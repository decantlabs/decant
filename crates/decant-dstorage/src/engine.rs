//! The queue engine (the former `storage.cpp`): files, status arrays, queues and the factory of the
//! project owned ABI, over PC DirectStorage when it is loaded or the Win32 CPU backend otherwise.
//! The engine never sees the console layout: `console` translates the console vtables into calls
//! on these objects.
//!
//! A queue records operations until `Submit`, then hands them to its own worker thread, which runs
//! them in order: `Submit` returns at once, as DirectStorage's does, and completion is reported
//! through status arrays, events and fences. Each queue has its own worker, so several queues read
//! in parallel. A worker only reads: decoding (zlib, GDeflate, BCPack) and writing into physical
//! pages run on a shared pool of decode threads (`pool`), and a status entry, an event or a fence
//! waits until the decodes of the reads before it have finished. Closing a queue lets the operation in flight finish, cancels everything after it
//! (status slots get `ERROR_CANCELLED`) and returns only once the worker has stopped.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, DuplicateHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{BY_HANDLE_FILE_INFORMATION, CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, OPEN_EXISTING, ReadFile};
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess, INFINITE, ResetEvent, SetEvent, WaitForSingleObject};

use crate::codecs;
use crate::log::storage_log;
use crate::pages::{PagePool, PageTarget, XmemPool};
use crate::pc::{self, Backend, Pc};
use crate::decant::*;

/// `GENERIC_READ`.
const GENERIC_READ: u32 = 0x8000_0000;
/// `ERROR_IO_PENDING`.
const ERROR_IO_PENDING: u32 = 997;
/// `EVENT_MODIFY_STATE`.
const EVENT_MODIFY_STATE: u32 = 0x0002;

/// Converts bytes processed over `duration` to mebibytes per second.
fn megabytes_per_second(bytes: u64, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds == 0.0 { 0.0 } else { bytes as f64 / (1024.0 * 1024.0) / seconds }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Something moved to the worker thread that holds raw pointers the caller keeps valid until
/// completion (a destination buffer, a console file object), or a COM reference.
struct Sendable<T>(T);

impl<T> Sendable<T> {
    fn get(&self) -> &T {
        &self.0
    }
}

// SAFETY: the engine's contract is that the caller retains the buffers a read points to until
// completion, and the worker thread is the only one that uses them while the read runs.
unsafe impl<T> Send for Sendable<T> {}
// SAFETY: `Sendable<T>` is only shared across threads when `T` itself is `Sync`.
unsafe impl<T: Sync> Sync for Sendable<T> {}

// ---- File ----

/// An open file: a Win32 handle on the CPU backend, a PC DirectStorage file otherwise.
#[repr(C)]
pub struct File {
    #[allow(dead_code)] // the first word of the object, read by callers through the vtable
    vtbl: &'static UnknownVtbl,
    refs: AtomicU32,
    backend: Arc<Backend>,
    handle: Handle,
    native: Option<Pc>,
    byte_length: Option<u64>,
}

static FILE_VTBL: UnknownVtbl = UnknownVtbl { query_interface: query_interface::<File>, add_ref: add_ref::<File>, release: release_object::<File> };

impl Com for File {
    const IID: GUID = IID_DECANT_FILE;
    fn refs(&self) -> &AtomicU32 {
        &self.refs
    }
    fn vtable_address() -> *const c_void {
        std::ptr::from_ref(&FILE_VTBL).cast()
    }
}

// ---- Status ----

/// A status array: one `HRESULT` per slot, `E_PENDING` while an operation owns the slot.
pub struct Status {
    #[allow(dead_code)] // the first word of the object, read by callers through the vtable
    vtbl: &'static StatusVtbl,
    refs: AtomicU32,
    slots: Mutex<Vec<HRESULT>>,
}

unsafe extern "system" fn status_result(this: *mut c_void, index: u32) -> HRESULT {
    // SAFETY: called through the vtable of a `Status`.
    unsafe { (*(this as *const Status)).result(index) }
}

static STATUS_VTBL: StatusVtbl =
    StatusVtbl { base: UnknownVtbl { query_interface: query_interface::<Status>, add_ref: add_ref::<Status>, release: release_object::<Status> }, result: status_result };

impl Com for Status {
    const IID: GUID = IID_DECANT_STATUS;
    fn refs(&self) -> &AtomicU32 {
        &self.refs
    }
    fn vtable_address() -> *const c_void {
        std::ptr::from_ref(&STATUS_VTBL).cast()
    }
}

impl Status {
    /// A status array of `count` slots, all `S_OK`, or `E_OUTOFMEMORY` when it cannot be allocated.
    fn new(count: u32) -> Result<Status, HRESULT> {
        let mut slots: Vec<HRESULT> = Vec::new();
        slots.try_reserve_exact(count as usize).map_err(|_| E_OUTOFMEMORY)?;
        slots.resize(count as usize, S_OK);
        Ok(Status { vtbl: &STATUS_VTBL, refs: AtomicU32::new(1), slots: Mutex::new(slots) })
    }

    /// The result of slot `index`.
    pub fn result(&self, index: u32) -> HRESULT {
        lock(&self.slots).get(index as usize).copied().unwrap_or(E_INVALIDARG)
    }

    fn reserve(&self, index: u32) -> HRESULT {
        let mut slots = lock(&self.slots);
        match slots.get_mut(index as usize) {
            None => E_INVALIDARG,
            Some(slot) if *slot == E_PENDING => hresult_from_win32(ERROR_BUSY),
            Some(slot) => {
                *slot = E_PENDING;
                S_OK
            }
        }
    }

    fn finish(&self, index: u32, hr: HRESULT) {
        if let Some(slot) = lock(&self.slots).get_mut(index as usize) {
            *slot = hr;
        }
    }
}

// ---- Queue ----

struct ReadOp {
    file: Ref<File>,
    read: Sendable<DecantRead>,
    pages: Option<PageTarget>,
}

type BarrierOp = Box<dyn FnOnce(&Arc<Core>, bool) -> HRESULT + Send>;

/// One recorded operation. A `Read` may be batched with consecutive reads, while a `Barrier`
/// settles all preceding decodes before signalling status, events, or fences.
enum QueueOp {
    Read(ReadOp),
    Barrier(BarrierOp),
}

/// Maximum consecutive reads batched into one submission or parallel CPU read pass.
const READ_BATCH_LIMIT: usize = 64;

/// The PC DirectStorage queue behind a project queue, with its status array and completion event.
struct Native {
    queue: Pc,
    status: Pc,
    done: Handle,
}

/// What the worker thread and the operations share.
struct Core {
    backend: Arc<Backend>,
    /// Where requests with a page destination write.
    pages: Arc<dyn PagePool>,
    /// Set by `init` when the backend is PC DirectStorage.
    native: OnceLock<Native>,
    /// First failure since the last status entry, of the worker or of a decode it handed on.
    interval: AtomicI32,
    /// Decodes handed to the pool and not finished yet.
    in_flight: Mutex<usize>,
    settled: Condvar,
    timing: Arc<Timing>,
}

/// Low overhead counters shared by this queue's worker and decode tasks.
struct Timing {
    window: Mutex<TimingWindow>,
}

struct TimingWindow {
    started: Instant,
    requests: u64,
    bytes: u64,
    written_bytes: u64,
    read: Duration,
    decode: Duration,
    write: Duration,
}

impl Timing {
    fn new() -> Timing {
        Timing { window: Mutex::new(TimingWindow { started: Instant::now(), requests: 0, bytes: 0, written_bytes: 0, read: Duration::ZERO, decode: Duration::ZERO, write: Duration::ZERO }) }
    }

    fn record(&self, bytes: u32, written_bytes: u32, read: Duration, decode: Duration, write: Duration) -> Option<String> {
        let mut window = lock(&self.window);
        window.requests += 1;
        window.bytes += u64::from(bytes);
        window.written_bytes += u64::from(written_bytes);
        window.read += read;
        window.decode += decode;
        window.write += write;
        if window.started.elapsed() < Duration::from_secs(5) {
            return None;
        }
        let line = format!(
            "dstorage 5s: requests {} bytes {} read {:.2} MB/s decode {:.2} MB/s write {:.2} MB/s\n",
            window.requests,
            window.bytes,
            megabytes_per_second(window.bytes, window.read),
            megabytes_per_second(window.bytes, window.decode),
            megabytes_per_second(window.written_bytes, window.write),
        );
        *window = TimingWindow { started: Instant::now(), requests: 0, bytes: 0, written_bytes: 0, read: Duration::ZERO, decode: Duration::ZERO, write: Duration::ZERO };
        Some(line)
    }

    fn record_and_log(&self, bytes: u32, written_bytes: u32, read: Duration, decode: Duration, write: Duration) {
        if let Some(line) = self.record(bytes, written_bytes, read, decode, write) {
            storage_log(&line);
        }
    }

    #[cfg(test)]
    fn summary_line(requests: u64, bytes: u64, written_bytes: u64, read: Duration, decode: Duration, write: Duration) -> String {
        format!(
            "dstorage 5s: requests {requests} bytes {bytes} read {:.2} MB/s decode {:.2} MB/s write {:.2} MB/s\n",
            megabytes_per_second(bytes, read),
            megabytes_per_second(bytes, decode),
            megabytes_per_second(written_bytes, write),
        )
    }
}

#[derive(Default)]
struct WorkState {
    submitted: VecDeque<QueueOp>,
    stopping: bool,
}

struct Work {
    state: Mutex<WorkState>,
    ready: Condvar,
}

pub struct Queue {
    #[allow(dead_code)] // the first word of the object, read by callers through the vtable
    vtbl: &'static QueueVtbl,
    refs: AtomicU32,
    capacity: u16,
    core: Arc<Core>,
    work: Arc<Work>,
    /// Recorded, not yet submitted.
    pending: Mutex<VecDeque<QueueOp>>,
    worker: Option<JoinHandle<()>>,
}

/// The vtable entries of a queue. Each one is called through the vtable of a `Queue`.
unsafe extern "system" fn queue_enqueue(this: *mut c_void, read: *const DecantRead) -> HRESULT {
    // SAFETY: `this` is a `Queue` and the caller passes a readable `DecantRead`.
    unsafe { (*(this as *const Queue)).enqueue(*read) }
}

unsafe extern "system" fn queue_enqueue_status(this: *mut c_void, slot: *mut c_void, index: u32) -> HRESULT {
    // SAFETY: `this` is a `Queue` and `slot` is null or a COM object.
    unsafe { (*(this as *const Queue)).enqueue_status(slot, index) }
}

unsafe extern "system" fn queue_enqueue_event(this: *mut c_void, event: HANDLE) -> HRESULT {
    // SAFETY: `this` is a `Queue`.
    unsafe { (*(this as *const Queue)).enqueue_event(event) }
}

unsafe extern "system" fn queue_enqueue_fence(this: *mut c_void, fence: *mut c_void, value: u64) -> HRESULT {
    // SAFETY: `this` is a `Queue` and `fence` is null or an `ID3D12Fence`.
    unsafe { (*(this as *const Queue)).enqueue_fence(fence, value) }
}

unsafe extern "system" fn queue_submit(this: *mut c_void) -> HRESULT {
    // SAFETY: `this` is a `Queue`.
    unsafe { (*(this as *const Queue)).submit() }
}

unsafe extern "system" fn queue_enqueue_console(this: *mut c_void, read: *const DecantConsoleRead) -> HRESULT {
    // SAFETY: `this` is a `Queue` and the caller passes a readable `DecantConsoleRead`.
    unsafe { (*(this as *const Queue)).enqueue_console(&*read) }
}

static QUEUE_VTBL: QueueVtbl = QueueVtbl {
    base: UnknownVtbl { query_interface: query_interface::<Queue>, add_ref: add_ref::<Queue>, release: release_object::<Queue> },
    enqueue: queue_enqueue,
    enqueue_status: queue_enqueue_status,
    enqueue_event: queue_enqueue_event,
    enqueue_fence: queue_enqueue_fence,
    submit: queue_submit,
    enqueue_console: queue_enqueue_console,
};

impl Com for Queue {
    const IID: GUID = IID_DECANT_QUEUE;
    fn refs(&self) -> &AtomicU32 {
        &self.refs
    }
    fn vtable_address() -> *const c_void {
        std::ptr::from_ref(&QUEUE_VTBL).cast()
    }
}

/// A reference to an `ID3D12Fence` held until its signal ran or was cancelled.
struct Fence(*mut c_void);

// SAFETY: a fence is a free threaded D3D12 object, the reference is used by the worker thread only.
unsafe impl Send for Fence {}

impl Fence {
    fn get(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: gives back the reference taken when the fence was enqueued.
        unsafe { release(self.0) };
    }
}

type WriteHookFn = unsafe extern "system" fn(usize, usize);

fn resolve_hook(slot: &OnceLock<WriteHookFn>, module_name: *const u16, proc_name: &std::ffi::CStr) -> Option<WriteHookFn> {
    if let Some(&f) = slot.get() {
        return Some(f);
    }
    // SAFETY: `module_name` and `proc_name` are NUL terminated constants.
    unsafe {
        let module = GetModuleHandleW(module_name);
        if module.is_null() {
            return None;
        }
        let proc = GetProcAddress(module, proc_name.as_ptr().cast())?;
        let f: WriteHookFn = std::mem::transmute(proc);
        let _ = slot.set(f);
        Some(f)
    }
}

/// Tells `d3d12_xs.dll`, when it is loaded, that `len` bytes at `destination` were written: the textures the title placed there
/// are detiled again before its next submission (`DecantD3D12XNoteWrite`).
fn note_written(destination: *mut c_void, len: u32) {
    static HOOK: OnceLock<WriteHookFn> = OnceLock::new();
    if let Some(f) = resolve_hook(&HOOK, windows_sys::w!("d3d12_xs.dll"), c"DecantD3D12XNoteWrite") {
        // SAFETY: `f` takes `(address, length)` and only looks the range up.
        unsafe { f(destination as usize, len as usize) };
    }
}

/// Pins `len` bytes at `destination` writable in `xmem.dll` using `DecantXmemBeginWrite` when
/// available, or falls back to `DecantXmemPrepareWrite`. Returns true if `finish_written` should
/// release the pin after the write completes.
fn begin_written(destination: *mut c_void, len: u32) -> bool {
    static BEGIN_HOOK: OnceLock<WriteHookFn> = OnceLock::new();
    if let Some(f) = resolve_hook(&BEGIN_HOOK, windows_sys::w!("xmem.dll"), c"DecantXmemBeginWrite") {
        // SAFETY: `f` takes `(address, length)` and pins the range writable.
        unsafe { f(destination as usize, len as usize) };
        return true;
    }
    static PREPARE_HOOK: OnceLock<WriteHookFn> = OnceLock::new();
    if let Some(f) = resolve_hook(&PREPARE_HOOK, windows_sys::w!("xmem.dll"), c"DecantXmemPrepareWrite") {
        // SAFETY: `f` takes `(address, length)` and marks the range writable.
        unsafe { f(destination as usize, len as usize) };
    }
    false
}

/// Releases a write pin acquired with `begin_written`.
fn finish_written(destination: *mut c_void, len: u32) {
    static FINISH_HOOK: OnceLock<WriteHookFn> = OnceLock::new();
    if let Some(f) = resolve_hook(&FINISH_HOOK, windows_sys::w!("xmem.dll"), c"DecantXmemFinishWrite") {
        // SAFETY: `f` takes `(address, length)` and releases the pin.
        unsafe { f(destination as usize, len as usize) };
    }
}

/// Decodes handed on by one queue that may wait or run at once. The queue's worker waits beyond
/// that, so the buffers they hold stay bounded.
const MAX_IN_FLIGHT: usize = 128;

/// A read whose bytes are in, to be decoded or written into physical pages on the decode pool.
struct Staged {
    input: Vec<u8>,
    codec: u32,
    bc_mode: u32,
    decode: bool,
    destination: Sendable<*mut c_void>,
    destination_size: u32,
    pages: Option<PageTarget>,
    core_timing: Arc<Timing>,
    source_bytes: u32,
    read_time: Duration,
}

impl Staged {
    /// Decodes into the destination, or decodes (or copies) into the physical pages.
    fn finish(self, pool: &dyn PagePool) -> HRESULT {
        let destination = *self.destination.get();
        let Some(target) = self.pages else {
            let write_started = Instant::now();
            let pinned = begin_written(destination, self.destination_size);
            let mut write_time = write_started.elapsed();
            // SAFETY: `input` holds the source bytes, and the title keeps `destination_size` bytes writable at the destination until the completion it waits for, which comes after this decode.
            let decode_started = Instant::now();
            let hr = unsafe { codecs::decode(self.codec, self.bc_mode, self.input.as_ptr(), self.input.len() as u32, destination.cast(), self.destination_size) };
            let decode_time = decode_started.elapsed();
            let write_started = Instant::now();
            if pinned {
                finish_written(destination, self.destination_size);
            }
            if hr >= 0 {
                note_written(destination, self.destination_size);
            }
            write_time += write_started.elapsed();
            self.core_timing.record_and_log(self.source_bytes, self.destination_size, self.read_time, decode_time, write_time);
            return hr;
        };
        let output = if self.decode {
            let Ok(mut output) = buffer_capacity(self.destination_size) else { return E_OUTOFMEMORY };
            // SAFETY: `input` holds the source bytes and `output` has capacity for `destination_size` bytes.
            let decode_started = Instant::now();
            let hr = unsafe { codecs::decode(self.codec, self.bc_mode, self.input.as_ptr(), self.input.len() as u32, output.as_mut_ptr(), self.destination_size) };
            let decode_time = decode_started.elapsed();
            if hr < 0 {
                return hr;
            }
            // SAFETY: `codecs::decode` initialized all `destination_size` bytes on success.
            unsafe { output.set_len(self.destination_size as usize) };
            (output, decode_time)
        } else {
            (self.input, Duration::ZERO)
        };
        let (output, decode_time) = output;
        let write_started = Instant::now();
        let frames = target.frames_now();
        let result = match pool.write(&frames, target.offset, &output) {
            Ok(views) => {
                static WRITTEN: AtomicU64 = AtomicU64::new(0);
                let n = WRITTEN.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(1000) {
                    storage_log(&format!("page destination: {n} reads written into physical pages so far\n"));
                }
                for (address, len) in views {
                    note_written(address as *mut c_void, len as u32);
                }
                S_OK
            }
            Err(error) => {
                storage_log(&format!("page destination frames={:x?} (at enqueue {:x?}) offset={:#x} bytes={} hr={:08x}\n", frames, target.frames, target.offset, output.len(), error as u32));
                error
            }
        };
        self.core_timing.record_and_log(self.source_bytes, output.len() as u32, self.read_time, decode_time, write_started.elapsed());
        result
    }
}

/// Logs a read that failed.
fn log_failure(r: &DecantRead, pages: bool, hr: HRESULT) {
    storage_log(&format!(
        "EnqueueRequest offset={} source={} dest={} codec={}{} hr={:08x}\n",
        r.offset,
        r.source_size,
        r.destination_size,
        r.codec,
        if pages { " to physical pages" } else { "" },
        hr as u32
    ));
}

impl Core {
    /// Notes a failure for the next status entry, unless one is noted already.
    fn fail(&self, hr: HRESULT) {
        let _ = self.interval.try_update(Ordering::AcqRel, Ordering::Acquire, |old| (old >= 0).then_some(hr));
    }

    /// Waits until every decode handed on has finished. A status entry, an event or a fence
    /// reports on the reads before it, so it runs only after this.
    fn settle(&self) {
        let mut in_flight = lock(&self.in_flight);
        while *in_flight > 0 {
            in_flight = self.settled.wait(in_flight).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Dispatches a completed read buffer to the shared decode pool.
    fn dispatch_staged(self: &Arc<Self>, op: &mut ReadOp, input: Vec<u8>, decode: bool, read_time: Duration) {
        let r = *op.read.get();
        let staged = Staged {
            input,
            codec: r.codec,
            bc_mode: r.bc_mode,
            decode,
            destination: Sendable(r.destination),
            destination_size: r.destination_size,
            pages: op.pages.take(),
            core_timing: self.timing.clone(),
            source_bytes: r.source_size,
            read_time,
        };
        {
            let mut in_flight = lock(&self.in_flight);
            while *in_flight >= MAX_IN_FLIGHT {
                in_flight = self.settled.wait(in_flight).unwrap_or_else(PoisonError::into_inner);
            }
            *in_flight += 1;
        }
        let core = self.clone();
        let read = Sendable(r);
        crate::pool::run(Box::new(move || {
            let to_pages = staged.pages.is_some();
            let hr = staged.finish(core.pages.as_ref());
            if hr < 0 {
                log_failure(read.get(), to_pages, hr);
                core.fail(hr);
            }
            *lock(&core.in_flight) -= 1;
            core.settled.notify_all();
        }));
    }

    /// Runs a batch of up to `READ_BATCH_LIMIT` consecutive reads on the active backend.
    fn read_batch(self: &Arc<Self>, batch: &mut [ReadOp]) {
        if batch.is_empty() {
            return;
        }
        if let Some(native) = self.native.get() {
            self.read_batch_native(native, batch);
            return;
        }
        if batch.len() == 1 {
            let op = &mut batch[0];
            let to_pages = op.pages.is_some();
            let hr = self.read_cpu(op);
            if hr < 0 {
                log_failure(op.read.get(), to_pages, hr);
                self.fail(hr);
            }
            return;
        }
        let base = batch.as_mut_ptr() as usize;
        let core = Sendable(self.clone());
        io_pool().run(batch.len(), |index| {
            // SAFETY: each `index` is unique across all threads in `IoPool::run`, and `batch` lives until `run` returns.
            let op = unsafe { &mut *(base as *mut ReadOp).add(index) };
            let to_pages = op.pages.is_some();
            let hr = core.get().read_cpu(op);
            if hr < 0 {
                log_failure(op.read.get(), to_pages, hr);
                core.get().fail(hr);
            }
        });
    }

    /// Batches up to `READ_BATCH_LIMIT` consecutive reads into one `IDStorageQueue1` submission.
    fn read_batch_native(self: &Arc<Self>, native: &Native, batch: &mut [ReadOp]) {
        enum PendingNative {
            Direct { destination: *mut c_void, destination_size: u32, source_size: u32, pinned: bool },
            Staged { index: usize, input: Vec<u8>, len: u32, decode: bool },
        }

        let mut pending = Vec::with_capacity(batch.len());
        let queue = native.queue.as_ptr();
        // SAFETY: `native.queue` is a live `IDStorageQueue1` owned by `Core`.
        let vtbl = unsafe { native.queue.vtbl::<pc::QueueVtbl>() };
        let started = Instant::now();

        for (index, op) in batch.iter().enumerate() {
            let r = op.read.get();
            if !matches!(r.codec, codec::COPY | codec::ZLIB | codec::GDEFLATE | codec::BCPACK) {
                decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("read with codec {}", r.codec));
                log_failure(r, op.pages.is_some(), E_NOTIMPL);
                self.fail(E_NOTIMPL);
                continue;
            }
            if r.codec == codec::COPY && r.source_size != r.destination_size {
                log_failure(r, op.pages.is_some(), E_INVALIDARG);
                self.fail(E_INVALIDARG);
                continue;
            }
            let Some(file_native) = op.file.native.as_ref() else {
                log_failure(r, op.pages.is_some(), E_INVALIDARG);
                self.fail(E_INVALIDARG);
                continue;
            };
            if let Some(length) = op.file.byte_length
                && (r.offset > length || u64::from(r.source_size) > length - r.offset)
            {
                let eof = hresult_from_win32(ERROR_HANDLE_EOF);
                log_failure(r, op.pages.is_some(), eof);
                self.fail(eof);
                continue;
            }
            let decode = r.codec == codec::ZLIB || r.codec == codec::BCPACK;
            let compression = if r.codec == codec::GDEFLATE { pc::COMPRESSION_GDEFLATE } else { 0 };
            if !decode && op.pages.is_none() {
                let pinned = begin_written(r.destination, r.destination_size);
                let request = pc::RawRequest::file_to_memory(compression, file_native.as_ptr(), r.offset, r.source_size, r.destination, r.destination_size);
                // SAFETY: `r.destination` remains valid until the completion wait below returns.
                unsafe { (vtbl.enqueue_request)(queue, &request) };
                pending.push(PendingNative::Direct { destination: r.destination, destination_size: r.destination_size, source_size: r.source_size, pinned });
            } else {
                let len = if decode { r.source_size } else { r.destination_size };
                let Ok(mut input) = buffer_capacity(len) else {
                    log_failure(r, op.pages.is_some(), E_OUTOFMEMORY);
                    self.fail(E_OUTOFMEMORY);
                    continue;
                };
                let request = pc::RawRequest::file_to_memory(compression, file_native.as_ptr(), r.offset, r.source_size, input.as_mut_ptr().cast(), len);
                // SAFETY: `input` buffer stays alive in `pending` until the wait below returns.
                unsafe { (vtbl.enqueue_request)(queue, &request) };
                pending.push(PendingNative::Staged { index, input, len, decode });
            }
        }

        if pending.is_empty() {
            return;
        }

        // SAFETY: `native.queue`, `native.status`, and `native.done` are live objects created in `Queue::create`.
        let hr = unsafe {
            (vtbl.enqueue_status)(queue, native.status.as_ptr(), 0);
            (vtbl.enqueue_set_event)(queue, native.done.0);
            (vtbl.submit)(queue);
            if WaitForSingleObject(native.done.0, INFINITE) != WAIT_OBJECT_0 {
                last_error()
            } else {
                (native.status.vtbl::<pc::StatusArrayVtbl>().get_hresult)(native.status.as_ptr(), 0).min(S_OK)
            }
        };
        let batch_read_time = started.elapsed();
        let per_read_time = batch_read_time / (pending.len() as u32).max(1);
        if hr < 0 {
            self.fail(hr);
        }

        for item in pending {
            match item {
                PendingNative::Direct { destination, destination_size, source_size, pinned } => {
                    let write_started = Instant::now();
                    if pinned {
                        finish_written(destination, destination_size);
                    }
                    if hr >= 0 {
                        note_written(destination, destination_size);
                        self.timing.record_and_log(source_size, destination_size, per_read_time, Duration::ZERO, write_started.elapsed());
                    }
                }
                PendingNative::Staged { index, mut input, len, decode } => {
                    if hr < 0 {
                        continue;
                    }
                    // SAFETY: `IDStorageQueue1` wrote all `len` bytes into `input` before signalling `native.done`.
                    unsafe { input.set_len(len as usize) };
                    self.dispatch_staged(&mut batch[index], input, decode, per_read_time);
                }
            }
        }
    }

    /// Runs one read on the Win32 CPU backend using a reusable thread local event handle.
    fn read_cpu(self: &Arc<Self>, op: &mut ReadOp) -> HRESULT {
        let r = op.read.get();
        if !matches!(r.codec, codec::COPY | codec::ZLIB | codec::GDEFLATE | codec::BCPACK) {
            decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("read with codec {}", r.codec));
            return E_NOTIMPL;
        }
        if r.codec == codec::COPY && r.source_size != r.destination_size {
            return E_INVALIDARG;
        }
        let decode = r.codec != codec::COPY;
        if !decode && op.pages.is_none() {
            let write_started = Instant::now();
            let pinned = begin_written(r.destination, r.destination_size);
            let mut write_time = write_started.elapsed();
            let read_started = Instant::now();
            let hr = read_file_cpu(op.file.handle.0, r.offset, r.source_size, r.destination);
            let read_time = read_started.elapsed();
            let write_started = Instant::now();
            if pinned {
                finish_written(r.destination, r.destination_size);
            }
            if hr >= 0 {
                note_written(r.destination, r.destination_size);
            }
            write_time += write_started.elapsed();
            if hr >= 0 {
                self.timing.record_and_log(r.source_size, r.destination_size, read_time, Duration::ZERO, write_time);
            }
            return hr;
        }
        let len = if decode { r.source_size } else { r.destination_size };
        let Ok(mut input) = buffer_capacity(len) else { return E_OUTOFMEMORY };
        let read_started = Instant::now();
        let hr = read_file_cpu(op.file.handle.0, r.offset, r.source_size, input.as_mut_ptr().cast());
        let read_time = read_started.elapsed();
        if hr < 0 {
            return hr;
        }
        // SAFETY: `read_file_cpu` verified `got == r.source_size` (`len` bytes initialized).
        unsafe { input.set_len(len as usize) };
        self.dispatch_staged(op, input, decode, read_time);
        S_OK
    }
}

struct ThreadEvent(HANDLE);

impl Drop for ThreadEvent {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: owned Win32 event handle created by this thread.
            unsafe { CloseHandle(self.0) };
        }
    }
}

thread_local! {
    static CPU_READ_EVENT: ThreadEvent = {
        // SAFETY: plain Win32 manual reset event created once per worker thread.
        ThreadEvent(unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) })
    };
}

fn read_file_cpu(handle: HANDLE, offset: u64, source_size: u32, buffer: *mut c_void) -> HRESULT {
    let event = CPU_READ_EVENT.with(|e| e.0);
    if event.is_null() {
        return last_error();
    }
    // SAFETY: `buffer` is writable for `source_size` bytes and `event` is a live thread local event.
    unsafe {
        let _ = ResetEvent(event);
        let mut overlapped: OVERLAPPED = std::mem::zeroed();
        overlapped.hEvent = event;
        overlapped.Anonymous.Anonymous.Offset = offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        let mut got = 0u32;
        if ReadFile(handle, buffer.cast(), source_size, std::ptr::null_mut(), &mut overlapped) == 0 && windows_sys::Win32::Foundation::GetLastError() != ERROR_IO_PENDING {
            return last_error();
        }
        if GetOverlappedResult(handle, &overlapped, &mut got, 1) == 0 {
            return last_error();
        }
        if got != source_size {
            return hresult_from_win32(ERROR_HANDLE_EOF);
        }
    }
    S_OK
}

struct IoBatchState {
    fn_ptr: usize,
    fn_data: usize,
    total: usize,
    workers_to_take: usize,
    active_workers: usize,
    epoch: u64,
}

struct IoPoolShared {
    state: Mutex<IoBatchState>,
    next: AtomicUsize,
    cv_work: Condvar,
    cv_done: Condvar,
}

struct IoPool {
    workers: usize,
    run_lock: Mutex<()>,
    shared: Arc<IoPoolShared>,
}

impl IoPool {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(IoPoolShared {
            state: Mutex::new(IoBatchState { fn_ptr: 0, fn_data: 0, total: 0, workers_to_take: 0, active_workers: 0, epoch: 0 }),
            next: AtomicUsize::new(0),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        for i in 0..workers {
            let shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new().name(format!("dstorage io {i}")).spawn(move || {
                let mut seen_epoch = 0u64;
                loop {
                    let (fn_ptr, fn_data, total) = {
                        let mut st = lock(&shared.state);
                        while st.epoch == seen_epoch || st.workers_to_take == 0 {
                            st = shared.cv_work.wait(st).unwrap_or_else(PoisonError::into_inner);
                        }
                        seen_epoch = st.epoch;
                        st.workers_to_take -= 1;
                        (st.fn_ptr, st.fn_data, st.total)
                    };
                    loop {
                        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
                        if idx >= total {
                            break;
                        }
                        // SAFETY: `run` keeps the closure alive until `active_workers` drops to 0.
                        unsafe {
                            let call: fn(usize, usize) = std::mem::transmute(fn_ptr);
                            call(fn_data, idx);
                        }
                    }
                    let mut st = lock(&shared.state);
                    st.active_workers -= 1;
                    if st.active_workers == 0 {
                        shared.cv_done.notify_one();
                    }
                }
            });
        }
        Self { workers, run_lock: Mutex::new(()), shared }
    }

    fn run<F: Fn(usize) + Sync>(&self, total: usize, f: F) {
        fn invoke<F: Fn(usize) + Sync>(data: usize, index: usize) {
            // SAFETY: `data` points to `f` for the entire duration of `run`.
            unsafe { (*(data as *const F))(index) };
        }
        let _guard = lock(&self.run_lock);
        let worker_count = total.saturating_sub(1).min(self.workers);
        if worker_count == 0 {
            for i in 0..total {
                f(i);
            }
            return;
        }
        {
            let mut st = lock(&self.shared.state);
            st.fn_ptr = invoke::<F> as *const () as usize;
            st.fn_data = std::ptr::from_ref(&f) as usize;
            st.total = total;
            self.shared.next.store(0, Ordering::Relaxed);
            st.workers_to_take = worker_count;
            st.active_workers = worker_count;
            st.epoch += 1;
            self.shared.cv_work.notify_all();
        }
        loop {
            let idx = self.shared.next.fetch_add(1, Ordering::Relaxed);
            if idx >= total {
                break;
            }
            f(idx);
        }
        let mut st = lock(&self.shared.state);
        while st.active_workers > 0 {
            st = self.shared.cv_done.wait(st).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

fn io_pool() -> &'static IoPool {
    static POOL: OnceLock<IoPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(8, 16);
        IoPool::new(threads)
    })
}

/// An uninitialized buffer with capacity for `len` bytes, or `Err` when it cannot be allocated.
fn buffer_capacity(len: u32) -> Result<Vec<u8>, ()> {
    let mut buffer: Vec<u8> = Vec::new();
    buffer.try_reserve_exact(len as usize).map_err(|_| ())?;
    Ok(buffer)
}

impl Queue {
    /// A queue and its worker thread, and, on PC DirectStorage, the native queue behind it.
    fn create(backend: Arc<Backend>, pages: Arc<dyn PagePool>, capacity: u16) -> Result<Queue, HRESULT> {
        let core = Arc::new(Core { backend, pages, native: OnceLock::new(), interval: AtomicI32::new(S_OK), in_flight: Mutex::new(0), settled: Condvar::new(), timing: Arc::new(Timing::new()) });
        let work = Arc::new(Work { state: Mutex::new(WorkState::default()), ready: Condvar::new() });
        let mut queue = Queue {
            vtbl: &QUEUE_VTBL,
            refs: AtomicU32::new(1),
            capacity,
            core: core.clone(),
            work: work.clone(),
            pending: Mutex::new(VecDeque::new()),
            worker: None,
        };
        let spawned = std::thread::Builder::new().spawn(move || run(&core, &work));
        queue.worker = Some(spawned.map_err(|_| E_OUTOFMEMORY)?);
        let Some(factory) = queue.core.backend.factory.as_ref() else { return Ok(queue) };
        let desc = pc::QueueDesc { source_type: 0, capacity: pc::MIN_QUEUE_CAPACITY, priority: 0, name: std::ptr::null(), device: std::ptr::null_mut() };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the factory is an `IDStorageFactory`, `desc` and `raw` live for the call.
        let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().create_queue)(factory.as_ptr(), &desc, &pc::IID_QUEUE1, &mut raw) };
        // SAFETY: on success `raw` is a queue reference we now own.
        let native = unsafe { Pc::from_raw(raw) };
        if hr < 0 {
            return Err(hr);
        }
        // SAFETY: plain Win32 call.
        let done = Handle(unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) });
        if done.0.is_null() {
            return Err(last_error());
        }
        decant_log::handles::created(done.0 as usize, "queue completion event");
        let mut raw = std::ptr::null_mut();
        // SAFETY: as above, with a zero terminated name.
        let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().create_status_array)(factory.as_ptr(), 1, c"DecantStorage".as_ptr().cast(), &pc::IID_STATUS_ARRAY, &mut raw) };
        // SAFETY: on success `raw` is a status array reference we now own.
        let status = unsafe { Pc::from_raw(raw) };
        if let (Some(queue_pc), Some(status)) = (native, status) {
            let _ = queue.core.native.set(Native { queue: queue_pc, status, done });
        }
        if hr < 0 { Err(hr) } else { Ok(queue) }
    }

    /// Hands every recorded operation to the worker. The caller holds the lock of `pending`.
    fn submit_locked(&self, pending: &mut VecDeque<QueueOp>) -> HRESULT {
        let mut batch = std::mem::take(pending);
        let mut state = lock(&self.work.state);
        if state.submitted.try_reserve(batch.len()).is_err() {
            drop(state);
            for operation in batch.drain(..) {
                if let QueueOp::Barrier(barrier) = operation {
                    barrier(&self.core, true);
                }
            }
            self.work.ready.notify_one();
            return E_OUTOFMEMORY;
        }
        state.submitted.extend(batch);
        drop(state);
        self.work.ready.notify_one();
        S_OK
    }

    /// Records an operation, and submits when half the capacity is recorded.
    fn append(&self, pending: &mut VecDeque<QueueOp>, operation: QueueOp) -> HRESULT {
        if pending.try_reserve(1).is_err() {
            return E_OUTOFMEMORY;
        }
        pending.push_back(operation);
        if pending.len() >= usize::from(self.capacity).div_ceil(2) { self.submit_locked(pending) } else { S_OK }
    }

    /// `Enqueue`: one read.
    pub fn enqueue(&self, r: DecantRead) -> HRESULT {
        self.enqueue_to(r, None)
    }

    /// One read, to `r.destination` or, with `pages`, to those physical pages.
    fn enqueue_to(&self, r: DecantRead, pages: Option<PageTarget>) -> HRESULT {
        if r.file.is_null() || (r.destination.is_null() && pages.is_none()) || r.source_size == 0 || r.destination_size == 0 {
            return E_INVALIDARG;
        }
        // SAFETY: `r.file` is not null and a COM object, as the interface requires.
        if !unsafe { is_object::<File>(r.file) } {
            return E_INVALIDARG;
        }
        // SAFETY: just checked to be one of our files.
        let file = unsafe { Ref::from_raw_add_ref(r.file.cast::<File>()) };
        if !Arc::ptr_eq(&file.backend, &self.core.backend) {
            return E_INVALIDARG;
        }
        let mut pending = lock(&self.pending);
        self.append(&mut pending, QueueOp::Read(ReadOp { file, read: Sendable(r), pages }))
    }

    /// `EnqueueStatus`: slot `index` of the status array gets the first failure since the last
    /// status entry, or `S_OK`, when the operations before it are done.
    ///
    /// # Safety
    /// `slot` is null or a COM object.
    pub unsafe fn enqueue_status(&self, slot: *mut c_void, index: u32) -> HRESULT {
        // SAFETY: the caller states that `slot` is null or a COM object.
        if !unsafe { is_object::<Status>(slot) } {
            return E_INVALIDARG;
        }
        // SAFETY: just checked to be one of our status arrays.
        let status = unsafe { Ref::from_raw_add_ref(slot.cast::<Status>()) };
        let mut pending = lock(&self.pending);
        let hr = status.reserve(index);
        if hr < 0 {
            return hr;
        }
        if pending.try_reserve(1).is_err() {
            status.finish(index, E_OUTOFMEMORY);
            return E_OUTOFMEMORY;
        }
        pending.push_back(QueueOp::Barrier(Box::new(move |core, cancel| {
            core.settle();
            status.finish(index, if cancel { hresult_from_win32(ERROR_CANCELLED) } else { core.interval.load(Ordering::Acquire) });
            core.interval.store(S_OK, Ordering::Release);
            S_OK
        })));
        if pending.len() >= usize::from(self.capacity).div_ceil(2) { self.submit_locked(&mut pending) } else { S_OK }
    }

    /// `EnqueueEvent`: sets `event` when the operations before it are done. It is not set when the
    /// queue is closed first.
    pub fn enqueue_event(&self, event: HANDLE) -> HRESULT {
        let mut duplicate: HANDLE = std::ptr::null_mut();
        // SAFETY: plain Win32 call, `duplicate` is writable.
        let duplicated = unsafe { DuplicateHandle(GetCurrentProcess(), event, GetCurrentProcess(), &mut duplicate, EVENT_MODIFY_STATE, 0, 0) };
        let hold = Handle(duplicate);
        if duplicated == 0 {
            return last_error();
        }
        let mut pending = lock(&self.pending);
        self.append(
            &mut pending,
            QueueOp::Barrier(Box::new(move |core, cancel| {
                core.settle();
                // SAFETY: the duplicate is owned by `hold`.
                if cancel || unsafe { SetEvent(hold.get()) } != 0 { S_OK } else { last_error() }
            })),
        )
    }

    /// `EnqueueFence`: signals `fence` with `value` when the operations before it are done. The
    /// queue holds a reference to the fence until then.
    ///
    /// # Safety
    /// `fence` is null or an `ID3D12Fence`.
    pub unsafe fn enqueue_fence(&self, fence: *mut c_void, value: u64) -> HRESULT {
        if fence.is_null() {
            return E_POINTER;
        }
        // SAFETY: a fence is a COM object, this takes the reference the `Fence` gives back.
        unsafe { (vtable::<UnknownVtbl>(fence).add_ref)(fence) };
        let fence = Fence(fence);
        let mut pending = lock(&self.pending);
        self.append(
            &mut pending,
            QueueOp::Barrier(Box::new(move |core, cancel| {
                core.settle();
                if cancel {
                    return S_OK;
                }
                type Signal = unsafe extern "system" fn(*mut c_void, u64) -> HRESULT;
                // SAFETY: `ID3D12Fence::Signal` is slot 10 (`IUnknown` 0 to 2, `ID3D12Object` 3 to 6, `GetDevice` 7, `GetCompletedValue` 8, `SetEventOnCompletion` 9).
                unsafe {
                    let object = fence.get();
                    let table = *(object as *const *const Signal);
                    (*table.add(10))(object, value)
                }
            })),
        )
    }

    /// `EnqueueConsole`: a read in the layout and with the option bits of the console request. The
    /// destination is an address, or physical pages (`console_options::DESTINATION_PAGES`).
    pub fn enqueue_console(&self, r: &DecantConsoleRead) -> HRESULT {
        let mode = ((r.options & console_options::BCPACK_MASK) >> console_options::BCPACK_SHIFT) as u32;
        let zlib = r.options & console_options::ZLIB != 0;
        if let Some(what) = console_refusal(r.options) {
            decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("EnqueueConsole {what} (options {:#x})", r.options));
            storage_log(&format!(
                "EnqueueConsole options={:x} ({what}) tag={} intermediate={} source={} dest={} hr=E_NOTIMPL\n",
                r.options, r.cancellation_tag, r.intermediate_size, r.source_size, r.destination_size
            ));
            return E_NOTIMPL;
        }
        if (zlib || mode != 0) && !r.offset.is_multiple_of(16) {
            return E_INVALIDARG;
        }
        let pages = if r.options & console_options::DESTINATION_PAGES != 0 {
            // SAFETY: with this option the destination is the title's array of frames, one per 64 KiB of the destination from the page offset.
            match unsafe { PageTarget::read(r.destination, r.destination_page_offset, r.destination_size) } {
                Ok(target) => Some(target),
                Err(hr) => return hr,
            }
        } else {
            None
        };
        self.enqueue_to(DecantRead {
            file: r.file,
            offset: r.offset,
            source_size: r.source_size,
            destination: r.destination,
            destination_size: r.destination_size,
            codec: if mode != 0 {
                codec::BCPACK
            } else if zlib {
                codec::ZLIB
            } else {
                codec::COPY
            },
            bc_mode: mode,
        }, pages)
    }

    /// `Submit`: hands the recorded operations to the worker and returns.
    pub fn submit(&self) -> HRESULT {
        let mut pending = lock(&self.pending);
        self.submit_locked(&mut pending)
    }
}

/// Why the engine refuses a console request with these options, `None` when it runs it. Each option
/// that is not implemented is named, for the log and the calls report.
fn console_refusal(options: u64) -> Option<String> {
    use console_options::*;
    let mode = (options & BCPACK_MASK) >> BCPACK_SHIFT;
    let swizzle = (options & SWIZZLE_MASK) >> SWIZZLE_SHIFT;
    Some(if swizzle != 0 {
        format!("swizzle mode {swizzle} (bits 5 to 20)")
    } else if options & SOURCE_PAGES != 0 {
        "source in physical pages (bit 23)".to_owned()
    } else if options & MEMORY_SOURCE != 0 {
        "memory source (bit 22)".to_owned()
    } else if options & RESERVED_MASK != 0 {
        format!("undeclared option bits {:#x}", options & RESERVED_MASK)
    } else if mode > 7 {
        format!("BCPack mode {mode}")
    } else if options & ZLIB != 0 && mode != 0 {
        format!("zlib then BCPack mode {mode}")
    } else {
        return None;
    })
}

/// The worker thread of a queue: runs consecutive reads in batches up to `READ_BATCH_LIMIT`,
/// runs barriers in order, and on close cancels what is left.
fn run(core: &Arc<Core>, work: &Work) {
    let mut batch = Vec::with_capacity(READ_BATCH_LIMIT);
    let mut state = lock(&work.state);
    loop {
        state = work.ready.wait_while(state, |s| !s.stopping && s.submitted.is_empty()).unwrap_or_else(PoisonError::into_inner);
        if state.submitted.is_empty() {
            drop(state);
            // The decodes handed on still write into the title's memory: the queue is closed only after them.
            core.settle();
            return;
        }
        if state.stopping {
            let rest = std::mem::take(&mut state.submitted);
            drop(state);
            for operation in rest {
                if let QueueOp::Barrier(barrier) = operation {
                    barrier(core, true);
                }
            }
            state = lock(&work.state);
            continue;
        }
        batch.clear();
        while batch.len() < READ_BATCH_LIMIT && matches!(state.submitted.front(), Some(QueueOp::Read(_))) {
            if let Some(QueueOp::Read(op)) = state.submitted.pop_front() {
                batch.push(op);
            }
        }
        if !batch.is_empty() {
            drop(state);
            core.read_batch(&mut batch);
            batch.clear();
            state = lock(&work.state);
            continue;
        }
        let Some(QueueOp::Barrier(operation)) = state.submitted.pop_front() else { continue };
        drop(state);
        let hr = operation(core, false);
        if hr < 0 {
            core.fail(hr);
        }
        state = lock(&work.state);
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        {
            let mut pending = lock(&self.pending);
            for operation in pending.drain(..) {
                if let QueueOp::Barrier(barrier) = operation {
                    barrier(&self.core, true);
                }
            }
        }
        if let Some(worker) = self.worker.take() {
            lock(&self.work.state).stopping = true;
            self.work.ready.notify_all();
            let _ = worker.join();
        }
    }
}

// ---- Factory ----

pub struct Factory {
    #[allow(dead_code)] // the first word of the object, read by callers through the vtable
    vtbl: &'static FactoryVtbl,
    refs: AtomicU32,
    backend: Arc<Backend>,
    pages: Arc<dyn PagePool>,
}

unsafe extern "system" fn factory_open_file(this: *mut c_void, path: *const u16, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: called through the vtable of a `Factory`, `out` is writable.
    unsafe {
        *out = std::ptr::null_mut();
        match (*(this as *const Factory)).open_file(path) {
            Ok(file) => {
                *out = file.into_raw().cast();
                S_OK
            }
            Err(hr) => hr,
        }
    }
}

unsafe extern "system" fn factory_create_queue(this: *mut c_void, capacity: u16, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: called through the vtable of a `Factory`, `out` is writable.
    unsafe {
        *out = std::ptr::null_mut();
        match (*(this as *const Factory)).create_queue(capacity) {
            Ok(queue) => {
                *out = queue.into_raw().cast();
                S_OK
            }
            Err(hr) => hr,
        }
    }
}

unsafe extern "system" fn factory_create_status_array(this: *mut c_void, count: u32, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: called through the vtable of a `Factory`, `out` is writable.
    unsafe {
        *out = std::ptr::null_mut();
        match (*(this as *const Factory)).create_status_array(count) {
            Ok(status) => {
                *out = status.into_raw().cast();
                S_OK
            }
            Err(hr) => hr,
        }
    }
}

static FACTORY_VTBL: FactoryVtbl = FactoryVtbl {
    base: UnknownVtbl { query_interface: query_interface::<Factory>, add_ref: add_ref::<Factory>, release: release_object::<Factory> },
    open_file: factory_open_file,
    create_queue: factory_create_queue,
    create_status_array: factory_create_status_array,
};

impl Com for Factory {
    const IID: GUID = IID_DECANT_FACTORY;
    fn refs(&self) -> &AtomicU32 {
        &self.refs
    }
    fn vtable_address() -> *const c_void {
        std::ptr::from_ref(&FACTORY_VTBL).cast()
    }
}

impl Factory {
    fn new(backend: Backend, pages: Arc<dyn PagePool>) -> Factory {
        Factory { vtbl: &FACTORY_VTBL, refs: AtomicU32::new(1), backend: Arc::new(backend), pages }
    }

    /// `OpenFile`.
    pub fn open_file(&self, path: *const u16) -> Result<Ref<File>, HRESULT> {
        if path.is_null() {
            return Err(E_INVALIDARG);
        }
        let mut file = File { vtbl: &FILE_VTBL, refs: AtomicU32::new(1), backend: self.backend.clone(), handle: Handle::null(), native: None, byte_length: None };
        if let Some(factory) = self.backend.factory.as_ref() {
            let mut raw = std::ptr::null_mut();
            // SAFETY: the factory is an `IDStorageFactory` and `path` a zero terminated string, as the interface requires.
            let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().open_file)(factory.as_ptr(), path, &pc::IID_FILE, &mut raw) };
            if hr < 0 {
                return Err(hr);
            }
            // SAFETY: on success `raw` is a file reference we now own.
            file.native = unsafe { Pc::from_raw(raw) };
            if let Some(file_native) = file.native.as_ref() {
                // SAFETY: `file_native` is a live `IDStorageFile` and `info` is writable.
                let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
                if unsafe { (file_native.vtbl::<pc::FileVtbl>().get_file_information)(file_native.as_ptr(), &mut info) } >= 0 {
                    file.byte_length = Some((u64::from(info.nFileSizeHigh) << 32) | u64::from(info.nFileSizeLow));
                }
            }
        } else {
            // SAFETY: `path` is a zero terminated string, as the interface requires.
            let handle = unsafe { CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, std::ptr::null(), OPEN_EXISTING, FILE_FLAG_OVERLAPPED, std::ptr::null_mut()) };
            if handle == INVALID_HANDLE_VALUE {
                return Err(last_error());
            }
            file.handle = Handle(handle);
        }
        Ok(Ref::new(file))
    }

    /// `CreateQueue`.
    pub fn create_queue(&self, capacity: u16) -> Result<Ref<Queue>, HRESULT> {
        if capacity == 0 {
            return Err(E_INVALIDARG);
        }
        Queue::create(self.backend.clone(), self.pages.clone(), capacity).map(Ref::new)
    }

    /// `CreateStatusArray`.
    pub fn create_status_array(&self, count: u32) -> Result<Ref<Status>, HRESULT> {
        if count == 0 {
            return Err(E_INVALIDARG);
        }
        Status::new(count).map(Ref::new)
    }
}

/// The engine factory singleton, with a new reference. Logs which backend it runs on.
pub fn engine_factory() -> Result<Ref<Factory>, HRESULT> {
    static SINGLETON: OnceLock<Ref<Factory>> = OnceLock::new();
    let factory = SINGLETON.get_or_init(|| Ref::new(Factory::new(Backend::new(), Arc::new(XmemPool))));
    if factory.backend.hr < 0 {
        storage_log(&format!("DStorageGetFactory PC initialization hr={:08x}\n", factory.backend.hr as u32));
        return Err(factory.backend.hr);
    }
    storage_log(&format!("DStorageGetFactory backend={}\n", if factory.backend.factory.is_some() { "PC DirectStorage" } else { "Win32 CPU" }));
    Ok(factory.clone())
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;

#[cfg(test)]
#[test]
fn byte_rate_uses_mebibytes_per_second() {
    assert_eq!(megabytes_per_second(4 * 1024 * 1024, Duration::from_secs(2)), 2.0);
    assert_eq!(megabytes_per_second(1, Duration::ZERO), 0.0);
    assert_eq!(Timing::summary_line(3, 4 * 1024 * 1024, 16 * 1024 * 1024, Duration::from_secs(2), Duration::from_secs(4), Duration::from_secs(8)), "dstorage 5s: requests 3 bytes 4194304 read 2.00 MB/s decode 1.00 MB/s write 2.00 MB/s\n");
}
