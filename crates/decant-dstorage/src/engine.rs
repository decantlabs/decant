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
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{DuplicateHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::Storage::FileSystem::{BY_HANDLE_FILE_INFORMATION, CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, OPEN_EXISTING, ReadFile};
use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess, INFINITE, SetEvent, WaitForSingleObject};

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

// ---- File ----

/// An open file: a Win32 handle on the CPU backend, a PC DirectStorage file otherwise.
pub struct File {
    #[allow(dead_code)] // the first word of the object, read by callers through the vtable
    vtbl: &'static UnknownVtbl,
    refs: AtomicU32,
    backend: Arc<Backend>,
    handle: Handle,
    native: Option<Pc>,
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

/// One recorded operation. It gets the queue's shared state and whether the queue is being
/// closed: a cancelled operation does nothing but release what it holds.
type Operation = Box<dyn FnOnce(&Arc<Core>, bool) -> HRESULT + Send>;

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
    submitted: VecDeque<Operation>,
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
    pending: Mutex<VecDeque<Operation>>,
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

/// Tells `d3d12_xs.dll`, when it is loaded, that `len` bytes at `destination` were written: the textures the title placed there
/// are detiled again before its next submission (`DecantD3D12XNoteWrite`).
fn note_written(destination: *mut c_void, len: u32) {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    // SAFETY: lookups by name in a module that is loaded or not, and a call of the export found, which takes (address, length).
    unsafe {
        let module = GetModuleHandleW(windows_sys::w!("d3d12_xs.dll"));
        if module.is_null() {
            return;
        }
        if let Some(f) = GetProcAddress(module, c"DecantD3D12XNoteWrite".as_ptr().cast()) {
            let f: unsafe extern "system" fn(usize, usize) = std::mem::transmute(f);
            f(destination as usize, len as usize);
        }
    }
}

/// Tells `xmem.dll`, when it is loaded, that the kernel or a decoder is about to write `len` bytes at
/// `destination`: pages the GPU mirror protected become writable and stay so a while
/// (`DecantXmemPrepareWrite`). Without it a write into such a page fails with `ERROR_NOACCESS`.
fn prepare_written(destination: *mut c_void, len: u32) {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    // SAFETY: lookups by name in a module that is loaded or not, and a call of the export found, which takes (address, length) and only looks the range up.
    unsafe {
        let module = GetModuleHandleW(windows_sys::w!("xmem.dll"));
        if module.is_null() {
            return;
        }
        if let Some(f) = GetProcAddress(module, c"DecantXmemPrepareWrite".as_ptr().cast()) {
            let f: unsafe extern "system" fn(usize, usize) = std::mem::transmute(f);
            f(destination as usize, len as usize);
        }
    }
}

/// Decodes handed on by one queue that may wait or run at once. The queue's worker waits beyond
/// that, so the buffers they hold stay bounded.
const MAX_IN_FLIGHT: usize = 16;

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
            prepare_written(destination, self.destination_size);
            let mut write_time = write_started.elapsed();
            // SAFETY: `input` holds the source bytes, and the title keeps `destination_size` bytes writable at the destination until the completion it waits for, which comes after this decode.
            let decode_started = Instant::now();
            let hr = unsafe { codecs::decode(self.codec, self.bc_mode, self.input.as_ptr(), self.input.len() as u32, destination.cast(), self.destination_size) };
            let decode_time = decode_started.elapsed();
            if hr >= 0 {
                let write_started = Instant::now();
                note_written(destination, self.destination_size);
                write_time += write_started.elapsed();
            }
            self.core_timing.record_and_log(self.source_bytes, self.destination_size, self.read_time, decode_time, write_time);
            return hr;
        };
        let output = if self.decode {
            let Ok(mut output) = buffer_of(self.destination_size) else { return E_OUTOFMEMORY };
            // SAFETY: `input` holds the source bytes and `output` `destination_size` bytes.
            let decode_started = Instant::now();
            let hr = unsafe { codecs::decode(self.codec, self.bc_mode, self.input.as_ptr(), self.input.len() as u32, output.as_mut_ptr(), self.destination_size) };
            let decode_time = decode_started.elapsed();
            if hr < 0 {
                return hr;
            }
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

    /// Runs one read. The bytes are read here, on the queue's worker. A decode, and the write
    /// into physical pages, go to the decode pool, so the worker reads the next request
    /// meanwhile. With `pages` the bytes go to those physical pages and `r.destination` is not used.
    fn read(self: &Arc<Self>, file: &File, r: &DecantRead, pages: Option<PageTarget>) -> HRESULT {
        if !matches!(r.codec, codec::COPY | codec::ZLIB | codec::GDEFLATE | codec::BCPACK) {
            decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("read with codec {}", r.codec));
            return E_NOTIMPL;
        }
        if r.codec == codec::COPY && r.source_size != r.destination_size {
            return E_INVALIDARG;
        }
        let decode = r.codec == codec::ZLIB || r.codec == codec::BCPACK || (self.native.get().is_none() && r.codec == codec::GDEFLATE);
        if !decode && pages.is_none() {
            // Straight into the destination.
            prepare_written(r.destination, r.destination_size);
            let read_started = Instant::now();
            let hr = self.fetch(file, r, r.destination, r.destination_size);
            let read_time = read_started.elapsed();
            let write_started = Instant::now();
            if hr >= 0 {
                note_written(r.destination, r.destination_size);
                self.timing.record_and_log(r.source_size, r.destination_size, read_time, Duration::ZERO, write_started.elapsed());
            }
            return hr;
        }
        // Into a buffer of ours first.
        let len = if decode { r.source_size } else { r.destination_size };
        let Ok(mut input) = buffer_of(len) else { return E_OUTOFMEMORY };
        let read_started = Instant::now();
        let hr = self.fetch(file, r, input.as_mut_ptr().cast(), len);
        let read_time = read_started.elapsed();
        if hr < 0 {
            return hr;
        }
        let staged = Staged { input, codec: r.codec, bc_mode: r.bc_mode, decode, destination: Sendable(r.destination), destination_size: r.destination_size, pages, core_timing: self.timing.clone(), source_bytes: r.source_size, read_time };
        {
            let mut in_flight = lock(&self.in_flight);
            while *in_flight >= MAX_IN_FLIGHT {
                in_flight = self.settled.wait(in_flight).unwrap_or_else(PoisonError::into_inner);
            }
            *in_flight += 1;
        }
        let core = self.clone();
        let read = Sendable(*r);
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
        S_OK
    }

    /// Reads `r.source_size` bytes of the file at `r.offset` into `buffer`, which holds `len`
    /// bytes (on PC DirectStorage a GDeflate read lands decompressed, `len` being the destination
    /// size).
    fn fetch(&self, file: &File, r: &DecantRead, buffer: *mut c_void, len: u32) -> HRESULT {
        if let Some(native) = self.native.get() {
            let Some(file_native) = file.native.as_ref() else { return E_INVALIDARG };
            // SAFETY: a zeroed structure of integers.
            let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: the file is an `IDStorageFile` and `info` is writable.
            let hr = unsafe { (file_native.vtbl::<pc::FileVtbl>().get_file_information)(file_native.as_ptr(), &mut info) };
            if hr < 0 {
                return hr;
            }
            let length = (u64::from(info.nFileSizeHigh) << 32) | u64::from(info.nFileSizeLow);
            if r.offset > length || u64::from(r.source_size) > length - r.offset {
                return hresult_from_win32(ERROR_HANDLE_EOF);
            }
            let compression = if r.codec == codec::GDEFLATE { pc::COMPRESSION_GDEFLATE } else { 0 };
            let request = pc::RawRequest::file_to_memory(compression, file_native.as_ptr(), r.offset, r.source_size, buffer, len);
            let queue = native.queue.as_ptr();
            // SAFETY: the queue is an `IDStorageQueue1` and the status array an `IDStorageStatusArray`, created by `init`. The buffer and the file stay valid until the wait below returns.
            unsafe {
                let vtbl = native.queue.vtbl::<pc::QueueVtbl>();
                (vtbl.enqueue_request)(queue, &request);
                (vtbl.enqueue_status)(queue, native.status.as_ptr(), 0);
                (vtbl.enqueue_set_event)(queue, native.done.0);
                (vtbl.submit)(queue);
                if WaitForSingleObject(native.done.0, INFINITE) != WAIT_OBJECT_0 {
                    return last_error();
                }
                return (native.status.vtbl::<pc::StatusArrayVtbl>().get_hresult)(native.status.as_ptr(), 0).min(S_OK);
            }
        }
        // SAFETY: plain Win32 calls, the buffer, the handle and the OVERLAPPED stay valid until `GetOverlappedResult` has waited for the read.
        unsafe {
            let event = Handle(CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()));
            if event.0.is_null() {
                return last_error();
            }
            let mut overlapped: OVERLAPPED = std::mem::zeroed();
            overlapped.hEvent = event.0;
            overlapped.Anonymous.Anonymous.Offset = r.offset as u32;
            overlapped.Anonymous.Anonymous.OffsetHigh = (r.offset >> 32) as u32;
            let mut got = 0u32;
            if ReadFile(file.handle.0, buffer.cast(), r.source_size, std::ptr::null_mut(), &mut overlapped) == 0 && windows_sys::Win32::Foundation::GetLastError() != ERROR_IO_PENDING {
                return last_error();
            }
            if GetOverlappedResult(file.handle.0, &overlapped, &mut got, 1) == 0 {
                return last_error();
            }
            if got != r.source_size {
                return hresult_from_win32(ERROR_HANDLE_EOF);
            }
        }
        S_OK
    }
}

/// A zeroed buffer of `len` bytes, or `Err` when it cannot be allocated.
fn buffer_of(len: u32) -> Result<Vec<u8>, ()> {
    let mut buffer: Vec<u8> = Vec::new();
    buffer.try_reserve_exact(len as usize).map_err(|_| ())?;
    buffer.resize(len as usize, 0);
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
    fn submit_locked(&self, pending: &mut VecDeque<Operation>) -> HRESULT {
        let mut batch = std::mem::take(pending);
        let mut state = lock(&self.work.state);
        if state.submitted.try_reserve(batch.len()).is_err() {
            drop(state);
            for operation in batch.drain(..) {
                operation(&self.core, true);
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
    fn append(&self, pending: &mut VecDeque<Operation>, operation: Operation) -> HRESULT {
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
        let read = Sendable(r);
        let mut pending = lock(&self.pending);
        self.append(
            &mut pending,
            Box::new(move |core, cancel| {
                if cancel {
                    return S_OK;
                }
                let r = read.get();
                let to_pages = pages.is_some();
                let hr = core.read(&file, r, pages);
                if hr < 0 {
                    log_failure(r, to_pages, hr);
                }
                hr
            }),
        )
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
        pending.push_back(Box::new(move |core, cancel| {
            core.settle();
            status.finish(index, if cancel { hresult_from_win32(ERROR_CANCELLED) } else { core.interval.load(Ordering::Acquire) });
            core.interval.store(S_OK, Ordering::Release);
            S_OK
        }));
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
            Box::new(move |core, cancel| {
                core.settle();
                // SAFETY: the duplicate is owned by `hold`.
                if cancel || unsafe { SetEvent(hold.get()) } != 0 { S_OK } else { last_error() }
            }),
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
            Box::new(move |core, cancel| {
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
            }),
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

/// The worker thread of a queue: runs the submitted operations in order, and on close cancels
/// what is left.
fn run(core: &Arc<Core>, work: &Work) {
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
                operation(core, true);
            }
            state = lock(&work.state);
            continue;
        }
        let Some(operation) = state.submitted.pop_front() else { continue };
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
                operation(&self.core, true);
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
        let mut file = File { vtbl: &FILE_VTBL, refs: AtomicU32::new(1), backend: self.backend.clone(), handle: Handle::null(), native: None };
        if let Some(factory) = self.backend.factory.as_ref() {
            let mut raw = std::ptr::null_mut();
            // SAFETY: the factory is an `IDStorageFactory` and `path` a zero terminated string, as the interface requires.
            let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().open_file)(factory.as_ptr(), path, &pc::IID_FILE, &mut raw) };
            if hr < 0 {
                return Err(hr);
            }
            // SAFETY: on success `raw` is a file reference we now own.
            file.native = unsafe { Pc::from_raw(raw) };
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
