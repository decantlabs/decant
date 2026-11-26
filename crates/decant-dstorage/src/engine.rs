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
const READ_BATCH_LIMIT: usize = 512;

/// The PC DirectStorage queue behind a project queue, with double-buffered status/event slots
/// for overlapped batch submission and decode/write pipelining.
struct Native {
    queue: Pc,
    status: Pc,
    done: [Handle; 2],
    in_flight: Mutex<Option<NativeInFlight>>,
    next_slot: AtomicUsize,
}

enum PendingNative {
    Direct { destination_size: u32, source_size: u32 },
    Staged { read: Sendable<DecantRead>, pages: Option<PageTarget>, input: PooledBuffer, len: u32, decode: bool },
    Coalesced { start: u64, span: u32, members: Vec<(Sendable<DecantRead>, Option<PageTarget>)>, master: PooledBuffer },
}

// SAFETY: `PendingNative` is only moved from the queue worker into `Native.in_flight` and consumed by the same worker.
unsafe impl Send for PendingNative {}

struct NativeInFlight {
    slot: usize,
    started: Instant,
    pending: Vec<PendingNative>,
    /// The merged write pins held for the direct requests of pending until they completed.
    pins: Vec<WritePin>,
}

/// A merged destination range pinned writable for direct requests of the native queue.
struct WritePin {
    start: usize,
    len: u32,
    pinned: bool,
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
    in_flight_count: AtomicUsize,
    in_flight_waiters: AtomicUsize,
    in_flight_lock: Mutex<()>,
    settled: Condvar,
    timing: Arc<Timing>,
}

/// Process-wide cumulative DirectStorage metrics snapshot exported via `DecantDstorageMetrics`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DecantDstorageMetricsSnapshot {
    pub requests: u64,
    pub bytes_read: u64,
    pub active_read_ns: u64,
    pub decode_ns: u64,
    pub write_ns: u64,
    pub in_flight: u64,
    pub batches: u64,
}

struct GlobalMetrics {
    requests: AtomicU64,
    bytes_read: AtomicU64,
    active_read_ns: AtomicU64,
    decode_ns: AtomicU64,
    write_ns: AtomicU64,
    in_flight: AtomicU64,
    batches: AtomicU64,
}

static GLOBAL_METRICS: GlobalMetrics = GlobalMetrics {
    requests: AtomicU64::new(0),
    bytes_read: AtomicU64::new(0),
    active_read_ns: AtomicU64::new(0),
    decode_ns: AtomicU64::new(0),
    write_ns: AtomicU64::new(0),
    in_flight: AtomicU64::new(0),
    batches: AtomicU64::new(0),
};

/// Returns a lock-free snapshot of cumulative DirectStorage metrics across all queues.
pub fn metrics_snapshot() -> DecantDstorageMetricsSnapshot {
    DecantDstorageMetricsSnapshot {
        requests: GLOBAL_METRICS.requests.load(Ordering::Relaxed),
        bytes_read: GLOBAL_METRICS.bytes_read.load(Ordering::Relaxed),
        active_read_ns: GLOBAL_METRICS.active_read_ns.load(Ordering::Relaxed),
        decode_ns: GLOBAL_METRICS.decode_ns.load(Ordering::Relaxed),
        write_ns: GLOBAL_METRICS.write_ns.load(Ordering::Relaxed),
        in_flight: GLOBAL_METRICS.in_flight.load(Ordering::Relaxed),
        batches: GLOBAL_METRICS.batches.load(Ordering::Relaxed),
    }
}

/// Low overhead counters shared by this queue's worker and decode tasks.
struct Timing {
    requests: AtomicU64,
    batches: AtomicU64,
    bytes: AtomicU64,
    written_bytes: AtomicU64,
    read_ns: AtomicU64,
    decode_ns: AtomicU64,
    write_ns: AtomicU64,
    window: Mutex<Instant>,
}

impl Timing {
    fn new() -> Timing {
        Timing {
            requests: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            written_bytes: AtomicU64::new(0),
            read_ns: AtomicU64::new(0),
            decode_ns: AtomicU64::new(0),
            write_ns: AtomicU64::new(0),
            window: Mutex::new(Instant::now()),
        }
    }

    fn record_batch(&self, active_read: Duration) {
        let ns = active_read.as_nanos() as u64;
        self.batches.fetch_add(1, Ordering::Relaxed);
        self.read_ns.fetch_add(ns, Ordering::Relaxed);
        GLOBAL_METRICS.batches.fetch_add(1, Ordering::Relaxed);
        GLOBAL_METRICS.active_read_ns.fetch_add(ns, Ordering::Relaxed);
    }

    fn record(&self, bytes: u32, written_bytes: u32, _read: Duration, decode: Duration, write: Duration) -> Option<String> {
        let count = self.requests.fetch_add(1, Ordering::Relaxed) + 1;
        let d_ns = decode.as_nanos() as u64;
        let w_ns = write.as_nanos() as u64;
        self.bytes.fetch_add(u64::from(bytes), Ordering::Relaxed);
        self.written_bytes.fetch_add(u64::from(written_bytes), Ordering::Relaxed);
        self.decode_ns.fetch_add(d_ns, Ordering::Relaxed);
        self.write_ns.fetch_add(w_ns, Ordering::Relaxed);

        GLOBAL_METRICS.requests.fetch_add(1, Ordering::Relaxed);
        GLOBAL_METRICS.bytes_read.fetch_add(u64::from(bytes), Ordering::Relaxed);
        GLOBAL_METRICS.decode_ns.fetch_add(d_ns, Ordering::Relaxed);
        GLOBAL_METRICS.write_ns.fetch_add(w_ns, Ordering::Relaxed);

        if !count.is_multiple_of(64) {
            return None;
        }
        let Ok(mut started) = self.window.try_lock() else { return None };
        let window_elapsed = started.elapsed();
        if window_elapsed < Duration::from_secs(5) {
            return None;
        }
        *started = Instant::now();
        drop(started);
        let requests = self.requests.swap(0, Ordering::Relaxed);
        let batches = self.batches.swap(0, Ordering::Relaxed);
        let bytes = self.bytes.swap(0, Ordering::Relaxed);
        let written_bytes = self.written_bytes.swap(0, Ordering::Relaxed);
        let read = Duration::from_nanos(self.read_ns.swap(0, Ordering::Relaxed));
        let decode = Duration::from_nanos(self.decode_ns.swap(0, Ordering::Relaxed));
        let write = Duration::from_nanos(self.write_ns.swap(0, Ordering::Relaxed));
        let avg_batch = if batches > 0 { requests as f64 / batches as f64 } else { requests as f64 };
        Some(format!(
            "dstorage 5s: requests {requests} batches {batches} avg_batch {avg_batch:.1} bytes {bytes} read {:.2} MB/s (window {:.2} MB/s) decode {:.2} MB/s write {:.2} MB/s\n",
            megabytes_per_second(bytes, read),
            megabytes_per_second(bytes, window_elapsed),
            megabytes_per_second(bytes, decode),
            megabytes_per_second(written_bytes, write),
        ))
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

/// Tells `d3d12_xs.dll` about the views a page write reported, adjacent views as one range.
fn note_views(views: &[(usize, usize)]) {
    if views.len() <= 1 {
        for &(address, len) in views {
            note_written(address as *mut c_void, len as u32);
        }
        return;
    }
    let ranges: Vec<(usize, u32)> = views.iter().map(|&(address, len)| (address, len as u32)).collect();
    for span in crate::spans::merge(&ranges) {
        note_written(span.start as *mut c_void, span.len);
    }
}

/// A read of the `COPY` codec whose bytes are already in memory and go to a plain destination.
struct DirectCopy {
    destination: *mut c_void,
    source: *const u8,
    len: u32,
    /// Compressed size of the request, for the timing line.
    source_size: u32,
}

/// Copies every item to its destination. Destinations that touch or overlap share one write pin and
/// one `NoteWrite`, so `n` adjacent requests cost one `BeginWrite`/`FinishWrite`/`NoteWrite` instead
/// of `n`. A byte is only written between the `BeginWrite` and `FinishWrite` of the span that covers it.
/// Returns the time spent, for the timing line.
///
/// # Safety
/// Every `source` is readable for `len` bytes and every `destination` writable for `len` bytes,
/// and no source overlaps a destination.
unsafe fn copy_direct_merged(items: &[DirectCopy]) -> Duration {
    let started = Instant::now();
    let ranges: Vec<(usize, u32)> = items.iter().map(|item| (item.destination as usize, item.len)).collect();
    for span in crate::spans::merge(&ranges) {
        let start = span.start as *mut c_void;
        let pinned = begin_written(start, span.len);
        for &member in &span.members {
            let item = &items[member];
            // SAFETY: by the contract; the span covers `item.destination..+item.len` and is pinned.
            unsafe { std::ptr::copy_nonoverlapping(item.source, item.destination.cast(), item.len as usize) };
        }
        if pinned {
            finish_written(start, span.len);
        }
        note_written(start, span.len);
    }
    started.elapsed()
}

/// Decodes handed on by one queue that may wait or run at once. The queue's worker waits beyond
/// that, so the buffers they hold stay bounded.
const MAX_IN_FLIGHT: usize = 1024;

/// Minimum pooled buffer size (4 KiB, one OS page).
const POOL_MIN_SHIFT: u32 = 12;
/// Maximum pooled buffer size (16 MiB).
const POOL_MAX_SHIFT: u32 = 24;
/// Number of power-of-two size classes in `BufferPool`.
const POOL_BUCKETS: usize = (POOL_MAX_SHIFT - POOL_MIN_SHIFT + 1) as usize;
/// Cap on total memory retained in `BufferPool` (128 MiB).
const MAX_POOL_BYTES: usize = 128 * 1024 * 1024;

struct BufferPoolInner {
    buckets: [Vec<Vec<u8>>; POOL_BUCKETS],
    total_bytes: usize,
}

impl BufferPoolInner {
    const fn new() -> Self {
        Self { buckets: [const { Vec::new() }; POOL_BUCKETS], total_bytes: 0 }
    }
}

struct BufferPool {
    inner: Mutex<BufferPoolInner>,
}

impl BufferPool {
    fn new() -> Self {
        Self { inner: Mutex::new(BufferPoolInner::new()) }
    }

    fn bucket_for(cap: usize) -> Option<usize> {
        if !(1 << POOL_MIN_SHIFT..=1 << POOL_MAX_SHIFT).contains(&cap) || !cap.is_power_of_two() {
            return None;
        }
        Some((cap.trailing_zeros() - POOL_MIN_SHIFT) as usize)
    }

    fn acquire(&self, len: u32) -> Result<PooledBuffer, ()> {
        let len = len as usize;
        if len == 0 {
            return Ok(PooledBuffer(Vec::new()));
        }
        let target_cap = len.next_power_of_two().max(1 << POOL_MIN_SHIFT);
        if let Some(bucket) = Self::bucket_for(target_cap) {
            let mut inner = lock(&self.inner);
            if let Some(mut buf) = inner.buckets[bucket].pop() {
                inner.total_bytes -= buf.capacity();
                buf.clear();
                return Ok(PooledBuffer(buf));
            }
            drop(inner);
            let mut buf = Vec::new();
            if buf.try_reserve_exact(target_cap).is_ok() {
                return Ok(PooledBuffer(buf));
            }
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(len).map_err(|_| ())?;
        Ok(PooledBuffer(buf))
    }

    fn release(&self, mut buf: Vec<u8>) {
        let cap = buf.capacity();
        let Some(bucket) = Self::bucket_for(cap) else { return };
        let mut inner = lock(&self.inner);
        if inner.total_bytes.saturating_add(cap) > MAX_POOL_BYTES {
            return;
        }
        buf.clear();
        inner.total_bytes += cap;
        inner.buckets[bucket].push(buf);
    }
}

fn buffer_pool() -> &'static BufferPool {
    static POOL: OnceLock<BufferPool> = OnceLock::new();
    POOL.get_or_init(BufferPool::new)
}

/// Largest buffer a thread keeps in its own cache (256 KiB): the many small requests are the ones
/// that hammer the shared pool's lock.
const TLS_MAX_BUFFER: usize = 256 * 1024;
/// Most bytes one thread keeps in its cache (2 MiB), so the memory held by threads stays bounded.
const TLS_MAX_BYTES: usize = 2 * 1024 * 1024;
/// Most buffers of one size class one thread keeps.
const TLS_MAX_PER_BUCKET: usize = 8;

/// The per thread front of BufferPool: small buffers released on a thread are reused by it
/// without taking the shared lock. Dropped (freed) with the thread.
struct TlsBuffers {
    buckets: [Vec<Vec<u8>>; POOL_BUCKETS],
    total_bytes: usize,
}

impl TlsBuffers {
    const fn new() -> Self {
        Self { buckets: [const { Vec::new() }; POOL_BUCKETS], total_bytes: 0 }
    }

    fn take(&mut self, bucket: usize) -> Option<Vec<u8>> {
        let buf = self.buckets[bucket].pop()?;
        self.total_bytes -= buf.capacity();
        Some(buf)
    }

    /// Keeps uf unless the caps are reached; gives it back otherwise.
    fn put(&mut self, bucket: usize, buf: Vec<u8>) -> Result<(), Vec<u8>> {
        let cap = buf.capacity();
        if cap > TLS_MAX_BUFFER || self.buckets[bucket].len() >= TLS_MAX_PER_BUCKET || self.total_bytes + cap > TLS_MAX_BYTES {
            return Err(buf);
        }
        self.total_bytes += cap;
        self.buckets[bucket].push(buf);
        Ok(())
    }
}

thread_local! {
    static TLS_BUFFERS: std::cell::RefCell<TlsBuffers> = const { std::cell::RefCell::new(TlsBuffers::new()) };
}

impl BufferPool {
    /// `acquire`, trying this thread's cache before the shared lock.
    fn acquire_cached(&self, len: u32) -> Result<PooledBuffer, ()> {
        let target_cap = (len as usize).next_power_of_two().max(1 << POOL_MIN_SHIFT);
        if len != 0
            && target_cap <= TLS_MAX_BUFFER
            && let Some(bucket) = Self::bucket_for(target_cap)
            && let Some(mut buf) = TLS_BUFFERS.try_with(|cache| cache.borrow_mut().take(bucket)).ok().flatten()
        {
            buf.clear();
            return Ok(PooledBuffer(buf));
        }
        self.acquire(len)
    }

    /// `release`, keeping a small buffer on this thread while its cache has room.
    fn release_cached(&self, buf: Vec<u8>) {
        let Some(bucket) = Self::bucket_for(buf.capacity()) else { return };
        let mut buf = buf;
        buf.clear();
        match TLS_BUFFERS.try_with(|cache| cache.borrow_mut().put(bucket, std::mem::take(&mut buf))) {
            Ok(Ok(())) => {}
            Ok(Err(back)) => self.release(back),
            // The thread is exiting: its cache is gone.
            Err(_) => self.release(buf),
        }
    }
}

/// Bytes the calling thread holds in its cache.
#[cfg(test)]
fn tls_cached_bytes() -> usize {
    TLS_BUFFERS.with(|cache| cache.borrow().total_bytes)
}

/// A `Vec<u8>` rented from `buffer_pool()` that returns to the pool on drop.
struct PooledBuffer(Vec<u8>);

impl std::ops::Deref for PooledBuffer {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if self.0.capacity() > 0 {
            buffer_pool().release_cached(std::mem::take(&mut self.0));
        }
    }
}

/// Source bytes for a `Staged` decode: either an owned `PooledBuffer` or a zero-copy slice of a
/// shared `PooledBuffer` from a coalesced read.
enum InputBuffer {
    Owned(PooledBuffer),
    Shared { buf: Arc<PooledBuffer>, start: usize, len: usize },
}

impl InputBuffer {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(buf) => buf.as_slice(),
            Self::Shared { buf, start, len } => &buf[*start..*start + *len],
        }
    }
}

/// Maximum gap between adjacent reads in the same file to coalesce into one sequential read.
const MAX_COALESCE_GAP: u64 = 64 * 1024;
/// Maximum total byte span of a single coalesced read.
const MAX_COALESCE_BYTES: u64 = 4 * 1024 * 1024;

struct CoalescedGroup {
    start: u64,
    span: u32,
    members: Vec<usize>,
}

struct CoalescePlan {
    groups: Vec<CoalescedGroup>,
    singles: Vec<usize>,
}

fn plan_coalesced_batches(batch: &[ReadOp], is_native: bool) -> CoalescePlan {
    let mut eligible: Vec<(usize, u64, u32, usize)> = Vec::with_capacity(batch.len());
    let mut singles = Vec::new();
    for (idx, op) in batch.iter().enumerate() {
        let r = op.read.get();
        let valid_codec = matches!(r.codec, codec::COPY | codec::ZLIB | codec::GDEFLATE | codec::BCPACK)
            && !(r.codec == codec::COPY && r.source_size != r.destination_size);
        let native_ok = !is_native
            || (r.codec != codec::GDEFLATE
                && op.file.native.is_some()
                && op.file.byte_length.is_none_or(|len| r.offset <= len && u64::from(r.source_size) <= len - r.offset));
        if valid_codec && native_ok && r.source_size > 0 && u64::from(r.source_size) <= MAX_COALESCE_BYTES {
            eligible.push((op.file.as_ptr() as usize, r.offset, r.source_size, idx));
        } else {
            singles.push(idx);
        }
    }
    eligible.sort_unstable_by_key(|&(file_ptr, offset, _, idx)| (file_ptr, offset, idx));

    let mut groups = Vec::new();
    let mut pos = 0;
    while pos < eligible.len() {
        let (file_ptr, run_start, first_size, first_idx) = eligible[pos];
        let mut run_end = run_start + u64::from(first_size);
        let mut members = vec![first_idx];
        pos += 1;
        while pos < eligible.len() {
            let (next_file, next_off, next_size, next_idx) = eligible[pos];
            if next_file != file_ptr {
                break;
            }
            let gap = next_off.saturating_sub(run_end);
            let new_end = run_end.max(next_off + u64::from(next_size));
            let new_span = new_end - run_start;
            if gap > MAX_COALESCE_GAP || new_span > MAX_COALESCE_BYTES {
                break;
            }
            run_end = new_end;
            members.push(next_idx);
            pos += 1;
        }
        if members.len() >= 2 {
            groups.push(CoalescedGroup { start: run_start, span: (run_end - run_start) as u32, members });
        } else {
            singles.push(first_idx);
        }
    }
    CoalescePlan { groups, singles }
}

/// A read whose bytes are in, to be decoded or written into physical pages on the decode pool.
struct Staged {
    input: InputBuffer,
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

/// Maximum size of the per-decode-thread scratch buffer kept without returning to `BufferPool`.
const THREAD_SCRATCH_MAX: usize = 4 * 1024 * 1024;

thread_local! {
    static DECODE_SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn with_decode_output<F: FnOnce(&mut [u8]) -> HRESULT>(size: usize, f: F) -> HRESULT {
    if size <= THREAD_SCRATCH_MAX {
        return DECODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            if scratch.capacity() < size && scratch.try_reserve_exact(size).is_err() {
                return E_OUTOFMEMORY;
            }
            // SAFETY: capacity >= size, and `f` initializes/writes the output slice before reading.
            unsafe { scratch.set_len(size) };
            f(&mut scratch[..size])
        });
    }
    let Ok(mut output) = buffer_capacity(size as u32) else { return E_OUTOFMEMORY };
    // SAFETY: `output` has capacity >= `size`.
    unsafe { output.set_len(size) };
    f(&mut output[..size])
}

impl Staged {
    /// Decodes into the destination, or decodes (or copies) into the physical pages.
    fn finish(self, pool: &dyn PagePool) -> HRESULT {
        let destination = *self.destination.get();
        let input_slice = self.input.as_slice();
        let Some(target) = self.pages else {
            let write_started = Instant::now();
            let pinned = begin_written(destination, self.destination_size);
            let mut write_time = write_started.elapsed();
            // SAFETY: `input` holds the source bytes, and the title keeps `destination_size` bytes writable at the destination until the completion it waits for, which comes after this decode.
            let decode_started = Instant::now();
            let hr = unsafe { codecs::decode(self.codec, self.bc_mode, input_slice.as_ptr(), input_slice.len() as u32, destination.cast(), self.destination_size) };
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
        if self.decode {
            let out_len = self.destination_size as usize;
            return with_decode_output(out_len, |output_slice| {
                let decode_started = Instant::now();
                // SAFETY: `input_slice` holds the source bytes and `output_slice` is writable for `out_len` bytes.
                let hr = unsafe { codecs::decode(self.codec, self.bc_mode, input_slice.as_ptr(), input_slice.len() as u32, output_slice.as_mut_ptr(), self.destination_size) };
                let decode_time = decode_started.elapsed();
                if hr < 0 {
                    return hr;
                }
                let write_started = Instant::now();
                let frames = target.frames_now_cow();
                let result = match pool.write(&frames, target.offset, output_slice) {
                    Ok(views) => {
                        static WRITTEN: AtomicU64 = AtomicU64::new(0);
                        let n = WRITTEN.fetch_add(1, Ordering::Relaxed) + 1;
                        if n == 1 || n.is_multiple_of(1000) {
                            storage_log(&format!("page destination: {n} reads written into physical pages so far\n"));
                        }
                        note_views(&views);
                        S_OK
                    }
                    Err(error) => {
                        storage_log(&format!("page destination frames={:x?} (at enqueue {:x?}) offset={:#x} bytes={} hr={:08x}\n", frames.as_ref(), target.frames, target.offset, output_slice.len(), error as u32));
                        error
                    }
                };
                self.core_timing.record_and_log(self.source_bytes, self.destination_size, self.read_time, decode_time, write_started.elapsed());
                result
            });
        }
        let write_started = Instant::now();
        let frames = target.frames_now_cow();
        let result = match pool.write(&frames, target.offset, input_slice) {
            Ok(views) => {
                static WRITTEN: AtomicU64 = AtomicU64::new(0);
                let n = WRITTEN.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(1000) {
                    storage_log(&format!("page destination: {n} reads written into physical pages so far\n"));
                }
                note_views(&views);
                S_OK
            }
            Err(error) => {
                storage_log(&format!("page destination frames={:x?} (at enqueue {:x?}) offset={:#x} bytes={} hr={:08x}\n", frames.as_ref(), target.frames, target.offset, input_slice.len(), error as u32));
                error
            }
        };
        self.core_timing.record_and_log(self.source_bytes, input_slice.len() as u32, self.read_time, Duration::ZERO, write_started.elapsed());
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
        if self.in_flight_count.load(Ordering::SeqCst) == 0 {
            return;
        }
        let mut guard = lock(&self.in_flight_lock);
        self.in_flight_waiters.fetch_add(1, Ordering::SeqCst);
        while self.in_flight_count.load(Ordering::SeqCst) > 0 {
            guard = self.settled.wait(guard).unwrap_or_else(PoisonError::into_inner);
        }
        self.in_flight_waiters.fetch_sub(1, Ordering::SeqCst);
    }

    fn wait_for_in_flight_room(&self, slots: usize) {
        if self.in_flight_count.load(Ordering::SeqCst) + slots > MAX_IN_FLIGHT {
            let mut guard = lock(&self.in_flight_lock);
            self.in_flight_waiters.fetch_add(1, Ordering::SeqCst);
            while self.in_flight_count.load(Ordering::SeqCst) + slots > MAX_IN_FLIGHT {
                guard = self.settled.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
            self.in_flight_waiters.fetch_sub(1, Ordering::SeqCst);
        }
        self.in_flight_count.fetch_add(slots, Ordering::SeqCst);
        GLOBAL_METRICS.in_flight.fetch_add(slots as u64, Ordering::Relaxed);
    }

    fn build_staged_job_from_parts(self: &Arc<Self>, r: DecantRead, pages: Option<PageTarget>, input: InputBuffer, decode: bool, read_time: Duration) -> crate::pool::Job {
        let staged = Staged {
            input,
            codec: r.codec,
            bc_mode: r.bc_mode,
            decode,
            destination: Sendable(r.destination),
            destination_size: r.destination_size,
            pages,
            core_timing: self.timing.clone(),
            source_bytes: r.source_size,
            read_time,
        };
        let core = self.clone();
        let read = Sendable(r);
        Box::new(move || {
            let to_pages = staged.pages.is_some();
            let hr = staged.finish(core.pages.as_ref());
            if hr < 0 {
                log_failure(read.get(), to_pages, hr);
                core.fail(hr);
            }
            core.in_flight_count.fetch_sub(1, Ordering::SeqCst);
            GLOBAL_METRICS.in_flight.fetch_sub(1, Ordering::Relaxed);
            if core.in_flight_waiters.load(Ordering::SeqCst) > 0 {
                let _guard = lock(&core.in_flight_lock);
                core.settled.notify_all();
            }
        })
    }

    fn build_staged_job(self: &Arc<Self>, op: &mut ReadOp, input: InputBuffer, decode: bool, read_time: Duration) -> crate::pool::Job {
        self.build_staged_job_from_parts(*op.read.get(), op.pages.take(), input, decode, read_time)
    }

    /// Hands decode jobs to the shared decode pool under one queue lock.
    fn submit_jobs(self: &Arc<Self>, jobs: Vec<crate::pool::Job>) {
        if !jobs.is_empty() {
            self.wait_for_in_flight_room(jobs.len());
            crate::pool::run_batch(jobs);
        }
    }

    /// Writes the plain-destination `COPY` members of a coalesced read straight from `master` (their
    /// adjacent destinations share one write pin and note), and turns the rest into decode jobs.
    fn complete_coalesced_group(self: &Arc<Self>, batch: &mut [ReadOp], group: &CoalescedGroup, master: PooledBuffer, read_time: Duration, staged_jobs: &mut Vec<crate::pool::Job>) {
        let per_read_time = read_time / (group.members.len() as u32).max(1);
        let mut direct = Vec::new();
        let mut staged = Vec::new();
        for &idx in &group.members {
            let op = &batch[idx];
            let r = *op.read.get();
            if r.codec == codec::COPY && op.pages.is_none() {
                let rel_start = (r.offset - group.start) as usize;
                // SAFETY: `rel_start + source_size <= group.span` by construction of the group.
                let source = unsafe { master.as_ptr().add(rel_start) };
                direct.push(DirectCopy { destination: r.destination, source, len: r.destination_size, source_size: r.source_size });
            } else {
                staged.push(idx);
            }
        }
        self.copy_direct_and_record(&direct, per_read_time);
        if staged.is_empty() {
            return;
        }
        let shared = Arc::new(master);
        for idx in staged {
            let op = &mut batch[idx];
            let r = *op.read.get();
            let rel_start = (r.offset - group.start) as usize;
            staged_jobs.push(self.build_staged_job(op, InputBuffer::Shared { buf: shared.clone(), start: rel_start, len: r.source_size as usize }, r.codec != codec::COPY, per_read_time));
        }
    }

    /// Copies `direct` under merged write pins and records one timing line per request.
    fn copy_direct_and_record(&self, direct: &[DirectCopy], per_read_time: Duration) {
        if direct.is_empty() {
            return;
        }
        // SAFETY: each source points into a coalesced read buffer that holds the request's bytes and each destination is writable for the request's size (the title's contract); buffer and destinations are distinct memory.
        let write_time = unsafe { copy_direct_merged(direct) };
        let per_write_time = write_time / (direct.len() as u32);
        for item in direct {
            self.timing.record_and_log(item.source_size, item.len, per_read_time, Duration::ZERO, per_write_time);
        }
    }
    fn read_coalesced_cpu(self: &Arc<Self>, batch: &mut [ReadOp], group: &CoalescedGroup) {
        let first_op = &batch[group.members[0]];
        let Ok(mut master) = buffer_capacity(group.span) else {
            for &idx in &group.members {
                let op = &batch[idx];
                log_failure(op.read.get(), op.pages.is_some(), E_OUTOFMEMORY);
                self.fail(E_OUTOFMEMORY);
            }
            return;
        };
        let read_started = Instant::now();
        let hr = read_file_cpu(first_op.file.handle.0, group.start, group.span, master.as_mut_ptr().cast());
        let read_time = read_started.elapsed();
        if hr < 0 {
            for &idx in &group.members {
                let op = &batch[idx];
                log_failure(op.read.get(), op.pages.is_some(), hr);
            }
            self.fail(hr);
            return;
        }
        // SAFETY: `read_file_cpu` verified `got == group.span`.
        unsafe { master.set_len(group.span as usize) };
        let mut staged_jobs = Vec::new();
        self.complete_coalesced_group(batch, group, master, read_time, &mut staged_jobs);
        self.submit_jobs(staged_jobs);
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
        let batch_started = Instant::now();
        if batch.len() == 1 {
            let op = &mut batch[0];
            let to_pages = op.pages.is_some();
            let sink = StagedSink::new();
            let hr = self.read_cpu(op, &sink);
            sink.flush(self);
            if hr < 0 {
                log_failure(op.read.get(), to_pages, hr);
                self.fail(hr);
            }
            self.timing.record_batch(batch_started.elapsed());
            return;
        }
        let plan = plan_coalesced_batches(batch, false);
        let total_tasks = plan.groups.len() + plan.singles.len();
        if total_tasks == 1 && plan.groups.len() == 1 {
            self.read_coalesced_cpu(batch, &plan.groups[0]);
            self.timing.record_batch(batch_started.elapsed());
            return;
        }
        let base = batch.as_mut_ptr() as usize;
        let batch_len = batch.len();
        let core = Sendable(self.clone());
        let groups_len = plan.groups.len();
        let sink = StagedSink::new();
        io_pool().run(total_tasks, |task_idx| {
            if task_idx < groups_len {
                // SAFETY: each group's member indices are disjoint across all tasks, and `batch` lives until `run` returns.
                let batch_slice = unsafe { std::slice::from_raw_parts_mut(base as *mut ReadOp, batch_len) };
                core.get().read_coalesced_cpu(batch_slice, &plan.groups[task_idx]);
            } else {
                let index = plan.singles[task_idx - groups_len];
                // SAFETY: `index` is disjoint from all other tasks, and `batch` lives until `run` returns.
                let op = unsafe { &mut *(base as *mut ReadOp).add(index) };
                let to_pages = op.pages.is_some();
                let hr = core.get().read_cpu(op, &sink);
                if hr < 0 {
                    log_failure(op.read.get(), to_pages, hr);
                    core.get().fail(hr);
                }
            }
        });
        sink.flush(self);
        self.timing.record_batch(batch_started.elapsed());
    }

    fn complete_coalesced_parts(
        self: &Arc<Self>,
        group_start: u64,
        members: Vec<(Sendable<DecantRead>, Option<PageTarget>)>,
        master: PooledBuffer,
        read_time: Duration,
        staged_jobs: &mut Vec<crate::pool::Job>,
    ) {
        let per_read_time = read_time / (members.len() as u32).max(1);
        let mut direct = Vec::new();
        let mut staged = Vec::new();
        for (read, pages) in members {
            let r = *read.get();
            if r.codec == codec::COPY && pages.is_none() {
                let rel_start = (r.offset - group_start) as usize;
                // SAFETY: `rel_start + source_size <= span` by construction of the group.
                let source = unsafe { master.as_ptr().add(rel_start) };
                direct.push(DirectCopy { destination: r.destination, source, len: r.destination_size, source_size: r.source_size });
            } else {
                staged.push((r, pages));
            }
        }
        self.copy_direct_and_record(&direct, per_read_time);
        if staged.is_empty() {
            return;
        }
        let shared = Arc::new(master);
        for (r, pages) in staged {
            let rel_start = (r.offset - group_start) as usize;
            staged_jobs.push(self.build_staged_job_from_parts(r, pages, InputBuffer::Shared { buf: shared.clone(), start: rel_start, len: r.source_size as usize }, r.codec != codec::COPY, per_read_time));
        }
    }
    fn finish_native_in_flight(self: &Arc<Self>, native: &Native, submitted: NativeInFlight) {
        // SAFETY: `native.done[slot]` and `native.status` are live objects created in `Queue::create`.
        let hr = unsafe {
            if WaitForSingleObject(native.done[submitted.slot].0, INFINITE) != WAIT_OBJECT_0 {
                last_error()
            } else {
                (native.status.vtbl::<pc::StatusArrayVtbl>().get_hresult)(native.status.as_ptr(), submitted.slot as u32).min(S_OK)
            }
        };
        let batch_read_time = submitted.started.elapsed();
        self.timing.record_batch(batch_read_time);
        let per_read_time = batch_read_time / (submitted.pending.len() as u32).max(1);
        if hr < 0 {
            self.fail(hr);
        }

        let pin_started = Instant::now();
        for pin in &submitted.pins {
            if pin.pinned {
                finish_written(pin.start as *mut c_void, pin.len);
            }
            if hr >= 0 {
                note_written(pin.start as *mut c_void, pin.len);
            }
        }
        let direct_count = submitted.pending.iter().filter(|item| matches!(item, PendingNative::Direct { .. })).count();
        let per_direct_write_time = pin_started.elapsed() / (direct_count as u32).max(1);
        let mut staged_jobs = Vec::new();
        for item in submitted.pending {
            match item {
                PendingNative::Direct { destination_size, source_size } => {
                    if hr >= 0 {
                        self.timing.record_and_log(source_size, destination_size, per_read_time, Duration::ZERO, per_direct_write_time);
                    }
                }
                PendingNative::Staged { read, pages, mut input, len, decode } => {
                    if hr < 0 {
                        continue;
                    }
                    // SAFETY: `IDStorageQueue1` wrote all `len` bytes into `input` before signalling `native.done`.
                    unsafe { input.set_len(len as usize) };
                    staged_jobs.push(self.build_staged_job_from_parts(*read.get(), pages, InputBuffer::Owned(input), decode, per_read_time));
                }
                PendingNative::Coalesced { start, span, members, mut master } => {
                    if hr < 0 {
                        continue;
                    }
                    // SAFETY: `IDStorageQueue1` wrote all `span` bytes into `master` before signalling `native.done`.
                    unsafe { master.set_len(span as usize) };
                    self.complete_coalesced_parts(start, members, master, per_read_time, &mut staged_jobs);
                }
            }
        }
        self.submit_jobs(staged_jobs);
    }

    fn flush_native(self: &Arc<Self>) {
        let Some(native) = self.native.get() else { return };
        if let Some(submitted) = lock(&native.in_flight).take() {
            self.finish_native_in_flight(native, submitted);
        }
    }

    fn has_native_in_flight(&self) -> bool {
        self.native.get().is_some_and(|n| lock(&n.in_flight).is_some())
    }

    /// Batches up to `READ_BATCH_LIMIT` consecutive reads into one `IDStorageQueue1` submission,
    /// overlapping submission of batch `N+1` with completion/decode of batch `N`.
    fn read_batch_native(self: &Arc<Self>, native: &Native, batch: &mut [ReadOp]) {
        let plan = if batch.len() >= 2 { Some(plan_coalesced_batches(batch, true)) } else { None };
        let mut pending = Vec::with_capacity(plan.as_ref().map_or(batch.len(), |p| p.groups.len() + p.singles.len()));
        let queue = native.queue.as_ptr();
        // SAFETY: `native.queue` is a live `IDStorageQueue1` owned by `Core`.
        let vtbl = unsafe { native.queue.vtbl::<pc::QueueVtbl>() };
        let started = Instant::now();
        let singles: Vec<usize> = match &plan {
            Some(plan) => plan.singles.clone(),
            None => (0..batch.len()).collect(),
        };
        let pins = pin_direct_destinations(batch, &singles);

        let enqueue_single = |op: &mut ReadOp, pending: &mut Vec<PendingNative>| {
            let r = *op.read.get();
            if !matches!(r.codec, codec::COPY | codec::ZLIB | codec::GDEFLATE | codec::BCPACK) {
                decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("read with codec {}", r.codec));
                log_failure(&r, op.pages.is_some(), E_NOTIMPL);
                self.fail(E_NOTIMPL);
                return;
            }
            if r.codec == codec::COPY && r.source_size != r.destination_size {
                log_failure(&r, op.pages.is_some(), E_INVALIDARG);
                self.fail(E_INVALIDARG);
                return;
            }
            let Some(file_native) = op.file.native.as_ref() else {
                log_failure(&r, op.pages.is_some(), E_INVALIDARG);
                self.fail(E_INVALIDARG);
                return;
            };
            if let Some(length) = op.file.byte_length
                && (r.offset > length || u64::from(r.source_size) > length - r.offset)
            {
                let eof = hresult_from_win32(ERROR_HANDLE_EOF);
                log_failure(&r, op.pages.is_some(), eof);
                self.fail(eof);
                return;
            }
            let decode = r.codec == codec::ZLIB || r.codec == codec::BCPACK;
            let compression = if r.codec == codec::GDEFLATE { pc::COMPRESSION_GDEFLATE } else { 0 };
            if !decode && op.pages.is_none() {
                // The destination is pinned by pin_direct_destinations before any request is enqueued.
                let request = pc::RawRequest::file_to_memory(compression, file_native.as_ptr(), r.offset, r.source_size, r.destination, r.destination_size);
                // SAFETY: `r.destination` remains valid until the completion wait in `finish_native_in_flight` returns.
                unsafe { (vtbl.enqueue_request)(queue, &request) };
                pending.push(PendingNative::Direct { destination_size: r.destination_size, source_size: r.source_size });
            } else {
                let len = if decode { r.source_size } else { r.destination_size };
                let Ok(mut input) = buffer_capacity(len) else {
                    log_failure(&r, op.pages.is_some(), E_OUTOFMEMORY);
                    self.fail(E_OUTOFMEMORY);
                    return;
                };
                let request = pc::RawRequest::file_to_memory(compression, file_native.as_ptr(), r.offset, r.source_size, input.as_mut_ptr().cast(), len);
                // SAFETY: `input` buffer stays alive in `pending` until `finish_native_in_flight` returns.
                unsafe { (vtbl.enqueue_request)(queue, &request) };
                pending.push(PendingNative::Staged { read: Sendable(r), pages: op.pages.take(), input, len, decode });
            }
        };

        if let Some(plan) = plan {
            for group in plan.groups {
                let first_op = &batch[group.members[0]];
                let Some(file_native) = first_op.file.native.as_ref() else { continue };
                let file_ptr = file_native.as_ptr();
                let Ok(mut master) = buffer_capacity(group.span) else {
                    for &idx in &group.members {
                        let op = &batch[idx];
                        log_failure(op.read.get(), op.pages.is_some(), E_OUTOFMEMORY);
                        self.fail(E_OUTOFMEMORY);
                    }
                    continue;
                };
                let request = pc::RawRequest::file_to_memory(0, file_ptr, group.start, group.span, master.as_mut_ptr().cast(), group.span);
                // SAFETY: `master` buffer stays alive in `pending` until `finish_native_in_flight` returns.
                unsafe { (vtbl.enqueue_request)(queue, &request) };
                let members = group.members.into_iter().map(|idx| {
                    let op = &mut batch[idx];
                    (Sendable(*op.read.get()), op.pages.take())
                }).collect();
                pending.push(PendingNative::Coalesced { start: group.start, span: group.span, members, master });
            }
            for index in plan.singles {
                enqueue_single(&mut batch[index], &mut pending);
            }
        } else {
            for op in batch.iter_mut() {
                enqueue_single(op, &mut pending);
            }
        }

        if pending.is_empty() {
            for pin in &pins {
                if pin.pinned {
                    finish_written(pin.start as *mut c_void, pin.len);
                }
            }
            return;
        }

        let slot = native.next_slot.fetch_xor(1, Ordering::Relaxed);
        // SAFETY: `native.queue`, `native.status`, and `native.done[slot]` are live objects created in `Queue::create`.
        unsafe {
            (vtbl.enqueue_status)(queue, native.status.as_ptr(), slot as u32);
            (vtbl.enqueue_set_event)(queue, native.done[slot].0);
            (vtbl.submit)(queue);
        }

        let prev = lock(&native.in_flight).replace(NativeInFlight { slot, started, pending, pins });
        if let Some(prev_submitted) = prev {
            self.finish_native_in_flight(native, prev_submitted);
        }
    }

    /// Runs one read on the Win32 CPU backend using a reusable thread local event handle.
    fn read_cpu(self: &Arc<Self>, op: &mut ReadOp, sink: &StagedSink) -> HRESULT {
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
        sink.push(self, self.build_staged_job(op, InputBuffer::Owned(input), decode, read_time));
        S_OK
    }
}

/// Decodes of one CPU read pass, handed to the decode pool in batches of STAGED_FLUSH (one queue
/// lock and wake per batch rather than per read). The pass flushes the rest when its reads are done,
/// before the queue worker waits for them.
struct StagedSink {
    jobs: Mutex<Vec<crate::pool::Job>>,
}

/// Decodes gathered before they go to the decode pool.
const STAGED_FLUSH: usize = 8;

impl StagedSink {
    fn new() -> Self {
        Self { jobs: Mutex::new(Vec::new()) }
    }

    fn push(&self, core: &Arc<Core>, job: crate::pool::Job) {
        let full = {
            let mut jobs = lock(&self.jobs);
            jobs.push(job);
            (jobs.len() >= STAGED_FLUSH).then(|| std::mem::take(&mut *jobs))
        };
        if let Some(full) = full {
            core.submit_jobs(full);
        }
    }

    fn flush(&self, core: &Arc<Core>) {
        let rest = std::mem::take(&mut *lock(&self.jobs));
        core.submit_jobs(rest);
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
        // Another queue is fanning out already: do not wait for it, read here on this queue's own thread.
        let _guard = match self.run_lock.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                for i in 0..total {
                    f(i);
                }
                return;
            }
        };
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

/// Pins writable, before they are enqueued, the destinations of the singles that go to the native queue
/// as direct requests (no decode, no page destination). Destinations that touch or overlap share one
/// pin; the pins are released when the batch completed. The conditions are those of the direct branch
/// of `read_batch_native`, so every pinned range is written by a request that is enqueued.
fn pin_direct_destinations(batch: &[ReadOp], singles: &[usize]) -> Vec<WritePin> {
    let ranges: Vec<(usize, u32)> = singles
        .iter()
        .filter_map(|&idx| {
            let op = &batch[idx];
            let r = op.read.get();
            let valid = matches!(r.codec, codec::COPY | codec::GDEFLATE) && !(r.codec == codec::COPY && r.source_size != r.destination_size);
            let in_bounds = op.file.byte_length.is_none_or(|len| r.offset <= len && u64::from(r.source_size) <= len - r.offset);
            (valid && in_bounds && op.file.native.is_some() && op.pages.is_none()).then_some((r.destination as usize, r.destination_size))
        })
        .collect();
    crate::spans::merge(&ranges)
        .into_iter()
        .map(|span| WritePin { start: span.start, len: span.len, pinned: begin_written(span.start as *mut c_void, span.len) })
        .collect()
}

/// A pooled buffer with capacity for at least `len` bytes, or `Err` when it cannot be allocated.
fn buffer_capacity(len: u32) -> Result<PooledBuffer, ()> {
    buffer_pool().acquire_cached(len)
}

impl Queue {
    /// A queue and its worker thread, and, on PC DirectStorage, the native queue behind it.
    fn create(backend: Arc<Backend>, pages: Arc<dyn PagePool>, capacity: u16) -> Result<Queue, HRESULT> {
        let core = Arc::new(Core {
            backend,
            pages,
            native: OnceLock::new(),
            interval: AtomicI32::new(S_OK),
            in_flight_count: AtomicUsize::new(0),
            in_flight_waiters: AtomicUsize::new(0),
            in_flight_lock: Mutex::new(()),
            settled: Condvar::new(),
            timing: Arc::new(Timing::new()),
        });
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
        let native_capacity = capacity.max(pc::MIN_QUEUE_CAPACITY).clamp(2048, 4096);
        let desc = pc::QueueDesc { source_type: 0, capacity: native_capacity, priority: 0, name: std::ptr::null(), device: std::ptr::null_mut() };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the factory is an `IDStorageFactory`, `desc` and `raw` live for the call.
        let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().create_queue)(factory.as_ptr(), &desc, &pc::IID_QUEUE1, &mut raw) };
        // SAFETY: on success `raw` is a queue reference we now own.
        let native = unsafe { Pc::from_raw(raw) };
        if hr < 0 {
            return Err(hr);
        }
        // SAFETY: plain Win32 calls.
        let done0 = Handle(unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) });
        if done0.0.is_null() {
            return Err(last_error());
        }
        let done1 = Handle(unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) });
        if done1.0.is_null() {
            return Err(last_error());
        }
        decant_log::handles::created(done0.0 as usize, "queue completion event 0");
        decant_log::handles::created(done1.0 as usize, "queue completion event 1");
        let mut raw = std::ptr::null_mut();
        // SAFETY: as above, with a zero terminated name.
        let hr = unsafe { (factory.vtbl::<pc::FactoryVtbl>().create_status_array)(factory.as_ptr(), 2, c"DecantStorage".as_ptr().cast(), &pc::IID_STATUS_ARRAY, &mut raw) };
        // SAFETY: on success `raw` is a status array reference we now own.
        let status = unsafe { Pc::from_raw(raw) };
        if let (Some(queue_pc), Some(status)) = (native, status) {
            let _ = queue.core.native.set(Native { queue: queue_pc, status, done: [done0, done1], in_flight: Mutex::new(None), next_slot: AtomicUsize::new(0) });
        }
        if hr < 0 { Err(hr) } else { Ok(queue) }
    }

    /// Hands every recorded operation to the worker. The caller holds the lock of `pending`.
    fn submit_locked(&self, pending: &mut VecDeque<QueueOp>) -> HRESULT {
        if pending.is_empty() {
            return S_OK;
        }
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
        if state.submitted.is_empty() && !state.stopping && core.has_native_in_flight() {
            drop(state);
            core.flush_native();
            state = lock(&work.state);
            continue;
        }
        state = work.ready.wait_while(state, |s| !s.stopping && s.submitted.is_empty()).unwrap_or_else(PoisonError::into_inner);
        if state.submitted.is_empty() {
            drop(state);
            core.flush_native();
            // The decodes handed on still write into the title's memory: the queue is closed only after them.
            core.settle();
            return;
        }
        if state.stopping {
            let rest = std::mem::take(&mut state.submitted);
            drop(state);
            core.flush_native();
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
        core.flush_native();
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

#[cfg(test)]
#[test]
fn buffer_pool_reuses_released_buffers() {
    let pool = BufferPool::new();
    let mut first = pool.acquire(1000).expect("allocates");
    assert!(first.capacity() >= 4096);
    let ptr = first.as_ptr();
    let cap = first.capacity();
    first.extend_from_slice(b"hello");
    let raw = std::mem::take(&mut first.0);
    pool.release(raw);
    let second = pool.acquire(2000).expect("reuses");
    assert_eq!(second.capacity(), cap);
    assert_eq!(second.as_ptr(), ptr);
    assert!(second.is_empty());
}

#[cfg(test)]
#[test]
fn thread_cache_reuses_small_buffers_and_stays_within_its_caps() {
    std::thread::spawn(|| {
        let pool = BufferPool::new();
        assert_eq!(tls_cached_bytes(), 0);

        // A released small buffer is reused on the same thread, without the shared pool.
        let mut first = pool.acquire_cached(1000).expect("allocates");
        let ptr = first.as_ptr();
        let raw = std::mem::take(&mut first.0);
        pool.release_cached(raw);
        assert_eq!(tls_cached_bytes(), 4096);
        assert!(lock(&pool.inner).buckets.iter().all(Vec::is_empty), "the shared pool was not used");
        let mut again = pool.acquire_cached(3000).expect("reuses");
        assert_eq!(again.as_ptr(), ptr);
        assert!(again.is_empty());
        assert_eq!(tls_cached_bytes(), 0);
        drop(std::mem::take(&mut again.0));

        // Per size class: at most `TLS_MAX_PER_BUCKET` stay, the rest go to the shared pool.
        for _ in 0..TLS_MAX_PER_BUCKET + 4 {
            pool.release_cached(Vec::with_capacity(8192));
        }
        assert_eq!(tls_cached_bytes(), TLS_MAX_PER_BUCKET * 8192);
        assert_eq!(lock(&pool.inner).buckets[1].len(), 4);
        // In total: never more than `TLS_MAX_BYTES`, whatever the size classes.
        let sizes = [8192usize, 16_384, 32_768, 65_536, 131_072, 262_144];
        for k in 0..60 {
            pool.release_cached(Vec::with_capacity(sizes[k % sizes.len()]));
            assert!(tls_cached_bytes() <= TLS_MAX_BYTES);
        }

        // A buffer larger than `TLS_MAX_BUFFER` is not kept by the thread.
        let before = tls_cached_bytes();
        pool.release_cached(Vec::with_capacity(TLS_MAX_BUFFER * 2));
        assert_eq!(tls_cached_bytes(), before);
        let bucket = (TLS_MAX_BUFFER * 2).trailing_zeros() as usize - POOL_MIN_SHIFT as usize;
        assert_eq!(lock(&pool.inner).buckets[bucket].len(), 1);
    })
    .join()
    .expect("thread cache test");
}

#[cfg(test)]
#[test]
fn merged_direct_copies_fill_adjacent_destinations_exactly() {
    let source: Vec<u8> = (0..4096u32).map(|i| (i * 7 + 3) as u8).collect();
    // Destination bytes outside every request stay untouched.
    let mut destination = vec![0xEEu8; 8192];
    let base = destination.as_mut_ptr();
    // Requests: [0..1000) [1000..2500) [2500..3000) (adjacent), a gap, [3500..4096).
    let spans = [(0usize, 0usize, 1000u32), (1000, 1000, 1500), (2500, 2500, 500), (3500, 3500, 596)];
    let items: Vec<DirectCopy> = spans
        .iter()
        .map(|&(src, dst, len)| DirectCopy {
            // SAFETY: inside the buffers above.
            destination: unsafe { base.add(dst) }.cast(),
            source: unsafe { source.as_ptr().add(src) },
            len,
            source_size: len,
        })
        .collect();
    // SAFETY: sources and destinations are the live, distinct buffers above.
    unsafe { copy_direct_merged(&items) };
    assert_eq!(&destination[..3000], &source[..3000]);
    assert!(destination[3000..3500].iter().all(|&b| b == 0xEE), "the gap is not written");
    assert_eq!(&destination[3500..4096], &source[3500..4096]);
    assert!(destination[4096..].iter().all(|&b| b == 0xEE));
}

#[cfg(test)]
#[test]
fn overlapping_direct_copies_keep_request_order() {
    let first = [1u8; 64];
    let second = [2u8; 64];
    let mut destination = vec![0u8; 128];
    let base = destination.as_mut_ptr();
    let items = [
        DirectCopy { destination: base.cast(), source: first.as_ptr(), len: 64, source_size: 64 },
        // SAFETY: inside `destination`.
        DirectCopy { destination: unsafe { base.add(32) }.cast(), source: second.as_ptr(), len: 64, source_size: 64 },
    ];
    // SAFETY: live, distinct buffers.
    unsafe { copy_direct_merged(&items) };
    assert!(destination[..32].iter().all(|&b| b == 1));
    assert!(destination[32..96].iter().all(|&b| b == 2), "the later request wins where they overlap");
    assert!(destination[96..].iter().all(|&b| b == 0));
}

#[cfg(test)]
#[test]
fn adjacent_reads_coalesce_and_complete_correctly() {
    use std::io::Write;
    use std::os::windows::ffi::OsStrExt;
    let mut data = vec![0u8; 64 * 1024];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i * 17 + i / 257) as u8;
    }
    let path = std::env::temp_dir().join(format!("dstorage_coalesce_{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("creates temp file");
        f.write_all(&data).expect("writes temp file");
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let factory = Ref::new(Factory::new(Backend::cpu(), Arc::new(XmemPool)));
    let file = factory.open_file(wide.as_ptr()).expect("opens file");
    let queue = factory.create_queue(16).expect("creates queue");
    let status = factory.create_status_array(1).expect("creates status");

    let mut d0 = vec![0u8; 4096];
    let mut d1 = vec![0u8; 4096];
    let mut d2 = vec![0u8; 4096];
    // Out-of-order and small-gap adjacent reads in the same file.
    for (offset, dst) in [(8192u64, &mut d2), (0u64, &mut d0), (4096u64, &mut d1)] {
        let r = DecantRead {
            file: file.as_ptr().cast(),
            offset,
            source_size: 4096,
            destination: dst.as_mut_ptr().cast(),
            destination_size: 4096,
            codec: codec::COPY,
            bc_mode: 0,
        };
        assert_eq!(queue.enqueue(r), S_OK);
    }
    // SAFETY: `status` is one of ours.
    assert_eq!(unsafe { queue.enqueue_status(status.as_ptr().cast(), 0) }, S_OK);
    assert_eq!(queue.submit(), S_OK);
    let start = Instant::now();
    while status.result(0) == E_PENDING {
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(status.result(0), S_OK);
    assert_eq!(d0, data[0..4096]);
    assert_eq!(d1, data[4096..8192]);
    assert_eq!(d2, data[8192..12288]);
    let _ = std::fs::remove_file(&path);
}

#[cfg(test)]
#[test]
fn multi_batch_pipelined_reads_settle_before_status_barrier() {
    use std::io::Write;
    let total_reads = READ_BATCH_LIMIT + 32;
    let chunk_size = 1024usize;
    let data: Vec<u8> = (0..total_reads * chunk_size).map(|i| (i * 29 + i / 131) as u8).collect();
    let path = std::env::temp_dir().join(format!("dstorage_pipeline_{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("creates temp file");
        f.write_all(&data).expect("writes temp file");
    }
    let wide: Vec<u16> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    };
    let factory = Ref::new(Factory::new(Backend::cpu(), Arc::new(XmemPool)));
    let file = factory.open_file(wide.as_ptr()).expect("opens file");
    let queue = factory.create_queue((total_reads + 4) as u16).expect("creates queue");
    let status = factory.create_status_array(1).expect("creates status");

    let mut bufs: Vec<Vec<u8>> = (0..total_reads).map(|_| vec![0u8; chunk_size]).collect();
    for (i, buf) in bufs.iter_mut().enumerate() {
        let r = DecantRead {
            file: file.as_ptr().cast(),
            offset: (i * chunk_size) as u64,
            source_size: chunk_size as u32,
            destination: buf.as_mut_ptr().cast(),
            destination_size: chunk_size as u32,
            codec: codec::COPY,
            bc_mode: 0,
        };
        assert_eq!(queue.enqueue(r), S_OK);
    }
    // SAFETY: `status` is one of ours.
    assert_eq!(unsafe { queue.enqueue_status(status.as_ptr().cast(), 0) }, S_OK);
    assert_eq!(queue.submit(), S_OK);
    let start = Instant::now();
    while status.result(0) == E_PENDING {
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(status.result(0), S_OK);
    for (i, buf) in bufs.iter().enumerate() {
        assert_eq!(buf.as_slice(), &data[i * chunk_size..(i + 1) * chunk_size], "mismatch at chunk {i}");
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg(test)]
#[test]
fn coalesces_across_64k_gap_and_reuses_decode_scratch() {
    use std::io::Write;
    let mut data = vec![0u8; 256 * 1024];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i * 11 + i / 97) as u8;
    }
    let path = std::env::temp_dir().join(format!("dstorage_gap64k_{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("creates temp file");
        f.write_all(&data).expect("writes temp file");
    }
    let wide: Vec<u16> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    };
    let factory = Ref::new(Factory::new(Backend::cpu(), Arc::new(XmemPool)));
    let file = factory.open_file(wide.as_ptr()).expect("opens file");
    let op0 = ReadOp {
        file: file.clone(),
        read: Sendable(DecantRead {
            file: file.as_ptr().cast(),
            offset: 0,
            source_size: 4096,
            destination: std::ptr::null_mut(),
            destination_size: 4096,
            codec: codec::COPY,
            bc_mode: 0,
        }),
        pages: None,
    };
    let op1 = ReadOp {
        file: file.clone(),
        read: Sendable(DecantRead {
            file: file.as_ptr().cast(),
            offset: 4096 + 48 * 1024, // 48 KiB gap <= 64 KiB
            source_size: 4096,
            destination: std::ptr::null_mut(),
            destination_size: 4096,
            codec: codec::COPY,
            bc_mode: 0,
        }),
        pages: None,
    };
    let plan = plan_coalesced_batches(&[op0, op1], false);
    assert_eq!(plan.groups.len(), 1);
    assert!(plan.singles.is_empty());
    assert_eq!(plan.groups[0].span, 4096 + 48 * 1024 + 4096);

    assert_eq!(with_decode_output(1024, |buf| { buf.fill(0xAA); S_OK }), S_OK);
    assert_eq!(with_decode_output(2048, |buf| { assert_eq!(buf.len(), 2048); S_OK }), S_OK);
    let _ = std::fs::remove_file(&path);
}

#[cfg(test)]
#[test]
fn metrics_snapshot_tracks_requests_bytes_and_batches() {
    use std::io::Write;
    let before = metrics_snapshot();
    let data = vec![0x5Au8; 8192];
    let path = std::env::temp_dir().join(format!("dstorage_metrics_{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("creates temp file");
        f.write_all(&data).expect("writes temp file");
    }
    let wide: Vec<u16> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
    };
    let factory = Ref::new(Factory::new(Backend::cpu(), Arc::new(XmemPool)));
    let file = factory.open_file(wide.as_ptr()).expect("opens file");
    let queue = factory.create_queue(8).expect("creates queue");
    let status = factory.create_status_array(1).expect("creates status");
    let mut dst = vec![0u8; 8192];
    let r = DecantRead {
        file: file.as_ptr().cast(),
        offset: 0,
        source_size: 8192,
        destination: dst.as_mut_ptr().cast(),
        destination_size: 8192,
        codec: codec::COPY,
        bc_mode: 0,
    };
    assert_eq!(queue.enqueue(r), S_OK);
    // SAFETY: `status` is one of ours.
    assert_eq!(unsafe { queue.enqueue_status(status.as_ptr().cast(), 0) }, S_OK);
    assert_eq!(queue.submit(), S_OK);
    let start = Instant::now();
    while status.result(0) == E_PENDING {
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(status.result(0), S_OK);
    let after = metrics_snapshot();
    assert!(after.requests > before.requests);
    assert!(after.bytes_read >= before.bytes_read + 8192);
    assert!(after.batches > before.batches);
    let _ = std::fs::remove_file(&path);
}
