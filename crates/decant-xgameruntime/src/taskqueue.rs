//! XTaskQueue of the local XAsync / XTaskQueue / XThread table (served only without a PC
//! runtime, see `local_table`). Contract: the public GDK `XTaskQueue.h`.
//!
//! Dispatch modes: Manual (the caller pumps `XTaskQueueDispatch`), ThreadPool,
//! SerializedThreadPool (pool threads, one callback at a time per port) and Immediate (run on the
//! submitting thread). `XTaskQueueTerminate` runs the queue's pending callbacks (delayed ones
//! included) with `canceled = true` before its own callback, and afterwards the queue takes no new
//! work (`E_ABORT`). Ports are shared objects: composite queues made from `XTaskQueueGetPort`
//! handles use the same port, and terminating one of them leaves the work of the others alone.
//!
//! Handles are opaque numbers looked up in a registry and never dereferenced, so a closed or
//! invented handle value is simply unknown.
#![allow(unsafe_code, non_snake_case)]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

pub(crate) const S_OK: i32 = 0;
pub(crate) const E_INVALIDARG: i32 = 0x8007_0057_u32 as i32;
pub(crate) const E_ABORT: i32 = 0x8000_4004_u32 as i32;
pub(crate) const E_FAIL: i32 = 0x8000_4005_u32 as i32;
/// The block names no queue and the process queue is disabled.
pub(crate) const E_NO_TASK_QUEUE: i32 = 0x8007_01AB_u32 as i32;

pub(crate) const MODE_MANUAL: u32 = 0;
pub(crate) const MODE_THREAD_POOL: u32 = 1;
pub(crate) const MODE_SERIALIZED_THREAD_POOL: u32 = 2;
pub(crate) const MODE_IMMEDIATE: u32 = 3;
pub(crate) const PORT_WORK: u32 = 0;
pub(crate) const PORT_COMPLETION: u32 = 1;

type TaskFn = unsafe extern "system" fn(*mut c_void, bool);
type TerminatedFn = unsafe extern "system" fn(*mut c_void);
type MonitorFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32);

fn grab<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn park<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(g).unwrap_or_else(PoisonError::into_inner)
}

fn park_for<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>, d: Duration) -> MutexGuard<'a, T> {
    cv.wait_timeout(g, d).unwrap_or_else(PoisonError::into_inner).0
}

/// A title callback and its context, kept as addresses so they can cross threads.
#[derive(Clone, Copy)]
pub(crate) struct Callback {
    pub(crate) func: usize,
    pub(crate) ctx: usize,
}

impl Callback {
    /// # Safety
    /// `func` is an `XTaskQueueCallback` that accepts `ctx`.
    unsafe fn invoke(self, canceled: bool) {
        // SAFETY: the caller's contract.
        unsafe {
            let f = std::mem::transmute::<usize, TaskFn>(self.func);
            f(self.ctx as *mut c_void, canceled);
        }
    }
}

/// One submitted callback waiting on a port, with the queue it was submitted through.
struct Entry {
    call: Callback,
    owner: Arc<Queue>,
    which: usize,
}

#[derive(Default)]
struct Lane {
    entries: VecDeque<Entry>,
    /// Serialized ports only: a pool job is currently draining `entries`.
    draining: bool,
}

pub(crate) struct Port {
    mode: u32,
    lane: Mutex<Lane>,
    arrived: Condvar,
}

impl Port {
    fn new(mode: u32) -> Arc<Port> {
        Arc::new(Port { mode, lane: Mutex::new(Lane::default()), arrived: Condvar::new() })
    }
}

#[derive(Default)]
struct Books {
    terminated: bool,
    /// Callbacks submitted and not yet finished, per port.
    open: [usize; 2],
}

#[derive(Clone, Copy)]
struct Monitor {
    token: u64,
    handle: usize,
    ctx: usize,
    func: usize,
}

pub(crate) struct Queue {
    ports: [Arc<Port>; 2],
    books: Mutex<Books>,
    quiet: Condvar,
    monitors: Mutex<Vec<Monitor>>,
}

impl Queue {
    pub(crate) fn new(work_mode: u32, completion_mode: u32) -> Arc<Queue> {
        Self::over(Port::new(work_mode), Port::new(completion_mode))
    }

    fn over(work: Arc<Port>, completion: Arc<Port>) -> Arc<Queue> {
        Arc::new(Queue { ports: [work, completion], books: Mutex::new(Books::default()), quiet: Condvar::new(), monitors: Mutex::new(Vec::new()) })
    }

    fn is_terminated(&self) -> bool {
        grab(&self.books).terminated
    }

    /// Records one more open callback on `which`, unless the queue is terminated.
    fn admit(&self, which: usize) -> bool {
        let mut b = grab(&self.books);
        if b.terminated {
            return false;
        }
        b.open[which] += 1;
        true
    }

    fn settle(&self, which: usize) {
        let mut b = grab(&self.books);
        b.open[which] -= 1;
        if b.open == [0, 0] {
            self.quiet.notify_all();
        }
    }

    fn tell_monitors(&self, which: usize) {
        let all = grab(&self.monitors).clone();
        for m in all {
            // SAFETY: `XTaskQueueRegisterMonitor` took `func` as an `XTaskQueueMonitorCallback`.
            unsafe {
                let f = std::mem::transmute::<usize, MonitorFn>(m.func);
                f(m.ctx as *mut c_void, m.handle as *mut c_void, which as u32);
            }
        }
    }
}

/// Runs an entry and settles its books.
///
/// # Safety
/// As [`Callback::invoke`].
unsafe fn run_entry(e: Entry) {
    let canceled = e.owner.is_terminated();
    // SAFETY: the caller's contract.
    unsafe { e.call.invoke(canceled) };
    e.owner.settle(e.which);
}

// Shared pool threads and the delay timer.

type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    wake: Condvar,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let p: &'static Pool = Box::leak(Box::new(Pool { jobs: Mutex::new(VecDeque::new()), wake: Condvar::new() }));
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(2, 8);
        for i in 0..threads {
            std::thread::Builder::new()
                .name(format!("decant-xgr-taskqueue-{i}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let mut jobs = grab(&p.jobs);
                            loop {
                                if let Some(j) = jobs.pop_front() {
                                    break j;
                                }
                                jobs = park(&p.wake, jobs);
                            }
                        };
                        job();
                    }
                })
                .expect("spawn task queue worker");
        }
        p
    })
}

fn run_on_pool(job: Job) {
    let p = pool();
    grab(&p.jobs).push_back(job);
    p.wake.notify_one();
}

struct Delayed {
    owner: Arc<Queue>,
    which: usize,
    call: Callback,
}

struct Timer {
    due: Mutex<BTreeMap<(Instant, u64), Delayed>>,
    changed: Condvar,
}

fn timer() -> &'static Timer {
    static TIMER: OnceLock<&'static Timer> = OnceLock::new();
    TIMER.get_or_init(|| {
        let t: &'static Timer = Box::leak(Box::new(Timer { due: Mutex::new(BTreeMap::new()), changed: Condvar::new() }));
        std::thread::Builder::new()
            .name("decant-xgr-taskqueue-timer".into())
            .spawn(move || {
                let mut due = grab(&t.due);
                loop {
                    let now = Instant::now();
                    let Some((&(at, _), _)) = due.first_key_value() else {
                        due = park(&t.changed, due);
                        continue;
                    };
                    if at > now {
                        due = park_for(&t.changed, due, at - now);
                        continue;
                    }
                    if let Some((_, d)) = due.pop_first() {
                        drop(due);
                        enqueue(&d.owner, d.which, d.call);
                        due = grab(&t.due);
                    }
                }
            })
            .expect("spawn timer");
        t
    })
}

static TIMER_SEQ: AtomicU64 = AtomicU64::new(0);
static MONITOR_SEQ: AtomicU64 = AtomicU64::new(1);

/// Drains a serialized port on a pool thread, one callback after the other.
fn drain(port: Arc<Port>) {
    loop {
        let next = {
            let mut lane = grab(&port.lane);
            match lane.entries.pop_front() {
                Some(e) => e,
                None => {
                    lane.draining = false;
                    return;
                }
            }
        };
        // SAFETY: entries come from the submit functions with title callbacks.
        unsafe { run_entry(next) };
    }
}

/// Hands a callback to port `which` of `q` right now. False when `q` is terminated. Locks are
/// always taken in the order port lane (or timer table), then queue books.
pub(crate) fn enqueue(q: &Arc<Queue>, which: usize, call: Callback) -> bool {
    let port = &q.ports[which];
    match port.mode {
        MODE_IMMEDIATE | MODE_THREAD_POOL => {
            if !q.admit(which) {
                return false;
            }
            q.tell_monitors(which);
            let entry = Entry { call, owner: q.clone(), which };
            if port.mode == MODE_IMMEDIATE {
                // SAFETY: a title callback with its context.
                unsafe { run_entry(entry) };
            } else {
                // SAFETY: as above.
                run_on_pool(Box::new(move || unsafe { run_entry(entry) }));
            }
        }
        MODE_SERIALIZED_THREAD_POOL => {
            let start_drain = {
                let mut lane = grab(&port.lane);
                if !q.admit(which) {
                    return false;
                }
                lane.entries.push_back(Entry { call, owner: q.clone(), which });
                !std::mem::replace(&mut lane.draining, true)
            };
            q.tell_monitors(which);
            if start_drain {
                let p = port.clone();
                run_on_pool(Box::new(move || drain(p)));
            }
        }
        _ => {
            {
                let mut lane = grab(&port.lane);
                if !q.admit(which) {
                    return false;
                }
                lane.entries.push_back(Entry { call, owner: q.clone(), which });
            }
            port.arrived.notify_one();
            q.tell_monitors(which);
        }
    }
    true
}

/// Hands a callback to port `which` of `q` after `delay_ms`. False when `q` is terminated.
pub(crate) fn enqueue_after(q: &Arc<Queue>, which: usize, delay_ms: u32, call: Callback) -> bool {
    if delay_ms == 0 {
        return enqueue(q, which, call);
    }
    let t = timer();
    let mut due = grab(&t.due);
    if q.is_terminated() {
        return false;
    }
    let key = (Instant::now() + Duration::from_millis(u64::from(delay_ms)), TIMER_SEQ.fetch_add(1, Ordering::Relaxed));
    due.insert(key, Delayed { owner: q.clone(), which, call });
    drop(due);
    t.changed.notify_all();
    true
}

/// Runs one callback of a Manual port, waiting up to `timeout_ms` for it. True when one ran.
fn pump(q: &Arc<Queue>, which: usize, timeout_ms: u32) -> bool {
    let port = &q.ports[which];
    if port.mode != MODE_MANUAL {
        return false;
    }
    let mut lane = grab(&port.lane);
    if lane.entries.is_empty() && timeout_ms != 0 {
        let patience = if timeout_ms == u32::MAX { Duration::from_secs(365 * 24 * 3600) } else { Duration::from_millis(u64::from(timeout_ms)) };
        let deadline = Instant::now() + patience;
        while lane.entries.is_empty() && !q.is_terminated() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            lane = park_for(&port.arrived, lane, deadline - now);
        }
    }
    let Some(entry) = lane.entries.pop_front() else { return false };
    drop(lane);
    // SAFETY: the entry holds a title callback.
    unsafe { run_entry(entry) };
    true
}

// Handle registry. Queue and port handles are numbers handed to the title and looked up here.

const HANDLE_BASE: usize = 0x5100_0000;
const HANDLE_STEP: usize = 0x10;

#[derive(Default)]
struct Registry {
    issued: usize,
    queues: HashMap<usize, Arc<Queue>>,
    /// The GDK has no close call for port handles, so they only refer weakly to their port.
    ports: HashMap<usize, Weak<Port>>,
}

impl Registry {
    fn fresh(&mut self) -> usize {
        self.issued += 1;
        HANDLE_BASE + self.issued * HANDLE_STEP
    }
}

fn registry() -> MutexGuard<'static, Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    grab(REG.get_or_init(|| Mutex::new(Registry::default())))
}

fn port_handle(p: &Arc<Port>) -> *mut c_void {
    let mut reg = registry();
    if reg.ports.len() > 64 {
        reg.ports.retain(|_, w| w.strong_count() > 0);
    }
    if let Some((&h, _)) = reg.ports.iter().find(|(_, w)| w.as_ptr() == Arc::as_ptr(p)) {
        return h as *mut c_void;
    }
    let h = reg.fresh();
    reg.ports.insert(h, Arc::downgrade(p));
    h as *mut c_void
}

fn port_of(h: *mut c_void) -> Option<Arc<Port>> {
    registry().ports.get(&(h as usize))?.upgrade()
}

fn issue_handle(q: Arc<Queue>) -> *mut c_void {
    let mut reg = registry();
    let h = reg.fresh();
    reg.queues.insert(h, q);
    h as *mut c_void
}

/// The queue behind a handle, the process queue for a null handle, `None` for an unknown or
/// closed value.
pub(crate) fn queue_from(h: *mut c_void) -> Option<Arc<Queue>> {
    if h.is_null() {
        return process_queue();
    }
    registry().queues.get(&(h as usize)).cloned()
}

enum ProcessSlot {
    Unset,
    Disabled,
    Set(Arc<Queue>),
}

static PROCESS_SLOT: Mutex<ProcessSlot> = Mutex::new(ProcessSlot::Unset);

fn process_queue() -> Option<Arc<Queue>> {
    let mut slot = grab(&PROCESS_SLOT);
    if matches!(*slot, ProcessSlot::Unset) {
        *slot = ProcessSlot::Set(Queue::new(MODE_THREAD_POOL, MODE_THREAD_POOL));
    }
    match &*slot {
        ProcessSlot::Set(q) => Some(q.clone()),
        _ => None,
    }
}

// The table's functions, with the public GDK signatures. Each `unsafe` function relies on the
// title passing what the GDK prototype says: writable out pointers, callbacks of the named type.

pub(crate) unsafe extern "system" fn XTaskQueueCreate(work: u32, completion: u32, out: *mut *mut c_void) -> i32 {
    if out.is_null() || work > MODE_IMMEDIATE || completion > MODE_IMMEDIATE {
        return E_INVALIDARG;
    }
    // SAFETY: `out` is the title's handle location, not null.
    unsafe { *out = issue_handle(Queue::new(work, completion)) };
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueGetPort(h: *mut c_void, port: u32, out: *mut *mut c_void) -> i32 {
    if out.is_null() || port > PORT_COMPLETION {
        return E_INVALIDARG;
    }
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    // SAFETY: as above.
    unsafe { *out = port_handle(&q.ports[port as usize]) };
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueCreateComposite(work_port: *mut c_void, completion_port: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let (Some(w), Some(c)) = (port_of(work_port), port_of(completion_port)) else { return E_INVALIDARG };
    // SAFETY: as above.
    unsafe { *out = issue_handle(Queue::over(w, c)) };
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueDuplicateHandle(h: *mut c_void, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    // SAFETY: as above.
    unsafe { *out = issue_handle(q) };
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueCloseHandle(h: *mut c_void) {
    if !h.is_null() {
        registry().queues.remove(&(h as usize));
    }
}

pub(crate) unsafe extern "system" fn XTaskQueueSubmitCallback(h: *mut c_void, port: u32, ctx: *mut c_void, cb: usize) -> i32 {
    if cb == 0 || port > PORT_COMPLETION {
        return E_INVALIDARG;
    }
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    if enqueue(&q, port as usize, Callback { func: cb, ctx: ctx as usize }) { S_OK } else { E_ABORT }
}

pub(crate) unsafe extern "system" fn XTaskQueueSubmitDelayedCallback(h: *mut c_void, port: u32, delay: u32, ctx: *mut c_void, cb: usize) -> i32 {
    if cb == 0 || port > PORT_COMPLETION {
        return E_INVALIDARG;
    }
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    // A delayed callback accepted after termination would run after the termination callback.
    if enqueue_after(&q, port as usize, delay, Callback { func: cb, ctx: ctx as usize }) { S_OK } else { E_ABORT }
}

pub(crate) unsafe extern "system" fn XTaskQueueDispatch(h: *mut c_void, port: u32, timeout: u32) -> bool {
    if port > PORT_COMPLETION {
        return false;
    }
    queue_from(h).is_some_and(|q| pump(&q, port as usize, timeout))
}

/// Takes the entries `q` submitted out of `lane`, leaving those of sibling queues in place.
fn take_own(lane: &mut Lane, q: &Arc<Queue>) -> Vec<Entry> {
    let (own, others): (Vec<Entry>, Vec<Entry>) = std::mem::take(&mut lane.entries).into_iter().partition(|e| Arc::ptr_eq(&e.owner, q));
    lane.entries = others.into();
    own
}

pub(crate) unsafe extern "system" fn XTaskQueueTerminate(h: *mut c_void, wait: bool, ctx: *mut c_void, cb: usize) -> i32 {
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    grab(&q.books).terminated = true;
    // Cancel what `q` itself queued, then wake manual dispatchers blocked on its ports.
    for port in &q.ports {
        let own = take_own(&mut grab(&port.lane), &q);
        port.arrived.notify_all();
        for e in own {
            // SAFETY: the entry holds a title callback.
            unsafe { run_entry(e) };
        }
    }
    // Delayed callbacks of `q` are canceled too, ahead of the termination callback. They only
    // count as open once due, so they are admitted here to keep the books balanced.
    let delayed: Vec<Delayed> = {
        let mut due = grab(&timer().due);
        let keys: Vec<(Instant, u64)> = due.iter().filter(|(_, d)| Arc::ptr_eq(&d.owner, &q)).map(|(k, _)| *k).collect();
        let taken = keys.into_iter().filter_map(|k| due.remove(&k)).collect();
        timer().changed.notify_all();
        taken
    };
    for d in delayed {
        grab(&d.owner.books).open[d.which] += 1;
        // SAFETY: as above.
        unsafe { run_entry(Entry { call: d.call, owner: d.owner, which: d.which }) };
    }
    let ctx = ctx as usize;
    let finish = move || {
        let mut b = grab(&q.books);
        while b.open != [0, 0] {
            b = park(&q.quiet, b);
        }
        drop(b);
        if cb != 0 {
            // SAFETY: the title passed `cb` as an `XTaskQueueTerminatedCallback`.
            unsafe {
                let f = std::mem::transmute::<usize, TerminatedFn>(cb);
                f(ctx as *mut c_void);
            }
        }
    };
    if wait {
        finish();
    } else {
        run_on_pool(Box::new(finish));
    }
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueGetCurrentProcessTaskQueue(out: *mut *mut c_void) -> bool {
    if out.is_null() {
        return false;
    }
    let found = process_queue();
    let ok = found.is_some();
    // SAFETY: `out` is the title's handle location, not null.
    unsafe { *out = found.map_or(std::ptr::null_mut(), issue_handle) };
    ok
}

pub(crate) unsafe extern "system" fn XTaskQueueSetCurrentProcessTaskQueue(h: *mut c_void) {
    if h.is_null() {
        *grab(&PROCESS_SLOT) = ProcessSlot::Disabled;
    } else if let Some(q) = queue_from(h) {
        *grab(&PROCESS_SLOT) = ProcessSlot::Set(q);
    }
}

/// `XTaskQueueRegistrationToken`.
#[repr(C)]
pub(crate) struct RegistrationToken {
    pub(crate) token: u64,
}

/// What a registered wait hands to its callback: where to submit, and the wait to unregister.
struct WaitRecord {
    owner: Arc<Queue>,
    which: usize,
    call: Callback,
    wait_handle: usize,
}

unsafe extern "system" fn wait_signalled(ctx: *mut c_void, _timed_out: bool) {
    // SAFETY: `ctx` is the boxed `WaitRecord`, freed only after `UnregisterWaitEx` returned.
    let rec = unsafe { &*(ctx as *const WaitRecord) };
    if !rec.owner.is_terminated() {
        enqueue(&rec.owner, rec.which, rec.call);
    }
}

pub(crate) unsafe extern "system" fn XTaskQueueRegisterWaiter(h: *mut c_void, port: u32, handle: *mut c_void, ctx: *mut c_void, cb: usize, token: *mut RegistrationToken) -> i32 {
    use windows_sys::Win32::System::Threading::{INFINITE, RegisterWaitForSingleObject, WT_EXECUTEDEFAULT};
    if port > PORT_COMPLETION || token.is_null() || cb == 0 {
        return E_INVALIDARG;
    }
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    if q.is_terminated() {
        return E_ABORT;
    }
    let rec = Box::into_raw(Box::new(WaitRecord { owner: q, which: port as usize, call: Callback { func: cb, ctx: ctx as usize }, wait_handle: 0 }));
    let mut wait = std::ptr::null_mut();
    // SAFETY: `rec` lives until `XTaskQueueUnregisterWaiter`, `handle` is the title's wait handle.
    let registered = unsafe { RegisterWaitForSingleObject(&mut wait, handle, Some(wait_signalled), rec as *const c_void, INFINITE, WT_EXECUTEDEFAULT) };
    if registered == 0 {
        // SAFETY: nothing was registered, so `rec` is still ours alone.
        drop(unsafe { Box::from_raw(rec) });
        return E_FAIL;
    }
    // SAFETY: `rec` is the box made above, `token` is the title's token location, not null.
    unsafe {
        (*rec).wait_handle = wait as usize;
        (*token).token = rec as u64;
    }
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueUnregisterWaiter(_h: *mut c_void, token: RegistrationToken) {
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Threading::UnregisterWaitEx;
    let rec = token.token as *mut WaitRecord;
    if rec.is_null() {
        return;
    }
    // SAFETY: the token came from `XTaskQueueRegisterWaiter`. Unregistering with
    // INVALID_HANDLE_VALUE waits for running callbacks, so the box is free to drop afterwards.
    unsafe {
        UnregisterWaitEx((*rec).wait_handle as *mut c_void, INVALID_HANDLE_VALUE);
        drop(Box::from_raw(rec));
    }
}

pub(crate) unsafe extern "system" fn XTaskQueueRegisterMonitor(h: *mut c_void, ctx: *mut c_void, cb: usize, token: *mut RegistrationToken) -> i32 {
    let Some(q) = queue_from(h) else { return E_INVALIDARG };
    if token.is_null() || cb == 0 {
        return E_INVALIDARG;
    }
    let id = MONITOR_SEQ.fetch_add(1, Ordering::Relaxed);
    grab(&q.monitors).push(Monitor { token: id, handle: h as usize, ctx: ctx as usize, func: cb });
    // SAFETY: the title's token location, not null.
    unsafe { (*token).token = id };
    S_OK
}

pub(crate) unsafe extern "system" fn XTaskQueueUnregisterMonitor(h: *mut c_void, token: RegistrationToken) {
    if let Some(q) = queue_from(h) {
        grab(&q.monitors).retain(|m| m.token != token.token);
    }
}

/// The live queue handles, and the callbacks the queues still hold: submitted and not finished
/// (queued, running or due), plus the delayed ones waiting for their time. For the heartbeat of
/// `calllog`.
pub(crate) fn stats() -> (usize, usize) {
    let (handles, mut distinct) = {
        let reg = registry();
        (reg.queues.len(), reg.queues.values().cloned().collect::<Vec<_>>())
    };
    distinct.sort_by_key(Arc::as_ptr);
    distinct.dedup_by(|a, b| Arc::ptr_eq(a, b));
    let open: usize = distinct.iter().map(|q| grab(&q.books).open.iter().sum::<usize>()).sum();
    (handles, open + grab(&timer().due).len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, AtomicUsize};

    fn addr(f: TaskFn) -> usize {
        f as usize
    }

    /// Context of the recording callbacks below: a log of what ran, in order.
    struct Log {
        ran: Mutex<Vec<(u32, bool)>>,
    }

    impl Log {
        fn new() -> Log {
            Log { ran: Mutex::new(Vec::new()) }
        }
        fn seen(&self) -> Vec<(u32, bool)> {
            grab(&self.ran).clone()
        }
    }

    /// Context of the tagged callbacks: a log plus the tag to record.
    struct Tagged {
        log: *const Log,
        tag: u32,
    }

    unsafe extern "system" fn record(ctx: *mut c_void, canceled: bool) {
        // SAFETY: `ctx` is a `Tagged` that outlives the test.
        let t = unsafe { &*(ctx as *const Tagged) };
        // SAFETY: the log outlives the test too.
        grab(&unsafe { &*t.log }.ran).push((t.tag, canceled));
    }

    unsafe extern "system" fn nothing(_ctx: *mut c_void, _canceled: bool) {}

    unsafe extern "system" fn bump(ctx: *mut c_void, _canceled: bool) {
        // SAFETY: `ctx` is an `AtomicU32` of the test.
        unsafe { &*(ctx as *const AtomicU32) }.fetch_add(1, Ordering::SeqCst);
    }

    fn make(work: u32, completion: u32) -> *mut c_void {
        let mut h = std::ptr::null_mut();
        // SAFETY: `h` is a writable location.
        assert_eq!(unsafe { XTaskQueueCreate(work, completion, &mut h) }, S_OK);
        assert!(!h.is_null());
        h
    }

    fn submit(h: *mut c_void, port: u32, t: &Tagged) -> i32 {
        // SAFETY: `t` and `record` fit the callback contract.
        unsafe { XTaskQueueSubmitCallback(h, port, t as *const Tagged as *mut c_void, addr(record)) }
    }

    #[test]
    fn create_rejects_bad_arguments() {
        let mut h = std::ptr::null_mut();
        // SAFETY: only the argument checks run.
        unsafe {
            assert_eq!(XTaskQueueCreate(4, 0, &mut h), E_INVALIDARG);
            assert_eq!(XTaskQueueCreate(0, 4, &mut h), E_INVALIDARG);
            assert_eq!(XTaskQueueCreate(0, 0, std::ptr::null_mut()), E_INVALIDARG);
        }
        assert!(h.is_null());
    }

    #[test]
    fn unknown_and_closed_handles_resolve_to_nothing() {
        let h = make(MODE_MANUAL, MODE_MANUAL);
        assert!(queue_from(h).is_some());
        // SAFETY: closing is idempotent and tolerates null.
        unsafe {
            XTaskQueueCloseHandle(h);
            XTaskQueueCloseHandle(h);
            XTaskQueueCloseHandle(std::ptr::null_mut());
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(nothing)), E_INVALIDARG);
            assert!(!XTaskQueueDispatch(h, PORT_WORK, 0));
        }
        assert!(queue_from(h).is_none());
        assert!(queue_from(0x77 as *mut c_void).is_none());
    }

    #[test]
    fn submit_rejects_a_missing_callback_and_a_bad_port() {
        let h = make(MODE_MANUAL, MODE_MANUAL);
        // SAFETY: only the argument checks run.
        unsafe {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), 0), E_INVALIDARG);
            assert_eq!(XTaskQueueSubmitCallback(h, 2, std::ptr::null_mut(), addr(nothing)), E_INVALIDARG);
            assert_eq!(XTaskQueueSubmitDelayedCallback(h, 2, 5, std::ptr::null_mut(), addr(nothing)), E_INVALIDARG);
            assert!(!XTaskQueueDispatch(h, 2, 0));
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn a_manual_port_runs_in_submission_order_only_when_pumped() {
        let log = Log::new();
        let tags = [1, 2, 3].map(|tag| Tagged { log: &log, tag });
        let h = make(MODE_MANUAL, MODE_MANUAL);
        for t in &tags {
            assert_eq!(submit(h, PORT_WORK, t), S_OK);
        }
        assert!(log.seen().is_empty());
        // SAFETY: `h` is a live handle.
        unsafe {
            assert!(!XTaskQueueDispatch(h, PORT_COMPLETION, 0), "the other port is empty");
            for _ in 0..3 {
                assert!(XTaskQueueDispatch(h, PORT_WORK, 0));
            }
            assert!(!XTaskQueueDispatch(h, PORT_WORK, 0));
            XTaskQueueCloseHandle(h);
        }
        assert_eq!(log.seen(), [(1, false), (2, false), (3, false)]);
    }

    #[test]
    fn a_dispatch_with_a_timeout_waits_for_late_work() {
        let h = make(MODE_MANUAL, MODE_MANUAL);
        let q = queue_from(h).unwrap();
        let feeder = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(40));
            enqueue(&q, 0, Callback { func: addr(nothing), ctx: 0 })
        });
        let started = Instant::now();
        // SAFETY: `h` is a live handle.
        assert!(unsafe { XTaskQueueDispatch(h, PORT_WORK, 5000) });
        assert!(started.elapsed() < Duration::from_secs(4));
        assert!(feeder.join().unwrap());
        // SAFETY: as above.
        unsafe {
            assert!(!XTaskQueueDispatch(h, PORT_WORK, 30), "an empty port times out");
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn dispatch_does_nothing_on_pool_and_immediate_ports() {
        for mode in [MODE_THREAD_POOL, MODE_SERIALIZED_THREAD_POOL, MODE_IMMEDIATE] {
            let h = make(mode, mode);
            // SAFETY: `h` is a live handle.
            unsafe {
                assert!(!XTaskQueueDispatch(h, PORT_WORK, 0));
                XTaskQueueCloseHandle(h);
            }
        }
    }

    #[test]
    fn an_immediate_port_runs_on_the_submitting_thread() {
        static RAN_ON: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);
        unsafe extern "system" fn whoami(_ctx: *mut c_void, _canceled: bool) {
            *grab(&RAN_ON) = Some(std::thread::current().id());
        }
        let h = make(MODE_IMMEDIATE, MODE_IMMEDIATE);
        // SAFETY: `h` is a live handle.
        unsafe {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_COMPLETION, std::ptr::null_mut(), addr(whoami)), S_OK);
            XTaskQueueCloseHandle(h);
        }
        assert_eq!(*grab(&RAN_ON), Some(std::thread::current().id()));
    }

    #[test]
    fn pool_modes_run_every_submitted_callback() {
        for mode in [MODE_THREAD_POOL, MODE_SERIALIZED_THREAD_POOL] {
            let hits = AtomicU32::new(0);
            let h = make(mode, mode);
            // SAFETY: `h` is live and `hits` outlives the wait in Terminate.
            unsafe {
                for _ in 0..40 {
                    assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, &hits as *const AtomicU32 as *mut c_void, addr(bump)), S_OK);
                }
                assert_eq!(XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0), S_OK);
                XTaskQueueCloseHandle(h);
            }
            assert_eq!(hits.load(Ordering::SeqCst), 40, "mode {mode}");
        }
    }

    #[test]
    fn a_serialized_port_never_overlaps_its_callbacks() {
        static INSIDE: AtomicUsize = AtomicUsize::new(0);
        static OVERLAPS: AtomicUsize = AtomicUsize::new(0);
        static FINISHED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "system" fn one_at_a_time(_ctx: *mut c_void, _canceled: bool) {
            if INSIDE.fetch_add(1, Ordering::SeqCst) != 0 {
                OVERLAPS.fetch_add(1, Ordering::SeqCst);
            }
            std::thread::sleep(Duration::from_millis(2));
            INSIDE.fetch_sub(1, Ordering::SeqCst);
            FINISHED.fetch_add(1, Ordering::SeqCst);
        }
        let h = make(MODE_SERIALIZED_THREAD_POOL, MODE_SERIALIZED_THREAD_POOL);
        // SAFETY: `h` is a live handle.
        unsafe {
            for _ in 0..12 {
                assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(one_at_a_time)), S_OK);
            }
            // Terminate would cancel queued work on this thread, so let the port drain first.
            let deadline = Instant::now() + Duration::from_secs(10);
            while FINISHED.load(Ordering::SeqCst) < 12 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert_eq!(XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0), S_OK);
            XTaskQueueCloseHandle(h);
        }
        assert_eq!(FINISHED.load(Ordering::SeqCst), 12);
        assert_eq!(OVERLAPS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn delayed_callbacks_wait_for_their_time_and_keep_their_order() {
        let log = Log::new();
        let (late, early) = (Tagged { log: &log, tag: 2 }, Tagged { log: &log, tag: 1 });
        let h = make(MODE_MANUAL, MODE_MANUAL);
        // SAFETY: `h` is live, the contexts outlive the test.
        unsafe {
            assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, 120, &late as *const Tagged as *mut c_void, addr(record)), S_OK);
            assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, 30, &early as *const Tagged as *mut c_void, addr(record)), S_OK);
            assert!(!XTaskQueueDispatch(h, PORT_WORK, 0), "nothing is due yet");
            assert!(XTaskQueueDispatch(h, PORT_WORK, 3000));
            assert!(XTaskQueueDispatch(h, PORT_WORK, 3000));
            XTaskQueueCloseHandle(h);
        }
        assert_eq!(log.seen(), [(1, false), (2, false)]);
    }

    #[test]
    fn terminate_cancels_pending_work_before_its_own_callback() {
        static ORDER: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
        unsafe extern "system" fn work(_ctx: *mut c_void, canceled: bool) {
            grab(&ORDER).push(if canceled { "work canceled" } else { "work ran" });
        }
        unsafe extern "system" fn finished(_ctx: *mut c_void) {
            grab(&ORDER).push("terminated");
        }
        let h = make(MODE_MANUAL, MODE_MANUAL);
        // SAFETY: `h` is a live handle.
        unsafe {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(work)), S_OK);
            assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_COMPLETION, 60_000, std::ptr::null_mut(), addr(work)), S_OK);
            assert_eq!(XTaskQueueTerminate(h, true, std::ptr::null_mut(), finished as *const () as usize), S_OK);
            XTaskQueueCloseHandle(h);
        }
        assert_eq!(*grab(&ORDER), ["work canceled", "work canceled", "terminated"]);
    }

    #[test]
    fn terminate_without_waiting_still_calls_back_later() {
        static DONE: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn finished(ctx: *mut c_void) {
            DONE.store(ctx as usize as u32, Ordering::SeqCst);
        }
        let h = make(MODE_THREAD_POOL, MODE_THREAD_POOL);
        // SAFETY: `h` is a live handle.
        unsafe {
            assert_eq!(XTaskQueueTerminate(h, false, 9 as *mut c_void, finished as *const () as usize), S_OK);
            XTaskQueueCloseHandle(h);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while DONE.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(DONE.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn terminate_waits_for_a_running_callback() {
        static STARTED: AtomicU32 = AtomicU32::new(0);
        static FINISHED: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn slow(_ctx: *mut c_void, _canceled: bool) {
            STARTED.store(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(80));
            FINISHED.store(1, Ordering::SeqCst);
        }
        let h = make(MODE_THREAD_POOL, MODE_THREAD_POOL);
        // SAFETY: `h` is a live handle.
        unsafe {
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(slow)), S_OK);
            while STARTED.load(Ordering::SeqCst) == 0 {
                std::thread::yield_now();
            }
            assert_eq!(XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0), S_OK);
            assert_eq!(FINISHED.load(Ordering::SeqCst), 1, "terminate returned while work still ran");
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn a_terminated_queue_refuses_new_work() {
        let h = make(MODE_THREAD_POOL, MODE_THREAD_POOL);
        let hits = AtomicU32::new(0);
        let ctx = &hits as *const AtomicU32 as *mut c_void;
        // SAFETY: `h` is a live handle.
        unsafe {
            assert_eq!(XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0), S_OK);
            assert_eq!(XTaskQueueSubmitCallback(h, PORT_WORK, ctx, addr(bump)), E_ABORT);
            assert_eq!(XTaskQueueSubmitDelayedCallback(h, PORT_WORK, 10, ctx, addr(bump)), E_ABORT);
            let mut token = RegistrationToken { token: 0 };
            assert_eq!(XTaskQueueRegisterWaiter(h, PORT_WORK, std::ptr::null_mut(), ctx, addr(bump), &mut token), E_ABORT);
            XTaskQueueCloseHandle(h);
        }
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(hits.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_duplicate_handle_is_the_same_queue_and_closes_on_its_own() {
        let h = make(MODE_MANUAL, MODE_MANUAL);
        let mut dup = std::ptr::null_mut();
        // SAFETY: handles are live where used.
        unsafe {
            assert_eq!(XTaskQueueDuplicateHandle(h, &mut dup), S_OK);
            assert_ne!(dup, h);
            assert!(Arc::ptr_eq(&queue_from(h).unwrap(), &queue_from(dup).unwrap()));
            XTaskQueueCloseHandle(h);
            assert!(queue_from(dup).is_some(), "the duplicate outlives the original");
            assert_eq!(XTaskQueueDuplicateHandle(h, &mut dup), E_INVALIDARG);
            XTaskQueueCloseHandle(dup);
        }
    }

    #[test]
    fn a_composite_shares_ports_and_terminating_it_spares_the_base() {
        let log = Log::new();
        let from_base = Tagged { log: &log, tag: 1 };
        let from_composite = Tagged { log: &log, tag: 2 };
        let base = make(MODE_MANUAL, MODE_MANUAL);
        let (mut work, mut completion, mut comp) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: handles are live where used.
        unsafe {
            assert_eq!(XTaskQueueGetPort(base, PORT_WORK, &mut work), S_OK);
            assert_eq!(XTaskQueueGetPort(base, PORT_COMPLETION, &mut completion), S_OK);
            let mut again = std::ptr::null_mut();
            assert_eq!(XTaskQueueGetPort(base, PORT_WORK, &mut again), S_OK);
            assert_eq!(again, work, "one handle per port");
            assert_eq!(XTaskQueueGetPort(base, 2, &mut again), E_INVALIDARG);
            assert_eq!(XTaskQueueCreateComposite(work, completion, &mut comp), S_OK);
        }
        assert_eq!(submit(base, PORT_WORK, &from_base), S_OK);
        assert_eq!(submit(comp, PORT_WORK, &from_composite), S_OK);
        // SAFETY: handles are live where used.
        unsafe {
            assert_eq!(XTaskQueueTerminate(comp, true, std::ptr::null_mut(), 0), S_OK);
            assert_eq!(log.seen(), [(2, true)], "only the composite's own work is canceled");
            assert!(XTaskQueueDispatch(base, PORT_WORK, 0));
            assert!(!XTaskQueueDispatch(base, PORT_WORK, 0));
            XTaskQueueCloseHandle(comp);
            XTaskQueueCloseHandle(base);
        }
        assert_eq!(log.seen(), [(2, true), (1, false)]);
    }

    #[test]
    fn a_composite_needs_known_ports() {
        let mut out = std::ptr::null_mut();
        // SAFETY: only the argument checks run.
        unsafe {
            assert_eq!(XTaskQueueCreateComposite(0x99 as *mut c_void, 0x98 as *mut c_void, &mut out), E_INVALIDARG);
            assert_eq!(XTaskQueueCreateComposite(std::ptr::null_mut(), std::ptr::null_mut(), &mut out), E_INVALIDARG);
        }
        assert!(out.is_null());
    }

    #[test]
    fn a_port_handle_stops_working_once_no_queue_holds_the_port() {
        let base = make(MODE_MANUAL, MODE_MANUAL);
        let mut port = std::ptr::null_mut();
        // SAFETY: handles are live where used.
        unsafe {
            assert_eq!(XTaskQueueGetPort(base, PORT_WORK, &mut port), S_OK);
            assert!(port_of(port).is_some());
            XTaskQueueCloseHandle(base);
        }
        assert!(port_of(port).is_none());
    }

    #[test]
    fn monitors_hear_submissions_until_unregistered() {
        unsafe extern "system" fn heard(ctx: *mut c_void, _queue: *mut c_void, port: u32) {
            // SAFETY: `ctx` is the test's mask.
            unsafe { &*(ctx as *const AtomicU32) }.fetch_or(1 << port, Ordering::SeqCst);
        }
        let mask = AtomicU32::new(0);
        let h = make(MODE_MANUAL, MODE_IMMEDIATE);
        let mut token = RegistrationToken { token: 0 };
        // SAFETY: `h` is live and `mask` outlives the registration.
        unsafe {
            assert_eq!(XTaskQueueRegisterMonitor(h, &mask as *const AtomicU32 as *mut c_void, 0, &mut token), E_INVALIDARG);
            assert_eq!(XTaskQueueRegisterMonitor(h, &mask as *const AtomicU32 as *mut c_void, heard as *const () as usize, &mut token), S_OK);
            assert_ne!(token.token, 0);
            XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(nothing));
            assert_eq!(mask.load(Ordering::SeqCst), 0b01);
            XTaskQueueSubmitCallback(h, PORT_COMPLETION, std::ptr::null_mut(), addr(nothing));
            assert_eq!(mask.load(Ordering::SeqCst), 0b11);
            XTaskQueueUnregisterMonitor(h, token);
            mask.store(0, Ordering::SeqCst);
            XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(nothing));
            assert_eq!(mask.load(Ordering::SeqCst), 0);
            XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0);
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn a_signalled_wait_handle_submits_its_callback() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};
        let hits = AtomicU32::new(0);
        let ctx = &hits as *const AtomicU32 as *mut c_void;
        let h = make(MODE_MANUAL, MODE_MANUAL);
        // SAFETY: `h` is live, the event is closed once, `hits` outlives the registration.
        unsafe {
            let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
            let mut token = RegistrationToken { token: 0 };
            assert_eq!(XTaskQueueRegisterWaiter(h, 2, event, ctx, addr(bump), &mut token), E_INVALIDARG);
            assert_eq!(XTaskQueueRegisterWaiter(h, PORT_WORK, event, ctx, 0, &mut token), E_INVALIDARG);
            assert_eq!(XTaskQueueRegisterWaiter(h, PORT_WORK, event, ctx, addr(bump), &mut token), S_OK);
            assert_ne!(SetEvent(event), 0);
            assert!(XTaskQueueDispatch(h, PORT_WORK, 3000));
            assert!(hits.load(Ordering::SeqCst) >= 1);
            XTaskQueueUnregisterWaiter(h, token);
            XTaskQueueTerminate(h, true, std::ptr::null_mut(), 0);
            CloseHandle(event);
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn the_process_queue_is_a_usable_handle_by_default() {
        let mut h = std::ptr::null_mut();
        // SAFETY: `h` is writable and the handle it receives is live.
        unsafe {
            assert!(XTaskQueueGetCurrentProcessTaskQueue(&mut h));
            assert!(!h.is_null());
            assert!(!XTaskQueueGetCurrentProcessTaskQueue(std::ptr::null_mut()));
            let hits = AtomicU32::new(0);
            assert_eq!(XTaskQueueSubmitCallback(std::ptr::null_mut(), PORT_WORK, &hits as *const AtomicU32 as *mut c_void, addr(bump)), S_OK);
            let deadline = Instant::now() + Duration::from_secs(5);
            while hits.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            assert_eq!(hits.load(Ordering::SeqCst), 1);
            XTaskQueueCloseHandle(h);
        }
    }

    #[test]
    fn stats_count_handles_and_open_callbacks() {
        let h = make(MODE_MANUAL, MODE_MANUAL);
        assert!(stats().0 >= 1);
        // SAFETY: `h` is a live handle.
        unsafe {
            XTaskQueueSubmitCallback(h, PORT_WORK, std::ptr::null_mut(), addr(nothing));
        }
        assert!(stats().1 >= 1);
        assert_eq!(grab(&queue_from(h).unwrap().books).open, [1, 0]);
        // SAFETY: as above.
        unsafe {
            assert!(XTaskQueueDispatch(h, PORT_WORK, 0));
            XTaskQueueCloseHandle(h);
        }
    }
}
