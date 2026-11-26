//! The order of the work a title sends to each queue, kept without holding up the title's thread.
//!
//! A console list may start with a memory wait (`Wait64BitValueX`, see `list_sync`). The GPU of the
//! console holds such a list back until a value in title memory passes a comparison, while the
//! thread that submitted it goes on at once. A title can submit the list that writes the value
//! later, from that same thread, on another queue. The layer answers memory waits on the CPU
//! (`cpu_writes`), so a submission whose wait is not met yet must not wait on the title's thread:
//! the title would never get to submit the write it waits for, and each such wait used to cost
//! the whole timeout.
//!
//! Each queue gets a first in first out list of work here. While the list is empty and the work is
//! ready (its waits are met), the caller runs the work at once on its own thread ([`inline_gate`]),
//! as before. Other work is queued ([`defer`], [`defer_holding`]) and one worker thread per queue
//! runs it in order and may wait as long as it needs to. Everything the title sends to that queue
//! afterwards (submissions, fence signals and waits, tile mappings, presents) queues behind it, so
//! the PC queue sees the title's order.
//!
//! The command lists and command allocators of a queued submission are held ([`Hold`]). A `Reset`
//! of any held list or allocator waits until its queued submission has finished reaching the PC
//! queue ([`wait_list_idle`], [`wait_allocator_idle`]), accelerating any stalled memory wait on
//! that queue after a grace period so `Reset` never races with `ExecuteCommandLists`.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::log::xs_log;

/// A piece of work for one queue, run once.
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

/// An object held by a queued submission until it finishes reaching the PC queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HoldKey {
    List(usize),
    Allocator(usize),
}

/// The work of one queue.
#[derive(Default)]
struct State {
    /// Queued work with its monotonic per-queue sequence number, oldest first.
    items: VecDeque<(u64, Job)>,
    /// Last sequence number assigned to a queued item on this queue.
    next_seq: u64,
    /// Someone runs work of this queue now: its worker, or a caller holding a [`Gate`].
    busy: bool,
    /// The worker of this queue is parked inside a memory wait (`cpu_writes::wait_all_until`).
    waiting: bool,
    /// The worker thread of this queue was started.
    worker: bool,
    /// The queue was released: its worker ends once the queued work is done.
    closed: bool,
}

impl State {
    fn idle(&self) -> bool {
        self.items.is_empty() && !self.busy
    }

    /// Whether this queue has work that can still make progress on its own. A queue parked in a
    /// memory wait cannot advance until another queue or `cpu_writes` job writes a value, so two
    /// mutually waiting queues do not keep each other stuck at the hard timeout.
    fn making_progress(&self) -> bool {
        !self.idle() && !self.waiting
    }
}

#[derive(Default)]
struct Fifo {
    state: Mutex<State>,
    /// Notified whenever work is queued or finished, or the queue is closed.
    changed: Condvar,
    /// Highest submission sequence number that a waiting `Reset` asked to accelerate through any
    /// blocking memory waits. Stored atomically so `wait_key_idle` never nests `Fifo::state` under
    /// `Held::holders`.
    urgent_until: AtomicU64,
}

impl Fifo {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Marks every submission up to `seq` as urgent. Returns true if the watermark advanced.
    fn request_urgent(&self, seq: u64) -> bool {
        self.urgent_until.fetch_max(seq, Ordering::AcqRel) < seq
    }

    fn is_urgent(&self, seq: u64) -> bool {
        self.urgent_until.load(Ordering::Acquire) >= seq
    }

    /// Waits until nothing is queued or running, for at most `timeout`. True when idle.
    fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = self.lock();
        while !st.idle() {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            st = self.changed.wait_timeout(st, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
        }
        true
    }
}

thread_local! {
    /// The queue and submission sequence number currently running on this worker thread.
    static CURRENT: RefCell<Option<(Arc<Fifo>, u64)>> = const { RefCell::new(None) };
}

struct CurrentGuard(Option<(Arc<Fifo>, u64)>);

impl CurrentGuard {
    fn enter(fifo: &Arc<Fifo>, seq: u64) -> Self {
        let prev = CURRENT.with(|c| c.borrow_mut().replace((Arc::clone(fifo), seq)));
        Self(prev)
    }
}

impl Drop for CurrentGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.0.take());
    }
}

/// Marks the running queue worker as parked inside a memory wait until dropped, and reports
/// whether a `Reset` asked this submission to stop waiting and execute.
pub(crate) struct WaitScope(Option<(Arc<Fifo>, u64)>);

pub(crate) fn enter_wait() -> WaitScope {
    let current = CURRENT.with(|c| c.borrow().clone());
    if let Some((ref fifo, _)) = current {
        fifo.lock().waiting = true;
    }
    WaitScope(current)
}

impl WaitScope {
    pub(crate) fn interrupted(&self) -> bool {
        self.0.as_ref().is_some_and(|(fifo, seq)| fifo.is_urgent(*seq))
    }
}

impl Drop for WaitScope {
    fn drop(&mut self) {
        if let Some((ref fifo, _)) = self.0 {
            fifo.lock().waiting = false;
        }
    }
}

/// The queues, by the address of their PC queue.
fn fifos() -> &'static RwLock<HashMap<usize, Arc<Fifo>>> {
    static FIFOS: OnceLock<RwLock<HashMap<usize, Arc<Fifo>>>> = OnceLock::new();
    FIFOS.get_or_init(Default::default)
}

fn find(queue: usize) -> Option<Arc<Fifo>> {
    fifos().read().unwrap_or_else(|e| e.into_inner()).get(&queue).cloned()
}

fn find_or_make(queue: usize) -> Arc<Fifo> {
    if let Some(fifo) = find(queue) {
        return fifo;
    }
    fifos().write().unwrap_or_else(|e| e.into_inner()).entry(queue).or_default().clone()
}

/// The right to run work of one queue on the calling thread. Dropping it lets queued work go on.
pub(crate) struct Gate(Arc<Fifo>);

impl Drop for Gate {
    fn drop(&mut self) {
        self.0.lock().busy = false;
        self.0.changed.notify_all();
    }
}

/// How many pieces of work were queued so far, for the log.
static DEFERRED: AtomicU64 = AtomicU64::new(0);

/// The right to run work on `queue` now, on the calling thread: given when nothing of that queue is
/// queued or running and `ready()` says the work can run at once. `ready` is asked under the
/// queue's lock, so queued work cannot start while it decides.
pub(crate) fn inline_gate(queue: usize, ready: impl FnOnce() -> bool) -> Option<Gate> {
    let fifo = find_or_make(queue);
    let mut st = fifo.lock();
    if !st.idle() || !ready() {
        return None;
    }
    st.busy = true;
    drop(st);
    Some(Gate(fifo))
}

/// Queues `job` behind the work of `queue`. The worker of the queue runs it and the caller goes on.
/// Returns how many pieces of work were queued so far in the process.
pub(crate) fn defer(queue: usize, job: Job) -> u64 {
    defer_holding(queue, Vec::new(), move |_| job)
}

/// Queues a submission on `queue` that holds `keys` until the [`Hold`] passed to `make_job` drops.
/// Holds are registered under the queue lock before the job becomes visible to the worker, so any
/// `Reset` after this call returns sees them and can accelerate the queue up to this submission.
pub(crate) fn defer_holding(queue: usize, keys: Vec<HoldKey>, make_job: impl FnOnce(Hold) -> Job) -> u64 {
    let fifo = find_or_make(queue);
    let start = {
        let mut st = fifo.lock();
        st.next_seq += 1;
        let seq = st.next_seq;
        let hold = Hold::acquire(Arc::clone(&fifo), seq, keys);
        let job = make_job(hold);
        st.items.push_back((seq, job));
        !std::mem::replace(&mut st.worker, true)
    };
    fifo.changed.notify_all();
    if start {
        start_worker(fifo);
    }
    DEFERRED.fetch_add(1, Ordering::Relaxed) + 1
}

fn start_worker(fifo: Arc<Fifo>) {
    let theirs = Arc::clone(&fifo);
    if let Err(e) = std::thread::Builder::new().name("d3d12xs-queue-order".into()).spawn(move || work(theirs)) {
        xs_log!("queue order: no worker thread ({e}), the queued work runs on the title's thread");
        fifo.lock().worker = false;
        run_queued_here(&fifo);
    }
}

/// Takes the next piece of work once nobody else runs work of this queue. `None` once the queue is
/// closed and empty, or (`stay` false) as soon as nothing is queued.
fn next(fifo: &Fifo, stay: bool) -> Option<(u64, Job)> {
    let mut st = fifo.lock();
    loop {
        if !st.busy
            && let Some(item) = st.items.pop_front()
        {
            st.busy = true;
            return Some(item);
        }
        if st.items.is_empty() && (st.closed || !stay) {
            if stay {
                st.worker = false;
            }
            return None;
        }
        st = fifo.changed.wait(st).unwrap_or_else(|e| e.into_inner());
    }
}

fn run_one(fifo: &Arc<Fifo>, seq: u64, job: Job) {
    let _current = CurrentGuard::enter(fifo, seq);
    if std::panic::catch_unwind(AssertUnwindSafe(job)).is_err() {
        xs_log!("queue order: a queued piece of work panicked, the queue goes on");
    }
    drop(_current);
    let mut st = fifo.lock();
    st.busy = false;
    st.waiting = false;
    drop(st);
    fifo.changed.notify_all();
}

/// The worker of one queue: runs its work in order until the queue is closed and empty.
fn work(fifo: Arc<Fifo>) {
    while let Some((seq, job)) = next(&fifo, true) {
        run_one(&fifo, seq, job);
    }
}

/// Runs what is queued on the calling thread, for a queue without a worker.
fn run_queued_here(fifo: &Arc<Fifo>) {
    while let Some((seq, job)) = next(fifo, false) {
        run_one(fifo, seq, job);
    }
}

/// Waits until everything queued on `queue` ran, for at most `timeout`. True when the queue is idle.
/// Never called by a holder of that queue's [`Gate`] or from its worker.
pub(crate) fn drain(queue: usize, timeout: Duration) -> bool {
    find(queue).is_none_or(|fifo| fifo.wait_idle(timeout))
}

/// Waits until every queue is idle, for at most `timeout` in all.
pub(crate) fn drain_all(timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let all: Vec<Arc<Fifo>> = fifos().read().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
    all.iter().all(|fifo| fifo.wait_idle(deadline.saturating_duration_since(Instant::now())))
}

/// The queue `queue` was released. Its queued work still runs, then its worker ends. A later queue
/// at the same address gets a fresh entry.
pub(crate) fn close(queue: usize) {
    let removed = fifos().write().unwrap_or_else(|e| e.into_inner()).remove(&queue);
    if let Some(fifo) = removed {
        fifo.lock().closed = true;
        fifo.changed.notify_all();
    }
}

/// Whether any queue other than `queue` has work queued or running that is not itself parked in a
/// memory wait. Such work may still make a write that a memory wait of `queue` waits for.
pub(crate) fn busy_elsewhere(queue: usize) -> bool {
    fifos()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|(&other, fifo)| other != queue && fifo.lock().making_progress())
}

/// One queued submission that holds an object: its queue and sequence number.
type Holder = (Arc<Fifo>, u64);

/// Objects held by queued submissions, mapped to each holding submission.
struct Held {
    holders: Mutex<HashMap<HoldKey, Vec<Holder>>>,
    changed: Condvar,
}

fn held() -> &'static Held {
    static HELD: OnceLock<Held> = OnceLock::new();
    HELD.get_or_init(|| Held { holders: Mutex::new(HashMap::new()), changed: Condvar::new() })
}

/// How many holds there are in all, so a `Reset` without any queued work takes no lock.
static HELD_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Holds `keys` against `Reset` until dropped.
pub(crate) struct Hold {
    fifo: Arc<Fifo>,
    seq: u64,
    keys: Vec<HoldKey>,
}

impl Hold {
    fn acquire(fifo: Arc<Fifo>, seq: u64, keys: Vec<HoldKey>) -> Self {
        if !keys.is_empty() {
            let mut holders = held().holders.lock().unwrap_or_else(|e| e.into_inner());
            for &key in &keys {
                holders.entry(key).or_default().push((Arc::clone(&fifo), seq));
            }
            HELD_TOTAL.fetch_add(keys.len(), Ordering::AcqRel);
        }
        Self { fifo, seq, keys }
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let h = held();
        let mut holders = h.holders.lock().unwrap_or_else(|e| e.into_inner());
        for &key in &self.keys {
            if let Some(entries) = holders.get_mut(&key) {
                if let Some(pos) = entries.iter().position(|(f, s)| *s == self.seq && Arc::ptr_eq(f, &self.fifo)) {
                    entries.swap_remove(pos);
                }
                if entries.is_empty() {
                    holders.remove(&key);
                }
            }
        }
        HELD_TOTAL.fetch_sub(self.keys.len(), Ordering::AcqRel);
        drop(holders);
        h.changed.notify_all();
    }
}

/// Waits until no queued submission holds `key`. Gives normal producers `grace` to satisfy their
/// memory wait naturally; if `key` is still held after `grace`, accelerates every queue holding it
/// through its blocking memory waits so the submission reaches the PC queue before `Reset` runs.
fn wait_key_idle(key: HoldKey, grace: Duration) {
    if HELD_TOTAL.load(Ordering::Acquire) == 0 {
        return;
    }
    let h = held();
    let mut holders = h.holders.lock().unwrap_or_else(|e| e.into_inner());
    if !holders.contains_key(&key) {
        return;
    }
    let urgent_at = Instant::now() + grace;
    let mut escalated = false;
    let mut warn_at = Instant::now() + Duration::from_secs(10);
    while let Some(entries) = holders.get(&key) {
        let now = Instant::now();
        if now >= urgent_at {
            let mut nudged = false;
            for (fifo, seq) in entries {
                if fifo.request_urgent(*seq) {
                    nudged = true;
                }
            }
            if !escalated {
                escalated = true;
                static ACCELERATED: AtomicU64 = AtomicU64::new(0);
                let n = ACCELERATED.fetch_add(1, Ordering::Relaxed) + 1;
                if n <= 8 || n.is_power_of_two() {
                    xs_log!("queue order: Reset of {key:?} accelerated its queued submission after {} ms ({n} so far)", grace.as_millis());
                }
            }
            if nudged {
                crate::cpu_writes::wake_waits();
            }
            if now >= warn_at {
                warn_at = now + Duration::from_secs(10);
                xs_log!("queue order: Reset of {key:?} is still waiting for its queued submission to finish");
            }
            holders = h.changed.wait_timeout(holders, Duration::from_millis(200)).unwrap_or_else(|e| e.into_inner()).0;
        } else {
            holders = h.changed.wait_timeout(holders, urgent_at - now).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

/// Waits until no queued submission holds command list `key`. Never returns while a queued
/// submission may still execute that PC command list.
pub(crate) fn wait_list_idle(key: usize, grace: Duration) {
    wait_key_idle(HoldKey::List(key), grace);
}

/// Waits until no queued submission holds command allocator `key`. Never returns while a queued
/// submission may still execute a command list recorded with that allocator.
pub(crate) fn wait_allocator_idle(key: usize, grace: Duration) {
    wait_key_idle(HoldKey::Allocator(key), grace);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_writes::{apply_writes, satisfied, wait_all_until, wait_until};
    use crate::list_sync::{PendingWait, WaitLimits, for_each_ready_slice, forget, record_cpu_write, record_wait};
    use std::sync::atomic::AtomicBool;

    // The tables are shared by the whole process and tests run in parallel, so each test uses
    // queue and list keys of its own.

    fn greater_equal(label: &AtomicU64, value: u64) -> PendingWait {
        PendingWait { va: std::ptr::from_ref(label) as u64, width: 8, comparison: 7, value, mask: u64::MAX }
    }

    #[test]
    fn ready_work_on_an_idle_queue_runs_inline() {
        let queue = 0x7100_0001;
        let label = Box::new(AtomicU64::new(3));
        let met = greater_equal(&label, 3);
        let gate = inline_gate(queue, || satisfied(&met));
        assert!(gate.is_some(), "an idle queue whose waits are met runs the work on the caller");
        assert!(inline_gate(queue, || true).is_none(), "only one caller runs work of a queue at a time");
        drop(gate);
        assert!(inline_gate(queue, || satisfied(&greater_equal(&label, 4))).is_none(), "work with an unmet wait is never run inline");
        assert!(inline_gate(queue, || true).is_some(), "the queue is idle again");
        assert_eq!(find(queue).map(|f| f.lock().worker), Some(false), "no worker thread is started for inline work");
    }

    #[test]
    fn work_queued_while_a_caller_runs_inline_waits_for_it() {
        let queue = 0x7100_0002;
        let gate = inline_gate(queue, || true).expect("the queue is idle");
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        defer(queue, Box::new(move || flag.store(true, Ordering::SeqCst)));
        std::thread::sleep(Duration::from_millis(30));
        assert!(!ran.load(Ordering::SeqCst), "queued work does not overtake the inline caller");
        assert!(inline_gate(queue, || true).is_none(), "queued work keeps later work off the inline path");
        drop(gate);
        assert!(drain(queue, Duration::from_secs(5)));
        assert!(ran.load(Ordering::SeqCst));
        assert!(inline_gate(queue, || true).is_some(), "an empty queue runs inline again");
    }

    /// What a piece of queued work did: its name, when it finished, and whether its wait was met.
    type Done = Arc<Mutex<Vec<(&'static str, Instant, bool)>>>;

    #[test]
    fn a_wait_queued_on_one_queue_is_met_by_a_later_write_of_the_same_thread_on_another() {
        let (consumer, producer) = (0x7100_0003, 0x7100_0004);
        let label = Box::new(AtomicU64::new(0));
        let wait = greater_equal(&label, 1);
        let order: Done = Arc::default();
        let limits = WaitLimits { idle: Duration::from_secs(5), hard: Duration::from_secs(10) };

        let begun = Instant::now();
        assert!(inline_gate(consumer, || satisfied(&wait)).is_none(), "an unmet wait is never run inline");
        let log = Arc::clone(&order);
        defer(
            consumer,
            Box::new(move || {
                let met = wait_until(&wait, limits, || false);
                log.lock().unwrap().push(("wait", Instant::now(), met));
            }),
        );
        let log = Arc::clone(&order);
        defer(consumer, Box::new(move || log.lock().unwrap().push(("after", Instant::now(), true))));
        assert!(begun.elapsed() < Duration::from_millis(100), "the submitting thread went on at once");

        // The same thread now submits the producer on its own queue, which is idle.
        let written_at = {
            let _gate = inline_gate(producer, || true).expect("the producer queue is idle");
            let at = Instant::now();
            apply_writes(&[(std::ptr::from_ref(&*label) as usize, vec![1, 0])], None);
            at
        };
        assert!(drain(consumer, Duration::from_secs(10)), "the consumer queue finishes");
        let order = order.lock().unwrap();
        let names: Vec<&str> = order.iter().map(|o| o.0).collect();
        assert_eq!(names, ["wait", "after"], "the queue keeps its order");
        assert!(order[0].2, "the wait was met, it did not time out");
        let latency = order[0].1.saturating_duration_since(written_at);
        assert!(latency < Duration::from_millis(50), "the waiting worker woke {latency:?} after the write");
    }

    #[test]
    fn an_unmet_wait_without_a_producer_times_out_and_lets_go_of_what_it_held() {
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let queue = 0x7100_0005;
        let label = Box::new(AtomicU64::new(0));
        let wait = greater_equal(&label, 1);
        let released = Arc::new(AtomicUsize::new(0));
        let held = Counted(Arc::clone(&released));
        let met = Arc::new(AtomicBool::new(true));
        let answer = Arc::clone(&met);
        let limits = WaitLimits { idle: Duration::from_millis(20), hard: Duration::from_millis(400) };
        let begun = Instant::now();
        defer(
            queue,
            Box::new(move || {
                let _held = held;
                answer.store(wait_until(&wait, limits, || false), Ordering::SeqCst);
            }),
        );
        assert!(drain(queue, Duration::from_secs(5)), "the queue never deadlocks");
        assert!(!met.load(Ordering::SeqCst), "nothing wrote the value");
        assert_eq!(released.load(Ordering::SeqCst), 1, "what the work held is released once");
        assert!(begun.elapsed() < Duration::from_secs(2), "the wait ended at its limit");
        drop(label);
    }

    #[test]
    fn reset_of_a_queued_list_accelerates_its_wait_and_runs_only_after_execution() {
        let (queue, key) = (0x7100_0006, 0x7100_1006);
        let label = Box::new(AtomicU64::new(0));
        let wait = greater_equal(&label, 1);
        let executed = Arc::new(AtomicBool::new(false));
        let executed_worker = Arc::clone(&executed);
        let limits = WaitLimits { idle: Duration::from_secs(10), hard: Duration::from_secs(10) };

        defer_holding(queue, vec![HoldKey::List(key)], move |hold| {
            Box::new(move || {
                let _hold = hold;
                let scope = enter_wait();
                let _ = wait_all_until(&[wait], limits, || true, || scope.interrupted());
                drop(scope);
                executed_worker.store(true, Ordering::SeqCst);
            })
        });

        let begun = Instant::now();
        wait_list_idle(key, Duration::from_millis(25));
        assert!(executed.load(Ordering::SeqCst), "Reset never overtakes ExecuteCommandLists");
        assert!(begun.elapsed() < Duration::from_secs(1), "Reset accelerated the stuck wait instead of timing out ({:?})", begun.elapsed());
        assert!(drain(queue, Duration::from_secs(5)));
        drop(label);
    }

    #[test]
    fn reset_of_a_queued_list_lets_a_fast_producer_satisfy_the_wait_first() {
        let (queue, key) = (0x7100_000A, 0x7100_100A);
        let label = Box::new(AtomicU64::new(0));
        let va = std::ptr::from_ref(&*label) as usize;
        let wait = greater_equal(&label, 1);
        let met = Arc::new(AtomicBool::new(false));
        let met_worker = Arc::clone(&met);
        let limits = WaitLimits { idle: Duration::from_secs(5), hard: Duration::from_secs(10) };

        defer_holding(queue, vec![HoldKey::List(key)], move |hold| {
            Box::new(move || {
                let _hold = hold;
                let scope = enter_wait();
                let unmet = wait_all_until(&[wait], limits, || true, || scope.interrupted());
                met_worker.store(unmet.is_empty(), Ordering::SeqCst);
            })
        });

        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(25));
            apply_writes(&[(va, vec![1, 0])], None);
        });
        wait_list_idle(key, Duration::from_millis(400));
        writer.join().unwrap();
        assert!(met.load(Ordering::SeqCst), "a producer arriving within grace satisfies the wait without interruption");
        assert!(drain(queue, Duration::from_secs(5)));
        drop(label);
    }

    #[test]
    fn reset_of_an_allocator_waits_for_queued_lists_using_it() {
        let (queue, alloc) = (0x7100_000B, 0x7100_200B);
        let label = Box::new(AtomicU64::new(0));
        let wait = greater_equal(&label, 1);
        let executed = Arc::new(AtomicBool::new(false));
        let executed_worker = Arc::clone(&executed);
        let limits = WaitLimits { idle: Duration::from_secs(10), hard: Duration::from_secs(10) };

        defer_holding(queue, vec![HoldKey::Allocator(alloc)], move |hold| {
            Box::new(move || {
                let _hold = hold;
                let scope = enter_wait();
                let _ = wait_all_until(&[wait], limits, || true, || scope.interrupted());
                executed_worker.store(true, Ordering::SeqCst);
            })
        });

        let begun = Instant::now();
        wait_allocator_idle(alloc, Duration::from_millis(20));
        assert!(executed.load(Ordering::SeqCst), "allocator Reset waited for the queued submission");
        assert!(begun.elapsed() < Duration::from_secs(1));
        assert!(drain(queue, Duration::from_secs(5)));
        drop(label);
    }

    #[test]
    fn a_closed_queue_still_runs_its_queued_work() {
        let queue = 0x7100_0007;
        let (sender, ran) = std::sync::mpsc::channel();
        let gate = inline_gate(queue, || true).expect("the queue is idle");
        defer(queue, Box::new(move || sender.send(()).unwrap()));
        close(queue);
        assert!(find(queue).is_none(), "a closed queue leaves the table");
        drop(gate);
        assert!(ran.recv_timeout(Duration::from_secs(5)).is_ok(), "work queued before the release still runs");
        assert!(drain(queue, Duration::ZERO), "a closed queue counts as idle");
    }

    #[test]
    fn busy_elsewhere_sees_other_queues_only() {
        let (mine, other) = (0x7100_0008, 0x7100_0009);
        let gate = inline_gate(other, || true).expect("the queue is idle");
        assert!(busy_elsewhere(mine), "work running on another queue counts");
        drop(gate);
        let _mine = inline_gate(mine, || true).expect("the queue is idle");
        assert!(!find(other).is_some_and(|f| f.lock().making_progress()), "the other queue is idle");
    }

    #[test]
    fn mutually_waiting_queues_break_at_idle_limit_not_hard_limit() {
        let (q_a, q_b) = (0x7100_000C, 0x7100_000D);
        let label_a = Box::new(AtomicU64::new(0));
        let label_b = Box::new(AtomicU64::new(0));
        let wait_a = greater_equal(&label_a, 1);
        let wait_b = greater_equal(&label_b, 1);
        let limits = WaitLimits { idle: Duration::from_millis(30), hard: Duration::from_secs(2) };

        let begun = Instant::now();
        defer(
            q_a,
            Box::new(move || {
                let scope = enter_wait();
                let _ = wait_all_until(&[wait_a], limits, || busy_elsewhere(q_a), || scope.interrupted());
            }),
        );
        defer(
            q_b,
            Box::new(move || {
                let scope = enter_wait();
                let _ = wait_all_until(&[wait_b], limits, || busy_elsewhere(q_b), || scope.interrupted());
            }),
        );
        assert!(drain(q_a, Duration::from_secs(5)));
        assert!(drain(q_b, Duration::from_secs(5)));
        let elapsed = begun.elapsed();
        assert!(elapsed < Duration::from_millis(500), "mutually waiting queues should break at idle limit, took {elapsed:?}");
        drop(label_a);
        drop(label_b);
    }

    #[test]
    fn multi_list_batch_ping_pong_with_another_queue_completes_without_timeout() {
        // Direct batch [L0 writes A=1, L1 waits B>=1, L2 writes C=1] paired with Compute [C0 waits
        // A>=1 and writes B=1]. Dispatching each executed slice's CPU writes before waiting lets
        // both queues finish immediately instead of deadlocking.
        let (direct, compute) = (0x7100_000E, 0x7100_000F);
        let (l0, l1, l2, c0) = (0x7100_3001, 0x7100_3002, 0x7100_3003, 0x7100_3004);
        let label_a = Box::new(AtomicU64::new(0));
        let label_b = Box::new(AtomicU64::new(0));
        let label_c = Box::new(AtomicU64::new(0));
        let va_a = std::ptr::from_ref(&*label_a) as usize;
        let va_b = std::ptr::from_ref(&*label_b) as usize;
        let va_c = std::ptr::from_ref(&*label_c) as usize;

        record_cpu_write(l0, va_a, vec![1, 0]);
        record_wait(l1, greater_equal(&label_b, 1));
        record_cpu_write(l2, va_c, vec![1, 0]);
        record_wait(c0, greater_equal(&label_a, 1));
        record_cpu_write(c0, va_b, vec![1, 0]);

        let limits = WaitLimits { idle: Duration::from_secs(2), hard: Duration::from_secs(5) };
        let timed_out = Arc::new(AtomicUsize::new(0));
        let to_d = Arc::clone(&timed_out);
        let to_c = Arc::clone(&timed_out);
        let begun = Instant::now();

        let direct_keys = vec![l0, l1, l2];
        defer(
            direct,
            Box::new(move || {
                for_each_ready_slice(
                    &direct_keys,
                    |k| crate::list_sync::all_waits(k, satisfied),
                    |s, e| {
                        let writes = crate::list_sync::cpu_writes_of(&direct_keys[s..e]);
                        apply_writes(&writes, None);
                    },
                    |k| {
                        let scope = enter_wait();
                        let unmet = wait_all_until(&crate::list_sync::waits_of(k), limits, || busy_elsewhere(direct), || scope.interrupted());
                        to_d.fetch_add(unmet.len(), Ordering::SeqCst);
                    },
                );
            }),
        );
        let compute_keys = vec![c0];
        defer(
            compute,
            Box::new(move || {
                for_each_ready_slice(
                    &compute_keys,
                    |k| crate::list_sync::all_waits(k, satisfied),
                    |s, e| {
                        let writes = crate::list_sync::cpu_writes_of(&compute_keys[s..e]);
                        apply_writes(&writes, None);
                    },
                    |k| {
                        let scope = enter_wait();
                        let unmet = wait_all_until(&crate::list_sync::waits_of(k), limits, || busy_elsewhere(compute), || scope.interrupted());
                        to_c.fetch_add(unmet.len(), Ordering::SeqCst);
                    },
                );
            }),
        );

        assert!(drain(direct, Duration::from_secs(5)));
        assert!(drain(compute, Duration::from_secs(5)));
        assert_eq!(timed_out.load(Ordering::SeqCst), 0, "neither queue timed out");
        assert_eq!(label_a.load(Ordering::Acquire), 1);
        assert_eq!(label_b.load(Ordering::Acquire), 1);
        assert_eq!(label_c.load(Ordering::Acquire), 1);
        assert!(begun.elapsed() < Duration::from_millis(200), "ping-pong completed in {:?}", begun.elapsed());

        for k in [l0, l1, l2, c0] {
            forget(k);
        }
    }
}
