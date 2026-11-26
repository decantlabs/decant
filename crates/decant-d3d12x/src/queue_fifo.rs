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
//! as before. Other work is queued ([`defer`]) and one worker thread per queue runs it in order and
//! may wait as long as it needs to. Everything the title sends to that queue afterwards
//! (submissions, fence signals and waits, tile mappings, presents) queues behind it, so the PC queue
//! sees the title's order.
//!
//! The command lists of a queued submission are held ([`hold_lists`]). A `Reset` of one of them
//! waits until its submission ran ([`wait_lists_idle`]), because the PC list is only read when the
//! queued submission reaches the PC queue.

use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::{Duration, Instant};

use crate::log::xs_log;

/// A piece of work for one queue, run once.
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

/// The work of one queue.
#[derive(Default)]
struct State {
    /// Queued work, oldest first.
    items: VecDeque<Job>,
    /// Someone runs work of this queue now: its worker, or a caller holding a [`Gate`].
    busy: bool,
    /// The worker thread of this queue was started.
    worker: bool,
    /// The queue was released: its worker ends once the queued work is done.
    closed: bool,
}

impl State {
    fn idle(&self) -> bool {
        self.items.is_empty() && !self.busy
    }
}

#[derive(Default)]
struct Fifo {
    state: Mutex<State>,
    /// Notified whenever work is queued or finished, or the queue is closed.
    changed: Condvar,
}

impl Fifo {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
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
    let fifo = find_or_make(queue);
    let start = {
        let mut st = fifo.lock();
        st.items.push_back(job);
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
    if let Err(e) = std::thread::Builder::new().name("d3d12xs-queue-order".into()).spawn(move || work(&theirs)) {
        xs_log!("queue order: no worker thread ({e}), the queued work runs on the title's thread");
        fifo.lock().worker = false;
        run_queued_here(&fifo);
    }
}

/// Takes the next piece of work once nobody else runs work of this queue. `None` once the queue is
/// closed and empty, or (`stay` false) as soon as nothing is queued.
fn next(fifo: &Fifo, stay: bool) -> Option<Job> {
    let mut st = fifo.lock();
    loop {
        if !st.busy
            && let Some(job) = st.items.pop_front()
        {
            st.busy = true;
            return Some(job);
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

fn run_one(fifo: &Fifo, job: Job) {
    if std::panic::catch_unwind(AssertUnwindSafe(job)).is_err() {
        xs_log!("queue order: a queued piece of work panicked, the queue goes on");
    }
    fifo.lock().busy = false;
    fifo.changed.notify_all();
}

/// The worker of one queue: runs its work in order until the queue is closed and empty.
fn work(fifo: &Fifo) {
    while let Some(job) = next(fifo, true) {
        run_one(fifo, job);
    }
}

/// Runs what is queued on the calling thread, for a queue without a worker.
fn run_queued_here(fifo: &Fifo) {
    while let Some(job) = next(fifo, false) {
        run_one(fifo, job);
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

/// Whether any queue other than `queue` has work queued or running. Such work may still make a
/// write that a memory wait of `queue` waits for.
pub(crate) fn busy_elsewhere(queue: usize) -> bool {
    fifos().read().unwrap_or_else(|e| e.into_inner()).iter().any(|(&other, fifo)| other != queue && !fifo.lock().idle())
}

/// The command lists of queued submissions, by key, and how many queued submissions hold each.
struct Held {
    counts: Mutex<HashMap<usize, u32>>,
    changed: Condvar,
}

fn held() -> &'static Held {
    static HELD: OnceLock<Held> = OnceLock::new();
    HELD.get_or_init(|| Held { counts: Mutex::new(HashMap::new()), changed: Condvar::new() })
}

/// How many holds there are in all, so a `Reset` without any queued work takes no lock.
static HELD_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// A queued submission holds the lists `keys` until [`release_lists`].
pub(crate) fn hold_lists(keys: &[usize]) {
    if keys.is_empty() {
        return;
    }
    let mut counts = held().counts.lock().unwrap_or_else(|e| e.into_inner());
    for &key in keys {
        *counts.entry(key).or_default() += 1;
    }
    HELD_TOTAL.fetch_add(keys.len(), Ordering::AcqRel);
}

/// The submission that held `keys` reached the PC queue.
pub(crate) fn release_lists(keys: &[usize]) {
    if keys.is_empty() {
        return;
    }
    let h = held();
    let mut counts = h.counts.lock().unwrap_or_else(|e| e.into_inner());
    for &key in keys {
        if let Some(n) = counts.get_mut(&key) {
            *n -= 1;
            if *n == 0 {
                counts.remove(&key);
            }
        }
    }
    HELD_TOTAL.fetch_sub(keys.len(), Ordering::AcqRel);
    drop(counts);
    h.changed.notify_all();
}

/// Waits until no queued submission holds list `key`, for at most `timeout`. True when none does.
pub(crate) fn wait_lists_idle(key: usize, timeout: Duration) -> bool {
    if HELD_TOTAL.load(Ordering::Acquire) == 0 {
        return true;
    }
    let h = held();
    let deadline = Instant::now() + timeout;
    let mut counts = h.counts.lock().unwrap_or_else(|e| e.into_inner());
    while counts.contains_key(&key) {
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        counts = h.changed.wait_timeout(counts, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_writes::{apply_writes, satisfied, wait_until};
    use crate::list_sync::{PendingWait, WaitLimits};
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
    fn a_held_list_keeps_its_reset_waiting_until_released() {
        let key = 0x7100_0006;
        assert!(wait_lists_idle(key, Duration::ZERO), "a list nobody holds resets at once");
        hold_lists(&[key, key]);
        assert!(!wait_lists_idle(key, Duration::from_millis(10)));
        release_lists(&[key]);
        assert!(!wait_lists_idle(key, Duration::from_millis(10)), "a list submitted twice is held twice");
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            release_lists(&[key]);
        });
        assert!(wait_lists_idle(key, Duration::from_secs(5)), "the reset goes on once the submission ran");
        releaser.join().unwrap();
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
        assert!(!find(other).is_some_and(|f| !f.lock().idle()), "the other queue is idle");
    }
}
