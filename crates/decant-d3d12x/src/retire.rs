//! Deferred release of PC objects. The last release of an object waits until the GPU has finished
//! every submission that could still use it.
//!
//! On the console a resource is only a description laid over memory that the title owns. The title
//! may drop the object while commands that use it are still in flight and nothing faults. On the PC
//! the layer made a real resource, and freeing it while a submitted list reads it is a page fault
//! that removes the device (DRED reports "recently freed allocation type 34"). The same holds for
//! the layer's own objects that the lists of the title read: the upload buffers handed out for
//! `AllocateUploadMemory`, and the backing heap and page map of a descriptor heap.
//!
//! Every submission of the title (and the present copy of the layer) is therefore opened with
//! [`begin`] and closed with [`submitted`], which signals a fence of this module on that queue. When
//! the last reference of an object goes, the object is parked together with one target per queue:
//! the last value signalled while the GPU is still behind it, or the next one while a submission is
//! under way. It is released when every fence has reached its target. The parked objects are looked
//! at on each release and at each present, and how many there are depends on how far the GPU falls
//! behind. When a device goes, its last release waits (bounded) for the queues of that device only,
//! then lets go of what only those queues held back and of their fences. Objects that are also
//! parked for a queue of another device stay parked for that queue.
//!
//! A queue whose fence could not be made is tracked all the same. What goes while it has work gets
//! a target that is never reached and stays parked for good, because nothing proves that the GPU
//! finished. The last release of the device proxy then does not destroy the device while such a
//! queue holds it.
//!
//! An object gets a name before it goes (`SetName`, which DRED reads as the name of a freed
//! allocation). The last [`RECENT`] freed objects are remembered with their GPU range, so a page
//! fault report can say what used to be at the faulting address.
#![allow(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{HSTRING, IUnknown, Interface};

use crate::log::xs_log;
use crate::mem_owners::Charge;

/// The longest teardown waits for one queue.
const FLUSH_WAIT_MS: u32 = 2000;
/// How many freed objects the fault report can look at.
const RECENT: usize = 512;
/// A summary line is logged every this many presents.
const SUMMARY_EVERY: u64 = 600;

/// One thing to wait for: a fence (or, for a queue without one, the queue itself) and the value
/// that fence must reach.
type Target = (usize, u64);

/// Queue-order work active in `queue_fifo` when an object was parked, paired with the queue's last
/// signalled fence value at snapshot time so any GPU work submitted before that queue-order item
/// finishes is also waited on before release.
#[derive(Clone)]
struct FifoWait {
    work: crate::queue_fifo::PendingWork,
    snapshot_signal: u64,
}

/// An object kept back and the targets it waits for.
struct Parked<T> {
    item: Option<T>,
    targets: Vec<Target>,
    fifo_waits: Vec<FifoWait>,
}

/// The parked objects in the order they were parked. It makes no GPU call, so tests run it on
/// invented data.
pub(crate) struct Parking<T> {
    entries: Vec<Parked<T>>,
    most: usize,
}

impl<T> Default for Parking<T> {
    fn default() -> Self {
        Self { entries: Vec::new(), most: 0 }
    }
}

impl<T> Parking<T> {
    /// Keeps `item` back until all `targets` are reached. When there are none nothing in flight can
    /// use it, so it is handed back at once.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn park(&mut self, item: T, targets: Vec<Target>) -> Option<T> {
        self.park_with_waits(item, targets, Vec::new())
    }

    /// Keeps `item` back until all `targets` and queue-order `fifo_waits` are finished. When both
    /// are empty nothing in flight can use it, so it is handed back at once.
    fn park_with_waits(&mut self, item: T, targets: Vec<Target>, fifo_waits: Vec<FifoWait>) -> Option<T> {
        if targets.is_empty() && fifo_waits.is_empty() {
            return Some(item);
        }
        self.entries.push(Parked { item: Some(item), targets, fifo_waits });
        self.most = self.most.max(self.entries.len());
        None
    }

    /// Splits off every entry for which `ready` says that it can go. `ready` may change what the
    /// entry waits for.
    fn extract(&mut self, mut ready: impl FnMut(&mut Vec<Target>, &mut Vec<FifoWait>) -> bool) -> Vec<T> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        self.entries.retain_mut(|entry| {
            if ready(&mut entry.targets, &mut entry.fifo_waits) {
                if let Some(item) = entry.item.take() {
                    out.push(item);
                }
                false
            } else {
                true
            }
        });
        out
    }

    /// Hands out every item whose targets are all reached. `reached(key)` is the value the fence
    /// `key` has got to.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn release_reached(&mut self, mut reached: impl FnMut(usize) -> u64) -> Vec<T> {
        self.extract(|targets, _| targets.iter().all(|&(key, value)| reached(key) >= value))
    }

    /// Hands out every item whose queue-order waits have finished (promoting any new GPU signals
    /// produced by those items into `targets`) and whose fence targets are all reached.
    fn release_reached_with(
        &mut self,
        mut reached: impl FnMut(usize) -> u64,
        mut resolve_waits: impl FnMut(&mut Vec<FifoWait>, &mut Vec<Target>) -> bool,
    ) -> Vec<T> {
        self.extract(|targets, waits| resolve_waits(waits, targets) && targets.iter().all(|&(key, value)| reached(key) >= value))
    }

    /// The queues behind `keys` are gone (their device was torn down after waiting for them). Their
    /// targets are dropped and the items that waited for nothing else are handed out.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn drop_keys(&mut self, keys: &[usize]) -> Vec<T> {
        self.drop_keys_and_queues(keys, &[])
    }

    /// Same as [`drop_keys`], also dropping finished queue-order waits for `torn_down_queues`.
    fn drop_keys_and_queues(&mut self, keys: &[usize], torn_down_queues: &[usize]) -> Vec<T> {
        self.extract(|targets, waits| {
            targets.retain(|(key, _)| !keys.contains(key));
            if !torn_down_queues.is_empty() {
                waits.retain(|fw| !(torn_down_queues.contains(&fw.work.queue()) && fw.work.is_finished()));
            }
            targets.is_empty() && waits.is_empty()
        })
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The most items that were parked at once.
    pub fn most(&self) -> usize {
        self.most
    }
}

/// A PC object to release, and what it is for its name and the fault report.
struct Retired {
    object: usize,
    what: &'static str,
    charge: Option<Charge>,
}

// SAFETY: the object pointers are COM objects of a free threaded device and are only used under the lock.
unsafe impl Send for Retired {}

/// A freed object: its GPU range (zero for a texture or a heap) and what it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Freed {
    pub va: u64,
    pub bytes: u64,
    pub label: String,
    pub at_ms: u64,
}

/// What is tracked for one queue.
#[derive(Clone, Copy)]
struct QueueTrack {
    /// The raw fence of the queue, released when its device goes. Zero if none could be made.
    fence: usize,
    /// A second fence signalled when a submission closes while another submission on this queue is
    /// still open, so callers draining already-executed GPU work never block on queued submissions
    /// that have not executed yet.
    exec_fence: usize,
    /// The `IUnknown` of the device of the queue.
    device: usize,
    /// The last value signalled on `fence`.
    last_signal: u64,
    /// The last value signalled on `exec_fence`.
    exec_signal: u64,
    /// How many submissions are under way: begun, their signal not yet sent.
    open: u32,
    /// Whether work was begun that the next signal has to cover.
    unsignalled: bool,
    /// Whether an intermediate submission failed to signal `exec_fence` since the last full signal.
    exec_failed: bool,
}

impl QueueTrack {
    /// Closes one submission. Overlapping submissions share one target, so only the last of them
    /// may send the signal: an earlier one would signal while another can still enqueue work.
    #[cfg(test)]
    fn close(&mut self, send_signal: impl FnOnce(u64) -> bool) {
        self.close_with_exec(send_signal, |_| true);
    }

    /// Closes one submission on `fence` when no other submission is open, or on `exec_fence` when
    /// other submissions remain queued on the same queue.
    fn close_with_exec(&mut self, send_main: impl FnOnce(u64) -> bool, send_exec: impl FnOnce(u64) -> bool) {
        if self.open == 0 {
            return;
        }
        self.open -= 1;
        if self.open == 0 {
            if send_main(self.last_signal + 1) {
                self.last_signal += 1;
                self.unsignalled = false;
                self.exec_failed = false;
            }
        } else if send_exec(self.exec_signal + 1) {
            self.exec_signal += 1;
            self.exec_failed = false;
        } else {
            self.exec_failed = true;
        }
    }

    /// What targets of this queue are keyed by: its fence, or the queue when it has no fence.
    fn key(&self, queue: usize) -> usize {
        if self.fence == 0 { queue } else { self.fence }
    }
}

/// The value a release has to wait for on a queue whose fence is at `reached`. While the GPU is
/// behind it is the last signal. While a submission is under way it is the next one, because its
/// lists may run before the signal is sent. An idle queue needs no wait. A removed device
/// (`u64::MAX`) needs none either.
fn wait_target(reached: u64, last_signal: u64, unsignalled: bool) -> Option<u64> {
    if reached == u64::MAX {
        None
    } else if unsignalled {
        Some(last_signal + 1)
    } else if reached < last_signal {
        Some(last_signal)
    } else {
        None
    }
}

#[derive(Default)]
struct FastUsizeHasher(u64);

impl Hasher for FastUsizeHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
        }
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<V> = HashMap<usize, V, BuildHasherDefault<FastUsizeHasher>>;

struct State {
    queues: FastMap<QueueTrack>,
    parked: Parking<Retired>,
    recent: VecDeque<Freed>,
}

fn state() -> MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State { queues: FastMap::default(), parked: Parking::default(), recent: VecDeque::new() })).lock().unwrap_or_else(|e| e.into_inner())
}

/// How many releases had to wait for work in flight, how many were done, how many presents there were.
static HELD_BACK: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);
static PRESENTS: AtomicU64 = AtomicU64::new(0);

/// Milliseconds since the first call.
fn ms_now() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// The value `fence` has reached. A removed device gives `u64::MAX`.
///
/// # Safety
/// `fence` is a live PC fence.
unsafe fn fence_value(fence: usize) -> u64 {
    let raw = fence as *mut c_void;
    // SAFETY: by the contract above.
    unsafe { ID3D12Fence::from_raw_borrowed(&raw).map_or(u64::MAX, |f| f.GetCompletedValue()) }
}

/// The value reached by the target key `key`. A queue without a fence never gets anywhere.
#[cfg(test)]
fn reached(queues: &FastMap<QueueTrack>, key: usize) -> u64 {
    if queues.get(&key).is_some_and(|q| q.fence == 0) {
        return 0;
    }
    // SAFETY: every other key is a fence from the table, and those are live.
    unsafe { fence_value(key) }
}

/// Queries each tracked queue's fence at most once for the current call.
fn snapshot_reached(queues: &FastMap<QueueTrack>) -> Vec<(usize, u64)> {
    queues
        .iter()
        .map(|(&queue, track)| {
            let key = track.key(queue);
            let val = if track.fence == 0 {
                0
            } else {
                // SAFETY: every non-zero fence in `queues` is live while `state()` is held.
                unsafe { fence_value(key) }
            };
            (key, val)
        })
        .collect()
}

fn reached_in(snapshot: &[(usize, u64)], key: usize) -> u64 {
    snapshot.iter().find(|&&(k, _)| k == key).map_or(u64::MAX, |&(_, v)| v)
}

/// Starts a submission to the PC queue `queue`, before any `ExecuteCommandLists` of it. Objects
/// released from now on wait for the signal that [`submitted`] sends. Every call that returns true
/// is followed by one call of `submitted` for the same queue.
///
/// # Safety
/// `queue` is a live PC command queue.
pub(crate) unsafe fn begin(queue: *mut c_void) -> bool {
    // SAFETY: by the contract above.
    let Some(q) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return false };
    let mut st = state();
    let key = queue as usize;
    if let Some(track) = st.queues.get_mut(&key) {
        track.open += 1;
        track.unsignalled = true;
        return true;
    }
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: a documented call on a live queue.
    let device = unsafe { q.GetDevice(&mut device) }.ok().and(device);
    let device_id = device.as_ref().map_or(0, device_identity);
    // SAFETY: a documented call on a live device.
    let made = device.as_ref().and_then(|d| unsafe { d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok());
    let exec_made = device.and_then(|d| unsafe { d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok());
    let fence = match made {
        Some(f) => f.into_raw() as usize,
        None => {
            static LOGGED: AtomicI32 = AtomicI32::new(0);
            if crate::proxy::log_due(&LOGGED) {
                xs_log!("deferred release: no fence for queue {key:#x}, what goes while it has work is held until its device goes");
            }
            0
        }
    };
    let exec_fence = exec_made.map_or(0, |f| f.into_raw() as usize);
    st.queues.insert(
        key,
        QueueTrack {
            fence,
            exec_fence,
            device: device_id,
            last_signal: 0,
            exec_signal: 0,
            open: 1,
            unsignalled: true,
            exec_failed: false,
        },
    );
    true
}

/// Ends a submission to the PC queue `queue` that [`begin`] opened: the queue signals its fence.
///
/// # Safety
/// `queue` is a live PC command queue.
pub(crate) unsafe fn submitted(queue: *mut c_void) {
    // SAFETY: by the contract above.
    let Some(q) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return };
    let mut st = state();
    let Some(track) = st.queues.get_mut(&(queue as usize)) else { return };
    let fence = track.fence as *mut c_void;
    let exec_fence = track.exec_fence as *mut c_void;
    // SAFETY: the table keeps both fences alive and the queue and fences belong to one device.
    // A queue without a fence signals nothing, so its target is never reached.
    track.close_with_exec(
        |value| unsafe { ID3D12Fence::from_raw_borrowed(&fence).is_some_and(|f| q.Signal(f, value).is_ok()) },
        |value| unsafe { ID3D12Fence::from_raw_borrowed(&exec_fence).is_some_and(|f| q.Signal(f, value).is_ok()) },
    );
}

/// Fence targets covering every completed submission across all tracked queues, excluding queued
/// submissions that have not executed yet (`open > 0`). Returns `None` when any queue has work
/// whose completion cannot be proven by a signalled fence.
fn submitted_targets(queues: &FastMap<QueueTrack>) -> Option<Vec<Target>> {
    let mut targets = Vec::new();
    for track in queues.values() {
        if track.fence == 0 {
            if track.last_signal > 0 || track.unsignalled || track.open > 0 {
                return None;
            }
            continue;
        }
        if track.exec_failed || (track.open == 0 && track.unsignalled) {
            return None;
        }
        if track.last_signal > 0 {
            targets.push((track.fence, track.last_signal));
        }
        if track.exec_fence != 0 && track.exec_signal > 0 {
            targets.push((track.exec_fence, track.exec_signal));
        }
    }
    Some(targets)
}

/// Waits up to `timeout_ms` for every already-submitted GPU command batch across all tracked
/// queues to complete, without waiting on queued submissions that have not executed yet.
/// Returns `true` when every tracked queue's completed submissions are proven finished.
pub(crate) fn wait_submitted_idle(timeout_ms: u32) -> bool {
    let fences: Vec<(ID3D12Fence, u64)> = {
        let st = state();
        let Some(targets) = submitted_targets(&st.queues) else {
            return false;
        };
        let mut owned = Vec::with_capacity(targets.len());
        for (fence, value) in targets {
            let raw = fence as *mut c_void;
            // SAFETY: every non-zero fence in `st.queues` is live while `state()` is held;
            // cloning takes an owned COM reference that stays valid after releasing the lock.
            if let Some(f) = unsafe { ID3D12Fence::from_raw_borrowed(&raw) } {
                owned.push((f.clone(), value));
            } else {
                return false;
            }
        }
        owned
    };
    if fences.is_empty() {
        return true;
    }
    let _pc_call = crate::watchdog_hold::pc_call();
    // SAFETY: an unnamed auto-reset event, closed below.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return false;
    };
    let deadline = Instant::now() + Duration::from_millis(u64::from(timeout_ms));
    let mut all_done = true;
    for (fence, value) in &fences {
        // SAFETY: `fence` is a live owned COM fence and `event` is the handle created above.
        unsafe {
            if fence.GetCompletedValue() >= *value {
                continue;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let wait_ms = u32::try_from(remaining.as_millis()).unwrap_or(u32::MAX);
            if wait_ms == 0 || fence.SetEventOnCompletion(*value, event).is_err() {
                if fence.GetCompletedValue() < *value {
                    all_done = false;
                    break;
                }
                continue;
            }
            WaitForSingleObject(event, wait_ms);
            if fence.GetCompletedValue() < *value {
                all_done = false;
                break;
            }
        }
    }
    // SAFETY: the event handle created above.
    unsafe {
        let _ = CloseHandle(event);
    }
    all_done
}

/// The targets of a release now: one per queue that has work in flight or a submission under way.
fn targets_from_snapshot(queues: &FastMap<QueueTrack>, snapshot: &[(usize, u64)]) -> Vec<Target> {
    queues
        .iter()
        .filter_map(|(&queue, track)| {
            let key = track.key(queue);
            wait_target(reached_in(snapshot, key), track.last_signal, track.unsignalled).map(|value| (key, value))
        })
        .collect()
}

fn collect_fifo_waits(queues: &FastMap<QueueTrack>, pending: &[crate::queue_fifo::PendingWork]) -> Vec<FifoWait> {
    pending
        .iter()
        .map(|work| FifoWait {
            work: work.clone(),
            snapshot_signal: queues.get(&work.queue()).map_or(0, |t| t.last_signal),
        })
        .collect()
}

fn resolve_fifo_waits(queues: &FastMap<QueueTrack>, snapshot: &[(usize, u64)], waits: &mut Vec<FifoWait>, targets: &mut Vec<Target>) -> bool {
    waits.retain(|fw| {
        if !fw.work.is_finished() {
            return true;
        }
        if let Some(track) = queues.get(&fw.work.queue())
            && (track.unsignalled || track.last_signal > fw.snapshot_signal)
        {
            let key = track.key(fw.work.queue());
            if let Some(target) = wait_target(reached_in(snapshot, key), track.last_signal, track.unsignalled)
                && !targets.contains(&(key, target))
            {
                targets.push((key, target));
            }
        }
        false
    });
    waits.is_empty()
}

/// The last reference of the PC object `object` goes. The object is released once the GPU is past
/// the work that is in flight now. `what` tells what it is (title texture, upload buffer, ...).
///
/// # Safety
/// `object` is null or a PC object whose last reference of the layer this call takes over.
pub(crate) unsafe fn release(object: *mut c_void, what: &'static str) {
    retire(object, what, None);
}

/// Like [`release`], for an allocation that was charged to [`crate::mem_owners`] when it was made.
/// The charge is only removed when the object is really released, after the queues are done with it.
///
/// # Safety
/// As [`release`], and `charge` is the one charge added for `object`.
pub(crate) unsafe fn release_owned(object: *mut c_void, what: &'static str, charge: Charge) {
    retire(object, what, Some(charge));
}

fn retire(object: *mut c_void, what: &'static str, charge: Option<Charge>) {
    if object.is_null() {
        return;
    }
    let ready = {
        let mut st = state();
        let st = &mut *st;
        let snapshot = snapshot_reached(&st.queues);
        let targets = targets_from_snapshot(&st.queues, &snapshot);
        let pending = crate::queue_fifo::pending_all();
        let fifo_waits = collect_fifo_waits(&st.queues, &pending);
        if !targets.is_empty() || !fifo_waits.is_empty() {
            let count = HELD_BACK.fetch_add(1, Ordering::Relaxed) + 1;
            if count <= 20 || count.is_multiple_of(1000) {
                let label = describe(object, what).2;
                xs_log!(
                    "deferred release {count}: {label} goes while {} queues and {} fifo items still have work in flight, held until it finishes",
                    targets.len(),
                    fifo_waits.len()
                );
            }
        }
        let queues = &st.queues;
        let mut ready: Vec<Retired> = st.parked.park_with_waits(Retired { object: object as usize, what, charge }, targets, fifo_waits).into_iter().collect();
        if !st.parked.is_empty() {
            ready.extend(st.parked.release_reached_with(
                |key| reached_in(&snapshot, key),
                |waits, tgts| resolve_fifo_waits(queues, &snapshot, waits, tgts),
            ));
        }
        ready
    };
    if !ready.is_empty() {
        // SAFETY: each item was parked once (or bypassed parking) and is released once outside the lock.
        let freed = unsafe { release_retired(ready) };
        record_freed(freed);
    }
}

/// Waits until each fence of `targets` has reached its value, for at most [`FLUSH_WAIT_MS`] each.
///
/// # Safety
/// The fences of `targets` are live.
unsafe fn wait_for(targets: &[Target]) {
    // SAFETY: an unnamed auto reset event, closed below.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else { return };
    for &(fence, value) in targets {
        let raw = fence as *mut c_void;
        // SAFETY: by the contract above.
        unsafe {
            if let Some(f) = ID3D12Fence::from_raw_borrowed(&raw)
                && f.GetCompletedValue() < value
                && f.SetEventOnCompletion(value, event).is_ok()
            {
                WaitForSingleObject(event, FLUSH_WAIT_MS);
            }
        }
    }
    // SAFETY: the event made above.
    unsafe {
        let _ = CloseHandle(event);
    }
}

/// Names and releases `items` without holding `state()`, returning fault-report records.
///
/// # Safety
/// The object of each item is a live PC object, and the reference of the layer is released here once.
unsafe fn release_retired(items: Vec<Retired>) -> Vec<Freed> {
    let mut freed = Vec::with_capacity(items.len());
    let now = ms_now();
    for item in items {
        let raw = item.object as *mut c_void;
        let (va, bytes, label) = describe(raw, item.what);
        // SAFETY: a live PC object, and every `ID3D12Object` has SetName.
        unsafe {
            if let Some(named) = ID3D12Object::from_raw_borrowed(&raw) {
                let _ = named.SetName(&HSTRING::from(format!("decant {label}")));
            }
        }
        freed.push(Freed { va, bytes, label, at_ms: now });
        RELEASED.fetch_add(1, Ordering::Relaxed);
        if let Some(charge) = item.charge {
            crate::mem_owners::remove_charge(charge);
        }
        // SAFETY: by the contract above.
        unsafe { drop(IUnknown::from_raw(raw)) };
    }
    freed
}

fn record_freed(freed: Vec<Freed>) {
    if freed.is_empty() {
        return;
    }
    let mut st = state();
    for entry in freed {
        if st.recent.len() == RECENT {
            st.recent.pop_front();
        }
        st.recent.push_back(entry);
    }
}

/// The GPU range and a label of the PC object `raw`. A buffer is described by its address range,
/// a texture by its size and format.
fn describe(raw: *mut c_void, what: &'static str) -> (u64, u64, String) {
    // SAFETY: a live PC object that is asked for one of its interfaces.
    let resource = unsafe { IUnknown::from_raw_borrowed(&raw) }.and_then(|u| u.cast::<ID3D12Resource>().ok());
    let Some(resource) = resource else { return (0, 0, format!("{what} {raw:p}")) };
    // SAFETY: documented calls on a live resource.
    let (desc, va) = unsafe { (resource.GetDesc(), resource.GetGPUVirtualAddress()) };
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        (va, desc.Width, format!("{what} {raw:p} buffer {:#x} bytes at GPU {va:#x}", desc.Width))
    } else {
        (0, 0, format!("{what} {raw:p} texture {}x{}x{} format {} mips {}", desc.Width, desc.Height, desc.DepthOrArraySize, desc.Format.0, desc.MipLevels))
    }
}

/// Called at each present. Releases what the GPU is done with and now and then logs the counters.
pub(crate) fn present() {
    let presents = PRESENTS.fetch_add(1, Ordering::Relaxed) + 1;
    let ready = {
        let mut st = state();
        if st.parked.is_empty() {
            if presents.is_multiple_of(SUMMARY_EVERY) {
                xs_log!(
                    "deferred release at present {presents}: {} objects held now (peak {}), {} released in all, {} of them held back for work in flight",
                    st.parked.len(),
                    st.parked.most(),
                    RELEASED.load(Ordering::Relaxed),
                    HELD_BACK.load(Ordering::Relaxed)
                );
            }
            return;
        }
        let st = &mut *st;
        let snapshot = snapshot_reached(&st.queues);
        let queues = &st.queues;
        st.parked.release_reached_with(
            |key| reached_in(&snapshot, key),
            |waits, tgts| resolve_fifo_waits(queues, &snapshot, waits, tgts),
        )
    };
    if !ready.is_empty() {
        // SAFETY: each item is released once outside the lock.
        let freed = unsafe { release_retired(ready) };
        record_freed(freed);
    }
    if presents.is_multiple_of(SUMMARY_EVERY) {
        let st = state();
        xs_log!(
            "deferred release at present {presents}: {} objects held now (peak {}), {} released in all, {} of them held back for work in flight",
            st.parked.len(),
            st.parked.most(),
            RELEASED.load(Ordering::Relaxed),
            HELD_BACK.load(Ordering::Relaxed)
        );
    }
}

/// The identity of a PC device is its `IUnknown`, which is the same pointer through any interface.
fn device_identity(device: &impl Interface) -> usize {
    device.cast::<IUnknown>().map_or(0, |u| u.as_raw() as usize)
}

/// The PC device `device` goes. Each of its queues is given up to [`FLUSH_WAIT_MS`] to finish the
/// work signalled so far. Then what only those queues held back is released and their fences are
/// dropped. Objects that are also held for a queue of another device stay held.
///
/// # Safety
/// `device` is a live PC device.
pub(crate) unsafe fn flush(device: *mut c_void) {
    // SAFETY: by the contract above.
    let device_id = unsafe { IUnknown::from_raw_borrowed(&device) }.map_or(0, device_identity);
    let ready = {
        let mut st = state();
        let active_queues: Vec<usize> = crate::queue_fifo::pending_all().into_iter().map(|w| w.queue()).collect();
        let teardown = Teardown::of_with_active(&st.queues, device_id, &active_queues);
        if !teardown.kept.is_empty() {
            xs_log!("deferred release: the device goes while {} of its queues are inside a submission or have no fence, what they hold back stays held", teardown.kept.len());
        }
        // SAFETY: the fences in the table are live.
        unsafe { wait_for(&teardown.to_wait_for) };
        let ready = st.parked.drop_keys_and_queues(&teardown.keys, &teardown.queues);
        for queue in teardown.queues {
            let Some(track) = st.queues.remove(&queue) else { continue };
            if track.fence != 0 {
                // SAFETY: the table held the one reference to the fence and it is released here once.
                unsafe { drop(ID3D12Fence::from_raw(track.fence as *mut c_void)) };
            }
            if track.exec_fence != 0 {
                // SAFETY: the table held the one reference to the exec fence and it is released here once.
                unsafe { drop(ID3D12Fence::from_raw(track.exec_fence as *mut c_void)) };
            }
        }
        ready
    };
    if !ready.is_empty() {
        // SAFETY: each item is released once outside the lock.
        let freed = unsafe { release_retired(ready) };
        record_freed(freed);
    }
}

/// The queues of a device that goes. A queue inside a submission (between [`begin`] and
/// [`submitted`]) is left alone. Its signal comes after the submission, which cannot end while the
/// teardown holds the lock, so it keeps its fence and its targets, and what it holds back is
/// released by a later release or present once the signal is reached. A queue without a fence is
/// left alone too, because nothing can prove that its work finished and what it holds back stays held.
#[derive(Debug, PartialEq, Eq)]
struct Teardown {
    /// The queues that are torn down, by their key in the table.
    queues: Vec<usize>,
    /// The queues that are kept: inside a submission, or without a fence.
    kept: Vec<usize>,
    /// The keys that targets of the torn down queues use.
    keys: Vec<usize>,
    /// Each fence to wait for and the last value signalled on it.
    to_wait_for: Vec<Target>,
}

impl Teardown {
    #[cfg(test)]
    fn of(queues: &FastMap<QueueTrack>, device: usize) -> Self {
        Self::of_with_active(queues, device, &[])
    }

    fn of_with_active(queues: &FastMap<QueueTrack>, device: usize, active_queues: &[usize]) -> Self {
        let mut own: Vec<(usize, QueueTrack)> = queues.iter().filter(|(_, t)| t.device == device).map(|(&q, &t)| (q, t)).collect();
        own.sort_by_key(|&(queue, _)| queue);
        let (going, kept): (Vec<_>, Vec<_>) = own.into_iter().partition(|(q, t)| t.open == 0 && t.fence != 0 && !active_queues.contains(q));
        Self {
            queues: going.iter().map(|&(q, _)| q).collect(),
            kept: kept.iter().map(|&(q, _)| q).collect(),
            keys: going.iter().map(|(q, t)| t.key(*q)).collect(),
            to_wait_for: going.iter().map(|(_, t)| (t.fence, t.last_signal)).collect(),
        }
    }
}

/// The freed objects whose range holds `va`. When there are none, the freed buffers right below
/// and right above it. This is for the page fault report.
pub(crate) fn near(va: u64) -> Vec<Freed> {
    nearest(&state().recent, va)
}

fn nearest(recent: &VecDeque<Freed>, va: u64) -> Vec<Freed> {
    let buffers = || recent.iter().filter(|f| f.bytes > 0);
    let mut found: Vec<Freed> = buffers().filter(|f| f.va <= va && va < f.va + f.bytes).cloned().collect();
    if found.is_empty() {
        found.extend(buffers().filter(|f| f.va + f.bytes <= va).max_by_key(|f| f.va + f.bytes).cloned());
        found.extend(buffers().filter(|f| f.va > va).min_by_key(|f| f.va).cloned());
    }
    found
}

/// How many objects were released in all, how many of them waited for work in flight, and how many
/// are held now. The fault report prints these.
pub(crate) fn counts() -> (u64, u64, usize) {
    (RELEASED.load(Ordering::Relaxed), HELD_BACK.load(Ordering::Relaxed), state().parked.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(fence: usize, device: usize, last_signal: u64) -> QueueTrack {
        QueueTrack { fence, exec_fence: 0, device, last_signal, exec_signal: 0, open: 0, unsignalled: false, exec_failed: false }
    }

    #[test]
    fn the_target_depends_on_how_far_the_gpu_is() {
        assert_eq!(wait_target(7, 7, false), None, "idle");
        assert_eq!(wait_target(2, 7, false), Some(7), "behind: the last signal");
        assert_eq!(wait_target(7, 7, true), Some(8), "a submission under way: the signal to come");
        assert_eq!(wait_target(0, 0, true), Some(1));
        assert_eq!(wait_target(u64::MAX, 7, true), None, "removed device");
    }

    #[test]
    fn an_item_without_targets_comes_straight_back() {
        let mut p = Parking::default();
        assert_eq!(p.park('a', vec![]), Some('a'));
        assert_eq!(p.len(), 0);
        assert_eq!(p.most(), 0);
    }

    #[test]
    fn an_item_goes_when_all_of_its_fences_are_there() {
        let mut p = Parking::default();
        assert!(p.park("both", vec![(1, 3), (2, 3)]).is_none());
        assert!(p.park("one", vec![(1, 4)]).is_none());
        assert_eq!(p.most(), 2);
        let at = |a: u64, b: u64| move |key: usize| if key == 1 { a } else { b };
        assert!(p.release_reached(at(3, 2)).is_empty());
        assert_eq!(p.release_reached(at(4, 2)), ["one"], "the item parked later can go first");
        assert_eq!(p.release_reached(at(4, 3)), ["both"]);
        assert_eq!((p.len(), p.most()), (0, 2));
    }

    #[test]
    fn items_come_out_in_the_order_they_were_parked() {
        let mut p = Parking::default();
        for n in 0..5 {
            p.park(n, vec![(9, 1)]);
        }
        assert_eq!(p.release_reached(|_| 1), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_removed_device_reaches_every_target() {
        let mut p = Parking::default();
        p.park(1, vec![(1, 100), (2, 200)]);
        assert_eq!(p.release_reached(|_| u64::MAX), [1]);
    }

    #[test]
    fn overlapping_submissions_signal_once_at_the_end() {
        let mut t = QueueTrack { open: 2, unsignalled: true, ..track(1, 0, 4) };
        let target = wait_target(4, t.last_signal, t.unsignalled).unwrap();
        let mut p = Parking::default();
        for n in 0..300 {
            p.park(n, vec![(t.fence, target)]);
        }
        t.close(|_| panic!("another submission can still enqueue lists"));
        assert!(p.release_reached(|_| t.last_signal).is_empty());
        t.close(|value| {
            assert_eq!(value, target);
            true
        });
        assert!(!t.unsignalled);
        assert_eq!(p.release_reached(|_| t.last_signal).len(), 300);
    }

    #[test]
    fn a_failed_signal_leaves_later_releases_behind_the_work() {
        let mut t = QueueTrack { open: 1, unsignalled: true, ..track(1, 0, 4) };
        t.close(|_| false);
        assert_eq!((t.last_signal, t.unsignalled), (4, true));
        assert_eq!(wait_target(4, t.last_signal, t.unsignalled), Some(5));
    }

    #[test]
    fn a_device_teardown_only_frees_what_its_own_queues_hold() {
        let queues: FastMap<QueueTrack> = [(100, track(10, 0xA, 3)), (200, track(20, 0xB, 7))].into_iter().collect();
        let t = Teardown::of(&queues, 0xA);
        assert_eq!(t, Teardown { queues: vec![100], kept: vec![], keys: vec![10], to_wait_for: vec![(10, 3)] });
        let mut p = Parking::default();
        p.park("only A", vec![(10, 3)]);
        p.park("only B", vec![(20, 8)]);
        p.park("A and B", vec![(10, 3), (20, 8)]);
        assert_eq!(p.drop_keys(&t.keys), ["only A"]);
        assert_eq!(p.len(), 2);
        let b_only = |key: usize| if key == 20 { 8 } else { panic!("fence {key} belongs to the torn down device") };
        assert_eq!(p.release_reached(b_only), ["only B", "A and B"]);
    }

    #[test]
    fn a_queue_without_a_fence_holds_what_it_gets_for_good() {
        let mut t = QueueTrack { open: 1, unsignalled: true, ..track(0, 0xA, 0) };
        let queues: FastMap<QueueTrack> = [(300, t)].into_iter().collect();
        assert_eq!(t.key(300), 300);
        assert_eq!(reached(&queues, 300), 0);
        let target = wait_target(0, t.last_signal, t.unsignalled).expect("held while it has work");
        let mut p = Parking::default();
        assert!(p.park("upload", vec![(300, target)]).is_none());
        t.close(|_| false);
        assert_eq!(wait_target(0, t.last_signal, t.unsignalled), Some(1), "later releases wait too");
        assert!(p.release_reached(|key| reached(&queues, key)).is_empty());
        let teardown = Teardown::of(&[(300, t)].into_iter().collect(), 0xA);
        assert_eq!(teardown, Teardown { queues: vec![], kept: vec![300], keys: vec![], to_wait_for: vec![] });
        assert!(p.drop_keys(&teardown.keys).is_empty());
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn a_teardown_leaves_a_queue_inside_a_submission_alone() {
        let busy = QueueTrack { open: 1, unsignalled: true, ..track(10, 0xA, 3) };
        let queues: FastMap<QueueTrack> = [(100, busy), (200, track(20, 0xA, 5)), (300, track(30, 0xC, 1))].into_iter().collect();
        let t = Teardown::of(&queues, 0xA);
        assert_eq!(t, Teardown { queues: vec![200], kept: vec![100], keys: vec![20], to_wait_for: vec![(20, 5)] });
        let mut p = Parking::default();
        p.park("busy only", vec![(10, 4)]);
        p.park("busy and idle", vec![(10, 4), (20, 5)]);
        assert!(p.drop_keys(&t.keys).is_empty(), "both still wait for the signal of the open submission");
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn the_fault_report_gives_the_hit_or_the_neighbours() {
        let freed = |va, bytes, label: &str| Freed { va, bytes, label: label.into(), at_ms: 0 };
        let recent: VecDeque<Freed> = [freed(0x1000, 0x1000, "low"), freed(0x4000, 0x100, "mid"), freed(0, 0, "texture"), freed(0x8000, 0x10, "high")].into();
        let labels = |va| nearest(&recent, va).into_iter().map(|f| f.label).collect::<Vec<_>>();
        assert_eq!(labels(0x1800), ["low"]);
        assert_eq!(labels(0x5000), ["mid", "high"]);
        assert_eq!(labels(0x100), ["low"], "nothing below");
        assert_eq!(labels(0x9000), ["high"], "nothing above");
    }

    #[test]
    fn empty_parking_and_snapshot_lookup_behave_cleanly() {
        let mut p: Parking<u32> = Parking::default();
        assert!(p.is_empty());
        assert!(p.release_reached(|_| panic!("should not query empty parking")).is_empty());
        assert!(p.drop_keys(&[1, 2]).is_empty());
        let snap = [(10usize, 5u64), (20usize, 9u64)];
        assert_eq!(reached_in(&snap, 10), 5);
        assert_eq!(reached_in(&snap, 20), 9);
        assert_eq!(reached_in(&snap, 99), u64::MAX);
    }

    #[test]
    fn submitted_targets_cover_executed_work_without_waiting_on_queued_submissions() {
        let mut q1 = QueueTrack { exec_fence: 101, open: 2, unsignalled: true, ..track(10, 0xA, 4) };
        q1.close_with_exec(|_| panic!("main fence stays unsignalled while open > 0"), |v| {
            assert_eq!(v, 1);
            true
        });
        assert_eq!((q1.open, q1.last_signal, q1.exec_signal, q1.exec_failed), (1, 4, 1, false));
        let queues: FastMap<QueueTrack> = [(1, q1), (2, track(20, 0xA, 7))].into_iter().collect();
        let mut targets = submitted_targets(&queues).expect("all executed work is fenced");
        targets.sort();
        assert_eq!(targets, vec![(10, 4), (20, 7), (101, 1)]);

        // A failed intermediate exec signal prevents proving idle until the queue closes cleanly.
        let mut q2 = QueueTrack { exec_fence: 0, open: 2, unsignalled: true, ..track(30, 0xA, 2) };
        q2.close_with_exec(|_| unreachable!(), |_| false);
        assert!(q2.exec_failed);
        let broken: FastMap<QueueTrack> = [(3, q2)].into_iter().collect();
        assert_eq!(submitted_targets(&broken), None);
        q2.close_with_exec(|v| {
            assert_eq!(v, 3);
            true
        }, |_| unreachable!());
        assert!(!q2.exec_failed);
        let recovered: FastMap<QueueTrack> = [(3, q2)].into_iter().collect();
        assert_eq!(submitted_targets(&recovered), Some(vec![(30, 3)]));

        // A queue without a fence cannot prove its executed work is idle.
        let unfenced: FastMap<QueueTrack> = [(4, QueueTrack { open: 1, unsignalled: true, ..track(0, 0xA, 0) })].into_iter().collect();
        assert_eq!(submitted_targets(&unfenced), None);
    }

    #[test]
    fn queue_fifo_pending_work_holds_release_until_execution_and_fence_signal() {
        let q_inline = 0x7200_0001usize;
        let q_tile = 0x7200_0002usize;
        let fence_inline = 0x9901usize;

        let gate = crate::queue_fifo::inline_gate(q_inline, || true).expect("queue idle");
        let (tile_done_tx, tile_done_rx) = std::sync::mpsc::channel::<()>();
        let (tile_go_tx, tile_go_rx) = std::sync::mpsc::channel::<()>();
        crate::queue_fifo::defer(q_tile, Box::new(move || {
            let _ = tile_done_tx.send(());
            let _ = tile_go_rx.recv();
        }));
        tile_done_rx.recv_timeout(Duration::from_secs(5)).unwrap();

        let pending = crate::queue_fifo::pending_all();
        let mut queues: FastMap<QueueTrack> = [(q_inline, track(fence_inline, 0xA, 4)), (q_tile, track(0x9902, 0xA, 7))].into_iter().collect();
        let waits_all = collect_fifo_waits(&queues, &pending);
        let inline_wait: Vec<_> = waits_all.iter().filter(|w| w.work.queue() == q_inline).cloned().collect();
        let tile_wait: Vec<_> = waits_all.iter().filter(|w| w.work.queue() == q_tile).cloned().collect();
        assert_eq!(inline_wait.len(), 1);
        assert_eq!(tile_wait.len(), 1);

        let mut p = Parking::default();
        assert!(p.park_with_waits("inline-op", vec![], inline_wait).is_none());
        assert!(p.park_with_waits("tile-op", vec![], tile_wait).is_none());

        let snap_before = [(fence_inline, 4u64), (0x9902, 7u64)];
        assert!(p.release_reached_with(|k| reached_in(&snap_before, k), |w, t| resolve_fifo_waits(&queues, &snap_before, w, t)).is_empty());

        // Simulate inline submission advancing `last_signal` to 5 before dropping `Gate`.
        queues.get_mut(&q_inline).unwrap().last_signal = 5;
        drop(gate);
        let _ = tile_go_tx.send(());
        assert!(crate::queue_fifo::drain(q_tile, Duration::from_secs(5)));

        // At fence 4, tile-op (no new signal) releases, while inline-op promoted fence 5 and stays held.
        let released = p.release_reached_with(|k| reached_in(&snap_before, k), |w, t| resolve_fifo_waits(&queues, &snap_before, w, t));
        assert_eq!(released, ["tile-op"]);
        assert_eq!(p.len(), 1);

        // Once fence reaches 5, inline-op is released cleanly.
        let snap_after = [(fence_inline, 5u64), (0x9902, 7u64)];
        let released2 = p.release_reached_with(|k| reached_in(&snap_after, k), |w, t| resolve_fifo_waits(&queues, &snap_after, w, t));
        assert_eq!(released2, ["inline-op"]);
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn teardown_with_active_keeps_fifo_busy_queue_and_close_never_underflows() {
        let mut t = track(10, 0xA, 3);
        t.close(|_| panic!("close when open == 0 must be a no-op"));
        assert_eq!((t.open, t.last_signal), (0, 3));

        let queues: FastMap<QueueTrack> = [(100, track(10, 0xA, 3)), (200, track(20, 0xA, 5))].into_iter().collect();
        let td = Teardown::of_with_active(&queues, 0xA, &[100]);
        assert_eq!(td, Teardown { queues: vec![200], kept: vec![100], keys: vec![20], to_wait_for: vec![(20, 5)] });
    }
}
