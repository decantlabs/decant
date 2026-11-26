//! Busy command-allocator swapping on `Reset` and in-flight list-resubmit diagnostics.
//!
//! Ported from ivall/xcl (GPL-2.0-only) and adapted to this codebase.
//!
//! On PC D3D12 a command list's recording lives inside its command allocator: after
//! `allocator.Reset()` any list recorded on that allocator must not be submitted again (debug
//! layer ID 541 -> device removal `DXGI_ERROR_DEVICE_REMOVED`). On the console a title can reset
//! its allocator while recordings on it are still held or still executing on the GPU.
//!
//! This module treats each title allocator as a logical name backed by a PC `ID3D12CommandAllocator`
//! that swaps on demand: when `allocator.Reset()` arrives while the active backing still has live
//! list recordings or unfinished GPU fence marks, that backing is parked in `aside` and replaced by
//! a free spare (or a newly created PC allocator of the same type). Subsequent `list.Reset` /
//! `CreateCommandList` calls record on the new active backing. Parked backings are reused as soon
//! as all lists have moved off them and their queue fences complete.
//!
//! Also tracks per-list in-flight GPU fences and logs debug-layer ID 553 violations (submitting a
//! command list while its previous submit has not completed on the GPU).
//!
//! Switch: `DECANT_D3D12X_ALLOC_SWAP=0` disables allocator swapping and resubmit diagnostics (default
//! on; `0` = legacy direct passthrough).

#[cfg(any(windows, test))]
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[cfg(windows)]
use crate::log::xs_log;
#[cfg(not(windows))]
macro_rules! xs_log {
    ($($t:tt)*) => {};
}

/// Whether busy command-allocator swapping is enabled (`DECANT_D3D12X_ALLOC_SWAP`, default on;
/// `0` = off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("DECANT_D3D12X_ALLOC_SWAP").is_none_or(|v| v != "0");
        if !on {
            xs_log!("alloc pool: off (DECANT_D3D12X_ALLOC_SWAP=0)");
        }
        on
    })
}

/// Minimal fence completion query abstracted so the entire pool state machine runs in host unit
/// tests without a GPU.
#[cfg(any(windows, test))]
trait FenceMark: Clone {
    fn completed(&self) -> u64;
}

#[cfg(any(windows, test))]
struct Backing<F> {
    /// PC lists whose current recording lives on this PC allocator backing.
    users: HashSet<usize>,
    /// PC queue -> `(fence, value)` after the last submit of any list on this backing.
    marks: HashMap<usize, (F, u64)>,
}

#[cfg(any(windows, test))]
impl<F> Default for Backing<F> {
    fn default() -> Self {
        Self { users: HashSet::new(), marks: HashMap::new() }
    }
}

#[cfg(any(windows, test))]
impl<F: FenceMark> Backing<F> {
    fn free(&self) -> bool {
        self.users.is_empty() && self.marks.values().all(|(f, v)| f.completed() >= *v)
    }
}

#[cfg(any(windows, test))]
struct LastSubmit<F> {
    queue: usize,
    fence: F,
    value: u64,
    recording: u64,
}

#[cfg(any(windows, test))]
struct StateCore<F> {
    /// Title PC allocator -> active PC allocator backing (absent: itself).
    cur: HashMap<usize, usize>,
    /// Title PC allocator -> parked PC allocators put aside while still busy.
    aside: HashMap<usize, Vec<usize>>,
    backings: HashMap<usize, Backing<F>>,
    /// PC list -> PC allocator backing holding its current recording.
    of_list: HashMap<usize, usize>,
    /// Title PC allocator -> `D3D12_COMMAND_LIST_TYPE` integer.
    types: HashMap<usize, i32>,
    /// PC queue -> `(fence, next_value)` used to track GPU completion.
    #[cfg_attr(not(windows), allow(dead_code))]
    fences: HashMap<usize, (F, u64)>,
    made: u64,
    swaps: u64,
    /// PC list -> its last submission `(queue, fence, value, recording)`.
    last_submit: HashMap<usize, LastSubmit<F>>,
    /// PC list -> recording generation counter.
    recordings: HashMap<usize, u64>,
    resubmits: u64,
}

#[cfg(any(windows, test))]
impl<F> Default for StateCore<F> {
    fn default() -> Self {
        Self {
            cur: HashMap::new(),
            aside: HashMap::new(),
            backings: HashMap::new(),
            of_list: HashMap::new(),
            types: HashMap::new(),
            fences: HashMap::new(),
            made: 0,
            swaps: 0,
            last_submit: HashMap::new(),
            recordings: HashMap::new(),
            resubmits: 0,
        }
    }
}

#[cfg(any(windows, test))]
impl<F: FenceMark> StateCore<F> {
    fn move_list(&mut self, list: usize, backing: usize) {
        if let Some(old) = self.of_list.insert(list, backing)
            && old != backing
            && let Some(b) = self.backings.get_mut(&old)
        {
            b.users.remove(&list);
        }
        self.backings.entry(backing).or_default().users.insert(list);
    }

    fn current(&self, title_alloc: usize) -> usize {
        self.cur.get(&title_alloc).copied().unwrap_or(title_alloc)
    }

    fn forget_list(&mut self, list: usize) {
        self.last_submit.remove(&list);
        self.recordings.remove(&list);
        if let Some(backing) = self.of_list.remove(&list)
            && let Some(b) = self.backings.get_mut(&backing)
        {
            b.users.remove(&list);
        }
    }

    /// Removes `title_alloc` and returns any pool-created spare PC allocators that are free to
    /// retire/release outside the lock.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn forget_alloc(&mut self, title_alloc: usize) -> Vec<usize> {
        self.types.remove(&title_alloc);
        let mut mine = self.aside.remove(&title_alloc).unwrap_or_default();
        mine.extend(self.cur.remove(&title_alloc));
        mine.push(title_alloc);
        let mut made = Vec::new();
        for b in mine {
            let free = self.backings.get(&b).is_none_or(Backing::free);
            if let Some(gone) = self.backings.remove(&b) {
                for l in gone.users {
                    self.of_list.remove(&l);
                }
            }
            if b != title_alloc && free {
                made.push(b);
            }
        }
        made
    }

    fn note_resubmits_in(&mut self, queue: usize, lists: &[usize]) -> usize {
        let mut found = 0usize;
        for &l in lists {
            let Some(last) = self.last_submit.get(&l) else { continue };
            if last.fence.completed() >= last.value {
                continue;
            }
            let same_recording = self.recordings.get(&l).copied().unwrap_or(0) == last.recording;
            let same_queue = last.queue == queue;
            found += 1;
            self.resubmits += 1;
            let n = self.resubmits;
            if n <= 8 || n.is_power_of_two() {
                let rec = if same_recording { "the same recording" } else { "a new recording" };
                let q = if same_queue { "the same queue" } else { "another queue" };
                let _ = (rec, q, n);
                xs_log!(
                    "alloc pool: a command list is submitted while its previous submit is still on the GPU ({rec}, {q}; #{n} so far) - PC forbids this (debug layer id 553)"
                );
            }
        }
        found
    }

    /// Core allocator-reset transition, parameterized by `reset` and `create` closures so failure
    /// retry and spare reuse are testable without a GPU.
    ///
    /// Returns `None` when the title's own PC allocator is free and should be reset directly by the
    /// caller; returns `Some(hr)` when handled here (current spare reset, or swapped to a spare /
    /// new allocator).
    fn title_reset_in(
        &mut self,
        title_alloc: usize,
        mut reset: impl FnMut(usize) -> i32,
        mut create: impl FnMut(i32) -> Option<usize>,
    ) -> Option<i32> {
        let cur = self.current(title_alloc);
        if self.backings.get(&cur).is_none_or(Backing::free) {
            if let Some(b) = self.backings.get_mut(&cur) {
                b.marks.clear();
            }
            if cur == title_alloc {
                return None;
            }
            return Some(reset(cur));
        }
        let ty = *self.types.get(&title_alloc)?;
        let reuse = self
            .aside
            .get(&title_alloc)
            .and_then(|v| v.iter().copied().find(|f| self.backings.get(f).is_none_or(Backing::free)));
        let next = match reuse {
            Some(spare) => {
                // Keep ownership in `aside` until Reset succeeds so a failed Reset can be retried.
                let hr = reset(spare);
                if hr < 0 {
                    return Some(hr);
                }
                if let Some(v) = self.aside.get_mut(&title_alloc) {
                    v.retain(|&x| x != spare);
                }
                if let Some(b) = self.backings.get_mut(&spare) {
                    b.marks.clear();
                }
                spare
            }
            None => {
                let raw = create(ty)?;
                self.made += 1;
                raw
            }
        };
        self.aside.entry(title_alloc).or_default().push(cur);
        self.cur.insert(title_alloc, next);
        self.swaps += 1;
        if self.swaps == 1 || self.swaps.is_power_of_two() {
            let aside: usize = self.aside.values().map(Vec::len).sum();
            let (swaps, made) = (self.swaps, self.made);
            let _ = (swaps, made, aside);
            xs_log!(
                "alloc pool: {swaps} allocator resets with recordings still held or on the GPU (DECANT_D3D12X_ALLOC_SWAP=1) - another PC allocator took its place ({made} made, {aside} put aside now)"
            );
        }
        Some(0)
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod win {
    use super::*;
    use std::ffi::c_void;
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;

    impl FenceMark for ID3D12Fence {
        fn completed(&self) -> u64 {
            // SAFETY: live D3D12 fence object.
            unsafe { self.GetCompletedValue() }
        }
    }

    struct WinState(StateCore<ID3D12Fence>);

    // SAFETY: D3D12 fences are free-threaded; all state access is serialized by `STATE`.
    unsafe impl Send for WinState {}

    static STATE: Mutex<Option<WinState>> = Mutex::new(None);

    /// Records the `D3D12_COMMAND_LIST_TYPE` of a newly created title allocator.
    pub fn note_allocator_type(title_alloc: usize, list_type: i32) {
        if !enabled() || title_alloc == 0 {
            return;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut g.get_or_insert_with(|| WinState(StateCore::default())).0;
        s.types.insert(title_alloc, list_type);
    }

    /// Returns the active PC allocator backing currently representing `title_alloc`.
    pub fn current(title_alloc: usize) -> usize {
        if !enabled() || title_alloc == 0 {
            return title_alloc;
        }
        STATE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(title_alloc, |s| s.0.current(title_alloc))
    }

    /// Records a newly created PC command list `pc_list` recording on `backing` (for `title_alloc`).
    pub fn created_on(pc_list: usize, title_alloc: usize, backing: usize, list_type: i32) {
        if !enabled() || pc_list == 0 || backing == 0 {
            return;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut g.get_or_insert_with(|| WinState(StateCore::default())).0;
        if title_alloc != 0 && list_type >= 0 {
            s.types.entry(title_alloc).or_insert(list_type);
        }
        s.move_list(pc_list, backing);
        *s.recordings.entry(pc_list).or_default() += 1;
    }

    /// Resets `pc_list` on the active PC backing of `title_alloc`, running the native D3D12
    /// `Reset` callback OUTSIDE the pool lock and updating ownership only on success.
    pub fn reset_list(pc_list: usize, title_alloc: usize, reset: impl FnOnce(usize) -> i32) -> i32 {
        let backing = current(title_alloc);
        let hr = reset(backing);
        if hr < 0 || !enabled() || pc_list == 0 || backing == 0 {
            return hr;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut g.get_or_insert_with(|| WinState(StateCore::default())).0;
        s.move_list(pc_list, backing);
        *s.recordings.entry(pc_list).or_default() += 1;
        hr
    }

    /// Handles `ID3D12CommandAllocator::Reset` on `title_alloc`. Returns `None` when `title_alloc`
    /// itself is free and should be reset directly; returns `Some(hr)` when handled by swapping or
    /// resetting the active spare.
    ///
    /// # Safety
    /// `title_alloc` is a live PC `ID3D12CommandAllocator`; `reset` calls PC slot 8 on the passed
    /// raw PC allocator pointer.
    pub unsafe fn title_reset(title_alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> i32) -> Option<i32> {
        if !enabled() || title_alloc.is_null() {
            return None;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut g.get_or_insert_with(|| WinState(StateCore::default())).0;
        s.title_reset_in(
            title_alloc as usize,
            |raw| reset(raw as *mut c_void),
            |ty| unsafe {
                let this = ID3D12CommandAllocator::from_raw_borrowed(&title_alloc)?;
                let mut dev: Option<ID3D12Device> = None;
                this.GetDevice(&mut dev).ok()?;
                let made: ID3D12CommandAllocator = dev?.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).ok()?;
                Some(made.into_raw() as usize)
            },
        )
    }

    /// Checks whether any PC list in `pc_lists` is being submitted while its previous GPU execution
    /// has not finished (debug layer ID 553 diagnostic).
    ///
    /// # Safety
    /// `pc_queue` is a live PC `ID3D12CommandQueue`; `pc_lists` are live PC command lists.
    pub unsafe fn note_resubmits(pc_queue: *mut c_void, pc_lists: &[*mut c_void]) -> usize {
        if !enabled() || pc_queue.is_null() || pc_lists.is_empty() {
            return 0;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = g.as_mut() else { return 0 };
        let ids: Vec<usize> = pc_lists.iter().map(|&p| p as usize).collect();
        s.0.note_resubmits_in(pc_queue as usize, &ids)
    }

    /// Records that `pc_lists` were just submitted to `pc_queue`, signalling a per-queue tracking
    /// fence so active allocator backings remain protected until the GPU finishes them.
    ///
    /// # Safety
    /// `pc_queue` is a live PC `ID3D12CommandQueue`; `pc_lists` are live PC command lists.
    pub unsafe fn executed(pc_queue: *mut c_void, pc_lists: &[*mut c_void]) {
        if !enabled() || pc_queue.is_null() || pc_lists.is_empty() {
            return;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let s = &mut g.get_or_insert_with(|| WinState(StateCore::default())).0;
        // SAFETY: by contract above.
        let Some(q) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&pc_queue) }) else { return };
        let key = pc_queue as usize;
        if let std::collections::hash_map::Entry::Vacant(e) = s.fences.entry(key) {
            let mut dev: Option<ID3D12Device> = None;
            let fence = unsafe {
                q.GetDevice(&mut dev)
                    .ok()
                    .and(dev)
                    .and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok())
            };
            let Some(f) = fence else { return };
            e.insert((f, 0));
        }
        let (fence, value) = {
            let Some((f, v)) = s.fences.get_mut(&key) else { return };
            // SAFETY: live queue and fence.
            if unsafe { q.Signal(&*f, *v + 1) }.is_err() {
                return;
            }
            *v += 1;
            (f.clone(), *v)
        };
        for &l in pc_lists {
            let id = l as usize;
            if let Some(&b) = s.of_list.get(&id) {
                s.backings.entry(b).or_default().marks.insert(key, (fence.clone(), value));
            }
            let recording = s.recordings.get(&id).copied().unwrap_or(0);
            s.last_submit.insert(id, LastSubmit { queue: key, fence: fence.clone(), value, recording });
        }
    }

    /// Drops tracking for a released PC command list while keeping submitted-work fence marks on
    /// its backing allocator.
    pub fn forget_list(pc_list: usize) {
        if !enabled() || pc_list == 0 {
            return;
        }
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.as_mut() {
            s.0.forget_list(pc_list);
        }
    }

    /// Drops tracking for a released title allocator and retires any free spare PC allocators
    /// created for it outside the pool lock.
    ///
    /// # Safety
    /// Called only on final Release of the title allocator proxy.
    pub unsafe fn forget(title_alloc: usize) {
        if !enabled() || title_alloc == 0 {
            return;
        }
        let made = {
            let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
            g.as_mut().map_or_else(Vec::new, |s| s.0.forget_alloc(title_alloc))
        };
        for raw in made {
            // SAFETY: `raw` is an owned PC `ID3D12CommandAllocator` created by `title_reset` and
            // free of recordings and pending GPU work; deferred through `retire` for extra safety.
            unsafe { crate::retire::release(raw as *mut c_void, "pooled command allocator") };
        }
    }
}

#[cfg(windows)]
pub use win::{created_on, current, executed, forget, forget_list, note_allocator_type, note_resubmits, reset_list, title_reset};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Clone, Default)]
    struct MockFence(Arc<AtomicU64>);

    impl FenceMark for MockFence {
        fn completed(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn a_spare_whose_reset_fails_stays_available_for_retry() {
        let (alloc, spare, list) = (0x1000usize, 0x2000usize, 0x3000usize);
        let mut s = StateCore::<MockFence>::default();
        s.types.insert(alloc, 0);
        s.move_list(list, alloc);
        s.aside.insert(alloc, vec![spare]);
        s.backings.insert(spare, Backing::default());
        let mut calls = Vec::new();

        for _ in 0..2 {
            let hr = s.title_reset_in(
                alloc,
                |raw| {
                    calls.push(raw);
                    -2147467259 // E_FAIL
                },
                |_| panic!("must not create a new allocator while a spare exists"),
            );
            assert_eq!(hr, Some(-2147467259));
            assert_eq!(s.aside.get(&alloc), Some(&vec![spare]));
            assert!(!s.cur.contains_key(&alloc));
            assert_eq!(s.of_list.get(&list), Some(&alloc));
            assert!(s.backings[&alloc].users.contains(&list));
            assert_eq!((s.made, s.swaps), (0, 0));
        }

        let hr = s.title_reset_in(
            alloc,
            |raw| {
                calls.push(raw);
                0
            },
            |_| panic!("must reuse spare"),
        );
        assert_eq!(hr, Some(0));
        assert_eq!(calls, vec![spare; 3]);
        assert_eq!(s.cur.get(&alloc), Some(&spare));
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]));
        assert_eq!(s.of_list.get(&list), Some(&alloc));
        assert_eq!((s.made, s.swaps), (0, 1));
    }

    #[test]
    fn released_lists_retain_pending_gpu_fence_until_completion_then_allow_reuse() {
        let (alloc, list1, list2, created_spare) = (0x1000usize, 0x3001usize, 0x3002usize, 0x9000usize);
        let mut s = StateCore::<MockFence>::default();
        s.types.insert(alloc, 0);
        s.move_list(list1, alloc);
        s.move_list(list2, alloc);
        let fence = MockFence::default();
        s.backings.get_mut(&alloc).unwrap().marks.insert(1, (fence.clone(), 7));

        // Release first list: second list still holds backing.
        s.forget_list(list1);
        assert!(!s.backings[&alloc].free());

        // Release second list: pending fence still holds backing.
        s.forget_list(list2);
        assert!(s.backings[&alloc].users.is_empty());
        assert!(!s.backings[&alloc].free());

        // Resetting while fence is incomplete swaps to a newly created spare.
        let hr = s.title_reset_in(alloc, |_| 0, |_| Some(created_spare));
        assert_eq!(hr, Some(0));
        assert_eq!(s.current(alloc), created_spare);
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]));

        // Fence completes -> original backing becomes free and reusable on next Reset.
        fence.0.store(7, Ordering::Relaxed);
        assert!(s.backings[&alloc].free());
        s.move_list(list1, created_spare);
        let mut reset_called = Vec::new();
        let hr2 = s.title_reset_in(
            alloc,
            |raw| {
                reset_called.push(raw);
                0
            },
            |_| panic!("original backing is free and must be reused"),
        );
        assert_eq!(hr2, Some(0));
        assert_eq!(reset_called, vec![alloc]);
        assert_eq!(s.current(alloc), alloc);
    }

    #[test]
    fn resubmit_while_previous_submit_in_flight_is_detected() {
        let (list, queue) = (0x4000usize, 0x5000usize);
        let mut s = StateCore::<MockFence>::default();
        let fence = MockFence::default();
        s.recordings.insert(list, 1);
        s.last_submit.insert(list, LastSubmit { queue, fence: fence.clone(), value: 3, recording: 1 });
        assert_eq!(s.note_resubmits_in(queue, &[list]), 1, "in-flight resubmit counted");
        fence.0.store(3, Ordering::Relaxed);
        assert_eq!(s.note_resubmits_in(queue, &[list]), 0, "completed submit is clean");
    }
}
