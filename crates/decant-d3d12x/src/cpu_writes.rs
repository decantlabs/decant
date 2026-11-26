//! The CPU side of the bottom of pipe writes into memory the GPU cannot reach (see `list_sync`).
//!
//! After a submission, the queue signals a fence that this module owns. A worker thread waits for
//! that value and then makes the recorded writes, in submission order. That is what "bottom of
//! pipe" can mean for a write that the GPU is unable to make: it happens after the GPU is done.
//!
//! The COM calls go through vtable slots of the public D3D12 headers, because this DLL is built
//! without the `windows` crate. The slots are `ID3D12DeviceChild::GetDevice` 7,
//! `ID3D12Device::CreateFence` 36, `ID3D12CommandQueue::Signal` 14, `ID3D12Fence::GetCompletedValue`
//! 8, `ID3D12Fence::SetEventOnCompletion` 9, `IUnknown::Release` 2,
//! `ID3D12CommandQueue::GetTimestampFrequency` 16 and `GetClockCalibration` 17.
//!
//! A timestamp write (no dwords, see `list_sync::CpuWrite`) stores the GPU clock as it is when the
//! CPU makes the write. Each queue is calibrated once, which pairs one GPU tick with one
//! `QueryPerformanceCounter` value, and the counter now is converted into GPU ticks. The result is a
//! little later than the moment the GPU reached the write and never earlier. The values advance at
//! the rate of the GPU clock frequency that the title asked for.
//!
//! The memory waits of the title (`list_sync::PendingWait`) are answered here too ([`wait_until`]).
//! Every set of writes wakes them, so a queue worker (`queue_fifo`) that waits for a value runs
//! its list as soon as the write it waits for is made.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, mpsc};
use std::time::{Duration, Instant};

use windows_sys::core::{GUID, HRESULT};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_MAPPED, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use crate::list_sync::{CpuWrite, PendingWait, WaitLimits};
use crate::log::xs_log;

/// Return value of `WaitForSingleObject` when the event was signalled.
const WAIT_OBJECT_0: u32 = 0;

/// How long a pending GPU fence wait counts as an active producer for memory waits before letting
/// other queues fall back to their idle timeout. A healthy GPU submission finishes far faster than
/// this, while a queue blocked on a cross-queue GPU fence stops pinning every wait to its hard
/// limit for the full `SUBMISSION_WAIT_MS`.
const ACTIVE_WAIT_MS: u32 = 500;

type Object = *mut c_void;

/// `IID_ID3D12Device` and `IID_ID3D12Fence` (d3d12.h).
const IID_ID3D12_DEVICE: GUID = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);
const IID_ID3D12_FENCE: GUID = GUID::from_u128(0x0a753dcf_c4d8_4b91_adf6_be5a60d95a76);

const SLOT_GET_DEVICE: usize = 7;
const SLOT_CREATE_FENCE: usize = 36;
const SLOT_SIGNAL: usize = 14;
const SLOT_GET_COMPLETED_VALUE: usize = 8;
const SLOT_SET_EVENT_ON_COMPLETION: usize = 9;
const SLOT_RELEASE: usize = 2;
const SLOT_GET_TIMESTAMP_FREQUENCY: usize = 16;
const SLOT_GET_CLOCK_CALIBRATION: usize = 17;

/// The longest the worker waits for a submission before it makes the writes anyway. That covers a
/// removed device or a queue that never gets to the signal.
const SUBMISSION_WAIT_MS: u32 = 10_000;
/// Short grace period used for backlogged jobs on a queue whose GPU fence already timed out for
/// [`SUBMISSION_WAIT_MS`] and still has not advanced a single tick. Avoids `N * 10s` cascading
/// stalls while still allowing immediate recovery the moment the fence moves.
const STALLED_FENCE_WAIT_MS: u32 = 200;

/// The function in vtable slot `slot` of `object`.
///
/// # Safety
/// `object` is a live COM object and `F` is the function type of the method in that slot.
unsafe fn vslot<F: Copy>(object: Object, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: a COM object starts with a pointer to its vtable.
    unsafe {
        let vtable = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*vtable.add(slot))
    }
}

type GetDevice = unsafe extern "system" fn(Object, *const GUID, *mut Object) -> HRESULT;
type CreateFence = unsafe extern "system" fn(Object, u64, u32, *const GUID, *mut Object) -> HRESULT;
type Signal = unsafe extern "system" fn(Object, Object, u64) -> HRESULT;
type GetCompletedValue = unsafe extern "system" fn(Object) -> u64;
type SetEventOnCompletion = unsafe extern "system" fn(Object, u64, HANDLE) -> HRESULT;
type Release = unsafe extern "system" fn(Object) -> u32;
type GetTimestampFrequency = unsafe extern "system" fn(Object, *mut u64) -> HRESULT;
type GetClockCalibration = unsafe extern "system" fn(Object, *mut u64, *mut u64) -> HRESULT;

/// A GPU tick and the performance counter value read together with it, plus both frequencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Clock {
    pub gpu: u64,
    pub cpu: u64,
    pub gpu_freq: u64,
    pub cpu_freq: u64,
}

impl Clock {
    /// The GPU tick that goes with the counter value `cpu_now`. It never goes below zero.
    pub fn gpu_at(&self, cpu_now: u64) -> u64 {
        if self.cpu_freq == 0 {
            return self.gpu;
        }
        let elapsed = i128::from(cpu_now) - i128::from(self.cpu);
        let gpu_elapsed = elapsed * i128::from(self.gpu_freq) / i128::from(self.cpu_freq);
        u64::try_from((i128::from(self.gpu) + gpu_elapsed).max(0)).unwrap_or(u64::MAX)
    }
}

/// What the module keeps for one queue.
struct QueueSlot {
    /// The raw fence of the queue and the last value signalled on it. The fence lives as long as
    /// the process.
    fence: Option<(usize, u64)>,
    /// The calibration, taken the first time a timestamp write needs it.
    clock: Option<Clock>,
    /// The sender for this queue's dedicated worker thread, so a slow or blocked fence on one
    /// queue never stalls CPU writes for another queue.
    sender: Option<mpsc::Sender<Job>>,
    /// Per-queue active-producer tracker shared with every job queued for this queue.
    activity: Arc<QueueActivity>,
}

impl Default for QueueSlot {
    fn default() -> Self {
        Self { fence: None, clock: None, sender: None, activity: Arc::new(QueueActivity::default()) }
    }
}

static QUEUES: Mutex<Option<HashMap<usize, QueueSlot>>> = Mutex::new(None);

fn queues() -> MutexGuard<'static, Option<HashMap<usize, QueueSlot>>> {
    QUEUES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reads the GPU clock of `queue` against the performance counter.
///
/// # Safety
/// `queue` is a live PC command queue.
unsafe fn calibrate(queue: Object) -> Option<Clock> {
    let (mut gpu_freq, mut gpu, mut cpu, mut cpu_freq) = (0u64, 0u64, 0u64, 0i64);
    // SAFETY: `queue` is live and the slots are the ones listed in the module header.
    unsafe {
        let get_freq: GetTimestampFrequency = vslot(queue, SLOT_GET_TIMESTAMP_FREQUENCY);
        let get_calibration: GetClockCalibration = vslot(queue, SLOT_GET_CLOCK_CALIBRATION);
        if get_freq(queue, &mut gpu_freq) < 0 || get_calibration(queue, &mut gpu, &mut cpu) < 0 {
            return None;
        }
        QueryPerformanceFrequency(&mut cpu_freq);
    }
    xs_log!("timestamp writes: GPU clock {gpu_freq} Hz, calibrated at GPU {gpu} and counter {cpu}");
    Some(Clock { gpu, cpu, gpu_freq, cpu_freq: u64::try_from(cpu_freq).unwrap_or(0) })
}

/// The performance counter now.
fn counter_now() -> u64 {
    let mut ticks = 0i64;
    // SAFETY: `ticks` is valid storage for the counter.
    unsafe { QueryPerformanceCounter(&mut ticks) };
    u64::try_from(ticks).unwrap_or(0)
}

/// State of one queue's pending bottom-of-pipe jobs for active-producer tracking.
#[derive(Default)]
struct ActivityState {
    /// Jobs enqueued onto this queue whose writes have not yet been applied.
    pending: usize,
    /// Whether this queue currently holds a `+1` claim on [`ACTIVE_QUEUES`].
    active: bool,
    /// Whether this queue's head job passed [`ACTIVE_WAIT_MS`] without its fence advancing. While
    /// set, backlogged jobs enqueued behind it do not re-claim [`ACTIVE_QUEUES`] until the fence
    /// advances or the queue drains back to idle.
    stalled: bool,
}

/// Per-queue active-producer tracker. Each queue contributes at most `1` to [`ACTIVE_QUEUES`], so
/// backlogged jobs sitting behind a stalled GPU fence never keep other queues pinned to the hard
/// memory wait timeout.
#[derive(Default)]
struct QueueActivity {
    state: Mutex<ActivityState>,
}

impl QueueActivity {
    fn lock(&self) -> MutexGuard<'_, ActivityState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers one newly enqueued job on this queue.
    fn enqueue(self: &Arc<Self>) -> JobActivityGuard {
        let mut st = self.lock();
        st.pending += 1;
        if !st.active && !st.stalled {
            st.active = true;
            ACTIVE_QUEUES.fetch_add(1, Ordering::AcqRel);
        }
        drop(st);
        JobActivityGuard { activity: Arc::clone(self), finished: false }
    }

    /// Marks this queue as stalled, releasing its single [`ACTIVE_QUEUES`] claim and waking waits.
    fn mark_stalled(&self) {
        let wake = {
            let mut st = self.lock();
            st.stalled = true;
            if st.active {
                st.active = false;
                ACTIVE_QUEUES.fetch_sub(1, Ordering::AcqRel);
                true
            } else {
                false
            }
        };
        if wake {
            wake_waits();
        }
    }

    /// Notes that this queue's GPU fence made progress again, restoring active tracking if further
    /// jobs remain queued behind the current one.
    fn mark_progressed(&self) {
        let mut st = self.lock();
        if st.stalled {
            st.stalled = false;
            if st.pending > 1 && !st.active {
                st.active = true;
                ACTIVE_QUEUES.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    /// Completes one job on this queue. Clears stalled state and releases [`ACTIVE_QUEUES`] once
    /// the queue drains back to zero pending jobs.
    fn finish_job(&self) {
        let wake = {
            let mut st = self.lock();
            st.pending = st.pending.saturating_sub(1);
            if st.pending == 0 {
                st.stalled = false;
                if st.active {
                    st.active = false;
                    ACTIVE_QUEUES.fetch_sub(1, Ordering::AcqRel);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if wake {
            wake_waits();
        }
    }

    fn is_stalled(&self) -> bool {
        self.lock().stalled
    }
}

/// Ensures every enqueued [`Job`] decrements its queue's pending count exactly once.
struct JobActivityGuard {
    activity: Arc<QueueActivity>,
    finished: bool,
}

impl JobActivityGuard {
    fn finish(&mut self) {
        if !self.finished {
            self.finished = true;
            self.activity.finish_job();
        }
    }
}

impl Drop for JobActivityGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Decides the total wait limit for a job whose fence currently reads `current` (`< target`),
/// given the completed value where the previous job on the same queue timed out (`stalled_at`).
fn fence_wait_limit_ms(current: u64, stalled_at: Option<u64>) -> (u32, bool) {
    if stalled_at == Some(current) {
        (STALLED_FENCE_WAIT_MS, true)
    } else {
        (SUBMISSION_WAIT_MS, false)
    }
}

/// The writes of one submission, waiting for the GPU to reach `value` on `fence`.
struct Job {
    queue: usize,
    fence: usize,
    value: u64,
    writes: Vec<CpuWrite>,
    /// The clock of the queue, present when some write is a timestamp.
    clock: Option<Clock>,
    guard: JobActivityGuard,
}

impl Job {
    /// Waits for the GPU, then makes the writes.
    ///
    /// # Safety
    /// The fence is live and `event` is an event handle that no other thread uses.
    unsafe fn run(mut self, event: HANDLE, stalled_fence_at: &mut Option<u64>) {
        let fence = self.fence as Object;
        // SAFETY: by the contract above.
        unsafe {
            let completed: GetCompletedValue = vslot(fence, SLOT_GET_COMPLETED_VALUE);
            let before = completed(fence);
            if before >= self.value {
                *stalled_fence_at = None;
                self.guard.activity.mark_progressed();
            } else {
                let (limit_ms, already_stalled) = fence_wait_limit_ms(before, *stalled_fence_at);
                if already_stalled {
                    self.guard.activity.mark_stalled();
                } else {
                    *stalled_fence_at = None;
                }
                let arm: SetEventOnCompletion = vslot(fence, SLOT_SET_EVENT_ON_COMPLETION);
                if arm(fence, self.value, event) >= 0 {
                    let started = Instant::now();
                    let active_limit = Duration::from_millis(u64::from(ACTIVE_WAIT_MS));
                    let total_limit = Duration::from_millis(u64::from(limit_ms));
                    let mut marked_stalled = already_stalled || self.guard.activity.is_stalled();
                    while completed(fence) < self.value {
                        let elapsed = started.elapsed();
                        if elapsed >= total_limit {
                            break;
                        }
                        if !marked_stalled && elapsed >= active_limit {
                            marked_stalled = true;
                            self.guard.activity.mark_stalled();
                        }
                        let remaining = if !marked_stalled {
                            active_limit.saturating_sub(elapsed).min(total_limit.saturating_sub(elapsed))
                        } else {
                            total_limit.saturating_sub(elapsed)
                        };
                        let ms = u32::try_from(remaining.as_millis()).unwrap_or(u32::MAX).max(1);
                        let waited = WaitForSingleObject(event, ms);
                        if waited == WAIT_OBJECT_0 && completed(fence) < self.value && arm(fence, self.value, event) < 0 {
                            break;
                        }
                    }
                }
                let after = completed(fence);
                if after >= self.value || after > before {
                    *stalled_fence_at = None;
                    self.guard.activity.mark_progressed();
                } else {
                    *stalled_fence_at = Some(after);
                    self.guard.activity.mark_stalled();
                    static STALL_LOGS: AtomicU32 = AtomicU32::new(0);
                    let n = STALL_LOGS.fetch_add(1, Ordering::Relaxed) + 1;
                    if n <= 8 || n.is_power_of_two() {
                        if let Some((map_done, map_pub, removed)) = crate::title_mem::map_fence_status() {
                            xs_log!(
                                "bottom-of-pipe writes: queue {:#x} fence stalled at {after} waiting for {} (map_fence {map_done}/{map_pub}, removed {removed:#x}), applying writes after {limit_ms} ms ({n} so far)",
                                self.queue,
                                self.value
                            );
                        } else {
                            xs_log!(
                                "bottom-of-pipe writes: queue {:#x} fence stalled at {after} waiting for {}, applying writes after {limit_ms} ms ({n} so far)",
                                self.queue,
                                self.value
                            );
                        }
                    }
                }
            }
        }
        apply_writes(&self.writes, self.clock);
        self.guard.finish();
    }
}

/// Makes `writes` now, in order, and wakes the memory waits ([`wait_until`]). A timestamp write
/// takes the GPU clock from `clock`, and is left out without one.
pub(crate) fn apply_writes(writes: &[CpuWrite], clock: Option<Clock>) {
    let mut cached_region: Option<(usize, usize)> = None;
    for (va, words) in writes {
        if !words.is_empty() {
            // SAFETY: the target is checked against the title's memory before the store.
            unsafe { write_now_cached(*va, words, &mut cached_region) };
        } else if let Some(clock) = clock {
            let tick = clock.gpu_at(counter_now());
            // SAFETY: as above.
            unsafe { write_now_cached(*va, &[tick as u32, (tick >> 32) as u32], &mut cached_region) };
        }
    }
    wake_waits();
}

/// Queues that currently have active, non-stalled bottom-of-pipe write jobs in flight.
static ACTIVE_QUEUES: AtomicUsize = AtomicUsize::new(0);

/// A count of the times writes were made, and the waits that sleep until it moves.
fn writes_made() -> &'static (Mutex<u64>, Condvar) {
    static MADE: OnceLock<(Mutex<u64>, Condvar)> = OnceLock::new();
    MADE.get_or_init(|| (Mutex::new(0), Condvar::new()))
}

fn writes_count() -> u64 {
    *writes_made().0.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wakes every memory wait, so it reads its value again at once.
pub(crate) fn wake_waits() {
    let (count, changed) = writes_made();
    *count.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    changed.notify_all();
}

/// Sleeps until writes are made after the count `seen`, for at most `timeout`.
fn sleep_until_written(seen: u64, timeout: Duration) {
    let (count, changed) = writes_made();
    let deadline = Instant::now() + timeout;
    let mut now_count = count.lock().unwrap_or_else(|e| e.into_inner());
    while *now_count == seen {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        now_count = changed.wait_timeout(now_count, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// Spawns one dedicated worker thread for a single queue, running `run` on each queued item in
/// order with its own fence event handle.
fn spawn_queue_worker<T: Send + 'static>(mut run: impl FnMut(T, HANDLE) + Send + 'static) -> Option<mpsc::Sender<T>> {
    let (sender, jobs) = mpsc::channel::<T>();
    let started = std::thread::Builder::new().name("decant-cpu-writes".into()).spawn(move || {
        // SAFETY: the event is made here and only this thread uses it.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        decant_log::handles::created(event as usize, "CPU write fence event");
        for job in jobs {
            run(job, event);
        }
        // SAFETY: `event` was created by this thread and is closed once on exit.
        unsafe { CloseHandle(event) };
    });
    match started {
        Ok(_) => Some(sender),
        Err(e) => {
            xs_log!("console proxies: no thread for the writes after the GPU: {e}");
            None
        }
    }
}

/// Applies a lazily committed pool page to `info`. A page of `xmem.dll` that the title has not
/// touched is only reserved in its view and the first access commits it (through the vectored
/// handler of `xmem.dll`). Such a page counts as committed with the protection of its view. In this
/// process only `xmem.dll` leaves pages of mapped views reserved.
pub(crate) fn lazily_committed(info: &mut MEMORY_BASIC_INFORMATION) {
    let untouched = info.State == MEM_RESERVE && info.Type == MEM_MAPPED && info.AllocationProtect != 0 && info.AllocationProtect & PAGE_NOACCESS == 0;
    if untouched {
        info.State = MEM_COMMIT;
        info.Protect = info.AllocationProtect;
    }
}

/// Whether all of `[va, va + bytes)` is committed and every region in it passes `accept`. The
/// dirty tracking of `xmem.dll` protects pages one by one, so one range of the title can be several
/// regions.
fn all_regions(va: usize, bytes: usize, mut accept: impl FnMut(&MEMORY_BASIC_INFORMATION) -> bool) -> bool {
    let Some(end) = va.checked_add(bytes.max(1)) else { return false };
    let mut at = va;
    while at < end {
        // SAFETY: a zeroed `MEMORY_BASIC_INFORMATION` is a valid value, it only holds integers.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid output buffer and its size is passed.
        let queried = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) } != 0;
        lazily_committed(&mut info);
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if !queried || info.State != MEM_COMMIT || !accept(&info) || region_end <= at {
            return false;
        }
        at = region_end;
    }
    true
}

/// Whether the CPU can write all of `[va, va + bytes)` now, reusing a cached writable region
/// `(base, end)` across consecutive writes in the same job.
fn writable_cached(va: usize, bytes: usize, cache: &mut Option<(usize, usize)>) -> bool {
    let Some(end) = va.checked_add(bytes.max(1)) else { return false };
    if let Some((c_lo, c_hi)) = *cache
        && va >= c_lo
        && end <= c_hi
    {
        return true;
    }
    let tracked_by_title = std::cell::OnceCell::new();
    let mut single_rw_region: Option<(usize, usize)> = None;
    let ok = all_regions(va, bytes, |r| {
        if r.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0 {
            let r_lo = r.BaseAddress as usize;
            let r_hi = r_lo.saturating_add(r.RegionSize);
            single_rw_region = match single_rw_region {
                None => Some((r_lo, r_hi)),
                Some((_, prev_hi)) if prev_hi == r_lo => Some((r_lo, r_hi)),
                Some(_) => None,
            };
            true
        } else {
            r.Protect & PAGE_READONLY != 0 && *tracked_by_title.get_or_init(|| crate::title_mem::writable(va, bytes))
        }
    });
    if ok && let Some(region) = single_rw_region {
        *cache = Some(region);
    }
    ok
}

/// Whether the CPU can write all of `[va, va + bytes)` now. A read only page counts as writable
/// when `xmem.dll` protected it to see the next write, which is how a clean page of the GPU
/// mirror looks.
pub(crate) fn writable(va: usize, bytes: usize) -> bool {
    let mut cache = None;
    writable_cached(va, bytes, &mut cache)
}

/// Whether the CPU can read all of `[va, va + bytes)` now.
pub(crate) fn readable(va: usize, bytes: usize) -> bool {
    all_regions(va, bytes, |r| r.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0)
}

/// The value of `width` bytes (4 or 8) at `va`, when that is readable title memory.
pub(crate) fn read_title(va: usize, width: usize) -> Option<u64> {
    if !matches!(width, 4 | 8) || !readable(va, width) {
        return None;
    }
    // SAFETY: the memory is committed and not guarded, and the caller passes an aligned address.
    unsafe {
        Some(if width == 4 { u64::from((*(va as *const AtomicU32)).load(Ordering::Acquire)) } else { (*(va as *const AtomicU64)).load(Ordering::Acquire) })
    }
}

/// Whether the memory now meets `wait`. Memory that cannot be read counts as meeting it, since the
/// CPU has no way to answer such a wait.
pub(crate) fn satisfied(wait: &PendingWait) -> bool {
    read_title(wait.va as usize, usize::from(wait.width)).is_none_or(|current| wait.satisfied_by(current))
}

/// The longest a memory wait sleeps before it reads its value again although no write of this
/// module was made: the title's own threads may write the value too, and they wake nobody.
const RECHECK: Duration = Duration::from_millis(2);

/// Waits until every memory wait in `waits` is met, sharing one `limits` budget across the whole
/// slice. Returns the waits that were still unmet when the wait gave up (either because
/// `interrupted()` asked the queue to advance, `limits.idle` passed with no active producer, or
/// `limits.hard` elapsed). Each write this module makes wakes the wait at once. Called by the
/// worker of a queue (`queue_fifo`), never on the title's thread.
pub(crate) fn wait_all_until(
    waits: &[PendingWait],
    limits: WaitLimits,
    elsewhere: impl Fn() -> bool,
    interrupted: impl Fn() -> bool,
) -> Vec<PendingWait> {
    let started = Instant::now();
    loop {
        let seen = writes_count();
        if waits.iter().all(satisfied) {
            return Vec::new();
        }
        if interrupted() {
            return waits.iter().copied().filter(|w| !satisfied(w)).collect();
        }
        let waited = started.elapsed();
        if waited >= limits.hard {
            return waits.iter().copied().filter(|w| !satisfied(w)).collect();
        }
        if waited >= limits.idle && ACTIVE_QUEUES.load(Ordering::Acquire) == 0 && !elsewhere() {
            return waits.iter().copied().filter(|w| !satisfied(w)).collect();
        }
        let until_idle = if waited < limits.idle { limits.idle - waited } else { RECHECK };
        sleep_until_written(seen, (limits.hard - waited).min(until_idle).min(RECHECK));
    }
}

/// Waits until the memory meets `wait`. Returns whether it was met. Convenience wrapper over
/// [`wait_all_until`] for a single wait without interruption.
#[cfg(test)]
pub(crate) fn wait_until(wait: &PendingWait, limits: WaitLimits, elsewhere: impl Fn() -> bool) -> bool {
    wait_all_until(&[*wait], limits, elsewhere, || false).is_empty()
}

/// Stores `words` at `va` if that memory is still there and the CPU can write it.
///
/// # Safety
/// Same safety contract as [`write_now`].
unsafe fn write_now_cached(va: usize, words: &[u32], cache: &mut Option<(usize, usize)>) {
    if !writable_cached(va, words.len() * 4, cache) {
        return;
    }
    for (index, &word) in words.iter().enumerate() {
        // SAFETY: checked above, alignment by the caller.
        unsafe { (*((va + index * 4) as *const AtomicU32)).store(word, Ordering::Release) };
    }
}

/// Stores `words` at `va` if that memory is still there and the CPU can write it.
///
/// # Safety
/// Nothing stops the title from unmapping `[va, va + 4 * words.len())` between the check and the
/// stores. The title owns that memory for as long as its submissions last, and `va` is dword aligned
/// (`list_sync` refuses other writes).
#[cfg(test)]
unsafe fn write_now(va: usize, words: &[u32]) {
    let mut cache = None;
    // SAFETY: forwarded from caller.
    unsafe { write_now_cached(va, words, &mut cache) };
}

/// How many failures of the bottom of pipe machinery are logged.
const FAILURE_LOGS: u32 = 5;

fn log_failure(what: &str, hr: HRESULT) {
    static LOGGED: AtomicU32 = AtomicU32::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < FAILURE_LOGS {
        xs_log!("bottom-of-pipe writes: {what} failed, hr {hr:#x}");
    }
}

/// Makes the fence for the device of `queue`.
///
/// # Safety
/// `queue` is a live PC command queue.
unsafe fn new_fence(queue: Object) -> Option<Object> {
    // SAFETY: `queue` is live and the slots are the ones listed in the module header.
    unsafe {
        let mut device: Object = std::ptr::null_mut();
        let get_device: GetDevice = vslot(queue, SLOT_GET_DEVICE);
        let hr = get_device(queue, &IID_ID3D12_DEVICE, &mut device);
        if hr < 0 || device.is_null() {
            log_failure("GetDevice of the queue", hr);
            return None;
        }
        let mut fence: Object = std::ptr::null_mut();
        let create: CreateFence = vslot(device, SLOT_CREATE_FENCE);
        let hr = create(device, 0, 0, &IID_ID3D12_FENCE, &mut fence);
        if hr < 0 {
            log_failure("CreateFence", hr);
        }
        // The fence keeps what it needs of the device, so the reference from GetDevice can go.
        let release: Release = vslot(device, SLOT_RELEASE);
        release(device);
        (hr >= 0 && !fence.is_null()).then_some(fence)
    }
}

/// Has the CPU make `writes` once the GPU is done with the submission just made to `queue`, a PC
/// `ID3D12CommandQueue`.
///
/// # Safety
/// `queue` is a live PC command queue.
pub(crate) unsafe fn after(queue: Object, writes: Vec<CpuWrite>) {
    static ANNOUNCED: AtomicU32 = AtomicU32::new(0);
    if queue.is_null() || writes.is_empty() {
        return;
    }
    let wants_clock = writes.iter().any(|(_, words)| words.is_empty());
    let (sender, job) = {
        let mut guard = queues();
        let slot = guard.get_or_insert_with(HashMap::new).entry(queue as usize).or_default();
        let clock = if !wants_clock {
            None
        } else {
            if slot.clock.is_none() {
                // SAFETY: `queue` is live by the contract above.
                slot.clock = unsafe { calibrate(queue) };
            }
            slot.clock
        };
        if slot.fence.is_none() {
            // SAFETY: `queue` is live by the contract above.
            let Some(fence) = (unsafe { new_fence(queue) }) else { return };
            slot.fence = Some((fence as usize, 0));
        }
        if slot.sender.is_none() {
            let mut stalled_fence_at: Option<u64> = None;
            slot.sender = spawn_queue_worker(move |job: Job, event| {
                // SAFETY: `QUEUES` keeps the fences alive for the process.
                unsafe { job.run(event, &mut stalled_fence_at) };
            });
        }
        let Some(sender) = slot.sender.clone() else {
            // Without a worker thread the title would never see these writes; apply them now so
            // memory waits do not starve.
            apply_writes(&writes, clock);
            return;
        };
        let (fence, signalled) = slot.fence.as_mut().expect("the fence was made above");
        *signalled += 1;
        // SAFETY: the queue and the fence are live COM objects of one device.
        let hr = unsafe { vslot::<Signal>(queue, SLOT_SIGNAL)(queue, *fence as Object, *signalled) };
        if hr < 0 {
            log_failure("Signal of the queue", hr);
            return;
        }
        let guard = slot.activity.enqueue();
        let job = Job { queue: queue as usize, fence: *fence, value: *signalled, writes, clock, guard };
        (sender, job)
    };
    if ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 3 {
        xs_log!("bottom-of-pipe writes outside the GPU's memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len());
    }
    let _ = sender.send(job);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_store_reaches_committed_memory() {
        let target = Box::new([0u32; 3]);
        let va = target.as_ptr() as usize;
        assert!(writable(va, 12));
        // SAFETY: `va` points into the live, dword aligned box.
        unsafe { write_now(va + 4, &[0xAB, 0xCD]) };
        assert_eq!(*target, [0, 0xAB, 0xCD]);
    }

    #[test]
    fn an_unmapped_address_is_refused() {
        assert!(!writable(0, 8));
        assert!(!readable(0, 4));
        assert!(!writable(usize::MAX - 2, 8), "the range wraps around");
        // SAFETY: the address is refused before anything is stored.
        unsafe { write_now(0, &[1]) };
    }

    #[test]
    fn titles_values_are_read_by_width() {
        let cell = Box::new(0x1122_3344_5566_7788u64);
        let va = &*cell as *const u64 as usize;
        assert_eq!(read_title(va, 8), Some(0x1122_3344_5566_7788));
        assert_eq!(read_title(va, 4), Some(0x5566_7788));
        assert_eq!(read_title(va, 2), None);
        assert_eq!(read_title(0, 4), None);
    }

    #[test]
    fn a_wait_is_met_by_memory_or_unreadable() {
        let cell = Box::new(5u64);
        let va = &*cell as *const u64 as u64;
        let at_least_5 = PendingWait { va, width: 8, comparison: 7, value: 5, mask: u64::MAX };
        assert!(satisfied(&at_least_5));
        assert!(!satisfied(&PendingWait { value: 6, ..at_least_5 }));
        assert!(satisfied(&PendingWait { va: 0, ..at_least_5 }), "nothing the CPU could wait for");
        let no_time = WaitLimits { idle: Duration::ZERO, hard: Duration::ZERO };
        assert!(wait_until(&at_least_5, no_time, || false));
        let short = WaitLimits { idle: Duration::from_millis(5), hard: Duration::from_millis(5) };
        assert!(!wait_until(&PendingWait { value: 6, ..at_least_5 }, short, || true));
    }

    #[test]
    fn a_waiting_thread_sees_a_value_the_title_stores() {
        let cell: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
        let wait = PendingWait { va: cell.as_ptr() as u64, width: 8, comparison: 3, value: 9, mask: u64::MAX };
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            cell.store(9, Ordering::Release);
        });
        let limits = WaitLimits { idle: Duration::from_secs(5), hard: Duration::from_secs(5) };
        assert!(wait_until(&wait, limits, || false), "a store that wakes nobody is seen at the next check");
        setter.join().unwrap();
    }

    #[test]
    fn a_write_of_this_module_wakes_the_wait_at_once() {
        let cell: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
        let va = cell.as_ptr() as usize;
        let wait = PendingWait { va: va as u64, width: 8, comparison: 7, value: 3, mask: u64::MAX };
        let writer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let at = Instant::now();
            apply_writes(&[(va, vec![3, 0])], None);
            at
        });
        let limits = WaitLimits { idle: Duration::from_secs(5), hard: Duration::from_secs(5) };
        assert!(wait_until(&wait, limits, || false));
        let woke = Instant::now();
        let written = writer.join().unwrap();
        let latency = woke.saturating_duration_since(written);
        assert!(latency < Duration::from_millis(50), "woke {latency:?} after the write");
    }

    #[test]
    fn a_wait_gives_up_early_only_when_no_producer_can_come() {
        let cell = Box::new(0u64);
        let wait = PendingWait { va: &*cell as *const u64 as u64, width: 8, comparison: 7, value: 1, mask: u64::MAX };
        let limits = WaitLimits { idle: Duration::from_millis(10), hard: Duration::from_millis(300) };
        let begun = Instant::now();
        assert!(!wait_until(&wait, limits, || false));
        let idle_end = begun.elapsed();
        assert!(idle_end < Duration::from_millis(250), "nothing can write it, so the idle limit ends the wait ({idle_end:?})");
        let begun = Instant::now();
        assert!(!wait_until(&wait, limits, || true));
        assert!(begun.elapsed() >= Duration::from_millis(300), "work elsewhere may still write it, so the wait runs to its hard limit");
    }

    #[test]
    fn an_untouched_xmem_page_counts_as_committed() {
        // SAFETY: a zeroed value of plain integers.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        info.State = MEM_RESERVE;
        info.Type = MEM_MAPPED;
        info.AllocationProtect = PAGE_READWRITE;
        lazily_committed(&mut info);
        assert_eq!((info.State, info.Protect), (MEM_COMMIT, PAGE_READWRITE));
        info.State = MEM_RESERVE;
        info.AllocationProtect = PAGE_NOACCESS;
        lazily_committed(&mut info);
        assert_eq!(info.State, MEM_RESERVE, "no access views stay reserved");
    }

    #[test]
    fn clock_ticks_follow_the_counter() {
        let clock = Clock { gpu: 5_000, cpu: 70_000, gpu_freq: 25_000_000, cpu_freq: 10_000_000 };
        assert_eq!(clock.gpu_at(70_000), 5_000);
        assert_eq!(clock.gpu_at(80_000), 5_000 + 25_000, "one millisecond on");
        assert_eq!(clock.gpu_at(0), 0, "no value below zero");
        assert_eq!(Clock { cpu_freq: 0, ..clock }.gpu_at(123), 5_000, "no counter frequency, no movement");
    }

    #[test]
    fn writable_cached_reuses_region_bounds_across_writes() {
        let target = Box::new([0u32; 4]);
        let va = target.as_ptr() as usize;
        let mut cache = None;
        assert!(writable_cached(va, 8, &mut cache));
        let (lo, hi) = cache.expect("readwrite region cached");
        assert!(va >= lo && va + 16 <= hi);
        // SAFETY: `va` points into the live, dword aligned box.
        unsafe { write_now_cached(va + 8, &[0x12, 0x34], &mut cache) };
        assert_eq!(*target, [0, 0, 0x12, 0x34]);
    }

    #[test]
    fn per_queue_workers_do_not_head_of_line_block_each_other() {
        let (block_tx, block_rx) = mpsc::channel::<()>();
        let (started_tx, started_rx) = mpsc::channel::<()>();
        let slow_queue = spawn_queue_worker(move |(): (), _| {
            let _ = started_tx.send(());
            let _ = block_rx.recv();
        })
        .expect("slow worker spawns");
        let fast_queue = spawn_queue_worker(|(va, val): (usize, u32), _| {
            apply_writes(&[(va, vec![val, 0])], None);
        })
        .expect("fast worker spawns");

        slow_queue.send(()).unwrap();
        started_rx.recv_timeout(Duration::from_secs(5)).expect("slow worker entered blocked job");

        let cell = Box::new(AtomicU64::new(0));
        let va = std::ptr::from_ref(&*cell) as usize;
        let wait = PendingWait { va: va as u64, width: 8, comparison: 7, value: 42, mask: u64::MAX };
        let begun = Instant::now();
        fast_queue.send((va, 42)).unwrap();
        assert!(wait_until(&wait, WaitLimits { idle: Duration::from_secs(2), hard: Duration::from_secs(5) }, || true));
        assert!(begun.elapsed() < Duration::from_millis(200), "fast queue completed while slow queue stayed blocked ({:?})", begun.elapsed());

        let _ = block_tx.send(());
        drop(slow_queue);
        drop(fast_queue);
    }

    #[test]
    fn wait_all_until_shares_budget_and_honours_interrupt() {
        let a = Box::new(AtomicU64::new(0));
        let b = Box::new(AtomicU64::new(0));
        let c = Box::new(AtomicU64::new(0));
        let waits = [
            PendingWait { va: std::ptr::from_ref(&*a) as u64, width: 8, comparison: 7, value: 1, mask: u64::MAX },
            PendingWait { va: std::ptr::from_ref(&*b) as u64, width: 8, comparison: 7, value: 1, mask: u64::MAX },
            PendingWait { va: std::ptr::from_ref(&*c) as u64, width: 8, comparison: 7, value: 1, mask: u64::MAX },
        ];
        // Three unmet waits share one idle budget instead of waiting 3x.
        let begun = Instant::now();
        let unmet = wait_all_until(&waits, WaitLimits { idle: Duration::from_millis(30), hard: Duration::from_secs(2) }, || false, || false);
        assert_eq!(unmet.len(), 3);
        assert!(begun.elapsed() < Duration::from_millis(250), "shared budget ended in {:?}", begun.elapsed());

        // An interrupt + wake_waits exits immediately without waiting for idle or hard limits.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = std::sync::Arc::clone(&stop);
        let kicker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            stop_flag.store(true, Ordering::Release);
            wake_waits();
        });
        let begun = Instant::now();
        let unmet = wait_all_until(
            &waits,
            WaitLimits { idle: Duration::from_secs(5), hard: Duration::from_secs(10) },
            || true,
            || stop.load(Ordering::Acquire),
        );
        kicker.join().unwrap();
        assert_eq!(unmet.len(), 3);
        assert!(begun.elapsed() < Duration::from_millis(250), "interrupt ended in {:?}", begun.elapsed());
    }

    #[test]
    fn stalled_queue_with_backlogged_jobs_drops_active_producers_to_zero() {
        let queue = Arc::new(QueueActivity::default());
        let cell = Box::new(AtomicU64::new(0));
        let wait = PendingWait { va: std::ptr::from_ref(&*cell) as u64, width: 8, comparison: 7, value: 1, mask: u64::MAX };

        // Enqueue 20 backlogged jobs on one queue: ACTIVE_QUEUES only rises by 1, not 20.
        let mut guards: Vec<JobActivityGuard> = (0..20).map(|_| queue.enqueue()).collect();
        let baseline = ACTIVE_QUEUES.load(Ordering::Acquire);
        assert!(baseline >= 1);

        // Once the head job marks the queue stalled, ACTIVE_QUEUES drops cleanly even while 19
        // backlogged jobs remain and even when more jobs are enqueued behind them.
        let queue_bg = Arc::clone(&queue);
        let staller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(35));
            queue_bg.mark_stalled();
            let extra = queue_bg.enqueue();
            std::thread::sleep(Duration::from_millis(10));
            drop(extra);
        });
        let begun = Instant::now();
        let unmet = wait_all_until(&[wait], WaitLimits { idle: Duration::from_millis(20), hard: Duration::from_secs(2) }, || false, || false);
        staller.join().unwrap();
        assert_eq!(unmet.len(), 1);
        assert!(begun.elapsed() < Duration::from_millis(400), "backlogged stalled queue must not pin waits to hard limit ({:?})", begun.elapsed());

        // Progress restores active tracking while remaining jobs exist, and draining releases it.
        queue.mark_progressed();
        assert_eq!(ACTIVE_QUEUES.load(Ordering::Acquire), baseline);
        for g in &mut guards {
            g.finish();
        }
        assert_eq!(ACTIVE_QUEUES.load(Ordering::Acquire), baseline - 1);
    }

    #[test]
    fn stalled_fence_wait_limit_shortens_until_fence_advances() {
        assert_eq!(fence_wait_limit_ms(10, None), (SUBMISSION_WAIT_MS, false));
        assert_eq!(fence_wait_limit_ms(10, Some(10)), (STALLED_FENCE_WAIT_MS, true));
        assert_eq!(fence_wait_limit_ms(11, Some(10)), (SUBMISSION_WAIT_MS, false));
    }
}
