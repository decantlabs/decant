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
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::core::{GUID, HRESULT};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_MAPPED, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use crate::list_sync::{CpuWrite, PendingWait};
use crate::log::xs_log;

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
/// `ID3D12CommandQueue::Wait` (PC queue vtable slot 15): the queue's own GPU timeline waits for
/// `fence` to reach `value` before running anything submitted to it after this call.
type QueueWait = unsafe extern "system" fn(Object, Object, u64) -> HRESULT;
/// `ID3D12Fence::Signal` (PC fence vtable slot 10), called from the CPU.
type FenceSignalCpu = unsafe extern "system" fn(Object, u64) -> HRESULT;

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
#[derive(Default)]
struct QueueSlot {
    /// The raw fence of the queue and the last value signalled on it. The fence lives as long as
    /// the process.
    fence: Option<(usize, u64)>,
    /// The calibration, taken the first time a timestamp write needs it.
    clock: Option<Clock>,
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

/// The writes of one submission, waiting for the GPU to reach `value` on `fence`.
struct Job {
    fence: usize,
    value: u64,
    writes: Vec<CpuWrite>,
    /// The clock of the queue, present when some write is a timestamp.
    clock: Option<Clock>,
}

impl Job {
    /// Waits for the GPU, then makes the writes.
    ///
    /// # Safety
    /// The fence is live and `event` is an event handle that no other thread uses.
    unsafe fn run(&self, event: HANDLE) {
        let fence = self.fence as Object;
        // SAFETY: by the contract above.
        unsafe {
            let completed: GetCompletedValue = vslot(fence, SLOT_GET_COMPLETED_VALUE);
            if completed(fence) < self.value {
                let arm: SetEventOnCompletion = vslot(fence, SLOT_SET_EVENT_ON_COMPLETION);
                if arm(fence, self.value, event) >= 0 {
                    WaitForSingleObject(event, SUBMISSION_WAIT_MS);
                }
            }
        }
        for (va, words) in &self.writes {
            if !words.is_empty() {
                // SAFETY: the target is checked against the title's memory before the store.
                unsafe { write_now(*va, words) };
            } else if let Some(clock) = self.clock {
                let tick = clock.gpu_at(counter_now());
                // SAFETY: as above.
                unsafe { write_now(*va, &[tick as u32, (tick >> 32) as u32]) };
            }
        }
    }
}

/// The sender of the worker thread. The thread starts on the first use.
fn worker() -> &'static Mutex<mpsc::Sender<Job>> {
    static WORKER: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    WORKER.get_or_init(|| {
        let (sender, jobs) = mpsc::channel::<Job>();
        let started = std::thread::Builder::new().name("decant-cpu-writes".into()).spawn(move || {
            // SAFETY: the event is made here and only this thread uses it.
            let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            decant_log::handles::created(event as usize, "CPU write fence event");
            for job in jobs {
                // SAFETY: `QUEUES` keeps the fences alive for the process.
                unsafe { job.run(event) };
            }
        });
        if let Err(e) = started {
            xs_log!("console proxies: no thread for the writes after the GPU: {e}");
        }
        Mutex::new(sender)
    })
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

/// Whether the CPU can write all of `[va, va + bytes)` now. A read only page counts as writable
/// when `xmem.dll` protected it to see the next write, which is how a clean page of the GPU
/// mirror looks.
pub(crate) fn writable(va: usize, bytes: usize) -> bool {
    let tracked_by_title = std::cell::OnceCell::new();
    all_regions(va, bytes, |r| {
        r.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0 || (r.Protect & PAGE_READONLY != 0 && *tracked_by_title.get_or_init(|| crate::title_mem::writable(va, bytes)))
    })
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

/// Checks `wait` again and again until the memory meets it or `timeout` is over. Returns whether
/// it was met.
pub(crate) fn wait_until(wait: &PendingWait, timeout: Duration) -> bool {
    let started = Instant::now();
    loop {
        if satisfied(wait) {
            return true;
        }
        let waited = started.elapsed();
        if waited >= timeout {
            return false;
        }
        std::thread::sleep(if waited < Duration::from_millis(10) { Duration::from_micros(100) } else { Duration::from_millis(1) });
    }
}

// ---- non-blocking deferral of unmet waits to a GPU fence (the one thread that submits every
// queue must never block synchronously on a wait the GPU itself would answer) ----

/// One queue's outstanding deferrals: its own wait fence, the next ticket to use, and the waits
/// still pending in submission order (a queue only ever grows its ticket from one thread, the
/// one that calls `queue_execute`, so FIFO order here matches submission order).
struct Deferred {
    fence: usize,
    ticket: u64,
    pending: std::collections::VecDeque<(u64, Vec<PendingWait>, Instant)>,
}

static WAIT_FENCES: Mutex<Option<HashMap<usize, Deferred>>> = Mutex::new(None);

/// Creates a fresh, zero-valued fence on `queue`'s device for deferred waits. `None` on any COM
/// failure (the caller falls back to the old synchronous wait).
///
/// # Safety
/// `queue` is a live PC command queue.
unsafe fn new_wait_fence(queue: Object) -> Option<usize> {
    static FAILS: AtomicU32 = AtomicU32::new(0);
    // SAFETY: `queue` is live by the contract above, and the slots are the ones named in the
    // module header (`GetDevice` 7, `CreateFence` 36).
    unsafe {
        let mut dev: Object = std::ptr::null_mut();
        let get: GetDevice = vslot(queue, SLOT_GET_DEVICE);
        let hr = get(queue, &IID_ID3D12_DEVICE, &mut dev);
        if hr < 0 || dev.is_null() {
            if FAILS.fetch_add(1, Ordering::Relaxed) < 5 {
                xs_log!("memory waits: GetDevice of the queue failed, hr {hr:#x}");
            }
            return None;
        }
        let mut fence: Object = std::ptr::null_mut();
        let create: CreateFence = vslot(dev, SLOT_CREATE_FENCE);
        let hr = create(dev, 0, 0, &IID_ID3D12_FENCE, &mut fence);
        let release: Release = vslot(dev, SLOT_RELEASE);
        release(dev);
        if hr < 0 || fence.is_null() {
            if FAILS.fetch_add(1, Ordering::Relaxed) < 5 {
                xs_log!("memory waits: CreateFence for a deferred wait failed, hr {hr:#x}");
            }
            return None;
        }
        Some(fence as usize)
    }
}

/// Pops every front entry of `pending` that is already met or past its deadline, in FIFO order,
/// and returns each popped ticket with whether it timed out rather than being met. An entry still
/// unmet and inside its deadline stops the scan, since later entries queued after it cannot be
/// due first. Pure logic, kept apart from the fence signalling so it can be unit tested on
/// synthetic input.
fn due_tickets<T>(pending: &mut std::collections::VecDeque<(u64, T, Instant)>, now: Instant, met: impl Fn(&T) -> bool) -> Vec<(u64, bool)> {
    let mut due = Vec::new();
    while let Some((_, waits, deadline)) = pending.front() {
        let is_met = met(waits);
        let expired = !is_met && now >= *deadline;
        if !is_met && !expired {
            break;
        }
        let (ticket, ..) = pending.pop_front().expect("front checked above");
        due.push((ticket, expired));
    }
    due
}

/// A watcher thread that advances every queue's deferred waits: the front one still unmet and
/// inside its deadline is left alone; a met or expired front one is cleared by signalling the
/// queue's own fence to its ticket, which releases the matching `ID3D12CommandQueue::Wait` on the
/// GPU without any CPU thread blocking on it. Checked every millisecond, same granularity as the
/// old synchronous poll (`wait_until`).
fn watcher() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("decant-memwait-watch".into()).spawn(|| loop {
            std::thread::sleep(Duration::from_millis(1));
            let mut g = WAIT_FENCES.lock().unwrap_or_else(|e| e.into_inner());
            let Some(m) = g.as_mut() else { continue };
            for state in m.values_mut() {
                let fence = state.fence as Object;
                for (ticket, expired) in due_tickets(&mut state.pending, Instant::now(), |waits| waits.iter().all(satisfied)) {
                    if expired {
                        static TIMED_OUT: AtomicU32 = AtomicU32::new(0);
                        if TIMED_OUT.fetch_add(1, Ordering::Relaxed) < 20 {
                            xs_log!("console list: a deferred memory wait timed out, the list runs anyway");
                        }
                    }
                    // SAFETY: the fence is kept alive by `WAIT_FENCES` for the life of the process.
                    unsafe {
                        let signal: FenceSignalCpu = vslot(fence, 10);
                        signal(fence, ticket);
                    }
                }
            }
        });
        if let Err(e) = spawned {
            xs_log!("console proxies: no thread to watch deferred memory waits: {e}");
        }
    });
}

/// Defers `waits` not already met on `queue`: an `ID3D12CommandQueue::Wait` is issued on `queue`
/// for a ticket on the queue's own wait fence, and a background thread signals that ticket once
/// the waits are met or `list_sync::WAIT_TIMEOUT` has passed, same timeout as the old synchronous
/// `wait_until` loop. Returns without blocking the calling thread (`queue_execute`'s one
/// submission thread, which must stay free to submit the producing queue's own list).
///
/// # Safety
/// `queue` is a live PC command queue.
pub(crate) unsafe fn defer_waits(queue: Object, waits: &[PendingWait]) {
    let unmet: Vec<PendingWait> = waits.iter().copied().filter(|w| !satisfied(w)).collect();
    if unmet.is_empty() {
        return;
    }
    watcher();
    let mut g = WAIT_FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    let state = match m.entry(queue as usize) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => {
            // SAFETY: `queue` is live by the contract above.
            let Some(fence) = (unsafe { new_wait_fence(queue) }) else {
                // no fence for this queue: fall back to the old synchronous wait so the title's
                // data is still correct even though this queue stalls the submission thread.
                drop(g);
                for w in &unmet {
                    wait_until(w, crate::list_sync::WAIT_TIMEOUT);
                }
                return;
            };
            v.insert(Deferred { fence, ticket: 0, pending: std::collections::VecDeque::new() })
        }
    };
    state.ticket += 1;
    let ticket = state.ticket;
    state.pending.push_back((ticket, unmet, Instant::now() + crate::list_sync::WAIT_TIMEOUT));
    let fence = state.fence as Object;
    drop(g);
    // SAFETY: `queue` is live by the contract above, `fence` is kept alive by `WAIT_FENCES`.
    unsafe {
        let wait: QueueWait = vslot(queue, 15);
        let hr = wait(queue, fence, ticket);
        if hr < 0 {
            static FAILS: AtomicU32 = AtomicU32::new(0);
            if FAILS.fetch_add(1, Ordering::Relaxed) < 5 {
                xs_log!("memory waits: ID3D12CommandQueue::Wait for a deferred wait failed, hr {hr:#x}");
            }
        }
    }
}

/// Releases every queue's still-pending deferred waits right away (called at present): a missing
/// or cross-frame-cyclic writer must never hang the GPU past the frame it was due in.
pub(crate) fn release_all_waits() {
    let mut g = WAIT_FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = g.as_mut() else { return };
    for state in m.values_mut() {
        if state.pending.is_empty() {
            continue;
        }
        let ticket = state.ticket;
        let fence = state.fence as Object;
        // SAFETY: the fence is kept alive by `WAIT_FENCES` for the life of the process.
        unsafe {
            let signal: FenceSignalCpu = vslot(fence, 10);
            signal(fence, ticket);
        }
        state.pending.clear();
    }
}

/// Stores `words` at `va` if that memory is still there and the CPU can write it.
///
/// # Safety
/// Nothing stops the title from unmapping `[va, va + 4 * words.len())` between the check and the
/// stores. The title owns that memory for as long as its submissions last, and `va` is dword aligned
/// (`list_sync` refuses other writes).
unsafe fn write_now(va: usize, words: &[u32]) {
    if !writable(va, words.len() * 4) {
        return;
    }
    for (index, &word) in words.iter().enumerate() {
        // SAFETY: checked above, alignment by the caller.
        unsafe { (*((va + index * 4) as *const AtomicU32)).store(word, Ordering::Release) };
    }
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
    let job = {
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
        let (fence, signalled) = slot.fence.as_mut().expect("the fence was made above");
        *signalled += 1;
        // SAFETY: the queue and the fence are live COM objects of one device.
        let hr = unsafe { vslot::<Signal>(queue, SLOT_SIGNAL)(queue, *fence as Object, *signalled) };
        if hr < 0 {
            log_failure("Signal of the queue", hr);
            return;
        }
        Job { fence: *fence, value: *signalled, writes, clock }
    };
    if ANNOUNCED.fetch_add(1, Ordering::Relaxed) < 3 {
        xs_log!("bottom-of-pipe writes outside the GPU's memory are done by the CPU after the GPU ({} in the first submission)", job.writes.len());
    }
    let _ = worker().lock().unwrap_or_else(|e| e.into_inner()).send(job);
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
        assert!(wait_until(&at_least_5, Duration::ZERO));
        assert!(!wait_until(&PendingWait { value: 6, ..at_least_5 }, Duration::from_millis(5)));
    }

    #[test]
    fn a_waiting_thread_sees_the_value_arrive() {
        let cell: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
        let wait = PendingWait { va: cell.as_ptr() as u64, width: 8, comparison: 3, value: 9, mask: u64::MAX };
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            cell.store(9, Ordering::Release);
        });
        assert!(wait_until(&wait, Duration::from_secs(5)));
        setter.join().unwrap();
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
    fn due_tickets_stops_at_the_first_unmet_unexpired_entry() {
        let now = Instant::now();
        let far_future = now + Duration::from_secs(60);
        let mut pending = std::collections::VecDeque::from([(1u64, false, far_future), (2u64, true, far_future)]);
        // The front ticket is neither met nor expired: nothing is due yet, even though a later
        // one queued after it is already met (FIFO order, same as submission order).
        assert_eq!(due_tickets(&mut pending, now, |met| *met), Vec::new());
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn due_tickets_pops_met_and_expired_fronts_in_order() {
        let now = Instant::now();
        let past = now - Duration::from_millis(1);
        let far_future = now + Duration::from_secs(60);
        // Ticket 1 is met, ticket 2 is unmet but past its deadline, ticket 3 is still pending.
        let mut pending = std::collections::VecDeque::from([(1u64, true, far_future), (2u64, false, past), (3u64, false, far_future)]);
        let due = due_tickets(&mut pending, now, |met| *met);
        assert_eq!(due, vec![(1, false), (2, true)], "ticket 1 met, ticket 2 timed out, ticket 3 left pending");
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.front().unwrap().0, 3);
    }

    #[test]
    fn due_tickets_on_an_empty_queue_is_a_no_op() {
        let mut pending: std::collections::VecDeque<(u64, bool, Instant)> = std::collections::VecDeque::new();
        assert_eq!(due_tickets(&mut pending, Instant::now(), |met| *met), Vec::new());
    }
}
