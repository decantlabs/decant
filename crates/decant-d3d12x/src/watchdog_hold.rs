//! Keeps a title's own hang watchdog quiet while this layer holds a title thread in a shader or
//! pipeline wait, or in a PC D3D12 call that costs no time on the console.
//!
//! Some titles run a stall check thread that ticks with `Sleep` and deliberately crashes the
//! process when no frame was presented for about 30 seconds. On a cold start a title thread can
//! wait on pipeline compilation or on slow driver calls (`Create*PipelineState`,
//! `ExecuteCommandLists`, placed resource creation) far longer than it ever would on the console.
//!
//! Two things count as "the title is held" (see [`title_held`]):
//! * a title thread is inside a pipeline bind wait ([`hold_bind`]),
//! * a PC D3D12 call is running ([`pc_call`]) while the PC call time spent since the last present
//!   ([`note_present`]) has reached [`PC_CALL_HOLDS_AFTER`].
//!
//! On Windows [`install`] patches the title executable's import table entries for `Sleep` and
//! `SleepEx` (from `kernel32.dll`, `kernelbase.dll` or `api-ms-win-core-synch-*.dll`). A sleep of
//! at least `DECANT_D3D12X_SLEEP_HOLD` milliseconds (default 100, `0` turns the feature off) stops
//! counting down while the title is held. Short sleeps are polls, so they pass through, as do
//! zero sleeps, `INFINITE` and alertable `SleepEx` calls.

#![allow(unsafe_code)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Longest single step of a held sleep, so time that was held only part of a step is not lost.
const SLEEP_STEP_MS: u32 = 20;
/// Poll interval of a sleep that is paused because the title is held.
const SLEEP_HELD_POLL_MS: u32 = 5;
/// Shortest sleep that is paused while held when the setting is not given.
const SLEEP_HOLD_MIN_MS: u32 = 100;
/// PC call time per frame after which a running PC call holds the title.
const PC_CALL_HOLDS_AFTER: Duration = Duration::from_millis(250);

/// Number of title threads currently waiting for a pipeline build.
static BIND_WAITERS: AtomicUsize = AtomicUsize::new(0);
/// Number of frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Book keeping of PC D3D12 calls: the ones running and the time of the finished ones this frame.
struct CallLedger {
    running: Vec<(u64, Instant)>,
    frame: u64,
    finished_time: Duration,
}

impl CallLedger {
    const fn new() -> Self {
        Self { running: Vec::new(), frame: 0, finished_time: Duration::ZERO }
    }

    /// Starts a fresh frame account when `frame` differs from the one being tracked.
    fn roll_to(&mut self, frame: u64) {
        if self.frame != frame {
            self.frame = frame;
            self.finished_time = Duration::ZERO;
        }
    }

    fn start(&mut self, id: u64, at: Instant) {
        self.running.push((id, at));
    }

    fn finish(&mut self, id: u64, frame: u64, at: Instant) {
        self.roll_to(frame);
        if let Some(pos) = self.running.iter().position(|&(known, _)| known == id) {
            let (_, began) = self.running.swap_remove(pos);
            self.finished_time += at.saturating_duration_since(began);
        }
    }

    /// True when some running call, together with the finished calls of this frame, has used up
    /// the allowance.
    fn is_holding(&mut self, frame: u64, at: Instant) -> bool {
        self.roll_to(frame);
        let banked = self.finished_time;
        self.running.iter().any(|&(_, began)| banked + at.saturating_duration_since(began) >= PC_CALL_HOLDS_AFTER)
    }
}

static LEDGER: Mutex<CallLedger> = Mutex::new(CallLedger::new());

fn ledger() -> std::sync::MutexGuard<'static, CallLedger> {
    LEDGER.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Records that the title presented a frame, which starts a new PC call account.
pub(crate) fn note_present() {
    FRAMES.fetch_add(1, Ordering::Release);
}

/// Whether a title thread is held in a pipeline bind wait or a long PC D3D12 call right now.
pub(crate) fn title_held() -> bool {
    BIND_WAITERS.load(Ordering::Acquire) != 0 || ledger().is_holding(FRAMES.load(Ordering::Acquire), Instant::now())
}

/// Guard that counts a title thread as waiting for a pipeline build while it lives.
pub(crate) struct Hold(&'static AtomicUsize);

/// Counts the calling thread as waiting for a pipeline build until the guard drops.
pub(crate) fn hold_bind() -> Hold {
    BIND_WAITERS.fetch_add(1, Ordering::AcqRel);
    Hold(&BIND_WAITERS)
}

impl Drop for Hold {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Guard that times a PC D3D12 call which would return at once on the console.
pub(crate) struct PcCall(u64);

/// Starts timing a PC D3D12 call, which ends when the guard drops.
pub(crate) fn pc_call() -> PcCall {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    ledger().start(id, Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        ledger().finish(self.0, FRAMES.load(Ordering::Acquire), Instant::now());
    }
}

/// Reads the `DECANT_D3D12X_SLEEP_HOLD` setting. Unset or unparsable gives the default, `0` gives
/// `None`, which means the feature is off.
fn sleep_hold_min(setting: Option<&str>) -> Option<u32> {
    let Some(text) = setting.map(str::trim) else {
        return Some(SLEEP_HOLD_MIN_MS);
    };
    match text.parse::<u32>() {
        Ok(0) => None,
        Ok(ms) => Some(ms),
        Err(_) => Some(SLEEP_HOLD_MIN_MS),
    }
}

/// Sleeps for `ms` of time in which the title was not held. `sleep` does the actual waiting and
/// `held` says whether the title is held at the moment.
fn title_sleep(ms: u32, min: u32, mut sleep: impl FnMut(u32), mut held: impl FnMut() -> bool) {
    let passes_through = ms == 0 || ms == u32::MAX || ms < min;
    if passes_through {
        sleep(ms);
        return;
    }
    let mut remaining = Duration::from_millis(u64::from(ms));
    while !remaining.is_zero() {
        if held() {
            sleep(SLEEP_HELD_POLL_MS);
            continue;
        }
        let step_ms = u32::try_from(remaining.as_millis()).unwrap_or(u32::MAX).clamp(1, SLEEP_STEP_MS);
        let begun = Instant::now();
        sleep(step_ms);
        remaining = remaining.saturating_sub(begun.elapsed());
    }
}

/// Whether an import descriptor naming this DLL can carry `Sleep` or `SleepEx`.
fn sleep_dll(dll: &str) -> bool {
    let name = dll.to_ascii_lowercase();
    matches!(name.as_str(), "kernel32.dll" | "kernelbase.dll") || (name.starts_with("api-ms-win-core-synch-") && name.ends_with(".dll"))
}

/// Slot of an imported function in the wrapper tables: 0 for `Sleep`, 1 for `SleepEx`.
fn sleep_slot(function: &str) -> Option<usize> {
    match function {
        "Sleep" => Some(0),
        "SleepEx" => Some(1),
        _ => None,
    }
}

#[cfg(windows)]
mod iat {
    use super::*;
    use crate::log::xs_log;
    use std::sync::{Once, OnceLock};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};

    type SleepFn = unsafe extern "system" fn(u32);
    type SleepExFn = unsafe extern "system" fn(u32, i32) -> u32;

    /// The real `Sleep` and `SleepEx`, as found in the import table before patching.
    static REAL: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];
    static PATCH_ONCE: Once = Once::new();

    fn min_sleep() -> Option<u32> {
        static SETTING: OnceLock<Option<u32>> = OnceLock::new();
        *SETTING.get_or_init(|| sleep_hold_min(std::env::var("DECANT_D3D12X_SLEEP_HOLD").ok().as_deref()))
    }

    unsafe extern "system" fn hooked_sleep(ms: u32) {
        // SAFETY: slot 0 only holds the address of the real `Sleep`, stored before the patch.
        let real: SleepFn = unsafe { std::mem::transmute(REAL[0].load(Ordering::Acquire)) };
        match min_sleep() {
            // SAFETY: calling the real function with the caller's argument.
            None => unsafe { real(ms) },
            // SAFETY: as above.
            Some(min) => title_sleep(ms, min, |part| unsafe { real(part) }, title_held),
        }
    }

    unsafe extern "system" fn hooked_sleep_ex(ms: u32, alertable: i32) -> u32 {
        // SAFETY: slot 1 only holds the address of the real `SleepEx`, stored before the patch.
        let real: SleepExFn = unsafe { std::mem::transmute(REAL[1].load(Ordering::Acquire)) };
        let Some(min) = min_sleep().filter(|_| alertable == 0) else {
            // SAFETY: calling the real function with the caller's arguments.
            return unsafe { real(ms, alertable) };
        };
        title_sleep(
            ms,
            min,
            |part| {
                // SAFETY: as above, never alertable here.
                unsafe { real(part, 0) };
            },
            title_held,
        );
        0
    }

    fn hook_address(slot: usize) -> usize {
        match slot {
            0 => hooked_sleep as *const () as usize,
            _ => hooked_sleep_ex as *const () as usize,
        }
    }

    /// Reads a NUL terminated name of at most 256 bytes.
    ///
    /// # Safety
    /// `at` is readable up to its terminator or 256 bytes.
    unsafe fn name_at(at: *const u8) -> Option<String> {
        let len = (0..256).find(|&i| unsafe { *at.add(i) } == 0).unwrap_or(256);
        String::from_utf8(unsafe { std::slice::from_raw_parts(at, len) }.to_vec()).ok()
    }

    /// Points one import address table entry at our hook and remembers the real function.
    ///
    /// # Safety
    /// `entry` is an 8 byte entry of the loaded executable's import address table.
    unsafe fn redirect(entry: *mut usize, slot: usize) -> bool {
        let current = unsafe { entry.read_unaligned() };
        if current == hook_address(slot) || current <= u32::MAX as usize {
            return false;
        }
        let mut previous = 0u32;
        if unsafe { VirtualProtect(entry.cast(), 8, PAGE_READWRITE, &mut previous) } == 0 {
            return false;
        }
        let _ = REAL[slot].compare_exchange(0, current, Ordering::AcqRel, Ordering::Acquire);
        unsafe {
            entry.write_unaligned(hook_address(slot));
            VirtualProtect(entry.cast(), 8, previous, &mut previous);
        }
        true
    }

    /// Walks the import descriptors of the executable at `base` and patches every `Sleep` and
    /// `SleepEx` entry. Returns how many entries changed.
    ///
    /// # Safety
    /// `base` is the base address of a loaded module.
    unsafe fn patch_imports(base: usize) -> usize {
        let at = |rva: u32| (base + rva as usize) as *const u8;
        let mut patched = 0;
        unsafe {
            if at(0).cast::<u16>().read_unaligned() != 0x5A4D {
                return 0;
            }
            let nt = at(at(0x3C).cast::<u32>().read_unaligned());
            let is_pe64 = nt.cast::<u32>().read_unaligned() == 0x0000_4550 && nt.add(0x18).cast::<u16>().read_unaligned() == 0x20B;
            if !is_pe64 {
                return 0;
            }
            let table = nt.add(0x90).cast::<u32>().read_unaligned();
            if table == 0 {
                return 0;
            }
            // Each descriptor is 20 bytes: lookup table at 0, DLL name at 12, address table at 16.
            let mut descriptor = at(table);
            loop {
                let lookups = descriptor.cast::<u32>().read_unaligned();
                let dll_name = descriptor.add(12).cast::<u32>().read_unaligned();
                let addresses = descriptor.add(16).cast::<u32>().read_unaligned();
                descriptor = descriptor.add(20);
                if dll_name == 0 {
                    break;
                }
                if lookups == 0 || !name_at(at(dll_name)).is_some_and(|dll| sleep_dll(&dll)) {
                    continue;
                }
                for index in 0.. {
                    let lookup = at(lookups).add(index * 8).cast::<u64>().read_unaligned();
                    if lookup == 0 {
                        break;
                    }
                    let by_name = lookup >> 63 == 0;
                    if !by_name {
                        continue;
                    }
                    // A by name lookup points at a 2 byte hint followed by the name.
                    let Some(slot) = name_at(at(lookup as u32 + 2)).as_deref().and_then(sleep_slot) else {
                        continue;
                    };
                    if redirect(at(addresses).add(index * 8) as *mut usize, slot) {
                        patched += 1;
                    }
                }
            }
        }
        patched
    }

    /// Patches the title executable's `Sleep` and `SleepEx` imports, once per process.
    pub(crate) fn install() {
        let Some(min) = min_sleep() else {
            return;
        };
        PATCH_ONCE.call_once(|| {
            // SAFETY: a null name asks for the executable of the host process, not this DLL.
            let base = unsafe { GetModuleHandleW(std::ptr::null()) } as usize;
            if base == 0 {
                return;
            }
            // SAFETY: `base` is the loaded host executable.
            let patched = unsafe { patch_imports(base) };
            if patched > 0 {
                xs_log!("watchdog hold: {patched} Sleep/SleepEx import(s) pause sleeps >= {min} ms while title threads wait on shaders or long PC calls");
            }
        });
    }
}

#[cfg(windows)]
pub(crate) use iat::install;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_running_call_holds_only_after_the_allowance() {
        let mut ledger = CallLedger::new();
        let t0 = Instant::now();
        ledger.start(1, t0);
        assert!(!ledger.is_holding(0, t0));
        assert!(!ledger.is_holding(0, t0 + PC_CALL_HOLDS_AFTER / 2));
        assert!(ledger.is_holding(0, t0 + PC_CALL_HOLDS_AFTER));
        ledger.finish(1, 0, t0 + PC_CALL_HOLDS_AFTER);
        assert!(!ledger.is_holding(0, t0 + PC_CALL_HOLDS_AFTER), "nothing is running any more");
    }

    #[test]
    fn finished_calls_of_a_frame_add_up_until_the_next_present() {
        let mut ledger = CallLedger::new();
        let mut now = Instant::now();
        let slice = PC_CALL_HOLDS_AFTER / 5;
        for id in 0..6 {
            ledger.start(id, now);
            now += slice;
            ledger.finish(id, 7, now);
        }
        ledger.start(100, now);
        assert!(ledger.is_holding(7, now), "six slices already used up the allowance");
        assert!(!ledger.is_holding(8, now), "a new frame starts with an empty account");
    }

    #[test]
    fn finishing_an_unknown_call_changes_nothing() {
        let mut ledger = CallLedger::new();
        let now = Instant::now();
        ledger.finish(42, 0, now + PC_CALL_HOLDS_AFTER * 4);
        ledger.start(1, now);
        assert!(!ledger.is_holding(0, now));
    }

    #[test]
    fn bind_guards_count_waiting_threads() {
        let first = hold_bind();
        let second = hold_bind();
        assert!(BIND_WAITERS.load(Ordering::Acquire) >= 2);
        assert!(title_held());
        drop(first);
        drop(second);
    }

    #[test]
    fn a_held_sleep_waits_for_release_then_counts_down() {
        let mut calls = 0u32;
        let mut slept = 0u32;
        let mut polls = 0u32;
        title_sleep(
            40,
            10,
            |ms| {
                calls += 1;
                slept += ms;
                std::thread::sleep(Duration::from_millis(u64::from(ms)));
            },
            || {
                polls += 1;
                polls <= 3
            },
        );
        assert!(calls >= 5, "three held polls plus real steps, got {calls}");
        assert!(slept >= 3 * SLEEP_HELD_POLL_MS + 40 - SLEEP_STEP_MS, "slept {slept}");
    }

    #[test]
    fn sleeps_that_are_short_zero_or_infinite_pass_through() {
        let mut seen = Vec::new();
        title_sleep(15, 100, |ms| seen.push(ms), || true);
        title_sleep(0, 100, |ms| seen.push(ms), || true);
        title_sleep(u32::MAX, 100, |ms| seen.push(ms), || true);
        assert_eq!(seen, [15, 0, u32::MAX]);
    }

    #[test]
    fn the_setting_parses_with_a_default() {
        assert_eq!(sleep_hold_min(None), Some(SLEEP_HOLD_MIN_MS));
        assert_eq!(sleep_hold_min(Some("0")), None);
        assert_eq!(sleep_hold_min(Some(" 0 ")), None);
        assert_eq!(sleep_hold_min(Some(" 250 ")), Some(250));
        assert_eq!(sleep_hold_min(Some("soon")), Some(SLEEP_HOLD_MIN_MS));
    }

    #[test]
    fn only_sleep_imports_of_synch_dlls_are_matched() {
        assert!(sleep_dll("KERNEL32.dll"));
        assert!(sleep_dll("KernelBase.dll"));
        assert!(sleep_dll("api-ms-win-core-synch-l1-2-0.dll"));
        assert!(!sleep_dll("api-ms-win-core-file-l1-1-0.dll"));
        assert!(!sleep_dll("user32.dll"));
        assert_eq!(sleep_slot("Sleep"), Some(0));
        assert_eq!(sleep_slot("SleepEx"), Some(1));
        assert_eq!(sleep_slot("SleepConditionVariableSRW"), None);
    }
}
