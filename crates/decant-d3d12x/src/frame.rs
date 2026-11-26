//! Frame pacing of the console device (slots 78 to 80: `SetFrameIntervalX`, `ScheduleFrameEventX`,
//! `WaitFrameEventX`), emulated with a timer.
//!
//! One pacer per process, because a title has one device. `SetFrameIntervalX` sets the frame
//! period (the interval in microseconds, the periodicity is ignored) and restarts the count.
//! `ScheduleFrameEventX` gives each event type an offset from every frame origin. `WaitFrameEventX`
//! sleeps until the next deadline of its type when it is early, and when it is late it hands back
//! the latest due frame at once. Until `SetFrameIntervalX` runs the period is 1/60 s. A deadline
//! beyond the caller's timeout makes the wait return `S_FALSE`. Without these waits a console
//! engine's game thread never sees its frame start, and its watchdog ends the process.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use windows_sys::Win32::Foundation::{CloseHandle, E_INVALIDARG, HANDLE, S_FALSE, S_OK};
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{
    CreateWaitableTimerExW, INFINITE, SetWaitableTimer, Sleep, TIMER_ALL_ACCESS, WaitForSingleObject,
};
use windows_sys::core::HRESULT;

use crate::log::xs_log;
use crate::proxy::log_due;

const EVENT_TYPES: usize = 8;
const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
/// `WaitFrameEventX` flags bit: on a timed out wait still issue a new token and advance the frame,
/// as if the deadline had been reached.
const TOKEN_ON_TIMEOUT: u32 = 0x2;
/// `ScheduleFrameEventX` (slot 79) result for the call that first schedules event type 0.
const STARTED_CLOCK: HRESULT = 0x1000_0000;
/// Microseconds of one frame at 60 Hz, what an interval of 0 stands for.
const SIXTIETH_US: u32 = 16_667;

/// How many issued type 0 tokens are remembered.
const TOKEN_MEMORY: usize = 64;
/// Streaming work units in one frame needed to trigger loading uncap in `Auto` mode.
const BURST_THRESHOLD: u64 = 16;
/// Frames to stay uncapped after streaming work drops below `BURST_THRESHOLD`.
const UNCAPPED_HOLD_FRAMES: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UncapMode {
    Off,
    Auto,
    Always,
}

fn parse_uncap_mode(value: Option<&str>) -> UncapMode {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("0" | "off" | "false") => UncapMode::Off,
        Some("always") => UncapMode::Always,
        _ => UncapMode::Auto,
    }
}

fn uncap_mode() -> UncapMode {
    static MODE: OnceLock<UncapMode> = OnceLock::new();
    *MODE.get_or_init(|| parse_uncap_mode(std::env::var("DECANT_D3D12X_LOAD_UNCAP").ok().as_deref()))
}

/// Records bulk streaming work (`detile` textures/bytes or `title_mem` pages/tile ranges) so
/// `wait_event` can skip frame pacing during heavy loading bursts.
#[allow(dead_code)]
pub(crate) fn note_streaming_work(units: u64) {
    crate::title_mem::note_streaming_work(units);
}
/// The last tokens handed out, oldest first. A caller may pass one back in, and only a token from
/// this window counts as genuine.
static TOKEN_WINDOW: Mutex<VecDeque<u64>> = Mutex::new(VecDeque::new());
static TOKEN_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Hands out a new type 0 frame token and remembers it. Zero is never issued.
fn issue_token() -> u64 {
    let token = loop {
        let candidate = TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
        if candidate != 0 {
            break candidate;
        }
    };
    let mut window = TOKEN_WINDOW.lock().unwrap_or_else(PoisonError::into_inner);
    if window.len() == TOKEN_MEMORY {
        window.pop_front();
    }
    window.push_back(token);
    token
}

/// Whether `token` is still in the window of recently issued tokens.
fn token_known(token: u64) -> bool {
    TOKEN_WINDOW.lock().unwrap_or_else(PoisonError::into_inner).contains(&token)
}

/// `PresentX` (console queue slot 30) names the token of the frame it flips. Nothing is keyed off
/// it yet, the first non zero token is only logged so a present shows up in the frame event log.
pub(crate) fn mark_flipped(token: u64) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if token != 0 {
        ONCE.call_once(|| xs_log!("PresentX: first non-zero frame token flipped (token {:016X})", token));
    }
}

/// Console device slot 81: `GetFrameStatisticsX(token, type, count in/out, statistics)`. The
/// layout of the records is not measured yet, so none are kept: `count` is set to 0 and the call
/// succeeds. A title calls it about once per frame.
pub(crate) fn statistics(token: u64, kind: u32, count: *mut u32, _out: *mut u8) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if log_due(&N) {
        xs_log!("GetFrameStatisticsX(token {token:016X}, type {kind}): no statistics kept, count 0");
    }
    if count.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `count` is the title's in/out pointer, checked not null above.
    unsafe { count.write(0) };
    S_OK
}

/// The state of the frame pacer. All counter values are `QueryPerformanceCounter` units.
struct Clock {
    started: bool,
    /// Set by the first schedule of event type 0, which is the call that answers `STARTED_CLOCK`.
    active: bool,
    /// Counter ticks per second.
    freq: i64,
    /// Where frame zero begins.
    origin: i64,
    /// Ticks per frame.
    period: f64,
    /// Ticks after each frame origin, per event type.
    offset: [i64; EVENT_TYPES],
    /// The first frame index not handed out yet, per event type.
    next: [u64; EVENT_TYPES],
    /// Remaining frames to stay uncapped after a streaming burst.
    uncap_hold: u32,
    /// Whether the clock was uncapped on the previous frame check.
    uncapped: bool,
}

static CLOCK: Mutex<Clock> = Mutex::new(Clock {
    started: false,
    active: false,
    freq: 0,
    origin: 0,
    period: 0.0,
    offset: [0; EVENT_TYPES],
    next: [0; EVENT_TYPES],
    uncap_hold: 0,
    uncapped: false,
});

fn lock() -> MutexGuard<'static, Clock> {
    CLOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

fn now() -> i64 {
    let mut t = 0i64;
    // SAFETY: `t` is writable storage for the counter.
    unsafe { QueryPerformanceCounter(&mut t) };
    t
}

impl Clock {
    /// Starts pacing at 60 Hz unless something did already.
    fn start(&mut self) {
        if self.started {
            return;
        }
        let mut f = 0i64;
        // SAFETY: `f` is writable storage for the frequency.
        unsafe { QueryPerformanceFrequency(&mut f) };
        self.freq = f;
        self.origin = now();
        self.period = f as f64 / 60.0;
        self.next = [1; EVENT_TYPES];
        self.started = true;
    }

    /// The frame an event of type `kind` should be answered with at counter value `at`, and the
    /// counter value of its deadline. After a stall the latest due frame is used, so missed
    /// frames are never replayed.
    fn next_event(&self, kind: usize, at: i64) -> (u64, i64) {
        let base = self.origin + self.offset[kind];
        let latest_due = ((at - base) as f64 / self.period) as u64;
        let frame = self.next[kind].max(latest_due);
        (frame, base + (frame as f64 * self.period).ceil() as i64)
    }

    /// Records that `frame` was handed out for `kind`.
    fn consume(&mut self, kind: usize, frame: u64) {
        self.next[kind] = self.next[kind].max(frame + 1);
    }

    /// Updates streaming burst uncap state and resets the clock origin when leaving uncap so
    /// steady-state pacing resumes smoothly without catch-up stutter.
    fn update_uncap(&mut self, mode: UncapMode, burst_units: u64, decay: bool, at: i64) -> bool {
        let active = match mode {
            UncapMode::Off => {
                self.uncap_hold = 0;
                false
            }
            UncapMode::Always => true,
            UncapMode::Auto => {
                if burst_units >= BURST_THRESHOLD {
                    self.uncap_hold = UNCAPPED_HOLD_FRAMES;
                } else if decay && self.uncap_hold > 0 {
                    self.uncap_hold -= 1;
                }
                self.uncap_hold > 0
            }
        };
        if self.uncapped && !active {
            self.origin = at;
            self.next = [1; EVENT_TYPES];
        }
        self.uncapped = active;
        active
    }
}

/// Sleeps until the counter reaches `deadline`: a high resolution waitable timer for the bulk and
/// a yield loop for the last millisecond.
/// The longest a frame event is waited for, in ticks: about four 60 Hz frames, so a distant or infinite
/// event never holds the caller for long. Pure, for the tests.
pub(crate) fn bounded_deadline(deadline: i64, now: i64, freq: i64) -> i64 {
    deadline.min(now + freq * 66 / 1000)
}

fn wait_until(deadline: i64, freq: i64) {
    // SAFETY: plain creation calls with null optional arguments.
    let timer: HANDLE = unsafe {
        let mut timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if timer.is_null() {
            timer = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), 0, TIMER_ALL_ACCESS);
        }
        timer
    };
    loop {
        let left = deadline - now();
        if left <= 0 {
            break;
        }
        let hundred_ns = left * 10_000_000 / freq;
        // SAFETY: `timer` is a live handle (checked) and `due` outlives the call.
        unsafe {
            if !timer.is_null() && hundred_ns > 20_000 {
                // Negative means relative. Wake one millisecond early.
                let due = -(hundred_ns - 10_000);
                if SetWaitableTimer(timer, &due, 0, None, std::ptr::null(), 0) != 0 {
                    WaitForSingleObject(timer, INFINITE);
                } else {
                    Sleep(1);
                }
            } else {
                Sleep(0);
            }
        }
    }
    if !timer.is_null() {
        // SAFETY: the handle was created above and is closed once.
        unsafe { CloseHandle(timer) };
    }
}

/// The interval a title asks for, where 0 means the default of one 60 Hz frame.
fn effective_interval(interval_us: u32) -> u32 {
    if interval_us == 0 { SIXTIETH_US } else { interval_us }
}

/// Console slot 78: `SetFrameIntervalX(output, interval in microseconds, periodicity, flags)`.
pub(crate) fn set_interval(output: usize, interval_us: u32, periodicity: u32, flags: u32) -> HRESULT {
    let interval_us = effective_interval(interval_us);
    {
        let mut clock = lock();
        clock.start();
        clock.origin = now();
        clock.period = clock.freq as f64 * f64::from(interval_us) / 1e6;
        clock.next = [1; EVENT_TYPES];
    }
    xs_log!(
        "SetFrameIntervalX(output={:016X}, {interval_us} us, periodicity {periodicity}, flags 0x{flags:X}): one frame every {interval_us} us from now (emulated clock)",
        output
    );
    S_OK
}

/// Console slot 79: `ScheduleFrameEventX(type, offset in microseconds, auxiliary, flags)`. For
/// event type 0 (the frame origin) the answer is `STARTED_CLOCK` the first time and `S_FALSE` on
/// every later call. The other types answer `S_OK`.
pub(crate) fn schedule_event(kind: u32, offset_us: u32, aux: usize, flags: u32) -> HRESULT {
    if kind as usize >= EVENT_TYPES {
        xs_log!("ScheduleFrameEventX(type {kind}, {offset_us} us, {:016X}, 0x{flags:X}): event type out of the emulated range, refused", aux);
        return E_INVALIDARG;
    }
    let hr = {
        let mut clock = lock();
        clock.start();
        clock.offset[kind as usize] = (clock.freq as f64 * f64::from(offset_us) / 1e6) as i64;
        match (kind, clock.active) {
            (0, false) => {
                clock.active = true;
                STARTED_CLOCK
            }
            (0, true) => S_FALSE,
            _ => S_OK,
        }
    };
    xs_log!("ScheduleFrameEventX(type {kind}, {offset_us} us after each frame origin, aux {:016X}, flags 0x{flags:X}) -> hr=0x{:08x}", aux, hr);
    hr
}

/// Console slot 80: `WaitFrameEventX(type, timeout in milliseconds, auxiliary, flags, token in/out)`.
/// Returns the result and the token to write back, if it changed. `token_in` is the caller's
/// current value, 0 when the title passes none.
///
/// A wait whose deadline lies beyond the timeout returns `S_FALSE` for every type. For type 0,
/// flag `TOKEN_ON_TIMEOUT` still issues a token and advances the frame on that timeout. A token
/// passed in must be a recently issued one and must not come with `TOKEN_ON_TIMEOUT`, or the call
/// fails with `E_INVALIDARG`. Types other than 0 use the frame counter as their token.
pub(crate) fn wait_event(kind: u32, timeout_ms: u32, aux: usize, flags: u32, token_in: u64) -> (HRESULT, Option<u64>) {
    static CALLS: AtomicI32 = AtomicI32::new(0);
    static TIMEOUTS: AtomicI32 = AtomicI32::new(0);
    if kind as usize >= EVENT_TYPES {
        xs_log!("WaitFrameEventX(type {kind}): event type out of the emulated range, refused");
        return (E_INVALIDARG, None);
    }
    let t = kind as usize;
    if t == 0 && token_in != 0 && (!token_known(token_in) || flags & TOKEN_ON_TIMEOUT != 0) {
        xs_log!("WaitFrameEventX(type 0, token {token_in:016X}, flags 0x{flags:X}): not a recently issued token, or passed together with 0x2, E_INVALIDARG");
        return (E_INVALIDARG, None);
    }
    let burst_units = if t == 0 { crate::title_mem::take_streaming_work() } else { crate::title_mem::peek_streaming_work() };
    let mode = uncap_mode();
    let (frame, deadline, arrival, freq, uncapped) = {
        let mut clock = lock();
        clock.start();
        let arrival = now();
        let uncapped = clock.update_uncap(mode, burst_units, t == 0, arrival);
        let (frame, deadline) = clock.next_event(t, arrival);
        if uncapped {
            clock.consume(t, frame);
        }
        (frame, deadline, arrival, clock.freq, uncapped)
    };
    if uncapped {
        return (S_OK, Some(if t == 0 { issue_token() } else { frame + 1 }));
    }

    let timeout_ticks = i64::from(timeout_ms) * freq / 1000;
    if timeout_ms != INFINITE && deadline - arrival > timeout_ticks {
        if timeout_ms != 0 {
            // SAFETY: no precondition.
            unsafe { Sleep(timeout_ms) };
        }
        if log_due(&TIMEOUTS) {
            xs_log!("WaitFrameEventX(type {kind}, {timeout_ms} ms): next frame event later than the timeout, S_FALSE");
        }
        if t == 0 && flags & TOKEN_ON_TIMEOUT != 0 {
            lock().consume(t, frame);
            return (S_FALSE, Some(issue_token()));
        }
        return (S_FALSE, None);
    }
    let deadline = bounded_deadline(deadline, arrival, freq);
    if deadline > arrival {
        wait_until(deadline, freq);
    }
    lock().consume(t, frame);
    if log_due(&CALLS) {
        xs_log!("WaitFrameEventX(type {kind}, {timeout_ms} ms, aux {:016X}, flags 0x{flags:X}): frame {} reached (emulated clock)", aux, frame + 1);
    }
    (S_OK, Some(if t == 0 { issue_token() } else { frame + 1 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A started clock at 1 MHz so tick values read as microseconds.
    fn micro_clock(period: f64) -> Clock {
        Clock { started: true, active: true, freq: 1_000_000, origin: 0, period, offset: [0; EVENT_TYPES], next: [1; EVENT_TYPES], uncap_hold: 0, uncapped: false }
    }

    #[test]
    fn zero_interval_means_one_sixtieth_of_a_second() {
        assert_eq!(effective_interval(0), 16_667);
        assert_eq!(effective_interval(8_333), 8_333);
    }

    #[test]
    fn an_early_arrival_waits_for_the_next_frame() {
        let clock = micro_clock(20_000.0);
        assert_eq!(clock.next_event(1, 5_000), (1, 20_000));
    }

    #[test]
    fn a_stall_skips_to_the_latest_due_frame_without_replaying() {
        let clock = micro_clock(20_000.0);
        let (frame, deadline) = clock.next_event(1, 205_000);
        assert_eq!(frame, 10);
        assert_eq!(deadline, 200_000);
    }

    #[test]
    fn consumed_frames_are_not_handed_out_twice() {
        let mut clock = micro_clock(20_000.0);
        let (frame, _) = clock.next_event(3, 25_000);
        clock.consume(3, frame);
        assert_eq!(clock.next_event(3, 25_000).0, frame + 1);
        // Consuming an older frame never moves the counter back.
        clock.consume(3, 0);
        assert_eq!(clock.next[3], frame + 1);
    }

    #[test]
    fn event_offsets_shift_the_deadline_of_their_type_only() {
        let mut clock = micro_clock(20_000.0);
        clock.offset[2] = 3_000;
        assert_eq!(clock.next_event(2, 0), (1, 23_000));
        assert_eq!(clock.next_event(4, 0), (1, 20_000));
    }

    #[test]
    fn fractional_periods_round_the_deadline_up() {
        let clock = micro_clock(33_333.5);
        assert_eq!(clock.next_event(0, 0), (1, 33_334));
    }

    #[test]
    fn tokens_are_nonzero_distinct_and_remembered() {
        let first = issue_token();
        let second = issue_token();
        assert_ne!(first, 0);
        assert_ne!(first, second);
        assert!(token_known(second));
        assert!(!token_known(0));
    }

    #[test]
    fn the_token_window_forgets_old_tokens() {
        let oldest = issue_token();
        for _ in 0..TOKEN_MEMORY {
            issue_token();
        }
        assert!(!token_known(oldest));
    }

    #[test]
    fn out_of_range_event_types_are_refused() {
        assert_eq!(schedule_event(EVENT_TYPES as u32, 0, 0, 0), E_INVALIDARG);
        assert_eq!(wait_event(EVENT_TYPES as u32, 0, 0, 0, 0), (E_INVALIDARG, None));
    }

    #[test]
    fn a_foreign_token_or_a_token_with_the_timeout_flag_is_refused() {
        assert_eq!(wait_event(0, 0, 0, 0, u64::MAX).0, E_INVALIDARG);
        let token = issue_token();
        assert_eq!(wait_event(0, 0, 0, TOKEN_ON_TIMEOUT, token).0, E_INVALIDARG);
    }

    #[test]
    fn statistics_reports_zero_records() {
        let mut count = 7u32;
        assert_eq!(statistics(1, 0, &mut count, std::ptr::null_mut()), S_OK);
        assert_eq!(count, 0);
        assert_eq!(statistics(1, 0, std::ptr::null_mut(), std::ptr::null_mut()), E_INVALIDARG);
    }

    #[test]
    fn uncap_mode_parses_auto_off_and_always() {
        assert_eq!(parse_uncap_mode(None), UncapMode::Auto);
        assert_eq!(parse_uncap_mode(Some("")), UncapMode::Auto);
        assert_eq!(parse_uncap_mode(Some("1")), UncapMode::Auto);
        assert_eq!(parse_uncap_mode(Some("auto")), UncapMode::Auto);
        assert_eq!(parse_uncap_mode(Some("0")), UncapMode::Off);
        assert_eq!(parse_uncap_mode(Some("off")), UncapMode::Off);
        assert_eq!(parse_uncap_mode(Some("FALSE")), UncapMode::Off);
        assert_eq!(parse_uncap_mode(Some("always")), UncapMode::Always);
    }

    #[test]
    fn streaming_burst_uncaps_decays_and_resets_clock_origin_on_exit() {
        let mut clock = micro_clock(33_333.0);
        clock.origin = 1_000;
        clock.next[0] = 5;

        // Below threshold keeps normal pacing.
        assert!(!clock.update_uncap(UncapMode::Auto, BURST_THRESHOLD - 1, true, 10_000));
        assert_eq!(clock.origin, 1_000);

        // Burst activates uncap for UNCAPPED_HOLD_FRAMES.
        assert!(clock.update_uncap(UncapMode::Auto, BURST_THRESHOLD, true, 20_000));
        assert_eq!(clock.uncap_hold, UNCAPPED_HOLD_FRAMES);

        // Secondary event check does not decay hold.
        assert!(clock.update_uncap(UncapMode::Auto, 0, false, 25_000));
        assert_eq!(clock.uncap_hold, UNCAPPED_HOLD_FRAMES);

        // Decay for UNCAPPED_HOLD_FRAMES - 1 primary frames stays uncapped.
        for i in 1..UNCAPPED_HOLD_FRAMES {
            assert!(clock.update_uncap(UncapMode::Auto, 0, true, 30_000 + i64::from(i) * 1_000));
        }
        assert_eq!(clock.uncap_hold, 1);

        // Final decay exits uncap and resets origin and next counters cleanly.
        let exit_at = 500_000;
        assert!(!clock.update_uncap(UncapMode::Auto, 0, true, exit_at));
        assert_eq!(clock.origin, exit_at);
        assert_eq!(clock.next, [1; EVENT_TYPES]);
        assert_eq!(clock.next_event(0, exit_at), (1, exit_at + 33_333));

        // Off and Always modes behave as configured.
        assert!(!clock.update_uncap(UncapMode::Off, BURST_THRESHOLD * 4, true, 600_000));
        assert!(clock.update_uncap(UncapMode::Always, 0, true, 610_000));
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    /// A distant frame event is waited for four frames at most.
    #[test]
    fn a_distant_frame_event_is_waited_for_at_most_four_frames() {
        let freq = 1_000_000;
        assert_eq!(bounded_deadline(10_000_000, 0, freq), 66_000);
        assert_eq!(bounded_deadline(5_000, 0, freq), 5_000, "a near deadline is kept");
        assert!(bounded_deadline(i64::MAX / 4, 0, freq) <= 66_000);
    }
}
