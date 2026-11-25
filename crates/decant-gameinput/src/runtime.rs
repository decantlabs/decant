//! The live pad of the DLL: the current state, the clock of the readings and the thread that polls the control file.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use decant_log::trace;

use crate::keys::Schedule;
use crate::pad::{Driver, PadState};

/// How often the thread looks at the control file.
pub const POLL: Duration = Duration::from_millis(16);
/// Parse errors logged per run, so a file full of nonsense cannot flood the log.
const MAX_LOGGED_ERRORS: usize = 20;

static START: OnceLock<Instant> = OnceLock::new();
static LAST_US: AtomicU64 = AtomicU64::new(0);
static STATE: Mutex<PadState> = Mutex::new(PadState { buttons: 0, left_trigger: 0.0, right_trigger: 0.0, left_x: 0.0, left_y: 0.0, right_x: 0.0, right_y: 0.0 });

fn start_instant() -> Instant {
    *START.get_or_init(Instant::now)
}

/// Microseconds since the first call, never the same twice and never going back: what the timestamps of readings need.
pub fn now_us() -> u64 {
    let real = start_instant().elapsed().as_micros() as u64 + 1;
    let mut previous = LAST_US.load(Ordering::Relaxed);
    loop {
        let next = real.max(previous + 1);
        match LAST_US.compare_exchange_weak(previous, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(seen) => previous = seen,
        }
    }
}

/// The state of the pad now.
pub fn state() -> PadState {
    *STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts the polling thread once. Without a control file and without `DECANT_KEYS` the pad stays
/// neutral and no thread runs. `on_change` is called from the thread with every new state, after
/// [`state`] returns it.
pub fn start(script: Option<PathBuf>, keys: Option<String>, on_change: fn(PadState)) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        start_instant();
        let schedule = keys.filter(|k| !k.trim().is_empty()).and_then(|k| match crate::keys::parse(&k) {
            Ok(entries) if !entries.is_empty() => {
                trace(&format!("gameinput: DECANT_KEYS plays {} timed inputs: {k}", entries.len()));
                Some(Schedule::new(entries))
            }
            Ok(_) => None,
            Err(e) => {
                trace(&format!("gameinput: DECANT_KEYS ignored: {e}"));
                None
            }
        });
        let real = crate::devices::enabled();
        if script.is_none() && schedule.is_none() && !real {
            trace("gameinput: DECANT_PAD_SCRIPT is not set, the gamepad stays neutral");
            return;
        }
        if let Some(path) = &script {
            trace(&format!("gameinput: the gamepad follows {}", path.display()));
        }
        if real && script.is_none() && schedule.is_none() {
            trace("gameinput: no script and no DECANT_KEYS, the gamepad follows a real controller or keyboard if one answers");
        }
        let spawned = std::thread::Builder::new().name("decant-gameinput-pad".into()).spawn(move || poll_forever(script, schedule, on_change));
        if let Err(e) = spawned {
            trace(&format!("gameinput: cannot start the polling thread: {e}"));
        }
    });
}

/// Milliseconds since the process started: the clock of `DECANT_KEYS`.
fn process_age_ms() -> u64 {
    decant_log::calls::now_ms().saturating_sub(decant_log::calls::process_start_ms())
}

fn poll_forever(path: Option<PathBuf>, mut schedule: Option<Schedule>, on_change: fn(PadState)) {
    let mut driver = path.map(Driver::new);
    let mut logged_errors = 0;
    let real_input = crate::devices::enabled();
    let mut last = PadState::default();
    loop {
        let now = start_instant().elapsed().as_millis() as u64;
        if let Some(driver) = driver.as_mut() {
            let (found, _) = driver.tick(now);
            if found.restarted {
                trace("gameinput: the control file was emptied or replaced, starting again from its first line");
            }
            for (line, why) in &found.errors {
                if logged_errors < MAX_LOGGED_ERRORS {
                    logged_errors += 1;
                    trace(&format!("gameinput: line {line} of the control file is ignored: {why}"));
                }
            }
        }
        if let Some(s) = schedule.as_mut() {
            for (at, c) in s.advance(process_age_ms()) {
                trace(&format!("gameinput: DECANT_KEYS at {:.3} s: {}", at as f64 / 1000.0, crate::keys::describe(c)));
            }
        }
        let script_state = driver.as_ref().map(|d| d.machine.state()).unwrap_or_default();
        let scripted = crate::keys::combine(script_state, schedule.as_ref().map(Schedule::state).unwrap_or_default());
        let real_state = if real_input { crate::devices::poll() } else { PadState::default() };
        let state = crate::devices::merge(scripted, real_state);
        if state != last {
            last = state;
            *STATE.lock().unwrap_or_else(|e| e.into_inner()) = state;
            trace(&format!(
                "gameinput: pad buttons {:#x}, left stick {} {}, right stick {} {}, triggers {} {}",
                state.buttons, state.left_x, state.left_y, state.right_x, state.right_y, state.left_trigger, state.right_trigger
            ));
            on_change(state);
        }
        // Nothing left to play, no file to follow, and no real input to keep watching: done.
        if driver.is_none() && schedule.as_ref().is_none_or(Schedule::done) && !real_input {
            return;
        }
        std::thread::sleep(POLL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_only_go_up() {
        let mut last = now_us();
        for _ in 0..1000 {
            let next = now_us();
            assert!(next > last);
            last = next;
        }
    }
}
