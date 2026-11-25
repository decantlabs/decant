//! `DECANT_KEYS=<second>:<key>[,...]`: pad input at fixed times since the process started, for a run
//! that must get past a menu without anybody at the controls. No Windows code here.
//!
//! ```text
//! 12:A            press A for 100 ms, 12 s after the start
//! 12.5:Menu*1000  hold Menu for 1000 ms
//! 20:L:0:1        left stick fully up (it stays there until the next L entry)
//! 21:R:-1:0       right stick fully left
//! 22:LT:1         left trigger fully pressed (stays)
//! 30:release      every button, stick and trigger back to neutral
//! ```
//!
//! Buttons are the names of the control file (`A B X Y Menu View Up Down Left Right LB RB LS RS`).
//! The entries add to the state of `DECANT_PAD_SCRIPT` when both are set.

use crate::pad::{Command, PadState, button_named};

/// Default length of a press, in milliseconds.
pub const PRESS_MS: u64 = 100;

fn axis(word: &str, low: f32) -> Result<f32, String> {
    let v: f32 = word.trim().parse().map_err(|_| format!("{word:?} is not a number"))?;
    if !v.is_finite() {
        return Err(format!("{word:?} is not finite"));
    }
    Ok(v.clamp(low, 1.0))
}

/// The timed commands of the list, in milliseconds since the start, sorted by time (entries at the
/// same time keep their order).
pub fn parse(value: &str) -> Result<Vec<(u64, Command)>, String> {
    let mut out = Vec::new();
    for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (time, what) = item.split_once(':').ok_or_else(|| format!("{item}: <second>:<key> expected"))?;
        let seconds: f64 = time.trim().trim_end_matches('s').parse().map_err(|_| format!("{item}: {time:?} is not a number of seconds"))?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(format!("{item}: {time:?} is not a number of seconds"));
        }
        let at = (seconds * 1000.0).round() as u64;
        let parts: Vec<&str> = what.split(':').map(str::trim).collect();
        match parts.as_slice() {
            [side @ ("L" | "l" | "R" | "r"), x, y] => out.push((at, Command::Stick { right: side.eq_ignore_ascii_case("r"), x: axis(x, -1.0)?, y: axis(y, -1.0)? })),
            [t, v] if t.eq_ignore_ascii_case("lt") || t.eq_ignore_ascii_case("rt") => out.push((at, Command::Trigger { right: t.eq_ignore_ascii_case("rt"), value: axis(v, 0.0)? })),
            [r] if r.eq_ignore_ascii_case("release") => out.push((at, Command::Release)),
            [key] => {
                let (name, ms) = match key.split_once('*') {
                    Some((n, ms)) => (n.trim(), ms.trim().parse::<u64>().map_err(|_| format!("{item}: {ms:?} is not a whole number of milliseconds"))?),
                    None => (*key, PRESS_MS),
                };
                let mask = button_named(name).ok_or_else(|| format!("{item}: {name:?} is not a button"))?;
                out.push((at, Command::Button { mask, down: true }));
                out.push((at + ms, Command::Button { mask, down: false }));
            }
            _ => return Err(format!("{item}: not a key, a stick (L:x:y) or a trigger (LT:v)")),
        }
    }
    out.sort_by_key(|(at, _)| *at);
    Ok(out)
}

/// Applies one command to a state.
pub fn apply(state: &mut PadState, command: Command) {
    match command {
        Command::Button { mask, down: true } => state.buttons |= mask,
        Command::Button { mask, down: false } => state.buttons &= !mask,
        Command::Stick { right: false, x, y } => (state.left_x, state.left_y) = (x, y),
        Command::Stick { right: true, x, y } => (state.right_x, state.right_y) = (x, y),
        Command::Trigger { right: false, value } => state.left_trigger = value,
        Command::Trigger { right: true, value } => state.right_trigger = value,
        Command::Release => *state = PadState::default(),
        Command::Wait(_) => {}
    }
}

/// The timed commands, played against the time since the start.
#[derive(Debug, Default)]
pub struct Schedule {
    entries: Vec<(u64, Command)>,
    next: usize,
    state: PadState,
    pressed_at: [Option<(u64, u64)>; 32],
}

impl Schedule {
    pub fn new(entries: Vec<(u64, Command)>) -> Schedule {
        Schedule { entries, next: 0, state: PadState::default(), pressed_at: [None; 32] }
    }

    /// Runs every entry due at `now_ms`, and returns the ones that ran.
    ///
    /// A release paired with an earlier press holds for the full planned length from the poll that
    /// actually applied the press, so a late poll cannot shorten or swallow the press.
    pub fn advance(&mut self, now_ms: u64) -> Vec<(u64, Command)> {
        let mut ran = Vec::new();
        while let Some(&(at, c)) = self.entries.get(self.next) {
            let due_at = match c {
                Command::Button { mask, down: false } => {
                    let slot = (mask.trailing_zeros() as usize).min(31);
                    match self.pressed_at[slot] {
                        Some((planned, actual)) => actual.saturating_add(at.saturating_sub(planned)),
                        None => at,
                    }
                }
                _ => at,
            };
            if due_at > now_ms {
                break;
            }
            if let Command::Button { mask, down } = c {
                let slot = (mask.trailing_zeros() as usize).min(31);
                self.pressed_at[slot] = if down { Some((at, now_ms)) } else { None };
            }
            apply(&mut self.state, c);
            ran.push((at, c));
            self.next += 1;
        }
        ran
    }

    pub fn state(&self) -> PadState {
        self.state
    }

    pub fn done(&self) -> bool {
        self.next >= self.entries.len()
    }
}

/// The state the title sees: the buttons of both, and for each axis the scheduled value when it is
/// not at rest, else the script's.
pub fn combine(script: PadState, keys: PadState) -> PadState {
    let pick = |k: f32, s: f32| if k != 0.0 { k } else { s };
    PadState {
        buttons: script.buttons | keys.buttons,
        left_trigger: pick(keys.left_trigger, script.left_trigger),
        right_trigger: pick(keys.right_trigger, script.right_trigger),
        left_x: pick(keys.left_x, script.left_x),
        left_y: pick(keys.left_y, script.left_y),
        right_x: pick(keys.right_x, script.right_x),
        right_y: pick(keys.right_y, script.right_y),
    }
}

/// One line for the log: `press A`, `release A`, `left stick 0 1`.
pub fn describe(c: Command) -> String {
    match c {
        Command::Button { mask, down } => format!("{} {}", if down { "press" } else { "release" }, button_text(mask)),
        Command::Stick { right, x, y } => format!("{} stick {x} {y}", if right { "right" } else { "left" }),
        Command::Trigger { right, value } => format!("{} trigger {value}", if right { "right" } else { "left" }),
        Command::Release => "release all".into(),
        Command::Wait(ms) => format!("wait {ms}"),
    }
}

fn button_text(mask: u32) -> &'static str {
    ["Menu", "View", "A", "B", "X", "Y", "Up", "Down", "Left", "Right", "LB", "RB", "LS", "RS"].get(mask.trailing_zeros() as usize).copied().unwrap_or("?")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pad::{A, B, MENU};

    #[test]
    fn a_list_becomes_timed_commands() {
        let c = parse("12:A, 2.5:Menu*1000, 20:L:0:1, 21:r:-2:0.5, 22:LT:1, 30:release").unwrap();
        assert_eq!(
            c,
            vec![
                (2500, Command::Button { mask: MENU, down: true }),
                (3500, Command::Button { mask: MENU, down: false }),
                (12_000, Command::Button { mask: A, down: true }),
                (12_100, Command::Button { mask: A, down: false }),
                (20_000, Command::Stick { right: false, x: 0.0, y: 1.0 }),
                (21_000, Command::Stick { right: true, x: -1.0, y: 0.5 }),
                (22_000, Command::Trigger { right: false, value: 1.0 }),
                (30_000, Command::Release),
            ]
        );
        for bad in ["A", "x:A", "-1:A", "5:Z", "5:A*soon", "5:L:1", "5:LT:x", "5:Q:1:1"] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        assert_eq!(parse("").unwrap(), vec![]);
    }

    #[test]
    fn the_schedule_plays_in_time() {
        let mut s = Schedule::new(parse("1:A, 1.05:B, 2:L:0:1").unwrap());
        assert!(s.advance(999).is_empty());
        assert_eq!(s.advance(1000).len(), 1);
        assert_eq!(s.state().buttons, A);
        s.advance(1100);
        assert_eq!(s.state().buttons, B, "A released at 1.1 s, B pressed at 1.05 s");
        s.advance(5000);
        assert_eq!(s.state(), PadState { left_y: 1.0, ..PadState::default() });
        assert!(s.done());
    }

    #[test]
    fn the_keys_add_to_the_script() {
        let script = PadState { buttons: A, left_x: 0.5, right_trigger: 1.0, ..PadState::default() };
        let keys = PadState { buttons: B, left_x: -1.0, ..PadState::default() };
        assert_eq!(combine(script, keys), PadState { buttons: A | B, left_x: -1.0, right_trigger: 1.0, ..PadState::default() });
        assert_eq!(describe(Command::Button { mask: MENU, down: true }), "press Menu");
    }

    #[test]
    fn a_late_poll_still_holds_a_press_for_its_full_duration() {
        // Reproduces the 20261125 000846 run at 90:A, where the polling thread woke at
        // 90_147 ms (past both 90_000 press and 90_100 release). The press must apply on that
        // poll and stay down for the full 100 ms so the title's 33 ms frame poll sees A down.
        let mut s = Schedule::new(parse("90:A").unwrap());
        assert!(s.advance(89_990).is_empty());
        assert_eq!(s.advance(90_147), vec![(90_000, Command::Button { mask: A, down: true })]);
        assert_eq!(s.state().buttons, A, "the title sees A down when polling at 90.147 s");
        assert!(s.advance(90_180).is_empty(), "still held one 33 ms frame later");
        assert_eq!(s.state().buttons, A);
        assert!(s.advance(90_246).is_empty());
        assert_eq!(s.advance(90_247), vec![(90_100, Command::Button { mask: A, down: false })]);
        assert_eq!(s.state().buttons, 0);
        assert!(s.done());

        // Reproduces the 20261124 201616 run at 140:A, where the press woke at
        // 140_124 ms and would otherwise release at 140_166 ms (only 42 ms held).
        let mut s2 = Schedule::new(parse("140:A").unwrap());
        assert_eq!(s2.advance(140_124), vec![(140_000, Command::Button { mask: A, down: true })]);
        assert_eq!(s2.state().buttons, A);
        assert!(s2.advance(140_166).is_empty(), "not released after only 42 ms");
        assert_eq!(s2.state().buttons, A);
        assert_eq!(s2.advance(140_224), vec![(140_100, Command::Button { mask: A, down: false })]);
        assert_eq!(s2.state().buttons, 0);
    }
}
