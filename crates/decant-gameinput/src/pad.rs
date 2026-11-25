//! The simulated pad: its state, the control file language, and the state machine that plays it.
//!
//! Nothing here touches Windows. The language is one command per line (see the crate README):
//!
//! ```text
//! A down                 press A and keep it pressed
//! A up                   release A
//! press A [ms]           A down, wait ms (default 100), A up
//! hold Start 1000        the same with a required time
//! wait 500               nothing happens for 500 ms
//! stick left 0.0 1.0     left stick x y, each from -1 to 1 (y positive is up)
//! trigger right 0.5      right trigger from 0 to 1
//! release                every button, stick and trigger back to neutral
//! # a comment
//! ```
//!
//! Commands run in order. `wait` (and the wait inside `press` and `hold`) holds back every command
//! after it, so a file reads as a timeline.

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

pub const MENU: u32 = 0x1;
pub const VIEW: u32 = 0x2;
pub const A: u32 = 0x4;
pub const B: u32 = 0x8;
pub const X: u32 = 0x10;
pub const Y: u32 = 0x20;
pub const DPAD_UP: u32 = 0x40;
pub const DPAD_DOWN: u32 = 0x80;
pub const DPAD_LEFT: u32 = 0x100;
pub const DPAD_RIGHT: u32 = 0x200;
pub const LEFT_SHOULDER: u32 = 0x400;
pub const RIGHT_SHOULDER: u32 = 0x800;
pub const LEFT_STICK: u32 = 0x1000;
pub const RIGHT_STICK: u32 = 0x2000;

/// Default length of `press` without a time, in milliseconds.
pub const DEFAULT_PRESS_MS: u32 = 100;

/// What a GameInput gamepad reading holds (the fields of `GameInputGamepadState`, in its order).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PadState {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_x: f32,
    pub left_y: f32,
    pub right_x: f32,
    pub right_y: f32,
}

/// The button of a name (any case), or `None`.
pub fn button_named(name: &str) -> Option<u32> {
    Some(match name.to_ascii_lowercase().as_str() {
        "a" => A,
        "b" => B,
        "x" => X,
        "y" => Y,
        "up" | "dpadup" => DPAD_UP,
        "down" | "dpaddown" => DPAD_DOWN,
        "left" | "dpadleft" => DPAD_LEFT,
        "right" | "dpadright" => DPAD_RIGHT,
        "menu" | "start" => MENU,
        "view" | "back" | "select" => VIEW,
        "lb" | "leftshoulder" => LEFT_SHOULDER,
        "rb" | "rightshoulder" => RIGHT_SHOULDER,
        "ls" | "l3" | "leftstick" => LEFT_STICK,
        "rs" | "r3" | "rightstick" => RIGHT_STICK,
        _ => return None,
    })
}

/// One step of the script.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Command {
    Button { mask: u32, down: bool },
    Wait(u32),
    Stick { right: bool, x: f32, y: f32 },
    Trigger { right: bool, value: f32 },
    Release,
}

fn number(word: Option<&str>, what: &str) -> Result<f32, String> {
    let word = word.ok_or_else(|| format!("{what} is missing"))?;
    let v: f32 = word.parse().map_err(|_| format!("{what} {word:?} is not a number"))?;
    if v.is_finite() { Ok(v) } else { Err(format!("{what} {word:?} is not finite")) }
}

fn millis(word: Option<&str>, what: &str) -> Result<u32, String> {
    let word = word.ok_or_else(|| format!("{what} is missing"))?;
    word.parse().map_err(|_| format!("{what} {word:?} is not a whole number of milliseconds"))
}

fn side(word: Option<&str>) -> Result<bool, String> {
    match word.map(str::to_ascii_lowercase).as_deref() {
        Some("left" | "l") => Ok(false),
        Some("right" | "r") => Ok(true),
        Some(other) => Err(format!("{other:?} is not left or right")),
        None => Err("left or right is missing".to_owned()),
    }
}

/// The commands of one line: none for a blank line or a comment, several for `press` and `hold`.
pub fn parse_line(line: &str) -> Result<Vec<Command>, String> {
    let line = line.split('#').next().unwrap_or("").trim();
    let mut words = line.split_whitespace();
    let Some(first) = words.next() else { return Ok(Vec::new()) };
    let first_lower = first.to_ascii_lowercase();
    let commands = match first_lower.as_str() {
        "wait" => vec![Command::Wait(millis(words.next(), "the time")?)],
        "release" | "reset" => vec![Command::Release],
        "stick" => {
            let right = side(words.next())?;
            let x = number(words.next(), "x")?.clamp(-1.0, 1.0);
            let y = number(words.next(), "y")?.clamp(-1.0, 1.0);
            vec![Command::Stick { right, x, y }]
        }
        "trigger" => {
            let right = side(words.next())?;
            vec![Command::Trigger { right, value: number(words.next(), "the value")?.clamp(0.0, 1.0) }]
        }
        "press" | "hold" => {
            let name = words.next().ok_or("the button is missing")?;
            let mask = button_named(name).ok_or_else(|| format!("{name:?} is not a button"))?;
            let ms = match (words.next(), first_lower.as_str()) {
                (None, "press") => DEFAULT_PRESS_MS,
                (word, _) => millis(word, "the time")?,
            };
            vec![Command::Button { mask, down: true }, Command::Wait(ms), Command::Button { mask, down: false }]
        }
        _ => {
            let mask = button_named(first).ok_or_else(|| format!("{first:?} is not a command or a button"))?;
            match words.next().map(str::to_ascii_lowercase).as_deref() {
                Some("down") => vec![Command::Button { mask, down: true }],
                Some("up") => vec![Command::Button { mask, down: false }],
                Some(other) => return Err(format!("{other:?} is not down or up")),
                None => return Err("down or up is missing".to_owned()),
            }
        }
    };
    match words.next() {
        Some(extra) => Err(format!("unexpected {extra:?} at the end of the line")),
        None => Ok(commands),
    }
}

/// Plays commands in order against a clock the caller supplies (milliseconds, any origin).
#[derive(Debug, Default)]
pub struct Machine {
    queue: VecDeque<Command>,
    state: PadState,
    /// Nothing runs before this time.
    deadline: u64,
    /// A command ran since the queue was last empty, so `deadline` is the schedule and not stale.
    chained: bool,
}

impl Machine {
    pub fn new() -> Machine {
        Machine::default()
    }

    pub fn state(&self) -> PadState {
        self.state
    }

    /// Commands not run yet.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn push(&mut self, commands: impl IntoIterator<Item = Command>) {
        self.queue.extend(commands);
    }

    /// Forgets the queue and puts the pad back to neutral.
    pub fn reset(&mut self) {
        *self = Machine::default();
    }

    /// Runs every command that is due at `now`. Returns true when the state changed.
    pub fn advance(&mut self, now: u64) -> bool {
        let before = self.state;
        while now >= self.deadline || !self.chained {
            let Some(command) = self.queue.pop_front() else { break };
            if !self.chained {
                self.chained = true;
                self.deadline = now;
            }
            match command {
                Command::Wait(ms) => self.deadline += u64::from(ms),
                Command::Button { mask, down } => {
                    if down {
                        self.state.buttons |= mask;
                    } else {
                        self.state.buttons &= !mask;
                    }
                }
                Command::Stick { right, x, y } => {
                    if right {
                        (self.state.right_x, self.state.right_y) = (x, y);
                    } else {
                        (self.state.left_x, self.state.left_y) = (x, y);
                    }
                }
                Command::Trigger { right, value } => {
                    if right {
                        self.state.right_trigger = value;
                    } else {
                        self.state.left_trigger = value;
                    }
                }
                Command::Release => self.state = PadState::default(),
            }
        }
        // Everything ran and its last wait is over: the next command starts from the clock again.
        if self.queue.is_empty() && now >= self.deadline {
            self.chained = false;
        }
        self.state != before
    }
}

/// What one look at the control file found.
#[derive(Debug, Default, PartialEq)]
pub struct Found {
    /// The file shrank or vanished: start again from nothing.
    pub restarted: bool,
    pub commands: Vec<Command>,
    /// (line number from 1, what is wrong) for every line that did not parse.
    pub errors: Vec<(usize, String)>,
}

/// Reads the new complete lines of the control file on each [`Feed::poll`].
#[derive(Debug)]
pub struct Feed {
    path: PathBuf,
    offset: u64,
    partial: Vec<u8>,
    line: usize,
}

impl Feed {
    pub fn new(path: PathBuf) -> Feed {
        Feed { path, offset: 0, partial: Vec::new(), line: 0 }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn poll(&mut self) -> Found {
        let mut found = Found::default();
        let length = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
        if length < self.offset + self.partial.len() as u64 {
            self.offset = 0;
            self.partial.clear();
            self.line = 0;
            found.restarted = true;
        }
        let read_from = self.offset + self.partial.len() as u64;
        if length > read_from {
            let mut fresh = Vec::new();
            let read = std::fs::File::open(&self.path).and_then(|mut f| {
                f.seek(SeekFrom::Start(read_from))?;
                f.take(length - read_from).read_to_end(&mut fresh)
            });
            if read.is_ok() {
                self.partial.extend_from_slice(&fresh);
            }
        }
        while let Some(end) = self.partial.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = self.partial.drain(..=end).collect();
            self.offset += raw.len() as u64;
            self.line += 1;
            match parse_line(&String::from_utf8_lossy(&raw)) {
                Ok(commands) => found.commands.extend(commands),
                Err(why) => found.errors.push((self.line, why)),
            }
        }
        found
    }
}

/// A feed and a machine together: what the shim's polling thread runs.
#[derive(Debug)]
pub struct Driver {
    pub feed: Feed,
    pub machine: Machine,
}

impl Driver {
    pub fn new(path: PathBuf) -> Driver {
        Driver { feed: Feed::new(path), machine: Machine::new() }
    }

    /// One poll at time `now` (milliseconds). Returns what was found and whether the state changed.
    pub fn tick(&mut self, now: u64) -> (Found, bool) {
        let found = self.feed.poll();
        let before = self.machine.state();
        if found.restarted {
            self.machine.reset();
        }
        self.machine.push(found.commands.iter().copied());
        self.machine.advance(now);
        let changed = self.machine.state() != before;
        (found, changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn run(text: &str) -> Vec<Command> {
        text.lines().flat_map(|l| parse_line(l).unwrap()).collect()
    }

    #[test]
    fn lines_parse_into_commands() {
        assert_eq!(parse_line("A down").unwrap(), [Command::Button { mask: A, down: true }]);
        assert_eq!(parse_line("  a UP  # let go").unwrap(), [Command::Button { mask: A, down: false }]);
        assert_eq!(parse_line("Up down").unwrap(), [Command::Button { mask: DPAD_UP, down: true }]);
        assert_eq!(parse_line("wait 500").unwrap(), [Command::Wait(500)]);
        assert_eq!(parse_line("stick left 0.0 1.0").unwrap(), [Command::Stick { right: false, x: 0.0, y: 1.0 }]);
        assert_eq!(parse_line("stick R -3 0.5").unwrap(), [Command::Stick { right: true, x: -1.0, y: 0.5 }]);
        assert_eq!(parse_line("trigger right 2").unwrap(), [Command::Trigger { right: true, value: 1.0 }]);
        assert_eq!(parse_line("release").unwrap(), [Command::Release]);
        assert_eq!(
            parse_line("press Start").unwrap(),
            [Command::Button { mask: MENU, down: true }, Command::Wait(DEFAULT_PRESS_MS), Command::Button { mask: MENU, down: false }]
        );
        assert_eq!(
            parse_line("hold B 1000").unwrap(),
            [Command::Button { mask: B, down: true }, Command::Wait(1000), Command::Button { mask: B, down: false }]
        );
        assert!(parse_line("").unwrap().is_empty());
        assert!(parse_line("   # only a comment").unwrap().is_empty());
    }

    #[test]
    fn bad_lines_say_what_is_wrong() {
        for bad in ["jump", "A", "A sideways", "wait", "wait soon", "stick left 1", "stick middle 0 0", "stick left nan 0", "hold A", "press Z", "A down now", "trigger left x"] {
            assert!(parse_line(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_machine_plays_a_timeline() {
        let mut m = Machine::new();
        m.push(run("A down\nwait 500\nA up\nstick left 0 1\nwait 100\nrelease"));
        assert!(m.advance(1000));
        assert_eq!(m.state().buttons, A);
        assert!(!m.advance(1499));
        assert!(m.advance(1500));
        assert_eq!(m.state(), PadState { left_y: 1.0, ..PadState::default() });
        assert!(!m.advance(1599));
        assert!(m.advance(1600));
        assert_eq!(m.state(), PadState::default());
        assert_eq!(m.pending(), 0);
    }

    #[test]
    fn waits_chain_without_drift_and_an_idle_gap_starts_fresh() {
        let mut m = Machine::new();
        m.push(run("A down\nwait 100\nA up\nwait 100\nB down"));
        m.advance(0);
        // Polled late: the schedule still follows the file, not the poll.
        assert!(m.advance(130));
        assert_eq!(m.state().buttons, 0);
        assert!(m.advance(200));
        assert_eq!(m.state().buttons, B);
        // A long idle gap: the next command runs at once and its wait counts from then.
        m.push(run("A down\nwait 50\nA up"));
        assert!(m.advance(10_000));
        assert_eq!(m.state().buttons, B | A);
        assert!(!m.advance(10_049));
        assert!(m.advance(10_050));
        assert_eq!(m.state().buttons, B);
    }

    #[test]
    fn press_holds_the_button_for_its_time_and_presses_queue_up() {
        let mut m = Machine::new();
        m.push(run("press A 100\npress A 100"));
        assert!(m.advance(0));
        assert_eq!(m.state().buttons, A);
        m.advance(100);
        assert_eq!(m.state().buttons, A, "the second press starts as the first ends");
        assert!(m.advance(200));
        assert_eq!(m.state().buttons, 0);
    }

    #[test]
    fn reset_forgets_everything() {
        let mut m = Machine::new();
        m.push(run("A down\nwait 100\nB down"));
        m.advance(0);
        m.reset();
        assert_eq!(m.state(), PadState::default());
        assert_eq!(m.pending(), 0);
    }

    #[test]
    fn the_feed_reads_appended_lines_and_waits_for_a_complete_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pad.txt");
        let mut feed = Feed::new(path.clone());
        assert_eq!(feed.poll(), Found::default(), "no file yet");
        std::fs::write(&path, "A down\nwa").unwrap();
        assert_eq!(feed.poll().commands, [Command::Button { mask: A, down: true }]);
        std::fs::OpenOptions::new().append(true).open(&path).unwrap().write_all(b"it 5\nnonsense\nB down\n").unwrap();
        let found = feed.poll();
        assert_eq!(found.commands, [Command::Wait(5), Command::Button { mask: B, down: true }]);
        assert_eq!(found.errors.len(), 1);
        assert_eq!(found.errors[0].0, 3);
        assert_eq!(feed.poll(), Found::default(), "nothing new");
    }

    #[test]
    fn a_shorter_file_restarts_the_driver() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pad.txt");
        std::fs::write(&path, "A down\nstick left 1 0\n").unwrap();
        let mut driver = Driver::new(path.clone());
        let (_, changed) = driver.tick(0);
        assert!(changed);
        assert_eq!(driver.machine.state().buttons, A);
        std::fs::write(&path, "B down\n").unwrap();
        let (found, changed) = driver.tick(16);
        assert!(found.restarted && changed);
        assert_eq!(driver.machine.state(), PadState { buttons: B, ..PadState::default() });
        std::fs::remove_file(&path).unwrap();
        let (found, changed) = driver.tick(32);
        assert!(found.restarted && changed);
        assert_eq!(driver.machine.state(), PadState::default());
    }
}
