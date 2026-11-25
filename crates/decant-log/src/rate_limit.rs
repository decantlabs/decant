//! Rate limiting for repeated log lines in the Decant logging system.
//!
//! Repeated lines within a time window are counted instead of being written
//! on every call. The first line is written immediately. When the window ends
//! or the text changes, a summary line stating repeated N times is written.
//! Error lines are never dropped and are always written.

use std::time::{Duration, Instant};

/// The default time window for grouping repeated lines.
pub const DEFAULT_WINDOW: Duration = Duration::from_secs(1);

/// The fixed number of slots in the rate limiter table.
pub const SLOTS: usize = 32;

/// Which log destination is being called.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Call {
    /// Debugger output via OutputDebugStringA.
    DebugString,
    /// Session log file append.
    SessionLog,
}

/// The action to take for a log message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilterAction {
    /// Write the message line directly.
    Write(String),
    /// Suppress the line because it is a repeat within the window.
    Suppress,
    /// Write the summary line first, then write the new line.
    SummaryAndWrite {
        summary: String,
        line: String,
    },
    /// Write only the summary line.
    Summary(String),
}

/// Checks whether rate limiting is enabled via the environment.
///
/// Enabled by default. Setting DECANT_LOG_RATE_LIMIT to 0 or off or false disables it.
pub fn rate_limit_enabled() -> bool {
    std::env::var("DECANT_LOG_RATE_LIMIT")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "off" | "false" | "no")
        })
        .unwrap_or(true)
}

/// Checks whether a message line carries an error.
///
/// Error lines are never suppressed or dropped.
pub fn is_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("exception")
}

/// Splits a message into a call identifier and the text key.
pub fn split_call_and_key(message: &str) -> (&str, &str) {
    let trimmed = message.trim();
    if let Some(idx) = trimmed.find('(') {
        let call = trimmed[..idx].trim();
        (call, trimmed)
    } else if let Some(idx) = trimmed.find(':') {
        let call = trimmed[..idx].trim();
        (call, trimmed)
    } else {
        ("", trimmed)
    }
}

/// One slot in the fixed rate limiter table.
#[derive(Clone, Debug)]
pub struct Slot {
    pub call: Call,
    pub call_name: String,
    pub text_key: String,
    pub count: u64,
    pub window_start: Option<Instant>,
    pub last_seen: Option<Instant>,
    pub occupied: bool,
}

impl Slot {
    /// Creates an empty slot.
    pub const fn new() -> Self {
        Self {
            call: Call::DebugString,
            call_name: String::new(),
            text_key: String::new(),
            count: 0,
            window_start: None,
            last_seen: None,
            occupied: false,
        }
    }
}

impl Default for Slot {
    fn default() -> Self {
        Self::new()
    }
}

const EMPTY_SLOT: Slot = Slot::new();

/// A fixed capacity table that rate limits repeated log lines.
pub struct Limiter {
    window: Duration,
    slots: [Slot; SLOTS],
}

impl Limiter {
    /// Creates a limiter with the given time window.
    pub const fn new(window: Duration) -> Self {
        Self {
            window,
            slots: [
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
                EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
            ],
        }
    }

    /// Filters a message line for the given call destination.
    pub fn filter_line(&mut self, call: Call, line: &str) -> FilterAction {
        self.filter_line_at(call, line, Instant::now())
    }

    /// Filters a message line with an explicit timestamp.
    pub fn filter_line_at(&mut self, call: Call, line: &str, now: Instant) -> FilterAction {
        let (call_name, key) = split_call_and_key(line);
        self.filter_at(call, call_name, key, line, now)
    }

    /// Filters with explicit call destination, call name and text key.
    pub fn filter(&mut self, call: Call, call_name: &str, text: &str) -> FilterAction {
        self.filter_at(call, call_name, text, text, Instant::now())
    }

    /// Filters with explicit timestamp.
    pub fn filter_at(
        &mut self,
        call: Call,
        call_name: &str,
        key: &str,
        full_line: &str,
        now: Instant,
    ) -> FilterAction {
        if is_error(full_line) {
            if let Some(idx) = self.find_slot(call, call_name) {
                let slot = &mut self.slots[idx];
                let repeats = slot.count.saturating_sub(1);
                slot.count = 0;
                slot.occupied = false;
                if repeats > 0 {
                    return FilterAction::SummaryAndWrite {
                        summary: format!("repeated {repeats} times"),
                        line: full_line.to_owned(),
                    };
                }
            }
            return FilterAction::Write(full_line.to_owned());
        }

        if let Some(idx) = self.find_slot(call, call_name) {
            let slot = &mut self.slots[idx];
            if slot.text_key == key {
                let elapsed = now.duration_since(slot.window_start.unwrap_or(now));
                if elapsed < self.window {
                    slot.count += 1;
                    slot.last_seen = Some(now);
                    return FilterAction::Suppress;
                }
                let repeats = slot.count.saturating_sub(1);
                slot.count = 1;
                slot.window_start = Some(now);
                slot.last_seen = Some(now);
                if repeats > 0 {
                    return FilterAction::SummaryAndWrite {
                        summary: format!("repeated {repeats} times"),
                        line: full_line.to_owned(),
                    };
                }
                return FilterAction::Write(full_line.to_owned());
            }

            let repeats = slot.count.saturating_sub(1);
            slot.text_key = key.to_owned();
            slot.count = 1;
            slot.window_start = Some(now);
            slot.last_seen = Some(now);
            if repeats > 0 {
                return FilterAction::SummaryAndWrite {
                    summary: format!("repeated {repeats} times"),
                    line: full_line.to_owned(),
                };
            }
            return FilterAction::Write(full_line.to_owned());
        }

        let idx = self.allocate_slot(now);
        let slot = &mut self.slots[idx];
        let prev_summary = if slot.occupied && slot.count > 1 {
            Some(format!("repeated {} times", slot.count - 1))
        } else {
            None
        };

        slot.call = call;
        slot.call_name = call_name.to_owned();
        slot.text_key = key.to_owned();
        slot.count = 1;
        slot.window_start = Some(now);
        slot.last_seen = Some(now);
        slot.occupied = true;

        if let Some(summary) = prev_summary {
            FilterAction::SummaryAndWrite {
                summary,
                line: full_line.to_owned(),
            }
        } else {
            FilterAction::Write(full_line.to_owned())
        }
    }

    /// Finds the slot index matching the call and call name.
    fn find_slot(&self, call: Call, call_name: &str) -> Option<usize> {
        self.slots.iter().position(|s| {
            s.occupied && s.call == call && s.call_name == call_name
        })
    }

    /// Allocates an empty slot or evicts the oldest occupied slot.
    fn allocate_slot(&self, now: Instant) -> usize {
        if let Some(empty) = self.slots.iter().position(|s| !s.occupied) {
            return empty;
        }
        let mut oldest_idx = 0;
        let mut oldest_time = now;
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(seen) = slot.last_seen {
                if seen < oldest_time {
                    oldest_time = seen;
                    oldest_idx = i;
                }
            } else {
                return i;
            }
        }
        oldest_idx
    }

    /// Flushes any pending repeat counts as summary lines.
    pub fn flush(&mut self) -> Vec<(Call, String)> {
        let mut out = Vec::new();
        for slot in &mut self.slots {
            if slot.occupied && slot.count > 1 {
                let repeats = slot.count - 1;
                out.push((slot.call, format!("repeated {repeats} times")));
                slot.count = 0;
                slot.occupied = false;
            }
        }
        out
    }

    /// Resets all slots in the table.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            *slot = Slot::new();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_lines_are_counted_not_written() {
        let mut limiter = Limiter::new(Duration::from_secs(1));
        let line = "[decant_xgr] QueryApiImpl(api XNetworking) -> 0x0";

        let first = limiter.filter_line(Call::DebugString, line);
        assert_eq!(first, FilterAction::Write(line.to_owned()));

        for _ in 0..79 {
            let next = limiter.filter_line(Call::DebugString, line);
            assert_eq!(next, FilterAction::Suppress);
        }

        let flushed = limiter.flush();
        assert_eq!(flushed, vec![(Call::DebugString, "repeated 79 times".to_owned())]);
    }

    #[test]
    fn an_error_line_is_always_written() {
        let mut limiter = Limiter::new(Duration::from_secs(1));
        let err1 = "error: file not found";
        let err2 = "failed to initialize runtime";
        let err3 = "exception 0xc0000005 access violation";

        assert_eq!(limiter.filter_line(Call::DebugString, err1), FilterAction::Write(err1.to_owned()));
        assert_eq!(limiter.filter_line(Call::DebugString, err1), FilterAction::Write(err1.to_owned()));
        assert_eq!(limiter.filter_line(Call::DebugString, err2), FilterAction::Write(err2.to_owned()));
        assert_eq!(limiter.filter_line(Call::DebugString, err3), FilterAction::Write(err3.to_owned()));
    }

    #[test]
    fn a_different_line_ends_the_run_of_repeats_with_the_summary() {
        let mut limiter = Limiter::new(Duration::from_secs(1));
        let line_a = "[decant_xgr] QueryApiImpl(api XNetworking) -> 0x0";
        let line_b = "[decant_xgr] QueryApiImpl(api XUser) -> 0x0";

        assert_eq!(limiter.filter_line(Call::DebugString, line_a), FilterAction::Write(line_a.to_owned()));
        assert_eq!(limiter.filter_line(Call::DebugString, line_a), FilterAction::Suppress);
        assert_eq!(limiter.filter_line(Call::DebugString, line_a), FilterAction::Suppress);

        let action = limiter.filter_line(Call::DebugString, line_b);
        assert_eq!(
            action,
            FilterAction::SummaryAndWrite {
                summary: "repeated 2 times".to_owned(),
                line: line_b.to_owned(),
            }
        );
    }

    #[test]
    fn window_expiration_writes_summary_and_starts_new_window() {
        let mut limiter = Limiter::new(Duration::from_millis(50));
        let line = "test_call: message";
        let t0 = Instant::now();

        assert_eq!(
            limiter.filter_line_at(Call::SessionLog, line, t0),
            FilterAction::Write(line.to_owned())
        );
        assert_eq!(
            limiter.filter_line_at(Call::SessionLog, line, t0 + Duration::from_millis(10)),
            FilterAction::Suppress
        );

        let action = limiter.filter_line_at(Call::SessionLog, line, t0 + Duration::from_millis(60));
        assert_eq!(
            action,
            FilterAction::SummaryAndWrite {
                summary: "repeated 1 times".to_owned(),
                line: line.to_owned(),
            }
        );
    }

    #[test]
    fn different_calls_do_not_interfere() {
        let mut limiter = Limiter::new(Duration::from_secs(1));
        let call_a = "[decant_xgr] QueryApiImpl(api A)";
        let call_b = "gameinput: GetCurrentReading";

        assert_eq!(limiter.filter_line(Call::DebugString, call_a), FilterAction::Write(call_a.to_owned()));
        assert_eq!(limiter.filter_line(Call::DebugString, call_b), FilterAction::Write(call_b.to_owned()));

        assert_eq!(limiter.filter_line(Call::DebugString, call_a), FilterAction::Suppress);
        assert_eq!(limiter.filter_line(Call::DebugString, call_b), FilterAction::Suppress);

        let flushed = limiter.flush();
        assert_eq!(flushed.len(), 2);
    }
}
