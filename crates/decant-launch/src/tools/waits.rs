//! `DECANT_WAITS=<seconds>`: every thread that sits in the same wait (same wait, same stack) for longer
//! than that is reported once, with what it waits for and, for an event or semaphore one of our DLLs
//! created, who created it. The launcher looks at all threads about once a second.

/// The threshold in seconds, `None` when absent or not a positive number.
pub fn threshold(value: Option<&str>) -> Option<f64> {
    let s: f64 = value?.trim().trim_end_matches('s').parse().ok()?;
    (s.is_finite() && s > 0.0).then_some(s)
}

/// How often the threads are looked at for a threshold of `s` seconds.
pub fn interval(s: f64) -> f64 {
    (s / 4.0).clamp(0.1, 1.0)
}

/// The file the reports also go to, with full stacks.
pub const FILE_NAME: &str = "decant_waits.txt";

#[cfg(all(windows, target_arch = "x86_64"))]
pub mod platform {
    use super::*;
    use crate::tools::Threads;
    use crate::tools::snapshot::platform::{Creators, take};
    use crate::tools::unwind::{Modules, platform::Process};
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    const FRAMES: usize = 48;

    struct Seen {
        key: String,
        since: f64,
        reported: bool,
        text: String,
    }

    pub struct Waits {
        threshold: f64,
        every: f64,
        next: f64,
        seen: HashMap<u32, Seen>,
        path: PathBuf,
    }

    impl Waits {
        pub fn new(value: Option<String>, folder: &std::path::Path, log: &mut dyn FnMut(&str)) -> Option<Waits> {
            let value = value.filter(|v| !v.trim().is_empty())?;
            let Some(threshold) = threshold(Some(&value)) else {
                log(&format!("DECANT_WAITS: {value} is not a number of seconds, ignored"));
                return None;
            };
            let path = folder.join(FILE_NAME);
            let _ = std::fs::write(&path, "");
            log(&format!("DECANT_WAITS: threads in the same wait for more than {threshold} s are reported here and in {}", path.display()));
            Some(Waits { threshold, every: interval(threshold), next: 0.0, seen: HashMap::new(), path })
        }

        pub fn poll(&mut self, start: Instant, mem: &Process, modules: &Modules, threads: &Threads, creators: &Creators, log: &mut dyn FnMut(&str)) {
            let now = start.elapsed().as_secs_f64();
            if now < self.next {
                return;
            }
            self.next = now + self.every;
            let states = take(mem, modules, threads, FRAMES, creators);
            let alive: Vec<u32> = states.iter().map(|s| s.tid).collect();
            self.seen.retain(|tid, _| alive.contains(tid));
            for s in states {
                let Some(blocked) = &s.blocked else {
                    self.seen.remove(&s.tid);
                    continue;
                };
                let entry = self.seen.entry(s.tid).or_insert_with(|| Seen { key: s.key.clone(), since: now, reported: false, text: String::new() });
                if entry.key != s.key {
                    *entry = Seen { key: s.key.clone(), since: now, reported: false, text: String::new() };
                }
                let waited = now - entry.since;
                if !entry.reported && waited >= self.threshold {
                    entry.reported = true;
                    // The first frames outside the system modules say where the title waits.
                    let title: Vec<&str> = s.frames.iter().filter(|f| !f.contains('!')).take(6).map(String::as_str).collect();
                    entry.text = format!("{} in {blocked}", s.label);
                    log(&format!("DECANT_WAITS: {} has waited {waited:.1} s in {blocked}, from {}", s.label, title.join(" < ")));
                    let block = format!("--- at {now:.3} s: {} has waited {waited:.1} s in {blocked}\n{}\n", s.label, s.frames.iter().map(|f| format!("    {f}\n")).collect::<String>());
                    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&self.path) {
                        let _ = f.write_all(block.as_bytes());
                    }
                }
            }
        }

        pub fn exit(&self, log: &mut dyn FnMut(&str)) {
            let mut still: Vec<&Seen> = self.seen.values().filter(|s| s.reported).collect();
            still.sort_by(|a, b| a.since.total_cmp(&b.since));
            for s in still {
                log(&format!("DECANT_WAITS: in that wait since {:.1} s at the last look before the exit: {}", s.since, s.text));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_threshold_is_a_positive_number_of_seconds() {
        assert_eq!(threshold(Some("10")), Some(10.0));
        assert_eq!(threshold(Some("2.5s")), Some(2.5));
        assert_eq!(threshold(Some("0")), None);
        assert_eq!(threshold(Some("x")), None);
        assert_eq!(threshold(None), None);
        assert_eq!(interval(10.0), 1.0);
        assert_eq!(interval(1.0), 0.25);
        assert_eq!(interval(0.1), 0.1);
    }
}
