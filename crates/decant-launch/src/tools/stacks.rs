//! `DECANT_STACKS=<seconds>,...`: at each of those times since the start, the stack of every thread of
//! the title, with its name and the wait it sits in, goes to `decant_stacks.txt` in the title folder.

/// The times of the list, sorted, in seconds. Fractions are allowed (`0.5`).
pub fn parse(value: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let s: f64 = item.trim_end_matches('s').parse().map_err(|_| format!("{item}: not a number of seconds"))?;
        if !s.is_finite() || s < 0.0 {
            return Err(format!("{item}: not a number of seconds"));
        }
        out.push(s);
    }
    out.sort_by(f64::total_cmp);
    Ok(out)
}

/// The file the pictures go to.
pub const FILE_NAME: &str = "decant_stacks.txt";

#[cfg(all(windows, target_arch = "x86_64"))]
pub mod platform {
    use super::*;
    use crate::tools::Threads;
    use crate::tools::snapshot::platform::{Creators, format, take};
    use crate::tools::unwind::{Modules, platform::Process};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Frames kept per thread.
    const FRAMES: usize = 48;

    pub struct Stacks {
        times: Vec<f64>,
        next: usize,
        path: PathBuf,
    }

    impl Stacks {
        pub fn new(value: Option<String>, folder: &std::path::Path, log: &mut dyn FnMut(&str)) -> Option<Stacks> {
            let value = value.filter(|v| !v.trim().is_empty())?;
            match parse(&value) {
                Ok(times) if !times.is_empty() => {
                    let path = folder.join(FILE_NAME);
                    let _ = std::fs::write(&path, "");
                    log(&format!("DECANT_STACKS: all thread stacks at {value} s into {}", path.display()));
                    Some(Stacks { times, next: 0, path })
                }
                Ok(_) => None,
                Err(e) => {
                    log(&format!("DECANT_STACKS: {e}, ignored"));
                    None
                }
            }
        }

        /// The milliseconds until the next picture is due.
        pub fn due_in_ms(&self, start: Instant) -> Option<u32> {
            let t = *self.times.get(self.next)?;
            Some(((t - start.elapsed().as_secs_f64()) * 1000.0).clamp(1.0, 100.0) as u32)
        }

        pub fn poll(&mut self, start: Instant, mem: &Process, modules: &Modules, threads: &Threads, creators: &Creators, log: &mut dyn FnMut(&str)) {
            let now = start.elapsed().as_secs_f64();
            if self.times.get(self.next).is_none_or(|&t| now < t) {
                return;
            }
            while self.times.get(self.next).is_some_and(|&t| now >= t) {
                self.next += 1;
            }
            let states = take(mem, modules, threads, FRAMES, creators);
            let waiting = states.iter().filter(|s| s.blocked.is_some()).count();
            let text = format!("--- at {now:.3} s: {} threads, {waiting} waiting\n{}\n", states.len(), format(&states));
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&self.path) {
                let _ = f.write_all(text.as_bytes());
            }
            log(&format!("DECANT_STACKS: {} threads at {now:.3} s ({waiting} waiting) in {}", states.len(), self.path.display()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_parse_and_sort() {
        assert_eq!(parse("30, 5,0.5s").unwrap(), vec![0.5, 5.0, 30.0]);
        assert!(parse("x").is_err());
        assert!(parse("-1").is_err());
        assert_eq!(parse("").unwrap(), Vec::<f64>::new());
    }
}
