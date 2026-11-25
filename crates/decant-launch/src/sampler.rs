//! Where the title's threads spend their time, behind `DECANT_SAMPLE_MS`: every that many
//! milliseconds the debugger stops each thread for a moment and notes where it is (the module and
//! offset of its instruction pointer, and the first stack words that fall inside a loaded module,
//! the likely return addresses). Every 30 seconds the counts go to a report file next to the log,
//! one block per thread, so a thread that waits shows where it waits and a busy one where it works.
//!
//! Off by default. Off, nothing here runs.

use std::collections::HashMap;
use std::time::Duration;

/// The environment switch: the sampling interval in milliseconds.
pub const ENV: &str = "DECANT_SAMPLE_MS";

/// Stack words kept per sample after the instruction pointer.
pub const DEPTH: usize = 10;

/// The sampling interval for the value of [`ENV`], `None` when absent, zero or not a number.
pub fn interval(value: Option<&str>) -> Option<Duration> {
    let ms: u64 = value?.trim().parse().ok()?;
    (ms > 0).then(|| Duration::from_millis(ms))
}

/// A loaded module: base, size of its image, file name.
pub type Module = (usize, usize, String);

/// `name+offset` when `address` lies inside one of `modules`, else `None`.
pub fn inside(modules: &[Module], address: usize) -> Option<String> {
    modules.iter().find(|(base, size, _)| (*base..base.saturating_add(*size)).contains(&address)).map(|(base, _, name)| format!("{name}+{:#x}", address - base))
}

/// The key of one sample: the instruction pointer, then up to [`DEPTH`] stack words inside a module.
pub fn key(modules: &[Module], rip: usize, stack: &[usize]) -> String {
    let mut parts = vec![inside(modules, rip).unwrap_or_else(|| format!("{rip:#x}"))];
    parts.extend(stack.iter().filter_map(|w| inside(modules, *w)).take(DEPTH));
    parts.join(" < ")
}

/// Sample counts per thread and key.
#[derive(Default)]
pub struct Profile {
    counts: HashMap<u32, HashMap<String, u64>>,
    ticks: u64,
}

impl Profile {
    /// One sample of thread `tid`.
    pub fn add(&mut self, tid: u32, key: String) {
        *self.counts.entry(tid).or_default().entry(key).or_default() += 1;
    }

    /// One sampling round done.
    pub fn tick(&mut self) {
        self.ticks += 1;
    }

    /// The report: one block per thread (the busiest first, busy meaning samples outside
    /// `ntdll.dll`), each with its `top` most frequent keys and their share. Empties the profile.
    pub fn take_report(&mut self, top: usize) -> String {
        let mut threads: Vec<(u32, HashMap<String, u64>)> = self.counts.drain().collect();
        let busy = |m: &HashMap<String, u64>| m.iter().filter(|(k, _)| !k.starts_with("ntdll.dll+")).map(|(_, n)| *n).sum::<u64>();
        threads.sort_by_key(|(tid, m)| (std::cmp::Reverse(busy(m)), *tid));
        let mut out = format!("{} rounds\n", self.ticks);
        for (tid, m) in threads {
            let total: u64 = m.values().sum();
            out += &format!("thread {tid}: {total} samples, {} outside ntdll\n", busy(&m));
            let mut keys: Vec<(String, u64)> = m.into_iter().collect();
            keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (k, n) in keys.into_iter().take(top) {
                out += &format!("  {:5.1}% {n:6} {k}\n", 100.0 * n as f64 / total as f64);
            }
        }
        self.ticks = 0;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_interval_needs_a_positive_number() {
        assert_eq!(interval(Some("20")), Some(Duration::from_millis(20)));
        assert_eq!(interval(Some("0")), None);
        assert_eq!(interval(Some("x")), None);
        assert_eq!(interval(None), None);
    }

    #[test]
    fn a_key_names_the_instruction_and_the_stack_words_inside_modules() {
        let modules = vec![(0x1000, 0x1000, "a.dll".to_owned()), (0x10_0000, 0x10_0000, "t.exe".to_owned())];
        assert_eq!(key(&modules, 0x1010, &[5, 0x10_0020, 0x3000, 0x1ff0]), "a.dll+0x10 < t.exe+0x20 < a.dll+0xff0");
        assert_eq!(key(&modules, 0x9, &[]), "0x9");
    }

    #[test]
    fn the_report_puts_the_busy_thread_first_and_empties_the_profile() {
        let mut p = Profile::default();
        p.add(7, "ntdll.dll+0x10".into());
        p.add(7, "ntdll.dll+0x10".into());
        p.add(9, "t.exe+0x20".into());
        p.add(9, "ntdll.dll+0x10".into());
        p.tick();
        let r = p.take_report(5);
        let (nine, seven) = (r.find("thread 9:").unwrap(), r.find("thread 7:").unwrap());
        assert!(nine < seven, "{r}");
        assert!(r.contains(" 50.0%      1 t.exe+0x20"), "{r}");
        assert!(r.starts_with("1 rounds"), "{r}");
        assert!(p.take_report(5).starts_with("0 rounds\n"), "emptied");
    }
}
