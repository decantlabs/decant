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
    key_with(modules, rip, stack, |_| None)
}

/// Like [`key`], using `resolve` to turn an address inside a module into a symbol name when available.
pub fn key_with<F>(modules: &[Module], rip: usize, stack: &[usize], mut resolve: F) -> String
where
    F: FnMut(usize) -> Option<String>,
{
    let head = resolve(rip).or_else(|| inside(modules, rip)).unwrap_or_else(|| format!("{rip:#x}"));
    let mut parts = Vec::with_capacity(1 + DEPTH);
    parts.push(head);
    for &w in stack {
        if parts.len() > DEPTH {
            break;
        }
        if let Some(fallback) = inside(modules, w) {
            parts.push(resolve(w).unwrap_or(fallback));
        }
    }
    parts.join(" < ")
}

fn is_ntdll_top(key: &str) -> bool {
    key.starts_with("ntdll.dll+") || key.starts_with("ntdll!")
}

fn is_idle_top(key: &str) -> bool {
    is_ntdll_top(key) || key.starts_with("win32u.dll+") || key.starts_with("win32u!")
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
        use std::fmt::Write as _;
        let mut threads: Vec<(u32, HashMap<String, u64>)> = self.counts.drain().collect();
        let busy = |m: &HashMap<String, u64>| m.iter().filter(|(k, _)| !is_ntdll_top(k)).map(|(_, n)| *n).sum::<u64>();
        threads.sort_by_key(|(tid, m)| (std::cmp::Reverse(busy(m)), *tid));
        let mut out = format!("{} rounds\n", self.ticks);
        for (tid, m) in threads {
            let total: u64 = m.values().sum();
            let _ = writeln!(out, "thread {tid}: {total} samples, {} outside ntdll", busy(&m));
            let mut keys: Vec<(String, u64)> = m.into_iter().collect();
            keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (k, n) in keys.into_iter().take(top) {
                let _ = writeln!(out, "  {:5.1}% {n:6} {k}", 100.0 * n as f64 / total as f64);
            }
        }
        self.ticks = 0;
        out
    }

    /// Windowed report for `[from_s..to_s]`: aggregated callstacks across all threads followed by
    /// per-thread top callstacks (busiest threads first). Empties the profile.
    pub fn take_window_report(&mut self, from_s: f64, to_s: f64, top_threads: usize, top_per_thread: usize, top_agg: usize) -> String {
        use std::fmt::Write as _;
        let mut threads: Vec<(u32, HashMap<String, u64>)> = self.counts.drain().collect();
        let busy = |m: &HashMap<String, u64>| m.iter().filter(|(k, _)| !is_ntdll_top(k)).map(|(_, n)| *n).sum::<u64>();
        let mut agg: HashMap<String, u64> = HashMap::new();
        let mut total_all = 0u64;
        let mut busy_all = 0u64;
        for (_, m) in &threads {
            for (k, &n) in m {
                total_all += n;
                if !is_ntdll_top(k) {
                    busy_all += n;
                }
                *agg.entry(k.clone()).or_default() += n;
            }
        }
        let mut out = format!(
            "--- window {from_s:.1}..{to_s:.1}s ({} rounds, {total_all} samples, {busy_all} outside ntdll) ---\n",
            self.ticks
        );
        if total_all > 0 {
            let mut agg_keys: Vec<(String, u64)> = agg.into_iter().collect();
            agg_keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let _ = writeln!(out, "aggregated (all threads, top {top_agg}):");
            for (k, n) in agg_keys.iter().take(top_agg) {
                let _ = writeln!(out, "  {:5.1}% {n:6} {k}", 100.0 * *n as f64 / total_all as f64);
            }
            let active_total: u64 = agg_keys.iter().filter(|(k, _)| !is_idle_top(k)).map(|(_, n)| *n).sum();
            if active_total > 0 {
                let _ = writeln!(out, "active (outside ntdll/win32u, top 20):");
                for (k, n) in agg_keys.iter().filter(|(k, _)| !is_idle_top(k)).take(20) {
                    let _ = writeln!(out, "  {:5.1}% {n:6} {k}", 100.0 * *n as f64 / active_total as f64);
                }
            }
        }
        threads.sort_by_key(|(tid, m)| (std::cmp::Reverse(busy(m)), *tid));
        for (tid, m) in threads.into_iter().take(top_threads) {
            let total: u64 = m.values().sum();
            let _ = writeln!(out, "thread {tid}: {total} samples, {} outside ntdll", busy(&m));
            let mut keys: Vec<(String, u64)> = m.into_iter().collect();
            keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (k, n) in keys.into_iter().take(top_per_thread) {
                let _ = writeln!(out, "  {:5.1}% {n:6} {k}", 100.0 * n as f64 / total as f64);
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

    #[test]
    fn windowed_report_aggregates_across_threads_and_orders_busy_first() {
        let mut p = Profile::default();
        p.add(7, "ntdll!NtWaitForSingleObject+0x14".into());
        p.add(7, "ntdll!NtWaitForSingleObject+0x14".into());
        p.add(9, "t.exe!work+0x20 < ntdll!RtlUserThreadStart+0x21".into());
        p.add(9, "ntdll!NtWaitForSingleObject+0x14".into());
        p.tick();
        p.tick();
        let r = p.take_window_report(30.0, 60.0, 10, 5, 10);
        assert!(r.starts_with("--- window 30.0..60.0s (2 rounds, 4 samples, 1 outside ntdll) ---"), "{r}");
        assert!(r.contains("aggregated (all threads, top 10):"), "{r}");
        assert!(r.contains(" 75.0%      3 ntdll!NtWaitForSingleObject+0x14"), "{r}");
        assert!(r.contains(" 25.0%      1 t.exe!work+0x20 < ntdll!RtlUserThreadStart+0x21"), "{r}");
        let (nine, seven) = (r.find("thread 9:").unwrap(), r.find("thread 7:").unwrap());
        assert!(nine < seven, "{r}");
    }

    #[test]
    fn key_with_prefers_resolved_symbol_when_inside_module() {
        let modules = vec![(0x1000, 0x1000, "a.dll".to_owned()), (0x10_0000, 0x10_0000, "t.exe".to_owned())];
        let k = key_with(&modules, 0x1010, &[0x10_0020, 0x1ff0], |addr| {
            (addr == 0x1010).then(|| "a!foo+0x10".to_owned())
        });
        assert_eq!(k, "a!foo+0x10 < t.exe+0x20 < a.dll+0xff0");
    }

    #[test]
    fn active_section_excludes_ntdll_and_win32u_idle_stacks() {
        let mut p = Profile::default();
        p.add(1, "ntdll!NtWaitForSingleObject+0x14".into());
        p.add(2, "win32u.dll+0x100".into());
        p.add(3, "win32u!NtUserMsgWaitForMultipleObjectsEx+0x14".into());
        p.add(4, "d3d12_xs!busy+0x10".into());
        p.add(4, "d3d12_xs!busy+0x10".into());
        p.add(5, "xmem!work+0x20".into());
        p.tick();
        let r = p.take_window_report(0.0, 30.0, 5, 5, 10);
        let active_idx = r.find("active (outside ntdll/win32u, top 20):").expect("active section missing");
        let thread_idx = r.find("thread 4:").expect("thread section missing");
        let active_sec = &r[active_idx..thread_idx];
        assert!(active_sec.contains(" 66.7%      2 d3d12_xs!busy+0x10"), "{active_sec}");
        assert!(active_sec.contains(" 33.3%      1 xmem!work+0x20"), "{active_sec}");
        assert!(!active_sec.contains("ntdll!") && !active_sec.contains("win32u!") && !active_sec.contains("win32u.dll"), "{active_sec}");
    }
}
