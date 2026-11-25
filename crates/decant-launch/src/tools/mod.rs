//! Debugging switches of a title run, each behind an environment variable and off by default. The
//! launcher is the title's debugger, so the switches that need the title's threads (data
//! breakpoints, stacks, waits) live here, on the debugger's side: it sees every thread from its
//! first instruction, receives every exception before the title does, and can stop and read a
//! thread without running any code inside the title. The switches that need to act inside the
//! title (key presses, frame capture, the count of unanswered calls) live in the Decant DLLs.
//! `docs/DEBUG_TOOLS.md` describes them all.
//!
//! [`Tools`] is the one hook point of the debug loop in `run.rs`: it is told about threads,
//! modules, debug strings and exceptions, and polled between events. Every switch off, it does
//! nothing.

#[cfg(windows)]
pub mod calls;
pub mod snapshot;
pub mod stacks;
pub mod unwind;
pub mod waits;
pub mod watch;

/// The value of `key`: the last one in `extra` (the title's own environment), else this process's.
pub fn env_value(extra: &[(String, String)], key: &str) -> Option<String> {
    extra.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v.clone()).or_else(|| std::env::var(key).ok())
}

#[cfg(all(windows, target_arch = "x86_64"))]
pub use platform::Tools;

#[cfg(all(windows, target_arch = "x86_64"))]
pub use platform::{ThreadInfo, Threads};

#[cfg(all(windows, target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod platform {
    use super::unwind::{Memory, Modules, platform::Process};
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::time::Instant;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, LocalFree};
    use windows_sys::Win32::System::Threading::{GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};

    /// A thread of the title as the debug events announced it.
    pub struct ThreadInfo {
        /// The handle of the debug event: get and set context, suspend and resume.
        pub handle: HANDLE,
        /// A name the title gave with the old `RaiseException(0x406D1388)` convention.
        pub raised_name: Option<String>,
    }

    /// The title's live threads, by id.
    #[derive(Default)]
    pub struct Threads {
        map: BTreeMap<u32, ThreadInfo>,
    }

    impl Threads {
        pub fn iter(&self) -> impl Iterator<Item = (&u32, &ThreadInfo)> {
            self.map.iter()
        }

        pub fn get(&self, tid: u32) -> Option<&ThreadInfo> {
            self.map.get(&tid)
        }

        /// The name of the thread: `SetThreadDescription`, else the raised one.
        pub fn name(&self, tid: u32) -> Option<String> {
            description(tid).or_else(|| self.map.get(&tid).and_then(|t| t.raised_name.clone()))
        }

        /// `thread 1234 "name"`, for the log.
        pub fn label(&self, tid: u32) -> String {
            match self.name(tid) {
                Some(n) => format!("thread {tid} \"{n}\""),
                None => format!("thread {tid}"),
            }
        }
    }

    /// The description the thread set for itself, if any.
    fn description(tid: u32) -> Option<String> {
        // SAFETY: a plain handle request.
        let handle = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) };
        if handle.is_null() {
            return None;
        }
        let mut text: *mut u16 = std::ptr::null_mut();
        // SAFETY: the call allocates the string with LocalAlloc, freed below.
        let hr = unsafe { GetThreadDescription(handle, &mut text) };
        let mut out = None;
        if hr >= 0 && !text.is_null() {
            // SAFETY: a NUL terminated string the call gave.
            let len = (0..).take_while(|&i| unsafe { *text.add(i) } != 0).count();
            // SAFETY: `len` units before the NUL.
            let s = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, len) });
            if !s.is_empty() {
                out = Some(s);
            }
        }
        // SAFETY: memory and handle the calls gave us.
        unsafe {
            if !text.is_null() {
                LocalFree(text.cast());
            }
            CloseHandle(handle);
        }
        out
    }

    /// The switches of one run.
    pub struct Tools {
        mem: Process,
        /// Any switch that needs the threads and modules is on.
        active: bool,
        threads: Threads,
        modules: Modules,
        calls: Option<super::calls::Calls>,
        watch: Option<super::watch::platform::Watch>,
        stacks: Option<super::stacks::platform::Stacks>,
        waits: Option<super::waits::platform::Waits>,
        /// Who created which handle, from the `[decant-handle]` lines of our DLLs.
        creators: super::snapshot::platform::Creators,
        start: Instant,
    }

    impl Tools {
        /// Reads the switches for the process `pid` just created, run from `folder`.
        pub fn new(pid: u32, process: HANDLE, folder: &Path, extra: &[(String, String)], start: Instant, log: &mut dyn FnMut(&str)) -> Tools {
            let calls = super::calls::Calls::new(pid, folder, extra, start, log);
            let watch = super::watch::platform::Watch::new(super::env_value(extra, "DECANT_WATCH"), super::env_value(extra, "DECANT_WATCH_READ"), log);
            let stacks = super::stacks::platform::Stacks::new(super::env_value(extra, "DECANT_STACKS"), folder, log);
            let waits = super::waits::platform::Waits::new(super::env_value(extra, "DECANT_WAITS"), folder, log);
            let active = watch.is_some() || stacks.is_some() || waits.is_some();
            Tools { mem: Process(process), active, threads: Threads::default(), modules: Modules::default(), calls, watch, stacks, waits, creators: Default::default(), start }
        }

        /// The longest the debug loop may wait for an event before [`Tools::poll`] is due.
        pub fn wait_ms(&self) -> Option<u32> {
            self.stacks.as_ref().and_then(|s| s.due_in_ms(self.start))
        }

        pub fn thread(&mut self, _process: HANDLE, tid: u32, handle: HANDLE, _log: &mut dyn FnMut(&str)) {
            if !self.active {
                return;
            }
            self.threads.map.insert(tid, ThreadInfo { handle, raised_name: None });
            if let Some(w) = &self.watch {
                w.arm(handle);
            }
        }

        pub fn thread_gone(&mut self, tid: u32) {
            self.threads.map.remove(&tid);
        }

        pub fn module(&mut self, _process: HANDLE, base: usize, name: &str, log: &mut dyn FnMut(&str)) {
            if !self.active {
                return;
            }
            self.modules.add(&self.mem, base, name);
            if let Some(w) = self.watch.as_mut() {
                w.module(&self.mem, base, name, &self.threads, log);
            }
        }

        pub fn unload(&mut self, base: usize) {
            if self.active {
                self.modules.remove(base);
            }
        }

        /// A debug line of the title: a `[decant-handle]` line of our DLLs names a handle's creator.
        pub fn debug_string(&mut self, text: &str) {
            if !self.active || !text.contains(decant_log::handles::PREFIX) {
                return;
            }
            for line in text.lines() {
                if let Some((handle, who)) = decant_log::handles::parse(line) {
                    self.creators.insert(handle, who);
                }
            }
        }

        /// An exception of thread `tid`: true when it was one of ours and the thread goes on as if
        /// nothing happened.
        #[allow(clippy::too_many_arguments)]
        pub fn exception(&mut self, _process: HANDLE, tid: u32, code: u32, _address: usize, info: &[usize], first_chance: bool, log: &mut dyn FnMut(&str)) -> bool {
            if !self.active || !first_chance {
                return false;
            }
            match code {
                // A data breakpoint.
                0x8000_0004 => self.watch.as_mut().is_some_and(|w| w.single_step(&self.mem, &self.modules, &self.threads, tid, log)),
                // A thread name, the old way: kept, the title's own handler still runs.
                0x406D_1388 if info.len() >= 3 && info[0] == 0x1000 => {
                    let target = if info[2] as u32 == u32::MAX { tid } else { info[2] as u32 };
                    let mut name = [0u8; 128];
                    let n = self.mem.read(info[1], &mut name);
                    let end = name[..n].iter().position(|&b| b == 0).unwrap_or(n);
                    if let Some(t) = self.threads.map.get_mut(&target) {
                        t.raised_name = Some(String::from_utf8_lossy(&name[..end]).into_owned());
                    }
                    false
                }
                _ => false,
            }
        }

        pub fn poll(&mut self, _process: HANDLE, log: &mut dyn FnMut(&str)) {
            if let Some(s) = self.stacks.as_mut() {
                s.poll(self.start, &self.mem, &self.modules, &self.threads, &self.creators, log);
            }
            if let Some(w) = self.waits.as_mut() {
                w.poll(self.start, &self.mem, &self.modules, &self.threads, &self.creators, log);
            }
        }

        /// The process ended: the final reports.
        pub fn exit(&mut self, log: &mut dyn FnMut(&str)) {
            if let Some(w) = &self.watch {
                w.exit(log);
            }
            if let Some(w) = &self.waits {
                w.exit(log);
            }
            if let Some(c) = self.calls.as_mut() {
                c.exit(log);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_title_environment_wins() {
        let extra = vec![("decant_x_test".to_owned(), "a".to_owned()), ("DECANT_X_TEST".to_owned(), "b".to_owned())];
        assert_eq!(env_value(&extra, "DECANT_X_TEST").as_deref(), Some("b"));
        assert_eq!(env_value(&[], "DECANT_X_TEST_NOT_SET_ANYWHERE"), None);
    }
}
