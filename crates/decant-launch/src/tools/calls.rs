//! The launcher's side of `DECANT_CALLS` (see `decant_log::calls`): it opens the title's shared
//! table before the title runs any code, keeps it open, and writes the final report when the
//! process has ended, when the title's own report thread can no longer run.

use std::path::{Path, PathBuf};
use std::time::Instant;

/// The report's path: `DECANT_CALLS_PATH`, else `decant_calls.txt` in the title folder (the title's
/// current folder, where its own report thread writes too).
pub fn report_path(folder: &Path, extra: &[(String, String)]) -> PathBuf {
    super::env_value(extra, decant_log::calls::PATH_ENV).filter(|p| !p.is_empty()).map_or_else(|| folder.join(decant_log::calls::FILE_NAME), PathBuf::from)
}

pub struct Calls {
    shared: decant_log::calls::Shared,
    pid: u32,
    path: PathBuf,
    start_ms: u64,
}

impl Calls {
    /// `None` unless `DECANT_CALLS` is on.
    pub fn new(pid: u32, folder: &Path, extra: &[(String, String)], start: Instant, log: &mut dyn FnMut(&str)) -> Option<Calls> {
        if !decant_log::calls::switch_on(super::env_value(extra, decant_log::calls::ENV).as_deref()) {
            return None;
        }
        let start_ms = decant_log::calls::now_ms().saturating_sub(start.elapsed().as_millis() as u64);
        let path = report_path(folder, extra);
        let Some(shared) = decant_log::calls::Shared::open(pid) else {
            log("DECANT_CALLS: cannot create the shared table, only the title's own report is written");
            return None;
        };
        log(&format!("DECANT_CALLS: counting unanswered shim calls, report in {}", path.display()));
        Some(Calls { shared, pid, path, start_ms })
    }

    /// Writes the report and puts its first lines in the log.
    pub fn exit(&mut self, log: &mut dyn FnMut(&str)) {
        let table = self.shared.table();
        let text = table.report(self.start_ms, &format!("DECANT_CALLS process {} at exit", self.pid));
        let _ = std::fs::write(&self.path, &text);
        log(&format!("DECANT_CALLS: {} distinct unanswered questions, report in {}", table.entries(), self.path.display()));
        for line in text.lines().skip(2).take(8) {
            log(&format!("  {line}"));
        }
    }
}
