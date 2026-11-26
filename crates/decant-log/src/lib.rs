//! The session log of the Decant DLLs, the Rust form of the former `src/decant/sessionlog.h`.
//!
//! `DECANT_LOG_PATH` names a file (an absolute path supplied by the launcher) and every line is
//! appended to it. Without it, the legacy opt in `DECANT_LOG` appends to `decant.log` in the
//! current folder. `DECANT_LOG` also sends each line to the debugger with `OutputDebugStringA`.
//! The variables are read on every call, as the C code did, so a launcher may set them late.

pub mod calls;
pub mod handles;
pub mod rate_limit;

use std::ffi::OsString;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rate_limit::{Call, FilterAction, Limiter, rate_limit_enabled};

static LIMITER: Mutex<Limiter> = Mutex::new(Limiter::new(rate_limit::DEFAULT_WINDOW));
static SESSION_WRITER: Mutex<Option<SessionWriter>> = Mutex::new(None);

const SESSION_BUFFER_BYTES: usize = 64 * 1024;
const SESSION_FLUSH_INTERVAL: Duration = Duration::from_millis(250);

struct SessionWriter {
    path: PathBuf,
    writer: BufWriter<std::fs::File>,
    last_flush: Instant,
}

impl SessionWriter {
    fn open(path: PathBuf) -> Option<Self> {
        let mut options = std::fs::OpenOptions::new();
        options.append(true).create(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // FILE_SHARE_READ | FILE_SHARE_WRITE, as the C code opened it.
            options.share_mode(0x1 | 0x2);
        }
        let file = options.open(&path).ok()?;
        Some(Self {
            path,
            writer: BufWriter::with_capacity(SESSION_BUFFER_BYTES, file),
            last_flush: Instant::now(),
        })
    }

    fn write_message(&mut self, message: &str, force_flush: bool) {
        let _ = self.writer.write_all(message.as_bytes());
        if !message.ends_with('\n') {
            let _ = self.writer.write_all(b"\r\n");
        }
        let now = Instant::now();
        if force_flush || now.duration_since(self.last_flush) >= SESSION_FLUSH_INTERVAL {
            let _ = self.writer.flush();
            self.last_flush = now;
        }
    }

    fn flush(&mut self) {
        let _ = self.writer.flush();
        self.last_flush = Instant::now();
    }
}

fn write_session_to_path(path: Option<PathBuf>, message: &str) {
    let mut guard = SESSION_WRITER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(target) = path else {
        if let Some(mut old) = guard.take() {
            old.flush();
        }
        return;
    };
    let reopen = match guard.as_ref() {
        Some(active) => active.path != target,
        None => true,
    };
    if reopen {
        if let Some(mut old) = guard.take() {
            old.flush();
        }
        *guard = SessionWriter::open(target);
    }
    if let Some(active) = guard.as_mut() {
        active.write_message(message, rate_limit::is_urgent(message));
    }
}

fn flush_session_writer() {
    let mut guard = SESSION_WRITER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(active) = guard.as_mut() {
        active.flush();
    }
}

fn reset_session_writer() {
    let mut guard = SESSION_WRITER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut old) = guard.take() {
        old.flush();
    }
}

/// Appends `message` to the cached session writer for `path`, opening or switching files as needed.
pub fn write_session_path(path: &Path, message: &str) {
    write_session_to_path(Some(path.to_path_buf()), message);
}

/// Longest path the C reader accepted, in UTF-16 units without the final zero.
const MAX_PATH_UNITS: usize = 32767;

/// Whether `DECANT_LOG` exists, even empty.
pub fn debugger_enabled() -> bool {
    std::env::var_os("DECANT_LOG").is_some()
}

/// Where a session line goes, if anywhere.
fn session_path(log_path: Option<OsString>, legacy: bool) -> Option<PathBuf> {
    match log_path.filter(|p| !p.is_empty()) {
        Some(path) if path.encode_wide_len() > MAX_PATH_UNITS => None,
        Some(path) => Some(PathBuf::from(path)),
        None if legacy => Some(PathBuf::from("decant.log")),
        None => None,
    }
}

trait WideLen {
    fn encode_wide_len(&self) -> usize;
}

impl WideLen for OsString {
    #[cfg(windows)]
    fn encode_wide_len(&self) -> usize {
        std::os::windows::ffi::OsStrExt::encode_wide(self.as_os_str()).count()
    }
    #[cfg(not(windows))]
    fn encode_wide_len(&self) -> usize {
        self.to_string_lossy().encode_utf16().count()
    }
}

/// The bytes appended for one message: the message, then CR LF unless it already ends a line.
#[cfg(test)]
fn session_bytes(message: &str) -> Vec<u8> {
    let mut bytes = message.as_bytes().to_vec();
    if !message.ends_with('\n') {
        bytes.extend_from_slice(b"\r\n");
    }
    bytes
}

/// Appends `message` to the session log, when one is configured. Repeated lines within
/// the window are counted and summarized. Errors are ignored: logging never changes what
/// the caller does.
pub fn session_log(message: &str) {
    if !rate_limit_enabled() {
        raw_session_log(message);
        return;
    }
    let action = {
        let mut limiter = LIMITER.lock().unwrap_or_else(|e| e.into_inner());
        limiter.filter_line(Call::SessionLog, message)
    };
    match action {
        FilterAction::Write(line) => raw_session_log(&line),
        FilterAction::Suppress => {}
        FilterAction::SummaryAndWrite { summary, line } => {
            raw_session_log(&summary);
            raw_session_log(&line);
        }
        FilterAction::Summary(summary) => raw_session_log(&summary),
    }
}

/// Appends `message` directly to the session log without rate limiting.
pub fn raw_session_log(message: &str) {
    let path = session_path(std::env::var_os("DECANT_LOG_PATH"), debugger_enabled());
    write_session_to_path(path, message);
}

/// Sends `message` to the debugger when `DECANT_LOG` is set, then to the session log.
pub fn trace(message: &str) {
    if debugger_enabled() {
        debug_string(message);
    }
    session_log(message);
}

/// `OutputDebugStringA` with the message bytes, rate limited for repeated lines.
#[cfg(windows)]
pub fn debug_string(message: &str) {
    if !rate_limit_enabled() {
        raw_debug_string(message);
        return;
    }
    let action = {
        let mut limiter = LIMITER.lock().unwrap_or_else(|e| e.into_inner());
        limiter.filter_line(Call::DebugString, message)
    };
    match action {
        FilterAction::Write(line) => raw_debug_string(&line),
        FilterAction::Suppress => {}
        FilterAction::SummaryAndWrite { summary, line } => {
            raw_debug_string(&summary);
            raw_debug_string(&line);
        }
        FilterAction::Summary(summary) => raw_debug_string(&summary),
    }
}

/// No debugger channel off Windows, but rate limit state is updated.
#[cfg(not(windows))]
pub fn debug_string(message: &str) {
    if !rate_limit_enabled() {
        return;
    }
    let mut limiter = LIMITER.lock().unwrap_or_else(|e| e.into_inner());
    let _ = limiter.filter_line(Call::DebugString, message);
}

/// `OutputDebugStringA` with the message bytes without rate limiting (cut at an inner zero byte, as C would).
#[cfg(windows)]
#[allow(unsafe_code)]
pub fn raw_debug_string(message: &str) {
    let raw = message.as_bytes();
    let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    if len < 1024 {
        let mut stack = [0u8; 1024];
        stack[..len].copy_from_slice(&raw[..len]);
        stack[len] = 0;
        // SAFETY: `stack` is NUL-terminated at `len` and lives for the whole call.
        unsafe { windows_sys::Win32::System::Diagnostics::Debug::OutputDebugStringA(stack.as_ptr()) };
        return;
    }
    let mut bytes = Vec::with_capacity(len + 1);
    bytes.extend_from_slice(&raw[..len]);
    bytes.push(0);
    // SAFETY: `bytes` is a zero terminated buffer that lives for the whole call.
    unsafe { windows_sys::Win32::System::Diagnostics::Debug::OutputDebugStringA(bytes.as_ptr()) };
}

/// No debugger channel off Windows.
#[cfg(not(windows))]
pub fn raw_debug_string(_message: &str) {}

/// Flushes any pending repeated line summaries and buffered session log bytes.
pub fn flush() {
    let items = {
        let mut limiter = LIMITER.lock().unwrap_or_else(|e| e.into_inner());
        limiter.flush()
    };
    for (call, summary) in items {
        match call {
            Call::DebugString => raw_debug_string(&summary),
            Call::SessionLog => raw_session_log(&summary),
        }
    }
    flush_session_writer();
}

/// Resets the rate limiter table and closes any cached session log file after flushing it.
pub fn reset() {
    let mut limiter = LIMITER.lock().unwrap_or_else(|e| e.into_inner());
    limiter.reset();
    drop(limiter);
    reset_session_writer();
}

/// Registers the shutdown flush from the C runtime startup initializer.
#[allow(unsafe_code)]
extern "C" fn on_process_attach() {
    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }
    let _ = unsafe { atexit(on_process_exit) };
}

extern "C" fn on_process_exit() {
    flush();
}

#[used]
#[allow(unsafe_code)]
#[unsafe(link_section = ".CRT$XCU")]
static ON_PROCESS_ATTACH: extern "C" fn() = on_process_attach;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unsafe_code)]
    fn process_exit_flushes_pending_repeat_summary_and_switches_paths_cleanly() {
        let pid = std::process::id();
        let log_path = std::env::temp_dir().join(format!("decant_log_exit_{pid}.log"));
        let path_a = std::env::temp_dir().join(format!("decant_log_switch_a_{pid}.log"));
        let path_b = std::env::temp_dir().join(format!("decant_log_switch_b_{pid}.log"));
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&path_a);
        let _ = std::fs::remove_file(&path_b);
        unsafe {
            std::env::set_var("DECANT_LOG_PATH", &log_path);
            std::env::set_var("DECANT_LOG_RATE_LIMIT", "1");
        }
        reset();
        for _ in 0..4 {
            session_log("exit summary probe");
        }

        on_process_exit();

        let content = std::fs::read_to_string(&log_path).unwrap();
        assert!(content.lines().any(|line| line == "repeated 3 times"), "{content}");

        write_session_path(&path_a, "first file line 1");
        write_session_path(&path_a, "first file line 2");
        write_session_path(&path_b, "second file line 1");
        flush();

        let content_a = std::fs::read_to_string(&path_a).unwrap();
        let content_b = std::fs::read_to_string(&path_b).unwrap();
        assert_eq!(content_a, "first file line 1\r\nfirst file line 2\r\n");
        assert_eq!(content_b, "second file line 1\r\n");

        unsafe {
            std::env::set_var("DECANT_LOG_PATH", &path_a);
        }
        raw_session_log("first file line 3");
        unsafe {
            std::env::set_var("DECANT_LOG_PATH", &path_b);
        }
        raw_session_log("second file line 2");
        flush();

        let content_a = std::fs::read_to_string(&path_a).unwrap();
        let content_b = std::fs::read_to_string(&path_b).unwrap();
        assert_eq!(content_a, "first file line 1\r\nfirst file line 2\r\nfirst file line 3\r\n");
        assert_eq!(content_b, "second file line 1\r\nsecond file line 2\r\n");

        reset();
        let _ = std::fs::remove_file(log_path);
        let _ = std::fs::remove_file(path_a);
        let _ = std::fs::remove_file(path_b);
    }

    #[test]
    fn the_path_variable_wins_and_the_legacy_switch_uses_the_current_folder() {
        assert_eq!(session_path(Some("x.log".into()), false), Some(PathBuf::from("x.log")));
        assert_eq!(session_path(Some("x.log".into()), true), Some(PathBuf::from("x.log")));
        assert_eq!(session_path(None, true), Some(PathBuf::from("decant.log")));
        assert_eq!(session_path(Some("".into()), true), Some(PathBuf::from("decant.log")));
        assert_eq!(session_path(None, false), None);
        assert_eq!(session_path(Some("".into()), false), None);
    }

    #[test]
    fn a_path_longer_than_the_c_buffer_is_refused() {
        assert_eq!(session_path(Some("a".repeat(MAX_PATH_UNITS + 1).into()), true), None);
        assert!(session_path(Some("a".repeat(MAX_PATH_UNITS).into()), true).is_some());
    }

    #[test]
    fn lines_end_with_cr_lf_once() {
        assert_eq!(session_bytes("a"), b"a\r\n");
        assert_eq!(session_bytes("a\r\n"), b"a\r\n");
        assert_eq!(session_bytes("a\n"), b"a\n");
        assert_eq!(session_bytes(""), b"\r\n");
    }
}
