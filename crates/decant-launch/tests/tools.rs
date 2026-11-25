//! The debugging switches of `decant_launch::tools`, run against the stand-in title `tools_probe`.

#![cfg(all(windows, target_arch = "x86_64"))]

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use decant_launch::run::run_cancellable;

/// A title folder of its own with a copy of the probe: each test has its own report files, so tests
/// running at the same time never read or remove each other's.
fn folder() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let folder = std::fs::canonicalize(dir.path()).unwrap();
    let folder = folder.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(folder.clone(), PathBuf::from);
    let exe = folder.join("tools_probe.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_tools_probe"), &exe).unwrap();
    (dir, exe)
}

fn run(exe: &Path, args: &[&str], env: &[(&str, &str)]) -> (Option<u32>, Vec<String>) {
    let mut log: Vec<String> = Vec::new();
    let args: Vec<String> = args.iter().map(|s| (*s).to_owned()).collect();
    let env: Vec<(String, String)> = env.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    let outcome = run_cancellable(exe, &args, exe.parent().unwrap(), &env, Some(DEADLINE), &AtomicBool::new(false), &mut log).unwrap();
    (outcome.exit_code, log)
}

#[test]
fn decant_calls_reports_each_unanswered_question_at_exit() {
    let (_dir, exe) = folder();
    let (code, log) = run(&exe, &["calls"], &[("DECANT_CALLS", "1")]);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("DECANT_CALLS: 2 distinct unanswered questions")), "{log:#?}");
    let report = std::fs::read_to_string(exe.parent().unwrap().join("decant_calls.txt")).unwrap();
    assert!(report.contains("      3  not implemented  tools_probe.exe!probe question  <-  "), "{report}");
    assert!(report.contains("      1  stub value       tools_probe.exe!probe stub"), "{report}");
}

/// The offset of the probe's watched value in its image, as the probe itself prints it.
fn watched_rva(exe: &Path) -> String {
    let out = std::process::Command::new(exe).arg("rva").output().unwrap();
    String::from_utf8(out.stdout).unwrap().trim().to_owned()
}

#[test]
fn decant_watch_logs_each_change_with_the_writer_and_its_stack() {
    let (_dir, exe) = folder();
    let rva = watched_rva(&exe);
    let (code, log) = run(&exe, &["watch"], &[("DECANT_WATCH", &format!("{rva}:8"))]);
    assert_eq!(code, Some(3), "{log:#?}");
    assert!(log.iter().any(|l| l.contains(&format!("DECANT_WATCH: #0 exe+{rva} at")) && l.contains("8 bytes, on writes, now 0x0")), "{log:#?}");
    for change in ["0x0 -> 0x1", "0x1 -> 0x2", "0x2 -> 0x3"] {
        let line = log.iter().find(|l| l.contains(change)).unwrap_or_else(|| panic!("{change}: {log:#?}"));
        assert!(line.contains("\"probe writer\": tools_probe.exe+0x"), "{line}");
        assert!(line.matches(" < ").count() >= 2, "a stack of several frames: {line}");
    }
    assert!(log.iter().any(|l| l.contains("3 hits, 3 changes, last value 0x3")), "{log:#?}");
}

#[test]
fn decant_watch_read_also_sees_the_reads() {
    let (_dir, exe) = folder();
    let rva = watched_rva(&exe);
    let (code, log) = run(&exe, &["watch"], &[("DECANT_WATCH_READ", &format!("exe+{rva}"))]);
    assert_eq!(code, Some(3), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("on reads and writes")), "{log:#?}");
    // The main thread reads the value for its exit code.
    assert!(log.iter().any(|l| l.contains("0x3 (unchanged) by thread")), "{log:#?}");
}

/// How long a test waits for what it looks for. Generous: the machine may be busy with many builds.
const DEADLINE: Duration = Duration::from_secs(150);

/// Runs the probe in `wait <done>` mode with `env`, polls the report `file` until `found` accepts
/// its text, then lets the probe end by creating its done file. Returns the exit code, the log and
/// the text `found` accepted. No fixed sleep stands for a state: the probe waits until the test has
/// seen the wait, however slow the machine is.
fn run_waiting(exe: &Path, env: &[(&str, &str)], file: &str, found: impl Fn(&str) -> Option<String>) -> (Option<u32>, Vec<String>, String) {
    let folder = exe.parent().unwrap().to_path_buf();
    let done = folder.join("done.flag");
    let report = folder.join(file);
    let mut log: Vec<String> = Vec::new();
    let args = vec!["wait".to_owned(), done.display().to_string()];
    let env: Vec<(String, String)> = env.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    let (outcome, seen) = std::thread::scope(|s| {
        let runner = s.spawn(|| run_cancellable(exe, &args, &folder, &env, Some(DEADLINE + Duration::from_secs(60)), &AtomicBool::new(false), &mut log).unwrap());
        let end = Instant::now() + DEADLINE;
        let mut seen = None;
        while seen.is_none() && Instant::now() < end && !runner.is_finished() {
            seen = std::fs::read_to_string(&report).ok().and_then(|t| found(&t));
            if seen.is_none() {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        // Whatever was found, the probe may end now.
        std::fs::write(&done, "").unwrap();
        (runner.join().unwrap(), seen)
    });
    let text = std::fs::read_to_string(&report).unwrap_or_default();
    let seen = seen.unwrap_or_else(|| panic!("not found within {DEADLINE:?} in {file}:\n{text}\nlog: {log:#?}"));
    (outcome.exit_code, log, seen)
}

/// The block of one picture in `decant_stacks.txt` that shows the waiter in its event wait and the main
/// thread waiting on the waiter's thread handle.
fn picture_with_both_waits(text: &str) -> Option<String> {
    text.split("--- at ").find_map(|block| {
        let waiter = block.lines().find(|l| l.contains("\"probe waiter\": waits in NtWaitForSingleObject"))?;
        if !waiter.contains(" Event (auto reset, not set)") {
            return None;
        }
        let tid = waiter.split_whitespace().nth(1)?;
        let joined = block.lines().any(|l| l.contains("\"main\": waits in NtWaitForSingleObject") && l.contains(&format!(" Thread {tid}")));
        joined.then(|| block.to_owned())
    })
}

#[test]
fn decant_stacks_writes_every_thread_with_its_wait() {
    let (_dir, exe) = folder();
    // A picture every half second: the test waits for the first one taken while both threads wait.
    let times: Vec<String> = (1..=2 * DEADLINE.as_secs()).map(|i| format!("{}", i as f64 / 2.0)).collect();
    let (code, log, block) = run_waiting(&exe, &[("DECANT_STACKS", &times.join(","))], "decant_stacks.txt", picture_with_both_waits);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("DECANT_STACKS: ") && l.contains(" threads at ")), "{log:#?}");
    assert!(block.contains("(ntdll.dll!RtlUserThreadStart+0x"), "walked to the thread start: {block}");
}

#[test]
fn decant_waits_reports_a_long_wait_with_the_creator_of_its_event() {
    let (_dir, exe) = folder();
    let waiter_reported = |text: &str| text.lines().find(|l| l.contains("\"probe waiter\" has waited")).map(|_| text.to_owned());
    let (code, log, text) = run_waiting(&exe, &[("DECANT_WAITS", "1")], "decant_waits.txt", waiter_reported);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(text.contains("(ntdll.dll!RtlUserThreadStart+0x"), "{text}");
    let line = log.iter().find(|l| l.contains("DECANT_WAITS: thread") && l.contains("\"probe waiter\" has waited")).unwrap_or_else(|| panic!("{log:#?}"));
    assert!(line.contains("Event (auto reset, not set), created by tools_probe.exe: probe event, called from "), "{line}");
    assert!(line.contains(", from tools_probe.exe+0x"), "{line}");
}
#[test]
fn decant_calls_off_writes_nothing() {
    let (_dir, exe) = folder();
    let (code, log) = run(&exe, &["calls"], &[]);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(!log.iter().any(|l| l.contains("DECANT_CALLS")), "{log:#?}");
    assert!(!exe.parent().unwrap().join("decant_calls.txt").exists());
}
