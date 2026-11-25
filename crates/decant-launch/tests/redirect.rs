//! The debugger sends a title's load of `xgameruntime.dll` to the copy in the title folder.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use decant_launch::run::{Outcome, run_with_redirect};

/// A title folder with the probe and a stand-in `xgameruntime.dll` (a copy of a small system DLL).
fn title_folder() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let folder = std::fs::canonicalize(dir.path()).unwrap();
    let folder = folder.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(folder.clone(), PathBuf::from);
    let exe = folder.join("load_probe.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_load_probe"), &exe).unwrap();
    let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
    let ours = folder.join("xgameruntime.dll");
    std::fs::copy(Path::new(&system).join("System32").join("version.dll"), &ours).unwrap();
    (dir, exe, ours)
}

fn run(exe: &Path, redirect: bool) -> (Outcome, Vec<String>) {
    let mut log: Vec<String> = Vec::new();
    let outcome = run_with_redirect(exe, &[], exe.parent().unwrap(), &[], Some(Duration::from_secs(60)), &AtomicBool::new(false), redirect, &mut log).unwrap();
    (outcome, log)
}

fn loaded(outcome: &Outcome, ours: &Path) -> bool {
    outcome.dlls.iter().any(|p| p.to_string_lossy().eq_ignore_ascii_case(&ours.to_string_lossy()))
}

#[test]
fn a_system32_only_load_of_the_runtime_gets_the_copy_in_the_title_folder() {
    let (_dir, exe, ours) = title_folder();
    let (outcome, log) = run(&exe, true);
    assert_eq!(outcome.exit_code, Some(0), "{log:#?}");
    assert!(loaded(&outcome, &ours), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("redirect LoadLibraryExW(\"xgameruntime.dll\", flags 0x800)")), "{log:#?}");
}

#[test]
fn with_the_redirect_off_the_copy_in_the_title_folder_is_not_loaded() {
    let (_dir, exe, ours) = title_folder();
    let (outcome, log) = run(&exe, false);
    assert!(!loaded(&outcome, &ours), "{log:#?}");
    assert!(!log.iter().any(|l| l.contains("redirect LoadLibraryExW")), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("no redirect of xgameruntime.dll")), "{log:#?}");
}
