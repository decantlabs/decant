//! A small stand-in title (not the game) calls every file function the debugger redirects for the game
//! drive with `X:\pack\...` paths, while `X:` is not mapped: each call must reach the title folder, and
//! only it (see T22 of the black-screen work: the title's own paths went to the real drive).

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use decant_launch::drive::GameDrive;
use decant_launch::run::{Outcome, run_with_drive};

/// The calls of `src/file_probe.rs`, by the bit each sets in the exit code when it fails.
const CALLS: [&str; 15] = [
    "CreateFileW",
    "CreateFileA",
    "CreateFile2",
    "CreateFileW extended form",
    "GetFileAttributesW",
    "GetFileAttributesA",
    "GetFileAttributesExW",
    "FindFirstFileExW",
    "CreateDirectoryW",
    "DeleteFileW",
    "RemoveDirectoryW",
    "SetFileAttributesW",
    "MoveFileExW",
    "CreateFileW NT form",
    "CreateDirectoryW at a top folder the folder lacks",
];

/// The functions whose counts the log must show with calls rewritten (the title folder is the place they land).
const REWRITTEN: [&str; 12] = [
    "CreateFileW",
    "CreateFileA",
    "CreateFile2",
    "GetFileAttributesW",
    "GetFileAttributesA",
    "GetFileAttributesExW",
    "FindFirstFileExW",
    "CreateDirectoryW",
    "DeleteFileW",
    "RemoveDirectoryW",
    "SetFileAttributesW",
    "MoveFileExW",
];

/// A drive letter nothing maps (the test never maps it).
fn free_letter() -> char {
    ('D'..='Z').rev().find(|c| !Path::new(&format!("{c}:\\")).exists()).expect("a free drive letter")
}

/// A title folder with `pack\a.dat` ("hello") and the files the probe deletes, moves or marks.
fn title_folder() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let folder = std::fs::canonicalize(dir.path()).unwrap();
    let folder = folder.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(folder.clone(), PathBuf::from);
    let pack = folder.join("pack");
    std::fs::create_dir_all(pack.join("emptydir")).unwrap();
    std::fs::write(pack.join("a.dat"), b"hello").unwrap();
    for name in ["del.txt", "attr.txt", "from.txt"] {
        std::fs::write(pack.join(name), b"x").unwrap();
    }
    (dir, folder)
}

fn run(folder: &Path, letter: char, redirect: bool) -> (Outcome, Vec<String>) {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_file_probe"));
    let mut log: Vec<String> = Vec::new();
    let mut drive = if redirect { GameDrive::redirecting(letter, folder) } else { GameDrive::acquire_letter(None, folder, &mut log) };
    let outcome = run_with_drive(&exe, &[letter.to_string()], folder, &[], Some(Duration::from_secs(60)), &AtomicBool::new(false), false, &mut drive, &mut log).unwrap();
    (outcome, log)
}

fn failed_calls(outcome: &Outcome) -> Vec<&'static str> {
    let code = outcome.exit_code.unwrap_or(u32::MAX);
    CALLS.iter().enumerate().filter(|(bit, _)| code & (1 << bit) != 0).map(|(_, name)| *name).collect()
}

#[test]
fn every_redirected_file_function_reaches_the_title_folder_and_only_it() {
    let (_dir, folder) = title_folder();
    let letter = free_letter();
    let (outcome, log) = run(&folder, letter, true);
    assert_eq!(outcome.exit_code, Some(0), "calls that failed: {:?}\n{log:#?}", failed_calls(&outcome));
    let pack = folder.join("pack");
    // What each call changed is in the title folder.
    assert!(pack.join("newdir").is_dir(), "CreateDirectoryW: {log:#?}");
    assert!(folder.join("newtop").is_dir(), "CreateDirectoryW at a top folder the folder lacks: {log:#?}");
    assert!(!pack.join("del.txt").exists(), "DeleteFileW: {log:#?}");
    assert!(!pack.join("emptydir").exists(), "RemoveDirectoryW: {log:#?}");
    assert!(std::fs::metadata(pack.join("attr.txt")).unwrap().permissions().readonly(), "SetFileAttributesW: {log:#?}");
    assert!(!pack.join("from.txt").exists() && pack.join("to.txt").exists(), "MoveFileExW moves both paths: {log:#?}");
    // Every function shows rewritten calls in the counts at the end.
    for name in REWRITTEN {
        let line = log.iter().find(|l| l.contains(&format!("file calls: {name} "))).unwrap_or_else(|| panic!("no count for {name}: {log:#?}"));
        assert!(!line.contains("(rewritten 0,"), "{line}");
    }
    // Nothing is left on the drive letter itself.
    assert!(!Path::new(&format!("{letter}:\\")).exists());
    // The test folder's read-only file must be removable when the temp folder goes.
    let mut permissions = std::fs::metadata(pack.join("attr.txt")).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    std::fs::set_permissions(pack.join("attr.txt"), permissions).unwrap();
}

#[test]
fn without_the_redirect_none_of_those_calls_reaches_the_title_folder() {
    let (_dir, folder) = title_folder();
    let letter = free_letter();
    let (outcome, log) = run(&folder, letter, false);
    assert_ne!(outcome.exit_code, Some(0), "the drive is not there: {log:#?}");
    assert!(!folder.join("pack").join("newdir").exists() && !folder.join("newtop").exists());
    assert!(folder.join("pack").join("del.txt").exists() && folder.join("pack").join("from.txt").exists());
}
