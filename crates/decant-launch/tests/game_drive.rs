//! A title that opens `X:/<file>` while `X:` is not mapped reads the file from the title folder
//! when the debugger redirects the game drive paths.

#![cfg(windows)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use decant_launch::drive::GameDrive;
use decant_launch::run::{Outcome, run_with_drive};

/// A drive letter nothing maps (the test never maps it).
fn free_letter() -> char {
    ('D'..='Z').rev().find(|c| !Path::new(&format!("{c}:\\")).exists()).expect("a free drive letter")
}

fn folder_with_file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let folder = std::fs::canonicalize(dir.path()).unwrap();
    let folder = folder.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(folder.clone(), PathBuf::from);
    std::fs::create_dir(folder.join("sub")).unwrap();
    std::fs::write(folder.join("hello.txt"), b"hello").unwrap();
    std::fs::write(folder.join("sub").join("deep.txt"), b"hello").unwrap();
    (dir, folder)
}

fn run(folder: &Path, argument: &str, redirect_letter: Option<char>) -> (Outcome, Vec<String>) {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_drive_probe"));
    let mut log: Vec<String> = Vec::new();
    let mut drive = match redirect_letter {
        Some(letter) => GameDrive::redirecting(letter, folder),
        None => GameDrive::acquire_letter(None, folder, &mut log),
    };
    let outcome = run_with_drive(&exe, &[argument.to_owned()], folder, &[], Some(Duration::from_secs(60)), &AtomicBool::new(false), false, &mut drive, &mut log).unwrap();
    (outcome, log)
}

fn redirect_lines(log: &[String]) -> usize {
    log.iter().filter(|l| l.contains(" redirect CreateFileW(") || l.contains(" redirect GetFileAttributesW(")).count()
}

#[test]
fn an_unmapped_drive_path_is_read_from_the_title_folder() {
    let (_dir, folder) = folder_with_file();
    let letter = free_letter();
    for path in [format!("{letter}:/hello.txt"), format!("{}:\\sub\\deep.txt", letter.to_ascii_lowercase()), format!("{letter}:/sub/deep.txt")] {
        let (outcome, log) = run(&folder, &path, Some(letter));
        assert_eq!(outcome.exit_code, Some(0), "{path}: {log:#?}");
        assert!(redirect_lines(&log) >= 1, "{log:#?}");
    }
}

#[test]
fn a_function_and_path_pair_is_logged_once_however_often_it_is_used() {
    let (_dir, folder) = folder_with_file();
    let letter = free_letter();
    // The probe uses the path for `exists` and for `read`, which is several calls of two functions.
    let (outcome, log) = run(&folder, &format!("{letter}:/hello.txt"), Some(letter));
    assert_eq!(outcome.exit_code, Some(0), "{log:#?}");
    let lines: Vec<&String> = log.iter().filter(|l| l.contains(" redirect ") && l.contains("(\"")).collect();
    assert!(!lines.is_empty() && lines.len() <= 2, "{log:#?}");
    let distinct: HashSet<&str> = lines.iter().map(|l| l.split_once(' ').map_or("", |(_, rest)| rest.trim_start())).collect();
    assert_eq!(distinct.len(), lines.len(), "a pair twice: {log:#?}");
}

#[test]
fn the_calls_are_counted_at_the_end_of_the_run() {
    let (_dir, folder) = folder_with_file();
    let letter = free_letter();
    let (outcome, log) = run(&folder, &format!("{letter}:/hello.txt"), Some(letter));
    assert_eq!(outcome.exit_code, Some(0), "{log:#?}");
    let line = log.iter().find(|l| l.contains("file calls: CreateFileW")).unwrap_or_else(|| panic!("{log:#?}"));
    assert!(line.contains("rewritten"), "{line}");
}

#[test]
fn without_the_redirect_the_drive_path_is_not_found() {
    let (_dir, folder) = folder_with_file();
    let letter = free_letter();
    let (outcome, log) = run(&folder, &format!("{letter}:/hello.txt"), None);
    assert_eq!(outcome.exit_code, Some(7), "{log:#?}");
    assert_eq!(redirect_lines(&log), 0, "{log:#?}");
}

#[test]
fn a_path_that_does_not_start_with_the_drive_is_untouched() {
    let (_dir, folder) = folder_with_file();
    let letter = free_letter();
    let direct = folder.join("hello.txt").display().to_string();
    let (outcome, log) = run(&folder, &direct, Some(letter));
    assert_eq!(outcome.exit_code, Some(0), "{log:#?}");
    assert_eq!(redirect_lines(&log), 0, "{log:#?}");
}
