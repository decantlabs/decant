//! A title that asks for a relaunch (`XLaunchNewGame` in our `xgameruntime.dll`) writes the request
//! file and ends with the relaunch exit code: the launcher starts it again with the new arguments.
//! `cmd.exe` plays the title here.
#![cfg(windows)]

use std::path::PathBuf;
use std::time::Duration;

use decant_launch::relaunch::{EXIT_CODE, FILE_NAME};
use decant_launch::run::run;

#[test]
fn a_relaunch_request_starts_the_title_again_with_its_arguments() {
    let cmd = PathBuf::from(std::env::var_os("ComSpec").expect("ComSpec"));
    let folder = tempfile::tempdir().unwrap();
    // First run: write "<cmd>" and "/C exit 7" to the request file, end with the relaunch code.
    let first = format!("(echo {}& echo /C exit 7)>%DECANT_RELAUNCH_FILE%& exit {EXIT_CODE}", cmd.display());
    let mut log: Vec<String> = Vec::new();
    let outcome = run(&cmd, &["/C".into(), first], folder.path(), &[], Some(Duration::from_secs(60)), &mut log).unwrap();
    assert_eq!(outcome.exit_code, Some(7), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("relaunch 1:") && l.contains(r#"["/C", "exit", "7"]"#)), "{log:#?}");
    assert!(!folder.path().join(FILE_NAME).exists(), "the request file is removed");
}

#[test]
fn the_relaunch_code_without_a_request_ends_the_run() {
    let cmd = PathBuf::from(std::env::var_os("ComSpec").expect("ComSpec"));
    let folder = tempfile::tempdir().unwrap();
    let mut log: Vec<String> = Vec::new();
    let outcome = run(&cmd, &["/C".into(), format!("exit {EXIT_CODE}")], folder.path(), &[], Some(Duration::from_secs(60)), &mut log).unwrap();
    assert_eq!(outcome.exit_code, Some(EXIT_CODE));
    assert!(!log.iter().any(|l| l.contains("relaunch 1:")), "{log:#?}");
}
