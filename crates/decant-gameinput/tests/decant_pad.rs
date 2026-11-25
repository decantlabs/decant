//! `decant-pad` appends checked lines to the control file named by `--file` or `DECANT_PAD_SCRIPT`.

use std::process::Command;

fn decant_pad(file: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_decant-pad")).env_remove("DECANT_PAD_SCRIPT").arg("--file").arg(file).args(args).output().unwrap()
}

#[test]
fn commands_are_appended_one_per_line_and_bad_ones_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pad.txt");
    // A last line without its end must not swallow the next command.
    std::fs::write(&file, "A down").unwrap();
    assert!(decant_pad(&file, &["press", "B", "250"]).status.success());
    assert!(decant_pad(&file, &["hold", "Start", "1000"]).status.success());
    assert!(decant_pad(&file, &["stick", "left", "0.0", "1.0"]).status.success());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "A down\npress B 250\nhold Start 1000\nstick left 0.0 1.0\n");
    let bad = decant_pad(&file, &["press", "Z"]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("not a button"));
    assert_eq!(std::fs::read_to_string(&file).unwrap().lines().count(), 4, "a refused command writes nothing");
    assert!(decant_pad(&file, &["clear"]).status.success());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "");
}

#[test]
fn the_file_comes_from_the_environment_without_the_option() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("env.txt");
    let ok = Command::new(env!("CARGO_BIN_EXE_decant-pad")).env("DECANT_PAD_SCRIPT", &file).args(["press", "A"]).status().unwrap();
    assert!(ok.success());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "press A\n");
    let none = Command::new(env!("CARGO_BIN_EXE_decant-pad")).env_remove("DECANT_PAD_SCRIPT").args(["press", "A"]).output().unwrap();
    assert_eq!(none.status.code(), Some(2));
}
