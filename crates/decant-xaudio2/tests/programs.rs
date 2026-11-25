//! Runs `xaudio2test` against the `decant_xaudio2.dll` Cargo built (or the one in
//! `DECANT_DLL_DIR`), and `gameinputtest`, as the review workflow ran the C tests.
#![cfg(windows)]

use std::process::Command;

fn passes(mut command: Command, what: &str) {
    let out = command.output().expect("the test program starts");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "{what} failed: {:?}", out.status);
}

#[test]
fn xaudio2test_passes() {
    let program = env!("CARGO_BIN_EXE_xaudio2test");
    let mut command = Command::new(program);
    command.arg(decant_testkit::dll_for(program, "decant_xaudio2.dll"));
    passes(command, "xaudio2test");
}

#[test]
fn gameinputtest_passes() {
    passes(Command::new(env!("CARGO_BIN_EXE_gameinputtest")), "gameinputtest");
}
