//! Runs `loadtest` and `xmem_abitest` against the `xmem.dll` Cargo built next to them (or the one
//! in `DECANT_DLL_DIR`), as the review workflow ran the C tests against the C DLL.
#![cfg(windows)]

use std::process::Command;

fn run(program: &str) {
    let dll = decant_testkit::dll_for(program, "xmem.dll");
    let out = Command::new(program).arg(&dll).output().expect("the test program starts");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "{program} {} failed: {:?}", dll.display(), out.status);
}

#[test]
fn loadtest_passes() {
    run(env!("CARGO_BIN_EXE_loadtest"));
}

#[test]
fn xmem_abitest_passes() {
    run(env!("CARGO_BIN_EXE_xmem_abitest"));
}
