//! Runs `xgameruntime_test` against the `xgameruntime.dll` Cargo built next to it (or the one in
//! `DECANT_DLL_DIR`), as the review workflow ran the C test against the C DLL. `offline` needs
//! no PC runtime. `direct` compares the interposer with the PC runtime of System32 and skips
//! itself (the program prints SKIP) on a machine without Gaming Services. `online` and `preload`
//! print SKIP unless Cargo found the public GDK through `DECANT_GDK_ROOT`.
#![cfg(windows)]

use std::process::Command;

fn run(mode: &str) -> String {
    let program = env!("CARGO_BIN_EXE_xgameruntime_test");
    let dll = decant_testkit::dll_for(program, "xgameruntime.dll");
    // The program writes `xgameruntime_test.log` in its folder: one folder per run, the tests run in parallel.
    let work = std::env::temp_dir().join(format!("xgameruntime_test_{}_{mode}", std::process::id()));
    std::fs::create_dir_all(&work).expect("a working folder");
    // `direct` compares every table with the PC runtime's, so the offline user tables that are
    // served locally by default are forwarded there too.
    let mut command = Command::new(program);
    if mode == "direct" {
        command.env("DECANT_XGR_REAL_USER", "1");
    }
    let out = command.arg(&dll).arg(mode).current_dir(&work).output().expect("the test program starts");
    let _ = std::fs::remove_dir_all(&work);
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    print!("{text}");
    assert!(out.status.success(), "xgameruntime_test {} {mode} failed: {:?}", dll.display(), out.status);
    text
}

#[test]
fn offline_passes() {
    let text = run("offline");
    assert_eq!(text.matches("PASS").count(), 25, "all twenty-five checks print PASS");
    assert!(!text.contains("FAIL"));
}

#[test]
fn direct_matches_the_pc_runtime() {
    let text = run("direct");
    assert!(text.contains("SKIP") || text.contains("PASS through the interposer"), "{text}");
}

#[test]
fn online_and_preload_pass_or_skip() {
    run("online");
    run("preload");
}
