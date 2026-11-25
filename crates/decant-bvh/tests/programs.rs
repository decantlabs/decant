//! Runs `decant_bvh_test` against the `decant_bvh.dll` Cargo built (or the one in `DECANT_DLL_DIR`).
//! A machine without a DXR device (exit code 2, such as a hosted runner) skips, unless
//! `DECANT_BVH_REQUIRE_DXR` is set.
#![cfg(windows)]

#[test]
fn decant_bvh_test_builds_a_triangle() {
    let program = env!("CARGO_BIN_EXE_decant_bvh_test");
    let dll = decant_testkit::dll_for(program, "decant_bvh.dll");
    let out = std::process::Command::new(program).arg(&dll).output().expect("decant_bvh_test starts");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    if out.status.code() == Some(2) && std::env::var_os("DECANT_BVH_REQUIRE_DXR").is_none() {
        println!("SKIPPED decant_bvh_test: no DXR device");
        return;
    }
    assert!(out.status.success(), "decant_bvh_test {} failed: {:?}", dll.display(), out.status);
}
