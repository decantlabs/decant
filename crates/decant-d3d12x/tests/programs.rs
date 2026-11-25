//! Runs `d3d12xs_test` against the `d3d12_xs.dll` Cargo built next to it (or the one in
//! `DECANT_DLL_DIR`), as the review workflow ran the C test against the C DLL. The device checks
//! skip on a machine without a D3D12 12_0 GPU, the program then still passes.
#![cfg(windows)]

use std::process::Command;

#[test]
fn d3d12xs_test_passes() {
    let program = env!("CARGO_BIN_EXE_d3d12xs_test");
    let dll = decant_testkit::dll_for(program, "d3d12_xs.dll");
    // A pool of 256 MiB for the title memory case.
    // The table of unanswered questions on, for the console slot checks, its report in the temp folder.
    let calls = std::env::temp_dir().join(format!("d3d12xs_test_sixcalls_{}.txt", std::process::id()));
    let out = Command::new(program)
        .arg(&dll)
        .env("DECANT_XMEM_PHYSICAL_MIB", "256")
        .env("DECANT_CALLS", "1")
        .env("DECANT_CALLS_PATH", &calls)
        .output()
        .expect("the test program starts");
    let _ = std::fs::remove_file(&calls);
    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "{program} {} failed: {:?}", dll.display(), out.status);
}

/// The device case of the direct shader path: the same program with `direct`, under
/// `DECANT_D3D12X_DIRECT=1` and a DXIL cache of its own. Skips without a GPU or without DXC.
#[test]
fn d3d12xs_test_draws_translated_shaders() {
    let program = env!("CARGO_BIN_EXE_d3d12xs_test");
    let dll = decant_testkit::dll_for(program, "d3d12_xs.dll");
    let scratch = std::env::temp_dir().join(format!("d3d12xs-direct-case-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let log = scratch.join("layer.log");
    std::fs::create_dir_all(&scratch).expect("scratch folder");
    let out = Command::new(program)
        .arg(&dll)
        .arg("direct")
        .env("DECANT_D3D12X_DIRECT", "1")
        .env("DECANT_D3D12X_DIRECT_CACHE", scratch.join("cache"))
        .env("DECANT_D3D12X_DIRECT_DUMP", scratch.join("dump"))
        .env("DECANT_LOG_PATH", &log)
        .env_remove("DECANT_D3D12X_DXIL_OVERRIDE")
        .output()
        .expect("the test program starts");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    print!("{text}");
    // What the layer said about the direct path, for a failure report.
    for line in std::fs::read_to_string(&log).unwrap_or_default().lines().filter(|l| l.contains("direct shaders") || l.contains("on first bind")) {
        println!("log   {line}");
    }
    let kept = out.status.success() && !text.contains("FAIL");
    if kept {
        let _ = std::fs::remove_dir_all(&scratch);
    } else {
        println!("kept  {}", scratch.display());
    }
    assert!(kept, "{program} {} direct failed: {:?}", dll.display(), out.status);
}
