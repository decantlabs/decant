//! Runs `ffxtest` when `DECANT_FFX_TEST_BACKEND` names a PC FidelityFX DLL (absolute path),
//! else prints why it skipped. `DECANT_DLL_DIR` points it at another build of `decant_ffx.dll`.
#![cfg(windows)]

#[test]
fn ffxtest_with_a_real_backend() {
    let Some(backend) = std::env::var_os("DECANT_FFX_TEST_BACKEND") else {
        println!("SKIPPED ffxtest: set DECANT_FFX_TEST_BACKEND to a PC FidelityFX DLL");
        return;
    };
    let program = env!("CARGO_BIN_EXE_ffxtest");
    let dir = decant_testkit::dll_for(program, "decant_ffx.dll").parent().map(std::path::Path::to_path_buf).unwrap_or_default();
    let out = std::process::Command::new(program).arg(&dir).arg(&backend).output().expect("ffxtest starts");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "ffxtest failed: {:?}", out.status);
}
