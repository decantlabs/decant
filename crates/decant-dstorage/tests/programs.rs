//! Runs `dstoragetest` and `dstorage_abi_test` against the `dstorage_xs.dll` Cargo built next to
//! them (or the one in `DECANT_DLL_DIR`, such as the former C++ DLL), as `build_storage.cmd`
//! built them for the review.
//!
//! Each test skips cleanly when an external piece is missing:
//! - `dstoragetest` makes its compressed data with libdeflate and the GDeflate compressor, which
//!   `build.rs` links only when `DECANT_GDEFLATE_ROOT` was set at build time (the program then
//!   prints `SKIP` and passes).
//! - The PC DirectStorage run needs `DECANT_DSTORAGE_PC_DLL` to name `dstorage.dll` from the
//!   `Microsoft.Direct3D.DirectStorage` package (`native/bin/x64`). Without it only the CPU
//!   backend runs.
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

const ABI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/abi");

/// Runs `run` (a copy of `program`, or `program` itself) against the DLL that goes with `program`,
/// and fails with its output unless it exits with 0.
fn passes_from(run: &Path, program: &str, args: &[&str], backend: &str) -> String {
    let dll = decant_testkit::dll_for(program, "dstorage_xs.dll");
    let mut command = Command::new(run);
    command.arg(&dll).args(args).env("DECANT_DSTORAGE_PC_DLL", backend).env_remove("DECANT_DSTORAGE_ABI").env_remove("DECANT_LOG").env_remove("DECANT_LOG_PATH");
    let out = command.output().expect("the test program starts");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    print!("{text}");
    assert!(out.status.success(), "{program} {} {args:?} on the {backend} backend failed: {:?}", dll.display(), out.status);
    text
}

/// Runs a test program and fails with its output unless it exits with 0.
fn passes(program: &str, args: &[&str], backend: &str) -> String {
    passes_from(Path::new(program), program, args, backend)
}

/// The PC DirectStorage runtime the developer named, if it exists.
fn pc_runtime() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("DECANT_DSTORAGE_PC_DLL")?);
    (path != Path::new("cpu") && path.is_file()).then_some(path)
}

#[test]
fn dstoragetest_passes_on_the_cpu_backend() {
    let text = passes(env!("CARGO_BIN_EXE_dstoragetest"), &[], "cpu");
    assert!(text.contains("PASS: asynchronous Submit") || text.contains("SKIP: dstoragetest needs GDeflate"), "{text}");
}

/// PC DirectStorage looks for `dstoragecore.dll` in the application folder, so the program runs from
/// a folder of its own that holds a copy of it and of the runtime.
#[test]
fn dstoragetest_passes_on_pc_directstorage() {
    let Some(runtime) = pc_runtime() else {
        println!("SKIP: set DECANT_DSTORAGE_PC_DLL to dstorage.dll of the DirectStorage package to run on PC DirectStorage");
        return;
    };
    let program = env!("CARGO_BIN_EXE_dstoragetest");
    let folder = std::env::temp_dir().join(format!("decant_storage_pc_{}", std::process::id()));
    std::fs::create_dir_all(&folder).expect("a temporary folder");
    let copy = |from: &Path| {
        let to = folder.join(from.file_name().expect("a file name"));
        std::fs::copy(from, &to).expect("the copy");
        to
    };
    let core = runtime.with_file_name("dstoragecore.dll");
    assert!(core.is_file(), "dstoragecore.dll must sit beside the dstorage.dll named by DECANT_DSTORAGE_PC_DLL");
    let moved = copy(Path::new(program));
    copy(&core);
    let runtime = copy(&runtime);
    let text = passes_from(&moved, program, &[], &runtime.to_string_lossy());
    let _ = std::fs::remove_dir_all(&folder);
    assert!(text.contains("PASS: queued reads") || text.contains("SKIP: dstoragetest needs GDeflate"), "{text}");
}

/// One process per spec, since the DLL reads its spec once. The scenarios share files in the
/// temporary folder, so they run one after the other.
#[test]
fn dstorage_abi_test_passes_for_every_mode() {
    let program = env!("CARGO_BIN_EXE_dstorage_abi_test");
    let spec = |name: &str| format!("{ABI}/{name}");
    let (pc, leads, template) = (spec("pc_hypothesis.abi"), spec("other_title.abi"), spec("console_hypothesis.abi"));
    let retail = spec("retail_title.abi");
    let runs: [Vec<&str>; 11] = [
        vec!["none"],
        vec!["spec", "a"],
        vec!["spec", "b"],
        vec!["bad"],
        vec!["parse", &pc],
        vec!["parse", &leads],
        vec!["parse", &retail],
        vec!["retail", &retail],
        vec!["template", &template],
        vec!["refused", &template],
        vec!["refused", &leads],
    ];
    for args in &runs {
        let text = passes(program, args, "cpu");
        assert!(text.contains("OK dstorage_abi_test"), "{text}");
    }
}

#[test]
fn dstorage_abi_test_refuses_a_template_without_iids_and_bad_arguments() {
    let program = env!("CARGO_BIN_EXE_dstorage_abi_test");
    let dll = decant_testkit::dll_for(program, "dstorage_xs.dll");
    // `parse` of the template fails: its factory has no IID to ask for. `template` of a spec that
    // has no `iid unknown` to fill fails too.
    let out = Command::new(program).arg(&dll).arg("parse").arg(format!("{ABI}/console_hypothesis.abi")).env("DECANT_DSTORAGE_PC_DLL", "cpu").output().expect("starts");
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    let leads = format!("{ABI}/other_title.abi");
    let out = Command::new(program).arg(&dll).arg("template").arg(leads).env("DECANT_DSTORAGE_PC_DLL", "cpu").output().expect("starts");
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    let out = Command::new(program).arg(&dll).arg("bogus").env("DECANT_DSTORAGE_PC_DLL", "cpu").output().expect("starts");
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(program).arg("no_such_dll.dll").arg("none").output().expect("starts");
    assert_eq!(out.status.code(), Some(2));
}
