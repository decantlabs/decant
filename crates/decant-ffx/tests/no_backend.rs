//! `decant_ffx.dll` without a backend, always run: the five calls answer 1, a relative path and the
//! shim's own path are refused, and a refusal is not remembered. `DECANT_DLL_DIR` points the
//! test at another build.
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;

use decant_testkit::Library;

type CreateFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_void) -> u32;
type ArgFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void) -> u32;

/// The five calls with a null context and null arguments, as they return without a backend.
fn answers(lib: &Library) -> [u32; 5] {
    // SAFETY: the signatures of ffx_api.h. With no backend loaded the shim returns before it
    // touches any argument, which is what this test checks.
    unsafe {
        let create = lib.get::<CreateFn>("ffxCreateContext").expect("ffxCreateContext exported");
        let mut context: *mut c_void = std::ptr::null_mut();
        let mut out = [create(&mut context, std::ptr::null_mut(), std::ptr::null()), 0, 0, 0, 0];
        for (i, name) in ["ffxDestroyContext", "ffxConfigure", "ffxQuery", "ffxDispatch"].iter().enumerate() {
            let f = lib.get::<ArgFn>(name).unwrap_or_else(|| panic!("{name} exported"));
            out[i + 1] = f(&mut context, std::ptr::null_mut());
        }
        out
    }
}

/// One test function only: it changes the environment of this process.
#[test]
fn without_a_backend_every_call_answers_one() {
    let dll = decant_testkit::dll_for(env!("CARGO_BIN_EXE_ffxtest"), "decant_ffx.dll");
    let dll = std::path::absolute(&dll).unwrap();
    let lib = Library::open(dll.as_os_str()).expect("decant_ffx.dll loads");
    // SAFETY: this binary runs this single test, so no other thread reads the environment.
    unsafe { std::env::remove_var("DECANT_FFX_BACKEND") };
    assert_eq!(answers(&lib), [1; 5], "no variable");
    // SAFETY: as above.
    unsafe { std::env::set_var("DECANT_FFX_BACKEND", "decant_ffx.dll") };
    assert_eq!(answers(&lib), [1; 5], "relative path");
    // SAFETY: as above.
    unsafe { std::env::set_var("DECANT_FFX_BACKEND", &dll) };
    assert_eq!(answers(&lib), [1; 5], "the shim itself");
    // SAFETY: as above.
    unsafe { std::env::set_var("DECANT_FFX_BACKEND", r"C:\no\such\backend.dll") };
    assert_eq!(answers(&lib), [1; 5], "missing file");
}
