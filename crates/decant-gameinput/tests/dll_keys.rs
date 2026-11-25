//! `DECANT_KEYS` through the real `GameInput.dll`, with no controller: its own file (not a test
//! inside `dll.rs`) because it reads the process environment that
//! `the_dll_exports_ordinal_one_and_plays_the_script` also sets, and integration test files are
//! each their own process, so the two no longer race when the whole suite runs together.
#![cfg(windows)]
#![allow(unsafe_code)]

use decant_testkit::{Library, dll_for};

mod common;
use common::{Create, pad_of, wait_for};

/// with real input disabled, DECANT_KEYS still presses and
/// releases on its own clock through the real DLL.
#[test]
fn the_dll_plays_six_keys_with_no_real_device() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "GameInput.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("GameInput.dll loads");
    // SAFETY: nothing else in this test process reads the environment concurrently.
    unsafe {
        std::env::remove_var("DECANT_PAD_SCRIPT");
        std::env::set_var("DECANT_NO_REAL_INPUT", "1");
        std::env::set_var("DECANT_KEYS", "0.05:A*150");
    }
    // SAFETY: the export and vtable slots have the public GameInput.h prototypes.
    unsafe {
        let create = lib.get::<Create>("GameInputCreate").unwrap();
        let mut input = std::ptr::null_mut();
        assert_eq!(create(&mut input), 0);
        assert!(!input.is_null());
        // Neutral before the scheduled press: no script, no controller.
        let (buttons, axes, _) = pad_of(input);
        assert_eq!((buttons, axes), (0, [0.0; 6]), "neutral with no controller and before DECANT_KEYS fires");
        wait_for(input, "DECANT_KEYS presses A at 0.05 s", |b, _| b == 0x4);
        wait_for(input, "DECANT_KEYS releases A after 150 ms", |b, _| b == 0);
    }
}
