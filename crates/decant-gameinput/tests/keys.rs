//! `DECANT_KEYS` through the built `GameInput.dll`: timed inputs reach the readings without a control file.
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::time::{Duration, Instant};

use decant_testkit::{Library, dll_for};

type Create = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type GetReading = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, *mut *mut c_void) -> i32;
type GamepadState = unsafe extern "system" fn(*mut c_void, *mut [f32; 7]) -> bool;
type Release = unsafe extern "system" fn(*mut c_void) -> u32;

const KIND_GAMEPAD: u32 = 0x4_0000;

unsafe fn method<F: Copy>(object: *mut c_void, slot: usize) -> F {
    // SAFETY: the caller names a slot of a COM object and the right function type.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(slot))
    }
}

/// Buttons and the six axes of the current gamepad reading.
unsafe fn pad_of(input: *mut c_void) -> (u32, [f32; 7]) {
    // SAFETY: slot 4 is GetCurrentReading, slot 22 of a reading GetGamepadState.
    unsafe {
        let mut reading = std::ptr::null_mut();
        assert_eq!(method::<GetReading>(input, 4)(input, KIND_GAMEPAD, std::ptr::null_mut(), &mut reading), 0);
        let mut state = [0f32; 7];
        assert!(method::<GamepadState>(reading, 22)(reading, &mut state));
        method::<Release>(reading, 2)(reading);
        (state[0].to_bits(), state)
    }
}

#[test]
fn timed_keys_reach_the_pad() {
    let exe = std::env::current_exe().unwrap();
    let lib = Library::open(dll_for(exe.to_str().unwrap(), "GameInput.dll").as_os_str()).expect("GameInput.dll loads");
    // SAFETY: nothing else in this test process reads the environment concurrently.
    unsafe {
        std::env::remove_var("DECANT_PAD_SCRIPT");
        std::env::set_var("DECANT_KEYS", "0:A*60000, 0:L:0.5:1, 1.5:B*60000");
        // A real pad on the test machine must not mix into the scripted state.
        std::env::set_var("DECANT_NO_REAL_INPUT", "1");
    }
    // SAFETY: the export has this prototype.
    unsafe {
        let mut input = std::ptr::null_mut();
        assert_eq!(lib.get::<Create>("GameInputCreate").unwrap()(&mut input), 0);
        let end = Instant::now() + Duration::from_secs(10);
        let mut seen_a = false;
        loop {
            let (buttons, s) = pad_of(input);
            if buttons == 0x4 && s[3] == 0.5 && s[4] == 1.0 {
                seen_a = true;
            }
            if buttons == 0x4 | 0x8 {
                break;
            }
            assert!(Instant::now() < end, "B never came: buttons {buttons:#x}");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(seen_a, "A and the stick first, B later");
    }
}
