//! COM call helpers shared by the `GameInput.dll` integration tests: reading a gamepad through
//! the real vtable and waiting for a state to show up.
#![allow(unsafe_code, dead_code)]

use std::ffi::c_void;
use std::time::{Duration, Instant};

pub type Hresult = i32;
pub type Create = unsafe extern "system" fn(*mut *mut c_void) -> Hresult;
pub type GetReading = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, *mut *mut c_void) -> Hresult;
pub type ReadingTimestamp = unsafe extern "system" fn(*mut c_void) -> u64;
pub type GamepadState = unsafe extern "system" fn(*mut c_void, *mut [f32; 7]) -> bool;
pub type Release = unsafe extern "system" fn(*mut c_void) -> u32;

pub const KIND_GAMEPAD: u32 = 0x4_0000;

/// The function in slot `slot` of the COM object `object`.
pub unsafe fn method<F: Copy>(object: *mut c_void, slot: usize) -> F {
    // SAFETY: the caller names a slot of a COM object and the right function type.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(slot))
    }
}

/// Buttons and the six axes of a reading of the gamepad, and the reading's timestamp.
pub unsafe fn pad_of(input: *mut c_void) -> (u32, [f32; 6], u64) {
    // SAFETY: slot 4 is GetCurrentReading, the reading is released after use.
    unsafe {
        let mut reading = std::ptr::null_mut();
        let hr = method::<GetReading>(input, 4)(input, KIND_GAMEPAD, std::ptr::null_mut(), &mut reading);
        assert_eq!(hr, 0);
        assert!(!reading.is_null());
        let mut state = [0f32; 7];
        assert!(method::<GamepadState>(reading, 22)(reading, &mut state));
        let stamp = method::<ReadingTimestamp>(reading, 5)(reading);
        method::<Release>(reading, 2)(reading);
        (state[0].to_bits(), [state[1], state[2], state[3], state[4], state[5], state[6]], stamp)
    }
}

/// Polls `pad_of` until `test` is true, or panics with `what` after 5 seconds.
pub unsafe fn wait_for(input: *mut c_void, what: &str, test: impl Fn(u32, [f32; 6]) -> bool) {
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        // SAFETY: a live input object.
        let (buttons, axes, _) = unsafe { pad_of(input) };
        if test(buttons, axes) {
            return;
        }
        assert!(Instant::now() < end, "the pad never showed: {what} (buttons {buttons:#x}, axes {axes:?})");
        std::thread::sleep(Duration::from_millis(5));
    }
}
