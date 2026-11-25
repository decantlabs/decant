//! `GameInput.dll`: a Decant shim that gives a title one always connected gamepad, driven by a control file.
//!
//! The title binds to `GameInputCreate` by ordinal 1 of `GameInput.dll`. This DLL exports the same three names under the
//! same ordinals as the system copy and answers the first GameInput interface set (`IGameInput`, `IGameInputDevice`,
//! `IGameInputReading`) itself, so a simulated pad can play without a controller. `DECANT_PAD_SCRIPT` names the control
//! file (the format is in `pad`, the tool that writes it is `decant-pad`). Without that variable the pad stays neutral.
//!
//! * `pad`: the state, the language of the file and the state machine. No Windows code, fully unit tested.
//! * `devices`: the seam where a real controller and keyboard merge into the pad (`DECANT_NO_REAL_INPUT`
//!   turns it off). The merge is plain code, fully unit tested; the actual device reads are Windows-only.
//! * `runtime`: the live state, the clock of the readings and the thread that polls the file every 16 ms.
//! * `com`: the objects and the exports.
#![allow(non_snake_case)]

pub mod devices;
pub mod keys;
pub mod pad;

#[cfg(windows)]
mod com;
#[cfg(windows)]
mod runtime;

/// The exports of the DLL, by name. `GameInputCreate` is ordinal 1.
pub const EXPORT_NAMES: [&str; 3] = ["GameInputCreate", "DllCanUnloadNow", "DllGetClassObject"];

#[cfg(windows)]
#[allow(unsafe_code)]
mod exports {
    use std::ffi::c_void;

    /// # Safety
    /// `out` is null or writable.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn GameInputCreate(out: *mut *mut c_void) -> i32 {
        // SAFETY: the same contract.
        unsafe { crate::com::game_input_create(out) }
    }

    #[unsafe(no_mangle)]
    pub extern "system" fn DllCanUnloadNow() -> i32 {
        crate::com::dll_can_unload_now()
    }

    /// # Safety
    /// The pointers are null or valid for their type.
    #[unsafe(no_mangle)]
    pub unsafe extern "system" fn DllGetClassObject(clsid: *const crate::com::Guid, iid: *const crate::com::Guid, out: *mut *mut c_void) -> i32 {
        // SAFETY: the same contract.
        unsafe { crate::com::dll_get_class_object(clsid, iid, out) }
    }
}
