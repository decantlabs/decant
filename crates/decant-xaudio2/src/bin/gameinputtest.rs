//! `gameinputtest`: GameInput needs no compatibility layer. The public GDK documentation states
//! the same API and the same header on PC and Xbox (docs/AUDIO_INPUT.md). This program checks
//! what that implies on a PC, with no input device and no GDK header: the inbox `GameInput.dll`
//! exports `GameInputCreate`, the call returns an `IGameInput`, and the object answers `IUnknown`.
//! The port of the former `tests/gameinputtest.cpp`, with the same lines and exit codes. It also
//! asks for the current gamepad and keyboard readings and accepts either a reading or the "no
//! reading" status (`GAMEINPUT_E_READING_NOT_FOUND`). The C program did that only when built with
//! the GDK header `GameInput.h`. Here the call goes through the vtable slot the header declares
//! (`IGameInput::GetCurrentReading`, after `GetCurrentTimestamp`), so no header is needed.
//! Synthetic: it presses no key and needs no controller. Without `GameInput.dll` it prints SKIP and passes, unless `DECANT_GI_REQUIRE` is set.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(run::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("gameinputtest runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::c_void;

    use decant_testkit::{Checks, Library};
    use windows::core::{HRESULT, IUnknown, IUnknown_Vtbl, Interface};

    type CreateFn = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;

    /// `IGameInput` up to `GetCurrentReading` (GameInput.h, `IGameInput`).
    #[repr(C)]
    struct GameInputVtbl {
        _base: IUnknown_Vtbl,
        _get_current_timestamp: unsafe extern "system" fn(*mut c_void) -> u64,
        get_current_reading: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, *mut *mut c_void) -> HRESULT,
    }

    const KIND_KEYBOARD: u32 = 0x0000_0010;
    const KIND_GAMEPAD: u32 = 0x0004_0000;
    /// `GAMEINPUT_E_READING_NOT_FOUND`.
    const E_READING_NOT_FOUND: i32 = 0x838A_0003_u32 as i32;

    pub fn main() -> u8 {
        let lib = match Library::open("GameInput.dll".as_ref()) {
            Ok(lib) => lib,
            Err(error) => {
                println!("  SKIP: GameInput.dll is not installed on this machine (error {error})");
                return u8::from(std::env::var_os("DECANT_GI_REQUIRE").is_some());
            }
        };
        let mut c = Checks::new(72, "OK", "FAIL");
        // SAFETY: GameInputCreate takes one output interface pointer (GameInput.h).
        let create = unsafe { lib.get::<CreateFn>("GameInputCreate") };
        c.check(create.is_some(), "GameInput.dll exports GameInputCreate");
        let Some(create) = create else {
            return 1;
        };
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: `raw` is a valid output location.
        let hr = unsafe { create(&mut raw) };
        c.check(hr.is_ok() && !raw.is_null(), "GameInputCreate returns an IGameInput");
        if raw.is_null() {
            return 1;
        }
        // SAFETY: a non null result is a COM object with one reference, owned here.
        let gi = unsafe { IUnknown::from_raw(raw) };
        c.check(gi.cast::<IUnknown>().is_ok(), "the object answers IUnknown");
        for (kind, what) in [(KIND_GAMEPAD, "gamepad"), (KIND_KEYBOARD, "keyboard")] {
            let mut reading: *mut c_void = std::ptr::null_mut();
            // SAFETY: `gi` is a live IGameInput whose vtable starts as `GameInputVtbl` says. The
            // device is null (any device) and `reading` is a valid output location.
            let hr = unsafe {
                let vtable = &**gi.as_raw().cast::<*const GameInputVtbl>();
                (vtable.get_current_reading)(gi.as_raw(), kind, std::ptr::null_mut(), &mut reading)
            };
            c.check(hr.is_ok() || hr.0 == E_READING_NOT_FOUND, &format!("GetCurrentReading({what}) gives a reading or 'not found'"));
            if !reading.is_null() {
                // SAFETY: a non null reading is a COM object with one reference, released here.
                drop(unsafe { IUnknown::from_raw(reading) });
            }
        }
        drop(gi);
        u8::from(c.failures() != 0)
    }
}
