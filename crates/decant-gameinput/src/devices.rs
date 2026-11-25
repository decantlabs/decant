//! Where a real controller and a real keyboard feed the simulated pad.
//!
//! Three things live here: the on/off switch, the merge of the scripted state (`DECANT_PAD_SCRIPT` and
//! `DECANT_KEYS`) with what real hardware reports, and the hardware reads themselves. The parsing of the script and
//! of `DECANT_KEYS` is not part of this module.
//!
//! Design of the real input: one XInput controller in slot 0 is enough for the one simulated pad. The keyboard is a
//! second source that is only sampled while a window of this process has the focus, so a background run never
//! reacts to keys typed elsewhere. Both sources are combined with the same merge as the script, so a controller does
//! not silence the keyboard and the other way round.
//!
#![allow(unsafe_code)]

use crate::pad::PadState;

/// `DECANT_NO_REAL_INPUT`, checked for presence like `DECANT_PAD_SCRIPT` and `DECANT_KEYS` (any non-empty
/// value turns real input off; unset or empty leaves it on). With it set, [`poll`] always answers
/// a neutral [`PadState`] and nothing here is read, so a scripted run behaves exactly as it did
/// before this module existed.
pub fn enabled() -> bool {
    std::env::var_os("DECANT_NO_REAL_INPUT").filter(|v| !v.is_empty()).is_none()
}

/// The state the title sees: the buttons of both (a button is down if either says so, the same
/// rule [`crate::keys::combine`] uses for the script and `DECANT_KEYS`), and for each axis the real
/// device's value when it is not at rest, else the scripted one's.
pub fn merge(base: PadState, real: PadState) -> PadState {
    let pick = |r: f32, b: f32| if r != 0.0 { r } else { b };
    PadState {
        buttons: base.buttons | real.buttons,
        left_trigger: pick(real.left_trigger, base.left_trigger),
        right_trigger: pick(real.right_trigger, base.right_trigger),
        left_x: pick(real.left_x, base.left_x),
        left_y: pick(real.left_y, base.left_y),
        right_x: pick(real.right_x, base.right_x),
        right_y: pick(real.right_y, base.right_y),
    }
}

/// Standard XInput thumbstick dead zones (`XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE` /
/// `XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE` in `XInput.h`), as a fraction of full deflection
/// (raw range -32768..=32767): 7849 for the left stick, 8689 for the right.
const LEFT_STICK_DEADZONE: f32 = 7849.0 / 32767.0;
const RIGHT_STICK_DEADZONE: f32 = 8689.0 / 32767.0;

/// Standard XInput trigger threshold (`XINPUT_GAMEPAD_TRIGGER_THRESHOLD` in `XInput.h`): 30 of
/// 255, as a fraction of full travel.
const TRIGGER_THRESHOLD: f32 = 30.0 / 255.0;

/// A radial dead zone on one stick, already in -1.0..=1.0 units: inside `deadzone` the stick
/// reads centered; outside, the remaining travel is rescaled so full deflection still reaches
/// 1.0.
fn deadzone_stick(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
    let magnitude = x.hypot(y);
    if magnitude <= deadzone {
        return (0.0, 0.0);
    }
    let scale = ((magnitude - deadzone) / (1.0 - deadzone)).min(1.0) / magnitude;
    (x * scale, y * scale)
}

/// A trigger below the documented threshold reads as fully released, otherwise unchanged.
fn deadzone_trigger(value: f32) -> f32 {
    if value < TRIGGER_THRESHOLD { 0.0 } else { value }
}

/// Applies the standard XInput dead zones to one reading straight off a real device, before it
/// is compared for change (`runtime::poll_forever`) or merged with the script. Only ever called
/// on what a real controller answered: `DECANT_PAD_SCRIPT` and `DECANT_KEYS` values never pass through
/// here and are unchanged by it.
fn deadzone(raw: PadState) -> PadState {
    let (left_x, left_y) = deadzone_stick(raw.left_x, raw.left_y, LEFT_STICK_DEADZONE);
    let (right_x, right_y) = deadzone_stick(raw.right_x, raw.right_y, RIGHT_STICK_DEADZONE);
    PadState {
        left_x,
        left_y,
        right_x,
        right_y,
        left_trigger: deadzone_trigger(raw.left_trigger),
        right_trigger: deadzone_trigger(raw.right_trigger),
        ..raw
    }
}

/// The real pad and keyboard fallback, combined into one [`PadState`]; neutral when [`enabled`]
/// is false or no device/window answers. The real reading has the standard XInput dead zones
/// applied before anything else sees it.
pub fn poll() -> PadState {
    if !enabled() {
        return PadState::default();
    }
    #[cfg(windows)]
    let raw = read::poll();
    #[cfg(not(windows))]
    let raw = PadState::default();
    deadzone(raw)
}

/// The hardware reads. Stick and trigger values leave this module raw, the dead zones are applied by [`deadzone`].
#[cfg(windows)]
mod read {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK, XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB, XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XInputGetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

    use crate::pad::*;

    /// Which XInput bit sets which pad button.
    const XINPUT_TO_PAD: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
        (XINPUT_GAMEPAD_DPAD_UP, DPAD_UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DPAD_DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, DPAD_LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, DPAD_RIGHT),
    ];

    #[derive(Clone, Copy)]
    enum Axis {
        LeftX,
        LeftY,
        RightX,
        RightY,
        LeftTrigger,
        RightTrigger,
    }

    /// What a held key does to the pad. A key has no travel, so an axis binding means full deflection.
    #[derive(Clone, Copy)]
    enum Effect {
        Press(u32),
        Push(Axis, f32),
    }
    use Effect::{Press, Push};

    /// The keyboard layout of the stand-in pad, by virtual key code. The letters WASD are the left stick and IJKL the
    /// right stick. The arrow keys are the directional pad, Z and C the triggers, Q and E the bumpers, Enter and
    /// Backspace accept and cancel, Esc and Tab the two small centre buttons.
    const KEYMAP: [(u8, Effect); 24] = [
        (0x1B, Press(MENU)),
        (0x09, Press(VIEW)),
        (0x0D, Press(A)),
        (0x08, Press(B)),
        (b'X', Press(X)),
        (b'Y', Press(Y)),
        (b'Q', Press(LEFT_SHOULDER)),
        (b'E', Press(RIGHT_SHOULDER)),
        (b'F', Press(LEFT_STICK)),
        (b'H', Press(RIGHT_STICK)),
        (0x26, Press(DPAD_UP)),
        (0x28, Press(DPAD_DOWN)),
        (0x25, Press(DPAD_LEFT)),
        (0x27, Press(DPAD_RIGHT)),
        (b'W', Push(Axis::LeftY, 1.0)),
        (b'S', Push(Axis::LeftY, -1.0)),
        (b'D', Push(Axis::LeftX, 1.0)),
        (b'A', Push(Axis::LeftX, -1.0)),
        (b'I', Push(Axis::RightY, 1.0)),
        (b'K', Push(Axis::RightY, -1.0)),
        (b'L', Push(Axis::RightX, 1.0)),
        (b'J', Push(Axis::RightX, -1.0)),
        (b'Z', Push(Axis::LeftTrigger, 1.0)),
        (b'C', Push(Axis::RightTrigger, 1.0)),
    ];

    fn key_down(vk: u8) -> bool {
        // SAFETY: any virtual key code is accepted and nothing is written.
        (unsafe { GetAsyncKeyState(i32::from(vk)) }) < 0
    }

    /// True when the foreground window belongs to this process.
    fn own_window_focused() -> bool {
        // SAFETY: no arguments.
        let foreground: HWND = unsafe { GetForegroundWindow() };
        if foreground.is_null() {
            return false;
        }
        let mut owner = 0u32;
        // SAFETY: the window handle is not null and the out pointer is valid.
        unsafe { GetWindowThreadProcessId(foreground, &mut owner) };
        // SAFETY: no arguments.
        owner == unsafe { GetCurrentProcessId() }
    }

    /// XInput reports sticks as `i16` and triggers as `u8`. Map them to -1..=1 and 0..=1.
    fn pad_from_xinput(raw: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: XINPUT_TO_PAD.iter().fold(0, |acc, &(bit, button)| if raw.wButtons & bit != 0 { acc | button } else { acc }),
            left_trigger: f32::from(raw.bLeftTrigger) / 255.0,
            right_trigger: f32::from(raw.bRightTrigger) / 255.0,
            left_x: stick(raw.sThumbLX),
            left_y: stick(raw.sThumbLY),
            right_x: stick(raw.sThumbRX),
            right_y: stick(raw.sThumbRY),
        }
    }

    /// Asking XInput about an empty slot is slow, so a missing controller is only looked for again after this pause.
    const MISSING_PAUSE: Duration = Duration::from_secs(1);

    /// The earliest moment XInput may be asked again, set while slot 0 is empty.
    static RETRY_AT: Mutex<Option<Instant>> = Mutex::new(None);

    /// The controller in XInput slot 0, or a neutral pad when there is none.
    fn controller() -> PadState {
        let mut retry_at = RETRY_AT.lock().unwrap_or_else(|e| e.into_inner());
        if retry_at.is_some_and(|t| Instant::now() < t) {
            return PadState::default();
        }
        let mut state = XINPUT_STATE::default();
        // SAFETY: the out structure is the size XInput expects.
        let status = unsafe { XInputGetState(0, &mut state) };
        if status == 0 {
            *retry_at = None;
            pad_from_xinput(&state.Gamepad)
        } else {
            *retry_at = Some(Instant::now() + MISSING_PAUSE);
            PadState::default()
        }
    }

    /// The pad the held keys describe, neutral while no window of this process has the focus.
    fn keyboard() -> PadState {
        let mut pad = PadState::default();
        if !own_window_focused() {
            return pad;
        }
        for &(vk, effect) in &KEYMAP {
            if !key_down(vk) {
                continue;
            }
            match effect {
                Press(mask) => pad.buttons |= mask,
                Push(axis, amount) => {
                    let slot = match axis {
                        Axis::LeftX => &mut pad.left_x,
                        Axis::LeftY => &mut pad.left_y,
                        Axis::RightX => &mut pad.right_x,
                        Axis::RightY => &mut pad.right_y,
                        Axis::LeftTrigger => &mut pad.left_trigger,
                        Axis::RightTrigger => &mut pad.right_trigger,
                    };
                    *slot = (*slot + amount).clamp(-1.0, 1.0);
                }
            }
        }
        pad
    }

    /// Controller and keyboard combined.
    pub fn poll() -> PadState {
        super::merge(controller(), keyboard())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pad::{A, B, LEFT_STICK};

    #[test]
    fn real_axes_win_over_the_script_when_not_at_rest() {
        let base = PadState { buttons: A, left_x: 0.5, right_trigger: 1.0, ..PadState::default() };
        let real = PadState { buttons: B, left_x: -1.0, ..PadState::default() };
        assert_eq!(merge(base, real), PadState { buttons: A | B, left_x: -1.0, right_trigger: 1.0, ..PadState::default() });
    }

    #[test]
    fn a_rested_real_axis_leaves_the_script_value() {
        let base = PadState { left_y: 0.75, ..PadState::default() };
        let real = PadState::default();
        assert_eq!(merge(base, real).left_y, 0.75);
    }

    #[test]
    fn buttons_are_ored_not_replaced() {
        let base = PadState { buttons: LEFT_STICK, ..PadState::default() };
        let real = PadState { buttons: A, ..PadState::default() };
        assert_eq!(merge(base, real).buttons, LEFT_STICK | A);
    }

    #[test]
    fn two_neutral_states_merge_to_neutral() {
        assert_eq!(merge(PadState::default(), PadState::default()), PadState::default());
    }

    #[test]
    fn drifting_real_stick_reads_as_centered() {
        let raw = PadState { left_x: 0.02, left_y: 0.0, right_x: -0.02, ..PadState::default() };
        let out = deadzone(raw);
        assert_eq!((out.left_x, out.left_y, out.right_x), (0.0, 0.0, 0.0));
    }

    #[test]
    fn full_deflection_still_reaches_one() {
        let raw = PadState { left_x: 1.0, right_y: -1.0, ..PadState::default() };
        let out = deadzone(raw);
        assert!((out.left_x - 1.0).abs() < 1e-6, "{}", out.left_x);
        assert!((out.right_y + 1.0).abs() < 1e-6, "{}", out.right_y);
    }

    #[test]
    fn just_past_the_deadzone_is_small_and_positive() {
        let raw = PadState { left_x: LEFT_STICK_DEADZONE + 0.001, ..PadState::default() };
        let out = deadzone(raw);
        assert!(out.left_x > 0.0 && out.left_x < 0.01, "{}", out.left_x);
    }

    #[test]
    fn a_trigger_below_the_threshold_reads_as_released() {
        let raw = PadState { left_trigger: 10.0 / 255.0, right_trigger: 200.0 / 255.0, ..PadState::default() };
        let out = deadzone(raw);
        assert_eq!(out.left_trigger, 0.0);
        assert_eq!(out.right_trigger, 200.0 / 255.0);
    }

    #[test]
    fn scripted_values_are_never_deadzoned() {
        // `deadzone` only ever sees what a real device answered (wired in `poll`); a scripted
        // value equal to the drift above stays exactly what DECANT_PAD_SCRIPT/DECANT_KEYS set once
        // merged, because merge's `base` operand never goes through `deadzone`.
        let scripted = PadState { left_x: 0.02, ..PadState::default() };
        assert_eq!(merge(scripted, PadState::default()).left_x, 0.02);
    }
}
