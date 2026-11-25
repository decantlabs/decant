//! The COM objects of the shim (`IGameInput`, `IGameInputDevice`, `IGameInputReading` of the first GameInput interface set)
//! and the exports that hand them out.
//!
//! Vtables are plain tables of function addresses in the order of the public `GameInput.h`. A method that is not written
//! answers `E_NOTIMPL` (or an empty answer for the readings) and is logged the first time it is called, so the log shows
//! what a title really uses.
#![allow(unsafe_code)]

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use decant_log::trace;

use crate::pad::PadState;
use crate::runtime;

pub type Hresult = i32;
pub const S_OK: Hresult = 0;
pub const S_FALSE: Hresult = 1;
pub const E_NOTIMPL: Hresult = 0x8000_4001_u32 as i32;
pub const E_NOINTERFACE: Hresult = 0x8000_4002_u32 as i32;
pub const E_POINTER: Hresult = 0x8000_4003_u32 as i32;
pub const CLASS_E_CLASSNOTAVAILABLE: Hresult = 0x8004_0111_u32 as i32;
pub const GAMEINPUT_E_DEVICE_NOT_FOUND: Hresult = 0x838A_0002_u32 as i32;
pub const GAMEINPUT_E_READING_NOT_FOUND: Hresult = 0x838A_0003_u32 as i32;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub d1: u32,
    pub d2: u16,
    pub d3: u16,
    pub d4: [u8; 8],
}

pub const IID_IUNKNOWN: Guid = Guid { d1: 0, d2: 0, d3: 0, d4: [0xC0, 0, 0, 0, 0, 0, 0, 0x46] };
pub const IID_IGAMEINPUT: Guid = Guid { d1: 0x11be2a7e, d2: 0x4254, d3: 0x445a, d4: [0x9c, 0x09, 0xff, 0xc4, 0x0f, 0x00, 0x69, 0x18] };
pub const IID_IGAMEINPUT_READING: Guid = Guid { d1: 0x2156947a, d2: 0xe1fa, d3: 0x4de0, d4: [0xa3, 0x0b, 0xd8, 0x12, 0x93, 0x1d, 0xbd, 0x8d] };
pub const IID_IGAMEINPUT_DEVICE: Guid = Guid { d1: 0x31dd86fb, d2: 0x4c1b, d3: 0x408a, d4: [0x86, 0x8f, 0x43, 0x9b, 0x3c, 0xd4, 0x71, 0x25] };

fn guid_text(g: &Guid) -> String {
    let d = g.d4;
    format!("{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", g.d1, g.d2, g.d3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7])
}

// GameInputKind
const KIND_CONTROLLER: u32 = 0xE;
const KIND_GAMEPAD: u32 = 0x4_0000;
const KIND_UI_NAVIGATION: u32 = 0x100_0000;
const KINDS: u32 = KIND_CONTROLLER | KIND_GAMEPAD | KIND_UI_NAVIGATION;
// GameInputDeviceStatus: connected, input enabled, output enabled, synchronized
const STATUS: u32 = 0x1 | 0x2 | 0x4 | 0x40;
// GameInputEnumerationKind
const ENUMERATE_BLOCKING: u32 = 2;

const DEVICE_ID: [u8; 32] = *b"Decant gamepad 0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

#[repr(C)]
struct Object {
    vtable: *const usize,
}

// The two static objects only ever point at tables that are leaked and never written again.
unsafe impl Sync for Object {}
unsafe impl Send for Object {}

#[repr(C)]
struct Reading {
    vtable: *const usize,
    refs: AtomicU32,
    state: PadState,
    sequence: u64,
    timestamp: u64,
}

/// The record `IGameInputDevice::GetDeviceInfo` returns (`GameInputDeviceInfo`). The field order, the field widths and
/// the 1024 byte total are the ABI of the public header, so a title can read it in place. Only the leading part is
/// filled in. Everything from `rest` on describes axes, buttons, switches, reports and force feedback that this
/// virtual pad does not have, and zero is the documented "none" for all of it.
#[repr(C)]
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage_page: u16,
    usage_id: u16,
    hardware_version: [u16; 4],
    firmware_version: [u16; 4],
    device_id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    rest: [u8; 896],
}

/// What the virtual pad claims to be. The choices are ours and the reasoning is:
/// * The USB ids are those of Microsoft's Xbox Series controller (a public fact), because a title that picks glyphs or
///   rumble layouts by vendor and product expects a real pad there.
/// * Usage page 1 (generic desktop) with usage 5 (game pad) is the HID description of any gamepad.
/// * The revision and the two version quadruples are fixed placeholders, there is no real firmware behind them.
/// * The device id is the text "Decant gamepad 0" padded with zeros, the same for the device and its root.
/// * The family is the Xbox One one, which the Series controllers report as well.
/// * The only capability is synchronization, and the rumble motors are all four of the Xbox set.
mod identity {
    pub const VENDOR_MICROSOFT: u16 = 0x045E;
    pub const PRODUCT_XBOX_SERIES_PAD: u16 = 0x0B12;
    pub const USAGE_PAGE_GENERIC_DESKTOP: u16 = 0x01;
    pub const USAGE_GAME_PAD: u16 = 0x05;
    pub const REVISION: u16 = 0x0100;
    pub const HARDWARE_VERSION: [u16; 4] = [1, 0, 0, 0];
    pub const FIRMWARE_VERSION: [u16; 4] = [5, 0, 0, 0];
    pub const FAMILY_XBOX_ONE: i32 = 1;
    pub const CAPABILITY_SYNCHRONIZATION: u32 = 0x8;
    pub const MOTOR_LOW_FREQUENCY: u32 = 1 << 0;
    pub const MOTOR_HIGH_FREQUENCY: u32 = 1 << 1;
    pub const MOTOR_LEFT_TRIGGER: u32 = 1 << 2;
    pub const MOTOR_RIGHT_TRIGGER: u32 = 1 << 3;
}

static DEVICE_INFO: DeviceInfo = {
    use identity::*;
    DeviceInfo {
        size: std::mem::size_of::<DeviceInfo>() as u32,
        vendor: VENDOR_MICROSOFT,
        product: PRODUCT_XBOX_SERIES_PAD,
        revision: REVISION,
        interface: 0,
        collection: 0,
        usage_page: USAGE_PAGE_GENERIC_DESKTOP,
        usage_id: USAGE_GAME_PAD,
        hardware_version: HARDWARE_VERSION,
        firmware_version: FIRMWARE_VERSION,
        device_id: DEVICE_ID,
        root_id: DEVICE_ID,
        family: FAMILY_XBOX_ONE,
        capabilities: CAPABILITY_SYNCHRONIZATION,
        supported_input: KINDS,
        rumble_motors: MOTOR_LOW_FREQUENCY | MOTOR_HIGH_FREQUENCY | MOTOR_LEFT_TRIGGER | MOTOR_RIGHT_TRIGGER,
        rest: [0; 896],
    }
};

fn device_info() -> &'static DeviceInfo {
    &DEVICE_INFO
}

// ---- method names, for the first call log ----

const INPUT_METHODS: [&str; 22] = [
    "QueryInterface",
    "AddRef",
    "Release",
    "GetCurrentTimestamp",
    "GetCurrentReading",
    "GetNextReading",
    "GetPreviousReading",
    "GetTemporalReading",
    "RegisterReadingCallback",
    "RegisterDeviceCallback",
    "RegisterSystemButtonCallback",
    "RegisterKeyboardLayoutCallback",
    "StopCallback",
    "UnregisterCallback",
    "CreateDispatcher",
    "CreateAggregateDevice",
    "FindDeviceFromId",
    "FindDeviceFromObject",
    "FindDeviceFromPlatformHandle",
    "FindDeviceFromPlatformString",
    "EnableOemDeviceSupport",
    "SetFocusPolicy",
];

const DEVICE_METHODS: [&str; 14] = [
    "QueryInterface",
    "AddRef",
    "Release",
    "GetDeviceInfo",
    "GetDeviceStatus",
    "GetBatteryState",
    "CreateForceFeedbackEffect",
    "IsForceFeedbackMotorPoweredOn",
    "SetForceFeedbackMotorGain",
    "SetHapticMotorState",
    "SetRumbleState",
    "SetInputSynchronizationState",
    "SendInputSynchronizationHint",
    "PowerOff",
];

const READING_METHODS: [&str; 25] = [
    "QueryInterface",
    "AddRef",
    "Release",
    "GetInputKind",
    "GetSequenceNumber",
    "GetTimestamp",
    "GetDevice",
    "GetRawReport",
    "GetControllerAxisCount",
    "GetControllerButtonCount",
    "GetControllerSwitchCount",
    "GetControllerAxisState",
    "GetControllerButtonState",
    "GetControllerSwitchState",
    "GetKeyCount",
    "GetKeyState",
    "GetMouseState",
    "GetTouchCount",
    "GetTouchState",
    "GetMotionState",
    "GetArcadeStickState",
    "GetFlightStickState",
    "GetGamepadState",
    "GetRacingWheelState",
    "GetUiNavigationState",
];

#[derive(Clone, Copy)]
enum Kind {
    Input = 0,
    Device = 1,
    Reading = 2,
}

fn method_name(kind: Kind, slot: usize) -> String {
    let (owner, names): (&str, &[&str]) = match kind {
        Kind::Input => ("IGameInput", &INPUT_METHODS),
        Kind::Device => ("IGameInputDevice", &DEVICE_METHODS),
        Kind::Reading => ("IGameInputReading", &READING_METHODS),
    };
    match names.get(slot) {
        Some(name) => format!("{owner}::{name}"),
        None => format!("{owner} slot {slot}"),
    }
}

/// Logs the first call of a method nobody wrote (once per method).
fn note_unwritten(kind: Kind, slot: usize, answer: &str) {
    static SEEN: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    let bit = 1u64 << (slot & 63);
    let unanswered = if answer == "E_NOTIMPL" { decant_log::calls::Kind::NotImplemented } else { decant_log::calls::Kind::Stub };
    decant_log::calls::record_with(unanswered, || method_name(kind, slot));
    if SEEN[kind as usize].fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        trace(&format!("gameinput: {} called, not implemented ({answer})", method_name(kind, slot)));
    }
}

/// A method that fails with `E_NOTIMPL`. Extra arguments are ignored, as the x64 convention allows.
unsafe extern "system" fn unimplemented_method<const KIND: u8, const SLOT: usize>() -> Hresult {
    let kind = match KIND {
        0 => Kind::Input,
        1 => Kind::Device,
        _ => Kind::Reading,
    };
    note_unwritten(kind, SLOT, "E_NOTIMPL");
    E_NOTIMPL
}

/// A reading method that counts something: this pad has none of it.
unsafe extern "system" fn empty_count<const SLOT: usize>() -> u32 {
    note_unwritten(Kind::Reading, SLOT, "none");
    0
}

/// A reading method that fills in a state of a device kind this pad is not.
unsafe extern "system" fn no_state<const SLOT: usize>() -> bool {
    note_unwritten(Kind::Reading, SLOT, "false");
    false
}

macro_rules! unimplemented_table {
    ($kind:literal; $($slot:literal)*) => { [$(unimplemented_method::<$kind, $slot> as *const () as usize),*] };
}

/// Makes a vtable: every slot answers `E_NOTIMPL`, then the written methods are put in. Leaked, as the objects live as long as the process.
fn leak_table(mut slots: Vec<usize>, written: &[(usize, usize)]) -> *const usize {
    for &(slot, function) in written {
        slots[slot] = function;
    }
    Box::leak(slots.into_boxed_slice()).as_ptr()
}

fn input_object() -> *mut Object {
    static OBJECT: OnceLock<usize> = OnceLock::new();
    *OBJECT.get_or_init(|| {
        let defaults = unimplemented_table!(0; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23);
        let table = leak_table(
            defaults.to_vec(),
            &[
                (0, input_query_interface as *const () as usize),
                (1, add_ref_static as *const () as usize),
                (2, release_static as *const () as usize),
                (3, get_current_timestamp as *const () as usize),
                (4, get_current_reading as *const () as usize),
                (5, other_reading as *const () as usize),
                (6, other_reading as *const () as usize),
                (7, get_temporal_reading as *const () as usize),
                (8, register_reading_callback as *const () as usize),
                (9, register_device_callback as *const () as usize),
                (10, register_other_callback::<5> as *const () as usize),
                (11, register_other_callback::<4> as *const () as usize),
                (12, stop_callback as *const () as usize),
                (13, unregister_callback as *const () as usize),
                (16, find_device_from_id as *const () as usize),
                (20, enable_oem_device_support as *const () as usize),
                (21, set_focus_policy as *const () as usize),
            ],
        );
        Box::leak(Box::new(Object { vtable: table })) as *mut Object as usize
    }) as *mut Object
}

fn device_object() -> *mut Object {
    static OBJECT: OnceLock<usize> = OnceLock::new();
    *OBJECT.get_or_init(|| {
        let defaults = unimplemented_table!(1; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23);
        let table = leak_table(
            defaults.to_vec(),
            &[
                (0, device_query_interface as *const () as usize),
                (1, add_ref_static as *const () as usize),
                (2, release_static as *const () as usize),
                (3, get_device_info as *const () as usize),
                (4, get_device_status as *const () as usize),
                (5, get_battery_state as *const () as usize),
                (7, motor_powered_on as *const () as usize),
                (8, do_nothing as *const () as usize),
                (10, set_rumble_state as *const () as usize),
                (11, do_nothing as *const () as usize),
                (12, do_nothing as *const () as usize),
                (13, do_nothing as *const () as usize),
            ],
        );
        Box::leak(Box::new(Object { vtable: table })) as *mut Object as usize
    }) as *mut Object
}

fn reading_table() -> *const usize {
    static TABLE: OnceLock<usize> = OnceLock::new();
    *TABLE.get_or_init(|| {
        let defaults = unimplemented_table!(2; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);
        leak_table(
            defaults.to_vec(),
            &[
                (0, reading_query_interface as *const () as usize),
                (1, reading_add_ref as *const () as usize),
                (2, reading_release as *const () as usize),
                (3, reading_kind as *const () as usize),
                (4, reading_sequence as *const () as usize),
                (5, reading_timestamp as *const () as usize),
                (6, reading_device as *const () as usize),
                (7, no_state::<7> as *const () as usize),
                (8, empty_count::<8> as *const () as usize),
                (9, empty_count::<9> as *const () as usize),
                (10, empty_count::<10> as *const () as usize),
                (11, empty_count::<11> as *const () as usize),
                (12, empty_count::<12> as *const () as usize),
                (13, empty_count::<13> as *const () as usize),
                (14, empty_count::<14> as *const () as usize),
                (15, empty_count::<15> as *const () as usize),
                (16, no_state::<16> as *const () as usize),
                (17, empty_count::<17> as *const () as usize),
                (18, empty_count::<18> as *const () as usize),
                (19, no_state::<19> as *const () as usize),
                (20, no_state::<20> as *const () as usize),
                (21, no_state::<21> as *const () as usize),
                (22, reading_gamepad_state as *const () as usize),
                (23, no_state::<23> as *const () as usize),
                (24, reading_ui_navigation_state as *const () as usize),
            ],
        ) as usize
    }) as *const usize
}

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn new_reading(state: PadState) -> *mut Reading {
    Box::into_raw(Box::new(Reading {
        vtable: reading_table(),
        refs: AtomicU32::new(1),
        state,
        sequence: SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
        timestamp: runtime::now_us(),
    }))
}

// ---- IUnknown ----

/// Answers `own` and `IUnknown`; any other interface is refused, and logged once.
///
/// # Safety
/// `iid` is null or readable, `out` is null or writable.
unsafe fn query(this: *mut c_void, iid: *const Guid, out: *mut *mut c_void, own: &Guid, which: &str, on_success: impl FnOnce()) -> Hresult {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `iid` is readable when not null.
    let wanted = unsafe { iid.as_ref() };
    if wanted.is_some_and(|g| g == own || *g == IID_IUNKNOWN) {
        on_success();
        // SAFETY: `out` is writable.
        unsafe { *out = this };
        return S_OK;
    }
    if let Some(g) = wanted {
        static SEEN: Mutex<Vec<[u8; 16]>> = Mutex::new(Vec::new());
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&g.d1.to_le_bytes());
        bytes[4..6].copy_from_slice(&g.d2.to_le_bytes());
        bytes[6..8].copy_from_slice(&g.d3.to_le_bytes());
        bytes[8..].copy_from_slice(&g.d4);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&bytes) && seen.len() < 16 {
            seen.push(bytes);
            trace(&format!("gameinput: {which}::QueryInterface({}) refused, E_NOINTERFACE", guid_text(g)));
        }
    }
    // SAFETY: `out` is writable.
    unsafe { *out = std::ptr::null_mut() };
    E_NOINTERFACE
}

unsafe extern "system" fn input_query_interface(this: *mut Object, iid: *const Guid, out: *mut *mut c_void) -> Hresult {
    // SAFETY: arguments as COM promises.
    unsafe { query(this.cast(), iid, out, &IID_IGAMEINPUT, "IGameInput", || {}) }
}

unsafe extern "system" fn device_query_interface(this: *mut Object, iid: *const Guid, out: *mut *mut c_void) -> Hresult {
    // SAFETY: arguments as COM promises.
    unsafe { query(this.cast(), iid, out, &IID_IGAMEINPUT_DEVICE, "IGameInputDevice", || {}) }
}

unsafe extern "system" fn reading_query_interface(this: *mut Reading, iid: *const Guid, out: *mut *mut c_void) -> Hresult {
    // SAFETY: arguments as COM promises, and `this` is a live reading.
    unsafe { query(this.cast(), iid, out, &IID_IGAMEINPUT_READING, "IGameInputReading", || {
        (*this).refs.fetch_add(1, Ordering::AcqRel);
    }) }
}

/// The input object and the device live for the whole process, so their counts mean nothing.
extern "system" fn add_ref_static(_this: *mut c_void) -> u32 {
    1
}

extern "system" fn release_static(_this: *mut c_void) -> u32 {
    1
}

unsafe extern "system" fn reading_add_ref(this: *mut Reading) -> u32 {
    // SAFETY: a live reading.
    unsafe { (*this).refs.fetch_add(1, Ordering::AcqRel) + 1 }
}

unsafe extern "system" fn reading_release(this: *mut Reading) -> u32 {
    // SAFETY: a live reading, freed when its last reference goes.
    unsafe {
        let left = (*this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            drop(Box::from_raw(this));
        }
        left
    }
}

extern "system" fn do_nothing() {}

// ---- IGameInputDevice ----

extern "system" fn get_device_info(_this: *mut Object) -> *const DeviceInfo {
    device_info()
}

extern "system" fn get_device_status(_this: *mut Object) -> u32 {
    STATUS
}

/// `GameInputBatteryState`: all zero, which is "unknown" (the pad is wired).
unsafe extern "system" fn get_battery_state(_this: *mut Object, state: *mut [u32; 5]) {
    if !state.is_null() {
        // SAFETY: writable as documented.
        unsafe { *state = [0; 5] };
    }
}

extern "system" fn motor_powered_on(_this: *mut Object, _motor: u32) -> bool {
    false
}

/// Rumble goes nowhere: there is no hardware.
extern "system" fn set_rumble_state(_this: *mut Object, _params: *const [f32; 4]) {}

// ---- IGameInput ----

extern "system" fn get_current_timestamp(_this: *mut Object) -> u64 {
    runtime::now_us()
}

/// Whether a call names our device (or none, which means any).
fn is_ours(device: *mut c_void) -> bool {
    device.is_null() || device == device_object().cast()
}

/// Logs a kind of query the first time a title asks for it (which input a title reads, and how).
fn note_query(what: &'static str, kind: u32) {
    static SEEN: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 32 && !seen.contains(&(what, kind)) {
        seen.push((what, kind));
        trace(&format!("gameinput: {what} for kinds {kind:#x}"));
    }
}

unsafe extern "system" fn get_current_reading(_this: *mut Object, kind: u32, device: *mut c_void, out: *mut *mut Reading) -> Hresult {
    note_query("GetCurrentReading", kind);
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: writable as checked.
    unsafe { *out = std::ptr::null_mut() };
    if kind & KINDS == 0 || !is_ours(device) {
        return GAMEINPUT_E_READING_NOT_FOUND;
    }
    // SAFETY: as above.
    unsafe { *out = new_reading(runtime::state()) };
    S_OK
}

/// Readings are not kept, so there is none before or after another.
unsafe extern "system" fn other_reading(_this: *mut Object, _reference: *mut c_void, kind: u32, _device: *mut c_void, out: *mut *mut Reading) -> Hresult {
    note_query("GetNextReading or GetPreviousReading", kind);
    if !out.is_null() {
        // SAFETY: writable as checked.
        unsafe { *out = std::ptr::null_mut() };
    }
    GAMEINPUT_E_READING_NOT_FOUND
}

unsafe extern "system" fn get_temporal_reading(this: *mut Object, _timestamp: u64, device: *mut c_void, out: *mut *mut Reading) -> Hresult {
    // SAFETY: the same arguments as the current reading takes.
    unsafe { get_current_reading(this, KIND_GAMEPAD, device, out) }
}

type ReadingCallback = unsafe extern "system" fn(u64, *mut c_void, *mut Reading, bool);
type DeviceCallback = unsafe extern "system" fn(u64, *mut c_void, *mut Object, u64, u32, u32);

/// A registered callback and what it takes to stop it safely.
struct Entry {
    token: u64,
    reading: Option<(usize, ReadingCallback)>,
    /// Calls of the callback running now, on any thread.
    running: AtomicU32,
    stopped: AtomicBool,
}

impl Entry {
    /// Counts a call in, unless the callback was stopped. Call [`Entry::leave`] after it.
    fn enter(&self) -> bool {
        self.running.fetch_add(1, Ordering::SeqCst);
        if self.stopped.load(Ordering::SeqCst) {
            self.running.fetch_sub(1, Ordering::SeqCst);
            return false;
        }
        CALLBACK_DEPTH.with(|d| d.set(d.get() + 1));
        true
    }

    fn leave(&self) {
        CALLBACK_DEPTH.with(|d| d.set(d.get() - 1));
        self.running.fetch_sub(1, Ordering::SeqCst);
    }
}

thread_local! {
    /// How many callbacks of ours this thread is inside, so a callback may unregister itself without waiting for itself.
    static CALLBACK_DEPTH: Cell<u32> = const { Cell::new(0) };
}

static TOKENS: AtomicU64 = AtomicU64::new(0);
static ENTRIES: Mutex<Vec<Arc<Entry>>> = Mutex::new(Vec::new());

fn entries() -> std::sync::MutexGuard<'static, Vec<Arc<Entry>>> {
    ENTRIES.lock().unwrap_or_else(|e| e.into_inner())
}

fn new_token(out: *mut u64) -> u64 {
    let token = TOKENS.fetch_add(1, Ordering::Relaxed) + 1;
    if !out.is_null() {
        // SAFETY: writable as checked.
        unsafe { *out = token };
    }
    token
}

/// Calls every reading callback with a reading of `state`. Called by the polling thread when the pad changes.
pub fn dispatch_reading_callbacks(state: PadState) {
    let list: Vec<Arc<Entry>> = entries().iter().filter(|e| e.reading.is_some()).cloned().collect();
    for entry in list {
        let Some((context, callback)) = entry.reading else { continue };
        if !entry.enter() {
            continue;
        }
        let reading = new_reading(state);
        // SAFETY: the callback and its context come from the registration, and `Entry::running` keeps UnregisterCallback
        // from returning (and the title from freeing the context) until the call is over.
        unsafe {
            callback(entry.token, context as *mut c_void, reading, false);
            reading_release(reading);
        }
        entry.leave();
    }
}

/// RegisterReadingCallback(device, kind, analog threshold, context, callback, token)
unsafe extern "system" fn register_reading_callback(
    _this: *mut Object,
    device: *mut c_void,
    kind: u32,
    _threshold: f32,
    context: *mut c_void,
    callback: Option<ReadingCallback>,
    token: *mut u64,
) -> Hresult {
    note_query("RegisterReadingCallback", kind);
    let token = new_token(token);
    if let Some(callback) = callback.filter(|_| kind & KINDS != 0 && is_ours(device)) {
        entries().push(Arc::new(Entry { token, reading: Some((context as usize, callback)), running: AtomicU32::new(0), stopped: AtomicBool::new(false) }));
    }
    S_OK
}

/// RegisterDeviceCallback(device, kind, status filter, enumeration kind, context, callback, token): the pad, connected.
/// A blocking enumeration calls at once, before this returns. An asynchronous one calls from another thread. Without
/// enumeration nothing is called, because the pad never comes or goes.
unsafe extern "system" fn register_device_callback(
    _this: *mut Object,
    device: *mut c_void,
    kind: u32,
    _status_filter: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    token: *mut u64,
) -> Hresult {
    let token = new_token(token);
    trace(&format!("gameinput: RegisterDeviceCallback for kinds {kind:#x}, enumeration {enumeration}"));
    let Some(callback) = callback.filter(|_| kind & KINDS != 0 && is_ours(device) && enumeration != 0) else { return S_OK };
    if enumeration == ENUMERATE_BLOCKING {
        // SAFETY: the callback and its context come from the caller.
        unsafe { callback(token, context, device_object(), runtime::now_us(), STATUS, 0) };
        return S_OK;
    }
    let entry = Arc::new(Entry { token, reading: None, running: AtomicU32::new(0), stopped: AtomicBool::new(false) });
    entries().push(Arc::clone(&entry));
    let context = context as usize;
    let spawned = std::thread::Builder::new().name("decant-gameinput-device".into()).spawn(move || {
        if entry.enter() {
            // SAFETY: as above, and `running` holds off UnregisterCallback until the call returns.
            unsafe { callback(token, context as *mut c_void, device_object(), runtime::now_us(), STATUS, 0) };
            entry.leave();
        }
        entries().retain(|e| e.token != token);
    });
    if spawned.is_err() {
        entries().retain(|e| e.token != token);
    }
    S_OK
}

/// RegisterSystemButtonCallback (device, buttons, context, callback, token: 5 arguments) and RegisterKeyboardLayoutCallback
/// (device, context, callback, token: 4): accepted and never called, as there are no system buttons and no keyboard.
unsafe extern "system" fn register_other_callback<const ARGS: usize>(_this: *mut Object, _p1: *mut c_void, _p2: usize, _p3: *mut c_void, p4: *mut c_void, p5: *mut u64) -> Hresult {
    new_token(if ARGS == 5 { p5 } else { p4.cast::<u64>() });
    S_OK
}

fn find(token: u64) -> Option<Arc<Entry>> {
    entries().iter().find(|e| e.token == token).cloned()
}

extern "system" fn stop_callback(_this: *mut Object, token: u64) {
    if let Some(entry) = find(token) {
        entry.stopped.store(true, Ordering::SeqCst);
        if entry.running.load(Ordering::SeqCst) == 0 {
            entries().retain(|e| e.token != token);
        }
    }
}

/// UnregisterCallback(token, timeout in microseconds): true once no call of the callback is running. A callback that
/// unregisters itself does not wait for itself.
extern "system" fn unregister_callback(_this: *mut Object, token: u64, timeout_us: u64) -> bool {
    let Some(entry) = find(token) else { return true };
    entry.stopped.store(true, Ordering::SeqCst);
    let own = CALLBACK_DEPTH.with(Cell::get);
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout_us));
    loop {
        if entry.running.load(Ordering::SeqCst) <= own {
            entries().retain(|e| e.token != token);
            return true;
        }
        if deadline.is_some_and(|end| Instant::now() >= end) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// FindDeviceFromId(&id, &device)
unsafe extern "system" fn find_device_from_id(_this: *mut Object, id: *const [u8; 32], out: *mut *mut Object) -> Hresult {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `id` is readable when not null, `out` is writable.
    unsafe {
        if !id.is_null() && *id == DEVICE_ID {
            *out = device_object();
            return S_OK;
        }
        *out = std::ptr::null_mut();
    }
    GAMEINPUT_E_DEVICE_NOT_FOUND
}

extern "system" fn enable_oem_device_support(_this: *mut Object, _vendor: u16, _product: u16, _interface: u8, _collection: u8) -> Hresult {
    S_OK
}

extern "system" fn set_focus_policy(_this: *mut Object, _policy: u32) {}

// ---- IGameInputReading ----

extern "system" fn reading_kind(_this: *mut Reading) -> u32 {
    KINDS
}

unsafe extern "system" fn reading_sequence(this: *mut Reading, _kind: u32) -> u64 {
    // SAFETY: a live reading.
    unsafe { (*this).sequence }
}

unsafe extern "system" fn reading_timestamp(this: *mut Reading) -> u64 {
    // SAFETY: a live reading.
    unsafe { (*this).timestamp }
}

unsafe extern "system" fn reading_device(_this: *mut Reading, out: *mut *mut Object) {
    if !out.is_null() {
        // SAFETY: writable as checked.
        unsafe { *out = device_object() };
    }
}

unsafe extern "system" fn reading_gamepad_state(this: *mut Reading, out: *mut PadState) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: a live reading and a writable out.
    unsafe { *out = (*this).state };
    true
}

/// The UI navigation view of a reading. Menu code reads a small fixed vocabulary (menu, view, accept, cancel, the four
/// directions, two extra face buttons and the bumpers as page keys), each a bit of `GameInputUiNavigationButtons`.
/// The thumbstick clicks have no navigation meaning and are dropped.
fn ui_navigation(buttons: u32) -> u32 {
    use crate::pad::*;
    const NAV_MENU: u32 = 1 << 0;
    const NAV_VIEW: u32 = 1 << 1;
    const NAV_ACCEPT: u32 = 1 << 2;
    const NAV_CANCEL: u32 = 1 << 3;
    const NAV_UP: u32 = 1 << 4;
    const NAV_DOWN: u32 = 1 << 5;
    const NAV_LEFT: u32 = 1 << 6;
    const NAV_RIGHT: u32 = 1 << 7;
    const NAV_CONTEXT1: u32 = 1 << 8;
    const NAV_CONTEXT2: u32 = 1 << 9;
    const NAV_PAGE_LEFT: u32 = 1 << 14;
    const NAV_PAGE_RIGHT: u32 = 1 << 15;
    let mut nav = 0;
    for (held, bit) in [
        (MENU, NAV_MENU),
        (VIEW, NAV_VIEW),
        (A, NAV_ACCEPT),
        (B, NAV_CANCEL),
        (DPAD_UP, NAV_UP),
        (DPAD_DOWN, NAV_DOWN),
        (DPAD_LEFT, NAV_LEFT),
        (DPAD_RIGHT, NAV_RIGHT),
        (X, NAV_CONTEXT1),
        (Y, NAV_CONTEXT2),
        (LEFT_SHOULDER, NAV_PAGE_LEFT),
        (RIGHT_SHOULDER, NAV_PAGE_RIGHT),
    ] {
        if buttons & held != 0 {
            nav |= bit;
        }
    }
    nav
}

unsafe extern "system" fn reading_ui_navigation_state(this: *mut Reading, out: *mut u32) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: a live reading and a writable out.
    unsafe { *out = ui_navigation((*this).state.buttons) };
    true
}

// ---- exports ----

/// GameInputCreate(&gameInput): the one input object of the process.
///
/// # Safety
/// `out` is null or writable.
pub unsafe extern "system" fn game_input_create(out: *mut *mut c_void) -> Hresult {
    if out.is_null() {
        return E_POINTER;
    }
    static FIRST: AtomicBool = AtomicBool::new(true);
    let script = std::env::var_os("DECANT_PAD_SCRIPT").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    runtime::start(script, std::env::var("DECANT_KEYS").ok(), dispatch_reading_callbacks);
    if FIRST.swap(false, Ordering::Relaxed) {
        trace("gameinput: GameInputCreate, one gamepad");
    }
    // SAFETY: writable as checked.
    unsafe { *out = input_object().cast() };
    S_OK
}

/// DllGetClassObject: no class objects.
///
/// # Safety
/// `out` is null or writable.
pub unsafe extern "system" fn dll_get_class_object(_clsid: *const Guid, _iid: *const Guid, out: *mut *mut c_void) -> Hresult {
    if !out.is_null() {
        // SAFETY: writable as checked.
        unsafe { *out = std::ptr::null_mut() };
    }
    CLASS_E_CLASSNOTAVAILABLE
}

/// DllCanUnloadNow: never, the objects and the polling thread live for the process.
pub extern "system" fn dll_can_unload_now() -> Hresult {
    S_FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_device_info_has_the_header_layout() {
        assert_eq!(std::mem::offset_of!(DeviceInfo, device_id), 32);
        assert_eq!(std::mem::offset_of!(DeviceInfo, root_id), 64);
        assert_eq!(std::mem::offset_of!(DeviceInfo, supported_input), 104);
        assert_eq!(std::mem::size_of::<PadState>(), 28);
    }

    #[test]
    fn navigation_buttons_follow_the_pad_buttons() {
        use crate::pad::*;
        assert_eq!(ui_navigation(A | B | DPAD_LEFT), 0x4 | 0x8 | 0x40);
        assert_eq!(ui_navigation(LEFT_STICK), 0);
    }

    #[test]
    fn the_device_reports_an_xbox_pad_with_four_motors() {
        let info = device_info();
        assert_eq!(info.size as usize, std::mem::size_of::<DeviceInfo>());
        assert_eq!((info.usage_page, info.usage_id), (1, 5));
        assert_eq!(info.rumble_motors, 0xF);
        assert_eq!(info.device_id, info.root_id);
        assert_eq!(info.supported_input, KINDS);
        assert!(info.rest.iter().all(|&b| b == 0));
    }

    #[test]
    fn every_navigation_button_has_its_own_bit() {
        use crate::pad::*;
        let all = [MENU, VIEW, A, B, DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, X, Y, LEFT_SHOULDER, RIGHT_SHOULDER];
        let mut seen = 0u32;
        for button in all {
            let bit = ui_navigation(button);
            assert_eq!(bit.count_ones(), 1);
            assert_eq!(seen & bit, 0);
            seen |= bit;
        }
        assert_eq!(ui_navigation(all.iter().fold(0, |a, b| a | b) | RIGHT_STICK), seen);
        assert_eq!(ui_navigation(0), 0);
    }

    #[test]
    fn guids_print_in_the_usual_form() {
        assert_eq!(guid_text(&IID_IGAMEINPUT), "11be2a7e-4254-445a-9c09-ffc40f006918");
        assert_eq!(guid_text(&IID_IUNKNOWN), "00000000-0000-0000-c000-000000000046");
    }
}
