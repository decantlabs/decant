//! Loads the `GameInput.dll` Cargo built next to the tests (or the one in `DECANT_DLL_DIR`), checks its exports and ordinal,
//! and reads the simulated pad through the interface: device callbacks, readings, the control file driving the state.
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::io::Write;
use std::time::{Duration, Instant};

use decant_testkit::{Library, dll_for};

mod common;
use common::{Create, GetReading, Release, KIND_GAMEPAD, method, pad_of, wait_for};

type Hresult = i32;
type Query = unsafe extern "system" fn(*mut c_void, *const [u8; 16], *mut *mut c_void) -> Hresult;
type RegisterDevice = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32, *mut c_void, *mut c_void, *mut u64) -> Hresult;
type Timestamp = unsafe extern "system" fn(*mut c_void) -> u64;
type DeviceStatus = unsafe extern "system" fn(*mut c_void) -> u32;

const KIND_KEYBOARD: u32 = 0x10;
const IID_IUNKNOWN: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0xC0, 0, 0, 0, 0, 0, 0, 0x46];
const IID_UNKNOWN_TO_US: [u8; 16] = [1; 16];
const GAMEINPUT_E_READING_NOT_FOUND: Hresult = 0x838A_0003_u32 as i32;
const E_NOINTERFACE: Hresult = 0x8000_4002_u32 as i32;

static DEVICE_CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

unsafe extern "system" fn on_device(_token: u64, _context: *mut c_void, device: *mut c_void, _stamp: u64, status: u32, _previous: u32) {
    assert!(!device.is_null());
    assert_eq!(status & 1, 1, "connected");
    DEVICE_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[test]
fn the_dll_exports_ordinal_one_and_plays_the_script() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "GameInput.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("GameInput.dll loads");
    for name in GameInput::EXPORT_NAMES {
        assert!(!lib.address(name).is_null(), "{name} is not exported");
    }
    let by_ordinal = {
        // GetProcAddress by ordinal 1: the title imports that way.
        #[link(name = "kernel32", kind = "raw-dylib")]
        unsafe extern "system" {
            fn GetProcAddress(module: *mut c_void, name: usize) -> *const c_void;
        }
        // SAFETY: a loaded module and an ordinal.
        unsafe { GetProcAddress(lib.handle(), 1) }
    };
    assert_eq!(by_ordinal, lib.address("GameInputCreate"), "GameInputCreate is ordinal 1");

    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("pad.txt");
    std::fs::write(&script, "").unwrap();
    // SAFETY: nothing else in this test process reads the environment concurrently.
    unsafe {
        std::env::set_var("DECANT_PAD_SCRIPT", &script);
        std::env::set_var("DECANT_NO_REAL_INPUT", "1");
    }

    // SAFETY: the exports and vtable slots have these prototypes (public GameInput.h order).
    unsafe {
        let create = lib.get::<Create>("GameInputCreate").unwrap();
        let mut input = std::ptr::null_mut();
        assert_eq!(create(&mut input), 0);
        assert!(!input.is_null());
        let mut again = std::ptr::null_mut();
        assert_eq!(create(&mut again), 0);
        assert_eq!(again, input, "one input object");
        assert_eq!(create(std::ptr::null_mut()) as u32, 0x8000_4003);

        // QueryInterface: IUnknown and IGameInput are served, others refused with a null out.
        let query = method::<Query>(input, 0);
        let iid_input = [0x7e, 0x2a, 0xbe, 0x11, 0x54, 0x42, 0x5a, 0x44, 0x9c, 0x09, 0xff, 0xc4, 0x0f, 0x00, 0x69, 0x18];
        for iid in [IID_IUNKNOWN, iid_input] {
            let mut out = std::ptr::null_mut();
            assert_eq!(query(input, &iid, &mut out), 0);
            assert_eq!(out, input);
        }
        let mut out = std::ptr::dangling_mut::<c_void>();
        assert_eq!(query(input, &IID_UNKNOWN_TO_US, &mut out), E_NOINTERFACE);
        assert!(out.is_null());

        // Device callback, blocking enumeration: called before the call returns.
        let mut token = 0u64;
        let before = DEVICE_CALLS.load(std::sync::atomic::Ordering::SeqCst);
        let hr = method::<RegisterDevice>(input, 9)(input, std::ptr::null_mut(), KIND_GAMEPAD, 0, 2, std::ptr::null_mut(), on_device as *const () as *mut c_void, &mut token);
        assert_eq!(hr, 0);
        assert_ne!(token, 0);
        assert_eq!(DEVICE_CALLS.load(std::sync::atomic::Ordering::SeqCst), before + 1);
        // Asynchronous enumeration: called from another thread soon after.
        let hr = method::<RegisterDevice>(input, 9)(input, std::ptr::null_mut(), KIND_GAMEPAD, 0, 1, std::ptr::null_mut(), on_device as *const () as *mut c_void, &mut token);
        assert_eq!(hr, 0);
        let end = Instant::now() + Duration::from_secs(5);
        while DEVICE_CALLS.load(std::sync::atomic::Ordering::SeqCst) < before + 2 {
            assert!(Instant::now() < end, "no asynchronous device callback");
            std::thread::sleep(Duration::from_millis(5));
        }

        // The device of a reading answers its status.
        let mut reading = std::ptr::null_mut();
        assert_eq!(method::<GetReading>(input, 4)(input, KIND_GAMEPAD, std::ptr::null_mut(), &mut reading), 0);
        let mut device = std::ptr::null_mut::<c_void>();
        method::<unsafe extern "system" fn(*mut c_void, *mut *mut c_void)>(reading, 6)(reading, &mut device);
        assert!(!device.is_null());
        assert_eq!(method::<DeviceStatus>(device, 4)(device) & 1, 1);
        method::<Release>(reading, 2)(reading);

        // A kind the pad is not has no reading.
        let mut none = std::ptr::dangling_mut::<c_void>();
        assert_eq!(method::<GetReading>(input, 4)(input, KIND_KEYBOARD, std::ptr::null_mut(), &mut none), GAMEINPUT_E_READING_NOT_FOUND);
        assert!(none.is_null());

        // Neutral to begin with, timestamps only go up.
        let (buttons, axes, first) = pad_of(input);
        assert_eq!((buttons, axes), (0, [0.0; 6]));
        let (_, _, second) = pad_of(input);
        assert!(second > first);
        let now = method::<Timestamp>(input, 3)(input);
        assert!(now > second);

        // The control file drives the state, also when written while the title runs.
        let mut file = std::fs::OpenOptions::new().append(true).open(&script).unwrap();
        file.write_all(b"A down\nstick left 0.5 1.0\ntrigger right 1\n").unwrap();
        wait_for(input, "A, left stick, right trigger", |b, a| b == 0x4 && a[0] == 0.0 && a[2] == 0.5 && a[3] == 1.0 && a[1] == 1.0);
        file.write_all(b"A up\nB down\nrelease\nY down\n").unwrap();
        wait_for(input, "Y only", |b, a| b == 0x20 && a == [0.0; 6]);
        file.write_all(b"wait 200\nY up\n").unwrap();
        wait_for(input, "neutral after the wait", |b, _| b == 0);
    }
}
