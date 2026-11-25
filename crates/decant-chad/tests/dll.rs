//! Loads the `chad.dll` Cargo built next to the tests (or the one in `DECANT_DLL_DIR`), checks that it exports all 7 names and
//! runs an FFT through the export table.
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;

use chad::abi::*;
use decant_testkit::{Library, dll_for};

#[test]
fn the_dll_exports_every_name_of_the_specification() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "chad.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("chad.dll loads");
    for name in chad::EXPORT_NAMES {
        assert!(!lib.address(name).is_null(), "{name} is not exported");
    }
    assert_eq!(chad::EXPORT_NAMES.len(), 7);

    type Connect = unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut XdspClient) -> i32;
    type Connect2 = unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32, *mut *mut XdspClient) -> i32;
    type Disconnect = unsafe extern "system" fn(*mut XdspClient) -> i32;
    type Activate = unsafe extern "system" fn(*mut XdspClient, *const RawActivationParameters, *mut *mut u32, *mut *mut XdspContext) -> i32;
    type Submit = unsafe extern "system" fn(*mut XdspContext, *const XdspCommand, *mut u32) -> i32;
    type SubmitEnv = unsafe extern "system" fn(*mut XdspContext, *const XdspCommand, *const f32, *mut u32) -> i32;
    type Deactivate = unsafe extern "system" fn(*mut XdspContext) -> i32;
    let mut memory = vec![0f32; 0x8000];
    // SAFETY: the exports have these prototypes and the arguments are valid for them.
    unsafe {
        let connect = lib.get::<Connect>("XDspConnect").unwrap();
        let connect2 = lib.get::<Connect2>("XDspConnectWithMaximumStreamLimit").unwrap();
        let disconnect = lib.get::<Disconnect>("XDspDisconnect").unwrap();
        let activate = lib.get::<Activate>("XDspActivate").unwrap();
        let submit = lib.get::<Submit>("XDspSubmitCommand").unwrap();
        let submit_env = lib.get::<SubmitEnv>("XDspSubmitCommandWithEnvelope").unwrap();
        let deactivate = lib.get::<Deactivate>("XDspDeactivate").unwrap();

        let mut client = std::ptr::null_mut();
        assert_eq!(connect2(memory.as_mut_ptr() as *mut _, 0xFFFF, 1, 1, 0, &mut client), E_INVALIDARG);
        assert_eq!(connect(memory.as_mut_ptr() as *mut _, 0x20000, 0x81, &mut client), E_INVALIDARG);
        assert_eq!(connect(memory.as_mut_ptr() as *mut _, 0x20000, 2, &mut client), 0);
        let p = ActivationParameters { process_type: PROCESS_FFT, flags: 0, block: 0x200, channels: 1, ir_length: 0, ir_pointer: 0 };
        let (mut status, mut context) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(activate(client, &p.to_bytes(), &mut status, &mut context), 0);
        // A DC input of 1.0 gives a DC bin of 512 and nothing else.
        memory[..0x200].fill(1.0);
        let cmd = XdspCommand { input_gain: 1.0, output_gain: 1.0, input: memory.as_ptr() as u64, output: memory.as_ptr() as u64 + 0x1000, multiplier: 0 };
        let mut seq = 0;
        assert_eq!(submit(context, &cmd, &mut seq), 0);
        assert_eq!(seq, 1);
        assert_eq!(submit_env(context, &cmd, std::ptr::null(), &mut seq), 0);
        assert_eq!(seq, 2);
        assert_eq!(memory[0x400], 512.0);
        assert!(memory[0x401..0x600].iter().all(|v| v.abs() < 1e-3));
        assert_eq!(std::slice::from_raw_parts(status, 4)[..2], [2, 2]);
        assert_eq!(disconnect(client), XDSP_E_CONTEXTS_ACTIVE);
        assert_eq!(deactivate(context), 0);
        assert_eq!(disconnect(client), 0);
    }
}
