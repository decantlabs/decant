//! Loads the `xapu.dll` Cargo built next to the tests (or the one in `DECANT_DLL_DIR`), checks that it exports all 6 names and
//! drives a connection through the export table.
#![cfg(windows)]
#![allow(unsafe_code)]

use decant_testkit::{Library, dll_for};
use xapu::abi::*;

#[test]
fn the_dll_exports_every_name_of_the_specification() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "xapu.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("xapu.dll loads");
    for name in xapu::EXPORT_NAMES {
        assert!(!lib.address(name).is_null(), "{name} is not exported");
    }
    assert_eq!(xapu::EXPORT_NAMES.len(), 6);

    type Connect = unsafe extern "system" fn(*const ConnectParams, *mut ConnectionRegion, *mut Handle) -> i32;
    type Enqueue = unsafe extern "system" fn(Handle, *const Command, *mut u32) -> i32;
    type Dequeue = unsafe extern "system" fn(Handle, *mut XapuResult) -> i32;
    type Disconnect = unsafe extern "system" fn(Handle) -> i32;
    type Util = unsafe extern "system" fn(u32, u32, *mut u32) -> i32;
    // SAFETY: the exports have these prototypes.
    unsafe {
        let connect = lib.get::<Connect>("XApuConnect").unwrap();
        let enqueue = lib.get::<Enqueue>("XApuEnqueueCommand").unwrap();
        let dequeue = lib.get::<Dequeue>("XApuDequeueResult").unwrap();
        let disconnect = lib.get::<Disconnect>("XApuDisconnect").unwrap();
        let util = lib.get::<Util>("XApuGetUtilization").unwrap();
        let p = ConnectParams { process_type: PROCESS_DECODE_OPUS, stream_count: 4, queue_depth: 2, region_size: 0, frame_count: 0, options: 0 };
        let mut region = ConnectionRegion { memory: std::ptr::null_mut(), result_event: 0 };
        let mut handle: Handle = 0;
        assert_eq!(connect(&p, &mut region, &mut handle), 0);
        assert_ne!(handle, 0);
        let mut free = [0u32; 2];
        assert_eq!(util(1, 1, free.as_mut_ptr()), 0);
        assert_eq!(free, [496, 500], "the DLL has its own stream budget");
        let mut seq = 0;
        assert_eq!(enqueue(handle, &Command::new(COMMAND_START, 0, 0).with_start(2, 1.0, 0), &mut seq), 0);
        assert_eq!(seq, 1);
        let mut r = XapuResult::default();
        assert_eq!(dequeue(handle, &mut r), 0);
        assert_eq!((r.kind(), r.code()), (COMMAND_START, 0));
        assert_eq!(dequeue(handle, &mut r), XAPU_E_NO_RESULT);
        assert_eq!(disconnect(handle), 0);
        assert_eq!(disconnect(handle), E_INVALIDARG);
    }
}
