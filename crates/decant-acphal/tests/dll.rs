//! Loads the `AcpHal.dll` Cargo built next to the tests (or the one in `DECANT_DLL_DIR`), checks that it exports all 10 names and
//! calls some of them through the export table.
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;

use decant_testkit::{Library, dll_for};

#[test]
fn the_dll_exports_every_name_of_the_specification() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "AcpHal.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("AcpHal.dll loads");
    for name in AcpHal::EXPORT_NAMES {
        assert!(!lib.address(name).is_null(), "{name} is not exported");
    }
    assert_eq!(AcpHal::EXPORT_NAMES.len(), 10);

    type Create = unsafe extern "system" fn(u32, u32) -> i32;
    type Alloc = unsafe extern "system" fn(*mut *mut c_void, *mut u32, u32, u32, u32) -> i32;
    type Map = unsafe extern "system" fn(*const c_void) -> u32;
    // SAFETY: the exports have these prototypes.
    unsafe {
        let create = lib.get::<Create>("ApuCreateHeap").unwrap();
        let alloc = lib.get::<Alloc>("ApuAlloc").unwrap();
        let map = lib.get::<Map>("ApuMapVirtualAddress").unwrap();
        // The DLL has its own heaps, apart from the ones of the test program.
        assert_eq!(create(0x10_0000, 0), 0);
        assert_eq!(create(0x10_0000, 0), 0x8ACA_0001_u32 as i32);
        let (mut va, mut apu) = (std::ptr::null_mut(), 0u32);
        assert_eq!(alloc(&mut va, &mut apu, 0x100, 256, 1), 0);
        assert_eq!(apu % 256, 0);
        assert_eq!(map(va), apu);
    }
}
