//! Loads the `xg_xs.dll` Cargo built next to the tests (or the one in `DECANT_DLL_DIR`), checks that it exports all 24 names and
//! calls two of them through the export table.
#![cfg(windows)]
#![allow(unsafe_code)]

use decant_testkit::{Library, dll_for};

#[test]
fn the_dll_exports_every_name_of_the_specification() {
    let exe = std::env::current_exe().unwrap();
    let dll = dll_for(exe.to_str().unwrap(), "xg_xs.dll");
    assert!(dll.is_file(), "{} was not built", dll.display());
    let lib = Library::open(dll.as_os_str()).expect("xg_xs.dll loads");
    for name in xg_xs::EXPORT_NAMES {
        assert!(!lib.address(name).is_null(), "{name} is not exported");
    }
    assert_eq!(xg_xs::EXPORT_NAMES.len(), 24);
    type Swizzle = unsafe extern "system" fn(u32, u32, u64, u32, u32, u32, u32, u32) -> u32;
    // SAFETY: the export has this prototype.
    let swizzle = unsafe { lib.get::<Swizzle>("XGComputeOptimalSwizzleMode") }.unwrap();
    // SAFETY: a plain computation.
    assert_eq!(unsafe { swizzle(3, 28, 1920, 1080, 1, 1, 0x28, 0) }, 27);
    type Init = unsafe extern "system" fn() -> i32;
    // SAFETY: the export has this prototype.
    let init = unsafe { lib.get::<Init>("XGInitializeLibrary") }.unwrap();
    // The DLL has its own state, apart from the one of the test program.
    // SAFETY: no arguments.
    assert_eq!(unsafe { init() }, 0);
    // SAFETY: no arguments.
    assert_eq!(unsafe { init() }, 1);
}
