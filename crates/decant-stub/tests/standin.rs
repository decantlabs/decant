//! Generates a stand in from an invented report, builds it with cargo, loads the DLL and calls it.
//! Skipped when cargo is not on the PATH. Windows only, like the DLLs.

#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::{CString, c_void};
use std::os::windows::ffi::OsStrExt;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetLastError() -> u32;
}

const REPORT: &str = r##"{
  "dll": {
    "zork_xs.dll": {"nom": "Zork_xs.dll", "origine": "absente", "fonctions": {"ZorkAlloc": false, "?Make@Zork@@QEAAXXZ": false, "#7": false, "#12": false}},
    "kernel32.dll": {"nom": "KERNEL32.dll", "origine": "PC", "fonctions": {"Sleep": true}}
  }
}"##;

fn cargo_available() -> bool {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    std::process::Command::new(cargo).arg("--version").output().is_ok_and(|o| o.status.success())
}

#[test]
fn generated_stand_in_builds_exports_by_name_and_ordinal_and_returns_e_notimpl() {
    if !cargo_available() {
        eprintln!("cargo is not on the PATH: skipped");
        return;
    }
    let mods = decant_stub::modules(REPORT, &[]).unwrap();
    assert_eq!(mods.len(), 1);
    let out = tempfile::tempdir().unwrap();
    decant_stub::write(&mods, out.path()).unwrap();
    let dll = decant_stub::build(&mods[0], out.path()).unwrap();
    assert_eq!(dll.file_name().unwrap(), "Zork_xs.dll");

    let wide: Vec<u16> = dll.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: loads the DLL built above, which only logs and returns a constant.
    unsafe {
        let module = LoadLibraryW(wide.as_ptr());
        assert!(!module.is_null(), "LoadLibrary failed: {}", GetLastError());
        for name in ["ZorkAlloc", "?Make@Zork@@QEAAXXZ"] {
            let c = CString::new(name).unwrap();
            assert!(!GetProcAddress(module, c.as_ptr().cast()).is_null(), "no export {name}");
        }
        for ordinal in [7usize, 12] {
            assert!(!GetProcAddress(module, ordinal as *const u8).is_null(), "no ordinal {ordinal}");
        }
        let c = CString::new("ZorkAlloc").unwrap();
        let f: extern "system" fn() -> i32 = std::mem::transmute(GetProcAddress(module, c.as_ptr().cast()));
        assert_eq!(f() as u32, 0x8000_4001);
        let by_ordinal: extern "system" fn() -> i32 = std::mem::transmute(GetProcAddress(module, 7usize as *const u8));
        assert_eq!(by_ordinal() as u32, 0x8000_4001);
        FreeLibrary(module);
    }
}
