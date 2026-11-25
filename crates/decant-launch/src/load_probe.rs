//! Test helper for `tests/redirect.rs`, not part of the launcher: loads `xgameruntime.dll` by name
//! with `LOAD_LIBRARY_SEARCH_SYSTEM32`, as the GDK thunk code in a title does, and exits 0 when the
//! load worked, 4 when it did not.

#[cfg(windows)]
#[allow(unsafe_code)]
fn main() {
    use std::ffi::c_void;

    #[link(name = "kernel32", kind = "raw-dylib")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    }
    const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
    let name: Vec<u16> = "xgameruntime.dll\0".encode_utf16().collect();
    // SAFETY: a NUL-terminated name and no file handle, as the API asks.
    let module = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
    std::process::exit(if module.is_null() { 4 } else { 0 });
}

#[cfg(not(windows))]
fn main() {}
