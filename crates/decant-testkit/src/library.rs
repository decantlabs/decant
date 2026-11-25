//! A DLL loaded with `LoadLibraryW`, and its exports.
#![allow(unsafe_code)]

use std::ffi::{OsStr, c_void};
use std::os::windows::ffi::OsStrExt;

use windows_sys::Win32::Foundation::{FreeLibrary, GetLastError, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW, LoadLibraryW};

pub struct Library(HMODULE);

impl Library {
    /// Loads `path`, or returns the Windows error code.
    pub fn open(path: &OsStr) -> Result<Library, u32> {
        let wide: Vec<u16> = path.encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a zero terminated path that lives for the call.
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        if module.is_null() {
            // SAFETY: no precondition.
            Err(unsafe { GetLastError() })
        } else {
            Ok(Library(module))
        }
    }

    /// Loads an absolute `path` with `LOAD_WITH_ALTERED_SEARCH_PATH`, so its dependencies are
    /// searched from its own folder first.
    pub fn open_altered(path: &OsStr) -> Result<Library, u32> {
        let wide: Vec<u16> = path.encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a zero terminated path that lives for the call.
        let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        if module.is_null() {
            // SAFETY: no precondition.
            Err(unsafe { GetLastError() })
        } else {
            Ok(Library(module))
        }
    }

    /// The address of an export, or null.
    pub fn address(&self, name: &str) -> *const c_void {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        // SAFETY: the module is loaded and `bytes` is a zero terminated name.
        unsafe { GetProcAddress(self.0, bytes.as_ptr()) }.map_or(std::ptr::null(), |f| f as *const c_void)
    }

    /// An export as a function of type `F`.
    ///
    /// # Safety
    /// `F` must be a function pointer type matching the export's real signature.
    pub unsafe fn get<F: Copy>(&self, name: &str) -> Option<F> {
        const { assert!(size_of::<F>() == size_of::<*const c_void>()) };
        let address = self.address(name);
        // SAFETY: the caller states that `F` is the export's function pointer type.
        (!address.is_null()).then(|| unsafe { std::mem::transmute_copy::<*const c_void, F>(&address) })
    }

    /// The module handle.
    pub fn handle(&self) -> HMODULE {
        self.0
    }

    /// Unloads the DLL, as the C tests did with `FreeLibrary` before they returned.
    pub fn free(self) {
        // SAFETY: the module was loaded by `open` and is freed once.
        unsafe { FreeLibrary(self.0) };
    }
}
