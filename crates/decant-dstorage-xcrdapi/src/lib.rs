//! `xcrdapi.dll`: the same implementation as `dstorage_xs.dll` (`decant-dstorage`) under the
//! second name a title may import `DStorageGetFactory` from. One implementation, two names, as the
//! two module definition files of the former `build_storage.cmd` made it.
#![allow(unsafe_code)]

/// Referenced from here so the linker keeps the export in this DLL. `build.rs` names it with its
/// ordinal.
#[cfg(windows)]
#[used]
static EXPORT: unsafe extern "system" fn(*const dstorage_xs::decant::GUID, *mut *mut std::ffi::c_void) -> i32 = dstorage_xs::DStorageGetFactory;
