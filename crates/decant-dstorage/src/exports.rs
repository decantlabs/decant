//! The one export of `dstorage_xs.dll` and `xcrdapi.dll` (the former `extern "C" DStorageGetFactory`
//! of `storage.cpp`): one implementation, two names, since a title may import `DStorageGetFactory`
//! from either.
#![allow(unsafe_code)]
#![allow(non_snake_case)]

use std::ffi::c_void;

use crate::console::console_get_factory;
use crate::engine::{Factory, engine_factory};
use crate::decant::*;

/// `HRESULT DStorageGetFactory(REFIID iid, void **result)`. The project's own factory IID (or
/// `IUnknown`) gives the engine factory. Any other IID goes through the spec driven console ABI:
/// `E_NOINTERFACE`, logged, without a spec.
///
/// # Safety
/// `iid` is readable and `result` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DStorageGetFactory(iid: *const GUID, result: *mut *mut c_void) -> HRESULT {
    if result.is_null() {
        return E_POINTER;
    }
    // SAFETY: `result` is writable and `iid` readable, as the caller guarantees.
    unsafe {
        *result = std::ptr::null_mut();
        let wanted = guid_value(&*iid);
        if wanted != guid_value(&IID_IUNKNOWN) && wanted != guid_value(&IID_DECANT_FACTORY) {
            return console_get_factory(iid, result);
        }
        let factory = match engine_factory() {
            Ok(factory) => factory,
            Err(hr) => return hr,
        };
        query_interface::<Factory>(factory.as_ptr().cast(), iid, result)
    }
}
