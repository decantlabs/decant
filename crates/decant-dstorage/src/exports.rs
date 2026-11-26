//! The one export of `dstorage_xs.dll` and `xcrdapi.dll` (the former `extern "C" DStorageGetFactory`
//! of `storage.cpp`): one implementation, two names, since a title may import `DStorageGetFactory`
//! from either.
#![allow(unsafe_code)]
#![allow(non_snake_case)]

use std::ffi::c_void;

use crate::console::console_get_factory;
use crate::engine::{DecantDstorageMetricsSnapshot, Factory, engine_factory, metrics_snapshot};
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

/// Fills `*out` with a lock-free snapshot of cumulative DirectStorage I/O metrics.
/// Returns `1` on success, `0` when `out` is null.
///
/// # Safety
/// `out` is null or writable for `DecantDstorageMetricsSnapshot`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DecantDstorageMetrics(out: *mut DecantDstorageMetricsSnapshot) -> u32 {
    if out.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees `out` is writable.
    unsafe { *out = metrics_snapshot() };
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decant_dstorage_metrics_handles_null_and_writes_snapshot() {
        // SAFETY: null is explicitly handled and returns 0.
        assert_eq!(unsafe { DecantDstorageMetrics(std::ptr::null_mut()) }, 0);
        let mut snap = DecantDstorageMetricsSnapshot::default();
        // SAFETY: `snap` is a valid writable `DecantDstorageMetricsSnapshot`.
        assert_eq!(unsafe { DecantDstorageMetrics(&mut snap) }, 1);
    }
}
