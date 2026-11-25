//! The 10 exports of `AcpHal.dll`, by name, with the prototypes of `docs/CONSOLE_LIBS.md` 2.1. Windows x64 convention.
#![allow(unsafe_code)]

use std::ffi::c_void;

use decant_log::trace;

use crate::abi::*;
use crate::hal::{self, IAcpHal};
use crate::heap::{Heaps, global, with_heaps};
use crate::shape;

pub const EXPORT_NAMES: [&str; 10] = [
    "ApuCreateHeap",
    "ApuAlloc",
    "ApuFree",
    "ApuMapVirtualAddress",
    "ApuMapApuAddress",
    "ApuIsVirtualAddressValid",
    "ApuHeapGetState",
    "AcpHalAllocateShapeContexts",
    "AcpHalReleaseShapeContexts",
    "AcpHalCreate",
];

fn failed(function: &str, code: Hresult) -> Hresult {
    trace(&format!("AcpHal: {function} failed with {code:#010x}"));
    code
}

#[unsafe(no_mangle)]
pub extern "system" fn ApuCreateHeap(cached_size: u32, non_cached_size: u32) -> Hresult {
    if cached_size == 0 {
        return failed("ApuCreateHeap", E_INVALIDARG);
    }
    let mut guard = global();
    if guard.is_some() {
        return failed("ApuCreateHeap", APU_E_HEAP_ALREADY_CREATED);
    }
    match Heaps::new(cached_size, non_cached_size) {
        Ok(heaps) => {
            *guard = Some(heaps);
            S_OK
        }
        Err(e) => failed("ApuCreateHeap", e),
    }
}

/// # Safety
/// `virtual_address` and `apu_address` are null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn ApuAlloc(virtual_address: *mut *mut c_void, apu_address: *mut u32, size: u32, alignment: u32, heap: u32) -> Hresult {
    // SAFETY: writable or null, as documented.
    unsafe {
        if !virtual_address.is_null() {
            *virtual_address = std::ptr::null_mut();
        }
        if !apu_address.is_null() {
            *apu_address = 0;
        }
    }
    if virtual_address.is_null() {
        return failed("ApuAlloc", E_INVALIDARG);
    }
    match with_heaps(|heaps| heaps.alloc(size, alignment, heap)) {
        Ok(Ok(a)) => {
            // SAFETY: `virtual_address` is non null and writable, `apu_address` is checked.
            unsafe {
                *virtual_address = a.host as *mut c_void;
                if !apu_address.is_null() {
                    *apu_address = a.apu;
                }
            }
            S_OK
        }
        Ok(Err(e)) | Err(e) => failed("ApuAlloc", e),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn ApuFree(virtual_address: *mut c_void) -> Hresult {
    if virtual_address.is_null() {
        return failed("ApuFree", E_INVALIDARG);
    }
    match global().as_mut() {
        Some(heaps) => heaps.free(virtual_address as *const u8).map_or_else(|e| failed("ApuFree", e), |()| S_OK),
        None => failed("ApuFree", E_INVALIDARG),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn ApuMapVirtualAddress(virtual_address: *const c_void) -> u32 {
    global().as_ref().and_then(|h| h.map_virtual(virtual_address as *const u8)).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "system" fn ApuMapApuAddress(apu_address: u32) -> *mut c_void {
    global().as_ref().and_then(|h| h.map_apu(apu_address)).map_or(std::ptr::null_mut(), |p| p as *mut c_void)
}

#[unsafe(no_mangle)]
pub extern "system" fn ApuIsVirtualAddressValid(virtual_address: *const c_void, alignment: u32) -> i32 {
    i32::from(global().as_ref().is_some_and(|h| h.is_valid(virtual_address as *const u8, alignment)))
}

/// # Safety
/// `state` is null or points at five writable UINT32.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn ApuHeapGetState(state: *mut ApuHeapState, heap: u32) -> Hresult {
    if state.is_null() || (heap != 1 && heap != 2) {
        return failed("ApuHeapGetState", E_INVALIDARG);
    }
    match global().as_ref() {
        None => failed("ApuHeapGetState", APU_E_HEAP_NOT_CREATED),
        Some(heaps) => match heaps.state(heap) {
            Ok(s) => {
                // SAFETY: five writable UINT32 as documented.
                unsafe { state.write_unaligned(s) };
                S_OK
            }
            Err(e) => failed("ApuHeapGetState", e),
        },
    }
}

/// # Safety
/// `contexts` is null or points at a writable `SHAPE_CONTEXTS` of 0x60 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn AcpHalAllocateShapeContexts(contexts: *mut ShapeContexts) -> Hresult {
    if contexts.is_null() {
        return failed("AcpHalAllocateShapeContexts", E_INVALIDARG);
    }
    // SAFETY: a 0x60 byte structure as documented.
    let mut c = unsafe { contexts.read_unaligned() };
    let code = shape::allocate(&mut c);
    if code == S_OK {
        // SAFETY: as above.
        unsafe { contexts.write_unaligned(c) };
        trace(&format!("AcpHal: SHAPE contexts allocated, counts {:?}", c.counts));
        S_OK
    } else {
        failed("AcpHalAllocateShapeContexts", code)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn AcpHalReleaseShapeContexts() -> Hresult {
    let code = shape::release();
    if code == S_OK { S_OK } else { failed("AcpHalReleaseShapeContexts", code) }
}

/// # Safety
/// `out` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn AcpHalCreate(out: *mut *mut IAcpHal) -> Hresult {
    if out.is_null() {
        return failed("AcpHalCreate", E_INVALIDARG);
    }
    // SAFETY: writable as documented.
    unsafe { *out = hal::create() };
    S_OK
}
