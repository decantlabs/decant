//! The local XDisplay HDR table, `api_id = {03F0FE74-FDD9-4E5C-B630-F9339C47ACC5}`, interface
//! `{35F07670-706E-4BFB-9476-090798C5EBF3}`, which `QueryApiImpl` serves only when there is no PC
//! runtime. The title executable asks for it during its display set up and breaks into the
//! debugger shortly after the refusal (Windows Sandbox run 9).
//!
//! It answers honestly that HDR is not available: `XDisplayTryEnableHdrMode` returns
//! `XDisplayHdrModeResult::Disabled` and, when the title passes an `XDisplayHdrModeInfo`, fills it
//! with zeros (no tone map luminance). Nothing is switched on the display.
//!
//! Slot order of the public PC GDK `xgameruntime.lib` thunks: `XDisplayTryEnableHdrMode` is slot 4
//! of this interface. Slot 3 is not identified (this crate's tables list slots 3 and 4): it logs
//! and answers `E_NOTIMPL`, as do the padding slots 5 to 7. Slots 0 to 2 are IUnknown on a static
//! object. Each slot takes `this` first and drops it; every call is logged (`calllog`).
//!
//! Temporary: this only keeps the title running without HDR. To do: real HDR support (relay the
//! PC display's HDR state and luminance once the presentation path can output HDR).
//!
//! Our own code, from the public GDK signature
//! `XDisplayHdrModeResult XDisplayTryEnableHdrMode(XDisplayHdrModePreference, XDisplayHdrModeInfo*)`
//! (`Unknown = 0, Enabled = 1, Disabled = 2`).
#![allow(unsafe_code, non_snake_case)]

use std::ffi::c_void;

use crate::E_NOTIMPL;
use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::tables::Guid;
use crate::taskqueue::S_OK;

/// The table's `api_id`.
pub(crate) const XDISPLAY_HDR_TABLE: Guid = Guid { data1: 0x03F0FE74, data2: 0xFDD9, data3: 0x4E5C, data4: [0xB6, 0x30, 0xF9, 0x33, 0x9C, 0x47, 0xAC, 0xC5] };
/// The interface the title asks for.
pub(crate) const XDISPLAY_HDR_IID: Guid = Guid { data1: 0x35F07670, data2: 0x706E, data3: 0x4BFB, data4: [0x94, 0x76, 0x09, 0x07, 0x98, 0xC5, 0xEB, 0xF3] };
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;

/// `XDisplayHdrModeResult::Disabled`.
pub(crate) const HDR_DISABLED: u32 = 2;

/// The object for a couple served without a PC runtime, with the table's name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&XDISPLAY_HDR_TABLE) && riid == Some(&XDISPLAY_HDR_IID)).then(|| (object(), "XDisplay"))
}

const SLOTS: usize = 8;
const MAIN: &str = "XDisplay";
static COUNTS: Counters<SLOTS> = Counters::new();

type This = *const Object;

/// `XDisplayHdrModeInfo`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct XDisplayHdrModeInfo {
    pub(crate) min_tone_map_luminance: f32,
    pub(crate) max_tone_map_luminance: f32,
    pub(crate) max_full_frame_tone_map_luminance: f32,
}

/// The vtable, in slot order.
#[repr(C)]
pub(crate) struct Vtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    slot3: unsafe extern "system" fn(This) -> i32,
    try_enable_hdr_mode: unsafe extern "system" fn(This, u32, *mut XDisplayHdrModeInfo) -> u32,
    slot5: unsafe extern "system" fn(This) -> i32,
    slot6: unsafe extern "system" fn(This) -> i32,
    slot7: unsafe extern "system" fn(This) -> i32,
}

/// The COM style object: a pointer to its vtable.
#[repr(C)]
pub(crate) struct Object {
    vtbl: *const c_void,
}

// SAFETY: the object only holds the address of a static vtable.
unsafe impl Sync for Object {}

static OBJECT: Object = Object { vtbl: (&raw const VTBL).cast() };

/// The object's address, as written to `*out`.
pub(crate) fn object() -> *mut c_void {
    (&raw const OBJECT).cast_mut().cast()
}

logged_shim!(MAIN, COUNTS, This, 4, try_enable_hdr_mode => XDisplayTryEnableHdrMode, (preference: u32, info: *mut XDisplayHdrModeInfo) -> u32);

/// Declares unknown slots of the table: each logs its index (bounded) and answers `E_NOTIMPL`.
macro_rules! unknown {
    ($($slot:literal $name:ident),*) => {
        $(
            unsafe extern "system" fn $name(_this: This) -> i32 {
                decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("local {MAIN} table slot {}", $slot));
                if let Some(n) = COUNTS.next($slot) {
                    log(&format!("local {MAIN} table: slot {} is not known -> E_NOTIMPL [call {n}]", $slot));
                }
                E_NOTIMPL
            }
        )*
    };
}

unknown!(3 slot3, 5 slot5, 6 slot6, 7 slot7);

static VTBL: Vtbl = Vtbl { query_interface, add_ref: no_count, release: no_count, slot3, try_enable_hdr_mode, slot5, slot6, slot7 };

/// The interface id answers the object.
unsafe extern "system" fn query_interface(_this: This, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract: `iid` is null or a GUID, `out` a writable pointer location.
    unsafe {
        let iid = if iid.is_null() { None } else { Some(iid.read_unaligned()) };
        if iid == Some(XDISPLAY_HDR_IID) {
            *out = object();
            S_OK
        } else {
            *out = std::ptr::null_mut();
            E_NOINTERFACE
        }
    }
}

/// AddRef and Release of a static object.
unsafe extern "system" fn no_count(_this: This) -> u32 {
    1
}

/// HDR is not available here: `Disabled`, and a zeroed mode info when one is passed.
unsafe extern "system" fn XDisplayTryEnableHdrMode(preference: u32, info: *mut XDisplayHdrModeInfo) -> u32 {
    if !info.is_null() {
        // SAFETY: the title's location, not null.
        unsafe { info.write_unaligned(XDisplayHdrModeInfo { min_tone_map_luminance: 0.0, max_tone_map_luminance: 0.0, max_full_frame_tone_map_luminance: 0.0 }) };
    }
    log(&format!("XDisplayTryEnableHdrMode(preference {preference}) -> Disabled (no PC runtime: HDR not available, nothing switched)"));
    HDR_DISABLED
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots() -> &'static [usize; SLOTS] {
        // SAFETY: `Vtbl` is `repr(C)` with SLOTS function pointer fields, the size checked below.
        unsafe { &*((&raw const VTBL).cast::<[usize; SLOTS]>()) }
    }

    #[test]
    fn the_vtable_has_the_gdk_layout() {
        assert_eq!(size_of::<XDisplayHdrModeInfo>(), 12);
        assert_eq!(size_of::<Vtbl>(), SLOTS * size_of::<usize>());
        // SAFETY: the object's first field is its vtable pointer.
        unsafe { assert!(std::ptr::eq(*(object() as *const *const Vtbl), &raw const VTBL)) };
        assert_eq!(slots()[3], slot3 as *const () as usize);
        assert_eq!(slots()[4], try_enable_hdr_mode as *const () as usize);
    }

    #[test]
    fn only_the_hdr_couple_is_served() {
        let other = Guid { data1: 1, data2: 2, data3: 3, data4: [0; 8] };
        assert_eq!(object_for(Some(&XDISPLAY_HDR_TABLE), Some(&XDISPLAY_HDR_IID)), Some((object(), "XDisplay")));
        assert_eq!(object_for(Some(&XDISPLAY_HDR_TABLE), Some(&other)), None);
        assert_eq!(object_for(Some(&XDISPLAY_HDR_TABLE), None), None);
        assert_eq!(object_for(Some(&other), Some(&XDISPLAY_HDR_IID)), None);
    }

    #[test]
    fn hdr_is_reported_disabled_with_a_zeroed_info() {
        let mut info = XDisplayHdrModeInfo { min_tone_map_luminance: 1.0, max_tone_map_luminance: 2.0, max_full_frame_tone_map_luminance: 3.0 };
        let vtbl = &VTBL;
        // SAFETY: the slot takes `this`, a preference and a writable info.
        let r = unsafe { (vtbl.try_enable_hdr_mode)(object().cast(), 1, &mut info) };
        assert_eq!(r, HDR_DISABLED);
        assert_eq!(info, XDisplayHdrModeInfo { min_tone_map_luminance: 0.0, max_tone_map_luminance: 0.0, max_full_frame_tone_map_luminance: 0.0 });
        // SAFETY: a null info is accepted.
        assert_eq!(unsafe { (vtbl.try_enable_hdr_mode)(object().cast(), 0, std::ptr::null_mut()) }, HDR_DISABLED);
        // SAFETY: the slot takes `this` only.
        assert_eq!(unsafe { (vtbl.slot3)(object().cast()) }, E_NOTIMPL);
    }

    #[test]
    fn query_interface_answers_the_hdr_iid_only() {
        let mut out = std::ptr::null_mut();
        // SAFETY: valid GUID and out location.
        assert_eq!(unsafe { query_interface(object().cast(), &XDISPLAY_HDR_IID, &mut out) }, S_OK);
        assert_eq!(out, object());
        // SAFETY: valid GUID and out location.
        assert_eq!(unsafe { query_interface(object().cast(), &XDISPLAY_HDR_TABLE, &mut out) }, E_NOINTERFACE);
        assert!(out.is_null());
    }
}
