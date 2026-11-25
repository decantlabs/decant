//! Offline XSystem tables. `api_id = {E349BD1A-FC20-4E40-B99C-4178CC6B409F}` with interface
//! `{1861CF2E-E18B-4834-A9F5-B4A4E6EFB4CF}`, and the analytics table whose `api_id` is also its
//! interface, `{B884675D-B738-4A9C-815D-9A9A1E0C6C9B}`. Forwarded to the PC runtime they returned the
//! PC's console id and sandbox (`decant-specs/hle-gap.md` row 10). Here the title sees one offline
//! retail Xbox Series X instead:
//!
//! * slot 3 `XSystemGetConsoleId` returns a fixed id of 38 characters.
//! * slot 4 `XSystemGetXboxLiveSandboxId` returns `RETAIL`.
//! * slot 5 `XSystemGetAppSpecificDeviceId` returns base64 of 32 fixed bytes (44 characters).
//! * slot 6 `XSystemHandleTrack` is accepted and its callback is never called.
//! * slot 8 `XSystemAllowFullDownloadBandwidth` is accepted, nothing downloads.
//! * analytics slot 3 `XSystemGetAnalyticsInfo` reports OS 10.0.25398.0, family `Windows.Xbox` and
//!   form `Xbox Series X`. The 0x90 byte structure is written into the caller's buffer, whose
//!   address is also the return value.
//!
//! The device type is not a runtime call. Code linked into the title reads the `ConsoleMode`
//! registry value, which `console_mode` answers as a Series X, matching [`DEVICE_FORM`]. Other slots
//! log a named "not implemented" (`local_title`). Every call is logged with a bound (`calllog`).
//! Signatures come from the public GDK `XSystem.h`.
#![allow(unsafe_code)]

use std::ffi::c_void;

use crate::calllog::{Counters, logged_shim};
use crate::local_title::{LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, put_string, query_interface, slot};
use crate::tables::Guid;
use crate::taskqueue::S_OK;

/// The XSystem `api_id`.
pub(crate) const XSYSTEM_TABLE: Guid = Guid { data1: 0xE349BD1A, data2: 0xFC20, data3: 0x4E40, data4: [0xB9, 0x9C, 0x41, 0x78, 0xCC, 0x6B, 0x40, 0x9F] };
/// The XSystem interface id.
pub(crate) const XSYSTEM_IID: Guid = Guid { data1: 0x1861CF2E, data2: 0xE18B, data3: 0x4834, data4: [0xA9, 0xF5, 0xB4, 0xA4, 0xE6, 0xEF, 0xB4, 0xCF] };
/// The analytics `api_id`, also its interface.
pub(crate) const XANALYTICS_TABLE: Guid = Guid { data1: 0xB884675D, data2: 0xB738, data3: 0x4A9C, data4: [0x81, 0x5D, 0x9A, 0x9A, 0x1E, 0x0C, 0x6C, 0x9B] };

/// 38 characters, 39 bytes with the NUL (`XSystemConsoleIdBytes`).
pub(crate) const CONSOLE_ID: &str = "00000000000000000000000000000000005358";
/// The retail sandbox.
pub(crate) const SANDBOX: &str = "RETAIL";
/// Base64 of the 32 bytes "Decant offline device identifier": 44 characters.
pub(crate) const DEVICE_ID: &str = "RGVjYW50IG9mZmxpbmUgZGV2aWNlIGlkZW50aWZpZXI=";
/// The console the title runs on, as `console_mode` reports it ([`crate::console_mode::SERIES_X`]).
pub(crate) const DEVICE_FORM: &str = "Xbox Series X";
const DEVICE_FAMILY: &str = "Windows.Xbox";
const OS: Version = Version { major: 10, minor: 0, build: 25398, revision: 0 };

const NAME: &str = "XSystem";
const ANALYTICS: &str = "XSystemAnalytics";
const SLOTS: usize = 12;
const ANALYTICS_SLOTS: usize = 8;
static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static ANALYTICS_COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);
static ANALYTICS_OBJECT: Object = Object::new(&ANALYTICS_VTBL);

/// The object for a pair these tables serve, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    if api == Some(&XSYSTEM_TABLE) && riid == Some(&XSYSTEM_IID) {
        Some((OBJECT.as_out(), NAME))
    } else if api == Some(&XANALYTICS_TABLE) && riid == Some(&XANALYTICS_TABLE) {
        Some((ANALYTICS_OBJECT.as_out(), ANALYTICS))
    } else {
        None
    }
}

struct System;

impl LocalTable for System {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[XSYSTEM_IID];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        match slot {
            3 => Some("XSystemGetConsoleId"),
            4 => Some("XSystemGetXboxLiveSandboxId"),
            5 => Some("XSystemGetAppSpecificDeviceId"),
            6 => Some("XSystemHandleTrack"),
            7 => Some("XSystemIsHandleValid"),
            8 => Some("XSystemAllowFullDownloadBandwidth"),
            _ => None,
        }
    }
    fn object() -> *mut c_void {
        OBJECT.as_out()
    }
}

struct Analytics;

impl LocalTable for Analytics {
    const NAME: &'static str = ANALYTICS;
    const IIDS: &'static [Guid] = &[XANALYTICS_TABLE];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &ANALYTICS_COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        (slot == 3).then_some("XSystemGetAnalyticsInfo")
    }
    fn object() -> *mut c_void {
        ANALYTICS_OBJECT.as_out()
    }
}

/// `XVersion`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Version {
    pub(crate) major: u16,
    pub(crate) minor: u16,
    pub(crate) build: u16,
    pub(crate) revision: u16,
}

/// `XSystemAnalyticsInfo`.
#[repr(C)]
pub(crate) struct AnalyticsInfo {
    pub(crate) os_version: Version,
    pub(crate) hosting_os_version: Version,
    /// NUL terminated.
    pub(crate) family: [u8; 64],
    /// NUL terminated.
    pub(crate) form: [u8; 64],
}

type This = *const c_void;

logged_shim!(NAME, COUNTS, This, 3 as "XSystemGetConsoleId", console_id_entry => report_console_id, (size: usize, buffer: *mut u8, used: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XSystemGetXboxLiveSandboxId", sandbox_id_entry => report_sandbox, (size: usize, buffer: *mut u8, used: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 5 as "XSystemGetAppSpecificDeviceId", device_id_entry => report_device_id, (size: usize, buffer: *mut u8, used: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 6 as "XSystemHandleTrack", handle_track_entry => accept_handle_tracking, (callback: usize, context: *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 8 as "XSystemAllowFullDownloadBandwidth", allow_bandwidth_entry => accept_bandwidth_hint, (allow: bool));
logged_shim!(ANALYTICS, ANALYTICS_COUNTS, This, 3 as "XSystemGetAnalyticsInfo", analytics_info_entry => fill_analytics, (out: *mut AnalyticsInfo) -> *mut AnalyticsInfo);

unsafe fn report_console_id(size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    put_string(CONSOLE_ID, size, buffer, used)
}

unsafe fn report_sandbox(size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    put_string(SANDBOX, size, buffer, used)
}

unsafe fn report_device_id(size: usize, buffer: *mut u8, used: *mut usize) -> i32 {
    put_string(DEVICE_ID, size, buffer, used)
}

/// No handle is ever reported: the callback stays unused.
unsafe fn accept_handle_tracking(_callback: usize, _context: *mut c_void) -> i32 {
    S_OK
}

unsafe fn accept_bandwidth_hint(_allow: bool) {}

fn zero_padded(text: &str) -> [u8; 64] {
    let mut out = [0; 64];
    out[..text.len()].copy_from_slice(text.as_bytes());
    out
}

/// The struct comes back through the caller's buffer, whose address is also the return value.
unsafe fn fill_analytics(out: *mut AnalyticsInfo) -> *mut AnalyticsInfo {
    if !out.is_null() {
        let info = AnalyticsInfo { os_version: OS, hosting_os_version: OS, family: zero_padded(DEVICE_FAMILY), form: zero_padded(DEVICE_FORM) };
        // SAFETY: the caller passes room for the struct.
        unsafe { out.write_unaligned(info) };
    }
    out
}

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<System>),
    slot!(no_count),
    slot!(no_count),
    slot!(console_id_entry),
    slot!(sandbox_id_entry),
    slot!(device_id_entry),
    slot!(handle_track_entry),
    slot!(not_implemented::<System, 7>),
    slot!(allow_bandwidth_entry),
    slot!(not_implemented::<System, 9>),
    slot!(not_implemented::<System, 10>),
    slot!(not_implemented::<System, 11>),
]);

static ANALYTICS_VTBL: Methods<ANALYTICS_SLOTS> = Methods([
    slot!(query_interface::<Analytics>),
    slot!(no_count),
    slot!(no_count),
    slot!(analytics_info_entry),
    slot!(not_implemented::<Analytics, 4>),
    slot!(not_implemented::<Analytics, 5>),
    slot!(not_implemented::<Analytics, 6>),
    slot!(not_implemented::<Analytics, 7>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::E_NOTIMPL;
    use crate::exports::query_without_runtime;
    use crate::local_title::{E_NOT_SUFFICIENT_BUFFER, method};

    fn table(api: &Guid, riid: &Guid, name: &str) -> *mut c_void {
        let mut out = std::ptr::null_mut();
        assert_eq!(query_without_runtime(Some(api), Some(riid), &mut out), (S_OK, Some(name)));
        out
    }

    #[test]
    fn identity_strings_fit_their_documented_sizes() {
        let t = table(&XSYSTEM_TABLE, &XSYSTEM_IID, NAME);
        type Get = unsafe extern "system" fn(This, usize, *mut u8, *mut usize) -> i32;
        let mut buffer = [0u8; 64];
        let mut used = 0;
        for (slot, size, value) in [(3, 39, CONSOLE_ID), (4, 16, SANDBOX), (5, 45, DEVICE_ID)] {
            // SAFETY: the slot gets its GDK arguments.
            unsafe {
                assert_eq!(method::<Get>(t, slot)(t, size, buffer.as_mut_ptr(), &mut used), S_OK, "slot {slot}");
                assert_eq!(&buffer[..used], format!("{value}\0").as_bytes());
                assert!(used <= size);
                assert_eq!(method::<Get>(t, slot)(t, used - 1, buffer.as_mut_ptr(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            }
        }
        assert_eq!(CONSOLE_ID.len(), 38);
        assert_eq!(DEVICE_ID.len(), 44);
        type Unknown = unsafe extern "system" fn(This) -> i32;
        // SAFETY: an unknown slot takes `this` only.
        unsafe {
            assert_eq!(method::<Unknown>(t, 7)(t), E_NOTIMPL);
            assert_eq!(method::<Unknown>(t, 11)(t), E_NOTIMPL);
        }
    }

    #[test]
    fn analytics_describe_a_series_x() {
        let t = table(&XANALYTICS_TABLE, &XANALYTICS_TABLE, ANALYTICS);
        type Get = unsafe extern "system" fn(This, *mut AnalyticsInfo) -> *mut AnalyticsInfo;
        let mut info = std::mem::MaybeUninit::<AnalyticsInfo>::uninit();
        // SAFETY: the slot gets room for the struct.
        let back = unsafe { method::<Get>(t, 3)(t, info.as_mut_ptr()) };
        assert_eq!(back, info.as_mut_ptr());
        // SAFETY: written by the call.
        let info = unsafe { info.assume_init() };
        assert_eq!(size_of::<AnalyticsInfo>(), 0x90);
        assert_eq!(info.os_version, OS);
        assert_eq!(&info.form[..14], b"Xbox Series X\0");
        assert_eq!(&info.family[..13], b"Windows.Xbox\0");
        assert_eq!(crate::console_mode::SERIES_X, 0x64, "the registry answer is a Series X too");
        assert!(object_for(Some(&XSYSTEM_TABLE), Some(&XSYSTEM_TABLE)).is_none());
    }
}
