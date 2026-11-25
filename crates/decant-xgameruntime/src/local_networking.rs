//! The local XNetworking table, `api_id = {37E56907-2F10-41E8-B72F-36EDB185331A}`, which
//! `QueryApiImpl` serves only when there is no PC runtime. One object answers both the `riid`
//! `{37E56907-...}` and `{BF2346B2-39AF-4658-B5EA-44713C7E83B3}`. The table reports an honest
//! offline state and reaches nothing: no socket, no name resolution, no WinHTTP, no `iphlpapi`,
//! no certificate chain. Nothing claims Xbox Live or an online state.
//!
//! * The connectivity hint is a fixed constant for a host without network: level 1 (`None`),
//!   cost 1 (`Unrestricted`), IANA interface type 6, `networkInitialized` true (the stack is up,
//!   nothing is reachable), no data limit, not roaming. Nothing is read from the host.
//! * The preferred local UDP multiplayer port is 3074, the well known Xbox Live port. Only the
//!   number is answered, no port is opened or reserved. The sync and async forms agree.
//! * The security information for a URL is TLS 1.2 plus 1.3 (`0x2800`) and no certificate pin,
//!   delivered through the async block. The URL is never contacted.
//! * `XNetworkingVerifyServerCertificate` rejects a null request or malformed pin metadata with
//!   `E_INVALIDARG` and fails everything else with `E_FAIL`. Offline there is no request whose
//!   certificate could be checked, and building a chain may fetch revocation lists.
//! * Both `Register...Changed` calls hand out a token and never call back, because offline
//!   nothing changes. Both `Unregister` calls answer true.
//!
//! The object is a pointer to a vtable of 24 slots in the public PC GDK `xgameruntime.lib` thunk
//! order. Each slot takes `this` first and drops it. Slots 0 to 2 are IUnknown on a static
//! object and 3 to 17 are the functions above. Slots 18 to 23 are not implemented. They log
//! their index and answer `E_NOTIMPL`. Every call is logged (`calllog`).
#![allow(unsafe_code, non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::E_NOTIMPL;
use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::tables::Guid;
use crate::taskqueue::{E_FAIL, E_INVALIDARG, RegistrationToken, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`, also the `riid` of its main interface.
pub(crate) const XNETWORKING_TABLE: Guid = Guid { data1: 0x37E56907, data2: 0x2F10, data3: 0x41E8, data4: [0xB7, 0x2F, 0x36, 0xED, 0xB1, 0x85, 0x33, 0x1A] };
/// The other `riid` the PC GDK thunks ask the same table for.
pub(crate) const XNETWORKING_BF2346B2: Guid = Guid { data1: 0xBF2346B2, data2: 0x39AF, data3: 0x4658, data4: [0xB5, 0xEA, 0x44, 0x71, 0x3C, 0x7E, 0x83, 0xB3] };
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;

/// The object for a couple served without a PC runtime, with the table's name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    if api != Some(&XNETWORKING_TABLE) {
        return None;
    }
    riid.filter(|r| is_own_interface(r)).map(|_| (object(), "XNetworking"))
}

fn is_own_interface(iid: &Guid) -> bool {
    *iid == XNETWORKING_TABLE || *iid == XNETWORKING_BF2346B2
}

const SLOTS: usize = 24;
const MAIN: &str = "XNetworking";
static COUNTS: Counters<SLOTS> = Counters::new();

type This = *const Object;
type Block = *mut XAsyncBlock;
type SecurityResult = unsafe extern "system" fn(This, Block, usize, *mut usize, *mut u8, *mut *mut XNetworkingSecurityInformation) -> i32;
type Register = unsafe extern "system" fn(This, *mut c_void, *mut c_void, usize, *mut RegistrationToken) -> i32;
type Unregister = unsafe extern "system" fn(This, RegistrationToken, bool) -> bool;

/// The vtable, in slot order.
#[repr(C)]
pub(crate) struct Vtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    query_udp_port: unsafe extern "system" fn(This, *mut u16) -> i32,
    query_udp_port_async: unsafe extern "system" fn(This, Block) -> i32,
    query_udp_port_async_result: unsafe extern "system" fn(This, Block, *mut u16) -> i32,
    register_udp_port_changed: Register,
    unregister_udp_port_changed: Unregister,
    security_info_async: unsafe extern "system" fn(This, *const u8, Block) -> i32,
    security_info_result_size: unsafe extern "system" fn(This, Block, *mut usize) -> i32,
    security_info_result: SecurityResult,
    security_info_utf16_async: unsafe extern "system" fn(This, *const u16, Block) -> i32,
    security_info_utf16_result_size: unsafe extern "system" fn(This, Block, *mut usize) -> i32,
    security_info_utf16_result: SecurityResult,
    verify_server_certificate: unsafe extern "system" fn(This, *mut c_void, *const XNetworkingSecurityInformation) -> i32,
    get_connectivity_hint: unsafe extern "system" fn(This, *mut XNetworkingConnectivityHint) -> i32,
    register_connectivity_hint_changed: Register,
    unregister_connectivity_hint_changed: Unregister,
    slot18: unsafe extern "system" fn(This) -> i32,
    slot19: unsafe extern "system" fn(This) -> i32,
    slot20: unsafe extern "system" fn(This) -> i32,
    slot21: unsafe extern "system" fn(This) -> i32,
    slot22: unsafe extern "system" fn(This) -> i32,
    slot23: unsafe extern "system" fn(This) -> i32,
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

macro_rules! shim {
    ($slot:literal as $gdk:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot as $gdk, $($rest)*);
    };
    ($slot:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot, $($rest)*);
    };
}

shim!(3, query_udp_port => XNetworkingQueryPreferredLocalUdpMultiplayerPort, (port: *mut u16) -> i32);
shim!(4, query_udp_port_async => XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync, (async_block: Block) -> i32);
shim!(5, query_udp_port_async_result => XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult, (async_block: Block, port: *mut u16) -> i32);
shim!(6 as "XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged", register_udp_port_changed => register_changed, (queue: *mut c_void, context: *mut c_void, callback: usize, token: *mut RegistrationToken) -> i32);
shim!(7 as "XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged", unregister_udp_port_changed => unregister_changed, (token: RegistrationToken, wait: bool) -> bool);
shim!(8, security_info_async => XNetworkingQuerySecurityInformationForUrlAsync, (url: *const u8, async_block: Block) -> i32);
shim!(9 as "XNetworkingQuerySecurityInformationForUrlAsyncResultSize", security_info_result_size => xa::XAsyncGetResultSize, (async_block: Block, size: *mut usize) -> i32);
shim!(10, security_info_result => XNetworkingQuerySecurityInformationForUrlAsyncResult, (async_block: Block, size: usize, used: *mut usize, buffer: *mut u8, ptr_to_buffer: *mut *mut XNetworkingSecurityInformation) -> i32);
shim!(11, security_info_utf16_async => XNetworkingQuerySecurityInformationForUrlUtf16Async, (url: *const u16, async_block: Block) -> i32);
shim!(12 as "XNetworkingQuerySecurityInformationForUrlUtf16AsyncResultSize", security_info_utf16_result_size => xa::XAsyncGetResultSize, (async_block: Block, size: *mut usize) -> i32);
shim!(13, security_info_utf16_result => XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult, (async_block: Block, size: usize, used: *mut usize, buffer: *mut u8, ptr_to_buffer: *mut *mut XNetworkingSecurityInformation) -> i32);
shim!(14, verify_server_certificate => XNetworkingVerifyServerCertificate, (request: *mut c_void, security_info: *const XNetworkingSecurityInformation) -> i32);
shim!(15, get_connectivity_hint => XNetworkingGetConnectivityHint, (hint: *mut XNetworkingConnectivityHint) -> i32);
shim!(16 as "XNetworkingRegisterConnectivityHintChanged", register_connectivity_hint_changed => register_changed, (queue: *mut c_void, context: *mut c_void, callback: usize, token: *mut RegistrationToken) -> i32);
shim!(17 as "XNetworkingUnregisterConnectivityHintChanged", unregister_connectivity_hint_changed => unregister_changed, (token: RegistrationToken, wait: bool) -> bool);

/// Declares the slots that have no known function: each logs its index (bounded) and answers
/// `E_NOTIMPL`.
macro_rules! unknown {
    ($($slot:literal $name:ident),*) => {
        $(
            unsafe extern "system" fn $name(_this: This) -> i32 {
                decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("local {MAIN} table slot {}", $slot));
                if let Some(n) = COUNTS.next($slot) {
                    log(&format!("local {MAIN} table: slot {} is not known, answering E_NOTIMPL [call {n}]", $slot));
                }
                E_NOTIMPL
            }
        )*
    };
}

unknown!(18 slot18, 19 slot19, 20 slot20, 21 slot21, 22 slot22, 23 slot23);

static VTBL: Vtbl = Vtbl {
    query_interface,
    add_ref: no_count,
    release: no_count,
    query_udp_port,
    query_udp_port_async,
    query_udp_port_async_result,
    register_udp_port_changed,
    unregister_udp_port_changed,
    security_info_async,
    security_info_result_size,
    security_info_result,
    security_info_utf16_async,
    security_info_utf16_result_size,
    security_info_utf16_result,
    verify_server_certificate,
    get_connectivity_hint,
    register_connectivity_hint_changed,
    unregister_connectivity_hint_changed,
    slot18,
    slot19,
    slot20,
    slot21,
    slot22,
    slot23,
};

/// Both interface ids answer the one object.
unsafe extern "system" fn query_interface(_this: This, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract, `iid` is null or a GUID and `out` a writable pointer location.
    unsafe {
        let known = !iid.is_null() && is_own_interface(&iid.read_unaligned());
        *out = if known { object() } else { std::ptr::null_mut() };
        if known { S_OK } else { E_NOINTERFACE }
    }
}

/// AddRef and Release of a static object.
unsafe extern "system" fn no_count(_this: This) -> u32 {
    1
}

/// Default Xbox multiplayer UDP port.
pub(crate) const DEFAULT_UDP_PORT: u16 = 3074;

/// `WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3`.
pub(crate) const DEFAULT_TLS_FLAGS: u32 = 0x0000_0800 | 0x0000_2000;

/// `XNetworkingThumbprint`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct XNetworkingThumbprint {
    pub(crate) thumbprint_type: u32,
    pub(crate) thumbprint_buffer_byte_count: usize,
    pub(crate) thumbprint_buffer: *const u8,
}

/// `XNetworkingSecurityInformation`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct XNetworkingSecurityInformation {
    pub(crate) enabled_http_security_protocol_flags: u32,
    pub(crate) thumbprint_count: usize,
    pub(crate) thumbprints: *const XNetworkingThumbprint,
}

/// `XNetworkingConnectivityHint`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XNetworkingConnectivityHint {
    pub(crate) connectivity_level: u32,
    pub(crate) connectivity_cost: u32,
    pub(crate) iana_interface_type: u32,
    pub(crate) network_initialized: bool,
    pub(crate) approaching_data_limit: bool,
    pub(crate) over_data_limit: bool,
    pub(crate) roaming: bool,
}

/// Tags that tie an async Result call to the Begin call of the same kind.
static UDP_PORT_ID: u8 = 0;
static SEC_INFO_ID: u8 = 0;
static SEC_INFO_UTF16_ID: u8 = 0;
/// The next registration token. Zero is never handed out.
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Writes `value` through `out`, or answers `E_INVALIDARG` when `out` is null.
///
/// # Safety
/// `out` is null or a writable location for a `T`.
unsafe fn store<T>(out: *mut T, value: T) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: not null, the title's location for a `T`.
    unsafe { out.write_unaligned(value) };
    S_OK
}

// UDP port

unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPort(port: *mut u16) -> i32 {
    // SAFETY: the title's location, null or writable.
    unsafe { store(port, DEFAULT_UDP_PORT) }
}

unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync(b: Block) -> i32 {
    let payload = || Ok(DEFAULT_UDP_PORT.to_le_bytes().to_vec());
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const UDP_PORT_ID as usize, payload) }
}

unsafe extern "system" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult(b: Block, port: *mut u16) -> i32 {
    if port.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block and a writable location of two bytes.
    unsafe { xa::XAsyncGetResult(b, &raw const UDP_PORT_ID as *const c_void, size_of::<u16>(), port.cast(), std::ptr::null_mut()) }
}

// Change registrations, shared by the port and the connectivity hint

unsafe extern "system" fn register_changed(_queue: *mut c_void, _ctx: *mut c_void, _cb: usize, token: *mut RegistrationToken) -> i32 {
    let fresh = RegistrationToken { token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed) };
    // SAFETY: the title's location, null or writable.
    unsafe { store(token, fresh) }
}

unsafe extern "system" fn unregister_changed(_token: RegistrationToken, _wait: bool) -> bool {
    true
}

// Security information

/// The security information every URL gets, as raw bytes. Built field by field so that struct
/// padding is never read: the TLS flags, a zero thumbprint count and a null thumbprint array.
fn default_security_info_bytes() -> Vec<u8> {
    let mut raw = vec![0u8; size_of::<XNetworkingSecurityInformation>()];
    let at = std::mem::offset_of!(XNetworkingSecurityInformation, enabled_http_security_protocol_flags);
    raw[at..at + 4].copy_from_slice(&DEFAULT_TLS_FLAGS.to_le_bytes());
    raw
}

/// Starts the async query for a URL pointer of either width. The URL is only checked for null.
///
/// # Safety
/// `b` is the title's block.
unsafe fn begin_security_info<C>(url: *const C, b: Block, identity: *const u8) -> i32 {
    if url.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the caller's contract.
    unsafe { xa::begin_with(b, identity as usize, || Ok(default_security_info_bytes())) }
}

unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlAsync(url: *const u8, b: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { begin_security_info(url, b, &raw const SEC_INFO_ID) }
}

unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlUtf16Async(url: *const u16, b: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { begin_security_info(url, b, &raw const SEC_INFO_UTF16_ID) }
}

/// The GDK orders the Result parameters as size, used, buffer, pointer to buffer.
unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlAsyncResult(
    b: Block,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> i32 {
    // SAFETY: the title's block, buffer and locations.
    unsafe { finish_security_info(&raw const SEC_INFO_ID, b, buffer_size, buffer_used, buffer, out_ptr) }
}

unsafe extern "system" fn XNetworkingQuerySecurityInformationForUrlUtf16AsyncResult(
    b: Block,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> i32 {
    // SAFETY: the title's block, buffer and locations.
    unsafe { finish_security_info(&raw const SEC_INFO_UTF16_ID, b, buffer_size, buffer_used, buffer, out_ptr) }
}

/// Copies the finished result into the title's buffer and points `out_ptr` at it on success.
///
/// # Safety
/// `b` is the title's block, `buffer` holds `buffer_size` bytes, `buffer_used` and `out_ptr` are
/// null or writable.
unsafe fn finish_security_info(
    identity: *const u8,
    b: Block,
    buffer_size: usize,
    buffer_used: *mut usize,
    buffer: *mut u8,
    out_ptr: *mut *mut XNetworkingSecurityInformation,
) -> i32 {
    let mut used = 0usize;
    // SAFETY: the caller's contract.
    unsafe {
        if !out_ptr.is_null() {
            *out_ptr = std::ptr::null_mut();
        }
        let hr = xa::XAsyncGetResult(b, identity.cast(), buffer_size, buffer.cast(), &mut used);
        if !buffer_used.is_null() {
            *buffer_used = used;
        }
        if hr >= 0 && !out_ptr.is_null() {
            *out_ptr = buffer.cast();
        }
        hr
    }
}

// Certificate verification

/// The length of a SHA-1 thumbprint.
const SHA1_LEN: usize = 20;
/// The highest `XNetworkingThumbprintType` value.
const THUMBPRINT_TYPE_MAX: u32 = 2;

/// Whether the pin metadata of a security information structure is well formed. A null
/// structure or a zero count has no pins and is fine. Otherwise the array must exist, its byte
/// size must fit an allocation, and every pin needs a buffer of exactly 20 bytes and a known type.
///
/// # Safety
/// `security_info` is null or the title's structure, its array as long as its count says.
unsafe fn pins_are_well_formed(security_info: *const XNetworkingSecurityInformation) -> bool {
    if security_info.is_null() {
        return true;
    }
    // SAFETY: the caller's contract.
    let info = unsafe { security_info.read_unaligned() };
    if info.thumbprint_count == 0 {
        return true;
    }
    let array_fits = info.thumbprint_count.checked_mul(size_of::<XNetworkingThumbprint>()).is_some_and(|bytes| bytes <= isize::MAX as usize);
    if info.thumbprints.is_null() || !array_fits {
        return false;
    }
    (0..info.thumbprint_count).all(|i| {
        // SAFETY: the caller's contract, `thumbprint_count` entries.
        let pin = unsafe { info.thumbprints.add(i).read_unaligned() };
        !pin.thumbprint_buffer.is_null() && pin.thumbprint_buffer_byte_count == SHA1_LEN && pin.thumbprint_type <= THUMBPRINT_TYPE_MAX
    })
}

unsafe extern "system" fn XNetworkingVerifyServerCertificate(request_handle: *mut c_void, security_info: *const XNetworkingSecurityInformation) -> i32 {
    // SAFETY: the title's structure, null or valid.
    if request_handle.is_null() || !unsafe { pins_are_well_formed(security_info) } {
        return E_INVALIDARG;
    }
    E_FAIL
}

// Connectivity hint

/// The hint of a host without network. The cost is the neutral `Unrestricted` and the interface
/// type is `IF_TYPE_ETHERNET_CSMACD`.
pub(crate) const OFFLINE_HINT: XNetworkingConnectivityHint = XNetworkingConnectivityHint {
    connectivity_level: 1,
    connectivity_cost: 1,
    iana_interface_type: 6,
    network_initialized: true,
    approaching_data_limit: false,
    over_data_limit: false,
    roaming: false,
};

unsafe extern "system" fn XNetworkingGetConnectivityHint(out: *mut XNetworkingConnectivityHint) -> i32 {
    static ANNOUNCED: AtomicBool = AtomicBool::new(false);
    if !out.is_null() && !ANNOUNCED.swap(true, Ordering::Relaxed) {
        let h = OFFLINE_HINT;
        log(&format!(
            "XNetworkingGetConnectivityHint -> offline: level {} (None), cost {} (Unrestricted), interface type {}, network initialized {}, no data limit, not roaming. There is no PC runtime and nothing is contacted.",
            h.connectivity_level, h.connectivity_cost, h.iana_interface_type, h.network_initialized
        ));
    }
    // SAFETY: the title's location, null or writable.
    unsafe { store(out, OFFLINE_HINT) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xasync::XAsyncGetStatus;

    const THIS: This = &OBJECT;

    /// Runs an async Begin call and waits for its block.
    unsafe fn finished(block: &mut XAsyncBlock) {
        assert_eq!(unsafe { XAsyncGetStatus(block, true) }, S_OK);
    }

    #[test]
    fn the_vtable_is_24_pointers_and_the_object_points_at_it() {
        assert_eq!(size_of::<Vtbl>(), SLOTS * size_of::<usize>());
        // SAFETY: the object's first field is its vtable pointer, and `Vtbl` is all function pointers.
        unsafe {
            assert!(std::ptr::eq(*(object() as *const *const Vtbl), &raw const VTBL));
            let table = &*((&raw const VTBL).cast::<[usize; SLOTS]>());
            assert_eq!(table[3], query_udp_port as *const () as usize);
            assert_eq!(table[14], verify_server_certificate as *const () as usize);
            assert_eq!(table[15], get_connectivity_hint as *const () as usize);
            assert_eq!(table[17], unregister_connectivity_hint_changed as *const () as usize);
            assert_eq!(table[23], slot23 as *const () as usize);
        }
    }

    #[test]
    fn the_structures_have_the_gdk_sizes() {
        assert_eq!(size_of::<XNetworkingThumbprint>(), 24);
        assert_eq!(size_of::<XNetworkingSecurityInformation>(), 24);
        assert_eq!(size_of::<XNetworkingConnectivityHint>(), 16);
        assert_eq!(std::mem::offset_of!(XNetworkingSecurityInformation, thumbprint_count), 8);
        assert_eq!(std::mem::offset_of!(XNetworkingConnectivityHint, network_initialized), 12);
    }

    #[test]
    fn only_the_networking_couples_are_served() {
        let unrelated = Guid { data1: 1, data2: 2, data3: 3, data4: [4; 8] };
        for riid in [XNETWORKING_TABLE, XNETWORKING_BF2346B2] {
            assert_eq!(object_for(Some(&XNETWORKING_TABLE), Some(&riid)), Some((object(), "XNetworking")));
        }
        assert_eq!(object_for(Some(&XNETWORKING_TABLE), Some(&unrelated)), None);
        assert_eq!(object_for(Some(&XNETWORKING_TABLE), None), None);
        assert_eq!(object_for(Some(&unrelated), Some(&XNETWORKING_TABLE)), None);
        assert_eq!(object_for(None, Some(&XNETWORKING_TABLE)), None);
    }

    #[test]
    fn query_interface_and_unknown_slots() {
        let unrelated = Guid { data1: 1, data2: 2, data3: 3, data4: [4; 8] };
        let mut out = std::ptr::dangling_mut::<c_void>();
        // SAFETY: valid GUIDs and out locations.
        unsafe {
            assert_eq!((VTBL.query_interface)(THIS, &XNETWORKING_BF2346B2, &mut out), S_OK);
            assert_eq!(out, object());
            assert_eq!((VTBL.query_interface)(THIS, &unrelated, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!((VTBL.query_interface)(THIS, std::ptr::null(), &mut out), E_NOINTERFACE);
            assert_eq!((VTBL.query_interface)(THIS, &unrelated, std::ptr::null_mut()), E_POINTER);
            assert_eq!((VTBL.add_ref)(THIS), 1);
            assert_eq!((VTBL.release)(THIS), 1);
            for slot in [VTBL.slot18, VTBL.slot19, VTBL.slot20, VTBL.slot21, VTBL.slot22, VTBL.slot23] {
                assert_eq!(slot(THIS), E_NOTIMPL);
            }
        }
    }

    #[test]
    fn the_hint_describes_a_host_without_network() {
        let mut hint = XNetworkingConnectivityHint {
            connectivity_level: 3,
            connectivity_cost: 0,
            iana_interface_type: 0,
            network_initialized: false,
            approaching_data_limit: true,
            over_data_limit: true,
            roaming: true,
        };
        // SAFETY: GDK arguments.
        unsafe {
            assert_eq!((VTBL.get_connectivity_hint)(THIS, &mut hint), S_OK);
            assert_eq!((VTBL.get_connectivity_hint)(THIS, std::ptr::null_mut()), E_INVALIDARG);
        }
        assert_eq!(hint, OFFLINE_HINT);
        assert_eq!((hint.connectivity_level, hint.connectivity_cost, hint.iana_interface_type), (1, 1, 6));
        assert!(hint.network_initialized);
        assert!(!hint.approaching_data_limit && !hint.over_data_limit && !hint.roaming);
    }

    #[test]
    fn the_udp_port_is_3074_sync_and_async() {
        // SAFETY: GDK arguments, the block runs on the process queue.
        unsafe {
            let mut port = 0u16;
            assert_eq!((VTBL.query_udp_port)(THIS, &mut port), S_OK);
            assert_eq!(port, 3074);
            assert_eq!((VTBL.query_udp_port)(THIS, std::ptr::null_mut()), E_INVALIDARG);

            let mut block: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.query_udp_port_async)(THIS, &mut block), S_OK);
            finished(&mut block);
            let mut later = 0u16;
            assert_eq!((VTBL.query_udp_port_async_result)(THIS, &mut block, &mut later), S_OK);
            assert_eq!(later, port);
            assert_eq!((VTBL.query_udp_port_async_result)(THIS, &mut block, std::ptr::null_mut()), E_INVALIDARG);
        }
    }

    #[test]
    fn registrations_give_distinct_nonzero_tokens() {
        let n = std::ptr::null_mut();
        let (mut first, mut second) = (RegistrationToken { token: 0 }, RegistrationToken { token: 0 });
        // SAFETY: GDK arguments.
        unsafe {
            assert_eq!((VTBL.register_connectivity_hint_changed)(THIS, n, n, 0x1234, &mut first), S_OK);
            assert_eq!((VTBL.register_udp_port_changed)(THIS, n, n, 0x1234, &mut second), S_OK);
            assert_ne!(first.token, 0);
            assert_ne!(first.token, second.token);
            assert_eq!((VTBL.register_udp_port_changed)(THIS, n, n, 0x1234, std::ptr::null_mut()), E_INVALIDARG);
            assert!((VTBL.unregister_connectivity_hint_changed)(THIS, first, true));
            assert!((VTBL.unregister_udp_port_changed)(THIS, second, false));
        }
    }

    /// A buffer that satisfies the alignment of the security information structure.
    #[repr(C)]
    struct AlignedBuffer {
        _align: [XNetworkingSecurityInformation; 0],
        bytes: [u8; 32],
    }

    #[test]
    fn security_information_is_tls_only_in_both_encodings() {
        let wide_url: Vec<u16> = "https://example.invalid/\0".encode_utf16().collect();
        let mut buf = AlignedBuffer { _align: [], bytes: [0xAA; 32] };
        // SAFETY: GDK arguments, the blocks run on the process queue.
        unsafe {
            let mut block: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.security_info_async)(THIS, c"https://example.invalid/".as_ptr().cast(), &mut block), S_OK);
            finished(&mut block);
            let mut size = 0usize;
            assert_eq!((VTBL.security_info_result_size)(THIS, &mut block, &mut size), S_OK);
            assert_eq!(size, 24);
            let (mut used, mut info) = (0usize, std::ptr::null_mut());
            assert_eq!((VTBL.security_info_result)(THIS, &mut block, 32, &mut used, buf.bytes.as_mut_ptr(), &mut info), S_OK);
            assert_eq!(used, size);
            assert_eq!(info as *const u8, buf.bytes.as_ptr());
            assert_eq!((*info).enabled_http_security_protocol_flags, 0x2800);
            assert_eq!((*info).thumbprint_count, 0);
            assert!((*info).thumbprints.is_null());

            let mut block: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.security_info_utf16_async)(THIS, wide_url.as_ptr(), &mut block), S_OK);
            finished(&mut block);
            buf.bytes.fill(0x55);
            assert_eq!((VTBL.security_info_utf16_result)(THIS, &mut block, 32, &mut used, buf.bytes.as_mut_ptr(), &mut info), S_OK);
            assert_eq!((*info).enabled_http_security_protocol_flags, 0x2800);
            assert_eq!((*info).thumbprint_count, 0);

            assert_eq!((VTBL.security_info_async)(THIS, std::ptr::null(), &mut block), E_INVALIDARG);
            assert_eq!((VTBL.security_info_utf16_async)(THIS, std::ptr::null(), &mut block), E_INVALIDARG);
        }
    }

    #[test]
    fn a_too_small_result_buffer_is_reported() {
        let mut buf = AlignedBuffer { _align: [], bytes: [0; 32] };
        // SAFETY: GDK arguments.
        unsafe {
            let mut block: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.security_info_async)(THIS, c"https://example.invalid/".as_ptr().cast(), &mut block), S_OK);
            finished(&mut block);
            let (mut used, mut info) = (0usize, std::ptr::dangling_mut());
            let hr = (VTBL.security_info_result)(THIS, &mut block, 4, &mut used, buf.bytes.as_mut_ptr(), &mut info);
            assert!(hr < 0);
            assert!(info.is_null());
        }
    }

    fn security_info(thumbprints: *const XNetworkingThumbprint, count: usize) -> XNetworkingSecurityInformation {
        XNetworkingSecurityInformation { enabled_http_security_protocol_flags: DEFAULT_TLS_FLAGS, thumbprint_count: count, thumbprints }
    }

    #[test]
    fn certificate_verification_validates_pins_then_fails() {
        let digest = [0xAAu8; 20];
        let pin = |kind: u32, len: usize, data: *const u8| XNetworkingThumbprint { thumbprint_type: kind, thumbprint_buffer_byte_count: len, thumbprint_buffer: data };
        let good = pin(0, 20, digest.as_ptr());
        let short = pin(0, 16, digest.as_ptr());
        let unknown_type = pin(3, 20, digest.as_ptr());
        let no_buffer = pin(0, 20, std::ptr::null());
        let two_good = [good, pin(2, 20, digest.as_ptr())];
        let request = std::ptr::dangling_mut::<c_void>();
        let verify = |request: *mut c_void, info: *const XNetworkingSecurityInformation| {
            // SAFETY: the structures are valid and the request is never dereferenced.
            unsafe { (VTBL.verify_server_certificate)(THIS, request, info) }
        };
        assert_eq!(verify(std::ptr::null_mut(), std::ptr::null()), E_INVALIDARG);
        for bad in [&short, &unknown_type, &no_buffer] {
            assert_eq!(verify(request, &security_info(bad, 1)), E_INVALIDARG);
        }
        assert_eq!(verify(request, &security_info(std::ptr::null(), 1)), E_INVALIDARG);
        assert_eq!(verify(request, &security_info(&good, usize::MAX)), E_INVALIDARG);
        assert_eq!(verify(request, &security_info(&good, 1)), E_FAIL);
        assert_eq!(verify(request, &security_info(two_good.as_ptr(), 2)), E_FAIL);
        assert_eq!(verify(request, &security_info(std::ptr::null(), 0)), E_FAIL);
        assert_eq!(verify(request, std::ptr::null()), E_FAIL);
    }
}
