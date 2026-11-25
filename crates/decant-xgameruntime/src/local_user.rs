//! The local XUser table, `api_id = riid = {01ACD177-91F9-4763-A38E-CCBB55CE32E0}`, and its
//! gamertag interface `(api_id, {CEF4FAC0-7676-4A94-A119-4C43F9EB5B74})`, which `QueryApiImpl`
//! serves only when there is no PC runtime. It offers one local offline user, signed in, so that
//! `XUserAddAsync` completes with it whatever the options (there is no other user to pick and no
//! UI to show). Nothing reaches a service and nothing is faked for the store or Xbox Live:
//!
//! * `XUserIsStoreUser` answers false, the local user has no store account.
//! * The token and signature calls fail through their async block with `0x80072EFD`
//!   (`WININET_E_CANNOT_CONNECT`, no Xbox Live). No token is made up.
//! * `XUserGetGamerPictureAsync` fails through its block with the same error, because there is
//!   no profile to download a picture from.
//! * `XUserCheckPrivilege` answers that the user has the privilege, because there is no Xbox
//!   Live to deny one. `XUserResolvePrivilegeWithUiAsync` and the `ResolveIssue` calls complete
//!   at once.
//!
//! The main object is a pointer to a vtable of 50 slots in the public PC GDK `xgameruntime.lib`
//! thunk order, each slot taking `this` first and dropping it. Slots 0 to 2 are IUnknown on a
//! static object. Slots 3 to 38 are the XUser functions. Slot 42 is `XUserIsStoreUser` and 47
//! `XUserIsSignOutPresent`, both false (they return a bool, which an `E_NOTIMPL` would turn into
//! true). The other slots (15, 39 to 41, 43 to 46, 48, 49) log their index and answer
//! `E_NOTIMPL`. The gamertag object has `XUserGetGamertag` at slot 3. Every call is logged
//! (`calllog`).
#![allow(unsafe_code, non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::E_NOTIMPL;
use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, RegistrationToken, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`, also the `riid` of its main interface.
pub(crate) const XUSER_TABLE: Guid = Guid { data1: 0x01ACD177, data2: 0x91F9, data3: 0x4763, data4: [0xA3, 0x8E, 0xCC, 0xBB, 0x55, 0xCE, 0x32, 0xE0] };
/// The `riid` of `XUserGetGamertag`.
pub(crate) const XUSER_GAMERTAG: Guid = Guid { data1: 0xCEF4FAC0, data2: 0x7676, data3: 0x4A94, data4: [0xA1, 0x19, 0x4C, 0x43, 0xF9, 0xEB, 0x5B, 0x74] };
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;
const E_NOT_SUFFICIENT_BUFFER: i32 = 0x8007_007A_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`.
const E_NOT_FOUND: i32 = 0x8007_0490_u32 as i32;
/// `WININET_E_CANNOT_CONNECT`: what a console that cannot reach Xbox Live gets.
pub(crate) const E_CANNOT_CONNECT: i32 = 0x8007_2EFD_u32 as i32;
const E_GAMEUSER_DEFERRAL_NOT_AVAILABLE: i32 = 0x8924_5103_u32 as i32;
const E_GAMEUSER_USER_NOT_FOUND: i32 = 0x8924_5104_u32 as i32;

/// The main object and the gamertag object for a couple served without a PC runtime, with the
/// table's name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    if api != Some(&XUSER_TABLE) {
        return None;
    }
    match riid {
        Some(r) if *r == XUSER_TABLE => Some((object(), "XUser")),
        Some(r) if *r == XUSER_GAMERTAG => Some((gamertag_object(), "XUser gamertag")),
        _ => None,
    }
}

const SLOTS: usize = 50;
const MAIN: &str = "XUser";
const GAMERTAG: &str = "XUser gamertag";
static COUNTS: Counters<SLOTS> = Counters::new();
static GAMERTAG_COUNTS: Counters<GAMERTAG_SLOTS> = Counters::new();

type This = *const Object;
type Handle = *mut c_void;
type Block = *mut XAsyncBlock;

/// The main vtable, in slot order.
#[repr(C)]
pub(crate) struct Vtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    duplicate_handle: unsafe extern "system" fn(This, Handle, *mut Handle) -> i32,
    close_handle: unsafe extern "system" fn(This, Handle),
    compare: unsafe extern "system" fn(This, Handle, Handle) -> i32,
    get_max_users: unsafe extern "system" fn(This, *mut u32) -> i32,
    add_async: unsafe extern "system" fn(This, u32, Block) -> i32,
    add_result: unsafe extern "system" fn(This, Block, *mut Handle) -> i32,
    get_local_id: unsafe extern "system" fn(This, Handle, *mut u64) -> i32,
    find_user_by_local_id: unsafe extern "system" fn(This, u64, *mut Handle) -> i32,
    get_id: unsafe extern "system" fn(This, Handle, *mut u64) -> i32,
    find_user_by_id: unsafe extern "system" fn(This, u64, *mut Handle) -> i32,
    get_is_guest: unsafe extern "system" fn(This, Handle, *mut bool) -> i32,
    get_state: unsafe extern "system" fn(This, Handle, *mut u32) -> i32,
    slot15: unsafe extern "system" fn(This) -> i32,
    get_gamer_picture_async: unsafe extern "system" fn(This, Handle, u32, Block) -> i32,
    get_gamer_picture_result_size: unsafe extern "system" fn(This, Block, *mut usize) -> i32,
    get_gamer_picture_result: unsafe extern "system" fn(This, Block, usize, *mut c_void, *mut usize) -> i32,
    get_age_group: unsafe extern "system" fn(This, Handle, *mut u32) -> i32,
    check_privilege: unsafe extern "system" fn(This, Handle, u32, u32, *mut bool, *mut u32) -> i32,
    resolve_privilege_with_ui_async: unsafe extern "system" fn(This, Handle, u32, u32, Block) -> i32,
    resolve_privilege_with_ui_result: unsafe extern "system" fn(This, Block) -> i32,
    get_token_and_signature_async: TokenAsync,
    get_token_and_signature_result_size: unsafe extern "system" fn(This, Block, *mut usize) -> i32,
    get_token_and_signature_result: TokenResult,
    get_token_and_signature_utf16_async: TokenAsync,
    get_token_and_signature_utf16_result_size: unsafe extern "system" fn(This, Block, *mut usize) -> i32,
    get_token_and_signature_utf16_result: TokenResult,
    resolve_issue_with_ui_async: unsafe extern "system" fn(This, Handle, *const c_void, Block) -> i32,
    resolve_issue_with_ui_result: unsafe extern "system" fn(This, Block) -> i32,
    resolve_issue_with_ui_utf16_async: unsafe extern "system" fn(This, Handle, *const c_void, Block) -> i32,
    resolve_issue_with_ui_utf16_result: unsafe extern "system" fn(This, Block) -> i32,
    register_for_change_event: unsafe extern "system" fn(This, *mut c_void, *mut c_void, usize, *mut RegistrationToken) -> i32,
    unregister_for_change_event: unsafe extern "system" fn(This, RegistrationToken, bool) -> bool,
    get_sign_out_deferral: unsafe extern "system" fn(This, *mut usize) -> i32,
    close_sign_out_deferral_handle: unsafe extern "system" fn(This, usize),
    add_by_id_with_ui_async: unsafe extern "system" fn(This, u64, Block) -> i32,
    add_by_id_with_ui_result: unsafe extern "system" fn(This, Block, *mut Handle) -> i32,
    slot39: unsafe extern "system" fn(This) -> i32,
    slot40: unsafe extern "system" fn(This) -> i32,
    slot41: unsafe extern "system" fn(This) -> i32,
    is_store_user: unsafe extern "system" fn(This, Handle) -> bool,
    slot43: unsafe extern "system" fn(This) -> i32,
    slot44: unsafe extern "system" fn(This) -> i32,
    slot45: unsafe extern "system" fn(This) -> i32,
    slot46: unsafe extern "system" fn(This) -> i32,
    is_sign_out_present: unsafe extern "system" fn(This) -> bool,
    slot48: unsafe extern "system" fn(This) -> i32,
    slot49: unsafe extern "system" fn(This) -> i32,
}

/// `XUserGetTokenAndSignature[Utf16]Async(user, options, method, url, header_count, headers,
/// body_size, body, async)`.
type TokenAsync = unsafe extern "system" fn(This, Handle, u32, *const c_void, *const c_void, usize, *const c_void, usize, *const c_void, Block) -> i32;
/// `XUserGetTokenAndSignature[Utf16]Result(async, size, buffer, ptr_to_buffer, used)`.
type TokenResult = unsafe extern "system" fn(This, Block, usize, *mut c_void, *mut *mut c_void, *mut usize) -> i32;

/// The COM style object: a pointer to its vtable.
#[repr(C)]
pub(crate) struct Object {
    vtbl: *const c_void,
}

// SAFETY: the object only holds the address of a static vtable.
unsafe impl Sync for Object {}

static OBJECT: Object = Object { vtbl: (&raw const VTBL).cast() };
static GAMERTAG_OBJECT: Object = Object { vtbl: (&raw const GAMERTAG_VTBL).cast() };

/// The main object's address, as written to `*out`.
pub(crate) fn object() -> *mut c_void {
    (&raw const OBJECT).cast_mut().cast()
}

/// The gamertag object's address.
pub(crate) fn gamertag_object() -> *mut c_void {
    (&raw const GAMERTAG_OBJECT).cast_mut().cast()
}

macro_rules! shim {
    ($slot:literal as $gdk:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot as $gdk, $($rest)*);
    };
    ($slot:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot, $($rest)*);
    };
}

shim!(3, duplicate_handle => XUserDuplicateHandle, (h: Handle, out: *mut Handle) -> i32);
shim!(4, close_handle => XUserCloseHandle, (h: Handle));
shim!(5, compare => XUserCompare, (a: Handle, b: Handle) -> i32);
shim!(6, get_max_users => XUserGetMaxUsers, (out: *mut u32) -> i32);
shim!(7, add_async => XUserAddAsync, (options: u32, async_block: Block) -> i32);
shim!(8, add_result => XUserAddResult, (async_block: Block, out: *mut Handle) -> i32);
shim!(9, get_local_id => XUserGetLocalId, (h: Handle, out: *mut u64) -> i32);
shim!(10 as "XUserFindUserByLocalId", find_user_by_local_id => find_user_by_any_id, (id: u64, out: *mut Handle) -> i32);
shim!(11, get_id => XUserGetId, (h: Handle, out: *mut u64) -> i32);
shim!(12 as "XUserFindUserById", find_user_by_id => find_user_by_any_id, (id: u64, out: *mut Handle) -> i32);
shim!(13, get_is_guest => XUserGetIsGuest, (h: Handle, out: *mut bool) -> i32);
shim!(14, get_state => XUserGetState, (h: Handle, out: *mut u32) -> i32);
shim!(16 as "XUserGetGamerPictureAsync", get_gamer_picture_async => gamer_picture_async, (h: Handle, size: u32, async_block: Block) -> i32);
shim!(17 as "XUserGetGamerPictureResultSize", get_gamer_picture_result_size => xa::XAsyncGetResultSize, (async_block: Block, size: *mut usize) -> i32);
shim!(18 as "XUserGetGamerPictureResult", get_gamer_picture_result => gamer_picture_result, (async_block: Block, size: usize, buffer: *mut c_void, used: *mut usize) -> i32);
shim!(19, get_age_group => XUserGetAgeGroup, (h: Handle, out: *mut u32) -> i32);
shim!(20, check_privilege => XUserCheckPrivilege, (h: Handle, options: u32, privilege: u32, has: *mut bool, reasons: *mut u32) -> i32);
shim!(21 as "XUserResolvePrivilegeWithUiAsync", resolve_privilege_with_ui_async => resolve_privilege_async, (h: Handle, options: u32, privilege: u32, async_block: Block) -> i32);
shim!(22 as "XUserResolvePrivilegeWithUiResult", resolve_privilege_with_ui_result => empty_result, (async_block: Block) -> i32);
shim!(23 as "XUserGetTokenAndSignatureAsync", get_token_and_signature_async => token_async_utf8, (h: Handle, options: u32, method: *const c_void, url: *const c_void, header_count: usize, headers: *const c_void, body_size: usize, body: *const c_void, async_block: Block) -> i32);
shim!(24 as "XUserGetTokenAndSignatureResultSize", get_token_and_signature_result_size => xa::XAsyncGetResultSize, (async_block: Block, size: *mut usize) -> i32);
shim!(25 as "XUserGetTokenAndSignatureResult", get_token_and_signature_result => token_result_utf8, (async_block: Block, size: usize, buffer: *mut c_void, ptr_to_buffer: *mut *mut c_void, used: *mut usize) -> i32);
shim!(26 as "XUserGetTokenAndSignatureUtf16Async", get_token_and_signature_utf16_async => token_async_utf16, (h: Handle, options: u32, method: *const c_void, url: *const c_void, header_count: usize, headers: *const c_void, body_size: usize, body: *const c_void, async_block: Block) -> i32);
shim!(27 as "XUserGetTokenAndSignatureUtf16ResultSize", get_token_and_signature_utf16_result_size => xa::XAsyncGetResultSize, (async_block: Block, size: *mut usize) -> i32);
shim!(28 as "XUserGetTokenAndSignatureUtf16Result", get_token_and_signature_utf16_result => token_result_utf16, (async_block: Block, size: usize, buffer: *mut c_void, ptr_to_buffer: *mut *mut c_void, used: *mut usize) -> i32);
shim!(29 as "XUserResolveIssueWithUiAsync", resolve_issue_with_ui_async => resolve_issue_async, (h: Handle, url: *const c_void, async_block: Block) -> i32);
shim!(30 as "XUserResolveIssueWithUiResult", resolve_issue_with_ui_result => empty_result, (async_block: Block) -> i32);
shim!(31 as "XUserResolveIssueWithUiUtf16Async", resolve_issue_with_ui_utf16_async => resolve_issue_async, (h: Handle, url: *const c_void, async_block: Block) -> i32);
shim!(32 as "XUserResolveIssueWithUiUtf16Result", resolve_issue_with_ui_utf16_result => empty_result, (async_block: Block) -> i32);
shim!(33, register_for_change_event => XUserRegisterForChangeEvent, (queue: *mut c_void, context: *mut c_void, callback: usize, token: *mut RegistrationToken) -> i32);
shim!(34, unregister_for_change_event => XUserUnregisterForChangeEvent, (token: RegistrationToken, wait: bool) -> bool);
shim!(35 as "XUserGetSignOutDeferral", get_sign_out_deferral => sign_out_deferral, (out: *mut usize) -> i32);
shim!(36 as "XUserCloseSignOutDeferralHandle", close_sign_out_deferral_handle => nothing, (deferral: usize));
shim!(37 as "XUserAddByIdWithUiAsync", add_by_id_with_ui_async => add_by_id_async, (id: u64, async_block: Block) -> i32);
shim!(38 as "XUserAddByIdWithUiResult", add_by_id_with_ui_result => XUserAddResult, (async_block: Block, out: *mut Handle) -> i32);
shim!(42 as "XUserIsStoreUser", is_store_user => is_store_user_impl, (h: Handle) -> bool);
shim!(47 as "XUserIsSignOutPresent", is_sign_out_present => no, () -> bool);

/// Declares unknown slots of a table: each logs its index (bounded) and answers `E_NOTIMPL`.
macro_rules! unknown {
    ($table:expr, $counts:expr, $this:ty, $($slot:literal $name:ident),*) => {
        $(
            unsafe extern "system" fn $name(_this: $this) -> i32 {
                decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("local {} table slot {}", $table, $slot));
                if let Some(n) = $counts.next($slot) {
                    log(&format!("local {} table: slot {} is not known -> E_NOTIMPL [call {n}]", $table, $slot));
                }
                E_NOTIMPL
            }
        )*
    };
}

unknown!(MAIN, COUNTS, This, 15 slot15, 39 slot39, 40 slot40, 41 slot41, 43 slot43, 44 slot44, 45 slot45, 46 slot46, 48 slot48, 49 slot49);

static VTBL: Vtbl = Vtbl {
    query_interface,
    add_ref: no_count,
    release: no_count,
    duplicate_handle,
    close_handle,
    compare,
    get_max_users,
    add_async,
    add_result,
    get_local_id,
    find_user_by_local_id,
    get_id,
    find_user_by_id,
    get_is_guest,
    get_state,
    slot15,
    get_gamer_picture_async,
    get_gamer_picture_result_size,
    get_gamer_picture_result,
    get_age_group,
    check_privilege,
    resolve_privilege_with_ui_async,
    resolve_privilege_with_ui_result,
    get_token_and_signature_async,
    get_token_and_signature_result_size,
    get_token_and_signature_result,
    get_token_and_signature_utf16_async,
    get_token_and_signature_utf16_result_size,
    get_token_and_signature_utf16_result,
    resolve_issue_with_ui_async,
    resolve_issue_with_ui_result,
    resolve_issue_with_ui_utf16_async,
    resolve_issue_with_ui_utf16_result,
    register_for_change_event,
    unregister_for_change_event,
    get_sign_out_deferral,
    close_sign_out_deferral_handle,
    add_by_id_with_ui_async,
    add_by_id_with_ui_result,
    slot39,
    slot40,
    slot41,
    is_store_user,
    slot43,
    slot44,
    slot45,
    slot46,
    is_sign_out_present,
    slot48,
    slot49,
};

/// The gamertag vtable: IUnknown, `XUserGetGamertag`, and unknown slots up to 7.
#[repr(C)]
pub(crate) struct GamertagVtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    get_gamertag: unsafe extern "system" fn(This, Handle, u32, usize, *mut u8, *mut usize) -> i32,
    slot4: unsafe extern "system" fn(This) -> i32,
    slot5: unsafe extern "system" fn(This) -> i32,
    slot6: unsafe extern "system" fn(This) -> i32,
    slot7: unsafe extern "system" fn(This) -> i32,
}

const GAMERTAG_SLOTS: usize = 8;

logged_shim!(GAMERTAG, GAMERTAG_COUNTS, This, 3, get_gamertag => XUserGetGamertag, (h: Handle, component: u32, size: usize, buffer: *mut u8, used: *mut usize) -> i32);
unknown!(GAMERTAG, GAMERTAG_COUNTS, This, 4 slot4, 5 slot5, 6 slot6, 7 slot7);

static GAMERTAG_VTBL: GamertagVtbl = GamertagVtbl { query_interface, add_ref: no_count, release: no_count, get_gamertag, slot4, slot5, slot6, slot7 };

/// Both objects answer both interfaces, each with its own object.
unsafe extern "system" fn query_interface(_this: This, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract: `iid` is null or a GUID, `out` a writable pointer location.
    unsafe {
        let iid = if iid.is_null() { None } else { Some(iid.read_unaligned()) };
        *out = match iid {
            Some(i) if i == XUSER_TABLE => object(),
            Some(i) if i == XUSER_GAMERTAG => gamertag_object(),
            _ => {
                *out = std::ptr::null_mut();
                return E_NOINTERFACE;
            }
        };
    }
    S_OK
}

/// AddRef and Release of a static object.
unsafe extern "system" fn no_count(_this: This) -> u32 {
    1
}

// The offline user

/// The user handle. It is a static object, so a handle never dangles.
#[repr(C)]
struct User {
    magic: u32,
    refs: AtomicU32,
}

static USER: User = User { magic: 0x5553_4552, refs: AtomicU32::new(1) };
/// The id `XUserGetLocalId` reports.
const LOCAL_ID: u64 = 1;
/// The Xbox user id `XUserGetId` reports.
pub(crate) const USER_ID: u64 = 0x0009_0000_0000_0001;
const GAMERTAG_TEXT: &str = "Player1";
/// Tags that tie an async Result call to the Begin call of the same kind.
static USER_ADD_IDENTITY: u8 = 0;
static TOKEN_UTF8_ID: u8 = 0;
static TOKEN_UTF16_ID: u8 = 0;
static PICTURE_ID: u8 = 0;
static EMPTY_ID: u8 = 0;

fn user_ptr() -> Handle {
    (&raw const USER).cast_mut().cast()
}

pub(crate) fn valid_user(h: Handle) -> bool {
    h == user_ptr()
}

/// Gives the title one more reference to the user.
fn retain_user() {
    USER.refs.fetch_add(1, Ordering::AcqRel);
}

/// Writes `value` through `out`, or answers `E_INVALIDARG` when `out` is null.
///
/// # Safety
/// `out` is null or a writable location for a `T`.
unsafe fn put<T>(out: *mut T, value: T) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: not null, the title's location for a `T`.
    unsafe { out.write_unaligned(value) };
    S_OK
}

/// A getter on the user: `E_INVALIDARG` for a foreign handle, otherwise `value` goes to `out`.
///
/// # Safety
/// `out` is null or a writable location for a `T`.
unsafe fn describe_user<T>(h: Handle, out: *mut T, value: T) -> i32 {
    if !valid_user(h) {
        return E_INVALIDARG;
    }
    // SAFETY: the caller's contract.
    unsafe { put(out, value) }
}

// Adding and finding users

unsafe extern "system" fn XUserAddAsync(options: u32, b: Block) -> i32 {
    log(&format!("XUserAddAsync(options 0x{options:x}, async {b:p}) -> local offline user '{GAMERTAG_TEXT}' (no PC runtime)"));
    let payload = || {
        retain_user();
        Ok((user_ptr() as usize).to_le_bytes().to_vec())
    };
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const USER_ADD_IDENTITY as usize, payload) }
}

unsafe extern "system" fn XUserAddResult(b: Block, out: *mut Handle) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block and a writable handle location of 8 bytes.
    let hr = unsafe { xa::XAsyncGetResult(b, &raw const USER_ADD_IDENTITY as *const c_void, 8, out.cast(), std::ptr::null_mut()) };
    // SAFETY: `out` is the title's handle location, not null.
    let user = if hr >= 0 { unsafe { *out } } else { std::ptr::null_mut() };
    log(&format!("XUserAddResult(async {b:p}) answers 0x{:08x}, user handle {user:p}", hr as u32));
    hr
}

/// Answers the user handle when `id` is the one the user has, counting the new reference.
///
/// # Safety
/// `out` is null or a writable handle location.
unsafe fn hand_out_user(id: u64, expected: u64, out: *mut Handle) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    if id != expected {
        return E_NOT_FOUND;
    }
    retain_user();
    // SAFETY: not null, checked above.
    unsafe { put(out, user_ptr()) }
}

unsafe extern "system" fn XUserFindUserByLocalId(id: u64, out: *mut Handle) -> i32 {
    // SAFETY: the title's location.
    unsafe { hand_out_user(id, LOCAL_ID, out) }
}

unsafe extern "system" fn XUserFindUserById(id: u64, out: *mut Handle) -> i32 {
    // SAFETY: the title's location.
    unsafe { hand_out_user(id, USER_ID, out) }
}

/// Slots 10 and 12 share this: the id may be the local id or the Xbox user id.
unsafe extern "system" fn find_user_by_any_id(id: u64, out: *mut Handle) -> i32 {
    // SAFETY: the title's location.
    unsafe {
        match XUserFindUserByLocalId(id, out) {
            hr if hr >= 0 => hr,
            _ => XUserFindUserById(id, out),
        }
    }
}

/// `XUserAddByIdWithUiAsync`: succeeds for the local user under either id, its Result is `XUserAddResult`.
unsafe extern "system" fn add_by_id_async(id: u64, b: Block) -> i32 {
    let mut probe: Handle = std::ptr::null_mut();
    // SAFETY: a local handle location and the title's block.
    unsafe {
        if find_user_by_any_id(id, &mut probe) < 0 {
            return E_GAMEUSER_USER_NOT_FOUND;
        }
        XUserCloseHandle(probe);
        XUserAddAsync(0, b)
    }
}

// Handles

unsafe extern "system" fn XUserDuplicateHandle(h: Handle, out: *mut Handle) -> i32 {
    if !valid_user(h) || out.is_null() {
        return E_INVALIDARG;
    }
    retain_user();
    // SAFETY: not null, checked above.
    unsafe { put(out, h) }
}

/// A close never drops the count below one, so the static user stays alive.
unsafe extern "system" fn XUserCloseHandle(h: Handle) {
    if valid_user(h) {
        let _ = USER.refs.try_update(Ordering::AcqRel, Ordering::Acquire, |n| Some(n.saturating_sub(1).max(1)));
    }
}

unsafe extern "system" fn XUserCompare(a: Handle, b: Handle) -> i32 {
    (a as usize).cmp(&(b as usize)) as i32
}

// Plain properties

unsafe extern "system" fn XUserGetMaxUsers(out: *mut u32) -> i32 {
    // SAFETY: the title's location.
    unsafe { put(out, 1) }
}

unsafe extern "system" fn XUserGetId(h: Handle, out: *mut u64) -> i32 {
    // SAFETY: the title's location.
    unsafe { describe_user(h, out, USER_ID) }
}

unsafe extern "system" fn XUserGetLocalId(h: Handle, out: *mut u64) -> i32 {
    // SAFETY: the title's location.
    unsafe { describe_user(h, out, LOCAL_ID) }
}

unsafe extern "system" fn XUserGetIsGuest(h: Handle, out: *mut bool) -> i32 {
    // SAFETY: the title's location.
    unsafe { describe_user(h, out, false) }
}

/// `XUserState::SignedIn` is 0.
unsafe extern "system" fn XUserGetState(h: Handle, out: *mut u32) -> i32 {
    // SAFETY: the title's location.
    unsafe { describe_user(h, out, 0) }
}

/// `XUserAgeGroup::Adult` is 3.
unsafe extern "system" fn XUserGetAgeGroup(h: Handle, out: *mut u32) -> i32 {
    // SAFETY: the title's location.
    unsafe { describe_user(h, out, 3) }
}

/// `XUserIsStoreUser`: the local user has no store account.
unsafe extern "system" fn is_store_user_impl(_user: Handle) -> bool {
    false
}

/// `XUserIsSignOutPresent`: nobody is signing out.
unsafe extern "system" fn no() -> bool {
    false
}

// Gamertag

/// The buffer sizes of the gamertag components, terminator included: Classic, Modern,
/// ModernSuffix, UniqueModern.
const GAMERTAG_SIZES: [usize; 4] = [16, 97, 15, 101];
/// `XUserGamertagComponent::ModernSuffix`.
const COMPONENT_MODERN_SUFFIX: u32 = 2;

/// The text of a gamertag component. The user has no modern suffix, the others are the gamertag.
fn gamertag_component(component: u32) -> &'static str {
    if component == COMPONENT_MODERN_SUFFIX { "" } else { GAMERTAG_TEXT }
}

unsafe extern "system" fn XUserGetGamertag(h: Handle, component: u32, size: usize, buf: *mut u8, used: *mut usize) -> i32 {
    if !valid_user(h) || component as usize >= GAMERTAG_SIZES.len() {
        return E_INVALIDARG;
    }
    let text = gamertag_component(component).as_bytes();
    debug_assert!(text.len() < GAMERTAG_SIZES[component as usize]);
    let needed = text.len() + 1;
    // SAFETY: `used` is null or the title's location.
    unsafe {
        if !used.is_null() {
            *used = needed;
        }
    }
    if buf.is_null() || size < needed {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the title's buffer holds `size >= needed` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buf, text.len());
        *buf.add(text.len()) = 0;
    }
    S_OK
}

// Privileges and issues

/// The local user has every privilege, there is no Xbox Live to deny one.
unsafe extern "system" fn XUserCheckPrivilege(_user: Handle, _options: u32, _privilege: u32, has: *mut bool, reasons: *mut u32) -> i32 {
    // SAFETY: the title's locations, null or writable.
    unsafe {
        if has.is_null() {
            return E_INVALIDARG;
        }
        *has = true;
        if !reasons.is_null() {
            *reasons = 0;
        }
    }
    S_OK
}

/// Starts an async call that completes at once without a result.
///
/// # Safety
/// `b` is the title's block.
unsafe fn complete_at_once(b: Block) -> i32 {
    // SAFETY: the caller's contract.
    unsafe { xa::begin_with(b, &raw const EMPTY_ID as usize, || Ok(Vec::new())) }
}

/// The Result of such a call is its status.
unsafe extern "system" fn empty_result(b: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { xa::XAsyncGetStatus(b, false) }
}

unsafe extern "system" fn resolve_privilege_async(_user: Handle, _options: u32, _privilege: u32, b: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { complete_at_once(b) }
}

unsafe extern "system" fn resolve_issue_async(_user: Handle, _url: *const c_void, b: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { complete_at_once(b) }
}

// Change events and sign out

/// Offline the user never changes. A token is handed out and nothing is ever called.
unsafe extern "system" fn XUserRegisterForChangeEvent(_q: *mut c_void, _ctx: *mut c_void, _cb: usize, token: *mut RegistrationToken) -> i32 {
    // SAFETY: the title's location.
    unsafe { put(token, RegistrationToken { token: 1 }) }
}

unsafe extern "system" fn XUserUnregisterForChangeEvent(_token: RegistrationToken, _wait: bool) -> bool {
    true
}

unsafe extern "system" fn sign_out_deferral(out: *mut usize) -> i32 {
    if !out.is_null() {
        // SAFETY: the title's location, not null.
        unsafe { *out = 0 };
    }
    E_GAMEUSER_DEFERRAL_NOT_AVAILABLE
}

unsafe extern "system" fn nothing(_deferral: usize) {}

// Tokens and the gamer picture, which need Xbox Live and so fail

/// Notes a token request in the log (the first 24 and every 64th). There is no token to give.
fn refuse_token(method: &str, url: &str) -> i32 {
    static REQUESTS: AtomicU64 = AtomicU64::new(0);
    let n = REQUESTS.fetch_add(1, Ordering::Relaxed);
    if n < 24 || n.is_multiple_of(64) {
        log(&format!("XUser: token request #{n} for {method} {url} gets none because there is no Xbox Live (0x80072EFD)"));
    }
    E_CANNOT_CONNECT
}

/// Both token calls fail their block with `E_CANNOT_CONNECT`. A foreign user or a null URL is
/// rejected up front.
///
/// # Safety
/// `b` is the title's block.
unsafe fn start_token_request(user: Handle, url_is_null: bool, method: String, url: String, b: Block, identity: *const u8) -> i32 {
    if !valid_user(user) || url_is_null {
        return E_INVALIDARG;
    }
    let refusal = refuse_token(&method, &url);
    // SAFETY: the caller's contract.
    unsafe { xa::begin_with(b, identity as usize, move || Err(refusal)) }
}

/// The method of a request, GET when the title gives none.
fn method_or_get(method: Option<String>) -> String {
    method.filter(|m| !m.is_empty()).unwrap_or_else(|| "GET".into())
}

unsafe extern "system" fn token_async_utf8(
    user: Handle,
    _options: u32,
    method: *const c_void,
    url: *const c_void,
    _header_count: usize,
    _headers: *const c_void,
    _body_size: usize,
    _body: *const c_void,
    b: Block,
) -> i32 {
    // SAFETY: the title's strings, null or zero terminated.
    unsafe {
        let method = method_or_get(narrow(method.cast()));
        start_token_request(user, url.is_null(), method, narrow(url.cast()).unwrap_or_default(), b, &raw const TOKEN_UTF8_ID)
    }
}

unsafe extern "system" fn token_async_utf16(
    user: Handle,
    _options: u32,
    method: *const c_void,
    url: *const c_void,
    _header_count: usize,
    _headers: *const c_void,
    _body_size: usize,
    _body: *const c_void,
    b: Block,
) -> i32 {
    // SAFETY: the title's strings, null or zero terminated.
    unsafe {
        let method = method_or_get(wide(method.cast()));
        start_token_request(user, url.is_null(), method, wide(url.cast()).unwrap_or_default(), b, &raw const TOKEN_UTF16_ID)
    }
}

/// The Result of a token call is the block's error. Nothing is ever written.
///
/// # Safety
/// The title's block, buffer and locations.
unsafe fn token_result(b: Block, identity: *const u8, size: usize, buffer: *mut c_void, ptr_to_buffer: *mut *mut c_void, used: *mut usize) -> i32 {
    // SAFETY: the caller's contract.
    unsafe {
        let hr = xa::XAsyncGetResult(b, identity.cast(), size, buffer, used);
        if hr >= 0 && !ptr_to_buffer.is_null() {
            *ptr_to_buffer = buffer;
        }
        hr
    }
}

unsafe extern "system" fn token_result_utf8(b: Block, size: usize, buffer: *mut c_void, ptr_to_buffer: *mut *mut c_void, used: *mut usize) -> i32 {
    // SAFETY: the title's block, buffer and locations.
    unsafe { token_result(b, &raw const TOKEN_UTF8_ID, size, buffer, ptr_to_buffer, used) }
}

unsafe extern "system" fn token_result_utf16(b: Block, size: usize, buffer: *mut c_void, ptr_to_buffer: *mut *mut c_void, used: *mut usize) -> i32 {
    // SAFETY: the title's block, buffer and locations.
    unsafe { token_result(b, &raw const TOKEN_UTF16_ID, size, buffer, ptr_to_buffer, used) }
}

/// The highest `XUserGamerPictureSize` value.
const PICTURE_SIZE_MAX: u32 = 3;

/// There is no profile to download a picture from, so the block fails with `E_CANNOT_CONNECT`.
unsafe extern "system" fn gamer_picture_async(user: Handle, size: u32, b: Block) -> i32 {
    if !valid_user(user) || size > PICTURE_SIZE_MAX {
        return E_INVALIDARG;
    }
    log("XUserGetGamerPictureAsync has no picture to give because there is no Xbox Live (0x80072EFD)");
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const PICTURE_ID as usize, || Err(E_CANNOT_CONNECT)) }
}

unsafe extern "system" fn gamer_picture_result(b: Block, size: usize, buffer: *mut c_void, used: *mut usize) -> i32 {
    // SAFETY: the title's block, buffer and `used`.
    unsafe { xa::XAsyncGetResult(b, &raw const PICTURE_ID as *const c_void, size, buffer, used) }
}

// Reading the title's strings for the log

/// Longest string read from the title for the log.
const TEXT_MAX: usize = 2048;

/// A zero terminated 8-bit string of the title, cut at [`TEXT_MAX`].
///
/// # Safety
/// `p` is null or a zero terminated string.
unsafe fn narrow(p: *const u8) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller's contract, reads stop at the terminator.
    let bytes: Vec<u8> = (0..TEXT_MAX).map(|i| unsafe { *p.add(i) }).take_while(|&c| c != 0).collect();
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// A zero terminated UTF-16 string of the title, cut at [`TEXT_MAX`].
///
/// # Safety
/// `p` is null or a zero terminated string.
unsafe fn wide(p: *const u16) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: the caller's contract, reads stop at the terminator.
    let units: Vec<u16> = (0..TEXT_MAX).map(|i| unsafe { p.add(i).read_unaligned() }).take_while(|&c| c != 0).collect();
    Some(String::from_utf16_lossy(&units))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xasync::XAsyncGetStatus;

    const THIS: This = &OBJECT;
    const TAG_THIS: *const Object = &GAMERTAG_OBJECT;

    /// Adds the default user through the table and returns the handle.
    unsafe fn add_user() -> Handle {
        // SAFETY: GDK arguments, the block runs on the process queue.
        unsafe {
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.add_async)(THIS, 1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut h: Handle = std::ptr::null_mut();
            assert_eq!((VTBL.add_result)(THIS, &mut b, &mut h), S_OK);
            h
        }
    }

    #[test]
    fn vtables_have_their_sizes_and_objects_point_at_them() {
        assert_eq!(size_of::<Vtbl>(), SLOTS * size_of::<usize>());
        assert_eq!(size_of::<GamertagVtbl>(), GAMERTAG_SLOTS * size_of::<usize>());
        // SAFETY: each object's first field is its vtable pointer, and both tables are function pointers only.
        unsafe {
            assert!(std::ptr::eq(*(object() as *const *const Vtbl), &raw const VTBL));
            assert!(std::ptr::eq(*(gamertag_object() as *const *const GamertagVtbl), &raw const GAMERTAG_VTBL));
            let table = &*((&raw const VTBL).cast::<[usize; SLOTS]>());
            assert_eq!(table[7], add_async as *const () as usize);
            assert_eq!(table[23], get_token_and_signature_async as *const () as usize);
            assert_eq!(table[38], add_by_id_with_ui_result as *const () as usize);
            assert_eq!(table[42], is_store_user as *const () as usize);
            assert_eq!(table[47], is_sign_out_present as *const () as usize);
            assert_eq!(table[49], slot49 as *const () as usize);
        }
    }

    #[test]
    fn the_two_interfaces_are_served_and_nothing_else() {
        let unrelated = Guid { data1: 9, data2: 8, data3: 7, data4: [6; 8] };
        assert_eq!(object_for(Some(&XUSER_TABLE), Some(&XUSER_TABLE)), Some((object(), "XUser")));
        assert_eq!(object_for(Some(&XUSER_TABLE), Some(&XUSER_GAMERTAG)), Some((gamertag_object(), "XUser gamertag")));
        assert_eq!(object_for(Some(&XUSER_TABLE), Some(&unrelated)), None);
        assert_eq!(object_for(Some(&XUSER_TABLE), None), None);
        assert_eq!(object_for(Some(&XUSER_GAMERTAG), Some(&XUSER_GAMERTAG)), None);
        let mut out = std::ptr::dangling_mut::<c_void>();
        // SAFETY: valid GUIDs and out locations.
        unsafe {
            assert_eq!((VTBL.query_interface)(THIS, &XUSER_GAMERTAG, &mut out), S_OK);
            assert_eq!(out, gamertag_object());
            assert_eq!((GAMERTAG_VTBL.query_interface)(TAG_THIS, &XUSER_TABLE, &mut out), S_OK);
            assert_eq!(out, object());
            assert_eq!((VTBL.query_interface)(THIS, &unrelated, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!((VTBL.query_interface)(THIS, &unrelated, std::ptr::null_mut()), E_POINTER);
            assert_eq!((VTBL.slot15)(THIS), E_NOTIMPL);
            assert_eq!((GAMERTAG_VTBL.slot7)(TAG_THIS), E_NOTIMPL);
        }
    }

    #[test]
    fn adding_a_user_gives_the_signed_in_local_user() {
        // SAFETY: GDK arguments.
        unsafe {
            let user = add_user();
            assert_eq!(user, user_ptr());
            assert!(valid_user(user));
            let (mut local, mut xuid) = (0u64, 0u64);
            assert_eq!((VTBL.get_local_id)(THIS, user, &mut local), S_OK);
            assert_eq!((VTBL.get_id)(THIS, user, &mut xuid), S_OK);
            assert_eq!((local, xuid), (1, 0x0009_0000_0000_0001));
            let (mut state, mut age, mut guest, mut max) = (9u32, 9u32, true, 0u32);
            assert_eq!((VTBL.get_state)(THIS, user, &mut state), S_OK);
            assert_eq!((VTBL.get_age_group)(THIS, user, &mut age), S_OK);
            assert_eq!((VTBL.get_is_guest)(THIS, user, &mut guest), S_OK);
            assert_eq!((VTBL.get_max_users)(THIS, &mut max), S_OK);
            assert_eq!((state, age, guest, max), (0, 3, false, 1));
            assert!(!(VTBL.is_store_user)(THIS, user));
            assert!(!(VTBL.is_sign_out_present)(THIS));
            (VTBL.close_handle)(THIS, user);
        }
    }

    #[test]
    fn foreign_handles_and_null_outputs_are_rejected() {
        let foreign = std::ptr::dangling_mut::<c_void>();
        // SAFETY: GDK arguments, the foreign handle is never dereferenced.
        unsafe {
            let (mut state, mut id) = (0u32, 0u64);
            assert_eq!((VTBL.get_state)(THIS, foreign, &mut state), E_INVALIDARG);
            assert_eq!((VTBL.get_id)(THIS, std::ptr::null_mut(), &mut id), E_INVALIDARG);
            assert_eq!((VTBL.get_id)(THIS, user_ptr(), std::ptr::null_mut()), E_INVALIDARG);
            let mut copy: Handle = std::ptr::null_mut();
            assert_eq!((VTBL.duplicate_handle)(THIS, foreign, &mut copy), E_INVALIDARG);
            assert_eq!((VTBL.duplicate_handle)(THIS, user_ptr(), std::ptr::null_mut()), E_INVALIDARG);
            assert_eq!((VTBL.get_max_users)(THIS, std::ptr::null_mut()), E_INVALIDARG);
            (VTBL.close_handle)(THIS, foreign);
            assert_eq!((VTBL.compare)(THIS, user_ptr(), user_ptr()), 0);
        }
    }

    #[test]
    fn users_are_found_by_either_id() {
        // SAFETY: GDK arguments.
        unsafe {
            let mut found: Handle;
            for id in [LOCAL_ID, USER_ID] {
                found = std::ptr::null_mut();
                assert_eq!((VTBL.find_user_by_local_id)(THIS, id, &mut found), S_OK);
                assert_eq!(found, user_ptr());
                (VTBL.close_handle)(THIS, found);
                found = std::ptr::null_mut();
                assert_eq!((VTBL.find_user_by_id)(THIS, id, &mut found), S_OK);
                assert_eq!(found, user_ptr());
                (VTBL.close_handle)(THIS, found);
            }
            found = std::ptr::null_mut();
            assert_eq!((VTBL.find_user_by_id)(THIS, 42, &mut found), E_NOT_FOUND);
            assert!(found.is_null());
            assert_eq!(XUserFindUserByLocalId(USER_ID, &mut found), E_NOT_FOUND);
            assert_eq!(XUserFindUserById(LOCAL_ID, &mut found), E_NOT_FOUND);
            assert_eq!((VTBL.find_user_by_id)(THIS, LOCAL_ID, std::ptr::null_mut()), E_INVALIDARG);
        }
    }

    #[test]
    fn closing_never_frees_the_static_user() {
        // SAFETY: GDK arguments.
        unsafe {
            for _ in 0..8 {
                (VTBL.close_handle)(THIS, user_ptr());
            }
            assert!(USER.refs.load(Ordering::Acquire) >= 1);
            assert_eq!(USER.magic, 0x5553_4552);
            let mut again: Handle = std::ptr::null_mut();
            assert_eq!((VTBL.duplicate_handle)(THIS, user_ptr(), &mut again), S_OK);
            assert_eq!(again, user_ptr());
        }
    }

    #[test]
    fn the_gamertag_fills_components_and_reports_short_buffers() {
        let mut buf = [0xFFu8; 32];
        let mut used = 0usize;
        // SAFETY: GDK arguments.
        unsafe {
            let user = user_ptr();
            for component in [0u32, 1, 3] {
                buf.fill(0xFF);
                assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, user, component, 32, buf.as_mut_ptr(), &mut used), S_OK);
                assert_eq!(&buf[..used], b"Player1\0");
            }
            assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, user, 2, 32, buf.as_mut_ptr(), &mut used), S_OK);
            assert_eq!((used, buf[0]), (1, 0), "no modern suffix");
            assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, user, 0, 3, buf.as_mut_ptr(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            assert_eq!(used, 8);
            assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, user, 0, 0, std::ptr::null_mut(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, user, 4, 32, buf.as_mut_ptr(), &mut used), E_INVALIDARG);
            assert_eq!((GAMERTAG_VTBL.get_gamertag)(TAG_THIS, std::ptr::null_mut(), 0, 32, buf.as_mut_ptr(), &mut used), E_INVALIDARG);
        }
    }

    #[test]
    fn privileges_are_granted_and_resolving_completes() {
        // SAFETY: GDK arguments, the blocks run on the process queue.
        unsafe {
            let user = user_ptr();
            let (mut has, mut reasons) = (false, 9u32);
            assert_eq!((VTBL.check_privilege)(THIS, user, 0, 254, &mut has, &mut reasons), S_OK);
            assert!(has);
            assert_eq!(reasons, 0);
            assert_eq!((VTBL.check_privilege)(THIS, user, 0, 254, std::ptr::null_mut(), &mut reasons), E_INVALIDARG);
            assert_eq!((VTBL.check_privilege)(THIS, user, 0, 254, &mut has, std::ptr::null_mut()), S_OK);

            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.resolve_privilege_with_ui_async)(THIS, user, 0, 254, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!((VTBL.resolve_privilege_with_ui_result)(THIS, &mut b), S_OK);

            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.resolve_issue_with_ui_async)(THIS, user, std::ptr::null(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!((VTBL.resolve_issue_with_ui_result)(THIS, &mut b), S_OK);
        }
    }

    #[test]
    fn tokens_and_pictures_fail_with_cannot_connect() {
        let url = c"https://example.invalid/";
        let wide_url: Vec<u16> = "https://example.invalid/\0".encode_utf16().collect();
        let none = std::ptr::null();
        // SAFETY: GDK arguments, the blocks run on the process queue.
        unsafe {
            let user = user_ptr();
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.get_token_and_signature_async)(THIS, user, 0, c"POST".as_ptr().cast(), url.as_ptr().cast(), 0, none, 0, none, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), E_CANNOT_CONNECT);
            let mut size = 5usize;
            assert_eq!((VTBL.get_token_and_signature_result_size)(THIS, &mut b, &mut size), E_CANNOT_CONNECT);
            let (mut buf, mut ptr) = ([0u8; 64], std::ptr::dangling_mut::<c_void>());
            assert_eq!((VTBL.get_token_and_signature_result)(THIS, &mut b, 64, buf.as_mut_ptr().cast(), &mut ptr, &mut size), E_CANNOT_CONNECT);
            assert_eq!(ptr, std::ptr::dangling_mut::<c_void>(), "no token pointer is written");

            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.get_token_and_signature_utf16_async)(THIS, user, 0, none, wide_url.as_ptr().cast(), 0, none, 0, none, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), E_CANNOT_CONNECT);
            assert_eq!((VTBL.get_token_and_signature_utf16_result)(THIS, &mut b, 64, buf.as_mut_ptr().cast(), &mut ptr, &mut size), E_CANNOT_CONNECT);

            assert_eq!((VTBL.get_token_and_signature_async)(THIS, std::ptr::null_mut(), 0, none, url.as_ptr().cast(), 0, none, 0, none, &mut b), E_INVALIDARG);
            assert_eq!((VTBL.get_token_and_signature_async)(THIS, user, 0, none, none, 0, none, 0, none, &mut b), E_INVALIDARG);

            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.get_gamer_picture_async)(THIS, user, 0, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), E_CANNOT_CONNECT);
            assert_eq!((VTBL.get_gamer_picture_result)(THIS, &mut b, 64, buf.as_mut_ptr().cast(), &mut size), E_CANNOT_CONNECT);
            assert_eq!((VTBL.get_gamer_picture_async)(THIS, user, 4, &mut b), E_INVALIDARG);
        }
    }

    #[test]
    fn sign_out_is_unavailable_and_change_events_hand_out_a_token() {
        // SAFETY: GDK arguments.
        unsafe {
            let mut deferral = 5usize;
            assert_eq!((VTBL.get_sign_out_deferral)(THIS, &mut deferral), E_GAMEUSER_DEFERRAL_NOT_AVAILABLE);
            assert_eq!(deferral, 0);
            (VTBL.close_sign_out_deferral_handle)(THIS, 0);

            let n = std::ptr::null_mut();
            let mut token = RegistrationToken { token: 0 };
            assert_eq!((VTBL.register_for_change_event)(THIS, n, n, 0x10, &mut token), S_OK);
            assert_ne!(token.token, 0);
            assert_eq!((VTBL.register_for_change_event)(THIS, n, n, 0x10, std::ptr::null_mut()), E_INVALIDARG);
            assert!((VTBL.unregister_for_change_event)(THIS, token, true));
        }
    }

    #[test]
    fn add_by_id_accepts_the_local_user_only() {
        // SAFETY: GDK arguments, the block runs on the process queue.
        unsafe {
            let mut b: XAsyncBlock = std::mem::zeroed();
            assert_eq!((VTBL.add_by_id_with_ui_async)(THIS, 99, &mut b), E_GAMEUSER_USER_NOT_FOUND);
            for id in [USER_ID, LOCAL_ID] {
                let mut b: XAsyncBlock = std::mem::zeroed();
                assert_eq!((VTBL.add_by_id_with_ui_async)(THIS, id, &mut b), S_OK);
                assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
                let mut h: Handle = std::ptr::null_mut();
                assert_eq!((VTBL.add_by_id_with_ui_result)(THIS, &mut b, &mut h), S_OK);
                assert_eq!(h, user_ptr());
            }
        }
    }
}
