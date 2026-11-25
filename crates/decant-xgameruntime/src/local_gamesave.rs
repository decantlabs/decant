//! The offline XGameSave table, `api_id = {704C3F58-E629-4CC2-B197-30511B996FE2}`. The same
//! object also answers to `{1BFFF3AF-F14A-40A3-8E35-9ADA906593F9}`, an interface id some titles
//! ask for, with the same slots in the same order. With Gaming Services installed the PC
//! runtime's XGameSave refuses our local user handle (`0x80070006`, "Handle does not exist ...
//! User"), so its save provider never opens. This table opens it for the local user
//! (`local_user`) and for nobody else.
//!
//! Saves live in `%LOCALAPPDATA%\Decant\<title id>\saves\<user id>\<service config id>`, the title
//! id coming from `MicrosoftGame.config` next to the executable (`DECANT_XGS_ROOT` replaces the
//! `saves` folder). Containers, generations and crash safety are the business of
//! `local_gamesave_store`. Each async form completes through our XAsync (`xasync::begin_with`):
//! the work runs on the block's work port and the completion routine on its completion port.
//!
//! The object is a pointer to a vtable in the thunk order of the public PC GDK
//! `xgameruntime.lib` (slots 3 to 32). Every slot takes `this` first and drops it. Slots 0 to 2 are
//! IUnknown on a static object. Slots 33 to 39 log their index and answer `E_NOTIMPL`. Every call
//! is logged (`calllog`), the first [`crate::calllog::FIRST`] of each slot in full.
//!
//! Where the public documentation leaves room, the answer is the one that lets a title reach its
//! menus: a container that does not exist is no error for `GetContainerInfo` (the callback just
//! does not run) and for `DeleteContainer`. The quota is 256 MiB less what the containers hold.
//! Nothing ever needs a sync.
#![allow(unsafe_code, non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::E_NOTIMPL;
use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::local_title::put;
use crate::local_gamesave_store::{self as store, BUFFER_TOO_SMALL, Change, ContainerSummary, INVALID_CONTAINER_NAME, Store};
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`, also the `riid` of its main interface.
pub(crate) const XGAMESAVE_TABLE: Guid = Guid { data1: 0x704C3F58, data2: 0xE629, data3: 0x4CC2, data4: [0xB1, 0x97, 0x30, 0x51, 0x1B, 0x99, 0x6F, 0xE2] };
/// The second interface id the table answers to.
pub(crate) const XGAMESAVE_ALIAS: Guid = Guid { data1: 0x1BFFF3AF, data2: 0xF14A, data3: 0x40A3, data4: [0x8E, 0x35, 0x9A, 0xDA, 0x90, 0x65, 0x93, 0xF9] };
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;
const E_NOT_SUFFICIENT_BUFFER: i32 = 0x8007_007A_u32 as i32;

/// The object for a couple, with the table's name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    if api != Some(&XGAMESAVE_TABLE) {
        return None;
    }
    match riid {
        Some(r) if *r == XGAMESAVE_TABLE || *r == XGAMESAVE_ALIAS => Some((object(), "XGameSave")),
        _ => None,
    }
}

const SLOTS: usize = 40;
const MAIN: &str = "XGameSave";
static COUNTS: Counters<SLOTS> = Counters::new();

type This = *const Object;
type Block = *mut XAsyncBlock;
type H = *mut c_void;
type Text = *const u8;
type Out = *mut H;

/// The vtable, in slot order.
#[repr(C)]
pub(crate) struct Vtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    initialize_provider: unsafe extern "system" fn(This, H, Text, u8, Out) -> i32,
    initialize_provider_async: unsafe extern "system" fn(This, H, Text, u8, Block) -> i32,
    initialize_provider_result: unsafe extern "system" fn(This, Block, Out) -> i32,
    close_provider: unsafe extern "system" fn(This, H),
    get_remaining_quota: unsafe extern "system" fn(This, H, *mut i64) -> i32,
    get_remaining_quota_async: unsafe extern "system" fn(This, H, Block) -> i32,
    get_remaining_quota_result: unsafe extern "system" fn(This, Block, *mut i64) -> i32,
    delete_container: unsafe extern "system" fn(This, H, Text) -> i32,
    delete_container_async: unsafe extern "system" fn(This, H, Text, Block) -> i32,
    delete_container_result: unsafe extern "system" fn(This, Block) -> i32,
    get_container_info: unsafe extern "system" fn(This, H, Text, *mut c_void, usize) -> i32,
    enumerate_container_info: unsafe extern "system" fn(This, H, *mut c_void, usize) -> i32,
    enumerate_container_info_by_name: unsafe extern "system" fn(This, H, Text, *mut c_void, usize) -> i32,
    create_container: unsafe extern "system" fn(This, H, Text, Out) -> i32,
    close_container: unsafe extern "system" fn(This, H),
    enumerate_blob_info: unsafe extern "system" fn(This, H, *mut c_void, usize) -> i32,
    enumerate_blob_info_by_name: unsafe extern "system" fn(This, H, Text, *mut c_void, usize) -> i32,
    read_blob_data: unsafe extern "system" fn(This, H, *const Text, *mut u32, usize, *mut u8) -> i32,
    read_blob_data_async: unsafe extern "system" fn(This, H, *const Text, u32, Block) -> i32,
    read_blob_data_result: unsafe extern "system" fn(This, Block, usize, *mut u8, *mut u32) -> i32,
    create_update: unsafe extern "system" fn(This, H, Text, Out) -> i32,
    close_update: unsafe extern "system" fn(This, H),
    submit_blob_write: unsafe extern "system" fn(This, H, Text, *const u8, usize) -> i32,
    submit_blob_delete: unsafe extern "system" fn(This, H, Text) -> i32,
    submit_update: unsafe extern "system" fn(This, H) -> i32,
    submit_update_async: unsafe extern "system" fn(This, H, Block) -> i32,
    submit_update_result: unsafe extern "system" fn(This, Block) -> i32,
    files_get_folder_with_ui_async: unsafe extern "system" fn(This, H, Text, Block) -> i32,
    files_get_folder_with_ui_result: unsafe extern "system" fn(This, Block, usize, *mut u8) -> i32,
    files_get_remaining_quota: unsafe extern "system" fn(This, H, Text, *mut i64) -> i32,
    slot33: unsafe extern "system" fn(This) -> i32,
    slot34: unsafe extern "system" fn(This) -> i32,
    slot35: unsafe extern "system" fn(This) -> i32,
    slot36: unsafe extern "system" fn(This) -> i32,
    slot37: unsafe extern "system" fn(This) -> i32,
    slot38: unsafe extern "system" fn(This) -> i32,
    slot39: unsafe extern "system" fn(This) -> i32,
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

/// The C++ `bool` arguments (`syncOnDemand`), taken as their low byte.
impl crate::calllog::Arg for u8 {
    fn text(&self) -> String {
        self.to_string()
    }
}

macro_rules! shim {
    ($slot:literal as $gdk:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot as $gdk, $($rest)*);
    };
    ($slot:literal, $($rest:tt)*) => {
        logged_shim!(MAIN, COUNTS, This, $slot, $($rest)*);
    };
}

shim!(3, initialize_provider => XGameSaveInitializeProvider, (user: H, config: Text, sync_on_demand: u8, provider: Out) -> i32);
shim!(4, initialize_provider_async => XGameSaveInitializeProviderAsync, (user: H, config: Text, sync_on_demand: u8, async_block: Block) -> i32);
shim!(5, initialize_provider_result => XGameSaveInitializeProviderResult, (async_block: Block, provider: Out) -> i32);
shim!(6, close_provider => XGameSaveCloseProvider, (provider: H));
shim!(7, get_remaining_quota => XGameSaveGetRemainingQuota, (provider: H, quota: *mut i64) -> i32);
shim!(8, get_remaining_quota_async => XGameSaveGetRemainingQuotaAsync, (provider: H, async_block: Block) -> i32);
shim!(9, get_remaining_quota_result => XGameSaveGetRemainingQuotaResult, (async_block: Block, quota: *mut i64) -> i32);
shim!(10, delete_container => XGameSaveDeleteContainer, (provider: H, name: Text) -> i32);
shim!(11, delete_container_async => XGameSaveDeleteContainerAsync, (provider: H, name: Text, async_block: Block) -> i32);
shim!(12, delete_container_result => XGameSaveDeleteContainerResult, (async_block: Block) -> i32);
shim!(13, get_container_info => XGameSaveGetContainerInfo, (provider: H, name: Text, context: *mut c_void, callback: usize) -> i32);
shim!(14, enumerate_container_info => XGameSaveEnumerateContainerInfo, (provider: H, context: *mut c_void, callback: usize) -> i32);
shim!(15, enumerate_container_info_by_name => XGameSaveEnumerateContainerInfoByName, (provider: H, prefix: Text, context: *mut c_void, callback: usize) -> i32);
shim!(16, create_container => XGameSaveCreateContainer, (provider: H, name: Text, container: Out) -> i32);
shim!(17, close_container => XGameSaveCloseContainer, (container: H));
shim!(18, enumerate_blob_info => XGameSaveEnumerateBlobInfo, (container: H, context: *mut c_void, callback: usize) -> i32);
shim!(19, enumerate_blob_info_by_name => XGameSaveEnumerateBlobInfoByName, (container: H, prefix: Text, context: *mut c_void, callback: usize) -> i32);
shim!(20, read_blob_data => XGameSaveReadBlobData, (container: H, names: *const Text, count: *mut u32, size: usize, blobs: *mut u8) -> i32);
shim!(21, read_blob_data_async => XGameSaveReadBlobDataAsync, (container: H, names: *const Text, count: u32, async_block: Block) -> i32);
shim!(22, read_blob_data_result => XGameSaveReadBlobDataResult, (async_block: Block, size: usize, blobs: *mut u8, count: *mut u32) -> i32);
shim!(23, create_update => XGameSaveCreateUpdate, (container: H, display_name: Text, update: Out) -> i32);
shim!(24, close_update => XGameSaveCloseUpdate, (update: H));
shim!(25, submit_blob_write => XGameSaveSubmitBlobWrite, (update: H, name: Text, data: *const u8, size: usize) -> i32);
shim!(26, submit_blob_delete => XGameSaveSubmitBlobDelete, (update: H, name: Text) -> i32);
shim!(27, submit_update => XGameSaveSubmitUpdate, (update: H) -> i32);
shim!(28, submit_update_async => XGameSaveSubmitUpdateAsync, (update: H, async_block: Block) -> i32);
shim!(29, submit_update_result => XGameSaveSubmitUpdateResult, (async_block: Block) -> i32);
shim!(30, files_get_folder_with_ui_async => XGameSaveFilesGetFolderWithUiAsync, (user: H, config: Text, async_block: Block) -> i32);
shim!(31, files_get_folder_with_ui_result => XGameSaveFilesGetFolderWithUiResult, (async_block: Block, size: usize, folder: *mut u8) -> i32);
shim!(32, files_get_remaining_quota => XGameSaveFilesGetRemainingQuota, (user: H, config: Text, quota: *mut i64) -> i32);

/// Declares unknown slots of the table: each logs its index (bounded) and answers `E_NOTIMPL`.
macro_rules! unknown {
    ($($slot:literal $name:ident),*) => {
        $(
            unsafe extern "system" fn $name(_this: This) -> i32 {
                if let Some(n) = COUNTS.next($slot) {
                    log(&format!("local {MAIN} table: slot {} is not known -> E_NOTIMPL [call {n}]", $slot));
                }
                E_NOTIMPL
            }
        )*
    };
}

unknown!(33 slot33, 34 slot34, 35 slot35, 36 slot36, 37 slot37, 38 slot38, 39 slot39);

static VTBL: Vtbl = Vtbl {
    query_interface,
    add_ref: no_count,
    release: no_count,
    initialize_provider,
    initialize_provider_async,
    initialize_provider_result,
    close_provider,
    get_remaining_quota,
    get_remaining_quota_async,
    get_remaining_quota_result,
    delete_container,
    delete_container_async,
    delete_container_result,
    get_container_info,
    enumerate_container_info,
    enumerate_container_info_by_name,
    create_container,
    close_container,
    enumerate_blob_info,
    enumerate_blob_info_by_name,
    read_blob_data,
    read_blob_data_async,
    read_blob_data_result,
    create_update,
    close_update,
    submit_blob_write,
    submit_blob_delete,
    submit_update,
    submit_update_async,
    submit_update_result,
    files_get_folder_with_ui_async,
    files_get_folder_with_ui_result,
    files_get_remaining_quota,
    slot33,
    slot34,
    slot35,
    slot36,
    slot37,
    slot38,
    slot39,
};

/// Both interface ids answer the one object.
unsafe extern "system" fn query_interface(_this: This, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract: `iid` is null or a GUID, `out` a writable pointer location.
    unsafe {
        let iid = if iid.is_null() { None } else { Some(iid.read_unaligned()) };
        if iid == Some(XGAMESAVE_TABLE) || iid == Some(XGAMESAVE_ALIAS) {
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

// The structures of the public API.

/// `XGameSaveContainerInfo`.
#[repr(C)]
pub(crate) struct ContainerInfo {
    pub(crate) name: *const u8,
    pub(crate) display_name: *const u8,
    pub(crate) blob_count: u32,
    pub(crate) total_size: u64,
    /// `time_t`.
    pub(crate) last_modified: i64,
    pub(crate) needs_sync: bool,
}

/// `XGameSaveBlobInfo`.
#[repr(C)]
pub(crate) struct BlobInfo {
    pub(crate) name: *const u8,
    pub(crate) size: u32,
}

/// `XGameSaveBlob`: the info, then the data pointer.
const BLOB: usize = 24;

/// The callbacks return a C++ `bool`: only its low byte is read.
type ContainerCallback = unsafe extern "system" fn(*const ContainerInfo, *mut c_void) -> u8;
type BlobCallback = unsafe extern "system" fn(*const BlobInfo, *mut c_void) -> u8;

// Handles. A handle is the address of a small allocation of ours and stands for an `Object` in
// the registry. A handle the title invented or already closed is not in the registry and is
// refused instead of dereferenced.

struct Provider {
    store: Store,
}

struct Container {
    provider: Arc<Provider>,
    name: String,
}

struct Update {
    container: Arc<Container>,
    display: Option<String>,
    changes: Mutex<Vec<Change>>,
}

#[derive(Clone)]
enum Kept {
    Provider(Arc<Provider>),
    Container(Arc<Container>),
    Update(Arc<Update>),
}

/// What a handle can stand for: lets [`lookup`] pick the right kind and refuse the others.
trait Handled: Sized {
    fn wrap(this: Arc<Self>) -> Kept;
    fn unwrap(kept: &Kept) -> Option<Arc<Self>>;
}

macro_rules! handled {
    ($kind:ident) => {
        impl Handled for $kind {
            fn wrap(this: Arc<Self>) -> Kept {
                Kept::$kind(this)
            }
            fn unwrap(kept: &Kept) -> Option<Arc<Self>> {
                match kept {
                    Kept::$kind(inner) => Some(inner.clone()),
                    _ => None,
                }
            }
        }
    };
}

handled!(Provider);
handled!(Container);
handled!(Update);

fn registry() -> std::sync::MutexGuard<'static, HashMap<usize, Kept>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Kept>>> = OnceLock::new();
    REGISTRY.get_or_init(Mutex::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Registers `thing` and returns the new handle.
fn issue<T: Handled>(thing: T) -> H {
    let address = Box::into_raw(Box::new(0u64)) as usize;
    registry().insert(address, T::wrap(Arc::new(thing)));
    address as H
}

/// Forgets handle `h` and frees its allocation. An unknown handle is ignored.
fn release(h: H) {
    if registry().remove(&(h as usize)).is_some() {
        // SAFETY: only `issue` puts addresses in the registry, each a leaked `Box<u64>`, and the
        // removal above makes this the single release.
        drop(unsafe { Box::from_raw(h as *mut u64) });
    }
}

/// What handle `h` stands for, when it is a live handle of kind `T`.
fn lookup<T: Handled>(h: H) -> Option<Arc<T>> {
    registry().get(&(h as usize)).and_then(T::unwrap)
}

/// The title's zero terminated UTF-8 text, `None` for a null pointer or invalid UTF-8.
fn text(p: Text) -> Option<String> {
    crate::local_title::text(p.cast())
}

fn result_of(r: Result<(), i32>) -> i32 {
    r.err().unwrap_or(S_OK)
}

// Where the saves live.

/// The folder all users' saves live in: `DECANT_XGS_ROOT`, else
/// `%LOCALAPPDATA%\Decant\<title id>\saves`.
fn saves_root() -> PathBuf {
    #[cfg(test)]
    return tests::root();
    #[cfg(not(test))]
    {
        static ROOT: OnceLock<PathBuf> = OnceLock::new();
        ROOT.get_or_init(|| {
            let root = std::env::var_os("DECANT_XGS_ROOT").map(PathBuf::from).unwrap_or_else(|| {
                let base = std::env::var_os("LOCALAPPDATA").map_or_else(std::env::temp_dir, PathBuf::from);
                base.join("Decant").join(crate::local_title::package_identifier()).join("saves")
            });
            log(&format!("XGameSave: saves live in {}", root.display()));
            root
        })
        .clone()
    }
}

/// The folder of the local user.
fn user_root() -> PathBuf {
    saves_root().join(format!("{:016X}", crate::local_user::USER_ID))
}

/// The service configuration id as a folder name; none or empty is `default`.
fn config_folder(config: Option<&str>) -> String {
    match config {
        Some(c) if !c.is_empty() => store::safe_name(c, 64),
        _ => "default".to_string(),
    }
}

/// Only the local user's handle opens saves: any other is refused.
fn check_user(user: H, call: &str) -> Result<(), i32> {
    if crate::local_user::valid_user(user) {
        Ok(())
    } else {
        log(&format!("XGameSave: {call} with user handle {user:p}, which is not the local user -> E_INVALIDARG"));
        Err(E_INVALIDARG)
    }
}

fn open_provider(config: Option<String>) -> Result<Provider, i32> {
    let folder = config_folder(config.as_deref());
    let store = Store::open(user_root().join(&folder))?;
    log(&format!("XGameSave: provider for configuration '{}' opened in {}", config.unwrap_or_default(), store.root().display()));
    Ok(Provider { store })
}

// The functions, with the public API's arguments. Each relies on the title passing what the API
// says: strings zero terminated, buffers as large as the size given, handles from this table.

static INIT_ID: u8 = 0;
static QUOTA_ID: u8 = 0;
static DELETE_ID: u8 = 0;
static READ_ID: u8 = 0;
static SUBMIT_ID: u8 = 0;
static FOLDER_ID: u8 = 0;

unsafe extern "system" fn XGameSaveInitializeProvider(user: H, config: Text, _sync_on_demand: u8, out: Out) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    if let Err(hr) = check_user(user, "XGameSaveInitializeProvider") {
        return hr;
    }
    match open_provider(text(config)) {
        Ok(p) => put(out, issue(p)),
        Err(hr) => hr,
    }
}

unsafe extern "system" fn XGameSaveInitializeProviderAsync(user: H, config: Text, _sync_on_demand: u8, b: Block) -> i32 {
    if b.is_null() {
        return E_INVALIDARG;
    }
    if let Err(hr) = check_user(user, "XGameSaveInitializeProviderAsync") {
        return hr;
    }
    let config = text(config);
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const INIT_ID as usize, move || open_provider(config).map(|p| (issue(p) as usize).to_le_bytes().to_vec())) }
}

unsafe extern "system" fn XGameSaveInitializeProviderResult(b: Block, out: Out) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block and a writable handle location of 8 bytes.
    unsafe { xa::XAsyncGetResult(b, &raw const INIT_ID as *const c_void, size_of::<H>(), out.cast(), std::ptr::null_mut()) }
}

unsafe extern "system" fn XGameSaveCloseProvider(p: H) {
    release(p);
}

unsafe extern "system" fn XGameSaveGetRemainingQuota(p: H, quota: *mut i64) -> i32 {
    let Some(p) = lookup::<Provider>(p) else { return E_INVALIDARG };
    put(quota, p.store.remaining() as i64)
}

unsafe extern "system" fn XGameSaveGetRemainingQuotaAsync(p: H, b: Block) -> i32 {
    let Some(p) = lookup::<Provider>(p) else { return E_INVALIDARG };
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const QUOTA_ID as usize, move || Ok((p.store.remaining() as i64).to_le_bytes().to_vec())) }
}

unsafe extern "system" fn XGameSaveGetRemainingQuotaResult(b: Block, quota: *mut i64) -> i32 {
    if quota.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block and an 8 byte location.
    unsafe { xa::XAsyncGetResult(b, &raw const QUOTA_ID as *const c_void, size_of::<i64>(), quota.cast(), std::ptr::null_mut()) }
}

/// The provider and the container name of a call, checked.
///
/// # Safety
/// `name` is null or a zero terminated string.
unsafe fn provider_and_name(p: H, name: Text) -> Result<(Arc<Provider>, String), i32> {
    let p = lookup::<Provider>(p).ok_or(E_INVALIDARG)?;
    let name = text(name).ok_or(E_INVALIDARG)?;
    if !store::valid_name(&name) {
        return Err(INVALID_CONTAINER_NAME);
    }
    Ok((p, name))
}

unsafe extern "system" fn XGameSaveDeleteContainer(p: H, name: Text) -> i32 {
    // SAFETY: the title's string.
    match unsafe { provider_and_name(p, name) } {
        Ok((p, name)) => result_of(p.store.delete(&name)),
        Err(hr) => hr,
    }
}

unsafe extern "system" fn XGameSaveDeleteContainerAsync(p: H, name: Text, b: Block) -> i32 {
    // SAFETY: the title's string.
    let (p, name) = match unsafe { provider_and_name(p, name) } {
        Ok(v) => v,
        Err(hr) => return hr,
    };
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const DELETE_ID as usize, move || p.store.delete(&name).map(|()| Vec::new())) }
}

unsafe extern "system" fn XGameSaveDeleteContainerResult(b: Block) -> i32 {
    // SAFETY: the title's block; the call has no payload.
    unsafe { xa::XAsyncGetResult(b, &raw const DELETE_ID as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) }
}

/// Hands each summary to the title's container callback until it answers false. Never called
/// under a lock of ours, because the callback may call this table again.
///
/// # Safety
/// `callback` is the title's `XGameSaveContainerInfoCallback`.
unsafe fn report_containers(list: Vec<ContainerSummary>, context: *mut c_void, callback: usize) {
    // SAFETY: the caller's contract.
    let callback = unsafe { std::mem::transmute::<usize, ContainerCallback>(callback) };
    for c in list {
        let name = cstring(&c.name);
        let display = cstring(&c.display);
        let info = ContainerInfo { name: name.as_ptr().cast(), display_name: display.as_ptr().cast(), blob_count: c.blob_count, total_size: c.total_size, last_modified: c.modified, needs_sync: false };
        // SAFETY: the title's callback, with an info that lives for the call.
        if unsafe { callback(&info, context) } == 0 {
            break;
        }
    }
}

fn cstring(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s.replace('\0', "")).unwrap_or_default()
}

fn starts_with_ignoring_case(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

unsafe extern "system" fn XGameSaveGetContainerInfo(p: H, name: Text, context: *mut c_void, callback: usize) -> i32 {
    if callback == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: the title's string.
    let (p, name) = match unsafe { provider_and_name(p, name) } {
        Ok(v) => v,
        Err(hr) => return hr,
    };
    match p.store.container(&name) {
        // SAFETY: the title's callback.
        Ok(found) => unsafe { report_containers(found.into_iter().collect(), context, callback) },
        Err(hr) => return hr,
    }
    S_OK
}

unsafe extern "system" fn XGameSaveEnumerateContainerInfo(p: H, context: *mut c_void, callback: usize) -> i32 {
    let Some(p) = lookup::<Provider>(p) else { return E_INVALIDARG };
    if callback == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: the title's callback.
    unsafe { report_containers(p.store.containers(), context, callback) };
    S_OK
}

unsafe extern "system" fn XGameSaveEnumerateContainerInfoByName(p: H, prefix: Text, context: *mut c_void, callback: usize) -> i32 {
    let Some(p) = lookup::<Provider>(p) else { return E_INVALIDARG };
    if callback == 0 {
        return E_INVALIDARG;
    }
    let prefix = text(prefix).unwrap_or_default();
    let list = p.store.containers().into_iter().filter(|c| starts_with_ignoring_case(&c.name, &prefix)).collect();
    // SAFETY: the title's callback.
    unsafe { report_containers(list, context, callback) };
    S_OK
}

/// A container handle needs no folder: the container exists once an update to it is submitted.
unsafe extern "system" fn XGameSaveCreateContainer(p: H, name: Text, out: Out) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's string.
    match unsafe { provider_and_name(p, name) } {
        Ok((provider, name)) => put(out, issue(Container { provider, name })),
        Err(hr) => hr,
    }
}

unsafe extern "system" fn XGameSaveCloseContainer(c: H) {
    release(c);
}

/// # Safety
/// `callback` is the title's `XGameSaveBlobInfoCallback`, `prefix` null or a string.
unsafe fn report_blobs(c: H, prefix: Text, context: *mut c_void, callback: usize) -> i32 {
    let Some(c) = lookup::<Container>(c) else { return E_INVALIDARG };
    if callback == 0 {
        return E_INVALIDARG;
    }
    let prefix = text(prefix).unwrap_or_default();
    let blobs = match c.provider.store.blobs(&c.name) {
        Ok(b) => b,
        Err(hr) => return hr,
    };
    // SAFETY: the caller's contract.
    let callback = unsafe { std::mem::transmute::<usize, BlobCallback>(callback) };
    for (name, size) in blobs.into_iter().filter(|(n, _)| starts_with_ignoring_case(n, &prefix)) {
        let name = cstring(&name);
        let info = BlobInfo { name: name.as_ptr().cast(), size: size as u32 };
        // SAFETY: the title's callback, with an info that lives for the call.
        if unsafe { callback(&info, context) } == 0 {
            break;
        }
    }
    S_OK
}

unsafe extern "system" fn XGameSaveEnumerateBlobInfo(c: H, context: *mut c_void, callback: usize) -> i32 {
    // SAFETY: the title's callback.
    unsafe { report_blobs(c, std::ptr::null(), context, callback) }
}

unsafe extern "system" fn XGameSaveEnumerateBlobInfoByName(c: H, prefix: Text, context: *mut c_void, callback: usize) -> i32 {
    // SAFETY: the title's string and callback.
    unsafe { report_blobs(c, prefix, context, callback) }
}

/// The blob names a title asks for: `count` strings, or every blob for a null array or a count
/// of 0.
///
/// # Safety
/// `names` is null or holds `count` strings.
unsafe fn wanted(names: *const Text, count: u32) -> Result<Option<Vec<String>>, i32> {
    if names.is_null() || count == 0 {
        return Ok(None);
    }
    // SAFETY: the caller's contract.
    (0..count as usize).map(|i| unsafe { text(*names.add(i)) }.ok_or(E_INVALIDARG)).collect::<Result<Vec<_>, _>>().map(Some)
}

/// Byte offsets inside one `XGameSaveBlob` entry: the name pointer, the size, the data pointer.
const BLOB_NAME_AT: usize = 0;
const BLOB_SIZE_AT: usize = 8;
const BLOB_DATA_AT: usize = 16;

/// The buffer `XGameSaveReadBlobData` fills: one entry per blob, then each blob's name (zero
/// terminated) and bytes. The pointer fields hold offsets from the buffer start until
/// [`bind_pointers`] makes them addresses, so the buffer can be built before its final place is known.
fn pack_blobs(blobs: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut buf = vec![0u8; blobs.len() * BLOB];
    for (n, (name, data)) in blobs.iter().enumerate() {
        let name_offset = buf.len() as u64;
        buf.extend_from_slice(name.as_bytes());
        buf.push(0);
        let data_offset = buf.len() as u64;
        buf.extend_from_slice(data);
        let entry = n * BLOB;
        buf[entry + BLOB_NAME_AT..][..8].copy_from_slice(&name_offset.to_le_bytes());
        buf[entry + BLOB_SIZE_AT..][..4].copy_from_slice(&(data.len() as u32).to_le_bytes());
        buf[entry + BLOB_DATA_AT..][..8].copy_from_slice(&data_offset.to_le_bytes());
    }
    buf
}

/// Adds the address of `buf` to the two pointer fields of each of the first `count` entries.
///
/// # Safety
/// `buf` holds a [`pack_blobs`] buffer of `count` blobs.
unsafe fn bind_pointers(buf: *mut u8, count: usize) {
    for n in 0..count {
        for field in [BLOB_NAME_AT, BLOB_DATA_AT] {
            // SAFETY: the caller's contract puts both fields inside the entry array.
            unsafe {
                let slot = buf.add(n * BLOB + field).cast::<usize>();
                slot.write_unaligned(slot.read_unaligned().wrapping_add(buf as usize));
            }
        }
    }
}

unsafe extern "system" fn XGameSaveReadBlobData(c: H, names: *const Text, count: *mut u32, size: usize, out: *mut u8) -> i32 {
    let Some(c) = lookup::<Container>(c) else { return E_INVALIDARG };
    if count.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's count and names.
    let wanted = match unsafe { wanted(names, *count) } {
        Ok(w) => w,
        Err(hr) => return hr,
    };
    let blobs = match c.provider.store.read(&c.name, wanted.as_deref()) {
        Ok(b) => b,
        Err(hr) => return hr,
    };
    let bytes = pack_blobs(&blobs);
    if out.is_null() || size < bytes.len() {
        return BUFFER_TOO_SMALL;
    }
    // SAFETY: the title's buffer holds `size` bytes, checked above.
    unsafe {
        out.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        bind_pointers(out, blobs.len());
        *count = blobs.len() as u32;
    }
    S_OK
}

unsafe extern "system" fn XGameSaveReadBlobDataAsync(c: H, names: *const Text, count: u32, b: Block) -> i32 {
    let Some(c) = lookup::<Container>(c) else { return E_INVALIDARG };
    // SAFETY: the title's names, copied before the call returns.
    let wanted = match unsafe { wanted(names, count) } {
        Ok(w) => w,
        Err(hr) => return hr,
    };
    // SAFETY: the title's block.
    unsafe { xa::begin_with(b, &raw const READ_ID as usize, move || c.provider.store.read(&c.name, wanted.as_deref()).map(|blobs| pack_blobs(&blobs))) }
}

/// The result keeps its exact size (what `XAsyncGetResultSize` answers is the buffer to give):
/// the count is not stored, it is the first name's offset divided by the entry size.
unsafe extern "system" fn XGameSaveReadBlobDataResult(b: Block, size: usize, out: *mut u8, count: *mut u32) -> i32 {
    let mut used = 0usize;
    // SAFETY: the title's block and buffer. A small buffer leaves the result for a retry.
    let hr = unsafe { xa::XAsyncGetResult(b, &raw const READ_ID as *const c_void, size, out.cast(), &mut used) };
    if hr == E_NOT_SUFFICIENT_BUFFER {
        return BUFFER_TOO_SMALL;
    }
    if hr < 0 {
        return hr;
    }
    // SAFETY: the result just written: an entry array whose first name follows it.
    let blobs = if used == 0 { 0 } else { unsafe { out.cast::<u64>().read_unaligned() as usize / BLOB } };
    // SAFETY: as above.
    unsafe { bind_pointers(out, blobs) };
    if !count.is_null() {
        // SAFETY: the title's location.
        unsafe { *count = blobs as u32 };
    }
    S_OK
}

unsafe extern "system" fn XGameSaveCreateUpdate(c: H, display: Text, out: Out) -> i32 {
    let Some(container) = lookup::<Container>(c) else { return E_INVALIDARG };
    let display = text(display);
    put(out, issue(Update { container, display, changes: Mutex::new(Vec::new()) }))
}

unsafe extern "system" fn XGameSaveCloseUpdate(u: H) {
    release(u);
}

/// # Safety
/// `name` is null or a string.
unsafe fn add_change(u: H, name: Text, data: Option<Vec<u8>>) -> i32 {
    let Some(u) = lookup::<Update>(u) else { return E_INVALIDARG };
    let Some(name) = text(name).filter(|n| store::valid_name(n)) else { return E_INVALIDARG };
    u.changes.lock().unwrap_or_else(|e| e.into_inner()).push((name, data));
    S_OK
}

unsafe extern "system" fn XGameSaveSubmitBlobWrite(u: H, name: Text, data: *const u8, size: usize) -> i32 {
    if data.is_null() && size > 0 {
        return E_INVALIDARG;
    }
    if size as u64 > store::MAX_UPDATE {
        return store::UPDATE_TOO_BIG;
    }
    // SAFETY: the title passes `size` bytes.
    let bytes = if size == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(data, size) }.to_vec() };
    // SAFETY: the title's string.
    unsafe { add_change(u, name, Some(bytes)) }
}

unsafe extern "system" fn XGameSaveSubmitBlobDelete(u: H, name: Text) -> i32 {
    // SAFETY: the title's string.
    unsafe { add_change(u, name, None) }
}

fn submit(u: &Update) -> Result<(), i32> {
    let changes = u.changes.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let c = &u.container;
    c.provider.store.submit(&c.name, u.display.as_deref(), &changes)
}

unsafe extern "system" fn XGameSaveSubmitUpdate(u: H) -> i32 {
    lookup::<Update>(u).map_or(E_INVALIDARG, |u| result_of(submit(&u)))
}

unsafe extern "system" fn XGameSaveSubmitUpdateAsync(u: H, b: Block) -> i32 {
    let Some(u) = lookup::<Update>(u) else { return E_INVALIDARG };
    // SAFETY: the title's block. The update's changes are taken as they are when the work runs;
    // the title may not touch the update while the call is pending.
    unsafe { xa::begin_with(b, &raw const SUBMIT_ID as usize, move || submit(&u).map(|()| Vec::new())) }
}

unsafe extern "system" fn XGameSaveSubmitUpdateResult(b: Block) -> i32 {
    // SAFETY: the title's block; the call has no payload.
    unsafe { xa::XAsyncGetResult(b, &raw const SUBMIT_ID as *const c_void, 0, std::ptr::null_mut(), std::ptr::null_mut()) }
}

/// The folder of the save files API for a configuration: beside the containers' folders, under a
/// name no configuration folder can have (`safe_name` never keeps a `~`).
fn files_folder(config: Option<&str>) -> PathBuf {
    user_root().join("files~").join(config_folder(config))
}

/// No UI: the folder is made and handed out at once.
unsafe extern "system" fn XGameSaveFilesGetFolderWithUiAsync(user: H, config: Text, b: Block) -> i32 {
    if let Err(hr) = check_user(user, "XGameSaveFilesGetFolderWithUiAsync") {
        return hr;
    }
    let config = text(config);
    // SAFETY: the title's block.
    unsafe {
        xa::begin_with(b, &raw const FOLDER_ID as usize, move || {
            let dir = files_folder(config.as_deref());
            std::fs::create_dir_all(&dir).map_err(|_| store::NO_ACCESS)?;
            log(&format!("XGameSave: save files folder {}", dir.display()));
            let mut bytes = dir.to_string_lossy().into_owned().into_bytes();
            bytes.push(0);
            Ok(bytes)
        })
    }
}

unsafe extern "system" fn XGameSaveFilesGetFolderWithUiResult(b: Block, size: usize, folder: *mut u8) -> i32 {
    // SAFETY: the title's block and a buffer of `size` bytes.
    let hr = unsafe { xa::XAsyncGetResult(b, &raw const FOLDER_ID as *const c_void, size, folder.cast(), std::ptr::null_mut()) };
    if hr == E_NOT_SUFFICIENT_BUFFER { BUFFER_TOO_SMALL } else { hr }
}

unsafe extern "system" fn XGameSaveFilesGetRemainingQuota(user: H, config: Text, quota: *mut i64) -> i32 {
    if let Err(hr) = check_user(user, "XGameSaveFilesGetRemainingQuota") {
        return hr;
    }
    let dir = files_folder(text(config).as_deref());
    put(quota, store::QUOTA.saturating_sub(store::folder_bytes(&dir)) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xasync::XAsyncGetStatus;

    /// One saves folder per test process; each test uses its own configuration id.
    pub(super) fn root() -> PathBuf {
        static ROOT: OnceLock<PathBuf> = OnceLock::new();
        ROOT.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("decant-xgs-table-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            dir
        })
        .clone()
    }

    fn slots() -> &'static [usize; SLOTS] {
        // SAFETY: `Vtbl` is `repr(C)` with 40 function pointer fields, the size checked below.
        unsafe { &*((&raw const VTBL).cast::<[usize; SLOTS]>()) }
    }

    /// The vtable as the title reaches it: through the object `QueryApiImpl` hands out.
    fn table() -> &'static Vtbl {
        // SAFETY: the object's first field is its vtable pointer.
        unsafe { &**(object() as *const *const Vtbl) }
    }

    fn user() -> H {
        // The handle `XUserAddAsync` hands out.
        let mut b: XAsyncBlock = unsafe { std::mem::zeroed() };
        let mut user: H = std::ptr::null_mut();
        // SAFETY: the XUser table's slots 7 and 8 with the GDK arguments.
        unsafe {
            let xuser = &**(crate::local_user::object() as *const *const [unsafe extern "system" fn(); 50]);
            let add: unsafe extern "system" fn(*const c_void, u32, Block) -> i32 = std::mem::transmute(xuser[7]);
            let result: unsafe extern "system" fn(*const c_void, Block, *mut H) -> i32 = std::mem::transmute(xuser[8]);
            assert_eq!(add(std::ptr::null(), 0, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!(result(std::ptr::null(), &mut b, &mut user), S_OK);
        }
        assert!(crate::local_user::valid_user(user));
        user
    }

    fn block() -> XAsyncBlock {
        // SAFETY: an all zero block is the valid "process queue, no callback" block.
        unsafe { std::mem::zeroed() }
    }

    /// Opens a provider through the async form, as the title does.
    fn open(config: &std::ffi::CStr) -> H {
        let t = table();
        let this = &OBJECT as This;
        let mut b = block();
        let mut p: H = std::ptr::null_mut();
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            assert_eq!((t.initialize_provider_async)(this, user(), config.as_ptr().cast(), 0, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!((t.initialize_provider_result)(this, &mut b, &mut p), S_OK);
        }
        assert!(!p.is_null());
        p
    }

    /// Writes blobs into container `name` through one update, submitted async.
    fn write(p: H, name: &std::ffi::CStr, display: &std::ffi::CStr, blobs: &[(&std::ffi::CStr, &[u8])], deletes: &[&std::ffi::CStr]) -> i32 {
        let t = table();
        let this = &OBJECT as This;
        let (mut c, mut u): (H, H) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            assert_eq!((t.create_container)(this, p, name.as_ptr().cast(), &mut c), S_OK);
            assert_eq!((t.create_update)(this, c, display.as_ptr().cast(), &mut u), S_OK);
            for (blob, data) in blobs {
                assert_eq!((t.submit_blob_write)(this, u, blob.as_ptr().cast(), data.as_ptr(), data.len()), S_OK);
            }
            for blob in deletes {
                assert_eq!((t.submit_blob_delete)(this, u, blob.as_ptr().cast()), S_OK);
            }
            let mut b = block();
            assert_eq!((t.submit_update_async)(this, u, &mut b), S_OK);
            let hr = XAsyncGetStatus(&mut b, true);
            assert_eq!((t.submit_update_result)(this, &mut b), hr);
            (t.close_update)(this, u);
            (t.close_container)(this, c);
            hr
        }
    }

    /// Reads blobs of container `name` async: name and data of each, in the order handed out.
    fn read(p: H, name: &std::ffi::CStr, wanted: &[&std::ffi::CStr]) -> Result<Vec<(String, Vec<u8>)>, i32> {
        let t = table();
        let this = &OBJECT as This;
        let mut c: H = std::ptr::null_mut();
        let names: Vec<Text> = wanted.iter().map(|n| n.as_ptr().cast()).collect();
        // SAFETY: the slots with the GDK arguments; the layout read back is the one written.
        unsafe {
            assert_eq!((t.create_container)(this, p, name.as_ptr().cast(), &mut c), S_OK);
            let mut b = block();
            assert_eq!((t.read_blob_data_async)(this, c, if names.is_empty() { std::ptr::null() } else { names.as_ptr() }, names.len() as u32, &mut b), S_OK);
            let hr = XAsyncGetStatus(&mut b, true);
            let out = if hr < 0 {
                let mut count = 0;
                assert_eq!((t.read_blob_data_result)(this, &mut b, 0, std::ptr::null_mut(), &mut count), hr);
                Err(hr)
            } else {
                let mut size = 0usize;
                assert_eq!(xa::XAsyncGetResultSize(&mut b, &mut size), S_OK);
                let mut buf = vec![0u64; size.div_ceil(8).max(1)];
                let mut count = 0u32;
                if size > 0 {
                    assert_eq!((t.read_blob_data_result)(this, &mut b, size - 1, buf.as_mut_ptr().cast(), &mut count), BUFFER_TOO_SMALL, "a small buffer leaves the result");
                }
                assert_eq!((t.read_blob_data_result)(this, &mut b, size, buf.as_mut_ptr().cast(), &mut count), S_OK);
                Ok((0..count as usize)
                    .map(|i| {
                        let entry = buf.as_ptr().cast::<u8>().add(i * BLOB);
                        let name = std::ffi::CStr::from_ptr(entry.cast::<*const std::ffi::c_char>().read_unaligned()).to_str().unwrap().to_string();
                        let len = entry.add(8).cast::<u32>().read_unaligned() as usize;
                        let data = std::slice::from_raw_parts(entry.add(16).cast::<*const u8>().read_unaligned(), len).to_vec();
                        (name, data)
                    })
                    .collect())
            };
            (t.close_container)(this, c);
            out
        }
    }

    fn owned(v: &[(&str, &[u8])]) -> Vec<(String, Vec<u8>)> {
        v.iter().map(|(n, d)| (n.to_string(), d.to_vec())).collect()
    }

    #[test]
    fn the_structures_and_the_vtable_have_the_gdk_layout() {
        assert_eq!(size_of::<ContainerInfo>(), 48);
        assert_eq!(std::mem::offset_of!(ContainerInfo, total_size), 24);
        assert_eq!(std::mem::offset_of!(ContainerInfo, needs_sync), 40);
        assert_eq!(size_of::<BlobInfo>(), 16);
        assert_eq!(size_of::<Vtbl>(), SLOTS * size_of::<usize>());
        assert_eq!(slots()[3], initialize_provider as *const () as usize);
        assert_eq!(slots()[4], initialize_provider_async as *const () as usize);
        assert_eq!(slots()[13], get_container_info as *const () as usize);
        assert_eq!(slots()[20], read_blob_data as *const () as usize);
        assert_eq!(slots()[28], submit_update_async as *const () as usize);
        assert_eq!(slots()[32], files_get_remaining_quota as *const () as usize);
        // SAFETY: an unknown slot takes only `this`.
        unsafe { assert_eq!((table().slot39)(&OBJECT), E_NOTIMPL) };
    }

    #[test]
    fn both_riids_get_the_one_object_and_nothing_else_is_served() {
        let other = Guid { data1: 0x0DD112AC, data2: 0x7C24, data3: 0x448C, data4: [0xB9, 0x2B, 0x39, 0x60, 0xFB, 0x5B, 0xD3, 0x0C] };
        assert_eq!(object_for(Some(&XGAMESAVE_TABLE), Some(&XGAMESAVE_TABLE)), Some((object(), "XGameSave")));
        assert_eq!(object_for(Some(&XGAMESAVE_TABLE), Some(&XGAMESAVE_ALIAS)), Some((object(), "XGameSave")));
        assert_eq!(object_for(Some(&XGAMESAVE_TABLE), Some(&other)), None);
        assert_eq!(object_for(Some(&XGAMESAVE_ALIAS), Some(&XGAMESAVE_ALIAS)), None);
        assert_eq!(object_for(Some(&XGAMESAVE_TABLE), None), None);
        let mut out = std::ptr::dangling_mut::<c_void>();
        // SAFETY: valid GUIDs and out location.
        unsafe {
            assert_eq!((table().query_interface)(&OBJECT, &XGAMESAVE_ALIAS, &mut out), S_OK);
            assert_eq!(out, object());
            assert_eq!((table().query_interface)(&OBJECT, &other, &mut out), E_NOINTERFACE);
        }
    }

    #[test]
    fn the_provider_opens_for_the_local_user_only() {
        let t = table();
        let this = &OBJECT as This;
        let mut p: H = std::ptr::null_mut();
        let stranger = 0x7fff_7f67_3000_usize as H;
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            assert_eq!((t.initialize_provider)(this, stranger, c"scid-user".as_ptr().cast(), 0, &mut p), E_INVALIDARG);
            let mut b = block();
            assert_eq!((t.initialize_provider_async)(this, stranger, c"scid-user".as_ptr().cast(), 0, &mut b), E_INVALIDARG);
            assert_eq!((t.initialize_provider)(this, user(), c"scid-user".as_ptr().cast(), 1, &mut p), S_OK);
            assert!(lookup::<Provider>(p).unwrap().store.root().ends_with(format!("{:016X}\\scid-user", crate::local_user::USER_ID)));
            (t.close_provider)(this, p);
            assert!(lookup::<Provider>(p).is_none(), "a closed handle is refused");
            let mut quota = 0i64;
            assert_eq!((t.get_remaining_quota)(this, p, &mut quota), E_INVALIDARG);
        }
    }

    #[test]
    fn a_write_and_a_read_round_trip_through_the_table() {
        let t = table();
        let this = &OBJECT as This;
        let p = open(c"00000000-0000-0000-0000-00007a8b9c0d");
        assert_eq!(write(p, c"Profile/Slot_1", c"Chapter 1", &[(c"world", b"hello"), (c"player.dat", b"xy"), (c"gone", b"z")], &[c"gone"]), S_OK);
        assert_eq!(read(p, c"Profile/Slot_1", &[c"world", c"player.dat"]), Ok(owned(&[("world", b"hello"), ("player.dat", b"xy")])));
        assert_eq!(read(p, c"Profile/Slot_1", &[]), Ok(owned(&[("player.dat", b"xy"), ("world", b"hello")])));
        assert_eq!(read(p, c"Profile/Slot_1", &[c"gone"]), Err(store::BLOB_NOT_FOUND));
        assert_eq!(read(p, c"Missing", &[]), Err(store::BLOB_NOT_FOUND));

        // The sync forms.
        let mut c: H = std::ptr::null_mut();
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            assert_eq!((t.create_container)(this, p, c"Profile/Slot_1".as_ptr().cast(), &mut c), S_OK);
            let wanted = [c"world".as_ptr().cast::<u8>()];
            let mut count = 1u32;
            let mut small = [0u64; 2];
            assert_eq!((t.read_blob_data)(this, c, wanted.as_ptr(), &mut count, 16, small.as_mut_ptr().cast()), BUFFER_TOO_SMALL);
            let mut buf = [0u64; 8];
            assert_eq!((t.read_blob_data)(this, c, wanted.as_ptr(), &mut count, 64, buf.as_mut_ptr().cast()), S_OK);
            assert_eq!(count, 1);
            let entry = buf.as_ptr().cast::<u8>();
            assert_eq!(std::ffi::CStr::from_ptr(entry.cast::<*const std::ffi::c_char>().read_unaligned()), c"world");
            assert_eq!(std::slice::from_raw_parts(entry.add(16).cast::<*const u8>().read_unaligned(), 5), b"hello");

            let mut u: H = std::ptr::null_mut();
            assert_eq!((t.create_update)(this, c, std::ptr::null(), &mut u), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, c"world".as_ptr().cast(), b"again".as_ptr(), 5), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, std::ptr::null(), b"x".as_ptr(), 1), E_INVALIDARG);
            assert_eq!((t.submit_update)(this, u), S_OK);
            (t.close_update)(this, u);
            (t.close_container)(this, c);
        }
        assert_eq!(read(p, c"Profile/Slot_1", &[c"world"]), Ok(owned(&[("world", b"again")])));
        // SAFETY: as above.
        unsafe { (t.close_provider)(this, p) };
        assert!(lookup::<Provider>(p).is_none(), "the provider handle was released");
    }

    unsafe extern "system" fn collect_containers(info: *const ContainerInfo, context: *mut c_void) -> u8 {
        // SAFETY: the info lives for the call, the context is the test's vector.
        unsafe {
            let info = &*info;
            let out = &mut *context.cast::<Vec<(String, String, u32, u64)>>();
            let s = |p: *const u8| std::ffi::CStr::from_ptr(p.cast()).to_str().unwrap().to_string();
            assert!(!info.needs_sync && info.last_modified > 0);
            out.push((s(info.name), s(info.display_name), info.blob_count, info.total_size));
        }
        1
    }

    unsafe extern "system" fn collect_blobs(info: *const BlobInfo, context: *mut c_void) -> u8 {
        // SAFETY: as above.
        unsafe {
            let out = &mut *context.cast::<Vec<(String, u32)>>();
            out.push((std::ffi::CStr::from_ptr((*info).name.cast()).to_str().unwrap().to_string(), (*info).size));
        }
        1
    }

    unsafe extern "system" fn stop_after_one(_info: *const ContainerInfo, context: *mut c_void) -> u8 {
        // SAFETY: the context is the test's counter.
        unsafe { *context.cast::<u32>() += 1 };
        0
    }

    #[test]
    fn enumerate_get_info_quota_and_delete() {
        let t = table();
        let this = &OBJECT as This;
        let p = open(c"scid-enumerate");
        assert_eq!(write(p, c"Slot_1", c"First", &[(c"a", b"1234")], &[]), S_OK);
        assert_eq!(write(p, c"Slot_2", c"Second", &[(c"a", b"1"), (c"b", b"22")], &[]), S_OK);
        assert_eq!(write(p, c"Settings", c"", &[(c"cfg", b"x")], &[]), S_OK);
        let mut found: Vec<(String, String, u32, u64)> = Vec::new();
        let ctx = (&raw mut found).cast();
        let cb = collect_containers as *const () as usize;
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            assert_eq!((t.enumerate_container_info)(this, p, ctx, cb), S_OK);
            assert_eq!(found, [("Settings".into(), "".into(), 1, 1), ("Slot_1".into(), "First".into(), 1, 4), ("Slot_2".into(), "Second".into(), 2, 3)]);
            found.clear();
            assert_eq!((t.enumerate_container_info_by_name)(this, p, c"slot".as_ptr().cast(), ctx, cb), S_OK);
            assert_eq!(found.len(), 2);
            found.clear();
            assert_eq!((t.get_container_info)(this, p, c"Slot_2".as_ptr().cast(), ctx, cb), S_OK);
            assert_eq!(found, [("Slot_2".into(), "Second".into(), 2, 3)]);
            found.clear();
            assert_eq!((t.get_container_info)(this, p, c"Nothing".as_ptr().cast(), ctx, cb), S_OK, "a missing container is no error");
            assert!(found.is_empty());
            assert_eq!((t.get_container_info)(this, p, c"".as_ptr().cast(), ctx, cb), INVALID_CONTAINER_NAME);
            let mut calls = 0u32;
            assert_eq!((t.enumerate_container_info)(this, p, (&raw mut calls).cast(), stop_after_one as *const () as usize), S_OK);
            assert_eq!(calls, 1, "false from the callback stops the enumeration");

            let mut c: H = std::ptr::null_mut();
            assert_eq!((t.create_container)(this, p, c"Slot_2".as_ptr().cast(), &mut c), S_OK);
            let mut blobs: Vec<(String, u32)> = Vec::new();
            assert_eq!((t.enumerate_blob_info)(this, c, (&raw mut blobs).cast(), collect_blobs as *const () as usize), S_OK);
            assert_eq!(blobs, [("a".into(), 1), ("b".into(), 2)]);
            blobs.clear();
            assert_eq!((t.enumerate_blob_info_by_name)(this, c, c"B".as_ptr().cast(), (&raw mut blobs).cast(), collect_blobs as *const () as usize), S_OK);
            assert_eq!(blobs, [("b".into(), 2)]);
            (t.close_container)(this, c);

            let mut quota = 0i64;
            assert_eq!((t.get_remaining_quota)(this, p, &mut quota), S_OK);
            assert_eq!(quota as u64, store::QUOTA - 8);
            let mut b = block();
            assert_eq!((t.get_remaining_quota_async)(this, p, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            quota = 0;
            assert_eq!((t.get_remaining_quota_result)(this, &mut b, &mut quota), S_OK);
            assert_eq!(quota as u64, store::QUOTA - 8);

            assert_eq!((t.delete_container)(this, p, c"Settings".as_ptr().cast()), S_OK);
            let mut b = block();
            assert_eq!((t.delete_container_async)(this, p, c"Slot_1".as_ptr().cast(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!((t.delete_container_result)(this, &mut b), S_OK);
            assert_eq!((t.delete_container)(this, p, c"Slot_1".as_ptr().cast()), S_OK, "deleting a missing container is no error");
            found.clear();
            assert_eq!((t.enumerate_container_info)(this, p, ctx, cb), S_OK);
            assert_eq!(found, [("Slot_2".into(), "Second".into(), 2, 3)]);
            (t.close_provider)(this, p);
        }
    }

    #[test]
    fn a_half_written_generation_after_a_kill_is_never_read() {
        let p = open(c"scid-kill");
        assert_eq!(write(p, c"Slot", c"one", &[(c"a", b"old a")], &[]), S_OK);
        let dir = lookup::<Provider>(p).unwrap().store.root().to_path_buf();
        let container = std::fs::read_dir(&dir).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with("Slot~")).unwrap().path();
        // What a process killed while writing generation 2 leaves: a part written blob file and a
        // part written next manifest.
        std::fs::write(container.join("d2_0"), b"new").unwrap();
        std::fs::write(container.join("manifest.next"), b"decant_gs 2
name").unwrap();
        // SAFETY: the provider handle is closed once.
        unsafe { (table().close_provider)(&OBJECT, p) };

        // A new run opens the provider again and reads the old version.
        let p = open(c"scid-kill");
        assert_eq!(read(p, c"Slot", &[]), Ok(owned(&[("a", b"old a")])));
        assert!(!container.join("d2_0").exists() && !container.join("manifest.next").exists());
        assert_eq!(write(p, c"Slot", c"two", &[(c"a", b"new a")], &[]), S_OK);
        assert_eq!(read(p, c"Slot", &[]), Ok(owned(&[("a", b"new a")])));
        // SAFETY: as above.
        unsafe { (table().close_provider)(&OBJECT, p) };
    }

    #[test]
    fn the_completion_runs_on_the_title_queue() {
        use crate::taskqueue::{self as tq, MODE_MANUAL, PORT_COMPLETION, PORT_WORK};
        use std::sync::atomic::{AtomicU32, Ordering};
        static DONE: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn on_done(_b: *mut XAsyncBlock) {
            DONE.fetch_add(1, Ordering::SeqCst);
        }
        let t = table();
        let this = &OBJECT as This;
        // SAFETY: the queue, block and callback are the test's.
        unsafe {
            let mut q = std::ptr::null_mut();
            assert_eq!(tq::XTaskQueueCreate(MODE_MANUAL, MODE_MANUAL, &mut q), S_OK);
            let mut b = block();
            b.queue = q;
            b.callback = on_done as *const () as usize;
            assert_eq!((t.initialize_provider_async)(this, user(), c"scid-queue".as_ptr().cast(), 0, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, false), xa::E_PENDING, "the work waits for the title's dispatch");
            assert!(tq::XTaskQueueDispatch(q, PORT_WORK, 0));
            assert_eq!(XAsyncGetStatus(&mut b, false), S_OK);
            assert_eq!(DONE.load(Ordering::SeqCst), 0);
            assert!(tq::XTaskQueueDispatch(q, PORT_COMPLETION, 0));
            assert_eq!(DONE.load(Ordering::SeqCst), 1);
            let mut p: H = std::ptr::null_mut();
            assert_eq!((t.initialize_provider_result)(this, &mut b, &mut p), S_OK);
            assert!(lookup::<Provider>(p).is_some());
            (t.close_provider)(this, p);
            tq::XTaskQueueCloseHandle(q);
        }
    }

    #[test]
    fn the_save_files_folder_and_its_quota() {
        let t = table();
        let this = &OBJECT as This;
        // SAFETY: the slots with the GDK arguments.
        unsafe {
            let mut b = block();
            assert_eq!((t.files_get_folder_with_ui_async)(this, user(), c"scid-files".as_ptr().cast(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut small = [0u8; 4];
            assert_eq!((t.files_get_folder_with_ui_result)(this, &mut b, small.len(), small.as_mut_ptr()), BUFFER_TOO_SMALL);
            let mut buf = [0u8; 1024];
            assert_eq!((t.files_get_folder_with_ui_result)(this, &mut b, buf.len(), buf.as_mut_ptr()), S_OK);
            let folder = PathBuf::from(std::ffi::CStr::from_ptr(buf.as_ptr().cast()).to_str().unwrap());
            assert!(folder.is_dir() && folder.ends_with("files~\\scid-files"));
            std::fs::write(folder.join("save.bin"), [0u8; 10]).unwrap();
            let mut quota = 0i64;
            assert_eq!((t.files_get_remaining_quota)(this, user(), c"scid-files".as_ptr().cast(), &mut quota), S_OK);
            assert_eq!(quota as u64, store::QUOTA - 10);
        }
    }

    fn pattern_bytes(len: usize, seed: u8) -> Vec<u8> {
        (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add((i >> 8) as u8).wrapping_add(seed)).collect()
    }

    #[test]
    fn full_save_roundtrip_and_failure_paths() {
        let t = table();
        let this = &OBJECT as This;
        let scid = c"00000000-0000-0000-0000-00000000cafe";
        let container_name = c"SaveData/Slot_01";
        let weird_name = c"dir/sub:name*with?chars <1> | ~!@#$.dat";
        let utf8_name = c"\xC3\xA9\xC3\xA0\xC3\xBC_save.bin";

        let small_data = vec![0x42u8];
        let med_data = pattern_bytes(64 * 1024, 3);
        let large_data = pattern_bytes(3 * 1024 * 1024, 17);
        let empty_data: Vec<u8> = Vec::new();
        let weird_data = pattern_bytes(1024, 99);
        let utf8_data = pattern_bytes(512, 55);

        let p = open(scid);
        let mut c: H = std::ptr::null_mut();
        // SAFETY: valid provider, container name, and output handle.
        unsafe {
            assert_eq!((t.create_container)(this, p, container_name.as_ptr().cast(), &mut c), S_OK);
        }

        // First update: write 1-byte, 64 KiB, 3 MiB, empty, and unusual-name blobs.
        let mut u: H = std::ptr::null_mut();
        // SAFETY: valid container, update, blob slices, and async block.
        unsafe {
            assert_eq!((t.create_update)(this, c, c"Chapter 1".as_ptr().cast(), &mut u), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, c"flag.bin".as_ptr().cast(), small_data.as_ptr(), small_data.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, c"settings.dat".as_ptr().cast(), med_data.as_ptr(), med_data.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, c"world_state.bin".as_ptr().cast(), large_data.as_ptr(), large_data.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, c"empty.tag".as_ptr().cast(), empty_data.as_ptr(), empty_data.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, weird_name.as_ptr().cast(), weird_data.as_ptr(), weird_data.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u, utf8_name.as_ptr().cast(), utf8_data.as_ptr(), utf8_data.len()), S_OK);

            let mut b = block();
            assert_eq!((t.submit_update_async)(this, u, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            assert_eq!((t.submit_update_result)(this, &mut b), S_OK);
            (t.close_update)(this, u);
        }

        // Read all blobs back through the async read call and compare every byte.
        let expected_first = owned(&[
            ("dir/sub:name*with?chars <1> | ~!@#$.dat", &weird_data),
            ("empty.tag", &empty_data),
            ("flag.bin", &small_data),
            ("settings.dat", &med_data),
            ("world_state.bin", &large_data),
            ("\u{e9}\u{e0}\u{fc}_save.bin", &utf8_data),
        ]);
        assert_eq!(read(p, container_name, &[]), Ok(expected_first));

        // Read specific blobs through the synchronous read call too.
        // SAFETY: valid container handle and adequately sized buffer.
        unsafe {
            let wanted = [c"world_state.bin".as_ptr().cast::<u8>(), c"empty.tag".as_ptr().cast::<u8>()];
            let mut count = wanted.len() as u32;
            let mut tiny = [0u8; 32];
            assert_eq!((t.read_blob_data)(this, c, wanted.as_ptr(), &mut count, tiny.len(), tiny.as_mut_ptr()), BUFFER_TOO_SMALL);
            let need = 2 * BLOB + 64 + large_data.len();
            let mut buf = vec![0u8; need];
            assert_eq!((t.read_blob_data)(this, c, wanted.as_ptr(), &mut count, buf.len(), buf.as_mut_ptr()), S_OK);
            assert_eq!(count, 2);
            let e0 = buf.as_ptr();
            let e1 = buf.as_ptr().add(BLOB);
            assert_eq!(std::ffi::CStr::from_ptr(e0.cast::<*const std::ffi::c_char>().read_unaligned()), c"world_state.bin");
            assert_eq!(e0.add(8).cast::<u32>().read_unaligned() as usize, large_data.len());
            assert_eq!(std::slice::from_raw_parts(e0.add(16).cast::<*const u8>().read_unaligned(), large_data.len()), large_data.as_slice());
            assert_eq!(std::ffi::CStr::from_ptr(e1.cast::<*const std::ffi::c_char>().read_unaligned()), c"empty.tag");
            assert_eq!(e1.add(8).cast::<u32>().read_unaligned(), 0);
        }

        // Update one blob, submit again, and verify unchanged blobs stay identical.
        let med_updated = pattern_bytes(64 * 1024, 42);
        assert_eq!(write(p, container_name, c"Chapter 2", &[(c"settings.dat", &med_updated)], &[]), S_OK);
        let expected_second = owned(&[
            ("dir/sub:name*with?chars <1> | ~!@#$.dat", &weird_data),
            ("empty.tag", &empty_data),
            ("flag.bin", &small_data),
            ("settings.dat", &med_updated),
            ("world_state.bin", &large_data),
            ("\u{e9}\u{e0}\u{fc}_save.bin", &utf8_data),
        ]);
        assert_eq!(read(p, container_name, &[]), Ok(expected_second));

        // Failure path: an update that is not submitted leaves no change visible.
        // SAFETY: valid container and update handles.
        unsafe {
            let mut u_discard: H = std::ptr::null_mut();
            assert_eq!((t.create_update)(this, c, c"Discarded".as_ptr().cast(), &mut u_discard), S_OK);
            assert_eq!((t.submit_blob_write)(this, u_discard, c"settings.dat".as_ptr().cast(), b"bad".as_ptr(), 3), S_OK);
            assert_eq!((t.submit_blob_write)(this, u_discard, c"ghost.bin".as_ptr().cast(), b"boo".as_ptr(), 3), S_OK);
            assert_eq!((t.submit_blob_delete)(this, u_discard, c"flag.bin".as_ptr().cast()), S_OK);
            (t.close_update)(this, u_discard);
        }
        assert_eq!(read(p, container_name, &[c"settings.dat", c"flag.bin"]), Ok(owned(&[("settings.dat", &med_updated), ("flag.bin", &small_data)])));
        assert_eq!(read(p, container_name, &[c"ghost.bin"]), Err(store::BLOB_NOT_FOUND));

        // Failure path: a blob larger than MAX_UPDATE (16 MiB) is refused with E_GS_UPDATE_TOO_BIG.
        // SAFETY: valid update handle and oversized buffer.
        unsafe {
            let mut u_big: H = std::ptr::null_mut();
            assert_eq!((t.create_update)(this, c, std::ptr::null(), &mut u_big), S_OK);
            let too_big = vec![0xAAu8; store::MAX_UPDATE as usize + 1];
            assert_eq!((t.submit_blob_write)(this, u_big, c"huge.bin".as_ptr().cast(), too_big.as_ptr(), too_big.len()), store::UPDATE_TOO_BIG);
            // Two blobs in one update whose sum exceeds MAX_UPDATE are refused at submit_update.
            let half = vec![0x55u8; 9 * 1024 * 1024];
            assert_eq!((t.submit_blob_write)(this, u_big, c"half1.bin".as_ptr().cast(), half.as_ptr(), half.len()), S_OK);
            assert_eq!((t.submit_blob_write)(this, u_big, c"half2.bin".as_ptr().cast(), half.as_ptr(), half.len()), S_OK);
            assert_eq!((t.submit_update)(this, u_big), store::UPDATE_TOO_BIG);
            (t.close_update)(this, u_big);
        }

        // Failure path: invalid container and blob names.
        // SAFETY: valid provider and update handles.
        unsafe {
            let mut bad_c: H = std::ptr::null_mut();
            assert_eq!((t.create_container)(this, p, c"".as_ptr().cast(), &mut bad_c), INVALID_CONTAINER_NAME);
            let long_name = std::ffi::CString::new("x".repeat(store::MAX_NAME + 1)).unwrap();
            assert_eq!((t.create_container)(this, p, long_name.as_ptr().cast(), &mut bad_c), INVALID_CONTAINER_NAME);
            let mut u_bad: H = std::ptr::null_mut();
            assert_eq!((t.create_update)(this, c, std::ptr::null(), &mut u_bad), S_OK);
            assert_eq!((t.submit_blob_write)(this, u_bad, c"".as_ptr().cast(), b"a".as_ptr(), 1), E_INVALIDARG);
            assert_eq!((t.submit_blob_write)(this, u_bad, c"a".as_ptr().cast(), std::ptr::null(), 1), E_INVALIDARG);
            (t.close_update)(this, u_bad);
        }

        // Two updates of one container created together and submitted sequentially.
        // SAFETY: valid container and update handles.
        unsafe {
            let (mut u1, mut u2): (H, H) = (std::ptr::null_mut(), std::ptr::null_mut());
            assert_eq!((t.create_update)(this, c, std::ptr::null(), &mut u1), S_OK);
            assert_eq!((t.create_update)(this, c, std::ptr::null(), &mut u2), S_OK);
            assert_eq!((t.submit_blob_write)(this, u1, c"u1_blob".as_ptr().cast(), b"one".as_ptr(), 3), S_OK);
            assert_eq!((t.submit_blob_write)(this, u2, c"u2_blob".as_ptr().cast(), b"two".as_ptr(), 3), S_OK);
            assert_eq!((t.submit_update)(this, u1), S_OK);
            assert_eq!((t.submit_update)(this, u2), S_OK);
            (t.close_update)(this, u1);
            (t.close_update)(this, u2);
        }
        assert_eq!(read(p, container_name, &[c"u1_blob", c"u2_blob"]), Ok(owned(&[("u1_blob", b"one"), ("u2_blob", b"two")])));

        // Delete a blob and verify enumeration reflects the change.
        assert_eq!(write(p, container_name, c"Chapter 2", &[], &[c"flag.bin"]), S_OK);
        assert_eq!(read(p, container_name, &[c"flag.bin"]), Err(store::BLOB_NOT_FOUND));
        let mut blobs_enum: Vec<(String, u32)> = Vec::new();
        let mut containers_enum: Vec<(String, String, u32, u64)> = Vec::new();
        // SAFETY: valid handles and test callback vectors.
        unsafe {
            assert_eq!((t.enumerate_blob_info)(this, c, (&raw mut blobs_enum).cast(), collect_blobs as *const () as usize), S_OK);
            assert!(!blobs_enum.iter().any(|(n, _)| n == "flag.bin"));
            assert_eq!(blobs_enum.len(), 7);
            assert_eq!((t.enumerate_container_info)(this, p, (&raw mut containers_enum).cast(), collect_containers as *const () as usize), S_OK);
            assert_eq!(containers_enum.len(), 1);
            assert_eq!(containers_enum[0].0, "SaveData/Slot_01");
            assert_eq!(containers_enum[0].1, "Chapter 2");
            assert_eq!(containers_enum[0].2, 7);
        }

        // Failure path: read of a missing container returns E_GS_BLOB_NOT_FOUND.
        assert_eq!(read(p, c"MissingContainer", &[]), Err(store::BLOB_NOT_FOUND));

        // Close container and provider, then reopen provider for the same user and config.
        // SAFETY: valid handles closed once.
        unsafe {
            (t.close_container)(this, c);
            (t.close_provider)(this, p);
        }
        assert!(lookup::<Provider>(p).is_none());
        assert!(lookup::<Container>(c).is_none());

        let p2 = open(scid);
        let expected_after_reopen = owned(&[
            ("dir/sub:name*with?chars <1> | ~!@#$.dat", &weird_data),
            ("empty.tag", &empty_data),
            ("settings.dat", &med_updated),
            ("u1_blob", b"one"),
            ("u2_blob", b"two"),
            ("world_state.bin", &large_data),
            ("\u{e9}\u{e0}\u{fc}_save.bin", &utf8_data),
        ]);
        assert_eq!(read(p2, container_name, &[]), Ok(expected_after_reopen));

        // Delete the container and confirm it is gone from enumeration and reads.
        // SAFETY: valid provider handle and container name.
        unsafe {
            assert_eq!((t.delete_container)(this, p2, container_name.as_ptr().cast()), S_OK);
            containers_enum.clear();
            assert_eq!((t.enumerate_container_info)(this, p2, (&raw mut containers_enum).cast(), collect_containers as *const () as usize), S_OK);
            assert!(containers_enum.is_empty());
            assert_eq!(read(p2, container_name, &[]), Err(store::BLOB_NOT_FOUND));
            (t.close_provider)(this, p2);
        }
    }
}
