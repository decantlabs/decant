//! Offline XPackage table: `api_id = {AF406016-E850-4AA8-A88D-2F3DCB9DAC7E}`, interface
//! `{0B9D556E-AB70-43F4-BAA0-9999AC4CB040}`. A title started as a plain exe has no package identity
//! on a PC. This table therefore describes the title's own installation as finished and
//! launchable, and denies that any other package exists.
//!
//! What it answers:
//! * The package identifier is the title id of `MicrosoftGame.config` in eight hex digits, and
//!   `XPackageIsPackagedProcess` is true.
//! * Every installation monitor is the same fixed handle. Its progress reports total, installed and
//!   launch bytes as the size of the game folder, launchable and completed. Progress callbacks get a
//!   token and never fire, `Update` never reports a change.
//! * Chunk availability is "ready" (0) for our package. Reordering or uninstalling chunks is accepted
//!   and changes nothing. `InstallChunksAsync` completes at once with the monitor, and the download
//!   estimate is 0 bytes without a confirmation request.
//! * Mounting our package (`XPackageMount`, `XPackageMountWithUiAsync`) yields one handle whose
//!   path is the game folder. Any other identifier fails with `0x89245208` (no such package).
//! * `XPackageEnumeratePackages` reports the game once for kind 0 and nothing for content (kind 1),
//!   because there are no add-ons.
//!
//! Slot order follows the thunks of the public PC GDK `xgameruntime.lib` for this interface (slots
//! 0x18 to 0x138). A slot without a known function logs a named "not implemented" and returns
//! `E_NOTIMPL` (see `local_title`). The vtable has 48 entries so a later slot never reads past it.
//! Every call is logged with a bound (`calllog`). Signatures come from the public GDK `XPackage.h`.
#![allow(unsafe_code)]

use std::ffi::{CString, c_char, c_void};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::local_title::{LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, package_identifier, put, put_string, query_interface, slot, text, title, token};
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`.
pub(crate) const XPACKAGE_TABLE: Guid = Guid { data1: 0xAF406016, data2: 0xE850, data3: 0x4AA8, data4: [0xA8, 0x8D, 0x2F, 0x3D, 0xCB, 0x9D, 0xAC, 0x7E] };
/// The interface id of the XPackage vtable.
pub(crate) const XPACKAGE_IID: Guid = Guid { data1: 0x0B9D556E, data2: 0xAB70, data3: 0x43F4, data4: [0xBA, 0xA0, 0x99, 0x99, 0xAC, 0x4C, 0xB0, 0x40] };

/// `E_GAMEPACKAGE_NO_PACKAGE_IDENTIFIER`: no package has this identifier.
pub(crate) const NO_PACKAGE: i32 = 0x8924_5208_u32 as i32;

const NAME: &str = "XPackage";
const SLOTS: usize = 48;

static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);

/// The object for the couple this table serves, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&XPACKAGE_TABLE) && riid == Some(&XPACKAGE_IID)).then(|| (OBJECT.as_out(), NAME))
}

struct Package;

impl LocalTable for Package {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[XPACKAGE_IID];
    fn counts() -> &'static Counters<MAX_SLOTS> {
        &COUNTS
    }
    fn slot_name(slot: usize) -> Option<&'static str> {
        SLOT_NAMES.iter().find(|(number, _)| *number == slot).map(|&(_, name)| name)
    }
    fn object() -> *mut c_void {
        OBJECT.as_out()
    }
}

/// GDK function names by slot. Slots 0 to 2 are the COM basics, absent numbers are unidentified.
const SLOT_NAMES: [(usize, &str); 35] = [
    (0, "QueryInterface"),
    (1, "AddRef"),
    (2, "Release"),
    (3, "XPackageGetCurrentProcessPackageIdentifier"),
    (4, "XPackageIsPackagedProcess"),
    (5, "XPackageCreateInstallationMonitor"),
    (6, "XPackageCloseInstallationMonitorHandle"),
    (7, "XPackageGetInstallationProgress"),
    (8, "XPackageUpdateInstallationMonitor"),
    (9, "XPackageRegisterInstallationProgressChanged"),
    (10, "XPackageUnregisterInstallationProgressChanged"),
    (11, "XPackageGetUserLocale"),
    (12, "XPackageFindChunkAvailability"),
    (13, "XPackageEnumerateChunkAvailability"),
    (14, "XPackageChangeChunkInstallOrder"),
    (15, "XPackageInstallChunks"),
    (16, "XPackageInstallChunksAsync"),
    (17, "XPackageInstallChunksResult"),
    (18, "XPackageEstimateDownloadSize"),
    (19, "XPackageUninstallChunks"),
    (22, "XPackageUnregisterPackageInstalled"),
    (23, "XPackageMount"),
    (24, "XPackageGetMountPathSize"),
    (25, "XPackageGetMountPath"),
    (26, "XPackageCloseMountHandle"),
    (27, "XPackageGetIdentifier"),
    (30, "XPackageGetWriteStats"),
    (32, "XPackageUninstallUWPInstance"),
    (33, "XPackageEnumerateFeatures"),
    (34, "XPackageUninstallPackage"),
    (37, "XPackageMountWithUiAsync"),
    (38, "XPackageMountWithUiResult"),
    (39, "XPackageEnumeratePackages"),
    (40, "XPackageRegisterPackageInstalled"),
    (41, "XPackageGetPackageKind"),
];

type This = *const c_void;
type Id = *const c_char;
type Block = *mut XAsyncBlock;

/// `XPackageInstallationProgress`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Progress {
    pub(crate) total_bytes: u64,
    pub(crate) installed_bytes: u64,
    pub(crate) launch_bytes: u64,
    pub(crate) launchable: bool,
    pub(crate) completed: bool,
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

/// `XPackageDetails`.
#[repr(C)]
pub(crate) struct PackageDetails {
    pub(crate) package_identifier: *const c_char,
    pub(crate) version: Version,
    pub(crate) kind: u32,
    pub(crate) display_name: *const c_char,
    pub(crate) description: *const c_char,
    pub(crate) publisher: *const c_char,
    pub(crate) store_id: *const c_char,
    pub(crate) installing: bool,
    pub(crate) index: u32,
    pub(crate) count: u32,
    pub(crate) age_restricted: bool,
    pub(crate) title_id: *const c_char,
}

type EnumeratePackagesCallback = unsafe extern "system" fn(*mut c_void, *const PackageDetails) -> bool;

/// `XPackageKind::Game`.
const KIND_GAME: u32 = 0;

/// Whether `id` names this title's package. Another identifier is logged (bounded).
fn names_us(id: Id) -> bool {
    let ours = package_identifier();
    let asked = text(id);
    let yes = asked.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(&ours));
    static OTHERS: AtomicU32 = AtomicU32::new(0);
    if !yes && OTHERS.fetch_add(1, Ordering::Relaxed) < 20 {
        log(&format!("local {NAME} table: package {asked:?} is not this title's ({ours}) -> no package"));
    }
    yes
}

/// Total size of the files below the game folder, measured once. Links are followed and the walk
/// stops at depth eight. An empty folder counts as one byte so that progress is never 0 of 0.
fn folder_bytes() -> u64 {
    fn walk(root: &Path) -> u64 {
        let mut pending = vec![(root.to_path_buf(), 0u32)];
        let mut sum = 0;
        while let Some((dir, depth)) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let Ok(meta) = std::fs::metadata(entry.path()) else { continue };
                if meta.is_file() {
                    sum += meta.len();
                } else if meta.is_dir() && depth < 8 {
                    pending.push((entry.path(), depth + 1));
                }
            }
        }
        sum
    }
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let bytes = walk(&title().folder).max(1);
        log(&format!("local {NAME} table: the package holds {bytes} bytes, all installed"));
        bytes
    })
}

// Every monitor watches the one finished installation: a fixed handle.
static MONITOR: u8 = 0;
fn monitor_handle() -> *mut c_void {
    (&raw const MONITOR).cast_mut().cast()
}

/// The one mount: our package, at the game folder.
static MOUNT: u8 = 0;
fn mount_handle() -> *mut c_void {
    (&raw const MOUNT).cast_mut().cast()
}

/// The mount path: the game folder.
fn mount_path() -> String {
    let path = title().folder.to_string_lossy().into_owned();
    path.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(path)
}

static INSTALL_ID: u8 = 0;
static MOUNT_ID: u8 = 0;

logged_shim!(NAME, COUNTS, This, 3 as "XPackageGetCurrentProcessPackageIdentifier", current_identifier_entry => own_identifier, (size: usize, buffer: *mut u8) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XPackageIsPackagedProcess", is_packaged_entry => always_packaged, () -> bool);
logged_shim!(NAME, COUNTS, This, 5 as "XPackageCreateInstallationMonitor", create_monitor_entry => open_monitor, (id: Id, count: u32, selectors: *const c_void, interval_ms: u32, queue: *mut c_void, out: *mut *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 6 as "XPackageCloseInstallationMonitorHandle", close_monitor_entry => ignore_handle, (monitor: *mut c_void));
logged_shim!(NAME, COUNTS, This, 7 as "XPackageGetInstallationProgress", progress_entry => report_progress, (monitor: *mut c_void, out: *mut Progress));
logged_shim!(NAME, COUNTS, This, 8 as "XPackageUpdateInstallationMonitor", update_monitor_entry => never_changed, (monitor: *mut c_void) -> bool);
logged_shim!(NAME, COUNTS, This, 9 as "XPackageRegisterInstallationProgressChanged", register_progress_entry => accept_progress_listener, (monitor: *mut c_void, context: *mut c_void, callback: usize, out: *mut u64) -> i32);
logged_shim!(NAME, COUNTS, This, 10 as "XPackageUnregisterInstallationProgressChanged", unregister_progress_entry => drop_progress_listener, (monitor: *mut c_void, token: u64, wait: bool) -> bool);
logged_shim!(NAME, COUNTS, This, 11 as "XPackageGetUserLocale", user_locale_entry => windows_locale, (size: usize, buffer: *mut u8) -> i32);
logged_shim!(NAME, COUNTS, This, 12 as "XPackageFindChunkAvailability", find_availability_entry => chunk_state, (id: Id, count: u32, selectors: *const c_void, out: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 13 as "XPackageEnumerateChunkAvailability", enumerate_chunks_entry => list_chunks, (id: Id, kind: u32, context: *mut c_void, callback: usize) -> i32);
logged_shim!(NAME, COUNTS, This, 14 as "XPackageChangeChunkInstallOrder", change_order_entry => acknowledge_chunks, (id: Id, count: u32, selectors: *const c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 15 as "XPackageInstallChunks", install_chunks_entry => begin_chunk_install, (id: Id, count: u32, selectors: *const c_void, interval_ms: u32, suppress: bool, queue: *mut c_void, out: *mut *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 16 as "XPackageInstallChunksAsync", install_chunks_async_entry => begin_chunk_install_async, (id: Id, count: u32, selectors: *const c_void, interval_ms: u32, suppress: bool, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 17 as "XPackageInstallChunksResult", install_chunks_result_entry => chunk_install_result, (block: Block, out: *mut *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 18 as "XPackageEstimateDownloadSize", estimate_download_entry => download_estimate, (id: Id, count: u32, selectors: *const c_void, bytes: *mut u64, confirm: *mut bool) -> i32);
logged_shim!(NAME, COUNTS, This, 19 as "XPackageUninstallChunks", uninstall_chunks_entry => acknowledge_chunks, (id: Id, count: u32, selectors: *const c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 22 as "XPackageUnregisterPackageInstalled", unregister_installed_entry => drop_install_listener, (token: u64, wait: bool) -> bool);
logged_shim!(NAME, COUNTS, This, 23 as "XPackageMount", mount_entry => mount_ours, (id: Id, out: *mut *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 24 as "XPackageGetMountPathSize", mount_path_size_entry => mount_path_len, (mount: *mut c_void, size: *mut usize) -> i32);
logged_shim!(NAME, COUNTS, This, 25 as "XPackageGetMountPath", mount_path_entry => mount_path_copy, (mount: *mut c_void, size: usize, buffer: *mut u8) -> i32);
logged_shim!(NAME, COUNTS, This, 26 as "XPackageCloseMountHandle", close_mount_entry => ignore_handle, (mount: *mut c_void));
logged_shim!(NAME, COUNTS, This, 27 as "XPackageGetIdentifier", identifier_entry => identifier_of, (path_or_name: *const c_char, size: usize, buffer: *mut u8) -> i32);
logged_shim!(NAME, COUNTS, This, 30 as "XPackageGetWriteStats", write_stats_entry => zero_write_stats, (out: *mut [u64; 4]) -> i32);
logged_shim!(NAME, COUNTS, This, 33 as "XPackageEnumerateFeatures", enumerate_features_entry => list_features, (id: Id, context: *mut c_void, callback: usize) -> i32);
logged_shim!(NAME, COUNTS, This, 34 as "XPackageUninstallPackage", uninstall_package_entry => refuse_uninstall, (id: Id) -> bool);
logged_shim!(NAME, COUNTS, This, 37 as "XPackageMountWithUiAsync", mount_async_entry => mount_ours_async, (id: Id, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 38 as "XPackageMountWithUiResult", mount_result_entry => mount_ours_result, (block: Block, out: *mut *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 39 as "XPackageEnumeratePackages", enumerate_packages_entry => list_packages, (kind: u32, scope: u32, context: *mut c_void, callback: usize) -> i32);
logged_shim!(NAME, COUNTS, This, 40 as "XPackageRegisterPackageInstalled", register_installed_entry => accept_install_listener, (queue: *mut c_void, context: *mut c_void, callback: usize, out: *mut u64) -> i32);

unsafe fn own_identifier(size: usize, buffer: *mut u8) -> i32 {
    put_string(&package_identifier(), size, buffer, std::ptr::null_mut())
}

unsafe fn always_packaged() -> bool {
    true
}

unsafe fn open_monitor(id: Id, _count: u32, _selectors: *const c_void, _interval_ms: u32, _queue: *mut c_void, out: *mut *mut c_void) -> i32 {
    if !names_us(id) {
        return NO_PACKAGE;
    }
    put(out, monitor_handle())
}

unsafe fn ignore_handle(_handle: *mut c_void) {}

unsafe fn report_progress(_monitor: *mut c_void, out: *mut Progress) {
    let bytes = folder_bytes();
    put(out, Progress { total_bytes: bytes, installed_bytes: bytes, launch_bytes: bytes, launchable: true, completed: true });
}

/// The installation is finished: it never changes.
unsafe fn never_changed(_monitor: *mut c_void) -> bool {
    false
}

unsafe fn accept_progress_listener(_monitor: *mut c_void, _context: *mut c_void, _callback: usize, out: *mut u64) -> i32 {
    put(out, token())
}

unsafe fn drop_progress_listener(_monitor: *mut c_void, _token: u64, _wait: bool) -> bool {
    true
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetUserDefaultLocaleName(name: *mut u16, size: i32) -> i32;
}

/// The user's locale of Windows, `en-US` when it has none.
unsafe fn windows_locale(size: usize, buffer: *mut u8) -> i32 {
    let mut name = [0u16; 85];
    // SAFETY: the buffer holds 85 units, as passed.
    let n = unsafe { GetUserDefaultLocaleName(name.as_mut_ptr(), name.len() as i32) };
    let locale = if n > 1 { String::from_utf16_lossy(&name[..n as usize - 1]) } else { "en-US".to_owned() };
    put_string(&locale, size, buffer, std::ptr::null_mut())
}

/// 0: ready, for every chunk of our package.
unsafe fn chunk_state(id: Id, _count: u32, _selectors: *const c_void, out: *mut u32) -> i32 {
    if !names_us(id) {
        return NO_PACKAGE;
    }
    put(out, 0)
}

/// The chunk layout is not read here, so no chunk is listed.
unsafe fn list_chunks(id: Id, _kind: u32, _context: *mut c_void, _callback: usize) -> i32 {
    if names_us(id) { S_OK } else { NO_PACKAGE }
}

/// Install order and uninstall of chunks: accepted, nothing changes, everything stays ready.
unsafe fn acknowledge_chunks(id: Id, _count: u32, _selectors: *const c_void) -> i32 {
    if names_us(id) { S_OK } else { NO_PACKAGE }
}

unsafe fn begin_chunk_install(id: Id, count: u32, selectors: *const c_void, interval_ms: u32, _suppress: bool, queue: *mut c_void, out: *mut *mut c_void) -> i32 {
    // SAFETY: the same arguments.
    unsafe { open_monitor(id, count, selectors, interval_ms, queue, out) }
}

/// Everything is installed: the call completes through the block with the monitor.
unsafe fn begin_chunk_install_async(id: Id, _count: u32, _selectors: *const c_void, _interval_ms: u32, _suppress: bool, block: Block) -> i32 {
    let result = if names_us(id) { Ok((monitor_handle() as usize).to_le_bytes().to_vec()) } else { Err(NO_PACKAGE) };
    // SAFETY: the title's block, valid until the call completes.
    unsafe { xa::begin_with(block, &raw const INSTALL_ID as usize, move || result) }
}

unsafe fn chunk_install_result(block: Block, out: *mut *mut c_void) -> i32 {
    // SAFETY: the title's block and an eight byte location.
    unsafe { xa::XAsyncGetResult(block, (&raw const INSTALL_ID).cast(), size_of::<usize>(), out.cast(), std::ptr::null_mut()) }
}

unsafe fn download_estimate(id: Id, _count: u32, _selectors: *const c_void, bytes: *mut u64, confirm: *mut bool) -> i32 {
    if !names_us(id) {
        return NO_PACKAGE;
    }
    if !confirm.is_null() {
        put(confirm, false);
    }
    put(bytes, 0)
}

unsafe fn drop_install_listener(_token: u64, _wait: bool) -> bool {
    true
}

unsafe fn mount_ours(id: Id, out: *mut *mut c_void) -> i32 {
    if !names_us(id) {
        return NO_PACKAGE;
    }
    log(&format!("local {NAME} table: package {} mounted at {}", package_identifier(), mount_path()));
    put(out, mount_handle())
}

unsafe fn mount_path_len(handle: *mut c_void, size: *mut usize) -> i32 {
    if handle != mount_handle() {
        return E_INVALIDARG;
    }
    put(size, mount_path().len() + 1)
}

unsafe fn mount_path_copy(handle: *mut c_void, size: usize, buffer: *mut u8) -> i32 {
    if handle != mount_handle() {
        return E_INVALIDARG;
    }
    put_string(&mount_path(), size, buffer, std::ptr::null_mut())
}

/// The identifier of the package at a path or of a name: only ours is installed.
unsafe fn identifier_of(_path_or_name: *const c_char, size: usize, buffer: *mut u8) -> i32 {
    put_string(&package_identifier(), size, buffer, std::ptr::null_mut())
}

/// `XPackageWriteStats`: nothing measured.
unsafe fn zero_write_stats(out: *mut [u64; 4]) -> i32 {
    put(out, [0; 4])
}

/// No optional feature.
unsafe fn list_features(id: Id, _context: *mut c_void, _callback: usize) -> i32 {
    if names_us(id) { S_OK } else { NO_PACKAGE }
}

/// Only other packages are uninstalled this way, and none is installed.
unsafe fn refuse_uninstall(id: Id) -> bool {
    log(&format!("local {NAME} table: uninstall of package {:?} refused", text(id)));
    false
}

unsafe fn mount_ours_async(id: Id, block: Block) -> i32 {
    let result = if names_us(id) {
        log(&format!("local {NAME} table: package {} mounted at {}", package_identifier(), mount_path()));
        Ok((mount_handle() as usize).to_le_bytes().to_vec())
    } else {
        Err(NO_PACKAGE)
    };
    // SAFETY: the title's block, valid until the call completes.
    unsafe { xa::begin_with(block, &raw const MOUNT_ID as usize, move || result) }
}

unsafe fn mount_ours_result(block: Block, out: *mut *mut c_void) -> i32 {
    // SAFETY: the title's block and an eight byte location.
    unsafe { xa::XAsyncGetResult(block, (&raw const MOUNT_ID).cast(), size_of::<usize>(), out.cast(), std::ptr::null_mut()) }
}

/// Kind 0 (game): the title itself, once. Content (kind 1) and the others: none.
unsafe fn list_packages(kind: u32, _scope: u32, context: *mut c_void, callback: usize) -> i32 {
    if callback == 0 {
        return E_INVALIDARG;
    }
    if kind != KIND_GAME {
        return S_OK;
    }
    let ours = package_identifier();
    let id = CString::new(ours.clone()).unwrap_or_default();
    let empty = CString::default();
    let details = PackageDetails {
        package_identifier: id.as_ptr(),
        version: Version { major: 1, minor: 0, build: 0, revision: 0 },
        kind: KIND_GAME,
        display_name: id.as_ptr(),
        description: empty.as_ptr(),
        publisher: empty.as_ptr(),
        store_id: empty.as_ptr(),
        installing: false,
        index: 0,
        count: 1,
        age_restricted: false,
        title_id: id.as_ptr(),
    };
    // SAFETY: the title's `XPackageEnumerationCallback`, not null.
    let callback: EnumeratePackagesCallback = unsafe { std::mem::transmute::<usize, EnumeratePackagesCallback>(callback) };
    // SAFETY: the details and their strings live for the call.
    unsafe { callback(context, &details) };
    S_OK
}

/// Nothing gets installed while the title runs: the callback never fires.
unsafe fn accept_install_listener(_queue: *mut c_void, _context: *mut c_void, _callback: usize, out: *mut u64) -> i32 {
    put(out, token())
}

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<Package>),
    slot!(no_count),
    slot!(no_count),
    slot!(current_identifier_entry),
    slot!(is_packaged_entry),
    slot!(create_monitor_entry),
    slot!(close_monitor_entry),
    slot!(progress_entry),
    slot!(update_monitor_entry),
    slot!(register_progress_entry),
    slot!(unregister_progress_entry),
    slot!(user_locale_entry),
    slot!(find_availability_entry),
    slot!(enumerate_chunks_entry),
    slot!(change_order_entry),
    slot!(install_chunks_entry),
    slot!(install_chunks_async_entry),
    slot!(install_chunks_result_entry),
    slot!(estimate_download_entry),
    slot!(uninstall_chunks_entry),
    slot!(not_implemented::<Package, 20>),
    slot!(not_implemented::<Package, 21>),
    slot!(unregister_installed_entry),
    slot!(mount_entry),
    slot!(mount_path_size_entry),
    slot!(mount_path_entry),
    slot!(close_mount_entry),
    slot!(identifier_entry),
    slot!(not_implemented::<Package, 28>),
    slot!(not_implemented::<Package, 29>),
    slot!(write_stats_entry),
    slot!(not_implemented::<Package, 31>),
    slot!(not_implemented::<Package, 32>),
    slot!(enumerate_features_entry),
    slot!(uninstall_package_entry),
    slot!(not_implemented::<Package, 35>),
    slot!(not_implemented::<Package, 36>),
    slot!(mount_async_entry),
    slot!(mount_result_entry),
    slot!(enumerate_packages_entry),
    slot!(register_installed_entry),
    slot!(not_implemented::<Package, 41>),
    slot!(not_implemented::<Package, 42>),
    slot!(not_implemented::<Package, 43>),
    slot!(not_implemented::<Package, 44>),
    slot!(not_implemented::<Package, 45>),
    slot!(not_implemented::<Package, 46>),
    slot!(not_implemented::<Package, 47>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::E_NOTIMPL;
    use crate::exports::query_without_runtime;
    use crate::local_title::method;
    use crate::xasync::XAsyncGetStatus;

    type Qi = unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32;
    type Async2 = unsafe extern "system" fn(This, Id, Block) -> i32;
    type InstallAsync = unsafe extern "system" fn(This, Id, u32, *const c_void, u32, bool, Block) -> i32;
    type ResultFn = unsafe extern "system" fn(This, Block, *mut *mut c_void) -> i32;

    /// The vtable object as a title obtains it from `QueryApiImpl`.
    fn object() -> *mut c_void {
        let mut out = std::ptr::null_mut();
        let outcome = query_without_runtime(Some(&XPACKAGE_TABLE), Some(&XPACKAGE_IID), &mut out);
        assert_eq!(outcome, (S_OK, Some(NAME)));
        assert!(!out.is_null());
        out
    }

    fn own_id() -> CString {
        CString::new(package_identifier()).unwrap()
    }

    fn fresh_block() -> XAsyncBlock {
        XAsyncBlock { queue: std::ptr::null_mut(), context: std::ptr::null_mut(), callback: 0, internal: [0; 32] }
    }

    /// Runs an async slot to the end and returns the status and the handle of its result.
    fn run_async(this: *mut c_void, ask: usize, get: usize, id: &std::ffi::CStr) -> (i32, *mut c_void) {
        let none = std::ptr::null();
        let mut block = fresh_block();
        let mut handle = std::ptr::null_mut();
        // SAFETY: both slots receive the arguments of their GDK declarations and the block outlives them.
        unsafe {
            let started = if ask == 16 { method::<InstallAsync>(this, ask)(this, id.as_ptr(), 0, none, 0, true, &mut block) } else { method::<Async2>(this, ask)(this, id.as_ptr(), &mut block) };
            assert_eq!(started, S_OK);
            let status = XAsyncGetStatus(&mut block, true);
            if status == S_OK {
                assert_eq!(method::<ResultFn>(this, get)(this, &mut block, &mut handle), S_OK);
            }
            (status, handle)
        }
    }

    #[test]
    fn serves_one_api_interface_pair() {
        let stranger = Guid { data1: 0xBE3F3E9F, data2: 0, data3: 0, data4: [0; 8] };
        assert!(object_for(Some(&XPACKAGE_TABLE), Some(&XPACKAGE_IID)).is_some());
        assert!(object_for(Some(&XPACKAGE_TABLE), Some(&stranger)).is_none());
        assert!(object_for(Some(&stranger), Some(&XPACKAGE_IID)).is_none());
        assert!(object_for(None, None).is_none());
    }

    #[test]
    fn layout_matches_the_abi() {
        assert_eq!(size_of::<Methods<SLOTS>>(), SLOTS * 8);
        assert_eq!(size_of::<Progress>(), 32);
        assert_eq!(size_of::<Version>(), 8);
    }

    #[test]
    fn query_interface_returns_the_same_object() {
        let this = object();
        let mut out = std::ptr::null_mut();
        // SAFETY: slot 0 takes a GUID and an out pointer.
        let hr = unsafe { method::<Qi>(this, 0)(this, &XPACKAGE_IID, &mut out) };
        assert_eq!((hr, out), (S_OK, this));
    }

    #[test]
    fn identity_and_availability() {
        type GetId = unsafe extern "system" fn(This, usize, *mut u8) -> i32;
        type Packaged = unsafe extern "system" fn(This) -> bool;
        type Find = unsafe extern "system" fn(This, Id, u32, *const c_void, *mut u32) -> i32;
        let this = object();
        let own = own_id();
        let mut name = [0xFFu8; 16];
        let mut state = 5u32;
        // SAFETY: arguments follow the GDK declarations.
        unsafe {
            assert_eq!(method::<GetId>(this, 3)(this, name.len(), name.as_mut_ptr()), S_OK);
            assert_eq!(&name[..own.as_bytes_with_nul().len()], own.as_bytes_with_nul());
            assert!(method::<Packaged>(this, 4)(this));
            assert_eq!(method::<Find>(this, 12)(this, own.as_ptr(), 0, std::ptr::null(), &mut state), S_OK);
            assert_eq!(state, 0);
            state = 5;
            assert_eq!(method::<Find>(this, 12)(this, c"someone.else".as_ptr(), 0, std::ptr::null(), &mut state), NO_PACKAGE);
            assert_eq!(state, 5, "an unknown package leaves the out value alone");
        }
    }

    #[test]
    fn monitor_reports_a_finished_install() {
        type Create = unsafe extern "system" fn(This, Id, u32, *const c_void, u32, *mut c_void, *mut *mut c_void) -> i32;
        type Get = unsafe extern "system" fn(This, *mut c_void, *mut Progress);
        type Update = unsafe extern "system" fn(This, *mut c_void) -> bool;
        type Listen = unsafe extern "system" fn(This, *mut c_void, *mut c_void, usize, *mut u64) -> i32;
        type Estimate = unsafe extern "system" fn(This, Id, u32, *const c_void, *mut u64, *mut bool) -> i32;
        let this = object();
        let own = own_id();
        let mut mon = std::ptr::null_mut();
        let mut progress = Progress { total_bytes: 0, installed_bytes: 0, launch_bytes: 0, launchable: false, completed: false };
        let (mut token, mut bytes, mut confirm) = (0u64, 99u64, true);
        // SAFETY: arguments follow the GDK declarations.
        unsafe {
            let null = std::ptr::null_mut();
            assert_eq!(method::<Create>(this, 5)(this, own.as_ptr(), 0, std::ptr::null(), 500, null, &mut mon), S_OK);
            assert_eq!(mon, monitor_handle());
            assert_eq!(method::<Create>(this, 5)(this, c"x".as_ptr(), 0, std::ptr::null(), 500, null, &mut mon), NO_PACKAGE);
            method::<Get>(this, 7)(this, mon, &mut progress);
            assert!(progress.launchable && progress.completed);
            assert!(progress.total_bytes >= 1);
            assert_eq!([progress.installed_bytes, progress.launch_bytes], [progress.total_bytes; 2]);
            assert!(!method::<Update>(this, 8)(this, mon));
            assert_eq!(method::<Listen>(this, 9)(this, mon, null, 0, &mut token), S_OK);
            assert_ne!(token, 0);
            assert_eq!(method::<Estimate>(this, 18)(this, own.as_ptr(), 0, std::ptr::null(), &mut bytes, &mut confirm), S_OK);
            assert_eq!((bytes, confirm), (0, false));
        }
    }

    #[test]
    fn chunk_install_hands_back_the_monitor() {
        let this = object();
        let (status, handle) = run_async(this, 16, 17, &own_id());
        assert_eq!(status, S_OK);
        assert_eq!(handle, monitor_handle());
        assert_eq!(run_async(this, 16, 17, c"nope").0, NO_PACKAGE);
    }

    #[test]
    fn mounting_exposes_the_game_folder() {
        type Len = unsafe extern "system" fn(This, *mut c_void, *mut usize) -> i32;
        type Copy_ = unsafe extern "system" fn(This, *mut c_void, usize, *mut u8) -> i32;
        let this = object();
        let (status, handle) = run_async(this, 37, 38, &own_id());
        assert_eq!((status, handle), (S_OK, mount_handle()));
        // SAFETY: arguments follow the GDK declarations.
        unsafe {
            let mut len = 0;
            assert_eq!(method::<Len>(this, 24)(this, handle, &mut len), S_OK);
            let mut buffer = vec![0u8; len];
            assert_eq!(method::<Copy_>(this, 25)(this, handle, len, buffer.as_mut_ptr()), S_OK);
            assert_eq!(buffer.pop(), Some(0));
            assert_eq!(String::from_utf8(buffer).unwrap(), mount_path());
            assert_eq!(method::<Len>(this, 24)(this, monitor_handle(), &mut len), E_INVALIDARG);
        }
        assert_eq!(run_async(this, 37, 38, c"nope").0, NO_PACKAGE);
    }

    #[test]
    fn only_the_game_kind_is_enumerated() {
        use std::sync::atomic::AtomicUsize;
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "system" fn visit(_ctx: *mut c_void, details: *const PackageDetails) -> bool {
            // SAFETY: the table passes details that live for the call.
            let details = unsafe { &*details };
            assert_eq!(text(details.title_id), Some(package_identifier()));
            assert_eq!((details.kind, details.index, details.count), (KIND_GAME, 0, 1));
            CALLS.fetch_add(1, Ordering::SeqCst);
            true
        }
        type List = unsafe extern "system" fn(This, u32, u32, *mut c_void, usize) -> i32;
        let this = object();
        let callback = visit as *const () as usize;
        // SAFETY: arguments follow the GDK declaration.
        unsafe {
            let list = method::<List>(this, 39);
            assert_eq!(list(this, 1, 0, std::ptr::null_mut(), callback), S_OK);
            assert_eq!(CALLS.load(Ordering::SeqCst), 0);
            assert_eq!(list(this, 0, 0, std::ptr::null_mut(), callback), S_OK);
            assert_eq!(CALLS.load(Ordering::SeqCst), 1);
            assert_eq!(list(this, 0, 0, std::ptr::null_mut(), 0), E_INVALIDARG);
        }
    }

    #[test]
    fn unidentified_slots_are_not_implemented() {
        type Bare = unsafe extern "system" fn(This) -> i32;
        let this = object();
        for slot in [20, 21, 28, 29, 31, 35, 36, 42, 47] {
            // SAFETY: an unidentified slot only takes `this`.
            assert_eq!(unsafe { method::<Bare>(this, slot)(this) }, E_NOTIMPL, "slot {slot}");
        }
        assert_eq!(Package::slot_name(41), Some("XPackageGetPackageKind"));
        assert_eq!(Package::slot_name(20), None);
    }
}
