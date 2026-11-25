//! Offline XStore table: `api_id = {0DD112AC-7C24-448C-B92B-3960FB5BD30C}`, interface
//! `{B09D803C-2414-4A05-82C6-66DFDC9E9A44}` plus the later interfaces that extend it. The PC runtime
//! does not know our local user handle (`XStoreCreateContext` failed with `0x8924010C` in the runs
//! ), so this table answers like an offline console for a retail title the user owns.
//! Nothing is bought, no store service is contacted and no purchase is faked.
//!
//! Behaviour:
//! * Any user gets the same fixed store context handle. Closing it does nothing.
//! * The game licence is active, no trial, no expiry, with the SKU `<StoreId>/0010` of
//!   `MicrosoftGame.config`. Only the licence of the title's own package is valid. Every other
//!   package and every durable gets an invalid one, and the add-on licence list is empty.
//! * All product queries (associated, by id, entitled, current game, package, store id) finish at
//!   once with an empty query that has no product and no further page.
//! * `CanAcquireLicense` says "licensable" for our package and store id and "not licensable" for
//!   anything else.
//! * Registrations for licence changed and package licence lost return a token and never fire.
//! * Consumables have a balance of 0, and no update exists for the game or any package.
//! * Our store id maps to our package identifier, any other store id to "no package".
//! * Purchase, redeem, review, gifting and product page dialogs, consumable fulfilment, user
//!   collection and purchase ids, licence tokens and downloads need the store service and answer a
//!   named "not implemented", like every slot without a known function (see `local_title`).
//!
//! Async calls finish through the local XAsync (`xasync::begin_with`) on the title's queue. Slot
//! order and structure layouts follow the public PC GDK `xgameruntime.lib` thunks and `XStore.h`.
//! The vtable is padded to slot 87 so a later slot never reads past it. Every call is logged with
//! its arguments, bounded (`calllog`).
#![allow(unsafe_code)]

use std::ffi::{c_char, c_void};

use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::local_package::NO_PACKAGE;
use crate::local_title::{LocalTable, MAX_SLOTS, Methods, Object, no_count, not_implemented, package_identifier, put, put_string, query_interface, slot, text, title, token};
use crate::tables::Guid;
use crate::taskqueue::{E_INVALIDARG, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`.
pub(crate) const XSTORE_TABLE: Guid = Guid { data1: 0x0DD112AC, data2: 0x7C24, data3: 0x448C, data4: [0xB9, 0x2B, 0x39, 0x60, 0xFB, 0x5B, 0xD3, 0x0C] };
/// The base interface, slots 3 to 69.
pub(crate) const XSTORE_IID: Guid = Guid { data1: 0xB09D803C, data2: 0x2414, data3: 0x4A05, data4: [0x82, 0xC6, 0x66, 0xDF, 0xDC, 0x9E, 0x9A, 0x44] };
/// Later interfaces of the same table, each adding slots behind the ones before.
const XSTORE_AVAILABILITY_IID: Guid = Guid { data1: 0x60B09F4E, data2: 0x1B85, data3: 0x45B1, data4: [0x82, 0x6C, 0x16, 0x91, 0x18, 0xE2, 0x30, 0xE1] };
const XSTORE_DURABLES_IID: Guid = Guid { data1: 0x2D42FEA5, data2: 0xE71D, data3: 0x4B76, data4: [0x97, 0xCD, 0xC5, 0x0A, 0xFB, 0xB3, 0xAE, 0x5D] };
const XSTORE_STORE_ID_IID: Guid = Guid { data1: 0x5C48DEDF, data2: 0x0B67, data3: 0x4492, data4: [0xA4, 0xB5, 0x68, 0x29, 0xB8, 0xE7, 0x96, 0xE1] };

const NAME: &str = "XStore";
const SLOTS: usize = 88;
const _: () = assert!(SLOTS <= MAX_SLOTS, "every slot has a call counter");
static COUNTS: Counters<MAX_SLOTS> = Counters::new();
static OBJECT: Object = Object::new(&VTBL);

/// The object for a couple this table serves, with its name for the log.
pub(crate) fn object_for(api: Option<&Guid>, riid: Option<&Guid>) -> Option<(*mut c_void, &'static str)> {
    (api == Some(&XSTORE_TABLE) && riid.is_some_and(|r| Store::IIDS.contains(r))).then(|| (OBJECT.as_out(), NAME))
}

struct Store;

impl LocalTable for Store {
    const NAME: &'static str = NAME;
    const IIDS: &'static [Guid] = &[XSTORE_IID, XSTORE_AVAILABILITY_IID, XSTORE_DURABLES_IID, XSTORE_STORE_ID_IID];
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

/// GDK function names by slot, only for the slots that are identified.
const SLOT_NAMES: [(usize, &str); 81] = [
    (0, "QueryInterface"),
    (1, "AddRef"),
    (2, "Release"),
    (3, "XStoreCreateContext"),
    (4, "XStoreCloseContextHandle"),
    (5, "XStoreQueryAssociatedProductsAsync"),
    (6, "XStoreQueryAssociatedProductsResult"),
    (7, "XStoreQueryProductsAsync"),
    (8, "XStoreQueryProductsResult"),
    (9, "XStoreQueryEntitledProductsAsync"),
    (10, "XStoreQueryEntitledProductsResult"),
    (11, "XStoreQueryProductForCurrentGameAsync"),
    (12, "XStoreQueryProductForCurrentGameResult"),
    (13, "XStoreQueryProductForPackageAsync"),
    (14, "XStoreQueryProductForPackageResult"),
    (15, "XStoreEnumerateProductsQuery"),
    (16, "XStoreProductsQueryHasMorePages"),
    (17, "XStoreProductsQueryNextPageAsync"),
    (18, "XStoreProductsQueryNextPageResult"),
    (19, "XStoreCloseProductsQueryHandle"),
    (20, "XStoreAcquireLicenseForPackageAsync"),
    (21, "XStoreAcquireLicenseForPackageResult"),
    (22, "XStoreIsLicenseValid"),
    (23, "XStoreCloseLicenseHandle"),
    (24, "XStoreCanAcquireLicenseForStoreIdAsync"),
    (25, "XStoreCanAcquireLicenseForStoreIdResult"),
    (26, "XStoreCanAcquireLicenseForPackageAsync"),
    (27, "XStoreCanAcquireLicenseForPackageResult"),
    (28, "XStoreQueryGameLicenseAsync"),
    (29, "XStoreQueryGameLicenseResult"),
    (30, "XStoreQueryAddOnLicensesAsync"),
    (31, "XStoreQueryAddOnLicensesResultCount"),
    (32, "XStoreQueryAddOnLicensesResult"),
    (33, "XStoreQueryConsumableBalanceRemainingAsync"),
    (34, "XStoreQueryConsumableBalanceRemainingResult"),
    (35, "XStoreReportConsumableFulfillmentAsync"),
    (36, "XStoreReportConsumableFulfillmentResult"),
    (37, "XStoreGetUserCollectionsIdAsync"),
    (38, "XStoreGetUserCollectionsIdResultSize"),
    (39, "XStoreGetUserCollectionsIdResult"),
    (40, "XStoreGetUserPurchaseIdAsync"),
    (41, "XStoreGetUserPurchaseIdResultSize"),
    (42, "XStoreGetUserPurchaseIdResult"),
    (43, "XStoreQueryLicenseTokenAsync"),
    (44, "XStoreQueryLicenseTokenResultSize"),
    (45, "XStoreQueryLicenseTokenResult"),
    (49, "XStoreShowPurchaseUIAsync"),
    (50, "XStoreShowPurchaseUIResult"),
    (51, "XStoreShowRateAndReviewUIAsync"),
    (52, "XStoreShowRateAndReviewUIResult"),
    (53, "XStoreShowRedeemTokenUIAsync"),
    (54, "XStoreShowRedeemTokenUIResult"),
    (55, "XStoreQueryGameAndDlcPackageUpdatesAsync"),
    (56, "XStoreQueryGameAndDlcPackageUpdatesResultCount"),
    (57, "XStoreQueryGameAndDlcPackageUpdatesResult"),
    (58, "XStoreDownloadPackageUpdatesAsync"),
    (59, "XStoreDownloadPackageUpdatesResult"),
    (60, "XStoreDownloadAndInstallPackageUpdatesAsync"),
    (61, "XStoreDownloadAndInstallPackageUpdatesResult"),
    (62, "XStoreDownloadAndInstallPackagesAsync"),
    (63, "XStoreDownloadAndInstallPackagesResultCount"),
    (64, "XStoreDownloadAndInstallPackagesResult"),
    (65, "XStoreQueryPackageIdentifier"),
    (66, "XStoreRegisterGameLicenseChanged"),
    (67, "XStoreUnregisterGameLicenseChanged"),
    (68, "XStoreRegisterPackageLicenseLost"),
    (69, "XStoreUnregisterPackageLicenseLost"),
    (70, "XStoreIsAvailabilityPurchasable"),
    (71, "XStoreAcquireLicenseForDurablesAsync"),
    (72, "XStoreAcquireLicenseForDurablesResult"),
    (73, "XStoreShowAssociatedProductsUIAsync"),
    (74, "XStoreShowAssociatedProductsUIResult"),
    (75, "XStoreShowProductPageUIAsync"),
    (76, "XStoreShowProductPageUIResult"),
    (77, "XStoreQueryAssociatedProductsForStoreIdAsync"),
    (78, "XStoreQueryAssociatedProductsForStoreIdResult"),
    (79, "XStoreQueryPackageUpdatesAsync"),
    (80, "XStoreQueryPackageUpdatesResultCount"),
    (81, "XStoreQueryPackageUpdatesResult"),
    (82, "XStoreShowGiftingUIAsync"),
    (83, "XStoreShowGiftingUIResult"),
];

type This = *const c_void;
type Handle = *mut c_void;
type Block = *mut XAsyncBlock;
type Text = *const c_char;
type Texts = *const *const c_char;

/// `XStoreGameLicense`: 104 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GameLicense {
    pub(crate) sku_store_id: [u8; 18],
    pub(crate) is_active: bool,
    pub(crate) is_trial_owned_by_this_user: bool,
    pub(crate) is_disc_license: bool,
    pub(crate) is_trial: bool,
    pub(crate) trial_time_remaining_in_seconds: u32,
    pub(crate) trial_unique_id: [u8; 64],
    pub(crate) expiration_date: i64,
}

/// `XStoreCanAcquireLicenseResult`: 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CanAcquireLicense {
    pub(crate) licensable_sku: [u8; 18],
    pub(crate) status: u32,
}

/// `XStoreCanLicenseStatus::NotLicensableToUser`.
pub(crate) const NOT_LICENSABLE: u32 = 0;
/// `XStoreCanLicenseStatus::Licensable`.
pub(crate) const LICENSABLE: u32 = 1;

/// `XStoreConsumableResult`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConsumableResult {
    pub(crate) quantity: u32,
}

/// The SKU store ids the GDK structures hold: 17 characters and a NUL.
const SKU_LEN: usize = 18;

/// A context and an empty product query hold nothing: fixed addresses. A licence handle says by
/// its address whether it is valid.
static CONTEXT: u8 = 0;
static EMPTY_QUERY: u8 = 0;
static VALID_LICENSE: u8 = 0;
static NO_LICENSE: u8 = 0;

fn fixed_handle(of: &'static u8) -> Handle {
    (of as *const u8).cast_mut().cast()
}

/// The identities that tie a Result call to its Async call, one per result shape.
static QUERY_ID: u8 = 0;
static LICENSE_ID: u8 = 0;
static CAN_ACQUIRE_ID: u8 = 0;
static GAME_LICENSE_ID: u8 = 0;
static ADD_ONS_ID: u8 = 0;
static CONSUMABLE_ID: u8 = 0;
static UPDATES_ID: u8 = 0;

fn identity_address(of: &'static u8) -> usize {
    of as *const u8 as usize
}

/// The SKU of the game licence: the title's store id with the store's first SKU, `0010`.
fn game_sku() -> [u8; SKU_LEN] {
    let store_id = &title().store_id;
    let text = if store_id.is_empty() { String::new() } else { format!("{store_id}/0010") };
    let mut sku = [0u8; SKU_LEN];
    let n = text.len().min(SKU_LEN - 1);
    sku[..n].copy_from_slice(&text.as_bytes()[..n]);
    sku
}

/// The game licence of a retail title the user owns: active, not a trial, no expiry.
pub(crate) fn game_license() -> GameLicense {
    GameLicense {
        sku_store_id: game_sku(),
        is_active: true,
        is_trial_owned_by_this_user: false,
        is_disc_license: false,
        is_trial: false,
        trial_time_remaining_in_seconds: 0,
        trial_unique_id: [0; 64],
        expiration_date: 0,
    }
}

/// The bytes of a plain `repr(C)` value, for an async result.
fn raw_bytes<T: Copy>(value: &T) -> Vec<u8> {
    let mut out = vec![0u8; size_of::<T>()];
    // SAFETY: `out` holds `size_of::<T>()` bytes; the value is copied whole, padding included
    // (zero in our values, which are built field by field from zeroed arrays).
    unsafe { out.as_mut_ptr().cast::<T>().write_unaligned(*value) };
    out
}

/// Whether `id` names this title's package, logged when not.
fn names_our_package(id: Text) -> bool {
    let asked = text(id);
    let yes = asked.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(&package_identifier()));
    if !yes {
        log(&format!("local {NAME} table: package {asked:?} is not this title's -> no licence"));
    }
    yes
}

/// Whether `id` is this title's store id, logged when not.
fn names_our_store_id(id: Text) -> bool {
    let asked = text(id);
    let ours = &title().store_id;
    let yes = !ours.is_empty() && asked.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(ours));
    if !yes {
        log(&format!("local {NAME} table: store id {asked:?} is not this title's ({ours:?}) -> not owned"));
    }
    yes
}

/// Completes `block` at once with `result` under `identity`.
unsafe fn finish(block: Block, identity: &'static u8, result: Result<Vec<u8>, i32>) -> i32 {
    // SAFETY: the title's block, valid until the call completes.
    unsafe { xa::begin_with(block, identity_address(identity), move || result) }
}

/// Reads a result of `size` bytes into `out`.
unsafe fn take_result(block: Block, identity: &'static u8, size: usize, out: *mut c_void) -> i32 {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block and a location of `size` bytes.
    unsafe { xa::XAsyncGetResult(block, (identity as *const u8).cast(), size, out, std::ptr::null_mut()) }
}

logged_shim!(NAME, COUNTS, This, 3 as "XStoreCreateContext", create_context_entry => open_context, (user: *mut c_void, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 4 as "XStoreCloseContextHandle", close_context_entry => ignore_handle, (context: Handle));
logged_shim!(NAME, COUNTS, This, 5 as "XStoreQueryAssociatedProductsAsync", associated_entry => query_paged, (context: Handle, kinds: u32, per_page: u32, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 6 as "XStoreQueryAssociatedProductsResult", associated_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 7 as "XStoreQueryProductsAsync", products_entry => query_by_ids, (context: Handle, kinds: u32, store_ids: Texts, store_id_count: usize, filters: Texts, filter_count: usize, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 8 as "XStoreQueryProductsResult", products_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 9 as "XStoreQueryEntitledProductsAsync", entitled_entry => query_paged, (context: Handle, kinds: u32, per_page: u32, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 10 as "XStoreQueryEntitledProductsResult", entitled_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 11 as "XStoreQueryProductForCurrentGameAsync", current_game_entry => query_current_game, (context: Handle, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 12 as "XStoreQueryProductForCurrentGameResult", current_game_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 13 as "XStoreQueryProductForPackageAsync", for_package_entry => query_package_product, (context: Handle, kinds: u32, package: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 14 as "XStoreQueryProductForPackageResult", for_package_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 15 as "XStoreEnumerateProductsQuery", enumerate_entry => visit_no_products, (query: Handle, context: *mut c_void, callback: usize) -> i32);
logged_shim!(NAME, COUNTS, This, 16 as "XStoreProductsQueryHasMorePages", more_pages_entry => query_is_last_page, (query: Handle) -> bool);
logged_shim!(NAME, COUNTS, This, 17 as "XStoreProductsQueryNextPageAsync", next_page_entry => next_empty_page, (query: Handle, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 18 as "XStoreProductsQueryNextPageResult", next_page_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 19 as "XStoreCloseProductsQueryHandle", close_query_entry => ignore_handle, (query: Handle));
logged_shim!(NAME, COUNTS, This, 20 as "XStoreAcquireLicenseForPackageAsync", license_for_package_entry => license_of_package, (context: Handle, package: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 21 as "XStoreAcquireLicenseForPackageResult", license_for_package_result_entry => take_license, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 22 as "XStoreIsLicenseValid", is_license_valid_entry => license_is_valid, (license: Handle) -> bool);
logged_shim!(NAME, COUNTS, This, 23 as "XStoreCloseLicenseHandle", close_license_entry => ignore_handle, (license: Handle));
logged_shim!(NAME, COUNTS, This, 24 as "XStoreCanAcquireLicenseForStoreIdAsync", can_acquire_store_id_entry => can_acquire_by_store_id, (context: Handle, store_id: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 25 as "XStoreCanAcquireLicenseForStoreIdResult", can_acquire_store_id_result_entry => take_can_acquire, (block: Block, out: *mut CanAcquireLicense) -> i32);
logged_shim!(NAME, COUNTS, This, 26 as "XStoreCanAcquireLicenseForPackageAsync", can_acquire_package_entry => can_acquire_by_package, (context: Handle, package: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 27 as "XStoreCanAcquireLicenseForPackageResult", can_acquire_package_result_entry => take_can_acquire, (block: Block, out: *mut CanAcquireLicense) -> i32);
logged_shim!(NAME, COUNTS, This, 28 as "XStoreQueryGameLicenseAsync", game_license_entry => start_game_license, (context: Handle, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 29 as "XStoreQueryGameLicenseResult", game_license_result_entry => take_game_license, (block: Block, out: *mut GameLicense) -> i32);
logged_shim!(NAME, COUNTS, This, 30 as "XStoreQueryAddOnLicensesAsync", add_ons_entry => start_add_ons, (context: Handle, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 31 as "XStoreQueryAddOnLicensesResultCount", add_ons_count_entry => add_ons_total, (block: Block, count: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 32 as "XStoreQueryAddOnLicensesResult", add_ons_result_entry => add_ons_copy, (block: Block, count: u32, licenses: *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 33 as "XStoreQueryConsumableBalanceRemainingAsync", consumable_entry => start_consumable, (context: Handle, store_id: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 34 as "XStoreQueryConsumableBalanceRemainingResult", consumable_result_entry => take_consumable, (block: Block, out: *mut ConsumableResult) -> i32);
logged_shim!(NAME, COUNTS, This, 55 as "XStoreQueryGameAndDlcPackageUpdatesAsync", game_updates_entry => start_game_updates, (context: Handle, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 56 as "XStoreQueryGameAndDlcPackageUpdatesResultCount", game_updates_count_entry => updates_total, (block: Block, count: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 57 as "XStoreQueryGameAndDlcPackageUpdatesResult", game_updates_result_entry => updates_copy, (block: Block, count: u32, updates: *mut c_void) -> i32);
logged_shim!(NAME, COUNTS, This, 65 as "XStoreQueryPackageIdentifier", package_identifier_entry => identifier_of_store_id, (store_id: Text, size: usize, buffer: *mut u8) -> i32);
logged_shim!(NAME, COUNTS, This, 66 as "XStoreRegisterGameLicenseChanged", register_license_changed_entry => listen_never_fires, (context: Handle, queue: *mut c_void, callback_context: *mut c_void, callback: usize, out: *mut u64) -> i32);
logged_shim!(NAME, COUNTS, This, 67 as "XStoreUnregisterGameLicenseChanged", unregister_license_changed_entry => stop_listening, (context: Handle, token: u64, wait: bool) -> bool);
logged_shim!(NAME, COUNTS, This, 68 as "XStoreRegisterPackageLicenseLost", register_license_lost_entry => listen_never_fires, (license: Handle, queue: *mut c_void, callback_context: *mut c_void, callback: usize, out: *mut u64) -> i32);
logged_shim!(NAME, COUNTS, This, 69 as "XStoreUnregisterPackageLicenseLost", unregister_license_lost_entry => stop_listening, (license: Handle, token: u64, wait: bool) -> bool);
logged_shim!(NAME, COUNTS, This, 71 as "XStoreAcquireLicenseForDurablesAsync", license_for_durables_entry => license_of_durable, (context: Handle, store_id: Text, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 72 as "XStoreAcquireLicenseForDurablesResult", license_for_durables_result_entry => take_license, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 77 as "XStoreQueryAssociatedProductsForStoreIdAsync", associated_for_store_id_entry => query_associated_for_id, (context: Handle, store_id: Text, kinds: u32, per_page: u32, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 78 as "XStoreQueryAssociatedProductsForStoreIdResult", associated_for_store_id_result_entry => take_query, (block: Block, out: *mut Handle) -> i32);
logged_shim!(NAME, COUNTS, This, 79 as "XStoreQueryPackageUpdatesAsync", package_updates_entry => start_package_updates, (context: Handle, packages: Texts, count: usize, block: Block) -> i32);
logged_shim!(NAME, COUNTS, This, 80 as "XStoreQueryPackageUpdatesResultCount", package_updates_count_entry => updates_total, (block: Block, count: *mut u32) -> i32);
logged_shim!(NAME, COUNTS, This, 81 as "XStoreQueryPackageUpdatesResult", package_updates_result_entry => updates_copy, (block: Block, count: u32, updates: *mut c_void) -> i32);

/// One context for any user: the store answers do not depend on who asks.
unsafe fn open_context(_user: *mut c_void, out: *mut Handle) -> i32 {
    put(out, fixed_handle(&CONTEXT))
}

unsafe fn ignore_handle(_handle: Handle) {}

/// An empty query, at once.
unsafe fn finish_with_empty_query(block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &QUERY_ID, Ok((fixed_handle(&EMPTY_QUERY) as usize).to_le_bytes().to_vec())) }
}

unsafe fn query_paged(_context: Handle, _kinds: u32, _per_page: u32, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

unsafe fn query_by_ids(_context: Handle, _kinds: u32, _store_ids: Texts, _store_id_count: usize, _filters: Texts, _filter_count: usize, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

unsafe fn query_current_game(_context: Handle, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

unsafe fn query_package_product(_context: Handle, _kinds: u32, _package: Text, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

unsafe fn query_associated_for_id(_context: Handle, _store_id: Text, _kinds: u32, _per_page: u32, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

/// A query has one page and it is empty: asking for the next one gives the same empty query.
unsafe fn next_empty_page(_query: Handle, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_with_empty_query(block) }
}

unsafe fn take_query(block: Block, out: *mut Handle) -> i32 {
    // SAFETY: the title's block and an eight byte location.
    unsafe { take_result(block, &QUERY_ID, size_of::<Handle>(), out.cast()) }
}

/// No product: the callback never runs.
unsafe fn visit_no_products(_query: Handle, _context: *mut c_void, callback: usize) -> i32 {
    if callback == 0 { E_INVALIDARG } else { S_OK }
}

unsafe fn query_is_last_page(_query: Handle) -> bool {
    false
}

unsafe fn finish_license(block: Block, valid: bool) -> i32 {
    let license = fixed_handle(if valid { &VALID_LICENSE } else { &NO_LICENSE });
    // SAFETY: the title's block.
    unsafe { finish(block, &LICENSE_ID, Ok((license as usize).to_le_bytes().to_vec())) }
}

/// Only the title's own package is installed and owned: only its licence is valid.
unsafe fn license_of_package(_context: Handle, package: Text, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_license(block, names_our_package(package)) }
}

/// The user owns no durable.
unsafe fn license_of_durable(_context: Handle, store_id: Text, block: Block) -> i32 {
    log(&format!("local {NAME} table: durable {:?} is not owned", text(store_id)));
    // SAFETY: the title's block.
    unsafe { finish_license(block, false) }
}

unsafe fn take_license(block: Block, out: *mut Handle) -> i32 {
    // SAFETY: the title's block and an eight byte location.
    unsafe { take_result(block, &LICENSE_ID, size_of::<Handle>(), out.cast()) }
}

unsafe fn license_is_valid(license: Handle) -> bool {
    license == fixed_handle(&VALID_LICENSE)
}

unsafe fn finish_can_acquire(block: Block, owned: bool) -> i32 {
    let result = if owned { CanAcquireLicense { licensable_sku: game_sku(), status: LICENSABLE } } else { CanAcquireLicense { licensable_sku: [0; SKU_LEN], status: NOT_LICENSABLE } };
    // SAFETY: the title's block.
    unsafe { finish(block, &CAN_ACQUIRE_ID, Ok(raw_bytes(&result))) }
}

unsafe fn can_acquire_by_store_id(_context: Handle, store_id: Text, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_can_acquire(block, names_our_store_id(store_id)) }
}

unsafe fn can_acquire_by_package(_context: Handle, package: Text, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish_can_acquire(block, names_our_package(package)) }
}

unsafe fn take_can_acquire(block: Block, out: *mut CanAcquireLicense) -> i32 {
    // SAFETY: the title's block and a structure location.
    unsafe { take_result(block, &CAN_ACQUIRE_ID, size_of::<CanAcquireLicense>(), out.cast()) }
}

unsafe fn start_game_license(_context: Handle, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &GAME_LICENSE_ID, Ok(raw_bytes(&game_license()))) }
}

unsafe fn take_game_license(block: Block, out: *mut GameLicense) -> i32 {
    // SAFETY: the title's block and a structure location.
    unsafe { take_result(block, &GAME_LICENSE_ID, size_of::<GameLicense>(), out.cast()) }
}

/// No add-on: an empty list.
unsafe fn start_add_ons(_context: Handle, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &ADD_ONS_ID, Ok(Vec::new())) }
}

/// The count of an empty list, once the call completed.
unsafe fn count_of_nothing(block: Block, count: *mut u32) -> i32 {
    // SAFETY: the title's block.
    let status = unsafe { xa::XAsyncGetStatus(block, false) };
    if status < 0 {
        return status;
    }
    put(count, 0)
}

/// An empty list into the title's array: nothing to copy, the call's status.
unsafe fn copy_of_nothing(block: Block, count: u32) -> i32 {
    // SAFETY: the title's block.
    let status = unsafe { xa::XAsyncGetStatus(block, false) };
    if status < 0 {
        status
    } else if count != 0 {
        E_INVALIDARG
    } else {
        S_OK
    }
}

unsafe fn add_ons_total(block: Block, count: *mut u32) -> i32 {
    // SAFETY: the title's block.
    unsafe { count_of_nothing(block, count) }
}

unsafe fn add_ons_copy(block: Block, count: u32, _licenses: *mut c_void) -> i32 {
    // SAFETY: the title's block.
    unsafe { copy_of_nothing(block, count) }
}

/// No consumable is owned: a balance of 0.
unsafe fn start_consumable(_context: Handle, _store_id: Text, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &CONSUMABLE_ID, Ok(raw_bytes(&ConsumableResult { quantity: 0 }))) }
}

unsafe fn take_consumable(block: Block, out: *mut ConsumableResult) -> i32 {
    // SAFETY: the title's block and a structure location.
    unsafe { take_result(block, &CONSUMABLE_ID, size_of::<ConsumableResult>(), out.cast()) }
}

/// No update for the game or any package.
unsafe fn start_game_updates(_context: Handle, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &UPDATES_ID, Ok(Vec::new())) }
}

unsafe fn start_package_updates(_context: Handle, _packages: Texts, _count: usize, block: Block) -> i32 {
    // SAFETY: the title's block.
    unsafe { finish(block, &UPDATES_ID, Ok(Vec::new())) }
}

unsafe fn updates_total(block: Block, count: *mut u32) -> i32 {
    // SAFETY: the title's block.
    unsafe { count_of_nothing(block, count) }
}

unsafe fn updates_copy(block: Block, count: u32, _updates: *mut c_void) -> i32 {
    // SAFETY: the title's block.
    unsafe { copy_of_nothing(block, count) }
}

/// The title's store id is its installed package. Nothing else is installed.
unsafe fn identifier_of_store_id(store_id: Text, size: usize, buffer: *mut u8) -> i32 {
    if names_our_store_id(store_id) { put_string(&package_identifier(), size, buffer, std::ptr::null_mut()) } else { NO_PACKAGE }
}

/// Licences never change offline: accepted, never fired.
unsafe fn listen_never_fires(_handle: Handle, _queue: *mut c_void, _context: *mut c_void, callback: usize, out: *mut u64) -> i32 {
    if callback == 0 {
        return E_INVALIDARG;
    }
    put(out, token())
}

unsafe fn stop_listening(_handle: Handle, _token: u64, _wait: bool) -> bool {
    true
}

static VTBL: Methods<SLOTS> = Methods([
    slot!(query_interface::<Store>),
    slot!(no_count),
    slot!(no_count),
    slot!(create_context_entry),
    slot!(close_context_entry),
    slot!(associated_entry),
    slot!(associated_result_entry),
    slot!(products_entry),
    slot!(products_result_entry),
    slot!(entitled_entry),
    slot!(entitled_result_entry),
    slot!(current_game_entry),
    slot!(current_game_result_entry),
    slot!(for_package_entry),
    slot!(for_package_result_entry),
    slot!(enumerate_entry),
    slot!(more_pages_entry),
    slot!(next_page_entry),
    slot!(next_page_result_entry),
    slot!(close_query_entry),
    slot!(license_for_package_entry),
    slot!(license_for_package_result_entry),
    slot!(is_license_valid_entry),
    slot!(close_license_entry),
    slot!(can_acquire_store_id_entry),
    slot!(can_acquire_store_id_result_entry),
    slot!(can_acquire_package_entry),
    slot!(can_acquire_package_result_entry),
    slot!(game_license_entry),
    slot!(game_license_result_entry),
    slot!(add_ons_entry),
    slot!(add_ons_count_entry),
    slot!(add_ons_result_entry),
    slot!(consumable_entry),
    slot!(consumable_result_entry),
    slot!(not_implemented::<Store, 35>),
    slot!(not_implemented::<Store, 36>),
    slot!(not_implemented::<Store, 37>),
    slot!(not_implemented::<Store, 38>),
    slot!(not_implemented::<Store, 39>),
    slot!(not_implemented::<Store, 40>),
    slot!(not_implemented::<Store, 41>),
    slot!(not_implemented::<Store, 42>),
    slot!(not_implemented::<Store, 43>),
    slot!(not_implemented::<Store, 44>),
    slot!(not_implemented::<Store, 45>),
    slot!(not_implemented::<Store, 46>),
    slot!(not_implemented::<Store, 47>),
    slot!(not_implemented::<Store, 48>),
    slot!(not_implemented::<Store, 49>),
    slot!(not_implemented::<Store, 50>),
    slot!(not_implemented::<Store, 51>),
    slot!(not_implemented::<Store, 52>),
    slot!(not_implemented::<Store, 53>),
    slot!(not_implemented::<Store, 54>),
    slot!(game_updates_entry),
    slot!(game_updates_count_entry),
    slot!(game_updates_result_entry),
    slot!(not_implemented::<Store, 58>),
    slot!(not_implemented::<Store, 59>),
    slot!(not_implemented::<Store, 60>),
    slot!(not_implemented::<Store, 61>),
    slot!(not_implemented::<Store, 62>),
    slot!(not_implemented::<Store, 63>),
    slot!(not_implemented::<Store, 64>),
    slot!(package_identifier_entry),
    slot!(register_license_changed_entry),
    slot!(unregister_license_changed_entry),
    slot!(register_license_lost_entry),
    slot!(unregister_license_lost_entry),
    slot!(not_implemented::<Store, 70>),
    slot!(license_for_durables_entry),
    slot!(license_for_durables_result_entry),
    slot!(not_implemented::<Store, 73>),
    slot!(not_implemented::<Store, 74>),
    slot!(not_implemented::<Store, 75>),
    slot!(not_implemented::<Store, 76>),
    slot!(associated_for_store_id_entry),
    slot!(associated_for_store_id_result_entry),
    slot!(package_updates_entry),
    slot!(package_updates_count_entry),
    slot!(package_updates_result_entry),
    slot!(not_implemented::<Store, 82>),
    slot!(not_implemented::<Store, 83>),
    slot!(not_implemented::<Store, 84>),
    slot!(not_implemented::<Store, 85>),
    slot!(not_implemented::<Store, 86>),
    slot!(not_implemented::<Store, 87>),
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::E_NOTIMPL;
    use crate::exports::query_without_runtime;
    use crate::local_title::method;
    use crate::xasync::XAsyncGetStatus;
    use std::ffi::CString;

    /// The table as the title gets it from `QueryApiImpl`.
    fn table() -> *mut c_void {
        let mut out = std::ptr::null_mut();
        assert_eq!(query_without_runtime(Some(&XSTORE_TABLE), Some(&XSTORE_IID), &mut out), (S_OK, Some(NAME)));
        assert!(!out.is_null());
        out
    }

    fn block() -> XAsyncBlock {
        XAsyncBlock { queue: std::ptr::null_mut(), context: std::ptr::null_mut(), callback: 0, internal: [0; 32] }
    }

    type CreateContext = unsafe extern "system" fn(This, *mut c_void, *mut Handle) -> i32;
    type Close = unsafe extern "system" fn(This, Handle);
    type ContextAsync = unsafe extern "system" fn(This, Handle, Block) -> i32;
    type TextAsync = unsafe extern "system" fn(This, Handle, Text, Block) -> i32;
    type HandleResult = unsafe extern "system" fn(This, Block, *mut Handle) -> i32;
    type Count = unsafe extern "system" fn(This, Block, *mut u32) -> i32;
    type List = unsafe extern "system" fn(This, Block, u32, *mut c_void) -> i32;

    /// A context, as the title opens it with its user handle.
    fn context(t: *mut c_void) -> Handle {
        let mut c = std::ptr::null_mut();
        let user = 0x7FFF_0000_1000usize as *mut c_void;
        // SAFETY: slot 3 with a user handle and an out location.
        assert_eq!(unsafe { method::<CreateContext>(t, 3)(t, user, &mut c) }, S_OK);
        assert!(!c.is_null());
        c
    }

    #[test]
    fn the_store_pairs_are_served_and_the_layouts_match_the_gdk() {
        for riid in Store::IIDS {
            assert!(object_for(Some(&XSTORE_TABLE), Some(riid)).is_some());
        }
        assert!(object_for(Some(&XSTORE_TABLE), Some(&XSTORE_TABLE)).is_none());
        assert!(object_for(Some(&XSTORE_IID), Some(&XSTORE_IID)).is_none());
        assert!(object_for(Some(&XSTORE_TABLE), None).is_none());
        assert_eq!(size_of::<Methods<SLOTS>>(), SLOTS * 8);
        assert_eq!(size_of::<GameLicense>(), 104);
        assert_eq!(std::mem::offset_of!(GameLicense, trial_time_remaining_in_seconds), 24);
        assert_eq!(std::mem::offset_of!(GameLicense, expiration_date), 96);
        assert_eq!(size_of::<CanAcquireLicense>(), 24);
        assert_eq!(std::mem::offset_of!(CanAcquireLicense, status), 20);
        assert_eq!(size_of::<ConsumableResult>(), 4);
        type Qi = unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32;
        let t = table();
        let mut out = std::ptr::null_mut();
        // SAFETY: slot 0 with a GUID and an out location.
        assert_eq!(unsafe { method::<Qi>(t, 0)(t, &XSTORE_DURABLES_IID, &mut out) }, S_OK);
        assert_eq!(out, t);
    }

    #[test]
    fn the_context_opens_and_the_game_licence_is_full_and_owned() {
        let t = table();
        let c = context(t);
        type LicenseResult = unsafe extern "system" fn(This, Block, *mut GameLicense) -> i32;
        let mut b = block();
        // SAFETY: the slots get their GDK arguments, the block outlives the call.
        unsafe {
            assert_eq!(method::<ContextAsync>(t, 28)(t, c, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut l = std::mem::zeroed::<GameLicense>();
            l.is_trial = true;
            assert_eq!(method::<LicenseResult>(t, 29)(t, &mut b, &mut l), S_OK);
            assert_eq!(l, game_license());
            assert!(l.is_active && !l.is_trial && !l.is_trial_owned_by_this_user && l.expiration_date == 0);
            method::<Close>(t, 4)(t, c);
        }
    }

    #[test]
    fn our_package_is_licensed_other_packages_and_durables_are_not() {
        let t = table();
        let c = context(t);
        type IsValid = unsafe extern "system" fn(This, Handle) -> bool;
        let ours = CString::new(package_identifier()).unwrap();
        // SAFETY: the slots get their GDK arguments, the blocks outlive the calls.
        unsafe {
            for (slot, id, valid) in [(20, ours.as_ptr(), true), (20, c"OTHER".as_ptr(), false), (71, c"9DURABLE0000".as_ptr(), false)] {
                let mut b = block();
                assert_eq!(method::<TextAsync>(t, slot)(t, c, id, &mut b), S_OK);
                assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
                let mut l = std::ptr::null_mut();
                assert_eq!(method::<HandleResult>(t, slot + 1)(t, &mut b, &mut l), S_OK);
                assert_eq!(method::<IsValid>(t, 22)(t, l), valid, "slot {slot}");
                method::<Close>(t, 23)(t, l);
            }
        }
    }

    #[test]
    fn can_acquire_is_licensable_for_our_package_only() {
        let t = table();
        let c = context(t);
        type Result_ = unsafe extern "system" fn(This, Block, *mut CanAcquireLicense) -> i32;
        let ours = CString::new(package_identifier()).unwrap();
        // SAFETY: the slots get their GDK arguments, the blocks outlive the calls.
        unsafe {
            for (slot, id, status) in [(26, ours.as_ptr(), LICENSABLE), (26, c"OTHER".as_ptr(), NOT_LICENSABLE), (24, c"9NOTOURS0000".as_ptr(), NOT_LICENSABLE)] {
                let mut b = block();
                assert_eq!(method::<TextAsync>(t, slot)(t, c, id, &mut b), S_OK);
                assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
                let mut r = CanAcquireLicense { licensable_sku: [7; SKU_LEN], status: 9 };
                assert_eq!(method::<Result_>(t, slot + 1)(t, &mut b, &mut r), S_OK);
                assert_eq!(r.status, status, "slot {slot}");
            }
        }
    }

    #[test]
    fn product_queries_complete_at_once_with_one_empty_page() {
        unsafe extern "system" fn never(_product: *const c_void, _context: *mut c_void) -> bool {
            panic!("an empty query has no product");
        }
        let t = table();
        let c = context(t);
        type Paged = unsafe extern "system" fn(This, Handle, u32, u32, Block) -> i32;
        type ById = unsafe extern "system" fn(This, Handle, u32, Texts, usize, Texts, usize, Block) -> i32;
        type ForPackage = unsafe extern "system" fn(This, Handle, u32, Text, Block) -> i32;
        type ForStoreId = unsafe extern "system" fn(This, Handle, Text, u32, u32, Block) -> i32;
        type Enumerate = unsafe extern "system" fn(This, Handle, *mut c_void, usize) -> i32;
        type HasMore = unsafe extern "system" fn(This, Handle) -> bool;
        type NextPage = unsafe extern "system" fn(This, Handle, Block) -> i32;
        let ids = [c"9ABC".as_ptr()];
        // SAFETY: the slots get their GDK arguments, the blocks outlive the calls. A block is known
        // by its address while its call is pending, so each one is boxed before its call.
        unsafe {
            let mut blocks: Vec<(usize, Box<XAsyncBlock>)> = Vec::new();
            for slot in [5, 9] {
                let mut b = Box::new(block());
                assert_eq!(method::<Paged>(t, slot)(t, c, 0xFF, 25, &mut *b), S_OK);
                blocks.push((slot + 1, b));
            }
            let mut b = Box::new(block());
            assert_eq!(method::<ById>(t, 7)(t, c, 0xFF, ids.as_ptr(), 1, std::ptr::null(), 0, &mut *b), S_OK);
            blocks.push((8, b));
            let mut b = Box::new(block());
            assert_eq!(method::<ContextAsync>(t, 11)(t, c, &mut *b), S_OK);
            blocks.push((12, b));
            let mut b = Box::new(block());
            assert_eq!(method::<ForPackage>(t, 13)(t, c, 0xFF, c"P".as_ptr(), &mut *b), S_OK);
            blocks.push((14, b));
            let mut b = Box::new(block());
            assert_eq!(method::<ForStoreId>(t, 77)(t, c, c"9ABC".as_ptr(), 0xFF, 25, &mut *b), S_OK);
            blocks.push((78, b));
            for (result_entry, b) in &mut blocks {
                assert_eq!(XAsyncGetStatus(&mut **b, true), S_OK);
                let mut q = std::ptr::null_mut();
                assert_eq!(method::<HandleResult>(t, *result_entry)(t, &mut **b, &mut q), S_OK, "slot {result_entry}");
                assert_eq!(method::<Enumerate>(t, 15)(t, q, std::ptr::null_mut(), never as *const () as usize), S_OK);
                assert!(!method::<HasMore>(t, 16)(t, q));
                let mut next = block();
                assert_eq!(method::<NextPage>(t, 17)(t, q, &mut next), S_OK);
                assert_eq!(XAsyncGetStatus(&mut next, true), S_OK);
                let mut q2 = std::ptr::null_mut();
                assert_eq!(method::<HandleResult>(t, 18)(t, &mut next, &mut q2), S_OK);
                method::<Close>(t, 19)(t, q2);
                method::<Close>(t, 19)(t, q);
            }
        }
    }

    #[test]
    fn no_add_on_no_consumable_and_no_update() {
        let t = table();
        let c = context(t);
        type Consumable = unsafe extern "system" fn(This, Block, *mut ConsumableResult) -> i32;
        type Updates = unsafe extern "system" fn(This, Handle, Texts, usize, Block) -> i32;
        // SAFETY: the slots get their GDK arguments, the blocks outlive the calls.
        unsafe {
            for (begin, count, list) in [(30, 31, 32), (55, 56, 57)] {
                let mut b = block();
                assert_eq!(method::<ContextAsync>(t, begin)(t, c, &mut b), S_OK);
                assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
                let mut n = 5;
                assert_eq!(method::<Count>(t, count)(t, &mut b, &mut n), S_OK);
                assert_eq!(n, 0, "slot {count}");
                assert_eq!(method::<List>(t, list)(t, &mut b, 0, std::ptr::null_mut()), S_OK);
            }
            let ids = [c"P".as_ptr()];
            let mut b = block();
            assert_eq!(method::<Updates>(t, 79)(t, c, ids.as_ptr(), 1, &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut n = 5;
            assert_eq!(method::<Count>(t, 80)(t, &mut b, &mut n), S_OK);
            assert_eq!(n, 0);

            let mut b = block();
            assert_eq!(method::<TextAsync>(t, 33)(t, c, c"9COINS".as_ptr(), &mut b), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
            let mut r = ConsumableResult { quantity: 9 };
            assert_eq!(method::<Consumable>(t, 34)(t, &mut b, &mut r), S_OK);
            assert_eq!(r.quantity, 0);
        }
    }

    #[test]
    fn licence_events_are_accepted_and_never_fire() {
        let t = table();
        let c = context(t);
        type Register = unsafe extern "system" fn(This, Handle, *mut c_void, *mut c_void, usize, *mut u64) -> i32;
        type Unregister = unsafe extern "system" fn(This, Handle, u64, bool) -> bool;
        // SAFETY: the slots get their GDK arguments.
        unsafe {
            for slot in [66, 68] {
                let mut token = 0;
                assert_eq!(method::<Register>(t, slot)(t, c, std::ptr::null_mut(), std::ptr::null_mut(), 1, &mut token), S_OK);
                assert_ne!(token, 0);
                assert!(method::<Unregister>(t, slot + 1)(t, c, token, true));
                assert_eq!(method::<Register>(t, slot)(t, c, std::ptr::null_mut(), std::ptr::null_mut(), 0, &mut token), E_INVALIDARG);
            }
        }
    }

    #[test]
    fn another_store_id_has_no_package() {
        let t = table();
        type Identifier = unsafe extern "system" fn(This, Text, usize, *mut u8) -> i32;
        let mut buffer = [0u8; 33];
        // SAFETY: slot 65 with a string and a buffer of its size.
        assert_eq!(unsafe { method::<Identifier>(t, 65)(t, c"9NOTOURS0000".as_ptr(), buffer.len(), buffer.as_mut_ptr()) }, NO_PACKAGE);
    }

    #[test]
    fn store_service_and_unknown_slots_answer_not_implemented() {
        let t = table();
        type Unknown = unsafe extern "system" fn(This) -> i32;
        for slot in [35, 37, 43, 46, 47, 48, 49, 53, 58, 62, 70, 73, 75, 82, 83, 87] {
            // SAFETY: a slot not served takes `this` only and reads nothing else.
            assert_eq!(unsafe { method::<Unknown>(t, slot)(t) }, E_NOTIMPL, "slot {slot}");
        }
        assert_eq!(Store::slot_name(49), Some("XStoreShowPurchaseUIAsync"));
        assert_eq!(Store::slot_name(47), None);
    }
}
