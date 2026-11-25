//! The local XAsync / XTaskQueue / XThread table, `api_id = riid = {073B7DCB-1FCF-4030-94BE-E3C9EB623428}`,
//! which `QueryApiImpl` serves only when there is no PC runtime. Every other table stays refused.
//!
//! The object is a pointer to a vtable of 32 slots. The public PC GDK's `xgameruntime.lib` thunks
//! call slot N as `vtbl[N](this, original arguments...)`, so each slot takes `this` first and
//! drops it. Slots 0 to 2 are IUnknown on a static object: the
//! reference count is a no-op answering 1, and QueryInterface answers `{073B7DCB-...}` with the
//! object itself, anything else with `E_NOINTERFACE`. Slots 8 and 31 are not known: they log
//! their index and answer `E_NOTIMPL`. The functions behind the slots are in `xasync` and
//! `taskqueue`, and the four XThread functions below. Every slot call is
//! logged with its arguments and its result, bounded (`calllog`).
//!
#![allow(unsafe_code, non_snake_case)]

use std::cell::Cell;
use std::ffi::c_void;

use crate::E_NOTIMPL;
use crate::calllog::{Counters, logged_shim};
use crate::exports::log;
use crate::tables::Guid;
use crate::taskqueue::{self as tq, RegistrationToken, S_OK};
use crate::xasync::{self as xa, XAsyncBlock};

/// The table's `api_id`, also its only `riid`.
pub(crate) const XASYNC_TABLE: Guid = Guid { data1: 0x073B7DCB, data2: 0x1FCF, data3: 0x4030, data4: [0x94, 0xBE, 0xE3, 0xC9, 0xEB, 0x62, 0x34, 0x28] };
const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_POINTER: i32 = 0x8000_4003_u32 as i32;

/// True for the one couple served without a PC runtime.
pub(crate) fn serves(api: Option<&Guid>, riid: Option<&Guid>) -> bool {
    api == Some(&XASYNC_TABLE) && riid == Some(&XASYNC_TABLE)
}

type This = *const Object;

/// The vtable, in slot order.
#[repr(C)]
pub(crate) struct Vtbl {
    query_interface: unsafe extern "system" fn(This, *const Guid, *mut *mut c_void) -> i32,
    add_ref: unsafe extern "system" fn(This) -> u32,
    release: unsafe extern "system" fn(This) -> u32,
    get_status: unsafe extern "system" fn(This, *mut XAsyncBlock, bool) -> i32,
    get_result_size: unsafe extern "system" fn(This, *mut XAsyncBlock, *mut usize) -> i32,
    cancel: unsafe extern "system" fn(This, *mut XAsyncBlock),
    run: unsafe extern "system" fn(This, *mut XAsyncBlock, usize) -> i32,
    begin: unsafe extern "system" fn(This, *mut XAsyncBlock, *mut c_void, *const c_void, *const u8, usize) -> i32,
    slot8: unsafe extern "system" fn(This) -> i32,
    schedule: unsafe extern "system" fn(This, *mut XAsyncBlock, u32) -> i32,
    complete: unsafe extern "system" fn(This, *mut XAsyncBlock, i32, usize),
    get_result: unsafe extern "system" fn(This, *mut XAsyncBlock, *const c_void, usize, *mut c_void, *mut usize) -> i32,
    queue_create: unsafe extern "system" fn(This, u32, u32, *mut *mut c_void) -> i32,
    queue_create_composite: unsafe extern "system" fn(This, *mut c_void, *mut c_void, *mut *mut c_void) -> i32,
    queue_get_port: unsafe extern "system" fn(This, *mut c_void, u32, *mut *mut c_void) -> i32,
    queue_duplicate_handle: unsafe extern "system" fn(This, *mut c_void, *mut *mut c_void) -> i32,
    queue_dispatch: unsafe extern "system" fn(This, *mut c_void, u32, u32) -> bool,
    queue_close_handle: unsafe extern "system" fn(This, *mut c_void),
    queue_submit_callback: unsafe extern "system" fn(This, *mut c_void, u32, *mut c_void, usize) -> i32,
    queue_submit_delayed_callback: unsafe extern "system" fn(This, *mut c_void, u32, u32, *mut c_void, usize) -> i32,
    queue_register_waiter: unsafe extern "system" fn(This, *mut c_void, u32, *mut c_void, *mut c_void, usize, *mut RegistrationToken) -> i32,
    queue_unregister_waiter: unsafe extern "system" fn(This, *mut c_void, RegistrationToken),
    queue_terminate: unsafe extern "system" fn(This, *mut c_void, bool, *mut c_void, usize) -> i32,
    queue_register_monitor: unsafe extern "system" fn(This, *mut c_void, *mut c_void, usize, *mut RegistrationToken) -> i32,
    queue_unregister_monitor: unsafe extern "system" fn(This, *mut c_void, RegistrationToken),
    queue_get_current_process_task_queue: unsafe extern "system" fn(This, *mut *mut c_void) -> bool,
    queue_set_current_process_task_queue: unsafe extern "system" fn(This, *mut c_void),
    thread_set_time_sensitive: unsafe extern "system" fn(This, bool) -> i32,
    thread_verify_not_time_sensitive: unsafe extern "system" fn(This) -> i32,
    thread_assert_not_time_sensitive: unsafe extern "system" fn(This),
    thread_is_time_sensitive: unsafe extern "system" fn(This) -> bool,
    slot31: unsafe extern "system" fn(This) -> i32,
}

/// The COM style object: a pointer to its vtable.
#[repr(C)]
pub(crate) struct Object {
    vtbl: &'static Vtbl,
}

/// The one object `QueryApiImpl` hands out.
pub(crate) static OBJECT: Object = Object { vtbl: &VTBL };

/// The object's address, as written to `*out`.
pub(crate) fn object() -> *mut c_void {
    (&raw const OBJECT).cast_mut().cast()
}

/// The table's name in the log.
const TABLE: &str = "XAsync/XTaskQueue/XThread";
/// The call counters of the 32 slots (`calllog`).
static COUNTS: Counters<32> = Counters::new();

/// Declares a slot that logs the call (`calllog`), drops `this` and calls the table function
/// with the other arguments.
macro_rules! shim {
    ($slot:literal, $($rest:tt)*) => {
        logged_shim!(TABLE, COUNTS, This, $slot, $($rest)*);
    };
}

shim!(3, get_status => xa::XAsyncGetStatus, (b: *mut XAsyncBlock, wait: bool) -> i32);
shim!(4, get_result_size => xa::XAsyncGetResultSize, (b: *mut XAsyncBlock, size: *mut usize) -> i32);
shim!(5, cancel => xa::XAsyncCancel, (b: *mut XAsyncBlock));
shim!(6, run => xa::XAsyncRun, (b: *mut XAsyncBlock, work: usize) -> i32);
shim!(7, begin => xa::XAsyncBegin, (b: *mut XAsyncBlock, context: *mut c_void, identity: *const c_void, name: *const u8, provider: usize) -> i32);
shim!(9, schedule => xa::XAsyncSchedule, (b: *mut XAsyncBlock, delay: u32) -> i32);
shim!(10, complete => xa::XAsyncComplete, (b: *mut XAsyncBlock, result: i32, required: usize));
shim!(11, get_result => xa::XAsyncGetResult, (b: *mut XAsyncBlock, identity: *const c_void, size: usize, buffer: *mut c_void, used: *mut usize) -> i32);
shim!(12, queue_create => tq::XTaskQueueCreate, (work: u32, completion: u32, out: *mut *mut c_void) -> i32);
shim!(13, queue_create_composite => tq::XTaskQueueCreateComposite, (work: *mut c_void, completion: *mut c_void, out: *mut *mut c_void) -> i32);
shim!(14, queue_get_port => tq::XTaskQueueGetPort, (h: *mut c_void, port: u32, out: *mut *mut c_void) -> i32);
shim!(15, queue_duplicate_handle => tq::XTaskQueueDuplicateHandle, (h: *mut c_void, out: *mut *mut c_void) -> i32);
shim!(16, queue_dispatch => tq::XTaskQueueDispatch, (h: *mut c_void, port: u32, timeout: u32) -> bool);
shim!(17, queue_close_handle => tq::XTaskQueueCloseHandle, (h: *mut c_void));
shim!(18, queue_submit_callback => tq::XTaskQueueSubmitCallback, (h: *mut c_void, port: u32, ctx: *mut c_void, cb: usize) -> i32);
shim!(19, queue_submit_delayed_callback => tq::XTaskQueueSubmitDelayedCallback, (h: *mut c_void, port: u32, delay: u32, ctx: *mut c_void, cb: usize) -> i32);
shim!(20, queue_register_waiter => tq::XTaskQueueRegisterWaiter, (h: *mut c_void, port: u32, handle: *mut c_void, ctx: *mut c_void, cb: usize, token: *mut RegistrationToken) -> i32);
shim!(21, queue_unregister_waiter => tq::XTaskQueueUnregisterWaiter, (h: *mut c_void, token: RegistrationToken));
shim!(22, queue_terminate => tq::XTaskQueueTerminate, (h: *mut c_void, wait: bool, ctx: *mut c_void, cb: usize) -> i32);
shim!(23, queue_register_monitor => tq::XTaskQueueRegisterMonitor, (h: *mut c_void, ctx: *mut c_void, cb: usize, token: *mut RegistrationToken) -> i32);
shim!(24, queue_unregister_monitor => tq::XTaskQueueUnregisterMonitor, (h: *mut c_void, token: RegistrationToken));
shim!(25, queue_get_current_process_task_queue => tq::XTaskQueueGetCurrentProcessTaskQueue, (out: *mut *mut c_void) -> bool);
shim!(26, queue_set_current_process_task_queue => tq::XTaskQueueSetCurrentProcessTaskQueue, (h: *mut c_void));
shim!(27, thread_set_time_sensitive => XThreadSetTimeSensitive, (value: bool) -> i32);
shim!(28, thread_verify_not_time_sensitive => XThreadVerifyNotTimeSensitive, () -> i32);
shim!(29, thread_assert_not_time_sensitive => XThreadAssertNotTimeSensitive, ());
shim!(30, thread_is_time_sensitive => XThreadIsTimeSensitive, () -> bool);

static VTBL: Vtbl = Vtbl {
    query_interface,
    add_ref: no_count,
    release: no_count,
    get_status,
    get_result_size,
    cancel,
    run,
    begin,
    slot8,
    schedule,
    complete,
    get_result,
    queue_create,
    queue_create_composite,
    queue_get_port,
    queue_duplicate_handle,
    queue_dispatch,
    queue_close_handle,
    queue_submit_callback,
    queue_submit_delayed_callback,
    queue_register_waiter,
    queue_unregister_waiter,
    queue_terminate,
    queue_register_monitor,
    queue_unregister_monitor,
    queue_get_current_process_task_queue,
    queue_set_current_process_task_queue,
    thread_set_time_sensitive,
    thread_verify_not_time_sensitive,
    thread_assert_not_time_sensitive,
    thread_is_time_sensitive,
    slot31,
};

unsafe extern "system" fn query_interface(this: This, iid: *const Guid, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM contract: `iid` is null or a GUID, `out` a writable pointer location.
    unsafe {
        if !iid.is_null() && iid.read_unaligned() == XASYNC_TABLE {
            *out = this.cast_mut().cast();
            return S_OK;
        }
        *out = std::ptr::null_mut();
    }
    E_NOINTERFACE
}

/// AddRef and Release of a static object.
unsafe extern "system" fn no_count(_this: This) -> u32 {
    1
}

fn unknown_slot(slot: usize) -> i32 {
    decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("local XAsync/XTaskQueue/XThread table slot {slot}"));
    if let Some(n) = COUNTS.next(slot) {
        log(&format!("local XAsync/XTaskQueue/XThread table: slot {slot} is not known -> E_NOTIMPL [call {n}]"));
    }
    E_NOTIMPL
}

unsafe extern "system" fn slot8(_this: This) -> i32 {
    unknown_slot(8)
}

unsafe extern "system" fn slot31(_this: This) -> i32 {
    unknown_slot(31)
}

// XThread: the "time sensitive" mark is a per thread flag. The two checking calls only log when
// a thread marked time sensitive reaches them, they never fail.

thread_local! {
    static TIME_SENSITIVE: Cell<bool> = const { Cell::new(false) };
}

unsafe extern "system" fn XThreadSetTimeSensitive(sensitive: bool) -> i32 {
    TIME_SENSITIVE.set(sensitive);
    S_OK
}

unsafe extern "system" fn XThreadIsTimeSensitive() -> bool {
    TIME_SENSITIVE.get()
}

unsafe extern "system" fn XThreadVerifyNotTimeSensitive() -> i32 {
    if TIME_SENSITIVE.get() {
        log("XThreadVerifyNotTimeSensitive called on a time-sensitive thread");
    }
    S_OK
}

unsafe extern "system" fn XThreadAssertNotTimeSensitive() {
    if TIME_SENSITIVE.get() {
        log("XThreadAssertNotTimeSensitive called on a time-sensitive thread");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// The object's vtable as the thunks see it: 32 addresses.
    fn slots() -> &'static [usize; 32] {
        // SAFETY: `Vtbl` is `repr(C)` with 32 function pointer fields, the size checked below.
        unsafe { &*((&raw const VTBL).cast::<[usize; 32]>()) }
    }

    #[test]
    fn the_vtable_has_32_slots_and_the_object_points_at_it() {
        assert_eq!(size_of::<Vtbl>(), 32 * size_of::<usize>());
        // SAFETY: the object's first field is the vtable pointer.
        let vtbl = unsafe { *(object() as *const *const Vtbl) };
        assert!(std::ptr::eq(vtbl, &raw const VTBL));
        assert_eq!(slots()[3], get_status as *const () as usize);
        assert_eq!(slots()[12], queue_create as *const () as usize);
        assert_eq!(slots()[30], thread_is_time_sensitive as *const () as usize);
    }

    #[test]
    fn query_interface_answers_the_table_iid_only() {
        let unknown = Guid { data1: 0, data2: 0, data3: 0, data4: [0xC0, 0, 0, 0, 0, 0, 0, 0x46] };
        let mut out = std::ptr::dangling_mut::<c_void>();
        // SAFETY: valid GUIDs and out location.
        unsafe {
            assert_eq!((VTBL.query_interface)(&OBJECT, &XASYNC_TABLE, &mut out), S_OK);
            assert_eq!(out, object());
            assert_eq!((VTBL.query_interface)(&OBJECT, &unknown, &mut out), E_NOINTERFACE);
            assert!(out.is_null());
            assert_eq!((VTBL.add_ref)(&OBJECT), 1);
            assert_eq!((VTBL.release)(&OBJECT), 1);
            assert_eq!((VTBL.slot8)(&OBJECT), E_NOTIMPL);
            assert_eq!((VTBL.slot31)(&OBJECT), E_NOTIMPL);
        }
    }

    /// A manual queue through the slots, called as the thunks call them (`this` first).
    #[test]
    fn a_manual_queue_through_the_slots() {
        static RAN: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn count(_ctx: *mut c_void, _canceled: bool) {
            RAN.fetch_add(1, Ordering::SeqCst);
        }
        let this: This = &OBJECT;
        // SAFETY: the slots get the GDK arguments.
        unsafe {
            let mut h = std::ptr::null_mut();
            assert_eq!((VTBL.queue_create)(this, tq::MODE_MANUAL, tq::MODE_MANUAL, &mut h), S_OK);
            assert_eq!((VTBL.queue_submit_callback)(this, h, tq::PORT_WORK, std::ptr::null_mut(), count as *const () as usize), S_OK);
            assert!((VTBL.queue_dispatch)(this, h, tq::PORT_WORK, 0));
            assert_eq!(RAN.load(Ordering::SeqCst), 1);
            (VTBL.queue_close_handle)(this, h);
            assert_eq!((VTBL.thread_set_time_sensitive)(this, true), S_OK);
            assert!((VTBL.thread_is_time_sensitive)(this));
            assert_eq!((VTBL.thread_set_time_sensitive)(this, false), S_OK);
            assert!(!(VTBL.thread_is_time_sensitive)(this));
        }
    }

    #[test]
    fn only_the_xasync_couple_is_served() {
        let user = Guid { data1: 0x01acd177, data2: 0x91f9, data3: 0x4763, data4: [0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0] };
        assert!(serves(Some(&XASYNC_TABLE), Some(&XASYNC_TABLE)));
        assert!(!serves(Some(&XASYNC_TABLE), Some(&user)));
        assert!(!serves(Some(&XASYNC_TABLE), None));
        assert!(!serves(Some(&user), Some(&user)));
        assert!(!serves(None, None));
    }
}
