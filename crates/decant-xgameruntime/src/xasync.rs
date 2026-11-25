//! XAsync of the local XAsync / XTaskQueue / XThread table (served only without a PC runtime, see
//! `local_table`). Contract: the public GDK `XAsync.h` and `XAsyncProvider.h`.
//!
//! An `XAsyncBlock` is { queue, context, callback, internal[32] }. While an operation is in
//! flight, or finished with a result nobody read yet, its `Op` sits in the `LIVE` table keyed by
//! the block address, and a stamp with the operation's serial number is written into `internal`.
//! Once the operation is finished and holds no unread result, the `Op` leaves the table and only
//! the stamp (status plus flags) stays in the block. A block on the title's stack therefore leaks
//! nothing, and a block that was never started or has been overwritten is recognised as unknown.
//!
//! `begin_with` is the entry point of the local tables (user, package, store and so on): it
//! completes an async call with bytes computed by a closure on the block's queue.
#![allow(unsafe_code, non_snake_case)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

use crate::taskqueue::{self, Callback, E_ABORT, E_FAIL, E_INVALIDARG, PORT_COMPLETION, PORT_WORK, S_OK};

// Provider operation codes of `XAsyncOp`.
const OP_BEGIN: u32 = 0;
const OP_DO_WORK: u32 = 1;
const OP_GET_RESULT: u32 = 2;
const OP_CANCEL: u32 = 3;
const OP_CLEANUP: u32 = 4;

pub(crate) const E_PENDING: i32 = 0x8000_000A_u32 as i32;
const E_ILLEGAL_METHOD_CALL: i32 = 0x8000_000E_u32 as i32;
const E_NOT_SUFFICIENT_BUFFER: i32 = 0x8007_007A_u32 as i32;

/// `XAsyncBlock`.
#[repr(C)]
pub(crate) struct XAsyncBlock {
    pub(crate) queue: *mut c_void,
    pub(crate) context: *mut c_void,
    pub(crate) callback: usize,
    pub(crate) internal: [u8; 32],
}

/// `XAsyncProviderData`.
#[repr(C)]
struct XAsyncProviderData {
    async_: *mut XAsyncBlock,
    buffer_size: usize,
    buffer: *mut c_void,
    context: *mut c_void,
}

type ProviderFn = unsafe extern "system" fn(u32, *const XAsyncProviderData) -> i32;
type WorkFn = unsafe extern "system" fn(*mut XAsyncBlock) -> i32;
type CompletionFn = unsafe extern "system" fn(*mut XAsyncBlock);

fn grab<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Started, not finished.
    Running,
    /// `XAsyncComplete` claimed the completion and is still running cleanup and publishing.
    Finishing,
    /// Finished. When the payload is not empty it is still unread.
    Finished,
    /// `XAsyncGetResult` claimed the single read and is running GetResult and Cleanup.
    Reading,
    /// Finished and released (no payload, an error, or the payload was read).
    Released,
}

struct Stage {
    status: i32,
    payload_len: usize,
    phase: Phase,
}

impl Stage {
    fn has_finished(&self) -> bool {
        !matches!(self.phase, Phase::Running | Phase::Finishing)
    }
}

/// One started async operation.
struct Op {
    serial: u64,
    provider: usize,
    context: usize,
    identity: usize,
    /// The work function of `XAsyncRun`, 0 for provider based operations.
    work: usize,
    stage: Mutex<Stage>,
    changed: Condvar,
}

impl Op {
    fn start(provider: usize, context: usize, identity: usize, work: usize) -> Arc<Op> {
        static SERIALS: AtomicU64 = AtomicU64::new(1);
        Arc::new(Op {
            serial: SERIALS.fetch_add(1, Ordering::Relaxed),
            provider,
            context,
            identity,
            work,
            stage: Mutex::new(Stage { status: E_PENDING, payload_len: 0, phase: Phase::Running }),
            changed: Condvar::new(),
        })
    }

    fn stage(&self) -> MutexGuard<'_, Stage> {
        grab(&self.stage)
    }

    /// Calls the provider with `op`, or does nothing for an operation without one.
    ///
    /// # Safety
    /// `self.provider` is 0 or the title's `XAsyncProvider`, and `b` is the operation's block.
    unsafe fn call(&self, op: u32, b: *mut XAsyncBlock, buffer: *mut c_void, size: usize) -> i32 {
        if self.provider == 0 {
            return S_OK;
        }
        let data = XAsyncProviderData { async_: b, buffer_size: size, buffer, context: self.context as *mut c_void };
        // SAFETY: the caller's contract.
        unsafe {
            let f = std::mem::transmute::<usize, ProviderFn>(self.provider);
            f(op, &data)
        }
    }
}

fn live_table() -> MutexGuard<'static, HashMap<usize, Arc<Op>>> {
    static LIVE: OnceLock<Mutex<HashMap<usize, Arc<Op>>>> = OnceLock::new();
    grab(LIVE.get_or_init(|| Mutex::new(HashMap::new())))
}

/// What stays in `XAsyncBlock::internal` of a started block.
#[derive(Clone, Copy)]
struct Stamp {
    status: i32,
    flags: u8,
    serial: u64,
}

const STAMP_TAG: u32 = 0x4443_4E54;
const STAMP_FINISHED: u8 = 1;
const STAMP_READ: u8 = 2;

/// # Safety
/// `b` is a writable `XAsyncBlock`.
unsafe fn put_stamp(b: *mut XAsyncBlock, s: Stamp) {
    let mut raw = [0u8; 32];
    raw[0..4].copy_from_slice(&STAMP_TAG.to_le_bytes());
    raw[4..8].copy_from_slice(&s.status.to_le_bytes());
    raw[8] = s.flags;
    raw[16..24].copy_from_slice(&s.serial.to_le_bytes());
    // SAFETY: the caller's contract.
    unsafe { (*b).internal = raw };
}

/// # Safety
/// `b` is null or a readable `XAsyncBlock`.
unsafe fn get_stamp(b: *mut XAsyncBlock) -> Option<Stamp> {
    if b.is_null() {
        return None;
    }
    // SAFETY: the caller's contract.
    let raw = unsafe { (*b).internal };
    if u32::from_le_bytes(raw[0..4].try_into().ok()?) != STAMP_TAG {
        return None;
    }
    Some(Stamp { status: i32::from_le_bytes(raw[4..8].try_into().ok()?), flags: raw[8], serial: u64::from_le_bytes(raw[16..24].try_into().ok()?) })
}

enum Found {
    Live(Arc<Op>),
    Settled(Stamp),
    Unknown,
}

/// Finds what a block refers to. The table lock is held while the stamp is read, so a completion
/// that swaps the table entry for the final stamp is seen as one step.
///
/// # Safety
/// As [`get_stamp`].
unsafe fn find(b: *mut XAsyncBlock) -> Found {
    let table = live_table();
    // SAFETY: the caller's contract.
    let Some(stamp) = (unsafe { get_stamp(b) }) else { return Found::Unknown };
    match table.get(&(b as usize)) {
        Some(op) if op.serial == stamp.serial => Found::Live(op.clone()),
        _ if stamp.flags & STAMP_FINISHED != 0 => Found::Settled(stamp),
        _ => Found::Unknown,
    }
}

/// Registers a new operation for a block and stamps the block.
///
/// # Safety
/// `b` is a writable `XAsyncBlock`.
unsafe fn adopt(b: *mut XAsyncBlock, op: Arc<Op>) {
    let mut table = live_table();
    // SAFETY: the caller's contract.
    unsafe { put_stamp(b, Stamp { status: E_PENDING, flags: 0, serial: op.serial }) };
    table.insert(b as usize, op);
}

/// Stamps the final state into the block and, when `release`, removes the operation from the
/// table, all under the table lock. The operation's phase is set before the lock is dropped.
///
/// # Safety
/// `b` is a writable `XAsyncBlock`.
unsafe fn settle(b: *mut XAsyncBlock, op: &Arc<Op>, status: i32, flags: u8, release: bool) {
    let mut table = live_table();
    // SAFETY: the caller's contract.
    unsafe { put_stamp(b, Stamp { status, flags, serial: op.serial }) };
    if release && table.get(&(b as usize)).is_some_and(|cur| cur.serial == op.serial) {
        table.remove(&(b as usize));
    }
    op.stage().phase = if release { Phase::Released } else { Phase::Finished };
}

unsafe extern "system" fn run_scheduled(ctx: *mut c_void, canceled: bool) {
    let b = ctx as *mut XAsyncBlock;
    // SAFETY: `ctx` is the title's block, scheduled by `XAsyncSchedule`.
    unsafe {
        let Found::Live(op) = find(b) else { return };
        if canceled {
            XAsyncComplete(b, E_ABORT, 0);
            return;
        }
        let hr = if op.work != 0 {
            let f = std::mem::transmute::<usize, WorkFn>(op.work);
            f(b)
        } else {
            op.call(OP_DO_WORK, b, std::ptr::null_mut(), 0)
        };
        // A provider may finish inside DoWork without calling XAsyncComplete itself.
        if hr != E_PENDING && op.stage().phase == Phase::Running {
            XAsyncComplete(b, hr, 0);
        }
    }
}

unsafe extern "system" fn run_completion(ctx: *mut c_void, _canceled: bool) {
    let b = ctx as *mut XAsyncBlock;
    // SAFETY: `ctx` is the title's block, whose callback is an `XAsyncCompletionRoutine`.
    unsafe {
        if (*b).callback != 0 {
            let f = std::mem::transmute::<usize, CompletionFn>((*b).callback);
            f(b);
        }
    }
}

// The table's functions, with the public GDK signatures. Each `unsafe` function relies on the
// title passing what the GDK prototype says: a valid `XAsyncBlock`, callbacks of the named type.

pub(crate) unsafe extern "system" fn XAsyncBegin(b: *mut XAsyncBlock, context: *mut c_void, identity: *const c_void, _identity_name: *const u8, provider: usize) -> i32 {
    if b.is_null() || provider == 0 {
        return E_INVALIDARG;
    }
    let op = Op::start(provider, context as usize, identity as usize, 0);
    // SAFETY: the title's block and provider.
    unsafe {
        adopt(b, op.clone());
        // A Begin that fails fails the async call, not XAsyncBegin: the error reaches the title
        // once, through the block.
        let hr = op.call(OP_BEGIN, b, std::ptr::null_mut(), 0);
        if hr < 0 {
            XAsyncComplete(b, hr, 0);
        }
    }
    S_OK
}

pub(crate) unsafe extern "system" fn XAsyncSchedule(b: *mut XAsyncBlock, delay_ms: u32) -> i32 {
    if b.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block.
    let queue = unsafe { (*b).queue };
    let Some(q) = taskqueue::queue_from(queue) else {
        return if queue.is_null() { taskqueue::E_NO_TASK_QUEUE } else { E_INVALIDARG };
    };
    if taskqueue::enqueue_after(&q, PORT_WORK as usize, delay_ms, Callback { func: run_scheduled as *const () as usize, ctx: b as usize }) { S_OK } else { E_ABORT }
}

pub(crate) unsafe extern "system" fn XAsyncComplete(b: *mut XAsyncBlock, result: i32, required: usize) {
    // E_PENDING means the provider has more to do, it is not a completion.
    if result == E_PENDING {
        return;
    }
    // SAFETY: the title's block.
    let Found::Live(op) = (unsafe { find(b) }) else { return };
    let release_now = {
        let mut st = op.stage();
        if st.phase != Phase::Running {
            return;
        }
        // Waiters see the operation as unfinished until cleanup and the stamp are done, so none
        // of them frees the block or races on the flags.
        st.phase = Phase::Finishing;
        st.status = result;
        st.payload_len = if result >= 0 { required } else { 0 };
        st.payload_len == 0
    };
    // The callback and queue are read before publishing: a waiter woken by the completion may
    // free a block that lives on its stack.
    // SAFETY: the title's block, still live.
    let (callback, queue_ptr) = unsafe { ((*b).callback, (*b).queue) };
    if release_now {
        // SAFETY: the title's provider.
        unsafe { op.call(OP_CLEANUP, b, std::ptr::null_mut(), 0) };
    }
    // SAFETY: the title's block.
    unsafe { settle(b, &op, result, STAMP_FINISHED, release_now) };
    if callback != 0
        && let Some(q) = taskqueue::queue_from(queue_ptr)
    {
        taskqueue::enqueue(&q, PORT_COMPLETION as usize, Callback { func: run_completion as *const () as usize, ctx: b as usize });
    }
    op.changed.notify_all();
}

pub(crate) unsafe extern "system" fn XAsyncGetStatus(b: *mut XAsyncBlock, wait: bool) -> i32 {
    // SAFETY: the title's block.
    match unsafe { find(b) } {
        Found::Live(op) => {
            let mut st = op.stage();
            while wait && !st.has_finished() {
                st = op.changed.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
            if st.has_finished() { st.status } else { E_PENDING }
        }
        Found::Settled(stamp) => stamp.status,
        Found::Unknown => E_INVALIDARG,
    }
}

pub(crate) unsafe extern "system" fn XAsyncGetResultSize(b: *mut XAsyncBlock, size: *mut usize) -> i32 {
    // SAFETY: the title's block. `size` is null or writable.
    unsafe {
        match find(b) {
            Found::Live(op) => {
                let st = op.stage();
                if !st.has_finished() {
                    return E_PENDING;
                }
                if !size.is_null() {
                    *size = st.payload_len;
                }
                st.status
            }
            Found::Settled(stamp) => {
                if !size.is_null() && stamp.status >= 0 {
                    *size = 0;
                }
                stamp.status
            }
            Found::Unknown => E_INVALIDARG,
        }
    }
}

pub(crate) unsafe extern "system" fn XAsyncGetResult(b: *mut XAsyncBlock, identity: *const c_void, size: usize, buffer: *mut c_void, used: *mut usize) -> i32 {
    // SAFETY: the title's block, a buffer of `size` bytes and `used` (null or writable).
    unsafe {
        let op = match find(b) {
            Found::Unknown => return E_INVALIDARG,
            Found::Settled(stamp) => {
                if stamp.status < 0 {
                    return stamp.status;
                }
                if stamp.flags & STAMP_READ != 0 {
                    return E_ILLEGAL_METHOD_CALL;
                }
                if !used.is_null() {
                    *used = 0;
                }
                return S_OK;
            }
            Found::Live(op) => op,
        };
        let (status, payload_len) = {
            let mut st = op.stage();
            if !st.has_finished() {
                return E_PENDING;
            }
            if !identity.is_null() && op.identity != 0 && identity as usize != op.identity {
                return E_INVALIDARG;
            }
            if st.status < 0 {
                return st.status;
            }
            if st.payload_len > 0 {
                if st.phase != Phase::Finished {
                    return E_ILLEGAL_METHOD_CALL;
                }
                if size < st.payload_len || buffer.is_null() {
                    return E_NOT_SUFFICIENT_BUFFER;
                }
                // Claim the one read of the payload together with its cleanup.
                st.phase = Phase::Reading;
            }
            (st.status, st.payload_len)
        };
        if !used.is_null() {
            *used = payload_len;
        }
        if payload_len == 0 {
            return S_OK;
        }
        let hr = op.call(OP_GET_RESULT, b, buffer, size);
        op.call(OP_CLEANUP, b, std::ptr::null_mut(), 0);
        settle(b, &op, status, STAMP_FINISHED | STAMP_READ, true);
        hr
    }
}

pub(crate) unsafe extern "system" fn XAsyncCancel(b: *mut XAsyncBlock) {
    // SAFETY: the title's block.
    let Found::Live(op) = (unsafe { find(b) }) else { return };
    if op.stage().phase != Phase::Running {
        return;
    }
    // Cancelling belongs to the provider: one that supports it completes with E_ABORT, now or
    // later. One that does not (XAsyncRun) runs to its normal end. Completing here instead would
    // run the provider's cleanup while its DoWork may still be running.
    // SAFETY: the title's provider.
    unsafe { op.call(OP_CANCEL, b, std::ptr::null_mut(), 0) };
}

pub(crate) unsafe extern "system" fn XAsyncRun(b: *mut XAsyncBlock, work: usize) -> i32 {
    if b.is_null() || work == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: the title's block.
    unsafe {
        adopt(b, Op::start(0, 0, 0, work));
        let hr = XAsyncSchedule(b, 0);
        if hr < 0 {
            XAsyncComplete(b, hr, 0);
        }
    }
    S_OK
}

type Compute = Box<dyn FnOnce() -> Result<Vec<u8>, i32> + Send>;

/// State of one `begin_with` call, owned by its provider until Cleanup.
struct Payload {
    compute: Mutex<Option<Compute>>,
    outcome: Mutex<Option<Result<Vec<u8>, i32>>>,
}

/// The provider behind `begin_with`. Begin schedules the work, DoWork computes and completes with
/// the bytes or the error, GetResult copies the bytes out and Cleanup frees the payload.
unsafe extern "system" fn payload_provider(op: u32, data: *const XAsyncProviderData) -> i32 {
    // SAFETY: `data` is built by `Op::call`. Its context is the `Payload` leaked by `begin_with`,
    // alive until OP_CLEANUP.
    unsafe {
        let d = &*data;
        let payload = &*(d.context as *const Payload);
        match op {
            OP_BEGIN => XAsyncSchedule(d.async_, 0),
            OP_DO_WORK => {
                let compute = grab(&payload.compute).take();
                let outcome = compute.map_or(Err(E_FAIL), |f| f());
                let (hr, len) = match &outcome {
                    Ok(bytes) => (S_OK, bytes.len()),
                    Err(e) => (*e, 0),
                };
                *grab(&payload.outcome) = Some(outcome);
                XAsyncComplete(d.async_, hr, len);
                S_OK
            }
            OP_GET_RESULT => {
                if let Some(Ok(bytes)) = grab(&payload.outcome).as_ref() {
                    let n = bytes.len().min(d.buffer_size);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), d.buffer as *mut u8, n);
                }
                S_OK
            }
            OP_CLEANUP => {
                drop(Box::from_raw(d.context as *mut Payload));
                S_OK
            }
            _ => S_OK,
        }
    }
}

/// Starts an async call whose result is the bytes `work` returns, computed on the work port of
/// the block's queue. `identity` ties the Result calls to the call. An error from `work` fails the
/// call through the block.
///
/// # Safety
/// `b` is null or the title's writable `XAsyncBlock`, valid until the call completes.
pub(crate) unsafe fn begin_with<F>(b: *mut XAsyncBlock, identity: usize, work: F) -> i32
where
    F: FnOnce() -> Result<Vec<u8>, i32> + Send + 'static,
{
    let payload = Box::into_raw(Box::new(Payload { compute: Mutex::new(Some(Box::new(work))), outcome: Mutex::new(None) }));
    // SAFETY: the caller's block. The provider owns `payload` from here on.
    let hr = unsafe { XAsyncBegin(b, payload as *mut c_void, identity as *const c_void, std::ptr::null(), payload_provider as *const () as usize) };
    if hr < 0 {
        // SAFETY: Begin refused the call before adopting the payload, so no Cleanup will run.
        drop(unsafe { Box::from_raw(payload) });
    }
    hr
}

/// The operations still in flight or holding an unread result, for the heartbeat of `calllog`.
pub(crate) fn active_count() -> usize {
    live_table().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn as_usize(f: unsafe extern "system" fn(*mut XAsyncBlock) -> i32) -> usize {
        f as usize
    }

    fn provider_addr(f: ProviderFn) -> usize {
        f as usize
    }

    fn is_live(b: &XAsyncBlock) -> bool {
        live_table().contains_key(&(b as *const XAsyncBlock as usize))
    }

    fn blank() -> XAsyncBlock {
        XAsyncBlock { queue: std::ptr::null_mut(), context: std::ptr::null_mut(), callback: 0, internal: [0; 32] }
    }

    unsafe extern "system" fn work_seven(_b: *mut XAsyncBlock) -> i32 {
        7
    }

    unsafe extern "system" fn inert(_op: u32, _data: *const XAsyncProviderData) -> i32 {
        S_OK
    }

    /// A provider that completes with E_ABORT when it is told to cancel.
    unsafe extern "system" fn abort_on_cancel(op: u32, data: *const XAsyncProviderData) -> i32 {
        if op == OP_CANCEL {
            // SAFETY: the provider data of the test's block.
            unsafe { XAsyncComplete((*data).async_, E_ABORT, 0) };
        }
        S_OK
    }

    fn begin(b: &mut XAsyncBlock, identity: usize, provider: ProviderFn) -> i32 {
        // SAFETY: `b` is the test's block.
        unsafe { XAsyncBegin(b, std::ptr::null_mut(), identity as *const c_void, std::ptr::null(), provider_addr(provider)) }
    }

    #[test]
    fn begin_and_run_validate_their_arguments() {
        let mut b = blank();
        // SAFETY: only the argument checks run.
        unsafe {
            assert_eq!(XAsyncBegin(std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), provider_addr(inert)), E_INVALIDARG);
            assert_eq!(XAsyncBegin(&mut b, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 0), E_INVALIDARG);
            assert_eq!(XAsyncRun(&mut b, 0), E_INVALIDARG);
            assert_eq!(XAsyncRun(std::ptr::null_mut(), as_usize(work_seven)), E_INVALIDARG);
            assert_eq!(XAsyncSchedule(std::ptr::null_mut(), 0), E_INVALIDARG);
        }
    }

    #[test]
    fn a_block_that_was_never_started_is_unknown() {
        let mut b = blank();
        let mut size = 5usize;
        // SAFETY: the block is the test's.
        unsafe {
            assert_eq!(XAsyncGetStatus(&mut b, false), E_INVALIDARG);
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), E_INVALIDARG);
            assert_eq!(XAsyncGetResult(&mut b, std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut()), E_INVALIDARG);
            assert_eq!(XAsyncGetStatus(std::ptr::null_mut(), false), E_INVALIDARG);
        }
        assert_eq!(size, 5);
    }

    #[test]
    fn run_on_a_manual_queue_waits_for_both_dispatches() {
        static COMPLETED: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn completed(b: *mut XAsyncBlock) {
            // SAFETY: the test's block.
            assert_eq!(unsafe { XAsyncGetStatus(b, false) }, 7);
            COMPLETED.fetch_add(1, Ordering::SeqCst);
        }
        let mut q = std::ptr::null_mut();
        let mut b = blank();
        // SAFETY: queue, block and callbacks belong to the test.
        unsafe {
            assert_eq!(taskqueue::XTaskQueueCreate(taskqueue::MODE_MANUAL, taskqueue::MODE_MANUAL, &mut q), S_OK);
            b.queue = q;
            b.callback = completed as *const () as usize;
            assert_eq!(XAsyncRun(&mut b, as_usize(work_seven)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, false), E_PENDING);
            assert!(taskqueue::XTaskQueueDispatch(q, PORT_WORK, 0));
            assert_eq!(XAsyncGetStatus(&mut b, false), 7);
            assert_eq!(COMPLETED.load(Ordering::SeqCst), 0);
            assert!(taskqueue::XTaskQueueDispatch(q, PORT_COMPLETION, 0));
            assert_eq!(COMPLETED.load(Ordering::SeqCst), 1);
            taskqueue::XTaskQueueCloseHandle(q);
        }
        assert!(!is_live(&b), "a finished call without payload leaves the live table");
    }

    #[test]
    fn run_on_the_process_queue_finishes_and_can_be_waited_for() {
        let mut b = blank();
        // SAFETY: the block is the test's.
        unsafe {
            assert_eq!(XAsyncRun(&mut b, as_usize(work_seven)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), 7);
            assert_eq!(XAsyncGetStatus(&mut b, false), 7, "the status stays readable");
            let mut size = 99usize;
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), 7);
        }
        assert!(!is_live(&b));
    }

    #[test]
    fn a_failing_work_function_reports_through_the_block() {
        unsafe extern "system" fn broken(_b: *mut XAsyncBlock) -> i32 {
            E_FAIL
        }
        let mut b = blank();
        // SAFETY: the block is the test's.
        unsafe {
            assert_eq!(XAsyncRun(&mut b, as_usize(broken)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), E_FAIL);
            let mut used = 1usize;
            assert_eq!(XAsyncGetResult(&mut b, std::ptr::null(), 0, std::ptr::null_mut(), &mut used), E_FAIL);
        }
    }

    #[test]
    fn schedule_needs_a_usable_queue() {
        let mut b = blank();
        // SAFETY: the block is the test's. The queue values are never dereferenced.
        unsafe {
            assert_eq!(XAsyncBegin(&mut b, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), provider_addr(inert)), S_OK);
            b.queue = 0x10 as *mut c_void;
            assert_eq!(XAsyncSchedule(&mut b, 0), E_INVALIDARG);
            XAsyncComplete(&mut b, S_OK, 0);
        }
    }

    #[test]
    fn work_on_a_terminated_queue_is_aborted() {
        let mut q = std::ptr::null_mut();
        let mut b = blank();
        // SAFETY: queue and block belong to the test.
        unsafe {
            assert_eq!(taskqueue::XTaskQueueCreate(taskqueue::MODE_THREAD_POOL, taskqueue::MODE_THREAD_POOL, &mut q), S_OK);
            assert_eq!(taskqueue::XTaskQueueTerminate(q, true, std::ptr::null_mut(), 0), S_OK);
            b.queue = q;
            assert_eq!(XAsyncRun(&mut b, as_usize(work_seven)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, true), E_ABORT);
            taskqueue::XTaskQueueCloseHandle(q);
        }
    }

    #[test]
    fn a_run_on_an_unknown_queue_fails_through_the_block() {
        let mut b = blank();
        b.queue = 0x10 as *mut c_void;
        // SAFETY: the block is the test's.
        unsafe {
            assert_eq!(XAsyncRun(&mut b, as_usize(work_seven)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut b, false), E_INVALIDARG);
        }
    }

    #[test]
    fn cancel_is_left_to_the_provider() {
        let mut stubborn = blank();
        let mut polite = blank();
        assert_eq!(begin(&mut stubborn, 0, inert), S_OK);
        assert_eq!(begin(&mut polite, 0, abort_on_cancel), S_OK);
        // SAFETY: the blocks are the test's.
        unsafe {
            XAsyncCancel(&mut stubborn);
            assert_eq!(XAsyncGetStatus(&mut stubborn, false), E_PENDING, "this provider ignores cancel");
            XAsyncComplete(&mut stubborn, S_OK, 0);
            assert_eq!(XAsyncGetStatus(&mut stubborn, true), S_OK);
            XAsyncCancel(&mut stubborn);
            XAsyncCancel(&mut polite);
            assert_eq!(XAsyncGetStatus(&mut polite, false), E_ABORT);
        }
    }

    #[test]
    fn completing_with_pending_is_ignored_and_only_the_first_completion_counts() {
        let mut b = blank();
        assert_eq!(begin(&mut b, 0, inert), S_OK);
        // SAFETY: the block is the test's.
        unsafe {
            XAsyncComplete(&mut b, E_PENDING, 0);
            assert_eq!(XAsyncGetStatus(&mut b, false), E_PENDING);
            XAsyncComplete(&mut b, 3, 0);
            XAsyncComplete(&mut b, 4, 0);
            assert_eq!(XAsyncGetStatus(&mut b, true), 3);
        }
    }

    #[test]
    fn a_begin_that_fails_completes_the_call_with_its_error() {
        unsafe extern "system" fn refuses(op: u32, _data: *const XAsyncProviderData) -> i32 {
            if op == OP_BEGIN { E_FAIL } else { S_OK }
        }
        let mut b = blank();
        assert_eq!(begin(&mut b, 0, refuses), S_OK);
        // SAFETY: the block is the test's.
        assert_eq!(unsafe { XAsyncGetStatus(&mut b, false) }, E_FAIL);
    }

    #[test]
    fn a_provider_may_finish_inside_do_work_by_returning() {
        unsafe extern "system" fn one_shot(op: u32, data: *const XAsyncProviderData) -> i32 {
            match op {
                // SAFETY: the provider data of the test's block.
                OP_BEGIN => unsafe { XAsyncSchedule((*data).async_, 0) },
                OP_DO_WORK => 12,
                _ => S_OK,
            }
        }
        let mut b = blank();
        assert_eq!(begin(&mut b, 0, one_shot), S_OK);
        // SAFETY: the block is the test's.
        assert_eq!(unsafe { XAsyncGetStatus(&mut b, true) }, 12);
    }

    #[test]
    fn a_payload_is_read_exactly_once_with_the_right_identity() {
        unsafe extern "system" fn with_bytes(op: u32, data: *const XAsyncProviderData) -> i32 {
            if op == OP_GET_RESULT {
                // SAFETY: `XAsyncGetResult` checked that the buffer holds the 4 bytes.
                unsafe { std::ptr::copy_nonoverlapping(b"abcd".as_ptr(), (*data).buffer as *mut u8, 4) };
            }
            S_OK
        }
        let mut b = blank();
        let id = 0xBEEF_usize;
        assert_eq!(begin(&mut b, id, with_bytes), S_OK);
        let (mut size, mut used, mut buf) = (0usize, 0usize, [0u8; 4]);
        // SAFETY: the block and buffers are the test's.
        unsafe {
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), E_PENDING);
            XAsyncComplete(&mut b, S_OK, 4);
            assert_eq!(XAsyncGetResultSize(&mut b, &mut size), S_OK);
            assert_eq!(size, 4);
            assert!(is_live(&b), "an unread payload keeps the call live");
            assert_eq!(XAsyncGetResult(&mut b, id as *const c_void, 3, buf.as_mut_ptr().cast(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            assert_eq!(XAsyncGetResult(&mut b, id as *const c_void, 4, std::ptr::null_mut(), &mut used), E_NOT_SUFFICIENT_BUFFER);
            assert_eq!(XAsyncGetResult(&mut b, 0x2 as *const c_void, 4, buf.as_mut_ptr().cast(), &mut used), E_INVALIDARG, "wrong identity");
            assert_eq!(XAsyncGetResult(&mut b, id as *const c_void, 4, buf.as_mut_ptr().cast(), &mut used), S_OK);
            assert_eq!((&buf, used), (b"abcd", 4));
            assert_eq!(XAsyncGetResult(&mut b, id as *const c_void, 4, buf.as_mut_ptr().cast(), &mut used), E_ILLEGAL_METHOD_CALL);
        }
        assert!(!is_live(&b));
    }

    #[test]
    fn a_result_asked_for_too_early_is_pending() {
        let mut b = blank();
        assert_eq!(begin(&mut b, 0, inert), S_OK);
        // SAFETY: the block is the test's.
        unsafe {
            assert_eq!(XAsyncGetResult(&mut b, std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut()), E_PENDING);
            XAsyncComplete(&mut b, S_OK, 0);
            let mut used = 9usize;
            assert_eq!(XAsyncGetResult(&mut b, std::ptr::null(), 0, std::ptr::null_mut(), &mut used), S_OK);
            assert_eq!(used, 0);
        }
    }

    #[test]
    fn begin_with_delivers_its_bytes_and_its_errors() {
        let mut ok = blank();
        let (mut size, mut used, mut buf) = (0usize, 0usize, [0u8; 8]);
        // SAFETY: blocks and buffers are the test's.
        unsafe {
            assert_eq!(begin_with(&mut ok, 0x44, || Ok(b"hello".to_vec())), S_OK);
            assert_eq!(XAsyncGetStatus(&mut ok, true), S_OK);
            assert_eq!(XAsyncGetResultSize(&mut ok, &mut size), S_OK);
            assert_eq!(size, 5);
            assert_eq!(XAsyncGetResult(&mut ok, 0x44 as *const c_void, 8, buf.as_mut_ptr().cast(), &mut used), S_OK);
            assert_eq!((&buf[..5], used), (&b"hello"[..], 5));
            assert_eq!(XAsyncGetResult(&mut ok, 0x44 as *const c_void, 8, buf.as_mut_ptr().cast(), &mut used), E_ILLEGAL_METHOD_CALL);

            let mut bad = blank();
            assert_eq!(begin_with(&mut bad, 1, || Err(E_FAIL)), S_OK);
            assert_eq!(XAsyncGetStatus(&mut bad, true), E_FAIL);
            assert_eq!(XAsyncGetResultSize(&mut bad, &mut size), E_FAIL);
            assert!(!is_live(&bad));
        }
        assert!(!is_live(&ok));
    }

    #[test]
    fn begin_with_rejects_a_null_block_and_frees_its_closure() {
        use std::sync::atomic::AtomicBool;
        struct Flag(Arc<AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = Flag(dropped.clone());
        // SAFETY: a null block is the case under test.
        let hr = unsafe {
            begin_with(std::ptr::null_mut(), 0, move || {
                let _keep = &guard;
                Ok(Vec::new())
            })
        };
        assert_eq!(hr, E_INVALIDARG);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn a_waiter_on_another_thread_is_woken_by_the_completion() {
        let mut b = blank();
        assert_eq!(begin(&mut b, 0, inert), S_OK);
        let block = &mut b as *mut XAsyncBlock;
        let addr = block as usize;
        let finisher = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(40));
            // SAFETY: the block outlives the join below.
            unsafe { XAsyncComplete(addr as *mut XAsyncBlock, 5, 0) };
        });
        // SAFETY: the block is the test's.
        assert_eq!(unsafe { XAsyncGetStatus(block, true) }, 5);
        finisher.join().unwrap();
    }
}
