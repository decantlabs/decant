//! The async blocks the title starts through a PC runtime table (`runtime_proxy`), watched until
//! they complete. A block that is still pending [`LATE_SECS`] s after its call returned is logged
//! once, and again when it completes after all.
//!
//! The proxy does not know which argument of a slot is an `XAsyncBlock`. Before the call, every
//! argument that points to 56 readable bytes looking like a block (queue null or one of our queue
//! handles, callback null or in executable memory) is a candidate, with a copy of its 32
//! internal bytes. After a successful call, a candidate whose queue and callback are unchanged
//! and whose internal bytes the PC runtime wrote is a started block. A watcher thread asks the PC
//! runtime's own `XAsyncGetStatus` (slot 3 of its XAsync table, without waiting) every
//! [`SWEEP_MS`] ms; any answer but `E_PENDING` ends the watch, and an answer of `E_INVALIDARG`
//! for a block never reported means it was not a block after all and is dropped silently.
//!
//! Our own code.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, VirtualQuery,
};

use crate::runtime_proxy::emit;
use crate::xasync::{E_PENDING, XAsyncBlock};

/// A block still pending this long after its call returned is logged.
pub(crate) const LATE_SECS: u64 = 5;
/// The watcher's period.
const SWEEP_MS: u64 = 500;
/// At most this many blocks are watched at once; more are logged as not watched.
const MAX_WATCHED: usize = 4096;
const E_INVALIDARG: i32 = 0x8007_0057_u32 as i32;

/// An argument that may be an async block, as it was before the call.
pub(crate) struct Candidate {
    block: usize,
    queue: usize,
    callback: usize,
    internal: [u8; 32],
}

struct Watched {
    block: usize,
    queue: usize,
    callback: usize,
    table: String,
    slot: usize,
    name: &'static str,
    caller: String,
    since: Instant,
    reported: bool,
}

static WATCHED: Mutex<Vec<Watched>> = Mutex::new(Vec::new());

fn executable(address: usize) -> bool {
    if address < 0x10000 {
        return false;
    }
    // SAFETY: an all zero `MEMORY_BASIC_INFORMATION` is a valid value of plain integers.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `VirtualQuery` only inspects the address and writes to `mbi`.
    let n = unsafe { VirtualQuery(address as *const c_void, &mut mbi, size_of::<MEMORY_BASIC_INFORMATION>()) };
    n == size_of::<MEMORY_BASIC_INFORMATION>()
        && mbi.State == MEM_COMMIT
        && mbi.Protect & PAGE_GUARD == 0
        && mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0
}

/// A copy of the block at `address` when it is readable.
fn read_block(address: usize) -> Option<XAsyncBlock> {
    if !(0x10000..0x8000_0000_0000).contains(&address) || !address.is_multiple_of(8) || !crate::exports::readable(address as *const c_void, size_of::<XAsyncBlock>()) {
        return None;
    }
    // SAFETY: readable, and every bit pattern is a valid `XAsyncBlock` of plain values.
    Some(unsafe { std::ptr::read(address as *const XAsyncBlock) })
}

/// The arguments of a call that look like async blocks.
pub(crate) fn candidates(args: &[usize]) -> Vec<Candidate> {
    let mut found = Vec::new();
    for &arg in args {
        let Some(b) = read_block(arg) else { continue };
        let queue = b.queue as usize;
        // `queue_from` of a null handle would make the process queue: check null first.
        let queue_ok = queue == 0 || crate::taskqueue::queue_from(b.queue).is_some();
        let callback_ok = b.callback == 0 || executable(b.callback);
        if queue_ok && callback_ok {
            found.push(Candidate { block: arg, queue, callback: b.callback, internal: b.internal });
        }
    }
    found
}

/// After a successful call: the candidates the PC runtime started are watched.
pub(crate) fn started(candidates: Vec<Candidate>, table: &str, slot: usize, name: &'static str, caller: &str, trace: bool) {
    for c in candidates {
        let Some(after) = read_block(c.block) else { continue };
        if after.queue as usize != c.queue || after.callback != c.callback || after.internal == c.internal || after.internal == [0; 32] {
            continue;
        }
        let mut watched = WATCHED.lock().unwrap_or_else(|e| e.into_inner());
        watched.retain(|w| w.block != c.block);
        if watched.len() >= MAX_WATCHED {
            drop(watched);
            emit(format!("windows {table} slot {slot} {name}: async block 0x{:x} not watched ({MAX_WATCHED} already)", c.block));
            continue;
        }
        watched.push(Watched { block: c.block, queue: c.queue, callback: c.callback, table: table.to_string(), slot, name, caller: caller.to_string(), since: Instant::now(), reported: false });
        drop(watched);
        if trace {
            emit(format!("windows {table} slot {slot} {name}: async block 0x{:x} started (queue 0x{:x}, callback 0x{:x}) from {caller}", c.block, c.queue, c.callback));
        }
        start_watcher();
    }
}

/// One pass over the watched blocks at `now`, with `status` the PC runtime's answer for a block.
pub(crate) fn sweep(now: Instant, status: &dyn Fn(usize) -> i32) {
    let mut lines = Vec::new();
    {
        let mut watched = WATCHED.lock().unwrap_or_else(|e| e.into_inner());
        watched.retain_mut(|w| {
            let age = now.saturating_duration_since(w.since);
            let hr = status(w.block);
            if hr == E_PENDING {
                if !w.reported && age >= Duration::from_secs(LATE_SECS) {
                    w.reported = true;
                    lines.push(format!(
                        "windows {} slot {} {}: async block 0x{:x} (queue 0x{:x}, callback 0x{:x}) from {} not completed after {LATE_SECS} s",
                        w.table, w.slot, w.name, w.block, w.queue, w.callback, w.caller
                    ));
                }
                return true;
            }
            if w.reported {
                lines.push(format!("windows {} slot {} {}: async block 0x{:x} completed late, after {} ms -> 0x{:08x}", w.table, w.slot, w.name, w.block, age.as_millis(), hr as u32));
            } else if hr != E_INVALIDARG && age >= Duration::from_secs(LATE_SECS) {
                lines.push(format!("windows {} slot {} {}: async block 0x{:x} completed after {} ms -> 0x{:08x}", w.table, w.slot, w.name, w.block, age.as_millis(), hr as u32));
            }
            false
        });
    }
    for line in lines {
        emit(line);
    }
}

/// `XAsyncGetStatus(block, false)` of the PC runtime, `E_INVALIDARG` when the block is no longer
/// readable or there is no PC XAsync table.
fn pc_status(block: usize) -> i32 {
    type GetStatus = unsafe extern "system" fn(*mut c_void, usize, bool) -> i32;
    static TABLE: OnceLock<usize> = OnceLock::new();
    let object = *TABLE.get_or_init(|| crate::exports::pc_query(&crate::local_table::XASYNC_TABLE, &crate::local_table::XASYNC_TABLE) as usize);
    if object == 0 || read_block(block).is_none() {
        return E_INVALIDARG;
    }
    // SAFETY: the PC runtime's XAsync object, its slot 3 is `XAsyncGetStatus`; the block was
    // started by the PC runtime and is still readable.
    unsafe {
        let f = std::mem::transmute::<usize, GetStatus>(*(*(object as *const *const usize)).add(3));
        f(object as *mut c_void, block, false)
    }
}

fn start_watcher() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if cfg!(test) || STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new().name("decant-xgr-async-watch".into()).spawn(|| {
        loop {
            std::thread::sleep(Duration::from_millis(SWEEP_MS));
            sweep(Instant::now(), &pc_status);
        }
    });
    if spawned.is_err() {
        emit("proxy: the async watcher thread could not be started".to_string());
    }
}
