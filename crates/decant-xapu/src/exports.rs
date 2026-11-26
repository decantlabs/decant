//! The 6 exports of `xapu.dll`, by name, as listed in `docs/CONSOLE_LIBS.md`, section 3. Windows x64 convention.
#![allow(unsafe_code)]

use decant_log::trace;

use crate::abi::*;
use crate::engine;

pub const EXPORT_NAMES: [&str; 6] = ["XApuConnect", "XApuDisconnect", "XApuEnqueueCommand", "XApuDequeueResult", "XApuGetResultCount", "XApuGetUtilization"];

fn failed(function: &str, code: Hresult) -> Hresult {
    trace(&format!("xapu: {function} failed with {code:#010x}"));
    code
}

/// # Safety
/// `params` is null or points at 6 UINT32. `region` and `handle` are null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XApuConnect(params: *const ConnectParams, region: *mut ConnectionRegion, handle: *mut Handle) -> Hresult {
    if region.is_null() || handle.is_null() {
        return failed("XApuConnect", E_POINTER);
    }
    // SAFETY: writable as documented.
    unsafe {
        *handle = 0;
        std::ptr::write_bytes(region as *mut u8, 0, std::mem::size_of::<ConnectionRegion>());
    }
    if params.is_null() {
        return failed("XApuConnect", E_INVALIDARG);
    }
    // SAFETY: 6 UINT32 as documented.
    let p = unsafe { params.read_unaligned() };
    match engine::connect(&p) {
        Ok(c) => {
            // SAFETY: writable as documented.
            unsafe {
                *handle = c.handle;
                region.write_unaligned(ConnectionRegion { memory: c.memory, result_event: c.result_event });
            }
            trace(&format!("xapu: connected, process type {}, {} streams, depth {}, frame count {:#x}", p.process_type, p.stream_count, p.queue_depth, p.frame_count));
            S_OK
        }
        Err(e) => failed("XApuConnect", e),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn XApuDisconnect(handle: Handle) -> Hresult {
    let code = engine::disconnect(handle);
    if code == S_OK { S_OK } else { failed("XApuDisconnect", code) }
}

/// # Safety
/// `command` is null or points at an `XAPU_COMMAND` (0x38 bytes, 0x40 for a data command). `sequence` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XApuEnqueueCommand(handle: Handle, command: *const Command, sequence: *mut u32) -> Hresult {
    if command.is_null() {
        return failed("XApuEnqueueCommand", E_INVALIDARG);
    }
    let mut cmd = Command::default();
    // SAFETY: the command has at least its first 0x38 bytes, and 0x40 when it is a data command.
    unsafe {
        std::ptr::copy_nonoverlapping(command as *const u8, cmd.0.as_mut_ptr(), COMMAND_READ_SIZE);
        if cmd.kind() == COMMAND_DATA {
            std::ptr::copy_nonoverlapping((command as *const u8).add(COMMAND_READ_SIZE), cmd.0.as_mut_ptr().add(COMMAND_READ_SIZE), COMMAND_SIZE - COMMAND_READ_SIZE);
        }
    }
    match engine::enqueue(handle, &cmd) {
        Ok(n) => {
            static ENQUEUED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let k = ENQUEUED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if k <= 60 || k.is_multiple_of(1000) {
                trace(&format!("xapu: XApuEnqueueCommand #{k} kind {:#x} -> sequence {n}", cmd.kind()));
            }
            if !sequence.is_null() {
                // SAFETY: writable as documented.
                unsafe { sequence.write_unaligned(n) };
            }
            S_OK
        }
        Err(e) => failed("XApuEnqueueCommand", e),
    }
}

/// # Safety
/// `result` is null or points at 0x50 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XApuDequeueResult(handle: Handle, result: *mut XapuResult) -> Hresult {
    if result.is_null() {
        if !engine::is_connected(handle) {
            return failed("XApuDequeueResult", E_INVALIDARG);
        }
        return failed("XApuDequeueResult", E_POINTER);
    }
    match engine::dequeue(handle) {
        Ok(r) => {
            static DEQUEUED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let k = DEQUEUED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if k <= 60 || k.is_multiple_of(1000) {
                trace(&format!("xapu: XApuDequeueResult #{k} -> a result"));
            }
            // SAFETY: 0x50 writable bytes as documented.
            unsafe { result.write_unaligned(r) };
            S_OK
        }
        // No result is an expected answer of a poll, not worth a log line.
        Err(e) if e == XAPU_E_NO_RESULT => e,
        Err(e) => failed("XApuDequeueResult", e),
    }
}

/// # Safety
/// `dequeued_count` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XApuGetResultCount(handle: Handle, dequeued_count: *mut u32) -> u32 {
    let (count, dequeued) = engine::result_count(handle);
    if !dequeued_count.is_null() {
        // SAFETY: writable as documented.
        unsafe { dequeued_count.write_unaligned(dequeued) };
    }
    count
}

/// # Safety
/// `out` is null or points at two writable UINT32.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XApuGetUtilization(process_type: u32, kind: u32, out: *mut u32) -> Hresult {
    if out.is_null() {
        return failed("XApuGetUtilization", E_POINTER);
    }
    // SAFETY: two writable UINT32 as documented.
    unsafe { std::ptr::write_bytes(out as *mut u8, 0, 8) };
    if !(1..=3).contains(&process_type) || !(1..=2).contains(&kind) {
        return failed("XApuGetUtilization", E_INVALIDARG);
    }
    let (free, total) = engine::utilization();
    // SAFETY: as above.
    unsafe {
        out.write_unaligned(free);
        out.add(1).write_unaligned(total);
    }
    S_OK
}
