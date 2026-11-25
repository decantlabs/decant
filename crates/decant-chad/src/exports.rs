//! The 7 exports of `chad.dll`, by name, with the prototypes of `docs/CONSOLE_LIBS.md` 4.1. Windows x64 convention.
#![allow(unsafe_code)]

use std::ffi::c_void;

use decant_log::trace;

use crate::abi::*;
use crate::engine;

pub const EXPORT_NAMES: [&str; 7] = [
    "XDspConnect",
    "XDspConnectWithMaximumStreamLimit",
    "XDspDisconnect",
    "XDspActivate",
    "XDspSubmitCommand",
    "XDspSubmitCommandWithEnvelope",
    "XDspDeactivate",
];

fn failed(function: &str, code: Hresult) -> Hresult {
    trace(&format!("chad: {function} failed with {code:#010x}"));
    code
}

/// # Safety
/// `memory` is memory of `memory_size` bytes the DSP may use, `client` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XDspConnectWithMaximumStreamLimit(memory: *mut c_void, memory_size: u32, max_activations: u32, max_streams: u32, priority: u32, client: *mut *mut XdspClient) -> Hresult {
    if !client.is_null() {
        // SAFETY: writable as documented.
        unsafe { *client = std::ptr::null_mut() };
    }
    if !engine::connect_args_valid(memory as usize, memory_size, max_activations, max_streams, priority) {
        return failed("XDspConnectWithMaximumStreamLimit", E_INVALIDARG);
    }
    if client.is_null() {
        return failed("XDspConnectWithMaximumStreamLimit", E_POINTER);
    }
    match engine::connect(memory as usize, memory_size, max_activations, max_streams, priority) {
        Ok(handle) => {
            // SAFETY: writable as documented.
            unsafe { *client = handle };
            S_OK
        }
        Err(e) => failed("XDspConnectWithMaximumStreamLimit", e),
    }
}

/// # Safety
/// As for `XDspConnectWithMaximumStreamLimit`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XDspConnect(memory: *mut c_void, memory_size: u32, max_activations: u32, client: *mut *mut XdspClient) -> Hresult {
    // SAFETY: the same contract.
    unsafe { XDspConnectWithMaximumStreamLimit(memory, memory_size, max_activations, MAX_STREAMS, MAX_PRIORITY, client) }
}

#[unsafe(no_mangle)]
pub extern "system" fn XDspDisconnect(client: *mut XdspClient) -> Hresult {
    if client.is_null() {
        return failed("XDspDisconnect", E_INVALIDARG);
    }
    let code = engine::disconnect(client as usize);
    if code == S_OK { S_OK } else { failed("XDspDisconnect", code) }
}

/// # Safety
/// `params` is null or points at 0x1C bytes. `status` and `context` are null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XDspActivate(client: *mut XdspClient, params: *const RawActivationParameters, status: *mut *mut u32, context: *mut *mut XdspContext) -> Hresult {
    // SAFETY: writable or null, as documented.
    unsafe {
        if !status.is_null() {
            *status = std::ptr::null_mut();
        }
        if !context.is_null() {
            *context = std::ptr::null_mut();
        }
    }
    if status.is_null() || context.is_null() {
        return failed("XDspActivate", E_POINTER);
    }
    if client.is_null() || params.is_null() {
        return failed("XDspActivate", E_INVALIDARG);
    }
    // SAFETY: 0x1C bytes as documented.
    let bytes = unsafe { params.read_unaligned() };
    let p = ActivationParameters::from_bytes(&bytes);
    match engine::activate(client as usize, &p) {
        Ok(a) => {
            // SAFETY: writable as checked above.
            unsafe {
                *status = a.status;
                *context = a.context;
            }
            S_OK
        }
        Err(e) => failed("XDspActivate", e),
    }
}

fn submit(context: *mut XdspContext, command: *const XdspCommand, envelope: u64, sequence: *mut u32, name: &str) -> Hresult {
    if context.is_null() || command.is_null() {
        return failed(name, E_INVALIDARG);
    }
    // SAFETY: an `XDSP_COMMAND` of 0x20 bytes as documented.
    let c = unsafe { command.read_unaligned() };
    match engine::submit(context as usize, &c, envelope) {
        Ok(n) => {
            if !sequence.is_null() {
                // SAFETY: writable as documented.
                unsafe { sequence.write_unaligned(n) };
            }
            S_OK
        }
        Err(e) => failed(name, e),
    }
}

/// # Safety
/// `command` is null or points at an `XDSP_COMMAND`, `sequence` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XDspSubmitCommand(context: *mut XdspContext, command: *const XdspCommand, sequence: *mut u32) -> Hresult {
    submit(context, command, 0, sequence, "XDspSubmitCommand")
}

/// # Safety
/// As for `XDspSubmitCommand`, and `envelope` is null or points at the gains.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XDspSubmitCommandWithEnvelope(context: *mut XdspContext, command: *const XdspCommand, envelope: *const f32, sequence: *mut u32) -> Hresult {
    submit(context, command, envelope as u64, sequence, "XDspSubmitCommandWithEnvelope")
}

#[unsafe(no_mangle)]
pub extern "system" fn XDspDeactivate(context: *mut XdspContext) -> Hresult {
    if context.is_null() {
        return failed("XDspDeactivate", E_INVALIDARG);
    }
    let code = engine::deactivate(context as usize);
    if code == S_OK { S_OK } else { failed("XDspDeactivate", code) }
}
