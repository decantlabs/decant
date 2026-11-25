//! The six exports, under the names the former `decant_xa2.def` gave them. The C functions were
//! called `DecantXa2Create` and so on and aliased there. Here they carry the exported names directly.
#![allow(unsafe_code)]

use std::ffi::c_void;

use windows::core::HRESULT;

use crate::engine::{create, forward_fx};
use crate::stats::{STATS_SIZE, DecantXa2Stats, reset, snapshot};

/// # Safety
/// Exported. `out` is null or a writable interface pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XAudio2Create(out: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    // SAFETY: the caller's contract is passed on.
    let hr = unsafe { create(out, flags, processor) };
    decant_log::trace(&format!("XAudio2: XAudio2Create(flags {flags:#x}, processor {processor:#x}) -> {:#010x}", hr.0 as u32));
    hr
}

/// The Windows 10 headers call this one when it exists. The extra argument is the SDK version the
/// caller was built for.
///
/// # Safety
/// Exported. `out` is null or a writable interface pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XAudio2CreateWithVersionInfo(out: *mut *mut c_void, flags: u32, processor: u32, _ntddi_version: u32) -> HRESULT {
    // SAFETY: the caller's contract is passed on.
    let hr = unsafe { create(out, flags, processor) };
    decant_log::trace(&format!("XAudio2: XAudio2CreateWithVersionInfo(flags {flags:#x}, processor {processor:#x}) -> {:#010x}", hr.0 as u32));
    hr
}

/// # Safety
/// Exported. `out` is null or a writable interface pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn CreateAudioVolumeMeter(out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the caller's contract is passed on.
    unsafe { forward_fx("CreateAudioVolumeMeter", out) }
}

/// # Safety
/// Exported. `out` is null or a writable interface pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn CreateAudioReverb(out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the caller's contract is passed on.
    unsafe { forward_fx("CreateAudioReverb", out) }
}

/// Copies the counters when `out.struct_size` is at least the size of `DecantXa2Stats`.
///
/// # Safety
/// Exported. `out` is null or points to a `DecantXa2Stats` whose `struct_size` is set.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXa2GetStats(out: *mut DecantXa2Stats) {
    // SAFETY: a non null `out` points to at least `struct_size` writable bytes.
    unsafe {
        if !out.is_null() && (*out).struct_size >= STATS_SIZE {
            out.write(snapshot());
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn DecantXa2ResetStats() {
    reset();
}
