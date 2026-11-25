//! The two exports of `decant_xma2.dll`: `DecantXma2GetDecoder` (ordinal 1) and `DecantXma2Pending`
//! (ordinal 2).
#![allow(unsafe_code)]

use std::ffi::c_void;

use crate::abi::{ABI_VERSION, DecantXma2Decoder};

use crate::decoder::{Context, close, decode, open};

/// Fills the decoder table. -1 when `out` is null.
///
/// # Safety
/// Exported. `out` is null or points to a writable `DecantXma2Decoder`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DecantXma2GetDecoder(out: *mut DecantXma2Decoder) -> i32 {
    if out.is_null() {
        return -1;
    }
    // SAFETY: `out` is a writable table. Only the four members are written.
    unsafe {
        (&raw mut (*out).abi_version).write(ABI_VERSION);
        (&raw mut (*out).open).write(Some(open));
        (&raw mut (*out).decode).write(Some(decode));
        (&raw mut (*out).close).write(Some(close));
    }
    0
}

/// Frames decoded but not yet handed out, for a caller that drains after the last block.
///
/// # Safety
/// Exported. `context` is null or a pointer `Open` returned that was not closed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DecantXma2Pending(context: *mut c_void) -> u32 {
    if context.is_null() {
        return 0;
    }
    // SAFETY: a non null context is a live `Context` from `open`.
    unsafe { (*context.cast::<Context>()).engine.pending_frames() as u32 }
}
