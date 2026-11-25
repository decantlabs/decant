//! The decoder behind the `DecantXma2Decoder` table: `Open`, `Decode` and `Close` over the `Engine`.
#![allow(unsafe_code)]

use std::ffi::c_void;

use crate::engine::Engine;
use crate::{FORMAT_HEADER, Format, FormatError};

/// One open decoder.
pub struct Context {
    pub engine: Engine,
    channels: usize,
}

/// A diagnostic line, sent to the debugger when `DECANT_LOG` is set (and nowhere else).
fn log(text: &str) {
    if decant_log::debugger_enabled() {
        decant_log::debug_string(text);
    }
}

/// `Close`.
///
/// # Safety
/// `context` is null or a pointer `open` returned and that was not closed yet.
pub unsafe extern "C" fn close(context: *mut c_void) {
    if !context.is_null() {
        // SAFETY: the pointer came from `Box::into_raw` in `open` and is closed once.
        drop(unsafe { Box::from_raw(context.cast::<Context>()) });
    }
}

/// `Open`. `format`: a `WAVEFORMATEX` with `wFormatTag` 0x166 followed by its `cbSize` extra
/// bytes (the XMA2 part). Null for a format that is not XMA2 or cannot be decoded.
///
/// # Safety
/// `format` is null or points to `format_bytes` readable bytes.
pub unsafe extern "C" fn open(format: *const c_void, format_bytes: u32) -> *mut c_void {
    if format.is_null() || (format_bytes as usize) < FORMAT_HEADER {
        return std::ptr::null_mut();
    }
    // SAFETY: the caller gives `format_bytes` readable bytes at `format`.
    let f = unsafe { std::slice::from_raw_parts(format.cast::<u8>(), format_bytes as usize) };
    let parsed = match Format::parse(f) {
        Ok(parsed) => parsed,
        Err(FormatError::TooShort) => return std::ptr::null_mut(),
        Err(FormatError::NotXma2) => {
            log("[decant_xma2] not an XMA2 format
");
            return std::ptr::null_mut();
        }
    };
    match Engine::new(&parsed) {
        Ok(engine) => Box::into_raw(Box::new(Context {
            engine,
            channels: usize::from(parsed.channels),
        }))
        .cast(),
        Err(why) => {
            log(&format!(
                "[decant_xma2] unsupported XMA2 format: {why}
"
            ));
            std::ptr::null_mut()
        }
    }
}

/// `Decode`: decodes `block` (whole XMA2 packets of 2048 bytes) and hands out at most
/// `max_frames`. What does not fit is kept for the next call. A call with `block_bytes` 0 only
/// hands out what is kept. A null block ends the stream: the decoder is flushed, and further null
/// calls hand out the rest.
///
/// # Safety
/// `context` is null or from `open`. `block` is null or `block_bytes` readable bytes. `pcm`
/// holds `max_frames * channels` samples when `max_frames` is not 0. `frames_out` is null or
/// writable.
pub unsafe extern "C" fn decode(
    context: *mut c_void,
    block: *const u8,
    block_bytes: u32,
    pcm: *mut i16,
    max_frames: u32,
    frames_out: *mut u32,
) -> i32 {
    if context.is_null() || frames_out.is_null() || (max_frames != 0 && pcm.is_null()) {
        return -1;
    }
    // SAFETY: a non null context is a live `Context` from `open`, and `frames_out` is writable.
    let (c, frames_out) = unsafe { (&mut *context.cast::<Context>(), &mut *frames_out) };
    *frames_out = 0;
    if block.is_null() {
        c.engine.flush();
    } else if block_bytes > 0 {
        // SAFETY: the caller gives `block_bytes` readable bytes at `block`.
        c.engine
            .push(unsafe { std::slice::from_raw_parts(block, block_bytes as usize) });
    }
    let give = if max_frames == 0 {
        0
    } else {
        // SAFETY: the caller's buffer holds `max_frames` frames.
        let out = unsafe { std::slice::from_raw_parts_mut(pcm, max_frames as usize * c.channels) };
        c.engine.take(out, max_frames as usize)
    };
    *frames_out = give as u32;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_entry_points_refuse_bad_arguments() {
        let mut frames = 5u32;
        // SAFETY: null pointers are the refusals under test.
        unsafe {
            assert_eq!(
                decode(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                    &mut frames
                ),
                -1
            );
            assert_eq!(frames, 5);
            assert!(open(std::ptr::null(), 100).is_null());
            assert!(open([0u8; 4].as_ptr().cast(), 4).is_null());
            close(std::ptr::null_mut());
        }
    }
}
