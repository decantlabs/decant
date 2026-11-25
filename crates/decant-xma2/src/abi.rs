//! The decoder table of `xma2_hook.h`, the ABI of a software XMA2 decoder DLL. The same types
//! live in `decant_xaudio2::xma2_hook`, the consumer. They are declared again here because a cdylib
//! that links the `decant-xaudio2` crate also exports that crate's `#[no_mangle]` functions,
//! and `decant_xma2.dll` must export `DecantXma2GetDecoder` and `DecantXma2Pending` only. A unit test
//! checks that both declarations have the same layout.

use std::ffi::c_void;

/// `DECANT_XMA2_ABI_VERSION`.
pub const ABI_VERSION: u32 = 1;

/// `Open`: `format` is the `XMA2WAVEFORMATEX` (`format_bytes` bytes, tag 0x166). Returns a
/// context or null.
pub type OpenFn = unsafe extern "C" fn(format: *const c_void, format_bytes: u32) -> *mut c_void;
/// `Decode`: interleaved signed 16 bit PCM, never more than `max_frames` frames. 0 on success.
pub type DecodeFn = unsafe extern "C" fn(context: *mut c_void, block: *const u8, block_bytes: u32, pcm: *mut i16, max_frames: u32, frames_out: *mut u32) -> i32;
/// `Close`.
pub type CloseFn = unsafe extern "C" fn(context: *mut c_void);

/// The table `DecantXma2GetDecoder` fills (32 bytes).
#[repr(C)]
#[derive(Default)]
pub struct DecantXma2Decoder {
    pub abi_version: u32,
    pub open: Option<OpenFn>,
    pub decode: Option<DecodeFn>,
    pub close: Option<CloseFn>,
}

/// `DecantXma2GetDecoder`, the one export a decoder DLL needs.
pub type GetDecoderFn = unsafe extern "C" fn(out: *mut DecantXma2Decoder) -> i32;

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use decant_xaudio2::xma2_hook as hook;

    #[test]
    fn the_table_has_the_layout_of_the_consumers_declaration() {
        assert_eq!(size_of::<DecantXma2Decoder>(), 32);
        assert_eq!(size_of::<DecantXma2Decoder>(), size_of::<hook::DecantXma2Decoder>());
        assert_eq!(offset_of!(DecantXma2Decoder, abi_version), offset_of!(hook::DecantXma2Decoder, abi_version));
        assert_eq!(offset_of!(DecantXma2Decoder, open), offset_of!(hook::DecantXma2Decoder, open));
        assert_eq!(offset_of!(DecantXma2Decoder, decode), offset_of!(hook::DecantXma2Decoder, decode));
        assert_eq!(offset_of!(DecantXma2Decoder, close), offset_of!(hook::DecantXma2Decoder, close));
        assert_eq!(ABI_VERSION, hook::ABI_VERSION);
    }
}
