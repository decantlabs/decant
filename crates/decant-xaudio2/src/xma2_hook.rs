//! The C ABI of an optional software XMA2 decoder DLL, for PCs without a hardware XMA decoder.
//!
//! XAudio2 2.9 on Windows accepts PCM, ADPCM and xWMA only. XMA is documented as Xbox only
//! (<https://learn.microsoft.com/en-us/windows/win32/xaudio2/xaudio2-redistributable>). A console
//! title that submits XMA2 data therefore needs a decoder somewhere. The layer does not contain
//! one. It looks for an optional decoder DLL named by the environment variable
//! `DECANT_XMA2_DECODER` and, when none is usable, rejects XMA formats with
//! `XAUDIO2_E_INVALID_CALL` and logs the rejection.
//!
//! A decoder DLL exports one C function, `int DecantXma2GetDecoder(DecantXma2Decoder* out)`, which
//! fills the table below and returns 0. The ABI is plain C, so any implementation (a licence
//! compatible library wrapped in a thin DLL, or the user's own) can be dropped in without
//! rebuilding the layer. Nothing in this repository downloads or bundles such a decoder. In C:
//!
//! ```c
//! typedef struct DecantXma2Decoder {
//!     uint32_t abi_version;  /* DECANT_XMA2_ABI_VERSION, 1 */
//!     void* (*Open)(const void* format, uint32_t format_bytes);
//!     int (*Decode)(void* context, const uint8_t* block, uint32_t block_bytes, int16_t* pcm,
//!                   uint32_t max_frames, uint32_t* frames_out);
//!     void (*Close)(void* context);
//! } DecantXma2Decoder;
//! ```
//!
//! One implementation is in this repository: the crate `decant-xma2` builds `decant_xma2.dll`, which
//! decodes with the pure Rust `decant-xma`.
//!
//! Status: the table is defined and the loader is in the layer. The layer does NOT yet feed
//! decoded PCM to a source voice. That needs a wrapper around
//! `IXAudio2SourceVoice::SubmitSourceBuffer`, which is the next step and is only worth writing
//! once a title is seen submitting XMA2 buffers.

use std::ffi::c_void;

/// `DECANT_XMA2_ABI_VERSION`.
pub const ABI_VERSION: u32 = 1;

/// `format`: the `XMA2WAVEFORMATEX` the title passed (`format_bytes` bytes, tag 0x166). Returns a
/// context or null.
pub type OpenFn = unsafe extern "C" fn(format: *const c_void, format_bytes: u32) -> *mut c_void;
/// Decodes one XMA2 data block (a multiple of 2048 bytes) into interleaved signed 16 bit PCM,
/// never more than `max_frames` frames. `*frames_out` receives the number written. 0 on success.
pub type DecodeFn = unsafe extern "C" fn(context: *mut c_void, block: *const u8, block_bytes: u32, pcm: *mut i16, max_frames: u32, frames_out: *mut u32) -> i32;
pub type CloseFn = unsafe extern "C" fn(context: *mut c_void);

/// The table a decoder DLL fills.
#[repr(C)]
#[derive(Default)]
pub struct DecantXma2Decoder {
    pub abi_version: u32,
    pub open: Option<OpenFn>,
    pub decode: Option<DecodeFn>,
    pub close: Option<CloseFn>,
}

/// `DecantXma2GetDecoder`, the one export of a decoder DLL.
pub type GetDecoderFn = unsafe extern "C" fn(out: *mut DecantXma2Decoder) -> i32;

impl DecantXma2Decoder {
    /// Whether the table is complete and of this ABI version.
    pub fn usable(&self) -> bool {
        self.abi_version == ABI_VERSION && self.open.is_some() && self.decode.is_some() && self.close.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_the_c_layout() {
        assert_eq!(size_of::<DecantXma2Decoder>(), 32);
        assert!(!DecantXma2Decoder::default().usable());
    }
}
