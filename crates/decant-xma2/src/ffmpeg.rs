//! FFmpeg, loaded at run time. The libraries are found in `DECANT_FFMPEG_DIR`, else this DLL's
//! own folder, and the functions are looked up by name. The structures are declared here by hand
//! for libavcodec 62 and libavutil 60 (FFmpeg 8.1): only the offsets of the few fields the decoder
//! touches. The offsets were measured with `offsetof` against the headers of FFmpeg n8.1. Inside
//! a major version FFmpeg only appends fields, so they hold for every 62.x and 60.x library.
#![allow(unsafe_code)]

use std::ffi::{OsStr, OsString, c_int, c_void};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW, GetProcAddress,
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW,
};

use crate::{avcodec_dll, avutil_dll, folder_from_variable, versions_match};

/// Opaque FFmpeg types, only ever used behind pointers FFmpeg gave out.
pub enum AVCodec {}
pub enum AVCodecContext {}
pub enum AVPacket {}
pub enum AVFrame {}
pub enum AVDictionary {}
/// `AVChannelLayout`, 24 bytes, only handed to `av_channel_layout_default`.
pub enum AVChannelLayout {}

/// `AV_CODEC_ID_XMA2`.
pub const AV_CODEC_ID_XMA2: c_int = 86096;
/// `AV_INPUT_BUFFER_PADDING_SIZE`.
pub const AV_INPUT_BUFFER_PADDING_SIZE: usize = 64;
/// `AV_SAMPLE_FMT_S16`, `AV_SAMPLE_FMT_FLT`, `AV_SAMPLE_FMT_S16P`, `AV_SAMPLE_FMT_FLTP`.
pub const AV_SAMPLE_FMT_S16: c_int = 1;
pub const AV_SAMPLE_FMT_FLT: c_int = 3;
pub const AV_SAMPLE_FMT_S16P: c_int = 6;
pub const AV_SAMPLE_FMT_FLTP: c_int = 8;
/// `AVERROR(EAGAIN)`, with the Windows value of `EAGAIN` (11).
pub const AVERROR_EAGAIN: c_int = -11;
/// `AVERROR_EOF`, `-MKTAG('E','O','F',' ')`.
pub const AVERROR_EOF: c_int = -541_478_725;
/// `AVERROR_INVALIDDATA`, `-MKTAG('I','N','D','A')`.
pub const AVERROR_INVALIDDATA: c_int = -1_094_995_529;

/// Byte offsets of the fields used, in the structures of libavcodec 62 and libavutil 60.
mod offset {
    pub const CODEC_EXTRADATA: usize = 72;
    pub const CODEC_EXTRADATA_SIZE: usize = 80;
    pub const CODEC_SAMPLE_RATE: usize = 344;
    pub const CODEC_CH_LAYOUT: usize = 352;
    pub const CODEC_BLOCK_ALIGN: usize = 380;
    pub const PACKET_DATA: usize = 24;
    pub const PACKET_SIZE: usize = 32;
    pub const FRAME_EXTENDED_DATA: usize = 96;
    pub const FRAME_NB_SAMPLES: usize = 112;
    pub const FRAME_FORMAT: usize = 116;
    pub const FRAME_CH_LAYOUT: usize = 384;
    /// `nb_channels` inside an `AVChannelLayout`.
    pub const LAYOUT_NB_CHANNELS: usize = 4;
}

/// Reads or writes a field at a byte offset of an FFmpeg structure.
///
/// # Safety
/// `base` points to a live FFmpeg structure of the declared version that has a `T` at `offset`.
unsafe fn field<T: Copy>(base: *const c_void, offset: usize) -> T {
    // SAFETY: the caller states that a `T` lives at `offset`. FFmpeg's fields are naturally aligned.
    unsafe { base.byte_add(offset).cast::<T>().read() }
}

/// # Safety
/// As `field`, and `base` is writable.
unsafe fn set_field<T>(base: *mut c_void, offset: usize, value: T) {
    // SAFETY: as `field`, and the structure is writable.
    unsafe { base.byte_add(offset).cast::<T>().write(value) }
}

impl Ff {
    // The accessors below only read or write what the C code did. Each takes a pointer FFmpeg
    // returned from the matching allocation function of the loaded version.

    pub unsafe fn set_sample_rate(&self, codec: *mut AVCodecContext, rate: c_int) {
        // SAFETY: `codec` is a live context.
        unsafe { set_field(codec.cast(), offset::CODEC_SAMPLE_RATE, rate) }
    }
    pub unsafe fn set_block_align(&self, codec: *mut AVCodecContext, block_align: c_int) {
        // SAFETY: `codec` is a live context.
        unsafe { set_field(codec.cast(), offset::CODEC_BLOCK_ALIGN, block_align) }
    }
    pub unsafe fn block_align(&self, codec: *const AVCodecContext) -> c_int {
        // SAFETY: `codec` is a live context.
        unsafe { field(codec.cast(), offset::CODEC_BLOCK_ALIGN) }
    }
    pub unsafe fn set_default_layout(&self, codec: *mut AVCodecContext, channels: c_int) {
        // SAFETY: `codec` is a live context and its `ch_layout` is the `AVChannelLayout` at the offset.
        unsafe { (self.av_channel_layout_default)(codec.cast::<u8>().add(offset::CODEC_CH_LAYOUT).cast(), channels) }
    }
    pub unsafe fn set_extradata(&self, codec: *mut AVCodecContext, data: *mut u8, size: c_int) {
        // SAFETY: `codec` is a live context.
        unsafe {
            set_field(codec.cast(), offset::CODEC_EXTRADATA, data);
            set_field(codec.cast(), offset::CODEC_EXTRADATA_SIZE, size);
        }
    }
    pub unsafe fn set_packet(&self, packet: *mut AVPacket, data: *mut u8, size: c_int) {
        // SAFETY: `packet` is a live packet.
        unsafe {
            set_field(packet.cast(), offset::PACKET_DATA, data);
            set_field(packet.cast(), offset::PACKET_SIZE, size);
        }
    }
    pub unsafe fn frame_nb_samples(&self, frame: *const AVFrame) -> c_int {
        // SAFETY: `frame` is a live frame.
        unsafe { field(frame.cast(), offset::FRAME_NB_SAMPLES) }
    }
    pub unsafe fn frame_format(&self, frame: *const AVFrame) -> c_int {
        // SAFETY: `frame` is a live frame.
        unsafe { field(frame.cast(), offset::FRAME_FORMAT) }
    }
    pub unsafe fn frame_channels(&self, frame: *const AVFrame) -> c_int {
        // SAFETY: `frame` is a live frame, its `ch_layout` an `AVChannelLayout`.
        unsafe { field(frame.cast(), offset::FRAME_CH_LAYOUT + offset::LAYOUT_NB_CHANNELS) }
    }
    pub unsafe fn frame_planes(&self, frame: *const AVFrame) -> *const *const u8 {
        // SAFETY: `frame` is a live frame.
        unsafe { field(frame.cast(), offset::FRAME_EXTENDED_DATA) }
    }
}

/// The FFmpeg functions used, looked up by name.
pub struct Ff {
    pub avcodec_version: unsafe extern "C" fn() -> u32,
    pub avcodec_find_decoder: unsafe extern "C" fn(c_int) -> *const AVCodec,
    pub avcodec_alloc_context3: unsafe extern "C" fn(*const AVCodec) -> *mut AVCodecContext,
    pub avcodec_open2: unsafe extern "C" fn(*mut AVCodecContext, *const AVCodec, *mut *mut AVDictionary) -> c_int,
    pub avcodec_free_context: unsafe extern "C" fn(*mut *mut AVCodecContext),
    pub avcodec_send_packet: unsafe extern "C" fn(*mut AVCodecContext, *const AVPacket) -> c_int,
    pub avcodec_receive_frame: unsafe extern "C" fn(*mut AVCodecContext, *mut AVFrame) -> c_int,
    pub av_packet_alloc: unsafe extern "C" fn() -> *mut AVPacket,
    pub av_packet_free: unsafe extern "C" fn(*mut *mut AVPacket),
    pub avutil_version: unsafe extern "C" fn() -> u32,
    pub av_frame_alloc: unsafe extern "C" fn() -> *mut AVFrame,
    pub av_frame_free: unsafe extern "C" fn(*mut *mut AVFrame),
    pub av_frame_unref: unsafe extern "C" fn(*mut AVFrame),
    pub av_mallocz: unsafe extern "C" fn(usize) -> *mut c_void,
    av_channel_layout_default: unsafe extern "C" fn(*mut AVChannelLayout, c_int),
}

/// A diagnostic line, sent to the debugger when `DECANT_LOG` is set (and nowhere else, as in the C code).
pub fn log(text: &str) {
    if decant_log::debugger_enabled() {
        decant_log::debug_string(text);
    }
}

/// The folder FFmpeg is loaded from: `DECANT_FFMPEG_DIR`, else this DLL's folder.
fn ffmpeg_folder() -> Option<OsString> {
    if let Some(folder) = folder_from_variable(std::env::var_os("DECANT_FFMPEG_DIR")) {
        return Some(folder);
    }
    let mut module: HMODULE = std::ptr::null_mut();
    // SAFETY: the address belongs to this module, and the reference count is left alone.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            ffmpeg_folder as *const () as *const u16,
            &mut module,
        )
    };
    if found == 0 {
        return None;
    }
    let mut path = [0u16; crate::FOLDER_UNITS];
    // SAFETY: `path` is writable for its length.
    let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    if n == 0 || n >= path.len() {
        return None;
    }
    let slash = path[..n].iter().rposition(|&u| u == u16::from(b'\\'))?;
    Some(OsString::from_wide(&path[..slash]))
}

fn load_from(folder: &OsStr, name: &str) -> Option<HMODULE> {
    let mut path: Vec<u16> = folder.encode_wide().collect();
    path.push(u16::from(b'\\'));
    path.extend(name.encode_utf16());
    // The C code formatted into a buffer of MAX_PATH * 2 units and gave up when it did not fit.
    if path.len() >= crate::FOLDER_UNITS * 2 {
        return None;
    }
    path.push(0);
    // The library's own folder first, so avcodec finds the avutil next to it.
    // SAFETY: `path` is a zero terminated path that lives for the call.
    let module = unsafe { LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) };
    (!module.is_null()).then_some(module)
}

/// The function `name` (zero terminated) of `module` as the function pointer type `F`.
///
/// # Safety
/// `F` is the function pointer type of the real signature of that function.
unsafe fn lookup<F: Copy>(module: HMODULE, name: &'static str) -> Option<F> {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    debug_assert!(name.ends_with('\0'));
    // SAFETY: the module is loaded and the name is zero terminated.
    let address = unsafe { GetProcAddress(module, name.as_ptr()) }?;
    // SAFETY: both are function pointers of the same size, and the caller states the type.
    Some(unsafe { std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, F>(&address) })
}

/// Looks one function up by name, as the function type of the field it fills.
macro_rules! get {
    ($module:expr, $name:ident) => {
        // SAFETY: the field's type is the function's signature in the FFmpeg version this crate declares.
        unsafe { lookup($module, concat!(stringify!($name), "\0")) }?
    };
}

fn resolve(codec: HMODULE, util: HMODULE) -> Option<Ff> {
    Some(Ff {
        avcodec_version: get!(codec, avcodec_version),
        avcodec_find_decoder: get!(codec, avcodec_find_decoder),
        avcodec_alloc_context3: get!(codec, avcodec_alloc_context3),
        avcodec_open2: get!(codec, avcodec_open2),
        avcodec_free_context: get!(codec, avcodec_free_context),
        avcodec_send_packet: get!(codec, avcodec_send_packet),
        avcodec_receive_frame: get!(codec, avcodec_receive_frame),
        av_packet_alloc: get!(codec, av_packet_alloc),
        av_packet_free: get!(codec, av_packet_free),
        avutil_version: get!(util, avutil_version),
        av_frame_alloc: get!(util, av_frame_alloc),
        av_frame_free: get!(util, av_frame_free),
        av_frame_unref: get!(util, av_frame_unref),
        av_mallocz: get!(util, av_mallocz),
        av_channel_layout_default: get!(util, av_channel_layout_default),
    })
}

fn load() -> Option<Ff> {
    let Some(folder) = ffmpeg_folder() else {
        log("[decant_xma2] no folder to load FFmpeg from\r\n");
        return None;
    };
    let util = load_from(&folder, &avutil_dll());
    let codec = util.and_then(|_| load_from(&folder, &avcodec_dll()));
    let (Some(util), Some(codec)) = (util, codec) else {
        log("[decant_xma2] avcodec/avutil of the expected major version not found (set DECANT_FFMPEG_DIR)\r\n");
        return None;
    };
    let Some(ff) = resolve(codec, util) else {
        log("[decant_xma2] an FFmpeg function is missing\r\n");
        return None;
    };
    // SAFETY: both are plain version queries of the libraries just loaded.
    let (avcodec, avutil) = unsafe { ((ff.avcodec_version)(), (ff.avutil_version)()) };
    if !versions_match(avcodec, avutil) {
        log("[decant_xma2] FFmpeg major version differs from the headers this DLL was built with\r\n");
        return None;
    }
    Some(ff)
}

/// Loads FFmpeg once. `None` when a library or function is missing or the versions differ. The
/// reason is logged the first time only, as the C code did.
pub fn ffmpeg() -> Option<&'static Ff> {
    static FF: OnceLock<Option<Ff>> = OnceLock::new();
    FF.get_or_init(load).as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_codes_are_ffmpegs() {
        let mktag = |a: u8, b: u8, c: u8, d: u8| i32::from_le_bytes([a, b, c, d]);
        assert_eq!(AVERROR_EOF, -mktag(b'E', b'O', b'F', b' '));
        assert_eq!(AVERROR_INVALIDDATA, -mktag(b'I', b'N', b'D', b'A'));
    }

    #[test]
    fn fields_are_read_at_their_offsets() {
        let mut block = [0u32; 128];
        // SAFETY: the buffer is large and aligned enough for every offset used.
        unsafe {
            let base = block.as_mut_ptr().cast::<c_void>();
            set_field(base, offset::CODEC_SAMPLE_RATE, 48_000i32);
            assert_eq!(field::<i32>(base, offset::CODEC_SAMPLE_RATE), 48_000);
        }
        assert_eq!(block[344 / 4], 48_000);
    }
}
