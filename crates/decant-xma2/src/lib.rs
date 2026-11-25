//! `decant_xma2.dll`: a software XMA2 decoder behind the `DecantXma2Decoder` ABI of `xma2_hook.h`
//! (declared in `abi`, consumed by `decant_xaudio2::xma2_hook`). A port of the former `src/decant/xaudio2/decant_xma2.c`, with
//! the same exports, ABI, FFmpeg calls, error handling and log text.
//!
//! The decoding itself is FFmpeg's xma2 decoder (libavcodec, LGPL). Nothing of FFmpeg is in this
//! repository or linked into this DLL: `avcodec` and `avutil` are loaded at run time with
//! `LoadLibraryExW` from the folder named by `DECANT_FFMPEG_DIR`, or else this DLL's own folder,
//! and their functions are looked up by name with `GetProcAddress`. The user supplies an LGPL
//! shared build of FFmpeg and the libraries stay replaceable. The
//! few FFmpeg structures used are declared by hand in `ffmpeg` for the major versions below, and
//! the loaded libraries must be exactly those (checked at load time). The crate therefore builds
//! without FFmpeg.
//!
//! XMA2 is a fallback: titles mostly use other codecs, and this only lets the
//! remaining XMA2 streams be heard instead of staying silent.
//!
//! The format rules, the sample conversion and the folder rules are plain functions here. The
//! FFmpeg loader, the decoder and the exports are in `ffmpeg`, `decoder` and `exports`, the only
//! modules with `unsafe` code.

pub mod abi;
#[cfg(windows)]
mod decoder;
#[cfg(windows)]
mod exports;
#[cfg(windows)]
mod ffmpeg;

use std::ffi::OsString;

/// Major version of libavcodec (FFmpeg 8.1) the declared structures follow.
pub const AVCODEC_MAJOR: u32 = 62;
/// Major version of libavutil (FFmpeg 8.1) the declared structures follow.
pub const AVUTIL_MAJOR: u32 = 60;

/// Size of the buffer the C code read `DECANT_FFMPEG_DIR` into, in UTF-16 units.
pub(crate) const FOLDER_UNITS: usize = 260;

/// The file name of the libavcodec this DLL loads.
pub fn avcodec_dll() -> String {
    format!("avcodec-{AVCODEC_MAJOR}.dll")
}

/// The file name of the libavutil this DLL loads.
pub fn avutil_dll() -> String {
    format!("avutil-{AVUTIL_MAJOR}.dll")
}

/// Whether the loaded libraries are the major versions the declared structures follow. `avcodec`
/// and `avutil` are the values of `avcodec_version()` and `avutil_version()`.
pub fn versions_match(avcodec: u32, avutil: u32) -> bool {
    avcodec >> 16 == AVCODEC_MAJOR && avutil >> 16 == AVUTIL_MAJOR
}

/// The folder named by `DECANT_FFMPEG_DIR`, if it is set, not empty and fits the 260 unit
/// buffer of the C code (a longer value was ignored there, so the DLL's own folder was used).
pub fn folder_from_variable(value: Option<OsString>) -> Option<OsString> {
    let value = value?;
    let units = wide_len(&value);
    (units > 0 && units < FOLDER_UNITS).then_some(value)
}

#[cfg(windows)]
fn wide_len(value: &OsString) -> usize {
    std::os::windows::ffi::OsStrExt::encode_wide(value.as_os_str()).count()
}

#[cfg(not(windows))]
fn wide_len(value: &OsString) -> usize {
    value.to_string_lossy().encode_utf16().count()
}

/// Bytes of a `WAVEFORMATEX` before its `cbSize` extra bytes.
pub const FORMAT_HEADER: usize = 18;
/// `WAVE_FORMAT_XMA2`.
pub const WAVE_FORMAT_XMA2: u16 = 0x166;

/// The parts of the `XMA2WAVEFORMATEX` the decoder needs.
#[derive(Debug, PartialEq, Eq)]
pub struct Format {
    pub channels: u16,
    pub rate: u32,
    /// 0 when the format leaves it open (the decoder then uses 2048).
    pub block_align: u16,
    /// The `cbSize` extra bytes after the `WAVEFORMATEX`: the decoder's extradata.
    pub extra: u16,
}

/// Why a format is refused.
#[derive(Debug, PartialEq, Eq)]
pub enum FormatError {
    /// Fewer bytes than a `WAVEFORMATEX` with `cbSize`: refused silently.
    TooShort,
    /// Not XMA2, an unusable channel count or rate, or `cbSize` beyond the bytes given.
    NotXma2,
}

impl Format {
    /// Reads the format the way the C code did.
    pub fn parse(bytes: &[u8]) -> Result<Format, FormatError> {
        if bytes.len() < FORMAT_HEADER {
            return Err(FormatError::TooShort);
        }
        let u16_at = |at: usize| u16::from(bytes[at]) | u16::from(bytes[at + 1]) << 8;
        let tag = u16_at(0);
        let channels = u16_at(2);
        let rate = u32::from(u16_at(4)) | u32::from(u16_at(6)) << 16;
        let block_align = u16_at(12);
        let extra = u16_at(16);
        if tag != WAVE_FORMAT_XMA2 || channels == 0 || channels > 8 || rate == 0 || FORMAT_HEADER + usize::from(extra) > bytes.len() {
            return Err(FormatError::NotXma2);
        }
        Ok(Format { channels, rate, block_align, extra })
    }
}

/// A float sample as signed 16 bit, rounded half away from zero and clipped.
pub fn to_short(v: f32) -> i16 {
    let scaled = v * 32768.0;
    if scaled >= 32767.0 {
        32767
    } else if scaled <= -32768.0 {
        -32768
    } else {
        (if scaled >= 0.0 { scaled + 0.5 } else { scaled - 0.5 }) as i32 as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xma2_format(channels: u16, rate: u32, block_align: u16, extra: u16) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend(WAVE_FORMAT_XMA2.to_le_bytes());
        f.extend(channels.to_le_bytes());
        f.extend(rate.to_le_bytes());
        f.extend((rate * 4).to_le_bytes());
        f.extend(block_align.to_le_bytes());
        f.extend(16u16.to_le_bytes());
        f.extend(extra.to_le_bytes());
        f.resize(FORMAT_HEADER + usize::from(extra), 0);
        f
    }

    #[test]
    fn a_good_format_is_read() {
        let f = Format::parse(&xma2_format(2, 48_000, 2048, 34)).unwrap();
        assert_eq!(f, Format { channels: 2, rate: 48_000, block_align: 2048, extra: 34 });
    }

    #[test]
    fn a_short_format_is_refused_silently() {
        assert_eq!(Format::parse(&[0x66, 1, 1, 0]), Err(FormatError::TooShort));
        assert_eq!(Format::parse(&xma2_format(1, 1, 0, 0)[..17]), Err(FormatError::TooShort));
    }

    #[test]
    fn other_formats_are_refused() {
        let mut other_tag = xma2_format(2, 48_000, 2048, 34);
        other_tag[0] = 0x65;
        assert_eq!(Format::parse(&other_tag), Err(FormatError::NotXma2));
        assert_eq!(Format::parse(&xma2_format(0, 48_000, 2048, 34)), Err(FormatError::NotXma2));
        assert_eq!(Format::parse(&xma2_format(9, 48_000, 2048, 34)), Err(FormatError::NotXma2));
        assert_eq!(Format::parse(&xma2_format(2, 0, 2048, 34)), Err(FormatError::NotXma2));
        // cbSize says 34 extra bytes, but only 10 follow.
        assert_eq!(Format::parse(&xma2_format(2, 48_000, 2048, 34)[..28]), Err(FormatError::NotXma2));
    }

    #[test]
    fn eight_channels_and_an_open_block_align_are_accepted() {
        let f = Format::parse(&xma2_format(8, 44_100, 0, 0)).unwrap();
        assert_eq!((f.channels, f.rate, f.block_align, f.extra), (8, 44_100, 0, 0));
    }

    #[test]
    fn the_library_names_follow_the_major_versions() {
        assert_eq!(avcodec_dll(), "avcodec-62.dll");
        assert_eq!(avutil_dll(), "avutil-60.dll");
    }

    #[test]
    fn only_the_expected_major_versions_are_accepted() {
        assert!(versions_match(62 << 16 | 12 << 8 | 100, 60 << 16 | 26 << 8 | 103));
        assert!(!versions_match(61 << 16, 60 << 16));
        assert!(!versions_match(62 << 16, 59 << 16));
        assert!(!versions_match(63 << 16, 61 << 16));
    }

    #[test]
    fn the_folder_variable_must_fit_the_c_buffer() {
        assert_eq!(folder_from_variable(None), None);
        assert_eq!(folder_from_variable(Some("".into())), None);
        assert_eq!(folder_from_variable(Some("C:\\ffmpeg\\bin".into())), Some("C:\\ffmpeg\\bin".into()));
        assert!(folder_from_variable(Some("a".repeat(259).into())).is_some());
        assert_eq!(folder_from_variable(Some("a".repeat(260).into())), None);
    }

    #[test]
    fn samples_are_rounded_away_from_zero_and_clipped() {
        assert_eq!(to_short(0.0), 0);
        assert_eq!(to_short(0.5), 16384);
        assert_eq!(to_short(-0.5), -16384);
        assert_eq!(to_short(1.0), 32767);
        assert_eq!(to_short(5.0), 32767);
        assert_eq!(to_short(-1.0), -32768);
        assert_eq!(to_short(-5.0), -32768);
        // 1.5 / 32768 scales to 1.5, which rounds away from zero to 2.
        assert_eq!(to_short(1.5 / 32768.0), 2);
        assert_eq!(to_short(-1.5 / 32768.0), -2);
        assert_eq!(to_short(0.4 / 32768.0), 0);
        assert_eq!(to_short(f32::NAN), 0);
    }
}
