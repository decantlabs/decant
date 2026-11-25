//! `decant_xma2.dll`: a software XMA2 decoder behind the `DecantXma2Decoder` ABI of `xma2_hook.h`
//! (declared in `abi`, consumed by `decant_xaudio2::xma2_hook`).
//!
//! The decoding is done by `decant-xma`, the pure Rust decoder of this repository, so the DLL has no
//! dependency and needs nothing installed next to it. `engine` maps an `XMA2WAVEFORMATEX` to one
//! `decant-xma` decoder per XMA stream (one or two channels each), splits the packets of a data
//! block between the streams by their skip counts and interleaves the output. `decoder` and
//! `exports` are the C entry points, the only modules with `unsafe` code.
//!
//! Supported: 1 to 8 channels in streams of two channels (the last one has one when the count is
//! odd), packets of 2048 bytes, sample rates mapped to the four XMA rates. Refused at `Open` with a
//! log line: other packet sizes and a stream count that does not match the channel count. The play
//! and loop fields of the format are not applied, the caller gets the whole stream.

pub mod abi;
mod decoder;
pub mod engine;
mod exports;

pub use exports::{DecantXma2GetDecoder, DecantXma2Pending};

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
    /// The `cbSize` extra bytes after the `WAVEFORMATEX`.
    pub extra: u16,
    /// `NumStreams`, the first field of the extra bytes. 0 when they are too short to hold it or
    /// say 0, and the stream count then follows from the channel count.
    pub streams: u16,
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
    /// Reads the `WAVEFORMATEX` and the stream count behind it.
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
        if tag != WAVE_FORMAT_XMA2
            || channels == 0
            || channels > 8
            || rate == 0
            || FORMAT_HEADER + usize::from(extra) > bytes.len()
        {
            return Err(FormatError::NotXma2);
        }
        let streams = if extra >= 2 { u16_at(FORMAT_HEADER) } else { 0 };
        Ok(Format {
            channels,
            rate,
            block_align,
            extra,
            streams,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn xma2_format(channels: u16, rate: u32, block_align: u16, extra: u16) -> Vec<u8> {
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
        assert_eq!(
            f,
            Format {
                channels: 2,
                rate: 48_000,
                block_align: 2048,
                extra: 34,
                streams: 0
            }
        );
        let mut with_streams = xma2_format(4, 48_000, 2048, 34);
        with_streams[FORMAT_HEADER] = 2;
        assert_eq!(Format::parse(&with_streams).unwrap().streams, 2);
    }

    #[test]
    fn a_short_format_is_refused_silently() {
        assert_eq!(Format::parse(&[0x66, 1, 1, 0]), Err(FormatError::TooShort));
        assert_eq!(
            Format::parse(&xma2_format(1, 1, 0, 0)[..17]),
            Err(FormatError::TooShort)
        );
    }

    #[test]
    fn other_formats_are_refused() {
        let mut other_tag = xma2_format(2, 48_000, 2048, 34);
        other_tag[0] = 0x65;
        assert_eq!(Format::parse(&other_tag), Err(FormatError::NotXma2));
        assert_eq!(
            Format::parse(&xma2_format(0, 48_000, 2048, 34)),
            Err(FormatError::NotXma2)
        );
        assert_eq!(
            Format::parse(&xma2_format(9, 48_000, 2048, 34)),
            Err(FormatError::NotXma2)
        );
        assert_eq!(
            Format::parse(&xma2_format(2, 0, 2048, 34)),
            Err(FormatError::NotXma2)
        );
        // cbSize says 34 extra bytes, but only 10 follow.
        assert_eq!(
            Format::parse(&xma2_format(2, 48_000, 2048, 34)[..28]),
            Err(FormatError::NotXma2)
        );
    }

    #[test]
    fn eight_channels_and_an_open_block_align_are_accepted() {
        let f = Format::parse(&xma2_format(8, 44_100, 0, 0)).unwrap();
        assert_eq!(
            (f.channels, f.rate, f.block_align, f.extra, f.streams),
            (8, 44_100, 0, 0, 0)
        );
    }
}
