//! XMA2 decoding in pure Rust.
//!
//! An XMA2 stream is a sequence of 2048 byte packets. A packet starts with a 32 bit header, most significant bit first. It
//! holds the number of frames that begin in the packet (6 bits), the number of bits at the start of the payload that still
//! belong to the frame of the previous packet (15 bits), 3 bits that are ignored and the number of packets of other streams
//! that follow before the next packet of this stream (8 bits). A frame starts with its own length in bits (15 bits) and ends
//! with a bit that tells whether another frame follows in the same packet. A frame holds 512 samples per channel, mono or
//! stereo, coded in the WMA Pro style.
//!
//! The [`stream`] module walks frames by bit position, which is the unit that console contexts count in. The [`Decoder`]
//! turns frames into interleaved 16 bit samples. Only what XMA uses is supported: at most two channels and subframes of 512,
//! 256 and 128 samples.

mod bitio;
mod decoder;
mod imdct;
pub mod stream;
pub mod synth;
mod tables;

pub use decoder::{Decoder, Error};
pub use stream::{Frame, PacketHeader, first_frame, packet_header, read_frame};

/// Bytes in one packet.
pub const PACKET_BYTES: usize = 2048;
/// Bits in one packet.
pub const PACKET_BITS: usize = PACKET_BYTES * 8;
/// Bits of the packet header.
pub const HEADER_BITS: usize = 32;
/// Samples per channel in one frame.
pub const FRAME_SAMPLES: usize = 512;
/// Samples per channel in the smallest subframe, the unit of the skips and loop ends of a context.
pub const SUBFRAME_SAMPLES: usize = 128;
/// The four sample rates of the rate codes 0 to 3 of a context.
pub const RATES: [u32; 4] = [24_000, 32_000, 44_100, 48_000];
