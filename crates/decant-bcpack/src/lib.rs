//! BCPack: the console's lossless packing of BCn texture blocks, which its DirectStorage hardware
//! unpacks on the way from disk to memory. This crate does the same on the CPU, for BC1, BC3, BC4,
//! BC5, BC6H and BC7 (BC2 has no packed form).
//!
//! A packed stream has no header. It is 12 or 16 rANS lanes sharing one byte stream ([`lanes`]).
//! The lanes' symbols make one record per block: two back references, one for the endpoints and
//! one for the indices, and the residuals of both against what they point at. A reference points at
//! one of the 256 blocks before, or for a block near the start of the stream at one of 255 seed
//! entries of the format, which stand in for blocks -1 to -255. The models, rank tables, BC7
//! control words and seed entries are constants of the format ([`tables`]).
//!
//! Blocks come out in stream order, which is the stored order of the texture, so the bytes can
//! replace the destination as they are (the detiling afterwards is that of any other data).

mod bc6;
mod bc7;
#[cfg(any(test, feature = "encode"))]
pub mod encode;
pub mod lanes;
mod model;
mod plain;
#[rustfmt::skip]
mod tables;

use lanes::{Lanes, Layout, SliceFeed};

/// A BCn format BCPack packs, by its `DSTORAGE_BCPACK_MODE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
}

impl Format {
    pub const ALL: [Format; 6] = [Format::Bc1, Format::Bc3, Format::Bc4, Format::Bc5, Format::Bc6h, Format::Bc7];

    /// The format of BCPack mode `mode` (1 = BC1 ... 7 = BC7); `None` for BC2 and unknown modes.
    pub fn from_mode(mode: u32) -> Option<Format> {
        Some(match mode {
            1 => Format::Bc1,
            3 => Format::Bc3,
            4 => Format::Bc4,
            5 => Format::Bc5,
            6 => Format::Bc6h,
            7 => Format::Bc7,
            _ => return None,
        })
    }

    /// The BCPack mode of the format.
    pub fn mode(self) -> u32 {
        match self {
            Format::Bc1 => 1,
            Format::Bc3 => 3,
            Format::Bc4 => 4,
            Format::Bc5 => 5,
            Format::Bc6h => 6,
            Format::Bc7 => 7,
        }
    }

    /// Bytes of one block.
    pub fn block_bytes(self) -> usize {
        match self {
            Format::Bc1 | Format::Bc4 => 8,
            _ => 16,
        }
    }

    /// The lanes of the format.
    pub fn layout(self) -> Layout {
        match self {
            Format::Bc1 => plain::BC1.layout,
            Format::Bc3 => plain::BC3.layout,
            Format::Bc4 => plain::BC4.layout,
            Format::Bc5 => plain::BC5.layout,
            Format::Bc6h => bc6::LAYOUT,
            Format::Bc7 => bc7::LAYOUT,
        }
    }
}

/// Why a stream did not unpack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The destination is empty or not a whole number of blocks.
    Size,
    /// The stream ends before every block is decoded (or is no BCPack stream of this format).
    Truncated,
    /// A record names something the format does not have (a BC7 control code or BC6H mode).
    Corrupt(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Size => write!(f, "the destination is not a whole number of blocks"),
            Error::Truncated => write!(f, "the stream ends before every block is decoded"),
            Error::Corrupt(what) => write!(f, "corrupt stream: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// What a decode found besides the blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Bytes of the stream the lanes took.
    pub taken: usize,
    /// BC6H and BC7 blocks with a value the block cannot hold (an anchor index with its top bit
    /// set, or two halves of a shared p-bit that differ). A correct stream has none.
    pub inconsistent: usize,
}

/// One format's step from records to blocks. It keeps the history the references read.
trait Blocks {
    /// Turns block `i`'s record into its bytes. Returns false when the block is inconsistent.
    fn block(&mut self, i: usize, record: &[u8], out: &mut [u8]) -> Result<bool, Error>;
}

/// Blocks decoded per batch of records (keeps the records in the cache).
const BATCH: usize = 512;

fn run(layout: Layout, block_bytes: usize, stage: &mut impl Blocks, src: &[u8], dst: &mut [u8]) -> Result<Report, Error> {
    let stride = layout.stride();
    let mut lanes = Lanes::new(layout);
    let mut feed = SliceFeed::new(src);
    let mut records = vec![0u8; BATCH * stride];
    let mut report = Report::default();
    for (batch, out) in dst.chunks_mut(BATCH * block_bytes).enumerate() {
        let blocks = out.len() / block_bytes;
        let records = &mut records[..blocks * stride];
        lanes.run(&mut feed, records)?;
        for (k, (record, block)) in records.chunks_exact(stride).zip(out.chunks_exact_mut(block_bytes)).enumerate() {
            if !stage.block(batch * BATCH + k, record, block)? {
                report.inconsistent += 1;
            }
        }
    }
    report.taken = feed.taken();
    Ok(report)
}

/// Unpacks the stream `src` of `format` into `dst`, whose length gives the number of blocks.
pub fn decode(format: Format, src: &[u8], dst: &mut [u8]) -> Result<Report, Error> {
    let bytes = format.block_bytes();
    if dst.is_empty() || !dst.len().is_multiple_of(bytes) {
        return Err(Error::Size);
    }
    let layout = format.layout();
    match format {
        Format::Bc1 => run(layout, bytes, &mut plain::Stage::new(&plain::BC1), src, dst),
        Format::Bc3 => run(layout, bytes, &mut plain::Stage::new(&plain::BC3), src, dst),
        Format::Bc4 => run(layout, bytes, &mut plain::Stage::new(&plain::BC4), src, dst),
        Format::Bc5 => run(layout, bytes, &mut plain::Stage::new(&plain::BC5), src, dst),
        Format::Bc6h => run(layout, bytes, &mut bc6::Stage::new(), src, dst),
        Format::Bc7 => run(layout, bytes, &mut bc7::Stage::new(), src, dst),
    }
}

/// [`decode`] into a new buffer of `len` bytes.
pub fn decode_to_vec(format: Format, src: &[u8], len: usize) -> Result<(Vec<u8>, Report), Error> {
    let mut out = vec![0u8; len];
    let report = decode(format, src, &mut out)?;
    Ok((out, report))
}

// Pieces the formats share.

/// Blocks a reference can reach back.
const HISTORY: usize = 256;
/// Seed entries per format.
const SEEDS: usize = 255;

/// Two 4-bit symbols as one byte: the first on the even bits, the second on the odd bits.
fn interleave(even: u8, odd: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0u8, |acc, k| acc | ((v >> k) & 1) << (2 * k));
    spread(even) | spread(odd) << 1
}

/// Zigzag code to a signed byte: 0, 1, 2, 3 -> 0, -1, 1, -2.
fn unzigzag(v: u8) -> u8 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

/// The endpoint and index references of a record. Paired layouts carry each as two 4-bit symbols
/// (lanes 0 and 1, two symbols each), the others as one symbol (lanes 0 and 1). The index
/// reference is direct; the endpoint reference is the index reference plus a zigzag offset.
fn references(record: &[u8], paired: bool) -> (usize, usize) {
    let (offset, index) = if paired { (interleave(record[0], record[1]), interleave(record[2], record[3])) } else { (record[0], record[1]) };
    (usize::from(index.wrapping_add(unzigzag(offset))), usize::from(index))
}

/// What reference `r` of block `i` reads: nothing (0), an earlier block, or a seed entry.
fn reach<'a, T>(i: usize, r: usize, history: &'a [T], seeds: &'a [T], zero: &'a T) -> &'a T {
    if r == 0 {
        zero
    } else if r <= i {
        &history[(i - r) % HISTORY]
    } else {
        &seeds[r - i - 1]
    }
}

/// Little endian bit writer for one block.
#[derive(Default)]
struct Bits {
    value: u128,
    at: u32,
}

impl Bits {
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 && self.at < 128 {
            self.value |= (u128::from(value) & ((1u128 << n) - 1)) << self.at;
        }
        self.at += n;
    }
}

/// A byte's even and odd bits as two zigzag coded 4-bit residuals, in pixel order: the odd half
/// first for rows 0 to 2 of an index set, the even half first for row 3.
fn index_pair(byte: u8, odd_first: bool) -> [i16; 2] {
    let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | ((v >> (2 * k)) & 1) << k);
    let signed = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (signed(half(byte)), signed(half(byte >> 1)));
    if odd_first { [odd, even] } else { [even, odd] }
}

/// Index residual plus reference: both left aligned at their own widths, summed, and brought back
/// to `bits` bits with the sign.
fn add_index(residual: i16, bits: u8, reference: i16, reference_bits: u8) -> i16 {
    let up = |v: i16, b: u8| if b == 0 || b > 16 { 0 } else { (v as u16) << (16 - b) };
    let sum = up(residual, bits).wrapping_add(up(reference, reference_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

#[cfg(test)]
mod tests;
