//! A small BCPack encoder for tests: rANS fragments per lane, interleaved by running the decoder's
//! own schedule, and block encoders that code every block against no reference. Its streams are
//! valid but not compact. Nothing here comes from a console or a game.

use std::collections::VecDeque;

use crate::lanes::{Feed, LOW, Lanes, Layout};
use crate::model::models;
use crate::{Format, bc6, bc7, plain};

/// One fragment of a lane: the model's shape (0 to 15), how far the header narrows the lane's
/// alphabet (0 to 3), and the symbols, each below `2^(group - narrow + 1)`.
#[derive(Clone, Debug)]
pub struct Fragment {
    pub shape: u8,
    pub narrow: u8,
    pub symbols: Vec<u8>,
}

/// The bytes of one lane: each fragment's header, length and payload. Every fragment clears the
/// state at its end. After the last one comes a filler fragment that parks the lane: its state
/// reaches exactly 2^24 with payload left, so it neither reads nor decodes again.
pub fn lane_bytes(group: u8, fragments: &[Fragment]) -> Vec<u8> {
    let mut out = Vec::new();
    for fragment in fragments {
        assert!(fragment.shape < 16 && fragment.narrow < 4 && fragment.narrow <= group);
        let model = &models()[usize::from(group - fragment.narrow) * 16 + usize::from(fragment.shape)];
        let mut x = LOW;
        let mut emitted = Vec::new();
        for &symbol in fragment.symbols.iter().rev() {
            assert!(u32::from(symbol) < 1 << (group - fragment.narrow + 1), "symbol {symbol} outside the alphabet");
            let (f, first) = model.interval(symbol);
            assert!(f > 0, "symbol {symbol} has no slot in this model");
            while u64::from(x) >= u64::from(f) << 16 {
                emitted.push(x as u8);
                x >>= 8;
            }
            x = ((x / f) << 16) + x % f + first;
        }
        emitted.extend_from_slice(&x.to_le_bytes());
        emitted.reverse();
        let n = emitted.len();
        let header = fragment.narrow << 6 | 0x10 | fragment.shape;
        if n > 256 {
            assert!(n - 257 < 1 << 16, "fragment too long");
            out.extend_from_slice(&[header | 0x20, (n - 257) as u8, ((n - 257) >> 8) as u8]);
        } else {
            out.extend_from_slice(&[header, (n - 1) as u8]);
        }
        out.extend(emitted);
    }
    // Filler: header 0, five payload bytes. From state 0 the first four make exactly 2^24.
    out.extend_from_slice(&[0, 4, 1, 0, 0, 0, 0]);
    out
}

/// Feeds each lane from its own queue and records the order: that order is the stream.
struct Queues {
    lanes: Vec<VecDeque<u8>>,
    stream: Vec<u8>,
}

impl Feed for Queues {
    fn next(&mut self, lane: usize) -> Option<u8> {
        let byte = self.lanes[lane].pop_front()?;
        self.stream.push(byte);
        Some(byte)
    }
}

/// Interleaves the lanes' bytes into one stream of `blocks` blocks, in the order the decoder takes
/// them.
pub fn interleave(layout: Layout, lanes: Vec<Vec<u8>>, blocks: usize) -> Vec<u8> {
    let mut queues = Queues { lanes: lanes.into_iter().map(VecDeque::from).collect(), stream: Vec::new() };
    let mut records = vec![0u8; blocks * layout.stride()];
    Lanes::new(layout).run(&mut queues, &mut records).expect("the lanes decode what they were given");
    queues.stream
}

/// A stream of `records` (one record of `layout.stride()` symbols per block). Each lane is cut into
/// fragments of at most `run` symbols; the fragments cycle through the shapes and, where the
/// symbols allow it, narrow the alphabet.
pub fn stream(layout: Layout, records: &[u8], run: usize) -> Vec<u8> {
    let stride = layout.stride();
    let blocks = records.len() / stride;
    let lanes = layout
        .groups
        .iter()
        .enumerate()
        .map(|(j, &group)| {
            let symbols: Vec<u8> = records.chunks_exact(stride).flat_map(|r| r[j * layout.per..(j + 1) * layout.per].to_vec()).collect();
            let fragments: Vec<Fragment> = symbols
                .chunks(run.max(1))
                .enumerate()
                .map(|(k, chunk)| {
                    let widest = chunk.iter().map(|&s| 8 - s.leading_zeros() as u8).max().unwrap_or(0);
                    // Narrow as far as the symbols allow, every other fragment.
                    let narrow = if k % 2 == 1 { (group + 1).saturating_sub(widest.max(1)).min(3) } else { 0 };
                    Fragment { shape: ((j + 5 * k) % 16) as u8, narrow, symbols: chunk.to_vec() }
                })
                .collect();
            lane_bytes(group, &fragments)
        })
        .collect();
    interleave(layout, lanes, blocks)
}

/// The values of a BC1 colour half (8 bytes) into `ep[..6]` and `ix[..16]` at `at`.
fn parse_colour(block: &[u8], ep: &mut [u8], ix: &mut [u8]) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let codes = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (k, (c, shift)) in [(c0, 11), (c0, 5), (c0, 0), (c1, 11), (c1, 5), (c1, 0)].into_iter().enumerate() {
        ep[k] = ((c >> shift) & if shift == 5 { 63 } else { 31 }) as u8;
    }
    for (p, value) in ix.iter_mut().enumerate().take(16) {
        let code = ((codes >> (2 * p)) & 3) as u8;
        *value = plain::BC1_CODE.iter().position(|&c| c == code).unwrap_or(0) as u8;
    }
}

/// The values of a BC4 half (8 bytes): endpoints into `ep[..2]`, indices into `ix[..16]`.
fn parse_single(block: &[u8], ep: &mut [u8], ix: &mut [u8]) {
    let v = u64::from_le_bytes(block[..8].try_into().unwrap_or_default());
    ep[0] = (v as u8) ^ 0x80;
    ep[1] = ((v >> 8) as u8) ^ 0x80;
    for (p, value) in ix.iter_mut().enumerate().take(16) {
        let code = ((v >> (16 + 3 * p)) & 7) as u8;
        *value = plain::BC4_CODE.iter().position(|&c| c == code).unwrap_or(0) as u8;
    }
}

/// Packs BC1, BC3, BC4 or BC5 blocks into a stream (every block against no reference).
pub fn pack_plain(format: Format, blocks: &[u8], run: usize) -> Vec<u8> {
    let (spec, bytes) = match format {
        Format::Bc1 => (&plain::BC1, 8),
        Format::Bc3 => (&plain::BC3, 16),
        Format::Bc4 => (&plain::BC4, 8),
        Format::Bc5 => (&plain::BC5, 16),
        _ => panic!("pack_plain takes BC1, BC3, BC4 and BC5"),
    };
    let records: Vec<u8> = blocks
        .chunks_exact(bytes)
        .flat_map(|b| {
            let mut v = plain::Values::default();
            match format {
                Format::Bc1 => parse_colour(b, &mut v.ep, &mut v.ix),
                Format::Bc3 => {
                    let (mut ep, mut ix) = ([0u8; 2], [0u8; 16]);
                    parse_single(&b[..8], &mut ep, &mut ix);
                    (v.ep[6], v.ep[7]) = (ep[0], ep[1]);
                    v.ix[..16].copy_from_slice(&ix);
                    parse_colour(&b[8..], &mut v.ep[..6], &mut v.ix[16..]);
                }
                Format::Bc4 => parse_single(b, &mut v.ep, &mut v.ix),
                _ => {
                    parse_single(&b[..8], &mut v.ep[..2], &mut v.ix[..16]);
                    parse_single(&b[8..], &mut v.ep[2..4], &mut v.ix[16..]);
                }
            }
            plain::record_of(spec, &v)
        })
        .collect();
    stream(spec.layout, &records, run)
}

/// BC7 values of one block, for [`pack_bc7`]: the control word, the endpoints of each subset and
/// channel as fields of the stored precision, and the indices.
pub use bc7::Values as Bc7Values;
/// BC6H values of one block, for [`pack_bc6`].
pub use bc6::Values as Bc6Values;

/// Packs BC7 blocks given by their values into a stream, and returns the blocks the decoder must
/// give. `None` when a control word is not one the format has.
pub fn pack_bc7(values: &[Bc7Values], run: usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut records = Vec::new();
    let mut blocks = vec![0u8; 16 * values.len()];
    for (v, out) in values.iter().zip(blocks.as_chunks_mut::<16>().0) {
        records.extend(bc7::record_of(v)?);
        bc7::pack(v, out);
    }
    Some((stream(bc7::LAYOUT, &records, run), blocks))
}

/// Packs BC6H blocks given by their values and partitions into a stream, and returns the blocks
/// the decoder must give.
pub fn pack_bc6(values: &[(Bc6Values, u8)], run: usize) -> (Vec<u8>, Vec<u8>) {
    let mut records = Vec::new();
    let mut blocks = vec![0u8; 16 * values.len()];
    for ((v, partition), out) in values.iter().zip(blocks.as_chunks_mut::<16>().0) {
        records.extend(bc6::record_of(v, *partition));
        bc6::pack(v, *partition, out);
    }
    (stream(bc6::LAYOUT, &records, run), blocks)
}

/// The BC7 control words the format has (the nonzero entries of its table).
pub fn bc7_controls() -> Vec<u16> {
    crate::tables::BC7_CONTROL.iter().copied().filter(|&w| w != 0).collect()
}

/// The bits of endpoint 0 and of the other endpoints' R G B of BC6H mode `mode`, and its index bits.
pub fn bc6_bits(mode: u8) -> ([u8; 4], u8) {
    (bc6::endpoint_bits(mode), bc6::index_bits(mode))
}

/// The BC7 precision of R G B A as stored (the p-bit as red's lowest bit, rotation applied), and
/// the index bits of the two sets, of a control word.
pub fn bc7_precision(word: u16) -> ([u8; 4], [u8; 2]) {
    bc7::precision_of(word)
}
