//! BC6H: 16 lanes, two symbols each per block.
//!
//! - lanes 0 and 1: the references (two 4-bit symbols each);
//! - lanes 2 to 4: endpoint 0's R G B residuals, each two symbols interleaved bit by bit to 16 bits;
//! - lanes 5 to 7: endpoint 1's R G B residuals, the same way;
//! - lanes 8 to 10: endpoints 2 and 3 (two region modes), one symbol each per channel;
//! - lanes 11 to 14: the indices, one row per lane, two pixels per symbol;
//! - lane 15: the mode minus 1, then the partition.
//!
//! Endpoints are kept left aligned at 16 bits with their top bit flipped, so a block can reference
//! one of another mode.

use std::sync::OnceLock;

use crate::bc7::ANCHOR_2;
use crate::lanes::Layout;
use crate::tables::SEED_BC6;
use crate::{Bits, Blocks, Error, HISTORY, SEEDS, add_index, index_pair, reach, references};

pub const LAYOUT: Layout = Layout { groups: &[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], per: 2 };

/// Per mode 1 to 14 (index 0 unused): the mode code and its width, the bits of endpoint 0 and of
/// the other endpoints' R G B, and the fields after the mode code, in the notation of the public
/// BC6H description (`r0[9:0]`: bits 9 down to 0 of endpoint 0's red, written from bit 0 up; `d`:
/// the 5-bit partition). Modes 13 and 14 write the high bits of endpoint 0 in ascending order,
/// which is what the console's unpacker writes.
const MODES: [(u64, u32, [u8; 4], &str); 15] = [
    (0, 0, [0; 4], ""),
    (0x00, 2, [10, 5, 5, 5], "g2[4] b2[4] b3[4] r0[9:0] g0[9:0] b0[9:0] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3] d"),
    (0x01, 2, [7, 6, 6, 6], "g2[5] g3[4] g3[5] r0[6:0] b3[0] b3[1] b2[4] g0[6:0] b2[5] b3[2] g2[4] b0[6:0] b3[3] b3[5] b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0] d"),
    (0x02, 5, [11, 5, 4, 4], "r0[9:0] g0[9:0] b0[9:0] r1[4:0] r0[10] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] b1[3:0] b0[10] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3] d"),
    (0x06, 5, [11, 4, 5, 4], "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] g3[4] g2[3:0] g1[4:0] g0[10] g3[3:0] b1[3:0] b0[10] b3[1] b2[3:0] r2[3:0] b3[0] b3[2] r3[3:0] g2[4] b3[3] d"),
    (0x0A, 5, [11, 4, 4, 5], "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] b2[4] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] b1[4:0] b0[10] b2[3:0] r2[3:0] b3[1] b3[2] r3[3:0] b3[4] b3[3] d"),
    (0x0E, 5, [9, 5, 5, 5], "r0[8:0] b2[4] g0[8:0] g2[4] b0[8:0] b3[4] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3] d"),
    (0x12, 5, [8, 6, 5, 5], "r0[7:0] g3[4] b2[4] g0[7:0] b3[2] g2[4] b0[7:0] b3[3] b3[4] r1[5:0] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[5:0] r3[5:0] d"),
    (0x16, 5, [8, 5, 6, 5], "r0[7:0] b3[0] b2[4] g0[7:0] g2[5] g2[4] b0[7:0] g3[5] b3[4] r1[4:0] g3[4] g2[3:0] g1[5:0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3] d"),
    (0x1A, 5, [8, 5, 5, 6], "r0[7:0] b3[1] b2[4] g0[7:0] b2[5] g2[4] b0[7:0] b3[5] b3[4] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[5:0] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3] d"),
    (0x1E, 5, [6, 6, 6, 6], "r0[5:0] g3[4] b3[0] b3[1] b2[4] g0[5:0] g2[5] b2[5] b3[2] g2[4] b0[5:0] g3[5] b3[3] b3[5] b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0] d"),
    (0x03, 5, [10, 10, 10, 10], "r0[9:0] g0[9:0] b0[9:0] r1[9:0] g1[9:0] b1[9:0]"),
    (0x07, 5, [11, 9, 9, 9], "r0[9:0] g0[9:0] b0[9:0] r1[8:0] r0[10] g1[8:0] g0[10] b1[8:0] b0[10]"),
    (0x0B, 5, [12, 8, 8, 8], "r0[9:0] g0[9:0] b0[9:0] r1[7:0] r0[11:10] g1[7:0] g0[11:10] b1[7:0] b0[11:10]"),
    (0x0F, 5, [16, 4, 4, 4], "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[15:10] g1[3:0] g0[15:10] b1[3:0] b0[15:10]"),
];

/// A run of the layout: `len` bits of channel `channel` (3: the partition) of endpoint `end`, from
/// bit `first` up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub channel: u8,
    pub end: u8,
    pub first: u8,
    pub len: u8,
}

/// Parses one mode's field list.
fn parse(text: &str) -> Vec<Run> {
    text.split_whitespace()
        .map(|item| {
            if item == "d" {
                return Run { channel: 3, end: 0, first: 0, len: 5 };
            }
            let bytes = item.as_bytes();
            let channel = match bytes[0] {
                b'r' => 0,
                b'g' => 1,
                _ => 2,
            };
            let end = bytes[1] - b'0';
            let inner = &item[3..item.len() - 1];
            let (high, low) = inner.split_once(':').unwrap_or((inner, inner));
            let (high, low): (u8, u8) = (high.parse().unwrap_or(0), low.parse().unwrap_or(0));
            Run { channel, end, first: low, len: high - low + 1 }
        })
        .collect()
}

/// The parsed layouts of modes 1 to 14.
pub fn layouts() -> &'static [Vec<Run>; 15] {
    static PARSED: OnceLock<[Vec<Run>; 15]> = OnceLock::new();
    PARSED.get_or_init(|| std::array::from_fn(|m| parse(MODES[m].3)))
}

/// Index bits of a mode: 3 with two regions, 4 with one (0 for no mode).
pub fn index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

/// Bits of endpoint 0, and of the other endpoints' R G B, of a mode.
pub fn endpoint_bits(mode: u8) -> [u8; 4] {
    MODES[usize::from(mode)].2
}

/// One block's values, as the history keeps them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Values {
    /// Endpoints 0 to 3, R G B, left aligned, top bit flipped.
    pub ends: [[u16; 3]; 4],
    /// The mode (1 to 14) the indices were coded at.
    pub mode: u8,
    pub index: [i16; 16],
}

/// What "no reference" reads: zero, with indices of mode 11.
const ZERO: Values = Values { ends: [[0; 3]; 4], mode: 11, index: [0; 16] };

static SEEDS_BUILT: OnceLock<Box<[Values; SEEDS]>> = OnceLock::new();

fn seeds() -> &'static [Values; SEEDS] {
    SEEDS_BUILT.get_or_init(|| {
        Box::new(std::array::from_fn(|k| {
            let (ends, mode, index) = &SEED_BC6[k];
            Values { ends: std::array::from_fn(|e| [ends[3 * e], ends[3 * e + 1], ends[3 * e + 2]]), mode: *mode, index: *index }
        }))
    })
}

/// Spreads 8 bits onto the even bits (0, 2, ..., 14) of a 16-bit integer.
#[inline(always)]
pub(crate) fn spread8(v: u8) -> u16 {
    let mut x = u16::from(v);
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    (x | (x << 1)) & 0x5555
}

/// Two symbols interleaved bit by bit (the first on the even bits) to 16 bits.
#[inline(always)]
fn wide(even: u8, odd: u8) -> u16 {
    spread8(even) | (spread8(odd) << 1)
}

#[inline(always)]
fn unzigzag16(v: u16) -> u16 {
    (v >> 1) ^ (v & 1).wrapping_neg()
}

pub struct Stage {
    seeds: &'static [Values; SEEDS],
    history: Box<[Values; HISTORY]>,
}

impl Stage {
    pub fn new() -> Stage {
        Stage { seeds: seeds(), history: Box::new([ZERO; HISTORY]) }
    }
}

impl Blocks for Stage {
    fn block(&mut self, i: usize, s: &[u8], out: &mut [u8]) -> Result<bool, Error> {
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error::Corrupt("BC6H mode out of range"));
        }
        let two = mode <= 10;
        let shift = endpoint_bits(mode).map(|b| 16 - u32::from(b));
        let residual = |k: usize, c: usize| match k {
            0 => unzigzag16(wide(s[4 + 2 * c], s[5 + 2 * c])),
            1 => unzigzag16(wide(s[10 + 2 * c], s[11 + 2 * c])),
            _ => unzigzag16(u16::from(s[14 + k + 2 * c])),
        };
        let base = reach(i, ep_ref, &self.history[..], &self.seeds[..], &ZERO);
        let mut v = Values { ends: [[0; 3]; 4], mode, index: [0; 16] };
        let mask = |sh: u32| 0xFFFFu16 << sh;
        for c in 0..3 {
            v.ends[0][c] = base.ends[0][c].wrapping_add(residual(0, c) << shift[0]) & mask(shift[0]);
            for k in 1..if two { 4 } else { 2 } {
                v.ends[k][c] = base.ends[k][c].wrapping_add(residual(k, c) << shift[1 + c]) & mask(shift[1 + c]);
            }
            if !two {
                // One region: endpoints 2 and 3 repeat 0 (at the others' precision) and 1, for
                // later blocks that reference this one.
                v.ends[2][c] = (v.ends[0][c] >> shift[0]) << shift[1 + c];
                v.ends[3][c] = v.ends[1][c];
            }
        }
        let base = reach(i, ix_ref, &self.history[..], &self.seeds[..], &ZERO);
        let bits = index_bits(mode);
        for k in 0..8 {
            for (h, r) in index_pair(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                v.index[p] = add_index(r, bits, base.index[p], index_bits(base.mode));
            }
        }
        self.history[i % HISTORY] = v;
        Ok(pack(&v, s[31] & 31, out))
    }
}

/// Writes the block. Returns false when an anchor index has its top bit set.
pub fn pack(v: &Values, partition: u8, out: &mut [u8]) -> bool {
    let (code, code_bits, bits, _) = MODES[usize::from(v.mode)];
    let shift = bits.map(|b| 16 - u32::from(b));
    let field = |k: usize, c: usize| u64::from((v.ends[k][c] ^ 0x8000) >> if k == 0 { shift[0] } else { shift[1 + c] });
    let mut w = Bits::default();
    w.put(code, code_bits);
    for run in &layouts()[usize::from(v.mode)] {
        if run.channel == 3 {
            w.put(u64::from(partition), 5);
        } else {
            w.put(field(usize::from(run.end), usize::from(run.channel)) >> run.first, u32::from(run.len));
        }
    }
    let ibits = index_bits(v.mode);
    let anchor = if v.mode <= 10 { ANCHOR_2[usize::from(partition)] } else { 0 };
    let anchor_mask: u32 = 1 | (1u32 << anchor);
    let mut fits = true;
    for (p, &value) in v.index.iter().enumerate() {
        let short = (anchor_mask >> p) & 1 != 0;
        let biased = u64::from((value as u16).wrapping_add(1 << (ibits - 1))) & ((1 << ibits) - 1);
        if short {
            fits &= biased >> (ibits - 1) == 0;
        }
        w.put(biased, u32::from(ibits) - u32::from(short));
    }
    out.copy_from_slice(&w.value.to_le_bytes());
    fits
}

/// For the encoder: the record of `v` against nothing (both references 0), with `partition`.
#[cfg(any(test, feature = "encode"))]
pub fn record_of(v: &Values, partition: u8) -> Vec<u8> {
    let mut r = vec![0u8; LAYOUT.stride()];
    r[30] = v.mode - 1;
    r[31] = partition;
    let shift = endpoint_bits(v.mode).map(|b| 16 - u32::from(b));
    let zig16 = |x: u16| {
        let s = x as i16;
        (if s >= 0 { 2 * s as i32 } else { -2 * s as i32 - 1 }) as u16
    };
    // Even and odd bits of a 16-bit code as two symbols.
    let split = |x: u16| {
        let half = |y: u16| (0..8).fold(0u8, |acc, k| acc | (((y >> (2 * k)) & 1) as u8) << k);
        (half(x), half(x >> 1))
    };
    let two = v.mode <= 10;
    for c in 0..3 {
        // A field of b bits as a signed residual: the left aligned value read as signed.
        let signed = |value: u16, sh: u32| zig16(((value as i16) >> sh) as u16);
        (r[4 + 2 * c], r[5 + 2 * c]) = split(signed(v.ends[0][c], shift[0]));
        (r[10 + 2 * c], r[11 + 2 * c]) = split(signed(v.ends[1][c], shift[1 + c]));
        if two {
            r[16 + 2 * c] = signed(v.ends[2][c], shift[1 + c]) as u8;
            r[17 + 2 * c] = signed(v.ends[3][c], shift[1 + c]) as u8;
        }
    }
    let bits = index_bits(v.mode);
    for k in 0..8 {
        let zig = |x: i16| {
            let s = (x << (16 - bits)) >> (16 - bits);
            (if s >= 0 { 2 * s } else { -2 * s - 1 }) as u8
        };
        let (first, second) = (zig(v.index[2 * k]), zig(v.index[2 * k + 1]));
        let (even, odd) = if k < 6 { (second, first) } else { (first, second) };
        r[22 + k] = crate::interleave(even, odd);
    }
    r
}
