// The anchor tables below come from texture2ddecoder 0.1.2.
//
// Copyright (c) 2022 Rudolf Kolbe
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! BC7: 16 lanes, two symbols each per block.
//!
//! - lanes 0 and 1: the references (two 4-bit symbols each);
//! - lanes 2 to 10: zigzag endpoint residuals, per subset and channel both endpoints of the channel
//!   on one lane, left aligned at the mode's precision (the p-bit is red's lowest bit);
//! - lanes 11 to 14: the first index set, one row per lane, two pixels per symbol; lanes 6, 7, 8
//!   and 10 carry the second set of modes 4 and 5 (which have one subset, so those lanes are free);
//! - lane 15: the control code, a word of [`BC7_CONTROL`] that names the mode, partition, rotation
//!   and index selection.
//!
//! Endpoints and indices are left aligned 16-bit values in the history, so a block can reference
//! one of another mode: the reference is cut to the current precision before the residual is added.

#![allow(clippy::needless_range_loop)] // set, row, subset and channel numbers index parallel tables

use crate::lanes::Layout;
use crate::tables::{BC7_CONTROL, SEED_BC7};
use crate::{Bits, Blocks, Error, HISTORY, SEEDS, add_index, index_pair, reach, references};

pub const LAYOUT: Layout = Layout { groups: &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], per: 2 };

/// Anchor pixel of subset 1 in the two-subset partitions (also BC6H's).
pub const ANCHOR_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, //
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];
/// Anchor pixels of subsets 1 and 2 in the three-subset partitions.
pub const ANCHOR_3: [[u8; 64]; 2] = [
    [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, //
        8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
    ],
    [
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, //
        15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
    ],
];

/// Endpoint bits of R G B A per mode, before p-bits and rotation (public BC7).
const CHANNEL_BITS: [[u8; 4]; 8] = [[4, 4, 4, 0], [6, 6, 6, 0], [5, 5, 5, 0], [7, 7, 7, 0], [5, 5, 5, 6], [7, 7, 7, 8], [7, 7, 7, 7], [5, 5, 5, 5]];
/// Index bits of the first and second set per mode (public BC7).
const INDEX_BITS: [[u8; 2]; 8] = [[3, 0], [3, 0], [2, 0], [2, 0], [3, 2], [2, 2], [4, 0], [2, 0]];
/// Record position of endpoint 0's residual per subset and channel (endpoint 1's is the next
/// one). Three-subset modes have no alpha: subset 2 takes the alpha lanes of subsets 0 and 1.
const ENDPOINT_AT: [[usize; 4]; 3] = [[4, 6, 8, 10], [12, 14, 16, 18], [20, 10, 18, 0]];
/// Record position of each row of the two index sets.
const INDEX_AT: [[usize; 4]; 2] = [[22, 24, 26, 28], [12, 14, 16, 20]];

fn has_pbits(mode: usize) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// A decoded control word.
#[derive(Clone, Copy)]
struct Control {
    mode: usize,
    rotation: usize,
    selection: u64,
    subsets: usize,
    partition: usize,
}

impl Control {
    fn of(word: u16) -> Control {
        Control {
            mode: usize::from(word & 7),
            rotation: usize::from((word >> 3) & 3),
            selection: u64::from((word >> 5) & 1),
            subsets: usize::from((word >> 6) & 3),
            partition: usize::from((word >> 8) & 63),
        }
    }

    /// Bits of R G B A as stored: the p-bit counts as red's lowest bit, and a rotation swaps alpha
    /// with the rotated channel.
    fn precision(&self) -> [u8; 4] {
        let mut p = CHANNEL_BITS[self.mode];
        if self.rotation != 0 {
            p.swap(3, self.rotation - 1);
        } else if has_pbits(self.mode) {
            p[0] += 1;
        }
        p
    }
}

/// One block's values, as the history keeps them.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Values {
    pub control: u16,
    /// Per subset, channel R G B A and endpoint: left aligned to 16 bits.
    pub ends: [[[u16; 2]; 4]; 3],
    pub index_bits: [u8; 2],
    pub index: [[i16; 16]; 2],
}

static SEEDS_BUILT: std::sync::OnceLock<Box<[Values; SEEDS]>> = std::sync::OnceLock::new();

fn seeds() -> &'static [Values; SEEDS] {
    SEEDS_BUILT.get_or_init(|| {
        Box::new(std::array::from_fn(|k| {
            let (control, rows, index_bits, index) = &SEED_BC7[k];
            Values {
                control: *control,
                ends: std::array::from_fn(|s| std::array::from_fn(|c| [rows[8 * s + 2 * c], rows[8 * s + 2 * c + 1]])),
                index_bits: *index_bits,
                index: [index[..16].try_into().unwrap_or_default(), index[16..].try_into().unwrap_or_default()],
            }
        }))
    })
}

pub struct Stage {
    seeds: &'static [Values; SEEDS],
    history: Box<[Values; HISTORY]>,
}

impl Stage {
    pub fn new() -> Stage {
        Stage { seeds: seeds(), history: Box::new([Values::default(); HISTORY]) }
    }
}

const ZERO: Values = Values { control: 0, ends: [[[0; 2]; 4]; 3], index_bits: [0; 2], index: [[0; 16]; 2] };

impl Blocks for Stage {
    fn block(&mut self, i: usize, s: &[u8], out: &mut [u8]) -> Result<bool, Error> {
        let (ep_ref, ix_ref) = references(s, true);
        let word = BC7_CONTROL[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if word == 0 {
            return Err(Error::Corrupt("unused BC7 control code"));
        }
        let control = Control::of(word);
        let precision = control.precision();
        let base = reach(i, ep_ref, &self.history[..], &self.seeds[..], &ZERO);
        let mut v = Values { control: word, index_bits: INDEX_BITS[control.mode], ..ZERO };
        // Subsets the reference lacks build on its subset 0.
        let common = control.subsets.min(usize::from((base.control >> 6) & 3));
        for (c, &p) in precision.iter().enumerate() {
            if p == 0 {
                continue;
            }
            let mask = 0xFFFFu16 << (16 - p);
            for subset in 0..control.subsets {
                let from = &base.ends[if subset < common { subset } else { 0 }][c];
                for e in 0..2 {
                    let symbol = s[ENDPOINT_AT[subset][c] + e];
                    let residual = (u16::from(symbol >> 1) ^ u16::from(symbol & 1).wrapping_neg()) << (16 - p);
                    v.ends[subset][c][e] = (from[e] & mask).wrapping_add(residual);
                }
            }
        }
        let base = reach(i, ix_ref, &self.history[..], &self.seeds[..], &ZERO);
        for set in 0..2 {
            let bits = v.index_bits[set];
            if bits == 0 {
                continue;
            }
            for (row, &at) in INDEX_AT[set].iter().enumerate() {
                for b in 0..2 {
                    for (h, residual) in index_pair(s[at + b], row < 3).into_iter().enumerate() {
                        let p = 4 * row + 2 * b + h;
                        v.index[set][p] = add_index(residual, bits, base.index[set][p], base.index_bits[set]);
                    }
                }
            }
        }
        self.history[i % HISTORY] = v;
        Ok(pack(&v, out))
    }
}

/// Writes the block. Returns false when a value did not fit: an anchor index with its top bit set,
/// or a shared p-bit whose two halves differ.
pub fn pack(v: &Values, out: &mut [u8]) -> bool {
    let control = Control::of(v.control);
    let precision = control.precision();
    let mode = control.mode;
    let subsets = control.subsets;
    let mut fits = true;
    let mut w = Bits::default();
    w.put(1 << mode, mode as u32 + 1);
    match mode {
        0 => w.put(control.partition as u64, 4),
        1 | 2 | 3 | 7 => w.put(control.partition as u64, 6),
        4 => {
            w.put(control.rotation as u64, 2);
            w.put(control.selection, 1);
        }
        5 => w.put(control.rotation as u64, 2),
        _ => {}
    }
    let field = |subset: usize, c: usize, e: usize| u64::from(v.ends[subset][c][e] >> (16 - precision[c]));
    let mut order = [0, 1, 2, 3];
    if control.rotation != 0 {
        order.swap(3, control.rotation - 1);
    }
    let pbits = has_pbits(mode);
    for c in order {
        let p = u32::from(precision[c]);
        if p == 0 {
            continue;
        }
        let drop = u32::from(c == 0 && pbits);
        for subset in 0..subsets {
            for e in 0..2 {
                w.put(field(subset, c, e) >> drop, p - drop);
            }
        }
    }
    if mode == 1 {
        for subset in 0..subsets {
            let (a, b) = (field(subset, 0, 0) & 1, field(subset, 0, 1) & 1);
            fits &= a == b;
            w.put(a, 1);
        }
    } else if pbits {
        for subset in 0..subsets {
            for e in 0..2 {
                w.put(field(subset, 0, e) & 1, 1);
            }
        }
    }
    let anchor_mask: u32 = match subsets {
        2 => 1 | (1u32 << ANCHOR_2[control.partition]),
        3 => 1 | (1u32 << ANCHOR_3[0][control.partition]) | (1u32 << ANCHOR_3[1][control.partition]),
        _ => 1,
    };
    let mut indices = |set: usize, w: &mut Bits| {
        let bits = u32::from(v.index_bits[set]);
        for (p, &value) in v.index[set].iter().enumerate() {
            let anchor = (anchor_mask >> p) & 1 != 0;
            let value = u64::from(value as u16) & ((1 << bits) - 1);
            if anchor {
                fits &= value >> (bits - 1) == 0;
            }
            w.put(value, bits - u32::from(anchor));
        }
    };
    match mode {
        4 => {
            indices(1, &mut w);
            indices(0, &mut w);
        }
        5 => {
            indices(0, &mut w);
            indices(1, &mut w);
        }
        _ => indices(0, &mut w),
    }
    out.copy_from_slice(&w.value.to_le_bytes());
    fits
}

/// For the encoder: the stored precision of R G B A and the index bits of a control word.
#[cfg(any(test, feature = "encode"))]
pub fn precision_of(word: u16) -> ([u8; 4], [u8; 2]) {
    let control = Control::of(word);
    (control.precision(), INDEX_BITS[control.mode])
}

/// For the encoder: the control code (lane 15's two symbols) of a control word.
#[cfg(any(test, feature = "encode"))]
pub fn control_code(word: u16) -> Option<(u8, u8)> {
    let k = BC7_CONTROL.iter().position(|&w| w == word)?;
    Some(((k & 31) as u8, (k >> 5) as u8))
}

/// For the encoder: the record of `v` against nothing (both references 0).
#[cfg(any(test, feature = "encode"))]
pub fn record_of(v: &Values) -> Option<Vec<u8>> {
    let control = Control::of(v.control);
    let precision = control.precision();
    let mut r = vec![0u8; LAYOUT.stride()];
    (r[30], r[31]) = control_code(v.control)?;
    for (c, &p) in precision.iter().enumerate() {
        if p == 0 {
            continue;
        }
        for subset in 0..control.subsets {
            for e in 0..2 {
                // The field as a signed residual of p bits, zigzag coded.
                let field = i32::from(v.ends[subset][c][e] >> (16 - p));
                let signed = if field >= 1 << (p - 1) { field - (1 << p) } else { field };
                r[ENDPOINT_AT[subset][c] + e] = if signed >= 0 { (2 * signed) as u8 } else { (-2 * signed - 1) as u8 };
            }
        }
    }
    for set in 0..2 {
        let bits = v.index_bits[set];
        if bits == 0 {
            continue;
        }
        for (row, &at) in INDEX_AT[set].iter().enumerate() {
            for b in 0..2 {
                let p = 4 * row + 2 * b;
                let zig = |x: i16| {
                    // Back to `bits` bits with the sign, then the 4-bit zigzag code.
                    let s = (x << (16 - bits)) >> (16 - bits);
                    (if s >= 0 { 2 * s } else { -2 * s - 1 }) as u8
                };
                let (first, second) = (zig(v.index[set][p]), zig(v.index[set][p + 1]));
                let (even, odd) = if row < 3 { (second, first) } else { (first, second) };
                r[at + b] = crate::interleave(even, odd);
            }
        }
    }
    Some(r)
}
