//! BC1, BC3, BC4 and BC5: endpoints and indices are small unsigned values, each the sum of its
//! residual and the referenced block's value modulo the value's width.
//!
//! Index residuals are coded through rank tables: a 2-bit set takes one symbol per 2x2 quad, a
//! 3-bit set one symbol per two pixels in raster order. The indices count along the endpoint line
//! and become BC1 and BC4 codes only when the block is written.

use crate::lanes::Layout;
use crate::tables::{RANK_BC1, RANK_BC3_ALPHA, RANK_BC3_COLOUR, RANK_BC4, RANK_BC5, SEED_BC1, SEED_BC3, SEED_BC4, SEED_BC5};
use crate::{Blocks, Error, HISTORY, SEEDS, reach, references};

/// Endpoint fields and indices of one block.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Values {
    pub ep: [u8; 8],
    pub ix: [u8; 32],
}

/// One plain format.
pub struct Spec {
    pub layout: Layout,
    /// Width of each endpoint field and index.
    pub ep_bits: [u8; 8],
    pub ix_bits: [u8; 32],
    /// The references are two 4-bit symbols each.
    paired: bool,
    /// The record's residuals.
    residual: fn(&[u8]) -> Values,
    /// The block's bytes from its values.
    pack: fn(&Values, &mut [u8]),
    seeds: fn() -> &'static [Values; SEEDS],
}

/// Index widths: `three` 3-bit indices, then `two` 2-bit indices.
const fn index_widths(three: usize, two: usize) -> [u8; 32] {
    let mut ix = [0; 32];
    let mut k = 0;
    while k < three + two {
        ix[k] = if k < three { 3 } else { 2 };
        k += 1;
    }
    ix
}

const BC1_EP: [u8; 8] = [5, 6, 5, 5, 6, 5, 0, 0];
const BC3_EP: [u8; 8] = [5, 6, 5, 5, 6, 5, 8, 8];
const BC4_EP: [u8; 8] = [8, 8, 0, 0, 0, 0, 0, 0];
const BC5_EP: [u8; 8] = [8, 8, 8, 8, 0, 0, 0, 0];

/// BC1: references on lanes 0 and 1, endpoints R0 G0 B0 R1 G1 B1 on lanes 2 to 7, the four index
/// quads on lanes 8 to 11.
pub static BC1: Spec = Spec {
    layout: Layout { groups: &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], per: 1 },
    ep_bits: BC1_EP,
    ix_bits: index_widths(0, 16),
    paired: false,
    residual: bc1_residual,
    pack: bc1_pack,
    seeds: || seeds(&SEED_BC1, &SEEDS_BC1),
};

/// BC3, two symbols per lane: alpha endpoints on lane 2, alpha index pairs on lanes 3 to 6, colour
/// endpoints per channel on lanes 7 to 9, colour quads on lanes 10 and 11.
pub static BC3: Spec = Spec {
    layout: Layout { groups: &[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], per: 2 },
    ep_bits: BC3_EP,
    ix_bits: index_widths(16, 16),
    paired: true,
    residual: bc3_residual,
    pack: bc3_pack,
    seeds: || seeds(&SEED_BC3, &SEEDS_BC3),
};

/// BC4: endpoints on lanes 2 and 3, index pairs on lanes 4 to 11.
pub static BC4: Spec = Spec {
    layout: Layout { groups: &[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], per: 1 },
    ep_bits: BC4_EP,
    ix_bits: index_widths(16, 0),
    paired: false,
    residual: bc4_residual,
    pack: bc4_pack,
    seeds: || seeds(&SEED_BC4, &SEEDS_BC4),
};

/// BC5, two symbols per lane: red and green endpoints on lanes 2 and 3, index pairs (red, then
/// green) on lanes 4 to 11.
pub static BC5: Spec = Spec {
    layout: Layout { groups: &[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], per: 2 },
    ep_bits: BC5_EP,
    ix_bits: index_widths(32, 0),
    paired: true,
    residual: bc5_residual,
    pack: bc5_pack,
    seeds: || seeds(&SEED_BC5, &SEEDS_BC5),
};

type SeedCell = std::sync::OnceLock<Box<[Values; SEEDS]>>;
static SEEDS_BC1: SeedCell = SeedCell::new();
static SEEDS_BC3: SeedCell = SeedCell::new();
static SEEDS_BC4: SeedCell = SeedCell::new();
static SEEDS_BC5: SeedCell = SeedCell::new();

fn seeds<const E: usize, const I: usize>(table: &[([u8; E], [u8; I]); SEEDS], cell: &'static SeedCell) -> &'static [Values; SEEDS] {
    cell.get_or_init(|| {
        Box::new(std::array::from_fn(|k| {
            let mut v = Values::default();
            v.ep[..E].copy_from_slice(&table[k].0);
            v.ix[..I].copy_from_slice(&table[k].1);
            v
        }))
    })
}

/// Zigzag residual of a field `bits` wide (0, -1, 1, -2 ... modulo 2^bits).
fn field(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Pixel `k` of quad `q`: quads TL, TR, BL, BR, pixels in raster order inside the quad.
pub const QUAD_PIXELS: [[usize; 4]; 4] = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];

/// 2-bit indices from four quad symbols.
fn quads(rank: &[u8; 256], symbols: &[u8], ix: &mut [u8]) {
    for (pixels, &symbol) in QUAD_PIXELS.iter().zip(symbols) {
        let v = rank[usize::from(symbol)];
        for (k, &p) in pixels.iter().enumerate() {
            ix[p] = (v >> (2 * k)) & 3;
        }
    }
}

/// 3-bit indices from symbols of two pixels each (low nibble first).
fn pairs(rank: &[u8; 64], symbols: &[u8], ix: &mut [u8]) {
    for (k, &symbol) in symbols.iter().enumerate() {
        let v = rank[usize::from(symbol & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = (v >> 4) & 7;
    }
}

fn bc1_residual(s: &[u8]) -> Values {
    let mut v = Values::default();
    for k in 0..6 {
        v.ep[k] = field(s[2 + k], u32::from(BC1_EP[k]));
    }
    quads(&RANK_BC1, &s[8..12], &mut v.ix);
    v
}

fn bc3_residual(s: &[u8]) -> Values {
    // The colour lanes carry both endpoints of one channel.
    let mut v = Values { ep: [field(s[14], 5), field(s[16], 6), field(s[18], 5), field(s[15], 5), field(s[17], 6), field(s[19], 5), field(s[4], 8), field(s[5], 8)], ..Values::default() };
    pairs(&RANK_BC3_ALPHA, &s[6..14], &mut v.ix[..16]);
    quads(&RANK_BC3_COLOUR, &s[20..24], &mut v.ix[16..]);
    v
}

fn bc4_residual(s: &[u8]) -> Values {
    let mut v = Values::default();
    v.ep[0] = field(s[2], 8);
    v.ep[1] = field(s[3], 8);
    pairs(&RANK_BC4, &s[4..12], &mut v.ix[..16]);
    v
}

fn bc5_residual(s: &[u8]) -> Values {
    let mut v = Values::default();
    for k in 0..4 {
        v.ep[k] = field(s[4 + k], 8);
    }
    pairs(&RANK_BC5, &s[8..24], &mut v.ix);
    v
}

/// Position along the endpoint line to BC1 code, and to BC4 code.
pub const BC1_CODE: [u8; 4] = [0, 2, 3, 1];
pub const BC4_CODE: [u8; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

#[inline(always)]
fn colour(ep: &[u8], ix: &[u8], out: &mut [u8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let ix = &ix[..16];
    let mut codes = 0u32;
    for (p, &idx) in ix.iter().enumerate() {
        codes |= u32::from(BC1_CODE[usize::from(idx & 3)]) << (2 * p);
    }
    out[..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&codes.to_le_bytes());
}

/// A BC4 half. The endpoints are kept 128 below their stored value.
#[inline(always)]
fn single(e0: u8, e1: u8, ix: &[u8], out: &mut [u8]) {
    let ix = &ix[..16];
    let mut codes = 0u64;
    for (p, &idx) in ix.iter().enumerate() {
        codes |= u64::from(BC4_CODE[usize::from(idx & 7)]) << (16 + 3 * p);
    }
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | codes;
    out[..8].copy_from_slice(&v.to_le_bytes());
}

fn bc1_pack(v: &Values, out: &mut [u8]) {
    colour(&v.ep, &v.ix, out);
}

fn bc3_pack(v: &Values, out: &mut [u8]) {
    single(v.ep[6], v.ep[7], &v.ix[..16], &mut out[..8]);
    colour(&v.ep, &v.ix[16..], &mut out[8..]);
}

fn bc4_pack(v: &Values, out: &mut [u8]) {
    single(v.ep[0], v.ep[1], &v.ix, out);
}

fn bc5_pack(v: &Values, out: &mut [u8]) {
    single(v.ep[0], v.ep[1], &v.ix[..16], &mut out[..8]);
    single(v.ep[2], v.ep[3], &v.ix[16..], &mut out[8..]);
}

/// A plain format's decoder state: the last 256 blocks' values.
pub struct Stage {
    spec: &'static Spec,
    seeds: &'static [Values; SEEDS],
    history: Box<[Values; HISTORY]>,
}

impl Stage {
    pub fn new(spec: &'static Spec) -> Stage {
        Stage { spec, seeds: (spec.seeds)(), history: Box::new([Values::default(); HISTORY]) }
    }
}

const ZERO: Values = Values { ep: [0; 8], ix: [0; 32] };

impl Blocks for Stage {
    fn block(&mut self, i: usize, record: &[u8], out: &mut [u8]) -> Result<bool, Error> {
        let spec = self.spec;
        let (ep_ref, ix_ref) = references(record, spec.paired);
        let ep_base = &reach(i, ep_ref, &self.history[..], &self.seeds[..], &ZERO).ep;
        let ix_base = &reach(i, ix_ref, &self.history[..], &self.seeds[..], &ZERO).ix;
        let mut v = (spec.residual)(record);
        for ((value, &base), &bits) in v.ep.iter_mut().zip(ep_base).zip(&spec.ep_bits) {
            let mask = ((1u16 << bits) - 1) as u8;
            *value = value.wrapping_add(base) & mask;
        }
        for ((value, &base), &bits) in v.ix.iter_mut().zip(ix_base).zip(&spec.ix_bits) {
            *value = value.wrapping_add(base) & ((1u8 << bits) - 1);
        }
        self.history[i % HISTORY] = v;
        (spec.pack)(&v, out);
        Ok(true)
    }
}

/// For the encoder: the residual symbols of `values` against nothing, as a record.
#[cfg(any(test, feature = "encode"))]
pub fn record_of(spec: &Spec, values: &Values) -> Vec<u8> {
    fn zig(v: u8, bits: u8) -> u8 {
        // The residual in -2^(bits-1) .. 2^(bits-1) - 1 with the same value modulo 2^bits.
        let half = 1i16 << (bits - 1);
        let mut s = i16::from(v);
        if s >= half {
            s -= 1 << bits;
        }
        if s >= 0 { (2 * s) as u8 } else { (-2 * s - 1) as u8 }
    }
    fn inverse<const N: usize>(rank: &[u8; N]) -> std::collections::HashMap<u8, u8> {
        rank.iter().enumerate().map(|(s, &v)| (v, s as u8)).collect()
    }
    let quad_symbols = |rank: &[u8; 256], ix: &[u8]| -> [u8; 4] {
        let back = inverse(rank);
        std::array::from_fn(|q| back[&QUAD_PIXELS[q].iter().enumerate().fold(0u8, |acc, (k, &p)| acc | ix[p] << (2 * k))])
    };
    let pair_symbols = |rank: &[u8; 64], ix: &[u8]| -> Vec<u8> {
        let back = inverse(rank);
        ix.chunks(2).map(|p| back[&(p[0] | p[1] << 4)]).collect()
    };
    let mut r = vec![0u8; spec.layout.stride()];
    let ep = |k: usize| zig(values.ep[k], spec.ep_bits[k]);
    if std::ptr::eq(spec, &BC1) {
        for k in 0..6 {
            r[2 + k] = ep(k);
        }
        r[8..12].copy_from_slice(&quad_symbols(&RANK_BC1, &values.ix[..16]));
    } else if std::ptr::eq(spec, &BC3) {
        [r[14], r[16], r[18], r[15], r[17], r[19], r[4], r[5]] = std::array::from_fn(ep);
        r[6..14].copy_from_slice(&pair_symbols(&RANK_BC3_ALPHA, &values.ix[..16]));
        r[20..24].copy_from_slice(&quad_symbols(&RANK_BC3_COLOUR, &values.ix[16..]));
    } else if std::ptr::eq(spec, &BC4) {
        r[2] = ep(0);
        r[3] = ep(1);
        r[4..12].copy_from_slice(&pair_symbols(&RANK_BC4, &values.ix[..16]));
    } else {
        for k in 0..4 {
            r[4 + k] = ep(k);
        }
        r[8..24].copy_from_slice(&pair_symbols(&RANK_BC5, &values.ix));
    }
    r
}

/// For the tests: a block's bytes from its values (the decoder's own packing).
#[cfg(test)]
pub fn pack(spec: &Spec, values: &Values, out: &mut [u8]) {
    (spec.pack)(values, out);
}
