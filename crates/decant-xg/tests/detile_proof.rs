//! Independent reference and property tests for every swizzle mode of `decant-xg`.
//!
//! For each mode, hardware version, format, dimension and PRNG generated size, this suite checks four properties:
//! 1. Bounds: every element offset stays inside the reported resource size.
//! 2. Bijection: distinct elements across all mips, slices and samples map to disjoint byte ranges.
//! 3. Identity: `CopyIntoSubresource` followed by `CopyFromSubresource` recovers the linear input.
//! 4. Reference match: `GetTexelElementOffsetBytes` equals an independent reference built from public GFX10 rules.
#![allow(unsafe_code, clippy::too_many_arguments)]

mod common;

use common::*;
use xg_xs::abi::{XgTexture1dDesc, XgTexture2dDesc, XgTexture3dDesc};
use xg_xs::exports::{XGCreateTexture1DComputer, XGCreateTexture2DComputer, XGCreateTexture3DComputer, XGSetHardwareVersion};

/// Deterministic SplitMix64 pseudo random generator (no external crate).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: u32, hi_inclusive: u32) -> u32 {
        let span = u64::from(hi_inclusive - lo + 1);
        lo + (self.next_u64() % span) as u32
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while i < len {
            let v = self.next_u64().to_le_bytes();
            let take = (len - i).min(8);
            out.extend_from_slice(&v[..take]);
            i += take;
        }
        out
    }
}

/// Every valid swizzle mode of `decant-xg`.
const ALL_MODES: [u32; 17] = [0, 1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27, 28, 31];

/// (DXGI_FORMAT, bytes per element, texels per element side).
const FORMATS: [(u32, u32, u32); 7] = [
    (61, 1, 1),
    (56, 2, 1),
    (28, 4, 1),
    (10, 8, 1),
    (2, 16, 1),
    (71, 8, 4),
    (98, 16, 4),
];

fn align_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

fn ref_block_bits(mode: u32, v4: bool) -> u32 {
    match mode & 0x1F {
        0 => 0,
        1..=3 => 8,
        4..=7 | 20..=23 => 12,
        8..=19 | 24..=27 => 16,
        _ => if v4 { 17 } else { 19 },
    }
}

fn ref_mode_valid_bpe(mode: u32, bpe: u32) -> bool {
    match mode & 0x1F {
        24 | 28 => bpe <= 8,
        0 | 1 | 2 | 5 | 6 | 9 | 10 | 17 | 18 | 21 | 22 | 25 | 26 | 27 | 31 => true,
        _ => false,
    }
}

fn ref_is_thick(mode: u32, is_3d: bool) -> bool {
    is_3d && matches!(mode & 0x1F, 5 | 9 | 17 | 21 | 25 | 26)
}

fn ref_block_shape_2d(bits: u32, lb: u32, ls: u32) -> (u32, u32) {
    if bits.is_multiple_of(2) {
        let e = bits - lb;
        (1 << (e.div_ceil(2) - ls.div_ceil(2)), 1 << (e / 2 - ls / 2))
    } else {
        let e = bits - lb - ls;
        (1 << e.div_ceil(2), 1 << (e / 2))
    }
}

fn ref_block_shape_thick(bits: u32, lb: u32) -> (u32, u32, u32) {
    const BASE_1K: [(u32, u32, u32); 5] = [(16, 8, 8), (8, 8, 8), (8, 8, 4), (8, 4, 4), (4, 4, 4)];
    let (w, h, d) = BASE_1K[lb as usize];
    let a = (bits - 10) / 3;
    let r = (bits - 10) % 3;
    (w << a, h << (a + r / 2), d << (a + u32::from(r > 0)))
}

fn ref_eval_terms(terms: &str, x: u32, y: u32, z: u32, s: u32) -> u32 {
    let rest = terms.strip_prefix("from bit ").expect("equation prefix");
    let (first_str, list) = rest.split_once(':').expect("equation colon");
    let first: u32 = first_str.trim().parse().expect("first bit");
    let mut out = 0u32;
    for (i, token) in list.split_whitespace().enumerate() {
        let mut parity = 0u32;
        for part in token.split('^') {
            let (axis, idx_str) = part.split_at(1);
            let idx: u32 = idx_str.parse().expect("bit index");
            let val = match axis {
                "x" => x,
                "y" => y,
                "z" => z,
                "s" => s,
                _ => panic!("unknown axis {axis}"),
            };
            parity ^= (val >> idx) & 1;
        }
        out |= parity << (first + i as u32);
    }
    out
}

fn ref_eval_masks(masks: &[(u16, u16, u16)], x: u32, y: u32, z: u32) -> u32 {
    let mut out = 0u32;
    for (bit, &(mx, my, mz)) in masks.iter().enumerate() {
        let p = ((x & u32::from(mx)).count_ones() + (y & u32::from(my)).count_ones() + (z & u32::from(mz)).count_ones()) & 1;
        out |= p << bit;
    }
    out
}

/// Straightforward 2D/thin bit equations for the non-XOR modes (1, 2, 5, 6, 9, 10) from GFX10 micro/macro rules.
fn ref_in_block_plain(mode: u32, bits: u32, lb: u32, x: u32, y: u32) -> u32 {
    let is_display = matches!(mode, 2 | 6 | 10);
    // Micro tile (256 B): bits lb..8.
    let micro_bits = 8 - lb;
    let mw_log2 = micro_bits.div_ceil(2);
    let mh_log2 = micro_bits / 2;
    let mut off = 0u32;
    if !is_display {
        // Standard 256B micro tile: 16 bytes along x (4 - lb bits), then mh_log2 bits of y, then remaining x bits.
        let mut b = lb;
        let first_x = (4 - lb).min(mw_log2);
        for xi in 0..first_x {
            off |= ((x >> xi) & 1) << b;
            b += 1;
        }
        for yi in 0..mh_log2 {
            off |= ((y >> yi) & 1) << b;
            b += 1;
        }
        for xi in first_x..mw_log2 {
            off |= ((x >> xi) & 1) << b;
            b += 1;
        }
    } else {
        // Display 256B micro tile (RB+ for modes 2, 6, 10).
        let eq = match lb {
            0 => "from bit 0: x0 x1 x2 y1 y0 y2 x3 y3",
            1 => "from bit 1: x0 x1 x2 y0 y1 y2 x3",
            2 => "from bit 2: x0 x1 y0 y1 x2 y2",
            3 => "from bit 3: x0 y0 x1 x2 y1",
            4 => "from bit 4: x0 y0 x1 y1",
            _ => unreachable!(),
        };
        off = ref_eval_terms(eq, x, y, 0, 0);
    }
    if bits > 8 {
        // Macro tile bits 8..bits: alternate y then x from micro tile boundary upward.
        let mut xi = mw_log2;
        let mut yi = mh_log2;
        let mut y_turn = true;
        for b in 8..bits {
            if y_turn {
                off |= ((y >> yi) & 1) << b;
                yi += 1;
            } else {
                off |= ((x >> xi) & 1) << b;
                xi += 1;
            }
            y_turn = !y_turn;
        }
    }
    off
}

/// Independent reference for GFX10 2D/thin in-block offset (all modes, v0 and v4, 1 sample and MSAA).
fn ref_in_block_2d(mode: u32, bpe: u32, samples: u32, v4: bool, x: u32, y: u32, z: u32, s: u32) -> u32 {
    let mode = mode & 0x1F;
    let lb = bpe.ilog2();
    let bits = ref_block_bits(mode, v4);
    if samples == 1 && matches!(mode, 1 | 2 | 5 | 6 | 9 | 10) {
        return ref_in_block_plain(mode, bits, lb, x, y);
    }
    if samples == 1 && v4 && matches!(mode, 21 | 22 | 24..=27) {
        return ref_in_block_v4_32pipe(mode, lb, x, y, z);
    }
    if samples > 1 {
        return ref_in_block_msaa(mode, lb, samples, v4, x, y, z, s);
    }
    let eq = match (v4, mode, lb) {
        (false, 17, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 y4 x4 x7^y5 x5^y7 x6^y6 x6 y7 x7",
        (false, 17, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 y3 x4 x7^y4 x5^y6 x6^y5 x6 y6 x7",
        (false, 17, 2) => "from bit 2: x0 x1 y0 y1 y2 x2 y3 x3 x6^y4 x4^y6 x5^y5 x5 y6 x6",
        (false, 17, 3) => "from bit 3: x0 y0 y1 x1 x2 y2 x3 x6^y3 x4^y5 x5^y4 x5 y5 x6",
        (false, 17, 4) => "from bit 4: y0 y1 x0 x1 y2 x2 x5^y3 x3^y5 x4^y4 x4 y5 x5",
        (false, 18, 0) => "from bit 0: x0 x1 x2 y1 y0 y2 x3 y3 y4 x4 x7^y5 x5^y7 x6^y6 x6 y7 x7",
        (false, 18, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 y3 x4 x7^y4 x5^y6 x6^y5 x6 y6 x7",
        (false, 18, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 y3 x3 x6^y4 x4^y6 x5^y5 x5 y6 x6",
        (false, 18, 3) => "from bit 3: x0 y0 x1 x2 y1 y2 x3 x6^y3 x4^y5 x5^y4 x5 y5 x6",
        (false, 18, 4) => "from bit 4: x0 y0 x1 y1 y2 x2 x5^y3 x3^y5 x4^y4 x4 y5 x5",
        (false, 21, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4 x4^y7 x6^y5^z2 x5^y6^z1",
        (false, 21, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y3 x4^y6 x6^y4^z2 x5^y5^z1",
        (false, 21, 2) => "from bit 2: x0 x1 y0 y1 y2 x2 x6^y3 x3^y6 x5^y4^z2 x4^y5^z1",
        (false, 21, 3) => "from bit 3: x0 y0 y1 x1 x2 x6^y2 x3^y5 x5^y3^z2 x4^y4^z1",
        (false, 21, 4) => "from bit 4: y0 y1 x0 x1 x5^y2 x2^y5 x4^y3^z2 x3^y4^z1",
        (false, 22, 0) => "from bit 0: x0 x1 x2 y1 y0 y2 x3 y3 x7^y4 x4^y7 x6^y5^z2 x5^y6^z1",
        (false, 22, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y3 x4^y6 x6^y4^z2 x5^y5^z1",
        (false, 22, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y3 x3^y6 x5^y4^z2 x4^y5^z1",
        (false, 22, 3) => "from bit 3: x0 y0 x1 x2 y1 x6^y2 x3^y5 x5^y3^z2 x4^y4^z1",
        (false, 22, 4) => "from bit 4: x0 y0 x1 y1 x5^y2 x2^y5 x4^y3^z2 x3^y4^z1",
        (false, 24, 0) => "from bit 0: x0 y0 x1 y1 x2 y2 x3 y3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y6 x7 x8^y7",
        (false, 24, 1) => "from bit 1: x0 y0 x1 y1 x2 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y3 y6 x7^y7",
        (false, 24, 2) => "from bit 2: x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y6",
        (false, 24, 3) => "from bit 3: x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^y6 y2 x3 x7^y3",
        (false, 25, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x8^y4 x4^y8 x7^y5^z2 x5^y7^z1 x6^y6^z0 x6 y7 x7",
        (false, 25, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x8^y3 x4^y7 x7^y4^z2 x5^y6^z1 x6^y5^z0 x6 y6 x7",
        (false, 25, 2) => "from bit 2: x0 x1 y0 y1 y2 x2 x7^y3 x3^y7 x6^y4^z2 x4^y6^z1 x5^y5^z0 x5 y6 x6",
        (false, 25, 3) => "from bit 3: x0 y0 y1 x1 x2 x7^y2 x3^y6 x6^y3^z2 x4^y5^z1 x5^y4^z0 x5 y5 x6",
        (false, 25, 4) => "from bit 4: y0 y1 x0 x1 x6^y2 x2^y6 x5^y3^z2 x3^y5^z1 x4^y4^z0 x4 y5 x5",
        (false, 26, 0) => "from bit 0: x0 x1 x2 y1 y0 y2 x3 y3 x8^y4 x4^y8 x7^y5^z2 x5^y7^z1 x6^y6^z0 x6 y7 x7",
        (false, 26, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x8^y3 x4^y7 x7^y4^z2 x5^y6^z1 x6^y5^z0 x6 y6 x7",
        (false, 26, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y3 x3^y7 x6^y4^z2 x4^y6^z1 x5^y5^z0 x5 y6 x6",
        (false, 26, 3) => "from bit 3: x0 y0 x1 x2 y1 x7^y2 x3^y6 x6^y3^z2 x4^y5^z1 x5^y4^z0 x5 y5 x6",
        (false, 26, 4) => "from bit 4: x0 y0 x1 y1 x6^y2 x2^y6 x5^y3^z2 x3^y5^z1 x4^y4^z0 x4 y5 x5",
        (false, 27, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 x7 x8^y7",
        (false, 27, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 y6 x7^y7",
        (false, 27, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x3 y3 x7^y6",
        (false, 27, 3) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y2 x3 x7^y3",
        (false, 27, 4) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x3^x8^y8 x2 y2 x6^y3",
        (false, 28, 0) => "from bit 0: x0 y0 x1 y1 x2 y2 x3 y3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y6 x7 x8^y10 x11^y7 x9^y9 x10^y8",
        (false, 28, 1) => "from bit 1: x0 y0 x1 y1 x2 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y3 y6 x7^y10 x10^y7 x8^y9 x9^y8",
        (false, 28, 2) => "from bit 2: x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y9 x10^y6 x8^y8 x9^y7",
        (false, 28, 3) => "from bit 3: x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^y6 y2 x3 x10^y3 x9^y6 x7^y8 x8^y7",
        (false, 31, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 x7 x8^y10 x11^y7 x9^y9 x10^y8",
        (false, 31, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 y6 x7^y10 x10^y7 x8^y9 x9^y8",
        (false, 31, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x3 y3 x7^y9 x10^y6 x8^y8 x9^y7",
        (false, 31, 3) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y2 x3 x10^y3 x9^y6 x7^y8 x8^y7",
        (false, 31, 4) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x2 y2 x3^y8 x9^y3 x7^y7 x8^y6",
        (true, 17, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4 x4^y6 x5^y5 x5 y6 x6 y7 x7",
        (true, 17, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y3 x4^y5 x5^y4 x5 y5 x6 y6 x7",
        (true, 17, 2) => "from bit 2: x0 x1 y0 y1 y2 x2 x5^y3 x3^y5 x4^y4 x4 y5 x5 y6 x6",
        (true, 17, 3) => "from bit 3: x0 y0 y1 x1 x2 x5^y2 x3^y4 x4^y3 x4 y4 x5 y5 x6",
        (true, 17, 4) => "from bit 4: y0 y1 x0 x1 x4^y2 x2^y4 x3^y3 x3 y4 x4 y5 x5",
        (true, 18, 0) => "from bit 0: x0 x1 x2 y1 y0 y2 x3 y3 x6^y4 x4^y6 x5^y5 x5 y6 x6 y7 x7",
        (true, 18, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y3 x4^y5 x5^y4 x5 y5 x6 y6 x7",
        (true, 18, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x5^y3 x3^y5 x4^y4 x4 y5 x5 y6 x6",
        (true, 18, 3) => "from bit 3: x0 y0 x1 x2 y1 x5^y2 x3^y4 x4^y3 x4 y4 x5 y5 x6",
        (true, 18, 4) => "from bit 4: x0 y0 x1 y1 x4^y2 x2^y4 x3^y3 x3 y4 x4 y5 x5",
        (true, 28, 0) => "from bit 0: x0 y0 x1 y1 x2 y2 x3 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x10^y6 x7^y9 x9^y7 x8^y8 x9 x10",
        (true, 28, 1) => "from bit 1: x0 y0 x1 y1 x2 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6 x7^y8 x8^y7 x8 x9",
        (true, 28, 2) => "from bit 2: x0 y0 x1 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y7 x8 x9",
        (true, 28, 3) => "from bit 3: x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x8^y3 x5^y8 x6^y7 x7^y6 x7 x8",
        (true, 31, 0) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x10^y6 x7^y9 x9^y7 x8^y8 x9 x10",
        (true, 31, 1) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6 x7^y8 x8^y7 x8 x9",
        (true, 31, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y7 x8 x9",
        (true, 31, 3) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x8^y3 x5^y8 x6^y7 x7^y6 x7 x8",
        (true, 31, 4) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x2 y2 x3^y8 x7^y3 x5^y7 x6^y6 x7 x8",
        _ => panic!("unsupported 2D equation v4={v4} mode={mode} lb={lb}"),
    };
    ref_eval_terms(eq, x, y, z, 0) & ((1 << bits) - 1)
}

/// Hardware version 4 RB+ 32-pipe equations (GFX10.3 addrlib) for modes 21, 22, 24..=27.
fn ref_in_block_v4_32pipe(mode: u32, lb: u32, x: u32, y: u32, z: u32) -> u32 {
    let masks: &[(u16, u16, u16)] = match (mode, lb) {
        (21, 0) => &[(1,0,0),(2,0,0),(4,0,0),(8,0,0),(0,1,0),(0,2,0),(0,4,0),(0,8,0),(64,16,0),(16,64,2),(32,32,1),(32,0,0)],
        (21, 1) => &[(0,0,0),(1,0,0),(2,0,0),(4,0,0),(0,1,0),(0,2,0),(0,4,0),(8,0,0),(64,8,0),(16,32,2),(32,16,1),(32,0,0)],
        (21, 2) => &[(0,0,0),(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(0,4,0),(4,0,0),(32,8,0),(8,32,2),(16,16,1),(16,0,0)],
        (21, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(0,2,0),(2,0,0),(4,0,0),(32,4,0),(8,16,2),(16,8,1),(16,0,0)],
        (21, 4) => &[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,2,0),(1,0,0),(2,0,0),(16,4,0),(4,16,2),(8,8,1),(8,0,0)],
        (22, 0) => &[(1,0,0),(2,0,0),(4,0,0),(0,2,0),(0,1,0),(0,4,0),(8,0,0),(0,8,0),(64,16,0),(16,64,2),(32,32,1),(32,0,0)],
        (22, 1) => &[(0,0,0),(1,0,0),(2,0,0),(4,0,0),(0,1,0),(0,2,0),(0,4,0),(8,0,0),(64,8,0),(16,32,2),(32,16,1),(32,0,0)],
        (22, 2) => &[(0,0,0),(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(4,0,0),(0,4,0),(32,8,0),(8,32,2),(16,16,1),(16,0,0)],
        (22, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(4,0,0),(0,2,0),(32,4,0),(8,16,2),(16,8,1),(16,0,0)],
        (22, 4) => &[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(16,4,0),(4,16,2),(8,8,1),(8,0,0)],
        (24, 0) => &[(1,0,0),(0,1,0),(2,0,0),(0,2,0),(4,0,0),(0,4,0),(8,0,0),(0,8,0),(64,80,0),(16,16,2),(32,32,1),(32,0,0),(64,0,0),(512,64,0),(128,256,0),(256,128,0)],
        (24, 1) => &[(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(4,0,0),(0,4,0),(8,0,0),(64,80,0),(16,16,2),(32,32,1),(0,8,0),(32,0,0),(64,256,0),(256,64,0),(128,128,0)],
        (24, 2) => &[(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(4,0,0),(0,4,0),(64,80,0),(16,16,2),(32,32,1),(8,0,0),(0,8,0),(32,256,0),(64,128,0),(128,64,0)],
        (24, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(4,0,0),(64,80,0),(16,16,2),(32,32,1),(0,4,0),(8,0,0),(128,8,0),(32,128,0),(64,64,0)],
        (25, 0) => &[(1,0,0),(2,0,0),(4,0,0),(8,0,0),(0,1,0),(0,2,0),(0,4,0),(0,8,0),(64,16,0),(16,64,2),(32,32,1),(32,0,0),(0,64,0),(64,0,0),(0,128,0),(128,0,0)],
        (25, 1) => &[(0,0,0),(1,0,0),(2,0,0),(4,0,0),(0,1,0),(0,2,0),(0,4,0),(8,0,0),(64,8,0),(16,32,2),(32,16,1),(32,0,0),(0,32,0),(64,0,0),(0,64,0),(128,0,0)],
        (25, 2) => &[(0,0,0),(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(0,4,0),(4,0,0),(32,8,0),(8,32,2),(16,16,1),(16,0,0),(0,32,0),(32,0,0),(0,64,0),(64,0,0)],
        (25, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(0,2,0),(2,0,0),(4,0,0),(32,4,0),(8,16,2),(16,8,1),(16,0,0),(0,16,0),(32,0,0),(0,32,0),(64,0,0)],
        (25, 4) => &[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,2,0),(1,0,0),(2,0,0),(16,4,0),(4,16,2),(8,8,1),(8,0,0),(0,16,0),(16,0,0),(0,32,0),(32,0,0)],
        (26, 0) => &[(1,0,0),(2,0,0),(4,0,0),(0,2,0),(0,1,0),(0,4,0),(8,0,0),(0,8,0),(64,16,0),(16,64,2),(32,32,1),(32,0,0),(0,64,0),(64,0,0),(0,128,0),(128,0,0)],
        (26, 1) => &[(0,0,0),(1,0,0),(2,0,0),(4,0,0),(0,1,0),(0,2,0),(0,4,0),(8,0,0),(64,8,0),(16,32,2),(32,16,1),(32,0,0),(0,32,0),(64,0,0),(0,64,0),(128,0,0)],
        (26, 2) => &[(0,0,0),(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(4,0,0),(0,4,0),(32,8,0),(8,32,2),(16,16,1),(16,0,0),(0,32,0),(32,0,0),(0,64,0),(64,0,0)],
        (26, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(4,0,0),(0,2,0),(32,4,0),(8,16,2),(16,8,1),(16,0,0),(0,16,0),(32,0,0),(0,32,0),(64,0,0)],
        (26, 4) => &[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(16,4,0),(4,16,2),(8,8,1),(8,0,0),(0,16,0),(16,0,0),(0,32,0),(32,0,0)],
        (27, 0) => &[(1,0,0),(2,0,0),(4,0,0),(8,0,0),(0,1,0),(0,2,0),(0,4,0),(0,8,0),(64,80,0),(16,16,2),(32,32,1),(32,0,0),(64,0,0),(512,64,0),(128,256,0),(256,128,0)],
        (27, 1) => &[(0,0,0),(1,0,0),(2,0,0),(4,0,0),(0,1,0),(0,2,0),(0,4,0),(8,0,0),(64,80,0),(16,16,2),(32,32,1),(0,8,0),(32,0,0),(64,256,0),(256,64,0),(128,128,0)],
        (27, 2) => &[(0,0,0),(0,0,0),(1,0,0),(2,0,0),(0,1,0),(0,2,0),(4,0,0),(0,4,0),(64,80,0),(16,16,2),(32,32,1),(8,0,0),(0,8,0),(32,256,0),(64,128,0),(128,64,0)],
        (27, 3) => &[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(4,0,0),(0,2,0),(64,80,0),(16,16,2),(32,32,1),(0,4,0),(8,0,0),(128,8,0),(32,128,0),(64,64,0)],
        (27, 4) => &[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,1,0),(2,0,0),(0,2,0),(64,80,0),(16,16,2),(32,32,1),(4,0,0),(0,4,0),(8,128,0),(64,8,0),(32,64,0)],
        _ => panic!("unsupported v4 32-pipe mode={mode} lb={lb}"),
    };
    ref_eval_masks(masks, x, y, z)
}

fn ref_in_block_msaa(mode: u32, lb: u32, samples: u32, v4: bool, x: u32, y: u32, z: u32, s: u32) -> u32 {
    let eq = match (v4, mode, lb, samples) {
        (false, 24, 0, 2) => "from bit 0: s0 x0 y0 x1 y1 x2 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y3 x6 x7^y7",
        (false, 24, 0, 4) => "from bit 0: s0 s1 x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y6",
        (false, 24, 0, 8) => "from bit 0: s0 s1 s2 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y2^y7",
        (false, 24, 1, 2) => "from bit 1: s0 x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y7",
        (false, 24, 1, 4) => "from bit 1: s0 s1 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y2^y6",
        (false, 24, 1, 8) => "from bit 1: s0 s1 s2 x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y2^y6 x3 y3 x2^x6^y6",
        (false, 24, 2, 2) => "from bit 2: s0 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y2^y7",
        (false, 24, 2, 4) => "from bit 2: s0 s1 x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y2^y6 x3 y3 x2^x6^y6",
        (false, 24, 2, 8) => "from bit 2: s0 s1 s2 x0 y0 x1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y2^y6 x2^x6^y6 x3 y3 x6^y1^y6",
        (false, 27, 0, 2) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 y7 x7^y8^s0",
        (false, 27, 0, 4) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 x7^s0 x8^y7^s1",
        (false, 27, 0, 8) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x8^y8^s2 y6 x6^s0 x7^y7^s1",
        (false, 27, 1, 2) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 y6 x7^y7^s0",
        (false, 27, 1, 4) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 x7^s0 x8^y6^s1",
        (false, 27, 1, 8) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x8^y8^s2 y3 x6^s0 x7^y6^s1",
        (false, 27, 2, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x8^y8^s0 x3 y3 x6^y6",
        (false, 27, 2, 4) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x8^y8^s1 x3 y3 x6^y6^s0",
        (false, 27, 2, 8) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^s2 x8^y8^s1 x3 y3 x6^y6^s0",
        (false, 27, 3, 2) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x8^y8^s0 y2 x3 x6^y3",
        (false, 27, 3, 4) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z1 x6^y5^s1 x5^y6^z0 x8^y8^s0 y2 x3 x6^y3",
        (false, 27, 3, 8) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z0 x6^y5^s2 x5^y6^s1 x8^y8^s0 y2 x3 x6^y3",
        (false, 27, 4, 2) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^s0 x3^x8^y8 x2 y2 x6^y3",
        (false, 27, 4, 4) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z0 x6^y5^s1 x5^y6^s0 x3^x8^y8 x2 y2 x6^y3",
        (false, 27, 4, 8) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^s2 x6^y5^s1 x5^y6^s0 x3^x8^y8 x2 y2 x6^y3",
        (false, 28, 0, 2) => "from bit 0: s0 x0 y0 x1 y1 x2 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 y3 x6 x7^y10 x10^y7 x8^y9 x9^y8",
        (false, 28, 0, 4) => "from bit 0: s0 s1 x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y9 x10^y6 x8^y8 x9^y7",
        (false, 28, 0, 8) => "from bit 0: s0 s1 s2 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y10 x7^y9 x9^y7 x8^y2^y8",
        (false, 28, 1, 2) => "from bit 1: s0 x0 y0 x1 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y10 x7^y9 x8^y8 x9^y7",
        (false, 28, 1, 4) => "from bit 1: s0 s1 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x10^y6 x7^y9 x9^y7 x8^y2^y8",
        (false, 28, 1, 8) => "from bit 1: s0 s1 s2 x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y10 x7^y9 x2^x8^y8 x9^y2^y7",
        (false, 28, 2, 2) => "from bit 2: s0 x0 y0 x1 y1 x2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y10 x7^y9 x9^y7 x8^y2^y8",
        (false, 28, 2, 4) => "from bit 2: s0 s1 x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x7^y9 x10^y6 x2^x8^y8 x9^y2^y7",
        (false, 28, 2, 8) => "from bit 2: s0 s1 s2 x0 y0 x1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6 x6^y6 x3 y3 x6^y10 x7^y1^y9 x2^x8^y8 x9^y2^y7",
        (false, 31, 0, 2) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 x7 x11^y7 x8^y10 x10^y8 x9^y9^s0",
        (false, 31, 0, 4) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y6 x7 x8^y10 x11^y7 x9^y9^s0 x10^y8^s1",
        (false, 31, 0, 8) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x6^x8^y8 y6 x7 x11^y7 x8^y10^s0 x10^y8^s1 x9^y9^s2",
        (false, 31, 1, 2) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 y6 x7^y10 x8^y9 x10^y7 x9^y8^s0",
        (false, 31, 1, 4) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y3 y6 x7^y10 x10^y7 x8^y9^s0 x9^y8^s1",
        (false, 31, 1, 8) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x6^x8^y8 y3 y6 x7^y10 x8^y9^s0 x10^y7^s1 x9^y8^s2",
        (false, 31, 2, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x3 y3 x10^y6 x7^y9 x9^y7 x8^y8^s0",
        (false, 31, 2, 4) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x3 y3 x7^y9 x10^y6 x8^y8^s0 x9^y7^s1",
        (false, 31, 2, 8) => "from bit 2: x0 x1 y0 y1 x2 y2 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x6^x8^y8 x3 y3 x10^y6 x7^y9^s0 x9^y7^s1 x8^y8^s2",
        (false, 31, 3, 2) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y2 x3 x10^y3 x7^y8 x9^y6 x8^y7^s0",
        (false, 31, 3, 4) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 y2 x3 x10^y3 x9^y6 x7^y8^s0 x8^y7^s1",
        (false, 31, 3, 8) => "from bit 3: x0 y0 x1 x2 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x6^x8^y8 y2 x3 x10^y3 x7^y8^s0 x9^y6^s1 x8^y7^s2",
        (false, 31, 4, 2) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x2 y2 x9^y3 x3^y8 x8^y6 x7^y7^s0",
        (false, 31, 4, 4) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z2 x6^y5^z1 x5^y6^z0 x6^x8^y8 x2 y2 x3^y8 x9^y3 x7^y7^s0 x8^y6^s1",
        (false, 31, 4, 8) => "from bit 4: x0 y0 x1 y1 x7^y4^y7 x4^y4^z1 x6^y5^z0 x5^y6^z2 x3^x8^y8 x2 y2 x9^y3 x6^y8^s0 x8^y6^s1 x7^y7^s2",
        (true, 24, 0, 2) => "from bit 0: s0 x0 y0 x1 y1 x2 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x8^y6 x6^y8 x7^y7",
        (true, 24, 0, 4) => "from bit 0: s0 s1 x0 y0 x1 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x6^y7 x7^y6",
        (true, 24, 0, 8) => "from bit 0: s0 s1 s2 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x7^y6 x6^y2^y7",
        (true, 24, 1, 2) => "from bit 1: s0 x0 y0 x1 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x6^y7 x7^y6",
        (true, 24, 1, 4) => "from bit 1: s0 s1 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x6^y7 x7^y2^y6",
        (true, 24, 1, 8) => "from bit 1: s0 s1 s2 x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x2^x6^y7 x7^y2^y6",
        (true, 24, 2, 2) => "from bit 2: s0 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x7^y6 x6^y2^y7",
        (true, 24, 2, 4) => "from bit 2: s0 s1 x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x2^x6^y7 x7^y2^y6",
        (true, 24, 2, 8) => "from bit 2: s0 s1 s2 x0 y0 x1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y1^y8 x2^x6^y7 x7^y2^y6",
        (true, 27, 0, 2) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x9^y6 x8^y7 x7^y8^s0",
        (true, 27, 0, 4) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x9^y6 x7^y8^s0 x8^y7^s1",
        (true, 27, 0, 8) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 y6 x6^y9^s0 x8^y7^s1 x7^y8^s2",
        (true, 27, 1, 2) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y8 x8^y6 x7^y7^s0",
        (true, 27, 1, 4) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y8 x8^y6^s0 x7^y7^s1",
        (true, 27, 1, 8) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y8^s0 x8^y6^s1 x7^y7^s2",
        (true, 27, 2, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x7^y6 x6^y7^s0",
        (true, 27, 2, 4) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8 x6^y7^s0 x7^y6^s1",
        (true, 27, 2, 8) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y8^s0 x7^y6^s1 x6^y7^s2",
        (true, 27, 3, 2) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x7^y3 x5^y7 x6^y6^s0",
        (true, 27, 3, 4) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x8^y3 x6^y6^s0 x7^y5^s1",
        (true, 27, 3, 8) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z0 x5^y5^s2 y2 x3 x7^y3 x5^y7^s0 x6^y6^s1",
        (true, 27, 4, 2) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x2 y2 x6^y3 x3^y7 x5^y6^s0",
        (true, 27, 4, 4) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z0 x5^y5^s1 x2 y2 x6^y3 x3^y7 x5^y6^s0",
        (true, 27, 4, 8) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^s2 x5^y5^s1 x2 y2 x6^y3 x3^y7 x5^y6^s0",
        (true, 28, 0, 2) => "from bit 0: s0 x0 y0 x1 y1 x2 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6 x7^y8 x8^y7",
        (true, 28, 0, 4) => "from bit 0: s0 s1 x0 y0 x1 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y7",
        (true, 28, 0, 8) => "from bit 0: s0 s1 s2 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y2^y7",
        (true, 28, 1, 2) => "from bit 1: s0 x0 y0 x1 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y7",
        (true, 28, 1, 4) => "from bit 1: s0 s1 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y2^y7",
        (true, 28, 1, 8) => "from bit 1: s0 s1 s2 x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x2^x8^y6 x7^y2^y7",
        (true, 28, 2, 2) => "from bit 2: s0 x0 y0 x1 y1 x2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y2^y7",
        (true, 28, 2, 4) => "from bit 2: s0 s1 x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x2^x8^y6 x7^y2^y7",
        (true, 28, 2, 8) => "from bit 2: s0 s1 s2 x0 y0 x1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y1^y8 x2^x8^y6 x7^y2^y7",
        (true, 31, 0, 2) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x10^y6 x7^y9 x9^y7 x8^y8^s0",
        (true, 31, 0, 4) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x10^y6 x7^y9 x9^y7^s0 x8^y8^s1",
        (true, 31, 0, 8) => "from bit 0: x0 x1 x2 x3 y0 y1 y2 y3 x6^y4^y6 x4^y4^z1 x5^y5^z0 x5 x6 x10^y6 x7^y9^s0 x9^y7^s1 x8^y8^s2",
        (true, 31, 1, 2) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6 x7^y8 x8^y7^s0",
        (true, 31, 1, 4) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6 x7^y8^s0 x8^y7^s1",
        (true, 31, 1, 8) => "from bit 1: x0 x1 x2 y0 y1 y2 x3 x6^y4^y6 x4^y4^z1 x5^y5^z0 y3 x5 x6^y9 x9^y6^s0 x7^y8^s1 x8^y7^s2",
        (true, 31, 2, 2) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6 x7^y7^s0",
        (true, 31, 2, 4) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8 x8^y6^s0 x7^y7^s1",
        (true, 31, 2, 8) => "from bit 2: x0 x1 y0 y1 x2 y2 x6^y4^y6 x4^y4^z1 x5^y5^z0 x3 y3 x5^y9 x6^y8^s0 x8^y6^s1 x7^y7^s2",
        (true, 31, 3, 2) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x8^y3 x5^y8 x6^y7 x7^y6^s0",
        (true, 31, 3, 4) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x8^y3 x5^y8 x6^y7^s0 x7^y6^s1",
        (true, 31, 3, 8) => "from bit 3: x0 y0 x1 x2 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 y2 x3 x9^y3 x6^y7^s0 x8^y5^s1 x7^y6^s2",
        (true, 31, 4, 2) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x2 y2 x3^y8 x7^y3 x5^y7 x6^y6^s0",
        (true, 31, 4, 4) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z1 x5^y5^z0 x2 y2 x8^y3 x3^y7 x6^y6^s0 x7^y5^s1",
        (true, 31, 4, 8) => "from bit 4: x0 y0 x1 y1 x6^y4^y6 x4^y4^z0 x5^y5^s2 x2 y2 x7^y3 x3^y8 x5^y7^s0 x6^y6^s1",
        _ => panic!("unsupported MSAA equation v4={v4} mode={mode} lb={lb} samples={samples}"),
    };
    let bits = ref_block_bits(mode, v4);
    ref_eval_terms(eq, x, y, z, s) & ((1 << bits) - 1)
}

/// Independent reference for thick 3D in-block offset (modes 5, 9, 17, 21, 25, 26).
fn ref_in_block_3d_thick(mode: u32, lb: u32, v4: bool, x: u32, y: u32, z: u32) -> u32 {
    let eq = match (v4, mode, lb) {
        (_, 5, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 y2 x3 z3 y3",
        (_, 5, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 y2 x2 z3 y3",
        (_, 5, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 y2 x2 z2 y3",
        (_, 5, 3) => "from bit 3: x0 z0 y0 x1 z1 y1 x2 z2 y2",
        (_, 5, 4) => "from bit 4: z0 y0 x0 z1 y1 x1 z2 y2",
        (_, 9, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 y2 x3 z3 y3 x4 z4 y4 x5",
        (_, 9, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 y2 x2 z3 y3 x3 z4 y4 x4",
        (_, 9, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 y2 x2 z2 y3 x3 z3 y4 x4",
        (_, 9, 3) => "from bit 3: x0 z0 y0 x1 z1 y1 x2 z2 y2 x3 z3 y3 x4",
        (_, 9, 4) => "from bit 4: z0 y0 x0 z1 y1 x1 z2 y2 x2 z3 y3 x3",
        (false, 17, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 y2 x3 z3 x5^y3 x4^y4^z4 z4 y4 x5",
        (false, 17, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 y2 x2 z3 x4^y3 x3^y4^z4 z4 y4 x4",
        (false, 17, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 y2 x2 z2 x4^y3 x3^y4^z3 z3 y4 x4",
        (false, 17, 3) => "from bit 3: x0 z0 y0 x1 z1 y1 x2 z2 x4^y2 x3^y3^z3 z3 y3 x4",
        (false, 17, 4) => "from bit 4: z0 y0 x0 z1 y1 x1 z2 x3^y2 x2^y3^z3 z3 y3 x3",
        (false, 21, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 x6^y2^z6 x3^y5^z5 x5^y4^z3 x4^y3^z4",
        (false, 21, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 x5^y2^z6 x2^y5^z5 x4^y4^z3 x3^y3^z4",
        (false, 21, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 x5^y2^z5 x2^y5^z4 x4^y4^z2 x3^y3^z3",
        (false, 21, 3) => "from bit 3: x0 z0 y0 x1 z1 x5^y1^z5 x2^y4^z4 x4^y3^z2 x3^y2^z3",
        (false, 21, 4) => "from bit 4: z0 y0 x0 z1 x4^y1^z5 x1^y4^z4 x3^y3^z2 x2^y2^z3",
        (false, 25, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 x7^y2^z7 x3^y6^z6 x6^y5^z3 x5^y3^z5 x4^y4^z4 z4 y4 x5",
        (false, 25, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 x6^y2^z7 x2^y6^z6 x5^y5^z3 x4^y3^z5 x3^y4^z4 z4 y4 x4",
        (false, 25, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 x6^y2^z6 x2^y6^z5 x5^y5^z2 x4^y3^z4 x3^y4^z3 z3 y4 x4",
        (false, 25, 3) => "from bit 3: x0 z0 y0 x1 z1 x6^y1^z6 x2^y5^z5 x5^y4^z2 x4^y2^z4 x3^y3^z3 z3 y3 x4",
        (false, 25, 4) => "from bit 4: z0 y0 x0 z1 x5^y1^z6 x1^y5^z5 x4^y4^z2 x3^y2^z4 x2^y3^z3 z3 y3 x3",
        (false, 26, 0) => "from bit 0: x0 x1 z0 y0 y1 z1 x2 z2 x7^y4^y7 x4^y4 x6^y2^y5 x5^y6 x7^z3 y3 x3 z4",
        (false, 26, 1) => "from bit 1: x0 z0 y0 x1 z1 y1 z2 x7^y4^y7 x4^y4 x6^y2^y5 x5^y3^y6 x7^z3 x2 z4 x3",
        (false, 26, 2) => "from bit 2: x0 y0 x1 z0 y1 z1 x7^y4^y7 x4^y4 x6^y2^y5 x5^y3^y6 x7^z2 x2 z3 x3",
        (true, 17, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 x5^y2 x3^y4^z4 x4^y3^z3 y3 x4 z4 y4 x5",
        (true, 17, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 x4^y2 x2^y4^z4 x3^y3^z3 y3 x3 z4 y4 x4",
        (true, 17, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 x4^y2 x2^y4^z3 x3^y3^z2 y3 x3 z3 y4 x4",
        (true, 17, 3) => "from bit 3: x0 z0 y0 x1 z1 x4^y1 x2^y3^z3 x3^y2^z2 y2 x3 z3 y3 x4",
        (true, 17, 4) => "from bit 4: z0 y0 x0 z1 x3^y1 x1^y3^z3 x2^y2^z2 y2 x2 z3 y3 x3",
        (true, 21, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 x5^y2^z5 x3^y4^z4 x4^y3^z3 y3",
        (true, 21, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 x4^y2^z5 x2^y4^z4 x3^y3^z3 y3",
        (true, 21, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 x4^y2^z4 x2^y4^z3 x3^y3^z2 y3",
        (true, 21, 3) => "from bit 3: x0 z0 y0 x1 z1 x4^y1^z4 x2^y3^z3 x3^y2^z2 y2",
        (true, 21, 4) => "from bit 4: z0 y0 x0 z1 x3^y1^z4 x1^y3^z3 x2^y2^z2 y2",
        (true, 25, 0) => "from bit 0: x0 x1 z0 y0 z1 y1 x2 z2 x5^y2^z5 x3^y4^z4 x4^y3^z3 y3 x4 z4 y4 x5",
        (true, 25, 1) => "from bit 1: x0 z0 y0 z1 y1 x1 z2 x4^y2^z5 x2^y4^z4 x3^y3^z3 y3 x3 z4 y4 x4",
        (true, 25, 2) => "from bit 2: x0 y0 z0 y1 x1 z1 x4^y2^z4 x2^y4^z3 x3^y3^z2 y3 x3 z3 y4 x4",
        (true, 25, 3) => "from bit 3: x0 z0 y0 x1 z1 x4^y1^z4 x2^y3^z3 x3^y2^z2 y2 x3 z3 y3 x4",
        (true, 25, 4) => "from bit 4: z0 y0 x0 z1 x3^y1^z4 x1^y3^z3 x2^y2^z2 y2 x2 z3 y3 x3",
        (true, 26, 0) => "from bit 0: x0 x1 z0 y0 y1 z1 x2 z2 x6^y4^y6 x4^y4 x5^y5 y2 y3 z3 x3 z4",
        (true, 26, 1) => "from bit 1: x0 z0 y0 x1 z1 y1 z2 x6^y4^y6 x4^y4 x5^y2^y5 z3 y3 x2 z4 x3",
        (true, 26, 2) => "from bit 2: x0 y0 x1 z0 y1 z1 x6^y4^y6 x4^y4 x5^y2^y5 z2 y3 x2 z3 x3",
        _ => panic!("unsupported 3D thick equation v4={v4} mode={mode} lb={lb}"),
    };
    let bits = ref_block_bits(mode, v4);
    ref_eval_terms(eq, x, y, z, 0) & ((1 << bits) - 1)
}

#[derive(Clone, Copy)]
struct RefLevel {
    offset_bytes: u64,
    pitch_bytes: u64,
    padded_width: u32,
    width_elements: u32,
    height_elements: u32,
    depth_or_array: u32,
    in_tail: bool,
    tail_x: u32,
    tail_y: u32,
}

struct RefLayout {
    mode: u32,
    bpe: u32,
    samples: u32,
    v4: bool,
    thick: bool,
    block_bits: u32,
    block_w: u32,
    block_h: u32,
    block_d: u32,
    slice_size: u64,
    size_bytes: u64,
    levels: Vec<RefLevel>,
}

fn ref_tail_slot_xy(slot: u64, kx: u32, ky: u32) -> (u32, u32) {
    let mut rx = kx;
    let mut ry = ky;
    let mut x_turn = true;
    let mut x = 0u32;
    let mut y = 0u32;
    for k in (0..kx + ky).rev() {
        let bit = ((slot >> k) & 1) as u32;
        if (x_turn && rx > 0) || ry == 0 {
            x = (x << 1) | bit;
            rx -= 1;
        } else {
            y = (y << 1) | bit;
            ry -= 1;
        }
        x_turn = !x_turn;
    }
    (x, y)
}

fn ref_compute_layout(w_tex: u32, h_tex: u32, depth_or_array: u32, mips_in: u32, bpe: u32, blk: u32, samples: u32, mode: u32, is_3d: bool, v4: bool) -> RefLayout {
    let mode = mode & 0x1F;
    let n = if mips_in == 0 {
        let top = if is_3d { w_tex.max(h_tex).max(depth_or_array) } else { w_tex.max(h_tex) };
        top.max(1).ilog2() + 1
    } else {
        mips_in
    };
    let n = if samples > 1 { 1 } else { n };
    let lb = bpe.ilog2();
    let ls = samples.ilog2();
    let w0 = u64::from(w_tex).div_ceil(u64::from(blk));
    let h0 = u64::from(h_tex).div_ceil(u64::from(blk));

    let level_wh = |i: u32| -> (u32, u32, u32) {
        let wi = w0.div_ceil(1u64 << i).max(1) as u32;
        let hi = h0.div_ceil(1u64 << i).max(1) as u32;
        let di = if is_3d { u64::from(depth_or_array).div_ceil(1u64 << i).max(1) as u32 } else { depth_or_array };
        (wi, hi, di)
    };

    if mode == 0 {
        let mut levels = vec![RefLevel { offset_bytes: 0, pitch_bytes: 0, padded_width: 0, width_elements: 0, height_elements: 0, depth_or_array: 0, in_tail: false, tail_x: 0, tail_y: 0 }; n as usize];
        let mut run = 0u64;
        for i in (0..n).rev() {
            let (wi, hi, di) = level_wh(i);
            let pitch = align_up(u64::from(wi) * u64::from(bpe), 256);
            let slice2d = pitch * u64::from(hi);
            levels[i as usize] = RefLevel {
                offset_bytes: run,
                pitch_bytes: pitch,
                padded_width: (pitch / u64::from(bpe)) as u32,
                width_elements: wi,
                height_elements: hi,
                depth_or_array: di,
                in_tail: false,
                tail_x: 0,
                tail_y: 0,
            };
            run += slice2d;
        }
        let size_bytes = run * u64::from(depth_or_array);
        return RefLayout { mode, bpe, samples, v4, thick: false, block_bits: 0, block_w: 1, block_h: 1, block_d: 1, slice_size: run, size_bytes, levels };
    }

    let bits = ref_block_bits(mode, v4);
    let block_size = 1u64 << bits;
    let thick = ref_is_thick(mode, is_3d);

    if thick {
        const MICRO_3D: [(u32, u32, u32); 5] = [(8, 4, 8), (4, 4, 8), (4, 4, 4), (4, 2, 4), (2, 2, 4)];
        const TAIL_3D: [(u32, u32); 10] = [(4, 0), (0, 4), (2, 0), (1, 2), (0, 3), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)];
        let (bw, bh, bd) = ref_block_shape_thick(bits, lb);
        let (mw, mh, _md) = MICRO_3D[lb as usize];
        let (fw, fh) = if bits.is_multiple_of(3) { (bw, bh / 2) } else { (bw / 2, bh) };
        let max_tail = if bits == 12 { 5 } else { 10 };
        let mut first = n;
        if n > 1 {
            for i in 0..n {
                let (wi, hi, _) = level_wh(i);
                if wi <= fw && hi <= fh {
                    first = i.max(n.saturating_sub(max_tail));
                    break;
                }
            }
        }
        let mut levels = vec![RefLevel { offset_bytes: 0, pitch_bytes: 0, padded_width: 0, width_elements: 0, height_elements: 0, depth_or_array: 0, in_tail: false, tail_x: 0, tail_y: 0 }; n as usize];
        let mut run = if first < n { block_size } else { 0 };
        for i in (0..first).rev() {
            let (wi, hi, di) = level_wh(i);
            let pw = align_up(u64::from(wi), u64::from(bw)) as u32;
            let ph = align_up(u64::from(hi), u64::from(bh)) as u32;
            let slice2d = u64::from(pw) * u64::from(ph) * u64::from(bpe);
            levels[i as usize] = RefLevel {
                offset_bytes: run,
                pitch_bytes: u64::from(pw) * u64::from(bpe),
                padded_width: pw,
                width_elements: wi,
                height_elements: hi,
                depth_or_array: di,
                in_tail: false,
                tail_x: 0,
                tail_y: 0,
            };
            run += slice2d * u64::from(bd);
        }
        let slab = align_up(run, block_size);
        let slot0 = if bits == 12 { 5 } else { 0 };
        for j in 0..(n - first) {
            let i = first + j;
            let (wi, hi, di) = level_wh(i);
            let pw = (fw >> j.min(31)).max(mw);
            let _ph = (fh >> j.min(31)).max(mh);
            let (px, py) = TAIL_3D[(slot0 + j as usize).min(TAIL_3D.len() - 1)];
            levels[i as usize] = RefLevel {
                offset_bytes: 0,
                pitch_bytes: u64::from(pw) * u64::from(bpe),
                padded_width: pw,
                width_elements: wi,
                height_elements: hi,
                depth_or_array: di,
                in_tail: true,
                tail_x: px * mw,
                tail_y: py * mh,
            };
        }
        let slabs = align_up(u64::from(depth_or_array), u64::from(bd)) / u64::from(bd);
        return RefLayout { mode, bpe, samples, v4, thick: true, block_bits: bits, block_w: bw, block_h: bh, block_d: bd, slice_size: slab, size_bytes: slab * slabs, levels };
    }

    const SLOTS: [u64; 16] = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 6, 5, 4, 3, 2, 1, 0];
    let (bw, bh) = ref_block_shape_2d(bits, lb, ls);
    let (mw, mh) = ref_block_shape_2d(8, lb, 0);
    let (tw, th) = (bw / 2, bh);
    let (fw, fh) = match (mode, bpe) {
        (24 | 28, 1) => (tw / 2, th / 2),
        (24 | 28, 2) => (tw / 2, th),
        _ => (tw, th),
    };
    let mut first = n;
    if bits > 8 && n > 1 {
        for i in 0..n {
            let (wi, hi, _) = level_wh(i);
            if wi <= fw && hi <= fh {
                first = i.max(n.saturating_sub(bits - 4));
                break;
            }
        }
    }
    let mut levels = vec![RefLevel { offset_bytes: 0, pitch_bytes: 0, padded_width: 0, width_elements: 0, height_elements: 0, depth_or_array: 0, in_tail: false, tail_x: 0, tail_y: 0 }; n as usize];
    let mut run = if first < n { block_size } else { 0 };
    for i in (0..first).rev() {
        let (wi, hi, di) = level_wh(i);
        let pw = align_up(u64::from(wi), u64::from(bw)) as u32;
        let ph = align_up(u64::from(hi), u64::from(bh)) as u32;
        let size = u64::from(pw) * u64::from(ph) * u64::from(bpe) * u64::from(samples);
        levels[i as usize] = RefLevel {
            offset_bytes: run,
            pitch_bytes: u64::from(pw) * u64::from(bpe),
            padded_width: pw,
            width_elements: wi,
            height_elements: hi,
            depth_or_array: di,
            in_tail: false,
            tail_x: 0,
            tail_y: 0,
        };
        run += size;
    }
    let slice_size = align_up(run, block_size);
    if first < n {
        let (kx, ky) = ((bw / mw).ilog2(), (bh / mh).ilog2());
        for j in 0..(n - first) {
            let i = first + j;
            let (wi, hi, di) = level_wh(i);
            let t = SLOTS[((20 - bits) + j).min(15) as usize];
            let (px, py) = ref_tail_slot_xy(t, kx, ky);
            let pw = (tw >> j.min(31)).max(mw);
            let _ph = (th >> j.min(31)).max(mh);
            levels[i as usize] = RefLevel {
                offset_bytes: if is_3d { 0 } else { t * 256 },
                pitch_bytes: u64::from(pw) * u64::from(bpe),
                padded_width: pw,
                width_elements: wi,
                height_elements: hi,
                depth_or_array: di,
                in_tail: true,
                tail_x: px * mw,
                tail_y: py * mh,
            };
        }
    }
    RefLayout { mode, bpe, samples, v4, thick: false, block_bits: bits, block_w: bw, block_h: bh, block_d: 1, slice_size, size_bytes: slice_size * u64::from(depth_or_array), levels }
}

impl RefLayout {
    fn element_offset(&self, mip: u32, x: u32, y: u32, z: u32, s: u32) -> u64 {
        let l = &self.levels[mip as usize];
        if self.mode == 0 {
            return u64::from(z) * self.slice_size + l.offset_bytes + u64::from(y) * l.pitch_bytes + u64::from(x) * u64::from(self.bpe);
        }
        let block = 1u64 << self.block_bits;
        let z_base = if self.thick { u64::from(z / self.block_d) * self.slice_size } else { u64::from(z) * self.slice_size };
        let (ex, ey) = if l.in_tail { (x + l.tail_x, y + l.tail_y) } else { (x, y) };
        let in_block = if self.thick {
            ref_in_block_3d_thick(self.mode, self.bpe.ilog2(), self.v4, ex, ey, z)
        } else {
            ref_in_block_2d(self.mode, self.bpe, self.samples, self.v4, ex, ey, z, s)
        };
        if l.in_tail {
            return z_base + u64::from(in_block);
        }
        let blocks_x = l.padded_width / self.block_w;
        let block_index = u64::from((y / self.block_h) * blocks_x + (x / self.block_w));
        z_base + l.offset_bytes + block_index * block + u64::from(in_block)
    }
}

fn create_2d_with_usage(w: u32, h: u32, array: u32, mips: u32, format: u32, samples: u32, mode: u32, bind_flags: u32) -> Computer {
    let desc = XgTexture2dDesc {
        width: w,
        height: h,
        mip_levels: mips,
        array_size: array,
        format,
        sample_count: samples,
        sample_quality: 0,
        usage: if mode == 0 { 3 } else { 0 },
        bind_flags,
        cpu_access_flags: 0,
        misc_flags: 0,
        swizzle_mode: mode,
        reserved: 0,
    };
    let mut out = std::ptr::null_mut();
    let hr = unsafe { XGCreateTexture2DComputer(&desc, &mut out) };
    assert_eq!(hr, OK, "XGCreateTexture2DComputer failed for mode {mode} format {format} {w}x{h}x{array} mips {mips} samples {samples}");
    Computer(out)
}

fn verify_bijection_bounds_and_ref(c: &Computer, r_layout: &RefLayout, label: &str) {
    let size = c.size();
    assert_eq!(size, r_layout.size_bytes, "{label}: resource size mismatch");
    let bpe = u64::from(r_layout.bpe);
    // Mark every occupied byte slot (element index = offset / bpe) in a flat bitset.
    let total_slots = (size / bpe) as usize;
    let mut used = vec![0u64; total_slots.div_ceil(64)];
    for (mip, l) in r_layout.levels.iter().enumerate() {
        let mip = mip as u32;
        for z in 0..l.depth_or_array {
            for s in 0..r_layout.samples {
                for y in 0..l.height_elements {
                    for x in 0..l.width_elements {
                        let crate_addr = c.texel(0, mip, u64::from(x), y, z, s);
                        let ref_addr = r_layout.element_offset(mip, x, y, z, s);
                        assert_ne!(crate_addr, u64::MAX, "{label}: out of range at mip {mip} ({x},{y},{z}) s {s}");
                        assert!(crate_addr + bpe <= size, "{label}: out of bounds at mip {mip} ({x},{y},{z}) s {s}: {crate_addr:#x} + {bpe} > {size:#x}");
                        assert_eq!(crate_addr % bpe, 0, "{label}: unaligned address {crate_addr:#x} for bpe {bpe}");
                        assert_eq!(
                            crate_addr, ref_addr,
                            "{label}: first differing texel mip {mip} ({x},{y},{z}) s {s}: crate={crate_addr:#x}, ref={ref_addr:#x}"
                        );
                        let slot = (crate_addr / bpe) as usize;
                        let word = slot / 64;
                        let bit = 1u64 << (slot % 64);
                        assert_eq!(used[word] & bit, 0, "{label}: collision at byte {crate_addr:#x} for mip {mip} ({x},{y},{z}) s {s}");
                        used[word] |= bit;
                    }
                }
            }
        }
    }
}

#[test]
fn prop_2d_all_modes_bijection_bounds_and_reference() {
    let _s = serial();
    let mut rng = Rng(0xD371_1E00_2026_0001);
    let mut cases_checked = 0u32;

    for v4 in [false, true] {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });
        for mode in ALL_MODES {
            for (format, bpe, blk) in FORMATS {
                if !ref_mode_valid_bpe(mode, bpe) {
                    continue;
                }
                // Deterministic edge sizes + PRNG sizes, keeping each surface small enough for exhaustive element bijection.
                let mut shapes: Vec<(u32, u32, u32, u32)> = vec![
                    (1, 1, 1, 0),
                    (17, 33, 1, 0),
                    (65, 31, 2, 0),
                    (129, 65, 1, 1),
                ];
                for _ in 0..3 {
                    let w = rng.range(1, 140);
                    let h = rng.range(1, 140);
                    let array = rng.range(1, 3);
                    let mips = if rng.next_u64().is_multiple_of(2) { 0 } else { rng.range(1, 4) };
                    shapes.push((w, h, array, mips));
                }
                // For VAR modes (512 KiB in v0, 128 KiB in v4), include a multi-block surface > 512 KiB.
                if matches!(mode, 28 | 31) && bpe >= 4 && blk == 1 {
                    shapes.push((260, 260, 1, 1));
                }

                for (w, h, array, mips) in shapes {
                    let bind = if matches!(mode, 24 | 28) && blk == 4 { 0x28 } else { 0x8 };
                    let mips_eff = if matches!(mode, 24 | 28) && blk == 4 { 1 } else { mips };
                    let c = create_2d_with_usage(w, h, array, mips_eff, format, 1, mode, bind);
                    let r_layout = ref_compute_layout(w, h, array, mips_eff, bpe, blk, 1, mode, false, v4);
                    let label = format!("2D v4={v4} mode={mode} fmt={format} (bpe={bpe},blk={blk}) {w}x{h}x{array} mips={mips_eff}");
                    verify_bijection_bounds_and_ref(&c, &r_layout, &label);
                    cases_checked += 1;
                }
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(cases_checked > 1000, "expected > 1000 2D property cases, ran {cases_checked}");
}

#[test]
fn prop_2d_tile_then_detile_identity() {
    let _s = serial();
    let mut rng = Rng(0xD371_1E00_2026_0002);
    let mut cases_checked = 0u32;

    for v4 in [false, true] {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });
        for mode in ALL_MODES {
            for (format, bpe, blk) in FORMATS {
                if !ref_mode_valid_bpe(mode, bpe) {
                    continue;
                }
                for _ in 0..4 {
                    let w = rng.range(1, 150);
                    let h = rng.range(1, 150);
                    let array = rng.range(1, 3);
                    let mips_req = if matches!(mode, 24 | 28) && blk == 4 { 1 } else if rng.next_u64().is_multiple_of(2) { 0 } else { rng.range(1, 5) };
                    let bind = if matches!(mode, 24 | 28) && blk == 4 { 0x28 } else { 0x8 };
                    let c = create_2d_with_usage(w, h, array, mips_req, format, 1, mode, bind);
                    let rec = Record(c.record());
                    let mips = rec.mips();
                    let mut memory = vec![0xA5u8; c.size() as usize];
                    let mut inputs = Vec::new();
                    let label = format!("2D roundtrip v4={v4} mode={mode} fmt={format} {w}x{h}x{array} mips={mips}");

                    for slice in 0..array {
                        for mip in 0..mips {
                            let we = w.div_ceil(blk).div_ceil(1 << mip).max(1) as usize;
                            let he = h.div_ceil(blk).div_ceil(1 << mip).max(1) as usize;
                            let pitch = we * bpe as usize;
                            let data = rng.bytes(pitch * he);
                            assert_eq!(c.copy_into(&mut memory, mip + slice * mips, &data, pitch as u32, 0), OK, "{label}");
                            inputs.push((we, he, pitch, data));
                        }
                    }

                    let mut idx = 0;
                    for slice in 0..array {
                        for mip in 0..mips {
                            let (we, _he, pitch, ref input) = inputs[idx];
                            let mut out = vec![0u8; input.len()];
                            assert_eq!(c.copy_from(&mut out, mip + slice * mips, &memory, (we * bpe as usize) as u32, 0), OK, "{label}");
                            if out != *input {
                                let pos = out.iter().zip(input.iter()).position(|(a, b)| a != b).unwrap();
                                panic!("{label} slice {slice} mip {mip}: mismatch at linear byte {pos} (pitch {pitch})");
                            }
                            idx += 1;
                        }
                    }
                    cases_checked += 1;
                }
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(cases_checked > 700, "expected > 700 2D roundtrip cases, ran {cases_checked}");
}

#[test]
fn prop_1d_and_3d_all_modes() {
    let _s = serial();
    let mut rng = Rng(0xD371_1E00_2026_0003);
    let mut cases_checked = 0u32;

    for v4 in [false, true] {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });

        // 1D modes: 0, 24, 27, 28, 31.
        for mode in [0u32, 24, 27, 28, 31] {
            for (format, bpe, blk) in FORMATS {
                if blk != 1 || !ref_mode_valid_bpe(mode, bpe) {
                    continue;
                }
                for _ in 0..3 {
                    let w = rng.range(1, 300);
                    let array = rng.range(1, 3);
                    let mips_req = if rng.next_u64().is_multiple_of(2) { 0 } else { rng.range(1, 4) };
                    let desc = XgTexture1dDesc {
                        width: w,
                        mip_levels: mips_req,
                        array_size: array,
                        format,
                        usage: if mode == 0 { 3 } else { 0 },
                        bind_flags: 0x8,
                        cpu_access_flags: 0,
                        misc_flags: 0,
                        swizzle_mode: mode,
                        pitch: 0,
                    };
                    let mut out = std::ptr::null_mut();
                    assert_eq!(unsafe { XGCreateTexture1DComputer(&desc, &mut out) }, OK);
                    let c = Computer(out);
                    let r_layout = ref_compute_layout(w, 1, array, mips_req, bpe, 1, 1, mode, false, v4);
                    let label = format!("1D v4={v4} mode={mode} fmt={format} w={w} a={array} mips={mips_req}");
                    verify_bijection_bounds_and_ref(&c, &r_layout, &label);

                    // Also check CopyIntoSubresource + CopyFromSubresource round trip for 1D.
                    let mips = r_layout.levels.len() as u32;
                    let mut mem = vec![0u8; c.size() as usize];
                    let mut inputs = Vec::new();
                    for s in 0..array {
                        for m in 0..mips {
                            let we = w.div_ceil(1 << m).max(1) as usize;
                            let data = rng.bytes(we * bpe as usize);
                            assert_eq!(c.copy_into(&mut mem, m + s * mips, &data, (we * bpe as usize) as u32, 0), OK, "{label}");
                            inputs.push(data);
                        }
                    }
                    for s in 0..array {
                        for m in 0..mips {
                            let input = &inputs[(s * mips + m) as usize];
                            let mut back = vec![0u8; input.len()];
                            assert_eq!(c.copy_from(&mut back, m + s * mips, &mem, input.len() as u32, 0), OK, "{label}");
                            assert_eq!(&back, input, "{label} 1D roundtrip slice {s} mip {m}");
                        }
                    }
                    cases_checked += 1;
                }
            }
        }

        // 3D modes: linear (0), thick (5, 9, 17, 21, 25, 26), thin (24, 27, 28, 31).
        for mode in [0u32, 5, 9, 17, 21, 24, 25, 26, 27, 28, 31] {
            for (format, bpe, blk) in FORMATS {
                if !ref_mode_valid_bpe(mode, bpe) {
                    continue;
                }
                if mode == 26 && bpe >= 8 {
                    continue;
                }
                if matches!(mode, 24 | 28) && blk == 4 {
                    continue;
                }
                for _ in 0..2 {
                    let w = rng.range(1, 36);
                    let h = rng.range(1, 36);
                    let d = rng.range(1, 20);
                    let mips_req = if rng.next_u64().is_multiple_of(2) { 0 } else { rng.range(1, 4) };
                    let desc = XgTexture3dDesc {
                        width: w,
                        height: h,
                        depth: d,
                        mip_levels: mips_req,
                        format,
                        usage: if mode == 0 { 3 } else { 0 },
                        bind_flags: 0x8,
                        cpu_access_flags: 0,
                        misc_flags: 0,
                        swizzle_mode: mode,
                        pitch: 0,
                    };
                    let mut out = std::ptr::null_mut();
                    assert_eq!(unsafe { XGCreateTexture3DComputer(&desc, &mut out) }, OK);
                    let c = Computer(out);
                    let r_layout = ref_compute_layout(w, h, d, mips_req, bpe, blk, 1, mode, true, v4);
                    let label = format!("3D v4={v4} mode={mode} fmt={format} {w}x{h}x{d} mips={mips_req}");
                    verify_bijection_bounds_and_ref(&c, &r_layout, &label);

                    // Verify CopyIntoSubresource places every element at its exact element offset.
                    let mut mem = vec![0u8; c.size() as usize];
                    for (mip, l) in r_layout.levels.iter().enumerate() {
                        let mip = mip as u32;
                        let row_pitch = l.width_elements as usize * bpe as usize;
                        let slice_pitch = row_pitch * l.height_elements as usize;
                        let data = rng.bytes(slice_pitch * l.depth_or_array as usize);
                        assert_eq!(c.copy_into(&mut mem, mip, &data, row_pitch as u32, slice_pitch as u32), OK, "{label}");
                        for z in 0..l.depth_or_array {
                            for y in 0..l.height_elements {
                                for x in 0..l.width_elements {
                                    let off = r_layout.element_offset(mip, x, y, z, 0) as usize;
                                    let lin = z as usize * slice_pitch + y as usize * row_pitch + x as usize * bpe as usize;
                                    assert_eq!(
                                        &mem[off..off + bpe as usize],
                                        &data[lin..lin + bpe as usize],
                                        "{label} CopyInto mismatch at mip {mip} ({x},{y},{z})"
                                    );
                                }
                            }
                        }
                    }
                    cases_checked += 1;
                }
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(cases_checked > 250, "expected > 250 1D/3D property cases, ran {cases_checked}");
}

#[test]
fn prop_msaa_all_modes() {
    let _s = serial();
    let mut rng = Rng(0xD371_1E00_2026_0004);
    let mut cases_checked = 0u32;

    for v4 in [false, true] {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });
        for mode in [24u32, 27, 28, 31] {
            for (format, bpe, blk) in FORMATS {
                if blk != 1 || !ref_mode_valid_bpe(mode, bpe) {
                    continue;
                }
                if matches!(mode, 24 | 28) && bpe > 4 {
                    continue;
                }
                for samples in [2u32, 4, 8] {
                    for _ in 0..2 {
                        let w = rng.range(1, 90);
                        let h = rng.range(1, 90);
                        let array = rng.range(1, 3);
                        let c = create_2d_with_usage(w, h, array, 1, format, samples, mode, 0x8);
                        let r_layout = ref_compute_layout(w, h, array, 1, bpe, 1, samples, mode, false, v4);
                        let label = format!("MSAA v4={v4} mode={mode} fmt={format} samples={samples} {w}x{h}x{array}");
                        verify_bijection_bounds_and_ref(&c, &r_layout, &label);
                        cases_checked += 1;
                    }
                }
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(cases_checked > 150, "expected > 150 MSAA property cases, ran {cases_checked}");
}
