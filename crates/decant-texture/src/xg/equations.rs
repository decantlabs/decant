//! The address equations of the console library: the offset inside a block as an exclusive or of coordinate bits.
//!
//! The modes without exclusive or (1, 2, 5, 6, 9, 10) and, after `XGSetHardwareVersion(4)`, the modes 21, 22 and 24 to
//! 27 use the addrlib tables of [`crate::tiling`]. All the others come from `equations_data.rs`, which holds
//! them as text rows (see `docs/CONSOLE_LIBS.md`, section 1).

use std::sync::OnceLock;

use super::equations_data::{GROUP_A, GROUP_B, GROUP_C};
use crate::tiling::{self, Bit, Layout};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Term {
    bit: u8,
    x: u32,
    y: u32,
    z: u32,
    s: u32,
}

/// One address equation: for each output bit, the coordinate bits whose parity it takes.
#[derive(Clone, Copy, Debug, Eq)]
pub struct Equation {
    terms: [Term; Self::MAX_TERMS],
    len: u8,
}

impl PartialEq for Equation {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Equation {
    const MAX_TERMS: usize = 24;

    #[inline]
    fn as_slice(&self) -> &[Term] {
        &self.terms[..usize::from(self.len)]
    }

    /// The byte offset for element (x, y), array or depth slice z, sample s. The result is not reduced to the block.
    #[inline]
    pub fn eval(&self, x: u32, y: u32, z: u32, s: u32) -> u32 {
        let mut offset = 0u32;
        for t in self.as_slice() {
            let parity = ((x & t.x) ^ (y & t.y) ^ (z & t.z) ^ (s & t.s)).count_ones() & 1;
            offset |= parity << t.bit;
        }
        offset
    }

    /// The terms as `(output bit, x mask, y mask, z mask)`, for a caller that evaluates the equation elsewhere (a shader). The sample
    /// mask is left out: it is 0 for every one sample surface.
    pub fn terms(&self) -> Vec<(u32, u32, u32, u32)> {
        self.as_slice().iter().map(|t| (u32::from(t.bit), t.x, t.y, t.z)).collect()
    }

    /// Writes the terms into `out` as `(output bit, x mask, y mask, z mask)` without allocating when `out` has capacity.
    pub fn write_terms(&self, out: &mut Vec<(u32, u32, u32, u32)>) {
        out.clear();
        out.extend(self.as_slice().iter().map(|t| (u32::from(t.bit), t.x, t.y, t.z)));
    }

    /// The number of the highest output bit plus one (the block size is a power of two at least this large).
    pub fn bit_count(&self) -> u32 {
        self.as_slice().iter().map(|t| u32::from(t.bit) + 1).max().unwrap_or(0)
    }

    fn from_bits(bits: &[Bit]) -> Option<Equation> {
        let mut terms = [Term::default(); Self::MAX_TERMS];
        let mut len = 0usize;
        for (bit, b) in bits.iter().enumerate() {
            if b.x == 0 && b.y == 0 && b.z == 0 {
                continue;
            }
            if len >= Self::MAX_TERMS {
                return None;
            }
            terms[len] = Term { bit: bit as u8, x: u32::from(b.x), y: u32::from(b.y), z: u32::from(b.z), s: 0 };
            len += 1;
        }
        Some(Equation { terms, len: len as u8 })
    }

    /// Parses a row of the appendices: `from bit N: x0 y0 x1^y1 ...`.
    fn parse(row: &str) -> Option<Equation> {
        let rest = row.strip_prefix("from bit ")?;
        let (first, list) = rest.split_once(':')?;
        let first: u8 = first.trim().parse().ok()?;
        let mut terms = [Term::default(); Self::MAX_TERMS];
        let mut len = 0usize;
        for (i, token) in list.split_whitespace().enumerate() {
            if len >= Self::MAX_TERMS {
                return None;
            }
            let mut term = Term { bit: first + i as u8, x: 0, y: 0, z: 0, s: 0 };
            for part in token.split('^') {
                let (axis, index) = part.split_at(1);
                let mask = 1u32 << index.parse::<u32>().ok()?;
                match axis {
                    "x" => term.x ^= mask,
                    "y" => term.y ^= mask,
                    "z" => term.z ^= mask,
                    "s" => term.s ^= mask,
                    _ => return None,
                }
            }
            terms[len] = term;
            len += 1;
        }
        Some(Equation { terms, len: len as u8 })
    }
}

fn from_core(mode: u32, bpe: usize, rb_plus: bool, pipes_log2: u32) -> Option<Equation> {
    let mode = super::modes::core_mode(mode)?;
    let layout = Layout { pipes_log2, rb_plus, ..Layout::default() };
    Equation::from_bits(tiling::equation_bits(mode, &layout, bpe)?)
}

fn compute_equation_2d_1s(mode: u32, bpe: u32, v4: bool) -> Option<Equation> {
    match mode {
        1 | 5 | 9 => from_core(mode, bpe as usize, false, 0),
        2 | 6 | 10 => from_core(mode, bpe as usize, true, 0),
        21 | 22 | 24..=27 if v4 => from_core(mode, bpe as usize, true, 5),
        _ => {
            let lb = bpe.ilog2() as u8;
            let row = GROUP_A.iter().find(|r| r.0 == v4 && u32::from(r.1) == mode && r.2 == lb)?;
            Equation::parse(row.3)
        }
    }
}

/// The equation of a 2D texture (also an array, a 1D texture and a depth slice of a thin volume), with `samples` 1, 2, 4 or
/// 8. `None` when the console library has none for the combination.
pub fn equation_2d(mode: u32, bpe: u32, samples: u32, v4: bool) -> Option<Equation> {
    if !bpe.is_power_of_two() || bpe > 16 {
        return None;
    }
    let mode = mode & 0x1F;
    let lb = bpe.ilog2();
    if samples > 1 {
        if !samples.is_power_of_two() || samples > 8 {
            return None;
        }
        static TABLE_MSAA: OnceLock<Box<[Option<Equation>; 2 * 32 * 5 * 3]>> = OnceLock::new();
        let table = TABLE_MSAA.get_or_init(|| {
            let mut entries = vec![None; 2 * 32 * 5 * 3].into_boxed_slice();
            for &(rv4, rmode, rlb, rsamples, row) in GROUP_B {
                if let Some(ls) = rsamples.checked_ilog2()
                    && (1..=3).contains(&ls)
                    && rmode < 32
                    && rlb <= 4
                {
                    let idx = usize::from(rv4) * 480 + usize::from(rmode) * 15 + usize::from(rlb) * 3 + (ls as usize - 1);
                    entries[idx] = Equation::parse(row);
                }
            }
            entries.try_into().expect("exact MSAA table size")
        });
        let ls = samples.ilog2() as usize;
        let idx = usize::from(v4) * 480 + (mode as usize) * 15 + (lb as usize) * 3 + (ls - 1);
        return table[idx];
    }
    static TABLE_1S: OnceLock<Box<[Option<Equation>; 2 * 32 * 5]>> = OnceLock::new();
    let table = TABLE_1S.get_or_init(|| {
        let mut entries = vec![None; 2 * 32 * 5].into_boxed_slice();
        for v4_idx in 0..2usize {
            let rv4 = v4_idx != 0;
            for rmode in 0..32u32 {
                for rlb in 0..5u32 {
                    let idx = v4_idx * 160 + (rmode as usize) * 5 + (rlb as usize);
                    entries[idx] = compute_equation_2d_1s(rmode, 1 << rlb, rv4);
                }
            }
        }
        entries.try_into().expect("exact 2D table size")
    });
    let idx = usize::from(v4) * 160 + (mode as usize) * 5 + (lb as usize);
    table[idx]
}

/// The equation of a thick 3D mode (5, 9, 17, 21, 25, 26). Mode 26 has no row for 8 and 16 byte elements.
pub fn equation_3d(mode: u32, bpe: u32, v4: bool) -> Option<Equation> {
    if !bpe.is_power_of_two() || bpe > 16 {
        return None;
    }
    let mode = mode & 0x1F;
    let lb = bpe.ilog2();
    static TABLE_3D: OnceLock<Box<[Option<Equation>; 2 * 32 * 5]>> = OnceLock::new();
    let table = TABLE_3D.get_or_init(|| {
        let mut entries = vec![None; 2 * 32 * 5].into_boxed_slice();
        for &(rv4, rmode, rlb, row) in GROUP_C {
            if rmode < 32 && rlb <= 4 {
                let idx = usize::from(rv4) * 160 + usize::from(rmode) * 5 + usize::from(rlb);
                entries[idx] = Equation::parse(row);
            }
        }
        entries.try_into().expect("exact 3D table size")
    });
    let idx = usize::from(v4) * 160 + (mode as usize) * 5 + (lb as usize);
    table[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_parse_and_the_first_bit_follows_the_element_size() {
        for (v4, mode, lb, row) in GROUP_A.iter().copied().chain(GROUP_C.iter().copied()) {
            let e = Equation::parse(row).unwrap_or_else(|| panic!("row {mode} {lb}"));
            assert_eq!(e.terms[0].bit, lb, "mode {mode} {lb} v4 {v4}");
        }
        for (_, mode, lb, samples, row) in GROUP_B {
            let e = Equation::parse(row).unwrap();
            assert_eq!(e.terms[0].bit, *lb, "mode {mode} {lb} {samples}");
        }
    }

    /// The block shape in elements of a 2D block (the layout module repeats this rule for the real layouts).
    fn shape(bits: u32, lb: u32, ls: u32) -> (u32, u32) {
        if bits.is_multiple_of(2) {
            let e = bits - lb;
            (1 << (e.div_ceil(2) - ls.div_ceil(2)), 1 << (e / 2 - ls / 2))
        } else {
            let e = bits - lb - ls;
            (1 << e.div_ceil(2), 1 << (e / 2))
        }
    }

    #[test]
    fn every_row_is_one_to_one_inside_its_block() {
        // The offsets of the elements (and samples) of one block are distinct multiples of the element size inside the
        // block, whatever the window of the surface (the high coordinate bits only mix in).
        for mode in 0..32u32 {
            for bpe in [1u32, 2, 4, 8, 16] {
                for samples in [1u32, 2, 4, 8] {
                    for v4 in [false, true] {
                        let Some(e) = equation_2d(mode, bpe, samples, v4) else { continue };
                        let bits = super::super::modes::block_bits(mode, v4);
                        let (bw, bh) = shape(bits, bpe.ilog2(), samples.ilog2());
                        for (ox, oy, oz) in [(0u32, 0u32, 0u32), (3 * bw, 5 * bh, 3), (7 * bw, 2 * bh, 1)] {
                            let mut seen = std::collections::HashSet::new();
                            for s in 0..samples {
                                for y in 0..bh {
                                    for x in 0..bw {
                                        let offset = e.eval(ox + x, oy + y, oz, s) & ((1 << bits) - 1);
                                        assert_eq!(offset % bpe, 0, "mode {mode} bpe {bpe} samples {samples} v4 {v4}");
                                        assert!(seen.insert(offset), "mode {mode} bpe {bpe} samples {samples} v4 {v4} window {ox},{oy}: {offset:#x} twice");
                                    }
                                }
                            }
                            assert_eq!(seen.len() as u32, (1 << bits) / bpe, "mode {mode} bpe {bpe} samples {samples} v4 {v4}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_default_tables_are_the_core_tables() {
        for (mode, rb) in [(1, false), (2, true), (5, false), (6, true), (9, false), (10, true)] {
            for bpe in [1u32, 2, 4, 8, 16] {
                let ours = equation_2d(mode, bpe, 1, false).unwrap();
                let core = from_core(mode, bpe as usize, rb, 0).unwrap();
                assert_eq!(ours, core);
                assert_eq!(equation_2d(mode, bpe, 1, true).unwrap(), ours, "no version 4 change for mode {mode}");
            }
        }
    }

    #[test]
    fn missing_combinations() {
        assert!(equation_2d(3, 4, 1, false).is_none());
        assert!(equation_2d(24, 16, 1, false).is_none());
        assert!(equation_2d(24, 4, 2, false).is_some());
        assert!(equation_2d(24, 8, 2, false).is_none());
        assert!(equation_2d(27, 4, 16, false).is_none());
        assert!(equation_3d(26, 8, false).is_none());
        assert!(equation_3d(26, 4, false).is_some());
        assert!(equation_3d(5, 16, true).is_some());
        assert!(equation_2d(5, 3, 1, false).is_none());
    }

    #[test]
    fn a_known_value_of_mode_24() {
        // 4 byte elements, position (64, 0): bit 10 and bit 12 of the offset.
        let e = equation_2d(24, 4, 1, false).unwrap();
        assert_eq!(e.eval(64, 0, 0, 0), 0x1400);
        assert_eq!(e.eval(1, 0, 0, 0), 4);
        assert_eq!(e.eval(0, 1, 0, 0), 8);
    }
}
