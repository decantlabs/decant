//! The address equations of the console library: the offset inside a block as an exclusive or of coordinate bits.
//!
//! The modes without exclusive or (1, 2, 5, 6, 9, 10) and, after `XGSetHardwareVersion(4)`, the modes 21, 22 and 24 to
//! 27 use the addrlib tables of [`crate::tiling`]. All the others come from `equations_data.rs`, which holds
//! them as text rows (see `docs/CONSOLE_LIBS.md`, section 1).

use super::equations_data::{GROUP_A, GROUP_B, GROUP_C};
use crate::tiling::{self, Layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Term {
    bit: u8,
    x: u32,
    y: u32,
    z: u32,
    s: u32,
}

/// One address equation: for each output bit, the coordinate bits whose parity it takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Equation {
    terms: Vec<Term>,
}

impl Equation {
    /// The byte offset for element (x, y), array or depth slice z, sample s. The result is not reduced to the block.
    pub fn eval(&self, x: u32, y: u32, z: u32, s: u32) -> u32 {
        self.terms.iter().fold(0u32, |offset, t| {
            let parity = ((x & t.x).count_ones() + (y & t.y).count_ones() + (z & t.z).count_ones() + (s & t.s).count_ones()) & 1;
            offset | (parity << t.bit)
        })
    }

    /// The terms as `(output bit, x mask, y mask, z mask)`, for a caller that evaluates the equation elsewhere (a shader). The sample
    /// mask is left out: it is 0 for every one sample surface.
    pub fn terms(&self) -> Vec<(u32, u32, u32, u32)> {
        self.terms.iter().map(|t| (u32::from(t.bit), t.x, t.y, t.z)).collect()
    }

    /// The number of the highest output bit plus one (the block size is a power of two at least this large).
    pub fn bit_count(&self) -> u32 {
        self.terms.iter().map(|t| u32::from(t.bit) + 1).max().unwrap_or(0)
    }

    fn from_masks(masks: &[(u16, u16, u16)]) -> Equation {
        let terms = masks.iter().enumerate().filter(|(_, m)| **m != (0, 0, 0)).map(|(bit, &(x, y, z))| Term { bit: bit as u8, x: u32::from(x), y: u32::from(y), z: u32::from(z), s: 0 }).collect();
        Equation { terms }
    }

    /// Parses a row of the appendices: `from bit N: x0 y0 x1^y1 ...`.
    fn parse(row: &str) -> Option<Equation> {
        let rest = row.strip_prefix("from bit ")?;
        let (first, list) = rest.split_once(':')?;
        let first: u8 = first.trim().parse().ok()?;
        let mut terms = Vec::new();
        for (i, token) in list.split_whitespace().enumerate() {
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
            terms.push(term);
        }
        Some(Equation { terms })
    }
}

fn from_core(mode: u32, bpe: usize, rb_plus: bool, pipes_log2: u32) -> Option<Equation> {
    let mode = super::modes::core_mode(mode)?;
    let layout = Layout { pipes_log2, rb_plus, ..Layout::default() };
    Some(Equation::from_masks(&tiling::equation_masks(mode, &layout, bpe)?))
}

/// The equation of a 2D texture (also an array, a 1D texture and a depth slice of a thin volume), with `samples` 1, 2, 4 or
/// 8. `None` when the console library has none for the combination.
pub fn equation_2d(mode: u32, bpe: u32, samples: u32, v4: bool) -> Option<Equation> {
    let mode = mode & 0x1F;
    let lb = bpe.checked_ilog2()? as u8;
    if samples > 1 {
        let row = GROUP_B.iter().find(|r| r.0 == v4 && u32::from(r.1) == mode && r.2 == lb && u32::from(r.3) == samples)?;
        return Equation::parse(row.4);
    }
    match mode {
        1 | 5 | 9 => from_core(mode, bpe as usize, false, 0),
        2 | 6 | 10 => from_core(mode, bpe as usize, true, 0),
        21 | 22 | 24..=27 if v4 => from_core(mode, bpe as usize, true, 5),
        _ => {
            let row = GROUP_A.iter().find(|r| r.0 == v4 && u32::from(r.1) == mode && r.2 == lb)?;
            Equation::parse(row.3)
        }
    }
}

/// The equation of a thick 3D mode (5, 9, 17, 21, 25, 26). Mode 26 has no row for 8 and 16 byte elements.
pub fn equation_3d(mode: u32, bpe: u32, v4: bool) -> Option<Equation> {
    let lb = bpe.checked_ilog2()? as u8;
    let row = GROUP_C.iter().find(|r| r.0 == v4 && u32::from(r.1) == (mode & 0x1F) && r.2 == lb)?;
    Equation::parse(row.3)
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
