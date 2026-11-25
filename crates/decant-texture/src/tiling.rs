// Bit tables from AMD addrlib `gfx10SwizzlePattern.h` (Mesa, `src/amd/addrlib`).
//
// Copyright (c) 2007-2026 Advanced Micro Devices, Inc.
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

//! AMD GFX10 (RDNA) tiling, turned back into rows of elements, and the place of each mip level in a
//! tiled surface.
//!
//! Inside a block, bit `i` of an element's byte offset is the exclusive or of chosen bits of its x
//! and y position (and of its slice, for the "X" modes). The chosen bits are the rows of AMD addrlib,
//! `gfx10SwizzlePattern.h` (the `GFX10_SW_*_PATINFO` tables and their NIBBLE tables), as shipped in
//! Mesa (`src/amd/addrlib`), under the MIT licence named at the top of this file; `tables.rs` is
//! generated from that header. The mip layout follows the same library's
//! `Gfx10Lib::ComputeSurfaceInfoMacroTiled` and `GetMaxNumMipsInTail`, written again from their
//! description, and the pipe/bank exclusive or its `ComputeSurfaceAddrFromCoordMacroTiled`.
//!
//! Only the bit choices and the layout rules are taken; no addrlib code is reused. Blocks follow one
//! another row by row. The texture databases of the Xbox Series build use `SW_64K_D` and `SW_4K_S`
//! (verified on rendered textures); the other modes are tabulated for the textures that
//! do not decode with those two (cubemaps, arrays) and are NOT yet verified against the game's data.

mod tables;

use tables as t;

/// One byte offset bit: the x, y and slice (z) bits whose parity it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bit {
    x: u16,
    y: u16,
    z: u16,
}

/// A GFX10 swizzle mode. `S`/`D`/`R`/`Z` is the in-block order family, `X` a mode that mixes in the
/// pipe and bank exclusive or and more position bits, and 256B/4K/64K the block size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// 64 KiB blocks, "display" order.
    Sw64kD,
    /// 4 KiB blocks, "standard" order.
    Sw4kS,
    /// 256 byte blocks, "standard" order (micro tiled: no mip tail).
    Sw256bS,
    /// 256 byte blocks, "display" order (micro tiled: no mip tail).
    Sw256bD,
    Sw4kD,
    Sw4kSX,
    Sw4kDX,
    Sw64kS,
    Sw64kSX,
    Sw64kDX,
    /// 64 KiB, render target order, one sample.
    Sw64kRX,
    /// 64 KiB, depth order, one sample.
    Sw64kZX,
}

/// What a surface's address depends on besides its mode. The default (everything 0, 256 byte pipe
/// interleave, plain tables) is what the two verified modes use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// log2 of the number of pipes (0 to 6): the "X" modes' bits depend on it.
    pub pipes_log2: u32,
    /// The RB+ variant of the tables (differs for the "D" and "X" modes).
    pub rb_plus: bool,
    /// The pipe and bank exclusive or of the surface (the tile swizzle in the first descriptor word),
    /// already masked to the pipe and bank bits; only used by the "X" modes.
    pub pipe_bank_xor: u32,
    /// log2 of the pipe interleave in bytes (8, 9 or 10).
    pub pipe_interleave_log2: u32,
    /// The array slice (or volume depth) read; only the "X" modes use it.
    pub slice: u32,
}

impl Default for Layout {
    fn default() -> Self {
        Layout { pipes_log2: 0, rb_plus: false, pipe_bank_xor: 0, pipe_interleave_log2: 8, slice: 0 }
    }
}

impl Mode {
    /// Every mode, in the order the reference vectors number them.
    pub const ALL: [Mode; 12] =
        [Mode::Sw64kD, Mode::Sw4kS, Mode::Sw256bS, Mode::Sw256bD, Mode::Sw4kD, Mode::Sw4kSX, Mode::Sw4kDX, Mode::Sw64kS, Mode::Sw64kSX, Mode::Sw64kDX, Mode::Sw64kRX, Mode::Sw64kZX];

    pub fn name(self) -> &'static str {
        match self {
            Mode::Sw64kD => "SW_64K_D",
            Mode::Sw4kS => "SW_4K_S",
            Mode::Sw256bS => "SW_256B_S",
            Mode::Sw256bD => "SW_256B_D",
            Mode::Sw4kD => "SW_4K_D",
            Mode::Sw4kSX => "SW_4K_S_X",
            Mode::Sw4kDX => "SW_4K_D_X",
            Mode::Sw64kS => "SW_64K_S",
            Mode::Sw64kSX => "SW_64K_S_X",
            Mode::Sw64kDX => "SW_64K_D_X",
            Mode::Sw64kRX => "SW_64K_R_X",
            Mode::Sw64kZX => "SW_64K_Z_X",
        }
    }

    /// log2 of the block size in bytes.
    pub fn block_log2(self) -> u32 {
        match self {
            Mode::Sw256bS | Mode::Sw256bD => 8,
            Mode::Sw4kS | Mode::Sw4kD | Mode::Sw4kSX | Mode::Sw4kDX => 12,
            _ => 16,
        }
    }

    /// Whether the mode mixes in the pipe and bank exclusive or (the "X" modes).
    pub fn is_xor(self) -> bool {
        matches!(self, Mode::Sw4kSX | Mode::Sw4kDX | Mode::Sw64kSX | Mode::Sw64kDX | Mode::Sw64kRX | Mode::Sw64kZX)
    }

    /// The "standard" order modes (`S`): thick blocks in 3D.
    pub fn is_standard(self) -> bool {
        matches!(self, Mode::Sw256bS | Mode::Sw4kS | Mode::Sw4kSX | Mode::Sw64kS | Mode::Sw64kSX)
    }

    /// The "display" order modes (`D`): thick blocks in 3D.
    pub fn is_display(self) -> bool {
        matches!(self, Mode::Sw64kD | Mode::Sw256bD | Mode::Sw4kD | Mode::Sw4kDX | Mode::Sw64kDX)
    }

    /// The table, the number of offset bits and whether its rows depend on the pipe count.
    fn source(self, rb_plus: bool) -> (&'static [Bit], usize, bool) {
        match (self, rb_plus) {
            (Mode::Sw64kD, false) => (&t::T_64K_D, 16, false),
            (Mode::Sw64kD, true) => (&t::T_64K_D_RB, 16, false),
            (Mode::Sw4kS, _) => (&t::T_4K_S, 12, false),
            (Mode::Sw256bS, _) => (&t::T_256B_S, 8, false),
            (Mode::Sw256bD, false) => (&t::T_256B_D, 8, false),
            (Mode::Sw256bD, true) => (&t::T_256B_D_RB, 8, false),
            (Mode::Sw4kD, false) => (&t::T_4K_D, 12, false),
            (Mode::Sw4kD, true) => (&t::T_4K_D_RB, 12, false),
            (Mode::Sw4kSX, false) => (&t::T_4K_S_X, 12, true),
            (Mode::Sw4kSX, true) => (&t::T_4K_S_X_RB, 12, true),
            (Mode::Sw4kDX, false) => (&t::T_4K_D_X, 12, true),
            (Mode::Sw4kDX, true) => (&t::T_4K_D_X_RB, 12, true),
            (Mode::Sw64kS, _) => (&t::T_64K_S, 16, false),
            (Mode::Sw64kSX, false) => (&t::T_64K_S_X, 16, true),
            (Mode::Sw64kSX, true) => (&t::T_64K_S_X_RB, 16, true),
            (Mode::Sw64kDX, false) => (&t::T_64K_D_X, 16, true),
            (Mode::Sw64kDX, true) => (&t::T_64K_D_X_RB, 16, true),
            (Mode::Sw64kRX, false) => (&t::T_64K_R_X, 16, true),
            (Mode::Sw64kRX, true) => (&t::T_64K_R_X_RB, 16, true),
            (Mode::Sw64kZX, false) => (&t::T_64K_Z_X, 16, true),
            (Mode::Sw64kZX, true) => (&t::T_64K_Z_X_RB, 16, true),
        }
    }

    /// The offset bits for elements of `bpe` bytes; `None` for an element size or pipe count the
    /// tables do not have.
    fn bits(self, layout: &Layout, bpe: usize) -> Option<&'static [Bit]> {
        if !bpe.is_power_of_two() || bpe > 16 || layout.pipes_log2 > 6 {
            return None;
        }
        let (table, n, by_pipes) = self.source(layout.rb_plus);
        let group = if by_pipes { layout.pipes_log2 as usize } else { 0 };
        let at = (group * 5 + bpe.trailing_zeros() as usize) * n;
        Some(&table[at..at + n])
    }

    /// The block shape for elements of `bpe` bytes: log2 of its width and height in elements (wider
    /// than high when they differ). `None` for an element size the modes do not have.
    pub fn block_shape(self, bpe: usize) -> Option<(u32, u32)> {
        let element_log2 = match bpe {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            _ => return None,
        };
        let elements_log2 = self.block_log2() - element_log2;
        let width_log2 = elements_log2.div_ceil(2);
        Some((width_log2, elements_log2 - width_log2))
    }

    /// The bytes one level of `width` × `height` elements occupies: whole blocks.
    pub fn tiled_size(self, width: usize, height: usize, bpe: usize) -> Option<usize> {
        let (w_log2, h_log2) = self.block_shape(bpe)?;
        Some((width.div_ceil(1 << w_log2) * height.div_ceil(1 << h_log2)) << self.block_log2())
    }
}

/// The address equation of `mode` for elements of `bpe` bytes, one entry per offset bit from bit 0 upward: the masks of
/// the x, y and slice bits whose parity the offset bit takes. The pipe/bank exclusive or of the layout is not part of it
/// (it is a constant, see [`detile_with`]). `None` for an element size or pipe count the tables do not have.
pub fn equation_masks(mode: Mode, layout: &Layout, bpe: usize) -> Option<Vec<(u16, u16, u16)>> {
    Some(mode.bits(layout, bpe)?.iter().map(|b| (b.x, b.y, b.z)).collect())
}

/// The offset bits that `v` selects, for `len` values of one coordinate.
fn table(bits: &[Bit], len: usize, pick: impl Fn(Bit) -> u16) -> Vec<u32> {
    (0..len)
        .map(|v| bits.iter().enumerate().fold(0u32, |offset, (i, &bit)| offset | (((v as u16 & pick(bit)).count_ones() & 1) << i)))
        .collect()
}

/// For each column and each row of a `width` × `height` surface, the offset bits its x and y give,
/// and what the slice and the pipe/bank exclusive or add to every offset. The pattern is evaluated on
/// the whole surface coordinates, because the "X" modes mix in position bits above the block's own.
fn offsets(mode: Mode, layout: &Layout, bpe: usize, width: usize, height: usize) -> Option<(Vec<u32>, Vec<u32>, u32)> {
    let bits = mode.bits(layout, bpe)?;
    let block_mask = (1u32 << mode.block_log2()) - 1;
    let xor = if mode.is_xor() { layout.pipe_bank_xor.checked_shl(layout.pipe_interleave_log2).unwrap_or(0) & block_mask } else { 0 };
    let slice_bits = if mode.is_xor() { bits.iter().enumerate().fold(0u32, |o, (i, b)| o | (((layout.slice as u16 & b.z).count_ones() & 1) << i)) } else { 0 };
    let constant = xor ^ slice_bits;
    Some((table(bits, width, |b| b.x), table(bits, height, |b| b.y), constant))
}

/// Rows of `width` × `height` elements of `bpe` bytes read from a surface tiled in `mode` with the
/// default layout. `None` when the element size is not one of the mode's or `tiled` is shorter than
/// the whole blocks.
pub fn detile(mode: Mode, tiled: &[u8], width: usize, height: usize, bpe: usize) -> Option<Vec<u8>> {
    detile_with(mode, &Layout::default(), tiled, width, height, bpe)
}

/// As [`detile`], with a layout (pipe count, RB+ tables, pipe/bank exclusive or, slice).
pub fn detile_with(mode: Mode, layout: &Layout, tiled: &[u8], width: usize, height: usize, bpe: usize) -> Option<Vec<u8>> {
    let (w_log2, h_log2) = mode.block_shape(bpe)?;
    if tiled.len() < mode.tiled_size(width, height, bpe)? {
        return None;
    }
    let block_log2 = mode.block_log2();
    let (along_x, along_y, constant) = offsets(mode, layout, bpe, width, height)?;
    let blocks_x = width.div_ceil(1 << w_log2);
    let mut out = vec![0u8; width * height * bpe];
    for (y, &ry) in along_y.iter().enumerate() {
        let row_base = ((y >> h_log2) * blocks_x) << block_log2;
        let fy = ry ^ constant;
        for (x, &cx) in along_x.iter().enumerate() {
            let at = row_base + (((x >> w_log2) << block_log2) | (cx ^ fy) as usize);
            let to = (y * width + x) * bpe;
            out[to..to + bpe].copy_from_slice(&tiled[at..at + bpe]);
        }
    }
    Some(out)
}

/// Where the levels of a mip chain sit in a tiled surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chain {
    /// Bytes from the start of the surface to level 0.
    pub level0: usize,
    /// Bytes of the tail block (0 when no level is in a tail).
    pub tail: usize,
    /// Bytes of the whole chain.
    pub total: usize,
}

/// The layout of `levels` levels of a `width` × `height` element surface: the smallest levels share
/// one "mip tail" block at the start, the other levels follow from the smallest to the largest, so
/// level 0 comes last. A level goes to the tail when it fits in half a block (the wider side halved)
/// and at most as many levels as a tail holds remain.
pub fn chain(mode: Mode, width: usize, height: usize, bpe: usize, levels: u32) -> Option<Chain> {
    if mode.block_log2() < 12 {
        return None; // 256 byte modes are micro tiled: no mip tail, no chain rule
    }
    let (w_log2, h_log2) = mode.block_shape(bpe)?;
    let block_log2 = mode.block_log2();
    let (tail_w, tail_h) = if block_log2.is_multiple_of(2) { (1usize << (w_log2 - 1), 1usize << h_log2) } else { (1 << w_log2, 1 << (h_log2 - 1)) };
    let max_in_tail = if block_log2 <= 11 { 1 + (1 << (block_log2 - 9)) } else { block_log2 - 4 };
    let mut sizes = Vec::new();
    let mut tail = 0;
    for i in 0..levels {
        let (w, h) = ((width >> i).max(1), (height >> i).max(1));
        if levels > 1 && w <= tail_w && h <= tail_h && levels - i <= max_in_tail {
            tail = 1 << block_log2;
            break;
        }
        sizes.push(mode.tiled_size(w, h, bpe)?);
    }
    let total = tail + sizes.iter().sum::<usize>();
    Some(Chain { level0: total - sizes.first().copied().unwrap_or(0), tail, total })
}

/// Width and height in elements of a 256 byte block, for 1, 2, 4, 8 and 16 byte elements.
const BLOCK_256: [(usize, usize); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];

/// Where level 0 sits, in elements, inside the tail block when the whole chain fits in the tail
/// (level 0 included); `None` otherwise. Addrlib gives tail level `i` the byte offset `16 << m` when
/// `m > 6`, else `m << 8`, with `m = max_in_tail - 1 - i`; that offset's even and odd bits from bit
/// 8 up are its y and x position in 256 byte blocks.
pub fn level0_in_tail(mode: Mode, width: usize, height: usize, bpe: usize) -> Option<(usize, usize)> {
    if mode.block_log2() < 12 {
        return None;
    }
    let (w_log2, h_log2) = mode.block_shape(bpe)?;
    let block_log2 = mode.block_log2();
    let (tail_w, tail_h) = if block_log2.is_multiple_of(2) { (1usize << (w_log2 - 1), 1usize << h_log2) } else { (1 << w_log2, 1 << (h_log2 - 1)) };
    if width > tail_w || height > tail_h {
        return None;
    }
    let max_in_tail = if block_log2 <= 11 { 1 + (1 << (block_log2 - 9)) } else { block_log2 - 4 };
    let m = max_in_tail - 1;
    let offset: usize = if m > 6 { 16 << m } else { (m as usize) << 8 };
    let spread = |first: u32| (0..6).fold(0usize, |v, k| v | (((offset >> (first + 2 * k)) & 1) << k));
    let (mut x, mut y) = (spread(9), spread(8));
    if block_log2 & 1 == 1 {
        (x, y) = (y, x);
        if bpe.trailing_zeros() & 1 == 1 {
            (y, x) = ((y << 1) | (x & 1), x >> 1);
        }
    }
    let (w256, h256) = BLOCK_256[bpe.trailing_zeros() as usize];
    Some((x * w256, y * h256))
}

/// Invented tiled surfaces for tests elsewhere in the crate.
/// Where one level of an array sits in a tiled surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelPlacement {
    /// Bytes from the start of the surface to slice 0 of this level (0 for a level in the tail).
    pub offset: usize,
    /// Bytes of one slice of this level (the whole tail block for a level in the tail).
    pub slice_bytes: usize,
    /// Whether the level is inside the shared mip tail block.
    pub in_tail: bool,
}

/// Where each level of `slices` array slices sits: the tail block first (once, shared by every slice),
/// then the levels that are not in the tail from the smallest to the largest, each with its slices one
/// after another. Returns the placement of every level and the total bytes. The tail rule is the one of
/// [`chain`], which stores a single slice and only needs level 0.
pub fn level_layout(mode: Mode, width: usize, height: usize, bpe: usize, levels: u32, slices: usize) -> Option<(Vec<LevelPlacement>, usize)> {
    if mode.block_log2() < 12 {
        return None; // 256 byte modes are micro tiled: no mip tail, no chain rule
    }
    let (w_log2, h_log2) = mode.block_shape(bpe)?;
    let block_log2 = mode.block_log2();
    let (tail_w, tail_h) = if block_log2.is_multiple_of(2) { (1usize << (w_log2 - 1), 1usize << h_log2) } else { (1 << w_log2, 1 << (h_log2 - 1)) };
    let max_in_tail = if block_log2 <= 11 { 1 + (1 << (block_log2 - 9)) } else { block_log2 - 4 };
    let mut sizes: Vec<Option<usize>> = Vec::new();
    let mut tail = 0;
    for i in 0..levels {
        let (lw, lh) = ((width >> i).max(1), (height >> i).max(1));
        if levels > 1 && lw <= tail_w && lh <= tail_h && levels - i <= max_in_tail {
            tail = 1usize << block_log2;
            sizes.extend(std::iter::repeat_n(None, (levels - i) as usize));
            break;
        }
        sizes.push(Some(mode.tiled_size(lw, lh, bpe)?));
    }
    let mut out = vec![LevelPlacement { offset: 0, slice_bytes: 0, in_tail: true }; levels as usize];
    let mut cursor = tail;
    for i in (0..levels as usize).rev() {
        if let Some(size) = sizes[i] {
            out[i] = LevelPlacement { offset: cursor, slice_bytes: size, in_tail: false };
            cursor += size * slices;
        } else {
            out[i] = LevelPlacement { offset: 0, slice_bytes: 1usize << block_log2, in_tail: true };
        }
    }
    Some((out, cursor))
}

#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// The inverse of `detile`.
    pub fn tile(mode: Mode, linear: &[u8], width: usize, height: usize, bpe: usize) -> Vec<u8> {
        tile_with(mode, &Layout::default(), linear, width, height, bpe)
    }

    /// The inverse of `detile_with`.
    pub fn tile_with(mode: Mode, layout: &Layout, linear: &[u8], width: usize, height: usize, bpe: usize) -> Vec<u8> {
        let (w_log2, h_log2) = mode.block_shape(bpe).unwrap();
        let (ax, ay, constant) = offsets(mode, layout, bpe, width, height).unwrap();
        let blocks_x = width.div_ceil(1 << w_log2);
        let mut out = vec![0u8; mode.tiled_size(width, height, bpe).unwrap()];
        for (y, &ry) in ay.iter().enumerate() {
            for (x, &cx) in ax.iter().enumerate() {
                let at = ((((y >> h_log2) * blocks_x) + (x >> w_log2)) << mode.block_log2()) | (cx ^ ry ^ constant) as usize;
                let from = (y * width + x) * bpe;
                out[at..at + bpe].copy_from_slice(&linear[from..from + bpe]);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::{tile, tile_with};
    use super::*;
    use crate::surface::{Resource, Surface, compute};

    const MODES: [Mode; 2] = [Mode::Sw64kD, Mode::Sw4kS];

    #[test]
    fn block_shapes_fill_the_block() {
        for (bpe, shape) in [(1, (8, 8)), (2, (8, 7)), (4, (7, 7)), (8, (7, 6)), (16, (6, 6))] {
            assert_eq!(Mode::Sw64kD.block_shape(bpe), Some(shape));
        }
        for (bpe, shape) in [(1, (6, 6)), (2, (6, 5)), (4, (5, 5)), (8, (5, 4)), (16, (4, 4))] {
            assert_eq!(Mode::Sw4kS.block_shape(bpe), Some(shape));
        }
        for mode in MODES {
            for bpe in [1usize, 2, 4, 8, 16] {
                let (w, h) = mode.block_shape(bpe).unwrap();
                assert_eq!(bpe << (w + h), 1 << mode.block_log2());
            }
            assert_eq!(mode.block_shape(3), None);
        }
    }

    #[test]
    fn every_element_lands_on_its_own_aligned_offset_inside_one_block() {
        for mode in MODES {
            for bpe in [1usize, 2, 4, 8, 16] {
                let (w_log2, h_log2) = mode.block_shape(bpe).unwrap();
                let bits = mode.bits(&Layout::default(), bpe).unwrap();
                let (ax, ay) = (table(bits, 1 << w_log2, |b| b.x), table(bits, 1 << h_log2, |b| b.y));
                let mut seen = vec![false; (1 << mode.block_log2()) / bpe];
                for &fy in &ay {
                    for &fx in &ax {
                        let offset = (fx ^ fy) as usize;
                        assert_eq!(offset % bpe, 0, "{}: {bpe} byte elements must stay aligned", mode.name());
                        assert!(!std::mem::replace(&mut seen[offset / bpe], true), "{} {bpe}: two elements share {offset:#x}", mode.name());
                    }
                }
            }
        }
    }

    #[test]
    fn the_first_elements_of_a_block_follow_the_table() {
        // SW_64K_D, 8 byte elements: x bit 0 moves by 8 bytes, y bit 0 by 16, x bit 1 by 32.
        let width = 128;
        let linear: Vec<u8> = (0..width * 64).flat_map(|i| (i as u64).to_le_bytes()).collect();
        let tiled = tile(Mode::Sw64kD, &linear, width, 64, 8);
        let at = |offset: usize| u64::from_le_bytes(tiled[offset..offset + 8].try_into().unwrap());
        assert_eq!([at(0), at(8), at(16), at(24), at(32)], [0, 1, width as u64, width as u64 + 1, 2]);
        // SW_4K_S, 8 byte elements: x bit 0 by 8, y bits 0 and 1 by 16 and 32, x bit 1 by 64.
        let tiled = tile(Mode::Sw4kS, &linear, width, 64, 8);
        let at = |offset: usize| u64::from_le_bytes(tiled[offset..offset + 8].try_into().unwrap());
        assert_eq!([at(0), at(8), at(16), at(32), at(64)], [0, 1, width as u64, 2 * width as u64, 2]);
    }

    #[test]
    fn surfaces_round_trip_in_both_modes_including_partial_blocks() {
        for mode in MODES {
            for (bpe, width, height) in [(8, 256, 192), (16, 100, 70), (4, 200, 130), (1, 300, 10), (2, 17, 300)] {
                let linear: Vec<u8> = (0..width * height * bpe).map(|i| (i * 7 + i / 251) as u8).collect();
                let tiled = tile(mode, &linear, width, height, bpe);
                assert_eq!(detile(mode, &tiled, width, height, bpe).unwrap(), linear, "{} {bpe} bytes, {width}×{height}", mode.name());
                assert_eq!(detile(mode, &tiled[..tiled.len() - 1], width, height, bpe), None);
            }
        }
    }

    #[test]
    fn a_level_0_small_enough_for_the_tail_sits_inside_it() {
        // 4 KiB block, 8 byte elements: block 32 × 16, tail 16 × 16; level 0 of a 64² BC1 (16 × 16
        // blocks) goes to offset 16 << 7 = 2048, that is x = 2 and y = 0 in 8 × 4 element units.
        assert_eq!(level0_in_tail(Mode::Sw4kS, 16, 16, 8), Some((16, 0)));
        assert_eq!(level0_in_tail(Mode::Sw4kS, 8, 8, 8), Some((16, 0)));
        assert_eq!(level0_in_tail(Mode::Sw4kS, 32, 16, 8), None, "wider than the tail");
        // 16 byte elements: block 16 × 16, tail 8 × 16.
        assert_eq!(level0_in_tail(Mode::Sw4kS, 16, 16, 16), None);
        assert_eq!(level0_in_tail(Mode::Sw4kS, 8, 8, 16), Some((8, 0)));
        // 64 KiB block, 4 byte elements: tail 64 × 128, m = 11, offset 32 KiB: x = 8, y = 0 in 8 × 8 units.
        assert_eq!(level0_in_tail(Mode::Sw64kD, 16, 16, 4), Some((64, 0)));
    }

    #[test]
    fn mip_chains_put_level_0_last() {
        // 1024² BC1 in SW_4K_S, 6 levels: levels 1 to 3 in whole blocks, 4 and 5 in one tail block.
        // These sizes match the space the build reserves for such textures.
        let c = chain(Mode::Sw4kS, 256, 256, 8, 6).unwrap();
        assert_eq!((c.level0, c.tail, c.total), (4096 + 131_072 + 32_768 + 8192, 4096, 700_416));
        // 1024² BC5 in SW_64K_D, 6 levels: 4 blocks, 1 block, then the tail.
        assert_eq!(chain(Mode::Sw64kD, 256, 256, 16, 6).unwrap(), Chain { level0: 393_216, tail: 65_536, total: 1_441_792 });
        // 1024 × 256 BC5 in SW_4K_S, 6 levels: the 16 × 4 level is too wide for the 8 × 16 tail.
        assert_eq!(chain(Mode::Sw4kS, 256, 64, 16, 6).unwrap().total, 360_448);
        // One level: nothing before level 0.
        assert_eq!(chain(Mode::Sw64kD, 256, 256, 8, 1).unwrap(), Chain { level0: 0, tail: 0, total: 524_288 });
    }

    /// Layouts that exercise every parameter of the "X" modes.
    fn layouts(mode: Mode) -> Vec<Layout> {
        let l = |pipes_log2, rb_plus, pipe_bank_xor, pipe_interleave_log2, slice| Layout { pipes_log2, rb_plus, pipe_bank_xor, pipe_interleave_log2, slice };
        if mode.is_xor() {
            vec![l(0, false, 0, 8, 0), l(2, false, 0, 8, 0), l(4, false, 3, 8, 1), l(6, false, 5, 9, 3), l(2, true, 0, 8, 0), l(4, true, 2, 9, 2), l(6, true, 7, 8, 5), l(3, false, 1, 8, 1)]
        } else {
            vec![l(0, false, 0, 8, 0), l(0, true, 0, 8, 0)]
        }
    }

    #[test]
    fn every_mode_puts_each_element_of_every_block_on_its_own_aligned_offset() {
        // Several blocks in each direction, because the "X" modes mix in position bits of the
        // neighbouring blocks: inside each block the map must still be one to one.
        for mode in Mode::ALL {
            for layout in layouts(mode) {
                for bpe in [1usize, 2, 4, 8, 16] {
                    let (w_log2, h_log2) = mode.block_shape(bpe).unwrap();
                    let (width, height) = (3usize << w_log2, 3usize << h_log2);
                    let (ax, ay, constant) = offsets(mode, &layout, bpe, width, height).unwrap();
                    for by in 0..3 {
                        for bx in 0..3 {
                            let mut seen = vec![false; (1usize << mode.block_log2()) / bpe];
                            for &ry in &ay[by << h_log2..(by + 1) << h_log2] {
                                for &cx in &ax[bx << w_log2..(bx + 1) << w_log2] {
                                    let offset = (cx ^ ry ^ constant) as usize;
                                    assert!(offset < 1 << mode.block_log2(), "{} {bpe} {layout:?}: offset {offset:#x} leaves the block", mode.name());
                                    assert_eq!(offset % bpe, 0, "{} {bpe} {layout:?}: unaligned", mode.name());
                                    assert!(!std::mem::replace(&mut seen[offset / bpe], true), "{} {bpe} {layout:?}: two elements share {offset:#x}", mode.name());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_mode_and_layout_round_trips_including_partial_blocks() {
        for mode in Mode::ALL {
            for layout in layouts(mode) {
                for (bpe, width, height) in [(8, 130, 70), (16, 100, 70), (4, 200, 130), (1, 300, 10), (2, 17, 300)] {
                    let linear: Vec<u8> = (0..width * height * bpe).map(|i| (i * 7 + i / 251) as u8).collect();
                    let tiled = tile_with(mode, &layout, &linear, width, height, bpe);
                    assert_eq!(detile_with(mode, &layout, &tiled, width, height, bpe).unwrap(), linear, "{} {bpe} bytes {width}×{height} {layout:?}", mode.name());
                    assert_eq!(detile_with(mode, &layout, &tiled[..tiled.len() - 1], width, height, bpe), None);
                }
            }
        }
    }

    #[test]
    fn the_verified_modes_still_give_their_old_offsets() {
        // The first rows of the two tables the texture databases use, written out as they were before
        // the tables were generated: SW_64K_D 1 byte elements, SW_4K_S 16 byte elements.
        let rows = |mode: Mode, bpe: usize| mode.bits(&Layout::default(), bpe).unwrap().iter().map(|b| (b.x, b.y, b.z)).collect::<Vec<_>>();
        assert_eq!(rows(Mode::Sw64kD, 1), [(1, 0, 0), (2, 0, 0), (4, 0, 0), (0, 2, 0), (0, 1, 0), (0, 4, 0), (8, 0, 0), (0, 8, 0), (0, 16, 0), (16, 0, 0), (0, 32, 0), (32, 0, 0), (0, 64, 0), (64, 0, 0), (0, 128, 0), (128, 0, 0)]);
        assert_eq!(rows(Mode::Sw4kS, 16), [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (2, 0, 0), (0, 4, 0), (4, 0, 0), (0, 8, 0), (8, 0, 0)]);
    }

    #[test]
    fn the_xor_modes_depend_on_pipes_slice_and_exclusive_or_and_the_others_do_not() {
        // Some element sizes or pipe counts of a mode may not use a given parameter (the slice bits sit
        // in the rows of the larger pipe counts), so look across all element sizes and ask for 64 pipes.
        let changes = |mode: Mode, layout: Layout| {
            [1usize, 2, 4, 8, 16].into_iter().any(|bpe| {
                let linear: Vec<u8> = (0..256 * 256 * bpe).map(|i| (i * 13 + i / 97) as u8).collect();
                tile_with(mode, &layout, &linear, 256, 256, bpe) != tile_with(mode, &Layout::default(), &linear, 256, 256, bpe)
            })
        };
        for mode in Mode::ALL {
            for layout in [Layout { pipes_log2: 4, ..Layout::default() }, Layout { pipe_bank_xor: 5, ..Layout::default() }, Layout { pipes_log2: 6, slice: 3, ..Layout::default() }] {
                assert_eq!(changes(mode, layout), mode.is_xor(), "{} {layout:?}", mode.name());
            }
        }
        // The pipe interleave only moves the exclusive or: with none it changes nothing.
        assert!(!changes(Mode::Sw64kSX, Layout { pipe_interleave_log2: 10, ..Layout::default() }));
        assert!(changes(Mode::Sw64kSX, Layout { pipe_bank_xor: 5, pipe_interleave_log2: 10, ..Layout::default() }));
    }

    #[test]
    fn micro_tiled_modes_have_no_mip_chain_and_the_others_keep_theirs() {
        for mode in [Mode::Sw256bS, Mode::Sw256bD] {
            assert_eq!(chain(mode, 64, 64, 4, 4), None);
            assert_eq!(level0_in_tail(mode, 8, 8, 4), None);
            assert_eq!(mode.tiled_size(33, 17, 4), Some(((33usize).div_ceil(8) * (17usize).div_ceil(8)) << 8));
        }
        assert_eq!(chain(Mode::Sw64kS, 256, 256, 8, 6), chain(Mode::Sw64kD, 256, 256, 8, 6));
    }

    #[test]
    fn level_layout_of_one_slice_agrees_with_the_chain() {
        for mode in MODES {
            for bpe in [1usize, 2, 4, 8, 16] {
                for (w, h) in [(1usize, 1usize), (16, 16), (33, 17), (100, 70), (256, 64), (1024, 1024)] {
                    for levels in [1u32, 3, 6, 11] {
                        let (placement, total) = level_layout(mode, w, h, bpe, levels, 1).unwrap();
                        let c = chain(mode, w, h, bpe, levels).unwrap();
                        assert_eq!(total, c.total, "{mode:?} bpe {bpe} {w}x{h} levels {levels}");
                        assert_eq!(placement[0].offset, if placement[0].in_tail { 0 } else { c.level0 });
                        assert_eq!(placement.iter().any(|p| p.in_tail), c.tail > 0);
                    }
                }
            }
        }
    }

    #[test]
    fn level_layout_shares_the_tail_and_repeats_the_other_levels_per_slice() {
        let (one, total_one) = level_layout(Mode::Sw4kS, 128, 128, 4, 8, 1).unwrap();
        let (three, total_three) = level_layout(Mode::Sw4kS, 128, 128, 4, 8, 3).unwrap();
        let tail = 1 << Mode::Sw4kS.block_log2();
        assert!(one.last().unwrap().in_tail && !one[0].in_tail);
        assert_eq!(total_three - tail, (total_one - tail) * 3, "only the levels outside the tail grow with the slices");
        assert_eq!(three[0].offset + three[0].slice_bytes * 3, total_three, "level 0 of the last slice ends the surface");
        assert_eq!(level_layout(Mode::Sw256bS, 64, 64, 4, 3, 1), None, "micro tiled modes have no chain");
    }

    // ---- Reference vectors (tests/detile_vectors.txt) ----

    /// The tiled input the vectors use: bytes that differ with position, kept fixed so the committed vectors stay comparable.
    fn vector_input(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(7) + i / 251 + 13) as u8).collect()
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn mode_id(mode: Mode) -> u32 {
        Mode::ALL.iter().position(|&m| m == mode).unwrap() as u32
    }

    fn vectors() -> String {
        let mut out = String::new();
        out.push_str("# Reference vectors for tiling.rs and surface.rs, written by the decant-texture test\n");
        out.push_str("# `reference_vectors_match_the_committed_file` (tiling.rs). Do not edit by hand: run\n");
        out.push_str("#   UPDATE_DETILE_VECTORS=1 cargo test -p decant-texture reference_vectors\n");
        out.push_str("# Modes: 0 = SW_64K_D, 1 = SW_4K_S, 2 = SW_256B_S, 3 = SW_256B_D, 4 = SW_4K_D, 5 = SW_4K_S_X, 6 = SW_4K_D_X,\n");
        out.push_str("# 7 = SW_64K_S, 8 = SW_64K_S_X, 9 = SW_64K_D_X, 10 = SW_64K_R_X, 11 = SW_64K_Z_X. Sizes in elements and bytes. `-` = none / absent.\n");
        out.push_str("# detilex: mode bpe w h pipes_log2 rb_plus pipe_bank_xor pipe_interleave_log2 slice tiled_len hash len tail full.\n");
        let dims = [(1usize, 1usize), (4, 4), (16, 16), (16, 8), (33, 17), (64, 64), (100, 70), (256, 64), (300, 10), (17, 300), (512, 512), (1024, 1024)];
        let level_counts = [1u32, 3, 6, 11];
        for mode in MODES {
            let m = mode_id(mode);
            for bpe in [1usize, 2, 3, 4, 8, 16] {
                match mode.block_shape(bpe) {
                    Some((a, b)) => out.push_str(&format!("shape {m} {bpe} {a} {b}\n")),
                    None => out.push_str(&format!("shape {m} {bpe} -\n")),
                }
                for &(w, h) in &dims {
                    match mode.tiled_size(w, h, bpe) {
                        Some(n) => out.push_str(&format!("size {m} {bpe} {w} {h} {n}\n")),
                        None => out.push_str(&format!("size {m} {bpe} {w} {h} -\n")),
                    }
                    match level0_in_tail(mode, w, h, bpe) {
                        Some((x, y)) => out.push_str(&format!("tail0 {m} {bpe} {w} {h} {x} {y}\n")),
                        None => out.push_str(&format!("tail0 {m} {bpe} {w} {h} -\n")),
                    }
                    for &levels in &level_counts {
                        match chain(mode, w, h, bpe, levels) {
                            Some(c) => out.push_str(&format!("chain {m} {bpe} {w} {h} {levels} {} {}\n", c.level0, c.total)),
                            None => out.push_str(&format!("chain {m} {bpe} {w} {h} {levels} -\n")),
                        }
                        for slices in [1usize, 3] {
                            if let Some((layout, total)) = level_layout(mode, w, h, bpe, levels, slices) {
                                let cells: Vec<String> = layout.iter().map(|p| format!("{}:{}:{}", p.offset, p.slice_bytes, u8::from(p.in_tail))).collect();
                                out.push_str(&format!("levels {m} {bpe} {w} {h} {levels} {slices} {total} {}\n", cells.join(" ")));
                            } else {
                                out.push_str(&format!("levels {m} {bpe} {w} {h} {levels} {slices} -\n"));
                            }
                        }
                    }
                }
            }
        }
        for mode in MODES {
            let m = mode_id(mode);
            for (bpe, w, h) in [(8usize, 256usize, 192usize), (16, 100, 70), (4, 200, 130), (1, 300, 10), (2, 17, 300), (4, 1, 1), (8, 129, 65), (16, 64, 64), (4, 257, 129), (2, 128, 64)] {
                let tiled = vector_input(mode.tiled_size(w, h, bpe).unwrap());
                let rows = detile(mode, &tiled, w, h, bpe).unwrap();
                let tail = &rows[rows.len().saturating_sub(16)..];
                let full = if rows.len() <= 512 { hex(&rows) } else { "-".to_string() };
                out.push_str(&format!("detile {m} {bpe} {w} {h} {} {:016x} {} {} {full}\n", tiled.len(), fnv1a64(&rows), rows.len(), hex(tail)));
                assert_eq!(detile(mode, &tiled[..tiled.len() - 1], w, h, bpe), None);
                out.push_str(&format!("detile_short {m} {bpe} {w} {h} {}\n", tiled.len() - 1));
            }
        }
        // The other modes: shapes, sizes, the mip rules and detiling with the layouts above.
        for &mode in &Mode::ALL[2..] {
            let m = mode_id(mode);
            for bpe in [1usize, 2, 4, 8, 16] {
                let (a, b) = mode.block_shape(bpe).unwrap();
                out.push_str(&format!("shape {m} {bpe} {a} {b}\n"));
                for (w, h) in [(1usize, 1usize), (16, 16), (33, 17), (100, 70), (256, 64), (300, 10), (1024, 1024)] {
                    out.push_str(&format!("size {m} {bpe} {w} {h} {}\n", mode.tiled_size(w, h, bpe).unwrap()));
                }
                for (w, h) in [(8usize, 8usize), (16, 16), (64, 64)] {
                    match level0_in_tail(mode, w, h, bpe) {
                        Some((x, y)) => out.push_str(&format!("tail0 {m} {bpe} {w} {h} {x} {y}\n")),
                        None => out.push_str(&format!("tail0 {m} {bpe} {w} {h} -\n")),
                    }
                }
                for levels in [1u32, 6] {
                    match chain(mode, 256, 256, bpe, levels) {
                        Some(c) => out.push_str(&format!("chain {m} {bpe} 256 256 {levels} {} {}\n", c.level0, c.total)),
                        None => out.push_str(&format!("chain {m} {bpe} 256 256 {levels} -\n")),
                    }
                }
            }
            for layout in layouts(mode) {
                for (bpe, w, h) in [(8usize, 130usize, 70usize), (16, 100, 70), (4, 200, 130), (1, 300, 10), (2, 17, 300), (4, 257, 129)] {
                    let tiled = vector_input(mode.tiled_size(w, h, bpe).unwrap());
                    let rows = detile_with(mode, &layout, &tiled, w, h, bpe).unwrap();
                    let tail = &rows[rows.len().saturating_sub(16)..];
                    let full = if rows.len() <= 512 { hex(&rows) } else { "-".to_string() };
                    out.push_str(&format!(
                        "detilex {m} {bpe} {w} {h} {} {} {} {} {} {} {:016x} {} {} {full}\n",
                        layout.pipes_log2,
                        u8::from(layout.rb_plus),
                        layout.pipe_bank_xor,
                        layout.pipe_interleave_log2,
                        layout.slice,
                        tiled.len(),
                        fnv1a64(&rows),
                        rows.len(),
                        hex(tail)
                    ));
                }
            }
        }
        // The surface layout calculation (surface.rs): block dimensions, pitch, mip tail, slice size, every level.
        out.push_str("# surf: mode resource(0=2D,1=3D) bpe width height slices levels | `-` or: block_w block_h block_d pitch height num_slices slice_size surf_size first_mip_in_tail chain_in_tail, then per level pitch:height:depth:offset:macro_block_offset:tail_offset:tail_x:tail_y:tail_z:in_tail.\n");
        for mode in Mode::ALL {
            for (res_id, resource) in [(0u32, Resource::Tex2d), (1, Resource::Tex3d)] {
                for bpe in [1usize, 4, 16] {
                    for (w, h) in [(32usize, 32usize), (100, 70)] {
                        for slices in [1usize, 6] {
                            for levels in [1usize, 6] {
                                let head = format!("surf {} {res_id} {bpe} {w} {h} {slices} {levels}", mode_id(mode));
                                match compute(&Surface { mode, resource, bpe, width: w, height: h, slices, levels }) {
                                    Some(i) => {
                                        let mips: Vec<String> = i
                                            .mips
                                            .iter()
                                            .map(|m| format!("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}", m.pitch, m.height, m.depth, m.offset, m.macro_block_offset, m.tail_offset, m.tail_x, m.tail_y, m.tail_z, u8::from(m.in_tail)))
                                            .collect();
                                        out.push_str(&format!(
                                            "{head} {} {} {} {} {} {} {} {} {} {} {}\n",
                                            i.block_width,
                                            i.block_height,
                                            i.block_depth,
                                            i.pitch,
                                            i.height,
                                            i.num_slices,
                                            i.slice_size,
                                            i.surf_size,
                                            i.first_mip_in_tail,
                                            u8::from(i.mip_chain_in_tail),
                                            mips.join(" ")
                                        ));
                                    }
                                    None => out.push_str(&format!("{head} -\n")),
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn reference_vectors_match_the_committed_file() {
        let text = vectors();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/detile_vectors.txt");
        if std::env::var_os("UPDATE_DETILE_VECTORS").is_some() {
            std::fs::write(&path, &text).unwrap();
            return;
        }
        let committed = std::fs::read_to_string(&path).expect("tests/detile_vectors.txt: run with UPDATE_DETILE_VECTORS=1 to write it");
        assert_eq!(committed.replace("\r\n", "\n"), text, "the committed vectors are stale: run UPDATE_DETILE_VECTORS=1 cargo test -p decant-texture reference_vectors");
    }
}
