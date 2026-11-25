// The layout rules in this file are those of AMD addrlib (Mesa `src/amd/addrlib`), `Gfx10Lib::ComputeSurfaceInfoMicroTiled`,
// `ComputeSurfaceInfoMacroTiled`, `GetMaxNumMipsInTail`, `Lib::GetMipTailDim`, `ComputeThinBlockDimension` and
// `ComputeThickBlockDimension`, written again here from their description (no addrlib code is reused). The library is MIT
// licensed, see the top of `tiling.rs`.

//! The size and place of every part of a GFX10 tiled surface: the pitch and height of each mip level, the block
//! dimensions, where the mip tail sits and what it holds, how slices are padded and where each (slice, mip) starts.
//!
//! This is the calculation the console's `xg_xs` library does when a title asks for a surface's layout; the offsets
//! here are what [`crate::tiling::detile_with`] needs to read one level of one slice. All sizes are in elements (one
//! texel, or one 4 × 4 block for the BC formats) and bytes. Only the layout is computed here; nothing in it is
//! verified against the game's data except for the single-slice 2D chains that `tiling::chain` covers.

use crate::tiling::Mode;

/// What kind of surface: a 2D image (an array or a cube is a 2D image with several slices) or a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resource {
    Tex2d,
    Tex3d,
}

/// The input of the calculation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    pub mode: Mode,
    pub resource: Resource,
    /// Bytes per element: 1, 2, 4, 8 or 16.
    pub bpe: usize,
    pub width: usize,
    pub height: usize,
    /// Array slices (cube faces count as slices), or the depth of a volume.
    pub slices: usize,
    /// Mip levels, at least 1.
    pub levels: usize,
}

/// One mip level, as addrlib describes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mip {
    pub pitch: usize,
    pub height: usize,
    pub depth: usize,
    /// Byte offset of the level inside one slice (inside the whole surface for a volume).
    pub offset: u64,
    /// Byte offset of the level's first block inside one slice's run of blocks.
    pub macro_block_offset: u64,
    /// For a level in the tail, its byte offset inside the tail block; 0 otherwise.
    pub tail_offset: u32,
    /// For a level in the tail, its position inside the tail block, in elements.
    pub tail_x: usize,
    pub tail_y: usize,
    pub tail_z: usize,
    pub in_tail: bool,
}

/// The output of the calculation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    pub block_width: usize,
    pub block_height: usize,
    pub block_depth: usize,
    /// Width and height of level 0 rounded up to whole blocks.
    pub pitch: usize,
    pub height: usize,
    /// Slices (or depth) rounded up to the block depth.
    pub num_slices: usize,
    /// Bytes of one slice's whole chain: the tail block (once) and every level outside the tail.
    pub slice_size: u64,
    /// Bytes of the whole surface.
    pub surf_size: u64,
    /// The first level in the tail; `levels` when none is.
    pub first_mip_in_tail: usize,
    pub mip_chain_in_tail: bool,
    pub mips: Vec<Mip>,
    thin: bool,
}

impl Info {
    /// Byte offset of the first block of `mip` in `slice`: slices follow one another, each holding its own whole chain
    /// (the tail block first, then the levels from the smallest to the largest). For a volume the slice is the block of
    /// depth slices it falls in.
    pub fn level_start(&self, slice: usize, mip: usize) -> Option<u64> {
        let level = self.mips.get(mip)?;
        if slice >= self.num_slices {
            return None;
        }
        let (slice_size, id) = if self.thin { (self.slice_size, slice) } else { (self.slice_size * self.block_depth as u64, slice / self.block_depth) };
        Some(slice_size * id as u64 + level.macro_block_offset)
    }
}

fn align(value: usize, to: usize) -> usize {
    value.div_ceil(to) * to
}

/// 256 byte block shapes in elements for 1, 2, 4, 8 and 16 byte elements, thin and thick.
const BLOCK_256_2D: [(usize, usize); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];
const BLOCK_256_3D: [(usize, usize, usize); 5] = [(8, 4, 8), (4, 4, 8), (4, 4, 4), (4, 2, 4), (2, 2, 4)];
/// 1 KiB thick block shapes, which the thick blocks of larger sizes amplify.
const BLOCK_1K_3D: [(usize, usize, usize); 5] = [(16, 8, 8), (8, 8, 8), (8, 8, 4), (8, 4, 4), (4, 4, 4)];

fn thick(s: &Surface) -> bool {
    s.resource == Resource::Tex3d && (s.mode.is_standard() || s.mode.is_display())
}

/// The block shape in elements: (width, height, depth).
fn block_dims(s: &Surface) -> (usize, usize, usize) {
    let log2_block = s.mode.block_log2();
    let element = s.bpe.trailing_zeros() as usize;
    if thick(s) {
        let in_kib = (log2_block - 10) as usize;
        let (average, rest) = (in_kib / 3, in_kib % 3);
        let (w, h, d) = BLOCK_1K_3D[element];
        (w << average, h << (average + rest / 2), d << (average + usize::from(rest != 0)))
    } else {
        let elements_log2 = log2_block as usize - element;
        let width_log2 = elements_log2.div_ceil(2);
        (1 << width_log2, 1 << (elements_log2 - width_log2), 1)
    }
}

/// How many levels the tail holds at most.
fn max_in_tail(log2_block: u32, thin: bool) -> u32 {
    let effective = if thin { log2_block } else { log2_block - (log2_block - 8) / 3 };
    if effective <= 11 { 1 + (1 << (effective - 9)) } else { effective - 4 }
}

fn mip_size(dim: usize, level: usize) -> usize {
    (dim >> level).max(1)
}

/// The layout of a surface. `None` for an element size, mode or resource the tables do not have (3D needs a block of at
/// least 4 KiB), or a zero dimension.
pub fn compute(s: &Surface) -> Option<Info> {
    if !matches!(s.bpe, 1 | 2 | 4 | 8 | 16) || s.width == 0 || s.height == 0 || s.slices == 0 || s.levels == 0 || s.levels > 16 {
        return None;
    }
    if s.resource == Resource::Tex3d && s.mode.block_log2() < 12 {
        return None;
    }
    let log2_block = s.mode.block_log2();
    let block_size = 1u64 << log2_block;
    let (bw, bh, bd) = block_dims(s);
    let is_thin = !thick(s);
    let element = s.bpe.trailing_zeros() as usize;
    let tex3d = s.resource == Resource::Tex3d;
    let pitch = align(s.width, bw);
    let height = align(s.height, bh);
    let num_slices = align(s.slices, bd);
    let mut info = Info {
        block_width: bw,
        block_height: bh,
        block_depth: bd,
        pitch,
        height,
        num_slices,
        slice_size: 0,
        surf_size: 0,
        first_mip_in_tail: s.levels,
        mip_chain_in_tail: false,
        mips: vec![Mip::default(); s.levels],
        thin: is_thin,
    };
    let depth_of_mip = |_level: usize| if tex3d { num_slices } else { 1 };

    if log2_block == 8 {
        // Micro tiled: no tail; the levels follow from the smallest to the largest.
        let mut run = 0u64;
        for i in (0..s.levels).rev() {
            let (w, h) = (align(mip_size(s.width, i), bw), align(mip_size(s.height, i), bh));
            info.mips[i] = Mip { pitch: w, height: h, depth: 1, offset: run, macro_block_offset: run, ..Mip::default() };
            run += (w * h * s.bpe) as u64;
        }
        if s.levels == 1 {
            run = (pitch * height * s.bpe) as u64;
        }
        info.slice_size = run;
        info.surf_size = run * num_slices as u64;
        return Some(info);
    }

    if s.levels == 1 {
        info.slice_size = (pitch * height * s.bpe) as u64;
        info.surf_size = info.slice_size * num_slices as u64;
        info.mips[0] = Mip { pitch, height, depth: depth_of_mip(0), ..Mip::default() };
        return Some(info);
    }

    // The largest level that still goes in the tail: the block with its width halved (thin) or one of its sides halved (thick).
    let (mut tail_w, mut tail_h) = (bw, bh);
    if is_thin {
        tail_w >>= 1;
    } else {
        match log2_block % 3 {
            0 => tail_h >>= 1,
            1 => tail_w >>= 1,
            _ => {} // the depth is halved: it does not enter the width and height test
        }
    }
    let mip0_depth = if is_thin { 1 } else { s.slices };
    let max_tail = max_in_tail(log2_block, is_thin) as usize;
    let mut first = s.levels;
    let mut chain = 0u64;
    let mut size = vec![0u64; s.levels];
    let mut slice_size = vec![0u64; s.levels];
    for i in 0..s.levels {
        let (w, h, d) = (mip_size(s.width, i), mip_size(s.height, i), mip_size(mip0_depth, i));
        if w <= tail_w && h <= tail_h && s.levels - i <= max_tail {
            first = i;
            chain += block_size / bd as u64;
            break;
        }
        let (p, hh, dd) = (align(w, bw), align(h, bh), align(d, bd));
        let one = (p * hh * s.bpe) as u64;
        size[i] = one * dd as u64;
        slice_size[i] = one * bd as u64;
        chain += one;
        info.mips[i].pitch = p;
        info.mips[i].height = hh;
        info.mips[i].depth = depth_of_mip(i);
    }
    info.slice_size = chain;
    info.surf_size = chain * num_slices as u64;
    info.mip_chain_in_tail = first == 0;
    info.first_mip_in_tail = first;

    let (mut offset, mut macro_offset, mut tail_depth) = (0u64, 0u64, 0usize);
    if first != s.levels {
        tail_depth = mip_size(mip0_depth, first);
        offset = block_size * align(tail_depth, bd) as u64 / bd as u64;
        macro_offset = block_size;
    }
    for i in (0..first).rev() {
        info.mips[i].offset = offset;
        info.mips[i].macro_block_offset = macro_offset;
        offset += size[i];
        macro_offset += slice_size[i];
    }
    let (mut p, mut h) = (tail_w, tail_h);
    let depth = if is_thin { 1 } else { align(tail_depth, BLOCK_256_3D[element].2) };
    let tail_depth = if is_thin { 1 } else { depth / BLOCK_256_3D[element].2 };
    #[allow(clippy::needless_range_loop)]
    for i in first..s.levels {
        let m = max_tail - 1 - (i - first);
        let tail_offset = if m > 6 { 16usize << m } else { m << 8 };
        let gather = |first_bit: u32| (0..6).fold(0usize, |v, k| v | (((tail_offset >> (first_bit + 2 * k)) & 1) << k));
        let (mut x, mut y) = (gather(9), gather(8));
        if log2_block & 1 == 1 {
            std::mem::swap(&mut x, &mut y);
            if element & 1 == 1 {
                y = (y << 1) | (x & 1);
                x >>= 1;
            }
        }
        let (w256, h256) = if is_thin { BLOCK_256_2D[element] } else { (BLOCK_256_3D[element].0, BLOCK_256_3D[element].1) };
        info.mips[i] = Mip {
            pitch: p,
            height: h,
            depth: depth_of_mip(i),
            offset: tail_offset as u64 * tail_depth as u64,
            macro_block_offset: 0,
            tail_offset: tail_offset as u32,
            tail_x: x * w256,
            tail_y: y * h256,
            tail_z: 0,
            in_tail: true,
        };
        p = (p >> 1).max(w256);
        h = (h >> 1).max(h256);
    }
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::{chain, level0_in_tail};

    fn surface(mode: Mode, bpe: usize, width: usize, height: usize, slices: usize, levels: usize) -> Surface {
        Surface { mode, resource: Resource::Tex2d, bpe, width, height, slices, levels }
    }

    #[test]
    fn a_single_slice_matches_the_older_chain_rule() {
        // `tiling::chain` was written first and checked against the space the build reserves for textures; the
        // general calculation must agree with it wherever both apply.
        for mode in [Mode::Sw4kS, Mode::Sw64kD, Mode::Sw4kD, Mode::Sw64kS, Mode::Sw64kSX] {
            for bpe in [1usize, 2, 4, 8, 16] {
                for (w, h) in [(1usize, 1usize), (16, 16), (64, 64), (256, 256), (300, 70), (1024, 512), (33, 129)] {
                    for levels in [1usize, 2, 3, 6, 8] {
                        let c = chain(mode, w, h, bpe, levels as u32).unwrap();
                        let i = compute(&surface(mode, bpe, w, h, 1, levels)).unwrap();
                        assert_eq!(i.slice_size as usize, c.total, "{} {bpe} {w}x{h} {levels}", mode.name());
                        if i.first_mip_in_tail != 0 {
                            // (when level 0 is itself in the tail, `chain` reports the end of the chain instead)
                            assert_eq!(i.mips[0].macro_block_offset as usize, c.level0, "{} {bpe} {w}x{h} {levels} level 0", mode.name());
                        }
                        assert_eq!(i.first_mip_in_tail == 0 && levels > 1, level0_in_tail(mode, w, h, bpe).is_some() && levels > 1, "{} {bpe} {w}x{h} levels {levels}", mode.name());
                    }
                }
            }
        }
    }

    #[test]
    fn slices_follow_one_another_each_with_its_own_chain() {
        // 128 × 128 BC7 (32 × 32 elements of 16 bytes) in 4 KiB blocks (16 × 16 elements), 6 mips, 6 slices: per slice
        // the tail block, level 1 (one block) and level 0 (four blocks) = 6 blocks, 36 for the surface.
        let i = compute(&surface(Mode::Sw4kS, 16, 32, 32, 6, 6)).unwrap();
        assert_eq!((i.block_width, i.block_height, i.block_depth), (16, 16, 1));
        assert_eq!((i.slice_size, i.surf_size), (6 * 4096, 36 * 4096));
        assert_eq!(i.first_mip_in_tail, 2);
        assert_eq!(i.mips[1].macro_block_offset, 4096);
        assert_eq!(i.mips[0].macro_block_offset, 2 * 4096);
        assert_eq!(i.level_start(3, 0), Some(3 * 6 * 4096 + 2 * 4096));
        assert_eq!(i.level_start(6, 0), None);
        // the levels in the tail share its block: level 2 is the first, 16 << 8 away from the last slot.
        assert!(i.mips[2].in_tail && i.mips[5].in_tail);
        assert!(i.mips[2].tail_offset > i.mips[5].tail_offset);
    }

    #[test]
    fn a_level_0_alone_in_the_tail_takes_one_block_per_slice() {
        // 64 × 64 BC7 (16 × 16 elements of 16 bytes), 4 KiB blocks: levels 1 to 5 fit the tail, level 0 is one block.
        let i = compute(&surface(Mode::Sw4kS, 16, 16, 16, 6, 6)).unwrap();
        assert_eq!((i.slice_size, i.surf_size, i.first_mip_in_tail), (2 * 4096, 12 * 4096, 1));
        // a 32 × 32 BC7 (8 × 8 elements): the whole chain sits in the tail, one block per slice.
        let j = compute(&surface(Mode::Sw4kS, 16, 8, 8, 6, 6)).unwrap();
        assert_eq!((j.slice_size, j.surf_size, j.first_mip_in_tail, j.mip_chain_in_tail), (4096, 6 * 4096, 0, true));
        assert_eq!((j.mips[0].tail_x, j.mips[0].tail_y), level0_in_tail(Mode::Sw4kS, 8, 8, 16).unwrap());
    }

    #[test]
    fn micro_tiled_surfaces_have_no_tail_and_put_level_0_last() {
        let i = compute(&surface(Mode::Sw256bS, 4, 40, 17, 2, 3)).unwrap();
        // 256 byte blocks of 4 byte elements: 8 × 8; levels 40 × 17 -> 40 × 24, 20 × 8 -> 24 × 8, 10 × 4 -> 16 × 8.
        assert_eq!((i.block_width, i.block_height), (8, 8));
        assert_eq!(i.first_mip_in_tail, 3);
        assert_eq!(i.mips[2].offset, 0);
        assert_eq!(i.mips[1].offset, (16 * 8 * 4) as u64);
        assert_eq!(i.mips[0].offset, (16 * 8 * 4 + 24 * 8 * 4) as u64);
        assert_eq!(i.slice_size, (16 * 8 * 4 + 24 * 8 * 4 + 40 * 24 * 4) as u64);
        assert_eq!(i.surf_size, 2 * i.slice_size);
    }

    #[test]
    fn volumes_use_thick_blocks_in_the_standard_and_display_modes() {
        // 4 KiB thick block of 16 byte elements: 4 × 8 × 8 elements.
        let v = |mode| compute(&Surface { mode, resource: Resource::Tex3d, bpe: 16, width: 32, height: 32, slices: 6, levels: 1 }).unwrap();
        let thick = v(Mode::Sw4kS);
        assert_eq!((thick.block_width, thick.block_height, thick.block_depth), (4, 8, 8));
        assert_eq!(thick.num_slices, 8);
        assert_eq!(thick.surf_size, (32 * 32 * 16 * 8) as u64);
        // The render target and depth modes are thin in 3D: a 2D block, and the depth is not padded.
        let thin = v(Mode::Sw64kRX);
        assert_eq!((thin.block_width, thin.block_height, thin.block_depth), (64, 64, 1));
        assert_eq!(thin.num_slices, 6);
        assert_eq!(compute(&Surface { mode: Mode::Sw256bS, resource: Resource::Tex3d, bpe: 4, width: 8, height: 8, slices: 4, levels: 1 }), None);
    }

    #[test]
    fn invalid_surfaces_are_refused() {
        assert_eq!(compute(&surface(Mode::Sw4kS, 3, 8, 8, 1, 1)), None);
        assert_eq!(compute(&surface(Mode::Sw4kS, 4, 0, 8, 1, 1)), None);
        assert_eq!(compute(&surface(Mode::Sw4kS, 4, 8, 8, 1, 17)), None);
    }
}
