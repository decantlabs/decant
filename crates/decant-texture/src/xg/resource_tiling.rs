//! The 64 KiB tiling of a texture (slot 10 of the texture computer, `docs/CONSOLE_LIBS.md` 1.8): what D3D12's
//! `GetResourceTiling` reports, over the blocks of a 64 KiB swizzle mode. The mip tail is the packed part: the first block
//! of each array slice. The standard levels are listed slice after slice. With a tail the console library lists two
//! slices at most and zeroes every entry after slice 0.

use super::XgError;
use super::layout::{Dim, TextureLayout};

/// Bytes of one tile.
pub const TILE_BYTES: u64 = 1 << 16;

/// `D3D12_PACKED_MIP_INFO`, 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedMipInfo {
    pub standard_mips: u8,
    pub packed_mips: u8,
    pub tiles_for_packed_mips: u32,
    pub start_tile: u32,
}

/// `D3D12_SUBRESOURCE_TILING`, 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubresourceTiling {
    pub width_in_tiles: u32,
    pub height_in_tiles: u16,
    pub depth_in_tiles: u16,
    pub start_tile: u32,
}

const _: () = assert!(size_of::<PackedMipInfo>() == 12);
const _: () = assert!(size_of::<SubresourceTiling>() == 12);

/// Everything the slot reports, with the full list of subresource tilings. The caller picks the range it was asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceTiling {
    pub total_tiles: u32,
    pub packed: PackedMipInfo,
    /// The tile shape in texels (width, height, depth).
    pub tile_shape: (u32, u32, u32),
    pub subresources: Vec<SubresourceTiling>,
}

impl TextureLayout {
    /// The 64 KiB tiling of plane 0. `Fail` when the blocks are not 64 KiB (linear, 256 B, 4 KiB and the larger modes),
    /// `NotImpl` for thick 3D modes.
    pub fn resource_tiling(&self) -> Result<ResourceTiling, XgError> {
        if self.block_bits() != 16 {
            return Err(XgError::Fail);
        }
        if self.is_thick() {
            return Err(XgError::NotImpl);
        }
        let plane = self.plane0();
        let mips = plane.levels.len() as u32;
        let standard = plane.first_mip_in_tail.min(mips);
        let tail = standard < mips;
        // The tail is the first block of a slice, the standard levels follow it.
        let tail_tiles = match standard {
            _ if !tail => 0,
            0 => self.slice_size() / TILE_BYTES,
            s => plane.levels[s as usize - 1].offset_bytes / TILE_BYTES,
        };
        let (bw, bh) = self.block_shape();
        let texels = self.texels_per_element();
        // A 3D subresource is one level with all its depth slices, a 2D one is one level of one array slice.
        let slices = match self.dim() {
            Dim::D3 => 1,
            _ if tail => self.array_or_depth().min(2),
            _ => self.array_or_depth(),
        };
        let mut subresources = Vec::with_capacity((standard * slices) as usize);
        for slice in 0..slices {
            for level in &plane.levels[..standard as usize] {
                subresources.push(if tail && slice > 0 {
                    SubresourceTiling::default()
                } else {
                    SubresourceTiling {
                        width_in_tiles: level.padded_width / bw,
                        height_in_tiles: (level.padded_height / bh) as u16,
                        depth_in_tiles: if self.dim() == Dim::D3 { level.depth_or_array as u16 } else { 1 },
                        start_tile: ((u64::from(slice) * self.slice_size() + level.offset_bytes) / TILE_BYTES) as u32,
                    }
                });
            }
        }
        Ok(ResourceTiling {
            total_tiles: (plane.size_bytes / TILE_BYTES) as u32,
            packed: PackedMipInfo { standard_mips: standard as u8, packed_mips: (mips - standard) as u8, tiles_for_packed_mips: tail_tiles as u32, start_tile: 0 },
            tile_shape: (bw * texels, bh * texels, self.block_depth()),
            subresources,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xg::{TextureDesc, compute_layout};

    fn tex(w: u32, h: u32, a: u32, n: u32, format: u32, mode: u32) -> TextureLayout {
        compute_layout(&TextureDesc { dim: Dim::D2, width: w, height: h, depth_or_array: a, mip_levels: n, format, sample_count: 1, usage: 0, bind_flags: 8, misc_flags: 0, swizzle_mode: mode, pitch: 0 }, false).unwrap()
    }

    fn sub(w: u32, h: u16, start: u32) -> SubresourceTiling {
        SubresourceTiling { width_in_tiles: w, height_in_tiles: h, depth_in_tiles: 1, start_tile: start }
    }

    #[test]
    fn a_full_chain_lists_the_standard_levels_after_the_tail_tile() {
        // 256 x 256 of 4 byte elements in 64 KiB blocks (128 x 128 elements). Level 2 (64 x 64) is the first to fit the tail.
        // Tail at tile 0, level 1 (one tile) at tile 1, level 0 (2 x 2 tiles) at tile 2.
        let t = tex(256, 256, 1, 0, 28, 9).resource_tiling().unwrap();
        assert_eq!(t.packed, PackedMipInfo { standard_mips: 2, packed_mips: 7, tiles_for_packed_mips: 1, start_tile: 0 });
        assert_eq!((t.total_tiles, t.tile_shape), (6, (128, 128, 1)));
        assert_eq!(t.subresources, vec![sub(2, 2, 2), sub(1, 1, 1)]);
    }

    #[test]
    fn without_a_tail_every_slice_is_listed() {
        // Vector G of 1.13 without its tail: 512 x 512 of 8 byte elements, 3 mips, an array of 6.
        let l = tex(512, 512, 6, 3, 10, 9);
        let t = l.resource_tiling().unwrap();
        assert_eq!(t.packed.packed_mips, 0);
        assert_eq!(t.packed.tiles_for_packed_mips, 0);
        assert_eq!(t.subresources.len(), 18);
        let per_slice = (l.slice_size() / TILE_BYTES) as u32;
        assert_eq!(t.subresources[3].start_tile, per_slice + t.subresources[0].start_tile);
        assert_eq!(t.total_tiles, per_slice * 6);
    }

    #[test]
    fn with_a_tail_two_slices_are_listed_and_the_second_is_zero() {
        // Vector G: cube of 512 x 512, 8 byte elements, full chain, first tail level 3.
        let t = tex(512, 512, 6, 0, 10, 9).resource_tiling().unwrap();
        assert_eq!((t.packed.standard_mips, t.packed.packed_mips, t.packed.tiles_for_packed_mips), (3, 7, 1));
        assert_eq!(t.tile_shape, (128, 64, 1));
        // Level offsets 0xB0000, 0x30000, 0x10000: 4 x 8, 2 x 4 and 1 x 2 tiles.
        assert_eq!(t.subresources[..3], [sub(4, 8, 11), sub(2, 4, 3), sub(1, 2, 1)]);
        assert!(t.subresources[3..].iter().all(|s| *s == SubresourceTiling::default()));
        assert_eq!(t.subresources.len(), 6);
    }

    #[test]
    fn block_compressed_tiles_count_texels() {
        // BC1 (8 bytes per 4 x 4 block): 128 x 64 blocks per tile, 512 x 256 texels.
        let t = tex(1024, 1024, 1, 1, 71, 9).resource_tiling().unwrap();
        assert_eq!(t.tile_shape, (512, 256, 1));
        assert_eq!(t.subresources, vec![sub(2, 4, 0)]);
    }

    #[test]
    fn a_whole_chain_in_the_tail_has_no_standard_level() {
        let t = tex(64, 64, 1, 0, 28, 9).resource_tiling().unwrap();
        assert_eq!(t.packed, PackedMipInfo { standard_mips: 0, packed_mips: 7, tiles_for_packed_mips: 1, start_tile: 0 });
        assert!(t.subresources.is_empty());
    }

    #[test]
    fn other_block_sizes_fail() {
        assert_eq!(tex(64, 64, 1, 1, 28, 5).resource_tiling(), Err(XgError::Fail));
        let linear = compute_layout(&TextureDesc { dim: Dim::D2, width: 64, height: 64, depth_or_array: 1, mip_levels: 1, format: 28, sample_count: 1, usage: 3, bind_flags: 8, misc_flags: 0, swizzle_mode: 0, pitch: 0 }, false).unwrap();
        assert_eq!(linear.resource_tiling(), Err(XgError::Fail));
        let volume = compute_layout(&TextureDesc { dim: Dim::D3, width: 64, height: 64, depth_or_array: 64, mip_levels: 1, format: 28, sample_count: 1, usage: 0, bind_flags: 8, misc_flags: 0, swizzle_mode: 9, pitch: 0 }, false).unwrap();
        assert_eq!(volume.resource_tiling(), Err(XgError::NotImpl));
    }
}
