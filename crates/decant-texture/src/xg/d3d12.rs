//! `D3D12_RESOURCE_DESC` to an XG description (`docs/CONSOLE_LIBS.md` 1.7).

use super::XgError;
use super::formats::is_depth;
use super::layout::{Dim, TextureDesc};
use super::swizzle::{BIND_DEPTH_STENCIL, BIND_RENDER_TARGET, MISC_64K_D, SwizzleQuery, compute_optimal_swizzle_mode};

/// The public `D3D12_RESOURCE_DESC` (0x38 bytes) fields XG reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3d12ResourceDesc {
    /// 1 buffer, 2 1D, 3 2D, 4 3D.
    pub dimension: u32,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub layout: u32,
    pub flags: u32,
}

impl D3d12ResourceDesc {
    /// Reads the structure from its 0x38 bytes.
    pub fn from_bytes(b: &[u8; 0x38]) -> D3d12ResourceDesc {
        let u32_at = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let u16_at = |at: usize| u16::from_le_bytes(b[at..at + 2].try_into().unwrap());
        D3d12ResourceDesc {
            dimension: u32_at(0),
            width: u64::from_le_bytes(b[0x10..0x18].try_into().unwrap()),
            height: u32_at(0x18),
            depth_or_array_size: u16_at(0x1C),
            mip_levels: u16_at(0x1E),
            format: u32_at(0x20),
            sample_count: u32_at(0x24),
            sample_quality: u32_at(0x28),
            layout: u32_at(0x2C),
            flags: u32_at(0x30),
        }
    }
}

/// The XG description of a D3D12 texture: bind and misc flags, the swizzle mode and the usage.
pub fn texture_desc_from_d3d12(d: &D3d12ResourceDesc) -> Result<TextureDesc, XgError> {
    let dim = match d.dimension {
        2 => Dim::D1,
        3 => Dim::D2,
        4 => Dim::D3,
        _ => return Err(XgError::InvalidArg),
    };
    let mut bind = 0x8u32;
    if d.flags & 0x1 != 0 {
        bind |= BIND_RENDER_TARGET;
    }
    if d.flags & 0x2 != 0 {
        bind |= BIND_DEPTH_STENCIL;
    }
    if d.flags & 0x4 != 0 {
        bind |= 0x80;
    }
    if d.flags & 0x8 != 0 {
        bind &= !0x8;
    }
    let mut misc = 0u32;
    if d.flags & 0x20 != 0 {
        misc |= 0x38_0000;
    }
    if d.flags & 0x30_0000 != 0 && bind & BIND_RENDER_TARGET != 0 {
        misc |= 0x8_0000;
    }
    if d.flags & 0x40_0000 != 0 {
        misc |= 0x40_0000;
    }
    if d.flags & 0x80_0000 != 0 {
        misc |= 0x8000;
    }
    let mode = match d.layout {
        0 | 2 => {
            if d.layout == 2 {
                misc |= MISC_64K_D;
            }
            if bind & BIND_DEPTH_STENCIL != 0 {
                24
            } else {
                compute_optimal_swizzle_mode(&SwizzleQuery {
                    dimension: d.dimension,
                    format: d.format,
                    width: d.width & 0xFFFF_FFFF,
                    height: d.height,
                    depth_or_array: u32::from(d.depth_or_array_size),
                    sample_count: d.sample_count,
                    bind_flags: bind,
                    misc_flags: misc,
                })
            }
        }
        1 => 0,
        0x100..=0x11F => {
            if dim == Dim::D2 && (0x110..=0x113).contains(&d.layout) && !is_depth(d.format) {
                misc |= MISC_64K_D;
            }
            d.layout & 0x1F
        }
        _ => return Err(XgError::InvalidArg),
    };
    Ok(TextureDesc {
        dim,
        width: d.width as u32,
        height: d.height,
        depth_or_array: u32::from(d.depth_or_array_size),
        mip_levels: u32::from(d.mip_levels),
        format: d.format,
        sample_count: d.sample_count,
        usage: if mode == 0 { 3 } else { 0 },
        bind_flags: bind,
        misc_flags: misc,
        swizzle_mode: mode,
        pitch: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(dimension: u32, layout: u32, flags: u32) -> D3d12ResourceDesc {
        D3d12ResourceDesc { dimension, width: 256, height: 256, depth_or_array_size: 1, mip_levels: 1, format: 28, sample_count: 1, sample_quality: 0, layout, flags }
    }

    #[test]
    fn flags_become_bind_and_misc() {
        let t = texture_desc_from_d3d12(&d(3, 0, 0x1 | 0x20 | 0x20_0000 | 0x80_0000 | 0x40_0000)).unwrap();
        assert_eq!(t.bind_flags, 0x28);
        assert_eq!(t.misc_flags, 0x38_0000 | 0x8_0000 | 0x8000 | 0x40_0000);
        assert_eq!(t.swizzle_mode, 27);
        assert_eq!(t.usage, 0);
        let t = texture_desc_from_d3d12(&d(3, 0, 0x8 | 0x4)).unwrap();
        assert_eq!(t.bind_flags, 0x80);
        assert_eq!(t.swizzle_mode, 27, "unordered access of 256 x 256: not small");
        assert_eq!(texture_desc_from_d3d12(&d(3, 1, 0)).unwrap().usage, 3);
    }

    #[test]
    fn layouts() {
        assert_eq!(texture_desc_from_d3d12(&d(3, 2, 0)).unwrap().swizzle_mode, 10);
        assert_eq!(texture_desc_from_d3d12(&d(3, 0x105, 0)).unwrap().swizzle_mode, 5);
        let t = texture_desc_from_d3d12(&d(3, 0x112, 0)).unwrap();
        assert_eq!((t.swizzle_mode, t.misc_flags), (18, MISC_64K_D));
        let t = texture_desc_from_d3d12(&D3d12ResourceDesc { format: 45, ..d(3, 0x112, 0) }).unwrap();
        assert_eq!(t.misc_flags, 0, "a depth format gets no 64 KiB D flag");
        assert_eq!(texture_desc_from_d3d12(&d(3, 0x102, 0)).unwrap().swizzle_mode, 2);
        assert_eq!(texture_desc_from_d3d12(&d(3, 3, 0)), Err(XgError::InvalidArg));
        assert_eq!(texture_desc_from_d3d12(&d(3, 0x120, 0)), Err(XgError::InvalidArg));
        assert_eq!(texture_desc_from_d3d12(&d(3, 0x1000, 0)), Err(XgError::InvalidArg));
        for dim in [0, 1, 5, 99] {
            assert_eq!(texture_desc_from_d3d12(&d(dim, 0, 0)), Err(XgError::InvalidArg));
        }
        assert_eq!(texture_desc_from_d3d12(&d(3, 0, 0x2)).unwrap().swizzle_mode, 24);
    }

    #[test]
    fn the_structure_is_read_from_its_bytes() {
        let mut b = [0u8; 0x38];
        b[0..4].copy_from_slice(&3u32.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&0x1_0000_0100u64.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&7u32.to_le_bytes());
        b[0x1C..0x1E].copy_from_slice(&6u16.to_le_bytes());
        b[0x1E..0x20].copy_from_slice(&5u16.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&4u32.to_le_bytes());
        b[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        b[0x30..0x34].copy_from_slice(&1u32.to_le_bytes());
        let r = D3d12ResourceDesc::from_bytes(&b);
        assert_eq!((r.dimension, r.width, r.height, r.depth_or_array_size, r.mip_levels, r.format, r.sample_count, r.layout, r.flags), (3, 0x1_0000_0100, 7, 6, 5, 28, 4, 0x105, 1));
        assert_eq!(texture_desc_from_d3d12(&r).unwrap().width, 0x100, "the width is truncated to 32 bits");
    }
}
