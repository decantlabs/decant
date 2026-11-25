//! The console GPU image descriptor (GFX10 `SQ_IMG_RSRC_WORD0` to `WORD7`) as `xg_xs` reads it (`docs/CONSOLE_LIBS.md`, section 1).

use super::XgError;
use super::layout::{Dim, TextureDesc};

fn bits(word: u32, lo: u32, len: u32) -> u32 {
    (word >> lo) & ((1u32 << len) - 1)
}

/// Whether one of the four destination swizzle selectors (3 bits each, in bits 0 to 11 of word 3) is 2 or 3.
fn bad_selector(word3: u32) -> bool {
    (0..4).any(|i| matches!(bits(word3, i * 3, 3), 2 | 3))
}

/// `XGValidateDescriptor`: `Ok` for a valid descriptor, `Err(Fail)` for an invalid one, `Err(InvalidArg)` for a bad call.
/// `words` is `None` for a null pointer.
pub fn validate_descriptor(size_dwords: u32, words: Option<&[u32]>, flags: u32) -> Result<(), XgError> {
    let words = match words {
        Some(w) if (size_dwords == 4 || size_dwords == 8) && flags == 0 && w.len() >= size_dwords as usize => w,
        _ => return Err(XgError::InvalidArg),
    };
    if words[..4].iter().all(|&w| w == 0) {
        return Ok(());
    }
    let (w1, w2, w3) = (words[1], words[2], words[3]);
    let invalid = Err(XgError::Fail);
    if w3 < 0x4000_0000 {
        // A buffer descriptor.
        let format = bits(w3, 12, 7);
        if bad_selector(w3) || format == 0 || format > 0x4E || bits(w1, 0, 16) > 0xFF {
            return invalid;
        }
        return Ok(());
    }
    let ty = bits(w3, 28, 4);
    if ty < 8 {
        return invalid;
    }
    let format = bits(w1, 20, 9);
    let format_ok = (0x80..=0xBF).contains(&format) || (0x01..=0x4E).contains(&format) || (0x109..=0x120).contains(&format);
    if bits(w1, 0, 8) != 0 || !format_ok || bad_selector(w3) || bits(w3, 16, 4) < bits(w3, 12, 4) {
        return invalid;
    }
    if size_dwords == 4 {
        return if ty == 8 || ty == 9 { Ok(()) } else { invalid };
    }
    let (w4, w7) = (words[4], words[7]);
    let (depth, base_array) = (bits(w4, 0, 16), bits(w4, 16, 16));
    let width = (bits(w1, 30, 2) | (bits(w2, 0, 14) << 2)) + 1;
    let height = bits(w2, 14, 16) + 1;
    if bits(w7, 24, 8) != 0 || base_array > depth || depth + 1 > 0x2000 || width > 0x4000 || height > 0x4000 || !(8..=15).contains(&ty) {
        return invalid;
    }
    Ok(())
}

/// The fields of an image descriptor the layout needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescriptor {
    /// The GFX10 `IMG_FORMAT` field.
    pub image_format: u32,
    pub width: u32,
    pub height: u32,
    /// Depth of a volume, or array slices (cube faces included) for the array types, else 1.
    pub depth_or_array: u32,
    pub swizzle_mode: u32,
    pub dim: Dim,
    pub sample_count: u32,
    /// Mip levels the descriptor addresses.
    pub mip_levels: u32,
    /// Mip levels allocated.
    pub allocated_mips: u32,
}

/// Reads an 8 dword image descriptor.
pub fn parse_image_descriptor(w: &[u32; 8]) -> Result<ImageDescriptor, XgError> {
    let ty = bits(w[3], 28, 4);
    let image_format = bits(w[1], 20, 9);
    if w[3] >> 31 == 0 || image_format == 0 {
        return Err(XgError::InvalidArg);
    }
    let dim = match ty {
        9 | 11 | 13 | 14 | 15 => Dim::D2,
        8 | 12 => Dim::D1,
        _ => Dim::D3,
    };
    let last_level = bits(w[3], 16, 4);
    let msaa = ty == 14 || ty == 15;
    let (sample_count, mip_levels, allocated_mips) = if msaa {
        (1 << last_level, 1, if last_level != 0 { 1 } else { bits(w[5], 4, 4) + 1 })
    } else {
        (1, last_level + 1, bits(w[5], 4, 4) + 1)
    };
    if allocated_mips > 15 {
        return Err(XgError::InvalidArg);
    }
    let depth_field = bits(w[4], 0, 16) + 1;
    let arrayed = matches!(ty, 11 | 12 | 13 | 15);
    Ok(ImageDescriptor {
        image_format,
        width: (bits(w[1], 30, 2) | (bits(w[2], 0, 14) << 2)) + 1,
        height: bits(w[2], 14, 16) + 1,
        depth_or_array: if dim == Dim::D3 || arrayed { depth_field } else { 1 },
        swizzle_mode: bits(w[3], 20, 5),
        dim,
        sample_count,
        mip_levels,
        allocated_mips,
    })
}

/// The element of a GFX10 `IMG_FORMAT`: bytes per element and the texels along a side of the element.
///
/// The format table (Mesa `gfx10_format_table`) is not in the specification and not in our repository, so every format is an
/// open point and gives `None`. A format with more than one element per texel or a block shape other than 1 x 1 or 4 x 4 would
/// give `None` too.
pub fn image_format_element(_image_format: u32) -> Option<(u32, u32)> {
    None
}

/// The representative `DXGI_FORMAT` of an element: 1 byte 61, 2 bytes 56, 4 bytes 28, 8 bytes 10, 16 bytes 2, block of 8 bytes
/// 71 and block of 16 bytes 77.
pub fn representative_format(bpe: u32, block: u32) -> Option<u32> {
    match (bpe, block) {
        (1, 1) => Some(61),
        (2, 1) => Some(56),
        (4, 1) => Some(28),
        (8, 1) => Some(10),
        (16, 1) => Some(2),
        (8, 4) => Some(71),
        (16, 4) => Some(77),
        _ => None,
    }
}

/// The description XG builds from a descriptor and the element of its format. The extra levels the descriptor addresses
/// beyond the allocated ones (placeholder records) are not modelled: `NotImpl`.
pub fn texture_desc_from_descriptor(d: &ImageDescriptor, bpe: u32, block: u32) -> Result<TextureDesc, XgError> {
    let format = representative_format(bpe, block).ok_or(XgError::NotImpl)?;
    if d.mip_levels != d.allocated_mips {
        return Err(XgError::NotImpl);
    }
    let mode = d.swizzle_mode;
    Ok(TextureDesc {
        dim: d.dim,
        width: d.width,
        height: if d.dim == Dim::D1 { 1 } else { d.height },
        depth_or_array: d.depth_or_array,
        mip_levels: d.allocated_mips,
        format,
        sample_count: d.sample_count,
        usage: if mode == 0 { 3 } else { 0 },
        bind_flags: 0x8,
        misc_flags: if matches!(mode, 10 | 16..=19) { 0x800_0000 } else { 0 },
        swizzle_mode: mode,
        pitch: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2D image descriptor: format 0x80, 256 x 128, mode 27, 3 levels.
    fn image() -> [u32; 8] {
        let width = 255u32;
        let height = 127u32;
        [0, (0x80 << 20) | ((width & 3) << 30), (width >> 2) | (height << 14), (1 << 31) | (9 << 28) | (27 << 20) | (2 << 16), 0, 2 << 4, 0, 0]
    }

    #[test]
    fn validation_arguments() {
        let w = image();
        assert_eq!(validate_descriptor(8, Some(&w), 0), Ok(()));
        assert_eq!(validate_descriptor(8, None, 0), Err(XgError::InvalidArg));
        assert_eq!(validate_descriptor(5, Some(&w), 0), Err(XgError::InvalidArg));
        assert_eq!(validate_descriptor(8, Some(&w), 1), Err(XgError::InvalidArg));
        assert_eq!(validate_descriptor(4, Some(&[0, 0, 0, 0]), 0), Ok(()));
    }

    #[test]
    fn invalid_image_descriptors() {
        let ok = image();
        let with = |f: &dyn Fn(&mut [u32; 8])| {
            let mut w = ok;
            f(&mut w);
            validate_descriptor(8, Some(&w), 0)
        };
        assert_eq!(with(&|w| w[1] |= 1), Err(XgError::Fail), "low base address bits");
        assert_eq!(with(&|w| w[1] = (w[1] & 0xFFF) | (0x50 << 20)), Err(XgError::Fail), "format outside the ranges");
        assert_eq!(with(&|w| w[3] |= 2), Err(XgError::Fail), "selector 2");
        assert_eq!(with(&|w| w[3] = (w[3] & !(0xF << 16)) | (1 << 12)), Err(XgError::Fail), "last level below base level");
        assert_eq!(with(&|w| w[7] = 1 << 24), Err(XgError::Fail));
        assert_eq!(with(&|w| w[4] = 3 << 16), Err(XgError::Fail), "base array beyond depth");
        assert_eq!(with(&|w| w[2] = 0x3FFF | (0x3FFF << 14)), Err(XgError::Fail), "wider than 0x4000");
        assert_eq!(with(&|w| w[3] = (w[3] & 0x0FFF_FFFF) | (7 << 28)), Err(XgError::Fail), "type below 8");
        assert_eq!(validate_descriptor(4, Some(&[0, 0x80 << 20, 0, (1 << 31) | (10 << 28)]), 0), Err(XgError::Fail), "4 dwords: 1D or 2D only");
    }

    #[test]
    fn buffer_descriptors() {
        let buf = |w1: u32, w3: u32| validate_descriptor(4, Some(&[0, w1, 0, w3]), 0);
        assert_eq!(buf(16, 0x1000 * 5), Ok(()));
        assert_eq!(buf(16, 0), Err(XgError::Fail), "format 0");
        assert_eq!(buf(16, 0x1000 * 0x4F), Err(XgError::Fail), "format above 0x4E");
        assert_eq!(buf(0x100, 0x1000 * 5), Err(XgError::Fail), "stride above 0xFF");
        assert_eq!(buf(16, (0x1000 * 5) | 2), Err(XgError::Fail), "selector 2");
    }

    #[test]
    fn parsing_and_the_layout() {
        let d = parse_image_descriptor(&image()).unwrap();
        assert_eq!((d.image_format, d.width, d.height, d.depth_or_array, d.swizzle_mode, d.dim, d.mip_levels, d.allocated_mips), (0x80, 256, 128, 1, 27, Dim::D2, 3, 3));
        let t = texture_desc_from_descriptor(&d, 4, 1).unwrap();
        assert_eq!((t.format, t.swizzle_mode, t.usage, t.mip_levels), (28, 27, 0, 3));
        assert!(super::super::layout::compute_layout(&t, false).is_ok());
        assert_eq!(texture_desc_from_descriptor(&d, 3, 1), Err(XgError::NotImpl));
        assert_eq!(texture_desc_from_descriptor(&ImageDescriptor { allocated_mips: 4, ..d }, 4, 1), Err(XgError::NotImpl));
        let mut w = image();
        w[3] &= 0x7FFF_FFFF;
        assert_eq!(parse_image_descriptor(&w), Err(XgError::InvalidArg));
        let mut w = image();
        w[1] &= 0x000F_FFFF;
        assert_eq!(parse_image_descriptor(&w), Err(XgError::InvalidArg));
        let mut w = image();
        w[5] = 15 << 4;
        assert_eq!(parse_image_descriptor(&w), Err(XgError::InvalidArg), "16 allocated mips");
        let mut w = image();
        w[3] = (w[3] & 0x0FFF_FFFF & !(0xF << 16)) | (14 << 28) | (3 << 16) | (1 << 31);
        let msaa = parse_image_descriptor(&w).unwrap();
        assert_eq!((msaa.sample_count, msaa.mip_levels, msaa.allocated_mips), (8, 1, 1));
        assert_eq!(texture_desc_from_descriptor(&ImageDescriptor { swizzle_mode: 10, ..d }, 4, 1).unwrap().misc_flags, 0x800_0000);
        assert_eq!(image_format_element(0x80), None);
    }
}
