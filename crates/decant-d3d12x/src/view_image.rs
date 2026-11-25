//! The image words of a texture view for the view shadow (`view_shadow`): dwords 2 and 3 of the row
//! of a heap slot that holds a texture SRV or UAV. A translated program that asks a texture for its
//! size where it cannot type a `GetDimensions` call reads them instead.
//!
//! - dword 2: `(width - 1) | (height - 1) << 16` of the view's most detailed mip.
//! - dword 3: the last slice in bits 0 to 12 (array size of the view - 1, depth - 1 for 3D,
//!   6 * cubes - 1 for cubes), the mip levels of the view - 1 in bits 16 to 19 (log2 of the sample
//!   count for multisample), and the image type in bits 28 to 31: 8 1D, 9 2D, 10 3D, 11 cube,
//!   12 1D array, 13 2D array, 14 2D multisample, 15 2D multisample array.
//!
//! Open fields of the view (`-1` mip levels or array size, a null description) are resolved
//! against the resource description.
//!
//! Layouts read here: `D3D12_RESOURCE_DESC` (dimension 0, width 0x10 u64, height 0x18,
//! depth or array size 0x1C u16, mip levels 0x1E u16, sample count 0x24), the texture members of
//! `D3D12_SHADER_RESOURCE_VIEW_DESC` from 0x10 and of `D3D12_UNORDERED_ACCESS_VIEW_DESC` from 0x08.

use crate::view_raw::{DESC_BYTES, ViewKind};

/// The bytes of a `D3D12_RESOURCE_DESC`.
pub const RESOURCE_BYTES: usize = 0x38;

const T1D: u32 = 8;
const T2D: u32 = 9;
const T3D: u32 = 10;
const CUBE: u32 = 11;
const T1D_ARRAY: u32 = 12;
const T2D_ARRAY: u32 = 13;
const T2D_MS: u32 = 14;
const T2D_MS_ARRAY: u32 = 15;

/// "All that are left" in a mip count, array size or 3D slice count of a view.
const ALL: u32 = u32::MAX;

fn get32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

/// The texture as its resource description has it.
struct Resource {
    dimension: u32,
    width: u32,
    height: u32,
    /// Depth of a 3D texture, array size of another.
    depth_or_array: u32,
    mips: u32,
    samples: u32,
}

impl Resource {
    fn read(d: &[u8; RESOURCE_BYTES]) -> Resource {
        let width = u32::try_from(u64::from_le_bytes(d[0x10..0x18].try_into().unwrap_or_default())).unwrap_or(u32::MAX);
        let height = get32(d, 0x18).max(1);
        let depth_or_array = u32::from(u16::from_le_bytes([d[0x1C], d[0x1D]])).max(1);
        let depth = if get32(d, 0) == 4 { depth_or_array } else { 1 };
        let declared = u32::from(u16::from_le_bytes([d[0x1E], d[0x1F]]));
        // 0 mip levels: the whole chain
        let mips = if declared == 0 { 32 - width.max(height).max(depth).max(1).leading_zeros() } else { declared };
        Resource { dimension: get32(d, 0), width: width.max(1), height, depth_or_array, mips, samples: get32(d, 0x24).max(1) }
    }

    fn array(&self) -> u32 {
        if self.dimension == 4 { 1 } else { self.depth_or_array }
    }

    fn depth_at(&self, mip: u32) -> u32 {
        if self.dimension == 4 { (self.depth_or_array >> mip.min(31)).max(1) } else { 1 }
    }
}

/// What the image words describe: type, most detailed mip, the value of bits 16 to 19 and the last
/// slice.
struct Shape {
    image: u32,
    mip: u32,
    levels_field: u32,
    last: u32,
}

/// The two image words of the texture view `view` (`None`: the resource's own view) of `kind` on the
/// resource described by `resource`. `None` for a buffer, a view of another kind or an unknown
/// dimension.
pub fn image_words(view: Option<&[u8; DESC_BYTES]>, kind: ViewKind, resource: &[u8; RESOURCE_BYTES]) -> Option<[u32; 2]> {
    let r = Resource::read(resource);
    if !(2..=4).contains(&r.dimension) {
        return None;
    }
    let shape = match view {
        Some(v) => view_shape(v, kind, &r)?,
        None => own_shape(kind, &r),
    };
    let one_d = matches!(shape.image, T1D | T1D_ARRAY);
    let width = (r.width >> shape.mip.min(31)).max(1);
    let height = if one_d { 1 } else { (r.height >> shape.mip.min(31)).max(1) };
    let size = ((width - 1) & 0xFFFF) | (((height - 1) & 0xFFFF) << 16);
    Some([size, (shape.last & 0x1FFF) | ((shape.levels_field & 0xF) << 16) | (shape.image << 28)])
}

/// The view the PC makes for a null description: the whole resource.
fn own_shape(kind: ViewKind, r: &Resource) -> Shape {
    let levels = if kind == ViewKind::ShaderResource { r.mips } else { 1 };
    let (image, last) = match (r.dimension, r.array() > 1, r.samples > 1) {
        (2, false, _) => (T1D, 0),
        (2, true, _) => (T1D_ARRAY, r.array() - 1),
        (3, false, true) => (T2D_MS, 0),
        (3, true, true) => (T2D_MS_ARRAY, r.array() - 1),
        (3, false, false) => (T2D, 0),
        (3, true, false) => (T2D_ARRAY, r.array() - 1),
        _ => (T3D, r.depth_at(0) - 1),
    };
    let levels_field = if matches!(image, T2D_MS | T2D_MS_ARRAY) { r.samples.trailing_zeros() } else { levels - 1 };
    Shape { image, mip: 0, levels_field, last }
}

fn view_shape(v: &[u8; DESC_BYTES], kind: ViewKind, r: &Resource) -> Option<Shape> {
    let dim = get32(v, 4);
    let samples = r.samples.trailing_zeros();
    // array slices of a view from its first slice and size
    let slices = |first: u32, size: u32| if size == ALL { r.array().saturating_sub(first) } else { size };
    let last = |count: u32| count.max(1) - 1;
    let shape = |image, mip, levels: u32, last| Shape { image, mip, levels_field: levels.max(1) - 1, last };
    Some(match kind {
        ViewKind::ShaderResource => {
            let mip = get32(v, 0x10);
            let levels = match get32(v, 0x14) {
                ALL => r.mips.saturating_sub(mip),
                n => n,
            };
            match dim {
                2 => shape(T1D, mip, levels, 0),
                3 => shape(T1D_ARRAY, mip, levels, last(slices(get32(v, 0x18), get32(v, 0x1C)))),
                4 => shape(T2D, mip, levels, 0),
                5 => shape(T2D_ARRAY, mip, levels, last(slices(get32(v, 0x18), get32(v, 0x1C)))),
                6 => Shape { image: T2D_MS, mip: 0, levels_field: samples, last: 0 },
                7 => Shape { image: T2D_MS_ARRAY, mip: 0, levels_field: samples, last: last(slices(get32(v, 0x10), get32(v, 0x14))) },
                8 => shape(T3D, mip, levels, last(r.depth_at(mip))),
                9 => shape(CUBE, mip, levels, 5),
                10 => shape(CUBE, mip, levels, last(6 * get32(v, 0x1C))),
                _ => return None,
            }
        }
        ViewKind::UnorderedAccess => {
            let mip = get32(v, 0x08);
            match dim {
                2 => shape(T1D, mip, 1, 0),
                3 => shape(T1D_ARRAY, mip, 1, last(slices(get32(v, 0x0C), get32(v, 0x10)))),
                4 => shape(T2D, mip, 1, 0),
                5 => shape(T2D_ARRAY, mip, 1, last(slices(get32(v, 0x0C), get32(v, 0x10)))),
                6 => Shape { image: T2D_MS, mip: 0, levels_field: samples, last: 0 },
                7 => Shape { image: T2D_MS_ARRAY, mip: 0, levels_field: samples, last: last(slices(get32(v, 0x08), get32(v, 0x0C))) },
                8 => {
                    let depth = match get32(v, 0x10) {
                        ALL => r.depth_at(mip).saturating_sub(get32(v, 0x0C)),
                        n => n,
                    };
                    shape(T3D, mip, 1, last(depth))
                }
                _ => return None,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{RESOURCE_BYTES, image_words};
    use crate::view_raw::{DESC_BYTES, ViewKind};

    fn resource(dimension: u32, width: u64, height: u32, depth_or_array: u16, mips: u16, samples: u32) -> [u8; RESOURCE_BYTES] {
        let mut d = [0u8; RESOURCE_BYTES];
        d[0..4].copy_from_slice(&dimension.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&width.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&depth_or_array.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&samples.to_le_bytes());
        d
    }

    fn view(dimension: u32, words: &[(usize, u32)]) -> [u8; DESC_BYTES] {
        let mut d = [0u8; DESC_BYTES];
        d[4..8].copy_from_slice(&dimension.to_le_bytes());
        for &(at, w) in words {
            d[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        d
    }

    fn words(w: u32, h: u32, last: u32, levels: u32, image: u32) -> [u32; 2] {
        [(w - 1) | ((h - 1) << 16), last | (levels << 16) | (image << 28)]
    }

    #[test]
    fn shader_resource_views_give_the_size_of_their_first_mip_and_their_shape() {
        let tex2d = resource(3, 64, 32, 1, 5, 1);
        let mip_range = view(4, &[(0x10, 1), (0x14, 2)]);
        assert_eq!(image_words(Some(&mip_range), ViewKind::ShaderResource, &tex2d), Some(words(32, 16, 0, 1, 9)));
        let all_mips = view(4, &[(0x10, 2), (0x14, u32::MAX)]);
        assert_eq!(image_words(Some(&all_mips), ViewKind::ShaderResource, &tex2d), Some(words(16, 8, 0, 2, 9)), "-1 mip levels: the rest of the chain");
        let tex3d = resource(4, 16, 8, 4, 3, 1);
        assert_eq!(image_words(Some(&view(8, &[(0x10, 1), (0x14, u32::MAX)])), ViewKind::ShaderResource, &tex3d), Some(words(8, 4, 1, 1, 10)));
        assert_eq!(image_words(None, ViewKind::ShaderResource, &tex3d), Some(words(16, 8, 3, 2, 10)), "a null description is the whole resource");
        let cubes = resource(3, 16, 16, 12, 1, 1);
        assert_eq!(image_words(Some(&view(9, &[(0x14, u32::MAX)])), ViewKind::ShaderResource, &cubes), Some(words(16, 16, 5, 0, 11)));
        assert_eq!(image_words(Some(&view(10, &[(0x14, 1), (0x1C, 2)])), ViewKind::ShaderResource, &cubes), Some(words(16, 16, 11, 0, 11)));
        let array = resource(3, 8, 8, 6, 1, 1);
        assert_eq!(image_words(Some(&view(5, &[(0x14, 1), (0x18, 2), (0x1C, 3)])), ViewKind::ShaderResource, &array), Some(words(8, 8, 2, 0, 13)));
        assert_eq!(image_words(Some(&view(5, &[(0x14, 1), (0x18, 2), (0x1C, u32::MAX)])), ViewKind::ShaderResource, &array), Some(words(8, 8, 3, 0, 13)));
        let ms = resource(3, 16, 16, 1, 1, 4);
        assert_eq!(image_words(Some(&view(6, &[])), ViewKind::ShaderResource, &ms), Some(words(16, 16, 0, 2, 14)));
        assert_eq!(image_words(None, ViewKind::ShaderResource, &ms), Some(words(16, 16, 0, 2, 14)));
        let tex1d = resource(2, 256, 1, 4, 0, 1);
        assert_eq!(image_words(None, ViewKind::ShaderResource, &tex1d), Some(words(256, 1, 3, 8, 12)), "0 mip levels: the whole chain of 9");
    }

    #[test]
    fn unordered_access_views_have_one_mip_and_buffers_have_no_image_words() {
        let tex3d = resource(4, 16, 8, 4, 3, 1);
        assert_eq!(image_words(Some(&view(8, &[(0x08, 1), (0x0C, 0), (0x10, u32::MAX)])), ViewKind::UnorderedAccess, &tex3d), Some(words(8, 4, 1, 0, 10)));
        let array = resource(3, 8, 8, 6, 1, 1);
        assert_eq!(image_words(Some(&view(5, &[(0x0C, 1), (0x10, u32::MAX)])), ViewKind::UnorderedAccess, &array), Some(words(8, 8, 4, 0, 13)));
        assert_eq!(image_words(None, ViewKind::UnorderedAccess, &resource(3, 64, 32, 1, 5, 1)), Some(words(64, 32, 0, 0, 9)));
        assert_eq!(image_words(None, ViewKind::ShaderResource, &resource(1, 256, 1, 1, 1, 1)), None);
        assert_eq!(image_words(Some(&view(1, &[])), ViewKind::ShaderResource, &resource(3, 8, 8, 1, 1, 1)), None, "a buffer view");
    }
}
