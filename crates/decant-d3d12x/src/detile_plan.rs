//! The console layout of a texture the title placed in its own memory, reduced to what the detile
//! shader (`shaders/detile.hlsl`) needs: per mip level the geometry of the blocks and the tail, and
//! for the whole texture the address equation. The numbers come from `decant_texture::xg`, the same
//! computation `xg_xs.dll` answers the title with, so the layout the title used and the layout read
//! here agree. Nothing in this module touches D3D12.
//!
//! The byte offset of element (x, y) of a mip in array slice z is, as in `TextureLayout::element_offset`:
//!
//! * linear: `z * slice + level offset + y * pitch + x * bpe`
//! * tiled, outside the mip tail: `z * slice + level offset + block index * block size + (equation(x, y, z) mod block size)`
//!   with `block index = (y / block height) * blocks per row + x / block width`
//! * tiled, in the mip tail: `z * slice + (equation(x + tail x, y + tail y, z) mod block size)`
//!
//! What is refused (with a reason, logged by the caller): anything but one sample 2D textures
//! (also arrays and cube maps), textures the GPU writes (render target, depth, unordered access),
//! thick 3D modes, formats the XG layout does not model, and textures of 4 GiB or more.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use decant_texture::xg::d3d12::{D3d12ResourceDesc, texture_desc_from_d3d12};
use decant_texture::xg::{Dim, compute_layout};

/// D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, ALLOW_DEPTH_STENCIL and ALLOW_UNORDERED_ACCESS: the GPU writes the texture.
pub const GPU_WRITTEN_FLAGS: u32 = 0x1 | 0x2 | 0x4;

/// The most entries a planner or footprint cache keeps before it clears on the next miss.
const CACHE_LIMIT: usize = 4096;

/// The key of a layout query: the 0x38 bytes of the console description with `Alignment` cleared
/// (bytes 8 to 16, which `D3d12ResourceDesc::from_bytes` ignores) and the hardware version flag.
type PlanKey = ([u8; 0x38], bool);
type PlanCache = HashMap<PlanKey, Result<Arc<Plan>, String>>;
type FootprintCache = HashMap<PlanKey, Option<(u64, u64)>>;

fn plan_key(raw: &[u8; 0x38], v4: bool) -> PlanKey {
    let mut canonical = *raw;
    canonical[8..16].fill(0);
    (canonical, v4)
}

/// One mip level, field for field what the shader reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipJob {
    pub width_elements: u32,
    pub height_elements: u32,
    pub in_tail: bool,
    pub tail_x: u32,
    pub tail_y: u32,
    pub blocks_x: u32,
    pub level_offset: u32,
    /// Row pitch of a linear level in bytes.
    pub pitch_bytes: u32,
}

/// The layout of one texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub bytes_per_element: u32,
    pub linear: bool,
    pub mode: u32,
    pub block_bits: u32,
    pub block_width: u32,
    pub block_height: u32,
    /// Distance in bytes between array slices.
    pub slice_size: u32,
    pub slices: u32,
    pub mips: Vec<MipJob>,
    /// `(output bit, x mask, y mask, z mask)`, empty for linear.
    pub terms: Vec<(u32, u32, u32, u32)>,
    /// Bytes the title's memory holds for the texture.
    pub size: u64,
}

impl Plan {
    /// The most terms the shader's parameter block holds.
    pub const MAX_TERMS: usize = 24;

    /// The byte offset of element (x, y) of mip `mip` in slice `z`, computed as the shader does.
    pub fn offset(&self, mip: usize, x: u32, y: u32, z: u32) -> u32 {
        let l = &self.mips[mip];
        let z_base = z.wrapping_mul(self.slice_size);
        if self.linear {
            return z_base.wrapping_add(l.level_offset).wrapping_add(y.wrapping_mul(l.pitch_bytes)).wrapping_add(x * self.bytes_per_element);
        }
        let (ex, ey) = if l.in_tail { (x + l.tail_x, y + l.tail_y) } else { (x, y) };
        let mut e = 0u32;
        for &(bit, mx, my, mz) in &self.terms {
            let parity = ((ex & mx).count_ones() + (ey & my).count_ones() + (z & mz).count_ones()) & 1;
            e |= parity << bit;
        }
        let reduced = e & ((1u32 << self.block_bits) - 1);
        if l.in_tail {
            return z_base.wrapping_add(reduced);
        }
        let block_index = (y / self.block_height).wrapping_mul(l.blocks_x).wrapping_add(x / self.block_width);
        z_base.wrapping_add(l.level_offset).wrapping_add(block_index << self.block_bits).wrapping_add(reduced)
    }
}

fn compute_plan(raw: &[u8; 0x38], v4: bool) -> Result<Plan, String> {
    let d = D3d12ResourceDesc::from_bytes(raw);
    if d.dimension != 3 {
        return Err(format!("dimension {} is not a 2D texture", d.dimension));
    }
    if d.sample_count > 1 {
        return Err(format!("{} samples", d.sample_count));
    }
    if d.flags & GPU_WRITTEN_FLAGS != 0 {
        return Err(format!("flags {:#x}: the GPU writes it", d.flags));
    }
    let desc = texture_desc_from_d3d12(&d).map_err(|e| format!("description refused: {e:?}"))?;
    if desc.dim != Dim::D2 {
        return Err("not 2D".into());
    }
    let layout = compute_layout(&desc, v4).map_err(|e| format!("layout refused (format {}, mode {}): {e:?}", desc.format, desc.swizzle_mode))?;
    if layout.is_thick() {
        return Err("thick 3D mode".into());
    }
    let main = layout.plane0();
    if layout.layout.size_bytes >= 1 << 32 || main.slice_size_bytes >= 1 << 32 {
        return Err("4 GiB or more".into());
    }
    let terms = layout.equation().map(|e| e.terms()).unwrap_or_default();
    if terms.len() > Plan::MAX_TERMS {
        return Err(format!("{} equation terms", terms.len()));
    }
    let linear = layout.equation().is_none();
    let (block_width, block_height) = layout.block_shape();
    let mips = main
        .levels
        .iter()
        .map(|l| MipJob {
            width_elements: l.width_elements,
            height_elements: l.height_elements,
            in_tail: l.in_tail,
            tail_x: l.tail_x,
            tail_y: l.tail_y,
            blocks_x: if linear { 0 } else { l.padded_width / block_width.max(1) },
            level_offset: l.offset_bytes as u32,
            pitch_bytes: l.pitch_bytes,
        })
        .collect();
    Ok(Plan {
        bytes_per_element: layout.bytes_per_element(),
        linear,
        mode: layout.swizzle_mode(),
        block_bits: layout.block_bits(),
        block_width,
        block_height,
        slice_size: main.slice_size_bytes as u32,
        slices: layout.array_or_depth(),
        mips,
        terms,
        size: layout.layout.size_bytes,
    })
}

/// Shared plan of the texture whose console description is `raw` (the 0x38 bytes the title passed),
/// laid out with the hardware version `v4`. Plans are cached by canonical description so repeated
/// placements and views of the same texture geometry share one allocation without recomputing XG
/// equations.
pub fn plan_shared(raw: &[u8; 0x38], v4: bool) -> Result<Arc<Plan>, String> {
    static CACHE: OnceLock<Mutex<PlanCache>> = OnceLock::new();
    let key = plan_key(raw, v4);
    let mutex = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = mutex.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned() {
        return hit;
    }
    let computed = compute_plan(&key.0, v4).map(Arc::new);
    let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if guard.len() >= CACHE_LIMIT {
        guard.clear();
    }
    guard.insert(key, computed.clone());
    computed
}

/// The plan of the texture whose console description is `raw` (the 0x38 bytes the title passed), laid
/// out with the hardware version `v4`. The error is the reason for refusing it.
pub fn plan(raw: &[u8; 0x38], v4: bool) -> Result<Plan, String> {
    plan_shared(raw, v4).map(|p| (*p).clone())
}

fn compute_console_footprint(raw: &[u8; 0x38], v4: bool) -> Option<(u64, u64)> {
    let desc = texture_desc_from_d3d12(&D3d12ResourceDesc::from_bytes(raw)).ok()?;
    let layout = compute_layout(&desc, v4).ok()?;
    (layout.layout.size_bytes != 0).then_some((layout.layout.size_bytes, layout.layout.base_alignment_bytes))
}

/// The bytes and alignment the console's XG layout gives the resource whose console description is
/// `raw`, laid out with the hardware version `v4`. The PC footprint (`GetResourceAllocationInfo`)
/// differs: the title sizes its display target block with XG and asserts the block is used up.
pub fn console_footprint(raw: &[u8; 0x38], v4: bool) -> Option<(u64, u64)> {
    static CACHE: OnceLock<Mutex<FootprintCache>> = OnceLock::new();
    let key = plan_key(raw, v4);
    let mutex = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&hit) = mutex.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return hit;
    }
    let computed = compute_console_footprint(&key.0, v4);
    let mut guard = mutex.lock().unwrap_or_else(|e| e.into_inner());
    if guard.len() >= CACHE_LIMIT {
        guard.clear();
    }
    guard.insert(key, computed);
    computed
}

#[cfg(test)]
mod tests {
    use super::*;
    use decant_texture::xg::TextureLayout;

    #[test]
    fn console_footprint_is_the_xg_layout_size() {
        for (w, h, format) in [(3840u64, 2160u32, 24u32), (3840, 2160, 28), (2560, 1440, 24), (1920, 1080, 28), (3840, 2160, 10)] {
            let r = raw(w, h, 1, 1, format, 0, 0x1);
            let (size, align) = console_footprint(&r, false).expect("a footprint");
            let xg = compute_layout(&texture_desc_from_d3d12(&D3d12ResourceDesc::from_bytes(&r)).unwrap(), false).unwrap();
            assert_eq!(size, xg.layout.size_bytes);
            assert!(size >= w * u64::from(h) * if format == 10 { 8 } else { 4 });
            assert!(align.is_power_of_two());
            println!("{w}x{h} format {format}: XG {size:#x} align {align:#x}");
        }
        assert_eq!(console_footprint(&[0u8; 0x38], false), None);
    }

    fn raw(width: u64, height: u32, slices: u16, mips: u16, format: u32, layout: u32, flags: u32) -> [u8; 0x38] {
        let mut b = [0u8; 0x38];
        b[0..4].copy_from_slice(&3u32.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&width.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
        b[0x1C..0x1E].copy_from_slice(&slices.to_le_bytes());
        b[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        b[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        b[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
        b
    }

    fn reference(raw: &[u8; 0x38], v4: bool) -> TextureLayout {
        let d = D3d12ResourceDesc::from_bytes(raw);
        compute_layout(&texture_desc_from_d3d12(&d).unwrap(), v4).unwrap()
    }

    /// Every element of every mip and slice: the plan's offset is the one of the XG layout.
    fn agrees(raw: &[u8; 0x38], v4: bool) {
        let p = plan(raw, v4).unwrap();
        let r = reference(raw, v4);
        let mut checked = 0u64;
        for (mip, l) in r.plane0().levels.iter().enumerate() {
            for z in 0..p.slices {
                for y in 0..l.height_elements {
                    for x in 0..l.width_elements {
                        let want = r.element_offset(mip as u32, u64::from(x), y, z, 0).unwrap();
                        assert_eq!(u64::from(p.offset(mip, x, y, z)), want, "format {} mode {} mip {mip} slice {z} at ({x}, {y})", u32::from_le_bytes(raw[0x20..0x24].try_into().unwrap()), p.mode);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn the_plan_offsets_are_the_xg_offsets_for_the_formats_of_sampled_textures() {
        // BC1, BC3, BC5, BC7, RGBA8, BGRA8, R8, R16 and RGBA16F, the default layout (the 4K S mode of a sampled texture).
        for format in [71u32, 77, 83, 98, 28, 87, 61, 56, 10] {
            for v4 in [false, true] {
                agrees(&raw(300, 150, 1, 0, format, 0, 0), v4);
                agrees(&raw(64, 64, 3, 5, format, 0, 0), v4);
            }
        }
    }

    #[test]
    fn the_explicit_modes_of_the_layout_field_agree_too() {
        // 0x100 + mode: 64K D (10), 64K R X (27), 4K S X (21), 64 KiB S (9) and 256 byte S (1).
        for mode in [10u32, 27, 21, 9, 1, 2, 5] {
            for format in [71u32, 28, 61] {
                for v4 in [false, true] {
                    agrees(&raw(256, 128, 2, 0, format, 0x100 + mode, 0), v4);
                }
            }
        }
    }

    #[test]
    fn a_linear_texture_is_a_pitch_copy() {
        let p = plan(&raw(100, 20, 2, 1, 28, 1, 0), false).unwrap();
        assert!(p.linear && p.terms.is_empty());
        agrees(&raw(100, 20, 2, 1, 28, 1, 0), false);
        assert_eq!(p.offset(0, 3, 2, 1), p.slice_size + 2 * p.mips[0].pitch_bytes + 12);
    }

    #[test]
    fn a_texture_the_gpu_writes_or_one_that_is_not_plain_2d_is_refused() {
        assert!(plan(&raw(64, 64, 1, 1, 28, 0, 0x1), false).unwrap_err().contains("GPU writes"));
        assert!(plan(&raw(64, 64, 1, 1, 28, 0, 0x4), false).is_err());
        assert!(plan(&raw(64, 64, 1, 1, 45, 0, 0x2), false).is_err());
        let mut volume = raw(64, 64, 4, 1, 28, 0, 0);
        volume[0..4].copy_from_slice(&4u32.to_le_bytes());
        assert!(plan(&volume, false).unwrap_err().contains("not a 2D"));
        let mut msaa = raw(64, 64, 1, 1, 28, 0, 0);
        msaa[0x24..0x28].copy_from_slice(&4u32.to_le_bytes());
        assert!(plan(&msaa, false).unwrap_err().contains("samples"));
    }

    #[test]
    fn shared_plans_and_footprints_reuse_cached_entries_across_alignments() {
        let mut a = raw(256, 256, 1, 9, 98, 0, 0);
        a[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
        let mut b = a;
        b[8..16].copy_from_slice(&0x10000u64.to_le_bytes());
        let plan_a = plan_shared(&a, false).expect("a plan");
        let plan_b = plan_shared(&b, false).expect("a plan");
        assert!(Arc::ptr_eq(&plan_a, &plan_b), "identical geometry with different alignment hints shares the same plan");
        let plan_v4 = plan_shared(&a, true).expect("a v4 plan");
        assert_eq!(plan_a.size, plan_v4.size);
        assert_eq!(console_footprint(&a, false), console_footprint(&b, false));
    }
}
