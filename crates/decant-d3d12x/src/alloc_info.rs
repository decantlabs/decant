//! What the console device answers for `GetResourceAllocationInfo` (device slots 25 and 108): the size
//! and alignment of a texture in its console layout, as `xg_xs.dll` computes it (`decant_texture::xg`),
//! not the PC footprint of the same description.
//!
//! The title keeps these sizes in its streaming tables and reads each texture with exactly that many bytes
//! (game+0x343e80 asks, game+0x51fb680 stores the size in pages, game+0x43c4850 reads mip by mip). The
//! file's compressed chunks are cut to the console sizes, and the reader asserts that a compressed chunk
//! is read whole (int3 at game+0x4ffd879, where it asked for 0x4000 bytes of a 0x10000 byte chunk with
//! the PC answer). For textures under 0x18001 bytes the title keeps the size as answered, larger ones it
//! rounds up to 64 KiB itself.
//!
//! For a small texture the title asks first with `Alignment` 0x1000 and layout 0x105 (`SW_4KB_S`) and
//! keeps the answer when the alignment comes back as 0x1000, otherwise it asks again with `Alignment` 0.
//! The console answers the layout's own alignment unless the description asks for more, and the size
//! rounded up to that alignment. So a 4 KiB mode is granted 4 KiB, a 64 KiB mode answers 64 KiB.
//!
//! Buffers and descriptions XG refuses keep the PC answer.

use crate::detile_plan::console_footprint;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct FastBytesHasher(u64);

impl Hasher for FastBytesHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0 ^ 0xCBF2_9CE4_8422_2325;
        let (chunks, rem) = bytes.as_chunks::<8>();
        for &c in chunks {
            let word = u64::from_le_bytes(c);
            h = (h ^ word).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        for &b in rem {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
        self.0 = h;
    }
}

type FastDescSet = HashSet<[u8; 0x38], BuildHasherDefault<FastBytesHasher>>;
type FastAllocMap = HashMap<([u8; 0x38], bool), Option<(u64, u64)>, BuildHasherDefault<FastBytesHasher>>;

/// `D3D12_RESOURCE_DIMENSION_BUFFER`.
const BUFFER: u32 = 1;

/// How many distinct descriptions are logged with both answers.
pub const LOGGED_DESCRIPTIONS: usize = 256;

/// Maximum cached `console_allocation` entries.
const ALLOC_CACHE_CAP: usize = 4096;

/// Whether `raw` is a description not logged yet, while fewer than `LOGGED_DESCRIPTIONS` were.
pub fn first_sight(raw: &[u8; 0x38]) -> bool {
    static SEEN_FULL: AtomicBool = AtomicBool::new(false);
    static SEEN: RwLock<Option<FastDescSet>> = RwLock::new(None);
    if SEEN_FULL.load(Ordering::Relaxed) {
        return false;
    }
    if let Some(seen) = SEEN.read().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref()
        && (seen.len() >= LOGGED_DESCRIPTIONS || seen.contains(raw))
    {
        return false;
    }
    let mut guard = SEEN.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    let seen = guard.get_or_insert_with(FastDescSet::default);
    if seen.len() >= LOGGED_DESCRIPTIONS {
        SEEN_FULL.store(true, Ordering::Relaxed);
        return false;
    }
    if !seen.insert(*raw) {
        return false;
    }
    if seen.len() >= LOGGED_DESCRIPTIONS {
        SEEN_FULL.store(true, Ordering::Relaxed);
    }
    true
}

/// The console's `(SizeInBytes, Alignment)` for the console description `raw`, laid out with the
/// hardware version `v4`. `None` for a buffer, for a description XG refuses and for a requested
/// alignment that is not a power of two: the caller keeps the PC answer.
pub fn console_allocation(raw: &[u8; 0x38], v4: bool) -> Option<(u64, u64)> {
    static CACHE: RwLock<Option<FastAllocMap>> = RwLock::new(None);
    if u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) == BUFFER {
        return None;
    }
    let key = (*raw, v4);
    if let Some(&cached) = CACHE.read().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().and_then(|m| m.get(&key)) {
        return cached;
    }
    let computed = compute_console_allocation(raw, v4);
    let mut guard = CACHE.write().unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = guard.get_or_insert_with(FastAllocMap::default);
    if map.len() < ALLOC_CACHE_CAP {
        map.insert(key, computed);
    }
    computed
}

fn compute_console_allocation(raw: &[u8; 0x38], v4: bool) -> Option<(u64, u64)> {
    let requested = u64::from_le_bytes([raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]]);
    let (size, layout_alignment) = console_footprint(raw, v4)?;
    let alignment = requested.max(layout_alignment);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some((size.checked_next_multiple_of(alignment)?, alignment))
}

/// One log line with the description and both answers, `xg` `None` when XG gave none.
pub fn describe(slot: u32, raw: &[u8; 0x38], pc: (u64, u64), xg: Option<(u64, u64)>) -> String {
    let u16_at = |o: usize| u16::from_le_bytes([raw[o], raw[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap_or_default());
    let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap_or_default());
    let xg_text = xg.map_or_else(|| "none, the PC answer stands".to_owned(), |(size, align)| format!("{size:#x} align {align:#x}"));
    format!(
        "console device slot {slot}: allocation info of dimension {}, {}x{}x{}, mips {}, format {}, layout {:#x}, flags {:#x}, asked alignment {:#x}: PC {:#x} align {:#x}, XG {xg_text}",
        u32_at(0),
        u64_at(0x10),
        u32_at(0x18),
        u16_at(0x1C),
        u16_at(0x1E),
        u32_at(0x20),
        u32_at(0x2C),
        u32_at(0x30),
        u64_at(8),
        pc.0,
        pc.1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use decant_texture::xg::compute_layout;
    use decant_texture::xg::d3d12::{D3d12ResourceDesc, texture_desc_from_d3d12};

    /// A console `D3D12_RESOURCE_DESC`: 2D unless `dimension` says otherwise, one sample.
    fn desc(dimension: u32, alignment: u64, width: u64, height: u32, mips: u16, format: u32, layout: u32) -> [u8; 0x38] {
        let mut b = [0u8; 0x38];
        b[0..4].copy_from_slice(&dimension.to_le_bytes());
        b[8..16].copy_from_slice(&alignment.to_le_bytes());
        b[0x10..0x18].copy_from_slice(&width.to_le_bytes());
        b[0x18..0x1C].copy_from_slice(&height.to_le_bytes());
        b[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        b[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        b[0x20..0x24].copy_from_slice(&format.to_le_bytes());
        b[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        b[0x2C..0x30].copy_from_slice(&layout.to_le_bytes());
        b
    }

    fn xg_layout(raw: &[u8; 0x38]) -> (u64, u64) {
        let l = compute_layout(&texture_desc_from_d3d12(&D3d12ResourceDesc::from_bytes(raw)).unwrap(), false).unwrap();
        (l.layout.size_bytes, l.layout.base_alignment_bytes)
    }

    /// The title's first question for a small texture: 4 KiB alignment, `SW_4KB_S`. The console grants
    /// 4 KiB and answers the XG size of the whole mip chain.
    #[test]
    fn a_small_bc7_texture_in_a_4k_mode_gets_the_xg_size_with_4k_alignment() {
        let raw = desc(3, 0x1000, 128, 128, 8, 98, 0x105);
        let (size, align) = console_allocation(&raw, false).expect("a texture XG lays out");
        let (xg_size, xg_align) = xg_layout(&raw);
        assert_eq!(xg_align, 0x1000, "SW_4KB_S blocks are 4 KiB");
        assert_eq!((size, align), (xg_size.next_multiple_of(0x1000), 0x1000));
        assert!(size > 0x4000, "the mips of the console layout take more than mip 0: {size:#x}");
        println!("128x128 BC7 SW_4KB_S with mips: XG {size:#x}");
    }

    #[test]
    fn a_texture_in_a_64k_mode_answers_its_size_rounded_to_64k() {
        // Layout 0x10A (SW_64KB_S): the title's 4 KiB question gets 64 KiB back, so it asks again.
        let asked_4k = desc(3, 0x1000, 2048, 2048, 12, 98, 0x10A);
        let (size, align) = console_allocation(&asked_4k, false).expect("a texture XG lays out");
        assert_eq!(align, 0x10000);
        assert_eq!(size, xg_layout(&asked_4k).0.next_multiple_of(0x10000));
        assert_eq!(console_allocation(&desc(3, 0, 2048, 2048, 12, 98, 0x10A), false), Some((size, align)));
    }

    #[test]
    fn layout_0_is_the_mode_xg_picks() {
        // XG picks a 4 KiB mode for a 2D texture that is not a target.
        let raw = desc(3, 0x1000, 2048, 2048, 12, 98, 0);
        assert_eq!(console_allocation(&raw, false), Some((xg_layout(&raw).0.next_multiple_of(0x1000), 0x1000)));
    }

    #[test]
    fn each_description_is_logged_once() {
        let raw = desc(3, 0x1000, 96, 32, 1, 71, 0x105);
        assert!(first_sight(&raw));
        assert!(!first_sight(&raw));
    }

    #[test]
    fn a_larger_requested_alignment_rounds_the_size_up() {
        let raw = desc(3, 0x10000, 128, 128, 8, 98, 0x105);
        let (size, align) = console_allocation(&raw, false).expect("a texture XG lays out");
        assert_eq!(align, 0x10000);
        assert_eq!(size, xg_layout(&raw).0.next_multiple_of(0x10000));
        assert_eq!(console_allocation(&desc(3, 0x3000, 128, 128, 8, 98, 0x105), false), None, "not a power of two");
    }

    #[test]
    fn buffers_and_refused_descriptions_keep_the_pc_answer() {
        let buffer = desc(BUFFER, 0, 0x12345, 1, 1, 0, 1);
        assert_eq!(console_allocation(&buffer, false), None);
        assert_eq!(console_allocation(&[0u8; 0x38], false), None);
        assert_eq!(console_allocation(&desc(3, 0, 128, 128, 1, 0, 0x105), false), None, "format 0 has no layout");
    }

    #[test]
    fn the_log_line_names_the_description_and_both_answers() {
        let raw = desc(3, 0x1000, 128, 128, 8, 98, 0x105);
        let line = describe(25, &raw, (0x4000, 0x1000), Some((0x8000, 0x1000)));
        assert!(line.contains("dimension 3, 128x128x1, mips 8, format 98, layout 0x105"), "{line}");
        assert!(line.contains("asked alignment 0x1000: PC 0x4000 align 0x1000, XG 0x8000 align 0x1000"), "{line}");
        assert!(describe(25, &raw, (1, 2), None).ends_with("XG none, the PC answer stands"));
    }

    #[test]
    fn repeated_console_allocation_hits_cache_consistently() {
        let raw = desc(3, 0x1000, 256, 256, 9, 98, 0x105);
        let first = console_allocation(&raw, false);
        let second = console_allocation(&raw, false);
        assert_eq!(first, second);
        assert!(first.is_some());
    }
}
