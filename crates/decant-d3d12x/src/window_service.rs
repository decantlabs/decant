//! The device memory window and the translator's service buffers of translated shaders
//! (`decant-dxil` README, "Memory read by device address" and "The translator's own buffers").
//!
//! A translated shader reads memory by 64 bit address as a raw buffer access at a 32 bit byte offset
//! into one window: the root UAV `u0` of register space 14 (its GPU address is the base) and the root
//! CBV `b0` of space 14 (the base, low dword first). Every address the shader forms must lie within
//! 4 GiB above the base. The page table (`t0` of space 13, one `u64` per 16 KiB console page) turns a
//! console address into the PC address the window serves, and `u0` to `u2` of space 13 are the bit
//! sets of pages touched and missing and a third buffer.
//!
//! This module holds what is pure: choosing the base from the title's section buffers, the range check,
//! filling the page table, the root parameter indices of a root signature and which lists have set them.
//! The GPU buffers and the calls on the list are in `service_mem.rs` and `proxy/list_mirror.rs`.
//!
//! `DECANT_D3D12X_WINDOW=0` turns it off (the root signatures then have no window and service parameters
//! and the shaders that need them keep the title's bytecode).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::rts0::RootLayout;
use crate::user_mirror::Bind;

/// The bytes above the base that a 32 bit offset reaches.
pub const WINDOW_BYTES: u64 = 1 << 32;
/// Log2 of the console page of the page table (16 KiB).
pub const PAGE_SHIFT: u32 = 14;
/// Bytes of a console page.
pub const PAGE_BYTES: u64 = 1 << PAGE_SHIFT;
/// Entries of the page table: 1 M pages cover the first 16 GiB of the console address space.
pub const TABLE_PAGES: usize = 1 << 20;
/// Console addresses from 8 TiB up are lowered by 7 TiB before they index the page table.
pub const LOWER_FROM: u64 = 8 << 40;
/// See [`LOWER_FROM`].
pub const LOWER_BY: u64 = 7 << 40;

/// Whether the layer makes the window and service parameters: overrides are on and
/// `DECANT_D3D12X_WINDOW` is not `0`.
pub fn wanted() -> bool {
    crate::dxil_override::configured_dir().is_some() && std::env::var_os("DECANT_D3D12X_WINDOW").is_none_or(|v| v != "0")
}

/// One PC buffer the title's memory lives in: its GPU address and size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    pub gpu: u64,
    pub bytes: u64,
}

/// The window: `base` is the lowest GPU address it covers, `end` the end of the highest buffer inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub base: u64,
    pub end: u64,
}

impl Window {
    /// Whether `[addr, addr + len)` is inside the window: not below the base and within the bytes a 32 bit
    /// offset reaches (and inside the buffers chosen, `end`).
    pub fn contains(&self, addr: u64, len: u64) -> bool {
        let Some(last) = addr.checked_add(len.max(1)) else { return false };
        addr >= self.base && last <= self.end && last - self.base <= WINDOW_BYTES
    }
}

/// The window [`choose_window`] picked and what it leaves out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chosen {
    pub window: Window,
    /// Buffers inside the window.
    pub covered: usize,
    /// Buffers given.
    pub total: usize,
}

/// The base of the window for the buffers `extents`: the start of the buffer from which the most bytes of
/// buffers lie wholly within 4 GiB above it (the lowest such start wins a tie). When the buffers are
/// contiguous in GPU address and fit, that is the lowest buffer and all are covered. Empty buffers are
/// ignored. `None` when there is no buffer.
pub fn choose_window(extents: &[Extent]) -> Option<Chosen> {
    let mut sorted: Vec<Extent> = extents.iter().copied().filter(|e| e.bytes > 0 && e.gpu.checked_add(e.bytes).is_some()).collect();
    sorted.sort_by_key(|e| e.gpu);
    let mut best: Option<(u64, Chosen)> = None;
    for (i, start) in sorted.iter().enumerate() {
        let base = start.gpu;
        let mut bytes = 0u64;
        let mut covered = 0usize;
        let mut end = base;
        for e in &sorted[i..] {
            let last = e.gpu + e.bytes;
            if last - base <= WINDOW_BYTES {
                bytes += e.bytes;
                covered += 1;
                end = end.max(last);
            }
        }
        if best.is_none_or(|(b, _)| bytes > b) {
            best = Some((bytes, Chosen { window: Window { base, end }, covered, total: sorted.len() }));
        }
    }
    best.map(|(_, c)| c)
}

/// The page table index of the console address `va`, `None` past the table of `pages` entries.
pub fn console_page(va: u64, pages: usize) -> Option<usize> {
    let lowered = if va >= LOWER_FROM { va - LOWER_BY } else { va };
    let page = usize::try_from(lowered >> PAGE_SHIFT).ok()?;
    (page < pages).then_some(page)
}

/// What [`fill_page_table`] found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FillStats {
    /// Pages with an entry.
    pub mapped: usize,
    /// Pages whose PC address is outside the window, left 0 (the shader sees them as missing).
    pub outside: usize,
}

/// Fills `table` (one `u64` per console page, page `i` is the console address `i * 16 KiB`): the PC GPU
/// address `resolve(console address, 16 KiB)` gives, or 0 when it gives none or when the page is not inside
/// `window` (a shader cannot reach it, so it is refused and counted). With no window every entry is 0.
pub fn fill_page_table(table: &mut [u64], window: Option<&Window>, mut resolve: impl FnMut(u64, u64) -> Option<u64>) -> FillStats {
    let mut stats = FillStats::default();
    for (page, entry) in table.iter_mut().enumerate() {
        *entry = 0;
        let Some(window) = window else { continue };
        let Some(gpu) = resolve((page as u64) << PAGE_SHIFT, PAGE_BYTES) else { continue };
        if window.contains(gpu, PAGE_BYTES) {
            *entry = gpu;
            stats.mapped += 1;
        } else {
            stats.outside += 1;
        }
    }
    stats
}

/// The indices of the window and service parameters of one root signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbiSig {
    pub window_uav: Option<u32>,
    pub window_base: Option<u32>,
    /// The page table SRV.
    pub table: Option<u32>,
    /// The three service UAVs.
    pub bits: [Option<u32>; 3],
}

impl AbiSig {
    /// `None` when `layout` has none of the parameters.
    pub fn from_layout(layout: &RootLayout) -> Option<AbiSig> {
        let idx = |i: Option<usize>| i.and_then(|i| u32::try_from(i).ok());
        let [table, b0, b1, b2] = layout.service_indices();
        let sig = AbiSig { window_uav: idx(layout.window_uav_index()), window_base: idx(layout.window_base_index()), table: idx(table), bits: [idx(b0), idx(b1), idx(b2)] };
        (sig != AbiSig::default()).then_some(sig)
    }
}

static ANY: AtomicBool = AtomicBool::new(false);
static SIGS: Mutex<Option<HashMap<usize, Arc<AbiSig>>>> = Mutex::new(None);
static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);

/// What a list has set: the root signature object of each kind of pipeline and the epoch at which the
/// parameters were set for it (`None` for not yet).
#[derive(Clone, Copy, Debug, Default)]
struct ListState {
    sig: [usize; 2],
    set_at: [Option<u64>; 2],
}

/// Whether any root signature with these parameters exists, so that the common case takes no lock.
pub fn active() -> bool {
    ANY.load(Ordering::Acquire)
}

/// Remembers the root signature `object` when its layout has window or service parameters.
pub fn remember_signature(object: usize, layout: &RootLayout) -> bool {
    let Some(sig) = AbiSig::from_layout(layout) else { return false };
    SIGS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(object, Arc::new(sig));
    ANY.store(true, Ordering::Release);
    true
}

/// The parameters of root signature `object`.
pub fn signature(object: usize) -> Option<Arc<AbiSig>> {
    SIGS.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&object).cloned()
}

/// `object` is the root signature of `bind` on `list` from now on: nothing is set for it yet.
pub fn note_signature(list: usize, bind: Bind, object: usize) {
    let mut g = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let st = g.get_or_insert_with(HashMap::new).entry(list).or_default();
    st.sig[bind as usize] = object;
    st.set_at[bind as usize] = None;
}

/// The parameters to set before a draw or dispatch of `bind` on `list` at `epoch`: the root signature has
/// them and they were not set for it at this epoch.
pub fn pending(list: usize, bind: Bind, epoch: u64) -> Option<Arc<AbiSig>> {
    let object = {
        let g = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        let st = g.as_ref()?.get(&list)?;
        if st.set_at[bind as usize] == Some(epoch) {
            return None;
        }
        st.sig[bind as usize]
    };
    signature(object)
}

/// The parameters of `bind` on `list` are set at `epoch`.
pub fn mark_set(list: usize, bind: Bind, epoch: u64) {
    if let Some(st) = LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.get_mut(&list)) {
        st.set_at[bind as usize] = Some(epoch);
    }
}

/// `list` was reset: its root signatures and arguments are gone.
pub fn forget(list: usize) {
    if let Some(m) = LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&list);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rts0::{RegisterClass, RootParam, RootParamKind};

    const GIB: u64 = 1 << 30;

    fn ext(gpu: u64, bytes: u64) -> Extent {
        Extent { gpu, bytes }
    }

    #[test]
    fn contiguous_buffers_get_the_lowest_one_as_base() {
        let c = choose_window(&[ext(0x2_0000_0000 + GIB, GIB), ext(0x2_0000_0000, GIB), ext(0x2_0000_0000 + 2 * GIB, GIB)]).unwrap();
        assert_eq!(c.window, Window { base: 0x2_0000_0000, end: 0x2_0000_0000 + 3 * GIB });
        assert_eq!((c.covered, c.total), (3, 3));
    }

    #[test]
    fn the_base_is_the_start_that_covers_the_most_bytes() {
        // a small buffer far below, then 3 GiB and 2 GiB buffers a bit apart: the window starts at the 3 GiB one
        let low = ext(0x1000_0000, 0x100_0000);
        let big = ext(0x10_0000_0000, 3 * GIB);
        let next = ext(0x10_0000_0000 + 3 * GIB + 0x1000, 2 * GIB);
        let c = choose_window(&[next, low, big]).unwrap();
        assert_eq!(c.window.base, big.gpu);
        assert_eq!((c.covered, c.total), (1, 3), "the 2 GiB buffer would end past 4 GiB above the base");
        assert!(c.window.contains(big.gpu, 4) && !c.window.contains(next.gpu, 4) && !c.window.contains(low.gpu, 4));
    }

    #[test]
    fn a_tie_takes_the_lowest_base_and_empty_input_has_none() {
        let c = choose_window(&[ext(0x40_0000_0000, 3 * GIB), ext(0x10_0000_0000, 3 * GIB)]).unwrap();
        assert_eq!(c.window.base, 0x10_0000_0000);
        assert_eq!(choose_window(&[]), None);
        assert_eq!(choose_window(&[ext(0x1000, 0)]), None, "an empty buffer is no window");
        assert_eq!(choose_window(&[ext(u64::MAX - 4, 100)]), None, "a buffer that wraps is ignored");
    }

    #[test]
    fn the_range_check_covers_both_ends() {
        let w = Window { base: 0x1000, end: 0x1000 + 0x8000 };
        assert!(w.contains(0x1000, 0x8000));
        assert!(!w.contains(0xFFF, 4), "below the base");
        assert!(!w.contains(0x1000 + 0x7FFD, 4), "past the end");
        assert!(w.contains(0x1000, 0), "a zero length counts as one byte");
        assert!(!w.contains(u64::MAX, 4), "wrapping");
        // end beyond 4 GiB above the base never counts as inside
        let wide = Window { base: 0, end: 3 * WINDOW_BYTES };
        assert!(wide.contains(WINDOW_BYTES - 4, 4));
        assert!(!wide.contains(WINDOW_BYTES - 3, 4));
    }

    /// A fake section table: console base, bytes and the GPU address of the section buffer.
    fn fake(sections: &'static [(u64, u64, u64)]) -> impl FnMut(u64, u64) -> Option<u64> {
        move |va, len| sections.iter().find(|(c, b, _)| va >= *c && va + len <= c + b).map(|(c, _, g)| g + (va - c))
    }

    #[test]
    fn the_page_table_maps_pages_of_sections_to_their_gpu_address() {
        // section A holds console pages 4..8 (0x10000..0x20000), section B pages 10..11
        static SECTIONS: [(u64, u64, u64); 2] = [(0x1_0000, 0x1_0000, 0x7000_0000), (0x2_8000, 0x4000, 0x7001_0000)];
        let mut table = vec![0xFFFF_FFFF_FFFF_FFFFu64; 16];
        let window = Window { base: 0x7000_0000, end: 0x7001_4000 };
        let stats = fill_page_table(&mut table, Some(&window), fake(&SECTIONS));
        assert_eq!(stats, FillStats { mapped: 5, outside: 0 });
        assert_eq!(table[3], 0, "an unmapped page is zero, stale content is cleared");
        assert_eq!(table[4], 0x7000_0000);
        assert_eq!(table[7], 0x7000_0000 + 3 * PAGE_BYTES);
        assert_eq!(table[8], 0);
        assert_eq!(table[10], 0x7001_0000);
        assert_eq!(table[11], 0);
    }

    #[test]
    fn pages_outside_the_window_are_refused_and_no_window_means_all_zero() {
        static SECTIONS: [(u64, u64, u64); 2] = [(0x0, 0x1_0000, 0x7000_0000), (0x1_0000, 0x1_0000, 0x9_0000_0000)];
        let mut table = vec![0u64; 8];
        let window = Window { base: 0x7000_0000, end: 0x7001_0000 };
        let stats = fill_page_table(&mut table, Some(&window), fake(&SECTIONS));
        assert_eq!(stats, FillStats { mapped: 4, outside: 4 });
        assert_eq!(table, [0x7000_0000, 0x7000_4000, 0x7000_8000, 0x7000_C000, 0, 0, 0, 0]);
        let stats = fill_page_table(&mut table, None, fake(&SECTIONS));
        assert_eq!(stats, FillStats::default());
        assert!(table.iter().all(|&e| e == 0));
    }

    #[test]
    fn console_addresses_index_the_table_and_high_ones_are_lowered() {
        assert_eq!(console_page(0, 100), Some(0));
        assert_eq!(console_page(PAGE_BYTES * 3 + 5, 100), Some(3));
        assert_eq!(console_page(PAGE_BYTES * 100, 100), None);
        assert_eq!(console_page(LOWER_FROM + PAGE_BYTES * 2, 1 << 30), Some(((1u64 << 40) >> PAGE_SHIFT) as usize + 2));
        assert_eq!(console_page(LOWER_FROM + PAGE_BYTES * 2, 100), None);
    }

    fn desc(class: RegisterClass, register: u32, space: u32) -> RootParam {
        RootParam { kind: RootParamKind::Descriptor { class, register, space }, visibility: 0 }
    }

    #[test]
    fn the_signature_finds_its_parameters_and_lists_set_them_once_per_epoch() {
        let layout = RootLayout {
            version: 2,
            flags: 0,
            static_samplers: vec![],
            params: vec![
                RootParam { kind: RootParamKind::Constants { register: 0, space: 0, count: 2 }, visibility: 0 },
                desc(RegisterClass::Uav, 0, 14),
                desc(RegisterClass::Cbv, 0, 14),
                desc(RegisterClass::Srv, 0, 13),
                desc(RegisterClass::Uav, 0, 13),
                desc(RegisterClass::Uav, 1, 13),
                desc(RegisterClass::Uav, 2, 13),
            ],
        };
        let sig = AbiSig::from_layout(&layout).unwrap();
        assert_eq!(sig, AbiSig { window_uav: Some(1), window_base: Some(2), table: Some(3), bits: [Some(4), Some(5), Some(6)] });
        let none = RootLayout { params: vec![], ..layout.clone() };
        assert_eq!(AbiSig::from_layout(&none), None);
        // a list: set at epoch 1, again at epoch 1 nothing, a new signature or epoch asks again
        let (list, object) = (0xA11CE_usize, 0xB0B_usize);
        assert!(remember_signature(object, &layout) && !remember_signature(object + 1, &none));
        assert!(pending(list, Bind::Compute, 1).is_none(), "no signature set yet");
        note_signature(list, Bind::Compute, object);
        assert_eq!(pending(list, Bind::Compute, 1).as_deref(), Some(&sig));
        mark_set(list, Bind::Compute, 1);
        assert!(pending(list, Bind::Compute, 1).is_none());
        assert!(pending(list, Bind::Compute, 2).is_some(), "the window moved");
        assert!(pending(list, Bind::Graphics, 1).is_none(), "graphics has no root signature on this list");
        note_signature(list, Bind::Compute, object);
        assert!(pending(list, Bind::Compute, 1).is_some(), "a new Set*RootSignature clears the parameters");
        forget(list);
        assert!(pending(list, Bind::Compute, 1).is_none());
        assert!(active());
    }
}
