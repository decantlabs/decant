//! Heaps for the textures of views at an address (`proxy::view_texture`). The title makes hundreds
//! of small textures a second that way while its world loads (measured: 16 by 4 `R16G16B16A16_FLOAT`,
//! 512 bytes), and a committed PC texture takes 64 KiB at least, so thousands of them took GiBs. Here
//! each one is placed in a heap of [`HEAP_BYTES`] at the small texture alignment of 4 KiB when the
//! runtime allows it (64 KiB else). The space is handed out in order and not given back: a texture
//! of this kind goes only when no slot holds its view any more, which is rare. A heap stays for the
//! life of the process, and is used again once every texture placed in it has been released
//! (`ViewTextures::collect`, past the memory cap) and [`REUSE_PRESENTS`] presents have passed since,
//! for the GPU to be done with them (`retire`).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Bytes of one large streaming heap.
pub const HEAP_BYTES: u64 = 64 << 20;
/// Bytes of one small persistent texture heap.
const SMALL_HEAP_BYTES: u64 = 16 << 20;
/// Textures up to this byte size use small heaps so tiny persistent views do not pin 64 MiB heaps.
const SMALL_TEXTURE_LIMIT: u64 = 64 << 10;
/// Maximum warm empty heaps kept per size class for instant reuse.
const WARM_FREE_HEAPS_PER_CLASS: usize = 1;
/// The small resource placement alignment of the PC runtime.
const SMALL: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SizeClass {
    Small = 0,
    Large = 1,
}

fn size_class_for(size: u64) -> SizeClass {
    if size <= SMALL_TEXTURE_LIMIT { SizeClass::Small } else { SizeClass::Large }
}

fn class_heap_bytes(class: SizeClass) -> u64 {
    match class {
        SizeClass::Small => SMALL_HEAP_BYTES,
        SizeClass::Large => HEAP_BYTES,
    }
}

fn fit_in(used: u64, size: u64, align: u64, capacity: u64) -> Option<u64> {
    let at = used.checked_next_multiple_of(align.max(1))?;
    (at.checked_add(size)? <= capacity).then_some(at)
}

/// Where the next placement of `size` bytes at `align` goes in a heap of [`HEAP_BYTES`] whose first
/// `used` bytes are taken: the offset, or `None` when it does not fit (pure, for the tests).
#[cfg(test)]
pub fn fit(used: u64, size: u64, align: u64) -> Option<u64> {
    fit_in(used, size, align, HEAP_BYTES)
}

/// Presents a heap whose textures were all released waits before it is filled again.
pub const REUSE_PRESENTS: u64 = 8;

/// Whether a heap with `live` textures placed in it, all released at present `free_at`, may be filled
/// again at present `now` (pure, for the tests).
pub fn reusable(live: u32, free_at: Option<u64>, now: u64) -> bool {
    live == 0 && free_at.is_some_and(|t| t + REUSE_PRESENTS <= now)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AllocKey {
    dimension: i32,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mip_levels: u16,
    format: i32,
    sample_count: u32,
    sample_quality: u32,
    layout: i32,
    flags: i32,
}

impl AllocKey {
    fn from_desc(desc: &D3D12_RESOURCE_DESC) -> Self {
        Self {
            dimension: desc.Dimension.0,
            width: desc.Width,
            height: desc.Height,
            depth_or_array: desc.DepthOrArraySize,
            mip_levels: desc.MipLevels,
            format: desc.Format.0,
            sample_count: desc.SampleDesc.Count,
            sample_quality: desc.SampleDesc.Quality,
            layout: desc.Layout.0,
            flags: desc.Flags.0,
        }
    }
}

static ALLOC_INFO_CACHE: LazyLock<RwLock<HashMap<AllocKey, (u64, u64)>>> = LazyLock::new(Default::default);

fn cached_alloc_info(key: AllocKey) -> Option<(u64, u64)> {
    ALLOC_INFO_CACHE.read().unwrap_or_else(|e| e.into_inner()).get(&key).copied()
}

fn cache_alloc_info(key: AllocKey, size: u64, align: u64) {
    if size != u64::MAX {
        ALLOC_INFO_CACHE.write().unwrap_or_else(|e| e.into_inner()).insert(key, (size, align));
    }
}

fn allocation_info(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<(u64, u64)> {
    let key = AllocKey::from_desc(desc);
    if let Some(hit) = cached_alloc_info(key) {
        return Some(hit);
    }
    let mut d = *desc;
    d.Alignment = SMALL;
    // SAFETY: one valid description.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.Alignment != SMALL {
        d.Alignment = 0;
        // SAFETY: as above.
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    cache_alloc_info(key, info.SizeInBytes, info.Alignment);
    Some((info.SizeInBytes, info.Alignment))
}

#[derive(Default)]
struct SlotMeta {
    used: u64,
    /// Textures placed in it and not released.
    live: u32,
    /// The present at which `live` fell to 0.
    free_at: Option<u64>,
}

struct Heap {
    heap: ID3D12Heap,
    class: SizeClass,
    meta: SlotMeta,
}

// SAFETY: D3D12 heaps are free threaded, and the list is only touched under its lock.
unsafe impl Send for Heap {}

#[derive(Default)]
struct HeapPool {
    heaps: Vec<Option<Heap>>,
    active_hint: [usize; 2],
}

static HEAPS: Mutex<HeapPool> = Mutex::new(HeapPool { heaps: Vec::new(), active_hint: [0, 0] });
static HEAP_SEQ: AtomicUsize = AtomicUsize::new(0);

fn insert_heap(pool: &mut HeapPool, heap: Heap) -> usize {
    if let Some(index) = pool.heaps.iter().position(Option::is_none) {
        pool.heaps[index] = Some(heap);
        index
    } else {
        pool.heaps.push(Some(heap));
        pool.heaps.len() - 1
    }
}

fn reclaim_empty_in_pool(pool: &mut HeapPool, now: u64) -> Vec<(ID3D12Heap, u64)> {
    let mut warm = [0usize; 2];
    let mut to_release = Vec::new();
    for slot in &mut pool.heaps {
        let Some(h) = slot.as_mut() else { continue };
        if !reusable(h.meta.live, h.meta.free_at, now) {
            continue;
        }
        let class_idx = h.class as usize;
        if warm[class_idx] < WARM_FREE_HEAPS_PER_CLASS {
            warm[class_idx] += 1;
            h.meta.used = 0;
            h.meta.free_at = None;
        } else if let Some(gone) = slot.take() {
            to_release.push((gone.heap, class_heap_bytes(gone.class)));
        }
    }
    to_release
}

fn reserve_in_pool(pool: &mut HeapPool, class: SizeClass, size: u64, align: u64) -> Option<(ID3D12Heap, usize, u64)> {
    let len = pool.heaps.len();
    if len == 0 {
        return None;
    }
    let capacity = class_heap_bytes(class);
    let hint = &mut pool.active_hint[class as usize];
    let start = (*hint).min(len - 1);
    for step in 0..len {
        let index = (start + step) % len;
        let Some(h) = pool.heaps[index].as_mut() else { continue };
        if h.class != class {
            continue;
        }
        if let Some(at) = fit_in(h.meta.used, size, align, capacity) {
            h.meta.used = at + size;
            h.meta.live += 1;
            h.meta.free_at = None;
            *hint = index;
            return Some((h.heap.clone(), index, at));
        }
    }
    None
}

fn release_excess_heaps(heaps: Vec<(ID3D12Heap, u64)>) {
    for (heap, bytes) in heaps {
        crate::mem_owners::remove(
            crate::mem_owners::Kind::ViewHeap,
            crate::mem_owners::Segment::Local,
            1,
            bytes,
        );
        // SAFETY: `pool.heaps` dropped its only reference to this empty heap; `retire::release`
        // holds it until in flight GPU work finishes.
        unsafe { crate::retire::release(heap.into_raw(), "view heap") };
    }
}

/// A texture of the PC description `desc` (after `abi::pc_resource_desc`) in state `state`, placed in
/// one of the heaps. `None` for a texture larger than its class heap or when the runtime refuses:
/// the caller makes a committed one then.
///
/// Returns the texture, the index of its heap and its size in bytes.
pub fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, state: u32) -> Option<(ID3D12Resource, usize, u64)> {
    let (size, align) = allocation_info(device, desc)?;
    let class = size_class_for(size);
    let heap_bytes = class_heap_bytes(class);
    if size > heap_bytes {
        return None;
    }
    let now = crate::title_heap::presents();
    let (reserved, excess) = {
        let mut pool = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        let excess = reclaim_empty_in_pool(&mut pool, now);
        let reserved = reserve_in_pool(&mut pool, class, size, align);
        (reserved, excess)
    };
    release_excess_heaps(excess);
    let (heap, index, at) = match reserved {
        Some(hit) => hit,
        None => {
            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: heap_bytes,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut created: Option<ID3D12Heap> = None;
            // SAFETY: a valid description and output, run outside HEAPS.lock().
            unsafe { device.CreateHeap(&heap_desc, &mut created) }.ok()?;
            let new_heap = created?;
            let seq = HEAP_SEQ.fetch_add(1, Ordering::Relaxed);
            // SAFETY: a live heap, named for DRED.
            let _ = unsafe { new_heap.SetName(&windows::core::HSTRING::from(format!("decant view texture heap {seq}"))) };
            let mut pool = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
            let choice = if let Some(hit) = reserve_in_pool(&mut pool, class, size, align) {
                insert_heap(&mut pool, Heap { heap: new_heap, class, meta: SlotMeta::default() });
                hit
            } else {
                let clone = new_heap.clone();
                let new_index = insert_heap(
                    &mut pool,
                    Heap {
                        heap: new_heap,
                        class,
                        meta: SlotMeta { used: size, live: 1, free_at: None },
                    },
                );
                pool.active_hint[class as usize] = new_index;
                (clone, new_index, 0)
            };
            drop(pool);
            crate::mem_owners::add(
                crate::mem_owners::Kind::ViewHeap,
                crate::mem_owners::Segment::Local,
                1,
                heap_bytes,
            );
            choice
        }
    };
    let mut d = *desc;
    d.Alignment = if align == SMALL { SMALL } else { 0 };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: the heap is live, the reserved range lies inside it, and CreatePlacedResource runs lock free.
    let placed = unsafe { device.CreatePlacedResource(&heap, at, &d, D3D12_RESOURCE_STATES(state as i32), None, &mut res) };
    if placed.is_err() || res.is_none() {
        released(index);
        return None;
    }
    Some((res?, index, size))
}

/// A texture placed in heap `index` was released (its last reference is held back until the GPU is
/// done, `retire`). The heap is used again once all of its textures are gone.
pub fn released(index: usize) {
    let now = crate::title_heap::presents();
    let mut pool = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Some(h)) = pool.heaps.get_mut(index) {
        h.meta.live = h.meta.live.saturating_sub(1);
        if h.meta.live == 0 {
            h.meta.free_at = Some(now);
        }
    }
}

/// The dedicated video memory and the budget (bytes, 0 when unknown) of the adapter of `device`.
#[cfg(windows)]
pub fn adapter_memory(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> (u64, u64) {
    use windows::Win32::Graphics::Dxgi::*;
    // SAFETY: plain queries on a live device and on adapters made here.
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return (0, 0) };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()) else { return (0, 0) };
        let dedicated = adapter.GetDesc().map_or(0, |d| d.DedicatedVideoMemory as u64);
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let budget = if adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).is_ok() { info.Budget } else { 0 };
        (dedicated, budget)
    }
}

/// Heaps made so far and the bytes taken in them.
pub fn usage() -> (usize, u64) {
    let pool = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    (pool.heaps.iter().flatten().count(), pool.heaps.iter().flatten().map(|h| h.meta.used).sum())
}

/// Heaps with no live texture: kept, and reusable once their wait is over.
pub fn empty_heaps() -> usize {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner()).heaps.iter().flatten().filter(|h| h.meta.live == 0).count()
}

#[cfg(test)]
mod tests {
    use super::{HEAP_BYTES, REUSE_PRESENTS, fit, reusable};

    #[test]
    fn placements_follow_each_other_at_their_alignment_and_stop_at_the_end() {
        assert_eq!(fit(0, 4096, 4096), Some(0));
        assert_eq!(fit(4096, 512, 4096), Some(4096));
        assert_eq!(fit(4100, 4096, 4096), Some(8192), "the next aligned offset");
        assert_eq!(fit(4096, 65536, 65536), Some(65536));
        assert_eq!(fit(HEAP_BYTES - 4096, 4096, 4096), Some(HEAP_BYTES - 4096), "the last page fits");
        assert_eq!(fit(HEAP_BYTES - 4096, 8192, 4096), None, "past the end");
    }

    #[test]
    fn a_heap_is_used_again_only_when_all_its_textures_are_gone_and_the_gpu_had_time() {
        assert!(!reusable(1, None, 1000), "a texture still lives in it");
        assert!(!reusable(0, None, 1000), "never filled or never freed: nothing to reset");
        assert!(!reusable(0, Some(100), 100 + REUSE_PRESENTS - 1), "released a moment ago");
        assert!(reusable(0, Some(100), 100 + REUSE_PRESENTS));
    }

    #[test]
    fn alloc_key_caches_valid_entries_and_skips_invalid_sentinel() {
        use super::{AllocKey, cache_alloc_info, cached_alloc_info};
        let key = AllocKey {
            dimension: 3,
            width: 777_001,
            height: 256,
            depth_or_array: 1,
            mip_levels: 4,
            format: 72,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 0,
        };
        assert_eq!(cached_alloc_info(key), None);
        cache_alloc_info(key, u64::MAX, 4096);
        assert_eq!(cached_alloc_info(key), None, "u64::MAX sentinel is not cached");
        cache_alloc_info(key, 65536, 4096);
        assert_eq!(cached_alloc_info(key), Some((65536, 4096)));
    }

    #[test]
    fn slot_reservation_uses_hint_and_wraps_around() {
        use super::SlotMeta;
        fn reserve(slots: &mut [SlotMeta], hint: &mut usize, size: u64, align: u64, now: u64) -> Option<(usize, u64)> {
            for slot in slots.iter_mut() {
                if slot.used != 0 && reusable(slot.live, slot.free_at, now) {
                    slot.used = 0;
                    slot.free_at = None;
                }
            }
            let len = slots.len();
            let start = (*hint).min(len - 1);
            for step in 0..len {
                let index = (start + step) % len;
                if let Some(at) = fit(slots[index].used, size, align) {
                    slots[index].used = at + size;
                    slots[index].live += 1;
                    slots[index].free_at = None;
                    *hint = index;
                    return Some((index, at));
                }
            }
            None
        }

        let mut slots = vec![
            SlotMeta { used: HEAP_BYTES, live: 0, free_at: Some(100) },
            SlotMeta { used: HEAP_BYTES - 4096, live: 1, free_at: None },
            SlotMeta { used: HEAP_BYTES, live: 1, free_at: None },
        ];
        let mut hint = 1;
        // Hint is at slot 1, which still has 4096 bytes left.
        assert_eq!(reserve(&mut slots, &mut hint, 4096, 4096, 100), Some((1, HEAP_BYTES - 4096)));
        assert_eq!(hint, 1);
        // Now slot 1 and 2 are full, slot 0 is not reusable yet at present 100.
        assert_eq!(reserve(&mut slots, &mut hint, 4096, 4096, 100), None);
        // At present 100 + REUSE_PRESENTS, slot 0 resets and wrap around from hint 1 finds slot 0.
        assert_eq!(reserve(&mut slots, &mut hint, 4096, 4096, 100 + REUSE_PRESENTS), Some((0, 0)));
        assert_eq!(hint, 0);
    }

    #[test]
    fn size_class_routes_small_and_large_textures() {
        use super::{SMALL_HEAP_BYTES, SMALL_TEXTURE_LIMIT, SizeClass, class_heap_bytes, fit_in, size_class_for};
        assert_eq!(size_class_for(4096), SizeClass::Small);
        assert_eq!(size_class_for(SMALL_TEXTURE_LIMIT), SizeClass::Small);
        assert_eq!(size_class_for(SMALL_TEXTURE_LIMIT + 4096), SizeClass::Large);
        assert_eq!(class_heap_bytes(SizeClass::Small), SMALL_HEAP_BYTES);
        assert_eq!(class_heap_bytes(SizeClass::Large), HEAP_BYTES);
        assert_eq!(fit_in(SMALL_HEAP_BYTES - 4096, 4096, 4096, SMALL_HEAP_BYTES), Some(SMALL_HEAP_BYTES - 4096));
        assert_eq!(fit_in(SMALL_HEAP_BYTES - 4096, 8192, 4096, SMALL_HEAP_BYTES), None);
    }

    #[test]
    fn excess_empty_heaps_keep_warm_pool_and_reclaim_rest() {
        use super::{SlotMeta, SizeClass, WARM_FREE_HEAPS_PER_CLASS};
        let mut slots = [
            Some((SizeClass::Small, SlotMeta { used: 4096, live: 0, free_at: Some(10) })),
            Some((SizeClass::Small, SlotMeta { used: 4096, live: 0, free_at: Some(10) })),
            Some((SizeClass::Small, SlotMeta { used: 4096, live: 0, free_at: Some(10) })),
            Some((SizeClass::Large, SlotMeta { used: 65536, live: 0, free_at: Some(10) })),
            Some((SizeClass::Large, SlotMeta { used: 65536, live: 0, free_at: Some(10) })),
            Some((SizeClass::Large, SlotMeta { used: 65536, live: 0, free_at: Some(10) })),
        ];
        let now = 10 + REUSE_PRESENTS;
        let mut warm = [0usize; 2];
        let mut reclaimed = Vec::new();
        for (idx, entry) in slots.iter_mut().enumerate() {
            let Some((class, meta)) = entry.as_mut() else { continue };
            if !reusable(meta.live, meta.free_at, now) {
                continue;
            }
            if warm[*class as usize] < WARM_FREE_HEAPS_PER_CLASS {
                warm[*class as usize] += 1;
                meta.used = 0;
                meta.free_at = None;
            } else {
                *entry = None;
                reclaimed.push(idx);
            }
        }
        assert_eq!(reclaimed, vec![1, 2, 4, 5], "excess empty heaps beyond warm pool per class are reclaimed");
        assert_eq!(slots.iter().flatten().count(), 2);
    }
}
