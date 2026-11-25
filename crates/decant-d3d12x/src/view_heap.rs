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

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

/// Bytes of one heap.
pub const HEAP_BYTES: u64 = 64 << 20;
/// The small resource placement alignment of the PC runtime.
const SMALL: u64 = 4096;

/// Where the next placement of `size` bytes at `align` goes in a heap of [`HEAP_BYTES`] whose first
/// `used` bytes are taken: the offset, or `None` when it does not fit (pure, for the tests).
pub fn fit(used: u64, size: u64, align: u64) -> Option<u64> {
    let at = used.checked_next_multiple_of(align.max(1))?;
    (at.checked_add(size)? <= HEAP_BYTES).then_some(at)
}

/// Presents a heap whose textures were all released waits before it is filled again.
pub const REUSE_PRESENTS: u64 = 60;

/// Whether a heap with `live` textures placed in it, all released at present `free_at`, may be filled
/// again at present `now` (pure, for the tests).
pub fn reusable(live: u32, free_at: Option<u64>, now: u64) -> bool {
    live == 0 && free_at.is_some_and(|t| t + REUSE_PRESENTS <= now)
}

struct Heap {
    heap: ID3D12Heap,
    used: u64,
    /// Textures placed in it and not released.
    live: u32,
    /// The present at which `live` fell to 0.
    free_at: Option<u64>,
}

// SAFETY: D3D12 heaps are free threaded, and the list is only touched under its lock.
unsafe impl Send for Heap {}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// A texture of the PC description `desc` (after `abi::pc_resource_desc`) in state `state`, placed in
/// one of the heaps. `None` for a texture larger than a heap or when the runtime refuses: the caller
/// makes a committed one then.
///
/// Returns the texture, the index of its heap and its size in bytes.
pub fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, state: u32) -> Option<(ID3D12Resource, usize, u64)> {
    let mut d = *desc;
    d.Alignment = SMALL;
    // SAFETY: one valid description.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.Alignment != SMALL {
        d.Alignment = 0;
        // SAFETY: as above.
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX || info.SizeInBytes > HEAP_BYTES {
        return None;
    }
    let now = crate::title_heap::presents();
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    for h in heaps.iter_mut() {
        if h.used != 0 && reusable(h.live, h.free_at, now) {
            h.used = 0;
            h.free_at = None;
        }
    }
    let found = heaps.iter().position(|h| fit(h.used, info.SizeInBytes, info.Alignment).is_some());
    let index = match found {
        Some(i) => i,
        None => {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: HEAP_BYTES,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: a valid description and output.
            unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
            let heap = heap?;
            // SAFETY: a live heap, named for DRED.
            let _ = unsafe { heap.SetName(&windows::core::HSTRING::from(format!("decant view texture heap {}", heaps.len()))) };
            heaps.push(Heap { heap, used: 0, live: 0, free_at: None });
            crate::mem_owners::add(
                crate::mem_owners::Kind::ViewHeap,
                crate::mem_owners::Segment::Local,
                1,
                HEAP_BYTES,
            );
            heaps.len() - 1
        }
    };
    let h = &mut heaps[index];
    let at = fit(h.used, info.SizeInBytes, info.Alignment)?;
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: the heap is live, the range lies inside it, the description is valid.
    unsafe { device.CreatePlacedResource(&h.heap, at, &d, D3D12_RESOURCE_STATES(state as i32), None, &mut res) }.ok()?;
    h.used = at + info.SizeInBytes;
    h.live += 1;
    h.free_at = None;
    Some((res?, index, info.SizeInBytes))
}

/// A texture placed in heap `index` was released (its last reference is held back until the GPU is
/// done, `retire`). The heap is used again once all of its textures are gone.
pub fn released(index: usize) {
    let now = crate::title_heap::presents();
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.get_mut(index) {
        h.live = h.live.saturating_sub(1);
        if h.live == 0 {
            h.free_at = Some(now);
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
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    (heaps.len(), heaps.iter().map(|h| h.used).sum())
}

/// Heaps with no live texture: kept, and reusable once their wait is over.
pub fn empty_heaps() -> usize {
    HEAPS.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|h| h.live == 0).count()
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
}
