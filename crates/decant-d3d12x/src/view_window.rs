//! Buffer views use byte offsets before raw conversion so structure strides do not constrain addresses.
//! Distant views use reserved buffers over the same mirror pages. Cached resources keep their pages
//! and source resource alive for descriptor lifetime. Existing descriptors retain their page mapping
//! when title pages are remapped, as do the placed buffers made by `title_mem::place_buffer`.

#![allow(unsafe_code)]

use crate::view_raw::{Outcome, Records, ViewKind};

const PAGE: u64 = 65536;
const RAW_REACH: u64 = u32::MAX as u64 * 4;

fn fields(kind: ViewKind) -> usize {
    match kind {
        ViewKind::ShaderResource => 16,
        ViewKind::UnorderedAccess => 8,
    }
}

fn word(desc: &[u8; 40], at: usize) -> u32 {
    u32::from_le_bytes(desc[at..at + 4].try_into().unwrap_or_default())
}

struct Plan {
    desc: [u8; 40],
    records: Records,
    start: u64,
    bytes: u64,
}

impl Plan {
    fn new(mut desc: [u8; 40], kind: ViewKind, offset: u64) -> Option<Self> {
        let at = fields(kind);
        let stride = word(&desc, at + 12);
        let element = if word(&desc, 32) & 1 != 0 { 4 } else if stride != 0 { stride } else { crate::views_at::format_bytes(word(&desc, 0)) };
        let first = u64::from_le_bytes(desc[at..at + 8].try_into().ok()?);
        let start = first.checked_mul(u64::from(element))?.checked_add(offset)?;
        let bytes = u64::from(word(&desc, at + 8)).checked_mul(u64::from(element))?;
        start.checked_add(bytes)?;
        desc[at..at + 8].fill(0);
        let Outcome::Raw(records) = crate::view_raw::make_raw(&mut desc, kind) else { return None };
        Some(Self { desc, records, start, bytes })
    }

    fn at(&self, kind: ViewKind, start: u64) -> Option<([u8; 40], Records)> {
        let lead = start % 16;
        let count = u32::try_from(self.bytes.checked_add(lead)?.div_ceil(4)).ok()?;
        if count == 0 || start.checked_add(self.bytes)?.div_ceil(4) > RAW_REACH / 4 {
            return None;
        }
        let mut desc = self.desc;
        let at = fields(kind);
        desc[at..at + 8].copy_from_slice(&((start - lead) / 4).to_le_bytes());
        desc[at + 8..at + 12].copy_from_slice(&count.to_le_bytes());
        Some((desc, Records { offset: lead as u32, ..self.records }))
    }

    fn window(&self, va: u64, offset: u64) -> Option<(u64, u64, u64)> {
        let target = va.checked_add(self.start.checked_sub(offset)?)?;
        let base = target / PAGE * PAGE;
        let local = target - base;
        let width = local.checked_add(self.bytes)?.div_ceil(PAGE).checked_mul(PAGE)?;
        base.checked_add(width)?;
        Some((base, width, local))
    }
}

/// Makes a raw descriptor at the exact byte address and retains the original record metadata.
/// Returns no result if the range cannot be represented or its mirror pages are unavailable.
///
/// # Safety
/// `source` is a live PC buffer resource for the device used by `title_mem`.
pub unsafe fn prepare(source: usize, va: u64, offset: u64, desc: [u8; 40], kind: ViewKind) -> Option<(usize, [u8; 40], Records)> {
    use std::collections::HashMap;
    use std::hash::{BuildHasherDefault, Hasher};
    use std::sync::{OnceLock, RwLock};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;

    #[derive(Default)]
    struct FastHasher(u64);

    impl Hasher for FastHasher {
        #[inline]
        fn finish(&self) -> u64 {
            self.0
        }

        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.rotate_left(5) ^ u64::from(b);
                self.0 = self.0.wrapping_mul(0x517C_C1B7_2722_0A95);
            }
        }

        #[inline]
        fn write_usize(&mut self, i: usize) {
            self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }

        #[inline]
        fn write_u32(&mut self, i: u32) {
            self.0 = (self.0 ^ u64::from(i)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.0 = (self.0 ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

    struct Cached {
        _source: ID3D12Resource,
        resource: ID3D12Resource,
    }
    type Key = (usize, u64, u64, Vec<u32>);
    static CACHE: OnceLock<RwLock<FastMap<Key, Cached>>> = OnceLock::new();
    let plan = Plan::new(desc, kind, offset)?;
    if let Some((desc, records)) = plan.at(kind, plan.start) {
        return Some((source, desc, records));
    }
    let (base, width, local) = plan.window(va, offset)?;
    let (desc, records) = plan.at(kind, local)?;
    let pages = crate::title_mem::tiles_of(base, width)?;
    let key = (source, base, width, pages);
    let cache_lock = CACHE.get_or_init(|| RwLock::new(FastMap::default()));
    {
        let cache = cache_lock.read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = cache.get(&key) {
            return Some((entry.resource.as_raw() as usize, desc, records));
        }
    }
    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let resource = crate::title_mem::place_buffer(base, &resource_desc, D3D12_RESOURCE_STATE_COMMON)?;
    // A mapping change during creation must not associate the new resource with the old pages.
    if crate::title_mem::tiles_of(base, width)? != key.3 {
        return None;
    }
    let mut cache = cache_lock.write().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = cache.get(&key) {
        return Some((entry.resource.as_raw() as usize, desc, records));
    }
    let result = resource.as_raw() as usize;
    // SAFETY: the caller supplies a live COM resource. Cloning owns a reference in the cache.
    let source_ptr = source as *mut core::ffi::c_void;
    let source = unsafe { ID3D12Resource::from_raw_borrowed(&source_ptr) }?.clone();
    cache.insert(key, Cached { _source: source, resource });
    use std::sync::atomic::{AtomicU64, Ordering};
    static MADE: AtomicU64 = AtomicU64::new(0);
    let made = MADE.fetch_add(1, Ordering::Relaxed) + 1;
    if made <= 8 || made.is_power_of_two() {
        crate::log::xs_log!("buffer view window: VA {va:016X}, base {base:016X}, bytes {width:#x}, local offset {local:#x}, {made} windows created");
    }
    Some((result, desc, records))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structured(kind: ViewKind, first: u64, count: u32, stride: u32) -> [u8; 40] {
        let mut desc = [0; 40];
        desc[4..8].copy_from_slice(&1u32.to_le_bytes());
        let at = fields(kind);
        desc[at..at + 8].copy_from_slice(&first.to_le_bytes());
        desc[at + 8..at + 12].copy_from_slice(&count.to_le_bytes());
        desc[at + 12..at + 16].copy_from_slice(&stride.to_le_bytes());
        desc
    }

    #[test]
    fn unaligned_structures_keep_bytes_and_metadata_for_both_views() {
        for kind in [ViewKind::ShaderResource, ViewKind::UnorderedAccess] {
            for offset in [4, 16, 65540] {
                let plan = Plan::new(structured(kind, 3, 7, 12), kind, offset).unwrap();
                let (desc, records) = plan.at(kind, plan.start).unwrap();
                let at = fields(kind);
                let first = u64::from_le_bytes(desc[at..at + 8].try_into().unwrap());
                assert_eq!(first * 4 + u64::from(records.offset), offset + 36);
                assert_eq!((records.stride, records.count, records.format), (12, 7, 0));
                assert!(u64::from(word(&desc, at + 8)) * 4 >= 84 + u64::from(records.offset));
            }
        }
    }

    #[test]
    fn distant_views_rebase_the_effective_address_including_first_element() {
        for kind in [ViewKind::ShaderResource, ViewKind::UnorderedAccess] {
            let offset = 48 << 30;
            let plan = Plan::new(structured(kind, 11, 17, 20), kind, offset).unwrap();
            assert!(plan.at(kind, plan.start).is_none());
            let va = 0x8000_0000_0004;
            let (base, width, local) = plan.window(va, offset).unwrap();
            assert_eq!(base + local, va + 220);
            assert_eq!(base % PAGE, 0);
            assert_eq!(width % PAGE, 0);
            assert!(local + 340 <= width);
            assert!(plan.at(kind, local).is_some());
        }
    }

    #[test]
    fn raw_and_small_typed_views_preserve_the_exact_first_byte() {
        for kind in [ViewKind::ShaderResource, ViewKind::UnorderedAccess] {
            for (format, flags, element) in [(39u32, 1u32, 4u64), (61, 0, 1), (57, 0, 2)] {
                let mut desc = structured(kind, 7, 9, 0);
                desc[..4].copy_from_slice(&format.to_le_bytes());
                desc[32..36].copy_from_slice(&flags.to_le_bytes());
                let plan = Plan::new(desc, kind, 3).unwrap();
                let (raw, records) = plan.at(kind, plan.start).unwrap();
                let at = fields(kind);
                let first = u64::from_le_bytes(raw[at..at + 8].try_into().unwrap());
                assert_eq!(first * 4 + u64::from(records.offset), 3 + 7 * element);
                assert_eq!(records.count, 9);
                assert_eq!(word(&raw, at + 12), 0);
                assert_eq!(word(&raw, 32) & 1, 1);
            }
        }
    }

    #[test]
    fn large_first_elements_and_crossing_pages_choose_a_covering_window() {
        let kind = ViewKind::UnorderedAccess;
        let plan = Plan::new(structured(kind, 1 << 32, 9, 16), kind, 0).unwrap();
        assert!(plan.at(kind, plan.start).is_none());
        let (base, width, local) = plan.window(0x1000_fff8, 0).unwrap();
        assert_eq!(base + local, 0x1000_fff8 + (1 << 36));
        assert_eq!(width, 2 * PAGE);
        let (desc, records) = plan.at(kind, local).unwrap();
        assert_eq!(records.offset, 8);
        assert_eq!(word(&desc, fields(kind) + 8), 38);
    }

    #[test]
    fn unaligned_views_at_addresses_read_and_write_on_the_gpu() {
        use crate::view_gpu_tests::*;
        use windows::core::Interface;
        type CreateAt = unsafe extern "system" fn(*mut core::ffi::c_void, *const core::ffi::c_void, *const u8, usize);
        for (name, device) in devices() {
            let dev = console_device(&device);
            let heap = ConsoleHeap::new(dev, &device, 2, true);
            let data = HostBuffer::new(&device, 64);
            let target = HostBuffer::new(&device, 64);
            data.fill(|i| 0x1000 + i as u32);
            target.fill(|_| 0);
            let source_va = 0x7000_0010_0000;
            let target_va = 0x7000_0020_0000;
            crate::views_at::record_resource(source_va, 256, data.res.as_raw() as usize, true);
            crate::views_at::record_resource(target_va, 256, target.res.as_raw() as usize, true);
            for (kind, slot, va, cpu) in [
                (ViewKind::ShaderResource, 70, source_va + 20, heap.cpu_at(0)),
                (ViewKind::UnorderedAccess, 71, target_va + 20, heap.cpu_at(1)),
            ] {
                let mut desc = [0; 48];
                desc[..40].copy_from_slice(&structured(kind, 2, 3, 12));
                desc[40..].copy_from_slice(&va.to_le_bytes());
                // SAFETY: console slots 70 and 71 take these descriptions and a live heap slot.
                unsafe { vt::<CreateAt>(dev, slot)(dev, core::ptr::null(), desc.as_ptr(), cpu) };
                let host = crate::title_heap::translate_cpu(cpu).unwrap();
                let expected = crate::view_shadow::records_row(Records { stride: 12, count: 3, format: 0, offset: 12 });
                assert_eq!(crate::view_shadow::row(host), Some(expected), "{name}: record metadata keeps the byte offset");
            }
            let out = HostBuffer::new(&device, 64);
            let words = read_write(&device, &heap, 0, &out);
            let expected: Vec<u32> = (8..20).map(|i| 0x1000 + i).collect();
            assert_eq!(&words[..12], expected, "{name}: the raw SRV covers the requested bytes and alignment prefix");
            assert_eq!((words[62], words[63]), (48, 48));
            let written = target.words();
            assert!(written[..8].iter().all(|&word| word == 0));
            for (i, &word) in written[8..20].iter().enumerate() {
                assert_eq!(word, 0xA000 + i as u32, "{name}: the raw UAV writes its byte range");
            }
            assert!(written[20..].iter().all(|&word| word == 0));
            crate::views_at::forget_resource(data.res.as_raw() as usize);
            crate::views_at::forget_resource(target.res.as_raw() as usize);
            drop(heap);
            release(dev);
        }
    }

    #[test]
    fn raw_reach_and_arithmetic_boundaries_are_checked() {
        let kind = ViewKind::ShaderResource;
        let plan = Plan::new(structured(kind, 0, 1, 4), kind, 0).unwrap();
        assert!(plan.at(kind, RAW_REACH - 4).is_some());
        assert!(plan.at(kind, RAW_REACH).is_none());
        assert!(plan.window(u64::MAX, 0).is_none());
        assert!(Plan::new(structured(kind, u64::MAX, 1, 16), kind, 0).is_none());
        assert!(Plan::new(structured(kind, 0, 0, 16), kind, 0).is_none());
        assert!(Plan::new(structured(kind, 0, 1, 3), kind, 0).is_none());
    }
}
