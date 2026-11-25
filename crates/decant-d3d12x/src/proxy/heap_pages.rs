//! The PC side of a paged shader visible view heap (`title_heap::Pager`): the backing heap made
//! beside a heap the PC refuses at the title's size, the host handle of each table the title binds,
//! the copies from the backing heap into the host pages, and the page map buffer of every shader
//! visible view heap.

use super::*;
use crate::title_heap::{self, TableAt};

type MapResource = unsafe extern "system" fn(Object, u32, *const c_void, *mut *mut c_void) -> HRESULT;

const IID_RESOURCE: GUID = GUID::from_u128(0x696442be_a72e_4059_bc79_5b5c98040fad);

/// Presents between two lines of the paging counters.
const REPORT_PRESENTS: u64 = 128;

/// Copies `n` descriptors of views with the PC device's CopyDescriptorsSimple (slot 24).
pub(super) fn copy_views(device: usize, dst: usize, src: usize, n: u32) {
    // SAFETY: `device` is the live PC device of the paged heap, `dst` a CPU handle of its shader
    // visible heap and `src` one of its backing heap, both with `n` slots from there.
    unsafe { method::<CopyDescriptorsSimplePc>(device as Object, 24)(device as Object, n, dst, src, 0) };
    shadow_copied(dst, src, n);
}

/// The layer copied `n` views from host CPU handle `src` to `dst`: the view shadow rows follow.
pub(super) fn shadow_copied(dst: usize, src: usize, n: u32) {
    crate::view_shadow::copied(dst, src, n);
}

/// A CPU only view heap of `count` slots beside a paged heap: the object and its CPU start.
///
/// # Safety
/// `pc` is the live PC device, `riid` the interface the title asked for its heap.
pub(super) unsafe fn backing(pc: Object, count: u32, node_mask: u32, riid: *const GUID) -> Option<(Object, usize, crate::mem_owners::Charge)> {
    let desc: [u32; 4] = [0, count, 0, node_mask];
    let mut h: Object = std::ptr::null_mut();
    // SAFETY: by the contract above, `desc` is a D3D12_DESCRIPTOR_HEAP_DESC.
    let hr = unsafe { method::<CreateHeap>(pc, 14)(pc, desc.as_ptr().cast(), riid, &mut h) };
    if hr < 0 || h.is_null() {
        xs_log!("descriptor pages: the backing heap of {count} views was refused, hr=0x{:08X}; slots past the PC limit stay unreachable", hr as u32);
        return None;
    }
    let mut cpu = 0u64;
    // SAFETY: `h` is the heap just made, `pc` its device.
    let increment = unsafe {
        method::<HeapStart>(h, 9)(h, &mut cpu);
        method::<GetIncSize>(pc, 15)(pc, 0)
    };
    // The backing heap holds the title's slots: the view shadow keeps their rows (CPU only, no buffer).
    crate::view_shadow::heap_created(pc as usize, h as usize, cpu as usize, 0, count, increment);
    let charge = crate::mem_owners::Charge {
        kind: crate::mem_owners::Kind::DescriptorHeap,
        segment: crate::mem_owners::Segment::System,
        bytes: count as u64 * increment as u64,
    };
    crate::mem_owners::add_charge(charge);
    Some((h, cpu as usize, charge))
}

/// The descriptors the table at root parameter `index` of the root signature `bind` of `list`
/// has, [`title_heap::UNBOUNDED_EXTENT`] when unbounded or unknown.
fn extent(list: usize, bind: crate::user_mirror::Bind, index: u32) -> u32 {
    static UNBOUNDED: AtomicI32 = AtomicI32::new(0);
    let sig = if crate::user_mirror::active() { crate::user_mirror::with_shadow(list, |s| s.signature(bind)) } else { 0 };
    let ranges = crate::dxil_override::root_layout(sig).and_then(|l| match l.params.get(index as usize).map(|p| &p.kind) {
        Some(rts0::RootParamKind::Table(r)) => Some(r.iter().map(|r| (r.count, None)).collect::<Vec<_>>()),
        _ => None,
    });
    match ranges.map(title_heap::table_extent) {
        Some(Some(n)) => n,
        known => {
            if log_due(&UNBOUNDED) {
                let why = if known.is_some() { "is unbounded" } else { "has no known root signature" };
                xs_log!("descriptor pages: table {index} of root signature {sig:#x} {why}, its run in the shader visible heap is {} descriptors", title_heap::UNBOUNDED_EXTENT);
            }
            title_heap::UNBOUNDED_EXTENT
        }
    }
}

/// The PC handle of the table the title binds at `handle` (root parameter `index` of `bind`) on
/// `list`, `handle` itself outside a paged heap.
pub(super) fn table_handle(list: usize, bind: crate::user_mirror::Bind, index: u32, handle: u64) -> u64 {
    host_handle(list, handle, || extent(list, bind, index))
}

/// The PC handle of the descriptor at the title's GPU handle `handle`, read by `list` for
/// `extent()` descriptors.
pub(super) fn host_handle(list: usize, handle: u64, extent: impl FnOnce() -> u32) -> u64 {
    static EARLY: AtomicI32 = AtomicI32::new(0);
    static FULL: AtomicI32 = AtomicI32::new(0);
    if !title_heap::paged(handle) {
        return handle;
    }
    match title_heap::table(list, handle, extent(), &mut copy_views) {
        TableAt::Same => handle,
        TableAt::Host(h, early) => {
            if early && log_due(&EARLY) {
                xs_log!("descriptor pages: a host page was taken for {handle:#x} before {} presents had passed since its last use", title_heap::KEEP_PRESENTS);
            }
            h
        }
        TableAt::Full => {
            if log_due(&FULL) {
                xs_log!("descriptor pages: no free run in the shader visible heap for the table at {handle:#x}, bound unmapped");
            }
            handle
        }
    }
}

/// Copies the slots written since the last draw or dispatch into the host pages.
pub(super) fn flush() {
    title_heap::flush(&mut copy_views, &mut write_map);
}

/// Writes `entries` into the mapped page map at `cpu`.
fn write_map(cpu: usize, entries: &[u32]) {
    // SAFETY: `cpu` is the mapped memory of a page map buffer made for this many entries, which
    // stays mapped as long as its heap lives.
    unsafe { std::ptr::copy_nonoverlapping(entries.as_ptr(), cpu as *mut u32, entries.len()) };
}

/// A page map buffer with `entries`: an upload buffer, mapped for good, `(object, CPU address,
/// GPU address)`.
///
/// # Safety
/// `pc` is the live PC device.
pub(super) unsafe fn page_map(pc: Object, entries: &[u32]) -> Option<(Object, usize, u64, crate::mem_owners::Charge)> {
    const HEAP_TYPE_UPLOAD: u32 = 2;
    const STATE_GENERIC_READ: u32 = 0xAC3;
    let bytes = (entries.len() as u64 * 4).next_multiple_of(256);
    let props: [u32; 5] = [HEAP_TYPE_UPLOAD, 0, 0, 0, 0];
    // D3D12_RESOURCE_DESC of a buffer: dimension 1, width, height 1, depth 1, mips 1, one sample,
    // layout ROW_MAJOR (1).
    let mut desc = [0u8; 0x38];
    desc[0..4].copy_from_slice(&1u32.to_le_bytes());
    desc[0x10..0x18].copy_from_slice(&bytes.to_le_bytes());
    desc[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
    desc[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
    desc[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
    desc[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    desc[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
    let mut res: Object = std::ptr::null_mut();
    // SAFETY: by the contract above, with a heap properties and a resource description of the
    // public layout; the resource is mapped once and released with its heap.
    unsafe {
        let hr = method::<CreateCommittedResource>(pc, 27)(pc, props.as_ptr().cast(), 0, desc.as_ptr().cast(), STATE_GENERIC_READ, std::ptr::null(), &IID_RESOURCE, &mut res);
        if hr < 0 || res.is_null() {
            xs_log!("descriptor pages: the page map buffer of {bytes} bytes was refused, hr=0x{:08X}", hr as u32);
            return None;
        }
        let mut cpu: *mut c_void = std::ptr::null_mut();
        if method::<MapResource>(res, 8)(res, 0, std::ptr::null(), &mut cpu) < 0 || cpu.is_null() {
            method::<Release>(res, 2)(res);
            return None;
        }
        write_map(cpu as usize, entries);
        let charge = crate::mem_owners::Charge {
            kind: crate::mem_owners::Kind::DescriptorPageMap,
            segment: crate::mem_owners::Segment::Unknown,
            bytes,
        };
        crate::mem_owners::add_charge(charge);
        Some((res, cpu as usize, method::<GpuVirtualAddress>(res, 11)(res), charge))
    }
}

/// A present: counts it and logs the paging counters now and then.
pub(super) fn present() {
    let n = title_heap::present();
    if !n.is_multiple_of(REPORT_PRESENTS) {
        return;
    }
    let c = &crate::slot_tags::COUNTS;
    let [kept, elsewhere, copied, lost, again] = [&c.kept, &c.elsewhere, &c.again_copied, &c.lost, &c.again].map(|n| n.load(Ordering::Relaxed));
    if kept + elsewhere + copied + lost + again > 0 {
        xs_log!("slot tags at present {n}: {kept} views of earlier draws kept, {elsewhere} old tags of other lists or presents ({copied} copies made again), {lost} views lost before their draw, {again} moved tables read again without being set");
    }
    for s in title_heap::stats() {
        let highest = s.highest_written.map_or("-".to_owned(), |h| h.to_string());
        xs_log!(
            "descriptor pages at present {n}: highest console slot written {highest}, pages written {} ({} table, {} index), host pages {} of {} in use (peak {}), evictions {}, taken early {}, refused {}",
            s.written_pages, s.table_pages, s.index_pages, s.in_use, s.host_pages, s.peak, s.evictions, s.early, s.refused
        );
        if s.refused > 0 && s.in_use == s.host_pages {
            xs_log!("descriptor pages: the shader visible heap is full ({} index pages kept for good, {} table pages): the title has more live descriptors than the PC heap holds", s.index_pages, s.table_pages);
        }
    }
}
