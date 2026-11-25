//! The view shadow: one row of four dwords per slot of a view heap, for the translated shaders of the
//! direct path. A raw view made from a structured one has `[record stride in bytes, record count, 0,
//! offset]`, one made from a typed view `[element size in bytes, element count, DXGI format,
//! offset]`, a raw view of the title's `[0, dwords, 0, offset]`, a texture view `[0, 0, size, shape]`
//! (`view_image`), any other view zeros. `offset` (0, 4, 8 or 12) is where the title's first byte
//! is in the PC view, which starts on 16 bytes (`view_raw`). Dword 2 is 0 and dword 0 not for a
//! structured view, the type bits of dword 3 (28 to 31) are 0 for a buffer and 8 to 15 for an
//! image.
//!
//! A console buffer descriptor carries its record stride and count, a PC view made raw
//! (`view_raw`) does not, and a program that cannot type a size query reads the size of a texture
//! here. The layer keeps the row of every slot of every view heap here and, for a shader visible
//! heap, in a raw buffer of upload memory that the shader reads through the root SRV `t1` of space
//! 15 ([`crate::rts0::VIEW_SHADOW_REGISTER`]), set before each draw and dispatch to the buffer of the
//! view heap the list has bound (`list_mirror`, with the heap map).
//!
//! Rows follow the descriptors: a view made in a slot writes its row ([`view_made`]), a copy of
//! descriptors copies the rows ([`copied`]). Both take host CPU handles: the heap and the slot are
//! found from the handle, so a copy between any two view heaps the layer knows carries its rows. The
//! rows of a heap that is not shader visible live in memory only, a copy into a shader visible heap
//! takes them along. Upload memory is written at once, like the descriptors themselves: the title
//! must not change a slot that work still in flight reads, on PC as on the console.
//!
//! The rows are kept for every view heap: a clear of a view made raw needs the format of the
//! title's view (`view_raw::clear_words`). The upload buffers and their binding exist only under
//! `DECANT_D3D12X_DIRECT` with the user data mirror ([`wanted`]), as the mirror's.

use std::sync::Mutex;

use crate::view_raw::Records;

/// The bytes of one row.
pub const ROW_BYTES: usize = 16;

/// One row: stride and record count of a raw view of records, the image words of a texture view.
pub type Row = [u32; 4];

/// The row of a raw view of `records`.
pub fn records_row(records: Records) -> Row {
    [records.stride, records.count, records.format, records.offset]
}

/// The rows of one view heap.
#[derive(Debug)]
struct HeapRows {
    /// The PC heap.
    heap: usize,
    cpu_start: usize,
    /// The GPU start of a shader visible heap, 0 for another.
    gpu_start: u64,
    increment: u32,
    rows: Vec<Row>,
    /// The view texture (`views_at::ViewTextures`) each slot's view was made of, 0 for none: a slot that
    /// holds one is a reference the texture must not be released under.
    owners: Vec<usize>,
    /// The upload buffer of a shader visible heap.
    buffer: Option<Buffer>,
}

/// The upload buffer of the rows of a shader visible heap: the PC resource (one reference the layer
/// owns), where it is mapped and its GPU address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub resource: usize,
    pub mapped: usize,
    pub gpu: u64,
    /// Bytes charged to the view-shadow owner.
    pub bytes: u64,
}

impl HeapRows {
    fn slot_of(&self, cpu: usize) -> Option<usize> {
        let off = cpu.checked_sub(self.cpu_start)?;
        let inc = self.increment as usize;
        let slot = off.checked_div(inc)?;
        (off % inc == 0 && slot < self.rows.len()).then_some(slot)
    }

    fn write(&mut self, slot: usize, row: Row) {
        self.rows[slot] = row;
        self.owners[slot] = 0;
        self.upload(slot..slot + 1);
    }

    fn upload(&self, range: std::ops::Range<usize>) {
        if let Some(b) = self.buffer {
            let offset = range.start * ROW_BYTES;
            let rows = &self.rows[range];
            // SAFETY: the buffer maps `ROW_BYTES` bytes per slot of the heap for as long as the heap
            // lives (made with the heap in `heap_created`, released with it).
            #[allow(unsafe_code)]
            unsafe {
                std::ptr::copy(rows.as_ptr(), (b.mapped + offset) as *mut Row, rows.len());
            }
        }
    }
}

/// Every view heap the layer made, by CPU start.
#[derive(Debug, Default)]
pub struct Shadows {
    heaps: Vec<HeapRows>,
}

impl Shadows {
    /// Registers a view heap of `count` slots. `buffer` maps `count` rows when the heap is shader
    /// visible (`gpu_start` not 0). A heap of the same PC object or CPU start is replaced.
    pub fn add(&mut self, heap: usize, cpu_start: usize, gpu_start: u64, count: u32, increment: u32, buffer: Option<Buffer>) {
        self.heaps.retain(|h| h.heap != heap && h.cpu_start != cpu_start);
        let at = self.heaps.partition_point(|h| h.cpu_start < cpu_start);
        self.heaps.insert(at, HeapRows { heap, cpu_start, gpu_start, increment, rows: vec![[0; 4]; count as usize], owners: vec![0; count as usize], buffer });
    }

    /// Forgets the heap `heap` and gives its buffer back to be released.
    pub fn remove(&mut self, heap: usize) -> Option<Buffer> {
        let at = self.heaps.iter().position(|h| h.heap == heap)?;
        self.heaps.remove(at).buffer
    }

    fn find(&self, cpu: usize) -> Option<(usize, usize)> {
        let at = self.heaps.partition_point(|h| h.cpu_start <= cpu).checked_sub(1)?;
        Some((at, self.heaps[at].slot_of(cpu)?))
    }

    /// The row of the slot at host CPU handle `cpu`, `None` when it is in no known heap.
    pub fn row(&self, cpu: usize) -> Option<Row> {
        let (h, s) = self.find(cpu)?;
        Some(self.heaps[h].rows[s])
    }

    /// A view was made in the slot at `cpu`: its row.
    pub fn view_made(&mut self, cpu: usize, row: Row) {
        if let Some((h, s)) = self.find(cpu) {
            self.heaps[h].write(s, row);
        }
    }

    /// `count` descriptors were copied from the slots at `src` to those at `dst`. A source slot in no
    /// known heap gives zeros.
    pub fn copied(&mut self, dst: usize, src: usize, count: u32) {
        if count == 0 {
            return;
        }
        let Some((dh, ds)) = self.find(dst) else { return };
        let inc = self.heaps[dh].increment as usize;
        let count = count as usize;
        let destination = ds..ds + count;
        if let Some((sh, ss)) = self.find(src)
            && self.heaps[sh].increment as usize == inc
            && count <= self.heaps[sh].rows.len() - ss
            && count <= self.heaps[dh].rows.len() - ds
        {
            let source = ss..ss + count;
            if sh == dh {
                self.heaps[dh].rows.copy_within(source.clone(), ds);
                self.heaps[dh].owners.copy_within(source, ds);
            } else {
                let [dst_heap, src_heap] = self.heaps.get_disjoint_mut([dh, sh]).expect("distinct heap indices");
                dst_heap.rows[destination.clone()].copy_from_slice(&src_heap.rows[source.clone()]);
                dst_heap.owners[destination.clone()].copy_from_slice(&src_heap.owners[source]);
            }
            self.heaps[dh].upload(destination);
            return;
        }

        // A range can leave a tracked heap or use another increment. Walk in address order
        // away from the overlap, as each destination is the same distance from its source.
        for k in 0..count {
            let offset = if dst > src { count - 1 - k } else { k } * inc;
            let row = self.row(src + offset).unwrap_or_default();
            let owner = self.owner(src + offset);
            self.view_made(dst + offset, row);
            self.set_owner(dst + offset, owner);
        }
    }

    /// The texture the slot at `cpu` holds a view of, 0 for none or outside the heaps.
    pub fn owner(&self, cpu: usize) -> usize {
        self.find(cpu).map_or(0, |(h, s)| self.heaps[h].owners[s])
    }

    /// The view just made in the slot at `cpu` (after `view_made`, which clears it) is of texture `res`.
    pub fn set_owner(&mut self, cpu: usize, res: usize) {
        if res != 0
            && let Some((h, s)) = self.find(cpu)
        {
            self.heaps[h].owners[s] = res;
        }
    }

    /// Every texture some slot of some view heap holds a view of.
    pub fn referenced(&self) -> std::collections::HashSet<usize> {
        self.heaps.iter().flat_map(|h| h.owners.iter().copied()).filter(|&r| r != 0).collect()
    }

    /// The buffer of the shader visible heap whose GPU start is `gpu_start`.
    pub fn buffer_at(&self, gpu_start: u64) -> Option<Buffer> {
        self.heaps.iter().find(|h| h.gpu_start == gpu_start && gpu_start != 0)?.buffer
    }
}

static SHADOWS: Mutex<Shadows> = Mutex::new(Shadows { heaps: Vec::new() });

fn shadows() -> std::sync::MutexGuard<'static, Shadows> {
    SHADOWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the layer gives the shadow to shaders: the direct path is on and the user data mirror is
/// made. The tests of the crate always do.
pub fn wanted() -> bool {
    cfg!(test) || (crate::compile_service::direct_enabled() && crate::user_mirror::wanted())
}

/// A view was made in host slot `cpu`: its row (zeros for a view that has none).
pub fn view_made(cpu: usize, row: Row) {
    let mut s = shadows();
    if !s.heaps.is_empty() {
        s.view_made(cpu, row);
    }
}

/// The view just made in host slot `cpu` is of the view texture `res`.
pub fn owner_made(cpu: usize, res: usize) {
    let mut s = shadows();
    if !s.heaps.is_empty() {
        s.set_owner(cpu, res);
    }
}

/// Every view texture some slot of some view heap still holds a view of (a scan of all the slots).
pub fn referenced() -> std::collections::HashSet<usize> {
    shadows().referenced()
}

/// `count` view descriptors were copied from host slot `src` on to host slot `dst`. To be called
/// after every copy of view descriptors the layer makes, with host CPU handles.
pub fn copied(dst: usize, src: usize, count: u32) {
    let mut s = shadows();
    if !s.heaps.is_empty() {
        s.copied(dst, src, count);
    }
}

/// The GPU address of the shadow buffer of the view heap at GPU start `gpu_start`.
pub fn buffer_gpu(gpu_start: u64) -> Option<u64> {
    shadows().buffer_at(gpu_start).map(|b| b.gpu)
}

/// The row of host slot `cpu`, `None` outside the view heaps.
pub fn row(cpu: usize) -> Option<Row> {
    shadows().row(cpu)
}

#[allow(unsafe_code)]
mod win {
    use super::{Buffer, ROW_BYTES, shadows};
    use crate::log::xs_log;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    /// An upload buffer of `rows` rows, mapped. Upload memory starts zeroed.
    fn make_buffer(device: usize, rows: u32) -> Option<Buffer> {
        let raw = device as *mut std::ffi::c_void;
        // SAFETY: `device` is the live PC device that made the heap.
        let device = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?;
        let bytes = (rows as u64 * ROW_BYTES as u64).max(256);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            Alignment: 0,
        };
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        // SAFETY: plain D3D12 calls; the layer keeps the reference and the mapping until the heap goes.
        unsafe {
            let mut res: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut res).ok()?;
            let res = res?;
            let mut mapped = std::ptr::null_mut();
            res.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut mapped)).ok()?;
            let gpu = res.GetGPUVirtualAddress();
            crate::mem_owners::add(
                crate::mem_owners::Kind::ViewShadow,
                crate::mem_owners::Segment::Unknown,
                1,
                bytes,
            );
            Some(Buffer { resource: res.into_raw() as usize, mapped: mapped as usize, gpu, bytes })
        }
    }

    /// A view heap was made: `heap` the PC heap of `count` slots of `increment` bytes from host CPU
    /// handle `cpu_start`, `gpu_start` its GPU start when shader visible (else 0), `device` the PC
    /// device. A shader visible heap gets its buffer.
    pub fn heap_created(device: usize, heap: usize, cpu_start: usize, gpu_start: u64, count: u32, increment: u32) {
        if count == 0 || increment == 0 {
            return;
        }
        let buffer = if gpu_start == 0 || !super::wanted() {
            None
        } else {
            let b = make_buffer(device, count);
            match b {
                Some(b) => xs_log!("view shadow: {count} rows at {:#x} for the view heap at GPU {gpu_start:#x}", b.gpu),
                None => xs_log!("view shadow: no upload buffer of {count} rows for the view heap at GPU {gpu_start:#x}, its shadow parameter reads address 0"),
            }
            b
        };
        shadows().add(heap, cpu_start, gpu_start, count, increment, buffer);
    }

    /// The PC heap `heap` is gone, and with it its rows and buffer.
    pub fn heap_released(heap: usize) {
        let buffer = shadows().remove(heap);
        if let Some(b) = buffer {
            crate::mem_owners::remove(
                crate::mem_owners::Kind::ViewShadow,
                crate::mem_owners::Segment::Unknown,
                1,
                b.bytes,
            );
            // SAFETY: the layer owns this one reference of the buffer (`make_buffer`).
            drop(unsafe { ID3D12Resource::from_raw(b.resource as *mut std::ffi::c_void) });
        }
    }
}

pub use win::{heap_created, heap_released};

#[cfg(test)]
mod tests {
    use super::{Buffer, Records, Row, Shadows, records_row};

    const INC: u32 = 32;

    fn rec(stride: u32, count: u32) -> Row {
        records_row(Records { stride, count, format: 0, offset: 0 })
    }

    #[test]
    fn a_slot_holds_its_texture_until_another_view_is_made_and_a_copy_carries_it() {
        let mut s = Shadows::default();
        s.add(0xA, 0x1000, 0, 8, INC, None);
        s.add(0xB, 0x9000, 0, 8, INC, None);
        s.view_made(0x1000, rec(1, 1));
        s.set_owner(0x1000, 0x77);
        assert_eq!(s.owner(0x1000), 0x77);
        s.copied(0x9000 + 32, 0x1000, 1);
        assert_eq!(s.owner(0x9000 + 32), 0x77, "a copy into another heap holds the texture too");
        s.view_made(0x1000, rec(2, 2));
        assert_eq!(s.owner(0x1000), 0, "a new view in the slot lets go of it");
        assert!(s.referenced().contains(&0x77), "the copy still holds it");
        s.view_made(0x9000 + 32, [0; 4]);
        assert!(s.referenced().is_empty());
    }

    #[test]
    fn a_view_writes_its_row_and_a_copy_carries_rows_between_heaps() {
        let mut memory = vec![0u32; 8 * 4];
        let mut s = Shadows::default();
        s.add(0xA, 0x1000, 0, 4, INC, None);
        s.add(0xB, 0x9000, 0x7_0000, 8, INC, Some(Buffer { resource: 0, mapped: memory.as_mut_ptr() as usize, gpu: 0x5_0000, bytes: 0 }));
        s.view_made(0x1000 + 32, rec(16, 10));
        s.view_made(0x1000 + 64, rec(12, 3));
        assert_eq!(s.row(0x1000 + 32), Some(rec(16, 10)));
        s.copied(0x9000 + 2 * 32, 0x1000 + 32, 2);
        assert_eq!((s.row(0x9000 + 64), s.row(0x9000 + 96)), (Some(rec(16, 10)), Some(rec(12, 3))));
        assert_eq!(&memory[8..16], &[16, 10, 0, 0, 12, 3, 0, 0], "the upload buffer holds the rows of slots 2 and 3");
        // a view without records clears the row, a copy from an unknown slot too
        s.view_made(0x9000 + 64, rec(0, 0));
        s.copied(0x9000 + 96, 0x4_0000, 1);
        assert_eq!(&memory[8..16], &[0; 8]);
        // a texture view's image words travel the same way
        s.view_made(0x1000, [0, 0, 0x000F_001F, 0x9001_0000]);
        s.copied(0x9000 + 32, 0x1000, 1);
        assert_eq!(&memory[4..8], &[0, 0, 0x000F_001F, 0x9001_0000]);
        assert_eq!(s.buffer_at(0x7_0000).map(|b| b.gpu), Some(0x5_0000));
        assert_eq!(s.buffer_at(0), None);
    }

    #[test]
    fn handles_outside_the_heaps_or_between_slots_are_ignored() {
        let mut s = Shadows::default();
        s.add(0xA, 0x1000, 0, 4, INC, None);
        s.view_made(0x1000 + 4 * 32, rec(4, 1));
        s.view_made(0x1000 + 8, rec(4, 1));
        s.view_made(0x10, rec(4, 1));
        assert_eq!(s.row(0x1000 + 4 * 32), None);
        assert_eq!(s.row(0x1000 + 8), None);
        assert!((0..4).all(|k| s.row(0x1000 + k * 32) == Some(rec(0, 0))));
        s.copied(0x10, 0x1000, 1);
        assert_eq!(s.remove(0xA), None);
        assert_eq!(s.row(0x1000), None);
    }

    #[test]
    fn an_overlapping_copy_reads_its_sources_first() {
        let mut s = Shadows::default();
        s.add(0xA, 0x1000, 0, 4, INC, None);
        s.view_made(0x1000, rec(4, 1));
        s.view_made(0x1000 + 32, rec(8, 2));
        s.copied(0x1000 + 32, 0x1000, 2);
        assert_eq!((s.row(0x1000 + 32), s.row(0x1000 + 64)), (Some(rec(4, 1)), Some(rec(8, 2))));
    }

    #[test]
    fn copies_match_a_source_snapshot_across_heap_ends_and_overlaps() {
        let inc = INC as usize;
        let handles: Vec<usize> = (0..18).map(|k| 0x1000 + k * inc).collect();
        for &src in &handles {
            for &dst in &handles {
                for count in 0..=9 {
                    let mut mapped = [[0; 4]; 5];
                    let mut s = Shadows::default();
                    s.add(1, handles[0], 0, 4, INC, None);
                    s.add(2, handles[4], 1, 5, INC, Some(Buffer { resource: 0, mapped: mapped.as_mut_ptr() as usize, gpu: 1, bytes: 0 }));
                    s.add(3, handles[11], 0, 4, INC * 2, None);
                    for (k, &handle) in handles.iter().enumerate() {
                        s.view_made(handle, [k as u32 + 1; 4]);
                    }
                    let mut expected: Vec<_> = handles.iter().map(|&handle| s.row(handle)).collect();
                    let original = expected.clone();
                    let dst_index = (dst - handles[0]) / inc;
                    let step = if dst >= handles[11] { 2 } else { 1 };
                    if original[dst_index].is_some() {
                        for k in 0..count as usize {
                            let to = dst_index + k * step;
                            let from = (src - handles[0]) / inc + k * step;
                            if let Some(Some(row)) = expected.get_mut(to) {
                                *row = original.get(from).copied().flatten().unwrap_or_default();
                            }
                        }
                    }
                    s.copied(dst, src, count);
                    for (&handle, row) in handles.iter().zip(&expected) {
                        assert_eq!(&s.row(handle), row, "src {src:#x}, dst {dst:#x}, count {count}");
                    }
                    assert!(mapped.iter().zip(&expected[4..9]).all(|(row, expected)| Some(*row) == *expected));
                }
            }
        }
    }
}
