//! Console command list methods of the newer layout that the PC list has no equal for, and the
//! methods that carry upload memory addresses (list slot 148).
//!
//! Slots and argument lists are the measured or inferred ones documented in `abi::LIST_RULES`. A
//! method whose slot means something else in the older layout checks `abi::newer_layout()` first.

use super::*;

type RootVa = unsafe extern "system" fn(Object, u32, u64);
type SetVertexBuffers = unsafe extern "system" fn(Object, u32, u32, *const [u64; 2]);
type SetIndexBuffer = unsafe extern "system" fn(Object, *const [u8; 16]);
type GetDevice = unsafe extern "system" fn(Object, *const GUID, *mut Object) -> HRESULT;
type MapResource = unsafe extern "system" fn(Object, u32, *const c_void, *mut *mut c_void) -> HRESULT;

const IID_DEVICE: GUID = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);
const IID_RESOURCE: GUID = GUID::from_u128(0x696442be_a72e_4059_bc79_5b5c98040fad);

const HEAP_TYPE_UPLOAD: u32 = 2;
const STATE_GENERIC_READ: u32 = 0xAC3;

/// A buffer of `bytes` bytes in an upload heap, mapped: `(resource, CPU address, GPU address)`. The
/// caller owns the reference to the resource.
///
/// # Safety
/// `list` is a live PC command list.
pub(super) unsafe fn new_upload_buffer(list: Object, bytes: u64) -> Option<(Object, usize, u64)> {
    // SAFETY: by the contract above, and the PC device answers the slots used here.
    unsafe {
        let mut dev: Object = std::ptr::null_mut();
        if method::<GetDevice>(list, 7)(list, &IID_DEVICE, &mut dev) < 0 || dev.is_null() {
            return None;
        }
        let props: [u32; 5] = [HEAP_TYPE_UPLOAD, 0, 0, 0, 0];
        // D3D12_RESOURCE_DESC of a buffer: dimension 1, width, height 1, depth 1, mips 1, one
        // sample, layout ROW_MAJOR (1)
        let mut desc = [0u8; 0x38];
        desc[0..4].copy_from_slice(&1u32.to_le_bytes());
        desc[0x10..0x18].copy_from_slice(&bytes.to_le_bytes());
        desc[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        desc[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        desc[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
        desc[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        desc[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
        let mut res: Object = std::ptr::null_mut();
        let hr = method::<CreateCommittedResource>(dev, 27)(dev, props.as_ptr().cast(), 0, desc.as_ptr().cast(), STATE_GENERIC_READ, std::ptr::null(), &IID_RESOURCE, &mut res);
        method::<Release>(dev, 2)(dev);
        if hr < 0 || res.is_null() {
            return None;
        }
        let mut cpu: *mut c_void = std::ptr::null_mut();
        if method::<MapResource>(res, 8)(res, 0, std::ptr::null(), &mut cpu) < 0 || cpu.is_null() {
            method::<Release>(res, 2)(res);
            return None;
        }
        let gpu = method::<GpuVirtualAddress>(res, 11)(res);
        Some((res, cpu as usize, gpu))
    }
}

/// Console list slot 148: `AllocateUploadMemory(size, alignment)`. The title writes through the
/// returned pointer and passes the same number to the GPU as an address. On the PC the pointer is the
/// CPU address of a mapped upload buffer, and the list methods that take addresses translate it
/// (`upload_mem`). The memory lives until the command allocator of the list is reset. Null when no
/// buffer can be made.
pub(super) extern "system" fn list_allocate_upload_memory(this: *mut Proxy, size: usize, align: usize) -> *mut u8 {
    allocate_upload(this, size, align).map_or(std::ptr::null_mut(), |a| a as *mut u8)
}

/// [`list_allocate_upload_memory`] as a number, `None` when no buffer can be made. Also what the user
/// data mirror is written into.
pub(super) fn allocate_upload(this: *mut Proxy, size: usize, align: usize) -> Option<usize> {
    static N: AtomicI32 = AtomicI32::new(0);
    let (size, align) = crate::upload_mem::normalize(size, align);
    let (owner, bytes) = match crate::upload_mem::with(|m| {
        let owner = m.owner_of(this as usize);
        m.bump(owner, size, align).ok_or_else(|| (owner, m.chunk_bytes(owner, size, align)))
    }) {
        Ok(at) => return Some(at),
        Err(need) => need,
    };
    // SAFETY: `this` is a live list proxy.
    let made = unsafe { new_upload_buffer((*this).pc, bytes as u64) };
    let Some((res, cpu, gpu)) = made else {
        if log_due(&N) {
            xs_log!("console list slot 148 (AllocateUploadMemory {size:#x}, align {align:#x}): no upload buffer of {bytes:#x} bytes, answering null");
        }
        return None;
    };
    let at = crate::upload_mem::with(|m| {
        m.add_chunk(owner, crate::upload_mem::Chunk { res: res as usize, cpu, gpu, size: bytes, used: 0 });
        m.bump(owner, size, align)
    });
    crate::mem_owners::add_charge(crate::mem_owners::Charge {
        kind: crate::mem_owners::Kind::UploadChunk,
        segment: crate::mem_owners::Segment::Unknown,
        bytes: bytes as u64,
    });
    crate::upload_mem::note_used();
    if log_due(&N) {
        xs_log!("console list slot 148 (AllocateUploadMemory {size:#x}, align {align:#x}): new upload buffer {bytes:#x} bytes, CPU {cpu:#x}, GPU {gpu:#x}, owner {owner:#x}");
    }
    at
}

/// The PC address for an index or vertex view address: upload memory, title memory or a placed
/// buffer, else the zeroed fallback buffer (an address no PC resource holds is an invalid call on
/// native D3D12, which removes the device). 0 stays 0. Root views go through
/// `root_views`, which counts them by source.
///
/// # Safety
/// `list` is a live PC command list.
unsafe fn root_address(list: Object, va: u64) -> u64 {
    static N: AtomicI32 = AtomicI32::new(0);
    if va == 0 {
        return 0;
    }
    if let Some(pc_va) = super::pc_gpu_address(va, 4) {
        return pc_va;
    }
    if log_due(&N) {
        xs_log!("console list: index or vertex view at title address {va:#x} is in no PC resource, the zeroed fallback buffer is bound");
    }
    // SAFETY: by the contract above.
    unsafe { crate::fallback::address(list) }
}

macro_rules! root_va_method {
    ($name:ident, $pc:literal, $bind:expr, $va_kind:expr) => {
        pub(super) extern "system" fn $name(this: *mut Proxy, index: u32, va: u64) {
            // The user data mirror gets the console address, as the shader sees it.
            super::list_mirror::note_address(this, $bind, index, va);
            // SAFETY: `this` is a live list proxy; the PC slot takes (root index, address).
            unsafe {
                let pc = (*this).pc;
                let gpu = super::root_views::bind_address(pc, $pc, va);
                if $bind == crate::user_mirror::Bind::Compute {
                    crate::indirect_clamp::note_compute_va(this as usize, $va_kind, index, gpu);
                }
                method::<RootVa>(pc, $pc)(pc, index, gpu);
            }
        }
    };
}

root_va_method!(list_set_compute_root_cbv, 37, crate::user_mirror::Bind::Compute, crate::indirect_clamp::RootVaKind::Cbv);
root_va_method!(list_set_graphics_root_cbv, 38, crate::user_mirror::Bind::Graphics, crate::indirect_clamp::RootVaKind::Cbv);
root_va_method!(list_set_compute_root_srv, 39, crate::user_mirror::Bind::Compute, crate::indirect_clamp::RootVaKind::Srv);
root_va_method!(list_set_graphics_root_srv, 40, crate::user_mirror::Bind::Graphics, crate::indirect_clamp::RootVaKind::Srv);
root_va_method!(list_set_compute_root_uav, 41, crate::user_mirror::Bind::Compute, crate::indirect_clamp::RootVaKind::Uav);
root_va_method!(list_set_graphics_root_uav, 42, crate::user_mirror::Bind::Graphics, crate::indirect_clamp::RootVaKind::Uav);

/// The bytes of a `D3D12_INDEX_BUFFER_VIEW` with its address translated: upload memory, title
/// memory in xmem pages or a placed buffer. An address no PC resource holds stays as it was.
fn translated_index_view(view: &[u8; 16]) -> [u8; 16] {
    let mut v = *view;
    let va = u64::from_le_bytes(v[0..8].try_into().expect("eight bytes"));
    let size = u32::from_le_bytes(v[8..12].try_into().expect("four bytes"));
    let gpu = super::pc_gpu_address(va, u64::from(size).max(1)).unwrap_or(va);
    v[0..8].copy_from_slice(&gpu.to_le_bytes());
    v
}

/// Console list slot 47: `IASetIndexBuffer(view)` (PC 43). The view that reaches PC is remembered, so
/// that `DrawIndexedX` can put it back.
pub(super) extern "system" fn list_ia_set_index_buffer(this: *mut Proxy, view: *const [u8; 16]) {
    // SAFETY: `this` is a live list proxy, `view` is null or a readable view.
    unsafe {
        let pc = (*this).pc;
        if view.is_null() {
            crate::list_sync::set_index_view(this as usize, None);
            crate::draw_diag::index_view(this as usize, None, [0; 16]);
            method::<SetIndexBuffer>(pc, 43)(pc, view);
            return;
        }
        let title = view.read_unaligned();
        let mut v = translated_index_view(&title);
        // An address no PC resource holds gets the zeroed fallback buffer, its size capped to that
        // buffer, as a vertex buffer does: the PC GPU faults on the title address and the device
        // is lost (the debug layer names this view).
        let va = u64::from_le_bytes(title[0..8].try_into().expect("eight bytes"));
        if va != 0 && v[0..8] == title[0..8] {
            let size = u32::from_le_bytes(v[8..12].try_into().expect("four bytes"));
            v[0..8].copy_from_slice(&root_address(pc, va).to_le_bytes());
            v[8..12].copy_from_slice(&size.min(crate::fallback::BYTES as u32).to_le_bytes());
        }
        crate::list_sync::set_index_view(this as usize, Some(v));
        crate::draw_diag::index_view(this as usize, Some(title), v);
        method::<SetIndexBuffer>(pc, 43)(pc, &v);
    }
}

/// The most vertex buffer views that are translated: the D3D12 limit is 32.
const MAX_VIEWS: u32 = 32;

/// Console list slot 48: `IASetVertexBuffers(start, count, views)` (PC 44), the addresses of the
/// views translated.
pub(super) extern "system" fn list_ia_set_vertex_buffers(this: *mut Proxy, start: u32, count: u32, views: *const [u64; 2]) {
    // SAFETY: `this` is a live list proxy, `views` is null or points to `count` views.
    unsafe {
        let pc = (*this).pc;
        if views.is_null() || count == 0 || count > MAX_VIEWS {
            method::<SetVertexBuffers>(pc, 44)(pc, start, count, views);
            return;
        }
        let mut copy: Vec<[u64; 2]> = (0..count as usize).map(|i| views.add(i).read_unaligned()).collect();
        for v in &mut copy {
            // {BufferLocation u64, SizeInBytes u32, StrideInBytes u32}: an address no PC resource
            // holds gets the zeroed fallback buffer, its size capped to that buffer.
            let size = v[1] & 0xFFFF_FFFF;
            let translated = super::pc_gpu_address(v[0], size.max(1));
            v[0] = match translated {
                Some(pc_va) => pc_va,
                None if v[0] == 0 => 0,
                None => {
                    v[1] = (v[1] & !0xFFFF_FFFF) | size.min(crate::fallback::BYTES);
                    root_address(pc, v[0])
                }
            };
        }
        for (i, pc_view) in copy.iter().enumerate() {
            crate::draw_diag::vertex_view(this as usize, start + i as u32, views.add(i).read_unaligned(), *pc_view);
        }
        if start == 0 {
            let title = views.read_unaligned();
            let vb0 = crate::rect_list::Vb0 { va: title[0], size: title[1] as u32, stride: (title[1] >> 32) as u32, view: copy[0] };
            crate::rect_list::set_vb0(this as usize, Some(vb0));
        }
        method::<SetVertexBuffers>(pc, 44)(pc, start, count, copy.as_ptr());
    }
}

/// Draws `vertices` vertices of a rectangle list from vertex buffer `vb` as triangles: the vertices
/// with the fourth corner of each rectangle, from a copy in upload memory (`rect_list`). Vertex
/// buffer 0 is put back afterwards. False when the vertices cannot be read, then nothing is drawn.
pub(super) fn draw_rects(this: *mut Proxy, vb: crate::rect_list::Vb0, vertices: u32, instances: u32, start_vertex: u32, start_instance: u32) -> bool {
    type DrawInstanced = unsafe extern "system" fn(Object, u32, u32, u32, u32);
    static N: AtomicI32 = AtomicI32::new(0);
    let stride = vb.stride as usize;
    let first = start_vertex as usize * stride;
    let len = vertices as usize * stride;
    let va = vb.va as usize + first;
    if vb.va == 0 || first + len > vb.size as usize {
        return false;
    }
    // Title memory the CPU reads: xmem pages, a descriptor heap's memory or upload memory of slot 148.
    let readable = crate::title_mem::writable(va, len)
        || crate::title_heap::translate_cpu(va) != Some(va)
        || crate::upload_mem::translate(va as u64) != va as u64;
    if !readable {
        return false;
    }
    // SAFETY: `len` bytes at `va` are readable title memory, checked above.
    let bytes = unsafe { std::slice::from_raw_parts(va as *const u8, len) };
    let Some(out) = crate::rect_list::expand(bytes, stride) else { return false };
    let Some(cpu) = allocate_upload(this, out.len(), 256) else { return false };
    let gpu = crate::upload_mem::translate(cpu as u64);
    if gpu == cpu as u64 {
        return false;
    }
    // SAFETY: `cpu` is a mapped block of `out.len()` bytes this list owns; `this` is a live list
    // proxy, PC 44 is IASetVertexBuffers and 12 DrawInstanced.
    unsafe {
        std::ptr::copy_nonoverlapping(out.as_ptr(), cpu as *mut u8, out.len());
        let pc = (*this).pc;
        let view = [gpu, (u64::from(vb.stride) << 32) | out.len() as u64];
        method::<SetVertexBuffers>(pc, 44)(pc, 0, 1, &view);
        let n = (out.len() / stride) as u32;
        method::<DrawInstanced>(pc, 12)(pc, n, instances, 0, start_instance);
        method::<SetVertexBuffers>(pc, 44)(pc, 0, 1, &vb.view);
    }
    if log_due(&N) {
        xs_log!("console list: rectangle list of {vertices} vertices at {:#x} drawn as {} triangle vertices", vb.va, out.len() / stride);
    }
    true
}

/// A DMA list method that is not known (slot 15 and up): answers S_OK and does nothing, logged.
pub(super) extern "system" fn dma_list_ok(this: *mut Proxy) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if log_due(&N) {
        xs_log!("console DMA list {}: a method past the copies was called, S_OK and nothing done", p(this.cast()));
    }
    S_OK
}

/// Console allocator slot 8: `Reset` (PC 8). Once the PC allocator is reset the upload memory the
/// lists took from it can go, after the GPU is past the work in flight (`retire`): the title resets
/// an allocator when its own fences say the work is done, which on the PC may be early.
pub(super) extern "system" fn allocator_reset(this: *mut Proxy) -> HRESULT {
    type AllocReset = unsafe extern "system" fn(Object) -> HRESULT;
    // SAFETY: `this` is a live allocator proxy.
    unsafe {
        let pc = (*this).pc;
        let hr = method::<AllocReset>(pc, 8)(pc);
        if hr >= 0 {
            for (res, bytes) in crate::upload_mem::with(|m| m.free_owner(pc as usize)) {
                // SAFETY: the chunk was charged when it was added and is released once.
                crate::retire::release_owned(
                    res as Object,
                    "upload buffer",
                    crate::mem_owners::Charge {
                        kind: crate::mem_owners::Kind::UploadChunk,
                        segment: crate::mem_owners::Segment::Unknown,
                        bytes: bytes as u64,
                    },
                );
            }
        }
        hr
    }
}
