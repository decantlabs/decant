//! The texture computer object (`docs/CONSOLE_LIBS.md`, section 1): a COM style object whose first 8 bytes point to a table of 12 function
//! pointers. Slot 0 is `AddRef`, there is no `QueryInterface`.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use decant_texture::xg::resource_tiling::{PackedMipInfo, SubresourceTiling};
use decant_texture::xg::{Dim, TextureLayout};
use decant_log::trace;

use crate::abi::{E_INVALIDARG, E_NOTIMPL, Hresult, S_OK, XgTileShape, hresult};

/// The object a title holds. The table pointer comes first, as for a C++ object with virtual methods.
#[repr(C)]
pub struct TextureComputer {
    vtable: *const Vtable,
    references: AtomicU32,
    texture: TextureLayout,
}

type This = *mut TextureComputer;

/// The table of 12 method pointers.
#[repr(transparent)]
pub struct Vtable([*const (); 12]);

// SAFETY: the table is read only and holds function addresses.
unsafe impl Sync for Vtable {}

static VTABLE: Vtable = Vtable([
    add_ref as *const (),
    release as *const (),
    get_resource_layout as *const (),
    get_resource_size_bytes as *const (),
    get_resource_base_alignment_bytes as *const (),
    get_mip_level_offset_bytes as *const (),
    get_texel_element_offset_bytes as *const (),
    get_texel_coordinate as *const (),
    copy_into_subresource as *const (),
    copy_from_subresource as *const (),
    get_resource_tiling as *const (),
    get_texture_view_descriptor as *const (),
]);

/// A new computer with a reference count of 1. The caller hands the pointer to the title.
pub fn create(texture: TextureLayout) -> *mut TextureComputer {
    Box::into_raw(Box::new(TextureComputer { vtable: &VTABLE, references: AtomicU32::new(1), texture }))
}

/// Reads the texture of a computer.
///
/// # Safety
/// `this` is a pointer returned by [`create`] that has not been released to zero.
pub unsafe fn texture<'a>(this: *const TextureComputer) -> &'a TextureLayout {
    // SAFETY: stated by the caller.
    unsafe { &(*this).texture }
}

unsafe extern "system" fn add_ref(this: This) -> u32 {
    // SAFETY: a live computer.
    unsafe { (*this).references.fetch_add(1, Ordering::SeqCst) + 1 }
}

unsafe extern "system" fn release(this: This) -> u32 {
    // SAFETY: a live computer. The last release frees the box `create` made.
    unsafe {
        let left = (*this).references.fetch_sub(1, Ordering::SeqCst) - 1;
        if left == 0 {
            drop(Box::from_raw(this));
        }
        left
    }
}

unsafe extern "system" fn get_resource_layout(this: This, out: *mut u8) -> Hresult {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a live computer, and `out` points to an `XG_RESOURCE_LAYOUT` of 0x1B40 bytes.
    unsafe {
        let record = (*this).texture.layout.to_record();
        std::ptr::copy_nonoverlapping(record.as_ptr(), out, record.len());
    }
    S_OK
}

unsafe extern "system" fn get_resource_size_bytes(this: This) -> u64 {
    // SAFETY: a live computer.
    unsafe { (*this).texture.layout.size_bytes }
}

unsafe extern "system" fn get_resource_base_alignment_bytes(this: This) -> u64 {
    // SAFETY: a live computer.
    unsafe { (*this).texture.layout.base_alignment_bytes }
}

unsafe extern "system" fn get_mip_level_offset_bytes(this: This, plane: u32, mip: u32) -> u64 {
    // SAFETY: a live computer.
    let t = unsafe { &(*this).texture };
    if plane != 0 {
        return 0;
    }
    match t.plane0().levels.get(mip as usize) {
        Some(level) => level.offset_bytes,
        None if t.dim() == Dim::D2 => 0,
        None => u64::MAX,
    }
}

unsafe extern "system" fn get_texel_element_offset_bytes(this: This, plane: u32, mip: u32, x: u64, y: u32, z_or_slice: u32, sample: u32, bit_offset: *mut u32) -> u64 {
    if !bit_offset.is_null() {
        // SAFETY: a non null pointer is a writable UINT.
        unsafe { bit_offset.write(0) };
    }
    if plane != 0 {
        return u64::MAX;
    }
    // SAFETY: a live computer.
    unsafe { (*this).texture.element_offset(mip, x, y, z_or_slice, sample) }.unwrap_or(u64::MAX)
}

unsafe extern "system" fn get_texel_coordinate(_this: This, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> Hresult {
    decant_log::calls::not_implemented("texture computer GetTexelCoordinate");
    trace(&format!("xg_xs: GetTexelCoordinate not implemented, arguments {a:#x} {b:#x} {c:#x} {d:#x} {e:#x} {f:#x}"));
    E_NOTIMPL
}

static TILING_CALLS: AtomicU32 = AtomicU32::new(0);

/// Slot 10, D3D12's `GetResourceTiling` over the 64 KiB tiles of plane 0: `(total tiles, packed mip info, tile shape in texels,
/// in: room / out: count of subresource tilings, first one asked, subresource tilings)`. Every output may be null.
unsafe extern "system" fn get_resource_tiling(
    this: This,
    total_tiles: *mut u32,
    packed: *mut PackedMipInfo,
    shape: *mut XgTileShape,
    count: *mut u32,
    first: u32,
    tilings: *mut SubresourceTiling,
) -> Hresult {
    // SAFETY: a live computer.
    let t = unsafe { &(*this).texture };
    let tiling = match t.resource_tiling() {
        Ok(tiling) => tiling,
        Err(e) => {
            if TILING_CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                trace(&format!("xg_xs: GetResourceTiling of a mode {} texture: {e:?}", t.swizzle_mode()));
            }
            return hresult(e);
        }
    };
    let available = (tiling.subresources.len() as u32).saturating_sub(first);
    // SAFETY: each output is null or room for its value, `tilings` has room for `*count` entries.
    let written = unsafe {
        if let Some(out) = total_tiles.as_mut() {
            *out = tiling.total_tiles;
        }
        if let Some(out) = packed.as_mut() {
            *out = tiling.packed;
        }
        if let Some(out) = shape.as_mut() {
            *out = XgTileShape { width: tiling.tile_shape.0, height: tiling.tile_shape.1, depth: tiling.tile_shape.2 };
        }
        match count.as_mut() {
            Some(n) => {
                let wanted = available.min(*n);
                if !tilings.is_null() {
                    let from = &tiling.subresources[first.min(tiling.subresources.len() as u32) as usize..][..wanted as usize];
                    std::ptr::copy_nonoverlapping(from.as_ptr(), tilings, from.len());
                }
                *n = wanted;
                wanted
            }
            None => 0,
        }
    };
    if TILING_CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
        let p = tiling.packed;
        trace(&format!(
            "xg_xs: GetResourceTiling mode {}: {} tiles, {} standard and {} packed mips, shape {:?}, {written} tilings from {first}",
            t.swizzle_mode(),
            tiling.total_tiles,
            p.standard_mips,
            p.packed_mips,
            tiling.tile_shape
        ));
    }
    S_OK
}

unsafe extern "system" fn get_texture_view_descriptor(_this: This, plane: u32, _out: *mut c_void) -> Hresult {
    decant_log::calls::not_implemented("texture computer GetTextureViewDescriptor");
    trace(&format!("xg_xs: GetTextureViewDescriptor not implemented, plane {plane}"));
    E_NOTIMPL
}

/// Which way a copy goes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Linear data into the resource memory.
    Into,
    /// The resource memory out to linear data.
    From,
}

/// Moves the elements of one 2D slice of a level between the resource memory and linear rows.
///
/// # Safety
/// `resource` and `linear` cover the whole resource and the rows of the level. Elements beyond the resource are skipped.
unsafe fn copy_slice(t: &TextureLayout, resource: *mut u8, linear: *mut u8, row_pitch: usize, mip: u32, z: u32, direction: Direction) {
    let level = &t.plane0().levels[mip as usize];
    let bpe = t.bytes_per_element() as usize;
    let size = t.layout.size_bytes;
    for row in 0..level.height_elements {
        for column in 0..level.width_elements {
            let Some(offset) = t.element_offset(mip, u64::from(column), row, z, 0) else { continue };
            if offset + bpe as u64 > size {
                continue;
            }
            // SAFETY: the element is inside the resource, and the linear row holds the element.
            unsafe {
                let tiled = resource.add(offset as usize);
                let flat = linear.add(row as usize * row_pitch + column as usize * bpe);
                if direction == Direction::Into {
                    std::ptr::copy_nonoverlapping(flat, tiled, bpe);
                } else {
                    std::ptr::copy_nonoverlapping(tiled, flat, bpe);
                }
            }
        }
    }
}

/// # Safety
/// The arguments are the ones of the two copy methods.
#[allow(clippy::too_many_arguments)]
unsafe fn copy(this: This, resource: *mut u8, plane: u32, subresource: u32, linear: *mut u8, row_pitch: u32, slice_pitch: u32, direction: Direction) -> Hresult {
    // SAFETY: a live computer.
    let t = unsafe { &(*this).texture };
    if plane != 0 || resource.is_null() || linear.is_null() {
        return E_INVALIDARG;
    }
    let mips = t.layout.mip_levels;
    let (mip, index) = (subresource % mips, subresource / mips);
    match t.dim() {
        Dim::D3 => {
            if direction == Direction::From || index >= t.array_or_depth() {
                return E_INVALIDARG;
            }
            let depth = t.plane0().levels[mip as usize].depth_or_array;
            for z in 0..depth {
                // SAFETY: slice `z` of the source starts `slice_pitch` bytes after the previous one.
                unsafe { copy_slice(t, resource, linear.add(z as usize * slice_pitch as usize), row_pitch as usize, mip, z, direction) };
            }
        }
        _ => {
            if index >= t.array_or_depth() {
                return E_INVALIDARG;
            }
            // SAFETY: stated by the caller.
            unsafe { copy_slice(t, resource, linear, row_pitch as usize, mip, index, direction) };
        }
    }
    S_OK
}

unsafe extern "system" fn copy_into_subresource(this: This, resource_base: *mut u8, plane: u32, subresource: u32, src: *const u8, row_pitch: u32, slice_pitch: u32) -> Hresult {
    // SAFETY: the caller's contract is passed on, the source is only read.
    unsafe { copy(this, resource_base, plane, subresource, src as *mut u8, row_pitch, slice_pitch, Direction::Into) }
}

unsafe extern "system" fn copy_from_subresource(this: This, dst: *mut u8, plane: u32, subresource: u32, resource_base: *const u8, row_pitch: u32, slice_pitch: u32) -> Hresult {
    // SAFETY: the caller's contract is passed on, the resource is only read.
    unsafe { copy(this, resource_base as *mut u8, plane, subresource, dst, row_pitch, slice_pitch, Direction::From) }
}
