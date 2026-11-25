//! The 24 exports of `xg_xs.dll`, under the names listed in `docs/CONSOLE_LIBS.md`, section 1. All use the Windows x64
//! convention. The computations are in `decant_texture::xg`, the state in `state`, the object in `computer`.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use decant_texture::xg::d3d12::{D3d12ResourceDesc, texture_desc_from_d3d12};
use decant_texture::xg::descriptor::{image_format_element, parse_image_descriptor, texture_desc_from_descriptor, validate_descriptor};
use decant_texture::xg::modes::is_z_family;
use decant_texture::xg::tile_shape::compute_tile_shape;
use decant_texture::xg::{Dim, SwizzleQuery, TextureDesc, XgError, buffer_layout, compute_layout, compute_optimal_swizzle_mode};
use decant_log::trace;

use crate::abi::{E_FAIL, E_INVALIDARG, E_NOTIMPL, Hresult, S_OK, XgBufferDesc, XgTexture1dDesc, XgTexture2dDesc, XgTexture3dDesc, XgTileShape, hresult};
use crate::computer::{TextureComputer, create};
use crate::state::{self, Guard};

/// The exported names, in the order of the specification.
pub const EXPORT_NAMES: [&str; 24] = [
    "XGInitializeLibrary",
    "XGTerminateLibrary",
    "XGSetHardwareVersion",
    "XGEnableExclusiveLocking",
    "XGEnterExclusiveLock",
    "XGLeaveExclusiveLock",
    "XGComputeOptimalSwizzleMode",
    "XGComputeOptimalDepthStencilSwizzleMode",
    "XGValidateDepthStencilSwizzleMode",
    "XGComputeBufferLayout",
    "XGComputeTexture1DLayout",
    "XGComputeTexture2DLayout",
    "XGComputeTexture3DLayout",
    "XGCreateTexture1DComputer",
    "XGCreateTexture2DComputer",
    "XGCreateTexture3DComputer",
    "XGCreateTextureComputer",
    "XGCreateTextureComputerFromDescriptor",
    "XGValidateDescriptor",
    "XGComputeTileShape",
    "XGEncodeDCC",
    "XGDecodeDCC",
    "XGCreateBVHComputer",
    "XGSuggestSwizzleMode",
];

/// Logs a failure with the function that returned it.
fn failed(function: &str, error: XgError) -> Hresult {
    let code = hresult(error);
    trace(&format!("xg_xs: {function} failed with {code:#010x} ({error:?})"));
    code
}

#[unsafe(no_mangle)]
pub extern "system" fn XGInitializeLibrary() -> Hresult {
    state::initialize()
}

#[unsafe(no_mangle)]
pub extern "system" fn XGTerminateLibrary() -> Hresult {
    state::terminate()
}

#[unsafe(no_mangle)]
pub extern "system" fn XGSetHardwareVersion(version: u32) -> Hresult {
    trace(&format!("xg_xs: XGSetHardwareVersion({version})"));
    state::set_hardware_version(version)
}

/// Not an `xg_xs` export of the console: whether the title selected the version 4 equations (`XGSetHardwareVersion(4)`). The
/// D3D12 layer reads it by name to lay out the textures it detiles as the title laid them out.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXgHardwareV4() -> u32 {
    u32::from(state::hardware_v4())
}

#[unsafe(no_mangle)]
pub extern "system" fn XGEnableExclusiveLocking(enable: i32) {
    state::enable_locking(enable != 0);
}

#[unsafe(no_mangle)]
pub extern "system" fn XGEnterExclusiveLock() -> i32 {
    state::enter_lock()
}

#[unsafe(no_mangle)]
pub extern "system" fn XGLeaveExclusiveLock() {
    state::leave_lock();
}

#[unsafe(no_mangle)]
pub extern "system" fn XGComputeOptimalSwizzleMode(dimension: u32, format: u32, width: u64, height: u32, depth_or_array_size: u32, sample_count: u32, bind_flags: u32, misc_flags: u32) -> u32 {
    let _guard = Guard::new();
    compute_optimal_swizzle_mode(&SwizzleQuery { dimension, format, width, height, depth_or_array: depth_or_array_size, sample_count, bind_flags, misc_flags })
}

static DEPTH_STENCIL_CALLS: AtomicU32 = AtomicU32::new(0);

/// The eight arguments before the stencil pointer are not identified. They are logged on the first calls.
///
/// # Safety
/// `stencil_mode` is null or a writable `UINT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeOptimalDepthStencilSwizzleMode(a0: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize, a7: usize, stencil_mode: *mut u32) -> u32 {
    if DEPTH_STENCIL_CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
        trace(&format!("xg_xs: XGComputeOptimalDepthStencilSwizzleMode({a0:#x}, {a1:#x}, {a2:#x}, {a3:#x}, {a4:#x}, {a5:#x}, {a6:#x}, {a7:#x}, {stencil_mode:p})"));
    }
    if !stencil_mode.is_null() {
        // SAFETY: a non null pointer is a writable UINT.
        unsafe { stencil_mode.write(0x18) };
    }
    0x18
}

static SUGGEST_CALLS: AtomicU32 = AtomicU32::new(0);

/// `HRESULT XGSuggestSwizzleMode(const D3D12_RESOURCE_DESC* desc, UINT flags, UINT* layout)`: the explicit layout
/// (`0x100 | mode`) the library picks for the description with `Layout` 0. The title passes flags 0 and the `Layout`
/// field of the same description as `layout` (logged runs: the third argument is always the first plus 0x2C).
///
/// # Safety
/// `desc` and `layout` are null or point to a `D3D12_RESOURCE_DESC` and a writable `UINT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGSuggestSwizzleMode(desc: *const [u8; 0x38], flags: u32, layout: *mut u32) -> Hresult {
    if desc.is_null() || layout.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `desc` is a readable `D3D12_RESOURCE_DESC`.
    let mut resource = D3d12ResourceDesc::from_bytes(&unsafe { desc.read_unaligned() });
    resource.layout = 0;
    let mode = match texture_desc_from_d3d12(&resource) {
        Ok(d) => d.swizzle_mode,
        Err(e) => return failed("XGSuggestSwizzleMode", e),
    };
    if SUGGEST_CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
        trace(&format!(
            "xg_xs: XGSuggestSwizzleMode(dimension {}, {}x{}x{}, format {}, flags {:#x}, call flags {flags:#x}) = layout {:#x}",
            resource.dimension,
            resource.width,
            resource.height,
            resource.depth_or_array_size,
            resource.format,
            resource.flags,
            0x100 | mode
        ));
    }
    // SAFETY: `layout` is writable. It may be the `Layout` field of `desc`, which was read before.
    unsafe { layout.write_unaligned(0x100 | mode) };
    S_OK
}

static VALIDATE_CALLS: AtomicU32 = AtomicU32::new(0);

#[unsafe(no_mangle)]
pub extern "system" fn XGValidateDepthStencilSwizzleMode(a0: usize, a1: usize, mode: u32) -> Hresult {
    if VALIDATE_CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
        trace(&format!("xg_xs: XGValidateDepthStencilSwizzleMode({a0:#x}, {a1:#x}, {mode})"));
    }
    if is_z_family(mode) { S_OK } else { E_FAIL }
}

/// # Safety
/// `desc` and `layout` are null or point to an `XG_BUFFER_DESC` and a writable `XG_RESOURCE_LAYOUT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeBufferLayout(desc: *const XgBufferDesc, layout: *mut u8) -> Hresult {
    if desc.is_null() || layout.is_null() {
        return E_INVALIDARG;
    }
    let _guard = Guard::new();
    // SAFETY: `desc` is a readable `XG_BUFFER_DESC`.
    let d = unsafe { desc.read_unaligned() };
    match buffer_layout(d.byte_width, d.bind_flags, d.misc_flags) {
        // SAFETY: `layout` is a writable record.
        Ok(l) => unsafe { write_record(layout, &l) },
        Err(e) => failed("XGComputeBufferLayout", e),
    }
}

/// # Safety
/// `out` points to a writable `XG_RESOURCE_LAYOUT`.
unsafe fn write_record(out: *mut u8, layout: &decant_texture::xg::ResourceLayout) -> Hresult {
    let record = layout.to_record();
    // SAFETY: the record is 0x1B40 bytes and `out` holds that many.
    unsafe { std::ptr::copy_nonoverlapping(record.as_ptr(), out, record.len()) };
    S_OK
}

fn layout_of(function: &str, desc: &TextureDesc) -> Result<decant_texture::xg::TextureLayout, Hresult> {
    compute_layout(desc, state::hardware_v4()).map_err(|e| failed(function, e))
}

/// # Safety
/// `desc` and `layout` are null or point to an `XG_TEXTURE1D_DESC` and a writable `XG_RESOURCE_LAYOUT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeTexture1DLayout(desc: *const XgTexture1dDesc, layout: *mut u8) -> Hresult {
    if desc.is_null() || layout.is_null() {
        return E_INVALIDARG;
    }
    let _guard = Guard::new();
    // SAFETY: `desc` is readable.
    let d = TextureDesc::from(&unsafe { desc.read_unaligned() });
    match layout_of("XGComputeTexture1DLayout", &d) {
        // SAFETY: `layout` is a writable record.
        Ok(t) => unsafe { write_record(layout, &t.layout) },
        Err(code) => code,
    }
}

/// # Safety
/// `desc` and `layout` are null or point to an `XG_TEXTURE2D_DESC` and a writable `XG_RESOURCE_LAYOUT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeTexture2DLayout(desc: *const XgTexture2dDesc, layout: *mut u8) -> Hresult {
    if desc.is_null() || layout.is_null() {
        return E_INVALIDARG;
    }
    let _guard = Guard::new();
    // SAFETY: `desc` is readable.
    let d = TextureDesc::from(&unsafe { desc.read_unaligned() });
    match layout_of("XGComputeTexture2DLayout", &d) {
        // SAFETY: `layout` is a writable record.
        Ok(t) => unsafe { write_record(layout, &t.layout) },
        Err(code) => code,
    }
}

/// # Safety
/// `desc` and `layout` are null or point to an `XG_TEXTURE3D_DESC` and a writable `XG_RESOURCE_LAYOUT`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeTexture3DLayout(desc: *const XgTexture3dDesc, layout: *mut u8) -> Hresult {
    if desc.is_null() || layout.is_null() {
        return E_INVALIDARG;
    }
    let _guard = Guard::new();
    // SAFETY: `desc` is readable.
    let d = TextureDesc::from(&unsafe { desc.read_unaligned() });
    match layout_of("XGComputeTexture3DLayout", &d) {
        // SAFETY: `layout` is a writable record.
        Ok(t) => unsafe { write_record(layout, &t.layout) },
        Err(code) => code,
    }
}

/// Computes the layout of `d` and stores a new computer in `*out`.
///
/// # Safety
/// `out` is a writable pointer location.
unsafe fn create_computer(function: &str, d: &TextureDesc, out: *mut *mut TextureComputer) -> Hresult {
    let _guard = Guard::new();
    match layout_of(function, d) {
        Ok(t) => {
            // SAFETY: `out` is writable.
            unsafe { out.write(create(t)) };
            S_OK
        }
        Err(code) => code,
    }
}

/// # Safety
/// `desc` and `out` are null or point to an `XG_TEXTURE1D_DESC` and a writable pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateTexture1DComputer(desc: *const XgTexture1dDesc, out: *mut *mut TextureComputer) -> Hresult {
    if desc.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `desc` is readable and `out` writable.
    unsafe { create_computer("XGCreateTexture1DComputer", &TextureDesc::from(&desc.read_unaligned()), out) }
}

/// # Safety
/// `desc` and `out` are null or point to an `XG_TEXTURE2D_DESC` and a writable pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateTexture2DComputer(desc: *const XgTexture2dDesc, out: *mut *mut TextureComputer) -> Hresult {
    if desc.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `desc` is readable and `out` writable.
    unsafe { create_computer("XGCreateTexture2DComputer", &TextureDesc::from(&desc.read_unaligned()), out) }
}

/// # Safety
/// `desc` and `out` are null or point to an `XG_TEXTURE3D_DESC` and a writable pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateTexture3DComputer(desc: *const XgTexture3dDesc, out: *mut *mut TextureComputer) -> Hresult {
    if desc.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `desc` is readable and `out` writable.
    unsafe { create_computer("XGCreateTexture3DComputer", &TextureDesc::from(&desc.read_unaligned()), out) }
}

/// # Safety
/// `desc` and `out` are null or point to a `D3D12_RESOURCE_DESC` and a writable pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateTextureComputer(desc: *const [u8; 0x38], out: *mut *mut TextureComputer) -> Hresult {
    if desc.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: `desc` is a readable `D3D12_RESOURCE_DESC`.
    let resource = D3d12ResourceDesc::from_bytes(&unsafe { desc.read_unaligned() });
    let d = match texture_desc_from_d3d12(&resource) {
        Ok(d) => d,
        Err(e) => return failed("XGCreateTextureComputer", e),
    };
    if d.dim != Dim::D2 {
        // SAFETY: `out` is writable.
        unsafe { out.write(std::ptr::null_mut()) };
    }
    // SAFETY: `out` is writable.
    unsafe { create_computer("XGCreateTextureComputer", &d, out) }
}

/// # Safety
/// `descriptor` and `out` are null or point to 8 readable dwords and a writable pointer location.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateTextureComputerFromDescriptor(descriptor: *const u32, out: *mut *mut TextureComputer) -> Hresult {
    if descriptor.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    let mut words = [0u32; 8];
    // SAFETY: 8 readable dwords.
    unsafe { std::ptr::copy_nonoverlapping(descriptor, words.as_mut_ptr(), 8) };
    let image = match parse_image_descriptor(&words) {
        Ok(i) => i,
        Err(e) => return failed("XGCreateTextureComputerFromDescriptor", e),
    };
    let Some((bpe, block)) = image_format_element(image.image_format) else {
        decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("XGCreateTextureComputerFromDescriptor image format {:#x}", image.image_format));
        trace(&format!("xg_xs: XGCreateTextureComputerFromDescriptor: image format {:#x} is not in the format table, not implemented", image.image_format));
        return E_NOTIMPL;
    };
    match texture_desc_from_descriptor(&image, bpe, block) {
        // SAFETY: `out` is writable.
        Ok(d) => unsafe { create_computer("XGCreateTextureComputerFromDescriptor", &d, out) },
        Err(e) => failed("XGCreateTextureComputerFromDescriptor", e),
    }
}

/// # Safety
/// `descriptor` is null or points to `size_in_dwords` readable dwords.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGValidateDescriptor(size_in_dwords: u32, descriptor: *const u32, flags: u32) -> Hresult {
    let words = if descriptor.is_null() || size_in_dwords > 8 {
        None
    } else {
        // SAFETY: `size_in_dwords` readable dwords.
        Some(unsafe { std::slice::from_raw_parts(descriptor, size_in_dwords as usize) })
    };
    match validate_descriptor(size_in_dwords, words, flags) {
        Ok(()) => S_OK,
        Err(XgError::Fail) => E_FAIL,
        Err(e) => hresult(e),
    }
}

/// # Safety
/// `texels` and `elements` are null or writable `XG_TILE_SHAPE`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGComputeTileShape(dimension: u32, format: u32, bytes_per_element: u32, sample_count: u32, texels: *mut XgTileShape, elements: *mut XgTileShape) -> Hresult {
    match compute_tile_shape(dimension, format, bytes_per_element, sample_count) {
        Ok(s) => {
            // SAFETY: non null pointers are writable.
            unsafe {
                if !texels.is_null() {
                    texels.write(XgTileShape { width: s.texels.width, height: s.texels.height, depth: s.texels.depth });
                }
                if !elements.is_null() {
                    elements.write(XgTileShape { width: s.elements.width, height: s.elements.height, depth: s.elements.depth });
                }
            }
            S_OK
        }
        Err(e) => failed("XGComputeTileShape", e),
    }
}

/// The software DCC codec is not specified: a logging stand in.
///
/// # Safety
/// The pointers are not used.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGEncodeDCC(count: u32, _src: *const c_void, _dst: *mut c_void, _metadata: *mut c_void, _sizes: *mut u32, format: u32, swizzle_mode: u32, dimension: u32, _params: *const u32) -> Hresult {
    decant_log::calls::not_implemented("XGEncodeDCC");
    trace(&format!("xg_xs: XGEncodeDCC not implemented (count {count}, format {format}, swizzle mode {swizzle_mode}, dimension {dimension})"));
    E_NOTIMPL
}

/// The software DCC codec is not specified: a logging stand in.
///
/// # Safety
/// The pointers are not used.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGDecodeDCC(count: u32, _src: *const c_void, _metadata: *const c_void, _dst: *mut c_void, format: u32, swizzle_mode: u32, dimension: u32, _params: *const u32) -> Hresult {
    decant_log::calls::not_implemented("XGDecodeDCC");
    trace(&format!("xg_xs: XGDecodeDCC not implemented (count {count}, format {format}, swizzle mode {swizzle_mode}, dimension {dimension})"));
    E_NOTIMPL
}

/// The ray tracing acceleration structure helper is not specified: a logging stand in.
///
/// # Safety
/// `out` is not used.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XGCreateBVHComputer(_out: *mut *mut c_void) -> Hresult {
    decant_log::calls::not_implemented("XGCreateBVHComputer");
    trace("xg_xs: XGCreateBVHComputer not implemented");
    E_NOTIMPL
}
