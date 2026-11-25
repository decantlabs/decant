//! The binary layout of the descriptions the exports read (`docs/CONSOLE_LIBS.md` 1.3) and the result codes.

use decant_texture::xg::{Dim, TextureDesc, XgError};

pub type Hresult = i32;

pub const S_OK: Hresult = 0;
pub const E_NOTIMPL: Hresult = 0x8000_4001_u32 as i32;
pub const E_INVALIDARG: Hresult = 0x8007_0057_u32 as i32;
pub const E_FAIL: Hresult = 0x8000_4005_u32 as i32;

/// The result code of a failed computation.
pub fn hresult(e: XgError) -> Hresult {
    match e {
        XgError::InvalidArg => E_INVALIDARG,
        XgError::NotImpl => E_NOTIMPL,
        XgError::Fail => E_FAIL,
    }
}

/// `XG_TEXTURE1D_DESC`, 0x28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XgTexture1dDesc {
    pub width: u32,
    pub mip_levels: u32,
    pub array_size: u32,
    pub format: u32,
    pub usage: u32,
    pub bind_flags: u32,
    pub cpu_access_flags: u32,
    pub misc_flags: u32,
    pub swizzle_mode: u32,
    pub pitch: u32,
}

/// `XG_TEXTURE2D_DESC`, 0x34 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XgTexture2dDesc {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub array_size: u32,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub usage: u32,
    pub bind_flags: u32,
    pub cpu_access_flags: u32,
    pub misc_flags: u32,
    pub swizzle_mode: u32,
    pub reserved: u32,
}

/// `XG_TEXTURE3D_DESC`, 0x2C bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XgTexture3dDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub format: u32,
    pub usage: u32,
    pub bind_flags: u32,
    pub cpu_access_flags: u32,
    pub misc_flags: u32,
    pub swizzle_mode: u32,
    pub pitch: u32,
}

/// `XG_BUFFER_DESC`, 0x18 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XgBufferDesc {
    pub byte_width: u32,
    pub usage: u32,
    pub bind_flags: u32,
    pub cpu_access_flags: u32,
    pub misc_flags: u32,
    pub structure_byte_stride: u32,
}

/// `XG_TILE_SHAPE`, 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XgTileShape {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

const _: () = assert!(size_of::<XgTexture1dDesc>() == 0x28);
const _: () = assert!(size_of::<XgTexture2dDesc>() == 0x34);
const _: () = assert!(size_of::<XgTexture3dDesc>() == 0x2C);
const _: () = assert!(size_of::<XgBufferDesc>() == 0x18);
const _: () = assert!(size_of::<XgTileShape>() == 12);

impl From<&XgTexture1dDesc> for TextureDesc {
    fn from(d: &XgTexture1dDesc) -> TextureDesc {
        TextureDesc {
            dim: Dim::D1,
            width: d.width,
            height: 1,
            depth_or_array: d.array_size,
            mip_levels: d.mip_levels,
            format: d.format,
            sample_count: 1,
            usage: d.usage,
            bind_flags: d.bind_flags,
            misc_flags: d.misc_flags,
            swizzle_mode: d.swizzle_mode,
            pitch: d.pitch,
        }
    }
}

impl From<&XgTexture2dDesc> for TextureDesc {
    fn from(d: &XgTexture2dDesc) -> TextureDesc {
        TextureDesc {
            dim: Dim::D2,
            width: d.width,
            height: d.height,
            depth_or_array: d.array_size,
            mip_levels: d.mip_levels,
            format: d.format,
            sample_count: d.sample_count,
            usage: d.usage,
            bind_flags: d.bind_flags,
            misc_flags: d.misc_flags,
            swizzle_mode: d.swizzle_mode,
            pitch: 0,
        }
    }
}

impl From<&XgTexture3dDesc> for TextureDesc {
    fn from(d: &XgTexture3dDesc) -> TextureDesc {
        TextureDesc {
            dim: Dim::D3,
            width: d.width,
            height: d.height,
            depth_or_array: d.depth,
            mip_levels: d.mip_levels,
            format: d.format,
            sample_count: 1,
            usage: d.usage,
            bind_flags: d.bind_flags,
            misc_flags: d.misc_flags,
            swizzle_mode: d.swizzle_mode,
            pitch: d.pitch,
        }
    }
}
