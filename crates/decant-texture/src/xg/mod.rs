//! The computation behind the console library `xg_xs` (`docs/CONSOLE_LIBS.md`, section 1): which GFX10 swizzle mode to use,
//! how large a texture is, where each mip level and slice lives, and the byte offset of any element. The DLL
//! (`crates/decant-xg`) only wraps these functions in the console ABI.
//!
//! The modes without exclusive or and, with hardware version 4, the RB+ ones come from the addrlib tables of
//! [`crate::tiling`]. The layout of a 2D level chain follows [`crate::surface::compute`] where the specification says they
//! agree (the tests compare them).

pub mod d3d12;
pub mod descriptor;
pub mod equations;
mod equations_data;
pub mod formats;
pub mod layout;
pub mod modes;
pub mod resource_tiling;
pub mod swizzle;
pub mod tile_shape;

/// Why a computation failed. The DLL maps them to `E_INVALIDARG`, `E_NOTIMPL` and `E_FAIL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XgError {
    InvalidArg,
    NotImpl,
    Fail,
}

pub use layout::{Dim, Level, Plane, ResourceLayout, TextureDesc, TextureLayout, buffer_layout, compute_footprint, compute_layout};
pub use swizzle::{SwizzleQuery, compute_optimal_swizzle_mode};
