//! `xg_xs.dll`: the texture and resource layout library of the console, in Rust (`docs/CONSOLE_LIBS.md`, section 1).
//!
//! Given a texture description it says which GFX10 swizzle mode to use, how large the resource is, where each mip level and
//! array slice lives and the byte offset of any element. The computation is `decant_texture::xg`, built on the GFX10 tables
//! and the surface layout of `decant-texture`. This crate holds what is specific to the DLL: the 23 exports with the console
//! prototypes, the binary layout of the descriptions, the process wide state (initialised flag, hardware version, exclusive
//! locking) and the texture computer object with its table of 12 methods.
//!
//! Written from the specification only. Open points return `E_NOTIMPL` and log: the DCC and BVH functions, the GFX10 image
//! format table behind `XGCreateTextureComputerFromDescriptor`, and three slots of the texture computer.

pub mod abi;
pub mod computer;
pub mod exports;
pub mod state;

pub use exports::EXPORT_NAMES;
