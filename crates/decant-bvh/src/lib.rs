//! `decant_bvh.dll`: a small native DXR build interface. A port of the former
//! `src/decant/decant_bvh.cpp` and `decant_bvh.h`, with the same two C exports:
//!
//! ```c
//! void DecantPrebuild(ID3D12Device5*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS*,
//!                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO*);
//! void DecantBuild(ID3D12GraphicsCommandList4*, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*,
//!               UINT, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*);
//! ```
//!
//! Both take native D3D12 objects and GPU addresses and call the device or the command list
//! directly. Callers own the resources, the synchronisation and the descriptor lifetimes. A null
//! device or command list is ignored (the C code would have crashed on it).

#[cfg(windows)]
mod exports;
#[cfg(windows)]
pub mod xbox_builder;
