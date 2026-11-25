//! The two exports. They forward the caller's pointers to the D3D12 vtables unchanged.
#![allow(unsafe_code)]

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO, ID3D12Device5,
    ID3D12GraphicsCommandList4,
};
use windows::core::Interface;

/// # Safety
/// Exported. `device` is an `ID3D12Device5` (or null), `inputs` and `info` are what
/// `GetRaytracingAccelerationStructurePrebuildInfo` takes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DecantPrebuild(
    device: *mut c_void,
    inputs: *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    info: *mut D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
) {
    // SAFETY: the caller passes a live ID3D12Device5 or null. It is borrowed, not released.
    let Some(device) = (unsafe { ID3D12Device5::from_raw_borrowed(&device) }) else {
        return;
    };
    // SAFETY: the vtable call takes the caller's two pointers as the C++ method does.
    unsafe { (Interface::vtable(device).GetRaytracingAccelerationStructurePrebuildInfo)(Interface::as_raw(device), inputs, info) };
}

/// # Safety
/// Exported. `list` is an `ID3D12GraphicsCommandList4` (or null), the other arguments are what
/// `BuildRaytracingAccelerationStructure` takes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DecantBuild(
    list: *mut c_void,
    desc: *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    count: u32,
    post: *const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC,
) {
    // SAFETY: the caller passes a live ID3D12GraphicsCommandList4 or null. It is borrowed, not released.
    let Some(list) = (unsafe { ID3D12GraphicsCommandList4::from_raw_borrowed(&list) }) else {
        return;
    };
    // SAFETY: the vtable call takes the caller's pointers and count as the C++ method does.
    unsafe { (Interface::vtable(list).BuildRaytracingAccelerationStructure)(Interface::as_raw(list), desc, count, post) };
}
