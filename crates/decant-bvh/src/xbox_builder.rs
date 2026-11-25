//! `XboxBvhBuilder.dll`: a stand in for the package's console BVH builder, so the title never
//! creates the 53 console compute pipelines that fail on a PC D3D12 device.
//!
//! The package class is a C++ class (x64 MSVC, `this` in `rcx`) with twelve exported members, all
//! exported here under their exact mangled names. The title imports seven of them (ctor, dtor,
//! `GetRaytracingAccelerationStructurePrebuildInfo`, `CopyRaytracingAccelerationStructure`,
//! `BuildRaytracingAccelerationStructureBatched2X`,
//! `RehydrateRaytracingAccelerationStructureBatched2X`, `Set64BitMortonCodes`).
//!
//! Object: the title allocates 0x18 bytes (`mov ecx, 0x18; call operator new`, then
//! `ctor(this, device, 0)`), so the constructor writes only inside 0x18 bytes:
//!
//! | offset | content |
//! |---|---|
//! | 0x00 | the PC `ID3D12Device5` (one reference, released by the destructor) |
//! | 0x08 | the `bool` of the constructor |
//! | 0x09 | the 64 bit Morton code flag of `Set64BitMortonCodes` |
//!
//! The device and the command lists the title passes are `d3d12_xs` proxies (console vtable).
//! Their `QueryInterface` answers a PC interface with the proxy itself, so it cannot give the PC
//! vtable. [`pc_object`] asks the `D3D12XsUnwrap` export of the loaded `d3d12_xs.dll` (resolved
//! once) for the PC object. Without that DLL or export (an older build) every object is taken as
//! a PC object already.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Foundation::S_OK;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC, D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO, ID3D12Device5,
    ID3D12GraphicsCommandList4,
};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::core::{Interface, s, w};

/// Bytes the title allocates for the object. The constructor never writes past this.
pub const OBJECT_BYTES: usize = 0x18;
/// Offset of the PC device in the object.
pub const OFF_DEVICE: usize = 0x00;
/// Offset of the constructor's `bool`.
pub const OFF_FLAG: usize = 0x08;
/// Offset of the Morton flag.
pub const OFF_MORTON: usize = 0x09;

fn log(message: &str) {
    decant_log::session_log(&format!("bvh: XboxBvhBuilder {message}"));
}

/// Logs the first call of a member, with its arguments, then stays silent.
macro_rules! log_once {
    ($($arg:tt)*) => {{
        static SEEN: AtomicBool = AtomicBool::new(false);
        if !SEEN.swap(true, Ordering::Relaxed) {
            log(&format!($($arg)*));
        }
    }};
}

/// `D3D12XsUnwrap` of `d3d12_xs.dll`: the PC COM pointer behind a console proxy, or the argument
/// unchanged for any other object (including null). The result is borrowed, not add referenced.
type UnwrapFn = unsafe extern "system" fn(*mut c_void) -> *mut c_void;

/// The `D3D12XsUnwrap` of the loaded `d3d12_xs.dll`, resolved once. `None` when the DLL is not
/// loaded or is an older build without the export, then every object is taken as a PC object.
fn unwrap_fn() -> Option<UnwrapFn> {
    static RESOLVED: OnceLock<Option<UnwrapFn>> = OnceLock::new();
    *RESOLVED.get_or_init(|| {
        // SAFETY: both calls only read the loader's module table.
        let found = unsafe { GetModuleHandleW(w!("d3d12_xs.dll")).ok().and_then(|module| GetProcAddress(module, s!("D3D12XsUnwrap"))) };
        if found.is_none() {
            log("no D3D12XsUnwrap in a loaded d3d12_xs.dll (older build or not loaded): objects are taken as PC objects");
        }
        // SAFETY: the export has the signature of `UnwrapFn` (d3d12-xs `exports.rs`).
        found.map(|address| unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, UnwrapFn>(address) })
    })
}

/// The PC object behind a console proxy through `unwrap`, or `object` itself without one.
///
/// # Safety
/// `object` is null or a COM object pointer, as `unwrap` requires.
unsafe fn unwrap_with(unwrap: Option<UnwrapFn>, object: *mut c_void) -> *mut c_void {
    match unwrap {
        // SAFETY: by the contract above.
        Some(f) => unsafe { f(object) },
        None => object,
    }
}

/// The PC object behind `object`, through the resolved `D3D12XsUnwrap`.
///
/// # Safety
/// As [`unwrap_with`].
unsafe fn pc_object(object: *mut c_void) -> *mut c_void {
    // SAFETY: by the contract above.
    unsafe { unwrap_with(unwrap_fn(), object) }
}

/// A PC command list of the interface `T`, from the title's list (proxy or PC).
///
/// # Safety
/// `list` is null or a live command list.
unsafe fn list_as<T: Interface>(list: *mut c_void) -> Option<T> {
    // SAFETY: by the contract above. The borrowed pointer is not released.
    unsafe {
        let pc = pc_object(list);
        windows::core::IUnknown::from_raw_borrowed(&pc)?.cast::<T>().ok()
    }
}

/// The PC device stored in the object.
///
/// # Safety
/// `this` is null or an object made by the constructor.
unsafe fn device_of<'a>(this: *mut c_void) -> Option<&'a ID3D12Device5> {
    if this.is_null() {
        return None;
    }
    // SAFETY: by the contract above, offset 0 holds an `ID3D12Device5` pointer or null.
    unsafe {
        let slot = this.cast::<u8>().add(OFF_DEVICE).cast::<*mut c_void>();
        ID3D12Device5::from_raw_borrowed(&*slot)
    }
}

/// `XboxBvhBuilder::XboxBvhBuilder(ID3D12Device11*, bool)`
///
/// # Safety
/// `this` points to at least [`OBJECT_BYTES`] writable bytes, `device` is a live device or null.
#[unsafe(export_name = "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z")]
pub unsafe extern "system" fn ctor(this: *mut c_void, device: *mut c_void, flag: u8) -> *mut c_void {
    if this.is_null() {
        return this;
    }
    // SAFETY: `this` has OBJECT_BYTES bytes, `device` is a live device.
    unsafe {
        let base = this.cast::<u8>();
        std::ptr::write_bytes(base, 0, OBJECT_BYTES);
        let pc = pc_object(device);
        let device5 = windows::core::IUnknown::from_raw_borrowed(&pc).and_then(|d| d.cast::<ID3D12Device5>().ok());
        log(&format!(
            "ctor(this={this:p}, device={device:p}, bool={flag}): proxy={} pc device={pc:p}, ID3D12Device5 {}",
            pc != device,
            if device5.is_some() { "obtained" } else { "NOT available" }
        ));
        base.add(OFF_FLAG).write(flag);
        if let Some(d) = device5 {
            base.add(OFF_DEVICE).cast::<*mut c_void>().write(d.into_raw());
        }
    }
    this
}

/// `XboxBvhBuilder::~XboxBvhBuilder()`
///
/// # Safety
/// `this` is null or an object made by [`ctor`].
#[unsafe(export_name = "??1XboxBvhBuilder@@QEAA@XZ")]
pub unsafe extern "system" fn dtor(this: *mut c_void) {
    if this.is_null() {
        return;
    }
    // SAFETY: offset 0 owns one reference taken by the constructor.
    unsafe {
        let slot = this.cast::<u8>().add(OFF_DEVICE).cast::<*mut c_void>();
        let device = slot.read();
        slot.write(std::ptr::null_mut());
        if !device.is_null() {
            drop(ID3D12Device5::from_raw(device));
        }
    }
    log_once!("dtor(this={this:p})");
}

/// `GetRaytracingAccelerationStructurePrebuildInfo(inputs, stream desc, info)`. The XDXR stream
/// description is ignored (its pointer is logged), the PC prebuild answers.
///
/// # Safety
/// `inputs` and `info` are what the PC method takes.
#[unsafe(export_name = "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z")]
pub unsafe extern "system" fn get_prebuild_info(
    this: *mut c_void,
    inputs: *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    stream: *const c_void,
    info: *mut D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
) {
    log_once!("GetRaytracingAccelerationStructurePrebuildInfo(this={this:p}, inputs={inputs:p}, xdxr stream={stream:p}, info={info:p}): PC prebuild, stream ignored");
    // SAFETY: `this` is made by the constructor, the pointers are what the PC method takes.
    unsafe {
        let Some(device) = device_of(this) else {
            log_once!("prebuild: no PC device, the output is left untouched");
            return;
        };
        (Interface::vtable(device).GetRaytracingAccelerationStructurePrebuildInfo)(Interface::as_raw(device), inputs, info);
    }
}

/// `BuildRaytracingAccelerationStructure(list4, desc, count, post)`: passes through.
///
/// # Safety
/// The pointers are what the PC method takes.
#[unsafe(export_name = "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z")]
pub unsafe extern "system" fn build(
    this: *mut c_void,
    list: *mut c_void,
    desc: *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    count: u32,
    post: *const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC,
) {
    log_once!("BuildRaytracingAccelerationStructure(this={this:p}, list={list:p}, desc={desc:p}, count={count}, post={post:p})");
    // SAFETY: `list` is a live command list.
    unsafe {
        let Some(list4) = list_as::<ID3D12GraphicsCommandList4>(list) else {
            log_once!("build: no ID3D12GraphicsCommandList4, nothing recorded");
            return;
        };
        (Interface::vtable(&list4).BuildRaytracingAccelerationStructure)(Interface::as_raw(&list4), desc, count, post);
    }
}

/// `CopyRaytracingAccelerationStructure(list4, dest address, source address, mode)`: passes through.
///
/// # Safety
/// `list` is a live command list.
#[unsafe(export_name = "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z")]
pub unsafe extern "system" fn copy(this: *mut c_void, list: *mut c_void, dest: u64, source: u64, mode: i32) {
    log_once!("CopyRaytracingAccelerationStructure(this={this:p}, list={list:p}, dest=0x{dest:X}, source=0x{source:X}, mode={mode})");
    // SAFETY: `list` is a live command list.
    unsafe {
        let Some(list4) = list_as::<ID3D12GraphicsCommandList4>(list) else {
            log_once!("copy: no ID3D12GraphicsCommandList4, nothing recorded");
            return;
        };
        (Interface::vtable(&list4).CopyRaytracingAccelerationStructure)(Interface::as_raw(&list4), dest, source, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE(mode));
    }
}

/// `BuildRaytracingAccelerationStructureBatched2X(list6, count, desc)`: the console batched
/// description layout is not confirmed, so this logs and records nothing.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z")]
pub unsafe extern "system" fn build_batched2x(this: *mut c_void, list: *mut c_void, count: u32, desc: *const c_void) {
    log_once!("BuildRaytracingAccelerationStructureBatched2X(this={this:p}, list={list:p}, count={count}, desc={desc:p}): console desc layout unknown, no op");
}

/// `CpuValidateResultX(list4, desc)`: no op.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?CpuValidateResultX@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z")]
pub unsafe extern "system" fn cpu_validate_result(this: *mut c_void, list: *mut c_void, desc: *const c_void) {
    log_once!("CpuValidateResultX(this={this:p}, list={list:p}, desc={desc:p}): no op");
}

/// `RehydrateRaytracingAccelerationStructureBatched2X(list8, count, desc, sizes)`. The title
/// ignores the result (`eax` is not read after either call), so `S_OK` with no work.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z")]
pub unsafe extern "system" fn rehydrate_batched2x(this: *mut c_void, list: *mut c_void, count: u32, desc: *const c_void, extra: *const u64) -> i32 {
    log_once!("RehydrateRaytracingAccelerationStructureBatched2X(this={this:p}, list={list:p}, count={count}, desc={desc:p}, array={extra:p}): S_OK, no work (the title ignores the result)");
    S_OK.0
}

/// `RehydrateRaytracingAccelerationStructureBatchedX(list8, count, desc)`: as the 2X form.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?RehydrateRaytracingAccelerationStructureBatchedX@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z")]
pub unsafe extern "system" fn rehydrate_batched(this: *mut c_void, list: *mut c_void, count: u32, desc: *const c_void) -> i32 {
    log_once!("RehydrateRaytracingAccelerationStructureBatchedX(this={this:p}, list={list:p}, count={count}, desc={desc:p}): S_OK, no work");
    S_OK.0
}

/// `RehydrateRaytracingAccelerationStructureX(device11, desc)`: as the batched forms.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?RehydrateRaytracingAccelerationStructureX@XboxBvhBuilder@@QEAAJPEAUID3D12Device11@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z")]
pub unsafe extern "system" fn rehydrate(this: *mut c_void, device: *mut c_void, desc: *const c_void) -> i32 {
    log_once!("RehydrateRaytracingAccelerationStructureX(this={this:p}, device={device:p}, desc={desc:p}): S_OK, no work");
    S_OK.0
}

/// `Set64BitMortonCodes(bool)`: the flag is stored and logged.
///
/// # Safety
/// `this` is null or an object made by [`ctor`].
#[unsafe(export_name = "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z")]
pub unsafe extern "system" fn set_64bit_morton_codes(this: *mut c_void, enable: u8) {
    log_once!("Set64BitMortonCodes(this={this:p}, {enable})");
    if !this.is_null() {
        // SAFETY: the object has OBJECT_BYTES bytes.
        unsafe { this.cast::<u8>().add(OFF_MORTON).write(enable) };
    }
}

/// `ValidateRehydrateAccelerationStructureBuildDesc(desc)`: logged, `S_OK`.
///
/// # Safety
/// Arguments are only logged.
#[unsafe(export_name = "?ValidateRehydrateAccelerationStructureBuildDesc@XboxBvhBuilder@@QEAAJPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z")]
pub unsafe extern "system" fn validate_rehydrate(this: *mut c_void, desc: *const c_void) -> i32 {
    log_once!("ValidateRehydrateAccelerationStructureBuildDesc(this={this:p}, desc={desc:p}): S_OK");
    S_OK.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_object_fields_fit_the_titles_allocation() {
        const { assert!(OBJECT_BYTES == 0x18) };
        const { assert!(OFF_DEVICE + size_of::<*mut c_void>() <= OBJECT_BYTES) };
        const { assert!(OFF_FLAG < OFF_MORTON && OFF_MORTON < OBJECT_BYTES) };
    }

    unsafe extern "system" fn fake_unwrap(object: *mut c_void) -> *mut c_void {
        if object.is_null() { object } else { 0x2000 as *mut c_void }
    }

    #[test]
    fn without_an_unwrap_function_the_object_is_a_pc_object() {
        let object = 0x1000 as *mut c_void;
        // SAFETY: nothing is dereferenced without a function.
        unsafe {
            assert_eq!(unwrap_with(None, object), object);
            assert!(unwrap_with(None, std::ptr::null_mut()).is_null());
        }
    }

    #[test]
    fn with_an_unwrap_function_its_answer_is_used() {
        // SAFETY: the fake never dereferences its argument.
        unsafe {
            assert_eq!(unwrap_with(Some(fake_unwrap), 0x1000 as *mut c_void), 0x2000 as *mut c_void);
            assert!(unwrap_with(Some(fake_unwrap), std::ptr::null_mut()).is_null());
        }
    }
}
