//! Loads the built DLL under the package's name, `XboxBvhBuilder.dll`, resolves the twelve mangled
//! exports, constructs the object on a real D3D12 device inside a buffer of the size the title
//! allocates (0x18 bytes, canaries around it), asks the prebuild size of one triangle and
//! destructs. `d3d12_xs.dll` is not loaded here, so the plain PC device goes through the fallback
//! of `D3D12XsUnwrap` resolution (the object is taken as a PC object). A machine with no DXR device skips (set `DECANT_BVH_REQUIRE_DXR` to make that a failure).
#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::c_void;

use decant_testkit::Library;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT;
use windows::core::{IUnknown, Interface};

const EXPORTS: [&str; 12] = [
    "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z",
    "??1XboxBvhBuilder@@QEAA@XZ",
    "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
    "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
    "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
    "?CpuValidateResultX@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z",
    "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
    "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
    "?RehydrateRaytracingAccelerationStructureBatchedX@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z",
    "?RehydrateRaytracingAccelerationStructureX@XboxBvhBuilder@@QEAAJPEAUID3D12Device11@@PEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z",
    "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z",
    "?ValidateRehydrateAccelerationStructureBuildDesc@XboxBvhBuilder@@QEAAJPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@@Z",
];

type Ctor = unsafe extern "system" fn(*mut c_void, *mut c_void, u8) -> *mut c_void;
type Dtor = unsafe extern "system" fn(*mut c_void);
type Prebuild = unsafe extern "system" fn(
    *mut c_void,
    *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
    *const c_void,
    *mut D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
);
type Morton = unsafe extern "system" fn(*mut c_void, u8);

fn skip_or_fail(why: &str) {
    assert!(std::env::var_os("DECANT_BVH_REQUIRE_DXR").is_none(), "no DXR device: {why}");
    println!("SKIPPED xbox_builder: {why}");
}

#[test]
fn xbox_bvh_builder_dll_constructs_and_prebuilds() {
    let program = env!("CARGO_BIN_EXE_decant_bvh_test");
    let built = decant_testkit::dll_for(program, "decant_bvh.dll");
    let folder = std::env::temp_dir().join(format!("xbox_bvh_builder_test_{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    let dll = folder.join("XboxBvhBuilder.dll");
    std::fs::copy(&built, &dll).unwrap_or_else(|e| panic!("copy {}: {e}", built.display()));
    let lib = Library::open(dll.as_os_str()).unwrap_or_else(|e| panic!("load {}: error {e}", dll.display()));
    for name in EXPORTS {
        assert!(!lib.address(name).is_null(), "missing export {name}");
    }

    let mut device: Option<ID3D12Device5> = None;
    // SAFETY: plain device creation on the default adapter.
    if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() || device.is_none() {
        return skip_or_fail("no D3D12 device");
    }
    let device = device.unwrap();
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: the structure this feature fills, with its size.
    let ok = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&raw mut options).cast(), size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32) };
    if ok.is_err() || options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED {
        return skip_or_fail("DXR tier missing");
    }

    // SAFETY: the signatures of the class members for these names.
    let (ctor, dtor, prebuild, morton) = unsafe {
        (
            lib.get::<Ctor>(EXPORTS[0]).unwrap(),
            lib.get::<Dtor>(EXPORTS[1]).unwrap(),
            lib.get::<Prebuild>(EXPORTS[6]).unwrap(),
            lib.get::<Morton>(EXPORTS[10]).unwrap(),
        )
    };

    // The title's object is 0x18 bytes. Canaries on both sides show a write outside it.
    let mut memory = [0xA5u8; 0x18 + 32];
    let object = memory[16..].as_mut_ptr().cast::<c_void>();
    let unknown: IUnknown = device.cast().unwrap();
    // SAFETY: `object` has 0x18 writable bytes, the device is live. References are counted with
    // AddRef and Release on the device itself.
    unsafe {
        let count_before = {
            let raw = Interface::as_raw(&unknown);
            let vtable = *raw.cast::<*const [usize; 3]>();
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vtable)[2]);
            let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vtable)[1]);
            add(raw);
            release(raw)
        };
        assert_eq!(ctor(object, device.as_raw(), 0), object);
        assert!(memory[..16].iter().all(|&b| b == 0xA5), "a write before the object");
        assert!(memory[16 + 0x18..].iter().all(|&b| b == 0xA5), "a write after the 0x18 bytes");
        morton(object, 1);
        assert_eq!(memory[16 + 0x09], 1);

        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    VertexCount: 3,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: 0x1_0000, StrideInBytes: 12 },
                    ..Default::default()
                },
            },
        };
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry },
        };
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        prebuild(object, &inputs, std::ptr::null(), &mut info);
        println!("prebuild: result {} bytes, scratch {} bytes", info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes);
        assert!(info.ResultDataMaxSizeInBytes > 0 && info.ScratchDataSizeInBytes > 0, "sizes must be nonzero");

        dtor(object);
        // The destructor released the reference of the constructor: the count is as before.
        let raw = Interface::as_raw(&unknown);
        let vtable = *raw.cast::<*const [usize; 3]>();
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vtable)[2]);
        let add: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute((*vtable)[1]);
        add(raw);
        assert_eq!(release(raw), count_before, "the destructor must release the device reference it took");
    }
    lib.free();
    let _ = std::fs::remove_dir_all(&folder);
}
