//! `decant_bvh_test [decant_bvh.dll]`: builds the acceleration structure of one triangle through
//! `decant_bvh.dll` on the GPU and reads back its compacted size after a fence. The port of the
//! former `tests/decant_bvh_test.cpp`, with the same lines and exit codes (0 passed, 1 failed, 2 no
//! DXR device). The C program linked `decant_bvh.dll` at load time. This one loads it from its own
//! folder, or from the path given. `gpu_diag` runs it from the build folder with no argument.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(gpu::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("decant_bvh_test runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod gpu {
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;

    use decant_testkit::Library;
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_1;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32B32_FLOAT, DXGI_SAMPLE_DESC};
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::{HRESULT, Interface};

    type PrebuildFn = unsafe extern "C" fn(
        *mut c_void,
        *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
        *mut D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO,
    );
    type BuildFn = unsafe extern "C" fn(
        *mut c_void,
        *const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
        u32,
        *const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC,
    );

    fn buffer(device: &ID3D12Device5, size: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
        let properties = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
        let description = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: both descriptions live for the call. The out parameter is a valid Option.
        unsafe { device.CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &description, state, None, &mut resource) }.ok()?;
        resource
    }

    pub fn main() -> u8 {
        let args = decant_testkit::args();
        let path = args.first().cloned().unwrap_or_else(|| decant_testkit::program_dir().join("decant_bvh.dll").into_os_string());
        let lib = match Library::open(&path) {
            Ok(lib) => lib,
            Err(error) => {
                println!("chargement de decant_bvh.dll impossible ({error})");
                return 2;
            }
        };
        // SAFETY: the signatures of decant_bvh.h for these names.
        let (Some(prebuild), Some(build)) = (unsafe { lib.get::<PrebuildFn>("DecantPrebuild") }, unsafe { lib.get::<BuildFn>("DecantBuild") }) else {
            println!("chargement de decant_bvh.dll impossible (exports absents)");
            return 2;
        };

        let mut device: Option<ID3D12Device5> = None;
        // SAFETY: plain device creation on the default adapter.
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_1, &mut device) }.is_err() {
            println!("peripherique DXR indisponible");
            return 2;
        }
        let Some(device) = device else {
            println!("peripherique DXR indisponible");
            return 2;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: `options` is the structure this feature fills, with its size.
        let supported = unsafe {
            device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&raw mut options).cast(), size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32)
        };
        if supported.is_err() || options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED {
            println!("DXR unavailable");
            return 2;
        }
        run(&device, prebuild, build).unwrap_or(1)
    }

    fn run(device: &ID3D12Device5, prebuild: PrebuildFn, build: BuildFn) -> Option<u8> {
        // SAFETY: plain D3D12 object creation and recording on objects this function owns. Every
        // structure passed lives until the call returns, and the GPU work is waited for below.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let commands: ID3D12GraphicsCommandList4 = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;

            let vertices = buffer(device, 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let triangle: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            let mut mapped: *mut c_void = std::ptr::null_mut();
            vertices.Map(0, None, Some(&mut mapped)).ok()?;
            std::ptr::copy_nonoverlapping(triangle.as_ptr().cast::<u8>(), mapped.cast::<u8>(), size_of_val(&triangle));
            vertices.Unmap(0, None);

            let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        VertexCount: 3,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: vertices.GetGPUVirtualAddress(), StrideInBytes: 12 },
                        ..Default::default()
                    },
                },
            };
            let mut desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                        | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
                    NumDescs: 1,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry },
                },
                ..Default::default()
            };

            let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            prebuild(device.as_raw(), &desc.Inputs, &mut info);
            println!("PrebuildInfo : resultat {} octets, travail {} octets", info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes);
            if info.ResultDataMaxSizeInBytes == 0 {
                println!("ESSAI ECHOUE");
                return Some(1);
            }
            let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let result = buffer(device, info.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)?;
            let scratch = buffer(device, info.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let size_gpu = buffer(device, 64, D3D12_HEAP_TYPE_DEFAULT, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
            let size_cpu = buffer(device, 64, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            desc.DestAccelerationStructureData = result.GetGPUVirtualAddress();
            desc.ScratchAccelerationStructureData = scratch.GetGPUVirtualAddress();
            let post = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: size_gpu.GetGPUVirtualAddress(),
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            };

            build(commands.as_raw(), &desc, 1, &post);

            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        // Borrowed without a reference of its own, as the C++ code passed the pointer.
                        pResource: std::mem::transmute_copy(&size_gpu),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            commands.ResourceBarrier(&[barrier]);
            commands.CopyResource(&size_cpu, &size_gpu);
            commands.Close().ok()?;
            queue.ExecuteCommandLists(&[Some(commands.cast::<ID3D12CommandList>().ok()?)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let waited = queue.Signal(&fence, 1).is_ok() && fence.SetEventOnCompletion(1, event).is_ok() && WaitForSingleObject(event, 30000) == WAIT_OBJECT_0;
            let _ = CloseHandle(event);
            if !waited {
                return None;
            }

            let removed: HRESULT = device.GetDeviceRemovedReason().map_or_else(|e| e.code(), |()| HRESULT(0));
            let mut compacted = 0u64;
            let mut mapped: *mut c_void = std::ptr::null_mut();
            if size_cpu.Map(0, None, Some(&mut mapped)).is_ok() {
                compacted = mapped.cast::<u64>().read();
                size_cpu.Unmap(0, None);
            }
            println!(
                "build execute sur la carte : etat du peripherique 0x{:08x}, taille compactee rendue par la carte : {compacted} octets",
                removed.0 as u32
            );
            Some(u8::from(removed.is_err() || compacted == 0 || compacted > info.ResultDataMaxSizeInBytes))
        }
    }
}
