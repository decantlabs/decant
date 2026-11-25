//! Synthetic draw bench for `d3d12_xs.dll` (W6 draw-bench).
//!
//! Exercises the console ABI proxy path without game data:
//! - `cargo bench -p decant-d3d12x --bench draws` (runs on WARP by default, safe during game runs)
//! - `cargo bench -p decant-d3d12x --bench draws -- --gpu` (runs on the high performance hardware GPU
//!   when `D:\decant_runs\.lock` is absent)

#![cfg(windows)]
#![allow(unsafe_code)]

use std::ffi::{OsStr, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use decant_testkit::Library;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
    IDXGIAdapter1, IDXGIFactory6,
};
use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};
use windows::core::{GUID, Interface, PCSTR};

const DRAW_COUNTS: [usize; 4] = [500, 1000, 3000, 5000];
const WARMUP_FRAMES: usize = 3;
const MEASURED_FRAMES: usize = 100;
const TABLE_COUNT: usize = 256;
const TABLE_STRIDE_SLOTS: usize = 16;
const HEAP_DESCRIPTORS: u32 = (TABLE_COUNT * TABLE_STRIDE_SLOTS) as u32 + 64;
const TARGET_SIZE: u32 = 64;
const PLACED_VA: u64 = 0x1000_0000;
const PLACED_BYTES: u64 = 65536;

unsafe extern "system" fn title_alloc(bytes: usize, _attributes: u64) -> *mut c_void {
    // SAFETY: allocates committed read-write memory for the title descriptor heap mirror.
    unsafe { VirtualAlloc(Some(std::ptr::null()), bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }
}

unsafe extern "system" fn title_free(at: *mut c_void, _attributes: u64) {
    // SAFETY: frees memory allocated by `title_alloc`.
    unsafe {
        let _ = VirtualFree(at, 0, MEM_RELEASE);
    }
}

#[repr(C)]
pub struct XMemHooks {
    alloc: unsafe extern "system" fn(usize, u64) -> *mut c_void,
    free: unsafe extern "system" fn(*mut c_void, u64),
}

#[unsafe(no_mangle)]
pub static __XMemAllocationHooks__: XMemHooks = XMemHooks {
    alloc: title_alloc,
    free: title_free,
};

#[unsafe(link_section = ".drectve")]
#[used]
static EXPORT_XMEM_HOOKS: [u8; 37] = *b"/EXPORT:__XMemAllocationHooks__,DATA ";

type XboxCreate =
    unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type SerializeVersioned = unsafe extern "system" fn(
    *const D3D12_VERSIONED_ROOT_SIGNATURE_DESC,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type CreateRs = unsafe extern "system" fn(
    *mut c_void,
    u32,
    *const c_void,
    usize,
    *const GUID,
    *mut *mut c_void,
) -> i32;
type CreatePso = unsafe extern "system" fn(
    *mut c_void,
    *const c_void,
    *const GUID,
    *mut *mut c_void,
) -> i32;
type AllocUpload = unsafe extern "system" fn(*mut c_void, usize, usize) -> *mut u8;

unsafe fn vt<F: Copy>(object: *mut c_void, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: `object` is a live COM proxy and `slot` matches `F`.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(slot))
    }
}

fn find_dll() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let dir = exe.parent().expect("exe parent");
    let candidates = [
        dir.join("d3d12_xs.dll"),
        dir.parent().map(|p| p.join("d3d12_xs.dll")).unwrap_or_default(),
    ];
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| panic!("d3d12_xs.dll not found next to {}", exe.display()))
}

fn pick_adapter_name(want_warp: bool) -> (String, String) {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory6>() }) else {
        return ("unknown".into(), String::new());
    };
    let mut i = 0u32;
    while let Ok(adapter) = unsafe {
        factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
    } {
        if let Ok(desc) = unsafe { adapter.GetDesc1() } {
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
            if want_warp == software {
                return (name, i.to_string());
            }
        }
        i += 1;
    }
    ("default".into(), String::new())
}

fn compile_hlsl(source: &str, entry: &CStr, target: &CStr) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    let hr = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(entry.as_ptr().cast()),
            PCSTR(target.as_ptr().cast()),
            0,
            0,
            &mut code,
            Some(&mut err),
        )
    };
    if let Err(e) = hr {
        let msg = err
            .map(|b| unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .to_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed: {e:?} {msg}");
    }
    let blob = code.expect("shader blob");
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    }
}

use std::ffi::CStr;

fn build_root_signature(serialize: SerializeVersioned) -> Vec<u8> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 1,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
            NumDescriptors: 1,
            BaseShaderRegister: 2,
            RegisterSpace: 0,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            OffsetInDescriptorsFromTableStart: 1,
        },
    ];
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                    NumDescriptorRanges: ranges.len() as u32,
                    pDescriptorRanges: ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        },
    };
    let (mut blob, mut err): (*mut c_void, *mut c_void) =
        (std::ptr::null_mut(), std::ptr::null_mut());
    let hr = unsafe { serialize(&desc, &mut blob, &mut err) };
    if !err.is_null() {
        drop(unsafe { ID3DBlob::from_raw(err) });
    }
    assert!(hr >= 0 && !blob.is_null(), "serialize root sig hr=0x{:08x}", hr as u32);
    let blob = unsafe { ID3DBlob::from_raw(blob) };
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    }
}

struct Bench {
    _dev: *mut c_void,
    queue: *mut c_void,
    alloc: *mut c_void,
    list: *mut c_void,
    _setup_alloc: *mut c_void,
    _setup_list: *mut c_void,
    fence: ID3D12Fence,
    fence_proxy: *mut c_void,
    fence_value: u64,
    root: *mut c_void,
    pso: *mut c_void,
    heap: *mut c_void,
    heap_gpu: u64,
    increment: u32,
    rtv: usize,
    _rtv_heap: ID3D12DescriptorHeap,
    _target: ID3D12Resource,
    _placed_buf: *mut c_void,
}

impl Bench {
    fn new(xs: &Library, pc: &Library) -> Self {
        let create = unsafe { xs.get::<XboxCreate>("D3D12XboxCreateDevice") }
            .expect("D3D12XboxCreateDevice");
        let serialize = unsafe {
            pc.get::<SerializeVersioned>("D3D12SerializeVersionedRootSignature")
        }
        .expect("D3D12SerializeVersionedRootSignature");

        let mut dev: *mut c_void = std::ptr::null_mut();
        let hr = unsafe { create(std::ptr::null_mut(), std::ptr::null(), &ID3D12Device::IID, &mut dev) };
        assert!(hr >= 0 && !dev.is_null(), "D3D12XboxCreateDevice failed: 0x{:08x}", hr as u32);

        unsafe {
            let pc_dev_raw = *(dev as *const *mut c_void).add(1);
            let pc_dev = ID3D12Device::from_raw_borrowed(&pc_dev_raw).expect("pc_dev");

            let rs_blob = build_root_signature(serialize);
            let mut root: *mut c_void = std::ptr::null_mut();
            let hr = vt::<CreateRs>(dev, 16)(
                dev,
                0,
                rs_blob.as_ptr().cast(),
                rs_blob.len(),
                &ID3D12RootSignature::IID,
                &mut root,
            );
            assert!(hr >= 0 && !root.is_null(), "CreateRootSignature failed: 0x{:08x}", hr as u32);

            let vs_src = "cbuffer C0 : register(b0) { float4 c0; };
                float4 main(uint vid : SV_VertexID) : SV_POSITION {
                    float2 p = float2((vid == 1) ? 3.0 : -1.0, (vid == 2) ? 3.0 : -1.0);
                    return float4(p, 0.0, 1.0) + c0 * 0.0;
                }";
            let ps_src = "cbuffer C1 : register(b1) { float4 c1; };
                StructuredBuffer<uint> sb0 : register(t0);
                Texture2D<float4> tex1 : register(t1);
                cbuffer C2 : register(b2) { float4 c2; };
                float4 main() : SV_TARGET {
                    return c1 * 0.0 + c2 * 0.0 + float4(0.1, 0.2, 0.3, 1.0) + asfloat(sb0[0]) * 0.0 + tex1.Load(int3(0,0,0)) * 0.0;
                }";
            let vs = compile_hlsl(vs_src, c"main", c"vs_5_1");
            let ps = compile_hlsl(ps_src, c"main", c"ps_5_1");

            let stencil = D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            };
            let mut pso_desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::mem::zeroed();
            pso_desc.pRootSignature = std::mem::transmute::<
                *mut c_void,
                ManuallyDrop<Option<ID3D12RootSignature>>,
            >(root);
            pso_desc.VS = D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            };
            pso_desc.PS = D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            };
            pso_desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_ZERO,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_ZERO,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: 0xF,
                ..Default::default()
            };
            pso_desc.SampleMask = u32::MAX;
            pso_desc.RasterizerState = D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            };
            pso_desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                FrontFace: stencil,
                BackFace: stencil,
                ..Default::default()
            };
            pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso_desc.NumRenderTargets = 1;
            pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            pso_desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
            pso_desc.SampleDesc = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };

            let mut pso: *mut c_void = std::ptr::null_mut();
            let hr = vt::<CreatePso>(dev, 10)(
                dev,
                (&raw const pso_desc).cast(),
                &ID3D12PipelineState::IID,
                &mut pso,
            );
            assert!(hr >= 0 && !pso.is_null(), "CreateGraphicsPipelineState failed: 0x{:08x}", hr as u32);

            let q_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            };
            let mut queue: *mut c_void = std::ptr::null_mut();
            let hr = vt::<
                unsafe extern "system" fn(
                    *mut c_void,
                    *const D3D12_COMMAND_QUEUE_DESC,
                    *const GUID,
                    *mut *mut c_void,
                ) -> i32,
            >(dev, 8)(dev, &q_desc, &ID3D12CommandQueue::IID, &mut queue);
            assert!(hr >= 0 && !queue.is_null(), "CreateCommandQueue failed");

            let make_alloc = || {
                let mut a: *mut c_void = std::ptr::null_mut();
                let hr = vt::<
                    unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32,
                >(dev, 9)(
                    dev,
                    D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                    &ID3D12CommandAllocator::IID,
                    &mut a,
                );
                assert!(hr >= 0 && !a.is_null(), "CreateCommandAllocator failed");
                a
            };
            let alloc = make_alloc();
            let setup_alloc = make_alloc();

            type CreateList = unsafe extern "system" fn(
                *mut c_void,
                u32,
                i32,
                *mut c_void,
                *mut c_void,
                *const GUID,
                *mut *mut c_void,
            ) -> i32;
            let make_list = |a: *mut c_void| {
                let mut l: *mut c_void = std::ptr::null_mut();
                let hr = vt::<CreateList>(dev, 12)(
                    dev,
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT.0,
                    a,
                    std::ptr::null_mut(),
                    &ID3D12GraphicsCommandList::IID,
                    &mut l,
                );
                assert!(hr >= 0 && !l.is_null(), "CreateCommandList failed");
                l
            };
            let list = make_list(alloc);
            let setup_list = make_list(setup_alloc);
            let _ = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(list, 9)(list);

            let mut fence_proxy: *mut c_void = std::ptr::null_mut();
            let hr = vt::<
                unsafe extern "system" fn(
                    *mut c_void,
                    u64,
                    i32,
                    *const GUID,
                    *mut *mut c_void,
                ) -> i32,
            >(dev, 36)(dev, 0, D3D12_FENCE_FLAG_NONE.0, &ID3D12Fence::IID, &mut fence_proxy);
            assert!(hr >= 0 && !fence_proxy.is_null(), "CreateFence failed");
            let fence = ID3D12Fence::from_raw(fence_proxy);

            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: HEAP_DESCRIPTORS,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let mut heap: *mut c_void = std::ptr::null_mut();
            let hr = vt::<
                unsafe extern "system" fn(
                    *mut c_void,
                    *const D3D12_DESCRIPTOR_HEAP_DESC,
                    *const GUID,
                    *mut *mut c_void,
                ) -> i32,
            >(dev, 14)(
                dev,
                &heap_desc,
                &ID3D12DescriptorHeap::IID,
                &mut heap,
            );
            assert!(hr >= 0 && !heap.is_null(), "CreateDescriptorHeap failed");
            let heap_cpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 9)(heap);
            let heap_gpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 10)(heap);
            let increment = vt::<unsafe extern "system" fn(*mut c_void, i32) -> u32>(dev, 15)(
                dev,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0,
            );

            // Placed buffer at PLACED_VA for root CBV/SRV and table SRVs.
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let buf_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: PLACED_BYTES,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: sample,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut placed_buf: *mut c_void = std::ptr::null_mut();
            let hr = vt::<
                unsafe extern "system" fn(
                    *mut c_void,
                    u64,
                    *const D3D12_RESOURCE_DESC,
                    u32,
                    *const c_void,
                    *const GUID,
                    *mut *mut c_void,
                ) -> i32,
            >(dev, 44)(
                dev,
                PLACED_VA,
                &buf_desc,
                0,
                std::ptr::null(),
                &ID3D12Resource::IID,
                &mut placed_buf,
            );
            assert!(hr >= 0 && !placed_buf.is_null(), "CreatePlacedResourceX failed");

            // Keep a permanent upload allocation on setup_list for unaligned CBV sources.
            let cb_upload = vt::<AllocUpload>(setup_list, 148)(setup_list, 4096, 256);
            assert!(!cb_upload.is_null(), "setup upload allocation failed");
            cb_upload.write_bytes(0x3F, 4096);
            let _ = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(setup_list, 9)(setup_list);

            // Populate TABLE_COUNT descriptor tables spaced by TABLE_STRIDE_SLOTS.
            // Each table has slot 0 = buffer SRV, slot 1 = unaligned CBV (va + 16).
            let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: 256,
                        StructureByteStride: 4,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };
            for k in 0..TABLE_COUNT {
                let base_slot = (k * TABLE_STRIDE_SLOTS) as u64;
                let cpu0 = (heap_cpu + base_slot * u64::from(increment)) as usize;
                let cpu1 = (heap_cpu + (base_slot + 1) * u64::from(increment)) as usize;
                vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, usize)>(
                    dev, 18,
                )(dev, placed_buf, (&raw const srv_desc).cast(), cpu0);
                let cbv_view: [u64; 2] = [(cb_upload as u64) + 16, 256];
                vt::<unsafe extern "system" fn(*mut c_void, *const c_void, usize)>(dev, 17)(
                    dev,
                    cbv_view.as_ptr().cast(),
                    cpu1,
                );
            }

            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(TARGET_SIZE),
                Height: TARGET_SIZE,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: sample,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let default_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut target: Option<ID3D12Resource> = None;
            pc_dev
                .CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .expect("CreateCommittedResource target");
            let target = target.expect("target");
            let rtv_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            };
            let rtv_heap: ID3D12DescriptorHeap =
                pc_dev.CreateDescriptorHeap(&rtv_desc).expect("rtv heap");
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
            pc_dev.CreateRenderTargetView(&target, None, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv });

            Self {
                _dev: dev,
                queue,
                alloc,
                list,
                _setup_alloc: setup_alloc,
                _setup_list: setup_list,
                fence,
                fence_proxy,
                fence_value: 0,
                root,
                pso,
                heap,
                heap_gpu,
                increment,
                rtv,
                _rtv_heap: rtv_heap,
                _target: target,
                _placed_buf: placed_buf,
            }
        }
    }

    fn run_frame(&mut self, draws: usize) -> (f64, f64, f64) {
        unsafe {
            let hr = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(self.alloc, 8)(self.alloc);
            assert!(hr >= 0, "alloc Reset failed");
            let hr = vt::<
                unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
            >(self.list, 10)(self.list, self.alloc, self.pso);
            assert!(hr >= 0, "list Reset failed");

            let list = self.list;
            let t_rec0 = Instant::now();
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(list, 32)(
                list, 1, &self.heap,
            );
            vt::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(list, 34)(list, self.root);
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: TARGET_SIZE as f32,
                Height: TARGET_SIZE as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: TARGET_SIZE as i32,
                bottom: TARGET_SIZE as i32,
            };
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(list, 21)(
                list, 1, &viewport,
            );
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const RECT)>(list, 22)(
                list, 1, &scissor,
            );
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(
                list, 50,
            )(list, 1, &self.rtv, 0, std::ptr::null());
            vt::<unsafe extern "system" fn(*mut c_void, i32)>(list, 20)(
                list,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0,
            );

            let set_consts = vt::<
                unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32),
            >(list, 40);
            let set_cbv = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 42);
            let set_srv = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 44);
            let set_table = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 36);
            let draw = vt::<unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32)>(list, 12);

            for d in 0..draws {
                let consts = [d as u32, 0x3f80_0000, 0, 0];
                set_consts(list, 0, 4, consts.as_ptr(), 0);
                set_cbv(list, 1, PLACED_VA);
                set_srv(list, 2, PLACED_VA + 256);
                let slot = ((d % TABLE_COUNT) * TABLE_STRIDE_SLOTS) as u64;
                set_table(list, 3, self.heap_gpu + slot * u64::from(self.increment));
                draw(list, 3, 1, 0, 0);
            }

            let hr = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(list, 9)(list);
            assert!(hr >= 0, "list Close failed");
            let rec_ms = t_rec0.elapsed().as_secs_f64() * 1000.0;

            let t_exec0 = Instant::now();
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(self.queue, 10)(
                self.queue, 1, &list,
            );
            let exec_ms = t_exec0.elapsed().as_secs_f64() * 1000.0;

            self.fence_value += 1;
            let hr = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32>(
                self.queue, 14,
            )(self.queue, self.fence_proxy, self.fence_value);
            assert!(hr >= 0, "queue Signal failed");

            let t_wait0 = Instant::now();
            while self.fence.GetCompletedValue() < self.fence_value {
                if t_wait0.elapsed() > Duration::from_secs(20) {
                    panic!("fence timeout at value {}", self.fence_value);
                }
                std::hint::spin_loop();
            }
            let wait_ms = t_wait0.elapsed().as_secs_f64() * 1000.0;
            (rec_ms, exec_ms, wait_ms)
        }
    }

    fn profile_breakdown(&mut self, draws: usize) {
        unsafe {
            let _ = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(self.alloc, 8)(self.alloc);
            let _ = vt::<
                unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
            >(self.list, 10)(self.list, self.alloc, self.pso);
            let list = self.list;
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(list, 32)(
                list, 1, &self.heap,
            );
            vt::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(list, 34)(list, self.root);
            vt::<unsafe extern "system" fn(*mut c_void, i32)>(list, 20)(
                list,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0,
            );

            let set_consts = vt::<
                unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32),
            >(list, 40);
            let set_cbv = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 42);
            let set_srv = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 44);
            let set_table = vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 36);
            let draw = vt::<unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32)>(list, 12);

            let (mut ns_consts, mut ns_cbv, mut ns_srv, mut ns_table, mut ns_draw) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for d in 0..draws {
                let consts = [d as u32, 0x3f80_0000, 0, 0];
                let t0 = Instant::now();
                set_consts(list, 0, 4, consts.as_ptr(), 0);
                let t1 = Instant::now();
                set_cbv(list, 1, PLACED_VA);
                let t2 = Instant::now();
                set_srv(list, 2, PLACED_VA + 256);
                let t3 = Instant::now();
                let slot = ((d % TABLE_COUNT) * TABLE_STRIDE_SLOTS) as u64;
                set_table(list, 3, self.heap_gpu + slot * u64::from(self.increment));
                let t4 = Instant::now();
                draw(list, 3, 1, 0, 0);
                let t5 = Instant::now();
                ns_consts += (t1 - t0).as_secs_f64() * 1000.0;
                ns_cbv += (t2 - t1).as_secs_f64() * 1000.0;
                ns_srv += (t3 - t2).as_secs_f64() * 1000.0;
                ns_table += (t4 - t3).as_secs_f64() * 1000.0;
                ns_draw += (t5 - t4).as_secs_f64() * 1000.0;
            }
            let _ = vt::<unsafe extern "system" fn(*mut c_void) -> i32>(list, 9)(list);
            let t_exec0 = Instant::now();
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(self.queue, 10)(
                self.queue, 1, &list,
            );
            let exec_ms = t_exec0.elapsed().as_secs_f64() * 1000.0;
            self.fence_value += 1;
            let _ = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32>(
                self.queue, 14,
            )(self.queue, self.fence_proxy, self.fence_value);
            while self.fence.GetCompletedValue() < self.fence_value {
                std::hint::spin_loop();
            }
            let sum = ns_consts + ns_cbv + ns_srv + ns_table + ns_draw + exec_ms;
            println!("\nPhase breakdown at {draws} draws (1 sampled frame):");
            println!("  SetRoot32BitConstants : {:7.3} ms ({:5.1}%)", ns_consts, ns_consts * 100.0 / sum);
            println!("  SetRootCBV            : {:7.3} ms ({:5.1}%)", ns_cbv, ns_cbv * 100.0 / sum);
            println!("  SetRootSRV            : {:7.3} ms ({:5.1}%)", ns_srv, ns_srv * 100.0 / sum);
            println!("  SetRootDescriptorTable: {:7.3} ms ({:5.1}%)", ns_table, ns_table * 100.0 / sum);
            println!("  DrawInstanced         : {:7.3} ms ({:5.1}%)", ns_draw, ns_draw * 100.0 / sum);
            println!("  ExecuteCommandLists   : {:7.3} ms ({:5.1}%)", exec_ms, exec_ms * 100.0 / sum);
        }
    }
}

fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p50 = sorted[sorted.len() / 2];
    let p99_idx = ((sorted.len() * 99) / 100).min(sorted.len() - 1);
    let p99 = sorted[p99_idx];
    (mean, p50, p99)
}

fn main() {
    let want_gpu = std::env::args().any(|a| a == "--gpu");
    let direct = std::env::var_os("DECANT_D3D12X_DIRECT").is_none_or(|v| v != "0");
    if !want_warp_only_check(want_gpu) {
        eprintln!("Refusing --gpu run because D:\\decant_runs\\.lock exists (a game run is active).");
        std::process::exit(1);
    }

    let (adapter_name, adapter_idx) = pick_adapter_name(!want_gpu);
    unsafe {
        if direct {
            std::env::set_var("DECANT_D3D12X_DIRECT", "1");
        }
        if !adapter_idx.is_empty() {
            std::env::set_var("DECANT_D3D12X_ADAPTER", &adapter_idx);
        }
    }

    let dll_path = find_dll();
    let xs = Library::open(dll_path.as_os_str()).expect("open d3d12_xs.dll");
    let pc = Library::open(OsStr::new("d3d12.dll")).expect("open d3d12.dll");
    let hooks_ok = unsafe {
        let h = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null());
        windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            h,
            c"__XMemAllocationHooks__".as_ptr().cast(),
        )
        .is_some()
    };

    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    println!(
        "draw-bench: adapter={adapter_name} ({}), profile={profile}, direct={}, xmem_hooks={hooks_ok}",
        if want_gpu { "hardware" } else { "WARP" },
        if direct { "on" } else { "off" }
    );

    let mut bench = Bench::new(&xs, &pc);

    println!(
        "{:>6} | {:>10} {:>10} {:>10} | {:>9} | {:>10} {:>10} {:>10}",
        "draws", "mean_ms", "p50_ms", "p99_ms", "us/draw", "rec_ms", "exec_ms", "wait_ms"
    );
    println!("{}", "-".repeat(86));

    for &n in &DRAW_COUNTS {
        for _ in 0..WARMUP_FRAMES {
            let _ = bench.run_frame(n);
        }
        let mut total_samples = Vec::with_capacity(MEASURED_FRAMES);
        let mut rec_sum = 0.0;
        let mut exec_sum = 0.0;
        let mut wait_sum = 0.0;
        for _ in 0..MEASURED_FRAMES {
            let (rec, exec, wait) = bench.run_frame(n);
            total_samples.push(rec + exec + wait);
            rec_sum += rec;
            exec_sum += exec;
            wait_sum += wait;
        }
        let (mean, p50, p99) = stats(&total_samples);
        let us_per_draw = (mean * 1000.0) / n as f64;
        let f = MEASURED_FRAMES as f64;
        println!(
            "{n:>6} | {mean:>10.3} {p50:>10.3} {p99:>10.3} | {us_per_draw:>9.2} | {:>10.3} {:>10.3} {:>10.3}",
            rec_sum / f,
            exec_sum / f,
            wait_sum / f
        );
    }

    bench.profile_breakdown(3000);
}

fn want_warp_only_check(want_gpu: bool) -> bool {
    if !want_gpu {
        return true;
    }
    !Path::new(r"D:\decant_runs\.lock").exists()
}
