//! Every Direct3D 12 call of `vkd3d_test`, through the `windows` crate, in the order of the
//! former C++ bench.
//!
//! No object is ever released (see [`leak`]): the C++ released nothing and the process ends right
//! after the image is written, so vkd3d-proton never runs its teardown here either.
#![allow(unsafe_code)]

use std::ffi::{OsString, c_void};
use std::io::Write;
use std::mem::ManuallyDrop;
use std::ptr;

use windows::Win32::Foundation::{HANDLE, RECT, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
};
use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject};
use windows::core::{GUID, HRESULT, HSTRING, Interface, PCSTR, s};

use decant_shader_bench::{
    Exit, Vertex, Vkd3dOptions, cfmt, d3d12core_beside, floats, out, pad256, pattern_texel,
    read_file,
};

const SIZE: u32 = 64;
const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R32G32B32A32_FLOAT;
const TEXTURE_SIZE: u32 = 4;

type CreateDevice = unsafe extern "system" fn(
    *mut c_void,
    D3D_FEATURE_LEVEL,
    *const GUID,
    *mut *mut c_void,
) -> HRESULT;
type Serialize = unsafe extern "system" fn(
    *const D3D12_ROOT_SIGNATURE_DESC,
    D3D_ROOT_SIGNATURE_VERSION,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HRESULT;

/// An export as `GetProcAddress` returns it.
type Proc = unsafe extern "system" fn() -> isize;

fn failed(what: &str, code: HRESULT) -> Exit {
    Exit::new(4, format!("echec {what} : 0x{:08x}", code.0 as u32))
}

fn check<T>(result: windows::core::Result<T>, what: &str) -> Result<T, Exit> {
    result.map_err(|e| failed(what, e.code()))
}

/// Keeps one reference forever, so the object is never destroyed, as in the C++.
fn leak<T: Interface>(object: T) -> T {
    std::mem::forget(object.clone());
    object
}

/// A borrowed interface in a structure field, without taking a reference.
fn field<T: Interface>(object: &T) -> ManuallyDrop<Option<T>> {
    // SAFETY: the field is never dropped (ManuallyDrop) and `object` outlives every use of it.
    unsafe { std::mem::transmute_copy(object) }
}

fn heap(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    }
}

fn buffer(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

fn texture_2d(width: u32, format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(width),
        Height: width,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: field(resource),
                Subresource: 0,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn committed(
    device: &ID3D12Device,
    heap: &D3D12_HEAP_PROPERTIES,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    clear: Option<&D3D12_CLEAR_VALUE>,
    what: &str,
) -> Result<ID3D12Resource, Exit> {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: the descriptions are valid for the call and `resource` receives the new interface.
    let result = unsafe {
        device.CreateCommittedResource(
            heap,
            D3D12_HEAP_FLAG_NONE,
            desc,
            state,
            clear.map(ptr::from_ref),
            &mut resource,
        )
    };
    check(result, what)?;
    resource.map(leak).ok_or_else(|| failed(what, HRESULT(0)))
}

/// Maps subresource 0 of an upload or readback resource.
fn map(resource: &ID3D12Resource, what: &str) -> Result<*mut u8, Exit> {
    let mut mapped = ptr::null_mut();
    // SAFETY: CPU visible resource, whole range, the pointer receives the mapping.
    check(unsafe { resource.Map(0, None, Some(&mut mapped)) }, what)?;
    Ok(mapped.cast::<u8>())
}

/// Copies `data` into an upload resource of at least that size.
fn upload(resource: &ID3D12Resource, data: &[u8], what: &str) -> Result<(), Exit> {
    let mapped = map(resource, what)?;
    // SAFETY: the resource was created with at least `data.len()` bytes and is mapped.
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), mapped, data.len());
        resource.Unmap(0, None);
    }
    Ok(())
}

fn load(path: &str) -> Result<windows::Win32::Foundation::HMODULE, Exit> {
    // SAFETY: loading the vkd3d-proton library runs its initialisation, which is what the bench is
    // for. It is never unloaded.
    unsafe { LoadLibraryExW(&HSTRING::from(path), None, LOAD_WITH_ALTERED_SEARCH_PATH) }.map_err(
        |e| {
            // The C++ printed GetLastError(), the error carries it as HRESULT_FROM_WIN32.
            let code = e.code().0 as u32;
            let last = if code & 0xFFFF_0000 == 0x8007_0000 {
                code & 0xFFFF
            } else {
                code
            };
            Exit::new(3, format!("chargement impossible : {path} (erreur {last})"))
        },
    )
}

/// Builds the pipeline from the game containers, draws the triangle, writes the image and prints
/// the first pixel. Returns the exit code (0, or 2 when the image could not be written).
pub fn render(args: &[OsString]) -> Result<i32, Exit> {
    // d3d12.dll looks for "d3d12core.dll" by name: the one of vkd3d-proton (same folder) is loaded
    // first by its full path, otherwise Windows would take Microsoft's D3D12Core.dll.
    let dll = args[1].to_string_lossy().into_owned();
    load(&d3d12core_beside(&dll))?;
    let library = load(&dll)?;
    // SAFETY: looking up two exports of the loaded module by their NUL terminated names.
    let (create_device, serialize) = unsafe {
        (
            GetProcAddress(library, s!("D3D12CreateDevice")),
            GetProcAddress(library, s!("D3D12SerializeRootSignature")),
        )
    };
    let (Some(create_device), Some(serialize)) = (create_device, serialize) else {
        return Err(Exit::new(
            3,
            "exports Direct3D 12 absents de la bibliotheque",
        ));
    };
    // SAFETY: both exports have these signatures in d3d12.h, which vkd3d-proton implements.
    let (create_device, serialize): (CreateDevice, Serialize) = unsafe {
        (
            std::mem::transmute::<Proc, CreateDevice>(create_device),
            std::mem::transmute::<Proc, Serialize>(serialize),
        )
    };
    let vs = read_file(&args[2])?;
    let ps = read_file(&args[3])?;

    let mut raw = ptr::null_mut();
    // SAFETY: no adapter (default), a valid interface id, and `raw` receives the device.
    let hr = unsafe {
        create_device(
            ptr::null_mut(),
            D3D_FEATURE_LEVEL_12_0,
            &ID3D12Device::IID,
            &mut raw,
        )
    };
    if hr.is_err() {
        return Err(failed("D3D12CreateDevice", hr));
    }
    // SAFETY: on success `raw` holds one reference to an ID3D12Device, now owned here.
    let device = leak(unsafe { ID3D12Device::from_raw(raw) });

    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };
    // SAFETY: valid description.
    let queue: ID3D12CommandQueue = leak(check(
        unsafe { device.CreateCommandQueue(&queue_desc) },
        "CreateCommandQueue",
    )?);
    // SAFETY: plain creation.
    let allocator: ID3D12CommandAllocator = leak(check(
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) },
        "CreateCommandAllocator",
    )?);

    // Option --cbv <register> <file>: a constant buffer as root parameter 0, visible to pixels.
    // Option --srv <register>: the 4 x 4 texture of decant_shader_run in a descriptor table as root
    // parameter 0, with a static sampler s0 "nearest, clamped to the edges".
    let options = Vkd3dOptions::parse(args);
    let cbv_data = match &options.cbv_file {
        Some(path) => pad256(read_file(path)?), // size must be a multiple of 256
        None => Vec::new(),
    };
    let (cbv_register, srv_register) = (options.cbv_register, options.srv_register);
    let range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: srv_register.max(0) as u32,
        ..Default::default()
    };
    let static_sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        MaxLOD: D3D12_FLOAT32_MAX,
        ShaderRegister: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let mut root_parameter = D3D12_ROOT_PARAMETER::default();
    if srv_register >= 0 {
        root_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        root_parameter.Anonymous.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: 1,
            pDescriptorRanges: &range,
        };
    } else {
        root_parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        root_parameter.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: cbv_register.max(0) as u32,
            RegisterSpace: 0,
        };
    }
    root_parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: if cbv_register < 0 && srv_register < 0 {
            0
        } else {
            1
        },
        pParameters: &root_parameter,
        NumStaticSamplers: if srv_register < 0 { 0 } else { 1 },
        pStaticSamplers: &static_sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    let mut blob_raw = ptr::null_mut();
    // SAFETY: the description and what it points at live for the call, no error blob is asked.
    let hr = unsafe {
        serialize(
            &root_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob_raw,
            ptr::null_mut(),
        )
    };
    if hr.is_err() {
        return Err(failed("D3D12SerializeRootSignature", hr));
    }
    // SAFETY: on success `blob_raw` holds one reference to an ID3DBlob, now owned here.
    let blob = leak(unsafe { ID3DBlob::from_raw(blob_raw) });
    // SAFETY: the blob owns `GetBufferSize()` bytes at `GetBufferPointer()` while it lives.
    let blob_bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    // SAFETY: a serialized root signature.
    let root: ID3D12RootSignature = leak(check(
        unsafe { device.CreateRootSignature(0, blob_bytes) },
        "CreateRootSignature",
    )?);

    let element = |name: PCSTR, format: DXGI_FORMAT, offset: u32| D3D12_INPUT_ELEMENT_DESC {
        SemanticName: name,
        SemanticIndex: 0,
        Format: format,
        InputSlot: 0,
        AlignedByteOffset: offset,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    };
    let layout = [
        element(
            s!("POSITION"),
            DXGI_FORMAT_R32G32B32_FLOAT,
            Vertex::POSITION,
        ),
        element(s!("COLOR"), DXGI_FORMAT_R32G32B32A32_FLOAT, Vertex::COLOR),
        element(s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, Vertex::UV),
    ];
    // The game containers are given as read, unchanged.
    let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: field(&root),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr().cast(),
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr().cast(),
            BytecodeLength: ps.len(),
        },
        SampleMask: 0xFFFF_FFFF,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: 3,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = true.into();
    pso.RTVFormats[0] = FORMAT;
    // SAFETY: the description, the bytecode and the input layout live for the call.
    let pipeline: ID3D12PipelineState = leak(check(
        unsafe { device.CreateGraphicsPipelineState(&pso) },
        "CreateGraphicsPipelineState",
    )?);
    out("pipeline cree a partir des conteneurs du jeu\n");

    let target_desc = texture_2d(SIZE, FORMAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    let default_heap = heap(D3D12_HEAP_TYPE_DEFAULT);
    let clear_color = [-1.0f32; 4];
    let clear = D3D12_CLEAR_VALUE {
        Format: FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let target = committed(
        &device,
        &default_heap,
        &target_desc,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        Some(&clear),
        "CreateCommittedResource(cible)",
    )?;
    let rtv_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        ..Default::default()
    };
    // SAFETY: valid description.
    let rtv_heap: ID3D12DescriptorHeap = leak(check(
        unsafe { device.CreateDescriptorHeap(&rtv_desc) },
        "CreateDescriptorHeap",
    )?);
    // SAFETY: the heap is live, the view goes into its first descriptor.
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    // SAFETY: a render target resource and a descriptor of an RTV heap.
    unsafe { device.CreateRenderTargetView(&target, None, rtv) };

    let vertices = [
        Vertex {
            position: [-1.0, 1.0, 0.0],
            color: [1.00, 0.00, 0.25, 1.00],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [3.0, 1.0, 0.0],
            color: [0.00, 1.00, 0.50, 0.50],
            uv: [2.0, 0.0],
        },
        Vertex {
            position: [-1.0, -3.0, 0.0],
            color: [0.50, 0.25, 1.00, 0.75],
            uv: [0.0, 2.0],
        },
    ];
    let vertex_bytes = Vertex::bytes(&vertices);
    let upload_heap = heap(D3D12_HEAP_TYPE_UPLOAD);
    let vertex_desc = buffer(vertex_bytes.len() as u64);
    let vertex_buffer = committed(
        &device,
        &upload_heap,
        &vertex_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        None,
        "CreateCommittedResource(sommets)",
    )?;
    upload(&vertex_buffer, &vertex_bytes, "Map(sommets)")?;
    let vertex_view = D3D12_VERTEX_BUFFER_VIEW {
        // SAFETY: a buffer resource.
        BufferLocation: unsafe { vertex_buffer.GetGPUVirtualAddress() },
        SizeInBytes: vertex_bytes.len() as u32,
        StrideInBytes: Vertex::SIZE,
    };

    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut readback_size = 0u64;
    // SAFETY: one subresource, outputs written into locals.
    unsafe {
        device.GetCopyableFootprints(
            &target_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut readback_size),
        )
    };
    let readback_heap = heap(D3D12_HEAP_TYPE_READBACK);
    let readback_desc = buffer(readback_size);
    let readback = committed(
        &device,
        &readback_heap,
        &readback_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        "CreateCommittedResource(lecture)",
    )?;

    // SAFETY: a fresh allocator and a pipeline of this device.
    let list: ID3D12GraphicsCommandList = leak(check(
        unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pipeline)
        },
        "CreateCommandList",
    )?);
    let viewport = D3D12_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: SIZE as f32,
        Height: SIZE as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: SIZE as i32,
        bottom: SIZE as i32,
    };
    // SAFETY: recording into the open list with a root signature of this device.
    unsafe { list.SetGraphicsRootSignature(&root) };
    if srv_register >= 0 {
        let texture_desc = texture_2d(
            TEXTURE_SIZE,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_NONE,
        );
        let texture = committed(
            &device,
            &default_heap,
            &texture_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            "CreateCommittedResource(texture)",
        )?;
        let mut texture_footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut staging_size = 0u64;
        // SAFETY: one subresource, outputs written into locals.
        unsafe {
            device.GetCopyableFootprints(
                &texture_desc,
                0,
                1,
                0,
                Some(&mut texture_footprint),
                None,
                None,
                Some(&mut staging_size),
            )
        };
        let staging_desc = buffer(staging_size);
        let staging = committed(
            &device,
            &upload_heap,
            &staging_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            "CreateCommittedResource(texels)",
        )?;
        let texels = map(&staging, "Map(texels)")?;
        for j in 0..TEXTURE_SIZE {
            let row = texture_footprint.Offset as usize
                + (j * texture_footprint.Footprint.RowPitch) as usize;
            for i in 0..TEXTURE_SIZE {
                let texel = pattern_texel(j * TEXTURE_SIZE + i);
                // SAFETY: the footprint places 4 rows of RowPitch bytes inside the `staging_size` bytes.
                unsafe {
                    ptr::copy_nonoverlapping(texel.as_ptr(), texels.add(row + (i * 4) as usize), 4)
                };
            }
        }
        // SAFETY: the staging buffer is mapped.
        unsafe { staging.Unmap(0, None) };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: field(&staging),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: texture_footprint,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: field(&texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let srv_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            ..Default::default()
        };
        // SAFETY: recording into the open list, both resources stay alive (leaked).
        unsafe {
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        // SAFETY: valid description.
        let srv_heap: ID3D12DescriptorHeap = leak(check(
            unsafe { device.CreateDescriptorHeap(&srv_desc) },
            "CreateDescriptorHeap(textures)",
        )?);
        // SAFETY: a texture resource, a descriptor of a shader visible heap, then recording.
        unsafe {
            device.CreateShaderResourceView(
                &texture,
                None,
                srv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            list.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            list.SetGraphicsRootDescriptorTable(0, srv_heap.GetGPUDescriptorHandleForHeapStart());
        }
    }
    if cbv_register >= 0 {
        let constants_desc = buffer(cbv_data.len() as u64);
        let constants = committed(
            &device,
            &upload_heap,
            &constants_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            "CreateCommittedResource(constantes)",
        )?;
        upload(&constants, &cbv_data, "Map(constantes)")?;
        // SAFETY: recording, the buffer stays alive (leaked).
        unsafe { list.SetGraphicsRootConstantBufferView(0, constants.GetGPUVirtualAddress()) };
    }
    let source = D3D12_TEXTURE_COPY_LOCATION {
        pResource: field(&target),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let destination = D3D12_TEXTURE_COPY_LOCATION {
        pResource: field(&readback),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    // SAFETY: recording into the open list with objects of this device that are never released.
    unsafe {
        list.RSSetViewports(&[viewport]);
        list.RSSetScissorRects(&[scissor]);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.ClearRenderTargetView(rtv, &clear_color, None);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.IASetVertexBuffers(0, Some(&[vertex_view]));
        list.DrawInstanced(3, 1, 0, 0);
        list.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
    }
    // SAFETY: the list is open.
    check(unsafe { list.Close() }, "Close")?;
    // SAFETY: a closed list of this device.
    unsafe { queue.ExecuteCommandLists(&[Some(list.clone().into())]) };
    // SAFETY: plain creation.
    let fence: ID3D12Fence = leak(check(
        unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) },
        "CreateFence",
    )?);
    // SAFETY: a fence of this device.
    check(unsafe { queue.Signal(&fence, 1) }, "Signal")?;
    // SAFETY: an unnamed auto reset event, a failure gives a null handle, as in the C++.
    let event =
        unsafe { CreateEventA(None, false, false, PCSTR::null()) }.unwrap_or(HANDLE::default());
    // SAFETY: the fence is live and the event handle stays open.
    check(
        unsafe { fence.SetEventOnCompletion(1, event) },
        "SetEventOnCompletion",
    )?;
    // SAFETY: waiting on the handle created above.
    let waited = unsafe { WaitForSingleObject(event, 20000) };
    // SAFETY: the fence is live.
    if waited != WAIT_OBJECT_0 || unsafe { fence.GetCompletedValue() } < 1 {
        // Without this check, a stuck card would give an image of zeros, taken for a rendering error.
        // SAFETY: the fence is live.
        let completed = unsafe { fence.GetCompletedValue() };
        return Err(Exit::new(
            5,
            format!(
                "echec : le travail du GPU n'est pas termine apres 20 s (attente {}, barriere {completed})",
                waited.0
            ),
        ));
    }

    let mapped = map(&readback, "Map(lecture)")?;
    let row_bytes = (SIZE * 4 * 4) as usize;
    let offset = footprint.Offset as usize;
    let pitch = footprint.Footprint.RowPitch as usize;
    // SAFETY: the readback buffer holds `readback_size` bytes, which cover every row of the footprint.
    let rows: Vec<&[u8]> = (0..SIZE as usize)
        .map(|y| unsafe { std::slice::from_raw_parts(mapped.add(offset + y * pitch), row_bytes) })
        .collect();
    let written = std::fs::File::create(&args[4])
        .and_then(|mut file| rows.iter().try_for_each(|row| file.write_all(row)))
        .is_ok();
    let first = floats(&rows[0][..16]);
    out(&format!("pixel (0,0) : {}\n", cfmt::fixed4_rgba(&first)));
    Ok(if written { 0 } else { 2 })
}
