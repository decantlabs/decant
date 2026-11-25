//! Rendering harness: a vertex and a pixel container drawn into a 64 by 64 `R8G8B8A8_UNORM`
//! target (one triangle), with an optional 2D texture and a point sampler for the pixel shader.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::time::Duration;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::core::PCSTR;

use super::gpu::{Gpu, buffer, transition, upload};

pub const TARGET_SIZE: u32 = 64;

/// One vertex attribute: the semantic name (NUL terminated), its index, format and byte offset.
pub struct Attribute {
    pub name: &'static [u8],
    pub index: u32,
    pub format: DXGI_FORMAT,
    pub offset: u32,
}

/// A constant buffer bound as a root descriptor visible to every stage, at register `b<register>`
/// of `space`.
pub struct Constants<'a> {
    pub register: u32,
    pub space: u32,
    pub bytes: &'a [u8],
}

/// A small RGBA8 texture for the pixel shader (`t0`, with a point sampler with clamping at `s0`).
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

fn location_resource(res: &ID3D12Resource, sub: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed copy of the interface pointer, never released (ManuallyDrop).
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
    }
}

fn location_footprint(res: &ID3D12Resource, width: u32, height: u32, pitch: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: width, Height: height, Depth: 1, RowPitch: pitch },
            },
        },
    }
}

fn texture_2d(device: &ID3D12Device, width: u32, height: u32, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        ..Default::default()
    };
    let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0, 0.0, 0.0, 0.0] } };
    let mut res: Option<ID3D12Resource> = None;
    let clear_ref = if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 != 0 { Some(&clear as *const _) } else { None };
    // SAFETY: valid descriptions, the result is checked.
    unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, clear_ref, &mut res).expect("CreateCommittedResource texture") };
    res.expect("texture")
}

impl Gpu {
    /// Draws `count` vertices of `vertices` (stride `stride`) with the containers `vs` and `ps` and
    /// returns the 64 by 64 target as RGBA8 bytes, rows tightly packed.
    #[allow(clippy::too_many_arguments)]
    pub fn render(&self, vs: &[u8], ps: &[u8], attributes: &[Attribute], stride: u32, vertices: &[u8], count: u32, texture: Option<&Texture>) -> Result<Vec<u8>, String> {
        self.render_with(vs, ps, attributes, stride, vertices, count, texture, None)
    }

    /// As [`Gpu::render`], with an optional constant buffer bound as a root descriptor after the
    /// texture table.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with(&self, vs: &[u8], ps: &[u8], attributes: &[Attribute], stride: u32, vertices: &[u8], count: u32, texture: Option<&Texture>, constants: Option<&Constants>) -> Result<Vec<u8>, String> {
        let device = &self.device;
        // SAFETY: standard command recording. Every resource outlives the wait on the fence.
        unsafe {
            // Root signature: one SRV table and a static point sampler when there is a texture.
            let range = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 1, BaseShaderRegister: 0, RegisterSpace: 0, OffsetInDescriptorsFromTableStart: 0 };
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxLOD: f32::MAX,
                ShaderRegister: 0,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let cbv = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: constants.map_or(0, |c| c.register), RegisterSpace: constants.map_or(0, |c| c.space) } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let mut params = Vec::new();
            if texture.is_some() {
                params.push(param);
            }
            if constants.is_some() {
                params.push(cbv);
            }
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: if params.is_empty() { std::ptr::null() } else { params.as_ptr() },
                NumStaticSamplers: texture.is_some() as u32,
                pStaticSamplers: if texture.is_some() { &sampler } else { std::ptr::null() },
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            };
            let mut blob = None;
            let mut err = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, Some(&mut err)).map_err(|e| format!("serialize: {e}"))?;
            let blob: windows::Win32::Graphics::Direct3D::ID3DBlob = blob.ok_or("no root signature")?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            let rs = device.CreateRootSignature::<ID3D12RootSignature>(0, bytes).map_err(|e| format!("CreateRootSignature: {e}"))?;

            // Pipeline state.
            let elements: Vec<D3D12_INPUT_ELEMENT_DESC> = attributes
                .iter()
                .map(|a| D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: PCSTR(a.name.as_ptr()),
                    SemanticIndex: a.index,
                    Format: a.format,
                    InputSlot: 0,
                    AlignedByteOffset: a.offset,
                    InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                    InstanceDataStepRate: 0,
                })
                .collect();
            let mut pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            pso.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
            let state = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&pso);
            drop(std::mem::ManuallyDrop::into_inner(pso.pRootSignature));
            let state = state.map_err(|e| format!("CreateGraphicsPipelineState: {e}"))?;

            // Target, vertex buffer, readback.
            let target = texture_2d(device, TARGET_SIZE, TARGET_SIZE, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET);
            let rtv_heap = device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 }).map_err(|e| e.to_string())?;
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&target, None, rtv);
            let vb = upload(device, vertices)?;
            let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: vb.GetGPUVirtualAddress(), SizeInBytes: vertices.len() as u32, StrideInBytes: stride };
            let pitch = TARGET_SIZE * 4;
            let readback = buffer(device, D3D12_HEAP_TYPE_READBACK, (pitch * TARGET_SIZE) as usize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;

            let alloc = device.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
            let list = device.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &state).map_err(|e| e.to_string())?;

            // The texture: uploaded with a copy, then an SRV in a shader visible heap.
            let mut keep: Vec<ID3D12Resource> = vec![vb];
            let srv_heap = match texture {
                Some(t) => {
                    let tex = texture_2d(device, t.width, t.height, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
                    let row = (t.width * 4).next_multiple_of(256);
                    let mut padded = vec![0u8; (row * t.height) as usize];
                    for y in 0..t.height as usize {
                        let (s, d) = (y * t.width as usize * 4, y * row as usize);
                        padded[d..d + t.width as usize * 4].copy_from_slice(&t.rgba[s..s + t.width as usize * 4]);
                    }
                    let up = upload(device, &padded)?;
                    let dst = location_resource(&tex, 0);
                    let src = location_footprint(&up, t.width, t.height, row);
                    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    list.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)]);
                    let heap = device
                        .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
                        .map_err(|e| e.to_string())?;
                    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 0x1688,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
                    };
                    device.CreateShaderResourceView(&tex, Some(&srv), heap.GetCPUDescriptorHandleForHeapStart());
                    keep.push(tex);
                    keep.push(up);
                    Some(heap)
                }
                None => None,
            };

            list.SetGraphicsRootSignature(&rs);
            if let Some(h) = &srv_heap {
                list.SetDescriptorHeaps(&[Some(h.clone())]);
                list.SetGraphicsRootDescriptorTable(0, h.GetGPUDescriptorHandleForHeapStart());
            }
            if let Some(c) = constants {
                let mut padded = c.bytes.to_vec();
                padded.resize(c.bytes.len().next_multiple_of(256), 0);
                let cb = upload(device, &padded)?;
                list.SetGraphicsRootConstantBufferView(params.len() as u32 - 1, cb.GetGPUVirtualAddress());
                keep.push(cb);
            }
            list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: TARGET_SIZE as f32, Height: TARGET_SIZE as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: TARGET_SIZE as i32, bottom: TARGET_SIZE as i32 }]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.IASetVertexBuffers(0, Some(&[view]));
            list.DrawInstanced(count, 1, 0, 0);
            list.ResourceBarrier(&[transition(&target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = location_footprint(&readback, TARGET_SIZE, TARGET_SIZE, pitch);
            let src = location_resource(&target, 0);
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            self.submit(&list, Duration::from_secs(20))?;
            let mut p: *mut c_void = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
            let out = std::slice::from_raw_parts(p.cast::<u8>(), (pitch * TARGET_SIZE) as usize).to_vec();
            readback.Unmap(0, None);
            drop(keep);
            Ok(out)
        }
    }
}

/// The RGBA bytes of pixel `(x, y)` of a rendered target.
pub fn pixel(target: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * TARGET_SIZE + x) * 4) as usize;
    [target[at], target[at + 1], target[at + 2], target[at + 3]]
}
