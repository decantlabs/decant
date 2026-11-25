//! Runs one side of a plan on a device: creates the generated resources, the pipeline and the
//! descriptors, draws or dispatches once and reads back every writable resource.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::time::Duration;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_POINTLIST, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::device::{Gpu, buffer, read_back, transition, upload};
use super::root::serialize;
use crate::compare::{Elem, Output, Sub};
use crate::input::{Pattern, Rng, buffer_bytes, f16_bits, texel};
use crate::model::{Class, Component, Stage};
use crate::plan::{BufView, Dim, Format, MIRROR_REGISTER, MIRROR_SPACE, Param, Plan, Res, WINDOW_SPACE, res_of, vertex_value};

/// How long one side may run on the GPU.
pub const DEADLINE: Duration = Duration::from_secs(10);

/// Slots of the descriptor heap before the tables when the shaders index it directly. Those of known
/// kind get a resource, the others a null descriptor.
const HEAP_EXTRA: u32 = 1024;
/// Samplers in the sampler heap when the shaders index it directly.
const SAMPLER_HEAP_SIZE: u32 = 16;

fn dxgi(f: Format) -> DXGI_FORMAT {
    match f {
        Format::Rgba32F => DXGI_FORMAT_R32G32B32A32_FLOAT,
        Format::Rgba32U => DXGI_FORMAT_R32G32B32A32_UINT,
        Format::Rgba32I => DXGI_FORMAT_R32G32B32A32_SINT,
        Format::Rgba16F => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Format::Rgba8 => DXGI_FORMAT_R8G8B8A8_UNORM,
    }
}

/// The bytes of one texel of `f` from four channels in `[0, 1]`.
fn encode_texel(f: Format, v: [f32; 4], out: &mut Vec<u8>) {
    for c in v {
        match f {
            Format::Rgba32F => out.extend_from_slice(&c.to_le_bytes()),
            Format::Rgba16F => out.extend_from_slice(&f16_bits(c).to_le_bytes()),
            Format::Rgba8 => out.push((c.clamp(0.0, 1.0) * 255.0).round() as u8),
            Format::Rgba32U | Format::Rgba32I => out.extend_from_slice(&((c * 255.0).round() as u32).to_le_bytes()),
        }
    }
}

/// The space [`key`] gives the slots of the descriptor heap.
const HEAP_SPACE: u32 = u32::MAX;

fn key(class: Class, space: u32, register: u32) -> String {
    if space == HEAP_SPACE {
        return format!("{}{register} heap", class.letter());
    }
    format!("{}{register} space{space}", class.letter())
}

/// A created resource and how to read it back.
struct Made {
    res: ID3D12Resource,
    /// The state it is left in for the draw or dispatch.
    state: D3D12_RESOURCE_STATES,
    /// The output it becomes when it is writable.
    output: Option<(String, Readback)>,
}

#[derive(Clone)]
enum Readback {
    Buffer { bytes: u32, record: Vec<Elem> },
    Texture { desc: D3D12_RESOURCE_DESC, format: Format, dim: Dim },
}

struct Ctx<'a> {
    gpu: &'a Gpu,
    list: &'a ID3D12GraphicsCommandList,
    plan: &'a Plan,
    keep: Vec<ID3D12Resource>,
}

impl Ctx<'_> {
    fn device(&self) -> &ID3D12Device {
        &self.gpu.device
    }

    /// A default heap buffer holding `data`, left in `state`.
    fn filled_buffer(&mut self, data: &[u8], uav: bool, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
        let flags = if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE };
        let r = buffer(self.device(), D3D12_HEAP_TYPE_DEFAULT, data.len(), flags, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let up = upload(self.device(), data)?;
        // SAFETY: both resources are alive until the list has run (`keep`).
        unsafe {
            self.list.CopyBufferRegion(&r, 0, &up, 0, data.len() as u64);
            self.list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
        self.keep.push(up);
        Ok(r)
    }

    fn texture(&mut self, k: &str, dim: Dim, size: [u32; 3], mips: u32, format: Format, uav: bool) -> Result<(ID3D12Resource, D3D12_RESOURCE_DESC), String> {
        let dimension = match dim {
            Dim::D1 | Dim::D1Array => D3D12_RESOURCE_DIMENSION_TEXTURE1D,
            Dim::D3 => D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            _ => D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: dimension,
            Width: u64::from(size[0]),
            Height: size[1],
            DepthOrArraySize: size[2] as u16,
            MipLevels: mips as u16,
            Format: dxgi(format),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
            ..Default::default()
        };
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut res: Option<ID3D12Resource> = None;
        // SAFETY: valid descriptions, the result is checked.
        unsafe { self.device().CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut res) }.map_err(|e| format!("CreateCommittedResource texture {k}: {e}"))?;
        let tex = res.ok_or("no texture")?;
        let subs = subresources(&desc);
        let (layouts, total) = footprints(self.device(), &desc, subs);
        let mut staging = vec![0u8; total as usize];
        let pattern = Pattern::for_key(self.plan.options.seed, k);
        for (i, l) in layouts.iter().enumerate() {
            let fp = l.Footprint;
            let mut rng = Rng::new(self.plan.options.seed, &format!("{k} sub{i}"));
            let mut row = Vec::new();
            for z in 0..fp.Depth {
                for y in 0..fp.Height {
                    row.clear();
                    for x in 0..fp.Width {
                        encode_texel(format, texel(pattern, &mut rng, i as u32, x, y, z, [fp.Width, fp.Height, fp.Depth]), &mut row);
                    }
                    let at = l.Offset as usize + (z * fp.Height + y) as usize * fp.RowPitch as usize;
                    staging[at..at + row.len()].copy_from_slice(&row);
                }
            }
        }
        let up = upload(self.device(), &staging)?;
        for (i, l) in layouts.iter().enumerate() {
            let dst = location_sub(&tex, i as u32);
            let src = location_placed(&up, *l);
            // SAFETY: both resources outlive the list.
            unsafe { self.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        self.keep.push(up);
        let state = if uav { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE };
        // SAFETY: as above.
        unsafe { self.list.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_COPY_DEST, state)]) };
        Ok((tex, desc))
    }

    /// Creates the resource `res` for register `register` of `space`, with its view at `cpu` when given.
    fn make(&mut self, class: Class, space: u32, register: u32, res: &Res, cpu: Option<D3D12_CPU_DESCRIPTOR_HANDLE>, made: &BTreeMap<String, Made>) -> Result<Option<Made>, String> {
        let k = key(class, space, register);
        let seed = self.plan.options.seed;
        let srv_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        match res {
            Res::Null => {
                if let Some(cpu) = cpu {
                    null_view(self.device(), class, cpu);
                }
                Ok(None)
            }
            Res::Cbv { bytes } => {
                // The user data mirror of a shader that indexes the descriptor heap holds the heap
                // indices: filled as the heap map assumed (see heapmap::mirror_bytes).
                let heap = self.plan.root.flags & (crate::plan::FLAG_RESOURCE_HEAP | crate::plan::FLAG_SAMPLER_HEAP) != 0;
                let mut data = if heap && space == MIRROR_SPACE && register == MIRROR_REGISTER { crate::heapmap::mirror_bytes(*bytes as usize) } else { buffer_bytes(seed, &k, *bytes as usize) };
                if let Some((_, _, given)) = self.plan.options.constants.iter().find(|c| c.0 == space && c.1 == register) {
                    data = given.clone();
                    data.resize((*bytes as usize).max(given.len()), 0);
                }
                // The window convention: b0 of the window space holds the address of u0 there.
                if space == WINDOW_SPACE && register == 0
                    && let Some(w) = made.get(&key(Class::Uav, WINDOW_SPACE, 0))
                {
                    // SAFETY: a plain query.
                    let va = unsafe { w.res.GetGPUVirtualAddress() };
                    data[..8].copy_from_slice(&va.to_le_bytes());
                }
                data.resize(data.len().next_multiple_of(256), 0);
                let r = upload(self.device(), &data)?;
                if let Some(cpu) = cpu {
                    // SAFETY: the view points at a live buffer.
                    unsafe { self.device().CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: r.GetGPUVirtualAddress(), SizeInBytes: data.len().min(65536) as u32 }), cpu) };
                }
                Ok(Some(Made { res: r, state: D3D12_RESOURCE_STATE_GENERIC_READ, output: None }))
            }
            Res::Buffer { bytes, view, uav } => {
                let state = if *uav { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { srv_state };
                let r = self.filled_buffer(&buffer_bytes(seed, &k, *bytes as usize), *uav, state)?;
                let (format, stride, flags_raw, record) = match view {
                    BufView::Raw => (DXGI_FORMAT_R32_TYPELESS, 4, true, vec![Elem::Auto]),
                    BufView::Structured(s) => (DXGI_FORMAT_UNKNOWN, *s, false, vec![Elem::Auto]),
                    BufView::Typed(f) => (dxgi(*f), f.texel_size(), false, vec![f.elem(); 4]),
                };
                let n = bytes / stride;
                if let Some(cpu) = cpu {
                    let structured = if matches!(view, BufView::Structured(_)) { stride } else { 0 };
                    // SAFETY: views of a live buffer.
                    unsafe {
                        if *uav {
                            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                                Format: format,
                                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: n, StructureByteStride: structured, CounterOffsetInBytes: 0, Flags: if flags_raw { D3D12_BUFFER_UAV_FLAG_RAW } else { D3D12_BUFFER_UAV_FLAG_NONE } } },
                            };
                            self.device().CreateUnorderedAccessView(&r, None::<&ID3D12Resource>, Some(&d), cpu);
                        } else {
                            let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                                Format: format,
                                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: n, StructureByteStride: structured, Flags: if flags_raw { D3D12_BUFFER_SRV_FLAG_RAW } else { D3D12_BUFFER_SRV_FLAG_NONE } } },
                            };
                            self.device().CreateShaderResourceView(&r, Some(&d), cpu);
                        }
                    }
                }
                let output = uav.then(|| (format!("{k} buffer"), Readback::Buffer { bytes: *bytes, record }));
                Ok(Some(Made { res: r, state, output }))
            }
            Res::Texture { dim, size, mips, format, uav } => {
                let (r, desc) = self.texture(&k, *dim, *size, *mips, *format, *uav)?;
                if let Some(cpu) = cpu {
                    texture_view(self.device(), &r, *dim, *size, *mips, *format, *uav, cpu);
                }
                let state = if *uav { D3D12_RESOURCE_STATE_UNORDERED_ACCESS } else { srv_state };
                let output = uav.then(|| (format!("{k} texture {:?} {}", dim, format.name()), Readback::Texture { desc, format: *format, dim: *dim }));
                Ok(Some(Made { res: r, state, output }))
            }
        }
    }
}

fn subresources(desc: &D3D12_RESOURCE_DESC) -> u32 {
    let layers = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(desc.DepthOrArraySize) };
    u32::from(desc.MipLevels) * layers
}

fn footprints(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, subs: u32) -> (Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>, u64) {
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
    let mut total = 0u64;
    // SAFETY: the array has room for `subs` layouts.
    unsafe { device.GetCopyableFootprints(desc, 0, subs, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total)) };
    (layouts, total)
}

fn location_sub(res: &ID3D12Resource, sub: u32) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed copy of the interface pointer, never released (ManuallyDrop).
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
    }
}

fn location_placed(res: &ID3D12Resource, fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
    }
}

fn null_view(device: &ID3D12Device, class: Class, cpu: D3D12_CPU_DESCRIPTOR_HANDLE) {
    // SAFETY: null descriptors with a full description.
    unsafe {
        match class {
            Class::Uav => {
                let d = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: DXGI_FORMAT_R32_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
                device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&d), cpu);
            }
            Class::Cbv => device.CreateConstantBufferView(Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: 0, SizeInBytes: 0 }), cpu),
            _ => {
                let d = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
                };
                device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), cpu);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn texture_view(device: &ID3D12Device, r: &ID3D12Resource, dim: Dim, size: [u32; 3], mips: u32, format: Format, uav: bool, cpu: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let layers = size[2];
    // SAFETY: views of a live texture whose description matches.
    unsafe {
        if uav {
            let (vd, a) = match dim {
                Dim::D1 => (D3D12_UAV_DIMENSION_TEXTURE1D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_UAV { MipSlice: 0 } }),
                Dim::D1Array => (D3D12_UAV_DIMENSION_TEXTURE1DARRAY, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture1DArray: D3D12_TEX1D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: layers } }),
                Dim::D2 => (D3D12_UAV_DIMENSION_TEXTURE2D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 } }),
                Dim::D2Array | Dim::Cube | Dim::CubeArray => (D3D12_UAV_DIMENSION_TEXTURE2DARRAY, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: layers, PlaneSlice: 0 } }),
                Dim::D3 => (D3D12_UAV_DIMENSION_TEXTURE3D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: size[2] } }),
            };
            let d = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: dxgi(format), ViewDimension: vd, Anonymous: a };
            device.CreateUnorderedAccessView(r, None::<&ID3D12Resource>, Some(&d), cpu);
        } else {
            let (vd, a) = match dim {
                Dim::D1 => (D3D12_SRV_DIMENSION_TEXTURE1D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1D: D3D12_TEX1D_SRV { MostDetailedMip: 0, MipLevels: mips, ResourceMinLODClamp: 0.0 } }),
                Dim::D1Array => (D3D12_SRV_DIMENSION_TEXTURE1DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture1DArray: D3D12_TEX1D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: layers, ResourceMinLODClamp: 0.0 } }),
                Dim::D2 => (D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: mips, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } }),
                Dim::D2Array => (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, FirstArraySlice: 0, ArraySize: layers, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } }),
                Dim::D3 => (D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: 0, MipLevels: mips, ResourceMinLODClamp: 0.0 } }),
                Dim::Cube => (D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MostDetailedMip: 0, MipLevels: mips, ResourceMinLODClamp: 0.0 } }),
                Dim::CubeArray => (D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV { MostDetailedMip: 0, MipLevels: mips, First2DArrayFace: 0, NumCubes: layers / 6, ResourceMinLODClamp: 0.0 } }),
            };
            let d = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: dxgi(format), ViewDimension: vd, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: a };
            device.CreateShaderResourceView(r, Some(&d), cpu);
        }
    }
}

fn sampler_desc(i: u32, comparison: bool) -> D3D12_SAMPLER_DESC {
    let (filter, address) = match i % 3 {
        _ if comparison => (D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
        0 => (D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP),
        1 => (D3D12_FILTER_MIN_MAG_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
        _ => (D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT, D3D12_TEXTURE_ADDRESS_MODE_MIRROR),
    };
    D3D12_SAMPLER_DESC { Filter: filter, AddressU: address, AddressV: address, AddressW: address, MaxAnisotropy: 1, ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL, MaxLOD: f32::MAX, ..Default::default() }
}

/// The layout of one table in its heap: the start of every range.
fn table_layout(ranges: &[crate::plan::Range]) -> (Vec<u32>, u32) {
    let mut starts = Vec::new();
    let mut next = 0u32;
    let mut end = 0u32;
    for r in ranges {
        let start = r.offset.unwrap_or(next);
        let n = r.slots.len() as u32;
        starts.push(start);
        next = start + n;
        end = end.max(next);
    }
    (starts, end)
}

/// One stream output entry: semantic, index, first component, component count and type.
type SoEntry = (CString, u32, u8, u8, Component);

/// One side's stream output entries and the byte stride of a vertex.
fn stream_output(outputs: &[crate::model::Element]) -> (Vec<SoEntry>, u32) {
    let mut v = Vec::new();
    let mut stride = 0;
    for e in outputs {
        let (start, count) = e.span();
        if count == 0 {
            continue;
        }
        v.push((CString::new(e.semantic.clone()).unwrap_or_default(), e.index, start, count, e.component));
        stride += u32::from(count) * 4;
    }
    (v, stride)
}

fn elem_of(c: Component) -> Elem {
    match c {
        Component::Float => Elem::F32,
        Component::Uint => Elem::U32,
        Component::Sint => Elem::I32,
    }
}

/// Runs side `which` (0 is A) of `plan` with its container, and for a pixel pair the compiled
/// feeder vertex shader. Returns every writable resource, or why the side failed.
pub fn run_side(gpu: &Gpu, plan: &Plan, which: usize, container: &[u8], feeder: Option<&[u8]>) -> Result<Vec<Output>, String> {
    let _ = gpu.messages();
    run_side_inner(gpu, plan, which, container, feeder).map_err(|e| {
        // A call that failed because the device is gone: say why it went.
        let e = match gpu.removed() {
            Some(r) if !e.contains("device removed") => format!("{e} ({r})"),
            _ => e,
        };
        let m = gpu.messages();
        if m.is_empty() { e } else { format!("{e} [debug layer: {}]", m.join(" | ")) }
    })
}

fn run_side_inner(gpu: &Gpu, plan: &Plan, which: usize, container: &[u8], feeder: Option<&[u8]>) -> Result<Vec<Output>, String> {
    let side = &plan.sides[which];
    let device = &gpu.device;
    let blob = serialize(&plan.root)?;
    // SAFETY: standard D3D12 object creation and command recording. Every resource the list uses
    // is held in `keep`, `made` or a local until after `submit` has waited for the GPU.
    unsafe {
        let rs = device.CreateRootSignature::<ID3D12RootSignature>(0, &blob).map_err(|e| format!("CreateRootSignature: {e}"))?;
        let bytecode = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr().cast(), BytecodeLength: b.len() };

        // Pipeline.
        let mut so_entries = Vec::new();
        let mut so_stride = 0u32;
        let mut input_names: Vec<CString> = Vec::new();
        let pso = match plan.stage {
            Stage::Compute => {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())), CS: bytecode(container), ..Default::default() };
                let p = device.CreateComputePipelineState::<ID3D12PipelineState>(&desc);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                p
            }
            Stage::Pixel | Stage::Vertex => {
                let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() },
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let decl: Vec<D3D12_SO_DECLARATION_ENTRY>;
                let strides: [u32; 1];
                let elements: Vec<D3D12_INPUT_ELEMENT_DESC>;
                if plan.stage == Stage::Pixel {
                    desc.VS = bytecode(feeder.ok_or("a pixel pair needs its feeder vertex shader")?);
                    desc.PS = bytecode(container);
                    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
                    let slots = plan.targets.iter().map(|t| t.index + 1).max().unwrap_or(0).min(8);
                    desc.NumRenderTargets = slots;
                    for s in 0..slots {
                        desc.RTVFormats[s as usize] = plan.targets.iter().find(|t| t.index == s).map_or(DXGI_FORMAT_R8G8B8A8_UNORM, |t| dxgi(t.format));
                        desc.BlendState.RenderTarget[s as usize].RenderTargetWriteMask = 0xF;
                    }
                } else {
                    desc.VS = bytecode(container);
                    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
                    let (entries, stride) = stream_output(&side.reflection.outputs);
                    so_entries = entries;
                    so_stride = stride;
                    decl = so_entries.iter().map(|(name, index, start, count, _)| D3D12_SO_DECLARATION_ENTRY { Stream: 0, SemanticName: PCSTR(name.as_ptr().cast()), SemanticIndex: *index, StartComponent: *start, ComponentCount: *count, OutputSlot: 0 }).collect();
                    strides = [so_stride];
                    if !decl.is_empty() {
                        desc.StreamOutput = D3D12_STREAM_OUTPUT_DESC { pSODeclaration: decl.as_ptr(), NumEntries: decl.len() as u32, pBufferStrides: strides.as_ptr(), NumStrides: 1, RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM };
                    }
                    input_names = side.vertex_inputs.iter().map(|e| CString::new(e.semantic.clone()).unwrap_or_default()).collect();
                    elements = side
                        .vertex_inputs
                        .iter()
                        .zip(&input_names)
                        .enumerate()
                        .map(|(i, (e, n))| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(n.as_ptr().cast()),
                            SemanticIndex: e.index,
                            Format: match e.component {
                                Component::Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
                                Component::Uint => DXGI_FORMAT_R32G32B32A32_UINT,
                                Component::Sint => DXGI_FORMAT_R32G32B32A32_SINT,
                            },
                            InputSlot: 0,
                            AlignedByteOffset: 16 * i as u32,
                            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                            InstanceDataStepRate: 0,
                        })
                        .collect();
                    desc.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 };
                }
                let p = device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc);
                drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
                p
            }
        };
        let pso = pso.map_err(|e| format!("pipeline refused: {e}{}", gpu.removed().map(|r| format!(" ({r})")).unwrap_or_default()))?;

        let alloc = device.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list = device.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(|e| e.to_string())?;
        let mut ctx = Ctx { gpu, list: &list, plan, keep: Vec::new() };

        // Descriptor heaps: the slots shaders index directly first (so that index n is slot n),
        // then one region per table.
        let resource_heap = plan.root.flags & crate::plan::FLAG_RESOURCE_HEAP != 0;
        let sampler_heap = plan.root.flags & crate::plan::FLAG_SAMPLER_HEAP != 0;
        let reach = |sampler: bool| plan.heap.iter().filter(|s| (s.class == Class::Sampler) == sampler).map(|s| s.index + 1).max().unwrap_or(0);
        let mut resource_slots = if resource_heap { HEAP_EXTRA.max(reach(false)) } else { 0 };
        let mut sampler_slots = if sampler_heap { SAMPLER_HEAP_SIZE.max(reach(true)) } else { 0 };
        let mut table_base = Vec::new();
        for p in &plan.root.params {
            if let Param::Table(ranges) = p {
                let (_, size) = table_layout(ranges);
                if ranges.first().is_some_and(|r| r.class == Class::Sampler) {
                    table_base.push(sampler_slots);
                    sampler_slots += size;
                } else {
                    table_base.push(resource_slots);
                    resource_slots += size;
                }
            } else {
                table_base.push(0);
            }
        }
        let make_heap = |ty, n: u32| -> Result<Option<ID3D12DescriptorHeap>, String> {
            if n == 0 {
                return Ok(None);
            }
            let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: ty, NumDescriptors: n, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
            device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d).map(Some).map_err(|e| format!("descriptor heap: {e}"))
        };
        let res_heap = make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, resource_slots)?;
        let smp_heap = make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, sampler_slots)?;
        let res_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let smp_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
        let cpu_at = |h: &ID3D12DescriptorHeap, step: usize, i: u32| {
            let mut c = h.GetCPUDescriptorHandleForHeapStart();
            c.ptr += step * i as usize;
            c
        };
        let gpu_at = |h: &ID3D12DescriptorHeap, step: usize, i: u32| {
            let mut c = h.GetGPUDescriptorHandleForHeapStart();
            c.ptr += (step * i as usize) as u64;
            c
        };
        if let Some(h) = &res_heap {
            for i in 0..resource_slots {
                null_view(device, Class::Srv, cpu_at(h, res_step, i));
            }
        }
        if let Some(h) = &smp_heap {
            for i in 0..sampler_slots {
                let comparison = plan.heap.iter().any(|s| s.class == Class::Sampler && s.index == i && matches!(s.shape, crate::model::Shape::Sampler { comparison: true }));
                device.CreateSampler(&sampler_desc(i, comparison), cpu_at(h, smp_step, i));
            }
        }

        // Resources: everything but constant buffers first, so that the window's address is known.
        let mut made: BTreeMap<String, Made> = BTreeMap::new();
        for pass in [false, true] {
            for (pi, p) in plan.root.params.iter().enumerate() {
                match p {
                    Param::Table(ranges) => {
                        if ranges.first().is_some_and(|r| r.class == Class::Sampler) {
                            continue;
                        }
                        let (starts, _) = table_layout(ranges);
                        let h = res_heap.as_ref().ok_or("no heap")?;
                        for (r, start) in ranges.iter().zip(starts) {
                            if (r.class == Class::Cbv) != pass {
                                continue;
                            }
                            for (i, res) in r.slots.iter().enumerate() {
                                let reg = r.base + i as u32;
                                let cpu = cpu_at(h, res_step, table_base[pi] + start + i as u32);
                                if let Some(m) = ctx.make(r.class, r.space, reg, res, Some(cpu), &made)? {
                                    made.entry(key(r.class, r.space, reg)).or_insert(m);
                                }
                            }
                        }
                    }
                    Param::Root { class, register, space, res } if (*class == Class::Cbv) == pass => {
                        if let Some(m) = ctx.make(*class, *space, *register, res, None, &made)? {
                            made.insert(format!("root{pi}"), m);
                        }
                    }
                    _ => {}
                }
            }
            // The heap slots of known kind.
            for s in plan.heap.iter().filter(|s| s.class != Class::Sampler && (s.class == Class::Cbv) == pass) {
                let h = res_heap.as_ref().ok_or("no heap")?;
                if let Some(m) = ctx.make(s.class, HEAP_SPACE, s.index, &res_of(s.class, s.shape), Some(cpu_at(h, res_step, s.index)), &made)? {
                    made.entry(key(s.class, HEAP_SPACE, s.index)).or_insert(m);
                }
            }
        }

        // Bindings.
        // The heaps first: a root signature that indexes them directly needs them bound already.
        let heaps: Vec<Option<ID3D12DescriptorHeap>> = [res_heap.clone(), smp_heap.clone()].into_iter().flatten().map(Some).collect();
        if !heaps.is_empty() {
            list.SetDescriptorHeaps(&heaps);
        }
        let graphics = plan.stage != Stage::Compute;
        if graphics {
            list.SetGraphicsRootSignature(&rs);
        } else {
            list.SetComputeRootSignature(&rs);
        }
        for (pi, p) in plan.root.params.iter().enumerate() {
            let i = pi as u32;
            match p {
                Param::Table(ranges) => {
                    let sampler = ranges.first().is_some_and(|r| r.class == Class::Sampler);
                    let handle = if sampler { gpu_at(smp_heap.as_ref().ok_or("no sampler heap")?, smp_step, table_base[pi]) } else { gpu_at(res_heap.as_ref().ok_or("no heap")?, res_step, table_base[pi]) };
                    if graphics {
                        list.SetGraphicsRootDescriptorTable(i, handle);
                    } else {
                        list.SetComputeRootDescriptorTable(i, handle);
                    }
                }
                Param::Constants { register, space, count } => {
                    let data = buffer_bytes(plan.options.seed, &format!("c{register} space{space}"), *count as usize * 4);
                    if graphics {
                        list.SetGraphicsRoot32BitConstants(i, *count, data.as_ptr().cast(), 0);
                    } else {
                        list.SetComputeRoot32BitConstants(i, *count, data.as_ptr().cast(), 0);
                    }
                }
                Param::Root { class, .. } => {
                    let va = made.get(&format!("root{pi}")).map_or(0, |m| m.res.GetGPUVirtualAddress());
                    match (class, graphics) {
                        (Class::Cbv, true) => list.SetGraphicsRootConstantBufferView(i, va),
                        (Class::Cbv, false) => list.SetComputeRootConstantBufferView(i, va),
                        (Class::Srv, true) => list.SetGraphicsRootShaderResourceView(i, va),
                        (Class::Srv, false) => list.SetComputeRootShaderResourceView(i, va),
                        (_, true) => list.SetGraphicsRootUnorderedAccessView(i, va),
                        (_, false) => list.SetComputeRootUnorderedAccessView(i, va),
                    }
                }
            }
        }

        // Draw or dispatch.
        let mut targets: Vec<(ID3D12Resource, D3D12_RESOURCE_DESC, Format, u32)> = Vec::new();
        let mut so_buffer: Option<(ID3D12Resource, u32)> = None;
        let mut rtv_keep: Option<ID3D12DescriptorHeap> = None;
        match plan.stage {
            Stage::Compute => {
                let g = plan.options.groups;
                list.Dispatch(g[0], g[1], g[2]);
            }
            Stage::Pixel => {
                let size = plan.options.size;
                let slots = plan.targets.iter().map(|t| t.index + 1).max().unwrap_or(0).min(8);
                let rtv_heap = device
                    .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: slots.max(1), Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 })
                    .map_err(|e| e.to_string())?;
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut handles = Vec::new();
                for s in 0..slots {
                    let format = plan.targets.iter().find(|t| t.index == s).map_or(Format::Rgba8, |t| t.format);
                    let desc = D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(size),
                        Height: size,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: dxgi(format),
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    };
                    let clear = D3D12_CLEAR_VALUE { Format: dxgi(format), Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] } };
                    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                    let mut res: Option<ID3D12Resource> = None;
                    device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&clear), &mut res).map_err(|e| format!("render target: {e}"))?;
                    let res = res.ok_or("no render target")?;
                    let mut h = rtv_heap.GetCPUDescriptorHandleForHeapStart();
                    h.ptr += rtv_step * s as usize;
                    device.CreateRenderTargetView(&res, None, h);
                    list.ClearRenderTargetView(h, &[0.0; 4], None);
                    handles.push(h);
                    targets.push((res, desc, format, s));
                }
                list.OMSetRenderTargets(slots, if handles.is_empty() { None } else { Some(handles.as_ptr()) }, false, None);
                list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: size as f32, Height: size as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
                list.RSSetScissorRects(&[RECT { left: 0, top: 0, right: size as i32, bottom: size as i32 }]);
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                let g = plan.options.grid;
                list.DrawInstanced(g * g * 6, 1, 0, 0);
                rtv_keep = Some(rtv_heap);
            }
            Stage::Vertex => {
                let n = plan.options.points;
                let inputs = &side.vertex_inputs;
                if !inputs.is_empty() {
                    let mut data = Vec::with_capacity(n as usize * inputs.len() * 16);
                    for v in 0..n {
                        for e in inputs {
                            for c in 0..4 {
                                data.extend_from_slice(&vertex_value(e, plan.options.feed, plan.options.seed, v, c).to_le_bytes());
                            }
                        }
                    }
                    let vb = upload(device, &data)?;
                    let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: vb.GetGPUVirtualAddress(), SizeInBytes: data.len() as u32, StrideInBytes: 16 * inputs.len() as u32 };
                    list.IASetVertexBuffers(0, Some(&[view]));
                    ctx.keep.push(vb);
                }
                if so_stride > 0 {
                    let bytes = so_stride * n;
                    let counter = bytes.next_multiple_of(16);
                    let so = buffer(device, D3D12_HEAP_TYPE_DEFAULT, counter as usize + 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_STREAM_OUT)?;
                    let va = so.GetGPUVirtualAddress();
                    list.SOSetTargets(0, Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW { BufferLocation: va, SizeInBytes: u64::from(bytes), BufferFilledSizeLocation: va + u64::from(counter) }]));
                    so_buffer = Some((so, bytes));
                }
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                list.DrawInstanced(n, 1, 0, 0);
            }
        }

        // Read back.
        let mut pending: Vec<(String, ID3D12Resource, Readback)> = Vec::new();
        for (name, m) in made.values().filter_map(|m| m.output.as_ref().map(|o| (o, m))) {
            list.ResourceBarrier(&[transition(&m.res, m.state, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            pending.push((name.0.clone(), m.res.clone(), name.1.clone()));
        }
        for (res, desc, format, slot) in &targets {
            if !plan.targets.iter().any(|t| t.index == *slot) {
                continue;
            }
            list.ResourceBarrier(&[transition(res, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            pending.push((format!("SV_Target{slot} {}", format.name()), res.clone(), Readback::Texture { desc: *desc, format: *format, dim: Dim::D2 }));
        }
        if let Some((so, bytes)) = &so_buffer {
            list.ResourceBarrier(&[transition(so, D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            pending.push(("stream output".into(), so.clone(), Readback::Buffer { bytes: *bytes, record: Vec::new() }));
        }
        let mut readbacks = Vec::new();
        for (_, res, rb) in &pending {
            match rb {
                Readback::Buffer { bytes, .. } => {
                    let r = buffer(device, D3D12_HEAP_TYPE_READBACK, *bytes as usize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    list.CopyBufferRegion(&r, 0, res, 0, u64::from(*bytes));
                    readbacks.push((r, Vec::new()));
                }
                Readback::Texture { desc, .. } => {
                    let subs = subresources(desc);
                    let (layouts, total) = footprints(device, desc, subs);
                    let r = buffer(device, D3D12_HEAP_TYPE_READBACK, total as usize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                    for (i, l) in layouts.iter().enumerate() {
                        list.CopyTextureRegion(&location_placed(&r, *l), 0, 0, 0, &location_sub(res, i as u32), None);
                    }
                    readbacks.push((r, layouts));
                }
            }
        }
        gpu.submit(&list, DEADLINE)?;

        let mut outputs = Vec::new();
        for ((name, _, rb), (r, layouts)) in pending.iter().zip(&readbacks) {
            match rb {
                Readback::Buffer { bytes, record } if name == "stream output" => {
                    let data = read_back(r, *bytes as usize)?;
                    outputs.extend(split_stream_output(&data, &so_entries, so_stride));
                    let _ = record;
                }
                Readback::Buffer { bytes, record } => {
                    let data = read_back(r, *bytes as usize)?;
                    let stride: usize = record.iter().map(|e| e.size()).sum();
                    outputs.push(Output { name: name.clone(), texture: false, subs: vec![Sub { label: String::new(), width: (data.len() / stride.max(1)) as u32, height: 1, depth: 1, record: record.clone(), bytes: data }] });
                }
                Readback::Texture { desc, format, dim } => {
                    let total = layouts.last().map_or(0, |l| l.Offset as usize + l.Footprint.RowPitch as usize * (l.Footprint.Height * l.Footprint.Depth) as usize);
                    let data = read_back(r, total)?;
                    let mips = u32::from(desc.MipLevels);
                    let texel_size = format.texel_size() as usize;
                    let mut subs = Vec::new();
                    for (i, l) in layouts.iter().enumerate() {
                        let fp = l.Footprint;
                        let mut bytes = Vec::with_capacity((fp.Width * fp.Height * fp.Depth) as usize * texel_size);
                        for z in 0..fp.Depth {
                            for y in 0..fp.Height {
                                let at = l.Offset as usize + (z * fp.Height + y) as usize * fp.RowPitch as usize;
                                bytes.extend_from_slice(&data[at..at + fp.Width as usize * texel_size]);
                            }
                        }
                        let (mip, slice) = (i as u32 % mips, i as u32 / mips);
                        let label = if *dim == Dim::D3 { format!("mip {mip}") } else { format!("mip {mip} slice {slice}") };
                        subs.push(Sub { label, width: fp.Width, height: fp.Height, depth: fp.Depth, record: vec![format.elem(); 4], bytes });
                    }
                    outputs.push(Output { name: name.clone(), texture: true, subs });
                }
            }
        }
        drop(ctx);
        drop(made);
        drop(input_names);
        drop(rtv_keep);
        // A device lost while the side ran makes its outputs worthless, even when the copies came
        // back. Some losses only show on the next submission, so one empty submission follows.
        let probe = device.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| format!("after the run: {e}"))?;
        gpu.submit(&probe, DEADLINE).map_err(|e| format!("after the run: {e}"))?;
        Ok(outputs)
    }
}

/// One output per stream output element, named by its semantic and index, one record per vertex.
fn split_stream_output(data: &[u8], entries: &[SoEntry], stride: u32) -> Vec<Output> {
    let stride = stride as usize;
    let vertices = data.len().checked_div(stride).unwrap_or(0);
    let mut out = Vec::new();
    let mut offset = 0usize;
    for (name, index, start, count, component) in entries {
        let n = *count as usize * 4;
        let mut bytes = Vec::with_capacity(vertices * n);
        for v in 0..vertices {
            bytes.extend_from_slice(&data[v * stride + offset..v * stride + offset + n]);
        }
        let semantic = name.to_string_lossy().to_ascii_uppercase();
        out.push(Output {
            name: format!("{semantic}{index} components {start}..{}", start + count),
            texture: false,
            subs: vec![Sub { label: String::new(), width: vertices as u32, height: 1, depth: 1, record: vec![elem_of(*component); *count as usize], bytes }],
        });
        offset += n;
    }
    out
}
