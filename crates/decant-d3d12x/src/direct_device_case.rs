//! The device case of the direct shader path (issue #172), a part of `d3d12xs_test`:
//! `d3d12xs_test <d3d12_xs.dll> direct`, run with `DECANT_D3D12X_DIRECT=1`.
//!
//! It hands the layer what the title does, through the console slots of the device and list
//! proxies: a root signature with one descriptor table, a graphics pipeline whose two shaders are
//! console containers (the synthetic user interface pair of `direct_fixture`), a constant buffer
//! view in a table of a shader visible heap, vertices in upload memory of the list, and a draw.
//! The layer translates the pair with `decant-shader`, compiles the HLSL with DXC, makes the
//! pipeline once compilation is ready and writes the user data mirror before the draw. The vertex shader
//! finds its constant buffer in the heap at the index the mirror holds, so the colour the centre
//! of the target reads back proves the whole path: adapter, DXC, cache, pipeline, mirror, draw.
//!
//! Four pipelines are drawn:
//! 1. the pair, new: the draw retries while compilation leaves the target clear,
//! 2. the same description again, with cached DXIL,
//! 3. a pair whose pixel stage does not link (the translator refuses it): the pipeline takes the
//!    title's shaders, which the PC runtime cannot run without an override, and nothing is drawn.
//! 4. a mesh pipeline that draws a white triangle.
//!
//! Skipped (not failed) without a D3D12 device or without DXC.

use std::ffi::{OsStr, c_void};
use std::mem::ManuallyDrop;
use std::time::{Duration, Instant};

use decant_testkit::Library;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::core::{GUID, Interface, PCSTR};

use crate::direct_fixture::{container, ui_pixel, ui_vertex};

/// The function in slot `slot` of the vtable of `object`.
///
/// # Safety
/// `object` is a live COM object, and `F` is the function pointer type of that slot.
unsafe fn vt<F: Copy>(object: *mut c_void, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: by the contract above.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(slot))
    }
}

const SIZE: u32 = 64;
/// The descriptor of the view heap the table starts at. Not 0, so a mirror that is not written
/// (all zero) reads another descriptor.
const TABLE_AT: u32 = 5;
const CLEAR: [u8; 4] = [0, 0, 0, 0];

/// The constant buffer: rows of four floats. The vertices pick row 1 (`COLOR1.z` is 1 / 255).
const ROWS: [[f32; 4]; 5] = [
    [9.0, 9.0, 9.0, 9.0],
    // Row k: TEXCOORD0.
    [0.2, 0.13, 0.0, 0.3],
    // Row k + 1: TEXCOORD1.
    [0.5, 0.25, 1.0, 1.0],
    // Rows k + 2 and k + 3: x and y of the position from POSITION.xy.
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
];
const COLOR0: [f32; 4] = [0.4, 0.8, 0.6, 0.5];
const COLOR1: [f32; 4] = [0.0, 0.0, 1.0 / 255.0, 0.5];
/// One triangle over the whole target.
const POSITIONS: [[f32; 2]; 3] = [[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]];
const VERTEX_BYTES: u32 = 40;

/// What the pixel program makes of the data above: `saturate(TEXCOORD0 + COLOR0 * TEXCOORD1)`,
/// its alpha times the alpha of `COLOR1`, as 8 bit values.
fn expected() -> [u8; 4] {
    let (t0, t1) = (ROWS[1], ROWS[2]);
    let mut out = [0u8; 4];
    for c in 0..4 {
        let mut v = (t0[c] + COLOR0[c] * t1[c]).clamp(0.0, 1.0);
        if c == 3 {
            v *= COLOR1[3];
        }
        out[c] = (v * 255.0 + 0.5) as u8;
    }
    out
}

fn close(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= 1)
}

struct Checks {
    failures: u32,
}

impl Checks {
    fn line(&mut self, ok: bool, text: &str) {
        println!("{} {text}", if ok { "ok    " } else { "FAIL  " });
        self.failures += u32::from(!ok);
    }
}

/// The objects of the case. The proxies are raw console objects, the rest are PC objects.
struct Scene {
    dev: *mut c_void,
    queue: *mut c_void,
    alloc: *mut c_void,
    list: *mut c_void,
    fence: ID3D12Fence,
    fence_proxy: *mut c_void,
    fence_value: u64,
    root: *mut c_void,
    heap: *mut c_void,
    heap_cpu: u64,
    heap_gpu: u64,
    increment: u32,
    target: ID3D12Resource,
    rtv: usize,
    _rtv_heap: ID3D12DescriptorHeap,
    readback: ID3D12Resource,
    /// Whether the list is open and empty (after its creation or a reset).
    fresh: bool,
}

type CreateRs = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> i32;
type CreatePso = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type SerializeVersioned = unsafe extern "system" fn(*const D3D12_VERSIONED_ROOT_SIGNATURE_DESC, *mut *mut c_void, *mut *mut c_void) -> i32;

fn transition(resource: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed interface pointer in a field that is never dropped.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

impl Scene {
    /// The root signature of the case as the title serializes the one of its user interface
    /// pipeline: one table for all stages with the ranges `t2` and `b0` of space 1, each appended
    /// to the one before. The PC has `b0` at descriptor 1 of the table. The console vertex program
    /// reads its constant buffer at descriptor 0, where this case (like the title) puts the view.
    fn root_blob(serialize: SerializeVersioned) -> Option<Vec<u8>> {
        let range = |kind, register| D3D12_DESCRIPTOR_RANGE1 {
            RangeType: kind,
            NumDescriptors: 1,
            BaseShaderRegister: register,
            RegisterSpace: 1,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };
        let ranges = [range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2), range(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 0)];
        let param = D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 2, pDescriptorRanges: ranges.as_ptr() } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: &param,
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                },
            },
        };
        let (mut blob, mut error): (*mut c_void, *mut c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: the description and what it points to live through the call, the outputs are writable.
        let hr = unsafe { serialize(&desc, &mut blob, &mut error) };
        if !error.is_null() {
            // SAFETY: a blob this program owns.
            drop(unsafe { windows::Win32::Graphics::Direct3D::ID3DBlob::from_raw(error) });
        }
        if hr < 0 || blob.is_null() {
            return None;
        }
        // SAFETY: a blob this program owns, with `GetBufferSize` bytes.
        unsafe {
            let blob = windows::Win32::Graphics::Direct3D::ID3DBlob::from_raw(blob);
            Some(std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec())
        }
    }

    /// A graphics pipeline with the console containers `vs` and `ps`, through device slot 10.
    fn pipeline(&self, vs: &[u8], ps: &[u8]) -> Option<*mut c_void> {
        let element = |name: &'static std::ffi::CStr, index: u32, format: DXGI_FORMAT, offset: u32| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(name.as_ptr().cast()),
            SemanticIndex: index,
            Format: format,
            InputSlot: 0,
            AlignedByteOffset: offset,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        };
        let elements = [
            element(c"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0),
            element(c"COLOR", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 16),
            element(c"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 32),
        ];
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        // SAFETY: the description is plain data, zero is a valid start for every field.
        let mut desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::mem::zeroed() };
        // SAFETY: a borrowed interface pointer in a field that is never dropped.
        desc.pRootSignature = unsafe { std::mem::transmute::<*mut c_void, ManuallyDrop<Option<ID3D12RootSignature>>>(self.root) };
        desc.VS = code(vs);
        desc.PS = code(ps);
        desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
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
        desc.SampleMask = u32::MAX;
        desc.RasterizerState = D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() };
        desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC { DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS, FrontFace: stencil, BackFace: stencil, ..Default::default() };
        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 };
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
        let mut pso: *mut c_void = std::ptr::null_mut();
        // SAFETY: device slot 10 takes the PC description, which lives through the call (the layer
        // copies it with its shaders and its input layout).
        let hr = unsafe { vt::<CreatePso>(self.dev, 10)(self.dev, (&raw const desc).cast(), &ID3D12PipelineState::IID, &mut pso) };
        (hr >= 0 && !pso.is_null()).then_some(pso)
    }

    /// A console stream with mesh and pixel shaders and a single color target.
    fn mesh_pipeline(&self, ms: &[u8], ps: &[u8]) -> Option<*mut c_void> {
        // The console stream subobjects, padded to eight bytes. The render target array has
        // eight 32-bit formats followed by its count. The depth stencil state is disabled.
        let stream: [u64; 20] = [
            0, self.root as u64,
            24, ms.as_ptr() as u64, ms.len() as u64,
            2, ps.as_ptr() as u64, ps.len() as u64,
            15 | (28u64 << 32), 0, 0, 0, 1u64 << 32,
            11, 0, 0, 0, 0, 0, 0,
        ];
        let desc = [size_of_val(&stream), stream.as_ptr() as usize];
        let mut pso = std::ptr::null_mut();
        // SAFETY: slot 86 takes the stream description and copies its live subobjects.
        let hr = unsafe { vt::<CreatePso>(self.dev, 86)(self.dev, desc.as_ptr().cast(), &ID3D12PipelineState::IID, &mut pso) };
        (hr >= 0 && !pso.is_null()).then_some(pso)
    }

    fn draw_when_ready(&mut self, pso: *mut c_void, mesh: bool) -> Option<[u8; 4]> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            // A pending bind skips the draw and leaves the cleared target intact.
            let pixel = self.frame(pso, mesh, deadline);
            let remaining = deadline.saturating_duration_since(Instant::now());
            if pixel != Some(CLEAR) || remaining.is_zero() {
                return pixel;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(10)));
            if Instant::now() >= deadline {
                return pixel;
            }
        }
    }

    /// One frame with `pso`: clear, draw the triangle, copy the target to the read back buffer, run
    /// the list and wait. The centre pixel of the target, `None` when a call failed.
    fn frame(&mut self, pso: *mut c_void, mesh: bool, deadline: Instant) -> Option<[u8; 4]> {
        // SAFETY: every object is live, each slot is called with the argument shapes of the console
        // order (`abi::LIST_RULES`), and what the calls point to lives until the list is closed.
        unsafe {
            if !self.fresh {
                if vt::<unsafe extern "system" fn(*mut c_void) -> i32>(self.alloc, 8)(self.alloc) < 0 {
                    return None;
                }
                if vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32>(self.list, 10)(self.list, self.alloc, std::ptr::null_mut()) < 0 {
                    return None;
                }
            }
            self.fresh = false;
            let list = self.list;
            // Upload memory of the list (slot 148), which the proxy takes as a GPU address: the
            // constant buffer on 256 bytes and the three vertices.
            type Alloc = unsafe extern "system" fn(*mut c_void, usize, usize) -> *mut u8;
            let constants = vt::<Alloc>(list, 148)(list, 256, 256);
            let vertices = vt::<Alloc>(list, 148)(list, 3 * VERTEX_BYTES as usize, 16);
            if constants.is_null() || vertices.is_null() {
                return None;
            }
            constants.write_bytes(0, 256);
            std::ptr::copy_nonoverlapping(ROWS.as_ptr().cast::<u8>(), constants, size_of_val(&ROWS));
            for (i, position) in POSITIONS.iter().enumerate() {
                let at = vertices.add(i * VERTEX_BYTES as usize).cast::<f32>();
                std::ptr::copy_nonoverlapping(COLOR0.as_ptr(), at, 4);
                std::ptr::copy_nonoverlapping(COLOR1.as_ptr(), at.add(4), 4);
                std::ptr::copy_nonoverlapping(position.as_ptr(), at.add(8), 2);
            }
            // The constant buffer view at descriptor 0 of the table (device slot 17).
            let view: [u64; 2] = [constants as u64, 256];
            let cpu = self.heap_cpu + u64::from(TABLE_AT) * u64::from(self.increment);
            vt::<unsafe extern "system" fn(*mut c_void, *const c_void, usize)>(self.dev, 17)(self.dev, view.as_ptr().cast(), cpu as usize);

            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(list, 32)(list, 1, &self.heap);
            vt::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(list, 34)(list, self.root);
            let table = self.heap_gpu + u64::from(TABLE_AT) * u64::from(self.increment);
            vt::<unsafe extern "system" fn(*mut c_void, u32, u64)>(list, 36)(list, 0, table);
            vt::<unsafe extern "system" fn(*mut c_void, *mut c_void)>(list, 25)(list, pso);
            let viewport = D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: SIZE as f32, Height: SIZE as f32, MinDepth: 0.0, MaxDepth: 1.0 };
            let scissor = windows::Win32::Foundation::RECT { left: 0, top: 0, right: SIZE as i32, bottom: SIZE as i32 };
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_VIEWPORT)>(list, 21)(list, 1, &viewport);
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const windows::Win32::Foundation::RECT)>(list, 22)(list, 1, &scissor);
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const usize, i32, *const usize)>(list, 50)(list, 1, &self.rtv, 0, std::ptr::null());
            let clear = [0.0f32; 4];
            vt::<unsafe extern "system" fn(*mut c_void, usize, *const f32, u32, *const c_void)>(list, 52)(list, self.rtv, clear.as_ptr(), 0, std::ptr::null());
            if mesh {
                vt::<unsafe extern "system" fn(*mut c_void, u32, u32, u32)>(list, 127)(list, 1, 1, 1);
            } else {
                vt::<unsafe extern "system" fn(*mut c_void, i32)>(list, 20)(list, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0);
                let vb: [u64; 2] = [vertices as u64, u64::from(3 * VERTEX_BYTES) | u64::from(VERTEX_BYTES) << 32];
                vt::<unsafe extern "system" fn(*mut c_void, u32, u32, *const [u64; 2])>(list, 48)(list, 0, 1, &vb);
                vt::<unsafe extern "system" fn(*mut c_void, u32, u32, u32, u32)>(list, 12)(list, 3, 1, 0, 0);
            }

            // The target to the read back buffer.
            let to_source = transition(&self.target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, 26)(list, 1, &to_source);
            let source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R8G8B8A8_UNORM, Width: SIZE, Height: SIZE, Depth: 1, RowPitch: SIZE * 4 },
            };
            let destination = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            type CopyRegion = unsafe extern "system" fn(*mut c_void, *const D3D12_TEXTURE_COPY_LOCATION, u32, u32, u32, *const D3D12_TEXTURE_COPY_LOCATION, *const D3D12_BOX);
            vt::<CopyRegion>(list, 16)(list, &destination, 0, 0, 0, &source, std::ptr::null());
            let to_target = transition(&self.target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(list, 26)(list, 1, &to_target);
            if vt::<unsafe extern "system" fn(*mut c_void) -> i32>(list, 9)(list) < 0 {
                return None;
            }
            vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(self.queue, 10)(self.queue, 1, &list);
            self.fence_value += 1;
            if vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32>(self.queue, 14)(self.queue, self.fence_proxy, self.fence_value) < 0 {
                return None;
            }
            let fence_deadline = deadline.min(Instant::now() + Duration::from_secs(20));
            while self.fence.GetCompletedValue() < self.fence_value {
                if Instant::now() >= fence_deadline {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let mut data: *mut c_void = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut data)).ok()?;
            let centre = (SIZE / 2 * SIZE * 4 + SIZE / 2 * 4) as usize;
            let pixel = std::ptr::read_unaligned(data.cast::<u8>().add(centre).cast::<[u8; 4]>());
            self.readback.Unmap(0, None);
            Some(pixel)
        }
    }
}

/// Builds the scene: the console device and its proxies, the PC target and read back buffer.
fn scene(xs: &Library, pc: &Library) -> Result<Scene, String> {
    type XboxCreate = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
    // SAFETY: each type is the signature of the export it names.
    let (create, serialize) = unsafe { (xs.get::<XboxCreate>("D3D12XboxCreateDevice"), pc.get::<SerializeVersioned>("D3D12SerializeVersionedRootSignature")) };
    let (Some(create), Some(serialize)) = (create, serialize) else { return Err("FAIL  missing export".into()) };
    let mut dev: *mut c_void = std::ptr::null_mut();
    // SAFETY: the export takes (adapter, parameters, riid, device).
    let hr = unsafe { create(std::ptr::null_mut(), std::ptr::null(), &ID3D12Device::IID, &mut dev) };
    if hr < 0 || dev.is_null() {
        return Err(format!("skip  no D3D12 12_0 device on this machine (hr=0x{:08x}): the direct shader case is skipped", hr as u32));
    }
    let fail = |what: &str, hr: i32| format!("FAIL  {what} (hr=0x{:08x})", hr as u32);
    // SAFETY: `dev` is the console device proxy, every slot is called with the argument shapes of
    // the console order (`abi::DEVICE_RULES`), and the PC device behind it (+0x08) is live.
    unsafe {
        let pc_dev_raw = *(dev as *const *mut c_void).add(1);
        let pc_dev = ID3D12Device::from_raw_borrowed(&pc_dev_raw).ok_or("FAIL  no PC device behind the proxy")?;

        let blob = Scene::root_blob(serialize).ok_or("FAIL  the root signature does not serialize")?;
        let mut root: *mut c_void = std::ptr::null_mut();
        let hr = vt::<CreateRs>(dev, 16)(dev, 0, blob.as_ptr().cast(), blob.len(), &ID3D12RootSignature::IID, &mut root);
        if hr < 0 || root.is_null() {
            return Err(fail("device slot 16 makes the root signature with the mirror and the heap flags", hr));
        }

        let queue_desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, Priority: 0, Flags: D3D12_COMMAND_QUEUE_FLAG_NONE, NodeMask: 0 };
        let (mut queue, mut alloc, mut list, mut fence_proxy, mut heap): (*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void) =
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        let hr = vt::<unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const GUID, *mut *mut c_void) -> i32>(dev, 8)(dev, &queue_desc, &ID3D12CommandQueue::IID, &mut queue);
        if hr < 0 || queue.is_null() {
            return Err(fail("device slot 8 makes a queue", hr));
        }
        let hr = vt::<unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32>(dev, 9)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT.0, &ID3D12CommandAllocator::IID, &mut alloc);
        if hr < 0 || alloc.is_null() {
            return Err(fail("device slot 9 makes an allocator", hr));
        }
        type CreateList = unsafe extern "system" fn(*mut c_void, u32, i32, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
        let hr = vt::<CreateList>(dev, 12)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0, alloc, std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list);
        if hr < 0 || list.is_null() {
            return Err(fail("device slot 12 makes a list", hr));
        }
        let hr = vt::<unsafe extern "system" fn(*mut c_void, u64, i32, *const GUID, *mut *mut c_void) -> i32>(dev, 36)(dev, 0, D3D12_FENCE_FLAG_NONE.0, &ID3D12Fence::IID, &mut fence_proxy);
        if hr < 0 || fence_proxy.is_null() {
            return Err(fail("device slot 36 makes a fence", hr));
        }
        let fence = ID3D12Fence::from_raw(fence_proxy);

        // The view heap of the tables, through the proxy: the mirror turns a table handle into an
        // index of this heap.
        let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 16, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let hr = vt::<unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32>(dev, 14)(dev, &heap_desc, &ID3D12DescriptorHeap::IID, &mut heap);
        if hr < 0 || heap.is_null() {
            return Err(fail("device slot 14 makes a view heap", hr));
        }
        let heap_cpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 9)(heap);
        let heap_gpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 10)(heap);
        let increment = vt::<unsafe extern "system" fn(*mut c_void, i32) -> u32>(dev, 15)(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV.0);

        // The target, its view and the read back buffer are plain PC objects.
        let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
        let texture = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIZE),
            Height: SIZE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: sample,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(SIZE * SIZE * 4),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: sample,
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let default_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let readback_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
        let (mut target, mut readback): (Option<ID3D12Resource>, Option<ID3D12Resource>) = (None, None);
        pc_dev.CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut target).map_err(|e| fail("the target", e.code().0))?;
        pc_dev.CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).map_err(|e| fail("the read back buffer", e.code().0))?;
        let (target, readback) = (target.ok_or("FAIL  no target")?, readback.ok_or("FAIL  no read back buffer")?);
        let rtv_desc = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() };
        let rtv_heap: ID3D12DescriptorHeap = pc_dev.CreateDescriptorHeap(&rtv_desc).map_err(|e| fail("the target view heap", e.code().0))?;
        let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart().ptr;
        pc_dev.CreateRenderTargetView(&target, None, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: rtv });

        Ok(Scene { dev, queue, alloc, list, fence, fence_proxy, fence_value: 0, root, heap, heap_cpu, heap_gpu, increment, target, rtv, _rtv_heap: rtv_heap, readback, fresh: true })
    }
}

/// Runs the case. 0 when every check passed or the case was skipped, 1 on a failure.
pub fn main(xs_path: &OsStr) -> u8 {
    if let Err(why) = decant_dxc::Dxc::load() {
        println!("skip  no DXC on this machine ({why}): the direct shader case is skipped");
        return 0;
    }
    let (Ok(xs), Ok(pc)) = (Library::open(xs_path), Library::open(OsStr::new("d3d12.dll"))) else {
        println!("FAIL  cannot load {} or d3d12.dll", xs_path.to_string_lossy());
        return 1;
    };
    let mut scene = match scene(&xs, &pc) {
        Ok(s) => s,
        Err(line) => {
            println!("{line}");
            return u8::from(line.starts_with("FAIL"));
        }
    };
    let mut t = Checks { failures: 0 };
    let want = expected();
    let (vs, ps) = (container(&ui_vertex()), container(&ui_pixel()));

    // 1. The pair, new to the layer.
    match scene.pipeline(&vs, &ps) {
        None => t.line(false, "device slot 10 takes a graphics pipeline with console shaders"),
        Some(pso) => {
            let pixel = scene.draw_when_ready(pso, false);
            t.line(
                pixel.is_some_and(|p| close(p, want)),
                &format!("the translated pair draws {want:?} through the mirror and the heap within 30 seconds: the centre is {pixel:?}"),
            );
            // 2. The same description as a second pipeline: every stage is cached.
            match scene.pipeline(&vs, &ps) {
                None => t.line(false, "device slot 10 takes the same pipeline again"),
                Some(again) => {
                    let pixel = scene.draw_when_ready(again, false);
                    t.line(pixel.is_some_and(|p| close(p, want)), &format!("a pipeline with cached DXIL draws within 30 seconds: the centre is {pixel:?}"));
                }
            }
        }
    }

    // 3. A pixel stage that does not link to the vertex stage: the translator refuses the pair.
    let mut unlinked = ui_pixel();
    unlinked.inputs[3].semantic = "NORMAL".to_owned();
    match scene.pipeline(&vs, &container(&unlinked)) {
        None => t.line(false, "device slot 10 takes a pipeline the translator will refuse"),
        Some(pso) => {
            let mut pixel = None;
            for _ in 0..3 {
                pixel = scene.frame(pso, false, Instant::now() + Duration::from_secs(20));
            }
            t.line(pixel == Some(CLEAR), &format!("a refused pair takes the title's shaders, which draw nothing here: the centre is {pixel:?}"));
            // The pipelines of the cases before still draw.
        }
    }

    // 4. A mesh pipeline reaches the PC runtime through the stream and emits one triangle.
    let ms = container(&crate::direct_fixture::mesh_triangle());
    match scene.mesh_pipeline(&ms, &ps) {
        None => t.line(false, "device slot 86 takes a mesh pipeline with console shaders"),
        Some(pso) => {
            let pixel = scene.draw_when_ready(pso, true);
            t.line(pixel == Some([255; 4]), &format!("a direct mesh pipeline draws white within 30 seconds: the centre is {pixel:?}"));
        }
    }

    if t.failures != 0 {
        println!("d3d12xs_test direct: {} failure(s)", t.failures);
    } else {
        println!("d3d12xs_test direct: all checks passed");
    }
    u8::from(t.failures != 0)
}
