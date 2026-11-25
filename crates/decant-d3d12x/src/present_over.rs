//! Second presentation plane compositing for `PresentX`.
//!
//! Ported from vitutu1/victory-perf (GPL-3.0-or-later) and adapted to this codebase.
//!
//! On the console the display controller composes multiple `PresentX` planes (for example a
//! lower-resolution scene on plane 0 and a 4K UI / loading overlay on plane 1 whose empty
//! pixels have alpha 0). Showing plane 0 alone leaves any content drawn only onto plane 1
//! invisible. When `PresentX` passes 2 or more planes with a distinct second resource, this pass
//! draws plane 1 over the back buffer after plane 0 has been copied into it: one full-screen
//! triangle sampled bilinearly across the whole back buffer with premultiplied alpha blending
//! (`ONE`, `INV_SRC_ALPHA`).
//!
//! The root signature is serialized explicitly via `D3D12SerializeRootSignature` rather than
//! embedded with `[RootSignature]` in HLSL: under Wine, `d3dcompiler_47` omits embedded root
//! signatures so `CreateRootSignature` on the shader blob fails.
//!
//! Switch: `DECANT_D3D12X_PRESENT_OVER=0` disables second-plane compositing (default on; 0 = plane 0
//! only, matching prior behaviour).

/// Byte stride between consecutive `D3D12XBOX_PRESENT_PLANE_PARAMETERS` entries passed to
/// `PresentX`, confirmed by measurement against the title's calls.
pub const PLANE_STRIDE: usize = 0x48;

/// Byte offset of `ResourceCount` (`u32`) inside one plane parameter struct.
pub const PLANE_COUNT_OFFSET: usize = 0x08;

/// Byte offset of `ppResources` (`*const *mut c_void`) inside one plane parameter struct.
pub const PLANE_RESOURCES_OFFSET: usize = 0x10;

/// Full-screen triangle shader sampling the second presentation plane.
/// Intentionally carries no `[RootSignature]` attribute (see module doc, Wine compatibility).
#[cfg(any(windows, test))]
const SOURCE: &str = r#"
Texture2D<float4> plane : register(t0);
SamplerState bilinear : register(s0);
struct Corner { float4 position : SV_Position; float2 at : TEXCOORD0; };
Corner corner(uint i : SV_VertexID) {
    Corner c;
    c.at = float2(float(i & 1) * 2.0, float(i >> 1) * 2.0);
    c.position = float4(c.at.x * 2.0 - 1.0, 1.0 - c.at.y * 2.0, 0.0, 1.0);
    return c;
}
float4 over(Corner c) : SV_Target { return plane.Sample(bilinear, c.at); }
"#;

/// Whether second-plane compositing is enabled (`DECANT_D3D12X_PRESENT_OVER`, default on; `0` = off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_PRESENT_OVER").is_none_or(|v| v != "0"))
}

/// Maps a typeless colour format (as raw `DXGI_FORMAT` integer) to its typed SRV/RTV form for
/// compositing; typed formats pass through unchanged.
pub fn typed_format(format: i32) -> i32 {
    match format {
        1 => 2,   // R32G32B32A32_TYPELESS -> FLOAT
        9 => 10,  // R16G16B16A16_TYPELESS -> FLOAT
        23 => 24, // R10G10B10A2_TYPELESS -> UNORM
        27 => 28, // R8G8B8A8_TYPELESS -> UNORM
        90 => 87, // B8G8R8A8_TYPELESS -> UNORM
        f => f,
    }
}

/// Reads the first raw console resource pointer of plane `plane_index` from a `PresentX` plane
/// array (`plane_count` entries, each [`PLANE_STRIDE`] bytes). Returns null when out of range,
/// null, or empty.
///
/// # Safety
/// When `plane_index < plane_count` and `planes` is non-null, `planes` must be readable for at
/// least `(plane_index + 1) * PLANE_STRIDE` bytes, and its `ppResources` pointer (when non-null
/// with non-zero count) must point to at least one readable pointer.
#[allow(unsafe_code)]
pub unsafe fn plane_resource(plane_count: u32, planes: *const std::ffi::c_void, plane_index: u32) -> *mut std::ffi::c_void {
    if plane_index >= plane_count || planes.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: by the contract above.
    unsafe {
        let entry = planes.cast::<u8>().add(plane_index as usize * PLANE_STRIDE);
        let count = (entry.add(PLANE_COUNT_OFFSET) as *const u32).read_unaligned();
        let array = (entry.add(PLANE_RESOURCES_OFFSET) as *const *const *mut std::ffi::c_void).read_unaligned();
        if count == 0 || array.is_null() {
            return std::ptr::null_mut();
        }
        array.read_unaligned()
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod win {
    use super::{SOURCE, typed_format};
    use crate::log::xs_log;
    use std::ffi::c_void;
    use std::sync::Mutex;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
    use windows::core::{Interface, PCSTR, PCWSTR};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    /// Ring slots kept so views are not overwritten while prior presents are still in flight.
    const RING: u32 = 16;

    type SerializeFn = unsafe extern "system" fn(
        *const D3D12_ROOT_SIGNATURE_DESC,
        D3D_ROOT_SIGNATURE_VERSION,
        *mut Option<ID3DBlob>,
        *mut Option<ID3DBlob>,
    ) -> windows::core::HRESULT;

    type CompileFn = unsafe extern "system" fn(
        *const c_void,
        usize,
        PCSTR,
        *const c_void,
        *mut c_void,
        PCSTR,
        PCSTR,
        u32,
        u32,
        *mut Option<ID3DBlob>,
        *mut Option<ID3DBlob>,
    ) -> windows::core::HRESULT;

    /// Loads an export dynamically so we resolve `d3d12.dll` (not our own exported
    /// `D3D12SerializeRootSignature`) and `d3dcompiler_47.dll` lazily.
    unsafe fn dll_fn<F: Copy>(dll: PCWSTR, name: &std::ffi::CStr) -> Option<F> {
        const { assert!(size_of::<F>() == size_of::<usize>()) };
        // SAFETY: zero-terminated wide DLL name and ASCII export name.
        unsafe {
            let module = LoadLibraryW(dll.0);
            if module.is_null() {
                return None;
            }
            let f = GetProcAddress(module, name.as_ptr().cast())?;
            Some(std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, F>(&f))
        }
    }

    fn compile(entry: &std::ffi::CStr, target: &std::ffi::CStr) -> windows::core::Result<Vec<u8>> {
        let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
        // SAFETY: standard export of d3dcompiler_47.dll.
        let func = unsafe { dll_fn::<CompileFn>(windows::core::w!("d3dcompiler_47.dll"), c"D3DCompile") }.ok_or_else(fail)?;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: valid source buffer, zero-terminated entry/target, valid out pointers.
        let hr = unsafe {
            func(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                PCSTR::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                PCSTR(entry.as_ptr().cast()),
                PCSTR(target.as_ptr().cast()),
                0,
                0,
                &mut code,
                &mut errors,
            )
        };
        if hr.is_err() || code.is_none() {
            if let Some(err) = errors {
                // SAFETY: blob owns its bytes while alive.
                let text = unsafe {
                    String::from_utf8_lossy(std::slice::from_raw_parts(err.GetBufferPointer().cast::<u8>(), err.GetBufferSize())).into_owned()
                };
                xs_log!("present over: shader compile failed ({hr:?}): {text}");
            }
            return Err(windows::core::Error::from_hresult(hr));
        }
        let blob = code.ok_or_else(fail)?;
        // SAFETY: blob owns its bytes while alive.
        Ok(unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }.to_vec())
    }

    fn root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
        let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
        let serialize = unsafe { dll_fn::<SerializeFn>(windows::core::w!("d3d12.dll"), c"D3D12SerializeRootSignature") }.ok_or_else(fail)?;
        let range = [D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        }];
        let params = [D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: range.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        }];
        let samplers = [D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        }];
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: samplers.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        // SAFETY: descriptions point to live stack slices; `blob` is a valid out pointer.
        unsafe {
            serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, std::ptr::null_mut()).ok()?;
            let blob = blob.ok_or_else(fail)?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            device.CreateRootSignature(0, bytes)
        }
    }

    struct Pass {
        device_raw: usize,
        format: i32,
        root: ID3D12RootSignature,
        pipeline: ID3D12PipelineState,
        targets: ID3D12DescriptorHeap,
        views: ID3D12DescriptorHeap,
        target_step: usize,
        view_step: u32,
        next: u32,
    }

    // SAFETY: D3D12 objects are free-threaded; access is serialized by `PASS`.
    unsafe impl Send for Pass {}

    static PASS: Mutex<Option<Pass>> = Mutex::new(None);

    fn make(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Pass> {
        let vs = compile(c"corner", c"vs_5_1")?;
        let ps = compile(c"over", c"ps_5_1")?;
        let root = root_signature(device)?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: live device, root signature, and shader byte slices.
        unsafe {
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                SrcBlend: D3D12_BLEND_ONE,
                DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            let targets: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RING,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
            let views: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RING,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            Ok(Pass {
                device_raw: device.as_raw() as usize,
                format: format.0,
                root,
                pipeline,
                target_step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize,
                view_step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                targets,
                views,
                next: 0,
            })
        }
    }

    /// Records the compositing draw of `plane` (in `PIXEL_SHADER_RESOURCE`, sampled as
    /// `plane_format`) over `back` (in `RENDER_TARGET`, `width`x`height`, of `format`).
    /// Returns `false` if the pass could not be initialized (nothing recorded).
    pub fn record(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        format: DXGI_FORMAT,
        (width, height): (u32, u32),
        plane: &ID3D12Resource,
        plane_format: DXGI_FORMAT,
    ) -> bool {
        let dev_raw = device.as_raw() as usize;
        let mut pass = PASS.lock().unwrap_or_else(|e| e.into_inner());
        if pass.as_ref().is_none_or(|p| p.format != format.0 || p.device_raw != dev_raw) {
            *pass = match make(device, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        xs_log!("present over: no pass to draw a present's second plane ({e}): it is not shown");
                    });
                    return false;
                }
            };
            xs_log!("present over: second presentation plane pass ready (target format {}, {width}x{height})", format.0);
        }
        let Some(p) = pass.as_mut() else { return false };
        let slot = p.next % RING;
        p.next = p.next.wrapping_add(1);
        let srv_fmt = DXGI_FORMAT(typed_format(plane_format.0));
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: srv_fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let target_desc = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        // SAFETY: open command list, valid ring descriptor slots, live D3D12 resources.
        unsafe {
            let target = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: p.targets.GetCPUDescriptorHandleForHeapStart().ptr + slot as usize * p.target_step,
            };
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: p.views.GetCPUDescriptorHandleForHeapStart().ptr + (slot * p.view_step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: p.views.GetGPUDescriptorHandleForHeapStart().ptr + u64::from(slot * p.view_step),
            };
            device.CreateRenderTargetView(back, Some(&target_desc), target);
            device.CreateShaderResourceView(plane, Some(&view), cpu);
            list.SetDescriptorHeaps(&[Some(p.views.clone())]);
            list.SetGraphicsRootSignature(&p.root);
            list.SetPipelineState(&p.pipeline);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.OMSetRenderTargets(1, Some(&target), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4));
            list.DrawInstanced(3, 1, 0, 0);
        }
        true
    }
}

#[cfg(windows)]
pub use win::record;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typeless_overlay_formats_resolve_to_typed_srv_formats() {
        for (typeless, typed) in [(1, 2), (9, 10), (23, 24), (27, 28), (90, 87)] {
            assert_eq!(typed_format(typeless), typed);
            assert_eq!(typed_format(typed), typed);
        }
        for kept in [29, 91, 26, 0] {
            assert_eq!(typed_format(kept), kept);
        }
    }

    #[test]
    fn shader_source_carries_no_embedded_root_signature() {
        // Wine's d3dcompiler_47 omits [RootSignature] blobs (#188).
        assert!(!SOURCE.contains("RootSignature"));
        assert!(!SOURCE.contains("#define RS"));
    }

    #[test]
    #[allow(unsafe_code)]
    fn plane_resource_reads_plane_0_and_plane_1_at_0x48_stride() {
        let mut r0 = 0x1111usize;
        let mut r1 = 0x2222usize;
        let arr0 = [&raw mut r0 as *mut std::ffi::c_void];
        let arr1 = [&raw mut r1 as *mut std::ffi::c_void];
        let mut buf = [0u8; PLANE_STRIDE * 2];
        // Plane 0: count 1, ptr = arr0
        buf[PLANE_COUNT_OFFSET..PLANE_COUNT_OFFSET + 4].copy_from_slice(&1u32.to_le_bytes());
        buf[PLANE_RESOURCES_OFFSET..PLANE_RESOURCES_OFFSET + 8].copy_from_slice(&(arr0.as_ptr() as u64).to_le_bytes());
        // Plane 1: count 1, ptr = arr1
        buf[PLANE_STRIDE + PLANE_COUNT_OFFSET..PLANE_STRIDE + PLANE_COUNT_OFFSET + 4].copy_from_slice(&1u32.to_le_bytes());
        buf[PLANE_STRIDE + PLANE_RESOURCES_OFFSET..PLANE_STRIDE + PLANE_RESOURCES_OFFSET + 8].copy_from_slice(&(arr1.as_ptr() as u64).to_le_bytes());
        unsafe {
            assert_eq!(plane_resource(2, buf.as_ptr().cast(), 0), arr0[0]);
            assert_eq!(plane_resource(2, buf.as_ptr().cast(), 1), arr1[0]);
            assert!(plane_resource(1, buf.as_ptr().cast(), 1).is_null());
            assert!(plane_resource(2, buf.as_ptr().cast(), 2).is_null());
            assert!(plane_resource(2, std::ptr::null(), 0).is_null());
        }
    }
}
