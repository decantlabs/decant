//! A small D3D12 harness for the tests: a device (default adapter, WARP as fallback), compute
//! pipeline creation from a container, and a dispatch with root descriptors and read back.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::time::Duration;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS;

pub use decant_shader_ab::win::device::{buffer, transition, upload};

/// The device of `decant-shader-ab`, which also finds the adapter and waits for the GPU. These tests
/// need Shader Model 6.6.
pub struct Gpu(decant_shader_ab::win::device::Gpu);

impl std::ops::Deref for Gpu {
    type Target = decant_shader_ab::win::device::Gpu;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// What a shader binds, in root parameter order. Registers count per type: `t0`, `u0`, `b0`...
pub enum Bind {
    /// A raw SRV filled with these bytes.
    Srv(Vec<u8>),
    /// A constant buffer filled with these bytes.
    Cbv(Vec<u8>),
    /// A raw UAV of this many bytes, zero filled. Its final contents are returned.
    Uav(usize),
    /// Like `Uav`, in register `register` of space 0 instead of the next free one.
    UavAt { register: u32, size: usize },
    /// A raw UAV of this many bytes in slot `slot` of a shader visible descriptor heap (the root
    /// signature gets `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`). Its final contents are returned, in
    /// the order of the `Uav` and `HeapUav` entries.
    HeapUav { slot: u32, size: usize },
    /// The device memory window of the SPIR-V importer: a root UAV in register space 14, register 0,
    /// holding these bytes, and a root CBV in the same space with the 64 bit address of the UAV in its
    /// first two dwords. Its final contents are returned, in the order of the `Uav` entries.
    Window(Vec<u8>),
    /// The user data mirror of the importer: a root CBV at register 1 of space 15 with these bytes.
    Mirror(Vec<u8>),
    /// The user data mirror with the address of the `Window` plus a byte offset written to the dwords `at` and `at + 1`.
    MirrorWithWindowAddress { bytes: Vec<u8>, at: usize, offset: u64 },
    /// A raw SRV whose first eight bytes are replaced by the address of the `Window`.
    SrvWithWindowAddress(Vec<u8>),
}

/// The default adapter when it supports Shader Model 6.6, else WARP. `DECANT_SHADER_DEVICE=warp` forces WARP,
/// `DECANT_SHADER_DEVICE=hardware` forbids it. `None` when no device runs SM 6.6.
pub fn open_device() -> Option<Gpu> {
    decant_shader_ab::win::device::open_device().filter(|g| g.model >= (6, 6)).map(Gpu)
}

fn blob_text(b: &ID3DBlob) -> String {
    // SAFETY: the blob owns its buffer.
    unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())).into_owned() }
}

impl Gpu {
    fn root_signature(&self, binds: &[Bind]) -> Result<ID3D12RootSignature, String> {
        let mut counts = [0u32; 3];
        let heap = binds.iter().any(|b| matches!(b, Bind::HeapUav { .. }));
        let param = |ty, reg, space| D3D12_ROOT_PARAMETER {
            ParameterType: ty,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: reg, RegisterSpace: space } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let params: Vec<D3D12_ROOT_PARAMETER> = binds
            .iter()
            .filter(|b| !matches!(b, Bind::HeapUav { .. }))
            .flat_map(|b| {
                let (ty, slot) = match b {
                    Bind::Srv(_) | Bind::SrvWithWindowAddress(_) => (D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
                    Bind::Uav(_) => (D3D12_ROOT_PARAMETER_TYPE_UAV, 1),
                    Bind::UavAt { register, .. } => return vec![param(D3D12_ROOT_PARAMETER_TYPE_UAV, *register, 0)],
                    Bind::Cbv(_) => (D3D12_ROOT_PARAMETER_TYPE_CBV, 2),
                    Bind::Window(_) => return vec![param(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 14), param(D3D12_ROOT_PARAMETER_TYPE_CBV, 0, 14)],
                    Bind::Mirror(_) | Bind::MirrorWithWindowAddress { .. } => return vec![param(D3D12_ROOT_PARAMETER_TYPE_CBV, 1, 15)],
                    Bind::HeapUav { .. } => unreachable!(),
                };
                let reg = counts[slot];
                counts[slot] += 1;
                vec![param(ty, reg, 0)]
            })
            .collect();
        let flags = if heap { D3D12_ROOT_SIGNATURE_FLAGS(0x400) } else { D3D12_ROOT_SIGNATURE_FLAG_NONE };
        let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), Flags: flags, ..Default::default() };
        let mut blob: Option<ID3DBlob> = None;
        let mut err: Option<ID3DBlob> = None;
        // SAFETY: the description and the parameter array outlive the call.
        unsafe {
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, Some(&mut err)).map_err(|e| format!("serialize: {e} {}", err.as_ref().map(blob_text).unwrap_or_default()))?;
            let blob = blob.ok_or("no root signature blob")?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            self.device.CreateRootSignature::<ID3D12RootSignature>(0, bytes).map_err(|e| format!("CreateRootSignature: {e}"))
        }
    }

    /// `CreateComputePipelineState` for the container with a root signature of `binds`.
    pub fn pipeline(&self, container: &[u8], binds: &[Bind]) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let rs = self.root_signature(binds)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: container.as_ptr().cast(), BytecodeLength: container.len() },
            ..Default::default()
        };
        // SAFETY: the description points at live data for the call.
        let pso = unsafe { self.device.CreateComputePipelineState::<ID3D12PipelineState>(&desc) };
        drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
        pso.map(|p| (rs, p)).map_err(|e| format!("CreateComputePipelineState: {e}"))
    }

    /// Creates the pipeline, dispatches `groups` and returns the contents of every `Bind::Uav`, in order.
    pub fn run(&self, container: &[u8], binds: &[Bind], groups: [u32; 3]) -> Result<Vec<Vec<u8>>, String> {
        let (rs, pso) = self.pipeline(container, binds)?;
        let device = &self.device;
        // SAFETY: standard command recording; every resource outlives the wait on the fence.
        unsafe {
            let alloc = device.CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
            let list = device.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso).map_err(|e| e.to_string())?;
            list.SetComputeRootSignature(&rs);
            let mut keep = Vec::new();
            let mut uavs: Vec<(ID3D12Resource, usize)> = Vec::new();
            let heap_slots = binds.iter().filter_map(|b| if let Bind::HeapUav { slot, .. } = b { Some(*slot + 1) } else { None }).max().unwrap_or(0);
            let heap: Option<ID3D12DescriptorHeap> = if heap_slots > 0 {
                let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: heap_slots, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
                Some(device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&d).map_err(|e| e.to_string())?)
            } else {
                None
            };
            if let Some(h) = &heap {
                list.SetDescriptorHeaps(&[Some(h.clone())]);
            }
            // The window first, its address goes into the other buffers.
            let mut window: Option<(ID3D12Resource, usize, u64)> = None;
            if let Some(Bind::Window(init)) = binds.iter().find(|b| matches!(b, Bind::Window(_))) {
                let r = buffer(device, D3D12_HEAP_TYPE_DEFAULT, init.len(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)?;
                let up = upload(device, init)?;
                list.CopyBufferRegion(&r, 0, &up, 0, init.len() as u64);
                list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                keep.push(up);
                let va = r.GetGPUVirtualAddress();
                window = Some((r, init.len(), va));
            }
            let mut root = 0u32;
            for b in binds {
                let i = root;
                match b {
                    Bind::HeapUav { .. } => {}
                    Bind::Window(_) => root += 2,
                    _ => root += 1,
                }
                match b {
                    Bind::SrvWithWindowAddress(d) => {
                        let mut data = d.clone();
                        let va = window.as_ref().map(|w| w.2).ok_or("the SRV needs a window")?;
                        data[..8].copy_from_slice(&va.to_le_bytes());
                        let r = upload(device, &data)?;
                        list.SetComputeRootShaderResourceView(i, r.GetGPUVirtualAddress());
                        keep.push(r);
                    }
                    Bind::Window(_) => {
                        let (r, n, va) = window.take().ok_or("one window only")?;
                        list.SetComputeRootUnorderedAccessView(i, va);
                        let mut base = Vec::new();
                        base.extend_from_slice(&va.to_le_bytes());
                        base.extend_from_slice(&[0u8; 8]);
                        let c = upload(device, &base)?;
                        list.SetComputeRootConstantBufferView(i + 1, c.GetGPUVirtualAddress());
                        keep.push(c);
                        uavs.push((r, n));
                    }
                    Bind::HeapUav { slot, size } => {
                        let r = buffer(device, D3D12_HEAP_TYPE_DEFAULT, *size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
                        let h = heap.as_ref().expect("heap");
                        let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
                        let mut cpu = h.GetCPUDescriptorHandleForHeapStart();
                        cpu.ptr += step * *slot as usize;
                        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: (*size / 4) as u32, StructureByteStride: 0, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_RAW },
                            },
                        };
                        device.CreateUnorderedAccessView(&r, None::<&ID3D12Resource>, Some(&view), cpu);
                        uavs.push((r, *size));
                    }
                    Bind::Srv(d) => {
                        let r = upload(device, d)?;
                        list.SetComputeRootShaderResourceView(i, r.GetGPUVirtualAddress());
                        keep.push(r);
                    }
                    Bind::MirrorWithWindowAddress { bytes, at, offset } => {
                        let mut data = bytes.clone();
                        let va = window.as_ref().map(|w| w.2).ok_or("the mirror needs a window")? + offset;
                        data[4 * at..4 * at + 8].copy_from_slice(&va.to_le_bytes());
                        let r = upload(device, &data)?;
                        list.SetComputeRootConstantBufferView(i, r.GetGPUVirtualAddress());
                        keep.push(r);
                    }
                    Bind::Cbv(d) | Bind::Mirror(d) => {
                        let r = upload(device, d)?;
                        list.SetComputeRootConstantBufferView(i, r.GetGPUVirtualAddress());
                        keep.push(r);
                    }
                    Bind::Uav(n) | Bind::UavAt { size: n, .. } => {
                        let r = buffer(device, D3D12_HEAP_TYPE_DEFAULT, *n, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
                        list.SetComputeRootUnorderedAccessView(i, r.GetGPUVirtualAddress());
                        uavs.push((r, *n));
                    }
                }
            }
            list.Dispatch(groups[0], groups[1], groups[2]);
            let mut readbacks = Vec::new();
            for (r, n) in &uavs {
                let rb = buffer(device, D3D12_HEAP_TYPE_READBACK, *n, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
                list.ResourceBarrier(&[transition(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.CopyBufferRegion(&rb, 0, r, 0, *n as u64);
                readbacks.push(rb);
            }
            self.submit(&list, Duration::from_secs(20))?;
            let mut out = Vec::new();
            for (rb, (_, n)) in readbacks.iter().zip(&uavs) {
                let mut p: *mut c_void = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
                out.push(std::slice::from_raw_parts(p.cast::<u8>(), *n).to_vec());
                rb.Unmap(0, None);
            }
            drop(keep);
            Ok(out)
        }
    }
}

impl Gpu {
    /// The wave lane count range the device reports, `(min, max)`.
    pub fn wave_lane_range(&self) -> (u32, u32) {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: the structure matches the feature.
        let ok = unsafe { self.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() };
        if ok && o.WaveOps.as_bool() { (o.WaveLaneCountMin, o.WaveLaneCountMax) } else { (0, 0) }
    }
}

pub fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

pub fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}
