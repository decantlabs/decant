//! A small compute copy on the PC GPU, for the test programs that read memory through a GPU
//! address: `d3d12xs_test` includes this file with `#[path]`, it is not part of the DLL. The shader of `shaders/section_copy.hlsl` copies words from a root SRV
//! to a root UAV, and the result lands in a buffer the CPU reads.
#![allow(unsafe_code)]
#![allow(dead_code)]

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCSTR, PCWSTR};

/// Bytes of the output buffer, and so the most a `copy` reads back.
pub const OUT_BYTES: usize = 0x1_0000;

/// Threads per group of `shaders/section_copy.hlsl` (its `numthreads`).
pub const GROUP_THREADS: u32 = 64;

/// Groups that cover `words` words: the last group may be partly past the end, and the shader
/// returns for those threads from the `words` root constant.
pub fn dispatch_groups(words: u32) -> u32 {
    words.div_ceil(GROUP_THREADS)
}

/// The device, one direct queue and the compute copy.
pub struct Gpu {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
    signature: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    /// A custom write back buffer in system memory that the shader writes and the CPU reads.
    out: ID3D12Resource,
    out_ptr: *mut u32,
}

pub fn buffer_desc(width: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        Alignment: 0,
    }
}

fn root_view(kind: D3D12_ROOT_PARAMETER_TYPE) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn compile_copy() -> Option<Vec<u8>> {
    let source = include_str!("../shaders/section_copy.hlsl");
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source is a valid buffer of the length passed and the output is a valid location.
    let made = unsafe { D3DCompile(source.as_ptr().cast(), source.len(), PCSTR::null(), None, None, PCSTR(c"main".as_ptr().cast()), PCSTR(c"cs_5_1".as_ptr().cast()), 0, 0, &mut code, None) };
    let code = code.filter(|_| made.is_ok())?;
    // SAFETY: the blob owns `GetBufferSize` bytes at `GetBufferPointer` while it lives.
    Some(unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) }.to_vec())
}

impl Gpu {
    /// A device on the default adapter and the copy pipeline. `None` without a D3D12 12_0 device or
    /// the shader compiler.
    pub fn new() -> Option<Gpu> {
        // SAFETY: plain D3D12 calls on objects this function creates and keeps.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            let device = device?;
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).ok()?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let words_param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 1 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let params = [root_view(D3D12_ROOT_PARAMETER_TYPE_SRV), root_view(D3D12_ROOT_PARAMETER_TYPE_UAV), words_param];
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 3, pParameters: params.as_ptr(), ..Default::default() };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).ok()?;
            let blob = blob?;
            let signature: ID3D12RootSignature = device.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())).ok()?;
            let shader = compile_copy()?;
            let state = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(signature.clone())),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr().cast(), BytecodeLength: shader.len() },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateComputePipelineState(&state).ok()?;
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
            let mut out: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer_desc(OUT_BYTES as u64, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut out).ok()?;
            let out = out?;
            let mut mapped = std::ptr::null_mut();
            out.Map(0, None, Some(&mut mapped)).ok()?;
            Some(Gpu { device, queue, allocator, list, fence, value: 0, signature, pipeline, out, out_ptr: mapped.cast() })
        }
    }

    /// Runs `record` on the queue and waits for the fence. False when the wait timed out or the device was removed.
    pub fn submit(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> bool {
        // SAFETY: the allocator and list are idle (every submission waits), and the event is closed below.
        unsafe {
            if self.allocator.Reset().is_err() || self.list.Reset(&self.allocator, None).is_err() {
                return false;
            }
            record(&self.list);
            let Ok(list) = self.list.cast::<ID3D12CommandList>() else { return false };
            if self.list.Close().is_err() {
                return false;
            }
            self.queue.ExecuteCommandLists(&[Some(list)]);
            self.value += 1;
            let Ok(event) = CreateEventW(None, false, false, PCWSTR::null()) else { return false };
            let armed = self.queue.Signal(&self.fence, self.value).is_ok() && self.fence.SetEventOnCompletion(self.value, event).is_ok();
            let done = armed && WaitForSingleObject(event, 5000).0 == 0;
            let _ = CloseHandle(event);
            done && self.device.GetDeviceRemovedReason().is_ok()
        }
    }

    /// The shader copies `words` words from GPU address `src` to `dst` (the output buffer when `None`).
    /// Returns what the output buffer holds afterwards, `None` when the submission failed.
    pub fn copy(&mut self, src: u64, dst: Option<u64>, words: u32) -> Option<Vec<u32>> {
        assert!(words as usize * 4 <= OUT_BYTES);
        // SAFETY: `out_ptr` maps `OUT_BYTES` bytes of the output buffer, and `words` fits it.
        let out_va = unsafe {
            std::ptr::write_bytes(self.out_ptr, 0xEE, words as usize);
            self.out.GetGPUVirtualAddress()
        };
        let (signature, pipeline) = (self.signature.clone(), self.pipeline.clone());
        let ok = self.submit(|list| {
            // SAFETY: recording on an open list with objects that outlive the submission.
            unsafe {
                list.SetComputeRootSignature(&signature);
                list.SetPipelineState(&pipeline);
                list.SetComputeRootShaderResourceView(0, src);
                list.SetComputeRootUnorderedAccessView(1, dst.unwrap_or(out_va));
                list.SetComputeRoot32BitConstant(2, words, 0);
                list.Dispatch(dispatch_groups(words), 1, 1);
            }
        });
        // SAFETY: as above, the GPU is done.
        ok.then(|| unsafe { std::slice::from_raw_parts(self.out_ptr, words as usize) }.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_cover_aligned_and_tail_sizes() {
        assert_eq!(dispatch_groups(0), 0);
        for (words, groups) in [(1, 1), (63, 1), (64, 1), (65, 2), (127, 2), (128, 2), (129, 3)] {
            assert_eq!(dispatch_groups(words), groups, "{words} words");
            assert!(groups * GROUP_THREADS >= words && (groups - 1) * GROUP_THREADS < words);
        }
    }

    #[test]
    fn shader_returns_before_touching_words_past_the_count() {
        let source = include_str!("../shaders/section_copy.hlsl");
        let guard = source.find("id.x >= words").expect("a bounds check on the word count");
        let load = source.find("src.Load").expect("the load");
        let store = source.find("dst.Store").expect("the store");
        assert!(guard < load && guard < store);
        assert!(source.contains("numthreads(64, 1, 1)"));
    }

    /// Aligned and non-multiple-of-64 sizes on a real device: every word arrives, and the guard
    /// words after the copy keep their value. Skipped without a D3D12 device.
    #[test]
    fn gpu_copy_keeps_tail_and_guard() {
        let Some(mut gpu) = Gpu::new() else { return };
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        let total = 256u32;
        let mut source: Option<ID3D12Resource> = None;
        // SAFETY: plain resource creation and a map of the buffer this test owns, `total` words long.
        let src_va = unsafe {
            gpu.device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer_desc(total as u64 * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut source).expect("a source buffer");
            let source = source.as_ref().expect("a source buffer");
            let mut mapped = std::ptr::null_mut();
            source.Map(0, None, Some(&mut mapped)).expect("map");
            for i in 0..total {
                mapped.cast::<u32>().add(i as usize).write(0x1000 + i);
            }
            source.GetGPUVirtualAddress()
        };
        for words in [1u32, 63, 64, 65, 127, 128, 129, 191, 192] {
            // SAFETY: the output buffer maps OUT_BYTES, and the guard region ends well inside it.
            unsafe { std::ptr::write_bytes(gpu.out_ptr.add(words as usize), 0xAB, GROUP_THREADS as usize) };
            let got = gpu.copy(src_va, None, words).expect("a finished copy");
            let expected: Vec<u32> = (0..words).map(|i| 0x1000 + i).collect();
            assert_eq!(got, expected, "{words} words");
            // SAFETY: as above.
            let guard = unsafe { std::slice::from_raw_parts(gpu.out_ptr.add(words as usize), GROUP_THREADS as usize) };
            assert!(guard.iter().all(|&w| w == 0xABAB_ABAB), "{words} words wrote past the end");
        }
    }
}