//! Device tests of the views the title makes through the console device: on the default adapter
//! and on WARP, each skipped when it has no D3D12 12_0 device. A view is made through the console
//! slots, as the title makes it, and read on the GPU by a shader that declares every buffer a
//! (RW)ByteAddressBuffer, as the translated shaders do.
#![allow(unsafe_code)]

use std::ffi::c_void;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_12_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{GUID, Interface, PCSTR, PCWSTR};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

use crate::abi::Kind;

/// The function in slot `slot` of the vtable of `object`.
///
/// # Safety
/// `object` is a live COM object or console proxy, and `F` is the function type of that slot.
pub unsafe fn vt<F: Copy>(object: *mut c_void, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: by the contract above.
    unsafe {
        let table = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*table.add(slot))
    }
}

pub type CreateHeapX = unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32;
pub type CreateSrvX = unsafe extern "system" fn(*mut c_void, *mut c_void, *const c_void, usize);
pub type CreateUavX = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const c_void, usize);
pub type CopySimpleX = unsafe extern "system" fn(*mut c_void, u32, usize, usize, u32);
pub type CopyX = unsafe extern "system" fn(*mut c_void, u32, *const usize, *const u32, u32, *const usize, *const u32, u32);
pub type HeapStartX = unsafe extern "system" fn(*mut c_void) -> u64;

/// The export `name` of the system `d3d12.dll`. This crate exports `D3D12CreateDevice` and the root
/// signature serialisers itself, so its tests cannot link the PC ones.
fn d3d12_export(name: &std::ffi::CStr) -> Option<usize> {
    let wide: Vec<u16> = "d3d12.dll\0".encode_utf16().collect();
    // SAFETY: a NUL terminated name, the module stays loaded for the process.
    unsafe {
        let module = LoadLibraryW(wide.as_ptr());
        if module.is_null() {
            return None;
        }
        GetProcAddress(module, name.as_ptr().cast()).map(|f| f as usize)
    }
}

type CreateDevicePc = unsafe extern "system" fn(*mut c_void, D3D_FEATURE_LEVEL, *const GUID, *mut *mut c_void) -> i32;
type SerializePc = unsafe extern "system" fn(*const D3D12_ROOT_SIGNATURE_DESC, D3D_ROOT_SIGNATURE_VERSION, *mut *mut c_void, *mut *mut c_void) -> i32;

/// A 12_0 device of the system `d3d12.dll` on `adapter` (null: the default adapter).
fn pc_device(adapter: *mut c_void) -> Option<ID3D12Device> {
    let create = d3d12_export(c"D3D12CreateDevice")?;
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: the export of that name has this signature, `raw` takes one reference.
    unsafe {
        let create = std::mem::transmute::<usize, CreateDevicePc>(create);
        if create(adapter, D3D_FEATURE_LEVEL_12_0, &ID3D12Device::IID, &mut raw) < 0 || raw.is_null() {
            return None;
        }
        Some(ID3D12Device::from_raw(raw))
    }
}

/// The devices to test on: the default adapter and WARP, each when it has a 12_0 device.
pub fn devices() -> Vec<(&'static str, ID3D12Device)> {
    let mut out = Vec::new();
    if let Some(d) = pc_device(std::ptr::null_mut()) {
        out.push(("default adapter", d));
    }
    // SAFETY: plain DXGI calls on objects this function creates.
    let warp = unsafe { CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
    if let Ok(adapter) = warp
        && let Some(d) = pc_device(adapter.as_raw())
    {
        out.push(("WARP", d));
    }
    if out.is_empty() {
        println!("no D3D12 12_0 device and no WARP: the device tests are skipped");
    }
    out
}

/// A console device proxy over `device` (it holds its own reference).
pub fn console_device(device: &ID3D12Device) -> *mut c_void {
    let raw = device.as_raw();
    // SAFETY: a live device, one more reference for the proxy, which takes it over.
    unsafe { vt::<unsafe extern "system" fn(*mut c_void) -> u32>(raw, 1)(raw) };
    let dev = crate::proxy::wrap(raw, Kind::Device);
    assert!(!dev.is_null(), "the console proxies are available");
    dev
}

/// Releases a console proxy.
pub fn release(object: *mut c_void) {
    // SAFETY: callers pass a live object they own a reference to.
    unsafe { vt::<unsafe extern "system" fn(*mut c_void) -> u32>(object, 2)(object) };
}

/// A view heap made through console slot 14: the proxy, its PC heap, CPU and GPU starts.
pub struct ConsoleHeap {
    pub proxy: *mut c_void,
    pub pc: ID3D12DescriptorHeap,
    pub cpu: usize,
    pub gpu: u64,
    pub increment: usize,
}

impl ConsoleHeap {
    pub fn new(dev: *mut c_void, device: &ID3D12Device, count: u32, visible: bool) -> ConsoleHeap {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: count,
            Flags: if visible { D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE } else { D3D12_DESCRIPTOR_HEAP_FLAG_NONE },
            NodeMask: 0,
        };
        let mut proxy: *mut c_void = std::ptr::null_mut();
        // SAFETY: `dev` is a console device proxy, the slots take these arguments.
        unsafe {
            let hr = vt::<CreateHeapX>(dev, 14)(dev, &desc, &ID3D12DescriptorHeap::IID, &mut proxy);
            assert!(hr >= 0 && !proxy.is_null(), "console slot 14 makes the heap (hr {hr:#x})");
            let pc_raw = *(proxy.cast::<u8>().add(8).cast::<*mut c_void>());
            let pc = ID3D12DescriptorHeap::from_raw_borrowed(&pc_raw).expect("the PC heap").clone();
            let cpu = vt::<HeapStartX>(proxy, 9)(proxy) as usize;
            let gpu = if visible { vt::<HeapStartX>(proxy, 10)(proxy) } else { 0 };
            let increment = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            ConsoleHeap { proxy, pc, cpu, gpu, increment }
        }
    }

    pub fn cpu_at(&self, slot: usize) -> usize {
        self.cpu + slot * self.increment
    }

    /// The CPU start of the PC heap (the title's handles start in title memory).
    pub fn pc_cpu(&self) -> usize {
        // SAFETY: a live heap.
        unsafe { self.pc.GetCPUDescriptorHandleForHeapStart() }.ptr
    }
}

impl Drop for ConsoleHeap {
    fn drop(&mut self) {
        release(self.proxy);
    }
}

pub fn buffer_desc(width: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        Alignment: 0,
    }
}

/// A buffer in system memory that the GPU reads and writes and the CPU maps: `words` words.
pub struct HostBuffer {
    pub res: ID3D12Resource,
    pub ptr: *mut u32,
    pub words: usize,
}

impl HostBuffer {
    pub fn new(device: &ID3D12Device, words: usize) -> HostBuffer {
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() };
        // SAFETY: plain D3D12 calls, the mapping lives as long as the resource.
        unsafe {
            let mut res: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &buffer_desc(words as u64 * 4), D3D12_RESOURCE_STATE_COMMON, None, &mut res).expect("a host buffer");
            let res = res.expect("a host buffer");
            let mut ptr = std::ptr::null_mut();
            res.Map(0, None, Some(&mut ptr)).expect("the host buffer maps");
            HostBuffer { res, ptr: ptr.cast(), words }
        }
    }

    pub fn words(&self) -> Vec<u32> {
        // SAFETY: the mapping holds `words` words.
        unsafe { std::slice::from_raw_parts(self.ptr, self.words) }.to_vec()
    }

    pub fn fill(&self, f: impl Fn(usize) -> u32) {
        for i in 0..self.words {
            // SAFETY: the mapping holds `words` words.
            unsafe { self.ptr.add(i).write(f(i)) };
        }
    }
}

/// Compiles compute shader `source` (entry `main`, cs_5_1).
pub fn compile(source: &str) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source is a valid buffer of the length passed, the outputs valid locations.
    let made = unsafe { D3DCompile(source.as_ptr().cast(), source.len(), PCSTR::null(), None, None, PCSTR(c"main".as_ptr().cast()), PCSTR(c"cs_5_1".as_ptr().cast()), 0, 0, &mut code, Some(&mut errors)) };
    let text = |b: &ID3DBlob| {
        // SAFETY: the blob owns `GetBufferSize` bytes while it lives.
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }).into_owned()
    };
    assert!(made.is_ok(), "the test shader compiles: {}", errors.as_ref().map(text).unwrap_or_default());
    let code = code.expect("a shader");
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) }.to_vec()
}

/// A root signature from parameters (version 1.0).
pub fn root_signature(device: &ID3D12Device, params: &[D3D12_ROOT_PARAMETER]) -> ID3D12RootSignature {
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    let serialize = d3d12_export(c"D3D12SerializeRootSignature").expect("d3d12.dll serialises root signatures");
    // SAFETY: the export of that name has this signature, the blob lives while it is read.
    unsafe {
        let serialize = std::mem::transmute::<usize, SerializePc>(serialize);
        let mut raw: *mut c_void = std::ptr::null_mut();
        assert!(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut raw, std::ptr::null_mut()) >= 0 && !raw.is_null(), "the root signature serialises");
        let blob = ID3DBlob::from_raw(raw);
        device.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())).expect("the root signature")
    }
}

/// A compute pipeline of `shader` on `signature`.
pub fn pipeline(device: &ID3D12Device, signature: &ID3D12RootSignature, shader: &[u8]) -> ID3D12PipelineState {
    let state = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::ManuallyDrop::new(Some(signature.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: shader.as_ptr().cast(), BytecodeLength: shader.len() },
        ..Default::default()
    };
    // SAFETY: a valid description whose shader outlives the call.
    unsafe { device.CreateComputePipelineState(&state) }.expect("the pipeline")
}

/// Records `record` on a fresh direct list, runs it and waits. False when the device was removed.
pub fn run(device: &ID3D12Device, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> bool {
    // SAFETY: objects made here live until the wait returned.
    unsafe {
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("a queue");
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("an allocator");
        let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).expect("a list");
        record(&list);
        list.Close().expect("the list closes");
        queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().expect("a command list"))]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
        let event = CreateEventW(None, false, false, PCWSTR::null()).expect("an event");
        let armed = queue.Signal(&fence, 1).is_ok() && fence.SetEventOnCompletion(1, event).is_ok();
        let done = armed && WaitForSingleObject(event, 10_000).0 == 0;
        let _ = CloseHandle(event);
        done && device.GetDeviceRemovedReason().is_ok()
    }
}

pub fn table_param(ranges: &[D3D12_DESCRIPTOR_RANGE]) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: ranges.len() as u32, pDescriptorRanges: ranges.as_ptr() } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

pub fn root_view(kind: D3D12_ROOT_PARAMETER_TYPE, register: u32, space: u32) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: register, RegisterSpace: space } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

pub fn range(kind: D3D12_DESCRIPTOR_RANGE_TYPE, register: u32, offset: u32) -> D3D12_DESCRIPTOR_RANGE {
    D3D12_DESCRIPTOR_RANGE { RangeType: kind, NumDescriptors: 1, BaseShaderRegister: register, RegisterSpace: 0, OffsetInDescriptorsFromTableStart: offset }
}

/// A structured buffer view description of the title: `first` and `count` records of `stride`.
pub fn structured_srv(first: u64, count: u32, stride: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Buffer: D3D12_BUFFER_SRV { FirstElement: first, NumElements: count, StructureByteStride: stride, Flags: D3D12_BUFFER_SRV_FLAG_NONE } },
    }
}

pub fn structured_uav(first: u64, count: u32, stride: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Buffer: D3D12_BUFFER_UAV { FirstElement: first, NumElements: count, StructureByteStride: stride, CounterOffsetInBytes: 0, Flags: D3D12_BUFFER_UAV_FLAG_NONE } },
    }
}

/// Copies what the view at table slot 0 (t0) reads to `out` (u0, a root view) with its size in
/// bytes at word 63, and writes `0xA000 + i` into every word of the view at table slot 1 (u1) with
/// its size at word 62.
const READ_WRITE: &str = r"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u1);
RWByteAddressBuffer out_ : register(u0);
[numthreads(1, 1, 1)]
void main()
{
    uint n;
    src.GetDimensions(n);
    for (uint i = 0; i < n / 4 && i < 60; i++)
        out_.Store(i * 4, src.Load(i * 4));
    out_.Store(252, n);
    uint m;
    dst.GetDimensions(m);
    for (uint j = 0; j < m / 4 && j < 60; j++)
        dst.Store(j * 4, 0xA000 + j);
    out_.Store(248, m);
}
";

/// Runs [`READ_WRITE`] with the table at slot `first` of `heap` (t0 there, u1 in the slot after).
/// The words of `out` (64) afterwards.
pub fn read_write(device: &ID3D12Device, heap: &ConsoleHeap, first: u64, out: &HostBuffer) -> Vec<u32> {
    let ranges = [range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 0), range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1)];
    let signature = root_signature(device, &[table_param(&ranges), root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)]);
    let state = pipeline(device, &signature, &compile(READ_WRITE));
    out.fill(|_| 0xEEEE_EEEE);
    let ran = run(device, |list| {
        // SAFETY: recording on an open list with objects that outlive the submission.
        unsafe {
            list.SetDescriptorHeaps(&[Some(heap.pc.clone())]);
            list.SetComputeRootSignature(&signature);
            list.SetPipelineState(&state);
            list.SetComputeRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: heap.gpu + first * heap.increment as u64 });
            list.SetComputeRootUnorderedAccessView(1, out.res.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
        }
    });
    assert!(ran, "the shader ran");
    out.words()
}

#[test]
fn a_constant_buffer_view_of_no_bytes_forgets_the_unaligned_view_its_slot_held() {
    type CreateCbvX = unsafe extern "system" fn(*mut c_void, *const c_void, usize);
    for (name, device) in devices() {
        let dev = console_device(&device);
        let staging = ConsoleHeap::new(dev, &device, 2, false);
        let data = HostBuffer::new(&device, 256);
        let host = crate::title_heap::translate_cpu(staging.cpu_at(0)).expect("a host slot");
        // SAFETY: console device slot 17 with descriptions of {address u64, size u32} and a slot of the heap.
        unsafe {
            let unaligned: (u64, u32, u32) = (data.res.GetGPUVirtualAddress() + 0x20, 64, 0);
            vt::<CreateCbvX>(dev, 17)(dev, (&raw const unaligned).cast(), staging.cpu_at(0));
            assert!(crate::title_heap::unaligned_cbv(host).is_some(), "{name}: the unaligned view is noted");
            let empty: (u64, u32, u32) = (0, 0, 0);
            vt::<CreateCbvX>(dev, 17)(dev, (&raw const empty).cast(), staging.cpu_at(0));
        }
        assert_eq!(crate::title_heap::unaligned_cbv(host), None, "{name}: the null view replaced it");
        // SAFETY: a plain query on a live device.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok(), "{name}: the device is still there");
    }
}

#[test]
fn an_empty_buffer_view_at_an_address_nothing_holds_replaces_the_view_of_its_slot() {
    type CreateViewAtX = unsafe extern "system" fn(*mut c_void, *const c_void, *const u8, usize);
    for (name, device) in devices() {
        let dev = console_device(&device);
        let staging = ConsoleHeap::new(dev, &device, 2, false);
        let data = HostBuffer::new(&device, 64);
        let row = |cpu: usize| crate::view_shadow::row(crate::title_heap::translate_cpu(cpu).expect("a host slot"));
        let srv = structured_srv(1, 3, 16);
        // the title's view at an address: the PC description, then the address at 0x28
        let mut at = [0u8; 0x30];
        let empty = structured_srv(0, 0, 16);
        // SAFETY: a plain copy of the 40 byte description.
        unsafe { std::ptr::copy_nonoverlapping((&raw const empty).cast::<u8>(), at.as_mut_ptr(), 0x28) };
        at[0x28..0x30].copy_from_slice(&0x7F00_0000_0000u64.to_le_bytes());
        // SAFETY: console device slots 18 and 70 with a live resource, the descriptions above and a slot of the heap.
        unsafe {
            vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const srv).cast(), staging.cpu_at(0));
            assert!(row(staging.cpu_at(0)).is_some_and(|r| r != [0; 4]), "{name}: the slot holds a view");
            vt::<CreateViewAtX>(dev, 70)(dev, std::ptr::null(), at.as_ptr(), staging.cpu_at(0));
        }
        assert_eq!(row(staging.cpu_at(0)), Some([0; 4]), "{name}: a null view replaced it");
        // SAFETY: a plain query on a live device.
        assert!(unsafe { device.GetDeviceRemovedReason() }.is_ok(), "{name}: the device is still there");
    }
}

#[test]
fn a_structured_view_of_the_title_reads_and_writes_its_records_as_raw_bytes() {
    for (name, device) in devices() {
        println!("raw views on {name}");
        let dev = console_device(&device);
        let heap = ConsoleHeap::new(dev, &device, 4, true);
        let data = HostBuffer::new(&device, 64);
        data.fill(|i| 0x1000 + i as u32);
        let target = HostBuffer::new(&device, 64);
        target.fill(|_| 0);
        let counter = HostBuffer::new(&device, 4);
        // The title's views: records 1 to 3 of 16 bytes, and records 2 to 5 of 8 bytes with a counter.
        let srv = structured_srv(1, 3, 16);
        let uav = structured_uav(2, 4, 8);
        // SAFETY: console device slots 18 and 19 with live resources and slots of `heap`.
        unsafe {
            vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const srv).cast(), heap.cpu_at(0));
            vt::<CreateUavX>(dev, 19)(dev, target.res.as_raw(), counter.res.as_raw(), (&raw const uav).cast(), heap.cpu_at(1));
        }
        let out = HostBuffer::new(&device, 64);
        let words = read_write(&device, &heap, 0, &out);
        let want: Vec<u32> = (4..16).map(|i| 0x1000 + i).collect();
        assert_eq!(&words[..12], &want[..], "{name}: the raw view reads bytes 16 to 64 of the buffer");
        assert_eq!((words[63], words[62]), (48, 32), "{name}: the views hold 48 and 32 bytes");
        let mut want = vec![0u32; 64];
        for (j, w) in want[4..12].iter_mut().enumerate() {
            *w = 0xA000 + j as u32;
        }
        assert_eq!(target.words(), want, "{name}: the raw unordered access view writes bytes 16 to 48 only");
        drop(heap);
        release(dev);
    }
}

/// Copies the first 32 words of the view shadow (`t1` of space 15, the rows of slots 0 to 7) to `out`.
const SHADOW_READ: &str = r"
ByteAddressBuffer shadow : register(t1, space15);
RWByteAddressBuffer out_ : register(u0);
[numthreads(1, 1, 1)]
void main()
{
    for (uint i = 0; i < 32; i++)
        out_.Store(i * 4, shadow.Load(i * 4));
}
";

#[test]
fn the_shadow_rows_follow_views_and_copies_and_the_gpu_reads_them() {
    use crate::view_raw::Records;
    use crate::view_shadow;
    let rec = |stride, count| Some(crate::view_shadow::records_row(Records { stride, count, format: 0, offset: 0 }));
    for (name, device) in devices() {
        println!("view shadow on {name}");
        let dev = console_device(&device);
        let visible = ConsoleHeap::new(dev, &device, 8, true);
        let staging = ConsoleHeap::new(dev, &device, 4, false);
        // The heaps have title memory (`title_heap` in the tests): rows are kept by host slot.
        let row = |cpu: usize| view_shadow::row(crate::title_heap::translate_cpu(cpu).expect("a host slot"));
        let data = HostBuffer::new(&device, 64);
        data.fill(|i| 0x1000 + i as u32);
        let target = HostBuffer::new(&device, 64);
        target.fill(|_| 0);
        let srv = structured_srv(1, 3, 16);
        let uav = structured_uav(2, 4, 8);
        // SAFETY: console device slots 18, 19, 24 and 23 with live resources and slots of the heaps.
        unsafe {
            vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const srv).cast(), staging.cpu_at(0));
            vt::<CreateUavX>(dev, 19)(dev, target.res.as_raw(), std::ptr::null_mut(), (&raw const uav).cast(), staging.cpu_at(1));
            assert_eq!((row(staging.cpu_at(0)), row(staging.cpu_at(1))), (rec(16, 3), rec(8, 4)), "{name}: rows after the views are made");
            // a raw view of the title's own: its row holds its dwords
            let mut raw = D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R32_TYPELESS, ..structured_srv(0, 4, 0) };
            raw.Anonymous.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const raw).cast(), visible.cpu_at(5));
            assert_eq!(row(visible.cpu_at(5)), Some([0, 4, 0, 0]));
            // copies into the shader visible heap carry the rows
            vt::<CopySimpleX>(dev, 24)(dev, 2, visible.cpu_at(2), staging.cpu_at(0), 0);
            let (dst, src, one) = ([visible.cpu_at(6)], [staging.cpu_at(0)], [1u32]);
            vt::<CopyX>(dev, 23)(dev, 1, dst.as_ptr(), one.as_ptr(), 1, src.as_ptr(), one.as_ptr(), 0);
        }
        assert_eq!((row(visible.cpu_at(2)), row(visible.cpu_at(3)), row(visible.cpu_at(6))), (rec(16, 3), rec(8, 4), rec(16, 3)), "{name}: rows after the copies");

        // The GPU reads the rows through the root SRV t1 of space 15.
        let buffer = view_shadow::buffer_gpu(visible.gpu).expect("the visible heap has a shadow buffer");
        let signature = root_signature(&device, &[root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1, 15), root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)]);
        let state = pipeline(&device, &signature, &compile(SHADOW_READ));
        let out = HostBuffer::new(&device, 64);
        out.fill(|_| 0xEEEE_EEEE);
        let ran = run(&device, |list| {
            // SAFETY: recording on an open list with objects that outlive the submission.
            unsafe {
                list.SetComputeRootSignature(&signature);
                list.SetPipelineState(&state);
                list.SetComputeRootShaderResourceView(0, buffer);
                list.SetComputeRootUnorderedAccessView(1, out.res.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
            }
        });
        assert!(ran, "{name}: the shadow shader ran");
        let mut want = [0u32; 32];
        want[8..16].copy_from_slice(&[16, 3, 0, 0, 8, 4, 0, 0]);
        want[20..28].copy_from_slice(&[0, 4, 0, 0, 16, 3, 0, 0]);
        assert_eq!(&out.words()[..32], &want[..], "{name}: the GPU reads the rows of slots 2, 3 and 6");

        // The copied descriptors are the raw views: slot 2 reads and slot 3 writes the records.
        let words = read_write(&device, &visible, 2, &out);
        let want: Vec<u32> = (4..16).map(|i| 0x1000 + i).collect();
        assert_eq!(&words[..12], &want[..], "{name}: the copied raw view reads bytes 16 to 64");
        assert_eq!((words[63], words[62]), (48, 32), "{name}: the copied views hold 48 and 32 bytes");
        drop(visible);
        drop(staging);
        release(dev);
    }
}

/// Reads the heap map entry of console slot `SLOT`, the host index it gives, and the view shadow
/// row of that host index, as a translated shader does.
const PAGED_READ: &str = r"
ByteAddressBuffer heap_map : register(t0, space15);
ByteAddressBuffer shadow : register(t1, space15);
RWByteAddressBuffer out_ : register(u0);
[numthreads(1, 1, 1)]
void main()
{
    uint slot = SLOT;
    uint page = heap_map.Load((slot >> 12) * 4);
    uint host = page + (slot & 4095);
    out_.Store(0, page);
    out_.Store(4, host);
    out_.Store4(16, shadow.Load4(host * 16));
}
";

pub type CreateListX = unsafe extern "system" fn(*mut c_void, u32, u32, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
pub type DispatchX = unsafe extern "system" fn(*mut c_void, u32, u32, u32);

#[test]
fn a_structured_view_above_the_first_page_of_a_paged_heap_has_its_row_at_the_host_slot_the_heap_map_names() {
    use crate::title_heap::{PAGE, UNMAPPED};
    // The title's view heap: more slots than the PC takes shader visible, so it is paged.
    const TITLE_SLOTS: u32 = 1_200_000;
    const SLOT: u32 = 3 * PAGE + 7;
    for (name, device) in devices() {
        println!("paged view heap on {name}");
        let dev = console_device(&device);
        let heap = ConsoleHeap::new(dev, &device, TITLE_SLOTS, true);
        let paged = crate::title_heap::paged(heap.gpu);
        println!("{name}: the view heap of {TITLE_SLOTS} slots is {}", if paged { "paged" } else { "not paged (the PC took it whole)" });
        let data = HostBuffer::new(&device, 64);
        data.fill(|i| 0x1000 + i as u32);
        let srv = structured_srv(1, 3, 16);
        // SAFETY: console device slot 18 with a live resource and a console slot of `heap`.
        unsafe { vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const srv).cast(), heap.cpu_at(SLOT as usize)) };

        // A console list: its dispatch copies the written slots into host pages before the PC dispatch.
        let signature = root_signature(&device, &[root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0, 15), root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1, 15), root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)]);
        let state = pipeline(&device, &signature, &compile(&PAGED_READ.replace("SLOT;", &format!("{SLOT};"))));
        let map = crate::title_heap::page_map_address(heap.gpu).expect("every shader visible view heap has a page map");
        let shadow = crate::view_shadow::buffer_gpu(heap.gpu).expect("the view heap has a shadow buffer");
        let out = HostBuffer::new(&device, 64);
        out.fill(|_| 0xEEEE_EEEE);
        // SAFETY: plain D3D12 calls and console slots 12 and 14 on the proxies made here, the list
        // runs once and is waited for.
        unsafe {
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("an allocator");
            let mut list: *mut c_void = std::ptr::null_mut();
            let hr = vt::<CreateListX>(dev, 12)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, allocator.as_raw(), std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list);
            assert!(hr >= 0 && !list.is_null(), "{name}: console slot 12 makes the list");
            let pc_raw = *(list.cast::<u8>().add(8).cast::<*mut c_void>());
            let pc = ID3D12GraphicsCommandList::from_raw_borrowed(&pc_raw).expect("the PC list").clone();
            pc.SetComputeRootSignature(&signature);
            pc.SetPipelineState(&state);
            pc.SetComputeRootShaderResourceView(0, map);
            pc.SetComputeRootShaderResourceView(1, shadow);
            pc.SetComputeRootUnorderedAccessView(2, out.res.GetGPUVirtualAddress());
            vt::<DispatchX>(list, 14)(list, 1, 1, 1);
            pc.Close().expect("the list closes");
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("a queue");
            queue.ExecuteCommandLists(&[Some(pc.cast::<ID3D12CommandList>().expect("a command list"))]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
            let event = CreateEventW(None, false, false, PCWSTR::null()).expect("an event");
            assert!(queue.Signal(&fence, 1).is_ok() && fence.SetEventOnCompletion(1, event).is_ok());
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0, "{name}: the dispatch finished");
            let _ = CloseHandle(event);
            drop(pc);
            release(list);
        }
        let words = out.words();
        let (page, host) = (words[0], words[1]);
        assert_ne!(page, UNMAPPED, "{name}: the written page has a host page");
        assert_eq!(host % PAGE, SLOT % PAGE, "{name}: the slot keeps its place in the page");
        assert_eq!(crate::title_heap::console_index(heap.gpu, host), Some(SLOT), "{name}: the host slot holds console slot {SLOT}");
        if paged {
            assert!(host < crate::title_heap::PC_MAX_SHADER_VISIBLE_VIEW_DESCRIPTORS, "{name}: the host slot is in the PC heap");
        } else {
            assert_eq!(host, SLOT, "{name}: an unpaged heap maps every page to itself");
        }
        assert_eq!(&words[4..8], &[16, 3, 0, 0], "{name}: the GPU reads the row of the view at the host slot");
        let host_cpu = heap.pc_cpu() + host as usize * heap.increment;
        assert_eq!(crate::view_shadow::row(host_cpu), Some([16, 3, 0, 0]), "{name}: the row of the host slot");
        drop(heap);
        release(dev);
    }
}

/// The first 32 words of the view shadow of the view heap at GPU start `gpu_start` (the rows of
/// slots 0 to 7), read by a shader through `t1` of space 15.
fn shadow_words(device: &ID3D12Device, gpu_start: u64) -> Vec<u32> {
    let buffer = crate::view_shadow::buffer_gpu(gpu_start).expect("the visible heap has a shadow buffer");
    let signature = root_signature(device, &[root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1, 15), root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)]);
    let state = pipeline(device, &signature, &compile(SHADOW_READ));
    let out = HostBuffer::new(device, 64);
    out.fill(|_| 0xEEEE_EEEE);
    let ran = run(device, |list| {
        // SAFETY: recording on an open list with objects that outlive the submission.
        unsafe {
            list.SetComputeRootSignature(&signature);
            list.SetPipelineState(&state);
            list.SetComputeRootShaderResourceView(0, buffer);
            list.SetComputeRootUnorderedAccessView(1, out.res.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
        }
    });
    assert!(ran, "the shadow shader ran");
    out.words()[..32].to_vec()
}

/// A texture in default memory: `dimension` 2 (1D), 3 (2D) or 4 (3D).
fn texture(device: &ID3D12Device, dimension: D3D12_RESOURCE_DIMENSION, size: (u64, u32, u16), mips: u16, samples: u32, flags: D3D12_RESOURCE_FLAGS) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: dimension,
        Width: size.0,
        Height: size.1,
        DepthOrArraySize: size.2,
        MipLevels: mips,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: flags,
        Alignment: 0,
    };
    let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: a plain D3D12 call with valid descriptions.
    unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res) }.expect("a texture");
    res.expect("a texture")
}

fn texture_srv(dimension: D3D12_SRV_DIMENSION, union: D3D12_SHADER_RESOURCE_VIEW_DESC_0) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC { Format: DXGI_FORMAT_R8G8B8A8_UNORM, ViewDimension: dimension, Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, Anonymous: union }
}

#[test]
fn texture_views_give_their_size_and_shape_in_the_shadow_row() {
    let words = |w: u32, h: u32, last: u32, levels: u32, image: u32| [0, 0, (w - 1) | ((h - 1) << 16), last | (levels << 16) | (image << 28)];
    for (name, device) in devices() {
        println!("texture rows on {name}");
        let dev = console_device(&device);
        let heap = ConsoleHeap::new(dev, &device, 8, true);
        let none = D3D12_RESOURCE_FLAG_NONE;
        let tex2d = texture(&device, D3D12_RESOURCE_DIMENSION_TEXTURE2D, (64, 32, 1), 5, 1, none);
        let tex3d = texture(&device, D3D12_RESOURCE_DIMENSION_TEXTURE3D, (16, 8, 4), 3, 1, none);
        let cubes = texture(&device, D3D12_RESOURCE_DIMENSION_TEXTURE2D, (16, 16, 12), 1, 1, none);
        let array = texture(&device, D3D12_RESOURCE_DIMENSION_TEXTURE2D, (8, 8, 6), 1, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        let ms = texture(&device, D3D12_RESOURCE_DIMENSION_TEXTURE2D, (16, 16, 1), 1, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        let views = [
            // slot 0: mips 1 and 2 of a 64 by 32 texture
            (&tex2d, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 1, MipLevels: 2, ..Default::default() } }))),
            // slot 1: a 3D texture from mip 1 on, mip levels left open
            (&tex3d, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURE3D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture3D: D3D12_TEX3D_SRV { MostDetailedMip: 1, MipLevels: u32::MAX, ..Default::default() } }))),
            // slot 2: the 3D texture's own view
            (&tex3d, None),
            // slot 3: a cube, slot 4: an array of two cubes
            (&cubes, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURECUBE, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCube: D3D12_TEXCUBE_SRV { MipLevels: u32::MAX, ..Default::default() } }))),
            (&cubes, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { TextureCubeArray: D3D12_TEXCUBE_ARRAY_SRV { MipLevels: 1, NumCubes: 2, ..Default::default() } }))),
            // slot 5: slices 2 to 4 of an array of six
            (&array, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, FirstArraySlice: 2, ArraySize: 3, ..Default::default() } }))),
            // slot 6: a texture of four samples
            (&ms, Some(texture_srv(D3D12_SRV_DIMENSION_TEXTURE2DMS, D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2DMS: D3D12_TEX2DMS_SRV::default() }))),
        ];
        // slot 7: unordered access to slices 1 to 5 of the array
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2DArray: D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 1, ArraySize: u32::MAX, PlaneSlice: 0 } },
        };
        // SAFETY: console device slots 18 and 19 with live resources and slots of `heap`.
        unsafe {
            for (slot, (res, desc)) in views.iter().enumerate() {
                let desc: *const c_void = desc.as_ref().map_or(std::ptr::null(), |d| (d as *const D3D12_SHADER_RESOURCE_VIEW_DESC).cast());
                vt::<CreateSrvX>(dev, 18)(dev, res.as_raw(), desc, heap.cpu_at(slot));
            }
            vt::<CreateUavX>(dev, 19)(dev, array.as_raw(), std::ptr::null_mut(), (&raw const uav).cast(), heap.cpu_at(7));
        }
        let want = [
            words(32, 16, 0, 1, 9),
            words(8, 4, 1, 1, 10),
            words(16, 8, 3, 2, 10),
            words(16, 16, 5, 0, 11),
            words(16, 16, 11, 0, 11),
            words(8, 8, 2, 0, 13),
            words(16, 16, 0, 2, 14),
            words(8, 8, 4, 0, 13),
        ];
        let got = shadow_words(&device, heap.gpu);
        for (slot, row) in want.iter().enumerate() {
            assert_eq!(&got[slot * 4..slot * 4 + 4], row, "{name}: the row of slot {slot}");
        }
        // a copy carries the image words
        // SAFETY: console device slot 24 with slots of `heap`.
        unsafe { vt::<CopySimpleX>(dev, 24)(dev, 1, heap.cpu_at(1), heap.cpu_at(6), 0) };
        let got = shadow_words(&device, heap.gpu);
        assert_eq!(&got[4..8], &want[6], "{name}: the copied row");
        drop(heap);
        release(dev);
    }
}

pub type ClearX = unsafe extern "system" fn(*mut c_void, u64, u64, *mut c_void, *const c_void, u32, *const c_void);

fn typed_srv(format: DXGI_FORMAT, first: u64, count: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC { Format: format, ..structured_srv(first, count, 0) }
}

fn typed_uav(format: DXGI_FORMAT, first: u64, count: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ..structured_uav(first, count, 0) }
}

#[test]
fn typed_views_of_the_title_read_and_write_their_elements_raw_and_their_rows_name_the_format() {
    // R16_UINT: elements of two bytes, element 1 inside the first dword of the PC view
    let formats = [(DXGI_FORMAT_R32G32B32A32_FLOAT, 16u32), (DXGI_FORMAT_R16G16_UNORM, 4), (DXGI_FORMAT_R8G8B8A8_UNORM, 4), (DXGI_FORMAT_R32_UINT, 4), (DXGI_FORMAT_R16_UINT, 2)];
    for (name, device) in devices() {
        println!("typed views on {name}");
        let dev = console_device(&device);
        let heap = ConsoleHeap::new(dev, &device, 10, true);
        let staging = ConsoleHeap::new(dev, &device, 2, false);
        let data = HostBuffer::new(&device, 64);
        data.fill(|i| 0x1000 + i as u32);
        for (k, &(format, size)) in formats.iter().enumerate() {
            // element 1 starts at byte `size`, the PC view on the 16 bytes at or below it
            let (lead, offset) = (size & !15, size % 16);
            let target = HostBuffer::new(&device, 64);
            target.fill(|_| 0);
            // elements 1 and 2 of the format, read through slot 2k and written through slot 2k + 1
            let (srv, uav) = (typed_srv(format, 1, 2), typed_uav(format, 1, 2));
            // SAFETY: console device slots 18 and 19 with live resources and slots of `heap`.
            unsafe {
                vt::<CreateSrvX>(dev, 18)(dev, data.res.as_raw(), (&raw const srv).cast(), heap.cpu_at(2 * k));
                vt::<CreateUavX>(dev, 19)(dev, target.res.as_raw(), std::ptr::null_mut(), (&raw const uav).cast(), heap.cpu_at(2 * k + 1));
            }
            let row = |slot: usize| crate::view_shadow::row(crate::title_heap::translate_cpu(heap.cpu_at(slot)).expect("a host slot"));
            let want_row = Some([size, 2, format.0 as u32, offset]);
            assert_eq!((row(2 * k), row(2 * k + 1)), (want_row, want_row), "{name}: format {}: the rows", format.0);
            let out = HostBuffer::new(&device, 64);
            let words = read_write(&device, &heap, 2 * k as u64, &out);
            // the dwords that hold the offset and the two elements
            let (first, n) = ((lead / 4) as usize, (2 * size + offset).div_ceil(4) as usize);
            let want: Vec<u32> = (first..first + n).map(|i| 0x1000 + i as u32).collect();
            assert_eq!(&words[..n], &want[..], "{name}: format {}: the raw view reads elements 1 and 2 from the row's offset on", format.0);
            let bytes = 4 * n as u32;
            assert_eq!((words[63], words[62]), (bytes, bytes), "{name}: format {}: the views hold two elements after the offset", format.0);
            let mut written = vec![0u32; 64];
            for (j, w) in written[first..first + n].iter_mut().enumerate() {
                *w = 0xA000 + j as u32;
            }
            assert_eq!(target.words(), written, "{name}: format {}: the raw unordered access view writes elements 1 and 2 only", format.0);
        }

        // Clears of typed views made raw: an R8G8B8A8_UNORM float clear and an R32G32B32A32_FLOAT one
        // of four equal values become uint clears of the raw views, of the title's elements only (the
        // first view starts 4 bytes after the start of the PC view).
        let target = HostBuffer::new(&device, 64);
        target.fill(|_| 0);
        let (unorm, wide) = (typed_uav(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 3), typed_uav(DXGI_FORMAT_R32G32B32A32_FLOAT, 2, 1));
        // SAFETY: console device slots 19 and 24 with a live resource and slots of the heaps, then
        // console list slots 12 and 54 on proxies made here, the list runs once and is waited for.
        unsafe {
            vt::<CreateUavX>(dev, 19)(dev, target.res.as_raw(), std::ptr::null_mut(), (&raw const unorm).cast(), staging.cpu_at(0));
            vt::<CreateUavX>(dev, 19)(dev, target.res.as_raw(), std::ptr::null_mut(), (&raw const wide).cast(), staging.cpu_at(1));
            vt::<CopySimpleX>(dev, 24)(dev, 2, heap.cpu_at(8), staging.cpu_at(0), 0);
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("an allocator");
            let mut list: *mut c_void = std::ptr::null_mut();
            let hr = vt::<CreateListX>(dev, 12)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, allocator.as_raw(), std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list);
            assert!(hr >= 0 && !list.is_null(), "{name}: console slot 12 makes the list");
            let pc_raw = *(list.cast::<u8>().add(8).cast::<*mut c_void>());
            let pc = ID3D12GraphicsCommandList::from_raw_borrowed(&pc_raw).expect("the PC list").clone();
            pc.SetDescriptorHeaps(&[Some(heap.pc.clone())]);
            let gpu = |slot: u64| heap.gpu + slot * heap.increment as u64;
            let colour = [1.0f32, 0.5, 0.0, 1.0];
            vt::<ClearX>(list, 54)(list, gpu(8), staging.cpu_at(0) as u64, target.res.as_raw(), colour.as_ptr().cast(), 0, std::ptr::null());
            let two = [2.0f32; 4];
            vt::<ClearX>(list, 54)(list, gpu(9), staging.cpu_at(1) as u64, target.res.as_raw(), two.as_ptr().cast(), 0, std::ptr::null());
            pc.Close().expect("the list closes");
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("a queue");
            queue.ExecuteCommandLists(&[Some(pc.cast::<ID3D12CommandList>().expect("a command list"))]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
            let event = CreateEventW(None, false, false, PCWSTR::null()).expect("an event");
            assert!(queue.Signal(&fence, 1).is_ok() && fence.SetEventOnCompletion(1, event).is_ok());
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0, "{name}: the clears finished");
            let _ = CloseHandle(event);
            assert!(device.GetDeviceRemovedReason().is_ok(), "{name}: the device is still there");
            drop(pc);
            release(list);
        }
        let mut want = vec![0u32; 64];
        want[1..4].fill(0xFF00_80FF);
        want[8..12].fill(0x4000_0000);
        assert_eq!(target.words(), want, "{name}: the clears wrote the title's values into elements of the views only");
        drop(heap);
        drop(staging);
        release(dev);
    }
}
