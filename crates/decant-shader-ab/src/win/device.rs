//! A D3D12 device for short GPU tests: the default adapter, WARP as fallback, buffers, barriers and
//! a submission that waits with a deadline and notices device removal.

use std::ffi::c_void;
use std::time::Duration;

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{IUnknown, Interface};

/// A device and its direct queue.
pub struct Gpu {
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    /// The adapter's description.
    pub name: String,
    /// The highest shader model the device runs, as `(major, minor)`.
    pub model: (u32, u32),
}

fn adapter_name(device: &ID3D12Device) -> String {
    // SAFETY: plain queries on a live device.
    unsafe {
        let luid = device.GetAdapterLuid();
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else { return "unknown adapter".into() };
        match factory.EnumAdapterByLuid::<IDXGIAdapter>(luid) {
            Ok(a) => {
                let d = a.GetDesc().unwrap_or_default();
                String::from_utf16_lossy(&d.Description).trim_end_matches('\0').to_owned()
            }
            Err(_) => "unknown adapter".into(),
        }
    }
}

fn highest_model(device: &ID3D12Device) -> (u32, u32) {
    for m in [0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60] {
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL(m) };
        // SAFETY: the structure matches the feature.
        let ok = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &mut sm as *mut _ as *mut c_void, std::mem::size_of_val(&sm) as u32).is_ok() };
        if ok {
            let v = sm.HighestShaderModel.0 as u32;
            return (v >> 4, v & 0xF);
        }
    }
    (5, 1)
}

fn make_gpu(device: ID3D12Device) -> Option<Gpu> {
    let model = highest_model(&device);
    if model.0 < 6 {
        return None;
    }
    let desc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
    // SAFETY: valid description.
    let queue = unsafe { device.CreateCommandQueue::<ID3D12CommandQueue>(&desc).ok()? };
    let name = adapter_name(&device);
    Some(Gpu { device, queue, name, model })
}

/// The default adapter, else WARP. `DECANT_SHADER_DEVICE=warp` forces WARP, `DECANT_SHADER_DEVICE=hardware`
/// forbids it. `DECANT_SHADER_AB_DEBUG=1` turns the D3D12 debug layer on, whose messages then come with
/// every failure, `2` adds GPU based validation. `None` when no device runs Shader Model 6.0.
pub fn open_device() -> Option<Gpu> {
    open_device_kind(&std::env::var("DECANT_SHADER_DEVICE").unwrap_or_default().to_lowercase())
}

/// As [`open_device`] with the choice given: `warp`, `hardware`, or anything else for the default.
pub fn open_device_kind(want: &str) -> Option<Gpu> {
    // SAFETY: documented D3D12 and DXGI entry points with valid arguments.
    unsafe {
        if debug_layer() {
            // DRED: the breadcrumbs and the page fault of a removal come with the error.
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(d) = dred
            {
                d.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                d.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
            let mut debug: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut debug).is_ok()
                && let Some(d) = debug
            {
                d.EnableDebugLayer();
                if std::env::var("DECANT_SHADER_AB_DEBUG").is_ok_and(|v| v == "2")
                    && let Ok(d1) = d.cast::<ID3D12Debug1>()
                {
                    d1.SetEnableGPUBasedValidation(true);
                }
            }
        }
        if want != "warp" {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None::<&IUnknown>, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok()
                && let Some(g) = dev.and_then(make_gpu)
            {
                return Some(g);
            }
        }
        if want != "hardware" {
            let factory = CreateDXGIFactory1::<IDXGIFactory4>().ok()?;
            let warp = factory.EnumWarpAdapter::<IDXGIAdapter>().ok()?;
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_ok() {
                return dev.and_then(make_gpu);
            }
        }
    }
    None
}

/// A buffer of at least `size` bytes (256 at least) on `heap`, in `state`.
pub fn buffer(device: &ID3D12Device, heap: D3D12_HEAP_TYPE, size: usize, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    let props = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: valid descriptions, the result is checked.
    unsafe { device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res) }.map_err(|e| format!("CreateCommittedResource buffer: {e}"))?;
    res.ok_or_else(|| "no buffer".into())
}

/// An upload buffer holding `data`.
pub fn upload(device: &ID3D12Device, data: &[u8]) -> Result<ID3D12Resource, String> {
    let r = buffer(device, D3D12_HEAP_TYPE_UPLOAD, data.len(), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
    // SAFETY: the upload buffer is at least `data.len()` bytes and mapped for the copy only.
    unsafe {
        let mut p: *mut c_void = std::ptr::null_mut();
        r.Map(0, None, Some(&mut p)).map_err(|e| format!("Map: {e}"))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u8>(), data.len());
        r.Unmap(0, None);
    }
    Ok(r)
}

/// The first `len` bytes of a mapped readback buffer.
pub fn read_back(res: &ID3D12Resource, len: usize) -> Result<Vec<u8>, String> {
    // SAFETY: the buffer is a readback buffer of at least `len` bytes, the GPU has finished with it.
    unsafe {
        let mut p: *mut c_void = std::ptr::null_mut();
        res.Map(0, None, Some(&mut p)).map_err(|e| format!("Map readback: {e}"))?;
        let out = std::slice::from_raw_parts(p.cast::<u8>(), len).to_vec();
        res.Unmap(0, None);
        Ok(out)
    }
}

/// A transition barrier of every subresource of `res`.
pub fn transition(res: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed copy of the interface pointer, never released (ManuallyDrop).
                pResource: unsafe { std::mem::transmute_copy(res) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
    }
}

/// A UAV barrier on every UAV access.
pub fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }) },
    }
}

fn debug_layer() -> bool {
    std::env::var("DECANT_SHADER_AB_DEBUG").is_ok_and(|v| v == "1" || v == "2")
}

impl Gpu {
    /// The messages the debug layer stored since the last call (errors and corruption only), empty
    /// without the debug layer.
    pub fn messages(&self) -> Vec<String> {
        let Ok(q) = self.device.cast::<ID3D12InfoQueue>() else { return Vec::new() };
        let mut out = Vec::new();
        // SAFETY: each message is read into a buffer of the length the queue asked for.
        unsafe {
            for i in 0..q.GetNumStoredMessages() {
                let mut len = 0usize;
                if q.GetMessage(i, None, &mut len).is_err() || len < std::mem::size_of::<D3D12_MESSAGE>() {
                    continue;
                }
                let mut buf = vec![0u64; len.div_ceil(8)];
                let m = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if q.GetMessage(i, Some(m), &mut len).is_ok() && (*m).Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && !(*m).pDescription.is_null() {
                    let text = std::slice::from_raw_parts((*m).pDescription, (*m).DescriptionByteLength.saturating_sub(1));
                    out.push(String::from_utf8_lossy(text).into_owned());
                }
            }
            q.ClearStoredMessages();
        }
        out
    }

    /// Whether the device was removed, with the reason, and what DRED recorded when it is on.
    pub fn removed(&self) -> Option<String> {
        // SAFETY: a plain query.
        let e = unsafe { self.device.GetDeviceRemovedReason() }.err()?;
        Some(format!("device removed: {e}{}", self.dred()))
    }

    /// The last breadcrumbs of every list and the page fault DRED recorded, empty without DRED.
    fn dred(&self) -> String {
        let Ok(d) = self.device.cast::<ID3D12DeviceRemovedExtendedData>() else { return String::new() };
        let mut out = String::new();
        // SAFETY: DRED's output lives as long as the device. Every pointer is checked before use.
        unsafe {
            if let Ok(pf) = d.GetPageFaultAllocationOutput()
                && pf.PageFaultVA != 0
            {
                out.push_str(&format!(" [page fault at 0x{:x}]", pf.PageFaultVA));
            }
            if let Ok(b) = d.GetAutoBreadcrumbsOutput() {
                let mut node = b.pHeadAutoBreadcrumbNode;
                while !node.is_null() {
                    let n = &*node;
                    let done = if n.pLastBreadcrumbValue.is_null() { 0 } else { *n.pLastBreadcrumbValue };
                    let ops: Vec<String> = (0..n.BreadcrumbCount).map(|i| format!("{}{:?}", if i == done { ">" } else { "" }, (*n.pCommandHistory.add(i as usize)).0)).collect();
                    out.push_str(&format!(" [breadcrumbs {done}/{}: {}]", n.BreadcrumbCount, ops.join(" ")));
                    node = n.pNext;
                }
            }
        }
        out
    }

    /// Closes `list`, runs it and waits up to `deadline`. A timeout or a removed device is an error.
    pub fn submit(&self, list: &ID3D12GraphicsCommandList, deadline: Duration) -> Result<(), String> {
        // SAFETY: the list is recorded and every resource it uses outlives this call.
        unsafe {
            list.Close().map_err(|e| format!("Close: {e}{}", self.removed().map(|r| format!(" ({r})")).unwrap_or_default()))?;
            self.queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().map_err(|e| e.to_string())?)]);
            let fence = self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).map_err(|e| e.to_string())?;
            let event: HANDLE = CreateEventW(None, false, false, None).map_err(|e| e.to_string())?;
            let waited = self.queue.Signal(&fence, 1).and_then(|_| fence.SetEventOnCompletion(1, event)).map(|_| WaitForSingleObject(event, deadline.as_millis() as u32));
            let _ = CloseHandle(event);
            if let Some(r) = self.removed() {
                return Err(r);
            }
            // A removed device signals its fences with the largest value, before it reports itself.
            if fence.GetCompletedValue() == u64::MAX {
                return Err("device removed (its fence reads all ones)".into());
            }
            match waited {
                Ok(w) if w == WAIT_OBJECT_0 => Ok(()),
                Ok(_) => Err(format!("the GPU did not finish within {} seconds", deadline.as_secs())),
                Err(e) => Err(format!("fence: {e}")),
            }
        }
    }
}
