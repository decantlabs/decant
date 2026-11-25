//! A zeroed GPU buffer for addresses no PC resource holds. Root views and index or vertex views
//! that point into title memory the PC GPU cannot see would be invalid calls, and native D3D12
//! removes the device for them; this buffer is bound instead, so the draw reads zeros.

#![allow(unsafe_code)]

use crate::log::xs_log;
use std::ffi::c_void;
use crate::device_cache::PerDevice;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::core::Interface;

/// Size of the buffer: large enough for any constant buffer (64 KiB) and small reads.
pub const BYTES: u64 = 1 << 20;

/// The GPU address of the zeroed buffer of the device of PC command list `list`, made on first use for each
/// device (a buffer is no address on another device). 0 when it cannot be made.
///
/// # Safety
/// `list` is a live PC `ID3D12GraphicsCommandList`.
pub unsafe fn address(list: *mut c_void) -> u64 {
    static VA: PerDevice<u64> = PerDevice::new();
    // SAFETY: by the contract above.
    let Some(device) = (unsafe { device_of(list) }) else { return 0 };
    VA.get_or_make(device.as_raw() as usize, || {
        // SAFETY: by the contract above; the documented creation calls.
        let made = unsafe {
            (|| -> windows::core::Result<u64> {
                let dev = device.clone();
                let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: BYTES,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    Alignment: 0,
                };
                let mut res: Option<ID3D12Resource> = None;
                // Committed default heap memory starts zeroed. COMMON: buffers promote to any
                // read state and back on their own.
                dev.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res)?;
                let res = res.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
                let va = res.GetGPUVirtualAddress();
                crate::mem_owners::add(
                    crate::mem_owners::Kind::CommittedOther,
                    crate::mem_owners::Segment::Local,
                    1,
                    BYTES,
                );
                // The buffer lives for the process.
                std::mem::forget(res);
                Ok(va)
            })()
        };
        match made {
            Ok(va) => {
                xs_log!("fallback: a zeroed {BYTES} byte buffer at GPU VA {va:#x} of device {:#x} stands in for title addresses the PC GPU cannot see", device.as_raw() as usize);
                va
            }
            Err(e) => {
                xs_log!("fallback: no zeroed buffer ({e})");
                0
            }
        }
    })
}

/// The device of PC command list `list`.
///
/// # Safety
/// `list` is a live PC `ID3D12GraphicsCommandList`.
unsafe fn device_of(list: *mut c_void) -> Option<ID3D12Device> {
    // SAFETY: by the contract above.
    unsafe {
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&list)?;
        let mut dev: Option<ID3D12Device> = None;
        list.GetDevice(&mut dev).ok()?;
        dev
    }
}
