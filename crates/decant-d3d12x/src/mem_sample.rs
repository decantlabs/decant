//! `DECANT_MEM_SAMPLE=<seconds>`: a thread logs the GPU side of the memory every that many seconds (30
//! when the value is not a number), from the first PC device on: the pool heaps of the title memory
//! mirror (system memory, pinned), its mirror tiles and a rotating count of entirely-zero title pages,
//! window buffers, the upload memory of list slot
//! 148, the page table of the device memory window and its service buffers, and what DXGI counts for
//! the process in video memory (local) and in system memory (non local). A second line, emitted from
//! the same sample, names the layer's memory owners and the unaccounted local usage. `xmem.dll` logs
//! the process and its engine with the same switch.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::Once;
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
    IDXGIFactory4,
};
use windows::core::Interface;

use crate::log::xs_log;
use crate::mem_owners::{self, MIB};

/// Starts the sampler once for the first PC device, when `DECANT_MEM_SAMPLE` is set.
///
/// # Safety
/// `device` is a live PC `ID3D12Device`.
pub unsafe fn start(device: *mut c_void) {
    static STARTED: Once = Once::new();
    let Some(value) = std::env::var_os("DECANT_MEM_SAMPLE") else {
        return;
    };
    let every = Duration::from_secs(
        value
            .to_str()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&s| s > 0)
            .unwrap_or(30),
    );
    // SAFETY: by the contract above.
    let Some(device) = (unsafe { ID3D12Device::from_raw_borrowed(&device) }).cloned() else {
        return;
    };
    STARTED.call_once(|| {
        // SAFETY: plain DXGI calls, the LUID names the adapter of the live device.
        let adapter = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
        };
        let adapter = match adapter {
            Ok(a) => Some(a),
            Err(e) => {
                xs_log!(
                    "memory: no IDXGIAdapter3 for the device ({e}), video memory is not sampled"
                );
                None
            }
        };
        let _ = std::thread::Builder::new()
            .name("d3d12xs-memsample".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(every);
                    for line in lines(adapter.as_ref()) {
                        xs_log!("{}", line);
                    }
                }
            });
    });
}

fn video(adapter: &IDXGIAdapter3, local: bool) -> Option<DXGI_QUERY_VIDEO_MEMORY_INFO> {
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    let group = if local {
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL
    } else {
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL
    };
    // SAFETY: a live adapter and a valid output.
    unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }
        .ok()
        .map(|()| info)
}

fn lines(adapter: Option<&IDXGIAdapter3>) -> [String; 2] {
    let local = adapter.and_then(|a| video(a, true));
    let non_local = adapter.and_then(|a| video(a, false));
    [
        line(local.as_ref(), non_local.as_ref()),
        mem_owners::format(
            &mem_owners::snapshot().with_empty_view_heaps(crate::view_heap::empty_heaps()),
            local.as_ref().map(|v| v.CurrentUsage),
        ),
    ]
}

fn line(
    local: Option<&DXGI_QUERY_VIDEO_MEMORY_INFO>,
    non_local: Option<&DXGI_QUERY_VIDEO_MEMORY_INFO>,
) -> String {
    let mut out = String::from("memory (GPU side):");
    match crate::title_mem::sample() {
        Some(sample) => {
            out += &format!(
                " title memory mirror {} pool heaps ({} MiB pinned), {} mirror tiles ({} MiB), {} window buffers;",
                sample.heaps,
                sample.heaps as u64 * crate::title_mem::POOL_HEAP_BYTES / MIB,
                sample.tiles,
                u64::from(sample.tiles) * 64 / 1024,
                sample.windows
            );
            let zero = sample.zero_pages;
            let read = zero.zero + zero.non_zero;
            out += &format!(
                " zero title pages {} of {read} read mirror tiles ({} MiB), {} non-zero, {} unreadable, {} unobserved; rotating scan {} tiles/sample ({} this sample);",
                zero.zero,
                zero.zero * 64 / 1024,
                zero.non_zero,
                zero.unreadable,
                zero.unknown,
                zero.scan_tiles,
                zero.scanned
            );
            if let Some(sweep) = zero.last_sweep {
                out += &format!(
                    " last full zero-page sweep {} of {} tiles zero ({} MiB), {} unreadable;",
                    sweep.zero,
                    sweep.tiles,
                    sweep.zero * 64 / 1024,
                    sweep.unreadable
                );
            }
        }
        None => out += " title memory mirror none;",
    }
    out += &format!(
        " upload memory {} MiB;",
        crate::upload_mem::with(|m| m.bytes()) as u64 / MIB
    );
    let (table, bits) = crate::service_mem::bytes();
    out += &format!(
        " page table {} MiB (upload), service buffers {} MiB;",
        table / MIB,
        bits / MIB
    );
    for (info, name) in [(local, "local"), (non_local, "non local")] {
        if let Some(v) = info {
            out += &format!(
                " video memory {name} {} MiB of a budget of {} MiB;",
                v.CurrentUsage / MIB,
                v.Budget / MIB
            );
        }
    }
    out
}
