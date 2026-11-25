//! Why the PC device was removed. Before the device is created, `DECANT_D3D12X_DEBUG=1` turns on the
//! D3D12 debug layer (its messages reach the debugger, so the decant-launch log), and DRED (device
//! removed extended data: automatic breadcrumbs and page fault data) is on unless
//! `DECANT_D3D12X_DRED=0`. After creation a watcher thread checks the removal reason every 5 ms
//! and, once the device is gone, logs the reason, the last command lists with the operation each
//! stopped at, and the page fault address with the allocations near it.

#![allow(unsafe_code)]

use crate::log::xs_log;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

fn switch(name: &str, default: bool) -> bool {
    std::env::var_os(name).map_or(default, |v| v == "1")
}

/// Turns the debug layer and DRED on as asked. Called before the PC device is created.
pub fn before_device() {
    // SAFETY: the documented calls, with interface pointers the runtime hands out.
    unsafe {
        if switch("DECANT_D3D12X_DEBUG", false) {
            let mut debug: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut debug) {
                Ok(()) => {
                    if let Some(d) = debug {
                        d.EnableDebugLayer();
                        if let Ok(d1) = d.cast::<ID3D12Debug1>() {
                            d1.SetEnableGPUBasedValidation(switch("DECANT_D3D12X_GBV", false));
                        }
                        xs_log!("diagnose: D3D12 debug layer on (DECANT_D3D12X_DEBUG=1)");
                    }
                }
                Err(e) => xs_log!("diagnose: no D3D12 debug layer ({e}), the Graphics Tools are not installed"),
            }
        }
        if switch("DECANT_D3D12X_DRED", true) {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(s) = dred
            {
                s.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                s.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                xs_log!("diagnose: DRED breadcrumbs and page faults on (DECANT_D3D12X_DRED=0 turns them off)");
            }
        }
    }
}

/// The device [`watch`] looks at, with one reference of its own that is never given back.
static WATCHED: OnceLock<usize> = OnceLock::new();
/// Set once a call of [`removed_after`] saw the device removed.
static SEEN_REMOVED: AtomicBool = AtomicBool::new(false);

/// Whether the device is checked after the layer's own GPU calls (on, `DECANT_D3D12X_REMOVAL_CHECK=0` turns it off).
fn removal_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var_os("DECANT_D3D12X_REMOVAL_CHECK").is_some_and(|v| v == "0"))
}

/// Unless `DECANT_D3D12X_REMOVAL_CHECK=0`: checks right after a call that can remove the device whether it
/// was removed, and logs the first call after which it was. The runtime removes the device inside the
/// call that it refuses, so this names the call (or one racing with it on another thread).
pub fn removed_after(what: impl FnOnce() -> String) {
    if !removal_check() || SEEN_REMOVED.load(Ordering::Relaxed) {
        return;
    }
    let Some(&raw) = WATCHED.get() else { return };
    let raw = raw as *mut c_void;
    // SAFETY: the watched device holds a reference that is never given back.
    let Some(device) = (unsafe { ID3D12Device::from_raw_borrowed(&raw) }) else { return };
    // SAFETY: a plain query on a live device.
    if unsafe { device.GetDeviceRemovedReason() }.is_err() && !SEEN_REMOVED.swap(true, Ordering::Relaxed) {
        xs_log!("diagnose: the device was found removed right after {} (thread {})", what(), unsafe { windows::Win32::System::Threading::GetCurrentThreadId() });
    }
}

/// Starts the watcher for the PC device `device` (a reference is taken for the thread).
pub fn watch(device: *mut c_void) {
    // SAFETY: `device` is a live PC device; the clone takes the thread's own reference.
    let Some(device) = (unsafe { ID3D12Device::from_raw_borrowed(&device) }).cloned() else { return };
    let _ = WATCHED.set(device.clone().into_raw() as usize);
    struct Owned(ID3D12Device);
    // SAFETY: D3D12 devices are free threaded.
    unsafe impl Send for Owned {}
    let owned = Owned(device);
    let _ = std::thread::Builder::new().name("d3d12xs-removal-watch".into()).spawn(move || {
        let owned = owned;
        loop {
            // Often enough to report before the title traps on its next failed call (a retail
            // title trapped 50 ms after the removal).
            std::thread::sleep(std::time::Duration::from_millis(5));
            // SAFETY: a plain query on a live device.
            let reason = unsafe { owned.0.GetDeviceRemovedReason() };
            if let Err(e) = reason {
                report(&owned.0, e.code().0 as u32);
                return;
            }
        }
    });
}

/// The name of a breadcrumb operation (`D3D12_AUTO_BREADCRUMB_OP`, d3d12.h), its number for the rest.
fn op_name(op: D3D12_AUTO_BREADCRUMB_OP) -> String {
    const NAMES: [&str; 21] = [
        "SetMarker", "BeginEvent", "EndEvent", "Draw", "DrawIndexed", "ExecuteIndirect", "Dispatch", "CopyBufferRegion",
        "CopyTextureRegion", "CopyResource", "CopyTiles", "ResolveSubresource", "ClearRenderTargetView",
        "ClearUnorderedAccessView", "ClearDepthStencilView", "ResourceBarrier", "ExecuteBundle", "Present",
        "ResolveQueryData", "BeginSubmission", "EndSubmission",
    ];
    match op.0 {
        n @ 0..=20 => NAMES[n as usize].to_string(),
        26 => "WriteBufferImmediate".into(),
        39 => "SetPipelineState1".into(),
        45 => "Barrier".into(),
        46 => "BeginCommandList".into(),
        n => format!("{n}"),
    }
}

/// Formats one faulting command list line for DRED diagnostics.
fn format_fault_list(
    list: Option<usize>,
    name: &str,
    queue: Option<usize>,
    done: u32,
    total: u32,
    around: &[String],
    active_pso: Option<&crate::pso_diag::ActivePso>,
) -> String {
    let list_ptr = list.map(|p| p as *const c_void);
    let queue_ptr = queue.map(|p| p as *const c_void);
    let base = format!(
        "list {list_ptr:?} {name:?} on queue {queue_ptr:?}: stopped at operation {done} of {total} (ops around it: {})",
        around.join(" ")
    );
    match active_pso {
        Some(pso) => format!("{base}, active {}", pso.format_summary()),
        None => base,
    }
}

fn report(device: &ID3D12Device, reason: u32) {
    xs_log!("diagnose: the PC device was removed, reason 0x{reason:08X}");
    // SAFETY: DRED queries on the removed device; the returned structures live as long as it does.
    unsafe {
        let Ok(dred) = device.cast::<ID3D12DeviceRemovedExtendedData>() else {
            xs_log!("diagnose: no DRED data (was it on before the device was created?)");
            return;
        };
        if let Ok(crumbs) = dred.GetAutoBreadcrumbsOutput() {
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            let mut shown = 0;
            // The layer's own lists (named "decant ..."), the last ones of them, finished or not.
            let mut ours: std::collections::VecDeque<String> = std::collections::VecDeque::new();
            let mut nodes = 0usize;
            while !node.is_null() {
                let n = &*node;
                nodes += 1;
                let done = if n.pLastBreadcrumbValue.is_null() { 0 } else { *n.pLastBreadcrumbValue };
                let name = if n.pCommandListDebugNameW.is_null() { String::new() } else { n.pCommandListDebugNameW.to_string().unwrap_or_default() };
                if name.starts_with("decant ") {
                    if ours.len() == 8 {
                        ours.pop_front();
                    }
                    ours.push_back(format!("{name} on queue {:?}: {done} of {} operations done", n.pCommandQueue.as_ref().map(|q| q.as_raw()), n.BreadcrumbCount));
                }
                if done < n.BreadcrumbCount && shown < 24 {
                    let ops = std::slice::from_raw_parts(n.pCommandHistory, n.BreadcrumbCount as usize);
                    let from = (done as usize).saturating_sub(4);
                    let to = (done as usize + 4).min(ops.len());
                    let around: Vec<String> = ops[from..to].iter().enumerate().map(|(k, op)| {
                        let at = from + k;
                        format!("{}{}", if at == done as usize { ">" } else { "" }, op_name(*op))
                    }).collect();
                    let list_ptr = n.pCommandList.as_ref().map(|l| l.as_raw() as usize);
                    let active = list_ptr.and_then(crate::pso_diag::active_pso);
                    let line = format_fault_list(
                        list_ptr,
                        &name,
                        n.pCommandQueue.as_ref().map(|q| q.as_raw() as usize),
                        done,
                        n.BreadcrumbCount,
                        &around,
                        active.as_ref(),
                    );
                    xs_log!("diagnose: {line}");
                    shown += 1;
                }
                node = n.pNext;
            }
            if shown == 0 {
                xs_log!("diagnose: every recorded list finished its operations (the fault is outside the breadcrumbs)");
            }
            xs_log!("diagnose: {nodes} lists in the breadcrumbs, the last {} of the layer's own:", ours.len());
            for line in ours {
                xs_log!("diagnose:   {line}");
            }
        }
        if let Ok(fault) = dred.GetPageFaultAllocationOutput() {
            xs_log!("diagnose: page fault at GPU VA 0x{:X}", fault.PageFaultVA);
            for (label, mut node) in [("existing", fault.pHeadExistingAllocationNode), ("recently freed", fault.pHeadRecentFreedAllocationNode)] {
                let (mut k, mut named) = (0, 0);
                while !node.is_null() {
                    let n = &*node;
                    let name = if n.ObjectNameW.is_null() { String::new() } else { n.ObjectNameW.to_string().unwrap_or_default() };
                    // the first allocations, and every one the layer named, up to 64
                    if k < 16 || (name.starts_with("decant ") && named < 64) {
                        named += usize::from(name.starts_with("decant "));
                        xs_log!("diagnose:   {label} allocation {k} type {} {name}", n.AllocationType.0);
                    }
                    node = n.pNext;
                    k += 1;
                }
                xs_log!("diagnose:   {k} {label} allocations in all");
            }
            let (released, held_back, held) = crate::retire::counts();
            xs_log!("diagnose: {released} objects released by the layer, {held_back} of them held back for work in flight, {held} held now");
            for f in crate::retire::near(fault.PageFaultVA) {
                xs_log!("diagnose:   freed at {:.3} s by the layer: {} (GPU {:#x} to {:#x})", f.at_ms as f64 / 1000.0, f.label, f.va, f.va + f.bytes);
            }
        }
        xs_log!("diagnose: {}", crate::detile::summary());
        let (heaps, placed) = crate::view_heap::usage();
        xs_log!("diagnose: view texture heaps {heaps} with {} MiB placed", placed >> 20);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pso_diag::ActivePso;

    #[test]
    fn format_fault_list_includes_active_pipeline_and_shader_hashes() {
        let around = vec![
            op_name(D3D12_AUTO_BREADCRUMB_OP(15)),
            format!(">{}", op_name(D3D12_AUTO_BREADCRUMB_OP(5))),
        ];
        let active = ActivePso {
            number: Some(42),
            pso: 0x5555,
            shaders: String::from("VS 1111222233334444 hit, PS 5555666677778888 miss"),
            missing: false,
        };

        let line = format_fault_list(
            Some(0x1000),
            "main",
            Some(0x2000),
            12,
            20,
            &around,
            Some(&active),
        );

        assert!(line.contains("stopped at operation 12 of 20"));
        assert!(line.contains("ResourceBarrier >ExecuteIndirect"));
        assert!(line.contains("active pipeline 42 [VS 1111222233334444 hit, PS 5555666677778888 miss]"));
    }

    #[test]
    fn format_fault_list_without_active_pipeline_omits_suffix() {
        let around = vec![format!(">{}", op_name(D3D12_AUTO_BREADCRUMB_OP(6)))];
        let line = format_fault_list(Some(0x3000), "", Some(0x4000), 0, 1, &around, None);

        assert!(line.contains("stopped at operation 0 of 1 (ops around it: >Dispatch)"));
        assert!(!line.contains("active"));
    }
}
