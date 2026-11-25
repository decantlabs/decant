//! `DECANT_MEM_SAMPLE=<seconds>`: a thread logs where the commit of the process goes, every that many
//! seconds (30 when the value is not a number), from the first use of the engine on. One line holds
//! the system commit (and how much of it the process added since the engine started), the process
//! counters, the committed bytes of the address space by kind, the process heaps and the engine's own
//! share. `d3d12_xs.dll` logs the GPU side with the same switch.
//!
//! The mapped bytes of the address space count each view: a pool page the title maps at two addresses,
//! and the engine's two private views of every section, count once per view. The engine's line says
//! what its sections really charge.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::Once;
use std::time::Duration;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Memory::{
    GetProcessHeaps, HEAP_SUMMARY, HeapSummary, MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE,
    MEMORY_BASIC_INFORMATION, VirtualQuery,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::engine::Engine;

const MIB: usize = 1 << 20;

/// The interval of `DECANT_MEM_SAMPLE`, `None` when it is not set.
pub fn interval() -> Option<Duration> {
    let value = std::env::var("DECANT_MEM_SAMPLE").ok()?;
    let seconds = value
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&s| s > 0)
        .unwrap_or(30);
    Some(Duration::from_secs(seconds))
}

/// Starts the sampler once, when `DECANT_MEM_SAMPLE` is set.
pub fn start() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let Some(every) = interval() else { return };
        let baseline = performance().map_or(0, |p| p.CommitTotal * p.PageSize);
        let _ = std::thread::Builder::new()
            .name("xmem-memsample".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(every);
                    decant_log::trace(&format!("{}\r\n", line(baseline)));
                }
            });
    });
}

fn performance() -> Option<PERFORMANCE_INFORMATION> {
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: `info` is a valid output buffer of the size its `cb` says.
    (unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0).then_some(info)
}

/// Committed bytes of the address space: private, mapped (each view counted) and image.
fn address_space() -> (usize, usize, usize) {
    let (mut at, mut private, mut mapped, mut image) = (0usize, 0, 0, 0);
    loop {
        // SAFETY: MEMORY_BASIC_INFORMATION is plain data.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid output buffer of the size passed. Querying changes nothing.
        if unsafe {
            VirtualQuery(
                at as *const c_void,
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        } == 0
        {
            break;
        }
        if info.State == MEM_COMMIT {
            match info.Type {
                MEM_PRIVATE => private += info.RegionSize,
                MEM_MAPPED => mapped += info.RegionSize,
                MEM_IMAGE => image += info.RegionSize,
                _ => {}
            }
        }
        match (info.BaseAddress as usize).checked_add(info.RegionSize) {
            Some(next) if next > at => at = next,
            _ => break,
        }
    }
    (private, mapped, image)
}

/// Committed bytes of the process heaps (ours, the C runtimes', DXC's and the drivers').
fn heaps() -> usize {
    let mut handles: Vec<HANDLE> = vec![std::ptr::null_mut(); 256];
    // SAFETY: `handles` has room for its length.
    let n = unsafe { GetProcessHeaps(handles.len() as u32, handles.as_mut_ptr()) } as usize;
    handles.truncate(n.min(handles.len()));
    handles
        .iter()
        .map(|&h| {
            let mut summary = HEAP_SUMMARY {
                cb: size_of::<HEAP_SUMMARY>() as u32,
                ..Default::default()
            };
            // SAFETY: `h` is a heap of this process, `summary` a valid output buffer.
            if unsafe { HeapSummary(h, 0, &mut summary) } != 0 {
                summary.cbCommitted
            } else {
                0
            }
        })
        .sum()
}

fn line(baseline: usize) -> String {
    let mut out = String::from("[xmem] memory:");
    if let Some(p) = performance() {
        let (total, limit) = (p.CommitTotal * p.PageSize, p.CommitLimit * p.PageSize);
        out += &format!(
            " system commit {} of {} MiB ({} MiB free, {} MiB more than at the engine start), RAM free {} of {} MiB;",
            total / MIB,
            limit / MIB,
            limit.saturating_sub(total) / MIB,
            total as i64 / MIB as i64 - baseline as i64 / MIB as i64,
            p.PhysicalAvailable * p.PageSize / MIB,
            p.PhysicalTotal * p.PageSize / MIB
        );
    }
    let mut counters = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: `counters` is a valid output buffer of the size its `cb` says.
    if unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
            counters.cb,
        )
    } != 0
    {
        out += &format!(
            " process private {} MiB, working set {} MiB;",
            counters.PrivateUsage / MIB,
            counters.WorkingSetSize / MIB
        );
    }
    let (private, mapped, image) = address_space();
    out += &format!(
        " address space committed: private {} MiB, mapped {} MiB (each view), image {} MiB; heaps {} MiB;",
        private / MIB,
        mapped / MIB,
        image / MIB,
        heaps() / MIB
    );
    if let Some(engine) = Engine::started() {
        let f = engine.footprint();
        out += &format!(
            " engine: {} sections of {} MiB, {} MiB of them committed, pool {} pages, {} used, {} mapped, arenas {} MiB with {} pages given to the GPU, private in reservations {} MiB",
            f.sections,
            f.section_bytes / MIB,
            f.committed_bytes / MIB,
            f.pool_pages,
            f.used_pages,
            f.mapped_pages,
            f.arena_bytes / MIB,
            f.given_pages,
            f.private_bytes / MIB
        );
    }
    out
}
