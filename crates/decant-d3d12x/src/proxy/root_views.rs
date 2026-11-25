//! Root views (console list slots 41 to 46: `SetCompute/GraphicsRootConstantBufferView`,
//! `ShaderResourceView`, `UnorderedAccessView`): where the PC address of a title address comes from,
//! counted per kind and source, and every address that still gets the zeroed fallback buffer named.
//!
//! The title passes GPU addresses. A translated shader reads a root SRV as a raw buffer at the PC
//! root parameter with the register and space of the title's root signature (measured: 10,287
//! programs read `t0` and `t4` of space 3 so, a header dword and data at computed offsets), so the
//! PC root view must point at the title's bytes:
//! * upload memory of list slot 148: its upload buffer,
//! * title memory (`title_mem`, `DECANT_D3D12X_TITLE_MEM=1`): the window buffer that mirrors it; for
//!   an SRV or a UAV every page up to the end of the title's reservation is given to the GPU, as the
//!   shader reads at offsets it takes from the data, and the bytes are current at each submission
//!   (`title_mem::sync` in `ExecuteCommandLists`),
//! * a buffer placed at the address (device slots 44, 133): that PC resource,
//! * else the zeroed fallback buffer, and the draw reads zeros (degenerate triangles for a mesh
//!   shader that reads its vertices so).
//!
//! Each fallback is counted in `DECANT_CALLS` and each new 64 KiB page of fallback addresses is logged
//! with what Windows knows of the memory there (allocation base, size, state, type, module), so the
//! allocation can be named. The first addresses of each source are logged with their PC address, and
//! every 4096 root views a summary gives the counts.

use std::sync::atomic::AtomicU64;

use super::*;

/// Where the PC address of a root view comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Source {
    /// Address 0: nothing bound.
    Null,
    /// Upload memory of list slot 148.
    Upload,
    /// Title memory in the window of `title_mem`.
    TitleMemory,
    /// A buffer placed at the address (device slots 44, 133).
    Placed,
    /// No PC resource: the zeroed fallback buffer.
    Fallback,
}

impl Source {
    const ALL: [Source; 5] = [Source::Null, Source::Upload, Source::TitleMemory, Source::Placed, Source::Fallback];

    fn name(self) -> &'static str {
        match self {
            Source::Null => "null",
            Source::Upload => "upload memory",
            Source::TitleMemory => "title memory",
            Source::Placed => "placed buffer",
            Source::Fallback => "zero buffer",
        }
    }
}

/// The PC GPU address of the title GPU address `va` (`len` bytes read) and where it comes from,
/// `None` when no PC resource holds it.
pub(super) fn pc_address(va: u64, len: u64) -> Option<(u64, Source)> {
    let up = crate::upload_mem::translate(va);
    if up != va {
        return Some((up, Source::Upload));
    }
    if let Some(span) = crate::title_mem::translate(va, len) {
        return Some((span.gpu, Source::TitleMemory));
    }
    let (res, offset) = crate::views_at::lookup_span(va, len)?;
    // SAFETY: the table holds live PC resources (forgotten when they are released).
    let base = unsafe { method::<GpuVirtualAddress>(res as Object, 11)(res as Object) };
    (base != 0).then_some((base + offset, Source::Placed))
}

/// The root view kind of PC list slot `pc_slot` (37 to 42).
fn kind_of(pc_slot: usize) -> usize {
    match pc_slot {
        37 | 38 => 0,
        39 | 40 => 1,
        _ => 2,
    }
}

const KINDS: [&str; 3] = ["CBV", "SRV", "UAV"];

/// Root views by kind and source.
static COUNTS: [[AtomicU64; 5]; 3] = [const { [const { AtomicU64::new(0) }; 5] }; 3];
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// The 64 KiB pages of fallback addresses logged so far, and the first addresses of each source.
static PAGES: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
static FIRST: [AtomicI32; 5] = [const { AtomicI32::new(0) }; 5];
/// The most distinct fallback pages that are described.
const MAX_PAGES: usize = 256;

/// The PC address for a root view the title sets at PC list slot `pc_slot` to title address `va`,
/// with the zeroed fallback buffer for an address no PC resource holds (an address no PC resource
/// holds is an invalid call on native D3D12, which removes the device). 0 stays 0.
///
/// # Safety
/// `list` is a live PC command list.
pub(super) unsafe fn bind_address(list: Object, pc_slot: usize, va: u64) -> u64 {
    let (pc_va, source) = if va == 0 {
        (0, Source::Null)
    } else {
        match pc_address(va, 4) {
            Some(found) => found,
            // SAFETY: by the contract above.
            None => (unsafe { crate::fallback::address(list) }, Source::Fallback),
        }
    };
    let kind = kind_of(pc_slot);
    if source == Source::TitleMemory && kind != 0 {
        give_to_reservation_end(va);
    }
    COUNTS[kind][source as usize].fetch_add(1, Ordering::Relaxed);
    if source == Source::Fallback {
        decant_log::calls::record_with(decant_log::calls::Kind::Stub, || format!("console list root {} at a title address in no PC resource (zero buffer bound)", KINDS[kind]));
        note_fallback(kind, va);
    } else if source != Source::Null && FIRST[source as usize].fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("console list root {} at title address {va:#x}: {} at PC address {pc_va:#x}", KINDS[kind], source.name());
    }
    if TOTAL.fetch_add(1, Ordering::Relaxed).wrapping_add(1).is_multiple_of(4096) {
        xs_log!("console list root views so far: {}", summary());
    }
    pc_va
}

/// A root SRV or UAV has no size, and the shader reads at offsets it takes from the data (measured: a
/// header dword names them), so every page from `va` to the end of the title's reservation must
/// reach the GPU, not only the first `title_mem` reach: the pages of a commit that come on demand are
/// given (`DecantXmemGpuUse` through `title_mem::translate`), and `title_mem::sync` copies their bytes at
/// the submission. Pages given once are not asked for again.
fn give_to_reservation_end(va: u64) {
    if let Some(span) = crate::title_mem::lookup(va, 4) {
        let _ = crate::title_mem::translate(va, 4 + span.left);
    }
}

/// The counts by kind and source, one line.
pub(super) fn summary() -> String {
    let mut parts = Vec::new();
    for (k, name) in KINDS.iter().enumerate() {
        let by: Vec<String> = Source::ALL.iter().map(|s| (s, COUNTS[k][*s as usize].load(Ordering::Relaxed))).filter(|(_, n)| *n > 0).map(|(s, n)| format!("{} {n}", s.name())).collect();
        if !by.is_empty() {
            parts.push(format!("{name}: {}", by.join(", ")));
        }
    }
    parts.join("; ")
}

/// Logs a fallback address the first time its 64 KiB page is seen (up to [`MAX_PAGES`] pages),
/// with what Windows knows of the memory there.
fn note_fallback(kind: usize, va: u64) {
    let page = va >> 16;
    let new = {
        let mut guard = PAGES.lock().unwrap_or_else(|e| e.into_inner());
        let pages = guard.get_or_insert_with(Default::default);
        pages.len() < MAX_PAGES && pages.insert(page)
    };
    if new {
        xs_log!("console list root {} at title address {va:#x} is in no PC resource, the zeroed fallback buffer is bound: {}", KINDS[kind], describe_memory(va));
    }
}

/// What Windows knows of the memory at `va`: the region, its allocation, state, protection and type,
/// and the module for an image.
pub(super) fn describe_memory(va: u64) -> String {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: VirtualQuery reads no memory at `va`, it describes the region; the output is plain data.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: as above, the size of the output is passed.
    let n = unsafe { VirtualQuery(va as usize as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
    if n == 0 {
        return "not a user address of this process".into();
    }
    let state = match m.State {
        MEM_COMMIT => "committed",
        MEM_RESERVE => "reserved",
        MEM_FREE => "free",
        _ => "?",
    };
    let kind = match m.Type {
        MEM_PRIVATE => "private",
        MEM_MAPPED => "mapped",
        MEM_IMAGE => "image",
        _ => "-",
    };
    let module = if m.Type == MEM_IMAGE { decant_log::calls::module_of(va as usize).map(|(_, name)| format!(", module {name}")).unwrap_or_default() } else { String::new() };
    format!(
        "{state} {kind} region {:#x}+{:#x}, allocation {:#x}, protection {:#x}{module}",
        m.BaseAddress as usize,
        m.RegionSize,
        m.AllocationBase as usize,
        m.Protect
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_root_view_kinds_follow_the_pc_slots() {
        assert_eq!([37, 38, 39, 40, 41, 42].map(kind_of), [0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn an_address_no_resource_holds_has_no_pc_address_and_its_memory_is_described() {
        let here = [0u32; 4];
        assert_eq!(pc_address(here.as_ptr() as u64, 4), None);
        assert!(describe_memory(here.as_ptr() as u64).starts_with("committed private"), "{}", describe_memory(here.as_ptr() as u64));
        let code = describe_memory(an_address_no_resource_holds_has_no_pc_address_and_its_memory_is_described as *const () as u64);
        assert!(code.starts_with("committed image") && code.contains("module"), "{code}");
    }
}
