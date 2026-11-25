//! Input assembler diagnostics behind `DECANT_D3D12X_PRESENT_PROBE`: for the draws the view probe
//! describes (the first and the 40th of each render target and pipeline pair), the index and vertex
//! buffer views as the title gave them and as they reached the PC list, the draw arguments, and the
//! first indices and vertices read from the title's own memory. A draw that leaves no pixel can so
//! be told apart: a view the layer sent to the zeroed fallback buffer, indices that are all zero,
//! or vertices that are.
//!
//! `DECANT_D3D12X_DRAW_ORDER=<n>` also logs the order of 60 draws from the nth on, with the list of
//! each and the lists of every submission in between, which tells whether a draw comes before or
//! after a full screen one.
//!
//! Off by default. Off, every hook here is one relaxed atomic load.

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};

/// A view as the title gave it and as the PC list got it.
#[derive(Clone, Copy, Default)]
struct Pair<T> {
    title: T,
    pc: T,
}

#[derive(Default)]
struct ListState {
    index: Option<Pair<[u8; 16]>>,
    vertex: HashMap<u32, Pair<[u64; 2]>>,
    /// How many viewports and scissor rectangles the title set last, and the first of each.
    viewports: Option<(u32, Option<[f32; 6]>)>,
    scissors: Option<(u32, Option<[i32; 4]>)>,
}

static LISTS: Mutex<Option<HashMap<usize, ListState>>> = Mutex::new(None);

fn with<R>(list: usize, f: impl FnOnce(&mut ListState) -> R) -> R {
    let mut g = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new).entry(list).or_default())
}

/// Whether the buffer views of the lists are kept: for the descriptions, or for the flicker probe
/// (`flicker_probe`), which reads the ranges a draw reads.
fn recording() -> bool {
    crate::rt_diag::enabled() || crate::flicker_probe::on()
}

/// `list` got an index buffer view: `title` as the title gave it, `pc` as the PC list got it.
pub fn index_view(list: usize, title: Option<[u8; 16]>, pc: [u8; 16]) {
    if recording() {
        with(list, |s| s.index = title.map(|title| Pair { title, pc }));
    }
}

/// `list` got a vertex buffer view in `slot`.
pub fn vertex_view(list: usize, slot: u32, title: [u64; 2], pc: [u64; 2]) {
    if recording() {
        with(list, |s| {
            s.vertex.insert(slot, Pair { title, pc });
        });
    }
}

/// The bound buffers of a list as the title gave them: the index buffer view and the vertex buffer
/// views by slot.
pub type Buffers = (Option<[u8; 16]>, Vec<(u32, [u64; 2])>);

/// The buffers bound on `list`.
pub fn buffers(list: usize) -> Buffers {
    let g = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(s) = g.as_ref().and_then(|m| m.get(&list)) else { return (None, Vec::new()) };
    (s.index.map(|p| p.title), s.vertex.iter().map(|(&slot, v)| (slot, v.title)).collect())
}

/// `list` got `count` viewports, the first is `first`.
pub fn viewports(list: usize, count: u32, first: Option<[f32; 6]>) {
    if crate::rt_diag::enabled() {
        with(list, |s| s.viewports = Some((count, first)));
    }
}

/// `list` got `count` scissor rectangles, the first is `first`.
pub fn scissors(list: usize, count: u32, first: Option<[i32; 4]>) {
    if crate::rt_diag::enabled() {
        with(list, |s| s.scissors = Some((count, first)));
    }
}

/// `list` was reset.
pub fn forget(list: usize) {
    if recording() {
        LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(|m| m.remove(&list));
    }
}

/// Whether `len` bytes at `va` are committed and readable in this process.
pub(crate) fn readable(va: usize, len: usize) -> bool {
    if va == 0 || len == 0 {
        return false;
    }
    let mut at = va;
    let end = va.saturating_add(len);
    while at < end {
        // SAFETY: a plain structure the call fills; the call only looks the address up.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { VirtualQuery(at as *const _, &mut info, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        crate::cpu_writes::lazily_committed(&mut info);
        if info.State != MEM_COMMIT || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || info.Protect == 0 {
            return false;
        }
        at = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

fn bytes_at(va: usize, len: usize) -> Option<Vec<u8>> {
    // SAFETY: the range was just found committed and readable.
    readable(va, len).then(|| unsafe { std::slice::from_raw_parts(va as *const u8, len) }.to_vec())
}

fn words(bytes: &[u8]) -> String {
    bytes.as_chunks::<4>().0.iter().map(|w| format!("{:x}", u32::from_le_bytes(*w))).collect::<Vec<_>>().join(" ")
}

/// Draws recorded so far, over all lists.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The draws whose order is logged: the 60 from the number `DECANT_D3D12X_DRAW_ORDER` names (the
/// count of draws since the start, over all lists), none without it.
fn in_order_window(sequence: u64) -> bool {
    static FROM: OnceLock<Option<u64>> = OnceLock::new();
    FROM.get_or_init(|| std::env::var("DECANT_D3D12X_DRAW_ORDER").ok().and_then(|v| v.trim().parse().ok())).is_some_and(|from| (from..from + 60).contains(&sequence))
}

/// The lists of one submission, logged while the order window is open.
pub fn submitted(lists: &[usize]) {
    if crate::rt_diag::enabled() && in_order_window(SEQUENCE.load(Ordering::Relaxed)) {
        crate::log::xs_log!("draw order: submission of lists {lists:x?}");
    }
}

/// Dispatches recorded so far, by compute pipeline.
static DISPATCHES: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);

/// `list` records a dispatch of `x` by `y` by `z` groups: how many its pipeline has now, 0 when
/// the probe is off. The first and the 40th of each pipeline are logged with their group counts.
pub fn dispatch(list: usize, x: u32, y: u32, z: u32) -> u64 {
    if !crate::rt_diag::enabled() {
        return 0;
    }
    let (_, pso) = crate::rt_diag::list_state(list);
    let n = {
        let mut g = DISPATCHES.lock().unwrap_or_else(|e| e.into_inner());
        let n = g.get_or_insert_with(HashMap::new).entry(pso).or_default();
        *n += 1;
        *n
    };
    if in_order_window(SEQUENCE.load(Ordering::Relaxed)) {
        crate::log::xs_log!("draw order: list {list:#x} pipeline {pso:#x} Dispatch({x}, {y}, {z})");
    }
    if crate::view_diag::picked(n) {
        crate::log::xs_log!("draw probe: dispatch {n} of pipeline {pso:#x}: {x} by {y} by {z} groups");
    }
    n
}

/// How many indices and vertices one description reads.
const SHOWN: usize = 6;

/// The line for draw `n` of `list`: `indices` is `Some((count, start index, base vertex))` for an
/// indexed draw, `first` the first vertex of a draw without indices.
pub fn describe(list: usize, n: u64, kind: &str, indices: Option<(u32, u32, i32)>, first: u32, instances: u32) {
    if !crate::rt_diag::enabled() {
        return;
    }
    let (rt, pso) = crate::rt_diag::list_state(list);
    if in_order_window(SEQUENCE.fetch_add(1, Ordering::Relaxed)) {
        crate::log::xs_log!("draw order: list {list:#x} target {rt:#x} pipeline {pso:#x} {kind} {indices:?} first {first}");
    }
    if !crate::view_diag::picked(n) {
        return;
    }
    let (index, vertex, viewports, scissors) = with(list, |s| (s.index, s.vertex.clone(), s.viewports, s.scissors));
    let mut out = format!("draw probe: draw {n} of target {rt:#x} pipeline {pso:#x}: {kind} {indices:?} first {first} instances {instances}; viewports {viewports:?}, scissors {scissors:?}");
    let mut picked: Vec<i64> = (0..SHOWN as i64).map(|i| i + i64::from(first)).collect();
    if let Some((count, start, base)) = indices {
        match index {
            None => out += "; no index buffer bound",
            Some(iv) => {
                let va = u64::from_le_bytes(iv.title[0..8].try_into().expect("eight bytes"));
                let pc = u64::from_le_bytes(iv.pc[0..8].try_into().expect("eight bytes"));
                let size = u32::from_le_bytes(iv.title[8..12].try_into().expect("four bytes"));
                let pc_size = u32::from_le_bytes(iv.pc[8..12].try_into().expect("four bytes"));
                let format = u32::from_le_bytes(iv.title[12..16].try_into().expect("four bytes"));
                let stride = if format == 42 { 4 } else { 2 };
                out += &format!("; index view title {va:#x} size {size} format {format} -> PC {pc:#x} size {pc_size}");
                let shown = (count as usize).min(SHOWN);
                match bytes_at(va as usize + start as usize * stride, shown * stride) {
                    Some(b) => {
                        let values: Vec<i64> = b.chunks(stride).map(|c| if stride == 4 { i64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])) } else { i64::from(u16::from_le_bytes([c[0], c[1]])) }).collect();
                        out += &format!(", first indices {values:?}");
                        picked = values.iter().map(|v| v + i64::from(base)).collect();
                    }
                    None => out += ", indices not readable by the CPU",
                }
            }
        }
    }
    let mut slots: Vec<u32> = vertex.keys().copied().collect();
    slots.sort_unstable();
    if slots.is_empty() {
        out += "; no vertex buffer bound";
    }
    for slot in slots {
        let v = vertex[&slot];
        let (va, size, stride) = (v.title[0], v.title[1] & 0xFFFF_FFFF, (v.title[1] >> 32) as usize);
        out += &format!("; vertex view {slot} title {va:#x} size {size} stride {stride} -> PC {:#x} size {}", v.pc[0], v.pc[1] & 0xFFFF_FFFF);
        if stride == 0 || stride > 256 {
            continue;
        }
        for &i in &picked {
            if i < 0 || (i as u64 + 1) * stride as u64 > size {
                out += &format!(" [{i}: outside]");
                continue;
            }
            match bytes_at(va as usize + i as usize * stride, stride) {
                Some(b) => out += &format!(" [{i}: {}]", words(&b)),
                None => out += &format!(" [{i}: not readable]"),
            }
        }
    }
    crate::log::xs_log!("{out}");
}
