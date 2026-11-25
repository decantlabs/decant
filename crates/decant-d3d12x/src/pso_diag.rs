//! Refused pipelines. Each pipeline the PC runtime refuses is logged once with the content hash of
//! every shader stage and what became of its DXIL override (hit, miss, refused and why), plus the
//! render target formats of a graphics pipeline.
//!
//! `DECANT_D3D12X_PSO_STANDIN=1` (default off) answers a refused pipeline with a stand in object and
//! `S_OK` instead of the failure, so the title goes on as if it had the pipeline. A list that binds a
//! stand in keeps its last real pipeline on the PC side, and its draws, dispatches and indirect
//! executions are left out until a real pipeline is bound again. Every 16th present logs which stand
//! ins the frame bound and how much work was left out, which names the pipelines a frame needs.
//!
//! A pipeline made on first bind (`lazy_pso`) that the PC runtime refuses is bound the same way,
//! switch or not ([`bind_missing`]): its list leaves out the work until a real pipeline is bound.

#![allow(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

use crate::dxil_override::Outcome;
use crate::log::xs_log;

/// One shader stage of a pipeline: its name (`VS`, `PS`, `CS` and so on), the content hash of the
/// title's bytecode and what became of its override.
pub struct Stage {
    pub name: &'static str,
    pub hash: u64,
    pub outcome: Outcome,
}

/// The fixed function state of a plain graphics desc that decides whether a draw writes anything:
/// culling, depth, the blend of target 0 and its write mask, the targets and the sample mask.
pub fn describe_state(g: &[u8; crate::stream_pso::GRAPHICS_DESC_SIZE]) -> String {
    let u = |o: usize| u32::from_le_bytes([g[o], g[o + 1], g[o + 2], g[o + 3]]);
    let formats: Vec<u32> = (0..u(576).min(8) as usize).map(|i| u(580 + i * 4)).collect();
    let layout = u64::from_le_bytes(g[0x228..0x230].try_into().unwrap_or_default()) as usize;
    let elements = u(0x230).min(16) as usize;
    let inputs: Vec<String> = if layout == 0 {
        Vec::new()
    } else {
        (0..elements)
            .map(|i| {
                // SAFETY: the input layout of a desc the runtime accepted holds `elements` element
                // descriptions of 32 bytes, each naming its semantic with a C string.
                let e = unsafe { std::ptr::read_unaligned((layout + i * 32) as *const [u32; 8]) };
                let at = e[0] as usize | (e[1] as usize) << 32;
                let name = if at == 0 { "?".into() } else { unsafe { std::ffi::CStr::from_ptr(at as *const std::ffi::c_char) }.to_string_lossy() };
                format!("{name}{} fmt {} slot {} at {}", e[2], e[3], e[4], e[5])
            })
            .collect()
    };
    format!(
        "; cull {} ccw {} depth clip {} depth {} write {} func {} stencil {}; blend0 {} {}/{}/{} write mask {:#x}; sample mask {:#x} count {}; targets {formats:?} depth format {}; topology type {}; inputs {inputs:?}",
        u(456), u(460), u(476), u(496), u(500), u(504), u(508), u(128), u(136), u(140), u(144), g[164], u(448), u(616), u(612), u(572)
    )
}

/// The stages of a pipeline as the kind check (`kind_check`) wants them.
pub fn kind_stages(stages: &[Stage]) -> Vec<crate::kind_check::StageIn> {
    stages.iter().map(|s| crate::kind_check::StageIn { stage: s.name, hash: s.hash, overridden: s.outcome == Outcome::Hit }).collect()
}

/// The stages and their override outcomes as one log fragment.
pub fn describe_stages(stages: &[Stage]) -> String {
    stages.iter().map(|s| format!("{} {:016x} {}", s.name, s.hash, s.outcome)).collect::<Vec<_>>().join(", ")
}

static STANDIN: AtomicU8 = AtomicU8::new(0);

/// Whether `DECANT_D3D12X_PSO_STANDIN` is set to something other than 0, read once.
pub fn standin_enabled() -> bool {
    match STANDIN.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_PSO_STANDIN").is_some_and(|v| v != "0");
            STANDIN.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Active pipeline metadata tracked for a command list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActivePso {
    /// Pipeline number when known (from deferred pipeline creation or a stand in).
    pub number: Option<u32>,
    /// Raw pipeline pointer value.
    pub pso: usize,
    /// Shader stage names and content hashes when recorded.
    pub shaders: String,
    /// Whether this pipeline was refused or missing on first bind.
    pub missing: bool,
}

impl ActivePso {
    /// Formats the active pipeline for DRED fault diagnostics.
    pub fn format_summary(&self) -> String {
        let mut head = match (self.number, self.pso) {
            (Some(n), _) => format!("pipeline {n}"),
            (None, pso) if pso != 0 => format!("pso {pso:#x}"),
            (None, 0) => String::from("pso 0x0"),
            _ => unreachable!(),
        };
        if self.missing {
            head.push_str(" (missing)");
        }
        if self.shaders.is_empty() {
            head
        } else {
            format!("{head} [{}]", self.shaders)
        }
    }
}

#[derive(Default)]
struct State {
    /// Stand in number to the description of the refused pipeline it stands for.
    refused: Vec<String>,
    /// Command list proxy to the stand in it has bound.
    bound: HashMap<usize, u32>,
    /// Stand ins bound since the last report, with the number of binds.
    frame_binds: BTreeMap<u32, u64>,
    /// Stand ins that had work left out since the last report, with the number of calls.
    frame_skips: BTreeMap<u32, u64>,
    /// Command list pointer (PC list when available, otherwise proxy) to its active pipeline.
    active: HashMap<usize, ActivePso>,
    /// Command list proxy pointer to its PC command list pointer.
    proxy_to_pc: HashMap<usize, usize>,
    /// Pipeline pointer to its recorded number and shader hashes.
    pipelines: HashMap<usize, ActivePso>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

static REFUSED_LOGGED: AtomicU64 = AtomicU64::new(0);

/// A pipeline was refused by the PC runtime with `hr`. Logs it with its stages, and with stand ins
/// on, writes a stand in to `out` and returns `S_OK`. Otherwise returns `hr` unchanged.
///
/// # Safety
/// `out` is null or writable.
pub unsafe fn refused(hr: i32, kind: &str, stages: &[Stage], extra: &str, out: *mut *mut c_void) -> i32 {
    let n = REFUSED_LOGGED.fetch_add(1, Ordering::Relaxed);
    let what = format!("{kind}: {}{extra}", describe_stages(stages));
    if !standin_enabled() || out.is_null() {
        if n < 4096 {
            xs_log!("refused pipeline {n}, hr=0x{:08X}, {what}", hr as u32);
        }
        return hr;
    }
    let id = with(|s| {
        s.refused.push(what.clone());
        (s.refused.len() - 1) as u32
    });
    if n < 4096 {
        xs_log!("refused pipeline {n}, hr=0x{:08X}, stand in {id}, {what}", hr as u32);
    }
    let object = Box::into_raw(Box::new(StandIn { vtbl: &VTBL, refs: AtomicU32::new(1), id }));
    // SAFETY: by the contract above.
    unsafe { *out = object.cast() };
    0
}

/// The stand in number of `object`, when it is a stand in.
///
/// # Safety
/// `object` is null or a COM object (starts with a vtable pointer).
pub unsafe fn standin_id(object: *mut c_void) -> Option<u32> {
    if object.is_null() || !standin_enabled() {
        return None;
    }
    // SAFETY: by the contract above.
    let vtbl = unsafe { *(object as *const *const VTable) };
    if std::ptr::eq(vtbl, &VTBL) {
        // SAFETY: the vtable is the stand in's, so the object is a `StandIn`.
        Some(unsafe { (*object.cast::<StandIn>()).id })
    } else {
        None
    }
}

/// List `list` bound pipeline `pso` (by SetPipelineState or Reset). Returns whether it is a stand in,
/// which the caller then does not relay.
///
/// # Safety
/// `list` is 0 or a live command list proxy whose second word is its PC command list pointer, and
/// `pso` is null or a COM object (starts with a vtable pointer).
pub unsafe fn bind(list: usize, pso: *mut c_void) -> bool {
    // SAFETY: by the contract above.
    let pc_list = if list != 0 { unsafe { *(list as *const usize).add(1) } } else { 0 };
    let track_key = if pc_list != 0 { pc_list } else { list };
    // SAFETY: by the contract above.
    let id = unsafe { standin_id(pso) };
    // SAFETY: `pso` is non null and starts with a vtable pointer. When `is_lazy` is true, its
    // `#[repr(C)]` header places `number: u32` right after `vtbl` and `refs`.
    let lazy_number = if id.is_none() && !pso.is_null() && unsafe { crate::lazy_pso::is_lazy(pso) } {
        Some(unsafe { *(pso as *const u8).add(std::mem::size_of::<usize>() + 4).cast::<u32>() })
    } else {
        None
    };

    with(|s| {
        if pc_list != 0 && list != 0 {
            s.proxy_to_pc.insert(list, pc_list);
        }
        if track_key != 0 {
            if pso.is_null() {
                s.active.remove(&track_key);
            } else if let Some(id) = id {
                let shaders = s.refused.get(id as usize).cloned().unwrap_or_default();
                s.active.insert(
                    track_key,
                    ActivePso { number: Some(id), pso: pso as usize, shaders, missing: true },
                );
            } else {
                let mut entry = s.pipelines.get(&(pso as usize)).cloned().unwrap_or_default();
                entry.pso = pso as usize;
                entry.number = entry.number.or(lazy_number);
                s.active.insert(track_key, entry);
            }
        }
        if standin_enabled() {
            match id {
                Some(id) => {
                    s.bound.insert(list, id);
                    *s.frame_binds.entry(id).or_default() += 1;
                }
                None => {
                    s.bound.remove(&list);
                }
            }
        } else if MISSING.load(Ordering::Relaxed) {
            s.bound.remove(&list);
        }
    });
    id.is_some()
}

/// Whether a pipeline made on first bind was ever refused, so lists are tracked without the switch.
static MISSING: AtomicBool = AtomicBool::new(false);

/// The stand in number noted for a pipeline made on first bind that the PC runtime refused.
pub const MISSING_ID: u32 = u32::MAX;

/// List `list` bound a pipeline made on first bind that the PC runtime refused: like a stand in, its
/// work is left out until a real pipeline is bound.
pub fn bind_missing(list: usize) {
    MISSING.store(true, Ordering::Relaxed);
    with(|s| {
        s.bound.insert(list, MISSING_ID);
        *s.frame_binds.entry(MISSING_ID).or_default() += 1;
        let key = s.proxy_to_pc.get(&list).copied().unwrap_or(list);
        if key != 0 {
            let entry = s.active.entry(key).or_default();
            entry.missing = true;
        }
    });
}

/// Records pipeline number and shader hashes for `pso` so later binds on any list report them.
#[cfg(test)]
pub fn register_pipeline(pso: usize, number: Option<u32>, stages: &[Stage]) {
    if pso == 0 {
        return;
    }
    let shaders = describe_stages(stages);
    with(|s| {
        s.pipelines.insert(pso, ActivePso { number, pso, shaders, missing: false });
    });
}

/// Records `active` as the active pipeline on command list `list`.
#[cfg(test)]
pub fn track_bind(list: usize, active: ActivePso) {
    if list == 0 {
        return;
    }
    with(|s| {
        let key = s.proxy_to_pc.get(&list).copied().unwrap_or(list);
        s.active.insert(key, active);
    });
}

/// Clears the active pipeline tracked for command list `list` when it is reset.
#[cfg(test)]
pub fn reset_list(list: usize) {
    forget_list(list);
}

/// Clears the active pipeline tracked for command list `list` when it is closed.
#[cfg(test)]
pub fn close_list(list: usize) {
    forget_list(list);
}

#[cfg(test)]
fn forget_list(list: usize) {
    if list == 0 {
        return;
    }
    with(|s| {
        s.active.remove(&list);
        if let Some(pc) = s.proxy_to_pc.get(&list).copied() {
            s.active.remove(&pc);
        }
    });
}

/// Returns the active pipeline currently tracked for command list `list`, if any.
pub fn active_pso(list: usize) -> Option<ActivePso> {
    if list == 0 {
        return None;
    }
    with(|s| {
        s.active
            .get(&list)
            .cloned()
            .or_else(|| s.proxy_to_pc.get(&list).and_then(|pc| s.active.get(pc)).cloned())
    })
}

/// Whether a draw, dispatch or indirect execution on `list` is left out because a stand in is bound.
pub fn skip(list: usize) -> bool {
    if !standin_enabled() && !MISSING.load(Ordering::Relaxed) {
        return false;
    }
    with(|s| match s.bound.get(&list).copied() {
        Some(id) => {
            *s.frame_skips.entry(id).or_default() += 1;
            true
        }
        None => false,
    })
}

static PRESENTS: AtomicU64 = AtomicU64::new(0);

/// A present. Every 16th logs the stand ins bound since the last report.
pub fn present() {
    if !standin_enabled() {
        return;
    }
    let n = PRESENTS.fetch_add(1, Ordering::Relaxed) + 1;
    if !n.is_multiple_of(16) {
        return;
    }
    let (binds, skips, total) = with(|s| (std::mem::take(&mut s.frame_binds), std::mem::take(&mut s.frame_skips), s.refused.len()));
    let fmt = |m: &BTreeMap<u32, u64>| m.iter().map(|(id, n)| format!("{id}:{n}")).collect::<Vec<_>>().join(" ");
    xs_log!("stand ins, 16 presents up to {n}: {total} made, bound {}, work left out {}", fmt(&binds), fmt(&skips));
    // The first time a stand in shows up in a report, its description once more, so the report can be
    // read without the line of its refusal.
    static NAMED: Mutex<Option<std::collections::HashSet<u32>>> = Mutex::new(None);
    let mut named = NAMED.lock().unwrap_or_else(|e| e.into_inner());
    let named = named.get_or_insert_with(Default::default);
    for id in binds.keys().chain(skips.keys()) {
        if named.insert(*id) {
            let what = with(|s| s.refused.get(*id as usize).cloned().unwrap_or_default());
            xs_log!("stand in {id} is used by the frame: {what}");
        }
    }
}

// ---- the stand in object: IUnknown, ID3D12Object, ID3D12DeviceChild, ID3D12PipelineState ----

#[repr(C)]
struct StandIn {
    vtbl: *const VTable,
    refs: AtomicU32,
    id: u32,
}

type Method = unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> i32;

#[repr(C)]
struct VTable([Method; 16]);

const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_NOTIMPL: i32 = 0x8000_4001_u32 as i32;
const DXGI_ERROR_NOT_FOUND: i32 = 0x887A_0002_u32 as i32;

unsafe extern "system" fn query_interface(this: *mut c_void, _riid: usize, out: usize, _: usize) -> i32 {
    // Any interface: the title only asks a pipeline for its own kinds.
    if out == 0 {
        return E_NOINTERFACE;
    }
    // SAFETY: `this` is a stand in, `out` the caller's writable pointer.
    unsafe {
        (*this.cast::<StandIn>()).refs.fetch_add(1, Ordering::Relaxed);
        *(out as *mut *mut c_void) = this;
    }
    0
}

unsafe extern "system" fn add_ref(this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    // SAFETY: `this` is a stand in.
    unsafe { (*this.cast::<StandIn>()).refs.fetch_add(1, Ordering::Relaxed) as i32 + 1 }
}

unsafe extern "system" fn release(this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    // Never freed: a list may still name it, and a stand in is a few bytes.
    // SAFETY: `this` is a stand in.
    unsafe { ((*this.cast::<StandIn>()).refs.fetch_sub(1, Ordering::Relaxed) as i32 - 1).max(0) }
}

unsafe extern "system" fn get_private_data(_this: *mut c_void, _: usize, size: usize, _: usize) -> i32 {
    if size != 0 {
        // SAFETY: the caller's in and out size.
        unsafe { *(size as *mut u32) = 0 };
    }
    DXGI_ERROR_NOT_FOUND
}

unsafe extern "system" fn accepted(_this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    0
}

unsafe extern "system" fn get_device(_this: *mut c_void, _riid: usize, out: usize, _: usize) -> i32 {
    if out != 0 {
        // SAFETY: the caller's writable pointer.
        unsafe { *(out as *mut *mut c_void) = std::ptr::null_mut() };
    }
    E_NOINTERFACE
}

unsafe extern "system" fn get_cached_blob(_this: *mut c_void, out: usize, _: usize, _: usize) -> i32 {
    if out != 0 {
        // SAFETY: the caller's writable pointer.
        unsafe { *(out as *mut *mut c_void) = std::ptr::null_mut() };
    }
    E_NOTIMPL
}

unsafe extern "system" fn not_known(_this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    E_NOTIMPL
}

static VTBL: VTable = VTable([
    query_interface,
    add_ref,
    release,
    get_private_data,
    accepted, // SetPrivateData
    accepted, // SetPrivateDataInterface
    accepted, // SetName
    get_device,
    get_cached_blob,
    not_known,
    not_known,
    not_known,
    not_known,
    not_known,
    not_known,
    not_known,
]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_active_pso_and_clears_on_reset_and_close() {
        let list_a = 0x1001_0001;
        let list_b = 0x1001_0002;
        assert_eq!(active_pso(list_a), None);

        track_bind(
            list_a,
            ActivePso {
                number: Some(12),
                pso: 0xABCD,
                shaders: String::from("VS 1111222233334444 hit, PS 5555666677778888 miss"),
                missing: false,
            },
        );
        track_bind(
            list_b,
            ActivePso {
                number: Some(99),
                pso: 0xDEAD,
                shaders: String::from("CS 9999aaaabbbbcccc hit"),
                missing: false,
            },
        );

        let a = active_pso(list_a).expect("list A should have an active pipeline");
        assert_eq!(a.number, Some(12));
        assert_eq!(a.pso, 0xABCD);
        assert!(a.shaders.contains("1111222233334444"));
        assert!(!a.missing);

        reset_list(list_a);
        assert_eq!(active_pso(list_a), None);
        assert!(active_pso(list_b).is_some());

        close_list(list_b);
        assert_eq!(active_pso(list_b), None);
    }

    #[test]
    fn bind_tracks_proxy_to_pc_list_and_registered_shaders() {
        let pc_list = 0x2001_0010usize;
        let proxy_stub: [usize; 2] = [0x9000, pc_list];
        let proxy_ptr = proxy_stub.as_ptr() as usize;

        let pso_vtbl: usize = 0x8000;
        let pso_stub: [usize; 2] = [&raw const pso_vtbl as usize, 0];
        let pso_ptr = pso_stub.as_ptr() as *mut c_void;

        register_pipeline(
            pso_ptr as usize,
            Some(7),
            &[
                Stage { name: "VS", hash: 0x0123_4567_89ab_cdef, outcome: Outcome::Hit },
                Stage { name: "PS", hash: 0xfedc_ba98_7654_3210, outcome: Outcome::Miss },
            ],
        );

        // SAFETY: `proxy_stub` has the PC command list pointer in its second word and `pso_stub`
        // starts with a readable vtable pointer.
        let skipped = unsafe { bind(proxy_ptr, pso_ptr) };
        assert!(!skipped);

        let by_pc = active_pso(pc_list).expect("PC command list should be tracked");
        assert_eq!(by_pc.number, Some(7));
        assert_eq!(by_pc.shaders, "VS 0123456789abcdef hit, PS fedcba9876543210 miss");
        assert_eq!(active_pso(proxy_ptr), Some(by_pc));

        bind_missing(proxy_ptr);
        let marked = active_pso(pc_list).expect("missing bind should preserve pipeline info");
        assert!(marked.missing);
        assert_eq!(marked.number, Some(7));

        // SAFETY: `proxy_stub` is live and null PSO clears the active pipeline.
        unsafe { bind(proxy_ptr, std::ptr::null_mut()) };
        assert_eq!(active_pso(pc_list), None);
        assert_eq!(active_pso(proxy_ptr), None);
    }

    #[test]
    fn active_pso_summary_formats_variants() {
        let with_shaders = ActivePso {
            number: Some(42),
            pso: 0x1234,
            shaders: String::from("CS aabbccdd11223344 hit"),
            missing: false,
        };
        assert_eq!(with_shaders.format_summary(), "pipeline 42 [CS aabbccdd11223344 hit]");

        let missing = ActivePso { number: Some(5), pso: 0x100, shaders: String::new(), missing: true };
        assert_eq!(missing.format_summary(), "pipeline 5 (missing)");

        let raw_only = ActivePso { number: None, pso: 0xdeadbeef, shaders: String::new(), missing: false };
        assert_eq!(raw_only.format_summary(), "pso 0xdeadbeef");
    }
}
