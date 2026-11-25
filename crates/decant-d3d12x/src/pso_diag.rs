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
/// As for [`standin_id`].
pub unsafe fn bind(list: usize, pso: *mut c_void) -> bool {
    if !standin_enabled() {
        if MISSING.load(Ordering::Relaxed) {
            with(|s| s.bound.remove(&list));
        }
        return false;
    }
    // SAFETY: by the contract above.
    let id = unsafe { standin_id(pso) };
    with(|s| match id {
        Some(id) => {
            s.bound.insert(list, id);
            *s.frame_binds.entry(id).or_default() += 1;
        }
        None => {
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
    });
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
