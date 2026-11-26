//! GPU indirect argument clamping with per-list work budget and command-signature parsing.
//!
//! Ported from vitutu1/victory-perf (GPL-3.0-or-later) and adapted to this codebase.
//! Ported from ivall/xcl (GPL-2.0-only) and adapted to this codebase.
//!
//! When compute passes that fill indirect argument buffers are refused or stale, `ExecuteIndirect`
//! can issue draws of billions of vertices or dispatches past host limits, triggering a GPU hang
//! (TDR). This module copies active indirect records into a per-list device-local scratch buffer
//! through a small SM 5.1 compute shader that clamps vertex/instance/group counts against a
//! per-list saturating work budget (`2^27`), restores the command list's compute root signature,
//! compute root arguments, and pipeline state, and runs `ExecuteIndirect` against the clamped
//! scratch buffer.
//!
//! The root signature is serialized explicitly via `D3D12SerializeRootSignature` rather than
//! embedded with `[RootSignature]` in HLSL for Wine `d3dcompiler_47` compatibility.
//!
//! Switches:
//! - `DECANT_D3D12X_INDIRECT_CLAMP=1` enables GPU indirect argument clamping (default off; `0` =
//!   unclamped passthrough).
//! - `DECANT_D3D12X_INDIRECT_CLAMP_SCOPED=1` uses resource-scoped UAV pre-barriers on the argument
//!   and count buffers instead of a global UAV pre-wait (default off; safe global wait is
//!   deduplicated via `uav_barrier_merge` because reserved-window / placed buffers may alias).

use std::collections::HashMap;
#[cfg(windows)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
use crate::log::xs_log;
#[cfg(not(windows))]
macro_rules! xs_log {
    ($($t:tt)*) => {};
}

/// Console-only `D3D12XBOX_INDIRECT_ARGUMENT_TYPE_DISPATCH_X` (105).
pub const INDIRECT_ARGUMENT_DISPATCH_X: u32 = 105;
/// PC `D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH` (2).
pub const PC_INDIRECT_ARGUMENT_DISPATCH: u32 = 2;

/// Clamping spec for a single-action command signature (`Draw`, `DrawIndexed`, `Dispatch`,
/// `DispatchMesh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClampSpec {
    pub kind: u32,
    pub stride: u32,
}

/// Translated `D3D12_COMMAND_SIGNATURE_DESC` ready for the PC runtime.
pub struct TranslatedSignatureDesc {
    pub desc: [u8; 24],
    pub args: Vec<[u32; 4]>,
    pub needs_root: bool,
    pub spec: Option<ClampSpec>,
}

/// Whether GPU indirect argument clamping is enabled (`DECANT_D3D12X_INDIRECT_CLAMP=1`, default off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("DECANT_D3D12X_INDIRECT_CLAMP").is_some_and(|v| v == "1");
        if on {
            xs_log!(
                "indirect clamp: enabled (DECANT_D3D12X_INDIRECT_CLAMP=1, scoped={})",
                scoped()
            );
        }
        on
    })
}

/// Whether pre-clamp UAV waits are scoped to `src`/`count` (`DECANT_D3D12X_INDIRECT_CLAMP_SCOPED=1`,
/// default off so reserved-window aliases stay ordered via global wait).
pub fn scoped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_INDIRECT_CLAMP_SCOPED").is_some_and(|v| v == "1"))
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
pub const fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),       // Draw
        1 => Some(20),       // DrawIndexed
        2 | 10 => Some(12),  // Dispatch / DispatchMesh
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
pub fn validate_layout(
    spec: ClampSpec,
    records: u32,
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(spec.kind)?;
    if records == 0 || spec.stride < min || !spec.stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(spec.stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Translates a console `D3D12_COMMAND_SIGNATURE_DESC` (24 bytes: `ByteStride @ 0`,
/// `NumArgumentDescs @ 4`, `pArgumentDescs @ 8`, `NodeMask @ 16`) into an owned PC copy with
/// console `DispatchX` (105) mapped to PC `Dispatch` (2), `needs_root` computed (arguments 5..=8),
/// and the optional single-action [`ClampSpec`] extracted.
///
/// # Safety
/// `desc` must be non-null and readable for 24 bytes; when `NumArgumentDescs > 0` and
/// `pArgumentDescs` is non-null, it must be readable for `NumArgumentDescs * 16` bytes.
#[allow(unsafe_code)]
pub unsafe fn translate_command_signature_desc(desc: *const u8) -> TranslatedSignatureDesc {
    let mut pc = [0u8; 24];
    // SAFETY: by contract above.
    unsafe { std::ptr::copy_nonoverlapping(desc, pc.as_mut_ptr(), 24) };
    let byte_stride = u32::from_le_bytes([pc[0], pc[1], pc[2], pc[3]]);
    let n = u32::from_le_bytes([pc[4], pc[5], pc[6], pc[7]]) as usize;
    let args_ptr = u64::from_le_bytes([pc[8], pc[9], pc[10], pc[11], pc[12], pc[13], pc[14], pc[15]]) as *const [u32; 4];
    let mut args: Vec<[u32; 4]> = if args_ptr.is_null() || n > 64 {
        Vec::new()
    } else {
        // SAFETY: by contract above.
        (0..n).map(|i| unsafe { args_ptr.add(i).read_unaligned() }).collect()
    };
    for a in &mut args {
        if a[0] == INDIRECT_ARGUMENT_DISPATCH_X {
            a[0] = PC_INDIRECT_ARGUMENT_DISPATCH;
        }
    }
    let needs_root = args.iter().any(|a| (5..=8).contains(&a[0]));
    let spec = if args.len() == 1 {
        let kind = args[0][0];
        min_stride(kind)
            .filter(|&min| byte_stride >= min && byte_stride.is_multiple_of(4))
            .map(|_| ClampSpec { kind, stride: byte_stride })
    } else {
        None
    };
    let mut out = TranslatedSignatureDesc { desc: pc, args, needs_root, spec };
    let ptr = if out.args.is_empty() { 0u64 } else { out.args.as_ptr() as u64 };
    out.desc[8..16].copy_from_slice(&ptr.to_le_bytes());
    out
}

/// Command signature registry: PC `ID3D12CommandSignature` pointer -> [`ClampSpec`].
static SIGNATURES: Mutex<Option<HashMap<usize, ClampSpec>>> = Mutex::new(None);

/// Records the clamping spec of a newly created PC `ID3D12CommandSignature`.
pub fn register_signature(pc_sig: usize, spec: ClampSpec) {
    if pc_sig == 0 {
        return;
    }
    let mut guard = SIGNATURES.lock().unwrap_or_else(|e| e.into_inner());
    guard.get_or_insert_with(HashMap::new).insert(pc_sig, spec);
}

/// Looks up the clamping spec of a PC `ID3D12CommandSignature`.
pub fn signature_spec(pc_sig: usize) -> Option<ClampSpec> {
    let guard = SIGNATURES.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref()?.get(&pc_sig).copied()
}

/// Compute root-descriptor kind for restoring clobbered compute root bindings after a clamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootVaKind {
    Cbv,
    Srv,
    Uav,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ComputeParam {
    Table(u64),
    Constants(Vec<u32>),
    Va(RootVaKind, u64),
}

#[derive(Default)]
struct SavedListState {
    pso: usize,
    compute_rs: usize,
    compute_params: Vec<Option<ComputeParam>>,
}

static LIST_SAVED: Mutex<Option<HashMap<usize, SavedListState>>> = Mutex::new(None);

fn with_saved(list: usize, f: impl FnOnce(&mut SavedListState)) {
    if !enabled() {
        return;
    }
    let mut guard = LIST_SAVED.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(HashMap::new).entry(list).or_default());
}

/// Notes the currently bound PC pipeline state on `list`.
pub fn note_pso(list: usize, pc_pso: usize) {
    with_saved(list, |s| s.pso = pc_pso);
}

/// Notes `SetComputeRootSignature` on `list` (clears saved compute root params per D3D12 spec).
pub fn note_compute_rs(list: usize, pc_rs: usize) {
    with_saved(list, |s| {
        s.compute_rs = pc_rs;
        s.compute_params.clear();
    });
}

/// Notes `SetComputeRootDescriptorTable` on `list`.
pub fn note_compute_table(list: usize, index: u32, handle: u64) {
    let idx = index as usize;
    if idx >= 64 {
        return;
    }
    with_saved(list, |s| {
        if s.compute_params.len() <= idx {
            s.compute_params.resize(idx + 1, None);
        }
        s.compute_params[idx] = Some(ComputeParam::Table(handle));
    });
}

/// Notes `SetComputeRoot32BitConstant(s)` on `list`.
pub fn note_compute_constants(list: usize, index: u32, offset: u32, values: &[u32]) {
    let idx = index as usize;
    let off = offset as usize;
    if idx >= 64 || off + values.len() > 64 {
        return;
    }
    with_saved(list, |s| {
        if s.compute_params.len() <= idx {
            s.compute_params.resize(idx + 1, None);
        }
        let buf = match &mut s.compute_params[idx] {
            Some(ComputeParam::Constants(v)) => v,
            slot => {
                *slot = Some(ComputeParam::Constants(Vec::new()));
                match slot {
                    Some(ComputeParam::Constants(v)) => v,
                    _ => unreachable!(),
                }
            }
        };
        if buf.len() < off + values.len() {
            buf.resize(off + values.len(), 0);
        }
        buf[off..off + values.len()].copy_from_slice(values);
    });
}

/// Notes `SetComputeRoot{ConstantBuffer,ShaderResource,UnorderedAccess}View` on `list` with the
/// already-translated PC GPU virtual address.
pub fn note_compute_va(list: usize, kind: RootVaKind, index: u32, gpu: u64) {
    let idx = index as usize;
    if idx >= 64 {
        return;
    }
    with_saved(list, |s| {
        if s.compute_params.len() <= idx {
            s.compute_params.resize(idx + 1, None);
        }
        s.compute_params[idx] = Some(ComputeParam::Va(kind, gpu));
    });
}

/// Resets saved compute/PSO bindings and marks the list's clamp budget due for zeroing.
pub fn list_reset(list: usize, initial_pc_pso: usize) {
    if !enabled() {
        return;
    }
    {
        let mut guard = LIST_SAVED.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.get_or_insert_with(HashMap::new).entry(list).or_default();
        entry.pso = initial_pc_pso;
        entry.compute_rs = 0;
        entry.compute_params.clear();
    }
    #[cfg(windows)]
    win::list_reset(list);
}

/// Drops saved state and retires per-list scratch/budget buffers on command-list release.
pub fn list_forget(list: usize) {
    if !enabled() {
        return;
    }
    if let Some(map) = LIST_SAVED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        map.remove(&list);
    }
    #[cfg(windows)]
    win::list_forget(list);
}

#[cfg(windows)]
static CLAMPED_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(windows)]
static CLAMPED_RECORDS: AtomicU64 = AtomicU64::new(0);
#[cfg(windows)]
static PASSTHROUGH_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(windows)]
static FIRST_CLAMP_LOGGED: AtomicBool = AtomicBool::new(false);

/// Compute shader clamping indirect argument records with a bounded 256-try atomic budget loop.
/// Intentionally carries no `[RootSignature]` attribute (Wine `d3dcompiler_47` bug #188).
#[cfg(windows)]
const SOURCE: &str = r#"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        uint2 asked = src_args.Load2(src_at);
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        uint3 g = src_args.Load3(src_at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        bool settled = false;
        [allow_uav_condition]
        for (uint tries = 0; tries < 256u && before < BUDGET; tries++) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) { settled = true; break; }
            before = prev;
        }
        over = !settled || before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

#[cfg(windows)]
#[allow(unsafe_code)]
mod win {
    use super::*;
    use std::ffi::c_void;
    use std::mem::ManuallyDrop;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::{Interface, PCSTR, PCWSTR};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    const MIN_SCRATCH: u64 = 64 << 10;
    const LOG_EVERY: u64 = 4096;

    type SerializeFn = unsafe extern "system" fn(
        *const D3D12_ROOT_SIGNATURE_DESC,
        D3D_ROOT_SIGNATURE_VERSION,
        *mut Option<ID3DBlob>,
        *mut Option<ID3DBlob>,
    ) -> windows::core::HRESULT;

    type CompileFn = unsafe extern "system" fn(
        *const c_void,
        usize,
        PCSTR,
        *const c_void,
        *mut c_void,
        PCSTR,
        PCSTR,
        u32,
        u32,
        *mut Option<ID3DBlob>,
        *mut Option<ID3DBlob>,
    ) -> windows::core::HRESULT;

    unsafe fn dll_fn<F: Copy>(dll: PCWSTR, name: &std::ffi::CStr) -> Option<F> {
        const { assert!(size_of::<F>() == size_of::<usize>()) };
        unsafe {
            let module = LoadLibraryW(dll.0);
            if module.is_null() {
                return None;
            }
            let f = GetProcAddress(module, name.as_ptr().cast())?;
            Some(std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, F>(&f))
        }
    }

    fn compile_cs() -> windows::core::Result<Vec<u8>> {
        let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
        let func = unsafe { dll_fn::<CompileFn>(windows::core::w!("d3dcompiler_47.dll"), c"D3DCompile") }.ok_or_else(fail)?;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        let hr = unsafe {
            func(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                PCSTR::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                PCSTR(c"main".as_ptr().cast()),
                PCSTR(c"cs_5_1".as_ptr().cast()),
                0,
                0,
                &mut code,
                &mut errors,
            )
        };
        if hr.is_err() || code.is_none() {
            if let Some(err) = errors {
                let text = unsafe {
                    String::from_utf8_lossy(std::slice::from_raw_parts(err.GetBufferPointer().cast::<u8>(), err.GetBufferSize())).into_owned()
                };
                xs_log!("indirect clamp: shader compile failed ({hr:?}): {text}");
            }
            return Err(windows::core::Error::from_hresult(hr));
        }
        let blob = code.ok_or_else(fail)?;
        Ok(unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }.to_vec())
    }

    fn build_root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
        let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
        let serialize = unsafe { dll_fn::<SerializeFn>(windows::core::w!("d3d12.dll"), c"D3D12SerializeRootSignature") }.ok_or_else(fail)?;
        let root_desc = |reg: u32| D3D12_ROOT_DESCRIPTOR { ShaderRegister: reg, RegisterSpace: 0 };
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 6 },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: root_desc(0) },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: root_desc(1) },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: root_desc(0) },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: root_desc(1) },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut blob: Option<ID3DBlob> = None;
        unsafe {
            serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, std::ptr::null_mut()).ok()?;
            let blob = blob.ok_or_else(fail)?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            device.CreateRootSignature(0, bytes)
        }
    }

    fn make_buffer(
        device: &ID3D12Device,
        heap_type: D3D12_HEAP_TYPE,
        bytes: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut res: Option<ID3D12Resource> = None;
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res) }?;
        res.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))
    }

    struct Clamp {
        device_raw: usize,
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        zero16: ID3D12Resource,
    }

    // SAFETY: D3D12 objects are free-threaded; access is synchronized by `CLAMP`.
    unsafe impl Send for Clamp {}

    static CLAMP: Mutex<Option<Clamp>> = Mutex::new(None);

    fn get_or_make_clamp(device: &ID3D12Device) -> Option<(ID3D12RootSignature, ID3D12PipelineState, ID3D12Resource)> {
        let dev_raw = device.as_raw() as usize;
        let mut guard = CLAMP.lock().unwrap_or_else(|e| e.into_inner());
        if guard.as_ref().is_none_or(|c| c.device_raw != dev_raw) {
            let made = (|| -> windows::core::Result<Clamp> {
                let cs = compile_cs()?;
                let root = build_root_signature(device)?;
                let pso: ID3D12PipelineState = unsafe {
                    device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                        ..Default::default()
                    })
                }?;
                let zero16 = make_buffer(
                    device,
                    D3D12_HEAP_TYPE_UPLOAD,
                    256,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut ptr = std::ptr::null_mut();
                unsafe {
                    zero16.Map(0, None, Some(&mut ptr))?;
                    std::ptr::write_bytes(ptr.cast::<u8>(), 0, 256);
                    zero16.Unmap(0, None);
                }
                Ok(Clamp { device_raw: dev_raw, root, pso, zero16 })
            })();
            match made {
                Ok(c) => *guard = Some(c),
                Err(e) => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| xs_log!("indirect clamp: pipeline creation failed ({e}); ExecuteIndirect stays unclamped"));
                    return None;
                }
            }
        }
        guard.as_ref().map(|c| (c.root.clone(), c.pso.clone(), c.zero16.clone()))
    }

    struct ListBuffers {
        budget: Option<ID3D12Resource>,
        budget_cleared: bool,
        scratch: Option<(ID3D12Resource, u64)>,
    }

    // SAFETY: D3D12 resources are free-threaded; protected by `LIST_BUFFERS`.
    unsafe impl Send for ListBuffers {}

    static LIST_BUFFERS: Mutex<Option<HashMap<usize, ListBuffers>>> = Mutex::new(None);

    pub fn list_reset(list: usize) {
        if let Some(map) = LIST_BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut()
            && let Some(b) = map.get_mut(&list)
        {
            b.budget_cleared = false;
        }
    }

    pub fn list_forget(list: usize) {
        let gone = LIST_BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.remove(&list));
        if let Some(b) = gone {
            if let Some(res) = b.budget {
                unsafe { crate::retire::release(res.into_raw(), "indirect clamp budget") };
            }
            if let Some((res, _)) = b.scratch {
                unsafe { crate::retire::release(res.into_raw(), "indirect clamp scratch") };
            }
        }
    }

    fn transition(res: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(unsafe { std::mem::transmute_copy(res) }),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    fn uav_barrier(res: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(match res {
                        Some(r) => unsafe { std::mem::transmute_copy(r) },
                        None => None,
                    }),
                }),
            },
        }
    }

    /// Clamped scratch resource returned to `list_execute_indirect`, left in `INDIRECT_ARGUMENT`.
    pub struct ClampedCall {
        pub scratch: ID3D12Resource,
    }

    /// Clamps `max` records from `pc_arg` at `arg_off` (optionally bounded by `pc_count` at
    /// `count_off`) into the list's scratch buffer, restores clobbered compute/PSO bindings, and
    /// returns the scratch buffer in `INDIRECT_ARGUMENT` state. Returns `None` if clamping is off
    /// or not applicable (caller falls back to passthrough).
    ///
    /// # Safety
    /// `pc_list` is a live open PC `ID3D12GraphicsCommandList`; `pc_sig`, `pc_arg`, and (when
    /// non-null) `pc_count` are live PC D3D12 objects.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn try_clamp(
        list_key: usize,
        pc_list: *mut c_void,
        pc_sig: *mut c_void,
        max: u32,
        pc_arg: *mut c_void,
        arg_off: u64,
        pc_count: *mut c_void,
        count_off: u64,
    ) -> Option<ClampedCall> {
        if !enabled() || pc_list.is_null() || pc_sig.is_null() || pc_arg.is_null() {
            return None;
        }
        let Some(spec) = signature_spec(pc_sig as usize) else {
            PASSTHROUGH_CALLS.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let count_opt = if pc_count.is_null() { None } else { Some(count_off) };
        let Some(bytes) = validate_layout(spec, max, arg_off, count_opt) else {
            PASSTHROUGH_CALLS.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let list = unsafe { ID3D12GraphicsCommandList::from_raw_borrowed(&pc_list) }?;
        // Only DIRECT (0) and COMPUTE (2) lists can run compute clamp shaders.
        let list_ty = unsafe { list.GetType() };
        if list_ty != D3D12_COMMAND_LIST_TYPE_DIRECT && list_ty != D3D12_COMMAND_LIST_TYPE_COMPUTE {
            PASSTHROUGH_CALLS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let src = unsafe { ID3D12Resource::from_raw_borrowed(&pc_arg) }?;
        let count_res = if pc_count.is_null() { None } else { unsafe { ID3D12Resource::from_raw_borrowed(&pc_count) } };
        let mut dev: Option<ID3D12Device> = None;
        unsafe { list.GetDevice(&mut dev).ok()? };
        let device = dev?;
        let (root, pso, zero16) = get_or_make_clamp(&device)?;

        // Prepare per-list budget & scratch buffers; retire any replaced smaller scratch buffer.
        let (budget, need_clear, scratch, old_scratch) = {
            let mut guard = LIST_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
            let entry = guard.get_or_insert_with(HashMap::new).entry(list_key).or_insert(ListBuffers {
                budget: None,
                budget_cleared: false,
                scratch: None,
            });
            if entry.budget.is_none() {
                entry.budget = make_buffer(
                    &device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    256,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
                .ok();
            }
            let budget = entry.budget.clone()?;
            let mut old_scratch = None;
            if entry.scratch.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
                let want = bytes.next_power_of_two().max(MIN_SCRATCH);
                let fresh = make_buffer(
                    &device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    want,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )
                .ok()?;
                old_scratch = entry.scratch.replace((fresh, want)).map(|(r, _)| r);
            }
            let scratch = entry.scratch.as_ref()?.0.clone();
            let need_clear = !entry.budget_cleared;
            entry.budget_cleared = true;
            (budget, need_clear, scratch, old_scratch)
        };
        if let Some(old) = old_scratch {
            unsafe { crate::retire::release(old.into_raw(), "indirect clamp scratch") };
        }

        let (src_base, dst_gpu, budget_gpu) = unsafe {
            (src.GetGPUVirtualAddress(), scratch.GetGPUVirtualAddress(), budget.GetGPUVirtualAddress())
        };
        if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget_gpu.is_multiple_of(4) {
            return None;
        }
        let (src_gpu, args_offset) = match arg_off.checked_add(bytes).filter(|&end| end <= u64::from(u32::MAX)) {
            Some(_) => (src_base, arg_off as u32),
            None => (src_base.checked_add(arg_off)?, 0u32),
        };
        let (count_gpu, has_count, count_offset) = match count_res {
            Some(buf) => {
                let base = unsafe { buf.GetGPUVirtualAddress() };
                if !base.is_multiple_of(4) {
                    return None;
                }
                match count_off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                    Some(_) => (base, 1u32, count_off as u32),
                    None => (base.checked_add(count_off)?, 1u32, 0u32),
                }
            }
            None => (src_gpu, 0u32, 0u32),
        };

        let constants = [spec.kind, spec.stride, max, args_offset, has_count, count_offset];
        unsafe {
            if need_clear {
                list.ResourceBarrier(&[transition(&budget, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)]);
                list.CopyBufferRegion(&budget, 0, &zero16, 0, 16);
                list.ResourceBarrier(&[transition(&budget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            }

            let want_scoped = scoped();
            let before = if crate::uav_barrier_merge::enabled() {
                crate::uav_barrier_merge::is_pending(list_key)
            } else {
                true
            };
            let mut pre = Vec::with_capacity(3);
            if before {
                if want_scoped {
                    pre.push(uav_barrier(Some(src)));
                    if let Some(c) = count_res.filter(|c| c.as_raw() != src.as_raw()) {
                        pre.push(uav_barrier(Some(c)));
                    }
                } else {
                    pre.push(uav_barrier(None));
                    crate::uav_barrier_merge::clear_pending(list_key);
                }
            }
            pre.push(transition(&scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            list.ResourceBarrier(&pre);
            crate::uav_barrier_merge::note_clamp(u64::from(before), 1 + u64::from(!before));

            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, src_gpu);
            list.SetComputeRootShaderResourceView(2, count_gpu);
            list.SetComputeRootUnorderedAccessView(3, dst_gpu);
            list.SetComputeRootUnorderedAccessView(4, budget_gpu);
            list.Dispatch(max.div_ceil(64), 1, 1);

            // Transition orders clamp output for ExecuteIndirect (no redundant global UAV barrier).
            list.ResourceBarrier(&[transition(&scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)]);

            // Restore clobbered compute root signature, compute root params, and PSO.
            restore_list_state(list_key, list);
        }

        let calls = CLAMPED_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let recs = CLAMPED_RECORDS.fetch_add(u64::from(max), Ordering::Relaxed) + u64::from(max);
        if !FIRST_CLAMP_LOGGED.swap(true, Ordering::Relaxed) {
            xs_log!(
                "indirect clamp: active (DECANT_D3D12X_INDIRECT_CLAMP=1, scoped={}), first clamped ExecuteIndirect (kind {}, stride {}, max {})",
                scoped(),
                spec.kind,
                spec.stride,
                max
            );
        } else if calls.is_multiple_of(LOG_EVERY) {
            let pass = PASSTHROUGH_CALLS.load(Ordering::Relaxed);
            xs_log!("indirect clamp: {calls} calls clamped ({recs} max records, {pass} passthrough)");
        }
        Some(ClampedCall { scratch })
    }

    unsafe fn restore_list_state(list_key: usize, list: &ID3D12GraphicsCommandList) {
        let snap = {
            let guard = LIST_SAVED.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().and_then(|m| m.get(&list_key)).map(|s| (s.pso, s.compute_rs, s.compute_params.clone()))
        };
        let Some((pso, rs, params)) = snap else { return };
        unsafe {
            if rs != 0 {
                let raw = rs as *mut c_void;
                if let Some(sig) = ID3D12RootSignature::from_raw_borrowed(&raw) {
                    list.SetComputeRootSignature(sig);
                    for (idx, param) in params.iter().enumerate() {
                        let Some(p) = param else { continue };
                        let i = idx as u32;
                        match p {
                            ComputeParam::Table(handle) => list.SetComputeRootDescriptorTable(i, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: *handle }),
                            ComputeParam::Constants(vals) => {
                                if !vals.is_empty() {
                                    list.SetComputeRoot32BitConstants(i, vals.len() as u32, vals.as_ptr().cast(), 0);
                                }
                            }
                            ComputeParam::Va(RootVaKind::Cbv, gpu) => list.SetComputeRootConstantBufferView(i, *gpu),
                            ComputeParam::Va(RootVaKind::Srv, gpu) => list.SetComputeRootShaderResourceView(i, *gpu),
                            ComputeParam::Va(RootVaKind::Uav, gpu) => list.SetComputeRootUnorderedAccessView(i, *gpu),
                        }
                    }
                }
            }
            if pso != 0 {
                let raw = pso as *mut c_void;
                if let Some(p) = ID3D12PipelineState::from_raw_borrowed(&raw) {
                    list.SetPipelineState(p);
                }
            }
        }
    }

    /// Returns the scratch buffer back to `COMMON` state right after `ExecuteIndirect`.
    ///
    /// # Safety
    /// `pc_list` is the same live open PC `ID3D12GraphicsCommandList` passed to [`try_clamp`].
    pub unsafe fn finish_clamp(pc_list: *mut c_void, call: ClampedCall) {
        if let Some(list) = unsafe { ID3D12GraphicsCommandList::from_raw_borrowed(&pc_list) } {
            unsafe {
                list.ResourceBarrier(&[transition(
                    &call.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
    }
}

#[cfg(windows)]
pub use win::{ClampedCall, finish_clamp, try_clamp};

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[test]
    fn command_signature_desc_translates_dispatch_x_and_extracts_clamp_spec() {
        let args: [[u32; 4]; 1] = [[INDIRECT_ARGUMENT_DISPATCH_X, 0, 0, 0]];
        let mut raw = [0u8; 24];
        raw[0..4].copy_from_slice(&12u32.to_le_bytes()); // ByteStride = 12
        raw[4..8].copy_from_slice(&1u32.to_le_bytes()); // NumArgumentDescs = 1
        raw[8..16].copy_from_slice(&(args.as_ptr() as u64).to_le_bytes());
        raw[16..20].copy_from_slice(&0u32.to_le_bytes()); // NodeMask = 0

        // SAFETY: stack buffer with live args array.
        let out = unsafe { translate_command_signature_desc(raw.as_ptr()) };
        assert_eq!(out.args.len(), 1);
        assert_eq!(out.args[0][0], PC_INDIRECT_ARGUMENT_DISPATCH);
        assert!(!out.needs_root);
        assert_eq!(out.spec, Some(ClampSpec { kind: 2, stride: 12 }));
        // Original caller memory must NOT be mutated.
        assert_eq!(args[0][0], INDIRECT_ARGUMENT_DISPATCH_X);
    }

    #[test]
    fn root_changing_signature_sets_needs_root_and_skips_simple_clamp() {
        let args: [[u32; 4]; 2] = [[5, 0, 0, 1], [1, 0, 0, 0]]; // CONSTANT + DRAW_INDEXED
        let mut raw = [0u8; 24];
        raw[0..4].copy_from_slice(&24u32.to_le_bytes());
        raw[4..8].copy_from_slice(&2u32.to_le_bytes());
        raw[8..16].copy_from_slice(&(args.as_ptr() as u64).to_le_bytes());

        // SAFETY: stack buffer with live args array.
        let out = unsafe { translate_command_signature_desc(raw.as_ptr()) };
        assert!(out.needs_root);
        assert_eq!(out.spec, None, "multi-arg root-changing signatures pass through unclamped");
    }

    #[test]
    fn layout_validation_checks_stride_alignment_and_bounds() {
        let draw = ClampSpec { kind: 0, stride: 16 };
        let indexed = ClampSpec { kind: 1, stride: 20 };
        let mesh = ClampSpec { kind: 10, stride: 12 };
        assert_eq!(validate_layout(draw, 4, 0, None), Some(64));
        assert_eq!(validate_layout(indexed, 2, 4, Some(8)), Some(40));
        assert_eq!(validate_layout(mesh, 10, 0, None), Some(120));
        assert_eq!(validate_layout(draw, 0, 0, None), None, "zero records");
        assert_eq!(validate_layout(ClampSpec { kind: 1, stride: 16 }, 1, 0, None), None, "stride < min 20");
        assert_eq!(validate_layout(draw, 1, 2, None), None, "unaligned src offset");
        assert_eq!(validate_layout(draw, 1, 0, Some(2)), None, "unaligned count offset");
        assert_eq!(validate_layout(ClampSpec { kind: 99, stride: 16 }, 1, 0, None), None, "unsupported kind");
    }
}
