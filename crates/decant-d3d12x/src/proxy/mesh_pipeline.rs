//! Mesh pipelines (console device slot 86, a stream with a mesh shader) and mesh dispatches (console
//! list slots 127, 129 and 100).
//!
//! A mesh pipeline is made on first bind like the graphics ones (`lazy_pso`): its shaders go to the
//! direct path (`direct_pso`, the mesh and the pixel shader in one call) when it is on, else to the
//! DXIL override files, then to `ID3D12Device2::CreatePipelineState` as a PC stream
//! (`mesh_pso::pc_stream`). A pipeline the PC runtime refuses (the console bytes of an
//! untranslated mesh shader) is refused by name at its first bind, its list leaves its work out,
//! and the refusal is counted in `DECANT_CALLS` as "mesh pipeline refused at bind".
//!
//! Mesh dispatches go to `ID3D12GraphicsCommandList6::DispatchMesh` with the user data mirror
//! prepared as for a draw.

use super::*;
use crate::mesh_pso::{AS_AT, MESH_DESC_SIZE, MS_AT, PS_AT};
use crate::stream_pso::GRAPHICS_DESC_SIZE;

/// `ID3D12Device2`, the interface with `CreatePipelineState` from a stream.
const IID_DEVICE2: GUID = GUID::from_u128(0x30baa41e_b15b_475c_a0bb_1af5c5b64328);
/// The PC slot of `ID3D12Device2::CreatePipelineState`.
pub(super) const PC_CREATE_PIPELINE_STATE: usize = 47;
/// `ID3D12GraphicsCommandList6`, the interface with `DispatchMesh`.
const IID_GRAPHICS_COMMAND_LIST6: GUID = GUID::from_u128(0xc3827890_e548_4cfa_96cf_5689a9370f80);
/// The PC slot of `ID3D12GraphicsCommandList6::DispatchMesh`.
pub(super) const PC_DISPATCH_MESH: usize = 79;
/// `D3D12_FEATURE_D3D12_OPTIONS7`, whose first field is the mesh shader tier.
const FEATURE_D3D12_OPTIONS7: u32 = 32;

/// The mesh shader tier of the PC device: unknown until a device is created, then the tier
/// (0 when the GPU has no mesh shaders).
static MESH_TIER: AtomicI32 = AtomicI32::new(-1);

/// Reads and logs the mesh shader tier of the PC device `dev` (`D3D12_OPTIONS7`), once per device
/// creation. Without mesh shaders every mesh pipeline is refused by name at its first bind.
///
/// # Safety
/// `dev` is a live PC device.
pub(crate) unsafe fn note_mesh_support(dev: Object) {
    type CheckFeature = unsafe extern "system" fn(Object, u32, *mut c_void, u32) -> HRESULT;
    let mut options7 = [0u32; 2];
    // SAFETY: by the contract above, slot 13 is CheckFeatureSupport, the structure is 8 bytes.
    let hr = unsafe { method::<CheckFeature>(dev, 13)(dev, FEATURE_D3D12_OPTIONS7, options7.as_mut_ptr().cast(), 8) };
    let tier = if hr >= 0 { options7[0] as i32 } else { 0 };
    MESH_TIER.store(tier, Ordering::Relaxed);
    if tier == 0 {
        xs_log!("PC device: no mesh shader support (D3D12_OPTIONS7 hr=0x{:08X}, tier {}): the title's mesh pipelines will be refused at their first bind", hr as u32, options7[0]);
    } else {
        xs_log!("PC device: mesh shader tier {tier}");
    }
}

/// Answers a mesh pipeline creation (slot 86) with a pipeline made on first bind, or makes it now
/// when that is off.
///
/// # Safety
/// `this` is a live device proxy, every pointer in `desc` readable as the title passed it, `out`
/// writable.
pub(super) unsafe fn defer_mesh(this: *mut Proxy, desc: &[u8; MESH_DESC_SIZE], riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        if !crate::lazy_pso::enabled() {
            return mesh_now(this as usize, desc.as_ptr().cast(), riid, out);
        }
        crate::lazy_pso::defer(this as usize, crate::lazy_pso::copy_mesh(desc), riid, out, mesh_now)
    }
}

/// The stages of a mesh description with their override lookups (`dxil_override`), the description
/// pointed at the overrides that hit. The override buffers are returned and must outlive the PC
/// call.
fn override_mesh_shaders(m: &mut [u8; MESH_DESC_SIZE]) -> (Vec<Vec<u8>>, Vec<crate::pso_diag::Stage>) {
    let mut kept = Vec::new();
    let mut stages = Vec::new();
    let root_sig = u64::from_le_bytes(m[0..8].try_into().unwrap_or_default()) as usize;
    // The stage, its offset and its D3D12_SHADER_VISIBILITY.
    for (stage, at, visibility) in [("AS", AS_AT, 6u32), ("MS", MS_AT, 7), ("PS", PS_AT, 5)] {
        if !crate::mesh_pso::has_shader(&m[..], at) {
            continue;
        }
        let ptr = u64::from_le_bytes(m[at..at + 8].try_into().unwrap_or_default()) as *const u8;
        let len = u64::from_le_bytes(m[at + 8..at + 16].try_into().unwrap_or_default()) as usize;
        // SAFETY: the description's shader pointer and length name bytes it (or its copy) owns.
        let original = unsafe { std::slice::from_raw_parts(ptr, len) };
        let (replacement, outcome) = dxil_override::lookup_for_pipeline_outcome(original, root_sig, visibility);
        stages.push(crate::pso_diag::Stage { name: stage, hash: dxil_override::content_hash(original), outcome });
        let Some(replacement) = replacement else { continue };
        m[at..at + 8].copy_from_slice(&(replacement.as_ptr() as u64).to_le_bytes());
        m[at + 8..at + 16].copy_from_slice(&(replacement.len() as u64).to_le_bytes());
        kept.push(replacement);
    }
    (kept, stages)
}

/// A mesh pipeline made now: the overrides of its stages, then the PC stream creation.
///
/// # Safety
/// `this` is a live device proxy (as an address), `desc` a readable mesh description whose
/// pointers are readable, `out` writable.
unsafe fn mesh_now(this: usize, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static FAIL_LOGGED: AtomicI32 = AtomicI32::new(0);
    let this = this as *mut Proxy;
    if desc.is_null() || out.is_null() {
        return E_POINTER;
    }
    // SAFETY: by the contract above.
    unsafe { *out = std::ptr::null_mut() };
    // SAFETY: by the contract above.
    let mut m: Box<[u8; MESH_DESC_SIZE]> = Box::new(unsafe { std::ptr::read_unaligned(desc.cast()) });
    let (kept, stages) = override_mesh_shaders(&mut m);
    let full = Some(kept.len() == stages.len());
    let graphics: &[u8; GRAPHICS_DESC_SIZE] = m[..GRAPHICS_DESC_SIZE].try_into().expect("the graphics part");
    let hr = if MESH_TIER.load(Ordering::Relaxed) == 0 {
        E_NOTIMPL
    } else {
        let stream = crate::mesh_pso::pc_stream(&m);
        // D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes, pPipelineStateSubobjectStream }.
        let stream_desc: [usize; 2] = [stream.len(), stream.as_ptr() as usize];
        // SAFETY: `this` is a live device proxy; the interface found by QueryInterface is released.
        unsafe {
            let pc = (*this).pc;
            let mut d2: Object = std::ptr::null_mut();
            if query_object(pc, &IID_DEVICE2, &mut d2) < 0 || d2.is_null() {
                E_NOINTERFACE
            } else {
                let hr = method::<CreatePso>(d2, PC_CREATE_PIPELINE_STATE)(d2, stream_desc.as_ptr().cast(), riid, out);
                method::<Release>(d2, 2)(d2);
                hr
            }
        }
    };
    drop(kept);
    crate::rt_diag::pipeline(hr >= 0, full);
    if hr >= 0 {
        // SAFETY: `out` was checked above, the runtime wrote the new pipeline to it.
        crate::rt_diag::pipeline_made(unsafe { *out } as usize, || crate::pso_diag::describe_stages(&stages) + &crate::pso_diag::describe_state(graphics));
        if crate::kind_check::tracking() {
            // SAFETY: as above.
            crate::kind_check::pipeline_made(unsafe { *out } as usize, true, &crate::pso_diag::kind_stages(&stages));
        }
        return hr;
    }
    decant_log::calls::error("mesh pipeline refused at bind (console device slot 86)");
    if log_due(&FAIL_LOGGED) {
        let why = if MESH_TIER.load(Ordering::Relaxed) == 0 { "the PC GPU has no mesh shaders" } else { "a mesh shader without a translation" };
        xs_log!("console device slot 86 (mesh pipeline): refused at its first bind, hr=0x{:08X} ({why})", hr as u32);
    }
    // SAFETY: `out` was checked above and is writable.
    unsafe { crate::pso_diag::refused(hr, "mesh, slot 86", &stages, &graphics_targets(graphics), out) }
}

/// One mesh dispatch of `x` groups on the list `this`: the user data mirror as for a draw, then
/// `DispatchMesh(x, y, z)` on the PC list. A count other than 1 in Y or Z is logged: the translated
/// mesh shaders take their group number from `SV_GroupID.x`.
fn dispatch_mesh(this: *mut Proxy, slot: u32, x: u32, y: u32, z: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    static YZ: AtomicI32 = AtomicI32::new(0);
    static MISSING: AtomicBool = AtomicBool::new(false);
    if (y != 1 || z != 1) && log_due(&YZ) {
        xs_log!("console list slot {slot} (DispatchMesh {x}, {y}, {z}): Y or Z is not 1, the translated mesh shaders read SV_GroupID.x only");
    }
    if x == 0 || y == 0 || z == 0 || crate::pso_diag::skip(this as usize) {
        return;
    }
    if log_due(&N) {
        xs_log!("console list slot {slot} (DispatchMesh): ({x}, {y}, {z})");
    }
    // SAFETY: `this` is a live list proxy; the interface found by QueryInterface is released again.
    unsafe {
        let pc = (*this).pc;
        let mut l6: Object = std::ptr::null_mut();
        if query_object(pc, &IID_GRAPHICS_COMMAND_LIST6, &mut l6) < 0 || l6.is_null() {
            if !MISSING.swap(true, Ordering::SeqCst) {
                xs_log!("console list slot {slot} (DispatchMesh): the PC list has no ID3D12GraphicsCommandList6, mesh dispatches are left out");
            }
            return;
        }
        super::list_mirror::prepare(this, crate::user_mirror::Bind::Graphics);
        let _ = crate::rt_diag::draw(this as usize, "DispatchMesh", x);
        crate::uav_barrier_merge::mark_pending(this as usize);
        method::<unsafe extern "system" fn(Object, u32, u32, u32)>(l6, PC_DISPATCH_MESH)(l6, x, y, z);
        method::<Release>(l6, 2)(l6);
    }
}

/// Console list slot 127, older layout: `DispatchMesh(x, y, z)`. In the newer layout the slot is
/// not known (the method moved to 129) and is logged as unknown.
pub(super) extern "system" fn list_slot127(this: *mut Proxy, x: u32, y: u32, z: u32) {
    if crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 127);
    }
    dispatch_mesh(this, 127, x, y, z);
}

/// Console list slot 129, newer layout (older 127): `DispatchMesh(x, y, z)`.
pub(super) extern "system" fn list_dispatch_mesh_newer(this: *mut Proxy, x: u32, y: u32, z: u32) {
    if super::list_draws::newer_or_unknown(this, 129) {
        dispatch_mesh(this, 129, x, y, z);
    }
}

/// Console list slot 100, newer layout (older 98): `DispatchMeshX(x, flags, indirect arguments,
/// payload)`. A dispatch whose group count is in GPU memory (indirect) is logged and left out.
pub(super) extern "system" fn list_dispatch_mesh_x(this: *mut Proxy, x: u32, flags: u64, indirect: u64, _payload: u64) {
    static INDIRECT: AtomicI32 = AtomicI32::new(0);
    if !super::list_draws::newer_or_unknown(this, 100) {
        return;
    }
    if indirect != 0 {
        if log_due(&INDIRECT) {
            xs_log!("console list slot 100 (DispatchMeshX, flags {flags:#x}): the group count is in GPU memory at {indirect:#x}, not ported, left out");
        }
        return;
    }
    dispatch_mesh(this, 100, x, 1, 1);
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_interfaces_are_those_of_the_public_headers() {
        use windows::Win32::Graphics::Direct3D12::{ID3D12Device2, ID3D12GraphicsCommandList6};
        use windows::core::Interface;
        let same = |a: windows::core::GUID, b: super::GUID| (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4);
        assert!(same(ID3D12Device2::IID, super::IID_DEVICE2));
        assert!(same(ID3D12GraphicsCommandList6::IID, super::IID_GRAPHICS_COMMAND_LIST6));
    }
}
