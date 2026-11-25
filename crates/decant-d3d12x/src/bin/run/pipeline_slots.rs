//! Section 7 of the test program: the console slots the title calls while it boots, through the proxy
//! of a device created as the title creates it (`ID3D12Device11`), with the argument shapes the
//! title passes: the pipeline cache loads (device 57 and 67), a mesh pipeline from a stream (device
//! 86) bound on a list (25) with a mesh dispatch (129), the pipeline syncs (list 131,
//! 132) and the depth order switch (list 75). With `DECANT_CALLS` on, the shared table of
//! unanswered questions is read back: none of these slots may be counted as a gap any more.

use std::ffi::c_void;

use decant_log::calls::{Entry, Kind};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12PipelineState};
use windows::core::{GUID, Interface};

use super::{Test, XboxCreate, hr, release_if, vt};

/// `ID3D12Device11`, the IID the title creates its device with.
const IID_DEVICE11: GUID = GUID::from_u128(0x5405c344_d457_444e_b4dd_2366e45aee39);
/// `D3D12_ERROR_DRIVER_VERSION_MISMATCH`.
const DRIVER_VERSION_MISMATCH: u32 = 0x887E_0002;

type Deserialize = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type CreateFromDesc = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type CreateAlloc = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;
type CreateListX = unsafe extern "system" fn(*mut c_void, *const c_void, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
type SetPipeline = unsafe extern "system" fn(*mut c_void, *mut c_void);
type DispatchMesh = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
type NoArgs = unsafe extern "system" fn(*mut c_void) -> i32;

/// Appends one pipeline stream subobject: type, padding to the payload's alignment, payload,
/// padding to the next pointer boundary.
fn push(stream: &mut Vec<u8>, ty: u32, align: usize, payload: &[u8]) {
    stream.extend_from_slice(&ty.to_le_bytes());
    while !stream.len().is_multiple_of(align) {
        stream.push(0);
    }
    stream.extend_from_slice(payload);
    while !stream.len().is_multiple_of(8) {
        stream.push(0);
    }
}

/// A shader as a `D3D12_SHADER_BYTECODE`.
fn bytecode(bytes: &[u8]) -> Vec<u8> {
    [(bytes.as_ptr() as u64).to_le_bytes(), (bytes.len() as u64).to_le_bytes()].concat()
}

/// The entries of this process's `DECANT_CALLS` table, `None` when the counter is off.
pub(super) fn calls() -> Option<Vec<Entry>> {
    if !decant_log::calls::switch_on(std::env::var(decant_log::calls::ENV).ok().as_deref()) {
        return None;
    }
    let shared = decant_log::calls::Shared::open(std::process::id())?;
    Some(shared.table().snapshot())
}

/// Whether `entries` count `what` (a part of the question) as answered with `kind`.
fn counted(entries: &[Entry], kind: Kind, what: &str) -> bool {
    entries.iter().any(|e| e.kind == kind && e.name.contains(what))
}

pub(super) fn console_pipeline_slots(t: &mut Test, xs_create: XboxCreate) {
    let mut dev: *mut c_void = std::ptr::null_mut();
    // SAFETY: the export takes (adapter, parameters, riid, device).
    let r = unsafe { xs_create(std::ptr::null_mut(), std::ptr::null(), &IID_DEVICE11, &mut dev) };
    if r < 0 || dev.is_null() {
        println!("skip  no D3D12 12_0 device on this machine (hr=0x{:08x}): console pipeline slots skipped", hr(r));
        return;
    }
    // SAFETY: `dev` is the console device proxy, every slot is called with the shapes the title uses.
    unsafe {
        // Device 57 and 67: (pointer to the cache entry, whose first field points at the serialized
        // data, IID_ID3D12PipelineState, out). Every entry is a miss with a cleared result.
        let data = [0xA5u8; 256];
        let entry: [usize; 2] = [data.as_ptr() as usize, data.len()];
        for (slot, name) in [(57usize, "graphics"), (67, "compute")] {
            let mut out: *mut c_void = data.as_ptr().cast_mut().cast();
            let r = vt::<Deserialize>(dev, slot)(dev, entry.as_ptr().cast(), &ID3D12PipelineState::IID, &mut out);
            check!(t, hr(r) == DRIVER_VERSION_MISMATCH && out.is_null(), "device slot {slot} (deserialize {name} pipeline) answers a cache miss with a cleared result (hr=0x{:08x})", hr(r));
        }
        // Device 86 with a mesh stream as the title passes it: root signature, mesh and pixel shader,
        // one target. The pipeline exists at once; made at its first bind, the PC runtime refuses the
        // console mesh bytes, and the list leaves out the mesh dispatch.
        let (ms, ps) = (b"console mesh shader bytes".to_vec(), b"console pixel shader bytes".to_vec());
        let mut stream = Vec::new();
        push(&mut stream, 0, 8, &0u64.to_le_bytes());
        push(&mut stream, 24, 8, &bytecode(&ms));
        push(&mut stream, 2, 8, &bytecode(&ps));
        let mut targets = [0u8; 36];
        targets[0..4].copy_from_slice(&28u32.to_le_bytes());
        targets[32..36].copy_from_slice(&1u32.to_le_bytes());
        push(&mut stream, 15, 4, &targets);
        let stream_desc: [usize; 2] = [stream.len(), stream.as_ptr() as usize];
        let mut mesh: *mut c_void = std::ptr::null_mut();
        let r = vt::<CreateFromDesc>(dev, 86)(dev, stream_desc.as_ptr().cast(), &ID3D12PipelineState::IID, &mut mesh);
        check!(t, r == 0 && !mesh.is_null(), "device slot 86 answers a mesh stream with a pipeline object (hr=0x{:08x})", hr(r));
        let mut alloc: *mut c_void = std::ptr::null_mut();
        let r = vt::<CreateAlloc>(dev, 9)(dev, 0, &ID3D12CommandAllocator::IID, &mut alloc);
        let mut list: *mut c_void = std::ptr::null_mut();
        let params = [0u32; 4];
        let r = if r >= 0 { vt::<CreateListX>(dev, 73)(dev, params.as_ptr().cast(), alloc, std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list) } else { r };
        check!(t, r >= 0 && !list.is_null(), "device slot 73 creates a direct list for the mesh checks (hr=0x{:08x})", hr(r));
        if !list.is_null() && !mesh.is_null() {
            vt::<SetPipeline>(list, 25)(list, mesh);
            vt::<DispatchMesh>(list, 129)(list, 4, 1, 1);
            // List 131 and 132: the pipeline syncs at the top and the bottom of the pipe, with no
            // memory named, as the title calls them right after creating its first lists.
            for slot in [131usize, 132] {
                vt::<unsafe extern "system" fn(*mut c_void, u32, u32, u64, u64)>(list, slot)(list, 0, 0, 0, 0);
            }
            // List 75: the pixel shader depth order switch, 1 then 0 as the title flips it.
            for enable in [1u32, 0] {
                vt::<unsafe extern "system" fn(*mut c_void, u32)>(list, 75)(list, enable);
            }
            let r = vt::<NoArgs>(list, 9)(list);
            check!(t, r >= 0, "a list that bound the refused mesh pipeline and dispatched (129) closes (hr=0x{:08x})", hr(r));
        }
        release_if(list);
        release_if(alloc);
        release_if(mesh);
        if let Some(entries) = calls() {
            // The counter itself works: section 4 called the unknown device slot 64.
            check!(t, counted(&entries, Kind::NotImplemented, "console device slot 64"), "DECANT_CALLS counts the unknown device slot 64 of section 4");
            for slot in [57, 67] {
                let what = format!("device slot {slot} ");
                let any = [Kind::NotImplemented, Kind::Stub, Kind::Error].iter().any(|k| counted(&entries, *k, &what));
                check!(t, !any, "DECANT_CALLS: the cache miss of device slot {slot} is an answer, not counted as a gap");
            }
            check!(t, !counted(&entries, Kind::NotImplemented, "device slot 86"), "DECANT_CALLS: device slot 86 is not counted as not implemented for a mesh stream");
            for slot in [131, 132] {
                check!(t, !counted(&entries, Kind::NotImplemented, &format!("command list slot {slot}")), "DECANT_CALLS: list slot {slot} (pipeline sync) is not counted as not implemented");
            }
            check!(t, !counted(&entries, Kind::NotImplemented, "command list slot 75"), "DECANT_CALLS: list slot 75 (pixel shader depth order) is not counted as not implemented");
            check!(t, !counted(&entries, Kind::NotImplemented, "command list slot 129"), "DECANT_CALLS: list slot 129 (DispatchMesh) is not counted as not implemented");
            check!(t, counted(&entries, Kind::Error, "mesh pipeline refused at bind"), "DECANT_CALLS: the mesh pipeline is refused by name at its first bind");
        } else {
            println!("skip  DECANT_CALLS is off: the table of unanswered questions is not checked");
        }
        release_if(dev);
    }
}
