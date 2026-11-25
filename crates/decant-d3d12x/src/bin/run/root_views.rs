//! Section 8 of the test program (with `DECANT_D3D12X_TITLE_MEM=1` and `xmem.dll`): a root SRV at a
//! title address is read by a shader through the proxy, as the title's translated programs read root
//! SRVs `t0` and `t4` of space 3: a header dword at byte 0, then data at the offset it names, as a
//! raw buffer. The title memory is an arena commit of `xmem.dll`, the root signature, the pipeline,
//! the list, the queue and the fence are made through console slots, and the result is written
//! through a root UAV at a buffer placed at a title address (device slot 44), copied to a readback
//! buffer. A root SRV at an address no PC resource holds reads the zeroed fallback buffer and is
//! counted in `DECANT_CALLS`.

use std::ffi::{OsStr, c_void};

use decant_testkit::Library;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, Interface, PCSTR};

use super::{Test, VirtualAlloc, XFree, XboxCreate, hr, release_if, vt, xmem_path};

const IID_DEVICE11: GUID = GUID::from_u128(0x5405c344_d457_444e_b4dd_2366e45aee39);

/// The shader: the header dword of the title's buffer, then the two dwords at the offset it names.
const SHADER: &str = "ByteAddressBuffer title : register(t0, space3);\n\
RWByteAddressBuffer result : register(u0);\n\
[numthreads(1, 1, 1)] void main() { uint at = title.Load(0); result.Store4(0, uint4(at, title.Load(at), title.Load(at + 4), 0x600D)); }\n";

fn compile() -> Option<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source is a valid buffer of the length passed and the output a valid location.
    let made = unsafe { D3DCompile(SHADER.as_ptr().cast(), SHADER.len(), PCSTR::null(), None, None, PCSTR(c"main".as_ptr().cast()), PCSTR(c"cs_5_1".as_ptr().cast()), 0, 0, &mut code, None) };
    let code = code.filter(|_| made.is_ok())?;
    // SAFETY: the blob owns `GetBufferSize` bytes while it lives.
    Some(unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) }.to_vec())
}

/// The root signature: a root SRV `t0` of space 3 and a root UAV `u0`.
fn root_signature_blob() -> Option<Vec<u8>> {
    let view = |kind, register, space| D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: register, RegisterSpace: space } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0, 3), view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)];
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 2, pParameters: params.as_ptr(), ..Default::default() };
    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: the description and its parameters live for the call.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None) }.ok()?;
    let blob = blob?;
    // SAFETY: as in `compile`.
    Some(unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }.to_vec())
}

/// A `D3D12_RESOURCE_DESC` of a buffer of `bytes` bytes with `flags`, as its 56 bytes.
fn buffer_desc(bytes: u64, flags: u32) -> [u8; 0x38] {
    let mut d = [0u8; 0x38];
    d[0..4].copy_from_slice(&1u32.to_le_bytes());
    d[0x10..0x18].copy_from_slice(&bytes.to_le_bytes());
    d[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
    d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
    d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
    d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    d[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
    d[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
    d
}

type CreateRs = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> i32;
type CreateCompute = unsafe extern "system" fn(*mut c_void, *const c_void, u64, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type PlacedAt = unsafe extern "system" fn(*mut c_void, u64, *const c_void, u32, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type Committed = unsafe extern "system" fn(*mut c_void, *const c_void, u32, *const c_void, u32, *const c_void, *const GUID, *mut *mut c_void) -> i32;
type CreateQueue = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const GUID, *mut *mut c_void) -> i32;
type CreateAlloc = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;
type CreateListX = unsafe extern "system" fn(*mut c_void, *const c_void, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
type CreateFence = unsafe extern "system" fn(*mut c_void, u64, i32, *const GUID, *mut *mut c_void) -> i32;
type SetObject = unsafe extern "system" fn(*mut c_void, *mut c_void);
type SetRootVa = unsafe extern "system" fn(*mut c_void, u32, u64);
type Dispatch = unsafe extern "system" fn(*mut c_void, u32, u32, u32);
type Barriers = unsafe extern "system" fn(*mut c_void, u32, *const c_void);
type Copy = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void);
type NoArgs = unsafe extern "system" fn(*mut c_void) -> i32;
type Execute = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
type Signal = unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32;
type Completed = unsafe extern "system" fn(*mut c_void) -> u64;
type Map = unsafe extern "system" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> i32;

/// A transition barrier of all subresources of `res`, as its 32 bytes.
fn transition(res: *mut c_void, before: u32, after: u32) -> [u64; 4] {
    // Type and flags, the resource, subresource and state before, state after.
    [0, res as u64, u64::from(u32::MAX) | (u64::from(before) << 32), u64::from(after)]
}

pub(super) fn root_view_reads(t: &mut Test, xs_path: &OsStr, xs_create: XboxCreate) {
    const ARENA: usize = 4 << 20;
    if !std::env::var_os("DECANT_D3D12X_TITLE_MEM").is_some_and(|v| v == "1") {
        println!("skip  root view checks: DECANT_D3D12X_TITLE_MEM=1 runs them");
        return;
    }
    let Some(xmem) = xmem_path(xs_path).and_then(|p| Library::open(p.as_os_str()).ok()) else {
        println!("skip  no xmem.dll next to {}: root view checks skipped", xs_path.to_string_lossy());
        return;
    };
    // SAFETY: each type is the signature of the export it names.
    let (Some(alloc), Some(free)) = (unsafe { xmem.get::<VirtualAlloc>("XMemVirtualAlloc") }, unsafe { xmem.get::<XFree>("XMemFree") }) else {
        println!("FAIL  xmem.dll lacks XMemVirtualAlloc or XMemFree");
        t.failures += 1;
        return;
    };
    let (Some(shader), Some(rs_blob)) = (compile(), root_signature_blob()) else {
        println!("skip  no shader compiler or root signature serializer: root view checks skipped");
        return;
    };
    let mut dev: *mut c_void = std::ptr::null_mut();
    // SAFETY: the export takes (adapter, parameters, riid, device).
    if unsafe { xs_create(std::ptr::null_mut(), std::ptr::null(), &IID_DEVICE11, &mut dev) } < 0 || dev.is_null() {
        println!("skip  no D3D12 12_0 device on this machine: root view checks skipped");
        return;
    }
    // SAFETY: `dev` is the console device proxy, every slot is called with the console shapes; the
    // title memory is an arena of xmem.dll written inside its bounds; each object is released once.
    unsafe {
        // The title's data: an arena commit with a console page bit, as the title's, a header dword at
        // byte 0 naming offset 3 MiB, data there: beyond the first reach of title memory (1 MiB), so
        // the pages up to the end of the reservation must reach the GPU.
        let arena = alloc(std::ptr::null_mut(), ARENA, 0x2000_3000, 0x183).cast::<u8>();
        check!(t, !arena.is_null(), "xmem gives an arena for the title's root SRV data");
        if arena.is_null() {
            release_if(dev);
            return;
        }
        let title = arena.add(0x1_0000);
        title.cast::<u32>().write_volatile(0x30_0000);
        title.add(0x30_0000).cast::<u32>().write_volatile(0xC0FF_EE01);
        title.add(0x30_0004).cast::<u32>().write_volatile(0xC0FF_EE02);
        let title_va = title as u64;
        let not_held = [0x40u32, 7, 7, 7];
        let out_va = 0x7E00_0000_0000u64;

        let mut rs = std::ptr::null_mut();
        let r = vt::<CreateRs>(dev, 16)(dev, 0, rs_blob.as_ptr().cast(), rs_blob.len(), &ID3D12RootSignature::IID, &mut rs);
        check!(t, r >= 0 && !rs.is_null(), "device slot 16 makes a root signature with a root SRV t0 of space 3 (hr=0x{:08x})", hr(r));
        let mut cdesc = [0u64; 7];
        (cdesc[0], cdesc[1], cdesc[2]) = (rs as u64, shader.as_ptr() as u64, shader.len() as u64);
        let mut pso = std::ptr::null_mut();
        let r = vt::<CreateCompute>(dev, 62)(dev, cdesc.as_ptr().cast(), 0, std::ptr::null(), &ID3D12PipelineState::IID, &mut pso);
        check!(t, r >= 0 && !pso.is_null(), "device slot 62 answers the compute pipeline (hr=0x{:08x})", hr(r));
        let mut placed = std::ptr::null_mut();
        let out_desc = buffer_desc(256, 4);
        let r = vt::<PlacedAt>(dev, 44)(dev, out_va, out_desc.as_ptr().cast(), 0, std::ptr::null(), &ID3D12Resource::IID, &mut placed);
        check!(t, r >= 0 && !placed.is_null(), "device slot 44 places the result buffer at a title address (hr=0x{:08x})", hr(r));
        let mut readback = std::ptr::null_mut();
        let props: [u32; 5] = [3, 0, 0, 1, 1];
        let rb_desc = buffer_desc(256, 0);
        let r = vt::<Committed>(dev, 27)(dev, props.as_ptr().cast(), 0, rb_desc.as_ptr().cast(), 0x400, std::ptr::null(), &ID3D12Resource::IID, &mut readback);
        check!(t, r >= 0 && !readback.is_null(), "device slot 27 makes a readback buffer (hr=0x{:08x})", hr(r));
        let qd = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
        let (mut q, mut allocator, mut list, mut fence) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        vt::<CreateQueue>(dev, 8)(dev, &qd, &ID3D12CommandQueue::IID, &mut q);
        vt::<CreateAlloc>(dev, 9)(dev, 0, &ID3D12CommandAllocator::IID, &mut allocator);
        let params = [0u32; 4];
        if !allocator.is_null() {
            vt::<CreateListX>(dev, 73)(dev, params.as_ptr().cast(), allocator, std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list);
        }
        vt::<CreateFence>(dev, 36)(dev, 0, 0, &ID3D12Fence::IID, &mut fence);
        let ready = [rs, pso, placed, readback, q, list, fence].iter().all(|o| !o.is_null());
        check!(t, ready, "queue, allocator, list and fence for the root view checks");
        if ready {
            // Twice: the title's root SRV, then one at an address no PC resource holds; each result
            // to its own 16 bytes of the placed buffer.
            for (i, src) in [title_va, not_held.as_ptr() as u64].into_iter().enumerate() {
                vt::<SetObject>(list, 33)(list, rs);
                vt::<SetObject>(list, 25)(list, pso);
                vt::<SetRootVa>(list, 43)(list, 0, src);
                vt::<SetRootVa>(list, 45)(list, 1, out_va + 16 * i as u64);
                vt::<Dispatch>(list, 14)(list, 1, 1, 1);
            }
            let to_copy = transition(placed, 0x8, 0x800);
            vt::<Barriers>(list, 26)(list, 1, to_copy.as_ptr().cast());
            vt::<Copy>(list, 17)(list, readback, placed);
            let r = vt::<NoArgs>(list, 9)(list);
            check!(t, r >= 0, "the list with the root view reads closes (hr=0x{:08x})", hr(r));
            vt::<Execute>(q, 10)(q, 1, &list);
            // Newer layout (the device is an ID3D12Device11): Signal is queue slot 16.
            let r = vt::<Signal>(q, 16)(q, fence, 1);
            let mut done = 0;
            for _ in 0..500 {
                done = vt::<Completed>(fence, 8)(fence);
                if done >= 1 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            check!(t, r >= 0 && done >= 1, "the GPU ran the root view reads (fence {done})");
            let mut mapped: *mut c_void = std::ptr::null_mut();
            let r = vt::<Map>(readback, 8)(readback, 0, std::ptr::null(), &mut mapped);
            if r >= 0 && !mapped.is_null() && done >= 1 {
                let words = std::slice::from_raw_parts(mapped.cast::<u32>(), 8);
                check!(t, words[..3] == [0x30_0000, 0xC0FF_EE01, 0xC0FF_EE02], "a root SRV t0 of space 3 at a title address reads the title's header and the data 3 MiB on ({:x?})", &words[..3]);
                check!(t, words[3] == 0x600D && words[7] == 0x600D, "both dispatches wrote their result through the placed root UAV ({:x?})", [words[3], words[7]]);
                check!(t, words[4..7] == [0, 0, 0], "a root SRV at an address no PC resource holds reads the zero buffer ({:x?})", &words[4..7]);
            } else {
                check!(t, false, "the readback buffer maps (hr=0x{:08x})", hr(r));
            }
        }
        if let Some(entries) = super::pipeline_slots::calls() {
            check!(t, entries.iter().any(|e| e.kind == decant_log::calls::Kind::Stub && e.name.contains("root SRV at a title address in no PC resource")), "DECANT_CALLS counts the root SRV that got the zero buffer");
        }
        for o in [list, allocator, q, fence, readback, placed, pso, rs] {
            release_if(o);
        }
        free(arena.cast(), 0);
        release_if(dev);
    }
}
