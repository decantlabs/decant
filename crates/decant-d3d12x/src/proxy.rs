//! Console ABI proxies over PC D3D12 objects (issue #70). The Rust form of the former
//! `src/gdk/d3d12x_proxy.c`.
//!
//! A title compiled for the console calls the device, command lists, queues and descriptor heaps
//! through console vtables whose order differs from the PC one (see `abi` for the order and its
//! sources). Each proxy is an object whose vtable follows that order:
//!
//! * a relayed slot is a small stub that replaces `this` with the PC object and jumps to the PC
//!   method, which works for any signature because the arguments are left untouched,
//! * a Rust method handles IUnknown, the creation of further proxied objects, the slots whose
//!   signature differs from the PC and the slots whose return convention differs (structures
//!   returned by value, see below),
//! * any other slot logs its number with what is known about it and answers `E_NOTIMPL`.
//!
//! Command lists: console compiled callers may write commands straight into a buffer inside the
//! list object while a cursor at +0x18 is below its limit (+0x28 for a dispatch, +0x30 for root
//! arguments). With the cursor at the limit every command goes through the vtable. A proxied list
//! keeps the cursor at the limit and is large enough for the fields those callers touch (+0x140,
//! +0x344, +0x544). Measured in a console BVH builder library, and seen again in another retail title.
//!
//! Return conventions: the PC runtime implements its methods as C++ members, which always return a
//! structure through a hidden pointer passed after `this`. Console callers measured in another
//! retail title do otherwise. An 8 byte handle comes back in rax (descriptor heap 9 and 10), and a
//! larger structure goes through a result pointer passed in rcx, with `this` moving to rdx (device
//! 25 and 26). Those slots are methods with the console convention that call the PC one.
//!
//! Frame pacing (device slots 78 to 80) lives in `frame`. Extension point for the full layer
//! (Sassy, #70): the tables are data, PresentX goes through `DecantD3D12X_SetPresentHook`, and [`wrap`]
//! is where richer proxies (resources) would plug in. x86 64 only.
//!
//! PC methods are called by slot index, so no D3D12 binding is needed. Public PC slots:
//! ID3D12Object 3 GetPrivateData, 4 SetPrivateData, 5 SetPrivateDataInterface. ID3D12Device 8
//! CreateCommandQueue, 10 CreateGraphicsPipelineState, 11 CreateComputePipelineState, 12
//! CreateCommandList, 14 CreateDescriptorHeap, 25 GetResourceAllocationInfo, 26
//! GetCustomHeapProperties, 34 MakeResident, 35 Evict, 43 GetAdapterLuid. ID3D12CommandQueue 10
//! ExecuteCommandLists, 14 Signal. ID3D12GraphicsCommandList 16 CopyTextureRegion, 27 ExecuteBundle,
//! 28 SetDescriptorHeaps. ID3D12DescriptorHeap 9 and 10 Get{CPU,GPU}DescriptorHandleForHeapStart.
//! IDXGIFactory4 26 EnumAdapterByLuid.
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};

use windows_sys::Win32::Foundation::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, E_OUTOFMEMORY, E_POINTER, LUID, S_OK};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows_sys::core::{GUID, HRESULT};

use crate::abi::{ALLOCATOR_SLOTS, DEVICE_SLOTS, HEAP_SLOTS, Kind, LIST_SLOTS, QUEUE_SLOTS, RESOURCE_SLOTS, RuleKind, pc_resource_desc};
use crate::dxil_override;
use crate::frame;
use crate::log::{guid_at, p, xs_log};
use crate::mem_owners::{self, Charge};
use crate::resource_args::{self, BARRIER_BYTES, COPY_LOCATION_BYTES};
use crate::rts0;
use crate::views_at::MapAnswer;

mod heap_pages;
mod immediate_samplers;
mod list_draws;
mod list_gaps;
mod list_raster;
mod list_mem;
mod list_misc;
mod list_mirror;
mod mesh_pipeline;
#[cfg(test)]
mod pc_slots;
mod pso_cache;
mod root_views;

pub(crate) use mesh_pipeline::note_mesh_support;

// The byte sizes the resource argument rewriting assumes (`resource_args`) are the public d3d12.h ones.
const _: () = assert!(size_of::<windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER>() == BARRIER_BYTES);
const _: () = assert!(size_of::<windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_COPY_LOCATION>() == COPY_LOCATION_BYTES);

/// A PC (or proxy) COM object: a pointer to a pointer to the vtable.
type Object = *mut c_void;

/// The function of a vtable slot of `object`, as the function pointer type `F`.
///
/// # Safety
/// `object` points to a live COM object whose vtable has at least `slot + 1` entries, and `F` is
/// the function pointer type of that method.
unsafe fn method<F: Copy>(object: Object, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: by the contract above, an object starts with its vtable pointer.
    unsafe {
        let vtbl = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*vtbl.add(slot))
    }
}

type CreateQueue = unsafe extern "system" fn(Object, *const c_void, *const GUID, *mut Object) -> HRESULT;
type CreatePso = unsafe extern "system" fn(Object, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;
type CreateList = unsafe extern "system" fn(Object, u32, u32, Object, Object, *const GUID, *mut Object) -> HRESULT;
type CreateHeap = unsafe extern "system" fn(Object, *const c_void, *const GUID, *mut Object) -> HRESULT;
type CreateCommittedResource = unsafe extern "system" fn(Object, *const c_void, u32, *const c_void, u32, *const c_void, *const GUID, *mut Object) -> HRESULT;
type CreatePlacedResource = unsafe extern "system" fn(Object, Object, u64, *const c_void, u32, *const c_void, *const GUID, *mut Object) -> HRESULT;
type CreateCommandSignature = unsafe extern "system" fn(Object, *const c_void, Object, *const GUID, *mut Object) -> HRESULT;
type CreateSampler = unsafe extern "system" fn(Object, *const c_void, usize) -> HRESULT;
type CreateRootSig = unsafe extern "system" fn(Object, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
type CreateComputePso = unsafe extern "system" fn(Object, *const ComputePsoDesc, *const GUID, *mut *mut c_void) -> HRESULT;

/// `D3D12_SHADER_BYTECODE` (public d3d12.h): a pointer and a byte length.
#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderBytecode {
    ptr: *const u8,
    len: usize,
}

/// `D3D12_COMPUTE_PIPELINE_STATE_DESC` (public d3d12.h). `CS` sits at the same offset (8) as the
/// first shader slot (`VS`) of `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, so a console graphics
/// description can be read with the same struct for that one field if that is ever needed. Only
/// compute shaders are overridden (see the `dxil_override` call sites and the device slot 11 and 62
/// handlers below).
#[repr(C)]
#[derive(Clone, Copy)]
struct ComputePsoDesc {
    root_signature: Object,
    cs: ShaderBytecode,
    node_mask: u32,
    _pad0: u32,
    cached_pso: ShaderBytecode,
    flags: u32,
    _pad1: u32,
}

type AllocInfo = unsafe extern "system" fn(Object, *mut c_void, u32, u32, *const c_void) -> *mut c_void;
type HeapProps = unsafe extern "system" fn(Object, *mut c_void, u32, u32) -> *mut c_void;
type Residency = unsafe extern "system" fn(Object, u32, *const Object) -> HRESULT;
type AdapterLuid = unsafe extern "system" fn(Object, *mut LUID) -> *mut LUID;
type Execute = unsafe extern "system" fn(Object, u32, *const Object);
type Signal = unsafe extern "system" fn(Object, Object, u64) -> HRESULT;
type Bundle = unsafe extern "system" fn(Object, Object);
type SetHeaps = unsafe extern "system" fn(Object, u32, *const Object);
type Reset = unsafe extern "system" fn(Object, Object, Object) -> HRESULT;
type CopyTexture = unsafe extern "system" fn(Object, *const c_void, u32, u32, u32, *const c_void, *const c_void);
type CopyResource = unsafe extern "system" fn(Object, Object, Object);
type DepthBounds = unsafe extern "system" fn(Object, f32, f32);
type HeapStart = unsafe extern "system" fn(Object, *mut u64) -> *mut c_void;
type PrivateGet = unsafe extern "system" fn(Object, *const GUID, *mut u32, *mut c_void) -> HRESULT;
type PrivateSet = unsafe extern "system" fn(Object, *const GUID, u32, *const c_void) -> HRESULT;
type PrivateIface = unsafe extern "system" fn(Object, *const GUID, *const c_void) -> HRESULT;
type ByLuid = unsafe extern "system" fn(Object, LUID, *const GUID, *mut Object) -> HRESULT;
type Query = unsafe extern "system" fn(Object, *const GUID, *mut Object) -> HRESULT;
type Release = unsafe extern "system" fn(Object) -> u32;
type GpuVirtualAddress = unsafe extern "system" fn(Object) -> u64;
type ResourceMap = unsafe extern "system" fn(Object, u32, *const c_void, *mut *mut c_void) -> HRESULT;
type ResourceUnmap = unsafe extern "system" fn(Object, u32, *const c_void);
type CopyBufferRegion = unsafe extern "system" fn(Object, Object, u64, Object, u64, u64);
type CopyTiles = unsafe extern "system" fn(Object, Object, *const c_void, *const c_void, Object, u64, u32);
type ResolveSubresource = unsafe extern "system" fn(Object, Object, u32, Object, u32, u32);
type Barriers = unsafe extern "system" fn(Object, u32, *const c_void);
type ClearUavView = unsafe extern "system" fn(Object, u64, u64, Object, *const c_void, u32, *const c_void);
type DiscardResource = unsafe extern "system" fn(Object, Object, *const c_void);
type ResolveQueryData = unsafe extern "system" fn(Object, Object, u32, u32, u32, Object, u64);
type SetPredication = unsafe extern "system" fn(Object, Object, u64, u32);
type ExecuteIndirect = unsafe extern "system" fn(Object, Object, u32, Object, u64, Object, u64);
type CreateSharedHandle = unsafe extern "system" fn(Object, Object, *const c_void, u32, *const c_void, *mut *mut c_void) -> HRESULT;
type GetResourceTiling = unsafe extern "system" fn(Object, Object, *mut u32, *mut c_void, *mut c_void, *mut u32, u32, *mut c_void);
type UpdateTileMappings = unsafe extern "system" fn(Object, Object, u32, *const c_void, *const c_void, Object, u32, *const c_void, *const c_void, *const c_void, u32);
type CopyTileMappings = unsafe extern "system" fn(Object, Object, *const c_void, Object, *const c_void, *const c_void, u32);

/// `PresentX` hook of the presentation layer: the PC queue and the three arguments untouched.
pub type PresentFn = unsafe extern "system" fn(Object, u32, *const c_void, *const c_void) -> HRESULT;
/// The hook as an address, 0 when none is installed.
pub(crate) static PRESENT_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Public PC interface IDs (d3d12.h, dxgi.h, dxgi1_4.h).
const IID_IDXGI_OBJECT: GUID = GUID::from_u128(0xaec22fb8_76f3_4639_9be0_28eb43a67a2e);
const IID_IDXGI_DEVICE: GUID = GUID::from_u128(0x54ec77fa_1377_44e6_8c32_88fd5f44c84c);
const IID_IDXGI_ADAPTER: GUID = GUID::from_u128(0x2411e7e1_12ac_4ccf_bd14_9798e8534dc0);
const IID_IDXGI_FACTORY4: GUID = GUID::from_u128(0x1bc6ea02_ef36_464f_bf0c_21ca39e5168a);
const IID_COMMAND_QUEUE: GUID = GUID::from_u128(0x0ec870a6_5d7e_4c22_8cfc_5baae07616ed);
const IID_COMMAND_ALLOCATOR: GUID = GUID::from_u128(0x6102dee4_af59_4b09_b999_b44d73f09b24);
const IID_GRAPHICS_COMMAND_LIST: GUID = GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
/// The console's DMA command list, which the PC runtime does not know.
const IID_DMA_COMMAND_LIST: GUID = GUID::from_u128(0x30bc77b9_8379_4b25_95de_b8588c42275d);

/// Whether `riid` is `guid`.
///
/// # Safety
/// `riid` points to a readable GUID.
unsafe fn is_iid(riid: *const GUID, guid: &GUID) -> bool {
    // SAFETY: by the contract above.
    { let a = unsafe { &*riid }; a.data1 == guid.data1 && a.data2 == guid.data2 && a.data3 == guid.data3 && a.data4 == guid.data4 }
}

/// True for the first three calls of a counter, then every thousandth: enough to see a method in
/// the log without flooding it from a per frame loop.
pub(crate) fn log_due(counter: &AtomicI32) -> bool {
    let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
    n <= 3 || n % 1000 == 0
}

/// The proxy of one PC object. The relay stubs read the PC object at +0x08.
#[repr(C)]
struct Proxy {
    /// +0x00, the console vtable.
    vtbl: *const usize,
    /// +0x08, the PC object.
    pc: Object,
    /// +0x10
    refs: AtomicI32,
    /// +0x14
    kind: i32,
}

/// Memory accounting of a proxy, kept out of the proxy: the title reads and writes the bytes after
/// +0x18 (the list cursor and limits, the placed address), so no field may be added to the structs.
#[derive(Clone, Copy, Default)]
struct Side {
    /// The additive allocation released with this PC object, if any.
    charge: Option<Charge>,
    /// Informational placed-resource size and segment.
    placed: Option<(mem_owners::Segment, u64)>,
}

static SIDE: std::sync::Mutex<Option<std::collections::HashMap<usize, Side>>> = std::sync::Mutex::new(None);

fn side_set(proxy: *mut Proxy, f: impl FnOnce(&mut Side)) {
    let mut g = SIDE.lock().unwrap_or_else(|e| e.into_inner());
    let e = g.get_or_insert_with(Default::default).entry(proxy as usize).or_default();
    f(e);
}

fn side_take(proxy: *mut Proxy) -> Side {
    let mut g = SIDE.lock().unwrap_or_else(|e| e.into_inner());
    g.as_mut().and_then(|m| m.remove(&(proxy as usize))).unwrap_or_default()
}

/// A resource placed at a console address (device slots 44 and 133): the proxy header, then the
/// address the title placed it at (0 when unknown), and whether the last Map gave that address
/// instead of a PC mapping (so Unmap must not reach the PC object).
#[repr(C)]
struct ResProxy {
    base: Proxy,
    /// +0x18, the title's GPU address of the placement.
    placed_va: u64,
    /// +0x20, the last Map returned the placed address (zeroed: false).
    map_fallback: AtomicBool,
}

/// A command allocator proxy: the base, then what the title's slot 9 hands over.
#[repr(C)]
struct AllocProxy {
    base: Proxy,
    /// +0x20, the function of the last slot 9 call (size and alignment to memory), never called.
    callback: AtomicUsize,
    /// +0x28, its context.
    context: AtomicUsize,
}

/// A command list proxy covers +0x544 + 32 written by console callers.
const LIST_BYTES: usize = 0x800;

fn layout_of(kind: i32) -> Layout {
    let bytes = if kind == Kind::List as i32 {
        LIST_BYTES
    } else if kind == Kind::Allocator as i32 {
        size_of::<AllocProxy>()
    } else if kind == Kind::Resource as i32 {
        size_of::<ResProxy>()
    } else {
        size_of::<Proxy>()
    };
    Layout::from_size_align(bytes, 16).expect("a fixed, valid layout")
}

/// The vtables of the kinds, as arrays of slot addresses that live as long as the process.
struct Tables {
    vtbls: [Box<[usize]>; Kind::ALL.len()],
}

static TABLES: OnceLock<Option<Tables>> = OnceLock::new();
/// Calls of unknown slots, per kind and slot.
static UNKNOWN_CALLS: [[AtomicI32; LIST_SLOTS as usize]; Kind::ALL.len()] = [const { [const { AtomicI32::new(0) }; LIST_SLOTS as usize] }; Kind::ALL.len()];

/// Reached through the unknown slot stubs: edx = kind << 16 | slot.
extern "system" fn unknown_method(this: Object, code: u32) -> HRESULT {
    let (kind, slot) = (code >> 16, code & 0xFFFF);
    if let Some(kind) = Kind::from_index(kind)
        && slot < u32::from(LIST_SLOTS)
    {
        if kind == Kind::Device && crate::abi::note_newer_device_method(slot) {
            xs_log!("console device slot {slot}: the title's runtime has the newer console layout (D-2026-11-23-2)");
        }
        let n = UNKNOWN_CALLS[kind as usize][slot as usize].fetch_add(1, Ordering::SeqCst) + 1;
        decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("console {} slot {slot}", kind.name()));
        if n <= 3 || n % 1000 == 0 {
            let what = kind.rule(slot).map_or("?", |r| r.what);
            xs_log!("UNIMPLEMENTED console {} slot {slot} (call {n}) on {}: {what}", kind.name(), p(this));
        }
    }
    E_NOTIMPL
}

/// An unknown command-list slot that is probed for its arguments: the first calls of the slot log
/// the words after `this` (six, the stack ones included), then the slot answers `E_NOTIMPL` like
/// the unknown-slot stub does.
type ProbeFn = unsafe extern "system" fn(Object, u64, u64, u64, u64, u64, u64) -> HRESULT;

/// One probe per command-list slot of [`abi::PROBE_LIST_SLOTS`]. The stub of an unknown slot
/// overwrites `rdx` (its edx code), so the first argument is lost there; a probe is the method
/// itself and keeps all of them. Win64: the 5th and 6th words after `this` are on the stack.
extern "system" fn unknown_probe<const SLOT: u32>(this: Object, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64, a6: u64) -> HRESULT {
    let kind = Kind::List;
    let n = UNKNOWN_CALLS[kind as usize][SLOT as usize].fetch_add(1, Ordering::SeqCst) + 1;
    decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("console {} slot {SLOT}", kind.name()));
    if n <= 3 || n % 1000 == 0 {
        let what = kind.rule(SLOT).map_or("?", |r| r.what);
        xs_log!(
            "UNIMPLEMENTED console {} slot {SLOT} (call {n}) on {}: {what}; words after this: {a1:#x} {a2:#x} {a3:#x} {a4:#x} {a5:#x} {a6:#x} (upper halves of 32-bit arguments may be stale)",
            kind.name(),
            p(this)
        );
    }
    E_NOTIMPL
}

/// The probe of command-list slot `slot`, if it has one.
fn list_probe(slot: u32) -> Option<usize> {
    let i = slot.checked_sub(*crate::abi::PROBE_LIST_SLOTS.start())? as usize;
    PROBES.get(i).map(|f| *f as usize)
}

macro_rules! probe_table {
    ($($n:literal)*) => { [$(unknown_probe::<$n> as ProbeFn),*] };
}

/// The probes of [`abi::PROBE_LIST_SLOTS`] (60 to 149), in slot order.
static PROBES: [ProbeFn; 90] = probe_table!(
    60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89
    90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
    120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149
);

/// The PC object behind one of our proxies, or `object` itself.
///
/// # Safety
/// `object` is null or points to a COM object (or to memory that starts with a vtable pointer).
pub(crate) unsafe fn unwrap(object: Object) -> Object {
    if object.is_null() {
        return object;
    }
    // A pipeline made on first bind (`lazy_pso`): made now if it was not yet, null if refused.
    // SAFETY: by the contract above.
    if let Some(made) = unsafe { crate::lazy_pso::resolve(object) } {
        return made;
    }
    // A stand in for a refused pipeline (`pso_diag`) has no PC object behind it.
    // SAFETY: by the contract above.
    if unsafe { crate::pso_diag::standin_id(object) }.is_some() {
        return std::ptr::null_mut();
    }
    let Some(Some(tables)) = TABLES.get() else {
        return object;
    };
    // SAFETY: by the contract above, an object starts with a pointer.
    let vtbl = unsafe { *(object as *const *const usize) };
    if tables.vtbls.iter().any(|t| std::ptr::eq(t.as_ptr(), vtbl)) {
        // SAFETY: that vtable is one of ours, so the object is a `Proxy`.
        return unsafe { (*(object as *const Proxy)).pc };
    }
    object
}

/// Unwraps an array of objects for a PC call.
///
/// # Safety
/// `input` is null or points to `n` readable object pointers, each as for [`unwrap`].
unsafe fn unwrap_array(n: u32, input: *const Object) -> Vec<Object> {
    (0..n as usize)
        // SAFETY: by the contract above.
        .map(|i| if input.is_null() { std::ptr::null_mut() } else { unsafe { unwrap(*input.add(i)) } })
        .collect()
}

/// Whether `object` is one of our own device proxies (so it is safe to read as a `Proxy`).
fn is_device_proxy(object: *mut c_void) -> bool {
    if object.is_null() {
        return false;
    }
    let Some(Some(tables)) = TABLES.get() else {
        return false;
    };
    // SAFETY: `object` is non null; if it is one of ours its vtable pointer is readable.
    let vtbl = unsafe { *(object as *const *const usize) };
    std::ptr::eq(tables.vtbls[Kind::Device as usize].as_ptr(), vtbl)
}

/// Console device slots 25 and 26 (`GetResourceAllocationInfo`, `GetCustomHeapProperties`): a
/// console caller may put the result pointer or `this` first. Whichever of `a`/`b` is one of our
/// own device proxies is `this`; the other is the result pointer. Anything that is not a device
/// proxy in the first position makes the second one `this`.
fn sret_order(first: *mut c_void, second: *mut c_void) -> (*mut Proxy, *mut c_void) {
    match is_device_proxy(first) {
        true => (first.cast(), second),
        false => (second.cast(), first),
    }
}

// ---- IUnknown, shared by every proxy ----

extern "system" fn add_ref(this: *mut Proxy) -> u32 {
    // SAFETY: `this` is a live proxy, as the caller holds a reference.
    (unsafe { &*this }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
}

/// What a released title object is called in `retire`'s names and fault report.
fn retired_label(kind: i32) -> &'static str {
    match Kind::from_index(kind as u32) {
        Some(Kind::Resource) => "title resource",
        Some(Kind::Heap) => "title descriptor heap",
        Some(Kind::List) | Some(Kind::DmaList) => "title command list",
        Some(Kind::Allocator) => "title command allocator",
        Some(Kind::Queue) => "title queue",
        Some(Kind::Fence) => "title fence",
        _ => "title object",
    }
}

extern "system" fn release(this: *mut Proxy) -> u32 {
    // SAFETY: `this` is a live proxy, as the caller holds a reference.
    let proxy = unsafe { &*this };
    let n = proxy.refs.fetch_sub(1, Ordering::SeqCst) - 1;
    if n == 0 {
        let side = side_take(this);
        if let Some((segment, bytes)) = side.placed {
            mem_owners::remove(mem_owners::Kind::PlacedResource, segment, 1, bytes);
        }
        if proxy.kind == Kind::Heap as i32 {
            crate::title_heap::heap_released(proxy.pc as usize);
            crate::title_heap::visible_released(proxy.pc as usize);
            crate::view_shadow::heap_released(proxy.pc as usize);
            let owned = [crate::title_heap::paged_released(proxy.pc as usize), crate::title_heap::page_map_released(proxy.pc as usize)];
            for item in owned.into_iter().flatten() {
                let (object, charge) = item;
                crate::view_shadow::heap_released(object);
                // SAFETY: the backing heap and the page map buffer were made for this heap and are
                // released with it, once.
                unsafe {
                    if let Some(charge) = charge {
                        crate::retire::release_owned(object as Object, "descriptor heap backing", charge);
                    } else {
                        crate::retire::release(object as Object, "descriptor heap backing");
                    }
                }
            }
        }
        if proxy.kind == Kind::List as i32 {
            crate::title_heap::list_reset(this as usize);
            crate::cbv_copies::forget(this as usize);
            crate::flicker_probe::forget(this as usize);
            crate::present::list_reset(proxy.pc as usize);
        }
        if proxy.kind == Kind::Resource as i32 {
            crate::views_at::forget_resource(proxy.pc as usize);
            crate::detile::forget(proxy.pc as usize);
            crate::rt_diag::forget(proxy.pc as usize);
            crate::present::forget_resource(proxy.pc as usize);
        }
        // SAFETY: the last reference is gone, so the PC object is released and the memory freed once.
        // Any object but the device is released once the GPU is past the work in flight now
        // (`retire`): the title's lists may still use it, as they may on the console.
        unsafe {
            if proxy.kind == Kind::Device as i32 {
                crate::retire::flush(proxy.pc);
                method::<Release>(proxy.pc, 2)(proxy.pc);
            } else if let Some(charge) = side.charge {
                crate::retire::release_owned(proxy.pc, retired_label(proxy.kind), charge);
            } else {
                crate::retire::release(proxy.pc, retired_label(proxy.kind));
            }
            dealloc(this.cast(), layout_of(proxy.kind));
        }
    }
    n as u32
}

/// An interface the PC object implements is answered with the proxy itself: the caller is console
/// code and must keep the console vtable. On the device, IDXGIDevice (the console merges the DXGI
/// device into the D3D12 device) is answered by a DXGI device object. Anything else is refused and
/// logged with its GUID.
///
/// Port note (branch `d3d12xs/desktop-iid-and-console-descs`): the C layer's own `obj_QueryInterface`
/// (`xs_wrap.c`) answers `IID_ID3D12Device` the same way this does - `S_OK` with the wrapper itself,
/// never the PC object - and its own test (`xs_census_synth.c`, case D6, "the known id is logged as
/// S_OK, not E_NOINTERFACE") asserts that is the intended behaviour, not a bug: the console vtable
/// is PC order for slots 0..43, so a caller asking for the base `ID3D12Device` identity gets back an
/// object that answers that identity correctly through the console vtable. No switch or code path
/// in the C layer refuses this id, and no run in `docs/private/research/r21/RESULT.md` records a
/// crash traced to it. A change here would make this proxy disagree with the project's own tested C
/// behaviour, so none was made; see the port report for the brief's conflicting claim.
extern "system" fn query_interface(this: *mut Proxy, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `out` is writable, `riid` readable, by the COM contract.
    unsafe {
        *out = std::ptr::null_mut();
        let proxy = &*this;
        if proxy.kind == Kind::Device as i32 && (is_iid(riid, &IID_IDXGI_DEVICE) || is_iid(riid, &IID_IDXGI_OBJECT)) {
            return dxgi_device_for(this, riid, out);
        }
        let mut tmp: Object = std::ptr::null_mut();
        let hr = method::<Query>(proxy.pc, 0)(proxy.pc, riid, &mut tmp);
        if hr >= 0 {
            method::<Release>(tmp, 2)(tmp);
            add_ref(this);
            *out = this.cast();
            return S_OK;
        }
        let kind = Kind::ALL[proxy.kind as usize];
        xs_log!("console {} proxy {}: QueryInterface {{{}}} not implemented by the PC object (hr=0x{:08X})", kind.name(), p(this.cast()), guid_at(riid), hr as u32);
        hr
    }
}

/// Wraps a freshly created PC object (one reference) into `*out`.
///
/// # Safety
/// `out` is writable. `obj` is null or a PC object whose reference this call takes over.
unsafe fn give(hr: HRESULT, obj: Object, kind: Kind, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe { give_charged(hr, obj, kind, None, out) }
}

/// As [`give`], with an allocation charge removed at the PC object's final deferred release.
///
/// # Safety
/// As [`give`].
unsafe fn give_charged(hr: HRESULT, obj: Object, kind: Kind, charge: Option<Charge>, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        *out = std::ptr::null_mut();
        if hr < 0 || obj.is_null() {
            return hr;
        }
        *out = wrap(obj, kind);
        if (*out).is_null() {
            method::<Release>(obj, 2)(obj);
            return E_OUTOFMEMORY;
        }
        if charge.is_some() {
            side_set((*out).cast::<Proxy>(), |s| s.charge = charge);
        }
        if let Some(charge) = charge {
            mem_owners::add_charge(charge);
        }
    }
    hr
}

/// Wraps a placed resource (one PC reference, taken over) as a `Kind::Resource` proxy that keeps the
/// title's address `va` of the placement, into `*out`. See [`give`].
///
/// # Safety
/// As [`give`].
unsafe fn give_placed(hr: HRESULT, obj: Object, va: u64, charge: Option<Charge>, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    let hr = unsafe { give_charged(hr, obj, Kind::Resource, charge, out) };
    if hr >= 0 && unsafe { !(*out).is_null() } {
        // SAFETY: `*out` is a fresh resource proxy, which is a `ResProxy`, and nobody else has it yet.
        unsafe {
            let proxy = &mut *(*out).cast::<ResProxy>();
            proxy.placed_va = va;
        }
    }
    hr
}

// ---- DXGI device (device QueryInterface(IDXGIDevice)) ----
//
// The PC device does not implement IDXGIDevice. When the PC object does, its own IDXGIDevice is
// handed out (PC DXGI vtable order, public). Otherwise this object answers it: private data relayed
// to the PC device, the adapter found by the PC device's LUID, the GPU thread priority kept (no
// effect on PC), surfaces and residency queries refused and logged.

#[repr(C)]
struct DxgiDevice {
    vtbl: *const *const c_void,
    /// An owned reference to the device proxy.
    dev: *mut Proxy,
    refs: AtomicI32,
    priority: AtomicI32,
}

/// The adapter of a PC device: found by its LUID through the DXGI factory.
///
/// # Safety
/// `pc_dev` is a live PC device and `out` is writable.
unsafe fn pc_adapter(pc_dev: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static DXGI: AtomicUsize = AtomicUsize::new(0);
    type CreateFactory = unsafe extern "system" fn(*const GUID, *mut Object) -> HRESULT;
    // SAFETY: by the contract above, and the loader calls take zero terminated names.
    unsafe {
        *out = std::ptr::null_mut();
        let mut module = DXGI.load(Ordering::SeqCst);
        if module == 0 {
            module = LoadLibraryW(windows_sys::w!("dxgi.dll")) as usize;
            DXGI.store(module, Ordering::SeqCst);
        }
        let create = if module == 0 { None } else { GetProcAddress(module as *mut c_void, c"CreateDXGIFactory1".as_ptr().cast()) };
        let Some(create) = create else {
            xs_log!("IDXGIDevice: no CreateDXGIFactory1 in dxgi.dll (module {})", p(module as *const c_void));
            return E_NOTIMPL;
        };
        let create = std::mem::transmute::<unsafe extern "system" fn() -> isize, CreateFactory>(create);
        let mut luid = LUID { LowPart: 0, HighPart: 0 };
        method::<AdapterLuid>(pc_dev, 43)(pc_dev, &mut luid);
        let mut factory: Object = std::ptr::null_mut();
        let hr = create(&IID_IDXGI_FACTORY4, &mut factory);
        if hr < 0 {
            xs_log!("IDXGIDevice: CreateDXGIFactory1(IDXGIFactory4) failed, hr=0x{:08X}", hr as u32);
            return hr;
        }
        // Always fetched as IDXGIAdapter first, then wrapped: `dxgi::wrap_adapter` gives the
        // real adapter a headless `EnumOutputs` fallback (see `dxgi`'s module doc), before the
        // caller's own requested interface (`riid`) is queried off the wrapper.
        let mut real: Object = std::ptr::null_mut();
        let hr = method::<ByLuid>(factory, 26)(factory, luid, &IID_IDXGI_ADAPTER, &mut real);
        method::<Release>(factory, 2)(factory);
        if hr < 0 {
            xs_log!("IDXGIDevice: no DXGI adapter for the device LUID {:08X}:{:08X}, hr=0x{:08X}", luid.HighPart as u32, luid.LowPart, hr as u32);
            return hr;
        }
        let shadow = crate::dxgi::wrap_adapter(real);
        let hr = method::<Query>(shadow, 0)(shadow, riid, out);
        method::<Release>(shadow, 2)(shadow);
        hr
    }
}

extern "system" fn dxgi_query_interface(this: *mut DxgiDevice, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live DXGI device object, `riid` readable, by the COM contract.
    unsafe {
        if is_iid(riid, &IID_IDXGI_DEVICE) || is_iid(riid, &IID_IDXGI_OBJECT) {
            (*this).refs.fetch_add(1, Ordering::SeqCst);
            *out = this.cast();
            return S_OK;
        }
        // IUnknown and the D3D12 interfaces: the device proxy (one COM identity).
        query_interface((*this).dev, riid, out)
    }
}

extern "system" fn dxgi_add_ref(this: *mut DxgiDevice) -> u32 {
    // SAFETY: `this` is a live DXGI device object.
    (unsafe { &*this }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
}

extern "system" fn dxgi_release(this: *mut DxgiDevice) -> u32 {
    // SAFETY: `this` is a live DXGI device object, freed once when its last reference goes.
    unsafe {
        let n = (*this).refs.fetch_sub(1, Ordering::SeqCst) - 1;
        if n == 0 {
            release((*this).dev);
            drop(Box::from_raw(this));
        }
        n as u32
    }
}

extern "system" fn dxgi_set_private(this: *mut DxgiDevice, guid: *const GUID, size: u32, data: *const c_void) -> HRESULT {
    // SAFETY: `this` is live, and its device proxy holds a live PC object.
    unsafe {
        let pc = (*(*this).dev).pc;
        method::<PrivateSet>(pc, 4)(pc, guid, size, data)
    }
}

extern "system" fn dxgi_set_private_interface(this: *mut DxgiDevice, guid: *const GUID, data: *const c_void) -> HRESULT {
    // SAFETY: as in `dxgi_set_private`.
    unsafe {
        let pc = (*(*this).dev).pc;
        method::<PrivateIface>(pc, 5)(pc, guid, data)
    }
}

extern "system" fn dxgi_get_private(this: *mut DxgiDevice, guid: *const GUID, size: *mut u32, data: *mut c_void) -> HRESULT {
    // SAFETY: as in `dxgi_set_private`.
    unsafe {
        let pc = (*(*this).dev).pc;
        method::<PrivateGet>(pc, 3)(pc, guid, size, data)
    }
}

/// The parent of a DXGI device is its adapter.
extern "system" fn dxgi_get_parent(this: *mut DxgiDevice, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: as in `dxgi_set_private`, and `out` is writable.
    unsafe { pc_adapter((*(*this).dev).pc, riid, out) }
}

extern "system" fn dxgi_get_adapter(this: *mut DxgiDevice, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: as in `dxgi_set_private`, and `out` is writable.
    unsafe { pc_adapter((*(*this).dev).pc, &IID_IDXGI_ADAPTER, out) }
}

extern "system" fn dxgi_create_surface(this: *mut DxgiDevice, desc: *const c_void, n: u32, _usage: u32, _shared: *const c_void, out: *mut *mut c_void) -> HRESULT {
    decant_log::calls::not_implemented("IDXGIDevice::CreateSurface");
    // SAFETY: `this` is live, `out` is null or writable.
    unsafe {
        xs_log!("UNIMPLEMENTED IDXGIDevice::CreateSurface({}, {n}) on the console device {}", p(desc), p((*this).dev.cast()));
        if !out.is_null() {
            *out = std::ptr::null_mut();
        }
    }
    E_NOTIMPL
}

extern "system" fn dxgi_query_resource_residency(this: *mut DxgiDevice, _resources: *const *mut c_void, _status: *mut c_void, n: u32) -> HRESULT {
    decant_log::calls::not_implemented("IDXGIDevice::QueryResourceResidency");
    // SAFETY: `this` is live.
    unsafe { xs_log!("UNIMPLEMENTED IDXGIDevice::QueryResourceResidency({n} resources) on the console device {}", p((*this).dev.cast())) };
    E_NOTIMPL
}

extern "system" fn dxgi_set_priority(this: *mut DxgiDevice, priority: i32) -> HRESULT {
    if !(-7..=7).contains(&priority) {
        return E_INVALIDARG;
    }
    // SAFETY: `this` is live.
    unsafe { (*this).priority.store(priority, Ordering::SeqCst) };
    xs_log!("IDXGIDevice::SetGPUThreadPriority({priority}): kept, no effect on PC");
    S_OK
}

extern "system" fn dxgi_get_priority(this: *mut DxgiDevice, priority: *mut i32) -> HRESULT {
    if priority.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live, `priority` is writable.
    unsafe { *priority = (*this).priority.load(Ordering::SeqCst) };
    S_OK
}

/// A vtable that can sit in a static: the entries are code addresses.
#[repr(transparent)]
struct Vtable<const N: usize>([*const c_void; N]);

// SAFETY: the entries are addresses of functions and are never written.
unsafe impl<const N: usize> Sync for Vtable<N> {}

/// IDXGIDevice order (public dxgi.h): IUnknown, IDXGIObject 3..6, IDXGIDevice 7..11.
static DXGI_VTABLE: Vtable<12> = Vtable([
    dxgi_query_interface as *const c_void,
    dxgi_add_ref as *const c_void,
    dxgi_release as *const c_void,
    dxgi_set_private as *const c_void,
    dxgi_set_private_interface as *const c_void,
    dxgi_get_private as *const c_void,
    dxgi_get_parent as *const c_void,
    dxgi_get_adapter as *const c_void,
    dxgi_create_surface as *const c_void,
    dxgi_query_resource_residency as *const c_void,
    dxgi_set_priority as *const c_void,
    dxgi_get_priority as *const c_void,
]);

/// # Safety
/// `dev` is a live device proxy, `riid` readable and `out` writable.
unsafe fn dxgi_device_for(dev: *mut Proxy, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        let pc = (*dev).pc;
        let hr = method::<Query>(pc, 0)(pc, riid, out.cast());
        if hr >= 0 {
            xs_log!("console device {}: QueryInterface(IDXGIDevice) relayed to the PC device", p(dev.cast()));
            return hr;
        }
        let object = Box::into_raw(Box::new(DxgiDevice {
            vtbl: DXGI_VTABLE.0.as_ptr(),
            dev,
            refs: AtomicI32::new(1),
            priority: AtomicI32::new(0),
        }));
        add_ref(dev);
        xs_log!(
            "console device {}: QueryInterface(IDXGIDevice) answered by the DXGI device object {} (the PC device has none, hr=0x{:08X})",
            p(dev.cast()),
            p(object.cast()),
            hr as u32
        );
        *out = object.cast();
    }
    S_OK
}

// ---- device ----

extern "system" fn dev_create_queue(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `desc` and `riid` are the caller's, `out` is writable.
    unsafe {
        let pc = (*this).pc;
        let host = if desc.is_null() { None } else { Some(host_queue_desc(std::ptr::read_unaligned(desc.cast()))) };
        let desc = host.as_ref().map_or(desc, |h| h.as_ptr().cast());
        let mut q: Object = std::ptr::null_mut();
        let mut hr = method::<CreateQueue>(pc, 8)(pc, desc, riid, &mut q);
        if hr == E_NOINTERFACE && !is_iid(riid, &IID_COMMAND_QUEUE) {
            xs_log!("console device: queue IID {{{}}} unknown to the PC runtime, created as ID3D12CommandQueue", guid_at(riid));
            hr = method::<CreateQueue>(pc, 8)(pc, desc, &IID_COMMAND_QUEUE, &mut q);
        }
        give(hr, q, Kind::Queue, out)
    }
}

/// # Safety
/// `this` is a live device proxy, `alloc` is an allocator proxy or a PC allocator, `pso` is a PC object or null, `riid` readable and
/// `out` writable.
unsafe fn create_list(this: *mut Proxy, mask: u32, kind: u32, alloc: Object, pso: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        let pc = (*this).pc;
        let alloc = unwrap(alloc);
        // A pipeline made on first bind is made here (null when the PC runtime refused it).
        let pso = unwrap(pso);
        let mut l: Object = std::ptr::null_mut();
        let mut hr = method::<CreateList>(pc, 12)(pc, mask, kind, alloc, pso, riid, &mut l);
        if hr == E_NOINTERFACE && !is_iid(riid, &IID_GRAPHICS_COMMAND_LIST) {
            xs_log!("console device: command list IID {{{}}} unknown to the PC runtime, created as ID3D12GraphicsCommandList", guid_at(riid));
            hr = method::<CreateList>(pc, 12)(pc, mask, kind, alloc, pso, &IID_GRAPHICS_COMMAND_LIST, &mut l);
        }
        let kind = if is_iid(riid, &IID_DMA_COMMAND_LIST) { Kind::DmaList } else { Kind::List };
        let hr = give(hr, l, kind, out);
        if hr >= 0 && !(*out).is_null() {
            crate::upload_mem::with(|m| m.set_allocator(*out as usize, alloc as usize));
        }
        hr
    }
}

extern "system" fn dev_create_list(this: *mut Proxy, mask: u32, kind: u32, alloc: Object, pso: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: the arguments are the caller's, by the COM contract.
    unsafe { create_list(this, mask, kind, alloc, pso, riid, out) }
}

type CreateAllocator = unsafe extern "system" fn(Object, u32, *const GUID, *mut Object) -> HRESULT;

/// Console device slot 9: CreateCommandAllocator (type, riid, out), the result wrapped.
extern "system" fn dev_create_allocator(this: *mut Proxy, kind: u32, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `riid` is the caller's, `out` is writable.
    unsafe {
        let pc = (*this).pc;
        let mut a: Object = std::ptr::null_mut();
        let mut hr = method::<CreateAllocator>(pc, 9)(pc, kind, riid, &mut a);
        if hr == E_NOINTERFACE && !is_iid(riid, &IID_COMMAND_ALLOCATOR) {
            xs_log!("console device: allocator IID {{{}}} unknown to the PC runtime, created as ID3D12CommandAllocator", guid_at(riid));
            hr = method::<CreateAllocator>(pc, 9)(pc, kind, &IID_COMMAND_ALLOCATOR, &mut a);
        }
        give(hr, a, Kind::Allocator, out)
    }
}

type GetIncSize = unsafe extern "system" fn(Object, u32) -> u32;
type CreateCbv = unsafe extern "system" fn(Object, *const c_void, usize);
type CreateSrv = unsafe extern "system" fn(Object, Object, *const c_void, usize);
type CreateUav = unsafe extern "system" fn(Object, Object, Object, *const c_void, usize);
type CreateRtv = unsafe extern "system" fn(Object, Object, *const c_void, usize);
type CreateDsv = unsafe extern "system" fn(Object, Object, *const c_void, usize);
type CreateSamplerPc = unsafe extern "system" fn(Object, *const c_void, usize);
type CopyDescriptorsPc = unsafe extern "system" fn(Object, u32, *const usize, *const u32, u32, *const usize, *const u32, u32);
type CopyDescriptorsSimplePc = unsafe extern "system" fn(Object, u32, usize, usize, u32);

/// Console slot 14: CreateDescriptorHeap. If a shader-visible CBV/SRV/UAV heap exceeds the PC
/// D3D12 Tier 2/3 cap (1,000,000 descriptors) and the host rejects it, retry clamped to 1,000,000.
/// When `xmem.dll` is loaded, allocate the full console descriptor range through `XMemAlloc`
/// (invoking the title's `XMemSetAllocationHooks`) and register the title -> host handle mapping.
/// Console device slot 15: GetDescriptorHandleIncrementSize, with the console's heap type 4 asked
/// as the PC view heap type 0 (the PC runtime answers 0 for type 4).
extern "system" fn dev_handle_increment(this: *mut Proxy, heap_type: u32) -> u32 {
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        method::<unsafe extern "system" fn(Object, u32) -> u32>(pc, 15)(pc, if heap_type == 4 { 0 } else { heap_type })
    }
}

/// Console device slot 52: SetDebugFlagsX, nothing to set on PC.
extern "system" fn dev_set_debug_flags(_this: *mut Proxy, _flags: u32) -> HRESULT {
    S_OK
}

/// Console device slots 56 and 66: SerializePipelineStateX. The console driver's serialised form
/// cannot be produced, so the title is told it may not, and keeps its pipelines as they are.
extern "system" fn dev_serialize_pso(_this: *mut Proxy, _pso: Object) -> HRESULT {
    0x8007_0005_u32 as HRESULT // E_ACCESSDENIED
}

extern "system" fn dev_create_heap(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `desc` (16 bytes when non-null) and `riid` are the caller's, `out` is writable.
    unsafe {
        let pc = (*this).pc;
        // Heap type 4 is the console's second CBV/SRV/UAV type, a PC view heap (type 0).
        let title_count = if desc.is_null() { 0 } else { std::ptr::read_unaligned(desc.cast::<[u32; 4]>())[1] };
        let direct = crate::compile_service::direct_enabled();
        let mapped = if desc.is_null() {
            None
        } else {
            let mut d: [u32; 4] = std::ptr::read_unaligned(desc.cast());
            if d[0] == 4 {
                d[0] = 0;
            }
            // Under the direct path a shader visible sampler heap gets the tail of immediate samplers.
            if direct {
                d[1] = crate::sampler_registry::heap_count(d[0], d[2], d[1]);
            }
            Some(d)
        };
        let desc: *const c_void = mapped.as_ref().map_or(desc, |d| d.as_ptr().cast());
        let mut h: Object = std::ptr::null_mut();
        let mut hr = method::<CreateHeap>(pc, 14)(pc, desc, riid, &mut h);
        let (mut heap_type, mut orig_num, mut host_num) = (0u32, 0u32, 0u32);
        // MOVED_SLOTS extra host slots a shader-visible view heap that fit whole (unclamped) was
        // given past the title's own count, for the ring `title_heap::move_table` uses; 0 when
        // it has none. Never fed to the title: every registration below still uses `host_num`.
        let mut ring_slots = 0u32;
        // A view heap the PC refuses at the title's size: shader visible at the PC limit, its
        // slots in a CPU only heap of the full size (`heap_pages`).
        let mut backing: Option<(Object, usize, Charge)> = None;
        let mut backing_charge = None;
        if !desc.is_null() {
            let d: [u32; 4] = std::ptr::read_unaligned(desc.cast());
            heap_type = d[0];
            orig_num = title_count;
            host_num = d[1];
            let flags = d[2];
            if hr < 0 && heap_type == 0 && (flags & 1) != 0 && orig_num > crate::title_heap::PC_MAX_SHADER_VISIBLE_VIEW_DESCRIPTORS {
                let mut clamped = d;
                clamped[1] = crate::title_heap::PC_MAX_SHADER_VISIBLE_VIEW_DESCRIPTORS;
                h = std::ptr::null_mut();
                hr = method::<CreateHeap>(pc, 14)(pc, clamped.as_ptr().cast(), riid, &mut h);
                if hr >= 0 {
                    host_num = clamped[1];
                    backing = heap_pages::backing(pc, orig_num, d[3], riid);
                    xs_log!(
                        "console device slot 14 (CreateDescriptorHeap): clamped shader-visible view heap {orig_num} -> {host_num} (PC D3D12 limit), {}",
                        if backing.is_some() { "paged over a CPU only heap of the full size" } else { "slots past the limit unreachable" }
                    );
                }
            } else if hr >= 0
                && !h.is_null()
                && heap_type == 0
                && (flags & 1) != 0
                && orig_num > 0
                && host_num.checked_add(crate::title_heap::UNPAGED_MOVED_SLOTS).is_some()
            {
                // The title's whole heap fit (vkd3d-proton: always, it never refuses one and so
                // never pages it, leaving `move_table`'s fix for 45f87e7 with no tail to use). Ask
                // for UNPAGED_MOVED_SLOTS more host slots in the same heap past the title's own count, for
                // that ring; fall back to the plain heap already made (no ring) if the PC refuses.
                let mut grown = d;
                grown[1] = host_num + crate::title_heap::UNPAGED_MOVED_SLOTS;
                let mut h2: Object = std::ptr::null_mut();
                let hr2 = method::<CreateHeap>(pc, 14)(pc, grown.as_ptr().cast(), riid, &mut h2);
                if hr2 >= 0 && !h2.is_null() {
                    method::<Release>(h, 2)(h);
                    h = h2;
                    ring_slots = crate::title_heap::UNPAGED_MOVED_SLOTS;
                } else if !h2.is_null() {
                    method::<Release>(h2, 2)(h2);
                }
            }
        }
        if hr >= 0 && !h.is_null() && orig_num > 0 {
            let inc = method::<GetIncSize>(pc, 15)(pc, heap_type);
            let mut host_cpu = 0u64;
            method::<HeapStart>(h, 9)(h, &mut host_cpu);
            // The title's CPU handles reach the backing heap's slots when paged.
            match backing {
                Some((_, start, charge)) => {
                    backing_charge = Some(charge);
                    crate::title_heap::heap_created(h as usize, heap_type, orig_num, orig_num, inc, start)
                }
                None => crate::title_heap::heap_created(h as usize, heap_type, orig_num, host_num, inc, host_cpu as usize),
            }
            if direct && heap_type == 1 {
                immediate_samplers::heap_made(pc, h, orig_num, host_num, inc, host_cpu as usize);
            }
            // The view shadow keeps rows for the host slots of the heap (a backing heap has its own).
            if heap_type == 0 {
                let mut gpu = 0u64;
                if (std::ptr::read_unaligned(desc.cast::<[u32; 4]>())[2] & 1) != 0 {
                    method::<HeapStart>(h, 10)(h, &mut gpu);
                }
                // the rows cover the ring too: a translated shader reads the row of a moved table's slot
                crate::view_shadow::heap_created(pc as usize, h as usize, host_cpu as usize, gpu, host_num + ring_slots, inc);
            }
            // Paging needs the title's handles in title memory, which only a registered heap has.
            if let Some((object, _, charge)) = backing
                && crate::title_heap::title_start_for_heap(h as usize).is_none()
            {
                xs_log!("console device slot 14 (CreateDescriptorHeap): the view heap has no title memory, not paged");
                crate::view_shadow::heap_released(object as usize);
                mem_owners::remove_charge(charge);
                method::<Release>(object, 2)(object);
                backing = None;
                backing_charge = None;
            }
            if let Some((object, start, _)) = backing {
                let mut gpu = 0u64;
                method::<HeapStart>(h, 10)(h, &mut gpu);
                let map = heap_pages::page_map(pc, &crate::title_heap::empty_map(orig_num));
                if let Some((buffer, _, address, charge)) = map {
                    crate::title_heap::page_map_created(crate::title_heap::PageMap { heap: h as usize, gpu_start: gpu, address, buffer: buffer as usize, charge: Some(charge) });
                }
                crate::title_heap::paged_created(crate::title_heap::Paged {
                    gpu_start: gpu,
                    increment: inc,
                    host_cpu: host_cpu as usize,
                    backing_cpu: start,
                    map_cpu: map.map_or(0, |(_, cpu, _, _)| cpu),
                    device: pc as usize,
                    heap: h as usize,
                    backing: object as usize,
                    backing_charge,
                    // the last slots stay outside the pages, for tables moved for one draw
                    pager: crate::title_heap::Pager::new(orig_num, host_num.saturating_sub(crate::title_heap::MOVED_SLOTS)),
                    host_count: host_num,
                    moved_next: 0,
                    moved: Vec::new(),
                });
            }
        } else if let Some((object, _, charge)) = backing {
            crate::view_shadow::heap_released(object as usize);
            mem_owners::remove_charge(charge);
            method::<Release>(object, 2)(object);
        }
        // Every other shader visible view heap gets a page map with page p at p * PAGE, so a directly
        // translated shader reads any bound heap the same way.
        let visible_views = mapped.is_some_and(|d| d[0] == 0 && d[2] & 1 != 0);
        if hr >= 0 && !h.is_null() && host_num > 0 && visible_views && backing.is_none()
            && let Some((buffer, _, address, charge)) = heap_pages::page_map(pc, &crate::title_heap::identity_map(host_num + ring_slots))
        {
            let mut gpu = 0u64;
            method::<HeapStart>(h, 10)(h, &mut gpu);
            crate::title_heap::page_map_created(crate::title_heap::PageMap { heap: h as usize, gpu_start: gpu, address, buffer: buffer as usize, charge: Some(charge) });
        }
        // Where the descriptors of the title's slots are written: the backing heap when paged.
        let slots = |h: Object| -> (usize, u32) {
            match backing {
                Some((_, start, _)) => (start, orig_num),
                None => {
                    let mut cpu = 0u64;
                    method::<HeapStart>(h, 9)(h, &mut cpu);
                    (cpu as usize, host_num)
                }
            }
        };
        // A shader visible heap is what a table of the user data mirror is an index into.
        if hr >= 0 && !h.is_null() && orig_num > 0 && matches!(heap_type, 0 | 1) && crate::user_mirror::wanted() {
            let mut gpu = 0u64;
            method::<HeapStart>(h, 10)(h, &mut gpu);
            let inc = method::<GetIncSize>(pc, 15)(pc, heap_type);
            crate::user_mirror::heap_created(h as usize, heap_type == 1, gpu, inc);
        }
        // A shader visible heap with title memory: its tables get the views of the tagged slots.
        let visible = !desc.is_null() && (std::ptr::read_unaligned(desc.cast::<[u32; 4]>())[2] & 1) != 0;
        if hr >= 0 && !h.is_null() && orig_num > 0 && visible && matches!(heap_type, 0 | 1)
            && let Some(title_base) = crate::title_heap::title_start_for_heap(h as usize)
        {
            let mut gpu = 0u64;
            method::<HeapStart>(h, 10)(h, &mut gpu);
            let (cpu, count) = slots(h);
            crate::title_heap::visible_created(crate::title_heap::Visible {
                gpu_start: gpu,
                increment: method::<GetIncSize>(pc, 15)(pc, heap_type),
                host_cpu: cpu,
                host_count: count,
                title_base,
                title_count: orig_num,
                heap_type,
                device: pc as usize,
                heap: h as usize,
                ring_slots: if heap_type == 0 { ring_slots } else { 0 },
                ring_next: 0,
            });
        }
        if hr >= 0 && !h.is_null() && orig_num > 0 && heap_type == 0 && crate::view_diag::enabled() {
            let mut gpu = 0u64;
            method::<HeapStart>(h, 10)(h, &mut gpu);
            let (cpu, count) = slots(h);
            let heap = crate::view_diag::Heap {
                host_cpu: cpu,
                increment: method::<GetIncSize>(pc, 15)(pc, 0),
                host_count: count,
                title_count: orig_num,
                title_base: crate::title_heap::title_start_for_heap(h as usize).unwrap_or(0),
            };
            if gpu != 0 {
                crate::view_diag::heap_made(gpu, heap);
            }
        }
        let heap_charge = (hr >= 0 && !h.is_null()).then(|| {
            let increment = method::<GetIncSize>(pc, 15)(pc, heap_type) as u64;
            Charge {
                kind: mem_owners::Kind::DescriptorHeap,
                segment: if visible { mem_owners::Segment::Local } else { mem_owners::Segment::System },
                bytes: (host_num as u64 + ring_slots as u64).saturating_mul(increment),
            }
        });
        give_charged(hr, h, Kind::Heap, heap_charge, out)
    }
}

/// The PC GPU address behind a title GPU address, for views that take an address: upload memory
/// of list slot 148, title memory that `xmem.dll` maps (the window of `title_mem`), or a
/// buffer placed at a title address. `None` when no PC resource holds it.
fn pc_gpu_address(va: u64, len: u64) -> Option<u64> {
    root_views::pc_address(va, len).map(|(pc_va, _)| pc_va)
}

/// Console device slot 17: CreateConstantBufferView. The console takes sizes in 128 byte steps and
/// addresses in title memory: the size is rounded up to the PC's 256 bytes and the address
/// translated to the PC resource that holds it. A view of memory no PC resource holds is left out
/// (native D3D12 removes the device on it) and logged.
extern "system" fn dev_create_cbv(this: *mut Proxy, desc: *const c_void, cpu: usize) {
    static SKIPPED: AtomicI32 = AtomicI32::new(0);
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return };
    // SAFETY: `this` is a live device proxy, `desc` null or a D3D12_CONSTANT_BUFFER_VIEW_DESC
    // {BufferLocation u64, SizeInBytes u32}.
    unsafe {
        let pc = (*this).pc;
        // a view of no bytes is a null view: the PC runtime removes the device on a size of 0
        if desc.is_null() || std::ptr::read_unaligned(desc.cast::<u8>().add(8).cast::<u32>()) == 0 {
            // the slot no longer holds an unaligned view either, or a draw would make the old one again
            crate::title_heap::note_cbv(host_cpu, None);
            method::<CreateCbv>(pc, 17)(pc, std::ptr::null(), host_cpu);
            crate::view_shadow::view_made(host_cpu, [0; 4]);
            return;
        }
        let va = std::ptr::read_unaligned(desc.cast::<u64>());
        let title_size = std::ptr::read_unaligned(desc.cast::<u8>().add(8).cast::<u32>());
        let size = title_size.next_multiple_of(256);
        // A view that does not start on 256 bytes is made again at each draw that reads it, over a
        // copy of its bytes in upload memory (`list_mirror::sync_tables`).
        crate::title_heap::note_cbv(host_cpu, (va & 255 != 0).then_some((va, title_size)));
        // PC constant buffers start on 256 bytes; the console allows less. The view starts at the
        // 256 byte boundary below (the shader then reads from that earlier start; the bytes are
        // still the title's), as an unaligned start is an invalid call that removes the device.
        let aligned = va & !255;
        if aligned != va {
            static UNALIGNED: AtomicI32 = AtomicI32::new(0);
            if log_due(&UNALIGNED) {
                xs_log!("console device slot 17 (CreateConstantBufferView): title address {va:#x} is not on 256 bytes, the view starts at {aligned:#x}");
            }
        }
        let va = aligned;
        let Some(location) = (if va == 0 { Some(0) } else { pc_gpu_address(va, u64::from(size)) }) else {
            if log_due(&SKIPPED) {
                xs_log!("console device slot 17 (CreateConstantBufferView): title address {va:#x} ({size} bytes) is in no PC resource, view left out");
            }
            return;
        };
        let location = location & !255;
        let mut pc_desc = [0u8; 16];
        pc_desc[..8].copy_from_slice(&location.to_le_bytes());
        pc_desc[8..12].copy_from_slice(&size.to_le_bytes());
        crate::view_diag::view_made(host_cpu, crate::view_diag::View { res: 0, shape: (0, 0, 0, 0), format: 0, dim: 0, va, kind: "cbv" });
        method::<CreateCbv>(pc, 17)(pc, pc_desc.as_ptr().cast(), host_cpu);
        crate::view_shadow::view_made(host_cpu, [0; 4]);
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
}

/// Notes the view written into `host_cpu` for the descriptor diagnostics (`view_diag`).
///
/// # Safety
/// `desc` is null or a view description, which starts with its format and dimension.
unsafe fn note_view(host_cpu: usize, res: Object, desc: *const c_void, va: u64, kind: &'static str) {
    if !crate::view_diag::enabled() {
        return;
    }
    // SAFETY: by the contract above.
    let (format, dim) = if desc.is_null() { (0, 0) } else { unsafe { (std::ptr::read_unaligned(desc.cast::<u32>()), std::ptr::read_unaligned(desc.cast::<u32>().add(1))) } };
    let shape = if res.is_null() {
        (0, 0, 0, 0)
    } else {
        // SAFETY: the title makes a view of a live resource.
        let d = unsafe { resource_desc(res) };
        let word = |at: usize| u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]);
        (u64::from_le_bytes(d[0x10..0x18].try_into().unwrap_or_default()), word(0x18), word(0x20), word(0x34))
    };
    crate::view_diag::view_made(host_cpu, crate::view_diag::View { res: res as usize, shape, format, dim, va, kind });
}

/// The `D3D12_RESOURCE_DESC` of a live PC resource (GetDesc, PC slot 10, which returns its 56 bytes
/// through a hidden pointer).
///
/// # Safety
/// `res` is a live PC resource.
unsafe fn resource_desc(res: Object) -> [u8; 0x38] {
    let mut d = [0u8; 0x38];
    // SAFETY: by the contract above.
    unsafe { method::<unsafe extern "system" fn(Object, *mut c_void) -> *mut c_void>(res, 10)(res, d.as_mut_ptr().cast()) };
    d
}

/// The view shadow row of a view just made of `res` (null: a null view) with the description `desc`
/// (`None`: the resource's own view): the records of a raw view, the image words of a texture view,
/// zeros for any other.
///
/// # Safety
/// `res` is null or a live PC resource.
unsafe fn shadow_row(res: Object, desc: Option<&[u8; 40]>, kind: crate::view_raw::ViewKind, records: Option<crate::view_raw::Records>) -> crate::view_shadow::Row {
    if let Some(r) = records {
        return crate::view_shadow::records_row(r);
    }
    // image rows are for the shaders only
    if res.is_null() || !crate::view_shadow::wanted() {
        return [0; 4];
    }
    // SAFETY: by the contract above.
    let resource = unsafe { resource_desc(res) };
    crate::view_image::image_words(desc, kind, &resource).map_or([0; 4], |[size, shape]| [0, 0, size, shape])
}

/// The resource flag a view needs on PC: `ALLOW_RENDER_TARGET`, `ALLOW_DEPTH_STENCIL`,
/// `ALLOW_UNORDERED_ACCESS`, and for a shader resource view the absence of `DENY_SHADER_RESOURCE`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewNeeds {
    ShaderResource,
    UnorderedAccess,
    RenderTarget,
    DepthStencil,
}

/// Little endian `u32` at `at` of a byte description.
fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Whether a resource with `flags` may be the target of a view of `format` that needs `needs`.
fn view_flags_allow(needs: ViewNeeds, flags: u32, format: u32) -> bool {
    // B5G6R5 (85), B5G5R5A1 (86) and B4G4R4A4 (115) have no typed unordered access on PC, and
    // B4G4R4A4 is no PC render target either
    match needs {
        ViewNeeds::ShaderResource => flags & 0x8 == 0,
        ViewNeeds::UnorderedAccess => flags & 0x4 != 0 && !matches!(format, 85 | 86 | 115),
        ViewNeeds::RenderTarget => flags & 0x1 != 0 && format != 115,
        ViewNeeds::DepthStencil => flags & 0x2 != 0,
    }
}

/// The PC resource a view is made of, or null (a null descriptor) when the PC refuses the view:
/// its format is not in the resource's format family, the resource lacks the flag the view needs,
/// or the PC has no such view of the format. The console allows all of these, PC D3D12 removes
/// the device on them. Seen in measured runs: an R9G9B9E5 view of an R32_UINT texture, and views of
/// a cube depth target whose PC copy lost its unordered access flag in `pc_resource_desc`. A
/// plane slice on a resource of one plane is set to 0 in `desc`, as seen with an R8_UINT unordered
/// access view of plane 1. `desc` is the view description, format first.
///
/// # Safety
/// `res` is null or a live PC resource.
unsafe fn view_resource(what: &str, res: Object, desc: Option<&mut [u8]>, needs: ViewNeeds) -> Object {
    static FORMAT_REFUSED: AtomicI32 = AtomicI32::new(0);
    static FLAGS_REFUSED: AtomicI32 = AtomicI32::new(0);
    if res.is_null() {
        return res;
    }
    // SAFETY: by the contract above.
    let resource_bytes = unsafe { resource_desc(res) };
    let resource_flags = le_u32(&resource_bytes, 0x30);
    let resource_format = le_u32(&resource_bytes, 0x20);
    let view_format = desc.as_deref().map_or(0, |v| le_u32(v, 0));
    // a view without a format takes the format of the resource
    let effective = if view_format == 0 { resource_format } else { view_format };
    if !view_flags_allow(needs, resource_flags, effective) {
        if log_due(&FLAGS_REFUSED) {
            xs_log!("console device: {what} of format {effective} on a resource with flags 0x{resource_flags:X} is not allowed on PC, a null view is made instead");
        }
        null_view_desc(desc, needs);
        return std::ptr::null_mut();
    }
    let Some(desc) = desc else { return res };
    // only the depth and stencil formats R32G8X24 (19..=22) and R24G8 (44..=47) have a second plane
    let two_planes = matches!(resource_format, 19..=22 | 44..=47);
    if !two_planes && let Some(at) = plane_slice_at(desc, needs) {
        desc[at..at + 4].fill(0);
    }
    // a depth view's format is the depth one of the family, which the family table does not hold
    if needs == ViewNeeds::DepthStencil || crate::abi::view_format_fits(view_format, resource_format) {
        return res;
    }
    if log_due(&FORMAT_REFUSED) {
        xs_log!("console device: {what} of format {view_format} on a resource of format {resource_format} is not allowed on PC, a null view is made instead");
    }
    null_view_desc(Some(desc), needs);
    std::ptr::null_mut()
}

// Where the plane slice of a 2D or 2D array view description lies, by view kind (none for a
/// depth view or another dimension). The members of a shader resource view start at 0x10, after the
/// component mapping: `MostDetailedMip`, `MipLevels`, then the plane slice of a 2D view, or
/// `FirstArraySlice`, `ArraySize` and then the plane slice of a 2D array view.
fn plane_slice_at(desc: &[u8], needs: ViewNeeds) -> Option<usize> {
    let dim = u32::from_le_bytes([desc[4], desc[5], desc[6], desc[7]]);
    let at = match (needs, dim) {
        (ViewNeeds::ShaderResource, 4) => 0x18,
        (ViewNeeds::ShaderResource, 5) => 0x20,
        (ViewNeeds::UnorderedAccess | ViewNeeds::RenderTarget, 4) => 12,
        (ViewNeeds::UnorderedAccess | ViewNeeds::RenderTarget, 5) => 20,
        _ => return None,
    };
    (desc.len() >= at + 4).then_some(at)
}

/// A view description the PC takes for a null descriptor: the PC checks it as for a real view, so
/// a refused view's plane slice is 0 and its format one every PC view of that kind takes
/// (measured: the stencil plane of a depth target refused as unordered access view removed the
/// device). Buffer views keep their description, their format can be no other.
fn null_view_desc(desc: Option<&mut [u8]>, needs: ViewNeeds) {
    let Some(desc) = desc else { return };
    let dim = u32::from_le_bytes([desc[4], desc[5], desc[6], desc[7]]);
    if dim == 1 {
        return;
    }
    if let Some(at) = plane_slice_at(desc, needs) {
        desc[at..at + 4].fill(0);
    }
    // R8G8B8A8_UNORM for views read or rendered, R32_UINT for unordered access, D32_FLOAT for depth
    let format: u32 = match needs {
        ViewNeeds::ShaderResource | ViewNeeds::RenderTarget => 28,
        ViewNeeds::UnorderedAccess => 42,
        ViewNeeds::DepthStencil => 40,
    };
    desc[0..4].copy_from_slice(&format.to_le_bytes());
}

/// A copy of the view description at `desc` (`N` bytes, format first) with the PC format of a
/// console only one (`abi::pc_dxgi_format`: the D16S8 family as D24S8, which is what
/// `pc_resource_desc` made the resource), `None` when `desc` is null. The PC removes the device on
/// a view in a console format, such as a stencil view in X16_TYPELESS_G8_UINT (120).
///
/// # Safety
/// `desc` is null or `N` readable bytes.
unsafe fn pc_view_desc<const N: usize>(desc: *const c_void) -> Option<[u8; N]> {
    if desc.is_null() {
        return None;
    }
    // SAFETY: by the contract above.
    let mut copy: [u8; N] = unsafe { std::ptr::read_unaligned(desc.cast()) };
    let mapped = crate::abi::pc_dxgi_format(le_u32(&copy, 0));
    copy[..4].copy_from_slice(&mapped.to_le_bytes());
    Some(copy)
}

/// `res` with the buffer view `desc` cut to fit it (`view_raw::fit`), or null with the description
/// of a null view when none of its elements is in the buffer.
///
/// # Safety
/// `res` is null or a live PC resource.
unsafe fn fitted(res: Object, desc: Option<&mut [u8; crate::view_raw::DESC_BYTES]>, kind: crate::view_raw::ViewKind) -> Object {
    let Some(desc) = desc else { return res };
    if res.is_null() {
        return res;
    }
    // SAFETY: by the contract above.
    let d = unsafe { resource_desc(res) };
    // a buffer resource (dimension 1) has its bytes in the width
    if u32::from_le_bytes([d[0], d[1], d[2], d[3]]) != 1 {
        return res;
    }
    let width = u64::from_le_bytes(d[0x10..0x18].try_into().unwrap_or_default());
    if crate::view_raw::fit(desc, kind, width) {
        return res;
    }
    // no element left: the description of a null view
    crate::view_raw::make_raw(desc, kind);
    std::ptr::null_mut()
}

/// The pointer the PC takes for a view description copied by [`pc_view_desc`].
fn view_desc_ptr<const N: usize>(copy: &Option<[u8; N]>) -> *const c_void {
    copy.as_ref().map_or(std::ptr::null(), |d| d.as_ptr().cast())
}

extern "system" fn dev_create_srv(this: *mut Proxy, res: Object, desc: *const c_void, cpu: usize) {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return };
    // SAFETY: a shader resource view description is 40 bytes.
    let mut copy = unsafe { pc_view_desc::<40>(desc) };
    // SAFETY: `this` is a live device proxy and `res` is null or a resource/proxy.
    unsafe {
        let pc = (*this).pc;
        let res = view_resource("shader resource view", unwrap(res), copy.as_mut().map(|d| &mut d[..]), ViewNeeds::ShaderResource);
        let outcome = if res.is_null() { crate::view_raw::Outcome::NotStructured } else { crate::view_raw::convert(copy.as_mut(), crate::view_raw::ViewKind::ShaderResource, false, "console device slot 18 (CreateShaderResourceView)") };
        let records = outcome.raw();
        let res = if outcome == crate::view_raw::Outcome::Empty { std::ptr::null_mut() } else { fitted(res, copy.as_mut(), crate::view_raw::ViewKind::ShaderResource) };
        let desc = view_desc_ptr(&copy);
        if res.is_null() && desc.is_null() {
            return;
        }
        note_view(host_cpu, res, desc, 0, "srv");
        crate::title_heap::note_cbv(host_cpu, None);
        method::<CreateSrv>(pc, 18)(pc, res, desc, host_cpu);
        crate::view_shadow::view_made(host_cpu, shadow_row(res, copy.as_ref(), crate::view_raw::ViewKind::ShaderResource, records));
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
}

extern "system" fn dev_create_uav(this: *mut Proxy, res: Object, counter: Object, desc: *const c_void, cpu: usize) {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return };
    // SAFETY: an unordered access view description is 40 bytes.
    let mut copy = unsafe { pc_view_desc::<40>(desc) };
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        let res = view_resource("unordered access view", unwrap(res), copy.as_mut().map(|d| &mut d[..]), ViewNeeds::UnorderedAccess);
        // A raw view has no counter (`view_raw`).
        let outcome = if res.is_null() { crate::view_raw::Outcome::NotStructured } else { crate::view_raw::convert(copy.as_mut(), crate::view_raw::ViewKind::UnorderedAccess, !counter.is_null(), "console device slot 19 (CreateUnorderedAccessView)") };
        let records = outcome.raw();
        let res = if outcome == crate::view_raw::Outcome::Empty { std::ptr::null_mut() } else { fitted(res, copy.as_mut(), crate::view_raw::ViewKind::UnorderedAccess) };
        let desc = view_desc_ptr(&copy);
        if res.is_null() && desc.is_null() {
            return;
        }
        let counter = if res.is_null() || records.is_some() { std::ptr::null_mut() } else { unwrap(counter) };
        note_view(host_cpu, res, desc, 0, "uav");
        crate::title_heap::note_cbv(host_cpu, None);
        method::<CreateUav>(pc, 19)(pc, res, counter, desc, host_cpu);
        crate::view_shadow::view_made(host_cpu, shadow_row(res, copy.as_ref(), crate::view_raw::ViewKind::UnorderedAccess, records));
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
}

/// The size, format, mip count and flags of a live PC resource (`D3D12_RESOURCE_DESC`, PC slot
/// 10, same generic vtable call as `view_resource`'s format check above). Read once, while the
/// resource is known live (at RTV creation): `rt_diag` keeps the string so no later diagnostic
/// calls `GetDesc()` on a resource that may already be released.
///
/// # Safety
/// `res` is null or a live PC resource.
unsafe fn describe_resource(res: Object) -> String {
    if res.is_null() {
        return "none".to_owned();
    }
    // SAFETY: by the contract above.
    let d = unsafe { resource_desc(res) };
    let u16_at = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
    let u64_at = |o: usize| u64::from_le_bytes(d[o..o + 8].try_into().unwrap_or_default());
    format!("{}x{} format {} mips {} flags {:#x}", u64_at(0x10), u32_at(0x18), u32_at(0x20), u16_at(0x1E), u32_at(0x30))
}

extern "system" fn dev_create_rtv(this: *mut Proxy, res: Object, desc: *const c_void, cpu: usize) {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else {
        crate::rt_diag::count(&crate::rt_diag::RTV_UNTRANSLATED);
        return;
    };
    // SAFETY: a render target view description is 24 bytes.
    let mut copy = unsafe { pc_view_desc::<24>(desc) };
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        let res = view_resource("render target view", unwrap(res), copy.as_mut().map(|d| &mut d[..]), ViewNeeds::RenderTarget);
        let desc = view_desc_ptr(&copy);
        if res.is_null() && desc.is_null() {
            return;
        }
        crate::rt_diag::rtv_made(host_cpu, res as usize, || describe_resource(res));
        method::<CreateRtv>(pc, 20)(pc, res, desc, host_cpu);
    }
}

extern "system" fn dev_create_dsv(this: *mut Proxy, res: Object, desc: *const c_void, cpu: usize) {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return };
    // SAFETY: a depth stencil view description is 24 bytes.
    let mut copy = unsafe { pc_view_desc::<24>(desc) };
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        let res = view_resource("depth stencil view", unwrap(res), copy.as_mut().map(|d| &mut d[..]), ViewNeeds::DepthStencil);
        let desc = view_desc_ptr(&copy);
        if res.is_null() && desc.is_null() {
            return;
        }
        method::<CreateDsv>(pc, 21)(pc, res, desc, host_cpu);
    }
}

extern "system" fn dev_create_sampler(this: *mut Proxy, desc: *const c_void, cpu: usize) {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return };
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        method::<CreateSamplerPc>(pc, 22)(pc, desc, host_cpu);
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
}

#[allow(clippy::too_many_arguments)]
extern "system" fn dev_copy_descriptors(
    this: *mut Proxy,
    n_dst: u32,
    dst_starts: *const usize,
    dst_sizes: *const u32,
    n_src: u32,
    src_starts: *const usize,
    src_sizes: *const u32,
    kind: u32,
) {
    // SAFETY: `this` is a live device proxy; arrays hold `n_dst` and `n_src` handles when non-null.
    unsafe {
        let pc = (*this).pc;
        let mut dst_buf: Vec<usize> = Vec::new();
        let mut src_buf: Vec<usize> = Vec::new();
        let dst_ptr = if !dst_starts.is_null() && n_dst > 0 {
            let slice = std::slice::from_raw_parts(dst_starts, n_dst as usize);
            dst_buf.reserve(slice.len());
            for &h in slice {
                let Some(t) = crate::title_heap::translate_cpu(h) else { return };
                dst_buf.push(t);
            }
            dst_buf.as_ptr()
        } else {
            dst_starts
        };
        let src_ptr = if !src_starts.is_null() && n_src > 0 {
            let slice = std::slice::from_raw_parts(src_starts, n_src as usize);
            src_buf.reserve(slice.len());
            for &h in slice {
                let Some(t) = crate::title_heap::translate_cpu(h) else { return };
                src_buf.push(t);
            }
            src_buf.as_ptr()
        } else {
            src_starts
        };
        if crate::view_diag::enabled() && kind == 0 && !dst_ptr.is_null() && !src_ptr.is_null() {
            note_copies(pc, n_dst, dst_ptr, dst_sizes, n_src, src_ptr, src_sizes);
        }
        method::<CopyDescriptorsPc>(pc, 23)(pc, n_dst, dst_ptr, dst_sizes, n_src, src_ptr, src_sizes, kind);
        if kind == 0 && !dst_ptr.is_null() && !src_ptr.is_null() {
            copy_shadow_rows(pc, (n_dst, dst_ptr, dst_sizes), (n_src, src_ptr, src_sizes));
        }
        if kind <= 1 && !dst_starts.is_null() && !src_starts.is_null() {
            copy_title_tags(pc, kind, (n_dst, dst_starts, dst_sizes), (n_src, src_starts, src_sizes));
        }
    }
}

/// Ranges of descriptor handles: their number, their starts and their sizes (null for ranges of one).
type Ranges = (u32, *const usize, *const u32);

/// Copies the slot tags of a `CopyDescriptors` in title memory, slot by slot.
///
/// # Safety
/// The arrays hold as many entries as the ranges say.
unsafe fn copy_title_tags(pc: Object, kind: u32, dst: Ranges, src: Ranges) {
    // SAFETY: by the contract above.
    unsafe {
        let inc = method::<GetIncSize>(pc, 15)(pc, kind) as usize;
        let size = |sizes: *const u32, i: usize| if sizes.is_null() { 1 } else { *sizes.add(i) as usize };
        let flat = |(n, starts, sizes): Ranges| -> Vec<usize> { (0..n as usize).flat_map(|i| (0..size(sizes, i)).map(move |k| *starts.add(i) + k * inc)).collect() };
        for (d, s) in flat(dst).into_iter().zip(flat(src)) {
            crate::title_heap::copy_tags(d, s, 1, inc as u32);
        }
    }
}

/// Copies the view shadow rows of a `CopyDescriptors` of views, slot by slot (host handles).
///
/// # Safety
/// The arrays hold as many entries as the ranges say.
unsafe fn copy_shadow_rows(pc: Object, dst: Ranges, src: Ranges) {
    // SAFETY: by the contract above.
    unsafe {
        let inc = method::<GetIncSize>(pc, 15)(pc, 0) as usize;
        let size = |sizes: *const u32, i: usize| if sizes.is_null() { 1 } else { *sizes.add(i) as usize };
        let flat = |(n, starts, sizes): Ranges| -> Vec<usize> { (0..n as usize).flat_map(|i| (0..size(sizes, i)).map(move |k| *starts.add(i) + k * inc)).collect() };
        for (d, s) in flat(dst).into_iter().zip(flat(src)) {
            crate::view_shadow::copied(d, s, 1);
        }
    }
}

/// Notes the slots a `CopyDescriptors` of views copies for the descriptor diagnostics.
///
/// # Safety
/// The arrays hold `n_dst` and `n_src` entries, a null size array means ranges of one.
unsafe fn note_copies(pc: Object, n_dst: u32, dst: *const usize, dst_sizes: *const u32, n_src: u32, src: *const usize, src_sizes: *const u32) {
    // SAFETY: by the contract above.
    unsafe {
        let inc = method::<GetIncSize>(pc, 15)(pc, 0) as usize;
        let size = |sizes: *const u32, i: usize| if sizes.is_null() { 1 } else { *sizes.add(i) as usize };
        let flat = |starts: *const usize, sizes: *const u32, n: u32| -> Vec<usize> { (0..n as usize).flat_map(|i| (0..size(sizes, i)).map(move |k| *starts.add(i) + k * inc)).collect() };
        for (d, s) in flat(dst, dst_sizes, n_dst).into_iter().zip(flat(src, src_sizes, n_src)) {
            crate::view_diag::copied(d, s, 1, inc as u32);
        }
    }
}

extern "system" fn dev_copy_descriptors_simple(this: *mut Proxy, count: u32, dst_start: usize, src_start: usize, kind: u32) {
    let Some(dst) = crate::title_heap::translate_cpu(dst_start) else { return };
    let Some(src) = crate::title_heap::translate_cpu(src_start) else { return };
    // SAFETY: `this` is a live device proxy.
    unsafe {
        let pc = (*this).pc;
        if kind == 0 && crate::view_diag::enabled() {
            crate::view_diag::copied(dst, src, count, method::<GetIncSize>(pc, 15)(pc, 0));
        }
        method::<CopyDescriptorsSimplePc>(pc, 24)(pc, count, dst, src, kind);
        if kind == 0 {
            crate::view_shadow::copied(dst, src, count);
        }
        if kind <= 1 {
            crate::title_heap::copy_tags(dst_start, src_start, count, method::<GetIncSize>(pc, 15)(pc, kind));
        }
    }
}

/// Root signature containers from the title reach the PC device only without the console-only
/// flag bits (round 13 of the port). Checked once per call: root signature creation is not a per
/// frame path.
const ROOT_SIGNATURE_HEAP_FLAG: u32 = 0x400;

/// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED`.
const ROOT_SIGNATURE_SAMPLER_HEAP_FLAG: u32 = 0x800;

/// `DECANT_D3D12X_RS_HEAP=1` (default off): also sets `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`
/// on every root signature the console device creates. The lowered builder compute shaders that
/// index a descriptor heap directly (SM 6.6 `createHandleFromHeap`) need the PC runtime to see
/// this bit or it never emits the binding they rely on. Off by default, named like `DECANT_D3D12X_PROXY`.
fn rs_heap_on() -> bool {
    std::env::var_os("DECANT_D3D12X_RS_HEAP").is_some_and(|v| v == "1")
}

/// The static samplers of the title's root signatures, by register and space, the first seen kept,
/// and the immediate samplers of the translated shaders (`immediate_samplers.txt` of the override
/// folder, written by `decant_shader_dxil_cache`).
mod static_samplers {
    use std::sync::Mutex;

    static SEEN: Mutex<Option<Vec<crate::rts0::SamplerWords>>> = Mutex::new(None);

    /// The immediate samplers the override folder lists: per line a first descriptor word, or a
    /// register and the four descriptor words, in hex (`rts0::immediate_samplers`).
    fn immediate() -> Vec<crate::rts0::SamplerWords> {
        let Some(dir) = crate::dxil_override::configured_dir() else { return Vec::new() };
        let text = std::fs::read_to_string(dir.join("immediate_samplers.txt")).unwrap_or_default();
        let words = crate::rts0::immediate_samplers(&text);
        crate::log::xs_log!("console device slot 16 (CreateRootSignature): {} immediate samplers of the translated shaders", words.len());
        words
    }

    /// Learns the static samplers of the root signature `blob`, then gives a copy of it with the ones
    /// it lacks and the number added, `None` when it lacks none.
    pub(super) fn complete(blob: &[u8]) -> Option<(Vec<u8>, usize)> {
        let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = guard.get_or_insert_with(immediate);
        for s in crate::rts0::static_sampler_words(blob) {
            if !seen.iter().any(|k| (k[10], k[11]) == (s[10], s[11])) {
                seen.push(s);
            }
        }
        crate::rts0::append_static_samplers(blob, seen)
    }
}

/// Console device slot 16 (PC form): CreateRootSignature, formerly a raw forward. Console-only
/// RTS0 flag bits are cleared from a copy of the blob before the host parses it (same fix as the
/// serializers and deserializers of `exports`), and `rs_heap_on` optionally sets the
/// heap-directly-indexed bit. Root signature objects are not proxied (their vtable matches the PC
/// one), so the host's object goes straight to `*out`.
extern "system" fn dev_create_root_signature(this: *mut Proxy, node_mask: u32, blob: *const c_void, len: usize, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy; `blob` is null or `len` readable bytes, `riid` readable and
    // `out` writable, by the COM contract.
    unsafe {
        *out = std::ptr::null_mut();
        let pc = (*this).pc;
        if blob.is_null() || len == 0 {
            return method::<CreateRootSig>(pc, 16)(pc, node_mask, blob, len, riid, out);
        }
        let bytes = std::slice::from_raw_parts(blob.cast::<u8>(), len);
        let mut copy: Option<Vec<u8>> = None;
        if rts0::flags_offset(bytes).is_some_and(|at| rts0::rd32(bytes, at) & !rts0::PUBLIC_FLAGS != 0) {
            let mut c = bytes.to_vec();
            let extra = rts0::strip_unknown_flags(&mut c);
            xs_log!("console device slot 16 (CreateRootSignature): cleared console-only flags 0x{extra:08X}");
            copy = Some(c);
        }
        let overriding = dxil_override::configured_dir().is_some();
        // Directly translated shaders read descriptors the root signature does not cover, and
        // their immediate samplers, from the heaps (`direct_pso`, `sampler_registry`).
        let direct = crate::compile_service::direct_enabled();
        if rs_heap_on() || overriding || direct {
            let mut c = copy.take().unwrap_or_else(|| bytes.to_vec());
            if rts0::add_public_flag(&mut c, ROOT_SIGNATURE_HEAP_FLAG) {
                xs_log!("console device slot 16 (CreateRootSignature): added CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED (DECANT_D3D12X_RS_HEAP, DECANT_D3D12X_DXIL_OVERRIDE or DECANT_D3D12X_DIRECT)");
            }
            // Only translated shaders need the sampler heap bit.
            if (overriding || direct) && rts0::add_public_flag(&mut c, ROOT_SIGNATURE_SAMPLER_HEAP_FLAG) {
                xs_log!("console device slot 16 (CreateRootSignature): added SAMPLER_HEAP_DIRECTLY_INDEXED (DECANT_D3D12X_DXIL_OVERRIDE or DECANT_D3D12X_DIRECT)");
            }
            copy = Some(c);
        }
        // Translated shaders read the static samplers they were compiled with, and the title's root
        // signatures do not all have them (the console never checks). Under an override every root
        // signature gets the static samplers of the title's other root signatures it lacks.
        if overriding {
            let base = copy.as_deref().unwrap_or(bytes);
            if let Some((c, added)) = static_samplers::complete(base) {
                static SHOWN: AtomicI32 = AtomicI32::new(0);
                if log_due(&SHOWN) {
                    xs_log!("console device slot 16 (CreateRootSignature): added {added} static samplers of other root signatures (DECANT_D3D12X_DXIL_OVERRIDE)");
                }
                copy = Some(c);
            }
        }
        // Under an override the root signature also gets the user data mirror (see `user_mirror`).
        // The same goes for the device memory window and the service buffers (`window_service`).
        let mut with_mirror: Option<Vec<u8>> = None;
        // Directly translated shaders also get the heap map and the view shadow of the bound view
        // heap, next to the mirror.
        let mirror = crate::user_mirror::wanted();
        let want = rts0::AbiWant { mirror, window: crate::window_service::wanted(), heap_map: mirror && direct, view_shadow: mirror && direct };
        if want.mirror || want.window {
            let base = copy.as_deref().unwrap_or(bytes);
            match rts0::append_abi_params(base, want) {
                Ok((c, added)) => {
                    with_mirror = Some(c);
                    let views_left_out = (want.heap_map && !added.heap_map) || (want.view_shadow && !added.view_shadow);
                    if (want.mirror && !added.mirror) || views_left_out || (want.window && !(added.window && added.service)) {
                        xs_log!("console device slot 16 (CreateRootSignature): parameters left out for the 64 dword limit, added {added:?}");
                    }
                }
                Err(why) => xs_log!("console device slot 16 (CreateRootSignature): no mirror, window or service parameters ({why:?})"),
            }
        }
        let first = with_mirror.as_ref().or(copy.as_ref());
        let (use_ptr, use_len) = match first {
            Some(c) => (c.as_ptr().cast(), c.len()),
            None => (blob, len),
        };
        let mut hr = method::<CreateRootSig>(pc, 16)(pc, node_mask, use_ptr, use_len, riid, out);
        let (use_ptr, use_len) = if hr < 0 && with_mirror.is_some() {
            xs_log!("console device slot 16 (CreateRootSignature): refused with the user data mirror (hr=0x{:08X}), made without it", hr as u32);
            let (p, l): (*const c_void, usize) = match &copy {
                Some(c) => (c.as_ptr().cast(), c.len()),
                None => (blob, len),
            };
            hr = method::<CreateRootSig>(pc, 16)(pc, node_mask, p, l, riid, out);
            (p, l)
        } else {
            (use_ptr, use_len)
        };
        // Keep the layout of the blob the host got, for the coverage check of DXIL overrides and
        // for the translator of the direct path.
        if (overriding || direct) && hr >= 0 && !(*out).is_null() {
            let seen = std::slice::from_raw_parts(use_ptr.cast::<u8>(), use_len);
            if let Some(layout) = rts0::parse_root_layout(seen) {
                static SHOWN: AtomicI32 = AtomicI32::new(0);
                if SHOWN.fetch_add(1, Ordering::Relaxed) < 6 {
                    xs_log!("console device slot 16 (CreateRootSignature): title layout {:?}", layout);
                }
                if crate::view_diag::enabled() {
                    crate::view_diag::signature_made(*out as usize, format!("{layout:?}"));
                }
                if crate::user_mirror::remember_signature(*out as usize, &layout) {
                    xs_log!(
                        "console device slot 16 (CreateRootSignature): the user data mirror is root parameter {}, the heap map {:?}, the view shadow {:?}",
                        layout.mirror_index().unwrap_or(0),
                        layout.heap_map_index(),
                        layout.view_shadow_index()
                    );
                }
                if crate::window_service::remember_signature(*out as usize, &layout) {
                    xs_log!(
                        "console device slot 16 (CreateRootSignature): window parameters {:?}, service parameters {:?}",
                        (layout.window_uav_index(), layout.window_base_index()),
                        layout.service_indices()
                    );
                }
                dxil_override::remember_root_layout(*out as usize, layout);
            }
        }
        hr
    }
}

/// Looks up a DXIL override (`dxil_override`) for `desc.cs` and, on a hit, returns a patched copy
/// of `desc` pointing at it plus the override buffer (kept alive by the caller until after the
/// host call). `None` on a miss: the caller uses `desc` unchanged.
///
/// # Safety
/// `desc` is a live, readable `ComputePsoDesc`.
unsafe fn overridden_compute_desc(desc: *const ComputePsoDesc) -> (Option<(ComputePsoDesc, Vec<u8>)>, Vec<crate::pso_diag::Stage>) {
    // SAFETY: by the contract above.
    let mut patched = unsafe { std::ptr::read_unaligned(desc) };
    let original = if patched.cs.ptr.is_null() || patched.cs.len == 0 {
        &[][..]
    } else {
        // SAFETY: `cs.ptr`/`cs.len` describe the bytecode the caller attached to this desc.
        unsafe { std::slice::from_raw_parts(patched.cs.ptr, patched.cs.len) }
    };
    let (overridden, outcome) = dxil_override::lookup_for_pipeline_outcome(original, patched.root_signature as usize, 0);
    let stages = vec![crate::pso_diag::Stage { name: "CS", hash: dxil_override::content_hash(original), outcome }];
    let Some(overridden) = overridden else { return (None, stages) };
    xs_log!(
        "DXIL override hit, hash={:016x}, size {}->{} (DECANT_D3D12X_DXIL_OVERRIDE)",
        dxil_override::content_hash(original),
        original.len(),
        overridden.len()
    );
    patched.cs = ShaderBytecode { ptr: overridden.as_ptr(), len: overridden.len() };
    (Some((patched, overridden)), stages)
}

/// The number of shader stages a plain graphics description has.
fn graphics_stages(g: &[u8; crate::stream_pso::GRAPHICS_DESC_SIZE]) -> usize {
    [0x08usize, 0x18, 0x28, 0x38, 0x48]
        .iter()
        .filter(|&&at| g[at..at + 8].iter().any(|b| *b != 0) && g[at + 8..at + 16].iter().any(|b| *b != 0))
        .count()
}

/// Looks up DXIL overrides (`dxil_override`) for the vertex, pixel, domain, hull and geometry
/// shaders of a plain graphics description and points the description at them. The override
/// buffers are returned and must outlive the host call, empty when nothing was replaced, and every
/// stage with what became of its override.
fn override_graphics_shaders(g: &mut [u8; crate::stream_pso::GRAPHICS_DESC_SIZE]) -> (Vec<Vec<u8>>, Vec<crate::pso_diag::Stage>) {
    let mut kept = Vec::new();
    let mut stages = Vec::new();
    let root_sig = crate::stream_pso::graphics_root_sig(g);
    // The stage and its D3D12_SHADER_VISIBILITY.
    for (stage, at, visibility) in [("VS", 0x08usize, 1u32), ("PS", 0x18, 5), ("DS", 0x28, 3), ("HS", 0x38, 2), ("GS", 0x48, 4)] {
        let ptr = u64::from_le_bytes(g[at..at + 8].try_into().unwrap_or_default()) as *const u8;
        let len = u64::from_le_bytes(g[at + 8..at + 16].try_into().unwrap_or_default()) as usize;
        if ptr.is_null() || len == 0 {
            continue;
        }
        // SAFETY: the description's shader bytecode pointer and length are the title's.
        let original = unsafe { std::slice::from_raw_parts(ptr, len) };
        let (replacement, outcome) = dxil_override::lookup_for_pipeline_outcome(original, root_sig, visibility);
        let hash = dxil_override::content_hash(original);
        stages.push(crate::pso_diag::Stage { name: stage, hash, outcome });
        let Some(replacement) = replacement else { continue };
        xs_log!("DXIL override hit ({stage}), hash={hash:016x}, size {}->{}", len, replacement.len());
        g[at..at + 8].copy_from_slice(&(replacement.as_ptr() as u64).to_le_bytes());
        g[at + 8..at + 16].copy_from_slice(&(replacement.len() as u64).to_le_bytes());
        kept.push(replacement);
    }
    (kept, stages)
}

/// The render target and depth formats of a plain graphics description, for the log of a refused
/// pipeline.
fn graphics_targets(g: &[u8; crate::stream_pso::GRAPHICS_DESC_SIZE]) -> String {
    let u32_at = |at: usize| u32::from_le_bytes(g[at..at + 4].try_into().unwrap_or_default());
    let n = u32_at(0x240).min(8) as usize;
    let formats: Vec<u32> = (0..n).map(|i| u32_at(0x244 + 4 * i)).collect();
    format!(", targets {formats:?}, depth {}", u32_at(0x264))
}

/// Console device slot 11 (PC form of CreateComputePipelineState; per `abi.rs`, not called by the
/// package but kept in the console order). A DXIL override is looked up before the host builds the
/// pipeline, same as `dxil_override`'s `slot11` call site in the C layer. Made on first bind
/// (`lazy_pso`) unless that is switched off.
extern "system" fn dev_create_compute_pipeline_state(this: *mut Proxy, desc: *const ComputePsoDesc, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the arguments are the caller's, by the PC contract.
    unsafe { defer_compute(this, desc.cast(), riid, out, compute_pso_pc_now) }
}

/// Answers a compute pipeline creation with a pipeline made on first bind (`lazy_pso`), or makes
/// it now with `now` when that is off or the arguments are not usable.
///
/// # Safety
/// `this` is a live device proxy, `desc` null or a readable `D3D12_COMPUTE_PIPELINE_STATE_DESC`,
/// `out` null or writable.
unsafe fn defer_compute(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void, now: crate::lazy_pso::Create) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        if !crate::lazy_pso::enabled() || desc.is_null() || out.is_null() {
            return now(this as usize, desc, riid, out);
        }
        crate::lazy_pso::defer(this as usize, crate::lazy_pso::copy_compute(desc), riid, out, now)
    }
}

/// Answers a graphics pipeline creation with a pipeline made on first bind (`lazy_pso`), or makes
/// it now with `now` when that is off or the arguments are not usable.
///
/// # Safety
/// `this` is a live device proxy, `desc` null or a readable `D3D12_GRAPHICS_PIPELINE_STATE_DESC`,
/// `out` null or writable.
unsafe fn defer_graphics(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void, now: crate::lazy_pso::Create) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        if !crate::lazy_pso::enabled() || desc.is_null() || out.is_null() {
            return now(this as usize, desc, riid, out);
        }
        let g: &[u8; crate::stream_pso::GRAPHICS_DESC_SIZE] = &*desc.cast();
        crate::lazy_pso::defer(this as usize, crate::lazy_pso::copy_graphics(g), riid, out, now)
    }
}

/// Device slot 11 made now: the override lookup and the PC creation.
///
/// # Safety
/// As for [`defer_compute`], `this` given as an address.
unsafe fn compute_pso_pc_now(this: usize, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    let (this, desc) = (this as *mut Proxy, desc.cast::<ComputePsoDesc>());
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `desc` is the caller's (null or readable), `out` writable.
    unsafe {
        *out = std::ptr::null_mut();
        if desc.is_null() {
            return E_INVALIDARG;
        }
        let pc = (*this).pc;
        let (overridden, stages) = overridden_compute_desc(desc);
        let effective = overridden.as_ref().map_or(desc, |(patched, _)| patched);
        let hr = method::<CreateComputePso>(pc, 11)(pc, effective, riid, out);
        crate::rt_diag::pipeline(hr >= 0, None);
        if hr < 0 { crate::pso_diag::refused(hr, "compute", &stages, "", out) } else { hr }
    }
}

/// Logs the first few console descriptions the PC runtime refused with an all ones size.
fn log_rejected_description(slot: u32, desc: &[u8; 0x38]) {
    static REJECTED: AtomicI32 = AtomicI32::new(0);
    if REJECTED.fetch_add(1, Ordering::Relaxed) >= 16 {
        return;
    }
    let word = |at: usize| le_u32(desc, at);
    let half = |at: usize| u16::from_le_bytes([desc[at], desc[at + 1]]);
    let quad = |at: usize| u64::from_le_bytes(desc[at..at + 8].try_into().unwrap_or_default());
    xs_log!(
        "console device slot {slot}: host rejected desc (dim {}, align {:#x}, {}x{}x{}, mips {}, format {}, samples {}/{}, layout {}, flags {:#x})",
        word(0), quad(8), quad(16), word(24), half(28), half(30), word(32), word(36), word(40), word(44), word(48)
    );
}

/// The allocation `(size, alignment)` of one console resource description `console_desc` as the title
/// sees it on device `pc`: the PC runtime is asked about the description made PC conform, the
/// answer is replaced by the console layout of a texture where XG knows it (`alloc_info`), buffers
/// keep the PC answer. The first few descriptions the PC runtime refused are logged.
///
/// # Safety
/// `pc` is a live PC device.
unsafe fn allocation_of(pc: Object, slot: u32, mask: u32, console_desc: &[u8; 0x38]) -> (u64, u64) {
    let mut pc_desc = *console_desc;
    pc_resource_desc(&mut pc_desc);
    let mut pc_answer = [0u64; 2];
    // SAFETY: by the contract above, the answer takes two words and the description is 0x38 bytes.
    unsafe { method::<AllocInfo>(pc, 25)(pc, (&raw mut pc_answer).cast(), mask, 1, pc_desc.as_ptr().cast()) };
    let answer = console_allocation_answer(slot, console_desc, (pc_answer[0], pc_answer[1]));
    if answer.0 == u64::MAX {
        log_rejected_description(slot, &pc_desc);
    }
    answer
}

/// Console slot 25: GetResourceAllocationInfo. The result pointer and `this` come in either
/// order, see `sret_order`. Every console description (0x38 bytes, console layout, flags and
/// depth formats) is asked of the PC runtime on its own through `allocation_of`, and the sizes are
/// packed one after the other into one total with the largest alignment.
extern "system" fn dev_alloc_info(a: *mut c_void, b: *mut c_void, mask: u32, n: u32, descs: *const c_void) -> *mut c_void {
    let (this, ret) = sret_order(a, b);
    if ret.is_null() {
        return ret;
    }
    // SAFETY: one of `a`/`b` is a live device proxy, `ret` is writable `[u64; 2]`, and `descs`
    // holds `n` 0x38-byte resource descriptions when non-null.
    unsafe {
        let pc = (*this).pc;
        let mut packed = Some((0u64, 0u64));
        for k in 0..n as usize {
            let console_desc: [u8; 0x38] = if descs.is_null() { [0; 0x38] } else { std::ptr::read_unaligned(descs.cast::<[u8; 0x38]>().add(k)) };
            let (size, alignment) = allocation_of(pc, 25, mask, &console_desc);
            let (total, align) = packed.unwrap_or_default();
            packed = crate::abi::accumulate_alloc_info(total, align, size, alignment).map(|(_, total, align)| (total, align));
            if packed.is_none() {
                break;
            }
        }
        // an overflow while packing answers an all ones size
        let (total, align) = packed.unwrap_or((u64::MAX, 0));
        std::ptr::write_unaligned(ret.cast::<[u64; 2]>(), [total, align]);
    }
    ret
}

// The answer of slots 25 and 108 for one console description `raw` whose PC answer is `pc`: the
/// console layout of a texture (`alloc_info`), `pc` for a buffer or what XG refuses. The first
/// distinct descriptions are logged with both answers.
fn console_allocation_answer(slot: u32, raw: &[u8; 0x38], pc: (u64, u64)) -> (u64, u64) {
    let xg = crate::alloc_info::console_allocation(raw, crate::detile::hardware_v4());
    if crate::alloc_info::first_sight(raw) {
        xs_log!("{}", crate::alloc_info::describe(slot, raw, pc, xg));
    }
    xg.unwrap_or(pc)
}

/// Console slot 26: same two possible argument orders as slot 25.
extern "system" fn dev_heap_props(a: *mut c_void, b: *mut c_void, node_mask: u32, heap_type: u32) -> *mut c_void {
    let (this, ret) = sret_order(a, b);
    // SAFETY: as in `dev_alloc_info`.
    unsafe {
        let pc = (*this).pc;
        method::<HeapProps>(pc, 26)(pc, ret, node_mask, heap_type);
    }
    ret
}

/// # Safety
/// `this` is a live device proxy and `objects` points to `n` objects.
unsafe fn dev_residency(this: *mut Proxy, slot: usize, n: u32, objects: *const Object) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        let mapped = unwrap_array(n, objects);
        let pc = (*this).pc;
        method::<Residency>(pc, slot)(pc, n, mapped.as_ptr())
    }
}

extern "system" fn dev_make_resident(this: *mut Proxy, n: u32, objects: *const Object) -> HRESULT {
    // SAFETY: the arguments are the caller's, by the COM contract.
    unsafe { dev_residency(this, 34, n, objects) }
}

extern "system" fn dev_evict(this: *mut Proxy, n: u32, objects: *const Object) -> HRESULT {
    // SAFETY: the arguments are the caller's, by the COM contract.
    unsafe { dev_residency(this, 35, n, objects) }
}

/// The legacy `D3D12_RESOURCE_STATES` bitmask of each newer `D3D12_BARRIER_LAYOUT` value that a
/// resource can be created in, indexed by the layout value.
const LAYOUT_STATES: [u32; 11] = [
    0,      // 0 undefined, common
    0xAC3,  // 1 generic read
    0x4,    // 2 render target
    0x8,    // 3 unordered access
    0x10,   // 4 depth write
    0x20,   // 5 depth read
    0xC0,   // 6 shader resource, all stages
    0x800,  // 7 copy source
    0x400,  // 8 copy dest
    0x2000, // 9 resolve source
    0x1000, // 10 resolve dest
];

/// Maps a newer layout `D3D12_BARRIER_LAYOUT` value to the legacy `D3D12_RESOURCE_STATES` bitmask
/// for `CreateCommittedResource` and `CreatePlacedResource`. An undefined or unknown layout is COMMON (0).
pub(crate) fn barrier_layout_to_state(layout: u32) -> u32 {
    LAYOUT_STATES.get(layout as usize).copied().unwrap_or(0)
}

// D3D12_HEAP_FLAG_ALLOW_DISPLAY: console display targets pass this to CreateCommittedResource3;
/// PC CreateCommittedResource rejects it without a swap-chain heap.
const HEAP_FLAG_ALLOW_DISPLAY: u32 = 0x8;

/// Shared backing for console placed-at-VA creation (slots 44 and 133): PC D3D12 has no placement
/// at a title-chosen GPU virtual address yet, so create a sanitized committed resource on a DEFAULT
/// heap and log the fallback.
#[allow(clippy::too_many_arguments)]
unsafe fn dev_placed_at_va_inner(
    this: *mut Proxy,
    slot: u32,
    va: u64,
    desc: *const c_void,
    state: u32,
    clear: *const c_void,
    riid: *const GUID,
    out: *mut *mut c_void,
) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    let _pc_call = crate::watchdog_hold::pc_call();
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable when non-null.
    unsafe { *out = std::ptr::null_mut() };
    if desc.is_null() {
        return E_INVALIDARG;
    }
    // D3D12_HEAP_PROPERTIES { Type = D3D12_HEAP_TYPE_DEFAULT (1), CPUPageProperty = 0,
    // MemoryPoolPreference = 0, CreationNodeMask = 0, VisibleNodeMask = 0 }.
    let default_props: [u32; 5] = [1, 0, 0, 0, 0];
    // SAFETY: `desc` is non-null and 56 readable bytes by the console contract.
    let raw: [u8; 0x38] = unsafe { std::ptr::read_unaligned(desc.cast()) };
    let mut sanitized = raw;
    pc_resource_desc(&mut sanitized);
    // SAFETY: the sanitized bytes are a plain D3D12_RESOURCE_DESC and `this` is a live device proxy.
    let (allocation_bytes, texture_kind) = unsafe {
        let d: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC = std::ptr::read_unaligned(sanitized.as_ptr().cast());
        let host = <windows::Win32::Graphics::Direct3D12::ID3D12Device as windows::core::Interface>::from_raw_borrowed(&(*this).pc);
        let bytes = host.map_or(0, |host| host.GetResourceAllocationInfo(0, &[d]).SizeInBytes);
        (bytes, mem_owners::committed_kind(d.Dimension.0, d.Flags.0))
    };
    // SAFETY: `desc` is non-null and `out` writable, by the console calling convention.
    let placed = unsafe { title_memory_buffer(va, desc, state, riid, out) };
    let mut hr = placed.unwrap_or_else(|| dev_create_committed_resource(this, default_props.as_ptr().cast(), 0, desc, state, clear, riid, out));
    if hr >= 0 {
        // SAFETY: `out` and `desc` were validated non-null above.
        let res = unsafe { *out };
        if !res.is_null() {
            let dim = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
            let width = u64::from_le_bytes([
                raw[0x10], raw[0x11], raw[0x12], raw[0x13], raw[0x14], raw[0x15], raw[0x16], raw[0x17],
            ]);
            let charge = placed.is_none().then_some(Charge {
                kind: texture_kind,
                segment: mem_owners::Segment::Local,
                bytes: allocation_bytes,
            });
            crate::views_at::record_resource(va, width, res as usize, dim == 1);
            if dim == 3 {
                // a texture in the console tiling: the PC texture is filled from title memory (detile)
                crate::detile::register(res as usize, va, &raw, state, allocation_bytes);
            }
            // a buffer is also known by the GPU address the PC gave it, which is what the title
            // gets from GetGPUVirtualAddress (list memory methods look addresses up by both)
            if dim == 1 {
                // SAFETY: `res` is the live PC resource just created.
                let pc_va = unsafe { method::<GpuVirtualAddress>(res, 11)(res) };
                if pc_va != 0 && pc_va != va {
                    crate::views_at::record_resource(pc_va, width, res as usize, true);
                }
            }
            // The title gets the resource as a console proxy; the PC object stays behind it, and
            // every relayed call that takes a resource unwraps it (see `resource_args`).
            // SAFETY: `res` is the PC object whose reference `dev_create_committed_resource` gave.
            hr = unsafe { give_placed(hr, res, va, charge, out) };
            if hr >= 0 {
                let segment = if placed.is_some() { mem_owners::Segment::System } else { mem_owners::Segment::Local };
                let bytes = if placed.is_some() { width } else { allocation_bytes };
                // SAFETY: `*out` is the fresh ResProxy made above.
                side_set(unsafe { (*out).cast::<Proxy>() }, |s| s.placed = Some((segment, bytes)));
                mem_owners::add(mem_owners::Kind::PlacedResource, segment, 1, bytes);
            }
        }
    }
    // A stand in holds none of the title's bytes and none of the GPU's writes reach the title: each of the
    // first ones is logged, not only every 1000th.
    static STAND_INS: AtomicI32 = AtomicI32::new(0);
    if log_due(&N) || (placed.is_none() && STAND_INS.fetch_add(1, Ordering::Relaxed) < 32) {
        xs_log!(
            "console device slot {slot} (CreatePlacedResourceX VA {va:016X}, state 0x{state:X}): backed by {} -> hr=0x{:08X}",
            if placed.is_some() { "a reserved buffer over its title memory" } else { "a committed PC resource standing in (the title's bytes are not in it)" },
            hr as u32
        );
    }
    if hr < 0 {
        // SAFETY: `desc` was checked non-null above and is 56 readable bytes by the console contract.
        let raw = unsafe { std::ptr::read_unaligned(desc.cast::<[u8; 0x38]>()) };
        // SAFETY: `clear` is null or 24 readable bytes by the console contract.
        let clear: Option<[u8; 24]> = (!clear.is_null()).then(|| unsafe { std::ptr::read_unaligned(clear.cast()) });
        xs_log!("console device slot {slot}: CreatePlacedResourceX VA {va:016X} failed with hr=0x{:08X}, state 0x{state:X}, desc {raw:02x?}, clear {clear:02x?}", hr as u32);
    }
    hr
}

/// A buffer the title places at `va` in title memory (`title_mem` on): a reserved PC buffer over the
/// pool tiles of its title pages (`title_mem::place_buffer`), handed out as `riid` into `*out`. `None`
/// for a texture, memory `title_mem` does not cover, or when it cannot be made: the committed way.
///
/// # Safety
/// `desc` is 56 readable bytes and `out` is writable.
unsafe fn title_memory_buffer(va: u64, desc: *const c_void, state: u32, riid: *const GUID, out: *mut *mut c_void) -> Option<HRESULT> {
    use windows::Win32::Graphics::Direct3D12::{D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES};
    use windows::core::Interface;
    if va == 0 || !crate::title_mem::enabled() {
        return None;
    }
    // SAFETY: by the contract above.
    let mut bytes: [u8; 0x38] = unsafe { std::ptr::read_unaligned(desc.cast()) };
    pc_resource_desc(&mut bytes);
    // SAFETY: `D3D12_RESOURCE_DESC` is 56 bytes of plain data in this layout.
    let pc_desc: D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast()) };
    let reserved = crate::title_mem::place_buffer(va, &pc_desc, D3D12_RESOURCE_STATES(state as i32))?;
    // SAFETY: `riid` is the caller's interface id, `out` writable; `query` adds the reference `*out` holds.
    Some(unsafe { reserved.query(riid.cast(), out) }.0)
}

// Console slot 44: CreatePlacedResourceX(va, desc, initial_state, clear, riid, out).
extern "system" fn dev_placed_at_va(this: *mut Proxy, va: u64, desc: *const c_void, state: u32, clear: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: `this` is a live device proxy and the remaining pointers follow the COM contract.
    unsafe { dev_placed_at_va_inner(this, 44, va, desc, state, clear, riid, out) }
}

/// Console slot 133: CreatePlacedResourceX(va, desc, initial_layout, clear, riid, out) of the
/// newer console device layout. Latches the newer layout and translates `D3D12_BARRIER_LAYOUT`
/// to `D3D12_RESOURCE_STATES`, then goes the way of slot 44.
extern "system" fn dev_placed_at_va_layout(this: *mut Proxy, va: u64, desc: *const c_void, layout: u32, clear: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if crate::abi::note_newer_device_method(133) {
        xs_log!("console device slot 133: the title's runtime has the newer console layout (D-2026-11-23-2)");
    }
    // SAFETY: `this` is a live device proxy and the remaining pointers follow the COM contract.
    unsafe { dev_placed_at_va_inner(this, 133, va, desc, barrier_layout_to_state(layout), clear, riid, out) }
}

/// Logs the XG and PC footprints of a display target and returns the byte size the title expects
/// the block of its display target to have: the XG one when XG knows the description, else the PC one.
///
/// # Safety
/// `pc` is a live PC device.
unsafe fn display_target_bytes(pc: Object, raw: &[u8; 0x38]) -> u64 {
    let mut pc_desc = *raw;
    pc_resource_desc(&mut pc_desc);
    let mut pc_answer = [0u64; 2];
    // SAFETY: by the contract above, `pc_answer` takes the two words of the result.
    unsafe { method::<AllocInfo>(pc, 25)(pc, pc_answer.as_mut_ptr().cast(), 0, 1, pc_desc.as_ptr().cast()) };
    // The title gives its display targets one block sized with XG and asserts it is used up
    // (issue #190): take the console XG footprint, not the PC one.
    let xg = crate::detile_plan::console_footprint(raw, crate::detile::hardware_v4());
    let (xg_size, xg_align) = xg.unwrap_or((0, 0));
    xs_log!(
        "console device slot 127: display target {}x{} format {} layout {:#x} flags {:#x}: XG {xg_size:#x} align {xg_align:#x}, PC {:#x} align {:#x}",
        u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap_or_default()),
        le_u32(raw, 0x18),
        le_u32(raw, 0x20),
        le_u32(raw, 0x2C),
        le_u32(raw, 0x30),
        pc_answer[0],
        pc_answer[1]
    );
    xg.map_or(pc_answer[0], |(size, _)| size)
}

/// Console slot 127: CreateCommittedResource3(props, heap_flags, desc, initial_layout, clear,
/// protected_session, num_castable, castable, riid, out) of the newer console device layout.
/// Latches the newer layout, strips `D3D12_HEAP_FLAG_ALLOW_DISPLAY`, translates
/// `D3D12_BARRIER_LAYOUT`, and forwards to slot 27. A display target also takes its block from the
/// title's allocation hook.
#[allow(clippy::too_many_arguments)]
extern "system" fn dev_create_committed_resource_layout(
    this: *mut Proxy,
    props: *const c_void,
    flags: u32,
    desc: *const c_void,
    layout: u32,
    clear: *const c_void,
    _protected_session: Object,
    _num_castable: u32,
    _castable: *const u32,
    riid: *const GUID,
    out: *mut *mut c_void,
) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if crate::abi::note_newer_device_method(127) {
        xs_log!("console device slot 127: the title's runtime has the newer console layout (D-2026-11-23-2)");
    }
    let state = barrier_layout_to_state(layout);
    let host_flags = flags & !HEAP_FLAG_ALLOW_DISPLAY;
    let hr = dev_create_committed_resource(this, props, host_flags, desc, state, clear, riid, out);
    if flags & HEAP_FLAG_ALLOW_DISPLAY != 0 && hr >= 0 && !desc.is_null() {
        // SAFETY: `this` is a live device proxy and `desc` is 56 readable bytes.
        unsafe {
            let raw: [u8; 0x38] = std::ptr::read_unaligned(desc.cast());
            let bytes = display_target_bytes((*this).pc, &raw);
            if bytes != 0 && bytes != u64::MAX {
                let at = crate::title_heap::display_target_alloc(bytes as usize);
                if at != 0 {
                    xs_log!("console device slot 127: display target allocated {bytes:#x} bytes from title hook at {at:016X}");
                    // the title's address stands for the PC resource (copies and views look it up)
                    if !out.is_null() && !(*out).is_null() {
                        crate::views_at::record_resource(at as u64, bytes, *out as usize, false);
                    }
                }
            }
        }
    }
    if log_due(&N) {
        xs_log!(
            "console device slot 127 (CreateCommittedResource3 flags 0x{flags:X} -> 0x{host_flags:X}, layout 0x{layout:X} -> state 0x{state:X}) -> hr=0x{:08X}",
            hr as u32
        );
    }
    if hr < 0 && !desc.is_null() {
        // SAFETY: `desc` is 56 readable bytes by the console contract.
        let raw = unsafe { std::ptr::read_unaligned(desc.cast::<[u8; 0x38]>()) };
        xs_log!("console device slot 127: CreateCommittedResource3 failed with hr=0x{:08X}, flags 0x{flags:X}, desc {raw:02x?}", hr as u32);
    }
    hr
}

// The queue flags the PC runtime knows (`D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT`). Console
/// bit 0x2 is unknown on the PC and ignored, the host rejects it with E_INVALIDARG.
const HOST_QUEUE_FLAGS: u32 = 0x1;

/// The console queue flags reduced to the ones the PC runtime accepts.
fn host_queue_flags(console_flags: u32) -> u32 {
    console_flags & HOST_QUEUE_FLAGS
}

/// Console slot 48: queue creation from a console description whose first 16 bytes are the PC
/// `D3D12_COMMAND_QUEUE_DESC`. The rest of the console description is not interpreted.
extern "system" fn dev_create_queue_x(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable, and a console description has at least the 16 bytes of the PC one.
    unsafe {
        *out = std::ptr::null_mut();
        if desc.is_null() {
            return E_INVALIDARG;
        }
        let pc_desc: [u32; 4] = std::ptr::read_unaligned(desc.cast());
        if log_due(&N) {
            xs_log!(
                "console device slot 48: queue type {}, priority {}, flags 0x{:X} (host 0x{:X}), node mask 0x{:X} (rest of the console desc ignored)",
                pc_desc[0],
                pc_desc[1] as i32,
                pc_desc[2],
                host_queue_flags(pc_desc[2]),
                pc_desc[3]
            );
        }
        // The title's memory is never written: the host gets a copy.
        let host_desc = host_queue_desc(pc_desc);
        dev_create_queue(this, host_desc.as_ptr().cast(), riid, out)
    }
}

/// A console queue description reduced to what the PC runtime accepts: the console DMA type
/// (0x10) is a copy queue (3), priorities other than normal, high and global realtime become high,
/// the flags keep only the PC ones, and the node mask is 0 or 1.
fn host_queue_desc(d: [u32; 4]) -> [u32; 4] {
    let kind = if d[0] == 0x10 { 3 } else { d[0] };
    let priority = if matches!(d[1], 0 | 100 | 10000) { d[1] } else { 100 };
    [kind, priority, host_queue_flags(d[2]), d[3].min(1)]
}

/// Console slot 55: a pipeline made from a base pipeline plus patches. Patch contents
/// are not known well enough to apply them: refused and logged.
extern "system" fn dev_derived_pso(_this: *mut Proxy, base: *mut c_void, count: u32, patches: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    decant_log::calls::not_implemented("console device slot 55 (derived pipeline)");
    if log_due(&N) {
        xs_log!("UNIMPLEMENTED console device slot 55 (derived pipeline from {}, {count} patches at {}): refused", p(base), p(patches));
    }
    if !out.is_null() {
        // SAFETY: `out` is writable when not null.
        unsafe { *out = std::ptr::null_mut() };
    }
    E_NOTIMPL
}

/// Console slots 60 and 62: (desc, extension count, extensions, riid, out). The description has the
/// PC layout. The extensions have no PC meaning and are logged, then ignored. `slots` is
/// the console slot and the PC slot it maps to.
///
/// # Safety
/// `this` is a live device proxy, the other arguments are the caller's.
unsafe fn dev_pso(this: *mut Proxy, (console_slot, pc_slot): (u32, usize), desc: *const c_void, count: u64, ext: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static FAIL_LOGGED: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
    let i = if console_slot == 60 { 0 } else { 1 };
    note_pso_extensions(console_slot, count, ext);
    // Compute only (console slot 62), matching the C layer's `dxil_override` call sites: the
    // graphics slot (60) is not looked up there either.
    let (overridden, mut stages) = if console_slot == 62 && !desc.is_null() {
        // SAFETY: slot 62's description has the PC `D3D12_COMPUTE_PIPELINE_STATE_DESC` layout
        // (abi.rs: "desc in PC layout"; measured in more than one retail title).
        unsafe { overridden_compute_desc(desc.cast()) }
    } else {
        (None, Vec::new())
    };
    // Graphics (slot 60): a copy with the state enums PC D3D12 knows (`stream_pso::valid_states`).
    let graphics = if console_slot == 60 && !desc.is_null() {
        // SAFETY: slot 60's description has the PC `D3D12_GRAPHICS_PIPELINE_STATE_DESC` layout.
        let mut g: Box<[u8; crate::stream_pso::GRAPHICS_DESC_SIZE]> = Box::new(unsafe { std::ptr::read_unaligned(desc.cast()) });
        crate::stream_pso::valid_states(&mut g);
        Some(g)
    } else {
        None
    };
    let mut graphics = graphics;
    let stage_count = graphics.as_ref().map(|g| graphics_stages(g));
    let _shaders_kept = graphics.as_mut().map(|g| {
        let (kept, graphics_stages) = override_graphics_shaders(g);
        stages = graphics_stages;
        kept
    });
    let full = stage_count.zip(_shaders_kept.as_ref()).map(|(n, k)| n == k.len());
    let effective_desc = match (&overridden, &graphics) {
        (Some((patched, _)), _) => patched as *const ComputePsoDesc as *const c_void,
        (None, Some(g)) => g.as_ptr().cast(),
        (None, None) => desc,
    };
    // SAFETY: by the contract above.
    let hr = unsafe {
        let pc = (*this).pc;
        method::<CreatePso>(pc, pc_slot)(pc, effective_desc, riid, out)
    };
    crate::rt_diag::pipeline(hr >= 0, full);
    if hr >= 0 && !out.is_null() {
        // SAFETY: the runtime wrote the new pipeline to `out`.
        crate::rt_diag::pipeline_made(unsafe { *out } as usize, || {
            let state = graphics.as_ref().map(|g| crate::pso_diag::describe_state(g)).unwrap_or_default();
            crate::pso_diag::describe_stages(&stages) + &state
        });
        if crate::kind_check::tracking() {
            // SAFETY: the runtime wrote the new pipeline to `out`.
            crate::kind_check::pipeline_made(unsafe { *out } as usize, graphics.is_some(), &crate::pso_diag::kind_stages(&stages));
        }
    }
    if hr < 0 && log_due(&FAIL_LOGGED[i]) {
        xs_log!("console device slot {console_slot}: the PC runtime refused the pipeline, hr=0x{:08X} (console shaders need a translator)", hr as u32);
    }
    if hr >= 0 {
        return hr;
    }
    let (kind, extra) = match &graphics {
        Some(g) => ("graphics, slot 60", graphics_targets(g)),
        None => ("compute, slot 62", String::new()),
    };
    // SAFETY: `out` is the caller's, null or writable.
    unsafe { crate::pso_diag::refused(hr, kind, &stages, &extra, out) }
}

/// Console device slot 10: CreateGraphicsPipelineState in PC form, with the state enums made valid
/// for PC as on slot 60.
extern "system" fn dev_graphics_pso_pc(this: *mut Proxy, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the arguments are the caller's, by the PC contract.
    unsafe { defer_graphics(this, desc, riid, out, graphics_pso_now) }
}

extern "system" fn dev_graphics_pso(this: *mut Proxy, desc: *const c_void, count: u64, ext: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    note_pso_extensions(60, count, ext);
    // SAFETY: the arguments are the caller's, by the console contract.
    unsafe { defer_graphics(this, desc, riid, out, graphics_pso_now) }
}

extern "system" fn dev_compute_pso(this: *mut Proxy, desc: *const c_void, count: u64, ext: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    note_pso_extensions(62, count, ext);
    // SAFETY: the arguments are the caller's, by the console contract.
    unsafe { defer_compute(this, desc, riid, out, compute_pso_now) }
}

/// Device slots 60 and 10 made now (the extensions were noted at creation).
///
/// # Safety
/// As for [`dev_pso`], `this` given as an address.
unsafe fn graphics_pso_now(this: usize, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe { dev_pso(this as *mut Proxy, (60, 10), desc, 0, std::ptr::null(), riid, out) }
}

/// Device slot 62 made now (the extensions were noted at creation).
///
/// # Safety
/// As for [`dev_pso`], `this` given as an address.
unsafe fn compute_pso_now(this: usize, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe { dev_pso(this as *mut Proxy, (62, 11), desc, 0, std::ptr::null(), riid, out) }
}

/// Logs (bounded) that a pipeline of console slot `console_slot` came with extensions, which have
/// no PC meaning and are ignored.
fn note_pso_extensions(console_slot: u32, count: u64, ext: *const c_void) {
    static EXT_LOGGED: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
    let i = if console_slot == 60 { 0 } else { 1 };
    if (count != 0 || !ext.is_null()) && log_due(&EXT_LOGGED[i]) {
        xs_log!("console device slot {console_slot}: {} pipeline extensions at {} ignored (meaning unknown)", count as u32, p(ext));
    }
}

/// Makes the view `pc_view` (a PC view description of 0x28 bytes) of no resource in the slot `cpu`
/// (a title heap CPU handle): a null view, which reads zeros. `false` when the slot has no host handle.
///
/// # Safety
/// `this` is a live device proxy, `pc_view` a view description of 0x28 bytes.
unsafe fn write_null_view(this: *mut Proxy, pc_view: &[u8; 0x28], va: u64, cpu: usize, uav: bool) -> bool {
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else { return false };
    // SAFETY: by the contract above.
    unsafe {
        let pc = (*this).pc;
        note_view(host_cpu, std::ptr::null_mut(), pc_view.as_ptr().cast(), va, if uav { "uav at" } else { "srv at" });
        crate::title_heap::note_cbv(host_cpu, None);
        if uav {
            method::<CreateUav>(pc, 19)(pc, std::ptr::null_mut(), std::ptr::null_mut(), pc_view.as_ptr().cast(), host_cpu);
        } else {
            method::<CreateSrv>(pc, 18)(pc, std::ptr::null_mut(), pc_view.as_ptr().cast(), host_cpu);
        }
        crate::view_shadow::view_made(host_cpu, [0; 4]);
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
    true
}

/// Where `FirstElement` lies in a PC buffer view description. `NumElements` follows at +8 and
/// `StructureByteStride` at +12 for both kinds of view, the flags lie at 0x20.
const fn first_element_at(uav: bool) -> usize {
    if uav { 0x08 } else { 0x10 }
}

/// The console slot of a view at an address: 71 for unordered access, 70 for shader resource.
const fn view_at_slot(uav: bool) -> u32 {
    if uav { 71 } else { 70 }
}

/// The console name of the view at an address creation call.
const fn view_at_name(uav: bool) -> &'static str {
    if uav { "CreateUavAt" } else { "CreateSrvAt" }
}

/// The resource dimension (`D3D12_RESOURCE_DIMENSION`) a view of dimension `dim`
/// (`D3D12_SRV_DIMENSION` or `D3D12_UAV_DIMENSION`) needs, if the dimension is known. A cube view
/// uses a 2D array resource, and an unknown view dimension keeps the usual validation path.
fn resource_dimension_of_view(uav: bool, dim: u32) -> Option<u32> {
    match (uav, dim) {
        (_, 1) | (false, 11) => Some(1),
        (_, 2 | 3) => Some(2),
        (_, 4..=7) | (false, 9 | 10) => Some(3),
        (_, 8) => Some(4),
        _ => None,
    }
}

/// Counts the views at an address that were skipped, for the log.
static VIEW_AT_SKIPS: AtomicI32 = AtomicI32::new(0);

/// Moves the buffer view `view` (a PC description whose format is already the PC one) `byte_off`
/// bytes into its buffer by advancing `FirstElement`. False, with the reason logged, when the offset
/// is no whole number of elements or the view would reach past what the PC runtime accepts.
fn shift_buffer_view(view: &mut [u8; 0x28], uav: bool, byte_off: u64, va: u64) -> bool {
    static FAR_LOGGED: AtomicI32 = AtomicI32::new(0);
    static FAR_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let first_at = first_element_at(uav);
    let stride = le_u32(view, first_at + 12);
    let flags = le_u32(view, 0x20);
    let element = match (flags & 1 != 0, stride) {
        (true, _) => 4,
        (false, 0) => crate::views_at::format_bytes(le_u32(view, 0)),
        (false, stride) => stride,
    };
    let slot = view_at_slot(uav);
    if element == 0 || !byte_off.is_multiple_of(u64::from(element)) {
        if log_due(&VIEW_AT_SKIPS) {
            xs_log!("console device slot {slot}: VA {va:016X} offset {byte_off:#x} not divisible by elem {element}; skipped");
        }
        return false;
    }
    let first = u64::from_le_bytes(view[first_at..first_at + 8].try_into().unwrap_or_default());
    let shifted = first.saturating_add(byte_off / u64::from(element));
    // The PC runtime takes no buffer view whose elements lie past 2^32 elements of its buffer: it
    // calls the view invalid and removes the device. That happened with a view 50 GiB into the window,
    // which the page mapping copies of its queue made reachable. Such a view is left out. The
    // elements are counted in the 4 byte ones of the raw view a structured one becomes (`view_raw`).
    let count = u64::from(le_u32(view, first_at + 8));
    if shifted.saturating_add(count).saturating_mul(u64::from(element)) / 4 > u64::from(u32::MAX) {
        let seen = FAR_VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
        if log_due(&FAR_LOGGED) || seen.is_power_of_two() {
            xs_log!(
                "console device slot {slot}: VA {va:016X} is element {shifted} (of {element} bytes) of the window buffer, past the 2^32 raw elements a PC buffer view reaches; left out ({seen} such views so far)"
            );
        }
        return false;
    }
    view[first_at..first_at + 8].copy_from_slice(&shifted.to_le_bytes());
    true
}

/// Console slots 70 and 71 (`CreateShaderResourceViewAt` and `CreateUnorderedAccessViewAt`) with
/// the arguments `(rdesc, view, cpu_dest)`, where `view` is a PC view description of 0x28 bytes
/// followed by the title GPU address at +0x28. The address is looked up in the table of placed
/// resources (a buffer view also in the window of `title_mem`, the title memory `xmem.dll` maps or
/// commits), a buffer view's `FirstElement` is moved by the offset into the resource, `cpu_dest`
/// is translated through `title_heap`, and the PC `CreateShaderResourceView` (18) or
/// `CreateUnorderedAccessView` (19) makes the view.
unsafe fn dev_create_view_at(this: *mut Proxy, rdesc: *const c_void, view: *const u8, cpu: usize, uav: bool) {
    static HITS: AtomicI32 = AtomicI32::new(0);
    if view.is_null() {
        return;
    }
    // SAFETY: a console view at an address is 0x30 bytes: the PC description of 0x28 bytes and the
    // GPU address as a u64 at +0x28.
    let console_view: [u8; 0x30] = unsafe { std::ptr::read_unaligned(view.cast()) };
    let va = u64::from_le_bytes(console_view[0x28..0x30].try_into().unwrap_or_default());
    let fmt = le_u32(&console_view, 0);
    let dim = le_u32(&console_view, 4);
    let is_buffer = dim == 1;
    let mut pc_view = [0u8; 0x28];
    pc_view.copy_from_slice(&console_view[..0x28]);
    let kind = if uav { crate::view_raw::ViewKind::UnorderedAccess } else { crate::view_raw::ViewKind::ShaderResource };
    let slot = view_at_slot(uav);
    let name = view_at_name(uav);
    // A buffer view of no elements is a null view wherever its address points (an address no
    // resource holds would otherwise leave the slot's old view in place).
    let mut probe = pc_view;
    if crate::view_raw::make_raw(&mut probe, kind) == crate::view_raw::Outcome::Empty {
        // SAFETY: `this` is a live device proxy, `probe` a null view description of 0x28 bytes.
        unsafe { write_null_view(this, &probe, va, cpu, uav) };
        return;
    }
    // A buffer view of title memory no placed resource holds is a view of the window of `title_mem`.
    let in_title_memory = || {
        let first_at = first_element_at(uav);
        let first = u64::from_le_bytes(pc_view[first_at..first_at + 8].try_into().unwrap_or_default());
        let count = le_u32(&pc_view, first_at + 8);
        let stride = le_u32(&pc_view, first_at + 12);
        let element = if stride != 0 { stride } else { crate::views_at::format_bytes(crate::abi::pc_dxgi_format(fmt)).max(4) };
        let span = crate::title_mem::translate(va, (first + u64::from(count)) * u64::from(element))?;
        Some((span.resource, span.offset))
    };
    // A texture view at an address no resource holds gets a texture of its own (`view_texture`).
    let texture = if !uav && !is_buffer && !rdesc.is_null() {
        // SAFETY: the console passes the 56 bytes of the texture's description with the view.
        unsafe { view_texture(this, va, rdesc) }
    } else {
        None
    };
    let found = texture
        .map(|res| (res, 0))
        .or_else(|| crate::views_at::lookup_resource(va, is_buffer))
        .or_else(|| is_buffer.then(in_title_memory).flatten());
    let Some((host_res, byte_off)) = found else {
        // On the console a view at address 0 is a null view: reads give zeros. Skipping it left the
        // slot's old (or never written) view in place, and a shader sampling it faulted at GPU address 0.
        // The same holds for a texture view at an address no resource holds (a 3D texture, a block
        // compressed one with VIEW_BC=0): a slot left at zero or at an old view is not a valid
        // descriptor either.
        if va == 0 || !is_buffer {
            static NULLS: AtomicI32 = AtomicI32::new(0);
            let mut null = pc_view;
            let mapped = crate::abi::pc_dxgi_format(fmt);
            null[0..4].copy_from_slice(&(if mapped == 0 { 28 } else { mapped }).to_le_bytes());
            // SAFETY: `this` is a live device proxy, `null` a view description of 0x28 bytes.
            if unsafe { write_null_view(this, &null, va, cpu, uav) } && log_due(&NULLS) {
                xs_log!("console device slot {slot} ({name}): a view of no resource at {va:016X} (dim={dim} fmt={fmt}) is made a null view, as on the console for address 0");
            }
            return;
        }
        if log_due(&VIEW_AT_SKIPS) {
            xs_log!("console device slot {slot} ({name}): VA {va:016X} (dim={dim} fmt={fmt}) not in placed-resource table nor in title memory; skipped");
        }
        return;
    };
    let Some(host_cpu) = crate::title_heap::translate_cpu(cpu) else {
        return;
    };
    pc_view[0..4].copy_from_slice(&crate::abi::pc_dxgi_format(fmt).to_le_bytes());
    if is_buffer && byte_off != 0 && !shift_buffer_view(&mut pc_view, uav, byte_off, va) {
        return;
    }
    if HITS.fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("console device slot {slot} ({name}): VA {va:016X} (+{byte_off:#x}) -> host res {host_res:016X} cpu {host_cpu:016X}");
    }
    // SAFETY: `this` is a live device proxy, `host_res` is a PC resource, `pc_view` is 0x28 bytes.
    unsafe {
        let pc = (*this).pc;
        let owner = host_res;
        let needs = if uav { ViewNeeds::UnorderedAccess } else { ViewNeeds::ShaderResource };
        let what = if uav { "unordered access view at an address" } else { "shader resource view at an address" };
        let mut host_res = host_res as Object;
        if let Some(wanted) = resource_dimension_of_view(uav, dim).filter(|_| !host_res.is_null()) {
            let actual = le_u32(&resource_desc(host_res), 0);
            if wanted != actual {
                static MISMATCHES: AtomicI32 = AtomicI32::new(0);
                if log_due(&MISMATCHES) {
                    xs_log!("console device: {what} of dimension {dim} on a resource of dimension {actual} is made a null view");
                }
                null_view_desc(Some(&mut pc_view), needs);
                host_res = std::ptr::null_mut();
            }
        }
        let host_res = view_resource(what, host_res, Some(&mut pc_view[..]), needs);
        let context = if uav { "console device slot 71 (CreateUavAt)" } else { "console device slot 70 (CreateSrvAt)" };
        let outcome = if host_res.is_null() { crate::view_raw::Outcome::NotStructured } else { crate::view_raw::convert(Some(&mut pc_view), kind, false, context) };
        let records = outcome.raw();
        let host_res = if outcome == crate::view_raw::Outcome::Empty { std::ptr::null_mut() } else { fitted(host_res, Some(&mut pc_view), kind) };
        note_view(host_cpu, host_res, pc_view.as_ptr().cast(), va, if uav { "uav at" } else { "srv at" });
        crate::title_heap::note_cbv(host_cpu, None);
        if uav {
            method::<CreateUav>(pc, 19)(pc, host_res, std::ptr::null_mut(), pc_view.as_ptr().cast(), host_cpu);
        } else {
            method::<CreateSrv>(pc, 18)(pc, host_res, pc_view.as_ptr().cast(), host_cpu);
        }
        crate::diagnose::removed_after(|| format!("the {} at {va:#x} on {host_res:p}, view {pc_view:02x?}", if uav { "UAV" } else { "SRV" }));
        crate::view_shadow::view_made(host_cpu, shadow_row(host_res, Some(&pc_view), kind, records));
        crate::view_shadow::owner_made(host_cpu, owner);
        crate::title_heap::tag_slot(cpu, host_cpu);
    }
}

/// `ID3D12Resource`.
const IID_RESOURCE: GUID = GUID::from_u128(0x696442be_a72e_4059_bc79_5b5c98040fad);

/// The PC texture for a shader resource view at `va` with the texture description `rdesc`, when no
/// placed resource starts there. On the console a view at an address needs no resource object, and
/// the title keeps small textures in its own memory that way (for example the gradient textures of a
/// loading screen UI, `R16G16B16A16_FLOAT`, in its arena pages). Their views were skipped, so the
/// table slot kept stale bytes and the UI shapes that sample them drew nothing over a black frame.
/// The texture is made once per address and description, registered at `va`, and `detile` fills it
/// from the title's bytes as for a placed texture. `None` for a placed resource at `va` (the caller
/// finds it), a description that is no 2D texture, or when it cannot be made.
///
/// # Safety
/// `this` is a live device proxy, `rdesc` 56 readable bytes.
unsafe fn view_texture(this: *mut Proxy, va: u64, rdesc: *const c_void) -> Option<usize> {
    static MADE: AtomicI32 = AtomicI32::new(0);
    static FULL: AtomicI32 = AtomicI32::new(0);
    static SAME: AtomicUsize = AtomicUsize::new(0);
    static CHANGED: AtomicUsize = AtomicUsize::new(0);
    // SAFETY: by the contract above.
    let raw: [u8; 0x38] = unsafe { std::ptr::read_unaligned(rdesc.cast()) };
    if u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) != 3 || va == 0 {
        return None;
    }
    // Block compressed textures are left out with `DECANT_D3D12X_VIEW_BC=0`. Loading artwork can be one.
    // With them the device was once removed during a world load in every measured run: a buffer view
    // at the address of such a texture took the texture (fixed in `VaTable::lookup`).
    if !crate::views_at::view_bc() && crate::views_at::block_compressed(crate::abi::pc_dxgi_format(u32::from_le_bytes(raw[0x20..0x24].try_into().unwrap_or([0; 4])))) {
        return None;

    }
    let old = match crate::views_at::with_view_textures(|t| t.lookup(va, &raw)) {
        // a resource the title placed there since wins
        crate::views_at::ViewTexture::Same(res) => {
            if crate::views_at::lookup_resource(va, false) != Some((res, 0)) {
                return None;
            }
            // a new view of the address: the title wrote the texture again
            crate::detile::mark_resource(res);
            crate::views_at::with_view_textures(|t| t.touch(va, crate::title_heap::presents(), view_clock_ms()));
            let n = SAME.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                xs_log!("console device slot 70: {n} views at an address reused the texture made for it, {} such textures made", crate::views_at::with_view_textures(|t| t.len()));
            }
            return Some(res);
        }
        crate::views_at::ViewTexture::Changed(res) => Some(res),
        crate::views_at::ViewTexture::New => {
            if crate::views_at::lookup_resource(va, false).is_some() {
                return None;
            }
            None
        }
    };
    if !crate::views_at::with_view_textures(|t| t.room()) {
        if log_due(&FULL) {
            xs_log!("console device slot 70: {} textures were made for views at an address, the view at {va:016X} is skipped", crate::views_at::MAX_VIEW_TEXTURES);
        }
        return None;
    }
    // in a heap of small textures (`view_heap`), else committed
    let mut d = raw;
    pc_resource_desc(&mut d);
    // read by shaders and written only by the detile copies, which leave it as they found it
    let state = crate::abi::pc_initial_state(&d, 0xC0);
    // SAFETY: `D3D12_RESOURCE_DESC` is 56 bytes of plain data in this layout.
    let pc_desc: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC = unsafe { std::ptr::read_unaligned(d.as_ptr().cast()) };
    // SAFETY: `this` is a live device proxy.
    let host_device = unsafe {
        <windows::Win32::Graphics::Direct3D12::ID3D12Device as windows::core::Interface>::from_raw_borrowed(&(*this).pc)
    };
    if let Some(host) = host_device {
        let _ = crate::views_at::view_texture_cap(host);
    }
    let device = crate::title_mem::device();
    let placed = device.as_ref().and_then(|device| crate::view_heap::place(device, &pc_desc, state));
    let (heap_index, mut bytes) = placed.as_ref().map_or((None, 0), |(_, h, b)| (Some(*h), *b));
    let mut out: *mut c_void = placed.map_or(std::ptr::null_mut(), |(r, _, _)| windows::core::Interface::into_raw(r));
    let hr = if out.is_null() {
        let default_props: [u32; 5] = [1, 0, 0, 0, 0];
        dev_create_committed_resource(this, default_props.as_ptr().cast(), 0, rdesc, state, std::ptr::null(), &IID_RESOURCE, &mut out)
    } else {
        S_OK
    };
    if bytes == 0 && hr >= 0 && !out.is_null() {
        // a committed texture: its size on the PC
        // SAFETY: one valid description.
        bytes = device.as_ref().map_or(0, |d| unsafe { d.GetResourceAllocationInfo(0, &[pc_desc]) }.SizeInBytes);
    }
    if hr < 0 || out.is_null() {
        if log_due(&MADE) {
            xs_log!("console device slot 70: no texture for the view at {va:016X}, hr=0x{:08X}, desc {raw:02x?}", hr as u32);
        }
        return None;
    }
    let res = out as usize;
    let committed = (heap_index.is_none()).then_some(Charge {
        kind: mem_owners::Kind::CommittedSampled,
        segment: mem_owners::Segment::Local,
        bytes,
    });
    if let Some(charge) = committed {
        mem_owners::add_charge(charge);
    }
    {
        let (height, format) = (u32::from_le_bytes(raw[0x18..0x1C].try_into().unwrap_or([0; 4])), u32::from_le_bytes(raw[0x20..0x24].try_into().unwrap_or([0; 4])));
        let width = u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap_or([0; 8]));
        // SAFETY: a live PC resource, named for DRED.
        if let Some(o) = unsafe { <windows::Win32::Graphics::Direct3D12::ID3D12Object as windows::core::Interface>::from_raw_borrowed(&out) } {
            let _ = unsafe { o.SetName(&windows::core::HSTRING::from(format!("decant view texture {width}x{height} format {format} at {va:#x}"))) };
        }
        crate::diagnose::removed_after(|| format!("the texture {width}x{height} format {format} for the view at {va:#x}"));
    }
    if old.is_some() {
        let n = CHANGED.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            xs_log!("console device slot 70: {n} views at an address came with another description than the texture made for it");
        }
    }
    if let Some(old) = old {
        // The old texture may still be read: it stays alive (`ViewTextures`), only unknown here.
        crate::detile::forget(old);
        crate::views_at::forget_resource(old);
    }
    crate::views_at::with_view_textures(|t| t.note(va, crate::views_at::ViewEntry::new(raw, res, heap_index, bytes, crate::title_heap::presents(), view_clock_ms())));
    mem_owners::add(
        mem_owners::Kind::ViewTexture,
        mem_owners::Segment::Local,
        1,
        bytes,
    );
    trim_view_textures();
    let width = u64::from_le_bytes(raw[0x10..0x18].try_into().unwrap_or([0; 8]));
    crate::views_at::record_resource(va, width, res, false);
    crate::detile::register(res, va, &raw, state, bytes);
    if log_due(&MADE) || crate::views_at::with_view_textures(|t| t.len()).is_power_of_two() {
        let height = u32::from_le_bytes(raw[0x18..0x1C].try_into().unwrap_or([0; 4]));
        let format = u32::from_le_bytes(raw[0x20..0x24].try_into().unwrap_or([0; 4]));
        xs_log!(
            "console device slot 70: a texture {width}x{height} format {format} made for the view at {va:016X}, filled from title memory ({} such textures, {} heaps with {} MiB placed)",
            crate::views_at::with_view_textures(|t| t.len()),
            crate::view_heap::usage().0,
            crate::view_heap::usage().1 >> 20
        );
    }
    Some(res)
}

/// Milliseconds since the first call (a monotonic clock for the age of the view textures).
fn view_clock_ms() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// Gives back the view textures that no view of their address asked for lately when the cache is past
/// its memory cap (`views_at::view_texture_cap_bytes`, `DECANT_D3D12X_VIEW_TEX_MIB`). Each goes through
/// `retire`, so the GPU finishes the submissions that may read it first; its heap is used again once
/// every texture placed in it is gone (`view_heap`). A line says how much went.
fn trim_view_textures() {
    static WAVES: AtomicI32 = AtomicI32::new(0);
    static STUCK: std::sync::Once = std::sync::Once::new();
    static LAST_SCAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(device) = crate::title_mem::device() else { return };
    let (cap, dedicated, budget) = crate::views_at::view_texture_cap(&device);
    static CAP_LOGGED: std::sync::Once = std::sync::Once::new();
    CAP_LOGGED.call_once(|| {
        xs_log!("console device slot 70: the view textures may take {} MiB of heaps ({} MiB dedicated video memory, budget {} MiB, DECANT_D3D12X_VIEW_TEX_MIB {}); past it the least recently used heap whose textures are all idle and held by no slot goes whole", cap >> 20, dedicated >> 20, budget >> 20, std::env::var("DECANT_D3D12X_VIEW_TEX_MIB").unwrap_or_else(|_| "not set".into()));
    });
    let (present, ms) = (crate::title_heap::presents(), view_clock_ms());
    if cap == 0 || crate::views_at::with_view_textures(|t| t.held_bytes(crate::view_heap::HEAP_BYTES)) <= cap {
        return;
    }
    // the scan of every slot of every view heap is not made at every texture
    if present < LAST_SCAN.load(Ordering::Relaxed) + 30 {
        return;
    }
    LAST_SCAN.store(present, Ordering::Relaxed);
    let referenced = crate::view_shadow::referenced();
    let pass = crate::views_at::with_view_textures(|t| t.collect(cap, crate::view_heap::HEAP_BYTES, present, ms, &referenced));
    if pass.stuck {
        STUCK.call_once(|| {
            xs_log!("view texture cache over its cap, nothing idle: {} MiB held for a cap of {} MiB, {} textures are held by a slot", pass.held_before >> 20, cap >> 20, referenced.len());
        });
    }
    if pass.freed.is_empty() {
        return;
    }
    mem_owners::note_view_heaps_evicted(pass.heaps_freed as u64);
    let mut bytes = 0u64;
    for e in &pass.freed {
        crate::detile::forget(e.res);
        crate::views_at::forget_resource(e.res);
        mem_owners::remove(
            mem_owners::Kind::ViewTexture,
            mem_owners::Segment::Local,
            1,
            e.bytes,
        );
        // SAFETY: the layer's own reference of a texture it made, held by no table of the layer and by no slot of a view heap.
        if e.heap.is_none() {
            unsafe {
                crate::retire::release_owned(
                    e.res as Object,
                    "view texture",
                    Charge {
                        kind: mem_owners::Kind::CommittedSampled,
                        segment: mem_owners::Segment::Local,
                        bytes: e.bytes,
                    },
                )
            };
        } else {
            // SAFETY: as above; the texture's bytes remain in its permanent view heap.
            unsafe { crate::retire::release(e.res as Object, "view texture") };
        }
        if let Some(h) = e.heap {
            crate::view_heap::released(h);
        }
        bytes += e.bytes;
    }
    if log_due(&WAVES) {
        let left = crate::views_at::with_view_textures(|t| t.len());
        xs_log!(
            "console device slot 70: {} textures ({} MiB) in {} heaps freed, {} heaps in use, {} MiB in heaps for a cap of {} MiB ({left} made in all, {} heaps with {} MiB placed)",
            pass.freed.len(),
            bytes >> 20,
            pass.heaps_freed,
            pass.heaps_in_use,
            pass.held_after >> 20,
            cap >> 20,
            crate::view_heap::usage().0,
            crate::view_heap::usage().1 >> 20
        );
    }
}

extern "system" fn dev_create_srv_at(this: *mut Proxy, rdesc: *const c_void, view: *const u8, cpu: usize) {
    // SAFETY: arguments are the caller's per console slot 70.
    unsafe { dev_create_view_at(this, rdesc, view, cpu, false) }
}

extern "system" fn dev_create_uav_at(this: *mut Proxy, rdesc: *const c_void, view: *const u8, cpu: usize) {
    // SAFETY: arguments are the caller's per console slot 71.
    unsafe { dev_create_view_at(this, rdesc, view, cpu, true) }
}

/// Console slot 74 (`RegisterPagePoolX(va, pages, index_out)`): registers the pool in the page pool
/// table and hands back its index.
extern "system" fn dev_register_page_pool(_this: *mut Proxy, va: u64, pages: u32, out: *mut u32) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if out.is_null() {
        return E_POINTER;
    }
    let idx = crate::views_at::register_page_pool(va, pages);
    // SAFETY: `out` is a writable `u32` pointer from the caller.
    unsafe { std::ptr::write_unaligned(out, idx) };
    if log_due(&N) {
        xs_log!("console device slot 74 (RegisterPagePoolX): va={va:#x} pages={pages} -> idx={idx}");
    }
    S_OK
}

/// Console slot 75 (`UnregisterPagePoolX(va, pages)`): drops the pool from the page pool table.
extern "system" fn dev_unregister_page_pool(_this: *mut Proxy, va: u64, pages: u32) -> HRESULT {
    let _ = crate::views_at::unregister_page_pool(va, pages);
    S_OK
}

/// Console slot 73: command list creation from a parameter block {type (u32) at 0, node mask (u32)
/// at 8}. Like the PC call, the list comes back open.
extern "system" fn dev_create_list_x(this: *mut Proxy, params: *const c_void, alloc: Object, pso: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable, and the parameter block holds the two words.
    unsafe {
        *out = std::ptr::null_mut();
        if params.is_null() {
            return E_INVALIDARG;
        }
        let kind: u32 = std::ptr::read_unaligned(params.cast());
        let mask: u32 = std::ptr::read_unaligned(params.cast::<u8>().add(8).cast());
        create_list(this, mask, kind, alloc, pso, riid, out)
    }
}

/// The title's clear value for a resource whose PC description is `desc`, as the PC takes it
/// (`abi::pc_clear_value`): `None` for no clear value.
///
/// # Safety
/// `clear` is null or 24 readable bytes, by the console calling convention.
unsafe fn pc_clear(desc: &[u8; 0x38], clear: *const c_void) -> Option<[u8; 24]> {
    if clear.is_null() {
        return None;
    }
    // SAFETY: by the contract above.
    let raw: [u8; 24] = unsafe { std::ptr::read_unaligned(clear.cast()) };
    crate::abi::pc_clear_value(desc, &raw)
}

/// Console slot 27: CreateCommittedResource, the console resource description sanitized, its
/// initial state and clear value made ones the PC takes for that description.
extern "system" fn dev_create_committed_resource(this: *mut Proxy, props: *const c_void, flags: u32, desc: *const c_void, state: u32, clear: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy, `desc` is the caller's (null or 56 readable bytes, by the
    // console calling convention), `out` is writable.
    unsafe {
        let pc = (*this).pc;
        let mut r: Object = std::ptr::null_mut();
        let (hr, actual_state) = if desc.is_null() {
            (method::<CreateCommittedResource>(pc, 27)(pc, props, flags, desc, state, clear, riid, &mut r), state)
        } else {
            let mut d: [u8; 0x38] = std::ptr::read_unaligned(desc.cast());
            pc_resource_desc(&mut d);
            let state = crate::abi::pc_initial_state(&d, state);
            let clear = pc_clear(&d, clear);
            (method::<CreateCommittedResource>(pc, 27)(pc, props, flags, d.as_ptr().cast(), state, clear.as_ref().map_or(std::ptr::null(), |c| c.as_ptr().cast()), riid, &mut r), state)
        };
        if hr >= 0 {
            if !r.is_null() {
                crate::present::note_resource_created(r as usize, actual_state);
            }
            *out = r;
        } else {
            *out = std::ptr::null_mut();
        }
        hr
    }
}

/// Console slot 29: CreatePlacedResource, the heap argument unwrapped and the same resource
/// description sanitizing as slot 27.
extern "system" fn dev_create_placed_resource(this: *mut Proxy, heap: Object, offset: u64, desc: *const c_void, state: u32, clear: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: as in `dev_create_committed_resource`, and `heap` is a proxy or a PC heap.
    unsafe {
        let pc = (*this).pc;
        let heap = unwrap(heap);
        let mut r: Object = std::ptr::null_mut();
        let (hr, actual_state) = if desc.is_null() {
            (method::<CreatePlacedResource>(pc, 29)(pc, heap, offset, desc, state, clear, riid, &mut r), state)
        } else {
            let mut d: [u8; 0x38] = std::ptr::read_unaligned(desc.cast());
            pc_resource_desc(&mut d);
            let state = crate::abi::pc_initial_state(&d, state);
            let clear = pc_clear(&d, clear);
            (method::<CreatePlacedResource>(pc, 29)(pc, heap, offset, d.as_ptr().cast(), state, clear.as_ref().map_or(std::ptr::null(), |c| c.as_ptr().cast()), riid, &mut r), state)
        };
        if hr >= 0 {
            if !r.is_null() {
                crate::present::note_resource_created(r as usize, actual_state);
            }
            *out = r;
        } else {
            *out = std::ptr::null_mut();
        }
        hr
    }
}

/// The `D3D12_INDIRECT_ARGUMENT_TYPE` of the console only `DispatchX` argument, which has no PC
/// equivalent.
const INDIRECT_ARGUMENT_DISPATCH_X: u32 = 105;

/// The PC `D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH` that a `DispatchX` argument is made.
const PC_INDIRECT_ARGUMENT_DISPATCH: u32 = 2;

/// Console slot 41: CreateCommandSignature. The console only `DispatchX` indirect argument type
/// (105) is rewritten in the title's description to the PC `Dispatch` type (2), then the PC call
/// is made with the root signature unwrapped.
extern "system" fn dev_create_command_signature(this: *mut Proxy, desc: *const u8, root_sig: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live proxy; `desc` is {NumArgumentDescs: u32, ArgumentDescs: *const
    // ArgDesc, ByteStride: u32} by the console calling convention, each ArgDesc at least 4 bytes.
    unsafe {
        let pc = (*this).pc;
        let count = std::ptr::read_unaligned::<u32>(desc.cast()) as usize;
        let first: *mut u8 = std::ptr::read_unaligned(desc.add(8).cast());
        let step = std::ptr::read_unaligned::<u32>(desc.add(16).cast()) as usize;
        for index in 0..count {
            let type_field = first.add(index * step).cast::<u32>();
            if std::ptr::read_unaligned(type_field) == INDIRECT_ARGUMENT_DISPATCH_X {
                std::ptr::write_unaligned(type_field, PC_INDIRECT_ARGUMENT_DISPATCH);
            }
        }
        let mut made: Object = std::ptr::null_mut();
        let hr = method::<CreateCommandSignature>(pc, 41)(pc, desc.cast(), unwrap(root_sig), riid, &mut made);
        *out = if hr >= 0 { made } else { std::ptr::null_mut() };
        hr
    }
}

/// Console slot 61: CreateSamplerX with a 0x6C byte description that only knows anisotropic
/// filtering (`MaxAnisotropy` at offset 0, `BorderColor` at offset 4, the destination CPU
/// descriptor handle as the last argument). It is relayed as a PC `D3D12_SAMPLER_DESC` with an
/// anisotropic filter.
extern "system" fn dev_create_sampler_x(this: *mut Proxy, desc: *const u8, dest: u64) -> HRESULT {
    const FILTER_ANISOTROPIC: u32 = 0x55;
    const ADDRESS_WRAP: u32 = 1;
    const COMPARISON_NEVER: u32 = 1;
    // SAFETY: `this` is a live proxy; `desc` is at least 8 bytes (MaxAnisotropy, BorderColor
    // index) by the console calling convention.
    unsafe {
        let pc = (*this).pc;
        // The destination is a CPU handle in title memory, like the other view slots (native
        // D3D12 removes the device on the untranslated handle).
        let Some(dest) = crate::title_heap::translate_cpu(dest as usize) else { return S_OK };
        let max_anisotropy: u32 = std::ptr::read_unaligned(desc.cast());
        // D3D12_SAMPLER_DESC, 0x34 bytes: Filter 0, AddressU/V/W 4/8/0xC, MipLODBias 0x10,
        // MaxAnisotropy 0x14, ComparisonFunc 0x18, BorderColor 0x1C..0x2C, MinLOD 0x2C, MaxLOD 0x30.
        let mut pc_desc = [0u8; 0x34];
        let mut put = |at: usize, word: [u8; 4]| pc_desc[at..at + 4].copy_from_slice(&word);
        put(0, FILTER_ANISOTROPIC.to_le_bytes());
        // zero is not a valid address mode on PC
        for at in [4, 8, 0xC] {
            put(at, ADDRESS_WRAP.to_le_bytes());
        }
        put(0x14, max_anisotropy.clamp(1, 16).to_le_bytes());
        put(0x18, COMPARISON_NEVER.to_le_bytes());
        put(0x30, f32::MAX.to_le_bytes());
        method::<CreateSampler>(pc, 22)(pc, pc_desc.as_ptr().cast(), dest);
        S_OK
    }
}

/// Console slot 108 (newer layout, D-2026-11-23-2): GetResourceAllocationInfo2. The result pointer
/// and `this` come in either order as for slot 25, and every descriptor is packed after the one
/// before it, the offset, alignment and size of each going to `infos`. A console descriptor is a
/// `D3D12_RESOURCE_DESC1` (0x48 bytes), of which the first 56 are read as the
/// `D3D12_RESOURCE_DESC` part. The trailing sampler feedback mip region is not modelled.
extern "system" fn dev_alloc_info2(a: *mut c_void, b: *mut c_void, mask: u32, n: u32, descs: *const u8, infos: *mut [u64; 3]) -> *mut c_void {
    if crate::abi::note_newer_device_method(108) {
        xs_log!("console device slot 108 (GetResourceAllocationInfo2): the title's runtime has the newer console layout (D-2026-11-23-2)");
    }
    let (this, ret) = sret_order(a, b);
    let (mut total, mut align) = (0u64, 0u64);
    let mut overflowed = false;
    // SAFETY: `this` is a live device proxy; `descs` has `n` entries of at least 0x38 bytes each,
    // and `infos` (when not null) is writable for `n` entries of 24 bytes each; by the console
    // calling convention.
    unsafe {
        let pc = (*this).pc;
        for k in 0..n as usize {
            let mut info = [0u64; 3];
            if !overflowed {
                let mut console_desc = [0u8; 0x38];
                if !descs.is_null() {
                    std::ptr::copy_nonoverlapping(descs.add(k * 0x48), console_desc.as_mut_ptr(), 0x38);
                }
                let (size, alignment) = allocation_of(pc, 108, mask, &console_desc);
                match crate::abi::accumulate_alloc_info(total, align, size, alignment) {
                    Some((offset, new_total, new_align)) => {
                        info = [offset, alignment, size];
                        (total, align) = (new_total, new_align);
                    }
                    None => {
                        overflowed = true;
                        total = u64::MAX;
                    }
                }
            }
            if !infos.is_null() {
                *infos.add(k) = info;
            }
        }
        if !ret.is_null() {
            *(ret as *mut [u64; 2]) = [total, align];
        }
    }
    ret
}

// ---- frame pacing (device 78 to 80), the clock is in `frame` ----

/// Console slot 86 (`CreatePipelineState(desc, riid, out)` from `D3D12_PIPELINE_STATE_STREAM_DESC`).
/// A stream with a graphics or compute pipeline is unpacked into a plain PC description, so root
/// signatures are unwrapped, compute shaders go through `dxil_override`, and console depth
/// stencil formats (`117..=120`) are mapped to the PC ones (`44..=47`). A stream with a mesh
/// shader becomes our mesh description, made on first bind (`mesh_pipeline`).
extern "system" fn dev_create_pipeline_state_stream(
    this: *mut Proxy,
    desc: *const c_void,
    riid: *const GUID,
    out: *mut *mut c_void,
) -> HRESULT {
    static LOGGED: AtomicI32 = AtomicI32::new(0);
    static FAIL_LOGGED: AtomicI32 = AtomicI32::new(0);
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable when non-null.
    unsafe { *out = std::ptr::null_mut() };
    if desc.is_null() {
        return E_INVALIDARG;
    }
    // D3D12_PIPELINE_STATE_STREAM_DESC { SIZE_T SizeInBytes; void *pPipelineStateSubobjectStream; }
    // SAFETY: caller supplies a readable 16 byte stream descriptor.
    let [size_in_bytes, stream_address]: [usize; 2] = unsafe { std::ptr::read_unaligned(desc.cast()) };
    let stream_ptr = stream_address as *const u8;
    if size_in_bytes == 0 || stream_ptr.is_null() {
        return E_INVALIDARG;
    }
    let note_unpacked = |what: &str| {
        if log_due(&LOGGED) {
            xs_log!("console device slot 86 (CreatePipelineState stream, {size_in_bytes} B): unpacked -> {what}");
        }
    };
    // SAFETY: caller guarantees `stream_ptr` points to `size_in_bytes` bytes of subobject stream.
    let bytes = unsafe { std::slice::from_raw_parts(stream_ptr, size_in_bytes) };
    match crate::stream_pso::unpack_stream(bytes) {
        Ok(Some(crate::stream_pso::UnpackedStream::Compute(mut compute))) => {
            // SAFETY: the root signature is a title object or a PC one.
            compute.root_sig = unsafe { unwrap(compute.root_sig as Object) } as usize;
            note_unpacked("compute PSO");
            // `PlainComputeDesc` is `#[repr(C)]` with the exact x86_64 layout of `ComputePsoDesc`.
            dev_create_compute_pipeline_state(this, (&raw const compute).cast::<ComputePsoDesc>(), riid, out)
        }
        Ok(Some(crate::stream_pso::UnpackedStream::Graphics(mut graphics))) => {
            let signature = crate::stream_pso::graphics_root_sig(&graphics);
            // SAFETY: the root signature is a title object or a PC one.
            crate::stream_pso::set_graphics_root_sig(&mut graphics, unsafe { unwrap(signature as Object) } as usize);
            note_unpacked("graphics PSO");
            // SAFETY: `this` is a live device proxy and `graphics` is a valid 656 byte D3D12_GRAPHICS_PIPELINE_STATE_DESC.
            unsafe { defer_graphics(this, graphics.as_ptr().cast(), riid, out, stream_graphics_now) }
        }
        Ok(Some(crate::stream_pso::UnpackedStream::Mesh(mut mesh))) => {
            let graphics_bytes = crate::stream_pso::GRAPHICS_DESC_SIZE;
            let mut graphics: [u8; crate::stream_pso::GRAPHICS_DESC_SIZE] = mesh[..graphics_bytes].try_into().expect("the graphics part");
            let signature = crate::stream_pso::graphics_root_sig(&graphics);
            // SAFETY: the root signature is a title object or a PC one.
            crate::stream_pso::set_graphics_root_sig(&mut graphics, unsafe { unwrap(signature as Object) } as usize);
            mesh[..graphics_bytes].copy_from_slice(&graphics);
            note_unpacked("mesh PSO");
            // SAFETY: `this` is a live device proxy, the shaders of `mesh` point into the title's stream.
            unsafe { mesh_pipeline::defer_mesh(this, &mesh, riid, out) }
        }
        Ok(None) => {
            decant_log::calls::not_implemented("console device slot 86 (stream without a CS, VS or MS)");
            if log_due(&FAIL_LOGGED) {
                xs_log!("console device slot 86 (stream, {size_in_bytes} B): no compute, vertex or mesh shader -> E_NOTIMPL");
            }
            E_NOTIMPL
        }
        Err(bad_ty) => {
            decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("console device slot 86 (subobject type {bad_ty})"));
            if log_due(&FAIL_LOGGED) {
                xs_log!("console device slot 86 (stream, {size_in_bytes} B): unknown subobject type {bad_ty} -> E_NOTIMPL");
            }
            E_NOTIMPL
        }
    }
}

// A graphics pipeline unpacked from a slot 86 stream, made now: the DXIL overrides of its stages
/// and the PC creation (slot 10).
///
/// # Safety
/// `this` is a live device proxy (as an address), `desc` a readable 656 byte
/// `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, `out` writable.
unsafe fn stream_graphics_now(this: usize, desc: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    static FAIL_LOGGED: AtomicI32 = AtomicI32::new(0);
    let this = this as *mut Proxy;
    if desc.is_null() || out.is_null() {
        return E_POINTER;
    }
    // SAFETY: by the contract above.
    let mut g: Box<[u8; crate::stream_pso::GRAPHICS_DESC_SIZE]> = Box::new(unsafe { std::ptr::read_unaligned(desc.cast()) });
    let stage_count = graphics_stages(&g);
    let (_shaders_kept, stages) = override_graphics_shaders(&mut g);
    let full = Some(stage_count == _shaders_kept.len());
    // SAFETY: `this` is a live device proxy and `g` is a valid 656-byte D3D12_GRAPHICS_PIPELINE_STATE_DESC.
    let hr = unsafe {
        let pc = (*this).pc;
        method::<CreatePso>(pc, 10)(pc, g.as_ptr().cast(), riid, out)
    };
    crate::rt_diag::pipeline(hr >= 0, full);
    if hr >= 0 {
        // SAFETY: `out` was checked above, the runtime wrote the new pipeline to it.
        crate::rt_diag::pipeline_made(unsafe { *out } as usize, || crate::pso_diag::describe_stages(&stages) + &crate::pso_diag::describe_state(&g));
        if crate::kind_check::tracking() {
            // SAFETY: as above.
            crate::kind_check::pipeline_made(unsafe { *out } as usize, true, &crate::pso_diag::kind_stages(&stages));
        }
    }
    if hr < 0 && log_due(&FAIL_LOGGED) {
        xs_log!(
            "console device slot 86 (stream -> graphics PSO): PC runtime refused hr=0x{:08X} (console VS/PS need translation)",
            hr as u32
        );
    }
    if hr >= 0 {
        return hr;
    }
    // SAFETY: `out` was checked above and is writable.
    unsafe { crate::pso_diag::refused(hr, "graphics, slot 86", &stages, &graphics_targets(&g), out) }
}

/// Console slot 78: `SetFrameIntervalX(output, interval in microseconds, periodicity, flags)`.
extern "system" fn dev_set_frame_interval(_this: *mut Proxy, output: *mut c_void, interval_us: u32, periodicity: u32, flags: u32) -> HRESULT {
    frame::set_interval(output as usize, interval_us, periodicity, flags)
}

/// Console slot 79: `ScheduleFrameEventX(type, offset in microseconds, auxiliary, flags)`.
extern "system" fn dev_schedule_frame_event(_this: *mut Proxy, kind: u32, offset_us: u32, aux: *const c_void, flags: u32) -> HRESULT {
    frame::schedule_event(kind, offset_us, aux as usize, flags)
}

/// Console slot 80: `WaitFrameEventX(type, timeout in milliseconds, auxiliary, flags, token*)`.
/// `token` is in/out: the caller's current value (0 for none) goes in, frame.rs's result token
/// (when it issues one) overwrites it.
extern "system" fn dev_wait_frame_event(_this: *mut Proxy, kind: u32, timeout_ms: u32, aux: *const c_void, flags: u32, token: *mut u64) -> HRESULT {
    // SAFETY: `token` is a title-supplied in/out pointer, readable/writable when not null.
    let token_in = if token.is_null() { 0 } else { unsafe { *token } };
    let (hr, value) = frame::wait_event(kind, timeout_ms, aux as usize, flags, token_in);
    if let Some(value) = value
        && !token.is_null()
    {
        // SAFETY: `token` is writable when not null.
        unsafe { *token = value };
    }
    hr
}

/// Console device slot 81: `GetFrameStatisticsX(token, type, count in/out, statistics)`, see
/// `frame::statistics`.
extern "system" fn dev_frame_statistics(_this: *mut Proxy, token: u64, kind: u32, count: *mut u32, out: *mut u8) -> HRESULT {
    frame::statistics(token, kind, count, out)
}

// ---- command list ----

/// The console primitive topology `D3D_PRIMITIVE_TOPOLOGY_RECTLIST`, which PC D3D12 does not accept.
const TOPOLOGY_RECT_LIST: u32 = 0x11;

/// `D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST`, which stands in for the rectangle list.
const TOPOLOGY_TRIANGLE_LIST: u32 = 4;

/// Command list slot 20 (`IASetPrimitiveTopology`). The rectangle list is no PC topology: it is
/// set as a triangle list, and `rect_list` is told so that the draws that follow get their fourth
/// corners.
extern "system" fn list_ia_set_primitive_topology(this: *mut Proxy, topology: u32) {
    type SetTopology = unsafe extern "system" fn(Object, u32);
    static MAPPED: AtomicI32 = AtomicI32::new(0);
    let is_rect_list = topology == TOPOLOGY_RECT_LIST;
    crate::rect_list::set_rect(this as usize, is_rect_list);
    if is_rect_list && MAPPED.fetch_add(1, Ordering::Relaxed) < 4 {
        xs_log!("list slot 20 (IASetPrimitiveTopology): mapped console RECT_LIST (0x11) -> TRIANGLELIST (4), draws from vertex buffer 0 get the fourth corners");
    }
    let pc_topology = if is_rect_list { TOPOLOGY_TRIANGLE_LIST } else { topology };
    // SAFETY: `this` is a live command list proxy; slot 20 is IASetPrimitiveTopology on PC.
    unsafe {
        let pc = (*this).pc;
        method::<SetTopology>(pc, 20)(pc, pc_topology);
    }
}

// Console slot 27: ExecuteBundle, the bundle is a proxied list.
extern "system" fn list_execute_bundle(this: *mut Proxy, bundle: Object) {
    // SAFETY: `this` is a live proxy, `bundle` is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        method::<Bundle>(pc, 27)(pc, unwrap(bundle));
    }
}

/// Console slot 32: SetDescriptorHeaps (PC 28), the heaps are proxied.
extern "system" fn list_set_heaps(this: *mut Proxy, n: u32, heaps: *const Object) {
    // SAFETY: `this` is a live proxy, `heaps` points to `n` objects.
    unsafe {
        let mapped = unwrap_array(n, heaps);
        let pc = (*this).pc;
        list_mirror::note_heaps(this, &mapped);
        let mapped = immediate_samplers::with_sampler_heap(mapped);
        method::<SetHeaps>(pc, 28)(pc, mapped.len() as u32, mapped.as_ptr());
    }
}

/// Console slot 67: texture cache flush. The PC runtime keeps its caches coherent
/// through barriers, so there is nothing to do.
extern "system" fn list_flush_texture_cache(this: *mut Proxy) {
    static N: AtomicI32 = AtomicI32::new(0);
    if log_due(&N) {
        xs_log!("console list slot 67 (texture cache flush) on {}: nothing to do on PC", p(this.cast()));
    }
}

/// Console slot 83: CopyTextureRegion with an 8th argument, always 0 in the measured titles.
extern "system" fn list_copy_texture_x(this: *mut Proxy, dst: *const c_void, x: u32, y: u32, z: u32, src: *const c_void, boxed: *const c_void, extra: u64) {
    static N: AtomicI32 = AtomicI32::new(0);
    if extra != 0 && log_due(&N) {
        xs_log!("console list slot 83 (CopyTextureRegion): 8th argument {:016X} not zero (meaning unknown, ignored)", extra);
    }
    // SAFETY: `this` is a live proxy; the copy locations go to PC as copies with their resources unwrapped.
    unsafe {
        let pc = (*this).pc;
        let (dst_loc, src_loc) = (unwrapped_location(dst), unwrapped_location(src));
        note_copy_dst(&dst_loc);
        method::<CopyTexture>(pc, 16)(pc, location_ptr(&dst_loc, dst), x, y, z, location_ptr(&src_loc, src), boxed);
    }
}

/// What the unknown-slot stub does for console `slot` of `kind`, for the newer-layout handlers
/// below when the newer layout is not in effect: the stub's log line and count, and its
/// `E_NOTIMPL`, which a caller of a void method ignores.
fn unknown_slot(this: Object, kind: Kind, slot: u32) {
    let _ = unknown_method(this, (kind as u32) << 16 | slot);
}

/// Flag bits of `Write64BitValueBottomOfPipeX` that ask for a GPU timestamp instead of a value.
const BOTTOM_TIMESTAMPS: u32 = 0x3;

/// Whether a diagnostic line is due for the first time of `flag`.
fn first_time(flag: &AtomicBool) -> bool {
    !flag.swap(true, Ordering::SeqCst)
}

/// Console list slot 71 (newer layout): `Write32BitValueBottomOfPipeX(address, value, flush,
/// flags)`. The address is title memory (seen: a small offset into a page the title mapped, with
/// the value 0). The write is recorded for the CPU to do once the list has run.
extern "system" fn list_write32_bottom(this: *mut Proxy, va: u64, value: u32, _flush: u32, _flags: u32) {
    if crate::abi::newer_layout() {
        bottom_write(this, va, &[value], 71);
    } else {
        unknown_slot(this.cast(), Kind::List, 71);
    }
}

/// Console list slot 72 (newer layout): `Write64BitValueBottomOfPipeX(address, value, flush,
/// flags)`. A measured call is `(address, 0, 1, ...)`. The timestamp form (flags
/// `BOTTOM_TIMESTAMPS`, a few per frame) writes the GPU clock as the CPU sees it once the list has
/// run (`cpu_writes`), so it records no words.
extern "system" fn list_write64_bottom(this: *mut Proxy, va: u64, value: u64, _flush: u32, flags: u32) {
    if !crate::abi::newer_layout() {
        unknown_slot(this.cast(), Kind::List, 72);
    } else if flags & BOTTOM_TIMESTAMPS != 0 {
        bottom_write(this, va, &[], 72);
    } else {
        bottom_write(this, va, &[value as u32, (value >> 32) as u32], 72);
    }
}

/// Why a bottom of pipe write of `bytes` bytes at `va` cannot be done by the CPU, if it cannot.
fn bottom_write_refusal(va: u64, bytes: usize) -> Option<&'static str> {
    if !va.is_multiple_of(bytes as u64) {
        Some("not aligned")
    } else if crate::views_at::lookup_resource(va, true).is_some() {
        Some("a GPU-visible resource (the GPU write for it is not done)")
    } else if !crate::cpu_writes::writable(va as usize, bytes) {
        Some("not committed, writable memory")
    } else {
        None
    }
}

/// Records one bottom-of-pipe write for the CPU to do after the list's submission (title memory),
/// or refuses it with the reason. No `words` is a GPU timestamp (`list_sync::CpuWrite`).
fn bottom_write(this: *mut Proxy, va: u64, words: &[u32], slot: u32) {
    static N: AtomicI32 = AtomicI32::new(0);
    if let Some(why) = bottom_write_refusal(va, crate::list_sync::write_bytes(words)) {
        if log_due(&N) {
            xs_log!("console list slot {slot}: bottom-of-pipe write at {va:#x} refused: {why}");
        }
        return;
    }
    if log_due(&N) {
        xs_log!("console list slot {slot}: bottom-of-pipe write at {va:#x}, words {words:x?}, recorded for the CPU after the GPU");
    }
    crate::list_sync::record_cpu_write(this as usize, va as usize, words.to_vec());
}

/// Console list slot 73 (newer layout): `Wait32BitValueX(address, comparison, value, mask, flags)`.
/// Not reached in the runs measured so far. The value is compared under the mask.
extern "system" fn list_wait32(this: *mut Proxy, va: u64, comparison: u32, value: u32, mask: u32, _flags: u32) {
    if crate::abi::newer_layout() {
        wait_memory(this, va, 4, comparison, u64::from(value), u64::from(mask), 73);
    } else {
        unknown_slot(this.cast(), Kind::List, 73);
    }
}

/// Console list slot 74 (newer layout): `Wait64BitValueX(address, comparison, value, flags)`. A
/// measured call is `(address, 7 = GREATER_EQUAL, 1, flags 1)`. The fourth argument is not a mask:
/// the whole 64 bit value is compared.
extern "system" fn list_wait64(this: *mut Proxy, va: u64, comparison: u32, value: u64, _flags: u64, _more: u32) {
    if crate::abi::newer_layout() {
        wait_memory(this, va, 8, comparison, value, u64::MAX, 74);
    } else {
        unknown_slot(this.cast(), Kind::List, 74);
    }
}

/// Console list slot 146 (newer layout): `uav_barrier_x(this, n, barriers)`. A console BVH builder
/// library calls it between its build passes so that the writes of one pass are visible to the
/// next. One global UAV barrier is recorded on the PC list. In the measured calls `n` is 1 and
/// the second argument a pointer, and the barrier array itself is not read.
extern "system" fn list_uav_barrier(this: *mut Proxy, n: u32, barriers: *const c_void) {
    type ResourceBarrier = unsafe extern "system" fn(Object, u32, *const c_void);
    static N: AtomicI32 = AtomicI32::new(0);
    if !crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 146);
    }
    let global = crate::list_sync::global_uav_barrier();
    // SAFETY: `this` is a live proxy; PC slot 26 of ID3D12GraphicsCommandList is ResourceBarrier.
    unsafe {
        let pc = (*this).pc;
        method::<ResourceBarrier>(pc, 26)(pc, 1, global.as_ptr().cast());
    }
    if log_due(&N) {
        xs_log!("console list slot 146 (UAV barrier, n {n}, barriers {barriers:?}): one global UAV barrier recorded on the PC list");
    }
}

/// Why a memory wait of `width` bytes at `va` is not recorded on a list, if it is not. A wait that
/// never comes true is refused, because the list would never run.
fn wait_refusal(va: u64, width: u8, comparison: u32) -> Option<&'static str> {
    use crate::list_sync::COMPARISON_NEVER;
    if !va.is_multiple_of(u64::from(width)) {
        Some("not aligned")
    } else if !crate::cpu_writes::readable(va as usize, usize::from(width)) {
        Some("not readable title memory (a GPU-visible resource is not mapped for the CPU)")
    } else if crate::list_sync::compare(comparison, 0, 0).is_none() {
        Some("unknown comparison")
    } else if comparison == COMPARISON_NEVER {
        Some("never true (refused: the list would never run)")
    } else {
        None
    }
}

/// Records one memory wait on list `this`, or leaves it out with the reason. A wait that is always
/// true needs nothing, and a wait on a CPU write of the same list is not recorded, since that write
/// runs after the list.
fn wait_memory(this: *mut Proxy, va: u64, width: u8, comparison: u32, value: u64, mask: u64, slot: u32) {
    use crate::list_sync::{COMPARISON_ALWAYS, PendingWait};
    static N: AtomicI32 = AtomicI32::new(0);
    let bits = u32::from(width) * 8;
    if let Some(why) = wait_refusal(va, width, comparison) {
        if log_due(&N) {
            xs_log!("console list slot {slot} (Wait{bits}BitValueX) at {va:#x}: not recorded: {why}");
        }
        return;
    }
    let skipped = if comparison == COMPARISON_ALWAYS {
        Some("always true")
    } else if crate::list_sync::has_cpu_write_over(this as usize, va, u64::from(width)) {
        Some("on a CPU write of this list")
    } else {
        None
    };
    if log_due(&N) {
        xs_log!(
            "console list slot {slot} (Wait{bits}BitValueX) at {va:#x}: comparison {comparison}, value {value:#x}, mask {mask:#x}, read now {:x?}: {}",
            crate::cpu_writes::read_title(va as usize, usize::from(width)),
            skipped.unwrap_or("recorded, the list waits for it")
        );
    }
    if skipped.is_none() {
        crate::list_sync::record_wait(this as usize, PendingWait { va, width, comparison, value, mask });
    }
}

/// Console list slot 70 (newer layout): `OMSetDepthBounds(min, max)`, forwarded to PC slot 62
/// (`ID3D12GraphicsCommandList1::OMSetDepthBounds`), the target of the older layout's slot 68.
extern "system" fn list_set_depth_bounds_x(this: *mut Proxy, min: f32, max: f32) {
    static SEEN: AtomicBool = AtomicBool::new(false);
    if !crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 70);
    }
    if first_time(&SEEN) {
        xs_log!("console list slot 70 (OMSetDepthBounds, newer layout): min {min}, max {max}");
    }
    // SAFETY: `this` is a live proxy; PC slot 62 is ID3D12GraphicsCommandList1::OMSetDepthBounds.
    unsafe {
        let pc = (*this).pc;
        method::<DepthBounds>(pc, 62)(pc, min, max);
    }
}

/// Console list slot 85 (newer layout): `CopyTextureRegionX(dst, x, y, z, src, box, flags)`,
/// forwarded to PC slot 16 like `list_copy_texture_x` (the older layout's slot 83). The locations
/// get their resources unwrapped, the flags are ignored.
extern "system" fn list_copy_texture_region_x(
    this: *mut Proxy,
    dst: *const c_void,
    x: u32,
    y: u32,
    z: u32,
    src: *const c_void,
    boxed: *const c_void,
    flags: u32,
) {
    static SEEN: AtomicBool = AtomicBool::new(false);
    static FLAGS: AtomicI32 = AtomicI32::new(0);
    if !crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 85);
    }
    if first_time(&SEEN) {
        xs_log!("console list slot 85 (CopyTextureRegionX, newer layout): first call at ({x}, {y}, {z})");
    }
    if flags != 0 && log_due(&FLAGS) {
        xs_log!("console list slot 85 (CopyTextureRegionX): flags {flags:#x} not zero (ignored)");
    }
    // SAFETY: `this` is a live proxy; the copy locations go to PC as copies with their resources unwrapped.
    unsafe {
        let pc = (*this).pc;
        let (dst_loc, src_loc) = (unwrapped_location(dst), unwrapped_location(src));
        note_copy_dst(&dst_loc);
        method::<CopyTexture>(pc, 16)(pc, location_ptr(&dst_loc, dst), x, y, z, location_ptr(&src_loc, src), boxed);
    }
}

/// Console list slot 86 (newer layout): `CopyResourceX(dst, src, flags)`, forwarded to PC slot 17
/// (`CopyResource`) with both proxies unwrapped. The flags are ignored.
extern "system" fn list_copy_resource_x(this: *mut Proxy, dst: Object, src: Object, flags: u32) {
    static SEEN: AtomicBool = AtomicBool::new(false);
    static FLAGS: AtomicI32 = AtomicI32::new(0);
    if !crate::abi::newer_layout() {
        return unknown_slot(this.cast(), Kind::List, 86);
    }
    if first_time(&SEEN) {
        xs_log!("console list slot 86 (CopyResourceX, newer layout): first call");
    }
    if flags != 0 && log_due(&FLAGS) {
        xs_log!("console list slot 86 (CopyResourceX): flags {flags:#x} not zero (ignored)");
    }
    // SAFETY: `this` is a live proxy; `dst` and `src` are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        let target = unwrap(dst);
        crate::rt_diag::copied_into(target as usize);
        crate::detile::note_gpu_write(target as usize);
        method::<CopyResource>(pc, 17)(pc, target, unwrap(src));
    }
}

/// The PC pipeline a Reset starts list `this` with, noted for the stand ins of `pso_diag`. A
/// pipeline made on first bind that the PC runtime refused starts the list without one, and its
/// work is left out.
///
/// # Safety
/// `pso` is null or a pipeline the title holds (ours or a PC one).
unsafe fn bound_at_reset(this: *mut Proxy, pso: Object) -> Object {
    // SAFETY: by the contract above.
    unsafe {
        crate::pso_diag::bind(this as usize, pso);
        let made = unwrap(pso);
        if made.is_null() && !pso.is_null() && crate::lazy_pso::is_lazy(pso) {
            crate::pso_diag::bind_missing(this as usize);
        }
        crate::kind_check::bound(this as usize, made as usize);
        made
    }
}

/// Resets list `this` with `allocator` and `pso` (proxies or PC objects): every table of the layer
/// that follows the recording of a list forgets it, then the PC list is reset.
fn reset_list(this: *mut Proxy, allocator: Object, pso: Object) -> HRESULT {
    let key = this as usize;
    crate::list_sync::forget(key);
    crate::user_mirror::forget(key);
    crate::title_heap::list_reset(key);
    crate::cbv_copies::forget(key);
    crate::window_service::forget(key);
    crate::rect_list::forget(key);
    crate::draw_diag::forget(key);
    crate::flicker_probe::forget(key);
    crate::kind_check::reset(key);
    // SAFETY: `this` is a live proxy, `allocator` and `pso` are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        forget_unclosed(pc);
        crate::detile::list_reset(pc as usize);
        crate::present::list_reset(pc as usize);
        let pso = bound_at_reset(this, pso);
        crate::rt_diag::list_reset(key, pso as usize);
        let allocator = unwrap(allocator);
        crate::upload_mem::with(|m| m.set_allocator(key, allocator as usize));
        method::<Reset>(pc, 10)(pc, allocator, pso)
    }
}

/// Console list slot 10 (older layout): `Reset(allocator, pso)`, the allocator may be a proxy.
extern "system" fn list_reset(this: *mut Proxy, allocator: Object, pso: Object) -> HRESULT {
    reset_list(this, allocator, pso)
}

/// Console list slot 139 (newer layout, D-2026-11-23-2): `Reset` with one extra argument that the
/// title passes but the PC method does not take (`(allocator, NULL, 0)` before it records a list
/// taken from its pool). Physical slot 139 only means this once the newer layout has been
/// observed (`abi::newer_layout`). Before that it is some other, unidentified method, treated as an
/// ordinary unknown slot.
extern "system" fn list_reset_late(this: *mut Proxy, allocator: Object, pso: Object, _extra: usize) -> HRESULT {
    if crate::abi::newer_layout() {
        reset_list(this, allocator, pso)
    } else {
        E_NOTIMPL
    }
}

// ---- queue ----

/// PC command lists whose last Close failed. Native D3D12 removes the device when such a list is
/// executed, so they are left out until a Reset records them anew.
static UNCLOSED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Console list slot 9: Close (PC order). A refusal is logged and remembered (`UNCLOSED`).
extern "system" fn list_close(this: *mut Proxy) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    // SAFETY: `this` is a live list proxy.
    unsafe {
        let pc = (*this).pc;
        crate::capture::flush(this as usize, pc);
        let hr = method::<unsafe extern "system" fn(Object) -> HRESULT>(pc, 9)(pc);
        if hr < 0 {
            UNCLOSED.lock().unwrap_or_else(|e| e.into_inner()).push(pc as usize);
            if log_due(&N) {
                xs_log!("console command list {}: Close refused by the PC runtime, hr=0x{:08X}; left out of its submissions", p(pc), hr as u32);
            }
        }
        hr
    }
}

/// A list was reset: whatever its last Close did no longer matters.
fn forget_unclosed(pc_list: Object) {
    UNCLOSED.lock().unwrap_or_else(|e| e.into_inner()).retain(|&l| l != pc_list as usize);
}

/// ExecuteCommandLists (PC 10) of the PC lists `lists` on PC queue `pc`, without the ones whose
/// Close failed. An empty submission still reaches the queue, as the title's did.
///
/// # Safety
/// `pc` is a live PC queue and `lists` live PC command lists.
unsafe fn execute_closed(pc: Object, lists: &[Object]) {
    static DROPPED: AtomicI32 = AtomicI32::new(0);
    // the title may have written memory while its lists waited behind a memory wait
    // SAFETY: by the contract above.
    unsafe { crate::title_mem::sync(pc) };
    let kept: Vec<Object> = {
        let unclosed = UNCLOSED.lock().unwrap_or_else(|e| e.into_inner());
        lists.iter().copied().filter(|l| !unclosed.contains(&(*l as usize))).collect()
    };
    if kept.len() != lists.len() && log_due(&DROPPED) {
        xs_log!("console queue: {} unclosed command lists left out of a submission of {}", lists.len() - kept.len(), lists.len());
    }
    // SAFETY: by the contract above.
    unsafe { method::<Execute>(pc, 10)(pc, kept.len() as u32, kept.as_ptr()) };
    crate::diagnose::removed_after(|| format!("a title submission of {} lists on queue {pc:p}", kept.len()));
}

extern "system" fn queue_execute(this: *mut Proxy, n: u32, lists: *const Object) {
    let _pc_call = crate::watchdog_hold::pc_call();
    // SAFETY: `this` is a live proxy, `lists` points to `n` objects.
    unsafe {
        let mapped = unwrap_array(n, lists);
        let pc = (*this).pc;
        // objects released from now on wait for this submission
        let tracked = crate::retire::begin(pc);
        // the title bytes that running submissions read and that this sync is about to copy (`flicker_probe`)
        let suspects = crate::flicker_probe::before_sync();
        // the memory the title wrote since its last submission is in the GPU mirror before its lists run
        crate::title_mem::sync(pc);
        // the textures the title wrote since its last submission are readable by the GPU before its lists run
        crate::detile::flush(pc);
        let keys: Vec<usize> = if n > 0 && !lists.is_null() { (0..n as usize).map(|i| *lists.add(i) as usize).collect() } else { Vec::new() };
        crate::draw_diag::submitted(&keys);
        // constant buffer copies of these lists take the bytes the title wrote since their recording
        let refresh = crate::cbv_copies::submitted(&keys);
        if refresh.stale > 0 || refresh.kept > 0 {
            static STALE: AtomicI32 = AtomicI32::new(0);
            if log_due(&STALE) {
                xs_log!("console queue: {} constant buffer copies of {} lists, {} changed since recording and taken again, {} of lists submitted again left as they are", refresh.copies, keys.len(), refresh.stale, refresh.kept);
            }
        }
        let watched = crate::flicker_probe::submitting(pc, &keys, suspects);
        // a list whose memory waits are not met runs after the lists before it, once they are
        // (the GPU has to run those first); the wait holds it back for at most WAIT_TIMEOUT
        let mut start = 0usize;
        for (i, &key) in keys.iter().enumerate() {
            let waits = crate::list_sync::waits_of(key);
            if waits.iter().all(crate::cpu_writes::satisfied) {
                continue;
            }
            if i > start {
                execute_closed(pc, &mapped[start..i]);
            }
            for w in &waits {
                if !crate::cpu_writes::wait_until(w, crate::list_sync::WAIT_TIMEOUT) {
                    static N: AtomicI32 = AtomicI32::new(0);
                    if log_due(&N) {
                        xs_log!("console list: a memory wait at {:#x} ({} bytes, comparison {}, value {:#x}, mask {:#x}) timed out, the list runs anyway", w.va, w.width, w.comparison, w.value, w.mask);
                    }
                }
            }
            start = i;
        }
        if start < mapped.len() || mapped.is_empty() {
            execute_closed(pc, &mapped[start..]);
        }
        crate::flicker_probe::submitted(pc, watched);
        crate::detile::submitted(&mapped.iter().map(|&l| l as usize).collect::<Vec<_>>());
        crate::present::list_submitted(&mapped.iter().map(|&l| l as usize).collect::<Vec<_>>());
        crate::rt_diag::executed(&keys);
        // the lists' bottom-of-pipe writes into title memory: done once the GPU has finished this
        if !keys.is_empty() {
            let writes = crate::list_sync::cpu_writes_of(&keys);
            static N: AtomicI32 = AtomicI32::new(0);
            if log_due(&N) {
                let waits: usize = keys.iter().map(|&k| crate::list_sync::waits_of(k).len()).sum();
                xs_log!("console queue: {} lists, {} CPU writes, {waits} waits recorded on them", keys.len(), writes.len());
            }
            crate::cpu_writes::after(pc, writes);
        }
        if tracked {
            crate::retire::submitted(pc);
        }
    }
}

/// Console queue slot 18. In the newer layout it is `GetTimestampFrequency(out u64)`, PC slot 16:
/// A retail title calls it with a pointer to one u64 and divides 1000.0 by the result.
/// Forwarding it to PC 18 (`GetDesc`, 16 bytes) overwrote the caller's saved register.
/// In the older layout console 18 is still `GetDesc` (PC 18, the descriptor returned through a
/// hidden result pointer, so the pointer comes back in rax).
extern "system" fn queue_slot18(this: *mut Proxy, out: *mut c_void) -> isize {
    // SAFETY: `this` is a live proxy, `out` is the caller's result pointer.
    unsafe {
        let pc = (*this).pc;
        if crate::abi::newer_layout() {
            static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::SeqCst) {
                xs_log!("console queue slot 18 (GetTimestampFrequency, newer layout): PC 16");
            }
            method::<unsafe extern "system" fn(Object, *mut u64) -> HRESULT>(pc, 16)(pc, out.cast()) as isize
        } else {
            method::<unsafe extern "system" fn(Object, *mut c_void) -> *mut c_void>(pc, 18)(pc, out) as isize
        }
    }
}

/// Console queue slot 19. In the newer layout it is `GetClockCalibration(gpu u64, cpu u64)`, PC
/// slot 17, by the same shift by two as slot 18 (inferred, not yet seen called). In the older
/// layout the slot is a console method we do not know.
extern "system" fn queue_slot19(this: *mut Proxy, gpu: *mut u64, cpu: *mut u64) -> HRESULT {
    if !crate::abi::newer_layout() {
        unknown_slot(this.cast(), Kind::Queue, 19);
        return E_NOTIMPL;
    }
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::SeqCst) {
        xs_log!("console queue slot 19 (GetClockCalibration, newer layout): PC 17");
    }
    // SAFETY: `this` is a live proxy, `gpu` and `cpu` are the caller's result pointers.
    unsafe {
        let pc = (*this).pc;
        method::<unsafe extern "system" fn(Object, *mut u64, *mut u64) -> HRESULT>(pc, 17)(pc, gpu, cpu)
    }
}

/// Console queue slots 16 and 17 in the newer layout. Slot 18 moved to GetTimestampFrequency, so
/// 16 and 17 are no longer GetTimestampFrequency and GetClockCalibration there. Forwarding them
/// to PC 16 and 17 wrote 1,000,000,000 (the NVIDIA timestamp frequency) over the first argument,
/// and the title later called through that value as a vtable. Until the
/// methods are identified they are logged and answer S_OK. The older layout forwards as before.
fn queue_slot16_17(this: *mut Proxy, slot: usize, a: usize, b: usize, c: usize) -> HRESULT {
    if !crate::abi::newer_layout() {
        // SAFETY: `this` is a live proxy; the older layout's 16 and 17 are the PC methods.
        return unsafe {
            let pc = (*this).pc;
            method::<unsafe extern "system" fn(Object, usize, usize, usize) -> HRESULT>(pc, slot)(pc, a, b, c)
        };
    }
    // Measured in a retail title: slot 16 gets (fence, 1), (fence, 2) on one fence, so it is Signal (PC 14),
    // two slots later than on PC like 18. Slot 17 is taken as Wait (PC 15) by the same shift. The
    // third argument (flags, as on slot 23) has no PC meaning.
    static N: [AtomicI32; 2] = [AtomicI32::new(0), AtomicI32::new(0)];
    if N[slot - 16].fetch_add(1, Ordering::Relaxed) < 4 {
        xs_log!("console queue slot {slot} (newer layout, {}): fence {a:#x}, value {b:#x}, third {c:#x}", if slot == 16 { "Signal, PC 14" } else { "Wait, PC 15" });
    }
    queue_fence_op(this, slot == 16, a as Object, b as u64)
}

/// Signal (PC 14) or Wait (PC 15) of `fence` (a proxy or a PC fence) to `value`. A signal also
/// tells the completed value cells (`fence_cell`) which value to watch for.
fn queue_fence_op(this: *mut Proxy, signal: bool, fence: Object, value: u64) -> HRESULT {
    // SAFETY: `this` is a live proxy, `fence` a fence proxy or a PC fence.
    unsafe {
        let pc = (*this).pc;
        let fence = unwrap(fence);
        let hr = method::<Signal>(pc, if signal { 14 } else { 15 })(pc, fence, value);
        if signal && hr >= 0 {
            crate::fence_cell::queued(fence, value);
        }
        hr
    }
}

/// Console queue slots 14 and 15: Signal and Wait in the older layout. The newer layout put two
/// other console methods here (Signal and Wait moved to 16 and 17); they are logged.
fn queue_slot14_15(this: *mut Proxy, slot: usize, a: usize, b: usize) -> HRESULT {
    if crate::abi::newer_layout() {
        unknown_slot(this.cast(), Kind::Queue, slot as u32);
        return E_NOTIMPL;
    }
    queue_fence_op(this, slot == 14, a as Object, b as u64)
}

extern "system" fn queue_slot14(this: *mut Proxy, a: usize, b: usize) -> HRESULT {
    queue_slot14_15(this, 14, a, b)
}

extern "system" fn queue_slot15(this: *mut Proxy, a: usize, b: usize) -> HRESULT {
    queue_slot14_15(this, 15, a, b)
}

// ---- fence ----

/// Console device slot 36: CreateFence (PC order), the fence wrapped for its console slot 12.
extern "system" fn dev_create_fence(this: *mut Proxy, initial: u64, flags: u32, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live device proxy, `out` writable.
    unsafe {
        let pc = (*this).pc;
        let mut fence: Object = std::ptr::null_mut();
        let hr = method::<unsafe extern "system" fn(Object, u64, u32, *const GUID, *mut Object) -> HRESULT>(pc, 36)(pc, initial, flags, riid, &mut fence);
        give(hr, fence, Kind::Fence, out)
    }
}

/// Console fence slot 10: Signal from the CPU (PC order), then the cells catch up.
extern "system" fn fence_signal(this: *mut Proxy, value: u64) -> HRESULT {
    // SAFETY: `this` is a live fence proxy.
    unsafe {
        let pc = (*this).pc;
        let hr = method::<unsafe extern "system" fn(Object, u64) -> HRESULT>(pc, 10)(pc, value);
        crate::fence_cell::cpu_signalled(pc);
        hr
    }
}

/// Console fence slot 12 (newer layout): the address of the completed value.
extern "system" fn fence_value_address(this: *mut Proxy) -> *const u64 {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    // SAFETY: `this` is a live fence proxy.
    let pc = unsafe { (*this).pc };
    if !LOGGED.swap(true, Ordering::SeqCst) {
        xs_log!("console fence slot 12 (completed value address): a cell kept equal to the PC fence {}", p(pc));
    }
    crate::fence_cell::address(pc)
}

extern "system" fn queue_slot16(this: *mut Proxy, a: usize, b: usize, c: usize) -> HRESULT {
    queue_slot16_17(this, 16, a, b, c)
}

extern "system" fn queue_slot17(this: *mut Proxy, a: usize, b: usize, c: usize) -> HRESULT {
    queue_slot16_17(this, 17, a, b, c)
}

/// Console queue slots 20 and up. The newer layout has two more methods before them, so its slot
/// is the older slot plus two from 16 on. One handler per slot translates to the older number and
/// runs the older method: 18 GetDesc, 23 SignalX, 30 PresentX, 33 ExecuteCommandListsX. Methods we
/// do not model answer S_OK and are logged, because a title needs it (frame pacing runs
/// through some of them and an error code makes it give up).
extern "system" fn queue_late<const SLOT: u32>(this: *mut Proxy, a: usize, b: usize, c: usize, d: usize) -> usize {
    let older = if crate::abi::newer_layout() { SLOT - 2 } else { SLOT };
    match older {
        // SAFETY: `this` is a live proxy, `a` the caller's result pointer (hidden return).
        18 => unsafe {
            let pc = (*this).pc;
            method::<unsafe extern "system" fn(Object, *mut c_void) -> *mut c_void>(pc, 18)(pc, a as *mut c_void) as usize
        },
        23 => queue_signal_x(this, a as Object, b as u64, c as u64) as u32 as usize,
        29 => queue_copy_page_mappings(a as u32, b as *const crate::page_maps::Batch, c as u64, d as u32) as u32 as usize,
        30 => queue_present_x(this, a as u32, b as *const c_void, c as *const c_void) as u32 as usize,
        33 => {
            static LOGGED: AtomicBool = AtomicBool::new(false);
            if !LOGGED.swap(true, Ordering::SeqCst) {
                xs_log!("console queue slot {SLOT} (ExecuteCommandListsX): {a} lists, value {c:#x}, flags {d:#x} ignored");
            }
            queue_execute(this, a as u32, b as *const Object);
            0
        }
        _ => {
            static N: [AtomicI32; QUEUE_SLOTS as usize] = [const { AtomicI32::new(0) }; QUEUE_SLOTS as usize];
            if log_due(&N[SLOT as usize]) {
                xs_log!("console queue slot {SLOT} (older {older}): not modelled, S_OK; args {a:#x} {b:#x} {c:#x} {d:#x}");
            }
            0
        }
    }
}

macro_rules! queue_late_table {
    ($($n:literal)*) => { [$(queue_late::<$n> as *const () as usize),*] };
}

/// The handlers of queue slots 20 to 47, in slot order.
fn queue_late_slot(slot: u32) -> Option<usize> {
    static LATE: OnceLock<[usize; 28]> = OnceLock::new();
    let table = LATE.get_or_init(|| queue_late_table!(20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47));
    table.get(slot.checked_sub(20)? as usize).copied()
}

/// Console queue slot 29 (31 in the newer layout): the page mapping copies,
/// `(batch count, batches, pool address, pool pages, flags)`. Each batch shows pool pages at its
/// destination address (`page_maps`), which `xmem.dll` does for the CPU and the GPU window alike.
/// Done at once, in the order of the calls. The title is always answered `S_OK`: a mapping that
/// cannot be made leaves a texture unfilled, which a failure of the call would not mend.
fn queue_copy_page_mappings(count: u32, batches: *const crate::page_maps::Batch, pool: u64, pool_pages: u32) -> HRESULT {
    use std::sync::atomic::AtomicU64;
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    static UNMAPPED: AtomicU64 = AtomicU64::new(0);
    static FAILED: AtomicU64 = AtomicU64::new(0);
    static REFUSED: AtomicU64 = AtomicU64::new(0);
    static UNAVAILABLE: AtomicBool = AtomicBool::new(false);
    let call = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    if batches.is_null() || count == 0 || count > 1 << 20 {
        return S_OK;
    }
    let mut first_destination = 0u64;
    let (mut ranges_seen, mut pages_seen) = (0u64, 0u64);
    for i in 0..count as usize {
        // SAFETY: the title passes `count` batches, and each batch `count` ranges.
        let (batch, ranges) = unsafe {
            let batch = batches.add(i).read_unaligned();
            if batch.ranges.is_null() || batch.count == 0 || batch.count > 1 << 20 {
                continue;
            }
            (batch, std::slice::from_raw_parts(batch.ranges, batch.count as usize))
        };
        if first_destination == 0 {
            first_destination = batch.destination;
        }
        ranges_seen += ranges.len() as u64;
        let (requests, refused) = crate::page_maps::requests(batch.destination, ranges, pool, pool_pages);
        REFUSED.fetch_add(u64::from(refused), Ordering::Relaxed);
        for r in requests {
            pages_seen += u64::from(r.pages);
            match crate::title_mem::show_pages(r.to, r.from, r.pages) {
                Some(true) if r.from == 0 => UNMAPPED.fetch_add(u64::from(r.pages), Ordering::Relaxed),
                Some(true) => SHOWN.fetch_add(u64::from(r.pages), Ordering::Relaxed),
                Some(false) => FAILED.fetch_add(u64::from(r.pages), Ordering::Relaxed),
                None => {
                    if !UNAVAILABLE.swap(true, Ordering::SeqCst) {
                        xs_log!("console queue page mappings: xmem.dll cannot show pool pages at another address (title memory off, or an older xmem.dll), the mappings are left out");
                    }
                    FAILED.fetch_add(u64::from(r.pages), Ordering::Relaxed)
                }
            };
        }
    }
    // The first calls in full, then the counts now and then: the line of a call must not be the cost of it.
    if call <= 4 || call.is_power_of_two() || call.is_multiple_of(1024) {
        xs_log!(
            "console queue page mappings: call {call}: {count} batches, {ranges_seen} ranges, {pages_seen} pages, first destination {first_destination:#x}, pool {pool:#x} of {pool_pages} pages; so far {} pages shown, {} unmapped, {} failed, {} ranges refused",
            SHOWN.load(Ordering::Relaxed),
            UNMAPPED.load(Ordering::Relaxed),
            FAILED.load(Ordering::Relaxed),
            REFUSED.load(Ordering::Relaxed)
        );
    }
    S_OK
}

/// Console queue slot 23: Signal(fence, value, flags). The PC Signal (slot 14) has no
/// flags, so flags that are not zero are logged and ignored.
extern "system" fn queue_signal_x(this: *mut Proxy, fence: Object, value: u64, flags: u64) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    if flags != 0 && log_due(&N) {
        xs_log!("console queue slot 23 (Signal): flags {:016X} ignored (meaning unknown)", flags);
    }
    queue_fence_op(this, true, fence, value)
}

/// Console queue slot 30, PresentX(plane count, planes, parameters) (seen in a retail title). A plane
/// starts with {token u64 at 0, resource count u32 at 8, resource array at 0x10}. The rest of the
/// plane and the parameter structure are not known. They are passed to the hook as pointers,
/// untouched. Without a hook the token is read (0 is a valid token, a console present does not
/// fail for the title) and handed to the frame clock (`frame::mark_flipped`), the first resource
/// of plane 0 goes to the built in present, and the answer is `S_OK` so a title that uses
/// `PresentX` on its own is not stuck here.
extern "system" fn queue_present_x(this: *mut Proxy, plane_count: u32, planes: *const c_void, parameters: *const c_void) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    heap_pages::present();
    crate::flicker_probe::present();
    crate::retire::present();
    let hook = PRESENT_HOOK.load(Ordering::SeqCst);
    if hook != 0 {
        // SAFETY: the hook was installed through `DecantD3D12X_SetPresentHook` with this signature.
        unsafe {
            let hook = std::mem::transmute::<usize, PresentFn>(hook);
            return hook((*this).pc, plane_count, planes, parameters);
        }
    }
    // Plane 0 is readable for its 0x18 bytes when there is a plane at all, and its array holds
    // `count` resource pointers.
    let plane_zero = (plane_count > 0 && !planes.is_null()).then_some(planes.cast::<u8>());
    // SAFETY: by the line above, `plane` points to a plane 0.
    let token = plane_zero.map_or(0, |plane| unsafe { plane.cast::<u64>().read_unaligned() });
    // SAFETY: as above, the resource array is read only when the plane has one.
    let shown = plane_zero.is_some_and(|plane| unsafe {
        let count = plane.add(8).cast::<u32>().read_unaligned();
        let array = plane.add(0x10).cast::<*const Object>().read_unaligned();
        if count == 0 || array.is_null() {
            return false;
        }
        let frame = unwrap(array.read_unaligned());
        crate::capture::presented((*this).pc, frame);
        crate::present::show((*this).pc, frame)
    });
    crate::pso_diag::present();
    crate::direct_pso::note_present();
    crate::watchdog_hold::note_present();
    crate::frame::mark_flipped(token);
    // Only plane 0 is shown. A change of the plane count is logged every time (a few dozen at
    // most): a screen the title composes from two planes shows without its second one.
    static PLANES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
    static PLANE_LINES: AtomicI32 = AtomicI32::new(0);
    if PLANES.swap(plane_count, Ordering::Relaxed) != plane_count && PLANE_LINES.fetch_add(1, Ordering::Relaxed) < 64 {
        xs_log!("console queue slot 30 (PresentX): the plane count is now {plane_count} (token {token:016X}), only plane 0 is shown");
    }
    if log_due(&N) {
        xs_log!(
            "console queue slot 30 (PresentX) ({plane_count}, {}, {}, token {token:016X}): no present hook installed, built in present {}",
            p(planes),
            p(parameters),
            if shown { "showed it" } else { "did not show it" }
        );
    }
    S_OK
}

// ---- command allocator ----

/// Console allocator slot 9: `(function, context)`, a "set memory callbacks" method. The function
/// (size and alignment, calls the title's allocator) and the context are kept in the proxy and not
/// called: the PC allocator has no use for them.
extern "system" fn allocator_set_callbacks(this: *mut AllocProxy, function: usize, context: usize) -> HRESULT {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    // SAFETY: `this` is a live allocator proxy (it has the larger layout).
    unsafe {
        (*this).callback.store(function, Ordering::SeqCst);
        (*this).context.store(context, Ordering::SeqCst);
    }
    if !LOGGED.swap(true, Ordering::SeqCst) {
        xs_log!("console allocator slot 9 (memory callbacks): recorded, not used");
    }
    S_OK
}

// ---- descriptor heap ----

/// Console slots 9 and 10: the handle comes back in rax, the PC writes it through a result pointer.
extern "system" fn heap_cpu_start(this: *mut Proxy) -> u64 {
    // SAFETY: `this` is a live proxy.
    let pc = unsafe { (*this).pc };
    if let Some(title_base) = crate::title_heap::title_start_for_heap(pc as usize) {
        return title_base as u64;
    }
    let mut handle = 0u64;
    // SAFETY: `handle` is writable storage for the PC result.
    unsafe {
        method::<HeapStart>(pc, 9)(pc, &mut handle);
    }
    handle
}

extern "system" fn heap_gpu_start(this: *mut Proxy) -> u64 {
    let mut handle = 0u64;
    // SAFETY: as in `heap_cpu_start`.
    unsafe {
        let pc = (*this).pc;
        method::<HeapStart>(pc, 10)(pc, &mut handle);
    }
    handle
}

// ---- resource arguments of the relayed calls ----
//
// A placed resource is a `Kind::Resource` proxy. Each relayed call that takes a resource, directly or
// inside a structure the title passes, gets the PC object here. Structures are copied and the copy is
// rewritten (`resource_args`); the title's memory is never written.

/// Console resource slot 8 (Map, PC 8). On PC a DEFAULT resource often cannot be mapped (the PC
/// runtime answers E_INVALIDARG), while on the console every placed resource is CPU visible at its
/// title address. A placed resource whose Map fails maps at that address, and its Unmap is
/// skipped (`res_unmap`). The decision is [`crate::views_at::map_answer`].
extern "system" fn res_map(this: *mut Proxy, sub: u32, range: *const c_void, out: *mut *mut c_void) -> HRESULT {
    static LOGGED: AtomicI32 = AtomicI32::new(0);
    // SAFETY: `this` is a live resource proxy, so a `ResProxy`; `range` and `out` are the caller's.
    unsafe {
        let res = &*this.cast::<ResProxy>();
        let pc_hr = method::<ResourceMap>(res.base.pc, 8)(res.base.pc, sub, range, out);
        let answer = crate::views_at::map_answer(pc_hr >= 0, res.placed_va);
        if !matches!(answer, MapAnswer::Failed) {
            res.map_fallback.store(matches!(answer, MapAnswer::Placed(_)), Ordering::Relaxed);
        }
        let MapAnswer::Placed(va) = answer else {
            return pc_hr;
        };
        if !out.is_null() {
            *out = va as usize as *mut c_void;
        }
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            xs_log!("ID3D12Resource slot 8 Map: PC hr=0x{:08X} -> placed title VA 0x{va:016X}", pc_hr as u32);
        }
        S_OK
    }
}

/// Console resource slot 9 (Unmap, PC 9): nothing to undo when the last Map gave the placed address.
// Console resource slot 9 (Unmap, PC 9): nothing to undo when the last Map gave the placed address.
extern "system" fn res_unmap(this: *mut Proxy, sub: u32, range: *const c_void) {
    // SAFETY: `this` is a live resource proxy, so a `ResProxy`; `range` is the caller's.
    unsafe {
        let res = &*this.cast::<ResProxy>();
        if res.map_fallback.load(Ordering::Relaxed) {
            // the title wrote through the placed address: the texture there is to be detiled again
            crate::detile::mark_resource(res.base.pc as usize);
            return;
        }
        method::<ResourceUnmap>(res.base.pc, 9)(res.base.pc, sub, range);
    }
}

/// Console resource slot 11 (GetGPUVirtualAddress, PC 11). PC answers 0 for a texture; a placed
/// resource answers the address the title placed it at (`views_at::title_gpu_va`).
extern "system" fn res_gpu_virtual_address(this: *mut Proxy) -> u64 {
    static LOGGED: AtomicI32 = AtomicI32::new(0);
    // SAFETY: `this` is a live resource proxy, so a `ResProxy`, and its PC object is live.
    unsafe {
        let res = &*this.cast::<ResProxy>();
        let pc_va = method::<GpuVirtualAddress>(res.base.pc, 11)(res.base.pc);
        let va = crate::views_at::title_gpu_va(pc_va, res.placed_va);
        if pc_va == 0 && va != 0 && LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            xs_log!("ID3D12Resource slot 11 GetGPUVirtualAddress: PC VA 0 -> placed title VA 0x{va:016X}");
        }
        va
    }
}

/// The PC object behind a pointer stored in a structure (the pointer itself when it is not a proxy).
///
/// # Safety
/// `p` is null or a COM object as the title passed it (see [`unwrap`]).
unsafe fn pc_pointer(p: u64) -> u64 {
    // SAFETY: by the contract above.
    unsafe { unwrap(p as usize as Object) as usize as u64 }
}

/// The copy of a `D3D12_TEXTURE_COPY_LOCATION` the title passed, with its resource unwrapped. `None`
/// for a null pointer.
///
/// # Safety
/// `location` is null or points to [`COPY_LOCATION_BYTES`] readable bytes.
unsafe fn unwrapped_location(location: *const c_void) -> Option<[u8; COPY_LOCATION_BYTES]> {
    if location.is_null() {
        return None;
    }
    // SAFETY: by the contract above.
    let mut copy: [u8; COPY_LOCATION_BYTES] = unsafe { std::ptr::read_unaligned(location.cast()) };
    resource_args::unwrap_copy_location(&mut copy, |p| unsafe { pc_pointer(p) });
    Some(copy)
}

/// The title copies into the texture of `location` on the GPU: it is no longer detiled from title memory.
fn note_copy_dst(location: &Option<[u8; COPY_LOCATION_BYTES]>) {
    if let Some(c) = location {
        let dst = u64::from_le_bytes(c[0..8].try_into().unwrap_or([0; 8])) as usize;
        crate::rt_diag::copied_into(dst);
        crate::detile::note_gpu_write(dst);
    }
}

/// The location pointer to pass to PC: the unwrapped copy when there is one, else the title's pointer
/// (which is then null).
fn location_ptr(copy: &Option<[u8; COPY_LOCATION_BYTES]>, title: *const c_void) -> *const c_void {
    copy.as_ref().map_or(title, |c| c.as_ptr().cast())
}

/// Console list slot 15 (CopyBufferRegion, PC 15): both buffers unwrapped.
extern "system" fn list_copy_buffer_region(this: *mut Proxy, dst: Object, dst_off: u64, src: Object, src_off: u64, n: u64) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        method::<CopyBufferRegion>(pc, 15)(pc, unwrap(dst), dst_off, unwrap(src), src_off, n);
    }
}

/// Console list slot 16 (CopyTextureRegion, PC 16): the copy locations go to PC as unwrapped copies.
extern "system" fn list_copy_texture_region(this: *mut Proxy, dst: *const c_void, x: u32, y: u32, z: u32, src: *const c_void, boxed: *const c_void) {
    // SAFETY: `this` is a live command list proxy; the locations are null or readable.
    unsafe {
        let pc = (*this).pc;
        let (dst_loc, src_loc) = (unwrapped_location(dst), unwrapped_location(src));
        note_copy_dst(&dst_loc);
        method::<CopyTexture>(pc, 16)(pc, location_ptr(&dst_loc, dst), x, y, z, location_ptr(&src_loc, src), boxed);
    }
}

/// Console list slot 17 (CopyResource, PC 17): both resources unwrapped (the newer layout's slot 86
/// does the same, see `list_copy_resource_x`).
extern "system" fn list_copy_resource(this: *mut Proxy, dst: Object, src: Object) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        crate::rt_diag::copied_into(unwrap(dst) as usize);
        crate::detile::note_gpu_write(unwrap(dst) as usize);
        method::<CopyResource>(pc, 17)(pc, unwrap(dst), unwrap(src));
    }
}

/// Console list slot 18 (CopyTiles, PC 18): the tiled resource and the buffer unwrapped.
extern "system" fn list_copy_tiles(this: *mut Proxy, tiled: Object, start: *const c_void, size: *const c_void, buffer: Object, buffer_off: u64, flags: u32) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        method::<CopyTiles>(pc, 18)(pc, unwrap(tiled), start, size, unwrap(buffer), buffer_off, flags);
    }
}

/// Console list slot 19 (ResolveSubresource, PC 19): both resources unwrapped.
extern "system" fn list_resolve_subresource(this: *mut Proxy, dst: Object, dst_sub: u32, src: Object, src_sub: u32, format: u32) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        crate::rt_diag::copied_into(unwrap(dst) as usize);
        method::<ResolveSubresource>(pc, 19)(pc, unwrap(dst), dst_sub, unwrap(src), src_sub, format);
    }
}

/// Console list slot 26 (ResourceBarrier, PC 26): the barriers go to PC as a copy with their
/// resources unwrapped.
extern "system" fn list_resource_barrier(this: *mut Proxy, n: u32, barriers: *const c_void) {
    // SAFETY: `this` is a live command list proxy; `barriers` is null or `n` readable barriers.
    unsafe {
        let pc = (*this).pc;
        crate::capture::flush(this as usize, pc);
        if barriers.is_null() || n == 0 {
            method::<Barriers>(pc, 26)(pc, n, barriers);
            return;
        }
        let mut copy: Vec<[u8; BARRIER_BYTES]> = (0..n as usize)
            .map(|i| std::ptr::read_unaligned(barriers.cast::<u8>().add(i * BARRIER_BYTES).cast()))
            .collect();
        for barrier in &mut copy {
            resource_args::unwrap_barrier(barrier, |p| pc_pointer(p));
        }
        let noted: Vec<(usize, u32, u32)> = copy
            .iter()
            .filter(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) == 0)
            .map(|b| (u64::from_le_bytes(b[8..16].try_into().unwrap_or([0; 8])) as usize, u32::from_le_bytes([b[16], b[17], b[18], b[19]]), u32::from_le_bytes([b[24], b[25], b[26], b[27]])))
            .collect();
        crate::detile::note_transitions(pc as usize, &noted);
        crate::present::note_list_transitions(pc as usize, &noted);
        method::<Barriers>(pc, 26)(pc, n, copy.as_ptr().cast());
    }
}

/// Console list slot 30 (ResolveQueryData, PC 54): the destination buffer unwrapped.
extern "system" fn list_resolve_query_data(this: *mut Proxy, heap: Object, ty: u32, start: u32, count: u32, dst: Object, dst_off: u64) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        method::<ResolveQueryData>(pc, 54)(pc, unwrap(heap), ty, start, count, unwrap(dst), dst_off);
    }
}

/// Console list slot 31 (SetPredication, PC 55): the buffer unwrapped.
extern "system" fn list_set_predication(this: *mut Proxy, buffer: Object, off: u64, op: u32) {
    // SAFETY: `this` is a live command list proxy; the buffer is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        method::<SetPredication>(pc, 55)(pc, unwrap(buffer), off, op);
    }
}

/// Console list slot 50 (OMSetRenderTargets, PC 46): the render target and depth stencil CPU
/// handles point into title memory (`title_heap`) and are translated to the host heap, as the
/// device's CreateRenderTargetView already does. With `single` set, one handle starts a range.
extern "system" fn list_om_set_render_targets(this: *mut Proxy, n: u32, rts: *const usize, single: i32, dsv: *const usize) {
    // SAFETY: `this` is a live command list proxy; `rts` holds `n` handles (one when `single`), `dsv` one or null.
    unsafe {
        let count = if rts.is_null() { 0 } else if single != 0 { n.min(1) } else { n } as usize;
        let mut host = Vec::with_capacity(count);
        for i in 0..count {
            let Some(t) = crate::title_heap::translate_cpu(*rts.add(i)) else {
                crate::rt_diag::count(&crate::rt_diag::RT_SET_UNTRANSLATED);
                return;
            };
            host.push(t);
        }
        let mut host_dsv = 0usize;
        if !dsv.is_null() {
            let Some(t) = crate::title_heap::translate_cpu(*dsv) else {
                crate::rt_diag::count(&crate::rt_diag::RT_SET_UNTRANSLATED);
                return;
            };
            host_dsv = t;
        }
        crate::rt_diag::bound(this as usize, &host);
        let pc = (*this).pc;
        crate::capture::flush(this as usize, pc);
        crate::capture::bound(this as usize, &host);
        method::<unsafe extern "system" fn(Object, u32, *const usize, i32, *const usize)>(pc, 46)(
            pc,
            n,
            if rts.is_null() { rts } else { host.as_ptr() },
            single,
            if dsv.is_null() { dsv } else { &host_dsv },
        );
    }
}

/// Console list slot 51 (ClearDepthStencilView, PC 47): the CPU handle translated (`title_heap`).
#[allow(clippy::too_many_arguments)]
extern "system" fn list_clear_dsv(this: *mut Proxy, dsv: usize, flags: u32, depth: f32, stencil: u8, n: u32, rects: *const c_void) {
    let Some(host) = crate::title_heap::translate_cpu(dsv) else { return };
    // SAFETY: `this` is a live command list proxy.
    unsafe {
        let pc = (*this).pc;
        method::<unsafe extern "system" fn(Object, usize, u32, f32, u8, u32, *const c_void)>(pc, 47)(pc, host, flags, depth, stencil, n, rects);
    }
}

/// Console list slot 52 (ClearRenderTargetView, PC 48): the CPU handle translated (`title_heap`).
/// Forwarded untranslated, vkd3d read the title memory as its own descriptor and faulted.
extern "system" fn list_clear_rtv(this: *mut Proxy, rtv: usize, color: *const f32, n: u32, rects: *const c_void) {
    let Some(host) = crate::title_heap::translate_cpu(rtv) else {
        crate::rt_diag::count(&crate::rt_diag::RT_CLEAR_UNTRANSLATED);
        return;
    };
    // SAFETY: a non null colour is the caller's four floats.
    crate::rt_diag::cleared(host, if color.is_null() { [0.0; 4] } else { unsafe { std::ptr::read_unaligned(color.cast::<[f32; 4]>()) } });
    // SAFETY: `this` is a live command list proxy.
    unsafe {
        let pc = (*this).pc;
        method::<unsafe extern "system" fn(Object, usize, *const f32, u32, *const c_void)>(pc, 48)(pc, host, color, n, rects);
    }
}

/// Console list slot 53 (ClearUnorderedAccessViewUint, PC 49): the resource unwrapped, the CPU
/// handle translated (`title_heap`).
#[allow(clippy::too_many_arguments)]
extern "system" fn list_clear_uav_uint(this: *mut Proxy, gpu: u64, cpu: u64, res: Object, values: *const c_void, n: u32, rects: *const c_void) {
    clear_uav(this, gpu, cpu, res, values, false, n, rects);
}

/// A clear of an unordered access view (console list slots 53 and 54, PC 49 and 50): the resource
/// unwrapped, the handles translated (`title_heap`, `heap_pages`). A view the layer made raw from a
/// structured or typed one clears with the word its first element holds after the title's clear
/// (`view_raw::clear_words`), a uint clear of the raw view; one that no word gives is left out and
/// logged. The rectangles cover the title's elements in dwords of the PC view
/// (`view_raw::clear_rects`).
#[allow(clippy::too_many_arguments)]
fn clear_uav(this: *mut Proxy, gpu: u64, cpu: u64, res: Object, values: *const c_void, float: bool, n: u32, rects: *const c_void) {
    static REFUSED: AtomicI32 = AtomicI32::new(0);
    let Some(cpu) = crate::title_heap::translate_cpu(cpu as usize) else { return };
    let row = crate::view_shadow::row(cpu).unwrap_or_default();
    let clear = if values.is_null() {
        crate::view_raw::Clear::AsIs
    } else {
        // SAFETY: a clear takes four values, uints or floats.
        let words: [u32; 4] = unsafe { std::ptr::read_unaligned(values.cast()) };
        let value = |w: u32| if float { crate::view_raw::ClearValue::Float(f32::from_bits(w)) } else { crate::view_raw::ClearValue::Uint(w) };
        crate::view_raw::clear_words(row, words.map(value))
    };
    // SAFETY: a clear has `n` rectangles of four words at `rects`.
    let title_rects: Vec<[i32; 4]> = if rects.is_null() || n == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(rects.cast::<[i32; 4]>(), n as usize) }.to_vec() };
    let moved = crate::view_raw::clear_rects(row, &title_rects);
    let (n, rects) = match &moved {
        crate::view_raw::Rects::Moved(r) => (r.len() as u32, r.as_ptr().cast::<c_void>()),
        _ => (n, rects),
    };
    // elements of 1 or 2 bytes whose clear starts or ends inside a dword: no clear of the raw view
    let clear = if moved == crate::view_raw::Rects::Unaligned { crate::view_raw::Clear::Unsupported } else { clear };
    // SAFETY: `this` is a live command list proxy; the resource is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        let gpu = heap_pages::host_handle(this as usize, gpu, || 1);
        match clear {
            crate::view_raw::Clear::AsIs => method::<ClearUavView>(pc, if float { 50 } else { 49 })(pc, gpu, cpu as u64, unwrap(res), values, n, rects),
            crate::view_raw::Clear::Words(w) => {
                let words = [w; 4];
                method::<ClearUavView>(pc, 49)(pc, gpu, cpu as u64, unwrap(res), words.as_ptr().cast(), n, rects);
            }
            crate::view_raw::Clear::Unsupported => {
                if log_due(&REFUSED) {
                    xs_log!("console list slot {}: the clear of a buffer view made raw (row {row:x?}) is no single word, left out", if float { 54 } else { 53 });
                }
            }
        }
    }
}

/// Console list slot 54 (ClearUnorderedAccessViewFloat, PC 50): the resource unwrapped, the CPU
/// handle translated (`title_heap`).
#[allow(clippy::too_many_arguments)]
extern "system" fn list_clear_uav_float(this: *mut Proxy, gpu: u64, cpu: u64, res: Object, values: *const c_void, n: u32, rects: *const c_void) {
    clear_uav(this, gpu, cpu, res, values, true, n, rects);
}

/// Console list slot 55 (DiscardResource, PC 51): the resource unwrapped.
extern "system" fn list_discard_resource(this: *mut Proxy, res: Object, region: *const c_void) {
    // SAFETY: `this` is a live command list proxy; the resource is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        method::<DiscardResource>(pc, 51)(pc, unwrap(res), region);
    }
}

/// Console list slot 25 (SetPipelineState, PC 25): relayed, and noted for the render target probe.
extern "system" fn list_set_pipeline_state(this: *mut Proxy, pso: Object) {
    // SAFETY: `this` is a live command list proxy; pso is a PC pipeline or null.
    unsafe {
        let pc = (*this).pc;
        if crate::pso_diag::bind(this as usize, pso) {
            return;
        }
        let pso = match unwrap(pso) {
            made if made.is_null() && !pso.is_null() => {
                // A pipeline made on first bind that the PC runtime refused: its work is left out.
                crate::pso_diag::bind_missing(this as usize);
                return;
            }
            made => made,
        };
        crate::rt_diag::pipeline_set(this as usize, pso as usize);
        crate::kind_check::bound(this as usize, pso as usize);
        method::<unsafe extern "system" fn(Object, Object)>(pc, 25)(pc, pso);
    }
}

/// Builds the `COMMON -> INDIRECT_ARGUMENT` (before) and `INDIRECT_ARGUMENT -> COMMON` (after)
/// transition barriers for an `ExecuteIndirect` on the argument buffer `arg` and the count buffer
/// `count`. A buffer is left out when it is null, when it may not be transitioned
/// (`*_needs_barrier` false, as an upload heap buffer), and the count buffer also when it is the
/// argument buffer, which already has its barrier.
fn indirect_argument_barriers(arg: u64, arg_needs_barrier: bool, count: u64, count_needs_barrier: bool) -> (Vec<[u8; BARRIER_BYTES]>, Vec<[u8; BARRIER_BYTES]>) {
    const ALL_SUBRESOURCES: u32 = 0xFFFF_FFFF;
    const STATE_COMMON: u32 = 0;
    const STATE_INDIRECT_ARGUMENT: u32 = 0x200;

    // a `D3D12_RESOURCE_BARRIER` of the transition type: resource at 8, subresource at 16, states at 20 and 24
    let transition = |resource: u64, from: u32, to: u32| -> [u8; BARRIER_BYTES] {
        let mut barrier = [0u8; BARRIER_BYTES];
        barrier[8..16].copy_from_slice(&resource.to_le_bytes());
        for (at, word) in [(16, ALL_SUBRESOURCES), (20, from), (24, to)] {
            barrier[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        barrier
    };

    let wanted = [(arg, arg_needs_barrier), (count, count_needs_barrier && count != arg)];
    wanted
        .into_iter()
        .filter(|&(resource, allowed)| resource != 0 && allowed)
        .map(|(resource, _)| (transition(resource, STATE_COMMON, STATE_INDIRECT_ARGUMENT), transition(resource, STATE_INDIRECT_ARGUMENT, STATE_COMMON)))
        .unzip()
}

/// True when `pc_res` is non-null and not on an upload or readback heap (those heaps have fixed
/// `GENERIC_READ` / `COPY_DEST` states and cannot be transitioned).
///
/// # Safety
/// `pc_res` is null or a live PC `ID3D12Resource`.
unsafe fn needs_indirect_barrier(pc_res: Object) -> bool {
    const D3D12_HEAP_TYPE_UPLOAD: u32 = 2;
    const D3D12_HEAP_TYPE_READBACK: u32 = 3;
    type GetHeapProperties = unsafe extern "system" fn(Object, *mut u32, *mut u32) -> HRESULT;

    if pc_res.is_null() {
        return false;
    }
    let mut props = [0u32; 5];
    // SAFETY: `pc_res` is a live PC `ID3D12Resource`; slot 14 is `GetHeapProperties`.
    let hr = unsafe { method::<GetHeapProperties>(pc_res, 14)(pc_res, props.as_mut_ptr(), std::ptr::null_mut()) };
    hr >= 0 && !matches!(props[0], D3D12_HEAP_TYPE_UPLOAD | D3D12_HEAP_TYPE_READBACK)
}

/// Console list slot 59 (ExecuteIndirect, PC 59): the argument and count buffers unwrapped, with
/// `COMMON -> INDIRECT_ARGUMENT` / `INDIRECT_ARGUMENT -> COMMON` barriers around the call on
/// non-bundle lists for non-upload buffers.
extern "system" fn list_execute_indirect(this: *mut Proxy, sig: Object, max: u32, arg: Object, arg_off: u64, count: Object, count_off: u64) {
    // SAFETY: `this` is a live command list proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        // The signature decides what runs, so both pipeline kinds get their mirror.
        if crate::pso_diag::skip(this as usize) {
            return;
        }
        list_mirror::prepare(this, crate::user_mirror::Bind::Graphics);
        list_mirror::prepare(this, crate::user_mirror::Bind::Compute);
        crate::rt_diag::count(&crate::rt_diag::INDIRECT);
        let pc_arg = unwrap(arg);
        let pc_count = unwrap(count);
        type GetListType = unsafe extern "system" fn(Object) -> u32;
        const D3D12_COMMAND_LIST_TYPE_BUNDLE: u32 = 1;
        let record_barriers = method::<GetListType>(pc, 8)(pc) != D3D12_COMMAND_LIST_TYPE_BUNDLE;
        let (before, after) = if record_barriers {
            indirect_argument_barriers(pc_arg as u64, needs_indirect_barrier(pc_arg), pc_count as u64, needs_indirect_barrier(pc_count))
        } else {
            (Vec::new(), Vec::new())
        };
        if !before.is_empty() {
            method::<Barriers>(pc, 26)(pc, before.len() as u32, before.as_ptr().cast());
        }
        method::<ExecuteIndirect>(pc, 59)(pc, unwrap(sig), max, pc_arg, arg_off, pc_count, count_off);
        if !after.is_empty() {
            method::<Barriers>(pc, 26)(pc, after.len() as u32, after.as_ptr().cast());
        }
    }
}

/// Console queue slot 8 (UpdateTileMappings, PC 8): the tiled resource unwrapped.
#[allow(clippy::too_many_arguments)]
extern "system" fn queue_update_tile_mappings(this: *mut Proxy, res: Object, n_regions: u32, coords: *const c_void, sizes: *const c_void, heap: Object, n_ranges: u32, flags: *const c_void, offsets: *const c_void, counts: *const c_void, mode: u32) {
    // SAFETY: `this` is a live queue proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        method::<UpdateTileMappings>(pc, 8)(pc, unwrap(res), n_regions, coords, sizes, unwrap(heap), n_ranges, flags, offsets, counts, mode);
    }
}

/// Console queue slot 9 (CopyTileMappings, PC 9): both resources unwrapped.
extern "system" fn queue_copy_tile_mappings(this: *mut Proxy, dst: Object, dst_start: *const c_void, src: Object, src_start: *const c_void, size: *const c_void, flags: u32) {
    // SAFETY: `this` is a live queue proxy; the objects are proxies or PC objects.
    unsafe {
        let pc = (*this).pc;
        method::<CopyTileMappings>(pc, 9)(pc, unwrap(dst), dst_start, unwrap(src), src_start, size, flags);
    }
}

/// Console device slot 31 (CreateSharedHandle, PC 31): the object unwrapped.
extern "system" fn dev_create_shared_handle(this: *mut Proxy, object: Object, attributes: *const c_void, access: u32, name: *const c_void, handle: *mut *mut c_void) -> HRESULT {
    // SAFETY: `this` is a live device proxy; `object` is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        method::<CreateSharedHandle>(pc, 31)(pc, unwrap(object), attributes, access, name, handle)
    }
}

/// Console device slot 42 (GetResourceTiling, PC 42): the tiled resource unwrapped.
#[allow(clippy::too_many_arguments)]
extern "system" fn dev_get_resource_tiling(this: *mut Proxy, res: Object, n_tiles: *mut u32, packed: *mut c_void, std_shape: *mut c_void, n_sub: *mut u32, first: u32, tilings: *mut c_void) {
    // SAFETY: `this` is a live device proxy; `res` is a proxy or a PC object.
    unsafe {
        let pc = (*this).pc;
        method::<GetResourceTiling>(pc, 42)(pc, unwrap(res), n_tiles, packed, std_shape, n_sub, first, tilings);
    }
}

// ---- vtable construction ----

/// The method behind a `Custom` rule, as a code address.
fn custom(kind: Kind, slot: u32) -> Option<usize> {
    match slot {
        0 => return Some(query_interface as *const () as usize),
        1 => return Some(add_ref as *const () as usize),
        2 => return Some(release as *const () as usize),
        _ => {}
    }
    Some(match (kind, slot) {
        (Kind::Device, 8) => dev_create_queue as *const () as usize,
        (Kind::Device, 9) => dev_create_allocator as *const () as usize,
        (Kind::Device, 11) => dev_create_compute_pipeline_state as *const () as usize,
        (Kind::Device, 12) => dev_create_list as *const () as usize,
        (Kind::Device, 14) => dev_create_heap as *const () as usize,
        (Kind::Device, 16) => dev_create_root_signature as *const () as usize,
        (Kind::Device, 17) => dev_create_cbv as *const () as usize,
        (Kind::Device, 18) => dev_create_srv as *const () as usize,
        (Kind::Device, 19) => dev_create_uav as *const () as usize,
        (Kind::Device, 20) => dev_create_rtv as *const () as usize,
        (Kind::Device, 21) => dev_create_dsv as *const () as usize,
        (Kind::Device, 22) => dev_create_sampler as *const () as usize,
        (Kind::Device, 23) => dev_copy_descriptors as *const () as usize,
        (Kind::Device, 24) => dev_copy_descriptors_simple as *const () as usize,
        (Kind::Device, 25) => dev_alloc_info as *const () as usize,
        (Kind::Device, 26) => dev_heap_props as *const () as usize,
        (Kind::Device, 27) => dev_create_committed_resource as *const () as usize,
        (Kind::Device, 29) => dev_create_placed_resource as *const () as usize,
        (Kind::Device, 31) => dev_create_shared_handle as *const () as usize,
        (Kind::Device, 41) => dev_create_command_signature as *const () as usize,
        (Kind::Device, 42) => dev_get_resource_tiling as *const () as usize,
        (Kind::Device, 61) => dev_create_sampler_x as *const () as usize,
        (Kind::Device, 108) => dev_alloc_info2 as *const () as usize,
        (Kind::Device, 127) => dev_create_committed_resource_layout as *const () as usize,
        (Kind::Device, 133) => dev_placed_at_va_layout as *const () as usize,
        (Kind::Device, 34) => dev_make_resident as *const () as usize,
        (Kind::Device, 35) => dev_evict as *const () as usize,
        (Kind::Device, 44) => dev_placed_at_va as *const () as usize,
        (Kind::Device, 48) => dev_create_queue_x as *const () as usize,
        (Kind::Device, 55) => dev_derived_pso as *const () as usize,
        (Kind::Device, 60) => dev_graphics_pso as *const () as usize,
        (Kind::Device, 62) => dev_compute_pso as *const () as usize,
        (Kind::Device, 67) => pso_cache::dev_deserialize_compute as *const () as usize,
        (Kind::Device, 70) => dev_create_srv_at as *const () as usize,
        (Kind::Device, 71) => dev_create_uav_at as *const () as usize,
        (Kind::Device, 73) => dev_create_list_x as *const () as usize,
        (Kind::Device, 74) => dev_register_page_pool as *const () as usize,
        (Kind::Device, 75) => dev_unregister_page_pool as *const () as usize,
        (Kind::Device, 78) => dev_set_frame_interval as *const () as usize,
        (Kind::Device, 79) => dev_schedule_frame_event as *const () as usize,
        (Kind::Device, 80) => dev_wait_frame_event as *const () as usize,
        (Kind::Device, 81) => dev_frame_statistics as *const () as usize,
        (Kind::Device, 86) => dev_create_pipeline_state_stream as *const () as usize,
        (Kind::List | Kind::DmaList, 9) => list_close as *const () as usize,
        (Kind::List, 10) => list_reset as *const () as usize,
        (Kind::List, 15) => list_copy_buffer_region as *const () as usize,
        (Kind::List, 16) => list_copy_texture_region as *const () as usize,
        (Kind::List, 17) => list_copy_resource as *const () as usize,
        (Kind::List, 18) => list_copy_tiles as *const () as usize,
        (Kind::List, 19) => list_resolve_subresource as *const () as usize,
        (Kind::List, 25) => list_set_pipeline_state as *const () as usize,
        (Kind::List, 26) => list_resource_barrier as *const () as usize,
        (Kind::List, 30) => list_resolve_query_data as *const () as usize,
        (Kind::List, 31) => list_set_predication as *const () as usize,
        (Kind::List, 50) => list_om_set_render_targets as *const () as usize,
        (Kind::List, 51) => list_clear_dsv as *const () as usize,
        (Kind::List, 52) => list_clear_rtv as *const () as usize,
        (Kind::List, 53) => list_clear_uav_uint as *const () as usize,
        (Kind::List, 54) => list_clear_uav_float as *const () as usize,
        (Kind::List, 55) => list_discard_resource as *const () as usize,
        (Kind::List, 59) => list_draws::list_slot59 as *const () as usize,
        (Kind::List, 61) => list_draws::list_execute_indirect_x as *const () as usize,
        (Kind::List, 80) => list_draws::list_dispatch_x as *const () as usize,
        (Kind::List, 75) => list_misc::list_depth_order_x as *const () as usize,
        (Kind::List, 87) => list_misc::list_set_predication_buffer_x as *const () as usize,
        (Kind::List, 90) => list_misc::list_write32_top as *const () as usize,
        (Kind::List, 91) => list_misc::list_write64_top as *const () as usize,
        (Kind::List, 92) => list_misc::list_write32_bottom_of_shader as *const () as usize,
        (Kind::List, 95) => list_misc::list_kickoff_x as *const () as usize,
        (Kind::List, 96) => list_misc::list_set_depth_bias_x as *const () as usize,
        (Kind::List, 126) => list_misc::list_shading_rate_x as *const () as usize,
        (Kind::List, 131) => list_misc::list_sync_top as *const () as usize,
        (Kind::List, 132) => list_misc::list_sync_bottom as *const () as usize,
        (Kind::List, 142) => list_misc::list_state_142 as *const () as usize,
        (Kind::List, 100) => mesh_pipeline::list_dispatch_mesh_x as *const () as usize,
        (Kind::List, 127) => mesh_pipeline::list_slot127 as *const () as usize,
        (Kind::List, 129) => mesh_pipeline::list_dispatch_mesh_newer as *const () as usize,
        (Kind::List, 62) => list_mem::list_copy_memory_x as *const () as usize,
        (Kind::List, 63) => list_mem::list_fill32_x as *const () as usize,
        (Kind::List, 64) => list_mem::list_fill64_x as *const () as usize,
        (Kind::List, 65) => list_mem::list_fill128_x as *const () as usize,
        (Kind::List, 20) => list_ia_set_primitive_topology as *const () as usize,
        (Kind::List, 21) => list_raster::list_rs_set_viewports as *const () as usize,
        (Kind::List, 22) => list_raster::list_rs_set_scissor_rects as *const () as usize,
        (Kind::List, 27) => list_execute_bundle as *const () as usize,
        (Kind::List, 32) => list_set_heaps as *const () as usize,
        (Kind::List, 67) => list_flush_texture_cache as *const () as usize,
        (Kind::List, 70) => list_set_depth_bounds_x as *const () as usize,
        (Kind::List, 71) => list_write32_bottom as *const () as usize,
        (Kind::List, 72) => list_write64_bottom as *const () as usize,
        (Kind::List, 73) => list_wait32 as *const () as usize,
        (Kind::List, 74) => list_wait64 as *const () as usize,
        (Kind::List, 146) => list_uav_barrier as *const () as usize,
        (Kind::List, 83) => list_draws::list_slot83 as *const () as usize,
        (Kind::List, 85) => list_copy_texture_region_x as *const () as usize,
        (Kind::List, 86) => list_copy_resource_x as *const () as usize,
        (Kind::List, 139) => list_reset_late as *const () as usize,
        (Kind::DmaList, 10) => list_reset as *const () as usize,
        (Kind::DmaList, 11) => list_copy_buffer_region as *const () as usize,
        (Kind::DmaList, 12) => list_copy_texture_region as *const () as usize,
        (Kind::DmaList, 13) => list_copy_resource as *const () as usize,
        (Kind::DmaList, 14) => list_copy_tiles as *const () as usize,
        (Kind::DmaList, 15..) => list_gaps::dma_list_ok as *const () as usize,
        (Kind::List, 12) => list_mirror::list_draw_instanced as *const () as usize,
        (Kind::List, 13) => list_mirror::list_draw_indexed_instanced as *const () as usize,
        (Kind::List, 14) => list_mirror::list_dispatch as *const () as usize,
        (Kind::List, 33) => list_mirror::list_set_compute_root_signature as *const () as usize,
        (Kind::List, 34) => list_mirror::list_set_graphics_root_signature as *const () as usize,
        (Kind::List, 35) => list_mirror::list_set_compute_root_table as *const () as usize,
        (Kind::List, 36) => list_mirror::list_set_graphics_root_table as *const () as usize,
        (Kind::List, 37) => list_mirror::list_set_compute_root_constant as *const () as usize,
        (Kind::List, 38) => list_mirror::list_set_graphics_root_constant as *const () as usize,
        (Kind::List, 39) => list_mirror::list_set_compute_root_constants as *const () as usize,
        (Kind::List, 40) => list_mirror::list_set_graphics_root_constants as *const () as usize,
        (Kind::List, 41) => list_gaps::list_set_compute_root_cbv as *const () as usize,
        (Kind::List, 42) => list_gaps::list_set_graphics_root_cbv as *const () as usize,
        (Kind::List, 43) => list_gaps::list_set_compute_root_srv as *const () as usize,
        (Kind::List, 44) => list_gaps::list_set_graphics_root_srv as *const () as usize,
        (Kind::List, 45) => list_gaps::list_set_compute_root_uav as *const () as usize,
        (Kind::List, 46) => list_gaps::list_set_graphics_root_uav as *const () as usize,
        (Kind::List, 47) => list_gaps::list_ia_set_index_buffer as *const () as usize,
        (Kind::List, 48) => list_gaps::list_ia_set_vertex_buffers as *const () as usize,
        (Kind::List, 148) => list_gaps::list_allocate_upload_memory as *const () as usize,
        (Kind::Allocator, 8) => list_gaps::allocator_reset as *const () as usize,
        (Kind::Resource, 8) => res_map as *const () as usize,
        (Kind::Resource, 9) => res_unmap as *const () as usize,
        (Kind::Resource, 11) => res_gpu_virtual_address as *const () as usize,
        (Kind::Queue, 8) => queue_update_tile_mappings as *const () as usize,
        (Kind::Queue, 9) => queue_copy_tile_mappings as *const () as usize,
        (Kind::Queue, 10) => queue_execute as *const () as usize,
        (Kind::Device, 36) => dev_create_fence as *const () as usize,
        (Kind::Device, 15) => dev_handle_increment as *const () as usize,
        (Kind::Device, 10) => dev_graphics_pso_pc as *const () as usize,
        (Kind::Device, 52) => dev_set_debug_flags as *const () as usize,
        (Kind::Device, 56 | 66) => dev_serialize_pso as *const () as usize,
        (Kind::Device, 57) => pso_cache::dev_deserialize_graphics as *const () as usize,
        (Kind::Device, 76) => dev_create_committed_resource as *const () as usize,
        (Kind::Fence, 10) => fence_signal as *const () as usize,
        (Kind::Fence, 12) => fence_value_address as *const () as usize,
        (Kind::Queue, 14) => queue_slot14 as *const () as usize,
        (Kind::Queue, 15) => queue_slot15 as *const () as usize,
        (Kind::Queue, 16) => queue_slot16 as *const () as usize,
        (Kind::Queue, 17) => queue_slot17 as *const () as usize,
        (Kind::Queue, 18) => queue_slot18 as *const () as usize,
        (Kind::Queue, 19) => queue_slot19 as *const () as usize,
        (Kind::Queue, s) if s >= 20 => return queue_late_slot(s),
        (Kind::Allocator, 9) => allocator_set_callbacks as *const () as usize,
        (Kind::Heap, 9) => heap_cpu_start as *const () as usize,
        (Kind::Heap, 10) => heap_gpu_start as *const () as usize,
        _ => return None,
    })
}

/// Bytes of code per slot.
const STUB: usize = 32;

/// Builds the vtables once. A rule marked `Custom` without a method is logged and left to the
/// unknown slot stub.
#[cfg(target_arch = "x86_64")]
fn build() -> Option<Tables> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let total = (usize::from(DEVICE_SLOTS) + usize::from(LIST_SLOTS) + usize::from(QUEUE_SLOTS) + usize::from(HEAP_SLOTS) + usize::from(ALLOCATOR_SLOTS) + usize::from(RESOURCE_SLOTS) + usize::from(crate::abi::DMA_LIST_SLOTS)) * STUB;
    // SAFETY: a fresh anonymous allocation, written below and then made executable.
    let code = unsafe { VirtualAlloc(std::ptr::null(), total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) }.cast::<u8>();
    if code.is_null() {
        // SAFETY: no precondition.
        xs_log!("console proxies: VirtualAlloc for the stubs failed ({})", unsafe { windows_sys::Win32::Foundation::GetLastError() });
        return None;
    }
    let mut at = 0usize;
    let vtbls = Kind::ALL.map(|kind| {
        let mut table = vec![0usize; usize::from(kind.slots())];
        for (slot, entry) in table.iter_mut().enumerate() {
            let rule = kind.rule(slot as u32);
            let method = match rule {
                Some(r) if r.kind == RuleKind::Custom => custom(kind, slot as u32),
                Some(r) if r.kind == RuleKind::Unknown && kind == Kind::List => list_probe(slot as u32),
                _ => None,
            };
            if let Some(method) = method {
                *entry = method;
                continue;
            }
            if matches!(rule, Some(r) if r.kind == RuleKind::Custom) {
                xs_log!("console proxies: {} slot {slot} is marked custom but has no method", kind.name());
            }
            let mut stub = [0u8; STUB];
            match rule {
                Some(r) if r.kind == RuleKind::Fwd => {
                    let disp = (u32::from(r.pc) + (slot as u32 - u32::from(r.first))) * 8;
                    // mov rcx,[rcx+8] then mov rax,[rcx] then jmp [rax+disp32]
                    let fwd = [0x48, 0x8B, 0x49, 0x08, 0x48, 0x8B, 0x01, 0xFF, 0xA0];
                    stub[..fwd.len()].copy_from_slice(&fwd);
                    stub[fwd.len()..fwd.len() + 4].copy_from_slice(&disp.to_le_bytes());
                }
                _ => {
                    let code_value = (kind as u32) << 16 | slot as u32;
                    let target = unknown_method as *const () as usize as u64;
                    // mov edx,imm32 then mov rax,imm64 then jmp rax
                    stub[0] = 0xBA;
                    stub[1..5].copy_from_slice(&code_value.to_le_bytes());
                    stub[5] = 0x48;
                    stub[6] = 0xB8;
                    stub[7..15].copy_from_slice(&target.to_le_bytes());
                    stub[15] = 0xFF;
                    stub[16] = 0xE0;
                }
            }
            // SAFETY: `at + STUB` stays inside the `total` bytes allocated above.
            unsafe { std::ptr::copy_nonoverlapping(stub.as_ptr(), code.add(at), STUB) };
            *entry = code as usize + at;
            at += STUB;
        }
        table.into_boxed_slice()
    });
    let mut old = 0u32;
    // SAFETY: `code` and `total` are the allocation above, written completely.
    unsafe {
        VirtualProtect(code.cast(), total, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), code.cast(), total);
    }
    Some(Tables { vtbls })
}

#[cfg(not(target_arch = "x86_64"))]
fn build() -> Option<Tables> {
    xs_log!("console proxies: x86-64 only");
    None
}

/// Wraps a PC object (one reference, taken over) in a console proxy. Null when the proxies are
/// unavailable.
pub(crate) fn wrap(pc: Object, kind: Kind) -> *mut c_void {
    let Some(tables) = TABLES.get_or_init(build) else {
        return std::ptr::null_mut();
    };
    if pc.is_null() {
        return std::ptr::null_mut();
    }
    let layout = layout_of(kind as i32);
    // SAFETY: the layout has a non zero size, and the zeroed block is initialised below.
    unsafe {
        let this = alloc_zeroed(layout).cast::<Proxy>();
        if this.is_null() {
            return std::ptr::null_mut();
        }
        this.write(Proxy { vtbl: tables.vtbls[kind as usize].as_ptr(), pc, refs: AtomicI32::new(1), kind: kind as i32 });
        if kind == Kind::List {
            // Cursor (+0x18) == dispatch limit (+0x28) == root argument limit (+0x30): no inline
            // writes, every command reaches the vtable. They point at scratch space in the object.
            let base = this.cast::<u8>();
            let scratch = base.add(0x780);
            for field in [0x18, 0x28, 0x30] {
                base.add(field).cast::<*mut u8>().write_unaligned(scratch);
            }
        }
        this.cast()
    }
}

/// Releases a PC object the layer holds one reference to.
///
/// # Safety
/// `object` is a live COM object and the caller owns one of its references.
pub(crate) unsafe fn release_object(object: Object) {
    // SAFETY: by the contract above.
    unsafe { method::<Release>(object, 2)(object) };
}

/// `QueryInterface` on a PC object.
///
/// # Safety
/// `object` is a live COM object, `riid` readable and `out` writable.
pub(crate) unsafe fn query_object(object: Object, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe { method::<Query>(object, 0)(object, riid, out) }
}

#[cfg(test)]
mod tests {
    use super::host_queue_flags;

    #[test]
    fn create_srv_at_makes_a_null_view_for_a_3d_view_over_a_2d_texture() {
        use super::{Object, Proxy, dev_create_view_at};
        use std::{ffi::c_void, ptr, sync::atomic::AtomicI32};

        unsafe extern "system" fn texture_desc(_: Object, out: *mut c_void) -> *mut c_void {
            let mut desc = [0u8; 0x38];
            desc[0] = 3; // D3D12_RESOURCE_DIMENSION_TEXTURE2D
            desc[0x20] = 10; // DXGI_FORMAT_R16G16B16A16_FLOAT
            // SAFETY: GetDesc provides a writable resource description.
            unsafe { out.cast::<[u8; 0x38]>().write_unaligned(desc) };
            out
        }

        unsafe extern "system" fn capture_srv(_: Object, res: Object, desc: *const c_void, cpu: usize) {
            // SAFETY: the test passes a writable capture as its CPU handle and a readable SRV description.
            unsafe { (cpu as *mut Option<(Object, [u8; 8])>).write(Some((res, desc.cast::<[u8; 8]>().read_unaligned()))) };
        }

        let mut resource_vtable = [0usize; 11];
        resource_vtable[10] = texture_desc as *const () as usize;
        let mut resource = resource_vtable.as_ptr();
        let resource = ptr::from_mut(&mut resource).cast::<c_void>();
        let mut device_vtable = [0usize; 19];
        device_vtable[18] = capture_srv as *const () as usize;
        let mut device = device_vtable.as_ptr();
        let mut proxy = Proxy {
            vtbl: ptr::null(),
            pc: ptr::from_mut(&mut device).cast(),
            refs: AtomicI32::new(1),
            kind: super::Kind::Device as i32,
        };
        let va = 0x1234_0000_u64;
        let mut view = [0u8; 0x30];
        view[0] = 10;
        view[4] = 8; // D3D12_SRV_DIMENSION_TEXTURE3D
        view[8..12].copy_from_slice(&0x1688_u32.to_le_bytes());
        view[0x14] = 1;
        view[0x28..0x30].copy_from_slice(&va.to_le_bytes());
        let mut captured: Option<(Object, [u8; 8])> = None;
        let cpu = ptr::from_mut(&mut captured) as usize;
        crate::views_at::record_resource(va, 1, resource as usize, false);
        // SAFETY: the mock objects and descriptor storage live through the call.
        unsafe { dev_create_view_at(&mut proxy, ptr::null(), view.as_ptr(), cpu, false) };
        crate::views_at::forget_resource(resource as usize);
        crate::view_shadow::owner_made(cpu, 0);
        assert_eq!(captured, Some((ptr::null_mut(), [28, 0, 0, 0, 8, 0, 0, 0])));
    }

    #[test]
    fn barrier_layout_maps_to_expected_resource_states() {
        assert_eq!(super::barrier_layout_to_state(0), 0);
        assert_eq!(super::barrier_layout_to_state(1), 0xAC3);
        assert_eq!(super::barrier_layout_to_state(2), 0x4);
        assert_eq!(super::barrier_layout_to_state(3), 0x8);
        assert_eq!(super::barrier_layout_to_state(4), 0x10);
        assert_eq!(super::barrier_layout_to_state(5), 0x20);
        assert_eq!(super::barrier_layout_to_state(6), 0xC0);
        assert_eq!(super::barrier_layout_to_state(7), 0x800);
        assert_eq!(super::barrier_layout_to_state(8), 0x400);
        assert_eq!(super::barrier_layout_to_state(9), 0x2000);
        assert_eq!(super::barrier_layout_to_state(10), 0x1000);
        assert_eq!(super::barrier_layout_to_state(0xFFFF_FFFF), 0);
    }

    #[test]
    fn console_queue_flags_are_masked_to_the_host_ones() {
        assert_eq!(host_queue_flags(0), 0);
        assert_eq!(host_queue_flags(1), 1);
        assert_eq!(host_queue_flags(2), 0);
        assert_eq!(host_queue_flags(3), 1);
    }

    #[test]
    fn execute_indirect_barriers_emit_for_non_upload_and_skip_upload_or_null() {
        use super::BARRIER_BYTES;
        let u32_at = |b: &[u8; BARRIER_BYTES], at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let u64_at = |b: &[u8; BARRIER_BYTES], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());

        // arg non-upload, count null -> 1 before (COMMON -> INDIRECT_ARGUMENT), 1 after (INDIRECT_ARGUMENT -> COMMON)
        let (before, after) = super::indirect_argument_barriers(0x1000, true, 0, false);
        assert_eq!(before.len(), 1);
        assert_eq!(after.len(), 1);
        assert_eq!(u32_at(&before[0], 0), 0);
        assert_eq!(u64_at(&before[0], 8), 0x1000);
        assert_eq!(u32_at(&before[0], 16), 0xFFFF_FFFF);
        assert_eq!(u32_at(&before[0], 20), 0);
        assert_eq!(u32_at(&before[0], 24), 0x200);
        assert_eq!(u32_at(&after[0], 20), 0x200);
        assert_eq!(u32_at(&after[0], 24), 0);

        // arg + distinct count both non-upload -> 2 barriers each
        let (before2, after2) = super::indirect_argument_barriers(0x1000, true, 0x2000, true);
        assert_eq!(before2.len(), 2);
        assert_eq!(after2.len(), 2);
        assert_eq!(u64_at(&before2[1], 8), 0x2000);
        assert_eq!(u64_at(&after2[1], 8), 0x2000);

        // arg == count -> deduplicated to 1 barrier each
        let (before_dup, after_dup) = super::indirect_argument_barriers(0x1000, true, 0x1000, true);
        assert_eq!(before_dup.len(), 1);
        assert_eq!(after_dup.len(), 1);

        // upload-heap arg skipped, non-upload count still barriered
        let (before_up, after_up) = super::indirect_argument_barriers(0x1000, false, 0x2000, true);
        assert_eq!(before_up.len(), 1);
        assert_eq!(after_up.len(), 1);
        assert_eq!(u64_at(&before_up[0], 8), 0x2000);

        // both upload-heap -> no barriers
        let (before_none, after_none) = super::indirect_argument_barriers(0x1000, false, 0x2000, false);
        assert!(before_none.is_empty());
        assert!(after_none.is_empty());
    }

    #[test]
    fn view_flags_follow_what_each_kind_of_view_needs() {
        use super::{ViewNeeds, view_flags_allow};
        assert!(view_flags_allow(ViewNeeds::ShaderResource, 0, 28));
        assert!(!view_flags_allow(ViewNeeds::ShaderResource, 0x8, 28));
        assert!(view_flags_allow(ViewNeeds::UnorderedAccess, 0x4, 28));
        assert!(!view_flags_allow(ViewNeeds::UnorderedAccess, 0, 28));
        assert!(!view_flags_allow(ViewNeeds::UnorderedAccess, 0x4, 85));
        assert!(view_flags_allow(ViewNeeds::RenderTarget, 0x1, 28));
        assert!(!view_flags_allow(ViewNeeds::RenderTarget, 0x1, 115));
        assert!(view_flags_allow(ViewNeeds::DepthStencil, 0x2, 45));
        assert!(!view_flags_allow(ViewNeeds::DepthStencil, 0x1, 45));
    }

    #[test]
    fn a_view_dimension_needs_the_matching_resource_dimension() {
        use super::resource_dimension_of_view;
        assert_eq!(resource_dimension_of_view(false, 1), Some(1));
        assert_eq!(resource_dimension_of_view(false, 11), Some(1));
        assert_eq!(resource_dimension_of_view(true, 11), None);
        assert_eq!(resource_dimension_of_view(true, 3), Some(2));
        assert_eq!(resource_dimension_of_view(false, 10), Some(3));
        assert_eq!(resource_dimension_of_view(true, 8), Some(4));
        assert_eq!(resource_dimension_of_view(false, 0), None);
    }

    #[test]
    fn a_buffer_view_moves_into_its_buffer_by_whole_elements() {
        use super::shift_buffer_view;
        // an unordered access view of 16 byte elements: FirstElement 4 at 8, NumElements 8 at 0x10, stride 16 at 0x14
        let mut view = [0u8; 0x28];
        view[0x08] = 4;
        view[0x10] = 8;
        view[0x14] = 16;
        assert!(shift_buffer_view(&mut view, true, 0x40, 0));
        assert_eq!(u64::from_le_bytes(view[8..16].try_into().unwrap()), 8);
        let before = view;
        assert!(!shift_buffer_view(&mut view, true, 0x41, 0));
        assert_eq!(view, before);
        // past 2^32 raw elements of 4 bytes
        assert!(!shift_buffer_view(&mut view, true, 0x40_0000_0000, 0));
    }

    #[cfg(windows)]
    #[test]
    fn execute_indirect_barrier_layout_matches_d3d12() {
        use windows::Win32::Graphics::Direct3D12::*;
        let (before, _) = super::indirect_argument_barriers(0x1234, true, 0, false);
        let b = &before[0];
        assert_eq!(u32::from_le_bytes(b[0..4].try_into().unwrap()), D3D12_RESOURCE_BARRIER_TYPE_TRANSITION.0 as u32);
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
        assert_eq!(u32::from_le_bytes(b[20..24].try_into().unwrap()), D3D12_RESOURCE_STATE_COMMON.0 as u32);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT.0 as u32);
    }
}
