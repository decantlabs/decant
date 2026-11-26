//! The exports of `d3d12_xs.dll` and the calls into the PC D3D12 behind them. Every function keeps
//! the name, the `WINAPI` convention and the arguments of the C version. Callers pass raw
//! pointers and COM objects, so the code here is `unsafe`, and each block states what it relies on.
//!
//! The PC D3D12 is whatever `d3d12.dll` resolves to: vkd3d-proton (Proton/Wine prefix, or an app
//! local copy on Windows) or Windows' own. Nothing here talks to Vulkan directly.
#![allow(unsafe_code, non_snake_case, non_upper_case_globals)]

use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};

use windows_sys::Win32::Foundation::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, E_OUTOFMEMORY, E_POINTER, HMODULE, S_OK};
use windows_sys::Win32::System::LibraryLoader::{DisableThreadLibraryCalls, GetProcAddress, LoadLibraryW};
use windows_sys::core::{GUID, HRESULT};

use crate::abi::Kind;
use crate::log::{guid_at, p, xs_log};
use crate::proxy::{PRESENT_HOOK, PresentFn, query_object, release_object, wrap};
use crate::rts0::{PUBLIC_FLAGS, flags_offset, rd32, strip_unknown_flags};
use crate::{FORWARDER_DIRECTIVES, bytes};

/// The forwarder, as an option the linker reads from the `.drectve` section of the object file, as
/// it read the `/export:D3D12GetDebugInterface=d3d12.D3D12GetDebugInterface` line of the C version.
/// Neither link.exe nor rustc's own module definition file can express a forwarder for a cdylib
/// otherwise.
// SAFETY: `.drectve` holds linker options only and is not part of the image, the bytes are ASCII.
#[used]
#[unsafe(link_section = ".drectve")]
static FORWARDERS: [u8; FORWARDER_DIRECTIVES.len()] = bytes(FORWARDER_DIRECTIVES);

/// Public IID of `ID3D12Device`, `{189819f1-1db6-4b57-be54-1821339b85f7}`, exported as data under its
/// public name: the console `d3d12_xs.dll` exports the PC IIDs this way and titles import them.
#[unsafe(no_mangle)]
pub static IID_ID3D12Device: GUID = GUID::from_u128(0x189819f1_1db6_4b57_be54_1821339b85f7);

const XS_FEATURE_LEVEL_12_0: u32 = 0xc000;
const XS_DXGI_ERROR_UNSUPPORTED: HRESULT = 0x887A_0004_u32 as HRESULT;

// ---- PC D3D12 ----

type SerializeFn = unsafe extern "system" fn(*const RsDesc, u32, *mut *mut c_void, *mut *mut c_void) -> HRESULT;
type SerializeVersionedFn = unsafe extern "system" fn(*const VersionedRsDesc, *mut *mut c_void, *mut *mut c_void) -> HRESULT;
type DeserializerFn = unsafe extern "system" fn(*const c_void, usize, *const GUID, *mut *mut c_void) -> HRESULT;
type CreateDeviceFn = unsafe extern "system" fn(*mut c_void, u32, *const GUID, *mut *mut c_void) -> HRESULT;

/// A function of `d3d12.dll` through the normal search order: an app local vkd3d-proton wins over
/// the system copy, as for the forwarded exports.
///
/// # Safety
/// `F` is the function pointer type of the export called `name`.
unsafe fn pc_d3d12<F: Copy>(name: &CStr) -> Option<F> {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    static MODULE: AtomicUsize = AtomicUsize::new(0);
    let mut module = MODULE.load(Ordering::SeqCst);
    if module == 0 {
        // SAFETY: a zero terminated wide string literal.
        module = unsafe { LoadLibraryW(windows_sys::w!("d3d12.dll")) } as usize;
        MODULE.store(module, Ordering::SeqCst);
    }
    // SAFETY: a loaded module and a zero terminated name.
    let function = if module == 0 { None } else { unsafe { GetProcAddress(module as HMODULE, name.as_ptr().cast()) } };
    match function {
        // SAFETY: by the contract above, `F` is that function's type.
        Some(function) => Some(unsafe { std::mem::transmute_copy::<unsafe extern "system" fn() -> isize, F>(&function) }),
        None => {
            xs_log!("PC d3d12.dll has no {} (module {})", name.to_string_lossy(), p(module as *const c_void));
            decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("{} (no PC d3d12.dll export)", name.to_string_lossy()));
            None
        }
    }
}

/// Same shape as `D3D12_ROOT_SIGNATURE_DESC`, `_DESC1` and `_DESC2` in the public d3d12.h: they only
/// differ in the pointee types, so `Flags` sits at the same place in all three.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsDesc {
    pub num_parameters: u32,
    pub parameters: *const c_void,
    pub num_static_samplers: u32,
    pub static_samplers: *const c_void,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VersionedRsDesc {
    /// `D3D_ROOT_SIGNATURE_VERSION`
    pub version: u32,
    pub desc: RsDesc,
}

/// Console only flag values already reported, so a title calling in a loop logs each once.
static REPORTED_FLAGS: [AtomicU32; 8] = [const { AtomicU32::new(0) }; 8];

fn report_flags(place: &str, extra: u32) {
    for slot in &REPORTED_FLAGS {
        match slot.compare_exchange(0, extra, Ordering::SeqCst, Ordering::SeqCst) {
            Err(seen) if seen == extra => return,
            Err(_) => {}
            Ok(_) => break,
        }
    }
    xs_log!("{place}: cleared console-only root signature flags 0x{extra:08X} (unknown to PC D3D12)");
}

// ---- root signature helpers ----

/// The PC serializer, handed a copy of the description without the console only flag bits.
///
/// # Safety
/// The arguments are those of `D3D12SerializeRootSignature`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12SerializeRootSignature(desc: *const RsDesc, version: u32, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    // SAFETY: the export is the one named, and `desc` is null or readable by the caller's contract.
    unsafe {
        let Some(real) = pc_d3d12::<SerializeFn>(c"D3D12SerializeRootSignature") else {
            return E_NOTIMPL;
        };
        if !desc.is_null() && (*desc).flags & !PUBLIC_FLAGS != 0 {
            let mut copy = *desc;
            report_flags("D3D12SerializeRootSignature", copy.flags & !PUBLIC_FLAGS);
            copy.flags &= PUBLIC_FLAGS;
            return real(&copy, version, blob, err);
        }
        real(desc, version, blob, err)
    }
}

/// The versioned form of [`D3D12SerializeRootSignature`].
///
/// # Safety
/// The arguments are those of `D3D12SerializeVersionedRootSignature`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12SerializeVersionedRootSignature(desc: *const VersionedRsDesc, blob: *mut *mut c_void, err: *mut *mut c_void) -> HRESULT {
    // SAFETY: the export is the one named, and `desc` is null or readable by the caller's contract.
    unsafe {
        let Some(real) = pc_d3d12::<SerializeVersionedFn>(c"D3D12SerializeVersionedRootSignature") else {
            return E_NOTIMPL;
        };
        if !desc.is_null() && (*desc).desc.flags & !PUBLIC_FLAGS != 0 {
            let mut copy = *desc;
            report_flags("D3D12SerializeVersionedRootSignature", copy.desc.flags & !PUBLIC_FLAGS);
            copy.desc.flags &= PUBLIC_FLAGS;
            return real(&copy, blob, err);
        }
        real(desc, blob, err)
    }
}

/// Hands the PC deserializer a cleaned copy when the container carries console only flags. The
/// deserializer parses the data during creation, so the copy is freed right after.
///
/// # Safety
/// `data` points to `size` readable bytes, `riid` is readable and `out` writable.
unsafe fn deserialize(name: &CStr, data: *const c_void, size: usize, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above, and the export is the one named.
    unsafe {
        let Some(real) = pc_d3d12::<DeserializerFn>(name) else {
            return E_NOTIMPL;
        };
        let bytes: &[u8] = if data.is_null() { &[] } else { std::slice::from_raw_parts(data.cast(), size) };
        let dirty = flags_offset(bytes).is_some_and(|at| rd32(bytes, at) & !PUBLIC_FLAGS != 0);
        if !dirty {
            return real(data, size, riid, out);
        }
        let mut copy = Vec::new();
        if copy.try_reserve_exact(size).is_err() {
            return E_OUTOFMEMORY;
        }
        copy.extend_from_slice(bytes);
        report_flags(&name.to_string_lossy(), strip_unknown_flags(&mut copy));
        real(copy.as_ptr().cast(), size, riid, out)
    }
}

/// # Safety
/// The arguments are those of `D3D12CreateRootSignatureDeserializer`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12CreateRootSignatureDeserializer(data: *const c_void, size: usize, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the caller's contract is that of `deserialize`.
    unsafe { deserialize(c"D3D12CreateRootSignatureDeserializer", data, size, riid, out) }
}

/// # Safety
/// The arguments are those of `D3D12CreateVersionedRootSignatureDeserializer`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12CreateVersionedRootSignatureDeserializer(data: *const c_void, size: usize, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: the caller's contract is that of `deserialize`.
    unsafe { deserialize(c"D3D12CreateVersionedRootSignatureDeserializer", data, size, riid, out) }
}

// ---- device ----

/// The device checks of the first two device creation calls: a PC device on the default adapter at
/// feature level 12_0 or above (the console GPU is 12_2 class, and nothing below 12_0 is offered as
/// if it were), returned inside the console device proxy.
///
/// # Safety
/// `riid` is readable and `out` null or writable.
unsafe fn create_console_device(who: &str, min_level: u32, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        let iid = guid_at(riid);
        let Some(create) = pc_d3d12::<CreateDeviceFn>(c"D3D12CreateDevice") else {
            return E_NOTIMPL;
        };
        let level = min_level.max(XS_FEATURE_LEVEL_12_0);
        if out.is_null() {
            // Capability check, as on PC: no device is created.
            let adapter = crate::adapter_choice::pick();
            let hr = create(adapter, level, &IID_ID3D12Device, std::ptr::null_mut());
            crate::adapter_choice::release(adapter);
            xs_log!("{who}: capability check at feature level 0x{level:X}, hr=0x{:08X}", hr as u32);
            return hr;
        }
        *out = std::ptr::null_mut();
        crate::watchdog_hold::install();
        crate::diagnose::before_device();
        let mut dev: *mut c_void = std::ptr::null_mut();
        let adapter = crate::adapter_choice::pick();
        let hr = create(adapter, level, &IID_ID3D12Device, &mut dev);
        crate::adapter_choice::release(adapter);
        if hr < 0 {
            xs_log!("{who}: PC D3D12CreateDevice (feature level 0x{level:X}) failed, hr=0x{:08X}", hr as u32);
            return hr;
        }
        crate::diagnose::watch(dev);
        crate::title_mem::device_created(dev);
        crate::mem_sample::start(dev);
        crate::proxy::note_mesh_support(dev);
        {
            use windows::Win32::Graphics::Direct3D12::ID3D12Device;
            use windows::core::Interface;
            if let Some(device) = ID3D12Device::from_raw_borrowed(&dev) {
                crate::direct_source::note_device_waves(device);
            }
        }
        if !riid.is_null() && crate::abi::note_device_iid(&std::ptr::read_unaligned(riid.cast::<[u8; 16]>())) {
            xs_log!("{who}: the title asks for ID3D12Device11, its runtime has the newer console layout");
        }
        // The caller is console code: it gets the console ABI proxy whatever IID it asked for (a
        // console IID may not exist on PC). The proxy takes over `dev`'s reference.
        if proxy_enabled() {
            let proxy = wrap(dev, Kind::Device);
            if proxy.is_null() {
                xs_log!("{who}: console device proxy unavailable for {{{iid}}}");
                release_object(dev);
                return E_NOINTERFACE;
            }
            xs_log!("{who}: returned the console device proxy {} for {{{iid}}}", p(proxy));
            *out = proxy;
            return S_OK;
        }
        let hr = query_object(dev, riid, out);
        if hr >= 0 {
            xs_log!("{who}: DECANT_D3D12X_PROXY=0, returned the bare PC device for {{{iid}}}");
        } else {
            xs_log!("{who}: DECANT_D3D12X_PROXY=0 and {{{iid}}} is not a PC D3D12 interface (hr=0x{:08X})", hr as u32);
        }
        release_object(dev);
        hr
    }
}

/// `DECANT_D3D12X_PROXY=0` returns the bare PC device instead (the old behaviour, for comparisons).
fn proxy_enabled() -> bool {
    std::env::var_os("DECANT_D3D12X_PROXY").is_none_or(|v| v != "0")
}

/// `D3D12XboxCreateDevice(adapter, parameters, riid, device)`: four arguments per the public
/// documentation. The parameters structure is not public, it is logged by address only and not
/// interpreted.
///
/// # Safety
/// `riid` is readable and `out` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12XboxCreateDevice(adapter: *mut c_void, params: *const c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        xs_log!("D3D12XboxCreateDevice(adapter={}, params={}, riid={{{}}})", p(adapter), p(params), guid_at(riid));
        if out.is_null() {
            return E_POINTER;
        }
        if !adapter.is_null() {
            xs_log!("D3D12XboxCreateDevice: console adapter object {} ignored, using the default PC adapter", p(adapter));
        }
        create_console_device("D3D12XboxCreateDevice", XS_FEATURE_LEVEL_12_0, riid, out)
    }
}

/// `D3D12CreateDevice(adapter, minimum feature level, riid, device)`, the public signature. Console
/// titles import it from `d3d12_xs` (another title): same path as `D3D12XboxCreateDevice`.
///
/// # Safety
/// `riid` is readable and `out` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12CreateDevice(adapter: *mut c_void, min_level: u32, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: by the contract above.
    unsafe {
        xs_log!("D3D12CreateDevice(adapter={}, feature level 0x{min_level:X}, riid={{{}}}, out={})", p(adapter), guid_at(riid), p(out.cast()));
        if !adapter.is_null() {
            xs_log!("D3D12CreateDevice: adapter object {} ignored, using the default PC adapter", p(adapter));
        }
        create_console_device("D3D12CreateDevice", min_level, riid, out)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn D3D12XboxSetProcessDebugFlags(flags: u32) -> HRESULT {
    xs_log!("D3D12XboxSetProcessDebugFlags(0x{flags:08X}): accepted, no effect on PC");
    S_OK
}

/// The description `D3D12ConfigureVirtualMemoryX` takes, 32 bytes. Shape measured on a retail
/// executable (facts only): the title passes flags 0, 4 page-table pages, 0 bytes.
#[repr(C)]
pub struct VirtualMemoryConfig {
    pub flags: u32,
    pub page_table_pages: u32,
    pub framebuffer_preserve_bytes: u32,
    pub reserved: u32,
    pub out0: u64,
    pub out1: u64,
}

/// The success code the console answers, not `S_OK`; the title goes on with it.
pub const CONFIGURE_VIRTUAL_MEMORY_OK: HRESULT = 0x1000_0000;

/// The title's first console call. `E_INVALIDARG` for a flag other than bit 0 or more than
/// 0x21 page-table pages; otherwise both output words are cleared and `CONFIGURE_VIRTUAL_MEMORY_OK`
/// is returned. Nothing is reserved on PC: the page table belongs to the console GPU. The console
/// also hands the sizes to xmem (`XMemRegisterGraphicsMemoryConfiguration`); ours has no such
/// export and the title does not need it. Every call is logged with its fields.
#[unsafe(no_mangle)]
pub extern "system" fn D3D12ConfigureVirtualMemoryX(config: *mut VirtualMemoryConfig) -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    let k = N.fetch_add(1, Ordering::SeqCst) + 1;
    if config.is_null() {
        xs_log!("D3D12ConfigureVirtualMemoryX (call {k}): null description, E_POINTER");
        return E_POINTER;
    }
    // SAFETY: the caller passes a writable 32-byte description (checked non-null above).
    let c = unsafe { &mut *config };
    let result = if c.flags & !1 != 0 || c.page_table_pages > 0x21 {
        E_INVALIDARG
    } else {
        c.out0 = 0;
        c.out1 = 0;
        CONFIGURE_VIRTUAL_MEMORY_OK
    };
    if k <= 3 || k % 1000 == 0 {
        xs_log!(
            "D3D12ConfigureVirtualMemoryX (call {k}) flags 0x{:X}, page-table pages {}, framebuffer preserve {} bytes, reserved 0x{:X}: 0x{:08X}",
            c.flags,
            c.page_table_pages,
            c.framebuffer_preserve_bytes,
            c.reserved,
            result as u32
        );
    }
    result
}

/// The console `d3d12_xs.dll` answers `DXGI_ERROR_UNSUPPORTED` from these four without reading
/// their arguments (disassembly of the console DLL), so their arguments are neither declared
/// nor read here: the x64 convention lets a caller pass any. Each call is logged.
fn dxgix_unsupported(name: &str, calls: &AtomicI32) -> HRESULT {
    let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
    if n <= 3 || n % 1000 == 0 {
        xs_log!("{name} (call {n}): DXGI_ERROR_UNSUPPORTED, as on the console");
    }
    XS_DXGI_ERROR_UNSUPPORTED
}

#[unsafe(no_mangle)]
pub extern "system" fn DXGIXPresentArray() -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    dxgix_unsupported("DXGIXPresentArray", &N)
}

#[unsafe(no_mangle)]
pub extern "system" fn DXGIXSetVLineNotification() -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    dxgix_unsupported("DXGIXSetVLineNotification", &N)
}

#[unsafe(no_mangle)]
pub extern "system" fn DXGIXSetFrameNotification() -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    dxgix_unsupported("DXGIXSetFrameNotification", &N)
}

#[unsafe(no_mangle)]
pub extern "system" fn DXGIXGetFrameStatistics() -> HRESULT {
    static N: AtomicI32 = AtomicI32::new(0);
    dxgix_unsupported("DXGIXGetFrameStatistics", &N)
}

/// Returns the underlying PC D3D12 COM pointer when `object` is one of this crate's console
/// proxies, or `object` unchanged otherwise (including null). Lets companion shims such as
/// `XboxBvhBuilder.dll` obtain PC interfaces like `ID3D12Device5` without reading proxy internals.
///
/// # Safety
/// `object` is null or a valid COM object pointer whose first pointer-sized word is readable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn D3D12XsUnwrap(object: *mut c_void) -> *mut c_void {
    // SAFETY: by the safety contract above.
    unsafe { crate::proxy::unwrap(object) }
}

/// Lets the presentation layer (PC swap chain) receive PresentX.
#[unsafe(no_mangle)]
pub extern "system" fn DecantD3D12X_SetPresentHook(hook: Option<PresentFn>) {
    PRESENT_HOOK.store(hook.map_or(0, |f| f as usize), Ordering::SeqCst);
    xs_log!("DecantD3D12X_SetPresentHook({})", p(hook.map_or(0, |f| f as usize) as *const c_void));
}

/// This module's handle, from the linker defined image base.
fn this_module() -> HMODULE {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    // Only the address of the linker defined symbol is taken, it is never read.
    (&raw const __ImageBase) as HMODULE
}

/// What the C `DllMain` did on `DLL_PROCESS_ATTACH`. It runs from the C runtime start up code of
/// the DLL, which calls the `.CRT$XCU` initializers once when the process attaches. A `DllMain`
/// written in Rust would be exported by the `cdylib`, and the C DLL has no such export.
extern "C" fn on_process_attach() {
    // SAFETY: the handle is this module, loaded while this code runs.
    unsafe { DisableThreadLibraryCalls(this_module()) };
    // A panic inside the layer ends the title with a fail fast and no text: the message and its
    // place go to the log first. The hook before it still runs, so a test program that links the
    // crate prints its failed asserts as usual.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        xs_log!("PANIC in d3d12_xs: {info}");
        previous(info);
    }));
}

#[used]
#[unsafe(link_section = ".CRT$XCU")]
static ON_PROCESS_ATTACH: extern "C" fn() = on_process_attach;

/// The title or a layer wrote `len` bytes at title address `address` (DirectStorage found by name in
/// `d3d12_xs.dll`): the placed textures that overlap them are detiled again before the next submission.
#[unsafe(no_mangle)]
pub extern "system" fn DecantD3D12XNoteWrite(address: usize, len: usize) {
    crate::detile::note_write(address as u64, len as u64);
}
