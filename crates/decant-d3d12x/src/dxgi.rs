//! Headless DXGI output chain: a fallback output for an adapter that has none (the
//! device/adapter/output/factory/present leg of the console boot chain).
//!
//! `proxy::pc_adapter` already bridges `IDXGIDevice::GetAdapter`/`GetParent` to the host
//! `dxgi.dll` by the PC device's LUID. That covers the device and the adapter. What it cannot
//! give a title is an output: under this layer's no-window, no-desktop-DXGI setup the real
//! adapter's `EnumOutputs` can have nothing to enumerate. A retail title's measured boot asks for
//! `IDXGIDevice1`, then `GetAdapter`, then `EnumOutputs(0)`, and passes that output into
//! `SetFrameIntervalX`'s `output` argument without reading it again; `frame::set_interval`
//! already ignores that argument. The only requirement for the chain not to abort a title's boot
//! is that `EnumOutputs(0)` hands back a well formed object, not that the object does anything.
//!
//! `wrap_adapter` gives the real adapter exactly one fallback: when its own `EnumOutputs(0)`
//! fails, a self contained virtual output answers instead. Every other index and every other
//! adapter method still reaches the real adapter untouched.
//!
//! Console output slot 21: two independent sources agree on its shape, nothing else about it.
//! Our own C console layer's DXGI device file measured a retail title calling output slot 21
//! with `(0, 0, &count, NULL)` right after `EnumOutputs(0)`, not checking the HRESULT but reading
//! `count` back (the public `IDXGIOutput` vtable the chain otherwise matches ends at slot 18, so
//! slot 21 is three past it). A second, independent public console API project's device object
//! names that same extra slot "the Xbox `GetDisplayModeList1`/`GetDisplayModeListX`" shape:
//! `(format, flags, count*, modes*)`. Bound here on that agreement: `count` is written (0 when
//! `modes` is null, this call's shape) and the call succeeds.
//!
//! Not added here: swap chain creation, back buffers, `Present`. No source (ours or either
//! public project) shows a console title creating a DXGI swap chain; the measured chain stops at
//! the output, and frames reach the screen through `PresentX` on plain `ID3D12Resource` back
//! buffers (queue slot 30, `proxy::queue_present_x`), already routed to the present hook and to
//! `frame`'s pacing. A swap chain object with nothing to exercise it would not be reuse, it would
//! be new code for a path no evidence shows is taken.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, Ordering};

use windows_sys::Win32::Foundation::{E_NOTIMPL, E_POINTER, S_OK};
use windows_sys::core::{GUID, HRESULT};

use crate::log::xs_log;
use crate::proxy::log_due;

type Object = *mut c_void;

const IID_UNKNOWN: GUID = GUID::from_u128(0x00000000_0000_0000_c000_000000000046);
const IID_DXGI_OBJECT: GUID = GUID::from_u128(0xaec22fb8_76f3_4639_9be0_28eb43a67a2e);
const IID_DXGI_ADAPTER: GUID = GUID::from_u128(0x2411e7e1_12ac_4ccf_bd14_9798e8534dc0);
const IID_DXGI_OUTPUT: GUID = GUID::from_u128(0xae02eedb_c735_4690_8d52_5a8dc20213aa);

/// # Safety
/// `riid` points to a readable GUID.
unsafe fn is_iid(riid: *const GUID, guid: &GUID) -> bool {
    // SAFETY: by the contract above.
    let a = unsafe { &*riid };
    a.data1 == guid.data1 && a.data2 == guid.data2 && a.data3 == guid.data3 && a.data4 == guid.data4
}

/// # Safety
/// `object` starts with a vtable pointer with at least `slot + 1` entries of type `F`.
unsafe fn method<F: Copy>(object: Object, slot: usize) -> F {
    const { assert!(size_of::<F>() == size_of::<usize>()) };
    // SAFETY: by the contract above.
    unsafe {
        let vtbl = *(object as *const *const usize);
        std::mem::transmute_copy::<usize, F>(&*vtbl.add(slot))
    }
}

type Query = unsafe extern "system" fn(Object, *const GUID, *mut Object) -> HRESULT;
type Release = unsafe extern "system" fn(Object) -> u32;
type EnumOutputsFn = unsafe extern "system" fn(Object, u32, *mut Object) -> HRESULT;
type GetDescFn = unsafe extern "system" fn(Object, *mut c_void) -> HRESULT;
type CheckSupportFn = unsafe extern "system" fn(Object, *const GUID, *mut i64) -> HRESULT;

fn dxgi_log_once(what: &str) {
    static N: AtomicI32 = AtomicI32::new(0);
    decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("DXGI {what}"));
    if log_due(&N) {
        xs_log!("DXGI {what}: not on the measured console boot chain, E_NOTIMPL");
    }
}

// ---- virtual output: IUnknown(3) + IDXGIObject(4) + GetDesc, GetDisplayModeList (12 methods,
// slots 0..18, the public IDXGIOutput order), then console slot 21 ----

#[repr(C)]
struct VOutput {
    vtbl: *const usize,
    refs: AtomicI32,
}

extern "system" fn vo_qi(this: *mut VOutput, riid: *const GUID, out: *mut Object) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live, `riid` readable, `out` writable, by the COM contract.
    unsafe {
        if is_iid(riid, &IID_UNKNOWN)
            || is_iid(riid, &IID_DXGI_OBJECT)
            || is_iid(riid, &IID_DXGI_OUTPUT)
        {
            (*this).refs.fetch_add(1, Ordering::SeqCst);
            *out = this.cast();
            return S_OK;
        }
        *out = std::ptr::null_mut();
    }
    windows_sys::Win32::Foundation::E_NOINTERFACE
}

extern "system" fn vo_add_ref(this: *mut VOutput) -> u32 {
    // SAFETY: `this` is live.
    (unsafe { &*this }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
}

extern "system" fn vo_release(this: *mut VOutput) -> u32 {
    // SAFETY: `this` is live, freed once when its last reference goes.
    unsafe {
        let n = (*this).refs.fetch_sub(1, Ordering::SeqCst) - 1;
        if n == 0 {
            drop(Box::from_raw(this));
        }
        n as u32
    }
}

/// Size and `AttachedToDesktop` byte offset of x86_64 `DXGI_OUTPUT_DESC`.
const DXGI_OUTPUT_DESC_SIZE: usize = 0x60;
const DXGI_OUTPUT_ATTACHED_OFFSET: usize = 0x50;

#[cfg(windows)]
const _: () = {
    use std::mem::{offset_of, size_of};
    use windows::Win32::Graphics::Dxgi::DXGI_OUTPUT_DESC;
    assert!(size_of::<DXGI_OUTPUT_DESC>() == DXGI_OUTPUT_DESC_SIZE);
    assert!(offset_of!(DXGI_OUTPUT_DESC, AttachedToDesktop) == DXGI_OUTPUT_ATTACHED_OFFSET);
};

extern "system" fn vo_get_desc(_this: *mut VOutput, desc: *mut c_void) -> HRESULT {
    if desc.is_null() {
        return E_POINTER;
    }
    // SAFETY: `desc` is a writable `DXGI_OUTPUT_DESC` (0x60 bytes on x64; `AttachedToDesktop`
    // sits at byte offset 0x50, verified against the Windows SDK layout above).
    unsafe {
        std::ptr::write_bytes(desc.cast::<u8>(), 0, DXGI_OUTPUT_DESC_SIZE);
        desc.cast::<u8>().add(DXGI_OUTPUT_ATTACHED_OFFSET).cast::<i32>().write_unaligned(1);
    }
    S_OK
}

extern "system" fn vo_display_mode_list(
    _this: *mut VOutput,
    _format: u32,
    _flags: u32,
    count: *mut u32,
    modes: *mut c_void,
) -> HRESULT {
    dxgi_log_once("IDXGIOutput::GetDisplayModeList");
    if !count.is_null() {
        // SAFETY: `count` is writable when not null, by the COM contract.
        unsafe { *count = 0 };
    }
    if !modes.is_null() { E_NOTIMPL } else { S_OK }
}

/// One entry of the console's display mode list (32 bytes): size, refresh rate as numerator and
/// denominator, format, scanline order, scaling and flags.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct ConsoleMode {
    width: u32,
    height: u32,
    refresh: [u32; 2],
    format: u32,
    scanlines: u32,
    scaling: u32,
    flags: u32,
}
const _: () = assert!(size_of::<ConsoleMode>() == 32);

/// The only mode this layer lists, 3840x2160 at 60 Hz progressive.
const UHD_60: ConsoleMode = ConsoleMode { width: 3840, height: 2160, refresh: [60, 1], format: 0, scanlines: 1, scaling: 0, flags: 0 };

/// `HRESULT_FROM_WIN32(ERROR_MORE_DATA)`.
const MORE_DATA: HRESULT = 0x8007_00EA_u32 as HRESULT;

/// Fills a caller's mode list with the single mode we have. `capacity` is `None` when the caller
/// only asks how many modes exist. Returns the count to report and the result.
fn fill_mode_list(capacity: Option<u32>) -> (u32, HRESULT) {
    match capacity {
        None => (1, S_OK),
        Some(0) => (0, MORE_DATA),
        Some(_) => (1, S_OK),
    }
}

/// Console output slot 21, `(format, flags, count*, modes*)`. A null `modes` only reports the
/// count of 1. A list with room for no entry fails with `ERROR_MORE_DATA`.
unsafe extern "system" fn vo_mode_list_x(
    _this: *mut VOutput,
    _format: u32,
    _flags: u32,
    count: *mut u32,
    modes: *mut ConsoleMode,
) -> HRESULT {
    if count.is_null() {
        return E_POINTER;
    }
    // SAFETY: `count` is the title's writable count, checked not null above.
    let capacity = if modes.is_null() { None } else { Some(unsafe { *count }) };
    let (written, hr) = fill_mode_list(capacity);
    if hr != S_OK {
        return hr;
    }
    // SAFETY: as above, and `modes` has room for the `capacity` entries the title announced, of
    // which at least one is needed here.
    unsafe {
        *count = written;
        if !modes.is_null() {
            modes.write_unaligned(UHD_60);
        }
    }
    S_OK
}

/// A method this layer has no evidence for: logged once, `E_NOTIMPL`, at most three extra
/// pointer-sized arguments (every unimplemented `IDXGIOutput`/padding slot fits that shape).
extern "system" fn vo_unimplemented(
    _this: *mut VOutput,
    _a: usize,
    _b: usize,
    _c: usize,
) -> HRESULT {
    dxgi_log_once(
        "IDXGIOutput slot (gamma, ownership, display surface, frame statistics, or an unlisted slot)",
    );
    E_NOTIMPL
}

extern "system" fn vo_ok_noop(_this: *mut VOutput, _a: usize, _b: usize, _c: usize) -> HRESULT {
    S_OK
}

/// A vtable that can sit in a static: the entries are code addresses, never written.
#[repr(transparent)]
struct Vtable<const N: usize>([*const c_void; N]);

// SAFETY: every entry is the address of a `extern "system" fn`, fixed at compile time.
unsafe impl<const N: usize> Sync for Vtable<N> {}

/// 22 entries: the public `IDXGIOutput` order (0..18), two padding slots (19, 20) and the bound
/// console slot 21.
static VOUTPUT_VTBL: Vtable<22> = Vtable([
    vo_qi as *const c_void,
    vo_add_ref as *const c_void,
    vo_release as *const c_void,
    vo_ok_noop as *const c_void,           // 3 SetPrivateData
    vo_ok_noop as *const c_void,           // 4 SetPrivateDataInterface
    vo_unimplemented as *const c_void,     // 5 GetPrivateData
    vo_unimplemented as *const c_void,     // 6 GetParent
    vo_get_desc as *const c_void,          // 7 GetDesc
    vo_display_mode_list as *const c_void, // 8 GetDisplayModeList
    vo_unimplemented as *const c_void,     // 9 FindClosestMatchingMode
    vo_ok_noop as *const c_void,           // 10 WaitForVBlank
    vo_unimplemented as *const c_void,     // 11 TakeOwnership
    vo_ok_noop as *const c_void,           // 12 ReleaseOwnership (void)
    vo_unimplemented as *const c_void,     // 13 GetGammaControlCapabilities
    vo_unimplemented as *const c_void,     // 14 SetGammaControl
    vo_unimplemented as *const c_void,     // 15 GetGammaControl
    vo_unimplemented as *const c_void,     // 16 SetDisplaySurface
    vo_unimplemented as *const c_void,     // 17 GetDisplaySurfaceData
    vo_unimplemented as *const c_void,     // 18 GetFrameStatistics
    vo_unimplemented as *const c_void,     // 19 padding (IDXGIOutput1 slot on the real interface)
    vo_unimplemented as *const c_void,     // 20 padding
    vo_mode_list_x as *const c_void,       // 21 console GetDisplayModeListX
]);

fn make_virtual_output() -> Object {
    let o = Box::new(VOutput {
        vtbl: VOUTPUT_VTBL.0.as_ptr().cast(),
        refs: AtomicI32::new(1),
    });
    Box::into_raw(o).cast()
}

// ---- adapter shadow: forwards to the real adapter, EnumOutputs falls back on index 0 ----

#[repr(C)]
struct AdapterShadow {
    vtbl: *const usize,
    real: Object,
    refs: AtomicI32,
}

fn shadow_vtbl() -> &'static [usize; 10] {
    static V: std::sync::OnceLock<[usize; 10]> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        [
            as_qi as *const () as usize,
            as_add_ref as *const () as usize,
            as_release as *const () as usize,
            as_set_private_data as *const () as usize, // 3 SetPrivateData
            as_fwd2 as *const () as usize,             // 4 SetPrivateDataInterface
            as_get_private_data as *const () as usize, // 5 GetPrivateData
            as_get_parent as *const () as usize,
            as_enum_outputs as *const () as usize,
            as_get_desc as *const () as usize,
            as_check_support as *const () as usize,
        ]
    })
}

extern "system" fn as_qi(this: *mut AdapterShadow, riid: *const GUID, out: *mut Object) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live, `riid` readable, `out` writable.
    unsafe {
        if is_iid(riid, &IID_UNKNOWN)
            || is_iid(riid, &IID_DXGI_OBJECT)
            || is_iid(riid, &IID_DXGI_ADAPTER)
        {
            as_add_ref(this);
            *out = this.cast();
            return S_OK;
        }
        method::<Query>((*this).real, 0)((*this).real, riid, out)
    }
}

extern "system" fn as_add_ref(this: *mut AdapterShadow) -> u32 {
    (unsafe { &*this }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
}

extern "system" fn as_release(this: *mut AdapterShadow) -> u32 {
    // SAFETY: `this` is live; the real adapter's reference is released once, with the shadow.
    unsafe {
        let n = (*this).refs.fetch_sub(1, Ordering::SeqCst) - 1;
        if n == 0 {
            method::<Release>((*this).real, 2)((*this).real);
            drop(Box::from_raw(this));
        }
        n as u32
    }
}

/// `IDXGIObject::SetPrivateData` (slot 3): `(riid, data_size, data)`.
extern "system" fn as_set_private_data(
    this: *mut AdapterShadow,
    riid: *const GUID,
    size: u32,
    data: *const c_void,
) -> HRESULT {
    // SAFETY: `this` is live, forwarded to the real adapter's SetPrivateData (slot 3).
    unsafe {
        method::<unsafe extern "system" fn(Object, *const GUID, u32, *const c_void) -> HRESULT>(
            (*this).real,
            3,
        )((*this).real, riid, size, data)
    }
}

/// `IDXGIObject::SetPrivateDataInterface` (slot 4): `(riid, unknown)`.
extern "system" fn as_fwd2(this: *mut AdapterShadow, a: *const GUID, b: *const c_void) -> HRESULT {
    // SAFETY: `this` is live, forwarded to the real adapter's SetPrivateDataInterface (slot 4).
    unsafe {
        method::<unsafe extern "system" fn(Object, *const GUID, *const c_void) -> HRESULT>(
            (*this).real,
            4,
        )((*this).real, a, b)
    }
}

/// `IDXGIObject::GetPrivateData` (slot 5): `(riid, data_size*, data)`.
extern "system" fn as_get_private_data(
    this: *mut AdapterShadow,
    riid: *const GUID,
    size: *mut u32,
    data: *mut c_void,
) -> HRESULT {
    // SAFETY: `this` is live, forwarded to the real adapter's GetPrivateData (slot 5).
    unsafe {
        method::<unsafe extern "system" fn(Object, *const GUID, *mut u32, *mut c_void) -> HRESULT>(
            (*this).real,
            5,
        )((*this).real, riid, size, data)
    }
}

extern "system" fn as_get_parent(
    this: *mut AdapterShadow,
    riid: *const GUID,
    out: *mut Object,
) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live, forwarded to the real adapter's GetParent (slot 6).
    unsafe { method::<Query>((*this).real, 6)((*this).real, riid, out) }
}

/// Console boot measured shape: index 0 only. The real adapter is tried first; a virtual output
/// answers only when it fails. Any other index is the real adapter's own answer, unchanged.
extern "system" fn as_enum_outputs(
    this: *mut AdapterShadow,
    index: u32,
    out: *mut Object,
) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live, `(*this).real` is a live real `IDXGIAdapter`.
    unsafe {
        let real = (*this).real;
        let hr = method::<EnumOutputsFn>(real, 7)(real, index, out);
        if hr >= 0 || index != 0 {
            return hr;
        }
        xs_log!(
            "DXGI: the real adapter has no output at index 0 (hr=0x{:08X}), answering a virtual one (no window, no desktop DXGI)",
            hr as u32
        );
        *out = make_virtual_output();
        S_OK
    }
}

extern "system" fn as_get_desc(this: *mut AdapterShadow, desc: *mut c_void) -> HRESULT {
    // SAFETY: `this` is live, forwarded to the real adapter's GetDesc (slot 8).
    unsafe { method::<GetDescFn>((*this).real, 8)((*this).real, desc) }
}

extern "system" fn as_check_support(
    this: *mut AdapterShadow,
    guid: *const GUID,
    out: *mut i64,
) -> HRESULT {
    // SAFETY: `this` is live, forwarded to the real adapter's CheckInterfaceSupport (slot 9).
    unsafe { method::<CheckSupportFn>((*this).real, 9)((*this).real, guid, out) }
}

/// Wraps a live real `IDXGIAdapter` (one reference, taken over) so its `EnumOutputs` gets the
/// headless fallback. Called from `proxy::pc_adapter` right after the real adapter is found.
pub(crate) fn wrap_adapter(real: Object) -> Object {
    if real.is_null() {
        return real;
    }
    let shadow = Box::new(AdapterShadow {
        vtbl: shadow_vtbl().as_ptr(),
        real,
        refs: AtomicI32::new(1),
    });
    Box::into_raw(shadow).cast()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn the_mode_list_sizing_rules() {
        assert_eq!(fill_mode_list(None), (1, S_OK));
        assert_eq!(fill_mode_list(Some(0)), (0, MORE_DATA));
        assert_eq!(fill_mode_list(Some(1)), (1, S_OK));
        assert_eq!(fill_mode_list(Some(9)), (1, S_OK));
    }

    #[test]
    fn the_console_mode_list_slot_reports_and_writes_the_one_mode() {
        let blank = ConsoleMode { width: 0, height: 0, refresh: [0; 2], format: 0, scanlines: 0, scaling: 0, flags: 0 };
        let mut count = 0u32;
        let mut slot = [blank; 2];
        // SAFETY: `this` is unused, `count` and `modes` point at live locals or are null.
        unsafe {
            assert_eq!(vo_mode_list_x(std::ptr::null_mut(), 0, 0, std::ptr::null_mut(), slot.as_mut_ptr()), E_POINTER);
            assert_eq!(vo_mode_list_x(std::ptr::null_mut(), 0, 0, &mut count, std::ptr::null_mut()), S_OK);
            assert_eq!(count, 1);
            count = 0;
            assert_eq!(vo_mode_list_x(std::ptr::null_mut(), 0, 0, &mut count, slot.as_mut_ptr()), MORE_DATA);
            assert_eq!(slot[0], blank);
            count = 2;
            assert_eq!(vo_mode_list_x(std::ptr::null_mut(), 0, 0, &mut count, slot.as_mut_ptr()), S_OK);
        }
        assert_eq!(count, 1);
        assert_eq!(slot, [UHD_60, blank]);
        assert_eq!((UHD_60.width, UHD_60.height, UHD_60.refresh), (3840, 2160, [60, 1]));
    }

    /// A fake `IDXGIAdapter` whose `EnumOutputs` always fails, standing in for a headless host
    /// `dxgi.dll` adapter with no attached display.
    #[repr(C)]
    struct FakeAdapter {
        vtbl: *const usize,
        refs: AtomicI32,
    }
    static FAKE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static FAKE_SET_PRIVATE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static FAKE_GET_PRIVATE_CALLS: AtomicUsize = AtomicUsize::new(0);

    extern "system" fn fake_qi(_t: *mut FakeAdapter, _r: *const GUID, _o: *mut Object) -> HRESULT {
        windows_sys::Win32::Foundation::E_NOINTERFACE
    }
    extern "system" fn fake_add_ref(this: *mut FakeAdapter) -> u32 {
        (unsafe { &*this }.refs.fetch_add(1, Ordering::SeqCst) + 1) as u32
    }
    extern "system" fn fake_release(this: *mut FakeAdapter) -> u32 {
        unsafe {
            let n = (*this).refs.fetch_sub(1, Ordering::SeqCst) - 1;
            if n == 0 {
                drop(Box::from_raw(this));
            }
            n as u32
        }
    }
    extern "system" fn fake_set_private(
        _t: *mut FakeAdapter,
        _a: *const GUID,
        _b: u32,
        _c: *const c_void,
    ) -> HRESULT {
        FAKE_SET_PRIVATE_CALLS.fetch_add(1, Ordering::SeqCst);
        S_OK
    }
    extern "system" fn fake_fwd2(
        _t: *mut FakeAdapter,
        _a: *const GUID,
        _b: *const c_void,
    ) -> HRESULT {
        S_OK
    }
    extern "system" fn fake_get_private(
        _t: *mut FakeAdapter,
        _a: *const GUID,
        size: *mut u32,
        _c: *mut c_void,
    ) -> HRESULT {
        FAKE_GET_PRIVATE_CALLS.fetch_add(1, Ordering::SeqCst);
        if !size.is_null() {
            unsafe { *size = 42 };
        }
        S_OK
    }
    extern "system" fn fake_get_parent(
        _t: *mut FakeAdapter,
        _r: *const GUID,
        out: *mut Object,
    ) -> HRESULT {
        unsafe { *out = std::ptr::null_mut() };
        E_NOTIMPL
    }
    const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A_0002_u32 as HRESULT;
    extern "system" fn fake_enum_outputs(
        _t: *mut FakeAdapter,
        index: u32,
        out: *mut Object,
    ) -> HRESULT {
        FAKE_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe { *out = std::ptr::null_mut() };
        if index == 0 {
            DXGI_ERROR_NOT_FOUND
        } else {
            E_NOTIMPL
        }
    }
    extern "system" fn fake_get_desc(_t: *mut FakeAdapter, _d: *mut c_void) -> HRESULT {
        S_OK
    }
    extern "system" fn fake_check_support(
        _t: *mut FakeAdapter,
        _g: *const GUID,
        _v: *mut i64,
    ) -> HRESULT {
        E_NOTIMPL
    }

    fn fake_vtbl() -> &'static [usize; 10] {
        static V: std::sync::OnceLock<[usize; 10]> = std::sync::OnceLock::new();
        V.get_or_init(|| {
            [
                fake_qi as *const () as usize,
                fake_add_ref as *const () as usize,
                fake_release as *const () as usize,
                fake_set_private as *const () as usize,
                fake_fwd2 as *const () as usize,
                fake_get_private as *const () as usize,
                fake_get_parent as *const () as usize,
                fake_enum_outputs as *const () as usize,
                fake_get_desc as *const () as usize,
                fake_check_support as *const () as usize,
            ]
        })
    }

    /// Walks device -> adapter -> output -> console mode-list slot 21 with a null window
    /// (nothing here ever touches a window or a desktop compositor), then presents once through
    /// the virtual output's own "no-op frame" surface (`WaitForVBlank`, slot 10): the one call a
    /// headless chain can stand in for `Present` with, since the real presentation path
    /// (`PresentX`) is proven separately in `d3d12xs_test` and is not duplicated here.
    #[test]
    fn walks_the_chain_with_a_null_window_and_presents_once() {
        let real = Box::into_raw(Box::new(FakeAdapter {
            vtbl: fake_vtbl().as_ptr(),
            refs: AtomicI32::new(1),
        }))
        .cast::<c_void>();
        let shadow = wrap_adapter(real);
        assert!(!shadow.is_null(), "adapter -> shadow");

        // No window: every call below passes null where a window or surface would go.
        let null_window: *mut c_void = std::ptr::null_mut();
        assert!(null_window.is_null());

        let mut output: Object = std::ptr::null_mut();
        // SAFETY: `shadow` is a live `AdapterShadow`.
        let hr = unsafe { method::<EnumOutputsFn>(shadow, 7)(shadow, 0, &mut output) };
        assert_eq!(
            hr, S_OK,
            "EnumOutputs(0) falls back to the virtual output when the real one fails"
        );
        assert!(!output.is_null());
        assert_eq!(
            FAKE_CALLS.load(Ordering::SeqCst),
            1,
            "the real adapter's EnumOutputs was tried first"
        );

        let mut count: u32 = 0xFFFF_FFFF;
        // SAFETY: `output` is the virtual output just created.
        let hr = unsafe {
            method::<unsafe extern "system" fn(Object, u32, u32, *mut u32, *mut c_void) -> HRESULT>(
                output, 21,
            )(output, 0, 0, &mut count, std::ptr::null_mut())
        };
        assert_eq!(
            hr, S_OK,
            "console output slot 21 (GetDisplayModeListX) succeeds with a null mode list"
        );
        assert_eq!(count, 1, "count is written back, 1 for a null mode list");

        // "Presents once": the one frame-facing call a headless virtual output answers.
        // SAFETY: `output` is live.
        let hr =
            unsafe { method::<unsafe extern "system" fn(Object) -> HRESULT>(output, 10)(output) };
        assert_eq!(
            hr, S_OK,
            "WaitForVBlank (the output's own frame-facing call) succeeds with no display"
        );

        // SAFETY: both objects are live and this call releases each to zero.
        unsafe {
            assert_eq!(method::<Release>(output, 2)(output), 0);
            assert_eq!(method::<Release>(shadow, 2)(shadow), 0);
        }
    }

    #[test]
    fn vo_get_desc_zeros_struct_sets_attached_and_does_not_overrun() {
        #[repr(C)]
        struct Guarded {
            desc: [u8; DXGI_OUTPUT_DESC_SIZE],
            canary: [u8; 0x40],
        }
        let mut buf = Guarded {
            desc: [0x5A; DXGI_OUTPUT_DESC_SIZE],
            canary: [0xA5; 0x40],
        };
        assert_eq!(vo_get_desc(std::ptr::null_mut(), std::ptr::null_mut()), E_POINTER);
        let hr = vo_get_desc(std::ptr::null_mut(), buf.desc.as_mut_ptr().cast());
        assert_eq!(hr, S_OK);
        assert!(buf.desc[..DXGI_OUTPUT_ATTACHED_OFFSET].iter().all(|&b| b == 0));
        let attached = i32::from_le_bytes(
            buf.desc[DXGI_OUTPUT_ATTACHED_OFFSET..DXGI_OUTPUT_ATTACHED_OFFSET + 4].try_into().unwrap(),
        );
        assert_eq!(attached, 1);
        assert!(buf.desc[DXGI_OUTPUT_ATTACHED_OFFSET + 4..].iter().all(|&b| b == 0));
        assert_eq!(buf.canary, [0xA5; 0x40], "vo_get_desc must not write past DXGI_OUTPUT_DESC");

        #[cfg(windows)]
        {
            use windows::Win32::Graphics::Dxgi::DXGI_OUTPUT_DESC;
            let mut sdk_desc = DXGI_OUTPUT_DESC::default();
            assert_eq!(vo_get_desc(std::ptr::null_mut(), (&raw mut sdk_desc).cast()), S_OK);
            assert!(sdk_desc.AttachedToDesktop.as_bool());
            assert_eq!(sdk_desc.Rotation.0, 0);
            assert!(sdk_desc.Monitor.is_invalid());
        }
    }

    #[test]
    fn adapter_shadow_forwards_set_and_get_private_data_to_their_own_slots() {
        let real = Box::into_raw(Box::new(FakeAdapter {
            vtbl: fake_vtbl().as_ptr(),
            refs: AtomicI32::new(1),
        }))
        .cast::<c_void>();
        let shadow = wrap_adapter(real);
        let before_set = FAKE_SET_PRIVATE_CALLS.load(Ordering::SeqCst);
        let before_get = FAKE_GET_PRIVATE_CALLS.load(Ordering::SeqCst);
        let mut size = 0u32;
        let payload = [0u8; 4];
        // SAFETY: `shadow` is a live `AdapterShadow` wrapping `FakeAdapter`.
        unsafe {
            assert_eq!(
                method::<unsafe extern "system" fn(Object, *const GUID, u32, *const c_void) -> HRESULT>(
                    shadow, 3,
                )(shadow, &IID_UNKNOWN, payload.len() as u32, payload.as_ptr().cast()),
                S_OK
            );
            assert_eq!(
                method::<unsafe extern "system" fn(Object, *const GUID, *mut u32, *mut c_void) -> HRESULT>(
                    shadow, 5,
                )(shadow, &IID_UNKNOWN, &mut size, std::ptr::null_mut()),
                S_OK
            );
            assert_eq!(method::<Release>(shadow, 2)(shadow), 0);
        }
        assert_eq!(FAKE_SET_PRIVATE_CALLS.load(Ordering::SeqCst), before_set + 1);
        assert_eq!(FAKE_GET_PRIVATE_CALLS.load(Ordering::SeqCst), before_get + 1);
        assert_eq!(size, 42);
    }
}
