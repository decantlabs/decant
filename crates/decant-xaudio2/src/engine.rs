//! The engine wrapper: a COM object with the `IXAudio2` vtable that forwards to the PC engine,
//! the PC engine loader, COM initialisation and the XMA2 decoder probe. The vtable functions
//! receive raw COM pointers from the title, so this module is `unsafe` by nature.
#![allow(unsafe_code)]

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Foundation::{E_NOINTERFACE, E_POINTER, ERROR_MOD_NOT_FOUND, ERROR_PROC_NOT_FOUND, GetLastError, HMODULE, S_OK};
use windows::Win32::Media::Audio::XAudio2::{
    IXAudio2, IXAudio2SourceVoice_Vtbl, IXAudio2_Vtbl, IXAudio2Extension, XAUDIO2_BUFFER, XAUDIO2_BUFFER_WMA, XAUDIO2_DEBUG_CONFIGURATION,
    XAUDIO2_EFFECT_CHAIN, XAUDIO2_PERFORMANCE_DATA, XAUDIO2_VOICE_SENDS,
};
use windows::Win32::Media::Audio::{AUDIO_STREAM_CATEGORY, WAVEFORMATEX};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW, LoadLibraryW};
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface, PCSTR, PCWSTR};

use crate::stats::{Counter, bump};
use crate::xma2_hook::{GetDecoderFn, DecantXma2Decoder};
use crate::{CREATE_KNOWN, MASTERING_KNOWN, USE_DEFAULT_PROCESSOR, XAUDIO2_E_INVALID_CALL, guid_text, is_pc_format, is_xma, log, source_format_line, trace_all};

type CreateFn = unsafe extern "system" fn(*mut *mut c_void, u32, u32) -> HRESULT;
type FxFn = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
type SubmitFn = unsafe extern "system" fn(*mut c_void, *const XAUDIO2_BUFFER, *const XAUDIO2_BUFFER_WMA) -> HRESULT;

static VOICE_SUBMITS: OnceLock<Mutex<HashMap<usize, (SubmitFn, bool)>>> = OnceLock::new();

fn voice_submits() -> &'static Mutex<HashMap<usize, (SubmitFn, bool)>> {
    VOICE_SUBMITS.get_or_init(|| Mutex::new(HashMap::new()))
}

unsafe extern "system" fn submit_source_buffer(this: *mut c_void, buffer: *const XAUDIO2_BUFFER, wma: *const XAUDIO2_BUFFER_WMA) -> HRESULT {
    let (submit, bytes_to_log) = {
        let mut voices = voice_submits().lock().unwrap_or_else(|e| e.into_inner());
        let Some((submit, logged)) = voices.get_mut(&(this as usize)) else {
            return HRESULT(XAUDIO2_E_INVALID_CALL);
        };
        let bytes = if !*logged && !buffer.is_null() {
            // SAFETY: XAudio2 passes a valid XAUDIO2_BUFFER; its packed byte count is unaligned.
            Some(unsafe { std::ptr::addr_of!((*buffer).AudioBytes).read_unaligned() })
        } else { None };
        *logged |= bytes.is_some();
        (*submit, bytes)
    };
    if let Some(bytes) = bytes_to_log {
        log(&format!("SubmitSourceBuffer: first buffer {bytes} bytes"));
    }
    // SAFETY: this is the original XAudio2 source voice method captured at creation.
    unsafe { submit(this, buffer, wma) }
}

unsafe fn instrument_source_voice(raw: *mut c_void) {
    if raw.is_null() { return; }
    // SAFETY: `raw` is an XAudio2 source voice returned by CreateSourceVoice.
    let vtable_slot = raw.cast::<*const IXAudio2SourceVoice_Vtbl>();
    let original = unsafe { *vtable_slot };
    if original.is_null() { return; }
    // Each voice receives a private copy so other XAudio2 objects keep their original vtable.
    let mut copy = Box::new(unsafe { std::ptr::read(original) });
    let submit = copy.SubmitSourceBuffer;
    copy.SubmitSourceBuffer = submit_source_buffer;
    let replacement = Box::into_raw(copy);
    voice_submits().lock().unwrap_or_else(|e| e.into_inner()).insert(raw as usize, (submit, false));
    // SAFETY: XAudio2 exposes a writable COM interface pointer; the copy lives for the process.
    unsafe { *vtable_slot = replacement };
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The PC engine needs COM on the calling thread (`CreateMasteringVoice` fails with
/// `CO_E_NOTINITIALIZED`, 0x800401F0, otherwise). Whether a console title calls `CoInitializeEx`
/// is not documented publicly. To keep such a title working the layer initialises COM
/// (multithreaded) once per thread that reaches it and logs the first time. `RPC_E_CHANGED_MODE`
/// means the thread already has an apartment: fine, nothing to do.
pub fn ensure_com() {
    thread_local! {
        static DONE: Cell<bool> = const { Cell::new(false) };
    }
    if DONE.with(|d| d.replace(true)) {
        return;
    }
    // SAFETY: plain COM initialisation of the calling thread. It is never undone, as in C.
    if unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) } == S_OK {
        bump(Counter::ComInitialised);
        log("COM was not initialised on this thread: initialised (multithreaded) by the layer");
    }
}

/// `XAudio2_9.dll` from System32, else `XAudio2_9Redist.dll`, loaded once.
fn pc_module() -> Option<HMODULE> {
    static MODULE: OnceLock<usize> = OnceLock::new();
    let module = *MODULE.get_or_init(|| {
        let system = wide("XAudio2_9.dll");
        let redist = wide("XAudio2_9Redist.dll");
        // SAFETY: zero terminated names. The first load is limited to System32.
        let loaded = unsafe { LoadLibraryExW(PCWSTR(system.as_ptr()), None, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            .or_else(|_| unsafe { LoadLibraryW(PCWSTR(redist.as_ptr())) });
        match loaded {
            Ok(m) => m.0 as usize,
            Err(_) => {
                // SAFETY: no precondition.
                let error = unsafe { GetLastError() }.0;
                log(&format!("cannot load XAudio2_9.dll (error {error}): the PC engine is missing"));
                0
            }
        }
    });
    (module != 0).then_some(HMODULE(module as *mut c_void))
}

/// The address of `name` in the PC engine.
fn pc_export(name: &str) -> Option<*const c_void> {
    let module = pc_module()?;
    let name = format!("{name}\0");
    // SAFETY: the module stays loaded for the life of the process. The name ends with a zero.
    unsafe { GetProcAddress(module, PCSTR(name.as_ptr())) }.map(|f| f as *const c_void)
}

/// Looks once for the decoder named by `DECANT_XMA2_DECODER`. Found is not enough: nothing
/// consumes decoded PCM yet, so the answer is always false, with the reason logged.
fn xma_decoder_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let Some(path) = std::env::var_os("DECANT_XMA2_DECODER").filter(|p| !p.is_empty() && p.encode_wide().count() < 260) else {
            return false;
        };
        let shown = path.to_string_lossy().into_owned();
        let name: Vec<u16> = path.encode_wide().chain(std::iter::once(0)).collect();
        // SAFETY: a zero terminated path. The DLL is the user's own decoder.
        let module = match unsafe { LoadLibraryW(PCWSTR(name.as_ptr())) } {
            Ok(m) => m,
            Err(_) => {
                // SAFETY: no precondition.
                let error = unsafe { GetLastError() }.0;
                log(&format!("XMA2 decoder {shown}: load failed ({error})"));
                return false;
            }
        };
        // SAFETY: the export of a decoder DLL has the signature of xma2_hook.rs.
        let get = unsafe { GetProcAddress(module, PCSTR(c"DecantXma2GetDecoder".as_ptr().cast())) }
            .map(|f| unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, GetDecoderFn>(f) });
        let mut decoder = DecantXma2Decoder::default();
        // SAFETY: `decoder` is a valid table for the call to fill.
        let filled = get.is_some_and(|get| unsafe { get(&mut decoder) } == 0);
        if !filled || !decoder.usable() {
            log(&format!("XMA2 decoder {shown}: no usable DecantXma2GetDecoder"));
            return false;
        }
        log(&format!("XMA2 decoder {shown} found, but decoded PCM is not yet fed to a source voice (see xma2_hook.h)"));
        false
    })
}

/// The wrapper object. The vtable pointer comes first, as COM requires.
#[repr(C)]
struct Engine {
    vtable: *const IXAudio2_Vtbl,
    inner: IXAudio2,
    refs: AtomicU32,
}

static VTABLE: IXAudio2_Vtbl = IXAudio2_Vtbl {
    base__: IUnknown_Vtbl { QueryInterface: query_interface, AddRef: add_ref, Release: release },
    RegisterForCallbacks: register_for_callbacks,
    UnregisterForCallbacks: unregister_for_callbacks,
    CreateSourceVoice: create_source_voice,
    CreateSubmixVoice: create_submix_voice,
    CreateMasteringVoice: create_mastering_voice,
    StartEngine: start_engine,
    StopEngine: stop_engine,
    CommitChanges: commit_changes,
    GetPerformanceData: get_performance_data,
    SetDebugConfiguration: set_debug_configuration,
};

/// The wrapper behind `this`.
///
/// # Safety
/// `this` is an `Engine` created by `create`, still referenced by the caller.
unsafe fn engine<'a>(this: *mut c_void) -> &'a Engine {
    // SAFETY: as the function's contract says.
    unsafe { &*this.cast::<Engine>() }
}

/// The PC engine's vtable and object.
fn inner(engine: &Engine) -> (&IXAudio2_Vtbl, *mut c_void) {
    (Interface::vtable(&engine.inner), engine.inner.as_raw())
}

fn forward(what: &str) {
    bump(Counter::ForwardedCalls);
    if trace_all() {
        log(&format!("-> {what}"));
    }
}

unsafe extern "system" fn query_interface(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() || iid.is_null() {
        return E_POINTER;
    }
    // SAFETY: COM passes this object, a valid IID and an output location.
    unsafe {
        let iid = *iid;
        if iid == IUnknown::IID || iid == IXAudio2::IID {
            *out = this;
            add_ref(this);
            return S_OK;
        }
        let e = engine(this);
        if iid == IXAudio2Extension::IID {
            // Processor and quantum: the same answer as the PC engine.
            let (vtable, raw) = inner(e);
            return (vtable.base__.QueryInterface)(raw, &iid, out);
        }
        *out = std::ptr::null_mut();
        bump(Counter::UnknownIids);
        log(&format!("QueryInterface: IID {} not answered (console-only interface?)", guid_text(iid.data1, iid.data2, iid.data3, iid.data4)));
        E_NOINTERFACE
    }
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: COM passes a live Engine.
    unsafe { engine(this) }.refs.fetch_add(1, Ordering::SeqCst) + 1
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: COM passes a live Engine.
    let left = unsafe { engine(this) }.refs.fetch_sub(1, Ordering::SeqCst) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone. The Box made by `create` is dropped once, which
        // releases the PC engine.
        drop(unsafe { Box::from_raw(this.cast::<Engine>()) });
    }
    left
}

unsafe extern "system" fn register_for_callbacks(this: *mut c_void, callback: *mut c_void) -> HRESULT {
    forward("RegisterForCallbacks");
    // SAFETY: the title's arguments go to the PC engine unchanged.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.RegisterForCallbacks)(raw, callback)
    }
}

unsafe extern "system" fn unregister_for_callbacks(this: *mut c_void, callback: *mut c_void) {
    forward("UnregisterForCallbacks");
    // SAFETY: the title's arguments go to the PC engine unchanged.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.UnregisterForCallbacks)(raw, callback)
    }
}

unsafe extern "system" fn create_source_voice(
    this: *mut c_void,
    voice: *mut *mut c_void,
    format: *const WAVEFORMATEX,
    flags: u32,
    max_ratio: f32,
    callback: *mut c_void,
    sends: *const XAUDIO2_VOICE_SENDS,
    chain: *const XAUDIO2_EFFECT_CHAIN,
) -> HRESULT {
    ensure_com();
    if !format.is_null() {
        // SAFETY: a non null format is a WAVEFORMATEX (packed: tag at 0, channels at 2, rate at 4).
        let (tag, channels, rate, bits, block) = unsafe {
            let bytes = format.cast::<u8>();
            (bytes.cast::<u16>().read_unaligned(), bytes.add(2).cast::<u16>().read_unaligned(), bytes.add(4).cast::<u32>().read_unaligned(), bytes.add(14).cast::<u16>().read_unaligned(), bytes.add(12).cast::<u16>().read_unaligned())
        };
        log(&source_format_line(tag, channels, rate, bits, block));
        if is_xma(tag) {
            if !xma_decoder_available() {
                bump(Counter::XmaRejected);
                log(&format!("CreateSourceVoice: XMA format tag 0x{tag:04X} ({channels} channels, {rate} Hz) rejected, no software decoder"));
                if !voice.is_null() {
                    // SAFETY: a non null output location of the title.
                    unsafe { *voice = std::ptr::null_mut() };
                }
                return HRESULT(XAUDIO2_E_INVALID_CALL);
            }
        } else if !is_pc_format(tag) {
            bump(Counter::FormatsUnsupported);
            log(&format!("CreateSourceVoice: format tag 0x{tag:04X} is not supported by the PC engine, passed on"));
        } else {
            forward("CreateSourceVoice");
        }
    } else {
        forward("CreateSourceVoice");
    }
    // SAFETY: the title's arguments go to the PC engine unchanged.
    let hr = unsafe {
        let (v, raw) = inner(engine(this));
        (v.CreateSourceVoice)(raw, voice, format, flags, max_ratio, callback, sends, chain)
    };
    if hr.is_ok() && !voice.is_null() {
        // SAFETY: successful creation returns a source voice at the requested output location.
        unsafe { instrument_source_voice(*voice) };
    }
    hr
}

unsafe extern "system" fn create_submix_voice(
    this: *mut c_void,
    voice: *mut *mut c_void,
    channels: u32,
    rate: u32,
    flags: u32,
    stage: u32,
    sends: *const XAUDIO2_VOICE_SENDS,
    chain: *const XAUDIO2_EFFECT_CHAIN,
) -> HRESULT {
    ensure_com();
    forward("CreateSubmixVoice");
    // SAFETY: the title's arguments go to the PC engine unchanged.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.CreateSubmixVoice)(raw, voice, channels, rate, flags, stage, sends, chain)
    }
}

unsafe extern "system" fn create_mastering_voice(
    this: *mut c_void,
    voice: *mut *mut c_void,
    channels: u32,
    rate: u32,
    flags: u32,
    device: PCWSTR,
    chain: *const XAUDIO2_EFFECT_CHAIN,
    category: AUDIO_STREAM_CATEGORY,
) -> HRESULT {
    ensure_com();
    let unknown = flags & !MASTERING_KNOWN;
    if unknown != 0 {
        bump(Counter::MasteringFlagsUnknown);
        log(&format!("CreateMasteringVoice: flag bits 0x{unknown:X} are not defined by the PC engine"));
    } else {
        forward("CreateMasteringVoice");
    }
    // SAFETY: the title's arguments go to the PC engine unchanged, the flags filtered on retry.
    unsafe {
        let (v, raw) = inner(engine(this));
        let mut hr = (v.CreateMasteringVoice)(raw, voice, channels, rate, flags, device, chain, category);
        if hr.is_err() && unknown != 0 {
            bump(Counter::MasteringRetried);
            log(&format!("CreateMasteringVoice: 0x{:08X} with the unknown bits, retrying without them", hr.0 as u32));
            hr = (v.CreateMasteringVoice)(raw, voice, channels, rate, flags & MASTERING_KNOWN, device, chain, category);
        }
        hr
    }
}

unsafe extern "system" fn start_engine(this: *mut c_void) -> HRESULT {
    ensure_com();
    forward("StartEngine");
    // SAFETY: forwarded to the PC engine.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.StartEngine)(raw)
    }
}

unsafe extern "system" fn stop_engine(this: *mut c_void) {
    forward("StopEngine");
    // SAFETY: forwarded to the PC engine.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.StopEngine)(raw)
    }
}

unsafe extern "system" fn commit_changes(this: *mut c_void, set: u32) -> HRESULT {
    forward("CommitChanges");
    // SAFETY: forwarded to the PC engine.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.CommitChanges)(raw, set)
    }
}

unsafe extern "system" fn get_performance_data(this: *mut c_void, data: *mut XAUDIO2_PERFORMANCE_DATA) {
    forward("GetPerformanceData");
    // SAFETY: forwarded to the PC engine.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.GetPerformanceData)(raw, data)
    }
}

unsafe extern "system" fn set_debug_configuration(this: *mut c_void, config: *const XAUDIO2_DEBUG_CONFIGURATION, reserved: *const c_void) {
    forward("SetDebugConfiguration");
    // SAFETY: forwarded to the PC engine.
    unsafe {
        let (v, raw) = inner(engine(this));
        (v.SetDebugConfiguration)(raw, config, reserved)
    }
}

/// `XAudio2Create` through the PC engine, with the console differences handled.
///
/// # Safety
/// `out` is null or a writable location for an interface pointer.
pub unsafe fn create(out: *mut *mut c_void, flags: u32, processor: u32) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable, as the contract says.
    unsafe { *out = std::ptr::null_mut() };
    ensure_com();
    let Some(address) = pc_export("XAudio2Create") else {
        return ERROR_MOD_NOT_FOUND.to_hresult();
    };
    // SAFETY: the PC engine's XAudio2Create has this signature (xaudio2.h).
    let pc_create = unsafe { std::mem::transmute::<*const c_void, CreateFn>(address) };
    let unknown = flags & !CREATE_KNOWN;
    if unknown != 0 {
        bump(Counter::CreateFlagsUnknown);
        log(&format!("XAudio2Create: flag bits 0x{unknown:X} are not defined by the PC engine"));
    }
    if processor != USE_DEFAULT_PROCESSOR {
        bump(Counter::ProcessorsNondefault);
        log(&format!("XAudio2Create: processor mask 0x{processor:08X} (console cores do not map to PC cores)"));
    }
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: `raw` is a valid output location.
    let mut hr = unsafe { pc_create(&mut raw, flags, processor) };
    if hr.is_err() && unknown != 0 {
        bump(Counter::CreateRetried);
        log(&format!("XAudio2Create: 0x{:08X} with the unknown bits, retrying without them", hr.0 as u32));
        // SAFETY: as above.
        hr = unsafe { pc_create(&mut raw, flags & CREATE_KNOWN, processor) };
    }
    if hr.is_err() {
        return hr;
    }
    // SAFETY: on success the PC engine returned an IXAudio2 with one reference, now owned here.
    let Some(inner) = (unsafe { IXAudio2::from_raw_borrowed(&raw) }).map(|_| unsafe { IXAudio2::from_raw(raw) }) else {
        return E_POINTER;
    };
    let engine = Box::new(Engine { vtable: &VTABLE, inner, refs: AtomicU32::new(1) });
    // SAFETY: `out` is writable. The caller owns the one reference.
    unsafe { *out = Box::into_raw(engine).cast() };
    bump(Counter::EnginesCreated);
    S_OK
}

/// `CreateAudioVolumeMeter` and `CreateAudioReverb`, passed to the PC engine.
///
/// # Safety
/// `out` is null or a writable location for an interface pointer.
pub unsafe fn forward_fx(name: &str, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `out` is writable, as the contract says.
    unsafe { *out = std::ptr::null_mut() };
    let Some(address) = pc_export(name) else {
        log(&format!("{name}: not exported by the PC engine"));
        return ERROR_PROC_NOT_FOUND.to_hresult();
    };
    // SAFETY: both effect factories take one output pointer (xaudio2fx.h).
    unsafe { std::mem::transmute::<*const c_void, FxFn>(address)(out) }
}
