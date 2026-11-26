//! The console DirectStorage ABI driven by a spec file (the former `abi.cpp`, format in
//! `docs/DSTORAGE_ABI.md`).
//!
//! A console title calls DirectStorage objects through raw vtables: it knows each interface by its
//! IID and each method by its slot. Neither is in the public documentation, so both come from a
//! spec: per interface, its IIDs and, per slot, the name of an engine handler from the fixed
//! catalogue below (`?` for a slot that only logs and returns `E_NOTIMPL`). The queue engine
//! (`engine`) never sees the console layout: the handlers translate calls into factory, queue,
//! file and status calls.
//!
//! The spec is looked up once, on the first factory request with a non project IID: the path in
//! `DECANT_DSTORAGE_ABI`, else `decant_storage_abi.txt` next to this DLL. Without a valid spec every
//! console IID keeps getting `E_NOINTERFACE`, logged.
#![allow(unsafe_code)]

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{OnceLock, PoisonError, RwLock};

use windows_sys::Win32::Foundation::{HANDLE, HMODULE};
use windows_sys::Win32::Storage::FileSystem::GetFileAttributesW;
use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};

use crate::engine::{self, Factory, Queue, Status};
use crate::log::storage_log;
use crate::decant::*;
use crate::spec::{self, Guid, Role, Spec};

/// A spec interface with its vtable built.
struct Live {
    iface: spec::Iface,
    /// Function addresses: `QueryInterface`, `AddRef`, `Release`, then a handler or a stub per slot.
    vtbl: Vec<usize>,
    slot_names: Vec<String>,
}

struct Loaded {
    spec: Spec,
    live: Vec<Live>,
}

impl Loaded {
    fn find_role(&self, role: Role, iid: Guid) -> Option<&Live> {
        self.live.iter().find(|l| l.iface.role == role && l.iface.accepts(iid))
    }
}

/// A console object handed to the title. The vtable pointer must stay the first member.
#[repr(C)]
struct ConsoleObject {
    vtbl: *const usize,
    refs: AtomicU32,
    live: &'static Live,
    /// Owned reference to the engine object behind it.
    engine: *mut c_void,
}

const _: () = assert!(std::mem::offset_of!(ConsoleObject, vtbl) == 0);

impl Drop for ConsoleObject {
    fn drop(&mut self) {
        // SAFETY: gives back the reference this object owns.
        unsafe { release(self.engine) };
    }
}

static SPEC: OnceLock<Option<&'static Loaded>> = OnceLock::new();
/// Sharded live console objects, to validate pointers passed back without global lock contention.
const REGISTRY_SHARDS: usize = 16;
static REGISTRY: [RwLock<BTreeSet<usize>>; REGISTRY_SHARDS] = [const { RwLock::new(BTreeSet::new()) }; REGISTRY_SHARDS];

fn registry_shard(addr: usize) -> &'static RwLock<BTreeSet<usize>> {
    &REGISTRY[(addr >> 4) & (REGISTRY_SHARDS - 1)]
}

fn registry_insert(addr: usize) {
    registry_shard(addr).write().unwrap_or_else(PoisonError::into_inner).insert(addr);
}

fn registry_remove(addr: usize) {
    registry_shard(addr).write().unwrap_or_else(PoisonError::into_inner).remove(&addr);
}

fn registry_contains(addr: usize) -> bool {
    registry_shard(addr).read().unwrap_or_else(PoisonError::into_inner).contains(&addr)
}

fn guid_of(iid: &GUID) -> Guid {
    Guid(guid_value(iid))
}

fn loaded() -> &'static Loaded {
    // The handlers run only for objects built from a loaded spec.
    SPEC.get().copied().flatten().expect("a console object exists only after its spec loaded")
}

/// The live console object of a given role at `p`, or `None` when `p` is not one of ours.
fn lookup(p: *mut c_void, role: Role) -> Option<&'static ConsoleObject> {
    if p.is_null() || !registry_contains(p as usize) {
        return None;
    }
    // SAFETY: registered pointers are live `ConsoleObject`s, which stay allocated until their last release removes them.
    let object = unsafe { &*(p as *const ConsoleObject) };
    (object.live.iface.role == role).then_some(object)
}

/// Wraps an engine object (taking its reference) in a console object of `live`, if the caller
/// asked for one of its IIDs.
///
/// # Safety
/// `engine` is an engine object whose reference the caller gives up, `out` is writable.
unsafe fn give_out(live: Option<&'static Live>, engine: *mut c_void, riid: &GUID, out: *mut *mut c_void, what: &str) -> HRESULT {
    let Some(live) = live.filter(|l| l.iface.accepts(guid_of(riid))) else {
        storage_log(&format!("console {what} riid={} hr=E_NOINTERFACE\n", guid_text(riid)));
        // SAFETY: the caller gave up this reference.
        unsafe { release(engine) };
        return E_NOINTERFACE;
    };
    let object = Box::into_raw(Box::new(ConsoleObject { vtbl: live.vtbl.as_ptr(), refs: AtomicU32::new(1), live, engine }));
    registry_insert(object as usize);
    // SAFETY: `out` is writable.
    unsafe { *out = object.cast() };
    S_OK
}

fn find_role(role: Role, riid: &GUID) -> Option<&'static Live> {
    loaded().find_role(role, guid_of(riid))
}

// ---- IUnknown, in slots 0 to 2 of every console interface ----

unsafe extern "system" fn query_interface_slot(this: *mut ConsoleObject, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a console object and `iid` is readable, as the caller guarantees.
    unsafe {
        *out = std::ptr::null_mut();
        let object = &*this;
        if guid_value(&*iid) != guid_value(&IID_IUNKNOWN) && !object.live.iface.accepts(guid_of(&*iid)) {
            storage_log(&format!("console {} QueryInterface iid={} hr=E_NOINTERFACE\n", object.live.iface.label, guid_text(&*iid)));
            return E_NOINTERFACE;
        }
        object.refs.fetch_add(1, Ordering::Relaxed);
        *out = this.cast();
    }
    S_OK
}

unsafe extern "system" fn add_ref_slot(this: *mut ConsoleObject) -> u32 {
    // SAFETY: `this` is a console object.
    unsafe { (*this).refs.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release_slot(this: *mut ConsoleObject) -> u32 {
    // SAFETY: `this` is a console object, the last reference removes it from the registry and frees it.
    unsafe {
        let n = (*this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
        if n == 0 {
            registry_remove(this as usize);
            drop(Box::from_raw(this));
        }
        n
    }
}

// ---- Engine handlers. Signatures follow the public PC interfaces (dstorage.h): a spec that maps a
// slot to a handler states that the console method at that slot has this signature. ----

unsafe extern "system" fn factory_create_queue(this: *mut ConsoleObject, desc: *const u8, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a console factory, `riid` readable, `out` writable and `desc` null or a queue description of at least `capacity_offset + 2` bytes.
    unsafe {
        *out = std::ptr::null_mut();
        if desc.is_null() {
            return E_INVALIDARG;
        }
        let capacity = std::ptr::read_unaligned(desc.add(loaded().spec.capacity_offset as usize).cast::<u16>());
        let factory = &*((*this).engine as *const Factory);
        match factory.create_queue(capacity) {
            Err(hr) => {
                storage_log(&format!("console CreateQueue capacity={capacity} hr={:08x}\n", hr as u32));
                hr
            }
            Ok(queue) => give_out(find_role(Role::Queue, &*riid), queue.into_raw().cast(), &*riid, out, "CreateQueue"),
        }
    }
}

unsafe extern "system" fn factory_open_file(this: *mut ConsoleObject, path: *const u16, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a console factory, `riid` readable, `out` writable and `path` null or a zero terminated string.
    unsafe {
        *out = std::ptr::null_mut();
        let factory = &*((*this).engine as *const Factory);
        match factory.open_file(path) {
            Err(hr) => {
                let shown = if path.is_null() { "(null)".to_owned() } else { wide_text(path) };
                storage_log(&format!("console OpenFile path={shown} hr={:08x}\n", hr as u32));
                hr
            }
            Ok(file) => give_out(find_role(Role::File, &*riid), file.into_raw().cast(), &*riid, out, "OpenFile"),
        }
    }
}

unsafe extern "system" fn factory_create_status_array(this: *mut ConsoleObject, count: u32, _name: *const u8, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a console factory, `riid` readable and `out` writable.
    unsafe {
        *out = std::ptr::null_mut();
        let factory = &*((*this).engine as *const Factory);
        match factory.create_status_array(count) {
            Err(hr) => {
                storage_log(&format!("console CreateStatusArray count={count} hr={:08x}\n", hr as u32));
                hr
            }
            Ok(status) => give_out(find_role(Role::Status, &*riid), status.into_raw().cast(), &*riid, out, "CreateStatusArray"),
        }
    }
}

// Tuning calls of the console factory (order 6, 7, 8 on retail titles). This engine has no CPU affinity
// setting, debug layer or staging buffer to size, the values are logged and the calls accepted, as a
// hint a runtime may ignore.

unsafe extern "system" fn factory_set_cpu_affinity(_this: *mut ConsoleObject, mask: u64) -> HRESULT {
    storage_log(&format!("console SetCpuAffinity mask={mask:x}: no effect on this engine\n"));
    S_OK
}

unsafe extern "system" fn factory_set_debug_flags(_this: *mut ConsoleObject, flags: u32) {
    storage_log(&format!("console SetDebugFlags flags={flags:x}: no effect on this engine\n"));
}

unsafe extern "system" fn factory_set_staging_buffer_size(_this: *mut ConsoleObject, size: u32) -> HRESULT {
    storage_log(&format!("console SetStagingBufferSize size={size}: no staging buffer in this engine\n"));
    S_OK
}

/// The engine queue behind a console queue object.
///
/// # Safety
/// `this` is a console queue object.
unsafe fn queue_of<'a>(this: *mut ConsoleObject) -> &'a Queue {
    // SAFETY: the engine pointer of a queue object is a `Queue`.
    unsafe { &*((*this).engine as *const Queue) }
}

// The console request is read with the layout of `DecantConsoleRead` (72 bytes, destination at
// offset 8 BEFORE the source file at offset 32, the reverse of the PC request).
unsafe extern "system" fn queue_enqueue_request(this: *mut ConsoleObject, request: *const u8) -> HRESULT {
    if request.is_null() {
        storage_log("console EnqueueRequest request=null hr=E_POINTER\n");
        return E_POINTER;
    }
    // SAFETY: `this` is a console queue and `request` points to a console request of 72 bytes.
    unsafe {
        let mut r = std::ptr::read_unaligned(request.cast::<DecantConsoleRead>());
        let Some(file) = lookup(r.file, Role::File) else {
            storage_log(&format!("console EnqueueRequest file={:016X} is not a console file of this layer hr=E_INVALIDARG\n", r.file as usize));
            return E_INVALIDARG;
        };
        r.file = file.engine;
        let hr = queue_of(this).enqueue_console(&r);
        if hr < 0 {
            storage_log(&format!("console EnqueueRequest hr={:08x}\n", hr as u32));
        }
        hr
    }
}

unsafe extern "system" fn queue_enqueue_status(this: *mut ConsoleObject, array: *mut c_void, index: u32) -> HRESULT {
    // SAFETY: `this` is a console queue.
    let hr = match lookup(array, Role::Status) {
        Some(status) => unsafe { queue_of(this).enqueue_status(status.engine, index) },
        None => E_INVALIDARG,
    };
    if hr < 0 {
        storage_log(&format!("console EnqueueStatus array={:016X} index={index} hr={:08x}\n", array as usize, hr as u32));
    }
    hr
}

unsafe extern "system" fn queue_enqueue_signal(this: *mut ConsoleObject, fence: *mut c_void, value: u64) -> HRESULT {
    // SAFETY: `this` is a console queue and `fence` is null or an `ID3D12Fence`.
    let hr = unsafe { queue_of(this).enqueue_fence(fence, value) };
    if hr < 0 {
        storage_log(&format!("console EnqueueSignal value={value} hr={:08x}\n", hr as u32));
    }
    hr
}

unsafe extern "system" fn queue_enqueue_set_event(this: *mut ConsoleObject, event: HANDLE) -> HRESULT {
    // SAFETY: `this` is a console queue.
    let hr = unsafe { queue_of(this).enqueue_event(event) };
    if hr < 0 {
        storage_log(&format!("console EnqueueSetEvent hr={:08x}\n", hr as u32));
    }
    hr
}

unsafe extern "system" fn queue_submit(this: *mut ConsoleObject) -> HRESULT {
    // SAFETY: `this` is a console queue.
    let hr = unsafe { queue_of(this).submit() };
    if hr < 0 {
        storage_log(&format!("console Submit hr={:08x}\n", hr as u32));
    }
    hr
}

unsafe extern "system" fn status_is_complete(this: *mut ConsoleObject, index: u32) -> bool {
    // SAFETY: `this` is a console status array, whose engine object is a `Status`.
    unsafe { (*((*this).engine as *const Status)).result(index) != E_PENDING }
}

unsafe extern "system" fn status_get_hresult(this: *mut ConsoleObject, index: u32) -> HRESULT {
    // SAFETY: as above.
    unsafe { (*((*this).engine as *const Status)).result(index) }
}

/// A slot the spec leaves as `?`: logged with its interface and slot, `E_NOTIMPL`. Extra arguments
/// a caller passes are ignored, which the x64 calling convention allows.
unsafe extern "system" fn unimplemented<const SLOT: usize>(this: *mut ConsoleObject) -> HRESULT {
    // SAFETY: `this` is a console object.
    let object = unsafe { &*this };
    decant_log::calls::record_with(decant_log::calls::Kind::NotImplemented, || format!("console {} slot {SLOT} ({})", object.live.iface.label, object.live.slot_names[SLOT]));
    storage_log(&format!("console {} slot {SLOT} ({}) not implemented hr=E_NOTIMPL\n", object.live.iface.label, object.live.slot_names[SLOT]));
    E_NOTIMPL
}

type Stub = unsafe extern "system" fn(*mut ConsoleObject) -> HRESULT;

macro_rules! stubs {
    ($($slot:literal)*) => { [$(unimplemented::<$slot> as Stub),*] };
}

static STUBS: [Stub; spec::MAX_SLOTS as usize] = stubs!(
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
);

/// The function a handler name stands for.
fn handler_address(name: &str) -> Option<usize> {
    Some(match name {
        "factory.create_queue" => factory_create_queue as *const () as usize,
        "factory.open_file" => factory_open_file as *const () as usize,
        "factory.create_status_array" => factory_create_status_array as *const () as usize,
        "factory.set_cpu_affinity" => factory_set_cpu_affinity as *const () as usize,
        "factory.set_debug_flags" => factory_set_debug_flags as *const () as usize,
        "factory.set_staging_buffer_size" => factory_set_staging_buffer_size as *const () as usize,
        "queue.enqueue_request" => queue_enqueue_request as *const () as usize,
        "queue.enqueue_status" => queue_enqueue_status as *const () as usize,
        "queue.enqueue_signal" => queue_enqueue_signal as *const () as usize,
        "queue.enqueue_set_event" => queue_enqueue_set_event as *const () as usize,
        "queue.submit" => queue_submit as *const () as usize,
        "status.is_complete" => status_is_complete as *const () as usize,
        "status.get_hresult" => status_get_hresult as *const () as usize,
        _ => return None,
    })
}

fn build_vtables(spec: &Spec) -> Vec<Live> {
    spec.ifaces
        .iter()
        .map(|iface| {
            let size = iface.methods.iter().map(|&(slot, _)| slot as usize + 1).fold(3, usize::max);
            let mut vtbl = vec![0usize; size];
            let mut slot_names = vec!["?".to_owned(); size];
            vtbl[0] = query_interface_slot as *const () as usize;
            vtbl[1] = add_ref_slot as *const () as usize;
            vtbl[2] = release_slot as *const () as usize;
            slot_names[0] = "QueryInterface".to_owned();
            slot_names[1] = "AddRef".to_owned();
            slot_names[2] = "Release".to_owned();
            for (i, entry) in vtbl.iter_mut().enumerate().skip(3) {
                *entry = STUBS[i] as usize;
            }
            for (slot, name) in &iface.methods {
                let handler = name.split(' ').next().unwrap_or("");
                slot_names[*slot as usize] = name.clone();
                if let Some(address) = handler_address(handler) {
                    vtbl[*slot as usize] = address;
                }
            }
            Live { iface: iface.clone(), vtbl, slot_names }
        })
        .collect()
}

/// A zero terminated UTF-16 string, as text for the log.
///
/// # Safety
/// `p` points to a zero terminated string.
unsafe fn wide_text(p: *const u16) -> String {
    let mut len = 0;
    // SAFETY: the string is zero terminated, as the caller guarantees.
    while unsafe { *p.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: `len` units were just read.
    String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(p, len) })
}

/// The path of the spec: `DECANT_DSTORAGE_ABI`, else `decant_storage_abi.txt` next to this DLL. Empty
/// when this DLL's own path cannot be read.
fn spec_path() -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    if let Some(env) = std::env::var_os("DECANT_DSTORAGE_ABI") {
        let units: Vec<u16> = env.encode_wide().collect();
        if !units.is_empty() && units.len() < 32768 {
            return units;
        }
    }
    let mut module: HMODULE = std::ptr::null_mut();
    let mut buffer = vec![0u16; 32768];
    // SAFETY: the address is one of this module's own functions, `module` and `buffer` are writable for their sizes.
    let len = unsafe {
        if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, spec_path as *const () as *const u16, &mut module) == 0 {
            return Vec::new();
        }
        GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) as usize
    };
    if len == 0 || len >= buffer.len() {
        return Vec::new();
    }
    buffer.truncate(len);
    let keep = buffer.iter().rposition(|&c| c == u16::from(b'\\') || c == u16::from(b'/')).map_or(0, |i| i + 1);
    buffer.truncate(keep);
    buffer.extend("decant_storage_abi.txt".encode_utf16());
    buffer
}

fn load_spec() -> Option<&'static Loaded> {
    use std::os::windows::ffi::OsStringExt;
    let path = spec_path();
    let shown = String::from_utf16_lossy(&path);
    let mut zero_terminated = path.clone();
    zero_terminated.push(0);
    // SAFETY: a zero terminated path.
    if path.is_empty() || unsafe { GetFileAttributesW(zero_terminated.as_ptr()) } == u32::MAX {
        storage_log(&format!("console ABI: no spec at {shown}; console IIDs are refused\n"));
        return None;
    }
    let parsed = match std::fs::read(std::ffi::OsString::from_wide(&path)) {
        Ok(text) => spec::parse_spec(&text),
        Err(_) => Err("cannot open".to_owned()),
    };
    let spec = match parsed {
        Ok(spec) => spec,
        Err(error) => {
            storage_log(&format!("console ABI: spec {shown} rejected, {error}; console IIDs are refused\n"));
            return None;
        }
    };
    let live = build_vtables(&spec);
    if let Some(f) = spec.ifaces.iter().find(|i| i.role == Role::Factory) {
        storage_log(&format!(
            "console ABI: {shown}, {} interfaces, factory '{}' status={}{}{}\n",
            spec.ifaces.len(),
            f.label,
            f.status,
            if f.status == "verified" { "" } else { " (not verified on console)" },
            if f.iids.is_empty() { "; factory IID unknown, fill it in from the executable" } else { "" },
        ));
    }
    Some(Box::leak(Box::new(Loaded { spec, live })))
}

/// A console factory when `iid` is a factory IID of the loaded spec, `E_NOINTERFACE` (logged)
/// otherwise.
///
/// # Safety
/// `iid` is readable and `out` writable.
pub unsafe fn console_get_factory(iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: `out` is writable and `iid` readable, as the caller guarantees.
    unsafe {
        *out = std::ptr::null_mut();
        let iid = &*iid;
        let live = SPEC.get_or_init(load_spec).and_then(|l| l.find_role(Role::Factory, guid_of(iid)));
        let Some(live) = live else {
            storage_log(&format!("DStorageGetFactory iid={} hr=E_NOINTERFACE\n", guid_text(iid)));
            return E_NOINTERFACE;
        };
        let engine = match engine::engine_factory() {
            Ok(engine) => engine,
            Err(hr) => return hr,
        };
        storage_log(&format!("DStorageGetFactory console iid={} interface={} status={}\n", guid_text(iid), live.iface.label, live.iface.status));
        give_out(Some(live), engine.into_raw().cast(), iid, out, "DStorageGetFactory")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharded_registry_tracks_and_removes_addresses() {
        let addrs: Vec<usize> = (0..64).map(|i| 0x1000_0000usize + i * 16).collect();
        for &a in &addrs {
            registry_insert(a);
        }
        for &a in &addrs {
            assert!(registry_contains(a));
        }
        for &a in &addrs {
            registry_remove(a);
            assert!(!registry_contains(a));
        }
        assert!(!registry_contains(0));
    }
}
