//! Keeps the kernel from writing into title pages it cannot write. A pool page the title never touched
//! is not committed yet, and while the GPU mirror tracks dirty pages a page that is clean is read only:
//! a `ReadFile` into either fails with `ERROR_NOACCESS` instead of faulting. Installed when the engine
//! starts. The title imports `ReadFile`, `ReadFileEx` and `ReadFileScatter` from the PC
//! system DLLs directly, nothing of ours sits in between. So this module patches the import address
//! tables of the title executable and of the modules loaded from its folder: the entries of those
//! three functions point to wrappers that call `Engine::prepare_write` on the buffer first (the
//! pages are committed, and become dirty and writable for a while) and then the original.
//!
//! `NtReadFile` called directly, IoRing, `DeviceIoControl` and socket receives are not covered. A
//! wrapper that still sees `ERROR_NOACCESS` logs it with the address.
#![allow(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use windows_sys::Win32::Foundation::{ERROR_IO_PENDING, ERROR_NOACCESS, GetLastError, HMODULE, SetLastError};
use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::engine::Engine;

/// The size of a segment of `ReadFileScatter`: one system page.
const SEGMENT_BYTES: usize = 4096;

type ReadFileFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut u32, *mut c_void) -> BOOL;
type ReadFileExFn = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, *mut c_void) -> BOOL;
type ReadFileScatterFn = unsafe extern "system" fn(*mut c_void, *const u64, u32, *mut u32, *mut c_void) -> BOOL;
type LoadLibraryExWFn = unsafe extern "system" fn(*const u16, *mut c_void, u32) -> HMODULE;
type GetOverlappedResultFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u32, BOOL) -> BOOL;
type GetOverlappedResultExFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u32, u32, BOOL) -> BOOL;
type GetQueuedCompletionStatusFn = unsafe extern "system" fn(*mut c_void, *mut u32, *mut usize, *mut *mut c_void, u32) -> BOOL;
type GetQueuedCompletionStatusExFn = unsafe extern "system" fn(*mut c_void, *mut OverlappedEntry, u32, *mut u32, u32, BOOL) -> BOOL;
type WaitForSingleObjectFn = unsafe extern "system" fn(*mut c_void, u32) -> u32;
type WaitForSingleObjectExFn = unsafe extern "system" fn(*mut c_void, u32, BOOL) -> u32;
type WaitForMultipleObjectsFn = unsafe extern "system" fn(u32, *const *mut c_void, BOOL, u32) -> u32;
type WaitForMultipleObjectsExFn = unsafe extern "system" fn(u32, *const *mut c_void, BOOL, u32, BOOL) -> u32;

#[repr(C)]
struct OverlappedEntry {
    key: usize,
    overlapped: *mut c_void,
    internal: usize,
    bytes: u32,
}

const MAX_PENDING_OVERLAPPED: usize = 4096;

#[derive(Default)]
struct PendingOverlapped {
    map: HashMap<usize, (usize, Vec<(usize, usize)>)>,
    events: HashMap<usize, Vec<usize>>,
    order: VecDeque<usize>,
}

/// The functions that were in the tables before the patch, 0 until a table was patched.
static ORIGINAL: [AtomicUsize; 12] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
/// The modules whose tables were looked at.
static DONE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
static XGR_PATH: OnceLock<Option<Vec<u16>>> = OnceLock::new();
static PENDING_OVERLAPPED: Mutex<Option<PendingOverlapped>> = Mutex::new(None);

fn xgr_path() -> Option<&'static [u16]> {
    XGR_PATH
        .get_or_init(|| {
            let val = std::env::var("DECANT_XGR_PATH").ok()?;
            let trimmed = val.trim();
            if trimmed.is_empty() {
                return None;
            }
            let mut wide: Vec<u16> = trimmed.encode_utf16().collect();
            wide.push(0);
            Some(wide)
        })
        .as_deref()
}

/// Whether `DECANT_XGR_PATH` is set for in-process `LoadLibraryExW` redirection.
pub fn has_xgr_redirect() -> bool {
    xgr_path().is_some()
}

/// Whether `name` is a load of `xgameruntime.dll`.
pub(crate) fn is_xgameruntime_load(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    lower == "xgameruntime.dll" || lower.ends_with("\\xgameruntime.dll") || lower.ends_with("/xgameruntime.dll")
}

/// Whether an import table named `dll` can hold the file or library loader functions.
fn wanted_dll(dll: &str) -> bool {
    let lower = dll.to_ascii_lowercase();
    lower == "kernel32.dll"
        || lower == "kernelbase.dll"
        || (lower.starts_with("api-ms-win-core-file-") && lower.ends_with(".dll"))
        || (lower.starts_with("api-ms-win-core-libraryloader-") && lower.ends_with(".dll"))
        || (lower.starts_with("api-ms-win-core-io-") && lower.ends_with(".dll"))
        || (lower.starts_with("api-ms-win-core-synch-") && lower.ends_with(".dll"))
}

/// The index in `ORIGINAL` of an imported function, when it is one of the hooked functions.
fn wanted_function(name: &str) -> Option<usize> {
    [
        "ReadFile",
        "ReadFileEx",
        "ReadFileScatter",
        "LoadLibraryExW",
        "GetOverlappedResult",
        "GetOverlappedResultEx",
        "GetQueuedCompletionStatus",
        "GetQueuedCompletionStatusEx",
        "WaitForSingleObject",
        "WaitForSingleObjectEx",
        "WaitForMultipleObjects",
        "WaitForMultipleObjectsEx",
    ]
    .iter()
    .position(|&n| n == name)
}

fn remove_event_entry(events: &mut HashMap<usize, Vec<usize>>, ev: usize, ov: usize) {
    if ev == 0 {
        return;
    }
    if let Some(list) = events.get_mut(&ev) {
        list.retain(|&x| x != ov);
        if list.is_empty() {
            events.remove(&ev);
        }
    }
}

fn record_pending_overlapped_with_event(ov: usize, event: usize, runs: Vec<(usize, usize)>) {
    if ov == 0 || runs.is_empty() {
        return;
    }
    let mut guard = PENDING_OVERLAPPED.lock().unwrap_or_else(PoisonError::into_inner);
    let pending = guard.get_or_insert_with(PendingOverlapped::default);
    if let Some((old_ev, _)) = pending.map.insert(ov, (event, runs)) {
        remove_event_entry(&mut pending.events, old_ev, ov);
    } else {
        pending.order.push_back(ov);
    }
    if event != 0 {
        let list = pending.events.entry(event).or_default();
        if !list.contains(&ov) {
            list.push(ov);
        }
    }
    while pending.map.len() > MAX_PENDING_OVERLAPPED {
        let Some(oldest) = pending.order.pop_front() else { break };
        if let Some((old_ev, _)) = pending.map.remove(&oldest) {
            remove_event_entry(&mut pending.events, old_ev, oldest);
        }
    }
}

fn take_pending_overlapped(ov: usize) -> Option<Vec<(usize, usize)>> {
    if ov == 0 {
        return None;
    }
    let mut guard = PENDING_OVERLAPPED.lock().unwrap_or_else(PoisonError::into_inner);
    let pending = guard.as_mut()?;
    let (ev, runs) = pending.map.remove(&ov)?;
    remove_event_entry(&mut pending.events, ev, ov);
    pending.order.retain(|&x| x != ov);
    Some(runs)
}

fn take_pending_by_event(event: usize) -> Vec<Vec<(usize, usize)>> {
    if event == 0 {
        return Vec::new();
    }
    let mut guard = PENDING_OVERLAPPED.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(pending) = guard.as_mut() else { return Vec::new() };
    let Some(ovs) = pending.events.remove(&event) else { return Vec::new() };
    let mut out = Vec::with_capacity(ovs.len());
    for ov in ovs {
        if let Some((_, runs)) = pending.map.remove(&ov) {
            pending.order.retain(|&x| x != ov);
            out.push(runs);
        }
    }
    out
}

fn complete_overlapped(ov: usize) {
    let Some(runs) = take_pending_overlapped(ov) else { return };
    if let Some(engine) = Engine::started() {
        engine.clear_io_hold_runs(&runs);
    }
}

fn complete_event(event: usize) {
    let list = take_pending_by_event(event);
    if list.is_empty() {
        return;
    }
    if let Some(engine) = Engine::started() {
        for runs in list {
            engine.clear_io_hold_runs(&runs);
        }
    }
}

fn overlapped_event(overlapped: *mut c_void) -> usize {
    if overlapped.is_null() {
        return 0;
    }
    // SAFETY: `overlapped` is a valid non-null Win32 `OVERLAPPED` struct (hEvent is at offset 24 on x64).
    let raw = unsafe { overlapped.cast::<usize>().add(3).read_unaligned() };
    raw & !1
}

fn finish_or_pend_read(engine: Option<&Engine>, overlapped: *mut c_void, result: BOOL, runs: Vec<(usize, usize)>) {
    let Some(e) = engine else { return };
    // SAFETY: GetLastError and SetLastError only touch the thread's error code, kept for the caller.
    unsafe {
        let err = GetLastError();
        if overlapped.is_null() || result != 0 || err != ERROR_IO_PENDING {
            e.finish_io_write_runs(&runs);
        } else {
            let ev = overlapped_event(overlapped);
            e.convert_pin_to_hold_runs(&runs);
            record_pending_overlapped_with_event(overlapped as usize, ev, runs);
        }
        SetLastError(err);
    }
}

/// Coalesces contiguous `SEGMENT_BYTES` scatter segments into maximal `(base, run_len)` runs.
pub(crate) fn coalesce_scatter_segments(segments: &[u64], len: usize) -> Vec<(usize, usize)> {
    if len == 0 || segments.is_empty() {
        return Vec::new();
    }
    let count = len.div_ceil(SEGMENT_BYTES).min(segments.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut remaining = len;
    for &seg in &segments[..count] {
        let seg_len = remaining.min(SEGMENT_BYTES);
        let addr = seg as usize;
        match runs.last_mut() {
            Some((base, run_len)) if base.checked_add(*run_len) == Some(addr) => *run_len += seg_len,
            _ => runs.push((addr, seg_len)),
        }
        remaining -= seg_len;
    }
    runs
}

/// Logs a read that still failed on a protected page, keeping the error code for the caller.
fn check(result: BOOL, buffer: *const c_void, len: usize) {
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    if result != 0 {
        return;
    }
    // SAFETY: GetLastError has no precondition.
    let error = unsafe { GetLastError() };
    if error == ERROR_NOACCESS && SEEN.fetch_add(1, Ordering::Relaxed) < 32 {
        let first = Engine::started().and_then(|e| e.first_unwritable(buffer as usize, len)).unwrap_or_else(|| "every page is writable now".into());
        decant_log::trace(&format!("[xmem] ReadFile : ERROR_NOACCESS sur 0x{:x}, {len} octets (page protegee que la garde n'a pas ouverte) ; {first}\r\n", buffer as usize));
        // SAFETY: SetLastError only stores the thread's error code.
        unsafe { SetLastError(error) };
    }
}

unsafe extern "system" fn read_file(file: *mut c_void, buffer: *mut c_void, len: u32, read: *mut u32, overlapped: *mut c_void) -> BOOL {
    let engine = Engine::started();
    let runs = if len != 0 { vec![(buffer as usize, len as usize)] } else { Vec::new() };
    if let Some(e) = engine {
        e.begin_io_write_runs(&runs);
    }
    // SAFETY: the wrapper is only installed after `ORIGINAL[0]` holds the function it replaces, and the arguments are the caller's.
    let result = unsafe { std::mem::transmute::<usize, ReadFileFn>(ORIGINAL[0].load(Ordering::Acquire))(file, buffer, len, read, overlapped) };
    check(result, buffer, len as usize);
    finish_or_pend_read(engine, overlapped, result, runs);
    result
}

unsafe extern "system" fn read_file_ex(file: *mut c_void, buffer: *mut c_void, len: u32, overlapped: *mut c_void, routine: *mut c_void) -> BOOL {
    let engine = Engine::started();
    let runs = if len != 0 { vec![(buffer as usize, len as usize)] } else { Vec::new() };
    if let Some(e) = engine {
        e.begin_io_write_runs(&runs);
    }
    // SAFETY: as in `read_file`, with `ORIGINAL[1]`.
    let result = unsafe { std::mem::transmute::<usize, ReadFileExFn>(ORIGINAL[1].load(Ordering::Acquire))(file, buffer, len, overlapped, routine) };
    check(result, buffer, len as usize);
    if let Some(e) = engine {
        // SAFETY: preserves last error across pin cleanup.
        unsafe {
            let err = GetLastError();
            if result == 0 {
                e.finish_io_write_runs(&runs);
            } else {
                e.convert_pin_to_hold_runs(&runs);
            }
            SetLastError(err);
        }
    }
    result
}

unsafe extern "system" fn read_file_scatter(file: *mut c_void, segments: *const u64, len: u32, reserved: *mut u32, overlapped: *mut c_void) -> BOOL {
    let engine = Engine::started();
    let runs = if !segments.is_null() && len != 0 {
        let count = (len as usize).div_ceil(SEGMENT_BYTES);
        // SAFETY: the caller's array has one element for each segment of `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(segments, count) };
        let runs = coalesce_scatter_segments(slice, len as usize);
        if let Some(e) = engine {
            e.begin_io_write_runs(&runs);
        }
        runs
    } else {
        Vec::new()
    };
    // SAFETY: as in `read_file`, with `ORIGINAL[2]`.
    let result = unsafe { std::mem::transmute::<usize, ReadFileScatterFn>(ORIGINAL[2].load(Ordering::Acquire))(file, segments, len, reserved, overlapped) };
    check(result, segments.cast(), len as usize);
    finish_or_pend_read(engine, overlapped, result, runs);
    result
}

unsafe extern "system" fn load_library_ex_w(lib_name: *const u16, file: *mut c_void, flags: u32) -> HMODULE {
    let target = if !lib_name.is_null()
        && let Some(redirect) = xgr_path()
    {
        let mut len = 0usize;
        // SAFETY: `lib_name` is a NUL-terminated UTF-16 string passed to `LoadLibraryExW`.
        while len < 1024 && unsafe { *lib_name.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `len` UTF-16 units were just checked before the terminator.
        let requested = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(lib_name, len) });
        if is_xgameruntime_load(&requested) {
            redirect.as_ptr()
        } else {
            lib_name
        }
    } else {
        lib_name
    };
    // SAFETY: `ORIGINAL[3]` holds the original `LoadLibraryExW` address before this wrapper was installed.
    let loaded = unsafe { std::mem::transmute::<usize, LoadLibraryExWFn>(ORIGINAL[3].load(Ordering::Acquire))(target, file, flags) };
    if !loaded.is_null() && Engine::started().is_some() {
        install();
    }
    loaded
}

unsafe extern "system" fn get_overlapped_result(file: *mut c_void, overlapped: *mut c_void, bytes: *mut u32, wait: BOOL) -> BOOL {
    // SAFETY: `ORIGINAL[4]` holds the original `GetOverlappedResult`.
    let result = unsafe { std::mem::transmute::<usize, GetOverlappedResultFn>(ORIGINAL[4].load(Ordering::Acquire))(file, overlapped, bytes, wait) };
    if result != 0 {
        // SAFETY: preserves the caller's last error across completion cleanup.
        unsafe {
            let err = GetLastError();
            complete_overlapped(overlapped as usize);
            SetLastError(err);
        }
    }
    result
}

unsafe extern "system" fn get_overlapped_result_ex(file: *mut c_void, overlapped: *mut c_void, bytes: *mut u32, timeout_ms: u32, alertable: BOOL) -> BOOL {
    // SAFETY: `ORIGINAL[5]` holds the original `GetOverlappedResultEx`.
    let result = unsafe { std::mem::transmute::<usize, GetOverlappedResultExFn>(ORIGINAL[5].load(Ordering::Acquire))(file, overlapped, bytes, timeout_ms, alertable) };
    if result != 0 {
        // SAFETY: preserves the caller's last error across completion cleanup.
        unsafe {
            let err = GetLastError();
            complete_overlapped(overlapped as usize);
            SetLastError(err);
        }
    }
    result
}

unsafe extern "system" fn get_queued_completion_status(port: *mut c_void, bytes: *mut u32, key: *mut usize, overlapped: *mut *mut c_void, timeout_ms: u32) -> BOOL {
    // SAFETY: `ORIGINAL[6]` holds the original `GetQueuedCompletionStatus`.
    let result = unsafe { std::mem::transmute::<usize, GetQueuedCompletionStatusFn>(ORIGINAL[6].load(Ordering::Acquire))(port, bytes, key, overlapped, timeout_ms) };
    if !overlapped.is_null() {
        // SAFETY: `overlapped` is the caller's output pointer populated by `GetQueuedCompletionStatus`.
        let ov = unsafe { *overlapped } as usize;
        if ov != 0 {
            // SAFETY: preserves the caller's last error across completion cleanup.
            unsafe {
                let err = GetLastError();
                complete_overlapped(ov);
                SetLastError(err);
            }
        }
    }
    result
}

unsafe extern "system" fn get_queued_completion_status_ex(port: *mut c_void, entries: *mut OverlappedEntry, count: u32, removed: *mut u32, timeout_ms: u32, alertable: BOOL) -> BOOL {
    // SAFETY: `ORIGINAL[7]` holds the original `GetQueuedCompletionStatusEx`.
    let result = unsafe { std::mem::transmute::<usize, GetQueuedCompletionStatusExFn>(ORIGINAL[7].load(Ordering::Acquire))(port, entries, count, removed, timeout_ms, alertable) };
    if result != 0 && !entries.is_null() && !removed.is_null() {
        // SAFETY: `removed` entries in `entries` were written by `GetQueuedCompletionStatusEx`.
        unsafe {
            let err = GetLastError();
            let n = (*removed).min(count) as usize;
            for i in 0..n {
                complete_overlapped((*entries.add(i)).overlapped as usize);
            }
            SetLastError(err);
        }
    }
    result
}

unsafe extern "system" fn wait_for_single_object(handle: *mut c_void, ms: u32) -> u32 {
    // SAFETY: `ORIGINAL[8]` holds the original `WaitForSingleObject`.
    let result = unsafe { std::mem::transmute::<usize, WaitForSingleObjectFn>(ORIGINAL[8].load(Ordering::Acquire))(handle, ms) };
    if result == 0 {
        complete_event(handle as usize);
    }
    result
}

unsafe extern "system" fn wait_for_single_object_ex(handle: *mut c_void, ms: u32, alertable: BOOL) -> u32 {
    // SAFETY: `ORIGINAL[9]` holds the original `WaitForSingleObjectEx`.
    let result = unsafe { std::mem::transmute::<usize, WaitForSingleObjectExFn>(ORIGINAL[9].load(Ordering::Acquire))(handle, ms, alertable) };
    if result == 0 {
        complete_event(handle as usize);
    }
    result
}

fn complete_wait_multiple(count: u32, handles: *const *mut c_void, wait_all: BOOL, result: u32) {
    if handles.is_null() {
        return;
    }
    if wait_all != 0 && result == 0 {
        for i in 0..count as usize {
            // SAFETY: `handles` has `count` entries.
            complete_event(unsafe { *handles.add(i) } as usize);
        }
    } else if wait_all == 0 && result < count {
        // SAFETY: `result` is a valid index `< count` in `handles`.
        complete_event(unsafe { *handles.add(result as usize) } as usize);
    }
}

unsafe extern "system" fn wait_for_multiple_objects(count: u32, handles: *const *mut c_void, wait_all: BOOL, ms: u32) -> u32 {
    // SAFETY: `ORIGINAL[10]` holds the original `WaitForMultipleObjects`.
    let result = unsafe { std::mem::transmute::<usize, WaitForMultipleObjectsFn>(ORIGINAL[10].load(Ordering::Acquire))(count, handles, wait_all, ms) };
    complete_wait_multiple(count, handles, wait_all, result);
    result
}

unsafe extern "system" fn wait_for_multiple_objects_ex(count: u32, handles: *const *mut c_void, wait_all: BOOL, ms: u32, alertable: BOOL) -> u32 {
    // SAFETY: `ORIGINAL[11]` holds the original `WaitForMultipleObjectsEx`.
    let result = unsafe { std::mem::transmute::<usize, WaitForMultipleObjectsExFn>(ORIGINAL[11].load(Ordering::Acquire))(count, handles, wait_all, ms, alertable) };
    complete_wait_multiple(count, handles, wait_all, result);
    result
}

fn wrapper(index: usize) -> usize {
    [
        read_file as *const () as usize,
        read_file_ex as *const () as usize,
        read_file_scatter as *const () as usize,
        load_library_ex_w as *const () as usize,
        get_overlapped_result as *const () as usize,
        get_overlapped_result_ex as *const () as usize,
        get_queued_completion_status as *const () as usize,
        get_queued_completion_status_ex as *const () as usize,
        wait_for_single_object as *const () as usize,
        wait_for_single_object_ex as *const () as usize,
        wait_for_multiple_objects as *const () as usize,
        wait_for_multiple_objects_ex as *const () as usize,
    ][index]
}

/// A zero terminated name at `address` as text, `None` when it is not UTF 8 or runs on.
///
/// # Safety
/// `address` is readable up to the end of the name.
unsafe fn c_name(address: *const u8) -> Option<String> {
    let mut len = 0;
    // SAFETY: by the contract above.
    while len < 256 && unsafe { *address.add(len) } != 0 {
        len += 1;
    }
    // SAFETY: the `len` bytes before the terminator were just read.
    String::from_utf8(unsafe { std::slice::from_raw_parts(address, len) }.to_vec()).ok()
}

/// Points the entries of `ReadFile`, `ReadFileEx` and `ReadFileScatter` in the import table of `module`
/// to the wrappers. Returns how many entries were patched.
///
/// # Safety
/// `module` is the base of a loaded 64 bit image.
unsafe fn patch(module: usize) -> usize {
    let at = |rva: u32| (module + rva as usize) as *const u8;
    let mut patched = 0;
    // SAFETY: the headers and the import table of a loaded image are readable. Each read stays inside
    // the structures the headers describe. An entry is written only after its page was made writable.
    unsafe {
        if at(0).cast::<u16>().read_unaligned() != 0x5A4D {
            return 0;
        }
        let nt = at(at(0x3C).cast::<u32>().read_unaligned());
        if nt.cast::<u32>().read_unaligned() != 0x0000_4550 || nt.add(0x18).cast::<u16>().read_unaligned() != 0x20B {
            return 0;
        }
        let imports = nt.add(0x90).cast::<u32>().read_unaligned();
        if imports == 0 {
            return 0;
        }
        let mut descriptor = at(imports);
        loop {
            let (names, name, addresses) = (descriptor.cast::<u32>().read_unaligned(), descriptor.add(12).cast::<u32>().read_unaligned(), descriptor.add(16).cast::<u32>().read_unaligned());
            if name == 0 {
                break;
            }
            if names != 0 && c_name(at(name)).is_some_and(|dll| wanted_dll(&dll)) {
                let mut k = 0;
                loop {
                    let entry = at(names).add(k * 8).cast::<u64>().read_unaligned();
                    if entry == 0 {
                        break;
                    }
                    // The high bit marks an import by ordinal: no name to compare.
                    if entry >> 63 == 0
                        && let Some(index) = c_name(at(entry as u32 + 2)).as_deref().and_then(wanted_function)
                    {
                        let slot = at(addresses).add(k * 8) as *mut usize;
                        let current = slot.read_unaligned();
                        if current != wrapper(index) {
                            let mut old = 0u32;
                            if VirtualProtect(slot.cast(), 8, PAGE_READWRITE, &mut old) != 0 {
                                // The first table patched tells the original. The others hold the same function.
                                let _ = ORIGINAL[index].compare_exchange(0, current, Ordering::AcqRel, Ordering::Acquire);
                                slot.write_unaligned(wrapper(index));
                                VirtualProtect(slot.cast(), 8, old, &mut old);
                                patched += 1;
                            }
                        }
                    }
                    k += 1;
                }
            }
            descriptor = descriptor.add(20);
        }
    }
    patched
}

/// The text of the path of `module`, lower case.
fn module_path(module: HMODULE) -> String {
    let mut buffer = [0u16; 1024];
    // SAFETY: the buffer holds 1024 wide characters, which the call is told.
    let n = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
    String::from_utf16_lossy(&buffer[..n.min(buffer.len())]).to_lowercase()
}

/// The folder of a path, with its trailing separator.
fn folder(path: &str) -> &str {
    path.rfind(['\\', '/']).map_or("", |i| &path[..=i])
}

/// Patches the import tables of the title executable and of every module loaded from its folder
/// that was not looked at yet. Cheap when nothing new is loaded: one module list.
pub fn install() {
    // SAFETY: a null name asks for the executable of the process.
    let exe = unsafe { GetModuleHandleW(std::ptr::null()) };
    if exe.is_null() {
        return;
    }
    let home = folder(&module_path(exe)).to_owned();
    let mut modules = vec![std::ptr::null_mut::<c_void>(); 1024];
    let mut needed = 0u32;
    // SAFETY: the array has the room in bytes that the call is told.
    let listed = unsafe { K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr().cast(), (modules.len() * size_of::<usize>()) as u32, &mut needed) };
    if listed == 0 {
        return;
    }
    modules.truncate((needed as usize / size_of::<usize>()).min(modules.len()));
    let mut done = DONE.lock().unwrap_or_else(PoisonError::into_inner);
    let done = done.get_or_insert_with(HashSet::new);
    let mut total = 0;
    for module in modules {
        let path = module_path(module);
        let ours = module as usize == this_module();
        if ours || !path.starts_with(&home) || !done.insert(module as usize) {
            continue;
        }
        // SAFETY: `module` was listed as loaded, it is a 64 bit image or `patch` finds out it is not.
        total += unsafe { patch(module as usize) };
    }
    if total > 0 {
        decant_log::trace(&format!("[xmem] garde des lectures : {total} entrees ReadFile, ReadFileEx ou ReadFileScatter redirigees\r\n"));
    }
}

/// This module's own base, which is never patched: its calls to `ReadFile` come with the engine lock held.
fn this_module() -> usize {
    unsafe extern "C" {
        static __ImageBase: u8;
    }
    (&raw const __ImageBase) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_imports_of_the_system_dlls_are_wanted() {
        assert!(wanted_dll("KERNEL32.dll") && wanted_dll("kernelbase.dll") && wanted_dll("api-ms-win-core-file-l1-1-0.dll") && wanted_dll("API-MS-WIN-CORE-FILE-L2-1-1.DLL"));
        assert!(!wanted_dll("ntdll.dll") && !wanted_dll("d3d12.dll") && !wanted_dll("api-ms-win-core-file"));
        assert_eq!(wanted_function("ReadFile"), Some(0));
        assert_eq!(wanted_function("ReadFileEx"), Some(1));
        assert_eq!(wanted_function("ReadFileScatter"), Some(2));
        assert_eq!(wanted_function("LoadLibraryExW"), Some(3));
        assert_eq!(wanted_function("GetOverlappedResult"), Some(4));
        assert_eq!(wanted_function("GetOverlappedResultEx"), Some(5));
        assert_eq!(wanted_function("GetQueuedCompletionStatus"), Some(6));
        assert_eq!(wanted_function("GetQueuedCompletionStatusEx"), Some(7));
        assert_eq!(wanted_function("WaitForSingleObject"), Some(8));
        assert_eq!(wanted_function("WaitForSingleObjectEx"), Some(9));
        assert_eq!(wanted_function("WaitForMultipleObjects"), Some(10));
        assert_eq!(wanted_function("WaitForMultipleObjectsEx"), Some(11));
        assert_eq!(wanted_function("WriteFile"), None);
        assert!(wanted_dll("api-ms-win-core-libraryloader-l1-2-0.dll"));
        assert!(wanted_dll("api-ms-win-core-io-l1-1-0.dll"));
        assert!(wanted_dll("api-ms-win-core-synch-l1-2-0.dll"));
        assert!(is_xgameruntime_load("xgameruntime.dll"));
        assert!(is_xgameruntime_load("C:\\Windows\\System32\\XGameRuntime.dll"));
        assert!(!is_xgameruntime_load("other_xgameruntime.dll"));
        assert_eq!(folder("c:\\game\\title.exe"), "c:\\game\\");
        assert_eq!(folder("title.exe"), "");
    }

    #[test]
    fn coalesce_scatter_segments_merges_contiguous_runs_and_handles_gaps() {
        assert_eq!(coalesce_scatter_segments(&[], 4096), Vec::<(usize, usize)>::new());
        assert_eq!(coalesce_scatter_segments(&[0x10_0000], 0), Vec::<(usize, usize)>::new());

        let segs = [0x10_0000u64, 0x10_1000, 0x10_2000, 0x20_0000, 0x20_1000];
        assert_eq!(coalesce_scatter_segments(&segs, 5 * SEGMENT_BYTES), vec![(0x10_0000, 3 * SEGMENT_BYTES), (0x20_0000, 2 * SEGMENT_BYTES)]);
        assert_eq!(coalesce_scatter_segments(&segs, 4 * SEGMENT_BYTES + 123), vec![(0x10_0000, 3 * SEGMENT_BYTES), (0x20_0000, SEGMENT_BYTES + 123)]);
    }

    #[test]
    fn pending_overlapped_records_takes_and_caps_at_limit() {
        for i in 1..=(MAX_PENDING_OVERLAPPED + 5) {
            record_pending_overlapped_with_event(i, 0, vec![(i * 0x1000, 0x1000)]);
        }
        assert!(take_pending_overlapped(1).is_none(), "oldest entries evicted past cap");
        assert_eq!(take_pending_overlapped(MAX_PENDING_OVERLAPPED + 5), Some(vec![((MAX_PENDING_OVERLAPPED + 5) * 0x1000, 0x1000)]));
        assert!(take_pending_overlapped(MAX_PENDING_OVERLAPPED + 5).is_none());

        record_pending_overlapped_with_event(100, 0xBEEF, vec![(0x10_0000, 0x1000)]);
        assert_eq!(take_pending_by_event(0xBEEF), vec![vec![(0x10_0000, 0x1000)]]);
        assert!(take_pending_overlapped(100).is_none());
    }
}
