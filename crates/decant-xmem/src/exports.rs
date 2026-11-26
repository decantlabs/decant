//! The exports of `xmem.dll` and the Windows calls behind them. Every function keeps the name,
//! the `WINAPI` convention and the arguments of the C version. Callers pass raw pointers, so the
//! code here is `unsafe`. Each block states what it relies on.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use windows_sys::Win32::Foundation::{ERROR_CALL_NOT_IMPLEMENTED, ERROR_INVALID_PARAMETER, ERROR_NOACCESS, FALSE, GetLastError, HANDLE, HMODULE, SetLastError, TRUE};
use windows_sys::Win32::System::LibraryLoader::DisableThreadLibraryCalls;
use windows_sys::Win32::System::Memory::{
    GetProcessHeap, HEAP_ZERO_MEMORY, HeapAlloc, HeapFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, VirtualAlloc, VirtualFree,
    VirtualQuery, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::engine::{Engine, EngineError, INLINE_WRITE_PAGES};
use crate::journal::GpuChange;
use crate::smallbuf::SmallBuf;
use crate::{
    HEADER_BYTES, HEADER_MAGIC, PAGES_64K, PAGES_LARGE, VirtualRoute, alignment_of, granularity_of, placement_plan, round_range, total_bytes, user_address, uses_pages, virtual_alloc_plan, virtual_route,
};

type AllocHook = unsafe extern "system" fn(usize, u64) -> *mut c_void;
type FreeHook = unsafe extern "system" fn(*mut c_void, u64);

/// The header in front of every allocation, as the C code laid it out (24 bytes).
#[repr(C)]
struct Header {
    /// Address returned by the system.
    raw: *mut c_void,
    /// Requested size.
    size: u64,
    magic: u32,
    /// 1: `VirtualAlloc`, 0: process heap.
    is_virtual: u32,
}

const _: () = assert!(size_of::<Header>() == HEADER_BYTES);

static CURRENT_BYTES: AtomicI64 = AtomicI64::new(0);
static PEAK_BYTES: AtomicI64 = AtomicI64::new(0);
static ALLOC_COUNT: AtomicI64 = AtomicI64::new(0);
/// The caller's hooks as addresses, 0 when absent.
static ALLOC_HOOK: AtomicUsize = AtomicUsize::new(0);
static FREE_HOOK: AtomicUsize = AtomicUsize::new(0);
/// Serialises `XMemFreeDefault` (the SRWLOCK of the C version).
static FREE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn count_allocation(size: usize) {
    let current = CURRENT_BYTES.fetch_add(size as i64, Ordering::SeqCst).wrapping_add(size as i64);
    PEAK_BYTES.fetch_max(current, Ordering::SeqCst);
    ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
}

/// # Safety
/// Exported allocator. The returned block is `size` bytes, aligned as the attributes ask.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    let alignment = alignment_of(attributes);
    let use_pages = uses_pages(size, attributes);
    let Some(total) = total_bytes(size, alignment) else {
        return std::ptr::null_mut();
    };
    // SAFETY: plain allocation calls. A null result is handled below.
    let raw = unsafe {
        if use_pages {
            VirtualAlloc(std::ptr::null(), total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } else {
            HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, total)
        }
    };
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    let user = user_address(raw as usize, alignment);
    let header = (user - HEADER_BYTES) as *mut Header;
    // SAFETY: `user - HEADER_BYTES` lies inside the `total` bytes just allocated (the user address
    // is at least HEADER_BYTES past `raw` and `size` bytes still fit after it), and it is 8 byte
    // aligned because `user` is aligned on at least 16.
    unsafe {
        header.write(Header { raw, size: size as u64, magic: HEADER_MAGIC, is_virtual: u32::from(use_pages) });
    }
    count_allocation(size);
    user as *mut c_void
}

/// True if `[start, start + size)` is committed and readable. A pointer already released by
/// `VirtualFree`, or one at the start of a page with no header in front of it, must not make
/// `XMemFree` crash.
fn readable(start: usize, size: usize) -> bool {
    const READ: u32 = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let Some(last) = start.checked_add(size.wrapping_sub(1)).filter(|_| size > 0) else {
        return false;
    };
    let mut at = start;
    while at <= last {
        // SAFETY: MEMORY_BASIC_INFORMATION is plain data. VirtualQuery only reads the address space.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid output buffer of the size passed.
        if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || info.State != MEM_COMMIT {
            return false;
        }
        if info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || info.Protect & READ == 0 {
            return false;
        }
        let end = (info.BaseAddress as usize).wrapping_add(info.RegionSize);
        if end <= at {
            return false;
        }
        at = end;
    }
    true
}

/// `XMemFree` of a reservation of the physical page engine, which the engine knows by its base
/// address: the views in it are unmapped and the address range is released. The attributes play no
/// part (a mapped page goes back to a reserved one through `XMemMapPhysicalPages` with frame 0, not
/// through `XMemFree`). True when the address lies in a reservation of the engine, whatever the
/// outcome. A block of the allocator never lies in one, so the attributes of a normal free are not
/// read here.
fn free_engine_range(address: usize) -> bool {
    let Some(engine) = Engine::started().filter(|e| e.owns(address)) else {
        return false;
    };
    // A commit the engine reserved itself (`commit_region`) goes away whole, as `VirtualFree` would release it.
    if engine.is_region(address) {
        match engine.release(address) {
            Ok(()) => decant_log::trace(&format!("[xmem] XMemFree : commit 0x{address:x} libere, ses pages rendues au pool\r\n")),
            Err(e) => {
                decant_log::trace(&format!("[xmem] XMemFree : 0x{address:x} refuse, {e}\r\n"));
                // SAFETY: SetLastError only stores the thread's error code.
                unsafe { SetLastError(e.win32_code()) };
            }
        }
        return true;
    }
    // Memory placed inside a reservation by XMemVirtualAlloc goes back to a placeholder, the reservation stays.
    if engine.is_allocation(address) {
        match engine.unplace(address) {
            Ok(size) => decant_log::trace(&format!("[xmem] XMemFree : 0x{address:x}, 0x{size:x} octets rendus a la reservation (placeholder conserve)\r\n")),
            Err(e) => {
                decant_log::trace(&format!("[xmem] XMemFree : 0x{address:x} refuse, {e}\r\n"));
                // SAFETY: SetLastError only stores the thread's error code.
                unsafe { SetLastError(e.win32_code()) };
            }
        }
        return true;
    }
    match engine.release(address) {
        Ok(()) => decant_log::trace(&format!("[xmem] XMemFree : reservation 0x{address:x} liberee\r\n")),
        Err(e) => {
            decant_log::trace(&format!("[xmem] XMemFree : 0x{address:x} refuse, {e}\r\n"));
            // SAFETY: SetLastError only stores the thread's error code.
            unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
        }
    }
    true
}

/// # Safety
/// Exported. `address` is null, a block of `XMemAllocDefault`, or anything else (then ignored
/// when no readable header with the magic precedes it).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemFreeDefault(address: *mut c_void, _attributes: u64) {
    if address.is_null() {
        return;
    }
    let at = address as usize;
    if free_engine_range(at) {
        return;
    }
    // The whole release is one critical section, as in the C version: the header check (readable +
    // magic) and the release. When two threads free the same pointer, the second only enters once
    // the first is done and sees the block gone, so there is no double VirtualFree/HeapFree, no
    // double count, and no access to a page already returned. A compare-and-swap on the magic is
    // not enough: on the VirtualAlloc path the second thread would still touch a released page.
    let guard = FREE_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if at < HEADER_BYTES || !readable(at - HEADER_BYTES, HEADER_BYTES) {
        drop(guard);
        decant_log::trace("[xmem] XMemFree : pointeur sans en-tete lisible (deja libere ?), ignore\r\n");
        return;
    }
    let header = (at - HEADER_BYTES) as *mut Header;
    // SAFETY: the 24 bytes in front of `address` were just checked readable, under the lock, so no
    // other free can release them meanwhile. The magic tells a live header from anything else
    // before it is trusted. Unaligned reads cover a foreign pointer.
    let magic = unsafe { std::ptr::addr_of!((*header).magic).read_unaligned() };
    if magic != HEADER_MAGIC {
        drop(guard);
        decant_log::trace("[xmem] XMemFree : pointeur inconnu ou deja libere, ignore\r\n");
        return;
    }
    // SAFETY: the magic matches, so this is a header written by XMemAllocDefault, aligned and
    // writable, and `raw` is the block the system returned.
    unsafe {
        CURRENT_BYTES.fetch_sub((*header).size as i64, Ordering::SeqCst);
        let raw = (*header).raw;
        let is_virtual = (*header).is_virtual != 0;
        (*header).magic = 0;
        if is_virtual {
            VirtualFree(raw, 0, MEM_RELEASE);
        } else {
            HeapFree(GetProcessHeap(), 0, raw);
        }
    }
    drop(guard);
}

fn alloc_hook() -> Option<AllocHook> {
    let address = ALLOC_HOOK.load(Ordering::SeqCst);
    // SAFETY: a non zero value was stored from an `AllocHook` by XMemSetAllocationHooks.
    (address != 0).then(|| unsafe { std::mem::transmute::<usize, AllocHook>(address) })
}

fn free_hook() -> Option<FreeHook> {
    let address = FREE_HOOK.load(Ordering::SeqCst);
    // SAFETY: a non zero value was stored from a `FreeHook` by XMemSetAllocationHooks.
    (address != 0).then(|| unsafe { std::mem::transmute::<usize, FreeHook>(address) })
}

/// # Safety
/// Exported. Calls the caller's hook when one is set.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    // SAFETY: the hook is the caller's own function with this signature, or the default.
    unsafe {
        match alloc_hook() {
            Some(hook) => hook(size, attributes),
            None => XMemAllocDefault(size, attributes),
        }
    }
}

/// # Safety
/// Exported. Calls the caller's hook when one is set.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    // SAFETY: the hook is the caller's own function with this signature, or the default.
    unsafe {
        match free_hook() {
            Some(hook) => hook(address, attributes),
            None => XMemFreeDefault(address, attributes),
        }
    }
}

/// The title may supply its own functions. Two null pointers bring the defaults back. The
/// public documentation of XMemSetAllocationHooks (docs.xbox.com) calls supplying only one an
/// error: that is logged, and the missing one stays the default.
///
/// # Safety
/// Exported. The hooks must stay callable while they are set.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemSetAllocationHooks(alloc: Option<AllocHook>, release: Option<FreeHook>) {
    if alloc.is_none() != release.is_none() {
        decant_log::trace("[xmem] XMemSetAllocationHooks : une seule fonction fournie (erreur selon la documentation), l'autre reste celle par defaut\r\n");
    }
    ALLOC_HOOK.store(alloc.map_or(0, |f| f as usize), Ordering::SeqCst);
    FREE_HOOK.store(release.map_or(0, |f| f as usize), Ordering::SeqCst);
}

#[unsafe(no_mangle)]
pub extern "system" fn XMemCheckDefaultHeaps() -> BOOL {
    TRUE
}

/// Logs an entry point and its first four raw arguments, as `TraceArgs` did in the C version.
fn trace_args(name: &str, a: usize, b: usize, c: usize, d: usize) {
    decant_log::trace(&format!(
        "[xmem] {name}(0x{a:x}, 0x{b:x}, 0x{c:x}, 0x{d:x})
"
    ));
}

/// Four arguments, matching the title's measured calls: address, size, type and protection.
/// Every reservation without `MEM_COMMIT` (`0x20406000`, `0x20402000`, `0x42000`, a plain
/// `MEM_RESERVE`) reserves a placeholder range that the title can map frames into afterwards with
/// `XMemMapPhysicalPages`, as on the console. It allocates no page. Both round the address down
/// and the end up to the granularity of the console bits, at least 64 KiB, and the mappings take
/// `protect`. A commit with a console page bit (`0x20003000`, `0x20403000`, `0x20401000`) is reserved
/// and committed by the engine with pool pages (`commit_pages`), below 16 GiB when there is room, so the
/// GPU can be given addresses in it. Any other request goes to `VirtualAlloc` with the PC bits only, and
/// a protection that cannot serve committed pages becomes `PAGE_READWRITE`.
///
/// # Safety
/// Exported. Same contract as `VirtualAlloc`: `address` is null or a placement the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemVirtualAlloc(address: *mut c_void, size: usize, allocation_type: u32, protect: u32) -> *mut c_void {
    trace_args("XMemVirtualAlloc", address as usize, size, allocation_type as usize, protect as usize);
    let inside = !address.is_null() && allocation_type & (MEM_RESERVE | MEM_COMMIT) != 0 && Engine::started().is_some_and(|e| e.owns(address as usize));
    let result = if inside {
        place_in_reservation(address as usize, size, allocation_type, protect)
    } else if virtual_route(allocation_type) == VirtualRoute::Placeholder {
        reserve_placeholder(address as usize, size, allocation_type, protect)
    } else if virtual_route(allocation_type) == VirtualRoute::Pool {
        commit_in_pool(address as usize, size, allocation_type, protect)
    } else {
        let plan = virtual_alloc_plan(allocation_type, protect);
        if plan.type_changed || plan.protect != protect {
            decant_log::trace(&format!(
                "[xmem] XMemVirtualAlloc : type Windows {:#x}, protection Windows {:#x}\r\n",
                plan.allocation_type, plan.protect
            ));
        }
        // SAFETY: VirtualAlloc validates its arguments and answers null on failure.
        unsafe { VirtualAlloc(address, size, plan.allocation_type, plan.protect) }
    };
    if result.is_null() {
        // SAFETY: GetLastError reads only this thread's error code.
        let error = unsafe { GetLastError() };
        decant_log::trace(&format!(
            "[xmem] XMemVirtualAlloc({address:p}, {size:#x}, {allocation_type:#x}, {protect:#x}) a echoue : GetLastError={error:#x}\r\n"
        ));
        // SAFETY: restore the thread's error code in case logging changed it.
        unsafe { SetLastError(error) };
    }
    result
}

/// Logs why an engine call failed and leaves the Win32 error code for the caller.
fn engine_failed(what: &str, error: &EngineError) {
    decant_log::trace(&format!("[xmem] {what} : {error}\r\n"));
    // SAFETY: SetLastError only stores the thread's error code.
    unsafe { SetLastError(error.win32_code()) };
}

/// `XMemVirtualAlloc` for `0x20406000` or `0x42000`: reserves the range as a placeholder.
fn reserve_placeholder(address: usize, size: usize, allocation_type: u32, protect: u32) -> *mut c_void {
    let granularity = granularity_of(allocation_type).max(crate::pages::PAGE_BYTES);
    let Some((start, length)) = round_range(address, size, granularity) else {
        engine_failed("XMemVirtualAlloc (reservation)", &EngineError::Invalid("taille nulle ou adresse hors limites"));
        return std::ptr::null_mut();
    };
    match Engine::global().and_then(|engine| engine.reserve(start, length, protect)) {
        Ok(at) => {
            decant_log::trace(&format!("[xmem] XMemVirtualAlloc : reservation 0x{at:x}, 0x{length:x} octets\r\n"));
            at as *mut c_void
        }
        Err(e) => {
            engine_failed("XMemVirtualAlloc (reservation)", &e);
            std::ptr::null_mut()
        }
    }
}

/// `XMemVirtualAlloc` for a commit with a console page bit: the engine reserves the range and commits it
/// with pool pages (`Engine::commit_region`), aligned on the granularity of the console bits.
fn commit_in_pool(address: usize, size: usize, allocation_type: u32, protect: u32) -> *mut c_void {
    let granularity = granularity_of(allocation_type).max(crate::pages::PAGE_BYTES);
    let Some((start, length)) = round_range(address, size, granularity) else {
        engine_failed("XMemVirtualAlloc (commit)", &EngineError::Invalid("taille nulle ou adresse hors limites"));
        return std::ptr::null_mut();
    };
    let (_, win_protect) = placement_plan(allocation_type, protect);
    match Engine::global().and_then(|engine| engine.commit_region(start, length, granularity, win_protect)) {
        Ok((at, pages)) => {
            let backing = if pages { "des pages du pool" } else { "de la memoire privee (invisible au GPU)" };
            decant_log::trace(&format!("[xmem] XMemVirtualAlloc : commit 0x{at:x}, 0x{length:x} octets par {backing}, protection Windows {win_protect:#x}\r\n"));
            at as *mut c_void
        }
        Err(e) => {
            engine_failed("XMemVirtualAlloc (commit)", &e);
            std::ptr::null_mut()
        }
    }
}

/// `XMemVirtualAlloc` at an address inside a reservation of the engine (the console way to name a
/// range of one). The placeholder is cut to the requested range, widened to the granularity of the
/// console bits (at least 64 KiB), as a reservation is. A range that is cut already is accepted. Without
/// `MEM_COMMIT` it stays a placeholder, for `XMemMapPhysicalPages`. With it, it is replaced by committed
/// private memory. The range must lie in one placeholder, else the call fails with `ERROR_INVALID_ADDRESS`.
fn place_in_reservation(address: usize, size: usize, allocation_type: u32, protect: u32) -> *mut c_void {
    let granularity = granularity_of(allocation_type).max(crate::pages::PAGE_BYTES);
    let Some((start, length)) = round_range(address, size, granularity) else {
        engine_failed("XMemVirtualAlloc (dans une reservation)", &EngineError::Invalid("taille nulle ou adresse hors limites"));
        return std::ptr::null_mut();
    };
    let (commit, win_protect) = placement_plan(allocation_type, protect);
    let result = Engine::global().and_then(|engine| {
        if commit {
            engine.place(start, length, true, win_protect).map(|placed| (placed.cuts, placed.pages))
        } else {
            engine.cut(start, length).map(|cuts| (cuts, false))
        }
    });
    match result {
        Ok((cuts, pages)) => {
            if commit {
                let backing = if pages { " par des pages du pool" } else { " par de la memoire privee" };
                decant_log::trace(&format!(
                    "[xmem] XMemVirtualAlloc : placeholder coupe ({cuts} coupe(s)) et remplace{backing}, 0x{start:x}, 0x{length:x} octets, commit, protection Windows {win_protect:#x}
"
                ));
            } else {
                decant_log::trace(&format!(
                    "[xmem] XMemVirtualAlloc : placeholder conserve pour le mappage physique ({cuts} coupe(s)), 0x{start:x}, 0x{length:x} octets
"
                ));
            }
            start as *mut c_void
        }
        Err(e) => {
            engine_failed("XMemVirtualAlloc (dans une reservation)", &e);
            std::ptr::null_mut()
        }
    }
}

/// Logs the raw arguments and fails with `ERROR_CALL_NOT_IMPLEMENTED`.
fn unimplemented_entry(name: &str, a: usize, b: usize, c: usize, d: usize) -> usize {
    decant_log::calls::not_implemented(name);
    trace_args(&format!("{name} (signature inconnue, non implementee)"), a, b, c, d);
    // SAFETY: SetLastError only stores the thread's error code.
    unsafe { SetLastError(ERROR_CALL_NOT_IMPLEMENTED) };
    0
}

/// The most entries a pointer and a count can describe, for `slice::from_raw_parts`.
fn entries_fit(count: usize) -> bool {
    count <= isize::MAX as usize / size_of::<u64>()
}

fn fail(code: u32) {
    // SAFETY: SetLastError only stores the thread's error code.
    unsafe { SetLastError(code) };
}

/// A frame number of the title as the engine's 32 bit number. A value that does not fit becomes 1,
/// which is no frame (frames are multiples of 16), so the engine refuses it.
fn frame32(frame: u64) -> u32 {
    u32::try_from(frame).unwrap_or(1)
}

/// `BOOL XMemAllocatePhysicalPages(ULONG64 flags, SIZE_T* numberOfPages, ULONG64* pageArray)`.
/// Allocates `*numberOfPages` pages of 64 KiB and writes their frame numbers, in units of 4 KiB
/// (page `i` of the backing section is frame `(i + 1) * 16`), to `pageArray`. `flags` carry the
/// same bits as the allocation type of `XMemVirtualAlloc`: with large pages (0x20000000 without
/// the 64 KiB bit) the count must be a multiple of 32. On failure the count becomes 0.
///
/// # Safety
/// Exported. `number_of_pages` is null or readable and writable, `page_array` is null or has room
/// for `*number_of_pages` entries.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemAllocatePhysicalPages(flags: u64, number_of_pages: *mut usize, page_array: *mut u64) -> BOOL {
    if number_of_pages.is_null() {
        fail(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    // SAFETY: non null, and the caller guarantees it is readable and writable.
    let count = unsafe { number_of_pages.read() };
    let large = flags as u32 & PAGES_64K != PAGES_64K && flags as u32 & PAGES_LARGE != 0;
    if (count != 0 && page_array.is_null()) || !entries_fit(count) || (large && !count.is_multiple_of(32)) {
        // SAFETY: as above.
        unsafe { number_of_pages.write(0) };
        fail(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if count == 0 {
        return TRUE;
    }
    match Engine::global().and_then(|engine| engine.allocate_pages(count)) {
        Ok(frames) => {
            for (i, frame) in frames.iter().enumerate() {
                // SAFETY: the caller provides room for `count` entries and `frames` holds `count`.
                unsafe { page_array.add(i).write_unaligned(u64::from(*frame)) };
            }
            TRUE
        }
        Err(e) => {
            // SAFETY: as above.
            unsafe { number_of_pages.write(0) };
            engine_failed("XMemAllocatePhysicalPages", &e);
            FALSE
        }
    }
}

/// `PVOID XMemMapPhysicalPages(PVOID virtualAddress, SIZE_T numberOfPages, ULONG64* pageArray)`.
/// Maps page `i` of the array at `virtualAddress` (rounded down to 64 KiB) plus `i * 64 KiB`, in a
/// range the title reserved with `XMemVirtualAlloc`. A null array, or an entry of 0, unmaps that
/// page, which goes back to the reserved state. A page that is mapped already gets the new mapping.
/// Returns `virtualAddress` as passed, or null with the error.
///
/// # Safety
/// Exported. `page_array` is null or readable for `number_of_pages` entries.
fn should_trace_call(n: i64) -> bool {
    let n = n as u64;
    n < 32 || n.is_power_of_two() || n.is_multiple_of(1000)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemMapPhysicalPages(virtual_address: *mut c_void, number_of_pages: usize, page_array: *const u64) -> *mut c_void {
    // Only the first calls, powers of two, and every 1000th call are traced so streaming cannot flood the log.
    static CALLS: AtomicI64 = AtomicI64::new(0);
    let logged = should_trace_call(CALLS.fetch_add(1, Ordering::Relaxed));
    if virtual_address.is_null() || !entries_fit(number_of_pages) {
        fail(ERROR_INVALID_PARAMETER);
        return std::ptr::null_mut();
    }
    if number_of_pages == 0 {
        return virtual_address;
    }
    let base = virtual_address as usize & !(crate::pages::PAGE_BYTES - 1);
    let result = Engine::global().and_then(|engine| {
        if page_array.is_null() {
            engine.unmap_pages(base, number_of_pages)
        } else {
            // SAFETY: the caller provides `number_of_pages` readable entries.
            let frames: Vec<u32> = unsafe { std::slice::from_raw_parts(page_array, number_of_pages) }.iter().map(|&f| frame32(f)).collect();
            if logged {
                let owned = Engine::started().is_some_and(|e| e.owns(base));
                let first = frames.first().copied().unwrap_or(0);
                let last = frames.last().copied().unwrap_or(0);
                decant_log::trace(&format!(
                    "[xmem] XMemMapPhysicalPages : 0x{base:x}, {number_of_pages} pages, frames {first}..{last}, dans une reservation connue: {owned}\r\n"
                ));
            }
            engine.map_frames(base, &frames)
        }
    });
    match result {
        Ok(()) => {
            if logged {
                decant_log::trace(&format!("[xmem] XMemMapPhysicalPages : 0x{base:x}, {number_of_pages} pages, reussi\r\n"));
            }
            virtual_address
        }
        Err(e) => {
            engine_failed("XMemMapPhysicalPages", &e);
            std::ptr::null_mut()
        }
    }
}

/// `BOOL XMemMapPhysicalPagesScatter(PVOID* virtualAddresses, SIZE_T numberOfPages, ULONG64* pageArray)`.
/// Page `i` goes to `virtualAddresses[i]` rounded down to 64 KiB. An entry of 0 or a null array
/// unmaps, as for `XMemMapPhysicalPages`.
///
/// # Safety
/// Exported. `virtual_addresses` is null or readable for `number_of_pages` entries, and so is
/// `page_array` when it is not null.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemMapPhysicalPagesScatter(virtual_addresses: *const *mut c_void, number_of_pages: usize, page_array: *const u64) -> BOOL {
    static CALLS: AtomicI64 = AtomicI64::new(0);
    let logged = should_trace_call(CALLS.fetch_add(1, Ordering::Relaxed));
    if virtual_addresses.is_null() || !entries_fit(number_of_pages) {
        fail(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    if number_of_pages == 0 {
        return TRUE;
    }
    // SAFETY: the caller provides `number_of_pages` readable entries in each array.
    let addresses = unsafe { std::slice::from_raw_parts(virtual_addresses, number_of_pages) };
    let entries: Vec<(usize, u32)> = addresses
        .iter()
        .enumerate()
        .map(|(i, &address)| {
            // SAFETY: as above, and `i` is below the count.
            let frame = if page_array.is_null() { 0 } else { unsafe { page_array.add(i).read_unaligned() } };
            (address as usize, frame32(frame))
        })
        .collect();
    if logged {
        let first = entries.first().copied().unwrap_or((0, 0));
        decant_log::trace(&format!("[xmem] XMemMapPhysicalPagesScatter : {number_of_pages} entrees, premiere 0x{:x} -> frame {}\r\n", first.0, first.1));
    }
    match Engine::global().and_then(|engine| engine.map_scatter(&entries)) {
        Ok(()) => {
            if logged {
                decant_log::trace(&format!("[xmem] XMemMapPhysicalPagesScatter : {number_of_pages} entrees, reussi\r\n"));
            }
            TRUE
        }
        Err(e) => {
            engine_failed("XMemMapPhysicalPagesScatter", &e);
            FALSE
        }
    }
}

/// `BOOL XMemFreePhysicalPages(SIZE_T numberOfPages, ULONG64* pageArray)`. Frees the frames, in
/// chunks of 4096 entries that are each checked before they are freed, so a failure leaves the
/// earlier chunks freed. A page that a view still maps stays readable through it until unmapped.
///
/// # Safety
/// Exported. `page_array` is null or readable for `number_of_pages` entries.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemFreePhysicalPages(number_of_pages: usize, page_array: *const u64) -> BOOL {
    if number_of_pages == 0 {
        return TRUE;
    }
    if page_array.is_null() {
        fail(ERROR_NOACCESS);
        return FALSE;
    }
    if !entries_fit(number_of_pages) {
        fail(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    // SAFETY: the caller provides `number_of_pages` readable entries.
    let frames = unsafe { std::slice::from_raw_parts(page_array, number_of_pages) };
    for chunk in frames.chunks(4096) {
        let list: Vec<u32> = chunk.iter().map(|&f| frame32(f)).collect();
        if let Err(e) = Engine::global().and_then(|engine| engine.free_pages(&list)) {
            engine_failed("XMemFreePhysicalPages", &e);
            return FALSE;
        }
    }
    TRUE
}

/// Imported by retail titles. There is no memory reserved for tools on PC, so it answers
/// FALSE and logs.
#[unsafe(no_mangle)]
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    decant_log::trace("[xmem] XMemIsToolsMemAvailable : FALSE (pas de memoire outils sur PC)\r\n");
    FALSE
}

/// Signature unknown, not implemented: logs its raw arguments and returns 0.
#[unsafe(no_mangle)]
pub extern "system" fn XMemTransferMemory(a: usize, b: usize, c: usize, d: usize) -> usize {
    unimplemented_entry("XMemTransferMemory", a, b, c, d)
}

/// 64 byte buffer: the first 48 are a MEMORY_BASIC_INFORMATION (State at 0x20, Protect at 0x24).
/// Byte 0x30 holds flags whose bits 0 and 1 mean "readable by the processor" to the caller (the
/// BVH builder refuses an address with both bits clear).
///
/// # Safety
/// Exported. `information` is null or points to 64 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemVirtualQuery(process: HANDLE, address: *const c_void, information: *mut c_void) -> BOOL {
    if information.is_null() {
        return FALSE;
    }
    let out = information.cast::<u8>();
    // SAFETY: the caller provides 64 writable bytes at `information`.
    unsafe { out.write_bytes(0, 64) };
    // SAFETY: GetCurrentProcess returns a pseudo handle and has no precondition.
    let process = if process.is_null() { unsafe { GetCurrentProcess() } } else { process };
    // SAFETY: MEMORY_BASIC_INFORMATION is plain data.
    let mut basic: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `basic` is a valid output buffer of the size passed. The address is only queried.
    if unsafe { VirtualQueryEx(process, address, &mut basic, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
        return FALSE;
    }
    // A pool page in a view is committed memory with the title's protection, for the title: one that was
    // never touched is only reserved until its first access, and one the dirty tracking made read only for
    // a while is still read write.
    if process as isize == -1
        && let Some(protect) = Engine::started().and_then(|e| e.title_protection(address as usize))
    {
        basic.State = MEM_COMMIT;
        basic.Protect = protect;
    }
    // SAFETY: 48 bytes of `basic` go into the 64 the caller provides, then the flag byte at 0x30.
    unsafe {
        std::ptr::copy_nonoverlapping((&raw const basic).cast::<u8>(), out, size_of::<MEMORY_BASIC_INFORMATION>());
        out.add(0x30).write(u8::from(basic.State == MEM_COMMIT));
    }
    TRUE
}

/// 32 byte structure, four u64 (OpenScarlett, MIT: 0 total, 8, 16 working set, 24 peak). Our package's
/// DLLs pass 128 byte buffers, but a caller that reserves only 32 would have its stack overwritten
/// if we wrote more, so only 32 bytes are written, which is safe for every caller. The only field
/// known callers read is the u64 at offset 8, compared before and after creating a resource: it keeps
/// the memory in use (OpenScarlett reads available memory there, to confirm on the executable). The
/// meaning of the selector is a guess.
///
/// # Safety
/// Exported. `statistics` is null or points to 32 writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn XMemGetWorkingSetStatistics(_selector: u32, statistics: *mut c_void) -> BOOL {
    if statistics.is_null() {
        return FALSE;
    }
    let out = statistics.cast::<u64>();
    // SAFETY: the caller provides 32 writable bytes at `statistics`.
    unsafe { statistics.cast::<u8>().write_bytes(0, 32) };
    // SAFETY: both structures are plain data, sized as their `cb` and `dwLength` fields say.
    let (mut counters, mut status): (PROCESS_MEMORY_COUNTERS_EX, MEMORYSTATUSEX) = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
    counters.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `counters` is a valid output buffer of the size passed.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(), counters.cb) };
    if ok == 0 {
        return FALSE;
    }
    // SAFETY: `status` is a valid output buffer with its length set.
    let total = if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullTotalPhys } else { 0 };
    // Wine 11 returns PrivateUsage = 0 (the field is not filled in). Without a fallback the caller
    // always sees 0 bytes in use. PagefileUsage (memory committed by the process) has the same
    // meaning on Windows. WorkingSetSize is the last resort.
    let mut used = counters.PrivateUsage as u64;
    if used == 0 {
        used = if counters.PagefileUsage != 0 { counters.PagefileUsage as u64 } else { counters.WorkingSetSize as u64 };
    }
    // SAFETY: offsets 0 to 24 are inside the 32 bytes. The caller's buffer may be unaligned.
    unsafe {
        out.write_unaligned(total);
        out.add(1).write_unaligned(used);
        out.add(2).write_unaligned(counters.WorkingSetSize as u64);
        out.add(3).write_unaligned(counters.PeakWorkingSetSize as u64);
    }
    TRUE
}

/// Outside the console API: Decant's counters, for the tests.
///
/// # Safety
/// Exported. Each pointer is null or points to a writable `u64`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXMemCounters(current: *mut u64, peak: *mut u64, count: *mut u64) {
    // SAFETY: each non null pointer is writable, as the caller guarantees.
    unsafe {
        if !current.is_null() {
            *current = CURRENT_BYTES.load(Ordering::SeqCst) as u64;
        }
        if !peak.is_null() {
            *peak = PEAK_BYTES.load(Ordering::SeqCst) as u64;
        }
        if !count.is_null() {
            *count = ALLOC_COUNT.load(Ordering::SeqCst) as u64;
        }
    }
}

/// Outside the console API: switches the dirty tracking of the GPU mirror on or off. On, every page
/// a view maps starts dirty, written pages are seen through page protection, and the I/O guard
/// patches the file reads of the title (`io_guard`).
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemGpuTrack(on: BOOL) {
    let Ok(engine) = Engine::global() else { return };
    engine.set_tracking(on != 0);
    if on != 0 {
        crate::io_guard::install();
        decant_log::trace("[xmem] suivi des pages pour le GPU : actif\r\n");
    }
}

/// Outside the console API: patches the file reads of the modules loaded since the last call, which
/// `d3d12_xs.dll` asks for now and then.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemGpuGuardModules() {
    if Engine::started().is_some() {
        crate::io_guard::install();
    }
}

/// Outside the console API: `pages` pages from `to` show the pool pages that the pages from `from`
/// show (one more view of them), or go back to the reserved state when `from` is 0. This is what the
/// page mapping copies of a console command queue do (`d3d12_xs.dll` calls it for them). FALSE when a
/// source page shows no pool page or `to` is not in a reservation of the title: nothing is mapped then.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemShowPages(to: usize, from: usize, pages: usize) -> BOOL {
    if to == 0 || !entries_fit(pages) {
        return FALSE;
    }
    if pages == 0 {
        return TRUE;
    }
    let result = Engine::global().and_then(|engine| if from == 0 { engine.unmap_pages(to & !(crate::pages::PAGE_BYTES - 1), pages) } else { engine.show_pages_at(to, from, pages) });
    match result {
        Ok(()) => TRUE,
        Err(e) => {
            static SEEN: AtomicI64 = AtomicI64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                decant_log::trace(&format!("[xmem] DecantXmemShowPages : 0x{to:x} <- 0x{from:x}, {pages} pages : {e}\r\n"));
            }
            FALSE
        }
    }
}

/// Outside the console API: the GPU is given `[address, address + len)`. Pages of it in a commit with a
/// console page bit reach the GPU mirror from now on (they come in `DecantXmemGpuChanges` as `MAP`). Returns
/// how many pages that made, 0 when all were known already or the range lies in no such commit.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemGpuUse(address: usize, len: usize) -> u32 {
    Engine::started().map_or(0, |engine| engine.gpu_use(address, len) as u32)
}

/// Outside the console API: the changes of the address space after sequence number `after`, at most
/// `cap` records, in order. `*written` is the count. FALSE when the ring no longer holds them (the
/// caller starts again from `DecantXmemGpuSnapshot`).
///
/// # Safety
/// Exported. `out` has room for `cap` records, `written` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXmemGpuChanges(after: u64, out: *mut GpuChange, cap: u32, written: *mut u32) -> BOOL {
    let set = |n: usize| {
        if !written.is_null() {
            // SAFETY: non null, and the caller guarantees it is writable.
            unsafe { written.write(n as u32) };
        }
    };
    let Some(engine) = Engine::started() else {
        set(0);
        return TRUE;
    };
    if engine.journal_seq() == after {
        set(0);
        return TRUE;
    }
    let Some(changes) = engine.changes(after, cap as usize) else {
        set(0);
        return FALSE;
    };
    if !changes.is_empty() && !out.is_null() {
        // SAFETY: the caller provides room for `cap` records and `changes` holds at most `cap`.
        unsafe { std::ptr::copy_nonoverlapping(changes.as_ptr(), out, changes.len()) };
    }
    set(changes.len());
    TRUE
}

/// Outside the console API: returns the current GPU journal sequence number without locking.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemJournalSeq() -> u64 {
    Engine::started().map_or(0, Engine::journal_seq)
}

/// Outside the console API: every live reservation and view as `RESERVE` and `MAP` records, and the
/// sequence number they are valid for. FALSE when `cap` is too small, then `*written` is the count needed.
///
/// # Safety
/// Exported. `out` has room for `cap` records, `written` and `seq` are null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXmemGpuSnapshot(out: *mut GpuChange, cap: u32, written: *mut u32, seq: *mut u64) -> BOOL {
    let (records, at) = Engine::started().map_or((Vec::new(), 0), Engine::snapshot);
    // SAFETY: each non null pointer is writable, as the caller guarantees.
    unsafe {
        if !written.is_null() {
            written.write(records.len() as u32);
        }
        if !seq.is_null() {
            seq.write(at);
        }
        if records.len() > cap as usize {
            return FALSE;
        }
        if !records.is_empty() && !out.is_null() {
            std::ptr::copy_nonoverlapping(records.as_ptr(), out, records.len());
        }
    }
    TRUE
}

/// Outside the console API: the pool pages whose mirror is stale, at most `cap`, each protected
/// against writes until it is written again. Returns how many. The caller copies them from the read
/// only section views afterwards.
///
/// # Safety
/// Exported. `out` has room for `cap` entries.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXmemGpuTakeDirty(out: *mut u32, cap: u32) -> u32 {
    let Some(engine) = Engine::started().filter(|e| e.tracking()) else { return 0 };
    let pages = engine.take_dirty(cap as usize);
    if !pages.is_empty() && !out.is_null() {
        // SAFETY: the caller provides room for `cap` entries and `pages` holds at most `cap`.
        unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), out, pages.len()) };
    }
    pages.len() as u32
}

/// Outside the console API: the private read only view of backing section `index`: its address, the
/// pool page it starts at and its page count. FALSE past the last section.
///
/// # Safety
/// Exported. Each pointer is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXmemGpuSectionView(index: u32, base: *mut usize, first_page: *mut u32, pages: *mut u32) -> BOOL {
    let Some((view, first, count)) = Engine::started().and_then(|e| e.section_view(index as usize)) else {
        return FALSE;
    };
    // SAFETY: each non null pointer is writable, as the caller guarantees.
    unsafe {
        if !base.is_null() {
            *base = view;
        }
        if !first_page.is_null() {
            *first_page = first;
        }
        if !pages.is_null() {
            *pages = count;
        }
    }
    TRUE
}

/// Outside the console API: a write by the kernel (a file read, a DirectStorage request) may land in
/// `[address, address + len)`. The pages of it that the title maps become dirty and writable, and stay
/// so for two seconds, which covers requests that complete later.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemPrepareWrite(address: usize, len: usize) {
    if let Some(engine) = Engine::started() {
        engine.prepare_write(address, len);
    }
}

/// Outside the console API: pins `[address, address + len)` writable for an active asynchronous write.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemBeginWrite(address: usize, len: usize) {
    if let Some(engine) = Engine::started() {
        engine.begin_io_write(address, len);
    }
}

/// Outside the console API: releases an active asynchronous write pin on `[address, address + len)`.
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemFinishWrite(address: usize, len: usize) {
    if let Some(engine) = Engine::started() {
        engine.finish_io_write(address, len);
    }
}

/// Outside the console API: writes `len` bytes at `data` into physical pages, for a DirectStorage request
/// whose destination is a page array. `page_array` holds `count` frame numbers as
/// `XMemAllocatePhysicalPages` gives them, one per 64 KiB, and the bytes start at `offset` in the first
/// page. The write goes through the pool, not through a mapping of the title, so a page that no view maps
/// yet holds the bytes once it is mapped. The pages become dirty for the GPU mirror. `views` gets up to
/// `cap` pairs (address, bytes): where views of the title show the bytes now. `*found` is the number of
/// pairs, which may exceed `cap`. FALSE with the error code set, and nothing written, when a frame the
/// bytes reach is not allocated or `count` is too small.
///
/// # Safety
/// Exported. `page_array` is readable for `count` entries, `data` for `len` bytes, `views` is null or
/// writable for `cap` pairs and `found` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn DecantXmemWritePhysicalPages(page_array: *const u64, count: usize, offset: usize, data: *const u8, len: usize, views: *mut [usize; 2], cap: u32, found: *mut u32) -> BOOL {
    let set = |n: usize| {
        if !found.is_null() {
            // SAFETY: non null, and the caller guarantees it is writable.
            unsafe { found.write(u32::try_from(n).unwrap_or(u32::MAX)) };
        }
    };
    set(0);
    if len == 0 {
        return TRUE;
    }
    if page_array.is_null() || data.is_null() || !entries_fit(count) {
        fail(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    // SAFETY: the caller guarantees both ranges.
    let (entries, bytes) = unsafe { (std::slice::from_raw_parts(page_array, count), std::slice::from_raw_parts(data, len)) };
    let frames: SmallBuf<u32, INLINE_WRITE_PAGES> = entries.iter().map(|&f| frame32(f)).collect();
    match Engine::global().and_then(|engine| engine.write_frames_buf(frames.as_slice(), offset, bytes)) {
        Ok(seen) => {
            if !views.is_null() {
                for (k, &(address, n)) in seen.as_slice().iter().take(cap as usize).enumerate() {
                    // SAFETY: the caller provides room for `cap` pairs.
                    unsafe { views.add(k).write([address, n]) };
                }
            }
            set(seen.as_slice().len());
            TRUE
        }
        Err(e) => {
            engine_failed("DecantXmemWritePhysicalPages", &e);
            FALSE
        }
    }
}

/// Outside the console API: whether `[address, address + len)` lies in views the title may write,
/// whatever the temporary protection of the dirty tracking (`d3d12_xs.dll` and `xgameruntime.dll`
/// ask before they write for the title).
#[unsafe(no_mangle)]
pub extern "system" fn DecantXmemGpuWritable(address: usize, len: usize) -> BOOL {
    BOOL::from(Engine::started().is_some_and(|e| e.writable(address, len)))
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
    #[cfg(target_arch = "x86_64")]
    crate::cpu_emu_veh::install();
    if crate::io_guard::has_xgr_redirect() {
        crate::io_guard::install();
    }
    decant_log::trace("[xmem] Decant xmem.dll charge\r\n");
}

#[used]
#[unsafe(link_section = ".CRT$XCU")]
static ON_PROCESS_ATTACH: extern "C" fn() = on_process_attach;
