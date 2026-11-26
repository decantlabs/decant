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

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use windows_sys::Win32::Foundation::{ERROR_NOACCESS, GetLastError, HMODULE, SetLastError};
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

/// The functions that were in the tables before the patch, 0 until a table was patched.
static ORIGINAL: [AtomicUsize; 3] = [AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0)];
/// The modules whose tables were looked at.
static DONE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// Whether an import table named `dll` can hold the file functions.
fn wanted_dll(dll: &str) -> bool {
    let lower = dll.to_ascii_lowercase();
    lower == "kernel32.dll" || lower == "kernelbase.dll" || (lower.starts_with("api-ms-win-core-file-") && lower.ends_with(".dll"))
}

/// The index in `ORIGINAL` of an imported function, when it is one of the three.
fn wanted_function(name: &str) -> Option<usize> {
    ["ReadFile", "ReadFileEx", "ReadFileScatter"].iter().position(|&n| n == name)
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

/// Makes the pages of `[buffer, buffer + len)` writable for a kernel write.
fn prepare(buffer: *const c_void, len: usize) {
    if let Some(engine) = Engine::started() {
        engine.prepare_write(buffer as usize, len);
    }
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
    // Without OVERLAPPED the read is done when the call returns: its pages stay open until then.
    let blocking = overlapped.is_null();
    let engine = Engine::started();
    match engine {
        Some(e) if blocking => e.begin_write(buffer as usize, len as usize),
        _ => prepare(buffer, len as usize),
    }
    // SAFETY: the wrapper is only installed after `ORIGINAL[0]` holds the function it replaces, and the arguments are the caller's.
    let result = unsafe { std::mem::transmute::<usize, ReadFileFn>(ORIGINAL[0].load(Ordering::Acquire))(file, buffer, len, read, overlapped) };
    check(result, buffer, len as usize);
    if let Some(e) = engine.filter(|_| blocking) {
        // SAFETY: GetLastError and SetLastError only touch the thread's error code, kept for the caller.
        unsafe {
            let error = GetLastError();
            e.end_write(buffer as usize, len as usize);
            SetLastError(error);
        }
    }
    result
}

unsafe extern "system" fn read_file_ex(file: *mut c_void, buffer: *mut c_void, len: u32, overlapped: *mut c_void, routine: *mut c_void) -> BOOL {
    prepare(buffer, len as usize);
    // SAFETY: as in `read_file`, with `ORIGINAL[1]`.
    let result = unsafe { std::mem::transmute::<usize, ReadFileExFn>(ORIGINAL[1].load(Ordering::Acquire))(file, buffer, len, overlapped, routine) };
    check(result, buffer, len as usize);
    result
}

unsafe extern "system" fn read_file_scatter(file: *mut c_void, segments: *const u64, len: u32, reserved: *mut u32, overlapped: *mut c_void) -> BOOL {
    if !segments.is_null() && len != 0 {
        let count = (len as usize).div_ceil(SEGMENT_BYTES);
        // SAFETY: the caller's array has one element for each segment of `len` bytes.
        let slice = unsafe { std::slice::from_raw_parts(segments, count) };
        let runs = coalesce_scatter_segments(slice, len as usize);
        if let Some(engine) = Engine::started() {
            engine.prepare_write_runs(&runs);
        }
    }
    // SAFETY: as in `read_file`, with `ORIGINAL[2]`.
    let result = unsafe { std::mem::transmute::<usize, ReadFileScatterFn>(ORIGINAL[2].load(Ordering::Acquire))(file, segments, len, reserved, overlapped) };
    check(result, segments.cast(), len as usize);
    result
}

fn wrapper(index: usize) -> usize {
    [read_file as *const () as usize, read_file_ex as *const () as usize, read_file_scatter as *const () as usize][index]
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
        assert_eq!(wanted_function("WriteFile"), None);
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
}
