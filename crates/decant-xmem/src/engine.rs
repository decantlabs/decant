//! The physical page engine: one pagefile backed section holds the pages, Windows placeholders
//! hold the addresses, and views of the section are mapped over the placeholders with
//! `MapViewOfFile3`. One page can appear at several addresses, as on the console. The rules (frame
//! numbers, page allocator, registry of placeholders and views) are the safe modules `pages` and
//! `regions`. This module only makes the Windows calls, each behind a `// SAFETY:` comment, and
//! keeps the registry in step with what Windows holds.
//!
//! Dirty tracking has its own mutex. Dirty takes and tracked write faults never acquire the
//! address space mutex. Mapping changes acquire the address space mutex first, then the dirty
//! mutex, and hold both through alias updates and protection changes. Pool view metadata stays
//! with the tracker so dirty takes can query commitment without the address space registry.
//!
//! `VirtualAlloc2`, `MapViewOfFile3` and `UnmapViewOfFile2` (Windows 10 1803 and later) are
//! looked up in `kernelbase.dll` when the engine starts. Without them the engine does not start
//! and every call answers `EngineError::Unsupported`, so `xmem.dll` still loads on older systems.
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt;
use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock, PoisonError};
use std::time::Instant;

use windows_sys::Win32::Foundation::{CloseHandle, EXCEPTION_ACCESS_VIOLATION, ERROR_NOT_ALL_ASSIGNED, GetLastError, HANDLE, INVALID_HANDLE_VALUE, NTSTATUS};
use windows_sys::Win32::Security::{AdjustTokenPrivileges, LookupPrivilegeValueW, SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, GetLargePageMinimum, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT, SEC_LARGE_PAGES,
    SEC_RESERVE, UnmapViewOfFile, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use crate::journal::{self, GpuChange, Journal};
use crate::pages::{FRAMES_PER_PAGE, Growth, MAX_POOL_PAGES, NO_FRAME, PAGE_BYTES, PageAllocator, PageError, growth, index_of};
use crate::regions::{Kind, RegionError, Regions, split_steps};
use crate::smallbuf::SmallBuf;
use crate::track::{self, Track};

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
/// `VirtualFree` and `UnmapViewOfFile2`: keep the range as a placeholder.
const MEM_PRESERVE_PLACEHOLDER: u32 = 0x2;
/// `VirtualFree`: join adjacent placeholders into one.
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;
const PAGE_NOACCESS: u32 = 0x01;

/// Pages of the first backing section when nothing else is asked: 1 GiB. Only the pages that are
/// touched use memory, but a section is committed against the page file. When a request needs
/// more, the pool grows by further sections (see `pages::growth`).
pub const DEFAULT_CAPACITY_PAGES: usize = 16 * 1024;

type VirtualAlloc2Fn = unsafe extern "system" fn(HANDLE, *const c_void, usize, u32, u32, *mut c_void, u32) -> *mut c_void;
type MapViewOfFile3Fn = unsafe extern "system" fn(HANDLE, HANDLE, *const c_void, u64, usize, u32, u32, *mut c_void, u32) -> *mut c_void;
type UnmapViewOfFile2Fn = unsafe extern "system" fn(HANDLE, *mut c_void, u32) -> i32;

/// What `Engine::place` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    /// Cuts made in the placeholder: 0 (it was the whole range), 1 (front or tail) or 2 (middle).
    pub cuts: usize,
    pub committed: bool,
    /// The commit is backed by pages of the pool (a view), not private memory.
    pub pages: bool,
}

/// Why a call of the engine failed. Nothing is left half done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineError {
    /// This Windows lacks the named function.
    Unsupported(&'static str),
    /// A Windows call failed with this error code.
    Os { call: &'static str, code: u32 },
    Pages(PageError),
    Regions(RegionError),
    /// The backing section has not enough free pages.
    OutOfPages,
    /// An argument is not valid (the text says which).
    Invalid(&'static str),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Unsupported(name) => write!(f, "{name} absente de ce Windows"),
            EngineError::Os { call, code } => write!(f, "{call} a echoue (erreur {code})"),
            EngineError::Pages(e) => write!(f, "pages : {e:?}"),
            EngineError::Regions(e) => write!(f, "adresses : {e:?}"),
            EngineError::OutOfPages => write!(f, "plus de pages physiques libres"),
            EngineError::Invalid(what) => write!(f, "argument invalide : {what}"),
        }
    }
}

impl EngineError {
    /// The Win32 error code a caller of an export gets with the failure.
    pub fn win32_code(&self) -> u32 {
        match self {
            EngineError::Unsupported(_) => 120,
            EngineError::Os { code, .. } => *code,
            EngineError::OutOfPages => 8,
            EngineError::Regions(RegionError::NotInPlaceholder | RegionError::Unknown) => 487,
            EngineError::Pages(_) | EngineError::Regions(_) | EngineError::Invalid(_) => 87,
        }
    }
}

impl From<PageError> for EngineError {
    fn from(e: PageError) -> Self {
        EngineError::Pages(e)
    }
}

impl From<RegionError> for EngineError {
    fn from(e: RegionError) -> Self {
        EngineError::Regions(e)
    }
}

/// The protection a view gets for a console protection: read only stays read only, anything else
/// is read write (the section is not executable).
pub fn view_protection(protect: u32) -> u32 {
    if protect & 0xFF == PAGE_READONLY { PAGE_READONLY } else { PAGE_READWRITE }
}

struct Api {
    virtual_alloc2: VirtualAlloc2Fn,
    map_view: MapViewOfFile3Fn,
    unmap_view: UnmapViewOfFile2Fn,
}

impl Api {
    fn load() -> Result<Api, EngineError> {
        // SAFETY: a plain lookup of a module that every process has loaded.
        let module = unsafe { GetModuleHandleA(c"kernelbase.dll".as_ptr().cast()) };
        if module.is_null() {
            return Err(EngineError::Unsupported("kernelbase.dll"));
        }
        // SAFETY: GetProcAddress only reads the export table of the module.
        let (va2, map, unmap) = unsafe {
            (
                GetProcAddress(module, c"VirtualAlloc2".as_ptr().cast()),
                GetProcAddress(module, c"MapViewOfFile3".as_ptr().cast()),
                GetProcAddress(module, c"UnmapViewOfFile2".as_ptr().cast()),
            )
        };
        let (Some(va2), Some(map), Some(unmap)) = (va2, map, unmap) else {
            let missing = if va2.is_none() {
                "VirtualAlloc2"
            } else if map.is_none() {
                "MapViewOfFile3"
            } else {
                "UnmapViewOfFile2"
            };
            return Err(EngineError::Unsupported(missing));
        };
        // SAFETY: each address is the documented function of that name, whose signature the
        // type aliases above repeat.
        unsafe {
            Ok(Api {
                virtual_alloc2: std::mem::transmute::<unsafe extern "system" fn() -> isize, VirtualAlloc2Fn>(va2),
                map_view: std::mem::transmute::<unsafe extern "system" fn() -> isize, MapViewOfFile3Fn>(map),
                unmap_view: std::mem::transmute::<unsafe extern "system" fn() -> isize, UnmapViewOfFile2Fn>(unmap),
            })
        }
    }

    fn process() -> HANDLE {
        // SAFETY: returns the pseudo handle of this process, with no precondition.
        unsafe { GetCurrentProcess() }
    }

    fn last_error(call: &'static str) -> EngineError {
        // SAFETY: GetLastError has no precondition.
        EngineError::Os { call, code: unsafe { GetLastError() } }
    }

    /// Reserves `size` bytes as one placeholder, at `address` or anywhere when it is 0.
    fn reserve(&self, address: usize, size: usize) -> Result<usize, EngineError> {
        // SAFETY: VirtualAlloc2 validates its arguments. No extended parameters are passed.
        let at = unsafe {
            (self.virtual_alloc2)(Self::process(), address as *const c_void, size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0)
        };
        if at.is_null() { Err(Self::last_error("VirtualAlloc2")) } else { Ok(at as usize) }
    }

    /// Reserves `size` bytes as one placeholder that ends at or below `highest_end`, aligned on
    /// `align` (a power of two, at least 64 KiB), wherever Windows finds room
    /// (`MEM_ADDRESS_REQUIREMENTS`).
    fn reserve_below(&self, size: usize, highest_end: usize, align: usize) -> Result<usize, EngineError> {
        #[repr(C)]
        struct Requirements {
            lowest: *mut c_void,
            highest: *mut c_void,
            alignment: usize,
        }
        #[repr(C)]
        struct Extended {
            kind: u64,
            pointer: *mut c_void,
        }
        const ADDRESS_REQUIREMENTS: u64 = 1;
        let mut req = Requirements { lowest: std::ptr::null_mut(), highest: (highest_end - 1) as *mut c_void, alignment: align };
        let mut param = Extended { kind: ADDRESS_REQUIREMENTS, pointer: (&raw mut req).cast() };
        // SAFETY: VirtualAlloc2 validates its arguments. The one extended parameter points to a
        // MEM_ADDRESS_REQUIREMENTS that lives for the call.
        let at = unsafe {
            (self.virtual_alloc2)(Self::process(), std::ptr::null(), size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, (&raw mut param).cast(), 1)
        };
        if at.is_null() { Err(Self::last_error("VirtualAlloc2 (sous 16 Gio)")) } else { Ok(at as usize) }
    }

    /// Reserves `size` bytes as one placeholder starting at or above `lowest`, with no upper bound
    /// (`MEM_ADDRESS_REQUIREMENTS`, `HighestEndingAddress = 0`), wherever Windows finds room. Used for
    /// an address-less reservation without a console commit bit (the title's own plain "anywhere"
    /// windows), so it does not land inside the low window `reserve_below` needs for the arenas on
    /// Wine. `lowest` must be a multiple of the allocation granularity (MS Docs,
    /// `MEM_ADDRESS_REQUIREMENTS`: "LowestStartingAddress ... must be a multiple of the system's
    /// allocation granularity"); `GPU_LOW_END` (16 GiB) already is.
    fn reserve_above(&self, lowest: usize, size: usize) -> Result<usize, EngineError> {
        #[repr(C)]
        struct Requirements {
            lowest: *mut c_void,
            highest: *mut c_void,
            alignment: usize,
        }
        #[repr(C)]
        struct Extended {
            kind: u64,
            pointer: *mut c_void,
        }
        const ADDRESS_REQUIREMENTS: u64 = 1;
        let mut req = Requirements { lowest: lowest as *mut c_void, highest: std::ptr::null_mut(), alignment: 0 };
        let mut param = Extended { kind: ADDRESS_REQUIREMENTS, pointer: (&raw mut req).cast() };
        // SAFETY: VirtualAlloc2 validates its arguments. The one extended parameter points to a
        // MEM_ADDRESS_REQUIREMENTS that lives for the call.
        let at = unsafe {
            (self.virtual_alloc2)(Self::process(), std::ptr::null(), size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, (&raw mut param).cast(), 1)
        };
        if at.is_null() { Err(Self::last_error("VirtualAlloc2 (au-dessus de 16 Gio)")) } else { Ok(at as usize) }
    }

    /// Cuts the placeholder at `address` into one of `first` bytes and the rest. A title on PC may
    /// have cut it already with its own `VirtualFree`: when Windows reports a placeholder of exactly
    /// `first` bytes at `address`, there is nothing left to cut and that counts as done.
    fn split(&self, address: usize, first: usize) -> Result<(), EngineError> {
        // SAFETY: VirtualFree validates that the range is a placeholder larger than `first`.
        let ok = unsafe { VirtualFree(address as *mut c_void, first, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        if ok != 0 {
            return Ok(());
        }
        let error = Self::last_error("VirtualFree (couper)");
        // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid output buffer of the size passed.
        let known = unsafe { VirtualQuery(address as *const c_void, &mut info, size_of_val(&info)) } != 0;
        if known && info.AllocationBase as usize == address && info.RegionSize == first && info.State == MEM_RESERVE && info.Type == MEM_PRIVATE {
            decant_log::trace(&format!("[xmem] couper 0x{address:x}, 0x{first:x} : deja coupe par le titre (placeholder de cette taille), rien a faire\r\n"));
            return Ok(());
        }
        Err(error)
    }

    /// Joins the adjacent placeholders of `[address, address + size)` into one.
    fn coalesce(&self, address: usize, size: usize) -> Result<(), EngineError> {
        // SAFETY: VirtualFree validates that the range is whole placeholders of one reservation.
        let ok = unsafe { VirtualFree(address as *mut c_void, size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) };
        if ok == 0 { Err(Self::last_error("VirtualFree (rejoindre)")) } else { Ok(()) }
    }

    /// Releases `[address, address + size)`, which the registry holds as one region but Windows may
    /// hold as several placeholders (a title can cut them itself): one `VirtualFree` per piece.
    fn release_pieces(&self, address: usize, size: usize) -> Result<(), EngineError> {
        let end = address + size;
        let mut at = address;
        while at < end {
            // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid output buffer of the size passed.
            let step = if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) } != 0 { info.RegionSize.max(PAGE_BYTES) } else { PAGE_BYTES };
            self.release_placeholder(at)?;
            at += step;
        }
        Ok(())
    }

    fn release_placeholder(&self, address: usize) -> Result<(), EngineError> {
        // SAFETY: VirtualFree validates the address, and refuses what is not a placeholder.
        let ok = unsafe { VirtualFree(address as *mut c_void, 0, MEM_RELEASE) };
        if ok == 0 { Err(Self::last_error("VirtualFree")) } else { Ok(()) }
    }

    /// Replaces the placeholder `[address, address + size)` by private memory: reserved only
    /// (`PAGE_NOACCESS`), or committed with `protect`.
    fn replace(&self, address: usize, size: usize, commit: bool, protect: u32) -> Result<(), EngineError> {
        let kind = MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | if commit { MEM_COMMIT } else { 0 };
        let protect = if commit { protect } else { PAGE_NOACCESS };
        // SAFETY: VirtualAlloc2 validates that the range is exactly one placeholder.
        let at = unsafe { (self.virtual_alloc2)(Self::process(), address as *const c_void, size, kind, protect, std::ptr::null_mut(), 0) };
        if at.is_null() { Err(Self::last_error("VirtualAlloc2 (remplacer)")) } else { Ok(()) }
    }

    /// Gives private memory that replaced a placeholder back as a placeholder of the same range.
    fn restore(&self, address: usize, size: usize) -> Result<(), EngineError> {
        // SAFETY: VirtualFree validates the range, and refuses what is not such an allocation.
        let ok = unsafe { VirtualFree(address as *mut c_void, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        if ok == 0 { Err(Self::last_error("VirtualFree (rendre)")) } else { Ok(()) }
    }

    /// Maps `size` bytes of `section` from `offset` over the placeholder `[address, address + size)`.
    fn map(&self, section: usize, address: usize, offset: u64, size: usize, protect: u32) -> Result<(), EngineError> {
        // SAFETY: MapViewOfFile3 validates the placeholder and the section. `section` is a live
        // section handle owned by the engine. No extended parameters are passed.
        let at = unsafe {
            (self.map_view)(section as HANDLE, Self::process(), address as *const c_void, offset, size, MEM_REPLACE_PLACEHOLDER, protect, std::ptr::null_mut(), 0)
        };
        if at.is_null() { Err(Self::last_error("MapViewOfFile3")) } else { Ok(()) }
    }

    /// Unmaps the view at `address`. With `keep`, the range goes back to a placeholder.
    fn unmap(&self, address: usize, keep: bool) -> Result<(), EngineError> {
        // SAFETY: UnmapViewOfFile2 validates that the address is the base of a view.
        let ok = unsafe { (self.unmap_view)(Self::process(), address as *mut c_void, if keep { MEM_PRESERVE_PLACEHOLDER } else { 0 }) };
        if ok == 0 { Err(Self::last_error("UnmapViewOfFile2")) } else { Ok(()) }
    }
}

/// Lets this process lock pages in memory, which large pages need. False without the privilege.
fn enable_lock_memory_privilege() -> bool {
    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: `token` is a valid output location. The pseudo handle needs no release.
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token) } == 0 {
        return false;
    }
    // SAFETY: TOKEN_PRIVILEGES is plain data.
    let mut privileges: TOKEN_PRIVILEGES = unsafe { std::mem::zeroed() };
    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    // SAFETY: the name is a NUL terminated wide string and the LUID a valid output location.
    let found = unsafe { LookupPrivilegeValueW(std::ptr::null(), windows_sys::w!("SeLockMemoryPrivilege"), &mut privileges.Privileges[0].Luid) };
    let enabled = found != 0
        // SAFETY: `token` is a live token handle and `privileges` is fully initialised.
        && unsafe { AdjustTokenPrivileges(token, 0, &privileges, 0, std::ptr::null_mut(), std::ptr::null_mut()) } != 0
        // SAFETY: GetLastError has no precondition. The call above succeeds even when not every
        // privilege was granted, and reports that case here.
        && unsafe { GetLastError() } != ERROR_NOT_ALL_ASSIGNED;
    // SAFETY: `token` was opened above.
    unsafe { CloseHandle(token) };
    enabled
}

/// Whether the process runs under Wine (ntdll exports `wine_get_version`). A committed section there is a
/// sparse file that costs nothing until it is touched, while committing a reserved one page by page costs
/// a wineserver request each.
fn on_wine() -> bool {
    static WINE: OnceLock<bool> = OnceLock::new();
    *WINE.get_or_init(|| {
        // SAFETY: a lookup of a module every process has loaded, and of an export by name.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    })
}

/// A section of normal pages is only reserved: a page is committed, and charged, on its first access
/// (`Engine::commit_on_access_in`, `Engine::prepare_write`), so pages the title never touches (render
/// targets, the unused tail of an arena, the slack of a growth) cost no commit. Large pages, and every
/// section under Wine (`on_wine`), are committed whole.
fn create_section(bytes: usize, large: bool) -> Option<usize> {
    let flags = PAGE_READWRITE
        | if large {
            SEC_COMMIT | SEC_LARGE_PAGES
        } else if on_wine() {
            SEC_COMMIT
        } else {
            SEC_RESERVE
        };
    let size = bytes as u64;
    // SAFETY: an anonymous, page file backed section. The arguments are plain values.
    let handle = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), flags, (size >> 32) as u32, size as u32, std::ptr::null()) };
    (!handle.is_null()).then_some(handle as usize)
}

fn close_section(handle: usize) {
    // SAFETY: `handle` is a section handle this module created and no longer uses.
    unsafe { CloseHandle(handle as HANDLE) };
}

/// One backing section: `pages` pages of the pool from page index `start`. The engine keeps two
/// views of the whole section for itself, read only (the copy source of the GPU mirror) and read
/// write (its own writes, which must not fault). They are never protected and never given to the
/// title, and a view of a pagefile section adds no commit.
struct Section {
    handle: usize,
    start: usize,
    pages: usize,
    ro_view: usize,
    rw_view: usize,
    /// Per page of the section, the writes that are copying into it right now (`Engine::write_frames`,
    /// outside every lock). A page that is pinned is not given back to the pool until the copy ends.
    /// Heap memory that stays put when `State::sections` grows.
    pins: Box<[AtomicU32]>,
    /// One bit per page of the section: it is known to be committed, which `commit_page` then does not ask the
    /// system again. A committed page of the private view stays committed (it is only ever reset).
    committed: Box<[AtomicU64]>,
}

impl Section {
    /// The section over `handle`, with its two private views. The handle is closed on failure.
    fn new(handle: usize, start: usize, pages: usize, call: &'static str) -> Result<Section, EngineError> {
        let bytes = pages * PAGE_BYTES;
        // SAFETY: `handle` is a live section of `bytes` bytes, the views cover all of it.
        let (ro, rw) = unsafe { (MapViewOfFile(handle as HANDLE, FILE_MAP_READ, 0, 0, bytes), MapViewOfFile(handle as HANDLE, FILE_MAP_WRITE, 0, 0, bytes)) };
        if ro.Value.is_null() || rw.Value.is_null() {
            let error = Api::last_error(call);
            // SAFETY: each non null view was just mapped above.
            unsafe {
                if !ro.Value.is_null() {
                    UnmapViewOfFile(ro);
                }
                if !rw.Value.is_null() {
                    UnmapViewOfFile(rw);
                }
            }
            close_section(handle);
            return Err(error);
        }
        Ok(Section {
            handle,
            start,
            pages,
            ro_view: ro.Value as usize,
            rw_view: rw.Value as usize,
            pins: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            committed: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        })
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: both views were mapped by `Section::new` and nothing uses them any more.
        unsafe {
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ro_view as *mut c_void });
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.rw_view as *mut c_void });
        }
        close_section(self.handle);
    }
}

/// What the engine holds, in bytes and pages (`Engine::footprint`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footprint {
    pub sections: usize,
    /// The size of the backing sections.
    pub section_bytes: usize,
    /// The bytes of the sections that are committed, which is what they charge.
    pub committed_bytes: usize,
    pub pool_pages: usize,
    /// Pages out of the pool.
    pub used_pages: usize,
    /// Pages that a view maps.
    pub mapped_pages: usize,
    /// The bytes of the commits with a console page bit (`commit_region`).
    pub arena_bytes: usize,
    /// The pages of those that the GPU was given.
    pub given_pages: usize,
    /// Private memory placed inside reservations.
    pub private_bytes: usize,
}

/// Whether the page at `address` is committed.
/// Where the time of `take_dirty` goes, logged every 200 calls: waiting for the lock, looking at
/// the pending pages (with a `VirtualQuery` each), and protecting them again.
fn take_stats(pages: usize, protected: usize, runs: usize, retries: usize, pending: usize, queries: usize, times: [std::time::Duration; 3]) {
    static STATS: Mutex<(u64, u64, u64, [u128; 3])> = Mutex::new((0, 0, 0, [0; 3]));
    static LOOKED: Mutex<(u64, u64, u64, u64)> = Mutex::new((0, 0, 0, 0));
    let looked = {
        let mut l = LOOKED.lock().unwrap_or_else(|e| e.into_inner());
        l.0 += pending as u64;
        l.1 += queries as u64;
        l.2 += runs as u64;
        l.3 += retries as u64;
        *l
    };
    let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
    s.0 += 1;
    s.1 += pages as u64;
    s.2 += protected as u64;
    for (sum, t) in s.3.iter_mut().zip(times) {
        *sum += t.as_micros();
    }
    if s.0.is_multiple_of(200) {
        decant_log::trace(&format!(
            "[xmem] written pages taken: {} calls, {} pages, {} pages protected again ({} runs, {} retries); {} us waiting for the lock, {} us looking at the pages ({} pending pages looked at, {} VirtualQuery), {} us protecting
",
            s.0, s.1, s.2, looked.2, looked.3, s.3[0], s.3[1], looked.0, looked.1, s.3[2]
        ));
        *s = (s.0, 0, 0, [0; 3]);
        *LOOKED.lock().unwrap_or_else(|e| e.into_inner()) = (0, 0, 0, 0);
    }
}

/// The last region `VirtualQuery` described, so that the next pages in it are not queried again: a region
/// is a run of pages with the same state, and `take_dirty` asked for each 64 KiB page on its own (a call
/// per page, microseconds each under the lock while a large world loaded).
#[derive(Default)]
struct CommittedRuns {
    start: usize,
    end: usize,
    committed: bool,
    /// `VirtualQuery` calls made.
    queries: usize,
}

impl CommittedRuns {
    /// Whether `address` is committed. `query` gives, for an address, the `[start, end)` of its region
    /// and whether it is committed (`None` when the query fails: not committed).
    fn is_committed(&mut self, address: usize, query: impl FnOnce(usize) -> Option<(usize, usize, bool)>) -> bool {
        if address >= self.start && address < self.end {
            return self.committed;
        }
        self.queries += 1;
        match query(address) {
            Some((start, end, committed)) if end > address => {
                (self.start, self.end, self.committed) = (start.min(address), end, committed);
                committed
            }
            _ => {
                (self.start, self.end) = (0, 0);
                false
            }
        }
    }
}

/// The region of `address` and whether it is committed, from `VirtualQuery`.
fn query_region(address: usize) -> Option<(usize, usize, bool)> {
    // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid output buffer of the size passed.
    if unsafe { VirtualQuery(address as *const c_void, &mut info, size_of_val(&info)) } == 0 {
        return None;
    }
    let start = info.BaseAddress as usize;
    Some((start, start + info.RegionSize, info.State == MEM_COMMIT))
}

fn page_committed(address: usize) -> bool {
    // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid output buffer of the size passed.
    unsafe { VirtualQuery(address as *const c_void, &mut info, size_of_val(&info)) != 0 && info.State == MEM_COMMIT }
}

/// The committed bytes of `[base, base + len)`, walked with `VirtualQuery`.
fn committed_bytes(base: usize, len: usize) -> usize {
    let end = base + len;
    let (mut at, mut total) = (base, 0);
    while at < end {
        // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a valid output buffer of the size passed.
        if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize).min(end);
        if next <= at {
            break;
        }
        if info.State == MEM_COMMIT {
            total += next - at;
        }
        at = next;
    }
    total
}

/// The commit charge the system has left, in bytes (`GlobalMemoryStatusEx`). 0 when the call fails.
fn available_commit() -> u64 {
    // SAFETY: MEMORYSTATUSEX is plain data. The length field is set as the call requires.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a valid output buffer with its length set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 { 0 } else { status.ullAvailPageFile }
}

struct State {
    sections: Vec<Section>,
    pages: PageAllocator,
    regions: Regions,
    /// Commits that `place` backed with pool pages: base address to first frame and page count.
    commits: BTreeMap<usize, (u32, usize)>,
    dirty: Arc<Mutex<Dirty>>,
    protect_sync: Arc<ProtectSync>,
    /// The commits of `commit_region` by base, with one flag per page: the GPU was given an address in
    /// it (`gpu_use`). Only such pages are tracked and reported to the GPU mirror.
    on_demand: BTreeMap<usize, Vec<bool>>,
}

const PROTECT_STRIPES: usize = 256;
const PROTECT_STRIPE_SHIFT: usize = 16;
const MAX_PROTECT_CHUNK_BYTES: usize = 32 * PAGE_BYTES;

type StripeMask = [u64; 4];
type RunSnapshot = (usize, usize, StripeMask, MaskGens);

const INLINE_STRIPE_CAP: usize = 8;

enum MaskGens {
    Inline { len: u8, buf: [u64; INLINE_STRIPE_CAP] },
    Heap(Vec<u64>),
}

impl MaskGens {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::Inline { len, buf } => &buf[..usize::from(*len)],
            Self::Heap(v) => v.as_slice(),
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

#[allow(dead_code)]
enum MaskGuards<'a> {
    Inline([Option<MutexGuard<'a, ()>>; INLINE_STRIPE_CAP]),
    Heap(Vec<MutexGuard<'a, ()>>),
}

/// Coordinates page protection changes outside the `dirty` mutex so `VirtualProtect` and
/// `VirtualQuery` syscalls do not block fault, I/O preparation, or tracking updates.
struct ProtectSync {
    locks: Vec<Mutex<()>>,
    gens: Vec<AtomicU64>,
    in_flight_rw: Vec<AtomicU64>,
    pending: AtomicBool,
}

impl Default for ProtectSync {
    fn default() -> Self {
        let mut locks = Vec::with_capacity(PROTECT_STRIPES);
        let mut gens = Vec::with_capacity(PROTECT_STRIPES);
        let mut in_flight_rw = Vec::with_capacity(PROTECT_STRIPES);
        for _ in 0..PROTECT_STRIPES {
            locks.push(Mutex::new(()));
            gens.push(AtomicU64::new(0));
            in_flight_rw.push(AtomicU64::new(0));
        }
        Self { locks, gens, in_flight_rw, pending: AtomicBool::new(false) }
    }
}

impl ProtectSync {
    fn stripe(addr: usize) -> usize {
        (addr >> PROTECT_STRIPE_SHIFT) & (PROTECT_STRIPES - 1)
    }

    fn set_bit(mask: &mut StripeMask, bit: usize) {
        mask[bit >> 6] |= 1u64 << (bit & 63);
    }

    fn merge_mask(dst: &mut StripeMask, src: StripeMask) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d |= s;
        }
    }

    fn mask_count(mask: StripeMask) -> usize {
        mask.iter().map(|w| w.count_ones() as usize).sum()
    }

    fn for_each_stripe(mask: StripeMask, mut f: impl FnMut(usize)) {
        for (word_idx, mut word) in mask.into_iter().enumerate() {
            let base = word_idx << 6;
            while word != 0 {
                let s = base + word.trailing_zeros() as usize;
                f(s);
                word &= word - 1;
            }
        }
    }

    fn stripe_mask(start: usize, bytes: usize) -> StripeMask {
        if bytes == 0 {
            return [0; 4];
        }
        let first = start >> PROTECT_STRIPE_SHIFT;
        let last = (start + bytes - 1) >> PROTECT_STRIPE_SHIFT;
        let span = last - first + 1;
        if span >= PROTECT_STRIPES {
            return [u64::MAX; 4];
        }
        let mut mask = [0u64; 4];
        for k in 0..span {
            Self::set_bit(&mut mask, (first + k) & (PROTECT_STRIPES - 1));
        }
        mask
    }

    fn has_pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }

    fn set_pending(&self, has: bool) {
        self.pending.store(has, Ordering::Release);
    }

    fn bump_addrs(&self, addrs: &[usize]) {
        let mut mask = [0u64; 4];
        for &addr in addrs {
            Self::set_bit(&mut mask, Self::stripe(addr));
        }
        Self::for_each_stripe(mask, |s| {
            self.gens[s].fetch_add(1, Ordering::Release);
        });
    }

    fn read_mask_gens(&self, mask: StripeMask) -> MaskGens {
        let count = Self::mask_count(mask);
        if count <= INLINE_STRIPE_CAP {
            let mut buf = [0u64; INLINE_STRIPE_CAP];
            let mut len = 0usize;
            Self::for_each_stripe(mask, |s| {
                buf[len] = self.gens[s].load(Ordering::Acquire);
                len += 1;
            });
            MaskGens::Inline { len: len as u8, buf }
        } else {
            let mut gens = Vec::with_capacity(count);
            Self::for_each_stripe(mask, |s| {
                gens.push(self.gens[s].load(Ordering::Acquire));
            });
            MaskGens::Heap(gens)
        }
    }

    fn gens_match(&self, mask: StripeMask, expected: &[u64]) -> bool {
        let mut idx = 0usize;
        for (word_idx, mut word) in mask.into_iter().enumerate() {
            let base = word_idx << 6;
            while word != 0 {
                let s = base + word.trailing_zeros() as usize;
                if self.gens[s].load(Ordering::Acquire) != expected[idx] {
                    return false;
                }
                idx += 1;
                word &= word - 1;
            }
        }
        true
    }

    fn lock_mask(&self, mask: StripeMask) -> MaskGuards<'_> {
        let count = Self::mask_count(mask);
        if count <= INLINE_STRIPE_CAP {
            let mut buf: [Option<MutexGuard<'_, ()>>; INLINE_STRIPE_CAP] = [const { None }; INLINE_STRIPE_CAP];
            let mut idx = 0usize;
            Self::for_each_stripe(mask, |s| {
                buf[idx] = Some(self.locks[s].lock().unwrap_or_else(PoisonError::into_inner));
                idx += 1;
            });
            MaskGuards::Inline(buf)
        } else {
            let mut guards = Vec::with_capacity(count);
            Self::for_each_stripe(mask, |s| {
                guards.push(self.locks[s].lock().unwrap_or_else(PoisonError::into_inner));
            });
            MaskGuards::Heap(guards)
        }
    }

    fn protect_rw_locked(&self, addrs: Vec<usize>) {
        for (start, bytes) in track::runs(addrs) {
            let _guards = self.lock_mask(Self::stripe_mask(start, bytes));
            Engine::protect_range(start, bytes, PAGE_READWRITE);
        }
    }

    fn snapshot_runs(&self, addrs: Vec<usize>) -> Vec<RunSnapshot> {
        let mut runs = Vec::new();
        for (start, bytes) in track::runs(addrs) {
            let end = start + bytes;
            let mut at = start;
            while at < end {
                let cb = (end - at).min(MAX_PROTECT_CHUNK_BYTES);
                let mask = Self::stripe_mask(at, cb);
                let gens = self.read_mask_gens(mask);
                runs.push((at, cb, mask, gens));
                at += cb;
            }
        }
        runs
    }

    fn snapshot_rw_runs(&self, addrs: Vec<usize>) -> Vec<RunSnapshot> {
        let runs = self.snapshot_runs(addrs);
        for &(_, _, mask, _) in &runs {
            Self::for_each_stripe(mask, |s| {
                self.in_flight_rw[s].fetch_add(1, Ordering::Release);
            });
        }
        runs
    }

    fn release_in_flight_rw(&self, mask: StripeMask) {
        Self::for_each_stripe(mask, |s| {
            self.in_flight_rw[s].fetch_sub(1, Ordering::Release);
        });
    }

    fn wait_in_flight_rw(&self, mask: StripeMask) {
        Self::for_each_stripe(mask, |s| {
            while self.in_flight_rw[s].load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
        });
    }

    fn protect_runs_checked(&self, runs: Vec<RunSnapshot>, protect: u32, want_clean: bool, release_rw: bool, dirty: &Mutex<Dirty>) -> (usize, usize) {
        let mut protected_runs = 0usize;
        let mut retries = 0usize;
        let mut retry_addrs = Vec::new();
        let mut rw_masks_to_release = Vec::new();
        for (start, bytes, mask, expected_gens) in runs {
            let guards = self.lock_mask(mask);
            if self.gens_match(mask, expected_gens.as_slice()) {
                Engine::protect_range(start, bytes, protect);
                protected_runs += 1;
                drop(guards);
                if release_rw {
                    self.release_in_flight_rw(mask);
                }
            } else {
                drop(guards);
                retries += 1;
                retry_addrs.extend((start..start + bytes).step_by(PAGE_BYTES));
                if release_rw {
                    rw_masks_to_release.push(mask);
                }
            }
        }
        while !retry_addrs.is_empty() {
            let sub_runs = {
                let dirty_guard = dirty.lock().unwrap_or_else(PoisonError::into_inner);
                let keep = match dirty_guard.track.as_ref() {
                    Some(track) => retry_addrs
                        .into_iter()
                        .filter(|&addr| track.is_clean_addr(addr) == want_clean)
                        .collect::<Vec<_>>(),
                    None if want_clean => Vec::new(),
                    None => retry_addrs,
                };
                self.snapshot_runs(keep)
            };
            if sub_runs.is_empty() {
                break;
            }
            let mut next_retry = Vec::new();
            for (sub_s, sub_b, sub_mask, fresh_gens) in sub_runs {
                let guards = self.lock_mask(sub_mask);
                if self.gens_match(sub_mask, fresh_gens.as_slice()) {
                    Engine::protect_range(sub_s, sub_b, protect);
                    protected_runs += 1;
                } else {
                    retries += 1;
                    next_retry.extend((sub_s..sub_s + sub_b).step_by(PAGE_BYTES));
                }
                drop(guards);
            }
            retry_addrs = next_retry;
        }
        for mask in rw_masks_to_release {
            self.release_in_flight_rw(mask);
        }
        (protected_runs, retries)
    }

    fn protect_rw_runs(&self, runs: Vec<RunSnapshot>, dirty: &Mutex<Dirty>) {
        let _ = self.protect_runs_checked(runs, PAGE_READWRITE, false, true, dirty);
    }

    fn protect_ro_runs(&self, runs: Vec<RunSnapshot>, dirty: &Mutex<Dirty>) -> (usize, usize) {
        self.protect_runs_checked(runs, PAGE_READONLY, true, false, dirty)
    }

    fn ensure_rw(&self, page: usize) -> bool {
        let _guard = self.locks[Self::stripe(page)].lock().unwrap_or_else(PoisonError::into_inner);
        if !accessible(page, true) {
            Engine::protect_range(page, PAGE_BYTES, PAGE_READWRITE);
        }
        accessible(page, true)
    }
}

/// The pages a write holds before it spills to the heap (2 MiB of data), and the places a title sees
/// the bytes (`Engine::write_frames_buf`).
pub const INLINE_WRITE_PAGES: usize = 32;
pub const INLINE_WRITE_VIEWS: usize = 16;

/// Where the title sees the bytes of a write, as `(address, bytes)`.
pub type WrittenViews = SmallBuf<(usize, usize), INLINE_WRITE_VIEWS>;

/// Pins pool pages for a copy that runs outside the locks: each page counts one more writer in
/// `Section::pins` until this drops, and `Engine::wait_unpinned` makes whoever gives the page back wait.
struct PinGuard<'a>(&'a [(usize, usize, usize)]);

impl<'a> PinGuard<'a> {
    /// Pins the pages of `targets`, each `(pool index, address, address of the pin)`. The pins live in
    /// sections, which stay while the engine does, and the caller borrows the engine for as long as this lives.
    fn pin(targets: &'a [(usize, usize, usize)]) -> Self {
        for &(_, _, pin) in targets {
            // SAFETY: `pin` is the address of an `AtomicU32` of `Section::pins`, heap memory of a section that
            // is only dropped with the engine.
            unsafe { &*(pin as *const AtomicU32) }.fetch_add(1, Ordering::AcqRel);
        }
        PinGuard(targets)
    }
}

impl Drop for PinGuard<'_> {
    fn drop(&mut self) {
        for &(_, _, pin) in self.0 {
            // SAFETY: as in `pin`.
            unsafe { &*(pin as *const AtomicU32) }.fetch_sub(1, Ordering::Release);
        }
    }
}

/// The tracker and its pool views share a lock separate from address space inspection.
/// Mutations take the state lock before this lock and release it before protection syscalls.
#[derive(Default)]
struct Dirty {
    track: Option<Track>,
    sections: Vec<(usize, usize, usize)>,
    sync: Arc<ProtectSync>,
}

thread_local! {
    /// Set while this thread holds the engine lock. The fault handler takes the lock too, so a title
    /// view written under the lock (a bug) must not wait for itself: the handler lets such a fault through.
    static IN_ENGINE: Cell<bool> = const { Cell::new(false) };
}

/// The engine lock, which marks its thread in `IN_ENGINE` for as long as it is held.
struct Locked<'a>(MutexGuard<'a, State>);

impl Deref for Locked<'_> {
    type Target = State;
    fn deref(&self) -> &State {
        &self.0
    }
}

impl DerefMut for Locked<'_> {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.0
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        IN_ENGINE.with(|f| f.set(false));
    }
}

/// The most pages one section of a growth holds: 2 GiB, the size D3D12 was seen to open as a
/// GPU heap (a section of several GiB was refused).
pub const MAX_SECTION_PAGES: usize = (2usize << 30) / PAGE_BYTES;

/// Commits of `commit_region` lie below this address when Windows has room there: the page table of
/// the device memory window in `d3d12_xs.dll` (`window_service::TABLE_PAGES`) covers the first 16 GiB
/// of title addresses, so translated shaders can reach memory there by address.
pub const GPU_LOW_END: usize = 16 << 30;

/// The most physical pages the pool grows to by default: 16 GiB, the memory of an Xbox Series X.
pub const CONSOLE_POOL_PAGES: usize = (16usize << 30) / PAGE_BYTES;

pub struct Engine {
    api: Api,
    /// The most pages the pool may reach, whatever the system allows.
    limit_pages: usize,
    /// Whether tracking is on, read by the fault handler without the lock.
    tracking: AtomicBool,
    /// The sequence number of the last journal record, readable without the lock.
    journal_seq: AtomicU64,
    /// What changed in the address space, for the GPU mirror, locked separately from `state`.
    journal: Mutex<Journal>,
    /// Boxed so that its address stays put when the engine moves: the fault handler finds it in `STATES`.
    state: Box<Mutex<State>>,
    dirty: Arc<Mutex<Dirty>>,
    protect_sync: Arc<ProtectSync>,
    /// Minimum interval in milliseconds between batched re-protection waves.
    reprotect_interval_ms: AtomicU64,
}

/// The state of every live engine, by address: the fault handler commits pages of any of them (the one
/// of `xmem.dll`, and those the tests make).
static STATES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Takes the engine lock of `state` and marks the thread in `IN_ENGINE` for as long as it is held.
fn lock_state(state: &Mutex<State>) -> Locked<'_> {
    let guard = state.lock().unwrap_or_else(PoisonError::into_inner);
    IN_ENGINE.with(|f| f.set(true));
    Locked(guard)
}

/// `Engine::commit_on_access` for every live engine.
fn commit_on_access(address: usize) -> bool {
    let states = STATES.lock().unwrap_or_else(PoisonError::into_inner);
    // SAFETY: an engine removes its state from the list (under this lock) before the state is freed.
    states.iter().any(|&state| Engine::commit_on_access_in(unsafe { &*(state as *const Mutex<State>) }, address))
}

static ENGINE: OnceLock<Result<Engine, EngineError>> = OnceLock::new();
static LARGE_PAGES_LOG: Once = Once::new();
static FAULT_HANDLER: Once = Once::new();

const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// Installs, once per process, the vectored handler that backs a page on demand.
///
/// A title may write into a 64 KiB range of a reservation before the call that places pages there
/// has run (another thread is still inside `XMemMapPhysicalPages`). Windows faults that write. The
/// handler backs the page at once, with pool pages when some are free and private memory otherwise,
/// the same way `place` does for a commit, and the write is retried. Only an address that lies in
/// a placeholder owned by an engine is handled. Any other fault is a real bug and still crashes.
fn install_fault_handler() {
    FAULT_HANDLER.call_once(|| {
        // SAFETY: the handler only reads the exception record that Windows passes in and touches
        // memory that an engine owns.
        unsafe { AddVectoredExceptionHandler(1, Some(fault_handler)) };
    });
}

/// How the faulting instruction touched memory, from `ExceptionInformation[0]`.
const ACCESS_WRITE: usize = 1;
const ACCESS_EXECUTE: usize = 8;

unsafe extern "system" fn fault_handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows passes a valid `EXCEPTION_POINTERS` and exception record for the call.
    let record = unsafe { &*(*info).ExceptionRecord };
    let is_access_violation = record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS && record.NumberParameters >= 2;
    // A fault while this thread holds the engine lock would wait for itself: let it crash loudly.
    if !is_access_violation || IN_ENGINE.with(Cell::get) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let access = record.ExceptionInformation[0];
    let address = record.ExceptionInformation[1];
    let write = access == ACCESS_WRITE;
    let engine = Engine::started();
    let handled = (write && engine.is_some_and(|e| e.on_write_fault(address)))
        || (access != ACCESS_EXECUTE && commit_on_access(address))
        || (write && engine.is_some_and(|e| e.on_write_fault(address)))
        || engine.is_some_and(|e| e.fault_in(address, write));
    if handled { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
}

/// Whether the page at `page` is committed now and allows the access that faulted (a write needs
/// a writable protection). Used when another thread mapped the page while the fault waited.
fn accessible(page: usize, write: bool) -> bool {
    // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    if unsafe { VirtualQuery(page as *const c_void, &mut info, size_of_val(&info)) } == 0 {
        return false;
    }
    const MEM_COMMIT: u32 = 0x1000;
    const PAGE_GUARD: u32 = 0x100;
    let p = info.Protect;
    if info.State != MEM_COMMIT || p & PAGE_GUARD != 0 || p & 0xff == PAGE_NOACCESS || p == 0 {
        return false;
    }
    // writable: PAGE_READWRITE 0x04, PAGE_WRITECOPY 0x08, PAGE_EXECUTE_READWRITE 0x40, PAGE_EXECUTE_WRITECOPY 0x80
    !write || p & (0x04 | 0x08 | 0x40 | 0x80) != 0
}

impl Engine {
    /// An engine over a backing section of `capacity_pages` pages of 64 KiB, created on first use.
    /// The pool grows up to the console's memory (`CONSOLE_POOL_PAGES`): a title cannot expect
    /// more physical pages than its console has, and without a bound a request for far more grew
    /// as far as the commit charge allowed (xmem_abitest's "more pages than the section has" then
    /// succeeded on a machine with a large page file).
    pub fn load(capacity_pages: usize) -> Result<Engine, EngineError> {
        Engine::load_limited(capacity_pages, CONSOLE_POOL_PAGES.max(capacity_pages).min(MAX_POOL_PAGES))
    }

    /// As `load`, and the pool never grows past `limit_pages` pages.
    pub fn load_limited(capacity_pages: usize, limit_pages: usize) -> Result<Engine, EngineError> {
        let api = Api::load()?;
        install_fault_handler();
        let protect_sync = Arc::new(ProtectSync::default());
        let dirty = Arc::new(Mutex::new(Dirty { track: None, sections: Vec::new(), sync: Arc::clone(&protect_sync) }));
        let state = State {
            sections: Vec::new(),
            pages: PageAllocator::new(capacity_pages),
            regions: Regions::new(),
            commits: BTreeMap::new(),
            dirty: Arc::clone(&dirty),
            protect_sync: Arc::clone(&protect_sync),
            on_demand: BTreeMap::new(),
        };
        let state = Box::new(Mutex::new(state));
        STATES.lock().unwrap_or_else(PoisonError::into_inner).push(&raw const *state as usize);
        let reprotect_interval_ms = AtomicU64::new(track::reprotect_interval().as_millis() as u64);
        Ok(Engine { api, limit_pages, tracking: AtomicBool::new(false), journal_seq: AtomicU64::new(0), journal: Mutex::new(Journal::new()), state, dirty, protect_sync, reprotect_interval_ms })
    }

    /// The engine of this process. The size of the section is `DECANT_XMEM_PHYSICAL_MIB` MiB when
    /// that variable is set, 1 GiB otherwise.
    pub fn global() -> Result<&'static Engine, EngineError> {
        ENGINE
            .get_or_init(|| {
                crate::memsample::start();
                // Kernel writes (file reads) into pages that are not committed yet fail instead of faulting.
                crate::io_guard::install();
                let pages = std::env::var("DECANT_XMEM_PHYSICAL_MIB").ok().and_then(|v| v.trim().parse::<usize>().ok()).filter(|&m| (1..=65536).contains(&m));
                Engine::load(pages.map_or(DEFAULT_CAPACITY_PAGES, |mib| mib * (1 << 20) / PAGE_BYTES))
            })
            .as_ref()
            .map_err(|e| *e)
    }

    /// The engine when it already started, without starting it.
    pub fn started() -> Option<&'static Engine> {
        ENGINE.get().and_then(|e| e.as_ref().ok())
    }

    fn lock(&self) -> Locked<'_> {
        lock_state(&self.state)
    }

    /// Records a change for the GPU mirror.
    fn note(&self, st: &mut State, kind: u32, va: usize, pages: usize, first_page: u32) {
        self.note_flags(st, kind, va, pages, first_page, 0);
    }

    fn note_flags(&self, _st: &mut State, kind: u32, va: usize, pages: usize, first_page: u32, flags: u32) {
        let seq = self.journal.lock().unwrap_or_else(PoisonError::into_inner).push(kind, va, pages, first_page, flags);
        self.journal_seq.store(seq, Ordering::Release);
    }

    /// Makes sure the first backing section exists, created on first use. Large pages are tried
    /// first and need the lock memory privilege. Without it normal pages serve, and that is logged once.
    fn section(&self, st: &mut State, dirty: &mut Dirty) -> Result<(), EngineError> {
        if !st.sections.is_empty() {
            return Ok(());
        }
        let bytes = st.pages.capacity() * PAGE_BYTES;
        // SAFETY: GetLargePageMinimum has no precondition.
        let granule = unsafe { GetLargePageMinimum() };
        let large = (granule != 0 && bytes.is_multiple_of(granule) && enable_lock_memory_privilege())
            .then(|| create_section(bytes, true))
            .flatten()
            .filter(|&handle| {
                let works = self.probe(handle);
                if !works {
                    close_section(handle);
                }
                works
            });
        let handle = match large {
            Some(handle) => handle,
            None => {
                LARGE_PAGES_LOG.call_once(|| {
                    decant_log::trace("[xmem] pages physiques : grandes pages indisponibles (privilege ou memoire), pages normales utilisees\r\n");
                });
                create_section(bytes, false).ok_or_else(|| Api::last_error("CreateFileMappingW"))?
            }
        };
        let pages = st.pages.capacity();
        st.sections.push(Section::new(handle, 0, pages, "MapViewOfFile")?);
        dirty.sections.push((0, pages, st.sections.last().unwrap().ro_view));
        Ok(())
    }

    /// Grows the pool by `extra` pages of normal pages, in sections of at most `MAX_SECTION_PAGES`:
    /// D3D12 opens a section of 2 GiB as a heap for the GPU but refused a page pool of several GiB
    /// as one section (E_INVALIDARG), so a larger growth becomes several sections.
    fn grow(&self, st: &mut State, dirty: &mut Dirty, extra: usize) -> Result<(), EngineError> {
        let mut left = extra;
        while left > 0 {
            let pages = left.min(MAX_SECTION_PAGES);
            let handle = create_section(pages * PAGE_BYTES, false).ok_or_else(|| Api::last_error("CreateFileMappingW (agrandir)"))?;
            let start = st.pages.capacity();
            st.sections.push(Section::new(handle, start, pages, "MapViewOfFile (agrandir)")?);
            dirty.sections.push((start, pages, st.sections.last().unwrap().ro_view));
            st.pages.grow(pages);
            let total = st.pages.capacity();
            if let Some(track) = dirty.track.as_mut() {
                track.grow(total);
            }
            left -= pages;
        }
        Ok(())
    }

    /// The section that holds page `index`, with the byte offset of the page in it and the pages
    /// from there to the end of the section.
    fn locate(st: &State, index: usize) -> Result<(usize, u64, usize), EngineError> {
        let s = st.sections.iter().find(|s| index >= s.start && index < s.start + s.pages).ok_or(PageError::BadFrame(0))?;
        Ok((s.handle, ((index - s.start) * PAGE_BYTES) as u64, s.start + s.pages - index))
    }

    /// Whether a section can be mapped one 64 KiB page at a time at a 64 KiB offset, which the
    /// engine does. Large page sections may refuse it.
    fn probe(&self, section: usize) -> bool {
        let Ok(at) = self.api.reserve(0, PAGE_BYTES) else {
            return false;
        };
        let mapped = self.api.map(section, at, PAGE_BYTES as u64, PAGE_BYTES, PAGE_READWRITE).is_ok();
        let freed = if mapped { self.api.unmap(at, false).is_ok() } else { self.api.release_placeholder(at).is_ok() };
        mapped && freed
    }

    /// Reserves a placeholder of `size` bytes at `address` (0: anywhere) and registers it. `protect`
    /// is the protection of the reservation, which every mapping inside it takes.
    pub fn reserve(&self, address: usize, size: usize, protect: u32) -> Result<usize, EngineError> {
        let mut st = self.lock();
        let at = if address != 0 {
            self.api.reserve(address, size)?
        } else {
            let (at, placement) = crate::reserve_plain(|| self.api.reserve_above(GPU_LOW_END, size), || self.api.reserve(0, size))?;
            match placement {
                crate::PlainPlacement::Above => {
                    decant_log::trace(&format!("[xmem] reservation 0x{size:x} octets : 0x{at:x} (au-dessus de 16 Gio)\r\n"));
                }
                crate::PlainPlacement::Anywhere => {
                    decant_log::trace(&format!("[xmem] reservation 0x{size:x} octets : 0x{at:x} (adresse quelconque, au-dessus de 16 Gio refuse)\r\n"));
                }
            }
            at
        };
        if let Err(e) = st.regions.insert_placeholder(at, size, protect) {
            let _ = self.api.release_placeholder(at);
            return Err(e.into());
        }
        self.note(&mut st, journal::RESERVE, at, size / PAGE_BYTES, 0);
        Ok(at)
    }

    /// Places memory at `[address, address + size)` inside a reservation of the engine: the
    /// placeholder is cut to exactly that range (front, middle or tail cut) and filled, committed
    /// with `protect` when `commit`. `address` and `size` are on 64 KiB. A commit is backed by one
    /// run of pages of one backing section, so that the GPU sees it as one span, and the range is a
    /// view of them. When the pool cannot give such a run (or `protect` is read only, which would
    /// leave no way to clear old bytes), the range is private memory as before. On failure the cuts
    /// are joined again.
    /// A commit over a range that is committed already (views or private memory, no placeholder left)
    /// changes nothing and succeeds, as `VirtualAlloc` does.
    pub fn place(&self, address: usize, size: usize, commit: bool, protect: u32) -> Result<Placed, EngineError> {
        let mut st = self.lock();
        if commit
            && let Ok(found) = st.regions.covering(address, size)
            && found.iter().all(|(_, r)| r.kind != Kind::Placeholder)
        {
            return Ok(Placed { cuts: 0, committed: true, pages: found.iter().all(|(_, r)| matches!(r.kind, Kind::View { .. })) });
        }
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        self.place_locked(&mut st, dirty, address, size, commit, protect)
    }

    fn place_locked(&self, st: &mut State, dirty: &mut Dirty, address: usize, size: usize, commit: bool, protect: u32) -> Result<Placed, EngineError> {
        if commit && view_protection(protect) == PAGE_READWRITE {
            match self.commit_pages(st, dirty, address, size, protect) {
                Ok(cuts) => return Ok(Placed { cuts, committed: true, pages: true }),
                Err(EngineError::OutOfPages) => {
                    decant_log::trace(&format!("[xmem] commit 0x{address:x}, 0x{size:x} octets : pas de suite de pages libre, memoire privee a la place\r\n"));
                }
                Err(e) => return Err(e),
            }
        }
        let cuts = self.carve(st, address, size)?;
        if let Err(e) = self.api.replace(address, size, commit, protect) {
            let _ = self.coalesce(st, address, size);
            return Err(e);
        }
        st.regions.set_allocation(address)?;
        Ok(Placed { cuts, committed: commit, pages: false })
    }

    /// `XMemVirtualAlloc` with `MEM_COMMIT` and a console page bit: reserves `size` bytes (at `address`,
    /// or below `GPU_LOW_END` aligned on `align` when it is 0, anywhere when there is no room there) and
    /// commits them with pool pages, as `place` does. The GPU mirror learns of the pages only as the GPU is
    /// given addresses in them (`gpu_use`): until then they cost the mirror nothing. Returns the base and
    /// whether pool pages back it (else private memory, which the GPU cannot see). `release` frees it.
    pub fn commit_region(&self, address: usize, size: usize, align: usize, protect: u32) -> Result<(usize, bool), EngineError> {
        if size == 0 || !size.is_multiple_of(PAGE_BYTES) {
            return Err(EngineError::Invalid("taille de commit"));
        }
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        let at = if address != 0 {
            self.api.reserve(address, size)?
        } else {
            match self.api.reserve_below(size, GPU_LOW_END, align.max(PAGE_BYTES)) {
                Ok(at) => {
                    decant_log::trace(&format!("[xmem] commit 0x{size:x} octets : 0x{at:x} (exigence sous 16 Gio)\r\n"));
                    at
                }
                Err(e) => {
                    let probe_align = align.max(PAGE_BYTES);
                    match crate::probe_reserve(size, GPU_LOW_END, probe_align, |addr| self.api.reserve(addr, size).is_ok()) {
                        Some(at) => {
                            decant_log::trace(&format!("[xmem] commit 0x{size:x} octets : 0x{at:x} (sondage sous 16 Gio, exigence refusee : {e})\r\n"));
                            at
                        }
                        None => {
                            let at = self.api.reserve(0, size)?;
                            decant_log::trace(&format!("[xmem] commit 0x{size:x} octets : 0x{at:x} (adresse quelconque, sondage sous 16 Gio epuise)\r\n"));
                            at
                        }
                    }
                }
            }
        };
        if let Err(e) = st.regions.insert_placeholder(at, size, protect) {
            let _ = self.api.release_placeholder(at);
            return Err(e.into());
        }
        st.on_demand.insert(at, vec![false; size / PAGE_BYTES]);
        self.note_flags(&mut st, journal::RESERVE, at, size / PAGE_BYTES, 0, journal::ON_DEMAND);
        match self.place_locked(&mut st, dirty, at, size, true, protect) {
            Ok(placed) => Ok((at, placed.pages)),
            Err(e) => {
                drop(dirty_guard);
                drop(st);
                let _ = self.release(at);
                Err(e)
            }
        }
    }

    /// Whether `address` is the base of a commit of `commit_region`.
    pub fn is_region(&self, address: usize) -> bool {
        self.lock().on_demand.contains_key(&address)
    }

    /// Whether the page at `addr` reaches the GPU mirror: any page outside the commits of `commit_region`,
    /// and the pages of those that `gpu_use` asked for.
    fn shown(st: &State, addr: usize) -> bool {
        match st.on_demand.range(..=addr).next_back() {
            Some((&base, active)) if addr - base < active.len() * PAGE_BYTES => active[(addr - base) / PAGE_BYTES],
            _ => true,
        }
    }

    /// The GPU is given `[address, address + len)`: the pages of it in commits of `commit_region` that
    /// it was not given before reach the GPU mirror from now on (dirty, tracked, reported as `MAP`).
    /// Returns how many pages that made.
    pub fn gpu_use(&self, address: usize, len: usize) -> usize {
        let Some(end) = address.checked_add(len.max(1)) else { return 0 };
        let mut st = self.lock();
        let Some((&base, active)) = st.on_demand.range(..=address).next_back() else { return 0 };
        let pages = active.len();
        if address - base >= pages * PAGE_BYTES {
            return 0;
        }
        let first = (address - base) / PAGE_BYTES;
        let last = (end - base).div_ceil(PAGE_BYTES).min(pages);
        let new: Vec<usize> = (first..last).filter(|&k| !active[k]).collect();
        if new.is_empty() {
            return 0;
        }
        if let Some(active) = st.on_demand.get_mut(&base) {
            for &k in &new {
                active[k] = true;
            }
        }
        // (title page address, pool page index, read write) of each page that a view maps.
        let mut mapped: Vec<(usize, u32, bool)> = Vec::new();
        for &k in &new {
            let addr = base + k * PAGE_BYTES;
            if let Some((at, region)) = st.regions.containing(addr)
                && let Kind::View { first_frame } = region.kind
                && let Some(index) = index_of(first_frame)
            {
                mapped.push((addr, (index + (addr - at) / PAGE_BYTES) as u32, view_protection(region.protect) == PAGE_READWRITE));
            }
        }
        if !mapped.is_empty() {
            let protect_sync = Arc::clone(&st.protect_sync);
            let dirty_state = Arc::clone(&st.dirty);
            let chunks = {
                let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
                let mut open = Vec::new();
                if let Some(track) = dirty_guard.track.as_mut() {
                    for &(addr, page, rw) in &mapped {
                        open.extend(track.on_map(page, addr, rw));
                    }
                    protect_sync.bump_addrs(&open);
                    if track.pending_len() > 0 {
                        protect_sync.set_pending(true);
                    }
                }
                protect_sync.snapshot_rw_runs(open)
            };
            protect_sync.protect_rw_runs(chunks, &dirty_state);
        }
        let mut i = 0;
        while i < mapped.len() {
            let (addr, page, _) = mapped[i];
            let mut n = 1;
            while i + n < mapped.len() && mapped[i + n].0 == addr + n * PAGE_BYTES && mapped[i + n].1 == page + n as u32 {
                n += 1;
            }
            self.note(&mut st, journal::MAP, addr, n, page);
            i += n;
        }
        new.len()
    }

    /// The commit of `place` in pages: a run of the pool mapped over the range, old bytes cleared.
    /// Returns the number of cuts made. `OutOfPages` when no run can be had.
    fn commit_pages(&self, st: &mut State, dirty: &mut Dirty, address: usize, size: usize, protect: u32) -> Result<usize, EngineError> {
        if size == 0 || !size.is_multiple_of(PAGE_BYTES) {
            return Err(EngineError::Invalid("taille de commit"));
        }
        let count = size / PAGE_BYTES;
        self.section(st, dirty)?;
        self.make_room(st, dirty, count, false)?;
        let first = st.pages.allocate_run(count).ok_or(EngineError::OutOfPages)?;
        let frames: Vec<u32> = (0..count as u32).map(|k| first + k * FRAMES_PER_PAGE).collect();
        self.track_changed(dirty, &frames);
        let mapped = self.carve(st, address, size).and_then(|cuts| self.map_one(st, dirty, address, first, count, view_protection(protect)).map(|()| cuts));
        let cuts = match mapped {
            Ok(cuts) => cuts,
            Err(e) => {
                let _ = st.pages.free(&frames);
                let _ = self.coalesce(st, address, size);
                return Err(e);
            }
        };
        for &frame in &frames {
            if st.pages.is_dirty(frame) {
                // Through the engine's own read write view, never the title's: that one may be
                // protected by the dirty tracking, and a fault under the lock cannot be served. A page
                // that was never committed holds zeros already (and a write would fault).
                if let Some(view) = index_of(frame).and_then(|i| Self::private_page(st, i)).filter(|&v| page_committed(v)) {
                    // SAFETY: `view` is a page of a private read write view of the section, 64 KiB long.
                    unsafe { std::ptr::write_bytes(view as *mut u8, 0, PAGE_BYTES) };
                }
                st.pages.mark_clean(frame);
            }
        }
        st.commits.insert(address, (first, count));
        Ok(cuts)
    }

    /// Cuts the placeholder to exactly `[address, address + size)` and keeps it a placeholder, for
    /// physical pages to be mapped into it. A range that is cut already (by the title) is accepted.
    /// Returns the number of cuts made.
    pub fn cut(&self, address: usize, size: usize) -> Result<usize, EngineError> {
        let mut st = self.lock();
        self.carve(&mut st, address, size)
    }

    /// Whether `address` is the base of memory that `place` put in a reservation.
    pub fn is_allocation(&self, address: usize) -> bool {
        let st = self.lock();
        st.commits.contains_key(&address) || st.regions.at(address).is_some_and(|r| r.kind == Kind::Allocation)
    }

    /// Gives the memory that `place` put at `address` back as a placeholder, joined with the
    /// placeholders next to it. The pages of a commit go back to the pool. Returns its size.
    pub fn unplace(&self, address: usize) -> Result<usize, EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        if let Some(&(first, count)) = st.commits.get(&address) {
            let size = count * PAGE_BYTES;
            self.clear_range(&mut st, dirty, address, size)?;
            st.commits.remove(&address);
            let frames: Vec<u32> = (0..count as u32).map(|k| first + k * FRAMES_PER_PAGE).collect();
            Self::wait_unpinned(&st, &frames);
            st.pages.free(&frames)?;
            Self::reset_unused_frames(&st, &frames);
            return Ok(size);
        }
        let size = st.regions.at(address).filter(|r| r.kind == Kind::Allocation).ok_or(RegionError::Unknown)?.size;
        self.api.restore(address, size)?;
        st.regions.clear_allocation(address)?;
        self.coalesce(&mut st, address, size)?;
        Ok(size)
    }

    /// Allocates `count` pages of the pool: one run inside one section when a section has it free,
    /// else a new section big enough for the whole request, else (the pool is at its limit) pages
    /// scattered over what is free, which may leave gaps for the GPU. The frame numbers come back
    /// in order. Fails when the pool is full.
    pub fn allocate_pages(&self, count: usize) -> Result<Vec<u32>, EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        self.section(&mut st, dirty)?;
        let free = st.pages.capacity() - st.pages.used_count();
        let grown = self.make_room(&mut st, dirty, count, true);
        // A section that could not be made does not fail a request the free pages can still serve.
        let outcome = match grown {
            Err(e) if free < count => Err(e),
            _ => st.pages.allocate(count).ok_or(EngineError::OutOfPages),
        };
        if let Ok(frames) = &outcome {
            self.track_changed(dirty, frames);
        }
        let (pool, used) = (st.pages.capacity(), st.pages.used_count());
        let refused = if outcome.is_err() { ", refuse" } else { "" };
        decant_log::trace(&format!("[xmem] XMemAllocatePhysicalPages : {count} pages demandees, reserve de {pool} pages, {used} en usage{refused}\r\n"));
        outcome
    }

    /// Writes `data` into pool pages through the engine's own read write view of their sections, so
    /// a page gets the bytes whether the title maps it or not (it may map it later). `data` starts at
    /// byte `offset` of the page of `frames[0]` and goes on page after page, one frame per 64 KiB.
    /// The pages become dirty for the GPU mirror. All or nothing: `frames` must cover the data and
    /// every frame the data reaches must be allocated. Returns where the title sees the bytes now: the
    /// written part of each view page that maps one of the pages, as `(address, bytes)`, adjacent parts
    /// joined (a texture placed there has to be read again).
    #[cfg(test)]
    pub fn write_frames(&self, frames: &[u32], offset: usize, data: &[u8]) -> Result<Vec<(usize, usize)>, EngineError> {
        self.write_frames_buf(frames, offset, data).map(|seen| seen.as_slice().to_vec())
    }

    /// `write_frames` without a heap allocation for a write of up to `INLINE_WRITE_PAGES` pages and
    /// `INLINE_WRITE_VIEWS` places the title sees it.
    ///
    /// In three steps, so that the copy of the bytes (the slow part, many 64 KiB pages) runs without
    /// any lock and several threads copy at the same time:
    /// 1. under the engine lock: the frames are checked, their pages committed and pinned (`Section::pins`),
    ///    and the views that show them looked up (`Regions::views_of_page`, no scan of every view);
    /// 2. no lock: the copy, into the private read write views, which are never protected, never unmapped
    ///    while the engine lives and never decommitted. A pinned page is not given back by `free_pages`,
    ///    `unplace` or `release` before the copy ends;
    /// 3. under the dirty lock only: the pages are noted as changed, after the bytes are in, so that the GPU
    ///    mirror never takes a page and then misses its bytes.
    ///
    /// Two writes that overlap on the same bytes at the same time leave either bytes or a mix of both
    /// (a title that does that has no defined result either), and no write is ever torn across pages it did
    /// not ask for.
    pub fn write_frames_buf(&self, frames: &[u32], offset: usize, data: &[u8]) -> Result<WrittenViews, EngineError> {
        if offset >= PAGE_BYTES {
            return Err(EngineError::Invalid("offset past the first page"));
        }
        let count = (offset + data.len()).div_ceil(PAGE_BYTES);
        let frames = frames.get(..count).ok_or(EngineError::Invalid("fewer frames than the data covers"))?;
        let mut seen = WrittenViews::new();
        if count == 0 {
            return Ok(seen);
        }
        // Per page written: its pool index, its address in the private view and its pin.
        let mut targets: SmallBuf<(usize, usize, usize), INLINE_WRITE_PAGES> = SmallBuf::with_capacity(count);
        let pinned;
        {
            let st = self.lock();
            for &frame in frames {
                if !st.pages.is_allocated(frame) {
                    return Err(PageError::NotAllocated(frame).into());
                }
                let index = index_of(frame).ok_or(PageError::BadFrame(frame))?;
                let section = Self::section_of(&st, index).ok_or(PageError::BadFrame(frame))?;
                let slot = index - section.start;
                targets.push((index, section.rw_view + slot * PAGE_BYTES, &raw const section.pins[slot] as usize));
            }
            // A page that was never touched is committed first, under the lock like the fault handler does it.
            for &(index, _, _) in targets.as_slice() {
                Self::commit_page(&st, index)?;
            }
            pinned = PinGuard::pin(targets.as_slice());
            let mut at = 0;
            for (k, &(index, _, _)) in targets.as_slice().iter().enumerate() {
                let start = if k == 0 { offset } else { 0 };
                let n = (PAGE_BYTES - start).min(data.len() - at);
                for (base, first) in st.regions.views_of_page(index) {
                    seen.push((base + (index - first) * PAGE_BYTES + start, n));
                }
                at += n;
            }
        }
        let mut at = 0;
        for (k, &(_, page, _)) in targets.as_slice().iter().enumerate() {
            let start = if k == 0 { offset } else { 0 };
            let n = (PAGE_BYTES - start).min(data.len() - at);
            // SAFETY: `page` is a 64 KiB page of a private read write view that lives as long as the engine
            // (`self` is borrowed), is committed and stays so, and is pinned: nothing gives it back to the
            // pool before `pinned` drops. `start + n` stays inside it, and `data` holds `at + n` bytes.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(at), (page + start) as *mut u8, n) };
            at += n;
        }
        // The pins go before the dirty lock is taken: a thread that waits for them holds the dirty lock.
        drop(pinned);
        {
            let mut dirty_guard = self.dirty.lock().unwrap_or_else(PoisonError::into_inner);
            self.track_changed(&mut dirty_guard, frames);
        }
        let runs = seen.as_mut_slice();
        runs.sort_unstable();
        let mut joined = 0;
        for k in 0..runs.len() {
            let (address, n) = runs[k];
            if joined > 0 && runs[joined - 1].0 + runs[joined - 1].1 == address {
                runs[joined - 1].1 += n;
            } else {
                runs[joined] = (address, n);
                joined += 1;
            }
        }
        seen.truncate(joined);
        Ok(seen)
    }

    /// The section that holds pool page `index`.
    fn section_of(st: &State, index: usize) -> Option<&Section> {
        st.sections.iter().find(|s| index >= s.start && index < s.start + s.pages)
    }

    /// Waits until no write is copying into any of `frames`. Called with the engine lock held, which is
    /// what pins a page, so no new pin appears: the wait ends when the copies in flight end. They take no lock.
    fn wait_unpinned(st: &State, frames: &[u32]) {
        for &frame in frames {
            let Some(index) = index_of(frame) else { continue };
            let Some(section) = Self::section_of(st, index) else { continue };
            let pin = &section.pins[index - section.start];
            while pin.load(Ordering::Acquire) != 0 {
                std::thread::yield_now();
            }
        }
    }

    /// Frees pages. All or nothing. A page that a view still maps stays readable through it until
    /// the last view ends.
    pub fn free_pages(&self, frames: &[u32]) -> Result<(), EngineError> {
        let mut st = self.lock();
        Self::wait_unpinned(&st, frames);
        st.pages.free(frames)?;
        Self::reset_unused_frames(&st, frames);
        Ok(())
    }

    /// Maps page `i` at `base + i * 64 KiB` for each entry. An entry of 0 unmaps that page, and a
    /// page that is mapped already gets the new mapping. The range must lie in one reservation.
    /// Consecutive frames share one view.
    pub fn map_frames(&self, base: usize, frames: &[u32]) -> Result<(), EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        let entries: Vec<(usize, u32)> = frames.iter().enumerate().map(|(i, &f)| (base + i * PAGE_BYTES, f)).collect();
        self.map_entries(&mut st, dirty, &entries)
    }

    /// Unmaps `count` pages from `base`, which return to the reserved state.
    pub fn unmap_pages(&self, base: usize, count: usize) -> Result<(), EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        let size = count.checked_mul(PAGE_BYTES).ok_or(EngineError::Invalid("plage trop grande"))?;
        self.clear_range(&mut st, dirty, base, size)
    }

    /// Shows at `to` the pool pages that the `count` pages from `from` show, page by page: one more
    /// view of the same frames, made by the same path as every other view. A console title has its
    /// GPU do this (the page mapping copies of its command queue): it writes pool pages at one
    /// address and places a texture at another. Both addresses are rounded down to 64 KiB. Nothing
    /// is mapped when a source page shows no pool page or `to` is not in a reservation.
    pub fn show_pages_at(&self, to: usize, from: usize, count: usize) -> Result<(), EngineError> {
        let (to, from) = (to & !(PAGE_BYTES - 1), from & !(PAGE_BYTES - 1));
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        let mut entries: Vec<(usize, u32)> = Vec::with_capacity(count);
        for k in 0..count {
            let page = k.checked_mul(PAGE_BYTES).and_then(|d| from.checked_add(d)).ok_or(EngineError::Invalid("plage trop grande"))?;
            let frame = match st.regions.containing(page).map(|(base, region)| (base, region.kind)) {
                Some((base, Kind::View { first_frame })) => first_frame + ((page - base) / PAGE_BYTES) as u32 * FRAMES_PER_PAGE,
                _ => return Err(EngineError::Invalid("la page source ne montre aucune page du pool")),
            };
            entries.push((to + k * PAGE_BYTES, frame));
        }
        self.map_entries(&mut st, dirty, &entries)
    }

    /// Maps each frame at its own address, one page each, with 0 unmapping, as `map_frames`.
    pub fn map_scatter(&self, entries: &[(usize, u32)]) -> Result<(), EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        self.map_entries(&mut st, dirty, entries)
    }

    /// Releases the reservation that starts at `base`: its views are unmapped, its placeholders
    /// released and its address range given back.
    pub fn release(&self, base: usize) -> Result<(), EngineError> {
        let mut st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        for at in st.regions.reservation_at(base)? {
            match st.regions.at(at).ok_or(RegionError::Unknown)?.kind {
                Kind::Placeholder => {
                    let size = st.regions.at(at).ok_or(RegionError::Unknown)?.size;
                    self.api.release_pieces(at, size)?;
                }
                Kind::Allocation => self.api.release_placeholder(at)?,
                Kind::View { .. } => {
                    self.api.unmap(at, false)?;
                    self.forget_view(&mut st, dirty, at)?;                    // The pages of a commit go back to the pool with the reservation.
                    if let Some((first, count)) = st.commits.remove(&at) {
                        let frames: Vec<u32> = (0..count as u32).map(|k| first + k * FRAMES_PER_PAGE).collect();
                        Self::wait_unpinned(&st, &frames);
                        if st.pages.free(&frames).is_ok() {
                            Self::reset_unused_frames(&st, &frames);
                        }
                    }
                }
            }
            st.regions.remove(at)?;
        }
        st.on_demand.remove(&base);
        self.note(&mut st, journal::RELEASE, base, 0, 0);
        Ok(())
    }

    /// Whether `address` lies in a reservation of the engine.
    pub fn owns(&self, address: usize) -> bool {
        self.lock().regions.containing(address).is_some()
    }

    /// Backs the 64 KiB page holding `address` when it still lies in an unmapped placeholder of a
    /// reservation of this engine (see `install_fault_handler`). Returns whether the faulting access
    /// can run again. `false` means nothing is reserved there, or the page is a view or an allocation
    /// that still refuses the access, so the fault is real.
    pub fn fault_in(&self, address: usize, write: bool) -> bool {
        static LOGGED: AtomicI64 = AtomicI64::new(0);
        const LOG_LIMIT: i64 = 64;
        let page = address & !(PAGE_BYTES - 1);
        let protect = {
            let st = self.lock();
            match st.regions.containing(page) {
                Some((_, region)) if region.kind == Kind::Placeholder => view_protection(region.protect),
                // Another thread placed pages here while the handler waited for the lock, so the
                // page is backed already: the access can run again unless the protection refuses it.
                Some(_) => {
                    let ok = accessible(page, write);
                    if ok && LOGGED.fetch_add(1, Ordering::Relaxed) < LOG_LIMIT {
                        decant_log::trace(&format!("[xmem] fault_in : 0x{page:x}, mappee entre-temps, l'acces reprend
"));
                    }
                    return ok;
                }
                None => return false,
            }
        };
        let backed = self.place(page, PAGE_BYTES, true, protect).is_ok();
        if backed && LOGGED.fetch_add(1, Ordering::Relaxed) < LOG_LIMIT {
            decant_log::trace(&format!("[xmem] fault_in : 0x{page:x}, commise par le gestionnaire vectoriel (course avec le mappage)
"));
        }
        backed
    }

    // The helpers below run with the lock held.

    /// Makes sure a request of `count` pages can be served: when no section has a free run of that
    /// size, the pool grows by one section big enough for the whole request. With `scatter`, a
    /// request that does not fit the room for a section may still grow the pool by what is missing,
    /// and is then served by scattered pages. The pool never grows past `limit_pages`.
    fn make_room(&self, st: &mut State, dirty: &mut Dirty, count: usize, scatter: bool) -> Result<(), EngineError> {
        let pool = st.pages.capacity();
        let free = pool - st.pages.used_count();
        let room = self.limit_pages.saturating_sub(pool);
        // A reserved section charges nothing until its pages are touched, so growth is bounded by the pool
        // limit alone and is the same on every machine (a request of the title never fails for the commit
        // the system has left at that moment). A committed one (under Wine) is charged when made: then
        // the commit left bounds it, half of it with the GPU mirror on, whose pinned copy is charged too.
        let reserved = !on_wine();
        let available = if reserved {
            u64::MAX
        } else if dirty.track.is_some() {
            available_commit() / 2
        } else {
            available_commit()
        };
        let whole = growth(pool, 0, count, available);
        let plan = if st.pages.has_run(count) {
            Growth::None
        } else if matches!(whole, Growth::Add(_)) && count <= room {
            whole
        } else if scatter {
            growth(pool, free, count, available)
        } else {
            Growth::Bound { room, missing: count }
        };
        match plan {
            Growth::None => Ok(()),
            Growth::Add(extra) if count.saturating_sub(if scatter { free } else { 0 }) <= room => {
                let extra = extra.min(room);
                self.grow(st, dirty, extra)?;
                let bound = if reserved {
                    format!("sections reservees, bornee par la limite du pool seule ({} pages)", self.limit_pages)
                } else {
                    format!("sections commises, bornee par la charge memoire libre ({} Mio pris en compte)", available >> 20)
                };
                decant_log::trace(&format!("[xmem] pages physiques : reserve agrandie de {extra} pages ({} Mio) a {} pages ({bound})\r\n", (extra * PAGE_BYTES) >> 20, pool + extra));
                Ok(())
            }
            Growth::Add(_) | Growth::Bound { .. } => {
                decant_log::trace(&format!(
                    "[xmem] pages physiques : limite atteinte, {} pages manquent, reserve de {pool} pages non agrandie (charge memoire libre ou limite du pool)\r\n",
                    count.saturating_sub(free)
                ));
                Ok(())
            }
        }
    }

    /// Makes `[base, base + size)` a placeholder of its own, splitting what surrounds it. Returns
    /// the number of cuts made (0 to 2).
    fn carve(&self, st: &mut State, base: usize, size: usize) -> Result<usize, EngineError> {
        let (start, whole) = st.regions.placeholder_holding(base, size)?;
        let steps = split_steps(start, whole, base, size);
        for &(at, first) in &steps {
            self.api.split(at, first)?;
            st.regions.split(at, first)?;
        }
        Ok(steps.len())
    }

    /// Turns the allocation covering `at` back into a placeholder, so that a later mapping call for
    /// the same address wins over a page the fault handler backed (`carve` only splits placeholders).
    /// Does nothing when `at` is a placeholder already or lies in no reservation.
    fn reclaim_allocation(&self, st: &mut State, at: usize) -> Result<(), EngineError> {
        match st.regions.containing(at) {
            Some((base, region)) if region.kind == Kind::Allocation => {
                self.api.restore(base, region.size)?;
                st.regions.clear_allocation(base)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Joins the placeholders around `[base, base + size)` that came from one reservation.
    fn coalesce(&self, st: &mut State, base: usize, size: usize) -> Result<(), EngineError> {
        if let Some((start, length)) = st.regions.coalescible(base, size) {
            self.api.coalesce(start, length)?;
            st.regions.merge(start, length);
        }
        Ok(())
    }

    /// Maps the entries (address, frame), validated first so that a bad entry changes nothing.
    fn map_entries(&self, st: &mut State, dirty: &mut Dirty, entries: &[(usize, u32)]) -> Result<(), EngineError> {
        let mut spans: Vec<(usize, u32, u32)> = Vec::with_capacity(entries.len());
        for &(address, frame) in entries {
            let base = address & !(PAGE_BYTES - 1);
            if frame != NO_FRAME && !st.pages.is_allocated(frame) {
                return Err(PageError::NotAllocated(frame).into());
            }
            let covering = st.regions.covering(base, PAGE_BYTES)?;
            spans.push((base, frame, covering[0].1.protect));
        }
        for &(base, _, _) in &spans {
            self.clear_range(st, dirty, base, PAGE_BYTES)?;
        }
        // Consecutive entries that go to consecutive addresses with consecutive frames share a view.
        let mut i = 0;
        while i < spans.len() {
            let (base, frame, protect) = spans[i];
            if frame == NO_FRAME {
                i += 1;
                continue;
            }
            let mut n = 1;
            while i + n < spans.len() && spans[i + n].0 == base + n * PAGE_BYTES && spans[i + n].1 == frame + n as u32 * FRAMES_PER_PAGE && spans[i + n].2 == protect {
                n += 1;
            }
            self.map_one(st, dirty, base, frame, n, view_protection(protect))?;
            i += n;
        }
        Ok(())
    }

    /// Gives `[base, base + size)` back as placeholders: the views that touch it are unmapped (the
    /// pages of a view that stick out are mapped again), and the placeholders are joined.
    fn clear_range(&self, st: &mut State, dirty: &mut Dirty, base: usize, size: usize) -> Result<(), EngineError> {
        let start = base & !(PAGE_BYTES - 1);
        let end = base.checked_add(size).and_then(|e| e.checked_next_multiple_of(PAGE_BYTES)).ok_or(RegionError::BadRange)?;
        let found = st.regions.covering(start, end - start)?;
        for (at, region) in found {
            let Kind::View { first_frame } = region.kind else {
                continue;
            };
            let count = region.size / PAGE_BYTES;
            self.api.unmap(at, true)?;
            st.regions.clear_view(at)?;
            st.pages.unmap(first_frame, count)?;
            self.track_unmap(st, dirty, at, first_frame, count);
            let protect = view_protection(region.protect);
            for k in 0..count {
                let page = at + k * PAGE_BYTES;
                if page < start || page >= end {
                    self.map_one(st, dirty, page, first_frame + k as u32 * FRAMES_PER_PAGE, 1, protect)?;
                }
            }
        }
        self.coalesce(st, start, end - start)
    }

    /// Maps `count` consecutive pages from `first` over `[at, at + count * 64 KiB)`, which lies in
    /// placeholders of one reservation.
    fn map_one(&self, st: &mut State, dirty: &mut Dirty, at: usize, first: u32, count: usize, protect: u32) -> Result<(), EngineError> {
        let start = index_of(first).ok_or(PageError::BadFrame(first))?;
        self.section(st, dirty)?;
        self.coalesce(st, at, count * PAGE_BYTES)?;
        // One view per backing section: a run that crosses the end of a section is mapped in parts.
        let mut done = 0;
        while done < count {
            let (handle, offset, left) = Self::locate(st, start + done)?;
            let n = left.min(count - done);
            let seg_at = at + done * PAGE_BYTES;
            let seg_first = first + done as u32 * FRAMES_PER_PAGE;
            self.reclaim_allocation(st, seg_at)?;
            self.carve(st, seg_at, n * PAGE_BYTES)?;
            self.api.map(handle, seg_at, offset, n * PAGE_BYTES, protect)?;
            st.regions.set_view(seg_at, seg_first)?;
            st.pages.map(seg_first, n)?;
            self.track_map(st, dirty, seg_at, seg_first, n, protect);
            done += n;
        }
        Ok(())
    }

    /// Registry side of ending a view: the pages lose a view and the range stays a placeholder.
    fn forget_view(&self, st: &mut State, dirty: &mut Dirty, base: usize) -> Result<(), EngineError> {
        let region = st.regions.clear_view(base)?;
        if let Kind::View { first_frame } = region.kind {
            st.pages.unmap(first_frame, region.size / PAGE_BYTES)?;
            self.track_unmap(st, dirty, base, first_frame, region.size / PAGE_BYTES);
        }
        Ok(())
    }

    /// Notes that the view of `count` pages from `first_frame` at `at` is gone: the dirty tracking
    /// forgets its addresses and the GPU mirror is told.
    fn track_unmap(&self, st: &mut State, dirty: &mut Dirty, at: usize, first_frame: u32, count: usize) {
        if let (Some(track), Some(first)) = (dirty.track.as_mut(), index_of(first_frame)) {
            for k in 0..count {
                track.on_unmap((first + k) as u32, at + k * PAGE_BYTES);
            }
        }
        self.note(st, journal::UNMAP, at, count, 0);
    }

    /// Notes that the view of `count` pages from `first_frame` at `at` exists, with the view protection `protect`.
    /// A new map counts as a write: the pages are dirty, and the other views of them read write again.
    /// Pages of a commit of `commit_region` that the GPU was not given yet stay out of both.
    fn track_map(&self, st: &mut State, dirty: &mut Dirty, at: usize, first_frame: u32, count: usize, protect: u32) {
        let Some(first) = index_of(first_frame) else { return };
        let shown: Vec<bool> = (0..count).map(|k| Self::shown(st, at + k * PAGE_BYTES)).collect();
        let mut open = Vec::new();
        if let Some(track) = dirty.track.as_mut() {
            for k in (0..count).filter(|&k| shown[k]) {
                open.extend(track.on_map((first + k) as u32, at + k * PAGE_BYTES, protect == PAGE_READWRITE));
            }
            dirty.sync.bump_addrs(&open);
            if track.pending_len() > 0 {
                dirty.sync.set_pending(true);
            }
        }
        dirty.sync.protect_rw_locked(open);
        let mut k = 0;
        while k < count {
            let n = shown[k..].iter().take_while(|&&s| s == shown[k]).count();
            if shown[k] {
                self.note(st, journal::MAP, at + k * PAGE_BYTES, n, (first + k) as u32);
            }
            k += n;
        }
    }

    /// Notes that the bytes of `frames` changed outside the title's views: the allocator handed them
    /// out, or the engine wrote them (`write_frames`).
    fn track_changed(&self, dirty: &mut Dirty, frames: &[u32]) {
        let mut open = Vec::new();
        if let Some(track) = dirty.track.as_mut() {
            for &frame in frames {
                if let Some(page) = index_of(frame) {
                    open.extend(track.on_changed(page as u32));
                }
            }
            dirty.sync.bump_addrs(&open);
            if track.pending_len() > 0 {
                dirty.sync.set_pending(true);
            }
        }
        dirty.sync.protect_rw_locked(open);
    }

    /// Sets the protection of `[start, start + bytes)` to `protect`. One call cannot cross from one
    /// view to the next (error 487), so a run over several views is set view by view.
    fn protect_range(start: usize, bytes: usize, protect: u32) {
        let mut old = 0u32;
        // SAFETY: the range is pages of views of the section that the registry holds.
        if unsafe { VirtualProtect(start as *const c_void, bytes, protect, &mut old) } != 0 {
            return;
        }
        let end = start + bytes;
        let mut at = start;
        while at < end {
            // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: as above.
            if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) } == 0 {
                break;
            }
            let next = (info.BaseAddress as usize + info.RegionSize).min(end);
            if info.State == MEM_COMMIT {
                // SAFETY: as above, a part of the same run that lies inside one view.
                if unsafe { VirtualProtect(at as *const c_void, next - at, protect, &mut old) } == 0 {
                    decant_log::trace(&format!("[xmem] VirtualProtect 0x{at:x}, 0x{:x} octets, {protect:#x} a echoue (erreur {})\r\n", next - at, Api::last_error("VirtualProtect").win32_code()));
                }
            }
            at = next;
        }
    }

    /// Sets the protection of the title view pages at `addrs`, adjacent pages in one call. One call
    /// cannot cross from one view to the next (error 487), so a run over several views is set view by
    /// view. A failure is logged: the page then keeps its protection, and the worst case is a lost or
    /// extra fault.
    #[cfg_attr(not(test), allow(dead_code))]
    fn protect_runs(addrs: Vec<usize>, protect: u32) {
        for (start, bytes) in track::runs(addrs) {
            Self::protect_range(start, bytes, protect);
        }
    }

    /// The address of pool page `index` in the engine's private read write view.
    fn private_page(st: &State, index: usize) -> Option<usize> {
        let s = st.sections.iter().find(|s| index >= s.start && index < s.start + s.pages)?;
        Some(s.rw_view + (index - s.start) * PAGE_BYTES)
    }

    /// Switches the dirty tracking of the GPU mirror on or off. On, every page that a view maps starts
    /// dirty, so the first sync copies them all.
    pub fn set_tracking(&self, on: bool) {
        let st = self.lock();
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let dirty = &mut *dirty_guard;
        if on == dirty.track.is_some() {
            return;
        }
        if !on {
            // Every protected page becomes writable again before the tracker goes.
            let mut open = Vec::new();
            for (at, region) in st.regions.views() {
                if view_protection(region.protect) == PAGE_READWRITE {
                    open.extend((0..region.size / PAGE_BYTES).map(|k| at + k * PAGE_BYTES));
                }
            }
            protect_sync.bump_addrs(&open);
            let chunks = protect_sync.snapshot_rw_runs(open);
            protect_sync.set_pending(false);
            dirty.track = None;
            self.tracking.store(false, Ordering::Release);
            drop(dirty_guard);
            protect_sync.protect_rw_runs(chunks, &dirty_state);
            return;
        }
        let mut track = Track::new(st.pages.capacity());
        for (at, region) in st.regions.views() {
            if let Kind::View { first_frame } = region.kind
                && let Some(first) = index_of(first_frame)
            {
                for k in (0..region.size / PAGE_BYTES).filter(|&k| Self::shown(&st, at + k * PAGE_BYTES)) {
                    track.on_map((first + k) as u32, at + k * PAGE_BYTES, view_protection(region.protect) == PAGE_READWRITE);
                }
            }
        }
        protect_sync.set_pending(track.pending_len() > 0);
        dirty.track = Some(track);
        self.tracking.store(true, Ordering::Release);
    }

    /// Whether the dirty tracking is on.
    pub fn tracking(&self) -> bool {
        self.tracking.load(Ordering::Acquire)
    }

    /// A write faulted at `address`. True when it was a write to a page the dirty tracking protected (or
    /// one another thread has just opened): the page is dirty and writable again and the write runs again.
    pub fn on_write_fault(&self, address: usize) -> bool {
        static FAULTS: AtomicI64 = AtomicI64::new(0);
        static HOT: AtomicU64 = AtomicU64::new(0);
        if !self.tracking.load(Ordering::Acquire) {
            return false;
        }
        let page = address & !(PAGE_BYTES - 1);
        let (index, was_hot, is_hot, chunks) = {
            let mut dirty = self.dirty.lock().unwrap_or_else(PoisonError::into_inner);
            let Some(track) = dirty.track.as_mut() else { return false };
            let Some(mut index) = track.writable_page(page) else { return false };
            if !track.is_committed(index) {
                drop(dirty);
                // An untouched section page needs the commit path before protection can change.
                if !accessible(page, false) {
                    return false;
                }
                dirty = self.dirty.lock().unwrap_or_else(PoisonError::into_inner);
                let Some(track) = dirty.track.as_mut() else { return false };
                let Some(rechecked) = track.writable_page(page) else { return false };
                index = rechecked;
                track.mark_committed(index);
            }
            let Some(track) = dirty.track.as_mut() else { return false };
            let now = Instant::now();
            let hot_threshold = now + track::guard_rest();
            let was_hot = track.is_held(index, hot_threshold);
            let open = track.on_fault(index, page, now);
            let is_hot = track.is_held(index, hot_threshold);
            let chunks = open.map(|addrs| {
                self.protect_sync.bump_addrs(&addrs);
                self.protect_sync.snapshot_rw_runs(addrs)
            });
            if track.pending_len() > 0 {
                self.protect_sync.set_pending(true);
            }
            (index, was_hot, is_hot, chunks)
        };
        let handled = match chunks {
            Some(chunks) => {
                self.protect_sync.protect_rw_runs(chunks, &self.dirty);
                true
            }
            None => self.protect_sync.ensure_rw(page),
        };
        if !was_hot && is_hot {
            let hot = HOT.fetch_add(1, Ordering::Relaxed) + 1;
            if hot <= 8 || hot.is_multiple_of(1000) {
                decant_log::trace(&format!("[xmem] suivi des pages : page {index} (0x{page:x}) ecrite a chaque copie, ouverte et copiee a chaque synchronisation pour {} ms ({hot} fois)\r\n", track::hot_hold().as_millis()));
            }
        }
        let n = FAULTS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 10000 == 0 {
            decant_log::trace(&format!("[xmem] suivi des pages : {n} ecritures interceptees\r\n"));
        }
        handled
    }

    /// Commits pool page `index` of its section, through the engine's read write view: every view of the
    /// page sees it committed from then on. `Ok(true)` when it was committed now, `Ok(false)` when it was
    /// already. A failure (the commit limit of the system) is logged.
    fn commit_page(st: &State, index: usize) -> Result<bool, EngineError> {
        static FAILED: AtomicI64 = AtomicI64::new(0);
        let section = Self::section_of(st, index).ok_or(PageError::BadFrame(0))?;
        let slot = index - section.start;
        let (word, bit) = (slot / 64, 1u64 << (slot % 64));
        if section.committed[word].load(Ordering::Relaxed) & bit != 0 {
            return Ok(false);
        }
        let page = section.rw_view + slot * PAGE_BYTES;
        if page_committed(page) {
            section.committed[word].fetch_or(bit, Ordering::Relaxed);
            return Ok(false);
        }
        // SAFETY: `page` is a page of the engine's read write view of a reserved section, which only the
        // engine maps there. Committing it changes no byte (a new page reads as zeros).
        if unsafe { VirtualAlloc(page as *const c_void, PAGE_BYTES, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
            let error = Api::last_error("VirtualAlloc (commit d'une page du pool)");
            if FAILED.fetch_add(1, Ordering::Relaxed) < 16 {
                decant_log::trace(&format!("[xmem] page {index} du pool : commit refuse ({error}), charge memoire du systeme epuisee ?\r\n"));
            }
            return Err(error);
        }
        section.committed[word].fetch_or(bit, Ordering::Relaxed);
        Ok(true)
    }

    /// An access faulted at `address`. True when it lies in a view of a pool page that was never committed:
    /// the page is committed now (it reads as zeros), it is dirty for the GPU mirror, and the access runs
    /// again. False for anything else, which the other halves of the fault handler look at.
    fn commit_on_access_in(state: &Mutex<State>, address: usize) -> bool {
        static COMMITS: AtomicU64 = AtomicU64::new(0);
        let page = address & !(PAGE_BYTES - 1);
        let st = lock_state(state);
        let Some((base, region)) = st.regions.containing(page) else { return false };
        let Kind::View { first_frame } = region.kind else { return false };
        let Some(first) = index_of(first_frame) else { return false };
        let index = first + (page - base) / PAGE_BYTES;
        if !matches!(Self::commit_page(&st, index), Ok(true)) {
            return false;
        }
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        drop(st);
        let chunks = {
            let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
            let mut chunks = None;
            if let Some(track) = dirty_guard.track.as_mut() {
                track.mark_committed(index as u32);
                if let Some(addrs) = track.on_fault(index as u32, page, Instant::now()) {
                    protect_sync.bump_addrs(&addrs);
                    chunks = Some(protect_sync.snapshot_rw_runs(addrs));
                }
                if track.pending_len() > 0 {
                    protect_sync.set_pending(true);
                }
            }
            chunks
        };
        if let Some(chunks) = chunks {
            protect_sync.protect_rw_runs(chunks, &dirty_state);
        }
        let n = COMMITS.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_multiple_of(16384) {
            decant_log::trace(&format!("[xmem] pages du pool commises au premier acces : {n}\r\n"));
        }
        true
    }

    /// The pool pages whose mirror is stale, at most `cap`, each now protected against writes until the
    /// next one. The caller copies them from the read only view of their section after this returns.
    pub fn take_dirty(&self, cap: usize) -> Vec<u32> {
        if !self.protect_sync.has_pending() {
            return Vec::new();
        }
        let asked = Instant::now();
        let mut dirty = self.dirty.lock().unwrap_or_else(PoisonError::into_inner);
        let locked = Instant::now();
        let (pending, candidates) = {
            let Dirty { track, sections, sync } = &mut *dirty;
            let Some(track) = track.as_mut() else {
                sync.set_pending(false);
                return Vec::new();
            };
            let pending = track.pending_len();
            if pending == 0 {
                sync.set_pending(false);
                return Vec::new();
            }
            let candidates: Vec<(u32, usize)> = track
                .uncommitted_pending(cap)
                .into_iter()
                .filter_map(|page| {
                    let i = page as usize;
                    sections
                        .iter()
                        .find(|&&(start, pages, _)| i >= start && i < start + pages)
                        .map(|&(start, _, view)| (page, view + (i - start) * PAGE_BYTES))
                })
                .collect();
            (pending, candidates)
        };
        let mut queries = 0usize;
        let mut uncommitted_set = std::collections::HashSet::new();
        if !candidates.is_empty() {
            drop(dirty);
            let mut runs_seen = CommittedRuns::default();
            let mut committed_pages = Vec::new();
            for (page, view_addr) in candidates {
                if runs_seen.is_committed(view_addr, query_region) {
                    committed_pages.push(page);
                } else {
                    uncommitted_set.insert(page);
                }
            }
            queries = runs_seen.queries;
            dirty = self.dirty.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(track) = dirty.track.as_mut() {
                for page in committed_pages {
                    track.mark_committed(page);
                }
            }
        }
        let (pages, chunks, runs) = {
            let Dirty { track, sections, sync } = &mut *dirty;
            let Some(track) = track.as_mut() else {
                sync.set_pending(false);
                return Vec::new();
            };
            let runs_seen = std::cell::RefCell::new(CommittedRuns::default());
            let committed = |page: u32| {
                if uncommitted_set.contains(&page) {
                    return false;
                }
                let i = page as usize;
                sections
                    .iter()
                    .find(|&&(start, pages, _)| i >= start && i < start + pages)
                    .is_some_and(|&(start, _, view)| runs_seen.borrow_mut().is_committed(view + (i - start) * PAGE_BYTES, query_region))
            };
            let interval = std::time::Duration::from_millis(self.reprotect_interval_ms.load(Ordering::Relaxed));
            let (pages, protect) = track.take_batched(Instant::now(), cap, interval, committed);
            let protect = track.bridge_clean_gaps(protect, 12);
            queries += runs_seen.borrow().queries;
            sync.set_pending(track.pending_len() > 0);
            sync.bump_addrs(&protect);
            let runs = protect.len();
            let snapshots = sync.snapshot_runs(protect);
            (pages, snapshots, runs)
        };
        let taken = Instant::now();
        drop(dirty);
        let (vp_runs, retries) = self.protect_sync.protect_ro_runs(chunks, &self.dirty);
        let done = Instant::now();
        take_stats(pages.len(), runs, vp_runs, retries, pending, queries, [locked - asked, taken - locked, done - taken]);
        pages
    }

    /// A kernel write (a file read, a DirectStorage request) may land in `[address, address + len)`: the
    /// pages of it that views map are committed (a kernel write into a page that is not fails instead of
    /// faulting), and with the dirty tracking on they become dirty and writable, and stay so for
    /// `track::io_hold_for(len)`.
    pub fn prepare_write(&self, address: usize, len: usize) {
        self.prepare_write_until(address, len, Instant::now() + track::io_hold_for(len));
    }

    fn collect_rw_view_indices_into(st: &State, address: usize, len: usize, indices: &mut Vec<u32>) {
        if len == 0 {
            return;
        }
        let Some(end) = address.checked_add(len) else { return };
        let mut page = address & !(PAGE_BYTES - 1);
        while page < end {
            let Some((base, region)) = st.regions.containing(page) else {
                page += PAGE_BYTES;
                continue;
            };
            let region_end = base + region.size;
            if let Kind::View { first_frame } = region.kind
                && view_protection(region.protect) == PAGE_READWRITE
                && let Some(first) = index_of(first_frame)
            {
                while page < end && page < region_end {
                    let index = first + (page - base) / PAGE_BYTES;
                    indices.push(index as u32);
                    page += PAGE_BYTES;
                }
            } else {
                page = region_end.max(page + PAGE_BYTES);
            }
        }
    }

    fn collect_rw_view_indices(st: &State, address: usize, len: usize) -> Vec<u32> {
        let mut indices = Vec::new();
        Self::collect_rw_view_indices_into(st, address, len, &mut indices);
        indices
    }

    /// Hints Windows to discard physical backing for freed, unmapped pool pages via `MEM_RESET`
    /// while keeping section reservations intact. Pages still mapped by any view are skipped.
    fn reset_unused_frames(st: &State, frames: &[u32]) {
        const MEM_RESET: u32 = 0x8_0000;
        if on_wine() || frames.is_empty() {
            return;
        }
        let mut unused: Vec<usize> = frames
            .iter()
            .copied()
            .filter(|&f| !st.pages.is_in_use(f))
            .filter_map(index_of)
            .collect();
        if unused.is_empty() {
            return;
        }
        unused.sort_unstable();
        unused.dedup();
        let mut i = 0;
        while i < unused.len() {
            let start_idx = unused[i];
            let Some(sec) = st.sections.iter().find(|s| start_idx >= s.start && start_idx < s.start + s.pages) else {
                i += 1;
                continue;
            };
            let sec_end = sec.start + sec.pages;
            let mut run_len = 1usize;
            while i + run_len < unused.len() {
                let next_idx = unused[i + run_len];
                if next_idx == start_idx + run_len && next_idx < sec_end {
                    run_len += 1;
                } else {
                    break;
                }
            }
            let base_addr = sec.rw_view + (start_idx - sec.start) * PAGE_BYTES;
            let run_bytes = run_len * PAGE_BYTES;
            if page_committed(base_addr) {
                // SAFETY: `base_addr..base_addr + run_bytes` is inside one section's private view and unmapped.
                unsafe { VirtualAlloc(base_addr as *const c_void, run_bytes, MEM_RESET, PAGE_READWRITE) };
            }
            i += run_len;
        }
    }

    /// Commits any uncommitted pool pages in `indices` in maximal contiguous section runs.
    /// Returns the indices that were not yet marked committed in `track`.
    #[cfg(test)]
    fn commit_uncommitted_runs(st: &State, indices: &[u32]) -> Vec<u32> {
        let uncommitted = {
            let dirty_state = Arc::clone(&st.dirty);
            let dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
            match dirty_guard.track.as_ref() {
                Some(track) => indices.iter().copied().filter(|&i| !track.is_committed(i)).collect::<Vec<u32>>(),
                None => indices.to_vec(),
            }
        };
        if uncommitted.is_empty() {
            return Vec::new();
        }
        Self::commit_index_runs(st, &uncommitted);
        uncommitted
    }

    /// Commits contiguous pool page indices within the same section with one `VirtualAlloc` per run,
    /// falling back to per-page `commit_page` on failure.
    fn commit_index_runs(st: &State, indices: &[u32]) {
        if indices.is_empty() {
            return;
        }
        let mut sorted = indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut i = 0;
        while i < sorted.len() {
            let start_idx = sorted[i] as usize;
            let Some(sec) = st.sections.iter().find(|s| start_idx >= s.start && start_idx < s.start + s.pages) else {
                i += 1;
                continue;
            };
            let sec_end = sec.start + sec.pages;
            let mut run_len = 1usize;
            while i + run_len < sorted.len() {
                let next_idx = sorted[i + run_len] as usize;
                if next_idx == start_idx + run_len && next_idx < sec_end {
                    run_len += 1;
                } else {
                    break;
                }
            }
            let base_addr = sec.rw_view + (start_idx - sec.start) * PAGE_BYTES;
            let run_bytes = run_len * PAGE_BYTES;
            // SAFETY: `base_addr..base_addr + run_bytes` lies inside one section's private read-write view.
            if unsafe { VirtualAlloc(base_addr as *const c_void, run_bytes, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
                for offset in 0..run_len {
                    let _ = Self::commit_page(st, start_idx + offset);
                }
            } else {
                // The pages are known committed from now on: `write_frames` need not ask the system.
                for slot in start_idx - sec.start..start_idx - sec.start + run_len {
                    sec.committed[slot / 64].fetch_or(1u64 << (slot % 64), Ordering::Relaxed);
                }
            }
            i += run_len;
        }
    }

    /// Commits and opens the writable view pages across all `(address, len)` runs for a kernel
    /// write, acquiring the state and dirty locks once across the whole batch.
    #[cfg(test)]
    pub fn prepare_write_runs(&self, runs: &[(usize, usize)]) {
        if runs.is_empty() {
            return;
        }
        let st = self.lock();
        let mut indices = Vec::new();
        let mut total_bytes = 0usize;
        for &(address, len) in runs {
            total_bytes = total_bytes.saturating_add(len);
            Self::collect_rw_view_indices_into(&st, address, len, &mut indices);
        }
        if indices.is_empty() {
            return;
        }
        let until = Instant::now() + track::io_hold_for(total_bytes);
        let mut wait_mask = [0u64; 4];
        for &(address, len) in runs {
            ProtectSync::merge_mask(&mut wait_mask, ProtectSync::stripe_mask(address, len));
        }
        Self::commit_and_prepare_until(&st, indices, until, wait_mask);
    }

    fn commit_and_prepare_until(st: &State, indices: Vec<u32>, until: Instant, wait_mask: StripeMask) {
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let uncommitted = match dirty_guard.track.as_ref() {
            Some(track) => indices.iter().copied().filter(|&i| !track.is_committed(i)).collect::<Vec<u32>>(),
            None => indices.clone(),
        };
        if !uncommitted.is_empty() {
            drop(dirty_guard);
            Self::commit_index_runs(st, &uncommitted);
            dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        }
        let mut open = Vec::new();
        if let Some(track) = dirty_guard.track.as_mut() {
            for index in uncommitted {
                track.mark_committed(index);
            }
            for index in indices {
                track.prepare_until_into(index, until, &mut open);
            }
            protect_sync.bump_addrs(&open);
            if track.pending_len() > 0 {
                protect_sync.set_pending(true);
            }
        }
        let chunks = protect_sync.snapshot_rw_runs(open);
        drop(dirty_guard);
        protect_sync.protect_rw_runs(chunks, &dirty_state);
        protect_sync.wait_in_flight_rw(wait_mask);
    }

    /// Commits and pins the writable view pages in `[address, address + len)` open for an active
    /// asynchronous write until [`Engine::finish_io_write`] balances every overlapping pin.
    pub fn begin_io_write(&self, address: usize, len: usize) {
        let _ = self.begin_io_write_checked(address, len);
    }

    /// As [`Engine::begin_io_write`], returning whether any writable view page was touched.
    pub(crate) fn begin_io_write_checked(&self, address: usize, len: usize) -> bool {
        if len == 0 {
            return false;
        }
        self.begin_io_write_runs_checked(&[(address, len)])
    }

    /// Releases one active write pin on each writable view page in `[address, address + len)`.
    /// Once the last pin drops, the next `take_dirty` copies the page once and protects it again.
    pub fn finish_io_write(&self, address: usize, len: usize) {
        if len == 0 {
            return;
        }
        self.finish_io_write_runs(&[(address, len)]);
    }

    /// Commits and pins the writable view pages across all `(address, len)` runs open for an active write.
    #[cfg(test)]
    pub fn begin_io_write_runs(&self, runs: &[(usize, usize)]) {
        let _ = self.begin_io_write_runs_checked(runs);
    }

    /// As [`Engine::begin_io_write_runs`], returning whether any writable view page was touched.
    pub(crate) fn begin_io_write_runs_checked(&self, runs: &[(usize, usize)]) -> bool {
        if runs.is_empty() {
            return false;
        }
        let st = self.lock();
        let mut indices = Vec::new();
        for &(address, len) in runs {
            Self::collect_rw_view_indices_into(&st, address, len, &mut indices);
        }
        if indices.is_empty() {
            return false;
        }
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        let uncommitted = match dirty_guard.track.as_ref() {
            Some(track) => indices.iter().copied().filter(|&i| !track.is_committed(i)).collect::<Vec<u32>>(),
            None => indices.clone(),
        };
        if !uncommitted.is_empty() {
            drop(dirty_guard);
            Self::commit_index_runs(&st, &uncommitted);
            dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        }
        let mut open = Vec::new();
        if let Some(track) = dirty_guard.track.as_mut() {
            for index in uncommitted {
                track.mark_committed(index);
            }
            for index in indices {
                track.begin_write_pin_into(index, &mut open);
            }
            protect_sync.bump_addrs(&open);
            if track.pending_len() > 0 {
                protect_sync.set_pending(true);
            }
        }
        let chunks = protect_sync.snapshot_rw_runs(open);
        drop(dirty_guard);
        drop(st);
        let mut wait_mask = [0u64; 4];
        for &(address, len) in runs {
            ProtectSync::merge_mask(&mut wait_mask, ProtectSync::stripe_mask(address, len));
        }
        protect_sync.protect_rw_runs(chunks, &dirty_state);
        protect_sync.wait_in_flight_rw(wait_mask);
        true
    }

    /// Releases one active write pin on each writable view page across all `(address, len)` runs.
    pub fn finish_io_write_runs(&self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.tracking.load(Ordering::Acquire) {
            return;
        }
        let st = self.lock();
        let mut indices = Vec::new();
        for &(address, len) in runs {
            Self::collect_rw_view_indices_into(&st, address, len, &mut indices);
        }
        if indices.is_empty() {
            return;
        }
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        drop(st);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(track) = dirty_guard.track.as_mut() {
            for index in indices {
                track.finish_write_pin(index);
            }
            if track.pending_len() > 0 {
                protect_sync.set_pending(true);
            }
        }
    }

    /// Transitions active write pins across `runs` into a timed fallback hold for pending overlapped I/O.
    pub fn convert_pin_to_hold_runs(&self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.tracking.load(Ordering::Acquire) {
            return;
        }
        let st = self.lock();
        let mut indices = Vec::new();
        let mut total_bytes = 0usize;
        for &(address, len) in runs {
            total_bytes = total_bytes.saturating_add(len);
            Self::collect_rw_view_indices_into(&st, address, len, &mut indices);
        }
        if indices.is_empty() {
            return;
        }
        let until = Instant::now() + track::io_hold_for(total_bytes);
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        drop(st);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(track) = dirty_guard.track.as_mut() {
            for index in indices {
                track.convert_pin_to_hold(index, until);
            }
            if track.pending_len() > 0 {
                protect_sync.set_pending(true);
            }
        }
    }

    /// Clears I/O fallback holds across `runs` when an overlapped read completes.
    pub fn clear_io_hold_runs(&self, runs: &[(usize, usize)]) {
        if runs.is_empty() || !self.tracking.load(Ordering::Acquire) {
            return;
        }
        let st = self.lock();
        let mut indices = Vec::new();
        for &(address, len) in runs {
            Self::collect_rw_view_indices_into(&st, address, len, &mut indices);
        }
        if indices.is_empty() {
            return;
        }
        let dirty_state = Arc::clone(&st.dirty);
        let protect_sync = Arc::clone(&st.protect_sync);
        drop(st);
        let mut dirty_guard = dirty_state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(track) = dirty_guard.track.as_mut() {
            for index in indices {
                track.clear_io_hold(index);
            }
            if track.pending_len() > 0 {
                protect_sync.set_pending(true);
            }
        }
    }

    fn prepare_write_until(&self, address: usize, len: usize, until: Instant) {
        let st = self.lock();
        let indices = Self::collect_rw_view_indices(&st, address, len);
        if indices.is_empty() {
            return;
        }
        Self::commit_and_prepare_until(&st, indices, until, ProtectSync::stripe_mask(address, len));
    }

    /// The first page of `[address, address + len)` the kernel cannot write now, with what the engine
    /// and the system know of it, for the log of a read that failed with `ERROR_NOACCESS`.
    pub fn first_unwritable(&self, address: usize, len: usize) -> Option<String> {
        let end = address.checked_add(len)?;
        let st = self.lock();
        let mut page = address & !(PAGE_BYTES - 1);
        while page < end {
            if !accessible(page, true) {
                // SAFETY: MEMORY_BASIC_INFORMATION is plain data and VirtualQuery only reads the address space.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: `info` is a valid output buffer of the size passed.
                unsafe { VirtualQuery(page as *const c_void, &mut info, size_of_val(&info)) };
                let region = st.regions.containing(page).map_or("no region of the engine".to_string(), |(base, r)| format!("region at 0x{base:x} of {} bytes, {:?}, protection 0x{:x}", r.size, r.kind, r.protect));
                return Some(format!("page 0x{page:x} ({} bytes into the buffer): state 0x{:x}, protection 0x{:x}, {region}", page.max(address) - address, info.State, info.Protect));
            }
            page += PAGE_BYTES;
        }
        None
    }

    /// Whether every page of `[address, address + len)` lies in a view the title may write, whatever the
    /// temporary protection of the dirty tracking.
    pub fn writable(&self, address: usize, len: usize) -> bool {
        let Some(end) = address.checked_add(len).filter(|_| len != 0) else { return false };
        let st = self.lock();
        let mut at = address;
        while at < end {
            match st.regions.containing(at) {
                Some((base, region)) if matches!(region.kind, Kind::View { .. }) && view_protection(region.protect) == PAGE_READWRITE => at = base + region.size,
                _ => return false,
            }
        }
        true
    }

    /// The protection the title gave the view that holds `address`, when there is one: what a query
    /// reports instead of the temporary `PAGE_READONLY` of the dirty tracking.
    pub fn title_protection(&self, address: usize) -> Option<u32> {
        let st = self.lock();
        let (_, region) = st.regions.containing(address)?;
        matches!(region.kind, Kind::View { .. }).then(|| view_protection(region.protect))
    }

    /// The private read only view of section `index`: its address, its first pool page and its pages.
    pub fn section_view(&self, index: usize) -> Option<(usize, u32, u32)> {
        self.lock().sections.get(index).map(|s| (s.ro_view, s.start as u32, s.pages as u32))
    }

    /// What the engine holds now, for the memory sampler (`memsample`).
    pub fn footprint(&self) -> Footprint {
        let (mut footprint, views) = {
            let st = self.lock();
            let (arena_bytes, given_pages) = st.on_demand.values().fold((0, 0), |(b, g), active| (b + active.len() * PAGE_BYTES, g + active.iter().filter(|&&a| a).count()));
            let views: Vec<(usize, usize)> = st.sections.iter().map(|s| (s.ro_view, s.pages * PAGE_BYTES)).collect();
            let footprint = Footprint {
                sections: views.len(),
                section_bytes: views.iter().map(|&(_, bytes)| bytes).sum(),
                committed_bytes: 0,
                pool_pages: st.pages.capacity(),
                used_pages: st.pages.used_count(),
                mapped_pages: st.pages.mapped_count(),
                arena_bytes,
                given_pages,
                private_bytes: st.regions.allocation_bytes(),
            };
            (footprint, views)
        };
        // Outside the lock: sections live as long as the engine, and the walk takes a while.
        footprint.committed_bytes = views.iter().map(|&(view, bytes)| committed_bytes(view, bytes)).sum();
        footprint
    }

    /// The sequence number of the last journal record, without the engine lock.
    pub fn journal_seq(&self) -> u64 {
        self.journal_seq.load(Ordering::Acquire)
    }

    /// The journal records after `after`, at most `cap`. `None` when the ring no longer holds them.
    pub fn changes(&self, after: u64, cap: usize) -> Option<Vec<GpuChange>> {
        self.journal.lock().unwrap_or_else(PoisonError::into_inner).since(after, cap)
    }

    /// Every live reservation and view as journal records (`RESERVE` and `MAP`), with the sequence
    /// number they are valid for: the changes after that number complete them.
    pub fn snapshot(&self) -> (Vec<GpuChange>, u64) {
        let st = self.lock();
        let seq = self.journal.lock().unwrap_or_else(PoisonError::into_inner).seq();
        let mut out: Vec<GpuChange> = st
            .regions
            .reservations()
            .into_iter()
            .map(|(va, size)| GpuChange { seq, kind: journal::RESERVE, pages: (size / PAGE_BYTES) as u32, va: va as u64, flags: if st.on_demand.contains_key(&va) { journal::ON_DEMAND } else { 0 }, ..GpuChange::default() })
            .collect();
        for (at, region) in st.regions.views() {
            if let Kind::View { first_frame } = region.kind
                && let Some(first) = index_of(first_frame)
            {
                let count = region.size / PAGE_BYTES;
                let mut k = 0;
                while k < count {
                    let shown = Self::shown(&st, at + k * PAGE_BYTES);
                    let n = (k..count).take_while(|&j| Self::shown(&st, at + j * PAGE_BYTES) == shown).count();
                    if shown {
                        out.push(GpuChange { seq, kind: journal::MAP, pages: n as u32, va: (at + k * PAGE_BYTES) as u64, first_page: (first + k) as u32, flags: 0 });
                    }
                    k += n;
                }
            }
        }
        (out, seq)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let me = &raw const *self.state as usize;
        STATES.lock().unwrap_or_else(PoisonError::into_inner).retain(|&s| s != me);
        let st = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        st.sections.clear();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pages_of_one_region_are_asked_to_the_system_once() {
        let mut runs = super::CommittedRuns::default();
        let mut calls = 0;
        let mut ask = |a: usize| {
            calls += 1;
            // committed from 0x100000 to 0x400000, free from there to 0x600000
            Some(if a < 0x400000 { (0x100000, 0x400000, true) } else { (0x400000, 0x600000, false) })
        };
        for page in (0x100000..0x400000).step_by(0x10000) {
            assert!(runs.is_committed(page, &mut ask));
        }
        assert_eq!(runs.queries, 1, "48 pages, one query");
        assert!(!runs.is_committed(0x500000, &mut ask));
        assert!(!runs.is_committed(0x510000, &mut ask));
        assert!(runs.is_committed(0x200000, &mut ask), "back in the first region: asked again");
        assert_eq!((runs.queries, calls), (3, 3));
        assert!(!runs.is_committed(0x700000, |_| None), "a failed query is not committed");
    }

    use super::*;

    const P: usize = PAGE_BYTES;

    /// Tests share one process and one address space, and `reserve(0, ..)` takes whatever range
    /// Windows offers, including a range another test has just released. A test that releases a
    /// reservation and reserves the same address again would then lose it to a neighbour. Each
    /// test holds this lock for its whole run, so no other test touches the address space meanwhile.
    static ADDRESS_SPACE: Mutex<()> = Mutex::new(());

    /// An engine together with the lock of the address space, released after the engine is dropped.
    struct Held {
        engine: Engine,
        _guard: MutexGuard<'static, ()>,
    }

    impl std::ops::Deref for Held {
        type Target = Engine;
        fn deref(&self) -> &Engine {
            &self.engine
        }
    }

    fn held(load: impl FnOnce() -> Result<Engine, EngineError>) -> Option<Held> {
        let guard = ADDRESS_SPACE.lock().unwrap_or_else(PoisonError::into_inner);
        load().ok().map(|engine| {
            engine.reprotect_interval_ms.store(0, Ordering::Relaxed);
            Held { engine, _guard: guard }
        })
    }

    fn engine() -> Option<Held> {
        held(|| Engine::load(64))
    }

    fn write(address: usize, offset: usize, value: u8) {
        // SAFETY: tests only write inside ranges they mapped with read write views.
        unsafe { ((address + offset) as *mut u8).write_volatile(value) };
    }

    fn read(address: usize, offset: usize) -> u8 {
        // SAFETY: tests only read inside ranges they mapped.
        unsafe { ((address + offset) as *const u8).read_volatile() }
    }

    /// The frame of the page that a view maps at `address`, and the pages in use.
    fn frame_at(e: &Engine, address: usize) -> Option<u32> {
        let st = e.lock();
        let (base, region) = st.regions.containing(address)?;
        match region.kind {
            Kind::View { first_frame } => Some(first_frame + ((address - base) / PAGE_BYTES) as u32 * FRAMES_PER_PAGE),
            Kind::Placeholder | Kind::Allocation => None,
        }
    }

    fn used(e: &Engine) -> usize {
        e.lock().pages.used_count()
    }

    /// A title's own `XMemMapPhysicalPages(Scatter)` still wins the race, even after the fault
    /// handler stood in: the real frame replaces the stand-in's zeroed private memory, and the
    /// bytes already written there are gone, as they would be when the title maps real content
    /// over a page it itself only reserved.
    #[test]
    fn a_real_map_can_still_replace_a_page_the_fault_handler_stood_in_for() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        assert!(e.fault_in(area, true), "the fault handler's stand-in commit");
        write(area, 0, 0xEE);
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        assert_eq!(frame_at(&e, area), Some(frames[0]), "the real page replaced the stand-in, not just grew around it");
        write(area, 1, 0x99);
        assert_eq!(read(area, 1), 0x99);
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// `Engine::fault_in`, called directly as the handler does: a reservation with nothing placed in
    /// it takes a write once `fault_in` has run, and reads back what was written.
    #[test]
    fn fault_in_backs_an_empty_reservation_page_for_a_write() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 4 * P, PAGE_READWRITE).unwrap();
        assert!(e.owns(area) && frame_at(&e, area).is_none(), "nothing placed there yet");
        assert!(e.fault_in(area + 2 * P, true));
        write(area + 2 * P, 7, 0x5a);
        assert_eq!(read(area + 2 * P, 7), 0x5a);
        assert!(e.fault_in(area + 2 * P, true), "a second call finds the page backed and places nothing twice");
        assert_eq!(read(area + 2 * P, 7), 0x5a, "the content is unchanged");
        e.release(area).unwrap();
    }

    /// Addresses `fault_in` must leave for the real crash: nothing reserved there, and a page
    /// already committed (a second fault on it is not this race).
    #[test]
    fn fault_in_refuses_what_it_does_not_own() {
        let Some(e) = engine() else { return };
        assert!(!e.fault_in(0, true), "address 0 is never a reservation of this engine");
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        assert_eq!(read(area, 0), 0, "the first access commits the page (a new page reads as zeros)");
        assert!(e.fault_in(area, true), "a page mapped by now (a mapper placed it while the fault waited for the lock): the write resumes");
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// What `VirtualQuery` says of a pool page that was never touched, which the readable and writable
    /// checks of the other DLLs rely on: reserved, mapped, with the view's protection as its allocation
    /// protection. After the first access (through any view) every view of it is committed.
    #[test]
    fn an_untouched_page_is_a_reserved_mapped_page_until_its_first_access_through_any_view() {
        use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEM_RESERVE};
        let query = |at: usize| {
            // SAFETY: plain data, filled by the call.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid output buffer of the size passed.
            assert_ne!(unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) }, 0);
            (info.State, info.Type, info.AllocationProtect, info.Protect)
        };
        let Some(e) = engine() else { return };
        let (rw, ro) = (e.reserve(0, P, PAGE_READWRITE).unwrap(), e.reserve(0, P, PAGE_READONLY).unwrap());
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(rw, &frames).unwrap();
        e.map_frames(ro, &frames).unwrap();
        assert_eq!(query(rw), (MEM_RESERVE, MEM_MAPPED, PAGE_READWRITE, 0));
        assert_eq!(query(ro), (MEM_RESERVE, MEM_MAPPED, PAGE_READONLY, 0));
        assert_eq!(read(ro, 9), 0, "a read through the read only view commits the page");
        assert_eq!(query(rw), (MEM_COMMIT, MEM_MAPPED, PAGE_READWRITE, PAGE_READWRITE), "and the other view sees it committed");
        write(rw, 9, 0x5A);
        assert_eq!(read(ro, 9), 0x5A);
        assert_eq!(query(ro).3, PAGE_READONLY);
        e.release(rw).unwrap();
        e.release(ro).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// A write fault on a page that is mapped but read-only is a real fault: `fault_in` must not
    /// resume it (that would loop on the same fault forever).
    #[test]
    fn fault_in_refuses_a_write_to_a_read_only_view() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READONLY).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        assert_eq!(read(area, 0), 0, "the first read commits the page, the read only view sees it committed");
        assert!(!e.fault_in(area, true), "a write to a read-only view still crashes");
        assert!(e.fault_in(area, false), "a read of it resumes");
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// End to end: a real access violation on an unmapped placeholder, caught by the process
    /// wide vectored handler `install_fault_handler` set up (every engine load installs it, and
    /// `Engine::global` is the one instance the handler itself can see, unlike the ad hoc
    /// `engine()` of the other tests here), recovers instead of crashing the test binary.
    #[test]
    fn a_direct_write_into_a_placeholder_faults_in_through_the_vectored_handler() {
        let _guard = ADDRESS_SPACE.lock().unwrap_or_else(PoisonError::into_inner);
        let Ok(e) = Engine::global() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        assert!(frame_at(e, area).is_none(), "nothing placed there: a direct write would AV without the handler");
        write(area, 10, 0x77);
        assert_eq!(read(area, 10), 0x77, "the vectored handler committed the page and the write landed");
        e.release(area).unwrap();
    }

    #[test]
    fn view_protection_keeps_read_only() {
        assert_eq!(view_protection(PAGE_READONLY), PAGE_READONLY);
        assert_eq!(view_protection(0x1), PAGE_READWRITE);
        assert_eq!(view_protection(0x40), PAGE_READWRITE);
        assert_eq!(view_protection(PAGE_READWRITE), PAGE_READWRITE);
    }

    #[test]
    fn the_pages_one_address_shows_can_be_shown_at_another() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 12 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(3).unwrap();
        e.map_frames(area, &frames).unwrap();
        write(area + P, 5, 0x77);
        e.show_pages_at(area + 6 * P, area, 3).unwrap();
        assert_eq!(read(area + 7 * P, 5), 0x77, "the second address shows the same page");
        write(area + 8 * P, 9, 0x21);
        assert_eq!(read(area + 2 * P, 9), 0x21, "a write at the second address shows at the first");
        assert_eq!((frame_at(&e, area + 6 * P), frame_at(&e, area + 8 * P)), (frame_at(&e, area), frame_at(&e, area + 2 * P)));
        // one source page shown several times, as a console title fills a range with one page
        for k in 0..3 {
            e.show_pages_at(area + (9 + k) * P, area + P, 1).unwrap();
        }
        assert_eq!((read(area + 9 * P, 5), read(area + 11 * P, 5)), (0x77, 0x77));
        // a source that shows nothing maps nothing and leaves the destination as it was
        assert!(e.show_pages_at(area + 6 * P, area + 3 * P, 1).is_err(), "page 3 of the area is a placeholder");
        assert_eq!(read(area + 6 * P, 0), read(area, 0));
        assert!(e.show_pages_at(0x10000, area, 1).is_err(), "the destination is not in a reservation");
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn one_page_appears_at_two_addresses() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        assert_eq!(frames, [16], "the first page of the section is frame 16");
        e.map_frames(area, &frames).unwrap();
        e.map_frames(area + 5 * P, &frames).unwrap();
        write(area, 0, 0xAB);
        write(area, P - 1, 0xCD);
        assert_eq!((read(area + 5 * P, 0), read(area + 5 * P, P - 1)), (0xAB, 0xCD));
        write(area + 5 * P, 77, 0x5A);
        assert_eq!(read(area, 77), 0x5A, "a write at the second address shows at the first");
        assert_eq!((frame_at(&e, area), frame_at(&e, area + 5 * P + 100), frame_at(&e, area + 2 * P)), (Some(16), Some(16), None));
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
        assert_eq!(used(&e), 0);
    }

    #[test]
    fn frame_zero_and_unmap_pages_give_reserved_pages_that_map_again_with_their_bytes() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 4 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(2).unwrap();
        assert_eq!(frames, [16, 32]);
        e.map_frames(area + P, &frames).unwrap();
        write(area + P, 1, 9);
        write(area + 2 * P, 2, 8);
        e.map_frames(area + P, &[0, 32]).unwrap();
        assert_eq!((frame_at(&e, area + P), frame_at(&e, area + 2 * P)), (None, Some(32)), "an entry of 0 unmaps its page only");
        assert_eq!(read(area + 2 * P, 2), 8, "the other page stays mapped");
        e.unmap_pages(area + P, 2).unwrap();
        assert_eq!(frame_at(&e, area + 2 * P), None);
        e.unmap_pages(area, 4).unwrap();
        e.map_frames(area + 2 * P, &frames[..1]).unwrap();
        assert_eq!(read(area + 2 * P, 1), 9, "the page kept its bytes while unmapped");
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn scatter_puts_each_frame_at_its_own_address() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 6 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(3).unwrap();
        e.map_scatter(&[(area + 4 * P, frames[0]), (area, frames[2]), (area + 2 * P + 123, frames[1])]).unwrap();
        write(area + 4 * P, 0, 1);
        write(area + 2 * P, 0, 2);
        write(area, 0, 3);
        e.map_frames(area + P, &frames[..1]).unwrap();
        assert_eq!(read(area + P, 0), 1);
        assert_eq!((frame_at(&e, area), frame_at(&e, area + 2 * P), frame_at(&e, area + 4 * P)), (Some(frames[2]), Some(frames[1]), Some(frames[0])));
        // One entry fails (a frame that was never allocated): nothing changes.
        let failed = e.map_scatter(&[(area + 3 * P, frames[1]), (area + 5 * P, 16 * 40)]);
        assert_eq!(failed, Err(EngineError::Pages(PageError::NotAllocated(16 * 40))));
        assert_eq!(frame_at(&e, area + 3 * P), None);
        e.map_scatter(&[(area, 0), (area + 2 * P, 0)]).unwrap();
        assert_eq!((frame_at(&e, area), frame_at(&e, area + 2 * P)), (None, None));
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn mapping_over_a_mapped_page_replaces_it_and_keeps_the_rest_of_its_view() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 6 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(4).unwrap();
        e.map_frames(area, &frames[..3]).unwrap();
        for k in 0..3 {
            write(area + k * P, 0, 10 + k as u8);
        }
        e.map_frames(area + P, &frames[3..]).unwrap();
        write(area + P, 0, 99);
        assert_eq!((read(area, 0), read(area + P, 0), read(area + 2 * P, 0)), (10, 99, 12), "the middle page is the new one, the others stay");
        assert_eq!((frame_at(&e, area), frame_at(&e, area + P), frame_at(&e, area + 2 * P)), (Some(frames[0]), Some(frames[3]), Some(frames[2])));
        e.map_frames(area, &frames[..3]).unwrap();
        assert_eq!(read(area + P, 0), 11, "the old page is back");
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn a_page_freed_while_mapped_stays_readable_and_comes_back_after_unmapping() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        write(area, 5, 0x77);
        e.free_pages(&frames).unwrap();
        assert_eq!(read(area, 5), 0x77, "the view still reads the page");
        assert_eq!(e.free_pages(&frames), Err(EngineError::Pages(PageError::NotAllocated(16))), "a second free is refused");
        assert_eq!(e.map_frames(area + P, &frames), Err(EngineError::Pages(PageError::NotAllocated(16))), "a freed frame is not mapped again");
        assert_eq!(e.allocate_pages(1).unwrap(), [32], "the page is not handed out while a view maps it");
        e.unmap_pages(area, 1).unwrap();
        assert_eq!(e.allocate_pages(1).unwrap(), [16], "now it is free again");
        e.release(area).unwrap();
    }

    #[test]
    fn mapping_a_run_inside_a_large_placeholder_uses_one_view() {
        let Some(e) = engine() else { return };
        const RESERVATION: usize = 0x1_A000_0000; // 6.5 GiB of virtual address space
        let area = e.reserve(0, RESERVATION, PAGE_READWRITE).unwrap();
        let tail = area + RESERVATION - 4 * P;
        let frames = e.allocate_pages(4).unwrap();
        e.map_frames(tail, &frames).unwrap();
        assert_eq!(e.lock().regions.views_in(tail, 4 * P).unwrap(), [tail]);
        assert_eq!(frame_at(&e, tail), Some(frames[0]));
        assert_eq!(frame_at(&e, tail + 3 * P), Some(frames[3]));
        write(tail, 0, 1);
        write(tail + 3 * P, P - 1, 2);
        assert_eq!(read(tail, 0), 1);
        assert_eq!(read(tail + 3 * P, P - 1), 2);
        e.unmap_pages(tail, 4).unwrap();
        e.free_pages(&frames).unwrap();
        assert_eq!(used(&e), 0);
        e.release(area).unwrap();
    }

    #[test]
    fn memory_placed_in_a_reservation_goes_back_to_a_placeholder() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        for (offset, cuts) in [(0, 1), (3 * P, 2), (6 * P, 1)] {
            let at = area + offset;
            assert_eq!(e.place(at, 2 * P, true, PAGE_READWRITE), Ok(Placed { cuts, committed: true, pages: true }));
            write(at, 0, 1);
            write(at, 2 * P - 1, 2);
            assert!(e.is_allocation(at) && !e.is_allocation(area + 2 * P));
            assert_eq!(e.place(at, P, true, PAGE_READWRITE), Ok(Placed { cuts: 0, committed: true, pages: true }), "already committed: nothing to do, as VirtualAlloc");
            assert_eq!(e.unplace(at), Ok(2 * P));
            assert!(!e.is_allocation(at));
        }
        assert_eq!(e.lock().regions.at(area).map(|r| r.size), Some(8 * P), "all joined again");
        assert_eq!(e.place(area + 7 * P, 2 * P, true, PAGE_READWRITE), Err(EngineError::Regions(RegionError::NotInPlaceholder)), "sticks out");
        e.place(area + P, P, false, PAGE_READWRITE).unwrap();
        e.release(area).unwrap();
        assert!(!e.owns(area));
    }

    #[test]
    fn a_commit_is_backed_by_pool_pages_that_come_back_cleared() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let placed = e.place(area + P, 3 * P, true, PAGE_READWRITE).unwrap();
        assert_eq!((placed.committed, placed.pages), (true, true));
        assert_eq!(used(&e), 3, "three pages of the pool");
        let first = frame_at(&e, area + P).unwrap();
        assert_eq!((frame_at(&e, area + 3 * P), frame_at(&e, area + 4 * P)), (Some(first + 32), None), "one run of consecutive frames");
        write(area + P, 5, 0x42);
        write(area + 4 * P - 1, 0, 0x43);
        assert!(e.is_allocation(area + P));
        assert_eq!(e.unplace(area + P), Ok(3 * P));
        assert!(!e.is_allocation(area + P));
        assert_eq!(used(&e), 0, "the pages went back");
        assert_eq!(e.lock().regions.at(area).map(|r| r.size), Some(8 * P), "the placeholder is whole again");
        // The same pages are handed out again, and a commit reads as zeros.
        e.place(area + 2 * P, 3 * P, true, PAGE_READWRITE).unwrap();
        assert_eq!(frame_at(&e, area + 2 * P), Some(first));
        assert_eq!((read(area + 2 * P, 5), read(area + 5 * P - 1, 0)), (0, 0));
        // Releasing the reservation gives the pages back too.
        e.release(area).unwrap();
        assert_eq!(used(&e), 0);
    }

    #[test]
    fn a_commit_without_a_free_run_is_private_memory() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let placed = e.place(area, 6 * P, true, PAGE_READWRITE).unwrap();
        assert_eq!((placed.committed, placed.pages), (true, false), "the pool has 4 pages, the request 6");
        write(area, 0, 1);
        write(area + 6 * P - 1, 0, 2);
        assert_eq!(used(&e), 0);
        assert!(e.is_allocation(area));
        assert_eq!(e.unplace(area), Ok(6 * P));
        e.release(area).unwrap();
    }

    #[test]
    fn a_cut_placeholder_takes_pages_and_a_cut_by_the_title_is_accepted() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        assert_eq!(e.cut(area, 3 * P), Ok(1));
        assert_eq!(e.cut(area, 3 * P), Ok(0), "already exact");
        // The title cuts a piece itself with VirtualFree, behind the registry's back.
        e.api.split(area + 3 * P, P).unwrap();
        assert_eq!(e.cut(area + 3 * P, P), Ok(1), "the registry cuts, Windows had cut already");
        let frames = e.allocate_pages(3).unwrap();
        e.map_frames(area, &frames).unwrap();
        write(area, 0, 5);
        write(area + 3 * P - 1, 0, 6);
        e.unmap_pages(area, 3).unwrap();
        e.free_pages(&frames).unwrap();
        e.release(area).unwrap();
        assert!(!e.owns(area));
        assert_eq!(e.reserve(area, 8 * P, PAGE_READWRITE), Ok(area), "every piece was released");
        e.release(area).unwrap();
    }

    #[test]
    fn a_range_outside_one_reservation_is_refused() {
        let Some(e) = engine() else { return };
        let one = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(3).unwrap();
        assert_eq!(e.map_frames(one, &frames), Err(EngineError::Regions(RegionError::NotInPlaceholder)), "three pages in a reservation of two");
        assert_eq!(frame_at(&e, one), None);
        assert_eq!(e.map_frames(one + 64 * P * 1024, &frames[..1]), Err(EngineError::Regions(RegionError::NotInPlaceholder)), "not reserved");
        assert_eq!(e.unmap_pages(one, 3), Err(EngineError::Regions(RegionError::NotInPlaceholder)));
        e.release(one).unwrap();
        assert_eq!(e.release(one), Err(EngineError::Regions(RegionError::Unknown)));
    }

    #[test]
    fn releasing_a_reservation_unmaps_its_views_and_gives_the_addresses_back() {
        let Some(e) = engine() else { return };
        let area = e.reserve(0, 6 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(3).unwrap();
        e.map_frames(area + P, &frames).unwrap();
        e.map_frames(area + 5 * P, &frames[..1]).unwrap();
        assert!(e.owns(area + 3 * P + 7), "an address inside the reservation");
        assert_eq!(e.release(area + P), Err(EngineError::Regions(RegionError::Unknown)), "not the start of the reservation");
        e.release(area).unwrap();
        assert!(!e.owns(area));
        e.free_pages(&frames).unwrap();
        assert_eq!(used(&e), 0, "no view is left that keeps a page");
        assert_eq!(e.reserve(area, 6 * P, PAGE_READWRITE), Ok(area), "the range can be reserved again");
        e.release(area).unwrap();
    }

    #[test]
    fn a_fixed_pool_runs_out_cleanly() {
        let Some(e) = held(|| Engine::load_limited(64, 64)) else { return };
        assert_eq!(e.allocate_pages(65), Err(EngineError::OutOfPages));
        assert_eq!(used(&e), 0);
        let all = e.allocate_pages(64).unwrap();
        assert_eq!(e.allocate_pages(1), Err(EngineError::OutOfPages));
        e.free_pages(&all).unwrap();
        assert_eq!(e.allocate_pages(64).map(|f| f.len()), Ok(64));
    }

    #[test]
    fn a_request_larger_than_the_free_pages_grows_the_pool_and_takes_one_run_over_both_sections() {
        let Some(e) = held(|| Engine::load(8)) else { return };
        let area = e.reserve(0, 24 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(20).unwrap();
        assert_eq!(e.lock().pages.capacity(), 8 + crate::pages::GROW_STEP_PAGES, "the pool grew by one step");
        assert_eq!(e.lock().sections.len(), 2);
        assert_eq!(frames[0], 16, "the run starts at the first free page and goes on into the new section");
        e.map_frames(area, &frames).unwrap();
        for page in [0, 7, 8, 19] {
            write(area + page * P, page, page as u8 + 1);
        }
        for page in [0, 7, 8, 19] {
            assert_eq!(read(area + page * P, page), page as u8 + 1);
        }
        assert_eq!(e.lock().regions.views_in(area, 20 * P).unwrap(), [area, area + 8 * P], "one view per section");
        e.unmap_pages(area, 24).unwrap();
        e.free_pages(&frames).unwrap();
        assert_eq!(used(&e), 0);
        e.release(area).unwrap();
    }

    #[test]
    fn scattered_pages_over_the_end_of_a_section_map_in_parts() {
        let step = crate::pages::GROW_STEP_PAGES;
        let Some(e) = held(|| Engine::load_limited(8, 8 + step)) else { return };
        let area = e.reserve(0, 24 * P, PAGE_READWRITE).unwrap();
        let first = e.allocate_pages(8).unwrap();
        let second = e.allocate_pages(step).unwrap();
        assert_eq!(e.lock().sections.len(), 2);
        // The pool is at its limit. Pages 5 to 7 and 8 to 10 are free: next to each other, in two sections.
        e.free_pages(&first[5..]).unwrap();
        e.free_pages(&second[..3]).unwrap();
        let frames = e.allocate_pages(6).unwrap();
        assert_eq!(frames, [16 * 6, 16 * 7, 16 * 8, 16 * 9, 16 * 10, 16 * 11]);
        e.map_frames(area, &frames).unwrap();
        for page in [0, 2, 3, 5] {
            write(area + page * P, page, page as u8 + 1);
        }
        for page in [0, 2, 3, 5] {
            assert_eq!(read(area + page * P, page), page as u8 + 1);
        }
        assert_eq!(e.lock().regions.views_in(area, 6 * P).unwrap(), [area, area + 3 * P], "one view for each section");
        // The pages of the second section map again, one by one, over the first view.
        e.map_frames(area + 21 * P, &frames[2..4]).unwrap();
        assert_eq!((read(area + 21 * P, 2), read(area + 22 * P, 3)), (3, 4));
        e.unmap_pages(area, 24).unwrap();
        e.free_pages(&frames).unwrap();
        e.release(area).unwrap();
    }

    #[test]
    fn the_private_views_of_the_sections_see_the_pages_and_the_journal_follows_the_maps() {
        let step = crate::pages::GROW_STEP_PAGES;
        let Some(e) = held(|| Engine::load_limited(8, 8 + step)) else { return };
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let first = e.allocate_pages(8).unwrap();
        let second = e.allocate_pages(step).unwrap();
        e.map_frames(area, &[first[2], first[3], second[0]]).unwrap();
        write(area, P + 7, 0x5A);
        write(area, 2 * P + 9, 0x6B);
        // Section 0 holds pool pages 0 to 7, section 1 the pages from 8.
        let (ro0, start0, pages0) = e.section_view(0).unwrap();
        let (ro1, start1, pages1) = e.section_view(1).unwrap();
        assert_eq!((start0, pages0, start1, pages1), (0, 8, 8, step as u32));
        assert_eq!(read(ro0, 3 * P + 7), 0x5A, "page 3 through the read only view");
        assert_eq!(read(ro1, 9), 0x6B, "the first page of the second section");
        assert!(e.section_view(2).is_none());
        let changes = e.changes(0, 100).unwrap();
        let kinds: Vec<(u32, u32, u32)> = changes.iter().map(|c| (c.kind, c.pages, c.first_page)).collect();
        assert_eq!(kinds, [(journal::RESERVE, 8, 0), (journal::MAP, 2, 2), (journal::MAP, 1, 8)], "a view for each section");
        let (snapshot, seq) = e.snapshot();
        assert_eq!(seq, e.journal_seq());
        assert_eq!(snapshot.iter().map(|c| (c.kind, c.va, c.pages)).collect::<Vec<_>>(), [(journal::RESERVE, area as u64, 8), (journal::MAP, area as u64, 2), (journal::MAP, (area + 2 * P) as u64, 1)]);
        e.unmap_pages(area, 8).unwrap();
        e.release(area).unwrap();
        assert_eq!(e.changes(seq, 100).unwrap().iter().map(|c| c.kind).collect::<Vec<_>>(), [journal::UNMAP, journal::UNMAP, journal::RELEASE]);
    }

    /// The process engine with the dirty tracking on, and the lock of the address space. The fault
    /// handler only serves the process engine, so the tracking tests use it, 64 MiB of pool.
    fn tracked() -> Option<(MutexGuard<'static, ()>, &'static Engine)> {
        let guard = ADDRESS_SPACE.lock().unwrap_or_else(PoisonError::into_inner);
        let engine = Engine::global().ok()?;
        engine.set_tracking(true);
        engine.reprotect_interval_ms.store(0, Ordering::Relaxed);
        Some((guard, engine))
    }

    fn page_of(frame: u32) -> u32 {
        index_of(frame).unwrap() as u32
    }

    #[test]
    fn dirty_take_and_write_faults_finish_while_the_state_lock_is_held() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        e.map_frames(area + P, &frames).unwrap();
        write(area, 0, 1);
        let (send, recv) = std::sync::mpsc::channel();
        let state = e.lock();
        let worker = std::thread::spawn(move || {
            let first = e.take_dirty(usize::MAX);
            write(area + P, 0, 2);
            let second = e.take_dirty(usize::MAX);
            send.send((first, second, read(area, 0))).unwrap();
        });
        let result = recv.recv_timeout(std::time::Duration::from_secs(5));
        drop(state);
        worker.join().unwrap();
        let (first, second, value) = result.expect("dirty tracking waited for the state lock");
        assert!(first.contains(&page_of(frames[0])));
        assert!(second.contains(&page_of(frames[0])));
        assert_eq!(value, 2);
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn concurrent_writes_alias_maps_and_dirty_takes_keep_the_last_write() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, 3 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(2).unwrap();
        e.map_frames(area, &frames[..1]).unwrap();
        e.map_frames(area + P, &frames[..1]).unwrap();
        write(area, 0, 1);
        e.take_dirty(usize::MAX);
        let barrier = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                for i in 0..500 {
                    write(area + (i % 2) * P, 0, (i % 251) as u8);
                    std::thread::yield_now();
                }
            });
            scope.spawn(|| {
                barrier.wait();
                for _ in 0..200 {
                    e.map_frames(area + 2 * P, &frames[1..]).unwrap();
                    e.unmap_pages(area + 2 * P, 1).unwrap();
                }
            });
            barrier.wait();
            for _ in 0..500 {
                let pages = e.take_dirty(1);
                assert!(pages.len() <= 1);
                std::thread::yield_now();
            }
        });
        write(area + P, 0, 253);
        assert!(e.take_dirty(usize::MAX).contains(&page_of(frames[0])));
        assert_eq!(read(area, 0), 253);
        assert!(e.dirty.lock().unwrap().track.as_ref().unwrap().writable_page(area + 2 * P).is_none());
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn a_write_after_take_dirty_faults_and_the_page_comes_back() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, 4 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(2).unwrap();
        e.map_frames(area, &frames).unwrap();
        let (a, b) = (page_of(frames[0]), page_of(frames[1]));
        write(area, 0, 0);
        write(area + P, 0, 0);
        let first = e.take_dirty(1 << 20);
        assert!(first.contains(&a) && first.contains(&b), "the first access commits a page and makes it dirty");
        assert!(!e.take_dirty(1 << 20).contains(&a), "a page comes back once");
        assert_eq!(e.title_protection(area), Some(PAGE_READWRITE), "the title still sees read write");
        write(area, 0, 1);
        assert_eq!(read(area, 0), 1, "the write went through the fault handler");
        let again = e.take_dirty(1 << 20);
        assert!(again.contains(&a) && !again.contains(&b), "only the written page");
        // A second view of the page: a write through it is seen too, and opens every view.
        let alias = e.reserve(0, P, PAGE_READWRITE).unwrap();
        e.map_frames(alias, &frames[..1]).unwrap();
        assert!(e.take_dirty(1 << 20).contains(&a), "a new view counts as a write");
        write(alias, 5, 9);
        assert_eq!(read(area, 5), 9, "one page, two views");
        assert!(e.take_dirty(1 << 20).contains(&a));
        write(area, 6, 3);
        assert_eq!(read(alias, 6), 3);
        assert!(e.take_dirty(1 << 20).contains(&a), "a write through the first view is seen as well");
        e.unmap_pages(alias, 1).unwrap();
        e.release(alias).unwrap();
        e.unmap_pages(area, 2).unwrap();
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn a_file_read_into_a_clean_page_fails_until_the_write_is_prepared() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        e.take_dirty(1 << 20);
        let path = std::env::temp_dir().join(format!("xmem_track_{}.bin", std::process::id()));
        std::fs::write(&path, [0x33u8; 512]).unwrap();
        let read_into = |at: usize| {
            let mut file = std::fs::File::open(&path).unwrap();
            // SAFETY: the page is mapped read write for 64 KiB, the kernel write may fail but does not fault in this process.
            let target = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, 512) };
            std::io::Read::read(&mut file, target)
        };
        let failed = read_into(area).unwrap_err();
        assert_eq!(failed.raw_os_error(), Some(998), "ERROR_NOACCESS, the hazard the guard exists for");
        e.prepare_write(area + 100, 8);
        assert_eq!(read_into(area).unwrap(), 512, "the page was opened");
        assert_eq!(read(area, 511), 0x33);
        assert!(e.take_dirty(1 << 20).contains(&page_of(frames[0])), "the kernel write is copied");
        assert!(e.take_dirty(1 << 20).contains(&page_of(frames[0])), "and the page stays held and unprotected");
        std::fs::remove_file(&path).unwrap();
        e.unmap_pages(area, 1).unwrap();
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// Bytes of a pattern that differs from page to page and inside a page.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 + i / PAGE_BYTES) as u8).collect()
    }

    #[test]
    fn a_pool_write_reaches_scattered_pages_that_are_mapped_afterwards() {
        let Some(e) = engine() else { return };
        let all = e.allocate_pages(6).unwrap();
        // Pages 4, 1 and 3 of the run, in that order, as a title's page table may hold them.
        let frames = [all[4], all[1], all[3]];
        let data = pattern(2 * P + 100);
        assert_eq!(e.write_frames(&frames, 300, &data).unwrap(), [], "no view shows the pages yet");
        let area = e.reserve(0, 3 * P, PAGE_READWRITE).unwrap();
        e.map_frames(area, &frames).unwrap();
        // SAFETY: the three pages are mapped read write.
        let seen = unsafe { std::slice::from_raw_parts(area as *const u8, 3 * P) };
        assert_eq!(&seen[300..300 + data.len()], &data[..], "the bytes follow the page array, not the frame order");
        assert!(seen[..300].iter().all(|&b| b == 0), "before the offset");
        assert!(seen[300 + data.len()..].iter().all(|&b| b == 0), "the write ends inside the last page");
        // Written again while mapped: the title addresses of the bytes come back, in address order.
        assert_eq!(e.write_frames(&frames, 300, &data).unwrap(), [(area + 300, data.len())]);
        let alias = e.reserve(0, P, PAGE_READWRITE).unwrap();
        e.map_frames(alias, &frames[2..]).unwrap();
        // The last 4 bytes of page 1 and the first 4 of page 3: one run in the area, page 3 again in the alias.
        let mut want = vec![(area + 2 * P - 4, 8), (alias, 4)];
        want.sort_unstable();
        assert_eq!(e.write_frames(&frames[1..], P - 4, &[9; 8]).unwrap(), want);
        e.unmap_pages(alias, 1).unwrap();
        e.release(alias).unwrap();
        e.unmap_pages(area, 3).unwrap();
        e.release(area).unwrap();
        e.free_pages(&all).unwrap();
    }

    #[test]
    fn a_pool_write_with_a_page_not_allocated_or_too_few_frames_writes_nothing() {
        let Some(e) = engine() else { return };
        let frames = e.allocate_pages(2).unwrap();
        let spare = e.allocate_pages(1).unwrap();
        e.free_pages(&spare).unwrap();
        assert!(matches!(e.write_frames(&[frames[0], spare[0]], 0, &[1; P + 1]), Err(EngineError::Pages(PageError::NotAllocated(_)))));
        assert!(matches!(e.write_frames(&[frames[0], 0], 0, &[1; P + 1]), Err(EngineError::Pages(_))));
        assert!(matches!(e.write_frames(&frames[..1], 1, &[1; P]), Err(EngineError::Invalid(_))), "the bytes reach a second page");
        assert!(matches!(e.write_frames(&frames, P, &[1]), Err(EngineError::Invalid(_))));
        assert!(e.write_frames(&[], 0, &[]).is_ok());
        let area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        e.map_frames(area, &frames[..1]).unwrap();
        assert_eq!(read(area, 0), 0, "the refused writes left the first page alone");
        e.unmap_pages(area, 1).unwrap();
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn a_pool_write_makes_a_clean_page_dirty_and_its_view_writable() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        e.take_dirty(1 << 20);
        assert!(!e.take_dirty(1 << 20).contains(&page_of(frames[0])), "clean, its view is protected");
        assert_eq!(e.write_frames(&frames, 8, &[0x5A; 16]).unwrap(), [(area + 8, 16)]);
        assert_eq!(read(area, 8), 0x5A, "the title sees the bytes");
        assert!(e.take_dirty(1 << 20).contains(&page_of(frames[0])), "the mirror copies the page again");
        e.unmap_pages(area, 1).unwrap();
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn a_reused_clean_page_is_zeroed_by_a_commit_without_a_fault() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        e.place(area, P, true, PAGE_READWRITE).unwrap();
        write(area, 0, 0xAA);
        e.take_dirty(1 << 20);
        e.unplace(area).unwrap();
        e.place(area, P, true, PAGE_READWRITE).unwrap();
        assert_eq!(read(area, 0), 0, "the old bytes are gone");
        assert!(e.writable(area, P));
        assert!(!e.writable(area, 2 * P), "the second page is a placeholder");
        e.unplace(area).unwrap();
        e.release(area).unwrap();
    }

    /// A commit with a console page bit (`commit_region`): below 16 GiB on the large page alignment, every
    /// page in the pool, and nothing of it reaches the GPU mirror until the GPU is given an address in it.
    #[test]
    fn a_commit_region_reaches_the_mirror_only_where_the_gpu_was_given_it() {
        const LARGE: usize = 0x20_0000;
        let Some((_guard, e)) = tracked() else { return };
        let used = used(e);
        let seq = e.journal_seq();
        let (at, pages) = e.commit_region(0, 8 * P, LARGE, PAGE_READWRITE).unwrap();
        assert!(pages, "pool pages back it");
        assert!(at + 8 * P <= GPU_LOW_END && at.is_multiple_of(LARGE), "below 16 GiB, 2 MiB aligned: 0x{at:x}");
        assert!(e.is_region(at) && !e.is_region(at + P));
        let page = |k: usize| page_of(frame_at(e, at + k * P).unwrap());
        assert_eq!(e.changes(seq, 100).unwrap().iter().map(|c| (c.kind, c.flags)).collect::<Vec<_>>(), [(journal::RESERVE, journal::ON_DEMAND)], "no MAP before the GPU asks");
        write(at + P, 3, 7);
        assert!(!e.take_dirty(1 << 20).contains(&page(1)), "a page the GPU was not given is neither copied nor protected");
        let seq = e.journal_seq();
        assert_eq!(e.gpu_use(at + P + 16, 100), 1);
        assert_eq!(e.gpu_use(at + P, P), 0, "known already");
        assert_eq!(e.gpu_use(at + 64 * P, 4), 0, "outside the commit");
        let maps = e.changes(seq, 100).unwrap();
        assert_eq!(maps.iter().map(|c| (c.kind, c.va, c.pages, c.first_page)).collect::<Vec<_>>(), [(journal::MAP, (at + P) as u64, 1, page(1))]);
        assert!(e.take_dirty(1 << 20).contains(&page(1)), "copied at the next sync");
        write(at + P, 4, 8);
        assert_eq!(read(at + P, 3), 7);
        assert!(e.take_dirty(1 << 20).contains(&page(1)), "a write after the sync faults and is seen");
        assert_eq!(e.gpu_use(at + 2 * P, 3 * P), 3);
        let (snapshot, _) = e.snapshot();
        let mine: Vec<(u32, u64, u32, u32)> = snapshot.iter().filter(|c| (at as u64..(at + 8 * P) as u64).contains(&c.va)).map(|c| (c.kind, c.va, c.pages, c.flags)).collect();
        assert_eq!(mine, [(journal::RESERVE, at as u64, 8, journal::ON_DEMAND), (journal::MAP, (at + P) as u64, 4, 0)], "the snapshot shows the pages the GPU was given");
        // A commit over the committed range changes nothing, as VirtualAlloc.
        assert_eq!(e.place(at, 2 * P, true, PAGE_READWRITE).map(|p| p.pages), Ok(true));
        assert_eq!(read(at + P, 3), 7);
        let seq = e.journal_seq();
        e.release(at).unwrap();
        assert!(!e.is_region(at) && !e.owns(at));
        assert_eq!(e.changes(seq, 100).unwrap().last().map(|c| c.kind), Some(journal::RELEASE));
        assert_eq!(super::tests::used(e), used, "the pages went back to the pool");
    }

    /// A commit larger than the free run of a section takes a run that goes on into a new section, mapped
    /// as one view per section (an arena above 2 GiB is larger than a section of 2 GiB).
    #[test]
    fn a_commit_region_may_cross_into_a_new_section() {
        let step = crate::pages::GROW_STEP_PAGES;
        let Some(e) = held(|| Engine::load_limited(8, 8 + step)) else { return };
        let taken = e.allocate_pages(4).unwrap();
        let (at, pages) = e.commit_region(0, 8 * P, 0, PAGE_READWRITE).unwrap();
        assert!(pages);
        let frames: Vec<u32> = (0..8).map(|k| frame_at(&e, at + k * P).unwrap()).collect();
        assert_eq!(frames, (0..8).map(|k| frames[0] + k * FRAMES_PER_PAGE).collect::<Vec<_>>(), "one run");
        assert_eq!(index_of(frames[0]), Some(4), "pages 4 to 7 of the first section, then the second");
        for k in 0..8 {
            write(at + k * P, 1, k as u8 + 1);
        }
        assert!((0..8).all(|k| read(at + k * P, 1) == k as u8 + 1));
        e.release(at).unwrap();
        e.free_pages(&taken).unwrap();
        assert_eq!(used(&e), 0);
    }

    /// The commit this process holds through views of sections, in bytes: the committed pages of
    /// `MEM_MAPPED` regions found by `VirtualQuery` over the address space. A section is charged to the
    /// system commit when it is made, and the process counters (`PrivateUsage`, `PagefileUsage`) do not
    /// show it, so this is the number of the process that follows the charge. Not the free commit of
    /// the system, which other processes (a game run of 14 GB) change meanwhile.
    fn mapped_commit() -> u64 {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
        let (mut at, mut total) = (0usize, 0u64);
        loop {
            // SAFETY: MEMORY_BASIC_INFORMATION is plain data.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid output buffer of the size passed. Querying an address changes nothing.
            if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
                return total;
            }
            if info.State == MEM_COMMIT && info.Type == MEM_MAPPED {
                total += info.RegionSize as u64;
            }
            match (info.BaseAddress as usize).checked_add(info.RegionSize) {
                Some(next) if next > at => at = next,
                _ => return total,
            }
        }
    }

    /// The commit charge of the pool: a commit region charges nothing until its pages are touched, then
    /// the touched pages and no more, its pages are reused after `XMemFree`, and the whole charge is back
    /// once the engine is gone (the process ends). The engine's own count of committed section bytes is
    /// exact. The mapped commit of this process counts each view of a page (the title's and the engine's
    /// two), so other processes changing the free commit of the system do not matter.
    #[test]
    fn a_commit_region_charges_only_touched_pages_while_the_engine_lives_and_gives_them_back() {
        const MIB: u64 = 1 << 20;
        let pages = 16384; // 1 GiB
        let touched = 8192; // 512 MiB
        // The lock for the whole test: no other test makes or drops sections between the readings.
        let _guard = ADDRESS_SPACE.lock().unwrap_or_else(PoisonError::into_inner);
        let before = mapped_commit();
        {
            let Ok(e) = Engine::load_limited(16, 16 + pages) else { return };
            let (at, _) = e.commit_region(0, pages * P, 0, PAGE_READWRITE).unwrap();
            let untouched = mapped_commit();
            assert_eq!(e.footprint().committed_bytes, 0, "a commit region charges nothing before its first access");
            for k in 0..touched {
                write(at, k * P + 7, 1);
            }
            let during = mapped_commit();
            assert_eq!(e.footprint().committed_bytes, touched * P, "each first access commits exactly its page");
            let pool = e.lock().pages.capacity();
            e.release(at).unwrap();
            assert_eq!(used(&e), 0, "XMemFree gives the pages back to the pool");
            let (again, _) = e.commit_region(0, pages * P, 0, PAGE_READWRITE).unwrap();
            assert_eq!(e.lock().pages.capacity(), pool, "a second commit reuses them, the pool does not grow");
            assert_eq!(read(again, (touched - 1) * P + 7), 0, "and finds them cleared");
            assert_eq!(read(again, (pages - 1) * P), 0, "a page never touched reads as zeros");
            e.release(again).unwrap();
            eprintln!("mapped commit: {} MiB before, {} MiB with 1 GiB committed and untouched, {} MiB with 512 MiB touched", before / MIB, untouched / MIB, during / MIB);
            assert!(untouched.saturating_sub(before) < 64 * MIB, "an untouched commit region is not charged");
            assert!(during.saturating_sub(before) >= 512 * MIB, "the touched pages are charged while the engine lives");
        }
        let after = mapped_commit();
        eprintln!("mapped commit: {} MiB after the engine is gone", after / MIB);
        assert!(after.abs_diff(before) < 256 * MIB, "the charge is back: {} MiB before, {} MiB after", before / MIB, after / MIB);
    }

    #[test]
    fn growth_stops_at_the_limit_of_the_pool() {
        let Some(e) = held(|| Engine::load_limited(8, 12)) else { return };
        assert_eq!(e.allocate_pages(13), Err(EngineError::OutOfPages));
        assert_eq!(e.lock().pages.capacity(), 8, "nothing grew for a request that cannot be met");
        assert_eq!(e.allocate_pages(12).map(|f| f.len()), Ok(12));
        assert_eq!(e.lock().pages.capacity(), 12);
    }

    #[test]
    fn the_engine_starts_or_says_what_is_missing() {
        let _guard = ADDRESS_SPACE.lock().unwrap_or_else(PoisonError::into_inner);
        match Engine::load(1) {
            Ok(_) => {}
            Err(e) => assert!(matches!(e, EngineError::Unsupported(_))),
        }
    }

    #[test]
    fn write_fault_recovers_when_concurrent_commit_and_take_dirty_protect_page() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();

        // Thread A faults on an uncommitted tracked view page: first on_write_fault sees !accessible.
        assert!(!accessible(area, false));
        assert!(!e.on_write_fault(area));

        // Thread B commits the page and take_dirty marks it clean with PAGE_READONLY.
        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.take_dirty(usize::MAX).contains(&page_of(frames[0])));
        assert!(accessible(area, false));
        assert!(!accessible(area, true));

        // Thread A continues: commit_on_access_in sees it already committed, fault_in alone would fail,
        // and the second on_write_fault reopens the page so the write succeeds.
        assert!(!Engine::commit_on_access_in(&e.state, area));
        assert!(!e.fault_in(area, true));
        assert!(e.on_write_fault(area));
        assert!(accessible(area, true));
        write(area, 3, 0x42);
        assert_eq!(read(area, 3), 0x42);

        // When the tracker already considers the page dirty (on_fault returns None) yet the page is
        // still PAGE_READONLY, on_write_fault must restore PAGE_READWRITE instead of returning false.
        Engine::protect_runs(vec![area], PAGE_READONLY);
        assert!(!accessible(area, true));
        assert!(e.on_write_fault(area));
        assert!(accessible(area, true));
        write(area, 4, 0x55);
        assert_eq!(read(area, 4), 0x55);

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn concurrent_writers_and_take_dirty_on_uncommitted_tracked_pages() {
        let Some((_guard, e)) = tracked() else { return };
        const PAGES: usize = 64;
        const WRITERS: usize = 4;
        let area = e.reserve(0, PAGES * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(PAGES).unwrap();
        e.map_frames(area, &frames).unwrap();

        let barrier = std::sync::Barrier::new(WRITERS + 1);
        std::thread::scope(|scope| {
            for tid in 0..WRITERS {
                let b = &barrier;
                scope.spawn(move || {
                    b.wait();
                    for round in 0..8u8 {
                        for p in 0..PAGES {
                            let idx = (p + tid * 7) % PAGES;
                            write(area + idx * P, tid, round.wrapping_add(idx as u8));
                        }
                    }
                });
            }
            barrier.wait();
            for _ in 0..400 {
                let _ = e.take_dirty(16);
                std::thread::yield_now();
            }
        });

        for p in 0..PAGES {
            for tid in 0..WRITERS {
                let expected = 7u8.wrapping_add(p as u8);
                assert_eq!(read(area + p * P, tid), expected);
            }
        }
        let _ = e.take_dirty(usize::MAX);
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn on_write_fault_and_commit_on_access_cache_committed_state() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        let page = page_of(frames[0]);

        assert!(!e.dirty.lock().unwrap().track.as_ref().unwrap().is_committed(page));
        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.dirty.lock().unwrap().track.as_ref().unwrap().is_committed(page));

        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(e.take_dirty(usize::MAX).is_empty());
        assert!(e.on_write_fault(area));
        assert!(accessible(area, true));

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn prepare_write_committing_many_pages_runs_concurrently_with_take_dirty_and_faults() {
        let Some((_guard, e)) = tracked() else { return };
        const PREPARE_PAGES: usize = 256;
        let prep_area = e.reserve(0, PREPARE_PAGES * P, PAGE_READWRITE).unwrap();
        let prep_frames = e.allocate_pages(PREPARE_PAGES).unwrap();
        e.map_frames(prep_area, &prep_frames).unwrap();

        let fault_area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let fault_frames = e.allocate_pages(2).unwrap();
        e.map_frames(fault_area, &fault_frames).unwrap();
        write(fault_area, 0, 1);
        assert!(e.take_dirty(usize::MAX).contains(&page_of(fault_frames[0])));

        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                e.prepare_write(prep_area, PREPARE_PAGES * P);
            });
            barrier.wait();
            write(fault_area, 1, 0x33);
            let taken = e.take_dirty(usize::MAX);
            assert!(taken.contains(&page_of(fault_frames[0])));
        });

        assert_eq!(read(fault_area, 0), 1);
        assert_eq!(read(fault_area, 1), 0x33);
        let _ = e.take_dirty(usize::MAX);
        e.release(prep_area).unwrap();
        e.free_pages(&prep_frames).unwrap();
        e.release(fault_area).unwrap();
        e.free_pages(&fault_frames).unwrap();
    }

    #[test]
    fn overlapping_begin_and_finish_io_write_protect_immediately_after_last_writer() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        let page = page_of(frames[0]);

        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(!accessible(area, true));

        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                e.begin_io_write(area, P);
            });
            barrier.wait();
            e.begin_io_write(area, P);
        });
        assert!(accessible(area, true));
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(accessible(area, true));

        e.finish_io_write(area, P);
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(accessible(area, true));

        e.finish_io_write(area, P);
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(!accessible(area, true));
        assert!(e.take_dirty(usize::MAX).is_empty());

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn prepare_write_runs_commits_and_opens_multiple_disjoint_runs_in_one_call() {
        let Some(e) = held(|| Engine::load_limited(8, 8)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(6).unwrap();
        e.map_frames(area, &frames).unwrap();

        // Commit and protect page 0 so the batch mixes a protected committed page and uncommitted pages.
        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.take_dirty(usize::MAX).contains(&page_of(frames[0])));
        assert!(!accessible(area, true));
        assert!(!accessible(area + 3 * P, false));
        assert!(!accessible(area + 4 * P, false));

        e.prepare_write_runs(&[(area, P), (area + 3 * P, 2 * P)]);
        assert!(accessible(area, true));
        assert!(accessible(area + 3 * P, true));
        assert!(accessible(area + 4 * P, true));

        let taken = e.take_dirty(usize::MAX);
        assert!(taken.contains(&page_of(frames[0])));
        assert!(taken.contains(&page_of(frames[3])));
        assert!(taken.contains(&page_of(frames[4])));

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn journal_seq_tracks_mapping_events_and_stays_constant_when_idle() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        let s0 = e.journal_seq();
        let frames = e.allocate_pages(1).unwrap();
        assert_eq!(e.journal_seq(), s0, "allocate_pages does not emit a journal event");

        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let s1 = e.journal_seq();
        assert!(s1 > s0);

        e.map_frames(area, &frames).unwrap();
        let s2 = e.journal_seq();
        assert!(s2 > s1);

        e.unmap_pages(area, 1).unwrap();
        let s3 = e.journal_seq();
        assert!(s3 > s2);

        e.release(area).unwrap();
        let s4 = e.journal_seq();
        assert!(s4 > s3);

        e.free_pages(&frames).unwrap();
        assert_eq!(e.journal_seq(), s4, "free_pages does not emit a journal event");
    }

    #[test]
    fn commit_index_runs_commits_contiguous_runs_and_skips_committed_pages() {
        let Some(e) = held(|| Engine::load_limited(8, 8)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 8 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(6).unwrap();
        e.map_frames(area, &frames).unwrap();
        let p0 = page_of(frames[0]);
        let p1 = page_of(frames[1]);
        let p2 = page_of(frames[2]);
        let p4 = page_of(frames[4]);
        let p5 = page_of(frames[5]);

        // Commit p0 via fault path and mark it committed in track.
        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.dirty.lock().unwrap().track.as_ref().unwrap().is_committed(p0));
        assert!(!accessible(area + P, false));
        assert!(!accessible(area + 2 * P, false));
        assert!(!accessible(area + 4 * P, false));
        assert!(!accessible(area + 5 * P, false));

        {
            let st = e.lock();
            let uncom = Engine::commit_uncommitted_runs(&st, &[p0, p1, p2, p4, p5, p2]);
            assert_eq!(uncom, vec![p1, p2, p4, p5, p2]);
        }
        assert!(accessible(area + P, true));
        assert!(accessible(area + 2 * P, true));
        assert!(!accessible(area + 3 * P, false));
        assert!(accessible(area + 4 * P, true));
        assert!(accessible(area + 5 * P, true));

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn convert_pin_to_hold_and_clear_io_hold_runs_protect_on_completion() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        let page = page_of(frames[0]);

        assert!(Engine::commit_on_access_in(&e.state, area));
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(!accessible(area, true));

        e.begin_io_write_runs(&[(area, P)]);
        assert!(accessible(area, true));
        e.convert_pin_to_hold_runs(&[(area, P)]);
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(accessible(area, true));

        e.clear_io_hold_runs(&[(area, P)]);
        assert!(e.take_dirty(usize::MAX).contains(&page));
        assert!(!accessible(area, true));

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn changes_succeeds_while_state_lock_is_held() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let seq = e.journal_seq();
        let (send, recv) = std::sync::mpsc::channel();
        let state_guard = e.lock();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let got = e.changes(0, 64);
                send.send(got.map(|v| v.len())).unwrap();
            });
            let count = recv.recv_timeout(std::time::Duration::from_secs(5)).expect("changes waited for state lock");
            assert!(count >= Some(1));
            assert_eq!(e.journal_seq(), seq);
            drop(state_guard);
        });
        e.release(area).unwrap();
    }

    #[test]
    fn free_pages_resets_unmapped_pages_and_preserves_still_mapped_pages() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(2).unwrap();
        e.map_frames(area, &frames[..1]).unwrap();
        write(area, 0, 0x7A);

        // frames[0] is still mapped, frames[1] is unmapped.
        e.free_pages(&frames).unwrap();
        assert_eq!(read(area, 0), 0x7A, "still-mapped freed page keeps its bytes");

        e.release(area).unwrap();
        let reused = e.allocate_pages(2).unwrap();
        let area2 = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        e.map_frames(area2, &reused).unwrap();
        write(area2, 0, 0x11);
        write(area2 + P, 0, 0x22);
        assert_eq!((read(area2, 0), read(area2 + P, 0)), (0x11, 0x22));
        e.release(area2).unwrap();
        e.free_pages(&reused).unwrap();
    }

    #[test]
    fn take_dirty_empty_fast_path_does_not_lock_dirty() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        assert!(!e.protect_sync.has_pending());
        let _guard = e.dirty.lock().unwrap();
        assert!(e.take_dirty(16).is_empty());
    }

    #[test]
    fn concurrent_take_dirty_io_pin_and_write_faults_keep_pinned_pages_writable() {
        let Some((_guard, e)) = tracked() else { return };
        let area = e.reserve(0, 32 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(32).unwrap();
        e.map_frames(area, &frames).unwrap();
        for k in 0..32 {
            write(area + k * P, 0, k as u8);
        }
        let _ = e.take_dirty(64);

        let e_ref = e;
        std::thread::scope(|s| {
            let io_thread = s.spawn(|| {
                for round in 0..100 {
                    let k = round % 16;
                    let addr = area + k * P;
                    e_ref.begin_io_write(addr, P);
                    assert!(accessible(addr, true), "pinned I/O page became non writable at round {round}");
                    write(addr, 0, (round & 0xff) as u8);
                    e_ref.finish_io_write(addr, P);
                }
            });
            let fault_thread = s.spawn(|| {
                for round in 0..150 {
                    let k = 16 + (round % 16);
                    let addr = area + k * P;
                    write(addr, 0, (round & 0xff) as u8);
                }
            });
            let take_thread = s.spawn(|| {
                for _ in 0..120 {
                    let _ = e_ref.take_dirty(8);
                }
            });
            io_thread.join().unwrap();
            fault_thread.join().unwrap();
            take_thread.join().unwrap();
        });

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn begin_io_write_checked_skips_non_view_addresses_and_pins_views() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        assert!(!e.begin_io_write_checked(0x1000, P));
        assert!(!e.begin_io_write_checked(0, 0));

        let area = e.reserve(0, 2 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(2).unwrap();
        e.map_frames(area, &frames).unwrap();
        assert!(e.begin_io_write_checked(area, P));
        assert!(accessible(area, true));
        e.finish_io_write(area, P);
        assert_eq!(e.take_dirty(10), vec![page_of(frames[0])]);

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn protect_sync_256_stripe_mask_and_32_page_chunk_split() {
        let sync = ProtectSync::default();
        // 40 contiguous pages span 40 stripes and split into 32 + 8 page chunks.
        let base = 0x10_0000usize;
        let addrs: Vec<usize> = (0..40).rev().map(|k| base + k * P).collect();
        let runs = sync.snapshot_runs(addrs);
        assert_eq!(runs.len(), 2);
        assert_eq!((runs[0].0, runs[0].1), (base, 32 * P));
        assert_eq!(ProtectSync::mask_count(runs[0].2), 32);
        assert_eq!(runs[0].3.len(), 32);
        assert_eq!((runs[1].0, runs[1].1), (base + 32 * P, 8 * P));
        assert_eq!(ProtectSync::mask_count(runs[1].2), 8);
        assert_eq!(runs[1].3.len(), 8);
    }

    #[test]
    fn begin_io_write_waits_for_in_flight_protect_rw_on_same_stripe() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        write(area, 0, 1);
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(!accessible(area, true));

        // Simulate another thread (e.g. on_write_fault) having marked the page dirty under `dirty`
        // and not yet finished VirtualProtect(PAGE_READWRITE) on that stripe.
        let chunks = {
            let mut dirty = e.dirty.lock().unwrap();
            let open = dirty.track.as_mut().unwrap().on_fault(page_of(frames[0]), area, Instant::now()).unwrap();
            e.protect_sync.bump_addrs(&open);
            e.protect_sync.snapshot_rw_runs(open)
        };
        let e_ref = &e;
        std::thread::scope(|s| {
            let opener = s.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(25));
                e_ref.protect_sync.protect_rw_runs(chunks, &e_ref.dirty);
            });
            assert!(e_ref.begin_io_write_checked(area, P));
            assert!(accessible(area, true), "begin_io_write returned before in-flight PAGE_READWRITE completed");
            e_ref.finish_io_write(area, P);
            opener.join().unwrap();
        });

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn one_view_per_fault_opens_only_faulting_alias_and_io_opens_siblings() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let a1 = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let a2 = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(a1, &frames).unwrap();
        e.map_frames(a2, &frames).unwrap();
        write(a1, 0, 1);
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(!accessible(a1, true));
        assert!(!accessible(a2, true));

        // Faulting on a1 opens only a1; a2 remains guarded until it faults or I/O prepares it.
        assert!(e.on_write_fault(a1));
        assert!(accessible(a1, true));
        assert!(!accessible(a2, true));

        // Faulting on a2 while page is already dirty still opens a2.
        assert!(e.on_write_fault(a2));
        assert!(accessible(a2, true));

        std::thread::sleep(track::GUARD_REST + std::time::Duration::from_millis(5));
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(!accessible(a1, true));
        assert!(!accessible(a2, true));

        // Fault a1 again, then begin_io_write on a2 opens a2 safely.
        assert!(e.on_write_fault(a1));
        assert!(!accessible(a2, true));
        assert!(e.begin_io_write_checked(a2, P));
        assert!(accessible(a2, true));
        e.finish_io_write(a2, P);

        e.release(a1).unwrap();
        e.release(a2).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn guard_rest_coalesces_writes_and_protects_after_rest_elapses() {
        let Some(e) = held(|| Engine::load_limited(4, 4)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(1).unwrap();
        e.map_frames(area, &frames).unwrap();
        write(area, 0, 1);
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(!accessible(area, true));

        assert!(e.on_write_fault(area));
        assert!(accessible(area, true));
        // Immediate take_dirty keeps the page writable during guard rest.
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(accessible(area, true));

        std::thread::sleep(track::GUARD_REST + std::time::Duration::from_millis(5));
        assert_eq!(e.take_dirty(10).len(), 1);
        assert!(!accessible(area, true));

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    #[test]
    fn protect_runs_checked_batches_failed_run_retries() {
        let Some(e) = held(|| Engine::load_limited(8, 8)) else { return };
        e.set_tracking(true);
        let area = e.reserve(0, 3 * P, PAGE_READWRITE).unwrap();
        let frames = e.allocate_pages(3).unwrap();
        e.map_frames(area, &frames).unwrap();
        for p in 0..3 {
            assert!(Engine::commit_on_access_in(&e.state, area + p * P));
        }
        let _ = e.take_dirty(usize::MAX);
        assert!(!accessible(area, true));
        assert!(!accessible(area + P, true));
        assert!(!accessible(area + 2 * P, true));

        // Open all 3 pages and snapshot two separate runs before bumping their stripe gens.
        assert!(e.on_write_fault(area));
        assert!(e.on_write_fault(area + 2 * P));
        let snapshots = {
            let mut dirty = e.dirty.lock().unwrap_or_else(PoisonError::into_inner);
            let track = dirty.track.as_mut().unwrap();
            let (_, _) = track.take(Instant::now() + track::GUARD_REST + std::time::Duration::from_millis(1), usize::MAX, |_| true);
            dirty.sync.snapshot_runs(vec![area, area + 2 * P])
        };
        // Simulate a concurrent fault bumping stripe gens on both runs, while leaving page 0 dirty and page 2 clean.
        {
            let mut dirty = e.dirty.lock().unwrap_or_else(PoisonError::into_inner);
            let Dirty { track, sync, .. } = &mut *dirty;
            let track = track.as_mut().unwrap();
            let _ = track.on_changed(page_of(frames[0]));
            sync.bump_addrs(&[area, area + 2 * P]);
        }

        let (protected_runs, retries) = e.protect_sync.protect_runs_checked(snapshots, PAGE_READONLY, true, false, &e.dirty);
        assert_eq!(retries, 2);
        assert_eq!(protected_runs, 1);
        assert!(accessible(area, true), "dirty page 0 must stay writable");
        assert!(!accessible(area + 2 * P, true), "clean page 2 must be protected on batched retry");

        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }

    /// The bytes that writer `w` writes in round `round`: tells writers, rounds and pages apart.
    fn stamp(w: usize, round: usize, len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 13 + w * 31 + round * 7 + i / PAGE_BYTES) as u8).collect()
    }

    /// Several threads write pool pages (no lock held while they copy) while others take dirty pages, map and
    /// unmap a view of some of those pages, and a kernel write fills a tracked page. No thread sees an
    /// error, and the last round of each writer is in its pages whole.
    #[test]
    fn concurrent_pool_writes_with_take_dirty_map_unmap_and_io_lose_no_bytes() {
        let Some((_guard, e)) = tracked() else { return };
        const WRITERS: usize = 6;
        const PAGES: usize = 4;
        const ROUNDS: usize = 60;
        let frames: Vec<Vec<u32>> = (0..WRITERS).map(|_| e.allocate_pages(PAGES).unwrap()).collect();
        let watch = e.reserve(0, PAGES * P, PAGE_READWRITE).unwrap();
        let io_area = e.reserve(0, P, PAGE_READWRITE).unwrap();
        let io_frames = e.allocate_pages(1).unwrap();
        e.map_frames(io_area, &io_frames).unwrap();
        let path = std::env::temp_dir().join(format!("xmem_lockless_{}.bin", std::process::id()));
        std::fs::write(&path, [0x44u8; 512]).unwrap();
        let length = |round: usize| PAGES * P - 300 - (round % 5) * 17;
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writers: Vec<_> = (0..WRITERS)
                .map(|w| {
                    let (frames, stop) = (&frames[w], &stop);
                    s.spawn(move || {
                        for round in 0..ROUNDS {
                            let data = stamp(w, round, length(round));
                            let seen = e.write_frames(frames, 300, &data).unwrap();
                            if w == 0 {
                                assert!(seen.is_empty() || seen == [(watch + 300, data.len())], "the view of writer 0 or none: {seen:?}");
                            } else {
                                assert!(seen.is_empty(), "no view shows the pages of writer {w}");
                            }
                        }
                        let _ = stop;
                    })
                })
                .collect();
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    e.take_dirty(1 << 20);
                    std::thread::yield_now();
                }
            });
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    e.map_frames(watch, &frames[0]).unwrap();
                    let mut sum = 0u8;
                    for k in 0..PAGES {
                        sum = sum.wrapping_add(read(watch, k * P + 5));
                    }
                    std::hint::black_box(sum);
                    e.unmap_pages(watch, PAGES).unwrap();
                }
            });
            s.spawn(|| {
                while !stop.load(Ordering::Acquire) {
                    e.begin_io_write(io_area, 512);
                    let mut file = std::fs::File::open(&path).unwrap();
                    // SAFETY: the page is mapped read write for 64 KiB, and `begin_io_write` opened it for the kernel.
                    let target = unsafe { std::slice::from_raw_parts_mut(io_area as *mut u8, 512) };
                    let read = std::io::Read::read(&mut file, target);
                    e.finish_io_write(io_area, 512);
                    assert_eq!(read.expect("no ERROR_NOACCESS for a pinned page").min(512), 512);
                }
            });
            for writer in writers {
                writer.join().unwrap();
            }
            stop.store(true, Ordering::Release);
        });
        for (w, pages) in frames.iter().enumerate() {
            e.map_frames(watch, pages).unwrap();
            let want = stamp(w, ROUNDS - 1, length(ROUNDS - 1));
            // SAFETY: the pages are mapped read write.
            let seen = unsafe { std::slice::from_raw_parts((watch + 300) as *const u8, want.len()) };
            assert!(seen == &want[..], "the last round of writer {w} is whole");
            e.unmap_pages(watch, PAGES).unwrap();
        }
        std::fs::remove_file(&path).unwrap();
        e.unmap_pages(io_area, 1).unwrap();
        e.release(io_area).unwrap();
        e.release(watch).unwrap();
        for pages in &frames {
            e.free_pages(pages).unwrap();
        }
        e.free_pages(&io_frames).unwrap();
    }

    /// A page that a write is copying into stays out of the pool until the copy ends.
    #[test]
    fn a_pinned_page_is_not_freed_until_its_copy_ends() {
        let Some(held) = engine() else { return };
        let e: &Engine = &held;
        let frames = e.allocate_pages(2).unwrap();
        let pin = {
            let st = e.lock();
            let index = index_of(frames[1]).unwrap();
            let section = Engine::section_of(&st, index).unwrap();
            &raw const section.pins[index - section.start] as usize
        };
        // SAFETY: the pin is in a section of `e`, which lives as long as the test.
        let pin = unsafe { &*(pin as *const AtomicU32) };
        pin.fetch_add(1, Ordering::AcqRel);
        let freed = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                e.free_pages(&frames).unwrap();
                freed.store(true, Ordering::Release);
            });
            std::thread::sleep(std::time::Duration::from_millis(150));
            assert!(!freed.load(Ordering::Acquire), "the free waits for the copy");
            pin.fetch_sub(1, Ordering::Release);
        });
        assert!(freed.load(Ordering::Acquire));
        assert!(matches!(e.write_frames(&frames, 0, &[1]), Err(EngineError::Pages(PageError::NotAllocated(_)))));
    }

    /// A write of more pages than the inline buffers hold goes through the heap fallback and is as exact as a small one.
    #[test]
    fn a_large_pool_write_spills_to_the_heap_and_a_small_one_stays_inline() {
        let Some(e) = engine() else { return };
        let n = INLINE_WRITE_PAGES + 8;
        let frames = e.allocate_pages(n).unwrap();
        let area = e.reserve(0, n * P, PAGE_READWRITE).unwrap();
        e.map_frames(area, &frames).unwrap();
        let data = pattern(n * P - 16);
        let seen = e.write_frames_buf(&frames, 16, &data).unwrap();
        assert_eq!(seen.as_slice(), [(area + 16, data.len())], "n page runs joined into one");
        // SAFETY: the pages are mapped read write.
        let mapped = unsafe { std::slice::from_raw_parts((area + 16) as *const u8, data.len()) };
        assert!(mapped == &data[..]);
        let small = e.write_frames_buf(&frames[..2], 0, &data[..P + 8]).unwrap();
        assert!(small.is_inline() && small.as_slice() == [(area, P + 8)]);
        e.unmap_pages(area, n).unwrap();
        e.release(area).unwrap();
        e.free_pages(&frames).unwrap();
    }
}
