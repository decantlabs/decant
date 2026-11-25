//! The exports of an `xmem.dll` as the test programs call them: typed, with the raw pointer work
//! kept here. Shared by `loadtest` and `xmem_abitest` (included with `#[path]`).
#![allow(unsafe_code)]
#![allow(dead_code)]

use std::ffi::c_void;

use decant_testkit::Library;
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::core::BOOL;

pub type AllocFn = unsafe extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeFn = unsafe extern "system" fn(*mut c_void, u64);
type QueryFn = unsafe extern "system" fn(HANDLE, *const c_void, *mut c_void) -> BOOL;
type StatsFn = unsafe extern "system" fn(u32, *mut c_void) -> BOOL;
type CountersFn = unsafe extern "system" fn(*mut u64, *mut u64, *mut u64);
type HooksFn = unsafe extern "system" fn(Option<AllocFn>, Option<FreeFn>);
type VirtualAllocFn = unsafe extern "system" fn(*mut c_void, usize, u32, u32) -> *mut c_void;
type PhysicalFn = unsafe extern "system" fn(usize, usize, usize, usize) -> usize;
type ToolsMemFn = unsafe extern "system" fn() -> BOOL;
type PhysAllocFn = unsafe extern "system" fn(u64, *mut usize, *mut u64) -> BOOL;
type PhysMapFn = unsafe extern "system" fn(*mut c_void, usize, *const u64) -> *mut c_void;
type PhysScatterFn = unsafe extern "system" fn(*const *mut c_void, usize, *const u64) -> BOOL;
type PhysFreeFn = unsafe extern "system" fn(usize, *const u64) -> BOOL;
type GpuTrackFn = unsafe extern "system" fn(BOOL);
type GpuTakeFn = unsafe extern "system" fn(*mut u32, u32) -> u32;
type GpuViewFn = unsafe extern "system" fn(u32, *mut usize, *mut u32, *mut u32) -> BOOL;
type PrepareFn = unsafe extern "system" fn(usize, usize);
type GpuWritableFn = unsafe extern "system" fn(usize, usize) -> BOOL;

/// The four physical page exports, as described in `docs/CONSOLE_LIBS.md` (section 5).
struct Physical {
    allocate: PhysAllocFn,
    map: PhysMapFn,
    scatter: PhysScatterFn,
    free: PhysFreeFn,
}

/// The entry points whose signature is not known (they log four raw arguments and fail).
pub const UNKNOWN_SIGNATURES: [&str; 1] = ["XMemTransferMemory"];

/// `GetCurrentProcess()`, the pseudo handle.
pub const CURRENT_PROCESS: HANDLE = -1isize as HANDLE;

pub const GRAPHICS: u64 = 1 << 29;
pub const PAGE_64K: u64 = 1 << 14;

pub const fn align(log2: u64) -> u64 {
    log2 << 24
}

/// The exports a test may look up. The four console functions are required.
pub struct XMem {
    alloc: AllocFn,
    free: FreeFn,
    query: QueryFn,
    stats: StatsFn,
    counters: Option<CountersFn>,
    hooks: Option<HooksFn>,
    pub default_alloc: Option<AllocFn>,
    pub default_free: Option<FreeFn>,
    virtual_alloc: Option<VirtualAllocFn>,
    unknown: [Option<PhysicalFn>; 1],
    tools_mem: Option<ToolsMemFn>,
    physical: Option<Physical>,
    gpu_track: Option<GpuTrackFn>,
    gpu_take: Option<GpuTakeFn>,
    gpu_view: Option<GpuViewFn>,
    prepare: Option<PrepareFn>,
    gpu_writable: Option<GpuWritableFn>,
}

/// Whether each export exists: `XMemAlloc`, `XMemFree`, `XMemVirtualQuery`,
/// `XMemGetWorkingSetStatistics`, `DecantXMemCounters`.
pub fn present(lib: &Library) -> [bool; 5] {
    ["XMemAlloc", "XMemFree", "XMemVirtualQuery", "XMemGetWorkingSetStatistics", "DecantXMemCounters"].map(|n| !lib.address(n).is_null())
}

impl XMem {
    pub fn load(lib: &Library) -> Option<XMem> {
        // SAFETY: each type is the signature the C header of xmem.dll declares for that name.
        unsafe {
            Some(XMem {
                alloc: lib.get("XMemAlloc")?,
                free: lib.get("XMemFree")?,
                query: lib.get("XMemVirtualQuery")?,
                stats: lib.get("XMemGetWorkingSetStatistics")?,
                counters: lib.get("DecantXMemCounters"),
                hooks: lib.get("XMemSetAllocationHooks"),
                default_alloc: lib.get("XMemAllocDefault"),
                default_free: lib.get("XMemFreeDefault"),
                virtual_alloc: lib.get("XMemVirtualAlloc"),
                unknown: UNKNOWN_SIGNATURES.map(|name| lib.get(name)),
                gpu_track: lib.get("DecantXmemGpuTrack"),
                gpu_take: lib.get("DecantXmemGpuTakeDirty"),
                gpu_view: lib.get("DecantXmemGpuSectionView"),
                prepare: lib.get("DecantXmemPrepareWrite"),
                gpu_writable: lib.get("DecantXmemGpuWritable"),
                tools_mem: lib.get("XMemIsToolsMemAvailable"),
                physical: (|| {
                    Some(Physical {
                        allocate: lib.get("XMemAllocatePhysicalPages")?,
                        map: lib.get("XMemMapPhysicalPages")?,
                        scatter: lib.get("XMemMapPhysicalPagesScatter")?,
                        free: lib.get("XMemFreePhysicalPages")?,
                    })
                })(),
            })
        }
    }

    pub fn alloc(&self, size: usize, attributes: u64) -> *mut u8 {
        // SAFETY: the allocator accepts any size and attributes and returns null when it cannot serve them.
        unsafe { (self.alloc)(size, attributes).cast() }
    }

    /// Frees any pointer: xmem ignores the ones it did not allocate.
    pub fn free(&self, address: *mut u8, attributes: u64) {
        // SAFETY: XMemFree accepts null, foreign and already freed pointers by contract.
        unsafe { (self.free)(address.cast(), attributes) }
    }

    /// `XMemVirtualQuery(process, address, buffer)`. The buffer is at least 64 bytes.
    pub fn query(&self, process: HANDLE, address: *const c_void, buffer: &mut [u8]) -> bool {
        assert!(buffer.len() >= 64);
        // SAFETY: `buffer` has the 64 writable bytes the function fills. The address is only queried.
        unsafe { (self.query)(process, address, buffer.as_mut_ptr().cast()) != 0 }
    }

    /// `XMemGetWorkingSetStatistics(selector, buffer)`. The function writes 32 bytes; callers pass 128.
    pub fn stats(&self, selector: u32, buffer: &mut [u8]) -> bool {
        assert!(buffer.len() >= 128);
        // SAFETY: `buffer` has the 128 writable bytes the function fills.
        unsafe { (self.stats)(selector, buffer.as_mut_ptr().cast()) != 0 }
    }

    pub fn has_virtual_alloc(&self) -> bool {
        self.virtual_alloc.is_some()
    }

    /// `XMemVirtualAlloc(address, size, type, protect)`, null when the export is missing.
    pub fn virtual_alloc(&self, address: *mut c_void, size: usize, allocation_type: u32, protect: u32) -> *mut u8 {
        let Some(f) = self.virtual_alloc else {
            return std::ptr::null_mut();
        };
        // SAFETY: same contract as VirtualAlloc, which validates its arguments and answers null on failure.
        unsafe { f(address, size, allocation_type, protect).cast() }
    }

    /// Read the calling thread's Win32 error after an export fails.
    pub fn last_error(&self) -> u32 {
        // SAFETY: GetLastError reads only this thread's error code.
        unsafe { windows_sys::Win32::Foundation::GetLastError() }
    }

    /// Whether the entry point `index` of `UNKNOWN_SIGNATURES` is exported.
    pub fn has_unknown(&self, index: usize) -> bool {
        self.unknown[index].is_some()
    }

    /// Calls the entry point `index` of `UNKNOWN_SIGNATURES` with raw arguments: its result and the
    /// last error it left. They only log and fail, so the arguments are never used as pointers.
    pub fn call_unknown(&self, index: usize, args: [usize; 4]) -> Option<(usize, u32)> {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        let f = self.unknown[index]?;
        // SAFETY: SetLastError only stores the thread's error code, and the entry points ignore
        // their arguments beyond logging them.
        let result = unsafe {
            SetLastError(0);
            f(args[0], args[1], args[2], args[3])
        };
        // SAFETY: GetLastError has no precondition.
        Some((result, unsafe { GetLastError() }))
    }

    /// `XMemIsToolsMemAvailable()`, `None` when the export is missing.
    pub fn tools_mem_available(&self) -> Option<bool> {
        let f = self.tools_mem?;
        // SAFETY: the function takes no argument and only logs.
        Some(unsafe { f() } != 0)
    }

    /// Whether the exports `d3d12_xs.dll` uses to follow the pages are there.
    pub fn has_gpu_exports(&self) -> bool {
        self.gpu_track.is_some() && self.gpu_take.is_some() && self.gpu_view.is_some() && self.prepare.is_some() && self.gpu_writable.is_some()
    }

    /// `DecantXmemGpuTrack(on)`.
    pub fn gpu_track(&self, on: bool) {
        if let Some(f) = self.gpu_track {
            // SAFETY: the function takes a flag.
            unsafe { f(BOOL::from(on)) }
        }
    }

    /// `DecantXmemGpuTakeDirty`: the pool pages whose mirror is stale, at most `cap`.
    pub fn gpu_take_dirty(&self, cap: u32) -> Vec<u32> {
        let Some(f) = self.gpu_take else { return Vec::new() };
        let mut out = vec![0u32; cap as usize];
        // SAFETY: `out` has room for `cap` entries.
        let n = unsafe { f(out.as_mut_ptr(), cap) };
        out.truncate(n as usize);
        out
    }

    /// `DecantXmemGpuSectionView(index)`: the address of the read only view, the first pool page and the pages.
    pub fn gpu_section_view(&self, index: u32) -> Option<(usize, u32, u32)> {
        let (mut base, mut first, mut pages) = (0usize, 0u32, 0u32);
        // SAFETY: the three outputs are valid locations.
        let ok = unsafe { self.gpu_view?(index, &mut base, &mut first, &mut pages) };
        (ok != 0).then_some((base, first, pages))
    }

    /// `DecantXmemPrepareWrite(address, len)`.
    pub fn prepare_write(&self, address: *const u8, len: usize) {
        if let Some(f) = self.prepare {
            // SAFETY: the range is only looked up.
            unsafe { f(address as usize, len) }
        }
    }

    /// `DecantXmemGpuWritable(address, len)`.
    pub fn gpu_writable(&self, address: *const u8, len: usize) -> bool {
        // SAFETY: the range is only looked up.
        self.gpu_writable.is_some_and(|f| unsafe { f(address as usize, len) } != 0)
    }

    pub fn has_physical(&self) -> bool {
        self.physical.is_some()
    }

    /// `XMemAllocatePhysicalPages(flags, &count, frames)` with room for `count` frames: the result,
    /// the count it leaves, the frames and the last error.
    pub fn allocate_pages(&self, flags: u64, count: usize) -> (bool, usize, Vec<u64>, u32) {
        let mut n = count;
        let mut frames = vec![0u64; count];
        let (ok, error) = self.call(|p| {
            // SAFETY: `n` is a valid count and `frames` has room for `count` entries.
            unsafe { (p.allocate)(flags, &mut n, frames.as_mut_ptr()) }
        });
        (ok != 0, n, frames, error)
    }

    /// `XMemAllocatePhysicalPages` with a null count pointer.
    pub fn allocate_pages_null_count(&self, flags: u64) -> (bool, u32) {
        // SAFETY: the export refuses a null count before it reads anything.
        let (ok, error) = self.call(|p| unsafe { (p.allocate)(flags, std::ptr::null_mut(), std::ptr::null_mut()) });
        (ok != 0, error)
    }

    /// `XMemAllocatePhysicalPages` with a count and a null array: the result, the count left and the error.
    pub fn allocate_pages_null_array(&self, flags: u64, count: usize) -> (bool, usize, u32) {
        let mut n = count;
        // SAFETY: `n` is a valid count, and the export refuses the null array before it writes.
        let (ok, error) = self.call(|p| unsafe { (p.allocate)(flags, &mut n, std::ptr::null_mut()) });
        (ok != 0, n, error)
    }

    /// `XMemMapPhysicalPages(address, count, frames)`, `frames` null for `None`: the address it
    /// returns and the last error.
    pub fn map_pages(&self, address: *mut u8, count: usize, frames: Option<&[u64]>) -> (*mut u8, u32) {
        if let Some(frames) = frames {
            assert!(frames.len() >= count || count == usize::MAX);
        }
        // SAFETY: `frames` is readable for `count` entries (asserted), or null. The export validates the address.
        let (result, error) = self.call(|p| unsafe { (p.map)(address.cast(), count, frames.map_or(std::ptr::null(), <[u64]>::as_ptr)) as usize });
        (result as *mut u8, error)
    }

    /// `XMemMapPhysicalPagesScatter(addresses, count, frames)`, either pointer null for `None`.
    pub fn map_scatter(&self, addresses: Option<&[*mut u8]>, count: usize, frames: Option<&[u64]>) -> (bool, u32) {
        let raw: Option<Vec<*mut c_void>> = addresses.map(|a| a.iter().map(|&p| p.cast()).collect());
        // SAFETY: the arrays are readable for the count the caller passes, which the tests keep within their length.
        let (ok, error) = self.call(|p| unsafe { (p.scatter)(raw.as_ref().map_or(std::ptr::null(), |r| r.as_ptr()), count, frames.map_or(std::ptr::null(), <[u64]>::as_ptr)) });
        (ok != 0, error)
    }

    /// `XMemFreePhysicalPages(count, frames)`, `frames` null for `None`.
    pub fn free_pages(&self, count: usize, frames: Option<&[u64]>) -> (bool, u32) {
        // SAFETY: `frames` is readable for `count` entries, which the tests keep within its length, or null.
        let (ok, error) = self.call(|p| unsafe { (p.free)(count, frames.map_or(std::ptr::null(), <[u64]>::as_ptr)) });
        (ok != 0, error)
    }

    /// Runs `f` with the physical exports and returns what it returned and the last error.
    fn call<T: Default>(&self, f: impl FnOnce(&Physical) -> T) -> (T, u32) {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        let Some(physical) = self.physical.as_ref() else {
            return (T::default(), 0);
        };
        // SAFETY: SetLastError and GetLastError only store and read the thread's error code.
        unsafe { SetLastError(0) };
        let result = f(physical);
        (result, unsafe { GetLastError() })
    }

    pub fn has_counters(&self) -> bool {
        self.counters.is_some()
    }

    /// (bytes in use, peak, allocation count).
    pub fn counters(&self) -> (u64, u64, u64) {
        let (mut current, mut peak, mut count) = (0, 0, 0);
        if let Some(counters) = self.counters {
            // SAFETY: three valid output locations.
            unsafe { counters(&mut current, &mut peak, &mut count) };
        }
        (current, peak, count)
    }

    pub fn has_hooks(&self) -> bool {
        self.hooks.is_some()
    }

    pub fn set_hooks(&self, alloc: Option<AllocFn>, free: Option<FreeFn>) {
        if let Some(hooks) = self.hooks {
            // SAFETY: the hooks passed are functions of this program with the allocator signatures.
            unsafe { hooks(alloc, free) };
        }
    }
}

/// Writes one byte at `block + offset`. An access violation here is a failure of the allocator.
/// Reads up to `len` bytes of the file at `path` into the memory at `block` with the program's own
/// imported `ReadFile` (the standard library reads with `NtReadFile`), and returns how many came.
/// `None` when the file cannot be opened or read.
pub fn read_file_into(path: &std::path::Path, block: *mut u8, len: usize) -> Option<usize> {
    use std::os::windows::io::AsRawHandle;
    let file = std::fs::File::open(path).ok()?;
    let mut read = 0u32;
    // SAFETY: the handle is the open file, the caller guarantees `len` writable bytes at `block`.
    let ok = unsafe { windows_sys::Win32::Storage::FileSystem::ReadFile(file.as_raw_handle(), block.cast(), len as u32, &mut read, std::ptr::null_mut()) };
    (ok != 0).then_some(read as usize)
}

pub fn poke(block: *mut u8, offset: usize, value: u8) {
    // SAFETY: callers pass a non null block returned by XMemAlloc for at least offset + 1 bytes.
    unsafe { block.add(offset).write_volatile(value) };
}

/// Fills `len` bytes of a block.
pub fn fill(block: *mut u8, value: u8, len: usize) {
    // SAFETY: callers pass a non null block returned by XMemAlloc for at least `len` bytes.
    unsafe { block.write_bytes(value, len) };
}

/// Calls a default allocator export, for the counting hooks.
pub fn call_alloc(f: AllocFn, size: usize, attributes: u64) -> *mut c_void {
    // SAFETY: `f` is XMemAllocDefault, which accepts any size and attributes.
    unsafe { f(size, attributes) }
}

/// Calls a default free export, for the counting hooks.
pub fn call_free(f: FreeFn, address: *mut c_void, attributes: u64) {
    // SAFETY: `f` is XMemFreeDefault, which accepts any pointer.
    unsafe { f(address, attributes) }
}

/// A committed page from `VirtualAlloc`, with no xmem header in front of it.
pub fn foreign_page() -> *mut u8 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    // SAFETY: plain allocation call.
    unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast() }
}

pub fn release_foreign_page(page: *mut u8) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    // SAFETY: `page` came from `foreign_page`.
    unsafe { VirtualFree(page.cast(), 0, MEM_RELEASE) };
}

/// Reads one byte at `block + offset`, which must be mapped.
pub fn peek(block: *mut u8, offset: usize) -> u8 {
    // SAFETY: callers pass an address they mapped for at least offset + 1 bytes.
    unsafe { block.add(offset).read_volatile() }
}

/// Releases pages `XMemVirtualAlloc` reserved.
pub fn release_pages(page: *mut u8) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    // SAFETY: `page` is the base address of a reservation made by VirtualAlloc.
    unsafe { VirtualFree(page.cast(), 0, MEM_RELEASE) };
}

/// The protection `VirtualQuery` reports for the page at `address`: 0 when it is only reserved or not mapped.
/// A pool page in a view that the title has not touched yet is reserved until its first access commits
/// it: it reports the protection of its view.
pub fn protection_of(address: *const u8) -> u32 {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data structure.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid output buffer of the size passed.
    let n = unsafe { VirtualQuery(address.cast(), &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    if n == 0 {
        0
    } else if info.State == MEM_RESERVE && info.Type == MEM_MAPPED {
        info.AllocationProtect
    } else {
        info.Protect
    }
}

/// (PrivateUsage, WorkingSetSize, PagefileUsage) of this process.
pub fn host_counters() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
    // SAFETY: plain data structure.
    let mut pmc: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    pmc.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: `pmc` is a valid output buffer of the size passed.
    let ok = unsafe { K32GetProcessMemoryInfo(CURRENT_PROCESS, (&raw mut pmc).cast::<PROCESS_MEMORY_COUNTERS>(), pmc.cb) };
    (ok != 0).then_some((pmc.PrivateUsage as u64, pmc.WorkingSetSize as u64, pmc.PagefileUsage as u64))
}
