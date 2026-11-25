//! Title memory seen by the PC GPU. On the console the GPU address of vertex, index and constant
//! data is its CPU address. Here those bytes sit in pages that `xmem.dll` maps with
//! `XMemMapPhysicalPages`, and no PC resource covers them. This module mirrors them for the GPU:
//!
//! * The pool is a set of D3D12 heaps of 64 MiB (custom, system memory, write back, made with
//!   `CreateHeap`), one buffer placed over each and mapped for the life of the process. A pool page of
//!   `xmem.dll` gets the next mirror tile when it is first copied (it has bytes: `xmem.dll` reports only
//!   committed pages, and a page is committed by its first access), when a buffer is placed over it, or
//!   when it lies in the first `ADDRESS_REACH` bytes of a span the GPU is given (`title_mirror`). Until
//!   then its window tiles show nothing and read as zeros, as the page does. The heaps are made as the
//!   tiles fill them. The mirror is written with a plain memcpy, and the GPU reads it over PCIe with no
//!   copy command.
//! * A window of reserved buffers (`DECANT_D3D12X_TITLE_WINDOW_GIB` GiB each, 64 by default) shows the
//!   pool at title addresses. A reservation of the title gets a slot of its own size in the window
//!   when its first page is mapped, and the window tiles of a slot are mapped to the heap tiles of the
//!   pages the title maps there (`UpdateTileMappings` on a private queue). A title address is the slot
//!   address plus its distance from the reservation base, so a span is contiguous on the GPU over
//!   gaps, aliases and scattered pages (`title_mirror` has the arithmetic).
//! * The title's commits with a console page bit (the title's arenas,`XMemVirtualAlloc` 0x20003000) are pool
//!   pages too, but their pages come on demand: `translate` asks `xmem.dll` for the pages of a span the
//!   GPU is given (`DecantXmemGpuUse`, at least `ADDRESS_REACH` bytes), and only those are mapped, copied and
//!   tracked. Their slots share window buffers of their own, packed for the 4 GiB a translated shader
//!   reaches. `lookup` answers without bringing pages in.
//! * `xmem.dll` says what changed (a journal of reservations and views) and which pool pages were
//!   written since the last time (page protection, see its `track` module). `sync` runs before every
//!   submission of the title: it applies the changes, copies the dirty pages from the read only view of
//!   their section into the heap, and makes the submitting queue wait for the tile mappings.
//!   Batches of at least 16 MiB use scoped copying threads sized to host parallelism and batch size.
//!   Each job covers at most 4 MiB within one section and heap. Threads join before tile mappings.
//!
//! Nothing here keeps a section committed: no heap is opened on memory of the title, so the commit
//! comes back when the process ends, killed or not. The two DLLs only meet through exports of
//! `xmem.dll`, looked up by name when first needed.
//!
//! What this cannot do yet: GPU writes into title memory do not reach the CPU side (the writes the
//! CPU does after the GPU, `cpu_writes::after`, do). Textures are not covered, console textures are
//! tiled (`detile` reads the tiled bytes through the window).
//!
//! Off by default: `DECANT_D3D12X_TITLE_MEM=1` turns the module on (the page protection changes how the
//! whole title runs, see `enabled`). When the exports are missing or a device object cannot be made,
//! that is logged once and everything stays as it was without the module.

#![allow(unsafe_code)]
// `d3d12xs_test` includes this file and uses only part of it.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};

use crate::log::xs_log;
use crate::title_mirror::{Change, HEAP_PAGES, Mirror, PAGE, TileCall, heap_of};

/// A span within this many bytes of the end of its reservation is logged: a root view has no bounds
/// check, so a shader that reads past its data would read past the slot in the window.
pub const NEAR_END_BYTES: u64 = 64 << 10;

/// The size of a pool heap: 1024 pages.
pub const POOL_HEAP_BYTES: u64 = HEAP_PAGES as u64 * PAGE;

/// Pages asked of `DecantXmemGpuTakeDirty` in one call.
const TAKE_PAGES: usize = 65536;
/// Bound each copy job to 1 MiB and assign at least 1 MiB per copying thread.
const COPY_CHUNK_PAGES: usize = 16;
const COPY_THREAD_PAGES: usize = 16;

#[derive(Clone, Copy, Debug)]
struct CopyJob {
    source: usize,
    destination: usize,
    pages: usize,
}

fn plan_copy(jobs: &mut Vec<CopyJob>, source: usize, destination: usize, pages: usize) {
    for start in (0..pages).step_by(COPY_CHUNK_PAGES) {
        jobs.push(CopyJob {
            source: source + start * PAGE as usize,
            destination: destination + start * PAGE as usize,
            pages: (pages - start).min(COPY_CHUNK_PAGES),
        });
    }
}

fn copy_threads(pages: usize, cores: usize) -> usize {
    (pages / COPY_THREAD_PAGES).max(1).min(cores.max(1))
}

struct PoolState {
    jobs_ptr: usize,
    jobs_len: usize,
    workers_to_take: usize,
    active_workers: usize,
    epoch: u64,
}

struct PoolShared {
    state: Mutex<PoolState>,
    next: std::sync::atomic::AtomicUsize,
    cv_work: Condvar,
    cv_done: Condvar,
}

struct CopyPool {
    workers: usize,
    run_lock: Mutex<()>,
    shared: Arc<PoolShared>,
}

impl CopyPool {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState { jobs_ptr: 0, jobs_len: 0, workers_to_take: 0, active_workers: 0, epoch: 0 }),
            next: std::sync::atomic::AtomicUsize::new(0),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        for i in 0..workers {
            let shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new().name(format!("title-mem-copy-{i}")).spawn(move || {
                let mut seen_gen = 0u64;
                loop {
                    let (ptr, len) = {
                        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        while st.epoch == seen_gen || st.workers_to_take == 0 {
                            st = shared.cv_work.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        seen_gen = st.epoch;
                        st.workers_to_take -= 1;
                        (st.jobs_ptr as *const CopyJob, st.jobs_len)
                    };
                    loop {
                        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
                        if idx >= len {
                            break;
                        }
                        // SAFETY: `run` keeps the `jobs` slice alive until `active_workers` reaches 0.
                        let job = unsafe { *ptr.add(idx) };
                        // SAFETY: caller guarantees live disjoint source and destination ranges.
                        unsafe { std::ptr::copy_nonoverlapping(job.source as *const u8, job.destination as *mut u8, job.pages * PAGE as usize) };
                    }
                    let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    st.active_workers -= 1;
                    if st.active_workers == 0 {
                        shared.cv_done.notify_one();
                    }
                }
            });
        }
        Self { workers, run_lock: Mutex::new(()), shared }
    }

    /// # Safety
    /// Same safety contract as `execute_copies`.
    unsafe fn run(&self, jobs: &[CopyJob], threads: usize) {
        let _run_guard = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let worker_count = threads.saturating_sub(1).min(self.workers);
        if worker_count == 0 || jobs.len() <= 1 {
            for job in jobs {
                // SAFETY: caller guarantees live disjoint ranges for every job.
                unsafe { std::ptr::copy_nonoverlapping(job.source as *const u8, job.destination as *mut u8, job.pages * PAGE as usize) };
            }
            return;
        }
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.jobs_ptr = jobs.as_ptr() as usize;
            st.jobs_len = jobs.len();
            self.shared.next.store(0, Ordering::Relaxed);
            st.workers_to_take = worker_count;
            st.active_workers = worker_count;
            st.epoch += 1;
            self.shared.cv_work.notify_all();
        }
        loop {
            let idx = self.shared.next.fetch_add(1, Ordering::Relaxed);
            if idx >= jobs.len() {
                break;
            }
            let job = jobs[idx];
            // SAFETY: caller guarantees live disjoint ranges for every job.
            unsafe { std::ptr::copy_nonoverlapping(job.source as *const u8, job.destination as *mut u8, job.pages * PAGE as usize) };
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.active_workers > 0 {
            st = self.shared.cv_done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn copy_pool(cores: usize) -> &'static CopyPool {
    static POOL: OnceLock<CopyPool> = OnceLock::new();
    POOL.get_or_init(|| CopyPool::new((cores / 2).clamp(2, 8)))
}

/// # Safety
/// Sources must be readable and destinations writable until this function returns.
/// All destination ranges must be disjoint and must not overlap any source.
unsafe fn execute_copies(jobs: &[CopyJob], cores: usize) {
    let pages = jobs.iter().map(|job| job.pages).sum();
    let threads = copy_threads(pages, cores).min(jobs.len());
    if threads <= 1 {
        for job in jobs {
            // SAFETY: the caller guarantees live disjoint ranges for every job.
            unsafe { std::ptr::copy_nonoverlapping(job.source as *const u8, job.destination as *mut u8, job.pages * PAGE as usize) };
        }
        return;
    }
    // SAFETY: caller guarantees live disjoint ranges for every job.
    unsafe { copy_pool(cores).run(jobs, threads) };
}

/// Journal records read in one call.
const CHANGE_RECORDS: usize = 4096;
/// How often the modules loaded since are looked at for file reads (`DecantXmemGpuGuardModules`).
const GUARD_EVERY: Duration = Duration::from_secs(5);
/// Syncs between two stats lines.
const STATS_EVERY: u64 = 100;
/// Mirror tiles read by the zero-page sampler on each tick: 256 MiB of title memory.
const ZERO_SCAN_TILES: usize = 4096;
/// Identifies a device's state while a sampler chunk is read outside the state lock.
static NEXT_STATE: AtomicU64 = AtomicU64::new(1);
/// Incremented whenever virtual to mirror tile mappings change.
static MAPPING_GEN: AtomicU64 = AtomicU64::new(1);

fn bump_mapping_gen() {
    MAPPING_GEN.fetch_add(1, Ordering::Release);
}

/// Current generation of virtual to mirror tile mappings.
pub fn mapping_generation() -> u64 {
    MAPPING_GEN.load(Ordering::Acquire)
}

/// One complete rotating pass over the mirror tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZeroSweep {
    pub tiles: u32,
    pub zero: u64,
    pub non_zero: u64,
    pub unreadable: u64,
}

/// The latest rotating observations of the title pages behind the mirror tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZeroPageSample {
    pub tiles: u32,
    pub scanned: u32,
    pub scan_tiles: usize,
    pub zero: u64,
    pub non_zero: u64,
    pub unreadable: u64,
    pub unknown: u64,
    pub last_sweep: Option<ZeroSweep>,
}

/// What the mirror holds and what the sampler could read of its title pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MirrorSample {
    pub heaps: usize,
    pub tiles: u32,
    pub windows: usize,
    pub zero_pages: ZeroPageSample,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PageZero {
    #[default]
    Unknown,
    Zero,
    NonZero,
    Unreadable,
}

/// Latest result per mirror tile and the counts kept while the scan rotates.
#[derive(Debug, Default)]
struct ZeroScan {
    next: usize,
    latest: Vec<PageZero>,
    zero: u64,
    non_zero: u64,
    unreadable: u64,
    unknown: u64,
    /// Results seen since the last complete sweep, so growth cannot make a partial pass look complete.
    sweep_seen: Vec<bool>,
    sweep_known: usize,
    sweep_zero: u64,
    sweep_non_zero: u64,
    sweep_unreadable: u64,
    last_sweep: Option<ZeroSweep>,
}

impl ZeroScan {
    fn resize(&mut self, tiles: u32) {
        let tiles = tiles as usize;
        if tiles > self.latest.len() {
            self.unknown += (tiles - self.latest.len()) as u64;
            self.latest.resize(tiles, PageZero::Unknown);
            self.sweep_seen.resize(tiles, false);
        }
    }

    fn set(&mut self, tile: usize, value: PageZero) {
        match self.latest[tile] {
            PageZero::Unknown => self.unknown -= 1,
            PageZero::Zero => self.zero -= 1,
            PageZero::NonZero => self.non_zero -= 1,
            PageZero::Unreadable => self.unreadable -= 1,
        }
        match value {
            PageZero::Unknown => self.unknown += 1,
            PageZero::Zero => self.zero += 1,
            PageZero::NonZero => self.non_zero += 1,
            PageZero::Unreadable => self.unreadable += 1,
        }
        self.latest[tile] = value;
    }

    fn record(&mut self, tiles: u32, results: &[(u32, PageZero)]) {
        self.resize(tiles);
        for &(tile, result) in results {
            let tile = tile as usize;
            self.set(tile, result);
            if result != PageZero::Unknown && !self.sweep_seen[tile] {
                self.sweep_seen[tile] = true;
                self.sweep_known += 1;
                match result {
                    PageZero::Unknown => {}
                    PageZero::Zero => self.sweep_zero += 1,
                    PageZero::NonZero => self.sweep_non_zero += 1,
                    PageZero::Unreadable => self.sweep_unreadable += 1,
                }
            }
        }
        if tiles > 0 && self.sweep_known == tiles as usize {
            self.last_sweep = Some(ZeroSweep { tiles, zero: self.sweep_zero, non_zero: self.sweep_non_zero, unreadable: self.sweep_unreadable });
            self.sweep_seen.fill(false);
            self.sweep_known = 0;
            self.sweep_zero = 0;
            self.sweep_non_zero = 0;
            self.sweep_unreadable = 0;
        }
    }

    fn sample(&self, tiles: u32, scanned: u32) -> ZeroPageSample {
        ZeroPageSample { tiles, scanned, scan_tiles: ZERO_SCAN_TILES, zero: self.zero, non_zero: self.non_zero, unreadable: self.unreadable, unknown: self.unknown, last_sweep: self.last_sweep }
    }
}

/// The next bounded range of mirror tiles. It never wraps inside one call, so a sample stays small.
fn rotated_range(next: usize, tiles: usize, limit: usize) -> (usize, usize, usize) {
    if tiles == 0 || limit == 0 {
        return (0, 0, 0);
    }
    let start = if next < tiles { next } else { 0 };
    let count = limit.min(tiles - start);
    let next = if start + count == tiles { 0 } else { start + count };
    (start, count, next)
}

#[derive(Clone, Copy)]
enum ZeroSource {
    /// `xmem.dll` was not available: leave the tile unobserved so a later tick retries it.
    Unknown,
    /// No live section view holds this pool page.
    Missing,
    Address(usize),
}

/// Source pages for one sampler tick, taken under the state lock and read after it is released.
struct ZeroPageChunk {
    state: u64,
    reads: Vec<(u32, ZeroSource)>,
}

impl ZeroPageChunk {
    fn read(self) -> Vec<(u32, PageZero)> {
        self.reads.into_iter().map(|(tile, source)| (tile, read_zero_page(source))).collect()
    }
}

/// Whether the whole 64 KiB source page is committed and readable, as the dirty-page path requires.
fn committed_readable(address: usize) -> bool {
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a valid output buffer of the size passed; VirtualQuery only reads the address space.
    if unsafe { VirtualQuery(address as *const c_void, &mut info, size_of_val(&info)) } == 0 {
        return false;
    }
    let start = info.BaseAddress as usize;
    address.checked_add(PAGE as usize).is_some_and(|end| {
        info.State == MEM_COMMIT
            && info.Protect != 0
            && info.Protect & PAGE_GUARD == 0
            && info.Protect & 0xff != PAGE_NOACCESS
            && address >= start
            && end <= start.saturating_add(info.RegionSize)
    })
}

/// Pure zero test used by the sampler; the compiler can vectorize the byte loop.
fn all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&byte| byte == 0)
}

fn read_zero_page(source: ZeroSource) -> PageZero {
    let address = match source {
        ZeroSource::Unknown => return PageZero::Unknown,
        ZeroSource::Missing => return PageZero::Unreadable,
        ZeroSource::Address(address) => address,
    };
    if !committed_readable(address) {
        return PageZero::Unreadable;
    }
    // SAFETY: the query above covers a full page of the read only section view, which lives as long as the engine.
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, PAGE as usize) };
    if all_zero(bytes) { PageZero::Zero } else { PageZero::NonZero }
}

/// Where a title span lives for the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The window buffer, a PC `ID3D12Resource` that lives as long as the process.
    pub resource: usize,
    /// The GPU address of the first byte of the span.
    pub gpu: u64,
    /// The byte offset of the span in the window buffer.
    pub offset: u64,
    /// The bytes of the slot after the span.
    pub left: u64,
}

/// One slot of the window: its buffer, its GPU address, its size in bytes and the title address of its
/// reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub resource: usize,
    pub gpu: u64,
    pub bytes: u64,
    pub va: u64,
}

/// A record of the journal of `xmem.dll` (`GpuChange` there).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuChange {
    seq: u64,
    kind: u32,
    pages: u32,
    va: u64,
    first_page: u32,
    flags: u32,
}

const RESERVE: u32 = 1;
const RELEASE: u32 = 2;
const MAP: u32 = 3;
const UNMAP: u32 = 4;
/// `flags` of a `RESERVE`: its pages come as the GPU is given addresses in them (`DecantXmemGpuUse`).
const ON_DEMAND: u32 = 1;
/// The bytes from a title address that the GPU may read when it is given only an address (a root view
/// has no size): asked of `xmem.dll` at once, so the pages after a root view's first one are there too.
const ADDRESS_REACH: u64 = 1 << 20;

type TrackFn = unsafe extern "system" fn(i32);
type ChangesFn = unsafe extern "system" fn(u64, *mut GpuChange, u32, *mut u32) -> i32;
type SnapshotFn = unsafe extern "system" fn(*mut GpuChange, u32, *mut u32, *mut u64) -> i32;
type TakeFn = unsafe extern "system" fn(*mut u32, u32) -> u32;
type ViewFn = unsafe extern "system" fn(u32, *mut usize, *mut u32, *mut u32) -> i32;
type GuardFn = unsafe extern "system" fn();
type WritableFn = unsafe extern "system" fn(usize, usize) -> i32;
type UseFn = unsafe extern "system" fn(usize, usize) -> u32;
type ShowFn = unsafe extern "system" fn(usize, usize, usize) -> i32;

/// The exports of `xmem.dll` this module uses.
struct Api {
    track: TrackFn,
    changes: ChangesFn,
    snapshot: SnapshotFn,
    take_dirty: TakeFn,
    section_view: ViewFn,
    guard_modules: GuardFn,
    writable: WritableFn,
    gpu_use: UseFn,
    /// `DecantXmemShowPages`, which an older `xmem.dll` does not have.
    show: Option<ShowFn>,
}

/// One pool heap: the buffer over it and where its memory is mapped.
struct Heap {
    _heap: ID3D12Heap,
    _buffer: ID3D12Resource,
    ptr: usize,
}

/// One window buffer: a reserved buffer, its GPU address.
struct Window {
    resource: ID3D12Resource,
    gpu: u64,
    bytes: u64,
}

#[derive(Default)]
struct Stats {
    syncs: u64,
    pages: u64,
    ranges: u64,
    micros: u64,
    worst_micros: u64,
    /// Of `micros`: asking `xmem.dll` for the written pages, copying them, updating the tile mappings.
    take_micros: u64,
    copy_micros: u64,
    tile_micros: u64,
}

struct State {
    /// Distinguishes a replacement device while the sampler reads outside the lock.
    id: u64,
    device: ID3D12Device,
    /// The private queue that applies the tile mappings, and the fence that says when.
    map_queue: ID3D12CommandQueue,
    map_fence: ID3D12Fence,
    fence_value: u64,
    mirror: Mirror,
    /// The sequence number of the last journal record applied.
    seq: u64,
    /// Pool heaps by index. A heap that cannot be made is `None` and is not tried again.
    heaps: Vec<Option<Heap>>,
    /// The heaps that could not be made.
    failed: HashSet<u32>,
    windows: Vec<Option<Window>>,
    /// The read only section views of `xmem.dll`: address, first pool page, pages.
    sections: Vec<(usize, u32, u32)>,
    /// The map fence value each submitting queue has waited for.
    waited: HashMap<usize, u64>,
    tracking: bool,
    last_guard: Instant,
    stats: Stats,
    /// Rotating zero-page observations, advanced only by the memory sampler.
    zero_scan: ZeroScan,
    /// Reused between syncs.
    scratch: Vec<GpuChange>,
    dirty: Vec<u32>,
    /// The mirror tiles of the pages the current sync copied: what the title wrote since the last one.
    written: Vec<u32>,
}

impl Drop for State {
    fn drop(&mut self) {
        let heaps = self.heaps.iter().flatten().count() as u64;
        crate::mem_owners::remove(
            crate::mem_owners::Kind::TitleMirrorHeap,
            crate::mem_owners::Segment::System,
            heaps,
            heaps * POOL_HEAP_BYTES,
        );
        let windows: Vec<_> = self.windows.iter().flatten().collect();
        crate::mem_owners::remove(
            crate::mem_owners::Kind::WindowReserved,
            crate::mem_owners::Segment::System,
            windows.len() as u64,
            windows.iter().map(|w| w.bytes).sum(),
        );
        let tiles = u64::from(self.mirror.tiles_given());
        crate::mem_owners::remove(
            crate::mem_owners::Kind::MirrorTile,
            crate::mem_owners::Segment::System,
            tiles,
            tiles * PAGE,
        );
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Serializes `sync` and `device_created` across queues so `execute_copies` runs without holding `STATE`.
static SYNC_LOCK: Mutex<()> = Mutex::new(());
/// Pages of commits on demand given to the GPU so far (`State::give`).
static GIVEN_PAGES: AtomicU64 = AtomicU64::new(0);

/// How many pages of commits on demand reached the GPU so far: the page table of the device memory
/// window is filled again when it changes (`service_mem`), so translated shaders see them too.
pub fn given_pages() -> u64 {
    GIVEN_PAGES.load(Ordering::Relaxed)
}

/// Whether the module is on: only with `DECANT_D3D12X_TITLE_MEM=1`. Phase 1 ships off by default: page
/// protection changes how the whole title runs (kernel writes into protected pages fail unless
/// `xmem.dll` opens them first, every first write to a clean page faults), so the first game runs
/// check the guard and the fault cost before it is on for everyone.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("DECANT_D3D12X_TITLE_MEM").is_some_and(|v| v == "1");
        if on {
            xs_log!("title memory: DECANT_D3D12X_TITLE_MEM=1, pool heaps and a window of reserved buffers, written pages found by page protection in xmem.dll");
        } else {
            xs_log!("title memory: off (DECANT_D3D12X_TITLE_MEM=1 turns it on), title addresses in xmem pages are not translated");
        }
        on
    })
}

/// The size of a window buffer in bytes: `DECANT_D3D12X_TITLE_WINDOW_GIB` GiB, 64 by default.
fn window_bytes() -> u64 {
    std::env::var("DECANT_D3D12X_TITLE_WINDOW_GIB").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|g| (1..=1024).contains(g)).unwrap_or(64) << 30
}

/// The exports of `xmem.dll`, looked up until found. `None` when the DLL is not loaded or lacks
/// them (an older `xmem.dll`), logged once.
fn api() -> Option<&'static Api> {
    static API: OnceLock<Api> = OnceLock::new();
    static MISSING: AtomicI32 = AtomicI32::new(0);
    if let Some(api) = API.get() {
        return Some(api);
    }
    // SAFETY: lookups of a module that is loaded or not, and of exports by name.
    let found = unsafe {
        let module = GetModuleHandleW(windows_sys::w!("xmem.dll"));
        if module.is_null() {
            None
        } else {
            let get = |name: &std::ffi::CStr| GetProcAddress(module, name.as_ptr().cast());
            // SAFETY: each address is the export of that name, whose signature the aliases above repeat.
            match (
                get(c"DecantXmemGpuTrack"),
                get(c"DecantXmemGpuChanges"),
                get(c"DecantXmemGpuSnapshot"),
                get(c"DecantXmemGpuTakeDirty"),
                get(c"DecantXmemGpuSectionView"),
                get(c"DecantXmemGpuGuardModules"),
                get(c"DecantXmemGpuWritable"),
                get(c"DecantXmemGpuUse"),
            ) {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g), Some(h)) => Some(Api {
                    track: std::mem::transmute::<unsafe extern "system" fn() -> isize, TrackFn>(a),
                    changes: std::mem::transmute::<unsafe extern "system" fn() -> isize, ChangesFn>(b),
                    snapshot: std::mem::transmute::<unsafe extern "system" fn() -> isize, SnapshotFn>(c),
                    take_dirty: std::mem::transmute::<unsafe extern "system" fn() -> isize, TakeFn>(d),
                    section_view: std::mem::transmute::<unsafe extern "system" fn() -> isize, ViewFn>(e),
                    guard_modules: std::mem::transmute::<unsafe extern "system" fn() -> isize, GuardFn>(f),
                    writable: std::mem::transmute::<unsafe extern "system" fn() -> isize, WritableFn>(g),
                    gpu_use: std::mem::transmute::<unsafe extern "system" fn() -> isize, UseFn>(h),
                    show: get(c"DecantXmemShowPages").map(|f| std::mem::transmute::<unsafe extern "system" fn() -> isize, ShowFn>(f)),
                }),
                _ => None,
            }
        }
    };
    match found {
        Some(api) => Some(API.get_or_init(|| api)),
        None => {
            if MISSING.fetch_add(1, Ordering::Relaxed) == 0 {
                xs_log!("title memory: xmem.dll is not loaded or has no DecantXmemGpu exports (yet), title addresses in xmem pages are not translated");
            }
            None
        }
    }
}

fn buffer_desc(width: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// A pool heap, the buffer over it and its mapped memory.
fn make_heap(device: &ID3D12Device, index: u32) -> windows::core::Result<Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_HEAP_BYTES,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain D3D12 calls on a live device with a description of plain data.
    unsafe { device.CreateHeap(&desc, &mut heap) }?;
    let heap = heap.ok_or_else(fail)?;
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: the heap is live. COMMON: a buffer promotes to the state each use needs and decays again.
    unsafe { device.CreatePlacedResource(&heap, 0, &buffer_desc(POOL_HEAP_BYTES, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) }?;
    let buffer = buffer.ok_or_else(fail)?;
    // names for DRED
    // SAFETY: live objects.
    unsafe {
        let _ = heap.SetName(&windows::core::HSTRING::from(format!("decant title pool heap {index}")));
        let _ = buffer.SetName(&windows::core::HSTRING::from(format!("decant title pool buffer {index}")));
    }
    let mut ptr = std::ptr::null_mut();
    // SAFETY: a buffer in a CPU visible heap, mapped for the life of the process (never unmapped).
    unsafe { buffer.Map(0, None, Some(&mut ptr)) }?;
    crate::mem_owners::add(
        crate::mem_owners::Kind::TitleMirrorHeap,
        crate::mem_owners::Segment::System,
        1,
        POOL_HEAP_BYTES,
    );
    Ok(Heap { _heap: heap, _buffer: buffer, ptr: ptr as usize })
}

/// A window buffer: a reserved buffer of `bytes` bytes.
fn make_window(device: &ID3D12Device, bytes: u64, index: usize) -> windows::core::Result<Window> {
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain D3D12 call on a live device.
    unsafe { device.CreateReservedResource(&buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) }?;
    let resource = buffer.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)))?;
    // SAFETY: a live resource.
    let gpu = unsafe { resource.GetGPUVirtualAddress() };
    // SAFETY: a live resource, named for DRED.
    let _ = unsafe { resource.SetName(&windows::core::HSTRING::from(format!("decant title window {index} of {} GiB", bytes >> 30))) };
    crate::mem_owners::add(
        crate::mem_owners::Kind::WindowReserved,
        crate::mem_owners::Segment::System,
        1,
        bytes,
    );
    Ok(Window { resource, gpu, bytes })
}

impl State {
    /// The pool heap `index`, made when first needed. `None` when it cannot be made.
    fn heap(&mut self, index: u32) -> Option<usize> {
        let i = index as usize;
        if self.heaps.len() <= i {
            self.heaps.resize_with(i + 1, || None);
        }
        if self.heaps[i].is_none() && !self.failed.contains(&index) {
            match make_heap(&self.device, index) {
                Ok(heap) => self.heaps[i] = Some(heap),
                Err(e) => {
                    self.failed.insert(index);
                    xs_log!("title memory: pool heap {index} cannot be made ({e}), its pages stay invisible to the GPU");
                }
            }
        }
        self.heaps[i].as_ref().map(|h| h.ptr)
    }

    /// Makes the window buffers the mirror has slots in.
    fn make_windows(&mut self) {
        let sizes = self.mirror.buffer_bytes();
        for (i, &bytes) in sizes.iter().enumerate().skip(self.windows.len()) {
            let made = make_window(&self.device, bytes, i);
            match &made {
                Ok(w) => xs_log!("title memory: window buffer {i} of {} GiB at GPU address {:#x}", bytes >> 30, w.gpu),
                Err(e) => xs_log!("title memory: window buffer {i} of {} GiB cannot be made ({e}), its slots stay invisible to the GPU", bytes >> 30),
            }
            self.windows.push(made.ok());
        }
    }

    /// The address of pool page `page` in the read only view of its section.
    fn source(&mut self, api: &Api, page: u32) -> Option<usize> {
        loop {
            if let Some(&(base, first, _)) = self.sections.iter().find(|&&(_, first, pages)| page >= first && page - first < pages) {
                return Some(base + (page - first) as usize * PAGE as usize);
            }
            let (mut base, mut first, mut pages) = (0usize, 0u32, 0u32);
            let index = self.sections.len() as u32;
            // SAFETY: the three outputs are valid locations.
            if unsafe { (api.section_view)(index, &mut base, &mut first, &mut pages) } == 0 {
                return None;
            }
            self.sections.push((base, first, pages));
        }
    }

    /// Snapshot the next rotating part of the mirror for the sampler. The page reads happen after the
    /// state lock is released; section views and pool heaps live for the life of the process.
    fn zero_page_chunk(&mut self, api: Option<&Api>) -> ZeroPageChunk {
        let tiles = self.mirror.tiles_given();
        self.zero_scan.resize(tiles);
        let (start, count, next) = rotated_range(self.zero_scan.next, tiles as usize, ZERO_SCAN_TILES);
        self.zero_scan.next = next;
        let mut reads = Vec::with_capacity(count);
        for tile in start..start + count {
            let source = self.mirror.page_of_tile(tile as u32).map_or(ZeroSource::Missing, |page| match api {
                Some(api) => self.source(api, page).map_or(ZeroSource::Missing, ZeroSource::Address),
                None => ZeroSource::Unknown,
            });
            reads.push((tile as u32, source));
        }
        ZeroPageChunk { state: self.id, reads }
    }

    /// The GPU is given `[va, va + len)`: when it lies in a commit whose pages come on demand and some are
    /// not mapped yet, `xmem.dll` is asked for them and the journal read again, so they translate now and
    /// are copied at the next sync.
    fn give(&mut self, api: &Api, va: u64, len: u64) {
        static GIVEN: AtomicI32 = AtomicI32::new(0);
        if !self.mirror.wants_use(va, len) {
            return;
        }
        // SAFETY: the export takes plain values.
        let pages = unsafe { (api.gpu_use)(va as usize, len as usize) };
        if pages > 0 {
            GIVEN_PAGES.fetch_add(u64::from(pages), Ordering::Relaxed);
            if GIVEN.fetch_add(1, Ordering::Relaxed) < 8 {
                xs_log!("title memory: {pages} pages from title address {va:#x} reach the GPU now (a commit whose pages come on demand)");
            }
            self.drain(api);
        }
    }

    fn apply(&mut self, c: &GpuChange) {
        static REFUSED: AtomicI32 = AtomicI32::new(0);
        let change = match c.kind {
            RESERVE => Change::Reserve { va: c.va, pages: c.pages, on_demand: c.flags & ON_DEMAND != 0 },
            RELEASE => Change::Release { va: c.va },
            MAP => Change::Map { va: c.va, pages: c.pages, first_page: c.first_page },
            UNMAP => Change::Unmap { va: c.va, pages: c.pages },
            _ => return,
        };
        if matches!(c.kind, RELEASE | MAP | UNMAP) {
            bump_mapping_gen();
        }
        if let Some(bytes) = self.mirror.apply(change)
            && REFUSED.fetch_add(1, Ordering::Relaxed) == 0
        {
            xs_log!("title memory: a reservation of {} GiB is larger than a window buffer can be (1 TiB), its pages stay invisible to the GPU", bytes >> 30);
        }
    }

    /// Reads what `xmem.dll` changed since the last time into the mirror, or starts again from a snapshot
    /// when its ring no longer holds it.
    fn drain(&mut self, api: &Api) {
        loop {
            let mut written = 0u32;
            self.scratch.resize(CHANGE_RECORDS, GpuChange::default());
            // SAFETY: `scratch` has room for `CHANGE_RECORDS` records, `written` is a valid location.
            let ok = unsafe { (api.changes)(self.seq, self.scratch.as_mut_ptr(), CHANGE_RECORDS as u32, &mut written) };
            if ok == 0 {
                self.resync(api);
                continue;
            }
            let n = (written as usize).min(CHANGE_RECORDS);
            for i in 0..n {
                let c = self.scratch[i];
                self.apply(&c);
                self.seq = c.seq;
            }
            if n < CHANGE_RECORDS {
                break;
            }
        }
        self.make_windows();
    }

    fn resync(&mut self, api: &Api) {
        let mut records = vec![GpuChange::default(); CHANGE_RECORDS];
        let (mut written, mut seq) = (0u32, 0u64);
        loop {
            // SAFETY: `records` has room for its length, the outputs are valid locations.
            let ok = unsafe { (api.snapshot)(records.as_mut_ptr(), records.len() as u32, &mut written, &mut seq) };
            if ok != 0 {
                break;
            }
            records.resize(written as usize + 1024, GpuChange::default());
        }
        records.truncate(written as usize);
        xs_log!("title memory: the journal of xmem.dll overran, starting again from a snapshot of {} records", records.len());
        self.mirror.reset();
        bump_mapping_gen();
        for c in &records {
            self.apply(c);
        }
        self.seq = seq;
    }

    /// Applies the pending tile mappings on the private queue and signals the fence. Returns the ranges.
    fn update_tiles(&mut self) -> u64 {
        let _trace = decant_trace::span("title_mem", "update_tiles");
        let calls = self.mirror.drain_updates();
        if calls.is_empty() {
            return 0;
        }
        let mut ranges = 0;
        for mut call in calls {
            // A heap that cannot be made shows nothing, as the tiles of a release do.
            if let Some(h) = call.heap
                && self.heap(h).is_none()
            {
                call.heap = None;
            }
            let Some(window) = self.windows.get(call.buffer).and_then(|w| w.as_ref()) else { continue };
            ranges += call.ranges.len() as u64;
            update_call(&self.map_queue, &window.resource, &call, call.heap.and_then(|h| self.heaps.get(h as usize)).and_then(|h| h.as_ref()).map(|h| &h._heap));
        }
        self.fence_value += 1;
        // SAFETY: both objects are live.
        if let Err(e) = unsafe { self.map_queue.Signal(&self.map_fence, self.fence_value) } {
            xs_log!("title memory: the map fence cannot be signalled ({e})");
        }
        ranges
    }

    /// Collects copy jobs for pool pages written since the last sync, ensuring heaps and sections exist.
    fn prepare_upload(&mut self, api: &Api, jobs: &mut Vec<CopyJob>) -> u64 {
        let _trace = decant_trace::span("title_mem", "upload");
        let mut total = 0;
        let mut new_tile = false;
        loop {
            self.dirty.resize(TAKE_PAGES, 0);
            // SAFETY: `dirty` has room for `TAKE_PAGES` entries.
            let asked = Instant::now();
            let n = {
                let _take = decant_trace::span("title_mem", "take_dirty");
                unsafe { (api.take_dirty)(self.dirty.as_mut_ptr(), TAKE_PAGES as u32) }
            } as usize;
            self.stats.take_micros += asked.elapsed().as_micros() as u64;
            let mut pages = std::mem::take(&mut self.dirty);
            pages.truncate(n);
            pages.sort_unstable();
            pages.dedup();
            // A page copied for the first time has bytes now: it gets its mirror tile here (in page order,
            // so consecutive pages get consecutive tiles), and the window shows it from this sync on.
            for &page in &pages {
                if self.mirror.tile_of(page).is_none() {
                    new_tile = true;
                }
                self.mirror.give_tile(page);
            }
            self.plan_runs(api, &pages, jobs, &mut total);
            self.dirty = pages;
            if n < TAKE_PAGES {
                break;
            }
        }
        if new_tile {
            bump_mapping_gen();
        }
        total
    }

    fn plan_runs(&mut self, api: &Api, pages: &[u32], jobs: &mut Vec<CopyJob>, copied: &mut u64) {
        let mut i = 0;
        while i < pages.len() {
            let first = pages[i];
            let Some(first_tile) = self.mirror.tile_of(first) else {
                i += 1;
                continue;
            };
            let (heap, tile) = heap_of(first_tile);
            let mut n = 1usize;
            while i + n < pages.len()
                && pages[i + n] == first + n as u32
                && self.mirror.tile_of(pages[i + n]) == Some(first_tile + n as u32)
                && heap_of(first_tile + n as u32).0 == heap
            {
                n += 1;
            }
            i += n;
            let (Some(dst), Some(src)) = (self.heap(heap), self.source(api, first)) else { continue };
            // One run never leaves its section: the tail of a section is cut off and copied by page.
            let in_section = self.sections.iter().find(|&&(_, f, p)| first >= f && first - f < p).map_or(1, |&(_, f, p)| (f + p - first) as usize);
            let (head, rest) = (n.min(in_section), n.saturating_sub(in_section));
            plan_copy(jobs, src, dst + tile as usize * PAGE as usize, head);
            *copied += head as u64;
            self.written.extend(first_tile..first_tile + head as u32);
            if rest > 0 {
                self.plan_runs(api, &pages[i - rest..i], jobs, copied);
            }
        }
    }
}

/// One `UpdateTileMappings` call of the private queue.
fn update_call(queue: &ID3D12CommandQueue, window: &ID3D12Resource, call: &TileCall, heap: Option<&ID3D12Heap>) {
    let n = call.ranges.len() as u32;
    let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = call.ranges.iter().map(|&(tile, _)| D3D12_TILED_RESOURCE_COORDINATE { X: tile, Y: 0, Z: 0, Subresource: 0 }).collect();
    let sizes: Vec<D3D12_TILE_REGION_SIZE> = call.ranges.iter().map(|&(_, count)| D3D12_TILE_REGION_SIZE { NumTiles: count, ..Default::default() }).collect();
    let counts: Vec<u32> = call.ranges.iter().map(|&(_, count)| count).collect();
    let flag = if call.heap.is_some() && heap.is_some() { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_NULL };
    let flags = vec![flag; n as usize];
    // SAFETY: the arrays have `n` entries each, the resource and the heap are live.
    unsafe { queue.UpdateTileMappings(window, n, Some(coords.as_ptr()), Some(sizes.as_ptr()), heap, n, Some(flags.as_ptr()), Some(call.heap_tiles.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE) };
    crate::diagnose::removed_after(|| format!("UpdateTileMappings of {n} ranges on window buffer {} (heap {:?}, first tile {:?})", call.buffer, call.heap, call.ranges.first()));
}

/// Notes the PC device that the mirror is made on, with its private queue and fence. A second device
/// starts a new mirror.
///
/// # Safety
/// `device` is a live PC `ID3D12Device`.
pub unsafe fn device_created(device: *mut c_void) {
    if !enabled() {
        return;
    }
    // SAFETY: by the contract above. `from_raw_borrowed` takes no reference, the clone takes one.
    let Some(device) = (unsafe { ID3D12Device::from_raw_borrowed(&device) }).cloned() else { return };
    let queue = unsafe { device.CreateCommandQueue::<ID3D12CommandQueue>(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }) };
    let fence = unsafe { device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) };
    let (Ok(map_queue), Ok(map_fence)) = (queue, fence) else {
        xs_log!("title memory: the queue or the fence of the tile mappings cannot be made, title memory stays off");
        return;
    };
    let _sync_guard = SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if st.is_some() {
        xs_log!("title memory: a second PC device, the heaps of the first are not used for it");
    }
    bump_mapping_gen();
    *st = Some(State {
        id: NEXT_STATE.fetch_add(1, Ordering::Relaxed),
        device,
        map_queue,
        map_fence,
        fence_value: 0,
        mirror: Mirror::new(window_bytes()),
        seq: 0,
        heaps: Vec::new(),
        failed: HashSet::new(),
        windows: Vec::new(),
        sections: Vec::new(),
        waited: HashMap::new(),
        tracking: false,
        last_guard: Instant::now(),
        stats: Stats::default(),
        zero_scan: ZeroScan::default(),
        scratch: Vec::new(),
        dirty: Vec::new(),
        written: Vec::new(),
    });
}

/// Runs `f` on the state, with the exports of `xmem.dll`, once tracking is on and the journal is read.
/// `None` when the module is off, has no device or `xmem.dll` is not there.
fn with_state<R>(f: impl FnOnce(&mut State, &Api) -> R) -> Option<R> {
    if !enabled() {
        return None;
    }
    let api = api()?;
    let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.lock().unwrap_or_else(|e| e.into_inner()));
    let st = guard.as_mut()?;
    if !st.tracking {
        st.tracking = true;
        // SAFETY: the export takes a flag.
        unsafe { (api.track)(1) };
        xs_log!("title memory: xmem.dll tracks written pages now, every mapped page is copied at the first sync");
    }
    st.drain(api);
    Some(f(st, api))
}

/// Samples the mirror footprint and one rotating part of its title pages for the memory sampler. Only
/// the sampler calls this: source addresses are taken under the state lock, then up to 256 MiB is read
/// and tested on the sampler thread after the lock is released. `None` before the first device, with
/// the module off, or when a replacement device won the race.
pub fn sample() -> Option<MirrorSample> {
    if !enabled() {
        return None;
    }
    let api = api();
    let chunk = {
        let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_mut()?.zero_page_chunk(api)
    };
    let state = chunk.state;
    let results = chunk.read();
    let scanned = results.len() as u32;
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let st = guard.as_mut()?;
    if st.id != state {
        return None;
    }
    st.zero_scan.record(st.mirror.tiles_given(), &results);
    Some(MirrorSample {
        heaps: st.heaps.iter().filter(|h| h.is_some()).count(),
        tiles: st.mirror.tiles_given(),
        windows: st.windows.iter().filter(|w| w.is_some()).count(),
        zero_pages: st.zero_scan.sample(st.mirror.tiles_given(), scanned),
    })
}

/// The PC device the mirror is made on, one more reference (`None` before the first device).
pub fn device() -> Option<ID3D12Device> {
    STATE.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|s| s.device.clone())
}

/// The generation of the window slots now: it changes when a slot is given or freed, not when a page
/// is mapped inside one. `None` when the module is off or `xmem.dll` is not there.
pub fn generation() -> Option<u32> {
    with_state(|st, _| st.mirror.generation())
}

/// The slots of the window (GPU address and bytes), with the generation they are for. Empty when there is
/// no device or no slot, `None` when the module is off.
pub fn extents() -> Option<(u32, Vec<Entry>)> {
    with_state(|st, _| {
        let entries = st
            .mirror
            .slots()
            .into_iter()
            .filter_map(|(va, buffer, offset, bytes)| {
                let w = st.windows.get(buffer)?.as_ref()?;
                Some(Entry { resource: w.resource.as_raw() as usize, gpu: w.gpu + offset, bytes, va })
            })
            .collect();
        (st.mirror.generation(), entries)
    })
}

/// The span of `[va, va + len)` in the window, when it lies in one reservation that has a slot and the page of
/// `va` is mapped. `None` for anything else (including the module being off). The GPU is about to be given
/// the span: pages of a commit whose pages come on demand are asked for first (at least `ADDRESS_REACH`
/// bytes from `va`, for the views that carry no size).
pub fn translate(va: u64, len: u64) -> Option<Span> {
    span_of(va, len, true)
}

/// As `translate`, but pages that come on demand are not asked for: for the page table of the device
/// memory window, which looks at every page and would bring every page of those commits in.
pub fn lookup(va: u64, len: u64) -> Option<Span> {
    span_of(va, len, false)
}

/// As `translate`, and whether a tile of the span has a mapping that only the next `sync` applies (the
/// pages it brought in now, or a remap since the last sync): GPU work that reads the span before that
/// sync may see the old mapping.
pub fn translate_fresh(va: u64, len: u64) -> Option<(Span, bool)> {
    let mut fresh = false;
    let span = span_of_with(va, len, true, &mut fresh)?;
    Some((span, fresh))
}

fn span_of(va: u64, len: u64, give: bool) -> Option<Span> {
    span_of_with(va, len, give, &mut false)
}

impl State {
    fn translate_span(&mut self, api: &Api, va: u64, len: u64, give: bool) -> Option<(Span, bool)> {
        if give {
            self.give(api, va, len.max(ADDRESS_REACH));
            // Pages the title never wrote have no mirror tile and read as zeros. The first `ADDRESS_REACH`
            // bytes of a span the GPU is given get tiles anyway: the GPU may write there (a root UAV, a
            // counter) and read it back later.
            let prev_tiles = self.mirror.tiles_given();
            self.mirror.give_span(va, len.clamp(1, ADDRESS_REACH));
            if self.mirror.tiles_given() != prev_tiles {
                bump_mapping_gen();
            }
        }
        let placed = self.mirror.translate(va, len)?;
        let w = self.windows.get(placed.buffer)?.as_ref()?;
        let is_fresh = self.mirror.pending_in(placed.buffer, placed.offset, len);
        Some((Span { resource: w.resource.as_raw() as usize, gpu: w.gpu + placed.offset, offset: placed.offset, left: placed.left }, is_fresh))
    }
}

/// Translates a batch of `(va, len)` spans under one state lock and one journal drain.
pub fn translate_fresh_batch(spans: &[(u64, u64)], out: &mut Vec<Option<(Span, bool)>>) {
    out.clear();
    if spans.is_empty() {
        return;
    }
    let filled = with_state(|st, api| {
        out.extend(spans.iter().map(|&(va, len)| st.translate_span(api, va, len, true)));
    });
    if filled.is_none() {
        out.resize(spans.len(), None);
    }
}

fn span_of_with(va: u64, len: u64, give: bool, fresh: &mut bool) -> Option<Span> {
    static SEEN: AtomicI32 = AtomicI32::new(0);
    static NEAR: AtomicI32 = AtomicI32::new(0);
    let (span, is_fresh) = with_state(|st, api| st.translate_span(api, va, len, give))??;
    *fresh = is_fresh;
    if SEEN.fetch_add(1, Ordering::Relaxed) < 3 {
        xs_log!("title memory: title address {va:#x} ({len} bytes) is at offset {:#x} of a window buffer, GPU address {:#x}", span.offset, span.gpu);
    }
    if span.left < NEAR_END_BYTES && NEAR.fetch_add(1, Ordering::Relaxed) < 8 {
        xs_log!("title memory: title address {va:#x} ({len} bytes) is {} bytes before the end of its reservation", span.left);
    }
    Some(span)
}

/// The tile runs of one pool heap: `(first buffer tile, tiles, first heap tile)`.
type HeapRuns = (u32, Vec<(u32, u32, u32)>);

/// Tile runs of a buffer over mirror tiles: `(first buffer tile, tiles, first heap tile)` for each
/// pool heap, a run ending where the next tile is not the next tile of the same heap.
fn runs_by_heap(pages: &[u32]) -> Vec<HeapRuns> {
    let mut by_heap: Vec<HeapRuns> = Vec::new();
    let mut last: Option<(u32, usize)> = None;
    for (k, &page) in pages.iter().enumerate() {
        let (heap, tile) = heap_of(page);
        if let Some((prev, i)) = last
            && prev + 1 == page
            && heap_of(prev).0 == heap
            && let Some(run) = by_heap[i].1.last_mut()
        {
            run.1 += 1;
            last = Some((page, i));
            continue;
        }
        let i = match by_heap.iter().position(|(h, _)| *h == heap) {
            Some(i) => i,
            None => {
                by_heap.push((heap, Vec::new()));
                by_heap.len() - 1
            }
        };
        by_heap[i].1.push((k as u32, 1, tile));
        last = Some((page, i));
    }
    by_heap
}

/// The description of the reserved buffer that stands in for a placed buffer. A reserved buffer takes
/// no flag but `ALLOW_UNORDERED_ACCESS`, which it also gets when the initial state needs it.
fn reserved_desc(desc: &D3D12_RESOURCE_DESC, state: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_DESC {
    let needs_uav = state.0 & (D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 | D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE.0) != 0;
    let mut flags = desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if needs_uav {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    D3D12_RESOURCE_DESC { Alignment: 0, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: flags, ..*desc }
}

impl State {
    /// Maps the tiles of `buffer` onto the mirror tiles `pages` (one per buffer tile, in order), one
    /// `UpdateTileMappings` call per pool heap, and signals the map fence. `None` when a heap is missing.
    fn map_to_pool(&mut self, buffer: &ID3D12Resource, pages: &[u32]) -> Option<()> {
        for (heap, runs) in runs_by_heap(pages) {
            self.heap(heap)?;
            let pool = self.heaps.get(heap as usize)?.as_ref()?;
            let call = TileCall {
                buffer: 0,
                heap: Some(heap),
                ranges: runs.iter().map(|&(at, n, _)| (at, n)).collect(),
                heap_tiles: runs.iter().map(|&(_, _, t)| t).collect(),
            };
            update_call(&self.map_queue, buffer, &call, Some(&pool._heap));
        }
        self.fence_value += 1;
        // SAFETY: both objects are live.
        if let Err(e) = unsafe { self.map_queue.Signal(&self.map_fence, self.fence_value) } {
            xs_log!("title memory: the map fence cannot be signalled ({e})");
        }
        Some(())
    }
}

/// A PC buffer for a buffer the title places at title address `va` (`CreatePlacedResourceX`): a reserved
/// buffer whose tiles are mapped to the pool tiles of the title pages there, the same tiles the window
/// shows at `va`. What the GPU writes into it is then at the title address for every later GPU read, and
/// what the title writes at `va` (copied into the pool at each sync) is in it. On the console a placed
/// buffer is that memory, and console code reads root views and V#s from the memory, not from a resource.
///
/// `desc` is the PC description (sanitized), `state` the PC initial state. `None` when the module is off,
/// `va` is not on a page, a page is not mapped or a call fails: the caller keeps its own way. The tile
/// mappings are made once: a later remap of those title pages does not follow (not seen in a retail title).
pub fn place_buffer(va: u64, desc: &D3D12_RESOURCE_DESC, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
    static FAILED: AtomicI32 = AtomicI32::new(0);
    static PLACED: AtomicI32 = AtomicI32::new(0);
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width == 0 {
        return None;
    }
    let tiles = u32::try_from(desc.Width.div_ceil(PAGE)).ok()?;
    let reserved = reserved_desc(desc, state);
    let made = with_state(|st, api| {
        st.give(api, va, u64::from(tiles) * PAGE);
        let pages = st.mirror.mirror_tiles(va, tiles)?;
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain D3D12 call on a live device with a description of plain data.
        let created = unsafe { st.device.CreateReservedResource(&reserved, state, None, &mut buffer) };
        if let Err(e) = created {
            if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
                xs_log!("title memory: the buffer placed at {va:#x} ({} bytes) cannot be a reserved buffer ({e})", desc.Width);
            }
            return None;
        }
        let buffer = buffer?;
        st.map_to_pool(&buffer, &pages)?;
        Some(buffer)
    })??;
    if PLACED.fetch_add(1, Ordering::Relaxed) < 8 {
        // SAFETY: a live resource.
        let gpu = unsafe { made.GetGPUVirtualAddress() };
        xs_log!("title memory: the buffer placed at {va:#x} ({} bytes) is a reserved buffer over its title pages, GPU address {gpu:#x}", desc.Width);
    }
    Some(made)
}

/// Whether `[va, va + len)` lies in memory the title may write, even where `xmem.dll` made the page read
/// only for a while (a clean page of the mirror). `false` when the module is off or `xmem.dll` is not there.
pub fn writable(va: usize, len: usize) -> bool {
    // SAFETY: the export only looks the range up.
    enabled() && api().is_some_and(|api| unsafe { (api.writable)(va, len) } != 0)
}

/// Brings the GPU mirror up to date for a submission on `queue`: applies what `xmem.dll` changed, copies the
/// pages written since the last time into the pool heaps, and makes `queue` wait for the tile mappings. Runs
/// on the CPU right before every `ExecuteCommandLists` of the title. One mutex serializes the queues.
///
/// # Safety
/// `queue` is a live PC command queue.
pub unsafe fn sync(queue: *mut c_void) {
    if !enabled() {
        return;
    }
    let _trace = decant_trace::span("title_mem", "sync");
    // SAFETY: by the contract above.
    let Some(queue_ref) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return };
    let _sync_guard = decant_trace::timed_lock("title_mem::SYNC_LOCK", || SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner()));
    let started = Instant::now();
    let Some((phase1_id, jobs, pages, written)) = with_state(|st, api| {
        let mut jobs = Vec::new();
        let pages = st.prepare_upload(api, &mut jobs);
        let written = std::mem::take(&mut st.written);
        (st.id, jobs, pages, written)
    }) else {
        return;
    };

    static CORES: OnceLock<usize> = OnceLock::new();
    let cores = *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from));
    let copy_micros = if !jobs.is_empty() {
        let copying = Instant::now();
        let _copy = decant_trace::span("title_mem", "copy_runs");
        // SAFETY: sorted unique pool pages have unique mirror tiles. Planning keeps each job
        // inside a live section and heap, and SYNC_LOCK keeps `device_created` from replacing heaps.
        unsafe { execute_copies(&jobs, cores) };
        copying.elapsed().as_micros() as u64
    } else {
        0
    };

    if let Some(api) = api() {
        let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(st) = guard.as_mut().filter(|st| st.id == phase1_id) {
            st.stats.copy_micros += copy_micros;
            let tiles = Instant::now();
            let ranges = st.update_tiles();
            st.stats.tile_micros += tiles.elapsed().as_micros() as u64;
            if st.fence_value > 0 && st.waited.get(&(queue as usize)).is_none_or(|&v| v < st.fence_value) {
                // SAFETY: both objects are live. The queue waits on the GPU, the CPU does not.
                match unsafe { queue_ref.Wait(&st.map_fence, st.fence_value) } {
                    Ok(()) => {
                        st.waited.insert(queue as usize, st.fence_value);
                    }
                    Err(e) => xs_log!("title memory: a queue cannot wait for the tile mappings ({e})"),
                }
            }
            crate::diagnose::removed_after(|| format!("a sync of {pages} pages and {ranges} tile ranges"));
            if st.last_guard.elapsed() >= GUARD_EVERY {
                st.last_guard = Instant::now();
                // SAFETY: the export takes no argument.
                unsafe { (api.guard_modules)() };
            }
            let micros = started.elapsed().as_micros() as u64;
            let s = &mut st.stats;
            s.syncs += 1;
            s.pages += pages;
            s.ranges += ranges;
            s.micros += micros;
            s.worst_micros = s.worst_micros.max(micros);
            if s.syncs.is_multiple_of(STATS_EVERY) {
                xs_log!(
                    "title memory: {} syncs, {} pages copied ({} MiB), {} tile ranges updated, {} us in total ({} us asking for written pages, {} us copying, {} us mapping tiles), {} us at worst, {} slots",
                    s.syncs,
                    s.pages,
                    (s.pages * PAGE) >> 20,
                    s.ranges,
                    s.micros,
                    s.take_micros,
                    s.copy_micros,
                    s.tile_micros,
                    s.worst_micros,
                    st.mirror.slots().len()
                );
                st.stats = Stats { syncs: st.stats.syncs, ..Stats::default() };
            }
        }
    }
    drop(_sync_guard);
    // A placed texture whose memory the title wrote is filled again (after this lock is given back:
    // the detile table asks for tiles under its own lock).
    if !written.is_empty() {
        let _note_tiles = decant_trace::span("title_mem", "note_tiles");
        crate::detile::note_tiles(&written);
    }
}

/// `pages` pages from title address `to` show the pool pages that the pages from `from` show, or
/// nothing when `from` is 0 (`DecantXmemShowPages`): for the CPU at once, for the GPU with the next
/// `sync`, which reads the change from the journal. `None` when `xmem.dll` cannot do it (the module
/// is off, or an older `xmem.dll`), else whether it did.
pub fn show_pages(to: u64, from: u64, pages: u32) -> Option<bool> {
    if !enabled() {
        return None;
    }
    let show = api()?.show?;
    // SAFETY: the export checks its arguments and maps nothing when it refuses.
    Some(unsafe { show(to as usize, from as usize, pages as usize) } != 0)
}

/// For each span `(va, len)` of `spans`: whether one of its pages shows a mirror tile of `written`,
/// `None` when its pages are not all mapped in one reservation. One lock for all spans: the detile
/// table asked `tiles_of` once per texture at every sync, tens of thousands of times while a title loads its world,
/// and held its own lock meanwhile.
pub fn spans_touch(spans: &[(u64, u64)], written: &HashSet<u32>) -> Vec<Option<bool>> {
    with_state(|st, _| {
        spans
            .iter()
            .map(|&(va, len)| {
                let (first, pages) = crate::title_mirror::page_span(va, len);
                st.mirror.touches(first, pages, written)
            })
            .collect()
    })
    .unwrap_or_else(|| vec![None; spans.len()])
}

/// The mirror tiles of the pages that hold the `len` bytes at title address `va`, when all of them are
/// mapped pages of one reservation. `None` for anything else (including the module being off).
pub fn tiles_of(va: u64, len: u64) -> Option<Vec<u32>> {
    let (first, pages) = crate::title_mirror::page_span(va, len);
    with_state(|st, _| st.mirror.mirror_tiles(first, pages))?
}

/// Mirror tiles for each `(va, len)` span under one state lock and one journal drain.
pub fn tiles_of_batch(spans: &[(u64, u64)], out: &mut Vec<Option<Vec<u32>>>) {
    out.clear();
    if spans.is_empty() {
        return;
    }
    let filled = with_state(|st, _| {
        out.extend(spans.iter().map(|&(va, len)| {
            let (first, pages) = crate::title_mirror::page_span(va, len);
            st.mirror.mirror_tiles(first, pages)
        }));
    });
    if filled.is_none() {
        out.resize(spans.len(), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_planning_bounds_chunks_and_thread_counts() {
        assert_eq!(copy_threads(0, 8), 1);
        assert_eq!(copy_threads(15, 8), 1);
        assert_eq!(copy_threads(32, 8), 2);
        assert_eq!(copy_threads(4096, 3), 3);
        assert_eq!(copy_threads(4096, 0), 1);
        let mut jobs = Vec::new();
        plan_copy(&mut jobs, 0, 1024, 0);
        assert!(jobs.is_empty());
        plan_copy(&mut jobs, 0, 1024, 35);
        assert_eq!(jobs.iter().map(|job| job.pages).collect::<Vec<_>>(), [16, 16, 3]);
        for (i, job) in jobs.iter().enumerate() {
            assert_eq!(job.source, i * 16 * PAGE as usize);
            assert_eq!(job.destination, 1024 + job.source);
        }
    }

    #[test]
    fn parallel_copies_match_inline_with_scattered_ranges_and_repeated_batches() {
        let pages = 259;
        let source: Vec<u8> = (0..pages * PAGE as usize).map(|i| (i / PAGE as usize + i % 251) as u8).collect();
        let mut inline = vec![0xa5; source.len()];
        let mut parallel = inline.clone();
        let mut jobs = Vec::new();
        for (output, cores) in [(&mut inline, 1), (&mut parallel, 4)] {
            jobs.clear();
            for (src, dst, count) in [(130, 0, 129), (0, 130, 128)] {
                plan_copy(&mut jobs, source.as_ptr() as usize + src * PAGE as usize, output.as_mut_ptr() as usize + dst * PAGE as usize, count);
            }
            for _ in 0..3 {
                // SAFETY: vectors live until the workers join and the destination spans are disjoint.
                unsafe { execute_copies(&jobs, cores) };
            }
        }
        assert_eq!(parallel, inline);
        assert_eq!(&parallel[..129 * PAGE as usize], &source[130 * PAGE as usize..]);
        assert!(parallel[129 * PAGE as usize..130 * PAGE as usize].iter().all(|&b| b == 0xa5));
        assert_eq!(&parallel[130 * PAGE as usize..258 * PAGE as usize], &source[..128 * PAGE as usize]);
        assert!(parallel[258 * PAGE as usize..].iter().all(|&b| b == 0xa5));
    }

    #[test]
    fn a_page_is_zero_only_when_every_byte_is_zero() {
        assert!(all_zero(&[]));
        assert!(all_zero(&[0; 257]));
        assert!(!all_zero(&[0, 0, 1, 0]), "a byte in the middle is enough");
        assert!(!all_zero(&[1, 0, 0]), "the first byte is checked");
        assert!(!all_zero(&[0, 0, 0x80]), "the last byte is checked");
        assert!(!all_zero(&[0xff; 64]));
    }

    #[test]
    fn the_zero_scan_rotates_in_bounded_chunks_and_covers_every_tile() {
        let mut next = 0;
        let mut seen = Vec::new();
        loop {
            let (start, count, after) = rotated_range(next, 10, 4);
            assert!(count > 0 && count <= 4);
            seen.extend(start..start + count);
            next = after;
            if next == 0 {
                break;
            }
        }
        assert_eq!(seen, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(next, 0, "the next sample starts a new sweep");
        assert_eq!(rotated_range(8, 10, 4), (8, 2, 0), "the tail wraps before the next sample");
        assert_eq!(rotated_range(4, 4, 4), (0, 4, 0), "a stale cursor starts again");
        assert_eq!(rotated_range(0, 0, 4), (0, 0, 0));
        assert_eq!(rotated_range(0, 10, 0), (0, 0, 0));
    }

    #[test]
    fn rotating_results_keep_coverage_and_the_latest_complete_sweep() {
        let mut scan = ZeroScan::default();
        scan.record(4, &[(0, PageZero::Zero), (1, PageZero::Zero), (2, PageZero::NonZero), (3, PageZero::Unreadable)]);
        assert_eq!(scan.sample(4, 4), ZeroPageSample {
            tiles: 4,
            scanned: 4,
            scan_tiles: ZERO_SCAN_TILES,
            zero: 2,
            non_zero: 1,
            unreadable: 1,
            unknown: 0,
            last_sweep: Some(ZeroSweep { tiles: 4, zero: 2, non_zero: 1, unreadable: 1 }),
        });
        scan.record(6, &[(4, PageZero::Zero), (5, PageZero::Unknown)]);
        assert_eq!((scan.zero, scan.unknown), (3, 1), "new tiles are unobserved until read");
        scan.record(6, &[(5, PageZero::NonZero)]);
        assert_eq!((scan.zero, scan.non_zero, scan.unknown), (3, 2, 0), "the rolling count is current before a full sweep");
        assert_eq!(scan.last_sweep, Some(ZeroSweep { tiles: 4, zero: 2, non_zero: 1, unreadable: 1 }), "the new tiles alone do not complete a sweep");
        scan.record(6, &[(0, PageZero::NonZero), (1, PageZero::Zero), (2, PageZero::NonZero), (3, PageZero::Unreadable)]);
        assert_eq!(scan.last_sweep, Some(ZeroSweep { tiles: 6, zero: 2, non_zero: 3, unreadable: 1 }), "every tile has now been seen in this sweep");
        scan.record(6, &[(0, PageZero::Zero), (1, PageZero::Zero), (2, PageZero::NonZero), (3, PageZero::Unreadable), (4, PageZero::Zero), (5, PageZero::NonZero)]);
        assert_eq!((scan.zero, scan.non_zero), (3, 2), "the rolling count follows the latest read");
        assert_eq!(scan.last_sweep, Some(ZeroSweep { tiles: 6, zero: 3, non_zero: 2, unreadable: 1 }), "the next full sweep replaces the previous one");
    }

    #[test]
    fn a_placed_buffer_maps_runs_of_consecutive_pages_per_pool_heap() {
        let runs = runs_by_heap(&[5, 6, 7, 1023, 1024, 1025, 9]);
        assert_eq!(runs, [(0, vec![(0, 3, 5), (3, 1, 1023), (6, 1, 9)]), (1, vec![(4, 2, 0)])], "a run ends at a heap's end and where the pages jump");
        assert!(runs_by_heap(&[]).is_empty());
    }

    #[test]
    fn translate_fresh_batch_clears_and_returns_none_when_disabled() {
        let mut out = Vec::with_capacity(8);
        out.push(Some((Span { resource: 1, gpu: 2, offset: 3, left: 4 }, true)));
        translate_fresh_batch(&[], &mut out);
        assert!(out.is_empty());
        translate_fresh_batch(&[(0x1000, 64), (0x2000, 128)], &mut out);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(Option::is_none));
    }

    #[test]
    fn mapping_generation_bumps_on_changes_and_sync_lock_does_not_block_translate() {
        let before = mapping_generation();
        bump_mapping_gen();
        assert!(mapping_generation() > before);

        let _sync_guard = SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let done = std::thread::spawn(|| translate(0x1000, 64)).join().unwrap();
        assert!(done.is_none());
    }
}
