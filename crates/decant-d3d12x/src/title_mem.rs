//! Title memory seen by the PC GPU. On the console the GPU address of vertex, index and constant
//! data is its CPU address. Here those bytes sit in pages that `xmem.dll` maps with
//! `XMemMapPhysicalPages`, and no PC resource covers them. This module mirrors them for the GPU:
//!
//! * The pool is a set of D3D12 heaps of 16 MiB (custom, system memory, write back, made with
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
//!   Batches of at least 2 MiB use scoped copying threads sized to host parallelism and batch size.
//!   Each job covers at most 512 KiB within one section and heap. Threads join before tile mappings.
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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, RwLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCWSTR};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};

use crate::log::xs_log;
use crate::title_mirror::{Change, HEAP_PAGES, Mirror, PAGE, TileCall, heap_of};

/// A span within this many bytes of the end of its reservation is logged: a root view has no bounds
/// check, so a shader that reads past its data would read past the slot in the window.
pub const NEAR_END_BYTES: u64 = 64 << 10;

/// The size of a pool heap: 256 pages.
pub const POOL_HEAP_BYTES: u64 = HEAP_PAGES as u64 * PAGE;

/// Pages asked of `DecantXmemGpuTakeDirty` in one call.
const TAKE_PAGES: usize = 65536;
/// Bound each copy job to 256 KiB, parallelize batches of at least 512 KiB, and assign at least 512 KiB per copying thread.
const COPY_CHUNK_PAGES: usize = 4;
const PARALLEL_THRESHOLD_PAGES: usize = 8;
const COPY_THREAD_PAGES: usize = 8;

const SUBPAGE_BYTES: usize = 4096;
const SUBPAGES_PER_PAGE: usize = PAGE as usize / SUBPAGE_BYTES;

fn parse_subpage_diff(raw: Option<&str>) -> bool {
    !matches!(raw.map(str::trim), Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
}

fn subpage_diff_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| parse_subpage_diff(std::env::var("DECANT_D3D12X_SUBPAGE_DIFF").ok().as_deref()))
}

#[derive(Clone, Copy, Debug)]
struct CopyJob {
    source: usize,
    destination: usize,
    bytes: usize,
    first_tile: u32,
    staged: bool,
}

fn plan_copy(jobs: &mut Vec<CopyJob>, source: usize, destination: usize, pages: usize, staged: bool) {
    plan_copy_tiles(jobs, source, destination, pages, staged, 0);
}

fn plan_copy_tiles(jobs: &mut Vec<CopyJob>, source: usize, destination: usize, pages: usize, staged: bool, first_tile: u32) {
    for start in (0..pages).step_by(COPY_CHUNK_PAGES) {
        let count = (pages - start).min(COPY_CHUNK_PAGES);
        jobs.push(CopyJob {
            source: source + start * PAGE as usize,
            destination: destination + start * PAGE as usize,
            bytes: count * PAGE as usize,
            first_tile: first_tile + start as u32,
            staged,
        });
    }
}

/// Executes one copy job, returning `(copied_bytes, skipped_bytes, changed_page_mask)`.
/// For direct L0 destinations when `diff_enabled` is true, compares 4 KiB sub-pages and copies
/// only contiguous changed runs.
///
/// # Safety
/// `job.source` must be readable and `job.destination` writable for `job.bytes` bytes, and the
/// two ranges must not overlap.
unsafe fn run_copy_job(job: CopyJob, diff_enabled: bool) -> (u64, u64, u64) {
    if job.staged || !diff_enabled {
        // SAFETY: caller guarantees live disjoint ranges for `job.bytes`.
        unsafe { std::ptr::copy_nonoverlapping(job.source as *const u8, job.destination as *mut u8, job.bytes) };
        let pages = (job.bytes / PAGE as usize).min(64);
        let mask = if job.staged { 0 } else if pages == 64 { u64::MAX } else { (1u64 << pages) - 1 };
        return (job.bytes as u64, 0, mask);
    }
    let total_subpages = job.bytes / SUBPAGE_BYTES;
    let mut copied = 0u64;
    let mut skipped = 0u64;
    let mut changed_mask = 0u64;
    let mut run_start: Option<usize> = None;
    for sp in 0..total_subpages {
        let off = sp * SUBPAGE_BYTES;
        // SAFETY: `off + SUBPAGE_BYTES <= job.bytes`, both slices are live and disjoint.
        let same = unsafe {
            let s = std::slice::from_raw_parts((job.source + off) as *const u8, SUBPAGE_BYTES);
            let d = std::slice::from_raw_parts((job.destination + off) as *const u8, SUBPAGE_BYTES);
            s == d
        };
        if same {
            if let Some(start_sp) = run_start.take() {
                let run_off = start_sp * SUBPAGE_BYTES;
                let run_bytes = (sp - start_sp) * SUBPAGE_BYTES;
                // SAFETY: contiguous changed sub-pages within `job.bytes`.
                unsafe { std::ptr::copy_nonoverlapping((job.source + run_off) as *const u8, (job.destination + run_off) as *mut u8, run_bytes) };
                copied += run_bytes as u64;
            }
            skipped += SUBPAGE_BYTES as u64;
        } else {
            changed_mask |= 1u64 << (sp / SUBPAGES_PER_PAGE);
            if run_start.is_none() {
                run_start = Some(sp);
            }
        }
    }
    if let Some(start_sp) = run_start {
        let run_off = start_sp * SUBPAGE_BYTES;
        let run_bytes = (total_subpages - start_sp) * SUBPAGE_BYTES;
        // SAFETY: contiguous changed tail sub-pages within `job.bytes`.
        unsafe { std::ptr::copy_nonoverlapping((job.source + run_off) as *const u8, (job.destination + run_off) as *mut u8, run_bytes) };
        copied += run_bytes as u64;
    }
    (copied, skipped, changed_mask)
}

fn push_masked_tiles(job: CopyJob, mut mask: u64, out: &mut Vec<u32>) {
    if job.staged {
        return;
    }
    while mask != 0 {
        let p = mask.trailing_zeros();
        out.push(job.first_tile + p);
        mask &= mask - 1;
    }
}

/// Rebases staged L1 copy destinations by `dst_base` while leaving direct L0 pointers untouched.
/// Returns the total byte size of staged upload runs.
fn apply_staging_base(jobs: &mut [CopyJob], runs: &[UploadRun], dst_base: usize) -> usize {
    for job in jobs.iter_mut() {
        if job.staged {
            job.destination += dst_base;
        }
    }
    runs.iter().map(|r| r.bytes as usize).sum()
}

fn copy_threads(pages: usize, cores: usize) -> usize {
    if pages < PARALLEL_THRESHOLD_PAGES {
        return 1;
    }
    (pages / COPY_THREAD_PAGES).max(2).min(cores.max(1))
}

struct PoolState {
    jobs_ptr: usize,
    jobs_len: usize,
    masks_ptr: usize,
    diff_enabled: bool,
    workers_to_take: usize,
    active_workers: usize,
    epoch: u64,
}

struct PoolShared {
    state: Mutex<PoolState>,
    next: std::sync::atomic::AtomicUsize,
    copied_bytes: AtomicU64,
    skipped_bytes: AtomicU64,
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
            state: Mutex::new(PoolState { jobs_ptr: 0, jobs_len: 0, masks_ptr: 0, diff_enabled: true, workers_to_take: 0, active_workers: 0, epoch: 0 }),
            next: std::sync::atomic::AtomicUsize::new(0),
            copied_bytes: AtomicU64::new(0),
            skipped_bytes: AtomicU64::new(0),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        for i in 0..workers {
            let shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new().name(format!("title-mem-copy-{i}")).spawn(move || {
                let mut seen_gen = 0u64;
                loop {
                    let (ptr, len, masks_ptr, diff_enabled) = {
                        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        while st.epoch == seen_gen || st.workers_to_take == 0 {
                            st = shared.cv_work.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        seen_gen = st.epoch;
                        st.workers_to_take -= 1;
                        (st.jobs_ptr as *const CopyJob, st.jobs_len, st.masks_ptr as *const AtomicU64, st.diff_enabled)
                    };
                    let mut local_copied = 0u64;
                    let mut local_skipped = 0u64;
                    loop {
                        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
                        if idx >= len {
                            break;
                        }
                        // SAFETY: `run` keeps the `jobs` and `masks` slices alive until `active_workers` reaches 0.
                        let job = unsafe { *ptr.add(idx) };
                        // SAFETY: caller guarantees live disjoint source and destination ranges.
                        let (c, s, mask) = unsafe { run_copy_job(job, diff_enabled) };
                        if !masks_ptr.is_null() {
                            // SAFETY: `masks_ptr` has `len` elements and `idx < len` is unique to this worker.
                            unsafe { (*masks_ptr.add(idx)).store(mask, Ordering::Relaxed) };
                        }
                        local_copied += c;
                        local_skipped += s;
                    }
                    shared.copied_bytes.fetch_add(local_copied, Ordering::Relaxed);
                    shared.skipped_bytes.fetch_add(local_skipped, Ordering::Relaxed);
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
    unsafe fn run(&self, jobs: &[CopyJob], threads: usize, diff_enabled: bool, changed_tiles: &mut Vec<u32>) -> (u64, u64) {
        let _run_guard = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let worker_count = threads.saturating_sub(1).min(self.workers);
        if worker_count == 0 || jobs.len() <= 1 {
            let mut copied = 0u64;
            let mut skipped = 0u64;
            for &job in jobs {
                // SAFETY: caller guarantees live disjoint ranges for every job.
                let (c, s, mask) = unsafe { run_copy_job(job, diff_enabled) };
                push_masked_tiles(job, mask, changed_tiles);
                copied += c;
                skipped += s;
            }
            return (copied, skipped);
        }
        let masks: Vec<AtomicU64> = (0..jobs.len()).map(|_| AtomicU64::new(0)).collect();
        self.shared.copied_bytes.store(0, Ordering::Relaxed);
        self.shared.skipped_bytes.store(0, Ordering::Relaxed);
        {
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            st.jobs_ptr = jobs.as_ptr() as usize;
            st.jobs_len = jobs.len();
            st.masks_ptr = masks.as_ptr() as usize;
            st.diff_enabled = diff_enabled;
            self.shared.next.store(0, Ordering::Relaxed);
            st.workers_to_take = worker_count;
            st.active_workers = worker_count;
            st.epoch += 1;
            self.shared.cv_work.notify_all();
        }
        let mut main_copied = 0u64;
        let mut main_skipped = 0u64;
        loop {
            let idx = self.shared.next.fetch_add(1, Ordering::Relaxed);
            if idx >= jobs.len() {
                break;
            }
            // SAFETY: caller guarantees live disjoint ranges for every job.
            let (c, s, mask) = unsafe { run_copy_job(jobs[idx], diff_enabled) };
            masks[idx].store(mask, Ordering::Relaxed);
            main_copied += c;
            main_skipped += s;
        }
        let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.active_workers > 0 {
            st = self.shared.cv_done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        for (&job, m) in jobs.iter().zip(masks.iter()) {
            push_masked_tiles(job, m.load(Ordering::Relaxed), changed_tiles);
        }
        (
            main_copied + self.shared.copied_bytes.load(Ordering::Relaxed),
            main_skipped + self.shared.skipped_bytes.load(Ordering::Relaxed),
        )
    }
}

fn copy_pool(cores: usize) -> &'static CopyPool {
    static POOL: OnceLock<CopyPool> = OnceLock::new();
    POOL.get_or_init(|| CopyPool::new((cores / 2).clamp(2, 8)))
}

/// # Safety
/// Sources must be readable and destinations writable until this function returns.
/// All destination ranges must be disjoint and must not overlap any source.
unsafe fn execute_copies(jobs: &[CopyJob], cores: usize, diff_enabled: bool, changed_tiles: &mut Vec<u32>) -> (u64, u64) {
    let pages = jobs.iter().map(|job| job.bytes.div_ceil(PAGE as usize)).sum();
    let threads = copy_threads(pages, cores).min(jobs.len());
    if threads <= 1 {
        let mut copied = 0u64;
        let mut skipped = 0u64;
        for &job in jobs {
            // SAFETY: the caller guarantees live disjoint ranges for every job.
            let (c, s, mask) = unsafe { run_copy_job(job, diff_enabled) };
            push_masked_tiles(job, mask, changed_tiles);
            copied += c;
            skipped += s;
        }
        return (copied, skipped);
    }
    // SAFETY: caller guarantees live disjoint ranges for every job.
    unsafe { copy_pool(cores).run(jobs, threads, diff_enabled, changed_tiles) }
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
/// Active `State::id` for out-of-lock journal draining.
static ACTIVE_STATE_ID: AtomicU64 = AtomicU64::new(0);
/// Incremented whenever virtual to pool mappings change structurally (MAP, UNMAP, RELEASE, resync, device_created).
static MAPPING_GEN: AtomicU64 = AtomicU64::new(1);
/// Incremented whenever virtual to pool mappings are unmapped or reset (UNMAP, RELEASE, resync, device_created).
static UNMAP_GEN: AtomicU64 = AtomicU64::new(1);
/// Incremented whenever previously tileless pages receive mirror tiles.
static TILE_GEN: AtomicU64 = AtomicU64::new(1);

fn bump_mapping_gen() {
    MAPPING_GEN.fetch_add(1, Ordering::Release);
}

fn bump_unmap_gen() {
    UNMAP_GEN.fetch_add(1, Ordering::Release);
}

fn bump_tile_gen() {
    TILE_GEN.fetch_add(1, Ordering::Release);
}

/// Current generation of structural virtual to pool mappings.
pub fn mapping_generation() -> u64 {
    MAPPING_GEN.load(Ordering::Acquire)
}

/// Current generation of virtual to pool unmap or reset events.
pub fn unmap_generation() -> u64 {
    UNMAP_GEN.load(Ordering::Acquire)
}

/// Current generation of mirror tile assignments on mapped pages.
pub fn tile_generation() -> u64 {
    TILE_GEN.load(Ordering::Acquire)
}

static STREAMING_WORK: AtomicU64 = AtomicU64::new(0);

/// Records bulk streaming work (`detile` textures/bytes or `title_mem` pages/tile ranges) so
/// frame pacing can skip waits during heavy loading bursts.
pub(crate) fn note_streaming_work(units: u64) {
    if units > 0 {
        STREAMING_WORK.fetch_add(units, Ordering::Relaxed);
    }
}

pub(crate) fn take_streaming_work() -> u64 {
    STREAMING_WORK.swap(0, Ordering::Relaxed)
}

pub(crate) fn peek_streaming_work() -> u64 {
    STREAMING_WORK.load(Ordering::Relaxed)
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
type JournalSeqFn = unsafe extern "system" fn() -> u64;

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
    /// `DecantXmemJournalSeq`, which an older `xmem.dll` does not have.
    journal_seq: Option<JournalSeqFn>,
}

/// One pool heap: the buffer over it, where its memory is mapped (0 when in L1 VRAM), and its pool.
struct Heap {
    _heap: ID3D12Heap,
    buffer: ID3D12Resource,
    ptr: usize,
    is_l1: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MirrorPool {
    L0,
    L1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MirrorPoolMode {
    Auto,
    StrictL0,
    PreferL1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VramBudget {
    r_drv: u64,
    hwm_total: u64,
    lwm_total: u64,
    m_cap: u64,
}

impl VramBudget {
    /// Computes Design Note F VRAM reserves, watermarks, and L1 mirror ceiling from local budget `budget`.
    fn from_budget(budget: u64) -> Self {
        let budget_mib = budget >> 20;
        let r_drv = 1536u64.max(budget_mib * 16 / 100) << 20;
        let hwm_total = (budget_mib << 20).saturating_sub(r_drv);
        let delta_h = 1024u64.max(budget_mib / 10) << 20;
        let lwm_total = hwm_total.saturating_sub(delta_h);
        let m_cap = if budget_mib < 10_240 {
            0
        } else if budget_mib < 14_336 {
            1_024 << 20
        } else if budget_mib < 20_480 {
            2_048 << 20
        } else {
            4_096 << 20
        };
        Self { r_drv, hwm_total, lwm_total, m_cap }
    }
}

const STAGING_BYTES: usize = 64 << 20;
const STAGING_BYTES_32G: usize = 32 << 20;
const STAGING_MAX_PAGES: usize = STAGING_BYTES / PAGE as usize;
const STAGING_ALLOCATORS: usize = 8;
const HOST_RAM_32G_THRESHOLD: u64 = 33_554 << 20;
const COMMIT_FLOOR_BYTES: u64 = 4_096 << 20;

fn parse_mirror_l0_mib(env: Option<&str>) -> Option<u64> {
    env.and_then(|v| v.trim().parse::<u64>().ok()).map(|mib| mib << 20)
}

fn host_ram_status() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut mem = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `mem.dwLength` is initialized to the exact struct size.
    if unsafe { GlobalMemoryStatusEx(&mut mem) } != 0 {
        Some((mem.ullTotalPhys, mem.ullAvailPhys, mem.ullAvailPageFile))
    } else {
        None
    }
}

fn host_ram_l0_cap(
    total_phys: u64,
    avail_phys: u64,
    avail_pagefile: u64,
    override_cap: Option<u64>,
) -> u64 {
    if let Some(cap) = override_cap {
        return cap;
    }
    if total_phys <= HOST_RAM_32G_THRESHOLD {
        let base = if total_phys <= (16_384 << 20) {
            1_024 << 20
        } else {
            2_048 << 20
        };
        if avail_phys < (2_560 << 20) || avail_pagefile < (2_048 << 20) {
            base.min(512 << 20)
        } else if avail_phys < (4_096 << 20) || avail_pagefile < (4_096 << 20) {
            base.min(1_024 << 20)
        } else {
            base
        }
    } else {
        let twenty_pct = ((total_phys >> 20) * 20 / 100) << 20;
        (6_144 << 20).min(twenty_pct)
    }
}

fn staging_ring_bytes(total_phys: u64) -> usize {
    if total_phys > 0 && total_phys <= HOST_RAM_32G_THRESHOLD {
        STAGING_BYTES_32G
    } else {
        STAGING_BYTES
    }
}

fn can_grow_l0(
    l0_mirror_bytes: u64,
    l0_cap: u64,
    avail_phys: u64,
    avail_pagefile: u64,
) -> bool {
    l0_mirror_bytes.saturating_add(POOL_HEAP_BYTES) <= l0_cap
        && avail_phys >= COMMIT_FLOOR_BYTES
        && avail_pagefile >= (2_048 << 20)
}

fn parse_mirror_pool_mode(env: Option<&str>) -> MirrorPoolMode {
    if let Some(val) = env {
        let trimmed = val.trim();
        if trimmed.eq_ignore_ascii_case("l0") {
            return MirrorPoolMode::StrictL0;
        }
        if trimmed.eq_ignore_ascii_case("l1") {
            return MirrorPoolMode::PreferL1;
        }
    }
    MirrorPoolMode::Auto
}

fn parse_mirror_l1_mib(env: Option<&str>) -> Option<u64> {
    env.and_then(|v| v.trim().parse::<u64>().ok()).map(|mib| mib << 20)
}

fn should_use_l1(
    mode: MirrorPoolMode,
    budget: u64,
    current_usage: u64,
    l1_mirror_bytes: u64,
    m_cap_override: Option<u64>,
) -> bool {
    let vram = VramBudget::from_budget(budget);
    let m_cap = m_cap_override.unwrap_or(vram.m_cap);
    match mode {
        MirrorPoolMode::StrictL0 => false,
        MirrorPoolMode::Auto => {
            m_cap > 0
                && current_usage.saturating_add(POOL_HEAP_BYTES) <= vram.hwm_total
                && l1_mirror_bytes.saturating_add(POOL_HEAP_BYTES) <= m_cap
        }
        MirrorPoolMode::PreferL1 => {
            current_usage.saturating_add(POOL_HEAP_BYTES) <= vram.hwm_total
        }
    }
}

fn local_adapter(device: &ID3D12Device) -> Option<IDXGIAdapter3> {
    // SAFETY: plain DXGI factory and adapter queries on a live D3D12 device.
    let factory = unsafe { CreateDXGIFactory1::<IDXGIFactory4>() }.ok()?;
    // SAFETY: `device` is live and `GetAdapterLuid` returns its LUID by value.
    unsafe { factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()) }.ok()
}

fn local_vram_info(adapter: &IDXGIAdapter3) -> Option<(u64, u64)> {
    use windows::Win32::Graphics::Dxgi::{
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: `adapter` is live and `info` is plain output data.
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
    Some((info.Budget, info.CurrentUsage))
}

fn dedicated_vram(adapter: Option<&IDXGIAdapter3>) -> u64 {
    let Some(adapter) = adapter else { return 0 };
    // SAFETY: `adapter` is live.
    unsafe { adapter.GetDesc() }.map_or(0, |d| d.DedicatedVideoMemory as u64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UploadRun {
    source: usize,
    heap: u32,
    heap_offset: u64,
    staging_offset: u64,
    bytes: u64,
}

fn has_avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    })
}

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^ (k >> 33)
}

#[inline]
fn fold_lanes_to_u128(lanes: [u64; 8]) -> u128 {
    let mut h1 = 0x243F_6A88_85A3_08D3u64;
    let mut h2 = 0x1319_8A2E_0370_7344u64;
    let mut i = 0usize;
    while i < 8 {
        h1 = h1.wrapping_add(lanes[i]).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h1 = h1.rotate_left(27) ^ lanes[i + 1];
        h2 = h2.wrapping_add(lanes[i + 1]).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h2 = h2.rotate_left(31) ^ lanes[i];
        i += 2;
    }
    h1 = fmix64(h1 ^ h2.rotate_left(17));
    h2 = fmix64(h2 ^ h1.rotate_left(29));
    (u128::from(h2) << 64) | u128::from(h1)
}

/// Scalar 4-way unrolled 128-bit fingerprint of `SUBPAGE_BYTES`.
///
/// # Safety
/// `src` must be readable for `SUBPAGE_BYTES` bytes.
unsafe fn hash_subpage_scalar(src: *const u8) -> u128 {
    const CHUNKS: usize = SUBPAGE_BYTES / 32;
    let mut a0 = 0x243F_6A88_85A3_08D3u64;
    let mut a1 = 0x1319_8A2E_0370_7344u64;
    let mut a2 = 0xA409_3822_299F_31D0u64;
    let mut a3 = 0x082E_FA98_EC4E_6C89u64;
    let words = src.cast::<u64>();
    for i in 0..CHUNKS {
        let base = i * 4;
        // SAFETY: `(base + 3) * 8 + 8 <= SUBPAGE_BYTES`.
        let w0 = unsafe { words.add(base).read_unaligned() };
        let w1 = unsafe { words.add(base + 1).read_unaligned() };
        let w2 = unsafe { words.add(base + 2).read_unaligned() };
        let w3 = unsafe { words.add(base + 3).read_unaligned() };
        a0 = a0.wrapping_add(w0).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23);
        a1 = (a1 ^ w1).wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29);
        a2 = a2.wrapping_add(w2).wrapping_mul(0x94D0_49BB_1331_11EB).rotate_left(31);
        a3 = (a3 ^ w3).wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(19);
    }
    let h1 = fmix64(a0 ^ a2.rotate_left(13) ^ a1.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let h2 = fmix64(a1 ^ a3.rotate_left(17) ^ a0.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    (u128::from(h2) << 64) | u128::from(h1)
}

/// AVX2 256-bit vectorized 128-bit fingerprint of `SUBPAGE_BYTES` processing 64 bytes per step.
///
/// # Safety
/// Caller must ensure AVX2 is supported and `src` is readable for `SUBPAGE_BYTES` bytes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hash_subpage_avx2(src: *const u8) -> u128 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const ITERS: usize = SUBPAGE_BYTES / 64;
    let mut acc0 = _mm256_set_epi64x(
        0x243F_6A88_85A3_08D3u64 as i64,
        0x1319_8A2E_0370_7344u64 as i64,
        0xA409_3822_299F_31D0u64 as i64,
        0x082E_FA98_EC4E_6C89u64 as i64,
    );
    let mut acc1 = _mm256_set_epi64x(
        0x4528_21E6_38D0_1377u64 as i64,
        0xBE54_66CF_34E9_0C6Cu64 as i64,
        0xC0AC_29B7_C97C_50DDu64 as i64,
        0x3F84_D5B5_B547_0917u64 as i64,
    );
    let mul0 = _mm256_set_epi64x(
        0x9E37_79B9_7F4A_7C15u64 as i64,
        0xBF58_476D_1CE4_E5B9u64 as i64,
        0x94D0_49BB_1331_11EBu64 as i64,
        0xD6E8_FEB8_6659_FD93u64 as i64,
    );
    let mul1 = _mm256_set_epi64x(
        0xFF51_AFD7_ED55_8CCDu64 as i64,
        0xC4CE_B9FE_1A85_EC53u64 as i64,
        0x2545_F491_4F6C_DD1Du64 as i64,
        0x9E37_79B1_85EB_CA87u64 as i64,
    );

    for i in 0..ITERS {
        let ptr = unsafe { src.add(i * 64) };
        let v0 = unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) };
        let v1 = unsafe { _mm256_loadu_si256(ptr.add(32).cast::<__m256i>()) };

        acc0 = _mm256_add_epi64(acc0, v0);
        acc0 = _mm256_xor_si256(acc0, _mm256_srli_epi64(acc0, 32));
        acc0 = _mm256_mul_epu32(acc0, mul0);
        acc0 = _mm256_xor_si256(acc0, _mm256_srli_epi64(acc0, 29));

        acc1 = _mm256_add_epi64(acc1, v1);
        acc1 = _mm256_xor_si256(acc1, _mm256_srli_epi64(acc1, 32));
        acc1 = _mm256_mul_epu32(acc1, mul1);
        acc1 = _mm256_xor_si256(acc1, _mm256_srli_epi64(acc1, 31));
    }

    let mut lanes = [0u64; 8];
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc0);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(4).cast::<__m256i>(), acc1);
    }
    fold_lanes_to_u128(lanes)
}

/// Deterministic 128-bit fingerprint of a 4 KiB sub-page (`SUBPAGE_BYTES`).
///
/// # Safety
/// `src` must be readable for `SUBPAGE_BYTES` bytes.
unsafe fn hash_subpage(src: *const u8) -> u128 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if has_avx2() {
        // SAFETY: `has_avx2()` verified AVX2 support and caller guarantees `SUBPAGE_BYTES` readable.
        return unsafe { hash_subpage_avx2(src) };
    }
    // SAFETY: caller guarantees `src` is readable for `SUBPAGE_BYTES` bytes.
    unsafe { hash_subpage_scalar(src) }
}

#[derive(Default)]
struct L1HeapPrints {
    valid: [u64; 4],
    hashes: Vec<[u128; SUBPAGES_PER_PAGE]>,
}

impl L1HeapPrints {
    fn is_valid(&self, tile: usize) -> bool {
        (self.valid[tile >> 6] & (1u64 << (tile & 63))) != 0
    }

    fn set_valid(&mut self, tile: usize) {
        self.valid[tile >> 6] |= 1u64 << (tile & 63);
    }

    fn ensure_hashes(&mut self) -> &mut [[u128; SUBPAGES_PER_PAGE]] {
        if self.hashes.is_empty() {
            self.hashes.resize(HEAP_PAGES as usize, [0u128; SUBPAGES_PER_PAGE]);
        }
        &mut self.hashes
    }
}

#[derive(Default)]
struct L1Fingerprints {
    heaps: Vec<Option<Box<L1HeapPrints>>>,
}

impl L1Fingerprints {
    fn clear(&mut self) {
        self.heaps.clear();
    }

    fn heap_mut(&mut self, heap: usize) -> &mut L1HeapPrints {
        if self.heaps.len() <= heap {
            self.heaps.resize_with(heap + 1, || None);
        }
        self.heaps[heap].get_or_insert_with(|| Box::new(L1HeapPrints::default()))
    }
}

const DIFF_PARALLEL_MIN_TILES: usize = 64;

#[derive(Default)]
struct DiffScratch {
    tile_srcs: Vec<usize>,
    computed: Vec<[u128; SUBPAGES_PER_PAGE]>,
}

struct DiffState {
    srcs_ptr: usize,
    out_ptr: usize,
    total_tiles: usize,
    workers_to_take: usize,
    active_workers: usize,
    epoch: u64,
}

struct DiffShared {
    state: Mutex<DiffState>,
    next: std::sync::atomic::AtomicUsize,
    cv_work: Condvar,
    cv_done: Condvar,
}

struct DiffPool {
    workers: usize,
    scratch: Mutex<DiffScratch>,
    shared: Arc<DiffShared>,
}

impl DiffPool {
    fn new(workers: usize) -> Self {
        let shared = Arc::new(DiffShared {
            state: Mutex::new(DiffState {
                srcs_ptr: 0,
                out_ptr: 0,
                total_tiles: 0,
                workers_to_take: 0,
                active_workers: 0,
                epoch: 0,
            }),
            next: std::sync::atomic::AtomicUsize::new(0),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
        });
        for i in 0..workers {
            let shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new().name(format!("title-mem-diff-{i}")).spawn(move || {
                let mut seen_gen = 0u64;
                loop {
                    let (srcs_ptr, out_ptr, total_tiles) = {
                        let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                        while st.epoch == seen_gen || st.workers_to_take == 0 {
                            st = shared.cv_work.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        seen_gen = st.epoch;
                        st.workers_to_take -= 1;
                        (
                            st.srcs_ptr as *const usize,
                            st.out_ptr as *mut [u128; SUBPAGES_PER_PAGE],
                            st.total_tiles,
                        )
                    };
                    loop {
                        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
                        if idx >= total_tiles {
                            break;
                        }
                        // SAFETY: `hash_and_consume` keeps `tile_srcs` and `computed` alive until `active_workers` is 0.
                        unsafe { hash_tile_slot(*srcs_ptr.add(idx), &mut *out_ptr.add(idx)) };
                    }
                    let mut st = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    st.active_workers -= 1;
                    if st.active_workers == 0 {
                        shared.cv_done.notify_one();
                    }
                }
            });
        }
        Self { workers, scratch: Mutex::new(DiffScratch::default()), shared }
    }

    /// # Safety
    /// Every `UploadRun` in `raw` must have a live readable `source` for `bytes` bytes.
    unsafe fn hash_and_consume<R>(
        &self,
        raw: &[UploadRun],
        total_tiles: usize,
        cores: usize,
        consume: impl FnOnce(&[[u128; SUBPAGES_PER_PAGE]]) -> R,
    ) -> R {
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.tile_srcs.clear();
        scratch.tile_srcs.reserve(total_tiles);
        for run in raw {
            let tile_count = (run.bytes / PAGE) as usize;
            for t in 0..tile_count {
                scratch.tile_srcs.push(run.source + t * PAGE as usize);
            }
        }
        if scratch.computed.len() < total_tiles {
            scratch.computed.resize(total_tiles, [0u128; SUBPAGES_PER_PAGE]);
        }
        let threads = (total_tiles / 16).clamp(2, cores).min(total_tiles);
        let worker_count = threads.saturating_sub(1).min(self.workers);
        let srcs_ptr = scratch.tile_srcs.as_ptr();
        let out_ptr = scratch.computed.as_mut_ptr();
        if worker_count > 0 {
            {
                let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                st.srcs_ptr = srcs_ptr as usize;
                st.out_ptr = out_ptr as usize;
                st.total_tiles = total_tiles;
                self.shared.next.store(0, Ordering::Relaxed);
                st.workers_to_take = worker_count;
                st.active_workers = worker_count;
                st.epoch += 1;
                self.shared.cv_work.notify_all();
            }
            loop {
                let idx = self.shared.next.fetch_add(1, Ordering::Relaxed);
                if idx >= total_tiles {
                    break;
                }
                // SAFETY: `idx < total_tiles` is unique to this thread and `scratch` stays locked.
                unsafe { hash_tile_slot(*srcs_ptr.add(idx), &mut *out_ptr.add(idx)) };
            }
            let mut st = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.active_workers > 0 {
                st = self.shared.cv_done.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        } else {
            for idx in 0..total_tiles {
                // SAFETY: `idx < total_tiles` and `scratch` stays locked.
                unsafe { hash_tile_slot(*srcs_ptr.add(idx), &mut *out_ptr.add(idx)) };
            }
        }
        consume(&scratch.computed[..total_tiles])
    }
}

/// # Safety
/// `tile_src` must be readable for `PAGE` bytes.
#[inline]
unsafe fn hash_tile_slot(tile_src: usize, out_tile: &mut [u128; SUBPAGES_PER_PAGE]) {
    for (sp, slot) in out_tile.iter_mut().enumerate() {
        // SAFETY: `sp * SUBPAGE_BYTES + SUBPAGE_BYTES <= PAGE`.
        *slot = unsafe { hash_subpage((tile_src + sp * SUBPAGE_BYTES) as *const u8) };
    }
}

fn diff_pool(cores: usize) -> &'static DiffPool {
    static POOL: std::sync::OnceLock<DiffPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| DiffPool::new((cores / 2).clamp(2, 6)))
}

/// # Safety
/// Every `UploadRun` in `raw` must have a live readable `source` for `bytes` bytes.
unsafe fn apply_staged_runs(
    raw: &[UploadRun],
    prints: &mut L1Fingerprints,
    runs: &mut Vec<UploadRun>,
    jobs: &mut Vec<CopyJob>,
    parallel_hashes: Option<&[[u128; SUBPAGES_PER_PAGE]]>,
) -> (usize, u64) {
    let mut staged_bytes = 0usize;
    let mut skipped_bytes = 0u64;
    let mut hash_cursor = 0usize;
    for &run in raw {
        let hp = prints.heap_mut(run.heap as usize);
        let first_tile = (run.heap_offset / PAGE) as usize;
        let tile_count = (run.bytes / PAGE) as usize;
        let mut run_src = 0usize;
        let mut run_heap_off = 0u64;
        let mut run_bytes = 0usize;
        let flush_run = |runs: &mut Vec<UploadRun>, jobs: &mut Vec<CopyJob>, staged_bytes: &mut usize, r_src: usize, r_hoff: u64, r_bytes: usize| {
            if r_bytes == 0 {
                return;
            }
            let st_off = *staged_bytes;
            jobs.push(CopyJob { source: r_src, destination: st_off, bytes: r_bytes, staged: true, first_tile: 0 });
            runs.push(UploadRun {
                source: 0,
                heap: run.heap,
                heap_offset: r_hoff,
                staging_offset: st_off as u64,
                bytes: r_bytes as u64,
            });
            *staged_bytes += r_bytes;
        };
        for t in 0..tile_count {
            let tile = first_tile + t;
            let tile_src = run.source + t * PAGE as usize;
            let tile_hoff = run.heap_offset + t as u64 * PAGE;
            let was_valid = hp.is_valid(tile);
            let hashes = hp.ensure_hashes();
            let tile_hashes = &mut hashes[tile];
            let precomputed = parallel_hashes.map(|v| &v[hash_cursor + t]);
            for (sp, slot) in tile_hashes.iter_mut().enumerate() {
                let sp_src = tile_src + sp * SUBPAGE_BYTES;
                let h = match precomputed {
                    Some(arr) => arr[sp],
                    // SAFETY: caller guarantees `run.source..run.source + run.bytes` is readable.
                    None => unsafe { hash_subpage(sp_src as *const u8) },
                };
                let changed = !was_valid || *slot != h;
                if changed {
                    *slot = h;
                    let sp_hoff = tile_hoff + sp as u64 * SUBPAGE_BYTES as u64;
                    if run_bytes > 0 && run_src + run_bytes == sp_src {
                        run_bytes += SUBPAGE_BYTES;
                    } else {
                        flush_run(runs, jobs, &mut staged_bytes, run_src, run_heap_off, run_bytes);
                        run_src = sp_src;
                        run_heap_off = sp_hoff;
                        run_bytes = SUBPAGE_BYTES;
                    }
                } else {
                    flush_run(runs, jobs, &mut staged_bytes, run_src, run_heap_off, run_bytes);
                    run_bytes = 0;
                    skipped_bytes += SUBPAGE_BYTES as u64;
                }
            }
            if !was_valid {
                hp.set_valid(tile);
            }
        }
        hash_cursor += tile_count;
        flush_run(runs, jobs, &mut staged_bytes, run_src, run_heap_off, run_bytes);
    }
    (staged_bytes, skipped_bytes)
}

/// Filters staged L1 `runs` to contiguous changed 4 KiB sub-page runs using `prints` when
/// `diff_enabled` is true, appending staged `CopyJob`s to `jobs` and returning
/// `(staged_bytes, skipped_bytes)`. Hashes sub-pages on a persistent worker pool when
/// `>= DIFF_PARALLEL_MIN_TILES` pages are staged.
///
/// # Safety
/// Every `UploadRun` in `runs` must have a live readable `source` for `bytes` bytes.
unsafe fn filter_staged_runs(
    runs: &mut Vec<UploadRun>,
    prints: &mut L1Fingerprints,
    jobs: &mut Vec<CopyJob>,
    diff_enabled: bool,
    cores: usize,
) -> (usize, u64) {
    if runs.is_empty() {
        return (0, 0);
    }
    let _trace = decant_trace::span("title_mem", "subpage_diff");
    if !diff_enabled {
        let mut staged_bytes = 0usize;
        for run in runs.iter_mut() {
            run.staging_offset = staged_bytes as u64;
            let pages = (run.bytes / PAGE) as usize;
            plan_copy(jobs, run.source, staged_bytes, pages, true);
            staged_bytes += run.bytes as usize;
        }
        return (staged_bytes, 0);
    }
    let raw = std::mem::take(runs);
    let total_tiles: usize = raw.iter().map(|r| (r.bytes / PAGE) as usize).sum();
    if total_tiles >= DIFF_PARALLEL_MIN_TILES && cores > 1 {
        // SAFETY: caller guarantees every `UploadRun` in `raw` has a live readable `source`.
        unsafe {
            diff_pool(cores).hash_and_consume(&raw, total_tiles, cores, |precomputed| {
                apply_staged_runs(&raw, prints, runs, jobs, Some(precomputed))
            })
        }
    } else {
        // SAFETY: caller guarantees every `UploadRun` in `raw` has a live readable `source`.
        unsafe { apply_staged_runs(&raw, prints, runs, jobs, None) }
    }
}

fn collect_changed_l1_tiles(runs: &[UploadRun], out: &mut Vec<u32>) {
    for run in runs {
        if run.bytes == 0 {
            continue;
        }
        let first_tile = run.heap * HEAP_PAGES + (run.heap_offset / PAGE) as u32;
        let last_tile = run.heap * HEAP_PAGES + ((run.heap_offset + run.bytes - 1) / PAGE) as u32;
        for t in first_tile..=last_tile {
            if out.last().copied() != Some(t) {
                out.push(t);
            }
        }
    }
}

struct StagingRing {
    buffer: ID3D12Resource,
    ptr: usize,
    size: usize,
    head: usize,
    in_flight: VecDeque<(usize, usize, u64)>,
    allocators: Vec<ID3D12CommandAllocator>,
    alloc_fences: Vec<u64>,
    list: ID3D12GraphicsCommandList,
    slot: usize,
    /// `HANDLE` stored as `usize` so `StagingRing` is `Send + Sync` inside `STATE`.
    event: usize,
}

impl Drop for StagingRing {
    fn drop(&mut self) {
        if self.event != 0 {
            // SAFETY: `event` was created by `CreateEventW` and is owned by `StagingRing`.
            unsafe {
                let _ = CloseHandle(HANDLE(self.event as *mut c_void));
            }
        }
    }
}

fn wait_fence(fence: &ID3D12Fence, event: usize, target: u64) {
    if target == 0 || event == 0 {
        return;
    }
    // SAFETY: `fence` and `event` are live; `GetCompletedValue` and `SetEventOnCompletion` are plain D3D12 calls.
    unsafe {
        if fence.GetCompletedValue() >= target {
            return;
        }
        if fence.SetEventOnCompletion(target, HANDLE(event as *mut c_void)).is_ok() {
            WaitForSingleObject(HANDLE(event as *mut c_void), 5000);
        }
    }
}

impl StagingRing {
    fn new(device: &ID3D12Device, size_bytes: usize) -> windows::core::Result<Self> {
        let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain D3D12 upload buffer creation on a live device.
        unsafe {
            device.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc(size_bytes as u64, D3D12_RESOURCE_FLAG_NONE),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut buffer,
            )
        }?;
        let fail = || windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32));
        let buffer = buffer.ok_or_else(fail)?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: upload buffer mapped for the life of the ring.
        unsafe { buffer.Map(0, None, Some(&mut ptr)) }?;
        // SAFETY: live resource.
        let _ = unsafe { buffer.SetName(&windows::core::HSTRING::from("decant title mirror staging")) };
        let mut allocators = Vec::with_capacity(STAGING_ALLOCATORS);
        for _ in 0..STAGING_ALLOCATORS {
            // SAFETY: plain D3D12 command allocator creation.
            let alloc: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }?;
            allocators.push(alloc);
        }
        // SAFETY: plain D3D12 command list creation and immediate close.
        let list: ID3D12GraphicsCommandList = unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocators[0], None) }?;
        // SAFETY: list was just opened above.
        unsafe { list.Close() }?;
        // SAFETY: plain Win32 auto-reset event.
        let event = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }?;
        Ok(Self {
            buffer,
            ptr: ptr as usize,
            size: size_bytes,
            head: 0,
            in_flight: VecDeque::new(),
            allocators,
            alloc_fences: vec![0; STAGING_ALLOCATORS],
            list,
            slot: 0,
            event: event.0 as usize,
        })
    }

    fn reserve(&mut self, bytes: usize, fence: &ID3D12Fence) -> Option<usize> {
        if bytes == 0 {
            return Some(self.head);
        }
        if bytes > self.size {
            return None;
        }
        if self.head + bytes > self.size {
            self.head = 0;
        }
        let start = self.head;
        let end = start + bytes;
        let mut wait_val = 0u64;
        for &(f_start, f_len, f_val) in &self.in_flight {
            if f_start < end && start < f_start + f_len {
                wait_val = wait_val.max(f_val);
            }
        }
        if wait_val > 0 {
            wait_fence(fence, self.event, wait_val);
            self.in_flight.retain(|&(f_start, f_len, f_val)| f_val > wait_val && !(f_start < end && start < f_start + f_len));
        }
        self.head = end;
        Some(start)
    }

    fn copy_to_heaps(
        &mut self,
        queue: &ID3D12CommandQueue,
        fence: &ID3D12Fence,
        heaps: &[Option<ID3D12Resource>],
        span: (usize, usize),
        runs: &[UploadRun],
        fence_value: &AtomicU64,
    ) -> bool {
        let (base_offset, total_bytes) = span;
        if runs.is_empty() {
            return false;
        }
        wait_fence(fence, self.event, self.alloc_fences[self.slot]);
        let alloc = &self.allocators[self.slot];
        let next_val;
        // SAFETY: `alloc_fences[slot]` has completed above, and `list` is closed before reset.
        unsafe {
            if alloc.Reset().is_err() || self.list.Reset(alloc, None).is_err() {
                return false;
            }
            for run in runs {
                if let Some(Some(dst)) = heaps.get(run.heap as usize) {
                    self.list.CopyBufferRegion(
                        dst,
                        run.heap_offset,
                        &self.buffer,
                        base_offset as u64 + run.staging_offset,
                        run.bytes,
                    );
                }
            }
            if self.list.Close().is_err() {
                return false;
            }
            let Ok(cmd) = self.list.cast::<ID3D12CommandList>() else { return false };
            queue.ExecuteCommandLists(&[Some(cmd)]);
            next_val = fence_value.fetch_add(1, Ordering::AcqRel) + 1;
            let _ = queue.Signal(fence, next_val);
        }
        self.in_flight.push_back((base_offset, total_bytes, next_val));
        self.alloc_fences[self.slot] = next_val;
        self.slot = (self.slot + 1) % self.allocators.len();
        true
    }
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
    copied_bytes: u64,
    skipped_bytes: u64,
    ranges: u64,
    micros: u64,
    worst_micros: u64,
    /// Of `micros`: asking `xmem.dll` for the written pages, copying them, updating the tile mappings.
    take_micros: u64,
    copy_micros: u64,
    tile_micros: u64,
}

#[derive(Default)]
struct TileScratch {
    coords: Vec<D3D12_TILED_RESOURCE_COORDINATE>,
    sizes: Vec<D3D12_TILE_REGION_SIZE>,
    counts: Vec<u32>,
    flags: Vec<D3D12_TILE_RANGE_FLAGS>,
}

struct State {
    /// Distinguishes a replacement device while the sampler reads outside the lock.
    id: u64,
    device: ID3D12Device,
    /// The private queue that applies the tile mappings, and the fence that says when.
    map_queue: ID3D12CommandQueue,
    map_fence: ID3D12Fence,
    fence_value: Arc<AtomicU64>,
    mirror: Mirror,
    /// The sequence number of the last journal record applied.
    seq: u64,
    /// Last `DecantXmemJournalSeq` observed before draining `xmem.dll`.
    last_journal_seq: Option<u64>,
    /// Pool heaps by index. A heap that cannot be made is `None` and is not tried again.
    heaps: Vec<Option<Heap>>,
    adapter: Option<IDXGIAdapter3>,
    pool_mode: MirrorPoolMode,
    fallback_budget: u64,
    m_cap_override: Option<u64>,
    l0_cap_override: Option<u64>,
    l1_heaps: u64,
    l0_heaps: u64,
    has_staging: bool,
    /// The heaps that could not be made.
    failed: HashSet<u32>,
    windows: Vec<Option<Window>>,
    /// The read only section views of `xmem.dll`: address, first pool page, pages.
    sections: Vec<(usize, u32, u32)>,
    /// Index of the most recently matched section in `sections`.
    last_section: usize,
    tracking: bool,
    /// Rotating zero-page observations, advanced only by the memory sampler.
    zero_scan: ZeroScan,
    /// Reused between syncs.
    scratch: Vec<GpuChange>,
    /// The mirror tiles of the pages the current sync copied: what the title wrote since the last one.
    written: Vec<u32>,
}

impl Drop for State {
    fn drop(&mut self) {
        if self.l1_heaps > 0 {
            crate::mem_owners::remove(
                crate::mem_owners::Kind::TitleMirrorHeap,
                crate::mem_owners::Segment::Local,
                self.l1_heaps,
                self.l1_heaps * POOL_HEAP_BYTES,
            );
        }
        if self.l0_heaps > 0 {
            crate::mem_owners::remove(
                crate::mem_owners::Kind::TitleMirrorHeap,
                crate::mem_owners::Segment::System,
                self.l0_heaps,
                self.l0_heaps * POOL_HEAP_BYTES,
            );
        }
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

static STATE: RwLock<Option<State>> = RwLock::new(None);
/// Serializes `DecantXmemGpuChanges` and `DecantXmemGpuSnapshot` calls outside `STATE`.
static DRAIN_LOCK: Mutex<DrainCursor> = Mutex::new(DrainCursor {
    state_id: 0,
    seq: 0,
    last_journal_seq: None,
    tracking: false,
    scratch: Vec::new(),
    records: Vec::new(),
    known_res: std::collections::BTreeMap::new(),
});

struct DrainCursor {
    state_id: u64,
    seq: u64,
    last_journal_seq: Option<u64>,
    tracking: bool,
    scratch: Vec<GpuChange>,
    records: Vec<GpuChange>,
    known_res: std::collections::BTreeMap<u64, u64>,
}

enum DrainBatch {
    Unchanged,
    Changes { state_id: u64, seq: u64, journal_seq: Option<u64> },
    Snapshot { state_id: u64, seq: u64, journal_seq: Option<u64> },
}

fn read_snapshot(api: &Api, out: &mut Vec<GpuChange>) -> u64 {
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
    out.clear();
    out.extend_from_slice(&records);
    seq
}

/// Drains `xmem.dll` journal records under `DRAIN_LOCK` without holding `STATE`.
fn drain_journal(api: &Api, cursor: &mut DrainCursor) -> DrainBatch {
    let active_id = ACTIVE_STATE_ID.load(Ordering::Acquire);
    if active_id == 0 {
        return DrainBatch::Unchanged;
    }
    if cursor.state_id != active_id {
        cursor.state_id = active_id;
        cursor.seq = 0;
        cursor.last_journal_seq = None;
        cursor.known_res.clear();
    }
    let cur_seq = api.journal_seq.map(|f| {
        // SAFETY: `DecantXmemJournalSeq` is a lock-free atomic load returning `u64`.
        unsafe { f() }
    });
    if should_skip_drain(cursor.last_journal_seq, cur_seq) {
        return DrainBatch::Unchanged;
    }
    let _trace = decant_trace::span("title_mem", "drain");
    cursor.records.clear();
    let mut is_snapshot = false;
    loop {
        let mut written = 0u32;
        cursor.scratch.resize(CHANGE_RECORDS, GpuChange::default());
        // SAFETY: `scratch` has room for `CHANGE_RECORDS` records, `written` is a valid location.
        let ok = unsafe { (api.changes)(cursor.seq, cursor.scratch.as_mut_ptr(), CHANGE_RECORDS as u32, &mut written) };
        if ok == 0 {
            cursor.seq = read_snapshot(api, &mut cursor.records);
            is_snapshot = true;
            continue;
        }
        let n = (written as usize).min(CHANGE_RECORDS);
        cursor.records.extend_from_slice(&cursor.scratch[..n]);
        if let Some(last) = cursor.scratch[..n].last() {
            cursor.seq = last.seq;
        }
        if n < CHANGE_RECORDS {
            break;
        }
    }
    cursor.last_journal_seq = cur_seq;
    if is_snapshot {
        DrainBatch::Snapshot { state_id: active_id, seq: cursor.seq, journal_seq: cur_seq }
    } else {
        DrainBatch::Changes { state_id: active_id, seq: cursor.seq, journal_seq: cur_seq }
    }
}

/// Last `DecantXmemJournalSeq` applied to `STATE` (`u64::MAX` when unknown).
static LAST_JOURNAL_SEQ: AtomicU64 = AtomicU64::new(u64::MAX);
/// Last `DecantXmemJournalSeq` drained into `PENDING_JOURNAL` (`u64::MAX` when unknown).
static DRAINED_JOURNAL_SEQ: AtomicU64 = AtomicU64::new(u64::MAX);
/// `ACTIVE_STATE_ID` associated with `DRAINED_JOURNAL_SEQ`.
static DRAINED_STATE_ID: AtomicU64 = AtomicU64::new(0);
/// True while `PENDING_JOURNAL` is non-empty or a drained batch is being applied under `APPLY_LOCK`.
static HAS_PENDING_JOURNAL: AtomicBool = AtomicBool::new(false);
/// True when `Mirror` may have un-drained `TileCall` updates for `sync()`.
static MIRROR_MAY_HAVE_PENDING: AtomicBool = AtomicBool::new(false);
/// True once `DecantXmemGpuTrack(1)` has been enabled, so `with_state` can skip `DRAIN_LOCK`.
static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
/// True when `Mirror` has added slots whose window buffers have not been confirmed built yet.
static MISSING_WINDOWS: AtomicBool = AtomicBool::new(false);

struct PendingBatch {
    state_id: u64,
    snapshot: bool,
    seq: u64,
    journal_seq: Option<u64>,
    records: Vec<GpuChange>,
}

/// Queue of coalesced journal batches drained under `DRAIN_LOCK` and waiting for `APPLY_LOCK`.
static PENDING_JOURNAL: Mutex<Vec<PendingBatch>> = Mutex::new(Vec::new());
/// Serializes `Mirror` batch application under `STATE.write()` without holding `DRAIN_LOCK`.
static APPLY_LOCK: Mutex<()> = Mutex::new(());

fn journal_unchanged(api: &Api) -> bool {
    if !TRACKING_ACTIVE.load(Ordering::Acquire) {
        return false;
    }
    let Some(get_seq) = api.journal_seq else { return false };
    // SAFETY: `DecantXmemJournalSeq` is a lock-free atomic load returning `u64`.
    let cur_seq = unsafe { get_seq() };
    let last = LAST_JOURNAL_SEQ.load(Ordering::Acquire);
    last != u64::MAX && last == cur_seq
}

fn drained_journal_unchanged(api: &Api, active_id: u64) -> bool {
    if !TRACKING_ACTIVE.load(Ordering::Acquire) || DRAINED_STATE_ID.load(Ordering::Acquire) != active_id {
        return false;
    }
    let Some(get_seq) = api.journal_seq else { return false };
    // SAFETY: `DecantXmemJournalSeq` is a lock-free atomic load returning `u64`.
    let cur_seq = unsafe { get_seq() };
    let drained = DRAINED_JOURNAL_SEQ.load(Ordering::Acquire);
    drained != u64::MAX && drained == cur_seq
}

fn known_holding(res_bounds: &std::collections::BTreeMap<u64, u64>, va: u64) -> Option<(u64, u64)> {
    let (&base, &end) = res_bounds.range(..=va).next_back()?;
    (va < end).then_some((base, end))
}

fn record_reserve(res_bounds: &mut std::collections::BTreeMap<u64, u64>, va: u64, pages: u32) -> u64 {
    let new_end = va.saturating_add(u64::from(pages).saturating_mul(PAGE));
    let stale: Vec<u64> = res_bounds
        .iter()
        .filter(|&(&b, &e)| b < new_end && e > va)
        .map(|(&b, _)| b)
        .collect();
    for b in stale {
        res_bounds.remove(&b);
    }
    res_bounds.insert(va, new_end);
    new_end
}

/// Flushes a contiguous run of `UNMAP` records at the tail of `out`: because `UNMAP` commutes
/// with `UNMAP`, known-reservation runs can be sorted by `(base, va)` and merged into minimal spans.
fn flush_unmap_run(out: &mut Vec<GpuChange>, run_start: usize, res_bounds: &std::collections::BTreeMap<u64, u64>) {
    if out.len() - run_start < 2 {
        return;
    }
    let slice = &out[run_start..];
    let mut known: Vec<(u64, u64, u64, GpuChange)> = Vec::with_capacity(slice.len());
    let mut unknown: Vec<GpuChange> = Vec::new();
    for &c in slice {
        if c.pages > 0
            && let Some((base, res_end)) = known_holding(res_bounds, c.va)
            && let Some(end) = c.va.checked_add(u64::from(c.pages) * PAGE)
        {
            known.push((base, c.va, end.min(res_end), c));
        } else {
            unknown.push(c);
        }
    }
    if known.len() < 2 {
        return;
    }
    known.sort_unstable_by_key(|&(base, va, _, c)| (base, va, c.seq));
    out.truncate(run_start);
    let mut cur: Option<(u64, u64, u64, GpuChange)> = None;
    for (base, va, end, c) in known {
        match &mut cur {
            Some((c_base, c_va, c_end, prev)) if *c_base == base && va <= *c_end => {
                *c_end = (*c_end).max(end);
                let merged = ((*c_end - *c_va) / PAGE).min(u64::from(u32::MAX)) as u32;
                prev.pages = merged;
                prev.seq = prev.seq.max(c.seq);
            }
            _ => {
                if let Some((_, _, _, prev)) = cur.take() {
                    out.push(prev);
                }
                let mut first = c;
                first.pages = ((end - va) / PAGE).min(u64::from(u32::MAX)) as u32;
                cur = Some((base, va, end, first));
            }
        }
    }
    if let Some((_, _, _, prev)) = cur {
        out.push(prev);
    }
    out.extend(unknown);
}

/// Flushes a contiguous run of `MAP` records at the tail of `out`: when every entry in the run
/// falls within `res_bounds` and has strictly disjoint VAs, sorting by `(base, va)` merges adjacent
/// `(va, first_page)` scatter entries without altering semantics.
fn flush_map_run(out: &mut Vec<GpuChange>, run_start: usize, res_bounds: &std::collections::BTreeMap<u64, u64>) {
    if out.len() - run_start < 2 {
        return;
    }
    let slice = &out[run_start..];
    let mut items: Vec<(u64, u64, u64, GpuChange)> = Vec::with_capacity(slice.len());
    for &c in slice {
        let Some((base, res_end)) = known_holding(res_bounds, c.va) else { return };
        let Some(end) = c.va.checked_add(u64::from(c.pages) * PAGE) else { return };
        if c.pages == 0 || end > res_end {
            return;
        }
        items.push((base, c.va, end, c));
    }
    items.sort_unstable_by_key(|&(base, va, _, _)| (base, va));
    // Reordering is safe only if every pair of MAP intervals in this run is disjoint.
    for w in items.windows(2) {
        if w[0].2 > w[1].1 {
            return;
        }
    }
    out.truncate(run_start);
    for (base, va, end, c) in items {
        if let Some(prev) = out[run_start..].last_mut()
            && prev.va.checked_add(u64::from(prev.pages) * PAGE) == Some(va)
            && prev.first_page.checked_add(prev.pages) == Some(c.first_page)
            && let Some(merged_pages) = prev.pages.checked_add(c.pages)
            && known_holding(res_bounds, prev.va).is_some_and(|(b0, e0)| b0 == base && end <= e0)
        {
            prev.pages = merged_pages;
            prev.seq = prev.seq.max(c.seq);
            continue;
        }
        out.push(c);
    }
}

/// Compacts drained journal records before acquiring `STATE.write()` while keeping `res_bounds`
/// synchronized across batches:
/// - merges contiguous and disjoint `MAP` runs inside known reservations
/// - merges contiguous and unsorted `UNMAP` runs inside known reservations
/// - collapses `RESERVE(va)` + intra-`va` `MAP`/`UNMAP` + `RELEASE(va)` in one batch into `RELEASE(va)`.
fn coalesce_changes(records: &mut Vec<GpuChange>, res_bounds: &mut std::collections::BTreeMap<u64, u64>, is_snapshot: bool) {
    if is_snapshot {
        res_bounds.clear();
    }
    if records.len() < 2 {
        for &c in records.iter() {
            match c.kind {
                RESERVE => {
                    record_reserve(res_bounds, c.va, c.pages);
                }
                RELEASE => {
                    res_bounds.remove(&c.va);
                }
                _ => {}
            }
        }
        return;
    }
    let mut out: Vec<GpuChange> = Vec::with_capacity(records.len());
    let mut open_res: HashMap<u64, (usize, bool)> = HashMap::new();
    let mut active_run: Option<(u32, usize)> = None;

    let flush_run = |out: &mut Vec<GpuChange>, run: &mut Option<(u32, usize)>, bounds: &std::collections::BTreeMap<u64, u64>| {
        if let Some((kind, start)) = run.take() {
            if kind == UNMAP {
                flush_unmap_run(out, start, bounds);
            } else if kind == MAP {
                flush_map_run(out, start, bounds);
            }
        }
    };

    for &c in records.iter() {
        match c.kind {
            RESERVE => {
                flush_run(&mut out, &mut active_run, res_bounds);
                let new_end = c.va.saturating_add(u64::from(c.pages).saturating_mul(PAGE));
                let stale: Vec<u64> = res_bounds
                    .iter()
                    .filter(|&(&b, &e)| b < new_end && e > c.va)
                    .map(|(&b, _)| b)
                    .collect();
                for b in stale {
                    res_bounds.remove(&b);
                    open_res.remove(&b);
                }
                for (_, pure) in open_res.values_mut() {
                    *pure = false;
                }
                res_bounds.insert(c.va, new_end);
                open_res.insert(c.va, (out.len(), true));
                out.push(c);
            }
            RELEASE => {
                flush_run(&mut out, &mut active_run, res_bounds);
                res_bounds.remove(&c.va);
                if let Some((idx, pure)) = open_res.remove(&c.va)
                    && pure
                    && idx < out.len()
                    && out[idx].kind == RESERVE
                    && out[idx].va == c.va
                {
                    out.truncate(idx);
                    out.push(c);
                    continue;
                }
                for (_, pure) in open_res.values_mut() {
                    *pure = false;
                }
                out.push(c);
            }
            MAP => {
                if active_run.is_some_and(|(k, _)| k != MAP) {
                    flush_run(&mut out, &mut active_run, res_bounds);
                }
                if active_run.is_none() {
                    active_run = Some((MAP, out.len()));
                }
                let run_start = active_run.map_or(out.len(), |(_, s)| s);
                let target_base = known_holding(res_bounds, c.va).map(|(b, _)| b);
                for (&base, (_, pure)) in open_res.iter_mut() {
                    if target_base != Some(base) {
                        *pure = false;
                    }
                }
                if c.pages > 0
                    && out.len() > run_start
                    && let Some(prev) = out.last_mut()
                    && prev.kind == MAP
                    && prev.pages > 0
                    && prev.va.checked_add(u64::from(prev.pages) * PAGE) == Some(c.va)
                    && prev.first_page.checked_add(prev.pages) == Some(c.first_page)
                    && let Some(merged_pages) = prev.pages.checked_add(c.pages)
                    && let Some((base, end)) = known_holding(res_bounds, prev.va)
                    && known_holding(res_bounds, c.va).is_some_and(|(b2, _)| b2 == base)
                    && c.va.checked_add(u64::from(c.pages) * PAGE).is_some_and(|e| e <= end)
                {
                    prev.pages = merged_pages;
                    prev.seq = prev.seq.max(c.seq);
                    continue;
                }
                out.push(c);
            }
            UNMAP => {
                if active_run.is_some_and(|(k, _)| k != UNMAP) {
                    flush_run(&mut out, &mut active_run, res_bounds);
                }
                if active_run.is_none() {
                    active_run = Some((UNMAP, out.len()));
                }
                let run_start = active_run.map_or(out.len(), |(_, s)| s);
                let target_base = known_holding(res_bounds, c.va).map(|(b, _)| b);
                for (&base, (_, pure)) in open_res.iter_mut() {
                    if target_base != Some(base) {
                        *pure = false;
                    }
                }
                if c.pages > 0
                    && out.len() > run_start
                    && let Some(prev) = out.last_mut()
                    && prev.kind == UNMAP
                    && prev.pages > 0
                    && prev.va.checked_add(u64::from(prev.pages) * PAGE) == Some(c.va)
                    && let Some(merged_pages) = prev.pages.checked_add(c.pages)
                    && let Some((base, end)) = known_holding(res_bounds, prev.va)
                    && known_holding(res_bounds, c.va).is_some_and(|(b2, _)| b2 == base)
                    && c.va.checked_add(u64::from(c.pages) * PAGE).is_some_and(|e| e <= end)
                {
                    prev.pages = merged_pages;
                    prev.seq = prev.seq.max(c.seq);
                    continue;
                }
                out.push(c);
            }
            _ => {
                flush_run(&mut out, &mut active_run, res_bounds);
                out.push(c);
            }
        }
    }
    flush_run(&mut out, &mut active_run, res_bounds);

    *records = out;
}

struct CachedSyncState {
    state_id: u64,
    map_queue: ID3D12CommandQueue,
    map_fence: ID3D12Fence,
    fence_value: Arc<AtomicU64>,
}

struct SyncScratch {
    jobs: Vec<CopyJob>,
    dirty: Vec<u32>,
    runs: Vec<UploadRun>,
    staging: Option<StagingRing>,
    l1_prints: L1Fingerprints,
    /// The map fence value each submitting queue has waited for.
    waited: Option<HashMap<usize, u64>>,
    /// Cached device queue/fence handles for `state_id`, cleared on `device_created`.
    cached: Option<CachedSyncState>,
    stats: Stats,
    last_guard: Option<Instant>,
}

/// Serializes `sync` and `device_created` across queues so `execute_copies` and staging GPU
/// submissions run without holding `STATE`, and reuses buffers across syncs.
static SYNC_LOCK: Mutex<SyncScratch> = Mutex::new(SyncScratch {
    jobs: Vec::new(),
    dirty: Vec::new(),
    runs: Vec::new(),
    staging: None,
    l1_prints: L1Fingerprints { heaps: Vec::new() },
    waited: None,
    cached: None,
    stats: Stats {
        syncs: 0,
        pages: 0,
        copied_bytes: 0,
        skipped_bytes: 0,
        ranges: 0,
        micros: 0,
        take_micros: 0,
        copy_micros: 0,
        tile_micros: 0,
        worst_micros: 0,
    },
    last_guard: None,
});

/// Serializes window buffer creation outside `STATE`.
static WINDOW_MAKE_LOCK: Mutex<()> = Mutex::new(());
/// Serializes pool heap creation outside `STATE`.
static HEAP_MAKE_LOCK: Mutex<()> = Mutex::new(());
/// Serializes `UpdateTileMappings` + `Signal` on `map_queue` outside `STATE`.
static MAP_QUEUE_LOCK: Mutex<TileScratch> = Mutex::new(TileScratch {
    coords: Vec::new(),
    sizes: Vec::new(),
    counts: Vec::new(),
    flags: Vec::new(),
});

fn should_skip_drain(last_seq: Option<u64>, current_seq: Option<u64>) -> bool {
    matches!((last_seq, current_seq), (Some(a), Some(b)) if a == b)
}
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
                    journal_seq: get(c"DecantXmemJournalSeq").map(|f| std::mem::transmute::<unsafe extern "system" fn() -> isize, JournalSeqFn>(f)),
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

/// A pool heap, the buffer over it and its mapped memory (or 0 when in L1 VRAM).
fn make_heap(device: &ID3D12Device, index: u32, pool: MirrorPool) -> windows::core::Result<Heap> {
    let props = match pool {
        MirrorPool::L0 => D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        MirrorPool::L1 => D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
    };
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: POOL_HEAP_BYTES,
        Properties: props,
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
    let ptr = if pool == MirrorPool::L0 {
        let mut p = std::ptr::null_mut();
        // SAFETY: a buffer in a CPU visible heap, mapped for the life of the process (never unmapped).
        unsafe { buffer.Map(0, None, Some(&mut p)) }?;
        p as usize
    } else {
        0
    };
    let seg = if pool == MirrorPool::L1 { crate::mem_owners::Segment::Local } else { crate::mem_owners::Segment::System };
    crate::mem_owners::add(
        crate::mem_owners::Kind::TitleMirrorHeap,
        seg,
        1,
        POOL_HEAP_BYTES,
    );
    Ok(Heap { _heap: heap, buffer, ptr, is_l1: pool == MirrorPool::L1 })
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
    /// Returns `(state_id, device, start_idx, new_sizes)` if new window buffers are needed.
    fn missing_window_sizes(&self) -> Option<(u64, ID3D12Device, usize, Vec<u64>)> {
        let sizes = self.mirror.buffer_bytes();
        if sizes.len() <= self.windows.len() {
            return None;
        }
        Some((self.id, self.device.clone(), self.windows.len(), sizes[self.windows.len()..].to_vec()))
    }

    /// The address of pool page `page` in the read only view of its section.
    fn source(&mut self, api: &Api, page: u32) -> Option<usize> {
        self.source_section(api, page).map(|(src, _)| src)
    }

    /// The address of pool page `page` and how many pages remain in its section from `page`.
    fn source_section(&mut self, api: &Api, page: u32) -> Option<(usize, usize)> {
        if let Some(&(base, first, pages)) = self.sections.get(self.last_section)
            && page >= first
            && page - first < pages
        {
            return Some((base + (page - first) as usize * PAGE as usize, (first + pages - page) as usize));
        }
        loop {
            if let Some((idx, &(base, first, pages))) = self.sections.iter().enumerate().find(|&(_, &(_, first, pages))| page >= first && page - first < pages) {
                self.last_section = idx;
                return Some((base + (page - first) as usize * PAGE as usize, (first + pages - page) as usize));
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
        if matches!(c.kind, RELEASE | UNMAP) {
            bump_unmap_gen();
        }
        MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
        if let Some(bytes) = self.mirror.apply(change)
            && REFUSED.fetch_add(1, Ordering::Relaxed) == 0
        {
            xs_log!("title memory: a reservation of {} GiB is larger than a window buffer can be (1 TiB), its pages stay invisible to the GPU", bytes >> 30);
        }
    }

    /// Reads what `xmem.dll` changed since the last time into the mirror, or starts again from a snapshot
    /// when its ring no longer holds it.
    fn drain(&mut self, api: &Api) {
        let cur_seq = api.journal_seq.map(|f| {
            // SAFETY: `DecantXmemJournalSeq` is a lock-free atomic load returning `u64`.
            unsafe { f() }
        });
        if should_skip_drain(self.last_journal_seq, cur_seq) {
            return;
        }
        let _trace = decant_trace::span("title_mem", "drain");
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
        self.last_journal_seq = cur_seq;
        if let Some(seq) = cur_seq {
            LAST_JOURNAL_SEQ.store(seq, Ordering::Release);
        }
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
        bump_unmap_gen();
        MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
        for c in &records {
            self.apply(c);
        }
        self.seq = seq;
    }

    /// Applies the pending tile mappings on the private queue and signals the fence. Returns the ranges.
    /// Collects copy jobs for sorted unique pool pages written since the last sync, ensuring heaps
    /// are either ready or returned in `Err(missing_heaps)` to be created outside `STATE`.
    fn plan_upload(&mut self, api: &Api, pages: &[u32], jobs: &mut Vec<CopyJob>, runs: &mut Vec<UploadRun>) -> Result<u64, Vec<u32>> {
        if pages.is_empty() {
            return Ok(0);
        }
        let _trace = decant_trace::span("title_mem", "upload");
        let mut total = 0;
        let mut new_tile = false;
        // A page copied for the first time has bytes now: it gets its mirror tile here (in page order,
        // so consecutive pages get consecutive tiles), and the window shows it from this sync on.
        let diff_enabled = subpage_diff_enabled();
        for &page in pages {
            if self.mirror.tile_of(page).is_none() {
                let tile = self.mirror.give_tile(page);
                if diff_enabled {
                    self.written.push(tile);
                }
                new_tile = true;
            }
        }
        let res = self.plan_runs(api, pages, jobs, runs, &mut total);
        if new_tile {
            bump_tile_gen();
            MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
        }
        res.map(|()| total)
    }

    fn plan_runs(&mut self, api: &Api, pages: &[u32], jobs: &mut Vec<CopyJob>, runs: &mut Vec<UploadRun>, copied: &mut u64) -> Result<(), Vec<u32>> {
        let mut missing_heaps = Vec::new();
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
            let heap_info = self.heaps.get(heap as usize).and_then(Option::as_ref).map(|h| (h.ptr, h.is_l1));
            if heap_info.is_none() && !self.failed.contains(&heap) {
                if !missing_heaps.contains(&heap) {
                    missing_heaps.push(heap);
                }
                i += n;
                continue;
            }
            let (Some((dst, use_staging)), Some((src, in_section))) = (heap_info, self.source_section(api, first)) else {
                i += n;
                continue;
            };
            // One run never leaves its section: the tail of a section is cut off and copied by page.
            let head = n.min(in_section);
            i += head;
            if !missing_heaps.is_empty() {
                continue;
            }
            if use_staging {
                runs.push(UploadRun {
                    source: src,
                    heap,
                    heap_offset: tile as u64 * PAGE,
                    staging_offset: 0,
                    bytes: head as u64 * PAGE,
                });
            } else {
                plan_copy_tiles(jobs, src, dst + tile as usize * PAGE as usize, head, false, first_tile);
            }
            *copied += head as u64;
            if !subpage_diff_enabled() {
                self.written.extend(first_tile..first_tile + head as u32);
            }
        }
        if missing_heaps.is_empty() { Ok(()) } else { Err(missing_heaps) }
    }
}

/// One `UpdateTileMappings` call of the private queue, reusing `scratch` buffers.
fn update_call(queue: &ID3D12CommandQueue, window: &ID3D12Resource, call: &TileCall, heap: Option<&ID3D12Heap>, scratch: &mut TileScratch) {
    let n = call.ranges.len() as u32;
    scratch.coords.clear();
    scratch.coords.extend(call.ranges.iter().map(|&(tile, _)| D3D12_TILED_RESOURCE_COORDINATE { X: tile, Y: 0, Z: 0, Subresource: 0 }));
    scratch.sizes.clear();
    scratch.sizes.extend(call.ranges.iter().map(|&(_, count)| D3D12_TILE_REGION_SIZE { NumTiles: count, ..Default::default() }));
    scratch.counts.clear();
    scratch.counts.extend(call.ranges.iter().map(|&(_, count)| count));
    let flag = if call.heap.is_some() && heap.is_some() { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_NULL };
    scratch.flags.clear();
    scratch.flags.resize(n as usize, flag);
    // SAFETY: the arrays have `n` entries each, the resource and the heap are live.
    unsafe { queue.UpdateTileMappings(window, n, Some(scratch.coords.as_ptr()), Some(scratch.sizes.as_ptr()), heap, n, Some(scratch.flags.as_ptr()), Some(call.heap_tiles.as_ptr()), Some(scratch.counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE) };
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
    let env_pool = std::env::var("DECANT_D3D12X_MIRROR_POOL").ok();
    let env_l1_mib = std::env::var("DECANT_D3D12X_MIRROR_L1_MIB").ok();
    let env_l0_mib = std::env::var("DECANT_D3D12X_MIRROR_L0_MIB").ok();
    let adapter = local_adapter(&device);
    let dedicated = dedicated_vram(adapter.as_ref());
    let (budget, _) = adapter.as_ref().and_then(local_vram_info).unwrap_or((dedicated, 0));
    let mut pool_mode = parse_mirror_pool_mode(env_pool.as_deref());
    let m_cap_override = parse_mirror_l1_mib(env_l1_mib.as_deref());
    let l0_cap_override = parse_mirror_l0_mib(env_l0_mib.as_deref());
    let vram_budget = VramBudget::from_budget(budget);
    let effective_m_cap = m_cap_override.unwrap_or(vram_budget.m_cap);
    let (total_phys, avail_phys, avail_pagefile) = host_ram_status().unwrap_or((u64::MAX, u64::MAX, u64::MAX));
    let l0_cap = host_ram_l0_cap(total_phys, avail_phys, avail_pagefile, l0_cap_override);
    let ring_bytes = staging_ring_bytes(total_phys);
    let needs_staging = match pool_mode {
        MirrorPoolMode::StrictL0 => false,
        MirrorPoolMode::Auto => effective_m_cap > 0,
        MirrorPoolMode::PreferL1 => true,
    };
    let staging = if needs_staging {
        match StagingRing::new(&device, ring_bytes) {
            Ok(ring) => Some(ring),
            Err(e) => {
                xs_log!("title memory: L1 staging ring cannot be made ({e}), falling back to L0 mirror heaps");
                pool_mode = MirrorPoolMode::StrictL0;
                None
            }
        }
    } else {
        None
    };
    xs_log!(
        "title memory: mirror mode {:?} (dedicated {} MiB, budget {} MiB, hwm {} MiB, lwm {} MiB, l1 cap {} MiB, l0 cap {} MiB, staging {} MiB)",
        pool_mode,
        dedicated >> 20,
        budget >> 20,
        vram_budget.hwm_total >> 20,
        vram_budget.lwm_total >> 20,
        effective_m_cap >> 20,
        l0_cap >> 20,
        staging.as_ref().map_or(0, |r| r.size >> 20)
    );
    let has_staging = staging.is_some();
    let mut sync_guard = SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    sync_guard.staging = staging;
    sync_guard.l1_prints.clear();
    sync_guard.waited = Some(HashMap::new());
    sync_guard.cached = None;
    sync_guard.stats = Stats::default();
    sync_guard.last_guard = Some(Instant::now());
    let mut st = STATE.write().unwrap_or_else(PoisonError::into_inner);
    if st.is_some() {
        xs_log!("title memory: a second PC device, the heaps of the first are not used for it");
    }
    bump_mapping_gen();
    bump_unmap_gen();
    PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner).clear();
    HAS_PENDING_JOURNAL.store(false, Ordering::Release);
    MIRROR_MAY_HAVE_PENDING.store(false, Ordering::Release);
    DRAINED_STATE_ID.store(0, Ordering::Release);
    DRAINED_JOURNAL_SEQ.store(u64::MAX, Ordering::Release);
    LAST_JOURNAL_SEQ.store(u64::MAX, Ordering::Release);
    MISSING_WINDOWS.store(false, Ordering::Release);
    let id = NEXT_STATE.fetch_add(1, Ordering::Relaxed);
    ACTIVE_STATE_ID.store(id, Ordering::Release);
    *st = Some(State {
        id,
        device,
        map_queue,
        map_fence,
        fence_value: Arc::new(AtomicU64::new(0)),
        mirror: Mirror::new(window_bytes()),
        seq: 0,
        last_journal_seq: None,
        heaps: Vec::new(),
        adapter,
        pool_mode,
        fallback_budget: budget,
        m_cap_override,
        l0_cap_override,
        l1_heaps: 0,
        l0_heaps: 0,
        has_staging,
        failed: HashSet::new(),
        windows: Vec::new(),
        sections: Vec::new(),
        last_section: 0,
        tracking: false,
        zero_scan: ZeroScan::default(),
        scratch: Vec::new(),
        written: Vec::new(),
    });
}

/// Creates any missing window buffers outside `STATE`, holding only `WINDOW_MAKE_LOCK` across
/// `CreateReservedResource` and taking `STATE.write()` only for the O(1) push.
fn ensure_windows_outside_state(state_id: u64, device: &ID3D12Device, start_idx: usize, sizes: &[u64]) {
    if sizes.is_empty() {
        return;
    }
    let _make_guard = WINDOW_MAKE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    for (offset, &bytes) in sizes.iter().enumerate() {
        let i = start_idx + offset;
        let already = {
            let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
            guard.as_ref().is_none_or(|st| st.id != state_id || st.windows.len() > i)
        };
        if already {
            continue;
        }
        let made = make_window(device, bytes, i);
        match &made {
            Ok(w) => xs_log!("title memory: window buffer {i} of {} GiB at GPU address {:#x}", bytes >> 30, w.gpu),
            Err(e) => xs_log!("title memory: window buffer {i} of {} GiB cannot be made ({e}), its slots stay invisible to the GPU", bytes >> 30),
        }
        let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.write().unwrap_or_else(PoisonError::into_inner));
        if let Some(st) = guard.as_mut().filter(|st| st.id == state_id && st.windows.len() == i) {
            st.windows.push(made.ok());
        }
    }
}

/// Creates any missing pool heaps in `indices` outside `STATE`, holding `HEAP_MAKE_LOCK` across
/// `CreateHeap`/`CreatePlacedResource`/`Map` and taking `STATE.write()` only to insert the handle.
fn ensure_heaps_outside_state(state_id: u64, indices: &[u32]) {
    if indices.is_empty() {
        return;
    }
    let _make_guard = HEAP_MAKE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    for &index in indices {
        let snap = {
            let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
            let Some(st) = guard.as_ref().filter(|st| st.id == state_id) else {
                return;
            };
            let i = index as usize;
            if st.heaps.get(i).and_then(Option::as_ref).is_some() || st.failed.contains(&index) {
                continue;
            }
            (
                st.device.clone(),
                st.adapter.clone(),
                st.pool_mode,
                st.fallback_budget,
                st.has_staging,
                st.l1_heaps,
                st.l0_heaps,
                st.m_cap_override,
                st.l0_cap_override,
            )
        };
        let (device, adapter, pool_mode, fallback_budget, has_staging, l1_heaps, l0_heaps, m_cap_override, l0_cap_override) = snap;
        let (budget, current_usage) = adapter.as_ref().and_then(local_vram_info).unwrap_or((fallback_budget, 0));
        let want_l1 = has_staging && should_use_l1(pool_mode, budget, current_usage, l1_heaps * POOL_HEAP_BYTES, m_cap_override);
        let made = if want_l1 {
            match make_heap(&device, index, MirrorPool::L1) {
                Ok(h) => Ok(Some(h)),
                Err(e) => {
                    xs_log!("title memory: L1 pool heap {index} failed ({e}), spilling to L0");
                    make_heap(&device, index, MirrorPool::L0).map(Some)
                }
            }
        } else {
            let allow_l0 = if let Some((total_phys, avail_phys, avail_pagefile)) = host_ram_status() {
                let l0_cap = host_ram_l0_cap(total_phys, avail_phys, avail_pagefile, l0_cap_override);
                let l0_bytes = l0_heaps * POOL_HEAP_BYTES;
                let ok = can_grow_l0(l0_bytes, l0_cap, avail_phys, avail_pagefile);
                if !ok {
                    static L0_PRESSURE_LOGGED: std::sync::Once = std::sync::Once::new();
                    L0_PRESSURE_LOGGED.call_once(|| {
                        xs_log!(
                            "title memory: L0 host RAM cap reached ({} MiB / {} MiB, avail phys {} MiB, avail pagefile {} MiB), skipping further L0 heap growth",
                            l0_bytes >> 20,
                            l0_cap >> 20,
                            avail_phys >> 20,
                            avail_pagefile >> 20
                        );
                    });
                }
                ok
            } else {
                true
            };
            if !allow_l0 { Ok(None) } else { make_heap(&device, index, MirrorPool::L0).map(Some) }
        };
        let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.write().unwrap_or_else(PoisonError::into_inner));
        let Some(st) = guard.as_mut().filter(|st| st.id == state_id) else {
            return;
        };
        let i = index as usize;
        if st.heaps.len() <= i {
            st.heaps.resize_with(i + 1, || None);
        }
        if st.heaps[i].is_some() || st.failed.contains(&index) {
            continue;
        }
        match made {
            Ok(Some(heap)) => {
                if heap.is_l1 {
                    st.l1_heaps += 1;
                } else {
                    st.l0_heaps += 1;
                }
                st.heaps[i] = Some(heap);
            }
            Ok(None) => {
                st.failed.insert(index);
            }
            Err(e) => {
                st.failed.insert(index);
                xs_log!("title memory: pool heap {index} cannot be made ({e}), its pages stay invisible to the GPU");
            }
        }
    }
}

fn ensure_missing_windows() {
    if !MISSING_WINDOWS.load(Ordering::Acquire) {
        return;
    }
    let missing = {
        let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
        guard.as_ref().and_then(State::missing_window_sizes)
    };
    if let Some((state_id, device, start_idx, sizes)) = missing {
        ensure_windows_outside_state(state_id, &device, start_idx, &sizes);
    }
    let still_missing = {
        let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
        guard.as_ref().and_then(State::missing_window_sizes).is_some()
    };
    if !still_missing {
        MISSING_WINDOWS.store(false, Ordering::Release);
    }
}

/// Drains any new `xmem.dll` journal records into `STATE` and drops `DRAIN_LOCK` immediately after
/// applying them, before window creation or any caller work runs.
fn sync_journal_and_windows(api: &Api) -> bool {
    let active_id = ACTIVE_STATE_ID.load(Ordering::Acquire);
    if active_id == 0 {
        return false;
    }
    if journal_unchanged(api) && !HAS_PENDING_JOURNAL.load(Ordering::Acquire) {
        ensure_missing_windows();
        return true;
    }
    let mut need_apply = HAS_PENDING_JOURNAL.load(Ordering::Acquire);
    if !drained_journal_unchanged(api, active_id) {
        let mut cursor = decant_trace::timed_lock("title_mem::DRAIN_LOCK", || DRAIN_LOCK.lock().unwrap_or_else(PoisonError::into_inner));
        let active_id = ACTIVE_STATE_ID.load(Ordering::Acquire);
        if active_id == 0 {
            return false;
        }
        if !cursor.tracking {
            cursor.tracking = true;
            // SAFETY: the export takes a flag.
            unsafe { (api.track)(1) };
            TRACKING_ACTIVE.store(true, Ordering::Release);
            xs_log!("title memory: xmem.dll tracks written pages now, every mapped page is copied at the first sync");
        }
        if !drained_journal_unchanged(api, active_id) {
            let batch = drain_journal(api, &mut cursor);
            match batch {
                DrainBatch::Unchanged => {
                    if api.journal_seq.is_none() || LAST_JOURNAL_SEQ.load(Ordering::Acquire) == u64::MAX {
                        PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner).push(PendingBatch {
                            state_id: active_id,
                            snapshot: false,
                            seq: cursor.seq,
                            journal_seq: cursor.last_journal_seq,
                            records: Vec::new(),
                        });
                        HAS_PENDING_JOURNAL.store(true, Ordering::Release);
                        need_apply = true;
                    }
                }
                DrainBatch::Changes { state_id, seq, journal_seq } => {
                    let mut records = std::mem::take(&mut cursor.records);
                    coalesce_changes(&mut records, &mut cursor.known_res, false);
                    PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner).push(PendingBatch {
                        state_id,
                        snapshot: false,
                        seq,
                        journal_seq,
                        records,
                    });
                    HAS_PENDING_JOURNAL.store(true, Ordering::Release);
                    if let Some(jseq) = journal_seq {
                        DRAINED_STATE_ID.store(state_id, Ordering::Release);
                        DRAINED_JOURNAL_SEQ.store(jseq, Ordering::Release);
                    }
                    need_apply = true;
                }
                DrainBatch::Snapshot { state_id, seq, journal_seq } => {
                    let mut records = std::mem::take(&mut cursor.records);
                    coalesce_changes(&mut records, &mut cursor.known_res, true);
                    PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner).push(PendingBatch {
                        state_id,
                        snapshot: true,
                        seq,
                        journal_seq,
                        records,
                    });
                    HAS_PENDING_JOURNAL.store(true, Ordering::Release);
                    if let Some(jseq) = journal_seq {
                        DRAINED_STATE_ID.store(state_id, Ordering::Release);
                        DRAINED_JOURNAL_SEQ.store(jseq, Ordering::Release);
                    }
                    need_apply = true;
                }
            }
        }
        // `DRAIN_LOCK` is released here so `APPLY_LOCK` / `STATE.write()` never stalls draining.
    }

    let mut missing = None;
    if need_apply || HAS_PENDING_JOURNAL.load(Ordering::Acquire) {
        let _apply = decant_trace::timed_lock("title_mem::APPLY_LOCK", || APPLY_LOCK.lock().unwrap_or_else(PoisonError::into_inner));
        loop {
            let mut batches = {
                let mut p = PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner);
                if p.is_empty() {
                    HAS_PENDING_JOURNAL.store(false, Ordering::Release);
                    break;
                }
                std::mem::take(&mut *p)
            };
            let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.write().unwrap_or_else(PoisonError::into_inner));
            let Some(st) = guard.as_mut() else {
                HAS_PENDING_JOURNAL.store(false, Ordering::Release);
                return false;
            };
            st.tracking = true;
            for b in batches.drain(..) {
                if b.state_id != st.id {
                    continue;
                }
                if b.snapshot {
                    xs_log!("title memory: the journal of xmem.dll overran, starting again from a snapshot of {} records", b.records.len());
                    st.mirror.reset();
                    bump_mapping_gen();
                    bump_unmap_gen();
                    MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
                    for c in &b.records {
                        st.apply(c);
                    }
                    st.seq = b.seq;
                    st.last_journal_seq = b.journal_seq;
                } else {
                    for c in &b.records {
                        if c.seq > st.seq {
                            st.apply(c);
                        }
                    }
                    st.seq = st.seq.max(b.seq);
                    if b.journal_seq.is_some() {
                        st.last_journal_seq = b.journal_seq;
                    }
                }
                if let Some(jseq) = b.journal_seq {
                    LAST_JOURNAL_SEQ.store(jseq, Ordering::Release);
                }
            }
            let m = st.missing_window_sizes();
            if m.is_some() {
                MISSING_WINDOWS.store(true, Ordering::Release);
            }
            missing = m;
        }
    }

    if let Some((state_id, device, start_idx, sizes)) = missing {
        ensure_windows_outside_state(state_id, &device, start_idx, &sizes);
        let still_missing = {
            let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
            guard.as_ref().and_then(State::missing_window_sizes).is_some()
        };
        if !still_missing {
            MISSING_WINDOWS.store(false, Ordering::Release);
        }
    } else {
        ensure_missing_windows();
    }
    true
}

/// Runs `f` on the state, with the exports of `xmem.dll`, once tracking is on and the journal is read.
/// `None` when the module is off, has no device or `xmem.dll` is not there.
fn with_state<R>(f: impl FnOnce(&mut State, &Api) -> R) -> Option<R> {
    if !enabled() {
        return None;
    }
    let api = api()?;
    if !sync_journal_and_windows(api) {
        return None;
    }
    let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.write().unwrap_or_else(PoisonError::into_inner));
    let st = guard.as_mut()?;
    st.tracking = true;
    Some(f(st, api))
}

/// Runs `f` under a shared read lock after bringing the journal up to date if needed.
fn with_state_read<R>(f: impl FnOnce(&State) -> Option<R>) -> Option<R> {
    if !enabled() {
        return None;
    }
    let api = api()?;
    if ACTIVE_STATE_ID.load(Ordering::Acquire) == 0 {
        return None;
    }
    let cur_seq = api.journal_seq.map(|g| {
        // SAFETY: `DecantXmemJournalSeq` is a lock-free atomic load returning `u64`.
        unsafe { g() }
    });
    if TRACKING_ACTIVE.load(Ordering::Acquire)
        && !HAS_PENDING_JOURNAL.load(Ordering::Acquire)
        && !MISSING_WINDOWS.load(Ordering::Acquire)
        && let Some(seq) = cur_seq
        && LAST_JOURNAL_SEQ.load(Ordering::Acquire) == seq
    {
        let guard = decant_trace::timed_lock("title_mem::STATE_READ", || STATE.read().unwrap_or_else(PoisonError::into_inner));
        let st = guard.as_ref()?;
        if st.tracking && st.last_journal_seq == Some(seq) {
            return f(st);
        }
    }
    if !sync_journal_and_windows(api) {
        return None;
    }
    let guard = decant_trace::timed_lock("title_mem::STATE_READ", || STATE.read().unwrap_or_else(PoisonError::into_inner));
    let st = guard.as_ref()?;
    f(st)
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
        let mut guard = STATE.write().unwrap_or_else(PoisonError::into_inner);
        guard.as_mut()?.zero_page_chunk(api)
    };
    let state = chunk.state;
    let results = chunk.read();
    let scanned = results.len() as u32;
    let mut guard = STATE.write().unwrap_or_else(PoisonError::into_inner);
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
    STATE.read().unwrap_or_else(PoisonError::into_inner).as_ref().map(|s| s.device.clone())
}

/// The generation of the window slots now: it changes when a slot is given or freed, not when a page
/// is mapped inside one. `None` when the module is off or `xmem.dll` is not there.
pub fn generation() -> Option<u32> {
    with_state_read(|st| Some(st.mirror.generation()))
}

/// The slots of the window (GPU address and bytes), with the generation they are for. Empty when there is
/// no device or no slot, `None` when the module is off.
pub fn extents() -> Option<(u32, Vec<Entry>)> {
    with_state_read(|st| {
        let entries = st
            .mirror
            .slots()
            .into_iter()
            .filter_map(|(va, buffer, offset, bytes)| {
                let w = st.windows.get(buffer)?.as_ref()?;
                Some(Entry { resource: w.resource.as_raw() as usize, gpu: w.gpu + offset, bytes, va })
            })
            .collect();
        Some((st.mirror.generation(), entries))
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
    fn translate_span_read(&self, va: u64, len: u64) -> Option<(Span, bool)> {
        let placed = self.mirror.translate(va, len)?;
        let w = self.windows.get(placed.buffer)?.as_ref()?;
        let is_fresh = self.mirror.pending_in(placed.buffer, placed.offset, len);
        Some((Span { resource: w.resource.as_raw() as usize, gpu: w.gpu + placed.offset, offset: placed.offset, left: placed.left }, is_fresh))
    }

    fn translate_span(&mut self, api: &Api, va: u64, len: u64, give: bool) -> Option<(Span, bool)> {
        if give {
            self.give(api, va, len.max(ADDRESS_REACH));
            // Pages the title never wrote have no mirror tile and read as zeros. The first `ADDRESS_REACH`
            // bytes of a span the GPU is given get tiles anyway: the GPU may write there (a root UAV, a
            // counter) and read it back later.
            let prev_tiles = self.mirror.tiles_given();
            self.mirror.give_span(va, len.clamp(1, ADDRESS_REACH));
            if self.mirror.tiles_given() != prev_tiles {
                bump_tile_gen();
                MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
            }
        }
        let placed = self.mirror.translate(va, len)?;
        let w = self.windows.get(placed.buffer)?.as_ref()?;
        let is_fresh = self.mirror.pending_in(placed.buffer, placed.offset, len);
        Some((Span { resource: w.resource.as_raw() as usize, gpu: w.gpu + placed.offset, offset: placed.offset, left: placed.left }, is_fresh))
    }
}

/// Asks `xmem.dll` to bring in on-demand pages outside `STATE` and `DRAIN_LOCK`, then syncs the
/// resulting `MAP` journal records before any caller acquires `STATE.write()`.
fn give_outside_state(api: &Api, spans: &[(u64, u64)]) {
    if spans.is_empty() {
        return;
    }
    static GIVEN: AtomicI32 = AtomicI32::new(0);
    let mut total_pages = 0u32;
    for &(va, len) in spans {
        // SAFETY: the export takes plain values.
        let pages = unsafe { (api.gpu_use)(va as usize, len as usize) };
        if pages > 0 {
            total_pages += pages;
            if GIVEN.fetch_add(1, Ordering::Relaxed) < 8 {
                xs_log!("title memory: {pages} pages from title address {va:#x} reach the GPU now (a commit whose pages come on demand)");
            }
        }
    }
    if total_pages > 0 {
        GIVEN_PAGES.fetch_add(u64::from(total_pages), Ordering::Relaxed);
        let _ = sync_journal_and_windows(api);
    }
}

/// Translates a batch of `(va, len)` spans under one state lock, calling `DecantXmemGpuUse`
/// outside `STATE` and `DRAIN_LOCK` when on-demand commits need pages.
pub fn translate_fresh_batch(spans: &[(u64, u64)], out: &mut Vec<Option<(Span, bool)>>) {
    out.clear();
    if spans.is_empty() {
        return;
    }
    let read_res = with_state_read(|st| {
        let mut tmp = Vec::with_capacity(spans.len());
        let mut to_give = Vec::new();
        let mut all_ready = true;
        for &(va, len) in spans {
            let reach = len.max(ADDRESS_REACH);
            if st.mirror.wants_use(va, reach) {
                to_give.push((va, reach));
                all_ready = false;
            } else if !st.mirror.has_tiles_for_span(va, len.clamp(1, ADDRESS_REACH)) {
                all_ready = false;
            } else if all_ready {
                tmp.push(st.translate_span_read(va, len));
            }
        }
        Some(if all_ready { Ok(tmp) } else { Err(to_give) })
    });
    match read_res {
        None => {
            out.resize(spans.len(), None);
            return;
        }
        Some(Ok(tmp)) => {
            *out = tmp;
            return;
        }
        Some(Err(to_give)) => {
            if !to_give.is_empty()
                && let Some(api) = api()
            {
                give_outside_state(api, &to_give);
            }
        }
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
    let (span, is_fresh) = if !give {
        with_state_read(|st| st.translate_span_read(va, len))?
    } else {
        match with_state_read(|st| {
            let reach = len.max(ADDRESS_REACH);
            if st.mirror.wants_use(va, reach) {
                Some(Err(true))
            } else if !st.mirror.has_tiles_for_span(va, len.clamp(1, ADDRESS_REACH)) {
                Some(Err(false))
            } else {
                Some(Ok(st.translate_span_read(va, len)))
            }
        })? {
            Ok(found) => found?,
            Err(needs_use) => {
                if needs_use && let Some(api) = api() {
                    give_outside_state(api, &[(va, len.max(ADDRESS_REACH))]);
                }
                with_state(|st, api| st.translate_span(api, va, len, true)).flatten()?
            }
        }
    };
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

/// Maps `buffer` onto `pages` outside `STATE`, ensuring any missing pool heaps exist first.
fn map_buffer_to_pool_outside_state(
    state_id: u64,
    map_queue: &ID3D12CommandQueue,
    map_fence: &ID3D12Fence,
    fence_value: &AtomicU64,
    buffer: &ID3D12Resource,
    pages: &[u32],
) -> Option<()> {
    let heap_runs = runs_by_heap(pages);
    let missing_indices: Vec<u32> = heap_runs.iter().map(|&(h, _)| h).collect();
    ensure_heaps_outside_state(state_id, &missing_indices);
    let heaps: Vec<ID3D12Heap> = {
        let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
        let st = guard.as_ref().filter(|st| st.id == state_id)?;
        let mut out = Vec::with_capacity(heap_runs.len());
        for &(heap, _) in &heap_runs {
            let pool = st.heaps.get(heap as usize)?.as_ref()?;
            out.push(pool._heap.clone());
        }
        out
    };
    let mut scratch = MAP_QUEUE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    for ((heap, runs), pool_heap) in heap_runs.into_iter().zip(heaps) {
        let call = TileCall {
            buffer: 0,
            heap: Some(heap),
            ranges: runs.iter().map(|&(at, n, _)| (at, n)).collect(),
            heap_tiles: runs.iter().map(|&(_, _, t)| t).collect(),
        };
        update_call(map_queue, buffer, &call, Some(&pool_heap), &mut scratch);
    }
    let next_val = fence_value.fetch_add(1, Ordering::AcqRel) + 1;
    // SAFETY: both objects are live.
    if let Err(e) = unsafe { map_queue.Signal(map_fence, next_val) } {
        xs_log!("title memory: the map fence cannot be signalled ({e})");
    }
    Some(())
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
    let span_len = u64::from(tiles) * PAGE;
    if with_state_read(|st| Some(st.mirror.wants_use(va, span_len))) == Some(true)
        && let Some(api) = api()
    {
        give_outside_state(api, &[(va, span_len)]);
    }
    let (state_id, device, map_queue, map_fence, fence_value, pages) = with_state(|st, api| {
        st.give(api, va, span_len);
        let prev_tiles = st.mirror.tiles_given();
        let pages = st.mirror.mirror_tiles(va, tiles)?;
        if st.mirror.tiles_given() != prev_tiles {
            MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
        }
        Some((st.id, st.device.clone(), st.map_queue.clone(), st.map_fence.clone(), st.fence_value.clone(), pages))
    })??;
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain D3D12 call on a live device with a description of plain data.
    let created = unsafe { device.CreateReservedResource(&reserved, state, None, &mut buffer) };
    if let Err(e) = created {
        if FAILED.fetch_add(1, Ordering::Relaxed) < 4 {
            xs_log!("title memory: the buffer placed at {va:#x} ({} bytes) cannot be a reserved buffer ({e})", desc.Width);
        }
        return None;
    }
    let made = buffer?;
    map_buffer_to_pool_outside_state(state_id, &map_queue, &map_fence, &fence_value, &made, &pages)?;
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
/// Takes dirty pool page indices from `xmem.dll` outside `STATE` and returns `(take_micros)`.
fn take_dirty_outside_state(api: &Api, dirty: &mut Vec<u32>) -> u64 {
    dirty.clear();
    let asked = Instant::now();
    loop {
        let base = dirty.len();
        dirty.reserve(TAKE_PAGES);
        // SAFETY: `dirty` has spare capacity for `TAKE_PAGES` entries past `base`.
        let n = {
            let _take = decant_trace::span("title_mem", "take_dirty");
            unsafe { (api.take_dirty)(dirty.as_mut_ptr().add(base), TAKE_PAGES as u32) }
        } as usize;
        let n = n.min(TAKE_PAGES);
        // SAFETY: `take_dirty` wrote `n` initialized page indices past `base`.
        unsafe { dirty.set_len(base + n) };
        if n < TAKE_PAGES {
            break;
        }
    }
    if !dirty.is_empty() {
        dirty.sort_unstable();
        dirty.dedup();
    }
    asked.elapsed().as_micros() as u64
}

#[derive(Default, Clone, Copy)]
struct SyncBatchMetrics {
    pages: u64,
    copied_bytes: u64,
    skipped_bytes: u64,
    ranges: u64,
    copy_micros: u64,
    tile_micros: u64,
}

fn maybe_stats_and_guard(
    stats: &mut Stats,
    guard_timer: &mut Option<Instant>,
    m: SyncBatchMetrics,
    started: Instant,
    run_guard: &mut bool,
) {
    let guard = guard_timer.get_or_insert_with(Instant::now);
    if guard.elapsed() >= GUARD_EVERY {
        *guard = Instant::now();
        *run_guard = true;
    }
    let micros = started.elapsed().as_micros() as u64;
    stats.syncs += 1;
    stats.pages += m.pages;
    stats.copied_bytes += m.copied_bytes;
    stats.skipped_bytes += m.skipped_bytes;
    stats.ranges += m.ranges;
    stats.copy_micros += m.copy_micros;
    stats.tile_micros += m.tile_micros;
    stats.micros += micros;
    stats.worst_micros = stats.worst_micros.max(micros);
    if stats.syncs.is_multiple_of(STATS_EVERY) {
        let slots = STATE
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map_or(0, |st| st.mirror.slots().len());
        xs_log!(
            "title memory: {} syncs, {} pages ({} MiB copied, {} MiB skipped by subpage diff), {} tile ranges updated, {} us in total ({} us asking for written pages, {} us copying, {} us mapping tiles), {} us at worst, {} slots",
            stats.syncs,
            stats.pages,
            stats.copied_bytes >> 20,
            stats.skipped_bytes >> 20,
            stats.ranges,
            stats.micros,
            stats.take_micros,
            stats.copy_micros,
            stats.tile_micros,
            stats.worst_micros,
            slots
        );
        *stats = Stats { syncs: stats.syncs, ..Stats::default() };
    }
}

pub unsafe fn sync(queue: *mut c_void) {
    if !enabled() {
        return;
    }
    let _trace = decant_trace::span("title_mem", "sync");
    // SAFETY: by the contract above.
    let Some(queue_ref) = (unsafe { ID3D12CommandQueue::from_raw_borrowed(&queue) }) else { return };
    let Some(api_ref) = api() else { return };
    if ACTIVE_STATE_ID.load(Ordering::Acquire) == 0 {
        return;
    }
    let _ = sync_journal_and_windows(api_ref);

    let mut sync_scratch = decant_trace::timed_lock("title_mem::SYNC_LOCK", || SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner()));
    let SyncScratch { jobs, dirty, runs, staging, l1_prints, waited, cached, stats, last_guard } = &mut *sync_scratch;
    let waited = waited.get_or_insert_with(HashMap::new);
    thread_local! {
        static SYNC_WRITTEN: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    }
    let mut written = SYNC_WRITTEN.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    written.clear();
    let started = Instant::now();
    let take_micros = take_dirty_outside_state(api_ref, dirty);
    stats.take_micros += take_micros;
    static CORES: OnceLock<usize> = OnceLock::new();
    let cores = *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, usize::from));

    if dirty.is_empty() && !MIRROR_MAY_HAVE_PENDING.load(Ordering::Acquire) {
        let cur_id = ACTIVE_STATE_ID.load(Ordering::Acquire);
        if cur_id == 0 {
            SYNC_WRITTEN.with(|cell| *cell.borrow_mut() = written);
            return;
        }
        if !cached.as_ref().is_some_and(|c| c.state_id == cur_id) {
            let guard = decant_trace::timed_lock("title_mem::STATE_READ", || STATE.read().unwrap_or_else(PoisonError::into_inner));
            if let Some(st) = guard.as_ref().filter(|st| st.id == cur_id) {
                *cached = Some(CachedSyncState {
                    state_id: st.id,
                    map_queue: st.map_queue.clone(),
                    map_fence: st.map_fence.clone(),
                    fence_value: st.fence_value.clone(),
                });
            }
        }
        if let Some(c) = cached.as_ref().filter(|c| c.state_id == cur_id) {
            let cur_fence = c.fence_value.load(Ordering::Acquire);
            if cur_fence > 0 && waited.get(&(queue as usize)).is_none_or(|&v| v < cur_fence) {
                // SAFETY: both objects are live. The queue waits on the GPU, the CPU does not.
                match unsafe { queue_ref.Wait(&c.map_fence, cur_fence) } {
                    Ok(()) => {
                        waited.insert(queue as usize, cur_fence);
                    }
                    Err(e) => xs_log!("title memory: a queue cannot wait for the tile mappings ({e})"),
                }
            }
            crate::diagnose::removed_after(|| "a sync of 0 pages and 0 tile ranges".to_string());
            let mut run_guard = false;
            maybe_stats_and_guard(stats, last_guard, SyncBatchMetrics::default(), started, &mut run_guard);
            drop(sync_scratch);
            if run_guard {
                // SAFETY: the export takes no argument.
                unsafe { (api_ref.guard_modules)() };
            }
            SYNC_WRITTEN.with(|cell| *cell.borrow_mut() = written);
            return;
        }
    }

    let mut total_pages = 0u64;
    let mut total_copied_bytes = 0u64;
    let mut total_skipped_bytes = 0u64;
    let mut total_ranges = 0u64;
    let mut total_copy_micros = 0u64;
    let mut total_tile_micros = 0u64;
    let mut last_fence: Option<(ID3D12Fence, Arc<AtomicU64>)> = None;
    let mut run_guard = false;
    let max_batch_pages = staging.as_ref().map_or(STAGING_MAX_PAGES, |r| r.size / PAGE as usize);
    let mut first_batch = true;
    let mut chunk_start = 0usize;
    while first_batch || chunk_start < dirty.len() {
        first_batch = false;
        let chunk_end = (chunk_start + max_batch_pages).min(dirty.len());
        let batch_pages = &dirty[chunk_start..chunk_end];
        chunk_start = chunk_end;
        jobs.clear();
        runs.clear();
        let Some((phase1_id, pages, map_queue, map_fence, fence_value, target_heaps)) = (if batch_pages.is_empty() {
            let cur_id = ACTIVE_STATE_ID.load(Ordering::Acquire);
            if cur_id == 0 {
                None
            } else {
                if !cached.as_ref().is_some_and(|c| c.state_id == cur_id) {
                    let guard = decant_trace::timed_lock("title_mem::STATE_READ", || STATE.read().unwrap_or_else(PoisonError::into_inner));
                    if let Some(st) = guard.as_ref().filter(|st| st.id == cur_id) {
                        *cached = Some(CachedSyncState {
                            state_id: st.id,
                            map_queue: st.map_queue.clone(),
                            map_fence: st.map_fence.clone(),
                            fence_value: st.fence_value.clone(),
                        });
                    }
                }
                cached
                    .as_ref()
                    .filter(|c| c.state_id == cur_id)
                    .map(|c| (c.state_id, 0u64, c.map_queue.clone(), c.map_fence.clone(), c.fence_value.clone(), Vec::new()))
            }
        } else {
            loop {
                jobs.clear();
                runs.clear();
                let res = with_state(|st, api| {
                    if !cached.as_ref().is_some_and(|c| c.state_id == st.id) {
                        *cached = Some(CachedSyncState {
                            state_id: st.id,
                            map_queue: st.map_queue.clone(),
                            map_fence: st.map_fence.clone(),
                            fence_value: st.fence_value.clone(),
                        });
                    }
                    match st.plan_upload(api, batch_pages, jobs, runs) {
                        Ok(pages) => {
                            let mut batch_written = std::mem::take(&mut st.written);
                            written.append(&mut batch_written);
                            st.written = batch_written;
                            let mut target_heaps: Vec<Option<ID3D12Resource>> = Vec::new();
                            if !runs.is_empty() {
                                let max_heap = runs.iter().map(|r| r.heap as usize).max().unwrap_or(0);
                                target_heaps.resize(max_heap + 1, None);
                                for r in runs.iter() {
                                    let idx = r.heap as usize;
                                    if target_heaps[idx].is_none()
                                        && let Some(Some(h)) = st.heaps.get(idx)
                                    {
                                        target_heaps[idx] = Some(h.buffer.clone());
                                    }
                                }
                            }
                            Ok((st.id, pages, st.map_queue.clone(), st.map_fence.clone(), st.fence_value.clone(), target_heaps))
                        }
                        Err(missing_heaps) => Err((st.id, missing_heaps)),
                    }
                });
                match res {
                    None => break None,
                    Some(Ok(ok)) => break Some(ok),
                    Some(Err((state_id, missing_heaps))) => {
                        ensure_heaps_outside_state(state_id, &missing_heaps);
                    }
                }
            }
        }) else {
            SYNC_WRITTEN.with(|cell| *cell.borrow_mut() = written);
            return;
        };

        let diff_enabled = subpage_diff_enabled();
        let (staged_bytes, l1_skipped_bytes) = if !runs.is_empty() && staging.is_some() {
            // SAFETY: every staged `UploadRun` points to a live section view for this sync.
            let res = unsafe { filter_staged_runs(runs, l1_prints, jobs, diff_enabled, cores) };
            if diff_enabled {
                collect_changed_l1_tiles(runs, &mut written);
            }
            res
        } else {
            (0, 0)
        };
        let staging_span = if let (true, Some(ring)) = (staged_bytes > 0, staging.as_mut()) {
            let Some(base) = ring.reserve(staged_bytes, &map_fence) else {
                SYNC_WRITTEN.with(|cell| *cell.borrow_mut() = written);
                return;
            };
            let dst_base = ring.ptr + base;
            apply_staging_base(jobs, runs, dst_base);
            Some((base, staged_bytes))
        } else {
            None
        };

        let (copy_micros, copied_bytes, l0_skipped_bytes) = if !jobs.is_empty() {
            let copying = Instant::now();
            let _copy = decant_trace::span("title_mem", "copy_runs");
            // SAFETY: sorted unique pool pages have unique mirror tiles. Planning keeps each job
            // inside a live section and heap (or reserved staging span), and SYNC_LOCK keeps `device_created` from replacing heaps.
            let (copied_bytes, l0_skipped_bytes) = unsafe { execute_copies(jobs, cores, diff_enabled, &mut written) };
            if let (Some((base, staged_bytes)), Some(ring)) = (staging_span, staging.as_mut()) {
                ring.copy_to_heaps(&map_queue, &map_fence, &target_heaps, (base, staged_bytes), runs, &fence_value);
            }
            (copying.elapsed().as_micros() as u64, copied_bytes, l0_skipped_bytes)
        } else {
            (0, 0, 0)
        };
        let skipped_bytes = l1_skipped_bytes + l0_skipped_bytes;

        let tile_started = Instant::now();
        let (calls, missing_heaps, window_resources, mut heap_objs) = if MIRROR_MAY_HAVE_PENDING.load(Ordering::Acquire) {
            let mut guard = decant_trace::timed_lock("title_mem::STATE", || STATE.write().unwrap_or_else(PoisonError::into_inner));
            if let Some(st) = guard.as_mut().filter(|st| st.id == phase1_id) {
                MIRROR_MAY_HAVE_PENDING.store(false, Ordering::Release);
                let calls = st.mirror.drain_updates();
                let mut missing_heaps = Vec::new();
                let mut window_resources: Vec<Option<ID3D12Resource>> = Vec::new();
                let mut heap_objs: HashMap<u32, ID3D12Heap> = HashMap::new();
                for call in &calls {
                    if let Some(h) = call.heap {
                        if let Some(Some(pool)) = st.heaps.get(h as usize) {
                            heap_objs.entry(h).or_insert_with(|| pool._heap.clone());
                        } else if !st.failed.contains(&h) && !missing_heaps.contains(&h) {
                            missing_heaps.push(h);
                        }
                    }
                    if window_resources.len() <= call.buffer {
                        window_resources.resize(call.buffer + 1, None);
                    }
                    if window_resources[call.buffer].is_none()
                        && let Some(Some(w)) = st.windows.get(call.buffer)
                    {
                        window_resources[call.buffer] = Some(w.resource.clone());
                    }
                }
                (calls, missing_heaps, window_resources, heap_objs)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), HashMap::new())
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), HashMap::new())
        };
        let mut ranges = 0u64;
        if !calls.is_empty() {
            let _tile_span = decant_trace::span("title_mem", "update_tiles");
            if !missing_heaps.is_empty() {
                ensure_heaps_outside_state(phase1_id, &missing_heaps);
                let guard = STATE.read().unwrap_or_else(PoisonError::into_inner);
                if let Some(st) = guard.as_ref().filter(|st| st.id == phase1_id) {
                    for h in missing_heaps {
                        if let Some(Some(pool)) = st.heaps.get(h as usize) {
                            heap_objs.insert(h, pool._heap.clone());
                        }
                    }
                }
            }
            {
                let mut scratch = MAP_QUEUE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
                for mut call in calls {
                    let Some(Some(window)) = window_resources.get(call.buffer) else { continue };
                    let heap = call.heap.and_then(|h| heap_objs.get(&h));
                    if call.heap.is_some() && heap.is_none() {
                        call.heap = None;
                    }
                    ranges += call.ranges.len() as u64;
                    update_call(&map_queue, window, &call, heap, &mut scratch);
                }
                let next_val = fence_value.fetch_add(1, Ordering::AcqRel) + 1;
                // SAFETY: both objects are live.
                if let Err(e) = unsafe { map_queue.Signal(&map_fence, next_val) } {
                    xs_log!("title memory: the map fence cannot be signalled ({e})");
                }
            }
        }
        total_pages += pages;
        total_copied_bytes += copied_bytes;
        total_skipped_bytes += skipped_bytes;
        total_ranges += ranges;
        total_copy_micros += copy_micros;
        total_tile_micros += tile_started.elapsed().as_micros() as u64;
        last_fence = Some((map_fence, fence_value));
    }
    if let Some((map_fence, fence_value)) = last_fence {
        let cur_fence = fence_value.load(Ordering::Acquire);
        if cur_fence > 0 && waited.get(&(queue as usize)).is_none_or(|&v| v < cur_fence) {
            // SAFETY: both objects are live. The queue waits on the GPU, the CPU does not.
            match unsafe { queue_ref.Wait(&map_fence, cur_fence) } {
                Ok(()) => {
                    waited.insert(queue as usize, cur_fence);
                }
                Err(e) => xs_log!("title memory: a queue cannot wait for the tile mappings ({e})"),
            }
        }
    }
    crate::diagnose::removed_after(|| format!("a sync of {total_pages} pages and {total_ranges} tile ranges"));
    maybe_stats_and_guard(
        stats,
        last_guard,
        SyncBatchMetrics {
            pages: total_pages,
            copied_bytes: total_copied_bytes,
            skipped_bytes: total_skipped_bytes,
            ranges: total_ranges,
            copy_micros: total_copy_micros,
            tile_micros: total_tile_micros,
        },
        started,
        &mut run_guard,
    );
    drop(sync_scratch);
    if run_guard {
        // SAFETY: the export takes no argument.
        unsafe { (api_ref.guard_modules)() };
    }
    let page_units = total_pages / 16;
    let range_units = total_ranges / 4;
    note_streaming_work(page_units.max(range_units));
    // A placed texture whose memory the title wrote is filled again (after this lock is given back:
    // the detile table asks for tiles under its own lock).
    if crate::detile::filter_overlapping_tiles(&mut written) {
        written.sort_unstable();
        written.dedup();
        let _note_tiles = decant_trace::span("title_mem", "note_tiles");
        crate::detile::note_tiles(&written);
    }
    written.clear();
    SYNC_WRITTEN.with(|cell| *cell.borrow_mut() = written);
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
    with_state_read(|st| {
        Some(
            spans
                .iter()
                .map(|&(va, len)| {
                    let (first, pages) = crate::title_mirror::page_span(va, len);
                    st.mirror.touches(first, pages, written)
                })
                .collect(),
        )
    })
    .unwrap_or_else(|| vec![None; spans.len()])
}

/// The mirror tiles of the pages that hold the `len` bytes at title address `va`, when all of them are
/// mapped pages of one reservation. `None` for anything else (including the module being off).
pub fn tiles_of(va: u64, len: u64) -> Option<Vec<u32>> {
    let (first, pages) = crate::title_mirror::page_span(va, len);
    with_state_read(|st| st.mirror.mirror_tiles_read(first, pages)).or_else(|| {
        with_state(|st, _| {
            let prev_tiles = st.mirror.tiles_given();
            let got = st.mirror.mirror_tiles(first, pages);
            if st.mirror.tiles_given() != prev_tiles {
                MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
            }
            got
        })
        .flatten()
    })
}

/// Mirror tiles for each `(va, len)` span under one state lock, taking only `STATE.read()` for
/// spans that already have mirror tiles and touching `STATE.write()` only for misses.
pub fn tiles_of_batch(spans: &[(u64, u64)], out: &mut Vec<Option<Vec<u32>>>) {
    out.clear();
    if spans.is_empty() {
        return;
    }
    let Some((mut partial, misses)) = with_state_read(|st| {
        let mut tmp = Vec::with_capacity(spans.len());
        let mut misses = 0usize;
        for &(va, len) in spans {
            let (first, pages) = crate::title_mirror::page_span(va, len);
            let got = st.mirror.mirror_tiles_read(first, pages);
            if got.is_none() {
                misses += 1;
            }
            tmp.push(got);
        }
        Some((tmp, misses))
    }) else {
        out.resize(spans.len(), None);
        return;
    };
    if misses == 0 {
        *out = partial;
        return;
    }
    if with_state(|st, _| {
        let prev_tiles = st.mirror.tiles_given();
        for (slot, &(va, len)) in partial.iter_mut().zip(spans) {
            if slot.is_none() {
                let (first, pages) = crate::title_mirror::page_span(va, len);
                *slot = st.mirror.mirror_tiles(first, pages);
            }
        }
        if st.mirror.tiles_given() != prev_tiles {
            MIRROR_MAY_HAVE_PENDING.store(true, Ordering::Release);
        }
    })
    .is_none()
    {
        out.resize(spans.len(), None);
        return;
    }
    *out = partial;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_planning_bounds_chunks_and_thread_counts() {
        assert_eq!(copy_threads(0, 8), 1);
        assert_eq!(copy_threads(7, 8), 1);
        assert_eq!(copy_threads(8, 8), 2);
        assert_eq!(copy_threads(24, 8), 3);
        assert_eq!(copy_threads(4096, 3), 3);
        assert_eq!(copy_threads(4096, 1), 1);
        assert_eq!(copy_threads(4096, 0), 1);
        let mut jobs = Vec::new();
        plan_copy(&mut jobs, 0, 1024, 0, false);
        assert!(jobs.is_empty());
        plan_copy(&mut jobs, 0, 1024, 15, false);
        assert_eq!(jobs.iter().map(|job| job.bytes / PAGE as usize).collect::<Vec<_>>(), [4, 4, 4, 3]);
        for (i, job) in jobs.iter().enumerate() {
            assert_eq!(job.source, i * COPY_CHUNK_PAGES * PAGE as usize);
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
                plan_copy(&mut jobs, source.as_ptr() as usize + src * PAGE as usize, output.as_mut_ptr() as usize + dst * PAGE as usize, count, false);
            }
            for _ in 0..3 {
                // SAFETY: vectors live until the workers join and the destination spans are disjoint.
                unsafe { execute_copies(&jobs, cores, false, &mut Vec::new()) };
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
        let runs = runs_by_heap(&[5, 6, 7, HEAP_PAGES - 1, HEAP_PAGES, HEAP_PAGES + 1, 9]);
        assert_eq!(runs, [(0, vec![(0, 3, 5), (3, 1, HEAP_PAGES - 1), (6, 1, 9)]), (1, vec![(4, 2, 0)])], "a run ends at a heap's end and where the pages jump");
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
        let before_layout = mapping_generation();
        let before_tile = tile_generation();
        bump_mapping_gen();
        assert!(mapping_generation() > before_layout);
        assert_eq!(tile_generation(), before_tile);
        bump_tile_gen();
        assert!(tile_generation() > before_tile);

        let _sync_guard = SYNC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let done = std::thread::spawn(|| translate(0x1000, 64)).join().unwrap();
        assert!(done.is_none());
    }

    #[test]
    fn journal_seq_skip_logic_and_rwlock_allow_concurrent_reads() {
        assert!(!should_skip_drain(None, None));
        assert!(!should_skip_drain(None, Some(0)));
        assert!(!should_skip_drain(Some(5), None));
        assert!(!should_skip_drain(Some(5), Some(6)));
        assert!(should_skip_drain(Some(5), Some(5)));

        let r1 = STATE.read().unwrap_or_else(PoisonError::into_inner);
        let r2 = std::thread::spawn(|| STATE.read().unwrap_or_else(PoisonError::into_inner).is_some()).join().unwrap();
        assert_eq!(r2, r1.is_some());
    }

    #[test]
    fn drain_lock_does_not_block_state_reads_or_writes() {
        let mut cursor = DRAIN_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        cursor.state_id = 0;
        let read_ok = std::thread::spawn(|| STATE.read().unwrap_or_else(PoisonError::into_inner).is_none()).join().unwrap();
        assert!(read_ok);
    }

    #[test]
    fn pool_heaps_are_16_mib_aligned_to_64_kib_pages() {
        assert_eq!(HEAP_PAGES, 256);
        assert_eq!(POOL_HEAP_BYTES, 16 * 1024 * 1024);
        assert_eq!(POOL_HEAP_BYTES % PAGE, 0);
    }

    #[test]
    fn journal_unchanged_fast_path_checks_tracking_and_last_seq() {
        unsafe extern "system" fn stub_track(_: i32) {}
        unsafe extern "system" fn stub_changes(_: u64, _: *mut GpuChange, _: u32, _: *mut u32) -> i32 { 1 }
        unsafe extern "system" fn stub_snapshot(_: *mut GpuChange, _: u32, _: *mut u32, _: *mut u64) -> i32 { 1 }
        unsafe extern "system" fn stub_take(_: *mut u32, _: u32) -> u32 { 0 }
        unsafe extern "system" fn stub_view(_: u32, _: *mut usize, _: *mut u32, _: *mut u32) -> i32 { 0 }
        unsafe extern "system" fn stub_guard() {}
        unsafe extern "system" fn stub_writable(_: usize, _: usize) -> i32 { 0 }
        unsafe extern "system" fn stub_use(_: usize, _: usize) -> u32 { 0 }
        unsafe extern "system" fn stub_seq_7() -> u64 { 7 }

        let api = Api {
            track: stub_track,
            changes: stub_changes,
            snapshot: stub_snapshot,
            take_dirty: stub_take,
            section_view: stub_view,
            guard_modules: stub_guard,
            writable: stub_writable,
            gpu_use: stub_use,
            show: None,
            journal_seq: Some(stub_seq_7),
        };
        let prev_track = TRACKING_ACTIVE.swap(false, Ordering::AcqRel);
        let prev_seq = LAST_JOURNAL_SEQ.swap(7, Ordering::AcqRel);
        assert!(!journal_unchanged(&api));
        TRACKING_ACTIVE.store(true, Ordering::Release);
        assert!(journal_unchanged(&api));
        LAST_JOURNAL_SEQ.store(6, Ordering::Release);
        assert!(!journal_unchanged(&api));
        TRACKING_ACTIVE.store(prev_track, Ordering::Release);
        LAST_JOURNAL_SEQ.store(prev_seq, Ordering::Release);
    }

    #[test]
    fn take_dirty_outside_state_sorts_dedups_and_does_not_lock_state() {
        unsafe extern "system" fn stub_track(_: i32) {}
        unsafe extern "system" fn stub_changes(_: u64, _: *mut GpuChange, _: u32, _: *mut u32) -> i32 { 1 }
        unsafe extern "system" fn stub_snapshot(_: *mut GpuChange, _: u32, _: *mut u32, _: *mut u64) -> i32 { 1 }
        unsafe extern "system" fn stub_take(out: *mut u32, cap: u32) -> u32 {
            assert!(cap >= 4);
            // SAFETY: `out` has room for `cap` u32 entries.
            unsafe {
                *out.add(0) = 9;
                *out.add(1) = 3;
                *out.add(2) = 9;
                *out.add(3) = 1;
            }
            4
        }
        unsafe extern "system" fn stub_view(_: u32, _: *mut usize, _: *mut u32, _: *mut u32) -> i32 { 0 }
        unsafe extern "system" fn stub_guard() {}
        unsafe extern "system" fn stub_writable(_: usize, _: usize) -> i32 { 0 }
        unsafe extern "system" fn stub_use(_: usize, _: usize) -> u32 { 0 }

        let api = Api {
            track: stub_track,
            changes: stub_changes,
            snapshot: stub_snapshot,
            take_dirty: stub_take,
            section_view: stub_view,
            guard_modules: stub_guard,
            writable: stub_writable,
            gpu_use: stub_use,
            show: None,
            journal_seq: None,
        };
        let mut dirty = Vec::new();
        let _ = take_dirty_outside_state(&api, &mut dirty);
        assert_eq!(dirty, [1, 3, 9]);
    }

    #[test]
    fn mirror_pool_mode_parses_auto_l0_and_l1_overrides() {
        assert_eq!(parse_mirror_pool_mode(None), MirrorPoolMode::Auto);
        assert_eq!(parse_mirror_pool_mode(Some("auto")), MirrorPoolMode::Auto);
        assert_eq!(parse_mirror_pool_mode(Some("l0")), MirrorPoolMode::StrictL0);
        assert_eq!(parse_mirror_pool_mode(Some("L0")), MirrorPoolMode::StrictL0);
        assert_eq!(parse_mirror_pool_mode(Some("l1")), MirrorPoolMode::PreferL1);
        assert_eq!(parse_mirror_pool_mode(Some("L1")), MirrorPoolMode::PreferL1);
        assert_eq!(parse_mirror_l1_mib(Some("2048")), Some(2048 << 20));
        assert_eq!(parse_mirror_l1_mib(None), None);
    }

    #[test]
    fn vram_budget_matches_design_note_f_tiers() {
        let b8 = VramBudget::from_budget(7_680 << 20);
        assert_eq!(b8.r_drv, 1_536 << 20);
        assert_eq!(b8.hwm_total, 6_144 << 20);
        assert_eq!(b8.lwm_total, 5_120 << 20);
        assert_eq!(b8.m_cap, 0);

        let b12 = VramBudget::from_budget(11_520 << 20);
        assert_eq!(b12.m_cap, 1_024 << 20);
        assert_eq!(b12.r_drv, 1_843 << 20);
        assert_eq!(b12.hwm_total, 9_677 << 20);

        let b16 = VramBudget::from_budget(15_360 << 20);
        assert_eq!(b16.m_cap, 2_048 << 20);
        assert_eq!(b16.hwm_total, 12_903 << 20);
        assert_eq!(b16.lwm_total, 11_367 << 20);

        let b24 = VramBudget::from_budget(23_558 << 20);
        assert_eq!(b24.r_drv, 3_769 << 20);
        assert_eq!(b24.hwm_total, 19_789 << 20);
        assert_eq!(b24.lwm_total, 17_434 << 20);
        assert_eq!(b24.m_cap, 4_096 << 20);
    }

    #[test]
    fn hybrid_mirror_spills_from_l1_to_l0_at_m_cap_and_hwm() {
        let budget_24 = 23_558 << 20;
        // Under both HWM_total and M_cap: Auto places heap in L1.
        assert!(should_use_l1(MirrorPoolMode::Auto, budget_24, 10_000 << 20, 0, None));
        assert!(should_use_l1(MirrorPoolMode::Auto, budget_24, 10_000 << 20, 4_080 << 20, None));
        // At M_cap (4096 MiB): Auto spills cleanly to L0.
        assert!(!should_use_l1(MirrorPoolMode::Auto, budget_24, 10_000 << 20, 4_096 << 20, None));
        // Near HWM_total (19789 MiB): Auto and PreferL1 both spill to L0 before OOM.
        assert!(!should_use_l1(MirrorPoolMode::Auto, budget_24, 19_780 << 20, 1_024 << 20, None));
        assert!(!should_use_l1(MirrorPoolMode::PreferL1, budget_24, 19_780 << 20, 4_096 << 20, None));
        // PreferL1 allows beyond M_cap while below HWM_total.
        assert!(should_use_l1(MirrorPoolMode::PreferL1, budget_24, 10_000 << 20, 5_000 << 20, None));
        // StrictL0 always stays on L0.
        assert!(!should_use_l1(MirrorPoolMode::StrictL0, budget_24, 0, 0, None));
        // 8 GiB class has M_cap = 0 so Auto stays on L0.
        assert!(!should_use_l1(MirrorPoolMode::Auto, 7_680 << 20, 1_000 << 20, 0, None));
    }

    #[test]
    fn mixed_l1_and_l0_jobs_rebase_only_staged_destinations() {
        let mut jobs = Vec::new();
        let mut runs = Vec::new();
        let l0_ptr = 0x7ff0_0000_0000usize;
        // Two staged L1 pages (offset 0) and three direct L0 pages (absolute host pointer).
        plan_copy(&mut jobs, 0x1000_0000, 0, 2, true);
        runs.push(UploadRun {
            source: 0x1000_0000,
            heap: 0,
            heap_offset: 0,
            staging_offset: 0,
            bytes: 2 * PAGE,
        });
        plan_copy(&mut jobs, 0x2000_0000, l0_ptr + PAGE as usize, 3, false);

        let dst_base = 0x1234_0000usize;
        let staged_bytes = apply_staging_base(&mut jobs, &runs, dst_base);
        assert_eq!(staged_bytes, 2 * PAGE as usize);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].destination, dst_base);
        assert_eq!(jobs[1].destination, l0_ptr + PAGE as usize);
    }

    #[test]
    fn read_fastpath_falls_back_on_on_demand_or_missing_tile_and_hits_when_ready() {
        let mut m = Mirror::new(1 << 30);
        m.apply(Change::Reserve { va: 0x1000_0000, pages: 2, on_demand: true });
        m.apply(Change::Map { va: 0x1000_0000, pages: 1, first_page: 5 });
        // Page 1 is unmapped in an on-demand reservation: wants_use is true, must fall back.
        assert!(m.wants_use(0x1000_0000, PAGE * 2));
        assert!(!m.wants_use(0x1000_0000, PAGE));
        // Page 0 has no tile yet: must fall back until given.
        assert!(!m.has_tiles_for_span(0x1000_0000, PAGE));
        m.give_span(0x1000_0000, PAGE);
        assert!(m.has_tiles_for_span(0x1000_0000, PAGE));
        assert!(should_skip_drain(Some(7), Some(7)));
        assert!(!should_skip_drain(Some(7), Some(8)));
    }

    #[test]
    fn sync_lock_staging_does_not_block_state_read() {
        let mut sync_guard = SYNC_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        sync_guard.waited.get_or_insert_with(HashMap::new).insert(0x1234, 7);
        let read_guard = STATE.try_read();
        assert!(read_guard.is_ok(), "STATE.read() must not block while SYNC_LOCK / staging is held");
        drop(read_guard);
        sync_guard.waited = None;
    }

    #[test]
    fn leaf_creation_and_map_queue_locks_do_not_block_state_read() {
        let _w = WINDOW_MAKE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let _h = HEAP_MAKE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let _m = MAP_QUEUE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(STATE.try_read().is_ok(), "STATE.read() must not block while leaf creation/map locks are held");
    }

    #[test]
    fn l0_subpage_diff_skips_unchanged_subpages_and_copies_changed_ones() {
        let mut src = vec![0xAAu8; PAGE as usize];
        let mut dst = vec![0xAAu8; PAGE as usize];
        // Mutate sub-pages 2, 3 (contiguous) and 11.
        src[2 * SUBPAGE_BYTES + 7] = 0x55;
        src[3 * SUBPAGE_BYTES + 19] = 0x66;
        src[11 * SUBPAGE_BYTES + 1] = 0x77;
        let jobs = [CopyJob {
            source: src.as_ptr() as usize,
            destination: dst.as_mut_ptr() as usize,
            bytes: PAGE as usize,
            staged: false,
            first_tile: 7,
        }];
        let mut tiles = Vec::new();
        let (copied, skipped) = unsafe { execute_copies(&jobs, 1, true, &mut tiles) };
        assert_eq!(copied, 3 * SUBPAGE_BYTES as u64);
        assert_eq!(skipped, (SUBPAGES_PER_PAGE - 3) as u64 * SUBPAGE_BYTES as u64);
        assert_eq!(src, dst);
        assert_eq!(tiles, vec![7]);
    }

    #[test]
    fn l1_subpage_diff_hashes_coalesces_changed_runs_and_skips_clean_pages() {
        let mut src = vec![0x11u8; PAGE as usize];
        let mut prints = L1Fingerprints::default();
        let mut runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 0,
            heap_offset: 2 * PAGE,
            staging_offset: 0,
            bytes: PAGE,
        }];
        let mut jobs = Vec::new();
        // First copy populates fingerprints and uploads the full 64 KiB page.
        let (staged1, skipped1) = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        assert_eq!((staged1, skipped1), (PAGE as usize, 0));
        assert_eq!(runs.len(), 1);
        assert_eq!(jobs.len(), 1);

        // Unchanged second copy skips 100% (0 staged bytes).
        runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 0,
            heap_offset: 2 * PAGE,
            staging_offset: 0,
            bytes: PAGE,
        }];
        jobs.clear();
        let (staged2, skipped2) = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        assert_eq!((staged2, skipped2), (0, PAGE));
        assert!(runs.is_empty());
        assert!(jobs.is_empty());

        // Mutate sub-pages 4, 5 (coalesced) and 14 -> 2 UploadRuns totaling 12 KiB.
        src[4 * SUBPAGE_BYTES] = 0x99;
        src[5 * SUBPAGE_BYTES + 100] = 0x88;
        src[14 * SUBPAGE_BYTES + 42] = 0x77;
        runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 0,
            heap_offset: 2 * PAGE,
            staging_offset: 0,
            bytes: PAGE,
        }];
        jobs.clear();
        let (staged3, skipped3) = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        assert_eq!((staged3, skipped3), (3 * SUBPAGE_BYTES, (SUBPAGES_PER_PAGE - 3) as u64 * SUBPAGE_BYTES as u64));
        assert_eq!(runs.len(), 2);
        assert_eq!((runs[0].heap_offset, runs[0].bytes), (2 * PAGE + 4 * SUBPAGE_BYTES as u64, 2 * SUBPAGE_BYTES as u64));
        assert_eq!((runs[1].heap_offset, runs[1].bytes), (2 * PAGE + 14 * SUBPAGE_BYTES as u64, SUBPAGE_BYTES as u64));
        assert_eq!(jobs.len(), 2);
    }

    #[test]
    fn subpage_diff_disabled_copies_full_64k_pages() {
        assert!(parse_subpage_diff(None));
        assert!(!parse_subpage_diff(Some("0")));
        assert!(!parse_subpage_diff(Some("OFF")));
        assert!(!parse_subpage_diff(Some(" false ")));

        let src = vec![0x33u8; PAGE as usize];
        let mut dst = vec![0x33u8; PAGE as usize];
        let jobs = [CopyJob {
            source: src.as_ptr() as usize,
            destination: dst.as_mut_ptr() as usize,
            bytes: PAGE as usize,
            staged: false,
            first_tile: 0,
        }];
        let (copied, skipped) = unsafe { execute_copies(&jobs, 1, false, &mut Vec::new()) };
        assert_eq!((copied, skipped), (PAGE, 0));

        let mut prints = L1Fingerprints::default();
        let mut runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 0,
            heap_offset: 0,
            staging_offset: 0,
            bytes: PAGE,
        }];
        let mut staged_jobs = Vec::new();
        let (staged, l1_skipped) = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut staged_jobs, false, 1) };
        assert_eq!((staged, l1_skipped), (PAGE as usize, 0));
    }

    #[test]
    fn parallel_filter_staged_runs_matches_single_threaded_output() {
        const PAGES: usize = 24;
        let mut src_seq = vec![0x22u8; PAGES * PAGE as usize];
        let mut src_par = src_seq.clone();
        let mut prints_seq = L1Fingerprints::default();
        let mut prints_par = L1Fingerprints::default();

        // Initial pass populates 24 tiles on both.
        let mut runs_seq = vec![UploadRun { source: src_seq.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: PAGES as u64 * PAGE }];
        let mut runs_par = vec![UploadRun { source: src_par.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: PAGES as u64 * PAGE }];
        let (mut jobs_seq, mut jobs_par) = (Vec::new(), Vec::new());
        let res1_seq = unsafe { filter_staged_runs(&mut runs_seq, &mut prints_seq, &mut jobs_seq, true, 1) };
        let res1_par = unsafe { filter_staged_runs(&mut runs_par, &mut prints_par, &mut jobs_par, true, 4) };
        assert_eq!(res1_seq, res1_par);
        assert_eq!(runs_seq, runs_par);

        // Mutate scattered sub-pages across several tiles in two separate runs.
        for &(tile, sp, val) in &[(0usize, 3usize, 0xA1u8), (0, 4, 0xA2), (7, 15, 0xB1), (16, 0, 0xC1), (23, 9, 0xD1)] {
            let off = tile * PAGE as usize + sp * SUBPAGE_BYTES + 17;
            src_seq[off] = val;
            src_par[off] = val;
        }
        runs_seq = vec![
            UploadRun { source: src_seq.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: 16 * PAGE },
            UploadRun { source: unsafe { src_seq.as_ptr().add(16 * PAGE as usize) } as usize, heap: 0, heap_offset: 16 * PAGE, staging_offset: 0, bytes: 8 * PAGE },
        ];
        runs_par = vec![
            UploadRun { source: src_par.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: 16 * PAGE },
            UploadRun { source: unsafe { src_par.as_ptr().add(16 * PAGE as usize) } as usize, heap: 0, heap_offset: 16 * PAGE, staging_offset: 0, bytes: 8 * PAGE },
        ];
        jobs_seq.clear();
        jobs_par.clear();
        let res2_seq = unsafe { filter_staged_runs(&mut runs_seq, &mut prints_seq, &mut jobs_seq, true, 1) };
        let res2_par = unsafe { filter_staged_runs(&mut runs_par, &mut prints_par, &mut jobs_par, true, 4) };
        assert_eq!(res2_seq, res2_par);
        assert_eq!(res2_par, (5 * SUBPAGE_BYTES, (PAGES * SUBPAGES_PER_PAGE - 5) as u64 * SUBPAGE_BYTES as u64));
        assert_eq!(runs_seq, runs_par);
        assert_eq!(jobs_seq.len(), jobs_par.len());
        assert_eq!(prints_seq.heap_mut(0).hashes, prints_par.heap_mut(0).hashes);
    }

    #[test]
    fn l0_unchanged_pages_emit_no_detile_tiles_and_changed_pages_emit_only_changed() {
        const PAGES: usize = 6;
        let mut src = vec![0x44u8; PAGES * PAGE as usize];
        let mut dst = src.clone();
        let mut jobs = Vec::new();
        plan_copy_tiles(&mut jobs, src.as_ptr() as usize, dst.as_mut_ptr() as usize, PAGES, false, 100);

        // Second sync with zero changed bytes emits no tiles.
        let mut unchanged_tiles = Vec::new();
        let (copied0, skipped0) = unsafe { execute_copies(&jobs, 2, true, &mut unchanged_tiles) };
        assert_eq!((copied0, skipped0), (0, PAGES as u64 * PAGE));
        assert!(unchanged_tiles.is_empty());

        // Mutate one subpage on pages 1 and 5 only.
        src[PAGE as usize + 5 * SUBPAGE_BYTES + 3] = 0x91;
        src[5 * PAGE as usize + 12 * SUBPAGE_BYTES + 8] = 0x92;
        let mut changed_tiles = Vec::new();
        let (copied1, skipped1) = unsafe { execute_copies(&jobs, 2, true, &mut changed_tiles) };
        assert_eq!(copied1, 2 * SUBPAGE_BYTES as u64);
        assert_eq!(skipped1, (PAGES * SUBPAGES_PER_PAGE - 2) as u64 * SUBPAGE_BYTES as u64);
        changed_tiles.sort_unstable();
        assert_eq!(changed_tiles, vec![101, 105]);
    }

    #[test]
    fn l1_unchanged_pages_emit_no_detile_tiles_and_first_or_changed_emit() {
        let mut src = vec![0x55u8; 3 * PAGE as usize];
        let mut prints = L1Fingerprints::default();
        let mut jobs = Vec::new();

        // First upload populates fingerprints and emits all three tiles.
        let mut runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 2,
            heap_offset: 10 * PAGE,
            staging_offset: 0,
            bytes: 3 * PAGE,
        }];
        let _ = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        let mut first_tiles = Vec::new();
        collect_changed_l1_tiles(&runs, &mut first_tiles);
        assert_eq!(first_tiles, vec![522, 523, 524]);

        // Unchanged second sync skips everything and emits no tiles.
        runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 2,
            heap_offset: 10 * PAGE,
            staging_offset: 0,
            bytes: 3 * PAGE,
        }];
        jobs.clear();
        let _ = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        let mut second_tiles = Vec::new();
        collect_changed_l1_tiles(&runs, &mut second_tiles);
        assert!(second_tiles.is_empty());

        // Mutate one subpage on the middle tile only.
        src[PAGE as usize + 7 * SUBPAGE_BYTES + 11] = 0xCC;
        runs = vec![UploadRun {
            source: src.as_ptr() as usize,
            heap: 2,
            heap_offset: 10 * PAGE,
            staging_offset: 0,
            bytes: 3 * PAGE,
        }];
        jobs.clear();
        let _ = unsafe { filter_staged_runs(&mut runs, &mut prints, &mut jobs, true, 1) };
        let mut third_tiles = Vec::new();
        collect_changed_l1_tiles(&runs, &mut third_tiles);
        assert_eq!(third_tiles, vec![523]);
    }

    #[test]
    fn host_ram_l0_cap_handles_16_32_64_gib_and_pressure_tiers() {
        let g16 = 16_384 << 20;
        let g32 = 32_768 << 20;
        let g64 = 65_536 << 20;
        assert_eq!(host_ram_l0_cap(g16, 8_000 << 20, 8_000 << 20, None), 1_024 << 20);
        assert_eq!(host_ram_l0_cap(g32, 12_000 << 20, 12_000 << 20, None), 2_048 << 20);
        assert_eq!(host_ram_l0_cap(g32, 3_500 << 20, 12_000 << 20, None), 1_024 << 20);
        assert_eq!(host_ram_l0_cap(g32, 12_000 << 20, 3_500 << 20, None), 1_024 << 20);
        assert_eq!(host_ram_l0_cap(g32, 2_000 << 20, 12_000 << 20, None), 512 << 20);
        assert_eq!(host_ram_l0_cap(g32, 12_000 << 20, 1_500 << 20, None), 512 << 20);
        assert_eq!(host_ram_l0_cap(g64, 30_000 << 20, 30_000 << 20, None), 6_144 << 20);
        assert_eq!(host_ram_l0_cap(g32, 2_000 << 20, 1_500 << 20, Some(768 << 20)), 768 << 20);
        assert_eq!(parse_mirror_l0_mib(Some("1536")), Some(1536 << 20));
        assert!(can_grow_l0(1_024 << 20, 2_048 << 20, 6_000 << 20, 6_000 << 20));
        assert!(!can_grow_l0(2_048 << 20, 2_048 << 20, 6_000 << 20, 6_000 << 20));
        assert!(!can_grow_l0(512 << 20, 2_048 << 20, 3_900 << 20, 6_000 << 20));
        assert!(!can_grow_l0(512 << 20, 2_048 << 20, 6_000 << 20, 1_900 << 20));
    }

    #[test]
    fn staging_ring_bytes_halves_to_32_mib_on_32_gib_hosts() {
        assert_eq!(staging_ring_bytes(16_384 << 20), 32 << 20);
        assert_eq!(staging_ring_bytes(32_768 << 20), 32 << 20);
        assert_eq!(staging_ring_bytes(65_536 << 20), 64 << 20);
        assert_eq!(staging_ring_bytes(0), 64 << 20);
    }

    #[test]
    fn hash_subpage_avalanche_and_repeatability_over_all_4k_bytes() {
        // Use an unaligned buffer (+3 bytes) to verify unaligned loads work safely.
        let mut backing = vec![0x3Cu8; SUBPAGE_BYTES + 16];
        for (i, b) in backing[3..3 + SUBPAGE_BYTES].iter_mut().enumerate() {
            *b = (i.wrapping_mul(37) ^ (i >> 3)) as u8;
        }
        let ptr = unsafe { backing.as_ptr().add(3) };
        let base_hash = unsafe { hash_subpage(ptr) };
        assert_eq!(base_hash, unsafe { hash_subpage(ptr) });
        let base_scalar = unsafe { hash_subpage_scalar(ptr) };
        assert_eq!(base_scalar, unsafe { hash_subpage_scalar(ptr) });

        // Flip every byte position across all 4096 bytes and verify both paths detect every change.
        for byte_idx in 0..SUBPAGE_BYTES {
            let bit = 1u8 << (byte_idx & 7);
            backing[3 + byte_idx] ^= bit;
            let h = unsafe { hash_subpage(ptr) };
            let hs = unsafe { hash_subpage_scalar(ptr) };
            assert_ne!(h, base_hash, "hash_subpage missed bit flip at byte {byte_idx}");
            assert_ne!(hs, base_scalar, "hash_subpage_scalar missed bit flip at byte {byte_idx}");
            backing[3 + byte_idx] ^= bit;
        }
        assert_eq!(unsafe { hash_subpage(ptr) }, base_hash);
    }

    #[test]
    fn parallel_diff_pool_matches_single_threaded_output_over_64_tiles() {
        const PAGES: usize = 80;
        let mut src_seq = vec![0x29u8; PAGES * PAGE as usize];
        let mut src_par = src_seq.clone();
        let mut prints_seq = L1Fingerprints::default();
        let mut prints_par = L1Fingerprints::default();

        let mut runs_seq = vec![UploadRun { source: src_seq.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: PAGES as u64 * PAGE }];
        let mut runs_par = vec![UploadRun { source: src_par.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: PAGES as u64 * PAGE }];
        let (mut jobs_seq, mut jobs_par) = (Vec::new(), Vec::new());
        let res1_seq = unsafe { filter_staged_runs(&mut runs_seq, &mut prints_seq, &mut jobs_seq, true, 1) };
        let res1_par = unsafe { filter_staged_runs(&mut runs_par, &mut prints_par, &mut jobs_par, true, 4) };
        assert_eq!(res1_seq, res1_par);
        assert_eq!(runs_seq, runs_par);

        for &(tile, sp, val) in &[(0usize, 1usize, 0xA1u8), (19, 15, 0xB2), (48, 0, 0xC3), (79, 8, 0xD4)] {
            let off = tile * PAGE as usize + sp * SUBPAGE_BYTES + 23;
            src_seq[off] = val;
            src_par[off] = val;
        }
        runs_seq = vec![
            UploadRun { source: src_seq.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: 48 * PAGE },
            UploadRun { source: unsafe { src_seq.as_ptr().add(48 * PAGE as usize) } as usize, heap: 0, heap_offset: 48 * PAGE, staging_offset: 0, bytes: 32 * PAGE },
        ];
        runs_par = vec![
            UploadRun { source: src_par.as_ptr() as usize, heap: 0, heap_offset: 0, staging_offset: 0, bytes: 48 * PAGE },
            UploadRun { source: unsafe { src_par.as_ptr().add(48 * PAGE as usize) } as usize, heap: 0, heap_offset: 48 * PAGE, staging_offset: 0, bytes: 32 * PAGE },
        ];
        jobs_seq.clear();
        jobs_par.clear();
        let res2_seq = unsafe { filter_staged_runs(&mut runs_seq, &mut prints_seq, &mut jobs_seq, true, 1) };
        let res2_par = unsafe { filter_staged_runs(&mut runs_par, &mut prints_par, &mut jobs_par, true, 4) };
        assert_eq!(res2_seq, res2_par);
        assert_eq!(res2_par, (4 * SUBPAGE_BYTES, (PAGES * SUBPAGES_PER_PAGE - 4) as u64 * SUBPAGE_BYTES as u64));
        assert_eq!(runs_seq, runs_par);
        assert_eq!(jobs_seq.len(), jobs_par.len());
    }

    #[test]
    fn window_make_lock_does_not_block_drain_lock_or_state_access() {
        let _w = WINDOW_MAKE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        std::thread::scope(|s| {
            s.spawn(|| {
                let _d = DRAIN_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
                assert!(STATE.try_read().is_ok());
            })
            .join()
            .unwrap();
        });
    }

    #[test]
    fn sync_journal_and_windows_fast_path_does_not_touch_drain_lock() {
        unsafe extern "system" fn stub_track(_: i32) {}
        unsafe extern "system" fn stub_changes(_: u64, _: *mut GpuChange, _: u32, _: *mut u32) -> i32 { 1 }
        unsafe extern "system" fn stub_snapshot(_: *mut GpuChange, _: u32, _: *mut u32, _: *mut u64) -> i32 { 1 }
        unsafe extern "system" fn stub_take(_: *mut u32, _: u32) -> u32 { 0 }
        unsafe extern "system" fn stub_view(_: u32, _: *mut usize, _: *mut u32, _: *mut u32) -> i32 { 0 }
        unsafe extern "system" fn stub_guard() {}
        unsafe extern "system" fn stub_writable(_: usize, _: usize) -> i32 { 0 }
        unsafe extern "system" fn stub_use(_: usize, _: usize) -> u32 { 0 }
        unsafe extern "system" fn stub_seq_11() -> u64 { 11 }

        let api = Api {
            track: stub_track,
            changes: stub_changes,
            snapshot: stub_snapshot,
            take_dirty: stub_take,
            section_view: stub_view,
            guard_modules: stub_guard,
            writable: stub_writable,
            gpu_use: stub_use,
            show: None,
            journal_seq: Some(stub_seq_11),
        };
        let prev_id = ACTIVE_STATE_ID.swap(99, Ordering::AcqRel);
        let prev_track = TRACKING_ACTIVE.swap(true, Ordering::AcqRel);
        let prev_seq = LAST_JOURNAL_SEQ.swap(11, Ordering::AcqRel);
        let prev_missing = MISSING_WINDOWS.swap(false, Ordering::AcqRel);

        // Hold DRAIN_LOCK on this thread: unchanged journal fast path must return true without blocking.
        let _drain_guard = DRAIN_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        let ok = std::thread::scope(|s| s.spawn(|| sync_journal_and_windows(&api)).join().unwrap());
        assert!(ok);
        drop(_drain_guard);

        MISSING_WINDOWS.store(prev_missing, Ordering::Release);
        LAST_JOURNAL_SEQ.store(prev_seq, Ordering::Release);
        TRACKING_ACTIVE.store(prev_track, Ordering::Release);
        ACTIVE_STATE_ID.store(prev_id, Ordering::Release);
    }

    #[test]
    fn give_outside_state_calls_gpu_use_without_holding_drain_lock_or_state() {
        unsafe extern "system" fn stub_track(_: i32) {}
        unsafe extern "system" fn stub_changes(_: u64, _: *mut GpuChange, _: u32, _: *mut u32) -> i32 { 1 }
        unsafe extern "system" fn stub_snapshot(_: *mut GpuChange, _: u32, _: *mut u32, _: *mut u64) -> i32 { 1 }
        unsafe extern "system" fn stub_take(_: *mut u32, _: u32) -> u32 { 0 }
        unsafe extern "system" fn stub_view(_: u32, _: *mut usize, _: *mut u32, _: *mut u32) -> i32 { 0 }
        unsafe extern "system" fn stub_guard() {}
        unsafe extern "system" fn stub_writable(_: usize, _: usize) -> i32 { 0 }
        unsafe extern "system" fn stub_use(_: usize, _: usize) -> u32 {
            assert!(DRAIN_LOCK.try_lock().is_ok(), "DRAIN_LOCK must not be held during gpu_use");
            assert!(STATE.try_write().is_ok(), "STATE must not be held during gpu_use");
            2
        }
        unsafe extern "system" fn stub_seq_19() -> u64 { 19 }

        let api = Api {
            track: stub_track,
            changes: stub_changes,
            snapshot: stub_snapshot,
            take_dirty: stub_take,
            section_view: stub_view,
            guard_modules: stub_guard,
            writable: stub_writable,
            gpu_use: stub_use,
            show: None,
            journal_seq: Some(stub_seq_19),
        };
        let prev_id = ACTIVE_STATE_ID.swap(1, Ordering::AcqRel);
        let prev_track = TRACKING_ACTIVE.swap(true, Ordering::AcqRel);
        let prev_seq = LAST_JOURNAL_SEQ.swap(19, Ordering::AcqRel);
        let prev_missing = MISSING_WINDOWS.swap(false, Ordering::AcqRel);

        give_outside_state(&api, &[(0x1000_0000, PAGE)]);

        MISSING_WINDOWS.store(prev_missing, Ordering::Release);
        LAST_JOURNAL_SEQ.store(prev_seq, Ordering::Release);
        TRACKING_ACTIVE.store(prev_track, Ordering::Release);
        ACTIVE_STATE_ID.store(prev_id, Ordering::Release);
    }

    #[test]
    fn drain_lock_never_blocks_on_state_write() {
        unsafe extern "system" fn stub_track(_: i32) {}
        unsafe extern "system" fn stub_changes(_: u64, out: *mut GpuChange, cap: u32, written: *mut u32) -> i32 {
            if cap >= 1 {
                // SAFETY: caller provides a valid buffer of at least `cap` elements.
                unsafe {
                    *out = GpuChange { seq: 100, kind: RESERVE, pages: 4, va: 0x3000_0000, first_page: 0, flags: 0 };
                    *written = 1;
                }
            }
            1
        }
        unsafe extern "system" fn stub_snapshot(_: *mut GpuChange, _: u32, _: *mut u32, _: *mut u64) -> i32 { 1 }
        unsafe extern "system" fn stub_take(_: *mut u32, _: u32) -> u32 { 0 }
        unsafe extern "system" fn stub_view(_: u32, _: *mut usize, _: *mut u32, _: *mut u32) -> i32 { 0 }
        unsafe extern "system" fn stub_guard() {}
        unsafe extern "system" fn stub_writable(_: usize, _: usize) -> i32 { 0 }
        unsafe extern "system" fn stub_use(_: usize, _: usize) -> u32 { 0 }
        unsafe extern "system" fn stub_seq_101() -> u64 { 101 }

        let api = Api {
            track: stub_track,
            changes: stub_changes,
            snapshot: stub_snapshot,
            take_dirty: stub_take,
            section_view: stub_view,
            guard_modules: stub_guard,
            writable: stub_writable,
            gpu_use: stub_use,
            show: None,
            journal_seq: Some(stub_seq_101),
        };
        let prev_id = ACTIVE_STATE_ID.swap(77, Ordering::AcqRel);
        let prev_track = TRACKING_ACTIVE.swap(true, Ordering::AcqRel);
        let prev_seq = LAST_JOURNAL_SEQ.swap(u64::MAX, Ordering::AcqRel);
        let prev_drained_seq = DRAINED_JOURNAL_SEQ.swap(u64::MAX, Ordering::AcqRel);
        let prev_drained_id = DRAINED_STATE_ID.swap(0, Ordering::AcqRel);
        let prev_has_pending = HAS_PENDING_JOURNAL.swap(false, Ordering::AcqRel);

        let _state_write = STATE.write().unwrap_or_else(PoisonError::into_inner);
        let drained_without_blocking = std::thread::scope(|s| {
            s.spawn(|| {
                let mut cursor = DRAIN_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
                let _ = drain_journal(&api, &mut cursor);
                assert!(DRAINED_JOURNAL_SEQ.load(Ordering::Acquire) == u64::MAX);
                true
            })
            .join()
            .unwrap()
        });
        assert!(drained_without_blocking);

        drop(_state_write);
        PENDING_JOURNAL.lock().unwrap_or_else(PoisonError::into_inner).clear();
        HAS_PENDING_JOURNAL.store(prev_has_pending, Ordering::Release);
        DRAINED_STATE_ID.store(prev_drained_id, Ordering::Release);
        DRAINED_JOURNAL_SEQ.store(prev_drained_seq, Ordering::Release);
        LAST_JOURNAL_SEQ.store(prev_seq, Ordering::Release);
        TRACKING_ACTIVE.store(prev_track, Ordering::Release);
        ACTIVE_STATE_ID.store(prev_id, Ordering::Release);
    }

    #[test]
    fn coalesce_changes_merges_contiguous_maps_and_unmaps_within_reservation() {
        let mut known_res = std::collections::BTreeMap::new();
        let mut recs = vec![
            // Adjacent MAPs with unknown reservation (pre-existing) must NOT be merged.
            GpuChange { seq: 1, kind: MAP, pages: 2, va: 0x1000_0000, first_page: 10, flags: 0 },
            GpuChange { seq: 2, kind: MAP, pages: 3, va: 0x1000_0000 + 2 * PAGE, first_page: 12, flags: 0 },
            // Known reservation: contiguous MAPs inside same RESERVE merge cleanly.
            GpuChange { seq: 3, kind: RESERVE, pages: 16, va: 0x2000_0000, first_page: 0, flags: 0 },
            GpuChange { seq: 4, kind: MAP, pages: 4, va: 0x2000_0000, first_page: 100, flags: 0 },
            GpuChange { seq: 5, kind: MAP, pages: 6, va: 0x2000_0000 + 4 * PAGE, first_page: 104, flags: 0 },
            // Discontiguous first_page must not merge.
            GpuChange { seq: 6, kind: MAP, pages: 2, va: 0x2000_0000 + 10 * PAGE, first_page: 200, flags: 0 },
            // Contiguous UNMAPs inside same known RESERVE merge cleanly.
            GpuChange { seq: 7, kind: UNMAP, pages: 4, va: 0x2000_0000, first_page: 0, flags: 0 },
            GpuChange { seq: 8, kind: UNMAP, pages: 6, va: 0x2000_0000 + 4 * PAGE, first_page: 0, flags: 0 },
            // Ephemeral RESERVE + MAP + UNMAP + RELEASE in one batch collapses to RELEASE.
            GpuChange { seq: 9, kind: RESERVE, pages: 8, va: 0x3000_0000, first_page: 0, flags: 0 },
            GpuChange { seq: 10, kind: MAP, pages: 8, va: 0x3000_0000, first_page: 300, flags: 0 },
            GpuChange { seq: 11, kind: UNMAP, pages: 8, va: 0x3000_0000, first_page: 0, flags: 0 },
            GpuChange { seq: 12, kind: RELEASE, va: 0x3000_0000, ..GpuChange::default() },
        ];
        coalesce_changes(&mut recs, &mut known_res, false);
        assert_eq!(recs.len(), 7);
        assert_eq!((recs[0].kind, recs[0].pages), (MAP, 2));
        assert_eq!((recs[1].kind, recs[1].pages), (MAP, 3));
        assert_eq!((recs[2].kind, recs[2].va, recs[2].pages), (RESERVE, 0x2000_0000, 16));
        assert_eq!((recs[3].kind, recs[3].pages, recs[3].first_page, recs[3].seq), (MAP, 10, 100, 5));
        assert_eq!((recs[4].kind, recs[4].pages, recs[4].first_page), (MAP, 2, 200));
        assert_eq!((recs[5].kind, recs[5].pages, recs[5].seq), (UNMAP, 10, 8));
        assert_eq!((recs[6].kind, recs[6].va, recs[6].seq), (RELEASE, 0x3000_0000, 12));

        // Second batch reuses persistent `known_res` for `0x2000_0000`: unsorted disjoint MAPs & UNMAPs coalesce!
        let mut batch2 = vec![
            GpuChange { seq: 13, kind: MAP, pages: 1, va: 0x2000_0000 + 2 * PAGE, first_page: 502, flags: 0 },
            GpuChange { seq: 14, kind: MAP, pages: 1, va: 0x2000_0000 + PAGE, first_page: 501, flags: 0 },
            GpuChange { seq: 15, kind: MAP, pages: 1, va: 0x2000_0000, first_page: 500, flags: 0 },
            GpuChange { seq: 16, kind: UNMAP, pages: 1, va: 0x2000_0000 + 2 * PAGE, first_page: 0, flags: 0 },
            GpuChange { seq: 17, kind: UNMAP, pages: 2, va: 0x2000_0000, first_page: 0, flags: 0 },
        ];
        coalesce_changes(&mut batch2, &mut known_res, false);
        assert_eq!(batch2.len(), 2);
        assert_eq!((batch2[0].kind, batch2[0].va, batch2[0].pages, batch2[0].first_page, batch2[0].seq), (MAP, 0x2000_0000, 3, 500, 15));
        assert_eq!((batch2[1].kind, batch2[1].va, batch2[1].pages, batch2[1].seq), (UNMAP, 0x2000_0000, 3, 17));
    }

    #[test]
    fn sync_fast_path_skips_state_write_when_dirty_empty_and_mirror_clean() {
        let prev_mirror_pending = MIRROR_MAY_HAVE_PENDING.swap(false, Ordering::AcqRel);
        let mut stats = Stats::default();
        let mut guard_timer = Some(Instant::now());
        let started = Instant::now();

        // Hold STATE.read() concurrently: any attempt to take STATE.write() on another thread blocks.
        let _state_read = STATE.read().unwrap_or_else(PoisonError::into_inner);
        assert!(STATE.try_write().is_err());

        let (ok, syncs) = std::thread::scope(|s| {
            s.spawn(|| {
                let mut run_guard = false;
                for _ in 0..(STATS_EVERY - 1) {
                    maybe_stats_and_guard(&mut stats, &mut guard_timer, SyncBatchMetrics::default(), started, &mut run_guard);
                }
                // Even on the STATS_EVERY tick, logging only uses STATE.read(), never STATE.write().
                maybe_stats_and_guard(&mut stats, &mut guard_timer, SyncBatchMetrics::default(), started, &mut run_guard);
                assert!(!MIRROR_MAY_HAVE_PENDING.load(Ordering::Acquire));
                (true, stats.syncs)
            })
            .join()
            .unwrap()
        });
        assert!(ok);
        assert_eq!(syncs, STATS_EVERY);

        drop(_state_read);
        MIRROR_MAY_HAVE_PENDING.store(prev_mirror_pending, Ordering::Release);
    }
}
