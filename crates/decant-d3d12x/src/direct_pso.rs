//! The direct shader path for pipelines made on first bind. When a pipeline of
//! `lazy_pso` is bound for the first time, its console shaders go to the translator
//! (`direct_source`, behind `shader_source`), the HLSL to DXC (`decant-dxc`) on the workers of
//! `compile_service`, and the DXIL to the disk cache (`direct_cache`).
//!
//! Order per pipeline:
//! 1. the direct path, when every stage translates and compiles,
//! 2. else the pipeline as the title created it, which takes the DXIL override files
//!    (`dxil_override`) as before,
//! 3. else the refusal behaviour of `lazy_pso` and `pso_diag`.
//!
//! Waiting is limited so that a burst of cold pipelines cannot freeze the recording thread.
//! The first bind of a graphics or mesh pipeline may wait up to [`FIRST_BIND_WAIT`], paid from a
//! budget shared by the whole frame ([`frame_wait_budget`], default 500 ms, refilled by
//! `PresentX`). When the budget is gone, cold graphics and mesh binds return at once.
//! A compute pipeline may wait up to [`compute_wait`] (default 5000 ms) from its first bind,
//! because dependent dispatches read what it writes. A compute pipeline that still is not ready
//! then is marked `late`, and its later binds do not wait again until the workers finish it.
//! A pipeline the PC runtime refuses with the compiled shaders is made from the title's shaders
//! instead (step 2).
//!
//! Switches (read once):
//! * `DECANT_D3D12X_DIRECT=1` turns the path on (default off).
//! * `DECANT_DXC_DIR=<dir>` is the first folder searched for `dxcompiler.dll` and `dxil.dll`.
//! * `DECANT_D3D12X_DIRECT_CACHE=<dir>` is the DXIL cache (default under `%LOCALAPPDATA%`, `0` off).
//! * `DECANT_D3D12X_DIRECT_DUMP=<dir>` gets the HLSL of each translated stage with its instruction
//!   listing, the reason of each refused pipeline, and the HLSL and the DXC message of each stage
//!   DXC refused.
//! * `DECANT_D3D12X_COMPUTE_WAIT_MS=<ms>` is how long a compute pipeline waits for its shaders and
//!   driver pipeline before being marked late (default 5000, `0` does not wait).
//! * `DECANT_D3D12X_FRAME_WAIT_MS=<ms>` is the total per-frame wait budget across cold graphics and
//!   mesh binds (default 500, `0` does not wait).
//! * `DECANT_D3D12X_ASYNC_BIND_FRAME_MS=<ms>` is the total per-frame wait budget across cold compute
//!   binds in the async bind mode of `lazy_pso` (default 16, `0` does not wait at all). `PresentX`
//!   refills it. Once it is spent, cold async binds return at once.
//! * `DECANT_D3D12X_PRECOMPILE_THREADS=<n>` sets the queue worker count (default two thirds of the
//!   cores, 4 to 32, 0 off).
//! * `DECANT_D3D12X_DIRECT_THREADS=<n>` sets the worker count for binds that bypass the queue
//!   (default half of the cores, 2 to 16).
//! * `DECANT_D3D12X_MAX_LIVE_HLSL_BYTES=<n>` is the largest stage HLSL compiled live (default 8 MiB,
//!   `0` no limit). A larger stage without cached DXIL is refused with its size in the log.
//! * `DECANT_D3D12X_PSO_CACHE=<dir>` is the library of driver compiled pipeline states
//!   (`pso_library`, default `%LOCALAPPDATA%\Decant\pso-library`, `0` off).
//!
//! A graphics or compute pipeline made with compiled shaders keeps its driver blob in the pipeline
//! library. The next run hands that blob to the PC runtime in the `CachedPSO` field of the
//! description, so the driver does not compile the pipeline again. The library opens at the first
//! direct pipeline the PC runtime made, because its folder is named after that device's adapter
//! and driver. Its counts are logged with the summary (`pso library:`).
//!
//! The startup line names the translator hash of the build (`translator_hash=`), the same one the
//! cache keys hold. `direct shaders summary:` is logged at the first pipeline, every 64 pipelines,
//! every 15 s when a counter changed, and at exit, so a run that is killed keeps recent counts.
//!
//! The root signature a translator gets is the one the PC runtime got (`dxil_override` keeps the
//! layouts of the root signatures the title creates while this path is on).

#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::direct_source::DirectSource;
use crate::precompile_queue::Queue;
use crate::compile_service::{o0_hlsl_bytes, PipelineKind, Poll, Service, Settings, MAX_LIVE_HLSL_BYTES};
use crate::lazy_pso::{Owned, COMPUTE_DESC_SIZE, GRAPHICS_DESC_SIZE, MESH_DESC_SIZE};
use crate::log::xs_log;
use crate::pso_library::Library;
use crate::shader_source::{InputElement, Linkage, Pipeline, Refusal, ShaderSource, StageInput, StageKind};

impl crate::compile_service::Compiler for decant_dxc::Dxc {
    fn version(&self) -> String {
        decant_dxc::Dxc::version(self)
    }

    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
        let _trace = decant_trace::span("direct_pso", "compile");
        let began = Instant::now();
        let res = decant_dxc::Dxc::compile(self, hlsl, entry, profile, args);
        let us = u64::try_from(began.elapsed().as_micros()).unwrap_or(u64::MAX);
        DXC_COMPILE_US.fetch_add(us, Ordering::Relaxed);
        res
    }
}

fn log_line(line: &str) {
    xs_log!("{line}");
}

/// Longest a single graphics or mesh pipeline may hold up its first bind.
const FIRST_BIND_WAIT: Duration = Duration::from_millis(50);

/// Frame wait budget when `DECANT_D3D12X_FRAME_WAIT_MS` is not set.
const DEFAULT_FRAME_WAIT_MS: u64 = 500;

/// Async bind frame budget when `DECANT_D3D12X_ASYNC_BIND_FRAME_MS` is not set.
const DEFAULT_ASYNC_FRAME_MS: u64 = 16;

/// Compute wait when `DECANT_D3D12X_COMPUTE_WAIT_MS` is not set.
const DEFAULT_COMPUTE_WAIT_MS: u64 = 5000;

/// How often the summary is logged when a counter changed.
const SUMMARY_PERIOD: Duration = Duration::from_secs(15);

/// Microseconds that cold graphics and mesh binds have waited since the last present.
static FRAME_SPENT_US: AtomicU64 = AtomicU64::new(0);

/// Microseconds that cold async compute binds have waited since the last present.
static ASYNC_SPENT_US: AtomicU64 = AtomicU64::new(0);

/// The translator hash of this build, part of every cache key.
const TRANSLATOR_HASH: &str = env!("DECANT_DIRECT_TRANSLATOR_HASH");

/// Workers for binds that bypass the queue on a host with `cores` cores: half of them, 2 to 16.
fn default_threads(cores: usize) -> usize {
    (cores / 2).clamp(2, 16)
}

/// Workers of the precompile queue on a host with `cores` cores: two thirds of them, 4 to 32.
fn default_precompile_threads(cores: usize) -> usize {
    (cores * 2 / 3).clamp(4, 32)
}

/// The core count of the host, 4 when it cannot be read.
fn host_cores() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

fn threads() -> usize {
    let configured = std::env::var("DECANT_D3D12X_DIRECT_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0);
    configured.unwrap_or_else(|| default_threads(host_cores()))
}

/// The live HLSL limit for the value of `DECANT_D3D12X_MAX_LIVE_HLSL_BYTES`: 8 MiB when unset or not
/// a number, no limit for `0`.
fn parse_live_limit(value: Option<&str>) -> usize {
    match value.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(0) => usize::MAX,
        Some(bytes) => bytes,
        None => MAX_LIVE_HLSL_BYTES,
    }
}

/// Starts the service: DXC, the cache and the workers. `None` when DXC is not found.
fn start() -> Option<Service> {
    let dxc = match decant_dxc::Dxc::load() {
        Ok(d) => d,
        Err(e) => {
            xs_log!("direct shaders: DECANT_D3D12X_DIRECT is on but no DXC loads, the path stays off ({e})");
            return None;
        }
    };
    let cache = crate::direct_cache::DiskCache::from_env();
    let dump = std::env::var_os("DECANT_D3D12X_DIRECT_DUMP").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    let source = SOURCE.get_or_init(|| Arc::new(DirectSource::for_layer(log_line, dump.clone()))).clone();
    let settings = Settings { threads: threads(), cache, dump, log: log_line };
    let live_limit = parse_live_limit(std::env::var("DECANT_D3D12X_MAX_LIVE_HLSL_BYTES").ok().as_deref());
    let o0_bytes = o0_hlsl_bytes();
    xs_log!(
        "direct shaders on: {}, translator {}, translator_hash={TRANSLATOR_HASH}, {} workers, cache {}, dump {}, -O0 from {o0_bytes} HLSL bytes, live limit {}",
        dxc.describe(),
        source.version(),
        settings.threads,
        settings.cache.as_ref().map_or("off".to_string(), |c| c.dir().display().to_string()),
        settings.dump.as_ref().map_or("off".to_string(), |d| d.display().to_string()),
        if live_limit == usize::MAX { "off".to_string() } else { format!("{live_limit} HLSL bytes") }
    );
    // The process-wide service is static, so its Drop does not run at normal exit.
    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
    }
    // SAFETY: the callback uses only the process-wide service and has the CRT ABI.
    if unsafe { atexit(log_final_summary) } != 0 {
        xs_log!("direct shaders: could not register the teardown summary");
    }
    // A run stopped by force never reaches `atexit`: log the counts on a timer as well.
    let ticker = std::thread::Builder::new().name("decant-direct-summary".into()).spawn(|| loop {
        std::thread::sleep(SUMMARY_PERIOD);
        if let Some(Some(service)) = SERVICE.get() {
            service.log_summary_if_changed();
        }
        log_library_if_changed();
    });
    if let Err(e) = ticker {
        xs_log!("direct shaders: could not start the summary timer ({e})");
    }
    LIBRARY.get_or_init(open_library);
    let service = Service::new(source, Arc::new(dxc), settings);
    service.set_live(live_limit);
    Some(service)
}

static SERVICE: OnceLock<Option<Service>> = OnceLock::new();
static SOURCE: OnceLock<Arc<DirectSource>> = OnceLock::new();
static QUEUE: OnceLock<Option<Arc<Queue>>> = OnceLock::new();
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);
static DXC_COMPILE_US: AtomicU64 = AtomicU64::new(0);
static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

struct TraceCounterState {
    pending_binds: AtomicU64,
    ready_total: AtomicU64,
    refused_total: AtomicU64,
    dxc_compiled_total: AtomicU64,
    dxc_compile_total_ms: AtomicU64,
    library_hits: AtomicU64,
    library_misses: AtomicU64,
}

impl TraceCounterState {
    const fn new() -> Self {
        Self {
            pending_binds: AtomicU64::new(u64::MAX),
            ready_total: AtomicU64::new(u64::MAX),
            refused_total: AtomicU64::new(u64::MAX),
            dxc_compiled_total: AtomicU64::new(u64::MAX),
            dxc_compile_total_ms: AtomicU64::new(u64::MAX),
            library_hits: AtomicU64::new(u64::MAX),
            library_misses: AtomicU64::new(u64::MAX),
        }
    }
}

static TRACE_COUNTERS: TraceCounterState = TraceCounterState::new();

#[inline]
fn emit_if_changed(slot: &AtomicU64, name: &'static str, value: u64) {
    if slot.load(Ordering::Relaxed) != value {
        slot.store(value, Ordering::Relaxed);
        decant_trace::record_counter(name, value as f64);
    }
}

fn emit_trace_counters_to(state: &TraceCounterState, values: [u64; 7]) {
    if !decant_trace::enabled() {
        return;
    }
    emit_if_changed(&state.pending_binds, "pso_pending_binds", values[0]);
    emit_if_changed(&state.ready_total, "pso_ready_total", values[1]);
    emit_if_changed(&state.refused_total, "pso_refused_total", values[2]);
    emit_if_changed(&state.dxc_compiled_total, "dxc_compiled_total", values[3]);
    emit_if_changed(&state.dxc_compile_total_ms, "dxc_compile_total_ms", values[4]);
    emit_if_changed(&state.library_hits, "pso_library_hits", values[5]);
    emit_if_changed(&state.library_misses, "pso_library_misses", values[6]);
}

fn emit_trace_counters() {
    if !decant_trace::enabled() {
        return;
    }
    let service = current();
    let (made, refused_pso) = crate::lazy_pso::trace_counts();
    let pending_binds = LEFT_OUT.load(Ordering::Relaxed);
    let ready_total = service.map_or(made, |s| s.counters().ready.load(Ordering::Relaxed));
    let service_refused = service.map_or(0, |s| s.counters().refused().values().sum::<u64>());
    let refused_total = refused_pso.saturating_add(service_refused);
    let dxc_compiled_total = service.map_or(0, |s| s.counters().compiled.load(Ordering::Relaxed));
    let dxc_compile_total_ms = DXC_COMPILE_US.load(Ordering::Relaxed) / 1000;
    let (library_hits, library_misses) = library().map_or((0, 0), |lib| {
        let c = lib.counts();
        (c.hits.load(Ordering::Relaxed), c.misses.load(Ordering::Relaxed))
    });
    emit_trace_counters_to(
        &TRACE_COUNTERS,
        [
            pending_binds,
            ready_total,
            refused_total,
            dxc_compiled_total,
            dxc_compile_total_ms,
            library_hits,
            library_misses,
        ],
    );
}

const WAITING_SHARDS: usize = 16;

type WaitingShard = Mutex<Option<HashMap<usize, (u128, bool)>>>;

struct WaitingTable {
    count: AtomicUsize,
    shards: [WaitingShard; WAITING_SHARDS],
}

impl WaitingTable {
    const fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            shards: [const { Mutex::new(None) }; WAITING_SHARDS],
        }
    }

    #[inline]
    fn shard(&self, addr: usize) -> &WaitingShard {
        &self.shards[(addr >> 4) & (WAITING_SHARDS - 1)]
    }

    fn insert(&self, addr: usize, entry: (u128, bool)) {
        let mut guard = self.shard(addr).lock().unwrap_or_else(|e| e.into_inner());
        let prev = guard.get_or_insert_with(HashMap::new).insert(addr, entry);
        if prev.is_none() {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn take(&self, addr: usize) -> Option<(u128, bool)> {
        if self.count.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let mut guard = self.shard(addr).lock().unwrap_or_else(|e| e.into_inner());
        let removed = guard.as_mut()?.remove(&addr);
        if removed.is_some() {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
        removed
    }
}

/// The pipeline library key of each pipeline handed out with compiled shaders and not made yet,
/// by the address of its console description copy, and whether it got a stored blob.
static WAITING: WaitingTable = WaitingTable::new();

/// The pipeline library of the run, once the direct service started with it on.
fn library() -> Option<&'static Library> {
    LIBRARY.get()?.as_ref()
}

/// The library of `DECANT_D3D12X_PSO_CACHE`, `None` when it is off.
fn open_library() -> Option<Library> {
    let Some(root) = crate::pso_library::configured_root() else {
        xs_log!("direct shaders: the pipeline library is off");
        return None;
    };
    xs_log!("direct shaders: pipeline library under {}", root.display());
    Some(Library::new(root, TRANSLATOR_HASH))
}

pub(crate) fn register(owned: &Owned, sequence: u32) -> Option<u128> {
    let count = precompile_threads();
    if count == 0 || !crate::compile_service::direct_enabled() {
        return None;
    }
    let queue = QUEUE.get_or_init(|| {
        let cache = crate::direct_cache::DiskCache::from_env()?;
        let service = current()?;
        let source = SOURCE.get()?;
        Queue::start(cache, count, move |job, fast| crate::precompile_queue::run_job(service, source, job, fast))
    }).as_ref()?;
    queue.register(current()?, pipeline_of(owned).ok()?, sequence)
}

fn precompile_threads() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        if std::env::var_os("DECANT_D3D12X_PRECOMPILE").is_some_and(|value| value == "0") {
            return 0;
        }
        std::env::var("DECANT_D3D12X_PRECOMPILE_THREADS").ok()
            .and_then(|value| value.parse::<usize>().ok()).unwrap_or_else(|| default_precompile_threads(host_cores())).min(64)
    })
}

pub(crate) fn note_bind() {
    if let Some(Some(queue)) = QUEUE.get() {
        queue.note_bind();
    }
}

extern "C" fn log_final_summary() {
    if let Some(Some(queue)) = QUEUE.get() {
        queue.save(true);
    }
    xs_log!("direct shaders: {} pending binds left work out", LEFT_OUT.load(Ordering::Relaxed));
    if let Some(Some(service)) = SERVICE.get() {
        service.log_summary();
    }
    if let Some(library) = library() {
        library.flush();
        xs_log!("direct shaders: {}", library.summary());
    }
    emit_trace_counters();
}

/// Logs the counts of the pipeline library when they changed since the last time, and saves the
/// recorded pipeline manifest periodically so interrupted runs keep their progress.
fn log_library_if_changed() {
    static LAST: Mutex<String> = Mutex::new(String::new());
    if let Some(Some(queue)) = QUEUE.get() {
        queue.save(false);
    }
    let Some(library) = library() else { return };
    let line = library.summary();
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last != line {
        xs_log!("direct shaders: {line}");
        *last = line;
    }
}

pub(crate) fn note_present() {
    refill_frame_budget();
    emit_trace_counters();
}

/// Refills the wait budgets of the frame, for graphics and mesh binds and for async compute binds.
pub(crate) fn refill_frame_budget() {
    FRAME_SPENT_US.store(0, Ordering::Relaxed);
    ASYNC_SPENT_US.store(0, Ordering::Relaxed);
}

pub(crate) fn current_service() -> Option<&'static Service> {
    current()
}

pub(crate) fn pipeline_done_for(service: Option<&'static Service>, owned: &Owned, creation: Option<(bool, std::time::Duration)>) {
    let kind = match owned.desc().len() {
        GRAPHICS_DESC_SIZE => PipelineKind::Graphics,
        COMPUTE_DESC_SIZE => PipelineKind::Compute,
        MESH_DESC_SIZE => PipelineKind::Mesh,
        _ => return,
    };
    if let Some((false, _)) = creation {
        library_refused(owned);
    }
    if let Some(service) = service.or_else(current) {
        service.pipeline_done(kind, creation);
    }
    emit_trace_counters();
}

/// The library key waiting for the PC creation of `owned`, taken out.
fn take_waiting(owned: &Owned) -> Option<(u128, bool)> {
    WAITING.take(owned.desc().as_ptr() as usize)
}

/// The PC runtime refused the pipeline of `owned` with its compiled shaders. A stored blob it was
/// given is deleted, since the blob may be the reason.
fn library_refused(owned: &Owned) {
    let Some(library) = library() else { return };
    let Some((key, true)) = take_waiting(owned) else { return };
    let on = library.reject(key);
    xs_log!("direct shaders: the PC runtime refused pipeline library blob {key:032x}{}", if on { "" } else { ", the library stops for this run" });
}

/// The PC runtime made `made` from the compiled shaders of `owned`. Opens the library with its
/// adapter on the first call, and stores the driver blob of a pipeline that came without one.
fn library_made(made: usize, owned: &Owned) {
    let Some(library) = library() else { return };
    let Some((key, attached)) = take_waiting(owned) else { return };
    let pso = made as *mut std::ffi::c_void;
    let opened = library.open(|| {
        let adapter = crate::pso_library::adapter_of(pso);
        match &adapter {
            Some(a) => xs_log!("direct shaders: pipeline library for adapter {:04X}:{:04X}, driver {:#x}", a.vendor_id, a.device_id, a.driver_version),
            None => xs_log!("direct shaders: DXGI gives no adapter or driver version, the pipeline library stays closed"),
        }
        adapter
    });
    if attached || opened.is_none() {
        return;
    }
    let Some(blob) = crate::pso_library::cached_blob(pso) else { return };
    if let Err(e) = library.keep(key, &blob) {
        xs_log!("direct shaders: pipeline library could not store {key:032x} ({e})");
    }
}

#[cfg(test)]
thread_local! {
    /// A service for the tests of this thread, in place of the switch.
    static TEST_SERVICE: std::cell::Cell<Option<&'static Service>> = const { std::cell::Cell::new(None) };
}

fn current() -> Option<&'static Service> {
    #[cfg(test)]
    if let Some(s) = TEST_SERVICE.get() {
        return Some(s);
    }
    if !crate::compile_service::direct_enabled() {
        return None;
    }
    SERVICE.get_or_init(start).as_ref()
}

/// The pipeline `made` was made with the direct shaders of `owned` (the console shaders it holds): keeps what each
/// stage reads from its tables for the kind check (`kind_check`), from the console containers, so a
/// pipeline taken from the disk cache has it too.
///
/// The pipeline library keeps the driver blob of the pipeline here as well.
pub fn kind_made(made: usize, owned: &Owned) {
    let _trace = decant_trace::span("direct_pso", "create");
    library_made(made, owned);
    if !crate::kind_check::tracking() {
        return;
    }
    let desc = owned.desc();
    let graphics = desc.len() != COMPUTE_DESC_SIZE;
    let mut stages = Vec::new();
    for &(kind, at) in stage_offsets(desc).iter() {
        let (ptr, len) = (read_u64(desc, at), read_u64(desc, at + 8) as usize);
        let Some(bytes) = owned.shaders().iter().find(|s| s.as_ptr() as u64 == ptr && s.len() == len) else { continue };
        let hash = crate::dxil_override::content_hash(bytes);
        let name = format!("{hash:016x}");
        if let Ok(Ok(program)) = std::panic::catch_unwind(|| decant_shader::container::program(bytes, &name)) {
            stages.push(crate::kind_check::StageDecl { stage: kind.short().to_ascii_uppercase(), hash, source: crate::kind_check::Source::Direct, views: crate::kind_check::views_of_program(&program) });
        }
    }
    crate::kind_check::direct_made(made, graphics, stages);
}

/// What to make a pipeline with.
pub enum Choice {
    /// The description as the title created it.
    Original,
    /// Not yet: bind it as missing and ask again at the next bind.
    Pending,
    /// A copy of the description with the compiled shaders.
    Direct(Patched),
}

/// A description pointing at compiled shaders, and at a stored driver blob when the pipeline
/// library has one, which it keeps alive.
pub struct Patched {
    desc: Vec<u8>,
    dxil: Vec<Arc<[u8]>>,
    _blob: Option<Vec<u8>>,
}

impl Patched {
    /// The description.
    pub fn desc(&self) -> &[u8] {
        &self.desc
    }
}

/// How long a bind may wait for shader translation, DXC and the PC pipeline.
///
/// A compute pipeline waits up to [`compute_wait`] unless it is already `late`. A graphics or mesh
/// pipeline waits only on its first bind, for [`FIRST_BIND_WAIT`] or what is left of the frame
/// budget, whichever is less.
pub(crate) fn bind_wait(is_compute: bool, first: bool, late: &AtomicBool) -> Duration {
    match (is_compute, first) {
        (true, _) if late.load(Ordering::Acquire) => Duration::ZERO,
        (true, _) => compute_wait(),
        (false, true) => FIRST_BIND_WAIT.min(frame_budget_left()),
        (false, false) => Duration::ZERO,
    }
}

/// What is left of this frame's wait budget for graphics and mesh binds.
pub(crate) fn frame_budget_left() -> Duration {
    budget_left(&FRAME_SPENT_US, frame_wait_budget())
}

/// Takes `elapsed` from this frame's wait budget.
pub(crate) fn spend_frame_wait(elapsed: Duration) {
    spend(&FRAME_SPENT_US, elapsed);
}

/// What is left of this frame's wait budget for cold async compute binds.
pub(crate) fn async_frame_left() -> Duration {
    budget_left(&ASYNC_SPENT_US, async_frame_budget())
}

/// Takes `elapsed` from this frame's async compute wait budget.
pub(crate) fn spend_async_wait(elapsed: Duration) {
    spend(&ASYNC_SPENT_US, elapsed);
}

/// What is left of `budget` after the microseconds in `spent`.
fn budget_left(spent: &AtomicU64, budget: Duration) -> Duration {
    budget.saturating_sub(Duration::from_micros(spent.load(Ordering::Relaxed)))
}

/// Adds `elapsed` to the microseconds in `spent`.
fn spend(spent: &AtomicU64, elapsed: Duration) {
    let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    spent.fetch_add(micros, Ordering::Relaxed);
}

/// Counts a bind that went on without its pipeline, and logs only the first one of each object.
pub(crate) fn note_pending(object: usize, logged: &AtomicBool) {
    let total = LEFT_OUT.fetch_add(1, Ordering::Relaxed) + 1;
    if decant_trace::enabled() {
        emit_if_changed(&TRACE_COUNTERS.pending_binds, "pso_pending_binds", total);
    }
    if !logged.swap(true, Ordering::Relaxed) {
        xs_log!("direct shaders: pipeline {object:#x} is still pending, bind left work out ({total} pending binds)");
    }
}

/// Picks the shaders to make `owned` with. Waits up to `wait` for translation and DXC, using the
/// precompile queue when the pipeline has a `key` there.
pub(crate) fn choose_with_budget(object: usize, owned: &Owned, key: Option<u128>, wait: Duration) -> Choice {
    let Some(service) = current() else { return Choice::Original };
    let _trace = decant_trace::span("direct_pso", "choose");
    let begun = Instant::now();
    let from_queue = key.and_then(|key| QUEUE.get()?.as_ref()?.bind(service, key, wait));
    let verdict = match from_queue {
        Some(verdict) => verdict,
        None => {
            let ask = || match SOURCE.get() {
                #[cfg(test)]
                _ if TEST_SERVICE.get().is_some() => service.poll(object, || pipeline_of(owned)),
                Some(source) => service.poll_deferred(object, source, || pipeline_of(owned)),
                None => Poll::Refused(String::new()),
            };
            poll_within(service, object, wait.saturating_sub(begun.elapsed()), ask)
        }
    };
    choice_from(owned, verdict)
}

/// The choice that follows a poll answer.
fn choice_from(owned: &Owned, verdict: Poll) -> Choice {
    match verdict {
        Poll::Ready(dxil) => {
            let mut patched = patch(owned.desc(), &dxil);
            if let Some(library) = library() {
                attach_blob(library, owned, &mut patched);
            }
            Choice::Direct(patched)
        }
        Poll::Refused(_) => Choice::Original,
        Poll::Pending => Choice::Pending,
    }
}

/// Asks `poll` once and, when the answer is still pending, waits up to `wait` for the object to
/// finish and asks again.
fn poll_within(service: &Service, object: usize, wait: Duration, poll: impl Fn() -> Poll) -> Poll {
    let begun = Instant::now();
    let first = poll();
    if first != Poll::Pending {
        return first;
    }
    let left = wait.saturating_sub(begun.elapsed());
    if !left.is_zero() {
        let _wait = decant_trace::span("direct_pso", "wait");
        if service.wait(object, left) { poll() } else { first }
    } else {
        first
    }
}

/// How long a compute pipeline waits before it is marked late (`DECANT_D3D12X_COMPUTE_WAIT_MS`,
/// default 5000 ms, read once).
pub(crate) fn compute_wait() -> Duration {
    static WAIT: OnceLock<Duration> = OnceLock::new();
    *WAIT.get_or_init(|| wait_from(std::env::var("DECANT_D3D12X_COMPUTE_WAIT_MS").ok().as_deref(), DEFAULT_COMPUTE_WAIT_MS))
}

/// The wait budget of one frame for cold graphics and mesh binds (`DECANT_D3D12X_FRAME_WAIT_MS`,
/// default 500 ms, read once).
fn frame_wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| wait_from(std::env::var("DECANT_D3D12X_FRAME_WAIT_MS").ok().as_deref(), DEFAULT_FRAME_WAIT_MS))
}

/// The wait budget of one frame for cold async compute binds (`DECANT_D3D12X_ASYNC_BIND_FRAME_MS`,
/// default 16 ms, `0` does not wait, read once).
fn async_frame_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| wait_from(std::env::var("DECANT_D3D12X_ASYNC_BIND_FRAME_MS").ok().as_deref(), DEFAULT_ASYNC_FRAME_MS))
}

/// A wait given in milliseconds by a setting, `default_ms` when it is missing or not a number.
fn wait_from(setting: Option<&str>, default_ms: u64) -> Duration {
    Duration::from_millis(setting.and_then(|text| text.trim().parse().ok()).unwrap_or(default_ms))
}

#[cfg(test)]
fn choose_within(service: &Service, object: usize, owned: &Owned, wait: Duration) -> Choice {
    choice_from(owned, poll_within(service, object, wait, || service.poll(object, || pipeline_of(owned))))
}

fn read_u64(d: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(d[at..at + 8].try_into().unwrap_or_default())
}

fn read_u32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(d[at..at + 4].try_into().unwrap_or_default())
}

const MAX_PIPELINE_STAGES: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StageOffsets {
    items: [(StageKind, usize); MAX_PIPELINE_STAGES],
    len: usize,
}

impl Deref for StageOffsets {
    type Target = [(StageKind, usize)];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.items[..self.len]
    }
}

/// The stages of a copied description with the offsets of their `D3D12_SHADER_BYTECODE`.
fn stage_offsets(desc: &[u8]) -> StageOffsets {
    let all: &[(StageKind, usize)] = match desc.len() {
        COMPUTE_DESC_SIZE => &[(StageKind::Compute, 0x08)],
        GRAPHICS_DESC_SIZE => &[(StageKind::Vertex, 0x08), (StageKind::Pixel, 0x18), (StageKind::Domain, 0x28), (StageKind::Hull, 0x38), (StageKind::Geometry, 0x48)],
        // A mesh pipeline: its stages in pipeline order, the pixel inputs follow the mesh outputs.
        MESH_DESC_SIZE => &[(StageKind::Amplification, crate::mesh_pso::AS_AT), (StageKind::Mesh, crate::mesh_pso::MS_AT), (StageKind::Pixel, crate::mesh_pso::PS_AT)],
        _ => &[],
    };
    let mut items = [(StageKind::Vertex, 0); MAX_PIPELINE_STAGES];
    let mut len = 0;
    for &(kind, at) in all {
        if read_u64(desc, at) != 0 && read_u64(desc, at + 8) != 0 {
            items[len] = (kind, at);
            len += 1;
        }
    }
    StageOffsets { items, len }
}

/// The translator's input for a copied description (also what `pipeline_manifest` records).
pub(crate) fn pipeline_of(owned: &Owned) -> Result<Pipeline, Refusal> {
    let _trace = decant_trace::span("direct_pso", "translate");
    let desc = owned.desc();
    let offsets = stage_offsets(desc);
    if offsets.is_empty() {
        return Err(Refusal::new("no shader stage", ""));
    }
    let mut stages = Vec::with_capacity(offsets.len());
    for &(kind, at) in &*offsets {
        let (ptr, len) = (read_u64(desc, at), read_u64(desc, at + 8) as usize);
        // The copy shares its shaders with `owned`, so the bytes are taken from there, not copied.
        let bytes = owned.shaders().iter().find(|s| s.as_ptr() as u64 == ptr && s.len() == len).cloned();
        let Some(bytes) = bytes else { return Err(Refusal::new("shader not owned", kind.short())) };
        stages.push(StageInput { kind, bytes });
    }
    let root_sig = read_u64(desc, 0) as usize;
    let Some(root) = crate::dxil_override::root_layout(root_sig) else {
        return Err(Refusal::new("root signature unknown", format!("{root_sig:#x}")));
    };
    let linkage = linkage_of(desc, offsets.iter().map(|(k, _)| *k).collect());
    Ok(Pipeline { stages, root, linkage })
}

/// The linkage of a copied description.
fn linkage_of(desc: &[u8], stages: Vec<StageKind>) -> Linkage {
    // Graphics and mesh descriptions share the graphics layout (a mesh one has no input layout).
    if desc.len() < GRAPHICS_DESC_SIZE {
        return Linkage { stages, ..Linkage::default() };
    }
    let layout = read_u64(desc, 0x228) as usize;
    let count = if layout == 0 { 0 } else { read_u32(desc, 0x230).min(32) as usize };
    let inputs = (0..count)
        .map(|i| {
            // SAFETY: the copy's input layout points to `count` elements of 32 bytes in a buffer the
            // copy keeps, each with its semantic name copied too (`lazy_pso::copy_graphics`).
            let e = unsafe { std::slice::from_raw_parts((layout + i * 32) as *const u8, 32) };
            let name = read_u64(e, 0);
            let semantic = if name == 0 {
                String::new()
            } else {
                // SAFETY: as above, a NUL terminated copy.
                unsafe { std::ffi::CStr::from_ptr(name as *const std::ffi::c_char) }.to_string_lossy().into_owned()
            };
            InputElement { semantic, index: read_u32(e, 8), format: read_u32(e, 12), slot: read_u32(e, 16), offset: read_u32(e, 20), per_instance: read_u32(e, 24) == 1, step_rate: read_u32(e, 28) }
        })
        .collect();
    Linkage {
        stages,
        inputs,
        targets: (0..read_u32(desc, 0x240).min(8) as usize).map(|i| read_u32(desc, 0x244 + 4 * i)).collect(),
        depth_format: read_u32(desc, 0x264),
        topology_type: read_u32(desc, 0x23C),
        sample_count: read_u32(desc, 0x268),
        stream_output: read_u32(desc, 0x60) != 0,
    }
}

/// A copy of `desc` whose shaders are `dxil`, one per stage in the order of [`stage_offsets`].
fn patch(desc: &[u8], dxil: &[Arc<[u8]>]) -> Patched {
    let mut copy = desc.to_vec();
    for (&(_, at), d) in stage_offsets(desc).iter().zip(dxil) {
        copy[at..at + 8].copy_from_slice(&(d.as_ptr() as u64).to_le_bytes());
        copy[at + 8..at + 16].copy_from_slice(&(d.len() as u64).to_le_bytes());
    }
    Patched { desc: copy, dxil: dxil.to_vec(), _blob: None }
}

/// Points `patched` at the stored driver blob of its pipeline, when the library has one, and
/// notes its key for [`library_made`] and [`library_refused`].
fn attach_blob(library: &Library, owned: &Owned, patched: &mut Patched) {
    let Some(at) = crate::pso_library::cached_field(patched.desc.len()) else { return };
    let Some(key) = library_key(patched) else { return };
    let blob = library.lookup(key);
    if let Some(blob) = &blob {
        patched.desc[at..at + 8].copy_from_slice(&(blob.as_ptr() as u64).to_le_bytes());
        patched.desc[at + 8..at + 16].copy_from_slice(&(blob.len() as u64).to_le_bytes());
    }
    WAITING.insert(owned.desc().as_ptr() as usize, (key, blob.is_some()));
    patched._blob = blob;
}

/// The library key of a patched description: its state and its compiled stages.
fn library_key(patched: &Patched) -> Option<u128> {
    let mut stack = [&[][..]; MAX_PIPELINE_STAGES];
    if patched.dxil.len() <= MAX_PIPELINE_STAGES {
        for (slot, d) in stack.iter_mut().zip(&patched.dxil) {
            *slot = &d[..];
        }
        return crate::pso_library::pipeline_key(&patched.desc, &stack[..patched.dxil.len()], 0);
    }
    let stages: Vec<&[u8]> = patched.dxil.iter().map(|d| &d[..]).collect();
    crate::pso_library::pipeline_key(&patched.desc, &stages, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_service::tests::FakeCompiler;
    use crate::rts0::RootLayout;
    use crate::shader_source::FixedSource;
    use std::ffi::c_void;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;
    use std::time::Duration;
    use windows_sys::core::{GUID, HRESULT};

    // A root signature object for `lazy_pso::defer`, which holds a reference to it.
    unsafe extern "system" fn no_op(_: *mut c_void) -> u32 {
        1
    }
    static ROOT_VTBL: [unsafe extern "system" fn(*mut c_void) -> u32; 3] = [no_op, no_op, no_op];

    fn fake_root(layout: RootLayout) -> usize {
        let object = Box::into_raw(Box::new(&ROOT_VTBL)) as usize;
        crate::dxil_override::remember_root_layout(object, layout);
        object
    }

    /// The shader bytes each creation saw, by device, and whether it was accepted.
    static SEEN: Mutex<Vec<(usize, Vec<u8>)>> = Mutex::new(Vec::new());

    fn seen(device: usize) -> Vec<Vec<u8>> {
        SEEN.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|(d, _)| *d == device).map(|(_, b)| b.clone()).collect()
    }

    /// The PC creation for the tests: notes the compute shader and answers the device address as
    /// the pipeline, except for a compiled shader on device 0xBAD0, which it refuses.
    unsafe fn create(device: usize, desc: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        // SAFETY: the tests pass compute descriptions, their shader pointers live.
        let d = unsafe { std::slice::from_raw_parts(desc.cast::<u8>(), COMPUTE_DESC_SIZE) };
        // SAFETY: as above.
        let cs = unsafe { std::slice::from_raw_parts(read_u64(d, 8) as *const u8, read_u64(d, 16) as usize) }.to_vec();
        let refuse = device == 0xBAD0 && cs.starts_with(b"DXBCcs");
        SEEN.lock().unwrap_or_else(|e| e.into_inner()).push((device, cs));
        if refuse {
            return 0x8007_0057_u32 as i32;
        }
        // SAFETY: `out` is the recipe's.
        unsafe { *out = device as *mut c_void };
        0
    }

    fn with_test_service(source: FixedSource) -> &'static Service {
        let service: &'static Service = Box::leak(Box::new(Service::new(Arc::new(source), Arc::new(FakeCompiler::default()), Settings::quiet(1))));
        TEST_SERVICE.set(Some(service));
        service
    }

    /// A pipeline made on first bind with compute shader `cs` and a root signature of known layout.
    fn lazy_compute(device: usize, cs: &[u8]) -> *mut c_void {
        let root = fake_root(RootLayout { version: 2, flags: 0x400, params: Vec::new(), static_samplers: Vec::new() });
        let mut desc = [0u8; COMPUTE_DESC_SIZE];
        desc[0..8].copy_from_slice(&(root as u64).to_le_bytes());
        desc[8..16].copy_from_slice(&(cs.as_ptr() as u64).to_le_bytes());
        desc[16..24].copy_from_slice(&(cs.len() as u64).to_le_bytes());
        // SAFETY: the description, its shader and its root signature are live.
        let owned = unsafe { crate::lazy_pso::copy_compute(desc.as_ptr().cast()) };
        let mut out = std::ptr::null_mut();
        // SAFETY: `out` is writable, the root signature is an object with AddRef.
        assert_eq!(unsafe { crate::lazy_pso::defer(device, owned, std::ptr::null(), &mut out, create) }, 0);
        out
    }

    #[test]
    fn compiled_mesh_and_pixel_bytes_are_written_into_the_pc_stream() {
        use crate::mesh_pso::{MS_AT, PS_AT};

        let root = fake_root(RootLayout { version: 2, flags: 0, params: Vec::new(), static_samplers: Vec::new() });
        let mut desc = [0; MESH_DESC_SIZE];
        desc[..8].copy_from_slice(&(root as u64).to_le_bytes());
        for (at, bytes) in [(MS_AT, b"console mesh".as_slice()), (PS_AT, b"console pixel".as_slice())] {
            desc[at..at + 8].copy_from_slice(&(bytes.as_ptr() as u64).to_le_bytes());
            desc[at + 8..at + 16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        }
        // SAFETY: the description's only pointers refer to live synthetic shaders and a fake root.
        let owned = unsafe { crate::lazy_pso::copy_mesh(&desc) };
        let mut source = FixedSource::simple();
        source.stages.push((StageKind::Mesh, crate::shader_source::StageHlsl { hlsl: "mesh".into(), entry: "main".into(), profile: "ms_6_6".into(), args: Vec::new() }));
        let service = with_test_service(source);
        let pipeline = pipeline_of(&owned).unwrap();
        assert_eq!(pipeline.linkage.stages, [StageKind::Mesh, StageKind::Pixel]);
        let Choice::Direct(patched) = choose_within(service, 0xD8, &owned, Duration::from_secs(10)) else {
            panic!("mesh pipeline did not compile");
        };
        let stream = crate::mesh_pso::pc_stream(patched.desc().try_into().unwrap());
        assert_eq!(read_u32(&stream, 16), 25);
        assert_eq!(read_u32(&stream, 40), 2);
        for (at, expected) in [(24, b"DXBCms_6_6"), (48, b"DXBCps_6_0")] {
            let ptr = read_u64(&stream, at) as *const u8;
            let len = read_u64(&stream, at + 8) as usize;
            // SAFETY: patched owns both compiled buffers until after the comparison.
            assert_eq!(unsafe { std::slice::from_raw_parts(ptr, len) }, expected);
        }
    }

    #[test]
    fn a_later_bind_picks_up_the_real_pipeline() {
        let service = with_test_service(FixedSource::simple());
        let object = lazy_compute(0xD1, b"console compute bytes one");
        // SAFETY: `object` is a pipeline of `lazy_pso`.
        unsafe {
            if crate::lazy_pso::resolve(object) == Some(std::ptr::null_mut()) {
                assert!(service.wait(object as usize, Duration::from_secs(1)));
            }
            assert_eq!(crate::lazy_pso::resolve(object), Some(0xD1 as *mut c_void));
        }
        assert_eq!(seen(0xD1), [b"DXBCcs_6_0".to_vec()], "made once, from the compiled shader");
        assert_eq!(service.counters().ready.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_compute_wait_is_5_s_unless_set() {
        let compute = |setting| wait_from(setting, DEFAULT_COMPUTE_WAIT_MS);
        let frame = |setting| wait_from(setting, DEFAULT_FRAME_WAIT_MS);
        assert_eq!(compute(None), Duration::from_secs(5));
        assert_eq!(compute(Some("0")), Duration::ZERO);
        assert_eq!(compute(Some(" 250 ")), Duration::from_millis(250));
        assert_eq!(compute(Some("soon")), Duration::from_secs(5));
        assert_eq!(frame(None), Duration::from_millis(500));
        assert_eq!(frame(Some("0")), Duration::ZERO);
        assert_eq!(frame(Some(" 125 ")), Duration::from_millis(125));
        assert_eq!(frame(Some("soon")), Duration::from_millis(500));
    }

    #[test]
    fn compute_waits_on_first_bind_and_stops_waiting_once_marked_late() {
        note_present();
        let late = AtomicBool::new(false);
        assert_eq!(bind_wait(true, true, &late), compute_wait(), "compute waits on first bind too");
        assert_eq!(bind_wait(true, false, &late), compute_wait());
        late.store(true, Ordering::Release);
        assert_eq!(bind_wait(true, false, &late), Duration::ZERO, "late compute bind does not stall again");

        assert_eq!(bind_wait(false, true, &late), FIRST_BIND_WAIT.min(frame_wait_budget()));
        assert_eq!(bind_wait(false, false, &late), Duration::ZERO);
        spend_frame_wait(frame_wait_budget());
        assert_eq!(bind_wait(false, true, &late), Duration::ZERO, "exhausted frame budget caps graphics waits");
        note_present();
        assert_eq!(bind_wait(false, true, &late), FIRST_BIND_WAIT.min(frame_wait_budget()), "present resets budget");
    }

    #[test]
    fn worker_defaults_scale_with_the_cores() {
        assert_eq!(default_precompile_threads(1), 4, "never below 4");
        assert_eq!(default_precompile_threads(6), 4);
        assert_eq!(default_precompile_threads(12), 8);
        assert_eq!(default_precompile_threads(24), 16);
        assert_eq!(default_precompile_threads(1000), 32, "never above 32");
        assert_eq!(default_threads(1), 2, "never below 2");
        assert_eq!(default_threads(8), 4);
        assert_eq!(default_threads(24), 12);
        assert_eq!(default_threads(1000), 16, "never above 16");
    }

    #[test]
    fn the_async_frame_budget_is_16_ms_unless_set() {
        let budget = |setting| wait_from(setting, DEFAULT_ASYNC_FRAME_MS);
        assert_eq!(budget(None), Duration::from_millis(16));
        assert_eq!(budget(Some("0")), Duration::ZERO, "0 does not wait at all");
        assert_eq!(budget(Some(" 40 ")), Duration::from_millis(40));
        assert_eq!(budget(Some("soon")), Duration::from_millis(16));
    }

    #[test]
    fn a_spent_budget_stays_empty_until_it_is_refilled() {
        let spent = AtomicU64::new(0);
        let budget = Duration::from_millis(16);
        assert_eq!(budget_left(&spent, budget), budget);
        spend(&spent, Duration::from_millis(5));
        assert_eq!(budget_left(&spent, budget), Duration::from_millis(11));
        spend(&spent, Duration::from_millis(20));
        assert_eq!(budget_left(&spent, budget), Duration::ZERO, "overspending leaves nothing, not a wrap");
        assert_eq!(budget_left(&spent, Duration::ZERO), Duration::ZERO, "a zero budget never waits");
    }

    #[test]
    fn a_present_refills_the_async_budget() {
        // The statics are shared with the other tests, so this test only uses the async one.
        spend_async_wait(async_frame_budget());
        assert_eq!(async_frame_left(), Duration::ZERO);
        refill_frame_budget();
        assert_eq!(async_frame_left(), async_frame_budget());
    }

    #[test]
    fn the_live_hlsl_limit_is_8_mib_unless_set() {
        assert_eq!(parse_live_limit(None), 16 * 1024 * 1024);
        assert_eq!(parse_live_limit(Some("soon")), 16 * 1024 * 1024);
        assert_eq!(parse_live_limit(Some(" 1048576 ")), 1 << 20);
        assert_eq!(parse_live_limit(Some("0")), usize::MAX, "0 lifts the limit");
        assert_eq!(TRANSLATOR_HASH.len(), 32, "the startup line names the full translator hash");
    }

    #[test]
    fn a_compile_slower_than_the_wait_leaves_the_pipeline_missing_until_it_is_done() {
        let service = with_test_service(FixedSource::simple());
        let cs = b"console compute bytes slow";
        let mut desc = [0u8; COMPUTE_DESC_SIZE];
        let root = fake_root(RootLayout { version: 2, flags: 0x400, params: Vec::new(), static_samplers: Vec::new() });
        desc[0..8].copy_from_slice(&(root as u64).to_le_bytes());
        desc[8..16].copy_from_slice(&(cs.as_ptr() as u64).to_le_bytes());
        desc[16..24].copy_from_slice(&(cs.len() as u64).to_le_bytes());
        // SAFETY: the description, its shader and its root signature are live.
        let owned = unsafe { crate::lazy_pso::copy_compute(desc.as_ptr().cast()) };
        // The fake compiler takes 20 ms: no wait at all sees it pending, the long wait sees it ready.
        assert!(matches!(choose_within(service, 0xD9, &owned, Duration::ZERO), Choice::Pending));
        assert!(matches!(choose_within(service, 0xD9, &owned, Duration::from_secs(10)), Choice::Direct(_)));
    }

    #[test]
    fn a_refused_translation_takes_the_title_shaders() {
        with_test_service(FixedSource::simple());
        let object = lazy_compute(0xD2, b"REFUSE this compute");
        // SAFETY: `object` is a pipeline of `lazy_pso`.
        unsafe {
            assert_eq!(crate::lazy_pso::resolve(object), Some(0xD2 as *mut c_void));
        }
        assert_eq!(seen(0xD2), [b"REFUSE this compute".to_vec()]);
    }

    #[test]
    fn a_compiled_pipeline_the_runtime_refuses_falls_back_to_the_title_shaders() {
        with_test_service(FixedSource::simple());
        let object = lazy_compute(0xBAD0, b"console compute bytes two");
        // SAFETY: `object` is a pipeline of `lazy_pso`.
        unsafe {
            if crate::lazy_pso::resolve(object) == Some(std::ptr::null_mut()) {
                assert!(TEST_SERVICE.get().unwrap().wait(object as usize, Duration::from_secs(1)));
            }
            assert_eq!(crate::lazy_pso::resolve(object), Some(0xBAD0 as *mut c_void));
        }
        assert_eq!(seen(0xBAD0), [b"DXBCcs_6_0".to_vec(), b"console compute bytes two".to_vec()]);
    }

    #[test]
    fn a_pipeline_whose_root_signature_is_unknown_is_made_at_once() {
        let service = with_test_service(FixedSource::simple());
        let cs = b"console compute bytes three";
        let mut desc = [0u8; COMPUTE_DESC_SIZE];
        desc[8..16].copy_from_slice(&(cs.as_ptr() as u64).to_le_bytes());
        desc[16..24].copy_from_slice(&(cs.len() as u64).to_le_bytes());
        // SAFETY: the description and its shader are live, there is no root signature.
        let owned = unsafe { crate::lazy_pso::copy_compute(desc.as_ptr().cast()) };
        let mut out = std::ptr::null_mut();
        // SAFETY: `out` is writable.
        unsafe {
            crate::lazy_pso::defer(0xD3, owned, std::ptr::null(), &mut out, create);
            assert_eq!(crate::lazy_pso::resolve(out), Some(0xD3 as *mut c_void));
        }
        assert_eq!(service.counters().refused().get("root signature unknown"), Some(&1));
    }

    #[test]
    fn the_linkage_of_a_graphics_description_is_read_from_its_copy() {
        let vs = b"vertex bytes".to_vec();
        let ps = b"pixel bytes".to_vec();
        let name = c"TEXCOORD";
        let mut element = [0u8; 32];
        element[0..8].copy_from_slice(&(name.as_ptr() as u64).to_le_bytes());
        element[8..12].copy_from_slice(&2u32.to_le_bytes());
        element[12..16].copy_from_slice(&16u32.to_le_bytes());
        element[24..28].copy_from_slice(&1u32.to_le_bytes());
        let mut g = [0u8; GRAPHICS_DESC_SIZE];
        for (at, s) in [(0x08, &vs), (0x18, &ps)] {
            g[at..at + 8].copy_from_slice(&(s.as_ptr() as u64).to_le_bytes());
            g[at + 8..at + 16].copy_from_slice(&(s.len() as u64).to_le_bytes());
        }
        g[0x228..0x230].copy_from_slice(&(element.as_ptr() as u64).to_le_bytes());
        g[0x230..0x234].copy_from_slice(&1u32.to_le_bytes());
        g[0x23C..0x240].copy_from_slice(&3u32.to_le_bytes());
        g[0x240..0x244].copy_from_slice(&2u32.to_le_bytes());
        g[0x244..0x248].copy_from_slice(&28u32.to_le_bytes());
        g[0x248..0x24C].copy_from_slice(&10u32.to_le_bytes());
        g[0x264..0x268].copy_from_slice(&40u32.to_le_bytes());
        g[0x268..0x26C].copy_from_slice(&1u32.to_le_bytes());
        // SAFETY: every pointer in `g` is live.
        let owned = unsafe { crate::lazy_pso::copy_graphics(&g) };
        let d = owned.desc();
        let p = pipeline_of(&owned);
        // The copy has no root signature: refused for that.
        assert_eq!(p.map(|_| ()), Err(Refusal::new("root signature unknown", "0x0")));
        let linkage = linkage_of(owned.desc(), vec![StageKind::Vertex, StageKind::Pixel]);
        assert_eq!(
            linkage,
            Linkage {
                stages: vec![StageKind::Vertex, StageKind::Pixel],
                inputs: vec![InputElement { semantic: "TEXCOORD".into(), index: 2, format: 16, slot: 0, offset: 0, per_instance: true, step_rate: 0 }],
                targets: vec![28, 10],
                depth_format: 40,
                topology_type: 3,
                sample_count: 1,
                stream_output: false,
            }
        );
        assert_eq!(&*stage_offsets(owned.desc()), [(StageKind::Vertex, 0x08), (StageKind::Pixel, 0x18)]);
        // The patched copy points at the compiled shaders, the rest is the same.
        let dxil: Vec<Arc<[u8]>> = vec![Arc::from(&b"DXBCvs"[..]), Arc::from(&b"DXBCps"[..])];
        let patched = patch(d, &dxil);
        assert_eq!(read_u64(patched.desc(), 0x18), dxil[1].as_ptr() as u64);
        assert_eq!(read_u64(patched.desc(), 0x20), 6);
        assert_eq!(patched.desc()[0x50..], d[0x50..]);
    }

    #[test]
    fn trace_emission_records_hot_path_spans_and_flushes_json() {
        let dir = std::env::temp_dir().join(format!("decant-d3d12x-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hotpaths.json");
        decant_trace::start(&path);

        let mut c = [0u8; COMPUTE_DESC_SIZE];
        let cs = b"cs_trace";
        c[0x08..0x10].copy_from_slice(&(cs.as_ptr() as u64).to_le_bytes());
        c[0x10..0x18].copy_from_slice(&(cs.len() as u64).to_le_bytes());
        // SAFETY: `c` points to live `cs`.
        let owned = unsafe { crate::lazy_pso::copy_compute(c.as_ptr().cast::<c_void>()) };
        let _ = pipeline_of(&owned);
        kind_made(0, &owned);
        decant_trace::frame_mark();
        decant_trace::stop();
        decant_trace::flush().unwrap();

        let json = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let summary = decant_trace::summarize(&json).unwrap();
        assert_eq!(summary.frames, 1);
        assert!(summary.spans.iter().any(|s| s.cat == "direct_pso" && s.name == "translate"));
        assert!(summary.spans.iter().any(|s| s.cat == "direct_pso" && s.name == "create"));
    }

    #[test]
    fn waiting_table_shards_and_tracks_count_across_concurrent_inserts_and_takes() {
        let table = WaitingTable::new();
        assert_eq!(table.take(0x1000), None);
        for i in 0..32usize {
            let addr = 0x1000 + (i << 4);
            table.insert(addr, (i as u128 + 1, i % 2 == 0));
        }
        assert_eq!(table.count.load(Ordering::Relaxed), 32);
        for i in 0..32usize {
            let addr = 0x1000 + (i << 4);
            assert_eq!(table.take(addr), Some((i as u128 + 1, i % 2 == 0)));
            assert_eq!(table.take(addr), None);
        }
        assert_eq!(table.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stage_offsets_collects_all_five_graphics_stages_without_allocation() {
        let mut g = [0u8; GRAPHICS_DESC_SIZE];
        for at in [0x08usize, 0x18, 0x28, 0x38, 0x48] {
            g[at..at + 8].copy_from_slice(&0x1000u64.to_le_bytes());
            g[at + 8..at + 16].copy_from_slice(&4u64.to_le_bytes());
        }
        assert_eq!(
            &*stage_offsets(&g),
            [
                (StageKind::Vertex, 0x08),
                (StageKind::Pixel, 0x18),
                (StageKind::Domain, 0x28),
                (StageKind::Hull, 0x38),
                (StageKind::Geometry, 0x48),
            ]
        );
    }

    #[test]
    fn trace_counters_emit_only_on_change_when_tracing_is_enabled() {
        let dir = std::env::temp_dir().join(format!("decant-d3d12x-pso-counters-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("counters.json");
        decant_trace::start(&path);

        let state = TraceCounterState::new();
        emit_trace_counters_to(&state, [3, 12, 1, 7, 450, 10, 2]);
        // Repeating the exact same values must not emit duplicate events.
        emit_trace_counters_to(&state, [3, 12, 1, 7, 450, 10, 2]);
        // Changing one counter emits only that counter.
        emit_trace_counters_to(&state, [4, 12, 1, 7, 450, 10, 2]);

        decant_trace::stop();
        decant_trace::flush().unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        for name in [
            "pso_pending_binds",
            "pso_ready_total",
            "pso_refused_total",
            "dxc_compiled_total",
            "dxc_compile_total_ms",
            "pso_library_hits",
            "pso_library_misses",
        ] {
            assert!(json.contains(name), "missing counter {name} in {json}");
        }
        assert_eq!(json.matches("\"pso_ready_total\"").count(), 1);
        assert_eq!(json.matches("\"pso_pending_binds\"").count(), 2);
    }

    #[test]
    fn library_key_does_not_require_pipeline_of_or_root_layout() {
        let mut desc = [0u8; COMPUTE_DESC_SIZE];
        desc[8..16].copy_from_slice(&0x1000u64.to_le_bytes());
        desc[16..24].copy_from_slice(&4u64.to_le_bytes());
        let dxil = vec![Arc::<[u8]>::from(&b"DXBCcs"[..])];
        let patched = patch(&desc, &dxil);
        let expected = crate::pso_library::pipeline_key(patched.desc(), &[&b"DXBCcs"[..]], 999);
        assert_eq!(library_key(&patched), expected);
    }
}
