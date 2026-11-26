//! The compile service of the direct shader path: translates and compiles the shaders
//! of a pipeline on worker threads. The thread that binds the pipeline can wait for them
//! ([`Service::wait`]) or come back later.
//!
//! [`Service::poll`] is asked each time a pipeline that is not made yet is bound. First it checks
//! the pipeline key in memory and in the disk cache: when the pipeline record and every stage DXIL
//! file are present the pipeline is ready at once without translating or compiling. Otherwise it
//! asks the translator whether it takes the pipeline (a refused one is made from the title's
//! shaders at once) and checks the stage keys in memory and on disk before queueing one job per
//! distinct pipeline content, answering [`Poll::Pending`] until the workers finish. The caller
//! waits for them at the first bind (`direct_pso`), because a title draws some things only once.
//! A pipeline that is still pending after that wait is bound like a refused one (its work is left
//! out) and asked for again at the next bind.
//!
//! Work is shared by content: pipelines with the same stages, root signature shape and linkage
//! share one job, and a stage that several pipelines have is compiled once (the first job compiles
//! it, the others wait for that result).
//!
//! Counted per run and logged: pipelines ready from their disk record (`pipe_hits`), stages found
//! in memory or on disk for the other pipelines (`stage_hits`), stages translated, stages compiled,
//! refusals by reason, DXC errors. `dump` (from `DECANT_D3D12X_DIRECT_DUMP`) is a folder that gets the
//! HLSL and the DXC message of each stage DXC refused.
//!
//! A live service ([`Service::set_live`], the layer) compiles a stage of at least
//! [`O0_HLSL_BYTES`] of HLSL with `-O0` on a cache miss: DXC still validates and signs it with
//! `dxil.dll` (`-Vd` is never passed). Smaller stages and offline services (`decant_shader_warm`) keep
//! the default `-O3`. Both store under the same stage key, so an offline `-O3` blob is taken first.
//! A live service also refuses a stage above its HLSL limit that has no DXIL yet, with its size in
//! the log, instead of holding a worker for many minutes.
//!
//! The stages of one pipeline compile one after the other on its worker, except on a machine with
//! cores to spare: [`Service::set_stage_slots`] (by default a quarter of the cores) lets a worker
//! compile the other stages of a graphics or mesh pipeline on short lived helper threads, so the
//! vertex and the pixel stage of a cold pipeline overlap. A helper only starts when a slot is free,
//! else the worker compiles that stage itself, so the slots bound the extra threads of the whole
//! service.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use crate::direct_cache::{pipeline_key, source_stage_key, stage_key, DiskCache};
use crate::shader_source::{Pipeline, Refusal, ShaderSource, StageHlsl};

static ENABLED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether the direct shader path is on: `DECANT_D3D12X_DIRECT` is set to something other than `0`,
/// read once.
pub fn direct_enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_DIRECT").is_some_and(|v| !v.is_empty() && v != "0");
            ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Compiles HLSL to signed DXIL: DXC (`decant-dxc`) or a fake in the tests.
pub trait Compiler: Send + Sync {
    /// The compiler version, part of the cache key.
    fn version(&self) -> String;

    /// The DXIL container of `hlsl` for `entry` and `profile` with the extra `args`, or the
    /// compiler's message. The same shape as the `Compiler` trait of `decant-shader`.
    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String>;
}

/// How the service runs.
pub struct Settings {
    /// Worker threads (at least one).
    pub threads: usize,
    /// The disk cache, none for memory only.
    pub cache: Option<DiskCache>,
    /// Where the HLSL and message of each DXC error go, none for nowhere.
    pub dump: Option<PathBuf>,
    /// Writes one log line.
    pub log: fn(&str),
}

impl Settings {
    /// Settings for the tests: memory only, no log.
    pub fn quiet(threads: usize) -> Settings {
        Settings { threads, cache: None, dump: None, log: |_| {} }
    }
}

/// What became of a pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poll {
    /// Not on the direct path, for the reason given: the caller makes it the usual way.
    Refused(String),
    /// Being translated or compiled: bind it as missing and ask again later.
    Pending,
    /// The DXIL of every stage, in the order of `Pipeline::stages`.
    Ready(Vec<Arc<[u8]>>),
}

/// The state of one pipeline job.
#[derive(Debug, Default)]
struct Ticket {
    done: OnceLock<Poll>,
}

impl Ticket {
    fn poll(&self) -> Poll {
        self.done.get().cloned().unwrap_or(Poll::Pending)
    }

    fn complete(&self, result: Poll) {
        let _ = self.done.set(result);
    }
}

/// The three pipeline families accepted by the layer.
#[derive(Clone, Copy)]
pub enum PipelineKind {
    Graphics,
    Compute,
    Mesh,
}

const SUMMARY_INTERVAL: u64 = 64;

/// A live cache miss of a stage with at least this much HLSL is compiled with `-O0` by default.
pub const O0_HLSL_BYTES: usize = 8 * 1024;

/// Parses `DECANT_D3D12X_DIRECT_O0_BYTES`: `0`, `off`, or `false` disables `-O0` (`usize::MAX`),
/// a positive integer sets the byte threshold, and missing or invalid values return [`O0_HLSL_BYTES`].
pub fn parse_o0_hlsl_bytes(setting: Option<&str>) -> usize {
    let Some(text) = setting.map(str::trim).filter(|s| !s.is_empty()) else {
        return O0_HLSL_BYTES;
    };
    if matches!(text.to_ascii_lowercase().as_str(), "0" | "off" | "false") {
        return usize::MAX;
    }
    text.parse::<usize>().unwrap_or(O0_HLSL_BYTES)
}

/// Returns the live `-O0` HLSL byte threshold from `DECANT_D3D12X_DIRECT_O0_BYTES`, read once.
pub fn o0_hlsl_bytes() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| parse_o0_hlsl_bytes(std::env::var("DECANT_D3D12X_DIRECT_O0_BYTES").ok().as_deref()))
}

/// A live cache miss of a stage with at least this much HLSL is throttled by [`large_dxc_limit`].
pub const LARGE_STAGE_HLSL_BYTES: usize = 256 * 1024;
/// Default maximum concurrent live DXC compiles of stages `>= LARGE_STAGE_HLSL_BYTES`.
pub const DEFAULT_LARGE_DXC_LIMIT: usize = 2;

/// Parses `DECANT_D3D12X_LARGE_DXC_LIMIT`: `0`, `off`, or `false` disables the cap (`usize::MAX`),
/// a positive integer sets the concurrent limit, and missing or invalid values return
/// [`DEFAULT_LARGE_DXC_LIMIT`].
pub fn parse_large_dxc_limit(setting: Option<&str>) -> usize {
    let Some(text) = setting.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_LARGE_DXC_LIMIT;
    };
    if matches!(text.to_ascii_lowercase().as_str(), "0" | "off" | "false") {
        return usize::MAX;
    }
    text.parse::<usize>().unwrap_or(DEFAULT_LARGE_DXC_LIMIT)
}

/// Returns the live large-stage DXC concurrency limit from `DECANT_D3D12X_LARGE_DXC_LIMIT`, read once.
pub fn large_dxc_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| parse_large_dxc_limit(std::env::var("DECANT_D3D12X_LARGE_DXC_LIMIT").ok().as_deref()))
}

/// The default HLSL limit of a live compile (`DECANT_D3D12X_MAX_LIVE_HLSL_BYTES` sets another).
pub const MAX_LIVE_HLSL_BYTES: usize = 16 * 1024 * 1024;

/// The counters of one run.
#[derive(Default)]
pub struct Counters {
    /// Stages the translator gave HLSL for.
    pub translated: AtomicU64,
    /// Stages found in memory or on disk, for every pipeline.
    pub cache_hits: AtomicU64,
    /// Pipelines ready from their disk record, without translation.
    pub pipe_hits: AtomicU64,
    /// Stages found in memory or on disk for pipelines without a record hit.
    pub stage_hits: AtomicU64,
    /// Stages DXC compiled.
    pub compiled: AtomicU64,
    /// Of those, stages compiled with `-O0` for their size.
    pub compiled_o0: AtomicU64,
    /// Stages DXC refused.
    pub dxc_errors: AtomicU64,
    /// Pipelines that got DXIL for every stage.
    pub ready: AtomicU64,
    /// Pipelines refused, by reason.
    refused: Mutex<BTreeMap<String, u64>>,
    pipelines: [AtomicU64; 3],
    made: [AtomicU64; 3],
    completed: AtomicU64,
    translation_us: AtomicU64,
    dxc_wait_us: AtomicU64,
    dxc_compile_us: AtomicU64,
    pc_creation_us: AtomicU64,
}

impl Counters {
    fn refuse(&self, reason: &str) -> u64 {
        let mut map = self.refused.lock().unwrap_or_else(|e| e.into_inner());
        let n = map.entry(reason.to_string()).or_default();
        *n += 1;
        *n
    }

    /// Refusals by reason.
    pub fn refused(&self) -> BTreeMap<String, u64> {
        self.refused.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The counters as one log fragment.
    pub fn summary(&self) -> String {
        let refused: Vec<String> = self.refused().iter().map(|(r, n)| format!("{r}: {n}")).collect();
        format!(
            "{} pipelines ready, {} from their record, {} stages translated, {} cache hits, {} compiled, {} DXC errors, refused {{{}}}",
            self.ready.load(Ordering::Relaxed),
            self.pipe_hits.load(Ordering::Relaxed),
            self.translated.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
            self.compiled.load(Ordering::Relaxed),
            self.dxc_errors.load(Ordering::Relaxed),
            refused.join(", ")
        )
    }
}

/// A stage result shared by every job that needs the same stage.
type StageCell = Arc<OnceLock<Result<Arc<[u8]>, String>>>;

struct Shared {
    source: Arc<dyn ShaderSource>,
    compiler: Arc<dyn Compiler>,
    translator_version: String,
    compiler_version: String,
    cache: Option<DiskCache>,
    dump: Option<PathBuf>,
    log: fn(&str),
    /// Jobs by pipeline key, while they run.
    jobs: Mutex<HashMap<u128, Arc<Ticket>>>,
    /// Number of entries currently in `jobs` so callers can skip locking when empty.
    active_jobs: AtomicUsize,
    /// Pipelines by the caller's id, until the caller got a final answer.
    objects: Mutex<HashMap<usize, Arc<Ticket>>>,
    /// Number of entries currently in `objects` so callers can skip locking when empty.
    active_objects: AtomicUsize,
    /// Stage keys by pipeline key, kept in memory when there is no disk cache.
    pipelines: Mutex<HashMap<u128, Vec<u128>>>,
    /// Stage results by stage key: in flight, or kept when there is no disk cache.
    stages: Mutex<HashMap<u128, StageCell>>,
    counters: Counters,
    /// Jobs queued or running.
    busy: AtomicUsize,
    idle: (Mutex<()>, Condvar),
    /// Signalled whenever a job finished, for [`Service::wait`].
    done: (Mutex<()>, Condvar),
    /// A live service: `-O0` for large stages and the HLSL limit below.
    live: AtomicBool,
    /// The largest HLSL a live service compiles.
    live_limit: AtomicUsize,
    /// The last summary line logged.
    last_summary: Mutex<String>,
    /// Helper threads that may compile stages beside the workers, free now.
    stage_slots: AtomicUsize,
    /// Concurrency gate for giant (`>= LARGE_STAGE_HLSL_BYTES`) live DXC compiles.
    large_gate: LargeCompileGate,
}

struct LargeCompileGate {
    active: Mutex<usize>,
    ready: Condvar,
    limit: AtomicUsize,
}

impl LargeCompileGate {
    fn new(limit: usize) -> Self {
        Self {
            active: Mutex::new(0),
            ready: Condvar::new(),
            limit: AtomicUsize::new(limit),
        }
    }

    fn acquire(&self) -> LargeCompileGuard<'_> {
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        while *active >= self.limit.load(Ordering::Relaxed) {
            active = self.ready.wait(active).unwrap_or_else(|e| e.into_inner());
        }
        *active += 1;
        #[cfg(windows)]
        let restore_below_normal = set_lowest_priority();
        LargeCompileGuard {
            gate: self,
            #[cfg(windows)]
            restore_below_normal,
        }
    }
}

struct LargeCompileGuard<'a> {
    gate: &'a LargeCompileGate,
    #[cfg(windows)]
    restore_below_normal: bool,
}

impl Drop for LargeCompileGuard<'_> {
    fn drop(&mut self) {
        #[cfg(windows)]
        if self.restore_below_normal {
            set_below_normal_priority();
        }
        let mut active = self.gate.active.lock().unwrap_or_else(|e| e.into_inner());
        *active = active.saturating_sub(1);
        self.gate.ready.notify_one();
    }
}

/// The HLSL of every stage of a queued pipeline and its stage keys, made on a precompile worker.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) keys: Vec<u128>,
    pub(crate) hlsl: Vec<StageHlsl>,
    pub(crate) started: std::time::Instant,
}

#[cfg(windows)]
impl Translated {
    /// The HLSL bytes of the largest stage.
    pub(crate) fn largest(&self) -> usize {
        self.hlsl.iter().map(|stage| stage.hlsl.len()).max().unwrap_or(0)
    }
}

struct Job {
    pipeline: Pipeline,
    keys: Vec<u128>,
    key: u128,
    ticket: Arc<Ticket>,
}

enum Work {
    Compile(Box<Job>),
    #[cfg(windows)]
    Deferred(Box<dyn FnOnce() + Send>),
}

/// The service: a pool of workers and what they share.
pub struct Service {
    shared: Arc<Shared>,
    queue: Mutex<Sender<Work>>,
}

/// Whether the `n`th event of a kind is logged: the first 16, then the powers of two.
fn log_due(n: u64) -> bool {
    n < 16 || n.is_power_of_two()
}

/// Caps concurrent DXC compile workers and stage helpers so at least half the CPU cores stay
/// free for render and streaming threads.
pub fn default_worker_cap(cores: usize) -> usize {
    (cores / 2).clamp(2, 8)
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn set_below_normal_priority() {
    // SAFETY: GetCurrentThread returns the pseudo handle of the calling thread.
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
fn set_lowest_priority() -> bool {
    // SAFETY: GetCurrentThread returns the pseudo handle of the calling thread.
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) != 0
    }
}

#[cfg(not(windows))]
fn set_below_normal_priority() {}

impl Service {
    /// Starts the workers.
    pub fn new(source: Arc<dyn ShaderSource>, compiler: Arc<dyn Compiler>, settings: Settings) -> Service {
        let (tx, rx) = std::sync::mpsc::channel::<Work>();
        let worker_cap = default_worker_cap(std::thread::available_parallelism().map_or(4, |cores| cores.get()));
        let worker_count = settings.threads.clamp(1, worker_cap);
        let shared = Arc::new(Shared {
            translator_version: source.version(),
            compiler_version: compiler.version(),
            source,
            compiler,
            cache: settings.cache,
            dump: settings.dump,
            log: settings.log,
            jobs: Mutex::new(HashMap::new()),
            active_jobs: AtomicUsize::new(0),
            objects: Mutex::new(HashMap::new()),
            active_objects: AtomicUsize::new(0),
            pipelines: Mutex::new(HashMap::new()),
            stages: Mutex::new(HashMap::new()),
            counters: Counters::default(),
            busy: AtomicUsize::new(0),
            idle: (Mutex::new(()), Condvar::new()),
            done: (Mutex::new(()), Condvar::new()),
            live: AtomicBool::new(false),
            live_limit: AtomicUsize::new(usize::MAX),
            last_summary: Mutex::new(String::new()),
            stage_slots: AtomicUsize::new(worker_cap),
            large_gate: LargeCompileGate::new(large_dxc_limit()),
        });
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..worker_count {
            let (worker_shared, rx) = (shared.clone(), rx.clone());
            let spawned = std::thread::Builder::new().name(format!("decant-direct-{i}")).stack_size(32 << 20).spawn(move || {
                set_below_normal_priority();
                worker(&worker_shared, &rx);
            });
            if let Err(e) = spawned {
                (settings.log)(&format!("direct shaders: worker {i} not started ({e})"));
            }
        }
        Service { shared, queue: Mutex::new(tx) }
    }

    /// The counters of this run.
    pub fn counters(&self) -> &Counters {
        &self.shared.counters
    }

    /// Marks the service as the live one of the layer: a cache miss of a stage with at least
    /// [`O0_HLSL_BYTES`] of HLSL compiles with `-O0`, and a stage above `max_hlsl_bytes` without
    /// DXIL is refused (`usize::MAX` for no limit).
    pub fn set_live(&self, max_hlsl_bytes: usize) {
        self.shared.live_limit.store(max_hlsl_bytes, Ordering::Relaxed);
        self.shared.live.store(true, Ordering::Relaxed);
    }

    /// Sets how many helper threads may compile stages beside the workers, `0` for none. Slots in
    /// use keep running, the new number applies to the stages that start after.
    pub fn set_stage_slots(&self, slots: usize) {
        self.shared.stage_slots.store(slots, Ordering::Relaxed);
    }

    /// The state of the pipeline the caller knows as `id` (its address). `build` gives its input
    /// and is only called the first time, or a refusal when the pipeline cannot take the direct
    /// path (no known root signature). A final answer (`Ready` or `Refused`) is given once: the
    /// caller keeps it, the service forgets `id`.
    pub fn poll(&self, id: usize, build: impl FnOnce() -> Result<Pipeline, Refusal>) -> Poll {
        let s = &self.shared;
        if let Some(p) = s.take_object(id) {
            return p;
        }
        let pipeline = match build() {
            Ok(p) => p,
            Err(r) if r.is_pending() => return Poll::Pending,
            Err(r) => return s.refused(&r),
        };
        let key = pipeline_key(&pipeline, &s.translator_version, &s.compiler_version);
        if s.active_jobs.load(Ordering::Acquire) > 0
            && let Some(ticket) = s.jobs.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned()
        {
            return s.track_object(id, &ticket);
        }
        // Pipeline record and all stage files present: ready at once without translation or compile.
        if let Some(dxil) = s.cached_pipeline(key, pipeline.stages.len()) {
            return s.hit(key, dxil, true);
        }
        // The direct source translates synchronously when asked whether it takes the pipeline.
        let started = std::time::Instant::now();
        let accepted = s.source.accepts(&pipeline);
        s.counters.translation_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        match accepted {
            Ok(()) => {}
            // a translation that goes on in the background: asked again at the next bind
            Err(r) if r.is_pending() => return Poll::Pending,
            Err(r) => return s.refused(&r),
        }
        // The translator's own keys when it has them (one per stage), else the input of each stage.
        let keys: Vec<u128> = match s.source.stage_keys(&pipeline).filter(|k| k.len() == pipeline.stages.len()) {
            Some(own) => pipeline.stages.iter().zip(own).map(|(st, k)| source_stage_key(st.kind, k, &s.translator_version, &s.compiler_version)).collect(),
            None => pipeline
                .stages
                .iter()
                .map(|st| stage_key(st.kind, &st.bytes, &pipeline.root, &pipeline.linkage, &s.translator_version, &s.compiler_version))
                .collect(),
        };
        // Every stage already compiled: store the pipeline mapping and answer at once.
        let cached: Option<Vec<Arc<[u8]>>> = keys.iter().map(|k| s.cached(*k)).collect();
        if let Some(dxil) = cached {
            s.store_pipeline(key, &keys);
            return s.hit(key, dxil, false);
        }
        let (ticket, fresh) = {
            let mut jobs = s.jobs.lock().unwrap_or_else(|e| e.into_inner());
            match jobs.get(&key) {
                Some(t) => (t.clone(), false),
                None => {
                    let t = Arc::new(Ticket::default());
                    jobs.insert(key, t.clone());
                    s.active_jobs.fetch_add(1, Ordering::AcqRel);
                    (t, true)
                }
            }
        };
        if fresh {
            s.busy.fetch_add(1, Ordering::SeqCst);
            let sent = self.queue.lock().unwrap_or_else(|e| e.into_inner()).send(Work::Compile(Box::new(Job { pipeline, keys, key, ticket: ticket.clone() })));
            if sent.is_err() {
                s.finish(key, &ticket, Poll::Refused("no compile worker".to_string()));
            }
        }
        s.track_object(id, &ticket)
    }

    #[cfg(windows)]
    pub(crate) fn input_key(&self, pipeline: &Pipeline) -> u128 {
        pipeline_key(pipeline, &self.shared.translator_version, &self.shared.compiler_version)
    }

    #[cfg(windows)]
    pub(crate) fn compile_deferred(
        &self, source: &crate::direct_source::DirectSource, pipeline: &Pipeline,
    ) -> (Vec<u128>, Poll) {
        match self.translate_deferred(source, pipeline) {
            Ok(translated) => self.compile_translated(pipeline, translated),
            Err(refused) => (Vec::new(), refused),
        }
    }

    /// Translates a queued pipeline on the calling worker. A refusal is final.
    #[cfg(windows)]
    pub(crate) fn translate_deferred(
        &self, source: &crate::direct_source::DirectSource, pipeline: &Pipeline,
    ) -> Result<Translated, Poll> {
        let s = &self.shared;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let started = std::time::Instant::now();
            let hlsl = match retry_translation(|| source.translate_on_worker(pipeline)) {
                Ok(hlsl) => hlsl,
                Err(_) => return Err(s.refused(&Refusal::new("translator panic", ""))),
            };
            s.counters.translation_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            let hlsl = match hlsl.into_iter().collect::<Result<Vec<_>, _>>() {
                Ok(hlsl) if hlsl.len() == pipeline.stages.len() => hlsl,
                Ok(_) => return Err(s.refused(&Refusal::new("translator stage count", ""))),
                Err(refusal) => return Err(s.refused(&refusal)),
            };
            let keys: Vec<_> = pipeline.stages.iter().zip(&hlsl).map(|(stage, hlsl)| {
                source_stage_key(stage.kind, crate::direct_source::hlsl_key(hlsl), &s.translator_version, &s.compiler_version)
            }).collect();
            Ok(Translated { keys, hlsl, started })
        }));
        result.unwrap_or_else(|_| Err(s.refused(&Refusal::new("translator panic", ""))))
    }

    /// Compiles the stages of a translated queued pipeline.
    #[cfg(windows)]
    pub(crate) fn compile_translated(&self, pipeline: &Pipeline, translated: Translated) -> (Vec<u128>, Poll) {
        let s = &self.shared;
        let Translated { keys, hlsl, started } = translated;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.compile_stages(self.input_key(pipeline), pipeline, &keys, &hlsl, started)
        }));
        match result {
            Ok(result) => (keys, result),
            Err(_) => (Vec::new(), s.refused(&Refusal::new("translator panic", ""))),
        }
    }

    /// Whether every stage of `translated` is below [`O0_HLSL_BYTES`] or already has DXIL in
    /// memory or on disk, or a live stage exceeds the HLSL limit (refused without running DXC):
    /// the fast lane of the precompile queue takes only those.
    #[cfg(windows)]
    pub(crate) fn small_or_cached(&self, translated: &Translated) -> bool {
        let s = &self.shared;
        if s.live.load(Ordering::Relaxed) {
            let limit = s.live_limit.load(Ordering::Relaxed);
            if translated.hlsl.iter().zip(&translated.keys).any(|(stage, key)| stage.hlsl.len() > limit && s.cached(*key).is_none()) {
                return true;
            }
        }
        let limit = o0_hlsl_bytes();
        translated.hlsl.iter().zip(&translated.keys).all(|(stage, key)| stage.hlsl.len() < limit || s.cached(*key).is_some())
    }

    /// Counts a pipeline the precompile queue found ready from its disk record.
    #[cfg(windows)]
    pub(crate) fn pipe_hit(&self, key: u128, dxil: Vec<Arc<[u8]>>) -> Poll {
        self.shared.hit(key, dxil, true)
    }

    /// Submits a first bind when background registration is disabled or a disk entry needs repair.
    #[cfg(windows)]
    pub(crate) fn poll_deferred(
        &'static self, id: usize, source: &'static crate::direct_source::DirectSource,
        build: impl FnOnce() -> Result<Pipeline, Refusal>,
    ) -> Poll {
        let s = &self.shared;
        if let Some(result) = s.take_object(id) {
            return result;
        }
        let pipeline = match build() {
            Ok(pipeline) => pipeline,
            Err(refusal) => return s.refused(&refusal),
        };
        let ticket = {
            let mut objects = s.objects.lock().unwrap_or_else(|e| e.into_inner());
            if objects.contains_key(&id) {
                return Poll::Pending;
            }
            let ticket = Arc::new(Ticket::default());
            objects.insert(id, ticket.clone());
            s.active_objects.fetch_add(1, Ordering::AcqRel);
            ticket
        };
        let sent = self.queue.lock().unwrap_or_else(|e| e.into_inner()).send(Work::Deferred(Box::new(move || {
            let (_, result) = self.compile_deferred(source, &pipeline);
            ticket.complete(result);
            let _guard = s.done.0.lock().unwrap_or_else(|e| e.into_inner());
            s.done.1.notify_all();
        })));
        if sent.is_err() {
            s.remove_object(id);
            return s.refused(&Refusal::new("no compile worker", ""));
        }
        Poll::Pending
    }

    /// Whether the service holds a job for the pipeline the caller knows as `id` (a pipeline whose
    /// translation is still running has none: there is nothing to wait for here).
    pub fn holds(&self, id: usize) -> bool {
        if self.shared.active_objects.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.shared.objects.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&id)
    }

    /// Waits until the job of the pipeline the caller knows as `id` is finished, at most `timeout`.
    /// Whether the next [`poll`](Service::poll) of `id` gives a final answer (also true for an `id`
    /// the service does not hold).
    pub fn wait(&self, id: usize, timeout: Duration) -> bool {
        let s = &self.shared;
        if s.active_objects.load(Ordering::Acquire) == 0 {
            return true;
        }
        let Some(ticket) = s.objects.lock().unwrap_or_else(|e| e.into_inner()).get(&id).cloned() else { return true };
        let started = std::time::Instant::now();
        let guard = s.done.0.lock().unwrap_or_else(|e| e.into_inner());
        // `finish` sets the state first and signals under this lock, so no signal is missed.
        let (_g, res) = s.done.1.wait_timeout_while(guard, timeout, |_| ticket.poll() == Poll::Pending).unwrap_or_else(|e| e.into_inner());
        s.counters.dxc_wait_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        !res.timed_out()
    }

    /// Adds `elapsed` spent waiting on a queued pipeline (such as `precompile_queue::Queue::bind`)
    /// to the `dxc_wait_us` summary counter.
    #[cfg(windows)]
    pub(crate) fn record_wait(&self, elapsed: Duration) {
        self.shared.counters.dxc_wait_us.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
    }

    /// Counts one final direct result, after the runtime creation attempt if there was one.
    pub fn pipeline_done(&self, kind: PipelineKind, creation: Option<(bool, Duration)>) {
        let counters = &self.shared.counters;
        counters.pipelines[kind as usize].fetch_add(1, Ordering::Relaxed);
        if let Some((made, elapsed)) = creation {
            counters.made[kind as usize].fetch_add(u64::from(made), Ordering::Relaxed);
            counters.pc_creation_us.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        }
        let completed = counters.completed.fetch_add(1, Ordering::AcqRel) + 1;
        // The first pipeline too, so a short run that is killed still has a summary line.
        if completed == 1 || completed.is_multiple_of(SUMMARY_INTERVAL) {
            self.log_summary();
        }
    }

    fn summary_line(&self) -> String {
        let counters = &self.shared.counters;
        let [vs_ps, cs, ms_ps] = counters.pipelines.each_ref().map(|n| n.load(Ordering::Relaxed));
        let [vs_ps_made, cs_made, ms_ps_made] = counters.made.each_ref().map(|n| n.load(Ordering::Relaxed));
        format!(
            "direct shaders summary: pipelines vs+ps={vs_ps} cs={cs} ms+ps={ms_ps}, PC made vs+ps={vs_ps_made} cs={cs_made} ms+ps={ms_ps_made}, ready={} pipe_hits={} stage_hits={} translated={} cache_hits={} compiled={} compiled_o0={} dxc_errors={}, translation_us={} dxc_wait_us={} dxc_compile_us={} pc_creation_us={}",
            counters.ready.load(Ordering::Relaxed),
            counters.pipe_hits.load(Ordering::Relaxed),
            counters.stage_hits.load(Ordering::Relaxed),
            counters.translated.load(Ordering::Relaxed),
            counters.cache_hits.load(Ordering::Relaxed),
            counters.compiled.load(Ordering::Relaxed),
            counters.compiled_o0.load(Ordering::Relaxed),
            counters.dxc_errors.load(Ordering::Relaxed),
            counters.translation_us.load(Ordering::Relaxed),
            counters.dxc_wait_us.load(Ordering::Relaxed),
            counters.dxc_compile_us.load(Ordering::Relaxed),
            counters.pc_creation_us.load(Ordering::Relaxed),
        )
    }

    /// Logs a cumulative snapshot for periodic reporting or teardown.
    pub fn log_summary(&self) {
        let line = self.summary_line();
        (self.shared.log)(&line);
        *self.shared.last_summary.lock().unwrap_or_else(|e| e.into_inner()) = line;
    }

    /// Logs a snapshot when a counter changed since the last one: the layer calls it on a timer,
    /// so a run that is killed keeps counts at most one period old.
    pub fn log_summary_if_changed(&self) {
        let line = self.summary_line();
        let mut last = self.shared.last_summary.lock().unwrap_or_else(|e| e.into_inner());
        if *last != line {
            (self.shared.log)(&line);
            *last = line;
        }
    }

    /// Waits until no job is queued or running, at most `timeout`. Whether it got idle.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let (lock, cv) = &self.shared.idle;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let (_g, res) = cv.wait_timeout_while(guard, timeout, |_| self.shared.busy.load(Ordering::SeqCst) != 0).unwrap_or_else(|e| e.into_inner());
        !res.timed_out()
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.log_summary();
    }
}

impl Shared {
    /// The DXIL of stage `key` from memory or disk.
    fn cached(&self, key: u128) -> Option<Arc<[u8]>> {
        let cell = self.stages.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
        if let Some(Ok(dxil)) = cell.as_ref().and_then(|c| c.get()) {
            return Some(dxil.clone());
        }
        self.cache.as_ref()?.read(key).map(Arc::from)
    }

    /// Every stage's DXIL for pipeline `key` from memory or disk, `None` when the pipeline record or
    /// any stage file is missing or damaged.
    fn cached_pipeline(&self, key: u128, stage_count: usize) -> Option<Vec<Arc<[u8]>>> {
        let keys = match &self.cache {
            Some(cache) => cache.read_pipeline(key)?,
            None => self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned()?,
        };
        if keys.len() != stage_count {
            return None;
        }
        keys.into_iter().map(|k| self.cached(k)).collect()
    }

    fn store_pipeline(&self, key: u128, keys: &[u128]) {
        if let Some(cache) = &self.cache {
            if let Err(e) = cache.write_pipeline(key, keys) {
                static SHOWN: AtomicU64 = AtomicU64::new(0);
                if log_due(SHOWN.fetch_add(1, Ordering::Relaxed)) {
                    (self.log)(&format!("direct shaders: cache write to {} failed ({e})", cache.dir().display()));
                }
            }
            let mut map = self.stages.lock().unwrap_or_else(|e| e.into_inner());
            for k in keys {
                map.remove(k);
            }
        } else {
            self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).insert(key, keys.to_vec());
        }
    }

    /// One stage found in memory or on disk while a pipeline without a record hit is made.
    fn stage_hit(&self) {
        self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
        self.counters.stage_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline ready without compiling: from its disk record (`pipe`) or from its stages.
    fn hit(&self, key: u128, dxil: Vec<Arc<[u8]>>, pipe: bool) -> Poll {
        self.counters.cache_hits.fetch_add(dxil.len() as u64, Ordering::Relaxed);
        if pipe {
            self.counters.pipe_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.stage_hits.fetch_add(dxil.len() as u64, Ordering::Relaxed);
        }
        let n = self.counters.ready.fetch_add(1, Ordering::Relaxed);
        if log_due(n) {
            (self.log)(&format!("direct shaders: pipeline {key:032x} ready from cache ({})", self.counters.summary()));
        }
        Poll::Ready(dxil)
    }

    fn refused(&self, r: &Refusal) -> Poll {
        let n = self.counters.refuse(&r.reason);
        if log_due(n - 1) {
            (self.log)(&format!("direct shaders: pipeline refused, {r} ({n} for this reason)"));
        }
        Poll::Refused(r.to_string())
    }

    fn take_object(&self, id: usize) -> Option<Poll> {
        if self.active_objects.load(Ordering::Acquire) == 0 {
            return None;
        }
        let mut objects = self.objects.lock().unwrap_or_else(|e| e.into_inner());
        let p = objects.get(&id)?.poll();
        if p != Poll::Pending && objects.remove(&id).is_some() {
            self.active_objects.fetch_sub(1, Ordering::AcqRel);
        }
        Some(p)
    }

    fn track_object(&self, id: usize, ticket: &Arc<Ticket>) -> Poll {
        let p = ticket.poll();
        if p != Poll::Pending {
            return p;
        }
        {
            let mut objects = self.objects.lock().unwrap_or_else(|e| e.into_inner());
            if objects.insert(id, ticket.clone()).is_none() {
                self.active_objects.fetch_add(1, Ordering::AcqRel);
            }
        }
        // A job of the same content may have finished before `objects` recorded `id`.
        let p = ticket.poll();
        if p != Poll::Pending {
            self.remove_object(id);
        }
        p
    }

    fn remove_object(&self, id: usize) {
        let mut objects = self.objects.lock().unwrap_or_else(|e| e.into_inner());
        if objects.remove(&id).is_some() {
            self.active_objects.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn finish(&self, key: u128, ticket: &Ticket, result: Poll) {
        ticket.complete(result);
        if self.jobs.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).is_some() {
            self.active_jobs.fetch_sub(1, Ordering::AcqRel);
        }
        {
            let _g = self.done.0.lock().unwrap_or_else(|e| e.into_inner());
            self.done.1.notify_all();
        }
        if self.busy.fetch_sub(1, Ordering::SeqCst) == 1 {
            let _g = self.idle.0.lock().unwrap_or_else(|e| e.into_inner());
            self.idle.1.notify_all();
        }
    }

    /// Translates and compiles one pipeline.
    fn run(&self, job: &Job) -> Poll {
        let started = std::time::Instant::now();
        let hlsl = self.source.translate(&job.pipeline);
        self.counters.translation_us.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        if hlsl.len() != job.pipeline.stages.len() {
            return self.refused(&Refusal::new("translator stage count", format!("{} answers for {} stages", hlsl.len(), job.pipeline.stages.len())));
        }
        let mut stages = Vec::with_capacity(hlsl.len());
        for h in hlsl {
            match h {
                Ok(h) => stages.push(h),
                Err(r) => return self.refused(&r),
            }
        }
        self.compile_stages(job.key, &job.pipeline, &job.keys, &stages, started)
    }

    fn compile_stages(&self, pipeline_key: u128, pipeline: &Pipeline, keys: &[u128], stages: &[StageHlsl], started: std::time::Instant) -> Poll {
        self.counters.translated.fetch_add(stages.len() as u64, Ordering::Relaxed);
        if self.live.load(Ordering::Relaxed) {
            let limit = self.live_limit.load(Ordering::Relaxed);
            let over = stages.iter().zip(keys).zip(&pipeline.stages).find(|((stage, key), _)| stage.hlsl.len() > limit && self.cached(**key).is_none());
            if let Some(((stage, key), input)) = over {
                return self.refused(&Refusal::new(
                    "live hlsl limit",
                    format!("{} {key:032x} has {} bytes of HLSL, over the live limit of {limit} (DECANT_D3D12X_MAX_LIVE_HLSL_BYTES), DXC skipped", input.kind.short(), stage.hlsl.len()),
                ));
            }
        }
        let mut resolved: Vec<Option<Arc<[u8]>>> = Vec::with_capacity(stages.len());
        let mut misses = Vec::new();
        for (idx, ((stage, &key), input)) in stages.iter().zip(keys).zip(&pipeline.stages).enumerate() {
            if let Some(dxil) = self.cached(key) {
                self.stage_hit();
                resolved.push(Some(dxil));
            } else {
                resolved.push(None);
                misses.push((idx, stage, key, input));
            }
        }
        let result = if misses.is_empty() {
            Ok(resolved.into_iter().flatten().collect())
        } else {
            let compiled = self.map_stages(&misses, |&(idx, stage, key, input)| {
                let cell = self.stages.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_default().clone();
                let mut fresh = false;
                let result = cell.get_or_init(|| {
                    fresh = true;
                    self.compile_stage(key, stage)
                });
                if !fresh && result.is_ok() {
                    self.stage_hit();
                }
                result
                    .as_ref()
                    .cloned()
                    .map(|dxil| (idx, dxil))
                    .map_err(|msg| {
                        let first = msg.lines().next().unwrap_or("");
                        Refusal::new("dxc error", format!("{} {key:032x}: {first}", input.kind.short()))
                    })
            });
            for item in compiled {
                match item {
                    Ok((idx, dxil)) => resolved[idx] = Some(dxil),
                    Err(refusal) => return self.refused(&refusal),
                }
            }
            Ok(resolved.into_iter().flatten().collect())
        };
        // With a disk cache the files keep the results. Leave cells still in use to their owners.
        if self.cache.is_some() && !misses.is_empty() {
            let mut map = self.stages.lock().unwrap_or_else(|e| e.into_inner());
            for &(_, _, k, _) in &misses {
                if map.get(&k).is_some_and(|cell| Arc::strong_count(cell) == 1) {
                    map.remove(&k);
                }
            }
        }
        let out = match result {
            Ok(out) => out,
            Err(refusal) => return self.refused(&refusal),
        };
        self.store_pipeline(pipeline_key, keys);
        let n = self.counters.ready.fetch_add(1, Ordering::Relaxed);
        if log_due(n) {
            (self.log)(&format!("direct shaders: pipeline {pipeline_key:032x} ready in {} ms ({})", started.elapsed().as_millis(), self.counters.summary()));
        }
        Poll::Ready(out)
    }

    /// Takes one helper thread slot, `false` when none is free.
    fn take_stage_slot(&self) -> bool {
        self.stage_slots.try_update(Ordering::AcqRel, Ordering::Acquire, |free| free.checked_sub(1)).is_ok()
    }

    /// Runs `work` for every item and gives the answers in the order of the items. The first item
    /// and every item that finds no free slot run on the calling worker, the others on helper
    /// threads that give their slot back when they end.
    fn map_stages<T: Sync, R: Send>(&self, items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
        if items.is_empty() {
            return Vec::new();
        }
        if items.len() == 1 {
            return vec![work(&items[0])];
        }
        let mut answers: Vec<Option<R>> = Vec::new();
        answers.resize_with(items.len(), || None);
        std::thread::scope(|scope| {
            let mut helpers = Vec::new();
            let mut handled_mask = 0u64;
            let mut handled_extra = Vec::new();
            for (at, item) in items.iter().enumerate().skip(1) {
                if !self.take_stage_slot() {
                    continue;
                }
                let work = &work;
                let spawned = std::thread::Builder::new().name("decant-direct-stage".into()).stack_size(32 << 20).spawn_scoped(scope, move || {
                    set_below_normal_priority();
                    let answer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(item)));
                    self.stage_slots.fetch_add(1, Ordering::AcqRel);
                    answer
                });
                match spawned {
                    Ok(handle) => {
                        if at < 64 {
                            handled_mask |= 1u64 << at;
                        } else {
                            handled_extra.push(at);
                        }
                        helpers.push((at, handle));
                    }
                    Err(_) => {
                        self.stage_slots.fetch_add(1, Ordering::AcqRel);
                    }
                }
            }
            for (at, item) in items.iter().enumerate() {
                let is_handled = if at < 64 { (handled_mask & (1u64 << at)) != 0 } else { handled_extra.contains(&at) };
                if !is_handled {
                    answers[at] = Some(work(item));
                }
            }
            for (at, handle) in helpers.drain(..) {
                match handle.join() {
                    Ok(Ok(answer)) => answers[at] = Some(answer),
                    Ok(Err(panic)) | Err(panic) => std::panic::resume_unwind(panic),
                }
            }
        });
        answers.into_iter().flatten().collect()
    }

    /// The DXIL of one stage: from disk when another run compiled it, else compiled and stored.
    fn compile_stage(&self, key: u128, stage: &StageHlsl) -> Result<Arc<[u8]>, String> {
        if let Some(dxil) = self.cache.as_ref().and_then(|c| c.read(key)) {
            self.stage_hit();
            return Ok(Arc::from(dxil));
        }
        // A large live miss compiles with `-O0`: several times faster in DXC, still validated and
        // signed by `dxil.dll`. Never `-Vd`, the PC runtime refuses unsigned DXIL.
        let o0 = self.live.load(Ordering::Relaxed) && stage.hlsl.len() >= o0_hlsl_bytes();
        let args: std::borrow::Cow<'_, [String]> = if o0 {
            if stage.args.is_empty() {
                static O0_ONLY: OnceLock<[String; 1]> = OnceLock::new();
                std::borrow::Cow::Borrowed(O0_ONLY.get_or_init(|| ["-O0".to_string()]))
            } else {
                std::borrow::Cow::Owned(stage.args.iter().cloned().chain(["-O0".to_string()]).collect())
            }
        } else {
            std::borrow::Cow::Borrowed(&stage.args)
        };
        let _large_guard = (self.live.load(Ordering::Relaxed) && stage.hlsl.len() >= LARGE_STAGE_HLSL_BYTES)
            .then(|| self.large_gate.acquire());
        let compile_started = std::time::Instant::now();
        let compile_result = self.compiler.compile(&stage.hlsl, &stage.entry, &stage.profile, &args);
        self.counters.dxc_compile_us.fetch_add(compile_started.elapsed().as_micros() as u64, Ordering::Relaxed);
        match compile_result {
            Ok(dxil) => {
                self.counters.compiled.fetch_add(1, Ordering::Relaxed);
                if o0 {
                    let n = self.counters.compiled_o0.fetch_add(1, Ordering::Relaxed);
                    if log_due(n) {
                        (self.log)(&format!("direct shaders: stage {key:032x} ({}) has {} bytes of HLSL, compiled with -O0 ({} so far)", stage.profile, stage.hlsl.len(), n + 1));
                    }
                }
                if let Some(cache) = &self.cache
                    && let Err(e) = cache.write(key, &dxil)
                {
                    static SHOWN: AtomicU64 = AtomicU64::new(0);
                    if log_due(SHOWN.fetch_add(1, Ordering::Relaxed)) {
                        (self.log)(&format!("direct shaders: cache write to {} failed ({e})", cache.dir().display()));
                    }
                }
                Ok(Arc::from(dxil))
            }
            Err(msg) => {
                let n = self.counters.dxc_errors.fetch_add(1, Ordering::Relaxed);
                if let Some(dir) = &self.dump {
                    let _ = std::fs::create_dir_all(dir);
                    let _ = std::fs::write(dir.join(format!("{key:032x}.{}.hlsl", stage.profile)), &stage.hlsl);
                    let _ = std::fs::write(
                        dir.join(format!("{key:032x}.{}.txt", stage.profile)),
                        format!("entry {}\nprofile {}\nargs {:?}\ncompiler {}\n\n{msg}\n", stage.entry, stage.profile, args, self.compiler_version),
                    );
                }
                if log_due(n) {
                    (self.log)(&format!("direct shaders: DXC refused stage {key:032x} ({}): {}", stage.profile, msg.lines().next().unwrap_or("")));
                }
                Err(msg)
            }
        }
    }
}

fn worker(shared: &Shared, rx: &Mutex<Receiver<Work>>) {
    loop {
        let work = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
        let job = match work {
            Ok(Work::Compile(job)) => job,
            #[cfg(windows)]
            Ok(Work::Deferred(run)) => {
                run();
                continue;
            }
            Err(_) => return,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| shared.run(&job)))
            .unwrap_or_else(|_| shared.refused(&Refusal::new("translator panic", format!("{:032x}", job.key))));
        shared.finish(job.key, &job.ticket, result);
    }
}

#[cfg(any(windows, test))]
fn retry_translation<T>(mut translate: impl FnMut() -> T) -> std::thread::Result<T> {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    catch_unwind(AssertUnwindSafe(&mut translate))
        .or_else(|_| catch_unwind(AssertUnwindSafe(translate)))
}

#[cfg(test)]
#[test]
fn translator_panic_gets_one_retry() {
    let mut attempts = 0;
    assert_eq!(retry_translation(|| {
        attempts += 1;
        assert_eq!(attempts, 2);
        42
    }).unwrap(), 42);
    assert_eq!(attempts, 2);

    attempts = 0;
    assert!(retry_translation(|| {
        attempts += 1;
        panic!("persistent translator panic");
    }).is_err());
    assert_eq!(attempts, 2);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::direct_cache::tests::TempDir;
    use crate::rts0::RootLayout;
    use crate::shader_source::{FixedSource, Linkage, StageInput, StageKind};

    /// A compiler that wraps the HLSL in a fake container and counts its calls. `FAIL` in the HLSL
    /// is a compile error. Slow enough that concurrent jobs overlap.
    #[derive(Default)]
    pub(crate) struct FakeCompiler {
        pub(crate) calls: AtomicU64,
    }

    impl Compiler for FakeCompiler {
        fn version(&self) -> String {
            "fake 1".to_string()
        }

        fn compile(&self, hlsl: &str, _entry: &str, profile: &str, _args: &[String]) -> Result<Vec<u8>, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
            if hlsl.contains("FAIL") {
                return Err("error: FAIL is not HLSL\nsecond line".to_string());
            }
            Ok([b"DXBC".as_slice(), profile.as_bytes()].concat())
        }
    }

    pub(crate) fn pipeline(stages: &[(StageKind, &[u8])]) -> Pipeline {
        Pipeline {
            stages: stages.iter().map(|(k, b)| StageInput { kind: *k, bytes: Arc::from(*b) }).collect(),
            root: RootLayout { version: 2, flags: 0, params: Vec::new(), static_samplers: Vec::new() },
            linkage: Linkage { stages: stages.iter().map(|(k, _)| *k).collect(), ..Linkage::default() },
        }
    }

    fn poll_until_final(service: &Service, id: usize, p: &Pipeline) -> Poll {
        for _ in 0..500 {
            match service.poll(id, || Ok(p.clone())) {
                Poll::Pending => std::thread::sleep(Duration::from_millis(5)),
                other => return other,
            }
        }
        panic!("pipeline {id} never finished");
    }

    #[test]
    fn pipeline_summary_logs_at_the_first_pipeline_periodically_on_change_and_at_teardown() {
        static SUMMARIES: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let settings = Settings {
            log: |line| SUMMARIES.lock().unwrap().push(line.to_owned()),
            ..Settings::quiet(1)
        };
        let service = Service::new(Arc::new(FixedSource::simple()), Arc::new(FakeCompiler::default()), settings);
        service.pipeline_done(PipelineKind::Graphics, None);
        assert_eq!(SUMMARIES.lock().unwrap().len(), 1, "a short run gets a summary at its first pipeline");
        for _ in 1..SUMMARY_INTERVAL - 2 {
            service.pipeline_done(PipelineKind::Graphics, None);
        }
        service.pipeline_done(PipelineKind::Compute, Some((false, Duration::from_micros(15))));
        assert_eq!(SUMMARIES.lock().unwrap().len(), 1);
        service.pipeline_done(PipelineKind::Mesh, Some((true, Duration::from_micros(25))));
        service.log_summary_if_changed();
        assert_eq!(SUMMARIES.lock().unwrap().len(), 2, "the timer logs nothing new when no counter changed");
        service.pipeline_done(PipelineKind::Mesh, None);
        service.log_summary_if_changed();
        drop(service);
        let lines = SUMMARIES.lock().unwrap();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("pipelines vs+ps=1 cs=0 ms+ps=0, PC made vs+ps=0 cs=0 ms+ps=0"));
        assert!(lines[1].contains("pipelines vs+ps=62 cs=1 ms+ps=1, PC made vs+ps=0 cs=0 ms+ps=1"));
        assert!(lines[2].contains("pipelines vs+ps=62 cs=1 ms+ps=2, PC made vs+ps=0 cs=0 ms+ps=1"));
        assert_eq!(lines[3], lines[2]);
        assert!(lines[3].contains("ready=0 pipe_hits=0 stage_hits=0 translated=0 cache_hits=0 compiled=0 compiled_o0=0 dxc_errors=0"));
        assert!(lines[3].ends_with("translation_us=0 dxc_wait_us=0 dxc_compile_us=0 pc_creation_us=40"));
    }

    /// A compiler that keeps the HLSL size and the extra arguments of each call.
    #[derive(Default)]
    struct ArgsCompiler {
        calls: Mutex<Vec<(usize, Vec<String>)>>,
    }

    impl Compiler for ArgsCompiler {
        fn version(&self) -> String {
            "args 1".to_string()
        }

        fn compile(&self, hlsl: &str, _entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
            self.calls.lock().unwrap().push((hlsl.len(), args.to_vec()));
            Ok([b"DXBC".as_slice(), profile.as_bytes()].concat())
        }
    }

    /// The simple source with a compute stage of exactly `len` bytes of HLSL.
    fn sized_source(len: usize) -> FixedSource {
        let mut source = FixedSource::simple();
        let mut hlsl = "[numthreads(1, 1, 1)] void main() {}".to_string();
        hlsl.extend(std::iter::repeat_n(' ', len - hlsl.len()));
        source.stages[0].1.hlsl = hlsl;
        source
    }

    #[test]
    fn a_live_miss_of_512_kib_or_more_compiles_with_o0_and_never_skips_signing() {
        for (live, len, o0) in [(true, O0_HLSL_BYTES - 1, false), (true, O0_HLSL_BYTES, true), (false, 2 * O0_HLSL_BYTES, false)] {
            let compiler = Arc::new(ArgsCompiler::default());
            let service = Service::new(Arc::new(sized_source(len)), compiler.clone(), Settings::quiet(1));
            if live {
                service.set_live(MAX_LIVE_HLSL_BYTES);
            }
            let compute = pipeline(&[(StageKind::Compute, b"cs sized")]);
            assert!(matches!(poll_until_final(&service, 1, &compute), Poll::Ready(_)));
            let calls = compiler.calls.lock().unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0, len);
            assert_eq!(calls[0].1.iter().any(|a| a == "-O0"), o0, "live {live}, {len} bytes");
            assert!(!calls[0].1.iter().any(|a| a == "-Vd"), "DXIL stays signed");
            assert_eq!(service.counters().compiled_o0.load(Ordering::Relaxed), u64::from(o0));
        }
    }

    #[test]
    fn a_live_stage_over_8_mib_skips_dxc_and_an_offline_build_still_serves_it() {
        let tmp = TempDir::new("live-limit");
        let settings = || Settings { cache: Some(DiskCache::new(&tmp.0)), ..Settings::quiet(1) };
        let compute = pipeline(&[(StageKind::Compute, b"cs huge")]);

        let compiler = Arc::new(ArgsCompiler::default());
        let at_limit = Service::new(Arc::new(sized_source(MAX_LIVE_HLSL_BYTES)), compiler.clone(), Settings::quiet(1));
        at_limit.set_live(MAX_LIVE_HLSL_BYTES);
        assert!(matches!(poll_until_final(&at_limit, 1, &compute), Poll::Ready(_)), "8 MiB itself still compiles");
        assert_eq!(compiler.calls.lock().unwrap().len(), 1);

        let compiler = Arc::new(ArgsCompiler::default());
        let over = Service::new(Arc::new(sized_source(MAX_LIVE_HLSL_BYTES + 1)), compiler.clone(), settings());
        over.set_live(MAX_LIVE_HLSL_BYTES);
        #[cfg(windows)]
        {
            let make = |len: usize| Translated {
                keys: vec![0x99],
                hlsl: vec![StageHlsl { hlsl: " ".repeat(len), entry: "main".into(), profile: "cs_6_6".into(), args: Vec::new() }],
                started: std::time::Instant::now(),
            };
            assert!(over.small_or_cached(&make(O0_HLSL_BYTES - 1)));
            assert!(!over.small_or_cached(&make(O0_HLSL_BYTES)));
            assert!(!over.small_or_cached(&make(MAX_LIVE_HLSL_BYTES)));
            assert!(over.small_or_cached(&make(MAX_LIVE_HLSL_BYTES + 1)), "over limit refuses without DXC so the fast lane handles it");
        }
        let refused = poll_until_final(&over, 1, &compute);
        let over_bytes = format!("{} bytes", MAX_LIVE_HLSL_BYTES + 1);
        assert!(matches!(&refused, Poll::Refused(r) if r.starts_with("live hlsl limit (cs ") && r.contains(&over_bytes)), "{refused:?}");
        assert!(compiler.calls.lock().unwrap().is_empty(), "DXC never runs");
        assert_eq!(over.counters().dxc_errors.load(Ordering::Relaxed), 0);
        assert_eq!(over.counters().refused().get("live hlsl limit"), Some(&1));

        // An offline service has no limit and keeps -O3, then the live one takes its DXIL.
        let offline = Service::new(Arc::new(sized_source(MAX_LIVE_HLSL_BYTES + 1)), compiler.clone(), settings());
        assert!(matches!(poll_until_final(&offline, 1, &compute), Poll::Ready(_)));
        assert_eq!(compiler.calls.lock().unwrap().iter().map(|(_, args)| args.len()).collect::<Vec<_>>(), [0]);
        let live = Service::new(Arc::new(sized_source(MAX_LIVE_HLSL_BYTES + 1)), compiler.clone(), settings());
        live.set_live(MAX_LIVE_HLSL_BYTES);
        assert!(matches!(live.poll(1, || Ok(compute.clone())), Poll::Ready(_)));
        assert_eq!(live.counters().pipe_hits.load(Ordering::Relaxed), 1);
        assert_eq!(compiler.calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn the_same_content_is_compiled_once_across_pipelines_and_threads() {
        let compiler = Arc::new(FakeCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(4));
        let graphics = pipeline(&[(StageKind::Vertex, b"vs bytes"), (StageKind::Pixel, b"ps bytes")]);
        // Eight objects of the same content, polled at once: one job, two compiles.
        let polls: Vec<Poll> = (0..8).map(|id| service.poll(id, || Ok(graphics.clone()))).collect();
        assert!(polls.iter().all(|p| *p == Poll::Pending));
        assert!(service.wait_idle(Duration::from_secs(10)));
        for id in 0..8 {
            match service.poll(id, || panic!("built once")) {
                Poll::Ready(dxil) => assert_eq!(dxil.iter().map(|d| d.to_vec()).collect::<Vec<_>>(), [b"DXBCvs_6_0".to_vec(), b"DXBCps_6_0".to_vec()]),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 2);
        // Another pipeline with the same vertex stage and linkage but another pixel stage: the
        // vertex stage has the same key and comes from memory.
        let other = pipeline(&[(StageKind::Vertex, b"vs bytes"), (StageKind::Pixel, b"other ps")]);
        assert_eq!(service.poll(100, || Ok(other.clone())), Poll::Pending);
        assert!(matches!(poll_until_final(&service, 100, &other), Poll::Ready(_)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 3, "only the new pixel stage is compiled");
        assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 3);
        assert_eq!(service.counters().cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(service.counters().stage_hits.load(Ordering::Relaxed), 1);
        assert_eq!(service.counters().pipe_hits.load(Ordering::Relaxed), 0);
        assert_eq!(service.counters().translated.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn a_second_service_reads_the_disk_cache_without_translating_or_compiling() {
        let tmp = TempDir::new("service");
        let settings = || Settings { cache: Some(DiskCache::new(&tmp.0)), ..Settings::quiet(2) };
        let compute = pipeline(&[(StageKind::Compute, b"cs bytes")]);
        let first = Service::new(Arc::new(FixedSource::simple()), Arc::new(FakeCompiler::default()), settings());
        assert!(matches!(poll_until_final(&first, 1, &compute), Poll::Ready(_)));
        let compiler = Arc::new(FakeCompiler::default());
        let second = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), settings());
        assert_eq!(second.poll(1, || Ok(compute.clone())), Poll::Ready(vec![Arc::from(&b"DXBCcs_6_0"[..])]), "a hit is ready on the first poll");
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(second.counters().cache_hits.load(Ordering::Relaxed), 1);
        assert_eq!(second.counters().pipe_hits.load(Ordering::Relaxed), 1);
        assert_eq!(second.counters().stage_hits.load(Ordering::Relaxed), 0);
        assert_eq!(second.counters().translated.load(Ordering::Relaxed), 0);
        // Another translator version is another key: compiled again.
        let third = Service::new(Arc::new(FixedSource { generation: 1, ..FixedSource::simple() }), compiler.clone(), settings());
        assert_eq!(third.poll(1, || Ok(compute.clone())), Poll::Pending);
        assert!(matches!(poll_until_final(&third, 1, &compute), Poll::Ready(_)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn direct_source_pipelines_hit_the_disk_cache_without_translating_or_compiling() {
        use crate::direct_fixture::{container, mesh_triangle, ui_pixel, ui_vertex};
        use crate::direct_source::DirectSource;
        use crate::rts0::{RegisterClass, RootParam, RootParamKind, RootRange};

        static DIRECT_LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let log = |line: &str| DIRECT_LOGS.lock().unwrap().push(line.to_owned());

        let tmp = TempDir::new("direct-source-cache");
        let root = RootLayout {
            version: 2,
            flags: 0xC01,
            params: vec![
                RootParam {
                    kind: RootParamKind::Table(vec![
                        RootRange { class: RegisterClass::Srv, count: 1, base: 2, space: 1, flags: 0, offset: 0 },
                        RootRange { class: RegisterClass::Cbv, count: 1, base: 0, space: 1, flags: 0, offset: 1 },
                    ]),
                    visibility: 0,
                },
                RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 },
            ],
            static_samplers: Vec::new(),
        };
        let ui = Pipeline {
            stages: vec![
                StageInput { kind: StageKind::Vertex, bytes: Arc::from(container(&ui_vertex())) },
                StageInput { kind: StageKind::Pixel, bytes: Arc::from(container(&ui_pixel())) },
            ],
            root: root.clone(),
            linkage: Linkage { stages: vec![StageKind::Vertex, StageKind::Pixel], ..Linkage::default() },
        };
        let mesh = Pipeline {
            stages: vec![
                StageInput { kind: StageKind::Mesh, bytes: Arc::from(container(&mesh_triangle())) },
                StageInput { kind: StageKind::Pixel, bytes: Arc::from(container(&ui_pixel())) },
            ],
            root,
            linkage: Linkage { stages: vec![StageKind::Mesh, StageKind::Pixel], ..Linkage::default() },
        };

        let first_source = Arc::new(DirectSource::new(Arc::new(decant_shader::api::CountingHost::default()), log, None));
        let first_compiler = Arc::new(FakeCompiler::default());
        let first = Service::new(first_source, first_compiler.clone(), Settings { threads: 2, cache: Some(DiskCache::new(&tmp.0)), dump: None, log });
        assert!(matches!(poll_until_final(&first, 1, &ui), Poll::Ready(_)));
        assert!(matches!(poll_until_final(&first, 2, &mesh), Poll::Ready(_)));
        assert!(first_compiler.calls.load(Ordering::SeqCst) >= 3);
        drop(first);
        DIRECT_LOGS.lock().unwrap().clear();

        let second_source = Arc::new(DirectSource::new(Arc::new(decant_shader::api::CountingHost::default()), log, None));
        let second_compiler = Arc::new(FakeCompiler::default());
        let second = Service::new(second_source, second_compiler.clone(), Settings { threads: 2, cache: Some(DiskCache::new(&tmp.0)), dump: None, log });
        assert_eq!(second.poll(99, || Ok(mesh.clone())), Poll::Ready(vec![Arc::from(&b"DXBCms_6_6"[..]), Arc::from(&b"DXBCps_6_6"[..])]));
        assert_eq!(second.poll(98, || Ok(ui.clone())), Poll::Ready(vec![Arc::from(&b"DXBCvs_6_6"[..]), Arc::from(&b"DXBCps_6_6"[..])]));
        assert_eq!(second_compiler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(second.counters().translated.load(Ordering::Relaxed), 0);
        assert_eq!(second.counters().cache_hits.load(Ordering::Relaxed), 4);
        drop(second);

        let lines = DIRECT_LOGS.lock().unwrap();
        assert!(!lines.iter().any(|l| l.contains("the translation of")), "DirectSource must not translate on a warm cache hit: {lines:?}");
        assert!(lines.iter().any(|l| l.contains("ready from cache") && l.contains("cache hits")), "{lines:?}");
        assert!(lines.iter().any(|l| l.contains("direct shaders summary:") && l.contains("cache_hits=4") && l.contains("compiled=0")), "{lines:?}");
    }

    #[test]
    fn refusals_and_dxc_errors_are_counted_and_dumped() {
        let tmp = TempDir::new("dump");
        let mut source = FixedSource::simple();
        source.stages.push((StageKind::Geometry, StageHlsl { hlsl: "FAIL".into(), entry: "main".into(), profile: "gs_6_0".into(), args: Vec::new() }));
        let service = Service::new(Arc::new(source), Arc::new(FakeCompiler::default()), Settings { dump: Some(tmp.0.clone()), ..Settings::quiet(1) });
        let refused = pipeline(&[(StageKind::Vertex, b"vs"), (StageKind::Pixel, b"REFUSE")]);
        assert_eq!(poll_until_final(&service, 1, &refused), Poll::Refused("test refusal (ps)".into()));
        let failing = pipeline(&[(StageKind::Geometry, b"gs")]);
        assert!(matches!(poll_until_final(&service, 2, &failing), Poll::Refused(r) if r.starts_with("dxc error (gs ") && r.ends_with(": error: FAIL is not HLSL)")));
        assert_eq!(service.poll(3, || Err(Refusal::new("root signature unknown", ""))), Poll::Refused("root signature unknown".into()));
        let c = service.counters();
        assert_eq!(c.dxc_errors.load(Ordering::Relaxed), 1);
        assert_eq!(c.refused().into_iter().collect::<Vec<_>>(), [("dxc error".to_string(), 1), ("root signature unknown".to_string(), 1), ("test refusal".to_string(), 1)]);
        let names: Vec<String> = std::fs::read_dir(&tmp.0).expect("dump folder").filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(names.len(), 2);
        let txt = names.iter().find(|n| n.ends_with(".gs_6_0.txt")).expect("message file");
        assert!(std::fs::read_to_string(tmp.0.join(txt)).expect("read").contains("second line"));
        assert!(c.summary().contains("1 DXC errors"));
    }

    #[test]
    fn a_caller_can_wait_for_its_pipeline() {
        let service = Service::new(Arc::new(FixedSource::simple()), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let p = pipeline(&[(StageKind::Vertex, b"vs to wait for"), (StageKind::Pixel, b"ps to wait for")]);
        assert_eq!(service.poll(7, || Ok(p.clone())), Poll::Pending);
        // Two compiles of 20 ms each: not done within a millisecond, done within the long limit.
        assert!(!service.wait(7, Duration::from_millis(1)));
        assert!(service.wait(7, Duration::from_secs(10)));
        assert!(matches!(service.poll(7, || panic!("built once")), Poll::Ready(_)));
        assert!(service.wait(7, Duration::ZERO), "nothing to wait for once the answer was taken");
    }

    /// A source that refuses every pipeline at the first look.
    struct Refusing;

    impl ShaderSource for Refusing {
        fn version(&self) -> String {
            "refusing 1".to_string()
        }

        fn accepts(&self, _pipeline: &Pipeline) -> Result<(), Refusal> {
            Err(Refusal::new("not taken", ""))
        }

        fn translate(&self, _pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
            panic!("never asked");
        }
    }

    #[test]
    fn a_source_that_refuses_up_front_queues_nothing() {
        let compiler = Arc::new(FakeCompiler::default());
        let service = Service::new(Arc::new(Refusing), compiler.clone(), Settings::quiet(1));
        let p = pipeline(&[(StageKind::Compute, b"cs")]);
        assert_eq!(service.poll(1, || Ok(p.clone())), Poll::Refused("not taken".into()));
        assert!(service.wait_idle(Duration::from_millis(1)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 0);
    }

    /// A source whose pixel HLSL follows the vertex bytes, and that says so in its keys.
    struct Linked;

    impl ShaderSource for Linked {
        fn version(&self) -> String {
            "linked 1".to_string()
        }

        fn stage_keys(&self, pipeline: &Pipeline) -> Option<Vec<u128>> {
            let vs = u128::from(pipeline.stages[0].bytes[0]);
            Some(pipeline.stages.iter().map(|st| vs << 8 | u128::from(st.bytes[0])).collect())
        }

        fn translate(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
            FixedSource::simple().translate(pipeline)
        }
    }

    #[test]
    fn the_keys_of_the_source_decide_what_is_shared() {
        let compiler = Arc::new(FakeCompiler::default());
        let service = Service::new(Arc::new(Linked), compiler.clone(), Settings::quiet(1));
        let first = pipeline(&[(StageKind::Vertex, b"a vs"), (StageKind::Pixel, b"p ps")]);
        assert!(matches!(poll_until_final(&service, 1, &first), Poll::Ready(_)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 2);
        // The same pixel bytes behind another vertex stage: another key, compiled again.
        let second = pipeline(&[(StageKind::Vertex, b"b vs"), (StageKind::Pixel, b"p ps")]);
        assert!(matches!(poll_until_final(&service, 2, &second), Poll::Ready(_)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 4);
        // Other bytes with the same keys (the first byte of each stage): nothing is compiled.
        let third = pipeline(&[(StageKind::Vertex, b"a vs, other tail"), (StageKind::Pixel, b"p ps, other tail")]);
        assert!(matches!(poll_until_final(&service, 3, &third), Poll::Ready(_)));
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 4);
    }

    /// A compiler that counts the compiles running at once. With `meet` each compile waits up to
    /// 2 seconds for another one to run beside it, else it takes 40 ms. HLSL with `PANIC` panics.
    #[derive(Default)]
    struct OverlapCompiler {
        meet: bool,
        running: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Compiler for OverlapCompiler {
        fn version(&self) -> String {
            "overlap 1".to_string()
        }

        fn compile(&self, hlsl: &str, _entry: &str, profile: &str, _args: &[String]) -> Result<Vec<u8>, String> {
            assert!(!hlsl.contains("PANIC"), "compiler panic for the test");
            self.peak.fetch_max(self.running.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
            if self.meet {
                let deadline = std::time::Instant::now() + Duration::from_secs(2);
                while self.running.load(Ordering::SeqCst) < 2 && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
            } else {
                std::thread::sleep(Duration::from_millis(40));
            }
            self.running.fetch_sub(1, Ordering::SeqCst);
            Ok([b"DXBC".as_slice(), profile.as_bytes()].concat())
        }
    }

    fn graphics() -> Pipeline {
        pipeline(&[(StageKind::Vertex, b"vs overlap"), (StageKind::Pixel, b"ps overlap")])
    }

    #[test]
    fn the_stages_of_one_pipeline_compile_side_by_side_when_a_slot_is_free() {
        let compiler = Arc::new(OverlapCompiler { meet: true, ..Default::default() });
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(1));
        service.set_stage_slots(1);
        let graphics = graphics();
        match poll_until_final(&service, 1, &graphics) {
            Poll::Ready(dxil) => assert_eq!(dxil.iter().map(|d| d.to_vec()).collect::<Vec<_>>(), [b"DXBCvs_6_0".to_vec(), b"DXBCps_6_0".to_vec()], "the order of the stages is kept"),
            other => panic!("{other:?}"),
        }
        assert_eq!(compiler.peak.load(Ordering::SeqCst), 2, "the vertex and pixel stage ran together");
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 1, "the slot came back");
        assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn without_a_free_slot_one_worker_compiles_its_stages_one_after_the_other() {
        let compiler = Arc::new(OverlapCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(1));
        service.set_stage_slots(0);
        assert!(matches!(poll_until_final(&service, 1, &graphics()), Poll::Ready(_)));
        assert_eq!(compiler.peak.load(Ordering::SeqCst), 1);
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn the_slots_bound_the_helpers_of_the_whole_service() {
        // Two workers with two pipelines of two stages each and one slot: at most 3 compiles at once.
        let compiler = Arc::new(OverlapCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(2));
        service.set_stage_slots(1);
        let first = pipeline(&[(StageKind::Vertex, b"vs one"), (StageKind::Pixel, b"ps one")]);
        let second = pipeline(&[(StageKind::Vertex, b"vs two"), (StageKind::Pixel, b"ps two")]);
        assert_eq!(service.poll(1, || Ok(first.clone())), Poll::Pending);
        assert_eq!(service.poll(2, || Ok(second.clone())), Poll::Pending);
        assert!(service.wait_idle(Duration::from_secs(10)));
        assert!(compiler.peak.load(Ordering::SeqCst) <= 3, "peak {}", compiler.peak.load(Ordering::SeqCst));
        assert!(matches!(service.poll(1, || panic!("built once")), Poll::Ready(_)));
        assert!(matches!(service.poll(2, || panic!("built once")), Poll::Ready(_)));
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failing_or_panicking_helper_gives_a_refusal_and_its_slot_back() {
        let mut source = FixedSource::simple();
        source.stages.push((StageKind::Geometry, StageHlsl { hlsl: "PANIC".into(), entry: "main".into(), profile: "gs_6_0".into(), args: Vec::new() }));
        let service = Service::new(Arc::new(source), Arc::new(OverlapCompiler::default()), Settings::quiet(1));
        service.set_stage_slots(1);
        let panicking = pipeline(&[(StageKind::Vertex, b"vs first"), (StageKind::Geometry, b"gs second")]);
        let answer = poll_until_final(&service, 1, &panicking);
        assert!(matches!(&answer, Poll::Refused(reason) if reason.starts_with("translator panic")), "{answer:?}");
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 1, "the slot came back after the panic");
        // The worker lives on and compiles the next pipeline.
        assert!(matches!(poll_until_final(&service, 2, &graphics()), Poll::Ready(_)));
    }

    #[test]
    fn a_dxc_error_in_a_helper_stage_refuses_the_pipeline_like_before() {
        let mut source = FixedSource::simple();
        source.stages.push((StageKind::Geometry, StageHlsl { hlsl: "FAIL".into(), entry: "main".into(), profile: "gs_6_0".into(), args: Vec::new() }));
        let service = Service::new(Arc::new(source), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        service.set_stage_slots(1);
        let failing = pipeline(&[(StageKind::Vertex, b"vs first"), (StageKind::Geometry, b"gs second")]);
        assert!(matches!(poll_until_final(&service, 1, &failing), Poll::Refused(r) if r.starts_with("dxc error (gs ")));
        assert_eq!(service.counters().dxc_errors.load(Ordering::Relaxed), 1);
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parse_o0_hlsl_bytes_handles_defaults_overrides_and_off() {
        assert_eq!(O0_HLSL_BYTES, 8 * 1024);
        assert_eq!(MAX_LIVE_HLSL_BYTES, 16 * 1024 * 1024);
        assert_eq!(parse_o0_hlsl_bytes(None), O0_HLSL_BYTES);
        assert_eq!(parse_o0_hlsl_bytes(Some("")), O0_HLSL_BYTES);
        assert_eq!(parse_o0_hlsl_bytes(Some("not-a-number")), O0_HLSL_BYTES);
        assert_eq!(parse_o0_hlsl_bytes(Some(" 262144 ")), 262_144);
        for off in ["0", "off", "OFF", "false"] {
            assert_eq!(parse_o0_hlsl_bytes(Some(off)), usize::MAX, "{off}");
        }
    }

    #[test]
    fn live_stage_at_8_kib_uses_o0_even_with_empty_extra_args() {
        let compiler = Arc::new(ArgsCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(1));
        service.set_live(MAX_LIVE_HLSL_BYTES);
        let stage = StageHlsl { hlsl: "x".repeat(O0_HLSL_BYTES), entry: "main".into(), profile: "ps_6_0".into(), args: Vec::new() };
        assert!(service.shared.compile_stage(0x1234, &stage).is_ok());
        let calls = compiler.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, ["-O0"]);
    }

    #[test]
    fn ticket_complete_is_lock_free_and_keeps_first_result() {
        let ticket = Ticket::default();
        assert_eq!(ticket.poll(), Poll::Pending);
        let dxil = vec![Arc::<[u8]>::from(&b"DXBC"[..])];
        ticket.complete(Poll::Ready(dxil.clone()));
        assert_eq!(ticket.poll(), Poll::Ready(dxil.clone()));
        ticket.complete(Poll::Refused("ignored".to_string()));
        assert_eq!(ticket.poll(), Poll::Ready(dxil));
    }

    #[test]
    fn single_stage_map_and_cached_poll_leave_active_counts_zero() {
        let cache_dir = crate::direct_cache::tests::TempDir::new("compile-service-fast-path");
        let mut settings = Settings::quiet(1);
        settings.cache = Some(DiskCache::new(cache_dir.0.clone()));
        let service = Service::new(Arc::new(FixedSource::simple()), Arc::new(FakeCompiler::default()), settings);
        let compute = pipeline(&[(StageKind::Compute, b"cs fast")]);
        assert!(matches!(poll_until_final(&service, 10, &compute), Poll::Ready(_)));
        assert_eq!(service.shared.active_jobs.load(Ordering::Acquire), 0);
        assert_eq!(service.shared.active_objects.load(Ordering::Acquire), 0);
        assert!(!service.holds(10));
        assert!(matches!(service.poll(11, || Ok(compute.clone())), Poll::Ready(_)));
        assert_eq!(service.shared.active_jobs.load(Ordering::Acquire), 0);
        assert_eq!(service.shared.active_objects.load(Ordering::Acquire), 0);
        assert!(service.shared.map_stages::<i32, i32>(&[], |x| x + 1).is_empty());
        assert_eq!(service.shared.map_stages(&[41], |x| x + 1), vec![42]);
    }

    #[test]
    fn compile_stages_skips_helper_spawns_when_one_stage_misses() {
        let cache_dir = crate::direct_cache::tests::TempDir::new("compile-stages-partial-hit");
        let mut settings = Settings::quiet(1);
        settings.cache = Some(DiskCache::new(cache_dir.0.clone()));
        let compiler = Arc::new(OverlapCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), settings);
        service.set_stage_slots(1);

        let first = pipeline(&[(StageKind::Vertex, b"vs cached"), (StageKind::Pixel, b"ps first")]);
        assert!(matches!(poll_until_final(&service, 1, &first), Poll::Ready(_)));
        assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 2);

        let second = pipeline(&[(StageKind::Vertex, b"vs cached"), (StageKind::Pixel, b"ps second")]);
        assert!(matches!(poll_until_final(&service, 2, &second), Poll::Ready(_)));
        assert_eq!(service.shared.stage_slots.load(Ordering::SeqCst), 1);
        assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn default_worker_cap_leaves_half_the_cores_free_and_caps_at_eight() {
        assert_eq!(default_worker_cap(1), 2);
        assert_eq!(default_worker_cap(4), 2);
        assert_eq!(default_worker_cap(8), 4);
        assert_eq!(default_worker_cap(16), 8);
        assert_eq!(default_worker_cap(32), 8);
    }

    #[test]
    fn parse_large_dxc_limit_handles_defaults_overrides_and_unlimited() {
        assert_eq!(LARGE_STAGE_HLSL_BYTES, 256 * 1024);
        assert_eq!(DEFAULT_LARGE_DXC_LIMIT, 2);
        assert_eq!(parse_large_dxc_limit(None), DEFAULT_LARGE_DXC_LIMIT);
        assert_eq!(parse_large_dxc_limit(Some("")), DEFAULT_LARGE_DXC_LIMIT);
        assert_eq!(parse_large_dxc_limit(Some("not-a-number")), DEFAULT_LARGE_DXC_LIMIT);
        assert_eq!(parse_large_dxc_limit(Some(" 4 ")), 4);
        for unlimited in ["0", "off", "OFF", "false"] {
            assert_eq!(parse_large_dxc_limit(Some(unlimited)), usize::MAX, "{unlimited}");
        }
    }

    #[test]
    fn live_large_stage_gate_caps_giant_compiles_without_blocking_small_or_offline() {
        #[derive(Default)]
        struct TrackingCompiler {
            large_active: AtomicUsize,
            large_peak: AtomicUsize,
            small_active: AtomicUsize,
            small_peak: AtomicUsize,
            release: (Mutex<bool>, Condvar),
        }

        impl Compiler for TrackingCompiler {
            fn version(&self) -> String {
                "tracking 1".to_string()
            }

            fn compile(&self, hlsl: &str, _entry: &str, profile: &str, _args: &[String]) -> Result<Vec<u8>, String> {
                let is_large = hlsl.len() >= LARGE_STAGE_HLSL_BYTES;
                if is_large {
                    let cur = self.large_active.fetch_add(1, Ordering::SeqCst) + 1;
                    self.large_peak.fetch_max(cur, Ordering::SeqCst);
                    let mut guard = self.release.0.lock().unwrap();
                    while !*guard {
                        guard = self.release.1.wait(guard).unwrap();
                    }
                    self.large_active.fetch_sub(1, Ordering::SeqCst);
                } else {
                    let cur = self.small_active.fetch_add(1, Ordering::SeqCst) + 1;
                    self.small_peak.fetch_max(cur, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(10));
                    self.small_active.fetch_sub(1, Ordering::SeqCst);
                }
                Ok([b"DXBC".as_slice(), profile.as_bytes()].concat())
            }
        }

        let compiler = Arc::new(TrackingCompiler::default());
        let service = Service::new(Arc::new(FixedSource::simple()), compiler.clone(), Settings::quiet(4));
        service.set_live(MAX_LIVE_HLSL_BYTES);
        service.shared.large_gate.limit.store(2, Ordering::Relaxed);

        let large_stage = StageHlsl { hlsl: "x".repeat(LARGE_STAGE_HLSL_BYTES), entry: "main".into(), profile: "cs_6_0".into(), args: Vec::new() };
        let small_stage = StageHlsl { hlsl: "x".repeat(1024), entry: "main".into(), profile: "cs_6_0".into(), args: Vec::new() };

        std::thread::scope(|scope| {
            for i in 0..4u128 {
                let svc = &service;
                let st = &large_stage;
                scope.spawn(move || {
                    assert!(svc.shared.compile_stage(100 + i, st).is_ok());
                });
            }
            // Wait until 2 large compiles are inside `compile`, then verify no more than 2 enter.
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while compiler.large_active.load(Ordering::SeqCst) < 2 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(2));
            }
            std::thread::sleep(Duration::from_millis(20));
            assert_eq!(compiler.large_active.load(Ordering::SeqCst), 2);
            assert_eq!(compiler.large_peak.load(Ordering::SeqCst), 2);

            // Small stages are not blocked by the large-stage gate.
            assert!(service.shared.compile_stage(200, &small_stage).is_ok());
            assert_eq!(compiler.small_peak.load(Ordering::SeqCst), 1);

            *compiler.release.0.lock().unwrap() = true;
            compiler.release.1.notify_all();
        });
        assert_eq!(compiler.large_peak.load(Ordering::SeqCst), 2);

        // Offline mode (`!live`) bypasses the large-stage gate.
        service.shared.live.store(false, Ordering::Relaxed);
        compiler.large_peak.store(0, Ordering::SeqCst);
        *compiler.release.0.lock().unwrap() = false;
        std::thread::scope(|scope| {
            for i in 0..3u128 {
                let svc = &service;
                let st = &large_stage;
                scope.spawn(move || {
                    assert!(svc.shared.compile_stage(300 + i, st).is_ok());
                });
            }
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while compiler.large_active.load(Ordering::SeqCst) < 3 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(2));
            }
            *compiler.release.0.lock().unwrap() = true;
            compiler.release.1.notify_all();
        });
        assert_eq!(compiler.large_peak.load(Ordering::SeqCst), 3);
    }
}
