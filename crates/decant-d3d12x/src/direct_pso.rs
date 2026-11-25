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
//! * `DECANT_D3D12X_PRECOMPILE_THREADS=<n>` sets the queue worker count (default 4, 0 off).
//! * `DECANT_D3D12X_DIRECT_THREADS=<n>` sets the worker count for binds that bypass the queue
//!   (default a quarter of the cores, 1 to 4).
//! * `DECANT_D3D12X_MAX_LIVE_HLSL_BYTES=<n>` is the largest stage HLSL compiled live (default 8 MiB,
//!   `0` no limit). A larger stage without cached DXIL is refused with its size in the log.
//!
//! The startup line names the translator hash of the build (`translator_hash=`), the same one the
//! cache keys hold. `direct shaders summary:` is logged at the first pipeline, every 64 pipelines,
//! every 15 s when a counter changed, and at exit, so a run that is killed keeps recent counts.
//!
//! The root signature a translator gets is the one the PC runtime got (`dxil_override` keeps the
//! layouts of the root signatures the title creates while this path is on).

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::direct_source::DirectSource;
use crate::precompile_queue::Queue;
use crate::compile_service::{PipelineKind, Poll, Service, Settings, MAX_LIVE_HLSL_BYTES, O0_HLSL_BYTES};
use crate::lazy_pso::{Owned, COMPUTE_DESC_SIZE, GRAPHICS_DESC_SIZE, MESH_DESC_SIZE};
use crate::log::xs_log;
use crate::shader_source::{InputElement, Linkage, Pipeline, Refusal, ShaderSource, StageInput, StageKind};

impl crate::compile_service::Compiler for decant_dxc::Dxc {
    fn version(&self) -> String {
        decant_dxc::Dxc::version(self)
    }

    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
        decant_dxc::Dxc::compile(self, hlsl, entry, profile, args)
    }
}

fn log_line(line: &str) {
    xs_log!("{line}");
}

/// Longest a single graphics or mesh pipeline may hold up its first bind.
const FIRST_BIND_WAIT: Duration = Duration::from_millis(50);

/// Frame wait budget when `DECANT_D3D12X_FRAME_WAIT_MS` is not set.
const DEFAULT_FRAME_WAIT_MS: u64 = 500;

/// Compute wait when `DECANT_D3D12X_COMPUTE_WAIT_MS` is not set.
const DEFAULT_COMPUTE_WAIT_MS: u64 = 5000;

/// How often the summary is logged when a counter changed.
const SUMMARY_PERIOD: Duration = Duration::from_secs(15);

/// Microseconds that cold graphics and mesh binds have waited since the last present.
static FRAME_SPENT_US: AtomicU64 = AtomicU64::new(0);

/// The translator hash of this build, part of every cache key.
const TRANSLATOR_HASH: &str = env!("DECANT_DIRECT_TRANSLATOR_HASH");

fn threads() -> usize {
    let configured = std::env::var("DECANT_D3D12X_DIRECT_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0);
    configured.unwrap_or_else(|| (std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).clamp(1, 4))
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
    xs_log!(
        "direct shaders on: {}, translator {}, translator_hash={TRANSLATOR_HASH}, {} workers, cache {}, dump {}, -O0 from {O0_HLSL_BYTES} HLSL bytes, live limit {}",
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
    });
    if let Err(e) = ticker {
        xs_log!("direct shaders: could not start the summary timer ({e})");
    }
    let service = Service::new(source, Arc::new(dxc), settings);
    service.set_live(live_limit);
    Some(service)
}

static SERVICE: OnceLock<Option<Service>> = OnceLock::new();
static SOURCE: OnceLock<Arc<DirectSource>> = OnceLock::new();
static QUEUE: OnceLock<Option<Arc<Queue>>> = OnceLock::new();
static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

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
            .and_then(|value| value.parse::<usize>().ok()).unwrap_or(4).min(64)
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
}

pub(crate) fn note_present() {
    FRAME_SPENT_US.store(0, Ordering::Relaxed);
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
    if let Some(service) = service.or_else(current) {
        service.pipeline_done(kind, creation);
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
pub fn kind_made(made: usize, owned: &Owned) {
    if !crate::kind_check::tracking() {
        return;
    }
    let desc = owned.desc();
    let graphics = desc.len() != COMPUTE_DESC_SIZE;
    let mut stages = Vec::new();
    for (kind, at) in stage_offsets(desc) {
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

/// A description pointing at compiled shaders it keeps alive.
pub struct Patched {
    desc: Vec<u8>,
    _dxil: Vec<Arc<[u8]>>,
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
    let spent = Duration::from_micros(FRAME_SPENT_US.load(Ordering::Relaxed));
    frame_wait_budget().saturating_sub(spent)
}

/// Takes `elapsed` from this frame's wait budget.
pub(crate) fn spend_frame_wait(elapsed: Duration) {
    let micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    FRAME_SPENT_US.fetch_add(micros, Ordering::Relaxed);
}

/// Counts a bind that went on without its pipeline, and logs only the first one of each object.
pub(crate) fn note_pending(object: usize, logged: &AtomicBool) {
    let total = LEFT_OUT.fetch_add(1, Ordering::Relaxed) + 1;
    if !logged.swap(true, Ordering::Relaxed) {
        xs_log!("direct shaders: pipeline {object:#x} is still pending, bind left work out ({total} pending binds)");
    }
}

/// Picks the shaders to make `owned` with. Waits up to `wait` for translation and DXC, using the
/// precompile queue when the pipeline has a `key` there.
pub(crate) fn choose_with_budget(object: usize, owned: &Owned, key: Option<u128>, wait: Duration) -> Choice {
    let Some(service) = current() else { return Choice::Original };
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
        Poll::Ready(dxil) => Choice::Direct(patch(owned.desc(), &dxil)),
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
    if !left.is_zero() && service.wait(object, left) { poll() } else { first }
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

/// The stages of a copied description with the offsets of their `D3D12_SHADER_BYTECODE`.
fn stage_offsets(desc: &[u8]) -> Vec<(StageKind, usize)> {
    let all: &[(StageKind, usize)] = match desc.len() {
        COMPUTE_DESC_SIZE => &[(StageKind::Compute, 0x08)],
        GRAPHICS_DESC_SIZE => &[(StageKind::Vertex, 0x08), (StageKind::Pixel, 0x18), (StageKind::Domain, 0x28), (StageKind::Hull, 0x38), (StageKind::Geometry, 0x48)],
        // A mesh pipeline: its stages in pipeline order, the pixel inputs follow the mesh outputs.
        MESH_DESC_SIZE => &[(StageKind::Amplification, crate::mesh_pso::AS_AT), (StageKind::Mesh, crate::mesh_pso::MS_AT), (StageKind::Pixel, crate::mesh_pso::PS_AT)],
        _ => &[],
    };
    all.iter().copied().filter(|&(_, at)| read_u64(desc, at) != 0 && read_u64(desc, at + 8) != 0).collect()
}

/// The translator's input for a copied description (also what `pipeline_manifest` records).
pub(crate) fn pipeline_of(owned: &Owned) -> Result<Pipeline, Refusal> {
    let desc = owned.desc();
    let offsets = stage_offsets(desc);
    if offsets.is_empty() {
        return Err(Refusal::new("no shader stage", ""));
    }
    let mut stages = Vec::with_capacity(offsets.len());
    for &(kind, at) in &offsets {
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
    Patched { desc: copy, _dxil: dxil.to_vec() }
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
    fn the_live_hlsl_limit_is_8_mib_unless_set() {
        assert_eq!(parse_live_limit(None), 8 * 1024 * 1024);
        assert_eq!(parse_live_limit(Some("soon")), 8 * 1024 * 1024);
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
        assert_eq!(stage_offsets(owned.desc()), [(StageKind::Vertex, 0x08), (StageKind::Pixel, 0x18)]);
        // The patched copy points at the compiled shaders, the rest is the same.
        let dxil: Vec<Arc<[u8]>> = vec![Arc::from(&b"DXBCvs"[..]), Arc::from(&b"DXBCps"[..])];
        let patched = patch(d, &dxil);
        assert_eq!(read_u64(patched.desc(), 0x18), dxil[1].as_ptr() as u64);
        assert_eq!(read_u64(patched.desc(), 0x20), 6);
        assert_eq!(patched.desc()[0x50..], d[0x50..]);
    }
}
