//! Pipelines that are made on their first bind instead of at creation.
//!
//! A title may create tens of thousands of pipelines while it boots and then bind only a few dozen
//! of them in its first frames. Made at creation, each one pays for the DXIL override lookup and a
//! driver compile of the translated shaders, often more than a megabyte of DXIL apiece. If the
//! title creates them from many jobs of its own and its boot flow waits for those jobs, the title
//! can sit in its boot state for minutes while every worker is inside the driver's compiler.
//!
//! Creation therefore answers at once with an object of ours. The object keeps a copy of the
//! description: the shaders (shared by content), the input layout, the stream output declaration
//! and the cached blob. When a list first binds the object (SetPipelineState, Reset,
//! CreateCommandList) `proxy` asks [`resolve`] for the PC pipeline, which is then made by the path
//! the creation took before. A pipeline the PC runtime refuses resolves to null, and the list
//! leaves out its work as it does for a stand in (`pso_diag`).
//!
//! The PC pipelines are made on a small pool of worker threads, so the title's recording threads
//! only wait for the time their bind is allowed (see `direct_pso::bind_wait`).
//!
//! `DECANT_D3D12X_LAZY_PSO=0` switches this off, and pipelines are made at creation again.
//!
//! Async bind mode. A compute pipeline normally waits up to `DECANT_D3D12X_COMPUTE_WAIT_MS` (default
//! 5000) on its first bind, so one cold pipeline can hold a recording thread for seconds while the
//! shader compiler works. With `DECANT_D3D12X_ASYNC_BIND=1` that wait is capped at
//! `DECANT_D3D12X_ASYNC_BIND_WAIT_MS` (default 20, `0` does not wait at all). A pipeline that is not
//! ready in time is bound as missing and its work is left out, exactly as for a compute pipeline
//! that timed out before. The workers go on with it in the background, the job is already urgent,
//! and the next bind takes the pipeline as soon as it is ready. Unset, empty or `0` keeps the default
//! behaviour. Graphics and mesh binds are not changed: their first bind already waits at most 50 ms
//! and is paid from the frame budget (`DECANT_D3D12X_FRAME_WAIT_MS`). A bind that was cut short is
//! counted and logged (`async bind: ...`), so a run shows how much work was left out.
//!
//! The waits of all cold async compute binds in one frame also share a budget,
//! `DECANT_D3D12X_ASYNC_BIND_FRAME_MS` (default 16, `0` does not wait at all). `PresentX` refills it
//! (`direct_pso::refill_frame_budget`). Once it is spent, a cold async bind returns at once, so
//! thousands of cold pipelines in one frame cost no more than the budget.

#![allow(unsafe_code)]

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::core::{GUID, HRESULT};

use crate::log::xs_log;

/// Size of `D3D12_GRAPHICS_PIPELINE_STATE_DESC`, from the public d3d12.h.
pub const GRAPHICS_DESC_SIZE: usize = crate::stream_pso::GRAPHICS_DESC_SIZE;
/// Size of `D3D12_COMPUTE_PIPELINE_STATE_DESC`, from the public d3d12.h.
pub const COMPUTE_DESC_SIZE: usize = 0x38;
/// Size of our mesh pipeline description (`mesh_pso`).
pub const MESH_DESC_SIZE: usize = crate::mesh_pso::MESH_DESC_SIZE;

/// Offsets inside the graphics description that [`copy_graphics`] follows.
mod graphics_at {
    /// The `D3D12_SHADER_BYTECODE` of vertex, pixel, domain, hull and geometry shader.
    pub const SHADERS: [usize; 5] = [0x08, 0x18, 0x28, 0x38, 0x48];
    /// `D3D12_STREAM_OUTPUT_DESC`: entry array, entry count, stride array, stride count.
    pub const SO_ENTRIES: usize = 0x58;
    pub const SO_ENTRY_COUNT: usize = 0x60;
    pub const SO_STRIDES: usize = 0x68;
    pub const SO_STRIDE_COUNT: usize = 0x70;
    /// `D3D12_INPUT_LAYOUT_DESC`: element array, element count.
    pub const LAYOUT: usize = 0x228;
    pub const LAYOUT_COUNT: usize = 0x230;
    /// `D3D12_CACHED_PIPELINE_STATE`: blob pointer, blob size.
    pub const CACHED_BLOB: usize = 0x278;
}

/// Offsets inside the compute description.
mod compute_at {
    pub const SHADER: usize = 0x08;
    pub const CACHED_BLOB: usize = 0x20;
}

/// Makes the PC pipeline from a description with `(device, desc, riid, out)`. This is what
/// creation did before pipelines were deferred.
pub type Create = unsafe fn(usize, *const c_void, *const GUID, *mut *mut c_void) -> HRESULT;

/// Whether deferring is on: unless `DECANT_D3D12X_LAZY_PSO` is `0`. Read once.
pub fn enabled() -> bool {
    // 0 means not read yet, 1 off, 2 on.
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("DECANT_D3D12X_LAZY_PSO").is_none_or(|value| value != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// A pipeline description copied out of the title's memory. Every pointer in the copy points into
/// storage that the copy owns.
pub struct Owned {
    desc: Vec<u8>,
    shaders: Vec<Arc<[u8]>>,
    // Only kept alive for the pointers in `desc`.
    _buffers: Vec<Vec<u8>>,
}

impl Owned {
    /// The copied description.
    pub fn desc(&self) -> &[u8] {
        &self.desc
    }

    /// The shaders the copied description points to, in the order of the description.
    pub fn shaders(&self) -> &[Arc<[u8]>] {
        &self.shaders
    }
}

/// Shader containers that were seen, by length and identity. Pipelines naming the same bytes share
/// one copy.
type ShaderIndex = HashMap<(usize, u128), Arc<[u8]>>;

static SEEN_SHADERS: LazyLock<Mutex<ShaderIndex>> = LazyLock::new(Default::default);

/// What tells two shader containers apart: the 16 byte checksum that follows the `DXBC` magic, or a
/// hash of all bytes when there is no such container.
fn identity(bytes: &[u8]) -> u128 {
    if let Some(sum) = bytes.strip_prefix(b"DXBC").and_then(|rest| rest.get(..16)) {
        return u128::from_le_bytes(sum.try_into().unwrap_or_default());
    }
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut low = DefaultHasher::new();
    bytes.hash(&mut low);
    let mut high = DefaultHasher::new();
    (bytes.len(), bytes.first(), bytes.last()).hash(&mut high);
    bytes.hash(&mut high);
    (u128::from(high.finish()) << 64) | u128::from(low.finish())
}

/// The shared copy of `bytes`.
fn shared_copy(bytes: &[u8]) -> Arc<[u8]> {
    let key = (bytes.len(), identity(bytes));
    SEEN_SHADERS.lock().unwrap_or_else(|e| e.into_inner()).entry(key).or_insert_with(|| Arc::from(bytes)).clone()
}

fn get_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default())
}

fn get_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default())
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

/// The `len` bytes at address `ptr`. Empty when the address is null or `len` is 0.
///
/// # Safety
/// A non null `ptr` points to `len` readable bytes.
unsafe fn read_region<'a>(ptr: u64, len: usize) -> &'a [u8] {
    if ptr == 0 || len == 0 {
        return &[];
    }
    // SAFETY: the caller promises the region is readable.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }
}

/// The C string at address `ptr`, including its terminator. Empty for a null address.
///
/// # Safety
/// A non null `ptr` points to a NUL terminated string.
unsafe fn read_c_string(ptr: u64) -> Vec<u8> {
    if ptr == 0 {
        return Vec::new();
    }
    // SAFETY: the caller promises a terminated string.
    unsafe { std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char) }.to_bytes_with_nul().to_vec()
}

/// Collects the storage a copied description points into while the pointers are redirected.
#[derive(Default)]
struct Anchors {
    shaders: Vec<Arc<[u8]>>,
    buffers: Vec<Vec<u8>>,
}

impl Anchors {
    /// Redirects the `D3D12_SHADER_BYTECODE` at `at` to a shared copy of the shader.
    ///
    /// # Safety
    /// The pair at `at` is null or describes readable bytecode.
    unsafe fn shader(&mut self, desc: &mut [u8], at: usize) {
        // SAFETY: by the contract above.
        let bytes = unsafe { read_region(get_u64(desc, at), get_u64(desc, at + 8) as usize) };
        if bytes.is_empty() {
            return;
        }
        let copy = shared_copy(bytes);
        put_u64(desc, at, copy.as_ptr() as u64);
        self.shaders.push(copy);
    }

    /// Redirects the array pointer at `at`, which has `count` records of `size` bytes, to a copy.
    /// When `name_at` is given, the C string pointer at that offset of each record is redirected
    /// to a copy too.
    ///
    /// # Safety
    /// The pointer at `at` is null or points to `count` readable records, and their string
    /// pointers are null or point to NUL terminated strings.
    unsafe fn array(&mut self, desc: &mut [u8], at: usize, count: usize, size: usize, name_at: Option<usize>) {
        let source = get_u64(desc, at);
        if source == 0 || count == 0 {
            return;
        }
        // SAFETY: by the contract above.
        let mut records = unsafe { read_region(source, count * size) }.to_vec();
        if let Some(name_at) = name_at {
            for index in 0..count {
                let field = index * size + name_at;
                // SAFETY: by the contract above.
                let text = unsafe { read_c_string(get_u64(&records, field)) };
                if !text.is_empty() {
                    put_u64(&mut records, field, text.as_ptr() as u64);
                    self.buffers.push(text);
                }
            }
        }
        put_u64(desc, at, records.as_ptr() as u64);
        self.buffers.push(records);
    }

    /// Redirects the `{pointer, size}` blob at `at` to a copy.
    ///
    /// # Safety
    /// The pair at `at` is null or describes readable bytes.
    unsafe fn blob(&mut self, desc: &mut [u8], at: usize) {
        // SAFETY: by the contract above.
        let bytes = unsafe { read_region(get_u64(desc, at), get_u64(desc, at + 8) as usize) };
        if bytes.is_empty() {
            return;
        }
        let copy = bytes.to_vec();
        put_u64(desc, at, copy.as_ptr() as u64);
        self.buffers.push(copy);
    }

    fn into_owned(self, desc: Vec<u8>) -> Owned {
        Owned { desc, shaders: self.shaders, _buffers: self.buffers }
    }
}

/// Copies the graphics part of a description: the five shaders, the stream output declaration with
/// its strides, the input layout and the cached blob.
///
/// # Safety
/// As for [`copy_graphics`].
unsafe fn anchor_graphics(anchors: &mut Anchors, desc: &mut [u8]) {
    // SAFETY: by the contract above, each field is read at its public layout offset.
    unsafe {
        for at in graphics_at::SHADERS {
            anchors.shader(desc, at);
        }
        // Stream output declaration entries are 24 bytes with the semantic name at offset 8.
        let entries = get_u32(desc, graphics_at::SO_ENTRY_COUNT) as usize;
        anchors.array(desc, graphics_at::SO_ENTRIES, entries, 24, Some(8));
        let strides = get_u32(desc, graphics_at::SO_STRIDE_COUNT) as usize;
        anchors.array(desc, graphics_at::SO_STRIDES, strides, 4, None);
        // Input layout elements are 32 bytes with the semantic name at offset 0.
        let elements = get_u32(desc, graphics_at::LAYOUT_COUNT) as usize;
        anchors.array(desc, graphics_at::LAYOUT, elements, 32, Some(0));
        anchors.blob(desc, graphics_at::CACHED_BLOB);
    }
}

/// Copies a plain graphics description (`D3D12_GRAPHICS_PIPELINE_STATE_DESC`).
///
/// # Safety
/// `desc` is a description the title passed, and every pointer in it is readable as d3d12.h says.
pub unsafe fn copy_graphics(desc: &[u8; GRAPHICS_DESC_SIZE]) -> Owned {
    let mut copy = desc.to_vec();
    let mut anchors = Anchors::default();
    // SAFETY: by the contract above.
    unsafe { anchor_graphics(&mut anchors, &mut copy) };
    anchors.into_owned(copy)
}

/// Copies our mesh description (`mesh_pso`): the graphics part like [`copy_graphics`], then the
/// amplification and mesh shaders.
///
/// # Safety
/// As for [`copy_graphics`].
pub unsafe fn copy_mesh(desc: &[u8; MESH_DESC_SIZE]) -> Owned {
    let mut copy = desc.to_vec();
    let mut anchors = Anchors::default();
    // SAFETY: by the contract above.
    unsafe {
        anchor_graphics(&mut anchors, &mut copy);
        anchors.shader(&mut copy, crate::mesh_pso::AS_AT);
        anchors.shader(&mut copy, crate::mesh_pso::MS_AT);
    }
    anchors.into_owned(copy)
}

/// Copies a plain compute description (`D3D12_COMPUTE_PIPELINE_STATE_DESC`): the shader and the
/// cached blob.
///
/// # Safety
/// `desc` points to a description the title passed, and every pointer in it is readable.
pub unsafe fn copy_compute(desc: *const c_void) -> Owned {
    // SAFETY: by the contract above.
    let mut copy = unsafe { std::slice::from_raw_parts(desc.cast::<u8>(), COMPUTE_DESC_SIZE) }.to_vec();
    let mut anchors = Anchors::default();
    // SAFETY: by the contract above.
    unsafe {
        anchors.shader(&mut copy, compute_at::SHADER);
        anchors.blob(&mut copy, compute_at::CACHED_BLOB);
    }
    anchors.into_owned(copy)
}

/// Everything needed to make the PC pipeline later.
struct Recipe {
    device: usize,
    create: Create,
    riid: GUID,
    owned: Owned,
}

// SAFETY: the recipe holds addresses (the device proxy, the copied description) that stay valid for
// the life of the process and are only read.
unsafe impl Send for Recipe {}
// SAFETY: as above.
unsafe impl Sync for Recipe {}

/// Our pipeline object. A console title copies the 32 bytes after the object header of a pipeline,
/// so `packet` exists and stays zero.
#[repr(C)]
struct Lazy {
    vtbl: *const VTable,
    refs: AtomicU32,
    number: u32,
    packet: [u32; 8],
    recipe: Recipe,
    /// The PC pipeline once made. 0 means the PC runtime refused it.
    made: OnceLock<usize>,
    /// Key of this pipeline in the precompile queue, if it has one.
    precompile_key: Option<u128>,
    /// A bind has already looked at this pipeline.
    waited: AtomicBool,
    /// A compute bind waited for the full compute wait and gave up. Later binds do not wait until
    /// the pipeline is ready.
    late: AtomicBool,
    /// A job that makes the PC pipeline is queued or running.
    making: AtomicBool,
    pending_logged: AtomicBool,
}
const _: () = assert!(std::mem::offset_of!(Lazy, packet) == 0x10);
// SAFETY: `vtbl` points to the immutable static `VTBL`, every other field of `Lazy` is `Sync`.
unsafe impl Sync for Lazy {}

/// Counters for the log lines: pipelines deferred, made and refused.
static DEFERRED: AtomicU64 = AtomicU64::new(0);
static MADE: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// The wait cap of the async bind mode when `DECANT_D3D12X_ASYNC_BIND_WAIT_MS` is not set.
const DEFAULT_ASYNC_BIND_WAIT_MS: u64 = 20;

/// The wait cap of the async bind mode for the values of `DECANT_D3D12X_ASYNC_BIND` and
/// `DECANT_D3D12X_ASYNC_BIND_WAIT_MS`. `None` when the mode is off: unset, empty, `0`, `off` or
/// `false`. A wait that is not a number gives the default.
fn parse_async_bind(mode: Option<&str>, wait_ms: Option<&str>) -> Option<Duration> {
    let on = mode.map(str::trim).is_some_and(|mode| !mode.is_empty() && !matches!(mode.to_ascii_lowercase().as_str(), "0" | "off" | "false"));
    on.then(|| Duration::from_millis(wait_ms.and_then(|text| text.trim().parse().ok()).unwrap_or(DEFAULT_ASYNC_BIND_WAIT_MS)))
}

/// The wait cap of the async bind mode, `None` when it is off. Read once.
fn async_bind_cap() -> Option<Duration> {
    static CAP: OnceLock<Option<Duration>> = OnceLock::new();
    *CAP.get_or_init(|| {
        let cap = parse_async_bind(
            std::env::var("DECANT_D3D12X_ASYNC_BIND").ok().as_deref(),
            std::env::var("DECANT_D3D12X_ASYNC_BIND_WAIT_MS").ok().as_deref(),
        );
        if let Some(cap) = cap {
            xs_log!("lazy pso: async bind on, a compute bind waits at most {} ms for its pipeline", cap.as_millis());
        }
        cap
    })
}

/// How long a bind may wait when `allowed` is what the usual rules give and `cap` is the async bind
/// cap. Only a compute bind is capped. The second value tells whether the cap shortened the wait.
fn capped_wait(cap: Option<Duration>, is_compute: bool, allowed: Duration) -> (Duration, bool) {
    match cap {
        Some(cap) if is_compute && allowed > cap => (cap, true),
        _ => (allowed, false),
    }
}

/// How long a bind may wait when the async bind mode is `on` and `left` is what is left of the
/// frame budget. Only a compute bind that may wait at all is limited. The second value tells whether
/// the budget shortened the wait.
fn budgeted_wait(on: bool, is_compute: bool, allowed: Duration, left: Duration) -> (Duration, bool) {
    if on && is_compute && allowed > left { (left, true) } else { (allowed, false) }
}

/// Binds the async bind mode left without their pipeline. Every one is counted, the first 16 and
/// then the powers of two are logged.
static ASYNC_LEFT_OUT: AtomicU64 = AtomicU64::new(0);

/// Notes a compute bind that came back without its pipeline after the async bind cap.
fn note_async_left_out(number: u32, waited: Duration) {
    let total = ASYNC_LEFT_OUT.fetch_add(1, Ordering::Relaxed) + 1;
    if total <= 16 || total.is_power_of_two() {
        xs_log!(
            "async bind: pipeline {number} was not ready after {} ms, its bind left work out and the workers go on ({total} so far)",
            waited.as_millis()
        );
    }
}

/// Whether the PC `Create*PipelineState` calls run on worker threads, away from the title's
/// recording threads. `DECANT_D3D12X_ASYNC_PSO=0` turns this off.
fn async_make_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DECANT_D3D12X_ASYNC_PSO").is_none_or(|value| value != "0"))
}

/// The description a job makes the PC pipeline from.
enum MakeSource {
    /// A copy of the description that uses the compiled direct shaders.
    Direct(crate::direct_pso::Patched),
    /// The description as the title created it.
    Original,
}

struct MakeJob {
    lazy: &'static Lazy,
    source: MakeSource,
    service: Option<&'static crate::compile_service::Service>,
}

/// Jobs for the maker workers. Compute jobs are served before graphics and mesh jobs.
#[derive(Default)]
struct MakerQueues {
    compute: VecDeque<MakeJob>,
    graphics: VecDeque<MakeJob>,
}

impl MakerQueues {
    fn next(&mut self) -> Option<MakeJob> {
        self.compute.pop_front().or_else(|| self.graphics.pop_front())
    }
}

static MAKER_JOBS: LazyLock<(Mutex<MakerQueues>, Condvar)> = LazyLock::new(Default::default);
/// Signalled whenever a worker has finished a pipeline.
static MAKER_FINISHED: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Maker workers on a host with `cores` cores: half of them, 2 to 16.
fn default_maker_count(cores: usize) -> usize {
    (cores / 2).clamp(2, 16)
}

fn maker_thread_count() -> usize {
    let configured = std::env::var("DECANT_D3D12X_PSO_MAKERS").ok().and_then(|text| text.parse::<usize>().ok()).filter(|&n| n > 0);
    configured.unwrap_or_else(|| default_maker_count(std::thread::available_parallelism().map_or(4, |cores| cores.get())))
}

/// Starts the maker workers on first use and returns how many run.
fn start_makers() -> usize {
    static RUNNING: OnceLock<usize> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        (0..maker_thread_count())
            .filter(|index| {
                let started = std::thread::Builder::new().name(format!("decant-pso-maker-{index}")).stack_size(16 << 20).spawn(maker_loop);
                if let Err(error) = &started {
                    xs_log!("lazy pso: could not start maker worker {index} ({error})");
                }
                started.is_ok()
            })
            .count()
    })
}

fn maker_loop() {
    // Below normal priority, so the title's recording and audio threads never wait for driver compiles.
    // SAFETY: GetCurrentThread returns the pseudo handle of this thread.
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    let (queues, wake) = &*MAKER_JOBS;
    loop {
        let job = {
            let mut waiting = queues.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(job) = waiting.next() {
                    break job;
                }
                waiting = wake.wait(waiting).unwrap_or_else(|e| e.into_inner());
            }
        };
        run_job(job.lazy, job.source, job.service);
    }
}

/// Hands a job to the workers. Without any worker, the calling thread makes the pipeline.
fn submit(job: MakeJob, is_compute: bool) {
    if start_makers() == 0 {
        let _call = crate::watchdog_hold::pc_call();
        run_job(job.lazy, job.source, job.service);
        return;
    }
    let (queues, wake) = &*MAKER_JOBS;
    {
        let mut waiting = queues.lock().unwrap_or_else(|e| e.into_inner());
        if is_compute { &mut waiting.compute } else { &mut waiting.graphics }.push_back(job);
    }
    wake.notify_one();
}

/// Makes the PC pipeline of `lazy` once and tells the waiters.
fn run_job(lazy: &Lazy, source: MakeSource, service: Option<&'static crate::compile_service::Service>) {
    lazy.made.get_or_init(|| match source {
        MakeSource::Direct(patched) => {
            let direct = make(lazy, patched.desc(), true, service);
            if direct == 0 {
                // The PC runtime refused the compiled shaders: use the title's own.
                return make(lazy, lazy.recipe.owned.desc(), false, service);
            }
            crate::direct_pso::kind_made(direct, &lazy.recipe.owned);
            direct
        }
        MakeSource::Original => {
            crate::direct_pso::pipeline_done_for(service, &lazy.recipe.owned, None);
            make(lazy, lazy.recipe.owned.desc(), false, service)
        }
    });
    let (lock, finished) = &MAKER_FINISHED;
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    finished.notify_all();
}

/// Waits up to `timeout` for the PC pipeline of `lazy`. `None` when it is not there in time.
fn wait_for_made(lazy: &Lazy, timeout: Duration) -> Option<usize> {
    if let Some(&made) = lazy.made.get() {
        return Some(made);
    }
    if timeout.is_zero() {
        return None;
    }
    let begun = Instant::now();
    let (lock, finished) = &MAKER_FINISHED;
    let mut guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    while lazy.made.get().is_none() {
        let left = timeout.saturating_sub(begun.elapsed());
        if left.is_zero() {
            break;
        }
        guard = finished.wait_timeout(guard, left).unwrap_or_else(|e| e.into_inner()).0;
    }
    lazy.made.get().copied()
}

/// Answers a creation with our object for `owned`, and `S_OK`. The PC pipeline is made by
/// `create` on `device` when a list first binds the object. The root signature at offset 0 of the
/// description is referenced until then.
///
/// # Safety
/// `out` is writable, and the root signature in `owned` is null or a live COM object.
pub unsafe fn defer(device: usize, owned: Owned, riid: *const GUID, out: *mut *mut c_void, create: Create) -> HRESULT {
    crate::pipeline_manifest::created(&owned);
    let root = get_u64(owned.desc(), 0) as *mut c_void;
    if !root.is_null() {
        // SAFETY: by the contract above. AddRef is slot 1 of the IUnknown vtable.
        unsafe {
            let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(root as *const *const usize)).add(1));
            add_ref(root);
        }
    }
    let riid = if riid.is_null() {
        GUID { data1: 0, data2: 0, data3: 0, data4: [0; 8] }
    } else {
        // SAFETY: a non null `riid` is the caller's GUID.
        unsafe { riid.read_unaligned() }
    };
    let number = DEFERRED.fetch_add(1, Ordering::Relaxed) as u32;
    let precompile_key = crate::direct_pso::register(&owned, number);
    let object = Box::new(Lazy {
        vtbl: &VTBL,
        refs: AtomicU32::new(1),
        number,
        packet: [0; 8],
        recipe: Recipe { device, create, riid, owned },
        made: OnceLock::new(),
        precompile_key,
        waited: AtomicBool::new(false),
        late: AtomicBool::new(false),
        making: AtomicBool::new(false),
        pending_logged: AtomicBool::new(false),
    });
    // SAFETY: by the contract above. The object is never freed, see `release`.
    unsafe { *out = Box::into_raw(object).cast() };
    0
}

/// Whether `object` is one of our pipelines.
///
/// # Safety
/// `object` is null or a COM object, which starts with a vtable pointer.
pub unsafe fn is_lazy(object: *mut c_void) -> bool {
    // SAFETY: by the contract above.
    !object.is_null() && std::ptr::eq(unsafe { *(object as *const *const VTable) }, &VTBL)
}

/// The PC pipeline behind `object`, if it is one of ours. The pipeline is queued for the workers
/// when it is not made yet. The answer is null when the PC runtime refused it, or when it is still
/// being made after the time this bind may wait. `None` for any other object.
///
/// # Safety
/// As for [`is_lazy`].
pub unsafe fn resolve(object: *mut c_void) -> Option<*mut c_void> {
    // SAFETY: by the contract above.
    if !unsafe { is_lazy(object) } {
        return None;
    }
    // SAFETY: the vtable is ours, so this is a `Lazy`, and those are never freed.
    let lazy: &'static Lazy = unsafe { &*object.cast::<Lazy>() };
    crate::direct_pso::note_bind();
    if let Some(&made) = lazy.made.get() {
        return Some(made as *mut c_void);
    }
    let is_compute = lazy.recipe.owned.desc().len() == COMPUTE_DESC_SIZE;
    let first_bind = !lazy.waited.swap(true, Ordering::Relaxed);
    let allowed = crate::direct_pso::bind_wait(is_compute, first_bind, &lazy.late);
    let cap = async_bind_cap();
    let (allowed, capped) = capped_wait(cap, is_compute, allowed);
    let (allowed, over_budget) = budgeted_wait(cap.is_some(), is_compute, allowed, crate::direct_pso::async_frame_left());
    let capped = capped || over_budget;
    let begun = Instant::now();
    let made = resolve_within(object as usize, lazy, is_compute, allowed);
    if cap.is_some() && is_compute && !allowed.is_zero() {
        crate::direct_pso::spend_async_wait(begun.elapsed());
    }
    if capped && made.is_null() && lazy.made.get().is_none() {
        note_async_left_out(lazy.number, begun.elapsed());
    }
    Some(made)
}

/// Makes the pipeline of `lazy` ready, waiting at most `allowed` for it. Null when it is not
/// ready in time.
fn resolve_within(object: usize, lazy: &'static Lazy, is_compute: bool, allowed: Duration) -> *mut c_void {
    let begun = Instant::now();
    let waits = is_compute && !allowed.is_zero();
    let _hold = waits.then(crate::watchdog_hold::hold_bind);
    // Waits of graphics and mesh binds come out of the frame budget, compute waits do not.
    let charge_frame = || {
        if !is_compute {
            crate::direct_pso::spend_frame_wait(begun.elapsed());
        }
    };
    let not_in_time = || {
        if waits {
            lazy.late.store(true, Ordering::Release);
        } else {
            charge_frame();
        }
        crate::direct_pso::note_pending(object, &lazy.pending_logged);
        std::ptr::null_mut::<c_void>()
    };
    let arrived = |made: Option<usize>| match made {
        Some(pipeline) => {
            charge_frame();
            pipeline as *mut c_void
        }
        None => not_in_time(),
    };

    if lazy.making.load(Ordering::Acquire) {
        return arrived(wait_for_made(lazy, allowed));
    }

    let service = crate::direct_pso::current_service();
    let source = match crate::direct_pso::choose_with_budget(object, &lazy.recipe.owned, lazy.precompile_key, allowed) {
        crate::direct_pso::Choice::Pending => return not_in_time(),
        crate::direct_pso::Choice::Direct(patched) => MakeSource::Direct(patched),
        crate::direct_pso::Choice::Original => MakeSource::Original,
    };

    if !async_make_enabled() {
        let _call = crate::watchdog_hold::pc_call();
        run_job(lazy, source, service);
        charge_frame();
        return lazy.made.get().copied().unwrap_or(0) as *mut c_void;
    }

    if !lazy.making.swap(true, Ordering::AcqRel) {
        submit(MakeJob { lazy, source, service }, is_compute);
    }
    arrived(wait_for_made(lazy, allowed.saturating_sub(begun.elapsed())))
}

/// Calls the creation path for `lazy` with `desc`. Returns the PC pipeline, or 0 when the PC
/// runtime refused it or only gave a stand in.
fn make(lazy: &Lazy, desc: &[u8], direct: bool, service: Option<&'static crate::compile_service::Service>) -> usize {
    let recipe = &lazy.recipe;
    let begun = Instant::now();
    let mut out: *mut c_void = std::ptr::null_mut();
    // SAFETY: the recipe holds the device and a description that the creation path took the same
    // way when the title called it, or a copy of it with other shaders.
    let hr = unsafe { (recipe.create)(recipe.device, desc.as_ptr().cast(), &recipe.riid, &mut out) };
    // SAFETY: `out` is null or the object the creation path wrote.
    let stand_in = unsafe { crate::pso_diag::standin_id(out) }.is_some();
    let made = if hr >= 0 && !out.is_null() && !stand_in { out as usize } else { 0 };
    let took = begun.elapsed();
    if direct {
        crate::direct_pso::pipeline_done_for(service, &recipe.owned, Some((made != 0, took)));
    }
    let ms = took.as_millis();
    let (nth, outcome) = if made != 0 { (MADE.fetch_add(1, Ordering::Relaxed), "made") } else { (REFUSED.fetch_add(1, Ordering::Relaxed), "refused") };
    if nth < 32 || ms >= 1000 || nth.is_power_of_two() {
        let how = if direct { " with direct shaders" } else { "" };
        xs_log!(
            "pipeline {} {outcome}{how} on first bind in {ms} ms, hr=0x{:08X} ({} deferred, {} made, {} refused so far)",
            lazy.number,
            hr as u32,
            DEFERRED.load(Ordering::Relaxed),
            MADE.load(Ordering::Relaxed),
            REFUSED.load(Ordering::Relaxed)
        );
    }
    made
}

// The COM object. Slots 0 to 2 are IUnknown, 3 to 6 ID3D12Object (GetPrivateData, SetPrivateData,
// SetPrivateDataInterface, SetName), 7 is ID3D12DeviceChild::GetDevice and 8 is
// ID3D12PipelineState::GetCachedBlob. The table has 16 slots so a stray call stays inside it.

type Method = unsafe extern "system" fn(*mut c_void, usize, usize, usize) -> i32;

#[repr(C)]
struct VTable([Method; 16]);

const E_NOINTERFACE: i32 = 0x8000_4002_u32 as i32;
const E_NOTIMPL: i32 = 0x8000_4001_u32 as i32;
const DXGI_ERROR_NOT_FOUND: i32 = 0x887A_0002_u32 as i32;

/// Hands out `this` for any interface: a title only asks a pipeline for its own kinds.
unsafe extern "system" fn query_interface(this: *mut c_void, _riid: usize, out: usize, _: usize) -> i32 {
    if out == 0 {
        return E_NOINTERFACE;
    }
    // SAFETY: `this` is one of ours and `out` is the caller's writable pointer.
    unsafe {
        (*this.cast::<Lazy>()).refs.fetch_add(1, Ordering::Relaxed);
        *(out as *mut *mut c_void) = this;
    }
    0
}

unsafe extern "system" fn add_ref(this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    // SAFETY: `this` is one of ours.
    unsafe { (*this.cast::<Lazy>()).refs.fetch_add(1, Ordering::Relaxed) as i32 + 1 }
}

/// Counts down but never frees: a list can still name the object, and the shaders are shared.
unsafe extern "system" fn release(this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    // SAFETY: `this` is one of ours.
    unsafe { ((*this.cast::<Lazy>()).refs.fetch_sub(1, Ordering::Relaxed) as i32 - 1).max(0) }
}

unsafe extern "system" fn get_private_data(_this: *mut c_void, _: usize, size: usize, _: usize) -> i32 {
    if size != 0 {
        // SAFETY: `size` is the caller's in and out size.
        unsafe { *(size as *mut u32) = 0 };
    }
    DXGI_ERROR_NOT_FOUND
}

/// SetPrivateData, SetPrivateDataInterface and SetName: accepted and forgotten.
unsafe extern "system" fn accept_and_forget(_this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    0
}

unsafe extern "system" fn get_device(_this: *mut c_void, _riid: usize, out: usize, _: usize) -> i32 {
    if out != 0 {
        // SAFETY: `out` is the caller's writable pointer.
        unsafe { *(out as *mut *mut c_void) = std::ptr::null_mut() };
    }
    E_NOINTERFACE
}

unsafe extern "system" fn get_cached_blob(_this: *mut c_void, out: usize, _: usize, _: usize) -> i32 {
    if out != 0 {
        // SAFETY: `out` is the caller's writable pointer.
        unsafe { *(out as *mut *mut c_void) = std::ptr::null_mut() };
    }
    E_NOTIMPL
}

unsafe extern "system" fn unsupported(_this: *mut c_void, _: usize, _: usize, _: usize) -> i32 {
    E_NOTIMPL
}

static VTBL: VTable = VTable([
    query_interface,
    add_ref,
    release,
    get_private_data,
    accept_and_forget,
    accept_and_forget,
    accept_and_forget,
    get_device,
    get_cached_blob,
    unsupported,
    unsupported,
    unsupported,
    unsupported,
    unsupported,
    unsupported,
    unsupported,
]);

#[cfg(test)]
mod tests {
    use super::*;

    /// A compute description whose shader is `shader`.
    fn compute_desc(shader: &[u8]) -> [u8; COMPUTE_DESC_SIZE] {
        let mut desc = [0u8; COMPUTE_DESC_SIZE];
        put_u64(&mut desc, compute_at::SHADER, shader.as_ptr() as u64);
        put_u64(&mut desc, compute_at::SHADER + 8, shader.len() as u64);
        desc
    }

    /// Defers a compute pipeline on `device` and returns our object.
    fn defer_compute(device: usize, shader: &[u8], create: Create) -> *mut c_void {
        let desc = compute_desc(shader);
        // SAFETY: the description and its shader are live for the call.
        let owned = unsafe { copy_compute(desc.as_ptr().cast()) };
        let mut out = std::ptr::null_mut();
        // SAFETY: `out` is writable and the description has no root signature.
        assert_eq!(unsafe { defer(device, owned, std::ptr::null(), &mut out, create) }, 0);
        out
    }

    /// The shader a creation was asked to build.
    fn shader_of(desc: *const c_void) -> Vec<u8> {
        // SAFETY: the tests pass compute descriptions whose shader pointer is live.
        unsafe {
            let desc = std::slice::from_raw_parts(desc.cast::<u8>(), COMPUTE_DESC_SIZE);
            read_region(get_u64(desc, compute_at::SHADER), get_u64(desc, compute_at::SHADER + 8) as usize).to_vec()
        }
    }

    static COUNTING_CALLS: AtomicU32 = AtomicU32::new(0);

    unsafe fn counting_create(device: usize, desc: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        COUNTING_CALLS.fetch_add(1, Ordering::Relaxed);
        if shader_of(desc) != b"DXBC0123456789abcdefcounted" {
            return E_NOTIMPL;
        }
        // SAFETY: `out` is the recipe's pointer.
        unsafe { *out = device as *mut c_void };
        0
    }

    #[test]
    fn nothing_is_made_at_creation_and_the_pipeline_is_made_once_from_the_copy() {
        let mut shader = b"DXBC0123456789abcdefcounted".to_vec();
        let object = defer_compute(0x1234_5670, &shader, counting_create);
        // The title may reuse its memory as soon as creation returned.
        shader.fill(0);
        assert_eq!(COUNTING_CALLS.load(Ordering::Relaxed), 0);
        // SAFETY: `object` is our pipeline.
        unsafe {
            assert!(is_lazy(object));
            assert_eq!(*(object.cast::<u8>().add(0x10).cast::<[u32; 8]>()), [0; 8], "the packet after the header is zero");
            assert_eq!(resolve(object), Some(0x1234_5670 as *mut c_void));
            assert_eq!(resolve(object), Some(0x1234_5670 as *mut c_void));
        }
        assert_eq!(COUNTING_CALLS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn objects_of_other_makers_are_not_ours() {
        let foreign = Box::into_raw(Box::new([0usize; 2])).cast::<c_void>();
        // SAFETY: `foreign` starts with a null vtable pointer.
        unsafe {
            assert!(!is_lazy(foreign));
            assert_eq!(resolve(foreign), None);
            assert!(!is_lazy(std::ptr::null_mut()));
        }
    }

    #[test]
    fn the_com_object_answers_the_basic_calls() {
        let object = defer_compute(0x77, b"DXBC0123456789abcdefcounted", counting_create);
        // SAFETY: `object` is our pipeline and the arguments are valid for each slot.
        unsafe {
            let table = &*(*object.cast::<*const VTable>());
            assert_eq!((table.0[1])(object, 0, 0, 0), 2, "AddRef");
            assert_eq!((table.0[2])(object, 0, 0, 0), 1, "Release");
            let mut other: *mut c_void = std::ptr::null_mut();
            assert_eq!((table.0[0])(object, 0, &mut other as *mut _ as usize, 0), 0, "QueryInterface");
            assert_eq!(other, object);
            assert_eq!((table.0[2])(object, 0, 0, 0), 1);
            assert_eq!((table.0[0])(object, 0, 0, 0), E_NOINTERFACE);
            assert_eq!((table.0[8])(object, 0, 0, 0), E_NOTIMPL, "GetCachedBlob");
        }
    }

    #[test]
    fn a_graphics_copy_is_independent_of_the_title_memory() {
        let vertex = b"DXBCvvvvvvvvvvvvvvvvvertex".to_vec();
        let semantic = c"POSITION";
        let mut element = [0u8; 32];
        put_u64(&mut element, 0, semantic.as_ptr() as u64);
        element[12..16].copy_from_slice(&6u32.to_le_bytes());
        let mut desc = [0u8; GRAPHICS_DESC_SIZE];
        put_u64(&mut desc, graphics_at::SHADERS[0], vertex.as_ptr() as u64);
        put_u64(&mut desc, graphics_at::SHADERS[0] + 8, vertex.len() as u64);
        put_u64(&mut desc, graphics_at::LAYOUT, element.as_ptr() as u64);
        desc[graphics_at::LAYOUT_COUNT..graphics_at::LAYOUT_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        desc[0x240..0x244].copy_from_slice(&1u32.to_le_bytes());
        // SAFETY: every pointer in `desc` is live.
        let first = unsafe { copy_graphics(&desc) };
        let copy = first.desc();
        assert_ne!(get_u64(copy, 0x08), vertex.as_ptr() as u64);
        assert_eq!(first.shaders().len(), 1);
        // SAFETY: the copy's pointers point into storage that `first` owns.
        unsafe {
            assert_eq!(read_region(get_u64(copy, 0x08), vertex.len()), &vertex[..]);
            let layout = read_region(get_u64(copy, graphics_at::LAYOUT), 32);
            assert_ne!(get_u64(layout, 0), semantic.as_ptr() as u64, "the semantic name is a copy too");
            assert_eq!(read_c_string(get_u64(layout, 0)), b"POSITION\0");
            assert_eq!(get_u32(layout, 12), 6);
        }
        assert_eq!(get_u32(copy, 0x240), 1, "other fields are carried over");
        // SAFETY: as above.
        let second = unsafe { copy_graphics(&desc) };
        assert_eq!(get_u64(second.desc(), 0x08), get_u64(copy, 0x08), "equal containers are shared");
    }

    #[test]
    fn stream_output_and_cached_blob_are_copied() {
        let blob = b"cached pipeline blob".to_vec();
        let name = c"SV_Position";
        let mut entry = [0u8; 24];
        put_u64(&mut entry, 8, name.as_ptr() as u64);
        let strides = [16u32];
        let mut desc = [0u8; GRAPHICS_DESC_SIZE];
        put_u64(&mut desc, graphics_at::SO_ENTRIES, entry.as_ptr() as u64);
        desc[graphics_at::SO_ENTRY_COUNT..graphics_at::SO_ENTRY_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        put_u64(&mut desc, graphics_at::SO_STRIDES, strides.as_ptr() as u64);
        desc[graphics_at::SO_STRIDE_COUNT..graphics_at::SO_STRIDE_COUNT + 4].copy_from_slice(&1u32.to_le_bytes());
        put_u64(&mut desc, graphics_at::CACHED_BLOB, blob.as_ptr() as u64);
        put_u64(&mut desc, graphics_at::CACHED_BLOB + 8, blob.len() as u64);
        // SAFETY: every pointer in `desc` is live.
        let owned = unsafe { copy_graphics(&desc) };
        let copy = owned.desc();
        // SAFETY: the copy's pointers point into storage that `owned` keeps.
        unsafe {
            let entries = read_region(get_u64(copy, graphics_at::SO_ENTRIES), 24);
            assert_eq!(read_c_string(get_u64(entries, 8)), b"SV_Position\0");
            assert_ne!(get_u64(entries, 8), name.as_ptr() as u64);
            assert_eq!(read_region(get_u64(copy, graphics_at::SO_STRIDES), 4), 16u32.to_le_bytes());
            assert_eq!(read_region(get_u64(copy, graphics_at::CACHED_BLOB), blob.len()), &blob[..]);
        }
        assert_ne!(get_u64(copy, graphics_at::CACHED_BLOB), blob.as_ptr() as u64);
    }

    #[test]
    fn a_dxbc_container_is_identified_by_its_checksum() {
        let a = b"DXBC0123456789abcdefAAAA";
        let b = b"DXBC0123456789abcdefBBBB";
        let c = b"DXBC0123456789abcdeXBBBB";
        assert_eq!(identity(a), identity(b), "same checksum, same container");
        assert_ne!(identity(a), identity(c));
        assert_ne!(identity(b"plain bytes one"), identity(b"plain bytes two"));
    }

    static GATE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

    unsafe fn gated_create(device: usize, _desc: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        let mut open = GATE.0.lock().unwrap_or_else(|e| e.into_inner());
        while !*open {
            open = GATE.1.wait_timeout(open, Duration::from_secs(5)).unwrap_or_else(|e| e.into_inner()).0;
        }
        // SAFETY: `out` is the recipe's pointer.
        unsafe { *out = device as *mut c_void };
        0
    }

    #[test]
    fn a_slow_compute_pipeline_goes_late_and_arrives_from_a_worker() {
        let object = defer_compute(0x5100_0001, b"DXBC0123456789abcdefslowcs", gated_create);
        // SAFETY: `object` is our `Lazy`.
        let lazy: &'static Lazy = unsafe { &*object.cast::<Lazy>() };

        let begun = Instant::now();
        assert!(resolve_within(object as usize, lazy, true, Duration::from_millis(25)).is_null());
        assert!(begun.elapsed() < Duration::from_millis(400), "the bind did not wait for the gate");
        assert!(lazy.making.load(Ordering::Acquire));
        assert!(lazy.late.load(Ordering::Acquire));

        let again = Instant::now();
        // SAFETY: `object` is our pipeline.
        assert_eq!(unsafe { resolve(object) }, Some(std::ptr::null_mut()));
        assert!(again.elapsed() < Duration::from_millis(50), "a late pipeline does not wait again");

        *GATE.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        GATE.1.notify_all();
        let deadline = Instant::now() + Duration::from_secs(5);
        // SAFETY: `object` is our pipeline.
        while unsafe { resolve(object) } != Some(0x5100_0001 as *mut c_void) {
            assert!(Instant::now() < deadline, "the worker never finished");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn async_bind_is_off_unless_asked_for_and_reads_its_wait() {
        for off in [None, Some(""), Some("  "), Some("0"), Some("off"), Some("OFF"), Some("false")] {
            assert_eq!(parse_async_bind(off, Some("5")), None, "{off:?}");
        }
        let default = Some(Duration::from_millis(DEFAULT_ASYNC_BIND_WAIT_MS));
        assert_eq!(parse_async_bind(Some("1"), None), default);
        assert_eq!(parse_async_bind(Some("on"), Some("not a number")), default);
        assert_eq!(parse_async_bind(Some("1"), Some(" 75 ")), Some(Duration::from_millis(75)));
        assert_eq!(parse_async_bind(Some("1"), Some("0")), Some(Duration::ZERO), "0 never waits");
    }

    #[test]
    fn the_async_cap_shortens_only_a_compute_wait_that_is_longer() {
        let cap = Some(Duration::from_millis(20));
        let long = Duration::from_millis(5000);
        assert_eq!(capped_wait(cap, true, long), (Duration::from_millis(20), true));
        assert_eq!(capped_wait(cap, true, Duration::from_millis(20)), (Duration::from_millis(20), false), "equal is not cut");
        assert_eq!(capped_wait(cap, true, Duration::ZERO), (Duration::ZERO, false), "a late pipeline stays at zero");
        assert_eq!(capped_wait(cap, false, long), (long, false), "graphics and mesh binds keep their own budget");
        assert_eq!(capped_wait(None, true, long), (long, false), "off changes nothing");
        assert_eq!(capped_wait(Some(Duration::ZERO), true, long), (Duration::ZERO, true));
    }

    static SLOW_GATE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

    unsafe fn slow_create(device: usize, _desc: *const c_void, _riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
        let mut open = SLOW_GATE.0.lock().unwrap_or_else(|e| e.into_inner());
        while !*open {
            open = SLOW_GATE.1.wait_timeout(open, Duration::from_secs(5)).unwrap_or_else(|e| e.into_inner()).0;
        }
        // SAFETY: `out` is the recipe's pointer.
        unsafe { *out = device as *mut c_void };
        0
    }

    #[test]
    fn a_capped_bind_returns_in_time_and_the_next_bind_takes_the_finished_pipeline() {
        let object = defer_compute(0x5100_0002, b"DXBC0123456789abcdefcapped", slow_create);
        // SAFETY: `object` is our `Lazy`.
        let lazy: &'static Lazy = unsafe { &*object.cast::<Lazy>() };
        // What the usual rules allow for a first compute bind, cut by a 15 ms cap.
        let (allowed, capped) = capped_wait(Some(Duration::from_millis(15)), true, Duration::from_secs(5));
        assert!(capped);
        let begun = Instant::now();
        assert!(resolve_within(object as usize, lazy, true, allowed).is_null());
        assert!(begun.elapsed() < Duration::from_millis(1000), "the bind did not wait for the slow compile");
        assert!(lazy.late.load(Ordering::Acquire), "the pipeline is late, later binds do not wait");
        assert!(lazy.making.load(Ordering::Acquire), "a worker has the job");
        assert!(lazy.made.get().is_none());

        *SLOW_GATE.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        SLOW_GATE.1.notify_all();
        let deadline = Instant::now() + Duration::from_secs(5);
        while lazy.made.get().is_none() {
            assert!(Instant::now() < deadline, "the worker never finished");
            std::thread::sleep(Duration::from_millis(5));
        }
        let (allowed, _) = capped_wait(Some(Duration::from_millis(15)), true, Duration::ZERO);
        assert_eq!(resolve_within(object as usize, lazy, true, allowed), 0x5100_0002 as *mut c_void);
    }

    #[test]
    fn maker_workers_scale_with_the_cores() {
        assert_eq!(default_maker_count(1), 2, "never below 2");
        assert_eq!(default_maker_count(8), 4);
        assert_eq!(default_maker_count(24), 12);
        assert_eq!(default_maker_count(1000), 16, "never above 16");
    }

    #[test]
    fn the_async_frame_budget_limits_only_a_cold_compute_wait() {
        let ms = Duration::from_millis;
        assert_eq!(budgeted_wait(true, true, ms(20), ms(16)), (ms(16), true));
        assert_eq!(budgeted_wait(true, true, ms(20), ms(5)), (ms(5), true));
        assert_eq!(budgeted_wait(true, true, ms(20), Duration::ZERO), (Duration::ZERO, true), "a spent budget does not wait");
        assert_eq!(budgeted_wait(true, true, ms(10), ms(16)), (ms(10), false), "a wait inside the budget is kept");
        assert_eq!(budgeted_wait(true, true, Duration::ZERO, Duration::ZERO), (Duration::ZERO, false), "a late pipeline stays at zero");
        assert_eq!(budgeted_wait(true, false, ms(50), Duration::ZERO), (ms(50), false), "graphics and mesh keep their own budget");
        assert_eq!(budgeted_wait(false, true, ms(5000), Duration::ZERO), (ms(5000), false), "off changes nothing");
    }

    #[test]
    fn compute_jobs_are_served_before_graphics_jobs() {
        let shader = b"DXBC0123456789abcdefqueued";
        let compute = defer_compute(1, shader, counting_create);
        let graphics = defer_compute(2, shader, counting_create);
        // SAFETY: both are our `Lazy` objects, which are never freed.
        let (compute, graphics): (&'static Lazy, &'static Lazy) = unsafe { (&*compute.cast::<Lazy>(), &*graphics.cast::<Lazy>()) };
        let mut queues = MakerQueues::default();
        queues.graphics.push_back(MakeJob { lazy: graphics, source: MakeSource::Original, service: None });
        queues.compute.push_back(MakeJob { lazy: compute, source: MakeSource::Original, service: None });
        assert!(std::ptr::eq(queues.next().unwrap().lazy, compute));
        assert!(std::ptr::eq(queues.next().unwrap().lazy, graphics));
        assert!(queues.next().is_none());
    }
}
