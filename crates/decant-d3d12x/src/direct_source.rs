//! The translator of the direct shader path: a [`ShaderSource`] over
//! `decant_shader::translate`, the clean room translator from console GFX10 code to HLSL.
//!
//! Per pipeline:
//! * Every stage is read with `decant_shader::container::program` and all stages go to one
//!   `translate` call, because the stages are linked there (the pixel input struct is made from
//!   the output signature of the stage before it).
//! * The root signature is the title's own part of the PC one ([`RootLayout::without_layer_params`]),
//!   parameter by parameter, with the table offsets resolved by `rts0`.
//! * The pipeline is only served when its PC root signature has the user data mirror and both
//!   heap flags (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, `SAMPLER_HEAP_DIRECTLY_INDEXED`): translated
//!   shaders read every table descriptor from the heap, at the index the mirror holds plus the
//!   slot the machine code names (a register of a range would read the PC slot of the range,
//!   which is another one when the table has several ranges). `list_mirror` writes the mirror
//!   before each draw.
//! * The heap map and the view shadow (root shader resource views `t0` and `t1` of the mirror's
//!   space) are named in the options when the PC root signature has them. Until the layer
//!   appends them, a program that needs one is refused by the translator.
//! * Immediate samplers get their absolute heap index from `sampler_registry`
//!   ([`RegistryHost`]): `HEAP_BASE` plus the registry index, the tail every sampler heap has.
//! * What a translated stage needs is checked against what the layer provides
//!   ([`check_stage`]), so a pipeline the PC runtime would refuse is refused here with a name.
//!
//! One stage that does not translate refuses the whole pipeline, which then takes the title's
//! shaders and the DXIL override files as before. A refusal carries the translator's stable key
//! (`op:<name>`, `resource:<what>`, `stage:<what>`, ...) as its reason, or an `adapter:<what>` key
//! of this file, so the counters in the layer's log read like the census of the translator.
//!
//! The translation starts when the pipeline is first bound and the binding thread waits for it:
//! a refused pipeline is then made from the title's shaders at once and never bound as missing,
//! and the DXIL cache is keyed by the HLSL text ([`ShaderSource::stage_keys`]), so a cached
//! pipeline is ready at its first bind and a changed translator never meets stale DXIL. The work
//! itself is done on a thread of its own with a large stack, and a panic of the translator is a
//! refusal. The binding thread waits `DECANT_D3D12X_DIRECT_TRANSLATE_WAIT_MS` (default 2000) at most:
//! past that the pipeline is left out for now ([`Refusal::PENDING`], bound as missing), the
//! translation goes on, and a later bind takes it. The start and the end of every translation are
//! logged with the stages' hashes, so a slow one is named while it runs. The last translations are
//! kept, so the compile worker does not translate again.
//!
//! `DECANT_D3D12X_DIRECT_DUMP=<dir>` gets, per translated stage, `<container hash>.<profile>.hlsl`
//! with the instruction listing as comments, and per refused pipeline `<container hash>.refused.txt`.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use decant_shader::api as sd;

use crate::direct_cache::Fnv128;
use crate::rts0::{self, RootLayout};
use crate::shader_source::{Pipeline, Refusal, ShaderSource, StageHlsl, StageKind};

/// Bumped when this file changes what the translator is asked or what a pipeline is refused for.
pub const ADAPTER_VERSION: u32 = 2;

/// The `t` register of the heap map in the mirror's space (`decant_shader::api::MirrorAbi`).
pub const HEAP_MAP_REGISTER: u32 = 0;
/// The `t` register of the view shadow in the mirror's space.
pub const VIEW_SHADOW_REGISTER: u32 = 1;

/// `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`.
const RESOURCE_HEAP_FLAG: u32 = 0x400;
/// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED`.
const SAMPLER_HEAP_FLAG: u32 = 0x800;

/// The stack of the translating thread. The translator walks expression trees by recursion, and
/// a title thread may have a small stack.
const TRANSLATE_STACK: usize = 32 << 20;
/// How many translations are kept for the compile worker.
const KEPT: usize = 64;
/// A translation slower than this is logged, as it holds a title thread.
const SLOW_MS: u128 = 50;

/// The sampler lookup of the layer: the registry index of the words, as the absolute index in
/// the sampler heap (`sampler_registry::HEAP_BASE` plus the index). A sampler with
/// `FORCE_UNNORMALIZED` gets its index like any other: D3D12 has no such sampler, the registry
/// makes it as a normalised one and the translated shader divides the texel coordinates by the
/// size of the texture itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegistryHost;

impl sd::Host for RegistryHost {
    fn sampler_index(&self, words: [u32; 4]) -> Option<u32> {
        crate::sampler_registry::index_of(words).map(|entry| crate::sampler_registry::HEAP_BASE + entry.index)
    }
}

fn class(c: rts0::RegisterClass) -> sd::RegisterClass {
    match c {
        rts0::RegisterClass::Srv => sd::RegisterClass::Srv,
        rts0::RegisterClass::Uav => sd::RegisterClass::Uav,
        rts0::RegisterClass::Cbv => sd::RegisterClass::Cbv,
        rts0::RegisterClass::Sampler => sd::RegisterClass::Sampler,
    }
}

/// The root signature the translator gets: the title's parameters of `layout` in order (the
/// layer's own appended ones left out), the static samplers, and the flags of the PC root signature.
pub fn root_signature(layout: &RootLayout) -> sd::RootSignature {
    let title = layout.without_layer_params();
    let params = title
        .params
        .iter()
        .map(|p| sd::RootParam {
            kind: match &p.kind {
                rts0::RootParamKind::Table(ranges) => sd::RootParamKind::Table(
                    ranges.iter().map(|r| sd::RootRange { class: class(r.class), count: r.count, base_register: r.base, space: r.space, offset: r.offset }).collect(),
                ),
                rts0::RootParamKind::Constants { register, space, count } => sd::RootParamKind::Constants { register: *register, space: *space, count: *count },
                rts0::RootParamKind::Descriptor { class: c, register, space } => sd::RootParamKind::Descriptor { class: class(*c), register: *register, space: *space },
            },
            visibility: p.visibility,
        })
        .collect();
    let static_samplers = title.static_samplers.iter().map(|s| sd::StaticSampler { register: s.register, space: s.space, visibility: s.visibility }).collect();
    sd::RootSignature { params, static_samplers, flags: layout.flags }
}

/// The widest wave of the PC device, from `D3D12_OPTIONS1.WaveLaneCountMax` at device creation
/// ([`note_device_waves`]); 0 until then.
static HOST_WAVE_LANES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Reads and logs the widest wave of the PC device (`D3D12_OPTIONS1.WaveLaneCountMax`) and notes it
/// for the options and the cache identity of the pipelines translated afterwards: a GPU whose waves
/// have at most 32 lanes refuses `[WaveSize(64)]` (RTX 4080, debug layer message 622), so console
/// 64 lane programs are emulated for it. Call it before the compile service is made.
#[cfg(windows)]
#[allow(unsafe_code)]
pub fn note_device_waves(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1};
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the structure is the one of this feature and its size is given.
    let read = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut options).cast(), size_of_val(&options) as u32) };
    let (min, max) = if read.is_ok() { (options.WaveLaneCountMin, options.WaveLaneCountMax) } else { (0, 0) };
    HOST_WAVE_LANES.store(max, std::sync::atomic::Ordering::Relaxed);
    crate::log::xs_log!(
        "PC device: waves of {min} to {max} lanes (D3D12_OPTIONS1 read: {}){}",
        read.is_ok(),
        if host_wave_emulated() { ": console 64 lane programs are emulated through group shared memory" } else { "" }
    );
}

/// Whether console 64 lane programs are emulated: the device is known and its waves have fewer
/// than 64 lanes (an NVIDIA GPU: 32). `DECANT_D3D12X_WAVE_EMULATION=0` keeps the native form
/// (`[WaveSize(64)]`, which such a device refuses), for comparison only.
pub fn host_wave_emulated() -> bool {
    wave_mode_for(HOST_WAVE_LANES.load(std::sync::atomic::Ordering::Relaxed), std::env::var_os("DECANT_D3D12X_WAVE_EMULATION").is_some_and(|v| v == "0")) == sd::WaveMode::Emulated
}

/// The wave mode for a device whose widest wave has `lanes` lanes (0: unknown), `off` when the
/// emulation is switched off.
fn wave_mode_for(lanes: u32, off: bool) -> sd::WaveMode {
    if lanes != 0 && lanes < 64 && !off { sd::WaveMode::Emulated } else { sd::WaveMode::Scalar }
}

/// The options for a pipeline with the root signature `layout`: the mirror when the PC root
/// signature has the mirror parameter, with the heap map and the view shadow when it has those
/// root views too, one lane per invocation (a 64 lane wave emulated on a device with smaller
/// waves, [`host_wave_emulated`]), shader model 6.6.
pub fn options(layout: &RootLayout) -> sd::Options {
    let view = |register: u32| layout.descriptor_index(rts0::RegisterClass::Srv, register, rts0::MIRROR_SPACE).map(|_| register);
    sd::Options {
        mirror: layout.mirror_index().map(|_| sd::MirrorAbi {
            register: rts0::MIRROR_REGISTER,
            space: rts0::MIRROR_SPACE,
            heap_map: view(HEAP_MAP_REGISTER),
            view_shadow: view(VIEW_SHADOW_REGISTER),
        }),
        wave: if host_wave_emulated() { sd::WaveMode::Emulated } else { sd::WaveMode::Scalar },
        shader_model_minor: 6,
        listing: false,
    }
}

/// The translator's stage of a stage of the layer. A mesh pipeline comes as its mesh and pixel
/// stage (and an amplification stage, which the translator refuses by its own key).
fn stage_of(kind: StageKind) -> sd::Stage {
    match kind {
        StageKind::Vertex => sd::Stage::Vertex,
        StageKind::Pixel => sd::Stage::Pixel,
        StageKind::Domain => sd::Stage::Domain,
        StageKind::Hull => sd::Stage::Hull,
        StageKind::Geometry => sd::Stage::Geometry,
        StageKind::Compute => sd::Stage::Compute,
        StageKind::Amplification => sd::Stage::Amplification,
        StageKind::Mesh => sd::Stage::Mesh,
    }
}

/// Whether the title's root signature has `register` of `class` and `space` for a stage with the
/// shader visibility `visibility`: in a table range, as a root descriptor, as root constants or as
/// a static sampler.
fn covered(root: &sd::RootSignature, visibility: u32, class: sd::RegisterClass, register: u32, space: u32) -> bool {
    let seen = |v: u32| v == 0 || v == visibility;
    let in_param = |p: &sd::RootParam| match &p.kind {
        sd::RootParamKind::Table(ranges) => ranges.iter().any(|r| r.class == class && r.space == space && register >= r.base_register && register - r.base_register < r.count),
        sd::RootParamKind::Constants { register: r, space: s, .. } => class == sd::RegisterClass::Cbv && (*r, *s) == (register, space),
        sd::RootParamKind::Descriptor { class: c, register: r, space: s } => (*c, *r, *s) == (class, register, space),
    };
    root.params.iter().any(|p| seen(p.visibility) && in_param(p))
        || (class == sd::RegisterClass::Sampler && root.static_samplers.iter().any(|s| seen(s.visibility) && (s.register, s.space) == (register, space)))
}

/// Checks what the translated `stage` needs against what the layer provides for a pipeline with
/// the PC root signature `layout` (`root` is its title part). The reason of a refusal is an
/// `adapter:<what>` key.
///
/// * A resource at a register must be in the title's root signature and visible to the stage.
/// * A descriptor read from the heap through the mirror needs the mirror and the heap flag of
///   its heap, and a mirror dword the layer writes (below 32).
/// * A descriptor taken by its console heap index needs the heap flag of its heap, and the
///   stage that reads the heap map or the view shadow needs that root view.
/// * An immediate sampler needs the sampler heap flag and an index in the tail of the heap.
pub fn check_stage(stage: &sd::TranslatedStage, kind: StageKind, layout: &RootLayout, root: &sd::RootSignature) -> Result<(), Refusal> {
    let refuse = |what: &str, detail: String| Err(Refusal::new(format!("adapter:{what}"), detail));
    let mirror = layout.mirror_index().is_some();
    let resource_heap = layout.flags & RESOURCE_HEAP_FLAG != 0;
    let sampler_heap = layout.flags & SAMPLER_HEAP_FLAG != 0;
    let abi = options(layout).mirror;
    if stage.needs.heap_map && abi.and_then(|m| m.heap_map).is_none() {
        return refuse("no-heap-map", String::new());
    }
    if stage.needs.view_shadow && abi.and_then(|m| m.view_shadow).is_none() {
        return refuse("no-view-shadow", String::new());
    }
    if stage.needs.resource_heap && !resource_heap {
        return refuse("no-resource-heap-flag", String::new());
    }
    if stage.needs.sampler_heap && !sampler_heap {
        return refuse("no-sampler-heap-flag", String::new());
    }
    if let Some(d) = stage.needs.mirror_dwords.iter().find(|d| !mirror || usize::from(**d) >= crate::user_mirror::DWORDS) {
        return refuse("mirror-dword", format!("dword {d}"));
    }
    for r in &stage.resources {
        match r.placement {
            sd::Placement::Register { class, register, space } => {
                if !covered(root, kind.visibility(), class, register, space) {
                    return refuse("register-not-covered", format!("{} is {}{register} of space {space}", r.name, class.letter()));
                }
            }
            sd::Placement::HeapAtMirror { class, dword, .. } => {
                let flag = if class == sd::RegisterClass::Sampler { sampler_heap } else { resource_heap };
                if !mirror || !flag || usize::from(dword) >= crate::user_mirror::DWORDS {
                    return refuse("heap-at-mirror", format!("{} needs mirror dword {dword} and the heap flag", r.name));
                }
            }
            sd::Placement::HeapAtConsoleIndex { class } => {
                let flag = if class == sd::RegisterClass::Sampler { sampler_heap } else { resource_heap };
                if !flag {
                    return refuse("heap-at-console-index", format!("{} needs the heap flag", r.name));
                }
            }
            sd::Placement::HeapSampler { index, .. } => {
                if !sampler_heap || !(crate::sampler_registry::HEAP_BASE..crate::sampler_registry::HEAP_SIZE).contains(&index) {
                    return refuse("heap-sampler", format!("{} at index {index}", r.name));
                }
            }
        }
    }
    Ok(())
}

/// The answer for one pipeline: per stage the HLSL, or for every stage the same refusal.
type Answer = Vec<Result<StageHlsl, Refusal>>;

/// One translation by `decant_shader::translate`.
type Translated = Result<sd::Translation, decant_shader::Refusal>;

/// What the translating thread hands back: the translation, the listed one for the dump, and its time.
type Outcome = (Result<(Translated, Option<Translated>), Refusal>, u128);

/// A translation running on its own thread: its outcome once there, and whether a bind gave up
/// waiting for it.
#[derive(Default)]
struct Running {
    outcome: Mutex<Option<Outcome>>,
    done: Condvar,
    left_out: AtomicBool,
}

/// How long a binding thread waits for a translation before the pipeline is left out for now
/// (`DECANT_D3D12X_DIRECT_TRANSLATE_WAIT_MS`, default 2000). Some compute shaders of a title took
/// longer than 18 s: the title thread that bound them held the frame,
/// and after about 35 s without a frame the title ended itself.
fn translate_wait() -> std::time::Duration {
    static MS: OnceLock<u64> = OnceLock::new();
    std::time::Duration::from_millis(*MS.get_or_init(|| std::env::var("DECANT_D3D12X_DIRECT_TRANSLATE_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000)))
}

/// The translator behind the layer's direct shader path.
pub struct DirectSource {
    host: Arc<dyn sd::Host + Send + Sync>,
    log: fn(&str),
    dump: Option<PathBuf>,
    /// The last translations, by the translator's cache key of their input.
    kept: Mutex<VecDeque<([u8; 20], Arc<Answer>)>>,
    /// Translations still running, by the same key.
    running: Mutex<HashMap<[u8; 20], Arc<Running>>>,
    /// The wait of a binding thread for a translation.
    wait: std::time::Duration,
    translated: AtomicU64,
    slow: AtomicU64,
    left_out: AtomicU64,
    taken_later: AtomicU64,
}

/// Whether the `n`th event of a kind is logged: the first 32, then the powers of two.
fn log_due(n: u64) -> bool {
    n < 32 || n.is_power_of_two()
}

/// The container hash the layer names override files by, as 16 hex digits.
fn hash_of(bytes: &[u8]) -> String {
    format!("{:016x}", decant_shader::container::fnv1a64(bytes))
}

/// Starts `work` on a thread of its own with the translator's stack; `running` gets its outcome
/// (a panic or a thread that does not start is a refusal) and its time.
fn start_translation(running: &Arc<Running>, work: impl FnOnce() -> (Translated, Option<Translated>) + Send + 'static) {
    let finish = |slot: &Running, outcome: Outcome| {
        *slot.outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);
        slot.done.notify_all();
    };
    let slot = running.clone();
    let started = std::time::Instant::now();
    let spawned = std::thread::Builder::new().name("decant-direct-translate".to_string()).stack_size(TRANSLATE_STACK).spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).map_err(|_| Refusal::new("adapter:translator-panic", ""));
        finish(&slot, (result, started.elapsed().as_millis()));
    });
    if let Err(e) = spawned {
        finish(running, (Err(Refusal::new("adapter:no-thread", e.to_string())), 0));
    }
}

/// Whether `running` has its outcome, after waiting for it `wait` at most.
fn outcome_within(running: &Running, wait: std::time::Duration) -> bool {
    let guard = running.outcome.lock().unwrap_or_else(|e| e.into_inner());
    let (guard, _) = running.done.wait_timeout_while(guard, wait, |o| o.is_none()).unwrap_or_else(|e| e.into_inner());
    guard.is_some()
}

impl DirectSource {
    /// A source that asks `host` for sampler indices, logs through `log` and dumps to `dump`.
    pub fn new(host: Arc<dyn sd::Host + Send + Sync>, log: fn(&str), dump: Option<PathBuf>) -> DirectSource {
        DirectSource::with_wait(host, log, dump, translate_wait())
    }

    /// The same with the wait of a binding thread for a translation.
    pub fn with_wait(host: Arc<dyn sd::Host + Send + Sync>, log: fn(&str), dump: Option<PathBuf>, wait: std::time::Duration) -> DirectSource {
        DirectSource {
            host,
            log,
            dump,
            kept: Mutex::new(VecDeque::new()),
            running: Mutex::new(HashMap::new()),
            wait,
            translated: AtomicU64::new(0),
            slow: AtomicU64::new(0),
            left_out: AtomicU64::new(0),
            taken_later: AtomicU64::new(0),
        }
    }

    /// The source of the layer: sampler indices from `sampler_registry`.
    pub fn for_layer(log: fn(&str), dump: Option<PathBuf>) -> DirectSource {
        DirectSource::new(Arc::new(RegistryHost), log, dump)
    }

    /// The answer for `pipeline`: kept from an earlier call, or translated now.
    fn answer(&self, pipeline: &Pipeline, inline: bool) -> Arc<Answer> {
        let all = |r: Refusal| Arc::new(pipeline.stages.iter().map(|_| Err(r.clone())).collect::<Answer>());
        let names: Vec<String> = pipeline.stages.iter().map(|s| hash_of(&s.bytes)).collect();
        let described = || pipeline.stages.iter().zip(&names).map(|(s, n)| format!("{} {n}", s.kind.short())).collect::<Vec<_>>().join(", ");
        // Served pipelines read their tables through the heap at the index the mirror holds.
        let layout = &pipeline.root;
        if layout.mirror_index().is_none() {
            return all(Refusal::new("adapter:no-mirror", described()));
        }
        if layout.flags & RESOURCE_HEAP_FLAG == 0 || layout.flags & SAMPLER_HEAP_FLAG == 0 {
            return all(Refusal::new("adapter:no-heap-flags", format!("root signature flags {:#x}, {}", layout.flags, described())));
        }
        // The containers are read here: bounds checked parsing without recursion.
        let mut programs = Vec::with_capacity(pipeline.stages.len());
        for (stage, name) in pipeline.stages.iter().zip(&names) {
            let kind = stage_of(stage.kind);
            let parsed = std::panic::catch_unwind(|| decant_shader::container::program(&stage.bytes, name));
            match parsed {
                Ok(Ok(p)) if p.stage == kind => programs.push(p),
                Ok(Ok(p)) => return all(Refusal::new("adapter:stage-mismatch", format!("{} {name} holds a {:?} program", stage.kind.short(), p.stage))),
                Ok(Err(r)) => return all(Refusal::new(r.key(), format!("{} {name}", stage.kind.short()))),
                Err(_) => return all(Refusal::new("adapter:translator-panic", format!("reading {} {name}", stage.kind.short()))),
            }
        }
        let root = root_signature(layout);
        let options = options(layout);
        if inline {
            let started = std::time::Instant::now();
            let translated = decant_shader::translate(&programs, &root, &options, &*self.host);
            let listed = self.dump.is_some().then(|| decant_shader::translate(
                &programs, &root, &sd::Options { listing: true, ..options.clone() }, &*self.host,
            ));
            return self.finish_translation(pipeline, &names, &root, Ok((translated, listed)), started.elapsed().as_millis());
        }
        // Registry indices are the same in every run (the registry file), so no host tag.
        let key = decant_shader::cache::key(&programs, &root, &options, 0);
        if let Some(found) = self.kept.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key).map(|(_, a)| a.clone()) {
            return found;
        }
        // A translation runs on a thread of its own. The binding thread waits for it a while; past
        // that the pipeline is left out for now and a later bind takes the finished translation.
        let (running, fresh) = {
            let mut map = self.running.lock().unwrap_or_else(|e| e.into_inner());
            match map.get(&key) {
                Some(r) => (r.clone(), false),
                None => {
                    let r = Arc::new(Running::default());
                    map.insert(key, r.clone());
                    (r, true)
                }
            }
        };
        if fresh {
            (self.log)(&format!("direct shaders: the translation of {} starts", described()));
            let (host, root, options, listing) = (self.host.clone(), root.clone(), options.clone(), self.dump.is_some());
            start_translation(&running, move || {
                let translation = decant_shader::translate(&programs, &root, &options, &*host);
                // The dump wants the instruction listing in the text, which is another translation.
                let listed = listing.then(|| decant_shader::translate(&programs, &root, &sd::Options { listing: true, ..options.clone() }, &*host));
                (translation, listed)
            });
        }
        // Only the first bind waits: a pipeline already left out is asked again without a wait.
        let wait = if running.left_out.load(Ordering::Relaxed) { std::time::Duration::ZERO } else { self.wait };
        let pending = || all(Refusal::new(Refusal::PENDING, described()));
        if !outcome_within(&running, wait) {
            if !running.left_out.swap(true, Ordering::Relaxed) {
                let n = self.left_out.fetch_add(1, Ordering::Relaxed) + 1;
                if log_due(n - 1) {
                    (self.log)(&format!("direct shaders: {} is not translated after {} ms, the pipeline is left out for now ({n} so far)", described(), wait.as_millis()));
                }
            }
            return pending();
        }
        // The binding thread that takes the entry out makes the answer; another one that waited for
        // the same translation finds it kept, or asks again at its next bind.
        let taken = self.running.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).and_then(|r| r.outcome.lock().unwrap_or_else(|e| e.into_inner()).take());
        let Some((translated, ms)) = taken else {
            return self.kept.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key).map(|(_, a)| a.clone()).unwrap_or_else(pending);
        };
        (self.log)(&format!("direct shaders: the translation of {} ended in {ms} ms", described()));
        if running.left_out.load(Ordering::Relaxed) {
            let n = self.taken_later.fetch_add(1, Ordering::Relaxed) + 1;
            if log_due(n - 1) {
                (self.log)(&format!("direct shaders: {} is taken later, translated in {ms} ms ({n} so far)", described()));
            }
        }
        let answer = self.finish_translation(pipeline, &names, &root, translated, ms);
        if ms >= SLOW_MS && log_due(self.slow.fetch_add(1, Ordering::Relaxed)) {
            (self.log)(&format!("direct shaders: the translation of {} took {ms} ms on the binding thread", described()));
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.len() >= KEPT {
            kept.pop_front();
        }
        kept.push_back((key, answer.clone()));
        answer
    }

    fn finish_translation(&self, pipeline: &Pipeline, names: &[String], root: &sd::RootSignature, translated: Result<(Translated, Option<Translated>), Refusal>, ms: u128) -> Arc<Answer> {
        let all = |r: Refusal| Arc::new(pipeline.stages.iter().map(|_| Err(r.clone())).collect::<Answer>());
        let described = || pipeline.stages.iter().zip(names).map(|(s, n)| format!("{} {n}", s.kind.short())).collect::<Vec<_>>().join(", ");
        match translated {
            Err(r) => all(Refusal::new(r.reason, described())),
            Ok((Err(r), _)) => {
                let at = r.stage.and_then(|s| pipeline.stages.iter().position(|p| stage_of(p.kind) == s));
                let who = at.map_or_else(described, |i| format!("{} {}", pipeline.stages[i].kind.short(), names[i]));
                let place = r.pc.map_or_else(String::new, |pc| format!(" at word {pc}"));
                let detail = if r.detail.is_empty() { format!("{who}{place}") } else { format!("{who}{place}: {}", r.detail) };
                self.dump_refusal(at.map_or(&names[0], |i| &names[i]), &r.key(), &detail);
                all(Refusal::new(r.key(), detail))
            }
            Ok((Ok(translation), listed)) => self.checked(pipeline, names, translation, listed.and_then(Result::ok), root, ms),
        }
    }

    #[cfg(windows)]
    pub(crate) fn translate_on_worker(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
        let answer = self.answer(pipeline, true);
        Arc::try_unwrap(answer).unwrap_or_else(|shared| shared.as_ref().clone())
    }

    /// The HLSL of a translation whose stages all pass [`check_stage`], else the refusal.
    fn checked(&self, pipeline: &Pipeline, names: &[String], translation: sd::Translation, listed: Option<sd::Translation>, root: &sd::RootSignature, ms: u128) -> Arc<Answer> {
        let all = |r: Refusal| Arc::new(pipeline.stages.iter().map(|_| Err(r.clone())).collect::<Answer>());
        if translation.stages.len() != pipeline.stages.len() {
            return all(Refusal::new("adapter:stage-count", format!("{} answers for {} stages", translation.stages.len(), pipeline.stages.len())));
        }
        for ((stage, input), name) in translation.stages.iter().zip(&pipeline.stages).zip(names) {
            if let Err(mut r) = check_stage(stage, input.kind, &pipeline.root, root) {
                r.detail = if r.detail.is_empty() { format!("{} {name}", input.kind.short()) } else { format!("{} {name}: {}", input.kind.short(), r.detail) };
                self.dump_refusal(name, &r.reason, &r.detail);
                return all(r);
            }
        }
        if let Some(dir) = &self.dump {
            let _ = std::fs::create_dir_all(dir);
            for (stage, name) in listed.as_ref().unwrap_or(&translation).stages.iter().zip(names) {
                let _ = std::fs::write(dir.join(format!("{name}.{}.hlsl", stage.profile)), &stage.hlsl);
            }
        }
        let n = self.translated.fetch_add(1, Ordering::Relaxed);
        if log_due(n) {
            let stages: Vec<String> = translation
                .stages
                .iter()
                .zip(&pipeline.stages)
                .zip(names)
                .map(|((t, input), name)| {
                    let heap = t.resources.iter().filter(|r| !matches!(r.placement, sd::Placement::Register { .. })).count();
                    format!("{} {name} (mirror dwords {:?}, {} of {} resources through the heap)", input.kind.short(), t.needs.mirror_dwords, heap, t.resources.len())
                })
                .collect();
            (self.log)(&format!("direct shaders: translated {} in {ms} ms ({} pipelines so far)", stages.join(", "), n + 1));
        }
        Arc::new(translation.stages.into_iter().map(|s| Ok(StageHlsl { hlsl: s.hlsl, entry: s.entry, profile: s.profile, args: s.dxc_args })).collect())
    }

    fn dump_refusal(&self, name: &str, reason: &str, detail: &str) {
        if let Some(dir) = &self.dump {
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("{name}.refused.txt")), format!("{reason}\n{detail}\n"));
        }
    }
}

pub(crate) fn hlsl_key(s: &StageHlsl) -> u128 {
    let mut h = Fnv128::default();
    h.field(s.hlsl.as_bytes()).field(s.entry.as_bytes()).field(s.profile.as_bytes());
    for a in &s.args {
        h.field(a.as_bytes());
    }
    h.finish()
}

impl ShaderSource for DirectSource {
    fn version(&self) -> String {
        // The wave form is part of what a pipeline file names: a pipeline cached with
        // `[WaveSize(64)]` stages must not be served to a device that refuses them.
        format!(
            "decant-shader {} adapter {ADAPTER_VERSION}{}",
            decant_shader::TRANSLATOR_VERSION,
            if host_wave_emulated() { " wave64-emulated" } else { "" }
        )
    }

    fn accepts(&self, pipeline: &Pipeline) -> Result<(), Refusal> {
        match self.answer(pipeline, false).iter().find_map(|s| s.as_ref().err()) {
            Some(r) => Err(r.clone()),
            None => Ok(()),
        }
    }

    fn stage_keys(&self, pipeline: &Pipeline) -> Option<Vec<u128>> {
        self.answer(pipeline, false)
            .iter()
            .map(|s| {
                let s = s.as_ref().ok()?;
                Some(hlsl_key(s))
            })
            .collect()
    }

    fn translate(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
        self.answer(pipeline, false).as_ref().clone()
    }
}

#[cfg(test)]
mod wave_mode_tests {
    use super::*;

    #[test]
    fn a_device_with_32_lane_waves_gets_the_emulated_form() {
        assert_eq!(wave_mode_for(32, false), sd::WaveMode::Emulated);
        assert_eq!(wave_mode_for(64, false), sd::WaveMode::Scalar);
        assert_eq!(wave_mode_for(128, false), sd::WaveMode::Scalar);
        // unknown device, or the switch off: the native form, as before
        assert_eq!(wave_mode_for(0, false), sd::WaveMode::Scalar);
        assert_eq!(wave_mode_for(32, true), sd::WaveMode::Scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_fixture::{container, ui_pixel, ui_vertex};
    use crate::rts0::{RootParam, RootParamKind, RootRange};
    use crate::shader_source::{Linkage, StageInput};

    fn mirror_param() -> RootParam {
        RootParam { kind: RootParamKind::Descriptor { class: rts0::RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 }
    }

    fn range(class: rts0::RegisterClass, base: u32, offset: u32) -> RootRange {
        RootRange { class, count: 1, base, space: 1, flags: 0, offset }
    }

    /// The PC root signature of a title one with a single table: mirror and heap flags added.
    fn layout(ranges: Vec<RootRange>) -> RootLayout {
        RootLayout { version: 2, flags: 0xC01, params: vec![RootParam { kind: RootParamKind::Table(ranges), visibility: 0 }, mirror_param()], static_samplers: Vec::new() }
    }

    fn pipeline(stages: &[sd::StageProgram], root: RootLayout) -> Pipeline {
        let kind = |s: sd::Stage| match s {
            sd::Stage::Vertex => StageKind::Vertex,
            sd::Stage::Pixel => StageKind::Pixel,
            sd::Stage::Mesh => StageKind::Mesh,
            _ => panic!("no fixture mapping for {s:?}"),
        };
        Pipeline {
            stages: stages.iter().map(|p| StageInput { kind: kind(p.stage), bytes: Arc::from(container(p)) }).collect(),
            root,
            linkage: Linkage { stages: stages.iter().map(|p| kind(p.stage)).collect(), ..Linkage::default() },
        }
    }

    /// The table of the title's user interface pipeline: `t2` then `b0`. The PC has `b0` at
    /// slot 1, the console program reads its constant buffer at slot 0.
    fn title_table() -> Vec<RootRange> {
        vec![range(rts0::RegisterClass::Srv, 2, 0), range(rts0::RegisterClass::Cbv, 0, 1)]
    }

    fn source() -> DirectSource {
        DirectSource::new(Arc::new(sd::CountingHost::default()), |_| {}, None)
    }

    #[test]
    fn the_container_of_a_program_reads_back_as_that_program() {
        for program in [ui_vertex(), ui_pixel()] {
            let read = decant_shader::container::program(&container(&program), &program.name).expect("reads");
            assert_eq!(read, program);
        }
        assert_ne!(container(&ui_vertex())[4..20], container(&ui_pixel())[4..20], "the containers are told apart by their checksum field");
    }

    #[test]
    fn mesh_and_pixel_translate_together_and_key_the_linked_output() {
        let mesh = crate::direct_fixture::mesh_triangle();
        let root = layout(Vec::new());
        let source = source();
        let linked = pipeline(&[mesh.clone(), ui_pixel()], root.clone());
        let standalone = pipeline(&[mesh], root);
        let answer = source.translate(&linked);
        let [ms, ps] = answer.as_slice() else { panic!("expected two stages") };
        let ms = ms.as_ref().unwrap();
        let ps = ps.as_ref().unwrap();
        assert_eq!((ms.profile.as_str(), ps.profile.as_str()), ("ms_6_6", "ps_6_6"));
        assert!(ms.hlsl.contains("SetMeshOutputCounts"));
        assert!(!ms.hlsl.contains("UNUSED"));
        assert_ne!(source.stage_keys(&linked).unwrap()[0], source.stage_keys(&standalone).unwrap()[0]);
    }

    #[test]
    fn the_root_signature_is_the_title_part_with_resolved_offsets() {
        let mut l = layout(vec![range(rts0::RegisterClass::Srv, 2, 0), RootRange { count: u32::MAX, ..range(rts0::RegisterClass::Cbv, 0, 1) }]);
        l.params.insert(1, RootParam { kind: RootParamKind::Constants { register: 3, space: 0, count: 4 }, visibility: 5 });
        l.params.insert(2, RootParam { kind: RootParamKind::Descriptor { class: rts0::RegisterClass::Uav, register: 7, space: 2 }, visibility: 0 });
        l.static_samplers.push(rts0::StaticSampler { register: 4, space: 0, visibility: 5 });
        let root = root_signature(&l);
        assert_eq!(
            root,
            sd::RootSignature {
                params: vec![
                    sd::RootParam {
                        kind: sd::RootParamKind::Table(vec![
                            sd::RootRange { class: sd::RegisterClass::Srv, count: 1, base_register: 2, space: 1, offset: 0 },
                            sd::RootRange { class: sd::RegisterClass::Cbv, count: u32::MAX, base_register: 0, space: 1, offset: 1 },
                        ]),
                        visibility: 0
                    },
                    sd::RootParam { kind: sd::RootParamKind::Constants { register: 3, space: 0, count: 4 }, visibility: 5 },
                    sd::RootParam { kind: sd::RootParamKind::Descriptor { class: sd::RegisterClass::Uav, register: 7, space: 2 }, visibility: 0 },
                ],
                static_samplers: vec![sd::StaticSampler { register: 4, space: 0, visibility: 5 }],
                flags: 0xC01,
            },
            "the mirror parameter is left out, the rest is copied in order"
        );
        assert_eq!(options(&l).mirror, Some(sd::MirrorAbi { register: 1, space: 15, heap_map: None, view_shadow: None }));
        // The heap map and the view shadow are named when the root signature has their root views.
        let view = |register| RootParam { kind: RootParamKind::Descriptor { class: rts0::RegisterClass::Srv, register, space: 15 }, visibility: 0 };
        let mut with_views = l.clone();
        with_views.params.extend([view(0), view(1)]);
        assert_eq!(options(&with_views).mirror, Some(sd::MirrorAbi { register: 1, space: 15, heap_map: Some(0), view_shadow: Some(1) }));
        assert_eq!(root_signature(&with_views), root, "they are the layer's parameters, not the title's");
        l.params.pop();
        assert_eq!(options(&l), sd::Options { mirror: None, wave: sd::WaveMode::Scalar, shader_model_minor: 6, listing: false });
    }

    #[test]
    fn the_ui_pair_translates_with_the_buffer_read_through_the_mirror() {
        let p = pipeline(&[ui_vertex(), ui_pixel()], layout(title_table()));
        let s = source();
        assert_eq!(s.accepts(&p), Ok(()));
        let out = s.translate(&p);
        let (vs, ps) = (out[0].as_ref().expect("vertex"), out[1].as_ref().expect("pixel"));
        assert_eq!((vs.profile.as_str(), ps.profile.as_str(), vs.entry.as_str()), ("vs_6_6", "ps_6_6", "main"));
        assert!(vs.hlsl.contains("ResourceDescriptorHeap[decant_mirror[2].x + 0u]"), "{}", vs.hlsl);
        assert!(vs.hlsl.contains("register(b1, space15)"), "{}", vs.hlsl);
        assert!(!vs.hlsl.contains("register(b0, space1)"), "never the register of the range: {}", vs.hlsl);
        // The container hash names the program in the text.
        assert!(vs.hlsl.contains(&hash_of(&p.stages[0].bytes)), "{}", vs.hlsl);
        // One key per stage, from the text: the same for the same pipeline, another for another text.
        let keys = s.stage_keys(&p).expect("keys");
        assert_eq!(keys.len(), 2);
        assert_eq!(s.stage_keys(&p), Some(keys.clone()));
        assert_ne!(keys[0], keys[1]);
        assert_eq!(s.kept.lock().unwrap().len(), 1, "translated once for the three calls");
    }

    #[test]
    fn a_translation_past_the_wait_is_left_out_for_now_and_taken_at_a_later_bind() {
        let p = pipeline(&[ui_vertex(), ui_pixel()], layout(title_table()));
        let s = DirectSource::with_wait(Arc::new(sd::CountingHost::default()), |_| {}, None, std::time::Duration::ZERO);
        // With no wait the first bind does not get the translation, unless the thread was faster.
        let first = s.accepts(&p);
        assert!(first.is_ok() || first.as_ref().is_err_and(Refusal::is_pending), "{first:?}");
        let started = std::time::Instant::now();
        let mut later = s.accepts(&p);
        while later.as_ref().is_err_and(Refusal::is_pending) && started.elapsed() < std::time::Duration::from_secs(20) {
            std::thread::sleep(std::time::Duration::from_millis(5));
            later = s.accepts(&p);
        }
        assert_eq!(later, Ok(()), "a later bind takes the finished translation");
        let (left, taken) = (s.left_out.load(Ordering::Relaxed), s.taken_later.load(Ordering::Relaxed));
        assert_eq!((left, taken), if first.is_ok() { (0, 0) } else { (1, 1) }, "counted once each");
        assert!(s.running.lock().unwrap().is_empty(), "nothing runs any more");
        assert_eq!(s.translate(&p).iter().filter(|r| r.is_ok()).count(), 2, "the kept translation serves the compile worker");
    }

    #[test]
    fn a_pending_translation_is_no_refusal_of_the_service() {
        let pending = Refusal::new(Refusal::PENDING, "vs 1, ps 2");
        assert!(pending.is_pending());
        assert!(!Refusal::new("op:x", "").is_pending());
    }

    #[test]
    fn a_pixel_stage_is_keyed_with_the_stage_it_links_to() {
        let s = source();
        let root = || layout(title_table());
        let base = s.stage_keys(&pipeline(&[ui_vertex(), ui_pixel()], root())).expect("keys");
        // Another output order of the vertex stage is another input struct of the same pixel bytes.
        let mut vs = ui_vertex();
        vs.outputs.swap(2, 3);
        vs.outputs[2].register = 2;
        vs.outputs[3].register = 3;
        let other = s.stage_keys(&pipeline(&[vs, ui_pixel()], root())).expect("keys");
        assert_ne!(other[1], base[1], "the pixel text follows the vertex outputs");
    }

    #[test]
    fn a_stage_the_translator_refuses_refuses_the_pipeline_by_its_key() {
        let mut ps = ui_pixel();
        ps.inputs[3].semantic = "NORMAL".to_owned();
        let p = pipeline(&[ui_vertex(), ps], layout(title_table()));
        let s = source();
        let refusal = s.accepts(&p).expect_err("refused");
        assert_eq!(refusal.reason, "stage:link-missing");
        assert!(refusal.detail.starts_with(&format!("ps {}", hash_of(&p.stages[1].bytes))), "{}", refusal.detail);
        // Every stage answers with it, and there is no key.
        let out = s.translate(&p);
        assert!(out.iter().all(|r| r.as_ref().is_err_and(|e| e.reason == "stage:link-missing")));
        assert_eq!(s.stage_keys(&p), None);
    }

    #[test]
    fn a_pipeline_without_the_mirror_or_the_heap_flags_is_refused() {
        let s = source();
        let mut no_mirror = layout(Vec::new());
        no_mirror.params.pop();
        assert_eq!(s.accepts(&pipeline(&[ui_vertex(), ui_pixel()], no_mirror)).map_err(|r| r.reason), Err("adapter:no-mirror".to_string()));
        for flags in [0x1, 0x401, 0x801] {
            let l = RootLayout { flags, ..layout(Vec::new()) };
            assert_eq!(s.accepts(&pipeline(&[ui_vertex(), ui_pixel()], l)).map_err(|r| r.reason), Err("adapter:no-heap-flags".to_string()), "flags {flags:#x}");
        }
    }

    #[test]
    fn bytes_that_are_no_console_container_are_refused_by_the_container_key() {
        let p = Pipeline {
            stages: vec![StageInput { kind: StageKind::Compute, bytes: Arc::from(&b"not a container"[..]) }],
            root: layout(Vec::new()),
            linkage: Linkage { stages: vec![StageKind::Compute], ..Linkage::default() },
        };
        let r = source().accepts(&p).expect_err("refused");
        assert!(r.reason.starts_with("container:"), "{}", r.reason);
        // A container of another stage than the pipeline says.
        let mut p = pipeline(&[ui_vertex()], layout(Vec::new()));
        p.stages[0].kind = StageKind::Geometry;
        assert_eq!(source().accepts(&p).map_err(|r| r.reason), Err("adapter:stage-mismatch".to_string()));
    }

    fn stage(resources: Vec<sd::Placement>, needs: sd::Needs) -> sd::TranslatedStage {
        sd::TranslatedStage {
            stage: sd::Stage::Pixel,
            hlsl: String::new(),
            entry: "main".into(),
            profile: "ps_6_6".into(),
            dxc_args: Vec::new(),
            resources: resources.into_iter().map(|placement| sd::BoundResource { name: "r".into(), placement }).collect(),
            needs,
        }
    }

    #[test]
    fn what_a_stage_needs_is_checked_against_the_root_signature() {
        let check = |s: &sd::TranslatedStage, l: &RootLayout| check_stage(s, StageKind::Pixel, l, &root_signature(l)).map_err(|r| r.reason);
        let mut l = layout(vec![range(rts0::RegisterClass::Cbv, 27, 0)]);
        l.params.insert(1, RootParam { kind: RootParamKind::Descriptor { class: rts0::RegisterClass::Srv, register: 3, space: 0 }, visibility: 1 });
        l.static_samplers.push(rts0::StaticSampler { register: 2, space: 0, visibility: 5 });
        let reg = |class, register, space| sd::Placement::Register { class, register, space };
        // Registers: in a range, as a static sampler, and what is missing or seen by another stage.
        assert_eq!(check(&stage(vec![reg(sd::RegisterClass::Cbv, 27, 1), reg(sd::RegisterClass::Sampler, 2, 0)], sd::Needs::default()), &l), Ok(()));
        assert_eq!(check(&stage(vec![reg(sd::RegisterClass::Cbv, 28, 1)], sd::Needs::default()), &l), Err("adapter:register-not-covered".into()));
        assert_eq!(check(&stage(vec![reg(sd::RegisterClass::Srv, 3, 0)], sd::Needs::default()), &l), Err("adapter:register-not-covered".into()), "a vertex only parameter");
        // Heap reads through the mirror, and immediate samplers in the tail.
        let heap = sd::Placement::HeapAtMirror { class: sd::RegisterClass::Srv, dword: 8, slot: 1 };
        let needs = sd::Needs { mirror_dwords: vec![8], resource_heap: true, sampler_heap: true, ..sd::Needs::default() };
        let sampler = |index| sd::Placement::HeapSampler { words: [0; 4], index };
        assert_eq!(check(&stage(vec![heap, sampler(crate::sampler_registry::HEAP_BASE + 3)], needs.clone()), &l), Ok(()));
        assert_eq!(check(&stage(vec![sampler(3)], needs.clone()), &l), Err("adapter:heap-sampler".into()), "an index outside the tail");
        assert_eq!(check(&stage(Vec::new(), sd::Needs { mirror_dwords: vec![32], ..sd::Needs::default() }), &l), Err("adapter:mirror-dword".into()));
        assert_eq!(check(&stage(vec![heap], needs.clone()), &RootLayout { flags: 0x801, ..l.clone() }), Err("adapter:no-resource-heap-flag".into()));
        assert_eq!(check(&stage(vec![heap], needs.clone()), &RootLayout { flags: 0x401, ..l.clone() }), Err("adapter:no-sampler-heap-flag".into()));
        // A descriptor by its console heap index, the heap map and the view shadow.
        let by_index = |class| sd::Placement::HeapAtConsoleIndex { class };
        assert_eq!(check(&stage(vec![by_index(sd::RegisterClass::Srv), by_index(sd::RegisterClass::Sampler)], needs.clone()), &l), Ok(()));
        assert_eq!(check(&stage(vec![by_index(sd::RegisterClass::Sampler)], sd::Needs::default()), &RootLayout { flags: 0x401, ..l.clone() }), Err("adapter:heap-at-console-index".into()));
        assert_eq!(check(&stage(Vec::new(), sd::Needs { heap_map: true, ..needs.clone() }), &l), Err("adapter:no-heap-map".into()));
        assert_eq!(check(&stage(Vec::new(), sd::Needs { view_shadow: true, ..needs.clone() }), &l), Err("adapter:no-view-shadow".into()));
        let view = |register| RootParam { kind: RootParamKind::Descriptor { class: rts0::RegisterClass::Srv, register, space: 15 }, visibility: 0 };
        let mut with_views = l.clone();
        with_views.params.extend([view(0), view(1)]);
        assert_eq!(check(&stage(Vec::new(), sd::Needs { heap_map: true, view_shadow: true, ..needs.clone() }), &with_views), Ok(()));
        let mut no_mirror = l.clone();
        no_mirror.params.pop();
        assert_eq!(check(&stage(vec![heap], needs), &no_mirror), Err("adapter:mirror-dword".into()));
    }

    #[test]
    fn the_registry_host_gives_absolute_indices_also_for_unnormalised_samplers() {
        use sd::Host;
        let plain = [0x0000_0492, 0x00FF_F000, 0x0850_0000, 0];
        let index = RegistryHost.sampler_index(plain).expect("room in the registry");
        assert!((crate::sampler_registry::HEAP_BASE..crate::sampler_registry::HEAP_SIZE).contains(&index));
        assert_eq!(RegistryHost.sampler_index(plain), Some(index), "the same words, the same index");
        // Another sampler for the registry, the same PC sampler: the shader divides the coordinates.
        let unnormalised = [plain[0] | 1 << 15, plain[1], plain[2], plain[3]];
        let other = RegistryHost.sampler_index(unnormalised).expect("room in the registry");
        assert!(other != index && other >= crate::sampler_registry::HEAP_BASE);
    }

    #[test]
    fn the_dump_gets_the_listing_and_the_refusal() {
        let tmp = crate::direct_cache::tests::TempDir::new("direct-dump");
        let s = DirectSource::new(Arc::new(sd::CountingHost::default()), |_| {}, Some(tmp.0.clone()));
        let root = || layout(title_table());
        let good = pipeline(&[ui_vertex(), ui_pixel()], root());
        assert_eq!(s.accepts(&good), Ok(()));
        let listing = std::fs::read_to_string(tmp.0.join(format!("{}.vs_6_6.hlsl", hash_of(&good.stages[0].bytes)))).expect("vertex dump");
        assert!(listing.contains("s_swappc_b64"), "{listing}");
        // The text that is compiled has no listing.
        assert!(!s.translate(&good)[0].as_ref().expect("vertex").hlsl.contains("s_swappc_b64"));
        let mut ps = ui_pixel();
        ps.inputs[3].semantic = "NORMAL".to_owned();
        let bad = pipeline(&[ui_vertex(), ps], root());
        assert!(s.accepts(&bad).is_err());
        let text = std::fs::read_to_string(tmp.0.join(format!("{}.refused.txt", hash_of(&bad.stages[1].bytes)))).expect("refusal dump");
        assert!(text.starts_with("stage:link-missing\n"), "{text}");
    }
}
