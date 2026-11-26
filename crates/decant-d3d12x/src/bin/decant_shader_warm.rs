//! `decant_shader_warm` fills the direct shader disk cache (`direct_cache`) from a pipeline manifest
//! (`pipeline_manifest`, `DECANT_D3D12X_PIPELINE_MANIFEST`) or directly from a shader archive so the
//! first bind of a warmed pipeline in a later process is a file read. It reuses the translator
//! (`decant-shader` via `DirectSource`), compiler (`decant-dxc::Dxc`) and disk cache
//! (`direct_cache::DiskCache`) through `compile_service::Service`, the same path the layer runs.
//!
//! `--warm-pso` adds a second stage that creates headless PC pipelines and stores driver blobs.
//! `--pso-cache` selects the runtime PSO cache directory. It otherwise follows
//! `DECANT_D3D12X_PSO_CACHE`. Progress uses `PROGRESS_PSO` and the report counts successful
//! creation, accepted cache hits, stored blobs, failures and unsupported records.
//! Graphics needs version 2 inline state. Version 1 compute uses node mask and flags of zero.
//! Static sampler layouts are skipped because the manifest lacks sampler descriptions.
//!
//! Shader bytes are never stored in a manifest or in this tool's report. Each stage is resolved by
//! its 64 bit FNV 1a container hash from `--shaders` archives first and then as `<hash>.bin` under
//! `--dump-dir`. When `--manifest` is omitted, single stage records are built from the resolved
//! containers in archive order. When a bind order file is given (`--bind-order`) or present in the
//! cache folder (`bind-order.txt`), pipelines are ranked by first bind order before `--top` cuts
//! the list.
//!
//! `DECANT_D3D12X_DIRECT_CACHE` is set to `--cache` before any translation starts so this tool and a
//! game run share `immediate-samplers.bin` (`sampler_registry::default_file` and `index_of`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

// Only `std::env::set_var` below needs this, once, before any worker thread starts.
#[allow(unsafe_code)]
mod env_setup {
    pub fn set(cache: &std::path::Path) {
        // SAFETY: called once at startup on the main thread before any other thread reads env vars.
        unsafe { std::env::set_var("DECANT_D3D12X_DIRECT_CACHE", cache) };
    }
}

use clap::Parser;
use d3d12_xs::compile_service::{Compiler, Poll, Service, Settings};
use d3d12_xs::direct_cache::DiskCache;
use d3d12_xs::direct_source::{DirectSource, RegistryHost};
use d3d12_xs::pipeline_manifest::{self, Manifest, Record};
use d3d12_xs::rts0::{RegisterClass, RootLayout, RootParam, RootParamKind, RootRange};
use d3d12_xs::shader_source::{Linkage, Pipeline, Refusal, StageInput, StageKind};

#[derive(Parser, Debug)]
#[command(about = "Fills the direct shader disk cache from a pipeline manifest or shader archive.")]
struct Args {
    /// Pipeline manifest written by a game run with DECANT_D3D12X_DIRECT=1 and DECANT_D3D12X_PIPELINE_MANIFEST.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Create headless PC pipeline states after compiling DXIL.
    #[arg(long, requires = "manifest")]
    warm_pso: bool,
    /// Driver PSO cache directory shared with the runtime.
    #[arg(long, requires = "warm_pso")]
    pso_cache: Option<PathBuf>,
    /// Optional bind order file used to rank pipelines before applying --top.
    #[arg(long = "bind-order")]
    bind_order: Option<PathBuf>,
    /// Shader archives used to resolve container hashes or build corpus pipelines. Repeatable.
    #[arg(long = "shaders")]
    shaders: Vec<PathBuf>,
    /// Folder of `<hash>.bin` container dumps tried when a hash is not in `--shaders`.
    #[arg(long = "dump-dir")]
    dump_dir: Option<PathBuf>,
    /// Disk cache folder (DECANT_D3D12X_DIRECT_CACHE) shared with the game.
    #[arg(long)]
    cache: PathBuf,
    /// Worker threads translating and compiling pipelines at once.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    /// Warm only the top N pipelines (by bind order when a bind order file exists, else manifest order).
    #[arg(long)]
    top: Option<usize>,
    /// Warm only pipelines that reference a stage hash from this file (one 16 hex digit hash per line).
    #[arg(long = "hash-list")]
    hash_list: Option<PathBuf>,
    /// Folder where DXC puts the HLSL and message of every stage it refuses.
    #[arg(long = "dxc-dump")]
    dxc_dump: Option<PathBuf>,
    /// Optional file where the text report is also written.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Base wait step in milliseconds used to derive the per pipeline wait cap.
    #[arg(long, default_value_t = 200)]
    poll_ms: u64,
    /// Print every per stage translation line from the layer adapter.
    #[arg(long)]
    verbose: bool,
    /// Print structured machine readable progress lines to stdout during warming.
    #[arg(long)]
    progress: bool,
}

/// Wraps `decant_dxc::Dxc` as a `compile_service::Compiler`.
struct DxcCompiler(decant_dxc::Dxc);

impl Compiler for DxcCompiler {
    fn version(&self) -> String {
        self.0.version()
    }
    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
        self.0.compile(hlsl, entry, profile, args)
    }
}

/// Loaded container bytes keyed by their 64 bit FNV 1a hash.
struct Containers {
    found: HashMap<u64, Arc<[u8]>>,
}

impl Containers {
    /// Resolves every hash of `wanted` from `shaders` archives, then `dump_dir`.
    fn load(wanted: &HashSet<u64>, shaders: &[PathBuf], dump_dir: Option<&Path>, log: impl Fn(&str)) -> Containers {
        let mut found: HashMap<u64, Arc<[u8]>> = HashMap::new();
        for archive in shaders {
            if found.len() >= wanted.len() {
                break;
            }
            let items = match decant_extract_shaders::find_shaders(archive) {
                Ok(i) => i,
                Err(e) => {
                    log(&format!("decant_shader_warm: {} not scanned ({e})", archive.display()));
                    continue;
                }
            };
            for item in &items {
                if found.len() >= wanted.len() {
                    break;
                }
                let bytes = item.container();
                let hash = decant_shader::container::fnv1a64(bytes);
                if wanted.contains(&hash) && !found.contains_key(&hash) {
                    found.insert(hash, Arc::from(bytes));
                }
            }
        }
        if let Some(dir) = dump_dir {
            for hash in wanted {
                if found.contains_key(hash) {
                    continue;
                }
                let path = dir.join(format!("{hash:016x}.bin"));
                if let Ok(bytes) = std::fs::read(&path) {
                    found.insert(*hash, Arc::from(bytes));
                }
            }
        }
        Containers { found }
    }
}

/// Bind rank table read from a bind order file or a game log.
#[derive(Default, Debug)]
struct BindOrder {
    by_stages: HashMap<Vec<(StageKind, u64)>, usize>,
    by_hash: HashMap<u64, usize>,
}

impl BindOrder {
    fn is_empty(&self) -> bool {
        self.by_stages.is_empty() && self.by_hash.is_empty()
    }

    fn rank(&self, record: &Record) -> Option<usize> {
        if let Some(&r) = self.by_stages.get(&record.stages) {
            return Some(r);
        }
        record.stages.iter().filter_map(|(_, h)| self.by_hash.get(h).copied()).min()
    }
}

fn stage_tag(token: &str) -> Option<StageKind> {
    match token.to_ascii_uppercase().as_str() {
        "VS" => Some(StageKind::Vertex),
        "PS" => Some(StageKind::Pixel),
        "DS" => Some(StageKind::Domain),
        "HS" => Some(StageKind::Hull),
        "GS" => Some(StageKind::Geometry),
        "CS" => Some(StageKind::Compute),
        "AS" => Some(StageKind::Amplification),
        "MS" => Some(StageKind::Mesh),
        _ => None,
    }
}

/// Parses a bind order file. Each line may name stage tag and hash pairs (like `VS <hash> PS <hash>`,
/// manifest `P` lines, or layer log lines) or a bare 16 hex digit container hash.
fn parse_bind_order(text: &str) -> BindOrder {
    let mut order = BindOrder::default();
    let mut next_rank = 0usize;
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with("R ") {
            continue;
        }
        let tokens: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | ':' | '(' | ')' | '[' | ']' | '=' | '+'))
            .filter(|t| !t.is_empty())
            .collect();
        let mut stages = Vec::new();
        let mut i = 0;
        while i + 1 < tokens.len() {
            if let Some(kind) = stage_tag(tokens[i])
                && tokens[i + 1].len() == 16
                && let Ok(hash) = u64::from_str_radix(tokens[i + 1], 16)
            {
                stages.push((kind, hash));
                i += 2;
                continue;
            }
            i += 1;
        }
        if !stages.is_empty() {
            let rank = next_rank;
            next_rank += 1;
            order.by_stages.entry(stages.clone()).or_insert(rank);
            for (_, hash) in stages {
                order.by_hash.entry(hash).or_insert(rank);
            }
            continue;
        }
        if let Some(first) = tokens.first() {
            let clean = first.trim_start_matches("0x").trim_start_matches("0X");
            if clean.len() == 16
                && let Ok(hash) = u64::from_str_radix(clean, 16)
            {
                order.by_hash.entry(hash).or_insert_with(|| {
                    let r = next_rank;
                    next_rank += 1;
                    r
                });
            }
        }
    }
    order
}

/// Reorders `pipelines` so entries present in `order` come first by bind rank while all remaining
/// entries keep their original manifest order. Returns how many records matched `order`.
fn apply_bind_order(pipelines: &mut [Record], order: &BindOrder) -> usize {
    if order.is_empty() {
        return 0;
    }
    let matched = pipelines.iter().filter(|r| order.rank(r).is_some()).count();
    pipelines.sort_by_key(|r| order.rank(r).unwrap_or(usize::MAX));
    matched
}

/// Discovers an implicit bind order file when `--bind-order` was not passed.
fn default_bind_order_path(cache: &Path, manifest: Option<&Path>) -> Option<PathBuf> {
    let mut candidates = vec![cache.join("bind-order.txt")];
    if let Some(m) = manifest {
        candidates.push(m.with_extension("bind-order"));
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn map_stage(stage: decant_shader::api::Stage) -> StageKind {
    match stage {
        decant_shader::api::Stage::Vertex => StageKind::Vertex,
        decant_shader::api::Stage::Pixel => StageKind::Pixel,
        decant_shader::api::Stage::Domain => StageKind::Domain,
        decant_shader::api::Stage::Hull => StageKind::Hull,
        decant_shader::api::Stage::Geometry => StageKind::Geometry,
        decant_shader::api::Stage::Compute => StageKind::Compute,
        decant_shader::api::Stage::Amplification => StageKind::Amplification,
        decant_shader::api::Stage::Mesh => StageKind::Mesh,
    }
}

fn map_class(class: decant_shader::api::RegisterClass) -> RegisterClass {
    match class {
        decant_shader::api::RegisterClass::Srv => RegisterClass::Srv,
        decant_shader::api::RegisterClass::Uav => RegisterClass::Uav,
        decant_shader::api::RegisterClass::Cbv => RegisterClass::Cbv,
        decant_shader::api::RegisterClass::Sampler => RegisterClass::Sampler,
    }
}

/// Synthesizes a PC `RootLayout` for a standalone corpus container from its declared user data,
/// declared resources and the layer's mirror and heap views.
fn corpus_root_layout(program: &decant_shader::api::StageProgram) -> RootLayout {
    use decant_shader::api::{Stage, UserDataKind};
    let mut params = Vec::new();
    let mut slots = program.user_data.clone();
    slots.sort_by_key(|u| u.sgpr);
    let mut next = if program.stage == Stage::Compute { 4u32 } else { 8u32 };
    for slot in &slots {
        let target = u32::from(slot.sgpr);
        if target > next {
            params.push(RootParam {
                kind: RootParamKind::Constants { register: 500 + next, space: 9, count: target - next },
                visibility: 0,
            });
            next = target;
        }
        if target < next {
            continue;
        }
        match slot.kind {
            UserDataKind::Table | UserDataKind::ResourceHeap | UserDataKind::SamplerHeap => {
                params.push(RootParam { kind: RootParamKind::Table(Vec::new()), visibility: 0 });
                next += 1;
            }
            UserDataKind::Constants { register, count } => {
                let step = count.max(1);
                params.push(RootParam { kind: RootParamKind::Constants { register, space: 0, count: step }, visibility: 0 });
                next += step;
            }
            UserDataKind::RootSrv { register } => {
                let space = program
                    .resources
                    .iter()
                    .find(|r| r.class == decant_shader::api::RegisterClass::Srv && (r.lower..=r.upper).contains(&register))
                    .map_or(0, |r| r.space);
                params.push(RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register, space }, visibility: 0 });
                next += 2;
            }
        }
    }
    if !program.resources.is_empty() {
        if next < 32 {
            params.push(RootParam {
                kind: RootParamKind::Constants { register: 600, space: 9, count: 32 - next },
                visibility: 0,
            });
        }
        let mut ranges = Vec::with_capacity(program.resources.len());
        let mut offset = 0u32;
        for r in &program.resources {
            let count = if r.upper == u32::MAX { u32::MAX } else { r.upper.saturating_sub(r.lower).saturating_add(1) };
            ranges.push(RootRange { class: map_class(r.class), count, base: r.lower, space: r.space, flags: 0, offset });
            offset = if count == u32::MAX || offset == u32::MAX { u32::MAX } else { offset.saturating_add(count) };
        }
        params.push(RootParam { kind: RootParamKind::Table(ranges), visibility: 0 });
    }
    params.push(RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 });
    params.push(RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 15 }, visibility: 0 });
    params.push(RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 1, space: 15 }, visibility: 0 });
    RootLayout { version: 2, flags: 0xc01, params, static_samplers: Vec::new() }
}

/// Builds a manifest directly from `--shaders` and `--dump-dir` when no `--manifest` file exists.
/// If neither a bind order nor a hash filter requires later entries, stops after `limit` records so
/// memory and startup stay small.
fn build_corpus_manifest(
    shaders: &[PathBuf],
    dump_dir: Option<&Path>,
    allow: Option<&HashSet<u64>>,
    limit: Option<usize>,
    log: impl Fn(&str),
) -> (Manifest, Option<Containers>) {
    let mut manifest = Manifest::default();
    let mut seen_hashes = HashSet::new();
    let mut kept_bytes: Option<HashMap<u64, Arc<[u8]>>> = limit.map(HashMap::with_capacity);

    let mut accept = |bytes: Vec<u8>| -> bool {
        let hash = decant_shader::container::fnv1a64(&bytes);
        if let Some(set) = allow
            && !set.contains(&hash)
        {
            return false;
        }
        if !seen_hashes.insert(hash) {
            return false;
        }
        let name = format!("{hash:016x}");
        let (kind, layout) = match decant_shader::container::program(&bytes, &name) {
            Ok(p) => (map_stage(p.stage), corpus_root_layout(&p)),
            Err(_) => (
                StageKind::Compute,
                RootLayout {
                    version: 2,
                    flags: 0xc01,
                    params: vec![RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 }],
                    static_samplers: Vec::new(),
                },
            ),
        };
        let root_id = pipeline_manifest::root_id(&pipeline_manifest::root_text(&layout));
        manifest.roots.entry(root_id).or_insert(layout);
        let linkage = Linkage {
            stages: vec![kind],
            targets: if kind == StageKind::Pixel { vec![28] } else { Vec::new() },
            topology_type: if matches!(kind, StageKind::Vertex | StageKind::Pixel | StageKind::Geometry | StageKind::Mesh) { 3 } else { 0 },
            sample_count: u32::from(kind != StageKind::Compute),
            ..Linkage::default()
        };
        manifest.pipelines.push(Record { pc_state: None, root: Some(root_id), linkage, stages: vec![(kind, hash)] });
        if let Some(map) = &mut kept_bytes {
            map.insert(hash, Arc::from(bytes));
        }
        limit.is_some_and(|cap| manifest.pipelines.len() >= cap)
    };

    let mut reached_limit = false;
    'outer: for archive in shaders {
        let items = match decant_extract_shaders::find_shaders(archive) {
            Ok(i) => i,
            Err(e) => {
                log(&format!("decant_shader_warm: {} not scanned ({e})", archive.display()));
                continue;
            }
        };
        for item in &items {
            if accept(item.container().to_vec()) {
                reached_limit = true;
                break 'outer;
            }
        }
    }

    if !reached_limit
        && let Some(dir) = dump_dir
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().and_then(|e| e.to_str()) == Some("bin")).collect();
        paths.sort();
        for path in paths {
            let Ok(bytes) = std::fs::read(&path) else { continue };
            if accept(bytes) {
                break;
            }
        }
    }

    (manifest, kept_bytes.map(|found| Containers { found }))
}

/// Whether a refusal key marks an unexpected tool, compiler or crash failure rather than a known
/// unsupported shader feature.
fn is_hard_failure_reason(reason: &str) -> bool {
    matches!(
        reason,
        "warm:stage-bytes-missing" | "dxc error" | "translator panic" | "adapter:translator-panic" | "adapter:no-thread" | "no compile worker" | "translator stage count" | "adapter:stage-count"
    )
}

/// Total hard failures across timeouts, DXC compile errors, missing stage bytes and panics.
fn count_failures(pending_timeout: u64, refused: &BTreeMap<String, u64>) -> u64 {
    let hard_refused: u64 = refused.iter().filter(|(k, _)| is_hard_failure_reason(k)).map(|(_, &n)| n).sum();
    pending_timeout + hard_refused
}

/// Summary and breakdown produced at the end of a warm run.
struct Report {
    total: usize,
    ready: u64,
    hits: u64,
    stages_compiled: u64,
    stage_cache_hits: u64,
    pending_timeout: u64,
    refused: BTreeMap<String, u64>,
    failures: u64,
    dxc_errors: u64,
    dxc_classes: BTreeMap<String, u64>,
    seconds: f64,
    cache_bytes: u64,
}

impl Report {
    fn refused_total(&self) -> u64 {
        self.refused.values().sum()
    }

    fn summary_line(&self) -> String {
        format!(
            "summary: pipelines={} ready={} hits={} stages_compiled={} stage_cache_hits={} refused={} failures={} seconds={:.2} cache_bytes={}",
            self.total,
            self.ready,
            self.hits,
            self.stages_compiled,
            self.stage_cache_hits,
            self.refused_total(),
            self.failures,
            self.seconds,
            self.cache_bytes,
        )
    }

    fn render(&self) -> String {
        let mut out = String::new();
        out += &format!("pipelines total {}\n", self.total);
        out += &format!("pipelines ready {}\n", self.ready);
        out += &format!("pipelines cache hits {}\n", self.hits);
        out += &format!("stages compiled {}\n", self.stages_compiled);
        out += &format!("stage cache hits {}\n", self.stage_cache_hits);
        out += &format!("pipelines refused {}\n", self.refused_total());
        out += &format!("failures {}\n", self.failures);
        out += &format!("pipelines pending after timeout {}\n", self.pending_timeout);
        out += "refused by key:\n";
        for (k, n) in &self.refused {
            out += &format!("  {k}: {n}\n");
        }
        out += &format!("DXC errors: {}\n", self.dxc_errors);
        out += "DXC errors by class:\n";
        for (k, n) in &self.dxc_classes {
            out += &format!("  {k}: {n}\n");
        }
        out += &format!("seconds: {:.2}\n", self.seconds);
        out += &format!("cache size: {} bytes\n", self.cache_bytes);
        out += &self.summary_line();
        out += "\n";
        out
    }
}

/// Extracts the first `error:` line of a DXC message and replaces a quoted identifier so repeats
/// of the same mistake group into one class.
fn classify_dxc_message(msg: &str) -> String {
    let line = msg.lines().find(|l| l.contains("error:")).unwrap_or_else(|| msg.lines().next().unwrap_or("(empty)"));
    let at = line.find("error:").map(|i| i + "error:".len()).unwrap_or(0);
    let text = line[at..].trim();
    match (text.find('\''), text.rfind('\'')) {
        (Some(a), Some(b)) if b > a => format!("{}<name>{}", text[..a].trim_end(), text[b + 1..].trim_start()),
        _ => text.to_string(),
    }
}

/// Reads every `.txt` DXC error file written at or after `since` and groups them by error class.
/// Translator `.refused.txt` files and dumps from earlier runs in a shared folder are skipped.
fn dxc_classes_since(dir: &Path, since: SystemTime) -> BTreeMap<String, u64> {
    let mut classes = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return classes };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.ends_with(".refused.txt")) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Ok(modified) = meta.modified() else { continue };
        if modified < since {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let message = text.split_once("\n\n").map(|(_, m)| m).unwrap_or(&text);
        *classes.entry(classify_dxc_message(message)).or_default() += 1;
    }
    classes
}

fn log(text: &str) {
    eprintln!("{text}");
}

fn quiet_adapter_log(text: &str) {
    if text.contains("DXC refused") || text.contains("cache write") || text.contains("not started") || text.contains("direct shaders summary:") {
        eprintln!("{text}");
    }
}

/// Builds the `Pipeline` of one manifest `Record`. Uses an empty `RootLayout` when `record.root`
/// is `None` so the adapter refuses it under `adapter:no-mirror`.
fn pipeline_of(record: &Record, manifest: &Manifest, bytes: &Containers) -> Result<Pipeline, Refusal> {
    let root = record
        .root
        .and_then(|id| manifest.roots.get(&id))
        .cloned()
        .unwrap_or(RootLayout { version: 0, flags: 0, params: Vec::new(), static_samplers: Vec::new() });
    let mut stages = Vec::with_capacity(record.stages.len());
    for (kind, hash) in &record.stages {
        let Some(found) = bytes.found.get(hash) else {
            return Err(Refusal::new("warm:stage-bytes-missing", format!("{} {hash:016x}", kind.short())));
        };
        stages.push(StageInput { kind: *kind, bytes: found.clone() });
    }
    Ok(Pipeline { stages, root, linkage: record.linkage.clone() })
}

/// Parses `--hash-list` contents (one 16 hex digit hash per line, blank and bad lines ignored).
fn parse_hash_list(text: &str) -> HashSet<u64> {
    text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).collect()
}

const MAX_THREADS: usize = 32;

fn default_threads() -> usize {
    std::thread::available_parallelism().map(usize::from).unwrap_or(4).clamp(4, 16)
}

fn effective_threads(requested: usize) -> usize {
    let threads = requested.max(1);
    if threads > MAX_THREADS {
        log(&format!("decant_shader_warm: capping threads from {threads} to {MAX_THREADS} (maximum worker count)"));
        return MAX_THREADS;
    }
    threads
}

fn format_progress_line(done: usize, total: usize, ready: u64, compiled: u64, cache_hits: u64, failed: u64) -> String {
    format!("PROGRESS {done}/{total} ready={ready} compiled={compiled} cache_hits={cache_hits} failed={failed}")
}

fn emit_progress(done: usize, total: usize, ready: u64, compiled: u64, cache_hits: u64, failed: u64) {
    use std::io::Write as _;
    let line = format_progress_line(done, total, ready, compiled, cache_hits, failed);
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// A pointer free description prepared without a device.
#[cfg(any(windows, test))]
fn prepare_pso(record: &Record, root: &RootLayout) -> Result<Vec<u8>, String> {
    if !root.static_samplers.is_empty() {
        return Err("root signature has static sampler bindings without descriptions".into());
    }
    let kinds: Vec<_> = record.stages.iter().map(|s| s.0).collect();
    if kinds != record.linkage.stages {
        return Err("stage linkage differs from the recorded stages".into());
    }
    let compute = kinds == [StageKind::Compute];
    if !compute && (kinds.first() != Some(&StageKind::Vertex)
        || kinds.iter().any(|k| matches!(k, StageKind::Compute | StageKind::Mesh | StageKind::Amplification))
        || kinds.iter().collect::<HashSet<_>>().len() != kinds.len()) {
        return Err("unsupported graphics stage combination".into());
    }
    if record.linkage.stream_output {
        return Err("stream output is not supported by the PSO cache".into());
    }
    let desc = match &record.pc_state {
        Some(state) => state.description().to_vec(),
        None if compute => vec![0; 0x38],
        None => return Err("graphics pipeline needs version 2 full state".into()),
    };
    if desc.len() != if compute { 0x38 } else { 0x290 } {
        return Err("PC state does not match the pipeline kind".into());
    }
    if !compute {
        let n = |at| u32::from_le_bytes(desc[at..at + 4].try_into().unwrap_or_default());
        let l = &record.linkage;
        if l.targets.len() > 8 || l.inputs.len() > 32 || l.sample_count == 0
            || n(0x230) as usize != l.inputs.len() || n(0x240) as usize != l.targets.len()
            || n(0x23c) != l.topology_type || n(0x264) != l.depth_format || n(0x268) != l.sample_count
            || l.targets.iter().enumerate().any(|(i, f)| n(0x244 + 4 * i) != *f) {
            return Err("PC graphics state differs from linkage".into());
        }
    }
    Ok(desc)
}

#[derive(Default)]
struct PsoReport {
    warm: AtomicU64,
    hits: AtomicU64,
    stored: AtomicU64,
    failed: AtomicU64,
    skipped: AtomicU64,
}

impl PsoReport {
    fn line(&self) -> String {
        format!("pso_warm={} pso_hits={} pso_stored={} pso_failed={} pso_skipped={}",
            self.warm.load(Ordering::Relaxed), self.hits.load(Ordering::Relaxed),
            self.stored.load(Ordering::Relaxed), self.failed.load(Ordering::Relaxed), self.skipped.load(Ordering::Relaxed))
    }
}

#[cfg(not(windows))]
fn warm_psos(args: &Args, records: &[Record], _: &Manifest, _: &Containers, _: &Service, _: usize) -> PsoReport {
    skip_psos(args, records.len(), "headless D3D12 PSO warming requires Windows")
}

fn skip_psos(args: &Args, total: usize, reason: &str) -> PsoReport {
    log(&format!("Stage 2 skipped: {reason}"));
    let report = PsoReport::default();
    report.skipped.store(total as u64, Ordering::Relaxed);
    if args.progress { println!("PROGRESS_PSO {total}/{total} {}", report.line()); }
    report
}

/// All native calls use the system runtime and owned buffers. No command queue is needed.
#[cfg(windows)]
#[allow(unsafe_code)]
mod offline_pso {
    use super::*;
    use std::ffi::{CStr, CString, c_void};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIDevice, IDXGIAdapter1, IDXGIFactory4};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    use windows::core::{GUID, Interface, PCSTR};
    use d3d12_xs::pso_library::{self, Library};

    type CreateDevice = unsafe extern "system" fn(*mut c_void, u32, *const GUID, *mut *mut c_void) -> i32;
    type Serialize = unsafe extern "system" fn(*const D3D12_VERSIONED_ROOT_SIGNATURE_DESC, *mut *mut c_void, *mut *mut c_void) -> i32;

    fn export(name: &CStr) -> Option<usize> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        let path: Vec<u16> = PathBuf::from(std::env::var_os("SystemRoot")?).join("System32/d3d12.dll")
            .as_os_str().to_string_lossy().encode_utf16().chain(Some(0)).collect();
        // SAFETY: both names are terminated. The system module stays loaded for this process.
        unsafe {
            let module = LoadLibraryW(path.as_ptr());
            if module.is_null() { return None }
            GetProcAddress(module, name.as_ptr().cast()).map(|f| f as usize)
        }
    }

    fn device() -> Option<ID3D12Device> {
        let create = export(c"D3D12CreateDevice")?;
        // SAFETY: the export has the declared signature. COM outputs are owned references.
        unsafe {
            let create: CreateDevice = std::mem::transmute(create);
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let mut i = 0;
            while let Ok(adapter) = factory.EnumAdapters1(i) {
                i += 1;
                let Ok(desc) = adapter.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 { continue }
                let mut raw = std::ptr::null_mut();
                if create(adapter.as_raw(), D3D_FEATURE_LEVEL_12_0.0 as u32, &ID3D12Device::IID, &mut raw) >= 0 && !raw.is_null() {
                    return Some(ID3D12Device::from_raw(raw));
                }
            }
        }
        None
    }

    fn identity(device: &ID3D12Device) -> Option<pso_library::AdapterIdentity> {
        // SAFETY: these queries borrow live COM objects and return owned adapter references.
        unsafe {
            let luid = device.GetAdapterLuid();
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).ok()?;
            let desc = adapter.GetDesc1().ok()?;
            Some(pso_library::AdapterIdentity {
                luid: u64::from(luid.LowPart) | (u64::from(luid.HighPart as u32) << 32),
                vendor_id: desc.VendorId, device_id: desc.DeviceId, subsys_id: desc.SubSysId,
                revision: desc.Revision, driver_version: adapter.CheckInterfaceSupport(&IDXGIDevice::IID).ok()? as u64,
            })
        }
    }

    fn root_signature(device: &ID3D12Device, layout: &RootLayout) -> Result<ID3D12RootSignature, String> {
        if !layout.static_samplers.is_empty() { return Err("static sampler descriptions unavailable".into()) }
        let ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE1>> = layout.params.iter().map(|p| {
            if let RootParamKind::Table(ranges) = &p.kind {
                ranges.iter().map(|r| D3D12_DESCRIPTOR_RANGE1 {
                    RangeType: D3D12_DESCRIPTOR_RANGE_TYPE(match r.class { RegisterClass::Srv => 0, RegisterClass::Uav => 1, RegisterClass::Cbv => 2, RegisterClass::Sampler => 3 }),
                    NumDescriptors: r.count, BaseShaderRegister: r.base, RegisterSpace: r.space,
                    Flags: D3D12_DESCRIPTOR_RANGE_FLAGS(r.flags as i32), OffsetInDescriptorsFromTableStart: r.offset,
                }).collect()
            } else { Vec::new() }
        }).collect();
        let params: Vec<_> = layout.params.iter().zip(&ranges).map(|(p, ranges)| {
            let (kind, data) = match &p.kind {
                RootParamKind::Table(_) => (0, D3D12_ROOT_PARAMETER1_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: ranges.len() as u32, pDescriptorRanges: ranges.as_ptr() } }),
                RootParamKind::Constants { register, space, count } => (1, D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: *register, RegisterSpace: *space, Num32BitValues: *count } }),
                RootParamKind::Descriptor { class, register, space } => (
                    match class { RegisterClass::Cbv => 2, RegisterClass::Srv => 3, RegisterClass::Uav => 4, RegisterClass::Sampler => 4 },
                    D3D12_ROOT_PARAMETER1_0 { Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: *register, RegisterSpace: *space, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE } }),
            };
            D3D12_ROOT_PARAMETER1 { ParameterType: D3D12_ROOT_PARAMETER_TYPE(kind), Anonymous: data, ShaderVisibility: D3D12_SHADER_VISIBILITY(p.visibility as i32) }
        }).collect();
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC { Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 { Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32, pParameters: params.as_ptr(), NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(), Flags: D3D12_ROOT_SIGNATURE_FLAGS(layout.flags as i32),
            } } };
        let serialize = export(c"D3D12SerializeVersionedRootSignature").ok_or("system root signature serializer unavailable")?;
        let (mut raw, mut errors) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: all descriptions and their nested arrays live through serialization.
        unsafe {
            let serialize: Serialize = std::mem::transmute(serialize);
            let hr = serialize(&desc, &mut raw, &mut errors);
            let error = if errors.is_null() { None } else { Some(ID3DBlob::from_raw(errors)) };
            let blob = if raw.is_null() { None } else { Some(ID3DBlob::from_raw(raw)) };
            if hr < 0 { return Err(format!("root signature serialization failed {hr:#x}")) }
            drop(error);
            let blob = blob.ok_or("root signature serializer returned no blob")?;
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            device.CreateRootSignature(0, bytes).map_err(|e| e.to_string())
        }
    }

    fn make(device: &ID3D12Device, root: &ID3D12RootSignature, record: &Record, desc: &[u8], stages: &[Arc<[u8]>], blob: Option<&[u8]>) -> Result<ID3D12PipelineState, String> {
        let names: Vec<_> = record.linkage.inputs.iter().map(|e| CString::new(e.semantic.as_str()).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
        let inputs: Vec<_> = record.linkage.inputs.iter().zip(&names).map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(name.as_ptr().cast()), SemanticIndex: e.index, Format: DXGI_FORMAT(e.format as i32),
            InputSlot: e.slot, AlignedByteOffset: e.offset, InputSlotClass: D3D12_INPUT_CLASSIFICATION(i32::from(e.per_instance)), InstanceDataStepRate: e.step_rate,
        }).collect();
        let mut bytes = desc.to_vec();
        bytes[..8].copy_from_slice(&(root.as_raw() as u64).to_le_bytes());
        for ((kind, _), stage) in record.stages.iter().zip(stages) {
            let at = match kind { StageKind::Vertex | StageKind::Compute => 8, StageKind::Pixel => 24, StageKind::Domain => 40, StageKind::Hull => 56, StageKind::Geometry => 72, _ => return Err("unsupported PSO stage".into()) };
            bytes[at..at + 8].copy_from_slice(&(stage.as_ptr() as u64).to_le_bytes());
            bytes[at + 8..at + 16].copy_from_slice(&(stage.len() as u64).to_le_bytes());
        }
        if bytes.len() == 0x290 {
            bytes[0x228..0x230].copy_from_slice(&(inputs.as_ptr() as u64).to_le_bytes());
            let graphics: &mut [u8; 0x290] = bytes.as_mut_slice().try_into().map_err(|_| "graphics description size")?;
            d3d12_xs::stream_pso::valid_states(graphics);
        }
        if let Some(blob) = blob {
            let at = pso_library::cached_field(bytes.len()).ok_or("description size")?;
            bytes[at..at + 8].copy_from_slice(&(blob.as_ptr() as u64).to_le_bytes());
            bytes[at + 8..at + 16].copy_from_slice(&(blob.len() as u64).to_le_bytes());
        }
        let aligned: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
        // SAFETY: the aligned public description has live root, shader, input and blob pointers.
        // It is borrowed only. The manually dropped root field owns no additional reference.
        unsafe {
            if bytes.len() == 0x38 {
                device.CreateComputePipelineState(&*aligned.as_ptr().cast::<D3D12_COMPUTE_PIPELINE_STATE_DESC>())
            } else {
                device.CreateGraphicsPipelineState(&*aligned.as_ptr().cast::<D3D12_GRAPHICS_PIPELINE_STATE_DESC>())
            }.map_err(|e| e.to_string())
        }
    }

    /// Tells the translator the wave size of the adapter the game will use, so that the cache keys
    /// written here match the ones the layer asks for.
    pub(super) fn note_waves() {
        match device() {
            Some(device) => d3d12_xs::direct_source::note_device_waves(&device),
            None => log("decant_shader_warm: no D3D12 hardware adapter, translating for the native wave size"),
        }
    }

    pub(super) fn run(args: &Args, records: &[Record], manifest: &Manifest, containers: &Containers, service: &Service, threads: usize) -> PsoReport {
        let report = PsoReport::default();
        let Some(device) = device() else {
            return skip_psos(args, records.len(), "no D3D12 hardware adapter is present");
        };
        let Some(folder) = args.pso_cache.clone().or_else(pso_library::configured_root) else {
            return skip_psos(args, records.len(), "the PSO disk cache is disabled");
        };
        let library = Library::new(folder, env!("DECANT_DIRECT_TRANSLATOR_HASH"));
        if library.open(|| identity(&device)).is_none() {
            return skip_psos(args, records.len(), "adapter or driver identity unavailable");
        }
        let cursor = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let progress_lock = std::sync::Mutex::new(());
        let emit = |n| {
            if args.progress { println!("PROGRESS_PSO {n}/{} {}", records.len(), report.line()); }
        };
        emit(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(record) = records.get(i) else { break };
                        let prepared = record.root.and_then(|id| manifest.roots.get(&id)).ok_or_else(|| "unknown root signature".to_string())
                            .and_then(|root| prepare_pso(record, root).map(|desc| (root, desc)));
                        match prepared {
                            Err(e) => { report.skipped.fetch_add(1, Ordering::Relaxed); log(&format!("Stage 2 pipeline {} skipped: {e}", i + 1)); }
                            Ok((layout, desc)) => {
                                let result = (|| {
                                    let pipeline = pipeline_of(record, manifest, containers).map_err(|e| e.to_string())?;
                                    let identity = d3d12_xs::direct_cache::pipeline_key(&pipeline, env!("DECANT_DIRECT_TRANSLATOR_HASH"), "");
                                    let Poll::Ready(stages) = service.poll(i + 1, || Ok(pipeline)) else { return Err("compiled DXIL unavailable".into()) };
                                    if stages.len() != record.stages.len() { return Err("compiled stage count differs from manifest".into()) }
                                    let stage_refs: Vec<_> = stages.iter().map(|s| &s[..]).collect();
                                    let key = pso_library::pipeline_key(&desc, &stage_refs, identity).ok_or("unsupported cache key")?;
                                    let root = root_signature(&device, layout)?;
                                    let cached = library.lookup(key);
                                    let made = match make(&device, &root, record, &desc, &stages, cached.as_deref()) {
                                        Ok(pso) => { if cached.is_some() { report.hits.fetch_add(1, Ordering::Relaxed); } pso }
                                        Err(_) if cached.is_some() => { library.reject(key); make(&device, &root, record, &desc, &stages, None)? }
                                        Err(e) => return Err(e),
                                    };
                                    report.warm.fetch_add(1, Ordering::Relaxed);
                                    let blob = pso_library::cached_blob(made.as_raw()).ok_or("driver returned no cached blob")?;
                                    library.keep(key, &blob).map_err(|e| e.to_string())?;
                                    report.stored.fetch_max(library.counts().stored.load(Ordering::Relaxed), Ordering::Relaxed);
                                    Ok::<_, String>(())
                                })();
                                if let Err(e) = result { report.failed.fetch_add(1, Ordering::Relaxed); log(&format!("Stage 2 pipeline {} failed: {e}", i + 1)); }
                            }
                        }
                        let _progress = progress_lock.lock().unwrap_or_else(|e| e.into_inner());
                        emit(done.fetch_add(1, Ordering::Relaxed) + 1);
                    }
                });
            }
        });
        report
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hardware_compute_blob_matches_the_shared_cache() {
            let Some(device) = device() else { println!("skipped: no D3D12 hardware adapter"); return };
            let Some(adapter) = identity(&device) else { println!("skipped: no driver identity"); return };
            let Ok(dxc) = decant_dxc::Dxc::load() else { println!("skipped: no DXC"); return };
            let shader = dxc.compile("groupshared uint data[2]; [numthreads(2,1,1)] void main(uint i : SV_GroupIndex) { data[i] = i; }", "main", "cs_6_0", &[]).unwrap();
            let layout = RootLayout { version: 2, flags: 0, params: vec![], static_samplers: vec![] };
            let record = pipeline_manifest::parse_pipeline("P - topo=0 msaa=0 ds=0 rt= so=0 in= CS 0000000000000001").unwrap();
            let desc = prepare_pso(&record, &layout).unwrap();
            let root = root_signature(&device, &layout).unwrap();
            let stages: Vec<Arc<[u8]>> = vec![Arc::from(shader)];
            let made = make(&device, &root, &record, &desc, &stages, None).unwrap();
            let Some(blob) = pso_library::cached_blob(made.as_raw()) else { println!("skipped: driver returned no cached blob"); return };
            let key = pso_library::pipeline_key(&desc, &[&stages[0]], 123).unwrap();
            let dir = crate::tests::temp_dir("offline-pso");
            let library = Library::new(dir.clone(), env!("DECANT_DIRECT_TRANSLATOR_HASH"));
            library.open(|| Some(adapter)).unwrap();
            library.keep(key, &blob).unwrap();
            let reopened = Library::new(dir.clone(), env!("DECANT_DIRECT_TRANSLATOR_HASH"));
            reopened.open(|| Some(adapter)).unwrap();
            let cached = reopened.lookup(key).unwrap();
            assert_eq!(cached, blob);
            assert!(make(&device, &root, &record, &desc, &stages, Some(&cached)).is_ok());
            assert_eq!(reopened.counts().hits.load(Ordering::Relaxed), 1);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

}

#[cfg(windows)]
use offline_pso::run as warm_psos;

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    if args.top == Some(0) {
        log("BLOCKED: --top must be at least 1");
        return std::process::ExitCode::FAILURE;
    }
    let run_started = SystemTime::now();
    env_setup::set(&args.cache);
    let cache_dir = DiskCache::configured_dir().unwrap_or_else(|| args.cache.clone());

    let allow: Option<HashSet<u64>> = match &args.hash_list {
        None => None,
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => {
                let set = parse_hash_list(&text);
                if set.is_empty() {
                    log(&format!("decant_shader_warm: --hash-list {} is empty: selecting nothing", path.display()));
                }
                Some(set)
            }
            Err(e) => {
                log(&format!("BLOCKED: reading --hash-list {}: {e}", path.display()));
                return std::process::ExitCode::FAILURE;
            }
        },
    };

    let bind_order_path = args.bind_order.clone().or_else(|| default_bind_order_path(&cache_dir, args.manifest.as_deref()));
    let bind_order = match &bind_order_path {
        None => None,
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => Some((path.clone(), parse_bind_order(&text))),
            Err(e) => {
                log(&format!("BLOCKED: reading bind order {}: {e}", path.display()));
                return std::process::ExitCode::FAILURE;
            }
        },
    };

    let (manifest, preloaded) = match &args.manifest {
        Some(path) => match pipeline_manifest::read(path) {
            Ok(m) => (m, None),
            Err(e) => {
                log(&format!("BLOCKED: reading manifest {}: {e}", path.display()));
                return std::process::ExitCode::FAILURE;
            }
        },
        None => {
            if args.shaders.is_empty() && args.dump_dir.is_none() {
                log("BLOCKED: pass --manifest or at least one --shaders archive or --dump-dir");
                return std::process::ExitCode::FAILURE;
            }
            let early_cap = if bind_order.is_none() { args.top } else { None };
            println!("decant_shader_warm: corpus mode keys will not match the game keys without a manifest");
            build_corpus_manifest(&args.shaders, args.dump_dir.as_deref(), allow.as_ref(), early_cap, log)
        }
    };

    let mut pipelines: Vec<Record> = manifest.pipelines.clone();
    if let Some(allow) = &allow {
        pipelines.retain(|r| r.stages.iter().any(|(_, h)| allow.contains(h)));
    }
    if let Some((path, order)) = &bind_order {
        let matched = apply_bind_order(&mut pipelines, order);
        log(&format!("decant_shader_warm: bind order {} matched {matched} of {} pipelines", path.display(), pipelines.len()));
    }
    if let Some(n) = args.top {
        pipelines.truncate(n);
    }
    let total = pipelines.len();
    if total == 0 {
        log("BLOCKED: 0 pipelines selected");
        return std::process::ExitCode::FAILURE;
    }

    let wanted: HashSet<u64> = pipelines.iter().flat_map(|r| r.stages.iter().map(|(_, h)| *h)).collect();
    let containers = match preloaded {
        Some(c) if wanted.iter().all(|h| c.found.contains_key(h)) => c,
        _ => Containers::load(&wanted, &args.shaders, args.dump_dir.as_deref(), log),
    };
    log(&format!("decant_shader_warm: {} of {} stage hashes resolved for {total} pipelines", containers.found.len(), wanted.len()));

    let dxc = match decant_dxc::Dxc::load() {
        Ok(d) => {
            log(&format!("decant_shader_warm: {}", d.describe()));
            d
        }
        Err(e) => {
            log(&format!("BLOCKED: DXC did not load: {e}"));
            return std::process::ExitCode::FAILURE;
        }
    };

    let threads = effective_threads(args.threads);
    let poll_wait = Duration::from_millis(args.poll_ms.max(1) * 600);
    let adapter_log: fn(&str) = if args.verbose { log } else { quiet_adapter_log };
    #[cfg(windows)]
    offline_pso::note_waves();
    let source = Arc::new(DirectSource::with_wait(Arc::new(RegistryHost), adapter_log, args.dxc_dump.clone(), poll_wait));
    let compiler = Arc::new(DxcCompiler(dxc));
    let settings = Settings { threads, cache: Some(DiskCache::new(&args.cache)), dump: args.dxc_dump.clone(), log: adapter_log };
    let service = Arc::new(Service::new(source, compiler, settings));

    let cursor = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));
    let pipeline_hits = Arc::new(AtomicU64::new(0));
    let pending_timeout = Arc::new(AtomicU64::new(0));
    let pipelines = Arc::new(pipelines);
    let manifest = Arc::new(manifest);
    let containers = Arc::new(containers);

    let started = Instant::now();
    let progress_enabled = args.progress;
    let progress_step = (total / 50).clamp(1, 25);
    let last_progress_ms = Arc::new(AtomicU64::new(0));
    if progress_enabled {
        emit_progress(0, total, 0, 0, 0, 0);
    }
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let (cursor, completed, pipeline_hits, pending_timeout, last_progress_ms, pipelines, manifest, containers, service) = (
                cursor.clone(),
                completed.clone(),
                pipeline_hits.clone(),
                pending_timeout.clone(),
                last_progress_ms.clone(),
                pipelines.clone(),
                manifest.clone(),
                containers.clone(),
                service.clone(),
            );
            std::thread::spawn(move || {
                loop {
                    let i = cursor.fetch_add(1, Ordering::SeqCst);
                    if i >= pipelines.len() {
                        break;
                    }
                    let record = &pipelines[i];
                    let id = i + 1;
                    let first_poll = service.poll(id, || pipeline_of(record, &manifest, &containers));
                    if matches!(first_poll, Poll::Ready(_)) {
                        pipeline_hits.fetch_add(1, Ordering::Relaxed);
                    } else if first_poll == Poll::Pending {
                        let deadline = Instant::now() + poll_wait;
                        loop {
                            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                                pending_timeout.fetch_add(1, Ordering::Relaxed);
                                break;
                            };
                            if service.holds(id) {
                                if !service.wait(id, remaining) {
                                    pending_timeout.fetch_add(1, Ordering::Relaxed);
                                    break;
                                }
                            } else {
                                std::thread::sleep(Duration::from_millis(10).min(remaining));
                            }
                            let next_poll = service.poll(id, || pipeline_of(record, &manifest, &containers));
                            if next_poll != Poll::Pending {
                                break;
                            }
                        }
                    }
                    let done = completed.fetch_add(1, Ordering::AcqRel) + 1;
                    if progress_enabled && done < total {
                        let now_ms = started.elapsed().as_millis() as u64;
                        let prev_ms = last_progress_ms.load(Ordering::Relaxed);
                        let time_due = now_ms.saturating_sub(prev_ms) >= 250
                            && last_progress_ms.compare_exchange(prev_ms, now_ms, Ordering::Relaxed, Ordering::Relaxed).is_ok();
                        if done.is_multiple_of(progress_step) || time_due {
                            let c = service.counters();
                            let refused_map = c.refused();
                            let fail_now = count_failures(pending_timeout.load(Ordering::Relaxed), &refused_map);
                            emit_progress(
                                done,
                                total,
                                c.ready.load(Ordering::Relaxed),
                                c.compiled.load(Ordering::Relaxed),
                                c.cache_hits.load(Ordering::Relaxed),
                                fail_now,
                            );
                        }
                    }
                    if done.is_multiple_of(100) {
                        let c = service.counters();
                        let refused_map = c.refused();
                        let refused_now: u64 = refused_map.values().sum();
                        let fail_now = count_failures(pending_timeout.load(Ordering::Relaxed), &refused_map);
                        log(&format!(
                            "decant_shader_warm: progress {done}/{total} pipelines (ready={} hits={} stages_compiled={} stage_cache_hits={} refused={} failures={} {:.2}s)",
                            c.ready.load(Ordering::Relaxed),
                            pipeline_hits.load(Ordering::Relaxed),
                            c.compiled.load(Ordering::Relaxed),
                            c.cache_hits.load(Ordering::Relaxed),
                            refused_now,
                            fail_now,
                            started.elapsed().as_secs_f64(),
                        ));
                    }
                }
            })
        })
        .collect();
    for w in workers {
        let _ = w.join();
    }
    let elapsed = started.elapsed().as_secs_f64();
    if progress_enabled {
        let c = service.counters();
        let refused_map = c.refused();
        let fail_now = count_failures(pending_timeout.load(Ordering::Relaxed), &refused_map);
        emit_progress(
            total,
            total,
            c.ready.load(Ordering::Relaxed),
            c.compiled.load(Ordering::Relaxed),
            c.cache_hits.load(Ordering::Relaxed),
            fail_now,
        );
    }

    let cache_bytes = std::fs::read_dir(&cache_dir)
        .map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
        .unwrap_or(0);
    let counters = service.counters();
    let refused = counters.refused();
    let timeouts = pending_timeout.load(Ordering::Relaxed);
    let failures = count_failures(timeouts, &refused);
    let report = Report {
        total,
        ready: counters.ready.load(Ordering::Relaxed),
        hits: pipeline_hits.load(Ordering::Relaxed),
        stages_compiled: counters.compiled.load(Ordering::Relaxed),
        stage_cache_hits: counters.cache_hits.load(Ordering::Relaxed),
        pending_timeout: timeouts,
        refused,
        failures,
        dxc_errors: counters.dxc_errors.load(Ordering::Relaxed),
        dxc_classes: args.dxc_dump.as_deref().map(|d| dxc_classes_since(d, run_started)).unwrap_or_default(),
        seconds: elapsed,
        cache_bytes,
    };
    let pso_report = if args.warm_pso {
        warm_psos(&args, &pipelines, &manifest, &containers, &service, threads)
    } else { PsoReport::default() };
    let mut text = report.render();
    if args.warm_pso {
        text += &format!("PSO summary: {}\n", pso_report.line());
    }
    print!("{text}");
    if let Some(path) = &args.report
        && let Err(e) = std::fs::write(path, &text)
    {
        log(&format!("BLOCKED: report not written to {} ({e})", path.display()));
        return std::process::ExitCode::FAILURE;
    }
    if report.failures > 0 || report.ready == 0 || pso_report.failed.load(Ordering::Relaxed) > 0 {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pso_options_require_a_manifest_and_accept_an_optional_cache() {
        let args = Args::try_parse_from(["warm", "--cache", "dxil", "--manifest", "pipelines.txt", "--warm-pso", "--pso-cache", "driver", "--threads", "3", "--progress"]).unwrap();
        assert!(args.warm_pso && args.progress);
        assert_eq!(args.pso_cache, Some(PathBuf::from("driver")));
        assert_eq!(args.threads, 3);
        assert!(!Args::try_parse_from(["warm", "--cache", "dxil"]).unwrap().warm_pso);
        assert!(Args::try_parse_from(["warm", "--cache", "dxil", "--warm-pso"]).is_err());
        assert!(Args::try_parse_from(["warm", "--cache", "dxil", "--pso-cache", "driver"]).is_err());
        assert!(Args::try_parse_from(["warm", "--cache", "dxil", "--manifest", "p", "--warm-pso"]).unwrap().pso_cache.is_none());
    }

    #[test]
    fn offline_descriptions_require_complete_graphics_state() {
        let root = RootLayout { version: 2, flags: 0, params: vec![], static_samplers: vec![] };
        let mut record = pipeline_manifest::parse_pipeline("P - topo=0 msaa=0 ds=0 rt= so=0 in= CS 0000000000000001").unwrap();
        assert_eq!(prepare_pso(&record, &root).unwrap(), vec![0; 0x38]);
        let mut compute = vec![0; 0x38];
        compute[0x18] = 1;
        compute[0x30] = 4;
        record.pc_state = pipeline_manifest::PcState::capture(&compute);
        assert_eq!(prepare_pso(&record, &root).unwrap(), compute);
        let mut graphics = pipeline_manifest::parse_pipeline("P - topo=3 msaa=1 ds=40 rt=28 so=0 in=POSITION.0.6.0.0.v.0 VS 0000000000000001 PS 0000000000000002").unwrap();
        assert!(prepare_pso(&graphics, &root).unwrap_err().contains("version 2"));
        let mut desc = vec![0; 0x290];
        for (at, value) in [(0x230, 1u32), (0x23c, 3), (0x240, 1), (0x244, 28), (0x264, 40), (0x268, 1), (0x26c, 7), (0x288, 4), (0x1c0, 0x1234)] {
            desc[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        graphics.pc_state = pipeline_manifest::PcState::capture(&desc);
        assert_eq!(prepare_pso(&graphics, &root).unwrap(), desc);
        let stages: [&[u8]; 2] = [b"compiled vertex", b"compiled pixel"];
        let prepared = prepare_pso(&graphics, &root).unwrap();
        assert_eq!(d3d12_xs::pso_library::pipeline_key(&prepared, &stages, 8), d3d12_xs::pso_library::pipeline_key(&desc, &stages, 8));
        graphics.linkage.sample_count = 2;
        assert!(prepare_pso(&graphics, &root).is_err());
        record.pc_state = graphics.pc_state.clone();
        assert!(prepare_pso(&record, &root).is_err());
        record.pc_state = None;
        let mut sampler_root = root;
        sampler_root.static_samplers.push(d3d12_xs::rts0::StaticSampler { register: 0, space: 0, visibility: 0 });
        assert!(prepare_pso(&record, &sampler_root).unwrap_err().contains("sampler"));
    }

    #[test]
    fn an_empty_list_selects_nothing() {
        assert!(parse_hash_list("").is_empty());
    }

    #[test]
    fn hex_lines_parse_and_bad_lines_are_skipped() {
        let text = "0000000000000001\nnot-hex\n\n  00000000deadbeef  \n";
        let set = parse_hash_list(text);
        assert_eq!(set, HashSet::from([0x1, 0xdead_beef]));
    }

    pub(super) fn temp_dir(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!("decant-shader-warm-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn an_earlier_runs_dump_is_not_counted() {
        let dir = temp_dir("earlier-run");
        std::fs::write(dir.join("old.ps.txt"), "header\n\nerror: 'foo' is old\nsecond line").unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let since = SystemTime::now();
        std::thread::sleep(Duration::from_millis(50));

        assert!(dxc_classes_since(&dir, since).is_empty());

        std::fs::write(dir.join("new.ps.txt"), "header\n\nerror: 'bar' is new\nsecond line").unwrap();
        let classes = dxc_classes_since(&dir, since);
        assert_eq!(classes.len(), 1, "{classes:?}");
        assert_eq!(classes.values().next(), Some(&1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_refused_txt_from_this_run_is_still_not_a_dxc_error() {
        let dir = temp_dir("refused");
        let since = SystemTime::now();
        std::thread::sleep(Duration::from_millis(10));
        std::fs::write(dir.join("vertex.refused.txt"), "header\n\nerror: 'x' refused by the translator").unwrap();
        assert!(dxc_classes_since(&dir, since).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bind_order_ranks_stage_lines_and_keeps_manifest_order_for_the_rest() {
        let rec = |stages: &[(StageKind, u64)]| Record {
            pc_state: None,
            root: None,
            linkage: Linkage { stages: stages.iter().map(|(k, _)| *k).collect(), ..Linkage::default() },
            stages: stages.to_vec(),
        };
        let p0 = rec(&[(StageKind::Compute, 0x10)]);
        let p1 = rec(&[(StageKind::Vertex, 0x20), (StageKind::Pixel, 0x21)]);
        let p2 = rec(&[(StageKind::Compute, 0x30)]);
        let p3 = rec(&[(StageKind::Mesh, 0x40), (StageKind::Pixel, 0x41)]);

        let text = concat!(
            "# comment\n",
            "  18.152 debug d3d12_xs: direct shaders: translated ms 0000000000000040, ps 0000000000000041\n",
            "0000000000000021\n",
            "VS 0000000000000020 PS 0000000000000021\n",
        );
        let order = parse_bind_order(text);
        let mut list = vec![p0.clone(), p1.clone(), p2.clone(), p3.clone()];
        let matched = apply_bind_order(&mut list, &order);
        assert_eq!(matched, 2);
        assert_eq!(list, vec![p3, p1, p0, p2]);
    }

    #[test]
    fn failure_classification_separates_translator_refusals_from_hard_failures() {
        let mut refused = BTreeMap::new();
        refused.insert("op:v_readlane_b32_e64".to_string(), 3);
        refused.insert("resource:untraced-descriptor".to_string(), 2);
        assert_eq!(count_failures(0, &refused), 0);

        refused.insert("warm:stage-bytes-missing".to_string(), 1);
        assert_eq!(count_failures(0, &refused), 1);

        refused.insert("dxc error".to_string(), 2);
        assert_eq!(count_failures(4, &refused), 7);
    }

    #[test]
    fn thread_count_defaults_and_clamps_sensibly() {
        assert!(default_threads() >= 4);
        assert_eq!(effective_threads(0), 1);
        assert_eq!(effective_threads(64), MAX_THREADS);
    }

    #[test]
    fn progress_line_formats_structured_fields() {
        assert_eq!(
            format_progress_line(12, 40, 10, 4, 8, 2),
            "PROGRESS 12/40 ready=10 compiled=4 cache_hits=8 failed=2"
        );
    }
}
