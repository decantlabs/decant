//! `decant_shader_warm` fills the direct shader disk cache (`direct_cache`) from a pipeline manifest
//! (`pipeline_manifest`, `DECANT_D3D12X_PIPELINE_MANIFEST`) or directly from a shader archive so the
//! first bind of a warmed pipeline in a later process is a file read. It reuses the translator
//! (`decant-shader` via `DirectSource`), compiler (`decant-dxc::Dxc`) and disk cache
//! (`direct_cache::DiskCache`) through `compile_service::Service`, the same path the layer runs.
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
    #[arg(long, default_value_t = 4)]
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
        manifest.pipelines.push(Record { root: Some(root_id), linkage, stages: vec![(kind, hash)] });
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

fn effective_threads(requested: usize) -> usize {
    let threads = requested.max(1);
    if threads > 4 {
        log(&format!("decant_shader_warm: capping threads from {threads} to 4 (maximum worker count)"));
        return 4;
    }
    threads
}

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
    let workers: Vec<_> = (0..threads)
        .map(|_| {
            let (cursor, completed, pipeline_hits, pending_timeout, pipelines, manifest, containers, service) = (
                cursor.clone(),
                completed.clone(),
                pipeline_hits.clone(),
                pending_timeout.clone(),
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
    let text = report.render();
    print!("{text}");
    if let Some(path) = &args.report
        && let Err(e) = std::fs::write(path, &text)
    {
        log(&format!("BLOCKED: report not written to {} ({e})", path.display()));
        return std::process::ExitCode::FAILURE;
    }
    if report.failures > 0 || report.ready == 0 {
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn temp_dir(tag: &str) -> PathBuf {
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
    fn thread_count_is_capped_at_four_without_a_game_lock() {
        assert_eq!(effective_threads(64), 4);
    }
}
