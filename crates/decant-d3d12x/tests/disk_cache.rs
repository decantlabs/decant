//! Cross instance and cross process tests for the direct shader disk cache.
//! Shader inputs are synthetic. The real DXC test runs without a GPU device.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use d3d12_xs::compile_service::{Compiler, PipelineKind, Poll, Service, Settings};
use d3d12_xs::direct_cache::{DiskCache, Fnv128};
use d3d12_xs::rts0::{RegisterClass, RootLayout, RootParam, RootParamKind, RootRange, StaticSampler};
use d3d12_xs::shader_source::{InputElement, Linkage, Pipeline, Refusal, ShaderSource, StageHlsl, StageInput, StageKind};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("d3d12xs_disk_cache_{tag}_{}_{nanos}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A translator that translates during `accepts` and keys stages by their HLSL text, just as
/// `DirectSource` does in the layer. Counts every translation call.
struct TranslatingSource {
    version: String,
    translations: AtomicU64,
    kept: Mutex<Vec<(u128, Vec<StageHlsl>)>>,
}

impl TranslatingSource {
    fn new(version: &str) -> TranslatingSource {
        TranslatingSource { version: version.to_owned(), translations: AtomicU64::new(0), kept: Mutex::new(Vec::new()) }
    }

    fn key_of(pipeline: &Pipeline) -> u128 {
        let mut h = Fnv128::default();
        h.field(&pipeline.root.flags.to_le_bytes());
        for s in &pipeline.stages {
            h.field(s.kind.short().as_bytes()).field(&s.bytes);
        }
        h.finish()
    }

    fn build_stages(&self, pipeline: &Pipeline) -> Result<Vec<StageHlsl>, Refusal> {
        let k = Self::key_of(pipeline);
        if let Some((_, found)) = self.kept.lock().unwrap().iter().find(|(id, _)| *id == k) {
            return Ok(found.clone());
        }
        self.translations.fetch_add(1, Ordering::SeqCst);
        let mut out = Vec::with_capacity(pipeline.stages.len());
        for (idx, s) in pipeline.stages.iter().enumerate() {
            if s.bytes.starts_with(b"REFUSE") {
                return Err(Refusal::new("synthetic refusal", s.kind.short()));
            }
            // Linked pixel text incorporates the preceding stage kind, matching real stage linking.
            let linked_from = if idx > 0 { pipeline.stages[idx - 1].kind.short() } else { "entry" };
            let tag = String::from_utf8_lossy(&s.bytes);
            let profile = format!("{}_6_6", s.kind.short());
            let threads = if s.kind == StageKind::Compute { "[numthreads(1, 1, 1)] " } else { "" };
            let hlsl = format!("// stage {} from {linked_from} root_flags={:#x} rt={:?}\n{threads}void main() {{ /* {tag} */ }}\n", s.kind.short(), pipeline.root.flags, pipeline.linkage.targets);
            out.push(StageHlsl { hlsl, entry: "main".to_owned(), profile, args: vec!["-Gis".to_owned()] });
        }
        self.kept.lock().unwrap().push((k, out.clone()));
        Ok(out)
    }
}

impl ShaderSource for TranslatingSource {
    fn version(&self) -> String {
        self.version.clone()
    }

    fn accepts(&self, pipeline: &Pipeline) -> Result<(), Refusal> {
        self.build_stages(pipeline).map(|_| ())
    }

    fn stage_keys(&self, pipeline: &Pipeline) -> Option<Vec<u128>> {
        let stages = self.build_stages(pipeline).ok()?;
        Some(
            stages
                .iter()
                .map(|s| {
                    let mut h = Fnv128::default();
                    h.field(s.hlsl.as_bytes()).field(s.entry.as_bytes()).field(s.profile.as_bytes());
                    for a in &s.args {
                        h.field(a.as_bytes());
                    }
                    h.finish()
                })
                .collect(),
        )
    }

    fn translate(&self, pipeline: &Pipeline) -> Vec<Result<StageHlsl, Refusal>> {
        match self.build_stages(pipeline) {
            Ok(stages) => stages.into_iter().map(Ok).collect(),
            Err(r) => pipeline.stages.iter().map(|_| Err(r.clone())).collect(),
        }
    }
}

#[derive(Default)]
struct CountingCompiler {
    calls: AtomicU64,
}

impl Compiler for CountingCompiler {
    fn version(&self) -> String {
        "dxc_synthetic_1_9".to_owned()
    }

    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut out = b"DXBC_SYNTHETIC:".to_vec();
        out.extend_from_slice(profile.as_bytes());
        out.push(b':');
        out.extend_from_slice(entry.as_bytes());
        out.push(b':');
        out.extend_from_slice(args.join(",").as_bytes());
        out.push(b':');
        out.extend_from_slice(hlsl.as_bytes());
        Ok(out)
    }
}

struct CountingDxc {
    dxc: decant_dxc::Dxc,
    calls: AtomicU64,
}

impl Compiler for CountingDxc {
    fn version(&self) -> String {
        self.dxc.version()
    }

    fn compile(&self, hlsl: &str, entry: &str, profile: &str, args: &[String]) -> Result<Vec<u8>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.dxc.compile(hlsl, entry, profile, args)
    }
}

fn real_dxc_child(cache: DiskCache, warm: bool) {
    let compiler = Arc::new(CountingDxc { dxc: decant_dxc::Dxc::load().expect("load DXC in child"), calls: AtomicU64::new(0) });
    let source = Arc::new(TranslatingSource::new("real_dxc_test"));
    let expected = cache.dir().join("expected.bin");
    let service = Service::new(source.clone(), compiler.clone(), Settings { cache: Some(cache), ..Settings::quiet(1) });
    let pipeline = compute_pipeline(b"real_dxc_compute");
    if warm {
        let Poll::Ready(stages) = service.poll(80, || Ok(pipeline)) else { panic!("warm DXC child missed the cache") };
        assert_eq!(stages[0].as_ref(), std::fs::read(expected).unwrap());
        assert_eq!(source.translations.load(Ordering::SeqCst), 0);
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 0);
    } else {
        let stages = poll_until_ready(&service, 10, &pipeline);
        assert!(stages[0].starts_with(b"DXBC"));
        std::fs::write(expected, &stages[0]).unwrap();
        assert_eq!(compiler.calls.load(Ordering::SeqCst), 1);
    }
}

fn sample_root() -> RootLayout {
    RootLayout {
        version: 2,
        flags: 0xC01,
        params: vec![
            RootParam {
                kind: RootParamKind::Table(vec![
                    RootRange { class: RegisterClass::Srv, count: 4, base: 0, space: 0, flags: 0, offset: 0 },
                    RootRange { class: RegisterClass::Cbv, count: 2, base: 0, space: 1, flags: 0, offset: 4 },
                ]),
                visibility: 0,
            },
            RootParam { kind: RootParamKind::Constants { register: 2, space: 0, count: 4 }, visibility: 1 },
            RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 },
        ],
        static_samplers: vec![StaticSampler { register: 0, space: 0, visibility: 5 }],
    }
}

fn graphics_pipeline(vs: &[u8], ps: &[u8]) -> Pipeline {
    Pipeline {
        stages: vec![StageInput { kind: StageKind::Vertex, bytes: Arc::from(vs) }, StageInput { kind: StageKind::Pixel, bytes: Arc::from(ps) }],
        root: sample_root(),
        linkage: Linkage {
            stages: vec![StageKind::Vertex, StageKind::Pixel],
            inputs: vec![InputElement { semantic: "POSITION".to_owned(), index: 0, format: 6, slot: 0, offset: 0, per_instance: false, step_rate: 0 }],
            targets: vec![28],
            depth_format: 40,
            topology_type: 3,
            sample_count: 1,
            stream_output: false,
        },
    }
}

fn compute_pipeline(cs: &[u8]) -> Pipeline {
    Pipeline {
        stages: vec![StageInput { kind: StageKind::Compute, bytes: Arc::from(cs) }],
        root: sample_root(),
        linkage: Linkage { stages: vec![StageKind::Compute], ..Linkage::default() },
    }
}

fn sample_pipelines() -> Vec<(PipelineKind, Pipeline)> {
    vec![
        (PipelineKind::Graphics, graphics_pipeline(b"synthetic_vs_alpha", b"synthetic_ps_alpha")),
        (PipelineKind::Graphics, graphics_pipeline(b"synthetic_vs_alpha", b"synthetic_ps_beta")),
        (PipelineKind::Compute, compute_pipeline(b"synthetic_cs_gamma")),
    ]
}

fn poll_until_ready(service: &Service, id: usize, p: &Pipeline) -> Vec<Vec<u8>> {
    let first = service.poll(id, || Ok(p.clone()));
    if let Poll::Ready(dxil) = first {
        return dxil.into_iter().map(|d| d.to_vec()).collect();
    }
    assert_eq!(first, Poll::Pending);
    assert!(service.wait(id, Duration::from_secs(10)), "pipeline {id} timed out");
    match service.poll(id, || panic!("build closure called twice")) {
        Poll::Ready(dxil) => dxil.into_iter().map(|d| d.to_vec()).collect(),
        other => panic!("unexpected poll outcome: {other:?}"),
    }
}

static LOG_LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn record_log(line: &str) {
    LOG_LINES.lock().unwrap_or_else(|e| e.into_inner()).push(line.to_owned());
}

fn take_logs() -> Vec<String> {
    std::mem::take(&mut *LOG_LINES.lock().unwrap_or_else(|e| e.into_inner()))
}

#[test]
fn two_services_in_sequence_skip_translation_and_compilation_and_log_hits() {
    let _ = take_logs();
    let tmp = TempDir::new("seq");
    let items = sample_pipelines();

    let source1 = Arc::new(TranslatingSource::new("translator_1"));
    let compiler1 = Arc::new(CountingCompiler::default());
    let service1 = Service::new(source1.clone(), compiler1.clone(), Settings { threads: 2, cache: Some(DiskCache::new(&tmp.0)), dump: None, log: record_log });
    let mut expected = Vec::new();
    for (i, (kind, p)) in items.iter().enumerate() {
        expected.push(poll_until_ready(&service1, i + 1, p));
        service1.pipeline_done(*kind, Some((true, Duration::from_micros(10))));
    }
    assert_eq!(compiler1.calls.load(Ordering::SeqCst), 4, "shared vertex stage compiles once");
    assert_eq!(service1.counters().compiled.load(Ordering::Relaxed), 4);
    drop(service1);
    let _ = take_logs();

    // Second service on the same folder: reverse order, other object ids, zero translations, zero compiles.
    let source2 = Arc::new(TranslatingSource::new("translator_1"));
    let compiler2 = Arc::new(CountingCompiler::default());
    let service2 = Service::new(source2.clone(), compiler2.clone(), Settings { threads: 2, cache: Some(DiskCache::new(&tmp.0)), dump: None, log: record_log });
    for (i, ((kind, p), want)) in items.iter().zip(&expected).enumerate().rev() {
        let got = match service2.poll(900 + i * 17, || Ok(p.clone())) {
            Poll::Ready(dxil) => dxil.into_iter().map(|d| d.to_vec()).collect::<Vec<_>>(),
            other => panic!("expected warm hit on first poll, got {other:?}"),
        };
        assert_eq!(&got, want);
        service2.pipeline_done(*kind, Some((true, Duration::from_micros(5))));
    }
    assert_eq!(source2.translations.load(Ordering::SeqCst), 0, "second service must skip translation via pipeline key");
    assert_eq!(compiler2.calls.load(Ordering::SeqCst), 0, "second service must compile zero stages");
    assert_eq!(service2.counters().compiled.load(Ordering::Relaxed), 0);
    assert_eq!(service2.counters().translated.load(Ordering::Relaxed), 0);
    assert_eq!(service2.counters().cache_hits.load(Ordering::Relaxed), 5);
    assert_eq!(service2.counters().ready.load(Ordering::Relaxed), 3);
    drop(service2);

    let logs = take_logs();
    assert!(logs.iter().any(|l| l.contains("cache hits")), "per pipeline log must report cache hits, got {logs:?}");
    assert!(logs.iter().any(|l| l.contains("direct shaders summary:") && l.contains("cache_hits=5") && l.contains("compiled=0")), "summary log must report cache_hits=5 and compiled=0, got {logs:?}");
}

/// Entry point invoked by child processes of this integration test binary.
#[test]
fn child_worker_entry() {
    let Ok(role) = std::env::var("DECANT_TEST_CACHE_ROLE") else {
        return;
    };
    let _ = take_logs();
    if role.starts_with("sampler_") {
        sampler_child(&role);
        return;
    }
    let cache = if role == "concurrent_writer" {
        // Stage publishing can still be exercised concurrently with explicit standalone caches.
        DiskCache::new(std::env::var_os("DECANT_D3D12X_DIRECT_CACHE").unwrap())
    } else {
        DiskCache::from_env().expect("DECANT_D3D12X_DIRECT_CACHE must configure DiskCache::from_env")
    };
    match role.as_str() {
        "populate_dxc" => real_dxc_child(cache, false),
        "verify_warm_dxc" => real_dxc_child(cache, true),
        "verify_relative" => {
            let expected = PathBuf::from(std::env::var_os("DECANT_TEST_EXPECTED_CACHE").unwrap());
            assert!(cache.dir().is_absolute());
            assert_eq!(cache.dir(), expected);
            assert_eq!(DiskCache::new("cache").dir(), expected);
            assert!(!Path::new("cache").exists());
        }
        "populate" => {
            let source = Arc::new(TranslatingSource::new("translator_1"));
            let compiler = Arc::new(CountingCompiler::default());
            let service = Service::new(source, compiler.clone(), Settings { threads: 2, cache: Some(cache), dump: None, log: record_log });
            for (i, (kind, p)) in sample_pipelines().iter().enumerate() {
                let _ = poll_until_ready(&service, 10 + i, p);
                service.pipeline_done(*kind, Some((true, Duration::from_micros(10))));
            }
            assert_eq!(compiler.calls.load(Ordering::SeqCst), 4);
            assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 4);
            drop(service);
            let logs = take_logs();
            assert!(!logs.iter().any(|l| l.contains("failed")), "unexpected cache error: {logs:?}");
        }
        "verify_warm" => {
            let reference_source = TranslatingSource::new("translator_1");
            let reference_compiler = CountingCompiler::default();
            let expected: Vec<Vec<Vec<u8>>> = sample_pipelines()
                .iter()
                .map(|(_, p)| {
                    reference_source
                        .translate(p)
                        .into_iter()
                        .map(|s| {
                            let s = s.expect("valid stage");
                            reference_compiler.compile(&s.hlsl, &s.entry, &s.profile, &s.args).expect("compiled")
                        })
                        .collect()
                })
                .collect();

            let source = Arc::new(TranslatingSource::new("translator_1"));
            let compiler = Arc::new(CountingCompiler::default());
            let service = Service::new(source.clone(), compiler.clone(), Settings { threads: 2, cache: Some(cache), dump: None, log: record_log });
            // Reverse order and distinct ids so process memory addresses and order cannot match.
            for ((i, (kind, p)), want) in sample_pipelines().iter().enumerate().zip(&expected).rev() {
                match service.poll(5000 + i * 31, || Ok(p.clone())) {
                    Poll::Ready(dxil) => {
                        let got: Vec<Vec<u8>> = dxil.into_iter().map(|b| b.to_vec()).collect();
                        assert_eq!(&got, want);
                    }
                    other => panic!("child process missed disk cache: {other:?}"),
                }
                service.pipeline_done(*kind, Some((true, Duration::from_micros(4))));
            }
            assert_eq!(source.translations.load(Ordering::SeqCst), 0, "warm child translated stages");
            assert_eq!(compiler.calls.load(Ordering::SeqCst), 0, "warm child compiled stages");
            assert_eq!(service.counters().compiled.load(Ordering::Relaxed), 0);
            assert_eq!(service.counters().translated.load(Ordering::Relaxed), 0);
            assert_eq!(service.counters().cache_hits.load(Ordering::Relaxed), 5);
            assert_eq!(service.counters().ready.load(Ordering::Relaxed), 3);
            drop(service);
            let logs = take_logs();
            assert!(logs.iter().any(|l| l.contains("cache hits")), "warm child log lacks cache hits: {logs:?}");
            assert!(logs.iter().any(|l| l.contains("direct shaders summary:") && l.contains("cache_hits=5") && l.contains("compiled=0")), "warm child summary lacks counters: {logs:?}");
        }
        "concurrent_writer" => {
            let source = Arc::new(TranslatingSource::new("translator_concurrent"));
            let compiler = Arc::new(CountingCompiler::default());
            let service = Service::new(source, compiler, Settings { threads: 4, cache: Some(cache), dump: None, log: record_log });
            let batch: Vec<Pipeline> = (0..12u8)
                .map(|n| {
                    let vs = format!("concurrent_vs_{}", n % 4).into_bytes();
                    let ps = format!("concurrent_ps_{n}").into_bytes();
                    graphics_pipeline(&vs, &ps)
                })
                .collect();
            for (i, p) in batch.iter().enumerate() {
                let got = poll_until_ready(&service, 100 + i, p);
                assert_eq!(got.len(), 2);
            }
            drop(service);
            let logs = take_logs();
            assert!(!logs.iter().any(|l| l.contains("failed")), "concurrent write failed: {logs:?}");
        }
        "rebuild" => {
            let source = Arc::new(TranslatingSource::new("translator_1"));
            let compiler = Arc::new(CountingCompiler::default());
            let service = Service::new(source, compiler.clone(), Settings { threads: 2, cache: Some(cache), dump: None, log: record_log });
            for (i, (_, p)) in sample_pipelines().iter().enumerate() {
                let got = poll_until_ready(&service, 70 + i, p);
                assert_eq!(got.len(), p.stages.len());
                for stage_bytes in &got {
                    assert!(stage_bytes.starts_with(b"DXBC_SYNTHETIC:"));
                }
            }
            assert!(compiler.calls.load(Ordering::SeqCst) >= 1, "damaged stages must be recompiled");
            drop(service);
            let logs = take_logs();
            assert!(!logs.iter().any(|l| l.contains("failed")), "rebuild write failed: {logs:?}");
        }
        other => panic!("unknown role {other}"),
    }
}

fn child_command(role: &str, cache_dir: &Path) -> Command {
    let exe = std::env::current_exe().expect("current test binary");
    let mut command = Command::new(exe);
    command
        .arg("--exact")
        .arg("child_worker_entry")
        .arg("--nocapture")
        .env("DECANT_TEST_CACHE_ROLE", role)
        .env("DECANT_D3D12X_DIRECT_CACHE", cache_dir);
    command
}

fn spawn_role(role: &str, cache_dir: &Path) -> std::process::Child {
    child_command(role, cache_dir).spawn().expect("spawn child test process")
}

fn run_role(role: &str, cache_dir: &Path) {
    let out = spawn_role(role, cache_dir).wait_with_output().expect("wait for child");
    assert!(
        out.status.success(),
        "role {role} failed ({:?}):\nstdout:\n{}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_second_process_reads_every_stage_without_calling_the_synthetic_compiler() {
    let tmp = TempDir::new("cross_proc");
    run_role("populate", &tmp.0);
    run_role("verify_warm", &tmp.0);
}

#[test]
fn a_warm_child_makes_zero_real_dxc_calls() {
    if let Err(err) = decant_dxc::Dxc::load() {
        eprintln!("skipping real DXC test: {err}");
        return;
    }
    let tmp = TempDir::new("real_dxc");
    run_role("populate_dxc", &tmp.0);
    run_role("verify_warm_dxc", &tmp.0);
}

#[test]
fn relative_overrides_resolve_below_the_default_cache_from_any_working_directory() {
    let tmp = TempDir::new("relative");
    let expected = tmp.0.join("Decant").join("direct-dxil").join("cache");
    for cwd in ["first", "second"] {
        let cwd = tmp.0.join(cwd);
        std::fs::create_dir_all(&cwd).unwrap();
        let status = child_command("verify_relative", Path::new("cache"))
            .env("DECANT_TEST_EXPECTED_CACHE", &expected)
            .env("LOCALAPPDATA", &tmp.0)
            .env("XDG_CACHE_HOME", &tmp.0)
            .current_dir(cwd)
            .status()
            .unwrap();
        assert!(status.success());
    }
}

fn wait_for_file(path: &Path) {
    let start = std::time::Instant::now();
    while !path.is_file() {
        assert!(start.elapsed() < Duration::from_secs(10), "sampler child timed out");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn sampler_child(role: &str) {
    use d3d12_xs::sampler_registry::index_of;

    let dir = PathBuf::from(std::env::var_os("DECANT_D3D12X_DIRECT_CACHE").unwrap());
    let original = [1, 2, 3, 4];
    let competing = [2, 3, 4, 5];
    match role {
        "sampler_owner" => {
            let cache = DiskCache::from_env().unwrap();
            assert_eq!(index_of(original).unwrap().index, 0);
            cache.write_pipeline(1, &[10]).unwrap();
            std::fs::write(dir.join("ready"), []).unwrap();
            wait_for_file(&dir.join("done"));
        }
        "sampler_contender" => {
            assert!(DiskCache::from_env().is_none());
            assert_eq!(index_of(competing).unwrap().index, 0);
            let explicit = DiskCache::new(&dir);
            assert_eq!(explicit.read_pipeline(1), None);
            explicit.write_pipeline(1, &[20]).unwrap();
            explicit.write_pipeline(2, &[20]).unwrap();
        }
        "sampler_reader" => {
            let cache = DiskCache::from_env().unwrap();
            assert_eq!(cache.read_pipeline(1), Some(vec![10]));
            assert_eq!(cache.read_pipeline(2), None);
            assert_eq!(index_of(original).unwrap().index, 0);
            assert_eq!(index_of(competing).unwrap().index, 1);
        }
        _ => panic!("unknown sampler role"),
    }
}

#[test]
fn concurrent_sampler_assignments_cannot_publish_mismatched_pipeline_records() {
    let tmp = TempDir::new("samplers");
    let mut owner = spawn_role("sampler_owner", &tmp.0);
    wait_for_file(&tmp.0.join("ready"));
    let contender = child_command("sampler_contender", &tmp.0).status();
    std::fs::write(tmp.0.join("done"), []).unwrap();
    assert!(owner.wait().unwrap().success());
    assert!(contender.unwrap().success());
    run_role("sampler_reader", &tmp.0);
}

#[test]
fn two_processes_writing_at_once_leave_valid_entries_and_no_temp_files() {
    let tmp = TempDir::new("two_procs");
    let c1 = spawn_role("concurrent_writer", &tmp.0);
    let c2 = spawn_role("concurrent_writer", &tmp.0);
    let o1 = c1.wait_with_output().expect("child 1");
    let o2 = c2.wait_with_output().expect("child 2");
    assert!(o1.status.success(), "child 1 failed:\n{}{}", String::from_utf8_lossy(&o1.stdout), String::from_utf8_lossy(&o1.stderr));
    assert!(o2.status.success(), "child 2 failed:\n{}{}", String::from_utf8_lossy(&o2.stdout), String::from_utf8_lossy(&o2.stderr));

    let leftovers: Vec<_> = std::fs::read_dir(&tmp.0)
        .expect("cache dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporary files left behind: {leftovers:?}");

    // A fresh reader finds all 12 pipelines warm without translating or compiling.
    let source = Arc::new(TranslatingSource::new("translator_concurrent"));
    let compiler = Arc::new(CountingCompiler::default());
    let service = Service::new(source.clone(), compiler.clone(), Settings { threads: 2, cache: Some(DiskCache::new(&tmp.0)), dump: None, log: |_| {} });
    for n in 0..12u8 {
        let vs = format!("concurrent_vs_{}", n % 4).into_bytes();
        let ps = format!("concurrent_ps_{n}").into_bytes();
        let p = graphics_pipeline(&vs, &ps);
        assert!(matches!(service.poll(usize::from(n) + 1, || Ok(p)), Poll::Ready(_)));
    }
    assert_eq!(source.translations.load(Ordering::SeqCst), 0);
    assert_eq!(compiler.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn corrupt_half_written_and_old_version_files_are_ignored_and_rebuilt() {
    let tmp = TempDir::new("corrupt");
    run_role("populate", &tmp.0);

    let mut dxil_files: Vec<PathBuf> = std::fs::read_dir(&tmp.0)
        .expect("cache dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("dxil"))
        .collect();
    dxil_files.sort();
    assert_eq!(dxil_files.len(), 4, "expected 4 distinct stage files");

    // 1. Flip a byte inside the DXIL payload while keeping header and file length intact.
    let mut flipped = std::fs::read(&dxil_files[0]).expect("read stage 0");
    let last = flipped.len() - 1;
    flipped[last] ^= 0xFF;
    std::fs::write(&dxil_files[0], &flipped).expect("write corrupted payload");

    // 2. Cut a stage file in half like an interrupted write.
    let mut half = std::fs::read(&dxil_files[1]).expect("read stage 1");
    half.truncate(half.len() / 2);
    std::fs::write(&dxil_files[1], &half).expect("write half file");

    // 3. Stamp an old format version 1 into the header of stage 2.
    let mut old_ver = std::fs::read(&dxil_files[2]).expect("read stage 2");
    old_ver[4..8].copy_from_slice(&1u32.to_le_bytes());
    std::fs::write(&dxil_files[2], &old_ver).expect("write old format");

    // 4. Truncate a pipeline record file and leave a stray temporary file in the folder.
    let pipe_files: Vec<PathBuf> = std::fs::read_dir(&tmp.0)
        .expect("cache dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("pipe"))
        .collect();
    assert_eq!(pipe_files.len(), 3, "expected 3 pipeline files");
    std::fs::write(&pipe_files[0], b"SXDP\x02").expect("truncate pipe");
    std::fs::write(tmp.0.join(".stray.99999.0.tmp"), b"incomplete").expect("write stray tmp");

    // Rebuild in one child process, then verify warm hits in a fresh child process.
    run_role("rebuild", &tmp.0);
    run_role("verify_warm", &tmp.0);
}
