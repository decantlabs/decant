//! Two process integration tests for `decant_shader_warm` using synthetic GFX10 containers.
//! No game data is read or written.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use d3d12_xs::pipeline_manifest::Writer;
use d3d12_xs::rts0::{RegisterClass, RootLayout, RootParam, RootParamKind, RootRange};
use d3d12_xs::shader_source::{Linkage, StageKind};
use decant_shader::api::{DeclaredResource, PixelInput, RegisterClass as SdClass, SigElement, Stage, StageProgram, TableBinding, UserDataKind, UserDataSlot};
use decant_shader::asm::*;
use decant_shader::tables::Op;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("decant_shader_warm_it_{tag}_{}_{nanos}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn signature_part(elements: &[SigElement]) -> Vec<u8> {
    let names_at = 8 + 32 * elements.len();
    let mut names = Vec::new();
    let mut out = Vec::new();
    put32(&mut out, elements.len() as u32);
    put32(&mut out, 8);
    for e in elements {
        put32(&mut out, 0);
        put32(&mut out, (names_at + names.len()) as u32);
        for v in [e.index, e.system_value, e.component_type, e.register] {
            put32(&mut out, v);
        }
        out.extend_from_slice(&[e.mask, e.rw_mask, 0, 0, 0, 0, 0, 0]);
        names.extend_from_slice(e.semantic.as_bytes());
        names.push(0);
    }
    out.extend_from_slice(&names);
    out
}

fn psv0_part(resources: &[DeclaredResource], threads: [u32; 3]) -> Vec<u8> {
    let mut out = Vec::new();
    put32(&mut out, 52);
    let mut info = [0u8; 52];
    info[36..40].copy_from_slice(&threads[0].to_le_bytes());
    info[40..44].copy_from_slice(&threads[1].to_le_bytes());
    info[44..48].copy_from_slice(&threads[2].to_le_bytes());
    out.extend_from_slice(&info);
    put32(&mut out, resources.len() as u32);
    if !resources.is_empty() {
        put32(&mut out, 24);
    }
    for r in resources {
        for v in [r.psv_type, r.space, r.lower, r.upper, r.psv_kind, 0] {
            put32(&mut out, v);
        }
    }
    out
}

fn fssm_block(program: &StageProgram) -> Vec<u8> {
    const HEADER: usize = 0x240;
    const CODE_END: u32 = 0xBF9F_0000;
    let mut out = b"FSSM".to_vec();
    out.resize(HEADER, 0);
    let code_bytes = (program.code.len() + 1) * 4;
    out[0xC0..0xC8].copy_from_slice(&(HEADER as u64).to_le_bytes());
    out[0xC8..0xD0].copy_from_slice(&(code_bytes as u64).to_le_bytes());
    for p in &program.pixel_inputs {
        let at = 0x138 + 8 * usize::from(p.attr);
        let second = p.high_register.map_or(0, |r| r as u8);
        out[at..at + 8].copy_from_slice(&[p.kind, p.register as u8, p.channels, p.attr, 0, second, 0, p.flags]);
    }
    out.extend_from_slice(&[0; 16]);
    for w in program.code.iter().chain(&[CODE_END]) {
        put32(&mut out, *w);
    }
    let layout_at = out.len() as u64;
    out[0xE8..0xF0].copy_from_slice(&layout_at.to_le_bytes());
    out[0xF8..0xFC].copy_from_slice(&(program.user_data.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for u in &program.user_data {
        let (kind, count, register) = match u.kind {
            UserDataKind::Table => (3, 0, 0),
            UserDataKind::Constants { register, count } => (0, count, register),
            UserDataKind::RootSrv { register } => (1, 0, register),
            UserDataKind::ResourceHeap => (5, 0, 0),
            UserDataKind::SamplerHeap => (6, 0, 0),
        };
        for v in [kind, 0, count, register, u32::from(u.sgpr)] {
            put32(&mut out, v);
        }
    }
    let table_at = out.len() as u64;
    out[0xF0..0xF8].copy_from_slice(&table_at.to_le_bytes());
    out[0xFC..0x100].copy_from_slice(&(program.binding_table.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for b in &program.binding_table {
        let kind = match b.class {
            SdClass::Srv => 0x300,
            SdClass::Cbv => 0x302,
            SdClass::Uav => 0x303,
            SdClass::Sampler => 0x401,
        };
        for v in [kind, b.word_offset, b.register, 0, u32::from(b.sgpr) << 8] {
            put32(&mut out, v);
        }
    }
    out
}

fn stored_zlib(data: &[u8]) -> Vec<u8> {
    let len = u16::try_from(data.len()).expect("fits stored zlib block");
    let mut out = vec![0x78, 0x01, 0x01];
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn dxil_part(stage: Stage, fssm: &[u8]) -> Vec<u8> {
    let stage_code: u32 = match stage {
        Stage::Pixel => 0,
        Stage::Vertex => 1,
        Stage::Geometry => 2,
        Stage::Hull => 3,
        Stage::Domain => 4,
        Stage::Compute => 5,
        Stage::Mesh => 13,
        Stage::Amplification => 14,
    };
    let stream = stored_zlib(fssm);
    let mut out = Vec::new();
    put32(&mut out, stage_code << 16 | 6 << 4 | 6);
    put32(&mut out, 0);
    out.extend_from_slice(b"XDAT");
    put32(&mut out, 1);
    put32(&mut out, 0);
    let xbch_at = out.len() + 12;
    for v in [0x0D, xbch_at as u32, (24 + stream.len()) as u32] {
        put32(&mut out, v);
    }
    out.extend_from_slice(b"XBCH\0\0\0\0");
    out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(&stream);
    out
}

fn container(program: &StageProgram) -> Vec<u8> {
    let parts: [(&[u8; 4], Vec<u8>); 4] = [
        (b"ISG1", signature_part(&program.inputs)),
        (b"OSG1", signature_part(&program.outputs)),
        (b"PSV0", psv0_part(&program.resources, program.threads)),
        (b"DXIL", dxil_part(program.stage, &fssm_block(program))),
    ];
    let mut body = Vec::new();
    let mut offsets = Vec::new();
    let table_end = 32 + 4 * parts.len();
    for (tag, data) in &parts {
        offsets.push((table_end + body.len()) as u32);
        body.extend_from_slice(*tag);
        put32(&mut body, data.len() as u32);
        body.extend_from_slice(data);
        body.resize(body.len().next_multiple_of(4), 0);
    }
    let mut out = b"DXBC".to_vec();
    let id = decant_shader::container::fnv1a64(&body);
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(&(!id).to_le_bytes());
    put32(&mut out, 1);
    put32(&mut out, (table_end + body.len()) as u32);
    put32(&mut out, parts.len() as u32);
    for o in offsets {
        put32(&mut out, o);
    }
    out.extend_from_slice(&body);
    out
}

fn sig_element(semantic: &str, index: u32, register: u32, mask: u8, rw_mask: u8) -> SigElement {
    let system_value = match semantic {
        "SV_Position" => 1,
        "SV_Target" => 64,
        _ => 0,
    };
    SigElement { semantic: semantic.to_owned(), index, system_value, component_type: 3, register, mask, rw_mask }
}

/// Synthetic vertex shader parameterized by a constant float `scale` so distinct variants compile
/// to distinct HLSL and DXIL.
fn synthetic_vertex(scale: f32) -> StageProgram {
    let mut a = Asm::new();
    a.push(sopp(Op::s_inst_prefetch, 3));
    a.push(vop1(Op::v_mov_b32, v(1), lit(scale.to_bits())));
    a.push(vop1(Op::v_mov_b32, v(2), int(0)));
    a.push(vop1(Op::v_mov_b32, v(3), int(0)));
    a.push(vop1(Op::v_mov_b32, v(4), float(1.0)));
    a.push(export(12, 0xf, [1, 2, 3, 4], false, true, false));
    a.push(export(32, 0xf, [1, 2, 3, 4], false, false, false));
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs: vec![sig_element("POSITION", 0, 0, 3, 3)],
        outputs: vec![sig_element("COLOR", 0, 0, 15, 0), sig_element("SV_Position", 0, 1, 15, 0)],
        name: format!("vs_{}", scale.to_bits()),
        ..StageProgram::new(Stage::Vertex, a.finish())
    }
}

/// Synthetic pixel shader parameterized by `tint`.
fn synthetic_pixel(tint: f32) -> StageProgram {
    let mut a = Asm::new();
    a.push(sopp(Op::s_inst_prefetch, 3));
    a.push(sop1(Op::s_mov_b32, M0, s(12)));
    for chan in 0..4u8 {
        a.push(interp(Op::v_interp_p1_f32, v(2 + chan), v(0), 0, chan));
        a.push(interp(Op::v_interp_p2_f32, v(2 + chan), v(1), 0, chan));
    }
    a.push(vop3(Op::v_fma_f32, v(2), v(2), lit(tint.to_bits()), int(0)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(2), v(3)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(4), v(5)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs: vec![sig_element("COLOR", 0, 0, 15, 15)],
        outputs: vec![sig_element("SV_Target", 0, 0, 15, 0)],
        pixel_inputs: vec![PixelInput { attr: 0, kind: 6, register: 0, high_register: None, channels: 0xf, flags: 0x02 }],
        name: format!("ps_{}", tint.to_bits()),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

/// Synthetic compute shader writing a constant word `value` to a UAV buffer in table `s4`.
fn synthetic_compute(value: u32) -> StageProgram {
    let mut a = Asm::new();
    a.push(sopp(Op::s_inst_prefetch, 3));
    a.push(sop1(Op::s_mov_b32, s(5), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(0), s(4), 0));
    a.push(vop1(Op::v_mov_b32, v(1), lit(value)));
    a.push(buffer_load(Op::buffer_store_dword, v(1), v(0), s(0), 0, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = StageProgram::new(Stage::Compute, a.finish());
    p.threads = [64, 1, 1];
    p.resources = vec![DeclaredResource { class: SdClass::Uav, lower: 0, upper: 0, space: 1, psv_type: 8, psv_kind: 11 }];
    p.binding_table = vec![TableBinding { class: SdClass::Uav, register: 0, sgpr: 4, word_offset: 0 }];
    p.user_data = vec![UserDataSlot { sgpr: 4, kind: UserDataKind::Table }];
    p.name = format!("cs_{value}");
    p
}

fn layer_root_layout() -> RootLayout {
    RootLayout {
        version: 2,
        flags: 0xc01,
        params: vec![
            RootParam {
                kind: RootParamKind::Table(vec![RootRange { class: RegisterClass::Uav, count: 1, base: 0, space: 1, flags: 0, offset: 0 }]),
                visibility: 0,
            },
            RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 1, space: 15 }, visibility: 0 },
            RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 15 }, visibility: 0 },
            RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Srv, register: 1, space: 15 }, visibility: 0 },
        ],
        static_samplers: Vec::new(),
    }
}

fn dump_container(dir: &Path, program: &StageProgram) -> Vec<u8> {
    let bytes = container(program);
    let hash = decant_shader::container::fnv1a64(&bytes);
    std::fs::write(dir.join(format!("{hash:016x}.bin")), &bytes).expect("write dump");
    bytes
}

fn parse_summary(stdout: &str) -> HashMap<String, String> {
    let line = stdout.lines().find(|l| l.starts_with("summary: ")).unwrap_or_else(|| panic!("missing summary line in:\n{stdout}"));
    line.trim_start_matches("summary: ")
        .split_whitespace()
        .filter_map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect()
}

#[test]
fn second_process_hits_the_cache_and_compiles_zero_stages() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm integration test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("two_process");
    let dumps = tmp.0.join("dumps");
    let cache = tmp.0.join("cache");
    let manifest_path = tmp.0.join("pipelines.manifest");
    std::fs::create_dir_all(&dumps).expect("dumps dir");

    let vs_a = dump_container(&dumps, &synthetic_vertex(1.25));
    let ps_a = dump_container(&dumps, &synthetic_pixel(0.5));
    let ps_b = dump_container(&dumps, &synthetic_pixel(0.75));
    let cs_a = dump_container(&dumps, &synthetic_compute(0x1234_5678));

    let root = layer_root_layout();
    let gfx = Linkage { stages: vec![StageKind::Vertex, StageKind::Pixel], targets: vec![28], topology_type: 3, sample_count: 1, ..Linkage::default() };
    let cmp = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
    {
        let mut w = Writer::open(&manifest_path).expect("open manifest");
        assert!(w.record(Some(&root), &gfx, &[&vs_a, &ps_a]).expect("record 1"));
        assert!(w.record(Some(&root), &gfx, &[&vs_a, &ps_b]).expect("record 2"));
        assert!(w.record(Some(&root), &cmp, &[&cs_a]).expect("record 3"));
    }

    let exe = env!("CARGO_BIN_EXE_decant_shader_warm");

    let first = Command::new(exe)
        .args(["--manifest", manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "2"])
        .output()
        .expect("run first process");
    let first_stdout = String::from_utf8_lossy(&first.stdout);
    let first_stderr = String::from_utf8_lossy(&first.stderr);
    assert!(first.status.success(), "first run failed:\nstdout:\n{first_stdout}\nstderr:\n{first_stderr}");
    let s1 = parse_summary(&first_stdout);
    assert_eq!(s1.get("pipelines").map(String::as_str), Some("3"));
    assert_eq!(s1.get("ready").map(String::as_str), Some("3"));
    assert_eq!(s1.get("stages_compiled").map(String::as_str), Some("4"));
    assert_eq!(s1.get("refused").map(String::as_str), Some("0"));
    assert_eq!(s1.get("failures").map(String::as_str), Some("0"));

    let dxil_files = std::fs::read_dir(&cache)
        .expect("cache dir")
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("dxil"))
        .count();
    assert_eq!(dxil_files, 4, "expected 4 distinct stage DXIL files in cache");

    let second = Command::new(exe)
        .args(["--manifest", manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "2"])
        .output()
        .expect("run second process");
    let second_stdout = String::from_utf8_lossy(&second.stdout);
    let second_stderr = String::from_utf8_lossy(&second.stderr);
    assert!(second.status.success(), "second run failed:\nstdout:\n{second_stdout}\nstderr:\n{second_stderr}");
    let s2 = parse_summary(&second_stdout);
    assert_eq!(s2.get("pipelines").map(String::as_str), Some("3"));
    assert_eq!(s2.get("ready").map(String::as_str), Some("3"));
    assert_eq!(s2.get("hits").map(String::as_str), Some("3"));
    assert_eq!(s2.get("stages_compiled").map(String::as_str), Some("0"));
    assert_eq!(s2.get("stage_cache_hits").map(String::as_str), Some("5"));
    assert_eq!(s2.get("refused").map(String::as_str), Some("0"));
    assert_eq!(s2.get("failures").map(String::as_str), Some("0"));
}

#[test]
fn bind_order_and_top_warm_highest_ranked_pipelines_first() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm bind order test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("bind_order");
    let dumps = tmp.0.join("dumps");
    let cache = tmp.0.join("cache");
    let manifest_path = tmp.0.join("all.manifest");
    let ranked_manifest_path = tmp.0.join("ranked.manifest");
    let bind_order_path = tmp.0.join("bind-order.txt");
    std::fs::create_dir_all(&dumps).expect("dumps dir");

    let cs0 = dump_container(&dumps, &synthetic_compute(10));
    let cs1 = dump_container(&dumps, &synthetic_compute(20));
    let cs2 = dump_container(&dumps, &synthetic_compute(30));
    let cs3 = dump_container(&dumps, &synthetic_compute(40));

    let root = layer_root_layout();
    let cmp = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
    {
        let mut w = Writer::open(&manifest_path).expect("open all manifest");
        for cs in [&cs0, &cs1, &cs2, &cs3] {
            assert!(w.record(Some(&root), &cmp, &[cs]).expect("record"));
        }
    }
    let h1 = decant_shader::container::fnv1a64(&cs1);
    let h3 = decant_shader::container::fnv1a64(&cs3);
    std::fs::write(&bind_order_path, format!("CS {h3:016x}\nCS {h1:016x}\n")).expect("write bind order");
    {
        let mut w = Writer::open(&ranked_manifest_path).expect("open ranked manifest");
        assert!(w.record(Some(&root), &cmp, &[&cs3]).expect("record cs3"));
        assert!(w.record(Some(&root), &cmp, &[&cs1]).expect("record cs1"));
    }

    let exe = env!("CARGO_BIN_EXE_decant_shader_warm");
    let warm = Command::new(exe)
        .args([
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--bind-order",
            bind_order_path.to_str().unwrap(),
            "--top",
            "2",
            "--dump-dir",
            dumps.to_str().unwrap(),
            "--cache",
            cache.to_str().unwrap(),
            "--threads",
            "2",
        ])
        .output()
        .expect("warm top 2");
    assert!(warm.status.success(), "warm top 2 failed: {}", String::from_utf8_lossy(&warm.stderr));
    let s_warm = parse_summary(&String::from_utf8_lossy(&warm.stdout));
    assert_eq!(s_warm.get("pipelines").map(String::as_str), Some("2"));
    assert_eq!(s_warm.get("stages_compiled").map(String::as_str), Some("2"));

    let check = Command::new(exe)
        .args(["--manifest", ranked_manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "2"])
        .output()
        .expect("check ranked");
    assert!(check.status.success());
    let s_check = parse_summary(&String::from_utf8_lossy(&check.stdout));
    assert_eq!(s_check.get("hits").map(String::as_str), Some("2"));
    assert_eq!(s_check.get("stages_compiled").map(String::as_str), Some("0"));
}

#[test]
fn relative_cache_uses_the_resolved_bind_order_and_counts_its_files() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm relative cache test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("relative_cache");
    let dumps = tmp.0.join("dumps");
    let cache_root = tmp.0.join("cache-root");
    let cache = cache_root.join("Decant").join("direct-dxil").join("warm");
    let manifest = tmp.0.join("all.manifest");
    std::fs::create_dir_all(&dumps).expect("dumps dir");
    std::fs::create_dir_all(&cache).expect("resolved cache dir");
    let first = container(&synthetic_compute(10));
    let ranked = dump_container(&dumps, &synthetic_compute(20));
    let root = layer_root_layout();
    let linkage = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
    {
        let mut w = Writer::open(&manifest).expect("open manifest");
        assert!(w.record(Some(&root), &linkage, &[&first]).expect("record first"));
        assert!(w.record(Some(&root), &linkage, &[&ranked]).expect("record ranked"));
    }
    let ranked_hash = decant_shader::container::fnv1a64(&ranked);
    std::fs::write(cache.join("bind-order.txt"), format!("CS {ranked_hash:016x}\n")).expect("write implicit bind order");

    let output = Command::new(env!("CARGO_BIN_EXE_decant_shader_warm"))
        .current_dir(&tmp.0)
        .env("LOCALAPPDATA", &cache_root)
        .env("XDG_CACHE_HOME", &cache_root)
        .args(["--manifest", manifest.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", "warm", "--top", "1"])
        .output()
        .expect("warm with relative cache");
    assert!(output.status.success(), "warm failed: {}", String::from_utf8_lossy(&output.stderr));
    let summary = parse_summary(&String::from_utf8_lossy(&output.stdout));
    let actual_bytes: u64 = std::fs::read_dir(cache).unwrap().map(|e| e.unwrap().metadata().unwrap().len()).sum();
    assert_eq!(summary.get("stages_compiled").map(String::as_str), Some("1"));
    assert_eq!(summary.get("cache_bytes").and_then(|n| n.parse().ok()), Some(actual_bytes));
}

#[test]
fn missing_stage_bytes_exit_with_failure_code() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm failure code test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("missing_stage");
    let dumps = tmp.0.join("dumps");
    let cache = tmp.0.join("cache");
    let manifest_path = tmp.0.join("missing.manifest");
    std::fs::create_dir_all(&dumps).expect("dumps dir");

    let root = layer_root_layout();
    let cmp = Linkage { stages: vec![StageKind::Compute], ..Linkage::default() };
    {
        let mut w = Writer::open(&manifest_path).expect("open manifest");
        assert!(w.record(Some(&root), &cmp, &[b"not in dumps"]).expect("record"));
    }

    let exe = env!("CARGO_BIN_EXE_decant_shader_warm");
    let out = Command::new(exe)
        .args(["--manifest", manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "1"])
        .output()
        .expect("run warm");
    assert!(!out.status.success(), "expected failure exit status for missing stage bytes");
    let s = parse_summary(&String::from_utf8_lossy(&out.stdout));
    assert_eq!(s.get("failures").map(String::as_str), Some("1"));
}

#[test]
fn corpus_mode_without_manifest_warms_top_n_and_hits_cache_on_second_run() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm corpus mode test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("corpus_mode");
    let dumps = tmp.0.join("dumps");
    let cache = tmp.0.join("cache");
    std::fs::create_dir_all(&dumps).expect("dumps dir");

    dump_container(&dumps, &synthetic_compute(101));
    dump_container(&dumps, &synthetic_compute(202));
    dump_container(&dumps, &synthetic_compute(303));

    let exe = env!("CARGO_BIN_EXE_decant_shader_warm");
    let first = Command::new(exe)
        .args(["--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--top", "2", "--threads", "2"])
        .output()
        .expect("run first corpus process");
    let first_stdout = String::from_utf8_lossy(&first.stdout);
    let first_stderr = String::from_utf8_lossy(&first.stderr);
    assert!(first.status.success(), "first corpus run failed:\nstdout:\n{first_stdout}\nstderr:\n{first_stderr}");
    assert!(first_stdout.contains("decant_shader_warm: corpus mode keys will not match the game keys without a manifest"));
    let s1 = parse_summary(&first_stdout);
    assert_eq!(s1.get("pipelines").map(String::as_str), Some("2"));
    assert_eq!(s1.get("ready").map(String::as_str), Some("2"));
    assert_eq!(s1.get("stages_compiled").map(String::as_str), Some("2"));

    let second = Command::new(exe)
        .args(["--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--top", "2", "--threads", "2"])
        .output()
        .expect("run second corpus process");
    let second_stdout = String::from_utf8_lossy(&second.stdout);
    assert!(second.status.success());
    let s2 = parse_summary(&second_stdout);
    assert_eq!(s2.get("hits").map(String::as_str), Some("2"));
    assert_eq!(s2.get("stages_compiled").map(String::as_str), Some("0"));
    assert_eq!(s2.get("stage_cache_hits").map(String::as_str), Some("2"));
}

#[test]
fn corpus_pipeline_from_manifest_hits_same_key_as_game_manifest_pipeline() {
    if let Err(e) = decant_dxc::Dxc::load() {
        println!("skipping decant_shader_warm manifest corpus key test without DXC: {e}");
        return;
    }

    let tmp = TempDir::new("manifest_corpus_key");
    let dumps = tmp.0.join("dumps");
    let cache = tmp.0.join("cache");
    let manifest_path = tmp.0.join("game.manifest");
    std::fs::create_dir_all(&dumps).expect("dumps dir");
    let cs = dump_container(&dumps, &synthetic_compute(4242));
    let linkage = Linkage { stages: vec![StageKind::Compute], sample_count: 7, ..Linkage::default() };
    let root = layer_root_layout();
    {
        let mut writer = Writer::open(&manifest_path).expect("open game manifest");
        assert!(writer.record(Some(&root), &linkage, &[&cs]).expect("record game pipeline"));
    }

    let exe = env!("CARGO_BIN_EXE_decant_shader_warm");
    let corpus = Command::new(exe)
        .args(["--manifest", manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "1"])
        .output()
        .expect("warm manifest corpus");
    let corpus_stdout = String::from_utf8_lossy(&corpus.stdout);
    assert!(corpus.status.success(), "manifest corpus failed:\n{corpus_stdout}\n{}", String::from_utf8_lossy(&corpus.stderr));
    assert_eq!(parse_summary(&corpus_stdout).get("stages_compiled").map(String::as_str), Some("1"));

    let game = Command::new(exe)
        .args(["--manifest", manifest_path.to_str().unwrap(), "--dump-dir", dumps.to_str().unwrap(), "--cache", cache.to_str().unwrap(), "--threads", "1"])
        .output()
        .expect("warm game manifest pipeline");
    let game_stdout = String::from_utf8_lossy(&game.stdout);
    assert!(game.status.success(), "game manifest run failed:\n{game_stdout}\n{}", String::from_utf8_lossy(&game.stderr));
    let summary = parse_summary(&game_stdout);
    assert_eq!(summary.get("hits").map(String::as_str), Some("1"));
    assert_eq!(summary.get("stages_compiled").map(String::as_str), Some("0"));
    assert_eq!(summary.get("stage_cache_hits").map(String::as_str), Some("1"));
}
