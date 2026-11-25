//! The shapes of the translator's real inputs on WARP, which removes its device for things a
//! hardware driver lets pass: descriptor heap slots read as another kind than they hold, binding
//! indices outside their range, a pixel shader that reads clip distances. Each test runs a side
//! against itself and must come back identical. Skips without WARP or `dxcompiler.dll`.
#![cfg(windows)]

use decant_shader_ab::compare::{Tolerance, Verdict};
use decant_shader_ab::model::{Class, Stage};
use decant_shader_ab::pairs::Outcome;
use decant_shader_ab::plan::{Options, Param, Range, Res, Root};
use decant_shader_ab::win::device::open_device_kind;
use decant_shader_ab::win::dxc::Dxc;
use decant_shader_ab::win::{Session, run_pair};

struct Env {
    dxc: Dxc,
    dir: tempfile::TempDir,
}

fn env() -> Option<Env> {
    let dxc = match Dxc::load() {
        Ok(d) => d,
        Err(e) => {
            println!("SKIPPED: {e}");
            return None;
        }
    };
    if open_device_kind("warp").is_none() {
        println!("SKIPPED: no WARP device with Shader Model 6.0");
        return None;
    }
    Some(Env { dxc, dir: tempfile::tempdir().expect("temp dir") })
}

impl Env {
    fn file(&self, name: &str, bytes: &[u8]) -> String {
        let p = self.dir.path().join(name);
        std::fs::write(&p, bytes).expect("write");
        p.to_string_lossy().into_owned()
    }

    fn run(&self, a: &str, b: &str, stage: Stage, root: Option<&std::path::Path>) -> Outcome {
        let mut session = Session::on("warp");
        let o = run_pair(&mut session, &self.dxc, "t", a, b, Some(stage), root, "6_6", Options::default(), Tolerance::default());
        println!("{}", o.line());
        for n in &o.notes {
            println!("note: {n}");
        }
        o
    }

    fn same(&self, name: &str, src: &str, stage: Stage) -> Outcome {
        let f = self.file(name, src.as_bytes());
        let o = self.run(&f, &f, stage, None);
        assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
        o
    }
}

/// A pixel shader of the direct translator's shape: heap slots indexed by the user data mirror
/// plus a constant, read as a texture, a constant buffer, a byte address buffer and a sampler.
const HEAP_PIXEL: &str = "
struct decant_words { uint4 w[4096]; };
cbuffer decant_mirror_cb : register(b1, space15) { uint4 decant_mirror[8]; };
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    Texture2D<float4> t = ResourceDescriptorHeap[decant_mirror[2].y + 3u];
    ConstantBuffer<decant_words> c = ResourceDescriptorHeap[decant_mirror[2].y + 5u];
    ByteAddressBuffer b = ResourceDescriptorHeap[decant_mirror[2].x + 1u];
    SamplerState s = SamplerDescriptorHeap[0u];
    return t.SampleLevel(s, uv, 0) + asfloat(c.w[1]) * K + asfloat(b.Load4(16));
}
";

#[test]
fn heap_slots_get_resources_of_the_kind_the_shader_reads() {
    let Some(e) = env() else { return };
    let o = e.same("heap.hlsl", &HEAP_PIXEL.replace("K", "1.0"), Stage::Pixel);
    assert!(o.notes.iter().any(|n| n.contains("4 slots of known kind")), "{:?}", o.notes);
    // The constant buffer slot holds data: changing what is read from it changes the output.
    let a = e.file("a.hlsl", HEAP_PIXEL.replace("K", "1.0").as_bytes());
    let b = e.file("b.hlsl", HEAP_PIXEL.replace("K", "2.0").as_bytes());
    assert_eq!(e.run(&a, &b, Stage::Pixel, None).verdict, Verdict::Different);
}

#[test]
fn a_vertex_shader_with_system_value_inputs_and_a_clip_distance_runs() {
    let Some(e) = env() else { return };
    e.same(
        "vs.hlsl",
        "cbuffer C : register(b0, space1) { float4 k[16]; };
ByteAddressBuffer B : register(t2, space1);
struct O { float4 p : SV_Position; float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1; float c : SV_ClipDistance0; };
O main(float3 pos : POSITION, float2 uv : TEXCOORD0, float col : COLOR0, uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    uint n; B.GetDimensions(n);
    O o; o.p = float4(pos, 1) * k[0]; o.t0 = float4(uv, col, asfloat(B.Load((vid * 4) % n))); o.t1 = k[1] + iid; o.c = pos.x; return o; }",
        Stage::Vertex,
    );
}

#[test]
fn a_pixel_shader_that_reads_a_clip_distance_gets_it_from_the_feeder() {
    let Some(e) = env() else { return };
    e.same("clip.hlsl", "float4 main(float4 p : SV_Position, float4 c : SV_ClipDistance0) : SV_Target { return c; }", Stage::Pixel);
}

#[test]
fn a_given_root_signature_with_tables_is_used_as_it_is() {
    let Some(e) = env() else { return };
    let range = |class, base, space| Range { class, base, space, declared: 1, offset: None, slots: Vec::new() };
    let root = Root {
        flags: decant_shader_ab::plan::FLAG_INPUT_LAYOUT,
        params: vec![
            Param::Table(vec![range(Class::Srv, 0, 0), range(Class::Cbv, 4, 1)]),
            Param::Table(vec![range(Class::Sampler, 0, 0)]),
            Param::Constants { register: 0, space: 2, count: 4 },
            Param::Root { class: Class::Cbv, register: 1, space: 15, res: Res::Null },
        ],
        samplers: Vec::new(),
        blob: None,
    };
    let blob = decant_shader_ab::win::root::serialize(&root).expect("serialize");
    let rs = e.dir.path().join("root.bin");
    std::fs::write(&rs, blob).expect("write");
    let src = "Texture2D t : register(t0); SamplerState s : register(s0); cbuffer C : register(b4, space1) { float4 k; };
cbuffer R : register(b0, space2) { float4 r; }; cbuffer M : register(b1, space15) { uint4 m; };
float4 main(float4 p : SV_Position, float2 uv : TEXCOORD0) : SV_Target { return t.Sample(s, uv) * k + r + m.x; }";
    let f = e.file("tables.hlsl", src.as_bytes());
    let o = e.run(&f, &f, Stage::Pixel, Some(&rs));
    assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
}

#[test]
fn a_binding_index_outside_its_range_is_rewritten_to_the_register() {
    let Some(e) = env() else { return };
    // The old DXIL writer passes index 0 for a binding at b27: made here from a DXC module by
    // rewriting the index, as the old override containers have it.
    let src = "cbuffer C : register(b27, space1) { uint4 a[16]; };
float4 main(float4 p : SV_Position, float4 c : COLOR0) : SV_Target { return asfloat(a[2]) * c; }";
    let good = e.dxc.compile(src, "main", "ps_6_6", &[]).expect("compile");
    let text = e.dxc.disassemble(&good).expect("disassemble");
    let broken = text.replace("i32 27, i32 27, i32 1, i8 2 }, i32 27,", "i32 27, i32 27, i32 1, i8 2 }, i32 0,");
    assert_ne!(broken, text, "the handle of b27 is where the test expects it");
    let bad = e.dxc.assemble(&broken).expect("assemble");
    let f = e.file("old.dxil", &bad);
    let o = e.run(&f, &f, Stage::Pixel, None);
    assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
    assert!(o.notes.iter().any(|n| n.starts_with("A: 1 createHandleFromBinding")), "{:?}", o.notes);
}
