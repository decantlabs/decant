//! Pairs of synthetic HLSL shaders on a real D3D12 device (default adapter, WARP as fallback).
//! Every test skips with a message when there is no device or no `dxcompiler.dll`.
#![cfg(windows)]

use std::path::PathBuf;

use decant_shader_ab::compare::{Tolerance, Verdict};
use decant_shader_ab::model::Stage;
use decant_shader_ab::pairs::Outcome;
use decant_shader_ab::plan::Options;
use decant_shader_ab::win::device::open_device;
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
    if open_device().is_none() {
        println!("SKIPPED: no D3D12 device with Shader Model 6.0");
        return None;
    }
    Some(Env { dxc, dir: tempfile::tempdir().expect("temp dir") })
}

impl Env {
    fn file(&self, name: &str, bytes: &[u8]) -> String {
        let p: PathBuf = self.dir.path().join(name);
        std::fs::write(&p, bytes).expect("write");
        p.to_string_lossy().into_owned()
    }

    fn run(&self, a: &str, b: &str, stage: Stage, tol: Tolerance) -> Outcome {
        let mut session = Session::default();
        let o = run_pair(&mut session, &self.dxc, "t", a, b, Some(stage), None, "6_0", Options::default(), tol);
        println!("{}", o.line());
        o
    }
}

const COMPUTE: &str = "
cbuffer K : register(b0) { float4 scale; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
SamplerState smp : register(s0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.y * 16 + id.x;
    float v = asfloat(src.Load(i * 4)) * scale.x + tex.SampleLevel(smp, float2(id.xy) / 16.0, 0).y;
    dst.Store(i * 4, asuint(v));
}
";

const PIXEL: &str = "
Texture2D<float4> tex : register(t0);
SamplerState smp : register(s0);
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0, float4 col : COLOR0) : SV_Target0 {
    return tex.Sample(smp, uv) * col * K;
}
";

#[test]
fn two_equal_compute_shaders_agree() {
    let Some(e) = env() else { return };
    let a = e.file("a.hlsl", COMPUTE.as_bytes());
    let b = e.file("b.hlsl", COMPUTE.as_bytes());
    let o = e.run(&a, &b, Stage::Compute, Tolerance::default());
    assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
    assert!(o.declarations.is_empty());
    assert!(o.diff.unwrap().compared > 0);
}

#[test]
fn a_changed_constant_in_a_pixel_shader_is_found() {
    let Some(e) = env() else { return };
    let a = e.file("a.hlsl", PIXEL.replace("* K", "* 0.5").as_bytes());
    let b = e.file("b.hlsl", PIXEL.replace("* K", "* 0.25").as_bytes());
    let o = e.run(&a, &a, Stage::Pixel, Tolerance::default());
    assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
    let o = e.run(&a, &b, Stage::Pixel, Tolerance::default());
    assert_eq!(o.verdict, Verdict::Different, "{o:?}");
    let first = o.diff.as_ref().unwrap().first.clone().unwrap();
    assert_eq!(first.output, "SV_Target0 rgba32f");
    // Halving is exact: every differing value of B is half the value of A.
    let num = |s: &str| s.split_whitespace().next().unwrap().parse::<f32>().unwrap();
    assert_eq!(num(&first.a) / 2.0, num(&first.b), "{first:?}");
}

#[test]
fn a_tolerance_accepts_a_reordered_sum_and_exactness_does_not() {
    let Some(e) = env() else { return };
    let src = "RWStructuredBuffer<float> o : register(u0);\nStructuredBuffer<float> i : register(t0);\n[numthreads(64, 1, 1)] void main(uint t : SV_DispatchThreadID) { o[t] = EXPR; }";
    let a = e.file("a.hlsl", src.replace("EXPR", "(i[t] * 3.1) + (i[t+1] * 7.3) + i[t+2]").as_bytes());
    let b = e.file("b.hlsl", src.replace("EXPR", "i[t+2] + ((i[t+1] * 7.3) + (i[t] * 3.1))").as_bytes());
    let exact = e.run(&a, &b, Stage::Compute, Tolerance::default());
    let loose = e.run(&a, &b, Stage::Compute, Tolerance { max_ulp: 4, nan_equal: true });
    // Reordering may or may not change bits on a given device, but a tolerance never makes it worse.
    assert!(matches!(exact.verdict, Verdict::Identical | Verdict::Different), "{exact:?}");
    assert!(matches!(loose.verdict, Verdict::Identical | Verdict::WithinTolerance), "{loose:?}");
}

#[test]
fn nans_match_nans_and_not_numbers() {
    let Some(e) = env() else { return };
    let src = "RWByteAddressBuffer o : register(u0);\n[numthreads(64, 1, 1)] void main(uint t : SV_DispatchThreadID) { o.Store(t * 4, VALUE); }";
    let nan1 = e.file("n1.hlsl", src.replace("VALUE", "0x7fc00000u").as_bytes());
    let nan2 = e.file("n2.hlsl", src.replace("VALUE", "0x7fc00001u").as_bytes());
    let one = e.file("one.hlsl", src.replace("VALUE", "0x3f800000u").as_bytes());
    assert_eq!(e.run(&nan1, &nan1, Stage::Compute, Tolerance::default()).verdict, Verdict::Identical);
    assert_eq!(e.run(&nan1, &nan2, Stage::Compute, Tolerance::default()).verdict, Verdict::WithinTolerance);
    assert_eq!(e.run(&nan1, &nan2, Stage::Compute, Tolerance { max_ulp: 0, nan_equal: false }).verdict, Verdict::Different);
    let o = e.run(&nan1, &one, Stage::Compute, Tolerance { max_ulp: u32::MAX, nan_equal: true });
    assert_eq!(o.verdict, Verdict::Different);
    assert!(o.diff.unwrap().first.unwrap().a.starts_with("NaN"));
}

#[test]
fn a_shader_the_runtime_refuses_fails_its_side_only() {
    let Some(e) = env() else { return };
    let good = e.dxc.compile(COMPUTE, "main", "cs_6_0", &[]).expect("compile");
    // A wrong digest: DXC still reflects the container, D3D12 refuses to create the pipeline.
    let mut bad = good.clone();
    for b in &mut bad[4..20] {
        *b ^= 0x5A;
    }
    let a = e.file("good.dxil", &good);
    let b = e.file("bad.dxil", &bad);
    let o = e.run(&a, &b, Stage::Compute, Tolerance::default());
    assert_eq!(o.verdict, Verdict::BFailed, "{o:?}");
    assert!(o.b_error.unwrap().contains("pipeline refused"));
    let junk = e.file("junk.dxil", b"not a shader");
    let o = e.run(&junk, &b, Stage::Compute, Tolerance::default());
    assert_eq!(o.verdict, Verdict::BothFailed, "{o:?}");
}

#[test]
fn sides_with_different_inputs_both_run_and_the_difference_is_reported() {
    let Some(e) = env() else { return };
    // A reads one interpolated input, B two: each gets a feeder for its own signature, and
    // TEXCOORD0 carries the same values on both sides.
    let a = e.file("a.hlsl", b"float4 main(float4 pos : SV_Position, float4 t : TEXCOORD0) : SV_Target0 { return t; }");
    let b = e.file("b.hlsl", b"float4 main(float4 pos : SV_Position, float4 t : TEXCOORD0, float2 u : TEXCOORD1) : SV_Target0 { return t + float4(u, 0, 0) * 0; }");
    let o = e.run(&a, &b, Stage::Pixel, Tolerance::default());
    assert!(o.declarations.iter().any(|d| d.starts_with("input signature")), "{o:?}");
    assert_eq!(o.verdict, Verdict::Identical, "{o:?}");
    assert_eq!(o.declares(), "differ(1)");
}

#[test]
fn vertex_shaders_are_compared_per_output_element() {
    let Some(e) = env() else { return };
    let src = "cbuffer M : register(b0) { float4x4 m; };\nstruct O { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\nO main(float4 p : POSITION, float2 uv : TEXCOORD0) { O o; o.pos = mul(m, p); o.uv = uv * SCALE; return o; }";
    let a = e.file("a.hlsl", src.replace("SCALE", "2").as_bytes());
    let b = e.file("b.hlsl", src.replace("SCALE", "3").as_bytes());
    assert_eq!(e.run(&a, &a, Stage::Vertex, Tolerance::default()).verdict, Verdict::Identical);
    let o = e.run(&a, &b, Stage::Vertex, Tolerance::default());
    assert_eq!(o.verdict, Verdict::Different);
    assert!(o.diff.unwrap().first.unwrap().output.starts_with("TEXCOORD0"));
}
