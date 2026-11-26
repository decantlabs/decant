//! Device test of the emulated 64 lane wave (`decant_shader::hlsl::wave_emu`): a compute shader
//! written as the translator prints a console wave64 program (`[WaveSize(64)]`, lane reads,
//! a 64 lane ballot, the lane index and count) goes through `emulate_wave64`, DXC and the GPU, and
//! what every thread stored is compared with the console's answer worked out on the CPU. Runs on
//! the default adapter and on WARP, each skipped when it has no 12_0 device; skipped without DXC.
//!
//! On a GPU whose waves have 32 lanes (an RTX 4080) the text before emulation is refused
//! (`E_INVALIDARG`, debug layer message 622); the emulated text must give the console's values.
#![allow(unsafe_code)]

use windows::Win32::Graphics::Direct3D12::*;

use crate::view_gpu_tests::{HostBuffer, devices, pipeline, root_signature, run};

/// Words each thread stores.
const WORDS: usize = 8;
/// Groups dispatched.
const GROUPS: usize = 4;
/// Threads of a group: one console wave.
const THREADS: usize = 64;

/// The text as the translator prints it for a 64 lane console program.
const NATIVE: &str = "RWByteAddressBuffer decant_out : register(u0);
uint decant_read_lane(uint value, uint lane) { return WaveReadLaneAt(value, lane); }
[WaveSize(64)]
[numthreads(8, 8, 1)]
void main(uint3 decant_gid : SV_GroupID, uint decant_gi : SV_GroupIndex) {
    uint g = decant_gid.x;
    uint v = decant_gi * 7u + g * 1000u + 3u;
    uint a = decant_read_lane(v, 0u);
    uint b = decant_read_lane(v, 33u);
    uint c = decant_read_lane(v, 63u - (decant_gi & 63u));
    uint4 m = WaveActiveBallot((v % 3u) == 0u);
    uint n = WaveGetLaneCount();
    uint l = WaveGetLaneIndex();
    float f = asfloat(decant_read_lane(asuint(float(v) * 0.5f), 17u));
    uint at = (g * 64u + decant_gi) * 32u;
    decant_out.Store(at, a);
    decant_out.Store(at + 4u, b);
    decant_out.Store(at + 8u, c);
    decant_out.Store(at + 12u, m.x);
    decant_out.Store(at + 16u, m.y);
    decant_out.Store(at + 20u, n);
    decant_out.Store(at + 24u, l);
    decant_out.Store(at + 28u, asuint(f));
}
";

/// What thread `i` of group `g` stores on the console.
fn console(g: usize, i: usize) -> [u32; WORDS] {
    let v = |lane: usize| (lane * 7 + g * 1000 + 3) as u32;
    let (mut lo, mut hi) = (0u32, 0u32);
    for lane in 0..64 {
        if v(lane) % 3 == 0 {
            if lane < 32 {
                lo |= 1 << lane;
            } else {
                hi |= 1 << (lane - 32);
            }
        }
    }
    [v(0), v(33), v(63 - i), lo, hi, 64, i as u32, (v(17) as f32 * 0.5).to_bits()]
}

#[test]
fn an_emulated_64_lane_wave_gives_the_console_values() {
    let text = decant_shader::hlsl::wave_emu::emulate_wave64(NATIVE).expect("the program is emulated");
    assert!(!text.contains("[WaveSize("), "{text}");
    let dxc = match decant_dxc::Dxc::load() {
        Ok(d) => d,
        Err(e) => {
            println!("no DXC ({e}): the emulated wave device test is skipped");
            return;
        }
    };
    let code = dxc.compile(&text, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{text}"));
    for (name, device) in devices() {
        let signature = root_signature(
            &device,
            &[D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            }],
        );
        let state = pipeline(&device, &signature, &code);
        let out = HostBuffer::new(&device, GROUPS * THREADS * WORDS);
        out.fill(|_| 0xdead_beef);
        assert!(
            run(&device, |list| {
                // SAFETY: the signature, the pipeline and the buffer outlive the run.
                unsafe {
                    list.SetComputeRootSignature(&signature);
                    list.SetPipelineState(&state);
                    list.SetComputeRootUnorderedAccessView(0, out.res.GetGPUVirtualAddress());
                    list.Dispatch(GROUPS as u32, 1, 1);
                }
            }),
            "{name}: the dispatch runs"
        );
        let words = out.words();
        for g in 0..GROUPS {
            for i in 0..THREADS {
                let at = (g * THREADS + i) * WORDS;
                assert_eq!(&words[at..at + WORDS], &console(g, i), "{name}: group {g} thread {i} (a, b, c, ballot lo, ballot hi, count, lane, float)");
            }
        }
        println!("{name}: {} threads give the console's lane reads and ballot", GROUPS * THREADS);
    }
}
