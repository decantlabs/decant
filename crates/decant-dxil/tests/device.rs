//! The test shaders on a real D3D12 device (default adapter, WARP as fallback): pipeline creation
//! for every shader, dispatch and read back for the buffer shaders and the loop. Skipped with a
//! message when no device runs Shader Model 6.6.
#![cfg(windows)]

mod common;

use common::gpu::{Bind, Gpu, bytes_of, open_device, words};
use common::shaders::{self, ARITH_OUTPUTS, ArithInput};
use common::write_and_validate;

fn device() -> Option<Gpu> {
    let gpu = open_device();
    match &gpu {
        Some(g) => println!("device: {}", g.name),
        None => println!("SKIPPED: no D3D12 device with Shader Model 6.6"),
    }
    gpu
}

#[test]
fn empty_compute_creates_a_pipeline() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("empty", &shaders::empty());
    gpu.pipeline(&c, &[]).expect("pipeline");
    // It also runs.
    gpu.run(&c, &[], [1, 1, 1]).expect("dispatch");
}

#[test]
fn thread_id_store_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("thread_id_store", &shaders::thread_id_store());
    let out = gpu.run(&c, &[Bind::Uav(128 * 4)], [2, 1, 1]).expect("run");
    assert_eq!(words(&out[0]), (0..128).collect::<Vec<u32>>());
}

#[test]
fn buffer_copy_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("buffer_copy", &shaders::buffer_copy());
    let input: Vec<u32> = (0..64 * 4).map(|i: u32| i.wrapping_mul(2_654_435_761) ^ 0xA5A5_0000).collect();
    let out = gpu.run(&c, &[Bind::Srv(bytes_of(&input)), Bind::Uav(input.len() * 4)], [1, 1, 1]).expect("run");
    assert_eq!(words(&out[0]), input);
}

#[test]
fn arithmetic_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("arithmetic", &shaders::arithmetic());
    let cases = [
        ArithInput { a: 1000, b: 7, x: 2.5, y: -3.0, z: 0.25 },
        ArithInput { a: 0xF0F0_1234, b: 5, x: 123.75, y: 1e9, z: -1.5 },
        ArithInput { a: 0x8000_0000, b: 33, x: -7.9, y: 7.9, z: 3.0 },
        ArithInput { a: 1, b: 0xFFFF_FFFF, x: 0.0, y: -0.0, z: 1.0 },
    ];
    for case in &cases {
        let out = gpu.run(&c, &[Bind::Cbv(case.bytes()), Bind::Uav(ARITH_OUTPUTS * 4)], [1, 1, 1]).expect("run");
        let got = words(&out[0]);
        let want = shaders::arith_reference(case);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "output {i} for a={:#x} b={:#x}: got {g:#x}, want {w:#x}", case.a, case.b);
        }
    }
}

#[test]
fn wave_ops_run_for_every_wave_size_the_device_supports() {
    let Some(gpu) = device() else { return };
    let (min, max) = gpu.wave_lane_range();
    let mut ran = 0;
    for size in [4u32, 8, 16, 32, 64] {
        let c = write_and_validate(&format!("wave_ops_{size}"), &shaders::wave_ops(size));
        if min == 0 || !(min..=max).contains(&size) {
            println!("SKIPPED wave size {size}: the device reports {min} to {max}");
            continue;
        }
        let out = gpu.run(&c, &[Bind::Uav(64 * shaders::WAVE_WORDS * 4)], [1, 1, 1]).expect("run");
        let got = words(&out[0]);
        for t in 0..64u32 {
            let at = t as usize * shaders::WAVE_WORDS;
            assert_eq!(got[at..at + shaders::WAVE_WORDS], shaders::wave_reference(t, size), "wave size {size}, thread {t}");
        }
        ran += 1;
    }
    println!("wave operations ran with {ran} wave sizes");
}

#[test]
fn heap_uav_runs_through_the_descriptor_heap() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("heap_uav", &shaders::heap_uav());
    // The index 2 comes from the constant buffer, slots 0 and 1 hold decoys.
    let binds = [Bind::Cbv(bytes_of(&[2, 0, 0, 0])), Bind::HeapUav { slot: 0, size: 256 }, Bind::HeapUav { slot: 1, size: 256 }, Bind::HeapUav { slot: 2, size: 256 }];
    let out = gpu.run(&c, &binds, [1, 1, 1]).expect("run");
    assert!(words(&out[0]).iter().all(|&w| w == 0), "decoy 0 untouched");
    assert!(words(&out[1]).iter().all(|&w| w == 0), "decoy 1 untouched");
    assert_eq!(words(&out[2]), (100..164).collect::<Vec<u32>>());
}

#[test]
fn structured_copy_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("structured_copy", &shaders::structured_copy());
    let input: Vec<u32> = (0..64 * 4).map(|i: u32| i * 3 + 1).collect();
    let out = gpu.run(&c, &[Bind::Srv(bytes_of(&input)), Bind::Uav(input.len() * 4)], [1, 1, 1]).expect("run");
    let got = words(&out[0]);
    for t in 0..64usize {
        assert_eq!(&got[t * 4..t * 4 + 3], &input[t * 4 + 1..t * 4 + 4], "element {t}");
    }
}

#[test]
fn loop_with_branch_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("loop_with_branch", &shaders::loop_with_branch());
    for n in [0u32, 1, 7, 1000] {
        let out = gpu.run(&c, &[Bind::Cbv(bytes_of(&[n, 0, 0, 0])), Bind::Uav(16)], [1, 1, 1]).expect("run");
        assert_eq!(words(&out[0])[0], shaders::loop_reference(n), "n = {n}");
    }
}

#[test]
fn groupshared_barrier_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("groupshared_barrier", &shaders::groupshared_barrier());
    let out = gpu.run(&c, &[Bind::Uav(128 * 4)], [2, 1, 1]).expect("run");
    let want: Vec<u32> = (0..128).map(|t| 63 - t % 64).collect();
    assert_eq!(words(&out[0]), want);
}

#[test]
fn atomics_run() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("atomics", &shaders::atomics());
    let out = gpu.run(&c, &[Bind::Uav(16)], [2, 1, 1]).expect("run");
    assert_eq!(words(&out[0]), vec![128, 64, 2, 5], "total threads, per group counter, one winner per group, one compare exchange");
}

/// `createHandleFromBinding` takes the absolute register, which must lie inside the range of the
/// binding. A hardware driver lets index 0 for u27 pass, WARP with GPU based validation removes the
/// device (`DECANT_SHADER_DEVICE=warp DECANT_SHADER_AB_DEBUG=2`).
#[test]
fn a_resource_at_a_high_register_runs() {
    let Some(gpu) = device() else { return };
    let c = write_and_validate("uav_at_register_27", &shaders::thread_id_store_at(27));
    let out = gpu.run(&c, &[Bind::UavAt { register: 27, size: 128 * 4 }], [2, 1, 1]).expect("run");
    assert_eq!(words(&out[0]), (0..128).collect::<Vec<u32>>());
}
