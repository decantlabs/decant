//! The DSP work: FFT, inverse FFT and partitioned convolution against straightforward reference computations.
#![allow(unsafe_code)]

mod common;

use std::f64::consts::PI;

use chad::abi::*;
use common::*;

const IN: usize = 0;
const OUT: usize = 0x4000;
const AUX: usize = 0x8000;

/// The packed spectrum by the definition of the DFT.
fn dft_packed(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let bin = |k: usize| {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, v) in x.iter().enumerate() {
            let a = -2.0 * PI * ((k * i) % n) as f64 / n as f64;
            re += v * a.cos();
            im += v * a.sin();
        }
        (re, im)
    };
    let mut out = vec![0.0; n];
    out[0] = bin(0).0;
    out[1] = bin(n / 2).0;
    for k in 1..n / 2 {
        let (re, im) = bin(k);
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
    out
}

fn close(a: &[f32], b: &[f64], tolerance: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((f64::from(*x) - y).abs() <= tolerance, "{what} [{i}]: {x} against {y}");
    }
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

fn fft_rig(process: u32, flags: u32, block: u32) -> Rig {
    Rig::new(&params(process, flags, block, 1, &[]))
}

#[test]
fn the_forward_fft_matches_the_dft_for_every_block_size() {
    let _s = serial();
    for block in [0x200usize, 0x400, 0x800] {
        let mut rig = fft_rig(PROCESS_FFT, 0, block as u32);
        let x = noise(block, block as u64);
        rig.write(IN, &x);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        let want = dft_packed(&f64s(&x));
        close(&rig.read(OUT, block), &want, 2e-3, &format!("block {block:#x}"));
    }
}

#[test]
fn gains_and_the_block_multiplier_apply_to_the_fft() {
    let _s = serial();
    let mut rig = fft_rig(PROCESS_FFT, 0, 0x200);
    let x = noise(0x200, 1);
    let window: Vec<f32> = (0..0x200).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / 512.0).cos() as f32).collect();
    rig.write(IN, &x);
    rig.write(AUX, &window);
    let c = rig.command(2.0, 0.25, IN, OUT, Some(AUX));
    assert_eq!(rig.submit(&c).0, S_OK);
    let windowed: Vec<f64> = x.iter().zip(&window).map(|(a, w)| 2.0 * f64::from(*a) * f64::from(*w)).collect();
    let want: Vec<f64> = dft_packed(&windowed).iter().map(|v| v * 0.25).collect();
    close(&rig.read(OUT, 0x200), &want, 2e-3, "windowed");
}

#[test]
fn polar_format_gives_magnitude_and_phase_and_leaves_the_first_pair() {
    let _s = serial();
    let mut rig = fft_rig(PROCESS_FFT, FLAG_POLAR, 0x200);
    let x = noise(0x200, 2);
    rig.write(IN, &x);
    let c = rig.command(1.0, 1.0, IN, OUT, None);
    assert_eq!(rig.submit(&c).0, S_OK);
    let rect = dft_packed(&f64s(&x));
    let polar = rig.read(OUT, 0x200);
    assert!((f64::from(polar[0]) - rect[0]).abs() < 2e-3, "DC as it is");
    assert!((f64::from(polar[1]) - rect[1]).abs() < 2e-3, "Nyquist as it is");
    for k in 1..0x100 {
        let (re, im) = (rect[2 * k], rect[2 * k + 1]);
        assert!((f64::from(polar[2 * k]) - re.hypot(im)).abs() < 2e-3, "magnitude {k}");
        assert!((f64::from(polar[2 * k + 1]) - im.atan2(re)).abs() < 1e-3, "phase {k}");
    }
    // The output gain scales the magnitude, not the phase.
    let c = rig.command(1.0, 3.0, IN, OUT, None);
    assert_eq!(rig.submit(&c).0, S_OK);
    let scaled = rig.read(OUT, 0x200);
    assert!((f64::from(scaled[6]) - 3.0 * f64::from(polar[6])).abs() < 5e-3);
    assert!((scaled[7] - polar[7]).abs() < 1e-5);
}

#[test]
fn the_inverse_fft_undoes_the_forward_fft_and_scales_by_one_over_n() {
    let _s = serial();
    for (block, flags) in [(0x200usize, 0), (0x400, FLAG_POLAR), (0x800, 0)] {
        let mut fwd = fft_rig(PROCESS_FFT, flags, block as u32);
        let mut inv = fft_rig(PROCESS_INVERSE_FFT, flags, block as u32);
        let x = noise(block, 5);
        fwd.write(IN, &x);
        let c = fwd.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(fwd.submit(&c).0, S_OK);
        let spectrum = fwd.read(OUT, block);
        inv.write(IN, &spectrum);
        let c = inv.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(inv.submit(&c).0, S_OK);
        close(&inv.read(OUT, block), &f64s(&x), 1e-4, &format!("round trip {block:#x} flags {flags}"));
    }
}

#[test]
fn the_inverse_fft_of_a_known_spectrum() {
    let _s = serial();
    let mut inv = fft_rig(PROCESS_INVERSE_FFT, 0, 0x200);
    // A DC bin of 512 is a constant of 1.0 after the scaling by 1 / N. A cosine of bin 4 with N / 2 gives amplitude 1.
    let mut spectrum = vec![0f32; 0x200];
    spectrum[0] = 512.0;
    spectrum[8] = 256.0;
    inv.write(IN, &spectrum);
    let c = inv.command(1.0, 2.0, IN, OUT, None);
    assert_eq!(inv.submit(&c).0, S_OK);
    let want: Vec<f64> = (0..512).map(|i| 2.0 * (1.0 + (2.0 * PI * 4.0 * i as f64 / 512.0).cos())).collect();
    close(&inv.read(OUT, 512), &want, 1e-4, "dc plus cosine");
    // Input gain scales the spectrum.
    let c = inv.command(0.5, 1.0, IN, OUT, None);
    assert_eq!(inv.submit(&c).0, S_OK);
    assert!((inv.read(OUT, 1)[0] - 1.0).abs() < 1e-4, "0.5 * 2 at sample 0");
    // Polar input: magnitude 256, phase pi / 2 at bin 4 is a sine.
    let mut polar = fft_rig(PROCESS_INVERSE_FFT, FLAG_POLAR, 0x200);
    let mut sp = vec![0f32; 0x200];
    sp[8] = 256.0;
    sp[9] = (PI / 2.0) as f32;
    polar.write(IN, &sp);
    let c = polar.command(1.0, 1.0, IN, OUT, None);
    assert_eq!(polar.submit(&c).0, S_OK);
    let want: Vec<f64> = (0..512).map(|i| -(2.0 * PI * 4.0 * i as f64 / 512.0).sin()).collect();
    close(&polar.read(OUT, 512), &want, 1e-4, "polar sine");
}

#[test]
fn values_that_are_not_finite_become_zero() {
    let _s = serial();
    let mut rig = fft_rig(PROCESS_FFT, 0, 0x200);
    let mut x = noise(0x200, 9);
    let mut clean = x.clone();
    for (i, v) in [(3, f32::NAN), (10, f32::INFINITY), (11, f32::NEG_INFINITY)] {
        x[i] = v;
        clean[i] = 0.0;
    }
    rig.write(IN, &x);
    let c = rig.command(1.0, 1.0, IN, OUT, None);
    assert_eq!(rig.submit(&c).0, S_OK);
    close(&rig.read(OUT, 0x200), &dft_packed(&f64s(&clean)), 2e-3, "non finite samples");
    // A non finite gain is a gain of zero.
    let c = rig.command(f32::NAN, 1.0, IN, OUT, None);
    assert_eq!(rig.submit(&c).0, S_OK);
    assert!(rig.read(OUT, 0x200).iter().all(|v| *v == 0.0));
}

#[test]
fn submit_validates_its_arguments() {
    let _s = serial();
    let mut rig = fft_rig(PROCESS_FFT, 0, 0x200);
    let good = rig.command(1.0, 1.0, IN, OUT, None);
    // SAFETY: null pointers are part of the contract.
    unsafe {
        assert_eq!(chad::exports::XDspSubmitCommand(std::ptr::null_mut(), &good, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(chad::exports::XDspSubmitCommand(rig.context, std::ptr::null(), std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(chad::exports::XDspSubmitCommand(0x1234 as *mut XdspContext, &good, std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(chad::exports::XDspSubmitCommandWithEnvelope(std::ptr::null_mut(), &good, std::ptr::null(), std::ptr::null_mut()), E_INVALIDARG);
        // A null sequence pointer is allowed.
        assert_eq!(chad::exports::XDspSubmitCommand(rig.context, &good, std::ptr::null_mut()), S_OK);
    }
    assert_eq!(rig.status()[0], 1);
    // Input, output and the multiplier must lie inside the client memory.
    let before = rig.status();
    let outside = XdspCommand { input: rig.addr(0) - 4, ..good };
    assert_eq!(rig.submit(&outside).0, E_INVALIDARG);
    let outside = XdspCommand { input: rig.addr(MEM_FLOATS - 0x1FF), ..good };
    assert_eq!(rig.submit(&outside).0, E_INVALIDARG, "ends one float too far");
    let inside = XdspCommand { input: rig.addr(MEM_FLOATS - 0x200), ..good };
    assert_eq!(rig.submit(&inside).0, S_OK, "ends exactly at the end");
    let outside = XdspCommand { output: rig.addr(MEM_FLOATS), ..good };
    assert_eq!(rig.submit(&outside).0, E_INVALIDARG);
    let outside = XdspCommand { output: 0, ..good };
    assert_eq!(rig.submit(&outside).0, E_INVALIDARG);
    let outside = XdspCommand { multiplier: rig.addr(MEM_FLOATS - 1), ..good };
    assert_eq!(rig.submit(&outside).0, E_INVALIDARG);
    rig.write(AUX, &[1.0; 0x200]);
    let ok = XdspCommand { multiplier: rig.addr(AUX), ..good };
    assert_eq!(rig.submit(&ok).0, S_OK);
    assert_eq!(rig.status()[0], before[0] + 2, "a refused command is not counted");
}

#[test]
fn sequence_numbers_and_the_status_array() {
    let _s = serial();
    let rig = fft_rig(PROCESS_FFT, 0, 0x200);
    let c = rig.command(1.0, 1.0, IN, OUT, None);
    assert_eq!(rig.status(), [0, 0, 0, 0]);
    for n in 1..=20u32 {
        assert_eq!(rig.submit(&c), (S_OK, n));
        assert_eq!(rig.status()[..2], [n, n], "completed equals submitted");
    }
    // The envelope variant counts in the same sequence, and ignores the envelope for FFT.
    assert_eq!(rig.submit_envelope(&c, 0), (S_OK, 21));
    // A refused command leaves the sequence output alone.
    let bad = XdspCommand { input: 0, ..c };
    assert_eq!(rig.submit(&bad), (E_INVALIDARG, 0xAAAA));
}

#[test]
fn the_sticky_error_stops_submissions_and_eight_outstanding_commands_have_no_slot() {
    let _s = serial();
    let rig = fft_rig(PROCESS_FFT, 0, 0x200);
    let c = rig.command(1.0, 1.0, IN, OUT, None);
    assert_eq!(rig.submit(&c).0, S_OK);
    // Eight commands submitted and not completed: no slot.
    rig.set_status(0, 9);
    assert_eq!(rig.submit(&c), (XDSP_E_NO_SLOT, 0xAAAA));
    rig.set_status(1, 1);
    assert_eq!(rig.submit(&c).0, XDSP_E_NO_SLOT, "9 - 1 = 8 outstanding");
    rig.set_status(1, 2);
    assert_eq!(rig.submit(&c), (S_OK, 10), "7 outstanding is allowed");
    rig.set_status(0, 1);
    rig.set_status(1, 1);
    // A negative HRESULT in status word 3 is returned by every submission, whatever else is wrong.
    rig.set_status(3, 0x8000_4005);
    assert_eq!(rig.submit(&c), (0x8000_4005_u32 as i32, 0xAAAA));
    let bad = XdspCommand { input: 0, ..c };
    assert_eq!(rig.submit(&bad).0, 0x8000_4005_u32 as i32);
    rig.set_status(3, 0);
    assert_eq!(rig.submit(&c).0, S_OK);
}

/// Direct convolution of `x` with the partitions of `h`, weighted per partition (the reference of the specification).
fn reference(h: &[f32], x: &[f32], block: usize, blocks: usize, gains: &dyn Fn(usize, usize) -> f64) -> Vec<f64> {
    let partitions = h.len() / block;
    let mut y = vec![0.0; blocks * block];
    for n in 0..blocks {
        for i in 0..block {
            let t = n * block + i;
            let mut acc = 0.0;
            for p in 0..partitions {
                let g = gains(n, p);
                for k in 0..block {
                    let src = t as i64 - (p * block + k) as i64;
                    if src >= 0 {
                        acc += g * f64::from(h[p * block + k]) * f64::from(x[src as usize]);
                    }
                }
            }
            y[t] = acc;
        }
    }
    y
}

fn decaying(n: usize, seed: u64) -> Vec<f32> {
    noise(n, seed).iter().enumerate().map(|(i, v)| v * (-(i as f32) / (n as f32 / 3.0)).exp()).collect()
}

/// A convolution rig. With flag 8 `ir` holds both impulse responses, one after the other, and the length given is the one of each.
fn conv_rig(flags: u32, block: u32, channels: u32, ir: &[f32]) -> Rig {
    let mut p = params(PROCESS_CONVOLUTION, flags, block, channels, ir);
    if flags & FLAG_STEREO_IR != 0 {
        p.ir_length /= 2;
    }
    Rig::new(&p)
}

const B: usize = 0x200;

#[test]
fn convolution_matches_direct_convolution_block_after_block() {
    let _s = serial();
    let h = decaying(9 * B, 3);
    let spectra = ir_spectra(&h, B);
    let mut rig = conv_rig(0, B as u32, 1, &spectra);
    let blocks = 12;
    let x = noise(blocks * B, 4);
    let want = reference(&h, &x, B, blocks, &|_, _| 1.0);
    for n in 0..blocks {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c), (S_OK, n as u32 + 1));
        close(&rig.read(OUT, B), &want[n * B..(n + 1) * B], 2e-3, &format!("block {n}"));
    }
}

#[test]
fn gains_apply_to_the_input_and_the_output_of_the_convolution() {
    let _s = serial();
    let h = decaying(9 * B, 5);
    let mut rig = conv_rig(0, B as u32, 1, &ir_spectra(&h, B));
    let x = noise(4 * B, 6);
    let want = reference(&h, &x, B, 4, &|_, _| 1.0);
    for n in 0..4 {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(0.5, 4.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        let scaled: Vec<f64> = want[n * B..(n + 1) * B].iter().map(|v| v * 2.0).collect();
        close(&rig.read(OUT, B), &scaled, 4e-3, &format!("block {n}"));
    }
}

#[test]
fn a_unit_impulse_response_passes_the_input_through() {
    let _s = serial();
    let mut h = vec![0f32; 9 * B];
    h[0] = 1.0;
    let mut rig = conv_rig(0, B as u32, 1, &ir_spectra(&h, B));
    let x = noise(3 * B, 7);
    for n in 0..3 {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, B), &f64s(&x[n * B..(n + 1) * B]), 1e-4, "pass through");
    }
    // A delay of 2 blocks: the IR has its one tap in partition 2.
    let mut h = vec![0f32; 9 * B];
    h[2 * B + 7] = 1.0;
    let mut rig = conv_rig(0, B as u32, 1, &ir_spectra(&h, B));
    let want = reference(&h, &x, B, 3, &|_, _| 1.0);
    for n in 0..3 {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, B), &want[n * B..(n + 1) * B], 1e-4, "delay");
    }
}

#[test]
fn a_metadata_partition_counts_as_zero() {
    let _s = serial();
    let h = decaying(9 * B, 8);
    let mut spectra = ir_spectra(&h, B);
    // Partition 3 becomes a metadata partition.
    spectra[3 * 2 * B] = f32::from_bits(METADATA_MARKER);
    let mut h_zeroed = h.clone();
    h_zeroed[3 * B..4 * B].fill(0.0);
    let mut rig = conv_rig(0, B as u32, 1, &spectra);
    let x = noise(6 * B, 9);
    let want = reference(&h_zeroed, &x, B, 6, &|_, _| 1.0);
    for n in 0..6 {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, B), &want[n * B..(n + 1) * B], 2e-3, &format!("block {n}"));
    }
}

#[test]
fn stereo_convolution_with_one_or_two_impulse_responses() {
    let _s = serial();
    let h0 = decaying(9 * B, 10);
    let h1 = decaying(9 * B, 11);
    let (s0, s1) = (ir_spectra(&h0, B), ir_spectra(&h1, B));
    let left = noise(5 * B, 12);
    let right = noise(5 * B, 13);
    // Deinterleaved input, a stereo impulse response (one after the other), output channels one after the other.
    let both: Vec<f32> = s0.iter().chain(&s1).copied().collect();
    let mut rig = conv_rig(FLAG_DEINTERLEAVED | FLAG_STEREO_IR, B as u32, 2, &both);
    let (want_l, want_r) = (reference(&h0, &left, B, 5, &|_, _| 1.0), reference(&h1, &right, B, 5, &|_, _| 1.0));
    for n in 0..5 {
        rig.write(IN, &left[n * B..(n + 1) * B]);
        rig.write(IN + B, &right[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, B), &want_l[n * B..(n + 1) * B], 2e-3, &format!("left {n}"));
        close(&rig.read(OUT + B, B), &want_r[n * B..(n + 1) * B], 2e-3, &format!("right {n}"));
    }
    // Interleaved input and one impulse response for both channels.
    let mut rig = conv_rig(0, B as u32, 2, &s0);
    let (want_l, want_r) = (reference(&h0, &left, B, 5, &|_, _| 1.0), reference(&h0, &right, B, 5, &|_, _| 1.0));
    for n in 0..5 {
        let inter: Vec<f32> = (0..B).flat_map(|i| [left[n * B + i], right[n * B + i]]).collect();
        rig.write(IN, &inter);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, B), &want_l[n * B..(n + 1) * B], 2e-3, &format!("left {n}"));
        close(&rig.read(OUT + B, B), &want_r[n * B..(n + 1) * B], 2e-3, &format!("right {n}"));
    }
}

#[test]
fn the_larger_block_size_works_too() {
    let _s = serial();
    let block = 0x400;
    let h = decaying(9 * block, 14);
    let mut rig = conv_rig(0, block as u32, 1, &ir_spectra(&h, block));
    let x = noise(4 * block, 15);
    let want = reference(&h, &x, block, 4, &|_, _| 1.0);
    for n in 0..4 {
        rig.write(IN, &x[n * block..(n + 1) * block]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit(&c).0, S_OK);
        close(&rig.read(OUT, block), &want[n * block..(n + 1) * block], 3e-3, &format!("block {n}"));
    }
}

#[test]
fn the_envelope_is_taken_at_the_first_command_of_each_group_of_8() {
    let _s = serial();
    let h = decaying(9 * B, 16);
    let mut rig = conv_rig(0, B as u32, 1, &ir_spectra(&h, B));
    let blocks = 18;
    let x = noise(blocks * B, 17);
    // One gain per partition for the first group, none for the second, other gains for the third.
    let g1: Vec<f32> = (0..9).map(|p| 1.0 - 0.1 * p as f32).collect();
    let g3: Vec<f32> = (0..9).map(|p| if p % 2 == 0 { 0.5 } else { 2.0 }).collect();
    rig.write(AUX, &g1);
    rig.write(AUX + 0x100, &g3);
    let gains = |n: usize, p: usize| match n / 8 {
        0 => f64::from(g1[p]),
        1 => 1.0,
        _ => f64::from(g3[p]),
    };
    let want = reference(&h, &x, B, blocks, &gains);
    for n in 0..blocks {
        rig.write(IN, &x[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        let (r, _) = match n {
            0 => rig.submit_envelope(&c, rig.addr(AUX)),
            // The envelope of a command inside a group is ignored.
            1 | 2 => rig.submit_envelope(&c, rig.addr(AUX + 0x100)),
            // A group start without envelope disables it.
            8 => rig.submit(&c),
            16 => rig.submit_envelope(&c, rig.addr(AUX + 0x100)),
            _ => rig.submit(&c),
        };
        assert_eq!(r, S_OK, "block {n}");
        close(&rig.read(OUT, B), &want[n * B..(n + 1) * B], 3e-3, &format!("block {n}"));
    }
}

#[test]
fn the_right_impulse_response_gains_start_at_the_partition_count_rounded_up_to_4() {
    let _s = serial();
    let h0 = decaying(9 * B, 18);
    let h1 = decaying(9 * B, 19);
    let both: Vec<f32> = ir_spectra(&h0, B).into_iter().chain(ir_spectra(&h1, B)).collect();
    let mut rig = conv_rig(FLAG_DEINTERLEAVED | FLAG_STEREO_IR, B as u32, 2, &both);
    // 9 partitions round up to 12: the left gains are 0 to 8, the right gains 12 to 20.
    let mut env = vec![0f32; 21];
    for p in 0..9 {
        env[p] = 1.0;
        env[12 + p] = if p == 0 { 0.5 } else { 0.0 };
    }
    rig.write(AUX, &env);
    let (left, right) = (noise(3 * B, 20), noise(3 * B, 21));
    let want_l = reference(&h0, &left, B, 3, &|_, _| 1.0);
    let want_r = reference(&h1, &right, B, 3, &|_, p| if p == 0 { 0.5 } else { 0.0 });
    for n in 0..3 {
        rig.write(IN, &left[n * B..(n + 1) * B]);
        rig.write(IN + B, &right[n * B..(n + 1) * B]);
        let c = rig.command(1.0, 1.0, IN, OUT, None);
        assert_eq!(rig.submit_envelope(&c, rig.addr(AUX)).0, S_OK);
        close(&rig.read(OUT, B), &want_l[n * B..(n + 1) * B], 3e-3, &format!("left {n}"));
        close(&rig.read(OUT + B, B), &want_r[n * B..(n + 1) * B], 3e-3, &format!("right {n}"));
    }
}

#[test]
fn convolution_buffers_and_the_envelope_must_lie_inside_the_client_memory() {
    let _s = serial();
    let h = decaying(9 * B, 22);
    let rig = conv_rig(0, B as u32, 1, &ir_spectra(&h, B));
    let good = rig.command(1.0, 1.0, IN, OUT, None);
    let c = XdspCommand { input: rig.addr(MEM_FLOATS - B + 1), ..good };
    assert_eq!(rig.submit(&c).0, E_INVALIDARG);
    let c = XdspCommand { output: rig.addr(MEM_FLOATS - B + 1), ..good };
    assert_eq!(rig.submit(&c).0, E_INVALIDARG);
    // 9 gains starting 8 floats before the end do not fit.
    assert_eq!(rig.submit_envelope(&good, rig.addr(MEM_FLOATS - 8)).0, E_INVALIDARG);
    assert_eq!(rig.submit_envelope(&good, rig.addr(MEM_FLOATS - 9)).0, S_OK);
    assert_eq!(rig.status()[0], 1);
}
