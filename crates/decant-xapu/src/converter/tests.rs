use super::*;

fn signal(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

fn run(c: &mut Converter, input: &[f32], target: f32, ramp: f32) -> Vec<f32> {
    c.push(&[input], target, ramp);
    let n = c.pending_frames();
    let mut out = vec![Vec::new()];
    c.take(n, &mut out);
    out.remove(0)
}

#[test]
fn clamps() {
    assert_eq!(clamp_pitch(0.0), 0.125);
    assert_eq!(clamp_pitch(100.0), 8.0);
    assert_eq!(clamp_pitch(1.5), 1.5);
    assert_eq!(clamp_ramp(0.0), RAMP_MIN);
    assert_eq!(clamp_ramp(-0.5), 0.5);
    assert_eq!(clamp_ramp(-100.0), 8.0);
    assert_eq!(RAMP_MIN, 2f32.powi(-18));
}

#[test]
fn at_pitch_1_the_output_is_the_input_delayed_by_5_frames() {
    let input = signal(200, |i| ((i * 37 % 101) as f32 - 50.0) / 50.0);
    let mut c = Converter::new(1, 1.0);
    let out = run(&mut c, &input, 1.0, 1.0);
    assert_eq!(out.len(), 200);
    for (n, v) in out.iter().enumerate() {
        let want = if n >= DELAY { input[n - DELAY] } else { 0.0 };
        assert!((v - want).abs() < 1e-6, "frame {n}: {v} against {want}");
    }
    assert_eq!(c.pitch(), 1.0);
}

#[test]
fn a_flush_brings_out_the_last_frames() {
    let input = signal(64, |i| i as f32 + 1.0);
    let mut c = Converter::new(1, 1.0);
    c.push(&[&input], 1.0, 1.0);
    assert_eq!(c.pending_frames(), 64);
    c.flush(1.0, 1.0);
    assert_eq!(c.pending_frames(), 64 + DELAY);
    let mut out = vec![Vec::new()];
    c.take(69, &mut out);
    assert!((out[0][68] - 64.0).abs() < 1e-5, "the last input frame comes out at frame 68");
    assert!(out[0][..5].iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn input_is_split_into_commands_without_a_seam() {
    let input = signal(300, |i| (i as f32 * 0.07).sin());
    let whole = run(&mut Converter::new(1, 1.7), &input, 1.7, 1.0);
    let mut c = Converter::new(1, 1.7);
    let mut parts = Vec::new();
    for chunk in input.chunks(37) {
        parts.extend(run(&mut c, chunk, 1.7, 1.0));
    }
    assert_eq!(whole.len(), parts.len());
    for (a, b) in whole.iter().zip(&parts) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn a_low_frequency_survives_pitch_changes() {
    let f = 0.01f64;
    let sine = |t: f64| (2.0 * std::f64::consts::PI * f * t).sin() as f32;
    let input = signal(2000, |i| sine(i as f64));
    for pitch in [0.5f32, 0.75, 1.0, 1.5, 2.0, 3.0] {
        let mut c = Converter::new(1, pitch);
        let out = run(&mut c, &input, pitch, 1.0);
        let expected_len = (2000.0 / f64::from(pitch)).ceil() as usize;
        assert!(out.len().abs_diff(expected_len) <= 1, "pitch {pitch}: {} frames", out.len());
        for (n, v) in out.iter().enumerate().skip(20) {
            let want = sine(n as f64 * f64::from(pitch) - DELAY as f64);
            assert!((v - want).abs() < 0.03, "pitch {pitch}, frame {n}: {v} against {want}");
        }
    }
}

#[test]
fn a_constant_keeps_its_level() {
    for pitch in [0.3f32, 0.7, 1.3, 3.0, 8.0] {
        let mut c = Converter::new(1, pitch);
        let out = run(&mut c, &vec![0.5; 400], pitch, 1.0);
        // The first frames read the silence before the signal.
        for v in &out[(10.0 / pitch).ceil() as usize + 2..] {
            assert!((v - 0.5).abs() < 1e-5, "pitch {pitch}: {v}");
        }
    }
}

#[test]
fn the_pitch_ramps_toward_the_target() {
    let mut c = Converter::new(1, 1.0);
    // Read positions 0, 1, 2.25, 3.75, 5.5 at pitches 1, 1.25, 1.5, 1.75, 2, then steps of 2.
    let out = run(&mut c, &[0.0; 20], 2.0, 0.25);
    assert_eq!(out.len(), 12);
    assert_eq!(c.pitch(), 2.0);
    // Down again.
    run(&mut c, &[0.0; 20], 1.0, 0.5);
    assert_eq!(c.pitch(), 1.0);
}

#[test]
fn the_ramp_has_a_minimum_and_the_target_a_range() {
    let mut c = Converter::new(1, 1.0);
    run(&mut c, &[0.0; 100], 2.0, 0.0);
    let p = c.pitch();
    assert!(p > 1.0 && p < 1.001, "{p}");
    let mut c = Converter::new(1, 1.0);
    run(&mut c, &[0.0; 1000], 100.0, 8.0);
    assert_eq!(c.pitch(), 8.0);
    let mut c = Converter::new(1, 1.0);
    run(&mut c, &[0.0; 100], 0.0, 8.0);
    assert_eq!(c.pitch(), 0.125);
}

#[test]
fn channels_are_converted_independently() {
    let left = signal(100, |i| i as f32);
    let right = signal(100, |i| -(i as f32));
    let mut c = Converter::new(2, 1.0);
    c.push(&[&left, &right], 1.0, 1.0);
    let mut out = vec![Vec::new(), Vec::new()];
    c.take(100, &mut out);
    for n in 5..100 {
        assert!((out[0][n] - left[n - 5]).abs() < 1e-5);
        assert!((out[1][n] + left[n - 5]).abs() < 1e-5);
    }
}

#[test]
fn unconverted_frames_and_reset() {
    let mut c = Converter::new(1, 0.5);
    c.push(&[&[1.0; 40]], 0.5, 1.0);
    assert!(c.pending_frames() > 70);
    assert_eq!(c.unconverted_frames(), 40 - (c.pending_frames() as f64 * 0.5) as usize);
    c.reset(2.0);
    assert_eq!(c.pending_frames(), 0);
    assert_eq!(c.unconverted_frames(), 0);
    assert_eq!(c.pitch(), 2.0);
}

#[test]
fn cursor_compaction_over_long_stream() {
    let total = 10_000usize;
    let input = signal(total, |i| ((i % 251) as f32) * 0.001);
    let mut c = Converter::new(1, 1.0);
    let mut out = vec![Vec::new()];
    for chunk in input.chunks(512) {
        c.push(&[chunk], 1.0, 1.0);
        let take_n = c.pending_frames().min(256);
        c.take(take_n, &mut out);
    }
    c.flush(1.0, 1.0);
    let rem = c.pending_frames();
    c.take(rem, &mut out);
    assert_eq!(out[0].len(), total + DELAY);
    for i in DELAY..total {
        assert!((out[0][i] - input[i - DELAY]).abs() < 1e-6, "mismatch at {i}");
    }
}
