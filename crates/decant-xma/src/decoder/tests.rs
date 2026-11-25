use super::*;
use crate::stream::{first_frame, read_frame};
use crate::synth::{pack, tone_frame, tone_frame_with, tone_stream};

/// Decodes every frame of a buffer and then the tail. Returns the samples, the frame count and the error count.
fn decode_all(data: &[u8], channels: usize, rate: u32) -> (Vec<i16>, usize, usize) {
    let mut decoder = Decoder::new(channels, rate);
    let mut pcm = Vec::new();
    let (mut frames, mut errors) = (0, 0);
    let mut at = first_frame(data);
    while let Some(pos) = at {
        let frame = read_frame(data, pos).expect("a frame where the walk says");
        frames += 1;
        errors += usize::from(decoder.decode(&frame, &mut pcm).is_err());
        at = frame.next;
    }
    decoder.flush(&mut pcm);
    (pcm, frames, errors)
}

/// Signal power at `freq` Hz, by the Goertzel recurrence.
fn power(signal: &[f64], freq: f64, rate: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / rate;
    let k = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in signal {
        let s = x + k * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    s1 * s1 + s2 * s2 - k * s1 * s2
}

fn channel(pcm: &[i16], channels: usize, c: usize) -> Vec<f64> {
    pcm.iter()
        .skip(c)
        .step_by(channels)
        .map(|&v| f64::from(v))
        .collect()
}

/// A frame repeats every 512 samples, so the spectrum has lines at multiples of rate / 512 only. This is the strongest line
/// below 4 kHz, in Hz.
fn strongest_line(signal: &[f64], rate: f64) -> f64 {
    let step = rate / 512.0;
    (1..(4000.0 / step) as usize)
        .map(|i| i as f64 * step)
        .max_by(|a, b| power(signal, *a, rate).total_cmp(&power(signal, *b, rate)))
        .unwrap()
}

#[test]
fn silent_frames_decode_to_silence() {
    let data = pack(&vec![Some(tone_frame(&[None], 48_000, 0)); 6]);
    let (pcm, frames, errors) = decode_all(&data, 1, 48_000);
    assert_eq!((frames, errors), (6, 0));
    assert_eq!(pcm.len(), 6 * 512 + 256);
    assert!(pcm.iter().all(|&v| v == 0));
}

#[test]
fn a_mono_tone_has_its_frequency() {
    // Bin 21 at 48 kHz is 21.5 * 48000 / 1024 = 1007.8 Hz.
    let data = tone_stream(&[Some((21, 1))], 48_000, 26, 40);
    let (pcm, frames, errors) = decode_all(&data, 1, 48_000);
    assert_eq!((frames, errors), (40, 0));
    let x = channel(&pcm, 1, 0);
    let steady = &x[2048..2048 + 8192];
    let peak = steady.iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(peak > 1000.0 && peak < 32767.0, "peak {peak}");
    let line = strongest_line(steady, 48_000.0);
    assert!(
        (line - 1007.8).abs() < 48_000.0 / 512.0,
        "strongest at {line} Hz"
    );
}

#[test]
fn stereo_channels_stay_apart() {
    // Left bin 10 (515.6 Hz) and right bin 40 (1984.4 Hz) at 32 kHz.
    let data = tone_stream(&[Some((10, 1)), Some((40, 1))], 32_000, 26, 30);
    let (pcm, frames, errors) = decode_all(&data, 2, 32_000);
    assert_eq!((frames, errors), (30, 0));
    assert_eq!(pcm.len(), (30 * 512 + 256) * 2);
    let (l, r) = (channel(&pcm, 2, 0), channel(&pcm, 2, 1));
    assert!((strongest_line(&l[2048..10240], 32_000.0) - 10.5 * 32_000.0 / 1024.0).abs() < 62.5);
    assert!((strongest_line(&r[2048..10240], 32_000.0) - 40.5 * 32_000.0 / 1024.0).abs() < 62.5);
}

#[test]
fn mid_and_side_put_a_mid_tone_in_both_channels() {
    let frame = tone_frame_with(&[Some((30, 1)), None], 48_000, 26, true);
    let data = pack(&vec![Some(frame); 10]);
    let (pcm, frames, errors) = decode_all(&data, 2, 48_000);
    assert_eq!((frames, errors), (10, 0));
    let (l, r) = (channel(&pcm, 2, 0), channel(&pcm, 2, 1));
    assert!(l[1024..4096].iter().any(|&v| v.abs() > 1000.0));
    assert_eq!(l, r);
}

#[test]
fn a_damaged_frame_gives_a_frame_of_silence() {
    let mut frame = tone_frame(&[Some((21, 1))], 48_000, 20);
    // The reserved bit of the subframe sits at bit 15 + 2 + 8 + 1 + 1.
    frame[27] = true;
    let data = pack(&[Some(frame)]);
    let (pcm, frames, errors) = decode_all(&data, 1, 48_000);
    assert_eq!((frames, errors), (1, 1));
    assert_eq!(pcm.len(), 512 + 256);
    assert!(pcm.iter().all(|&v| v == 0));
}

#[test]
fn the_error_names_what_went_wrong() {
    let mut frame = tone_frame(&[Some((21, 1))], 48_000, 20);
    frame[27] = true;
    let data = pack(&[Some(frame)]);
    let frame = read_frame(&data, first_frame(&data).unwrap()).unwrap();
    let err = Decoder::new(1, 48_000)
        .decode(&frame, &mut Vec::new())
        .unwrap_err();
    assert_eq!(err, Error("reserved bit"));
    assert_eq!(err.to_string(), "XMA frame: reserved bit");
}

#[test]
fn a_frame_cut_short_is_an_error() {
    let frame = tone_frame(&[Some((21, 1))], 48_000, 20);
    // Keep the length field intact but drop the last 20 bits of content, so the contents read past the end.
    let mut cut = frame[..frame.len() - 20].to_vec();
    let len = cut.len();
    for (i, b) in cut.iter_mut().take(15).enumerate() {
        *b = len >> (14 - i) & 1 == 1;
    }
    let data = pack(&[Some(cut)]);
    let (pcm, _, errors) = decode_all(&data, 1, 48_000);
    assert_eq!(errors, 1);
    assert_eq!(pcm.len(), 512 + 256);
}

#[test]
fn reset_forgets_the_overlap() {
    let data = tone_stream(&[Some((21, 1))], 48_000, 26, 4);
    let mut at = first_frame(&data);
    let mut decoder = Decoder::new(1, 48_000);
    let mut first_run = Vec::new();
    let mut second_run = Vec::new();
    for run in [&mut first_run, &mut second_run] {
        decoder.reset();
        let mut pos = at;
        while let Some(p) = pos {
            let frame = read_frame(&data, p).unwrap();
            decoder.decode(&frame, run).unwrap();
            pos = frame.next;
        }
        decoder.flush(run);
        at = first_frame(&data);
    }
    assert_eq!(first_run, second_run);
}

#[test]
fn rate_and_channel_arguments_are_clamped() {
    assert_eq!(Decoder::new(0, 48_000).channels(), 1);
    assert_eq!(Decoder::new(6, 48_000).channels(), 2);
    // Rates round up to the next of the four known ones.
    assert_eq!(
        Decoder::new(1, 40_000).bands(0),
        Decoder::new(1, 44_100).bands(0)
    );
    assert_eq!(
        Decoder::new(1, 1).bands(0),
        Decoder::new(1, 24_000).bands(0)
    );
    assert_eq!(
        Decoder::new(1, 99_999).bands(0),
        Decoder::new(1, 48_000).bands(0)
    );
}

#[test]
fn band_counts_follow_the_block_size() {
    let d = Decoder::new(1, 48_000);
    assert!(d.bands(0) > d.bands(1) && d.bands(1) > d.bands(2));
    // A size past the last counts as the last.
    assert_eq!(d.bands(7), d.bands(2));
    assert!(d.bands(0) <= MAX_BANDS);
}

#[test]
fn louder_gain_gives_louder_output() {
    let peak = |gain| {
        let data = tone_stream(&[Some((21, 1))], 48_000, gain, 12);
        let (pcm, ..) = decode_all(&data, 1, 48_000);
        pcm.iter().map(|v| i32::from(*v).abs()).max().unwrap()
    };
    let (quiet, loud) = (peak(10), peak(20));
    assert!(loud > quiet * 2, "{quiet} vs {loud}");
}

#[test]
fn larger_levels_give_larger_amplitude() {
    let peak = |level| {
        let data = tone_stream(&[Some((21, level))], 48_000, 10, 12);
        let (pcm, ..) = decode_all(&data, 1, 48_000);
        pcm.iter().map(|v| i32::from(*v).abs()).max().unwrap()
    };
    assert!(peak(3) > peak(1));
}
