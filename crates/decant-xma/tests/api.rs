//! The public surface as the console audio layer uses it: constants, frame walking by bit position, decoding, errors.

use decant_xma::synth::{pack, tone_frame, tone_stream};
use decant_xma::{
    Decoder, Error, FRAME_SAMPLES, HEADER_BITS, PACKET_BITS, PACKET_BYTES, RATES, SUBFRAME_SAMPLES,
    first_frame, packet_header, read_frame,
};

fn decode_stream(data: &[u8], channels: usize, rate: u32) -> (Vec<i16>, Vec<Error>) {
    let mut decoder = Decoder::new(channels, rate);
    let (mut pcm, mut errors) = (Vec::new(), Vec::new());
    let mut at = first_frame(data);
    while let Some(pos) = at {
        let frame = read_frame(data, pos).unwrap();
        if let Err(e) = decoder.decode(&frame, &mut pcm) {
            errors.push(e);
        }
        at = frame.next;
    }
    decoder.flush(&mut pcm);
    (pcm, errors)
}

#[test]
fn constants_agree_with_each_other() {
    assert_eq!(PACKET_BITS, PACKET_BYTES * 8);
    assert_eq!(HEADER_BITS, 32);
    assert_eq!(FRAME_SAMPLES, 4 * SUBFRAME_SAMPLES);
    assert_eq!(RATES, [24_000, 32_000, 44_100, 48_000]);
}

#[test]
fn a_synthetic_stream_decodes_end_to_end() {
    let data = tone_stream(&[Some((21, 1))], 48_000, 26, 94);
    assert_eq!(data.len() % PACKET_BYTES, 0);
    let (pcm, errors) = decode_stream(&data, 1, 48_000);
    assert!(errors.is_empty());
    assert_eq!(pcm.len(), 94 * FRAME_SAMPLES + FRAME_SAMPLES / 2);
    assert!(pcm.iter().any(|&v| v.abs() > 1000));
}

#[test]
fn frames_chain_by_their_next_position() {
    let data = tone_stream(&[Some((10, 1)), Some((40, 1))], 44_100, 20, 50);
    let mut count = 0;
    let mut at = first_frame(&data);
    let mut last_start = 0;
    while let Some(pos) = at {
        let frame = read_frame(&data, pos).unwrap();
        assert!(frame.start >= last_start);
        assert_eq!(frame.start, pos);
        assert_eq!(frame.data.len(), frame.bits.div_ceil(8));
        last_start = frame.start;
        count += 1;
        at = frame.next;
    }
    assert_eq!(count, 50);
    let frames_in_headers: u32 = (0..data.len() / PACKET_BYTES)
        .map(|p| packet_header(&data, p).unwrap().frames)
        .sum();
    assert_eq!(frames_in_headers, 50);
}

#[test]
fn a_bad_frame_does_not_stop_the_stream() {
    let good = tone_frame(&[Some((21, 1))], 48_000, 20);
    let mut bad = good.clone();
    bad[27] = true;
    let data = pack(&[Some(good.clone()), Some(bad), Some(good)]);
    let (pcm, errors) = decode_stream(&data, 1, 48_000);
    assert_eq!(errors.len(), 1);
    assert_eq!(pcm.len(), 3 * FRAME_SAMPLES + FRAME_SAMPLES / 2);
}

#[test]
fn errors_print_and_compare() {
    let e = Error("x");
    assert_eq!(e.to_string(), "XMA frame: x");
    assert_eq!(e, Error("x"));
    let boxed: Box<dyn std::error::Error> = Box::new(e);
    assert!(boxed.to_string().contains('x'));
}

/// Decodes a real stream. `DECANT_XMA_SAMPLE` names a file of XMA2 packets of one stream, `DECANT_XMA_CHANNELS` (default 2)
/// and `DECANT_XMA_RATE` (default 48000) describe it. When `DECANT_XMA_PCM` names a file of expected little endian 16 bit
/// samples, the output must match it exactly.
#[test]
#[ignore = "needs local sample data, set DECANT_XMA_SAMPLE"]
fn a_real_stream_matches_its_reference() {
    let Ok(path) = std::env::var("DECANT_XMA_SAMPLE") else {
        panic!("DECANT_XMA_SAMPLE is not set")
    };
    let channels = std::env::var("DECANT_XMA_CHANNELS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let rate = std::env::var("DECANT_XMA_RATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48_000);
    let data = std::fs::read(path).unwrap();
    let (pcm, errors) = decode_stream(&data, channels, rate);
    assert!(errors.is_empty(), "{} frames failed", errors.len());
    assert!(!pcm.is_empty());
    if let Ok(reference) = std::env::var("DECANT_XMA_PCM") {
        let expected: Vec<i16> = std::fs::read(reference)
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect();
        let n = expected.len().min(pcm.len());
        assert_eq!(&pcm[..n], &expected[..n]);
    }
}
