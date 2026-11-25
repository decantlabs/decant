//! Synthetic XMA2 input and the reference decode shared by the test programs.
#![allow(dead_code)]

use decant_xma::{Decoder, PACKET_BYTES, first_frame, read_frame, synth};

pub const RATE: u32 = 48_000;

/// Decodes one stream of packets straight with `decant-xma`, flush included. Interleaved, `channels` per frame.
pub fn direct(data: &[u8], channels: usize) -> Vec<i16> {
    let mut decoder = Decoder::new(channels, RATE);
    let mut out = Vec::new();
    let mut at = first_frame(data);
    while let Some(start) = at {
        let Some(frame) = read_frame(data, start) else {
            break;
        };
        let _ = decoder.decode(&frame, &mut out);
        at = frame.next;
    }
    decoder.flush(&mut out);
    out
}

/// A stream of `frames` frames, one tone per channel, as packets.
pub fn tones(channels: usize, shift: usize, frames: usize) -> Vec<u8> {
    let tones: Vec<_> = (0..channels)
        .map(|c| Some((10 + shift + c * 17, 1u16)))
        .collect();
    synth::tone_stream(&tones, RATE, 26, frames)
}

/// A stream of `channels` channels with exactly `packets` packets: the longest tone stream that does not need more.
pub fn tones_packets(channels: usize, shift: usize, packets: usize) -> Vec<u8> {
    let mut best = None;
    for frames in 1.. {
        let data = tones(channels, shift, frames);
        if data.len() / PACKET_BYTES > packets {
            break;
        }
        best = Some(data);
    }
    let data = best.expect("a stream");
    assert_eq!(data.len() / PACKET_BYTES, packets);
    data
}

/// The packets of several streams, one after the other in turn while each has some. The header of each packet tells how many
/// packets of the other streams follow before the next packet of its own stream.
pub fn interleave(streams: &[Vec<u8>]) -> Vec<u8> {
    let counts: Vec<usize> = streams.iter().map(|s| s.len() / PACKET_BYTES).collect();
    let mut order = Vec::new();
    for round in 0..*counts.iter().max().unwrap() {
        order.extend((0..streams.len()).filter(|&s| round < counts[s]).map(|s| (s, round)));
    }
    let mut out = Vec::new();
    for (at, &(s, round)) in order.iter().enumerate() {
        let mut packet = streams[s][round * PACKET_BYTES..(round + 1) * PACKET_BYTES].to_vec();
        let next = order[at + 1..].iter().position(|&(other, _)| other == s);
        packet[3] = next.unwrap_or(0) as u8;
        out.extend(packet);
    }
    out
}

/// The expected interleaved output of streams of `channels[i]` channels: each decoded alone, then merged frame by frame.
pub fn expected(streams: &[Vec<u8>], channels: &[usize]) -> Vec<i16> {
    let decoded: Vec<Vec<i16>> = streams
        .iter()
        .zip(channels)
        .map(|(s, &c)| direct(s, c))
        .collect();
    let frames = decoded
        .iter()
        .zip(channels)
        .map(|(d, &c)| d.len() / c)
        .max()
        .unwrap();
    let mut out = Vec::new();
    for f in 0..frames {
        for (d, &c) in decoded.iter().zip(channels) {
            // A stream that ended earlier is padded with silence.
            out.extend((f * c..(f + 1) * c).map(|i| d.get(i).copied().unwrap_or(0)));
        }
    }
    out
}

/// An `XMA2WAVEFORMATEX`: the `WAVEFORMATEX` and its 34 extra bytes.
pub fn format(channels: u16, streams: u16, block_align: u16, packets: usize) -> Vec<u8> {
    let samples = (packets * 512) as u32;
    let mut f = Vec::new();
    f.extend(0x166u16.to_le_bytes());
    f.extend(channels.to_le_bytes());
    f.extend(RATE.to_le_bytes());
    f.extend((RATE * 2 * u32::from(channels)).to_le_bytes());
    f.extend(block_align.to_le_bytes());
    f.extend(16u16.to_le_bytes());
    f.extend(34u16.to_le_bytes());
    f.extend(streams.to_le_bytes());
    for v in [3u32, samples, 2048, 0, samples, 0, 0] {
        f.extend(v.to_le_bytes());
    }
    f.extend([0u8, 2]);
    f.extend((packets as u16).to_le_bytes());
    f
}

/// A RIFF WAVE file with a `fmt ` and a `data` chunk.
pub fn riff(format: &[u8], data: &[u8]) -> Vec<u8> {
    let mut body = b"WAVEfmt ".to_vec();
    body.extend((format.len() as u32).to_le_bytes());
    body.extend(format);
    body.extend(b"data");
    body.extend((data.len() as u32).to_le_bytes());
    body.extend(data);
    let mut file = b"RIFF".to_vec();
    file.extend((body.len() as u32).to_le_bytes());
    file.extend(body);
    file
}
