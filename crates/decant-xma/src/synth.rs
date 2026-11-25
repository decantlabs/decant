//! Small valid XMA streams built from nothing, for tests. A frame holds one tone per channel, or silence. A tone is a single
//! MDCT coefficient repeated in every frame, which the decoder turns into a steady sine at (bin + 1/2) * rate / 1024 Hz.

use crate::tables::{COEF_A, COEF_A_STEPS, QUAD, SCALE_DELTA};
use crate::{HEADER_BITS, PACKET_BITS, PACKET_BYTES};

/// Writes bits, most significant bit first.
pub struct Writer {
    pub data: Vec<u8>,
    pub len: usize,
}

impl Default for Writer {
    fn default() -> Writer {
        Writer::new()
    }
}

impl Writer {
    pub fn new() -> Writer {
        Writer {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Writes the low `n` bits of `value`.
    pub fn put(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.put_bit(value >> i & 1 == 1);
        }
    }

    pub fn put_bit(&mut self, bit: bool) {
        if self.len.is_multiple_of(8) {
            self.data.push(0);
        }
        if bit {
            let last = self.data.len() - 1;
            self.data[last] |= 0x80 >> (self.len % 8);
        }
        self.len += 1;
    }

    /// Writes the code of `symbol` from a table of (symbol, code, length) triples sorted by symbol.
    pub(crate) fn put_symbol(&mut self, table: &[(u16, u32, u8)], symbol: u16) {
        let at = table
            .binary_search_by_key(&symbol, |entry| entry.0)
            .unwrap_or_else(|_| panic!("symbol {symbol} is not in the code"));
        let (_, code, len) = table[at];
        self.put(u32::from(len), code);
    }

    /// The bits written so far, one bool each.
    pub fn bits(&self) -> Vec<bool> {
        (0..self.len)
            .map(|i| self.data[i / 8] >> (7 - i % 8) & 1 == 1)
            .collect()
    }
}

/// Packs frames (their bits, the length field first) into the packets of one stream. A `None` pads the rest of the current
/// packet, so the next frame opens the next packet. The last bit of every frame is set to whether the next frame begins in the
/// same packet. Padding is all ones.
pub fn pack(frames: &[Option<Vec<bool>>]) -> Vec<u8> {
    let payload = PACKET_BITS - HEADER_BITS;
    // Positions here count payload bits only, packet headers left out.
    let mut starts = Vec::new();
    let mut bodies = Vec::new();
    let mut at = 0usize;
    for frame in frames {
        match frame {
            None => at = at.div_ceil(payload) * payload,
            Some(bits) => {
                starts.push(at);
                bodies.push(bits);
                at += bits.len();
            }
        }
    }
    let packets = at.div_ceil(payload).max(1);
    let mut data = vec![0xFFu8; packets * PACKET_BYTES];
    for p in 0..packets {
        let (from, to) = (p * payload, (p + 1) * payload);
        let begin: Vec<usize> = starts
            .iter()
            .copied()
            .filter(|&s| s >= from && s < to)
            .collect();
        let first = begin.first().map_or(payload, |s| s - from);
        let mut header = Writer::new();
        header.put(6, begin.len() as u32);
        header.put(15, first as u32);
        header.put(11, 0);
        data[p * PACKET_BYTES..p * PACKET_BYTES + 4].copy_from_slice(&header.data);
    }
    for (i, (&start, bits)) in starts.iter().zip(&bodies).enumerate() {
        let len = bits.len();
        for (b, &bit) in bits.iter().enumerate() {
            let bit = if b == len - 1 {
                starts
                    .get(i + 1)
                    .is_some_and(|&next| next / payload == (start + len - 1) / payload)
            } else {
                bit
            };
            let payload_bit = start + b;
            let pos = payload_bit / payload * PACKET_BITS + HEADER_BITS + payload_bit % payload;
            let mask = 0x80 >> (pos % 8);
            if bit {
                data[pos / 8] |= mask;
            } else {
                data[pos / 8] &= !mask;
            }
        }
    }
    data
}

/// One frame whose channels each hold one coefficient: `tones[c]` is `Some((bin, level))`, or `None` for silence. `gain` moves
/// the quantisation step in dB (-31 to 30). The bins must be at least 4.
pub fn tone_frame(tones: &[Option<(usize, u16)>], rate: u32, gain: i32) -> Vec<bool> {
    tone_frame_with(tones, rate, gain, false)
}

/// As [`tone_frame`]. With `mid_side` a stereo frame carries mid and side, so a tone in channel 0 plays in both channels.
pub fn tone_frame_with(
    tones: &[Option<(usize, u16)>],
    rate: u32,
    gain: i32,
    mid_side: bool,
) -> Vec<bool> {
    let channels = tones.len().clamp(1, 2);
    let mut w = Writer::new();
    // The length field, patched at the end.
    w.put(15, 0);
    // Tiling: the same for all channels, one subframe of 512.
    w.put_bit(true);
    w.put_bit(false);
    if channels == 2 {
        // No post processing transform.
        w.put_bit(false);
    }
    // Dynamic range gain, then no trim.
    w.put(8, 0);
    w.put_bit(false);
    // Subframe: no extended header, then the reserved bit.
    w.put_bit(false);
    w.put_bit(false);
    if channels == 2 {
        // The channel transform bit, then a group of both channels with the mid and side transform on every band, or none.
        w.put_bit(false);
        w.put_bit(!mid_side);
        w.put_bit(mid_side);
    }
    let present: Vec<bool> = tones.iter().take(channels).map(Option::is_some).collect();
    for &p in &present {
        w.put_bit(p);
    }
    if present.iter().any(|&p| p) {
        // Every coefficient vector coded, then the quantisation step.
        w.put_bit(false);
        w.put(6, gain.clamp(-31, 30) as u32 & 0x3F);
        if channels == 2 {
            // No step modifiers.
            w.put(3, 0);
            for _ in 0..channels {
                w.put_bit(false);
            }
        }
        let bands = band_count(rate);
        for _ in &present {
            // Scale factors: step 1, every band at 45 (a difference of 0 is symbol 60).
            w.put(2, 0);
            for _ in 0..bands {
                w.put_symbol(&SCALE_DELTA, 60);
            }
        }
        for tone in tones.iter().take(channels) {
            let Some((bin, level)) = *tone else { continue };
            // Coefficient table 0, one vector of four zeros (symbol 1), which switches to run mode.
            w.put_bit(false);
            w.put_symbol(&QUAD, 1);
            let run = bin - 4;
            let symbol = (2..COEF_A_STEPS.len())
                .find(|&s| {
                    usize::from(COEF_A_STEPS[s].0) == run && COEF_A_STEPS[s].1 == level.max(1)
                })
                .expect("a code for this run and level");
            w.put_symbol(&COEF_A, symbol as u16);
            // A positive sign, then the end of the block.
            w.put_bit(true);
            w.put_symbol(&COEF_A, 1);
        }
    }
    // One spare bit and the trailer bit.
    w.put(2, 0);
    let mut bits = w.bits();
    let len = bits.len();
    for (i, b) in bits.iter_mut().take(15).enumerate() {
        *b = len >> (14 - i) & 1 == 1;
    }
    bits
}

/// The number of scale factor bands of a block of 512 samples at `rate`.
fn band_count(rate: u32) -> usize {
    crate::Decoder::new(1, rate).bands(0)
}

/// A stream of `frames` frames with one tone per channel as (bin, level), packed.
pub fn tone_stream(tones: &[Option<(usize, u16)>], rate: u32, gain: i32, frames: usize) -> Vec<u8> {
    let frame = tone_frame(tones, rate, gain);
    pack(&vec![Some(frame); frames])
}
