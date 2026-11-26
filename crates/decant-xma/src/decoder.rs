//! The frame decoder: frame tiling, channel transform, scale factors, coefficients, dequantisation, inverse MDCT and the
//! windowed overlap of blocks.
//!
//! The XMA setup of the bitstream is fixed: frames of 512 samples that carry their own length (15 bits), up to 32 subframes
//! per channel of 512, 256 or 128 samples, a dynamic range byte, and 16 bit output.

use std::sync::OnceLock;

use crate::FRAME_SAMPLES;
use crate::bitio::{BitReader, PrefixCode};
use crate::imdct::{HalfImdct, overlap_add, sine_window};
use crate::stream::Frame;
use crate::tables::*;

const MAX_BANDS: usize = 29;
const MAX_SUBFRAMES: usize = 32;
const SIZES: usize = 3;
const SMALLEST: usize = FRAME_SAMPLES >> (SIZES - 1);
/// Room for a frame and the half block that overlaps into the next frame.
const BUF_LEN: usize = FRAME_SAMPLES * 2;
/// The symbol of the single value code that continues in a wide value.
const SINGLE_ESCAPE: u16 = 100;
/// Gain applied to both channels when no mid and side transform is used for a band.
const PLAIN_GAIN: f32 = 181.0 / 128.0;

/// A frame that does not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XMA frame: {}", self.0)
    }
}

impl std::error::Error for Error {}

const GAIN_MIN: i32 = -256;
const GAIN_MAX: i32 = 256;
const GAIN_COUNT: usize = (GAIN_MAX - GAIN_MIN + 1) as usize;

/// Tables that every decoder shares.
struct Shared {
    scale_delta: PrefixCode,
    scale_run: PrefixCode,
    coef: [PrefixCode; 2],
    quad: PrefixCode,
    pair: PrefixCode,
    single: PrefixCode,
    imdct: [HalfImdct; SIZES],
    windows: [Vec<f32>; SIZES],
    gains: [f32; GAIN_COUNT],
}

fn shared() -> &'static Shared {
    static SHARED: OnceLock<Shared> = OnceLock::new();
    SHARED.get_or_init(|| {
        let gains = std::array::from_fn(|i| 10f64.powf((i as i32 + GAIN_MIN) as f64 / 20.0) as f32);
        Shared {
            scale_delta: PrefixCode::new(&SCALE_DELTA),
            scale_run: PrefixCode::new(&SCALE_RUN),
            coef: [PrefixCode::new(&COEF_A), PrefixCode::new(&COEF_B)],
            quad: PrefixCode::new(&QUAD),
            pair: PrefixCode::new(&PAIR),
            single: PrefixCode::new(&SINGLE),
            imdct: [
                HalfImdct::new(512),
                HalfImdct::new(256),
                HalfImdct::new(128),
            ],
            windows: [sine_window(512), sine_window(256), sine_window(128)],
            gains,
        }
    })
}

/// 0, 1 or 2 for a length of 512, 256 or 128.
fn size_index(len: usize) -> usize {
    (FRAME_SAMPLES / len).trailing_zeros() as usize
}

/// floor(log2(v)) for v of at least 1.
fn floor_log2(v: usize) -> u32 {
    usize::BITS - 1 - v.max(1).leading_zeros()
}

/// The decoding state of one channel.
#[derive(Clone)]
struct Channel {
    /// Length of the block before the current one.
    prev_len: usize,
    has_coefs: bool,
    sub_count: usize,
    sub_lens: [usize; MAX_SUBFRAMES],
    sub_index: usize,
    done: usize,
    quant_step: i32,
    /// Whether scale factors were sent in this frame already.
    sf_known: bool,
    sf_step: i32,
    sf_max: i32,
    /// Two sets of scale factors: the ones sent last and the ones of the current subframe.
    sf: [[i32; MAX_BANDS]; 2],
    sf_sent: usize,
    sf_used: usize,
    /// Block size index of the scale factors sent last.
    sf_size: usize,
    vector_len: usize,
    /// Where the current subframe sits in `buf`.
    at: usize,
    buf: Vec<f32>,
}

impl Channel {
    fn new() -> Self {
        Channel {
            prev_len: FRAME_SAMPLES,
            has_coefs: false,
            sub_count: 0,
            sub_lens: [0; MAX_SUBFRAMES],
            sub_index: 0,
            done: 0,
            quant_step: 0,
            sf_known: false,
            sf_step: 1,
            sf_max: 0,
            sf: [[0; MAX_BANDS]; 2],
            sf_sent: 0,
            sf_used: 0,
            sf_size: 0,
            vector_len: 0,
            at: 0,
            buf: vec![0.0; BUF_LEN],
        }
    }

    fn reset(&mut self) {
        self.prev_len = FRAME_SAMPLES;
        self.has_coefs = false;
        self.sub_count = 0;
        self.sub_index = 0;
        self.done = 0;
        self.quant_step = 0;
        self.sf_known = false;
        self.sf_step = 1;
        self.sf_max = 0;
        self.sf_sent = 0;
        self.sf_used = 0;
        self.sf_size = 0;
        self.vector_len = 0;
        self.at = 0;
        self.buf.fill(0.0);
    }
}

/// What is known about the subframe being decoded.
struct Subframe {
    len: usize,
    size: usize,
    bands: usize,
    /// Bits of an offset escape in the run coded coefficients.
    offset_bits: u32,
    members: [usize; 2],
    member_count: usize,
    /// Whether the number of vector coded coefficients is sent for each channel.
    sends_vector_len: bool,
}

impl Subframe {
    fn members(&self) -> &[usize] {
        &self.members[..self.member_count]
    }
}

/// Mid and side handling for the channels of a subframe.
struct Transform {
    enabled: bool,
    bands: [bool; MAX_BANDS],
}

/// Decodes the frames of one XMA stream, one or two channels, to interleaved 16 bit samples.
pub struct Decoder {
    channels: usize,
    band_count: [usize; SIZES],
    band_edges: [[usize; MAX_BANDS + 1]; SIZES],
    /// For a band of one block size, the band of another block size that holds its centre: `[to][from][band]`.
    band_map: [[[usize; MAX_BANDS]; SIZES]; SIZES],
    chans: Vec<Channel>,
    scratch: Vec<f32>,
}

impl Decoder {
    /// A decoder for `channels` (1 or 2, anything more counts as 2) at `rate` Hz.
    pub fn new(channels: usize, rate: u32) -> Decoder {
        let channels = channels.clamp(1, 2);
        let rate = match rate {
            r if r > 44_100 => 48_000,
            r if r > 32_000 => 44_100,
            r if r > 24_000 => 32_000,
            _ => 24_000,
        } as usize;
        let mut band_count = [0; SIZES];
        let mut band_edges = [[0usize; MAX_BANDS + 1]; SIZES];
        for (size, edges) in band_edges.iter_mut().enumerate() {
            let len = FRAME_SAMPLES >> size;
            let mut n = 1;
            for &hz in BAND_EDGES_HZ.iter().take(MAX_BANDS - 1) {
                if edges[n - 1] >= len {
                    break;
                }
                let edge = (len * 2 * hz as usize / rate + 2) & !3;
                if edge > edges[n - 1] {
                    edges[n] = edge;
                    n += 1;
                }
                if edge >= len {
                    break;
                }
            }
            edges[n - 1] = len;
            band_count[size] = n - 1;
        }
        let mut band_map = [[[0usize; MAX_BANDS]; SIZES]; SIZES];
        for to in 0..SIZES {
            for band in 0..band_count[to] {
                let centre = ((band_edges[to][band] + band_edges[to][band + 1] - 1) << to) >> 1;
                for from in 0..SIZES {
                    let mut found = 0;
                    while found + 1 < MAX_BANDS && (band_edges[from][found + 1] << from) < centre {
                        found += 1;
                    }
                    band_map[to][from][band] = found;
                }
            }
        }
        Decoder {
            channels,
            band_count,
            band_edges,
            band_map,
            chans: vec![Channel::new(); channels],
            scratch: vec![0.0; FRAME_SAMPLES],
        }
    }

    /// The number of channels this decoder was made for.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The number of scale factor bands of block size `size` (0 for 512 samples, 1 for 256, 2 for 128).
    pub fn bands(&self, size: usize) -> usize {
        self.band_count[size.min(SIZES - 1)]
    }

    /// Forgets the overlap of the previous frame, as at the start of a stream.
    pub fn reset(&mut self) {
        for c in &mut self.chans {
            c.reset();
        }
    }

    /// Decodes `frame` and appends its 512 samples per channel, interleaved. A frame that does not decode appends silence of
    /// the same length and gives the error.
    pub fn decode(&mut self, frame: &Frame, out: &mut Vec<i16>) -> Result<(), Error> {
        let result = self.decode_frame(frame);
        out.reserve(FRAME_SAMPLES * self.channels);
        if result.is_ok() {
            match self.chans.as_slice() {
                [c0] => {
                    for &s in &c0.buf[..FRAME_SAMPLES] {
                        out.push(to_pcm(s));
                    }
                }
                [c0, c1] => {
                    for i in 0..FRAME_SAMPLES {
                        out.push(to_pcm(c0.buf[i]));
                        out.push(to_pcm(c1.buf[i]));
                    }
                }
                _ => {
                    for i in 0..FRAME_SAMPLES {
                        out.extend(self.chans.iter().map(|c| to_pcm(c.buf[i])));
                    }
                }
            }
            for c in &mut self.chans {
                c.buf
                    .copy_within(FRAME_SAMPLES..FRAME_SAMPLES + FRAME_SAMPLES / 2, 0);
            }
        } else {
            out.resize(out.len() + FRAME_SAMPLES * self.channels, 0);
            self.reset();
        }
        result
    }

    /// Ends the stream: appends the half frame still held for overlap, 256 samples per channel.
    pub fn flush(&mut self, out: &mut Vec<i16>) {
        out.reserve(FRAME_SAMPLES / 2 * self.channels);
        match self.chans.as_slice() {
            [c0] => {
                for &s in &c0.buf[..FRAME_SAMPLES / 2] {
                    out.push(to_pcm(s));
                }
            }
            [c0, c1] => {
                for i in 0..FRAME_SAMPLES / 2 {
                    out.push(to_pcm(c0.buf[i]));
                    out.push(to_pcm(c1.buf[i]));
                }
            }
            _ => {
                for i in 0..FRAME_SAMPLES / 2 {
                    out.extend(self.chans.iter().map(|c| to_pcm(c.buf[i])));
                }
            }
        }
        self.reset();
    }

    fn decode_frame(&mut self, frame: &Frame) -> Result<(), Error> {
        let mut r = BitReader::new(&frame.data, frame.bits);
        r.take(15);
        self.read_tiling(&mut r)?;
        if self.channels > 1 && r.flag() && r.flag() {
            // A post processing transform: its matrix is read past and not applied.
            r.skip(4 * self.channels * self.channels);
        }
        // The dynamic range gain is not applied.
        r.take(8);
        if r.flag() {
            // Trims at the start and the end: the skips of the context do the trimming.
            for _ in 0..2 {
                if r.flag() {
                    r.take(10);
                }
            }
        }
        for c in &mut self.chans {
            c.done = 0;
            c.sub_index = 0;
            c.sf_known = false;
        }
        while !self.decode_subframe(&mut r)? {}
        if r.overrun() {
            return Err(Error("the frame is shorter than its contents"));
        }
        Ok(())
    }

    /// Reads how each channel's 512 samples are cut into subframes.
    fn read_tiling(&mut self, r: &mut BitReader) -> Result<(), Error> {
        let n = self.channels;
        for c in &mut self.chans {
            c.sub_count = 0;
        }
        let same_for_all = r.flag();
        let mut covered = [0usize; 2];
        let mut takes_part = [false; 2];
        let mut waiting = n;
        let mut cursor = 0;
        loop {
            for c in 0..n {
                takes_part[c] = covered[c] == cursor
                    && (same_for_all
                        || waiting == 1
                        || cursor == FRAME_SAMPLES - SMALLEST
                        || r.flag());
            }
            let len = if cursor == FRAME_SAMPLES - SMALLEST {
                SMALLEST
            } else if r.pos() >= r.end() {
                return Err(Error("subframe length past the end"));
            } else {
                FRAME_SAMPLES >> if r.flag() { 1 + r.take(1) } else { 0 }
            };
            cursor += len;
            for c in 0..n {
                let ch = &mut self.chans[c];
                if takes_part[c] {
                    if ch.sub_count >= MAX_SUBFRAMES {
                        return Err(Error("too many subframes"));
                    }
                    ch.sub_lens[ch.sub_count] = len;
                    ch.sub_count += 1;
                    covered[c] += len;
                    if covered[c] > FRAME_SAMPLES {
                        return Err(Error("subframes longer than the frame"));
                    }
                } else if covered[c] <= cursor {
                    if covered[c] < cursor {
                        waiting = 0;
                        cursor = covered[c];
                    }
                    waiting += 1;
                }
            }
            if cursor >= FRAME_SAMPLES {
                return Ok(());
            }
        }
    }

    /// Decodes the next subframe of the frame. Returns whether the frame is complete.
    fn decode_subframe(&mut self, r: &mut BitReader) -> Result<bool, Error> {
        let mut sub = self.pick_subframe()?;
        let done =
            self.chans.iter().map(|c| c.done).sum::<usize>() == FRAME_SAMPLES * self.channels;

        if r.flag() {
            // An extended header holding a count of fill bits.
            let mut fill = r.take(2) as usize;
            if fill == 0 {
                let width = r.take(4);
                fill = r.take(width) as usize + 1;
            }
            if r.pos() + fill > r.end() {
                return Err(Error("fill bits past the end"));
            }
            r.skip(fill);
        }
        if r.flag() {
            return Err(Error("reserved bit"));
        }
        let transform = self.read_channel_transform(r, &sub)?;

        let mut any = false;
        for &c in sub.members() {
            let has = r.flag();
            self.chans[c].has_coefs = has;
            any |= has;
        }
        if any {
            self.read_quantisation(r, &mut sub)?;
            self.read_scale_factors(r, &sub)?;
        }

        for &c in sub.members() {
            if self.chans[c].has_coefs && r.pos() < r.end() {
                // A damaged coefficient block keeps what was read so far, and decoding goes on.
                let _ = self.read_coefficients(r, &sub, c);
            } else {
                let at = self.chans[c].at;
                self.chans[c].buf[at..at + sub.len].fill(0.0);
            }
        }

        if any {
            self.apply_transform(&sub, &transform);
            self.dequantise_and_invert(&sub);
        }
        self.overlap_blocks(&sub);

        for &c in sub.members() {
            let ch = &mut self.chans[c];
            if ch.sub_index >= ch.sub_count {
                return Err(Error("broken subframe"));
            }
            ch.sub_index += 1;
        }
        Ok(done)
    }

    /// Chooses the subframe to read next: the one that starts earliest, with all channels that start one of the same length
    /// at that point.
    fn pick_subframe(&mut self) -> Result<Subframe, Error> {
        let mut start = FRAME_SAMPLES;
        let mut len = FRAME_SAMPLES;
        for c in &self.chans {
            if start > c.done {
                start = c.done;
                len = *c.sub_lens.get(c.sub_index).ok_or(Error("subframe index"))?;
            }
        }
        if len == 0 {
            return Err(Error("empty subframe"));
        }
        let mut members = [0usize; 2];
        let mut member_count = 0;
        for (i, c) in self.chans.iter_mut().enumerate() {
            if c.done == start && c.sub_index < c.sub_count && len == c.sub_lens[c.sub_index] {
                c.done += len;
                members[member_count] = i;
                member_count += 1;
                c.at = start + FRAME_SAMPLES / 2;
            }
        }
        if member_count == 0 {
            return Err(Error("no channel for the subframe"));
        }
        let size = size_index(len);
        Ok(Subframe {
            len,
            size,
            bands: self.band_count[size],
            offset_bits: floor_log2(len - 1) + 1,
            members,
            member_count,
            sends_vector_len: false,
        })
    }

    fn read_channel_transform(
        &mut self,
        r: &mut BitReader,
        sub: &Subframe,
    ) -> Result<Transform, Error> {
        let mut transform = Transform {
            enabled: false,
            bands: [false; MAX_BANDS],
        };
        if self.channels < 2 {
            return Ok(transform);
        }
        if r.flag() {
            return Err(Error("channel transform bit"));
        }
        if sub.member_count == 2 {
            if r.flag() {
                if r.flag() {
                    return Err(Error("unknown channel transform"));
                }
            } else {
                transform.enabled = true;
            }
        }
        if transform.enabled {
            if r.flag() {
                transform.bands = [true; MAX_BANDS];
            } else {
                for flag in transform.bands.iter_mut().take(sub.bands) {
                    *flag = r.flag();
                }
            }
        }
        Ok(transform)
    }

    /// Reads whether vector lengths are sent, the quantisation step and its per channel modifiers.
    fn read_quantisation(&mut self, r: &mut BitReader, sub: &mut Subframe) -> Result<(), Error> {
        // 90 corresponds to 16 bit samples.
        let mut step = 90;
        sub.sends_vector_len = r.flag();
        if sub.sends_vector_len {
            let width = floor_log2(sub.len.div_ceil(4)) + 1;
            for &c in sub.members() {
                let count = (r.take(width) as usize) << 2;
                if count > sub.len {
                    return Err(Error("too many vector coefficients"));
                }
                self.chans[c].vector_len = count;
            }
        } else {
            for &c in sub.members() {
                self.chans[c].vector_len = sub.len;
            }
        }
        let first = r.take_signed(6);
        step += first;
        if first == -32 || first == 31 {
            // Larger steps continue in groups of 5 bits, 31 meaning more follows.
            let mut extra = 0;
            let mut last = first;
            while r.pos() + 5 < r.end() {
                last = r.take(5) as i32;
                if last != 31 {
                    break;
                }
                extra += 31;
            }
            step += if first == -32 {
                -(extra + last)
            } else {
                extra + last
            };
        }
        if sub.member_count == 1 {
            self.chans[sub.members[0]].quant_step = step;
        } else {
            let width = r.take(3);
            for &c in sub.members() {
                self.chans[c].quant_step = step;
                if r.flag() {
                    self.chans[c].quant_step += if width != 0 {
                        r.take(width) as i32 + 1
                    } else {
                        1
                    };
                }
            }
        }
        Ok(())
    }

    fn read_scale_factors(&mut self, r: &mut BitReader, sub: &Subframe) -> Result<(), Error> {
        let tables = shared();
        for &c in sub.members() {
            let ch = &mut self.chans[c];
            let (old, new) = (ch.sf_sent, 1 - ch.sf_sent);
            ch.sf_used = new;
            if ch.sf_known {
                // Carry the last scale factors over to this block size.
                let map = &self.band_map[sub.size][ch.sf_size];
                for (band, &from) in map.iter().enumerate().take(sub.bands) {
                    ch.sf[new][band] = ch.sf[old][from];
                }
            }
            if ch.sub_index == 0 || r.flag() {
                if ch.sf_known {
                    // Changes to the carried values, as a list of runs.
                    let mut band = 0;
                    while band < sub.bands {
                        let symbol = tables
                            .scale_run
                            .decode(r)
                            .ok_or(Error("scale factor run code"))?
                            as usize;
                        let (skip, level, negative) = match symbol {
                            0 => {
                                let raw = r.take(14);
                                (
                                    ((raw & 0x3F) >> 1) as usize,
                                    (raw >> 6) as i32,
                                    raw & 1 == 0,
                                )
                            }
                            1 => break,
                            _ => {
                                let (skip, level) = SCALE_RUN_STEPS[symbol];
                                (usize::from(skip), i32::from(level), !r.flag())
                            }
                        };
                        band += skip;
                        if band >= sub.bands {
                            return Err(Error("scale factor run past the bands"));
                        }
                        ch.sf[new][band] += if negative { -level } else { level };
                        band += 1;
                    }
                } else {
                    // The first set of the frame: a step size, then a difference for every band.
                    ch.sf_step = r.take(2) as i32 + 1;
                    let mut value = 45 / ch.sf_step;
                    for band in 0..sub.bands {
                        value += i32::from(
                            tables
                                .scale_delta
                                .decode(r)
                                .ok_or(Error("scale factor code"))?,
                        ) - 60;
                        ch.sf[new][band] = value;
                    }
                }
                ch.sf_sent = new;
                ch.sf_size = sub.size;
                ch.sf_known = true;
            }
            ch.sf_max = ch.sf[new][..sub.bands.max(1)]
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
        }
        Ok(())
    }

    /// Reads the coefficients of one channel: first vectors of four, then runs of zeros with levels.
    fn read_coefficients(
        &mut self,
        r: &mut BitReader,
        sub: &Subframe,
        c: usize,
    ) -> Result<(), Error> {
        let tables = shared();
        let second_table = r.flag();
        let (code, steps): (&PrefixCode, &[(u16, u16)]) = if second_table {
            (&tables.coef[1], &COEF_B_STEPS)
        } else {
            (&tables.coef[0], &COEF_A_STEPS)
        };
        let len = sub.len;
        let ch = &mut self.chans[c];
        let base = ch.at;
        let mut count = 0;
        let mut run_mode = false;
        let mut zeros = 0;
        while (sub.sends_vector_len || !run_mode) && count + 3 < ch.vector_len {
            let symbol = tables.quad.decode(r).ok_or(Error("vector code"))?;
            let values: [u32; 4] = if symbol == 0 {
                let mut values = [0u32; 4];
                for half in values.chunks_mut(2) {
                    let pair = tables.pair.decode(r).ok_or(Error("vector code"))?;
                    if pair == 0 {
                        for v in half {
                            let mut x =
                                u32::from(tables.single.decode(r).ok_or(Error("vector code"))?);
                            if x == u32::from(SINGLE_ESCAPE) {
                                x += r.take_wide();
                            }
                            *v = x;
                        }
                    } else {
                        let packed = u32::from(pair - 1);
                        half[0] = packed >> 4;
                        half[1] = packed & 0xF;
                    }
                }
                values
            } else {
                let packed = u32::from(symbol - 1);
                [
                    packed >> 12,
                    packed >> 8 & 0xF,
                    packed >> 4 & 0xF,
                    packed & 0xF,
                ]
            };
            for v in values {
                if v != 0 {
                    let magnitude = v as f32;
                    ch.buf[base + count] = if r.flag() { magnitude } else { -magnitude };
                    zeros = 0;
                } else {
                    ch.buf[base + count] = 0.0;
                    zeros += 1;
                    run_mode |= zeros > len >> 8;
                }
                count += 1;
            }
        }
        if count < len {
            ch.buf[base + count..base + len].fill(0.0);
            let mask = len - 1;
            let mut at = count;
            while at < len {
                let symbol = usize::from(code.decode(r).ok_or(Error("coefficient code"))?);
                let level = match symbol {
                    1 => break,
                    0 => {
                        let level = r.take_wide();
                        if r.flag() {
                            if r.flag() {
                                if r.flag() {
                                    return Err(Error("broken escape"));
                                }
                                at += r.take(sub.offset_bits) as usize + 4;
                            } else {
                                at += r.take(2) as usize + 1;
                            }
                        }
                        level as f32
                    }
                    _ => {
                        let (skip, level) = steps[symbol];
                        at += usize::from(skip);
                        f32::from(level)
                    }
                };
                ch.buf[base + (at & mask)] = if r.flag() { level } else { -level };
                at += 1;
            }
            if at > len {
                return Err(Error("coefficient run past the subframe"));
            }
        }
        Ok(())
    }

    /// Undoes the mid and side transform of a pair of channels, band by band.
    fn apply_transform(&mut self, sub: &Subframe, transform: &Transform) {
        if !transform.enabled || sub.member_count != 2 {
            return;
        }
        let edges = self.band_edges[sub.size];
        let (a, b) = (sub.members[0], sub.members[1]);
        let (base_a, base_b) = (self.chans[a].at, self.chans[b].at);
        for band in 0..sub.bands {
            let range = edges[band]..edges[band + 1].min(sub.len);
            for y in range {
                let (x0, x1) = (self.chans[a].buf[base_a + y], self.chans[b].buf[base_b + y]);
                if transform.bands[band] {
                    self.chans[a].buf[base_a + y] = x0 - x1;
                    self.chans[b].buf[base_b + y] = x0 + x1;
                } else {
                    self.chans[a].buf[base_a + y] = x0 * PLAIN_GAIN;
                    self.chans[b].buf[base_b + y] = x1 * PLAIN_GAIN;
                }
            }
        }
    }

    /// Scales every band by its quantisation step and runs the inverse MDCT.
    fn dequantise_and_invert(&mut self, sub: &Subframe) {
        let tables = shared();
        let edges = self.band_edges[sub.size];
        let scale = 2.0 / sub.len as f64 / 32768.0;
        for &c in sub.members() {
            let ch = &mut self.chans[c];
            for band in 0..sub.bands {
                let exponent = ch.quant_step - (ch.sf_max - ch.sf[ch.sf_used][band]) * ch.sf_step;
                let gain = if (GAIN_MIN..=GAIN_MAX).contains(&exponent) {
                    tables.gains[(exponent - GAIN_MIN) as usize]
                } else {
                    10f32.powf(exponent as f32 / 20.0)
                };
                for y in edges[band]..edges[band + 1].min(sub.len) {
                    self.scratch[y] = ch.buf[ch.at + y] * gain;
                }
            }
            let at = ch.at;
            tables.imdct[sub.size].run(
                &self.scratch[..sub.len],
                &mut ch.buf[at..at + sub.len],
                scale,
            );
        }
    }

    /// Windows the new block and adds it to the tail of the previous one.
    fn overlap_blocks(&mut self, sub: &Subframe) {
        let tables = shared();
        for &c in sub.members() {
            let ch = &mut self.chans[c];
            let mut width = ch.prev_len;
            let mut from = ch.at - width / 2;
            if sub.len < width {
                from += (width - sub.len) / 2;
                width = sub.len;
            }
            overlap_add(
                &mut ch.buf[from..from + width],
                &tables.windows[size_index(width)],
                width / 2,
            );
            ch.prev_len = sub.len;
        }
    }
}

fn to_pcm(v: f32) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests;
