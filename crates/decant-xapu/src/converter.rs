//! The pitch converting sample rate converter of the XAPU (`docs/CONSOLE_LIBS.md` 3.4).
//!
//! The console filter is a polyphase converter with unknown coefficients, so this one is a windowed sinc as the specification
//! allows. Output frame `n` is the band limited input signal read at the time `pos - 5`, so the group delay is 5 frames: at
//! pitch 1.0 the output is the input delayed by exactly 5 frames. After each output frame the read position advances by the
//! pitch and the pitch moves toward the target by the ramp. The kernel has 10 taps and a Hann window, its cutoff drops to
//! `1 / pitch` when the pitch is above 1 so the conversion does not alias.

/// Group delay in frames, and the half width of the kernel.
pub const DELAY: usize = 5;
pub const PITCH_MIN: f32 = 0.125;
pub const PITCH_MAX: f32 = 8.0;
/// Smallest ramp magnitude, 2^-18.
pub const RAMP_MIN: f32 = 1.0 / 262_144.0;
pub const RAMP_MAX: f32 = 8.0;

/// Clamps a pitch to 0.125 .. 8.
pub fn clamp_pitch(p: f32) -> f32 {
    p.clamp(PITCH_MIN, PITCH_MAX)
}

/// Clamps the magnitude of a ramp to 2^-18 .. 8.
pub fn clamp_ramp(r: f32) -> f32 {
    r.abs().clamp(RAMP_MIN, RAMP_MAX)
}

pub struct Converter {
    channels: usize,
    /// Input history per channel, `hist[c][i]` is input frame `base + i`.
    hist: Vec<Vec<f32>>,
    base: i64,
    /// Input frames received since the last reset.
    total: i64,
    /// Read position of the next output frame, in input frames.
    pos: f64,
    pitch: f32,
    /// Output frames waiting to be taken, per channel.
    pending: Vec<Vec<f32>>,
}

fn kernel(x: f64, cutoff: f64) -> f64 {
    if x.abs() >= DELAY as f64 {
        return 0.0;
    }
    let t = x * cutoff;
    let sinc = if t.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
    cutoff * sinc * (0.5 + 0.5 * (std::f64::consts::PI * x / DELAY as f64).cos())
}

impl Converter {
    pub fn new(channels: usize, pitch: f32) -> Converter {
        Converter { channels, hist: vec![Vec::new(); channels], base: 0, total: 0, pos: 0.0, pitch: clamp_pitch(pitch), pending: vec![Vec::new(); channels] }
    }

    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Clears the history and the pending output and sets the pitch.
    pub fn reset(&mut self, pitch: f32) {
        *self = Converter::new(self.channels, pitch);
    }

    /// Output frames waiting.
    pub fn pending_frames(&self) -> usize {
        self.pending.first().map_or(0, Vec::len)
    }

    /// Input frames at or ahead of the read position, not yet converted.
    pub fn unconverted_frames(&self) -> usize {
        (self.total - (self.pos.floor() as i64).min(self.total)).max(0) as usize
    }

    /// Adds `input` (one slice per channel, the same length) and converts as far as the input allows.
    pub fn push(&mut self, input: &[&[f32]], target: f32, ramp: f32) {
        let frames = input.first().map_or(0, |c| c.len());
        for (hist, chan) in self.hist.iter_mut().zip(input) {
            hist.extend_from_slice(chan);
        }
        self.total += frames as i64;
        self.produce(clamp_pitch(target), clamp_ramp(ramp));
        self.trim();
    }

    /// Feeds the delay with silence so the last input frames come out. The converter then starts over, keeping its pitch.
    pub fn flush(&mut self, target: f32, ramp: f32) {
        let zeros = vec![0.0f32; DELAY];
        let slices: Vec<&[f32]> = (0..self.channels).map(|_| zeros.as_slice()).collect();
        self.push(&slices, target, ramp);
    }

    /// Forgets the input history but keeps the pending output: used after a flush, when the output is collected.
    pub fn restart(&mut self) {
        self.hist = vec![Vec::new(); self.channels];
        self.base = 0;
        self.total = 0;
        self.pos = 0.0;
    }

    /// Removes the first `frames` pending frames and appends them to `out` (one vector per channel).
    pub fn take(&mut self, frames: usize, out: &mut [Vec<f32>]) {
        for (p, o) in self.pending.iter_mut().zip(out) {
            o.extend(p.drain(..frames));
        }
    }

    fn produce(&mut self, target: f32, ramp: f32) {
        while (self.pos.floor() as i64) < self.total {
            let centre = self.pos - DELAY as f64;
            let first = centre.floor() as i64 - (DELAY as i64 - 1);
            let cutoff = f64::from((1.0 / self.pitch).min(1.0));
            let mut weights = [0.0f64; 2 * DELAY];
            let mut sum = 0.0;
            for (i, w) in weights.iter_mut().enumerate() {
                *w = kernel(centre - (first + i as i64) as f64, cutoff);
                sum += *w;
            }
            if sum.abs() > 1e-9 {
                weights.iter_mut().for_each(|w| *w /= sum);
            }
            for c in 0..self.channels {
                let mut acc = 0.0f64;
                for (i, w) in weights.iter().enumerate() {
                    let k = first + i as i64;
                    if k >= 0 && *w != 0.0 {
                        let idx = (k - self.base) as usize;
                        acc += w * f64::from(self.hist[c].get(idx).copied().unwrap_or(0.0));
                    }
                }
                self.pending[c].push(acc as f32);
            }
            self.pos += f64::from(self.pitch);
            if self.pitch < target {
                self.pitch = (self.pitch + ramp).min(target);
            } else if self.pitch > target {
                self.pitch = (self.pitch - ramp).max(target);
            }
        }
    }

    /// Drops input frames the next output can no longer reach.
    fn trim(&mut self) {
        let keep_from = (self.pos - DELAY as f64).floor() as i64 - DELAY as i64 - 2;
        let drop = (keep_from - self.base).max(0) as usize;
        if drop > 0 {
            for h in &mut self.hist {
                let n = drop.min(h.len());
                h.drain(..n);
            }
            self.base += drop as i64;
        }
    }
}

#[cfg(test)]
mod tests;
