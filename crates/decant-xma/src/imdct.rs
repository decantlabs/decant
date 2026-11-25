//! The middle half of the inverse MDCT, and the sine windows.
//!
//! For `n` coefficients X the full inverse MDCT gives 2n samples y[t] = sum over k of X[k] cos(pi/n (t + 1/2 + n/2)(k + 1/2)).
//! The decoder only needs y[n/2 .. 3n/2), the rest follows by symmetry. That middle part is a DCT IV of X, reversed and
//! negated. It is computed through a complex FFT of n/2 points.

use std::f64::consts::PI;

type Cx = (f64, f64);

fn unit(angle: f64) -> Cx {
    (angle.cos(), angle.sin())
}

/// Tables for one transform size.
pub(crate) struct HalfImdct {
    n: usize,
    /// exp(-i pi (j + 1/8) / n) for j below n/2, used before and after the FFT.
    rotate: Vec<Cx>,
    /// exp(-2 pi i j / (n/2)) for j below n/4.
    roots: Vec<Cx>,
}

impl HalfImdct {
    pub(crate) fn new(n: usize) -> Self {
        let m = n / 2;
        HalfImdct {
            n,
            rotate: (0..m)
                .map(|j| unit(-PI * (j as f64 + 0.125) / n as f64))
                .collect(),
            roots: (0..m / 2)
                .map(|j| unit(-2.0 * PI * j as f64 / m as f64))
                .collect(),
        }
    }

    /// Writes the middle half of the inverse MDCT of `input` (n values), times `scale`, to `output` (n values).
    pub(crate) fn run(&self, input: &[f32], output: &mut [f32], scale: f64) {
        let n = self.n;
        let m = n / 2;
        let mut re = vec![0f64; m];
        let mut im = vec![0f64; m];
        // Fold the input into m complex values and rotate them.
        for (j, &(c, s)) in self.rotate.iter().enumerate() {
            let a = f64::from(input[2 * j]);
            let b = f64::from(input[n - 1 - 2 * j]);
            re[j] = a * c - b * s;
            im[j] = a * s + b * c;
        }
        self.fft(&mut re, &mut im);
        // Rotate again and unfold: even outputs come from the imaginary parts, odd ones from the real parts, reversed.
        for (j, &(c, s)) in self.rotate.iter().enumerate() {
            let real = re[j] * c - im[j] * s;
            let imag = re[j] * s + im[j] * c;
            output[n - 1 - 2 * j] = (-real * scale) as f32;
            output[2 * j] = (imag * scale) as f32;
        }
    }

    /// An in place radix 2 FFT over the m points.
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let m = re.len();
        // Bit reversal permutation.
        let mut rev = 0usize;
        for i in 1..m {
            let mut bit = m >> 1;
            while rev & bit != 0 {
                rev ^= bit;
                bit >>= 1;
            }
            rev |= bit;
            if i < rev {
                re.swap(i, rev);
                im.swap(i, rev);
            }
        }
        // Butterflies, doubling the block size each pass.
        let mut block = 2;
        while block <= m {
            let stride = m / block;
            let half = block / 2;
            for base in (0..m).step_by(block) {
                for k in 0..half {
                    let (c, s) = self.roots[k * stride];
                    let (lo, hi) = (base + k, base + k + half);
                    let tr = re[hi] * c - im[hi] * s;
                    let ti = re[hi] * s + im[hi] * c;
                    re[hi] = re[lo] - tr;
                    im[hi] = im[lo] - ti;
                    re[lo] += tr;
                    im[lo] += ti;
                }
            }
            block *= 2;
        }
    }
}

/// The rising sine window of `len` samples: sin((i + 1/2) pi / (2 len)).
pub(crate) fn sine_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f64 + 0.5) * PI / (2.0 * len as f64)).sin() as f32)
        .collect()
}

/// Windows and overlap adds in place. `buf[..2 half]` holds the tail of the previous block in its first half and the start of
/// the new block in its second half. `window` holds 2 `half` values.
pub(crate) fn overlap_add(buf: &mut [f32], window: &[f32], half: usize) {
    for k in 0..half {
        let j = 2 * half - 1 - k;
        let (old, new) = (buf[k], buf[j]);
        let (rise, fall) = (window[k], window[j]);
        buf[k] = old * fall - new * rise;
        buf[j] = old * rise + new * fall;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(input: &[f32]) -> Vec<f64> {
        let n = input.len();
        (n / 2..3 * n / 2)
            .map(|t| {
                (0..n)
                    .map(|k| {
                        f64::from(input[k])
                            * (PI / n as f64 * (t as f64 + 0.5 + n as f64 / 2.0) * (k as f64 + 0.5))
                                .cos()
                    })
                    .sum()
            })
            .collect()
    }

    #[test]
    fn the_fast_transform_matches_the_definition() {
        for n in [128usize, 256, 512] {
            let input: Vec<f32> = (0..n)
                .map(|i| ((i * 7919 % 113) as f32 - 56.0) / 17.0)
                .collect();
            let mut out = vec![0f32; n];
            HalfImdct::new(n).run(&input, &mut out, 1.0);
            for (t, (a, b)) in out.iter().zip(direct(&input)).enumerate() {
                assert!(
                    (f64::from(*a) - b).abs() < 1e-3 * (1.0 + b.abs()),
                    "n {n} t {t}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn windowed_blocks_add_up_to_the_signal() {
        // Time domain aliasing cancellation: windowed blocks of 2n, transformed, inverted and overlap added.
        let n = 256;
        let signal: Vec<f64> = (0..4 * n)
            .map(|i| (i as f64 * 0.05).sin() + 0.3 * (i as f64 * 0.31).cos())
            .collect();
        let rising = sine_window(n);
        let window: Vec<f32> = (0..2 * n)
            .map(|t| {
                if t < n {
                    rising[t]
                } else {
                    rising[2 * n - 1 - t]
                }
            })
            .collect();
        let tf = HalfImdct::new(n);
        let mut out = vec![0f32; 5 * n];
        for block in 0..3 {
            let at = block * n;
            let coeffs: Vec<f32> = (0..n)
                .map(|k| {
                    (0..2 * n)
                        .map(|t| {
                            signal[at + t]
                                * f64::from(window[t])
                                * (PI / n as f64
                                    * (t as f64 + 0.5 + n as f64 / 2.0)
                                    * (k as f64 + 0.5))
                                    .cos()
                        })
                        .sum::<f64>() as f32
                })
                .collect();
            let mut half = vec![0f32; n];
            tf.run(&coeffs, &mut half, 2.0 / n as f64);
            let mut full = vec![0f32; 2 * n];
            full[n / 2..3 * n / 2].copy_from_slice(&half);
            for k in 0..n / 2 {
                full[k] = -half[n / 2 - 1 - k];
                full[2 * n - 1 - k] = half[n / 2 + k];
            }
            for t in 0..2 * n {
                out[at + t] += full[t] * window[t];
            }
        }
        for t in n..3 * n {
            assert!((f64::from(out[t]) - signal[t]).abs() < 1e-3, "t {t}");
        }
    }

    #[test]
    fn overlap_of_equal_blocks_keeps_the_energy() {
        // With the sine window, rotating a pair (old, new) keeps old^2 + new^2.
        let half = 8;
        let window = sine_window(2 * half);
        let mut buf: Vec<f32> = (0..2 * half).map(|i| i as f32 - 5.0).collect();
        let before: f32 = buf.iter().map(|v| v * v).sum();
        overlap_add(&mut buf, &window, half);
        let after: f32 = buf.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-3);
    }
}
