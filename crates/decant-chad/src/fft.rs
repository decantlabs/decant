//! The transforms of `chad.dll`, in double precision.
//!
//! The packed real format of the specification: for `N` real samples the spectrum is `N` values, `out[0]` the DC bin, `out[1]` the
//! Nyquist bin (both real), then the pairs `(re, im)` of bins 1 to `N / 2 - 1`. The forward transform is unscaled, the inverse
//! divides by `N`.

use std::f64::consts::PI;

/// In place radix 2 complex transform of a power of two length. `inverse` flips the sign of the exponent, no scaling.
pub fn fft_complex(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    assert!(n.is_power_of_two() && im.len() == n, "a power of two length");
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) };
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let step = sign * 2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (step * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len *= 2;
    }
}

/// Forward transform of `x` (a power of two number of real samples, at least 4) into the packed format.
pub fn forward_packed(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut re = x.to_vec();
    let mut im = vec![0.0; n];
    fft_complex(&mut re, &mut im, false);
    let mut out = vec![0.0; n];
    out[0] = re[0];
    out[1] = re[n / 2];
    for k in 1..n / 2 {
        out[2 * k] = re[k];
        out[2 * k + 1] = im[k];
    }
    out
}

/// Inverse transform of a packed spectrum back to `N` real samples, scaled by `1 / N`.
pub fn inverse_packed(spectrum: &[f64]) -> Vec<f64> {
    let n = spectrum.len();
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    re[0] = spectrum[0];
    re[n / 2] = spectrum[1];
    for k in 1..n / 2 {
        re[k] = spectrum[2 * k];
        im[k] = spectrum[2 * k + 1];
        re[n - k] = spectrum[2 * k];
        im[n - k] = -spectrum[2 * k + 1];
    }
    fft_complex(&mut re, &mut im, true);
    re.iter().map(|v| v / n as f64).collect()
}

/// Product of two packed spectra, bin by bin, added to `acc` after scaling by `gain`.
pub fn multiply_accumulate(acc: &mut [f64], a: &[f64], b: &[f64], gain: f64) {
    acc[0] += gain * a[0] * b[0];
    acc[1] += gain * a[1] * b[1];
    for k in 1..acc.len() / 2 {
        let (ar, ai, br, bi) = (a[2 * k], a[2 * k + 1], b[2 * k], b[2 * k + 1]);
        acc[2 * k] += gain * (ar * br - ai * bi);
        acc[2 * k + 1] += gain * (ar * bi + ai * br);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition: `X[k] = sum x[n] exp(-2 pi i k n / N)`.
    fn dft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        for k in 0..n {
            for (i, v) in x.iter().enumerate() {
                let a = -2.0 * PI * ((k * i) % n) as f64 / n as f64;
                re[k] += v * a.cos();
                im[k] += v * a.sin();
            }
        }
        (re, im)
    }

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_packed_forward_transform_matches_the_dft() {
        for n in [8usize, 64, 512] {
            let x = noise(n, n as u64);
            let (re, im) = dft(&x);
            let p = forward_packed(&x);
            assert!((p[0] - re[0]).abs() < 1e-9);
            assert!((p[1] - re[n / 2]).abs() < 1e-9);
            for k in 1..n / 2 {
                assert!((p[2 * k] - re[k]).abs() < 1e-9, "n {n} bin {k} re");
                assert!((p[2 * k + 1] - im[k]).abs() < 1e-9, "n {n} bin {k} im");
            }
        }
    }

    #[test]
    fn the_inverse_undoes_the_forward_transform() {
        for n in [4usize, 16, 512, 2048] {
            let x = noise(n, 7);
            let back = inverse_packed(&forward_packed(&x));
            for (a, b) in x.iter().zip(&back) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn known_signals() {
        // An impulse has a flat spectrum, a constant has only a DC bin.
        let mut x = vec![0.0; 16];
        x[0] = 1.0;
        let p = forward_packed(&x);
        assert!(p.iter().step_by(2).all(|v| (v - 1.0).abs() < 1e-12));
        assert!((p[1] - 1.0).abs() < 1e-12, "the Nyquist bin");
        assert!(p.iter().skip(3).step_by(2).all(|v| v.abs() < 1e-12), "no imaginary parts");
        let p = forward_packed(&[2.0; 16]);
        assert!((p[0] - 32.0).abs() < 1e-12);
        assert!(p[1..].iter().all(|v| v.abs() < 1e-12));
        // An alternating signal has only a Nyquist bin.
        let alt: Vec<f64> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let p = forward_packed(&alt);
        assert!((p[1] - 16.0).abs() < 1e-12);
        assert!(p[0].abs() < 1e-12 && p[2..].iter().all(|v| v.abs() < 1e-12));
        // A cosine of bin 3 puts N / 2 in the real part of bin 3.
        let c: Vec<f64> = (0..16).map(|i| (2.0 * PI * 3.0 * i as f64 / 16.0).cos()).collect();
        let p = forward_packed(&c);
        assert!((p[6] - 8.0).abs() < 1e-12 && p[7].abs() < 1e-12);
    }

    #[test]
    fn the_product_of_spectra_is_a_circular_convolution() {
        let n = 32;
        let (a, b) = (noise(n, 1), noise(n, 2));
        let mut acc = vec![0.0; n];
        multiply_accumulate(&mut acc, &forward_packed(&a), &forward_packed(&b), 1.0);
        let c = inverse_packed(&acc);
        for (i, v) in c.iter().enumerate() {
            let want: f64 = (0..n).map(|j| a[j] * b[(i + n - j) % n]).sum();
            assert!((v - want).abs() < 1e-12, "{i}");
        }
    }
}
