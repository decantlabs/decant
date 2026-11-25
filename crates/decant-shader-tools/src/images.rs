//! `check_blit.py` and `check_texture.py`: compare a 64 x 64 RGBA image of 32 bit floats written
//! by `decant_shader_run` with the expected result, computed independently. The console shader writes
//! half floats rounded toward zero, so the comparison works at half float precision.
//!
//! Every value is computed in the order the scripts compute it, in double precision, so that the
//! printed worst difference and its position are the same digits.

use std::fmt::Write;

use decant_shader_analysis::NL;

use crate::pyfmt::{float_repr, half_rtz};

pub const SIZE: usize = 64;
const TEX: f64 = 4.0;
/// The vertex colours of the bench triangle (the same vertices as `decant_shader_run.cpp`).
const COLORS: [[f64; 4]; 3] = [[1.00, 0.00, 0.25, 1.00], [0.00, 1.00, 0.50, 0.50], [0.50, 0.25, 1.00, 0.75]];

/// What a check prints, and whether the image passed.
pub struct Verdict {
    pub text: String,
    pub ok: bool,
}

fn pixels(data: &[u8]) -> Vec<f64> {
    data.as_chunks::<4>().0.iter().map(|w| f64::from(f32::from_le_bytes(*w))).collect()
}

/// `(x, y, c, got, want)` as Python prints the tuple, or `None`.
fn worst_at(at: Option<(usize, usize, usize, f64, f64)>) -> String {
    match at {
        Some((x, y, c, got, want)) => format!("({x}, {y}, {c}, {}, {})", float_repr(got), float_repr(want)),
        None => "None".to_owned(),
    }
}

/// `check_blit.py`: the interpolated colour of the full screen triangle, times `factor` per
/// component when given.
pub fn blit(data: &[u8], factor: Option<[f64; 4]>) -> Result<Verdict, String> {
    if data.len() != SIZE * SIZE * 16 {
        return Err(format!("AssertionError: {}", data.len()));
    }
    let pixels = pixels(data);
    let factor = factor.unwrap_or([1.0; 4]);
    let (mut worst, mut exact, mut total, mut undrawn) = (0.0f64, 0usize, 0usize, 0usize);
    let mut at = None;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let ndc_x = -1.0 + 2.0 * (x as f64 + 0.5) / SIZE as f64;
            let ndc_y = -1.0 + 2.0 * (y as f64 + 0.5) / SIZE as f64;
            let (b1, b2) = ((ndc_x + 1.0) / 4.0, (ndc_y + 1.0) / 4.0);
            let b0 = 1.0 - b1 - b2;
            for c in 0..4 {
                let got = pixels[(y * SIZE + x) * 4 + c];
                if got == -1.0 {
                    undrawn += 1;
                }
                let want = (b0 * COLORS[0][c] + b1 * COLORS[1][c] + b2 * COLORS[2][c]) * factor[c];
                let error = (got - want).abs();
                total += 1;
                exact += usize::from(got == half_rtz(want)?);
                if error > worst {
                    worst = error;
                    at = Some((x, y, c, got, want));
                }
            }
        }
    }
    let mut text = format!("composantes : {total}, non dessinees : {undrawn}{NL}");
    write!(text, "ecart maximal : {worst:.6} en {}{NL}", worst_at(at)).unwrap();
    write!(text, "egales au demi-flottant attendu (arrondi vers zero) : {exact} ({:.1} %){NL}", 100.0 * exact as f64 / total as f64).unwrap();
    let ok = undrawn == 0 && worst < 2e-3;
    write!(text, "RESULTAT : {}{NL}", if ok { "CONFORME" } else { "ECART" }).unwrap();
    Ok(Verdict { text, ok })
}

fn texel(i: i64, j: i64) -> [f64; 4] {
    [(40 * i + 10) as f64 / 255.0, (60 * j + 20) as f64 / 255.0, (15 * (i + 4 * j)) as f64 / 255.0, 1.0]
}

/// `check_texture.py`: the 4 x 4 bench texture sampled at the nearest texel, at the coordinate the
/// first input carries: the vertex colour, or with `linked` the real texture coordinate. Pixels
/// less than 0.002 texel from a texel edge are skipped.
pub fn texture(data: &[u8], linked: bool) -> Result<Verdict, String> {
    if data.len() != SIZE * SIZE * 16 {
        return Err(format!("struct.error: unpack requires a buffer of {} bytes", SIZE * SIZE * 16));
    }
    let pixels = pixels(data);
    let (mut checked, mut skipped, mut exact) = (0usize, 0usize, 0usize);
    let (mut worst, mut at) = (0.0f64, None);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let b1 = (2.0 * (x as f64 + 0.5) / SIZE as f64) / 4.0;
            let b2 = (2.0 * (y as f64 + 0.5) / SIZE as f64) / 4.0;
            let b0 = 1.0 - b1 - b2;
            let (u, v) = if linked {
                (2.0 * b1, 2.0 * b2)
            } else {
                (b0 * COLORS[0][0] + b1 * COLORS[1][0] + b2 * COLORS[2][0], b0 * COLORS[0][1] + b1 * COLORS[1][1] + b2 * COLORS[2][1])
            };
            let (fu, fv) = (u * TEX, v * TEX);
            if (fu - fu.round_ties_even()).abs().min((fv - fv.round_ties_even()).abs()) < 0.002 {
                skipped += 1;
                continue;
            }
            let clamp = |f: f64| (f.trunc() as i64).clamp(0, TEX as i64 - 1);
            let want = texel(clamp(fu), clamp(fv));
            for (c, &w) in want.iter().enumerate() {
                let got = pixels[(y * SIZE + x) * 4 + c];
                checked += 1;
                exact += usize::from(got == half_rtz(f64::from(w as f32))?);
                if (got - w).abs() > worst {
                    worst = (got - w).abs();
                    at = Some((x, y, c, got, w));
                }
            }
        }
    }
    let mut text = format!("composantes verifiees : {checked}, pixels ignores (frontiere de texel) : {skipped}{NL}");
    write!(text, "ecart maximal : {worst:.6} en {}{NL}", worst_at(at)).unwrap();
    write!(text, "egales au demi-flottant attendu : {exact} ({:.1} %){NL}", 100.0 * exact as f64 / checked.max(1) as f64).unwrap();
    let ok = worst < 2e-3 && checked > 0;
    write!(text, "RESULTAT : {}{NL}", if ok { "CONFORME" } else { "ECART" }).unwrap();
    Ok(Verdict { text, ok })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::{blit_image, texture_image};

    #[test]
    fn a_correct_blit_passes_and_a_wrong_one_does_not() {
        let good = blit(&blit_image([1.0; 4], false), None).unwrap();
        assert!(good.ok, "{}", good.text);
        let bad = blit(&blit_image([1.0, 1.0, 0.5, 1.0], false), None).unwrap();
        assert!(!bad.ok);
        let undrawn = blit(&blit_image([1.0; 4], true), None).unwrap();
        assert!(!undrawn.ok);
        assert!(blit(&[0; 16], None).is_err());
        let scaled = blit(&blit_image([0.5, 2.0, 1.0, 0.25], false), Some([0.5, 2.0, 1.0, 0.25])).unwrap();
        assert!(scaled.ok, "{}", scaled.text);
    }

    #[test]
    fn a_correct_texture_passes_in_both_modes() {
        for linked in [false, true] {
            let good = texture(&texture_image(linked, 0.0), linked).unwrap();
            assert!(good.ok, "{}", good.text);
            assert!(!texture(&texture_image(linked, 0.01), linked).unwrap().ok);
            assert!(!texture(&texture_image(!linked, 0.0), linked).unwrap().ok);
        }
    }
}
