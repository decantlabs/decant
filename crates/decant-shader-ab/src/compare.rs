//! Comparison of what two runs wrote: per element, exact and within a tolerance, and the verdict of
//! a pair. The tool reports differences, it does not decide which side is right.

use serde::{Deserialize, Serialize};

use crate::input::f16_to_f32;

/// How the bytes of one component are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Elem {
    F32,
    U32,
    I32,
    F16,
    Unorm8,
    /// A dword of unknown type (raw and structured buffers): compared as a float when both values
    /// are normal floats or both NaN, else exactly. Small integers look like denormals and so stay
    /// exact.
    Auto,
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::F16 => 2,
            Elem::Unorm8 => 1,
            _ => 4,
        }
    }
}

/// One subresource or buffer: `width * height * depth` records of `record.len()` components.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sub {
    /// `mip 0 slice 0`, empty for buffers.
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub record: Vec<Elem>,
    pub bytes: Vec<u8>,
}

/// Everything one writable resource held after the run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Output {
    /// `SV_Target0 rgba32f`, `u0 space0 RawBuffer`, `stream output`.
    pub name: String,
    /// Texture coordinates `(x, y, z)` when true, record indices otherwise.
    pub texture: bool,
    pub subs: Vec<Sub>,
}

/// What counts as equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tolerance {
    /// Largest distance in units in the last place for floats (and in steps for UNORM8).
    pub max_ulp: u32,
    /// Two NaNs count as equal whatever their payload.
    pub nan_equal: bool,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance { max_ulp: 0, nan_equal: true }
    }
}

/// How two components relate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rel {
    Same,
    /// Different bits, within the tolerance, this many ULP apart.
    Near(u64),
    Far,
}

fn ordered(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 { -i64::from(bits & 0x7FFF_FFFF) } else { i64::from(bits) }
}

fn ordered16(bits: u16) -> i64 {
    if bits & 0x8000 != 0 { -i64::from(bits & 0x7FFF) } else { i64::from(bits) }
}

fn relate(elem: Elem, a: &[u8], b: &[u8], tol: Tolerance) -> Rel {
    if a == b {
        return Rel::Same;
    }
    let near = |d: u64| if d <= u64::from(tol.max_ulp) { Rel::Near(d) } else { Rel::Far };
    match elem {
        Elem::U32 | Elem::I32 => Rel::Far,
        Elem::Unorm8 => near(u64::from(a[0].abs_diff(b[0]))),
        Elem::F16 => {
            let (x, y) = (u16::from_le_bytes([a[0], a[1]]), u16::from_le_bytes([b[0], b[1]]));
            let (fx, fy) = (f16_to_f32(x), f16_to_f32(y));
            if fx.is_nan() || fy.is_nan() {
                return if fx.is_nan() && fy.is_nan() && tol.nan_equal { Rel::Near(0) } else { Rel::Far };
            }
            near(ordered16(x).abs_diff(ordered16(y)))
        }
        Elem::F32 | Elem::Auto => {
            let (x, y) = (u32::from_le_bytes(a.try_into().unwrap()), u32::from_le_bytes(b.try_into().unwrap()));
            let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
            if elem == Elem::Auto && !(fx.is_normal() && fy.is_normal()) && !(fx.is_nan() && fy.is_nan()) {
                return Rel::Far;
            }
            if fx.is_nan() || fy.is_nan() {
                return if fx.is_nan() && fy.is_nan() && tol.nan_equal { Rel::Near(0) } else { Rel::Far };
            }
            near(ordered(x).abs_diff(ordered(y)))
        }
    }
}

/// One component written out: the float and its bits, or the integer.
pub fn show(elem: Elem, bytes: &[u8]) -> String {
    match elem {
        Elem::F32 | Elem::Auto => {
            let w = u32::from_le_bytes(bytes.try_into().unwrap());
            format!("{:?} (0x{w:08x})", f32::from_bits(w))
        }
        Elem::U32 => format!("{}", u32::from_le_bytes(bytes.try_into().unwrap())),
        Elem::I32 => format!("{}", i32::from_le_bytes(bytes.try_into().unwrap())),
        Elem::F16 => {
            let h = u16::from_le_bytes([bytes[0], bytes[1]]);
            format!("{:?} (0x{h:04x})", f16_to_f32(h))
        }
        Elem::Unorm8 => format!("{}", bytes[0]),
    }
}

/// The first component two outputs disagree on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstDiff {
    pub output: String,
    /// `(12, 30) component 2 mip 0 slice 0` or `element 7 component 0`.
    pub at: String,
    pub a: String,
    pub b: String,
}

/// The comparison of all outputs of a pair that both ran.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Diff {
    /// Components compared.
    pub compared: u64,
    /// Components whose bits differ.
    pub inexact: u64,
    /// Components outside the tolerance.
    pub different: u64,
    /// Largest ULP distance among the components within the tolerance.
    pub max_ulp: u64,
    pub first: Option<FirstDiff>,
    /// Structural problems: an output on one side only, a size that does not match.
    pub mismatched: Vec<String>,
}

/// Compares every output of A with the output of the same name of B.
pub fn compare(a: &[Output], b: &[Output], tol: Tolerance) -> Diff {
    let mut d = Diff::default();
    for oa in a {
        let Some(ob) = b.iter().find(|o| o.name == oa.name) else {
            d.mismatched.push(format!("{} only in A", oa.name));
            continue;
        };
        if oa.subs.len() != ob.subs.len() {
            d.mismatched.push(format!("{}: {} subresources in A, {} in B", oa.name, oa.subs.len(), ob.subs.len()));
            continue;
        }
        for (sa, sb) in oa.subs.iter().zip(&ob.subs) {
            if sa.record != sb.record || sa.bytes.len() != sb.bytes.len() || (sa.width, sa.height, sa.depth) != (sb.width, sb.height, sb.depth) {
                d.mismatched.push(format!("{} {}: layouts differ", oa.name, sa.label));
                continue;
            }
            compare_sub(&oa.name, oa.texture, sa, sb, tol, &mut d);
        }
    }
    for ob in b {
        if !a.iter().any(|o| o.name == ob.name) {
            d.mismatched.push(format!("{} only in B", ob.name));
        }
    }
    d
}

fn compare_sub(name: &str, texture: bool, sa: &Sub, sb: &Sub, tol: Tolerance, d: &mut Diff) {
    let stride: usize = sa.record.iter().map(|e| e.size()).sum();
    if stride == 0 {
        return;
    }
    let records = sa.bytes.len() / stride;
    for r in 0..records {
        let mut at = r * stride;
        for (c, &elem) in sa.record.iter().enumerate() {
            let n = elem.size();
            let (x, y) = (&sa.bytes[at..at + n], &sb.bytes[at..at + n]);
            d.compared += 1;
            match relate(elem, x, y, tol) {
                Rel::Same => {}
                Rel::Near(u) => {
                    if x != y {
                        d.inexact += 1;
                    }
                    d.max_ulp = d.max_ulp.max(u);
                }
                Rel::Far => {
                    d.inexact += 1;
                    d.different += 1;
                    if d.first.is_none() {
                        let place = if texture {
                            let w = sa.width.max(1) as usize;
                            let h = sa.height.max(1) as usize;
                            let (px, py, pz) = (r % w, (r / w) % h, r / (w * h));
                            let xyz = if sa.depth > 1 { format!("({px}, {py}, {pz})") } else { format!("({px}, {py})") };
                            format!("{xyz} component {c} {}", sa.label)
                        } else {
                            format!("element {r} component {c} (byte {at})")
                        };
                        d.first = Some(FirstDiff { output: name.to_string(), at: place.trim_end().to_string(), a: show(elem, x), b: show(elem, y) });
                    }
                }
            }
            at += n;
        }
    }
}

/// The verdict on one pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Identical,
    WithinTolerance,
    Different,
    AFailed,
    BFailed,
    BothFailed,
    /// Not run: a stage or a feature the runner does not support, or a broken input.
    Skipped,
    /// Both ran, but neither wrote anything the runner reads back (for example a shader that only
    /// writes through descriptor heap indices the runner fills with null descriptors).
    NoOutput,
}

impl Verdict {
    pub fn word(&self) -> &'static str {
        match self {
            Verdict::Identical => "identical",
            Verdict::WithinTolerance => "within_tolerance",
            Verdict::Different => "different",
            Verdict::AFailed => "a_failed",
            Verdict::BFailed => "b_failed",
            Verdict::BothFailed => "both_failed",
            Verdict::Skipped => "skipped",
            Verdict::NoOutput => "no_output",
        }
    }

    pub const ALL: [Verdict; 8] = [Verdict::Identical, Verdict::WithinTolerance, Verdict::Different, Verdict::AFailed, Verdict::BFailed, Verdict::BothFailed, Verdict::NoOutput, Verdict::Skipped];
}

/// The verdict from the two sides: each either its outputs or why it failed.
pub fn judge(a: &Result<Vec<Output>, String>, b: &Result<Vec<Output>, String>, tol: Tolerance) -> (Verdict, Option<Diff>) {
    match (a, b) {
        (Err(_), Err(_)) => (Verdict::BothFailed, None),
        (Err(_), Ok(_)) => (Verdict::AFailed, None),
        (Ok(_), Err(_)) => (Verdict::BFailed, None),
        (Ok(x), Ok(y)) => {
            let d = compare(x, y, tol);
            let v = if d.different > 0 || !d.mismatched.is_empty() {
                Verdict::Different
            } else if d.compared == 0 {
                Verdict::NoOutput
            } else if d.inexact > 0 {
                Verdict::WithinTolerance
            } else {
                Verdict::Identical
            };
            (v, Some(d))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floats(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn target(v: &[f32]) -> Vec<Output> {
        vec![Output { name: "SV_Target0".into(), texture: true, subs: vec![Sub { label: "mip 0 slice 0".into(), width: 2, height: (v.len() / 8) as u32, depth: 1, record: vec![Elem::F32; 4], bytes: floats(v) }] }]
    }

    #[test]
    fn equal_outputs_are_identical() {
        let a = target(&[0.5; 16]);
        assert_eq!(judge(&Ok(a.clone()), &Ok(a), Tolerance::default()).0, Verdict::Identical);
    }

    #[test]
    fn one_ulp_is_within_a_tolerance_of_one_and_different_without() {
        let mut v = [0.25f32; 16];
        let a = target(&v);
        v[13] = f32::from_bits(0.25f32.to_bits() + 1);
        let b = target(&v);
        let (verdict, d) = judge(&Ok(a.clone()), &Ok(b.clone()), Tolerance { max_ulp: 1, nan_equal: true });
        assert_eq!(verdict, Verdict::WithinTolerance);
        assert_eq!(d.unwrap().max_ulp, 1);
        let (verdict, d) = judge(&Ok(a), &Ok(b), Tolerance::default());
        assert_eq!(verdict, Verdict::Different);
        let first = d.unwrap().first.unwrap();
        // Record 3 of a 2 wide target is (1, 1), component 1.
        assert_eq!(first.at, "(1, 1) component 1 mip 0 slice 0");
        assert!(first.a.starts_with("0.25 "), "{}", first.a);
    }

    #[test]
    fn zero_and_negative_zero_are_zero_ulp_apart() {
        let (v, d) = judge(&Ok(target(&[0.0; 8])), &Ok(target(&[-0.0; 8])), Tolerance::default());
        assert_eq!(v, Verdict::WithinTolerance);
        assert_eq!(d.unwrap().max_ulp, 0);
    }

    #[test]
    fn nans_are_equal_only_when_asked_and_never_against_a_number() {
        let a = target(&[f32::NAN; 8]);
        let b = target(&[f32::from_bits(0x7FC0_0001); 8]);
        assert_eq!(judge(&Ok(a.clone()), &Ok(b.clone()), Tolerance { max_ulp: 0, nan_equal: true }).0, Verdict::WithinTolerance);
        assert_eq!(judge(&Ok(a.clone()), &Ok(b), Tolerance { max_ulp: 0, nan_equal: false }).0, Verdict::Different);
        let c = target(&[1.0; 8]);
        assert_eq!(judge(&Ok(a), &Ok(c), Tolerance { max_ulp: u32::MAX, nan_equal: true }).0, Verdict::Different);
    }

    #[test]
    fn auto_dwords_compare_small_integers_exactly() {
        let sub = |w: u32| vec![Output { name: "u0".into(), texture: false, subs: vec![Sub { label: String::new(), width: 1, height: 1, depth: 1, record: vec![Elem::Auto], bytes: w.to_le_bytes().to_vec() }] }];
        let tol = Tolerance { max_ulp: 4, nan_equal: true };
        assert_eq!(judge(&Ok(sub(5)), &Ok(sub(6)), tol).0, Verdict::Different);
        assert_eq!(judge(&Ok(sub(1.0f32.to_bits())), &Ok(sub(1.0f32.to_bits() + 2)), tol).0, Verdict::WithinTolerance);
        let d = judge(&Ok(sub(5)), &Ok(sub(6)), tol).1.unwrap();
        assert_eq!(d.first.unwrap().at, "element 0 component 0 (byte 0)");
    }

    #[test]
    fn failures_and_missing_outputs_are_reported() {
        let a = target(&[1.0; 8]);
        assert_eq!(judge(&Err("x".into()), &Ok(a.clone()), Tolerance::default()).0, Verdict::AFailed);
        assert_eq!(judge(&Ok(a.clone()), &Err("x".into()), Tolerance::default()).0, Verdict::BFailed);
        assert_eq!(judge(&Err("x".into()), &Err("y".into()), Tolerance::default()).0, Verdict::BothFailed);
        assert_eq!(judge(&Ok(vec![]), &Ok(vec![]), Tolerance::default()).0, Verdict::NoOutput);
        let (v, d) = judge(&Ok(a), &Ok(vec![]), Tolerance::default());
        assert_eq!(v, Verdict::Different);
        assert_eq!(d.unwrap().mismatched, vec!["SV_Target0 only in A".to_string()]);
    }
}
