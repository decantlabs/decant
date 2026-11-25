//! What these tools need of Python beyond `decant_shader_analysis::py`: the iteration order of a set of
//! integers, `repr` of a float, `struct.pack("<e")` and `json.dumps(indent=1)`.

use decant_shader_analysis::py::Py;

/// A CPython `set` of non negative integers, iterated in CPython's table order. The order decides
/// which of two equal counts `Counter.most_common` lists first, so it is reproduced exactly: an
/// open addressing table of eight slots that grows fourfold once it is three fifths full, nine
/// linear probes, then the perturbed jump of `setobject.c`.
#[derive(Clone, Debug)]
pub struct IntSet {
    table: Vec<Option<u64>>,
    used: usize,
}

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

impl Default for IntSet {
    fn default() -> IntSet {
        IntSet { table: vec![None; 8], used: 0 }
    }
}

/// The free slot `set_insert_clean` and `set_add_entry` choose for `key`, or the slot holding it.
fn slot(table: &[Option<u64>], key: u64) -> usize {
    let mask = table.len() - 1;
    let mut perturb = key as usize;
    let mut i = key as usize & mask;
    loop {
        let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
        for (j, entry) in table.iter().enumerate().skip(i).take(probes + 1) {
            match entry {
                None => return j,
                Some(k) if *k == key => return j,
                Some(_) => {}
            }
        }
        perturb >>= PERTURB_SHIFT;
        i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
    }
}

impl IntSet {
    pub fn insert(&mut self, key: u64) {
        let i = slot(&self.table, key);
        if self.table[i].is_some() {
            return;
        }
        self.table[i] = Some(key);
        self.used += 1;
        let mask = self.table.len() - 1;
        if self.used * 5 < mask * 3 {
            return;
        }
        let minused = if self.used > 50_000 { self.used * 2 } else { self.used * 4 };
        let mut size = 8;
        while size <= minused {
            size <<= 1;
        }
        let mut table = vec![None; size];
        for key in self.table.iter().flatten() {
            let j = slot(&table, *key);
            table[j] = Some(*key);
        }
        self.table = table;
    }

    /// The keys in the order `for key in set` gives them.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.table.iter().flatten().copied()
    }
}

/// `repr(x)` for a float: the shortest digits that read back, in fixed notation when the decimal
/// exponent is between -4 and 16, else in exponent notation with a sign and two digits at least.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let point = exp + 1;
    let body = if (-3..=16).contains(&point) {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        }
    } else {
        let rest = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        format!("{}{rest}e{}{:02}", &digits[..1], if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    format!("{sign}{body}")
}

/// `struct.pack("<e", x)` as bits: round to the nearest half float, ties to even. Values too large
/// raise `OverflowError` in Python and are an error here.
pub fn half_bits(x: f64) -> Result<u16, String> {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return Ok(sign | 0x7E00);
    }
    if a.is_infinite() {
        return Ok(sign | 0x7C00);
    }
    // In units of the smallest subnormal (2^-24), rounded to nearest even.
    let scaled = a * 16_777_216.0;
    let bits = if scaled < 1024.0 {
        scaled.round_ties_even() as u16
    } else {
        let e = a.log2().floor() as i32;
        let mut e = e.clamp(-14, 15);
        // Correct a log2 off by one at the edges.
        while a >= 2f64.powi(e + 1) {
            e += 1;
        }
        while a < 2f64.powi(e) {
            e -= 1;
        }
        let m = (a / 2f64.powi(e - 10)).round_ties_even() as u32;
        let (m, e) = if m == 2048 { (1024, e + 1) } else { (m, e) };
        if e > 15 {
            return Err("OverflowError: float too large to pack with e format".to_owned());
        }
        (((e + 15) as u16) << 10) | (m as u16 & 0x3FF)
    };
    Ok(sign | bits)
}

/// The value of half float bits.
pub fn half_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((bits >> 10) & 0x1F);
    let m = f64::from(bits & 0x3FF);
    match e {
        0 => sign * m * 2f64.powi(-24),
        31 if m == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// The half float nearest `value` toward zero: rounded to nearest, then one step down in magnitude
/// when that went past it (`to_half_rtz` of `check_blit.py` and `check_texture.py`).
pub fn half_rtz(value: f64) -> Result<f64, String> {
    let bits = half_bits(value)?;
    let half = half_value(bits);
    if half.abs() > value.abs() { Ok(half_value(bits.wrapping_sub(1))) } else { Ok(half) }
}

/// `json.dumps(value, indent=1)`: one item per line, one space per level, `,` after each item and
/// `": "` between key and value, empty lists and dictionaries kept on one line.
pub fn json_indent(value: &Py) -> String {
    let mut out = String::new();
    write_indent(value, 0, &mut out);
    out
}

fn write_indent(value: &Py, level: usize, out: &mut String) {
    let pad = |n: usize| format!("\n{}", " ".repeat(n));
    match value {
        Py::List(items) | Py::Tuple(items) if !items.is_empty() => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&pad(level + 1));
                write_indent(v, level + 1, out);
            }
            out.push_str(&pad(level));
            out.push(']');
        }
        Py::Dict(pairs) if !pairs.is_empty() => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&pad(level + 1));
                match k {
                    Py::Str(_) => out.push_str(&k.json(false)),
                    other => out.push_str(&Py::Str(other.repr()).json(false)),
                }
                out.push_str(": ");
                write_indent(v, level + 1, out);
            }
            out.push_str(&pad(level));
            out.push('}');
        }
        other => out.push_str(&other.json(false)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(keys: &[u64]) -> Vec<u64> {
        let mut s = IntSet::default();
        for &k in keys {
            s.insert(k);
        }
        s.iter().collect()
    }

    /// The expected orders were printed by CPython 3.13 for the same insertions.
    #[test]
    fn set_order_matches_cpython() {
        assert_eq!(order(&[1, 5347, 4427, 5301, 17, 33, 9, 2, 6, 11, 5345, 64, 4423, 5302, 3, 4]), [
            64, 1, 33, 5347, 2, 5345, 6, 4423, 3, 9, 4, 4427, 11, 17, 5301, 5302
        ]);
        assert_eq!(order(&[24, 8, 0, 16, 1, 41, 7, 15]), [0, 1, 7, 8, 41, 15, 16, 24]);
        let down: Vec<u64> = (1..=100).rev().step_by(7).collect();
        assert_eq!(order(&down), [65, 2, 100, 37, 72, 9, 44, 79, 16, 51, 86, 23, 58, 93, 30]);
        assert_eq!(order(&[3, 3, 1]), [1, 3]);
    }

    #[test]
    fn floats_print_like_python() {
        let cases = [
            (0.25, "0.25", "0.2", "0.250000"),
            (0.35, "0.35", "0.3", "0.350000"),
            (2.5, "2.5", "2.5", "2.500000"),
            (1e-05, "1e-05", "0.0", "0.000010"),
            (1.5e16, "1.5e+16", "15000000000000000.0", "15000000000000000.000000"),
            (1e16, "1e+16", "10000000000000000.0", "10000000000000000.000000"),
            (0.1, "0.1", "0.1", "0.100000"),
            (1.0 / 3.0, "0.3333333333333333", "0.3", "0.333333"),
            (-0.0, "-0.0", "-0.0", "-0.000000"),
            (65.0, "65.0", "65.0", "65.000000"),
            (0.000123, "0.000123", "0.0", "0.000123"),
        ];
        for (x, repr, one, six) in cases {
            assert_eq!(float_repr(x), repr);
            assert_eq!(format!("{x:.1}"), one, "{x}");
            assert_eq!(format!("{x:.6}"), six, "{x}");
        }
        assert_eq!(float_repr(123_456_789_012_345_680.0), "1.2345678901234568e+17");
    }

    #[test]
    fn half_floats_round_like_struct() {
        assert_eq!(half_bits(1.0).unwrap(), 0x3C00);
        assert_eq!(half_bits(65504.0).unwrap(), 0x7BFF);
        assert!(half_bits(65520.0).is_err());
        assert_eq!(half_bits(65519.0).unwrap(), 0x7BFF);
        assert_eq!(half_bits(2f64.powi(-25)).unwrap(), 0, "a tie rounds to the even zero");
        assert_eq!(half_bits(3.0 * 2f64.powi(-25)).unwrap(), 2);
        assert_eq!(half_bits(-2.0).unwrap(), 0xC000);
        // 1 + 1/2048 is halfway between two halves and rounds to the even one, 1.0.
        assert_eq!(half_bits(1.0 + 1.0 / 2048.0).unwrap(), 0x3C00);
        assert_eq!(half_rtz(0.1).unwrap(), 0.0999755859375);
        assert_eq!(half_rtz(0.5).unwrap(), 0.5);
        assert_eq!(half_rtz(-0.1).unwrap(), -0.0999755859375);
    }

    #[test]
    fn indented_json_matches_python() {
        let v = Py::Dict(vec![
            ("a".into(), Py::List(vec![Py::Int(1), Py::Bool(true)])),
            ("b".into(), Py::List(vec![])),
            ("c".into(), Py::Dict(vec![("x".into(), Py::None)])),
        ]);
        assert_eq!(json_indent(&v), "{\n \"a\": [\n  1,\n  true\n ],\n \"b\": [],\n \"c\": {\n  \"x\": null\n }\n}");
    }
}
