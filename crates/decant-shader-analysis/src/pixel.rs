//! `decant_shader_pixel_inputs.py`: how a pixel shader numbers its interpolated inputs. Hypothesis:
//! hardware attribute k (the operand of `V_INTERP`) is non system `ISG1` input register number k,
//! in order, counting only registers with at least one component read.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::LazyLock;

use regex::Regex;

use crate::NL;
use crate::py::{Counter, Py};

static INTERP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"V_INTERP_(?:P1|P2|MOV)_F\d+ v\d+, [^,]+, (\d+), (\d+)").unwrap());

/// The attributes and components the `V_INTERP` instructions of a listing read.
pub fn interpolated(listing: &str) -> BTreeMap<i64, BTreeSet<i64>> {
    let mut used: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for m in INTERP.captures_iter(listing) {
        let int = |s: &str| s.parse::<i64>().unwrap_or(i64::MAX);
        used.entry(int(&m[1])).or_default().insert(int(&m[2]));
    }
    used
}

#[derive(Default)]
pub struct Inputs {
    stat: Counter<String>,
    bad: Vec<Py>,
}

impl Inputs {
    /// Adds one shader: the attributes of its listing against the inputs of its metadata line.
    pub fn add(&mut self, name: &str, used: &BTreeMap<i64, BTreeSet<i64>>, inputs: &serde_json::Value) -> Result<(), String> {
        let list = inputs.as_array().ok_or("TypeError: inputs is not a list")?;
        let field = |e: &serde_json::Value, k: usize| e.get(k).and_then(|v| v.as_i64()).ok_or_else(|| format!("TypeError: input field {k} of {e}"));
        // Several elements may share a register (TEXCOORD2.xy and TEXCOORD3.zw): count per register.
        let (mut masks, mut reads) = (BTreeMap::<i64, i64>::new(), BTreeMap::<i64, i64>::new());
        for e in list {
            if field(e, 5)? != 0 {
                continue;
            }
            *masks.entry(field(e, 2)?).or_default() |= field(e, 3)?;
            *reads.entry(field(e, 2)?).or_default() |= field(e, 4)?;
        }
        let ordinary: Vec<i64> = masks.keys().copied().collect();
        let read: Vec<i64> = ordinary.iter().copied().filter(|r| reads[r] != 0).collect();
        let keys: BTreeSet<i64> = used.keys().copied().collect();
        let a = keys == (0..read.len() as i64).collect();
        let b = keys == (0..ordinary.len() as i64).collect();
        let comps = a && used.iter().all(|(&k, comps)| comps.iter().all(|&c| (0..4).contains(&c) && reads[&read[k as usize]] >> c & 1 != 0));
        self.stat.add("shaders".to_owned(), 1);
        self.stat.add("A. attributs = entrees lues, dans l ordre".to_owned(), i64::from(a));
        self.stat.add("A'. et composantes incluses dans le masque lu".to_owned(), i64::from(comps));
        self.stat.add("B. attributs = toutes les entrees declarees".to_owned(), i64::from(b));
        if !a && self.bad.len() < 5 {
            let elements = list
                .iter()
                .map(|e| Py::Tuple([0, 1, 2, 4, 5].iter().map(|&k| e.get(k).map_or(Py::None, Py::from_json)).collect()))
                .collect();
            self.bad.push(Py::Tuple(vec![name.into(), keys.into_iter().collect::<Vec<_>>().into(), Py::List(elements)]));
        }
        Ok(())
    }

    pub fn report(&self) -> String {
        let mut out = crate::py::sorted_lines(&self.stat, 5);
        for item in &self.bad {
            write!(out, "ECART {}{NL}", item.repr()).unwrap();
        }
        out
    }
}

/// The whole script: `count` pixel shaders of `meta` drawn with seed 31, each listed by
/// `listing`, names that several shaders share left out. Returns the printed report.
pub fn run(meta: &[serde_json::Value], count: i64, listing: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<String, String> {
    let name = |m: &serde_json::Value| m.get("name").and_then(|v| v.as_str()).map(str::to_owned).ok_or_else(|| "KeyError: 'name'".to_owned());
    let pixels: Vec<&serde_json::Value> = meta.iter().filter(|m| m.get("stage").is_some_and(|s| s == "pixel")).collect();
    let mut per_name: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &pixels {
        *per_name.entry(name(m)?).or_default() += 1;
    }
    let count = usize::try_from(count).map_err(|_| "ValueError: Sample larger than population or is negative")?;
    let chosen = crate::random::Random::seed(31).sample(&pixels, count).map_err(|e| format!("ValueError: {e}"))?;
    let mut check = Inputs::default();
    let mut seen = std::collections::HashSet::new();
    for m in chosen {
        let n = name(m)?;
        // The disassembler picks by name: one variant per name only.
        if seen.contains(&n) || per_name[&n] > 1 {
            continue;
        }
        seen.insert(n.clone());
        check.add(&n, &interpolated(&listing(&n)?), m.get("inputs").ok_or("KeyError: 'inputs'")?)?;
    }
    Ok(check.report())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_count_read_registers_in_order() {
        let listing = "0x00000000: V_INTERP_P1_F32 v2, v0, 0, 1\n\
                       0x00000004: V_INTERP_P2_F32 v2, v1, 0, 0\n\
                       0x00000008: V_INTERP_MOV_F32 v3, p0, 1, 2\n";
        let used = interpolated(listing);
        assert_eq!(used.len(), 2);
        let inputs: serde_json::Value = serde_json::json!([
            ["SV_Position", 0, 0, 15, 0, 1, 3, 0, 0],
            ["TEXCOORD", 0, 1, 3, 3, 0, 3, 0, 0],
            ["TEXCOORD", 1, 1, 12, 0, 0, 3, 0, 0],
            ["COLOR", 0, 2, 15, 0, 0, 3, 0, 0],
            ["NORMAL", 0, 3, 7, 4, 0, 3, 0, 0]
        ]);
        let mut check = Inputs::default();
        check.add("PS_A", &used, &inputs).unwrap();
        check.add("PS_B", &interpolated(""), &inputs).unwrap();
        assert_eq!(
            check.report(),
            [
                "    1  A'. et composantes incluses dans le masque lu",
                "    1  A. attributs = entrees lues, dans l ordre",
                "    0  B. attributs = toutes les entrees declarees",
                "    2  shaders",
                "ECART ('PS_B', [], [('SV_Position', 0, 0, 0, 1), ('TEXCOORD', 0, 1, 3, 0), ('TEXCOORD', 1, 1, 0, 0), ('COLOR', 0, 2, 0, 0), ('NORMAL', 0, 3, 4, 0)])",
            ]
            .map(|l| format!("{l}{NL}"))
            .concat()
        );
    }
}
