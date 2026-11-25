//! `decant_shader_vertex_abi.py`: the input convention of vertex shaders that call an external
//! routine (`S_SWAPPC` first). The vector registers live at entry, from a liveness analysis on the
//! control flow graph rebuilt from the listing, are compared with the `ISG1` signature under the
//! hypothesis that register r, component c arrives in v(4 + 4r + c).

use std::collections::BTreeSet;
use std::fmt::Write;
use std::sync::LazyLock;

use regex::Regex;

use crate::NL;
use crate::py::{Counter, Py};

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap()
}

static LINE: LazyLock<Regex> = LazyLock::new(|| re(r"^(0x[0-9a-f]{8}): (\S+) ?(.*)"));
static VREG: LazyLock<Regex> = LazyLock::new(|| re(r"^v(\d+)(?:\.|$)"));
static ANYV: LazyLock<Regex> = LazyLock::new(|| re(r"\bv(\d+)\b"));
/// Operations that also read their destination: MAC, FMAC and PK_FMAC (D = S0 * S1 + D), DOT2C
/// and DOT4C (D = dot(S0, S1) + D). MADMK and MADAK do not read it (RDNA2 instruction set).
static READS_DST: LazyLock<Regex> = LazyLock::new(|| re(r"^V_(?:F?MAC_|PK_FMAC_|DOT\dC_)"));
static TARGET: LazyLock<Regex> = LazyLock::new(|| re(r"0x([0-9a-f]{8})"));
static WIDE: LazyLock<Regex> = LazyLock::new(|| re(r"DWORDX(\d+)"));
static FORMAT: LazyLock<Regex> = LazyLock::new(|| re(r"FORMAT_(X[YZW]*)"));
static DMASK: LazyLock<Regex> = LazyLock::new(|| re(r"dmask=0x([0-9a-f]+)"));

fn int(text: &str) -> i64 {
    text.parse().unwrap_or(i64::MAX)
}

/// `str.strip()`, whose whitespace also holds the separators U+001C to U+001F.
fn strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// The operands of an instruction: split on commas outside parentheses.
pub fn operands(text: &str) -> Vec<String> {
    let (mut out, mut depth, mut cur) = (Vec::new(), 0i64, String::new());
    for ch in text.chars() {
        if ch == '(' {
            depth += 1;
        }
        if ch == ')' {
            depth -= 1;
        }
        if ch == ',' && depth == 0 {
            out.push(strip(&cur).to_owned());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    if !strip(&cur).is_empty() {
        out.push(strip(&cur).to_owned());
    }
    out
}

/// The vector registers an instruction writes and reads.
pub fn decode(op: &str, rest: &str) -> (BTreeSet<i64>, BTreeSet<i64>) {
    let ops = operands(rest.split(" ;").next().unwrap_or(""));
    let up = op.to_uppercase();
    let regs: Vec<Vec<i64>> = ops.iter().map(|o| ANYV.captures_iter(o).map(|m| int(&m[1])).collect()).collect();
    let flat: BTreeSet<i64> = regs.iter().flatten().copied().collect();
    if up.starts_with("EXP") || up.contains("STORE") || ops.is_empty() {
        return (BTreeSet::new(), flat);
    }
    let Some(first) = VREG.captures(&ops[0]) else {
        // A scalar, mask or vcc destination: every v register is a source.
        return (BTreeSet::new(), flat);
    };
    let dst = int(&first[1]);
    let mut src: BTreeSet<i64> = regs[1..].iter().flatten().copied().collect();
    if READS_DST.is_match(&up) {
        src.insert(dst);
    }
    // Loads of several components write consecutive registers.
    let count = if let Some(m) = WIDE.captures(&up) {
        int(&m[1])
    } else if let Some(m) = FORMAT.captures(&up) {
        m[1].chars().count() as i64
    } else if let Some(m) = DMASK.captures(rest).filter(|_| up.starts_with("IMAGE")) {
        m[1].chars().map(|c| c.to_digit(16).unwrap_or(0).count_ones() as i64).sum()
    } else {
        1
    };
    ((dst..dst.saturating_add(count)).collect(), src)
}

/// The vector registers live at entry, or `None` when the listing does not start with `S_SWAPPC`
/// after its header lines.
pub fn live_in(text: &str) -> Option<BTreeSet<i64>> {
    let mut insts: Vec<(i64, &str, &str)> = crate::decoder::splitlines(text)
        .into_iter()
        .filter_map(|raw| LINE.captures(raw).map(|m| (i64::from_str_radix(&m[1][2..], 16).unwrap(), m.get(2).unwrap().as_str(), m.get(3).unwrap().as_str())))
        .collect();
    let swappc = |op: &str| op.to_uppercase().starts_with("S_SWAPPC");
    let mut skip = 0;
    while skip < insts.len() && matches!(insts[skip].1, "signature" | "S_INST_PREFETCH") && insts.len() - skip > 1 && !swappc(insts[skip].1) {
        skip += 1;
    }
    insts.drain(..skip);
    if insts.is_empty() || !swappc(insts[0].1) {
        return None;
    }
    let index: std::collections::HashMap<i64, usize> = insts.iter().enumerate().map(|(i, &(pc, _, _))| (pc, i)).collect();
    let (mut succ, mut defs, mut uses) = (Vec::new(), Vec::new(), Vec::new());
    for (i, &(_, op, rest)) in insts.iter().enumerate() {
        let up = op.to_uppercase();
        let target = if up.starts_with("S_CBRANCH") || up.starts_with("S_BRANCH") { TARGET.captures(rest) } else { None };
        let mut next = Vec::new();
        if !(up.starts_with("S_ENDPGM") || up.starts_with("S_BRANCH")) && i + 1 < insts.len() {
            next.push(i + 1);
        }
        if let Some(&t) = target.and_then(|m| index.get(&i64::from_str_radix(&m[1], 16).unwrap())) {
            next.push(t);
        }
        succ.push(next);
        let (d, u) = if i > 0 { decode(op, rest) } else { (BTreeSet::new(), BTreeSet::new()) };
        defs.push(d);
        uses.push(u);
    }
    let mut live: Vec<BTreeSet<i64>> = vec![BTreeSet::new(); insts.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..insts.len()).rev() {
            let mut new = uses[i].clone();
            for &s in &succ[i] {
                new.extend(live[s].iter().filter(|r| !defs[i].contains(r)));
            }
            if new != live[i] {
                live[i] = new;
                changed = true;
            }
        }
    }
    Some(if insts.len() > 1 { live.swap_remove(1) } else { BTreeSet::new() })
}

/// An `ISG1` element of a metadata line, `[semantic, index, register, mask, read, system, ...]`.
struct Input<'a> {
    raw: &'a serde_json::Value,
    register: i64,
    mask: i64,
    read: i64,
    system: i64,
}

fn inputs(list: &serde_json::Value) -> Result<Vec<Input<'_>>, String> {
    let list = list.as_array().ok_or("TypeError: inputs is not a list")?;
    list.iter()
        .map(|e| {
            let field = |k: usize| e.get(k).and_then(|v| v.as_i64()).ok_or_else(|| format!("TypeError: input field {k} of {e}"));
            Ok(Input { raw: e, register: field(2)?, mask: field(3)?, read: field(4)?, system: field(5)? })
        })
        .collect()
}

fn slots(elements: &[&Input], mask: impl Fn(&Input) -> i64) -> BTreeSet<i64> {
    elements.iter().flat_map(|e| (0..4).filter(|c| mask(e) >> c & 1 != 0).map(|c| 4 + 4 * e.register + c).collect::<Vec<_>>()).collect()
}

/// The counts and examples of the script.
#[derive(Default)]
pub struct Abi {
    stat: Counter<String>,
    system: Counter<Py>,
    bad: Vec<Py>,
}

impl Abi {
    pub fn without_swappc(&mut self) {
        self.stat.add("sans S_SWAPPC en tete".to_owned(), 1);
    }

    /// Adds one shader and returns its line for the optional output file, without line end.
    pub fn add(&mut self, name: &str, live: &BTreeSet<i64>, meta_inputs: &serde_json::Value) -> Result<String, String> {
        let all = inputs(meta_inputs)?;
        let normal: Vec<&Input> = all.iter().filter(|e| e.system == 0).collect();
        let special: Vec<&Input> = all.iter().filter(|e| e.system != 0).collect();
        let used = slots(&normal, |e| e.read);
        let declared = slots(&normal, |e| e.mask);
        let taken = slots(&special, |_| 0xF);
        let attrs: BTreeSet<i64> = live.iter().copied().filter(|&r| r >= 4).collect();
        let free: BTreeSet<i64> = attrs.difference(&taken).copied().collect();
        self.stat.add("total".to_owned(), 1);
        self.stat.add("A. vivants == composantes utilisees".to_owned(), i64::from(free == used));
        self.stat.add("B. vivants inclus dans les composantes declarees".to_owned(), i64::from(free.is_subset(&declared)));
        self.stat.add("C. composantes utilisees toutes vivantes".to_owned(), i64::from(used.is_subset(&attrs)));
        let mut elements = special
            .iter()
            .map(|e| match e.raw.get(0) {
                Some(serde_json::Value::String(s)) => Ok((s.clone(), e.register)),
                _ => Err("TypeError: semantic is not a string".to_owned()),
            })
            .collect::<Result<Vec<_>, String>>()?;
        elements.sort();
        let ints = |set: &mut dyn Iterator<Item = i64>| Py::Tuple(set.map(Py::Int).collect());
        let key = Py::Tuple(vec![
            Py::Tuple(elements.into_iter().map(|(s, r)| crate::tuple!(s, r)).collect()),
            ints(&mut live.iter().copied().filter(|r| (0..4).contains(r))),
            ints(&mut attrs.intersection(&taken).copied()),
        ]);
        self.system.add(key, 1);
        if !free.is_subset(&declared) && self.bad.len() < 6 {
            let outside: Vec<i64> = free.difference(&declared).copied().collect();
            self.bad.push(Py::Tuple(vec![name.into(), outside.into(), Py::from_json(meta_inputs)]));
        }
        let row = Py::Dict(vec![
            ("name".into(), name.into()),
            ("live".into(), live.iter().copied().collect::<Vec<_>>().into()),
            ("inputs".into(), Py::from_json(meta_inputs)),
        ]);
        Ok(row.json(false))
    }

    pub fn report(&self) -> String {
        let mut out = crate::py::sorted_lines(&self.stat, 5);
        write!(out, "valeurs systeme (elements, v0-v3 vivants, emplacements 4+4r vivants) :{NL}").unwrap();
        for (key, count) in self.system.most_common(Some(10)) {
            write!(out, "{count:5}  {}{NL}", key.repr()).unwrap();
        }
        for item in &self.bad {
            write!(out, "ECART {}{NL}", item.repr()).unwrap();
        }
        out
    }
}

/// The whole script: the shaders `rows` flags as linked, each listed by `listing`, compared with
/// the metadata lines `meta`. `line` receives each line of the optional output file. Returns the
/// printed report.
pub fn run(
    meta: &[serde_json::Value],
    rows: &[serde_json::Value],
    listing: &mut dyn FnMut(&str) -> Result<String, String>,
    line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<String, String> {
    let mut by_name = std::collections::HashMap::new();
    for m in meta {
        by_name.insert(m.get("name").and_then(|v| v.as_str()).ok_or("KeyError: 'name'")?, m);
    }
    let mut abi = Abi::default();
    for name in linked_names(rows)? {
        let Some(live) = live_in(&listing(&name)?) else {
            abi.without_swappc();
            continue;
        };
        let m = by_name.get(name.as_str()).ok_or_else(|| format!("KeyError: '{name}'"))?;
        line(&abi.add(&name, &live, m.get("inputs").ok_or("KeyError: 'inputs'")?)?)?;
    }
    Ok(abi.report())
}

/// The names of the shaders the batch results flag as needing a link: `T` is not `OK` and its
/// message holds `must be linked`.
pub fn linked_names(rows: &[serde_json::Value]) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for r in rows {
        let Some(t) = r.get("T") else { continue };
        let ok = t.get(0).ok_or("IndexError: T is empty")?;
        if ok == "OK" {
            continue;
        }
        let message = t.get(1).ok_or("IndexError: T has no message")?;
        let flagged = match message {
            serde_json::Value::String(s) => s.contains("must be linked"),
            serde_json::Value::Array(items) => items.iter().any(|v| v == "must be linked"),
            _ => return Err("TypeError: T message is not text".to_owned()),
        };
        if flagged {
            let name = r.get("name").and_then(|n| n.as_str()).ok_or("KeyError: 'name'")?;
            names.push(name.to_owned());
        }
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operands_keep_parentheses_together() {
        assert_eq!(operands(" v1, s[0:3] , offen(v2, v3),, "), ["v1", "s[0:3]", "offen(v2, v3)", ""]);
        assert!(operands("  ").is_empty());
    }

    #[test]
    fn decode_follows_the_script_rules() {
        let set = |v: &[i64]| v.iter().copied().collect::<BTreeSet<i64>>();
        assert_eq!(decode("V_ADD_F32", "v1, v2, v3"), (set(&[1]), set(&[2, 3])));
        assert_eq!(decode("V_FMAC_F32", "v1, v2, v3 ; comment v9"), (set(&[1]), set(&[1, 2, 3])));
        assert_eq!(decode("V_MADMK_F32", "v1, v2, 0x1, v3"), (set(&[1]), set(&[2, 3])));
        assert_eq!(decode("BUFFER_LOAD_DWORDX3", "v4, v0, s[0:3]"), (set(&[4, 5, 6]), set(&[0])));
        assert_eq!(decode("TBUFFER_LOAD_FORMAT_XY", "v8, v1"), (set(&[8, 9]), set(&[1])));
        assert_eq!(decode("IMAGE_SAMPLE", "v10, v[0:1], s[0:7] dmask=0xb"), (set(&[10, 11, 12]), set(&[])));
        assert_eq!(decode("BUFFER_STORE_DWORD", "v1, v2"), (set(&[]), set(&[1, 2])));
        assert_eq!(decode("exp", "pos0 v4, v5, v6, v7 done"), (set(&[]), set(&[4, 5, 6, 7])));
        assert_eq!(decode("V_CMP_GT_F32", "vcc, v1, v2"), (set(&[]), set(&[1, 2])));
        assert_eq!(decode("V_MOV_B32", "v3.x, v1"), (set(&[3]), set(&[1])));
        assert_eq!(decode("S_NOP", ""), (set(&[]), set(&[])));
    }

    #[test]
    fn liveness_crosses_branches() {
        let listing = "header line\n\
            0x00000000: signature 1\n\
            0x00000004: S_SWAPPC_B64 s[0:1], s[2:3]\n\
            0x00000008: V_MOV_B32 v1, v4\n\
            0x0000000c: S_CBRANCH_SCC0 0x00000018\n\
            0x00000010: V_ADD_F32 v1, v1, v5\n\
            0x00000014: S_BRANCH 0x0000001c\n\
            0x00000018: V_MUL_F32 v1, v1, v9\n\
            0x0000001c: exp pos0 v1, v1, v1, v1 done\n\
            0x00000020: S_ENDPGM\n";
        assert_eq!(live_in(listing).unwrap().into_iter().collect::<Vec<_>>(), [4, 5, 9]);
        assert_eq!(live_in("0x00000000: V_MOV_B32 v0, v1\n"), None);
        assert_eq!(live_in("0x00000000: S_SWAPPC_B64 s[0:1], s[2:3]\n"), Some(BTreeSet::new()));
        assert_eq!(live_in(""), None);
    }
}
