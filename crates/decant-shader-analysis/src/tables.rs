//! The tables placed after the code in an FSSM block (`fssm_tables.py`) and the binding records
//! of its table B (`fssm_bindings.py`).
//!
//! Header fields: 0xE8 and 0xF0 point to tables A and B, 0xF8 and 0xFC count their entries, 0x120
//! points to a name. Positions count from byte 16 of the block.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use regex::Regex;

use crate::NL;
use crate::dxbc::{u32_at, u64_at};
use crate::py::Py;

/// The dump of `fssm_tables.py` for one shader, appended to `out` line by line so that a bound
/// that raises in Python leaves the same lines printed before it.
pub fn dump(out: &mut String, fs: &[u8], n: i64, name: &str, bindings: &Py) -> Result<(), String> {
    let (a, b) = (u64_at(fs, 0xE8)?, u64_at(fs, 0xF0)?);
    let (ca, cb) = (u32_at(fs, 0xF8)?, u32_at(fs, 0xFC)?);
    let name_at = u64_at(fs, 0x120)?;
    write!(out, "== {n} {name}  liaisons PSV0 : {}{NL}", bindings.repr()).unwrap();
    write!(out, "   table A a {a:#x} ({ca} entree(s)), table B a {b:#x} ({cb} entree(s)), nom a {name_at:#x}{NL}").unwrap();
    let b = if b > fs.len() as u64 { name_at } else { b };
    for (label, start, end) in [("A", a, b), ("B", b, name_at)] {
        let (start, end) = (u128::from(start) + 16, u128::from(end) + 16);
        let count = end.saturating_sub(start) / 4;
        if start + 4 * count > fs.len() as u128 {
            return Err(format!("struct.error: unpack_from requires a buffer of at least {} bytes", start + 4 * count));
        }
        let words: Vec<String> = (0..count as usize).map(|i| format!("{:x}", u32_at(fs, start as usize + 4 * i).unwrap())).collect();
        write!(out, "   {label}: {}{NL}", words.join(" ")).unwrap();
    }
    Ok(())
}

/// `[(r[0], r[1]) for r in meta["bindings"]]` from a metadata line.
pub fn binding_pairs(meta: &serde_json::Value) -> Result<Py, String> {
    let list = meta.get("bindings").and_then(|b| b.as_array()).ok_or("KeyError: 'bindings'")?;
    let pairs = list
        .iter()
        .map(|r| match (r.get(0), r.get(1)) {
            (Some(a), Some(b)) => Ok(Py::Tuple(vec![Py::from_json(a), Py::from_json(b)])),
            _ => Err("IndexError: list index out of range".to_owned()),
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Py::List(pairs))
}

/// One record of table B: `[kind, offset in 32 bit words, register, extra, flags]`, with
/// `flags = class << 16 | table SGPR << 8 | low byte`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub kind: u32,
    pub letter: String,
    pub offset: u32,
    pub register: u32,
    pub extra: u32,
    pub class: u32,
    pub sgpr: u32,
    pub dwords: u32,
}

/// The register letter of a record kind, `?<hex>` when it is not identified.
pub fn letter(kind: u32) -> String {
    match kind {
        0x300 => "t".to_owned(),
        0x302 => "b".to_owned(),
        0x401 => "s".to_owned(),
        other => format!("?{other:x}"),
    }
}

/// `records(fs)`: no records when table B points past the block.
pub fn records(fs: &[u8]) -> Result<Vec<Record>, String> {
    let start = u64_at(fs, 0xF0)?;
    let count = u32_at(fs, 0xFC)?;
    if start > fs.len() as u64 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = start as usize + 16 + 20 * i;
        let w = |k: usize| u32_at(fs, at + 4 * k);
        let (kind, offset, register, extra, flags) = (w(0)?, w(1)?, w(2)?, w(3)?, w(4)?);
        out.push(Record { kind, letter: letter(kind), offset, register, extra, class: flags >> 16, sgpr: (flags >> 8) & 0xFF, dwords: flags & 0xFF });
    }
    Ok(out)
}

/// The decoded table line of one record.
pub fn record_line(r: &Record) -> String {
    format!(
        "   {}{:<4} table s{:<2} octet {:#06x}  classe {}  bas {}  extra {:#x}",
        r.letter,
        r.register,
        r.sgpr,
        u64::from(r.offset) * 4,
        r.class,
        r.dwords,
        r.extra
    )
}

/// The scalar table loads of a decoder listing, as `(table register, byte offset)`, with copies
/// of the input registers s8 to s15 followed back to their source.
pub fn table_loads(listing: &str) -> BTreeSet<(i64, i64)> {
    static LOAD: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"S_LOAD_DWORDX?(\d*) s\d+, s(\d+), null ; offset=(\d+) ").unwrap());
    static MOVE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| Regex::new(r"^S_MOV_B32 s(\d+), s(\d+)$").unwrap());
    let int = |s: &str| s.parse::<i64>().unwrap_or(i64::MAX);
    let mut alias: HashMap<i64, i64> = HashMap::new();
    let mut loads = BTreeSet::new();
    for line in crate::decoder::splitlines(listing) {
        let body = line.split_once(": ").map_or(line, |(_, rest)| rest);
        if let Some(m) = MOVE.captures(body) {
            let (dst, src) = (int(&m[1]), int(&m[2]));
            if (8..16).contains(&src) && !alias.values().any(|&v| v == dst) {
                alias.insert(dst, src);
            }
        }
        if let Some(m) = LOAD.captures(body) {
            let reg = int(&m[2]);
            loads.insert((alias.get(&reg).copied().unwrap_or(reg), int(&m[3])));
        }
    }
    loads
}

/// The counts of `fssm_bindings.py --verifie`.
#[derive(Default)]
pub struct Check {
    pub stat: crate::py::Counter<String>,
    pub letters: crate::py::Counter<String>,
}

impl Check {
    /// Adds one shader: its `t` and `b` records against the loads of its listing.
    pub fn add(&mut self, recs: &[Record], loads: &BTreeSet<(i64, i64)>) {
        self.stat.add("shaders".to_owned(), 1);
        for r in recs {
            self.letters.add(r.letter.clone(), 1);
            self.stat.add("entrees de table".to_owned(), 1);
            for scale in [1i64, 4] {
                let hit = loads.contains(&(i64::from(r.sgpr), i64::from(r.offset) * scale));
                self.stat.add(format!("classe {:2} table s{:<2} : retrouvees avec decalage x{scale}", r.class, r.sgpr), i64::from(hit));
            }
            self.stat.add(format!("classe {:2} table s{:<2} : entrees", r.class, r.sgpr), 1);
        }
    }

    pub fn report(&self) -> String {
        let letters = Py::Dict(self.letters.items().iter().map(|(k, v)| (Py::from(k.clone()), Py::Int(*v))).collect());
        format!("{}genres : {}{NL}", crate::py::sorted_lines(&self.stat, 6), letters.repr())
    }
}

/// `--verifie`: `count` shaders drawn with seed 41 among those whose name is unique, their `t`
/// and `b` records from `records_of(index)` against the loads of `listing(name)`. Returns the
/// printed report.
pub fn verify(
    names: &[&str],
    count: i64,
    records_of: &dyn Fn(usize) -> Result<Vec<Record>, String>,
    listing: &mut dyn FnMut(&str) -> Result<String, String>,
) -> Result<String, String> {
    let mut per_name: crate::py::Counter<&str> = crate::py::Counter::default();
    for name in names {
        per_name.add(name, 1);
    }
    let candidates: Vec<usize> = (0..names.len()).filter(|&n| per_name.get(&names[n]) == 1).collect();
    let count = usize::try_from(count).map_err(|_| "ValueError: Sample larger than population or is negative")?;
    let chosen = crate::random::Random::seed(41).sample(&candidates, count).map_err(|e| format!("ValueError: {e}"))?;
    let mut check = Check::default();
    for n in chosen {
        let recs: Vec<Record> = records_of(n)?.into_iter().filter(|r| r.letter == "t" || r.letter == "b").collect();
        if recs.is_empty() {
            continue;
        }
        check.add(&recs, &table_loads(&listing(names[n])?));
    }
    Ok(check.report())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::Container;

    #[test]
    fn dump_prints_both_tables() {
        let mut c = Container::new(0);
        c.table_b = Some(vec![[0x300, 4, 0, 0, 0x0002_0804]]);
        let fs = c.fssm();
        let mut out = String::new();
        let pairs = Py::List(vec![crate::tuple!("t", 0u32)]);
        dump(&mut out, &fs, 3, "PS_X", &pairs).unwrap();
        let expected = [
            "== 3 PS_X  liaisons PSV0 : [('t', 0)]",
            "   table A a 0x15c (2 entree(s)), table B a 0x164 (1 entree(s)), nom a 0x178",
            "   A: 11 22",
            "   B: 300 4 0 0 20804",
        ];
        assert_eq!(out, expected.map(|l| format!("{l}{NL}")).concat());
        // Without table B, its words run from the name position to itself: an empty line.
        c.table_b = None;
        let mut out = String::new();
        dump(&mut out, &c.fssm(), 0, "PS_Y", &Py::List(vec![])).unwrap();
        assert!(out.ends_with(&format!("   A: 11 22{NL}   B: {NL}")), "{out}");
        assert!(out.contains("table B a 0xdeadbeefdeadbeef (0 entree(s))"));
    }

    #[test]
    fn a_table_past_the_block_raises_after_the_header_lines() {
        let mut fs = Container::new(0).fssm();
        fs[0xE8..0xF0].copy_from_slice(&0x1000u64.to_le_bytes());
        fs[0x120..0x128].copy_from_slice(&0x2000u64.to_le_bytes());
        let mut out = String::new();
        assert!(dump(&mut out, &fs, 1, "PS", &Py::List(vec![])).is_err());
        assert_eq!(out.matches(NL).count(), 2);
    }

    #[test]
    fn records_decode_flags() {
        let mut c = Container::new(0);
        c.table_b = Some(vec![[0x302, 3, 1, 9, 0x0001_0A08], [0x123, 0, 0, 0, 0]]);
        let recs = records(&c.fssm()).unwrap();
        assert_eq!(record_line(&recs[0]), "   b1    table s10 octet 0x000c  classe 1  bas 8  extra 0x9");
        assert_eq!(record_line(&recs[1]), "   ?1230    table s0  octet 0x0000  classe 0  bas 0  extra 0x0");
        c.table_b = None;
        assert!(records(&c.fssm()).unwrap().is_empty());
    }

    #[test]
    fn loads_follow_input_register_copies() {
        let listing = "0x00000000: S_MOV_B32 s20, s9\n\
                       0x00000004: S_MOV_B32 s21, s20\n\
                       0x00000008: S_LOAD_DWORDX4 s0, s20, null ; offset=16 \n\
                       0x0000000c: S_LOAD_DWORD s4, s21, null ; offset=8 \n\
                       0x00000010: S_LOAD_DWORDX8 s[4:11], s2, null ; offset=16 \n\
                       0x00000014: S_LOAD_DWORDX8 s4, s2, null ; offset=0x10 \n";
        let loads = table_loads(listing);
        assert_eq!(loads.into_iter().collect::<Vec<_>>(), [(9, 16), (21, 8)]);
        let mut check = Check::default();
        let rec = |offset, sgpr| Record { kind: 0x300, letter: "t".into(), offset, register: 0, extra: 0, class: 2, sgpr, dwords: 4 };
        check.add(&[rec(4, 9), rec(2, 21)], &table_loads(listing));
        assert_eq!(
            check.report(),
            [
                "     1  classe  2 table s21 : entrees",
                "     0  classe  2 table s21 : retrouvees avec decalage x1",
                "     1  classe  2 table s21 : retrouvees avec decalage x4",
                "     1  classe  2 table s9  : entrees",
                "     0  classe  2 table s9  : retrouvees avec decalage x1",
                "     1  classe  2 table s9  : retrouvees avec decalage x4",
                "     2  entrees de table",
                "     1  shaders",
                "genres : {'t': 2}",
            ]
            .map(|l| format!("{l}{NL}"))
            .concat()
        );
    }
}
