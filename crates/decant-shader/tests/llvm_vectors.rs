//! The decoder against the instruction words of LLVM's GFX10 disassembler tests
//! (`tests/data/llvm_gfx10.txt`, taken from llvm/llvm-project under Apache-2.0 with the LLVM
//! exception). Each line holds the words and the text LLVM prints for them.

use std::collections::BTreeMap;

use decant_shader::decode::{Insn, decode_one, encode};

const VECTORS: &str = include_str!("data/llvm_gfx10.txt");

/// LLVM names that the table (which follows Mesa) spells differently.
const ALIASES: &[(&str, &str)] = &[
    ("v_add_nc_u32", "v_add_u32"),
    ("v_sub_nc_u32", "v_sub_u32"),
    ("v_subrev_nc_u32", "v_subrev_u32"),
    ("v_add_co_ci_u32", "v_addc_co_u32"),
    ("v_sub_co_ci_u32", "v_subb_co_u32"),
    ("v_subrev_co_ci_u32", "v_subbrev_co_u32"),
    ("v_add_nc_u16", "v_add_u16"),
    ("v_sub_nc_u16", "v_sub_u16"),
    ("v_add_nc_i16", "v_add_i16"),
    ("v_sub_nc_i16", "v_sub_i16"),
    ("v_add_nc_i32", "v_add_i32"),
    ("v_sub_nc_i32", "v_sub_i32"),
];

/// The name LLVM would print for a table name, without any encoding suffix.
fn llvm_name(table: &str) -> String {
    let base = table.strip_suffix("_e64").unwrap_or(table);
    for (llvm, mesa) in ALIASES {
        if base == *mesa {
            return (*llvm).to_owned();
        }
    }
    // The integer compares: Mesa says lg and tru where LLVM says ne and t.
    let is_int_cmp = (base.starts_with("v_cmp_") || base.starts_with("v_cmpx_"))
        && ["_i16", "_i32", "_i64", "_u16", "_u32", "_u64"]
            .iter()
            .any(|s| base.ends_with(s));
    if is_int_cmp {
        if let Some(rest) = base.strip_prefix("v_cmp_lg_") {
            return format!("v_cmp_ne_{rest}");
        }
        if let Some(rest) = base.strip_prefix("v_cmpx_lg_") {
            return format!("v_cmpx_ne_{rest}");
        }
        if let Some(rest) = base.strip_prefix("v_cmp_tru_") {
            return format!("v_cmp_t_{rest}");
        }
        if let Some(rest) = base.strip_prefix("v_cmpx_tru_") {
            return format!("v_cmpx_t_{rest}");
        }
    }
    base.to_owned()
}

fn strip_suffix(name: &str) -> &str {
    for s in ["_e32", "_e64", "_sdwa", "_dpp8", "_dpp"] {
        if let Some(n) = name.strip_suffix(s) {
            return n;
        }
    }
    name
}

struct Vector {
    words: Vec<u32>,
    text: &'static str,
}

fn vectors() -> Vec<Vector> {
    VECTORS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let (w, text) = l.split_once(" | ").unwrap();
            Vector {
                words: w
                    .split(' ')
                    .map(|x| u32::from_str_radix(x, 16).unwrap())
                    .collect(),
                text,
            }
        })
        .collect()
}

/// What the decoder gets wrong, as `reason -> (count, first example)`.
#[test]
fn every_llvm_vector_decodes_to_its_instruction() {
    let mut bad: BTreeMap<String, (u32, String)> = BTreeMap::new();
    let mut total = 0;
    for v in vectors() {
        total += 1;
        let mut fail = |reason: String| {
            let e = bad.entry(reason).or_insert((0, v.text.to_owned()));
            e.0 += 1;
        };
        let insn = match decode_one(&v.words, 0) {
            Ok(i) => i,
            Err(e) => {
                fail(format!(
                    "decode error {}",
                    e.to_string().split(" at ").next().unwrap()
                ));
                continue;
            }
        };
        let want = strip_suffix(v.text.split(' ').next().unwrap());
        // GFX10.1 only: GFX10.3 (the table) gives these two numbers to the fused forms.
        if matches!(want, "v_mac_legacy_f32" | "v_mad_legacy_f32") {
            continue;
        }
        let got = llvm_name(insn.op.name());
        if want != got {
            fail(format!("name {want} decoded as {got}"));
            continue;
        }
        if usize::from(insn.len) != v.words.len() {
            fail(format!(
                "length of {want}: {} words, decoded {}",
                v.words.len(),
                insn.len
            ));
            continue;
        }
        let again = encode(&insn);
        match decode_one(&again, 0) {
            Ok(back) if Insn { pc: 0, ..back } == insn => {}
            _ => fail(format!("round trip of {want}")),
        }
    }
    let report: Vec<String> = bad
        .iter()
        .map(|(k, (n, ex))| format!("{n:5} {k}   e.g. {ex}"))
        .collect();
    assert!(
        report.is_empty(),
        "{} vectors, {} reasons:\n{}",
        total,
        report.len(),
        report.join("\n")
    );
}
