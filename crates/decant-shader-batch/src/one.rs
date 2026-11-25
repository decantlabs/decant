//! `decant_shader_one.py`: replays one shader (or a few) in the probe with the `DECANT_SHADER_TRACE` trace.
//! The output folder gets `one.pack`, `one.pack.inputs`, `one.pack.vinputs`, `one.tsv` and the
//! modules.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use regex::Regex;
use decant_shader_core::py;

/// The shaders asked for: numbers are pack positions (names are not unique), anything else is a name.
pub fn select(args: &[String]) -> Result<(HashSet<String>, HashSet<i64>), String> {
    let digits = |a: &str| !a.is_empty() && a.bytes().all(|b| b.is_ascii_digit());
    let names = args.iter().filter(|a| !digits(a)).cloned().collect();
    let ids = args.iter().filter(|a| digits(a)).map(|a| py::int(a)).collect::<Result<_, _>>()?;
    Ok((names, ids))
}

fn last_chars(text: &str, n: usize) -> &str {
    match text.char_indices().rev().nth(n.saturating_sub(1)) {
        Some((i, _)) if n > 0 => &text[i..],
        _ if n == 0 => "",
        _ => text,
    }
}

fn first_chars(text: &str, n: usize) -> &str {
    text.char_indices().nth(n).map_or(text, |(i, _)| &text[..i])
}

/// What the tool prints for one replayed shader before it reads the log, line ends included.
/// `code` is the probe's exit code and `stderr` its error output.
pub fn trace(index: usize, name: &str, stage: u8, code: i64, stderr: &[u8]) -> String {
    let err = String::from_utf8_lossy(stderr);
    let mut out = format!("== {index} {name} etape {stage} code {code}\n");

    let marks = Regex::new(r"decant_trace \d+|decant:[A-Za-z_0-9]+").expect("valid pattern");
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for m in marks.find_iter(&err) {
        *counts.entry(m.as_str()).or_default() += 1;
    }
    let items: Vec<String> = counts.iter().map(|(m, c)| format!("({}, {c})", py::repr_str(m))).collect();
    out += &format!("trace : [{}]\n", items.join(", "));
    out += &format!("fin : {}\n", last_chars(&err, 300));

    let trees = Regex::new(r"DECANT_TREE ([^;]*);").expect("valid pattern");
    let found: BTreeSet<&str> = trees.captures_iter(&err).filter_map(|c| c.get(1)).map(|m| m.as_str()).collect();
    for t in found {
        out += &format!("ARBRE {}\n", first_chars(t, 900));
    }
    out
}

/// The start of the log `one.tsv`, printed last.
pub fn log(log: &[u8]) -> String {
    format!("{}\n", first_chars(&py::read_text(log), 900))
}

/// [`trace`] then [`log`].
pub fn report(index: usize, name: &str, stage: u8, code: i64, stderr: &[u8], log_bytes: &[u8]) -> String {
    trace(index, name, stage, code, stderr) + &log(log_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_matches_the_python_layout() {
        let stderr = "decant_trace 3\ndecant:load_a decant:load_a decant_trace 12 DECANT_TREE b(a);DECANT_TREE a;DECANT_TREE b(a);\nend".as_bytes();
        let text = report(42, "PS_X", 0, 3_221_225_477, stderr, b"B\t0\tPS_X\r\nC\t0\tOK\r\n");
        let expected = concat!(
            "== 42 PS_X etape 0 code 3221225477\n",
            "trace : [('decant:load_a', 2), ('decant_trace 12', 1), ('decant_trace 3', 1)]\n",
            "fin : decant_trace 3\ndecant:load_a decant:load_a decant_trace 12 DECANT_TREE b(a);DECANT_TREE a;DECANT_TREE b(a);\nend\n",
            "ARBRE a\n",
            "ARBRE b(a)\n",
            "B\t0\tPS_X\nC\t0\tOK\n\n",
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn long_texts_are_cut_by_characters() {
        let err = format!("{}{}", "é".repeat(10), "z".repeat(299));
        let text = report(0, "a", 1, 0, err.as_bytes(), &[]);
        assert!(text.contains(&format!("fin : é{}\n", "z".repeat(299))));
        assert_eq!(first_chars("abc", 5), "abc");
        assert_eq!(last_chars("abc", 0), "");
        assert_eq!(last_chars("abc", 2), "bc");
    }

    #[test]
    fn numbers_select_by_position() {
        let args: Vec<String> = ["12", "PS_Water", "007", "-3", ""].iter().map(|s| s.to_string()).collect();
        let (names, ids) = select(&args).unwrap();
        assert_eq!(ids, HashSet::from([12, 7]));
        assert_eq!(names, HashSet::from(["PS_Water".to_owned(), "-3".to_owned(), String::new()]));
    }
}
