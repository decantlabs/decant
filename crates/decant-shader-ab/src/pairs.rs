//! The batch list (one pair per TSV line), the result line of a pair and the final table.

use serde::{Deserialize, Serialize};

use crate::compare::{Diff, Output, Tolerance, Verdict, judge};
use crate::model::Stage;

/// One line of the batch list: `hash, path A, path B, stage, root signature`. The last two may be
/// `auto` and `-`, or left out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    pub hash: String,
    pub a: String,
    pub b: String,
    pub stage: Option<Stage>,
    pub root: Option<String>,
}

/// Reads the batch list. Empty lines, `#` comments and a header line starting with `hash` are skipped.
pub fn parse(text: &str) -> Result<Vec<Pair>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.starts_with('#') || (n == 0 && line.to_ascii_lowercase().starts_with("hash\t")) {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            return Err(format!("line {}: want hash, path A, path B, stage, root signature separated by tabs", n + 1));
        }
        let stage = Stage::parse(cols.get(3).copied().unwrap_or("auto")).map_err(|e| format!("line {}: {e}", n + 1))?;
        let root = cols.get(4).map(|s| s.trim()).filter(|s| !s.is_empty() && *s != "-").map(str::to_string);
        out.push(Pair { hash: cols[0].to_string(), a: cols[1].to_string(), b: cols[2].to_string(), stage, root });
    }
    Ok(out)
}

/// The outcome of one pair, as the batch prints it and `run --json` writes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub hash: String,
    pub stage: Option<Stage>,
    pub verdict: Verdict,
    /// What the two sides declare differently. A finding of its own, whatever the outputs say.
    pub declarations: Vec<String>,
    pub diff: Option<Diff>,
    pub a_error: Option<String>,
    pub b_error: Option<String>,
    /// Why the pair was skipped, and approximations the runner made.
    pub notes: Vec<String>,
}

impl Outcome {
    /// The outcome of a pair from its plan and the results of both sides.
    pub fn from_sides(hash: &str, plan: &crate::plan::Plan, a: Result<Vec<Output>, String>, b: Result<Vec<Output>, String>, tol: Tolerance) -> Outcome {
        let (verdict, diff) = judge(&a, &b, tol);
        Outcome { hash: hash.to_string(), stage: Some(plan.stage), verdict, declarations: plan.declarations.clone(), diff, a_error: a.err(), b_error: b.err(), notes: plan.notes.clone() }
    }

    pub fn skipped(hash: &str, why: String) -> Outcome {
        Outcome { hash: hash.to_string(), stage: None, verdict: Verdict::Skipped, declarations: Vec::new(), diff: None, a_error: None, b_error: None, notes: vec![why] }
    }

    /// `same` or `differ(n)`.
    pub fn declares(&self) -> String {
        if self.declarations.is_empty() { "same".into() } else { format!("differ({})", self.declarations.len()) }
    }

    /// One line: hash, stage, verdict, declarations, detail. Tabs separate the columns.
    pub fn line(&self) -> String {
        let stage = self.stage.map_or("-", Stage::short);
        let detail = self.detail().replace(['\t', '\n', '\r'], " ");
        format!("{}\t{stage}\t{}\t{}\t{detail}", self.hash, self.verdict.word(), self.declares())
    }

    /// The most useful single sentence about the pair.
    pub fn detail(&self) -> String {
        let mut parts = Vec::new();
        match self.verdict {
            Verdict::AFailed => parts.push(format!("A: {}", self.a_error.as_deref().unwrap_or("?"))),
            Verdict::BFailed => parts.push(format!("B: {}", self.b_error.as_deref().unwrap_or("?"))),
            Verdict::BothFailed => parts.push(format!("A: {} | B: {}", self.a_error.as_deref().unwrap_or("?"), self.b_error.as_deref().unwrap_or("?"))),
            Verdict::Skipped => parts.push(self.notes.first().cloned().unwrap_or_default()),
            _ => {}
        }
        if let Some(d) = &self.diff {
            if let Some(f) = &d.first {
                parts.push(format!("{} of {} components differ, first {} at {}: A {} B {}", d.different, d.compared, f.output, f.at, f.a, f.b));
            } else if d.inexact > 0 {
                parts.push(format!("{} of {} components inexact, max {} ulp", d.inexact, d.compared, d.max_ulp));
            } else {
                parts.push(format!("{} components", d.compared));
            }
            if let Some(m) = d.mismatched.first() {
                parts.push(m.clone());
            }
        }
        if let Some(first) = self.declarations.first() {
            parts.push(format!("declares: {first}"));
        }
        parts.join("; ")
    }
}

/// The final table: pairs per verdict, and how many of them declare different resources.
pub fn table(outcomes: &[Outcome]) -> String {
    let mut s = format!("{:<18}{:>8}{:>18}\n", "verdict", "pairs", "declare differ");
    for v in Verdict::ALL {
        let of: Vec<&Outcome> = outcomes.iter().filter(|o| o.verdict == v).collect();
        let differ = of.iter().filter(|o| !o.declarations.is_empty()).count();
        s.push_str(&format!("{:<18}{:>8}{:>18}\n", v.word(), of.len(), differ));
    }
    let differ = outcomes.iter().filter(|o| !o.declarations.is_empty()).count();
    s.push_str(&format!("{:<18}{:>8}{:>18}\n", "total", outcomes.len(), differ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_takes_optional_columns_comments_and_a_header() {
        let text = "hash\tpath_a\tpath_b\tstage\troot\n# a comment\n\nab12\ta.dxil\tb.hlsl\tps\t-\ncd34\tx.dxil\ty.dxil\r\nef56\tp\tq\tauto\trs.bin\n";
        let p = parse(text).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], Pair { hash: "ab12".into(), a: "a.dxil".into(), b: "b.hlsl".into(), stage: Some(Stage::Pixel), root: None });
        assert_eq!(p[1].stage, None);
        assert_eq!(p[1].b, "y.dxil");
        assert_eq!(p[2].root.as_deref(), Some("rs.bin"));
        assert!(parse("x\ty\n").is_err());
        assert!(parse("x\ty\tz\tgs\n").is_err());
    }

    #[test]
    fn the_table_counts_verdicts_and_declaration_findings() {
        let mut a = Outcome::skipped("1", "geometry shader".into());
        a.verdict = Verdict::Identical;
        let mut b = a.clone();
        b.declarations = vec!["only B: t0 space0 RawBuffer".into()];
        let c = Outcome::skipped("3", "geometry shader".into());
        let t = table(&[a, b.clone(), c]);
        assert!(t.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["identical", "2", "1"]), "{t}");
        assert!(t.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["skipped", "1", "0"]), "{t}");
        assert!(t.lines().any(|l| l.split_whitespace().collect::<Vec<_>>() == ["total", "3", "1"]), "{t}");
        assert_eq!(b.line(), "1\t-\tidentical\tdiffer(1)\tgeometry shader; declares: only B: t0 space0 RawBuffer".replace("geometry shader; ", ""));
    }
}
