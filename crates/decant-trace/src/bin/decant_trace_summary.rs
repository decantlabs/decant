//! `decant_trace_summary <trace.json> [--top N] [--from-s S] [--to-s S] [--json-out <path>]`
//! or `decant_trace_summary --compare <base.json> <head.json>`.

use std::process::ExitCode;

const DEFAULT_TOP: usize = 10;
const USAGE: &str = "usage: decant_trace_summary <trace.json> [--top N] [--from-s S] [--to-s S] [--json-out <path>] | --compare <base.json> <head.json>";

#[derive(Debug, PartialEq)]
enum Command {
    Summarize {
        file: String,
        top: usize,
        from_s: Option<f64>,
        to_s: Option<f64>,
        json_out: Option<String>,
    },
    Compare {
        base: String,
        head: String,
    },
}

/// Parses command line options for summary or compare mode.
fn parse_args(args: &[String]) -> Result<Command, String> {
    let mut file = None;
    let mut top = DEFAULT_TOP;
    let mut from_s = None;
    let mut to_s = None;
    let mut json_out = None;
    let mut compare: Option<(String, String)> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--top" => {
                let n = it.next().ok_or("--top needs a number")?;
                top = n.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("bad row count {n}"))?;
            }
            "--from-s" => {
                let s = it.next().ok_or("--from-s needs a number")?;
                let v = s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0).ok_or_else(|| format!("bad --from-s {s}"))?;
                from_s = Some(v);
            }
            "--to-s" => {
                let s = it.next().ok_or("--to-s needs a number")?;
                let v = s.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0).ok_or_else(|| format!("bad --to-s {s}"))?;
                to_s = Some(v);
            }
            "--json-out" => {
                let p = it.next().ok_or("--json-out needs a path")?;
                json_out = Some(p.to_owned());
            }
            "--compare" => {
                if compare.is_some() {
                    return Err("only one --compare pair is accepted".to_owned());
                }
                let base = it.next().ok_or("--compare needs <base.json> <head.json>")?;
                let head = it.next().ok_or("--compare needs <base.json> <head.json>")?;
                compare = Some((base.to_owned(), head.to_owned()));
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path if file.is_none() => file = Some(path.to_owned()),
            _ => return Err("only one trace file is accepted".to_owned()),
        }
    }
    if let Some((base, head)) = compare {
        if file.is_some() || json_out.is_some() || from_s.is_some() || to_s.is_some() {
            return Err("--compare cannot be combined with a trace file or range options".to_owned());
        }
        return Ok(Command::Compare { base, head });
    }
    if let (Some(f), Some(t)) = (from_s, to_s)
        && f > t
    {
        return Err("--from-s must be <= --to-s".to_owned());
    }
    file.map(|file| Command::Summarize { file, top, from_s, to_s, json_out }).ok_or_else(|| "no trace file given".to_owned())
}

fn run(args: &[String]) -> Result<String, String> {
    match parse_args(args)? {
        Command::Compare { base, head } => {
            let base_json = std::fs::read_to_string(&base).map_err(|e| format!("cannot read {base}: {e}"))?;
            let head_json = std::fs::read_to_string(&head).map_err(|e| format!("cannot read {head}: {e}"))?;
            decant_trace::Summary::compare_metrics_json(&base_json, &head_json)
        }
        Command::Summarize { file, top, from_s, to_s, json_out } => {
            let json = std::fs::read_to_string(&file).map_err(|e| format!("cannot read {file}: {e}"))?;
            let summary = decant_trace::summarize_range(&json, from_s, to_s).map_err(|e| format!("{file}: {e}"))?;
            if let Some(out_path) = json_out {
                let metrics = summary.to_metrics_json(top);
                std::fs::write(&out_path, metrics).map_err(|e| format!("cannot write {out_path}: {e}"))?;
            }
            Ok(summary.report(top))
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(report) => {
            print!("{report}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("decant_trace_summary: {e}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_file_top_and_range() {
        assert_eq!(
            parse_args(&args(&["t.json"])),
            Ok(Command::Summarize { file: "t.json".to_owned(), top: 10, from_s: None, to_s: None, json_out: None })
        );
        assert_eq!(
            parse_args(&args(&["--top", "3", "--from-s", "10.5", "--to-s", "30", "--json-out", "m.json", "t.json"])),
            Ok(Command::Summarize { file: "t.json".to_owned(), top: 3, from_s: Some(10.5), to_s: Some(30.0), json_out: Some("m.json".to_owned()) })
        );
        assert_eq!(
            parse_args(&args(&["--compare", "a.json", "b.json"])),
            Ok(Command::Compare { base: "a.json".to_owned(), head: "b.json".to_owned() })
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [
            &[][..],
            &["--top"],
            &["--top", "0", "t"],
            &["--top", "x", "t"],
            &["--from-s", "-1", "t"],
            &["--from-s", "20", "--to-s", "10", "t"],
            &["--json-out"],
            &["--compare", "a.json"],
            &["--compare", "a.json", "b.json", "t.json"],
            &["--what", "t"],
            &["a", "b"],
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn summarizes_a_file_writes_json_and_compares() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.json");
        let m1 = dir.path().join("m1.json");
        let m2 = dir.path().join("m2.json");
        std::fs::write(
            &path,
            r#"{"traceEvents":[
                {"name":"frame","cat":"frame","ph":"i","s":"g","ts":0,"pid":1,"tid":1},
                {"name":"frame","cat":"frame","ph":"i","s":"g","ts":20000,"pid":1,"tid":1},
                {"name":"a","cat":"c","ph":"X","ts":0,"dur":1000,"pid":1,"tid":1}
            ]}"#,
        )
        .unwrap();
        let report = run(&args(&[path.to_str().unwrap(), "--json-out", m1.to_str().unwrap()])).unwrap();
        assert!(report.contains("c:a"));
        assert!(m1.exists());
        std::fs::copy(&m1, &m2).unwrap();
        let cmp = run(&args(&["--compare", m1.to_str().unwrap(), m2.to_str().unwrap()])).unwrap();
        assert!(cmp.contains("overall comparison"));
        assert!(cmp.contains("0.0%"));
        std::fs::write(&path, "oops").unwrap();
        assert!(run(&args(&[path.to_str().unwrap()])).unwrap_err().contains("not valid JSON"));
        assert!(run(&args(&["missing_trace_file.json"])).unwrap_err().contains("cannot read"));
    }
}
