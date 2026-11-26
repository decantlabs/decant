//! `decant_trace_summary <trace.json> [--top N] [--from-s S] [--to-s S]`: prints the top spans
//! by total and maximum duration, thread active time, lock wait hotspots, 10s frame windows, and
//! worst stutter frames of a trace written by `decant-trace`.

use std::process::ExitCode;

const DEFAULT_TOP: usize = 10;
const USAGE: &str = "usage: decant_trace_summary <trace.json> [--top N] [--from-s S] [--to-s S]";

#[derive(Debug, PartialEq)]
struct Args {
    file: String,
    top: usize,
    from_s: Option<f64>,
    to_s: Option<f64>,
}

/// The file, row count, and optional time range from the command line.
fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut file = None;
    let mut top = DEFAULT_TOP;
    let mut from_s = None;
    let mut to_s = None;
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
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path if file.is_none() => file = Some(path.to_owned()),
            _ => return Err("only one trace file is accepted".to_owned()),
        }
    }
    if let (Some(f), Some(t)) = (from_s, to_s)
        && f > t
    {
        return Err("--from-s must be <= --to-s".to_owned());
    }
    file.map(|file| Args { file, top, from_s, to_s }).ok_or_else(|| "no trace file given".to_owned())
}

fn run(args: &[String]) -> Result<String, String> {
    let parsed = parse_args(args)?;
    let json = std::fs::read_to_string(&parsed.file).map_err(|e| format!("cannot read {}: {e}", parsed.file))?;
    let summary = decant_trace::summarize_range(&json, parsed.from_s, parsed.to_s).map_err(|e| format!("{}: {e}", parsed.file))?;
    Ok(summary.report(parsed.top))
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
        assert_eq!(parse_args(&args(&["t.json"])), Ok(Args { file: "t.json".to_owned(), top: 10, from_s: None, to_s: None }));
        assert_eq!(parse_args(&args(&["--top", "3", "--from-s", "10.5", "--to-s", "30", "t.json"])), Ok(Args { file: "t.json".to_owned(), top: 3, from_s: Some(10.5), to_s: Some(30.0) }));
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [&[][..], &["--top"], &["--top", "0", "t"], &["--top", "x", "t"], &["--from-s", "-1", "t"], &["--from-s", "20", "--to-s", "10", "t"], &["--what", "t"], &["a", "b"]] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn summarizes_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.json");
        std::fs::write(&path, r#"{"traceEvents":[{"name":"a","cat":"c","ph":"X","ts":0,"dur":1000,"pid":1,"tid":1}]}"#).unwrap();
        let report = run(&args(&[path.to_str().unwrap()])).unwrap();
        assert!(report.contains("c:a"));
        std::fs::write(&path, "oops").unwrap();
        assert!(run(&args(&[path.to_str().unwrap()])).unwrap_err().contains("not valid JSON"));
        assert!(run(&args(&["missing_trace_file.json"])).unwrap_err().contains("cannot read"));
    }
}
