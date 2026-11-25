//! `decant_trace_summary <trace.json> [--top N]`: prints the top spans by total and maximum duration,
//! the active time of each thread and the lock wait hotspots of a trace written by `decant-trace`.

use std::process::ExitCode;

const DEFAULT_TOP: usize = 10;
const USAGE: &str = "usage: decant_trace_summary <trace.json> [--top N]";

/// The file and the row count from the command line.
fn parse_args(args: &[String]) -> Result<(String, usize), String> {
    let mut file = None;
    let mut top = DEFAULT_TOP;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--top" => {
                let n = it.next().ok_or("--top needs a number")?;
                top = n.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("bad row count {n}"))?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path if file.is_none() => file = Some(path.to_owned()),
            _ => return Err("only one trace file is accepted".to_owned()),
        }
    }
    file.map(|f| (f, top)).ok_or_else(|| "no trace file given".to_owned())
}

fn run(args: &[String]) -> Result<String, String> {
    let (file, top) = parse_args(args)?;
    let json = std::fs::read_to_string(&file).map_err(|e| format!("cannot read {file}: {e}"))?;
    let summary = decant_trace::summarize(&json).map_err(|e| format!("{file}: {e}"))?;
    Ok(summary.report(top))
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
    fn parses_file_and_top() {
        assert_eq!(parse_args(&args(&["t.json"])), Ok(("t.json".to_owned(), 10)));
        assert_eq!(parse_args(&args(&["--top", "3", "t.json"])), Ok(("t.json".to_owned(), 3)));
    }

    #[test]
    fn rejects_bad_arguments() {
        for bad in [&[][..], &["--top"], &["--top", "0", "t"], &["--top", "x", "t"], &["--what", "t"], &["a", "b"]] {
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
