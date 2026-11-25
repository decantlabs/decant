//! Translates shaders and writes, next to each SPIR-V module, a JSON description of what the
//! module expects, as `src/shaders/decant_shader/tools/decant_shader_describe.py` does. See
//! [`decant_shader_bridge::describe`].
//!
//! Usage:
//! * `decant_shader_describe <output folder> <pack number> [...]`
//! * `decant_shader_describe <output folder> --all [--stage pixel|vertex|compute|mesh] [--jobs N] [--limit N] [--sample N]`

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use decant_shader_bridge::describe::{self, Rows, Summary};
use decant_shader_bridge::py::{self, Failure};
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;

const USAGE: &str = "usage: decant_shader_describe [-h] [--all] [--stage {pixel,vertex,compute,mesh}] [--jobs JOBS] [--limit LIMIT] [--sample SAMPLE]\n                         output [ids ...]";
const OPTIONS: [&str; 6] = ["--help", "--all", "--stage", "--jobs", "--limit", "--sample"];
const STAGES: [&str; 4] = ["pixel", "vertex", "compute", "mesh"];

#[derive(Debug, Default, PartialEq)]
struct Args {
    output: PathBuf,
    ids: Vec<i64>,
    all: bool,
    stage: Option<String>,
    jobs: i64,
    limit: Option<i64>,
    sample: Option<i64>,
}

fn usage_error(message: &str) -> String {
    format!("{USAGE}\ndecant_shader_describe: error: {message}")
}

fn number(option: &str, text: &str) -> Result<i64, String> {
    py::int(text).map_err(|_| usage_error(&format!("argument {option}: invalid int value: {}", py::repr_str(text))))
}

/// The command line as argparse reads it: options anywhere, long options by any unique prefix.
fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args { jobs: 8, ..Args::default() };
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if !arg.starts_with("--") && arg != "-h" {
            positional.push(arg.clone());
            continue;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) => (n, Some(v.to_owned())),
            None => (arg.as_str(), None),
        };
        let matches: Vec<&str> = if name == "-h" { vec!["--help"] } else { OPTIONS.iter().copied().filter(|o| o.starts_with(name) && name.len() > 2).collect() };
        let option = match matches.as_slice() {
            [one] => *one,
            [] => return Err(usage_error(&format!("unrecognized arguments: {arg}"))),
            many => return Err(usage_error(&format!("ambiguous option: {name} could match {}", many.join(", ")))),
        };
        let mut value = || inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| usage_error(&format!("argument {option}: expected one argument")));
        match option {
            "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--all" => parsed.all = true,
            "--stage" => {
                let v = value()?;
                if !STAGES.contains(&v.as_str()) {
                    return Err(usage_error(&format!(
                        "argument --stage: invalid choice: {} (choose from 'pixel', 'vertex', 'compute', 'mesh')",
                        py::repr_str(&v)
                    )));
                }
                parsed.stage = Some(v);
            }
            "--jobs" => parsed.jobs = number(option, &value()?)?,
            "--limit" => parsed.limit = Some(number(option, &value()?)?),
            "--sample" => parsed.sample = Some(number(option, &value()?)?),
            _ => unreachable!(),
        }
    }
    let Some((output, ids)) = positional.split_first() else {
        return Err(usage_error("the following arguments are required: output"));
    };
    parsed.output = PathBuf::from(output);
    for id in ids {
        parsed.ids.push(py::int(id).map_err(|_| usage_error(&format!("argument ids: invalid int value: {}", py::repr_str(id))))?);
    }
    Ok(parsed)
}

fn run(session: &Session, args: Args) -> Result<(), Failure> {
    let out_dir = std::path::absolute(&args.output).map_err(|e| Failure::io(&e, &args.output))?;
    std::fs::create_dir_all(&out_dir).map_err(|e| Failure::io(&e, &out_dir))?;
    let rows = Rows::load(&session.config)?;
    py::index(&rows.meta, 0)?;
    let ids = describe::select(&rows.meta, args.ids, args.all, args.stage.as_deref(), args.sample, args.limit)?;
    if ids.is_empty() {
        eprintln!("{}", usage_error("give pack ids or --all"));
        std::process::exit(2);
    }
    let summary = Mutex::new(Summary::default());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.jobs.max(1).min(ids.len() as i64) {
            scope.spawn(|| {
                while let Some(&pack_id) = ids.get(next.fetch_add(1, Ordering::SeqCst)) {
                    let outcome = describe::one(session, &rows, pack_id, &out_dir);
                    summary.lock().expect("no worker panics").add(&outcome);
                }
            });
        }
    });
    let summary = summary.into_inner().expect("no worker panics");
    let path = out_dir.join("summary.json");
    std::fs::write(&path, py::native_lines(&summary.file())).map_err(|e| Failure::io(&e, &path))?;
    println!("{}", summary.line());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The Python opens the archive when it imports decant_shader_identify, before reading its arguments.
    let session = match Session::open(Config::from_env()) {
        Ok(s) => s,
        Err(failure) => {
            eprintln!("{failure}");
            return ExitCode::from(1);
        }
    };
    let args = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&session, args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("{failure}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        parse(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn arguments_parse_like_argparse() {
        let a = args(&["out", "--all", "--stage", "pixel", "--jobs=3", "--lim", "10", "--sample", "50"]).unwrap();
        assert_eq!((a.all, a.stage.as_deref(), a.jobs, a.limit, a.sample), (true, Some("pixel"), 3, Some(10), Some(50)));
        let a = args(&["out", "5", "12"]).unwrap();
        assert_eq!((a.output, a.ids, a.jobs), (PathBuf::from("out"), vec![5, 12], 8));
        assert!(args(&["out", "--stage", "hull"]).unwrap_err().contains("invalid choice: 'hull'"));
        assert!(args(&[]).unwrap_err().ends_with("required: output"));
        assert!(args(&["out", "x"]).unwrap_err().ends_with("invalid int value: 'x'"));
    }
}
