//! Measures the shaders of the pack in batches of 48, as
//! `src/shaders/decant_shader/tools/decant_shader_batch.py` does. A probe run that hangs or crashes is
//! stopped and the batch goes on after the shader the probe log names.
//!
//! Usage: `decant_shader_batch <output> <stage> [<stage> ...] [--ir-only] [--metadata <jsonl>]
//! [--placeholders] [--validate <spirv-val>]`, with stages 0 (pixel), 1 (vertex), 2 (geometry), 5 (compute)
//! and 13 (mesh). The output folder must not exist yet.
//!
//! The pack comes from `src/shaders/out`, the probe from `src/shaders/decant_shader` or `DECANT_SHADER_PROBE` (see
//! `decant_shader_core::paths`).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use decant_extract_shaders::pack;
use decant_shader_batch::batch::{self, External};
use decant_shader_core::{meta, paths, py};

const USAGE: &str = "usage: decant_shader_batch [-h] [--ir-only] [--metadata METADATA] [--placeholders] [--validate VALIDATE]\n                      output {0,1,5,13} [{0,1,5,13} ...]";
const STAGES: [u8; 5] = [0, 1, 2, 5, 13];

#[derive(Debug, Default)]
struct Args {
    output: PathBuf,
    stages: Vec<u8>,
    ir_only: bool,
    metadata: Option<PathBuf>,
    placeholders: bool,
    validate: Option<PathBuf>,
}

/// An argument error, reported as argparse reports it.
fn usage_error(message: &str) -> String {
    format!("{USAGE}\ndecant_shader_batch: error: {message}")
}

/// Long options, which argparse also accepts as any unique prefix.
const OPTIONS: [&str; 5] = ["--help", "--ir-only", "--metadata", "--placeholders", "--validate"];

fn parse(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut positional = Vec::new();
    let mut it = args.iter();
    let mut only_positional = false;
    while let Some(arg) = it.next() {
        if only_positional || !arg.starts_with('-') || arg == "-" || py::int(arg).is_ok() {
            positional.push(arg.clone());
            continue;
        }
        if arg == "--" {
            only_positional = true;
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
        let mut value = || {
            inline.clone().or_else(|| it.next().cloned()).map(PathBuf::from).ok_or_else(|| usage_error(&format!("argument {option}: expected one argument")))
        };
        match option {
            "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--ir-only" => parsed.ir_only = true,
            "--placeholders" => parsed.placeholders = true,
            "--metadata" => parsed.metadata = Some(value()?),
            "--validate" => parsed.validate = Some(value()?),
            _ => unreachable!(),
        }
    }
    let Some((output, stages)) = positional.split_first() else {
        return Err(usage_error("the following arguments are required: output, stages"));
    };
    if stages.is_empty() {
        return Err(usage_error("the following arguments are required: stages"));
    }
    parsed.output = PathBuf::from(output);
    for s in stages {
        let n = py::int(s).map_err(|_| usage_error(&format!("argument stages: invalid int value: {}", py::repr_str(s))))?;
        match STAGES.iter().find(|&&c| i64::from(c) == n) {
            Some(&c) => parsed.stages.push(c),
            None => {
                return Err(usage_error(&format!("argument stages: invalid choice: {} (choose from '0', '1', '2', '5', '13')", py::repr_str(s))));
            }
        }
    }
    Ok(parsed)
}

fn run(args: &Args) -> Result<(), String> {
    let out = std::path::absolute(&args.output).map_err(|e| e.to_string())?;
    fs::create_dir(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let validator = match &args.validate {
        Some(v) => {
            let v = std::path::absolute(v).map_err(|e| e.to_string())?;
            if !v.is_file() {
                return Err(format!("FileNotFoundError: {}", v.display()));
            }
            Some(v)
        }
        None => None,
    };
    let meta = match &args.metadata {
        Some(path) => Some(meta::rows(path)?),
        None => None,
    };
    let pack_path = paths::out().join("shaders_gpu.pack");
    let (header, records) =
        pack::read(&pack_path, |_, _, stage| args.stages.contains(&stage)).map_err(|e| format!("{}: {e}", pack_path.display()))?;
    if !header.is_current() {
        return Err(format!("AssertionError: {} is not an XSGC version 1 pack", pack_path.display()));
    }
    let buckets = batch::group(&args.stages, records);
    let jobs = batch::jobs(&buckets);
    let tools = External { probe: paths::probe(), validator };
    let results = out.join("results.jsonl");
    let mut file = BufWriter::new(File::create(&results).map_err(|e| format!("{}: {e}", results.display()))?);
    let mode = batch::mode(args.ir_only, args.placeholders);
    batch::run_all(&out, &jobs, meta.as_deref(), mode, &tools, &mut |rows| {
        let mut text = String::new();
        for row in rows {
            text += &row.dumps();
            text.push('\n');
        }
        file.write_all(py::native_newlines(&text).as_bytes()).and_then(|()| file.flush()).map_err(|e| format!("{}: {e}", results.display()))
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
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
        let a = args(&["runs", "0", "13", "--ir", "--metadata=m.jsonl", "--val", "spirv-val.exe"]).unwrap();
        assert_eq!((a.output, a.stages, a.ir_only, a.placeholders), (PathBuf::from("runs"), vec![0, 13], true, false));
        assert_eq!((a.metadata, a.validate), (Some(PathBuf::from("m.jsonl")), Some(PathBuf::from("spirv-val.exe"))));
        assert_eq!(args(&["runs", "2"]).unwrap().stages, vec![2]);
        assert!(args(&["runs", "3"]).unwrap_err().ends_with("invalid choice: '3' (choose from '0', '1', '2', '5', '13')"));
        assert!(args(&["runs"]).unwrap_err().ends_with("required: stages"));
        assert!(args(&["runs", "0", "--bogus"]).unwrap_err().ends_with("unrecognized arguments: --bogus"));
        assert!(args(&["runs", "0", "--metadata"]).unwrap_err().ends_with("expected one argument"));
    }
}
