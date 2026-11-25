//! Sums up one or more `results.jsonl`: valid modules and failure causes grouped together, as
//! `src/shaders/decant_shader/tools/decant_shader_summary.py` does.
//!
//! Usage: `decant_shader_summary <folder> [<folder> ...]`

use std::process::ExitCode;

use decant_shader_batch::summary::{self, Summary};

fn run() -> Result<(), String> {
    let mut total = Summary::default();
    for folder in std::env::args_os().skip(1) {
        let folder = folder.to_string_lossy().into_owned();
        println!("{}", total.add(&folder, &summary::rows(&folder)?)?);
    }
    for line in total.finish() {
        println!("{line}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
