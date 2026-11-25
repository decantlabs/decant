//! `decant_stubgen <rapport_imports.json> --out <folder> [--dll name.dll]... [--build]`
//!
//! Writes a Rust `cdylib` project for a stand in DLL of every module that `audit_xbox_imports`
//! reports as absent. With `--build` it runs `cargo build --release` in each project and copies the
//! DLL next to the projects. See the library documentation for what the stand ins do. Use a
//! folder outside the repository.

use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!("usage: decant_stubgen <rapport_imports.json> --out <folder> [--dll name.dll]... [--build]");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut report = None;
    let mut out = None;
    let mut only = Vec::new();
    let mut build = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => match args.next() {
                Some(dir) => out = Some(PathBuf::from(dir)),
                None => return usage(),
            },
            "--dll" => match args.next() {
                Some(name) => only.push(name),
                None => return usage(),
            },
            "--build" => build = true,
            _ if report.is_none() => report = Some(PathBuf::from(arg)),
            _ => return usage(),
        }
    }
    let (Some(report), Some(out)) = (report, out) else { return usage() };
    match run(&report, &out, &only, build) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn run(report: &std::path::Path, out: &std::path::Path, only: &[String], build: bool) -> Result<(), String> {
    let text = std::fs::read_to_string(report).map_err(|e| format!("{}: {e}", report.display()))?;
    let modules = decant_stub::modules(&text, only)?;
    if modules.is_empty() {
        println!("No module to stand in for: every import is resolved, or no module matches --dll.");
        return Ok(());
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    decant_stub::write(&modules, out)?;
    for m in &modules {
        println!("{:<32} {} export(s)", m.file, m.exports.len());
    }
    println!("Projects written to {}.", out.display());
    if build {
        for m in &modules {
            let dll = decant_stub::build(m, out)?;
            println!("Built {}", dll.display());
        }
    } else {
        println!("Build each with `cargo build --release` in its folder, or run again with --build.");
    }
    Ok(())
}
