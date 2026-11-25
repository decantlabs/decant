//! Audits the imports of a Xbox GDK (Scarlett) build, read only, as
//! `src/audit_imports/audit_xbox_imports.py` does, with the same report files.
//!
//! For every `.exe` and `.dll` in the folder it reads the import and delayed import tables and
//! sorts each imported DLL: shipped in the game folder ("paquet", checked against its exports),
//! present in this PC's System32 ("PC", checked function by function), or absent ("absente", to
//! reimplement). No game file is run or changed: PE files are read as data.
//!
//! Usage: `audit_xbox_imports <folder or file> [--out <report folder>] [--shims <folder of our DLLs>]`
//!
//! With `--shims`, a DLL that is neither in the package nor on this PC but is in that folder is
//! reported as "shim" and checked function by function against its exports, as the Windows loader
//! would find it when it is staged next to the title.
//!
//! `audit_xbox_imports --dstorage-probe <exe or dll> [--out <folder>]` instead reads one PE for what it
//! expects from DirectStorage (interface IDs passed to `DStorageGetFactory`, vtable slots called on
//! the result) and writes `dstorage_probe.md` and a draft ABI spec `decant_storage_abi.draft.abi`.

mod audit;
mod host;
mod pe;
mod probe;
mod pyfmt;
mod report;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut target = None;
    let mut out = PathBuf::from("rapport");
    let mut shims = None;
    let mut storage_probe = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => match args.next() {
                Some(dir) => out = PathBuf::from(dir),
                None => return usage(),
            },
            "--dstorage-probe" => match args.next() {
                Some(file) => storage_probe = Some(PathBuf::from(file)),
                None => return usage(),
            },
            "--shims" => match args.next() {
                Some(dir) => shims = Some(PathBuf::from(dir)),
                None => return usage(),
            },
            _ if target.is_none() => target = Some(PathBuf::from(arg)),
            _ => return usage(),
        }
    }
    if let Some(file) = storage_probe {
        if target.is_some() || shims.is_some() {
            return usage();
        }
        return match run_storage_probe(&file, &out) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        };
    }
    let Some(target) = target else { return usage() };
    match run(&target, &out, shims.as_deref()) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: audit_xbox_imports <folder or file> [--out <report folder>] [--shims <folder of our DLLs>]");
    eprintln!("       audit_xbox_imports --dstorage-probe <exe or dll> [--out <report folder>]");
    ExitCode::from(2)
}

fn run_storage_probe(file: &Path, out: &Path) -> Result<(), String> {
    let pe = pe::Pe::open(file)?;
    let found = probe::probe(&pe);
    let name = file.file_name().map_or_else(|| file.display().to_string(), |n| n.to_string_lossy().into_owned());
    let md = probe::markdown(&found, &name);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(out.join("dstorage_probe.md"), native_lines(&md)).map_err(|e| e.to_string())?;
    std::fs::write(out.join("decant_storage_abi.draft.abi"), probe::draft_spec(&found, &name)).map_err(|e| e.to_string())?;
    print!("{md}");
    println!("\nDraft spec: {}", out.join("decant_storage_abi.draft.abi").display());
    Ok(())
}

fn run(target: &Path, out: &Path, shims: Option<&Path>) -> Result<u8, String> {
    let (root, files) = audit::targets(target);
    if files.is_empty() {
        println!("Aucun .exe / .dll dans {}", pyfmt::text(&pyfmt::normalize(target)));
        return Ok(2);
    }
    let mut host = host::Host::new();
    let audit = audit::run(target, files, root, &mut host, shims);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let md = out.join("rapport_imports.md");
    std::fs::write(&md, native_lines(&report::markdown(&audit, &host))).map_err(|e| e.to_string())?;
    std::fs::write(out.join("rapport_imports.json"), native_lines(&report::json(&audit))).map_err(|e| e.to_string())?;

    let exes = report::exes(&audit);
    println!("Fichiers PE lus : {} ({} .exe)", audit.files.len(), exes.len());
    for ((g, name), (count, missing)) in report::GROUPS.iter().zip(report::totals(&audit)) {
        if *g == "shim" && !audit.shims {
            continue;
        }
        println!("  {name:<36} {count:>6} fonctions, {missing:>5} introuvables");
    }
    let mut absent: Vec<&str> = audit.dlls.iter().filter(|(_, d)| d.origin == "absente").map(|(_, d)| d.name.as_str()).collect();
    absent.sort_unstable();
    println!("DLL absentes du PC : {}", if absent.is_empty() { "aucune".to_owned() } else { absent.join(", ") });
    if exes.is_empty() {
        println!("ATTENTION : aucun .exe dans le dossier, seules les DLL ont ete auditees.");
    }
    println!("Rapport : {}", pyfmt::text(&md));
    Ok(0)
}

/// Line ends as Python's text mode writes them: CRLF on Windows, LF elsewhere.
fn native_lines(text: &str) -> String {
    if cfg!(windows) { text.replace('\n', "\r\n") } else { text.to_owned() }
}
