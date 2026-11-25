//! Ties every resource of a translated shader to its Direct3D register, as
//! `src/shaders/decant_shader/tools/decant_shader_identify.py` does, with the same output.
//!
//! Usage:
//! * `decant_shader_identify <pack number> [...]`: detail for a few shaders
//! * `decant_shader_identify --mesure <n> <results.jsonl>`: identification rate on `n` valid shaders
//!   drawn at random (the same draw as the Python)
//!
//! Each translation runs in a scratch folder of this call only, outside the repository, removed
//! at the end even on failure.

use std::path::Path;
use std::process::ExitCode;

use decant_shader_bridge::identify::{self, Measure};
use decant_shader_bridge::py::{self, Failure};
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::{self, Config};

fn run(session: &Session, args: &[String], scratch: &Path) -> Result<(), Failure> {
    if args[0] != "--mesure" {
        for arg in args {
            let pack_id = py::int(arg)?;
            let lines = identify::sources(&session.config, pack_id, scratch)?;
            println!("== {pack_id} {}", session.name(pack_id)?);
            for r in session.identify(pack_id, lines.as_deref().unwrap_or_default())? {
                println!("{}", identify::detail_line(&r));
            }
        }
        return Ok(());
    }
    let missing = || Failure::error("IndexError", "list index out of range");
    let count = py::int(args.get(1).ok_or_else(missing)?)?;
    let path = Path::new(args.get(2).ok_or_else(missing)?);
    let text = std::fs::read_to_string(path).map_err(|e| Failure::io(&e, path))?;
    let valid = identify::valid_shaders(&shaders::json_lines(&text)?)?;
    let mut measure = Measure::default();
    for pack_id in identify::draw(&valid, count)? {
        let Some(lines) = identify::sources(&session.config, pack_id, scratch)? else {
            measure.untranslated();
            continue;
        };
        let resources = session.identify(pack_id, &lines)?;
        measure.add(&resources, &session.heap_flags(pack_id)?);
    }
    for line in measure.report() {
        println!("{line}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = Session::open(Config::from_env()).and_then(|session| {
        if args.is_empty() {
            return Err(Failure::Exit("usage : decant_shader_identify <numero dans le pack> [...] | --mesure <n> <results.jsonl>".into()));
        }
        let scratch = tempfile::Builder::new().prefix("decant_shader_identify_").tempdir().map_err(|e| Failure::io(&e, &std::env::temp_dir()))?;
        let result = run(&session, &args, scratch.path());
        let _ = std::fs::remove_dir_all(scratch.path());
        result
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("{failure}");
            ExitCode::from(1)
        }
    }
}
