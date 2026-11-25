//! Replays one shader (or a few) in the DecantShader probe with the `DECANT_SHADER_TRACE` trace, as
//! `src/shaders/decant_shader/tools/decant_shader_one.py` does.
//!
//! Usage: `decant_shader_one <output folder> <name or pack number> [...]`
//!
//! The output folder gets `one.pack`, `one.pack.inputs`, `one.pack.vinputs`, `one.tsv` and the
//! modules. The pack and metadata come from `src/shaders/out`, the probe from
//! `src/shaders/decant_shader` or `DECANT_SHADER_PROBE` (see `decant_shader_core::paths`).

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::time::Duration;

use decant_extract_shaders::pack;
use decant_shader_batch::{batch, one, process};
use decant_shader_core::{meta, paths};

fn remove(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("{}: {e}", path.display())),
        _ => Ok(()),
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let Some((out, wanted)) = args.split_first() else {
        return Err("usage: decant_shader_one <dossier de sortie> <nom> [<nom> ...]".into());
    };
    let out = std::path::absolute(out).map_err(|e| format!("{out}: {e}"))?;
    let (names, ids) = one::select(wanted)?;
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let meta = meta::rows(&paths::out().join("decant_shader_metadata.jsonl"))?;
    let pack_path = paths::out().join("shaders_gpu.pack");
    let (_, found) = pack::read(&pack_path, |index, name, _| names.contains(name) || ids.contains(&(index as i64)))
        .map_err(|e| format!("{}: {e}", pack_path.display()))?;
    let probe = paths::probe();
    let stdout = std::io::stdout();
    for record in &found {
        let one_pack = out.join("one.pack");
        batch::write_inputs(&one_pack, std::slice::from_ref(record), Some(&meta))?;
        let log = out.join("one.tsv");
        remove(&log)?;
        remove(&out.join("0.spv"))?;
        let mut command = Command::new(&probe);
        command.arg(&one_pack).arg("--stage").arg(&log).arg("0").arg(record.stage.to_string()).arg(&out).arg("1");
        command.env("DECANT_SHADER_TRACE", "1");
        let result = process::run(&mut command, Duration::from_secs(120), true).map_err(|e| format!("{}: {e}", probe.display()))?;
        let code = match result.ended {
            process::Ended::Exit(code) => code,
            process::Ended::Timeout => return Err(format!("TimeoutExpired: {} timed out after 120 seconds", probe.display())),
        };
        let text = one::trace(record.index, &record.name, record.stage, code, &result.stderr);
        stdout.lock().write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        let log_bytes = fs::read(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        stdout.lock().write_all(one::log(&log_bytes).as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
