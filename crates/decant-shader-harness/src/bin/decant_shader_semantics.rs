//! Runs the semantic cases of `decant_shader_harness::semantics` on the driver.
//!
//! Every case is first ASSEMBLED (spirv-as) and VALIDATED (spirv-val). A case that does not
//! assemble or validate is reported as an invalid case and is NOT counted as a success, so a badly
//! written module shows at once instead of skewing the score. SPIRV-Tools come from `SPIRV_TOOLS`,
//! `VULKAN_SDK` or the `PATH`, as for `spirv_exercise`, and the bench `decant_shader_run` needs the
//! VS_Blit module `442.spv` next to it (the shader gallery writes it).
//!
//! Usage: `decant_shader_semantics [--bench <decant_shader_run.exe>] [--keep <folder>]`
//!
//! Output: one JSON line per case (expected, got, verdict), then a JSON summary. The exit code is
//! 0 only when every case is `ok`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use serde_json::{Map, Value, json};
use decant_shader_bridge::describe::compact;
use decant_shader_harness::Harness;
use decant_shader_harness::semantics::{cases, module_text, read_pixel};
use decant_spirv_exercise::Tools;

const USAGE: &str = "usage : decant_shader_semantics [--bench <decant_shader_run.exe>] [--keep <dossier>]";

fn parse(args: &[String]) -> Result<(Option<PathBuf>, Option<PathBuf>), String> {
    let (mut bench, mut keep) = (None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (option, inline) = match arg.split_once('=') {
            Some((o, v)) => (o, Some(v.to_owned())),
            None => (arg.as_str(), None),
        };
        let mut value = || inline.clone().or_else(|| it.next().cloned()).map(PathBuf::from).ok_or_else(|| format!("{USAGE}\n{option} attend une valeur"));
        match option {
            "--bench" => bench = Some(value()?),
            "--keep" => keep = Some(value()?),
            _ => return Err(USAGE.into()),
        }
    }
    Ok((bench, keep))
}

/// Exit code and the first 200 characters of the trimmed error output (or standard output).
fn run(command: &mut Command) -> (bool, String) {
    match command.stdin(Stdio::null()).output() {
        Ok(o) => {
            let shown = if o.stderr.is_empty() { o.stdout } else { o.stderr };
            (o.status.success(), String::from_utf8_lossy(&shown).trim().chars().take(200).collect())
        }
        Err(e) => (false, e.to_string()),
    }
}

fn case_row(work: &Path, tools: &Tools, bench: &Path, vs: &Path, case: &decant_shader_harness::semantics::Case) -> Value {
    let failed = |verdict: &str, detail: String| json!({"case": case.name, "verdict": verdict, "detail": detail});
    let (asm, spv) = (work.join(format!("{}.spvasm", case.name)), work.join(format!("{}.spv", case.name)));
    if let Err(e) = std::fs::write(&asm, module_text(case)) {
        return failed("cas_invalide_assemblage", e.to_string());
    }
    let (ok, detail) = run(Command::new(&tools.assembler).args(["--target-env", "vulkan1.2"]).arg(&asm).arg("-o").arg(&spv));
    if !ok {
        return failed("cas_invalide_assemblage", detail);
    }
    let (ok, detail) = run(Command::new(&tools.validator).args(["--target-env", "vulkan1.2"]).arg(&spv));
    if !ok {
        return failed("cas_invalide_validation", detail);
    }
    if !vs.exists() {
        return failed("banc_absent", format!("VS_Blit introuvable : {}", vs.display()));
    }
    let image = work.join(format!("{}.f32", case.name));
    let (ok, detail) = run(Command::new(bench).arg(vs).arg(&spv).arg(&image));
    if !ok || !image.exists() {
        return failed("execution_echouee", detail);
    }
    let got = std::fs::read(&image).ok().and_then(|data| read_pixel(&data));
    let verdict = if got == Some(case.expected) { "ok" } else { "FAUX" };
    json!({"case": case.name, "expected": case.expected, "got": got, "verdict": verdict})
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (bench, keep) = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let Some(tools) = Tools::find() else {
        eprintln!("SPIRV-Tools introuvable : renseigne SPIRV_TOOLS ou VULKAN_SDK, ou mets spirv-as/spirv-val dans le PATH");
        return ExitCode::from(2);
    };
    let bench = bench.unwrap_or_else(|| Harness::from_env().run_bench);
    let scratch = match &keep {
        Some(_) => None,
        None => match tempfile::Builder::new().prefix("decant_sem_").tempdir() {
            Ok(dir) => Some(dir),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(1);
            }
        },
    };
    let work = keep.clone().unwrap_or_else(|| scratch.as_ref().expect("a scratch folder without --keep").path().to_path_buf());
    if let Err(e) = std::fs::create_dir_all(&work) {
        eprintln!("{}: {e}", work.display());
        return ExitCode::from(1);
    }
    // VS_Blit: the bench needs it, taken from the shader gallery's output next to the bench.
    let vs = bench.parent().unwrap_or(Path::new(".")).join("442.spv");
    let all = cases();
    let rows: Vec<Value> = all.iter().map(|case| case_row(&work, &tools, &bench, &vs, case)).collect();
    let mut counts = Map::new();
    for row in &rows {
        let verdict = row["verdict"].as_str().unwrap_or_default();
        let n = counts.get(verdict).and_then(Value::as_i64).unwrap_or(0);
        counts.insert(verdict.to_owned(), json!(n + 1));
    }
    for row in &rows {
        println!("{}", compact(row));
    }
    let passed = counts.get("ok").and_then(Value::as_i64).unwrap_or(0) as usize == all.len();
    println!("{}", compact(&json!({"cases": all.len(), "counts": counts})));
    if passed { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments() {
        let list = |l: &[&str]| parse(&l.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(list(&[]).unwrap(), (None, None));
        assert_eq!(list(&["--bench", "b.exe", "--keep=k"]).unwrap(), (Some("b.exe".into()), Some("k".into())));
        assert!(list(&["--other"]).is_err());
        assert!(list(&["--bench"]).is_err());
    }
}
