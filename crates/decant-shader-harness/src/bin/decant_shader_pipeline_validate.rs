//! Validation by pipeline creation: a stronger check than spirv-val alone.
//!
//! spirv-val says whether a module follows the SPIR-V grammar, not whether the DRIVER compiles a
//! pipeline from it. This tool makes the driver create the pipeline (through the bench
//! `decant_shader_run`, which calls vkCreateGraphicsPipelines) for a sample of translated pixel
//! shaders, and counts how many it accepts or refuses. A shared VkPipelineCache (the bench's
//! `--pipeline-cache`) is kept from one run to the next.
//!
//! The bench builds a graphics pipeline (VS_Blit plus the shader as fragment stage), so only pixel
//! shaders are validated at pipeline level here. The other stages are counted apart.
//!
//! Usage: `decant_shader_pipeline_validate <output> [--sample N | --all] [--seed S] [--jobs N]
//! [--timeout S] [--bench <decant_shader_run.exe>]`
//!
//! Classes: `pipeline_ok`, `pipeline_refuse` (vkCreate* refused), `non_traduit`, `saute` (a
//! resource the bench does not make), `autre` (a crash after creation). Everything comes from
//! title files: keep it locally.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use decant_shader_batch::process::{self, Ended};
use decant_shader_bridge::describe::{self, Outcome, Rows, compact, pretty};
use decant_shader_bridge::py;
use decant_shader_bridge::random::Random;
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;
use decant_shader_core::py::{native_newlines, read_text};
use decant_shader_harness::Harness;
use decant_shader_harness::gallery::{VERTEX, float, layout_lines, push_words};
use decant_shader_tools::reflect::reflect;

const USAGE: &str = "usage : decant_shader_pipeline_validate <dossier de sortie> [--sample N | --all] [--seed S] [--jobs N]\n        [--timeout S] [--bench <decant_shader_run.exe>]";

#[derive(Debug)]
struct Args {
    output: PathBuf,
    sample: usize,
    all: bool,
    seed: i64,
    jobs: usize,
    timeout: f64,
    bench: Option<PathBuf>,
}

fn error(message: &str) -> String {
    format!("{USAGE}\ndecant_shader_pipeline_validate: error: {message}")
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args { output: PathBuf::new(), sample: 300, all: false, seed: 7, jobs: 10, timeout: 20.0, bench: None };
    let mut positional = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if !arg.starts_with("--") {
            positional.push(arg.clone());
            continue;
        }
        let (option, inline) = match arg.split_once('=') {
            Some((o, v)) => (o, Some(v.to_owned())),
            None => (arg.as_str(), None),
        };
        if option == "--all" {
            a.all = true;
            continue;
        }
        if option == "--help" {
            return Err(USAGE.into());
        }
        let value = inline.or_else(|| it.next().cloned()).ok_or_else(|| error(&format!("argument {option}: expected one argument")))?;
        let bad = || error(&format!("argument {option}: invalid value: '{value}'"));
        match option {
            "--sample" => a.sample = value.trim().parse().map_err(|_| bad())?,
            "--seed" => a.seed = value.trim().parse().map_err(|_| bad())?,
            "--jobs" => a.jobs = value.trim().parse().map_err(|_| bad())?,
            "--timeout" => a.timeout = value.trim().parse().map_err(|_| bad())?,
            "--bench" => a.bench = Some(PathBuf::from(value)),
            _ => return Err(error(&format!("unrecognized arguments: {arg}"))),
        }
    }
    match positional.as_slice() {
        [output] => a.output = PathBuf::from(output),
        [] => return Err(error("the following arguments are required: output")),
        [_, rest @ ..] => return Err(error(&format!("unrecognized arguments: {}", rest.join(" ")))),
    }
    Ok(a)
}

/// The class of one shader and its detail.
fn outcome(out: &Path, bench: &Path, vertex_spv: &Path, pack_id: i64, d: &Outcome, cache: &Path, timeout: f64) -> Result<(&'static str, String), String> {
    let Some(description) = d.description.as_ref().filter(|_| d.translated) else {
        return Ok(("non_traduit", d.error.clone().unwrap_or_default()));
    };
    let spv = out.join("modules").join(format!("{pack_id}.spv"));
    let reflection = reflect(&std::fs::read(&spv).map_err(|e| format!("{}: {e}", spv.display()))?)?;
    let layout = out.join("layouts").join(format!("{pack_id}.txt"));
    std::fs::write(&layout, native_newlines(&(layout_lines(&reflection)?.join("\n") + "\n"))).map_err(|e| format!("{}: {e}", layout.display()))?;
    let image = out.join("images").join(format!("{pack_id}.f32"));
    let _ = std::fs::remove_file(&image);
    let mut command = Command::new(bench);
    command.arg(vertex_spv).arg(&spv).arg(&image).arg("--layout").arg(&layout).arg("--pipeline-cache").arg(cache);
    for (word, value) in push_words(description)? {
        command.arg("--push").arg(word.to_string()).arg(value.to_string());
    }
    let ran = process::run(&mut command, Duration::from_secs_f64(timeout), true).map_err(|e| format!("{}: {e}", bench.display()))?;
    let code = match ran.ended {
        Ended::Timeout => return Ok(("autre", format!("delai {}s", float(timeout)))),
        Ended::Exit(code) => code,
    };
    let (stdout, stderr) = (read_text(&ran.stdout), read_text(&ran.stderr));
    let shown = if stderr.is_empty() { stdout } else { stderr };
    let message = py::splitlines(py::strip(&shown)).last().map_or(String::new(), |l| (*l).to_owned());
    if code == 5 {
        return Ok(("saute", message));
    }
    let low = message.to_lowercase();
    if ["vkcreategraphicspipelines", "vkcreateshadermodule", "vkcreatepipelinecache"].iter().any(|k| low.contains(k)) {
        return Ok(("pipeline_refuse", message));
    }
    if code == 0 {
        return Ok(("pipeline_ok", String::new()));
    }
    Ok(("autre", format!("code {code} : {message}")))
}

/// Counts in first seen order, as `collections.Counter` keeps them.
fn count<'a>(items: impl IntoIterator<Item = &'a str>) -> Map<String, Value> {
    let mut counts = Map::new();
    for item in items {
        let n = counts.get(item).and_then(Value::as_i64).unwrap_or(0);
        counts.insert(item.to_owned(), json!(n + 1));
    }
    counts
}

fn run(session: &Session, harness: &Harness, args: &Args) -> Result<(), String> {
    let fail = |f: py::Failure| f.stderr().to_owned();
    let out = std::path::absolute(&args.output).map_err(|e| e.to_string())?;
    for sub in ["modules", "layouts", "images"] {
        let dir = out.join(sub);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let bench = args.bench.clone().unwrap_or_else(|| harness.run_bench.clone());
    let cache = out.join("pipeline.cache");
    let rows = Rows::load(&session.config).map_err(fail)?;
    let stage = |m: &Value| m.get("stage").and_then(Value::as_str).unwrap_or_default().to_owned();
    let mut ids: Vec<i64> = rows.meta.iter().filter(|m| stage(m) == "pixel").filter_map(|m| m.get("id").and_then(Value::as_i64)).collect();
    let others: Vec<String> = rows.meta.iter().map(stage).filter(|s| s != "pixel").collect();
    if !args.all {
        ids = Random::seed(args.seed.unsigned_abs()).sample(&ids, args.sample.min(ids.len()));
        ids.sort_unstable();
    }
    let started = Instant::now();
    let modules = out.join("modules");
    if !describe::one(session, &rows, VERTEX, &modules).translated {
        return Err("traduction du shader de sommets VS_Blit impossible".into());
    }
    let descriptions: Vec<Mutex<Option<Outcome>>> = ids.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.jobs.max(1).min(ids.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&pack_id) = ids.get(n) else { break };
                    *descriptions[n].lock().expect("no worker panics") = Some(describe::one(session, &rows, pack_id, &modules));
                }
            });
        }
    });
    let vertex_spv = modules.join(format!("{VERTEX}.spv"));
    let path = out.join("pipeline_validation.jsonl");
    let mut log = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut verdicts = Vec::new();
    for (pack_id, d) in ids.iter().zip(descriptions) {
        let d = d.into_inner().expect("no worker panics").unwrap_or_default();
        let (verdict, detail) = outcome(&out, &bench, &vertex_spv, *pack_id, &d, &cache, args.timeout)?;
        let row = json!({"pack_id": pack_id, "name": d.name, "verdict": verdict, "detail": detail.chars().take(200).collect::<String>()});
        log.write_all(native_newlines(&(compact(&row) + "\n")).as_bytes()).and_then(|()| log.flush()).map_err(|e| e.to_string())?;
        verdicts.push(verdict);
    }
    let summary = json!({
        "pixel_sample": verdicts.len(),
        "counts": count(verdicts.iter().copied()),
        "cache_bytes": std::fs::metadata(&cache).map_or(0, |m| m.len()),
        "stages_not_exercised_here": count(others.iter().map(String::as_str)),
        "seconds": started.elapsed().as_secs_f64().round_ties_even() as i64,
    });
    let path = out.join("pipeline_validation.json");
    std::fs::write(&path, native_newlines(&pretty(&summary))).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{}", compact(&summary));
    Ok(())
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
    let session = match Session::open(Config::from_env()) {
        Ok(s) => s,
        Err(failure) => {
            eprintln!("{failure}");
            return ExitCode::from(1);
        }
    };
    match run(&session, &Harness::from_env(), &args) {
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
    fn arguments() {
        let a = args(&["out"]).unwrap();
        assert_eq!((a.sample, a.all, a.seed, a.jobs, a.timeout, a.bench), (300, false, 7, 10, 20.0, None));
        let a = args(&["out", "--all", "--bench=b.exe", "--timeout", "5"]).unwrap();
        assert_eq!((a.all, a.timeout, a.bench), (true, 5.0, Some("b.exe".into())));
        assert!(args(&[]).is_err());
        assert!(args(&["out", "--jobs", "x"]).unwrap_err().contains("invalid value: 'x'"));
    }

    #[test]
    fn counts_keep_first_seen_order() {
        assert_eq!(compact(&Value::Object(count(["mesh", "vertex", "mesh"]))), r#"{"mesh": 2, "vertex": 1}"#);
    }

    #[test]
    fn an_untranslated_shader_is_not_run() {
        let d = Outcome { translated: false, error: Some("C : opcode".into()), ..Default::default() };
        let r = outcome(Path::new("."), Path::new("missing"), Path::new("v.spv"), 1, &d, Path::new("c"), 1.0).unwrap();
        assert_eq!(r, ("non_traduit", "C : opcode".to_owned()));
    }
}
