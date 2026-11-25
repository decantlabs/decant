//! The shader gallery, as `src/shaders/decant_shader/harness/decant_shader_gallery.py` runs it: executes
//! thousands of translated pixel shaders on the GPU and sorts the result. See
//! `decant_shader_harness::gallery`.
//!
//! Classes: `ok`, `zero`, `nan`, `sans_couleur`, `rejete`, `pipeline`, `plantage`, `delai`,
//! `saute`, `non_traduit`, and `appareil_perdu` for shaders given with `--skip`. After a lost
//! device the gallery waits 10 s, and it stops at the third one in two minutes.
//!
//! Usage: `decant_shader_gallery <output> [--sample N | --all] [--seed S] [--jobs N] [--timeout S]
//! [--no-robust] [--keep-images] [--skip list] [--resume]`. The output gets `modules/`,
//! `layouts/`, `images/` (removed unless `--keep-images`), `results.jsonl` and `summary.json`.
//! Everything comes from title files: keep it locally.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use decant_shader_bridge::describe::{self, Outcome, Rows};
use decant_shader_bridge::py;
use decant_shader_bridge::random::Random;
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;
use decant_shader_harness::Harness;
use decant_shader_harness::gallery::{self, Bench, VERTEX, dumps};

const USAGE: &str = "usage: decant_shader_gallery [-h] [--sample SAMPLE] [--all] [--seed SEED] [--jobs JOBS] [--timeout TIMEOUT] [--no-robust]\n                        [--keep-images] [--skip SKIP] [--resume]\n                        output";
const OPTIONS: [&str; 10] = ["--help", "--sample", "--all", "--seed", "--jobs", "--timeout", "--no-robust", "--keep-images", "--skip", "--resume"];

#[derive(Debug, PartialEq)]
struct Args {
    output: PathBuf,
    sample: i64,
    all: bool,
    seed: i64,
    jobs: i64,
    timeout: f64,
    no_robust: bool,
    keep_images: bool,
    skip: String,
    resume: bool,
}

fn usage_error(message: &str) -> String {
    format!("{USAGE}\ndecant_shader_gallery: error: {message}")
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args { output: PathBuf::new(), sample: 300, all: false, seed: 7, jobs: 10, timeout: 10.0, no_robust: false, keep_images: false, skip: String::new(), resume: false };
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
        let exact = OPTIONS.contains(&name);
        let matches: Vec<&str> = if name == "-h" {
            vec!["--help"]
        } else if exact {
            vec![name]
        } else {
            OPTIONS.iter().copied().filter(|o| o.starts_with(name) && name.len() > 2).collect()
        };
        let option = match matches.as_slice() {
            [one] => *one,
            [] => return Err(usage_error(&format!("unrecognized arguments: {arg}"))),
            many => return Err(usage_error(&format!("ambiguous option: {name} could match {}", many.join(", ")))),
        };
        let mut value = || inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| usage_error(&format!("argument {option}: expected one argument")));
        let int = |option: &str, v: String| py::int(&v).map_err(|_| usage_error(&format!("argument {option}: invalid int value: {}", py::repr_str(&v))));
        match option {
            "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--sample" => a.sample = int(option, value()?)?,
            "--all" => a.all = true,
            "--seed" => a.seed = int(option, value()?)?,
            "--jobs" => a.jobs = int(option, value()?)?,
            "--timeout" => {
                let v = value()?;
                a.timeout = v.trim().parse().map_err(|_| usage_error(&format!("argument --timeout: invalid float value: {}", py::repr_str(&v))))?;
            }
            "--no-robust" => a.no_robust = true,
            "--keep-images" => a.keep_images = true,
            "--skip" => a.skip = value()?,
            "--resume" => a.resume = true,
            _ => unreachable!(),
        }
    }
    match positional.as_slice() {
        [output] => a.output = PathBuf::from(output),
        [] => return Err(usage_error("the following arguments are required: output")),
        [_, rest @ ..] => return Err(usage_error(&format!("unrecognized arguments: {}", rest.join(" ")))),
    }
    Ok(a)
}

/// `round(x)`: to the nearest integer, halves to even.
fn round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn run(session: &Session, harness: &Harness, args: &Args) -> Result<(), String> {
    let fail = |f: py::Failure| f.stderr().to_owned();
    let out = std::path::absolute(&args.output).map_err(|e| e.to_string())?;
    for sub in ["modules", "layouts", "images"] {
        let dir = out.join(sub);
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let rows = Rows::load(&session.config).map_err(fail)?;
    py::index(&rows.meta, 0).map_err(fail)?;
    let mut ids: Vec<i64> = Vec::new();
    for m in &rows.meta {
        if m.get("stage").ok_or("KeyError: 'stage'")? == "pixel" {
            ids.push(m.get("id").and_then(Value::as_i64).ok_or("KeyError: 'id'")?);
        }
    }
    if !args.all {
        let k = args.sample.min(ids.len() as i64);
        if k < 0 {
            return Err("ValueError: Sample larger than population or is negative".into());
        }
        ids = Random::seed(args.seed.unsigned_abs()).sample(&ids, k as usize);
        ids.sort_unstable();
    }
    let mut skip = HashSet::new();
    for x in args.skip.split(',') {
        if !py::strip(x).is_empty() {
            skip.insert(py::int(x).map_err(fail)?);
        }
    }
    let results_path = out.join("results.jsonl");
    let mut done: HashMap<i64, Value> = HashMap::new();
    if args.resume && results_path.exists() {
        let text = fs::read(&results_path).map_err(|e| format!("{}: {e}", results_path.display()))?;
        for line in py::splitlines(&String::from_utf8_lossy(&text).replace("\r\n", "\n")) {
            if !py::strip(line).is_empty() {
                let row: Value = serde_json::from_str(line).map_err(|e| format!("json.decoder.JSONDecodeError: {e}"))?;
                let id = row.get("pack_id").and_then(Value::as_i64).ok_or("KeyError: 'pack_id'")?;
                done.insert(id, row);
            }
        }
    }
    let started = Instant::now();
    let modules = out.join("modules");
    let vertex = describe::one(session, &rows, VERTEX, &modules);
    if !vertex.translated {
        return Err("traduction du shader de sommets impossible".into());
    }
    let descriptions: Vec<Mutex<Option<Outcome>>> = ids.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.jobs.max(1).min(ids.len().max(1) as i64) {
            scope.spawn(|| {
                loop {
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    let Some(&pack_id) = ids.get(n) else { break };
                    *descriptions[n].lock().expect("no worker panics") = Some(describe::one(session, &rows, pack_id, &modules));
                }
            });
        }
    });
    let translated_at = Instant::now();
    let vertex_spv = modules.join(format!("{VERTEX}.spv"));
    let bench = Bench { program: &harness.run_bench, out: &out, vertex_spv: &vertex_spv, timeout: args.timeout, robust: !args.no_robust };
    let mut results: Vec<Value> = Vec::new();
    let mut lost: VecDeque<Instant> = VecDeque::new();
    let mut log = fs::File::create(&results_path).map_err(|e| format!("{}: {e}", results_path.display()))?;
    let mut write = |row: &Value| log.write_all(py::native_lines(&(dumps(row) + "\n")).as_bytes()).and_then(|()| log.flush()).map_err(|e| e.to_string());
    for (pack_id, d) in ids.iter().zip(descriptions) {
        let d = d.into_inner().expect("no worker panics").unwrap_or_default();
        if let Some(row) = done.get(pack_id) {
            results.push(row.clone());
            write(row)?;
            continue;
        }
        let (verdict, detail) = if skip.contains(pack_id) {
            ("appareil_perdu".to_owned(), "saute (--skip)".to_owned())
        } else if !d.translated {
            ("non_traduit".to_owned(), d.error.clone().unwrap_or_default())
        } else {
            gallery::run_one(&bench, *pack_id, d.description.as_ref().unwrap_or(&Value::Null))?
        };
        let mut row = Map::new();
        row.insert("pack_id".into(), json!(pack_id));
        row.insert("name".into(), d.name.clone());
        row.insert("verdict".into(), json!(verdict));
        row.insert("detail".into(), json!(detail.chars().take(300).collect::<String>()));
        let row = Value::Object(row);
        results.push(row.clone());
        write(&row)?;
        if verdict == "plantage" && (detail.contains("-4") || detail.to_uppercase().contains("DEVICE_LOST")) {
            let now = Instant::now();
            lost.push_back(now);
            while lost.front().is_some_and(|t| now.duration_since(*t) > Duration::from_secs(120)) {
                lost.pop_front();
            }
            if lost.len() >= 3 {
                println!("trois appareils perdus en deux minutes : arret par prudence");
                break;
            }
            std::thread::sleep(Duration::from_secs(10));
        }
    }
    if !args.keep_images {
        let _ = fs::remove_dir_all(out.join("images"));
    }
    let mut counts: Vec<(String, i64)> = Vec::new();
    for r in &results {
        let verdict = r.get("verdict").map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned)).ok_or("KeyError: 'verdict'")?;
        match counts.iter_mut().find(|(v, _)| *v == verdict) {
            Some((_, n)) => *n += 1,
            None => counts.push((verdict, 1)),
        }
    }
    let mut counted = Map::new();
    for (verdict, n) in counts {
        counted.insert(verdict, json!(n));
    }
    let mut summary = Map::new();
    summary.insert("shaders".into(), json!(results.len()));
    summary.insert("robust".into(), json!(!args.no_robust));
    summary.insert("timeout".into(), json!(args.timeout));
    summary.insert("counts".into(), Value::Object(counted));
    summary.insert("translate_seconds".into(), json!(round(translated_at.duration_since(started).as_secs_f64())));
    summary.insert("run_seconds".into(), json!(round(translated_at.elapsed().as_secs_f64())));
    let summary = Value::Object(summary);
    let path = out.join("summary.json");
    fs::write(&path, py::native_lines(&describe::pretty(&summary))).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("{}", dumps(&summary));
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // The Python opens the archive when it imports decant_shader_describe, before reading its arguments.
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
    fn arguments_parse_like_argparse() {
        let a = args(&["out", "--sample", "50", "--seed=3", "--timeout", "2.5", "--no-r", "--keep", "--skip", "1, 2", "--resume"]).unwrap();
        assert_eq!((a.sample, a.seed, a.timeout, a.no_robust, a.keep_images, a.skip.as_str(), a.resume), (50, 3, 2.5, true, true, "1, 2", true));
        let a = args(&["out", "--all"]).unwrap();
        assert_eq!((a.all, a.sample, a.seed, a.jobs, a.timeout), (true, 300, 7, 10, 10.0));
        assert!(args(&["out", "--s", "1"]).unwrap_err().contains("ambiguous option: --s could match --sample, --seed, --skip"));
        assert!(args(&["out", "--timeout", "x"]).unwrap_err().ends_with("invalid float value: 'x'"));
        assert!(args(&[]).unwrap_err().ends_with("required: output"));
        assert_eq!((round(2.5), round(3.5), round(0.4)), (2, 4, 0));
    }
}
