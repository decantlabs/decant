//! Builds the offline adaptation cache the bridge reads through `DECANT_SHADER_ADAPT_CACHE`. See
//! `decant_shader_bridge::adapt_cache`.
//!
//! Usage: `decant_shader_adapt_cache <capture> <cache> [--jobs N] [--limit N]`
//!
//! * `<capture>`: the `<hash>.dxbc` and `<hash>.layout` pairs the game and vkd3d-proton write to
//!   `VKD3D_DECANT_SHADER_DIR` during a session (the menu, for example). Containers without console
//!   code are skipped.
//! * `<cache>`: created or completed, one entry per distinct `(shader, layout)`, plus
//!   `adapt_cache.json`.
//!
//! With `DECANT_SHADER_SPV_CACHE` set to the translation cache, the probe does not run again. With
//! `DECANT_SHADER_SPIRV_OPT`, each entry goes through spirv-opt before it is published. Everything in
//! the cache comes from title files: keep it locally, never in the repository.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use decant_shader_bridge::adapt_cache::{build_one, entries, summary};
use decant_shader_bridge::describe::{compact, pretty};
use decant_shader_bridge::py::{self, Failure};
use decant_shader_bridge::serve::container_index;
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;

const USAGE: &str = "usage : decant_shader_adapt_cache <capture> <cache> [--jobs N] [--limit N]";

#[derive(Debug)]
struct Args {
    capture: PathBuf,
    cache: PathBuf,
    jobs: usize,
    limit: usize,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let (mut positional, mut jobs, mut limit) = (Vec::new(), 8, 0);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_owned())),
            _ => (arg.as_str(), None),
        };
        let mut number = || -> Result<usize, String> {
            let v = inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| format!("{USAGE}\n{name} attend une valeur"))?;
            v.trim().parse().map_err(|_| format!("{USAGE}\n{name} : nombre invalide {}", py::repr_str(&v)))
        };
        match name {
            "--jobs" => jobs = number()?,
            "--limit" => limit = number()?,
            "-h" | "--help" => return Err(USAGE.into()),
            _ if arg.starts_with("--") => return Err(format!("{USAGE}\noption inconnue : {arg}")),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    let [capture, cache] = <[PathBuf; 2]>::try_from(positional).map_err(|_| USAGE.to_owned())?;
    Ok(Args { capture, cache, jobs: jobs.max(1), limit })
}

fn run(args: &Args) -> Result<(), Failure> {
    let session = Session::open(Config::from_env())?;
    std::fs::create_dir_all(&args.cache).map_err(|e| Failure::io(&e, &args.cache))?;
    let capture = std::path::absolute(&args.capture).map_err(|e| Failure::io(&e, &args.capture))?;
    let mut rows = entries(&capture, &container_index(&session)?)?;
    if args.limit > 0 {
        rows.truncate(args.limit);
    }
    let outcomes: Vec<Mutex<String>> = rows.iter().map(|_| Mutex::default()).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.jobs.min(rows.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    let Some(row) = rows.get(n) else { break };
                    *outcomes[n].lock().expect("no worker panics") = build_one(&session, row, &args.cache);
                }
            });
        }
    });
    let outcomes: Vec<String> = outcomes.into_iter().map(|o| o.into_inner().expect("no worker panics")).collect();
    let _ = std::fs::remove_dir_all(args.cache.join(".work"));
    let info = summary(&args.cache, rows.len(), &outcomes);
    let path = args.cache.join("adapt_cache.json");
    std::fs::write(&path, pretty(&info)).map_err(|e| Failure::io(&e, &path))?;
    let shown = serde_json::json!({"entries": info["entries"], "distinct_pairs": info["distinct_pairs"], "outcomes": info["outcomes"]});
    println!("{}", compact(&shown));
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
    match run(&args) {
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
    fn arguments() {
        let a = args(&["cap", "cache"]).unwrap();
        assert_eq!((a.capture, a.cache, a.jobs, a.limit), ("cap".into(), "cache".into(), 8, 0));
        let a = args(&["cap", "--jobs", "3", "cache", "--limit=5"]).unwrap();
        assert_eq!((a.jobs, a.limit), (3, 5));
        assert_eq!(args(&["cap", "cache", "--jobs", "0"]).unwrap().jobs, 1);
        assert!(args(&["cap"]).is_err());
        assert!(args(&["cap", "cache", "--jobs", "x"]).unwrap_err().contains("nombre invalide 'x'"));
        assert!(args(&["cap", "cache", "--other"]).is_err());
    }
}
