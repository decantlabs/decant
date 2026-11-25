//! `decant-shader-ab`: runs two shaders on the GPU with the same inputs and compares what they wrote.
//!
//! ```text
//! decant-shader-ab run A B [--stage ps] [--root-signature rs.bin] [--max-ulp 2] [--json]
//! decant-shader-ab batch pairs.tsv [-j 2] [--timeout 60] [--out results.tsv] [--json-out results.jsonl]
//! decant-shader-ab reflect FILE
//! ```
//!
//! A and B are DXIL containers or HLSL files (`file.hlsl` or `file.hlsl@entry`). The batch list has
//! one pair per line: `hash, path A, path B, stage, root signature`, separated by tabs, the last
//! two optional (`auto` and `-`).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use decant_shader_ab::compare::{Tolerance, Verdict};
use decant_shader_ab::model::Stage;
use decant_shader_ab::pairs::{Outcome, Pair, parse, table};
use decant_shader_ab::plan::{Feed, Format, Options};

#[derive(Parser)]
#[command(name = "decant-shader-ab", about = "Runs shader A and shader B on the GPU with the same inputs and compares their outputs.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// vs, ps, cs or auto (from the containers). HLSL inputs need it.
    #[arg(long, default_value = "auto")]
    stage: String,
    /// A serialized root signature (or a container with an RTS0 part) to run both sides with.
    #[arg(long)]
    root_signature: Option<PathBuf>,
    /// Shader model HLSL inputs are compiled for.
    #[arg(long, default_value = "6_6")]
    sm: String,
    /// Seed of every generated input.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Largest distance in units in the last place that still counts as equal for floats.
    #[arg(long, default_value_t = 0)]
    max_ulp: u32,
    /// Two NaNs with different bits count as different.
    #[arg(long)]
    nan_strict: bool,
    /// Render target width and height.
    #[arg(long, default_value_t = 64)]
    size: u32,
    /// Compute dispatch, `x,y,z` groups.
    #[arg(long, default_value = "2,2,1")]
    groups: String,
    /// Vertices of a vertex pair.
    #[arg(long, default_value_t = 256)]
    points: u32,
    /// Grid cells per side of the mesh that feeds a pixel pair.
    #[arg(long, default_value_t = 4)]
    grid: u32,
    /// Render target format for every target: rgba32f, rgba32u, rgba32i, rgba16f, rgba8.
    #[arg(long)]
    target_format: Option<String>,
    /// How inputs match between the sides: semantic or location.
    #[arg(long, default_value = "semantic")]
    feed: String,
    /// Constant buffer contents from a file: `space:register=path`, repeatable.
    #[arg(long)]
    cbv: Vec<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compares one pair in this process.
    Run {
        a: String,
        b: String,
        #[command(flatten)]
        common: Common,
        /// Prints the outcome as JSON instead of a line.
        #[arg(long)]
        json: bool,
    },
    /// Compares every pair of a TSV list, each side in its own process.
    Batch {
        list: PathBuf,
        #[command(flatten)]
        common: Common,
        /// Pairs run at the same time.
        #[arg(short = 'j', long, default_value_t = 2)]
        jobs: usize,
        /// Seconds one side may take.
        #[arg(long, default_value_t = 60)]
        timeout: u64,
        /// Writes the result lines here too.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Writes one JSON outcome per line here.
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Prints what a shader declares.
    Reflect {
        file: String,
        #[arg(long, default_value = "auto")]
        stage: String,
        #[arg(long, default_value = "6_6")]
        sm: String,
    },
    /// Internal: runs one side of a pair and writes its outputs (used by `batch`).
    #[command(hide = true)]
    Side {
        which: usize,
        a: String,
        b: String,
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        plan: Option<PathBuf>,
    },
}

impl Common {
    fn stage(&self) -> Result<Option<Stage>, String> {
        Stage::parse(&self.stage)
    }

    fn tolerance(&self) -> Tolerance {
        Tolerance { max_ulp: self.max_ulp, nan_equal: !self.nan_strict }
    }

    fn options(&self) -> Result<Options, String> {
        let g: Vec<u32> = self.groups.split(',').map(|s| s.trim().parse::<u32>().map_err(|e| format!("--groups: {e}"))).collect::<Result<_, _>>()?;
        let groups = match g.as_slice() {
            [x] => [*x, 1, 1],
            [x, y] => [*x, *y, 1],
            [x, y, z] => [*x, *y, *z],
            _ => return Err("--groups wants x,y,z".into()),
        };
        let feed = match self.feed.as_str() {
            "semantic" => Feed::Semantic,
            "location" => Feed::Location,
            other => return Err(format!("--feed {other:?}: semantic or location")),
        };
        let mut constants = Vec::new();
        for c in &self.cbv {
            let (at, path) = c.split_once('=').ok_or("--cbv wants space:register=path")?;
            let (space, reg) = at.split_once(':').ok_or("--cbv wants space:register=path")?;
            let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            constants.push((space.parse().map_err(|e| format!("--cbv space: {e}"))?, reg.trim_start_matches('b').parse().map_err(|e| format!("--cbv register: {e}"))?, bytes));
        }
        Ok(Options { seed: self.seed, size: self.size.max(1), groups, points: self.points.max(1), grid: self.grid.max(1), target_format: self.target_format.as_deref().map(Format::parse).transpose()?, feed, constants })
    }

    /// The arguments that give a child process the same options.
    fn forward(&self) -> Vec<String> {
        let mut v = vec!["--sm".into(), self.sm.clone(), "--seed".into(), self.seed.to_string(), "--max-ulp".into(), self.max_ulp.to_string()];
        v.extend(["--size".into(), self.size.to_string(), "--groups".into(), self.groups.clone(), "--points".into(), self.points.to_string(), "--grid".into(), self.grid.to_string(), "--feed".into(), self.feed.clone()]);
        if self.nan_strict {
            v.push("--nan-strict".into());
        }
        if let Some(f) = &self.target_format {
            v.extend(["--target-format".into(), f.clone()]);
        }
        for c in &self.cbv {
            v.extend(["--cbv".into(), c.clone()]);
        }
        v
    }
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("decant-shader-ab: {e}");
            ExitCode::from(3)
        }
    }
}

#[cfg(not(windows))]
fn real_main() -> Result<ExitCode, String> {
    let _ = Cli::parse();
    Err("the GPU runner needs Windows and Direct3D 12".into())
}

#[cfg(windows)]
fn real_main() -> Result<ExitCode, String> {
    use decant_shader_ab::win::{Session, dxc::Dxc, load_shader, prepare, run_pair, run_prepared_side};
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { a, b, common, json } => {
            let dxc = Dxc::load()?;
            let mut session = Session::default();
            let o = run_pair(&mut session, &dxc, "pair", &a, &b, common.stage()?, common.root_signature.as_deref(), &common.sm, common.options()?, common.tolerance());
            if json {
                println!("{}", serde_json::to_string_pretty(&o).map_err(|e| e.to_string())?);
            } else {
                println!("{}", o.line());
                if let Some(d) = &o.diff {
                    println!("compared {} components, {} inexact, {} outside the tolerance, max {} ulp within", d.compared, d.inexact, d.different, d.max_ulp);
                    for m in &d.mismatched {
                        println!("structure: {m}");
                    }
                }
                for d in &o.declarations {
                    println!("declares: {d}");
                }
                for n in &o.notes {
                    println!("note: {n}");
                }
                if !session.adapter.is_empty() {
                    println!("device: {}", session.adapter);
                }
            }
            Ok(ExitCode::from(match o.verdict {
                Verdict::Identical | Verdict::WithinTolerance => 0,
                Verdict::Different => 1,
                _ => 2,
            }))
        }
        Cmd::Side { which, a, b, common, out, plan } => {
            let result = (|| {
                let dxc = Dxc::load()?;
                let p = prepare(&dxc, &a, &b, common.stage()?, common.root_signature.as_deref(), &common.sm, common.options()?)?;
                if let Some(path) = &plan {
                    std::fs::write(path, serde_json::to_vec(&p.plan).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                }
                let mut session = Session::default();
                run_prepared_side(&mut session, &p, which.min(1))
            })();
            std::fs::write(&out, decant_shader_ab::dump::encode(&result)).map_err(|e| format!("{}: {e}", out.display()))?;
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Reflect { file, stage, sm } => {
            let dxc = Dxc::load()?;
            let c = load_shader(&dxc, &file, Stage::parse(&stage)?, &sm)?;
            let r = dxc.reflect(&c)?;
            println!("{}", serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Batch { list, common, jobs, timeout, out, json_out } => {
            let text = std::fs::read_to_string(&list).map_err(|e| format!("{}: {e}", list.display()))?;
            let pairs = parse(&text)?;
            batch(&pairs, &common, jobs.max(1), timeout, out.as_deref(), json_out.as_deref())
        }
    }
}

/// Whether a failure may come from another pair's device loss, so the pair is run again alone.
fn retry_worthy(o: &Outcome) -> bool {
    let bad = |e: &Option<String>| e.as_deref().is_some_and(|e| decant_shader_ab::win::device_lost(e) || e.contains("timeout") || e.contains("process ended"));
    bad(&o.a_error) || bad(&o.b_error)
}

/// Runs one side of `pair` in a child process.
fn side_process(pair: &Pair, which: usize, common: &Common, timeout: u64, dir: &Path, n: usize) -> (Result<Vec<decant_shader_ab::compare::Output>, String>, Option<decant_shader_ab::plan::Plan>) {
    let out = dir.join(format!("{n}-{which}.bin"));
    let plan_path = dir.join(format!("{n}-{which}.json"));
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return (Err(format!("current exe: {e}")), None),
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("side").arg(which.to_string()).arg(&pair.a).arg(&pair.b).arg("--out").arg(&out).arg("--plan").arg(&plan_path);
    cmd.args(common.forward());
    cmd.arg("--stage").arg(pair.stage.map_or("auto", Stage::short));
    if let Some(r) = pair.root.as_deref().map(PathBuf::from).or_else(|| common.root_signature.clone()) {
        cmd.arg("--root-signature").arg(r);
    }
    let ended = decant_shader_batch::process::run(&mut cmd, std::time::Duration::from_secs(timeout), false);
    let plan = std::fs::read(&plan_path).ok().and_then(|b| serde_json::from_slice(&b).ok());
    let result = match ended {
        Err(e) => Err(format!("could not start the side process: {e}")),
        Ok(o) => match o.ended {
            decant_shader_batch::process::Ended::Timeout => Err(format!("timeout after {timeout} s")),
            decant_shader_batch::process::Ended::Exit(code) => match std::fs::read(&out) {
                Ok(bytes) => decant_shader_ab::dump::decode(&bytes),
                Err(_) => Err(format!("process ended with code 0x{code:x} before writing its outputs")),
            },
        },
    };
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&plan_path);
    (result, plan)
}

fn run_one(pair: &Pair, common: &Common, timeout: u64, dir: &Path, n: usize) -> Outcome {
    let (a, plan_a) = side_process(pair, 0, common, timeout, dir, n);
    let (b, plan_b) = side_process(pair, 1, common, timeout, dir, n);
    match plan_a.or(plan_b) {
        Some(plan) => Outcome::from_sides(&pair.hash, &plan, a, b, common.tolerance()),
        None => {
            // Neither side got as far as a plan: the preparation failed for both.
            let mut o = Outcome::skipped(&pair.hash, String::new());
            o.notes.clear();
            o.verdict = Verdict::BothFailed;
            o.a_error = a.err();
            o.b_error = b.err();
            o
        }
    }
}

fn batch(pairs: &[Pair], common: &Common, jobs: usize, timeout: u64, out: Option<&Path>, json_out: Option<&Path>) -> Result<ExitCode, String> {
    let dir = tempfile_dir()?;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<Option<Outcome>>> = std::sync::Mutex::new(vec![None; pairs.len()]);
    let print = std::sync::Mutex::new(());
    std::thread::scope(|s| {
        for _ in 0..jobs.min(pairs.len().max(1)) {
            s.spawn(|| {
                loop {
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some(pair) = pairs.get(n) else { break };
                    let o = run_one(pair, common, timeout, &dir, n);
                    {
                        let _g = print.lock().unwrap_or_else(|e| e.into_inner());
                        println!("{}", o.line());
                    }
                    results.lock().unwrap_or_else(|e| e.into_inner())[n] = Some(o);
                }
            });
        }
    });
    let mut results: Vec<Outcome> = results.into_inner().unwrap_or_else(|e| e.into_inner()).into_iter().flatten().collect();
    if jobs > 1 {
        let again: Vec<usize> = (0..results.len()).filter(|&i| retry_worthy(&results[i])).collect();
        if !again.is_empty() {
            println!("# running {} pairs again alone, a device loss may have come from another pair", again.len());
            for i in again {
                let o = run_one(&pairs[i], common, timeout, &dir, i);
                println!("{}", o.line());
                results[i] = o;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    if let Some(path) = out {
        let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(f, "hash\tstage\tverdict\tdeclares\tdetail").map_err(|e| e.to_string())?;
        for o in &results {
            writeln!(f, "{}", o.line()).map_err(|e| e.to_string())?;
        }
    }
    if let Some(path) = json_out {
        let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        for o in &results {
            writeln!(f, "{}", serde_json::to_string(o).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        }
    }
    println!();
    print!("{}", table(&results));
    Ok(ExitCode::SUCCESS)
}

fn tempfile_dir() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("decant-shader-ab-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}
