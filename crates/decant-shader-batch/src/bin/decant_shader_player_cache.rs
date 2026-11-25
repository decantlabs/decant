//! Builds the ahead-of-time translation cache (`DECANT_SHADER_SPV_CACHE`) on a player's machine, from
//! the player's own game package, by running `extract_shaders`, `decant_shader_metadata`,
//! `decant_shader_batch` and `decant_shader_cache` as separate processes. See [`decant_shader_batch::player`].
//!
//! Usage: `decant_shader_player_cache --package <package dir> --cache <cache root> [--probe <exe>]
//! [--shaders <path>] [--work <dir>] [--tools <dir>] [--plan]`
//!
//! Writes `<cache root>\spv\` (the cache), the empty `adapt\`, `override\` and `scratch\` folders,
//! and `<cache root>\run_env.txt`: the environment a game run needs, one `NAME=value` per line.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use decant_shader_batch::cache::digest;
use decant_shader_batch::player::{self, Freshness, Layout, Options, Step};
use decant_shader_core::py;

fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

fn absolute_options(o: &Options) -> Options {
    let opt = |p: &Option<PathBuf>| p.as_deref().map(absolute);
    Options {
        package: absolute(&o.package),
        cache: absolute(&o.cache),
        probe: opt(&o.probe),
        shaders: opt(&o.shaders),
        work: opt(&o.work),
        tools: opt(&o.tools),
        plan: o.plan,
    }
}

fn print_step(index: usize, step: &Step) {
    println!("[{index}/5] {}", step.name);
    for (name, value) in &step.env {
        println!("      set {name}={}", value.to_string_lossy());
    }
    println!("      {}", step.line());
}

/// Runs one step, logging its duration and exit code. A failure stops the whole run.
fn run_step(index: usize, step: &Step) -> Result<(), String> {
    print_step(index, step);
    let start = Instant::now();
    let status = Command::new(&step.program)
        .args(&step.args)
        .envs(step.env.iter().map(|(k, v)| (k, v)))
        .status()
        .map_err(|e| format!("step {index}/5 {} could not start {}: {e}", step.name, step.program.display()))?;
    let seconds = start.elapsed().as_secs_f64();
    let code = status.code().map_or("none (terminated)".to_owned(), |c| c.to_string());
    println!("      exit code {code}, {seconds:.1} s");
    if status.success() {
        Ok(())
    } else {
        Err(format!("step {index}/5 {} failed (exit code {code} after {seconds:.1} s); stopping, later steps were not run", step.name))
    }
}

fn create(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

fn remove(dir: &Path) -> Result<(), String> {
    if dir.exists() { std::fs::remove_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display())) } else { Ok(()) }
}

fn print_plan(layout: &Layout, steps: &[Step; 5]) {
    println!("Plan (nothing is run):");
    for (i, step) in steps.iter().enumerate() {
        print_step(i + 1, step);
    }
    println!("Steps 4 and 5 are skipped when {} records the SHA-256 of the probe and of {}.", layout.cache_json().display(), layout.pack().display());
    println!("Folders to create: {}, {}, {}, {}", layout.adapt.display(), layout.override_dir.display(), layout.scratch.display(), layout.work.join("decant_shader").display());
    println!("{}:", layout.run_env_file().display());
    print!("{}", player::run_env_text(layout));
}

fn run(layout: &Layout, steps: &[Step; 5]) -> Result<(), String> {
    if !layout.shaders.exists() {
        return Err(format!("shader input not found: {}", layout.shaders.display()));
    }
    if !layout.probe.is_file() {
        return Err(format!("probe not found: {} (pass --probe)", layout.probe.display()));
    }
    for step in steps {
        if !step.program.is_file() {
            return Err(format!("{} not found: {} (pass --tools)", step.name, step.program.display()));
        }
    }
    for name in ["decant_shader_serve", "decant_shader_one", "decant_shader_locate"] {
        let program = player::program(&layout.tools, name);
        if !program.is_file() {
            eprintln!("warning: {} not found; the game run needs it next to decant_shader_serve", program.display());
        }
    }
    // `decant_shader_serve` starts its helpers in `DECANT_SHADERS_DIR\decant_shader`.
    for dir in [&layout.root, &layout.adapt, &layout.override_dir, &layout.scratch, &layout.out, &layout.work.join("decant_shader")] {
        create(dir)?;
    }
    let total = Instant::now();
    run_step(1, &steps[0])?;
    run_step(2, &steps[1])?;
    run_step(3, &steps[2])?;
    let pack_sha256 = digest(&layout.pack())?;
    let probe_sha256 = digest(&layout.probe)?;
    match player::freshness(&layout.cache_json(), &pack_sha256, &probe_sha256) {
        Freshness::Current => {
            println!("[4/5] [5/5] skipped: {} matches the current probe and pack", layout.cache_json().display());
        }
        state => {
            if let Freshness::Stale(why) = state {
                println!("cache out of date ({why}): rebuilding {}", layout.spv.display());
                remove(&layout.spv)?;
                // The adaptation cache goes on every new probe or pack, the container
                // index only when the pack changed.
                remove(&layout.adapt)?;
                create(&layout.adapt)?;
                if why.contains("pack") {
                    let index = layout.container_index();
                    if index.exists() {
                        std::fs::remove_file(&index).map_err(|e| format!("{}: {e}", index.display()))?;
                    }
                }
            }
            // `decant_shader_batch` wants a measure folder that does not exist yet.
            remove(&layout.measure)?;
            run_step(4, &steps[3])?;
            run_step(5, &steps[4])?;
        }
    }
    let path = layout.run_env_file();
    std::fs::write(&path, py::native_newlines(&player::run_env_text(layout))).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("Done in {:.1} s. Cache: {}. Run environment: {}", total.elapsed().as_secs_f64(), layout.spv.display(), path.display());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", player::USAGE);
        return ExitCode::SUCCESS;
    }
    let options = match player::parse(&args) {
        Ok(o) => absolute_options(&o),
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let shaders = options.shaders.clone().unwrap_or_else(|| options.package.clone());
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
    let layout = Layout::new(&options, shaders, &exe_dir);
    let steps = player::plan(&layout);
    if options.plan {
        print_plan(&layout, &steps);
        return ExitCode::SUCCESS;
    }
    match run(&layout, &steps) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
