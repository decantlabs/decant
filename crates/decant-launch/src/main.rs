//! `decant-launch`: stage a GDK package and run its title on Windows with the Decant shims, logging
//! everything. See `src/gdk/README.md`.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use decant_launch::{run, stage, title_executable};

#[derive(Parser)]
#[command(name = "decant-launch", about = "Stage a GDK package and run its title on Windows with the Decant shims")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Link the package into the overlay folder and copy the shim DLLs there.
    Stage(StageArgs),
    /// Run the staged title under the debugger and log the run.
    Run(RunArgs),
    /// Stage, then run.
    All {
        #[command(flatten)]
        stage: StageArgs,
        #[command(flatten)]
        run: RunOptions,
    },
}

#[derive(Args)]
struct StageArgs {
    /// The package folder (read only; nothing is written there).
    #[arg(long)]
    package: PathBuf,
    /// The writable folder the title runs from.
    #[arg(long)]
    overlay: PathBuf,
    /// A folder of shim DLLs (xmem.dll, d3d12_xs.dll...), copied into the overlay.
    #[arg(long)]
    shims: Option<PathBuf>,
    /// A folder with vkd3d-proton's d3d12.dll and d3d12core.dll (default: env DECANT_VKD3D_DIR). Without it
    /// the title keeps the native D3D12.
    #[arg(long)]
    vkd3d: Option<PathBuf>,
}

#[derive(Args)]
struct RunArgs {
    /// The staged overlay folder.
    #[arg(long)]
    overlay: PathBuf,
    #[command(flatten)]
    options: RunOptions,
}

#[derive(Args)]
struct RunOptions {
    /// The executable to run, instead of the config's primary executable.
    #[arg(long)]
    exe: Option<String>,
    /// Where the log goes (default: decant_launch.log in the overlay).
    #[arg(long)]
    log: Option<PathBuf>,
    /// Stop the title after this many seconds.
    #[arg(long)]
    timeout: Option<u64>,
    /// Run the title at below normal CPU and GPU scheduling priority (same as DECANT_GENTLE=1).
    #[arg(long)]
    gentle: bool,
    /// Arguments for the title.
    #[arg(last = true)]
    args: Vec<String>,
}

/// Writes each line to the console and to the log file.
struct Both(Option<std::fs::File>);

impl run::Sink for Both {
    fn line(&mut self, text: &str) {
        println!("{text}");
        if let Some(file) = &mut self.0 {
            let _ = writeln!(file, "{text}");
        }
    }
}

fn do_stage(a: &StageArgs) -> Result<(), String> {
    let vkd3d = a.vkd3d.clone().or_else(|| std::env::var_os("DECANT_VKD3D_DIR").map(PathBuf::from));
    let vkd3d = vkd3d.as_deref().filter(|p| !p.as_os_str().is_empty());
    let made = stage::stage_with_renderer(&a.package, &a.overlay, a.shims.as_deref(), vkd3d).map_err(|e| e.to_string())?;
    if let Some(dir) = vkd3d {
        for name in stage::VKD3D_DLLS {
            let sha = stage::sha256_file(&dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
            println!("vkd3d {name} sha256 {sha}");
        }
    }
    for entry in &made {
        println!("{:?}\t{}", entry.made, entry.name);
    }
    println!("{} entries staged in {}", made.len(), a.overlay.display());
    Ok(())
}

fn do_run(overlay: &std::path::Path, o: &RunOptions) -> Result<ExitCode, String> {
    let overlay = std::fs::canonicalize(overlay).map_err(|e| format!("{}: {e}", overlay.display()))?;
    // Without the `\\?\` prefix canonicalize adds, as the loader reports DLL paths.
    let overlay = overlay.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(overlay.clone(), PathBuf::from);
    let exe = title_executable(&overlay, o.exe.as_deref())?;
    let log_path = o.log.clone().unwrap_or_else(|| overlay.join("decant_launch.log"));
    let mut sink = Both(std::fs::File::create(&log_path).ok());
    run::Sink::line(&mut sink, &format!("decant-launch {}: {} from {}", env!("CARGO_PKG_VERSION"), exe.display(), overlay.display()));
    for line in decant_launch::runtime::report() {
        run::Sink::line(&mut sink, &line);
    }
    let mut env = vec![("DECANT_LOG".to_owned(), "1".to_owned())];
    env.extend(stage::renderer_env(&overlay, |k| std::env::var(k).ok()));
    if stage::has_vkd3d(&overlay) {
        run::Sink::line(&mut sink, "renderer: vkd3d-proton (d3d12.dll and d3d12core.dll staged in the overlay)");
    }
    if o.gentle {
        env.push((decant_launch::gentle::ENV.to_owned(), "1".to_owned()));
    }
    let outcome = run::run(&exe, &o.args, &overlay, &env, o.timeout.map(Duration::from_secs), &mut sink).map_err(|e| e.to_string())?;
    let ours: Vec<_> = outcome.dlls.iter().filter(|p| p.starts_with(&overlay)).collect();
    run::Sink::line(&mut sink, &format!("{} DLLs loaded, {} from the title folder; log: {}", outcome.dlls.len(), ours.len(), log_path.display()));
    Ok(match outcome.exit_code {
        Some(0) => ExitCode::SUCCESS,
        _ => ExitCode::FAILURE,
    })
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Stage(a) => do_stage(a).map(|()| ExitCode::SUCCESS),
        Command::Run(a) => do_run(&a.overlay, &a.options),
        Command::All { stage, run } => do_stage(stage).and_then(|()| do_run(&stage.overlay, run)),
    };
    result.unwrap_or_else(|e| {
        eprintln!("decant-launch: {e}");
        ExitCode::from(2)
    })
}
