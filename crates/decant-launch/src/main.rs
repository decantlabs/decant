//! `decant-launch`: stage a GDK package and run its title on Windows with the Decant shims, logging
//! everything. See `src/gdk/README.md`.

use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use decant_launch::{precompile, run, stage, title_executable};

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
    /// Precompile direct shaders into the disk cache from a manifest or shader archives.
    Precompile(PrecompileArgs),
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
struct PrecompileArgs {
    /// The package folder (read only).
    #[arg(long)]
    package: Option<PathBuf>,
    /// The staged overlay folder.
    #[arg(long)]
    overlay: Option<PathBuf>,
    /// Direct shader disk cache folder (default: DECANT_D3D12X_DIRECT_CACHE or %LOCALAPPDATA%\Decant\direct-dxil).
    #[arg(long)]
    cache: Option<PathBuf>,
    /// Pipeline manifest file (default: %LOCALAPPDATA%\Decant\<title-id>\manifest.txt when present).
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Worker threads for decant_shader_warm.
    #[arg(long)]
    threads: Option<usize>,
    /// Show a lightweight native Win32 progress window while warming.
    #[arg(long, alias = "precompile-ui")]
    ui: bool,
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
    /// Precompile direct shaders before starting the title.
    #[arg(long)]
    precompile: bool,
    /// Precompile direct shaders with a lightweight native Win32 progress window before starting the title.
    #[arg(long = "precompile-ui")]
    precompile_ui: bool,
    /// Arguments for the title.
    #[arg(last = true)]
    args: Vec<String>,
}

/// Writes each line to the console and to the log file on a background thread so debug events do not wait on I/O.
struct Both {
    tx: Option<Sender<String>>,
    worker: Option<JoinHandle<()>>,
}

impl Both {
    fn new(file: Option<std::fs::File>) -> Self {
        Self::with_console(file, true)
    }

    fn with_console(file: Option<std::fs::File>, console: bool) -> Self {
        let (tx, rx) = mpsc::channel::<String>();
        let worker = std::thread::Builder::new()
            .name("decant-launch-sink".to_owned())
            .spawn(move || {
                let mut file_writer = file.map(BufWriter::new);
                while let Ok(first) = rx.recv() {
                    let mut batch = Vec::with_capacity(64);
                    batch.push(first);
                    while batch.len() < 512 {
                        match rx.try_recv() {
                            Ok(next) => batch.push(next),
                            Err(_) => break,
                        }
                    }
                    if console {
                        let mut out = BufWriter::new(std::io::stdout().lock());
                        for line in &batch {
                            let _ = writeln!(out, "{line}");
                        }
                        let _ = out.flush();
                    }
                    if let Some(writer) = &mut file_writer {
                        for line in &batch {
                            let _ = writeln!(writer, "{line}");
                        }
                        let _ = writer.flush();
                    }
                }
                if let Some(writer) = &mut file_writer {
                    let _ = writer.flush();
                }
            })
            .ok();
        Self {
            tx: Some(tx),
            worker,
        }
    }
}

impl run::Sink for Both {
    fn line(&mut self, text: &str) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(text.to_owned());
        }
    }
}

impl Drop for Both {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
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

fn do_precompile(a: &PrecompileArgs) -> Result<(), String> {
    let cfg = precompile::PrecompileConfig {
        package: a.package.clone(),
        overlay: a.overlay.clone(),
        cache: a.cache.clone(),
        manifest: a.manifest.clone(),
        threads: a.threads,
        ui: a.ui,
    };
    precompile::precompile(&cfg).map(|_| ())
}

fn do_run(package: Option<&std::path::Path>, overlay: &std::path::Path, o: &RunOptions) -> Result<ExitCode, String> {
    let overlay = std::fs::canonicalize(overlay).map_err(|e| format!("{}: {e}", overlay.display()))?;
    // Without the `\\?\` prefix canonicalize adds, as the loader reports DLL paths.
    let overlay = overlay.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(overlay.clone(), PathBuf::from);
    if o.precompile || o.precompile_ui {
        precompile::precompile(&precompile::PrecompileConfig {
            package: package.map(PathBuf::from),
            overlay: Some(overlay.clone()),
            cache: None,
            manifest: None,
            threads: None,
            ui: o.precompile_ui,
        })?;
    }
    let exe = title_executable(&overlay, o.exe.as_deref())?;
    let log_path = o.log.clone().unwrap_or_else(|| overlay.join("decant_launch.log"));
    let mut sink = Both::new(std::fs::File::create(&log_path).ok());
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
        Command::Precompile(a) => do_precompile(a).map(|()| ExitCode::SUCCESS),
        Command::Run(a) => do_run(None, &a.overlay, &a.options),
        Command::All { stage, run } => do_stage(stage).and_then(|()| do_run(Some(&stage.package), &stage.overlay, run)),
    };
    result.unwrap_or_else(|e| {
        eprintln!("decant-launch: {e}");
        ExitCode::from(2)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn async_sink_delivers_all_lines_in_order_on_drop() {
        let path = std::env::temp_dir().join(format!("decant_launch_both_{}.log", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut sink = Both::with_console(Some(file), false);
            for i in 0..1000 {
                run::Sink::line(&mut sink, &format!("line {i}"));
            }
        }
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 1000);
        assert_eq!(lines.first().copied(), Some("line 0"));
        assert_eq!(lines.last().copied(), Some("line 999"));
        let _ = std::fs::remove_file(path);
    }
}
