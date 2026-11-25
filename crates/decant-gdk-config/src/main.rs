//! `gdk_config_reader`, the command line of the former `src/gdk/gdk_config_reader.cpp`, used by
//! `src/gdk/launch_prep.sh` and by hand:
//!
//! * `gdk_config_reader [--exec] CONFIG` prints the primary executable: the first ExecutableList
//!   entry without IsDevOnly (dev only tools never launch). Exit 1 when there is none.
//! * `gdk_config_reader --pe-check FILE` runs the PE pre-flight on the staged title executable.
//!   Exit 0 means it passed.
//!
//! Exit 2 means bad usage. Errors go to stderr and results to stdout, so callers can capture the
//! name safely. Lines end as the C runtime ends them: `\r\n` on Windows, `\n` elsewhere.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const NL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

fn main() -> ExitCode {
    let mut config = None;
    let mut pe = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        match text.as_ref() {
            "--exec" => continue, // default mode, the path is positional
            "--pe-check" => match args.next() {
                Some(path) => pe = Some(PathBuf::from(path)),
                None => return usage(),
            },
            "--help" | "-h" => return usage(),
            t if t.starts_with('-') && t != "-" => return usage(),
            _ if config.is_some() || pe.is_some() => return usage(), // one path only
            _ => config = Some(PathBuf::from(arg)),
        }
    }

    if let Some(pe) = pe {
        return match decant_gdk_config::check_pe_image(&pe) {
            Ok(()) => out(&format!("ok: {} is a plain x86-64 PE (GUI/CUI subsystem)", pe.display())),
            Err(e) => fail(&e),
        };
    }
    let Some(config) = config else { return usage() };
    let info = match decant_gdk_config::read_gdk_config(&config) {
        Ok(info) => info,
        Err(e) => return fail(&format!("gdk_config_reader: {e}")),
    };
    match info.executables.iter().find(|e| !e.is_dev_only) {
        Some(exe) => out(&exe.name),
        None => fail(&format!(
            "gdk_config_reader: {}: every ExecutableList entry is IsDevOnly; nothing to launch (set DECANT_EXE_NAME to override)",
            config.display()
        )),
    }
}

fn usage() -> ExitCode {
    let text = [
        "usage: gdk_config_reader [--exec] <MicrosoftGame.config>",
        "       print the primary (non-dev-only) executable name",
        "       gdk_config_reader --pe-check <file>",
        "       PE pre-flight: DOS+PE headers, x86-64, GUI/CUI (exit 0 = ok)",
    ];
    let _ = std::io::stderr().write_all(text.map(|l| format!("{l}{NL}")).concat().as_bytes());
    ExitCode::from(2)
}

fn out(line: &str) -> ExitCode {
    let _ = std::io::stdout().write_all(format!("{line}{NL}").as_bytes());
    ExitCode::SUCCESS
}

fn fail(line: &str) -> ExitCode {
    let _ = std::io::stderr().write_all(format!("{line}{NL}").as_bytes());
    ExitCode::from(1)
}
