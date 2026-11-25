//! `decant-pad`: writes commands for the simulated pad of `GameInput.dll` by appending lines to its control file.
//!
//! ```text
//! decant-pad [--file PATH] press A [ms]
//! decant-pad [--file PATH] hold Start 1000
//! decant-pad [--file PATH] stick left 0.0 1.0
//! decant-pad [--file PATH] trigger right 0.5
//! decant-pad [--file PATH] A down | A up | wait 500 | release
//! decant-pad [--file PATH] clear
//! ```
//!
//! The file is `--file` or `DECANT_PAD_SCRIPT`. Every command is checked with the parser of the DLL before it is written.
//! `clear` empties the file, which makes the DLL put the pad back to neutral and read the file again from the start.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::ExitCode;

fn run(mut args: Vec<String>) -> Result<(), String> {
    let mut file = None;
    if args.first().is_some_and(|a| a == "--file") {
        if args.len() < 2 {
            return Err("--file needs a path".to_owned());
        }
        file = Some(PathBuf::from(args.remove(1)));
        args.remove(0);
    }
    let path = file.or_else(|| std::env::var_os("DECANT_PAD_SCRIPT").filter(|v| !v.is_empty()).map(PathBuf::from)).ok_or("no control file: pass --file PATH or set DECANT_PAD_SCRIPT")?;
    if args.is_empty() {
        return Err("no command, see the header of decant-pad.rs or the crate README".to_owned());
    }
    if args[0].eq_ignore_ascii_case("clear") {
        std::fs::write(&path, b"").map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok(());
    }
    let line = args.join(" ");
    let commands = GameInput::pad::parse_line(&line)?;
    if commands.is_empty() {
        return Err("nothing to write".to_owned());
    }
    let mut out = std::fs::OpenOptions::new().read(true).append(true).create(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    // A line a person left without its end must not swallow ours.
    let length = out.metadata().map_err(|e| e.to_string())?.len();
    let mut text = String::new();
    if length > 0 {
        let mut last = [0u8; 1];
        out.seek(SeekFrom::Start(length - 1)).and_then(|_| out.read_exact(&mut last)).map_err(|e| e.to_string())?;
        if last[0] != b'\n' {
            text.push('\n');
        }
    }
    text.push_str(&line);
    text.push('\n');
    out.write_all(text.as_bytes()).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("decant-pad: {why}");
            ExitCode::from(2)
        }
    }
}
