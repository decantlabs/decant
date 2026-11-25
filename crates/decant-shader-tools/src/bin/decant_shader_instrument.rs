//! Replaces `src/shaders/decant_shader/tools/decant_shader_instrument.py`: puts a temporary trace into
//! `ResourceTracking.cpp` of the modified translator checkout named by `DECANT_TRANSLATOR_SRC`, or
//! takes it out. `on` keeps a `.decant_bak` copy, `off` restores it as it was. Nothing else in the
//! file should change between the two.
//!
//! Usage: `decant_shader_instrument on|off`

use std::path::PathBuf;
use std::process::ExitCode;

use decant_shader_analysis::NL;
use decant_shader_tools::instrument::{off, on, target};

fn main() -> ExitCode {
    decant_shader_analysis::run(|| {
        let src = std::env::var_os("DECANT_TRANSLATOR_SRC").ok_or("KeyError: 'DECANT_TRANSLATOR_SRC'")?;
        let path = target(&PathBuf::from(src));
        let mode = std::env::args().nth(1).ok_or("IndexError: list index out of range")?;
        let done = if mode == "off" { off(&path)? } else { on(&path)? };
        print!("{done}{NL}");
        Ok(())
    })
}
