//! Port of `src/shaders/decant_shader/tools/decant_shader_capabilities.py`: counts, in the SPIR-V modules of a
//! measurement, those that rely on an emulator mechanism (physical addresses, image tables and so
//! on), and the most frequent bindings.
//!
//! Usage: `decant_shader_capabilities <measurement folder> [...]`

use std::process::ExitCode;

fn main() -> ExitCode {
    decant_shader_analysis::run(|| {
        for folder in std::env::args().skip(1) {
            print!("{}", decant_shader_tools::capabilities::folder(&folder)?);
        }
        Ok(())
    })
}
