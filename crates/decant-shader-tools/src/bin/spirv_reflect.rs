//! Port of the command line of `src/shaders/decant_shader/tools/spirv_reflect.py`: prints the
//! reflection of one SPIR-V module as JSON indented by one space.
//!
//! Usage: `spirv_reflect <module.spv>`

use std::process::ExitCode;

use decant_shader_analysis::NL;
use decant_shader_tools::pyfmt::json_indent;
use decant_shader_tools::reflect::reflect;

fn main() -> ExitCode {
    decant_shader_analysis::run(|| {
        let path = std::env::args().nth(1).ok_or("IndexError: list index out of range")?;
        let data = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
        print!("{}{NL}", json_indent(&reflect(&data)?.py()).replace('\n', NL));
        Ok(())
    })
}
