//! Puts the inputs and outputs of a DecantShader SPIR-V module at the locations Direct3D 12 expects,
//! as `src/shaders/decant_shader/tools/decant_shader_locate.py` does. See [`decant_shader_core::locate`].
//!
//! Usage: `decant_shader_locate <shader number in the pack> <module.spv> <output.spv>`

use std::path::Path;
use std::process::ExitCode;

use decant_shader_core::{locate, paths, py, spirv};

fn run(args: &[String]) -> Result<(), String> {
    let [id, module, output, ..] = args else {
        return Err("usage: decant_shader_locate <numero du shader dans le pack> <module.spv> <sortie.spv>".into());
    };
    let (inputs, outputs) = locate::tables_for(&paths::out(), py::int(id)?)?;
    let module = std::fs::read(module).map_err(|e| format!("{module}: {e}"))?;
    std::fs::write(Path::new(output), spirv::locate(&module, &inputs, &outputs)?).map_err(|e| format!("{output}: {e}"))?;
    println!("{}", locate::report(&inputs, &outputs));
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
