//! Links the outputs of a vertex shader to the inputs of a pixel shader by semantic and rewrites
//! the pixel module, as `src/shaders/decant_shader/tools/decant_shader_link.py` does. See
//! [`decant_shader_core::link`].
//!
//! Usage: `decant_shader_link <vertex shader number> <pixel shader number> <pixel.spv> <output.spv>`

use std::path::Path;
use std::process::ExitCode;

use decant_shader_core::{link, paths, py, spirv};

fn run(args: &[String]) -> Result<(), String> {
    let [vertex, pixel, module, output, ..] = args else {
        return Err("usage: decant_shader_link <numero du shader de sommets> <numero du shader de pixels> <pixels.spv> <sortie.spv>".into());
    };
    let table = link::mapping_for(&paths::out(), py::int(vertex)?, py::int(pixel)?)?;
    let module = std::fs::read(module).map_err(|e| format!("{module}: {e}"))?;
    std::fs::write(Path::new(output), spirv::link(&module, &table)?).map_err(|e| format!("{output}: {e}"))?;
    println!("{}", link::report(&table));
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
