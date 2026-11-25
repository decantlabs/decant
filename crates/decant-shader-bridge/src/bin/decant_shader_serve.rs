//! Called by vkd3d-proton (the build with the DecantShader hook, variable `VKD3D_DECANT_SHADER_COMMAND`)
//! when it creates a pipeline, as `src/shaders/decant_shader/harness/decant_shader_serve.py` is: same
//! arguments, same environment variables, same `<hash>.spv` in `VKD3D_SHADER_OVERRIDE` and the
//! same `<hash>.log` journal in `VKD3D_DECANT_SHADER_DIR`.
//!
//! Usage: `decant_shader_serve <hexadecimal hash>`
//!
//! Prototype, as in the Python: one process per shader. A real integration would do this work in
//! the game's process.

use std::path::PathBuf;
use std::process::ExitCode;

use decant_shader_bridge::py;
use decant_shader_bridge::serve;
use decant_shader_bridge::shaders::Config;

fn main() -> ExitCode {
    let Some(name) = std::env::args().nth(1) else {
        eprintln!("IndexError: list index out of range");
        return ExitCode::from(1);
    };
    let mut dirs = Vec::new();
    for variable in ["VKD3D_DECANT_SHADER_DIR", "VKD3D_SHADER_OVERRIDE"] {
        match std::env::var_os(variable) {
            Some(dir) => dirs.push(PathBuf::from(dir)),
            None => {
                eprintln!("KeyError: {}", py::repr_str(variable));
                return ExitCode::from(1);
            }
        }
    }
    let (work, override_dir) = (&dirs[0], &dirs[1]);
    let (log, exit) = serve::serve(Config::from_env(), &name, work, override_dir);
    let path = work.join(format!("{name}.log"));
    if let Err(e) = std::fs::write(&path, py::native_lines(&(log.join("\n") + "\n"))) {
        eprintln!("{}", py::Failure::io(&e, &path));
        return ExitCode::from(1);
    }
    match exit {
        Some(failure) => {
            eprintln!("{failure}");
            ExitCode::from(1)
        }
        None => ExitCode::SUCCESS,
    }
}
