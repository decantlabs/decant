//! `vkd3d_test`, the test of the "Direct3D 12 -> vkd3d-proton -> Vulkan" path with the game
//! shaders. Port of the former `src/shaders/decant_shader/harness/vkd3d_test.cpp`, same command line,
//! output file, printed lines and exit codes.
//!
//! The program speaks Direct3D 12, but to the `d3d12.dll` of vkd3d-proton (loaded by its full
//! path). It creates a pipeline with the game's shader containers AS THEY ARE (console code, no
//! DXIL bytecode), vkd3d-proton must then take the DecantShader SPIR-V modules in their place
//! (`VKD3D_SHADER_OVERRIDE`, set by the driver script). Same full screen triangle and same output
//! as `decant_shader_run`: 64 x 64 pixels, 4 floats per pixel.
//!
//! ```text
//! vkd3d_test <d3d12.dll de vkd3d-proton> <sommets.dxbc> <pixels.dxbc> <sortie.f32> [--cbv <registre> <fichier> | --srv <registre>]
//! ```

#[cfg(windows)]
mod d3d12;

use decant_shader_bench::{Exit, finish};

pub const USAGE: &str = "usage : vkd3d_test <d3d12.dll> <sommets.dxbc> <pixels.dxbc> <sortie.f32>";

fn main() -> std::process::ExitCode {
    finish(run())
}

fn run() -> Result<i32, Exit> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() < 5 {
        return Err(Exit::new(1, USAGE));
    }
    #[cfg(windows)]
    return d3d12::render(&args);
    #[cfg(not(windows))]
    Err(Exit::new(3, "Direct3D 12 : Windows seulement"))
}
