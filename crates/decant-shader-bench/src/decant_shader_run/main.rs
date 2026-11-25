//! `decant_shader_run`, the Vulkan test bench: runs a vertex and a pixel shader translated by DecantShader
//! (SPIR-V) on a full screen triangle and writes the image as floats. Port of the former
//! `src/shaders/decant_shader/harness/decant_shader_run.cpp`, same command line, output file, printed
//! lines and exit codes.
//!
//! * `<sortie.f32>`: 64 x 64 pixels, 4 floats per pixel (RGBA), row by row from the top.
//! * `--buffer`: storage buffer bound at set 0, given binding (content = file).
//! * `--push`: 32 bit word of the push constants (32 words, zero by default). Their meaning depends
//!   on the module: for a buffer, the translator stores the offset in the bound buffer and its size in bytes.
//! * `--texture`: the 4 x 4 bench texture and a nearest sampler at the two set 0 bindings.
//! * `--layout`: binds every resource the module declares (see `decant_shader_bench::layout`).
//! * `--texture-file`: a real RGBA8 texture for the 2D samplers of `--layout`, instead of the pattern.
//! * `--pipeline-cache <file>`: seeds a `VkPipelineCache` from the file (when it exists) around the
//!   pipeline creation, then writes it back, as the C++ bench does since #85.
//! * `--no-robust`: creates the device without `robustBufferAccess` and `robustImageAccess`.
//!
//! Inputs made by the bench (nothing comes from the game): three vertices with position
//! (x, y, 0), colour (r, g, b, a) and texture coordinate (u, v) at locations 0, 1, 2. Vulkan is
//! loaded at run time (`vulkan-1.dll`): no SDK needed.

mod vulkan;

use decant_shader_bench::{Exit, finish, read_file, run_args};

fn main() -> std::process::ExitCode {
    finish(run())
}

fn run() -> Result<i32, Exit> {
    let args: Vec<_> = std::env::args_os().collect();
    let run = run_args::parse(&args, read_file)?;
    let layout = run_args::load_layout(&run, |path| std::fs::read(path).ok())?;
    let vs = read_file(&run.vs)?;
    let ps = read_file(&run.ps)?;
    vulkan::render(&run, &layout, &vs, &ps)
}
