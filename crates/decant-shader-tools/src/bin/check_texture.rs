//! Port of `src/shaders/decant_shader/harness/check_texture.py`: compares the image
//! `decant_shader_run --texture` rendered with the expected texels of the bench texture. With `--lie`
//! the pixel module was linked to the vertex shader, so the texture input carries the real
//! coordinate instead of the colour. Exit code 0 when it matches, 1 otherwise.
//!
//! Usage: `check_texture <image.f32> [--lie]`

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut ok = false;
    let code = decant_shader_analysis::run(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let linked = args.iter().any(|a| a == "--lie");
        let path = args.first().ok_or("IndexError: list index out of range")?;
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let verdict = decant_shader_tools::images::texture(&data, linked)?;
        print!("{}", verdict.text);
        ok = verdict.ok;
        Ok(())
    });
    if code == ExitCode::SUCCESS && !ok { ExitCode::FAILURE } else { code }
}
