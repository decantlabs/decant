//! Port of `src/shaders/decant_shader/harness/check_blit.py`: compares the image `decant_shader_run`
//! rendered with the expected interpolated colour of the full screen triangle, times a factor per
//! component when four are given. Exit code 0 when it matches, 1 otherwise.
//!
//! Usage: `check_blit <image.f32> [factor_r factor_g factor_b factor_a]`

use std::process::ExitCode;

fn float(text: &str) -> Result<f64, String> {
    text.trim().parse().map_err(|_| format!("ValueError: could not convert string to float: '{text}'"))
}

fn main() -> ExitCode {
    let mut ok = false;
    let code = decant_shader_analysis::run(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let path = args.first().ok_or("IndexError: list index out of range")?;
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let factor = if args.len() >= 5 { Some([float(&args[1])?, float(&args[2])?, float(&args[3])?, float(&args[4])?]) } else { None };
        let verdict = decant_shader_tools::images::blit(&data, factor)?;
        print!("{}", verdict.text);
        ok = verdict.ok;
        Ok(())
    });
    if code == ExitCode::SUCCESS && !ok { ExitCode::FAILURE } else { code }
}
