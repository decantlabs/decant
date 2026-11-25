//! Port of `src/shaders/decant_shader/tools/decant_shader_pixel_inputs.py`: checks how a pixel shader
//! numbers its interpolated inputs, on shaders drawn at random (seed 31, as in Python).
//!
//! Hypothesis: hardware attribute k (operand of `V_INTERP`) is non system `ISG1` input register
//! number k, in order, counting only registers with at least one component read. The listing
//! comes from the decoder test bench, run as a separate program on `out/shaders_gpu.pack`. The
//! signatures come from `out/decant_shader_metadata.jsonl`.
//!
//! Usage: `decant_shader_pixel_inputs <number of shaders drawn> <tmp folder> [--out folder]
//! [--decoder program]`

use std::path::PathBuf;
use std::process::ExitCode;

use decant_shader_analysis::decoder::Decoder;
use decant_shader_analysis::{args, int_arg, pixel, read_jsonl, run};

fn main() -> ExitCode {
    run(|| {
        let (positional, paths) = args()?;
        let meta = read_jsonl(&paths.metadata())?;
        let missing = || "IndexError: list index out of range".to_owned();
        let tmp = PathBuf::from(positional.get(1).ok_or_else(missing)?);
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let count = int_arg(positional.first().ok_or_else(missing)?)?;
        let decoder = Decoder { exe: paths.decoder.clone(), pack: paths.pack() };
        print!("{}", pixel::run(&meta, count, &mut |name| decoder.listing(&tmp, name))?);
        Ok(())
    })
}
