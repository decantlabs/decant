//! Port of `src/shaders/decant_shader_metadata.py`: `PSV0` and `ISG1` reflection of every shader and an
//! independent comparison with fields of the FSSM header. Reads the archive named by
//! `DECANT_SHADERS_RPF`, writes `out/decant_shader_metadata.jsonl` (one line per shader) and prints a
//! summary line.
//!
//! Usage: `decant_shader_metadata [--out folder]`

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use decant_extract_shaders::{Shader, container};
use decant_shader_analysis::metadata::{Summary, analyse};
use decant_shader_analysis::{NL, args, load_blob, open_corpus, run};

fn main() -> ExitCode {
    run(|| {
        let (_, paths) = args()?;
        let items = open_corpus()?;
        let path = paths.metadata();
        let mut out = BufWriter::new(File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?);
        let shaders: Vec<&Shader> = items.iter().collect();
        let results = decant_extract_shaders::map_shaders(&shaders, |item| {
            let blob = load_blob(item)?;
            let fssm = container::gpu_code(&blob);
            analyse(&blob, fssm.as_deref()).map_err(|e| format!("{}: {e}", item.name))
        })?;
        let mut summary = Summary::default();
        let io = |e: std::io::Error| e.to_string();
        for (id, (item, result)) in shaders.iter().zip(results).enumerate() {
            // Rows before a failing shader stay written, as the script leaves them.
            let shader = result.inspect_err(|_| {
                let _ = out.flush();
            })?;
            write!(out, "{}{NL}", shader.row(id, &item.name)).map_err(io)?;
            summary.add(&shader);
        }
        out.flush().map_err(io)?;
        print!("{}{NL}", summary.line());
        Ok(())
    })
}
