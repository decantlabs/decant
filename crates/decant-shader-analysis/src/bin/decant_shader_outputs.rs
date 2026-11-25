//! Port of `src/shaders/decant_shader_outputs.py`: the output signature (`OSG1`) of every shader, as a
//! complement to `decant_shader_metadata`. Reads the archive named by `DECANT_SHADERS_RPF` and writes
//! `out/decant_shader_outputs.jsonl`: one line per shader with the same numbering, and elements of the
//! same form as `inputs`.
//!
//! Usage: `decant_shader_outputs [--out folder]`

use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::ExitCode;

use decant_extract_shaders::Shader;
use decant_shader_analysis::metadata::outputs_row;
use decant_shader_analysis::{NL, args, load_blob, open_corpus, run};

fn main() -> ExitCode {
    run(|| {
        let (_, paths) = args()?;
        let items = open_corpus()?;
        let path = paths.out.join("decant_shader_outputs.jsonl");
        let mut out = BufWriter::new(File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?);
        let shaders: Vec<&Shader> = items.iter().collect();
        let blobs = decant_extract_shaders::map_shaders(&shaders, load_blob)?;
        let io = |e: std::io::Error| e.to_string();
        let mut count = 0;
        for (item, blob) in shaders.iter().zip(blobs) {
            // Rows before a failing shader stay written, as the script leaves them.
            let line = blob.and_then(|b| outputs_row(count, &item.name, &b)).map_err(|e| {
                let _ = out.flush();
                format!("{}: {e}", item.name)
            })?;
            write!(out, "{line}{NL}").map_err(io)?;
            count += 1;
        }
        out.flush().map_err(io)?;
        print!("shaders {count}{NL}");
        Ok(())
    })
}
