//! Port of `src/shaders/decant_shader/tools/decant_shader_vertex_abi.py`: the input convention of vertex
//! shaders that call an external routine (`S_SWAPPC` first).
//!
//! Hypothesis tested: the `ISG1` input element of register r, component c arrives in v(4 + 4r + c).
//! Method: vector registers live at entry (read before written on at least one path), from a
//! liveness analysis on the control flow graph rebuilt from the listing, compared with the
//! signature. The listing comes from the decoder test bench, run as a separate program on
//! `out/shaders_gpu.pack`. The signatures come from `out/decant_shader_metadata.jsonl`.
//!
//! Usage: `decant_shader_vertex_abi <results.jsonl of the vertex stage> <tmp folder> [output .jsonl]
//! [--out folder] [--decoder program]`

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use decant_shader_analysis::decoder::Decoder;
use decant_shader_analysis::{NL, args, read_jsonl, run, vertex};

fn main() -> ExitCode {
    run(|| {
        let (positional, paths) = args()?;
        let meta = read_jsonl(&paths.metadata())?;
        let missing = || "IndexError: list index out of range".to_owned();
        let rows = read_jsonl(&PathBuf::from(positional.first().ok_or_else(missing)?))?;
        let tmp = PathBuf::from(positional.get(1).ok_or_else(missing)?);
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let mut out = match positional.get(2) {
            Some(path) => Some(BufWriter::new(File::create(path).map_err(|e| format!("{path}: {e}"))?)),
            None => None,
        };
        let decoder = Decoder { exe: paths.decoder.clone(), pack: paths.pack() };
        let report = vertex::run(&meta, &rows, &mut |name| decoder.listing(&tmp, name), &mut |line| match &mut out {
            Some(out) => write!(out, "{line}{NL}").map_err(|e| e.to_string()),
            None => Ok(()),
        });
        // Lines written before a failure stay in the file, as Python flushes them at exit.
        if let Some(out) = &mut out {
            out.flush().map_err(|e| e.to_string())?;
        }
        print!("{}", report?);
        Ok(())
    })
}
