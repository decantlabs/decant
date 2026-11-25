//! The read only shader research tools of `src/shaders`, with the same outputs as the Python scripts they replace:
//!
//! | Binary | Python script |
//! |---|---|
//! | `decant_shader_metadata` | `src/shaders/decant_shader_metadata.py` |
//! | `decant_shader_outputs` | `src/shaders/decant_shader_outputs.py` |
//! | `fssm_tables` | `src/shaders/fssm_tables.py` |
//! | `fssm_bindings` | `src/shaders/fssm_bindings.py` |
//! | `decant_shader_vertex_abi` | `src/shaders/decant_shader/tools/decant_shader_vertex_abi.py` |
//! | `decant_shader_pixel_inputs` | `src/shaders/decant_shader/tools/decant_shader_pixel_inputs.py` |
//!
//! The scripts find their files next to themselves. The ports find them from the current folder,
//! so running them from `src/shaders` reads and writes the same files: `out/` for the metadata,
//! the pack and the JSON lines, `decant_shader/xbox_decode_test_xbox.exe` for the decoder. `--out` and
//! `--decoder` point elsewhere. The shaders come from the file or folder named by `DECANT_SHADERS_RPF`, as in Python.
//!
//! The decoder is a separate program. It is only ever run as a
//! separate process whose listing is read, as the scripts do.

pub mod decoder;
pub mod dxbc;
pub mod metadata;
pub mod pixel;
pub mod py;
pub mod random;
pub mod tables;
pub mod vertex;

#[cfg(test)]
mod parity;
#[cfg(test)]
pub(crate) mod synthetic;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use decant_extract_shaders::{Shader, container};

pub use py::NL;

/// The folders and programs the tools use, from `--out` and `--decoder`.
pub struct Paths {
    /// The script's `out` folder: metadata, pack and JSON lines.
    pub out: PathBuf,
    /// The decoder test bench and disassembler.
    pub decoder: PathBuf,
}

impl Paths {
    pub fn metadata(&self) -> PathBuf {
        self.out.join("decant_shader_metadata.jsonl")
    }

    pub fn pack(&self) -> PathBuf {
        self.out.join("shaders_gpu.pack")
    }
}

/// The command line: the positional arguments of the script, and the paths.
pub fn args() -> Result<(Vec<String>, Paths), String> {
    let mut positional = Vec::new();
    let mut paths = Paths { out: PathBuf::from("out"), decoder: Path::new("decant_shader").join("xbox_decode_test_xbox.exe") };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => paths.out = it.next().map(PathBuf::from).ok_or("--out needs a folder")?,
            "--decoder" => paths.decoder = it.next().map(PathBuf::from).ok_or("--decoder needs a program")?,
            _ => positional.push(arg),
        }
    }
    Ok((positional, paths))
}

/// The shader containers found in the file or folder named by `DECANT_SHADERS_RPF`, in corpus order.
pub fn open_corpus() -> Result<Vec<Shader>, String> {
    let path = std::env::var_os("DECANT_SHADERS_RPF").ok_or("KeyError: 'DECANT_SHADERS_RPF'")?;
    decant_extract_shaders::find_shaders(&PathBuf::from(path))
}

/// The DXBC container of a shader.
pub fn load_blob(shader: &Shader) -> Result<Vec<u8>, String> {
    Ok(shader.container().to_vec())
}

/// The FSSM block of a shader, or the `TypeError` of `bytes(None)`.
pub fn load_fssm(shader: &Shader) -> Result<Vec<u8>, String> {
    container::gpu_code(shader.container()).ok_or_else(|| format!("TypeError: {} has no FSSM block", shader.name))
}

/// Every line of a JSON lines file, read as `json.loads` reads it.
pub fn read_jsonl(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines().map(|line| serde_json::from_str(line).map_err(|e| format!("{}: {e}", path.display()))).collect()
}

/// Python's `int(text)` for the decimal arguments of the scripts.
pub fn int_arg(text: &str) -> Result<i64, String> {
    let t = text.trim().replace('_', "");
    t.strip_prefix('+').unwrap_or(&t).parse().map_err(|_| format!("ValueError: invalid literal for int() with base 10: '{text}'"))
}

/// Runs a tool: an error goes to the error stream with exit code 1, where Python stops with a
/// traceback.
pub fn run(tool: impl FnOnce() -> Result<(), String>) -> ExitCode {
    match tool() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            use std::io::Write;
            let _ = std::io::stdout().flush();
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
