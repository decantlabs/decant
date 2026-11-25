//! Where the tools find their data. The Python tools locate everything from their own file:
//! `src/shaders/out` for the pack and metadata, `src/shaders/decant_shader` for the probe.

use std::path::PathBuf;

/// The `src/shaders` folder: `DECANT_SHADERS_DIR` if set, otherwise the one of the checkout this
/// tool was built from.
pub fn shaders() -> PathBuf {
    let dir = std::env::var_os("DECANT_SHADERS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/shaders")));
    std::path::absolute(&dir).unwrap_or(dir)
}

/// `src/shaders/out`, written by `extract_shaders` and the metadata scripts.
pub fn out() -> PathBuf {
    shaders().join("out")
}

/// The workspace binary `name` built next to the running program: `cargo build` puts every tool
/// of the workspace in the same `target` folder, so the tools find each other there.
pub fn sibling(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    exe.with_file_name(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// The DecantShader probe: `DECANT_SHADER_PROBE` if set, otherwise
/// `src/shaders/decant_shader/decant_shader_probe.exe`. It is only ever run as a separate process.
pub fn probe() -> PathBuf {
    std::env::var_os("DECANT_SHADER_PROBE").map(PathBuf::from).unwrap_or_else(|| shaders().join("decant_shader").join("decant_shader_probe.exe"))
}
