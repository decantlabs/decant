//! Port of `src/shaders/decant_shader/tools/decant_shader_hlsl_probe.py`: converts a sample of the SPIR-V
//! modules of a measurement to HLSL (`spirv-cross`) and then to DXIL (`dxc`), and sorts the
//! refusals. The sample is drawn with seed 21, as in Python.
//!
//! Usage: `decant_shader_hlsl_probe <tmp folder> <count per stage> <measurement folder>:<dxc profile> [...]`
//!
//! `DECANT_SPIRV_CROSS` names `spirv-cross`, `DECANT_DXC` names `dxc` (default `dxc`).

use std::path::PathBuf;
use std::process::ExitCode;

use decant_shader_tools::hlsl::{Tools, probe};

fn main() -> ExitCode {
    decant_shader_analysis::run(|| {
        let cross = std::env::var_os("DECANT_SPIRV_CROSS").ok_or("KeyError: 'DECANT_SPIRV_CROSS'")?;
        let dxc = std::env::var_os("DECANT_DXC").unwrap_or_else(|| "dxc".into());
        let args: Vec<String> = std::env::args().skip(1).collect();
        let missing = || "IndexError: list index out of range".to_owned();
        let tmp = PathBuf::from(args.first().ok_or_else(missing)?);
        std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let count = decant_shader_core::py::int(args.get(1).ok_or_else(missing)?)?;
        let tools = Tools { cross: cross.into(), dxc: dxc.into() };
        probe(count, &args[2..], &|module, profile, index| tools.convert(module, profile, index, &tmp), &mut |text| print!("{text}"))
    })
}
