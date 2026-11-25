//! Rewrites the resources of a DecantShader SPIR-V module for another engine's layout, as
//! `src/shaders/decant_shader/tools/decant_shader_adapt.py` does, with the same module and report.
//!
//! Usage: `decant_shader_adapt <pack number> <layout.json> <output.spv>`
//!
//! The translation runs in a scratch folder of this call only, outside the repository, removed
//! even on failure: vkd3d-proton compiles on several threads, so several calls can run at once.

use std::path::Path;
use std::process::ExitCode;

use decant_shader_bridge::adapt;
use decant_shader_bridge::py::{self, Failure};
use decant_shader_bridge::session::Session;
use decant_shader_bridge::shaders::Config;

fn run(args: &[String]) -> Result<(), Failure> {
    let session = Session::open(Config::from_env())?;
    let [pack_id, layout, target] = args else {
        return Err(Failure::Exit("usage : decant_shader_adapt <numero dans le pack> <disposition.json> <sortie.spv>".into()));
    };
    let pack_id = py::int(pack_id)?;
    let layout_path = Path::new(layout);
    let text = std::fs::read_to_string(layout_path).map_err(|e| Failure::io(&e, layout_path))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| Failure::error("json.decoder.JSONDecodeError", &e.to_string()))?;
    let layout = adapt::layout_from_json(&value)?;
    let work = tempfile::Builder::new().prefix("decant_shader_adapt_").tempdir().map_err(|e| Failure::io(&e, &std::env::temp_dir()))?;
    let (module, report) = session.adapt(pack_id, &layout, work.path())?;
    let target = Path::new(target);
    std::fs::write(target, module).map_err(|e| Failure::io(&e, target))?;
    for line in report {
        println!("   {line}");
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("{failure}");
            ExitCode::from(1)
        }
    }
}
