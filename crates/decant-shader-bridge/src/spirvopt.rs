//! The optional spirv-opt step of the bridge, run on the FINAL module (after adaptation and
//! relocation), as `src/shaders/decant_shader/tools/decant_shader_spirvopt.py` does, with the same journal
//! lines.
//!
//! Off by default. `DECANT_SHADER_SPIRV_OPT` turns it on: unset, empty or `0` does nothing, `1` takes
//! `spirv-opt` from the `PATH`, any other value is the path of spirv-opt. The optimised module is
//! checked by spirv-val (`DECANT_SHADER_SPIRV_VAL`, otherwise `spirv-val` from the `PATH`) before it
//! replaces the original. When spirv-opt fails or runs out of time, or its result does not
//! validate, the module is kept as it was.
//!
//! `--preserve-bindings` and `--preserve-interface` are required: the module must keep exactly the
//! set/binding pairs of vkd3d-proton's layout and every interface variable.

use std::ffi::OsString;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const OPTIONS: [&str; 3] = ["-O", "--preserve-bindings", "--preserve-interface"];
/// For each of the two tools.
pub const TIMEOUT: Duration = Duration::from_secs(60);

/// (spirv-opt, spirv-val) to run, or `None` when the step is off.
pub fn tools(get: impl Fn(&str) -> Option<OsString>) -> Option<(OsString, OsString)> {
    let setting = get("DECANT_SHADER_SPIRV_OPT").unwrap_or_default();
    let setting = setting.to_string_lossy().trim().to_owned();
    if setting.is_empty() || setting == "0" {
        return None;
    }
    let optimizer = if setting == "1" { OsString::from("spirv-opt") } else { OsString::from(setting) };
    let validator = get("DECANT_SHADER_SPIRV_VAL")
        .map(|v| v.to_string_lossy().trim().to_owned())
        .filter(|v| !v.is_empty())
        .map_or_else(|| OsString::from("spirv-val"), OsString::from);
    Some((optimizer, validator))
}

/// Exit code, or the reason the tool gave none (`introuvable`, `delai`), like the Python `_run`.
fn run(tool: &OsString, args: &[&std::ffi::OsStr]) -> Result<i32, &'static str> {
    let mut child = match Command::new(tool).args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
        Ok(child) => child,
        Err(_) => return Err("introuvable"),
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status.code().unwrap_or(-1)),
            Ok(None) if start.elapsed() >= TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("delai");
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => return Err("introuvable"),
        }
    }
}

/// Optimises the module in place with the tools of the environment. Gives the journal line, or
/// `None` when the step is off.
pub fn optimize(module: &Path) -> Option<String> {
    tools(|name| std::env::var_os(name)).map(|(optimizer, validator)| optimize_with(module, &optimizer, &validator))
}

/// [`optimize`] with explicit tools.
pub fn optimize_with(module: &Path, optimizer: &OsString, validator: &OsString) -> String {
    let before = std::fs::metadata(module).map(|m| m.len()).unwrap_or(0);
    let name = module.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let candidate = module.with_file_name(format!("{name}.opt.{}", std::process::id()));
    let line = (|| {
        let mut args: Vec<&std::ffi::OsStr> = OPTIONS.iter().map(std::ffi::OsStr::new).collect();
        args.extend([module.as_os_str(), "-o".as_ref(), candidate.as_os_str()]);
        match run(optimizer, &args) {
            Err(problem) => return format!("spirv-opt : echec (spirv-opt {problem}), module non optimise garde"),
            Ok(code) if code != 0 || !candidate.exists() => {
                return format!("spirv-opt : echec (spirv-opt code {code}), module non optimise garde");
            }
            Ok(_) => {}
        }
        match run(validator, &["--target-env".as_ref(), "vulkan1.2".as_ref(), candidate.as_os_str()]) {
            Err(problem) => format!("spirv-opt : echec (spirv-val {problem}), module non optimise garde"),
            Ok(0) => {
                let after = std::fs::metadata(&candidate).map(|m| m.len()).unwrap_or(0);
                match std::fs::rename(&candidate, module) {
                    Ok(()) => format!("spirv-opt : {before} -> {after} octets"),
                    Err(e) => format!("spirv-opt : echec ({e}), module non optimise garde"),
                }
            }
            Ok(_) => "spirv-opt : echec (spirv-val refuse le module optimise), module non optimise garde".into(),
        }
    })();
    let _ = std::fs::remove_file(&candidate);
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| pairs.iter().find(|(k, _)| *k == name).map(|(_, v)| OsString::from(*v))
    }

    #[test]
    fn the_step_is_off_unless_asked_for() {
        assert_eq!(tools(env(&[])), None);
        assert_eq!(tools(env(&[("DECANT_SHADER_SPIRV_OPT", "")])), None);
        assert_eq!(tools(env(&[("DECANT_SHADER_SPIRV_OPT", "0")])), None);
        assert_eq!(tools(env(&[("DECANT_SHADER_SPIRV_OPT", "1")])), Some(("spirv-opt".into(), "spirv-val".into())));
        assert_eq!(
            tools(env(&[("DECANT_SHADER_SPIRV_OPT", "C:/sdk/spirv-opt.exe"), ("DECANT_SHADER_SPIRV_VAL", "C:/sdk/spirv-val.exe")])),
            Some(("C:/sdk/spirv-opt.exe".into(), "C:/sdk/spirv-val.exe".into()))
        );
    }

    #[test]
    fn a_missing_optimiser_keeps_the_module() {
        let dir = tempfile::tempdir().unwrap();
        let module = dir.path().join("m.spv");
        std::fs::write(&module, b"module").unwrap();
        let line = optimize_with(&module, &"no-such-spirv-opt-here".into(), &"no-such-spirv-val".into());
        assert_eq!(line, "spirv-opt : echec (spirv-opt introuvable), module non optimise garde");
        assert_eq!(std::fs::read(&module).unwrap(), b"module");
        // No candidate left behind.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    /// A SPIRV-Tools program from `DECANT_SHADER_TEST_SPIRV_BIN`, otherwise from the `PATH`.
    fn spirv_tool(name: &str) -> Option<std::path::PathBuf> {
        let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
        let mut dirs: Vec<std::path::PathBuf> = std::env::var_os("DECANT_SHADER_TEST_SPIRV_BIN").map(Into::into).into_iter().collect();
        dirs.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
        dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
    }

    /// A minimal compute shader written by hand (no game data). An unused local variable leaves
    /// spirv-opt something to remove.
    const SYNTHETIC: &str = "OpCapability Shader\nOpMemoryModel Logical GLSL450\nOpEntryPoint GLCompute %main \"main\"\n\
        OpExecutionMode %main LocalSize 1 1 1\n%void = OpTypeVoid\n%fn = OpTypeFunction %void\n%float = OpTypeFloat 32\n\
        %ptr = OpTypePointer Function %float\n%one = OpConstant %float 1\n%main = OpFunction %void None %fn\n%entry = OpLabel\n\
        %unused = OpVariable %ptr Function\nOpStore %unused %one\nOpReturn\nOpFunctionEnd\n";

    #[test]
    fn a_synthetic_module_is_optimised_and_still_validates() {
        let (Some(assembler), Some(optimizer), Some(validator)) = (spirv_tool("spirv-as"), spirv_tool("spirv-opt"), spirv_tool("spirv-val")) else {
            eprintln!("skipped: SPIRV-Tools not found (DECANT_SHADER_TEST_SPIRV_BIN or PATH)");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let (source, module) = (dir.path().join("m.spvasm"), dir.path().join("m.spv"));
        std::fs::write(&source, SYNTHETIC).unwrap();
        assert!(Command::new(&assembler).args(["--target-env", "vulkan1.2"]).arg(&source).arg("-o").arg(&module).status().unwrap().success());
        let before = std::fs::metadata(&module).unwrap().len();
        let line = optimize_with(&module, &optimizer.into_os_string(), &validator.clone().into_os_string());
        let after = std::fs::metadata(&module).unwrap().len();
        assert_eq!(line, format!("spirv-opt : {before} -> {after} octets"));
        assert!(after < before);
        assert!(Command::new(&validator).args(["--target-env", "vulkan1.2"]).arg(&module).status().unwrap().success());
    }
}
