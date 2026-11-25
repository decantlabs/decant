//! An independent check of SPIR-V validation: one original minimal module, and variants that each
//! break one declaration. A locally installed SPIRV-Tools assembles and validates them; nothing
//! here ships a binary or a tool. Generated modules live in a temporary folder and are deleted.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// The original module, as SPIR-V assembly text: one entry point that does nothing. Every invalid
/// variant changes one declaration of it. Kept in the source rather than in a `.spvasm` file,
/// which the repository check does not admit.
const MINIMAL: &str = "; A minimal compute module written for the validator exercise.
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main \"main\"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
      %float = OpTypeFloat 32
       %func = OpTypeFunction %void
       %main = OpFunction %void None %func
      %entry = OpLabel
               OpReturn
               OpFunctionEnd
";

/// `MINIMAL` with line feeds only, whatever line ends the checkout gave the source file.
pub fn minimal() -> String {
    MINIMAL.replace("\r\n", "\n")
}

/// The validation environment: Vulkan 1.3 semantics, SPIR-V 1.6.
pub const TARGET_ENV: &str = "vulkan1.3";

/// First word of every SPIR-V module.
const MAGIC: u32 = 0x0723_0203;

/// One module of the exercise: the valid one, or one deliberate fault in it.
pub struct Variant {
    pub name: &'static str,
    /// What was changed, in words.
    pub change: &'static str,
    pub text: String,
}

/// The minimal module with `from` replaced by `to`, exactly once.
fn edit(from: &str, to: &str) -> String {
    let base = minimal();
    assert_eq!(base.matches(from).count(), 1, "the edit must hit one line: {from}");
    base.replacen(from, to, 1)
}

/// The valid module first, then the invalid variants.
pub fn variants() -> Vec<Variant> {
    let v = |name, change, text| Variant { name, change, text };
    vec![
        v("minimal", "none", minimal()),
        v("duplicate-type", "a second OpTypeVoid", edit("%void = OpTypeVoid\n", "%void = OpTypeVoid\n%void2 = OpTypeVoid\n")),
        v("return-type", "the function's result type differs from its function type", edit("%main = OpFunction %void", "%main = OpFunction %float")),
        v("no-memory-model", "OpMemoryModel removed", edit("               OpMemoryModel Logical GLSL450\n", "")),
        v("storage-class", "a module scope variable whose storage class differs from its pointer type", edit("%func = OpTypeFunction %void\n", "%func = OpTypeFunction %void\n%ptr = OpTypePointer Function %float\n%var = OpVariable %ptr Private\n")),
        v("missing-capability", "a 64 bit float without the Float64 capability", edit("%float = OpTypeFloat 32\n", "%float = OpTypeFloat 32\n%double = OpTypeFloat 64\n")),
        v("entry-not-function", "the entry point names a type, not a function", edit("OpEntryPoint GLCompute %main", "OpEntryPoint GLCompute %void")),
    ]
}

/// The two SPIRV-Tools programs used.
pub struct Tools {
    pub assembler: PathBuf,
    pub validator: PathBuf,
}

const EXE: &str = std::env::consts::EXE_SUFFIX;

impl Tools {
    /// Looks in `SPIRV_TOOLS`, then `VULKAN_SDK`'s `Bin` folder, then the `PATH`.
    pub fn find() -> Option<Tools> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        dirs.extend(std::env::var_os("SPIRV_TOOLS").map(PathBuf::from));
        dirs.extend(std::env::var_os("VULKAN_SDK").map(|d| PathBuf::from(d).join("Bin")));
        dirs.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
        dirs.into_iter().find_map(|d| {
            let (assembler, validator) = (d.join(format!("spirv-as{EXE}")), d.join(format!("spirv-val{EXE}")));
            (assembler.is_file() && validator.is_file()).then_some(Tools { assembler, validator })
        })
    }

    /// The first line of `spirv-val --version`.
    pub fn version(&self) -> String {
        Command::new(&self.validator).arg("--version").stdin(Stdio::null()).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).and_then(|s| s.lines().next().map(str::to_owned)).unwrap_or_default()
    }
}

/// What happened to one variant.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The assembler refused the text.
    NotAssembled { exit: Option<i32> },
    /// Assembled to a SPIR-V module of this many bytes, then validated with this exit code and
    /// the validator's first message line.
    Validated { bytes: usize, exit: Option<i32>, message: String },
}

impl Outcome {
    pub fn valid(&self) -> bool {
        matches!(self, Outcome::Validated { exit: Some(0), .. })
    }
}

/// A temporary folder removed when dropped.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn first_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_owned()
}

/// Assembles and validates `text` in a fresh temporary folder.
pub fn check(tools: &Tools, name: &str, text: &str) -> std::io::Result<Outcome> {
    let scratch = Scratch(scratch_dir(name));
    std::fs::create_dir_all(&scratch.0)?;
    let (source, module) = (scratch.0.join("module.spvasm"), scratch.0.join("module.spv"));
    std::fs::write(&source, text)?;
    let assembled = Command::new(&tools.assembler).args(["--target-env", TARGET_ENV, "-o"]).arg(&module).arg(&source).stdin(Stdio::null()).output()?;
    if !assembled.status.success() {
        return Ok(Outcome::NotAssembled { exit: assembled.status.code() });
    }
    let bytes = std::fs::read(&module)?;
    if bytes.len() < 20 || u32::from_le_bytes(bytes[..4].try_into().unwrap()) != MAGIC {
        return Err(std::io::Error::other("the assembler did not write a SPIR-V module"));
    }
    let validated = Command::new(&tools.validator).args(["--target-env", TARGET_ENV]).arg(&module).stdin(Stdio::null()).output()?;
    let message = first_line(&validated.stderr).replace(&*scratch.0.to_string_lossy(), "<tmp>");
    Ok(Outcome::Validated { bytes: bytes.len(), exit: validated.status.code(), message })
}

/// The temporary folder a check uses, removed when the check ends.
fn scratch_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("spirv-exercise-{}-{name}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_differs_from_the_minimal_module_by_one_edit() {
        let all = variants();
        assert_eq!(all[0].text, minimal());
        let mut names: Vec<&str> = all.iter().map(|v| v.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len());
        for v in &all[1..] {
            assert_ne!(v.text, all[0].text, "{}", v.name);
        }
    }

    /// Needs SPIRV-Tools; skipped, with a note, where it is not installed.
    #[test]
    fn the_minimal_module_validates_and_every_variant_is_refused() {
        let Some(tools) = Tools::find() else {
            eprintln!("skipped: SPIRV-Tools not found (set SPIRV_TOOLS or VULKAN_SDK)");
            return;
        };
        for v in variants() {
            let outcome = check(&tools, v.name, &v.text).unwrap();
            assert_eq!(outcome.valid(), v.name == "minimal", "{}: {outcome:?}", v.name);
            assert!(!scratch_dir(v.name).exists(), "{}: generated module left behind", v.name);
        }
    }
}
