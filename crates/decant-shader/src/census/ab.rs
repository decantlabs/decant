//! Feeds the translated HLSL to the GPU comparer `decant-shader-ab` against the old overrides.
//!
//! For every translated program that has an old override (`<hash>.dxil` in one of the override
//! folders) this writes `<hash>.hlsl` and one line of the comparer's batch list:
//! `hash, old override (A), new HLSL (B), stage, root signature`. The comparer reports the
//! differences. Neither side is the truth: the old path is wrong for pixel programs with more
//! than one input, for indexed constant buffer reads and in some packed colour exports.
//!
//! The comparer runs on WARP only (`DECANT_SHADER_DEVICE=warp`): the old overrides with generated
//! inputs can hang a real GPU, and the game runs on the same machine.

use std::path::{Path, PathBuf};

use crate::api::{CountingHost, Options, RootSignature, Stage, StageProgram};

/// What became of one program.
#[derive(Debug, PartialEq, Eq)]
pub enum Emitted {
    /// A line for the batch list, the HLSL file is written.
    Pair(String),
    /// Translated, but no old override exists for this hash.
    NoOverride,
    /// The translator refuses the program.
    NotTranslated,
    /// A stage the comparer does not run (mesh, geometry).
    Unsupported,
}

/// The comparer's word for a stage.
pub fn stage_word(stage: Stage) -> Option<&'static str> {
    match stage {
        Stage::Vertex => Some("vs"),
        Stage::Pixel => Some("ps"),
        Stage::Compute => Some("cs"),
        _ => None,
    }
}

/// The old override of `hash`: the first folder that has `<hash>.dxil`.
pub fn find_override(overrides: &[PathBuf], hash: u64) -> Option<PathBuf> {
    overrides
        .iter()
        .map(|d| d.join(format!("{hash:016x}.dxil")))
        .find(|p| p.is_file())
}

/// Translates `program` alone and writes the batch list line when an old override exists.
pub fn emit(
    dir: &Path,
    overrides: &[PathBuf],
    hash: u64,
    program: &StageProgram,
) -> Result<Emitted, String> {
    let Some(word) = stage_word(program.stage) else {
        return Ok(Emitted::Unsupported);
    };
    let Ok(t) = crate::translate(
        std::slice::from_ref(program),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    ) else {
        return Ok(Emitted::NotTranslated);
    };
    let Some(old) = find_override(overrides, hash) else {
        return Ok(Emitted::NoOverride);
    };
    let stage = &t.stages[0];
    let file = dir.join(format!("{hash:016x}.hlsl"));
    std::fs::write(&file, &stage.hlsl).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(Emitted::Pair(format!(
        "{hash:016x}\t{}\t{}@{}\t{word}\t-",
        old.display(),
        file.display(),
        stage.entry
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_words() {
        assert_eq!(stage_word(Stage::Pixel), Some("ps"));
        assert_eq!(stage_word(Stage::Mesh), None);
    }

    #[test]
    fn override_lookup_takes_the_first_folder_with_the_file() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        std::fs::write(b.path().join("00000000000000ff.dxil"), b"x").unwrap();
        let dirs = [a.path().to_path_buf(), b.path().to_path_buf()];
        assert_eq!(
            find_override(&dirs, 0xff),
            Some(b.path().join("00000000000000ff.dxil"))
        );
        assert_eq!(find_override(&dirs, 1), None);
    }
}
