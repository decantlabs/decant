//! Repository checks in Rust, in place of the former Python scripts:
//!
//! | Binary | Former script |
//! |---|---|
//! | `check_repository` | `scripts/check_repository.py`: tracked paths and their text before review |
//! | `third_party` | `scripts/third_party.py`: `list`, `check` and `setup` of the pieces in `third_party/` |
//!
//! Both work on the repository that holds the current folder, found with
//! `git rev-parse --show-toplevel`, or on the folder given with `--root`.

pub mod rules;
pub mod third;

use std::path::PathBuf;

/// The line end Python's text mode wrote: CRLF on Windows, LF elsewhere.
pub const NL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

/// The repository root: `--root <folder>` taken out of `args`, else the git top level of the
/// current folder.
pub fn root(args: &mut Vec<String>) -> Result<PathBuf, String> {
    if let Some(i) = args.iter().position(|a| a == "--root") {
        if i + 1 >= args.len() {
            return Err("--root needs a folder".to_owned());
        }
        let folder = args.remove(i + 1);
        args.remove(i);
        return Ok(PathBuf::from(folder));
    }
    let out = std::process::Command::new("git").args(["rev-parse", "--show-toplevel"]).output().map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err("not inside a git checkout: give --root".to_owned());
    }
    let top = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    Ok(PathBuf::from(if cfg!(windows) { top.replace('/', "\\") } else { top }))
}
