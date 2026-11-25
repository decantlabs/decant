//! Shared parts of the Decant test programs (`loadtest`, `xmem_abitest`, `ffxtest`,
//! `decant_bvh_test`, `xaudio2test`, `gameinputtest`). Each program keeps the command line, the
//! printed lines and the exit codes of the C test it replaces, so the build folder and
//! `gpu_diag` use them the same way.

use std::path::{Path, PathBuf};

#[cfg(windows)]
mod library;
#[cfg(windows)]
pub use library::Library;

/// Prints one check line as the C tests did (`"  %-Ns %s\n"`) and counts the failures.
pub struct Checks {
    width: usize,
    ok: &'static str,
    fail: &'static str,
    failures: u32,
}

impl Checks {
    /// `width` is the padding of the description, `ok` and `fail` the two verdict words.
    pub fn new(width: usize, ok: &'static str, fail: &'static str) -> Checks {
        Checks { width, ok, fail, failures: 0 }
    }

    pub fn check(&mut self, ok: bool, what: &str) {
        println!("  {what:<width$} {}", if ok { self.ok } else { self.fail }, width = self.width);
        if !ok {
            self.failures += 1;
        }
    }

    pub fn failures(&self) -> u32 {
        self.failures
    }
}

/// The program's arguments without the ones `cargo test` may append (`--nocapture` and the like).
pub fn args() -> Vec<std::ffi::OsString> {
    std::env::args_os().skip(1).filter(|a| !a.to_string_lossy().starts_with("--")).collect()
}

/// The DLL an integration test hands to a test program: `DECANT_DLL_DIR` when set (to run
/// the same program against another build, such as the former C DLLs), else the `deps` folder
/// next to the program, where `cargo test` writes the package's DLL, else the program's folder.
pub fn dll_for(program: &str, dll: &str) -> PathBuf {
    if let Some(dir) = std::env::var_os("DECANT_DLL_DIR") {
        return Path::new(&dir).join(dll);
    }
    let deps = Path::new(program).with_file_name("deps").join(dll);
    if deps.is_file() { deps } else { Path::new(program).with_file_name(dll) }
}

/// The folder of the running program.
pub fn program_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dll_sits_next_to_the_program() {
        if std::env::var_os("DECANT_DLL_DIR").is_none() {
            assert_eq!(dll_for("a/b/loadtest.exe", "xmem.dll"), Path::new("a/b/xmem.dll"));
        }
    }
}
