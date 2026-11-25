//! Ports of the on screen test drivers in `src/shaders/decant_shader/harness`: `run_tests.py`
//! (binary `decant_shader_run_tests`), `run_vkd3d_test.py` (binary `decant_shader_run_vkd3d_test`) and the
//! shader gallery `decant_shader_gallery.py` (binary `decant_shader_gallery`, see [`gallery`]).
//!
//! The drivers only start other programs and compare what they print: the translation
//! (`decant_shader_one`), the linking (`decant_shader_link`), the GPU benches `decant_shader_run.exe` and
//! `vkd3d_test.exe`, the image checkers `check_blit.py` and `check_texture.py`, and through the
//! vkd3d-proton hook `decant_shader_serve`. [`Harness`] says where each one is, and each has an
//! environment override so a port or a stand in can take its place.
//!
//! Everything the drivers write goes to `harness/out`, which only holds files made from
//! title files and must not be shared.

pub mod gallery;
pub mod regress;
pub mod semantics;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use decant_shader_bridge::py;
use decant_shader_bridge::shaders::Config;

/// The programs and folders the drivers use.
#[derive(Clone, Debug)]
pub struct Harness {
    /// The bridge configuration: the `src/shaders` folder and the `DECANT_SHADER_ONE` and
    /// `DECANT_SHADER_LOCATE` replacements.
    pub config: Config,
    /// A program to run instead of the `decant_shader_link` built next to this one (`DECANT_SHADER_LINK`).
    pub link: Option<OsString>,
    /// The Vulkan bench (`DECANT_SHADER_RUN_BENCH`, default `harness/decant_shader_run.exe`).
    pub run_bench: PathBuf,
    /// The Direct3D 12 bench (`DECANT_SHADER_VKD3D_TEST`, default `harness/vkd3d_test.exe`).
    pub vkd3d_test: PathBuf,
    /// Programs to run instead of the `check_blit` and `check_texture` built next to this one
    /// (`DECANT_SHADER_CHECK_BLIT`, `DECANT_SHADER_CHECK_TEXTURE`).
    pub check_blit: Option<OsString>,
    pub check_texture: Option<OsString>,
    /// The program the vkd3d-proton hook runs per shader (`DECANT_SHADER_SERVE`, default the
    /// `decant_shader_serve` built next to this one).
    pub serve: Option<OsString>,
}

impl Harness {
    pub fn from_env() -> Harness {
        let mut config = Config::from_env();
        config.shaders = std::path::absolute(&config.shaders).unwrap_or(config.shaders);
        let dir = config.root().join("harness");
        let var = |name: &str| std::env::var_os(name);
        Harness {
            link: var("DECANT_SHADER_LINK"),
            run_bench: var("DECANT_SHADER_RUN_BENCH").map_or_else(|| dir.join("decant_shader_run.exe"), PathBuf::from),
            vkd3d_test: var("DECANT_SHADER_VKD3D_TEST").map_or_else(|| dir.join("vkd3d_test.exe"), PathBuf::from),
            check_blit: var("DECANT_SHADER_CHECK_BLIT"),
            check_texture: var("DECANT_SHADER_CHECK_TEXTURE"),
            serve: var("DECANT_SHADER_SERVE"),
            config,
        }
    }

    /// `src/shaders/decant_shader/harness`.
    pub fn dir(&self) -> PathBuf {
        self.config.root().join("harness")
    }

    /// `harness/out`.
    pub fn out(&self) -> PathBuf {
        self.dir().join("out")
    }

    /// The checker `name` (`check_blit` or `check_texture`), or its replacement.
    pub fn checker(&self, name: &str) -> Command {
        let replacement = match name {
            "check_blit" => &self.check_blit,
            "check_texture" => &self.check_texture,
            _ => &None,
        };
        Command::new(Config::program(name, replacement))
    }

    /// The text of the `decant_shader_serve.cmd` launcher the vkd3d driver writes, as bytes on disk.
    pub fn serve_launcher(&self) -> String {
        let program = Config::program("decant_shader_serve", &self.serve);
        let line = format!("@echo off\r\n\"{}\" %*\r\n", program.to_string_lossy());
        // `write_text` in text mode turns each "\n" into "\r\n" on Windows, so "\r\n" becomes "\r\r\n".
        py::native_lines(&line)
    }
}

/// What `subprocess.run(..., capture_output=True, text=True, errors="replace")` gives: the exit
/// code and both outputs decoded in the locale's code page with universal newlines.
pub struct Ran {
    pub code: i64,
    pub stdout: String,
    pub stderr: String,
}

pub fn run_text(command: &mut Command) -> std::io::Result<Ran> {
    command.stdin(Stdio::inherit());
    let Output { status, stdout, stderr } = command.output()?;
    Ok(Ran { code: return_code(status), stdout: decant_shader_core::py::read_text(&stdout), stderr: decant_shader_core::py::read_text(&stderr) })
}

/// `returncode`: unsigned on Windows, the negated signal elsewhere.
pub fn return_code(status: std::process::ExitStatus) -> i64 {
    #[cfg(windows)]
    {
        i64::from(status.code().unwrap_or(0) as u32)
    }
    #[cfg(not(windows))]
    {
        use std::os::unix::process::ExitStatusExt;
        status.code().map(i64::from).or_else(|| status.signal().map(|s| -i64::from(s))).unwrap_or(0)
    }
}

/// `text.strip()[-160:]`.
pub fn tail160(text: &str) -> &str {
    py::tail(py::strip(text), 160)
}

/// The last line a checker printed, or why there is none.
pub fn verdict(checked: &Ran) -> (String, bool) {
    let stdout = py::strip(&checked.stdout);
    let lines = py::splitlines(stdout);
    match lines.last() {
        Some(last) => ((*last).to_owned(), checked.code == 0),
        None => (format!("ECHEC du verificateur (code {}) : {}", checked.code, tail160(&checked.stderr)), false),
    }
}

/// The constant buffer both drivers write: 16 floats, the factors 0.5, 2.0, 0.25 and 1.0 at 8 to 11.
pub fn constants() -> Vec<u8> {
    let mut floats = [0.0f32; 16];
    floats[8..12].copy_from_slice(&[0.5, 2.0, 0.25, 1.0]);
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// `shutil.rmtree(folder, ignore_errors=True); folder.mkdir()`.
pub fn fresh(folder: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_dir_all(folder);
    std::fs::create_dir(folder)
}

/// `f"{text:width}"`: left aligned, padded with spaces to `width` characters.
pub fn pad(text: &str, width: usize) -> String {
    format!("{text:width$}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts_take_the_last_line() {
        let ok = Ran { code: 0, stdout: "x\nCONFORME : 4096 pixels\n\n".into(), stderr: String::new() };
        assert_eq!(verdict(&ok), ("CONFORME : 4096 pixels".into(), true));
        let bad = Ran { code: 1, stdout: "ECART : 3 pixels\n".into(), stderr: String::new() };
        assert_eq!(verdict(&bad), ("ECART : 3 pixels".into(), false));
        let crashed = Ran { code: 2, stdout: " \n".into(), stderr: format!("{}end\n", "z".repeat(200)) };
        let (text, passed) = verdict(&crashed);
        assert!(!passed);
        assert_eq!(text, format!("ECHEC du verificateur (code 2) : {}end", "z".repeat(157)));
    }

    #[test]
    fn files_match_the_python() {
        let c = constants();
        assert_eq!(c.len(), 64);
        assert_eq!(&c[32..48], [0.5f32, 2.0, 0.25, 1.0].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>().as_slice());
        assert_eq!(pad("copie de couleur", 20), "copie de couleur    ");
        assert_eq!(pad(&"n".repeat(40), 34).len(), 40);
        let mut h = Harness::from_env();
        h.serve = Some("C:\\x\\decant_shader_serve.exe".into());
        let nl = if cfg!(windows) { "\r\r\n" } else { "\r\n" };
        assert_eq!(h.serve_launcher(), format!("@echo off{nl}\"C:\\x\\decant_shader_serve.exe\" %*{nl}"));
    }
}
