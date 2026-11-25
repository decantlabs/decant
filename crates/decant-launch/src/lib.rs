//! Native Windows launcher for a staged GDK title, the Windows counterpart of
//! `src/gdk/launch_prep.sh`: stage the package into a writable overlay with the Decant shims in
//! the title's own folder, read `MicrosoftGame.config` for the executable, check its PE headers,
//! then run it under the debugging API with a full log. No game file, key or runtime is part of it.

pub mod cpu_emu;
pub mod drive;
pub mod gentle;
pub mod pls_drive;
pub mod precompile;
pub mod relaunch;
pub mod run;
pub mod runtime;
pub mod sampler;
pub mod stage;
pub mod tools;

use std::path::{Path, PathBuf};

/// The executable to start: `name` when given, else the primary executable of the staged
/// `MicrosoftGame.config` (the first that is not dev-only). Its PE headers must pass the check.
pub fn title_executable(overlay: &Path, name: Option<&str>) -> Result<PathBuf, String> {
    let name = match name {
        Some(n) => n.to_owned(),
        None => {
            let config = decant_gdk_config::read_gdk_config(&overlay.join("MicrosoftGame.config"))?;
            if config.primary_executable.is_empty() {
                return Err("MicrosoftGame.config names no executable that is not dev-only".into());
            }
            config.primary_executable
        }
    };
    let exe = overlay.join(&name);
    decant_gdk_config::check_pe_image(&exe)?;
    Ok(exe)
}
