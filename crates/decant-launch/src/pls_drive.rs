//! The drive of the title's persistent local storage. On the console that storage is a drive of
//! its own, and the local XPersistentLocalStorage table of our `xgameruntime.dll` hands the title
//! its root. For the length of a run the launcher maps a free letter over the host folder
//! `%LOCALAPPDATA%\Decant\<title id>\pls` the way it maps `G:` over the game folder
//! ([`crate::drive`], per logon session, removed at the end) and tells the title process by
//! environment: `DECANT_PLS_DIR` names the host folder, `DECANT_PLS_ROOT` the drive root (only when a
//! letter is mapped). Without them the table serves the host folder itself.

use std::path::Path;

use crate::drive::GameDrive;
use crate::run::Sink;

/// Environment variable of the launcher: `0` (or `off`, `false`, `no`) maps no letter, a single
/// letter maps that one. Unset: the first free letter of [`CANDIDATES`].
pub const PLS_DRIVE_ENV: &str = "DECANT_PLS_DRIVE";

/// The letters tried, in order, when [`PLS_DRIVE_ENV`] names none.
pub const CANDIDATES: [char; 8] = ['T', 'U', 'V', 'W', 'P', 'Q', 'R', 'S'];

/// The host folder, for the title process.
pub const DIR_ENV: &str = "DECANT_PLS_DIR";
/// The drive root over it, for the title process.
pub const ROOT_ENV: &str = "DECANT_PLS_ROOT";

/// The letter to map for this value of [`PLS_DRIVE_ENV`], given which letters are free. `None`
/// when switched off or when no candidate is free.
pub fn choose_letter(value: Option<&str>, free: impl Fn(char) -> bool) -> Option<char> {
    let v = value.map(str::trim).unwrap_or("");
    if matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no") {
        return None;
    }
    let mut chars = v.trim_end_matches(':').chars();
    if let (Some(c), None) = (chars.next(), chars.next())
        && c.is_ascii_alphabetic()
    {
        return Some(c.to_ascii_uppercase());
    }
    CANDIDATES.into_iter().find(|&c| free(c))
}

/// The mapping and the environment of one run.
pub struct PlsDrive {
    drive: Option<GameDrive>,
    env: Vec<(String, String)>,
}

impl PlsDrive {
    /// Makes the host folder of the title in `title_folder` (from its `MicrosoftGame.config`)
    /// and maps a letter over it, with a log line for what happened. A title without a config or
    /// title id gets nothing: the table then finds its folder by itself.
    pub fn acquire(title_folder: &Path, log: &mut dyn Sink) -> PlsDrive {
        let none = PlsDrive { drive: None, env: Vec::new() };
        let id = decant_gdk_config::read_gdk_config(&title_folder.join("MicrosoftGame.config")).ok().and_then(|c| decant_gdk_config::title_id_number(&c.title_id));
        let Some(folder) = id.and_then(decant_gdk_config::persistent_local_storage_folder) else {
            log.line("persistent local storage: no title id or no LOCALAPPDATA, no drive mapped");
            return none;
        };
        if let Err(e) = std::fs::create_dir_all(&folder) {
            log.line(&format!("persistent local storage: {} could not be made ({e}), no drive mapped", folder.display()));
            return none;
        }
        let mut env = vec![(DIR_ENV.to_owned(), folder.to_string_lossy().into_owned())];
        let Some(letter) = choose_letter(std::env::var(PLS_DRIVE_ENV).ok().as_deref(), crate::drive::is_free) else {
            log.line(&format!("persistent local storage: no drive letter ({PLS_DRIVE_ENV} or none free), the title gets {}", folder.display()));
            return PlsDrive { drive: None, env };
        };
        log.line(&format!("persistent local storage: drive {letter}: over {}", folder.display()));
        let drive = GameDrive::acquire_letter(Some(letter), &folder, log);
        if drive.redirect().is_none() {
            env.push((ROOT_ENV.to_owned(), format!("{letter}:\\")));
            PlsDrive { drive: Some(drive), env }
        } else {
            // Taken or not mappable: nothing is mapped and no debugger redirect exists for this letter.
            log.line(&format!("persistent local storage: drive {letter}: not usable, the title gets {}", folder.display()));
            PlsDrive { drive: None, env }
        }
    }

    /// What the title process gets in its environment.
    pub fn env(&self) -> &[(String, String)] {
        &self.env
    }

    /// Removes the mapping now, with a log line. Dropping does the same without one.
    pub fn release(&mut self, log: &mut dyn Sink) {
        if let Some(drive) = self.drive.as_mut() {
            drive.release(log);
        }
        self.drive = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_letter_is_the_named_one_or_the_first_free_candidate() {
        assert_eq!(choose_letter(Some("x"), |_| false), Some('X'));
        assert_eq!(choose_letter(Some(" y: "), |_| false), Some('Y'));
        for off in ["0", "off", "FALSE", " No "] {
            assert_eq!(choose_letter(Some(off), |_| true), None, "{off}");
        }
        assert_eq!(choose_letter(None, |_| true), Some('T'));
        assert_eq!(choose_letter(Some("nonsense"), |c| c == 'W'), Some('W'));
        assert_eq!(choose_letter(None, |_| false), None);
    }

    #[test]
    fn a_folder_without_a_config_maps_nothing() {
        let mut log = Vec::new();
        let dir = std::env::temp_dir().join(format!("pls_drive_none_{}", std::process::id()));
        let mut pls = PlsDrive::acquire(&dir, &mut log);
        assert!(pls.env().is_empty());
        pls.release(&mut log);
        assert_eq!(log.len(), 1, "{log:?}");
    }
}
