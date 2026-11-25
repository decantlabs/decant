//! The package drive of an Xbox title. A title built for the console reads its install files from the
//! package drive, `G:/`, a literal in its executable, and fails when a file it expects is missing there.
//! For the length of a run the launcher maps `G:` to the title folder the way `subst` does
//! (`DefineDosDeviceW`, per logon session, no admin rights needed) and removes the mapping when the run
//! ends, also on error, cancel or panic (a guard with `Drop`). When the letter is taken (a real drive),
//! cannot be mapped, or `DECANT_GAME_DRIVE_MODE=redirect` says so, nothing is mapped and the debugger
//! rewrites the title's `G:/` paths instead (see `run::redirect_game_path`).

use crate::run::Sink;
use std::path::Path;

/// The letter of the package drive.
pub const PACKAGE_DRIVE: char = 'G';

/// Environment variable of the launcher: `0` (or `off`, `false`, `no`) turns the mapping off, a
/// single letter such as `H` maps that letter instead of [`PACKAGE_DRIVE`].
pub const DRIVE_ENV: &str = "DECANT_GAME_DRIVE";

/// Environment variable of the launcher: `redirect` skips the mapping and sends the title's file
/// paths under the drive to the folder inside the title process (what happens anyway when the
/// letter is taken). For tests on a PC where the letter is free.
pub const DRIVE_MODE_ENV: &str = "DECANT_GAME_DRIVE_MODE";

/// Whether this value of [`DRIVE_MODE_ENV`] forces the redirect.
pub fn redirect_forced(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("redirect"))
}

/// The letter to map for this value of [`DRIVE_ENV`], or `None` when it is switched off. Unset,
/// empty or unreadable values mean [`PACKAGE_DRIVE`].
pub fn drive_letter(value: Option<&str>) -> Option<char> {
    let v = value.map(str::trim).unwrap_or("");
    if matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no") {
        return None;
    }
    let mut chars = v.trim_end_matches(':').chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) if c.is_ascii_alphabetic() => Some(c.to_ascii_uppercase()),
        _ => Some(PACKAGE_DRIVE),
    }
}

/// Whether no device is defined for `letter` in this logon session.
pub fn is_free(letter: char) -> bool {
    platform::query(letter).is_none()
}

/// What to do with the drive, from what `QueryDosDeviceW` said about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The letter is free: map it, and remove it at the end.
    Map,
    /// It already points to the folder: keep it, and leave it at the end.
    Keep,
    /// It points elsewhere (the target is carried): do not touch it.
    Taken(String),
}

/// A device target or folder reduced for comparing: no `\??\` prefix, `/` as `\`, no trailing
/// separator, lower case.
fn normal(path: &str) -> String {
    let p = path.replace('/', "\\");
    let p = p.strip_prefix("\\??\\").or_else(|| p.strip_prefix("\\\\?\\")).unwrap_or(&p);
    p.trim_end_matches('\\').to_lowercase()
}

/// Keep, map or leave alone: `existing` is the target of the letter (`None` when free), `folder`
/// the title folder.
pub fn decide(existing: Option<&str>, folder: &str) -> Decision {
    match existing {
        None => Decision::Map,
        Some(t) if normal(t) == normal(folder) => Decision::Keep,
        Some(t) => Decision::Taken(t.to_owned()),
    }
}

/// A mapping made for a run. Dropping it removes the mapping when this guard made it.
pub struct GameDrive {
    letter: char,
    folder: String,
    mapped: bool,
    /// The letter whose paths the debugger redirects inside the title (no mapping is in place).
    redirect: Option<char>,
}

impl GameDrive {
    /// Maps the drive for `folder` as [`DRIVE_ENV`] and [`DRIVE_MODE_ENV`] say, with one log line
    /// for what happened.
    pub fn acquire(folder: &Path, log: &mut dyn Sink) -> GameDrive {
        let letter = drive_letter(std::env::var(DRIVE_ENV).ok().as_deref());
        let forced = redirect_forced(std::env::var(DRIVE_MODE_ENV).ok().as_deref());
        GameDrive::acquire_with(letter, forced, folder, log)
    }

    /// Like [`GameDrive::acquire`] for a given letter (`None` maps nothing).
    pub fn acquire_letter(letter: Option<char>, folder: &Path, log: &mut dyn Sink) -> GameDrive {
        GameDrive::acquire_with(letter, false, folder, log)
    }

    /// A drive that maps nothing and has the title's `letter:/` paths redirected to `folder`.
    pub fn redirecting(letter: char, folder: &Path) -> GameDrive {
        GameDrive { letter, folder: plain_path(folder), mapped: false, redirect: Some(letter) }
    }

    /// Maps the letter, or (when `forced`, when it is taken, or when mapping fails) chooses the
    /// redirect inside the title. `None` does neither.
    pub fn acquire_with(letter: Option<char>, forced: bool, folder: &Path, log: &mut dyn Sink) -> GameDrive {
        let Some(letter) = letter else {
            return GameDrive { letter: PACKAGE_DRIVE, folder: String::new(), mapped: false, redirect: None };
        };
        let folder_text = plain_path(folder);
        let mut drive = GameDrive { letter, folder: folder_text.clone(), mapped: false, redirect: None };
        if forced {
            drive.redirect = Some(letter);
            log.line(&format!("game drive {letter}: redirect mode ({DRIVE_MODE_ENV}), paths under {letter}:/ go to {folder_text} inside the title"));
            return drive;
        }
        match decide(platform::query(letter).as_deref(), &folder_text) {
            Decision::Keep => log.line(&format!("game drive {letter}: already points to {folder_text}, kept")),
            Decision::Taken(target) => {
                drive.redirect = Some(letter);
                log.line(&format!("game drive {letter}: is taken by {target}, redirect mode: paths under {letter}:/ go to {folder_text} inside the title"));
            }
            Decision::Map => match platform::define(letter, &folder_text) {
                Ok(()) => {
                    drive.mapped = true;
                    log.line(&format!("game drive {letter}: mapped to {folder_text}"));
                }
                Err(e) => {
                    drive.redirect = Some(letter);
                    log.line(&format!("game drive {letter}: could not be mapped to {folder_text} ({e}), redirect mode: paths under {letter}:/ go there inside the title"));
                }
            },
        }
        drive
    }

    /// The letter whose paths the debugger redirects inside the title, when that mode is chosen.
    pub fn redirect(&self) -> Option<char> {
        self.redirect
    }

    /// Whether this guard made the mapping (and so removes it).
    pub fn mapped(&self) -> bool {
        self.mapped
    }

    /// Removes the mapping now, with one log line. Dropping does the same without a line.
    pub fn release(&mut self, log: &mut dyn Sink) {
        if self.mapped {
            match platform::remove(self.letter, &self.folder) {
                Ok(()) => log.line(&format!("game drive {}: removed", self.letter)),
                Err(e) => log.line(&format!("game drive {}: could not be removed ({e})", self.letter)),
            }
            self.mapped = false;
        }
    }
}

impl Drop for GameDrive {
    fn drop(&mut self) {
        if self.mapped {
            let _ = platform::remove(self.letter, &self.folder);
            self.mapped = false;
        }
    }
}

/// The folder as an absolute path with backslashes and without the `\\?\` prefix.
fn plain_path(folder: &Path) -> String {
    let abs = std::fs::canonicalize(folder).unwrap_or_else(|_| folder.to_path_buf());
    let text = abs.to_string_lossy().replace('/', "\\");
    text.strip_prefix("\\\\?\\").map(str::to_owned).unwrap_or(text)
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use windows_sys::Win32::Storage::FileSystem::{DefineDosDeviceW, QueryDosDeviceW, DDD_EXACT_MATCH_ON_REMOVE, DDD_REMOVE_DEFINITION};

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// The first target of `X:`, `None` when the letter is free.
    pub fn query(letter: char) -> Option<String> {
        let name = wide(&format!("{letter}:"));
        let mut buffer = vec![0u16; 1024];
        // SAFETY: `name` is NUL-terminated and `buffer` holds the length given.
        let n = unsafe { QueryDosDeviceW(name.as_ptr(), buffer.as_mut_ptr(), buffer.len() as u32) };
        if n == 0 {
            return None;
        }
        let first = buffer[..n as usize].split(|&c| c == 0).next().unwrap_or(&[]);
        Some(String::from_utf16_lossy(first))
    }

    pub fn define(letter: char, folder: &str) -> Result<(), std::io::Error> {
        let (name, target) = (wide(&format!("{letter}:")), wide(folder));
        // SAFETY: both strings are NUL-terminated buffers of this frame.
        if unsafe { DefineDosDeviceW(0, name.as_ptr(), target.as_ptr()) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    pub fn remove(letter: char, folder: &str) -> Result<(), std::io::Error> {
        let (name, target) = (wide(&format!("{letter}:")), wide(folder));
        // SAFETY: both strings are NUL-terminated buffers of this frame.
        if unsafe { DefineDosDeviceW(DDD_REMOVE_DEFINITION | DDD_EXACT_MATCH_ON_REMOVE, name.as_ptr(), target.as_ptr()) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(not(windows))]
#[allow(unsafe_code)]
mod platform {
    pub fn query(_: char) -> Option<String> {
        None
    }
    pub fn define(_: char, _: &str) -> Result<(), std::io::Error> {
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }
    pub fn remove(_: char, _: &str) -> Result<(), std::io::Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_letter_comes_from_the_environment_value() {
        assert_eq!(drive_letter(None), Some('G'));
        assert_eq!(drive_letter(Some("")), Some('G'));
        assert_eq!(drive_letter(Some("h")), Some('H'));
        assert_eq!(drive_letter(Some(" Z: ")), Some('Z'));
        assert_eq!(drive_letter(Some("nonsense")), Some('G'));
        for off in ["0", "off", "FALSE", " No "] {
            assert_eq!(drive_letter(Some(off)), None, "{off}");
        }
    }

    #[test]
    fn the_decision_follows_the_query_result() {
        assert_eq!(decide(None, r"D:\title_overlay"), Decision::Map);
        assert_eq!(decide(Some(r"\??\D:\title_overlay"), r"D:\title_overlay"), Decision::Keep);
        assert_eq!(decide(Some(r"\??\d:\title_overlay\"), "D:/title_overlay"), Decision::Keep);
        assert_eq!(decide(Some(r"\Device\HarddiskVolume3"), r"D:\title_overlay"), Decision::Taken(r"\Device\HarddiskVolume3".into()));
        assert_eq!(decide(Some(r"\??\D:\Other"), r"D:\title_overlay"), Decision::Taken(r"\??\D:\Other".into()));
    }

    #[test]
    fn only_the_word_redirect_forces_the_redirect_mode() {
        assert!(redirect_forced(Some("redirect")));
        assert!(redirect_forced(Some(" Redirect ")));
        for other in [None, Some(""), Some("map"), Some("1")] {
            assert!(!redirect_forced(other));
        }
    }

    #[test]
    fn a_forced_redirect_maps_nothing_and_says_so_once() {
        let mut log: Vec<String> = Vec::new();
        let drive = GameDrive::acquire_with(Some('G'), true, Path::new("."), &mut log);
        assert!(!drive.mapped() && drive.redirect() == Some('G'));
        assert_eq!(log.len(), 1);
        assert!(log[0].contains("redirect mode"), "{log:#?}");
        let off = GameDrive::acquire_with(None, true, Path::new("."), &mut log);
        assert!(off.redirect().is_none());
    }

    #[test]
    fn a_switched_off_drive_maps_nothing() {
        let mut log: Vec<String> = Vec::new();
        let drive = GameDrive::acquire_letter(None, Path::new("."), &mut log);
        assert!(!drive.mapped() && log.is_empty());
    }

    #[cfg(windows)]
    fn free_letter() -> (char, std::sync::MutexGuard<'static, ()>) {
        // The tests share the drive letters of the session, so they take turns.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        (('D'..='Z').rev().find(|&c| platform::query(c).is_none()).expect("a free drive letter"), guard)
    }

    #[cfg(windows)]
    #[test]
    fn a_free_letter_is_mapped_for_the_guard_and_removed_after() {
        let (letter, _turn) = free_letter();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.dat"), b"x").unwrap();
        let mut log: Vec<String> = Vec::new();
        let mut drive = GameDrive::acquire_letter(Some(letter), dir.path(), &mut log);
        assert!(drive.mapped(), "{log:#?}");
        assert!(Path::new(&format!("{letter}:/package.dat")).exists());
        assert!(log.iter().any(|l| l.contains("mapped to")));
        drive.release(&mut log);
        assert!(!Path::new(&format!("{letter}:/package.dat")).exists());
        assert!(log.iter().any(|l| l.contains("removed")));
        assert!(platform::query(letter).is_none());
        // The guard alone removes the mapping when it is dropped.
        let drive = GameDrive::acquire_letter(Some(letter), dir.path(), &mut log);
        assert!(drive.mapped());
        drop(drive);
        assert!(platform::query(letter).is_none());
    }

    #[cfg(windows)]
    #[test]
    fn a_letter_that_points_elsewhere_is_left_alone() {
        let (letter, _turn) = free_letter();
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let mut log: Vec<String> = Vec::new();
        let first = GameDrive::acquire_letter(Some(letter), a.path(), &mut log);
        let kept = GameDrive::acquire_letter(Some(letter), a.path(), &mut log);
        assert!(first.mapped() && !kept.mapped());
        let taken = GameDrive::acquire_letter(Some(letter), b.path(), &mut log);
        assert!(!taken.mapped());
        assert!(log.iter().any(|l| l.contains("is taken by")), "{log:#?}");
        drop((kept, taken));
        assert!(platform::query(letter).is_some());
        drop(first);
        assert!(platform::query(letter).is_none());
    }
}
