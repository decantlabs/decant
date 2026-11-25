//! The relaunch a title asks for with `XLaunchNewGame`. The title starts itself again through that
//! call, which on a console ends the process and starts the new one. Our `xgameruntime.dll`
//! (`decant-xgameruntime`, `local_game.rs`) writes the request to the file named by
//! [`FILE_ENV`] and ends the title with [`EXIT_CODE`]. The launcher then runs the title again,
//! under the same debugger and game drive, with the arguments of the request.
//!
//! The file: the executable on the first line, the arguments as one command line tail on the
//! second.

use std::path::{Path, PathBuf};

use crate::run::Outcome;

/// The variable that names the request file to the title (`local_game.rs`, same value).
pub const FILE_ENV: &str = "DECANT_RELAUNCH_FILE";
/// The exit code of a title that asks for a relaunch ("SXRL", `local_game.rs`, same value).
pub const EXIT_CODE: u32 = 0x5358_524C;
/// The request file, in the title's folder.
pub const FILE_NAME: &str = "decant_relaunch.txt";
/// The most relaunches of one run, so a title that asks again and again ends.
pub const MAX: usize = 8;

/// A relaunch request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// The executable the title named, empty for itself.
    pub exe: String,
    /// The arguments, split as the MSVC runtime splits a command line.
    pub args: Vec<String>,
}

/// The request in the file's text, `None` when it has no line.
pub fn parse(text: &str) -> Option<Request> {
    let mut lines = text.lines();
    let exe = lines.next()?.trim().to_owned();
    let args = split_args(lines.next().unwrap_or_default());
    Some(Request { exe, args })
}

/// Splits a command line tail as the MSVC runtime does: blanks separate, double quotes group,
/// backslashes escape a quote only when they come before one.
pub fn split_args(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_arg = false;
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let mut n = 1;
                while chars.peek() == Some(&'\\') {
                    chars.next();
                    n += 1;
                }
                if chars.peek() == Some(&'"') {
                    current.extend(std::iter::repeat_n('\\', n / 2));
                    if n % 2 == 1 {
                        chars.next();
                        current.push('"');
                    }
                } else {
                    current.extend(std::iter::repeat_n('\\', n));
                }
                in_arg = true;
            }
            '"' => {
                if quoted && chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    quoted = !quoted;
                }
                in_arg = true;
            }
            ' ' | '\t' if !quoted => {
                if in_arg {
                    out.push(std::mem::take(&mut current));
                    in_arg = false;
                }
            }
            _ => {
                current.push(c);
                in_arg = true;
            }
        }
    }
    if in_arg {
        out.push(current);
    }
    out
}

/// The request of a run that ended with [`EXIT_CODE`] and left the file, `None` otherwise (a
/// cancelled or timed out run never relaunches). The file is removed.
pub fn requested(outcome: &Outcome, file: &Path) -> Option<Request> {
    if outcome.exit_code != Some(EXIT_CODE) || outcome.cancelled || outcome.timed_out {
        return None;
    }
    let text = std::fs::read_to_string(file).ok()?;
    let _ = std::fs::remove_file(file);
    parse(&text)
}

/// The executable to start for `asked`: itself when empty, the path when it is a file, else the
/// file of that name in `folder` (a path on the game drive in redirect mode), else `current`.
pub fn executable(asked: &str, current: &Path, folder: &Path) -> PathBuf {
    if asked.is_empty() {
        return current.to_path_buf();
    }
    let path = Path::new(asked);
    if path.is_file() {
        return path.to_path_buf();
    }
    match path.file_name().map(|name| folder.join(name)) {
        Some(local) if local.is_file() => local,
        _ => current.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(code: u32) -> Outcome {
        Outcome { exit_code: Some(code), dlls: Vec::new(), timed_out: false, cancelled: false }
    }

    #[test]
    fn a_request_is_the_exe_and_its_arguments() {
        assert_eq!(parse("G:\\title.exe\n-a \"b c\" d\\\"e\n"), Some(Request { exe: "G:\\title.exe".into(), args: vec!["-a".into(), "b c".into(), "d\"e".into()] }));
        assert_eq!(parse("x.exe\r\n"), Some(Request { exe: "x.exe".into(), args: vec![] }));
        assert_eq!(parse(""), None);
    }

    #[test]
    fn arguments_split_as_the_msvc_runtime_does() {
        assert_eq!(split_args(r#"  a\\b "c d"  "" e\\"f g" "#), ["a\\\\b", "c d", "", "e\\f g"]);
        assert_eq!(split_args(r#"-x=1	-y"#), ["-x=1", "-y"]);
        assert!(split_args("   ").is_empty());
    }

    #[test]
    fn only_the_relaunch_code_with_a_file_relaunches() {
        let dir = std::env::temp_dir().join(format!("decant_launch_relaunch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(FILE_NAME);
        std::fs::write(&file, "t.exe\n-a\n").unwrap();
        assert_eq!(requested(&outcome(0), &file), None);
        let mut timed_out = outcome(EXIT_CODE);
        timed_out.timed_out = true;
        assert_eq!(requested(&timed_out, &file), None);
        assert_eq!(requested(&outcome(EXIT_CODE), &file), Some(Request { exe: "t.exe".into(), args: vec!["-a".into()] }));
        assert!(!file.exists(), "the request is taken once");
        assert_eq!(requested(&outcome(EXIT_CODE), &file), None);

        let current = dir.join("current.exe");
        std::fs::write(dir.join("title.exe"), b"").unwrap();
        assert_eq!(executable("", &current, &dir), current);
        assert_eq!(executable(r"G:\title.exe", &current, &dir), dir.join("title.exe"));
        assert_eq!(executable(r"G:\missing.exe", &current, &dir), current);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(EXIT_CODE.to_be_bytes(), *b"SXRL");
    }
}
