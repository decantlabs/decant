//! The decoder test bench `xbox_decode_test_xbox.exe`, run as a separate process exactly as the
//! scripts run it: `<exe> <pack> <tmp>/r.md <shader name>`, keeping only its standard output.
//! It is a separate program and stays outside this crate.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct Decoder {
    pub exe: PathBuf,
    pub pack: PathBuf,
}

impl Decoder {
    /// The listing of one shader of the pack, as `subprocess.run(..., capture_output=True,
    /// text=True, errors="replace").stdout`. The exit code is ignored, as in the scripts.
    pub fn listing(&self, tmp: &Path, name: &str) -> Result<String, String> {
        let output = Command::new(&self.exe)
            .arg(&self.pack)
            .arg(tmp.join("r.md"))
            .arg(name)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("{}: {e}", self.exe.display()))?;
        Ok(text_mode(&output.stdout))
    }
}

/// Bytes read in Python's text mode: undecodable bytes replaced, CRLF and lone CR made LF.
pub fn text_mode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n").replace('\r', "\n")
}

/// `str.splitlines()`: every line boundary Python knows, no empty piece after a final boundary.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        match rest.char_indices().find(|&(_, c)| matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}'..='\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}')) {
            Some((i, c)) => {
                out.push(&rest[..i]);
                let mut next = i + c.len_utf8();
                if c == '\r' && rest[next..].starts_with('\n') {
                    next += 1;
                }
                rest = &rest[next..];
            }
            None => {
                out.push(rest);
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_split_like_python() {
        assert_eq!(splitlines("a\nb\n"), ["a", "b"]);
        assert_eq!(splitlines("a\r\n\nb\u{b}c\u{2028}"), ["a", "", "b", "c"]);
        assert!(splitlines("").is_empty());
        assert_eq!(text_mode(b"x\r\ny\rz\xff"), "x\ny\nz\u{fffd}");
    }
}
