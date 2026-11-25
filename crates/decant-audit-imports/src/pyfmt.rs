//! Numbers and paths written the way the Python auditor writes them, so both reports compare
//! byte for byte.

use std::path::{Component, Path, PathBuf};

/// `round(x, 2)`: the value of `x` rounded to two decimals.
pub fn round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}

/// `str(x)` for a float: the shortest form that reads back, with `.0` on whole numbers.
pub fn float(x: f64) -> String {
    let text = format!("{x}");
    if text.contains(['.', 'e', 'E']) || !x.is_finite() { text } else { format!("{text}.0") }
}

/// `str(Path(text))` on Windows: separators made native and trailing ones dropped.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir if path.components().count() > 1 => {}
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() { PathBuf::from(".") } else { out }
}

/// A path as text with the platform's separator.
pub fn text(path: &Path) -> String {
    let s = path.to_string_lossy().into_owned();
    if cfg!(windows) { s.replace('/', "\\") } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_read_like_python() {
        assert_eq!(float(round2(0.0)), "0.0");
        assert_eq!(float(round2(12.0)), "12.0");
        assert_eq!(float(round2(0.254_9)), "0.25");
        assert_eq!(float(round2(7.999_6)), "8.0");
        assert_eq!(float(round2(2.675)), "2.67", "the binary value of 2.675 is just below it");
    }

    #[test]
    fn paths_lose_trailing_separators() {
        assert_eq!(text(&normalize(Path::new("a/b/"))), if cfg!(windows) { "a\\b" } else { "a/b" });
        assert_eq!(text(&normalize(Path::new("./x"))), "x");
        assert_eq!(text(&normalize(Path::new("."))), ".");
    }
}
