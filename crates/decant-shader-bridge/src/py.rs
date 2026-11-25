//! The bits of Python behaviour the bridge output depends on: whitespace, `int()`, `repr()`, text
//! mode line ends, and the exceptions a run can end with.

use std::fmt;
use std::path::Path;

/// `str.isspace()` for one character. Python counts the four separators U+001C to U+001F as
/// whitespace, Rust does not.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// `str.split()` without arguments.
pub fn split(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|f| !f.is_empty()).collect()
}

/// `str.splitlines()`: every line boundary Python knows, with CR LF as one.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            lines.push(&text[start..at]);
            start = at + c.len_utf8();
            if c == '\r' && chars.peek().map(|&(_, n)| n) == Some('\n') {
                chars.next();
                start += 1;
            }
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Text as a pipe gives it to `subprocess.run(text=True)`: CR LF and lone CR become LF.
pub fn universal_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// Line ends as Python's text mode writes them: CR LF on Windows, LF elsewhere.
pub fn native_lines(text: &str) -> String {
    if cfg!(windows) { text.replace('\n', "\r\n") } else { text.to_owned() }
}

/// `text[-n:]`: the last `n` characters.
pub fn tail(text: &str, n: usize) -> &str {
    match text.char_indices().rev().nth(n.saturating_sub(1)) {
        Some((at, _)) if n > 0 => &text[at..],
        _ if n == 0 => "",
        _ => text,
    }
}

/// `int(text)`: optional sign, decimal digits with single underscores between them, surrounding
/// whitespace allowed.
pub fn int(text: &str) -> Result<i64, Failure> {
    parse_int(text, 10).ok_or_else(|| Failure::value(format!("invalid literal for int() with base 10: {}", repr_str(text))))
}

/// `int(text, 16)`, which also takes a `0x` prefix.
pub fn int16(text: &str) -> Result<i64, Failure> {
    parse_int(text, 16).ok_or_else(|| Failure::value(format!("invalid literal for int() with base 16: {}", repr_str(text))))
}

fn parse_int(text: &str, radix: u32) -> Option<i64> {
    let body = strip(text);
    let (negative, body) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let body = if radix == 16 { body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")).map(|b| b.strip_prefix('_').unwrap_or(b)).unwrap_or(body) } else { body };
    if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let digits: String = body.chars().filter(|&c| c != '_').collect();
    if !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    let value = i64::from_str_radix(&digits, radix).ok()?;
    Some(if negative { -value } else { value })
}

/// `repr()` of a string: single quotes unless the text holds a single quote and no double quote.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (0x80..0xa0).contains(&(c as u32)) => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `str()` of an optional value, `None` when absent.
pub fn opt<T: fmt::Display>(value: &Option<T>) -> String {
    value.as_ref().map_or_else(|| "None".to_owned(), ToString::to_string)
}

/// `repr()` of a JSON value once `json.loads` has read it.
pub fn repr_json(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i.to_string(),
            (_, Some(u), _) => u.to_string(),
            (_, _, Some(f)) if f.fract() == 0.0 && f.abs() < 1e16 => format!("{f:.1}"),
            (_, _, Some(f)) => f.to_string(),
            _ => n.to_string(),
        },
        Value::String(s) => repr_str(s),
        Value::Array(items) => format!("[{}]", items.iter().map(repr_json).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr_json(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// `seq[i]` with Python's negative indexes.
pub fn index<T>(items: &[T], i: i64) -> Result<&T, Failure> {
    let at = if i < 0 { i.checked_add(items.len() as i64) } else { Some(i) };
    at.and_then(|at| usize::try_from(at).ok()).and_then(|at| items.get(at)).ok_or_else(|| Failure::error("IndexError", "list index out of range"))
}

/// How a Python run ends early.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// `raise SystemExit(message)`: the message on stderr and exit status 1. `except Exception`
    /// does not catch it.
    Exit(String),
    /// Any other exception, by its `repr()` (what `{error!r}` prints) and the last line of its
    /// traceback.
    Error { repr: String, line: String },
}

impl Failure {
    /// An exception built from one message, such as `RuntimeError('...')`.
    pub fn error(kind: &str, message: &str) -> Failure {
        let line = if message.is_empty() { kind.to_owned() } else { format!("{kind}: {message}") };
        let repr = if message.is_empty() { format!("{kind}()") } else { format!("{kind}({})", repr_str(message)) };
        Failure::Error { repr, line }
    }

    pub fn runtime(message: &str) -> Failure {
        Failure::error("RuntimeError", message)
    }

    pub fn value(message: String) -> Failure {
        Failure::error("ValueError", &message)
    }

    /// `KeyError` for a key whose `repr()` is given.
    pub fn key(key_repr: String) -> Failure {
        Failure::Error { line: format!("KeyError: {key_repr}"), repr: format!("KeyError({key_repr})") }
    }

    /// The `OSError` subclass Python raises for an I/O error on `path`.
    pub fn io(error: &std::io::Error, path: &Path) -> Failure {
        use std::io::ErrorKind;
        let (kind, errno, text) = match error.kind() {
            ErrorKind::NotFound => ("FileNotFoundError", 2, "No such file or directory".to_owned()),
            ErrorKind::PermissionDenied => ("PermissionError", 13, "Permission denied".to_owned()),
            _ => ("OSError", error.raw_os_error().unwrap_or(0), error.to_string()),
        };
        let path = repr_str(&path.to_string_lossy());
        Failure::Error { repr: format!("{kind}({errno}, {})", repr_str(&text)), line: format!("{kind}: [Errno {errno}] {text}: {path}") }
    }

    /// What reaches stderr: the message of `SystemExit`, the traceback's last line otherwise.
    pub fn stderr(&self) -> &str {
        match self {
            Failure::Exit(message) => message,
            Failure::Error { line, .. } => line,
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stderr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_helpers_follow_python() {
        assert_eq!(split(" a\tb \u{1c}c "), ["a", "b", "c"]);
        assert_eq!(splitlines("a\r\nb\rc\n\nd"), ["a", "b", "c", "", "d"]);
        assert_eq!(splitlines("a\n"), ["a"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(strip("\u{1f} x \n"), "x");
        assert_eq!(tail("abcdef", 3), "def");
        assert_eq!(tail("ab", 3), "ab");
        assert_eq!(tail("ab", 0), "");
        assert_eq!(universal_newlines("a\r\nb\rc"), "a\nb\nc");
    }

    #[test]
    fn integers_parse_like_int() {
        assert_eq!(int(" 12 ").unwrap(), 12);
        assert_eq!(int("+1_000").unwrap(), 1000);
        assert_eq!(int("-7").unwrap(), -7);
        assert_eq!(int16("0x1f").unwrap(), 31);
        assert_eq!(int16("1F").unwrap(), 31);
        assert!(int("1__0").is_err());
        assert!(int("_1").is_err());
        assert!(int("").is_err());
        assert_eq!(int("x").unwrap_err().stderr(), "ValueError: invalid literal for int() with base 10: 'x'");
    }

    #[test]
    fn reprs_match_python() {
        assert_eq!(repr_str("a'b"), "\"a'b\"");
        assert_eq!(repr_str("a'b\""), "'a\\'b\"'");
        assert_eq!(repr_str("t\\n\n"), "'t\\\\n\\n'");
        let value: serde_json::Value = serde_json::from_str(r#"{"a": [1, null, true, "x"], "b": 2.0}"#).unwrap();
        assert_eq!(repr_json(&value), "{'a': [1, None, True, 'x'], 'b': 2.0}");
        assert_eq!(Failure::runtime("x a echoue"), Failure::Error { repr: "RuntimeError('x a echoue')".into(), line: "RuntimeError: x a echoue".into() });
        assert_eq!(Failure::key("'DECANT_SHADERS_RPF'".into()).stderr(), "KeyError: 'DECANT_SHADERS_RPF'");
        assert_eq!(*index(&[1, 2, 3], -1).unwrap(), 3);
        assert!(index(&[1], 1).is_err());
    }
}
