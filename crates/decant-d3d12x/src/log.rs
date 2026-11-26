//! The log of the layer. Every line goes to `OutputDebugString` (`WINEDEBUG=+debugstr` under Wine)
//! and to the Decant session log (`DECANT_LOG_PATH`, set by the launcher), as `xs_log` did.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::fmt;

use windows_sys::core::GUID;

/// Writes one log line, `d3d12_xs: <text>` and a line feed.
macro_rules! xs_log {
    ($($arg:tt)*) => { $crate::log::write(format_args!($($arg)*)) };
}
pub(crate) use xs_log;

/// Longest text of a line: `wvsprintfA` of the C version cut the text at 1000 characters to leave
/// room for the prefix, and a debugger reads a debug string of about that length at most.
const MAX_TEXT: usize = 1000;
/// What a line starts with that goes on from the line before it.
const MORE: &str = "... ";

/// `text` in pieces of at most `max` bytes, cut between characters. A text that fits is one piece.
pub(crate) fn pieces(text: &str, max: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while rest.len() > max {
        let mut end = max;
        while end > 0 && !rest.is_char_boundary(end) {
            end -= 1;
        }
        if end == 0 {
            break;
        }
        out.push(&rest[..end]);
        rest = &rest[end..];
    }
    out.push(rest);
    out
}

struct StackBuf {
    buf: [u8; 1024],
    len: usize,
    overflow: bool,
}

impl StackBuf {
    fn new(prefix: &str) -> Self {
        let mut buf = [0u8; 1024];
        let n = prefix.len();
        buf[..n].copy_from_slice(prefix.as_bytes());
        Self { buf, len: n, overflow: false }
    }

    fn as_str(&self) -> &str {
        // SAFETY: only valid UTF-8 (`prefix`, `fmt::Write`, and `"\n"`) is written into `buf[..len]`.
        unsafe { std::str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

impl fmt::Write for StackBuf {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.overflow || self.len + s.len() > self.buf.len() - 1 {
            self.overflow = true;
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + s.len()].copy_from_slice(s.as_bytes());
        self.len += s.len();
        Ok(())
    }
}

/// Writes the text as one line, or as several when it is longer than a line holds: the lines after
/// the first start with `... ` (a long probe line used to lose everything past 1000 characters).
pub(crate) fn write(args: fmt::Arguments<'_>) {
    const PREFIX: &str = "d3d12_xs: ";
    let mut stack = StackBuf::new(PREFIX);
    if fmt::Write::write_fmt(&mut stack, args).is_ok() && stack.len - PREFIX.len() <= MAX_TEXT - MORE.len() {
        stack.buf[stack.len] = b'\n';
        stack.len += 1;
        let line = stack.as_str();
        decant_log::debug_string(line);
        decant_log::session_log(line);
        return;
    }
    let text = args.to_string();
    for (i, piece) in pieces(&text, MAX_TEXT - MORE.len()).into_iter().enumerate() {
        let line = format!("{PREFIX}{}{piece}\n", if i == 0 { "" } else { MORE });
        decant_log::debug_string(&line);
        decant_log::session_log(&line);
    }
}

/// A pointer as `wsprintfA("%p")` printed it: 16 upper case hex digits.
pub(crate) struct Ptr(pub usize);

impl Ptr {
    pub(crate) fn of<T>(pointer: *const T) -> Ptr {
        Ptr(pointer as usize)
    }
}

impl fmt::Display for Ptr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016X}", self.0)
    }
}

/// A GUID in the layout of the log lines: `189819F1-1DB6-4B57-BE54-1821339B85F7`.
pub(crate) fn guid_text(guid: &GUID) -> String {
    let d = guid.data4;
    format!(
        "{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        guid.data1, guid.data2, guid.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    )
}

/// The GUID behind `riid`.
///
/// # Safety
/// `riid` points to a readable GUID, as every D3D12 creation call requires.
pub(crate) unsafe fn guid_at(riid: *const GUID) -> String {
    // SAFETY: by the contract above.
    guid_text(unsafe { &*riid })
}

/// For `*const c_void` arguments.
pub(crate) fn p(pointer: *const c_void) -> Ptr {
    Ptr::of(pointer)
}

#[cfg(test)]
mod tests {
    use super::pieces;

    #[test]
    fn a_long_text_is_cut_into_pieces_between_characters() {
        assert_eq!(pieces("short", 10), vec!["short"]);
        assert_eq!(pieces("", 10), vec![""]);
        assert_eq!(pieces("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        // two byte characters are not cut in the middle
        assert_eq!(pieces("ääää", 3), vec!["ä", "ä", "ä", "ä"]);
        assert_eq!(pieces("abcd", 4), vec!["abcd"]);
    }

    #[test]
    fn stack_buf_formats_short_lines_without_overflow() {
        use std::fmt::Write;
        let mut b = super::StackBuf::new("d3d12_xs: ");
        write!(b, "value={}", 42).unwrap();
        assert_eq!(b.as_str(), "d3d12_xs: value=42");
        assert!(write!(b, "{}", "x".repeat(1024)).is_err());
    }
}
