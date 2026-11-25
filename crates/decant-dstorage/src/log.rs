//! `StorageLog` of the C++: a line to the session log and to standard error, cut at 511 bytes as
//! the 512 byte buffer of `_vsnprintf_s(..., _TRUNCATE, ...)` cut it.

use std::io::Write;

/// Longest text written, in bytes.
const MAX_BYTES: usize = 511;

/// The text as the C buffer held it: at most `MAX_BYTES` bytes, never cut inside a character.
fn truncated(text: &str) -> &str {
    if text.len() <= MAX_BYTES {
        return text;
    }
    let mut end = MAX_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Writes one log message. It carries its own line end, as the C format strings did.
pub fn storage_log(text: &str) {
    let text = truncated(text);
    decant_log::session_log(text);
    let _ = std::io::stderr().write_all(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_text_is_cut_at_511_bytes() {
        assert_eq!(truncated("short\n"), "short\n");
        assert_eq!(truncated(&"a".repeat(600)).len(), 511);
        assert_eq!(truncated(&"a".repeat(511)).len(), 511);
        let wide = "\u{e9}".repeat(300);
        assert_eq!(truncated(&wide).len(), 510);
    }
}
