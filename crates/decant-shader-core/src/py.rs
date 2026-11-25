//! Text read and written the way the Python tools read and write it, so that the ports print the
//! same lines and write the same files.

use serde_json::Value;

/// Line ends of a file the Python wrote in text mode: `\n` becomes `\r\n` on Windows.
pub fn native_newlines(text: &str) -> String {
    if cfg!(windows) { text.replace('\n', "\r\n") } else { text.to_owned() }
}

/// The lines of a file as Python iterates a file opened in text mode: `\r\n`, `\r` and `\n` all end
/// a line. Line ends are not included and there is no empty line after the last end.
pub fn universal_lines(bytes: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&bytes[start..i]);
                start = i + 1;
            }
            b'\r' => {
                lines.push(&bytes[start..i]);
                if bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < bytes.len() {
        lines.push(&bytes[start..]);
    }
    lines
}

/// `Path.read_text(errors="replace")`: the bytes in the locale's code page with replacement
/// characters, and every `\r\n` or `\r` made `\n`. On Windows the code page is taken as
/// Windows 1252, elsewhere as UTF 8.
pub fn read_text(bytes: &[u8]) -> String {
    let text = if cfg!(windows) { cp1252(bytes) } else { String::from_utf8_lossy(bytes).into_owned() };
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// `Path.read_text(encoding="utf8", errors="replace")`: UTF 8 with replacement characters, and
/// every CR LF or lone CR made LF.
pub fn read_text_utf8(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n").replace('\r', "\n")
}

/// Windows 1252 as Python decodes it with `errors="replace"`. Five bytes have no character.
pub fn cp1252(bytes: &[u8]) -> String {
    const HIGH: [u16; 32] = [
        0x20AC, 0xFFFD, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0xFFFD, 0x017D, 0xFFFD,
        0xFFFD, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0xFFFD, 0x017E, 0x0178,
    ];
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => char::from_u32(u32::from(HIGH[usize::from(b - 0x80)])).unwrap_or('\u{FFFD}'),
            _ => char::from(b),
        })
        .collect()
}

/// `str.splitlines()`: every line boundary Python knows, with no empty line after the last one.
pub fn splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let boundary = matches!(c, '\n' | '\r' | '\u{0B}' | '\u{0C}' | '\u{1C}' | '\u{1D}' | '\u{1E}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if boundary {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = chars.peek()
            {
                chars.next();
                end = j + 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// `int(text)` for a decimal number: surrounding spaces, a sign and single underscores between
/// digits are allowed.
pub fn int(text: &str) -> Result<i64, String> {
    let bad = || format!("invalid literal for int() with base 10: {}", repr_str(text));
    let body = text.trim();
    let (negative, digits) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return Err(bad());
    }
    let mut value: i64 = 0;
    for c in digits.chars().filter(|&c| c != '_') {
        let d = c.to_digit(10).ok_or_else(bad)?;
        value = value.checked_mul(10).and_then(|v| v.checked_add(i64::from(d))).ok_or_else(bad)?;
    }
    Ok(if negative { -value } else { value })
}

/// Whether Python's `str.isprintable()` holds for one character. Control, format, separator and
/// private use characters are not printable, the space is. Unassigned code points are not known
/// here and count as printable.
fn printable(c: char) -> bool {
    !matches!(c,
        '\0'..='\u{1F}' | '\u{7F}'..='\u{A0}' | '\u{AD}' | '\u{600}'..='\u{605}' | '\u{61C}' | '\u{6DD}' | '\u{70F}'
        | '\u{1680}' | '\u{180E}' | '\u{2000}'..='\u{200F}' | '\u{2028}'..='\u{202F}' | '\u{205F}'..='\u{2064}'
        | '\u{2066}'..='\u{206F}' | '\u{3000}' | '\u{E000}'..='\u{F8FF}' | '\u{FEFF}' | '\u{FFF9}'..='\u{FFFB}'
        | '\u{F0000}'..='\u{10FFFF}')
}

/// `repr(text)` for a string.
pub fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if printable(c) => out.push(c),
            _ if u32::from(c) < 0x100 => out.push_str(&format!("\\x{:02x}", u32::from(c))),
            _ if u32::from(c) < 0x10000 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            _ => out.push_str(&format!("\\U{:08x}", u32::from(c))),
        }
    }
    out.push(quote);
    out
}

/// `str(x)` for a float: the shortest form that reads back, with `.0` on whole numbers.
fn float(x: f64) -> String {
    let text = format!("{x}");
    if text.contains(['.', 'e', 'E', 'N', 'n']) { text } else { format!("{text}.0") }
}

/// `str(value)` for a value read from JSON.
pub fn str_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => repr_value(other),
    }
}

/// `repr(value)` for a value read from JSON.
pub fn repr_value(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => repr_str(s),
        Value::Array(items) => format!("[{}]", items.iter().map(repr_value).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => {
            format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", repr_str(k), repr_value(v))).collect::<Vec<_>>().join(", "))
        }
    }
}

/// Python truth: `None`, `False`, zero, empty strings and empty lists are false.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `repr(dict)` for integer keys and values, in the order given.
pub fn repr_int_dict<'a>(pairs: impl IntoIterator<Item = (&'a i64, &'a i64)>) -> String {
    format!("{{{}}}", pairs.into_iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", "))
}

/// `repr(set)` of integers and `None`, added in the given order. The order of a printed set is the
/// order of CPython's hash table, so the table is rebuilt here as `setobject.c` (3.13) fills it.
pub fn repr_set(items: &[Option<i64>]) -> String {
    let order = set_order(items);
    if order.is_empty() {
        return "set()".into();
    }
    let text: Vec<String> = order.iter().map(|v| v.map_or("None".into(), |i| i.to_string())).collect();
    format!("{{{}}}", text.join(", "))
}

/// `hash(x)` for an integer that fits a machine word and for `None` (a constant since 3.12).
fn hash(item: Option<i64>) -> u64 {
    const MODULUS: i64 = (1 << 61) - 1;
    match item {
        None => 0xFCA8_6420,
        Some(i) => {
            let h = i.signum() * (i.unsigned_abs() % MODULUS as u64) as i64;
            (if h == -1 { -2 } else { h }) as u64
        }
    }
}

const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;

/// The iteration order of a set built by adding `items` one by one.
fn set_order(items: &[Option<i64>]) -> Vec<Option<i64>> {
    let mut table: Vec<Option<Option<i64>>> = vec![None; 8];
    let mut used = 0usize;
    for &item in items {
        let h = hash(item);
        let mask = table.len() - 1;
        let (mut i, mut perturb) = (h as usize & mask, h);
        let slot = 'search: loop {
            let probes = if i + LINEAR_PROBES <= mask { LINEAR_PROBES } else { 0 };
            for (j, entry) in table.iter().enumerate().skip(i).take(probes + 1) {
                match entry {
                    None => break 'search Some(j),
                    Some(existing) if *existing == item => break 'search None,
                    Some(_) => {}
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        };
        let Some(slot) = slot else { continue };
        table[slot] = Some(item);
        used += 1;
        if used * 5 >= mask * 3 {
            table = resize(&table, if used > 50_000 { used * 2 } else { used * 4 });
        }
    }
    table.into_iter().flatten().collect()
}

/// `set_table_resize`: a larger table, filled in the order of the old one.
fn resize(old: &[Option<Option<i64>>], minused: usize) -> Vec<Option<Option<i64>>> {
    let mut size = 8;
    while size <= minused {
        size <<= 1;
    }
    let mut table = vec![None; size];
    let mask = size - 1;
    for item in old.iter().flatten() {
        let h = hash(*item);
        let (mut i, mut perturb) = (h as usize & mask, h);
        let slot = 'search: loop {
            if table[i].is_none() {
                break i;
            }
            if i + LINEAR_PROBES <= mask
                && let Some(j) = (i + 1..=i + LINEAR_PROBES).find(|&j| table[j].is_none())
            {
                break 'search j;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb as usize)) & mask;
        };
        table[slot] = Some(*item);
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_print_in_cpython_order() {
        // Each expected text is what CPython 3.13 prints after `set.add` in the same order.
        assert_eq!(repr_set(&[]), "set()");
        assert_eq!(repr_set(&[None]), "{None}");
        assert_eq!(repr_set(&[Some(3), None, Some(0)]), "{None, 0, 3}");
        assert_eq!(repr_set(&[Some(8), None, Some(0)]), "{8, 0, None}");
        assert_eq!(repr_set(&[Some(1), Some(9), Some(17), Some(25), Some(33)]), "{1, 33, 9, 17, 25}");
        assert_eq!(repr_set(&[Some(40), None, Some(7), Some(15), Some(23), Some(-1)]), "{None, 7, 40, 15, 23, -1}");
        assert_eq!(repr_set(&[Some(2), Some(2), Some(1)]), "{1, 2}");
    }

    #[test]
    fn strings_repr_like_python() {
        assert_eq!(repr_str("a'b"), "\"a'b\"");
        assert_eq!(repr_str("a\"b'"), "'a\"b\\''");
        assert_eq!(repr_str("\u{85}\u{AD}\u{2028}\t\\é"), "'\\x85\\xad\\u2028\\t\\\\é'");
        assert_eq!(repr_value(&serde_json::json!(["12", "timeout 45s"])), "['12', 'timeout 45s']");
        assert_eq!(repr_value(&serde_json::json!([1, 2.5, 3.0, null, true])), "[1, 2.5, 3.0, None, True]");
    }

    #[test]
    fn lines_split_like_python() {
        assert_eq!(universal_lines(b"a\r\nb\rc\n\nd"), [&b"a"[..], b"b", b"c", b"", b"d"]);
        assert_eq!(splitlines("a\r\nb\u{85}c\x0bd\n"), ["a", "b", "c", "d"]);
        assert_eq!(read_text(b"x\r\ny\rz"), "x\ny\nz");
        assert_eq!(cp1252(b"\x80\x81\x9f\xe9"), "\u{20AC}\u{FFFD}\u{178}é");
    }

    #[test]
    fn integers_parse_like_python() {
        assert_eq!(int(" 12 "), Ok(12));
        assert_eq!(int("-1_000"), Ok(-1000));
        assert!(int("1__0").is_err());
        assert!(int("").is_err());
        assert!(int("0x10").is_err());
    }
}
