//! The text conversions of the Microsoft C runtime that the C++ benches relied on, so the Rust
//! benches read their arguments and print their numbers exactly as they did.

/// `printf("%.4f", x)` for a `float` promoted to `double`, as the Universal CRT writes it:
/// correctly rounded (ties to even on the exact binary value), `inf`, and `nan`, `-nan(ind)` or
/// `-nan` for the NaNs (the promotion quiets a signalling NaN, the payload decides the rest).
pub fn fixed4(x: f32) -> String {
    if x.is_nan() {
        let bits = x.to_bits();
        let negative = bits >> 31 == 1;
        let payload = bits & 0x003F_FFFF; // mantissa without the quiet bit
        return match (negative, payload) {
            (false, _) => "nan",
            (true, 0) => "-nan(ind)",
            (true, _) => "-nan",
        }
        .to_owned();
    }
    format!("{:.4}", f64::from(x))
}

/// Four components joined by spaces, as `"%.4f %.4f %.4f %.4f"`.
pub fn fixed4_rgba(pixel: &[f32]) -> String {
    pixel
        .iter()
        .take(4)
        .map(|&v| fixed4(v))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Line endings of a C stream in text mode: `\r\n` on Windows, `\n` elsewhere.
pub fn text_mode(text: &str) -> String {
    if cfg!(windows) {
        text.replace('\n', "\r\n")
    } else {
        text.to_owned()
    }
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r')
}

/// `atoi`: leading white space, an optional sign, decimal digits, 0 without digits. The Universal
/// CRT clamps an overflow to `INT_MAX` or `INT_MIN`.
pub fn atoi(text: &str) -> i32 {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_space(bytes[i]) {
        i += 1;
    }
    let negative = i < bytes.len() && bytes[i] == b'-';
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        i += 1;
    }
    let mut value: i64 = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = (value * 10 + i64::from(bytes[i] - b'0')).min(1 << 32);
        i += 1;
    }
    let value = if negative { -value } else { value };
    value.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// `strtoul(text, nullptr, 0)` with a 32 bit `unsigned long`: base 16 after `0x`, base 8 after a
/// leading `0`, base 10 otherwise, a minus sign negates modulo 2^32, an overflow gives
/// `ULONG_MAX`, 0 without digits.
pub fn strtoul0(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_space(bytes[i]) {
        i += 1;
    }
    let negative = i < bytes.len() && bytes[i] == b'-';
    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
        i += 1;
    }
    let mut base = 10;
    if i < bytes.len() && bytes[i] == b'0' {
        let hex = bytes.get(i + 1).is_some_and(|&c| c == b'x' || c == b'X')
            && bytes.get(i + 2).is_some_and(u8::is_ascii_hexdigit);
        if hex {
            base = 16;
            i += 2;
        } else {
            base = 8;
        }
    }
    let mut value: u64 = 0;
    let mut overflow = false;
    while i < bytes.len() {
        let Some(digit) = (bytes[i] as char).to_digit(base) else {
            break;
        };
        value = value * u64::from(base) + u64::from(digit);
        if value > u64::from(u32::MAX) {
            overflow = true;
            value = u64::from(u32::MAX);
        }
        i += 1;
    }
    if overflow {
        return u32::MAX;
    }
    let value = value as u32;
    if negative {
        value.wrapping_neg()
    } else {
        value
    }
}

/// Reads whitespace separated fields the way `std::istream >>` does on a text file opened by
/// the Microsoft C runtime: a Ctrl Z ends the file, numbers stop at the first character that
/// cannot continue them, and a failed extraction stops every later one.
pub struct Fields<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Fields<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        let end = bytes.iter().position(|&b| b == 0x1A).unwrap_or(bytes.len());
        Self {
            bytes: &bytes[..end],
            pos: 0,
        }
    }

    fn skip_space(&mut self) {
        while self.pos < self.bytes.len() && is_space(self.bytes[self.pos]) {
            self.pos += 1;
        }
    }

    /// `>> std::string`.
    pub fn word(&mut self) -> Option<String> {
        self.skip_space();
        let start = self.pos;
        while self.pos < self.bytes.len() && !is_space(self.bytes[self.pos]) {
            self.pos += 1;
        }
        (self.pos > start)
            .then(|| String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned())
    }

    /// Sign and decimal digits of a number, as `num_get` collects them.
    fn number(&mut self) -> Option<(bool, u64)> {
        self.skip_space();
        let mut negative = false;
        if self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b'-' || self.bytes[self.pos] == b'+')
        {
            negative = self.bytes[self.pos] == b'-';
            self.pos += 1;
        }
        let start = self.pos;
        let mut value: u64 = 0;
        let mut overflow = false;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            value = value * 10 + u64::from(self.bytes[self.pos] - b'0');
            if value > 1 << 33 {
                overflow = true;
                value = 1 << 33;
            }
            self.pos += 1;
        }
        (self.pos > start && !overflow).then_some((negative, value))
    }

    /// `>> uint32_t`: a minus sign negates modulo 2^32, as `strtoul` does.
    pub fn unsigned(&mut self) -> Option<u32> {
        let (negative, value) = self.number()?;
        let value = u32::try_from(value).ok()?;
        Some(if negative {
            value.wrapping_neg()
        } else {
            value
        })
    }

    /// `>> int`.
    pub fn signed(&mut self) -> Option<i32> {
        let (negative, value) = self.number()?;
        let value = if negative {
            -(value as i64)
        } else {
            value as i64
        };
        i32::try_from(value).ok()
    }
}
