//! The RESULT and SUMMARY lines, formatted as the C version's `printf` formatted them, so that
//! `launch_prep` and anyone reading an old log see the same text.

use std::io::Write;

/// Lines end as the C runtime ends them in text mode: `\r\n` on Windows (also under Wine).
const NL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

/// Counts failures and prints one line per check, flushed at once (a crashing DLL must not take
/// the earlier lines with it).
pub struct Report<W: Write> {
    out: W,
    pub failures: i32,
}

impl<W: Write> Report<W> {
    pub fn new(out: W) -> Self {
        Self { out, failures: 0 }
    }

    /// `RESULT %-4s %-48s <detail>`, as the C macro `CHECK` printed it.
    pub fn check(&mut self, name: &str, ok: bool, detail: &str) {
        if !ok {
            self.failures += 1;
        }
        let status = if ok { "OK" } else { "FAIL" };
        self.write(&format!("RESULT {status:<4} {name:<48} {detail}"));
    }

    pub fn summary(&mut self) {
        self.write(&format!("SUMMARY failures={}", self.failures));
    }

    fn write(&mut self, line: &str) {
        // A closed stdout cannot be reported anywhere; the exit code still counts the failures.
        let _ = self.out.write_all(format!("{line}{NL}").as_bytes());
        let _ = self.out.flush();
    }
}

/// `%p` of the Microsoft C runtime (and of mingw-w64's msvcrt): 16 upper case hex digits.
pub fn pointer(address: usize) -> String {
    format!("{address:016X}")
}

/// `0x%08lx`.
pub fn hresult(hr: i32) -> String {
    format!("0x{:08x}", hr as u32)
}

/// `load %.46s`: the file name cut to 46 bytes.
pub fn load_label(file: &str) -> String {
    let mut end = file.len().min(46);
    while !file.is_char_boundary(end) {
        end -= 1;
    }
    format!("load {}", &file[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(run: impl FnOnce(&mut Report<&mut Vec<u8>>)) -> (String, i32) {
        let mut out = Vec::new();
        let mut report = Report::new(&mut out);
        run(&mut report);
        let failures = report.failures;
        (String::from_utf8(out).unwrap(), failures)
    }

    #[test]
    fn lines_match_the_c_printf() {
        let (text, failures) = lines(|r| {
            r.check("XMemAlloc graphics 16MiB", true, &pointer(0x1F2_A3B4_0000));
            r.check("d3d12.dll pulled in by forwarder", false, "");
            r.check(&load_label("xmem.dll"), true, "err=0");
            r.summary();
        });
        let expected = [
            "RESULT OK   XMemAlloc graphics 16MiB                         000001F2A3B40000",
            "RESULT FAIL d3d12.dll pulled in by forwarder                 ",
            "RESULT OK   load xmem.dll                                    err=0",
            "SUMMARY failures=1",
        ];
        assert_eq!(text, expected.map(|l| format!("{l}{NL}")).concat());
        assert_eq!(failures, 1);
    }

    #[test]
    fn formats() {
        assert_eq!(pointer(0), "0000000000000000");
        assert_eq!(hresult(-2147467263), "0x80004001");
        assert_eq!(hresult(0), "0x00000000");
        let long = "a".repeat(60);
        assert_eq!(load_label(&long), format!("load {}", "a".repeat(46)));
        // A name longer than the column still pushes the detail right, as printf does.
        let (text, _) = lines(|r| r.check(&"n".repeat(50), true, "x"));
        assert_eq!(text, format!("RESULT OK   {} x{NL}", "n".repeat(50)));
    }
}
