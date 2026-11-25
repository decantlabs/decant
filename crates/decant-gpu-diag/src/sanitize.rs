//! Placeholders for anything that could identify a person or a machine. The record holds no field
//! for such data, but text from the system (adapter and CPU names, driver strings) is scrubbed all
//! the same before it is shown.

/// Known identifiers of this run, replaced wherever they appear, plus shapes that look like paths
/// or serial numbers.
pub struct Scrubber {
    known: Vec<(String, &'static str)>,
}

/// Tokens of at least this many hex digits read as serial numbers or keys.
const LONG_ID: usize = 12;

impl Scrubber {
    /// A scrubber for the given user, machine and local paths. Empty or very short values are
    /// skipped, so a one letter name does not blank out ordinary words.
    pub fn new(user: &[String], machine: &[String], paths: &[String]) -> Scrubber {
        let mut known = Vec::new();
        for (values, placeholder) in [(paths, "<path>"), (user, "<user>"), (machine, "<machine>")] {
            for v in values {
                let v = v.trim().trim_end_matches(['\\', '/']);
                if v.chars().count() >= 3 {
                    known.push((v.to_owned(), placeholder));
                }
            }
        }
        // Longest first, so a path is replaced before the user name inside it.
        known.sort_by_key(|(v, _)| std::cmp::Reverse(v.len()));
        Scrubber { known }
    }

    /// A scrubber for this process: user, machine and home from the environment, plus `paths`.
    pub fn for_this_run(paths: &[String]) -> Scrubber {
        let env = |name: &str| std::env::var(name).ok().into_iter().collect::<Vec<_>>();
        let user = [env("USERNAME"), env("USER")].concat();
        let machine = [env("COMPUTERNAME"), env("USERDOMAIN"), env("HOSTNAME")].concat();
        let mut local = paths.to_vec();
        local.extend([env("USERPROFILE"), env("HOME")].concat());
        Scrubber::new(&user, &machine, &local)
    }

    pub fn clean(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for (value, placeholder) in &self.known {
            out = replace_ignore_case(&out, value, placeholder);
        }
        scrub_shapes(&out)
    }
}

fn replace_ignore_case(text: &str, needle: &str, with: &str) -> String {
    let (low_text, low_needle) = (text.to_lowercase(), needle.to_lowercase());
    // Lower casing can change byte lengths outside ASCII; fall back to an exact match then.
    if low_text.len() != text.len() || low_needle.len() != needle.len() {
        return text.replace(needle, with);
    }
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    while let Some(found) = low_text[at..].find(&low_needle) {
        out.push_str(&text[at..at + found]);
        out.push_str(with);
        at += found + needle.len();
    }
    out.push_str(&text[at..]);
    out
}

fn is_path_start(token: &str) -> bool {
    let b = token.as_bytes();
    let drive = b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/');
    // A known path already replaced, with more of the path after it.
    let rest_of_known = token.starts_with("<path>") && token.len() > "<path>".len();
    drive || rest_of_known || token.starts_with("\\\\") || ["/home/", "/Users/", "/root", "/mnt/", "/tmp/"].iter().any(|p| token.starts_with(p))
}

fn is_guid(token: &str) -> bool {
    let t = token.trim_matches(['{', '}', '(', ')']);
    let parts: Vec<&str> = t.split('-').collect();
    parts.iter().map(|p| p.len()).eq([8, 4, 4, 4, 12]) && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_hexdigit()))
}

fn is_long_id(token: &str) -> bool {
    let t = token.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    t.len() >= LONG_ID && t.chars().all(|c| c.is_ascii_hexdigit()) && t.chars().any(|c| c.is_ascii_digit())
}

/// Replaces whole tokens shaped like paths, GUIDs, long serial numbers or mail addresses. A path
/// swallows the tokens after it that still contain a separator, so `C:\Program Files\x` goes as
/// one.
fn scrub_shapes(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_path = false;
    for token in text.split(' ') {
        if in_path && token.contains(['\\', '/']) {
            continue;
        }
        in_path = false;
        if is_path_start(token) {
            in_path = true;
            out.push("<path>".to_owned());
        } else if is_guid(token) || is_long_id(token) || token.contains('@') {
            out.push("<id>".to_owned());
        } else {
            out.push(token.to_owned());
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invented() -> Scrubber {
        Scrubber::new(&["jdoe-test".to_owned()], &["WS-4471".to_owned()], &[r"D:\work\decant\build\".to_owned(), r"C:\Profiles\jdoe-test".to_owned()])
    }

    #[test]
    fn invented_identifiers_become_placeholders() {
        let s = invented();
        let cases = [
            (r"opened D:\work\decant\build\loadtest.exe", "opened <path>"),
            (r"profile C:\Profiles\jdoe-test\AppData\x.dll", "profile <path>"),
            ("user JDOE-TEST on ws-4471", "user <user> on <machine>"),
            (r"backend C:\Program Files\Vendor Kit\loader.dll found", "backend <path> found"),
            (r"share \\filer\gpu\log.txt", "share <path>"),
            ("home /home/jdoe-test/x", "home <path>"),
            ("serial 0123456789ABCDEF", "serial <id>"),
            ("adapter {6B29FC40-CA47-1067-B31D-00DD010662DA}", "adapter <id>"),
            ("mail jdoe@example.invalid", "mail <id>"),
        ];
        for (input, expected) in cases {
            assert_eq!(s.clean(input), expected, "{input}");
        }
    }

    #[test]
    fn public_hardware_and_versions_are_kept() {
        let s = invented();
        for kept in ["NVIDIA GeForce RTX 3090", "Intel(R) Core(TM) i9-13900K", "32.0.16.1742", "Windows 11 Pro 24H2 build 26300.1000", "Microsoft Basic Render Driver", "gpu_diag 0.4.0, commit 1a2b3c4"] {
            assert_eq!(s.clean(kept), kept);
        }
    }

    #[test]
    fn short_values_are_not_used_as_identifiers() {
        let s = Scrubber::new(&["a".to_owned()], &["PC".to_owned()], &[]);
        assert_eq!(s.clean("a PC and a GPU"), "a PC and a GPU");
    }

    #[test]
    fn a_scrubbed_record_holds_no_invented_identifier() {
        let mut record = crate::record::tests::sample();
        record.adapters[0].name = r"Invented GPU at D:\work\decant\build for jdoe-test".to_owned();
        record.system.cpu = "Invented CPU serial 00112233445566778899".to_owned();
        record.tool = "gpu_diag 0.4.0 on WS-4471".to_owned();
        let scrubbed = record.scrubbed(&invented());
        for text in [scrubbed.markdown(), scrubbed.json()] {
            for secret in ["jdoe", "WS-4471", r"D:\work", "00112233445566778899"] {
                assert!(!text.to_lowercase().contains(&secret.to_lowercase()), "{secret} left in:\n{text}");
            }
        }
    }
}
