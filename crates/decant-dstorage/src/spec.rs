//! The console ABI spec: parsing of the text format of `docs/DSTORAGE_ABI.md`, no Windows calls.
//!
//! The interface IDs and the method order a console title expects are not in the public
//! documentation. Instead of hard coding a guess, they are read from a text spec: each console
//! interface lists its IIDs and, per vtable slot, the engine handler it maps to. The parser keeps
//! the rules of the C++ it replaces: the same directives, the same limits, the same error text
//! (`line N: reason`).

/// What a console interface is: the engine object behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Factory,
    File,
    Queue,
    Status,
}

/// The engine handlers a spec may map a slot to, each usable in its own role only.
pub const HANDLERS: &[(&str, Role)] = &[
    ("factory.create_queue", Role::Factory),
    ("factory.open_file", Role::Factory),
    ("factory.create_status_array", Role::Factory),
    ("factory.set_cpu_affinity", Role::Factory),
    ("factory.set_debug_flags", Role::Factory),
    ("factory.set_staging_buffer_size", Role::Factory),
    ("queue.enqueue_request", Role::Queue),
    ("queue.enqueue_status", Role::Queue),
    ("queue.enqueue_signal", Role::Queue),
    ("queue.enqueue_set_event", Role::Queue),
    ("queue.submit", Role::Queue),
    ("status.is_complete", Role::Status),
    ("status.get_hresult", Role::Status),
];

/// The role of a handler name, if it is in the catalogue.
pub fn handler_role(name: &str) -> Option<Role> {
    HANDLERS.iter().find(|(n, _)| *n == name).map(|&(_, role)| role)
}

/// Slots of a console vtable: 0 to 2 are `IUnknown`, a spec may name 3 to 63.
pub const MAX_SLOTS: u32 = 64;

/// A GUID as the number its text spells (`{00112233-4455-6677-8899-aabbccddeeff}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guid(pub u128);

/// One console interface of a spec.
#[derive(Clone, Debug)]
pub struct Iface {
    pub role: Role,
    pub label: String,
    /// Free word, logged (`template`, `lead`, `verified`, ...).
    pub status: String,
    pub iids: Vec<Guid>,
    /// A template: `iid unknown`, nothing can reach this interface yet.
    pub unknown_iid: bool,
    /// Slot and handler name as written (`factory.create_queue CreateQueue`, `?`).
    pub methods: Vec<(u32, String)>,
}

impl Iface {
    /// Whether this interface answers `iid`.
    pub fn accepts(&self, iid: Guid) -> bool {
        self.iids.contains(&iid)
    }
}

/// A parsed spec.
#[derive(Clone, Debug)]
pub struct Spec {
    /// Byte offset of the `u16` capacity in the console queue description.
    pub capacity_offset: u32,
    pub ifaces: Vec<Iface>,
}

impl Spec {
    /// The first interface of `role` that answers `iid`.
    pub fn find_role(&self, role: Role, iid: Guid) -> Option<&Iface> {
        self.ifaces.iter().find(|i| i.role == role && i.accepts(iid))
    }
}

fn parse_role(word: &str) -> Option<Role> {
    match word {
        "factory" => Some(Role::Factory),
        "file" => Some(Role::File),
        "queue" => Some(Role::Queue),
        "status" => Some(Role::Status),
        _ => None,
    }
}

/// The text of a GUID: `{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}`, braces optional. As
/// `IIDFromString` reads it: exactly that shape, hexadecimal digits of either case.
pub fn parse_guid(text: &str) -> Option<Guid> {
    if text.is_empty() {
        return None;
    }
    let braced;
    let text = if text.starts_with('{') {
        text
    } else {
        braced = format!("{{{text}}}");
        &braced
    };
    let b = text.as_bytes();
    if b.len() != 38 || b[0] != b'{' || b[37] != b'}' {
        return None;
    }
    let mut value = 0u128;
    for (i, &c) in b[1..37].iter().enumerate() {
        if matches!(i, 8 | 13 | 18 | 23) {
            if c != b'-' {
                return None;
            }
            continue;
        }
        value = (value << 4) | u128::from(char::from(c).to_digit(16)?);
    }
    Some(Guid(value))
}

/// A number as C's `strtoul(text, &end, 0)` reads it on Windows (32 bit `unsigned long`): decimal,
/// `0x` hexadecimal or leading zero octal, an optional sign, saturating on overflow. `None` when
/// something other than a number follows, or there is none.
pub fn parse_number(text: &str) -> Option<u32> {
    if text.is_empty() || text.len() > 10 {
        return None;
    }
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | 0x0B | 0x0C | b'\r') {
        i += 1;
    }
    let negative = match b.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let mut base = 10u32;
    if b.get(i) == Some(&b'0') {
        if matches!(b.get(i + 1), Some(b'x' | b'X')) && b.get(i + 2).is_some_and(u8::is_ascii_hexdigit) {
            base = 16;
            i += 2;
        } else {
            base = 8;
        }
    }
    let start = i;
    let mut value = 0u64;
    let mut overflow = false;
    while let Some(digit) = b.get(i).and_then(|&c| char::from(c).to_digit(base)) {
        value = value * u64::from(base) + u64::from(digit);
        if value > u64::from(u32::MAX) {
            overflow = true;
            value = u64::from(u32::MAX);
        }
        i += 1;
    }
    if i == start && base != 8 || i != b.len() {
        return None;
    }
    let value = value as u32;
    Some(if overflow {
        u32::MAX
    } else if negative {
        value.wrapping_neg()
    } else {
        value
    })
}

/// The words of a spec line: split on space, tab, CR and LF, a `#` starts a comment, a zero byte
/// ends the line.
fn words(line: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    for &c in line {
        if c == 0 || c == b'#' {
            break;
        }
        if matches!(c, b' ' | b'\t' | b'\r' | b'\n') {
            if !current.is_empty() {
                out.push(String::from_utf8_lossy(&std::mem::take(&mut current)).into_owned());
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        out.push(String::from_utf8_lossy(&current).into_owned());
    }
    out
}

/// The lines of the file as `fgets` with a 1024 byte buffer reads them: a line of 1023 bytes or
/// more is cut into several, each counted as a line.
fn lines(text: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let mut end = rest.len().min(1023);
        if let Some(newline) = rest[..end].iter().position(|&c| c == b'\n') {
            end = newline + 1;
        }
        out.push(&rest[..end]);
        rest = &rest[end..];
    }
    out
}

/// Parses a spec. On error the text is `line N: reason`.
pub fn parse_spec(text: &[u8]) -> Result<Spec, String> {
    let mut spec = Spec { capacity_offset: 8, ifaces: Vec::new() };
    let mut number = 0u32;
    let mut version = false;
    let mut open = false;
    for line in lines(text) {
        number += 1;
        let fail = |why: &str| Err(format!("line {number}: {why}"));
        let w = words(line);
        let Some(k) = w.first().map(String::as_str) else { continue };
        if !version {
            if k != "abi" || w.len() != 2 || w[1] != "1" {
                return fail("the first directive must be 'abi 1'");
            }
            version = true;
            continue;
        }
        if k == "queue_desc_capacity_offset" {
            match (open, w.len(), w.get(1).and_then(|t| parse_number(t))) {
                (false, 2, Some(v)) if v <= 256 => spec.capacity_offset = v,
                _ => return fail("queue_desc_capacity_offset <0..256> outside an interface"),
            }
        } else if k == "interface" {
            let role = w.get(1).and_then(|r| parse_role(r));
            let (false, 3, Some(role)) = (open, w.len(), role) else {
                return fail("interface <factory|file|queue|status> <label>");
            };
            spec.ifaces.push(Iface { role, label: w[2].clone(), status: "unknown".to_owned(), iids: Vec::new(), unknown_iid: false, methods: Vec::new() });
            open = true;
        } else if !open {
            return fail(&format!("'{k}' outside an interface"));
        } else if k == "status" {
            if w.len() != 2 {
                return fail("status <word>");
            }
            if let Some(iface) = spec.ifaces.last_mut() {
                iface.status = w[1].clone();
            }
        } else if k == "iid" {
            if (2..=3).contains(&w.len()) && w[1] == "unknown" {
                // A template: the order is known, the IID is still to be read from the executable.
                if let Some(iface) = spec.ifaces.last_mut() {
                    iface.unknown_iid = true;
                }
                continue;
            }
            let guid = if (2..=3).contains(&w.len()) { parse_guid(&w[1]) } else { None };
            let Some(iid) = guid else {
                return fail("iid <guid|unknown> [name]");
            };
            if spec.ifaces.iter().any(|other| other.accepts(iid)) {
                return fail("IID listed twice");
            }
            if let Some(iface) = spec.ifaces.last_mut() {
                iface.iids.push(iid);
            }
        } else if k == "method" {
            let slot = if (3..=4).contains(&w.len()) { parse_number(&w[1]) } else { None };
            let Some(slot) = slot else {
                return fail("method <slot> <handler|?> [name]");
            };
            if !(3..MAX_SLOTS).contains(&slot) {
                return fail("slot must be 3..63 (0..2 are IUnknown)");
            }
            let Some(iface) = spec.ifaces.last_mut() else { return fail("method outside an interface") };
            if iface.methods.iter().any(|&(s, _)| s == slot) {
                return fail("slot listed twice");
            }
            if w[2] != "?" {
                match handler_role(&w[2]) {
                    None => return fail(&format!("unknown handler '{}'", w[2])),
                    Some(role) if role != iface.role => return fail(&format!("handler '{}' does not belong to this interface's role", w[2])),
                    Some(_) => {}
                }
            }
            iface.methods.push((slot, if w.len() == 4 { format!("{} {}", w[2], w[3]) } else { w[2].clone() }));
        } else if k == "end" {
            if spec.ifaces.last().is_some_and(|i| i.iids.is_empty() && !i.unknown_iid) {
                return fail("interface without an iid (write 'iid unknown' for a template)");
            }
            open = false;
        } else {
            return fail(&format!("unknown directive '{k}'"));
        }
    }
    number += 1;
    let fail = |why: &str| Err(format!("line {number}: {why}"));
    if !version {
        return fail("empty spec");
    }
    if open {
        return fail("missing 'end'");
    }
    if !spec.ifaces.iter().any(|i| i.role == Role::Factory) {
        return fail("no factory interface");
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACTORY: &str = "{7e57a000-0000-4000-8000-000000000001}";
    const GOOD: &str = "abi 1\n\
        # a comment\n\
        queue_desc_capacity_offset 0x10\n\
        interface factory f   # trailing comment\n\
        \x20 status test\n\
        \x20 iid {7E57A000-0000-4000-8000-000000000001} IDStorageFactoryX\n\
        \x20 method 3 factory.create_queue CreateQueue\n\
        \x20 method 6 ? SetDebugFlags\n\
        end\n\
        interface queue q\n\
        \x20 iid 7e57a000-0000-4000-8000-000000000003\n\
        \x20 method 3 queue.submit\n\
        end\n";

    fn error(text: &str) -> String {
        parse_spec(text.as_bytes()).expect_err("the spec is rejected")
    }

    #[test]
    fn a_spec_parses_into_interfaces_and_slots() {
        let spec = parse_spec(GOOD.as_bytes()).expect("parses");
        assert_eq!(spec.capacity_offset, 16);
        assert_eq!(spec.ifaces.len(), 2);
        let f = &spec.ifaces[0];
        assert_eq!((f.role, f.label.as_str(), f.status.as_str()), (Role::Factory, "f", "test"));
        assert_eq!(f.methods, vec![(3, "factory.create_queue CreateQueue".to_owned()), (6, "? SetDebugFlags".to_owned())]);
        assert_eq!(spec.ifaces[1].status, "unknown");
        let iid = parse_guid(FACTORY).expect("guid");
        assert!(f.accepts(iid));
        assert!(spec.find_role(Role::Factory, iid).is_some());
        assert!(spec.find_role(Role::Queue, iid).is_none());
        assert!(spec.find_role(Role::Queue, parse_guid("7e57a000-0000-4000-8000-000000000003").expect("guid")).is_some());
    }

    #[test]
    fn the_default_capacity_offset_is_eight() {
        let spec = parse_spec(b"abi 1\ninterface factory f\niid unknown\nend\n").expect("parses");
        assert_eq!(spec.capacity_offset, 8);
        assert!(spec.ifaces[0].unknown_iid && spec.ifaces[0].iids.is_empty());
    }

    #[test]
    fn guids_are_read_as_iidfromstring_reads_them() {
        assert_eq!(parse_guid(FACTORY), Some(Guid(0x7e57a000_0000_4000_8000_000000000001)));
        assert_eq!(parse_guid("7E57A000-0000-4000-8000-000000000001"), parse_guid(FACTORY));
        for bad in ["", "{}", "{7e57a000-0000-4000-8000-00000000000}", "{7e57a000_0000-4000-8000-000000000001}", "{7e57a000-0000-4000-8000-00000000000g}", "7e57a000-0000-4000-8000-000000000001}"] {
            assert_eq!(parse_guid(bad), None, "{bad}");
        }
    }

    #[test]
    fn numbers_are_read_as_strtoul_reads_them() {
        assert_eq!(parse_number("12"), Some(12));
        assert_eq!(parse_number("0x1F"), Some(31));
        assert_eq!(parse_number("010"), Some(8));
        assert_eq!(parse_number("0"), Some(0));
        assert_eq!(parse_number("+7"), Some(7));
        assert_eq!(parse_number("-1"), Some(u32::MAX));
        assert_eq!(parse_number("9999999999"), Some(u32::MAX));
        for bad in ["", "x", "1x", "0x", "08", "12345678901", "1 2"] {
            assert_eq!(parse_number(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_first_directive_must_be_the_version() {
        assert_eq!(error("interface factory f\n"), "line 1: the first directive must be 'abi 1'");
        assert_eq!(error("abi 2\n"), "line 1: the first directive must be 'abi 1'");
        assert_eq!(error(""), "line 1: empty spec");
        assert_eq!(error("# only a comment\n\n"), "line 3: empty spec");
    }

    #[test]
    fn a_handler_of_the_wrong_role_rejects_the_whole_spec() {
        let text = "abi 1\ninterface factory f\n iid {7e57a000-0000-4000-8000-000000000001}\n method 3 queue.submit\nend\n";
        assert_eq!(error(text), "line 4: handler 'queue.submit' does not belong to this interface's role");
        assert_eq!(error("abi 1\ninterface factory f\n iid unknown\n method 3 nothing.here\n"), "line 4: unknown handler 'nothing.here'");
    }

    #[test]
    fn slots_are_checked() {
        let head = "abi 1\ninterface factory f\n iid unknown\n";
        assert_eq!(error(&format!("{head} method 2 ?\n")), "line 4: slot must be 3..63 (0..2 are IUnknown)");
        assert_eq!(error(&format!("{head} method 64 ?\n")), "line 4: slot must be 3..63 (0..2 are IUnknown)");
        assert_eq!(error(&format!("{head} method 3 ?\n method 3 ?\n")), "line 5: slot listed twice");
        assert_eq!(error(&format!("{head} method three ?\n")), "line 4: method <slot> <handler|?> [name]");
        assert!(parse_spec(format!("{head} method 63 ?\nend\n").as_bytes()).is_ok());
    }

    #[test]
    fn structure_errors_name_their_line() {
        assert_eq!(error("abi 1\nmethod 3 ?\n"), "line 2: 'method' outside an interface");
        assert_eq!(error("abi 1\ninterface factory f\nend\n"), "line 3: interface without an iid (write 'iid unknown' for a template)");
        assert_eq!(error("abi 1\ninterface factory f\n iid unknown\n"), "line 4: missing 'end'");
        assert_eq!(error("abi 1\ninterface queue q\n iid unknown\nend\n"), "line 5: no factory interface");
        assert_eq!(error("abi 1\ninterface thing t\n"), "line 2: interface <factory|file|queue|status> <label>");
        assert_eq!(error("abi 1\ninterface factory f\ninterface queue q\n"), "line 3: interface <factory|file|queue|status> <label>");
        assert_eq!(error("abi 1\ninterface factory f\n frobnicate\n"), "line 3: unknown directive 'frobnicate'");
        assert_eq!(error("abi 1\nqueue_desc_capacity_offset 257\n"), "line 2: queue_desc_capacity_offset <0..256> outside an interface");
        assert_eq!(error("abi 1\ninterface factory f\n status\n"), "line 3: status <word>");
    }

    #[test]
    fn an_iid_may_be_listed_once_in_the_whole_spec() {
        let text = format!("abi 1\ninterface factory f\n iid {FACTORY}\nend\ninterface queue q\n iid {FACTORY}\nend\n");
        assert_eq!(error(&text), "line 6: IID listed twice");
        let twice = format!("abi 1\ninterface factory f\n iid {FACTORY}\n iid {FACTORY}\nend\n");
        assert_eq!(error(&twice), "line 4: IID listed twice");
        assert_eq!(error("abi 1\ninterface factory f\n iid nonsense\n"), "line 3: iid <guid|unknown> [name]");
    }

    #[test]
    fn a_long_line_counts_as_several_like_fgets_reads_it() {
        let mut text = String::from("abi 1\n# ");
        text.push_str(&"x".repeat(2100));
        text.push_str("\nbogus\n");
        // The comment line is read as 1023 + 1023 + 57 bytes: only the first part is a comment, the
        // second one is read as a directive, as the C++ read it.
        let message = error(&text);
        assert!(message.starts_with("line 3: 'xxxxxxxx") && message.ends_with("' outside an interface"), "{message}");
    }

    #[test]
    fn crlf_and_tabs_separate_words() {
        let spec = parse_spec(b"abi 1\r\ninterface\tfactory\tf\r\n iid unknown\r\nend\r\n").expect("parses");
        assert_eq!(spec.ifaces[0].label, "f");
    }

    #[test]
    fn the_shipped_specs_parse() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("abi");
        for name in ["console_hypothesis.abi", "pc_hypothesis.abi", "other_title.abi", "retail_title.abi"] {
            let text = std::fs::read(dir.join(name)).expect("the spec is in the crate");
            let spec = parse_spec(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(spec.ifaces.iter().any(|i| i.role == Role::Factory), "{name}");
        }
    }
}
