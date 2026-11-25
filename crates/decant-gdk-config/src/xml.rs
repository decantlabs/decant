//! The small XML tokenizer of the former `gdk_config_reader.cpp`. It respects element nesting, so an
//! attribute name is matched only in the element it belongs to.

#[derive(Debug, Default)]
pub struct Node {
    pub name: Vec<u8>,
    pub attrs: Vec<(Vec<u8>, Vec<u8>)>,
    pub children: Vec<Node>,
    /// Text of the element, its parts between children joined, trimmed.
    pub text: Vec<u8>,
}

impl Node {
    pub fn attr(&self, key: &[u8]) -> Option<&[u8]> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_slice())
    }

    pub fn child(&self, name: &[u8]) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }
}

/// The root element of `src`.
pub fn parse(src: &[u8]) -> Result<Node, String> {
    let mut p = Parser { s: src, pos: 0 };
    let mut root = Node::default(); // synthetic document node
    p.skip_prolog();
    while p.pos < src.len() {
        p.skip_ws_and_comments();
        if p.pos >= src.len() {
            break;
        }
        if src[p.pos] != b'<' {
            return Err("unexpected text at top level".to_owned());
        }
        p.element(&mut root)?;
    }
    match root.children.len() {
        1 => Ok(root.children.remove(0)),
        _ => Err("expected exactly one root element".to_owned()),
    }
}

/// Replaces the five predefined entities. Any other `&...;` stays as written.
fn unescape(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == b'&'
            && let Some(semi) = s[i..].iter().position(|&c| c == b';').map(|n| i + n)
        {
            match &s[i + 1..semi] {
                b"amp" => out.push(b'&'),
                b"lt" => out.push(b'<'),
                b"gt" => out.push(b'>'),
                b"quot" => out.push(b'"'),
                b"apos" => out.push(b'\''),
                _ => out.extend_from_slice(&s[i..=semi]),
            }
            i = semi + 1;
            continue;
        }
        out.push(s[i]);
        i += 1;
    }
    out
}

/// The bytes C `isspace` accepts in the C locale.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn is_name_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b':')
}

fn find(s: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    s.get(from..)?.windows(needle.len()).position(|w| w == needle).map(|n| from + n)
}

fn trim(text: &[u8]) -> Vec<u8> {
    let keep = |c: &u8| !matches!(c, b' ' | b'\t' | b'\r' | b'\n');
    match (text.iter().position(keep), text.iter().rposition(keep)) {
        (Some(a), Some(b)) => text[a..=b].to_vec(),
        _ => Vec::new(),
    }
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn at(&self, lit: &[u8]) -> bool {
        self.s.get(self.pos..).is_some_and(|rest| rest.starts_with(lit))
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(is_space) {
            self.pos += 1;
        }
    }

    /// Moves past the end of the next `end`, or to the end of the input.
    fn skip_past(&mut self, end: &[u8]) {
        self.pos = find(self.s, self.pos, end).map_or(self.s.len(), |e| e + end.len());
    }

    /// Skips a BOM, `<?xml ... ?>` declarations and comments.
    fn skip_prolog(&mut self) {
        if self.s.starts_with(b"\xEF\xBB\xBF") {
            self.pos = 3;
        }
        loop {
            self.skip_ws();
            if self.at(b"<?") {
                self.skip_past(b"?>");
            } else if self.at(b"<!--") {
                self.skip_past(b"-->");
            } else {
                break;
            }
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            self.skip_ws();
            if !self.at(b"<!--") {
                break;
            }
            self.skip_past(b"-->");
        }
    }

    fn name(&mut self) -> Vec<u8> {
        let start = self.pos;
        while self.peek().is_some_and(is_name_char) {
            self.pos += 1;
        }
        self.s[start..self.pos].to_vec()
    }

    /// Parses one element starting at `<` and appends it to `parent`.
    fn element(&mut self, parent: &mut Node) -> Result<(), String> {
        let s = self.s;
        self.pos += 1;
        if self.peek() == Some(b'/') {
            return Err("unexpected closing tag".to_owned());
        }
        let mut node = Node { name: self.name(), ..Node::default() };
        let tag = String::from_utf8_lossy(&node.name).into_owned();
        if node.name.is_empty() {
            return Err("empty element name".to_owned());
        }
        loop {
            self.skip_ws();
            match self.peek() {
                None => return Err(format!("unterminated tag <{tag}>")),
                Some(b'/' | b'>') => break,
                Some(_) => {}
            }
            let key = self.name();
            let key_text = String::from_utf8_lossy(&key).into_owned();
            if key.is_empty() {
                return Err(format!("bad attribute in <{tag}>"));
            }
            self.skip_ws();
            if self.peek() != Some(b'=') {
                return Err(format!("expected '=' after {key_text}"));
            }
            self.pos += 1;
            self.skip_ws();
            let Some(quote) = self.peek().filter(|&c| c == b'"' || c == b'\'') else {
                return Err(format!("expected quoted value for {key_text}"));
            };
            self.pos += 1;
            let Some(end) = s[self.pos..].iter().position(|&c| c == quote).map(|n| self.pos + n) else {
                return Err(format!("unterminated attribute {key_text}"));
            };
            node.attrs.push((key, unescape(&s[self.pos..end])));
            self.pos = end + 1;
        }
        if self.peek() == Some(b'/') {
            self.pos += 1;
            if self.peek() != Some(b'>') {
                return Err("bad self-closing tag".to_owned());
            }
            self.pos += 1;
            parent.children.push(node);
            return Ok(());
        }
        self.pos += 1; // '>'
        loop {
            let Some(lt) = s[self.pos..].iter().position(|&c| c == b'<').map(|n| self.pos + n) else {
                return Err(format!("unterminated <{tag}>"));
            };
            node.text.extend(unescape(&s[self.pos..lt]));
            self.pos = lt;
            if self.at(b"<!--") {
                let Some(e) = find(s, self.pos, b"-->") else { return Err("unterminated comment".to_owned()) };
                self.pos = e + 3;
                continue;
            }
            if self.at(b"</") {
                self.pos += 2;
                let closing = self.name();
                self.skip_ws();
                if self.peek() != Some(b'>') {
                    return Err("bad closing tag".to_owned());
                }
                self.pos += 1;
                if closing != node.name {
                    return Err(format!("mismatched </{}> for <{tag}>", String::from_utf8_lossy(&closing)));
                }
                break;
            }
            self.element(&mut node)?;
        }
        node.text = trim(&node.text);
        parent.children.push(node);
        Ok(())
    }
}
