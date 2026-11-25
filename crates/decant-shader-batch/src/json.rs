//! Result rows written as `json.dumps(row, ensure_ascii=True)` writes them: `", "` and `": "`
//! separators, every character outside ASCII escaped.

/// One value of a result row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Field {
    Int(i64),
    Str(String),
    List(Vec<String>),
}

/// A result row: keys in insertion order, as a Python dict keeps them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Row(pub Vec<(String, Field)>);

impl Row {
    /// `row[key] = value`: a new key goes last, an existing key keeps its place.
    pub fn set(&mut self, key: &str, value: Field) {
        match self.0.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.0.push((key.to_owned(), value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Field> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn dumps(&self) -> String {
        let fields: Vec<String> = self.0.iter().map(|(k, v)| format!("{}: {}", string(k), value(v))).collect();
        format!("{{{}}}", fields.join(", "))
    }
}

fn value(field: &Field) -> String {
    match field {
        Field::Int(i) => i.to_string(),
        Field::Str(s) => string(s),
        Field::List(items) => format!("[{}]", items.iter().map(|s| string(s)).collect::<Vec<_>>().join(", ")),
    }
}

/// A JSON string with Python's `ensure_ascii` escapes.
pub fn string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            ' '..='~' | '\u{7F}' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_dump_like_python() {
        let mut row = Row::default();
        row.set("pack_id", Field::Int(7));
        row.set("name", Field::Str("PS_é\"x".into()));
        row.set("stage", Field::Int(0));
        row.set("B", Field::List(vec!["0".into(), "a\tb\u{1}\u{1F600}\u{7F}".into()]));
        row.set("name", Field::List(vec![]));
        assert_eq!(
            row.dumps(),
            r#"{"pack_id": 7, "name": [], "stage": 0, "B": ["0", "a\tb\u0001\ud83d\ude00"#.to_owned() + "\u{7F}" + "\"]}"
        );
    }
}
