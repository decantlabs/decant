//! Values printed and written the way the Python research scripts print and write them, so the
//! outputs of both compare byte for byte: `repr`, `collections.Counter`, `json.dumps` with its
//! default `ensure_ascii`, and the line end of Python's text mode.

use std::collections::HashMap;
use std::hash::Hash;

/// The line end Python's text mode writes: CRLF on Windows, LF elsewhere.
pub const NL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

/// A Python value, enough for what the scripts print.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Py {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    /// A dictionary in insertion order.
    Dict(Vec<(Py, Py)>),
    /// Text already in its final form, such as a nested `Counter` or a float.
    Raw(String),
}

impl From<i64> for Py {
    fn from(v: i64) -> Py {
        Py::Int(v)
    }
}

impl From<u32> for Py {
    fn from(v: u32) -> Py {
        Py::Int(i64::from(v))
    }
}

impl From<usize> for Py {
    fn from(v: usize) -> Py {
        Py::Int(v as i64)
    }
}

impl From<&str> for Py {
    fn from(v: &str) -> Py {
        Py::Str(v.to_owned())
    }
}

impl From<String> for Py {
    fn from(v: String) -> Py {
        Py::Str(v)
    }
}

impl<T: Into<Py>> From<Vec<T>> for Py {
    fn from(v: Vec<T>) -> Py {
        Py::List(v.into_iter().map(Into::into).collect())
    }
}

/// A Python tuple of the given values.
#[macro_export]
macro_rules! tuple {
    ($($v:expr),* $(,)?) => { $crate::py::Py::Tuple(vec![$($crate::py::Py::from($v)),*]) };
}

/// `repr(text)`: single quotes unless the text holds one and no double quote.
pub fn str_repr(text: &str) -> String {
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
            c if (c as u32) < 0x20 || (0x7F..0xA1).contains(&(c as u32)) || c == '\u{AD}' => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

impl Py {
    /// `repr(value)`.
    pub fn repr(&self) -> String {
        let join = |items: &[Py]| items.iter().map(Py::repr).collect::<Vec<_>>().join(", ");
        match self {
            Py::None => "None".to_owned(),
            Py::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
            Py::Int(v) => v.to_string(),
            Py::Str(s) => str_repr(s),
            Py::List(items) => format!("[{}]", join(items)),
            Py::Tuple(items) if items.len() == 1 => format!("({},)", items[0].repr()),
            Py::Tuple(items) => format!("({})", join(items)),
            Py::Dict(pairs) => {
                format!("{{{}}}", pairs.iter().map(|(k, v)| format!("{}: {}", k.repr(), v.repr())).collect::<Vec<_>>().join(", "))
            }
            Py::Raw(text) => text.clone(),
        }
    }

    /// `json.dumps(value)` with `separators=(",", ":")` when `compact`, else the default `", "`
    /// and `": "`. Tuples become lists, as in Python.
    pub fn json(&self, compact: bool) -> String {
        let mut out = String::new();
        self.write_json(&mut out, compact);
        out
    }

    fn write_json(&self, out: &mut String, compact: bool) {
        let (item, key) = if compact { (",", ":") } else { (", ", ": ") };
        match self {
            Py::None => out.push_str("null"),
            Py::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Py::Int(v) => out.push_str(&v.to_string()),
            Py::Str(s) => json_str(out, s),
            Py::List(items) | Py::Tuple(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    v.write_json(out, compact);
                }
                out.push(']');
            }
            Py::Dict(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item);
                    }
                    match k {
                        Py::Str(s) => json_str(out, s),
                        other => json_str(out, &other.repr()),
                    }
                    out.push_str(key);
                    v.write_json(out, compact);
                }
                out.push('}');
            }
            Py::Raw(text) => out.push_str(text),
        }
    }

    /// The value `json.loads` gives for a JSON value.
    pub fn from_json(value: &serde_json::Value) -> Py {
        use serde_json::Value;
        match value {
            Value::Null => Py::None,
            Value::Bool(b) => Py::Bool(*b),
            Value::Number(n) => n.as_i64().map(Py::Int).unwrap_or_else(|| Py::Raw(n.to_string())),
            Value::String(s) => Py::Str(s.clone()),
            Value::Array(items) => Py::List(items.iter().map(Py::from_json).collect()),
            Value::Object(map) => Py::Dict(map.iter().map(|(k, v)| (Py::Str(k.clone()), Py::from_json(v))).collect()),
        }
    }
}

/// A JSON string with every character outside printable ASCII escaped, as `ensure_ascii` does.
fn json_str(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// `collections.Counter`: counts in key insertion order. Adding zero still inserts the key, as
/// `counter[key] += 0` does.
#[derive(Clone, Debug)]
pub struct Counter<K> {
    entries: Vec<(K, i64)>,
    index: HashMap<K, usize>,
}

impl<K> Default for Counter<K> {
    fn default() -> Counter<K> {
        Counter { entries: Vec::new(), index: HashMap::new() }
    }
}

impl<K: Clone + Eq + Hash> Counter<K> {
    pub fn add(&mut self, key: K, n: i64) {
        match self.index.get(&key) {
            Some(&i) => self.entries[i].1 += n,
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, n));
            }
        }
    }

    pub fn get(&self, key: &K) -> i64 {
        self.index.get(key).map_or(0, |&i| self.entries[i].1)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries in insertion order.
    pub fn items(&self) -> &[(K, i64)] {
        &self.entries
    }

    /// `most_common(n)`: highest counts first, equal counts in insertion order.
    pub fn most_common(&self, n: Option<usize>) -> Vec<(K, i64)> {
        let mut sorted = self.entries.clone();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.1));
        sorted.truncate(n.unwrap_or(usize::MAX));
        sorted
    }
}

impl<K: Clone + Eq + Hash + Into<Py>> Counter<K> {
    /// `repr(counter)`: `Counter({key: count, ...})` in `most_common` order, `Counter()` when empty.
    pub fn repr(&self) -> String {
        if self.is_empty() {
            return "Counter()".to_owned();
        }
        let body: Vec<String> = self.most_common(None).into_iter().map(|(k, n)| format!("{}: {n}", k.into().repr())).collect();
        format!("Counter({{{}}})", body.join(", "))
    }

    /// `counter.most_common(n)` as a Python list of `(key, count)` tuples.
    pub fn most_common_py(&self, n: usize) -> Py {
        Py::List(self.most_common(Some(n)).into_iter().map(|(k, c)| Py::Tuple(vec![k.into(), Py::Int(c)])).collect())
    }
}

/// The counts of `stat` sorted by key, one line each, as `for k in sorted(stat): print(f"{stat[k]:w}  {k}")`.
pub fn sorted_lines(stat: &Counter<String>, width: usize) -> String {
    let mut items: Vec<&(String, i64)> = stat.items().iter().collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    items.iter().map(|(k, v)| format!("{v:width$}  {k}{NL}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python() {
        assert_eq!(tuple!("pixel", 32u32, 64u32).repr(), "('pixel', 32, 64)");
        assert_eq!(tuple!(tuple!("a", 1i64)).repr(), "(('a', 1),)");
        assert_eq!(Py::Tuple(vec![]).repr(), "()");
        assert_eq!(str_repr("it's"), "\"it's\"");
        assert_eq!(str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(str_repr("\u{1}\u{e9}"), "'\\x01\u{e9}'");
        assert_eq!(Py::Dict(vec![(Py::Int(0), Py::None)]).repr(), "{0: None}");
    }

    #[test]
    fn json_matches_python() {
        let row = Py::Dict(vec![("name".into(), "\u{e9}\"\u{7f}\u{1F600}".into()), ("l".into(), vec![1i64, 2].into())]);
        assert_eq!(row.json(true), r#"{"name":"\u00e9\"\u007f\ud83d\ude00","l":[1,2]}"#);
        assert_eq!(row.json(false), r#"{"name": "\u00e9\"\u007f\ud83d\ude00", "l": [1, 2]}"#);
    }

    #[test]
    fn counter_orders_like_python() {
        let mut c: Counter<String> = Counter::default();
        assert_eq!(c.repr(), "Counter()");
        for (k, n) in [("b", 1), ("a", 2), ("c", 1), ("z", 0)] {
            c.add(k.to_owned(), n);
        }
        assert_eq!(c.repr(), "Counter({'a': 2, 'b': 1, 'c': 1, 'z': 0})");
        assert_eq!(c.most_common_py(2).repr(), "[('a', 2), ('b', 1)]");
        assert_eq!(sorted_lines(&c, 3), format!("  2  a{NL}  1  b{NL}  1  c{NL}  0  z{NL}"));
    }
}
