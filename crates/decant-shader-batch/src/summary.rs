//! `decant_shader_summary.py`: sums up one or more `results.jsonl`, valid modules and failure causes
//! grouped together.

use regex::Regex;
use serde_json::Value;
use decant_shader_core::py;

/// Running totals over the folders given.
pub struct Summary {
    ok: usize,
    n: usize,
    /// Causes in order of first appearance, with their counts.
    why: Vec<(String, usize)>,
    numbers: Regex,
}

impl Default for Summary {
    fn default() -> Self {
        Summary { ok: 0, n: 0, why: Vec::new(), numbers: Regex::new(r"pc=0x[0-9a-f]+|hash=0x[0-9a-f]+|\b\d+\b|0x[0-9a-f]+").expect("valid pattern") }
    }
}

/// `r.get(key, default)` for a list field.
fn list<'a>(row: &'a Value, key: &str) -> Option<&'a [Value]> {
    row.get(key).and_then(Value::as_array).map(Vec::as_slice)
}

/// `r.get(key, [default])[0]`.
fn first<'a>(row: &'a Value, key: &str, default: &'a Value) -> Result<&'a Value, String> {
    match row.get(key) {
        None => Ok(default),
        Some(Value::Array(items)) => items.first().ok_or_else(|| "IndexError: list index out of range".to_owned()),
        Some(other) => Err(format!("unexpected {key}: {}", py::repr_value(other))),
    }
}

/// `(r.get(key, ["", ""]) + [""])[1]`, which must be text.
fn second(row: &Value, key: &str) -> Result<String, String> {
    let value = match list(row, key) {
        None if row.get(key).is_some() => return Err(format!("unexpected {key}")),
        None => return Ok(String::new()),
        Some(items) => items.get(1),
    };
    match value {
        None => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!("TypeError: expected string, got {}", py::repr_value(other))),
    }
}

impl Summary {
    /// Adds the rows of one folder and returns the line printed for it.
    pub fn add(&mut self, folder: &str, rows: &[Value]) -> Result<String, String> {
        let ok_text = Value::String("OK".into());
        let empty = Value::String(String::new());
        let mut ok = 0;
        for r in rows {
            if *first(r, "V", &empty)? == ok_text {
                ok += 1;
            }
        }
        let line = format!("{folder}: {ok}/{} modules valides", rows.len());
        self.ok += ok;
        self.n += rows.len();
        for r in rows {
            if *first(r, "V", &empty)? == ok_text {
                continue;
            }
            let msg = if *first(r, "T", &ok_text)? != ok_text {
                second(r, "T")?
            } else if *first(r, "C", &ok_text)? != ok_text {
                second(r, "C")?
            } else if r.get("V").is_some() {
                format!("spirv-val: {}", second(r, "V")?.chars().take(80).collect::<String>())
            } else {
                format!("arret: {}", r.get("X").map_or("None".into(), py::repr_value))
            };
            let msg = self.numbers.replace_all(&msg, "#");
            let chars: Vec<char> = msg.chars().collect();
            let tail: String = chars[chars.len().saturating_sub(95)..].iter().collect();
            match self.why.iter_mut().find(|(m, _)| *m == tail) {
                Some((_, c)) => *c += 1,
                None => self.why.push((tail, 1)),
            }
        }
        Ok(line)
    }

    /// The closing lines: the total, then the 14 most common causes. Equal counts keep their first
    /// appearance, as `Counter.most_common` does.
    pub fn finish(&self) -> Vec<String> {
        let mut lines = vec![format!("TOTAL {}/{}", self.ok, self.n)];
        let mut causes: Vec<&(String, usize)> = self.why.iter().collect();
        causes.sort_by_key(|c| std::cmp::Reverse(c.1));
        lines.extend(causes.iter().take(14).map(|(m, c)| format!("{c:6}  {m}")));
        lines
    }
}

/// The rows of `<folder>/results.jsonl`.
pub fn rows(folder: &str) -> Result<Vec<Value>, String> {
    decant_shader_core::meta::rows(&std::path::Path::new(folder).join("results.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn causes_are_grouped_without_their_numbers() {
        let rows = vec![
            json!({"pack_id": 0, "B": ["0"], "T": ["OK"], "C": ["OK"], "V": ["OK"]}),
            json!({"pack_id": 1, "T": ["FAIL", "unknown instruction 0x7e at pc=0x1f0"]}),
            json!({"pack_id": 2, "T": ["FAIL", "unknown instruction 0x7f at pc=0x2a8"]}),
            json!({"pack_id": 3, "T": ["OK"], "C": ["FAIL", "binding 12 out of range 3"]}),
            json!({"pack_id": 4, "T": ["OK"], "C": ["OK"], "V": ["FAIL", format!("error: line 7: {}", "x".repeat(90))]}),
            json!({"pack_id": 5, "B": ["5"], "X": ["5", "timeout 45s"]}),
            json!({"pack_id": 6, "B": ["6"]}),
            json!({"pack_id": 7, "T": ["FAIL"]}),
            json!({"pack_id": 8, "C": ["FAIL", "hash=0xdeadbeef x2y 3.5 a_1"]}),
        ];
        let mut summary = Summary::default();
        assert_eq!(summary.add("run1", &rows).unwrap(), "run1: 1/9 modules valides");
        assert_eq!(summary.add("run2", &rows[..1]).unwrap(), "run2: 1/1 modules valides");
        let lines = summary.finish();
        assert_eq!(lines[0], "TOTAL 2/10");
        assert_eq!(lines[1], "     2  unknown instruction # at #");
        assert_eq!(lines[2], "     1  binding # out of range #");
        assert_eq!(lines[3], format!("     1  {}", &format!("spirv-val: error: line #: {}", "x".repeat(80 - 15))[..]));
        assert_eq!(lines[4], "     1  arret: ['#', 'timeout 45s']");
        assert_eq!(lines[5], "     1  arret: None");
        assert_eq!(lines[6], "     1  ");
        assert_eq!(lines[7], "     1  # x2y #.# a_1");
    }

    #[test]
    fn long_causes_keep_their_end_and_only_14_are_listed() {
        let rows: Vec<Value> = (0..20).map(|i| json!({"T": ["FAIL", format!("{}{}", "y".repeat(100), "abcdefghijklmnopqrst".chars().nth(i).unwrap())]})).collect();
        let mut summary = Summary::default();
        summary.add("f", &rows).unwrap();
        let lines = summary.finish();
        assert_eq!(lines.len(), 15);
        assert_eq!(lines[1], format!("     1  {}a", "y".repeat(94)));
    }
}
