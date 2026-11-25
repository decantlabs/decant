//! Rows of `out/decant_shader_metadata.jsonl` and `out/decant_shader_outputs.jsonl`, written by
//! `src/shaders/decant_shader_metadata.py` and `src/shaders/decant_shader_outputs.py`: one JSON object per
//! shader, in pack order.
//!
//! A signature element (`inputs`, `outputs`) is a list:
//! `[semantic, index, register, mask, read or written mask, system value, component type, stream, precision]`.

use std::path::Path;

use serde_json::Value;

use crate::py;

/// Positions inside a signature element.
pub const SEMANTIC: usize = 0;
pub const SEMANTIC_INDEX: usize = 1;
pub const REGISTER: usize = 2;
pub const MASK: usize = 3;
pub const USED: usize = 4;
pub const SYSTEM: usize = 5;
pub const COMPONENT: usize = 6;

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse(line: &[u8]) -> Result<Value, String> {
    let text = std::str::from_utf8(line).map_err(|e| format!("UnicodeDecodeError: {e}"))?;
    serde_json::from_str(text).map_err(|e| format!("JSONDecodeError: {e}"))
}

/// Row `index` of a JSON lines file, as `row()` in `decant_shader_locate.py` reads it. Only that line is
/// decoded.
pub fn row(path: &Path, index: i64) -> Result<Value, String> {
    let bytes = read(path)?;
    let line = usize::try_from(index).ok().and_then(|i| py::universal_lines(&bytes).get(i).copied());
    parse(line.ok_or_else(|| format!("IndexError: {index}"))?)
}

/// Every row of a JSON lines file.
pub fn rows(path: &Path) -> Result<Vec<Value>, String> {
    py::universal_lines(&read(path)?).into_iter().map(parse).collect()
}

/// `value[key]` for a JSON object.
pub fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("KeyError: {}", py::repr_str(key)))
}

/// `value[key]` for a list field.
pub fn list<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    field(value, key)?.as_array().map(Vec::as_slice).ok_or_else(|| format!("{key} is not a list"))
}

/// `entry[k]` for a signature element.
pub fn at(entry: &Value, k: usize) -> Result<&Value, String> {
    entry.get(k).ok_or_else(|| "IndexError: list index out of range".to_owned())
}

/// `entry[k]` where an integer is expected.
pub fn int(entry: &Value, k: usize) -> Result<i64, String> {
    let v = at(entry, k)?;
    v.as_i64().ok_or_else(|| format!("element {k} is not an integer: {}", py::repr_value(v)))
}

/// Whether the element is an ordinary attribute, not a system value (`e[5] == 0`).
pub fn ordinary(entry: &Value) -> Result<bool, String> {
    Ok(int(entry, SYSTEM)? == 0)
}

/// The values of a list as `str()` prints them, separated by spaces, as `print(*values)` does.
fn spaced(values: &[Value]) -> String {
    values.iter().map(py::str_value).collect::<Vec<_>>().join(" ")
}

/// One line of a `.inputs` file for the probe, without its line end:
/// `print(*(m["threads"] or [0,0,0]), m["wave"] or 0, *m.get("mesh_limits",[0,0]))`.
pub fn inputs_line(meta: &Value) -> Result<String, String> {
    let threads = field(meta, "threads")?;
    let threads = if py::truthy(threads) {
        spaced(threads.as_array().ok_or("threads is not a list")?)
    } else {
        "0 0 0".to_owned()
    };
    let wave = field(meta, "wave")?;
    let wave = if py::truthy(wave) { py::str_value(wave) } else { "0".to_owned() };
    let limits = match meta.get("mesh_limits") {
        Some(v) => spaced(v.as_array().ok_or("mesh_limits is not a list")?),
        None => "0 0".to_owned(),
    };
    Ok([threads, wave, limits].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" "))
}

/// One line of a `.vinputs` file for the probe, without its line end: per register below 32, the
/// mask and component type of the ordinary attribute in it (`e[3] | e[6] << 4`), or 0.
pub fn vinputs_line(meta: &Value) -> Result<String, String> {
    let mut line = [0i64; 32];
    for e in list(meta, "inputs")? {
        if ordinary(e)? {
            let register = int(e, REGISTER)?;
            if register < 32 {
                // A negative register counts from the end, as a Python list index does.
                let slot = if register < 0 { register + 32 } else { register };
                let slot = usize::try_from(slot).map_err(|_| "IndexError: list assignment index out of range")?;
                line[slot] = int(e, MASK)? | int(e, COMPONENT)? << 4;
            }
        }
    }
    Ok(line.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
}

/// `m["name"] == name`, the check the tools make before using a row for a pack record.
pub fn check_name(meta: &Value, name: &str) -> Result<(), String> {
    if field(meta, "name")?.as_str() == Some(name) {
        Ok(())
    } else {
        Err(format!("AssertionError: metadata row {} is not shader {}", py::repr_value(field(meta, "name")?), py::repr_str(name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn probe_lines_follow_the_python() {
        let compute = json!({"name": "CS", "threads": [8, 4, 1], "wave": 64, "mesh_limits": [0, 0], "inputs": []});
        assert_eq!(inputs_line(&compute).unwrap(), "8 4 1 64 0 0");
        let mesh = json!({"name": "MS", "threads": null, "wave": null, "mesh_limits": [64, 126], "inputs": []});
        assert_eq!(inputs_line(&mesh).unwrap(), "0 0 0 0 64 126");
        let old = json!({"name": "VS", "threads": [], "wave": 32, "inputs": []});
        assert_eq!(inputs_line(&old).unwrap(), "0 0 0 32 0 0");

        let vertex = json!({"inputs": [
            ["POSITION", 0, 0, 7, 7, 0, 3, 0, 0],
            ["TEXCOORD", 0, 2, 3, 3, 0, 2, 0, 0],
            ["SV_VertexID", 0, 1, 1, 1, 6, 1, 0, 0],
            ["FAR", 0, 40, 15, 15, 0, 3, 0, 0]
        ]});
        let line = vinputs_line(&vertex).unwrap();
        let mut expected = vec!["0"; 32];
        expected[0] = "55";
        expected[2] = "35";
        assert_eq!(line, expected.join(" "));
    }

    #[test]
    fn rows_are_read_by_line() {
        let dir = std::env::temp_dir().join(format!("decant-shader-core-meta-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.jsonl");
        std::fs::write(&path, b"{\"id\":0}\r\nnot json\r\n{\"id\":2}\r\n").unwrap();
        assert_eq!(row(&path, 2).unwrap(), json!({"id": 2}));
        assert!(row(&path, 1).is_err());
        assert_eq!(row(&path, 3).unwrap_err(), "IndexError: 3");
        assert_eq!(row(&path, -1).unwrap_err(), "IndexError: -1");
        assert!(rows(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
