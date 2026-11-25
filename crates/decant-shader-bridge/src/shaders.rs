//! Where the bridge finds its inputs, and the shared state `decant_shader_identify.py` builds when it is
//! imported: the shader input, its shaders by position, the pack positions from `shaders.csv` and
//! the rows of `decant_shader_metadata.jsonl`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use decant_extract_shaders::{Shader, find_shaders};

use crate::py::{self, Failure};

/// The folders and commands the tools use. Every value has an environment override.
#[derive(Clone, Debug)]
pub struct Config {
    /// The repository's `src/shaders` folder (`DECANT_SHADERS_DIR`). Its `out` folder holds the
    /// extractor and metadata outputs, `decant_shader/` the probe and the bench.
    pub shaders: PathBuf,
    /// A program to run instead of the `decant_shader_one` built next to this one (`DECANT_SHADER_ONE`).
    pub one: Option<OsString>,
    /// A program to run instead of the `decant_shader_locate` built next to this one (`DECANT_SHADER_LOCATE`).
    pub locate: Option<OsString>,
}

impl Config {
    pub fn from_env() -> Config {
        let shaders = std::env::var_os("DECANT_SHADERS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/shaders"));
        Config { shaders, one: std::env::var_os("DECANT_SHADER_ONE"), locate: std::env::var_os("DECANT_SHADER_LOCATE") }
    }

    /// `src/shaders/decant_shader`, the working folder of the tools.
    pub fn root(&self) -> PathBuf {
        self.shaders.join("decant_shader")
    }

    pub fn out(&self) -> PathBuf {
        self.shaders.join("out")
    }

    /// The program for the workspace tool `name` (`decant_shader_one`, `decant_shader_locate`, ...): its
    /// replacement, or the binary of that name next to the running one.
    pub fn program(name: &str, replacement: &Option<OsString>) -> OsString {
        replacement.clone().unwrap_or_else(|| decant_shader_core::paths::sibling(name).into_os_string())
    }

    /// The command for the workspace tool `name`, run from `decant_shader/` as the tools expect.
    pub fn tool(&self, name: &str, replacement: &Option<OsString>) -> Command {
        let mut command = Command::new(Config::program(name, replacement));
        command.current_dir(self.root());
        command
    }
}

/// `DECANT_SHADERS_RPF` (a shader file or a folder of them), read when `decant_shader_identify.py` is imported.
pub fn archive_path() -> Result<PathBuf, Failure> {
    std::env::var_os("DECANT_SHADERS_RPF").map(PathBuf::from).ok_or_else(|| Failure::key(py::repr_str("DECANT_SHADERS_RPF")))
}

/// The shader containers of the input file or folder, by their position in it.
pub struct Archive {
    by_position: HashMap<u64, Shader>,
}

impl Archive {
    pub fn open(path: &Path) -> Result<Archive, Failure> {
        let shaders = find_shaders(path).map_err(|e| Failure::error("OSError", &e))?;
        let by_position = shaders.into_iter().map(|s| (s.position(), s)).collect();
        Ok(Archive { by_position })
    }

    /// The shader that pack number `pack_id` came from, found by its position.
    pub fn item_of(&self, positions: &[u64], pack_id: i64) -> Result<&Shader, Failure> {
        let position = py::index(positions, pack_id)?;
        self.by_position
            .get(position)
            .ok_or_else(|| Failure::Exit(format!("shader {pack_id} du pack introuvable dans l'entree : refaire l'extraction")))
    }

    /// The DXBC container of a shader.
    pub fn container(&self, item: &Shader) -> Result<Vec<u8>, Failure> {
        Ok(item.container().to_vec())
    }
}

/// The archive position of every pack entry, in pack order: the `position` column of
/// `out/shaders.csv`. The pack keeps only shaders whose code was extracted, so a pack number may
/// differ from the rank in the archive.
pub fn pack_positions(config: &Config) -> Result<Vec<u64>, Failure> {
    let path = config.out().join("shaders.csv");
    let bytes = std::fs::read(&path).map_err(|e| Failure::io(&e, &path))?;
    let text = String::from_utf8_lossy(&bytes);
    let mut positions = Vec::new();
    for row in csv_rows(&text, ';').into_iter().skip(1) {
        let field = row.get(2).ok_or_else(|| Failure::error("IndexError", "list index out of range"))?;
        positions.push(py::int16(field)? as u64);
    }
    Ok(positions)
}

/// Rows as Python's `csv.reader` reads them with the given delimiter: `"` quotes a field, `""`
/// inside quotes is one quote, CR LF or LF ends a row outside quotes.
pub fn csv_rows(text: &str, delimiter: char) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let (mut row, mut field) = (Vec::new(), String::new());
    let (mut quoted, mut started) = (false, false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => {
                quoted = true;
                started = true;
            }
            c if c == delimiter => {
                row.push(std::mem::take(&mut field));
                started = true;
            }
            '\r' | '\n' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                if started || !field.is_empty() || !row.is_empty() {
                    row.push(std::mem::take(&mut field));
                }
                rows.push(std::mem::take(&mut row));
                started = false;
            }
            c => {
                field.push(c);
                started = true;
            }
        }
    }
    if started || !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// The rows of `out/decant_shader_metadata.jsonl`, one per pack entry.
pub fn metadata(config: &Config) -> Result<Vec<serde_json::Value>, Failure> {
    let path = config.out().join("decant_shader_metadata.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| Failure::io(&e, &path))?;
    json_lines(&text)
}

/// `[json.loads(l) for l in file]`.
pub fn json_lines(text: &str) -> Result<Vec<serde_json::Value>, Failure> {
    text.split_inclusive('\n')
        .map(|line| serde_json::from_str(line).map_err(|e| Failure::error("json.decoder.JSONDecodeError", &e.to_string())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_reads_rows_the_extractor_writes() {
        let mut csv = Vec::new();
        for row in [
            decant_extract_shaders::output::CSV_HEADER.map(str::to_owned).to_vec(),
            vec!["a;b".into(), "pixel".into(), "0x10".into(), "1".into()],
            vec!["say \"hi\"".into(), "vertex".into(), "0x2a".into()],
        ] {
            decant_extract_shaders::output::csv_row(&mut csv, &row).unwrap();
        }
        let rows = csv_rows(std::str::from_utf8(&csv).unwrap(), ';');
        assert_eq!(rows[0][2], "position");
        assert_eq!(rows[1], ["a;b", "pixel", "0x10", "1"]);
        assert_eq!(rows[2], ["say \"hi\"", "vertex", "0x2a"]);
        assert_eq!(csv_rows("\r\nx\n", ';'), [Vec::<String>::new(), vec!["x".to_owned()]]);
    }

    #[test]
    fn json_lines_read_each_line() {
        let rows = json_lines("{\"stage\": \"pixel\", \"heap_flags\": 1}\n{\"stage\": \"vertex\", \"heap_flags\": 0}\n").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["stage"], "vertex");
        assert!(json_lines("{\n").is_err());
    }
}
