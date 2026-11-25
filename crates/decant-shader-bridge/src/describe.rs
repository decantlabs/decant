//! Translates shaders and writes, next to each SPIR-V module, a JSON description of what the
//! module expects, as `src/shaders/decant_shader/tools/decant_shader_describe.py` does.
//!
//! For each shader: `<output>/<number>.spv` and `<output>/<number>.json`. The module has its inputs
//! and outputs at the register numbers of the Direct3D signature (as `decant_shader_locate` puts them).
//! Its resources keep the translator's layout, and the JSON says for each one which Direct3D
//! register or heap index it stands for and where it sits in the module (set, binding, array rank).
//!
//! The outputs come from title files: keep them locally, never in the repository.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{Map, Value, json};
use sha1::{Digest, Sha1};

use decant_shader_core::meta;

use crate::identify::Resource;
use crate::py::{self, Failure};
use crate::session::Session;
use crate::shaders::{self, Config};

pub const FORMAT_VERSION: i64 = 1;
/// `D3D_NAME_TARGET`: a colour output, whose location is the target number.
pub const SV_TARGET: i64 = 64;

/// The capability names the description spells out. Others stay numbers.
fn capability(c: u32) -> Value {
    let name = match c {
        1 => "Shader",
        9 => "Float16",
        11 => "Int64",
        22 => "Int16",
        32 => "ClipDistance",
        39 => "Int8",
        55 => "StorageImageReadWithoutFormat",
        56 => "StorageImageWriteWithoutFormat",
        61 => "GroupNonUniform",
        62 => "GroupNonUniformVote",
        63 => "GroupNonUniformArithmetic",
        64 => "GroupNonUniformBallot",
        65 => "GroupNonUniformShuffle",
        4448 => "StorageBuffer8BitAccess",
        4466 => "SignedZeroInfNanPreserve",
        5283 => "MeshShadingEXT",
        5302 => "RuntimeDescriptorArray",
        5345 => "VulkanMemoryModel",
        5347 => "PhysicalStorageBufferAddresses",
        _ => return json!(c),
    };
    json!(name)
}

/// `ORIGIN`: the identify origins in English.
fn origin(how: &str) -> &str {
    match how {
        "tas" => "heap",
        "tas-samplers" => "sampler-heap",
        "immediat" => "immediate",
        "autre" => "other",
        other => other,
    }
}

/// What `module_facts()` reads: the version word, the capabilities in order, and for each decorated
/// binding number the (set, binding) of the variable that carries it.
#[derive(Debug, PartialEq, Eq)]
pub struct Facts {
    pub version: u32,
    pub capabilities: Vec<u32>,
    pub by_binding: HashMap<u32, (u32, u32)>,
}

fn word(words: &[u32], i: usize) -> Result<u32, Failure> {
    words.get(i).copied().ok_or_else(|| Failure::error("IndexError", "tuple index out of range"))
}

pub fn module_facts(spv: &[u8]) -> Result<Facts, Failure> {
    let words: Vec<u32> = spv.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    if !spv.len().is_multiple_of(4) {
        return Err(Failure::error("struct.error", &format!("unpack requires a buffer of {} bytes", words.len() * 4)));
    }
    let mut capabilities = Vec::new();
    let mut sets: HashMap<u32, u32> = HashMap::new();
    // Variable to binding, in the order a Python dict keeps them: a new variable goes last.
    let mut bindings: Vec<(u32, u32)> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (op, count) = (words[i] & 0xFFFF, (words[i] >> 16) as usize);
        if op == 17 {
            capabilities.push(word(&words, i + 1)?);
        } else if op == 71 && word(&words, i + 2)? == 34 {
            sets.insert(word(&words, i + 1)?, word(&words, i + 3)?);
        } else if op == 71 && word(&words, i + 2)? == 33 {
            let (variable, binding) = (word(&words, i + 1)?, word(&words, i + 3)?);
            match bindings.iter_mut().find(|(v, _)| *v == variable) {
                Some((_, b)) => *b = binding,
                None => bindings.push((variable, binding)),
            }
        }
        // The Python loops forever on an instruction of zero words.
        if count == 0 {
            return Err(Failure::runtime(&format!("instruction of zero words at word {i}")));
        }
        i += count;
    }
    let mut by_binding = HashMap::new();
    for (variable, binding) in bindings {
        by_binding.insert(binding, (sets.get(&variable).copied().unwrap_or(0), binding));
    }
    Ok(Facts { version: word(&words, 1)?, capabilities, by_binding })
}

/// The nine fields of a signature element.
fn fields(row: &Value) -> Result<&[Value], Failure> {
    match row.as_array() {
        Some(items) if items.len() == 9 => Ok(items),
        Some(items) if items.len() > 9 => Err(Failure::value("too many values to unpack (expected 9)".into())),
        Some(items) => Err(Failure::value(format!("not enough values to unpack (expected 9, got {})", items.len()))),
        None => Err(Failure::error("TypeError", "cannot unpack non-iterable object")),
    }
}

/// `inputs_of()`: `used_mask` is the components read. An input never read is not declared in the
/// module, so its location is null.
pub fn inputs_of(rows: &[Value]) -> Result<Vec<Value>, Failure> {
    rows.iter()
        .map(|row| {
            let f = fields(row)?;
            let location = if f[5] == 0 && f[4] != 0 { f[2].clone() } else { Value::Null };
            Ok(element(f, "used_mask", location))
        })
        .collect()
}

/// `outputs_of()`: `never_written_mask` is the components the shader never writes.
pub fn outputs_of(rows: &[Value]) -> Result<Vec<Value>, Failure> {
    rows.iter()
        .map(|row| {
            let f = fields(row)?;
            let location = if f[5] == 0 || f[5] == SV_TARGET { f[2].clone() } else { Value::Null };
            Ok(element(f, "never_written_mask", location))
        })
        .collect()
}

fn element(f: &[Value], mask_name: &str, location: Value) -> Value {
    let mut m = Map::new();
    m.insert("semantic".into(), f[0].clone());
    m.insert("index".into(), f[1].clone());
    m.insert("register".into(), f[2].clone());
    m.insert("mask".into(), f[3].clone());
    m.insert(mask_name.into(), f[4].clone());
    m.insert("system_value".into(), f[5].clone());
    m.insert("component_type".into(), f[6].clone());
    m.insert("location".into(), location);
    Value::Object(m)
}

/// One resource of the description.
fn resource(r: &Resource, facts: &Facts, base: i64) -> Value {
    let place = r.binding.and_then(|b| u32::try_from(base + b).ok()).and_then(|b| facts.by_binding.get(&b));
    let mut m = Map::new();
    m.insert("kind".into(), json!(r.kind));
    m.insert("index".into(), json!(r.index));
    m.insert("d3d_register".into(), json!(r.register));
    m.insert("heap_index".into(), json!(r.heap));
    m.insert("origin".into(), json!(origin(r.origin)));
    m.insert("descriptor_set".into(), json!(place.map(|p| p.0)));
    m.insert("binding".into(), json!(place.map(|p| p.1)));
    m.insert("array_rank".into(), json!(r.rank.filter(|&k| k < 0xFFFF_FFFF)));
    m.insert("descriptor_table_sgpr".into(), json!(r.table));
    m.insert("table_offset_bytes".into(), json!(r.offset));
    Value::Object(m)
}

/// The metadata rows and the `outputs` of every row of `decant_shader_outputs.jsonl`, loaded once.
pub struct Rows {
    pub meta: Vec<Value>,
    pub outputs: Vec<Value>,
}

impl Rows {
    pub fn load(config: &Config) -> Result<Rows, Failure> {
        let meta = shaders::metadata(config)?;
        let path = config.out().join("decant_shader_outputs.jsonl");
        let text = std::fs::read_to_string(&path).map_err(|e| Failure::io(&e, &path))?;
        let outputs = shaders::json_lines(&text)?
            .into_iter()
            .map(|mut row| row.get_mut("outputs").map(Value::take).ok_or_else(|| Failure::key(py::repr_str("outputs"))))
            .collect::<Result<_, _>>()?;
        Ok(Rows { meta, outputs })
    }
}

/// What `one()` returns for one shader: what the summary counts, and for the shader gallery its
/// name, the description of a translated shader, or the `repr()` of the error that stopped it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub translated: bool,
    pub resources: usize,
    pub resolved: usize,
    /// `d.get("name")`: null when an error stopped the shader.
    pub name: Value,
    pub description: Option<Value>,
    /// `repr(error)[:300]`.
    pub error: Option<String>,
}

fn field<'a>(row: &'a Value, key: &str) -> Result<&'a Value, Failure> {
    row.get(key).ok_or_else(|| Failure::key(py::repr_str(key)))
}

fn core(error: String) -> Failure {
    match error.split_once(": ") {
        Some((kind, message)) if !kind.contains(' ') => Failure::error(kind, message),
        _ => Failure::Exit(error),
    }
}

/// The text a JSON value is written as: `json.dumps(value, indent=1, ensure_ascii=False)`.
pub fn pretty(value: &Value) -> String {
    use serde::Serialize;
    let mut out = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, serde_json::ser::PrettyFormatter::with_indent(b" "));
    value.serialize(&mut serializer).expect("writing to memory");
    String::from_utf8(out).expect("serde_json writes UTF 8")
}

/// One line of JSON with `", "` and `": "` separators, as the tools print their summaries.
pub fn compact(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let items: Vec<String> = map.iter().map(|(k, v)| format!("{}: {}", compact(&Value::String(k.clone())), compact(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(items) => format!("[{}]", items.iter().map(compact).collect::<Vec<_>>().join(", ")),
        other => serde_json::to_string(other).expect("a JSON value"),
    }
}

/// `describe()`: translates `pack_id` in `scratch` and writes its module and description in `out_dir`.
pub fn describe(session: &Session, rows: &Rows, pack_id: i64, out_dir: &Path, scratch: &Path) -> Result<Outcome, Failure> {
    let meta = py::index(&rows.meta, pack_id)?;
    let outputs = py::index(&rows.outputs, pack_id)?;
    let Some(lines) = crate::identify::sources(&session.config, pack_id, scratch)? else {
        let name = field(meta, "name")?.clone();
        field(meta, "stage")?;
        return Ok(Outcome { name, ..Outcome::default() });
    };
    let raw_path = scratch.join("0.spv");
    let raw = std::fs::read(&raw_path).map_err(|e| Failure::io(&e, &raw_path))?;
    let stage = field(meta, "stage")?;
    let located = stage == "vertex" || stage == "pixel";
    let mut notes: Vec<String> = Vec::new();
    let spv = if located {
        let outputs = outputs.as_array().ok_or_else(|| Failure::error("TypeError", "outputs is not a list"))?;
        let (inputs_map, outputs_map) = decant_shader_core::locate::tables(meta, outputs).map_err(core)?;
        decant_shader_core::spirv::locate(&raw, &inputs_map, &outputs_map).map_err(core)?
    } else {
        raw
    };
    let facts = module_facts(&spv)?;
    let found = match session.identify(pack_id, &lines) {
        Ok(found) => found,
        Err(Failure::Exit(refusal)) => {
            notes.push(format!("resources not identified: {refusal}"));
            Vec::new()
        }
        Err(other) => return Err(other),
    };
    let base = if stage == "pixel" { 60 } else { 0 };
    let resources: Vec<Value> = found.iter().map(|r| resource(r, &facts, base)).collect();
    if found.iter().any(|r| r.origin == "autre") {
        notes.push("some resources are chosen at run time and are not resolved".into());
    }
    if found.iter().filter(|r| r.origin == "immediat").count() > 1 {
        notes.push("several immediate samplers: matching them to static samplers is not solved".into());
    }
    let push = match lines.iter().find(|l| l.starts_with("push ")) {
        Some(line) => Some(py::split(line)[1..].iter().map(|f| py::int(f)).collect::<Result<Vec<i64>, Failure>>()?),
        None => None,
    };
    let addressed = facts.capabilities.contains(&5347);
    if addressed {
        notes.push("reads memory by address: needs an address table on the Direct3D side".into());
    }
    let push_constants = match &push {
        None => Value::Null,
        Some(p) => {
            let at = |i: usize| p.get(i).copied().ok_or_else(|| Failure::error("IndexError", "list index out of range"));
            json!({"buffer_offsets_dword": at(0)?, "buffer_sizes_dword": at(1)?, "shader_data_dwords": p.get(2)})
        }
    };
    let mut inputs = inputs_of(meta::list(meta, "inputs").map_err(core)?)?;
    let mut outputs = outputs_of(outputs.as_array().ok_or_else(|| Failure::error("TypeError", "outputs is not a list"))?)?;
    if !located {
        notes.push("input and output locations keep the translator's numbering".into());
        for e in inputs.iter_mut().chain(outputs.iter_mut()) {
            e["location"] = Value::Null;
        }
    }
    let version = facts.version;
    let mut d = Map::new();
    d.insert("format".into(), json!("decant-shader-description"));
    d.insert("format_version".into(), json!(FORMAT_VERSION));
    d.insert("pack_id".into(), json!(pack_id));
    d.insert("name".into(), field(meta, "name")?.clone());
    d.insert("stage".into(), stage.clone());
    d.insert("translated".into(), json!(true));
    d.insert("spirv".into(), json!(format!("{pack_id}.spv")));
    d.insert("spirv_sha1".into(), json!(Sha1::digest(&spv).iter().map(|b| format!("{b:02x}")).collect::<String>()));
    d.insert("spirv_version".into(), json!(format!("{}.{}", version >> 16 & 0xFF, version >> 8 & 0xFF)));
    d.insert("capabilities".into(), Value::Array(facts.capabilities.iter().map(|&c| capability(c)).collect()));
    d.insert("addressed_memory".into(), json!(addressed));
    d.insert("threads".into(), meta.get("threads").cloned().unwrap_or(Value::Null));
    d.insert("mesh_limits".into(), if stage == "mesh" { meta.get("mesh_limits").cloned().unwrap_or(Value::Null) } else { Value::Null });
    d.insert("inputs".into(), Value::Array(inputs));
    d.insert("outputs".into(), Value::Array(outputs));
    d.insert("resources".into(), Value::Array(resources));
    d.insert("push_constants".into(), push_constants);
    d.insert("notes".into(), json!(notes));
    let spv_path = out_dir.join(format!("{pack_id}.spv"));
    std::fs::write(&spv_path, &spv).map_err(|e| Failure::io(&e, &spv_path))?;
    let json_path = out_dir.join(format!("{pack_id}.json"));
    let description = Value::Object(d);
    std::fs::write(&json_path, py::native_lines(&pretty(&description))).map_err(|e| Failure::io(&e, &json_path))?;
    Ok(Outcome {
        translated: true,
        resources: found.len(),
        resolved: found.iter().filter(|r| r.known()).count(),
        name: field(meta, "name")?.clone(),
        description: Some(description),
        error: None,
    })
}


/// `one()`: [`describe`] in a scratch folder of its own, removed afterwards. A failure counts as
/// an untranslated shader and does not stop the run.
pub fn one(session: &Session, rows: &Rows, pack_id: i64, out_dir: &Path) -> Outcome {
    let Ok(scratch) = tempfile::Builder::new().prefix("decant_shader_describe_").tempdir() else {
        return Outcome::default();
    };
    let outcome = describe(session, rows, pack_id, out_dir, scratch.path()).unwrap_or_else(|failure| {
        let repr = match failure {
            Failure::Error { repr, .. } => repr,
            Failure::Exit(message) => format!("SystemExit({})", py::repr_str(&message)),
        };
        Outcome { error: Some(repr.chars().take(300).collect()), ..Outcome::default() }
    });
    let _ = std::fs::remove_dir_all(scratch.path());
    outcome
}

/// The counts written to `summary.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub translated: usize,
    pub failed: usize,
    pub resources: usize,
    pub resolved: usize,
}

impl Summary {
    pub fn add(&mut self, outcome: &Outcome) {
        if !outcome.translated {
            self.failed += 1;
            return;
        }
        self.translated += 1;
        self.resources += outcome.resources;
        self.resolved += outcome.resolved;
    }

    fn value(&self) -> Value {
        json!({"translated": self.translated, "failed": self.failed, "resources": self.resources, "resolved": self.resolved})
    }

    /// `json.dumps(summary, indent=1)`.
    pub fn file(&self) -> String {
        pretty(&self.value())
    }

    /// `json.dumps(summary)`.
    pub fn line(&self) -> String {
        format!("{{\"translated\": {}, \"failed\": {}, \"resources\": {}, \"resolved\": {}}}", self.translated, self.failed, self.resources, self.resolved)
    }
}

/// The shaders a run describes: the ids given, or every row (of one stage) with `--all`, then
/// `--sample` (a draw with `random.seed(21)`, sorted) and `--limit`.
pub fn select(rows: &[Value], ids: Vec<i64>, all: bool, stage: Option<&str>, sample: Option<i64>, limit: Option<i64>) -> Result<Vec<i64>, Failure> {
    let mut ids = ids;
    if all {
        ids = Vec::new();
        for m in rows {
            if stage.is_none_or(|s| field(m, "stage").is_ok_and(|v| v == s)) {
                ids.push(field(m, "id")?.as_i64().ok_or_else(|| Failure::error("TypeError", "id is not an integer"))?);
            }
        }
    }
    if let Some(n) = sample.filter(|&n| n != 0) {
        let k = n.min(ids.len() as i64);
        if k < 0 {
            return Err(Failure::value("Sample larger than population or is negative".into()));
        }
        ids = crate::random::Random::seed(21).sample(&ids, k as usize);
        ids.sort_unstable();
    }
    if let Some(n) = limit.filter(|&n| n != 0) {
        let len = ids.len() as i64;
        let end = if n < 0 { (len + n).max(0) } else { n.min(len) };
        ids.truncate(end as usize);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(code: u32, operands: &[u32]) -> Vec<u32> {
        let mut out = vec![((operands.len() as u32 + 1) << 16) | code];
        out.extend_from_slice(operands);
        out
    }

    /// An invented module: SPIR-V 1.5, capabilities Shader, Int64 and an unnamed one, variable 10
    /// at set 2 binding 60, variable 11 at binding 61 without a set, and 12 decorated twice.
    fn module() -> Vec<u8> {
        let mut w = vec![0x0723_0203, 0x0001_0500, 0, 20, 0];
        w.extend(op(17, &[1]));
        w.extend(op(17, &[11]));
        w.extend(op(17, &[777]));
        w.extend(op(71, &[10, 34, 2]));
        w.extend(op(71, &[10, 33, 60]));
        w.extend(op(71, &[11, 33, 61]));
        w.extend(op(71, &[12, 33, 70]));
        w.extend(op(71, &[12, 33, 61]));
        w.extend(op(71, &[12, 34, 5]));
        w.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn module_facts_follow_the_python_dicts() {
        let facts = module_facts(&module()).unwrap();
        assert_eq!(facts.version, 0x0001_0500);
        assert_eq!(facts.capabilities, [1, 11, 777]);
        // Variable 12 comes after 11 in the dict, so binding 61 takes its set.
        assert_eq!(facts.by_binding, HashMap::from([(60, (2, 60)), (61, (5, 61))]));
        assert_eq!(capability(11), json!("Int64"));
        assert_eq!(capability(777), json!(777));
        assert!(module_facts(&module()[..30]).is_err());
    }

    #[test]
    fn signature_elements_get_their_locations() {
        let rows = vec![json!(["SV_Position", 0, 0, 15, 15, 1, 3, 0, 0]), json!(["TEXCOORD", 0, 1, 3, 3, 0, 3, 0, 0]), json!(["TEXCOORD", 1, 2, 3, 0, 0, 3, 0, 0])];
        let inputs = inputs_of(&rows).unwrap();
        assert_eq!(inputs[0]["location"], Value::Null);
        assert_eq!(inputs[1]["location"], json!(1));
        assert_eq!(inputs[2]["location"], Value::Null);
        assert_eq!(inputs[1]["used_mask"], json!(3));
        let outputs = outputs_of(&[json!(["SV_Target", 0, 0, 15, 0, 64, 3, 0, 0]), json!(["SV_Depth", 0, 4294967295u32, 1, 14, 65, 3, 0, 0])]).unwrap();
        assert_eq!(outputs[0]["location"], json!(0));
        assert_eq!(outputs[1]["location"], Value::Null);
        assert_eq!(outputs[0]["never_written_mask"], json!(0));
        assert!(inputs_of(&[json!(["X", 0, 1])]).is_err());
    }

    #[test]
    fn resources_and_layout_match_the_python() {
        let facts = module_facts(&module()).unwrap();
        let r = Resource { kind: "image".into(), index: 0, register: Some("t188".into()), origin: "table", heap: None, binding: Some(1), rank: Some(0xFFFF_FFFF), table: Some(4), offset: Some(8) };
        let value = resource(&r, &facts, 60);
        let text = pretty(&value);
        let expected = "{\n \"kind\": \"image\",\n \"index\": 0,\n \"d3d_register\": \"t188\",\n \"heap_index\": null,\n \"origin\": \"table\",\n \"descriptor_set\": 5,\n \"binding\": 61,\n \"array_rank\": null,\n \"descriptor_table_sgpr\": 4,\n \"table_offset_bytes\": 8\n}";
        assert_eq!(text, expected);
        assert_eq!(pretty(&json!({"a": [], "b": {}, "c": "é\u{1}"})), "{\n \"a\": [],\n \"b\": {},\n \"c\": \"é\\u0001\"\n}");
        assert_eq!(origin("tas-samplers"), "sampler-heap");
    }

    #[test]
    fn summary_and_selection() {
        let mut s = Summary::default();
        s.add(&Outcome { translated: true, resources: 3, resolved: 2, ..Outcome::default() });
        s.add(&Outcome::default());
        assert_eq!(s.line(), "{\"translated\": 1, \"failed\": 1, \"resources\": 3, \"resolved\": 2}");
        assert_eq!(s.file(), "{\n \"translated\": 1,\n \"failed\": 1,\n \"resources\": 3,\n \"resolved\": 2\n}");
        let rows: Vec<Value> = (0..6).map(|i| json!({"id": i, "stage": if i % 2 == 0 { "pixel" } else { "vertex" }})).collect();
        assert_eq!(select(&rows, vec![], true, Some("pixel"), None, None).unwrap(), [0, 2, 4]);
        assert_eq!(select(&rows, vec![5, 1, 3], false, None, None, Some(2)).unwrap(), [5, 1]);
        assert_eq!(select(&rows, vec![5, 1, 3], false, None, None, Some(-1)).unwrap(), [5, 1]);
        assert_eq!(select(&rows, vec![], true, None, Some(100), None).unwrap(), [0, 1, 2, 3, 4, 5]);
    }
}
