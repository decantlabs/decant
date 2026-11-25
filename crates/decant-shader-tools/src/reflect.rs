//! `spirv_reflect.py`: a minimal reading of a SPIR-V module. Resource variables (set, binding,
//! kind, array size, image type), push constants, capabilities and inputs and outputs. The
//! shader gallery of the harness uses it, and other crates can use it as a library.
//!
//! The reading is the script's own and keeps its rules: a word count of zero ends it, the last
//! definition of an id wins, a decoration without operand counts as `true`, an unknown image type
//! gives `?` fields. A module the script would fail on (cut instructions, a pointer to a type it
//! cannot unpack) is an error here. It is word for word, not a full parser such as `rspirv`, so
//! that it accepts and refuses exactly what the script does.

use std::collections::HashMap;

use decant_shader_analysis::py::Py;

/// A decoration operand: a number, `true` for a decoration without operand, or the `?` of an
/// image type the module does not define.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Int(u32),
    Flag,
    Unknown,
}

impl Operand {
    pub fn py(self) -> Py {
        match self {
            Operand::Int(v) => Py::Int(i64::from(v)),
            Operand::Flag => Py::Bool(true),
            Operand::Unknown => Py::Str("?".to_owned()),
        }
    }

    /// `bool(value)`.
    fn truthy(self) -> bool {
        !matches!(self, Operand::Int(0))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    CombinedImage,
    Unknown,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Sampler => "sampler",
            Kind::SampledImage => "sampled_image",
            Kind::StorageImage => "storage_image",
            Kind::UniformTexelBuffer => "uniform_texel_buffer",
            Kind::StorageTexelBuffer => "storage_texel_buffer",
            Kind::UniformBuffer => "uniform_buffer",
            Kind::StorageBuffer => "storage_buffer",
            Kind::CombinedImage => "combined_image",
            Kind::Unknown => "unknown",
        }
    }
}

/// The image type of an image resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub dim: &'static str,
    pub depth: Operand,
    pub arrayed: bool,
    pub ms: bool,
    pub sampled_type: &'static str,
    pub format: Operand,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    pub set: Operand,
    pub binding: Operand,
    /// The array size, 1 for a single resource, 0 for an array of unknown size.
    pub count: u32,
    pub name: String,
    pub kind: Kind,
    pub image: Option<Image>,
}

/// An input or output variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interface {
    pub location: Option<Operand>,
    pub builtin: Option<Operand>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reflection {
    pub version: u32,
    /// Every `OpCapability` in module order, repeats kept.
    pub capabilities: Vec<u32>,
    pub resources: Vec<Resource>,
    /// 128 when the module declares push constants, else 0.
    pub push_size: u32,
    pub inputs: Vec<Interface>,
    pub outputs: Vec<Interface>,
}

const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_INPUT: u32 = 1;
const SC_UNIFORM: u32 = 2;
const SC_OUTPUT: u32 = 3;
const SC_PUSH: u32 = 9;
const SC_STORAGE: u32 = 12;

fn dim(value: Operand) -> &'static str {
    match value {
        Operand::Int(0) => "1d",
        Operand::Int(1) => "2d",
        Operand::Int(2) => "3d",
        Operand::Int(3) => "cube",
        Operand::Int(4) => "rect",
        Operand::Int(5) => "buffer",
        Operand::Int(6) => "subpass",
        _ => "?",
    }
}

/// A type as the script stores it: its tag and its operands. A struct keeps its members in
/// `fields`, but counts as a pair, as the tuple `("struct", members)` does.
#[derive(Clone, Debug)]
struct Type {
    tag: &'static str,
    fields: Vec<u32>,
}

impl Type {
    fn len(&self) -> usize {
        if self.tag == "struct" { 2 } else { 1 + self.fields.len() }
    }
}

/// `_string(words)`: the bytes up to the first zero, decoded as UTF 8 with replacement.
fn string(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Reflects a module, or says where the script would raise.
pub fn reflect(data: &[u8]) -> Result<Reflection, String> {
    if !data.len().is_multiple_of(4) {
        return Err("struct.error: the module is not a whole number of words".to_owned());
    }
    let words: Vec<u32> = data.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    let mut types: HashMap<u32, Type> = HashMap::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    let mut decorations: HashMap<u32, HashMap<u32, Operand>> = HashMap::new();
    let mut buffer_blocks: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut variables: Vec<(u32, u32, u32)> = Vec::new();
    let mut capabilities = Vec::new();
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut i = 5;
    while i < words.len() {
        let (op, count) = (words[i] & 0xFFFF, (words[i] >> 16) as usize);
        if count == 0 {
            break;
        }
        let ins = &words[i..words.len().min(i + count)];
        let at = |k: usize| ins.get(k).copied().ok_or_else(|| "IndexError: tuple index out of range".to_owned());
        let ty = |tag: &'static str, n: usize| -> Result<(u32, Type), String> {
            let fields = (2..2 + n).map(at).collect::<Result<Vec<_>, _>>()?;
            Ok((at(1)?, Type { tag, fields }))
        };
        let mut define = |(id, t): (u32, Type)| {
            types.insert(id, t);
        };
        match op {
            17 => capabilities.push(at(1)?),
            5 => {
                names.insert(at(1)?, string(ins.get(2..).unwrap_or_default()));
            }
            71 => {
                let (target, decoration) = (at(1)?, at(2)?);
                let value = if count > 3 { Operand::Int(at(3)?) } else { Operand::Flag };
                decorations.entry(target).or_default().insert(decoration, value);
                if decoration == 3 {
                    buffer_blocks.insert(target);
                }
            }
            19 => define(ty("void", 0)?),
            20 => define(ty("bool", 0)?),
            21 => define(ty("int", 2)?),
            22 => define(ty("float", 1)?),
            23 => define(ty("vector", 2)?),
            25 => define(ty("image", 7)?),
            26 => define(ty("sampler", 0)?),
            27 => define(ty("sampled_image", 1)?),
            28 => define(ty("array", 2)?),
            29 => define(ty("runtime_array", 1)?),
            30 => define((at(1)?, Type { tag: "struct", fields: ins.get(2..).unwrap_or_default().to_vec() })),
            32 => define(ty("pointer", 2)?),
            43 if count == 4 => {
                constants.insert(at(2)?, at(3)?);
            }
            59 => variables.push((at(2)?, at(1)?, at(3)?)),
            _ => {}
        }
        i += count;
    }

    let scalar_kind = |id: Option<u32>| match id.and_then(|id| types.get(&id)) {
        Some(t) if t.tag == "float" => "float",
        Some(t) if t.tag == "int" => {
            if t.fields[1] != 0 {
                "sint"
            } else {
                "uint"
            }
        }
        _ => "?",
    };
    let (mut resources, mut push_size, mut inputs, mut outputs) = (Vec::new(), 0, Vec::new(), Vec::new());
    let empty = HashMap::new();
    for &(ident, pointer, storage) in &variables {
        let pointee = match types.get(&pointer) {
            None => None,
            Some(t) if t.len() == 3 => Some(t.fields[1]),
            Some(t) => return Err(format!("ValueError: cannot unpack the {} type {pointer} as a pointer", t.tag)),
        };
        let dec = decorations.get(&ident).unwrap_or(&empty);
        if storage == SC_PUSH {
            push_size = 128;
            continue;
        }
        if storage == SC_INPUT || storage == SC_OUTPUT {
            let entry = Interface { location: dec.get(&30).copied(), builtin: dec.get(&11).copied() };
            if storage == SC_INPUT { inputs.push(entry) } else { outputs.push(entry) }
            continue;
        }
        if ![SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE].contains(&storage) {
            continue;
        }
        let (mut count, mut element) = (1, pointee);
        match element.and_then(|e| types.get(&e)) {
            Some(t) if t.tag == "array" => {
                count = constants.get(&t.fields[1]).copied().unwrap_or(1);
                element = Some(t.fields[0]);
            }
            Some(t) if t.tag == "runtime_array" => {
                count = 0;
                element = Some(t.fields[0]);
            }
            _ => {}
        }
        let t = element.and_then(|e| types.get(&e));
        let (kind, image) = match t {
            Some(t) if t.tag == "sampler" => (Kind::Sampler, None),
            Some(t) if t.tag == "sampled_image" => {
                let image = match types.get(&t.fields[0]) {
                    None => Image { dim: "?", depth: Operand::Unknown, arrayed: true, ms: true, sampled_type: "?", format: Operand::Unknown },
                    Some(i) if i.tag == "image" => image_of(&i.fields, scalar_kind),
                    Some(i) => return Err(format!("IndexError: the {} type {} is not an image", i.tag, t.fields[0])),
                };
                (Kind::CombinedImage, Some(image))
            }
            Some(t) if t.tag == "image" => {
                let image = image_of(&t.fields, scalar_kind);
                let sampled = t.fields[5] == 1;
                let kind = match (image.dim == "buffer", sampled) {
                    (true, true) => Kind::UniformTexelBuffer,
                    (true, false) => Kind::StorageTexelBuffer,
                    (false, true) => Kind::SampledImage,
                    (false, false) => Kind::StorageImage,
                };
                (kind, Some(image))
            }
            Some(_) if t.is_some_and(|t| t.tag == "struct") => {
                let buffer_block = element.is_some_and(|e| buffer_blocks.contains(&e));
                (if storage == SC_STORAGE || buffer_block { Kind::StorageBuffer } else { Kind::UniformBuffer }, None)
            }
            _ => (Kind::Unknown, None),
        };
        resources.push(Resource {
            set: dec.get(&34).copied().unwrap_or(Operand::Int(0)),
            binding: dec.get(&33).copied().unwrap_or(Operand::Int(0)),
            count,
            name: names.get(&ident).cloned().unwrap_or_default(),
            kind,
            image,
        });
    }
    let version = *words.get(1).ok_or("IndexError: tuple index out of range")?;
    Ok(Reflection { version, capabilities, resources, push_size, inputs, outputs })
}

fn image_of(f: &[u32], scalar_kind: impl Fn(Option<u32>) -> &'static str) -> Image {
    Image {
        dim: dim(Operand::Int(f[1])),
        depth: Operand::Int(f[2]),
        arrayed: Operand::Int(f[3]).truthy(),
        ms: Operand::Int(f[4]).truthy(),
        sampled_type: scalar_kind(Some(f[0])),
        format: Operand::Int(f[6]),
    }
}

impl Reflection {
    /// The dictionary `reflect()` returns, keys in its order.
    pub fn py(&self) -> Py {
        let ints = |v: &[u32]| Py::List(v.iter().map(|&c| Py::Int(i64::from(c))).collect());
        let opt = |o: Option<Operand>| o.map_or(Py::None, Operand::py);
        let io = |v: &[Interface]| {
            Py::List(v.iter().map(|e| Py::Dict(vec![("location".into(), opt(e.location)), ("builtin".into(), opt(e.builtin))])).collect())
        };
        let resources = self
            .resources
            .iter()
            .map(|r| {
                let mut d: Vec<(Py, Py)> = vec![
                    ("set".into(), r.set.py()),
                    ("binding".into(), r.binding.py()),
                    ("count".into(), r.count.into()),
                    ("name".into(), r.name.clone().into()),
                    ("kind".into(), r.kind.as_str().into()),
                ];
                if let Some(i) = &r.image {
                    d.extend([
                        ("dim".into(), i.dim.into()),
                        ("depth".into(), i.depth.py()),
                        ("arrayed".into(), Py::Bool(i.arrayed)),
                        ("ms".into(), Py::Bool(i.ms)),
                        ("sampled_type".into(), i.sampled_type.into()),
                        ("format".into(), i.format.py()),
                    ]);
                }
                Py::Dict(d)
            })
            .collect();
        Py::Dict(vec![
            ("version".into(), self.version.into()),
            ("capabilities".into(), ints(&self.capabilities)),
            ("resources".into(), Py::List(resources)),
            ("push_size".into(), self.push_size.into()),
            ("inputs".into(), io(&self.inputs)),
            ("outputs".into(), io(&self.outputs)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::Module;

    #[test]
    fn resources_follow_the_script_rules() {
        let m = Module::reflect_sample();
        let r = reflect(&m.bytes()).unwrap();
        let kinds: Vec<&str> = r.resources.iter().map(|r| r.kind.as_str()).collect();
        assert_eq!(kinds, ["combined_image", "storage_image", "uniform_texel_buffer", "uniform_buffer", "storage_buffer", "sampler", "combined_image", "unknown"]);
        assert_eq!(r.resources[0].count, 4, "an array takes its length from the constant");
        assert_eq!(r.resources[1].count, 0, "a runtime array has size 0");
        assert_eq!(r.resources[0].name, "tex");
        assert_eq!(r.resources[6].image.as_ref().unwrap().dim, "?", "an undefined image type gives ? fields");
        assert_eq!(r.push_size, 128);
        assert_eq!(r.inputs, [Interface { location: Some(Operand::Int(2)), builtin: None }]);
        assert_eq!(r.outputs, [Interface { location: None, builtin: Some(Operand::Flag) }]);
        assert_eq!(r.capabilities, [1, 1, 5347]);
    }

    #[test]
    fn modules_the_script_refuses_are_errors() {
        assert!(reflect(&[0; 6]).is_err(), "not whole words");
        let mut words = Module::new().words();
        words.push((4 << 16) | 71);
        words.push(1);
        assert!(reflect(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()).is_err(), "a cut decoration");
    }
}
