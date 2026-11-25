//! Rewrites the resources of a DecantShader SPIR-V module so it follows the layout of another engine
//! (vkd3d-proton), from the automatic identification of `identify`, as
//! `src/shaders/decant_shader/tools/decant_shader_adapt.py` does.
//!
//! The translator keeps its resources by kind, in arrays (buffers at the base binding, images,
//! samplers). Each resource whose Direct3D register is known and appears in the requested layout
//! leaves its array, becomes a variable of its own, and gets the requested set and binding. A
//! constant buffer can become a uniform buffer of fixed size. Reads of the translator's push constants
//! (offset and size of each bound buffer) are replaced by constants: offset 0, largest size.
//!
//! The module is read and written with `rspirv`. Its instructions are walked in the order of the
//! binary, and the checks that work on raw words in the Python (which words may be ids) still
//! work on each instruction's words, so the same modules are accepted and refused.
//!
//! Layout (JSON): `{"b27": {"set": 103, "binding": 0, "genre": "uniforme"}, "t188": {"set": 0, "binding": 0}}`

use std::collections::{HashMap, HashSet};

use rspirv::binary::Assemble;
use rspirv::dr::{self, Instruction, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::identify::Resource;
use crate::py::{self, Failure};

/// 64 KiB: the largest Direct3D 12 constant buffer.
pub const UNIFORM_WORDS: u32 = 16384;

/// Where one register goes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub set: u32,
    pub binding: u32,
    /// `"genre": "uniforme"`: the buffer becomes a uniform buffer of `UNIFORM_WORDS` words.
    pub uniform: bool,
}

/// Register to target, in the order of the JSON object.
pub type Layout = Vec<(String, Target)>;

/// Reads a layout as `json.loads` gives it to the Python.
pub fn layout_from_json(value: &serde_json::Value) -> Result<Layout, Failure> {
    let object = value.as_object().ok_or_else(|| Failure::error("AttributeError", "the layout is not a JSON object"))?;
    let mut layout = Layout::new();
    for (register, entry) in object {
        let number = |key: &str| -> Result<u32, Failure> {
            let field = entry.get(key).ok_or_else(|| Failure::key(py::repr_str(key)))?;
            field.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| Failure::error("struct.error", "argument out of range"))
        };
        let target = Target { set: number("set")?, binding: number("binding")?, uniform: entry.get("genre").and_then(|g| g.as_str()) == Some("uniforme") };
        layout.push((register.clone(), target));
    }
    Ok(layout)
}

fn lookup<'a>(layout: &'a Layout, register: &str) -> Option<&'a Target> {
    layout.iter().find(|(r, _)| r == register).map(|(_, t)| t)
}

/// One word of an operand, as it sits in the binary.
fn word(operand: &Operand) -> Option<u32> {
    match *operand {
        Operand::IdRef(w) | Operand::LiteralBit32(w) => Some(w),
        Operand::StorageClass(s) => Some(s as u32),
        Operand::Decoration(d) => Some(d as u32),
        _ => None,
    }
}

fn operand(inst: &Instruction, i: usize) -> Option<u32> {
    inst.operands.get(i).and_then(word)
}

fn opcode(inst: &Instruction) -> Op {
    inst.class.opcode
}

fn nop() -> Instruction {
    Instruction::new(Op::Nop, None, None, Vec::new())
}

fn is_chain(op: Op) -> bool {
    matches!(op, Op::AccessChain | Op::InBoundsAccessChain)
}

/// Operations whose words from the first index on (to the second, when given) are literals.
fn literals_from(op: u32) -> Option<(usize, Option<usize>)> {
    match op {
        12 => Some((4, Some(5))),
        79 | 82 => Some((5, None)),
        81 => Some((4, None)),
        246 => Some((3, None)),
        247 => Some((2, None)),
        8 => Some((1, None)),
        _ => None,
    }
}

/// The words of an instruction that may be ids, without the literals of the operations that carry
/// some: the instruction number of OpExtInst, the indexes of extracts, inserts and shuffles, the
/// flags of merges, and all of OpLine. Every word is kept for the other operations: a literal equal to an id by chance then
/// refuses the module, which is the careful side.
pub fn id_operands(words: &[u32]) -> Vec<u32> {
    let op = words[0] & 0xFFFF;
    if op == 251 {
        // OpSwitch: selector, default, then (literal, label) pairs.
        let mut ids = words[1..3.min(words.len())].to_vec();
        ids.extend(words.iter().skip(4).step_by(2));
        return ids;
    }
    match literals_from(op) {
        Some((start, end)) => {
            let mut ids = words[1..start.min(words.len())].to_vec();
            if let Some(end) = end {
                ids.extend(words.iter().skip(end));
            }
            ids
        }
        None => words[1..].to_vec(),
    }
}

struct Module {
    m: dr::Module,
    uint: u32,
    constants: HashMap<u32, u32>,
    /// Pointer types in module order, as the Python dict keeps them.
    pointers: Vec<(u32, (u32, u32))>,
    arrays: HashMap<u32, u32>,
    variables: HashMap<u32, (u32, u32)>,
    bindings: Vec<(u32, u32)>,
    new_globals: Vec<Instruction>,
    new_decorations: Vec<Instruction>,
    new_interface: Vec<u32>,
}

fn stop() -> Failure {
    Failure::error("StopIteration", "")
}

impl Module {
    fn load(data: &[u8]) -> Result<Module, Failure> {
        if !data.len().is_multiple_of(4) {
            return Err(Failure::error("struct.error", &format!("unpack requires a buffer of {} bytes", data.len() / 4 * 4)));
        }
        let words: Vec<u32> = data.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        let mut m = dr::load_words(&words).map_err(|e| Failure::Exit(format!("module SPIR-V illisible ({e}) : non gere")))?;
        // The loader puts its own generator number in the header: keep the translator's.
        if let Some(header) = m.header.as_mut() {
            header.generator = words[2];
        }
        // The rewrite must keep every other word as it is: refuse a module rspirv would change.
        if m.assemble() != words {
            return Err(Failure::Exit("module SPIR-V que rspirv ne reecrit pas a l'identique : non gere".into()));
        }
        let mut module = Module {
            m,
            uint: 0,
            constants: HashMap::new(),
            pointers: Vec::new(),
            arrays: HashMap::new(),
            variables: HashMap::new(),
            bindings: Vec::new(),
            new_globals: Vec::new(),
            new_decorations: Vec::new(),
            new_interface: Vec::new(),
        };
        module.scan()?;
        Ok(module)
    }

    fn scan(&mut self) -> Result<(), Failure> {
        let all: Vec<&Instruction> = self.m.all_inst_iter().collect();
        self.uint = all
            .iter()
            .find(|i| opcode(i) == Op::TypeInt && operand(i, 0) == Some(32) && operand(i, 1) == Some(0))
            .and_then(|i| i.result_id)
            .ok_or_else(stop)?;
        for inst in all {
            let (Some(id), op) = (inst.result_id, opcode(inst)) else {
                if opcode(inst) == Op::Decorate && operand(inst, 1) == Some(Decoration::Binding as u32)
                    && let (Some(target), Some(binding)) = (operand(inst, 0), operand(inst, 2))
                {
                    match self.bindings.iter_mut().find(|(t, _)| *t == target) {
                        Some(entry) => entry.1 = binding,
                        None => self.bindings.push((target, binding)),
                    }
                }
                continue;
            };
            match op {
                Op::Constant if inst.result_type == Some(self.uint) => {
                    if let Some(value) = operand(inst, 0) {
                        self.constants.insert(id, value);
                    }
                }
                Op::TypePointer => {
                    if let (Some(storage), Some(pointee)) = (operand(inst, 0), operand(inst, 1)) {
                        match self.pointers.iter_mut().find(|(p, _)| *p == id) {
                            Some(entry) => entry.1 = (storage, pointee),
                            None => self.pointers.push((id, (storage, pointee))),
                        }
                    }
                }
                Op::TypeArray | Op::TypeRuntimeArray => {
                    if let Some(element) = operand(inst, 0) {
                        self.arrays.insert(id, element);
                    }
                }
                Op::Variable => {
                    if let (Some(pointer), Some(storage)) = (inst.result_type, operand(inst, 0)) {
                        self.variables.insert(id, (pointer, storage));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn header(&mut self) -> &mut dr::ModuleHeader {
        self.m.header.as_mut().expect("a loaded module has a header")
    }

    fn new_id(&mut self) -> u32 {
        let header = self.header();
        header.bound += 1;
        header.bound - 1
    }

    fn constant(&mut self, value: u32) -> u32 {
        let id = self.new_id();
        self.new_globals.push(Instruction::new(Op::Constant, Some(self.uint), Some(id), vec![Operand::LiteralBit32(value)]));
        id
    }

    fn pointer(&mut self, storage: u32, pointee: u32) -> Result<u32, Failure> {
        if let Some((id, _)) = self.pointers.iter().find(|(_, p)| *p == (storage, pointee)) {
            return Ok(*id);
        }
        let id = self.new_id();
        self.new_globals.push(Instruction::new(Op::TypePointer, None, Some(id), vec![Operand::StorageClass(storage_class(storage)?), Operand::IdRef(pointee)]));
        self.pointers.push((id, (storage, pointee)));
        Ok(id)
    }

    fn variable(&mut self, pointer: u32, storage: u32, set: u32, binding: u32) -> Result<u32, Failure> {
        let id = self.new_id();
        self.new_globals.push(Instruction::new(Op::Variable, Some(pointer), Some(id), vec![Operand::StorageClass(storage_class(storage)?)]));
        self.new_decorations.push(decorate(id, Decoration::DescriptorSet, &[set]));
        self.new_decorations.push(decorate(id, Decoration::Binding, &[binding]));
        // Required from SPIR-V 1.4 on only.
        if self.header().version >= 0x0001_0400 {
            self.new_interface.push(id);
        }
        Ok(id)
    }

    fn serialize(mut self) -> Result<Vec<u8>, Failure> {
        let first_type = self.m.types_global_values.iter().position(|i| (19..=39).contains(&(opcode(i) as u32))).ok_or_else(stop)?;
        if self.m.functions.is_empty() {
            return Err(stop());
        }
        let decorations = std::mem::take(&mut self.new_decorations);
        self.m.types_global_values.splice(first_type..first_type, decorations);
        self.m.types_global_values.append(&mut self.new_globals);
        for entry in &mut self.m.entry_points {
            entry.operands.extend(self.new_interface.iter().map(|&id| Operand::IdRef(id)));
        }
        Ok(self.m.assemble().iter().flat_map(|w| w.to_le_bytes()).collect())
    }
}

fn storage_class(value: u32) -> Result<StorageClass, Failure> {
    StorageClass::from_u32(value).ok_or_else(|| Failure::error("ValueError", &format!("storage class {value}")))
}

fn decorate(id: u32, decoration: Decoration, literals: &[u32]) -> Instruction {
    let mut operands = vec![Operand::IdRef(id), Operand::Decoration(decoration)];
    operands.extend(literals.iter().map(|&l| Operand::LiteralBit32(l)));
    Instruction::new(Op::Decorate, None, None, operands)
}

struct Moved {
    variable: u32,
    uniform: bool,
    pointer: u32,
}

/// Rewrites `data` for `layout`. `push` is the probe's `push` line (its numbers), `pixel` whether
/// the shader is a pixel shader (its arrays start at binding 60). Gives the module and the report
/// lines, or the refusal.
pub fn adapt(data: &[u8], resources: &[Resource], push: Option<&[i64]>, layout: &Layout, pixel: bool) -> Result<(Vec<u8>, Vec<String>), Failure> {
    let mut m = Module::load(data)?;
    let base: i64 = if pixel { 60 } else { 0 };
    let mut by_binding: HashMap<u32, u32> = HashMap::new();
    for &(id, binding) in &m.bindings {
        by_binding.insert(binding, id);
    }
    let mut moved: HashMap<(u32, i64), Moved> = HashMap::new();
    let mut uniform_uint_pointer = None;
    let mut report = Vec::new();
    for r in resources {
        let target = r.register.as_deref().and_then(|reg| lookup(layout, reg));
        let Some(target) = target else {
            report.push(format!("{} {} : laisse en place ({})", r.kind, r.index, r.register.as_deref().unwrap_or(r.origin)));
            continue;
        };
        let register = r.register.as_deref().unwrap_or_default();
        let Some(binding) = r.binding else {
            return Err(Failure::error("TypeError", "unsupported operand type(s) for +: 'int' and 'NoneType'"));
        };
        let array_variable = u32::try_from(base + binding).ok().and_then(|b| by_binding.get(&b).copied());
        let (Some(array_variable), Some(rank)) = (array_variable, r.rank.filter(|&rank| rank < 0xFFFF_FFFF)) else {
            return Err(Failure::Exit(format!("{} {} ({register}) : pas de place dans le module", r.kind, r.index)));
        };
        let &(pointer_type, _) = m.variables.get(&array_variable).ok_or_else(|| Failure::key(array_variable.to_string()))?;
        let &(_, (storage, array_type)) = m.pointers.iter().find(|(p, _)| *p == pointer_type).ok_or_else(|| Failure::key(pointer_type.to_string()))?;
        let &element = m.arrays.get(&array_type).ok_or_else(|| Failure::key(array_type.to_string()))?;
        let (variable, pointer) = if target.uniform {
            let fixed = m.new_id();
            let block = m.new_id();
            let length = m.constant(UNIFORM_WORDS);
            m.new_globals.push(Instruction::new(Op::TypeArray, None, Some(fixed), vec![Operand::IdRef(m.uint), Operand::IdRef(length)]));
            m.new_globals.push(Instruction::new(Op::TypeStruct, None, Some(block), vec![Operand::IdRef(fixed)]));
            m.new_decorations.push(decorate(fixed, Decoration::ArrayStride, &[4]));
            m.new_decorations.push(decorate(block, Decoration::Block, &[]));
            m.new_decorations.push(Instruction::new(
                Op::MemberDecorate,
                None,
                None,
                vec![Operand::IdRef(block), Operand::LiteralBit32(0), Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(0)],
            ));
            let pointer = m.pointer(StorageClass::Uniform as u32, block)?;
            uniform_uint_pointer = Some(m.pointer(StorageClass::Uniform as u32, m.uint)?);
            (m.variable(pointer, StorageClass::Uniform as u32, target.set, target.binding)?, pointer)
        } else {
            let pointer = m.pointer(storage, element)?;
            (m.variable(pointer, storage, target.set, target.binding)?, pointer)
        };
        moved.insert((array_variable, rank), Moved { variable, uniform: target.uniform, pointer });
        report.push(format!(
            "{} {} : {register} -> jeu {}, binding {}{}",
            r.kind,
            r.index,
            target.set,
            target.binding,
            if target.uniform { ", tampon uniforme" } else { "" }
        ));
    }

    let push_variables: HashSet<u32> = m.variables.iter().filter(|(_, (_, sc))| *sc == StorageClass::PushConstant as u32).map(|(&id, _)| id).collect();
    let mut push_chains: HashMap<u32, u32> = HashMap::new();
    // Push constant reads are rewritten in one form only: an access chain with a constant index,
    // read by OpLoad alone. Any other form is refused here rather than giving an invalid module.
    for inst in m.m.all_inst_iter() {
        if is_chain(opcode(inst)) && operand(inst, 0).is_some_and(|s| push_variables.contains(&s)) {
            let last = inst.operands.last().and_then(word);
            let constant = last.filter(|_| inst.operands.len() >= 2).and_then(|l| m.constants.get(&l));
            let Some(&constant) = constant else {
                return Err(Failure::Exit(format!("constantes poussees lues a un indice non constant (resultat %{}) : non gere", inst.result_id.unwrap_or(0))));
            };
            push_chains.insert(inst.result_id.unwrap_or(0), constant);
        }
    }
    for inst in m.m.functions.iter().flat_map(|f| f.all_inst_iter()) {
        let op = opcode(inst);
        // OpLoad: only its pointer is an id, the rest are literals.
        if op == Op::Load || is_chain(op) && inst.result_id.is_some_and(|r| push_chains.contains_key(&r)) {
            continue;
        }
        let users: Vec<u32> = id_operands(&inst.assemble()).into_iter().filter(|w| push_chains.contains_key(w)).collect();
        if let Some(first) = users.iter().min() {
            return Err(Failure::Exit(format!(
                "constante poussee (resultat %{first}) utilisee par l'operation {}, pas seulement lue par OpLoad : non gere",
                op as u32
            )));
        }
    }

    let mut push_chains: HashMap<u32, u32> = HashMap::new();
    let mut converted: HashSet<u32> = HashSet::new();
    let (mut zero, mut size) = (None, None);
    let mut functions = std::mem::take(&mut m.m.functions);
    for function in &mut functions {
        for block in &mut function.blocks {
            let mut out = Vec::with_capacity(block.instructions.len());
            for inst in std::mem::take(&mut block.instructions) {
                let op = opcode(&inst);
                let length = inst.assemble().len();
                if is_chain(op) {
                    let (result_type, result) = (inst.result_type.unwrap_or(0), inst.result_id.unwrap_or(0));
                    let source = operand(&inst, 0).unwrap_or(0);
                    if push_variables.contains(&source) {
                        // A constant index, checked above.
                        push_chains.insert(result, m.constants[&inst.operands.last().and_then(word).unwrap_or(0)]);
                        out.extend(std::iter::repeat_with(nop).take(length));
                        continue;
                    }
                    let key = if inst.operands.len() > 1 { operand(&inst, 1).and_then(|i| m.constants.get(&i)).map(|&c| (source, i64::from(c))) } else { None };
                    if let Some(target) = key.and_then(|k| moved.get(&k)) {
                        let rest = &inst.operands[2..];
                        let new = if rest.is_empty() {
                            if target.uniform {
                                converted.insert(result);
                            }
                            Instruction::new(Op::CopyObject, Some(if target.uniform { target.pointer } else { result_type }), Some(result), vec![Operand::IdRef(target.variable)])
                        } else {
                            let new_type = if target.uniform { uniform_uint_pointer.unwrap_or(0) } else { result_type };
                            let mut operands = vec![Operand::IdRef(target.variable)];
                            operands.extend_from_slice(rest);
                            Instruction::new(op, Some(new_type), Some(result), operands)
                        };
                        let padding = length - new.assemble().len();
                        out.push(new);
                        out.extend(std::iter::repeat_with(nop).take(padding));
                        continue;
                    }
                    if converted.contains(&source) {
                        let mut new = inst;
                        new.result_type = uniform_uint_pointer;
                        out.push(new);
                        continue;
                    }
                }
                if op == Op::ArrayLength && operand(&inst, 0).is_some_and(|s| converted.contains(&s)) {
                    return Err(Failure::Exit("OpArrayLength sur un tampon converti : non gere".into()));
                }
                if op == Op::Load
                    && let Some(&word) = operand(&inst, 0).and_then(|p| push_chains.get(&p))
                {
                    let word = i64::from(word);
                    // `push` is a tuple in the Python.
                    let limit = |i: usize| -> Result<i64, Failure> {
                        push.and_then(|p| p.get(i).copied()).ok_or_else(|| Failure::error("IndexError", "tuple index out of range"))
                    };
                    if push.is_none() || word >= limit(2)? {
                        return Err(Failure::Exit(format!("constante poussee {word} : sens inconnu, non gere")));
                    }
                    let value = if word < limit(1)? {
                        // Binding offsets, 8 bits each.
                        *zero.get_or_insert_with(|| m.constant(0))
                    } else {
                        // Bound size of each buffer, in bytes.
                        *size.get_or_insert_with(|| m.constant(UNIFORM_WORDS * 4))
                    };
                    out.push(Instruction::new(Op::CopyObject, inst.result_type, inst.result_id, vec![Operand::IdRef(value)]));
                    continue;
                }
                out.push(inst);
            }
            block.instructions = out;
        }
    }
    m.m.functions = functions;
    Ok((m.serialize()?, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_words_are_not_ids() {
        // OpExtInst %type %result %set 7 %a %b: the instruction number 7 is a literal.
        assert_eq!(id_operands(&[7 << 16 | 12, 1, 2, 3, 7, 5, 6]), [1, 2, 3, 5, 6]);
        // OpCompositeExtract %type %result %composite 4 5.
        assert_eq!(id_operands(&[6 << 16 | 81, 1, 2, 3, 4, 5]), [1, 2, 3]);
        // OpSwitch %selector %default 1 %a 2 %b.
        assert_eq!(id_operands(&[7 << 16 | 251, 9, 10, 1, 11, 2, 12]), [9, 10, 11, 12]);
        // OpLine: no id at all.
        assert_eq!(id_operands(&[4 << 16 | 8, 1, 2, 3]), Vec::<u32>::new());
        // OpStore keeps every word, the memory operand too.
        assert_eq!(id_operands(&[4 << 16 | 62, 1, 2, 3]), [1, 2, 3]);
    }
}
