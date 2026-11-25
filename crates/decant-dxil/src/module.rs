//! The LLVM 3.7 module model and its bitcode encoding (specification section 2): types,
//! constants, globals, function declarations, metadata and one function body, numbered and
//! written with unabbreviated records.

use std::collections::HashMap;

use crate::Error;
use crate::bitstream::{BitWriter, signed};
use crate::dxop::Attr;

pub type TypeId = u32;
pub type ConstId = u32;
pub type MdId = u32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Void,
    Int(u32),
    Half,
    Float,
    Ptr(TypeId, u32),
    Array(u64, TypeId),
    Struct { name: Option<String>, fields: Vec<TypeId> },
    Func { ret: TypeId, params: Vec<TypeId> },
}

#[derive(Default)]
pub struct Types {
    pub list: Vec<Type>,
    map: HashMap<Type, TypeId>,
}

impl Types {
    pub fn intern(&mut self, t: Type) -> TypeId {
        if let Some(&id) = self.map.get(&t) {
            return id;
        }
        let id = self.list.len() as TypeId;
        self.list.push(t.clone());
        self.map.insert(t, id);
        id
    }
    pub fn void(&mut self) -> TypeId {
        self.intern(Type::Void)
    }
    pub fn int(&mut self, width: u32) -> TypeId {
        self.intern(Type::Int(width))
    }
    pub fn float(&mut self) -> TypeId {
        self.intern(Type::Float)
    }
    pub fn half(&mut self) -> TypeId {
        self.intern(Type::Half)
    }
    pub fn ptr(&mut self, to: TypeId, addrspace: u32) -> TypeId {
        self.intern(Type::Ptr(to, addrspace))
    }
    pub fn named(&mut self, name: &str, fields: Vec<TypeId>) -> TypeId {
        self.intern(Type::Struct { name: Some(name.to_owned()), fields })
    }
    pub fn anon(&mut self, fields: Vec<TypeId>) -> TypeId {
        self.intern(Type::Struct { name: None, fields })
    }
    pub fn func(&mut self, ret: TypeId, params: Vec<TypeId>) -> TypeId {
        self.intern(Type::Func { ret, params })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstKind {
    Undef,
    /// A sign extended integer.
    Int(i64),
    /// The bit pattern of a half, float or double.
    Float(u64),
    Aggregate(Vec<ConstId>),
}

#[derive(Default)]
pub struct Consts {
    pub list: Vec<(TypeId, ConstKind)>,
    map: HashMap<(TypeId, ConstKind), ConstId>,
}

impl Consts {
    pub fn intern(&mut self, ty: TypeId, kind: ConstKind) -> ConstId {
        if let Some(&id) = self.map.get(&(ty, kind.clone())) {
            return id;
        }
        let id = self.list.len() as ConstId;
        self.list.push((ty, kind.clone()));
        self.map.insert((ty, kind), id);
        id
    }
}

/// A value of the module: what an operand refers to. `Inst` counts the values produced in the
/// function body (numbered from 0 in emission order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ref {
    Const(ConstId),
    Global(u32),
    Func(u32),
    Inst(u32),
    /// A phi input that is defined later, named by the IR value id. Resolved before writing.
    Late(u32),
}

/// An operand of an instruction record.
#[derive(Clone, Copy, Debug)]
pub enum LOp {
    /// Written as is.
    Imm(u64),
    /// A relative value id (current id minus the id of the value), unsigned. Backward only.
    Rel(Ref),
    /// A relative value id as a signed VBR (phi inputs, which may be forward).
    SRel(Ref),
}

/// One instruction record of the function body.
#[derive(Clone, Debug)]
pub struct LInst {
    pub code: u32,
    pub ops: Vec<LOp>,
    /// Whether the instruction takes a value id.
    pub has_value: bool,
}

pub struct Global {
    pub name: String,
    /// The type of the value, not of the pointer.
    pub ty: TypeId,
    pub addrspace: u32,
    pub init: Option<ConstId>,
}

pub struct Func {
    pub name: String,
    /// The function type.
    pub ty: TypeId,
    pub attr: Option<Attr>,
    pub proto: bool,
}

/// Metadata node contents. Operands of a node are earlier ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Md {
    Str(String),
    Value(TypeId, Ref),
    Node(Vec<Option<MdId>>),
}

#[derive(Default)]
pub struct Metadata {
    pub list: Vec<Md>,
    map: HashMap<Md, MdId>,
    pub named: Vec<(String, Vec<MdId>)>,
}

impl Metadata {
    pub fn intern(&mut self, md: Md) -> MdId {
        if let Some(&id) = self.map.get(&md) {
            return id;
        }
        let id = self.list.len() as MdId;
        self.list.push(md.clone());
        self.map.insert(md, id);
        id
    }
    pub fn string(&mut self, s: &str) -> MdId {
        self.intern(Md::Str(s.to_owned()))
    }
    pub fn node(&mut self, ops: Vec<Option<MdId>>) -> MdId {
        self.intern(Md::Node(ops))
    }
}

#[derive(Default)]
pub struct Module {
    pub types: Types,
    pub consts: Consts,
    pub globals: Vec<Global>,
    pub funcs: Vec<Func>,
    pub md: Metadata,
    pub body: Vec<LInst>,
    pub block_count: u32,
}

/// Record codes (LLVMBitCodes 3.7).
mod code {
    pub const MODULE_VERSION: u32 = 1;
    pub const MODULE_TRIPLE: u32 = 2;
    pub const MODULE_DATALAYOUT: u32 = 3;
    pub const MODULE_GLOBALVAR: u32 = 7;
    pub const MODULE_FUNCTION: u32 = 8;
    pub const TYPE_NUMENTRY: u32 = 1;
    pub const TYPE_VOID: u32 = 2;
    pub const TYPE_FLOAT: u32 = 3;
    pub const TYPE_DOUBLE: u32 = 4;
    pub const TYPE_INTEGER: u32 = 7;
    pub const TYPE_POINTER: u32 = 8;
    pub const TYPE_HALF: u32 = 10;
    pub const TYPE_ARRAY: u32 = 11;
    pub const TYPE_STRUCT_ANON: u32 = 18;
    pub const TYPE_STRUCT_NAME: u32 = 19;
    pub const TYPE_STRUCT_NAMED: u32 = 20;
    pub const TYPE_FUNCTION: u32 = 21;
    pub const CST_SETTYPE: u32 = 1;
    pub const CST_UNDEF: u32 = 3;
    pub const CST_INTEGER: u32 = 4;
    pub const CST_FLOAT: u32 = 6;
    pub const CST_AGGREGATE: u32 = 7;
    pub const MD_STRING: u32 = 1;
    pub const MD_VALUE: u32 = 2;
    pub const MD_NODE: u32 = 3;
    pub const MD_NAME: u32 = 4;
    pub const MD_KIND: u32 = 6;
    pub const MD_NAMED_NODE: u32 = 10;
    pub const VST_ENTRY: u32 = 1;
    pub const FUNC_DECLAREBLOCKS: u32 = 1;
    pub const PARAMATTR_ENTRY: u32 = 2;
    pub const PARAMATTR_GRP_ENTRY: u32 = 3;
}

/// Block ids (LLVMBitCodes 3.7).
mod block {
    pub const MODULE: u64 = 8;
    pub const PARAMATTR: u64 = 9;
    pub const PARAMATTR_GROUP: u64 = 10;
    pub const CONSTANTS: u64 = 11;
    pub const FUNCTION: u64 = 12;
    pub const VALUE_SYMTAB: u64 = 14;
    pub const METADATA: u64 = 15;
    pub const TYPE_NEW: u64 = 17;
}

pub const TRIPLE: &str = "dxil-ms-dx";
pub const DATALAYOUT: &str = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64";

/// The fixed metadata kinds of the 3.7 `LLVMContext`, written as the reference tool chain does.
const FIXED_KINDS: [&str; 14] = [
    "dbg",
    "tbaa",
    "prof",
    "fpmath",
    "range",
    "tbaa.struct",
    "invariant.load",
    "alias.scope",
    "noalias",
    "nontemporal",
    "llvm.mem.parallel_loop_access",
    "nonnull",
    "dereferenceable",
    "dereferenceable_or_null",
];

fn chars(s: &str) -> Vec<u64> {
    s.bytes().map(u64::from).collect()
}

impl Module {
    fn attr_groups(&self) -> Vec<Attr> {
        let mut used = Vec::new();
        for f in &self.funcs {
            if let Some(a) = f.attr
                && !used.contains(&a)
            {
                used.push(a);
            }
        }
        used
    }

    /// Writes the module as a bitcode stream (without a wrapper). `kinds` selects whether the
    /// fixed metadata kind records are written.
    pub fn write_bitcode(&self, kinds: bool) -> Result<Vec<u8>, Error> {
        let mut w = BitWriter::new();
        w.enter_block(block::MODULE, 3);
        w.record(code::MODULE_VERSION, &[1]);

        let groups = self.attr_groups();
        if !groups.is_empty() {
            w.enter_block(block::PARAMATTR_GROUP, 3);
            for (i, a) in groups.iter().enumerate() {
                let attrs: &[u64] = match a {
                    Attr::ReadNone => &[0, 18, 0, 20],
                    Attr::NoUnwind => &[0, 18],
                    Attr::ReadOnly => &[0, 18, 0, 21],
                    Attr::NoDuplicate => &[0, 18, 0, 12],
                };
                let mut ops = vec![i as u64 + 1, 0xFFFF_FFFF];
                ops.extend_from_slice(attrs);
                w.record(code::PARAMATTR_GRP_ENTRY, &ops);
            }
            w.end_block();
            w.enter_block(block::PARAMATTR, 3);
            for i in 0..groups.len() {
                w.record(code::PARAMATTR_ENTRY, &[i as u64 + 1]);
            }
            w.end_block();
        }

        self.write_types(&mut w);
        w.record(code::MODULE_TRIPLE, &chars(TRIPLE));
        w.record(code::MODULE_DATALAYOUT, &chars(DATALAYOUT));

        let ng = self.globals.len() as u64;
        let nf = self.funcs.len() as u64;
        // Constants are grouped by type, in creation order inside a type.
        let mut order: Vec<ConstId> = (0..self.consts.list.len() as ConstId).collect();
        order.sort_by_key(|&c| self.consts.list[c as usize].0);
        let mut pos = vec![0u64; order.len()];
        for (p, &c) in order.iter().enumerate() {
            pos[c as usize] = p as u64;
        }
        let base_inst = ng + nf + order.len() as u64;
        let abs = |r: Ref| -> Result<u64, Error> {
            Ok(match r {
                Ref::Global(g) => g as u64,
                Ref::Func(f) => ng + f as u64,
                Ref::Const(c) => ng + nf + pos[c as usize],
                Ref::Inst(i) => base_inst + i as u64,
                Ref::Late(v) => return Err(Error::Invalid(format!("value {v} was never defined"))),
            })
        };

        for g in &self.globals {
            // [value type, flags (explicit type, address space), initialiser + 1, linkage (internal), alignment]
            let init = g.init.map_or(0, |c| ng + nf + pos[c as usize] + 1);
            w.record(code::MODULE_GLOBALVAR, &[g.ty as u64, 2 | (g.addrspace as u64) << 2, init, 3, 3, 0, 0, 0, 0, 0, 0, 0]);
        }
        for f in &self.funcs {
            let attr = f.attr.map_or(0, |a| groups.iter().position(|g| *g == a).unwrap() as u64 + 1);
            w.record(code::MODULE_FUNCTION, &[f.ty as u64, 0, f.proto as u64, 0, attr, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        }

        self.write_constants(&mut w, &order, &pos, ng + nf);
        self.write_metadata(&mut w, &abs, kinds)?;

        w.enter_block(block::VALUE_SYMTAB, 4);
        for (i, g) in self.globals.iter().enumerate() {
            let mut ops = vec![i as u64];
            ops.extend(chars(&g.name));
            w.record(code::VST_ENTRY, &ops);
        }
        for (i, f) in self.funcs.iter().enumerate() {
            let mut ops = vec![ng + i as u64];
            ops.extend(chars(&f.name));
            w.record(code::VST_ENTRY, &ops);
        }
        w.end_block();

        // The one function body (the first function).
        w.enter_block(block::FUNCTION, 4);
        w.record(code::FUNC_DECLAREBLOCKS, &[self.block_count as u64]);
        let mut cur = base_inst;
        for inst in &self.body {
            let mut ops = Vec::with_capacity(inst.ops.len());
            for op in &inst.ops {
                ops.push(match *op {
                    LOp::Imm(v) => v,
                    LOp::Rel(r) => {
                        let a = abs(r)?;
                        if a > cur {
                            return Err(Error::Invalid("an operand refers forward outside a phi".into()));
                        }
                        cur - a
                    }
                    LOp::SRel(r) => signed(cur as i64 - abs(r)? as i64),
                });
            }
            w.record(inst.code, &ops);
            if inst.has_value {
                cur += 1;
            }
        }
        w.end_block();
        w.end_block();
        Ok(w.finish())
    }

    fn write_types(&self, w: &mut BitWriter) {
        w.enter_block(block::TYPE_NEW, 4);
        w.record(code::TYPE_NUMENTRY, &[self.types.list.len() as u64]);
        for t in &self.types.list {
            match t {
                Type::Void => w.record(code::TYPE_VOID, &[]),
                Type::Int(n) => w.record(code::TYPE_INTEGER, &[*n as u64]),
                Type::Half => w.record(code::TYPE_HALF, &[]),
                Type::Float => w.record(code::TYPE_FLOAT, &[]),
                Type::Ptr(to, a) => w.record(code::TYPE_POINTER, &[*to as u64, *a as u64]),
                Type::Array(n, e) => w.record(code::TYPE_ARRAY, &[*n, *e as u64]),
                Type::Struct { name, fields } => {
                    let mut ops = vec![0u64];
                    ops.extend(fields.iter().map(|&f| f as u64));
                    match name {
                        Some(n) => {
                            w.record(code::TYPE_STRUCT_NAME, &chars(n));
                            w.record(code::TYPE_STRUCT_NAMED, &ops);
                        }
                        None => w.record(code::TYPE_STRUCT_ANON, &ops),
                    }
                }
                Type::Func { ret, params } => {
                    let mut ops = vec![0u64, *ret as u64];
                    ops.extend(params.iter().map(|&p| p as u64));
                    w.record(code::TYPE_FUNCTION, &ops);
                }
            }
        }
        // `Double` is never created, the constant only documents the code.
        let _ = code::TYPE_DOUBLE;
        w.end_block();
    }

    fn write_constants(&self, w: &mut BitWriter, order: &[ConstId], pos: &[u64], first: u64) {
        if order.is_empty() {
            return;
        }
        w.enter_block(block::CONSTANTS, 4);
        let mut cur_ty = None;
        for &c in order {
            let (ty, kind) = &self.consts.list[c as usize];
            if cur_ty != Some(*ty) {
                w.record(code::CST_SETTYPE, &[*ty as u64]);
                cur_ty = Some(*ty);
            }
            match kind {
                ConstKind::Undef => w.record(code::CST_UNDEF, &[]),
                ConstKind::Int(v) => w.record(code::CST_INTEGER, &[signed(*v)]),
                ConstKind::Float(bits) => w.record(code::CST_FLOAT, &[*bits]),
                ConstKind::Aggregate(elems) => {
                    let ops: Vec<u64> = elems.iter().map(|&e| first + pos[e as usize]).collect();
                    w.record(code::CST_AGGREGATE, &ops);
                }
            }
        }
        w.end_block();
    }

    fn write_metadata(&self, w: &mut BitWriter, abs: &dyn Fn(Ref) -> Result<u64, Error>, kinds: bool) -> Result<(), Error> {
        if !self.md.list.is_empty() {
            w.enter_block(block::METADATA, 3);
            for md in &self.md.list {
                match md {
                    Md::Str(s) => w.record(code::MD_STRING, &chars(s)),
                    Md::Value(ty, r) => w.record(code::MD_VALUE, &[*ty as u64, abs(*r)?]),
                    Md::Node(ops) => {
                        let ops: Vec<u64> = ops.iter().map(|o| o.map_or(0, |id| id as u64 + 1)).collect();
                        w.record(code::MD_NODE, &ops);
                    }
                }
            }
            for (name, ids) in &self.md.named {
                w.record(code::MD_NAME, &chars(name));
                let ops: Vec<u64> = ids.iter().map(|&i| i as u64).collect();
                w.record(code::MD_NAMED_NODE, &ops);
            }
            w.end_block();
        }
        if kinds {
            w.enter_block(block::METADATA, 3);
            for (i, k) in FIXED_KINDS.iter().enumerate() {
                let mut ops = vec![i as u64];
                ops.extend(chars(k));
                w.record(code::MD_KIND, &ops);
            }
            w.end_block();
        }
        Ok(())
    }
}
