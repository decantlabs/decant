//! A SPIR-V importer: reads a compute module (the kind the DecantShader translator emits) and builds
//! the IR of [`crate::ir`], scalarising vectors and composites, taking local variables to SSA with
//! phis, and mapping the descriptor variables to bound resources.
//!
//! Scope of this version: compute shaders, logical addressing, 32 and 64 bit integers, 32 bit
//! floats. What the importer cannot translate is reported as an [`ImportError`] with a short
//! category (such as `op:OpFunctionCall`), so that a corpus can be counted by cause.
//!
//! Resource mapping is synthetic: every descriptor variable element (set, binding, array index)
//! becomes one resource with the next register of its class in the register space of its set.
//! The host that knows the game's layout renames them (see [`Imported::resources`]).

mod inline;
mod io;
mod mem;
mod ops;

use std::collections::{HashMap, HashSet};

use rspirv::dr::{self, Instruction, Operand as SOp};
use rspirv::spirv::{self, Decoration, Dim, Op, StorageClass, Word};

use crate::ir::*;

/// Why a module could not be imported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportError {
    /// A short stable key to count failures by cause, for example `op:OpFunctionCall`.
    pub category: String,
    pub detail: String,
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.category, self.detail)
    }
}

impl std::error::Error for ImportError {}

pub(crate) type R<T> = Result<T, ImportError>;

pub(crate) fn fail<T>(category: impl Into<String>, detail: impl Into<String>) -> R<T> {
    Err(ImportError { category: category.into(), detail: detail.into() })
}

/// A descriptor variable element and the IR resource it became.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedResource {
    pub set: u32,
    pub binding: u32,
    /// The register space of the IR resource (the set, except for unbounded arrays, which get a space of their own).
    pub space: u32,
    /// The index in a descriptor array (0 for a single descriptor).
    pub array_index: u32,
    /// Index in [`Shader::resources`].
    pub index: u32,
    /// The push constant block.
    pub push_constant: bool,
}

/// The result of an import.
#[derive(Clone, Debug)]
pub struct Imported {
    pub shader: Shader,
    pub resources: Vec<ImportedResource>,
    /// The opcodes the module contains, for statistics.
    pub opcodes: Vec<String>,
    /// How the module reaches memory by device address, see [`DeviceMemory`].
    pub device_memory: Option<DeviceMemory>,
    /// The stage inputs and outputs of a vertex or pixel shader.
    pub io: Vec<ImportedIo>,
    /// Set when a heap index is read from the user data mirror: a root constant buffer
    /// `b`[`USER_DATA_REGISTER`] of space [`USER_DATA_SPACE`], [`USER_DATA_DWORDS`] dwords.
    pub user_data: Option<UserData>,
    /// The resources the shader reads from the resource heap at an index it computes (a table read
    /// through the user data mirror), with the class and kind each is read as.
    pub heap_views: Vec<HeapView>,
}

/// One resource read from the resource descriptor heap, and how the shader reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeapView {
    pub class: ResourceClass,
    pub kind: ResourceKind,
    /// The heap index the shader computes.
    pub index: IndexExpr,
}

/// The binding contract of memory read and written through device addresses
/// (`PhysicalStorageBuffer` pointers). DXIL has no device pointers, so every such access becomes a
/// raw buffer access at a 32 bit byte offset into a window of device memory:
///
/// * `window`: the IR resource that is a raw UAV in register space [`DEVICE_MEMORY_SPACE`], register 0
///   (`RWByteAddressBuffer`). The host binds it as a root UAV whose GPU virtual address is the base
///   address of the window.
/// * `base`: the IR resource that is a constant buffer in the same space, register 0, 16 bytes. Its first
///   two dwords are the 64 bit GPU virtual address of the window (low dword first). The host binds it
///   as root constants or a root CBV.
///
/// The offset of an access is `address - base` truncated to 32 bits, so every address the shader
/// uses must lie within 4 GiB above the base and inside the memory the window root descriptor makes
/// visible. The shader does not check this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemory {
    pub window: u32,
    pub base: u32,
}

/// The register space of the device memory window and its base address.
pub const DEVICE_MEMORY_SPACE: u32 = 14;

/// Imports the first entry point of a SPIR-V module: a compute, vertex or fragment shader.
pub fn import(words: &[u32]) -> Result<Imported, ImportError> {
    import_inner(words, None)
}

/// The Direct3D 12 class of a register that a descriptor is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// `t`
    Srv,
    /// `u`
    Uav,
    /// `b`
    Cbv,
    /// `s`
    Sampler,
}

impl RegisterClass {
    fn code(self) -> u8 {
        match self {
            RegisterClass::Srv => 0,
            RegisterClass::Uav => 1,
            RegisterClass::Cbv => 2,
            RegisterClass::Sampler => 3,
        }
    }
}

/// The most (block, local slot) pairs a module may have: beyond it the SSA construction needs gigabytes.
pub const MAX_SSA_CELLS: u64 = 40_000_000;

/// The register space, register and size of the user data mirror, see [`IndexExpr::User`].
pub const USER_DATA_SPACE: u32 = 15;
/// The register of the user data mirror (a constant buffer `b1` of [`USER_DATA_SPACE`]).
pub const USER_DATA_REGISTER: u32 = 1;
/// Dwords of the user data mirror: the 32 user SGPRs of a console stage.
pub const USER_DATA_DWORDS: u32 = 32;

/// A heap index computed by the shader. Every value is a 32 bit integer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexExpr {
    Const(u32),
    /// Dword `K` of the user data mirror: the value console user SGPR `sK` has (a table is the heap
    /// index of its first descriptor, see the README).
    User(u32),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Shl(Box<IndexExpr>, u32),
    Shr(Box<IndexExpr>, u32),
    Mul(Box<IndexExpr>, Box<IndexExpr>),
    And(Box<IndexExpr>, Box<IndexExpr>),
    /// The dword at a byte offset of a buffer the shader has, given by its descriptor `(set, binding, element)`.
    BufferWord { buffer: (u32, u32, u32), offset: Box<IndexExpr> },
}

impl IndexExpr {
    pub fn plus(self, other: IndexExpr) -> IndexExpr {
        IndexExpr::Add(Box::new(self), Box::new(other))
    }
}

/// Where one descriptor of the module lives on the Direct3D 12 side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    /// A register of a register space. A storage buffer bound to a `Cbv` register becomes a constant
    /// buffer of 64 KiB (the declared size cannot be known), one bound to `Srv` or `Uav` a raw buffer.
    Register { class: RegisterClass, space: u32, register: u32 },
    /// An element of the resource descriptor heap at a constant index (Shader Model 6.6 direct
    /// indexing). The class follows from the use: constant buffers, read only buffers and sampled
    /// images are read as such, writable ones as UAVs.
    Heap(u32),
    /// An element of the sampler descriptor heap at a constant index.
    SamplerHeap(u32),
    /// An element of the resource descriptor heap at an index the shader computes.
    HeapAt(IndexExpr),
    /// The same for a buffer whose descriptor is a constant buffer view (a `b` register of the console
    /// binding table): a storage buffer there becomes a constant buffer of 64 KiB, as at a `Cbv` register.
    /// A raw buffer handle of a heap slot that holds a constant buffer view is undefined.
    ConstantHeapAt(IndexExpr),
    /// An element of the sampler descriptor heap at an index the shader computes.
    SamplerHeapAt(IndexExpr),
    /// A buffer that is a root descriptor: user registers `sK` and `sK+1` (mirror dwords `K` and `K+1`) hold
    /// the 64 bit GPU address of its first byte, and the shader reaches it through the device memory
    /// window (`DEVICE_MEMORY_SPACE`). The host puts the address of PC memory in the mirror.
    Address(u32),
}

/// What the words of the push constant block mean, from the `push A B C` line of the translator.
///
/// Words `[0, data_end)` hold values of scalar registers the shader reads as data, words
/// `[data_end, offsets_end)` the byte offsets of the buffers inside their bindings (several to a
/// dword), words `[offsets_end, sizes_end)` the byte size of each buffer, one dword per buffer.
/// A Direct3D view starts at its first byte, so every offset word reads as 0, and a size word is the
/// size the buffer reports. A data word is dword `N` of the user data mirror when `registers` names its
/// console register `sN`, the workgroup id when it is listed in `group_id`, and refused (`push:prefix`) otherwise.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PushLayout {
    pub data_end: u32,
    pub offsets_end: u32,
    pub sizes_end: u32,
    /// The descriptor `(set, binding, array index)` of buffer number `k`, for the size word `offsets_end + k`.
    pub buffers: Vec<Option<(u32, u32, u32)>>,
    /// Data words that are the workgroup id: `(word, axis)`.
    pub group_id: Vec<(u32, u8)>,
    /// Data words that are the value of console user register `sN`: `(word, N)`. They read dword `N` of
    /// the user data mirror.
    pub registers: Vec<(u32, u32)>,
}

/// The places of descriptors and the names of interface elements, from the host that knows the
/// console shader. See [`import_with`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bindings {
    map: HashMap<(u32, u32, u32), Target>,
    /// Placements used instead of `map` when the array is indexed at run time.
    dynamic: HashMap<(u32, u32, u32), Target>,
    /// The push constant block, when its words are known.
    pub push: Option<PushLayout>,
    names: HashMap<(bool, u32, u32), (String, u32)>,
    system: HashMap<(bool, String), u32>,
}

impl Bindings {
    pub fn new() -> Bindings {
        Bindings::default()
    }

    /// Places the descriptor `array_index` of `binding` in `set` (0 for a binding that is not an array).
    pub fn bind(&mut self, set: u32, binding: u32, array_index: u32, target: Target) -> &mut Bindings {
        self.map.insert((set, binding, array_index), target);
        self
    }

    /// Names the signature element of the interface variable that has `Location` `location` and
    /// `Component` `component` (0 when it has none) with the
    /// console semantic and index, so that the title's input layout (`POSITION`, `NORMAL`, ...)
    /// matches a vertex shader. Inputs and outputs are numbered apart. The register of the element is the
    /// location, the register of the console signature. Pixel outputs keep `SV_Target`.
    pub fn name(&mut self, output: bool, location: u32, component: u32, semantic: &str, index: u32) -> &mut Bindings {
        self.names.insert((output, location, component), (semantic.to_owned(), index));
        self
    }

    /// Puts the built in element `semantic` (`SV_Position`, `SV_ClipDistance`, `SV_CullDistance`) at the
    /// register the console signature has it, so that it does not share a register with a named element.
    pub fn system_register(&mut self, output: bool, semantic: &str, register: u32) -> &mut Bindings {
        self.system.insert((output, semantic.to_owned()), register);
        self
    }

    pub(crate) fn system_of(&self, output: bool, semantic: &str) -> Option<u32> {
        self.system.get(&(output, semantic.to_owned())).copied()
    }

    pub(crate) fn name_of(&self, output: bool, location: u32, component: u32) -> Option<&(String, u32)> {
        self.names.get(&(output, location, component))
    }

    /// Where a descriptor was placed.
    pub fn target(&self, set: u32, binding: u32, array_index: u32) -> Option<Target> {
        self.map.get(&(set, binding, array_index)).cloned()
    }

    /// The placed elements of a binding for a run time index: the dynamic placement where there is one.
    pub(crate) fn elements(&self, set: u32, binding: u32) -> Vec<(u32, Target)> {
        let mut out: Vec<(u32, Target)> = self.map.iter().filter(|((s, b, _), _)| *s == set && *b == binding).map(|((_, _, k), t)| (*k, self.dynamic.get(&(set, binding, *k)).unwrap_or(t).clone())).collect();
        out.sort_by_key(|(k, _)| *k);
        out
    }

    /// Places the element another way when its array is indexed at run time: a register cannot be
    /// chosen by an index, a heap element can (the shader selects among the heap indices).
    pub fn bind_dynamic(&mut self, set: u32, binding: u32, array_index: u32, target: Target) -> &mut Bindings {
        self.dynamic.insert((set, binding, array_index), target);
        self
    }
}

/// Imports a module and places its descriptors as `bindings` says, instead of numbering registers
/// in order of first use. Every descriptor the shader uses must be placed (`binding:unmapped`
/// otherwise) and an array indexed at run time needs heap placements (`binding:dynamic-index`). Two variables at
/// one placed register (views of one buffer as `uint`, `ushort`, `uchar`) share one IR resource.
pub fn import_with(words: &[u32], bindings: &Bindings) -> Result<Imported, ImportError> {
    import_inner(words, Some(bindings))
}

fn import_inner(words: &[u32], bindings: Option<&Bindings>) -> Result<Imported, ImportError> {
    let trace = std::env::var_os("DECANT_SHADER_TRACE").is_some();
    let t0 = std::time::Instant::now();
    let mut module = dr::load_words(words).map_err(|e| ImportError { category: "parse".into(), detail: format!("{e:?}") })?;
    if trace {
        eprintln!("  trace: parsed in {:?}", t0.elapsed());
    }
    let entry = module.entry_points.first().and_then(|e| operand_id(e, 1).ok());
    if let Some(entry) = entry {
        inline::inline_calls(&mut module, entry)?;
    }
    if trace {
        eprintln!("  trace: inlined in {:?}, {} blocks", t0.elapsed(), module.functions.iter().map(|f| f.blocks.len()).sum::<usize>());
    }
    let mut imp = Imp::new(&module)?;
    imp.trace = trace;
    imp.bindings = bindings.cloned();
    imp.run()
}

/// The SPIR-V types the importer understands.
#[derive(Clone, Debug)]
pub(crate) enum T {
    Void,
    Bool,
    Int(u32, bool),
    Float(u32),
    Vector(Word, u32),
    Matrix(Word, u32),
    Array(Word, Option<u64>),
    Struct(Vec<Word>),
    Pointer(StorageClass, Word),
    Image { sampled: Word, dim: Dim, arrayed: bool, ms: bool, kind: u32 },
    Sampler,
    SampledImage,
    Function,
}

/// The decorations of an id.
pub(crate) type Decos = Vec<(Decoration, Vec<SOp>)>;

/// A SPIR-V phi still to resolve: IR block, the IR phi values, and the (value id, predecessor) pairs.
pub(crate) type PhiFix = (usize, Vec<usize>, Vec<(Word, usize)>);

/// A reference to an image descriptor (variable and descriptor array index).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DescRef {
    pub var: Word,
    pub arr: u32,
    /// The index in an array of descriptors with more than one element.
    pub idx: Option<Operand>,
    /// The type of the image or sampler.
    pub ty: Word,
}

/// A value of the module being translated.
#[derive(Clone, Debug)]
pub(crate) enum Val {
    /// A numeric value, flattened to scalars (vectors, matrices, arrays and structs in layout order).
    S(Vec<Operand>),
    Ptr(Ptr),
    Image(DescRef),
    Sampler(DescRef),
    Sampled(DescRef, DescRef),
    /// `OpImageTexelPointer`: a texel of an image, the target of an atomic.
    Texel(DescRef, Vec<Operand>),
}

/// A pointer: a variable, a position inside it and the pointee type.
#[derive(Clone, Debug)]
pub(crate) struct Ptr {
    pub storage: StorageClass,
    pub var: Word,
    pub ty: Word,
    /// The descriptor array index, when the variable is an array of one descriptor.
    pub arr: u32,
    /// The index in an array of descriptors with more than one element.
    pub arr_idx: Option<Operand>,
    /// The constant part of the offset: bytes for buffers, scalars for flat storage.
    pub off: u32,
    /// The dynamic part of the offset, same unit.
    pub dynoff: Option<Operand>,
    /// No step was taken yet, so an array type is the descriptor array.
    pub root: bool,
    /// The device address (`I64`) a `PhysicalStorageBuffer` pointer starts at.
    pub addr: Option<Operand>,
}

/// A local variable (Function or Private storage) kept in SSA slots.
pub(crate) struct Local {
    pub base: usize,
}

/// Per variable information.
pub(crate) struct Var {
    pub storage: StorageClass,
    pub ty: Word,
    pub init: Option<Word>,
    pub set: u32,
    pub binding: u32,
}

pub(crate) struct Imp<'m> {
    pub m: &'m dr::Module,
    pub types: HashMap<Word, T>,
    pub decos: HashMap<Word, Decos>,
    pub member_decos: HashMap<(Word, u32), Decos>,
    pub vals: HashMap<Word, Val>,
    pub vars: HashMap<Word, Var>,
    pub val_types: HashMap<Word, Word>,
    // IR under construction.
    pub resources: Vec<Resource>,
    pub imported: Vec<ImportedResource>,
    pub res_map: HashMap<(Word, u32), u32>,
    pub next_reg: HashMap<(u8, u32), u32>,
    pub groupshared: Vec<Groupshared>,
    pub gs_map: HashMap<Word, usize>,
    pub blocks: Vec<Block>,
    pub cur: usize,
    pub next_value: u32,
    // Locals.
    pub locals: HashMap<Word, Local>,
    pub slot_tys: Vec<Ty>,
    pub slot_init: Vec<Operand>,
    pub defs: HashMap<(usize, usize), Operand>,
    pub incomplete: HashMap<usize, Vec<(usize, ValueId)>>,
    pub sealed: HashSet<usize>,
    pub preds: Vec<Vec<usize>>,
    /// Phis of the module still waiting for their incoming values: (ir block, phi index in block, [(value id, spirv pred)]).
    pub phi_fix: Vec<PhiFix>,
    /// For each (spirv block, spirv target) the IR blocks that branch to it.
    pub edges: HashMap<(usize, usize), Vec<usize>>,
    pub label_index: HashMap<Word, usize>,
    pub opcodes: HashSet<String>,
    pub ext_glsl: Option<Word>,
    /// The type of every value operand the importer defined.
    pub op_ty: HashMap<u32, Ty>,
    pub window: Option<u32>,
    pub next_space: u32,
    pub window_base: Option<u32>,
    pub trace: bool,
    pub stage: Option<Stage>,
    pub io: HashMap<Word, Vec<io::IoSlot>>,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub io_notes: Vec<ImportedIo>,
    /// The host's placement of descriptors and names, see [`import_with`].
    pub bindings: Option<Bindings>,
    /// The resource at each placed register: (class code, space, register).
    pub placed: HashMap<(u8, u32, u32), u32>,
    /// The heap index of each resource placed in a heap.
    pub heap_of: HashMap<u32, HeapRef>,
    /// The user data mirror resource, when a heap index reads it.
    pub mirror: Option<u32>,
    pub mirror_words: std::collections::BTreeSet<u32>,
    /// A root descriptor was read through the device memory window.
    pub window_ok: bool,
}

/// The heap index of a resource in the heap: one expression, or the element of an array chosen by
/// the run time index.
#[derive(Clone, Debug)]
pub(crate) enum HeapRef {
    One(IndexExpr),
    Select(Vec<(u32, IndexExpr)>),
}

/// The user data mirror a shader reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserData {
    /// The dwords of the mirror the shader reads, ascending.
    pub dwords: Vec<u32>,
}

/// What a signature element is, for a host that wires vertex buffers and render targets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedIo {
    pub output: bool,
    /// Index in [`Shader::inputs`] or [`Shader::outputs`].
    pub element: u32,
    /// The `Location` decoration, for variables that have one.
    pub location: Option<u32>,
    /// The built in, as its SPIR-V name.
    pub builtin: Option<String>,
}

pub(crate) fn operand_id(i: &Instruction, n: usize) -> R<Word> {
    match i.operands.get(n) {
        Some(SOp::IdRef(w) | SOp::IdScope(w) | SOp::IdMemorySemantics(w)) => Ok(*w),
        other => fail("internal", format!("operand {n} of {:?} is {other:?}, expected an id", i.class.opcode)),
    }
}

pub(crate) fn operand_lit(i: &Instruction, n: usize) -> R<u32> {
    match i.operands.get(n) {
        Some(SOp::LiteralBit32(v) | SOp::LiteralExtInstInteger(v)) => Ok(*v),
        other => fail("internal", format!("operand {n} of {:?} is {other:?}, expected a literal", i.class.opcode)),
    }
}

impl<'m> Imp<'m> {
    fn new(m: &'m dr::Module) -> R<Imp<'m>> {
        let mut imp = Imp {
            m,
            types: HashMap::new(),
            decos: HashMap::new(),
            member_decos: HashMap::new(),
            vals: HashMap::new(),
            vars: HashMap::new(),
            val_types: HashMap::new(),
            resources: Vec::new(),
            imported: Vec::new(),
            res_map: HashMap::new(),
            next_reg: HashMap::new(),
            groupshared: Vec::new(),
            gs_map: HashMap::new(),
            blocks: Vec::new(),
            cur: 0,
            next_value: 0,
            locals: HashMap::new(),
            slot_tys: Vec::new(),
            slot_init: Vec::new(),
            defs: HashMap::new(),
            incomplete: HashMap::new(),
            sealed: HashSet::new(),
            preds: Vec::new(),
            phi_fix: Vec::new(),
            edges: HashMap::new(),
            label_index: HashMap::new(),
            opcodes: HashSet::new(),
            ext_glsl: None,
            op_ty: HashMap::new(),
            window: None,
            next_space: 1000,
            window_base: None,
            trace: false,
            stage: None,
            io: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            io_notes: Vec::new(),
            bindings: None,
            placed: HashMap::new(),
            heap_of: HashMap::new(),
            mirror: None,
            mirror_words: Default::default(),
            window_ok: false,
        };
        for i in &m.ext_inst_imports {
            if let Some(SOp::LiteralString(name)) = i.operands.first()
                && name == "GLSL.std.450"
            {
                imp.ext_glsl = i.result_id;
            }
        }
        for a in &m.annotations {
            match a.class.opcode {
                Op::Decorate => {
                    let (Ok(t), Some(SOp::Decoration(d))) = (operand_id(a, 0), a.operands.get(1)) else { continue };
                    imp.decos.entry(t).or_default().push((*d, a.operands[2..].to_vec()));
                }
                Op::MemberDecorate => {
                    let (Ok(t), Ok(mi), Some(SOp::Decoration(d))) = (operand_id(a, 0), operand_lit(a, 1), a.operands.get(2)) else { continue };
                    imp.member_decos.entry((t, mi)).or_default().push((*d, a.operands[3..].to_vec()));
                }
                _ => {}
            }
        }
        for i in &m.types_global_values {
            imp.global(i)?;
        }
        Ok(imp)
    }

    /// Reads one instruction of the types, constants and variables section.
    fn global(&mut self, i: &Instruction) -> R<()> {
        let id = i.result_id;
        let ty = |_: &Imp, n: usize| -> R<Word> { operand_id(i, n) };
        match i.class.opcode {
            Op::TypeVoid => {
                self.types.insert(id.unwrap(), T::Void);
            }
            Op::TypeBool => {
                self.types.insert(id.unwrap(), T::Bool);
            }
            Op::TypeInt => {
                self.types.insert(id.unwrap(), T::Int(operand_lit(i, 0)?, operand_lit(i, 1)? != 0));
            }
            Op::TypeFloat => {
                self.types.insert(id.unwrap(), T::Float(operand_lit(i, 0)?));
            }
            Op::TypeVector => {
                self.types.insert(id.unwrap(), T::Vector(ty(self, 0)?, operand_lit(i, 1)?));
            }
            Op::TypeMatrix => {
                self.types.insert(id.unwrap(), T::Matrix(ty(self, 0)?, operand_lit(i, 1)?));
            }
            Op::TypeArray => {
                let len = self.const_u32(operand_id(i, 1)?)?;
                self.types.insert(id.unwrap(), T::Array(ty(self, 0)?, Some(len as u64)));
            }
            Op::TypeRuntimeArray => {
                self.types.insert(id.unwrap(), T::Array(ty(self, 0)?, None));
            }
            Op::TypeStruct => {
                let members = (0..i.operands.len()).map(|n| operand_id(i, n)).collect::<R<Vec<_>>>()?;
                self.types.insert(id.unwrap(), T::Struct(members));
            }
            Op::TypePointer => {
                let Some(SOp::StorageClass(sc)) = i.operands.first() else { return fail("internal", "pointer without storage class") };
                self.types.insert(id.unwrap(), T::Pointer(*sc, operand_id(i, 1)?));
            }
            Op::TypeImage => {
                let sampled = operand_id(i, 0)?;
                let Some(SOp::Dim(dim)) = i.operands.get(1) else { return fail("internal", "image without dim") };
                let arrayed = operand_lit(i, 3)? != 0;
                let ms = operand_lit(i, 4)? != 0;
                let kind = operand_lit(i, 5)?;
                self.types.insert(id.unwrap(), T::Image { sampled, dim: *dim, arrayed, ms, kind });
            }
            Op::TypeSampler => {
                self.types.insert(id.unwrap(), T::Sampler);
            }
            Op::TypeSampledImage => {
                self.types.insert(id.unwrap(), T::SampledImage);
            }
            Op::TypeFunction => {
                self.types.insert(id.unwrap(), T::Function);
            }
            Op::Constant | Op::SpecConstant => {
                let t = i.result_type.unwrap();
                let (sc, w) = self.scalar_of(t)?;
                let bits: u64 = match i.operands.first() {
                    Some(SOp::LiteralBit32(v)) => *v as u64,
                    Some(SOp::LiteralBit64(v)) => *v,
                    _ => return fail("internal", "constant without value"),
                };
                let c = match sc {
                    Ty::I32 if w < 32 => {
                        let signed = matches!(self.types.get(&t), Some(T::Int(_, true)));
                        let v = bits as u32 & ((1u32 << w) - 1);
                        Operand::Const(Const::I32(if signed && v >> (w - 1) != 0 { v | !((1u32 << w) - 1) } else { v }))
                    }
                    Ty::I32 => Operand::Const(Const::I32(bits as u32)),
                    Ty::I64 => Operand::Const(Const::I64(bits)),
                    Ty::F32 if w == 16 => Operand::Const(Const::F32(half_to_f32(bits as u16).to_bits())),
                    Ty::F32 => Operand::Const(Const::F32(bits as u32)),
                    _ => return fail("type", format!("a {w} bit constant")),
                };
                self.vals.insert(id.unwrap(), Val::S(vec![c]));
                self.val_types.insert(id.unwrap(), t);
            }
            Op::ConstantTrue | Op::SpecConstantTrue => {
                self.vals.insert(id.unwrap(), Val::S(vec![Operand::Const(Const::Bool(true))]));
                self.val_types.insert(id.unwrap(), i.result_type.unwrap());
            }
            Op::ConstantFalse | Op::SpecConstantFalse => {
                self.vals.insert(id.unwrap(), Val::S(vec![Operand::Const(Const::Bool(false))]));
                self.val_types.insert(id.unwrap(), i.result_type.unwrap());
            }
            Op::ConstantComposite | Op::SpecConstantComposite => {
                let mut flat = Vec::new();
                for n in 0..i.operands.len() {
                    let w = operand_id(i, n)?;
                    match self.vals.get(&w) {
                        Some(Val::S(v)) => flat.extend(v.iter().copied()),
                        _ => return fail("internal", "constant composite member"),
                    }
                }
                self.vals.insert(id.unwrap(), Val::S(flat));
                self.val_types.insert(id.unwrap(), i.result_type.unwrap());
            }
            Op::ConstantNull | Op::Undef => {
                let t = i.result_type.unwrap();
                let tys = self.flat(t)?;
                self.vals.insert(id.unwrap(), Val::S(tys.iter().map(|t| zero(*t)).collect()));
                self.val_types.insert(id.unwrap(), t);
            }
            Op::SpecConstantOp => return fail("op:OpSpecConstantOp", "specialisation constant operations"),
            Op::Variable => {
                let Some(SOp::StorageClass(sc)) = i.operands.first() else { return fail("internal", "variable without storage class") };
                let ptr_ty = i.result_type.unwrap();
                let Some(T::Pointer(_, pointee)) = self.types.get(&ptr_ty).cloned() else { return fail("internal", "variable of a non pointer type") };
                let (set, binding) = (self.deco_lit(id.unwrap(), Decoration::DescriptorSet), self.deco_lit(id.unwrap(), Decoration::Binding));
                let init = operand_id(i, 1).ok();
                self.vars.insert(id.unwrap(), Var { storage: *sc, ty: pointee, init, set: set.unwrap_or(0), binding: binding.unwrap_or(0) });
            }
            _ => {}
        }
        Ok(())
    }

    pub fn deco_lit(&self, id: Word, d: Decoration) -> Option<u32> {
        self.decos.get(&id)?.iter().find(|(x, _)| *x == d).and_then(|(_, ops)| match ops.first() {
            Some(SOp::LiteralBit32(v)) => Some(*v),
            _ => None,
        })
    }

    pub fn has_deco(&self, id: Word, d: Decoration) -> bool {
        self.decos.get(&id).is_some_and(|v| v.iter().any(|(x, _)| *x == d))
    }

    pub fn member_deco_lit(&self, id: Word, member: u32, d: Decoration) -> Option<u32> {
        self.member_decos.get(&(id, member))?.iter().find(|(x, _)| *x == d).and_then(|(_, ops)| match ops.first() {
            Some(SOp::LiteralBit32(v)) => Some(*v),
            _ => None,
        })
    }

    pub fn member_has_deco(&self, id: Word, member: u32, d: Decoration) -> bool {
        self.member_decos.get(&(id, member)).is_some_and(|v| v.iter().any(|(x, _)| *x == d))
    }

    /// The value of an integer constant.
    pub fn const_u32(&self, id: Word) -> R<u32> {
        match self.vals.get(&id) {
            Some(Val::S(v)) => match v.first() {
                Some(Operand::Const(Const::I32(x))) => Ok(*x),
                Some(Operand::Const(Const::I64(x))) => Ok(*x as u32),
                _ => fail("const", "an integer constant was expected"),
            },
            _ => fail("const", format!("id {id} is not a constant")),
        }
    }

    /// The IR type of a scalar SPIR-V type, and its width.
    pub fn scalar_of(&self, t: Word) -> R<(Ty, u32)> {
        match self.types.get(&t) {
            Some(T::Bool) => Ok((Ty::Bool, 1)),
            Some(T::Int(32, _)) => Ok((Ty::I32, 32)),
            Some(T::Int(64, _)) => Ok((Ty::I64, 64)),
            // 8 and 16 bit integers are held as `I32` values that are zero or sign extended.
            Some(T::Int(w @ (8 | 16), _)) => Ok((Ty::I32, *w)),
            Some(T::Int(w, _)) => fail("type:int", format!("{w} bit integers")),
            Some(T::Float(32)) => Ok((Ty::F32, 32)),
            // Half precision values are kept as `F32` holding exactly representable values.
            Some(T::Float(16)) => Ok((Ty::F32, 16)),
            Some(T::Float(w)) => fail("type:float", format!("{w} bit floats")),
            other => fail("internal", format!("type {t} is {other:?}, expected a scalar")),
        }
    }

    /// The scalars of a numeric type in layout order.
    pub fn flat(&self, t: Word) -> R<Vec<Ty>> {
        Ok(match self.types.get(&t) {
            Some(T::Bool | T::Int(..) | T::Float(_)) => vec![self.scalar_of(t)?.0],
            Some(T::Vector(e, n)) => {
                let one = self.flat(*e)?;
                (0..*n).flat_map(|_| one.clone()).collect()
            }
            Some(T::Matrix(c, n)) => {
                let one = self.flat(*c)?;
                (0..*n).flat_map(|_| one.clone()).collect()
            }
            Some(T::Array(e, Some(n))) => {
                let one = self.flat(*e)?;
                (0..*n).flat_map(|_| one.clone()).collect()
            }
            Some(T::Struct(members)) => {
                let mut out = Vec::new();
                for m in members {
                    out.extend(self.flat(*m)?);
                }
                out
            }
            Some(T::Array(_, None)) => return fail("type:runtime-array", "a runtime array as a value"),
            other => return fail("internal", format!("type {t} ({other:?}) has no scalar layout")),
        })
    }

    /// The number of scalars of a type.
    pub fn flat_count(&self, t: Word) -> R<u32> {
        Ok(self.flat(t)?.len() as u32)
    }

    // ---- IR building

    pub fn value(&mut self) -> ValueId {
        let v = ValueId(self.next_value);
        self.next_value += 1;
        v
    }

    /// Emits an instruction with `n` results.
    pub fn emit(&mut self, n: usize, op: Op_) -> Vec<Operand> {
        let dsts: Vec<ValueId> = (0..n).map(|_| self.value()).collect();
        let out = dsts.iter().map(|d| Operand::Value(*d)).collect();
        self.blocks[self.cur].insts.push(Inst { dsts, op });
        out
    }

    pub fn emit1(&mut self, op: Op_) -> Operand {
        self.emit(1, op)[0]
    }

    pub fn emit0(&mut self, op: Op_) {
        self.blocks[self.cur].insts.push(Inst { dsts: Vec::new(), op });
    }

    // ---- CFG and driver

    fn run(&mut self) -> R<Imported> {
        for i in &self.m.capabilities {
            self.opcodes.insert(format!("cap:{:?}", i.operands.first()));
        }
        let Some(entry) = self.m.entry_points.first() else { return fail("stage", "no entry point") };
        let Some(SOp::ExecutionModel(model)) = entry.operands.first() else { return fail("internal", "an entry point without a model") };
        let stage = match model {
            spirv::ExecutionModel::GLCompute => Stage::Compute { threads: [1; 3], wave_size: None },
            spirv::ExecutionModel::Vertex => Stage::Vertex,
            spirv::ExecutionModel::Fragment => Stage::Pixel,
            other => return fail(format!("stage:{other:?}"), "this shader stage is not imported"),
        };
        self.stage = Some(stage.clone());
        let entry_fn = operand_id(entry, 1)?;
        let mut threads = [1u32; 3];
        for em in &self.m.execution_modes {
            if operand_id(em, 0).ok() == Some(entry_fn)
                && let Some(SOp::ExecutionMode(spirv::ExecutionMode::LocalSize)) = em.operands.get(1)
            {
                for (k, t) in threads.iter_mut().enumerate() {
                    *t = operand_lit(em, 2 + k)?;
                }
            }
        }
        let func = self.m.functions.iter().find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(entry_fn)).ok_or_else(|| ImportError { category: "internal".into(), detail: "entry function missing".into() })?;
        // Blocks, in order, with their terminators.
        for (n, b) in func.blocks.iter().enumerate() {
            self.label_index.insert(b.label.as_ref().and_then(|l| l.result_id).unwrap_or(0), n);
        }
        let nsp = func.blocks.len();
        self.blocks = (0..nsp).map(|_| Block { phis: Vec::new(), insts: Vec::new(), term: Terminator::Ret }).collect();
        // Terminators and the extra blocks of switches.
        let mut term_plan: Vec<TermPlan> = Vec::new();
        for (n, b) in func.blocks.iter().enumerate() {
            let t = b.instructions.last().ok_or_else(|| ImportError { category: "internal".into(), detail: "empty block".into() })?;
            term_plan.push(match t.class.opcode {
                Op::Branch => TermPlan::Br(self.label(operand_id(t, 0)?)?),
                Op::BranchConditional => TermPlan::Cond(operand_id(t, 0)?, self.label(operand_id(t, 1)?)?, self.label(operand_id(t, 2)?)?),
                Op::Switch => {
                    let sel = operand_id(t, 0)?;
                    let default = self.label(operand_id(t, 1)?)?;
                    let mut cases = Vec::new();
                    let mut k = 2;
                    while k + 1 < t.operands.len() {
                        let lit = match &t.operands[k] {
                            SOp::LiteralBit32(v) => *v as u64,
                            SOp::LiteralBit64(v) => *v,
                            other => return fail("internal", format!("switch literal {other:?}")),
                        };
                        cases.push((lit, self.label(operand_id(t, k + 1)?)?));
                        k += 2;
                    }
                    let mut chain = vec![n];
                    for _ in 1..cases.len() {
                        self.blocks.push(Block { phis: Vec::new(), insts: Vec::new(), term: Terminator::Ret });
                        chain.push(self.blocks.len() - 1);
                    }
                    TermPlan::Switch { sel, default, cases, chain }
                }
                Op::Return | Op::Unreachable => TermPlan::Ret,
                Op::Kill | Op::TerminateInvocation => {
                    if !matches!(self.stage, Some(Stage::Pixel)) {
                        return fail("op:OpKill", "a discard outside a pixel shader");
                    }
                    TermPlan::Kill
                }
                other => return fail("internal", format!("block ends with {other:?}")),
            });
        }
        // IR edges and predecessors.
        self.preds = vec![Vec::new(); self.blocks.len()];
        let mut ir_succ: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for (n, plan) in term_plan.iter().enumerate() {
            let mut add = |from: usize, to: usize, sp_from: usize, sp_to: usize, this: &mut Imp| {
                if !ir_succ[from].contains(&to) {
                    ir_succ[from].push(to);
                    this.preds[to].push(from);
                }
                this.edges.entry((sp_from, sp_to)).or_default().push(from);
            };
            match plan {
                TermPlan::Br(t) => add(n, *t, n, *t, self),
                TermPlan::Cond(_, t, f) => {
                    add(n, *t, n, *t, self);
                    add(n, *f, n, *f, self);
                }
                TermPlan::Switch { default, cases, chain, .. } => {
                    for (k, (_, t)) in cases.iter().enumerate() {
                        add(chain[k], *t, n, *t, self);
                        if k + 1 < chain.len() {
                            add(chain[k], chain[k + 1], n, usize::MAX, self);
                        }
                    }
                    let last = *chain.last().unwrap();
                    add(last, *default, n, *default, self);
                }
                TermPlan::Ret | TermPlan::Kill => {}
            }
        }
        for v in self.edges.values_mut() {
            v.dedup();
        }
        // Local variables and workgroup memory.
        self.setup_variables(func)?;
        self.setup_io()?;
        self.register_vars();
        if self.trace {
            eprintln!("  trace: variables set up, {} IR blocks, {} slots, {} local variables, {} workgroup words", self.blocks.len(), self.slot_tys.len(), self.locals.len(), self.groupshared.iter().map(|g| g.len as usize).sum::<usize>());
        }
        // The SSA construction keeps one definition per block and local slot, so memory grows with their
        // product: a module that inlines into tens of thousands of blocks with thousands of slots needs gigabytes.
        let (blocks, slots) = (self.blocks.len() as u64, self.slot_tys.len() as u64);
        if blocks * slots > MAX_SSA_CELLS {
            return fail("limit:ssa-size", format!("{blocks} blocks times {slots} local slots is more than {MAX_SSA_CELLS}"));
        }
        // Reverse postorder of the IR graph.
        let mut order = Vec::new();
        let mut seen = vec![false; self.blocks.len()];
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some((b, k)) = stack.pop() {
            if k < ir_succ[b].len() {
                stack.push((b, k + 1));
                let s = ir_succ[b][k];
                if !seen[s] {
                    seen[s] = true;
                    stack.push((s, 0));
                }
            } else {
                order.push(b);
            }
        }
        order.reverse();
        self.sealed.insert(0);
        let mut processed: HashSet<usize> = HashSet::new();
        self.seal_ready(&processed, &ir_succ, &[0]);
        for &b in &order {
            if b >= nsp {
                continue; // a chain block, translated with its switch
            }
            self.cur = b;
            if self.trace && processed.len().is_multiple_of(2000) {
                eprintln!("  trace: {} blocks done, {} values, {} defs", processed.len(), self.next_value, self.defs.len());
            }
            let blk = &func.blocks[b];
            self.translate_block(b, blk, &term_plan[b])?;
            processed.insert(b);
            let mut just = vec![b];
            if let TermPlan::Switch { chain, .. } = &term_plan[b] {
                processed.extend(chain.iter().copied());
                for c in chain {
                    self.sealed.insert(*c);
                }
                just = chain.clone();
            }
            self.seal_ready(&processed, &ir_succ, &just);
        }
        if self.trace {
            eprintln!("  trace: blocks translated, {} values", self.next_value);
        }
        // Unreachable blocks keep a `ret`, they are dropped by the normalisation.
        self.resolve_phis(func)?;
        let stage = match stage {
            Stage::Compute { .. } => Stage::Compute { threads, wave_size: None },
            other => other,
        };
        let shader = Shader {
            stage,
            resources: std::mem::take(&mut self.resources),
            groupshared: std::mem::take(&mut self.groupshared),
            blocks: std::mem::take(&mut self.blocks),
            inputs: std::mem::take(&mut self.inputs),
            outputs: std::mem::take(&mut self.outputs),
        };
        let mut opcodes: Vec<String> = self.opcodes.iter().cloned().collect();
        opcodes.sort();
        let device_memory = match (self.window, self.window_base) {
            (Some(window), Some(base)) => Some(DeviceMemory { window, base }),
            (Some(window), None) => Some(DeviceMemory { window, base: u32::MAX }),
            _ => None,
        };
        let user_data = self.mirror.map(|_| UserData { dwords: self.mirror_words.iter().copied().collect() });
        let mut heap_views = Vec::new();
        let mut heaped: Vec<_> = self.heap_of.iter().collect();
        heaped.sort_by_key(|(index, _)| **index);
        for (index, heap) in heaped {
            let Some(r) = shader.resources.get(*index as usize) else { continue };
            if r.class == ResourceClass::Sampler {
                continue;
            }
            let exprs: Vec<&IndexExpr> = match heap {
                HeapRef::One(e) => vec![e],
                HeapRef::Select(picks) => picks.iter().map(|(_, e)| e).collect(),
            };
            heap_views.extend(exprs.into_iter().map(|e| HeapView { class: r.class, kind: r.kind, index: e.clone() }));
        }
        Ok(Imported { shader, resources: std::mem::take(&mut self.imported), opcodes, device_memory, io: std::mem::take(&mut self.io_notes), user_data, heap_views })
    }

    fn label(&self, id: Word) -> R<usize> {
        self.label_index.get(&id).copied().ok_or_else(|| ImportError { category: "internal".into(), detail: format!("label {id}") })
    }

    /// Seals the successors of the blocks `just` whose predecessors are all processed.
    fn seal_ready(&mut self, processed: &HashSet<usize>, succ: &[Vec<usize>], just: &[usize]) {
        let mut cands: Vec<usize> = just.iter().flat_map(|b| succ[*b].iter().copied()).collect();
        cands.sort_unstable();
        cands.dedup();
        for s in cands {
            if !self.sealed.contains(&s) && self.preds[s].iter().all(|p| processed.contains(p)) {
                self.seal(s);
            }
        }
    }

    // ---- local variables, SSA construction (Braun et al.)

    fn setup_variables(&mut self, func: &dr::Function) -> R<()> {
        // Private and Function variables, then workgroup variables.
        let mut all: Vec<(Word, Word, StorageClass, Option<Word>)> = Vec::new();
        let mut gs: Vec<Word> = Vec::new();
        let ids: Vec<Word> = self.vars.keys().copied().collect();
        for id in ids {
            let v = &self.vars[&id];
            match v.storage {
                StorageClass::Private | StorageClass::Function => all.push((id, v.ty, v.storage, v.init)),
                StorageClass::Workgroup => gs.push(id),
                _ => {}
            }
        }
        for b in &func.blocks {
            for i in &b.instructions {
                if i.class.opcode == Op::Variable
                    && let (Some(id), Some(pt)) = (i.result_id, i.result_type)
                    && let Some(T::Pointer(sc, pointee)) = self.types.get(&pt).cloned()
                {
                    all.push((id, pointee, sc, operand_id(i, 1).ok()));
                    self.vars.insert(id, Var { storage: sc, ty: pointee, init: operand_id(i, 1).ok(), set: 0, binding: 0 });
                }
            }
        }
        all.sort_by_key(|a| a.0);
        for (id, ty, _sc, init) in all {
            let tys = self.flat(ty)?;
            let base = self.slot_tys.len();
            let init_vals: Vec<Operand> = match init.and_then(|w| self.vals.get(&w)) {
                Some(Val::S(v)) if v.len() == tys.len() => v.clone(),
                _ => tys.iter().map(|t| zero(*t)).collect(),
            };
            for (t, v) in tys.iter().zip(init_vals) {
                self.slot_tys.push(*t);
                self.slot_init.push(v);
            }
            self.locals.insert(id, Local { base });
        }
        gs.sort_unstable();
        for id in gs {
            let ty = self.vars[&id].ty;
            let n = self.flat_count(ty)?;
            self.gs_map.insert(id, self.groupshared.len());
            self.groupshared.push(Groupshared { len: n.max(1), ty: Ty::I32 });
        }
        Ok(())
    }

    pub fn read_slot(&mut self, slot: usize, block: usize) -> Operand {
        if let Some(v) = self.defs.get(&(slot, block)) {
            return *v;
        }
        let v = if !self.sealed.contains(&block) {
            let (phi, op) = self.new_phi(slot, block);
            self.incomplete.entry(block).or_default().push((slot, phi));
            op
        } else if self.preds[block].is_empty() {
            self.slot_init[slot]
        } else if self.preds[block].len() == 1 {
            let p = self.preds[block][0];
            self.read_slot(slot, p)
        } else {
            let (phi, op) = self.new_phi(slot, block);
            self.defs.insert((slot, block), op);
            self.fill_phi(slot, block, phi);
            op
        };
        self.defs.insert((slot, block), v);
        v
    }

    fn new_phi(&mut self, slot: usize, block: usize) -> (ValueId, Operand) {
        let dst = self.value();
        let ty = self.slot_tys[slot];
        self.blocks[block].phis.push(Phi { dst, ty, incoming: Vec::new() });
        let op = Operand::Value(dst);
        self.defs.insert((slot, block), op);
        (dst, op)
    }

    fn fill_phi(&mut self, slot: usize, block: usize, phi: ValueId) {
        let preds = self.preds[block].clone();
        let mut incoming = Vec::new();
        for p in preds {
            let v = self.read_slot(slot, p);
            incoming.push((BlockId(p as u32), v));
        }
        if let Some(ph) = self.blocks[block].phis.iter_mut().find(|p| p.dst == phi) {
            ph.incoming = incoming;
        }
    }

    fn seal(&mut self, block: usize) {
        self.sealed.insert(block);
        if let Some(list) = self.incomplete.remove(&block) {
            for (slot, phi) in list {
                self.fill_phi(slot, block, phi);
            }
        }
    }

    /// Resolves the incoming values of the SPIR-V phis once every value is known.
    fn resolve_phis(&mut self, _func: &dr::Function) -> R<()> {
        let fixes = std::mem::take(&mut self.phi_fix);
        for (block, phi_ids, incoming) in fixes {
            // phi_ids: the IR phi dsts (as indices into the block's phis by dst value) for each scalar.
            let mut by_pred: Vec<(usize, Vec<Operand>)> = Vec::new();
            for (val_id, pred_sp) in &incoming {
                let Some(Val::S(v)) = self.vals.get(val_id) else { return fail("internal", format!("phi input {val_id} is not a value")) };
                let from_blocks = self.edges.get(&(*pred_sp, self.sp_of(block))).cloned().unwrap_or_default();
                for fb in from_blocks {
                    if let Some(e) = by_pred.iter_mut().find(|(b, _)| *b == fb) {
                        e.1 = v.clone();
                    } else {
                        by_pred.push((fb, v.clone()));
                    }
                }
            }
            for (k, dst) in phi_ids.iter().enumerate() {
                let incoming: Vec<(BlockId, Operand)> = by_pred.iter().map(|(b, v)| (BlockId(*b as u32), v[k])).collect();
                if let Some(ph) = self.blocks[block].phis.iter_mut().find(|p| p.dst.0 as usize == *dst) {
                    ph.incoming = incoming;
                }
            }
        }
        Ok(())
    }

    /// The SPIR-V block index of an IR block (they are the same below the block count).
    fn sp_of(&self, block: usize) -> usize {
        block
    }

    // ---- block translation

    fn translate_block(&mut self, b: usize, blk: &dr::Block, plan: &TermPlan) -> R<()> {
        let n = blk.instructions.len();
        for i in &blk.instructions[..n.saturating_sub(1)] {
            self.opcodes.insert(format!("{:?}", i.class.opcode));
            self.instruction(i, b)?;
        }
        if let Some(t) = blk.instructions.last() {
            self.opcodes.insert(format!("{:?}", t.class.opcode));
        }
        match plan {
            TermPlan::Br(t) => self.blocks[b].term = Terminator::Br(BlockId(*t as u32)),
            TermPlan::Cond(c, t, f) => {
                let cond = self.scalar(*c)?;
                self.blocks[b].term = Terminator::CondBr(cond, BlockId(*t as u32), BlockId(*f as u32));
            }
            TermPlan::Ret => self.blocks[b].term = Terminator::Ret,
            TermPlan::Kill => {
                self.emit0(Op_::Discard(Operand::Const(Const::Bool(true))));
                self.blocks[b].term = Terminator::Ret;
            }
            TermPlan::Switch { sel, default, cases, chain } => {
                let selv = self.scalar(*sel)?;
                let sel_ty = self.val_ty_scalar(*sel)?;
                for (k, (lit, target)) in cases.iter().enumerate() {
                    self.cur = chain[k];
                    let lit_op = if sel_ty == Ty::I64 { Operand::Const(Const::I64(*lit)) } else { Operand::Const(Const::I32(*lit as u32)) };
                    let c = self.emit1(Op_::ICmp(IPred::Eq, selv, lit_op));
                    let els = if k + 1 < chain.len() { chain[k + 1] } else { *default };
                    self.blocks[chain[k]].term = Terminator::CondBr(c, BlockId(*target as u32), BlockId(els as u32));
                }
                if cases.is_empty() {
                    self.blocks[b].term = Terminator::Br(BlockId(*default as u32));
                }
                self.cur = b;
            }
        }
        Ok(())
    }

    /// The operand of a scalar SPIR-V value.
    pub fn scalar(&self, id: Word) -> R<Operand> {
        match self.vals.get(&id) {
            Some(Val::S(v)) if v.len() == 1 => Ok(v[0]),
            other => fail("internal", format!("id {id} is {other:?}, expected a scalar")),
        }
    }

    pub fn val_ty_scalar(&self, id: Word) -> R<Ty> {
        let t = *self.val_types.get(&id).ok_or_else(|| ImportError { category: "internal".into(), detail: format!("type of {id}") })?;
        Ok(self.scalar_of(t)?.0)
    }

    pub fn scalars(&self, id: Word) -> R<Vec<Operand>> {
        match self.vals.get(&id) {
            Some(Val::S(v)) => Ok(v.clone()),
            other => fail("internal", format!("id {id} is {other:?}, expected a value")),
        }
    }
}

// The IR operation type under a short name (the SPIR-V `Op` is also imported).
pub(crate) type Op_ = crate::ir::Op;

pub(crate) fn zero(t: Ty) -> Operand {
    Operand::Const(match t {
        Ty::Bool => Const::Bool(false),
        Ty::I32 => Const::I32(0),
        Ty::I64 => Const::I64(0),
        Ty::F16 => Const::F16(0),
        Ty::F32 => Const::F32(0),
    })
}

pub(crate) enum TermPlan {
    Br(usize),
    Cond(Word, usize, usize),
    Switch { sel: Word, default: usize, cases: Vec<(u64, usize)>, chain: Vec<usize> },
    Ret,
    Kill,
}

/// The value of a half precision float.
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}
