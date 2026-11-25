//! The intermediate representation.
//!
//! It is a register level IR, not SSA: the HLSL compiler builds SSA itself, so the translator
//! only has to say what each instruction does to the registers of ONE lane. A program is a
//! tree of structured statements over three kinds of state:
//!
//! * `S(n)`: a scalar register, 32 bits, the same value for every lane of the wave.
//! * `V(n)`: a vector register, 32 bits, the value of this lane.
//! * lane bits (`Exec`, `Vcc`, `Mask(n)`): this lane's bit of EXEC, of VCC, or of the 64 bit
//!   mask a scalar register pair holds. `crate::exec` decides which pairs are masks.
//!
//! Every value is a 32 bit pattern (`Ty::U32`), a lane bit or condition (`Ty::Bool`) or a 64 bit
//! pattern (`Ty::U64`). Floats are patterns too: a float operation reinterprets its operands,
//! which costs nothing after compilation and keeps one register one variable.
//!
//! The core operation set below is small and fixed. Everything else is a [`Helper`]: an HLSL
//! function a lifter file defines next to the rule that uses it, so that adding an instruction
//! touches one file.
//!
//! The passes, in order: `lift` writes [`Stmt`] lists per native block with raw memory
//! operations ([`Load`], [`ImageOp`], [`BufferOp`] naming registers), `structure` builds the
//! [`Node`] tree, `resource` replaces every raw descriptor use by a [`ResId`], the `stage`
//! module adds the prologue and resolves [`Export`] and [`Input`], `hlsl` prints.

mod print;

use crate::api::Placement;

/// The type of an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Bool,
    U32,
    U64,
}

/// A piece of machine state of one lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    S(u8),
    V(u8),
    Ttmp(u8),
    M0,
    /// The scalar condition code.
    Scc,
    /// This lane's bit of VCC.
    Vcc,
    /// This lane's bit of EXEC.
    Exec,
    /// This lane's bit of the mask in the scalar register pair that starts at `n`.
    Mask(u8),
    /// A 32 bit temporary of the translator.
    Tmp(u16),
    /// A boolean temporary of the translator.
    Flag(u16),
    /// One lane of a vector register that the program only uses to park a scalar value
    /// (`v_writelane` and `v_readlane` with a constant lane). 32 bits. A parked mask is 0 or 1.
    Spill {
        reg: u8,
        lane: u8,
    },
}

impl Var {
    pub fn ty(self) -> Ty {
        match self {
            Var::Scc | Var::Vcc | Var::Exec | Var::Mask(_) | Var::Flag(_) => Ty::Bool,
            _ => Ty::U32,
        }
    }
}

/// Operations with one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Un {
    /// Bitwise not of a 32 bit value.
    Not,
    /// Not of a boolean.
    LNot,
    /// Two's complement negation.
    Neg,
    FNeg,
    FAbs,
    /// Boolean to 1 or 0.
    BoolToU32,
    /// Nonzero test.
    U32ToBool,
    /// Zero extension to 64 bits and truncation back.
    U32ToU64,
    U64ToU32,
}

/// Operations with two operands. The float ones take and give 32 bit patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Shifts take the count modulo 32, as the machine does.
    Shl,
    LShr,
    AShr,
    FAdd,
    FSub,
    FMul,
    Eq,
    Ne,
    ULt,
    ULe,
    SLt,
    SLe,
    /// Ordered float compares: false when an operand is not a number.
    FLt,
    FLe,
    FEq,
    /// Unordered float inequality: true when an operand is not a number.
    FNe,
    LAnd,
    LOr,
    LXor,
}

impl Bin {
    pub fn result(self) -> Ty {
        match self {
            Bin::Eq
            | Bin::Ne
            | Bin::ULt
            | Bin::ULe
            | Bin::SLt
            | Bin::SLe
            | Bin::FLt
            | Bin::FLe
            | Bin::FEq
            | Bin::FNe
            | Bin::LAnd
            | Bin::LOr
            | Bin::LXor => Ty::Bool,
            _ => Ty::U32,
        }
    }
}

/// An HLSL function that a lifter rule calls. `text` is the whole definition. The printer emits
/// each helper a shader uses once, after the helpers it depends on.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Helper {
    /// The function name. It must be unique in the crate and start with `decant_`.
    pub name: &'static str,
    pub result: Ty,
    pub text: &'static str,
    pub needs: &'static [&'static Helper],
}

/// An input of the stage that is not a register at entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// Component `chan` of the interpolated pixel input attribute `attr` (the hardware number).
    Attr {
        attr: u8,
        chan: u8,
    },
    /// Component `chan` of the vertex input element in signature register `register`.
    VertexElement {
        register: u8,
        chan: u8,
    },
    VertexId,
    InstanceId,
    /// Component `chan` of the pixel position.
    FragCoord {
        chan: u8,
    },
    FrontFace,
    /// Whether this invocation is a helper pixel.
    HelperLane,
    /// The number of this lane in its console wave, 0 to 31 or 0 to 63. The stage says what
    /// it is (`Interface::lane_index`): the thread index in its wave for a compute program,
    /// the thread number of a mesh program.
    LaneIndex,
    /// Linear index of this invocation in a compute or mesh workgroup.
    GroupIndex,
    /// Component `axis` of the thread id in the group, of the group id.
    LocalThreadId {
        axis: u8,
    },
    GroupId {
        axis: u8,
    },
}

/// An index into [`Function::resources`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResId(pub u32);

/// A scalar memory read before resource tracing: `words` are not known yet to be a descriptor
/// or data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Load {
    /// `s_load_dword` reads at an address (the pair starting at `base`), `s_buffer_load_dword`
    /// reads the buffer whose descriptor is in the four registers starting at `base`.
    pub buffer: bool,
    pub base: u8,
    /// Byte offset of the dword read.
    pub offset: Box<Expr>,
    pub pc: u32,
}

/// One node intersection in the native BVH format, before backend lowering.
/// Address words hold the node pointer, extent, origin, direction and inverse direction.
/// A16 packs the six direction components into three words.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BvhIntersect {
    pub wide: bool,
    pub descriptor: Vec<Expr>,
    pub address: Vec<Expr>,
    pub encoding: crate::decode::Mimg,
    pub pc: u32,
}

/// An expression. It has no side effect.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// One of the four native result words of a node intersection.
    BvhWord(Box<BvhIntersect>, u8),
    U32(u32),
    U64(u64),
    Bool(bool),
    Var(Var),
    Un(Un, Box<Expr>),
    Bin(Bin, Box<Expr>, Box<Expr>),
    /// `cond ? a : b`, both sides evaluated.
    Select(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(&'static Helper, Vec<Expr>),
    Input(Input),
    /// The value user data register `K` has at entry (what the root signature put there).
    UserData(u8),
    /// A raw scalar memory read. Gone after the resource pass.
    Load(Load),
    /// A raw read of one dword at the 64 bit address a lane holds in vector registers `base`
    /// and `base + 1`, plus `offset` (the global loads). Gone after the resource pass, which
    /// knows where the address came from (so far: the program's own code).
    VectorLoad {
        base: u8,
        offset: Box<Expr>,
        pc: u32,
    },
    /// The dword at byte `offset` of a buffer resource.
    BufferWord {
        res: ResId,
        offset: Box<Expr>,
    },
    /// Word `word` of the mirror (see `MirrorAbi`), as placement code reads it.
    Mirror(u8),
    /// The byte address of a place in the program's own code, counted from the first word of the
    /// code (the signature included). `s_getpc_b64` gives the address of the instruction after
    /// it. Programs add a constant to it and read tables the compiler placed behind the code.
    /// As a number it is this offset: the real address is never needed, because every use is a
    /// far jump (folded away in `cfg`) or a read of the code ([`Expr::CodeWord`]).
    CodeAddress(u32),
    /// The dword of the program's own code at byte address `E` (counted like `CodeAddress`). The
    /// resource pass makes it from a scalar load whose base is a code address. The printer keeps
    /// the code words as `static const uint decant_code[N]` and folds a constant address.
    CodeWord(Box<Expr>),
}

impl Expr {
    pub fn var(v: Var) -> Expr {
        Expr::Var(v)
    }

    pub fn un(op: Un, a: Expr) -> Expr {
        Expr::Un(op, Box::new(a))
    }

    pub fn bin(op: Bin, a: Expr, b: Expr) -> Expr {
        Expr::Bin(op, Box::new(a), Box::new(b))
    }

    pub fn select(c: Expr, a: Expr, b: Expr) -> Expr {
        Expr::Select(Box::new(c), Box::new(a), Box::new(b))
    }

    pub fn ty(&self) -> Ty {
        match self {
            Expr::BvhWord(..)
            | Expr::U32(_)
            | Expr::UserData(_)
            | Expr::Load(_)
            | Expr::BufferWord { .. }
            | Expr::Mirror(_)
            | Expr::CodeAddress(_)
            | Expr::CodeWord(_)
            | Expr::VectorLoad { .. } => Ty::U32,
            Expr::U64(_) => Ty::U64,
            Expr::Bool(_) => Ty::Bool,
            Expr::Var(v) => v.ty(),
            Expr::Un(op, a) => match op {
                Un::LNot | Un::U32ToBool => Ty::Bool,
                Un::U32ToU64 => Ty::U64,
                Un::U64ToU32 | Un::BoolToU32 => Ty::U32,
                Un::Not | Un::Neg | Un::FNeg | Un::FAbs => a.ty(),
            },
            Expr::Bin(op, ..) => op.result(),
            Expr::Select(_, a, _) => a.ty(),
            Expr::Call(h, _) => h.result,
            Expr::Input(i) => match i {
                Input::FrontFace | Input::HelperLane => Ty::Bool,
                _ => Ty::U32,
            },
        }
    }
}

/// When a statement takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Guard {
    /// Always: scalar instructions.
    Always,
    /// Only when this lane is active: vector instructions.
    Exec,
}

/// What an image instruction does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageKind {
    /// Filtered sample. The flags follow the instruction name. `clamp` (the `_cl` forms) adds one
    /// address operand after the coordinates: the smallest level of detail allowed.
    Sample {
        lod: SampleLod,
        compare: bool,
        offset: bool,
        gather: bool,
        clamp: bool,
    },
    /// Texel fetch, with a mip level operand when `mip`.
    Fetch {
        mip: bool,
    },
    Store {
        mip: bool,
    },
    /// Size and mip count.
    Info,
    /// The level of detail a sample would use.
    QueryLod,
    /// A read, modify and write of one texel. With `returns` the old value goes to the data
    /// registers.
    Atomic {
        op: ImageAtomic,
        returns: bool,
    },
}

/// What an atomic image instruction does to the texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageAtomic {
    Swap,
    CmpSwap,
    Add,
    Sub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    Inc,
    Dec,
}

/// Where a sample takes its level of detail from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleLod {
    /// Screen space derivatives of the coordinates.
    Implicit,
    /// Derivatives plus a bias operand.
    Bias,
    /// An explicit level operand.
    Level,
    /// Level zero.
    Zero,
    /// Explicit derivative operands.
    Grad,
}

/// The descriptor an operation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Desc {
    /// The descriptor is in the scalar registers starting at this one (before the resource pass).
    Regs(u8),
    Res(ResId),
}

/// An image instruction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageOp {
    pub kind: ImageKind,
    /// The dimension code of the instruction, see `decode::Mimg::dim`.
    pub dim: u8,
    pub dmask: u8,
    /// 16 bit data, two components per register.
    pub d16: bool,
    /// One more result register after the data: the residency status.
    pub tfe: bool,
    /// Coordinates in texels instead of 0 to 1.
    pub unnormalized: bool,
    /// The first of the result registers (one per bit of `dmask`), or of the stored data.
    pub data: u8,
    /// The address operands in machine order (offset, bias, compare value, derivatives,
    /// coordinates, level: whatever the instruction takes), each a 32 bit pattern. The lifter
    /// unpacks 16 bit address halves, so there is always one entry per component.
    pub address: Vec<Expr>,
    pub image: Desc,
    pub sampler: Option<Desc>,
    pub pc: u32,
}

impl ImageOp {
    /// Registers of `data` the instruction reads: the stored components, or the operand dwords
    /// of an atomic.
    pub fn data_reads(&self) -> std::ops::Range<u8> {
        let n = match self.kind {
            ImageKind::Store { .. } if self.d16 => (self.dmask.count_ones() as u8).div_ceil(2),
            ImageKind::Store { .. } => self.dmask.count_ones() as u8,
            // An atomic names its operand dwords in the mask: one for 32 bits, two for 64 bits
            // or a 32 bit compare and swap, four for a 64 bit compare and swap.
            ImageKind::Atomic { .. } => self.dmask.count_ones() as u8,
            _ => 0,
        };
        self.data..self.data + n
    }

    /// Registers of `data` the instruction writes.
    pub fn data_writes(&self) -> std::ops::Range<u8> {
        let comps = match self.kind {
            ImageKind::Store { .. } => 0,
            ImageKind::Atomic { returns: false, .. } => 0,
            // The old value: as wide as the operand, half of the mask for a compare and swap.
            ImageKind::Atomic {
                op: ImageAtomic::CmpSwap,
                ..
            } => self.dmask.count_ones() as u8 / 2,
            ImageKind::Atomic { .. } => self.dmask.count_ones() as u8,
            ImageKind::Sample { gather: true, .. } => 4,
            _ => self.dmask.count_ones() as u8,
        };
        let regs = if self.d16 { comps.div_ceil(2) } else { comps };
        let status = u8::from(self.tfe && comps > 0);
        self.data..self.data + regs + status
    }
}

/// A vector buffer instruction (`buffer_load_*`, `buffer_store_*`, the typed forms).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferOp {
    /// The instruction, for what the fields below do not say: sign extension of the byte and
    /// short loads, the source of a typed format, the kind of an atomic. An atomic has `store` set.
    pub op: crate::tables::Op,
    /// An atomic gives the old value back in the data registers only with this bit.
    pub glc: bool,
    pub store: bool,
    /// Bytes per element moved: 1, 2 or 4. `count` elements go to or come from `count` registers.
    pub bytes: u8,
    pub count: u8,
    pub data: u8,
    /// The element index (`idxen`) and the byte offset (immediate plus register plus `offen`).
    pub index: Option<Expr>,
    pub offset: Expr,
    /// The data format of a typed access, 0 for raw.
    pub format: u8,
    pub buffer: Desc,
    pub pc: u32,
}

/// An export instruction, before the stage module gives it a meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Export {
    /// The target number: 0 to 7 colour targets, 8 depth, 12 to 15 positions, 20 primitive
    /// data, 32 and up parameters.
    pub target: u8,
    pub enabled: u8,
    /// Two 16 bit halves per source register.
    pub compressed: bool,
    pub done: bool,
    pub valid_mask: bool,
    pub src: [u8; 4],
    pub pc: u32,
}

/// A stage output component, after the stage module resolved an export.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputRef {
    /// Index into `StageProgram::outputs`.
    pub element: u16,
    pub chan: u8,
}

/// A statement.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    Set {
        dst: Var,
        value: Expr,
        guard: Guard,
    },
    Image(ImageOp, Guard),
    Buffer(BufferOp, Guard),
    Export(Export),
    /// Writes a 32 bit float pattern to an output component.
    Output {
        to: OutputRef,
        value: Expr,
    },
    /// Kills the pixel when `cond` holds.
    Discard {
        cond: Expr,
    },
    /// A call with a side effect: a store to group shared memory, an atomic, a barrier. `call`
    /// is an `Expr::Call`. The dead code pass never removes it. With a `dst` the result is
    /// written there (the old value of an atomic), under the same guard as the call.
    Effect {
        dst: Option<Var>,
        call: Expr,
        guard: Guard,
    },
    /// `s_sendmsg` with message number `id`. The payload is in `Var::M0`. Raw like `Export`:
    /// the stage resolves or drops it, the printer refuses one that is left.
    Message {
        id: u16,
        pc: u32,
    },
    /// `s_barrier`. Raw like `Export`: the stage resolves or drops it.
    Barrier {
        pc: u32,
    },
    /// A source line for the listing option.
    Comment(String),
}

/// A node of the structured program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Block(Vec<Stmt>),
    If {
        cond: Expr,
        then: Vec<Node>,
        els: Vec<Node>,
    },
    /// An endless loop left by `Break`.
    Loop(Vec<Node>),
    /// Statements that run once and can be left early: a `Break` inside (not inside a loop or
    /// another `Once` in it) goes on behind it, as does its end. It is no loop: lanes that
    /// leave it at different places meet again behind it. The structure pass writes it for
    /// code that several jumps lead to and nesting alone cannot express, and never puts a
    /// `Continue` directly into it. In HLSL it is `do { .. } while (false);`.
    Once(Vec<Node>),
    /// Leaves the innermost `Loop` or `Once`.
    Break,
    /// Starts the next round of the innermost `Loop`.
    Continue,
    Return,
}

/// What a resource is used as. It decides the HLSL type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceUse {
    /// Read as dwords at byte offsets (`s_buffer_load_dword`, raw vector loads).
    Words {
        written: bool,
    },
    /// An image of this dimension code.
    Image {
        dim: u8,
        written: bool,
        compare: bool,
    },
    Sampler {
        compare: bool,
    },
}

/// Where the console program takes a descriptor from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// The descriptor at byte `offset` of the table whose address user data register `sgpr` holds.
    Table { sgpr: u8, offset: u32 },
    /// The buffer whose address is in user data registers `sgpr` and `sgpr + 1`.
    RootDescriptor { sgpr: u8 },
    /// A sampler the program builds from four constants.
    Immediate { words: [u32; 4] },
    /// A finite set of immediate samplers selected by the descriptor words in the lane.
    SamplerChoices {
        first: u8,
        choices: Vec<([u32; 4], u32)>,
    },
    /// A descriptor the program chooses at run time: `index` holds its heap index at the
    /// moment of each use (the resource pass turns the first descriptor word into that index).
    /// The same variable can hold different indices at different uses, so the printer makes
    /// the resource at each use and never once at the top.
    HeapIndex { index: Var, sampler: bool },
}

/// One resource of a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resource {
    pub origin: Origin,
    pub usage: ResourceUse,
    /// The resource the container declares for the descriptor behind a table origin: its `PSV0`
    /// type (raw, structured or typed view, constant buffer) and kind (5 a cube, 9 a cube array,
    /// 7 a 2D array, 12 a structured buffer, ...). `None` when the container does not say. Filled
    /// by `resource::place`. The HLSL type has to match the view the title made.
    pub declared: Option<crate::api::DeclaredResource>,
    /// Filled by `resource::place`.
    pub placement: Option<Placement>,
    /// The HLSL name, filled with the placement.
    pub name: String,
}

/// One translated program before printing.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Function {
    pub body: Vec<Node>,
    pub resources: Vec<Resource>,
    /// The size of the group shared memory (the local data share of the machine) in dwords, 0
    /// for none. The printer declares `groupshared uint decant_lds[N];` before the helpers.
    pub groupshared_dwords: u32,
    /// The wave size the program was compiled for: 32 or 64.
    pub lanes: u8,
}

impl Function {
    /// Calls `f` on every statement, in program order.
    pub fn for_each_stmt(&self, f: &mut impl FnMut(&Stmt)) {
        fn walk(nodes: &[Node], f: &mut impl FnMut(&Stmt)) {
            for n in nodes {
                match n {
                    Node::Block(stmts) => stmts.iter().for_each(&mut *f),
                    Node::If { then, els, .. } => {
                        walk(then, f);
                        walk(els, f);
                    }
                    Node::Loop(body) | Node::Once(body) => walk(body, f),
                    Node::Break | Node::Continue | Node::Return => {}
                }
            }
        }
        walk(&self.body, f);
    }

    /// Calls `f` on every statement list, so a pass can rewrite statements in place.
    pub fn for_each_block_mut(&mut self, f: &mut impl FnMut(&mut Vec<Stmt>)) {
        fn walk(nodes: &mut [Node], f: &mut impl FnMut(&mut Vec<Stmt>)) {
            for n in nodes {
                match n {
                    Node::Block(stmts) => f(stmts),
                    Node::If { then, els, .. } => {
                        walk(then, f);
                        walk(els, f);
                    }
                    Node::Loop(body) | Node::Once(body) => walk(body, f),
                    Node::Break | Node::Continue | Node::Return => {}
                }
            }
        }
        walk(&mut self.body, f);
    }
}
