//! The input IR: a small scalar SSA form, one function per shader (see the crate README).
//!
//! Everything here is `serde` serialisable. The text form is JSON, see [`Shader::to_sxir`].

use serde::{Deserialize, Serialize};

/// Value types. No vectors: a front end scalarises.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Bool,
    I32,
    I64,
    F16,
    F32,
}

/// The id of an SSA value (an instruction result or a phi). Ids are chosen by the front end and
/// are unique in one shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// The index of a basic block in [`Shader::blocks`]. Block 0 is the entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// A constant. Floats are bit patterns, so NaN payloads and negative zero survive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Const {
    Bool(bool),
    I32(u32),
    I64(u64),
    F16(u16),
    F32(u32),
}

impl Const {
    pub fn ty(self) -> Ty {
        match self {
            Const::Bool(_) => Ty::Bool,
            Const::I32(_) => Ty::I32,
            Const::I64(_) => Ty::I64,
            Const::F16(_) => Ty::F16,
            Const::F32(_) => Ty::F32,
        }
    }

    /// An `f32` constant from a value.
    pub fn f32(v: f32) -> Const {
        Const::F32(v.to_bits())
    }
}

/// An instruction operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operand {
    Value(ValueId),
    Const(Const),
}

impl Operand {
    pub fn i32(v: u32) -> Operand {
        Operand::Const(Const::I32(v))
    }
    pub fn f32(v: f32) -> Operand {
        Operand::Const(Const::f32(v))
    }
    pub fn value(v: u32) -> Operand {
        Operand::Value(ValueId(v))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreadComponentKind {
    /// `SV_DispatchThreadID`
    Dispatch,
    /// `SV_GroupID`
    Group,
    /// `SV_GroupThreadID`
    InGroup,
}

/// Two operand arithmetic. The type is the type of the operands, which must agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    /// Shift counts are masked with the width minus one by the lowering (HLSL semantics).
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

/// One operand operations that map to `dx.op` unary calls, special float tests and bit counting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnOp {
    FAbs,
    Saturate,
    Floor,
    Ceil,
    Trunc,
    /// Round to nearest even.
    RoundNe,
    Fract,
    Sqrt,
    Rsqrt,
    Exp2,
    Log2,
    Sin,
    Cos,
    IsNan,
    IsInf,
    IsFinite,
    BitReverse,
    /// Population count, returns `I32`.
    CountBits,
    /// Index of the lowest set bit, 0xFFFFFFFF when none. Returns `I32`.
    FirstBitLow,
    /// Position of the highest set bit counted from the most significant bit (the DXIL
    /// operation itself, 0 means the top bit is set), 0xFFFFFFFF when none.
    FirstBitHigh,
    /// The same for signed values (first bit that differs from the sign bit).
    FirstBitHighSigned,
    /// `f32` to the bits of an `f16` in the low half of an `I32`.
    F32ToF16Bits,
    /// The reverse.
    F16BitsToF32,
    /// Derivatives (pixel shaders and compute with derivative groups): coarse and fine, x and y.
    DdxCoarse,
    DdyCoarse,
    DdxFine,
    DdyFine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MinMaxOp {
    FMin,
    FMax,
    UMin,
    UMax,
    SMin,
    SMax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MadOp {
    /// `a * b + c` on floats (not fused).
    FMad,
    IMad,
    UMad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CastOp {
    Trunc,
    ZExt,
    SExt,
    FpToUi,
    FpToSi,
    UiToFp,
    SiToFp,
    FpTrunc,
    FpExt,
    /// Between a float and an integer of the same width.
    Bitcast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IPred {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FPred {
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Uno,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
}

/// Atomic operations on buffers (DXIL `AtomicBinOpCode`). Subtraction is an add of the negation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomicOp {
    Add,
    And,
    Or,
    Xor,
    SMin,
    SMax,
    UMin,
    UMax,
    Xchg,
}

/// Atomic operations on groupshared memory (LLVM `atomicrmw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GsAtomicOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    SMin,
    SMax,
    UMin,
    UMax,
    Xchg,
}

/// `BarrierMode` flags of the `barrier` operation.
pub mod barrier {
    pub const SYNC_GROUP: u32 = 1;
    pub const UAV_FENCE_GLOBAL: u32 = 2;
    pub const UAV_FENCE_GROUP: u32 = 4;
    pub const GROUPSHARED_FENCE: u32 = 8;
}

/// The operation of one instruction. Resource and groupshared operands are indices into
/// [`Shader::resources`] and [`Shader::groupshared`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Bin(BinOp, Operand, Operand),
    Un(UnOp, Operand),
    MinMax(MinMaxOp, Operand, Operand),
    Mad(MadOp, Operand, Operand, Operand),
    /// `width`, `offset`, `value`: extracts `width` bits starting at `offset`.
    BitfieldExtract { signed: bool, width: Operand, offset: Operand, value: Operand },
    /// `width`, `offset`, `insert`, `base`: the DXIL `bfi` operands in that order.
    BitfieldInsert { width: Operand, offset: Operand, insert: Operand, base: Operand },
    /// Two results: the sum and the carry (`Bool`).
    AddCarry(Operand, Operand),
    /// Two results: the difference and the borrow (`Bool`).
    SubBorrow(Operand, Operand),
    Cast(CastOp, Operand, Ty),
    ICmp(IPred, Operand, Operand),
    FCmp(FPred, Operand, Operand),
    /// `condition`, `if true`, `if false`.
    Select(Operand, Operand, Operand),
    /// One component of a thread id (0 to 2), result `I32`.
    ThreadId(ThreadComponentKind, u8),
    /// `SV_GroupIndex`, result `I32`.
    FlattenedThreadIdInGroup,
    /// Reads components `comps` of the 16 byte row `row` of the constant buffer `res`, one result
    /// per entry of `comps`, of type `ty` (`I32` or `F32`).
    CbufLoad { res: ResRef, row: Operand, comps: Vec<u8>, ty: Ty },
    /// Loads `count` (1 to 4) consecutive 32 bit components. Raw buffer: `index` is the byte
    /// offset, `offset` must be `None`. Structured buffer: `index` is the element, `offset` the
    /// byte offset in it (`None` means 0). Typed buffer: `index` is the element, `offset` `None`.
    /// One result per component, of type `ty` (`I32` or `F32`).
    BufLoad { res: ResRef, index: Operand, offset: Option<Operand>, count: u8, ty: Ty },
    /// Stores 1 to 4 consecutive components (a typed buffer needs exactly 4). Addressing as `BufLoad`.
    BufStore { res: ResRef, index: Operand, offset: Option<Operand>, values: Vec<Operand> },
    /// Atomic on a raw or structured UAV, addressing as `BufLoad`, result is the old `I32` value.
    BufAtomic { op: AtomicOp, res: ResRef, index: Operand, offset: Option<Operand>, value: Operand },
    /// Compare exchange on a raw or structured UAV, result is the old `I32` value.
    BufCmpXchg { res: ResRef, index: Operand, offset: Option<Operand>, compare: Operand, new: Operand },
    /// Load of element `index` of groupshared variable `var`.
    GsLoad { var: u32, index: Operand },
    GsStore { var: u32, index: Operand, value: Operand },
    /// Result is the old value.
    GsAtomic { op: GsAtomicOp, var: u32, index: Operand, value: Operand },
    /// Result is the old value.
    GsCmpXchg { var: u32, index: Operand, compare: Operand, new: Operand },
    /// `dx.op.barrier` with the flags of [`barrier`]. Only in uniform control flow.
    Barrier { flags: u32 },
    /// `IsHelperLane` (Shader Model 6.6), `Bool`.
    IsHelperLane,
    /// Reads component `col` of row `row` of input element `elem`. One result of the element type.
    LoadInput { elem: u32, row: Operand, col: u8 },
    /// Writes component `col` of row `row` of output element `elem`.
    StoreOutput { elem: u32, row: Operand, col: u8, value: Operand },
    /// Discards the pixel when the `Bool` is true (pixel shaders).
    Discard(Operand),
    /// `Texture.Load`: `coords` are the integer texel coordinates (array slice last, 1 to 3),
    /// `mip` the mip level (or sample index), `offsets` constant offsets (or empty). Four results
    /// of type `ty` (`I32` or `F32`).
    TexLoad { tex: ResRef, mip: Option<Operand>, coords: Vec<Operand>, offsets: Vec<i32>, ty: Ty },
    /// Stores four texel components to a UAV texture.
    TexStore { tex: ResRef, coords: Vec<Operand>, values: Vec<Operand> },
    /// An atomic on a texel of a UAV texture of 32 bit integers. One result: the old value.
    TexAtomic { op: AtomicOp, tex: ResRef, coords: Vec<Operand>, value: Operand },
    /// Compare exchange on a texel of a UAV texture. One result: the old value.
    TexCmpXchg { tex: ResRef, coords: Vec<Operand>, compare: Operand, new: Operand },
    /// The sample family: `coords` are `F32` (array slice last), `offsets` constants. Four results
    /// of type `ty` (`F32`, or `I32` for integer textures with the plain and level forms).
    Sample { tex: ResRef, sampler: ResRef, coords: Vec<Operand>, offsets: Vec<i32>, mode: SampleMode, clamp: Option<Operand>, ty: Ty },
    /// `Gather` of component `channel` (0 to 3), or `GatherCmp` with `compare`. Four results.
    Gather { tex: ResRef, sampler: ResRef, coords: Vec<Operand>, offsets: Vec<i32>, channel: u8, compare: Option<Operand>, ty: Ty },
    /// `GetDimensions`: four `I32` results (width, height, depth or array size, mip count).
    GetDimensions { tex: ResRef, mip: Option<Operand> },
    /// `CalculateLevelOfDetail` (`clamped`) or the unclamped one, one `F32` result.
    CalculateLod { tex: ResRef, sampler: ResRef, coords: Vec<Operand>, clamped: bool },
    /// `WaveGetLaneIndex`, `I32`.
    WaveLaneIndex,
    /// `WaveGetLaneCount`, `I32`.
    WaveLaneCount,
    /// `WaveIsFirstLane`, `Bool`.
    WaveIsFirstLane,
    /// `WaveActiveAnyTrue`, `Bool`.
    WaveAnyTrue(Operand),
    /// `WaveActiveAllTrue`, `Bool`.
    WaveAllTrue(Operand),
    /// `WaveActiveAllEqual`, `Bool`. The operand is `I32`, `I64` or `F32`.
    WaveAllEqual(Operand),
    /// `WaveActiveBallot`: four results (`I32`), the lane mask words 0 to 3.
    WaveBallot(Operand),
    /// `WaveReadLaneAt`: `lane` is an `I32`.
    WaveReadLaneAt { value: Operand, lane: Operand },
    WaveReadLaneFirst(Operand),
    /// `WaveActiveSum`, `Product`, `Min`, `Max`. `signed` only matters for integers.
    WaveActive { op: WaveOp, signed: bool, value: Operand },
    /// `WaveActiveBitAnd`, `Or`, `Xor` on `I32` or `I64`.
    WaveActiveBit { op: WaveBitOp, value: Operand },
    /// `WavePrefixSum` and `WavePrefixProduct` (exclusive).
    WavePrefix { op: WaveOp, signed: bool, value: Operand },
    /// `WaveActiveCountBits`, `I32`.
    WaveActiveCountBits(Operand),
    /// `WavePrefixCountBits`, `I32`.
    WavePrefixCountBits(Operand),
    /// `QuadReadLaneAt`: `lane` is an `I32` from 0 to 3.
    QuadReadLaneAt { value: Operand, lane: Operand },
    /// `QuadReadAcrossX`, `Y` and `Diagonal`.
    QuadOp { kind: QuadKind, value: Operand },
}

/// The form of a sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SampleMode {
    /// `Sample`: implicit derivatives.
    Implicit,
    /// `SampleBias`.
    Bias(Operand),
    /// `SampleLevel`.
    Level(Operand),
    /// `SampleGrad`: the derivatives, one operand per coordinate (without the array slice).
    Grad { ddx: Vec<Operand>, ddy: Vec<Operand> },
    /// `SampleCmp`.
    Compare(Operand),
    /// `SampleCmpLevelZero`.
    CompareLevelZero(Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveOp {
    Sum,
    Product,
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveBitOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuadKind {
    AcrossX,
    AcrossY,
    Diagonal,
}

/// How an instruction names a resource: a bound resource by its index in [`Shader::resources`],
/// or an element of the descriptor heap (`ResourceDescriptorHeap[index]`, Shader Model 6.6) of
/// the kind described by that resource, which must be declared with [`Binding::Heap`]. The handle
/// of a heap resource is created at each use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResRef {
    pub res: u32,
    /// The heap index, for a resource declared with [`Binding::Heap`].
    #[serde(default)]
    pub index: Option<Operand>,
    /// The index may differ between lanes of a wave.
    #[serde(default)]
    pub non_uniform: bool,
}

impl ResRef {
    pub fn bound(res: u32) -> ResRef {
        ResRef { res, index: None, non_uniform: false }
    }
    pub fn heap(res: u32, index: Operand, non_uniform: bool) -> ResRef {
        ResRef { res, index: Some(index), non_uniform }
    }
}

impl From<u32> for ResRef {
    fn from(res: u32) -> ResRef {
        ResRef::bound(res)
    }
}

/// Where a resource lives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Binding {
    /// A register range of a register space.
    #[default]
    Bound,
    /// The descriptor heap, indexed at each use. Needs the root signature flag
    /// `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (or the sampler one).
    Heap,
}

/// One instruction with the ids of its results: none (stores, barrier), one, or several
/// (loads of several components, carry operations).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Inst {
    pub dsts: Vec<ValueId>,
    pub op: Op,
}

impl Inst {
    pub fn new(dsts: &[u32], op: Op) -> Inst {
        Inst { dsts: dsts.iter().map(|&d| ValueId(d)).collect(), op }
    }
}

/// A phi: one incoming operand for each predecessor edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Phi {
    pub dst: ValueId,
    pub ty: Ty,
    pub incoming: Vec<(BlockId, Operand)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    Br(BlockId),
    /// `condition` (a `Bool`), `if true`, `if false`.
    CondBr(Operand, BlockId, BlockId),
    Ret,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub term: Terminator,
}

impl Block {
    pub fn new(insts: Vec<Inst>, term: Terminator) -> Block {
        Block { phis: Vec::new(), insts, term }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceClass {
    Srv,
    Uav,
    CBuffer,
    Sampler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceKind {
    TypedBuffer,
    RawBuffer,
    StructuredBuffer,
    CBuffer,
    Sampler,
    Texture1D,
    Texture2D,
    Texture3D,
    TextureCube,
    Texture1DArray,
    Texture2DArray,
    TextureCubeArray,
}

impl ResourceKind {
    /// Whether this is one of the texture kinds (multisampled textures are not supported yet).
    pub fn is_texture(self) -> bool {
        use ResourceKind::*;
        matches!(self, Texture1D | Texture2D | Texture3D | TextureCube | Texture1DArray | Texture2DArray | TextureCubeArray)
    }

    /// Number of coordinates a sample takes, array slice included.
    pub fn coord_count(self) -> usize {
        use ResourceKind::*;
        match self {
            Texture1D => 1,
            Texture2D | Texture1DArray => 2,
            Texture3D | TextureCube | Texture2DArray => 3,
            TextureCubeArray => 4,
            _ => 0,
        }
    }

    /// Number of offsets a sample takes (the array slice has none, cube maps have none).
    pub fn offset_count(self) -> usize {
        use ResourceKind::*;
        match self {
            Texture1D | Texture1DArray => 1,
            Texture2D | Texture2DArray => 2,
            Texture3D => 3,
            _ => 0,
        }
    }
}

/// DXIL `ComponentType` of a typed buffer element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    I32,
    U32,
    F32,
}

/// Element type of a typed buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Element {
    pub component: ComponentType,
    /// 1 to 4.
    pub count: u8,
}

/// A bound resource: a register range of a register space, in the root signature the host builds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Resource {
    #[serde(default)]
    pub binding: Binding,
    pub class: ResourceClass,
    pub kind: ResourceKind,
    pub space: u32,
    pub lower_bound: u32,
    /// Only 1 is supported for now.
    pub range_size: u32,
    pub name: String,
    /// Typed buffers.
    pub element: Option<Element>,
    /// Structured buffers: bytes per element.
    pub stride: u32,
    /// Constant buffers: size in bytes, a multiple of 16.
    pub size_bytes: u32,
    /// UAV flags.
    pub globally_coherent: bool,
    pub has_counter: bool,
    pub rasterizer_ordered: bool,
    /// Samplers: a comparison sampler.
    #[serde(default)]
    pub comparison: bool,
}

impl Resource {
    /// A texture SRV or UAV with float or integer texels.
    pub fn texture(class: ResourceClass, kind: ResourceKind, space: u32, register: u32, element: Element) -> Resource {
        Resource { element: Some(element), ..Resource::base(class, kind, space, register) }
    }
    /// A sampler.
    pub fn sampler(space: u32, register: u32, comparison: bool) -> Resource {
        Resource { comparison, ..Resource::base(ResourceClass::Sampler, ResourceKind::Sampler, space, register) }
    }
    /// The same resource, taken from the descriptor heap: register and space are ignored.
    pub fn in_heap(self) -> Resource {
        Resource { binding: Binding::Heap, ..self }
    }
    fn base(class: ResourceClass, kind: ResourceKind, space: u32, register: u32) -> Resource {
        Resource {
            binding: Binding::Bound,
            class,
            kind,
            space,
            lower_bound: register,
            range_size: 1,
            name: String::new(),
            element: None,
            stride: 0,
            size_bytes: 0,
            globally_coherent: false,
            has_counter: false,
            rasterizer_ordered: false,
            comparison: false,
        }
    }
    /// A raw (byte address) buffer, `ByteAddressBuffer` or `RWByteAddressBuffer`.
    pub fn raw(class: ResourceClass, space: u32, register: u32) -> Resource {
        Resource::base(class, ResourceKind::RawBuffer, space, register)
    }
    pub fn structured(class: ResourceClass, space: u32, register: u32, stride: u32) -> Resource {
        Resource { stride, ..Resource::base(class, ResourceKind::StructuredBuffer, space, register) }
    }
    pub fn typed(class: ResourceClass, space: u32, register: u32, element: Element) -> Resource {
        Resource { element: Some(element), ..Resource::base(class, ResourceKind::TypedBuffer, space, register) }
    }
    pub fn cbuffer(space: u32, register: u32, size_bytes: u32) -> Resource {
        Resource { size_bytes, ..Resource::base(ResourceClass::CBuffer, ResourceKind::CBuffer, space, register) }
    }
}

/// A groupshared array of 32 bit elements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Groupshared {
    pub len: u32,
    /// `I32` or `F32`.
    pub ty: Ty,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Stage {
    Compute {
        threads: [u32; 3],
        /// `WaveSize` (Shader Model 6.6): a power of two from 4 to 128.
        #[serde(default)]
        wave_size: Option<u32>,
    },
    Vertex,
    Pixel,
}

/// DXIL `SemanticKind` of a signature element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Semantic {
    Arbitrary,
    VertexId,
    InstanceId,
    Position,
    RenderTargetArrayIndex,
    ViewportArrayIndex,
    ClipDistance,
    CullDistance,
    PrimitiveId,
    SampleIndex,
    IsFrontFace,
    Coverage,
    InnerCoverage,
    /// `SV_Barycentrics`: index 0 is perspective correct, 1 is linear in screen space.
    Barycentrics,
    Target,
    Depth,
    DepthLessEqual,
    DepthGreaterEqual,
    StencilRef,
}

/// DXIL `InterpolationMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interp {
    Undefined,
    Constant,
    Linear,
    LinearCentroid,
    LinearNoperspective,
    LinearNoperspectiveCentroid,
    LinearSample,
    LinearNoperspectiveSample,
}

/// One element of the input or output signature. The element is the unit of `LoadInput` and
/// `StoreOutput`: they name it by its index in [`Shader::inputs`] or [`Shader::outputs`], a row
/// inside it (0 to `rows` minus 1) and a column (0 to `cols` minus 1, relative to `start_col`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SigElement {
    pub name: String,
    pub index: u32,
    pub semantic: Semantic,
    /// `I32`, `U32` or `F32`.
    pub component: ComponentType,
    pub interp: Interp,
    /// The register (row) of the first row.
    pub start_row: u32,
    /// The first component (0 to 3) in the register.
    pub start_col: u8,
    pub rows: u32,
    /// Components per row, 1 to 4.
    pub cols: u8,
}

impl SigElement {
    /// `start_row` of an element that has no register (`SV_Depth`, `SV_Coverage`, `SV_SampleIndex` and
    /// the like). Then `start_col` is ignored.
    pub const UNALLOCATED: u32 = u32::MAX;

    /// Whether the element has a register.
    pub fn allocated(&self) -> bool {
        self.start_row != Self::UNALLOCATED
    }

    pub fn new(name: &str, index: u32, semantic: Semantic, component: ComponentType, interp: Interp, start_row: u32, cols: u8) -> SigElement {
        SigElement { name: name.to_owned(), index, semantic, component, interp, start_row, start_col: 0, rows: 1, cols }
    }
}

/// A whole shader.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shader {
    pub stage: Stage,
    pub resources: Vec<Resource>,
    pub groupshared: Vec<Groupshared>,
    pub blocks: Vec<Block>,
    /// Input signature (vertex and pixel stages).
    #[serde(default)]
    pub inputs: Vec<SigElement>,
    /// Output signature (vertex and pixel stages).
    #[serde(default)]
    pub outputs: Vec<SigElement>,
}

impl Shader {
    /// A vertex shader, no resources and no signature elements yet.
    pub fn vertex(blocks: Vec<Block>) -> Shader {
        Shader { stage: Stage::Vertex, resources: Vec::new(), groupshared: Vec::new(), blocks, inputs: Vec::new(), outputs: Vec::new() }
    }

    /// A pixel shader.
    pub fn pixel(blocks: Vec<Block>) -> Shader {
        Shader { stage: Stage::Pixel, ..Shader::vertex(blocks) }
    }

    /// A compute shader with the given thread group size and no resources.
    pub fn compute(threads: [u32; 3], blocks: Vec<Block>) -> Shader {
        Shader { stage: Stage::Compute { threads, wave_size: None }, ..Shader::vertex(blocks) }
    }

    /// The `.sxir` form: this shader as pretty printed JSON (the `serde` representation of the
    /// types of this module), UTF 8, with a trailing newline.
    pub fn to_sxir(&self) -> String {
        let mut text = serde_json::to_string_pretty(self).expect("the IR serialises");
        text.push('\n');
        text
    }

    /// Reads the `.sxir` form.
    pub fn from_sxir(text: &str) -> Result<Shader, serde_json::Error> {
        serde_json::from_str(text)
    }
}
