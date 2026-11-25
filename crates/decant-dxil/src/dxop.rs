//! The `dx.op` subset used by this crate: class names, opcodes and function attributes.
//!
//! Sources: DxilConstants.h `OpCode`, DXIL.rst operation table and hctdb.py (function attribute
//! of each operation), all public.

/// Function attribute lists of the declarations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attr {
    /// `nounwind readnone`
    ReadNone,
    /// `nounwind`
    NoUnwind,
    /// `nounwind readonly`
    ReadOnly,
    /// `nounwind noduplicate`
    NoDuplicate,
}

/// One operation: the class that makes the function name `dx.op.<class>[.<overload>]`, the
/// opcode (first argument, an `i32` constant) and the attribute.
#[derive(Clone, Copy, Debug)]
pub struct DxOp {
    pub class: &'static str,
    pub opcode: u32,
    pub attr: Attr,
}

const fn op(class: &'static str, opcode: u32, attr: Attr) -> DxOp {
    DxOp { class, opcode, attr }
}

use Attr::*;

pub const FABS: DxOp = op("unary", 6, ReadNone);
pub const SATURATE: DxOp = op("unary", 7, ReadNone);
pub const IS_NAN: DxOp = op("isSpecialFloat", 8, ReadNone);
pub const IS_INF: DxOp = op("isSpecialFloat", 9, ReadNone);
pub const IS_FINITE: DxOp = op("isSpecialFloat", 10, ReadNone);
pub const COS: DxOp = op("unary", 12, ReadNone);
pub const SIN: DxOp = op("unary", 13, ReadNone);
pub const EXP2: DxOp = op("unary", 21, ReadNone);
pub const FRACT: DxOp = op("unary", 22, ReadNone);
pub const LOG2: DxOp = op("unary", 23, ReadNone);
pub const SQRT: DxOp = op("unary", 24, ReadNone);
pub const RSQRT: DxOp = op("unary", 25, ReadNone);
pub const ROUND_NE: DxOp = op("unary", 26, ReadNone);
pub const FLOOR: DxOp = op("unary", 27, ReadNone);
pub const CEIL: DxOp = op("unary", 28, ReadNone);
pub const TRUNC: DxOp = op("unary", 29, ReadNone);
pub const BIT_REVERSE: DxOp = op("unary", 30, ReadNone);
pub const COUNT_BITS: DxOp = op("unaryBits", 31, ReadNone);
pub const FIRST_BIT_LOW: DxOp = op("unaryBits", 32, ReadNone);
pub const FIRST_BIT_HIGH: DxOp = op("unaryBits", 33, ReadNone);
pub const FIRST_BIT_HIGH_SIGNED: DxOp = op("unaryBits", 34, ReadNone);
pub const FMAX: DxOp = op("binary", 35, ReadNone);
pub const FMIN: DxOp = op("binary", 36, ReadNone);
pub const SMAX: DxOp = op("binary", 37, ReadNone);
pub const SMIN: DxOp = op("binary", 38, ReadNone);
pub const UMAX: DxOp = op("binary", 39, ReadNone);
pub const UMIN: DxOp = op("binary", 40, ReadNone);
pub const ADD_CARRY: DxOp = op("binaryWithCarryOrBorrow", 44, ReadNone);
pub const SUB_BORROW: DxOp = op("binaryWithCarryOrBorrow", 45, ReadNone);
pub const FMAD: DxOp = op("tertiary", 46, ReadNone);
pub const IMAD: DxOp = op("tertiary", 48, ReadNone);
pub const UMAD: DxOp = op("tertiary", 49, ReadNone);
pub const BFE_SIGNED: DxOp = op("tertiary", 51, ReadNone);
pub const BFE_UNSIGNED: DxOp = op("tertiary", 52, ReadNone);
pub const BFI: DxOp = op("quaternary", 53, ReadNone);
pub const CBUFFER_LOAD: DxOp = op("cbufferLoadLegacy", 59, ReadOnly);
pub const BUFFER_LOAD: DxOp = op("bufferLoad", 68, ReadOnly);
pub const BUFFER_STORE: DxOp = op("bufferStore", 69, NoUnwind);
pub const ATOMIC_BIN_OP: DxOp = op("atomicBinOp", 78, NoUnwind);
pub const ATOMIC_COMPARE_EXCHANGE: DxOp = op("atomicCompareExchange", 79, NoUnwind);
pub const BARRIER: DxOp = op("barrier", 80, NoDuplicate);
pub const THREAD_ID: DxOp = op("threadId", 93, ReadNone);
pub const GROUP_ID: DxOp = op("groupId", 94, ReadNone);
pub const THREAD_ID_IN_GROUP: DxOp = op("threadIdInGroup", 95, ReadNone);
pub const FLATTENED_THREAD_ID_IN_GROUP: DxOp = op("flattenedThreadIdInGroup", 96, ReadNone);
pub const LOAD_INPUT: DxOp = op("loadInput", 4, ReadNone);
pub const STORE_OUTPUT: DxOp = op("storeOutput", 5, NoUnwind);
pub const SAMPLE: DxOp = op("sample", 60, ReadOnly);
pub const SAMPLE_BIAS: DxOp = op("sampleBias", 61, ReadOnly);
pub const SAMPLE_LEVEL: DxOp = op("sampleLevel", 62, ReadOnly);
pub const SAMPLE_GRAD: DxOp = op("sampleGrad", 63, ReadOnly);
pub const SAMPLE_CMP: DxOp = op("sampleCmp", 64, ReadOnly);
pub const SAMPLE_CMP_LEVEL_ZERO: DxOp = op("sampleCmpLevelZero", 65, ReadOnly);
pub const TEXTURE_LOAD: DxOp = op("textureLoad", 66, ReadOnly);
pub const TEXTURE_STORE: DxOp = op("textureStore", 67, NoUnwind);
pub const GET_DIMENSIONS: DxOp = op("getDimensions", 72, ReadOnly);
pub const TEXTURE_GATHER: DxOp = op("textureGather", 73, ReadOnly);
pub const TEXTURE_GATHER_CMP: DxOp = op("textureGatherCmp", 74, ReadOnly);
pub const CALCULATE_LOD: DxOp = op("calculateLOD", 81, ReadOnly);
pub const DISCARD: DxOp = op("discard", 82, NoUnwind);
pub const DDX_COARSE: DxOp = op("unary", 83, ReadNone);
pub const DDY_COARSE: DxOp = op("unary", 84, ReadNone);
pub const DDX_FINE: DxOp = op("unary", 85, ReadNone);
pub const DDY_FINE: DxOp = op("unary", 86, ReadNone);
pub const IS_HELPER_LANE: DxOp = op("isHelperLane", 221, NoUnwind);
pub const WAVE_IS_FIRST_LANE: DxOp = op("waveIsFirstLane", 110, NoUnwind);
pub const WAVE_LANE_INDEX: DxOp = op("waveGetLaneIndex", 111, NoUnwind);
pub const WAVE_LANE_COUNT: DxOp = op("waveGetLaneCount", 112, NoUnwind);
pub const WAVE_ANY_TRUE: DxOp = op("waveAnyTrue", 113, NoUnwind);
pub const WAVE_ALL_TRUE: DxOp = op("waveAllTrue", 114, NoUnwind);
pub const WAVE_ALL_EQUAL: DxOp = op("waveActiveAllEqual", 115, NoUnwind);
pub const WAVE_BALLOT: DxOp = op("waveActiveBallot", 116, NoUnwind);
pub const WAVE_READ_LANE_AT: DxOp = op("waveReadLaneAt", 117, NoUnwind);
pub const WAVE_READ_LANE_FIRST: DxOp = op("waveReadLaneFirst", 118, NoUnwind);
pub const WAVE_ACTIVE_OP: DxOp = op("waveActiveOp", 119, NoUnwind);
pub const WAVE_ACTIVE_BIT: DxOp = op("waveActiveBit", 120, NoUnwind);
pub const WAVE_PREFIX_OP: DxOp = op("wavePrefixOp", 121, NoUnwind);
pub const QUAD_READ_LANE_AT: DxOp = op("quadReadLaneAt", 122, NoUnwind);
pub const QUAD_OP: DxOp = op("quadOp", 123, NoUnwind);
pub const WAVE_ALL_BIT_COUNT: DxOp = op("waveAllOp", 135, NoUnwind);
pub const WAVE_PREFIX_BIT_COUNT: DxOp = op("wavePrefixBitCount", 136, NoUnwind);
pub const LEGACY_F32_TO_F16: DxOp = op("legacyF32ToF16", 130, ReadNone);
pub const LEGACY_F16_TO_F32: DxOp = op("legacyF16ToF32", 131, ReadNone);
pub const RAW_BUFFER_LOAD: DxOp = op("rawBufferLoad", 139, ReadOnly);
pub const RAW_BUFFER_STORE: DxOp = op("rawBufferStore", 140, NoUnwind);
pub const CREATE_HANDLE_FROM_HEAP: DxOp = op("createHandleFromHeap", 218, ReadNone);
pub const ANNOTATE_HANDLE: DxOp = op("annotateHandle", 216, ReadNone);
pub const CREATE_HANDLE_FROM_BINDING: DxOp = op("createHandleFromBinding", 217, ReadNone);

/// `AtomicBinOpCode` (DxilConstants.h).
pub fn atomic_code(op: crate::ir::AtomicOp) -> u32 {
    use crate::ir::AtomicOp::*;
    match op {
        Add => 0,
        And => 1,
        Or => 2,
        Xor => 3,
        SMin => 4,
        SMax => 5,
        UMin => 6,
        UMax => 7,
        Xchg => 8,
    }
}
