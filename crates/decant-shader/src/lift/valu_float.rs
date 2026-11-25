//! Vector float instructions: 32 and 16 bit arithmetic, conversions, rounding, the
//! transcendental approximations, the cube face instructions and the mixed precision multiply
//! add.
//!
//! Values travel as 32 bit patterns. A float rule reinterprets them only inside its HLSL helper,
//! so a pattern that is no float survives every move. A 16 bit float is the low 16 bits of a
//! pattern, upper bits zero, until it is written into its half of a register.
//!
//! Operand order of the two operand forms: `src[0]` is the left operand. The `mac` and `fmac`
//! forms accumulate into the destination. The `mk` and `ak` forms find their literal in `src[2]`:
//! `madmk` is `src0 * K + src1`, `madak` is `src0 * src1 + K`.
//!
//! Modifiers, as the hardware applies them:
//! * ABS then NEG on each float source, on the bit pattern (sign bit 31, or 15 for 16 bit).
//! * the output modifier (`omod`: times 2, times 4, times 0.5) on the result, then the clamp
//!   bit (to [0, 1], not a number to 0). LLVM `SIFoldOperands.cpp`: "Clamp is applied after
//!   omod". Mesa `aco_select_nir_alu.cpp`: clamp turns not a number into 0.
//! * `op_sel` on the 16 bit instructions that Mesa's `can_use_opsel` allows on GFX10 (the three
//!   operand ones and `v_pack_b32_f16`): bit i picks the high half of source i, bit 3 the half of
//!   the destination that is written. A 16 bit result leaves the other half of the register as
//!   it was (Mesa `instr_is_16bit`, GFX10 and later).
//!
//! Exactness, helper by helper:
//! * add, subtract, multiply, compares, conversions between integers and floats, the 16 bit
//!   conversions, rounding to integers, minimum, maximum, median, clamp, the legacy multiply and
//!   the cube instructions are bit exact. The 16 bit conversions and the minimum family work on
//!   bit patterns, so they do not depend on how the device treats denormals.
//! * 16 bit add, subtract and multiply run in 32 bit and round once to 16 bit. That is exact:
//!   24 bits of precision are at least 2 * 11 + 2, so the double rounding is harmless.
//! * The fused multiply add (`v_fma_f32`, `v_fmac_f32`, `fmaak`, `fmamk`, the mixed forms) uses
//!   HLSL `mad`. DXC turns it into `dx.op.fmad`, which a driver may fuse or not, so the last bit
//!   can differ from the console where the driver does not fuse. A device test reports which.
//!   An exact form needs doubles (a design decision: not worth their cost). Every fused
//!   multiply add goes through [`FMA`], the one place to change.
//! * The unfused `v_mad_f32` family flushes denormal inputs, its product and its result
//!   explicitly, as the hardware does whatever MODE says.
//! * `v_rcp`, `v_rsq`, `v_sqrt`, `v_exp`, `v_log`, `v_sin`, `v_cos` use the HLSL intrinsics. The
//!   console instructions approximate to about one unit in the last place and so do the device's,
//!   but not with the same error: these are not bit exact and cannot be without measured tables.
//!   `v_sin` and `v_cos` take revolutions: the argument is reduced exactly by its nearest integer
//!   before the multiply by 2 pi.
//! * 16 bit fused multiply add and the 16 bit transcendental instructions run in 32 bit and round
//!   once, so a rare result can differ in the last bit.
//! * A not a number from an operation that creates one (0 times infinity and so on) is whatever
//!   the device makes. The payload of an input that is not a number is kept where the helper
//!   passes the pattern through.

use super::Lifter;
use super::operand::{bits, form, set_vdst};
use super::valu_int::{sdwa_dst, sdwa_src};
use crate::decode::{Ext, FloatConst, Form, Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Helper, Ty, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

macro_rules! helper {
    ($id:ident, $ty:expr, $name:literal, $needs:expr, $text:literal) => {
        pub(crate) static $id: Helper = Helper {
            name: $name,
            result: $ty,
            text: $text,
            needs: $needs,
        };
    };
}

helper!(
    NAN,
    Ty::Bool,
    "decant_vf_nan",
    &[],
    "bool decant_vf_nan(uint a) { return (a & 0x7fffffffu) > 0x7f800000u; }"
);
helper!(
    FTZ,
    Ty::U32,
    "decant_vf_ftz",
    &[],
    "uint decant_vf_ftz(uint a) { return ((a & 0x7f800000u) == 0u) ? (a & 0x80000000u) : a; }"
);
// The minimum family with GFX10 rules (IEEE mode off): a not a number loses against a number,
// negative zero is below positive zero, and a returned not a number is not quieted. LLVM
// AMDGPUInstCombineIntrinsic.cpp, the med3 notes: "with ieee=0, like llvm.minimumnum except a
// returned signaling nan will not be quieted". Denormal inputs are flushed first, as the console
// does in its f32 flush mode: measured on a device, a compiler may turn the select into a float
// minimum that flushes, so without the explicit flush the result would depend on the driver.
helper!(
    MIN,
    Ty::U32,
    "decant_vf_min",
    &[&NAN, &FTZ],
    "uint decant_vf_min(uint a, uint b) { a = decant_vf_ftz(a); b = decant_vf_ftz(b); if (decant_vf_nan(a)) return decant_vf_nan(b) ? a : b; if (decant_vf_nan(b)) return a; if (((a | b) & 0x7fffffffu) == 0u) return a | b; return (asfloat(a) < asfloat(b)) ? a : b; }"
);
helper!(
    MAX,
    Ty::U32,
    "decant_vf_max",
    &[&NAN, &FTZ],
    "uint decant_vf_max(uint a, uint b) { a = decant_vf_ftz(a); b = decant_vf_ftz(b); if (decant_vf_nan(a)) return decant_vf_nan(b) ? a : b; if (decant_vf_nan(b)) return a; if (((a | b) & 0x7fffffffu) == 0u) return a & b; return (asfloat(a) > asfloat(b)) ? a : b; }"
);
helper!(
    MIN3,
    Ty::U32,
    "decant_vf_min3",
    &[&MIN],
    "uint decant_vf_min3(uint a, uint b, uint c) { return decant_vf_min(decant_vf_min(a, b), c); }"
);
helper!(
    MAX3,
    Ty::U32,
    "decant_vf_max3",
    &[&MAX],
    "uint decant_vf_max3(uint a, uint b, uint c) { return decant_vf_max(decant_vf_max(a, b), c); }"
);
// LLVM AMDGPUInstCombineIntrinsic.cpp, ieee=0: "s0 _nan: min(s1, s2)" and so on.
helper!(
    MED3,
    Ty::U32,
    "decant_vf_med3",
    &[&NAN, &MIN, &MAX],
    "uint decant_vf_med3(uint a, uint b, uint c) { if (decant_vf_nan(a)) return decant_vf_min(b, c); if (decant_vf_nan(b)) return decant_vf_min(a, c); if (decant_vf_nan(c)) return decant_vf_min(a, b); return decant_vf_max(decant_vf_min(a, b), decant_vf_min(decant_vf_max(a, b), c)); }"
);
helper!(
    SAT,
    Ty::U32,
    "decant_vf_sat",
    &[],
    "uint decant_vf_sat(uint a) { float x = asfloat(a); return (x > 0.0f) ? ((x < 1.0f) ? a : 0x3f800000u) : 0u; }"
);
helper!(
    FMA,
    Ty::U32,
    "decant_vf_fma",
    &[],
    "uint decant_vf_fma(uint a, uint b, uint c) { return asuint(mad(asfloat(a), asfloat(b), asfloat(c))); }"
);
helper!(
    MAD,
    Ty::U32,
    "decant_vf_mad",
    &[&FTZ],
    "uint decant_vf_mad(uint a, uint b, uint c) { float p = asfloat(decant_vf_ftz(asuint(asfloat(decant_vf_ftz(a)) * asfloat(decant_vf_ftz(b))))); return decant_vf_ftz(asuint(p + asfloat(decant_vf_ftz(c)))); }"
);
// The legacy multiply: a zero operand (a denormal counts as zero where the device flushes)
// gives +0 even against infinity or not a number. LLVM AMDGPUInstCombineIntrinsic.cpp: "The
// legacy behaviour is that multiplying +/-0.0 by anything, even NaN or infinity, gives +0.0".
helper!(
    MUL_LEGACY,
    Ty::U32,
    "decant_vf_mul_legacy",
    &[],
    "uint decant_vf_mul_legacy(uint a, uint b) { float x = asfloat(a); float y = asfloat(b); return (x == 0.0f || y == 0.0f) ? 0u : asuint(x * y); }"
);
// The fused legacy form adds the addend to that +0, so a -0 addend gives +0.
helper!(
    FMA_LEGACY,
    Ty::U32,
    "decant_vf_fma_legacy",
    &[],
    "uint decant_vf_fma_legacy(uint a, uint b, uint c) { float x = asfloat(a); float y = asfloat(b); float z = asfloat(c); return (x == 0.0f || y == 0.0f) ? asuint(0.0f + z) : asuint(mad(x, y, z)); }"
);
helper!(
    RCP,
    Ty::U32,
    "decant_vf_rcp",
    &[],
    "uint decant_vf_rcp(uint a) { return asuint(rcp(asfloat(a))); }"
);
helper!(
    RSQ,
    Ty::U32,
    "decant_vf_rsq",
    &[],
    "uint decant_vf_rsq(uint a) { return asuint(rsqrt(asfloat(a))); }"
);
helper!(
    SQRT,
    Ty::U32,
    "decant_vf_sqrt",
    &[],
    "uint decant_vf_sqrt(uint a) { return asuint(sqrt(asfloat(a))); }"
);
helper!(
    EXP,
    Ty::U32,
    "decant_vf_exp",
    &[],
    "uint decant_vf_exp(uint a) { return asuint(exp2(asfloat(a))); }"
);
helper!(
    LOG,
    Ty::U32,
    "decant_vf_log",
    &[],
    "uint decant_vf_log(uint a) { return asuint(log2(asfloat(a))); }"
);
helper!(
    SIN,
    Ty::U32,
    "decant_vf_sin",
    &[],
    "uint decant_vf_sin(uint a) { float x = asfloat(a); return asuint(sin(6.2831855f * (x - round(x)))); }"
);
helper!(
    COS,
    Ty::U32,
    "decant_vf_cos",
    &[],
    "uint decant_vf_cos(uint a) { float x = asfloat(a); return asuint(cos(6.2831855f * (x - round(x)))); }"
);
// fract(inf) is not a number (Mesa aco_select_nir_alu.cpp), the result stays below 1.
helper!(
    FRACT,
    Ty::U32,
    "decant_vf_fract",
    &[&NAN],
    "uint decant_vf_fract(uint a) { if (decant_vf_nan(a)) return a | 0x400000u; if ((a & 0x7fffffffu) == 0x7f800000u) return 0x7fc00000u; float x = asfloat(a); float f = x - floor(x); return (f >= 1.0f) ? 0x3f7fffffu : asuint(f); }"
);
helper!(
    TRUNC,
    Ty::U32,
    "decant_vf_trunc",
    &[],
    "uint decant_vf_trunc(uint a) { return asuint(trunc(asfloat(a))); }"
);
helper!(
    FLOOR,
    Ty::U32,
    "decant_vf_floor",
    &[],
    "uint decant_vf_floor(uint a) { return asuint(floor(asfloat(a))); }"
);
helper!(
    CEIL,
    Ty::U32,
    "decant_vf_ceil",
    &[],
    "uint decant_vf_ceil(uint a) { return asuint(ceil(asfloat(a))); }"
);
// HLSL round is round half to even (DXIL Round_ne), as v_rndne.
helper!(
    RNDNE,
    Ty::U32,
    "decant_vf_rndne",
    &[],
    "uint decant_vf_rndne(uint a) { return asuint(round(asfloat(a))); }"
);
// Float to integer: toward zero, saturating, zero for not a number.
helper!(
    CVT_I32,
    Ty::U32,
    "decant_vf_cvt_i32_f32",
    &[],
    "uint decant_vf_cvt_i32_f32(uint a) { float x = asfloat(a); if (!(x == x)) return 0u; if (x >= 2147483648.0f) return 0x7fffffffu; if (x <= -2147483648.0f) return 0x80000000u; return uint(int(x)); }"
);
helper!(
    CVT_U32,
    Ty::U32,
    "decant_vf_cvt_u32_f32",
    &[],
    "uint decant_vf_cvt_u32_f32(uint a) { float x = asfloat(a); if (!(x > 0.0f)) return 0u; if (x >= 4294967296.0f) return 0xffffffffu; return uint(x); }"
);
helper!(
    CVT_F32_I32,
    Ty::U32,
    "decant_vf_cvt_f32_i32",
    &[],
    "uint decant_vf_cvt_f32_i32(uint a) { return asuint(float(int(a))); }"
);
// Unsigned to float in two exact halves and one rounding add: WARP was measured to convert
// values from 2^31 up without nearest even rounding.
helper!(
    CVT_F32_U32,
    Ty::U32,
    "decant_vf_cvt_f32_u32",
    &[],
    "uint decant_vf_cvt_f32_u32(uint a) { return asuint(float(a >> 16u) * 65536.0f + float(a & 0xffffu)); }"
);
// Float to a byte inserted into a register: rounded per MODE (Mesa aco_insert_fp_mode.cpp lists
// it under the f32 round mode, nearest even here), saturated to 0..255, not a number to 0, at the
// byte S1[1:0] of S2 (Mesa aco_lower_to_hw_instr.cpp builds byte constants with it).
helper!(
    CVT_PK_U8,
    Ty::U32,
    "decant_vf_cvt_pk_u8",
    &[],
    "uint decant_vf_cvt_pk_u8(uint a, uint b, uint c) { float x = asfloat(a); uint v = (x > 0.0f) ? uint(min(round(x), 255.0f)) : 0u; uint sh = (b & 3u) * 8u; return (c & ~(0xffu << sh)) | (v << sh); }"
);
// 16 bit float to 32 bit, exact, from bits: denormals become normals, a not a number keeps its
// payload.
helper!(
    H2F,
    Ty::U32,
    "decant_vf_h2f",
    &[],
    "uint decant_vf_h2f(uint h) { uint s = (h & 0x8000u) << 16; uint e = (h >> 10) & 0x1fu; uint m = h & 0x3ffu; if (e == 31u) return s | 0x7f800000u | (m << 13); if (e != 0u) return s | ((e + 112u) << 23) | (m << 13); if (m == 0u) return s; uint sh = 10u - firstbithigh(m); return s | ((113u - sh) << 23) | (((m << sh) & 0x3ffu) << 13); }"
);
// 32 bit float to 16 bit, round to nearest even, from bits. Overflow gives infinity, a not a
// number stays one with the top of its payload and the quiet bit.
helper!(
    F2H,
    Ty::U32,
    "decant_vf_f2h",
    &[],
    "uint decant_vf_f2h(uint f) { uint s = (f >> 16) & 0x8000u; uint a = f & 0x7fffffffu; if (a > 0x7f800000u) return s | 0x7e00u | ((a >> 13) & 0x3ffu); if (a >= 0x477ff000u) return s | 0x7c00u; if (a >= 0x38800000u) { uint r = a - 0x38000000u; return s | ((r + 0xfffu + ((r >> 13) & 1u)) >> 13); } if (a <= 0x33000000u) return s; uint m = (a & 0x7fffffu) | 0x800000u; uint sh = 126u - (a >> 23); uint q = m >> sh; uint rem = m & ((1u << sh) - 1u); uint half = 1u << (sh - 1u); return s | (q + (((rem > half) || (rem == half && (q & 1u) != 0u)) ? 1u : 0u)); }"
);
// 32 bit float to 16 bit, toward zero: a finite value too large for 16 bits stays at the largest
// finite one.
helper!(
    F2H_RTZ,
    Ty::U32,
    "decant_vf_f2h_rtz",
    &[],
    "uint decant_vf_f2h_rtz(uint f) { uint s = (f >> 16) & 0x8000u; uint m = f & 0x7fffffu; int e = int((f >> 23) & 0xffu); if (e == 255) return s | ((m != 0u) ? (0x7e00u | (m >> 13)) : 0x7c00u); e = e - 112; if (e >= 31) return s | 0x7bffu; if (e <= 0) { if (e < -10) return s; return s | ((m | 0x800000u) >> uint(14 - e)); } return s | (uint(e) << 10) | (m >> 13); }"
);
helper!(
    PKRTZ,
    Ty::U32,
    "decant_vf_cvt_pkrtz",
    &[&F2H_RTZ],
    "uint decant_vf_cvt_pkrtz(uint a, uint b) { return decant_vf_f2h_rtz(a) | (decant_vf_f2h_rtz(b) << 16); }"
);
// The cube instructions take the direction (x, y, z). The major axis is z on a tie with x or y,
// then y. Rows of semantics-arith.tsv, checked against Mesa ac_nir_lower_tex_coords.c. The
// inputs are flushed like those of every f32 operation in flush mode: S and T pass an input
// through, which a device would otherwise return as a denormal.
helper!(
    CUBEID,
    Ty::U32,
    "decant_vf_cubeid",
    &[&FTZ],
    "uint decant_vf_cubeid(uint a, uint b, uint c) { float x = asfloat(decant_vf_ftz(a)); float y = asfloat(decant_vf_ftz(b)); float z = asfloat(decant_vf_ftz(c)); float r; if (abs(z) >= abs(x) && abs(z) >= abs(y)) r = (z < 0.0f) ? 5.0f : 4.0f; else if (abs(y) >= abs(x)) r = (y < 0.0f) ? 3.0f : 2.0f; else r = (x < 0.0f) ? 1.0f : 0.0f; return asuint(r); }"
);
helper!(
    CUBESC,
    Ty::U32,
    "decant_vf_cubesc",
    &[&FTZ],
    "uint decant_vf_cubesc(uint a, uint b, uint c) { float x = asfloat(decant_vf_ftz(a)); float y = asfloat(decant_vf_ftz(b)); float z = asfloat(decant_vf_ftz(c)); float r; if (abs(z) >= abs(x) && abs(z) >= abs(y)) r = (z < 0.0f) ? -x : x; else if (abs(y) >= abs(x)) r = x; else r = (x < 0.0f) ? z : -z; return asuint(r); }"
);
helper!(
    CUBETC,
    Ty::U32,
    "decant_vf_cubetc",
    &[&FTZ],
    "uint decant_vf_cubetc(uint a, uint b, uint c) { float x = asfloat(decant_vf_ftz(a)); float y = asfloat(decant_vf_ftz(b)); float z = asfloat(decant_vf_ftz(c)); float r; if (abs(z) >= abs(x) && abs(z) >= abs(y)) r = -y; else if (abs(y) >= abs(x)) r = (y < 0.0f) ? -z : z; else r = -y; return asuint(r); }"
);
helper!(
    CUBEMA,
    Ty::U32,
    "decant_vf_cubema",
    &[&FTZ],
    "uint decant_vf_cubema(uint a, uint b, uint c) { float x = asfloat(decant_vf_ftz(a)); float y = asfloat(decant_vf_ftz(b)); float z = asfloat(decant_vf_ftz(c)); float r; if (abs(z) >= abs(x) && abs(z) >= abs(y)) r = z; else if (abs(y) >= abs(x)) r = y; else r = x; return asuint(2.0f * r); }"
);

// The 64 bit float instructions work on register pairs: the helpers take the halves and give
// the result as one 64 bit integer, split again by the caller. The f32 side of a conversion is
// flushed explicitly: devices were measured to keep a denormal there (both directions on WARP,
// the narrowing on an RTX 3090) although the program runs in the f32 flush mode.
helper!(
    HI64,
    Ty::U32,
    "decant_vf_hi64",
    &[],
    "uint decant_vf_hi64(uint64_t x) { return uint(x >> 32); }"
);
helper!(
    ADD_F64,
    Ty::U64,
    "decant_vf_add_f64",
    &[],
    "uint64_t decant_vf_add_f64(uint a, uint b, uint c, uint d) { uint lo; uint hi; asuint(asdouble(a, b) + asdouble(c, d), lo, hi); return (uint64_t(hi) << 32) | uint64_t(lo); }"
);
helper!(
    MUL_F64,
    Ty::U64,
    "decant_vf_mul_f64",
    &[],
    "uint64_t decant_vf_mul_f64(uint a, uint b, uint c, uint d) { uint lo; uint hi; asuint(asdouble(a, b) * asdouble(c, d), lo, hi); return (uint64_t(hi) << 32) | uint64_t(lo); }"
);
helper!(
    FMA_F64,
    Ty::U64,
    "decant_vf_fma_f64",
    &[],
    "uint64_t decant_vf_fma_f64(uint a, uint b, uint c, uint d, uint e, uint f) { uint lo; uint hi; asuint(fma(asdouble(a, b), asdouble(c, d), asdouble(e, f)), lo, hi); return (uint64_t(hi) << 32) | uint64_t(lo); }"
);
helper!(
    CVT_F64_F32,
    Ty::U64,
    "decant_vf_cvt_f64_f32",
    &[&FTZ],
    "uint64_t decant_vf_cvt_f64_f32(uint a) { uint lo; uint hi; asuint(double(asfloat(decant_vf_ftz(a))), lo, hi); return (uint64_t(hi) << 32) | uint64_t(lo); }"
);
helper!(
    CVT_F32_F64,
    Ty::U32,
    "decant_vf_cvt_f32_f64",
    &[&FTZ],
    "uint decant_vf_cvt_f32_f64(uint a, uint b) { return decant_vf_ftz(asuint(float(asdouble(a, b)))); }"
);

fn call(h: &'static Helper, args: Vec<Expr>) -> Expr {
    Expr::Call(h, args)
}

fn k(v: u32) -> Expr {
    Expr::U32(v)
}

fn bin(o: Bin, a: Expr, b: Expr) -> Expr {
    Expr::bin(o, a, b)
}

/// The modifiers of an instruction. Instructions without the two word encoding have none.
#[derive(Clone, Default)]
pub(crate) struct Mods {
    pub abs: u8,
    pub neg: u8,
    pub opsel: u8,
    pub omod: u8,
    pub clamp: bool,
    /// The value of the first source when the caller resolved it: the lane rules give the value
    /// a data parallel (`dpp`) form reads from another lane. The modifiers still apply.
    pub src0: Option<Expr>,
}

/// The modifiers, or a refusal for the data parallel forms: the lane rules resolve those and
/// call [`lift_with_src0`].
pub(crate) fn mods(insn: &Insn) -> Result<Mods, Refusal> {
    match insn.form {
        Form::Dpp16 | Form::Dpp8 => return Err(form(insn, "dpp")),
        Form::Plain | Form::AsVop3 | Form::Sdwa => {}
    }
    Ok(word_mods(insn))
}

/// The modifiers of whatever encoding the instruction has.
pub(crate) fn word_mods(insn: &Insn) -> Mods {
    match insn.ext {
        // The 16 lane data parallel word: NEG and ABS of source 0 in bits 20 and 21, of source 1
        // in bits 22 and 23. No output modifiers.
        Ext::Dpp(x) if insn.form == Form::Dpp16 => Mods {
            neg: (((x >> 20) & 1) | (((x >> 22) & 1) << 1)) as u8,
            abs: (((x >> 21) & 1) | (((x >> 23) & 1) << 1)) as u8,
            ..Mods::default()
        },
        // The sub dword word: NEG and ABS of source 0 in bits 20 and 21, of source 1 in bits 28
        // and 29, CLAMP in bit 13 and OMOD in bits 15:14. A compare has its result register in
        // bits 15:8 instead, so neither output modifier.
        Ext::Sdwa(x) => {
            let compare = insn.format() == crate::tables::Format::Vopc;
            Mods {
                abs: (((x >> 21) & 1) | (((x >> 29) & 1) << 1)) as u8,
                neg: (((x >> 20) & 1) | (((x >> 28) & 1) << 1)) as u8,
                opsel: 0,
                omod: if compare { 0 } else { ((x >> 14) & 3) as u8 },
                clamp: !compare && (x >> 13) & 1 != 0,
                src0: None,
            }
        }
        Ext::Valu {
            abs,
            neg,
            opsel,
            omod,
            clamp,
        } => Mods {
            abs,
            neg,
            opsel,
            omod,
            clamp,
            src0: None,
        },
        _ => Mods::default(),
    }
}

/// The first source before modifiers: the caller's value, or the operand.
pub(crate) fn raw(insn: &Insn, m: &Mods, i: usize) -> Result<Expr, Refusal> {
    match (&m.src0, i) {
        (Some(e), 0) => Ok(e.clone()),
        _ => bits(insn, i),
    }
}

/// The 16 bit pattern of an inline float constant.
pub(crate) fn half_const(c: FloatConst) -> u32 {
    match c {
        FloatConst::Half => 0x3800,
        FloatConst::MinusHalf => 0xb800,
        FloatConst::One => 0x3c00,
        FloatConst::MinusOne => 0xbc00,
        FloatConst::Two => 0x4000,
        FloatConst::MinusTwo => 0xc000,
        FloatConst::Four => 0x4400,
        FloatConst::MinusFour => 0xc400,
        FloatConst::InvTwoPi => 0x3118,
    }
}

/// Source `i` of a 32 bit float operation with ABS and NEG applied. In the sub dword form the
/// selected byte or word, extended, is the operand the modifiers act on.
pub(crate) fn f32_src(insn: &Insn, m: &Mods, i: usize) -> Result<Expr, Refusal> {
    let e = sdwa_src(insn, i, raw(insn, m, i)?);
    Ok(sign_mods(e, m, i, 0x8000_0000))
}

fn sign_mods(mut e: Expr, m: &Mods, i: usize, sign: u32) -> Expr {
    if (m.abs >> i) & 1 != 0 {
        e = bin(Bin::And, e, k(sign - 1));
    }
    if (m.neg >> i) & 1 != 0 {
        e = bin(Bin::Xor, e, k(sign));
    }
    e
}

/// Source `i` of a 16 bit float operation: the half `hi` selects, in the low 16 bits, with ABS
/// and NEG applied. In the sub dword form the word select picks the half (a whole dword select
/// reads the low half).
pub(crate) fn f16_src(insn: &Insn, m: &Mods, i: usize, hi: bool) -> Result<Expr, Refusal> {
    let given = m.src0.is_some() && i == 0;
    let e = match insn.src[i] {
        _ if insn.form == Form::Sdwa => {
            bin(Bin::And, sdwa_src(insn, i, raw(insn, m, i)?), k(0xffff))
        }
        // An inline constant is a 32 bit value: a float one is the 16 bit pattern with a zero
        // high half, an integer one is sign extended (LLVM AMDGPUBaseInfo.cpp,
        // getInlineEncodingV216: "the actual HW behavior").
        Operand::Float(c) if !given => k(if hi { 0 } else { half_const(c) }),
        Operand::Int(v) if !given => k(match (hi, v < 0) {
            (false, _) => u32::from(v as i16 as u16),
            (true, true) => 0xffff,
            (true, false) => 0,
        }),
        Operand::Lit(v) if !given => k(if hi { v >> 16 } else { v & 0xffff }),
        _ => {
            let r = raw(insn, m, i)?;
            if hi {
                bin(Bin::LShr, r, k(16))
            } else {
                bin(Bin::And, r, k(0xffff))
            }
        }
    };
    Ok(sign_mods(e, m, i, 0x8000))
}

/// The output modifier and the clamp bit on a 32 bit float result.
fn out32(m: &Mods, mut v: Expr) -> Expr {
    match m.omod {
        0 => {}
        1 => v = bin(Bin::FMul, v, k(2.0f32.to_bits())),
        2 => v = bin(Bin::FMul, v, k(4.0f32.to_bits())),
        _ => v = bin(Bin::FMul, v, k(0.5f32.to_bits())),
    }
    if m.clamp {
        v = call(&SAT, vec![v]);
    }
    v
}

/// The output modifier and the clamp bit on a 16 bit float result (in the low bits).
fn out16(m: &Mods, v: Expr) -> Expr {
    if m.omod == 0 && !m.clamp {
        return v;
    }
    let mut f = call(&H2F, vec![v]);
    match m.omod {
        0 => {}
        1 => f = bin(Bin::FMul, f, k(2.0f32.to_bits())),
        2 => f = bin(Bin::FMul, f, k(4.0f32.to_bits())),
        _ => f = bin(Bin::FMul, f, k(0.5f32.to_bits())),
    }
    if m.clamp {
        f = call(&SAT, vec![f]);
    }
    call(&F2H, vec![f])
}

fn vdst(insn: &Insn) -> Result<u8, Refusal> {
    match insn.dst {
        Operand::Reg(Reg::V(d)) => Ok(d),
        _ => Err(form(insn, "operand")),
    }
}

/// Writes a 32 bit result, placed as the sub dword word says in that form.
fn write32(cx: &mut Lifter, insn: &Insn, v: Expr) -> Result<(), Refusal> {
    let old = Expr::Var(Var::V(vdst(insn)?));
    let v = sdwa_dst(insn, old, v).map_err(|_| form(insn, "sdwa-dst"))?;
    set_vdst(cx, insn, v)
}

/// Writes a 16 bit result into one half of the destination, the other half kept. In the sub
/// dword form the destination select and the unused bits rule decide instead.
fn write_half(cx: &mut Lifter, insn: &Insn, hi: bool, h: Expr) -> Result<(), Refusal> {
    let d = vdst(insn)?;
    let old = Expr::Var(Var::V(d));
    if insn.form == Form::Sdwa {
        let v = sdwa_dst(insn, old, h).map_err(|_| form(insn, "sdwa-dst"))?;
        return set_vdst(cx, insn, v);
    }
    let value = if hi {
        bin(
            Bin::Or,
            bin(Bin::And, old, k(0xffff)),
            bin(Bin::Shl, h, k(16)),
        )
    } else {
        bin(Bin::Or, bin(Bin::And, old, k(0xffff_0000)), h)
    };
    set_vdst(cx, insn, value)
}

/// The 16 bit instructions whose two word form takes `op_sel` on GFX10 (Mesa `can_use_opsel`).
fn takes_opsel(op: Op) -> bool {
    matches!(
        op,
        Op::v_fma_f16
            | Op::v_med3_f16
            | Op::v_min3_f16
            | Op::v_max3_f16
            | Op::v_div_fixup_f16
            | Op::v_pack_b32_f16
    )
}

/// A 32 bit float operation of 32 bit float sources.
fn f32_value(
    op: Op,
    s: &dyn Fn(usize) -> Result<Expr, Refusal>,
    acc: Expr,
) -> Result<Option<Expr>, Refusal> {
    let one =
        |h: &'static Helper| -> Result<Option<Expr>, Refusal> { Ok(Some(call(h, vec![s(0)?]))) };
    let two = |h: &'static Helper| -> Result<Option<Expr>, Refusal> {
        Ok(Some(call(h, vec![s(0)?, s(1)?])))
    };
    let three = |h: &'static Helper| -> Result<Option<Expr>, Refusal> {
        Ok(Some(call(h, vec![s(0)?, s(1)?, s(2)?])))
    };
    Ok(match op {
        Op::v_add_f32 => Some(bin(Bin::FAdd, s(0)?, s(1)?)),
        Op::v_sub_f32 => Some(bin(Bin::FSub, s(0)?, s(1)?)),
        Op::v_subrev_f32 => Some(bin(Bin::FSub, s(1)?, s(0)?)),
        Op::v_mul_f32 => Some(bin(Bin::FMul, s(0)?, s(1)?)),
        Op::v_mul_legacy_f32 => return two(&MUL_LEGACY),
        Op::v_min_f32 => return two(&MIN),
        Op::v_max_f32 => return two(&MAX),
        Op::v_min3_f32 => return three(&MIN3),
        Op::v_max3_f32 => return three(&MAX3),
        Op::v_med3_f32 => return three(&MED3),
        Op::v_fma_f32 => return three(&FMA),
        Op::v_fma_legacy_f32 => return three(&FMA_LEGACY),
        Op::v_mad_f32 => return three(&MAD),
        Op::v_fmac_f32 => Some(call(&FMA, vec![s(0)?, s(1)?, acc])),
        Op::v_fmac_legacy_f32 => Some(call(&FMA_LEGACY, vec![s(0)?, s(1)?, acc])),
        Op::v_mac_f32 => Some(call(&MAD, vec![s(0)?, s(1)?, acc])),
        Op::v_fmamk_f32 => Some(call(&FMA, vec![s(0)?, s(2)?, s(1)?])),
        Op::v_fmaak_f32 => return three(&FMA),
        Op::v_madmk_f32 => Some(call(&MAD, vec![s(0)?, s(2)?, s(1)?])),
        Op::v_madak_f32 => return three(&MAD),
        Op::v_rcp_f32 | Op::v_rcp_iflag_f32 => return one(&RCP),
        Op::v_rsq_f32 => return one(&RSQ),
        Op::v_sqrt_f32 => return one(&SQRT),
        Op::v_exp_f32 => return one(&EXP),
        Op::v_log_f32 => return one(&LOG),
        Op::v_sin_f32 => return one(&SIN),
        Op::v_cos_f32 => return one(&COS),
        Op::v_fract_f32 => return one(&FRACT),
        Op::v_trunc_f32 => return one(&TRUNC),
        Op::v_floor_f32 => return one(&FLOOR),
        Op::v_ceil_f32 => return one(&CEIL),
        Op::v_rndne_f32 => return one(&RNDNE),
        Op::v_cubeid_f32 => return three(&CUBEID),
        Op::v_cubesc_f32 => return three(&CUBESC),
        Op::v_cubetc_f32 => return three(&CUBETC),
        Op::v_cubema_f32 => return three(&CUBEMA),
        _ => None,
    })
}

/// A 16 bit float operation: the sources are converted to 32 bit, the result rounded once.
fn f16_value(
    op: Op,
    s: &dyn Fn(usize) -> Result<Expr, Refusal>,
    acc: Expr,
) -> Result<Option<Expr>, Refusal> {
    let w = |i: usize| -> Result<Expr, Refusal> { Ok(call(&H2F, vec![s(i)?])) };
    let wide = |e: Expr| call(&F2H, vec![e]);
    let one = |h: &'static Helper| -> Result<Option<Expr>, Refusal> {
        Ok(Some(wide(call(h, vec![w(0)?]))))
    };
    Ok(Some(match op {
        Op::v_add_f16 => wide(bin(Bin::FAdd, w(0)?, w(1)?)),
        Op::v_sub_f16 => wide(bin(Bin::FSub, w(0)?, w(1)?)),
        Op::v_subrev_f16 => wide(bin(Bin::FSub, w(1)?, w(0)?)),
        Op::v_mul_f16 => wide(bin(Bin::FMul, w(0)?, w(1)?)),
        Op::v_fma_f16 => wide(call(&FMA, vec![w(0)?, w(1)?, w(2)?])),
        Op::v_fmac_f16 => wide(call(&FMA, vec![w(0)?, w(1)?, call(&H2F, vec![acc])])),
        Op::v_fmamk_f16 => wide(call(&FMA, vec![w(0)?, w(2)?, w(1)?])),
        Op::v_fmaak_f16 => wide(call(&FMA, vec![w(0)?, w(1)?, w(2)?])),
        // The minimum family is exact on the widened values and widening keeps the order.
        Op::v_min_f16 => wide(call(&MIN, vec![w(0)?, w(1)?])),
        Op::v_max_f16 => wide(call(&MAX, vec![w(0)?, w(1)?])),
        Op::v_min3_f16 => wide(call(&MIN3, vec![w(0)?, w(1)?, w(2)?])),
        Op::v_max3_f16 => wide(call(&MAX3, vec![w(0)?, w(1)?, w(2)?])),
        Op::v_med3_f16 => wide(call(&MED3, vec![w(0)?, w(1)?, w(2)?])),
        Op::v_rcp_f16 => return one(&RCP),
        Op::v_rsq_f16 => return one(&RSQ),
        Op::v_sqrt_f16 => return one(&SQRT),
        Op::v_exp_f16 => return one(&EXP),
        Op::v_log_f16 => return one(&LOG),
        Op::v_sin_f16 => return one(&SIN),
        Op::v_cos_f16 => return one(&COS),
        Op::v_fract_f16 => return one(&FRACT),
        Op::v_trunc_f16 => return one(&TRUNC),
        Op::v_floor_f16 => return one(&FLOOR),
        Op::v_ceil_f16 => return one(&CEIL),
        Op::v_rndne_f16 => return one(&RNDNE),
        _ => return Ok(None),
    }))
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    lift_in(cx, insn, None)
}

/// Lifts a float instruction whose first source the caller has resolved: the lane rules call
/// this for the data parallel forms with the value read from the source lane. ABS and NEG of the
/// data parallel word still apply. Whether a lane writes at all (bound control, row and bank
/// masks) is the caller's to decide. `Ok(false)` when the instruction is no float rule's.
pub fn lift_with_src0(cx: &mut Lifter, insn: &Insn, src0: Expr) -> Result<bool, Refusal> {
    lift_in(cx, insn, Some(src0))
}

fn lift_in(cx: &mut Lifter, insn: &Insn, given: Option<Expr>) -> Result<bool, Refusal> {
    let op = insn.op;
    let info = op.info();
    let Some(def) = info.defs.first() else {
        return Ok(false);
    };
    let float_def = def.base == crate::tables::Base::Float;
    let float_src = info
        .srcs
        .iter()
        .any(|s| s.base == crate::tables::Base::Float);
    if !float_def && !float_src {
        return Ok(false);
    }
    // Ops of other files with float slots: interpolation (pixel stage), packed math, compares.
    if matches!(
        info.format,
        crate::tables::Format::Vintrp | crate::tables::Format::Vopc
    ) || op.name().starts_with("v_interp")
    {
        return Ok(false);
    }
    match op {
        Op::v_fma_mix_f32 | Op::v_fma_mixlo_f16 | Op::v_fma_mixhi_f16 => return mix(cx, insn),
        Op::v_pack_b32_f16 => return pack(cx, insn),
        Op::v_pk_add_f16
        | Op::v_pk_mul_f16
        | Op::v_pk_fma_f16
        | Op::v_pk_min_f16
        | Op::v_pk_max_f16 => {
            return packed(cx, insn);
        }
        _ => {}
    }
    if info.format == crate::tables::Format::Vop3p {
        return Ok(false);
    }
    let m = match given {
        None => mods(insn)?,
        Some(e) => Mods {
            src0: Some(e),
            ..word_mods(insn)
        },
    };
    if op == Op::v_cvt_pkrtz_f16_f32 {
        return convert(cx, insn, &m);
    }
    let acc = || -> Result<Expr, Refusal> { Ok(Expr::Var(Var::V(vdst(insn)?))) };
    let def_bits = def.bits;
    let src_bits = info.srcs.first().map_or(32, |s| s.bits);
    match (def_bits, src_bits) {
        (32, 32)
            if float_def
                && info.srcs.iter().all(|s| {
                    s.base == crate::tables::Base::Float || s.fixed == crate::tables::Fixed::Imm
                }) =>
        {
            if m.opsel != 0 {
                return Err(form(insn, "opsel"));
            }
            let s = |i: usize| f32_src(insn, &m, i);
            let Some(v) = f32_value(op, &s, acc().unwrap_or(k(0)))? else {
                return Ok(false);
            };
            let v = out32(&m, v);
            write32(cx, insn, v)?;
            Ok(true)
        }
        (16, 16) => {
            if m.opsel != 0 && !takes_opsel(op) {
                return Err(form(insn, "opsel"));
            }
            let s = |i: usize| f16_src(insn, &m, i, (m.opsel >> i) & 1 != 0);
            let old = acc().map(|a| bin(Bin::And, a, k(0xffff)));
            let Some(v) = f16_value(op, &s, old.unwrap_or(k(0)))? else {
                return Ok(false);
            };
            let v = out16(&m, v);
            write_half(cx, insn, (m.opsel >> 3) & 1 != 0, v)?;
            Ok(true)
        }
        (64, _) | (_, 64) => double(cx, insn, &m),
        _ => convert(cx, insn, &m),
    }
}

/// The conversions.
fn convert(cx: &mut Lifter, insn: &Insn, m: &Mods) -> Result<bool, Refusal> {
    let op = insn.op;
    if m.opsel != 0 {
        return Err(form(insn, "opsel"));
    }
    let int_src = |i: usize| -> Result<Expr, Refusal> {
        if (m.abs | m.neg) >> i & 1 != 0 {
            return Err(form(insn, "absneg"));
        }
        Ok(sdwa_src(insn, i, bits(insn, i)?))
    };
    // Float results take the output modifiers, integer results take neither.
    let int_out = |v: Expr| -> Result<Expr, Refusal> {
        if m.omod != 0 {
            return Err(form(insn, "omod"));
        }
        // The float to integer conversions already saturate, so clamp changes nothing.
        Ok(v)
    };
    let value = match op {
        Op::v_cvt_f32_i32 => out32(m, call(&CVT_F32_I32, vec![int_src(0)?])),
        Op::v_cvt_f32_u32 => out32(m, call(&CVT_F32_U32, vec![int_src(0)?])),
        Op::v_cvt_f32_ubyte0
        | Op::v_cvt_f32_ubyte1
        | Op::v_cvt_f32_ubyte2
        | Op::v_cvt_f32_ubyte3 => {
            let n = match op {
                Op::v_cvt_f32_ubyte0 => 0,
                Op::v_cvt_f32_ubyte1 => 8,
                Op::v_cvt_f32_ubyte2 => 16,
                _ => 24,
            };
            let byte = bin(Bin::And, bin(Bin::LShr, int_src(0)?, k(n)), k(0xff));
            out32(m, call(&CVT_F32_U32, vec![byte]))
        }
        Op::v_cvt_i32_f32 => int_out(call(&CVT_I32, vec![f32_src(insn, m, 0)?]))?,
        Op::v_cvt_u32_f32 => int_out(call(&CVT_U32, vec![f32_src(insn, m, 0)?]))?,
        Op::v_cvt_pk_u8_f32 => int_out(call(
            &CVT_PK_U8,
            vec![f32_src(insn, m, 0)?, int_src(1)?, int_src(2)?],
        ))?,
        Op::v_cvt_f32_f16 => out32(m, call(&H2F, vec![f16_src(insn, m, 0, false)?])),
        Op::v_cvt_pkrtz_f16_f32 => {
            if m.omod != 0 || m.clamp {
                return Err(form(insn, if m.clamp { "clamp" } else { "omod" }));
            }
            call(&PKRTZ, vec![f32_src(insn, m, 0)?, f32_src(insn, m, 1)?])
        }
        Op::v_cvt_f16_f32 => {
            let h = out16(m, call(&F2H, vec![f32_src(insn, m, 0)?]));
            write_half(cx, insn, false, h)?;
            return Ok(true);
        }
        _ => return Ok(false),
    };
    write32(cx, insn, value)?;
    Ok(true)
}

/// Source `i` of a 64 bit float operation as its (low, high) halves, with ABS and NEG on the
/// sign bit of the high half. An inline float constant is the 64 bit float, an inline integer
/// is sign extended, a literal is the high half (LLVM AMDGPUBaseInfo.cpp, isValid32BitLiteral).
fn d_src(insn: &Insn, m: &Mods, i: usize) -> Result<(Expr, Expr), Refusal> {
    let (lo, hi) = match insn.src[i] {
        _ if i == 0 && m.src0.is_some() => return Err(form(insn, "operand")),
        Operand::Float(c) => {
            let v = match c {
                FloatConst::InvTwoPi => 0x3fc4_5f30_6dc9_c882u64,
                c => f64::from(c.f32()).to_bits(),
            };
            (k(v as u32), k((v >> 32) as u32))
        }
        Operand::Int(v) => {
            let v = i64::from(v) as u64;
            (k(v as u32), k((v >> 32) as u32))
        }
        Operand::Lit(v) => (k(0), k(v)),
        Operand::Reg(Reg::V(n)) => (Expr::Var(Var::V(n)), Expr::Var(Var::V(n + 1))),
        Operand::Reg(Reg::S(n)) => (Expr::Var(Var::S(n)), Expr::Var(Var::S(n + 1))),
        _ => return Err(form(insn, "operand")),
    };
    Ok((lo, sign_mods(hi, m, i, 0x8000_0000)))
}

/// The 64 bit float instructions the title uses.
fn double(cx: &mut Lifter, insn: &Insn, m: &Mods) -> Result<bool, Refusal> {
    if m.omod != 0 || m.clamp || m.opsel != 0 || insn.form == Form::Sdwa {
        return Err(form(insn, if m.clamp { "clamp" } else { "omod" }));
    }
    let wide = |h: &'static Helper, n: usize| -> Result<Expr, Refusal> {
        let mut args = Vec::new();
        for i in 0..n {
            let (lo, hi) = d_src(insn, m, i)?;
            args.push(lo);
            args.push(hi);
        }
        Ok(call(h, args))
    };
    let value = match insn.op {
        Op::v_add_f64_e64 => wide(&ADD_F64, 2)?,
        Op::v_mul_f64_e64 => wide(&MUL_F64, 2)?,
        Op::v_fma_f64 => wide(&FMA_F64, 3)?,
        Op::v_cvt_f64_f32 => call(&CVT_F64_F32, vec![f32_src(insn, m, 0)?]),
        Op::v_cvt_f32_f64 => {
            let v = call(&CVT_F32_F64, {
                let (lo, hi) = d_src(insn, m, 0)?;
                vec![lo, hi]
            });
            write32(cx, insn, v)?;
            return Ok(true);
        }
        _ => return Ok(false),
    };
    // Both halves come from one value, held in temporaries so that a destination that overlaps a
    // source reads the old registers.
    let d = vdst(insn)?;
    let (lo, hi) = (cx.tmp(), cx.tmp());
    cx.set(lo, Expr::un(crate::ir::Un::U64ToU32, value.clone()));
    cx.set(hi, call(&HI64, vec![value]));
    cx.set_lane(Var::V(d), Expr::Var(lo));
    cx.set_lane(Var::V(d + 1), Expr::Var(hi));
    Ok(true)
}

/// `v_pack_b32_f16 d, a, b`: `a` in the low half, `b` in the high half, each a 16 bit source
/// with ABS, NEG and `op_sel`.
fn pack(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let m = mods(insn)?;
    if m.omod != 0 || m.clamp || m.opsel & 8 != 0 {
        return Err(form(insn, if m.clamp { "clamp" } else { "omod" }));
    }
    let a = f16_src(insn, &m, 0, m.opsel & 1 != 0)?;
    let b = f16_src(insn, &m, 1, m.opsel & 2 != 0)?;
    set_vdst(cx, insn, bin(Bin::Or, a, bin(Bin::Shl, b, k(16))))?;
    Ok(true)
}

/// The packed 16 bit instructions: two independent 16 bit operations, one per half. For the low
/// result source `i` is the half `op_sel` bit `i` names, negated by NEG bit `i`; for the high
/// result the half `op_sel_hi` bit `i` names, negated by the high NEG bit `i` (the field the other
/// forms use for ABS). Clamp applies to both halves.
fn packed(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Ext::Valu {
        abs: neg_hi,
        neg,
        opsel,
        omod,
        clamp,
    } = insn.ext
    else {
        return Err(form(insn, "operand"));
    };
    let opsel_hi = omod | ((opsel >> 3) & 1) << 2;
    let scalar = match insn.op {
        Op::v_pk_add_f16 => Op::v_add_f16,
        Op::v_pk_mul_f16 => Op::v_mul_f16,
        Op::v_pk_fma_f16 => Op::v_fma_f16,
        Op::v_pk_min_f16 => Op::v_min_f16,
        _ => Op::v_max_f16,
    };
    let half = |sel: u8, negs: u8| -> Result<Expr, Refusal> {
        let m = Mods {
            neg: negs,
            ..Mods::default()
        };
        let s = |i: usize| f16_src(insn, &m, i, (sel >> i) & 1 != 0);
        let v = f16_value(scalar, &s, k(0))?.ok_or_else(|| form(insn, "operand"))?;
        Ok(out16(
            &Mods {
                clamp,
                ..Mods::default()
            },
            v,
        ))
    };
    let lo = half(opsel & 7, neg)?;
    let hi = half(opsel_hi, neg_hi)?;
    set_vdst(cx, insn, bin(Bin::Or, lo, bin(Bin::Shl, hi, k(16))))?;
    Ok(true)
}

/// The mixed precision multiply add. For each source, `op_sel_hi` says it is a 16 bit float
/// (the half `op_sel` names) and not a 32 bit one. The ABS bits sit where the packed forms have
/// their high NEG bits (Mesa `aco_ir.h`: `abs` and `neg_hi` share the field for `v_fma_mix`).
/// The 32 bit result is fused as [`FMA`] is. `mixlo` and `mixhi` round it to 16 bit and write
/// one half.
fn mix(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Ext::Valu {
        abs,
        neg,
        opsel,
        omod,
        clamp,
    } = insn.ext
    else {
        return Err(form(insn, "operand"));
    };
    // `omod` holds op_sel_hi bits 0 and 1, `opsel` bit 3 holds op_sel_hi bit 2.
    let hi_sel = omod | ((opsel >> 3) & 1) << 2;
    let m = Mods {
        abs,
        neg,
        ..Mods::default()
    };
    let src = |i: usize| -> Result<Expr, Refusal> {
        if (hi_sel >> i) & 1 != 0 {
            let h = f16_src(insn, &Mods::default(), i, (opsel >> i) & 1 != 0)?;
            Ok(sign_mods(call(&H2F, vec![h]), &m, i, 0x8000_0000))
        } else {
            f32_src(insn, &m, i)
        }
    };
    let mut v = call(&FMA, vec![src(0)?, src(1)?, src(2)?]);
    if clamp {
        v = call(&SAT, vec![v]);
    }
    match insn.op {
        Op::v_fma_mix_f32 => set_vdst(cx, insn, v)?,
        op => write_half(cx, insn, op == Op::v_fma_mixhi_f16, call(&F2H, vec![v]))?,
    }
    Ok(true)
}
