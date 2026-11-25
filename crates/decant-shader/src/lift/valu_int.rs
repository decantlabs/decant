//! Vector integer and bit instructions, moves and selects.
//!
//! The `rev` forms take their operands the other way round: `v_lshlrev_b32 d, a, b` is `b << a`.
//!
//! Every result is computed from the old register values before any register is written, so an
//! instruction whose destination is also a source is right. The sub dword encoding (`sdwa`) is
//! handled here for every instruction of this file: [`sdwa_src`] picks the byte or word of a
//! source and [`sdwa_dst`] places the result. The float rules can use both.

use super::Lifter;
use super::operand::{bits, fbits, form, mask, no_float_mods, plain, set_vdst};
use crate::decode::{Ext, Form, Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Guard, Helper, Stmt, Ty, Un, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

macro_rules! helper {
    ($id:ident, $name:literal, $needs:expr, $text:literal) => {
        static $id: Helper = Helper {
            name: $name,
            result: Ty::U32,
            text: $text,
            needs: $needs,
        };
    };
}

helper!(
    MUL_HI_U32,
    "decant_vi_mul_hi_u32",
    &[],
    "uint decant_vi_mul_hi_u32(uint a, uint b) { uint a0 = a & 0xffffu; uint a1 = a >> 16; uint b0 = b & 0xffffu; uint b1 = b >> 16; uint t = a1 * b0 + ((a0 * b0) >> 16); uint u = a0 * b1 + (t & 0xffffu); return a1 * b1 + (t >> 16) + (u >> 16); }"
);
helper!(
    MUL_HI_I32,
    "decant_vi_mul_hi_i32",
    &[&MUL_HI_U32],
    "uint decant_vi_mul_hi_i32(uint a, uint b) { return decant_vi_mul_hi_u32(a, b) - ((int(a) < 0) ? b : 0u) - ((int(b) < 0) ? a : 0u); }"
);
helper!(
    MIN_I32,
    "decant_vi_min_i32",
    &[],
    "uint decant_vi_min_i32(uint a, uint b) { return uint(min(int(a), int(b))); }"
);
helper!(
    MAX_I32,
    "decant_vi_max_i32",
    &[],
    "uint decant_vi_max_i32(uint a, uint b) { return uint(max(int(a), int(b))); }"
);
helper!(
    MIN_U32,
    "decant_vi_min_u32",
    &[],
    "uint decant_vi_min_u32(uint a, uint b) { return min(a, b); }"
);
helper!(
    MAX_U32,
    "decant_vi_max_u32",
    &[],
    "uint decant_vi_max_u32(uint a, uint b) { return max(a, b); }"
);
helper!(
    MIN3_I32,
    "decant_vi_min3_i32",
    &[],
    "uint decant_vi_min3_i32(uint a, uint b, uint c) { return uint(min(int(a), min(int(b), int(c)))); }"
);
helper!(
    MAX3_I32,
    "decant_vi_max3_i32",
    &[],
    "uint decant_vi_max3_i32(uint a, uint b, uint c) { return uint(max(int(a), max(int(b), int(c)))); }"
);
helper!(
    MIN3_U32,
    "decant_vi_min3_u32",
    &[],
    "uint decant_vi_min3_u32(uint a, uint b, uint c) { return min(a, min(b, c)); }"
);
helper!(
    MAX3_U32,
    "decant_vi_max3_u32",
    &[],
    "uint decant_vi_max3_u32(uint a, uint b, uint c) { return max(a, max(b, c)); }"
);
helper!(
    MED3_I32,
    "decant_vi_med3_i32",
    &[],
    "uint decant_vi_med3_i32(uint a, uint b, uint c) { int x = int(a); int y = int(b); int z = int(c); return uint(max(min(x, y), min(max(x, y), z))); }"
);
helper!(
    MED3_U32,
    "decant_vi_med3_u32",
    &[],
    "uint decant_vi_med3_u32(uint a, uint b, uint c) { return max(min(a, b), min(max(a, b), c)); }"
);
helper!(
    BFE_U32,
    "decant_vi_bfe_u32",
    &[],
    "uint decant_vi_bfe_u32(uint a, uint b, uint c) { uint w = c & 31u; return (w == 0u) ? 0u : ((a >> (b & 31u)) & ((1u << w) - 1u)); }"
);
helper!(
    BFE_I32,
    "decant_vi_bfe_i32",
    &[],
    "uint decant_vi_bfe_i32(uint a, uint b, uint c) { uint w = c & 31u; int v = int(a) >> (b & 31u); return (w == 0u) ? 0u : uint((v << (32u - w)) >> (32u - w)); }"
);
helper!(
    BFI,
    "decant_vi_bfi",
    &[],
    "uint decant_vi_bfi(uint a, uint b, uint c) { return (a & b) | (~a & c); }"
);
helper!(
    BFM,
    "decant_vi_bfm",
    &[],
    "uint decant_vi_bfm(uint a, uint b) { return ((1u << (a & 31u)) - 1u) << (b & 31u); }"
);
helper!(
    BCNT,
    "decant_vi_bcnt",
    &[],
    "uint decant_vi_bcnt(uint a, uint b) { return countbits(a) + b; }"
);
helper!(
    BFREV,
    "decant_vi_bfrev",
    &[],
    "uint decant_vi_bfrev(uint a) { return reversebits(a); }"
);
helper!(
    FFBH_U32,
    "decant_vi_ffbh_u32",
    &[],
    "uint decant_vi_ffbh_u32(uint a) { return (a == 0u) ? 0xffffffffu : (31u - firstbithigh(a)); }"
);
helper!(
    FFBH_I32,
    "decant_vi_ffbh_i32",
    &[],
    "uint decant_vi_ffbh_i32(uint a) { return (a == 0u || a == 0xffffffffu) ? 0xffffffffu : (31u - uint(firstbithigh(int(a)))); }"
);
helper!(
    FFBL,
    "decant_vi_ffbl",
    &[],
    "uint decant_vi_ffbl(uint a) { return (a == 0u) ? 0xffffffffu : firstbitlow(a); }"
);
helper!(
    ALIGNBIT,
    "decant_vi_alignbit",
    &[],
    "uint decant_vi_alignbit(uint a, uint b, uint c) { uint n = c & 31u; return (n == 0u) ? b : ((b >> n) | (a << (32u - n))); }"
);
helper!(
    ALIGNBYTE,
    "decant_vi_alignbyte",
    &[],
    "uint decant_vi_alignbyte(uint a, uint b, uint c) { uint n = (c & 3u) * 8u; return (n == 0u) ? b : ((b >> n) | (a << (32u - n))); }"
);
helper!(
    PERM_BYTE,
    "decant_vi_perm_byte",
    &[],
    "uint decant_vi_perm_byte(uint a, uint b, uint s) { if (s < 4u) return (b >> (8u * s)) & 0xffu; if (s < 8u) return (a >> (8u * (s - 4u))) & 0xffu; if (s == 8u) return (b & 0x8000u) != 0u ? 0xffu : 0u; if (s == 9u) return (b & 0x80000000u) != 0u ? 0xffu : 0u; if (s == 10u) return (a & 0x8000u) != 0u ? 0xffu : 0u; if (s == 11u) return (a & 0x80000000u) != 0u ? 0xffu : 0u; if (s == 12u) return 0u; return 0xffu; }"
);
helper!(
    PERM,
    "decant_vi_perm",
    &[&PERM_BYTE],
    "uint decant_vi_perm(uint a, uint b, uint c) { return decant_vi_perm_byte(a, b, c & 0xffu) | (decant_vi_perm_byte(a, b, (c >> 8u) & 0xffu) << 8u) | (decant_vi_perm_byte(a, b, (c >> 16u) & 0xffu) << 16u) | (decant_vi_perm_byte(a, b, c >> 24u) << 24u); }"
);
helper!(
    SAD_U32,
    "decant_vi_sad_u32",
    &[],
    "uint decant_vi_sad_u32(uint a, uint b, uint c) { return ((a > b) ? (a - b) : (b - a)) + c; }"
);
helper!(
    SAD_U8,
    "decant_vi_sad_u8",
    &[],
    "uint decant_vi_sad_u8(uint a, uint b, uint c) { uint r = c; for (uint i = 0u; i < 4u; i++) { uint x = (a >> (8u * i)) & 0xffu; uint y = (b >> (8u * i)) & 0xffu; r += (x > y) ? (x - y) : (y - x); } return r; }"
);
helper!(
    LERP_U8,
    "decant_vi_lerp_u8",
    &[],
    "uint decant_vi_lerp_u8(uint a, uint b, uint c) { uint r = 0u; for (uint i = 0u; i < 4u; i++) { uint x = (a >> (8u * i)) & 0xffu; uint y = (b >> (8u * i)) & 0xffu; r |= (((x + y + ((c >> (8u * i)) & 1u)) >> 1u) & 0xffu) << (8u * i); } return r; }"
);

fn call(h: &'static Helper, args: Vec<Expr>) -> Expr {
    Expr::Call(h, args)
}

fn bin(o: Bin, a: Expr, b: Expr) -> Expr {
    Expr::bin(o, a, b)
}

fn k(v: u32) -> Expr {
    Expr::U32(v)
}

/// The low 24 bits of `e`, sign extended.
fn sext24(e: Expr) -> Expr {
    bin(Bin::AShr, bin(Bin::Shl, e, k(8)), k(8))
}

fn mask24(e: Expr) -> Expr {
    bin(Bin::And, e, k(0xff_ffff))
}

/// The raw sub dword word of the instruction.
fn sdwa_word(insn: &Insn) -> Option<u32> {
    match insn.ext {
        Ext::Sdwa(x) if insn.form == Form::Sdwa => Some(x),
        _ => None,
    }
}

/// Source `i` (0 or 1) of an instruction in the sub dword encoding: the selected byte or word,
/// zero or sign extended. `value` is the whole register. Other encodings pass `value` through.
pub fn sdwa_src(insn: &Insn, i: usize, value: Expr) -> Expr {
    let Some(x) = sdwa_word(insn) else {
        return value;
    };
    if i > 1 {
        return value;
    }
    let sel = (x >> (16 + 8 * i)) & 7;
    let sext = (x >> (19 + 8 * i)) & 1 != 0;
    // (width, bit offset)
    let (width, at) = match sel {
        0..=3 => (8, 8 * sel),
        4 => (16, 0),
        5 => (16, 16),
        _ => return value,
    };
    if sext {
        // Move the field to the top, then back with the sign.
        bin(
            Bin::AShr,
            bin(Bin::Shl, value, k(32 - width - at)),
            k(32 - width),
        )
    } else {
        let field = bin(Bin::LShr, value, k(at));
        if at + width == 32 {
            field
        } else {
            bin(Bin::And, field, k((1u32 << width) - 1))
        }
    }
}

/// The new value of the destination register of an instruction in the sub dword encoding:
/// `value` is the 32 bit result, `old` the register before. Other encodings pass `value` through.
pub fn sdwa_dst(insn: &Insn, old: Expr, value: Expr) -> Result<Expr, Refusal> {
    let Some(x) = sdwa_word(insn) else {
        return Ok(value);
    };
    let sel = (x >> 8) & 7;
    let unused = (x >> 11) & 3;
    let (width, at) = match sel {
        0..=3 => (8, 8 * sel),
        4 => (16, 0),
        5 => (16, 16),
        _ => return Ok(value),
    };
    let field = (1u32 << width) - 1;
    let placed = bin(Bin::Shl, bin(Bin::And, value.clone(), k(field)), k(at));
    match unused {
        0 => Ok(placed),
        2 => Ok(bin(Bin::Or, bin(Bin::And, old, k(!(field << at))), placed)),
        // The bits of the register above the field are the sign of the field.
        _ if at == 0 => Ok(bin(
            Bin::AShr,
            bin(Bin::Shl, value, k(32 - width)),
            k(32 - width),
        )),
        _ => Err(form_name("sdwa-dst")),
    }
}

fn form_name(what: &'static str) -> Refusal {
    Refusal::new(
        crate::refusal::RefusalKind::Form(Op::v_nop, what),
        String::new(),
    )
}

/// Refuses the modifiers of the sub dword word that an integer instruction cannot take: ABS,
/// NEG, the output modifier and the clamp bit.
fn sdwa_plain(insn: &Insn) -> Result<(), Refusal> {
    let Some(x) = sdwa_word(insn) else {
        return Ok(());
    };
    // CLAMP bit 13, OMOD bits 15:14, SRC0 NEG and ABS bits 20 and 21, SRC1 bits 28 and 29.
    if (x >> 13) & 7 != 0 {
        return Err(form(insn, "clamp"));
    }
    if (x >> 20) & 3 != 0 || (x >> 28) & 3 != 0 {
        return Err(form(insn, "absneg"));
    }
    Ok(())
}

/// Checks the two word modifiers an integer instruction cannot take. The sub dword word has its
/// own check.
fn integer_plain(insn: &Insn) -> Result<(), Refusal> {
    if insn.form == Form::Sdwa {
        return sdwa_plain(insn);
    }
    plain(insn)?;
    no_float_mods(insn)
}

/// The carry or borrow flavour of an add or subtract.
#[derive(Clone, Copy, PartialEq)]
enum Carry {
    /// No carry at all.
    None,
    /// A carry out, no carry in.
    Out,
    /// A carry in and a carry out.
    InOut,
}

fn known(op: Op) -> bool {
    matches!(
        op,
        Op::v_nop
            | Op::v_mov_b32
            | Op::v_not_b32
            | Op::v_bfrev_b32
            | Op::v_ffbh_u32
            | Op::v_ffbh_i32
            | Op::v_ffbl_b32
            | Op::v_swap_b32
            | Op::v_add_u32
            | Op::v_sub_u32
            | Op::v_subrev_u32
            | Op::v_add_co_u32_e64
            | Op::v_sub_co_u32_e64
            | Op::v_subrev_co_u32_e64
            | Op::v_addc_co_u32
            | Op::v_subb_co_u32
            | Op::v_subbrev_co_u32
            | Op::v_mul_lo_u32
            | Op::v_mul_lo_i32
            | Op::v_mul_hi_u32
            | Op::v_mul_hi_i32
            | Op::v_mul_u32_u24
            | Op::v_mul_i32_i24
            | Op::v_mul_hi_u32_u24
            | Op::v_mul_hi_i32_i24
            | Op::v_mad_u32_u24
            | Op::v_mad_i32_i24
            | Op::v_and_b32
            | Op::v_or_b32
            | Op::v_xor_b32
            | Op::v_xnor_b32
            | Op::v_lshlrev_b32
            | Op::v_lshrrev_b32
            | Op::v_ashrrev_i32
            | Op::v_min_i32
            | Op::v_max_i32
            | Op::v_min_u32
            | Op::v_max_u32
            | Op::v_min3_i32
            | Op::v_max3_i32
            | Op::v_min3_u32
            | Op::v_max3_u32
            | Op::v_med3_i32
            | Op::v_med3_u32
            | Op::v_bfe_u32
            | Op::v_bfe_i32
            | Op::v_bfi_b32
            | Op::v_bfm_b32
            | Op::v_bcnt_u32_b32
            | Op::v_alignbit_b32
            | Op::v_alignbyte_b32
            | Op::v_perm_b32
            | Op::v_sad_u32
            | Op::v_sad_u8
            | Op::v_lerp_u8
            | Op::v_xor3_b32
            | Op::v_or3_b32
            | Op::v_add3_u32
            | Op::v_and_or_b32
            | Op::v_lshl_or_b32
            | Op::v_lshl_add_u32
            | Op::v_add_lshl_u32
            | Op::v_xad_u32
            | Op::v_cndmask_b32
    )
}

/// The 16 bit two word instructions: the sources are halves of registers (`opsel` bit `i` takes
/// the high half of source `i`), the result is 16 bits, zero extended, or in the high half with
/// the low half kept when `opsel` bit 3 is set.
fn lift16(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let signed = matches!(
        insn.op,
        Op::v_ashrrev_i16_e64 | Op::v_max_i16_e64 | Op::v_min_i16_e64
    );
    let wanted = matches!(
        insn.op,
        Op::v_add_u16_e64
            | Op::v_sub_u16_e64
            | Op::v_mul_lo_u16_e64
            | Op::v_lshlrev_b16_e64
            | Op::v_lshrrev_b16_e64
            | Op::v_ashrrev_i16_e64
            | Op::v_max_u16_e64
            | Op::v_min_u16_e64
            | Op::v_max_i16_e64
            | Op::v_min_i16_e64
    );
    if !wanted {
        return Ok(false);
    }
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
    if abs != 0 || neg != 0 {
        return Err(form(insn, "absneg"));
    }
    if omod != 0 {
        return Err(form(insn, "omod"));
    }
    if clamp {
        return Err(form(insn, "clamp"));
    }
    let half = |i: usize| -> Result<Expr, Refusal> {
        let e = bits(insn, i)?;
        let hi = (opsel >> i) & 1 != 0;
        Ok(match (hi, signed) {
            (false, false) => bin(Bin::And, e, k(0xffff)),
            (true, false) => bin(Bin::LShr, e, k(16)),
            (false, true) => bin(Bin::AShr, bin(Bin::Shl, e, k(16)), k(16)),
            (true, true) => bin(Bin::AShr, e, k(16)),
        })
    };
    let (a, b) = (half(0)?, half(1)?);
    let value = match insn.op {
        Op::v_add_u16_e64 => bin(Bin::Add, a, b),
        Op::v_sub_u16_e64 => bin(Bin::Sub, a, b),
        Op::v_mul_lo_u16_e64 => bin(Bin::Mul, a, b),
        Op::v_lshlrev_b16_e64 => bin(Bin::Shl, b, bin(Bin::And, a, k(15))),
        Op::v_lshrrev_b16_e64 => bin(Bin::LShr, b, bin(Bin::And, a, k(15))),
        Op::v_ashrrev_i16_e64 => bin(Bin::AShr, b, bin(Bin::And, a, k(15))),
        Op::v_max_u16_e64 => call(&MAX_U32, vec![a, b]),
        Op::v_min_u16_e64 => call(&MIN_U32, vec![a, b]),
        Op::v_max_i16_e64 => call(&MAX_I32, vec![a, b]),
        _ => call(&MIN_I32, vec![a, b]),
    };
    let low = bin(Bin::And, value, k(0xffff));
    let Operand::Reg(Reg::V(d)) = insn.dst else {
        return Err(form(insn, "operand"));
    };
    let value = if opsel & 8 != 0 {
        bin(
            Bin::Or,
            bin(Bin::And, Expr::Var(Var::V(d)), k(0xffff)),
            bin(Bin::Shl, low, k(16)),
        )
    } else {
        low
    };
    set_vdst(cx, insn, value)?;
    Ok(true)
}

/// The packed 16 bit shifts: each half shifts the half of the second source by the low four
/// bits of the half of the first. The low result reads the halves `op_sel` names, the high result
/// those `op_sel_hi` names (`omod` holds its bits 0 and 1, `opsel` bit 3 its bit 2).
fn packed_shift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if !matches!(insn.op, Op::v_pk_lshlrev_b16 | Op::v_pk_lshrrev_b16) {
        return Ok(false);
    }
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
    if abs != 0 || neg != 0 || clamp {
        return Err(form(insn, if clamp { "clamp" } else { "absneg" }));
    }
    let hi_sel = omod | ((opsel >> 3) & 1) << 2;
    let half = |i: usize, hi: bool| -> Result<Expr, Refusal> {
        let e = bits(insn, i)?;
        Ok(if hi {
            bin(Bin::LShr, e, k(16))
        } else {
            bin(Bin::And, e, k(0xffff))
        })
    };
    let one = |sel: u8| -> Result<Expr, Refusal> {
        let count = bin(Bin::And, half(0, sel & 1 != 0)?, k(15));
        let v = half(1, sel & 2 != 0)?;
        let r = if insn.op == Op::v_pk_lshlrev_b16 {
            bin(Bin::Shl, v, count)
        } else {
            bin(Bin::LShr, v, count)
        };
        Ok(bin(Bin::And, r, k(0xffff)))
    };
    let value = bin(
        Bin::Or,
        one(opsel & 3)?,
        bin(Bin::Shl, one(hi_sel & 3)?, k(16)),
    );
    set_vdst(cx, insn, value)?;
    Ok(true)
}

/// The 64 bit vector shifts: the value is the register pair (or a constant) in source 1, the
/// count is the low six bits of source 0, the result goes to the destination pair.
fn shift64(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    use super::salu::{ASHR64, HI64, LSHR64, PACK64, SHL64};
    let helper = match insn.op {
        Op::v_lshlrev_b64_e64 => &SHL64,
        Op::v_lshrrev_b64 => &LSHR64,
        Op::v_ashrrev_i64 => &ASHR64,
        _ => return Ok(false),
    };
    integer_plain(insn)?;
    let count = super::operand::bits(insn, 0)?;
    let (lo, hi) = match insn.src[1] {
        Operand::Int(v) => {
            let v = i64::from(v) as u64;
            (k(v as u32), k((v >> 32) as u32))
        }
        Operand::Reg(Reg::V(n)) => (Expr::Var(Var::V(n)), Expr::Var(Var::V(n + 1))),
        // A literal is one word: it cannot be a 64 bit operand. Not seen, not taken.
        _ => return Err(form(insn, "operand")),
    };
    let Operand::Reg(Reg::V(d)) = insn.dst else {
        return Err(form(insn, "operand"));
    };
    let value = call(helper, vec![call(&PACK64, vec![lo, hi]), count]);
    // Both halves are computed before the first is written: the destination may be the source.
    let (t0, t1) = (cx.tmp(), cx.tmp());
    cx.set(t0, Expr::un(Un::U64ToU32, value.clone()));
    cx.set(t1, call(&HI64, vec![value]));
    cx.set_lane(Var::V(d), Expr::Var(t0));
    cx.set_lane(Var::V(d + 1), Expr::Var(t1));
    Ok(true)
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if shift64(cx, insn)? {
        return Ok(true);
    }
    if lift16(cx, insn)? || packed_shift(cx, insn)? {
        return Ok(true);
    }
    lift_in(cx, insn, None)
}

/// Lifts an instruction of this file whose first source the caller has resolved: the lane rules
/// call this for the data parallel forms with the value read from the source lane. Whether a
/// lane writes at all (bound control, row and bank masks) is the caller's to decide. The reverse
/// shifts and subtracts are refused: the data parallel forms read their second source from the
/// other lane (Mesa README-ISA.md), and the title has none of them.
pub fn lift_with_src0(cx: &mut Lifter, insn: &Insn, src0: Expr) -> Result<bool, Refusal> {
    if matches!(
        insn.op,
        Op::v_subrev_u32
            | Op::v_lshlrev_b32
            | Op::v_lshrrev_b32
            | Op::v_ashrrev_i32
            | Op::v_subbrev_co_u32
    ) {
        return Err(form(insn, "dpp"));
    }
    lift_in(cx, insn, Some(src0))
}

/// The NEG and ABS bits of the 16 lane data parallel word (20 to 23).
fn dpp_mods(insn: &Insn) -> u32 {
    match insn.ext {
        Ext::Dpp(x) => (x >> 20) & 0xf,
        _ => 0,
    }
}

fn lift_in(cx: &mut Lifter, insn: &Insn, given: Option<Expr>) -> Result<bool, Refusal> {
    let op = insn.op;
    if !known(op) {
        return Ok(false);
    }
    if op == Op::v_nop {
        return Ok(true);
    }
    if op == Op::v_cndmask_b32 {
        return cndmask(cx, insn, given).map(|()| true);
    }
    if given.is_some() {
        if dpp_mods(insn) != 0 {
            return Err(form(insn, "absneg"));
        }
    } else {
        integer_plain(insn)?;
    }
    if op == Op::v_swap_b32 {
        if given.is_some() {
            return Err(form(insn, "dpp"));
        }
        swap(cx, insn)?;
        return Ok(true);
    }
    // Sources as 32 bit patterns, with the byte or word of the sub dword form picked.
    let src = |i: usize| -> Result<Expr, Refusal> {
        match (&given, i) {
            (Some(e), 0) => Ok(e.clone()),
            _ => Ok(sdwa_src(insn, i, bits(insn, i)?)),
        }
    };
    let (a, b, c) = (
        src(0)?,
        if insn.src[1] == Operand::None {
            k(0)
        } else {
            src(1)?
        },
        if insn.src[2] == Operand::None {
            k(0)
        } else {
            bits(insn, 2)?
        },
    );
    let carry = match op {
        Op::v_add_co_u32_e64 | Op::v_sub_co_u32_e64 | Op::v_subrev_co_u32_e64 => Carry::Out,
        Op::v_addc_co_u32 | Op::v_subb_co_u32 | Op::v_subbrev_co_u32 => Carry::InOut,
        _ => Carry::None,
    };
    if carry != Carry::None {
        return add_sub_carry(cx, insn, carry, a, b).map(|()| true);
    }
    let value = match op {
        Op::v_mov_b32 => a,
        Op::v_not_b32 => Expr::un(Un::Not, a),
        Op::v_bfrev_b32 => call(&BFREV, vec![a]),
        Op::v_ffbh_u32 => call(&FFBH_U32, vec![a]),
        Op::v_ffbh_i32 => call(&FFBH_I32, vec![a]),
        Op::v_ffbl_b32 => call(&FFBL, vec![a]),
        Op::v_add_u32 => bin(Bin::Add, a, b),
        Op::v_sub_u32 => bin(Bin::Sub, a, b),
        Op::v_subrev_u32 => bin(Bin::Sub, b, a),
        Op::v_mul_lo_u32 | Op::v_mul_lo_i32 => bin(Bin::Mul, a, b),
        Op::v_mul_hi_u32 => call(&MUL_HI_U32, vec![a, b]),
        Op::v_mul_hi_i32 => call(&MUL_HI_I32, vec![a, b]),
        Op::v_mul_u32_u24 => bin(Bin::Mul, mask24(a), mask24(b)),
        Op::v_mul_i32_i24 => bin(Bin::Mul, sext24(a), sext24(b)),
        Op::v_mul_hi_u32_u24 => call(&MUL_HI_U32, vec![mask24(a), mask24(b)]),
        Op::v_mul_hi_i32_i24 => call(&MUL_HI_I32, vec![sext24(a), sext24(b)]),
        Op::v_mad_u32_u24 => bin(Bin::Add, bin(Bin::Mul, mask24(a), mask24(b)), c),
        Op::v_mad_i32_i24 => bin(Bin::Add, bin(Bin::Mul, sext24(a), sext24(b)), c),
        Op::v_and_b32 => bin(Bin::And, a, b),
        Op::v_or_b32 => bin(Bin::Or, a, b),
        Op::v_xor_b32 => bin(Bin::Xor, a, b),
        Op::v_xnor_b32 => Expr::un(Un::Not, bin(Bin::Xor, a, b)),
        Op::v_lshlrev_b32 => bin(Bin::Shl, b, a),
        Op::v_lshrrev_b32 => bin(Bin::LShr, b, a),
        Op::v_ashrrev_i32 => bin(Bin::AShr, b, a),
        Op::v_min_i32 => call(&MIN_I32, vec![a, b]),
        Op::v_max_i32 => call(&MAX_I32, vec![a, b]),
        Op::v_min_u32 => call(&MIN_U32, vec![a, b]),
        Op::v_max_u32 => call(&MAX_U32, vec![a, b]),
        Op::v_min3_i32 => call(&MIN3_I32, vec![a, b, c]),
        Op::v_max3_i32 => call(&MAX3_I32, vec![a, b, c]),
        Op::v_min3_u32 => call(&MIN3_U32, vec![a, b, c]),
        Op::v_max3_u32 => call(&MAX3_U32, vec![a, b, c]),
        Op::v_med3_i32 => call(&MED3_I32, vec![a, b, c]),
        Op::v_med3_u32 => call(&MED3_U32, vec![a, b, c]),
        Op::v_bfe_u32 => call(&BFE_U32, vec![a, b, c]),
        Op::v_bfe_i32 => call(&BFE_I32, vec![a, b, c]),
        Op::v_bfi_b32 => call(&BFI, vec![a, b, c]),
        Op::v_bfm_b32 => call(&BFM, vec![a, b]),
        Op::v_bcnt_u32_b32 => call(&BCNT, vec![a, b]),
        Op::v_alignbit_b32 => call(&ALIGNBIT, vec![a, b, c]),
        Op::v_alignbyte_b32 => call(&ALIGNBYTE, vec![a, b, c]),
        Op::v_perm_b32 => call(&PERM, vec![a, b, c]),
        Op::v_sad_u32 => call(&SAD_U32, vec![a, b, c]),
        Op::v_sad_u8 => call(&SAD_U8, vec![a, b, c]),
        Op::v_lerp_u8 => call(&LERP_U8, vec![a, b, c]),
        Op::v_xor3_b32 => bin(Bin::Xor, bin(Bin::Xor, a, b), c),
        Op::v_or3_b32 => bin(Bin::Or, bin(Bin::Or, a, b), c),
        Op::v_add3_u32 => bin(Bin::Add, bin(Bin::Add, a, b), c),
        Op::v_and_or_b32 => bin(Bin::Or, bin(Bin::And, a, b), c),
        Op::v_lshl_or_b32 => bin(Bin::Or, bin(Bin::Shl, a, b), c),
        Op::v_lshl_add_u32 => bin(Bin::Add, bin(Bin::Shl, a, b), c),
        Op::v_add_lshl_u32 => bin(Bin::Shl, bin(Bin::Add, a, b), c),
        Op::v_xad_u32 => bin(Bin::Add, bin(Bin::Xor, a, b), c),
        _ => return Ok(false),
    };
    write(cx, insn, value)?;
    Ok(true)
}

/// Writes a result to the destination register, placed as the sub dword word says.
fn write(cx: &mut Lifter, insn: &Insn, value: Expr) -> Result<(), Refusal> {
    let Operand::Reg(Reg::V(d)) = insn.dst else {
        return Err(form(insn, "operand"));
    };
    let value = sdwa_dst(insn, Expr::Var(Var::V(d)), value).map_err(|_| form(insn, "sdwa-dst"))?;
    set_vdst(cx, insn, value)
}

/// `v_cndmask_b32`: the mask is VCC in the one word forms and the third source in the two word
/// form. The ABS and NEG modifiers of the two word form are float modifiers of the two data
/// sources.
fn cndmask(cx: &mut Lifter, insn: &Insn, given: Option<Expr>) -> Result<(), Refusal> {
    // The data parallel word has NEG and ABS for both sources, as float modifiers.
    let dpp = dpp_mods(insn);
    if given.is_none() {
        if insn.form == Form::Sdwa {
            sdwa_plain(insn)?;
        } else {
            plain(insn)?;
        }
    }
    let m = if insn.form == Form::AsVop3 {
        mask(cx, insn, insn.src[2])?
    } else {
        Var::Vcc
    };
    let pick = |i: usize| -> Result<Expr, Refusal> {
        let Some(e) = &given else {
            return Ok(sdwa_src(insn, i, fbits(insn, i)?));
        };
        let mut v = if i == 0 { e.clone() } else { bits(insn, i)? };
        // NEG of source i is bit 2i, ABS bit 2i + 1.
        if (dpp >> (2 * i + 1)) & 1 != 0 {
            v = Expr::un(Un::FAbs, v);
        }
        if (dpp >> (2 * i)) & 1 != 0 {
            v = Expr::un(Un::FNeg, v);
        }
        Ok(v)
    };
    let value = Expr::select(Expr::Var(m), pick(1)?, pick(0)?);
    write(cx, insn, value)
}

/// `v_swap_b32 d, s` exchanges two vector registers.
fn swap(cx: &mut Lifter, insn: &Insn) -> Result<(), Refusal> {
    let (Operand::Reg(Reg::V(d)), Operand::Reg(Reg::V(s))) = (insn.dst, insn.src[0]) else {
        return Err(form(insn, "operand"));
    };
    let t = cx.tmp();
    cx.set(t, Expr::Var(Var::V(s)));
    cx.set_lane(Var::V(s), Expr::Var(Var::V(d)));
    cx.set_lane(Var::V(d), Expr::Var(t));
    Ok(())
}

/// The adds and subtracts with a carry out (and in). The carry bit of an inactive lane is 0.
fn add_sub_carry(
    cx: &mut Lifter,
    insn: &Insn,
    carry: Carry,
    a: Expr,
    b: Expr,
) -> Result<(), Refusal> {
    let Operand::Reg(Reg::V(d)) = insn.dst else {
        return Err(form(insn, "operand"));
    };
    let out = if insn.dst2 == Operand::None {
        Var::Vcc
    } else {
        mask(cx, insn, insn.dst2)?
    };
    let cin = if carry == Carry::InOut {
        let m = if insn.form == Form::AsVop3
            || insn.form == Form::Plain && insn.src[2] != Operand::None
        {
            mask(cx, insn, insn.src[2])?
        } else {
            Var::Vcc
        };
        Some(Expr::un(Un::BoolToU32, Expr::Var(m)))
    } else {
        None
    };
    // Which operand is subtracted from which.
    let sub = matches!(
        insn.op,
        Op::v_sub_co_u32_e64 | Op::v_subrev_co_u32_e64 | Op::v_subb_co_u32 | Op::v_subbrev_co_u32
    );
    let reversed = matches!(insn.op, Op::v_subrev_co_u32_e64 | Op::v_subbrev_co_u32);
    let (x, y) = if reversed { (b, a) } else { (a, b) };
    let (value, flag) = if sub {
        // x - y - cin, with a borrow when x < y + cin.
        let mut diff = bin(Bin::Sub, x.clone(), y.clone());
        let mut flag = bin(Bin::ULt, x.clone(), y.clone());
        if let Some(c) = cin {
            diff = bin(Bin::Sub, diff, c.clone());
            flag = bin(
                Bin::LOr,
                flag,
                bin(Bin::LAnd, Expr::un(Un::U32ToBool, c), bin(Bin::Eq, x, y)),
            );
        }
        (diff, flag)
    } else {
        // x + y + cin, with a carry when either step wraps.
        let first = bin(Bin::Add, x.clone(), y);
        let mut flag = bin(Bin::ULt, first.clone(), x);
        let mut sum = first.clone();
        if let Some(c) = cin {
            sum = bin(Bin::Add, first.clone(), c);
            flag = bin(Bin::LOr, flag, bin(Bin::ULt, sum.clone(), first));
        }
        (sum, flag)
    };
    let value = sdwa_dst(insn, Expr::Var(Var::V(d)), value).map_err(|_| form(insn, "sdwa-dst"))?;
    let t = cx.tmp();
    let f = cx.flag();
    cx.set(t, value);
    cx.set(f, flag);
    cx.set_lane(Var::V(d), Expr::Var(t));
    cx.stmts.push(Stmt::Set {
        dst: out,
        value: bin(Bin::LAnd, Expr::Var(Var::Exec), Expr::Var(f)),
        guard: Guard::Always,
    });
    Ok(())
}
