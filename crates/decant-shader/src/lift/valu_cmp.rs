//! Vector compares (`v_cmp_*`, `v_cmpx_*`, `v_cmp_class_*`).
//!
//! A compare writes one bit per lane: to VCC, to a mask pair in the two word form, or to EXEC for
//! the `cmpx` forms. The bit of an inactive lane is 0, so the result is combined with EXEC. On
//! GFX10 `cmpx` writes EXEC only, also in the two word form, whose destination field is unused
//! (LLVM `VOPCInstructions.td`, the `_nosdst_e64` forms of `VOPCX_Real_gfx10`).
//!
//! Types: 16, 32 and 64 bit integers and 16 and 32 bit floats. A 16 bit float is widened to 32
//! bit, which is exact and keeps the order, so its compares are the 32 bit ones. The float
//! conditions: the ones starting with `n` and `u` are true when an operand is not a number, the
//! others except `o` are false then. ABS and NEG apply to float sources only.

use super::Lifter;
use super::operand::{form, mask, var};
use super::valu_float::{H2F, Mods, f16_src, f32_src, mods, raw, word_mods};
use super::valu_int::sdwa_src;
use crate::decode::{Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Helper, Ty, Un, Var};
use crate::refusal::Refusal;
use crate::tables::Format;

// v_cmp_class: the class bits are 0 signalling not a number, 1 quiet not a number, 2 negative
// infinity, 3 negative normal, 4 negative denormal, 5 negative zero, 6 positive zero, 7 positive
// denormal, 8 positive normal, 9 positive infinity. Mesa `aco_select_nir_alu.cpp` builds
// `fisnormal`, `fisfinite` and the not a number tests from these bits. From the bit pattern, so
// a denormal is a denormal whatever the device flushes.
pub(crate) static CLASS32: Helper = Helper {
    name: "decant_vc_class32",
    result: Ty::Bool,
    text: "bool decant_vc_class32(uint a, uint m) { uint e = (a >> 23) & 0xffu; uint f = a & 0x7fffffu; bool s = (a >> 31) != 0u; uint c; if (e == 0xffu) c = (f == 0u) ? (s ? 4u : 512u) : (((f & 0x400000u) != 0u) ? 2u : 1u); else if (e == 0u) c = (f == 0u) ? (s ? 32u : 64u) : (s ? 16u : 128u); else c = s ? 8u : 256u; return (c & m) != 0u; }",
    needs: &[],
};
pub(crate) static CLASS16: Helper = Helper {
    name: "decant_vc_class16",
    result: Ty::Bool,
    text: "bool decant_vc_class16(uint a, uint m) { uint e = (a >> 10) & 0x1fu; uint f = a & 0x3ffu; bool s = ((a >> 15) & 1u) != 0u; uint c; if (e == 0x1fu) c = (f == 0u) ? (s ? 4u : 512u) : (((f & 0x200u) != 0u) ? 2u : 1u); else if (e == 0u) c = (f == 0u) ? (s ? 32u : 64u) : (s ? 16u : 128u); else c = s ? 8u : 256u; return (c & m) != 0u; }",
    needs: &[],
};

fn not(e: Expr) -> Expr {
    Expr::un(Un::LNot, e)
}

fn bin(o: Bin, a: &Expr, b: &Expr) -> Expr {
    Expr::bin(o, a.clone(), b.clone())
}

/// A float condition on two 32 bit float patterns.
fn float_cond(cond: &str, a: &Expr, b: &Expr) -> Option<Expr> {
    let ordered = || Expr::bin(Bin::LAnd, bin(Bin::FEq, a, a), bin(Bin::FEq, b, b));
    let lg = || Expr::bin(Bin::LOr, bin(Bin::FLt, a, b), bin(Bin::FLt, b, a));
    Some(match cond {
        "f" => Expr::Bool(false),
        "lt" => bin(Bin::FLt, a, b),
        "eq" => bin(Bin::FEq, a, b),
        "le" => bin(Bin::FLe, a, b),
        "gt" => bin(Bin::FLt, b, a),
        "lg" => lg(),
        "ge" => bin(Bin::FLe, b, a),
        "o" => ordered(),
        "u" => not(ordered()),
        "nge" => not(bin(Bin::FLe, b, a)),
        "nlg" => not(lg()),
        "ngt" => not(bin(Bin::FLt, b, a)),
        "nle" => not(bin(Bin::FLe, a, b)),
        "neq" => bin(Bin::FNe, a, b),
        "nlt" => not(bin(Bin::FLt, a, b)),
        "tru" => Expr::Bool(true),
        _ => return None,
    })
}

/// An integer condition on 32 bit patterns (16 bit values are extended first).
fn int_cond(cond: &str, signed: bool, a: &Expr, b: &Expr) -> Option<Expr> {
    let (lt, le) = if signed {
        (Bin::SLt, Bin::SLe)
    } else {
        (Bin::ULt, Bin::ULe)
    };
    Some(match cond {
        "f" => Expr::Bool(false),
        "lt" => bin(lt, a, b),
        "eq" => bin(Bin::Eq, a, b),
        "le" => bin(le, a, b),
        "gt" => bin(lt, b, a),
        "ne" | "lg" => bin(Bin::Ne, a, b),
        "ge" => bin(le, b, a),
        "t" | "tru" => Expr::Bool(true),
        _ => return None,
    })
}

/// An integer condition on 64 bit values given as (low, high) halves.
fn int64_cond(cond: &str, signed: bool, a: &(Expr, Expr), b: &(Expr, Expr)) -> Option<Expr> {
    let hi_lt = if signed { Bin::SLt } else { Bin::ULt };
    let eq = || {
        Expr::bin(
            Bin::LAnd,
            bin(Bin::Eq, &a.0, &b.0),
            bin(Bin::Eq, &a.1, &b.1),
        )
    };
    // x < y: the high halves decide, unless they are equal: then the low halves, unsigned.
    let lt = |x: &(Expr, Expr), y: &(Expr, Expr)| {
        Expr::bin(
            Bin::LOr,
            bin(hi_lt, &x.1, &y.1),
            Expr::bin(
                Bin::LAnd,
                bin(Bin::Eq, &x.1, &y.1),
                bin(Bin::ULt, &x.0, &y.0),
            ),
        )
    };
    Some(match cond {
        "f" => Expr::Bool(false),
        "lt" => lt(a, b),
        "eq" => eq(),
        "le" => not(lt(b, a)),
        "gt" => lt(b, a),
        "ne" | "lg" => not(eq()),
        "ge" => not(lt(a, b)),
        "t" | "tru" => Expr::Bool(true),
        _ => return None,
    })
}

/// Source `i` of a 64 bit integer compare as its two halves.
fn pair(insn: &Insn, i: usize) -> Result<(Expr, Expr), Refusal> {
    match insn.src[i] {
        Operand::Int(v) => {
            let v = i64::from(v) as u64;
            Ok((Expr::U32(v as u32), Expr::U32((v >> 32) as u32)))
        }
        Operand::Reg(Reg::V(n)) => Ok((Expr::Var(Var::V(n)), Expr::Var(Var::V(n + 1)))),
        Operand::Reg(Reg::S(n)) => Ok((
            Expr::Var(var(insn, Operand::Reg(Reg::S(n)))?),
            Expr::Var(var(insn, Operand::Reg(Reg::S(n + 1)))?),
        )),
        // A literal or a float constant in a 64 bit integer operand: not seen, not taken.
        _ => Err(form(insn, "operand")),
    }
}

/// Refuses ABS and NEG on an integer compare.
fn int_mods(insn: &Insn, m: &Mods) -> Result<(), Refusal> {
    if m.abs != 0 || m.neg != 0 {
        return Err(form(insn, "absneg"));
    }
    Ok(())
}

/// Source `i` as a 32 bit pattern, with the byte or word of the sub dword form picked.
fn bits(insn: &Insn, m: &Mods, i: usize) -> Result<Expr, Refusal> {
    Ok(sdwa_src(insn, i, raw(insn, m, i)?))
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    lift_in(cx, insn, None)
}

/// Lifts a compare whose first source the caller has resolved (the data parallel forms, see
/// [`super::valu_float::lift_with_src0`]).
pub fn lift_with_src0(cx: &mut Lifter, insn: &Insn, src0: Expr) -> Result<bool, Refusal> {
    lift_in(cx, insn, Some(src0))
}

fn lift_in(cx: &mut Lifter, insn: &Insn, given: Option<Expr>) -> Result<bool, Refusal> {
    if insn.format() != Format::Vopc {
        return Ok(false);
    }
    let name = insn.op.name();
    let (cmpx, rest) = if let Some(r) = name.strip_prefix("v_cmpx_") {
        (true, r)
    } else if let Some(r) = name.strip_prefix("v_cmp_") {
        (false, r)
    } else {
        return Ok(false);
    };
    let Some((cond, ty)) = rest.rsplit_once('_') else {
        return Ok(false);
    };
    if !matches!(
        ty,
        "f16" | "f32" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64"
    ) {
        return Ok(false);
    }
    let m = match given {
        None => mods(insn)?,
        Some(e) => Mods {
            src0: Some(e),
            ..word_mods(insn)
        },
    };
    // On GFX10 the compares take no op_sel (Mesa `can_use_opsel`), and the output modifiers
    // mean nothing for a lane mask.
    if m.opsel != 0 {
        return Err(form(insn, "opsel"));
    }
    if m.omod != 0 || m.clamp {
        return Err(form(insn, if m.clamp { "clamp" } else { "omod" }));
    }
    let value = match (cond, ty) {
        ("class", "f32") => {
            int_mods_on(insn, &m, 1)?;
            Some(Expr::Call(
                &CLASS32,
                vec![f32_src(insn, &m, 0)?, bits(insn, &m, 1)?],
            ))
        }
        ("class", "f16") => {
            int_mods_on(insn, &m, 1)?;
            Some(Expr::Call(
                &CLASS16,
                vec![f16_src(insn, &m, 0, false)?, bits(insn, &m, 1)?],
            ))
        }
        (_, "f32") => float_cond(cond, &f32_src(insn, &m, 0)?, &f32_src(insn, &m, 1)?),
        (_, "f16") => {
            let w = |i: usize| -> Result<Expr, Refusal> {
                Ok(Expr::Call(&H2F, vec![f16_src(insn, &m, i, false)?]))
            };
            float_cond(cond, &w(0)?, &w(1)?)
        }
        (_, "i32" | "u32") => {
            int_mods(insn, &m)?;
            int_cond(cond, ty == "i32", &bits(insn, &m, 0)?, &bits(insn, &m, 1)?)
        }
        (_, "i16" | "u16") => {
            int_mods(insn, &m)?;
            let signed = ty == "i16";
            let half = |i: usize| -> Result<Expr, Refusal> {
                let e = bits(insn, &m, i)?;
                Ok(if signed {
                    Expr::bin(
                        Bin::AShr,
                        Expr::bin(Bin::Shl, e, Expr::U32(16)),
                        Expr::U32(16),
                    )
                } else {
                    Expr::bin(Bin::And, e, Expr::U32(0xffff))
                })
            };
            int_cond(cond, signed, &half(0)?, &half(1)?)
        }
        _ => {
            int_mods(insn, &m)?;
            if insn.form == crate::decode::Form::Sdwa || m.src0.is_some() {
                return Err(form(insn, "sdwa"));
            }
            int64_cond(cond, ty == "i64", &pair(insn, 0)?, &pair(insn, 1)?)
        }
    };
    let Some(cond) = value else {
        return Ok(false);
    };
    let dst = if cmpx {
        Var::Exec
    } else {
        mask(cx, insn, insn.dst)?
    };
    cx.set(dst, Expr::bin(Bin::LAnd, Expr::Var(Var::Exec), cond));
    if cx.stage == crate::api::Stage::Compute
        && let Var::Mask(first) = dst
    {
        let words = cx.masks.materialize_ballot();
        if words & 1 != 0 {
            cx.set(
                Var::S(first),
                Expr::Call(&super::wave::MASK_WORD, vec![Expr::Var(dst), Expr::U32(0)]),
            );
        }
        if words & 2 != 0 && cx.masks.lanes() == 64 {
            cx.set(
                Var::S(first + 1),
                Expr::Call(&super::wave::MASK_WORD, vec![Expr::Var(dst), Expr::U32(1)]),
            );
        }
    }
    Ok(true)
}

/// Refuses ABS and NEG on the integer source `i` of a class compare.
fn int_mods_on(insn: &Insn, m: &Mods, i: usize) -> Result<(), Refusal> {
    if ((m.abs | m.neg) >> i) & 1 != 0 {
        return Err(form(insn, "absneg"));
    }
    Ok(())
}
