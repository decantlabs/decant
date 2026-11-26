//! Scalar instructions: `Sop1`, `Sop2`, `Sopk`, `Sopc` and the `Sopp` instructions that are not
//! branches.
//!
//! Lane masks: the instructions that move and combine masks never come here. `lift::wave` is
//! asked first and takes them (see `crate::exec`). What is left of masks in this file is the
//! refusal of a 64 bit data instruction on a register pair that holds one.

use super::Lifter;
use super::operand::{bits, form, var};
use crate::cfg::imm16;
use crate::decode::{Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Helper, Ty, Un, Var};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::{Format, Op};

static BREV: Helper = Helper {
    name: "decant_brev",
    result: Ty::U32,
    text: "uint decant_brev(uint a) { return reversebits(a); }",
    needs: &[],
};

fn not(e: Expr) -> Expr {
    Expr::un(Un::LNot, e)
}

fn nonzero(e: Expr) -> Expr {
    Expr::un(Un::U32ToBool, e)
}

// 64 bit data. A scalar register pair that holds data is two 32 bit variables. The IR has no 64
// bit arithmetic, so an operation that needs one takes the pair as a `uint64_t` (`decant_pack64`),
// computes in a helper and returns the pair as two halves again.

pub(super) static PACK64: Helper = Helper {
    name: "decant_pack64",
    result: Ty::U64,
    text: "uint64_t decant_pack64(uint lo, uint hi) { return (uint64_t(hi) << 32) | uint64_t(lo); }",
    needs: &[],
};

pub(super) static HI64: Helper = Helper {
    name: "decant_hi64",
    result: Ty::U32,
    text: "uint decant_hi64(uint64_t a) { return uint(a >> 32); }",
    needs: &[],
};

pub(super) static SHL64: Helper = Helper {
    name: "decant_shl64",
    result: Ty::U64,
    text: "uint64_t decant_shl64(uint64_t a, uint n) { return a << (n & 63u); }",
    needs: &[],
};

pub(super) static LSHR64: Helper = Helper {
    name: "decant_lshr64",
    result: Ty::U64,
    text: "uint64_t decant_lshr64(uint64_t a, uint n) { return a >> (n & 63u); }",
    needs: &[],
};

pub(super) static ASHR64: Helper = Helper {
    name: "decant_ashr64",
    result: Ty::U64,
    text: "uint64_t decant_ashr64(uint64_t a, uint n) { return uint64_t(int64_t(a) >> (n & 63u)); }",
    needs: &[],
};

static BREV64: Helper = Helper {
    name: "decant_brev64",
    result: Ty::U64,
    text: "uint64_t decant_brev64(uint64_t a) { return (uint64_t(reversebits(uint(a))) << 32) | uint64_t(reversebits(uint(a >> 32))); }",
    needs: &[],
};

static BFE_U32: Helper = Helper {
    name: "decant_bfe_u32",
    result: Ty::U32,
    text: "uint decant_bfe_u32(uint a, uint s) { uint w = (s >> 16) & 127u; uint x = a >> (s & 31u); return w == 0u ? 0u : (w >= 32u ? x : (x & ((1u << w) - 1u))); }",
    needs: &[],
};

static BFE_I32: Helper = Helper {
    name: "decant_bfe_i32",
    result: Ty::U32,
    text: "uint decant_bfe_i32(uint a, uint s) { uint w = (s >> 16) & 127u; int x = int(a) >> (s & 31u); uint up = 32u - min(w, 32u); return w == 0u ? 0u : uint((x << up) >> up); }",
    needs: &[],
};

static BFE_U64: Helper = Helper {
    name: "decant_bfe_u64",
    result: Ty::U64,
    text: "uint64_t decant_bfe_u64(uint64_t a, uint s) { uint w = (s >> 16) & 127u; uint64_t x = a >> (s & 63u); return w == 0u ? 0ull : (w >= 64u ? x : (x & ((1ull << w) - 1ull))); }",
    needs: &[],
};

static BFE_I64: Helper = Helper {
    name: "decant_bfe_i64",
    result: Ty::U64,
    text: "uint64_t decant_bfe_i64(uint64_t a, uint s) { uint w = (s >> 16) & 127u; int64_t x = int64_t(a) >> (s & 63u); uint up = 64u - min(w, 64u); return w == 0u ? 0ull : uint64_t((x << up) >> up); }",
    needs: &[],
};

static BFM_B32: Helper = Helper {
    name: "decant_bfm_b32",
    result: Ty::U32,
    text: "uint decant_bfm_b32(uint w, uint o) { return ((1u << (w & 31u)) - 1u) << (o & 31u); }",
    needs: &[],
};

static BFM_B64: Helper = Helper {
    name: "decant_bfm_b64",
    result: Ty::U64,
    text: "uint64_t decant_bfm_b64(uint w, uint o) { return ((1ull << (w & 63u)) - 1ull) << (o & 63u); }",
    needs: &[],
};

static MUL_HI_U32: Helper = Helper {
    name: "decant_mul_hi_u32",
    result: Ty::U32,
    text: "uint decant_mul_hi_u32(uint a, uint b) { return uint((uint64_t(a) * uint64_t(b)) >> 32); }",
    needs: &[],
};

static MUL_HI_I32: Helper = Helper {
    name: "decant_mul_hi_i32",
    result: Ty::U32,
    text: "uint decant_mul_hi_i32(uint a, uint b) { return uint((int64_t(int(a)) * int64_t(int(b))) >> 32); }",
    needs: &[],
};

/// 1 when the sum `(a << n) + b` does not fit 32 bits, the bits shifted out of `a` included.
static SHL_ADD_CARRY: Helper = Helper {
    name: "decant_shl_add_carry",
    result: Ty::U32,
    text: "uint decant_shl_add_carry(uint a, uint b, uint n) { return (((uint64_t(a) << n) + uint64_t(b)) >> 32) != 0ull ? 1u : 0u; }",
    needs: &[],
};

static FF1_32: Helper = Helper {
    name: "decant_ff1_b32",
    result: Ty::U32,
    text: "uint decant_ff1_b32(uint a) { return a == 0u ? 0xffffffffu : firstbitlow(a); }",
    needs: &[],
};

static FF1_64: Helper = Helper {
    name: "decant_ff1_b64",
    result: Ty::U32,
    text: "uint decant_ff1_b64(uint64_t a) { uint lo = uint(a); uint hi = uint(a >> 32); return lo != 0u ? firstbitlow(lo) : (hi != 0u ? 32u + firstbitlow(hi) : 0xffffffffu); }",
    needs: &[],
};

/// Leading zeros, -1 for zero.
static FLBIT_32: Helper = Helper {
    name: "decant_flbit_b32",
    result: Ty::U32,
    text: "uint decant_flbit_b32(uint a) { return a == 0u ? 0xffffffffu : 31u - firstbithigh(a); }",
    needs: &[],
};

static FLBIT_64: Helper = Helper {
    name: "decant_flbit_b64",
    result: Ty::U32,
    text: "uint decant_flbit_b64(uint64_t a) { uint lo = uint(a); uint hi = uint(a >> 32); return hi != 0u ? 31u - firstbithigh(hi) : (lo != 0u ? 63u - firstbithigh(lo) : 0xffffffffu); }",
    needs: &[],
};

/// Leading bits equal to the sign bit, the sign bit itself not counted, -1 for 0 and -1.
static FLBIT_S32: Helper = Helper {
    name: "decant_flbit_i32",
    result: Ty::U32,
    text: "uint decant_flbit_i32(uint a) { uint y = int(a) < 0 ? ~a : a; return y == 0u ? 0xffffffffu : 31u - firstbithigh(y); }",
    needs: &[],
};

static FLBIT_S64: Helper = Helper {
    name: "decant_flbit_i64",
    result: Ty::U32,
    text: "uint decant_flbit_i64(uint64_t a) { return decant_flbit_b64(int64_t(a) < 0 ? ~a : a); }",
    needs: &[&FLBIT_64],
};

static BCNT_64: Helper = Helper {
    name: "decant_bcnt_b64",
    result: Ty::U32,
    text: "uint decant_bcnt_b64(uint64_t a) { return countbits(uint(a)) + countbits(uint(a >> 32)); }",
    needs: &[],
};

static BIT_64: Helper = Helper {
    name: "decant_bit_b64",
    result: Ty::U32,
    text: "uint decant_bit_b64(uint64_t a, uint i) { return uint((a >> (i & 63u)) & 1ull); }",
    needs: &[],
};

static BITSET1_64: Helper = Helper {
    name: "decant_bitset1_b64",
    result: Ty::U64,
    text: "uint64_t decant_bitset1_b64(uint64_t a, uint i) { return a | (1ull << (i & 63u)); }",
    needs: &[],
};

static BITSET0_64: Helper = Helper {
    name: "decant_bitset0_b64",
    result: Ty::U64,
    text: "uint64_t decant_bitset0_b64(uint64_t a, uint i) { return a & ~(1ull << (i & 63u)); }",
    needs: &[],
};

static SEXT8: Helper = Helper {
    name: "decant_sext_i8",
    result: Ty::U32,
    text: "uint decant_sext_i8(uint a) { return uint(int(a << 24) >> 24); }",
    needs: &[],
};

static SEXT16: Helper = Helper {
    name: "decant_sext_i16",
    result: Ty::U32,
    text: "uint decant_sext_i16(uint a) { return uint(int(a << 16) >> 16); }",
    needs: &[],
};

static BCNT_32: Helper = Helper {
    name: "decant_bcnt_b32",
    result: Ty::U32,
    text: "uint decant_bcnt_b32(uint a) { return countbits(a); }",
    needs: &[],
};

fn call(h: &'static Helper, args: Vec<Expr>) -> Expr {
    Expr::Call(h, args)
}

fn bin(o: Bin, a: Expr, b: Expr) -> Expr {
    Expr::bin(o, a, b)
}

fn mask_as_data(insn: &Insn) -> Refusal {
    Refusal::new(RefusalKind::Exec("mask-as-data"), insn.to_string()).at(insn.pc)
}

/// The 32 bit destination. The null register drops the write.
fn dst32(insn: &Insn) -> Result<Option<Var>, Refusal> {
    match insn.dst {
        Operand::Reg(Reg::Null) => Ok(None),
        o => var(insn, o).map(Some),
    }
}

fn put(cx: &mut Lifter, insn: &Insn, value: Expr) -> Result<(), Refusal> {
    if let Some(d) = dst32(insn)? {
        cx.set(d, value);
    }
    Ok(())
}

/// `dst = value`, then SCC from the result.
fn with_scc(
    cx: &mut Lifter,
    insn: &Insn,
    value: Expr,
    scc: impl FnOnce(Expr) -> Expr,
) -> Result<(), Refusal> {
    let dst = dst32(insn)?;
    let t = cx.tmp();
    cx.set(t, value);
    cx.set(Var::Scc, scc(Expr::Var(t)));
    if let Some(d) = dst {
        cx.set(d, Expr::Var(t));
    }
    Ok(())
}

/// Both halves of a 64 bit source operand.
fn pair(cx: &Lifter, insn: &Insn, op: Operand) -> Result<(Expr, Expr), Refusal> {
    match op {
        Operand::Int(v) => {
            let w = i64::from(v) as u64;
            Ok((Expr::U32(w as u32), Expr::U32((w >> 32) as u32)))
        }
        Operand::Lit(v) => Ok((Expr::U32(v), Expr::U32(0))),
        Operand::Reg(Reg::Null) => Ok((Expr::U32(0), Expr::U32(0))),
        Operand::Reg(Reg::S(n)) if cx.masks.is_mask(n) => Err(mask_as_data(insn)),
        Operand::Reg(Reg::S(n)) => Ok((Expr::Var(Var::S(n)), Expr::Var(Var::S(n + 1)))),
        Operand::Reg(Reg::VccLo)
            if !cx.masks.is_mask(crate::exec::VCC) && !cx.masks.is_mask(crate::exec::VCC + 1) =>
        {
            Ok((
                Expr::Var(Var::S(crate::exec::VCC)),
                Expr::Var(Var::S(crate::exec::VCC + 1)),
            ))
        }
        Operand::Reg(Reg::ExecLo | Reg::VccLo) => Err(mask_as_data(insn)),
        _ => Err(form(insn, "operand")),
    }
}

/// A 64 bit source operand as one value.
fn src64(cx: &Lifter, insn: &Insn, op: Operand) -> Result<Expr, Refusal> {
    Ok(match op {
        Operand::Int(v) => Expr::U64(i64::from(v) as u64),
        Operand::Lit(v) => Expr::U64(u64::from(v)),
        other => {
            let (lo, hi) = pair(cx, insn, other)?;
            call(&PACK64, vec![lo, hi])
        }
    })
}

/// Writes a 64 bit result from its halves, and SCC when asked: "the result is not zero".
fn set_pair(cx: &mut Lifter, insn: &Insn, lo: Expr, hi: Expr, scc: bool) -> Result<(), Refusal> {
    let (t0, t1) = (cx.tmp(), cx.tmp());
    cx.set(t0, lo);
    cx.set(t1, hi);
    if scc {
        let both = bin(Bin::Or, Expr::Var(t0), Expr::Var(t1));
        cx.set(Var::Scc, nonzero(both));
    }
    match insn.dst {
        Operand::Reg(Reg::Null) => {}
        Operand::Reg(Reg::S(d)) if !cx.masks.is_mask(d) => {
            cx.set(Var::S(d), Expr::Var(t0));
            cx.set(Var::S(d + 1), Expr::Var(t1));
        }
        Operand::Reg(Reg::VccLo)
            if !cx.masks.is_mask(crate::exec::VCC) && !cx.masks.is_mask(crate::exec::VCC + 1) =>
        {
            cx.set(Var::S(crate::exec::VCC), Expr::Var(t0));
            cx.set(Var::S(crate::exec::VCC + 1), Expr::Var(t1));
        }
        Operand::Reg(Reg::S(_) | Reg::ExecLo | Reg::VccLo) => return Err(mask_as_data(insn)),
        _ => return Err(form(insn, "operand")),
    }
    Ok(())
}

/// Writes a 64 bit result held in one value.
fn set_u64(cx: &mut Lifter, insn: &Insn, value: Expr, scc: bool) -> Result<(), Refusal> {
    let lo = Expr::un(Un::U64ToU32, value.clone());
    set_pair(cx, insn, lo, call(&HI64, vec![value]), scc)
}

/// A bitwise 64 bit instruction on data: each half alone.
fn bitwise_data(cx: &mut Lifter, insn: &Insn) -> Result<(), Refusal> {
    let (a0, a1) = pair(cx, insn, insn.src[0])?;
    let half = |a: Expr, b: Option<Expr>| -> Expr {
        let b = || b.clone().unwrap_or(Expr::U32(0));
        let nb = || Expr::un(Un::Not, b());
        match insn.op {
            Op::s_not_b64 => Expr::un(Un::Not, a),
            Op::s_and_b64 => bin(Bin::And, a, b()),
            Op::s_or_b64 => bin(Bin::Or, a, b()),
            Op::s_xor_b64 => bin(Bin::Xor, a, b()),
            Op::s_andn2_b64 => bin(Bin::And, a, nb()),
            Op::s_orn2_b64 => bin(Bin::Or, a, nb()),
            Op::s_nand_b64 => Expr::un(Un::Not, bin(Bin::And, a, b())),
            Op::s_nor_b64 => Expr::un(Un::Not, bin(Bin::Or, a, b())),
            _ => Expr::un(Un::Not, bin(Bin::Xor, a, b())),
        }
    };
    let (b0, b1) = if insn.op == Op::s_not_b64 {
        (None, None)
    } else {
        let (b0, b1) = pair(cx, insn, insn.src[1])?;
        (Some(b0), Some(b1))
    };
    set_pair(cx, insn, half(a0, b0), half(a1, b1), true)
}

fn compare(op: Op) -> Option<(Bin, bool)> {
    // The operation and whether the operands are swapped.
    Some(match op {
        Op::s_cmp_eq_i32 | Op::s_cmp_eq_u32 | Op::s_cmpk_eq_i32 | Op::s_cmpk_eq_u32 => {
            (Bin::Eq, false)
        }
        Op::s_cmp_lg_i32 | Op::s_cmp_lg_u32 | Op::s_cmpk_lg_i32 | Op::s_cmpk_lg_u32 => {
            (Bin::Ne, false)
        }
        Op::s_cmp_gt_i32 | Op::s_cmpk_gt_i32 => (Bin::SLt, true),
        Op::s_cmp_ge_i32 | Op::s_cmpk_ge_i32 => (Bin::SLe, true),
        Op::s_cmp_lt_i32 | Op::s_cmpk_lt_i32 => (Bin::SLt, false),
        Op::s_cmp_le_i32 | Op::s_cmpk_le_i32 => (Bin::SLe, false),
        Op::s_cmp_gt_u32 | Op::s_cmpk_gt_u32 => (Bin::ULt, true),
        Op::s_cmp_ge_u32 | Op::s_cmpk_ge_u32 => (Bin::ULe, true),
        Op::s_cmp_lt_u32 | Op::s_cmpk_lt_u32 => (Bin::ULt, false),
        Op::s_cmp_le_u32 | Op::s_cmpk_le_u32 => (Bin::ULe, false),
        _ => return None,
    })
}

/// The sign extended 16 bit immediate of a `Sopk` instruction.
fn simm16(insn: &Insn) -> Expr {
    Expr::U32(i32::from(imm16(insn).unwrap_or(0) as i16) as u32)
}

/// `dst = a + b` with SCC set to the signed overflow.
fn add_i32(cx: &mut Lifter, insn: &Insn, a: Expr, b: Expr) -> Result<(), Refusal> {
    let (ta, tb) = (cx.tmp(), cx.tmp());
    cx.set(ta, a);
    cx.set(tb, b);
    let (a, b) = (Expr::Var(ta), Expr::Var(tb));
    // Signed overflow: both operands differ in sign from the sum.
    with_scc(cx, insn, bin(Bin::Add, a.clone(), b.clone()), |r| {
        sign(bin(
            Bin::And,
            bin(Bin::Xor, a, r.clone()),
            bin(Bin::Xor, b, r),
        ))
    })
}

fn sign(e: Expr) -> Expr {
    nonzero(bin(Bin::LShr, e, Expr::U32(31)))
}

/// A 32 bit result of a helper call that is also tested against zero for SCC.
fn call_scc(cx: &mut Lifter, insn: &Insn, value: Expr) -> Result<(), Refusal> {
    with_scc(cx, insn, value, nonzero)
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let op = insn.op;
    // Instructions without an effect on the translated program. `s_setreg_imm32_b32` and
    // `s_setreg_b32` set the float mode register (rounding and denormals): the float work
    // package decides what to honour. `s_sleep` and the counters only order memory accesses.
    if matches!(
        op,
        Op::s_nop
            | Op::s_waitcnt
            | Op::s_waitcnt_vscnt
            | Op::s_waitcnt_vmcnt
            | Op::s_waitcnt_lgkmcnt
            | Op::s_waitcnt_expcnt
            | Op::s_inst_prefetch
            | Op::s_clause
            | Op::s_code_end
            | Op::s_waitcnt_depctr
            | Op::s_swappc_b64
            | Op::s_setreg_imm32_b32
            | Op::s_setreg_b32
            | Op::s_sleep
            | Op::s_version
    ) {
        return Ok(true);
    }
    if let Some((cmp, swap)) = compare(op) {
        let (a, b) = if insn.format() == Format::Sopk {
            let imm = imm16(insn).unwrap_or(0);
            // The signed compares sign extend the immediate, the unsigned ones zero extend it.
            let wide = if op.name().ends_with("_i32") {
                i32::from(imm as i16) as u32
            } else {
                u32::from(imm)
            };
            (Expr::Var(var(insn, insn.dst)?), Expr::U32(wide))
        } else {
            (bits(insn, 0)?, bits(insn, 1)?)
        };
        let (a, b) = if swap { (b, a) } else { (a, b) };
        cx.set(Var::Scc, bin(cmp, a, b));
        return Ok(true);
    }
    match op {
        Op::s_mov_b32 => {
            let value = bits(insn, 0)?;
            put(cx, insn, value)?;
        }
        Op::s_movk_i32 => put(cx, insn, simm16(insn))?,
        Op::s_cmovk_i32 => {
            let d = Expr::Var(var(insn, insn.dst)?);
            put(cx, insn, Expr::select(Expr::Var(Var::Scc), simm16(insn), d))?;
        }
        Op::s_cmov_b32 => {
            let d = Expr::Var(var(insn, insn.dst)?);
            put(
                cx,
                insn,
                Expr::select(Expr::Var(Var::Scc), bits(insn, 0)?, d),
            )?;
        }
        Op::s_getpc_b64 => {
            // The byte address of the next instruction, as an offset into the program's own code.
            // Far jumps were folded away before the lifter runs, so what is left adds a constant
            // and reads a table the compiler placed behind the code.
            let next = (insn.pc + u32::from(insn.len)) * 4;
            set_pair(cx, insn, Expr::CodeAddress(next), Expr::U32(0), false)?;
        }
        Op::s_mov_b64 => {
            // A 64 bit move of data: the low half and the high half.
            let (lo, hi) = pair(cx, insn, insn.src[0])?;
            set_pair(cx, insn, lo, hi, false)?;
        }
        Op::s_cselect_b64 | Op::s_cmov_b64 => {
            let (a0, a1) = pair(cx, insn, insn.src[0])?;
            let (b0, b1) = if op == Op::s_cmov_b64 {
                pair(cx, insn, insn.dst)?
            } else {
                pair(cx, insn, insn.src[1])?
            };
            let scc = Expr::Var(Var::Scc);
            set_pair(
                cx,
                insn,
                Expr::select(scc.clone(), a0, b0),
                Expr::select(scc, a1, b1),
                false,
            )?;
        }
        Op::s_not_b64
        | Op::s_and_b64
        | Op::s_or_b64
        | Op::s_xor_b64
        | Op::s_andn2_b64
        | Op::s_orn2_b64
        | Op::s_nand_b64
        | Op::s_nor_b64
        | Op::s_xnor_b64 => bitwise_data(cx, insn)?,
        Op::s_lshl_b64 | Op::s_lshr_b64 | Op::s_ashr_i64 => {
            let h = match op {
                Op::s_lshl_b64 => &SHL64,
                Op::s_lshr_b64 => &LSHR64,
                _ => &ASHR64,
            };
            let value = call(h, vec![src64(cx, insn, insn.src[0])?, bits(insn, 1)?]);
            set_u64(cx, insn, value, true)?;
        }
        Op::s_bfe_u64 | Op::s_bfe_i64 => {
            let h = if op == Op::s_bfe_u64 {
                &BFE_U64
            } else {
                &BFE_I64
            };
            let value = call(h, vec![src64(cx, insn, insn.src[0])?, bits(insn, 1)?]);
            set_u64(cx, insn, value, true)?;
        }
        Op::s_bfm_b64 => {
            let value = call(&BFM_B64, vec![bits(insn, 0)?, bits(insn, 1)?]);
            set_u64(cx, insn, value, false)?;
        }
        Op::s_brev_b64 => {
            let value = call(&BREV64, vec![src64(cx, insn, insn.src[0])?]);
            set_u64(cx, insn, value, false)?;
        }
        Op::s_bitset0_b64 | Op::s_bitset1_b64 => {
            let h = if op == Op::s_bitset1_b64 {
                &BITSET1_64
            } else {
                &BITSET0_64
            };
            let d = src64(cx, insn, insn.dst)?;
            set_u64(cx, insn, call(h, vec![d, bits(insn, 0)?]), false)?;
        }
        Op::s_cmp_eq_u64 | Op::s_cmp_lg_u64 => {
            let (a0, a1) = pair(cx, insn, insn.src[0])?;
            let (b0, b1) = pair(cx, insn, insn.src[1])?;
            let equal = bin(Bin::LAnd, bin(Bin::Eq, a0, b0), bin(Bin::Eq, a1, b1));
            let value = if op == Op::s_cmp_eq_u64 {
                equal
            } else {
                not(equal)
            };
            cx.set(Var::Scc, value);
        }
        Op::s_bcnt1_i32_b64 | Op::s_bcnt0_i32_b64 => {
            let n = call(&BCNT_64, vec![src64(cx, insn, insn.src[0])?]);
            let n = if op == Op::s_bcnt1_i32_b64 {
                n
            } else {
                bin(Bin::Sub, Expr::U32(64), n)
            };
            call_scc(cx, insn, n)?;
        }
        Op::s_ff1_i32_b64 | Op::s_ff0_i32_b64 => {
            let mut a = src64(cx, insn, insn.src[0])?;
            if op == Op::s_ff0_i32_b64 {
                a = Expr::un(Un::Not, a);
            }
            put(cx, insn, call(&FF1_64, vec![a]))?;
        }
        Op::s_flbit_i32_b64 | Op::s_flbit_i32_i64 => {
            let h = if op == Op::s_flbit_i32_b64 {
                &FLBIT_64
            } else {
                &FLBIT_S64
            };
            put(cx, insn, call(h, vec![src64(cx, insn, insn.src[0])?]))?;
        }
        Op::s_brev_b32 => put(cx, insn, call(&BREV, vec![bits(insn, 0)?]))?,
        Op::s_not_b32 => with_scc(cx, insn, Expr::un(Un::Not, bits(insn, 0)?), nonzero)?,
        Op::s_abs_i32 => {
            let t = cx.tmp();
            cx.set(t, bits(insn, 0)?);
            let x = Expr::Var(t);
            let negative = bin(Bin::SLt, x.clone(), Expr::U32(0));
            let value = Expr::select(negative, Expr::un(Un::Neg, x.clone()), x);
            with_scc(cx, insn, value, nonzero)?;
        }
        Op::s_absdiff_i32 => {
            let t = cx.tmp();
            cx.set(t, bin(Bin::Sub, bits(insn, 0)?, bits(insn, 1)?));
            let x = Expr::Var(t);
            let negative = bin(Bin::SLt, x.clone(), Expr::U32(0));
            let value = Expr::select(negative, Expr::un(Un::Neg, x.clone()), x);
            with_scc(cx, insn, value, nonzero)?;
        }
        Op::s_sext_i32_i8 => put(cx, insn, call(&SEXT8, vec![bits(insn, 0)?]))?,
        Op::s_sext_i32_i16 => put(cx, insn, call(&SEXT16, vec![bits(insn, 0)?]))?,
        Op::s_bcnt1_i32_b32 => call_scc(cx, insn, call(&BCNT_32, vec![bits(insn, 0)?]))?,
        Op::s_bcnt0_i32_b32 => {
            let n = call(&BCNT_32, vec![bits(insn, 0)?]);
            call_scc(cx, insn, bin(Bin::Sub, Expr::U32(32), n))?;
        }
        Op::s_ff1_i32_b32 => put(cx, insn, call(&FF1_32, vec![bits(insn, 0)?]))?,
        Op::s_ff0_i32_b32 => put(
            cx,
            insn,
            call(&FF1_32, vec![Expr::un(Un::Not, bits(insn, 0)?)]),
        )?,
        Op::s_flbit_i32_b32 => put(cx, insn, call(&FLBIT_32, vec![bits(insn, 0)?]))?,
        Op::s_flbit_i32 => put(cx, insn, call(&FLBIT_S32, vec![bits(insn, 0)?]))?,
        Op::s_bitset1_b32 | Op::s_bitset0_b32 => {
            let d = Expr::Var(var(insn, insn.dst)?);
            let bit = bin(Bin::Shl, Expr::U32(1), bits(insn, 0)?);
            let value = if op == Op::s_bitset1_b32 {
                bin(Bin::Or, d, bit)
            } else {
                bin(Bin::And, d, Expr::un(Un::Not, bit))
            };
            put(cx, insn, value)?;
        }
        Op::s_bitcmp0_b32 | Op::s_bitcmp1_b32 => {
            let bit = bin(
                Bin::And,
                bin(Bin::LShr, bits(insn, 0)?, bits(insn, 1)?),
                Expr::U32(1),
            );
            let set = nonzero(bit);
            cx.set(
                Var::Scc,
                if op == Op::s_bitcmp1_b32 {
                    set
                } else {
                    not(set)
                },
            );
        }
        Op::s_bitcmp0_b64 | Op::s_bitcmp1_b64 => {
            let bit = call(&BIT_64, vec![src64(cx, insn, insn.src[0])?, bits(insn, 1)?]);
            let set = nonzero(bit);
            cx.set(
                Var::Scc,
                if op == Op::s_bitcmp1_b64 {
                    set
                } else {
                    not(set)
                },
            );
        }
        Op::s_add_u32 => {
            let (ta, tb) = (cx.tmp(), cx.tmp());
            cx.set(ta, bits(insn, 0)?);
            cx.set(tb, bits(insn, 1)?);
            let a = Expr::Var(ta);
            // The carry: the sum wrapped below its first operand.
            with_scc(cx, insn, bin(Bin::Add, a.clone(), Expr::Var(tb)), |r| {
                bin(Bin::ULt, r, a)
            })?;
        }
        Op::s_sub_u32 => {
            let (a, b) = (cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            with_scc(cx, insn, bin(Bin::Sub, Expr::Var(a), Expr::Var(b)), |_| {
                bin(Bin::ULt, Expr::Var(a), Expr::Var(b))
            })?;
        }
        Op::s_addc_u32 => {
            let (a, b, carry) = (cx.tmp(), cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            cx.set(carry, Expr::un(Un::BoolToU32, Expr::Var(Var::Scc)));
            let (sum, total) = (cx.tmp(), cx.tmp());
            cx.set(sum, bin(Bin::Add, Expr::Var(a), Expr::Var(b)));
            cx.set(total, bin(Bin::Add, Expr::Var(sum), Expr::Var(carry)));
            let wrapped = bin(Bin::ULt, Expr::Var(sum), Expr::Var(a));
            let wrapped_again = bin(Bin::ULt, Expr::Var(total), Expr::Var(sum));
            cx.set(Var::Scc, bin(Bin::LOr, wrapped, wrapped_again));
            put(cx, insn, Expr::Var(total))?;
        }
        Op::s_subb_u32 => {
            let (a, b, carry) = (cx.tmp(), cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            cx.set(carry, Expr::un(Un::BoolToU32, Expr::Var(Var::Scc)));
            let (a, b, carry) = (Expr::Var(a), Expr::Var(b), Expr::Var(carry));
            let diff = bin(Bin::Sub, bin(Bin::Sub, a.clone(), b.clone()), carry.clone());
            // The borrow: b + SCC is larger than a, as a 33 bit number.
            let borrow = bin(
                Bin::LOr,
                bin(Bin::ULt, a.clone(), b.clone()),
                bin(Bin::LAnd, bin(Bin::Eq, a, b), nonzero(carry)),
            );
            let t = cx.tmp();
            cx.set(t, diff);
            cx.set(Var::Scc, borrow);
            put(cx, insn, Expr::Var(t))?;
        }
        Op::s_add_i32 => {
            let (a, b) = (bits(insn, 0)?, bits(insn, 1)?);
            add_i32(cx, insn, a, b)?;
        }
        Op::s_addk_i32 => {
            let a = Expr::Var(var(insn, insn.dst)?);
            add_i32(cx, insn, a, simm16(insn))?;
        }
        Op::s_sub_i32 => {
            let (a, b) = (cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            let (a, b) = (Expr::Var(a), Expr::Var(b));
            // Signed overflow: the operands differ in sign and the result differs from the first.
            with_scc(cx, insn, bin(Bin::Sub, a.clone(), b.clone()), |r| {
                sign(bin(
                    Bin::And,
                    bin(Bin::Xor, a.clone(), b),
                    bin(Bin::Xor, a, r),
                ))
            })?;
        }
        Op::s_lshl1_add_u32 | Op::s_lshl2_add_u32 | Op::s_lshl3_add_u32 | Op::s_lshl4_add_u32 => {
            let n = match op {
                Op::s_lshl1_add_u32 => 1,
                Op::s_lshl2_add_u32 => 2,
                Op::s_lshl3_add_u32 => 3,
                _ => 4,
            };
            let (a, b) = (cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            let (a, b) = (Expr::Var(a), Expr::Var(b));
            let carry = call(&SHL_ADD_CARRY, vec![a.clone(), b.clone(), Expr::U32(n)]);
            let sum = bin(Bin::Add, bin(Bin::Shl, a, Expr::U32(n)), b);
            let t = cx.tmp();
            cx.set(t, sum);
            cx.set(Var::Scc, nonzero(carry));
            put(cx, insn, Expr::Var(t))?;
        }
        Op::s_mul_i32 => {
            let value = bin(Bin::Mul, bits(insn, 0)?, bits(insn, 1)?);
            put(cx, insn, value)?;
        }
        Op::s_mulk_i32 => {
            let d = Expr::Var(var(insn, insn.dst)?);
            put(cx, insn, bin(Bin::Mul, d, simm16(insn)))?;
        }
        Op::s_mul_hi_u32 => put(
            cx,
            insn,
            call(&MUL_HI_U32, vec![bits(insn, 0)?, bits(insn, 1)?]),
        )?,
        Op::s_mul_hi_i32 => put(
            cx,
            insn,
            call(&MUL_HI_I32, vec![bits(insn, 0)?, bits(insn, 1)?]),
        )?,
        Op::s_min_u32 | Op::s_min_i32 | Op::s_max_u32 | Op::s_max_i32 => {
            let (a, b) = (cx.tmp(), cx.tmp());
            cx.set(a, bits(insn, 0)?);
            cx.set(b, bits(insn, 1)?);
            let (a, b) = (Expr::Var(a), Expr::Var(b));
            let signed = matches!(op, Op::s_min_i32 | Op::s_max_i32);
            let less = if signed { Bin::SLt } else { Bin::ULt };
            let min = matches!(op, Op::s_min_u32 | Op::s_min_i32);
            // SCC is set when S0 was chosen. On a tie S1 is chosen (the GCN text, RDNA does not say).
            let pick_a = if min {
                bin(less, a.clone(), b.clone())
            } else {
                bin(less, b.clone(), a.clone())
            };
            let f = cx.flag();
            cx.set(f, pick_a);
            cx.set(Var::Scc, Expr::Var(f));
            put(cx, insn, Expr::select(Expr::Var(f), a, b))?;
        }
        Op::s_and_b32 => with_scc(
            cx,
            insn,
            bin(Bin::And, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_or_b32 => with_scc(
            cx,
            insn,
            bin(Bin::Or, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_xor_b32 => with_scc(
            cx,
            insn,
            bin(Bin::Xor, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_xnor_b32 => with_scc(
            cx,
            insn,
            Expr::un(Un::Not, bin(Bin::Xor, bits(insn, 0)?, bits(insn, 1)?)),
            nonzero,
        )?,
        Op::s_nand_b32 => with_scc(
            cx,
            insn,
            Expr::un(Un::Not, bin(Bin::And, bits(insn, 0)?, bits(insn, 1)?)),
            nonzero,
        )?,
        Op::s_nor_b32 => with_scc(
            cx,
            insn,
            Expr::un(Un::Not, bin(Bin::Or, bits(insn, 0)?, bits(insn, 1)?)),
            nonzero,
        )?,
        Op::s_andn2_b32 => with_scc(
            cx,
            insn,
            bin(Bin::And, bits(insn, 0)?, Expr::un(Un::Not, bits(insn, 1)?)),
            nonzero,
        )?,
        Op::s_orn2_b32 => with_scc(
            cx,
            insn,
            bin(Bin::Or, bits(insn, 0)?, Expr::un(Un::Not, bits(insn, 1)?)),
            nonzero,
        )?,
        Op::s_lshl_b32 => with_scc(
            cx,
            insn,
            bin(Bin::Shl, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_lshr_b32 => with_scc(
            cx,
            insn,
            bin(Bin::LShr, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_ashr_i32 => with_scc(
            cx,
            insn,
            bin(Bin::AShr, bits(insn, 0)?, bits(insn, 1)?),
            nonzero,
        )?,
        Op::s_bfe_u32 => call_scc(
            cx,
            insn,
            call(&BFE_U32, vec![bits(insn, 0)?, bits(insn, 1)?]),
        )?,
        Op::s_bfe_i32 => call_scc(
            cx,
            insn,
            call(&BFE_I32, vec![bits(insn, 0)?, bits(insn, 1)?]),
        )?,
        Op::s_bfm_b32 => put(
            cx,
            insn,
            call(&BFM_B32, vec![bits(insn, 0)?, bits(insn, 1)?]),
        )?,
        Op::s_pack_ll_b32_b16 | Op::s_pack_lh_b32_b16 | Op::s_pack_hh_b32_b16 => {
            // The low half comes from S0 (its low half, or its high half for `hh`), the high
            // half from S1 (its low half for `ll`, its high half otherwise).
            let low = if op == Op::s_pack_hh_b32_b16 {
                bin(Bin::LShr, bits(insn, 0)?, Expr::U32(16))
            } else {
                bin(Bin::And, bits(insn, 0)?, Expr::U32(0xffff))
            };
            let high = if op == Op::s_pack_ll_b32_b16 {
                bin(Bin::Shl, bits(insn, 1)?, Expr::U32(16))
            } else {
                bin(Bin::And, bits(insn, 1)?, Expr::U32(0xffff_0000))
            };
            put(cx, insn, bin(Bin::Or, low, high))?;
        }
        Op::s_cselect_b32 => {
            let value = Expr::select(Expr::Var(Var::Scc), bits(insn, 0)?, bits(insn, 1)?);
            put(cx, insn, value)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
