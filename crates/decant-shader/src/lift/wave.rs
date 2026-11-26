//! Lane rules: every instruction that works on a lane mask, and the instructions that look at
//! other lanes of the wave. This file is asked first for every instruction.
//!
//! One shader invocation is one lane, so a mask is one boolean: this lane's bit
//! ([`Var::Exec`], [`Var::Vcc`], [`Var::Mask`]). `crate::exec` says for each instruction which
//! scalar registers hold masks. The rules here cover both wave sizes: a 64 lane program keeps a
//! mask in a register pair and works on it with the `_b64` instructions, a 32 lane program
//! keeps it in one register and uses the `_b32` instructions.
//!
//! What "one lane" changes, and why it is still right (design document, "Lane masks"):
//!
//! * SCC after a mask instruction says "the result has a lane". Here it says "the result has
//!   THIS lane". The same holds for a compare of a mask with zero and for the branches on EXEC
//!   and VCC. A lane that sees a different answer than the wave is a lane the result does not
//!   contain: it is inactive in the code the answer guards, and that code cannot change what
//!   the lane keeps (its vector registers, its own mask bits). Scalar values written there are
//!   dead or undefined for the lane when it becomes active again, because the compilers these
//!   programs come from only let a scalar value leave a masked region through a vector register.
//! * `s_wqm` makes every lane of a quad active that has a lane in the source. Here the result is
//!   "active" for every invocation. Helper invocations exist exactly for the quads that have a
//!   covered pixel, so this is exact for the EXEC a pixel program starts with. For a mask of
//!   pixels that survived a kill it keeps the invocations of a fully killed quad running until
//!   the program narrows EXEC to the surviving pixels again, which it does before every export
//!   and store. They cost time and change nothing.
//!
//! * `v_readfirstlane_b32` gives the value of the first active lane to a scalar register. Here
//!   it gives this lane's own value. Measured on the title's programs, the instruction has two
//!   uses. One moves a value that is the same in every lane (a select on a uniform condition, a
//!   converted scalar, a load at a uniform address) to a scalar register: every lane has it.
//!   The other starts a loop that serves the lanes group by group (the value of the first lane,
//!   then `v_cmpx_eq` of that value with the register in every lane, the work, then the lanes
//!   served leave the mask of lanes to go). With its own value every lane is in the first
//!   group, does the work with its own value, and leaves: the same result per lane, which is
//!   all the source program can observe. A program that reads the first lane and then uses the
//!   value for other lanes without that compare asks what a wave with one lane answers: itself.
//!
//! The number of a lane in its wave is an input the stage defines ([`Input::LaneIndex`]): the
//! thread index modulo the wave size in a compute or mesh program. `v_mbcnt_lo_u32_b32` and
//! `v_mbcnt_hi_u32_b32` count the bits of a mask below the lane, in the low and in the high 32
//! lanes. With a mask of all ones, which is how programs ask for the lane number, the two
//! counts are known from the lane number alone.
//!
//! A program that runs out of scalar registers parks one in a lane of a vector register
//! (`v_writelane_b32`) and takes it back later (`v_readlane_b32`). Both ignore EXEC, and the
//! register is used for nothing else, so the lane is a scalar variable of its own
//! ([`Var::Spill`]). A parked mask is kept there as 0 or 1: this lane's bit. In a 64 lane
//! program the two words of a mask are parked one by one: the low word carries the bit, the
//! high word is nothing. `crate::exec` tells such a parking place from a real reach into
//! another lane. Compute programs use HLSL wave intrinsics for those reads and writes.
//! A mask written into vector data uses a ballot to reconstruct its low word in a converged
//! compute block. Direct EXEC and VCC reads can also collect the high word. A scalar high
//! word without its original mask identity remains refused.
//!
//! A wave of one lane. Some instructions take a whole mask as a number: they count its lanes
//! (`s_bcnt1`), search the first or the last one (`s_ff1`, `s_flbit`), test one (`s_bitcmp`),
//! count the lanes below this one (`v_mbcnt` of a real mask), or read the lane a search found
//! (`v_readlane_b32` with a computed lane). They are the wave intrinsics of the source program
//! (any, all, count, first lane, prefix count) and the loops the compiler builds to serve equal
//! lanes together. The source program has to be right for every wave the hardware forms, down
//! to a wave with one active lane, because which pixels or threads share a wave is not its
//! business. That wave is what the translation gives every lane, since an invocation knows no
//! other lane: a mask is this lane's bit and nothing else, so its count is 0 or 1, its first
//! and last lane is this lane's number ([`Input::LaneIndex`], 0 in the stages that have none)
//! or -1, no lane of it is below this one, and the lane a search finds holds this lane's value.
//! A reduction over the wave (the sum, the minimum, the bitwise or of one value per lane) is
//! built from lane permutations. `crate::exec` finds the instructions of one, and in a wave of
//! one lane its steps change nothing: the result is this lane's value, or the identity of the
//! operation when this lane does not take part (see [`Reduction`]).
//!
//! Quads. The four pixels of a 2 by 2 block sit in four lanes next to each other, in the
//! order left to right, then top to bottom, and a pixel program reads its neighbours in the
//! block to take differences: `ds_swizzle_b32` in its quad mode and the data parallel forms
//! with a quad permutation (lane `i` of each quad reads lane `perm[i]`). HLSL has exactly this
//! for pixel programs, with the same lane order: `QuadReadAcrossX`, `QuadReadAcrossY`,
//! `QuadReadAcrossDiagonal` and `QuadReadLaneAt`. The neighbour's value is what the neighbour
//! invocation holds in the same register, helper invocations included. A permutation that is
//! not the same for the four lanes needs the place of this pixel in its block, which comes
//! from the pixel position. Outside pixel programs there are no quads: refused.
//!
//! A program that depends on how lanes are laid out in a real wave says so with instructions
//! this file can refuse: a constant used as a lane pattern, a cross-lane access outside
//! compute, or a permutation of lanes.

use super::operand::{bits, form, set_vdst, var};
use super::{Lifter, valu_cmp, valu_float, valu_int};
use crate::api::Stage;
use crate::decode::{DppCtrl, Ext, Insn, Operand, Reg};
use crate::exec::{Logic, MaskOp, Parked, Reduction, VCC, mask_number, mask_op, zero_test};
use crate::ir::{Bin, Expr, Guard, Helper, Input, Node, Stmt, Ty, Un, Var};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::{Format, Op};

static MBCNT_LO: Helper = Helper {
    name: "decant_mbcnt_lo",
    result: Ty::U32,
    text: "uint decant_mbcnt_lo(uint m, uint lane) { return countbits(m & (lane >= 32u ? 0xffffffffu : ((1u << lane) - 1u))); }",
    needs: &[],
};

static MBCNT_HI: Helper = Helper {
    name: "decant_mbcnt_hi",
    result: Ty::U32,
    text: "uint decant_mbcnt_hi(uint m, uint lane) { return countbits(m & (lane <= 32u ? 0u : ((1u << (lane - 32u)) - 1u))); }",
    needs: &[],
};

static QUAD_X: Helper = Helper {
    name: "decant_quad_x",
    result: Ty::U32,
    text: "uint decant_quad_x(uint a) { return QuadReadAcrossX(a); }",
    needs: &[],
};

static QUAD_Y: Helper = Helper {
    name: "decant_quad_y",
    result: Ty::U32,
    text: "uint decant_quad_y(uint a) { return QuadReadAcrossY(a); }",
    needs: &[],
};

static QUAD_DIAGONAL: Helper = Helper {
    name: "decant_quad_diagonal",
    result: Ty::U32,
    text: "uint decant_quad_diagonal(uint a) { return QuadReadAcrossDiagonal(a); }",
    needs: &[],
};

static QUAD_AT: Helper = Helper {
    name: "decant_quad_at",
    result: Ty::U32,
    text: "uint decant_quad_at(uint a, uint lane) { return QuadReadLaneAt(a, lane); }",
    needs: &[],
};

pub(crate) static READ_LANE: Helper = Helper {
    name: "decant_read_lane",
    result: Ty::U32,
    text: "uint decant_read_lane(uint value, uint lane) { return WaveReadLaneAt(value, lane); }",
    needs: &[],
};

pub(crate) static WRITE_LANE: Helper = Helper {
    name: "decant_write_lane",
    result: Ty::U32,
    text: "uint decant_write_lane(uint value, uint lane, uint old) { return WaveGetLaneIndex() == lane ? value : old; }",
    needs: &[],
};

/// The lane of a pixel in its 2 by 2 block, from the position (as float patterns).
static QUAD_INDEX: Helper = Helper {
    name: "decant_quad_index",
    result: Ty::U32,
    text: "uint decant_quad_index(uint x, uint y) { return (uint(asfloat(x)) & 1u) | ((uint(asfloat(y)) & 1u) << 1); }",
    needs: &[],
};

fn refuse(insn: &Insn, what: &'static str) -> Refusal {
    Refusal::new(RefusalKind::Exec(what), insn.to_string()).at(insn.pc)
}

fn not(e: Expr) -> Expr {
    Expr::un(Un::LNot, e)
}

/// Whether an operand that is read holds a mask at this instruction.
fn reads_mask(cx: &Lifter, op: Operand) -> bool {
    match op {
        Operand::Reg(Reg::ExecLo) => true,
        Operand::Reg(Reg::VccLo) => cx.masks.reads_mask(VCC),
        Operand::Reg(Reg::VccHi) => cx.masks.reads_mask(VCC + 1),
        Operand::Reg(Reg::S(n)) => cx.masks.reads_mask(n),
        _ => false,
    }
}

/// This lane's bit of a constant lane mask in a compute or mesh wave.
fn constant_lane_bit(lanes: u8, op: Operand) -> Option<Expr> {
    let (lo, hi) = match op {
        Operand::Int(v) => {
            let w = i64::from(v) as u64;
            (w as u32, (w >> 32) as u32)
        }
        Operand::Lit(v) => (v, 0),
        _ => return None,
    };
    let lane = Expr::Input(Input::LaneIndex);
    let word = if lanes == 32 || lo == hi {
        Expr::U32(lo)
    } else {
        let upper = Expr::bin(Bin::ULe, Expr::U32(32), lane.clone());
        Expr::select(upper, Expr::U32(hi), Expr::U32(lo))
    };
    let bit = Expr::bin(
        Bin::And,
        Expr::bin(Bin::LShr, word, Expr::bin(Bin::And, lane, Expr::U32(31))),
        Expr::U32(1),
    );
    Some(Expr::bin(Bin::Ne, bit, Expr::U32(0)))
}

/// This lane's bit of a mask operand that is read.
fn source(cx: &Lifter, insn: &Insn, op: Operand) -> Result<Expr, Refusal> {
    Ok(match op {
        Operand::Int(0) | Operand::Lit(0) => Expr::Bool(false),
        Operand::Int(-1) => Expr::Bool(true),
        Operand::Lit(u32::MAX) if cx.masks.lanes() == 32 => Expr::Bool(true),
        Operand::Reg(Reg::ExecLo) => Expr::Var(Var::Exec),
        Operand::Reg(Reg::VccLo) if cx.masks.reads_mask(VCC) => Expr::Var(Var::Vcc),
        Operand::Reg(Reg::S(n)) if cx.masks.reads_mask(n) => Expr::Var(Var::Mask(n)),
        // A constant that names lanes by their position in the wave.
        Operand::Int(_) | Operand::Lit(_) if matches!(cx.stage, Stage::Compute | Stage::Mesh) => {
            constant_lane_bit(cx.masks.lanes(), op).ok_or_else(|| refuse(insn, "lane-pattern"))?
        }
        Operand::Int(_) | Operand::Lit(_) => return Err(refuse(insn, "lane-pattern")),
        _ => return Err(refuse(insn, "mask-as-data")),
    })
}

/// The variable a mask result goes to.
fn target(cx: &mut Lifter, insn: &Insn) -> Result<Var, Refusal> {
    match insn.dst {
        Operand::Reg(Reg::ExecLo) => Ok(Var::Exec),
        Operand::Reg(Reg::VccLo) if cx.masks.writes_mask(VCC) => Ok(Var::Vcc),
        Operand::Reg(Reg::S(n)) if cx.masks.writes_mask(n) => Ok(Var::Mask(n)),
        // Only SCC is wanted.
        Operand::Reg(Reg::Null) => Ok(cx.flag()),
        _ => Err(refuse(insn, "mask-as-data")),
    }
}

fn apply(op: Logic, a: Expr, b: Expr) -> Expr {
    let and = |x, y| Expr::bin(Bin::LAnd, x, y);
    let or = |x, y| Expr::bin(Bin::LOr, x, y);
    let xor = |x, y| Expr::bin(Bin::LXor, x, y);
    match op {
        Logic::Mov => a,
        Logic::Not => not(a),
        Logic::And => and(a, b),
        Logic::Or => or(a, b),
        Logic::Xor => xor(a, b),
        Logic::Andn1 => and(not(a), b),
        Logic::Andn2 => and(a, not(b)),
        Logic::Orn1 => or(not(a), b),
        Logic::Orn2 => or(a, not(b)),
        Logic::Nand => not(and(a, b)),
        Logic::Nor => not(or(a, b)),
        Logic::Xnor => not(xor(a, b)),
    }
}

/// The instructions of `Sop1` and `Sop2` that move and combine masks.
fn mask_instruction(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some((kind, lanes)) = mask_op(insn.op) else {
        return Ok(false);
    };
    let ours = lanes == cx.masks.lanes();
    let exec = Expr::Var(Var::Exec);
    match kind {
        MaskOp::Logic(_) | MaskOp::Select => {
            let on_exec = std::iter::once(insn.dst)
                .chain(insn.src)
                .any(|o| o == Operand::Reg(Reg::ExecLo));
            // The same instruction on numbers belongs to the scalar rules. With EXEC in the
            // other width they refuse it: half a mask is a number made of lanes.
            if !ours || !(on_exec || cx.masks.touches_mask()) {
                return Ok(false);
            }
            let a = source(cx, insn, insn.src[0])?;
            match kind {
                MaskOp::Select => {
                    let b = source(cx, insn, insn.src[1])?;
                    let dst = target(cx, insn)?;
                    cx.set(dst, Expr::select(Expr::Var(Var::Scc), a, b));
                }
                MaskOp::Logic(Logic::Mov) => {
                    let dst = target(cx, insn)?;
                    cx.set(dst, a);
                }
                MaskOp::Logic(op) => {
                    let value = if op == Logic::Not {
                        not(a)
                    } else {
                        apply(op, a, source(cx, insn, insn.src[1])?)
                    };
                    let t = cx.flag();
                    cx.set(t, value);
                    let dst = target(cx, insn)?;
                    cx.set(dst, Expr::Var(t));
                    cx.set(Var::Scc, Expr::Var(t));
                }
                _ => unreachable!("the outer match selected these two kinds"),
            }
        }
        MaskOp::SaveExec(_) | MaskOp::WrExec(_) | MaskOp::Wqm if !ours => {
            return Err(refuse(insn, "wave-size"));
        }
        MaskOp::SaveExec(op) => {
            let a = source(cx, insn, insn.src[0])?;
            let t = cx.flag();
            cx.set(t, apply(op, a, exec.clone()));
            let dst = target(cx, insn)?;
            cx.set(dst, exec);
            cx.set(Var::Exec, Expr::Var(t));
            cx.set(Var::Scc, Expr::Var(t));
        }
        MaskOp::WrExec(op) => {
            let a = source(cx, insn, insn.src[0])?;
            let t = cx.flag();
            cx.set(t, apply(op, a, exec));
            let dst = target(cx, insn)?;
            cx.set(dst, Expr::Var(t));
            cx.set(Var::Exec, Expr::Var(t));
            cx.set(Var::Scc, Expr::Var(t));
        }
        MaskOp::Wqm => {
            // See the module text: every invocation is in a quad that has a lane.
            source(cx, insn, insn.src[0])?;
            let dst = target(cx, insn)?;
            cx.set(dst, Expr::Bool(true));
            cx.set(Var::Scc, Expr::Bool(true));
        }
    }
    Ok(true)
}

/// A compare of a mask with zero: "no lane" and "some lane" become this lane's bit.
fn mask_compare(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some((at, lanes, equal)) = zero_test(insn) else {
        return Ok(false);
    };
    if lanes != cx.masks.lanes() || !reads_mask(cx, insn.src[at]) {
        return Ok(false);
    }
    let bit = source(cx, insn, insn.src[at])?;
    cx.set(Var::Scc, if equal { not(bit) } else { bit });
    Ok(true)
}

/// A whole mask taken as a number, in a wave of one lane (see the module text). Whether the
/// instruction was one.
fn mask_number_instruction(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let lanes = cx.masks.lanes();
    if mask_number(insn.op) != Some(lanes) || !reads_mask(cx, insn.src[0]) {
        return Ok(false);
    }
    let bit = cx.flag();
    let value = source(cx, insn, insn.src[0])?;
    cx.set(bit, value);
    let bit = Expr::Var(bit);
    let lane = Expr::Input(Input::LaneIndex);
    let number = |n: u32| Expr::U32(n);
    let lanes = u32::from(lanes);
    let none = number(u32::MAX);
    let (value, scc) = match insn.op {
        // How many lanes: this one or none. SCC says whether the count is not zero.
        Op::s_bcnt1_i32_b32 | Op::s_bcnt1_i32_b64 => {
            (Expr::select(bit.clone(), number(1), number(0)), Some(bit))
        }
        // How many lanes are not in the mask: all the others.
        Op::s_bcnt0_i32_b32 | Op::s_bcnt0_i32_b64 => (
            Expr::select(bit, number(lanes - 1), number(lanes)),
            Some(Expr::Bool(true)),
        ),
        // The first lane of the mask, -1 for an empty mask.
        Op::s_ff1_i32_b32 | Op::s_ff1_i32_b64 => (Expr::select(bit, lane, none), None),
        // The first lane that is not in the mask: lane 0, or lane 1 when this is lane 0.
        Op::s_ff0_i32_b32 | Op::s_ff0_i32_b64 => (
            Expr::select(
                Expr::bin(Bin::LAnd, bit, Expr::bin(Bin::Eq, lane, number(0))),
                number(1),
                number(0),
            ),
            None,
        ),
        // The last lane of the mask, counted from the top bit, -1 for an empty mask.
        Op::s_flbit_i32_b32 | Op::s_flbit_i32_b64 => (
            Expr::select(bit, Expr::bin(Bin::Sub, number(lanes - 1), lane), none),
            None,
        ),
        // Whether the lane the second source names is in the mask (or is not).
        Op::s_bitcmp0_b32 | Op::s_bitcmp1_b32 | Op::s_bitcmp0_b64 | Op::s_bitcmp1_b64 => {
            let named = Expr::bin(Bin::And, bits(insn, 1)?, number(lanes - 1));
            let set = Expr::bin(Bin::LAnd, bit, Expr::bin(Bin::Eq, named, lane));
            let wanted = matches!(insn.op, Op::s_bitcmp1_b32 | Op::s_bitcmp1_b64);
            cx.set(Var::Scc, if wanted { set } else { not(set) });
            return Ok(true);
        }
        _ => return Ok(false),
    };
    let dst = var(insn, insn.dst)?;
    cx.set(dst, value);
    if let Some(scc) = scc {
        cx.set(Var::Scc, scc);
    }
    Ok(true)
}

/// `v_mbcnt_lo_u32_b32` and `v_mbcnt_hi_u32_b32`: the bits of the mask half below this lane,
/// plus the second source.
fn count_below(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let lane = Expr::Input(Input::LaneIndex);
    let half = Expr::U32(32);
    let low = insn.op == Op::v_mbcnt_lo_u32_b32;
    let below = match insn.src[0] {
        // No bit at all.
        Operand::Int(0) => Expr::U32(0),
        // Every bit: as many as there are lanes below this one in the half.
        Operand::Int(-1) if low => {
            Expr::select(Expr::bin(Bin::ULt, lane.clone(), half.clone()), lane, half)
        }
        Operand::Int(-1) => Expr::select(
            Expr::bin(Bin::ULt, half.clone(), lane.clone()),
            Expr::bin(Bin::Sub, lane, half),
            Expr::U32(0),
        ),
        // A real mask holds this lane's bit and no other: no lane of it is below this one.
        Operand::Reg(Reg::ExecLo | Reg::ExecHi) => Expr::U32(0),
        op if reads_mask(cx, op) => Expr::U32(0),
        // A number: its bits below the lane number.
        _ => Expr::Call(
            if low { &MBCNT_LO } else { &MBCNT_HI },
            vec![bits(insn, 0)?, lane],
        ),
    };
    let value = Expr::bin(Bin::Add, below, bits(insn, 1)?);
    set_vdst(cx, insn, value)?;
    Ok(true)
}

/// `v_writelane_b32` and `v_readlane_b32` on a parking place.
fn park(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some(kind) = cx.masks.parked() else {
        return Ok(false);
    };
    let write = insn.op == Op::v_writelane_b32_e64;
    let place = match (if write { insn.dst } else { insn.src[0] }, insn.src[1]) {
        (Operand::Reg(Reg::V(reg)), Operand::Int(lane)) => Var::Spill {
            reg,
            lane: lane as u8,
        },
        _ => return Ok(false),
    };
    match (kind, write) {
        // The low word carries this lane's bit.
        (Parked::MaskHigh, _) => {}
        (Parked::Number, true) => {
            let value = bits(insn, 0)?;
            cx.set(place, value);
        }
        (Parked::Number, false) => {
            let dst = var(insn, insn.dst)?;
            cx.set(dst, Expr::Var(place));
        }
        (Parked::Mask, true) => {
            let bit = if insn.src[0] == Operand::Lit(u32::MAX) {
                Expr::Bool(true)
            } else {
                source(cx, insn, insn.src[0])?
            };
            cx.set(place, Expr::un(Un::BoolToU32, bit));
        }
        (Parked::Mask, false) => {
            let dst = target(cx, insn)?;
            cx.set(dst, Expr::un(Un::U32ToBool, Expr::Var(place)));
        }
    }
    Ok(true)
}

static MASK_WORD: Helper = Helper {
    name: "decant_mask_word",
    result: Ty::U32,
    text: "uint decant_mask_word(bool value, uint word) { return WaveActiveBallot(value)[word]; }",
    needs: &[],
};

fn compute_lane(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let dst = var(insn, insn.dst)?;
    let lane = Expr::bin(
        Bin::And,
        bits(insn, 1)?,
        Expr::U32(u32::from(cx.masks.lanes()) - 1),
    );
    let value = if insn.op == Op::v_writelane_b32_e64
        && (reads_mask(cx, insn.src[0]) || matches!(insn.src[0], Operand::Reg(Reg::ExecHi)))
    {
        if !cx.masks.converged() {
            return Err(refuse(insn, "mask-as-data"));
        }
        let (value, high) = match insn.src[0] {
            Operand::Reg(Reg::ExecLo) => (Expr::Var(Var::Exec), false),
            Operand::Reg(Reg::ExecHi) => (Expr::Var(Var::Exec), true),
            Operand::Reg(Reg::S(n)) if cx.masks.reads_high_mask(n) => {
                // A separately restored high word need not be next to its low word.
                return Err(refuse(insn, "mask-as-data"));
            }
            Operand::Reg(Reg::VccHi) => (Expr::Var(Var::Vcc), true),
            op => (source(cx, insn, op)?, false),
        };
        Expr::Call(&MASK_WORD, vec![value, Expr::U32(u32::from(high))])
    } else {
        bits(insn, 0)?
    };
    let call = if insn.op == Op::v_readlane_b32_e64 {
        Expr::Call(&READ_LANE, vec![value, lane])
    } else {
        Expr::Call(&WRITE_LANE, vec![value, lane, Expr::Var(dst)])
    };
    // Both instructions ignore the console EXEC mask.
    cx.set(dst, call);
    Ok(true)
}

/// What the lane `perm[i]` of this pixel's quad holds of `value`, for the pixel in lane `i`.
fn quad_read(cx: &Lifter, insn: &Insn, value: Expr, perm: [u8; 4]) -> Result<Expr, Refusal> {
    if cx.stage != Stage::Pixel {
        return Err(refuse(insn, "cross-lane"));
    }
    let call = |h: &'static Helper, args: Vec<Expr>| Expr::Call(h, args);
    Ok(match perm {
        [0, 1, 2, 3] => value,
        [1, 0, 3, 2] => call(&QUAD_X, vec![value]),
        [2, 3, 0, 1] => call(&QUAD_Y, vec![value]),
        [3, 2, 1, 0] => call(&QUAD_DIAGONAL, vec![value]),
        [a, b, c, d] if a == b && b == c && c == d => {
            call(&QUAD_AT, vec![value, Expr::U32(u32::from(a))])
        }
        _ => {
            // Two bits per lane, picked by the place of this pixel in its block.
            let packed = (0..4).fold(0u32, |p, i| p | (u32::from(perm[i] & 3) << (2 * i)));
            let index = call(
                &QUAD_INDEX,
                vec![
                    Expr::Input(Input::FragCoord { chan: 0 }),
                    Expr::Input(Input::FragCoord { chan: 1 }),
                ],
            );
            let lane = Expr::bin(
                Bin::And,
                Expr::bin(
                    Bin::LShr,
                    Expr::U32(packed),
                    Expr::bin(Bin::Shl, index, Expr::U32(1)),
                ),
                Expr::U32(3),
            );
            call(&QUAD_AT, vec![value, lane])
        }
    })
}

pub(crate) static SWIZZLE: Helper = Helper {
    name: "decant_swizzle",
    result: Ty::U32,
    text: "uint decant_swizzle(uint thread, uint pattern, uint value) {
    uint lane = (((thread & (pattern & 31u)) | ((pattern >> 5u) & 31u)) ^ (pattern >> 10u));
    uint source = (pattern & 0x8000u) != 0u
        ? (thread & ~3u) | ((pattern >> ((thread & 3u) * 2u)) & 3u)
        : (thread & ~31u) | lane;
    decant_lds[decant_lds_words + thread] = value;
    GroupMemoryBarrierWithGroupSync();
    uint result = source < decant_swizzle_threads ? decant_lds[decant_lds_words + source] : 0u;
    GroupMemoryBarrierWithGroupSync();
    return result;
}",
    needs: &[],
};

/// Wave-level `ds_swizzle_b32` for divergent or non-converged control flow.
/// Both quad mode and bitmask mode only read within a 32-lane half-wave, so
/// `[WaveSize(32)]` and `[WaveSize(64)]` produce identical lane mappings.
/// Inactive lanes (either by HLSL control flow or by `exec`) are absent from
/// `WaveActiveBallot(exec)` and read as zero without reading an inactive lane.
pub(crate) static SWIZZLE_WAVE: Helper = Helper {
    name: "decant_swizzle_wave",
    result: Ty::U32,
    text: "uint decant_swizzle_wave(uint pattern, uint value, bool exec) {
    uint lane = WaveGetLaneIndex();
    uint sub = (((lane & (pattern & 31u)) | ((pattern >> 5u) & 31u)) ^ ((pattern >> 10u) & 31u));
    uint source = (pattern & 0x8000u) != 0u
        ? (lane & ~3u) | ((pattern >> ((lane & 3u) * 2u)) & 3u)
        : (lane & ~31u) | sub;
    uint4 active = WaveActiveBallot(exec);
    bool valid = source < WaveGetLaneCount() && ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(value, valid ? source : lane);
    return valid ? selected : 0u;
}",
    needs: &[],
};

/// Whether a `ds_swizzle_b32` pattern only ever reads from the caller's 4-lane quad
/// (`lane & ~3 ..= lane | 3`). Such a swizzle is well-defined for every D3D12 wave size
/// (`>= 4`) without needing `[WaveSize(32)]`.
fn is_quad_local(pattern: u32) -> bool {
    if pattern & 0x8000 != 0 {
        return true;
    }
    let and = pattern & 31;
    let or = (pattern >> 5) & 31;
    let xor = (pattern >> 10) & 31;
    (and & 0x1c) == 0x1c && (or & 0x1c) == 0 && (xor & 0x1c) == 0
}

/// A group exchange using barriers must execute once per thread with no earlier
/// exit bypassing it. When any swizzle stands outside converged workgroup
/// control flow, rewrite all swizzles in the function to wave-intrinsic reads
/// (`SWIZZLE_WAVE`) which require no group barrier and respect `exec`.
///
/// Compute shaders declare `[WaveSize(32)]` (or `[WaveSize(func.lanes)]`), so
/// both quad and 32-lane bitmask patterns are sound. Mesh shaders in SM 6.6
/// cannot declare `[WaveSize(...)]`, so divergent mesh swizzles are only
/// accepted when every swizzle pattern stays within a 4-lane quad.
pub(crate) fn lower_swizzles(stage: Stage, body: &mut [Node]) -> Result<(), Refusal> {
    fn has_non_converged(body: &[Node], mut converged: bool) -> (bool, bool) {
        let mut bypasses = false;
        for node in body {
            let (node_bypasses, found) = match node {
                Node::Block(stmts) => {
                    let non_conv = !converged
                        && stmts.iter().any(|s| {
                            matches!(s, Stmt::Effect { call: Expr::Call(h, _), .. }
                            if h.name == SWIZZLE.name)
                        });
                    (false, non_conv)
                }
                Node::If { then, els, .. } => {
                    let (then_b, then_f) = has_non_converged(then, false);
                    let (else_b, else_f) = has_non_converged(els, false);
                    (then_b || else_b, then_f || else_f)
                }
                Node::Loop(inner) => {
                    let (_, f) = has_non_converged(inner, false);
                    (true, f)
                }
                Node::Once(inner) => has_non_converged(inner, false),
                Node::Return => (true, false),
                Node::Break | Node::Continue => (false, false),
            };
            if found {
                return (true, true);
            }
            bypasses |= node_bypasses;
            converged &= !bypasses;
        }
        (bypasses, false)
    }

    fn all_quad_local(body: &[Node]) -> bool {
        body.iter().all(|node| match node {
            Node::Block(stmts) => stmts.iter().all(|stmt| match stmt {
                Stmt::Effect {
                    call: Expr::Call(h, args),
                    ..
                } if h.name == SWIZZLE.name => {
                    matches!(args.get(1), Some(Expr::U32(p)) if is_quad_local(*p))
                }
                _ => true,
            }),
            Node::If { then, els, .. } => all_quad_local(then) && all_quad_local(els),
            Node::Loop(inner) | Node::Once(inner) => all_quad_local(inner),
            Node::Return | Node::Break | Node::Continue => true,
        })
    }

    fn rewrite(body: &mut [Node]) {
        for node in body {
            match node {
                Node::Block(stmts) => {
                    for stmt in stmts {
                        if let Stmt::Effect {
                            call: Expr::Call(h, args),
                            ..
                        } = stmt
                            && h.name == SWIZZLE.name
                            && args.len() == 3
                        {
                            let mut drained = std::mem::take(args).into_iter();
                            let _thread = drained.next().expect("swizzle arg 0");
                            let pattern = drained.next().expect("swizzle arg 1");
                            let masked = drained.next().expect("swizzle arg 2");
                            let (value, exec) = match masked {
                                Expr::Select(cond, then, _) => (*then, *cond),
                                other => (other, Expr::Var(Var::Exec)),
                            };
                            *h = &SWIZZLE_WAVE;
                            *args = vec![pattern, value, exec];
                        }
                    }
                }
                Node::If { then, els, .. } => {
                    rewrite(then);
                    rewrite(els);
                }
                Node::Loop(inner) | Node::Once(inner) => rewrite(inner),
                Node::Return | Node::Break | Node::Continue => {}
            }
        }
    }

    if has_non_converged(body, true).1 {
        if stage == Stage::Mesh && !all_quad_local(body) {
            return Err(Refusal::new(
                RefusalKind::Exec("cross-lane"),
                "a non-quad mesh swizzle outside converged workgroup control flow",
            ));
        }
        rewrite(body);
    }
    Ok(())
}

/// `ds_swizzle_b32`: every lane gets the value another lane of its group holds.
fn swizzle(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Ext::Ds {
        offset0,
        offset1,
        gds,
    } = insn.ext
    else {
        return Ok(false);
    };
    if matches!(cx.stage, Stage::Compute | Stage::Mesh) {
        if gds {
            return Err(refuse(insn, "cross-lane"));
        }
        let pattern = u32::from(u16::from_le_bytes([offset0, offset1]));
        let value = Expr::select(Expr::Var(Var::Exec), bits(insn, 0)?, Expr::U32(0));
        let result = cx.tmp();
        cx.stmts.push(Stmt::Effect {
            dst: Some(result),
            call: Expr::Call(
                &SWIZZLE,
                vec![Expr::Input(Input::GroupIndex), Expr::U32(pattern), value],
            ),
            guard: Guard::Always,
        });
        set_vdst(cx, insn, Expr::Var(result))?;
        return Ok(true);
    }
    if offset1 & 0x80 == 0 {
        return Err(refuse(insn, "cross-lane"));
    }
    let perm = [0, 2, 4, 6].map(|shift| (offset0 >> shift) & 3);
    let value = quad_read(cx, insn, bits(insn, 0)?, perm)?;
    set_vdst(cx, insn, value)?;
    Ok(true)
}

/// Native lane reads permit scalar control flow when its condition is uniform.
/// Mask tests and scalar wave substitutions can make that condition differ by invocation.
/// Compute DPP, permlane and bpermute helpers emulate lane reads with `WaveActiveBallot` and
/// `WaveReadLaneAt` over active lanes, so only scalar wave substitutions are refused here.
pub(crate) fn check_dpp_context(
    graph: &crate::cfg::Cfg,
    masks: &crate::exec::Masks,
) -> Result<(), Refusal> {
    let reads_mask_operand = |masks: &crate::exec::Masks, op| match op {
        Operand::Reg(Reg::ExecLo | Reg::ExecHi) => true,
        Operand::Reg(Reg::VccLo) => masks.reads_mask(VCC),
        Operand::Reg(Reg::VccHi) => masks.reads_mask(VCC + 1),
        Operand::Reg(Reg::S(n)) => masks.reads_mask(n),
        _ => false,
    };
    for insn in graph.insns() {
        masks.enter(insn.pc);
        if masks.reduction().is_some()
            || (mask_number(insn.op).is_some() && reads_mask_operand(masks, insn.src[0]))
            || zero_test(insn).is_some_and(|(at, ..)| reads_mask_operand(masks, insn.src[at]))
            || matches!(
                mask_op(insn.op),
                Some((MaskOp::Wqm, _))
                    if masks.touches_mask()
                        || insn.dst == Operand::Reg(Reg::ExecLo)
                        || insn.src.contains(&Operand::Reg(Reg::ExecLo))
            )
            || matches!(
                insn.op,
                Op::v_readlane_b32_e64
                    | Op::v_writelane_b32_e64
                    | Op::v_mbcnt_lo_u32_b32
                    | Op::v_mbcnt_hi_u32_b32_e64
            )
        {
            return Err(refuse(insn, "dpp-scalar-wave"));
        }
    }
    Ok(())
}

pub(crate) static DPP_ADD: Helper = Helper {
    name: "decant_dpp_add",
    result: Ty::U32,
    text: "uint decant_dpp_add(uint a, uint b, uint old, bool exec, uint ctrl, bool bound, uint rows, uint banks, bool halfOp) {
    uint lane = WaveGetLaneIndex();
    uint col = lane & 15u;
    uint row = lane & ~15u;
    uint delta = ctrl & 15u;
    uint source = lane;
    bool inRange = true;
    if (ctrl < 256u) {
        source = (lane & ~3u) | ((ctrl >> (2u * (lane & 3u))) & 3u);
    } else if (ctrl < 272u) {
        inRange = col + delta < 16u;
        source = lane + delta;
    } else if (ctrl < 288u) {
        inRange = col >= delta;
        source = lane - delta;
    } else if (ctrl < 304u) {
        source = row | ((col - delta) & 15u);
    } else if (ctrl == 320u) {
        source = lane ^ 15u;
    } else if (ctrl == 321u) {
        source = lane ^ 7u;
    } else if (ctrl < 352u) {
        source = row | delta;
    } else {
        source = lane ^ delta;
    }
    // EXEC is tracked separately from HLSL control flow. Both masks matter for a read.
    uint4 active = WaveActiveBallot(exec);
    uint safeSource = inRange ? source : lane;
    bool valid = inRange && ((active[safeSource >> 5u] >> (safeSource & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(a, valid ? safeSource : lane);
    bool write = ((rows >> (lane >> 4u)) & (banks >> (col >> 2u)) & 1u) != 0u;
    uint operand = valid ? selected : 0u;
    uint sum = halfOp ? ((old & 0xffff0000u) | f32tof16(f16tof32(b) + f16tof32(operand))) : b + operand;
    return write && (valid || bound) ? sum : old;
}",
    needs: &[],
};

// AMD RDNA2 ISA, V_PERMLANE16_B32 and V_PERMLANEX16_B32.
// Each destination column selects one nibble from the two scalar words.
pub(crate) static PERMLANE: Helper = Helper {
    name: "decant_permlane",
    result: Ty::U32,
    text: "uint decant_permlane(uint a, uint lo, uint hi, uint old, bool exec, bool crossRow, bool fi, bool bound) {
    uint lane = WaveGetLaneIndex();
    uint col = lane & 15u;
    uint selects = col < 8u ? lo : hi;
    uint source = ((lane & ~15u) ^ (crossRow ? 16u : 0u)) | ((selects >> ((col & 7u) * 4u)) & 15u);
    uint4 active = WaveActiveBallot(exec || fi);
    bool valid = source < WaveGetLaneCount() && ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(a, valid ? source : lane);
    return valid ? selected : (bound ? 0u : old);
}",
    needs: &[],
};

fn permlane(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if !matches!(insn.op, Op::v_permlane16_b32 | Op::v_permlanex16_b32) {
        return Ok(false);
    }
    if cx.stage != Stage::Compute {
        return Err(refuse(insn, "permlane-stage"));
    }
    let Ext::Valu {
        abs: 0,
        neg: 0,
        opsel,
        omod: 0,
        clamp: false,
    } = insn.ext
    else {
        return Err(form(insn, "permlane-modifiers"));
    };
    // Bit 3 is the generic VOP3 destination half selector. This operation
    // writes a full dword, so it has no half to select. Bit 2 is reserved.
    if opsel & !11 != 0 {
        return Err(form(insn, "permlane-opsel"));
    }
    if !matches!(insn.src[0], Operand::Reg(Reg::V(_)))
        || insn.src[1..3]
            .iter()
            .any(|s| matches!(s, Operand::Reg(Reg::V(_))))
    {
        return Err(form(insn, "permlane-operands"));
    }
    let dst = var(insn, insn.dst)?;
    let result = cx.tmp();
    cx.set(
        result,
        Expr::Call(
            &PERMLANE,
            vec![
                bits(insn, 0)?,
                bits(insn, 1)?,
                bits(insn, 2)?,
                Expr::Var(dst),
                Expr::Var(Var::Exec),
                Expr::Bool(insn.op == Op::v_permlanex16_b32),
                Expr::Bool(opsel & 1 != 0),
                Expr::Bool(opsel & 2 != 0),
            ],
        ),
    );
    set_vdst(cx, insn, Expr::Var(result))?;
    Ok(true)
}

pub(crate) static DPP_MOV: Helper = Helper {
    name: "decant_dpp_mov",
    result: Ty::U32,
    text: "uint decant_dpp_mov(uint a, uint old, bool exec, uint ctrl, bool fi, bool bound, uint rows, uint banks) {
    uint lane = WaveGetLaneIndex();
    uint col = lane & 15u;
    uint row = lane & ~15u;
    uint delta = ctrl & 15u;
    uint source = lane;
    bool inRange = true;
    if (ctrl < 256u) {
        source = (lane & ~3u) | ((ctrl >> (2u * (lane & 3u))) & 3u);
    } else if (ctrl < 272u) {
        inRange = col + delta < 16u;
        source = lane + delta;
    } else if (ctrl < 288u) {
        inRange = col >= delta;
        source = lane - delta;
    } else if (ctrl < 304u) {
        source = row | ((col - delta) & 15u);
    } else if (ctrl == 320u) {
        source = lane ^ 15u;
    } else if (ctrl == 321u) {
        source = lane ^ 7u;
    } else if (ctrl < 352u) {
        source = row | delta;
    } else {
        source = lane ^ delta;
    }
    // EXEC is tracked separately from HLSL control flow. WaveActiveBallot checks both active
    // HLSL invocations and EXEC (unless fi is set) so WaveReadLaneAt never reads an inactive lane.
    uint4 active = WaveActiveBallot(exec || fi);
    bool inWave = inRange && source < WaveGetLaneCount();
    uint safeSource = inWave ? source : lane;
    bool valid = inWave && ((active[safeSource >> 5u] >> (safeSource & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(a, valid ? safeSource : lane);
    bool write = ((rows >> (lane >> 4u)) & (banks >> (col >> 2u)) & 1u) != 0u;
    return write && (valid || bound) ? (valid ? selected : 0u) : old;
}",
    needs: &[],
};

fn check_dpp16_ctrl(insn: &Insn, word: crate::decode::Dpp16) -> Result<(), Refusal> {
    if word.0 & (1 << 17) != 0 {
        return Err(form(insn, "dpp-reserved-bit"));
    }
    if let DppCtrl::Other(ctrl) = word.ctrl() {
        // The older wave shifts and cross-row broadcasts are outside GFX10 DPP16.
        return Err(form(
            insn,
            match ctrl {
                0x130 => "dpp-wave-shl",
                0x134 => "dpp-wave-rol",
                0x138 => "dpp-wave-shr",
                0x13c => "dpp-wave-ror",
                0x142 => "dpp-row-bcast15",
                0x143 => "dpp-row-bcast31",
                _ => "dpp-reserved-control",
            },
        ));
    }
    Ok(())
}

fn dpp_add(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if cx.stage != Stage::Compute || !matches!(insn.op, Op::v_add_u32 | Op::v_add_f16) {
        return Ok(false);
    }
    let Some(word) = insn.dpp16() else {
        return Ok(false);
    };
    if word.fetch_inactive() {
        return Err(form(insn, "dpp-fetch-inactive"));
    }
    if word.0 & (1 << 17) != 0 {
        return Err(form(insn, "dpp-reserved-bit"));
    }
    if word.0 & 0x00f0_0000 != 0 {
        return Err(form(insn, "dpp-absneg"));
    }
    check_dpp16_ctrl(insn, word)?;
    let dst = var(insn, insn.dst)?;
    let result = cx.tmp();
    cx.set(
        result,
        Expr::Call(
            &DPP_ADD,
            vec![
                bits(insn, 0)?,
                bits(insn, 1)?,
                Expr::Var(dst),
                Expr::Var(Var::Exec),
                Expr::U32((word.0 >> 8) & 0x1ff),
                Expr::Bool(word.bound_ctrl()),
                Expr::U32(u32::from(word.row_mask())),
                Expr::U32(u32::from(word.bank_mask())),
                Expr::Bool(insn.op == Op::v_add_f16),
            ],
        ),
    );
    set_vdst(cx, insn, Expr::Var(result))?;
    Ok(true)
}

pub(crate) static DPP8_MOV: Helper = Helper {
    name: "decant_dpp8_mov",
    result: Ty::U32,
    text: "uint decant_dpp8_mov(uint value, uint selects, bool exec, bool fi) {
    uint lane = WaveGetLaneIndex();
    uint source = (lane & ~7u) | ((selects >> (3u * (lane & 7u))) & 7u);
    uint4 active = WaveActiveBallot(exec || fi);
    bool valid = source < WaveGetLaneCount() && ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(value, valid ? source : lane);
    return valid ? selected : 0u;
}",
    needs: &[],
};

pub(crate) static BPERMUTE: Helper = Helper {
    name: "decant_bpermute",
    result: Ty::U32,
    text: "uint decant_bpermute(uint value, uint address, uint offset, bool exec) {
    uint lane = WaveGetLaneIndex();
    uint source = ((address + offset) >> 2u) & (WaveGetLaneCount() - 1u);
    uint4 active = WaveActiveBallot(exec);
    bool valid = ((active[source >> 5u] >> (source & 31u)) & 1u) != 0u;
    uint selected = WaveReadLaneAt(value, valid ? source : lane);
    return valid ? selected : 0u;
}",
    needs: &[],
};

fn bpermute(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if insn.op != Op::ds_bpermute_b32 {
        return Ok(false);
    }
    if cx.stage != Stage::Compute {
        return Err(refuse(insn, "bpermute-stage"));
    }
    let Ext::Ds {
        offset0,
        offset1,
        gds: false,
    } = insn.ext
    else {
        return Err(form(insn, "bpermute-gds"));
    };
    let result = cx.tmp();
    cx.set(
        result,
        Expr::Call(
            &BPERMUTE,
            vec![
                bits(insn, 1)?,
                bits(insn, 0)?,
                Expr::U32(u32::from(offset0) | (u32::from(offset1) << 8)),
                Expr::Var(Var::Exec),
            ],
        ),
    );
    set_vdst(cx, insn, Expr::Var(result))?;
    Ok(true)
}

fn dpp_mov(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if cx.stage != Stage::Compute || insn.op != Op::v_mov_b32 {
        return Ok(false);
    }
    if let Some((word, fi)) = insn.dpp8() {
        let result = cx.tmp();
        cx.set(
            result,
            Expr::Call(
                &DPP8_MOV,
                vec![
                    bits(insn, 0)?,
                    Expr::U32(word.0 >> 8),
                    Expr::Var(Var::Exec),
                    Expr::Bool(fi),
                ],
            ),
        );
        set_vdst(cx, insn, Expr::Var(result))?;
        return Ok(true);
    }
    let Some(word) = insn.dpp16() else {
        return Ok(false);
    };
    if word.0 & (1 << 17) != 0 {
        return Err(form(insn, "dpp-reserved-bit"));
    }
    if word.0 & 0x00f0_0000 != 0 {
        return Err(form(insn, "dpp-absneg"));
    }
    check_dpp16_ctrl(insn, word)?;
    let dst = var(insn, insn.dst)?;
    let result = cx.tmp();
    cx.set(
        result,
        Expr::Call(
            &DPP_MOV,
            vec![
                bits(insn, 0)?,
                Expr::Var(dst),
                Expr::Var(Var::Exec),
                Expr::U32((word.0 >> 8) & 0x1ff),
                Expr::Bool(word.fetch_inactive()),
                Expr::Bool(word.bound_ctrl()),
                Expr::U32(u32::from(word.row_mask())),
                Expr::U32(u32::from(word.bank_mask())),
            ],
        ),
    );
    set_vdst(cx, insn, Expr::Var(result))?;
    Ok(true)
}

/// A data parallel form with a quad permutation: the instruction with its first source read
/// from another lane of the quad. Whether the form was one.
fn quad_form(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some(word) = insn.dpp16() else {
        return Ok(false);
    };
    let DppCtrl::QuadPerm(perm) = word.ctrl() else {
        return Ok(false);
    };
    // Every row and bank writes. A lane whose source lane is off reads zero with bound
    // control, and without it would keep its old value, which no program of the title asks.
    if word.row_mask() != 0xf || word.bank_mask() != 0xf {
        return Ok(false);
    }
    let own = bits(insn, 0)?;
    let src0 = if cx.stage == Stage::Compute {
        if word.0 & (1 << 17) != 0 {
            return Err(form(insn, "dpp-reserved-bit"));
        }
        if !word.bound_ctrl() {
            return Ok(false);
        }
        let fetched = cx.tmp();
        cx.set(
            fetched,
            Expr::Call(
                &DPP_MOV,
                vec![
                    own,
                    Expr::U32(0),
                    Expr::Var(Var::Exec),
                    Expr::U32((word.0 >> 8) & 0x1ff),
                    Expr::Bool(word.fetch_inactive()),
                    Expr::Bool(true),
                    Expr::U32(0xf),
                    Expr::U32(0xf),
                ],
            ),
        );
        Expr::Var(fetched)
    } else {
        let seen = if word.fetch_inactive() {
            own
        } else if word.bound_ctrl() {
            Expr::select(Expr::Var(Var::Exec), own, Expr::U32(0))
        } else {
            return Ok(false);
        };
        quad_read(cx, insn, seen, perm)?
    };
    Ok(valu_int::lift_with_src0(cx, insn, src0.clone())?
        || valu_cmp::lift_with_src0(cx, insn, src0.clone())?
        || valu_float::lift_with_src0(cx, insn, src0)?)
}

/// The part of an instruction in a reduction over the wave, in a wave with one active lane
/// (see [`Reduction`]).
fn reduce(cx: &mut Lifter, insn: &Insn, part: Reduction) -> Result<(), Refusal> {
    match part {
        Reduction::Step => {}
        Reduction::OtherRow { identity } => set_vdst(cx, insn, Expr::U32(identity))?,
        Reduction::Result {
            identity,
            upper,
            combined,
        } => {
            let own = bits(insn, 0)?;
            let other = match combined {
                Some(scalar) => Expr::Var(Var::S(scalar)),
                None => Expr::U32(identity),
            };
            let value = if cx.masks.lanes() == 32 {
                own
            } else {
                // Whether this lane is in the half of the wave the lane read is in.
                let in_upper = Expr::bin(Bin::ULe, Expr::U32(32), Expr::Input(Input::LaneIndex));
                let here = if upper { in_upper } else { not(in_upper) };
                Expr::select(here, own, other)
            };
            let dst = var(insn, insn.dst)?;
            cx.set(dst, value);
        }
    }
    Ok(())
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    cx.masks.enter(insn.pc);
    if let Some(what) = cx.masks.refusal() {
        return Err(refuse(insn, what));
    }
    if cx.masks.lanes() == 64
        && insn.op == Op::s_mov_b32
        && let Operand::Reg(Reg::S(dst)) = insn.dst
        && !cx.masks.writes_mask(dst)
    {
        let mask_src = match insn.src[0] {
            Operand::Reg(Reg::ExecLo) => Some((Var::Exec, false)),
            Operand::Reg(Reg::ExecHi) => Some((Var::Exec, true)),
            Operand::Reg(Reg::VccLo) if cx.masks.reads_mask(VCC) => Some((Var::Vcc, false)),
            Operand::Reg(Reg::VccHi) if cx.masks.reads_mask(VCC + 1) => Some((Var::Vcc, true)),
            Operand::Reg(Reg::S(n)) if cx.masks.reads_mask(n) => {
                if cx.masks.reads_high_mask(n) {
                    if !n.is_multiple_of(2) {
                        Some((Var::Mask(n - 1), true))
                    } else {
                        return Err(refuse(insn, "mask-as-data"));
                    }
                } else {
                    Some((Var::Mask(n), false))
                }
            }
            _ => None,
        };
        if let Some((mask_var, high)) = mask_src {
            if !cx.masks.converged() {
                return Err(refuse(insn, "mask-as-data"));
            }
            cx.set(
                Var::S(dst),
                Expr::Call(&MASK_WORD, vec![Expr::Var(mask_var), Expr::U32(u32::from(high))]),
            );
            return Ok(true);
        }
    }
    // A scalar move replaces only the named 32 bits. Every invocation keeps
    // its old bit when it belongs to the other half, even under empty EXEC.
    if cx.masks.lanes() == 64
        && insn.op == Op::s_mov_b32
        && let Operand::Reg(half @ (Reg::ExecLo | Reg::ExecHi)) = insn.dst
        && let Some(value) = match insn.src[0] {
            Operand::Int(v) => Some(i32::from(v) as u32),
            Operand::Lit(v) => Some(v),
            _ => None,
        }
    {
        if !matches!(
            cx.stage,
            crate::api::Stage::Compute | crate::api::Stage::Mesh
        ) {
            return Err(refuse(insn, "lane-pattern"));
        }
        let lane = Expr::Input(Input::LaneIndex);
        let upper = Expr::bin(Bin::ULe, Expr::U32(32), lane.clone());
        let here = if half == Reg::ExecHi {
            upper
        } else {
            not(upper)
        };
        let bit = Expr::bin(
            Bin::And,
            Expr::bin(
                Bin::LShr,
                Expr::U32(value),
                Expr::bin(Bin::And, lane, Expr::U32(31)),
            ),
            Expr::U32(1),
        );
        cx.set(
            Var::Exec,
            Expr::select(
                here,
                Expr::bin(Bin::Ne, bit, Expr::U32(0)),
                Expr::Var(Var::Exec),
            ),
        );
        return Ok(true);
    }
    if let Some(part) = cx.masks.reduction() {
        reduce(cx, insn, part)?;
        return Ok(true);
    }
    if insn.op == Op::ds_swizzle_b32 {
        return swizzle(cx, insn);
    }
    // EXEC made from a count: `size` lanes from lane `offset` on. This lane is in it by its
    // number. (The same instruction with a scalar result makes a number: the scalar rules.)
    let field = match insn.op {
        Op::s_bfm_b32 => Some(32),
        Op::s_bfm_b64 => Some(64),
        _ => None,
    };
    if field == Some(cx.masks.lanes()) && insn.dst == Operand::Reg(Reg::ExecLo) {
        let top = Expr::U32(u32::from(cx.masks.lanes()) - 1);
        let size = Expr::bin(Bin::And, bits(insn, 0)?, top.clone());
        let offset = cx.tmp();
        let first = Expr::bin(Bin::And, bits(insn, 1)?, top);
        cx.set(offset, first);
        let lane = Expr::Input(Input::LaneIndex);
        let inside = Expr::bin(
            Bin::LAnd,
            Expr::bin(Bin::ULe, Expr::Var(offset), lane.clone()),
            Expr::bin(Bin::ULt, lane, Expr::bin(Bin::Add, Expr::Var(offset), size)),
        );
        cx.set(Var::Exec, inside);
        return Ok(true);
    }
    if bpermute(cx, insn)?
        || permlane(cx, insn)?
        || dpp_add(cx, insn)?
        || dpp_mov(cx, insn)?
        || quad_form(cx, insn)?
    {
        return Ok(true);
    }
    if mask_number_instruction(cx, insn)? {
        return Ok(true);
    }
    match insn.format() {
        Format::Sop1 | Format::Sop2 => mask_instruction(cx, insn),
        Format::Sopc => mask_compare(cx, insn),
        Format::Vop1 if insn.op == Op::v_readfirstlane_b32 => {
            // A scalar write: the machine does it whatever EXEC says.
            let value = bits(insn, 0)?;
            let dst = var(insn, insn.dst)?;
            cx.set(dst, value);
            Ok(true)
        }
        _ if matches!(insn.op, Op::v_mbcnt_lo_u32_b32 | Op::v_mbcnt_hi_u32_b32_e64) => {
            count_below(cx, insn)
        }
        _ if matches!(insn.op, Op::v_writelane_b32_e64 | Op::v_readlane_b32_e64) => {
            if park(cx, insn)? {
                return Ok(true);
            }
            if cx.stage == Stage::Compute {
                return compute_lane(cx, insn);
            }
            // A lane read with a computed lane: the lane a search of a mask found. In a wave of
            // one lane that is this lane (see the module text). A fixed lane of a register that
            // holds vector data, and every lane write, reach into another lane.
            match (insn.op, insn.src[1]) {
                (Op::v_readlane_b32_e64, Operand::Reg(Reg::S(_))) => {
                    let value = bits(insn, 0)?;
                    let dst = var(insn, insn.dst)?;
                    cx.set(dst, value);
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::*;

    #[test]
    fn divergent_compute_dpp_lowers_with_wave_emulation() {
        let mut converged = Asm::new();
        let skip = converged.label();
        let join = converged.label();
        converged.branch(Op::s_cbranch_execz, skip);
        converged.push(vop1(Op::v_mov_b32, v(3), v(1)));
        converged.branch(Op::s_branch, join);
        converged.bind(skip);
        converged.push(vop1(Op::v_mov_b32, v(3), v(2)));
        converged.bind(join);
        converged.push(dpp16(Op::v_add_u32, v(2), v(0), v(3), 0xff08_b100));
        converged.push(sopp(Op::s_endpgm, 0));
        let converged_prog = crate::api::StageProgram::new(Stage::Compute, converged.finish());
        assert!(crate::lift_program(&converged_prog).is_ok());

        let mut divergent = Asm::new();
        let done = divergent.label();
        divergent.branch(Op::s_cbranch_execz, done);
        divergent.push(dpp16(Op::v_add_u32, v(2), v(0), v(1), 0xff08_b100));
        divergent.bind(done);
        divergent.push(sopp(Op::s_endpgm, 0));
        let divergent_prog = crate::api::StageProgram::new(Stage::Compute, divergent.finish());
        assert!(crate::lift_program(&divergent_prog).is_ok());
    }

    #[test]
    fn move_b32_extracts_64_lane_mask_word_with_wave_ballot() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, s(26), EXEC));
        a.push(sop1(Op::s_mov_b32, s(40), s(26)));
        a.push(sop1(Op::s_mov_b32, s(41), s(27)));
        a.push(sopp(Op::s_endpgm, 0));
        let prog = crate::api::StageProgram::new(Stage::Compute, a.finish());
        let lifted = crate::lift_program(&prog).expect("64 lane s_mov_b32 of mask word lifts cleanly");
        assert_eq!(lifted.lanes, 64);
    }
}
