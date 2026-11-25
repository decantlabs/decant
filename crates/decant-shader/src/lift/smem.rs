//! Scalar memory reads (`s_load_dword*`, `s_buffer_load_dword*`).
//!
//! A read becomes one [`crate::ir::Load`] per dword. What the dwords are (words of a descriptor
//! in a table, or data of a buffer) is found by `crate::resource`.

use super::Lifter;
use super::operand::{form, var};
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Load, Var};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::Op;

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let (buffer, count) = match insn.op {
        Op::s_load_dword => (false, 1),
        Op::s_load_dwordx2 => (false, 2),
        Op::s_load_dwordx4 => (false, 4),
        Op::s_load_dwordx8 => (false, 8),
        Op::s_load_dwordx16 => (false, 16),
        Op::s_buffer_load_dword => (true, 1),
        Op::s_buffer_load_dwordx2 => (true, 2),
        Op::s_buffer_load_dwordx4 => (true, 4),
        Op::s_buffer_load_dwordx8 => (true, 8),
        Op::s_buffer_load_dwordx16 => (true, 16),
        _ => return Ok(false),
    };
    let Ext::Smem {
        offset, soffset, ..
    } = insn.ext
    else {
        return Ok(false);
    };
    let base = match insn.src[0] {
        Operand::Reg(Reg::S(n)) => n,
        _ => return Err(form(insn, "operand")),
    };
    let dst = match insn.dst {
        Operand::Reg(Reg::S(n)) => n,
        // A load into VCC or EXEC sets lane masks from memory.
        Operand::Reg(Reg::VccLo | Reg::ExecLo) => {
            return Err(
                Refusal::new(RefusalKind::Exec("mask-as-data"), insn.to_string()).at(insn.pc),
            );
        }
        _ => return Err(form(insn, "operand")),
    };
    // The immediate of `s_load` is a signed 21 bit byte offset, the one of `s_buffer_load` an
    // unsigned one. The two low bits of each offset are dropped on their own (memory-ops.md 1.3).
    let imm = if buffer {
        offset
    } else {
        ((offset << 11) as i32 >> 11) as u32
    } & !3;
    // The offset register may be any scalar, VCC as a plain number included (then refused as
    // a mask read as data).
    let reg = match soffset {
        Operand::Reg(Reg::Null) | Operand::None => None,
        op => Some(var(insn, op)?),
    };
    let at = |k: u32| {
        let imm = Expr::U32(imm.wrapping_add(4 * k));
        match reg {
            // A table read keeps the plain sum, which resource tracing reads as a slot. Table
            // offsets are whole slots, so the low bits are zero there anyway.
            Some(r) if !buffer => Expr::bin(Bin::Add, Expr::Var(r), imm),
            Some(r) => Expr::bin(
                Bin::Add,
                Expr::bin(Bin::And, Expr::Var(r), Expr::U32(!3)),
                imm,
            ),
            None => imm,
        }
    };
    // Every dword is read before the first result register is written: the results may cover
    // the base or the offset register.
    let temps: Vec<Var> = (0..count).map(|_| cx.tmp()).collect();
    for (k, t) in temps.iter().enumerate() {
        cx.set(
            *t,
            Expr::Load(Load {
                buffer,
                base,
                offset: Box::new(at(k as u32)),
                pc: insn.pc,
            }),
        );
    }
    for (k, t) in temps.iter().enumerate() {
        cx.set(Var::S(dst + k as u8), Expr::Var(*t));
    }
    Ok(true)
}

/// Global dword and unsigned short loads at the address in a vector register pair,
/// plus a 12 bit offset. Raw dword reads stay visible to the resource pass.
pub fn lift_global(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let count = match insn.op {
        Op::global_load_ushort | Op::global_load_dword => 1u8,
        Op::global_load_dwordx2 => 2,
        Op::global_load_dwordx3 => 3,
        Op::global_load_dwordx4 => 4,
        _ => return Ok(false),
    };
    let Ext::Flat(w0, _) = insn.ext else {
        return Ok(false);
    };
    let (Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(base))) = (insn.dst, insn.src[0]) else {
        return Err(form(insn, "operand"));
    };
    // A scalar base register next to the vector address: not seen in the title.
    if !matches!(insn.src[1], Operand::Reg(Reg::Null)) {
        return Err(form(insn, "scalar-base"));
    }
    let offset = w0 & 0xFFF;
    if insn.op == Op::global_load_ushort {
        // Keep each raw read at statement level for address tracing. CodeWord resolves
        // the containing dword. Extract both bytes so an unaligned short can cross it.
        // Snapshot the address bits before writing a possibly overlapping destination.
        let address = cx.tmp();
        cx.set(
            address,
            Expr::bin(Bin::Add, Expr::Var(Var::V(base)), Expr::U32(offset)),
        );
        let [low, high] = [0, 1].map(|k| {
            let word = cx.tmp();
            cx.set(
                word,
                Expr::VectorLoad {
                    base,
                    offset: Box::new(Expr::U32(offset + k)),
                    pc: insn.pc,
                },
            );
            let shift = Expr::bin(
                Bin::Mul,
                Expr::bin(
                    Bin::And,
                    Expr::bin(Bin::Add, Expr::Var(address), Expr::U32(k)),
                    Expr::U32(3),
                ),
                Expr::U32(8),
            );
            Expr::bin(
                Bin::And,
                Expr::bin(Bin::LShr, Expr::Var(word), shift),
                Expr::U32(255),
            )
        });
        let high = Expr::bin(Bin::Shl, high, Expr::U32(8));
        cx.set_lane(Var::V(dst), Expr::bin(Bin::Or, low, high));
        return Ok(true);
    }
    // Every dword is read before the first result register is written: the results may cover
    // the address pair.
    let temps: Vec<Var> = (0..count).map(|_| cx.tmp()).collect();
    for (k, t) in temps.iter().enumerate() {
        let at = Box::new(Expr::U32(offset + 4 * k as u32));
        cx.set(
            *t,
            Expr::VectorLoad {
                base,
                offset: at,
                pc: insn.pc,
            },
        );
    }
    for (k, t) in temps.iter().enumerate() {
        cx.set_lane(Var::V(dst + k as u8), Expr::Var(*t));
    }
    Ok(true)
}
