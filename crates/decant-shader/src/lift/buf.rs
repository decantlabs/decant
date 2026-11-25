//! Vector buffer instructions (`buffer_load_*`, `buffer_store_*`, the typed `tbuffer` forms and
//! the 32 bit atomics).
//!
//! The address of an access is `index * stride + offset`: the index register when `idxen` is
//! set (the stride is a field of the buffer descriptor), plus the 12 bit immediate, the scalar
//! offset operand and the offset register when `offen` is set. With both `idxen` and `offen` the
//! index is in the address register and the offset in the next one.
//!
//! The lifter only records the access. How it is written for the view the title made (constant
//! buffer, raw, structured or typed buffer) is decided by `crate::hlsl` in `buffer.rs`, which
//! reads the instruction's [`Shape`] again.

use super::Lifter;
use super::operand::{bits, form};
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{Bin, BufferOp, Desc, Expr, Guard, Stmt, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

/// What an atomic does to the dword in memory (`old`) with the data operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atomic {
    Swap,
    /// Two data registers: the new value, then the value compared with.
    CmpSwap,
    Add,
    Sub,
    /// Subtraction clamped at 0.
    CSub,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    /// `old >= data ? 0 : old + 1`.
    Inc,
    /// `old == 0 || old > data ? data : old - 1`.
    Dec,
    /// Two data registers like `CmpSwap`, compared as floats.
    FCmpSwap,
    FMin,
    FMax,
}

/// Where the conversion of a format access comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// Raw bytes.
    None,
    /// The format of the descriptor (`buffer_*_format_*`): the view converts.
    View,
    /// The format field of the instruction (`tbuffer_*`).
    Instruction,
}

/// What a buffer instruction moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub store: bool,
    pub atomic: Option<Atomic>,
    /// Bytes per element: 1, 2 or 4.
    pub bytes: u8,
    /// Components, before packing them into data registers.
    pub count: u8,
    pub d16: bool,
    /// The byte and short loads that sign extend.
    pub signed: bool,
    pub conversion: Conversion,
}

const fn plain(store: bool, bytes: u8, count: u8, signed: bool, conversion: Conversion) -> Shape {
    Shape {
        store,
        atomic: None,
        bytes,
        count,
        signed,
        conversion,
        d16: false,
    }
}

const fn packed(store: bool, count: u8) -> Shape {
    Shape {
        d16: true,
        ..plain(store, 4, count, false, Conversion::View)
    }
}

const fn atomic(a: Atomic) -> Shape {
    Shape {
        store: true,
        atomic: Some(a),
        bytes: 4,
        count: if matches!(a, Atomic::CmpSwap | Atomic::FCmpSwap) {
            2
        } else {
            1
        },
        signed: false,
        conversion: Conversion::None,
        d16: false,
    }
}

/// The shape of a buffer instruction the translator handles.
pub fn shape(op: Op) -> Option<Shape> {
    use Conversion::{Instruction as I, None as N, View as V};
    Some(match op {
        Op::buffer_load_dword => plain(false, 4, 1, false, N),
        Op::buffer_load_dwordx2 => plain(false, 4, 2, false, N),
        Op::buffer_load_dwordx3 => plain(false, 4, 3, false, N),
        Op::buffer_load_dwordx4 => plain(false, 4, 4, false, N),
        Op::buffer_load_ubyte => plain(false, 1, 1, false, N),
        Op::buffer_load_sbyte => plain(false, 1, 1, true, N),
        Op::buffer_load_ushort => plain(false, 2, 1, false, N),
        Op::buffer_load_sshort => plain(false, 2, 1, true, N),
        Op::buffer_load_format_x => plain(false, 4, 1, false, V),
        Op::buffer_load_format_xy => plain(false, 4, 2, false, V),
        Op::buffer_load_format_xyz => plain(false, 4, 3, false, V),
        Op::buffer_load_format_xyzw => plain(false, 4, 4, false, V),
        Op::buffer_load_format_d16_x => packed(false, 1),
        Op::buffer_load_format_d16_xy => packed(false, 2),
        Op::buffer_load_format_d16_xyz => packed(false, 3),
        Op::buffer_load_format_d16_xyzw => packed(false, 4),
        Op::tbuffer_load_format_x => plain(false, 4, 1, false, I),
        Op::tbuffer_load_format_xy => plain(false, 4, 2, false, I),
        Op::tbuffer_load_format_xyz => plain(false, 4, 3, false, I),
        Op::tbuffer_load_format_xyzw => plain(false, 4, 4, false, I),
        Op::buffer_store_dword => plain(true, 4, 1, false, N),
        Op::buffer_store_dwordx2 => plain(true, 4, 2, false, N),
        Op::buffer_store_dwordx3 => plain(true, 4, 3, false, N),
        Op::buffer_store_dwordx4 => plain(true, 4, 4, false, N),
        Op::buffer_store_byte => plain(true, 1, 1, false, N),
        Op::buffer_store_short => plain(true, 2, 1, false, N),
        Op::buffer_store_format_x => plain(true, 4, 1, false, V),
        Op::buffer_store_format_xy => plain(true, 4, 2, false, V),
        Op::buffer_store_format_xyz => plain(true, 4, 3, false, V),
        Op::buffer_store_format_xyzw => plain(true, 4, 4, false, V),
        Op::buffer_store_format_d16_x => packed(true, 1),
        Op::buffer_store_format_d16_xy => packed(true, 2),
        Op::buffer_store_format_d16_xyz => packed(true, 3),
        Op::buffer_store_format_d16_xyzw => packed(true, 4),
        Op::tbuffer_store_format_x => plain(true, 4, 1, false, I),
        Op::tbuffer_store_format_xy => plain(true, 4, 2, false, I),
        Op::tbuffer_store_format_xyz => plain(true, 4, 3, false, I),
        Op::tbuffer_store_format_xyzw => plain(true, 4, 4, false, I),
        Op::buffer_atomic_swap => atomic(Atomic::Swap),
        Op::buffer_atomic_cmpswap => atomic(Atomic::CmpSwap),
        Op::buffer_atomic_add => atomic(Atomic::Add),
        Op::buffer_atomic_sub => atomic(Atomic::Sub),
        Op::buffer_atomic_csub => atomic(Atomic::CSub),
        Op::buffer_atomic_smin => atomic(Atomic::SMin),
        Op::buffer_atomic_umin => atomic(Atomic::UMin),
        Op::buffer_atomic_smax => atomic(Atomic::SMax),
        Op::buffer_atomic_umax => atomic(Atomic::UMax),
        Op::buffer_atomic_and => atomic(Atomic::And),
        Op::buffer_atomic_or => atomic(Atomic::Or),
        Op::buffer_atomic_xor => atomic(Atomic::Xor),
        Op::buffer_atomic_inc => atomic(Atomic::Inc),
        Op::buffer_atomic_dec => atomic(Atomic::Dec),
        Op::buffer_atomic_fcmpswap => atomic(Atomic::FCmpSwap),
        Op::buffer_atomic_fmin => atomic(Atomic::FMin),
        Op::buffer_atomic_fmax => atomic(Atomic::FMax),
        _ => return None,
    })
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Ext::Buf {
        offset,
        offen,
        idxen,
        glc,
        lds,
        tfe,
        format,
        ..
    } = insn.ext
    else {
        return Ok(false);
    };
    let Some(shape) = shape(insn.op) else {
        let name = insn.op.name();
        if name.contains("d16") {
            return Err(form(insn, "d16"));
        }
        if name.starts_with("buffer_atomic") {
            return Err(form(insn, "64bit"));
        }
        return Ok(false);
    };
    if lds {
        return Err(form(insn, "lds"));
    }
    if tfe {
        return Err(form(insn, "tfe"));
    }
    let (Operand::Reg(Reg::V(data)), Operand::Reg(Reg::S(desc))) = (insn.dst, insn.src[1]) else {
        return Err(form(insn, "operand"));
    };
    let addr = match insn.src[0] {
        Operand::Reg(Reg::V(n)) => Some(n),
        _ if offen || idxen => return Err(form(insn, "operand")),
        _ => None,
    };
    let mut byte = Expr::U32(u32::from(offset));
    match insn.src[2] {
        Operand::Int(0) | Operand::Reg(Reg::Null) | Operand::None => {}
        _ => byte = Expr::bin(Bin::Add, byte, bits(insn, 2)?),
    }
    if let (true, Some(a)) = (offen, addr) {
        byte = Expr::bin(
            Bin::Add,
            byte,
            Expr::Var(Var::V(if idxen { a + 1 } else { a })),
        );
    }
    let index = match (idxen, addr) {
        (true, Some(a)) => Some(Expr::Var(Var::V(a))),
        _ => None,
    };
    cx.stmts.push(Stmt::Buffer(
        BufferOp {
            op: insn.op,
            glc,
            store: shape.store,
            bytes: shape.bytes,
            count: if shape.d16 {
                shape.count.div_ceil(2)
            } else {
                shape.count
            },
            data,
            index,
            offset: byte,
            format: if shape.conversion == Conversion::Instruction {
                format
            } else {
                0
            },
            buffer: Desc::Regs(desc),
            pc: insn.pc,
        },
        Guard::Exec,
    ));
    Ok(true)
}
