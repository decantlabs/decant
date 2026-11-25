//! The lifter: one decoded instruction to IR statements.
//!
//! This file only dispatches. Each instruction class has its own file, and a rule that needs an
//! HLSL function defines it as a [`crate::ir::Helper`] in the same file, so two people adding
//! instructions of different classes never touch the same file. `operand.rs` holds what the
//! rules share.
//!
//! `wave.rs` is asked first for every instruction: it takes the instructions that touch a lane
//! mask or look at other lanes, whatever their format, and answers `Ok(false)` for the rest.
//!
//! A rule file exports `lift(cx, insn) -> Result<bool, Refusal>`: `Ok(true)` when it handled the
//! instruction, `Ok(false)` when the instruction is not one of its own, an `Err` when it knows
//! the instruction but not the form it came in.

#[cfg(test)]
mod bounded_tests;
pub mod buf;
pub mod ds;
pub mod exp;
pub mod interp;
pub mod mimg;
pub mod msg;
pub mod operand;
mod relative;
pub mod salu;
pub mod smem;
pub mod valu_cmp;
pub mod valu_float;
pub mod valu_int;
pub mod wave;

use crate::api::Stage;
use crate::decode::Insn;
use crate::exec::Masks;
use crate::ir::{Expr, Guard, Stmt, Var};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::Format;

/// The state a rule works with.
pub struct Lifter<'a> {
    pub stage: Stage,
    pub masks: &'a Masks,
    /// The statements of the block being lifted.
    pub stmts: Vec<Stmt>,
    next_tmp: u16,
    relative: relative::Offsets,
}

impl<'a> Lifter<'a> {
    pub fn new(stage: Stage, masks: &'a Masks) -> Lifter<'a> {
        Lifter {
            stage,
            masks,
            stmts: Vec::new(),
            next_tmp: 0,
            relative: relative::Offsets::new(),
        }
    }

    pub(crate) fn with_relative(mut self, graph: &crate::cfg::Cfg) -> Self {
        self.relative = relative::analyse(graph, self.stage, self.masks);
        self
    }

    /// A fresh 32 bit temporary.
    pub fn tmp(&mut self) -> Var {
        self.next_tmp += 1;
        Var::Tmp(self.next_tmp - 1)
    }

    /// A fresh boolean temporary.
    pub fn flag(&mut self) -> Var {
        self.next_tmp += 1;
        Var::Flag(self.next_tmp - 1)
    }

    /// A scalar write: it happens whatever EXEC says.
    pub fn set(&mut self, dst: Var, value: Expr) {
        self.stmts.push(Stmt::Set {
            dst,
            value,
            guard: Guard::Always,
        });
    }

    /// A vector write: it happens only in an active lane.
    pub fn set_lane(&mut self, dst: Var, value: Expr) {
        self.stmts.push(Stmt::Set {
            dst,
            value,
            guard: Guard::Exec,
        });
    }

    /// Takes the statements lifted so far.
    pub fn take(&mut self) -> Vec<Stmt> {
        std::mem::take(&mut self.stmts)
    }
}

/// Lifts one instruction into `cx.stmts`.
pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<(), Refusal> {
    // The lane rules see every instruction first, whatever its format.
    if wave::lift(cx, insn)? || relative::lift(cx, insn)? {
        return Ok(());
    }
    let handled = match insn.format() {
        Format::Sop1 | Format::Sop2 | Format::Sopk | Format::Sopc => salu::lift(cx, insn)?,
        // Messages and barriers have a file of their own, before the other scalar rules.
        Format::Sopp => msg::lift(cx, insn)? || salu::lift(cx, insn)?,
        Format::Smem => smem::lift(cx, insn)?,
        Format::Vop1 | Format::Vop2 | Format::Vop3 | Format::Vop3p | Format::Vopc => {
            // The integer file goes first: it also takes the sub dword forms of the integer to
            // float conversions, which the float file would refuse by form.
            // and the interpolation file before all of them: the two word forms of the
            // interpolation instructions arrive in this format.
            interp::lift(cx, insn)?
                || valu_int::lift(cx, insn)?
                || valu_cmp::lift(cx, insn)?
                || valu_float::lift(cx, insn)?
        }
        Format::Vintrp => interp::lift(cx, insn)?,
        Format::Ds => ds::lift(cx, insn)?,
        Format::Mubuf | Format::Mtbuf => buf::lift(cx, insn)?,
        Format::Mimg => mimg::lift(cx, insn)?,
        Format::Exp => exp::lift(cx, insn)?,
        Format::Global => smem::lift_global(cx, insn)?,
        Format::Flat | Format::Scratch => false,
    };
    if handled {
        Ok(())
    } else {
        Err(Refusal::new(RefusalKind::Op(insn.op), insn.to_string()).at(insn.pc))
    }
}
