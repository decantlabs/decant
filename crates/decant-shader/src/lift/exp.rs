//! The export instruction. The lifter only records it: what a target means depends on the
//! stage, so `crate::stage` turns the record into output writes.

use super::Lifter;
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{Export, Stmt};
use crate::refusal::Refusal;

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Ext::Exp {
        enabled,
        target,
        compressed,
        done,
        valid_mask,
    } = insn.ext
    else {
        return Ok(false);
    };
    let mut src = [0u8; 4];
    for (slot, op) in src.iter_mut().zip(insn.src) {
        if let Operand::Reg(Reg::V(n)) = op {
            *slot = n;
        }
    }
    cx.stmts.push(Stmt::Export(Export {
        target,
        enabled,
        compressed,
        done,
        valid_mask,
        src,
        pc: insn.pc,
    }));
    Ok(true)
}
