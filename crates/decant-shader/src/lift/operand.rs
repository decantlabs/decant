//! What the lifter rules share: reading operands as expressions and writing results.

use super::Lifter;
use crate::decode::{Ext, Form, Insn, Operand, Reg};
use crate::ir::{Expr, Un, Var};
use crate::refusal::{Refusal, RefusalKind};

/// A refusal for a form of an instruction the rule does not handle.
pub fn form(insn: &Insn, what: &'static str) -> Refusal {
    Refusal::new(RefusalKind::Form(insn.op, what), insn.to_string()).at(insn.pc)
}

/// Where a write to the null register goes: a temporary nothing reads, so the dead code pass
/// removes the write and keeps the other effects of the instruction (SCC). A read of the null
/// register never comes here, [`bits`] gives zero for it.
pub const NULL_SINK: Var = Var::Tmp(u16::MAX);

/// The variable a register operand names in a 32 bit context.
pub fn var(insn: &Insn, op: Operand) -> Result<Var, Refusal> {
    match op {
        Operand::Reg(Reg::S(n)) => Ok(Var::S(n)),
        Operand::Reg(Reg::V(n)) => Ok(Var::V(n)),
        Operand::Reg(Reg::Ttmp(n)) => Ok(Var::Ttmp(n)),
        Operand::Reg(Reg::M0) => Ok(Var::M0),
        Operand::Reg(Reg::Null) => Ok(NULL_SINK),
        // VCC as a plain scratch register (its encoding is scalar register 106 and 107): the lane
        // rules, which see every instruction first, refuse a read or write of VCC as a number
        // while it holds a mask, so what arrives here holds a number.
        Operand::Reg(Reg::VccLo) => Ok(Var::S(106)),
        Operand::Reg(Reg::VccHi) => Ok(Var::S(107)),
        // The halves of EXEC as 32 bit data are whole masks, with the bits of other lanes.
        Operand::Reg(Reg::ExecLo | Reg::ExecHi) => {
            Err(Refusal::new(RefusalKind::Exec("mask-as-data"), insn.to_string()).at(insn.pc))
        }
        _ => Err(form(insn, "operand")),
    }
}

/// Source `i` as a 32 bit pattern, without modifiers. Inline integers are sign extended, inline
/// floats are their 32 bit float pattern.
pub fn bits(insn: &Insn, i: usize) -> Result<Expr, Refusal> {
    match insn.src[i] {
        Operand::Int(v) => Ok(Expr::U32(i32::from(v) as u32)),
        Operand::Float(c) => Ok(Expr::U32(c.f32().to_bits())),
        Operand::Lit(v) => Ok(Expr::U32(v)),
        Operand::Reg(Reg::Scc) => Ok(Expr::un(Un::BoolToU32, Expr::Var(Var::Scc))),
        Operand::Reg(Reg::Null) => Ok(Expr::U32(0)),
        Operand::None | Operand::Unknown(_) => Err(form(insn, "operand")),
        reg => Ok(Expr::Var(var(insn, reg)?)),
    }
}

/// Source `i` of a float instruction: the pattern with the ABS and NEG modifiers of the two
/// word encoding applied.
pub fn fbits(insn: &Insn, i: usize) -> Result<Expr, Refusal> {
    let mut e = bits(insn, i)?;
    if let Ext::Valu { abs, neg, .. } = insn.ext {
        if (abs >> i) & 1 != 0 {
            e = Expr::un(Un::FAbs, e);
        }
        if (neg >> i) & 1 != 0 {
            e = Expr::un(Un::FNeg, e);
        }
    }
    Ok(e)
}

/// Refuses the encodings with an extra word, and the modifiers no rule applies yet: operand
/// selectors, the output modifier and the clamp bit.
pub fn plain(insn: &Insn) -> Result<(), Refusal> {
    match insn.form {
        Form::Sdwa => return Err(form(insn, "sdwa")),
        Form::Dpp16 | Form::Dpp8 => return Err(form(insn, "dpp")),
        Form::Plain | Form::AsVop3 => {}
    }
    if let Ext::Valu {
        opsel, omod, clamp, ..
    } = insn.ext
    {
        if opsel != 0 {
            return Err(form(insn, "opsel"));
        }
        if omod != 0 {
            return Err(form(insn, "omod"));
        }
        if clamp {
            return Err(form(insn, "clamp"));
        }
    }
    Ok(())
}

/// Refuses ABS and NEG on an instruction whose operands are not floats.
pub fn no_float_mods(insn: &Insn) -> Result<(), Refusal> {
    match insn.ext {
        Ext::Valu { abs, neg, .. } if abs != 0 || neg != 0 => Err(form(insn, "absneg")),
        _ => Ok(()),
    }
}

/// Writes the result of a vector instruction to its destination register.
pub fn set_vdst(cx: &mut Lifter, insn: &Insn, value: Expr) -> Result<(), Refusal> {
    match insn.dst {
        Operand::Reg(Reg::V(n)) => {
            cx.set_lane(Var::V(n), value);
            Ok(())
        }
        _ => Err(form(insn, "operand")),
    }
}

/// The bit of this lane in a mask operand of a 64 bit scalar instruction or of a compare result.
pub fn mask(cx: &Lifter, insn: &Insn, op: Operand) -> Result<Var, Refusal> {
    match op {
        Operand::Reg(Reg::ExecLo) => Ok(Var::Exec),
        Operand::Reg(Reg::VccLo) => Ok(Var::Vcc),
        Operand::Reg(Reg::S(n)) if cx.masks.is_mask(n) => Ok(Var::Mask(n)),
        _ => Err(Refusal::new(RefusalKind::Exec("mask-as-data"), insn.to_string()).at(insn.pc)),
    }
}
