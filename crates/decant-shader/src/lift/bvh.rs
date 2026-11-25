//! Preserve native node intersections until a backend can implement their result encoding.
//! Operand facts were checked in the independent prior art lowerer.

use super::{Lifter, operand};
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{BvhIntersect, Expr, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

pub(super) fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if !matches!(
        insn.op,
        Op::image_bvh_intersect_ray | Op::image_bvh64_intersect_ray
    ) {
        return Ok(false);
    }
    let Ext::Mimg(m) = insn.ext else {
        return Err(operand::form(insn, "bvh-encoding"));
    };
    let (Operand::Reg(Reg::V(dst)), Operand::Reg(Reg::V(first)), Operand::Reg(Reg::S(desc))) =
        (insn.dst, insn.src[0], insn.src[1])
    else {
        return Err(operand::form(insn, "bvh-operands"));
    };
    if dst > 252 || desc > 102 {
        return Err(operand::form(insn, "bvh-register-range"));
    }
    let wide = insn.op == Op::image_bvh64_intersect_ray;
    let count = usize::from(wide) + 5 + if m.a16 { 3 } else { 6 };
    let address = (0..count)
        .map(|i| {
            m.address(first, i)
                .map(|reg| Expr::Var(Var::V(reg)))
                .ok_or_else(|| operand::form(insn, "bvh-address-range"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let op = BvhIntersect {
        wide,
        descriptor: (0..4).map(|i| Expr::Var(Var::S(desc + i))).collect(),
        address,
        encoding: m,
        pc: insn.pc,
    };
    // Capture all result words before any destination can replace an address operand.
    let results = (0..4)
        .map(|word| {
            let tmp = cx.tmp();
            cx.set_lane(tmp, Expr::BvhWord(Box::new(op.clone()), word));
            tmp
        })
        .collect::<Vec<_>>();
    for (word, tmp) in results.into_iter().enumerate() {
        cx.set_lane(Var::V(dst + word as u8), Expr::Var(tmp));
    }
    Ok(true)
}
