//! Pixel input interpolation (`v_interp_*`).
//!
//! The machine interpolates in two steps: `v_interp_p1_f32 d, i, attr` computes
//! `P0 + i * P10` and `v_interp_p2_f32 d, j, attr` adds `j * P20`, with the barycentric
//! weights `i` and `j` in vector registers and the three vertex values of the attribute behind
//! M0. In HLSL the interpolated value is simply the input, so the second step writes the input
//! and the first step writes a value nothing reads.
//!
//! Measured on the title's pixel programs: every ordinary input is either perspective
//! interpolated at the pixel centre or flat (the interpolation modes of `PSV0`), and every
//! weight pair is `v0`, `v1`. So the weights carry no information the translation needs.
//!
//! * `v_interp_mov_f32 d, p0, attr` reads the value of the provoking vertex: a flat input. The
//!   other two selectors (`p10`, `p20`) read differences of vertex values. The title never uses
//!   them and they are refused.
//! * The 16 bit forms (`v_interp_p1ll_f16`, `v_interp_p1lv_f16`, `v_interp_p2_f16`) read one
//!   half of a 32 bit attribute channel: `high` selects the upper half. The second step rounds
//!   the interpolated value to a half float and writes it to the low half of `d`, or to the high
//!   half when bit 3 of `opsel` is set. GFX10 keeps the other half ("On gfx10, all 16-bit
//!   instructions preserve the high bits", LLVM `GCNSubtarget::zeroesHigh16BitsOfDest`).
//!
//! The input read is [`Input::Attr`] with the hardware attribute number. `chan` 0 to 3 is a 32
//! bit channel. `chan` 4 to 11 is half `chan - 4` of a 16 bit attribute, as a 32 bit float:
//! half `h` lies in channel `h / 2`, low half for even `h`. `crate::stage::pixel` resolves both
//! kinds to members of the input struct.

use super::Lifter;
use super::operand::{form, set_vdst};
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Helper, Input, Ty, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

/// The first `chan` value of an [`Input::Attr`] that names a half instead of a channel.
pub const HALF: u8 = 4;

/// A 32 bit float pattern rounded to the nearest half float, in the low 16 bits.
static TO_F16: Helper = Helper {
    name: "decant_interp_f16",
    result: Ty::U32,
    text: "uint decant_interp_f16(uint a) { return f32tof16(asfloat(a)); }",
    needs: &[],
};

/// The attribute, channel and half of an interpolation instruction, after checking that it
/// carries no modifier.
fn fields(insn: &Insn) -> Result<(u8, u8, bool, u8), Refusal> {
    match insn.ext {
        Ext::Vintrp { attr, chan } => Ok((attr, chan, false, 0)),
        Ext::VintrpE64 {
            attr,
            chan,
            high,
            abs,
            neg,
            opsel,
            omod,
            clamp,
        } => {
            if abs != 0 || neg != 0 || omod != 0 || clamp {
                return Err(form(insn, "mods"));
            }
            Ok((attr, chan, high, opsel))
        }
        _ => Err(form(insn, "operand")),
    }
}

fn dst_reg(insn: &Insn) -> Result<u8, Refusal> {
    match insn.dst {
        Operand::Reg(Reg::V(n)) => Ok(n),
        _ => Err(form(insn, "operand")),
    }
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if !matches!(insn.ext, Ext::Vintrp { .. } | Ext::VintrpE64 { .. }) {
        return Ok(false);
    }
    let (attr, chan, high, opsel) = fields(insn)?;
    match insn.op {
        // The first steps compute a partial value only the second step reads.
        Op::v_interp_p1_f32 | Op::v_interp_p1ll_f16 | Op::v_interp_p1lv_f16 => {
            set_vdst(cx, insn, Expr::U32(0))?
        }
        Op::v_interp_p2_f32 => set_vdst(cx, insn, Expr::Input(Input::Attr { attr, chan }))?,
        Op::v_interp_mov_f32 => match insn.src[0] {
            // Selector 2 is P0, the value of the provoking vertex.
            Operand::Int(2) => set_vdst(cx, insn, Expr::Input(Input::Attr { attr, chan }))?,
            _ => return Err(form(insn, "vertex")),
        },
        Op::v_interp_p2_f16 | Op::v_interp_p2_hi_f16 => {
            let d = dst_reg(insn)?;
            let half = HALF + chan * 2 + u8::from(high);
            let value = Expr::Call(&TO_F16, vec![Expr::Input(Input::Attr { attr, chan: half })]);
            let upper = opsel & 8 != 0 || insn.op == Op::v_interp_p2_hi_f16;
            let old = Expr::Var(Var::V(d));
            let merged = if upper {
                Expr::bin(
                    Bin::Or,
                    Expr::bin(Bin::And, old, Expr::U32(0xffff)),
                    Expr::bin(Bin::Shl, value, Expr::U32(16)),
                )
            } else {
                Expr::bin(
                    Bin::Or,
                    Expr::bin(Bin::And, old, Expr::U32(0xffff_0000)),
                    value,
                )
            };
            cx.set_lane(Var::V(d), merged);
        }
        _ => return Ok(false),
    }
    Ok(true)
}
