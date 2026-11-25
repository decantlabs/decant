//! The encoding side: an [`Insn`] back to words. For every instruction the decoder returns,
//! `decode_one(&encode(&insn), 0)` gives the instruction back (with `pc` 0). The assembler and
//! the round trip tests rest on that.

use super::fields::{carry_out, code_of_operand, encoded_sources, scalar_result};
use super::{Ext, Form, Insn, Operand, Reg};
use crate::tables::{Format, Op};

fn code(op: Operand) -> u32 {
    u32::from(code_of_operand(op))
}

/// The 8 bit register number of a vector or scalar operand.
fn low8(op: Operand) -> u32 {
    if op == Operand::None {
        0
    } else {
        code(op) & 0xFF
    }
}

fn bit(flag: bool, at: u32) -> u32 {
    u32::from(flag) << at
}

fn literal(src: &[Operand; 4]) -> Option<u32> {
    src.iter().find_map(|s| {
        if let Operand::Lit(v) = s {
            Some(*v)
        } else {
            None
        }
    })
}

fn quad(op: Operand) -> u32 {
    match op {
        Operand::Reg(Reg::S(n)) => u32::from(n >> 2),
        _ => 0,
    }
}

/// The words of one instruction. Fields the record does not hold are zero.
pub fn encode(insn: &Insn) -> Vec<u32> {
    let info = insn.op.info();
    let number = u32::from(info.opcode);
    let (dst, dst2, src) = (insn.dst, insn.dst2, insn.src);
    let mut out = Vec::with_capacity(4);
    match info.format {
        Format::Sop2 => out.push(
            (0b10 << 30)
                | (number << 23)
                | ((code(dst) & 0x7F) << 16)
                | (low8(src[1]) << 8)
                | low8(src[0]),
        ),
        Format::Sopk => {
            let Ext::Imm16(imm) = insn.ext else {
                unreachable!("a Sopk record carries Imm16")
            };
            out.push((0b1011 << 28) | (number << 23) | ((code(dst) & 0x7F) << 16) | u32::from(imm));
        }
        Format::Sop1 => out.push(
            (0b1_0111_1101 << 23) | ((code(dst) & 0x7F) << 16) | (number << 8) | low8(src[0]),
        ),
        Format::Sopc => {
            out.push((0b1_0111_1110 << 23) | (number << 16) | (low8(src[1]) << 8) | low8(src[0]))
        }
        Format::Sopp => {
            let Ext::Imm16(imm) = insn.ext else {
                unreachable!("a Sopp record carries Imm16")
            };
            out.push((0b1_0111_1111 << 23) | (number << 16) | u32::from(imm));
        }
        Format::Smem => {
            let Ext::Smem {
                offset,
                soffset,
                glc,
                dlc,
            } = insn.ext
            else {
                unreachable!("a Smem record carries Smem")
            };
            let base = match src[0] {
                Operand::Reg(Reg::S(n)) => u32::from(n >> 1),
                _ => 0,
            };
            out.push(
                (0b111101 << 26)
                    | (number << 18)
                    | bit(glc, 16)
                    | bit(dlc, 14)
                    | ((code(dst) & 0x7F) << 6)
                    | base,
            );
            out.push((offset & 0x1F_FFFF) | ((code(soffset) & 0x7F) << 25));
        }
        Format::Vintrp | Format::Vop3 if matches!(insn.ext, Ext::VintrpE64 { .. }) => {
            let Ext::VintrpE64 {
                attr,
                chan,
                high,
                abs,
                neg,
                opsel,
                omod,
                clamp,
            } = insn.ext
            else {
                unreachable!()
            };
            let number = if info.format == Format::Vintrp {
                number + 0x200
            } else {
                number
            };
            out.push(
                (0b110101 << 26)
                    | (number << 16)
                    | bit(clamp, 15)
                    | (u32::from(opsel & 0xF) << 11)
                    | bit(abs & 4 != 0, 10)
                    | bit(abs & 1 != 0, 9)
                    | low8(dst),
            );
            let third = if src[1] == Operand::None {
                128
            } else {
                code(src[1])
            };
            out.push(
                u32::from(attr & 0x3F)
                    | (u32::from(chan & 3) << 6)
                    | bit(high, 8)
                    | (match src[0] {
                        Operand::Int(v) if insn.op == Op::v_interp_mov_f32 => v as u32 & 3,
                        other => code(other),
                    } << 9)
                    | (third << 18)
                    | (u32::from(omod & 3) << 27)
                    | bit(neg & 1 != 0, 30)
                    | bit(neg & 4 != 0, 31),
            );
        }
        Format::Vop1 | Format::Vop2 | Format::Vopc if insn.form != Form::AsVop3 => {
            let (first, extra) = match (insn.form, insn.ext) {
                (Form::Sdwa, Ext::Sdwa(x)) => (249, Some(x)),
                (Form::Dpp16, Ext::Dpp(x)) => (250, Some(x)),
                (
                    Form::Dpp8,
                    Ext::Dpp8 {
                        word,
                        fetch_inactive,
                    },
                ) => (233 + u32::from(fetch_inactive), Some(word)),
                _ => (code(src[0]), None),
            };
            out.push(match info.format {
                Format::Vop1 => (0b011_1111 << 25) | (low8(dst) << 17) | (number << 9) | first,
                Format::Vopc => (0b011_1110 << 25) | (number << 17) | (low8(src[1]) << 9) | first,
                _ => (number << 25) | (low8(dst) << 17) | (low8(src[1]) << 9) | first,
            });
            out.extend(extra);
        }
        Format::Vop1 | Format::Vop2 | Format::Vopc | Format::Vop3 => {
            let Ext::Valu {
                abs,
                neg,
                opsel,
                omod,
                clamp,
            } = insn.ext
            else {
                unreachable!("a Vop3 record carries Valu")
            };
            let number = match (insn.form, info.format) {
                (Form::AsVop3, Format::Vop2) => number + 0x100,
                (Form::AsVop3, Format::Vop1) => number + 0x180,
                _ => number,
            };
            let second = if carry_out(insn.op) {
                (code(dst2) & 0x7F) << 8
            } else {
                u32::from(abs & 7) << 8
            };
            let first = if info.defs.is_empty() { 0 } else { low8(dst) };
            debug_assert!(!scalar_result(insn.op) || !matches!(dst, Operand::Reg(Reg::V(_))));
            out.push(
                (0b110101 << 26)
                    | (number << 16)
                    | bit(clamp, 15)
                    | if carry_out(insn.op) {
                        0
                    } else {
                        u32::from(opsel & 0xF) << 11
                    }
                    | second
                    | first,
            );
            let mut w1 = (u32::from(omod & 3) << 27) | (u32::from(neg & 7) << 29);
            for (i, operand) in src.iter().enumerate().take(3) {
                w1 |= if i < encoded_sources(insn.op) {
                    code(*operand)
                } else {
                    128
                } << (9 * i);
            }
            out.push(w1);
        }
        Format::Vop3p => {
            let Ext::Valu {
                abs,
                neg,
                opsel,
                omod,
                clamp,
            } = insn.ext
            else {
                unreachable!("a Vop3p record carries Valu")
            };
            out.push(
                (0b110011 << 26)
                    | (number << 16)
                    | bit(clamp, 15)
                    | (u32::from(opsel & 0xF) << 11)
                    | (u32::from(abs & 7) << 8)
                    | low8(dst),
            );
            let mut w1 = (u32::from(omod & 3) << 27) | (u32::from(neg & 7) << 29);
            for (i, operand) in src.iter().enumerate().take(3) {
                w1 |= if i < encoded_sources(insn.op) {
                    code(*operand)
                } else {
                    128
                } << (9 * i);
            }
            out.push(w1);
        }
        Format::Vintrp => {
            let Ext::Vintrp { attr, chan } = insn.ext else {
                unreachable!("a Vintrp record carries Vintrp")
            };
            let first = match src[0] {
                Operand::Int(v) => v as u32 & 3,
                other => low8(other),
            };
            out.push(
                (0b110010 << 26)
                    | (low8(dst) << 18)
                    | (number << 16)
                    | (u32::from(attr & 0x3F) << 10)
                    | (u32::from(chan & 3) << 8)
                    | first,
            );
        }
        Format::Ds => {
            let Ext::Ds {
                offset0,
                offset1,
                gds,
            } = insn.ext
            else {
                unreachable!("a Ds record carries Ds")
            };
            out.push(
                (0b110110 << 26)
                    | (number << 18)
                    | bit(gds, 17)
                    | (u32::from(offset1) << 8)
                    | u32::from(offset0),
            );
            out.push((low8(dst) << 24) | (low8(src[2]) << 16) | (low8(src[1]) << 8) | low8(src[0]));
        }
        Format::Flat | Format::Global | Format::Scratch => {
            let Ext::Flat(w0, w1) = insn.ext else {
                unreachable!("a Flat record carries Flat")
            };
            out.extend([w0, w1]);
        }
        Format::Mubuf | Format::Mtbuf => {
            let Ext::Buf {
                offset,
                offen,
                idxen,
                glc,
                slc,
                dlc,
                lds,
                tfe,
                format,
            } = insn.ext
            else {
                unreachable!("a buffer record carries Buf")
            };
            let flags = bit(offen, 12)
                | bit(idxen, 13)
                | bit(glc, 14)
                | bit(dlc, 15)
                | u32::from(offset & 0xFFF);
            let w1 = (low8(src[2]) << 24)
                | bit(tfe, 23)
                | bit(slc, 22)
                | (quad(src[1]) << 16)
                | (low8(dst) << 8)
                | low8(src[0]);
            if info.format == Format::Mubuf {
                out.extend([(0b111000 << 26) | (number << 18) | bit(lds, 16) | flags, w1]);
            } else {
                out.extend([
                    (0b111010 << 26)
                        | (u32::from(format & 0x7F) << 19)
                        | ((number & 7) << 16)
                        | flags,
                    w1 | (((number >> 3) & 1) << 21),
                ]);
            }
        }
        Format::Mimg => {
            let Ext::Mimg(m) = insn.ext else {
                unreachable!("an image record carries Mimg")
            };
            out.push(
                (0b111100 << 26)
                    | bit(m.slc, 25)
                    | ((number & 0x7F) << 18)
                    | bit(m.lwe, 17)
                    | bit(m.tfe, 16)
                    | bit(m.r128, 15)
                    | bit(m.glc, 13)
                    | bit(m.unrm, 12)
                    | (u32::from(m.dmask & 0xF) << 8)
                    | bit(m.dlc, 7)
                    | (u32::from(m.dim & 7) << 3)
                    | (u32::from(m.nsa_words & 3) << 1)
                    | ((number >> 7) & 1),
            );
            out.push(
                bit(m.d16, 31)
                    | bit(m.a16, 30)
                    | (quad(src[2]) << 21)
                    | (quad(src[1]) << 16)
                    | (low8(dst) << 8)
                    | low8(src[0]),
            );
            for k in 0..usize::from(m.nsa_words) {
                out.push(u32::from_le_bytes([
                    m.nsa[k * 4],
                    m.nsa[k * 4 + 1],
                    m.nsa[k * 4 + 2],
                    m.nsa[k * 4 + 3],
                ]));
            }
        }
        Format::Exp => {
            let Ext::Exp {
                enabled,
                target,
                compressed,
                done,
                valid_mask,
            } = insn.ext
            else {
                unreachable!("an export record carries Exp")
            };
            out.push(
                (0b111110 << 26)
                    | bit(valid_mask, 12)
                    | bit(done, 11)
                    | bit(compressed, 10)
                    | (u32::from(target & 0x3F) << 4)
                    | u32::from(enabled & 0xF),
            );
            out.push(
                low8(src[0]) | (low8(src[1]) << 8) | (low8(src[2]) << 16) | (low8(src[3]) << 24),
            );
        }
    }
    if let Some(value) = literal(&src) {
        out.push(value);
    }
    out
}
