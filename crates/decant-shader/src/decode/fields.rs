//! The bit fields of every format, decoding side. `encode.rs` is the inverse.

use super::{DecodeError, Ext, FloatConst, Form, Insn, Mimg, Operand, Reg, vop3_lookup};
use crate::tables::{Base, Fixed, Format, Op, lookup};

/// The operand a 9 bit source code names. Code 255 (a literal follows) gives `Lit(0)`, the
/// caller fills the value in. Codes 249, 250, 233 and 234 select an extra word and are handled
/// before this is called.
pub fn operand_of_code(code: u16) -> Operand {
    match code {
        0..=105 => Operand::Reg(Reg::S(code as u8)),
        106 => Operand::Reg(Reg::VccLo),
        107 => Operand::Reg(Reg::VccHi),
        108..=123 => Operand::Reg(Reg::Ttmp((code - 108) as u8)),
        124 => Operand::Reg(Reg::M0),
        125 => Operand::Reg(Reg::Null),
        126 => Operand::Reg(Reg::ExecLo),
        127 => Operand::Reg(Reg::ExecHi),
        128..=192 => Operand::Int((code - 128) as i8),
        193..=208 => Operand::Int(-((code - 192) as i8)),
        235 => Operand::Reg(Reg::SharedBase),
        236 => Operand::Reg(Reg::SharedLimit),
        237 => Operand::Reg(Reg::PrivateBase),
        238 => Operand::Reg(Reg::PrivateLimit),
        239 => Operand::Reg(Reg::PopsExitingWaveId),
        240 => Operand::Float(FloatConst::Half),
        241 => Operand::Float(FloatConst::MinusHalf),
        242 => Operand::Float(FloatConst::One),
        243 => Operand::Float(FloatConst::MinusOne),
        244 => Operand::Float(FloatConst::Two),
        245 => Operand::Float(FloatConst::MinusTwo),
        246 => Operand::Float(FloatConst::Four),
        247 => Operand::Float(FloatConst::MinusFour),
        248 => Operand::Float(FloatConst::InvTwoPi),
        251 => Operand::Reg(Reg::Vccz),
        252 => Operand::Reg(Reg::Execz),
        253 => Operand::Reg(Reg::Scc),
        255 => Operand::Lit(0),
        256..=511 => Operand::Reg(Reg::V((code - 256) as u8)),
        _ => Operand::Unknown(code),
    }
}

/// The 9 bit code of an operand, the inverse of [`operand_of_code`]. `None` has code 128 (zero),
/// which is what compilers put in unused `Vop3` sources.
pub fn code_of_operand(op: Operand) -> u16 {
    match op {
        Operand::None => 128,
        Operand::Reg(Reg::S(n)) => u16::from(n),
        Operand::Reg(Reg::VccLo) => 106,
        Operand::Reg(Reg::VccHi) => 107,
        Operand::Reg(Reg::Ttmp(n)) => 108 + u16::from(n),
        Operand::Reg(Reg::M0) => 124,
        Operand::Reg(Reg::Null) => 125,
        Operand::Reg(Reg::ExecLo) => 126,
        Operand::Reg(Reg::ExecHi) => 127,
        Operand::Int(v @ 0..=64) => 128 + v as u16,
        Operand::Int(v) => (192 - i16::from(v)) as u16,
        Operand::Float(f) => 240 + f as u16,
        Operand::Reg(Reg::SharedBase) => 235,
        Operand::Reg(Reg::SharedLimit) => 236,
        Operand::Reg(Reg::PrivateBase) => 237,
        Operand::Reg(Reg::PrivateLimit) => 238,
        Operand::Reg(Reg::PopsExitingWaveId) => 239,
        Operand::Reg(Reg::Vccz) => 251,
        Operand::Reg(Reg::Execz) => 252,
        Operand::Reg(Reg::Scc) => 253,
        Operand::Lit(_) => 255,
        Operand::Reg(Reg::V(n)) => 256 + u16::from(n),
        Operand::Unknown(code) => code,
    }
}

fn scalar(code: u32) -> Operand {
    operand_of_code((code & 0xFF) as u16)
}

fn vector(code: u32) -> Operand {
    Operand::Reg(Reg::V((code & 0xFF) as u8))
}

struct Reader<'a> {
    words: &'a [u32],
    pc: u32,
    at: u32,
}

impl Reader<'_> {
    fn next(&mut self) -> Result<u32, DecodeError> {
        let w = self
            .words
            .get(self.at as usize)
            .copied()
            .ok_or(DecodeError::Truncated { pc: self.pc })?;
        self.at += 1;
        Ok(w)
    }

    /// Replaces every `Lit(0)` placeholder among the sources by the literal word. One literal
    /// word serves every source that names it.
    fn literal(&mut self, src: &mut [Operand; 4]) -> Result<(), DecodeError> {
        if src.iter().any(|s| matches!(s, Operand::Lit(_))) {
            let value = Operand::Lit(self.next()?);
            for s in src.iter_mut().filter(|s| matches!(s, Operand::Lit(_))) {
                *s = value;
            }
        }
        Ok(())
    }

    fn done(
        self,
        op: Op,
        form: Form,
        dst: Operand,
        dst2: Operand,
        src: [Operand; 4],
        ext: Ext,
    ) -> Insn {
        Insn {
            pc: self.pc,
            len: (self.at - self.pc) as u8,
            op,
            form,
            dst,
            dst2,
            src,
            ext,
        }
    }
}

fn op(format: Format, opcode: u32, pc: u32, word: u32) -> Result<Op, DecodeError> {
    lookup(format, opcode as u16).ok_or(DecodeError::UnknownOpcode {
        pc,
        word,
        format,
        opcode: opcode as u16,
    })
}

const NONE: Operand = Operand::None;

/// How many sources of a `Vop3` or `Vop3p` instruction are in the source fields: every slot
/// that is not a fixed register, and a lane mask (the mask of a select, a carry in).
pub(super) fn encoded_sources(o: Op) -> usize {
    o.info()
        .srcs
        .iter()
        .filter(|s| s.fixed == Fixed::No || s.base == Base::LaneMask)
        .count()
        .min(3)
}

/// Whether the first result of `op` is a scalar register even though the format has a vector
/// destination field.
pub(super) fn scalar_result(o: Op) -> bool {
    matches!(o, Op::v_readfirstlane_b32 | Op::v_readlane_b32_e64)
        || o.info()
            .defs
            .first()
            .is_some_and(|d| d.base == Base::LaneMask)
}

/// Whether a `Vop3` instruction has a second result in bits 8 to 14 (a carry out).
pub(super) fn carry_out(o: Op) -> bool {
    o.info().defs.len() == 2 && o.info().format != Format::Vopc
}

pub(super) fn decode(words: &[u32], pc: u32) -> Result<Insn, DecodeError> {
    let mut r = Reader { words, pc, at: pc };
    let w = r.next()?;
    if w >> 31 == 0 {
        return vector_one_word(r, w);
    }
    if w >> 30 == 0b10 {
        return scalar_word(r, w);
    }
    match w >> 26 {
        0b110101 => vop3(r, w),
        0b110011 => vop3p(r, w),
        0b110010 => {
            let o = op(Format::Vintrp, (w >> 16) & 3, pc, w)?;
            let ext = Ext::Vintrp {
                attr: ((w >> 10) & 0x3F) as u8,
                chan: ((w >> 8) & 3) as u8,
            };
            // The move form selects a vertex with a 2 bit code instead of a register.
            let src0 = if o == Op::v_interp_mov_f32 {
                Operand::Int((w & 3) as i8)
            } else {
                vector(w)
            };
            Ok(r.done(
                o,
                Form::Plain,
                vector(w >> 18),
                NONE,
                [src0, NONE, NONE, NONE],
                ext,
            ))
        }
        0b110110 => {
            let o = op(Format::Ds, (w >> 18) & 0xFF, pc, w)?;
            let w1 = r.next()?;
            let ext = Ext::Ds {
                offset0: (w & 0xFF) as u8,
                offset1: ((w >> 8) & 0xFF) as u8,
                gds: (w >> 17) & 1 != 0,
            };
            Ok(r.done(
                o,
                Form::Plain,
                vector(w1 >> 24),
                NONE,
                [vector(w1), vector(w1 >> 8), vector(w1 >> 16), NONE],
                ext,
            ))
        }
        0b110111 => {
            let format = match (w >> 14) & 3 {
                1 => Format::Scratch,
                2 => Format::Global,
                _ => Format::Flat,
            };
            let o = op(format, (w >> 18) & 0x7F, pc, w)?;
            let w1 = r.next()?;
            // The second word: the address register (the first of a 64 bit pair per lane), the
            // stored data, the scalar base (null when unused) and the result register.
            let src = [vector(w1), scalar((w1 >> 16) & 0x7F), vector(w1 >> 8), NONE];
            Ok(r.done(
                o,
                Form::Plain,
                vector(w1 >> 24),
                NONE,
                src,
                Ext::Flat(w, w1),
            ))
        }
        0b111000 | 0b111010 => {
            let mtbuf = w >> 26 == 0b111010;
            let w1 = r.next()?;
            let o = if mtbuf {
                op(
                    Format::Mtbuf,
                    ((w >> 16) & 7) | (((w1 >> 21) & 1) << 3),
                    pc,
                    w,
                )?
            } else {
                op(Format::Mubuf, (w >> 18) & 0xFF, pc, w)?
            };
            let ext = Ext::Buf {
                offset: (w & 0xFFF) as u16,
                offen: (w >> 12) & 1 != 0,
                idxen: (w >> 13) & 1 != 0,
                glc: (w >> 14) & 1 != 0,
                dlc: (w >> 15) & 1 != 0,
                lds: !mtbuf && (w >> 16) & 1 != 0,
                slc: (w1 >> 22) & 1 != 0,
                tfe: (w1 >> 23) & 1 != 0,
                format: if mtbuf { ((w >> 19) & 0x7F) as u8 } else { 0 },
            };
            let resource = Operand::Reg(Reg::S((((w1 >> 16) & 0x1F) << 2) as u8));
            Ok(r.done(
                o,
                Form::Plain,
                vector(w1 >> 8),
                NONE,
                [vector(w1), resource, scalar(w1 >> 24), NONE],
                ext,
            ))
        }
        0b111100 => {
            let o = op(Format::Mimg, ((w >> 18) & 0x7F) | ((w & 1) << 7), pc, w)?;
            let w1 = r.next()?;
            let nsa_words = ((w >> 1) & 3) as u8;
            let mut nsa = [0u8; 12];
            for k in 0..usize::from(nsa_words) {
                let extra = r.next()?;
                nsa[k * 4..k * 4 + 4].copy_from_slice(&extra.to_le_bytes());
            }
            let ext = Ext::Mimg(Mimg {
                dmask: ((w >> 8) & 0xF) as u8,
                dim: ((w >> 3) & 7) as u8,
                unrm: (w >> 12) & 1 != 0,
                glc: (w >> 13) & 1 != 0,
                slc: (w >> 25) & 1 != 0,
                dlc: (w >> 7) & 1 != 0,
                r128: (w >> 15) & 1 != 0,
                a16: (w1 >> 30) & 1 != 0,
                d16: (w1 >> 31) & 1 != 0,
                tfe: (w >> 16) & 1 != 0,
                lwe: (w >> 17) & 1 != 0,
                nsa,
                nsa_words,
            });
            let resource = Operand::Reg(Reg::S((((w1 >> 16) & 0x1F) << 2) as u8));
            let sampler = Operand::Reg(Reg::S((((w1 >> 21) & 0x1F) << 2) as u8));
            Ok(r.done(
                o,
                Form::Plain,
                vector(w1 >> 8),
                NONE,
                [vector(w1), resource, sampler, NONE],
                ext,
            ))
        }
        0b111101 => {
            let o = op(Format::Smem, (w >> 18) & 0xFF, pc, w)?;
            let w1 = r.next()?;
            let ext = Ext::Smem {
                offset: w1 & 0x1F_FFFF,
                soffset: scalar(w1 >> 25),
                glc: (w >> 16) & 1 != 0,
                dlc: (w >> 14) & 1 != 0,
            };
            let base = Operand::Reg(Reg::S(((w & 0x3F) << 1) as u8));
            Ok(r.done(
                o,
                Form::Plain,
                scalar((w >> 6) & 0x7F),
                NONE,
                [base, NONE, NONE, NONE],
                ext,
            ))
        }
        0b111110 => {
            let w1 = r.next()?;
            let ext = Ext::Exp {
                enabled: (w & 0xF) as u8,
                target: ((w >> 4) & 0x3F) as u8,
                compressed: (w >> 10) & 1 != 0,
                done: (w >> 11) & 1 != 0,
                valid_mask: (w >> 12) & 1 != 0,
            };
            Ok(r.done(
                Op::exp,
                Form::Plain,
                NONE,
                NONE,
                [
                    vector(w1),
                    vector(w1 >> 8),
                    vector(w1 >> 16),
                    vector(w1 >> 24),
                ],
                ext,
            ))
        }
        _ => Err(DecodeError::UnknownEncoding { pc, word: w }),
    }
}

fn scalar_word(mut r: Reader, w: u32) -> Result<Insn, DecodeError> {
    let pc = r.pc;
    let (s0, s1, dst) = (
        operand_of_code((w & 0xFF) as u16),
        operand_of_code(((w >> 8) & 0xFF) as u16),
        scalar((w >> 16) & 0x7F),
    );
    match w >> 23 {
        0b1_0111_1101 => {
            let o = op(Format::Sop1, (w >> 8) & 0xFF, pc, w)?;
            let mut src = [s0, NONE, NONE, NONE];
            r.literal(&mut src)?;
            Ok(r.done(o, Form::Plain, dst, NONE, src, Ext::None))
        }
        0b1_0111_1110 => {
            let o = op(Format::Sopc, (w >> 16) & 0x7F, pc, w)?;
            let mut src = [s0, s1, NONE, NONE];
            r.literal(&mut src)?;
            Ok(r.done(o, Form::Plain, NONE, NONE, src, Ext::None))
        }
        0b1_0111_1111 => {
            let o = op(Format::Sopp, (w >> 16) & 0x7F, pc, w)?;
            Ok(r.done(o, Form::Plain, NONE, NONE, [NONE; 4], Ext::Imm16(w as u16)))
        }
        _ if w >> 28 == 0b1011 => {
            let o = op(Format::Sopk, (w >> 23) & 0x1F, pc, w)?;
            let mut src = [NONE; 4];
            if o == Op::s_setreg_imm32_b32 {
                src[0] = Operand::Lit(r.next()?);
            }
            Ok(r.done(o, Form::Plain, dst, NONE, src, Ext::Imm16(w as u16)))
        }
        _ => {
            let o = op(Format::Sop2, (w >> 23) & 0x7F, pc, w)?;
            let mut src = [s0, s1, NONE, NONE];
            r.literal(&mut src)?;
            Ok(r.done(o, Form::Plain, dst, NONE, src, Ext::None))
        }
    }
}

/// Handles the extra word that source code 249, 250, 233 or 234 selects. Returns the form, the
/// real first source and the raw word.
fn extra_word(r: &mut Reader, src0: u32) -> Result<Option<(Form, Operand, Ext)>, DecodeError> {
    let form = match src0 {
        249 => Form::Sdwa,
        250 => Form::Dpp16,
        233 | 234 => Form::Dpp8,
        _ => return Ok(None),
    };
    let x = r.next()?;
    let first = match form {
        // Bit 23 says that the first source is a scalar register.
        Form::Sdwa if (x >> 23) & 1 != 0 => scalar(x),
        _ => vector(x),
    };
    Ok(Some((
        form,
        first,
        match form {
            Form::Sdwa => Ext::Sdwa(x),
            Form::Dpp8 => Ext::Dpp8 {
                word: x,
                fetch_inactive: src0 == 234,
            },
            _ => Ext::Dpp(x),
        },
    )))
}

fn vector_one_word(mut r: Reader, w: u32) -> Result<Insn, DecodeError> {
    let pc = r.pc;
    let src0_code = w & 0x1FF;
    let (format, number) = match w >> 25 {
        0b011_1111 => (Format::Vop1, (w >> 9) & 0xFF),
        0b011_1110 => (Format::Vopc, (w >> 17) & 0xFF),
        number => (Format::Vop2, number),
    };
    let o = op(format, number, pc, w)?;
    let (form, first, ext) = extra_word(&mut r, src0_code)?.unwrap_or((
        Form::Plain,
        operand_of_code(src0_code as u16),
        Ext::None,
    ));
    let mut src = [first, NONE, NONE, NONE];
    let mut dst = vector(w >> 17);
    match format {
        Format::Vop1 => {
            if scalar_result(o) {
                dst = scalar((w >> 17) & 0x7F);
            }
            if o.info().defs.is_empty() {
                dst = NONE;
            }
            if o.info().srcs.is_empty() {
                src[0] = NONE;
            }
        }
        Format::Vopc => {
            src[1] = vector(w >> 9);
            dst = Operand::Reg(if o.name().starts_with("v_cmpx") {
                Reg::ExecLo
            } else {
                Reg::VccLo
            });
            if let Ext::Sdwa(x) = ext {
                // Bit 15 says that the result goes to the scalar register in bits 8 to 14.
                if (x >> 15) & 1 != 0 {
                    dst = scalar((x >> 8) & 0x7F);
                }
                if (x >> 31) & 1 != 0 {
                    src[1] = scalar(w >> 9);
                }
            }
        }
        _ => {
            src[1] = vector(w >> 9);
            if let Ext::Sdwa(x) = ext
                && (x >> 31) & 1 != 0
            {
                src[1] = scalar(w >> 9);
            }
        }
    }
    // The `ak` and `mk` multiply add forms always take the literal word as their constant.
    if constant_operand(o) {
        src[2] = Operand::Lit(0);
    }
    r.literal(&mut src)?;
    Ok(r.done(o, form, dst, NONE, src, ext))
}

/// Whether the destination register is also the last source and the `Vop3` field of that source
/// is not used (the multiply accumulate forms, `v_writelane_b32`).
pub(super) fn accumulates(o: Op) -> bool {
    matches!(
        o,
        Op::v_mac_f32
            | Op::v_mac_legacy_f32
            | Op::v_fmac_f32
            | Op::v_fmac_f16
            | Op::v_fmac_legacy_f32
            | Op::v_writelane_b32_e64
    )
}

/// Whether a two operand instruction has a third operand that is always the literal word.
pub(super) fn constant_operand(o: Op) -> bool {
    matches!(
        o,
        Op::v_madmk_f32
            | Op::v_madak_f32
            | Op::v_fmamk_f32
            | Op::v_fmaak_f32
            | Op::v_fmamk_f16
            | Op::v_fmaak_f16
    )
}

fn vop3(mut r: Reader, w: u32) -> Result<Insn, DecodeError> {
    let pc = r.pc;
    let number = ((w >> 16) & 0x3FF) as u16;
    let (o, form) = vop3_lookup(number).ok_or(DecodeError::UnknownOpcode {
        pc,
        word: w,
        format: Format::Vop3,
        opcode: number,
    })?;
    let w1 = r.next()?;
    if o.name().starts_with("v_interp_") {
        return Ok(vop3_interp(r, w, w1, o, form));
    }
    let two = carry_out(o);
    let dst = if o.info().defs.is_empty() {
        NONE
    } else if scalar_result(o) {
        scalar(w & 0xFF)
    } else {
        vector(w)
    };
    let dst2 = if two { scalar((w >> 8) & 0x7F) } else { NONE };
    let mut src = [NONE; 4];
    for (i, slot) in src.iter_mut().enumerate().take(encoded_sources(o)) {
        *slot = operand_of_code(((w1 >> (9 * i)) & 0x1FF) as u16);
    }
    r.literal(&mut src)?;
    if accumulates(o) {
        // The destination is the third source: the field is not used.
        src[2] = NONE;
    }
    let ext = Ext::Valu {
        abs: if two { 0 } else { ((w >> 8) & 7) as u8 },
        neg: (w1 >> 29) as u8,
        // For the carry forms bits 8 to 14 are the second result.
        opsel: if two { 0 } else { ((w >> 11) & 0xF) as u8 },
        omod: ((w1 >> 27) & 3) as u8,
        clamp: (w >> 15) & 1 != 0,
    };
    Ok(r.done(o, form, dst, dst2, src, ext))
}

/// The interpolation instructions in the `Vop3` encoding put the attribute in the second word
/// and the vector source after it, so they do not follow the generic source layout.
fn vop3_interp(r: Reader, w: u32, w1: u32, o: Op, form: Form) -> Insn {
    let source = |at: u32| operand_of_code(((w1 >> at) & 0x1FF) as u16);
    let mut src = [source(9), NONE, NONE, NONE];
    if o == Op::v_interp_mov_f32 {
        // The parameter selector is the low two bits of the source field.
        src[0] = Operand::Int(((w1 >> 9) & 3) as i8);
    }
    if matches!(
        o,
        Op::v_interp_p1lv_f16 | Op::v_interp_p2_f16 | Op::v_interp_p2_hi_f16
    ) {
        src[1] = source(18);
    }
    let ext = Ext::VintrpE64 {
        attr: (w1 & 0x3F) as u8,
        chan: ((w1 >> 6) & 3) as u8,
        high: (w1 >> 8) & 1 != 0,
        abs: (((w >> 9) & 1) | (((w >> 10) & 1) << 2)) as u8,
        neg: (((w1 >> 30) & 1) | (((w1 >> 31) & 1) << 2)) as u8,
        opsel: ((w >> 11) & 0xF) as u8,
        omod: ((w1 >> 27) & 3) as u8,
        clamp: (w >> 15) & 1 != 0,
    };
    r.done(o, form, vector(w), NONE, src, ext)
}

fn vop3p(mut r: Reader, w: u32) -> Result<Insn, DecodeError> {
    let pc = r.pc;
    let o = op(Format::Vop3p, (w >> 16) & 0x7F, pc, w)?;
    let w1 = r.next()?;
    let mut src = [NONE; 4];
    for (i, slot) in src.iter_mut().enumerate().take(encoded_sources(o)) {
        *slot = operand_of_code(((w1 >> (9 * i)) & 0x1FF) as u16);
    }
    r.literal(&mut src)?;
    let ext = Ext::Valu {
        abs: ((w >> 8) & 7) as u8,
        neg: (w1 >> 29) as u8,
        opsel: ((w >> 11) & 0xF) as u8,
        omod: ((w1 >> 27) & 3) as u8,
        clamp: (w >> 15) & 1 != 0,
    };
    Ok(r.done(o, Form::Plain, vector(w), NONE, src, ext))
}
