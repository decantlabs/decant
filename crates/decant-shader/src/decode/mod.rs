//! The decoder: instruction words to [`Insn`] records.
//!
//! One record type serves every format, so that the assembler (`crate::asm`) can be the exact
//! inverse and the lifter matches on [`Op`] alone. The bit layouts are the public GFX10 ones,
//! read from the Mesa assembler (`aco_assembler.cpp`, MIT) as a description.
//!
//! Operand order follows the instruction syntax of the public ISA documents: `src[0]` is the
//! first source after the destination.

mod encode;
mod fields;
mod print;
mod words;

use crate::tables::{Format, Op, lookup};

pub use encode::encode;
pub use fields::{code_of_operand, operand_of_code};
pub use print::listing;
pub use words::{Dpp8, Dpp16, DppCtrl, Sdwa, SdwaSel, SdwaUnused};

/// A register an operand can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    /// Scalar register `s0` to `s105`.
    S(u8),
    /// Trap handler temporaries `ttmp0` to `ttmp15`.
    Ttmp(u8),
    /// Vector register `v0` to `v255`.
    V(u8),
    VccLo,
    VccHi,
    M0,
    /// The register that reads zero and drops writes.
    Null,
    ExecLo,
    ExecHi,
    /// The scalar condition code as a source.
    Scc,
    /// Whether VCC is zero.
    Vccz,
    /// Whether EXEC is zero.
    Execz,
    /// The aperture registers (codes 235 to 238).
    SharedBase,
    SharedLimit,
    PrivateBase,
    PrivateLimit,
    /// The ordered wave id of primitive ordered pixel shading (code 239).
    PopsExitingWaveId,
}

/// One operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operand {
    /// The slot is not used by this instruction.
    None,
    Reg(Reg),
    /// An inline integer constant, -16 to 64.
    Int(i8),
    /// An inline float constant, see [`FloatConst`]. Its bits depend on the width of the operation.
    Float(FloatConst),
    /// The literal word that follows the instruction.
    Lit(u32),
    /// An operand code the decoder does not know. Kept so that the census can count it.
    Unknown(u16),
}

/// The inline float constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatConst {
    Half,
    MinusHalf,
    One,
    MinusOne,
    Two,
    MinusTwo,
    Four,
    MinusFour,
    /// 1 / (2 * pi).
    InvTwoPi,
}

impl FloatConst {
    /// The value as a 32 bit float.
    pub fn f32(self) -> f32 {
        match self {
            FloatConst::Half => 0.5,
            FloatConst::MinusHalf => -0.5,
            FloatConst::One => 1.0,
            FloatConst::MinusOne => -1.0,
            FloatConst::Two => 2.0,
            FloatConst::MinusTwo => -2.0,
            FloatConst::Four => 4.0,
            FloatConst::MinusFour => -4.0,
            FloatConst::InvTwoPi => 0.159_154_94,
        }
    }
}

/// How a vector instruction was encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// The format the table names.
    Plain,
    /// A `Vop1`, `Vop2` or `Vopc` instruction in the two word `Vop3` encoding.
    AsVop3,
    /// With a sub dword word.
    Sdwa,
    /// With a 16 lane data parallel word.
    Dpp16,
    /// With an 8 lane data parallel word.
    Dpp8,
}

/// The fields of an instruction that are not registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ext {
    None,
    /// `Sopk` and `Sopp`: the 16 bit immediate. For branches it is the signed word distance from
    /// the next instruction.
    Imm16(u16),
    /// `Vop3` and `Vop3p` modifiers. For `Vop3p`, `abs` holds the high negate bits and `omod` the
    /// low two bits of the high selector, as they sit in the words.
    Valu {
        abs: u8,
        neg: u8,
        opsel: u8,
        omod: u8,
        clamp: bool,
    },
    /// The sub dword word, raw. `crate::decode::fields` has accessors.
    Sdwa(u32),
    /// The 16 lane data parallel word, raw.
    Dpp(u32),
    /// The 8 lane data parallel word, raw, and whether the source code asked to fetch inactive
    /// lanes (234 instead of 233).
    Dpp8 {
        word: u32,
        fetch_inactive: bool,
    },
    Smem {
        offset: u32,
        soffset: Operand,
        glc: bool,
        dlc: bool,
    },
    /// `Mubuf` and `Mtbuf`. `format` is the `Mtbuf` data format, 0 for `Mubuf`.
    Buf {
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
        slc: bool,
        dlc: bool,
        lds: bool,
        tfe: bool,
        format: u8,
    },
    Mimg(Mimg),
    Ds {
        offset0: u8,
        offset1: u8,
        gds: bool,
    },
    /// An interpolation instruction in the two word `Vop3` encoding: `v_interp_p1_f32`,
    /// `v_interp_p2_f32` and `v_interp_mov_f32` as `_e64`, and the 16 bit forms that exist only
    /// there. `src[0]` is the vector register (the parameter selector for the move), `src[1]` the
    /// third source of `v_interp_p1lv_f16` and `v_interp_p2_f16`. `abs` and `neg` use the source
    /// numbers as bits (bit 0 for `src[0]`, bit 2 for the third source).
    VintrpE64 {
        attr: u8,
        chan: u8,
        /// Use the high half of the attribute.
        high: bool,
        abs: u8,
        neg: u8,
        opsel: u8,
        omod: u8,
        clamp: bool,
    },
    Exp {
        enabled: u8,
        target: u8,
        compressed: bool,
        done: bool,
        valid_mask: bool,
    },
    Vintrp {
        attr: u8,
        chan: u8,
    },
    /// `Flat`, `Global` and `Scratch`: the two words, raw. The registers of the second word are
    /// also in the record: `dst` the result, `src` the address register (the first of a 64 bit
    /// pair per lane), the scalar base and the stored data. The 12 bit offset is in the low
    /// bits of the first word. The title uses the global loads to read tables behind its code.
    Flat(u32, u32),
}

/// The fields of an image instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mimg {
    pub dmask: u8,
    /// The dimension code (`SQ_RSRC_IMG_*` minus 8 on GFX10: 0 is 1D, 1 2D, 2 3D, 3 cube, 4 1D
    /// array, 5 2D array, 6 2D multisample, 7 2D multisample array).
    pub dim: u8,
    pub unrm: bool,
    pub glc: bool,
    pub slc: bool,
    pub dlc: bool,
    pub r128: bool,
    pub a16: bool,
    pub d16: bool,
    pub tfe: bool,
    pub lwe: bool,
    /// Address registers after the first, from the extra address words. `nsa_count` of them are used.
    pub nsa: [u8; 12],
    /// How many extra address words the instruction has (0 to 3).
    pub nsa_words: u8,
}

/// One decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Insn {
    /// Word index of the first word.
    pub pc: u32,
    /// Words the instruction takes, with literal and extra words.
    pub len: u8,
    pub op: Op,
    pub form: Form,
    /// The result, or the stored data register of a store. Instructions that write a fixed
    /// register (SCC) do not list it.
    pub dst: Operand,
    /// The second result of a `Vop3` instruction with a carry out.
    pub dst2: Operand,
    /// Sources in syntax order. Memory formats: `Smem` base (first of the pair), `Mubuf` and
    /// `Mtbuf` address, resource (first of four) and offset register, `Mimg` first address,
    /// resource (first of eight) and sampler (first of four), `Ds` address, data 0 and data 1,
    /// `Exp` its four sources.
    pub src: [Operand; 4],
    pub ext: Ext,
}

impl Insn {
    /// The table format of the instruction.
    pub fn format(&self) -> Format {
        self.op.info().format
    }

    /// The word index a branch goes to when taken.
    pub fn branch_target(&self) -> Option<u32> {
        match (self.op.info().class, self.ext) {
            (crate::tables::Class::Branch, Ext::Imm16(imm)) if self.format() == Format::Sopp => {
                Some((self.pc as i64 + 1 + i64::from(imm as i16)) as u32)
            }
            _ => None,
        }
    }
}

/// Why words do not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, thiserror::Error)]
pub enum DecodeError {
    #[error("word {word:#010x} at {pc} is no known {format:?} instruction (opcode {opcode:#x})")]
    UnknownOpcode {
        pc: u32,
        word: u32,
        format: Format,
        opcode: u16,
    },
    #[error("word {word:#010x} at {pc} has no known encoding")]
    UnknownEncoding { pc: u32, word: u32 },
    #[error("the instruction at {pc} runs past the end of the code")]
    Truncated { pc: u32 },
}

/// Decodes the instruction at word `pc`.
pub fn decode_one(words: &[u32], pc: u32) -> Result<Insn, DecodeError> {
    fields::decode(words, pc)
}

/// Decodes a whole program front to back. Decoding does not stop at `s_endpgm`: compilers
/// place far branch code after it.
pub fn decode_all(words: &[u32]) -> Result<Vec<Insn>, DecodeError> {
    let mut out = Vec::new();
    let mut pc = 0u32;
    while (pc as usize) < words.len() {
        let insn = decode_one(words, pc)?;
        pc += u32::from(insn.len);
        out.push(insn);
    }
    Ok(out)
}

/// Like [`decode_all`], but an undecodable word becomes an `Err` item and decoding goes on at the
/// next word. For the census.
pub fn decode_lossy(words: &[u32]) -> Vec<Result<Insn, DecodeError>> {
    let mut out = Vec::new();
    let mut pc = 0u32;
    while (pc as usize) < words.len() {
        match decode_one(words, pc) {
            Ok(insn) => {
                pc += u32::from(insn.len);
                out.push(Ok(insn));
            }
            Err(e) => {
                pc += 1;
                out.push(Err(e));
            }
        }
    }
    out
}

/// The opcode lookup with the `Vop3` forms of the one word vector formats.
pub(crate) fn vop3_lookup(opcode: u16) -> Option<(Op, Form)> {
    if let Some(op) = lookup(Format::Vop3, opcode) {
        return Some((op, Form::Plain));
    }
    let (format, number) = match opcode {
        0x000..=0x0FF => (Format::Vopc, opcode),
        0x200..=0x202 => (Format::Vintrp, opcode - 0x200),
        0x100..=0x13F => (Format::Vop2, opcode - 0x100),
        0x180..=0x1FF => (Format::Vop1, opcode - 0x180),
        _ => return None,
    };
    lookup(format, number).map(|op| (op, Form::AsVop3))
}
