//! An assembler for synthetic test programs.
//!
//! Tests in the repository never use game data. They build programs here: [`Asm`] collects
//! instructions (as the records the decoder returns, encoded by `decode::encode`), resolves
//! branch labels, and puts the six word signature in front, so the result is what a
//! `StageProgram` carries. The constructor functions cover the formats, one per format.
//!
//! ```
//! use decant_shader::asm::*;
//! use decant_shader::tables::Op;
//! let mut a = Asm::new();
//! a.push(vop2(Op::v_add_f32, v(2), v(0), v(1)));
//! a.push(sopp(Op::s_endpgm, 0));
//! let code = a.finish();
//! assert_eq!(code.len(), 6 + 2);
//! ```

use crate::decode::{Ext, FloatConst, Form, Insn, Mimg, Operand, Reg, encode};
use crate::tables::Op;

/// The signature words: two do nothing instructions in the two word encoding, each followed by
/// a data word. Real programs carry other payload bits, the shape is the same.
pub const SIGNATURE: [u32; 6] = [0xD580_0000, 0x03FF_0000, 0, 0xD580_0000, 0x03FF_0000, 0];

pub fn s(n: u8) -> Operand {
    Operand::Reg(Reg::S(n))
}

pub fn v(n: u8) -> Operand {
    Operand::Reg(Reg::V(n))
}

/// An inline integer, -16 to 64.
pub fn int(value: i8) -> Operand {
    assert!((-16..=64).contains(&value), "inline integers are -16 to 64");
    Operand::Int(value)
}

/// A literal word.
pub fn lit(value: u32) -> Operand {
    Operand::Lit(value)
}

/// A float: inline when the machine has the constant, else a literal.
pub fn float(value: f32) -> Operand {
    let inline = [
        FloatConst::Half,
        FloatConst::MinusHalf,
        FloatConst::One,
        FloatConst::MinusOne,
        FloatConst::Two,
        FloatConst::MinusTwo,
        FloatConst::Four,
        FloatConst::MinusFour,
    ];
    if value.to_bits() == 0 {
        return Operand::Int(0);
    }
    inline
        .into_iter()
        .find(|c| c.f32().to_bits() == value.to_bits())
        .map_or(Operand::Lit(value.to_bits()), Operand::Float)
}

pub const EXEC: Operand = Operand::Reg(Reg::ExecLo);
pub const VCC: Operand = Operand::Reg(Reg::VccLo);
pub const M0: Operand = Operand::Reg(Reg::M0);
const NONE: Operand = Operand::None;

fn insn(op: Op, form: Form, dst: Operand, src: [Operand; 4], ext: Ext) -> Insn {
    Insn {
        pc: 0,
        len: 0,
        op,
        form,
        dst,
        dst2: NONE,
        src,
        ext,
    }
}

pub fn sop1(op: Op, dst: Operand, src: Operand) -> Insn {
    insn(op, Form::Plain, dst, [src, NONE, NONE, NONE], Ext::None)
}

pub fn sop2(op: Op, dst: Operand, a: Operand, b: Operand) -> Insn {
    insn(op, Form::Plain, dst, [a, b, NONE, NONE], Ext::None)
}

pub fn sopc(op: Op, a: Operand, b: Operand) -> Insn {
    insn(op, Form::Plain, NONE, [a, b, NONE, NONE], Ext::None)
}

pub fn sopk(op: Op, dst: Operand, imm: u16) -> Insn {
    insn(op, Form::Plain, dst, [NONE; 4], Ext::Imm16(imm))
}

pub fn sopp(op: Op, imm: u16) -> Insn {
    insn(op, Form::Plain, NONE, [NONE; 4], Ext::Imm16(imm))
}

/// A scalar load: `dst` is the first result register, `base` the first base register.
pub fn smem(op: Op, dst: Operand, base: Operand, offset: u32) -> Insn {
    insn(
        op,
        Form::Plain,
        dst,
        [base, NONE, NONE, NONE],
        Ext::Smem {
            offset,
            soffset: Operand::Reg(Reg::Null),
            glc: false,
            dlc: false,
        },
    )
}

pub fn vop1(op: Op, dst: Operand, src: Operand) -> Insn {
    insn(op, Form::Plain, dst, [src, NONE, NONE, NONE], Ext::None)
}

/// A two operand vector instruction. The second source must be a vector register.
pub fn vop2(op: Op, dst: Operand, a: Operand, b: Operand) -> Insn {
    insn(op, Form::Plain, dst, [a, b, NONE, NONE], Ext::None)
}

/// A compare in the one word form: the result goes to VCC, or to EXEC for the `cmpx` forms.
pub fn vopc(op: Op, a: Operand, b: Operand) -> Insn {
    let dst = if op.name().starts_with("v_cmpx") {
        EXEC
    } else {
        VCC
    };
    insn(op, Form::Plain, dst, [a, b, NONE, NONE], Ext::None)
}

/// A three operand vector instruction, or a one word format instruction in the two word encoding.
pub fn vop3(op: Op, dst: Operand, a: Operand, b: Operand, c: Operand) -> Insn {
    let form = if op.info().format == crate::tables::Format::Vop3 {
        Form::Plain
    } else {
        Form::AsVop3
    };
    insn(
        op,
        form,
        dst,
        [a, b, c, NONE],
        Ext::Valu {
            abs: 0,
            neg: 0,
            opsel: 0,
            omod: 0,
            clamp: false,
        },
    )
}

/// The register number of the first source, which the extra word of a sub dword or data
/// parallel form carries in its low byte.
fn low_byte(src0: Operand) -> u32 {
    u32::from(crate::decode::code_of_operand(src0)) & 0xFF
}

/// A one word format instruction with the sub dword word. `src0` is the vector (or scalar)
/// register the word names, `word` the raw SDWA dword without the register number.
pub fn sdwa(op: Op, dst: Operand, src0: Operand, src1: Operand, word: u32) -> Insn {
    insn(
        op,
        Form::Sdwa,
        dst,
        [src0, src1, NONE, NONE],
        Ext::Sdwa((word & !0xFF) | low_byte(src0)),
    )
}

/// The same with a 16 lane data parallel word.
pub fn dpp16(op: Op, dst: Operand, src0: Operand, src1: Operand, word: u32) -> Insn {
    insn(
        op,
        Form::Dpp16,
        dst,
        [src0, src1, NONE, NONE],
        Ext::Dpp((word & !0xFF) | low_byte(src0)),
    )
}

/// The same with an 8 lane data parallel word.
pub fn dpp8(
    op: Op,
    dst: Operand,
    src0: Operand,
    src1: Operand,
    word: u32,
    fetch_inactive: bool,
) -> Insn {
    insn(
        op,
        Form::Dpp8,
        dst,
        [src0, src1, NONE, NONE],
        Ext::Dpp8 {
            word: (word & !0xFF) | low_byte(src0),
            fetch_inactive,
        },
    )
}

/// A local data share instruction. `dst` is `Operand::None` for the ones without a result.
pub fn ds(
    op: Op,
    dst: Operand,
    addr: Operand,
    data0: Operand,
    data1: Operand,
    offset0: u8,
    offset1: u8,
) -> Insn {
    insn(
        op,
        Form::Plain,
        dst,
        [addr, data0, data1, NONE],
        Ext::Ds {
            offset0,
            offset1,
            gds: false,
        },
    )
}

/// A raw buffer store: `data` is the first stored register. Same operands as [`buffer_load`].
pub fn buffer_store(
    op: Op,
    data: Operand,
    addr: Operand,
    desc: Operand,
    offset: u16,
    index: bool,
) -> Insn {
    buffer_load(op, data, addr, desc, offset, index)
}

/// A typed buffer access (`tbuffer_load_*`, `tbuffer_store_*`) with the data format `format`.
#[allow(clippy::too_many_arguments)]
pub fn tbuffer(
    op: Op,
    data: Operand,
    addr: Operand,
    desc: Operand,
    soffset: Operand,
    offset: u16,
    format: u8,
    offen: bool,
    idxen: bool,
) -> Insn {
    insn(
        op,
        Form::Plain,
        data,
        [addr, desc, soffset, NONE],
        Ext::Buf {
            offset,
            offen,
            idxen,
            glc: false,
            slc: false,
            dlc: false,
            lds: false,
            tfe: false,
            format,
        },
    )
}

/// An image instruction with the separate address registers: `first` is the first address
/// register, `more` the following ones (up to twelve, in groups of four per extra word).
#[allow(clippy::too_many_arguments)]
pub fn image_nsa(
    op: Op,
    data: Operand,
    first: Operand,
    more: &[u8],
    image: Operand,
    sampler: Operand,
    dim: u8,
    dmask: u8,
) -> Insn {
    let mut i = self::image(op, data, first, image, sampler, dim, dmask);
    if let Ext::Mimg(m) = &mut i.ext {
        assert!(
            more.len() <= 12,
            "an instruction has at most three extra address words"
        );
        m.nsa[..more.len()].copy_from_slice(more);
        m.nsa_words = more.len().div_ceil(4) as u8;
    }
    i
}

/// An interpolation step of attribute `attr`, component `chan`.
pub fn interp(op: Op, dst: Operand, weight: Operand, attr: u8, chan: u8) -> Insn {
    insn(
        op,
        Form::Plain,
        dst,
        [weight, NONE, NONE, NONE],
        Ext::Vintrp { attr, chan },
    )
}

/// A raw buffer load of dwords into `data` and the registers after it. `desc` is the first of
/// the four descriptor registers, `index` selects the indexed form.
pub fn buffer_load(
    op: Op,
    data: Operand,
    addr: Operand,
    desc: Operand,
    offset: u16,
    index: bool,
) -> Insn {
    let ext = Ext::Buf {
        offset,
        offen: false,
        idxen: index,
        glc: false,
        slc: false,
        dlc: false,
        lds: false,
        tfe: false,
        format: 0,
    };
    insn(
        op,
        Form::Plain,
        data,
        [addr, desc, Operand::Int(0), NONE],
        ext,
    )
}

/// An image instruction with consecutive address registers from `addr`. `image` is the first of
/// eight descriptor registers, `sampler` the first of four.
pub fn image(
    op: Op,
    data: Operand,
    addr: Operand,
    image: Operand,
    sampler: Operand,
    dim: u8,
    dmask: u8,
) -> Insn {
    let m = Mimg {
        dmask,
        dim,
        unrm: false,
        glc: false,
        slc: false,
        dlc: false,
        r128: false,
        a16: false,
        d16: false,
        tfe: false,
        lwe: false,
        nsa: [0; 12],
        nsa_words: 0,
    };
    insn(
        op,
        Form::Plain,
        data,
        [addr, image, sampler, NONE],
        Ext::Mimg(m),
    )
}

/// A global load (`global_load_dword` and its wider forms) of dwords into `dst` and the registers
/// after it, at the 64 bit address the lane holds in `addr` and the register after it, plus a
/// 12 bit `offset`. No scalar base.
pub fn global_load(op: Op, dst: u8, addr: u8, offset: u16) -> Insn {
    let w0 = (0b110111 << 26)
        | (u32::from(op.info().opcode) << 18)
        | (2 << 14)
        | u32::from(offset & 0xFFF);
    let w1 = (u32::from(dst) << 24) | (125 << 16) | u32::from(addr);
    let null = Operand::Reg(Reg::Null);
    insn(
        op,
        Form::Plain,
        v(dst),
        [v(addr), null, v(0), NONE],
        Ext::Flat(w0, w1),
    )
}

/// A flat address load. The title's programs never use that format and the lifter has no rule
/// file for it, so tests use it as "an instruction the translator refuses", whatever the rule
/// files learn.
pub fn refused() -> Insn {
    let op = Op::flat_load_dword;
    let word = (0b110111 << 26) | (u32::from(op.info().opcode) << 18);
    // The registers of an all zero second word, as the decoder reads them back.
    insn(
        op,
        Form::Plain,
        v(0),
        [v(0), s(0), v(0), NONE],
        Ext::Flat(word, 0),
    )
}

/// An export of four vector registers.
pub fn export(
    target: u8,
    enabled: u8,
    src: [u8; 4],
    compressed: bool,
    done: bool,
    valid_mask: bool,
) -> Insn {
    insn(
        Op::exp,
        Form::Plain,
        NONE,
        src.map(v),
        Ext::Exp {
            enabled,
            target,
            compressed,
            done,
            valid_mask,
        },
    )
}

/// A place in the program a branch can go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label(usize);

/// A program under construction.
#[derive(Debug, Default)]
pub struct Asm {
    words: Vec<u32>,
    /// Word positions of the labels, `None` until bound.
    labels: Vec<Option<usize>>,
    /// Branch words to patch: position and label.
    fixups: Vec<(usize, Label)>,
}

impl Asm {
    pub fn new() -> Asm {
        Asm::default()
    }

    /// Appends one instruction.
    pub fn push(&mut self, insn: Insn) -> &mut Asm {
        self.words.extend(encode(&insn));
        self
    }

    /// A new label, not bound yet.
    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }

    /// Binds `label` to the next instruction.
    pub fn bind(&mut self, label: Label) -> &mut Asm {
        self.labels[label.0] = Some(self.words.len());
        self
    }

    /// Appends a branch (`s_branch`, `s_cbranch_*`) to `label`.
    pub fn branch(&mut self, op: Op, label: Label) -> &mut Asm {
        self.fixups.push((self.words.len(), label));
        self.push(sopp(op, 0))
    }

    /// The code words with the signature in front. Panics on a label that was never bound.
    pub fn finish(mut self) -> Vec<u32> {
        for (at, label) in &self.fixups {
            let target = self.labels[label.0].expect("every label a branch uses is bound");
            let distance = target as i64 - (*at as i64 + 1);
            self.words[*at] |= u32::from(distance as i16 as u16);
        }
        let mut out = SIGNATURE.to_vec();
        out.extend(self.words);
        out
    }
}
