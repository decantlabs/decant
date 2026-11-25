//! Image instructions (`image_sample*`, `image_gather4*`, `image_load*`, `image_store*`,
//! `image_get_resinfo`, `image_get_lod`).
//!
//! The address operands sit in consecutive vector registers from the address register, or, with
//! extra address words (NSA), in the registers those words list. Their order is one fixed rule
//! for every instruction:
//!
//! ```text
//! [offset] [bias] [zcompare] [gradients] coordinates [slice] [fragid] [lod | clamp | mip]
//! ```
//!
//! The modifiers come from the instruction name. With A16 the coordinates and the last operand
//! are 16 bit halves, two per register, while offset, bias and compare value keep a register
//! each. The lifter unpacks the halves, so [`ImageOp::address`] always holds one 32 bit pattern
//! per operand: floats for samples, unsigned integers for loads and stores.

use super::Lifter;
use super::operand::form;
use crate::decode::{Ext, Insn, Mimg, Operand, Reg};
use crate::ir::{
    Bin, Desc, Expr, Guard, Helper, ImageAtomic, ImageKind, ImageOp, SampleLod, Stmt, Ty, Var,
};
use crate::refusal::Refusal;

/// How many address components the coordinates of dimension code `dim` take, slice, face and
/// fragment index included.
pub fn coordinates(dim: u8) -> Option<usize> {
    Some(match dim {
        0 => 1,
        1 | 4 => 2,
        2 | 3 | 5 | 6 => 3,
        7 => 4,
        _ => return None,
    })
}

/// How many derivative components one direction has for dimension code `dim`.
pub fn gradients(dim: u8) -> Option<usize> {
    Some(match dim {
        0 | 4 => 1,
        1 | 3 | 5 => 2,
        2 => 3,
        _ => return None,
    })
}

/// A 16 bit float half of a register, as a 32 bit float pattern.
static F16_LO: Helper = Helper {
    name: "decant_img_f16lo",
    result: Ty::U32,
    text: "uint decant_img_f16lo(uint v) { return asuint(f16tof32(v)); }",
    needs: &[],
};

static F16_HI: Helper = Helper {
    name: "decant_img_f16hi",
    result: Ty::U32,
    text: "uint decant_img_f16hi(uint v) { return asuint(f16tof32(v >> 16)); }",
    needs: &[],
};

/// What the instruction name says.
#[derive(Clone, Copy, Debug, Default)]
struct Mods {
    compare: bool,
    grad: bool,
    level: bool,
    zero: bool,
    bias: bool,
    clamp: bool,
    offset: bool,
    g16: bool,
}

fn mods(suffix: &str) -> Option<Mods> {
    let mut m = Mods::default();
    for token in suffix.split('_').filter(|t| !t.is_empty()) {
        match token {
            "c" => m.compare = true,
            "d" | "cd" => m.grad = true,
            "l" => m.level = true,
            "lz" => m.zero = true,
            "b" => m.bias = true,
            "cl" => m.clamp = true,
            "o" => m.offset = true,
            "g16" => m.g16 = true,
            _ => return None,
        }
    }
    Some(m)
}

/// The instruction class and its modifiers.
enum Class {
    Sample {
        gather: bool,
        mods: Mods,
    },
    QueryLod,
    Load {
        mip: bool,
    },
    Store {
        mip: bool,
    },
    Info,
    /// `None` for the float atomics, which have no HLSL form.
    Atomic(Option<ImageAtomic>),
}

fn atomic(name: &str) -> Option<Option<ImageAtomic>> {
    Some(Some(match name {
        "swap" => ImageAtomic::Swap,
        "cmpswap" => ImageAtomic::CmpSwap,
        "add" => ImageAtomic::Add,
        "sub" => ImageAtomic::Sub,
        "smin" => ImageAtomic::SMin,
        "umin" => ImageAtomic::UMin,
        "smax" => ImageAtomic::SMax,
        "umax" => ImageAtomic::UMax,
        "and" => ImageAtomic::And,
        "or" => ImageAtomic::Or,
        "xor" => ImageAtomic::Xor,
        "inc" => ImageAtomic::Inc,
        "dec" => ImageAtomic::Dec,
        "fcmpswap" | "fmin" | "fmax" => return Some(None),
        _ => return None,
    }))
}

fn class(name: &str) -> Option<Class> {
    let rest = name.strip_prefix("image_")?;
    if let Some(op) = rest.strip_prefix("atomic_") {
        return atomic(op).map(Class::Atomic);
    }
    Some(match rest {
        "load" => Class::Load { mip: false },
        "load_mip" => Class::Load { mip: true },
        "store" => Class::Store { mip: false },
        "store_mip" => Class::Store { mip: true },
        "get_resinfo" => Class::Info,
        "get_lod" => Class::QueryLod,
        _ => {
            if let Some(s) = rest.strip_prefix("sample") {
                Class::Sample {
                    gather: false,
                    mods: mods(s)?,
                }
            } else {
                Class::Sample {
                    gather: true,
                    mods: mods(rest.strip_prefix("gather4")?)?,
                }
            }
        }
    })
}

fn flags(insn: &Insn, m: &Mimg) -> Result<(), Refusal> {
    for (set, what) in [(m.lwe, "image-lwe"), (m.r128, "image-r128")] {
        if set {
            return Err(form(insn, what));
        }
    }
    Ok(())
}

/// The address registers in order: from the first one on, or from the NSA bytes.
fn registers(m: &Mimg, first: u8, count: usize) -> Result<Vec<u8>, ()> {
    let mut regs = vec![first];
    if m.nsa_words == 0 {
        regs.extend((1..count).map(|k| first.wrapping_add(k as u8)));
    } else {
        if count - 1 > usize::from(m.nsa_words) * 4 {
            return Err(());
        }
        regs.extend(m.nsa.iter().take(count - 1));
    }
    Ok(regs)
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some(class) = class(insn.op.name()) else {
        return Ok(false);
    };
    let Ext::Mimg(m) = insn.ext else {
        return Ok(false);
    };
    let (
        Operand::Reg(Reg::V(data)),
        Operand::Reg(Reg::V(first)),
        Operand::Reg(Reg::S(image)),
        Operand::Reg(Reg::S(sampler)),
    ) = (insn.dst, insn.src[0], insn.src[1], insn.src[2])
    else {
        return Err(form(insn, "operand"));
    };
    flags(insn, &m)?;
    let coords = coordinates(m.dim).ok_or_else(|| form(insn, "image-dim"))?;
    // Operands before the coordinates (one register each, also with A16), the gradients, the
    // coordinates and the one operand after them.
    let (lead, grads, tail, kind, uses_sampler, float) = match class {
        Class::Sample { gather, mods: md } => {
            if md.g16 {
                return Err(form(insn, "image-g16"));
            }
            let lod = match (md.grad, md.level, md.zero, md.bias) {
                (false, false, false, false) => SampleLod::Implicit,
                (true, false, false, false) => SampleLod::Grad,
                (false, true, false, false) => SampleLod::Level,
                (false, false, true, false) => SampleLod::Zero,
                (false, false, false, true) => SampleLod::Bias,
                _ => return Err(form(insn, "image-modifiers")),
            };
            if md.clamp && (md.level || md.zero) {
                return Err(form(insn, "image-modifiers"));
            }
            let lead = usize::from(md.offset) + usize::from(md.bias) + usize::from(md.compare);
            let grads = if md.grad {
                2 * gradients(m.dim).ok_or_else(|| form(insn, "image-dim"))?
            } else {
                0
            };
            let tail = usize::from(md.level || md.clamp);
            let kind = ImageKind::Sample {
                lod,
                compare: md.compare,
                offset: md.offset,
                gather,
                clamp: md.clamp,
            };
            (lead, grads, tail, kind, true, true)
        }
        Class::QueryLod => (0, 0, 0, ImageKind::QueryLod, true, true),
        Class::Load { mip } => (
            0,
            0,
            usize::from(mip),
            ImageKind::Fetch { mip },
            false,
            false,
        ),
        Class::Store { mip } => (
            0,
            0,
            usize::from(mip),
            ImageKind::Store { mip },
            false,
            false,
        ),
        Class::Info => (0, 0, 1, ImageKind::Info, false, false),
        Class::Atomic(None) => return Err(form(insn, "image-float-atomic")),
        Class::Atomic(Some(op)) => (
            0,
            0,
            0,
            ImageKind::Atomic { op, returns: m.glc },
            false,
            false,
        ),
    };
    // Resinfo takes only the mip level.
    let packed = if matches!(kind, ImageKind::Info) {
        1
    } else {
        coords + tail
    };
    let packed_regs = if m.a16 { packed.div_ceil(2) } else { packed };
    let count = lead + grads + packed_regs;
    let regs = registers(&m, first, count).map_err(|()| form(insn, "image-address"))?;
    let mut address: Vec<Expr> = regs[..lead + grads]
        .iter()
        .map(|r| Expr::Var(Var::V(*r)))
        .collect();
    for k in 0..packed {
        if !m.a16 {
            address.push(Expr::Var(Var::V(regs[lead + grads + k])));
            continue;
        }
        let reg = Expr::Var(Var::V(regs[lead + grads + k / 2]));
        let high = k % 2 == 1;
        address.push(match (float, high) {
            (true, false) => Expr::Call(&F16_LO, vec![reg]),
            (true, true) => Expr::Call(&F16_HI, vec![reg]),
            (false, false) => Expr::bin(Bin::And, reg, Expr::U32(0xFFFF)),
            (false, true) => Expr::bin(Bin::LShr, reg, Expr::U32(16)),
        });
    }
    // An atomic names its operand dwords in the mask: one per operand (two for a compare and
    // swap), twice that for the 64 bit forms.
    if let ImageKind::Atomic { op, .. } = kind {
        let ok: &[u8] = if op == ImageAtomic::CmpSwap {
            &[0x3, 0xF]
        } else {
            &[0x1, 0x3]
        };
        if !ok.contains(&m.dmask) {
            return Err(form(insn, "image-atomic-mask"));
        }
    }
    cx.stmts.push(Stmt::Image(
        ImageOp {
            kind,
            dim: m.dim,
            dmask: m.dmask,
            d16: m.d16,
            tfe: m.tfe,
            // Loads, stores and atomics take integer coordinates whatever the bit says.
            unnormalized: m.unrm && uses_sampler,
            data,
            address,
            image: Desc::Regs(image),
            sampler: uses_sampler.then_some(Desc::Regs(sampler)),
            pc: insn.pc,
        },
        Guard::Exec,
    ));
    Ok(true)
}
