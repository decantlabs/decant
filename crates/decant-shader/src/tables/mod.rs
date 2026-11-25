//! The GFX10 instruction table and the lookup from an encoding to an instruction.
//!
//! `gfx10.rs` is generated from Mesa's `aco_opcodes.py` (MIT). This file holds the types the
//! table uses and the lookup. GFX10 here means GFX10.3 (RDNA2): where GFX10.1 and GFX10.3 give
//! one number to two instructions, the GFX10.3 one wins.

mod gfx10;

use std::collections::HashMap;
use std::sync::LazyLock;

pub use gfx10::{OP_COUNT, OPS, Op};

/// The encoding format of an instruction, as the table names it. `Vop1`, `Vop2` and `Vopc`
/// instructions also exist in the `Vop3` encoding, the decoder reports that in the decoded form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Format {
    Sop1,
    Sop2,
    Sopk,
    Sopp,
    Sopc,
    Smem,
    Ds,
    Mtbuf,
    Mubuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
    Vintrp,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vop3p,
}

/// The timing class of an instruction. The census groups by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    Valu32,
    ValuConvert32,
    Valu64,
    ValuQuarterRate32,
    ValuFma,
    ValuTranscendental32,
    ValuDouble,
    ValuDoubleAdd,
    ValuDoubleConvert,
    ValuDoubleTranscendental,
    ValuPseudoScalarTrans,
    Wmma,
    Salu,
    Sfpu,
    Smem,
    Barrier,
    Branch,
    Sendmsg,
    Ds,
    Exp,
    Vmem,
    Waitcnt,
    Other,
}

/// What kind of value an operand slot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    None,
    /// One bit per lane: VCC, EXEC or a scalar register pair used as a mask.
    LaneMask,
    Uint,
    Int,
    Float,
    BFloat,
}

/// A slot that is always one register and is not encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fixed {
    No,
    Exec,
    Scc,
    M0,
    Imm,
}

/// One result or source of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    pub base: Base,
    /// Bits per component.
    pub bits: u8,
    /// Components packed into the register (two 16 bit halves for the packed operations).
    pub comps: u8,
    pub fixed: Fixed,
    /// Whether the source takes the NEG and ABS modifiers (and the result the output modifier).
    pub mods: bool,
}

/// One row of the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpInfo {
    pub op: Op,
    pub name: &'static str,
    pub format: Format,
    /// The opcode number inside its format.
    pub opcode: u16,
    pub class: Class,
    /// A memory operation that reads and writes in one step.
    pub atomic: bool,
    /// Results. The memory formats list none here, the decoder knows their operands.
    pub defs: &'static [Slot],
    pub srcs: &'static [Slot],
}

impl Op {
    /// The table row of this instruction.
    pub fn info(self) -> &'static OpInfo {
        &OPS[self as usize]
    }

    pub fn name(self) -> &'static str {
        self.info().name
    }
}

/// Instructions that lose when two share one number, see the module text.
const SHADOWED: [Op; 2] = [Op::v_mad_legacy_f32, Op::v_mac_legacy_f32];

static BY_NUMBER: LazyLock<HashMap<(Format, u16), Op>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for info in OPS.iter().filter(|i| !SHADOWED.contains(&i.op)) {
        // `v_interp_p2_hi_f16` is `v_interp_p2_f16` with an operand selector set: the first one stays.
        map.entry((info.format, info.opcode)).or_insert(info.op);
    }
    map
});

static BY_NAME: LazyLock<HashMap<&'static str, Op>> =
    LazyLock::new(|| OPS.iter().map(|i| (i.name, i.op)).collect());

/// The instruction with this number in this format.
pub fn lookup(format: Format, opcode: u16) -> Option<Op> {
    BY_NUMBER.get(&(format, opcode)).copied()
}

/// The instruction with this name.
pub fn by_name(name: &str) -> Option<Op> {
    BY_NAME.get(name).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_in_enum_order_and_numbers_are_unique() {
        for (i, info) in OPS.iter().enumerate() {
            assert_eq!(info.op as usize, i, "{}", info.name);
        }
        let mut seen = HashMap::new();
        for info in OPS
            .iter()
            .filter(|i| !SHADOWED.contains(&i.op) && i.op != Op::v_interp_p2_hi_f16)
        {
            if let Some(other) = seen.insert((info.format, info.opcode), info.name) {
                panic!(
                    "{} and {} share {:?} {:#x}",
                    other, info.name, info.format, info.opcode
                );
            }
        }
    }

    #[test]
    fn known_numbers_of_the_public_isa_documents() {
        assert_eq!(lookup(Format::Sopp, 1), Some(Op::s_endpgm));
        assert_eq!(lookup(Format::Sop1, 3), Some(Op::s_mov_b32));
        assert_eq!(lookup(Format::Vop2, 3), Some(Op::v_add_f32));
        assert_eq!(lookup(Format::Vop1, 1), Some(Op::v_mov_b32));
        assert_eq!(lookup(Format::Smem, 0), Some(Op::s_load_dword));
        assert_eq!(by_name("v_fma_f32"), Some(Op::v_fma_f32));
        assert_eq!(Op::exp.info().format, Format::Exp);
    }
}
