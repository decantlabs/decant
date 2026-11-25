//! Accessors for the extra dword of the sub dword (SDWA) and data parallel (DPP) forms and for
//! the address list of image instructions. Bit positions: `docs/gfx10/encoding.md`
//! sections 5 and 6.

use super::{Ext, Form, Insn, Mimg};

fn bit(x: u32, at: u32) -> bool {
    (x >> at) & 1 != 0
}

/// Which part of a dword an SDWA operand selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaSel {
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Word0,
    Word1,
    Dword,
}

impl SdwaSel {
    fn of(code: u32) -> Option<SdwaSel> {
        Some(match code {
            0 => SdwaSel::Byte0,
            1 => SdwaSel::Byte1,
            2 => SdwaSel::Byte2,
            3 => SdwaSel::Byte3,
            4 => SdwaSel::Word0,
            5 => SdwaSel::Word1,
            6 => SdwaSel::Dword,
            _ => return None,
        })
    }

    /// The first bit and the width of the selected part.
    pub fn bits(self) -> (u32, u32) {
        match self {
            SdwaSel::Byte0 => (0, 8),
            SdwaSel::Byte1 => (8, 8),
            SdwaSel::Byte2 => (16, 8),
            SdwaSel::Byte3 => (24, 8),
            SdwaSel::Word0 => (0, 16),
            SdwaSel::Word1 => (16, 16),
            SdwaSel::Dword => (0, 32),
        }
    }
}

/// What happens to the bits of the destination outside the selected part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SdwaUnused {
    Pad,
    Sext,
    Preserve,
}

/// The extra dword of an SDWA instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sdwa(pub u32);

impl Sdwa {
    /// The destination part. `None` for an undefined code (and for compares, which have none).
    pub fn dst_sel(self) -> Option<SdwaSel> {
        SdwaSel::of((self.0 >> 8) & 7)
    }
    pub fn dst_unused(self) -> Option<SdwaUnused> {
        match (self.0 >> 11) & 3 {
            0 => Some(SdwaUnused::Pad),
            1 => Some(SdwaUnused::Sext),
            2 => Some(SdwaUnused::Preserve),
            _ => None,
        }
    }
    pub fn clamp(self) -> bool {
        bit(self.0, 13)
    }
    /// The output modifier: 0 none, 1 times 2, 2 times 4, 3 divide by 2.
    pub fn omod(self) -> u8 {
        ((self.0 >> 14) & 3) as u8
    }
    pub fn src0_sel(self) -> Option<SdwaSel> {
        SdwaSel::of((self.0 >> 16) & 7)
    }
    pub fn src0_sext(self) -> bool {
        bit(self.0, 19)
    }
    pub fn src0_neg(self) -> bool {
        bit(self.0, 20)
    }
    pub fn src0_abs(self) -> bool {
        bit(self.0, 21)
    }
    pub fn src0_is_scalar(self) -> bool {
        bit(self.0, 23)
    }
    pub fn src1_sel(self) -> Option<SdwaSel> {
        SdwaSel::of((self.0 >> 24) & 7)
    }
    pub fn src1_sext(self) -> bool {
        bit(self.0, 27)
    }
    pub fn src1_neg(self) -> bool {
        bit(self.0, 28)
    }
    pub fn src1_abs(self) -> bool {
        bit(self.0, 29)
    }
    pub fn src1_is_scalar(self) -> bool {
        bit(self.0, 31)
    }
    /// Compares: the compare result goes to the scalar register `sdst()` when this is set, to VCC
    /// otherwise.
    pub fn sdst_enabled(self) -> bool {
        bit(self.0, 15)
    }
    pub fn sdst(self) -> u8 {
        ((self.0 >> 8) & 0x7F) as u8
    }
}

/// The meaning of the control field of a 16 lane DPP word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DppCtrl {
    /// Lane `i` of each group of four reads lane `perm[i]`.
    QuadPerm([u8; 4]),
    RowShl(u8),
    RowShr(u8),
    RowRor(u8),
    RowMirror,
    RowHalfMirror,
    RowShare(u8),
    RowXmask(u8),
    /// Codes GFX10 does not have (wave shifts, broadcasts).
    Other(u16),
}

/// The extra dword of a 16 lane DPP instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dpp16(pub u32);

impl Dpp16 {
    pub fn ctrl(self) -> DppCtrl {
        let c = ((self.0 >> 8) & 0x1FF) as u16;
        let n = (c & 0xF) as u8;
        match c {
            0..=0xFF => DppCtrl::QuadPerm([
                (c & 3) as u8,
                ((c >> 2) & 3) as u8,
                ((c >> 4) & 3) as u8,
                ((c >> 6) & 3) as u8,
            ]),
            0x101..=0x10F => DppCtrl::RowShl(n),
            0x111..=0x11F => DppCtrl::RowShr(n),
            0x121..=0x12F => DppCtrl::RowRor(n),
            0x140 => DppCtrl::RowMirror,
            0x141 => DppCtrl::RowHalfMirror,
            0x150..=0x15F => DppCtrl::RowShare(n),
            0x160..=0x16F => DppCtrl::RowXmask(n),
            other => DppCtrl::Other(other),
        }
    }
    /// Fetch inactive: read lanes that are off instead of treating them as out of bounds.
    pub fn fetch_inactive(self) -> bool {
        bit(self.0, 18)
    }
    pub fn bound_ctrl(self) -> bool {
        bit(self.0, 19)
    }
    pub fn src0_neg(self) -> bool {
        bit(self.0, 20)
    }
    pub fn src0_abs(self) -> bool {
        bit(self.0, 21)
    }
    pub fn src1_neg(self) -> bool {
        bit(self.0, 22)
    }
    pub fn src1_abs(self) -> bool {
        bit(self.0, 23)
    }
    pub fn bank_mask(self) -> u8 {
        ((self.0 >> 24) & 0xF) as u8
    }
    pub fn row_mask(self) -> u8 {
        ((self.0 >> 28) & 0xF) as u8
    }
}

/// The extra dword of an 8 lane DPP instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dpp8(pub u32);

impl Dpp8 {
    /// The lane of the group of eight that lane `lane` reads.
    pub fn lane_sel(self, lane: u8) -> u8 {
        ((self.0 >> (8 + 3 * u32::from(lane & 7))) & 7) as u8
    }
}

impl Insn {
    pub fn sdwa(&self) -> Option<Sdwa> {
        match (self.form, self.ext) {
            (Form::Sdwa, Ext::Sdwa(x)) => Some(Sdwa(x)),
            _ => None,
        }
    }
    pub fn dpp16(&self) -> Option<Dpp16> {
        match (self.form, self.ext) {
            (Form::Dpp16, Ext::Dpp(x)) => Some(Dpp16(x)),
            _ => None,
        }
    }
    /// The 8 lane word and whether the instruction fetches inactive lanes (source code 234).
    pub fn dpp8(&self) -> Option<(Dpp8, bool)> {
        match (self.form, self.ext) {
            (
                Form::Dpp8,
                Ext::Dpp8 {
                    word,
                    fetch_inactive,
                },
            ) => Some((Dpp8(word), fetch_inactive)),
            _ => None,
        }
    }
}

impl Mimg {
    /// The vector register of address component `i`. `first` is the register of the first
    /// address (`src[0]` of the instruction). Without the separate address form the components
    /// follow in order.
    pub fn address(&self, first: u8, i: usize) -> Option<u8> {
        if i == 0 {
            return Some(first);
        }
        if self.nsa_words == 0 {
            return u8::try_from(usize::from(first) + i).ok();
        }
        if i <= usize::from(self.nsa_words) * 4 {
            Some(self.nsa[i - 1])
        } else {
            None
        }
    }
}
