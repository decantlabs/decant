//! The 32 swizzle mode numbers of the GFX10 `SW_MODE` field and what the console library `xg_xs` accepts for each
//! (`docs/CONSOLE_LIBS.md`, section 1). Only the low 5 bits of a mode field are used.

use crate::tiling::Mode;

/// How a mode lays out a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Volume {
    /// Rejected for 3D textures.
    No,
    /// Row by row like a 2D texture (mode 0).
    Linear,
    /// Each depth slice is laid out like an array slice.
    Thin,
    /// 3D blocks.
    Thick,
}

/// log2 of the block size in bytes: 0 for mode 0, 8, 12, 16, and 19 for the VAR modes (17 with hardware version 4).
pub fn block_bits(mode: u32, v4: bool) -> u32 {
    match mode & 0x1F {
        0 => 0,
        1..=3 => 8,
        4..=7 | 20..=23 => 12,
        8..=19 | 24..=27 => 16,
        _ => {
            if v4 {
                17
            } else {
                19
            }
        }
    }
}

/// Whether `xg_xs` has an address equation for elements of `bpe` bytes in `mode` (mode 0 accepts every size).
pub fn valid_element(mode: u32, bpe: u32) -> bool {
    if !matches!(bpe, 1 | 2 | 4 | 8 | 16) {
        return false;
    }
    match mode & 0x1F {
        0 | 1 | 2 | 5 | 6 | 9 | 10 | 17 | 18 | 21 | 22 | 25 | 26 | 27 | 31 => true,
        24 | 28 => bpe <= 8,
        _ => false,
    }
}

/// The mode reported for the levels in the mip tail, `None` for a mode without equation and for mode 0.
pub fn tail_mode(mode: u32) -> Option<u32> {
    match mode & 0x1F {
        1 | 5 | 9 | 17 | 21 | 25 => Some(1),
        2 | 6 | 10 | 18 | 22 | 26 => Some(2),
        24 => Some(24),
        27 | 31 => Some(3),
        28 => Some(28),
        _ => None,
    }
}

/// How the mode lays out a volume.
pub fn volume(mode: u32) -> Volume {
    match mode & 0x1F {
        0 => Volume::Linear,
        5 | 9 | 17 | 21 | 25 | 26 => Volume::Thick,
        24 | 27 | 28 | 31 => Volume::Thin,
        _ => Volume::No,
    }
}

/// Whether a 1D texture accepts the mode.
pub fn accepts_1d(mode: u32) -> bool {
    matches!(mode & 0x1F, 0 | 24 | 27 | 28 | 31)
}

/// The modes of the Z family (depth), valid for `XGValidateDepthStencilSwizzleMode`.
pub fn is_z_family(mode: u32) -> bool {
    mode.is_multiple_of(4) && (4..=28).contains(&mode)
}

/// The mode of `decant-texture`'s own tables that has the same equation (no exclusive or modes of the default hardware
/// version and the RB+ ones of version 4 only), when there is one.
pub fn core_mode(mode: u32) -> Option<Mode> {
    Some(match mode & 0x1F {
        1 => Mode::Sw256bS,
        2 => Mode::Sw256bD,
        5 => Mode::Sw4kS,
        6 => Mode::Sw4kD,
        9 => Mode::Sw64kS,
        10 => Mode::Sw64kD,
        21 => Mode::Sw4kSX,
        22 => Mode::Sw4kDX,
        24 => Mode::Sw64kZX,
        25 => Mode::Sw64kSX,
        26 => Mode::Sw64kDX,
        27 => Mode::Sw64kRX,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_bits_follow_the_table() {
        assert_eq!(block_bits(0, false), 0);
        assert_eq!((1..=3).map(|m| block_bits(m, false)).collect::<Vec<_>>(), [8, 8, 8]);
        assert_eq!([4, 7, 20, 23].map(|m| block_bits(m, false)), [12; 4]);
        assert_eq!([8, 19, 24, 27].map(|m| block_bits(m, false)), [16; 4]);
        assert_eq!([28, 31].map(|m| block_bits(m, false)), [19, 19]);
        assert_eq!([28, 31].map(|m| block_bits(m, true)), [17, 17]);
        assert_eq!(block_bits(27 + 32, false), 16, "only the low 5 bits count");
    }

    #[test]
    fn validity_matches_the_table() {
        for bpe in [1, 2, 4, 8, 16] {
            assert!(valid_element(0, bpe));
            assert_eq!(valid_element(24, bpe), bpe <= 8);
            assert_eq!(valid_element(28, bpe), bpe <= 8);
            for m in [1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 25, 26, 27, 31] {
                assert!(valid_element(m, bpe), "{m} {bpe}");
            }
            for m in [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 29, 30] {
                assert!(!valid_element(m, bpe), "{m} {bpe}");
            }
        }
        assert!(!valid_element(5, 3));
    }

    #[test]
    fn tail_and_volume_classes() {
        assert_eq!([1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27, 28, 31].map(|m| tail_mode(m).unwrap()), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 24, 1, 2, 3, 28, 3]);
        assert_eq!(tail_mode(0), None);
        assert_eq!(tail_mode(3), None);
        assert_eq!(volume(5), Volume::Thick);
        assert_eq!(volume(27), Volume::Thin);
        assert_eq!(volume(10), Volume::No);
        assert_eq!(volume(0), Volume::Linear);
        assert!(accepts_1d(28) && !accepts_1d(5));
        assert!(is_z_family(4) && is_z_family(28) && !is_z_family(0) && !is_z_family(29) && !is_z_family(32));
    }
}
