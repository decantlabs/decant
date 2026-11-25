//! The formats `xg_xs` lays out: element size and block size (`docs/CONSOLE_LIBS.md`, section 1).

/// What the layout needs of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Laid out: bytes per element, and the texels along each side of an element (1, or 4 for the block compressed formats).
    Element { bpe: u32, block: u32 },
    /// A valid format the layout does not model (planar depth stencil, video, packed, console range): `E_NOTIMPL`.
    NotModelled,
    /// Not a format: `E_INVALIDARG`.
    Unknown,
}

fn in_ranges(value: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&value))
}

const NOT_MODELLED: &[(u32, u32)] = &[
    (0, 0),
    (19, 20),
    (22, 22),
    (44, 45),
    (47, 47),
    (68, 69),
    (103, 110),
    (116, 116),
    (118, 118),
    (120, 188),
    (193, 195),
    (211, 211),
    (217, 217),
    (226, 226),
];

/// The format table of 1.2.
pub fn classify(format: u32) -> Format {
    let element = |bpe, block| Format::Element { bpe, block };
    if in_ranges(format, &[(70, 72), (79, 81)]) {
        return element(8, 4);
    }
    if in_ranges(format, &[(73, 78), (82, 84), (94, 99)]) {
        return element(16, 4);
    }
    if in_ranges(format, &[(60, 66), (111, 113), (190, 190), (192, 192), (199, 199), (219, 219), (223, 223), (225, 225), (229, 231), (235, 237), (255, 255)]) {
        return element(1, 1);
    }
    if in_ranges(format, &[(48, 59), (85, 86), (114, 115), (119, 119), (205, 205), (207, 207), (209, 209), (221, 221), (228, 228)]) {
        return element(2, 1);
    }
    if in_ranges(format, &[(23, 43), (46, 46), (67, 67), (87, 93), (100, 101), (117, 117), (189, 189), (191, 191), (203, 203)]) {
        return element(4, 1);
    }
    if in_ranges(format, &[(9, 18), (21, 21), (102, 102), (197, 197), (201, 201), (213, 213), (215, 215)]) {
        return element(8, 1);
    }
    if in_ranges(format, &[(1, 4)]) {
        return element(16, 1);
    }
    if in_ranges(format, NOT_MODELLED) {
        return Format::NotModelled;
    }
    Format::Unknown
}

/// Depth formats and their views, which do not get the "64 KiB D layout" flag from a console layout of 0x110 to 0x113.
pub fn is_depth(format: u32) -> bool {
    matches!(format, 19 | 20 | 22 | 44 | 45 | 47 | 103..=106 | 110 | 118 | 120)
}

/// Stencil formats (19041 build rule 2 of the swizzle suggestion).
pub fn is_stencil(format: u32) -> bool {
    matches!(format, 19 | 20 | 22 | 44 | 45 | 47 | 62 | 118 | 119 | 120)
}

/// Bits per pixel for the swizzle suggestion (per texel for the BC formats).
pub fn bits_per_pixel(format: u32) -> u32 {
    match format {
        0 => 0,
        1..=4 => 128,
        5..=8 => 96,
        9..=22 => 64,
        23..=47 => 32,
        48..=59 => 16,
        60..=65 => 8,
        66 => 1,
        67 => 32,
        68 | 69 => 16,
        70..=72 => 4,
        73..=78 => 8,
        79..=81 => 4,
        82..=84 => 8,
        85 | 86 => 16,
        87..=93 => 32,
        94..=99 => 8,
        100 | 101 => 32,
        102 => 64,
        103 => 12,
        104 | 105 => 24,
        106 => 12,
        107 => 16,
        108 | 109 => 32,
        110 => 9,
        111 | 112 => 4,
        113 => 8,
        114 | 115 => 16,
        116 | 117 => 32,
        118..=120 => 16,
        189 | 191 => 32,
        190 | 192 => 8,
        other => match classify(other) {
            Format::Element { bpe, .. } => bpe * 8,
            _ => 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_sizes() {
        assert_eq!(classify(28), Format::Element { bpe: 4, block: 1 });
        assert_eq!(classify(61), Format::Element { bpe: 1, block: 1 });
        assert_eq!(classify(56), Format::Element { bpe: 2, block: 1 });
        assert_eq!(classify(10), Format::Element { bpe: 8, block: 1 });
        assert_eq!(classify(2), Format::Element { bpe: 16, block: 1 });
        assert_eq!(classify(71), Format::Element { bpe: 8, block: 4 });
        assert_eq!(classify(98), Format::Element { bpe: 16, block: 4 });
        assert_eq!(classify(40), Format::Element { bpe: 4, block: 1 });
    }

    #[test]
    fn modelled_or_not() {
        for f in [0, 19, 20, 22, 44, 45, 47, 68, 69, 103, 110, 116, 118, 120, 150, 188, 193, 195, 211, 217, 226] {
            assert_eq!(classify(f), Format::NotModelled, "{f}");
        }
        for f in [5, 6, 7, 8, 196, 198, 300, 0xFFFF_FFFF] {
            assert_eq!(classify(f), Format::Unknown, "{f}");
        }
    }

    #[test]
    fn every_format_lands_in_one_class() {
        let elements = (0..=300).filter(|&f| matches!(classify(f), Format::Element { .. })).count();
        assert!(elements > 100);
        assert_eq!(bits_per_pixel(28), 32);
        assert_eq!(bits_per_pixel(71), 4);
        assert_eq!(bits_per_pixel(98), 8);
        assert_eq!(bits_per_pixel(61), 8);
        assert_eq!(bits_per_pixel(0), 0);
        assert_eq!(bits_per_pixel(10), 64);
        assert_eq!(bits_per_pixel(2), 128);
        assert_eq!(bits_per_pixel(45), 32);
        assert_eq!(bits_per_pixel(255), 8);
    }
}
