//! Every test vector of `docs/CONSOLE_LIBS.md` 1.13, called through the exports and the texture computer table as a C client does.
#![allow(unsafe_code, clippy::type_complexity)]

mod common;

use common::*;
use xg_xs::abi::{Hresult, XgTileShape};
use xg_xs::exports::*;

#[test]
fn optimal_swizzle_mode_vectors() {
    let _s = serial();
    // (dimension, format, width, height, depth or array, samples, bind, misc) -> mode, rows 1 to 41 of the specification.
    let rows: [((u32, u32, u64, u32, u32, u32, u32, u32), u32); 41] = [
        ((1, 0, 65536, 1, 1, 1, 0x0, 0x0), 0),
        ((2, 28, 256, 1, 1, 1, 0x8, 0x0), 0),
        ((2, 28, 4096, 1, 1, 1, 0x80, 0x0), 0),
        ((3, 28, 256, 256, 1, 1, 0x8, 0x0), 5),
        ((3, 71, 1024, 1024, 1, 1, 0x8, 0x0), 5),
        ((3, 98, 4096, 4096, 1, 1, 0x8, 0x0), 5),
        ((3, 28, 1920, 1080, 1, 1, 0x28, 0x0), 27),
        ((3, 10, 1920, 1080, 1, 1, 0xA8, 0x0), 27),
        ((3, 28, 1920, 1080, 1, 4, 0x20, 0x0), 27),
        ((3, 40, 1920, 1080, 1, 1, 0x48, 0x0), 24),
        ((3, 45, 1920, 1080, 1, 1, 0x40, 0x0), 24),
        ((3, 28, 128, 128, 1, 1, 0x88, 0x0), 21),
        ((3, 28, 256, 256, 1, 1, 0x88, 0x0), 27),
        ((3, 28, 362, 362, 1, 1, 0x88, 0x0), 27),
        ((3, 28, 363, 363, 1, 1, 0x88, 0x0), 27),
        ((3, 61, 1024, 512, 1, 1, 0x88, 0x0), 27),
        ((3, 61, 1024, 513, 1, 1, 0x88, 0x0), 27),
        ((3, 2, 128, 64, 1, 1, 0x88, 0x0), 27),
        ((3, 2, 128, 65, 1, 1, 0x88, 0x0), 27),
        ((3, 28, 128, 128, 1, 2, 0x88, 0x0), 27),
        ((3, 28, 256, 256, 1, 1, 0x8, 0x800_0000), 10),
        ((3, 28, 256, 256, 1, 1, 0x28, 0x800_0000), 10),
        ((4, 28, 64, 64, 64, 1, 0x8, 0x0), 5),
        ((4, 28, 32, 32, 32, 1, 0x88, 0x0), 27),
        ((4, 28, 64, 64, 32, 1, 0x88, 0x0), 27),
        ((4, 28, 64, 64, 33, 1, 0x88, 0x0), 27),
        ((4, 28, 64, 64, 64, 1, 0xA8, 0x0), 27),
        ((4, 28, 64, 64, 64, 1, 0x28, 0x0), 27),
        ((4, 28, 64, 64, 64, 1, 0x8, 0x800_0000), 10),
        ((5, 28, 64, 64, 1, 1, 0x8, 0x0), 0),
        ((3, 0, 256, 256, 1, 1, 0x88, 0x0), 27),
        ((3, 61, 256, 256, 1, 1, 0x88, 0x0), 21),
        ((3, 61, 256, 257, 1, 1, 0x88, 0x0), 27),
        ((3, 28, 128, 129, 1, 1, 0x88, 0x0), 27),
        ((3, 98, 64, 64, 1, 1, 0x88, 0x0), 21),
        ((4, 28, 16, 16, 16, 1, 0x88, 0x0), 21),
        ((4, 28, 16, 16, 17, 1, 0x88, 0x0), 21),
        ((4, 28, 16, 16, 16, 1, 0xA8, 0x0), 27),
        ((3, 45, 256, 256, 1, 1, 0x40, 0x800_8000), 24),
        ((3, 28, 256, 256, 1, 1, 0x40, 0x800_0000), 24),
        ((2, 28, 4096, 1, 1, 1, 0x40, 0x0), 24),
    ];
    for (i, ((dim, format, width, height, depth, samples, bind, misc), mode)) in rows.into_iter().enumerate() {
        assert_eq!(XGComputeOptimalSwizzleMode(dim, format, width, height, depth, samples, bind, misc), mode, "vector {}", i + 1);
    }
}

/// A level of the specification: offset, padded width, padded height, mode, in the tail.
struct Lv(u64, u32, u32, u32, bool);

/// What a case gives. `None` fields are not stated by the specification.
#[derive(Default)]
struct Expect {
    size: Option<u64>,
    align: Option<u64>,
    mips: Option<u32>,
    first_tail: Option<u32>,
    tail_count: Option<u32>,
    slice: Option<u64>,
    planes: Option<u32>,
    levels: Vec<Lv>,
    /// (mip, x, y, z, offset).
    elements: Vec<(u32, u64, u32, u32, u64)>,
}

fn check(name: &str, d: D3d, e: Expect) -> Computer {
    let c = Computer::create(&d).unwrap_or_else(|hr| panic!("{name}: {hr:#x}"));
    let r = Record(c.record());
    if let Some(v) = e.size {
        assert_eq!(c.size(), v, "{name} size");
        assert_eq!(r.size(), v, "{name} record size");
    }
    if let Some(v) = e.align {
        assert_eq!(c.align(), v, "{name} alignment");
        assert_eq!(r.align(), v, "{name} record alignment");
    }
    if let Some(v) = e.mips {
        assert_eq!(r.mips(), v, "{name} mips");
    }
    if let Some(v) = e.first_tail {
        assert_eq!(r.first_tail(), v, "{name} first tail");
    }
    if let Some(v) = e.tail_count {
        assert_eq!(r.tail_count(), v, "{name} tail count");
    }
    if let Some(v) = e.slice {
        assert_eq!(r.plane_slice(0), v, "{name} slice");
    }
    if let Some(v) = e.planes {
        assert_eq!(r.planes(), v, "{name} planes");
    }
    for (i, Lv(off, pw, ph, mode, tail)) in e.levels.iter().enumerate() {
        assert_eq!(r.offset(i), *off, "{name} level {i} offset");
        assert_eq!(c.mip_offset(0, i as u32), *off, "{name} level {i} MipOffset");
        if *pw != 0 {
            let (w, h, _) = r.padded(i);
            assert_eq!((w, h), (*pw, *ph), "{name} level {i} padded size");
        }
        assert_eq!(r.mode(i), *mode, "{name} level {i} mode");
        assert_eq!(r.in_tail(i), *tail, "{name} level {i} tail flag");
    }
    for (mip, x, y, z, off) in e.elements {
        assert_eq!(c.element(mip, x, y, z), off, "{name} element ({mip},{x},{y},{z})");
    }
    c
}

fn tex(w: u64, h: u32, a: u16, mips: u16, format: u32, layout: u32, flags: u32) -> D3d {
    D3d { dimension: 3, width: w, height: h, depth: a, mips, format, samples: 1, layout, flags }
}

fn volume(w: u64, h: u32, d: u16, mips: u16, layout: u32) -> D3d {
    D3d { dimension: 4, width: w, height: h, depth: d, mips, format: 28, samples: 1, layout, flags: 0 }
}

#[test]
fn computer_case_a() {
    let _s = serial();
    let c = check(
        "A",
        tex(256, 256, 1, 1, 28, 0, 0),
        Expect {
            size: Some(0x40000),
            align: Some(0x1000),
            mips: Some(1),
            slice: Some(0x40000),
            levels: vec![Lv(0, 256, 256, 5, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10), (0, 5, 3, 0, 0xB4), (0, 37, 21, 0, 0x14D4)],
            ..Expect::default()
        },
    );
    let r = Record(c.record());
    assert_eq!(r.level_u32(0, 0, 0x18), 256, "pitch in texels");
    assert_eq!(r.level_u32(0, 0, 0x1C), 1024, "pitch in bytes");
    assert_eq!(r.dimension(), 3);
    assert_eq!(r.bpe(), 4);
}

#[test]
fn computer_case_b() {
    let _s = serial();
    check(
        "B",
        tex(256, 256, 1, 0, 28, 0, 0),
        Expect {
            size: Some(0x56000),
            align: Some(0x1000),
            mips: Some(9),
            first_tail: Some(4),
            tail_count: Some(5),
            levels: vec![
                Lv(0x16000, 256, 256, 5, false),
                Lv(0x6000, 128, 128, 5, false),
                Lv(0x2000, 64, 64, 5, false),
                Lv(0x1000, 32, 32, 5, false),
                Lv(0x800, 16, 32, 1, true),
                Lv(0x600, 8, 16, 1, true),
                Lv(0x500, 8, 8, 1, true),
                Lv(0x400, 8, 8, 1, true),
                Lv(0x300, 8, 8, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x16000), (0, 9, 9, 0, 0x16314), (4, 0, 0, 0, 0x800), (4, 3, 2, 0, 0x82C), (8, 0, 0, 0, 0x300)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_case_c_and_c4() {
    let _s = serial();
    let d = tex(1920, 1080, 1, 1, 28, 0, 1);
    let c = check(
        "C",
        d,
        Expect {
            size: Some(0x878000),
            align: Some(0x10000),
            planes: Some(2),
            levels: vec![Lv(0, 1920, 1152, 27, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10), (0, 100, 50, 0, 0x1B60), (0, 1919, 1079, 0, 0x8629FC)],
            ..Expect::default()
        },
    );
    let r = Record(c.record());
    assert_eq!(r.plane_size(0), 0x870000);
    assert_eq!((r.level_u32(0, 0, 0x50), r.level_u32(0, 0, 0x58)), (0x8000, 0x8000));
    assert_eq!((r.plane_usage(1), r.plane_size(1), r.plane_offset(1), r.plane_align(1)), (2, 0x8000, 0x870000, 0x2000));
    assert_eq!(r.level_u64(1, 0, 0), 0x8000, "CMASK size at mip 0");
    assert_eq!(r.level_u32(1, 0, 0x20), 0x2000, "CMASK alignment at mip 0");
    assert_eq!(c.texel(1, 0, 0, 0, 0, 0), u64::MAX, "plane 1 has no element offsets");
    assert_eq!(c.mip_offset(1, 0), 0);
    XGSetHardwareVersion(4);
    let c4 = check("C4", d, Expect { size: Some(0x878000), planes: Some(2), elements: vec![(0, 100, 50, 0, 0x6260)], ..Expect::default() });
    let r4 = Record(c4.record());
    assert_eq!((r4.level_u32(0, 0, 0x50), r4.level_u32(0, 0, 0x58)), (0xE000, 0x2000));
}

#[test]
fn computer_case_d() {
    let _s = serial();
    let c = check(
        "D",
        tex(1024, 1024, 1, 11, 71, 0, 0),
        Expect {
            size: Some(0xAB000),
            align: Some(0x1000),
            first_tail: Some(4),
            tail_count: Some(7),
            levels: vec![
                Lv(0x2B000, 256, 256, 5, false),
                Lv(0xB000, 128, 128, 5, false),
                Lv(0x3000, 64, 64, 5, false),
                Lv(0x1000, 32, 32, 5, false),
                Lv(0x800, 16, 16, 1, true),
                Lv(0x600, 8, 8, 1, true),
                Lv(0x500, 8, 4, 1, true),
                Lv(0x400, 8, 4, 1, true),
                Lv(0x300, 8, 4, 1, true),
                Lv(0x200, 8, 4, 1, true),
                Lv(0x100, 8, 4, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x2B000), (0, 3, 1, 0, 0x2B058), (6, 0, 0, 0, 0x500)],
            ..Expect::default()
        },
    );
    let r = Record(c.record());
    assert_eq!(r.bpe(), 8);
    assert_eq!((r.level_u32(0, 0, 0x18), r.level_u32(0, 0, 0x1C)), (1024, 2048));
}

#[test]
fn computer_case_e() {
    let _s = serial();
    check(
        "E",
        tex(4096, 4096, 1, 0, 98, 0, 0),
        Expect {
            size: Some(0x1556000),
            align: Some(0x1000),
            mips: Some(13),
            first_tail: Some(7),
            tail_count: Some(6),
            levels: vec![
                Lv(0x556000, 1024, 1024, 5, false),
                Lv(0x156000, 512, 512, 5, false),
                Lv(0x56000, 256, 256, 5, false),
                Lv(0x16000, 128, 128, 5, false),
                Lv(0x6000, 64, 64, 5, false),
                Lv(0x2000, 32, 32, 5, false),
                Lv(0x1000, 16, 16, 5, false),
                Lv(0x800, 8, 16, 1, true),
                Lv(0x600, 4, 8, 1, true),
                Lv(0x500, 4, 4, 1, true),
                Lv(0x400, 4, 4, 1, true),
                Lv(0x300, 4, 4, 1, true),
                Lv(0x200, 4, 4, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x556000), (0, 17, 5, 0, 0x557150)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_case_f() {
    let _s = serial();
    check(
        "F",
        tex(300, 70, 1, 3, 61, 0x105, 0),
        Expect {
            size: Some(0xF000),
            align: Some(0x1000),
            first_tail: Some(3),
            levels: vec![Lv(0x5000, 320, 128, 5, false), Lv(0x2000, 192, 64, 5, false), Lv(0x0, 128, 64, 5, false)],
            elements: vec![(0, 0, 0, 0, 0x5000), (0, 31, 7, 0, 0x527F), (2, 1, 1, 0, 0x11)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_case_g() {
    let _s = serial();
    check(
        "G",
        tex(512, 512, 6, 0, 10, 0x109, 0),
        Expect {
            size: Some(0x1020000),
            align: Some(0x10000),
            mips: Some(10),
            slice: Some(0x2B0000),
            first_tail: Some(3),
            levels: vec![
                Lv(0xB0000, 512, 512, 9, false),
                Lv(0x30000, 256, 256, 9, false),
                Lv(0x10000, 128, 128, 9, false),
                Lv(0x8000, 64, 64, 1, true),
                Lv(0x4000, 32, 32, 1, true),
                Lv(0x2000, 16, 16, 1, true),
                Lv(0x1000, 8, 8, 1, true),
                Lv(0x800, 8, 4, 1, true),
                Lv(0x600, 8, 4, 1, true),
                Lv(0x500, 8, 4, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0xB0000), (0, 0, 0, 1, 0x360000), (0, 0, 0, 5, 0xE20000), (5, 0, 0, 3, 0x812000)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_case_h() {
    let _s = serial();
    check(
        "H",
        tex(128, 128, 4, 8, 28, 0x105, 0),
        Expect {
            size: Some(0x58000),
            slice: Some(0x16000),
            first_tail: Some(3),
            levels: vec![
                Lv(0x6000, 128, 128, 5, false),
                Lv(0x2000, 64, 64, 5, false),
                Lv(0x1000, 32, 32, 5, false),
                Lv(0x800, 16, 32, 1, true),
                Lv(0x600, 8, 16, 1, true),
                Lv(0x500, 8, 8, 1, true),
                Lv(0x400, 8, 8, 1, true),
                Lv(0x300, 8, 8, 1, true),
            ],
            elements: vec![(0, 0, 0, 2, 0x32000), (3, 1, 1, 3, 0x42814)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_cases_i_and_j_linear() {
    let _s = serial();
    let c = check(
        "I",
        tex(1000, 33, 1, 1, 2, 1, 0),
        Expect {
            size: Some(0x81F00),
            align: Some(0x100),
            levels: vec![Lv(0, 1008, 33, 0, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x10), (0, 0, 1, 0, 0x3F00)],
            ..Expect::default()
        },
    );
    assert_eq!(Record(c.record()).level_u32(0, 0, 0x1C), 16128);
    check(
        "J",
        tex(300, 33, 1, 0, 28, 1, 0),
        Expect {
            size: Some(0xF700),
            align: Some(0x100),
            mips: Some(9),
            first_tail: Some(9),
            tail_count: Some(0),
            levels: vec![
                Lv(0x5200, 320, 33, 0, false),
                Lv(0x1F00, 192, 17, 0, false),
                Lv(0xD00, 128, 9, 0, false),
                Lv(0x800, 64, 5, 0, false),
                Lv(0x500, 64, 3, 0, false),
                Lv(0x300, 64, 2, 0, false),
                Lv(0x200, 64, 1, 0, false),
                Lv(0x100, 64, 1, 0, false),
                Lv(0x0, 64, 1, 0, false),
            ],
            elements: vec![(0, 0, 1, 0, 0x5700), (1, 0, 1, 0, 0x2200)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_cases_k_and_l() {
    let _s = serial();
    check(
        "K",
        tex(1920, 1080, 1, 1, 40, 0, 2),
        Expect {
            size: Some(0x870000),
            align: Some(0x10000),
            planes: Some(1),
            levels: vec![Lv(0, 1920, 1152, 24, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x8)],
            ..Expect::default()
        },
    );
    let c = check(
        "L",
        D3d { samples: 4, ..tex(1920, 1080, 1, 1, 28, 0, 1) },
        Expect {
            size: Some(0x1FE0000),
            align: Some(0x10000),
            planes: Some(1),
            levels: vec![Lv(0, 1920, 1088, 27, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4)],
            ..Expect::default()
        },
    );
    assert_eq!(Record(c.record()).level_u32(0, 0, 0x44), 4);
    assert_eq!(c.texel(0, 0, 0, 0, 0, 4), u64::MAX, "sample out of range");
    assert_ne!(c.texel(0, 0, 3, 3, 0, 3), u64::MAX);
}

#[test]
fn computer_cases_m_n_o() {
    let _s = serial();
    check(
        "M",
        tex(64, 64, 1, 0, 28, 0, 4),
        Expect {
            size: Some(0x6000),
            align: Some(0x1000),
            mips: Some(7),
            first_tail: Some(2),
            levels: vec![
                Lv(0x2000, 64, 64, 21, false),
                Lv(0x1000, 32, 32, 21, false),
                Lv(0x800, 16, 32, 1, true),
                Lv(0x600, 8, 16, 1, true),
                Lv(0x500, 8, 8, 1, true),
                Lv(0x400, 8, 8, 1, true),
                Lv(0x300, 8, 8, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x2000), (0, 1, 1, 0, 0x2014)],
            ..Expect::default()
        },
    );
    check("N", tex(1, 1, 1, 1, 28, 0x109, 0), Expect { size: Some(0x10000), align: Some(0x10000), levels: vec![Lv(0, 128, 128, 9, false)], elements: vec![(0, 0, 0, 0, 0)], ..Expect::default() });
    check(
        "O",
        tex(256, 256, 1, 1, 28, 2, 0),
        Expect { size: Some(0x40000), align: Some(0x10000), levels: vec![Lv(0, 0, 0, 10, false)], elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10)], ..Expect::default() },
    );
}

#[test]
fn computer_cases_p_and_q() {
    let _s = serial();
    let p = tex(256, 256, 1, 1, 28, 0x11B, 0);
    let c = check(
        "P",
        p,
        Expect {
            size: Some(0x40000),
            align: Some(0x10000),
            planes: Some(1),
            levels: vec![Lv(0, 0, 0, 27, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10), (0, 64, 0, 0, 0x1400), (0, 0, 64, 0, 0x8800), (0, 200, 130, 0, 0x3B420)],
            ..Expect::default()
        },
    );
    let r = Record(c.record());
    assert_eq!((r.level_u32(0, 0, 0x50), r.level_u32(0, 0, 0x58)), (0x8000, 0x8000));
    XGSetHardwareVersion(4);
    let c4 = check(
        "P4",
        p,
        Expect {
            size: Some(0x40000),
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10), (0, 64, 0, 0, 0x4100), (0, 0, 64, 0, 0x8100), (0, 200, 130, 0, 0x38920)],
            ..Expect::default()
        },
    );
    let r4 = Record(c4.record());
    assert_eq!((r4.level_u32(0, 0, 0x50), r4.level_u32(0, 0, 0x58)), (0xE000, 0x2000));
    XGSetHardwareVersion(0);
    check(
        "Q",
        tex(256, 256, 2, 1, 28, 0x119, 0),
        Expect {
            size: Some(0x80000),
            slice: Some(0x40000),
            levels: vec![Lv(0, 0, 0, 25, false)],
            elements: vec![(0, 0, 0, 0, 0x0), (0, 1, 0, 0, 0x4), (0, 0, 1, 0, 0x10), (0, 64, 0, 0, 0x8400), (0, 0, 64, 0, 0x4800), (0, 0, 0, 1, 0x41000)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_cases_r_s_t_u() {
    let _s = serial();
    let c = check(
        "R",
        volume(64, 64, 64, 0, 0),
        Expect {
            size: Some(0x158000),
            align: Some(0x1000),
            mips: Some(7),
            slice: Some(0x2B000),
            first_tail: Some(3),
            levels: vec![
                Lv(0xB000, 64, 64, 5, false),
                Lv(0x3000, 32, 32, 5, false),
                Lv(0x1000, 16, 16, 5, false),
                Lv(0, 8, 8, 1, true),
                Lv(0, 4, 4, 1, true),
                Lv(0, 4, 4, 1, true),
                Lv(0, 4, 4, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0xB000), (0, 1, 0, 0, 0xB004), (0, 0, 0, 1, 0xB010), (0, 33, 17, 9, 0x4201C)],
            ..Expect::default()
        },
    );
    let r = Record(c.record());
    assert_eq!(r.dimension(), 4);
    assert_eq!(r.padded(3).2, 8, "tail level 3 padded depth");
    assert_eq!(r.padded(0).2, 64);
    assert_eq!(r.level_u32(0, 0, 0x60), 8, "block depth");
    let s = check(
        "S",
        volume(64, 32, 16, 0, 0x105),
        Expect {
            size: Some(0x2E000),
            slice: Some(0x17000),
            first_tail: Some(3),
            levels: vec![Lv(0x7000, 64, 32, 5, false), Lv(0x3000, 32, 16, 5, false), Lv(0x1000, 16, 16, 5, false), Lv(0, 0, 0, 1, true), Lv(0, 0, 0, 1, true), Lv(0, 0, 0, 1, true), Lv(0, 0, 0, 1, true)],
            elements: vec![(0, 0, 0, 0, 0x7000), (0, 1, 1, 1, 0x701C), (3, 0, 0, 0, 0x800)],
            ..Expect::default()
        },
    );
    let rs = Record(s.record());
    assert_eq!([3, 4, 5, 6].map(|i| rs.level_u32(0, i, 0x68)), [0x4000, 0x2004, 0x4, 0x2000]);
    check(
        "T",
        volume(100, 60, 7, 0, 0x11B),
        Expect {
            size: Some(0xE0000),
            align: Some(0x10000),
            slice: Some(0x20000),
            first_tail: Some(1),
            levels: vec![
                Lv(0x10000, 128, 128, 27, false),
                Lv(0, 64, 128, 3, true),
                Lv(0, 32, 64, 3, true),
                Lv(0, 16, 32, 3, true),
                Lv(0, 8, 16, 3, true),
                Lv(0, 8, 8, 3, true),
                Lv(0, 8, 8, 3, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x10000), (0, 0, 0, 6, 0xD0600)],
            ..Expect::default()
        },
    );
    let u = check(
        "U",
        volume(100, 60, 7, 0, 1),
        Expect {
            size: Some(0x4EC00),
            align: Some(0x100),
            slice: Some(0xB400),
            levels: vec![
                Lv(0x3C00, 128, 60, 0, false),
                Lv(0x1E00, 64, 30, 0, false),
                Lv(0xF00, 64, 15, 0, false),
                Lv(0x700, 64, 8, 0, false),
                Lv(0x300, 64, 4, 0, false),
                Lv(0x100, 64, 2, 0, false),
                Lv(0x0, 64, 1, 0, false),
            ],
            elements: vec![(0, 0, 0, 0, 0x3C00), (0, 0, 0, 1, 0xF000)],
            ..Expect::default()
        },
    );
    assert_eq!(Record(u.record()).padded(0).2, 7);
}

#[test]
fn computer_cases_v_w_1d() {
    let _s = serial();
    let v = check(
        "V",
        D3d { dimension: 2, width: 4096, height: 1, depth: 1, mips: 0, format: 28, samples: 1, layout: 0, flags: 0 },
        Expect {
            size: Some(0x8500),
            align: Some(0x100),
            mips: Some(13),
            levels: [0x4500u64, 0x2500, 0x1500, 0xD00, 0x900, 0x700, 0x600, 0x500, 0x400, 0x300, 0x200, 0x100, 0x0].into_iter().map(|o| Lv(o, 0, 0, 0, false)).collect(),
            elements: vec![(0, 0, 0, 0, 0x4500), (0, 100, 0, 0, 0x4690)],
            ..Expect::default()
        },
    );
    let r = Record(v.record());
    assert_eq!(r.dimension(), 2);
    assert_eq!([0, 5, 6, 12].map(|i| r.padded(i).0), [4096, 128, 64, 64], "the last seven levels are padded to 64 elements");
    assert_eq!(v.mip_offset(0, 13), u64::MAX, "1D mip beyond the count");
    let w = check(
        "W",
        D3d { dimension: 2, width: 256, height: 1, depth: 3, mips: 1, format: 28, samples: 1, layout: 1, flags: 0 },
        Expect { size: Some(0xC00), slice: Some(0x400), elements: vec![(0, 5, 0, 2, 0x814)], ..Expect::default() },
    );
    assert_eq!(w.element(0, 5, 0, 3), u64::MAX, "slice beyond the array");
}

#[test]
fn computer_error_cases() {
    let _s = serial();
    let hr = |d: D3d| Computer::create(&d).err().unwrap_or(OK);
    assert_eq!(hr(tex(256, 256, 1, 1, 28, 0x103, 0)), INVALIDARG, "X");
    assert_eq!(hr(tex(256, 256, 1, 16, 28, 0, 0)), INVALIDARG, "Y");
    assert_eq!(hr(tex(0, 256, 1, 1, 28, 0, 0)), INVALIDARG, "Z");
    assert_eq!(hr(tex(256, 256, 1, 1, 28, 3, 0)), INVALIDARG, "AB");
    assert_eq!(hr(D3d { dimension: 1, ..tex(256, 1, 1, 1, 28, 0, 0) }), INVALIDARG, "AC");
    assert_eq!(hr(D3d { dimension: 5, ..tex(256, 256, 1, 1, 28, 0, 0) }), INVALIDARG, "AD");
    assert_eq!(hr(D3d { dimension: 0, ..tex(256, 256, 1, 1, 28, 0, 0) }), INVALIDARG);
    assert_eq!(hr(tex(256, 256, 1, 1, 0, 0, 0)), NOTIMPL, "AE");
    assert_eq!(hr(tex(256, 256, 1, 1, 45, 0, 2)), NOTIMPL, "AF");
    let aa = check(
        "AA",
        tex(256, 256, 1, 1, 71, 1, 0),
        Expect { size: Some(0x8000), align: Some(0x100), levels: vec![Lv(0, 64, 64, 0, false)], ..Expect::default() },
    );
    let r = Record(aa.record());
    assert_eq!((r.level_u32(0, 0, 0x18), r.level_u32(0, 0, 0x1C)), (256, 512));
    check("AG", tex(4096, 4096, 1, 1, 28, 0x109, 0), Expect { size: Some(0x4000000), align: Some(0x10000), elements: vec![(0, 4095, 4095, 0, 0x3FFFFFC)], ..Expect::default() });
}

#[test]
fn computer_cases_ah_ai_aj() {
    let _s = serial();
    check(
        "AH",
        tex(33, 129, 1, 0, 56, 0x105, 0),
        Expect {
            size: Some(0xB000),
            mips: Some(8),
            first_tail: Some(3),
            levels: vec![
                Lv(0x6000, 64, 160, 5, false),
                Lv(0x3000, 64, 96, 5, false),
                Lv(0x1000, 64, 64, 5, false),
                Lv(0x800, 32, 32, 1, true),
                Lv(0x600, 16, 16, 1, true),
                Lv(0x500, 16, 8, 1, true),
                Lv(0x400, 16, 8, 1, true),
                Lv(0x300, 16, 8, 1, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x6000), (0, 32, 128, 0, 0xA800)],
            ..Expect::default()
        },
    );
    check(
        "AI",
        tex(64, 64, 1, 0, 28, 0x118, 0),
        Expect {
            size: Some(0x10000),
            align: Some(0x10000),
            mips: Some(7),
            first_tail: Some(0),
            levels: vec![
                Lv(0x8000, 64, 128, 24, true),
                Lv(0x4000, 32, 64, 24, true),
                Lv(0x2000, 16, 32, 24, true),
                Lv(0x1000, 8, 16, 24, true),
                Lv(0x800, 8, 8, 24, true),
                Lv(0x600, 0, 0, 24, true),
                Lv(0x500, 0, 0, 24, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x1400), (0, 1, 0, 0, 0x1404), (0, 0, 1, 0, 0x1408)],
            ..Expect::default()
        },
    );
    check(
        "AJ",
        tex(2048, 2048, 1, 0, 61, 0x118, 0),
        Expect {
            size: Some(0x570000),
            mips: Some(12),
            first_tail: Some(5),
            levels: vec![
                Lv(0x170000, 2048, 2048, 24, false),
                Lv(0x70000, 1024, 1024, 24, false),
                Lv(0x30000, 512, 512, 24, false),
                Lv(0x20000, 256, 256, 24, false),
                Lv(0x10000, 256, 256, 24, false),
                Lv(0x8000, 128, 256, 24, true),
                Lv(0x4000, 64, 128, 24, true),
                Lv(0x2000, 32, 64, 24, true),
                Lv(0x1000, 16, 32, 24, true),
                Lv(0x800, 16, 16, 24, true),
                Lv(0x600, 0, 0, 24, true),
                Lv(0x500, 0, 0, 24, true),
            ],
            elements: vec![(0, 0, 0, 0, 0x170000)],
            ..Expect::default()
        },
    );
}

#[test]
fn computer_object_methods() {
    let _s = serial();
    let c = Computer::create(&tex(64, 64, 1, 1, 28, 0, 0)).unwrap();
    assert_eq!(c.add_ref(), 2);
    assert_eq!(c.release(), 1);
    assert_eq!(c.null_layout(), INVALIDARG);
    assert_eq!(c.mip_offset(0, 7), 0, "2D mip beyond the count");
    assert_eq!(c.mip_offset(1, 0), 0, "plane other than 0");
    assert_eq!(c.texel(1, 0, 0, 0, 0, 0), u64::MAX);
    assert_eq!(c.texel(0, 1, 0, 0, 0, 0), u64::MAX);
    assert_eq!(c.texel(0, 0, 64, 0, 0, 0), u64::MAX, "x beyond the padded level");
    assert_eq!(c.texel(0, 0, 0, 64, 0, 0), u64::MAX);
    assert_eq!(c.texel(0, 0, 0, 0, 1, 0), u64::MAX, "slice beyond the array");
    assert_eq!((c.slot7(), c.slot11()), (NOTIMPL, NOTIMPL));
    assert_eq!(c.tiling_null(), xg_xs::abi::E_FAIL, "4 KiB blocks have no 64 KiB tiling");
    let volume = Computer::create(&volume(16, 16, 16, 1, 0)).unwrap();
    assert_eq!(volume.mip_offset(0, 3), u64::MAX, "3D mip beyond the count");
    let mut bit = 7u32;
    // SAFETY: the pointers are valid.
    let _ = &mut bit;
    let mut out = [0u8; 4];
    assert_eq!(volume.copy_from(&mut out, 0, &[0; 4], 4, 4), INVALIDARG, "CopyFromSubresource on a 3D computer");
    assert_eq!(c.copy_into(&mut [0; 16], 1, &[0; 4], 4, 4), INVALIDARG, "slice beyond the array");
}

#[test]
fn copies_agree_with_the_core_detiler() {
    use decant_texture::tiling::{Layout, Mode, detile_with};
    let _s = serial();
    // 2D, one level, mode 10 (64KB_D) of 4 byte elements, and mode 5 (4KB_S) of 8 byte elements.
    for (format, layout, mode, bpe, w, h) in [(28u32, 0x10Au32, Mode::Sw64kD, 4usize, 100usize, 70usize), (10, 0x105, Mode::Sw4kS, 8, 33, 17)] {
        let c = Computer::create(&tex(w as u64, h as u32, 1, 1, format, layout, 0)).unwrap();
        let linear: Vec<u8> = (0..w * h * bpe).map(|i| (i * 7 + i / 251) as u8).collect();
        let mut memory = vec![0u8; c.size() as usize];
        assert_eq!(c.copy_into(&mut memory, 0, &linear, (w * bpe) as u32, 0), OK);
        // Mode 10 uses the RB+ tables, mode 5 the plain ones.
        let layout = Layout { rb_plus: mode == Mode::Sw64kD, ..Layout::default() };
        assert_eq!(detile_with(mode, &layout, &memory, w, h, bpe).unwrap(), linear, "{mode:?}");
        let mut back = vec![0u8; linear.len()];
        assert_eq!(c.copy_from(&mut back, 0, &memory, (w * bpe) as u32, 0), OK);
        assert_eq!(back, linear);
    }
    // The second slice of an array of 2D textures goes after the first.
    let c = Computer::create(&tex(64, 64, 2, 1, 28, 0x10A, 0)).unwrap();
    let linear = vec![0xABu8; 64 * 64 * 4];
    let mut memory = vec![0u8; c.size() as usize];
    assert_eq!(c.copy_into(&mut memory, 1, &linear, 256, 0), OK);
    assert!(memory[..0x10000].iter().all(|&b| b == 0));
    assert_eq!(memory[0x10000..].iter().filter(|&&b| b == 0xAB).count(), 64 * 64 * 4);
    // A 3D texture, mode 5 thick: each depth slice goes where the element offsets say.
    let c = Computer::create(&volume(16, 16, 8, 1, 0)).unwrap();
    let linear: Vec<u8> = (0..16 * 16 * 8 * 4).map(|i| (i * 3 + i / 13) as u8).collect();
    let mut memory = vec![0u8; c.size() as usize];
    assert_eq!(c.copy_into(&mut memory, 0, &linear, 64, 64 * 16), OK);
    for (z, y, x) in [(0u32, 0u32, 0u32), (3, 5, 9), (7, 15, 15)] {
        let at = c.element(0, u64::from(x), y, z) as usize;
        let from = (z as usize * 16 * 16 + y as usize * 16 + x as usize) * 4;
        assert_eq!(memory[at..at + 4], linear[from..from + 4]);
    }
}

#[test]
fn tile_shape_vectors() {
    let _s = serial();
    // (dimension, format, bytes, samples) -> result, texels, elements.
    let rows: [((u32, u32, u32, u32), Hresult, (u32, u32, u32), (u32, u32, u32)); 14] = [
        ((1, 0, 0, 1), OK, (0, 0, 0), (65536, 1, 1)),
        ((1, 28, 0, 1), INVALIDARG, (0, 0, 0), (0, 0, 0)),
        ((3, 28, 0, 1), OK, (128, 128, 1), (128, 128, 1)),
        ((3, 71, 0, 1), OK, (512, 256, 1), (128, 64, 1)),
        ((3, 98, 0, 1), OK, (256, 256, 1), (64, 64, 1)),
        ((3, 61, 0, 1), OK, (256, 256, 1), (256, 256, 1)),
        ((3, 0, 8, 1), OK, (0, 0, 0), (128, 64, 1)),
        ((3, 28, 0, 2), OK, (128, 64, 1), (128, 64, 1)),
        ((3, 28, 0, 4), OK, (64, 64, 1), (64, 64, 1)),
        ((3, 28, 0, 8), OK, (64, 32, 1), (64, 32, 1)),
        ((3, 28, 0, 16), INVALIDARG, (0, 0, 0), (0, 0, 0)),
        ((4, 28, 0, 1), NOTIMPL, (0, 0, 0), (0, 0, 0)),
        ((2, 28, 0, 1), NOTIMPL, (0, 0, 0), (0, 0, 0)),
        ((3, 0, 3, 1), INVALIDARG, (0, 0, 0), (0, 0, 0)),
    ];
    for (i, ((dim, format, bytes, samples), expected, texels, elements)) in rows.into_iter().enumerate() {
        let (mut t, mut e) = (XgTileShape { width: 9, height: 9, depth: 9 }, XgTileShape { width: 9, height: 9, depth: 9 });
        // SAFETY: valid pointers.
        let hr = unsafe { XGComputeTileShape(dim, format, bytes, samples, &mut t, &mut e) };
        assert_eq!(hr, expected, "row {}", i + 1);
        if expected == OK {
            assert_eq!((t.width, t.height, t.depth), texels, "texels of row {}", i + 1);
            assert_eq!((e.width, e.height, e.depth), elements, "elements of row {}", i + 1);
        } else {
            assert_eq!(t, XgTileShape { width: 9, height: 9, depth: 9 }, "nothing is written on failure");
        }
    }
    // SAFETY: null pointers are allowed.
    assert_eq!(unsafe { XGComputeTileShape(3, 28, 0, 1, std::ptr::null_mut(), std::ptr::null_mut()) }, OK);
}

#[test]
fn buffer_layout_vectors() {
    use xg_xs::abi::XgBufferDesc;
    let _s = serial();
    let rows: [((u32, u32, u32), Hresult, u64, u64); 7] = [
        ((0x64, 0x0, 0x0), OK, 0x64, 4),
        ((0x0, 0x0, 0x0), INVALIDARG, 0, 0),
        ((0x64, 0x100, 0x0), INVALIDARG, 0, 0),
        ((0x64, 0x0, 0x400_0000), INVALIDARG, 0, 0),
        ((0x20000, 0x0, 0x400_0000), OK, 0x20000, 0x10000),
        ((0x64, 0x8, 0x0), OK, 0x64, 4),
        ((0x64, 0x0, 0x800_0000), OK, 0x64, 0x10000),
    ];
    for (i, ((width, bind, misc), expected, size, align)) in rows.into_iter().enumerate() {
        let desc = XgBufferDesc { byte_width: width, bind_flags: bind, misc_flags: misc, ..XgBufferDesc::default() };
        let mut record = vec![0u8; 0x1B40];
        // SAFETY: valid pointers.
        let hr = unsafe { XGComputeBufferLayout(&desc, record.as_mut_ptr()) };
        assert_eq!(hr, expected, "row {}", i + 1);
        if expected == OK {
            let r = Record(record);
            assert_eq!((r.size(), r.align(), r.u64(0x10), r.dimension()), (size, align, 1, 1), "row {}", i + 1);
            assert_eq!(r.planes(), 0);
        }
    }
    let desc = XgBufferDesc { byte_width: 4, ..XgBufferDesc::default() };
    // SAFETY: null pointers are rejected.
    unsafe {
        assert_eq!(XGComputeBufferLayout(std::ptr::null(), [0u8; 4].as_mut_ptr()), INVALIDARG);
        assert_eq!(XGComputeBufferLayout(&desc, std::ptr::null_mut()), INVALIDARG);
    }
}

#[test]
fn other_function_vectors() {
    let _s = serial();
    for (mode, ok) in [(0, false), (4, true), (8, true), (12, true), (24, true), (28, true), (29, false), (32, false)] {
        assert_eq!(XGValidateDepthStencilSwizzleMode(0, 0, mode), if ok { OK } else { xg_xs::abi::E_FAIL }, "mode {mode}");
    }
    let mut stencil = 0u32;
    // SAFETY: a valid pointer, or null.
    unsafe {
        assert_eq!(XGComputeOptimalDepthStencilSwizzleMode(0, 0, 0, 0, 0, 0, 0, 0, &mut stencil), 0x18);
        assert_eq!(stencil, 0x18);
        assert_eq!(XGComputeOptimalDepthStencilSwizzleMode(1, 2, 3, 4, 5, 6, 7, 8, std::ptr::null_mut()), 0x18);
    }
    // The state flags, in a fresh state: the tests reset the version only, so the initialised flag is read from its own steps.
    XGTerminateLibrary();
    assert_eq!(XGInitializeLibrary(), 0);
    assert_eq!(XGInitializeLibrary(), 1);
    assert_eq!(XGTerminateLibrary(), 0);
    assert_eq!(XGTerminateLibrary(), 1);
    assert_eq!(XGSetHardwareVersion(4), 0, "behaves like the initialisation");
    assert_eq!(XGSetHardwareVersion(0), 1);
    XGTerminateLibrary();
}

#[test]
fn suggest_swizzle_mode_writes_the_layout_field_of_the_description() {
    let _s = serial();
    // As the title calls it: flags 0, the answer goes into the description's own Layout field at 0x2C.
    let suggest = |d: D3d| {
        let mut bytes = d.bytes();
        let layout = bytes[0x2C..].as_mut_ptr().cast::<u32>();
        // SAFETY: `layout` points into `bytes`, the export reads the description before it writes.
        let hr = unsafe { XGSuggestSwizzleMode(bytes.as_ptr().cast(), 0, layout) };
        (hr, u32::from_le_bytes(bytes[0x2C..0x30].try_into().unwrap()))
    };
    assert_eq!(suggest(tex(256, 256, 1, 0, 28, 0, 0)), (OK, 0x105), "a plain texture: 4KB_S");
    assert_eq!(suggest(tex(256, 256, 1, 0, 28, 0x10A, 0)), (OK, 0x105), "the Layout field the title left there is ignored");
    assert_eq!(suggest(tex(1920, 1080, 1, 1, 28, 0, 1)), (OK, 0x11B), "a render target: 64KB_R_X");
    assert_eq!(suggest(tex(1920, 1080, 1, 1, 40, 0, 2)), (OK, 0x118), "a depth buffer: 64KB_Z_X");
    assert_eq!(suggest(D3d { dimension: 2, ..tex(256, 1, 1, 1, 28, 0, 0) }), (OK, 0x100), "1D: linear");
    // The suggestion is what the computer picks for Layout 0.
    for d in [tex(64, 64, 1, 0, 28, 0, 4), tex(4096, 4096, 1, 0, 98, 0, 0), tex(512, 512, 6, 0, 10, 0, 0)] {
        let (hr, layout) = suggest(d);
        assert_eq!(hr, OK);
        let picked = Record(Computer::create(&d).unwrap().record()).mode(0);
        let explicit = Record(Computer::create(&D3d { layout, ..d }).unwrap().record());
        assert_eq!((layout & 0x1F, explicit.mode(0)), (picked, picked));
    }
    assert_eq!(suggest(D3d { dimension: 1, ..tex(256, 1, 1, 1, 28, 0, 0) }), (INVALIDARG, 0), "a buffer is refused, the field stays");
    let mut out = 0u32;
    // SAFETY: null pointers are refused.
    unsafe {
        assert_eq!(XGSuggestSwizzleMode(std::ptr::null(), 0, &mut out), INVALIDARG);
        assert_eq!(XGSuggestSwizzleMode(&tex(4, 4, 1, 1, 28, 0, 0).bytes(), 0, std::ptr::null_mut()), INVALIDARG);
    }
}

#[test]
fn exclusive_locking() {
    let _s = serial();
    assert_eq!(XGEnterExclusiveLock(), 0, "locking is off");
    XGLeaveExclusiveLock();
    XGEnableExclusiveLocking(1);
    assert_eq!(XGEnterExclusiveLock(), 1);
    // The same thread can enter again, and the layout functions take the lock too.
    assert_eq!(XGEnterExclusiveLock(), 1);
    assert!(Computer::create(&tex(16, 16, 1, 1, 28, 0, 0)).is_ok());
    let other = std::thread::spawn(|| {
        XGComputeOptimalSwizzleMode(3, 28, 64, 64, 1, 1, 8, 0)
    });
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!other.is_finished(), "another thread waits while the lock is held");
    XGLeaveExclusiveLock();
    XGLeaveExclusiveLock();
    assert_eq!(other.join().unwrap(), 5);
    XGEnableExclusiveLocking(0);
}

#[test]
fn dcc_and_bvh_are_logging_stand_ins() {
    let _s = serial();
    // SAFETY: the pointers are not used.
    unsafe {
        let null = std::ptr::null_mut::<std::ffi::c_void>();
        assert_eq!(XGEncodeDCC(1, null, null, null, std::ptr::null_mut(), 28, 24, 3, std::ptr::null()), NOTIMPL);
        assert_eq!(XGDecodeDCC(1, null, null, null, 28, 24, 3, std::ptr::null()), NOTIMPL);
        assert_eq!(XGCreateBVHComputer(std::ptr::null_mut()), NOTIMPL);
    }
}

#[test]
fn texture_layout_functions_and_typed_computers() {
    use xg_xs::abi::{XgTexture1dDesc, XgTexture2dDesc, XgTexture3dDesc};
    use xg_xs::computer::TextureComputer;
    let _s = serial();
    let d2 = XgTexture2dDesc { width: 256, height: 256, mip_levels: 1, array_size: 1, format: 28, sample_count: 1, swizzle_mode: 5, ..XgTexture2dDesc::default() };
    let mut record = vec![0u8; 0x1B40];
    // SAFETY: valid pointers.
    unsafe {
        assert_eq!(XGComputeTexture2DLayout(&d2, record.as_mut_ptr()), OK);
        let r = Record(record.clone());
        assert_eq!((r.size(), r.align(), r.planes(), r.dimension()), (0x40000, 0x1000, 1, 3));
        assert_eq!(XGComputeTexture2DLayout(std::ptr::null(), record.as_mut_ptr()), INVALIDARG);
        assert_eq!(XGComputeTexture2DLayout(&d2, std::ptr::null_mut()), INVALIDARG);
        let mut out: *mut TextureComputer = std::ptr::null_mut();
        assert_eq!(XGCreateTexture2DComputer(&d2, &mut out), OK);
        assert!(!out.is_null());
        Computer(out);
        assert_eq!(XGCreateTexture2DComputer(&d2, std::ptr::null_mut()), INVALIDARG);
        let bad = XgTexture2dDesc { swizzle_mode: 3, ..d2 };
        let mut none: *mut TextureComputer = std::ptr::null_mut();
        assert_eq!(XGCreateTexture2DComputer(&bad, &mut none), INVALIDARG);
        assert!(none.is_null());
        // 1D: linear, 300 elements of 4 bytes, pitch 512 bytes.
        let d1 = XgTexture1dDesc { width: 300, mip_levels: 1, array_size: 1, format: 28, usage: 3, ..XgTexture1dDesc::default() };
        assert_eq!(XGComputeTexture1DLayout(&d1, record.as_mut_ptr()), OK);
        let r1 = Record(record.clone());
        assert_eq!((r1.size(), r1.dimension(), r1.level_u32(0, 0, 0x1C)), (1280, 2, 1280));
        assert_eq!(XGCreateTexture1DComputer(&d1, &mut out), OK);
        Computer(out);
        // 3D: thick mode 5.
        let d3 = XgTexture3dDesc { width: 64, height: 64, depth: 64, mip_levels: 0, format: 28, swizzle_mode: 5, ..XgTexture3dDesc::default() };
        assert_eq!(XGComputeTexture3DLayout(&d3, record.as_mut_ptr()), OK);
        assert_eq!(Record(record.clone()).size(), 0x158000);
        assert_eq!(XGCreateTexture3DComputer(&d3, &mut out), OK);
        Computer(out);
        assert_eq!(XGComputeTexture3DLayout(&XgTexture3dDesc { misc_flags: 0x800_0000, ..d3 }, record.as_mut_ptr()), INVALIDARG);
        assert_eq!(XGComputeTexture3DLayout(&XgTexture3dDesc { width: 20000, ..d3 }, record.as_mut_ptr()), NOTIMPL);
    }
}

#[test]
fn descriptors() {
    let _s = serial();
    // SAFETY: valid pointers.
    unsafe {
        let image: [u32; 8] = [0, (0x80 << 20) | (3 << 30), 63 | (127 << 14), (9 << 28) | (27 << 20) | (2 << 16), 0, 2 << 4, 0, 0];
        assert_eq!(XGValidateDescriptor(8, image.as_ptr(), 0), OK);
        assert_eq!(XGValidateDescriptor(8, std::ptr::null(), 0), INVALIDARG);
        assert_eq!(XGValidateDescriptor(6, image.as_ptr(), 0), INVALIDARG);
        assert_eq!(XGValidateDescriptor(8, image.as_ptr(), 1), INVALIDARG);
        assert_eq!(XGValidateDescriptor(4, [0u32; 4].as_ptr(), 0), OK);
        let mut bad = image;
        bad[1] |= 1;
        assert_eq!(XGValidateDescriptor(8, bad.as_ptr(), 0), xg_xs::abi::E_FAIL);
        let mut out: *mut xg_xs::computer::TextureComputer = std::ptr::null_mut();
        assert_eq!(XGCreateTextureComputerFromDescriptor(std::ptr::null(), &mut out), INVALIDARG);
        assert_eq!(XGCreateTextureComputerFromDescriptor(image.as_ptr(), std::ptr::null_mut()), INVALIDARG);
        let mut invalid = image;
        invalid[3] &= 0x7FFF_FFFF;
        assert_eq!(XGCreateTextureComputerFromDescriptor(invalid.as_ptr(), &mut out), INVALIDARG);
        // The GFX10 image format table is an open point: a valid descriptor gives E_NOTIMPL.
        assert_eq!(XGCreateTextureComputerFromDescriptor(image.as_ptr(), &mut out), NOTIMPL);
        assert!(out.is_null());
    }
}
