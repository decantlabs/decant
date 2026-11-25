//! Tiling and detiling through the texture computer table, as a title calls it: `CopyIntoSubresource` (slot 8) of every
//! subresource then `CopyFromSubresource` (slot 9) gives the input back, for every 2D swizzle mode, the element sizes, the
//! sizes of `tests/detile_vectors.txt`, mip chains with their tail, arrays and cubes. The tiled bytes are checked against the
//! core detiler where its tables are the console's, and slot 10 (`GetResourceTiling`) against the layout.
#![allow(unsafe_code)]

mod common;

use common::*;
use decant_texture::tiling::{Layout, detile_with};
use decant_texture::xg::modes::core_mode;
use xg_xs::abi::{E_FAIL, XgTileShape};
use xg_xs::exports::XGSetHardwareVersion;

/// Every 2D tiled mode of the specification.
const MODES: [u32; 16] = [1, 2, 5, 6, 9, 10, 17, 18, 21, 22, 24, 25, 26, 27, 28, 31];

/// (format, bytes per element, texels per element side).
const FORMATS: [(u32, usize, usize); 7] = [(61, 1, 1), (56, 2, 1), (28, 4, 1), (10, 8, 1), (2, 16, 1), (71, 8, 4), (98, 16, 4)];

/// The texel sizes of the reference vectors (`size`, `detile` and `detilex` lines).
const SIZES: [(u32, u32); 14] = [(1, 1), (4, 4), (16, 16), (16, 8), (33, 17), (64, 64), (100, 70), (256, 64), (300, 10), (17, 300), (130, 70), (200, 130), (257, 129), (512, 512)];

fn tex(w: u32, h: u32, a: u16, mips: u16, format: u32, mode: u32) -> D3d {
    D3d { dimension: 3, width: u64::from(w), height: h, depth: a, mips, format, samples: 1, layout: 0x100 | mode, flags: 0 }
}

/// Elements of a level along one side: the element count of level 0 halved and rounded up (1.6).
fn elements(texels: usize, side: usize, mip: u32) -> usize {
    texels.div_ceil(side).div_ceil(1 << mip).max(1)
}

/// Distinct bytes for each subresource.
fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(7) ^ (i >> 8).wrapping_mul(13) ^ seed.wrapping_mul(101)) as u8).collect()
}

/// Copies every subresource in, then each one out again. Returns the tiled memory and the linear inputs, `None` when the
/// computer refuses the description.
fn round_trip(d: &D3d, bpe: usize, side: usize) -> Option<(Computer, Vec<u8>, Vec<Vec<u8>>)> {
    let c = Computer::create(d).ok()?;
    let r = Record(c.record());
    let mips = r.mips();
    let mut memory = vec![0u8; c.size() as usize];
    let mut inputs = Vec::new();
    let what = format!("mode {} format {} {}x{} array {} mips {mips}", d.layout & 0x1F, d.format, d.width, d.height, d.depth);
    for slice in 0..u32::from(d.depth) {
        for mip in 0..mips {
            let (w, h) = (elements(d.width as usize, side, mip), elements(d.height as usize, side, mip));
            let pitch = w * bpe;
            let input = pattern(pitch * h, inputs.len() + 1);
            assert_eq!(c.copy_into(&mut memory, mip + slice * mips, &input, pitch as u32, 0), OK, "{what}");
            inputs.push(input);
        }
    }
    // Read back after every subresource was written: a byte two subresources share would differ here.
    let mut k = 0;
    for slice in 0..u32::from(d.depth) {
        for mip in 0..mips {
            let input = &inputs[k];
            let w = elements(d.width as usize, side, mip);
            let mut back = vec![0u8; input.len()];
            assert_eq!(c.copy_from(&mut back, mip + slice * mips, &memory, (w * bpe) as u32, 0), OK, "{what}");
            assert!(back == *input, "{what}: slice {slice} mip {mip} differs after the round trip");
            k += 1;
        }
    }
    Some((c, memory, inputs))
}

#[test]
fn every_mode_size_chain_and_array_survives_the_round_trip() {
    let _s = serial();
    let mut done = 0;
    for v4 in [false, true] {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });
        for mode in MODES {
            for (format, bpe, side) in FORMATS {
                for (w, h) in SIZES {
                    // Arrays and cubes on the small sizes, one chain on the middle ones and one level of the largest keep a debug build fast.
                    let shapes: &[(u16, u16)] = match w * h {
                        0..=0x1000 => &[(1, 1), (1, 0), (6, 0), (3, 4)],
                        0x1001..=0x10000 => &[(1, 0)],
                        _ if v4 => &[],
                        _ => &[(1, 1)],
                    };
                    for &(array, mips) in shapes {
                        if round_trip(&tex(w, h, array, mips, format, mode), bpe, side).is_some() {
                            done += 1;
                        }
                    }
                }
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(done > 2000, "only {done} descriptions were accepted");
}

#[test]
fn tiled_bytes_match_the_core_detiler() {
    let _s = serial();
    // Where the specification says the core tables are the console's: the modes without exclusive or (RB+ for the D
    // modes 2, 6 and 10), and the RB+ X modes of hardware version 4 with 32 pipes.
    let cases: [(u32, bool, Layout); 12] = [
        (1, false, Layout::default()),
        (2, false, Layout { rb_plus: true, ..Layout::default() }),
        (5, false, Layout::default()),
        (6, false, Layout { rb_plus: true, ..Layout::default() }),
        (9, false, Layout::default()),
        (10, false, Layout { rb_plus: true, ..Layout::default() }),
        (21, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
        (22, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
        (24, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
        (25, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
        (26, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
        (27, true, Layout { rb_plus: true, pipes_log2: 5, ..Layout::default() }),
    ];
    let mut compared = 0;
    for (mode, v4, layout) in cases {
        XGSetHardwareVersion(if v4 { 4 } else { 0 });
        let core = core_mode(mode).unwrap();
        for (format, bpe, _) in FORMATS.into_iter().filter(|f| f.2 == 1) {
            for (w, h) in SIZES {
                let Some((_, memory, inputs)) = round_trip(&tex(w, h, 1, 1, format, mode), bpe, 1) else { continue };
                // A level that fits the tail sits at its tail position, which the single level detiler does not model.
                let c = Computer::create(&tex(w, h, 1, 1, format, mode)).unwrap();
                if Record(c.record()).in_tail(0) {
                    continue;
                }
                let detiled = detile_with(core, &layout, &memory, w as usize, h as usize, bpe).unwrap();
                assert!(detiled == inputs[0], "mode {mode} bpe {bpe} {w}x{h}");
                compared += 1;
            }
        }
    }
    XGSetHardwareVersion(0);
    assert!(compared > 300, "only {compared} compared");
}

#[test]
fn resource_tiling_through_the_table() {
    let _s = serial();
    // Vector G of 1.13: a cube of 512 x 512, 8 byte elements, full chain, 64KB_S. Tail tile first, levels 2, 1, 0 after it.
    let c = Computer::create(&tex(512, 512, 6, 0, 10, 9)).unwrap();
    let (hr, tiles, packed, shape, subs) = c.tiling(0, 16);
    assert_eq!(hr, OK);
    assert_eq!(tiles, (0x1020000 / 0x10000) as u32);
    assert_eq!(packed, PackedMipInfo { standard_mips: 3, packed_mips: 7, tiles_for_packed_mips: 1, start_tile: 0 });
    assert_eq!(shape, XgTileShape { width: 128, height: 64, depth: 1 });
    let t = |w, h, start| SubresourceTiling { width_in_tiles: w, height_in_tiles: h, depth_in_tiles: 1, start_tile: start };
    assert_eq!(subs, [t(4, 8, 11), t(2, 4, 3), t(1, 2, 1), SubresourceTiling::default(), SubresourceTiling::default(), SubresourceTiling::default()]);
    // A window from the second entry, and a room smaller than what is left.
    let (_, _, _, _, window) = c.tiling(1, 1);
    assert_eq!(window, [t(2, 4, 3)]);
    let (_, _, _, _, past) = c.tiling(40, 4);
    assert!(past.is_empty());
    // Each standard level starts where the element offsets say.
    for (mip, s) in subs[..3].iter().enumerate() {
        assert_eq!(u64::from(s.start_tile) * 0x10000, c.element(mip as u32, 0, 0, 0));
    }

    // No tail: every slice is listed and the slices follow each other.
    let c = Computer::create(&tex(256, 256, 2, 1, 28, 27)).unwrap();
    let (hr, tiles, packed, shape, subs) = c.tiling(0, 8);
    assert_eq!(hr, OK);
    assert_eq!((tiles, packed.standard_mips, packed.packed_mips, packed.tiles_for_packed_mips), (8, 1, 0, 0));
    assert_eq!(shape, XgTileShape { width: 128, height: 128, depth: 1 });
    assert_eq!(subs, [t(2, 2, 0), t(2, 2, 4)]);

    // Block compressed: the shape is in texels.
    let c = Computer::create(&tex(1024, 1024, 1, 1, 71, 9)).unwrap();
    assert_eq!(c.tiling(0, 1).3, XgTileShape { width: 512, height: 256, depth: 1 });
    // Null outputs are skipped.
    assert_eq!(c.tiling_null(), OK);

    // Linear and 4 KiB modes fail, thick volumes are not implemented.
    for layout in [1u32, 0x105] {
        let c = Computer::create(&D3d { layout, ..tex(64, 64, 1, 1, 28, 0) }).unwrap();
        assert_eq!(c.tiling_null(), E_FAIL);
    }
    let v = Computer::create(&D3d { dimension: 4, width: 64, height: 64, depth: 64, mips: 1, format: 28, samples: 1, layout: 0x109, flags: 0 }).unwrap();
    assert_eq!(v.tiling_null(), NOTIMPL);
}
