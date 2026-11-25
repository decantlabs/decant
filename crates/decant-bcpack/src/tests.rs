//! Tests on streams made by the test encoder (`encode`): no game data.
#![allow(clippy::needless_range_loop)] // set and pixel numbers index parallel arrays

use crate::encode::{self, Bc6Values, Bc7Values, Fragment};
use crate::lanes::{Lanes, SliceFeed};
use crate::{Error, Format, decode, decode_to_vec};

/// xorshift64, deterministic.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

/// Random records that fit every lane's alphabet. Small values are more likely, as in real data.
fn random_records(format: Format, blocks: usize, random: &mut Random) -> Vec<u8> {
    let layout = format.layout();
    let mut records = vec![0u8; blocks * layout.stride()];
    for record in records.chunks_exact_mut(layout.stride()) {
        for (k, symbol) in record.iter_mut().enumerate() {
            let bits = u32::from(layout.groups[k / layout.per]) + 1;
            let wide = if random.below(4) == 0 { bits } else { bits.min(3) };
            *symbol = random.below(1 << wide) as u8;
        }
    }
    records
}

fn decode_records(format: Format, stream: &[u8], blocks: usize) -> Result<Vec<u8>, Error> {
    let mut records = vec![0u8; blocks * format.layout().stride()];
    Lanes::new(format.layout()).run(&mut SliceFeed::new(stream), &mut records)?;
    Ok(records)
}

#[test]
fn every_layout_gives_back_its_records() {
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    for format in Format::ALL {
        for (blocks, run) in [(1, 1), (2, 3), (17, 5), (300, 64), (1500, 4000), (2500, 700)] {
            let records = random_records(format, blocks, &mut random);
            let stream = encode::stream(format.layout(), &records, run);
            let got = decode_records(format, &stream, blocks).unwrap_or_else(|e| panic!("{format:?} {blocks} blocks: {e}"));
            assert_eq!(got, records, "{format:?}, {blocks} blocks, fragments of {run}");
        }
    }
}

#[test]
fn long_fragments_and_every_shape_decode() {
    // One lane per shape and narrowing, BC4's layout (12 lanes of 1 symbol per block).
    let layout = Format::Bc4.layout();
    let mut random = Random(77);
    let blocks = 3000;
    for shape in 0..16u8 {
        let lanes: Vec<Vec<u8>> = layout
            .groups
            .iter()
            .enumerate()
            .map(|(j, &group)| {
                let narrow = (j % 4) as u8;
                let bits = group - narrow + 1;
                let symbols: Vec<u8> = (0..blocks).map(|_| random.below(1 << bits) as u8).collect();
                encode::lane_bytes(group, &[Fragment { shape, narrow, symbols }])
            })
            .collect();
        let stream = encode::interleave(layout, lanes, blocks);
        assert!(decode_records(Format::Bc4, &stream, blocks).is_ok(), "shape {shape}");
    }
}

#[test]
fn a_short_stream_is_truncated_and_noise_does_not_panic() {
    let mut random = Random(5);
    let records = random_records(Format::Bc1, 400, &mut random);
    let stream = encode::stream(Format::Bc1.layout(), &records, 100);
    let mut out = vec![0u8; 400 * 8];
    assert!(decode(Format::Bc1, &stream, &mut out).is_ok());
    assert_eq!(decode(Format::Bc1, &stream[..stream.len() / 2], &mut out), Err(Error::Truncated));
    assert_eq!(decode(Format::Bc1, &[], &mut out), Err(Error::Truncated));
    // More blocks than were coded: the lanes run dry.
    assert_eq!(decode(Format::Bc1, &stream, &mut vec![0u8; 401 * 8]), Err(Error::Truncated));
    assert_eq!(decode(Format::Bc1, &stream, &mut out[..12]), Err(Error::Size));
    assert_eq!(decode(Format::Bc7, &stream, &mut []), Err(Error::Size));
    for round in 0..200 {
        let src = random.bytes(round * 13 % 1000);
        for format in Format::ALL {
            let _ = decode(format, &src, &mut vec![0u8; 16 * (1 + round % 50)]);
        }
    }
}

#[test]
fn modes_map_to_formats() {
    for format in Format::ALL {
        assert_eq!(Format::from_mode(format.mode()), Some(format));
    }
    for mode in [0, 2, 8, 15] {
        assert_eq!(Format::from_mode(mode), None);
    }
}

/// Random BCn blocks of a plain format, packed against no reference and decoded back.
#[test]
fn plain_blocks_come_back_byte_for_byte() {
    let mut random = Random(0xB10C);
    for format in [Format::Bc1, Format::Bc3, Format::Bc4, Format::Bc5] {
        for blocks in [1, 9, 1000] {
            let want = random.bytes(blocks * format.block_bytes());
            let stream = encode::pack_plain(format, &want, 333);
            let (got, report) = decode_to_vec(format, &stream, want.len()).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            assert_eq!(got, want, "{format:?}, {blocks} blocks");
            assert_eq!(report.taken, stream.len());
        }
    }
}

/// References by hand: a block that repeats an earlier one with no residual, and a first block that
/// reads seed entry 1 (reference 1 before block 0).
#[test]
fn references_reach_earlier_blocks_and_the_seeds() {
    // BC4 records: [endpoint reference offset, index reference, endpoint 0, endpoint 1, 8 pairs].
    // Index residuals go through the rank table: `none` is the pair symbol of residual 0.
    let none = crate::tables::RANK_BC4.iter().position(|&v| v == 0).unwrap() as u8;
    let mut block0 = vec![0u8, 0, 20, 7];
    block0.extend([9, 18, 27, 36, 45, 54, 63, 1]);
    // Index reference 1 and offset 0: both references read block 0.
    let mut block1 = vec![0u8, 1, 0, 0];
    block1.extend([none; 8]);
    let stream = encode::stream(Format::Bc4.layout(), &[block0.clone(), block1].concat(), 10);
    let (got, _) = decode_to_vec(Format::Bc4, &stream, 16).unwrap();
    assert_eq!(got[..8], got[8..], "a reference with no residual repeats the block");
    // Block 0 reading seed entry 1 for both, no residual.
    let mut seed = vec![0u8, 1, 0, 0];
    seed.extend([none; 8]);
    let stream = encode::stream(Format::Bc4.layout(), &seed, 10);
    let (got, _) = decode_to_vec(Format::Bc4, &stream, 8).unwrap();
    let (ep, ix) = crate::tables::SEED_BC4[0];
    let mut want = [0u8; 8];
    let mut values = crate::plain::Values::default();
    values.ep[..2].copy_from_slice(&ep);
    values.ix[..16].copy_from_slice(&ix);
    crate::plain::pack(&crate::plain::BC4, &values, &mut want);
    assert_eq!(got, want);
    // An endpoint offset: offset zigzag 1 (-1) from index reference 2 reads block 1 for the
    // endpoints and block 0 for the indices.
    let mut block2 = vec![1u8, 2, 0, 0];
    block2.extend([none; 8]);
    let mut other = vec![0u8, 0, 99, 3];
    other.extend([5; 8]);
    let stream = encode::stream(Format::Bc4.layout(), &[block0, other, block2].concat(), 10);
    let (got, _) = decode_to_vec(Format::Bc4, &stream, 24).unwrap();
    assert_eq!(got[16..18], got[8..10], "endpoints from block 1");
    assert_eq!(got[18..24], got[2..8], "indices from block 0");
}

/// Random BC7 values that a block can hold: fields at their precision, a shared p-bit equal in
/// both halves, anchors with their top bit clear.
fn random_bc7(random: &mut Random, controls: &[u16]) -> Bc7Values {
    let word = controls[random.below(controls.len() as u64) as usize];
    let (precision, index_bits) = encode::bc7_precision(word);
    let subsets = usize::from((word >> 6) & 3);
    let partition = usize::from((word >> 8) & 63);
    let mut v = Bc7Values { control: word, index_bits, ..Bc7Values::default() };
    for subset in 0..subsets {
        for (c, &p) in precision.iter().enumerate() {
            if p == 0 {
                continue;
            }
            for e in 0..2 {
                v.ends[subset][c][e] = (random.below(1 << p) as u16) << (16 - p);
            }
        }
        if word & 7 == 1 {
            // Mode 1 shares one p-bit per subset.
            let bit = v.ends[subset][0][0] & (1 << (16 - precision[0]));
            v.ends[subset][0][1] = (v.ends[subset][0][1] & !(1 << (16 - precision[0]))) | bit;
        }
    }
    let anchors: Vec<usize> = match subsets {
        2 => vec![0, usize::from(crate::bc7::ANCHOR_2[partition])],
        3 => vec![0, usize::from(crate::bc7::ANCHOR_3[0][partition]), usize::from(crate::bc7::ANCHOR_3[1][partition])],
        _ => vec![0],
    };
    for set in 0..2 {
        let bits = index_bits[set];
        if bits == 0 {
            continue;
        }
        for p in 0..16 {
            let top = if anchors.contains(&p) || p == 0 { bits - 1 } else { bits };
            let value = random.below(1 << top) as i16;
            // Kept as the decoder keeps it: sign extended from `bits`.
            v.index[set][p] = (value << (16 - bits)) >> (16 - bits);
        }
    }
    v
}

/// Reads a BC7 block the way the public format describes it, independently of the packer.
fn read_bc7(block: &[u8; 16]) -> (usize, Vec<u64>) {
    let bits = u128::from_le_bytes(*block);
    let mut at = 0;
    let mut take = |n: u32| {
        let v = ((bits >> at) & ((1u128 << n) - 1)) as u64;
        at += n;
        v
    };
    let mut mode = 0;
    while take(1) == 0 {
        mode += 1;
    }
    let (subsets, partition_bits, rotation_bits, selection_bits, colour, alpha, pbits, i1, i2) = [
        (3, 4, 0, 0, 4, 0, 6, 3, 0),
        (2, 6, 0, 0, 6, 0, 2, 3, 0),
        (3, 6, 0, 0, 5, 0, 0, 2, 0),
        (2, 6, 0, 0, 7, 0, 4, 2, 0),
        (1, 0, 2, 1, 5, 6, 0, 2, 3),
        (1, 0, 2, 0, 7, 8, 0, 2, 2),
        (1, 0, 0, 0, 7, 7, 2, 4, 0),
        (2, 6, 0, 0, 5, 5, 4, 2, 0),
    ][mode];
    let mut fields = vec![take(partition_bits), take(rotation_bits), take(selection_bits)];
    for channel in 0..4 {
        let width = if channel < 3 { colour } else { alpha };
        for _ in 0..2 * subsets {
            fields.push(take(width));
        }
    }
    for _ in 0..pbits {
        fields.push(take(1));
    }
    let partition = fields[0] as usize;
    let anchors: Vec<usize> = match subsets {
        2 => vec![0, usize::from(crate::bc7::ANCHOR_2[partition])],
        3 => vec![0, usize::from(crate::bc7::ANCHOR_3[0][partition]), usize::from(crate::bc7::ANCHOR_3[1][partition])],
        _ => vec![0],
    };
    for width in [i1, i2] {
        for p in 0..16 {
            if width > 0 {
                fields.push(take(width - u32::from(anchors.contains(&p))));
            }
        }
    }
    assert_eq!(at, 128, "mode {mode} fills the block");
    (mode, fields)
}

#[test]
fn bc7_blocks_follow_the_public_layout_and_come_back() {
    let controls = encode::bc7_controls();
    assert!(controls.len() > 200);
    let mut random = Random(0x0BC7);
    let values: Vec<Bc7Values> = (0..3000).map(|_| random_bc7(&mut random, &controls)).collect();
    let (stream, want) = encode::pack_bc7(&values, 900).expect("every control word has a code");
    for (v, block) in values.iter().zip(want.as_chunks::<16>().0) {
        let (mode, fields) = read_bc7(block);
        assert_eq!(mode, usize::from(v.control & 7));
        let (precision, _) = encode::bc7_precision(v.control);
        let subsets = usize::from((v.control >> 6) & 3);
        // The first endpoint field: red of subset 0, endpoint 0 (alpha's place for rotation 1).
        let rotation = usize::from((v.control >> 3) & 3);
        let first = if rotation == 1 { 3 } else { 0 };
        let pbit = usize::from(matches!(mode, 0 | 1 | 3 | 6 | 7) && first == 0);
        assert_eq!(fields[3], u64::from(v.ends[0][first][0] >> (16 - precision[first])) >> pbit, "mode {mode}");
        if mode < 4 || mode == 7 {
            assert_eq!(fields[0], u64::from((v.control >> 8) & 63), "partition");
        }
        assert!(subsets >= 1);
    }
    let (got, report) = decode_to_vec(Format::Bc7, &stream, want.len()).unwrap();
    assert_eq!(got, want);
    assert_eq!(report.inconsistent, 0);
}

#[test]
fn bc6_layouts_place_every_bit_once() {
    for mode in 1..=14u8 {
        let (bits, index) = encode::bc6_bits(mode);
        let layout = &crate::bc6::layouts()[usize::from(mode)];
        let two = mode <= 10;
        let mut seen = std::collections::HashSet::new();
        let mut total = 0u32;
        for run in layout {
            total += u32::from(run.len);
            if run.channel == 3 {
                continue;
            }
            for b in run.first..run.first + run.len {
                assert!(seen.insert((run.end, run.channel, b)), "mode {mode}: bit {b} of {run:?} twice");
                let width = if run.end == 0 { bits[0] } else { bits[1 + usize::from(run.channel)] };
                assert!(b < width, "mode {mode}: bit {b} of {run:?} past {width}");
            }
        }
        let ends = if two { 4 } else { 2 };
        let want: u32 = (0..ends).map(|e| (0..3).map(|c| u32::from(if e == 0 { bits[0] } else { bits[1 + c] })).sum::<u32>()).sum();
        assert_eq!(seen.len() as u32, want, "mode {mode}: every endpoint bit");
        let code_bits = if mode <= 2 { 2 } else { 5 };
        let index_bits = 16 * u32::from(index) - if two { 2 } else { 1 };
        assert_eq!(code_bits + total + index_bits, 128, "mode {mode} fills the block");
    }
}

#[test]
fn bc6_blocks_come_back() {
    let mut random = Random(0xBC6);
    let values: Vec<(Bc6Values, u8)> = (0..2000)
        .map(|_| {
            let mode = 1 + random.below(14) as u8;
            let (bits, index_bits) = encode::bc6_bits(mode);
            let two = mode <= 10;
            let mut ends = [[0u16; 3]; 4];
            for (e, end) in ends.iter_mut().enumerate().take(if two { 4 } else { 2 }) {
                for (c, value) in end.iter_mut().enumerate() {
                    let b = if e == 0 { bits[0] } else { bits[1 + c] };
                    *value = (random.below(1 << b) as u16) << (16 - b);
                }
            }
            if !two {
                for c in 0..3 {
                    ends[2][c] = (ends[0][c] >> (16 - bits[0])) << (16 - bits[1 + c]);
                    ends[3][c] = ends[1][c];
                }
            }
            let partition = if two { random.below(32) as u8 } else { 0 };
            let anchor = if two { usize::from(crate::bc7::ANCHOR_2[usize::from(partition)]) } else { 0 };
            let index = std::array::from_fn(|p| {
                let top = if p == 0 || p == anchor { index_bits - 1 } else { index_bits };
                // Stored biased by half the range; kept signed.
                random.below(1 << top) as i16 - (1 << (index_bits - 1))
            });
            (Bc6Values { ends, mode, index }, partition)
        })
        .collect();
    let (stream, want) = encode::pack_bc6(&values, 800);
    let (got, report) = decode_to_vec(Format::Bc6h, &stream, want.len()).unwrap();
    assert_eq!(got, want);
    assert_eq!(report.inconsistent, 0);
}

/// MiB of blocks per second per format on synthetic streams:
/// `cargo test --release -p decant-bcpack throughput -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn throughput() {
    let mut random = Random(1);
    for format in Format::ALL {
        let blocks = 1 << 15;
        let records = random_records(format, blocks, &mut random);
        let mut records = records;
        if format == Format::Bc7 {
            for r in records.as_chunks_mut::<32>().0 {
                (r[30], r[31]) = (0, 0);
            }
        }
        if format == Format::Bc6h {
            for r in records.as_chunks_mut::<32>().0 {
                r[30] %= 14;
            }
        }
        let stream = encode::stream(format.layout(), &records, 20000);
        let mut out = vec![0u8; blocks * format.block_bytes()];
        let _ = decode(format, &stream, &mut out);
        let runs = 10;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            let _ = decode(format, std::hint::black_box(&stream), &mut out);
        }
        let seconds = started.elapsed().as_secs_f64() / f64::from(runs);
        let started = std::time::Instant::now();
        for _ in 0..runs {
            let _ = decode_records(format, std::hint::black_box(&stream), blocks);
        }
        let lanes = started.elapsed().as_secs_f64() / f64::from(runs);
        let mib = out.len() as f64 / (1024.0 * 1024.0);
        println!("{format:?}: {:.1} MiB/s out, the lanes alone {:.1} MiB/s", mib / seconds, mib / lanes);
    }
}
