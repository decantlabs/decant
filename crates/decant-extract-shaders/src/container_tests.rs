use super::*;
use crate::test_util::dxbc;

#[test]
fn malformed_containers_give_nothing() {
    assert!(chunks(b"DXBC").is_empty());
    let mut blob = dxbc(0, b"x");
    blob[32] = 0xff;
    assert!(chunks(&blob).is_empty(), "part offset outside the blob");
    assert_eq!(kind(b"nope"), ("?".to_owned(), 0, 0));
    assert!(gpu_code(b"DXBC").is_none());
}

/// Where the parts of `dxbc(_, gpu)` sit: the DXIL body starts at 44, `XDAT` at 8 in the body,
/// the `XBCH` part at 40 in the body, its zlib stream at 64.
const BODY: usize = 44;
const XDAT: usize = BODY + 8;
const XBCH: usize = BODY + 40;
const STREAM: usize = BODY + 64;

/// `blob` with a little endian u32 written at `at`.
fn put(mut blob: Vec<u8>, at: usize, value: u32) -> Vec<u8> {
    blob[at..at + 4].copy_from_slice(&value.to_le_bytes());
    blob
}

/// `blob` with its DXIL part size set to `size`, so the part ends early.
fn dxil_size(blob: Vec<u8>, size: usize) -> Vec<u8> {
    put(blob, 40, size as u32)
}

/// The fixture table: each invented container, and where the walk from DXBC to FSSM stops.
#[test]
fn fixture_table_separates_missing_truncated_and_well_formed_gpu_code() {
    let gpu = b"FSSM invented code";
    let whole = dxbc(0, gpu);
    let mut no_xdat = whole.clone();
    no_xdat[XDAT..XDAT + 4].copy_from_slice(b"XDAX");
    let mut no_xbch = whole.clone();
    no_xbch[XBCH..XBCH + 4].copy_from_slice(b"XBCX");
    let mut damaged = whole.clone();
    damaged[STREAM + 2] ^= 0xFF;
    let mut other_part = whole.clone();
    other_part[36..40].copy_from_slice(b"SHEX");
    type Case<'a> = (&'a str, Vec<u8>, Result<&'a [u8], GpuGap>);
    let cases: Vec<Case> = vec![
        ("well formed", whole.clone(), Ok(gpu)),
        ("empty", Vec::new(), Err(GpuGap::NotDxbc)),
        ("container header cut", whole[..31].to_vec(), Err(GpuGap::NotDxbc)),
        ("part table cut", whole[..35].to_vec(), Err(GpuGap::NotDxbc)),
        ("no DXIL part", other_part, Err(GpuGap::NoDxil)),
        ("DXIL body empty", dxil_size(whole.clone(), 0), Err(GpuGap::NoXdat)),
        ("no XDAT table", no_xdat, Err(GpuGap::NoXdat)),
        ("XDAT header cut", dxil_size(whole.clone(), 8 + 11), Err(GpuGap::XdatCut)),
        ("XDAT entry cut", dxil_size(whole.clone(), 8 + 23), Err(GpuGap::XdatCut)),
        ("XDAT count zero", put(whole.clone(), XDAT + 4, 0), Err(GpuGap::NoXbch)),
        ("entry of another kind", put(whole.clone(), XDAT + 12, 0x0C), Err(GpuGap::NoXbch)),
        ("entry not pointing at XBCH", no_xbch, Err(GpuGap::NoXbch)),
        ("XBCH header cut", dxil_size(whole.clone(), 40 + 23), Err(GpuGap::XbchCut)),
        ("zlib stream cut", whole[..whole.len() - 3].to_vec(), Err(GpuGap::Inflate)),
        ("zlib stream damaged", damaged, Err(GpuGap::Inflate)),
    ];
    for (name, blob, expected) in cases {
        assert_eq!(find_gpu_code(&blob).as_deref().map_err(|gap| *gap), expected, "{name}");
        assert_eq!(gpu_code(&blob).is_some(), expected.is_ok(), "{name}: gpu_code disagrees");
    }
}

/// A DXBC container with two parts: `first` as the first DXIL body, then a well formed one.
fn two_dxil(first: &[u8], gpu: &[u8]) -> Vec<u8> {
    let second = dxbc(0, gpu);
    let second_body = &second[BODY..];
    let mut blob = b"DXBC".to_vec();
    blob.extend([0u8; 24]);
    blob.extend(2u32.to_le_bytes());
    let first_at = 40u32;
    let second_at = first_at + 8 + first.len() as u32;
    blob.extend(first_at.to_le_bytes());
    blob.extend(second_at.to_le_bytes());
    for body in [first, second_body] {
        blob.extend(b"DXIL");
        blob.extend((body.len() as u32).to_le_bytes());
        blob.extend(body);
    }
    blob
}

#[test]
fn a_later_dxil_part_is_searched_only_when_the_earlier_one_has_an_xdat_table() {
    // First part: an XDAT table with no XBCH entry. The walk moves on to the next DXIL part.
    let mut table = 0u32.to_le_bytes().to_vec();
    table.extend([0u8; 4]);
    table.extend(b"XDAT");
    table.extend([0u8; 8]);
    assert_eq!(find_gpu_code(&two_dxil(&table, b"second")).unwrap(), b"second");
    // First part: no XDAT table at all. The walk stops there.
    assert_eq!(find_gpu_code(&two_dxil(&[0u8; 16], b"second")), Err(GpuGap::NoXdat));
}

/// An invented FSSM 1.1 block: code at 0xD0 after a 16 byte sub header, then one padding word.
fn fssm(words: &[u32]) -> Vec<u8> {
    let mut block = b"FSSM".to_vec();
    block.resize(0xD0, 0);
    block[0xC0..0xC8].copy_from_slice(&0xD0u64.to_le_bytes());
    block[0xC8..0xD0].copy_from_slice(&((words.len() * 4 + 4) as u64).to_le_bytes());
    block.extend([0u8; 16]);
    block.extend(words.iter().chain(&[CODE_END]).flat_map(|w| w.to_le_bytes()));
    block
}

/// A DXBC container whose one DXIL part has an `XDAT` table with one `XBCH` part per program.
fn programs(blocks: &[Vec<u8>]) -> Vec<u8> {
    let streams: Vec<Vec<u8>> = blocks.iter().map(|b| crate::test_util::zlib(b)).collect();
    let mut body = vec![0u8; 8];
    let x = body.len();
    let table_end = x + 12 + 12 * blocks.len();
    body.extend(b"XDAT");
    body.extend((blocks.len() as u32).to_le_bytes());
    body.extend([0u8; 4]);
    let mut at = table_end;
    for s in &streams {
        body.extend(0x0Du32.to_le_bytes());
        body.extend((at as u32).to_le_bytes());
        body.extend(((24 + s.len()) as u32).to_le_bytes());
        at += 24 + s.len();
    }
    for s in &streams {
        body.extend(b"XBCH");
        body.extend([0u8; 4]);
        body.extend((s.len() as u64).to_le_bytes());
        body.extend([0u8; 8]);
        body.extend(s);
    }
    let mut blob = b"DXBC".to_vec();
    blob.extend([0u8; 24]);
    blob.extend(1u32.to_le_bytes());
    blob.extend(36u32.to_le_bytes());
    blob.extend(b"DXIL");
    blob.extend((body.len() as u32).to_le_bytes());
    blob.extend(body);
    blob
}

#[test]
fn a_standalone_program_is_recognised_from_its_last_word_and_its_jumps() {
    assert!(is_setpc_b64(0xBE80_2018), "s_setpc_b64 s[24:25]");
    assert!(!is_setpc_b64(0xBE80_2118), "s_swappc_b64 is a call, not a jump away");
    assert!(fssm_standalone(&fssm(&[1, S_ENDPGM])));
    assert!(!fssm_standalone(&fssm(&[0xBE80_2018, S_ENDPGM])), "jumps before s_endpgm");
    assert!(!fssm_standalone(&fssm(&[1, 2])), "does not end with s_endpgm");
    assert!(!fssm_standalone(&fssm(&[])), "no code");
    assert!(!fssm_standalone(b"FSSM"), "cut header");
}

/// The fixture table: which program a container with several `XBCH` parts gives.
#[test]
fn with_several_programs_the_standalone_one_is_chosen() {
    let jump = fssm(&[7, 0xBE80_2018, S_ENDPGM]);
    let alone = fssm(&[8, 9, S_ENDPGM]);
    let other = fssm(&[10, S_ENDPGM]);
    let open = fssm(&[11, 12]);
    /// Name, the programs in container order, and the program expected.
    type Case<'a> = (&'a str, Vec<Vec<u8>>, &'a Vec<u8>);
    let cases: Vec<Case> = vec![
        ("one jumping program is kept as before", vec![jump.clone()], &jump),
        ("jumping, then standalone", vec![jump.clone(), alone.clone()], &alone),
        ("standalone first stays first", vec![alone.clone(), other.clone()], &alone),
        ("no standalone program keeps the first", vec![jump.clone(), open.clone()], &jump),
        ("the first standalone after the head", vec![jump.clone(), open.clone(), alone.clone(), other.clone()], &alone),
    ];
    for (name, blocks, expected) in cases {
        assert_eq!(&find_gpu_code(&programs(&blocks)).unwrap(), expected, "{name}");
    }
}
