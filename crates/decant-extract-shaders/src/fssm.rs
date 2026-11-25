//! The GPU code inside an FSSM block. In FSSM 1.1 the code's position and size are at 0xC0 and
//! 0xC8. A 16 byte sub header precedes the code, and `s_code_end` words pad it at the end.

/// The instruction that pads the end of the code.
pub const CODE_END: u32 = 0xBF9F_0000;

/// The code of one shader, ready for the pack.
#[derive(Debug, PartialEq, Eq)]
pub struct Code {
    pub sub_header: [u8; 16],
    /// Instruction words without the trailing padding.
    pub words: Vec<u32>,
    /// How many padding words were removed.
    pub padding: usize,
}

/// Why an FSSM block gives no code. The checks and their order follow `gpu_code` in
/// `extract_shaders.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Shorter than the 0xD0 byte header.
    ShortHeader,
    /// The first four bytes are not `FSSM`.
    NotFssm,
    /// The code size is not a whole number of words.
    PartialWord,
    /// The code would start inside the header.
    InsideHeader,
    /// The sub header and code run past the end of the block.
    PastEnd,
}

/// Splits an FSSM block, or says which bound does not hold.
pub fn split(fssm: &[u8]) -> Result<Code, Refusal> {
    if fssm.len() < 0xD0 {
        return Err(Refusal::ShortHeader);
    }
    if &fssm[..4] != b"FSSM" {
        return Err(Refusal::NotFssm);
    }
    let read = |at: usize| u64::from_le_bytes(fssm[at..at + 8].try_into().unwrap());
    let (offset, size) = (read(0xC0), read(0xC8));
    if size % 4 != 0 {
        return Err(Refusal::PartialWord);
    }
    let end = offset.checked_add(16).and_then(|e| e.checked_add(size)).ok_or(Refusal::PastEnd)?;
    if end > fssm.len() as u64 {
        return Err(Refusal::PastEnd);
    }
    if offset < 0xD0 {
        return Err(Refusal::InsideHeader);
    }
    let (offset, size) = (offset as usize, size as usize);
    let sub_header: [u8; 16] = fssm[offset..offset + 16].try_into().unwrap();
    let mut words: Vec<u32> = fssm[offset + 16..offset + 16 + size].as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    let mut padding = 0;
    while words.last() == Some(&CODE_END) {
        words.pop();
        padding += 1;
    }
    Ok(Code { sub_header, words, padding })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(words: &[u32]) -> Vec<u8> {
        let mut fssm = b"FSSM".to_vec();
        fssm.resize(0xD0, 0);
        fssm[0xC0..0xC8].copy_from_slice(&0xD0u64.to_le_bytes());
        fssm[0xC8..0xD0].copy_from_slice(&((words.len() * 4) as u64).to_le_bytes());
        fssm.extend(1..=16u8);
        fssm.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        fssm
    }

    #[test]
    fn padding_is_stripped_and_counted() {
        let code = split(&block(&[7, 8, CODE_END, CODE_END])).unwrap();
        assert_eq!(code.words, [7, 8]);
        assert_eq!(code.padding, 2);
        assert_eq!(code.sub_header[0], 1);
    }

    /// `block` with the code position and size fields overwritten.
    fn with_bounds(words: &[u32], offset: u64, size: u64) -> Vec<u8> {
        let mut fssm = block(words);
        fssm[0xC0..0xC8].copy_from_slice(&offset.to_le_bytes());
        fssm[0xC8..0xD0].copy_from_slice(&size.to_le_bytes());
        fssm
    }

    /// The fixture table: every input is invented, the expected outcome is either the number of
    /// code words and padding words, or the bound that refuses the block.
    #[test]
    fn fixture_table_separates_well_formed_truncated_and_foreign_blocks() {
        let whole = block(&[7, 8, CODE_END]);
        let mut foreign = whole.clone();
        foreign[..4].copy_from_slice(b"FSSN");
        /// Name, block, and code words with padding words or the refusal.
        type Case = (&'static str, Vec<u8>, Result<(usize, usize), Refusal>);
        let cases: Vec<Case> = vec![
            ("well formed, one padding word", whole.clone(), Ok((2, 1))),
            ("well formed, no code at all", block(&[]), Ok((0, 0))),
            ("well formed, only padding", block(&[CODE_END, CODE_END]), Ok((0, 2))),
            ("well formed, code after a gap", with_bounds(&[0, 0, 0, 0, 5], 0xE0, 4), Ok((1, 0))),
            ("empty", Vec::new(), Err(Refusal::ShortHeader)),
            ("magic only", b"FSSM".to_vec(), Err(Refusal::ShortHeader)),
            ("header cut one byte short", whole[..0xCF].to_vec(), Err(Refusal::ShortHeader)),
            ("header without sub header", whole[..0xD0].to_vec(), Err(Refusal::PastEnd)),
            ("sub header cut", whole[..0xD8].to_vec(), Err(Refusal::PastEnd)),
            ("last code word cut", whole[..whole.len() - 1].to_vec(), Err(Refusal::PastEnd)),
            ("other magic", foreign, Err(Refusal::NotFssm)),
            ("size not whole words", with_bounds(&[1], 0xD0, 3), Err(Refusal::PartialWord)),
            ("code starts in the header", with_bounds(&[1], 0xC0, 4), Err(Refusal::InsideHeader)),
            ("position past the end", with_bounds(&[1], 0x1000, 4), Err(Refusal::PastEnd)),
            ("position and size overflow", with_bounds(&[1], u64::MAX - 8, 8), Err(Refusal::PastEnd)),
        ];
        for (name, fssm, expected) in cases {
            let got = split(&fssm).map(|c| (c.words.len(), c.padding));
            assert_eq!(got, expected, "{name}");
        }
    }

    /// Python checks the size, then the end, then the start, in one expression; the first that
    /// fails here is the one reported, and any failure refuses the block as in Python.
    #[test]
    fn a_block_breaking_several_bounds_reports_the_first_in_python_order() {
        assert_eq!(split(&with_bounds(&[1], 0x10, 3)).unwrap_err(), Refusal::PartialWord);
        assert_eq!(split(&with_bounds(&[1], 0x10, 0x1000)).unwrap_err(), Refusal::PastEnd);
    }

    /// A zlib stream made of one stored block, so the test needs no compressor.
    fn stored_zlib(data: &[u8]) -> Vec<u8> {
        let len = data.len() as u16;
        let mut out = vec![0x78, 0x01, 0x01];
        out.extend(len.to_le_bytes());
        out.extend((!len).to_le_bytes());
        out.extend(data);
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in data {
            a = (a + u32::from(byte)) % 65521;
            b = (b + a) % 65521;
        }
        out.extend(((b << 16) | a).to_be_bytes());
        out
    }

    /// An invented DXBC container: one DXIL part whose `XDAT` table holds one `XBCH` per block.
    fn container(blocks: &[Vec<u8>]) -> Vec<u8> {
        let streams: Vec<Vec<u8>> = blocks.iter().map(|b| stored_zlib(b)).collect();
        let mut body = b"XDAT".to_vec();
        body.extend((blocks.len() as u32).to_le_bytes());
        body.extend([0u8; 4]);
        let mut at = 12 + 12 * blocks.len();
        for s in &streams {
            for v in [0x0D, at as u32, (24 + s.len()) as u32] {
                body.extend(v.to_le_bytes());
            }
            at += 24 + s.len();
        }
        for s in &streams {
            body.extend(b"XBCH\0\0\0\0");
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

    /// Two programs in one container, as the voxelisation vertex shaders have: the first writes
    /// to LDS and jumps to the geometry shader, the second exports its outputs. The pack gets
    /// the second; a container with one program still gives that program.
    #[test]
    fn the_pack_gets_the_standalone_program_of_a_two_program_container() {
        use crate::container::{S_ENDPGM, gpu_code};
        let jump = block(&[7, 0xBE80_2018, S_ENDPGM, CODE_END]);
        let alone = block(&[8, 9, S_ENDPGM, CODE_END]);
        let words = |blob: &[u8]| split(&gpu_code(blob).unwrap()).unwrap().words;
        assert_eq!(words(&container(&[jump.clone(), alone])), [8, 9, S_ENDPGM]);
        assert_eq!(words(&container(&[jump])), [7, 0xBE80_2018, S_ENDPGM]);
    }
}
