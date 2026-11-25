//! The LLVM bitstream writer (BitCodeFormat, "Bitstream Format"): fixed fields, VBR fields, blocks
//! and unabbreviated records. Abbreviations are not used, every record is `UNABBREV_RECORD`.

/// Abbreviation ids that are fixed by the format.
const END_BLOCK: u64 = 0;
const ENTER_SUBBLOCK: u64 = 1;
const UNABBREV_RECORD: u64 = 3;

/// A bit writer that packs little endian 32 bit words, least significant bit first.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u64,
    nbits: u32,
    /// Open blocks: abbreviation width to restore, and byte offset of the length word.
    stack: Vec<(u32, usize)>,
    width: u32,
}

impl BitWriter {
    /// A writer at top level (abbreviation width 2) that already holds the magic `BC C0DE`.
    pub fn new() -> BitWriter {
        let mut w = BitWriter { width: 2, ..Default::default() };
        for b in [b'B', b'C', 0xC0, 0xDE] {
            w.fixed(b as u64, 8);
        }
        w
    }

    /// Writes the low `n` bits of `value`, `n <= 32`.
    pub fn fixed(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 32 && (n == 32 || value >> n == 0), "value {value} does not fit {n} bits");
        if n == 0 {
            return;
        }
        self.cur |= value << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.bytes.push(self.cur as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }

    /// Writes `value` as a VBR with chunks of `n` bits (`n - 1` payload bits and a continuation bit).
    pub fn vbr(&mut self, mut value: u64, n: u32) {
        let hi = 1u64 << (n - 1);
        while value >= hi {
            self.fixed((value & (hi - 1)) | hi, n);
            value >>= n - 1;
        }
        self.fixed(value, n);
    }

    /// Writes `value` as a signed VBR: the sign goes into the lowest bit (BitCodeFormat, "Signed VBRs").
    pub fn svbr(&mut self, value: i64, n: u32) {
        self.vbr(signed(value), n);
    }

    /// Pads with zero bits to the next 32 bit boundary.
    pub fn align32(&mut self) {
        if self.nbits > 0 {
            self.fixed(0, 8 - self.nbits);
        }
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
    }

    /// `ENTER_SUBBLOCK`: opens block `id` with the abbreviation width `width`.
    pub fn enter_block(&mut self, id: u64, width: u32) {
        self.fixed(ENTER_SUBBLOCK, self.width);
        self.vbr(id, 8);
        self.vbr(width as u64, 4);
        self.align32();
        let at = self.bytes.len();
        self.bytes.extend_from_slice(&[0; 4]);
        self.stack.push((self.width, at));
        self.width = width;
    }

    /// `END_BLOCK`: closes the innermost block and fills in its length in words.
    pub fn end_block(&mut self) {
        self.fixed(END_BLOCK, self.width);
        self.align32();
        let (outer, at) = self.stack.pop().expect("end_block without enter_block");
        let words = ((self.bytes.len() - at - 4) / 4) as u32;
        self.bytes[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = outer;
    }

    /// An unabbreviated record: code and operand count as VBR6, each operand as VBR6.
    pub fn record(&mut self, code: u32, ops: &[u64]) {
        self.fixed(UNABBREV_RECORD, self.width);
        self.vbr(code as u64, 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// The finished stream, padded to a multiple of 4 bytes. All blocks must be closed.
    pub fn finish(mut self) -> Vec<u8> {
        assert!(self.stack.is_empty(), "unclosed block");
        self.align32();
        self.bytes
    }
}

/// The signed VBR transform: `v << 1` for `v >= 0`, `(-v << 1) | 1` otherwise.
pub fn signed(value: i64) -> u64 {
    if value >= 0 { (value as u64) << 1 } else { ((value.wrapping_neg() as u64) << 1) | 1 }
}

/// The inverse of [`signed`].
pub fn unsigned_to_signed(v: u64) -> i64 {
    if v & 1 == 0 { (v >> 1) as i64 } else { (v >> 1).wrapping_neg() as i64 }
}

/// A reader of the same format, for tests: walks the stream back into blocks and records.
pub mod reader {
    /// One item of a decoded stream.
    #[derive(Debug, PartialEq, Eq, Clone)]
    pub enum Item {
        Block { id: u64, width: u32, items: Vec<Item> },
        Record { code: u32, ops: Vec<u64> },
    }

    struct Bits<'a> {
        data: &'a [u8],
        pos: usize,
    }

    impl Bits<'_> {
        fn fixed(&mut self, n: u32) -> Result<u64, String> {
            let mut v = 0u64;
            for i in 0..n {
                let byte = *self.data.get(self.pos / 8).ok_or("end of stream")?;
                v |= (((byte >> (self.pos % 8)) & 1) as u64) << i;
                self.pos += 1;
            }
            Ok(v)
        }
        fn vbr(&mut self, n: u32) -> Result<u64, String> {
            let hi = 1u64 << (n - 1);
            let (mut v, mut shift) = (0u64, 0);
            loop {
                let chunk = self.fixed(n)?;
                v |= (chunk & (hi - 1)) << shift;
                if chunk & hi == 0 {
                    return Ok(v);
                }
                shift += n - 1;
            }
        }
        fn align32(&mut self) {
            self.pos = self.pos.div_ceil(32) * 32;
        }
    }

    /// Decodes a stream written by [`super::BitWriter`]: magic, then top level blocks.
    pub fn read(data: &[u8]) -> Result<Vec<Item>, String> {
        if !data.len().is_multiple_of(4) || data.len() < 4 || data[..4] != [b'B', b'C', 0xC0, 0xDE] {
            return Err("bad magic or length".into());
        }
        let mut bits = Bits { data, pos: 32 };
        let mut out = Vec::new();
        while bits.pos < data.len() * 8 {
            let id = bits.fixed(2)?;
            if id != 1 {
                return Err(format!("top level abbreviation id {id}"));
            }
            out.push(block(&mut bits)?);
        }
        Ok(out)
    }

    fn block(bits: &mut Bits) -> Result<Item, String> {
        let id = bits.vbr(8)?;
        let width = bits.vbr(4)? as u32;
        bits.align32();
        let words = bits.fixed(32)? as usize;
        let start = bits.pos;
        let mut items = Vec::new();
        loop {
            match bits.fixed(width)? {
                0 => {
                    bits.align32();
                    break;
                }
                1 => items.push(block(bits)?),
                3 => {
                    let code = bits.vbr(6)? as u32;
                    let n = bits.vbr(6)?;
                    let ops = (0..n).map(|_| bits.vbr(6)).collect::<Result<Vec<_>, _>>()?;
                    items.push(Item::Record { code, ops });
                }
                other => return Err(format!("abbreviation id {other}")),
            }
        }
        if (bits.pos - start) / 32 != words {
            return Err(format!("block {id}: length word {words}, actual {}", (bits.pos - start) / 32));
        }
        Ok(Item::Block { id, width, items })
    }
}

#[cfg(test)]
mod tests {
    use super::reader::{Item, read};
    use super::*;

    #[test]
    fn vbr6_of_5_is_one_chunk_and_of_100_is_two() {
        let mut w = BitWriter::default();
        w.vbr(5, 6);
        w.align32();
        assert_eq!(w.bytes, [5, 0, 0, 0]);
        let mut w = BitWriter::default();
        w.vbr(100, 6);
        w.align32();
        // 100 = 0b11_00100: first chunk 0b100100 (payload 00100, continuation), second chunk 0b000011.
        assert_eq!(u32::from_le_bytes(w.bytes[..4].try_into().unwrap()), 0b000011_100100);
    }

    #[test]
    fn signed_vbr_matches_the_format() {
        assert_eq!(signed(-3), 7);
        assert_eq!(signed(2), 4);
        assert_eq!(signed(0), 0);
        assert_eq!(signed(-1), 3);
        for v in [-1000i64, -3, -1, 0, 1, 2, 77, i32::MIN as i64, i32::MAX as i64] {
            assert_eq!(unsigned_to_signed(signed(v)), v);
        }
    }

    #[test]
    fn blocks_records_and_alignment_round_trip() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        w.record(1, &[1]);
        w.enter_block(17, 4);
        w.record(1, &[3]);
        w.record(7, &[32, 1 << 40, 0, 63, 64]);
        w.end_block();
        w.record(2, &[]);
        w.end_block();
        let bytes = w.finish();
        assert_eq!(bytes.len() % 4, 0);
        assert_eq!(&bytes[..4], b"BC\xC0\xDE");
        let items = read(&bytes).unwrap();
        assert_eq!(
            items,
            vec![Item::Block {
                id: 8,
                width: 3,
                items: vec![
                    Item::Record { code: 1, ops: vec![1] },
                    Item::Block {
                        id: 17,
                        width: 4,
                        items: vec![Item::Record { code: 1, ops: vec![3] }, Item::Record { code: 7, ops: vec![32, 1 << 40, 0, 63, 64] }]
                    },
                    Item::Record { code: 2, ops: vec![] },
                ]
            }]
        );
    }

    #[test]
    fn the_block_length_counts_words_after_the_length_word() {
        let mut w = BitWriter::new();
        w.enter_block(8, 3);
        w.record(1, &[1]);
        w.end_block();
        let bytes = w.finish();
        // magic (4), then abbrev id, block id, width and padding (4), the length word at 8.
        let words = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 12 + 4 * words);
    }
}
