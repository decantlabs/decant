//! Bit level input: an MSB first reader and prefix code lookup.

/// Reads bits from a byte slice, most significant bit first. Bits at or past `end` read as zero, and the reader
/// remembers that it ran past the end.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], end: usize) -> Self {
        BitReader {
            data,
            pos: 0,
            end: end.min(data.len() * 8),
        }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn end(&self) -> usize {
        self.end
    }

    pub(crate) fn overrun(&self) -> bool {
        self.pos > self.end
    }

    /// The next 32 bits, not consumed. Bits beyond the end are zero.
    pub(crate) fn peek(&self) -> u32 {
        let first = self.pos / 8;
        let mut acc = 0u64;
        for i in 0..5 {
            let at = first + i;
            let byte = if at * 8 < self.end {
                self.data.get(at).copied().unwrap_or(0)
            } else {
                0
            };
            acc = (acc << 8) | u64::from(byte);
        }
        let aligned = ((acc << (self.pos % 8)) >> 8) as u32;
        let live = self.end.saturating_sub(self.pos).min(32) as u32;
        if live == 0 {
            0
        } else {
            aligned & (u32::MAX << (32 - live))
        }
    }

    pub(crate) fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    /// Takes `n` bits (0 to 32) as an unsigned number.
    pub(crate) fn take(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let value = self.peek() >> (32 - n);
        self.pos += n as usize;
        value
    }

    pub(crate) fn flag(&mut self) -> bool {
        self.take(1) != 0
    }

    /// Takes `n` bits (1 to 32) as a two's complement number.
    pub(crate) fn take_signed(&mut self, n: u32) -> i32 {
        ((self.take(n) << (32 - n)) as i32) >> (32 - n)
    }

    /// A number whose width is announced by up to three flag bits: 8, 16, 24 or 31 bits.
    pub(crate) fn take_wide(&mut self) -> u32 {
        let mut width = 8;
        for extra in [8, 8, 7] {
            if !self.flag() {
                break;
            }
            width += extra;
        }
        self.take(width)
    }
}

const NO_NODE: u32 = u32::MAX;
const LEAF: u32 = 1 << 31;
const LOOKUP_BITS: u32 = 10;

/// A prefix code over symbols, decoded through a lookup of the first bits and a binary tree for longer codes.
pub(crate) struct PrefixCode {
    /// Indexed by the next `LOOKUP_BITS` bits: `symbol << 8 | length`, or 0 when the code is longer.
    lookup: Vec<u32>,
    /// Children of each tree node: `LEAF | symbol` for a leaf, `NO_NODE` when nothing is there.
    nodes: Vec<[u32; 2]>,
}

impl PrefixCode {
    /// Builds the code from (symbol, code, length) triples. The code is right aligned in `length` bits.
    pub(crate) fn new(entries: &[(u16, u32, u8)]) -> Self {
        let mut lookup = vec![0u32; 1 << LOOKUP_BITS];
        let mut nodes = vec![[NO_NODE; 2]];
        for &(symbol, code, len) in entries {
            let len = u32::from(len);
            let mut at = 0usize;
            for i in (0..len).rev() {
                let bit = (code >> i & 1) as usize;
                if i == 0 {
                    nodes[at][bit] = LEAF | u32::from(symbol);
                } else {
                    if nodes[at][bit] == NO_NODE {
                        nodes.push([NO_NODE; 2]);
                        nodes[at][bit] = (nodes.len() - 1) as u32;
                    }
                    at = nodes[at][bit] as usize;
                }
            }
            if len <= LOOKUP_BITS {
                let spare = LOOKUP_BITS - len;
                let first = (code << spare) as usize;
                for slot in &mut lookup[first..first + (1 << spare)] {
                    *slot = u32::from(symbol) << 8 | len;
                }
            }
        }
        PrefixCode { lookup, nodes }
    }

    /// Decodes one symbol, or `None` when the bits match no code (nothing is consumed then).
    pub(crate) fn decode(&self, reader: &mut BitReader) -> Option<u16> {
        let bits = reader.peek();
        let hit = self.lookup[(bits >> (32 - LOOKUP_BITS)) as usize];
        if hit != 0 {
            reader.skip((hit & 0xFF) as usize);
            return Some((hit >> 8) as u16);
        }
        let mut at = 0usize;
        for i in 0..32 {
            match self.nodes[at][(bits >> (31 - i) & 1) as usize] {
                NO_NODE => return None,
                node if node & LEAF != 0 => {
                    reader.skip(i + 1);
                    return Some((node & 0xFFFF) as u16);
                }
                node => at = node as usize,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::*;

    const ALL: [&[(u16, u32, u8)]; 7] = [
        &SCALE_DELTA,
        &SCALE_RUN,
        &COEF_A,
        &COEF_B,
        &QUAD,
        &PAIR,
        &SINGLE,
    ];

    #[test]
    fn bits_come_most_significant_first() {
        let data = [0b1010_0000, 0xFF, 0x00, 0x12, 0x34];
        let mut r = BitReader::new(&data, 40);
        assert!(r.flag());
        assert!(!r.flag());
        assert_eq!(r.take(2), 0b10);
        assert_eq!(r.take(8), 0b0000_1111);
        assert_eq!(r.take_signed(4), -1);
        assert_eq!(r.take(24), 0x1234);
        assert!(!r.overrun());
        assert_eq!(r.take(8), 0);
        assert!(r.overrun());
    }

    #[test]
    fn the_end_limit_cuts_the_last_byte() {
        let data = [0xFF, 0xFF];
        let mut r = BitReader::new(&data, 12);
        assert_eq!(r.take(16), 0xFFF0);
    }

    #[test]
    fn wide_values_have_four_widths() {
        let data = [0xFF; 8];
        let mut r = BitReader::new(&data, 64);
        assert_eq!(r.take_wide(), 0x7FFF_FFFF);
        assert_eq!(r.pos(), 3 + 31);
        let data = [0b0101_0101, 0x00];
        let mut r = BitReader::new(&data, 16);
        assert_eq!(r.take_wide(), 0b1010_1010);
        assert_eq!(r.pos(), 9);
    }

    #[test]
    fn every_code_of_every_table_reads_back() {
        for table in ALL {
            let code = PrefixCode::new(table);
            let mut bits = Vec::new();
            for &(_, c, len) in table {
                for i in (0..len).rev() {
                    bits.push(c >> i & 1 == 1);
                }
            }
            let mut data = vec![0u8; bits.len().div_ceil(8)];
            for (i, &b) in bits.iter().enumerate() {
                data[i / 8] |= u8::from(b) << (7 - i % 8);
            }
            let mut r = BitReader::new(&data, bits.len());
            for &(symbol, _, _) in table {
                assert_eq!(code.decode(&mut r), Some(symbol));
            }
            assert_eq!(r.pos(), bits.len());
        }
    }

    #[test]
    fn the_code_tables_fit_in_the_code_space() {
        for table in ALL {
            let sum: u64 = table
                .iter()
                .map(|&(_, _, len)| 1u64 << (32 - u32::from(len)))
                .sum();
            assert!(sum <= 1 << 32);
            assert!(table.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }

    #[test]
    fn bits_that_match_no_code_consume_nothing() {
        // The scale code leaves no gaps, but a table with a gap does.
        let code = PrefixCode::new(&[(0, 0b0, 1), (1, 0b10, 2)]);
        let data = [0b1100_0000];
        let mut r = BitReader::new(&data, 8);
        assert_eq!(code.decode(&mut r), None);
        assert_eq!(r.pos(), 0);
    }
}
