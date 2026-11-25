//! The 128 static rANS models of the lanes, and one decode step.
//!
//! A model belongs to an alphabet group `g` (symbols of `g + 1` bits) and has `g + 2` buckets:
//! bucket 0 holds symbol 0, bucket `b > 0` the `2^(b-1)` symbols from `2^(b-1)` on, all with the
//! same frequency. The buckets tile the 65536 slots of the state's low half in order. Slot 0 is
//! never a symbol's: symbol 0 owns slots 1 to `f0 - 1` and decodes with frequency `f0 - 1`.

use crate::tables::MODEL_FREQ;

/// Number of models: 8 alphabet groups of 16 shapes.
pub const MODELS: usize = 128;

/// One model, in the form the decode step uses.
#[derive(Clone, Copy)]
pub struct Model {
    /// First slot of each bucket; 65536 for the buckets past the last one and for the end mark.
    start: [u32; 10],
    freq: [u32; 9],
    /// `ceil(2^32 / freq)`: divides an offset inside a bucket exactly (offset and frequency are
    /// both below 2^16).
    inverse: [u64; 9],
    /// The bucket of the first slot of each run of 256 slots: the search starts there.
    coarse: [u8; 256],
}

impl Model {
    #[allow(clippy::needless_range_loop)] // bucket numbers index four parallel arrays
    fn build(freq_row: &[u16; 9], group: usize) -> Model {
        let used = group + 2;
        let mut model = Model { start: [1 << 16; 10], freq: [0; 9], inverse: [0; 9], coarse: [0; 256] };
        let mut at = 0u32;
        for b in 0..used {
            let f = u32::from(freq_row[b]);
            assert!(f > 0, "every bucket of a model has a frequency");
            model.start[b] = at;
            model.freq[b] = f;
            model.inverse[b] = (1u64 << 32).div_ceil(u64::from(f));
            at += f << b.saturating_sub(1);
        }
        assert_eq!(at, 1 << 16, "the buckets of a model tile 65536 slots");
        for (run, bucket) in model.coarse.iter_mut().enumerate() {
            let first = (run as u32) << 8;
            *bucket = model.start[1..].iter().filter(|&&s| s <= first).count() as u8;
        }
        model
    }

    /// The bucket that holds `slot`.
    #[inline(always)]
    fn bucket(&self, slot: u32) -> usize {
        let mut b = usize::from(self.coarse[(slot >> 8) as usize]);
        while self.start[b + 1] <= slot {
            b += 1;
        }
        b
    }

    /// Decodes one symbol from a state above 2^24: the symbol and the next state.
    #[inline(always)]
    pub fn decode(&self, x: u32) -> (u8, u32) {
        let slot = x & 0xFFFF;
        let b = self.bucket(slot);
        let (symbol, f, rest) = if b == 0 {
            // Slot 0 wraps the rest to -1, as the hardware's subtraction does.
            (0, self.freq[0] - 1, slot.wrapping_sub(1))
        } else {
            let offset = slot - self.start[b];
            let q = ((u64::from(offset) * self.inverse[b]) >> 32) as u32;
            ((1u32 << (b - 1)) + q, self.freq[b], offset - q * self.freq[b])
        };
        (symbol as u8, (x >> 16).wrapping_mul(f).wrapping_add(rest))
    }

    /// The frequency and first slot of `symbol` (the encoder's view of [`Model::decode`]).
    #[cfg(any(test, feature = "encode"))]
    pub fn interval(&self, symbol: u8) -> (u32, u32) {
        if symbol == 0 {
            return (self.freq[0] - 1, 1);
        }
        let b = (8 - symbol.leading_zeros()) as usize;
        (self.freq[b], self.start[b] + (u32::from(symbol) - (1 << (b - 1))) * self.freq[b])
    }
}

/// All 128 models, built once.
pub fn models() -> &'static [Model; MODELS] {
    static MODELS_BUILT: std::sync::OnceLock<Box<[Model; MODELS]>> = std::sync::OnceLock::new();
    MODELS_BUILT.get_or_init(|| Box::new(std::array::from_fn(|m| Model::build(&MODEL_FREQ[m], m / 16))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slot of every model decodes to the symbol whose interval holds it, with the state the
    /// rANS step gives, computed here with plain division.
    #[test]
    fn every_slot_of_every_model_decodes_like_plain_division() {
        for (m, model) in models().iter().enumerate() {
            let group = m / 16;
            let mut owner = vec![u8::MAX; 1 << 16];
            for symbol in 0..(1u32 << (group + 1)) {
                let (f, first) = model.interval(symbol as u8);
                for slot in first..first + f {
                    assert_eq!(owner[slot as usize], u8::MAX, "model {m}: slot {slot} owned twice");
                    owner[slot as usize] = symbol as u8;
                }
            }
            for high in [0x100u32, 0x7531, 0xFFFF] {
                for slot in 1..1u32 << 16 {
                    let x = high << 16 | slot;
                    let symbol = owner[slot as usize];
                    let (f, first) = model.interval(symbol);
                    assert_eq!(model.decode(x), (symbol, (x >> 16) * f + slot - first), "model {m} x {x:#x}");
                }
            }
        }
    }

    #[test]
    fn model_frequencies_tile_the_slots() {
        // `Model::build` asserts it; building all of them is the test.
        assert_eq!(models().len(), MODELS);
        // Group 7 shape 11, as the 8-bit model of US patent 11,234,023 prints it.
        let m = &models()[7 * 16 + 11];
        assert_eq!(m.start[..9], [0, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408]);
    }
}
