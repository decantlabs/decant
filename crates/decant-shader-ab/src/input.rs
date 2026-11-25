//! The deterministic input generator: every byte both sides receive comes from a seed and the name
//! of the resource it fills, so A and B get the same bytes whatever order they are created in.

/// SplitMix64, a small generator with good statistics for test data.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator for the resource `key` under `seed`.
    pub fn new(seed: u64, key: &str) -> Rng {
        // FNV-1a over the key, mixed with the seed.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in key.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Rng(h ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A float in `[0, 1)` with 24 bits of precision.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }
}

/// Buffer contents (constant, raw, structured and the window): dwords that are mostly floats in
/// `[0, 1)`, with every fourth one a small integer (0 to 63) so that indices and counts stay in range.
pub fn buffer_bytes(seed: u64, key: &str, len: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed, key);
    let mut out = Vec::with_capacity(len.next_multiple_of(4));
    let mut i = 0usize;
    while out.len() < len {
        let w = if i % 4 == 3 { rng.next_u32() % 64 } else { rng.unit().to_bits() };
        out.extend_from_slice(&w.to_le_bytes());
        i += 1;
    }
    out.truncate(len);
    out
}

/// The procedural pattern of a texture, picked from its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// Red follows x, green follows y, blue follows z, alpha is one minus red.
    Gradient,
    /// Independent noise per texel and channel.
    Noise,
    /// An 8 by 8 checkerboard of two known colours.
    Checker,
    /// Known solid colours, one per mip level and slice.
    Solid,
}

impl Pattern {
    pub fn for_key(seed: u64, key: &str) -> Pattern {
        match Rng::new(seed, key).next_u32() % 4 {
            0 => Pattern::Gradient,
            1 => Pattern::Noise,
            2 => Pattern::Checker,
            _ => Pattern::Solid,
        }
    }
}

/// The four channels of texel `(x, y, z)` of a `w` by `h` by `d` subresource, as floats in `[0, 1]`.
/// `rng` is only consumed by the noise pattern, in texel order.
pub fn texel(pattern: Pattern, rng: &mut Rng, sub: u32, x: u32, y: u32, z: u32, size: [u32; 3]) -> [f32; 4] {
    let f = |v: u32, n: u32| if n <= 1 { 0.5 } else { v as f32 / (n - 1) as f32 };
    match pattern {
        Pattern::Gradient => {
            let r = f(x, size[0]);
            [r, f(y, size[1]), f(z, size[2]), 1.0 - r]
        }
        Pattern::Noise => [rng.unit(), rng.unit(), rng.unit(), rng.unit()],
        Pattern::Checker => {
            if ((x / 8) + (y / 8) + (z / 8)).is_multiple_of(2) {
                [1.0, 0.5, 0.0, 1.0]
            } else {
                [0.0, 0.25, 1.0, 0.5]
            }
        }
        Pattern::Solid => {
            const COLOURS: [[f32; 4]; 6] = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]];
            COLOURS[sub as usize % COLOURS.len()]
        }
    }
}

/// Converts a float to IEEE half precision (round to nearest even), for `R16G16B16A16_FLOAT` data.
pub fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = u32::from(rest > mid || (rest == mid && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1FFF;
    let round = u32::from(rest > 0x1000 || (rest == 0x1000 && half & 1 == 1));
    sign | (half + round) as u16
}

/// Converts IEEE half precision bits to a float.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
        0x1F => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_key_and_seed_give_the_same_bytes() {
        assert_eq!(buffer_bytes(7, "t0 space0", 256), buffer_bytes(7, "t0 space0", 256));
        assert_ne!(buffer_bytes(7, "t0 space0", 256), buffer_bytes(8, "t0 space0", 256));
        assert_ne!(buffer_bytes(7, "t0 space0", 256), buffer_bytes(7, "t1 space0", 256));
        assert_eq!(buffer_bytes(1, "k", 10).len(), 10);
    }

    #[test]
    fn every_fourth_dword_is_a_small_integer_and_the_rest_unit_floats() {
        let b = buffer_bytes(3, "cb", 4096);
        for (i, c) in b.as_chunks::<4>().0.iter().enumerate() {
            let w = u32::from_le_bytes(*c);
            if i % 4 == 3 {
                assert!(w < 64);
            } else {
                let f = f32::from_bits(w);
                assert!((0.0..1.0).contains(&f), "{f}");
            }
        }
    }

    #[test]
    fn half_conversion_round_trips_known_values() {
        for v in [0.0f32, 1.0, -2.5, 0.5, 65504.0, 6.1035156e-5, 5.9604645e-8] {
            assert_eq!(f16_to_f32(f16_bits(v)), v, "{v}");
        }
        assert_eq!(f16_bits(1.0), 0x3C00);
        assert_eq!(f16_bits(f32::INFINITY), 0x7C00);
        assert!(f16_to_f32(f16_bits(f32::NAN)).is_nan());
        assert_eq!(f16_bits(1e9), 0x7C00);
    }
}
