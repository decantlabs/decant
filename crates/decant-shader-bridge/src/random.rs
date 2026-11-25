//! Python's `random` module, as far as `random.seed(n)` and `random.sample` go, so the
//! measurement mode of `decant_shader_identify` draws the same shaders as the Python script.
//!
//! The generator is MT19937 seeded with `init_by_array`, as CPython seeds it from an integer.

const N: usize = 624;
const M: usize = 397;

pub struct Random {
    state: [u32; N],
    index: usize,
}

impl Random {
    /// `random.seed(seed)` for a non negative integer.
    pub fn seed(seed: u64) -> Random {
        let mut key: Vec<u32> = Vec::new();
        let mut rest = seed;
        while rest > 0 {
            key.push(rest as u32);
            rest >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = Random { state: [0; N], index: N };
        r.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let previous = r.state[i - 1] ^ (r.state[i - 1] >> 30);
            r.state[i] = (r.state[i] ^ previous.wrapping_mul(1_664_525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                r.state[0] = r.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = r.state[i - 1] ^ (r.state[i - 1] >> 30);
            r.state[i] = (r.state[i] ^ previous.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                r.state[0] = r.state[N - 1];
                i = 1;
            }
        }
        r.state[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let previous = self.state[i - 1] ^ (self.state[i - 1] >> 30);
            self.state[i] = 1_812_433_253u32.wrapping_mul(previous).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn next_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX: u32 = 0x9908_b0df;
        if self.index >= N {
            for k in 0..N {
                let y = (self.state[k] & UPPER) | (self.state[(k + 1) % N] & LOWER);
                self.state[k] = self.state[(k + M) % N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for `k` up to 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 { 0 } else { self.next_u32() >> (32 - k) }
    }

    /// `_randbelow(n)`: a value in `0..n`, for `n` up to 2³².
    fn below(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        let mut r = u64::from(self.getrandbits(k));
        while r >= n {
            r = u64::from(self.getrandbits(k));
        }
        r
    }

    /// `random.sample(population, k)` with `k` at most the population size.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21usize;
        if k > 5 {
            // math.log(x, 4) is log(x) / log(4), as f64::log computes it.
            setsize += 4usize.pow(((k * 3) as f64).log(4.0).ceil() as u32);
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool = population.to_vec();
            for i in 0..k {
                let j = self.below((n - i) as u64) as usize;
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.below(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.below(n as u64) as usize;
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values printed by CPython 3.13 after `random.seed(51)`, the seed the measurement uses.
    #[test]
    fn draws_match_cpython() {
        let mut r = Random::seed(51);
        let big: Vec<u32> = (0..1000).collect();
        assert_eq!(r.sample(&big, 10), [249, 858, 514, 565, 165, 251, 840, 236, 852, 261]);
        let small: Vec<u32> = (0..20).collect();
        assert_eq!(r.sample(&small, 5), [12, 17, 14, 16, 10]);
        assert_eq!(r.getrandbits(32), 2_517_127_322);
        let mut r = Random::seed(51);
        assert_eq!(r.sample(&[7, 3, 9], 3), [7, 9, 3]);
        let huge: Vec<u32> = (0..100_000).collect();
        assert_eq!(r.sample(&huge, 3), [30329, 72394, 33533]);
    }
}
