//! Python's `random` module, as far as the scripts use it: `random.seed(n)` with an integer and
//! `random.sample`. The scripts draw their shaders with fixed seeds, so the port must draw the very
//! same ones: this is CPython's Mersenne Twister (`_randommodule.c`) and the selection of
//! `Random.sample` in `random.py`, step for step.

const N: usize = 624;
const M: usize = 397;

pub struct Random {
    mt: [u32; N],
    index: usize,
}

impl Random {
    /// `random.seed(n)`: the key is the absolute value as 32 bit words, least significant first.
    pub fn seed(n: u64) -> Random {
        let mut key = vec![n as u32];
        if n >> 32 != 0 {
            key.push((n >> 32) as u32);
        }
        let mut r = Random { mt: [0; N], index: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1_812_433_253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ prev.wrapping_mul(1_664_525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1] ^ (self.mt[i - 1] >> 30);
            self.mt[i] = (self.mt[i] ^ prev.wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn next_u32(&mut self) -> u32 {
        const MAG: [u32; 2] = [0, 0x9908_B0DF];
        if self.index >= N {
            for k in 0..N {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % N] & 0x7FFF_FFFF);
                self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ MAG[(y & 1) as usize];
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for `k` up to 64: whole words first, least significant first.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return u64::from(self.next_u32() >> (32 - k));
        }
        let low = u64::from(self.next_u32());
        let high = u64::from(self.next_u32() >> (64 - k));
        low | (high << 32)
    }

    /// `_randbelow(n)` for `n > 0`, by rejection on `n.bit_length()` bits.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k = 64 - n.leading_zeros();
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.sample(population, k)`, in selection order.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>, String> {
        let n = population.len();
        if k > n {
            return Err("Sample larger than population or is negative".to_owned());
        }
        let mut setsize = 21usize;
        if k > 5 {
            setsize += 4usize.pow(((k * 3) as f64).log(4.0).ceil() as u32);
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool = population.to_vec();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expected values were printed by CPython 3.13 with the same seeds and calls.
    #[test]
    fn draws_match_cpython() {
        let mut r = Random::seed(41);
        assert_eq!([r.getrandbits(32), r.getrandbits(32), r.getrandbits(32)], [1_636_471_393, 1_426_894_585, 990_931_328]);
        assert_eq!(r.getrandbits(40), 184_651_813_940);
        assert_eq!(Random::seed((1 << 40) + 5).getrandbits(32), 2_166_296_868);
        let range = |n: usize| (0..n).collect::<Vec<_>>();
        assert_eq!(Random::seed(41).sample(&range(10), 4).unwrap(), [6, 5, 3, 1]);
        assert_eq!(Random::seed(31).sample(&range(1000), 7).unwrap(), [12, 480, 115, 782, 402, 144, 700]);
        assert_eq!(Random::seed(41).sample(&range(50), 12).unwrap(), [24, 21, 14, 10, 49, 36, 18, 35, 17, 45, 44, 0]);
        assert!(Random::seed(1).sample(&range(3), 4).is_err());
    }
}
