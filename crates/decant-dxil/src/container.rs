//! The DXBC container: header, parts, the container digest, SFI0, signatures, PSV0, HASH and the
//! DXIL program header.
//!
//! The digest function is a copy of `dxbc_checksum` of `d3d12_xs::rts0` (our own code, written
//! from RFC 1321 and the public description of the DXBC hash). `d3d12-xs` is left untouched for now.

/// A little endian u32 at `at`.
pub fn rd32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Writes a little endian u32 at `at`.
pub fn wr32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// MD5 round constants (RFC 1321).
const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1,
    0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453,
    0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942,
    0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d,
    0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// MD5 rotation amounts.
const R: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4,
    11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// One MD5 block.
fn md5_block(state: &mut [u32; 4], block: &[u8]) {
    let mut m = [0u32; 16];
    for (i, word) in m.iter_mut().enumerate() {
        *word = rd32(block, 4 * i);
    }
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64 {
        let (f, g) = match i {
            0..=15 => ((b & c) | (!b & d), i),
            16..=31 => ((d & b) | (!d & c), (5 * i + 1) & 15),
            32..=47 => (b ^ c ^ d, (3 * i + 5) & 15),
            _ => (c ^ (b | !d), (7 * i) & 15),
        };
        let t = d;
        d = c;
        c = b;
        b = b.wrapping_add(a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]).rotate_left(R[i]));
        a = t;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// The container digest (`ComputeHashRetail`): MD5 state over bytes `[20, total)`, with DXBC's
/// final block layout instead of MD5's. The bit count goes in the first word of the last block and
/// `(bits >> 2) | 1` in its last word. The four words are written little endian at offset 4.
pub fn dxbc_checksum(blob: &[u8], total: usize) -> [u32; 4] {
    let data = &blob[20..total];
    let n = data.len();
    let full = n & !63;
    let rest = n - full;
    let mut state = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let bits = (n as u32).wrapping_mul(8);
    let mut block = [0u8; 64];
    for chunk in data[..full].as_chunks::<64>().0 {
        md5_block(&mut state, chunk);
    }
    if rest >= 56 {
        block[..rest].copy_from_slice(&data[full..]);
        block[rest] = 0x80;
        md5_block(&mut state, &block);
        block = [0u8; 64];
    } else {
        block[4..4 + rest].copy_from_slice(&data[full..]);
        block[4 + rest] = 0x80;
    }
    wr32(&mut block, 0, bits);
    wr32(&mut block, 60, (bits >> 2) | 1);
    md5_block(&mut state, &block);
    state
}

/// The digest of a whole container.
pub fn digest(container: &[u8]) -> [u32; 4] {
    dxbc_checksum(container, container.len())
}

/// Writes the digest into bytes 4 to 20 of the container.
pub fn sign(container: &mut [u8]) {
    let sum = digest(container);
    for (i, word) in sum.iter().enumerate() {
        wr32(container, 4 + 4 * i, *word);
    }
}

/// Builds a container from parts (FourCC and data) and signs it.
pub fn assemble(parts: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
    let header = 32 + 4 * parts.len();
    let total = header + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = vec![0u8; header];
    out[..4].copy_from_slice(b"DXBC");
    out[20] = 1; // major version 1, minor 0
    wr32(&mut out, 24, total as u32);
    wr32(&mut out, 28, parts.len() as u32);
    for (i, (cc, data)) in parts.iter().enumerate() {
        let at = out.len();
        wr32(&mut out, 32 + 4 * i, at as u32);
        out.extend_from_slice(cc);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    debug_assert_eq!(out.len(), total);
    sign(&mut out);
    out
}

/// The parts of a container: FourCC and data. Test and diagnostic helper, bounds checked.
pub fn parts(container: &[u8]) -> Option<Vec<([u8; 4], &[u8])>> {
    if container.len() < 32 || &container[..4] != b"DXBC" {
        return None;
    }
    let total = rd32(container, 24) as usize;
    let count = rd32(container, 28) as usize;
    if total > container.len() || count > (total - 32) / 4 {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..count {
        let off = rd32(container, 32 + 4 * i) as usize;
        if off.checked_add(8)? > total {
            return None;
        }
        let len = rd32(container, off + 4) as usize;
        if len > total - off - 8 {
            return None;
        }
        out.push((container[off..off + 4].try_into().ok()?, &container[off + 8..off + 8 + len]));
    }
    Some(out)
}

/// `SFI0`: the feature flags as a u64.
pub fn sfi0(features: u64) -> Vec<u8> {
    features.to_le_bytes().to_vec()
}

/// An empty signature part (`ISG1`, `OSG1`): no elements, first element offset 8.
pub fn empty_signature() -> Vec<u8> {
    let mut v = vec![0u8; 8];
    wr32(&mut v, 4, 8);
    v
}

/// `HASH`: flags and the 16 byte shader digest, all zero.
pub fn hash_part() -> Vec<u8> {
    vec![0u8; 20]
}

/// The `DXIL` part: program header and bitcode. `kind` is the DXIL shader kind (compute is 5).
pub fn dxil_part(bitcode: &[u8], kind: u32, sm_major: u32, sm_minor: u32, dxil_major: u32, dxil_minor: u32) -> Vec<u8> {
    let mut v = vec![0u8; 24];
    wr32(&mut v, 0, kind << 16 | sm_major << 4 | sm_minor);
    wr32(&mut v, 4, ((24 + bitcode.len()) / 4) as u32);
    v[8..12].copy_from_slice(b"DXIL");
    wr32(&mut v, 12, dxil_major << 8 | dxil_minor);
    wr32(&mut v, 16, 16);
    wr32(&mut v, 20, bitcode.len() as u32);
    v.extend_from_slice(bitcode);
    v
}

/// One resource of the PSV0 resource table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PsvResource {
    /// `PSVResourceType`: sampler 1, CBV 2, SRV typed 3, SRV raw 4, SRV structured 5, UAV typed 6,
    /// UAV raw 7, UAV structured 8, UAV structured with counter 9.
    pub ty: u32,
    pub space: u32,
    pub lower: u32,
    pub upper: u32,
    /// DXIL `ResourceKind`.
    pub kind: u32,
    /// Bit 0: 64 bit atomics.
    pub flags: u32,
}

/// PSV0 of a compute shader (validator 1.6, runtime info version 2): no signature elements.
/// `resources` must already be in PSV order: CBVs, samplers, SRVs, UAVs.
pub fn psv0_compute(threads: [u32; 3], wave_size: Option<u32>, resources: &[PsvResource]) -> Vec<u8> {
    let mut v = Vec::new();
    let put = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    put(&mut v, 48);
    // Runtime info: 16 bytes of the stage union, then the wave lane count range.
    v.extend_from_slice(&[0u8; 16]);
    put(&mut v, wave_size.unwrap_or(0));
    put(&mut v, wave_size.unwrap_or(0xFFFF_FFFF));
    // shader stage (compute 5), uses view id, 2 stage specific bytes, then the element counts.
    v.extend_from_slice(&[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for t in threads {
        put(&mut v, t);
    }
    put(&mut v, resources.len() as u32);
    if !resources.is_empty() {
        put(&mut v, 24);
        for r in resources {
            for x in [r.ty, r.space, r.lower, r.upper, r.kind, r.flags] {
                put(&mut v, x);
            }
        }
    }
    // String table: the empty string, padded to 4 bytes. Semantic index table: empty.
    put(&mut v, 4);
    put(&mut v, 0);
    put(&mut v, 0);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straightforward statement of the digest rule of the container format, written
    /// separately from `dxbc_checksum`: last block with the bit count first and `n * 2 + 1` last.
    fn reference_digest(data: &[u8]) -> [u32; 4] {
        let n = data.len();
        let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        let mut i = 0;
        while n - i >= 64 {
            md5_block(&mut state, &data[i..i + 64]);
            i += 64;
        }
        let r = n - i;
        let mut last = Vec::new();
        if r < 56 {
            last.extend_from_slice(&((n as u32).wrapping_mul(8)).to_le_bytes());
            last.extend_from_slice(&data[i..]);
            last.push(0x80);
            last.resize(60, 0);
            last.extend_from_slice(&((n as u32).wrapping_mul(2).wrapping_add(1)).to_le_bytes());
        } else {
            last.extend_from_slice(&data[i..]);
            last.push(0x80);
            last.resize(64, 0);
            last.extend_from_slice(&((n as u32).wrapping_mul(8)).to_le_bytes());
            last.resize(124, 0);
            last.extend_from_slice(&((n as u32).wrapping_mul(2).wrapping_add(1)).to_le_bytes());
        }
        for block in last.chunks(64) {
            md5_block(&mut state, block);
        }
        state
    }

    #[test]
    fn the_digest_follows_the_two_block_rule_for_every_length() {
        for len in 0..=200usize {
            let mut blob = vec![0u8; 20 + len];
            for (i, b) in blob.iter_mut().enumerate().skip(20) {
                *b = (i * 7 + 3) as u8;
            }
            assert_eq!(digest(&blob), reference_digest(&blob[20..]), "length {len}");
        }
    }

    #[test]
    fn md5_state_matches_the_rfc_1321_empty_message() {
        // MD5("") through the plain padding, built from the block function: 0x80 then zeros.
        let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        let mut block = [0u8; 64];
        block[0] = 0x80;
        md5_block(&mut state, &block);
        let hex: String = state.iter().flat_map(|w| w.to_le_bytes()).map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn assembled_containers_walk_back_into_their_parts() {
        let c = assemble(&[(*b"AAAA", vec![1, 2, 3, 4]), (*b"BBBB", vec![])]);
        assert_eq!(rd32(&c, 24) as usize, c.len());
        let p = parts(&c).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], (*b"AAAA", &[1u8, 2, 3, 4][..]));
        assert_eq!(p[1].1.len(), 0);
        let sum = digest(&c);
        assert!((0..4).all(|i| rd32(&c, 4 + 4 * i) == sum[i]));
    }

    #[test]
    fn psv0_of_a_compute_shader_without_resources_is_68_bytes() {
        let v = psv0_compute([1, 1, 1], None, &[]);
        assert_eq!(v.len(), 68);
        assert_eq!(v[24 + 4], 5, "stage byte");
    }
}
