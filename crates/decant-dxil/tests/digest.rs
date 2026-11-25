//! The container digest of this crate against the one of `d3d12_xs::rts0` (the code it was copied from).

use decant_dxil::container::{digest, dxbc_checksum};

#[test]
fn the_copied_digest_equals_the_original_for_lengths_0_to_200() {
    for len in 0..=200usize {
        let mut blob = vec![0u8; 20 + len];
        for (i, b) in blob.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[2];
        }
        assert_eq!(digest(&blob), d3d12_xs::rts0::dxbc_checksum(&blob, blob.len()), "length {len}");
        assert_eq!(dxbc_checksum(&blob, blob.len()), d3d12_xs::rts0::dxbc_checksum(&blob, blob.len()));
    }
}
