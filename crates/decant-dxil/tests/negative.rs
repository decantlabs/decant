//! Proof that the validator tests can fail: damaged containers are rejected, and the validator
//! really writes the digest in place (skipped with a message when dxil.dll is absent).
#![cfg(windows)]

mod common;

use common::shaders;
use decant_dxil::container::{self, rd32, wr32};
use decant_dxil::validate::{ValidateError, find_dxil, validate_in_place};

fn available() -> bool {
    let found = find_dxil().is_some();
    if !found {
        println!("SKIPPED: dxil.dll not found");
    }
    found
}

/// The offset of the data of the part `fourcc`.
fn part_offset(blob: &[u8], fourcc: &[u8; 4]) -> usize {
    (0..rd32(blob, 28) as usize).map(|i| rd32(blob, 32 + 4 * i) as usize).find(|&o| &blob[o..o + 4] == fourcc).unwrap() + 8
}

fn resign(blob: &mut [u8]) {
    let sum = container::digest(blob);
    for (i, w) in sum.iter().enumerate() {
        wr32(blob, 4 + 4 * i, *w);
    }
}

fn good() -> Vec<u8> {
    decant_dxil::write(&shaders::buffer_copy()).unwrap().container
}

#[test]
fn the_validator_writes_the_digest_in_place() {
    if !available() {
        return;
    }
    let ours = good();
    let mut blob = ours.clone();
    blob[4..20].fill(0);
    validate_in_place(&mut blob).expect("a zero digest is filled in");
    assert_eq!(blob, ours, "dxil.dll writes exactly the bytes we wrote");
}

#[test]
fn a_wrong_feature_info_part_is_rejected() {
    if !available() {
        return;
    }
    let mut blob = good();
    let at = part_offset(&blob, b"SFI0");
    wr32(&mut blob, at, 0x8000);
    resign(&mut blob);
    match validate_in_place(&mut blob) {
        Err(ValidateError::Rejected(m)) => println!("rejected as expected: {m}"),
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn a_wrong_pipeline_state_validation_part_is_rejected() {
    if !available() {
        return;
    }
    let mut blob = good();
    let at = part_offset(&blob, b"PSV0");
    // The thread group x size: 4 bytes of size, then offset 36 of the runtime info.
    wr32(&mut blob, at + 4 + 36, 63);
    resign(&mut blob);
    match validate_in_place(&mut blob) {
        Err(ValidateError::Rejected(m)) => println!("rejected as expected: {m}"),
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn a_damaged_bitcode_is_rejected() {
    if !available() {
        return;
    }
    let mut blob = good();
    let at = part_offset(&blob, b"DXIL");
    let mid = at + 24 + (rd32(&blob, at + 20) as usize) / 2;
    blob[mid] ^= 0xFF;
    resign(&mut blob);
    assert!(matches!(validate_in_place(&mut blob), Err(ValidateError::Rejected(_))));
}
