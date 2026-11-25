//! Shared by the integration tests: the test shaders built in
//! the IR, and the validator check.
#![allow(dead_code)]

use decant_dxil::container;
use decant_dxil::ir::*;

pub mod golden;
pub mod shaders;
#[cfg(windows)]
pub mod gpu;
#[cfg(windows)]
pub mod gfx;

/// Writes `shader` and checks it with the validator of dxil.dll. Returns the container bytes the
/// crate wrote. When the validator is absent it prints a skip message and still returns them.
pub fn write_and_validate(name: &str, shader: &Shader) -> Vec<u8> {
    let out = decant_dxil::write(shader).unwrap_or_else(|e| panic!("{name}: write failed: {e}"));
    // Debug aid: DECANT_SHADER_DUMP_DIR=<folder> keeps the containers, for `dxc -dumpbin`.
    if let Some(dir) = std::env::var_os("DECANT_SHADER_DUMP_DIR") {
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.dxbc")), &out.container).expect("dump");
    }
    check_validator(name, &out.container);
    out.container
}

#[cfg(windows)]
pub fn check_validator(name: &str, bytes: &[u8]) {
    use decant_dxil::validate::{ValidateError, validate_in_place};
    let mut copy = bytes.to_vec();
    match validate_in_place(&mut copy) {
        Ok(()) => {
            // The validator writes its own digest in place: it has to equal ours.
            let ours = container::digest(bytes);
            let theirs: Vec<u32> = (0..4).map(|i| container::rd32(&copy, 4 + 4 * i)).collect();
            assert_eq!(theirs, ours, "{name}: the digest written by dxil.dll differs from ours");
            println!("{name}: validated by dxil.dll, digest {:08x?}", ours);
        }
        Err(ValidateError::Missing(why)) => println!("{name}: SKIPPED validation, {why}"),
        Err(e) => panic!("{name}: {e}"),
    }
}

#[cfg(not(windows))]
pub fn check_validator(name: &str, _bytes: &[u8]) {
    println!("{name}: SKIPPED validation, not Windows");
}

pub fn br(t: u32) -> Terminator {
    Terminator::Br(BlockId(t))
}
