//! More shader research tools of `src/shaders/decant_shader`, with the same outputs as the Python scripts they replace:
//!
//! | Binary | Python script |
//! |---|---|
//! | `decant_shader_capabilities` | `tools/decant_shader_capabilities.py` |
//! | `decant_shader_hlsl_probe` | `tools/decant_shader_hlsl_probe.py` |
//! | `decant_shader_instrument` | `tools/decant_shader_instrument.py` |
//! | `spirv_reflect` | `tools/spirv_reflect.py`, also a library: [`reflect::reflect`] |
//! | `check_blit` | `harness/check_blit.py` |
//! | `check_texture` | `harness/check_texture.py` |
//! | `aco_opcode_crosscheck` | `tools/aco_opcode_crosscheck.py`, reads the `aco_opcodes.cpp` of a Mesa build |
//!
//! `spirv-cross` and `dxc` run as separate programs, as in the scripts. The instrumentation edits
//! a file of the user's own translator checkout and copies nothing of it.

pub mod aco;
pub mod capabilities;
pub mod files;
pub mod hlsl;
pub mod images;
pub mod instrument;
pub mod pyfmt;
pub mod reflect;

#[cfg(test)]
mod parity;
#[cfg(test)]
pub(crate) mod synthetic;
