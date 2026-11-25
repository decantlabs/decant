//! The part of the shader research tools in `src/shaders/decant_shader/tools` that other tools build on:
//! the metadata rows, the location tables of `decant_shader_locate.py` and `decant_shader_link.py`, the
//! SPIR-V relocation, and text formatted the way the Python tools format it.

pub mod link;
pub mod locate;
pub mod meta;
pub mod paths;
pub mod py;
pub mod spirv;
