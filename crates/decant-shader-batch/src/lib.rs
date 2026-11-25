//! Ports of the batch and replay tools in `src/shaders/decant_shader/tools`: `decant_shader_batch.py`,
//! `decant_shader_summary.py`, `decant_shader_one.py`, and the translation cache
//! `decant_shader_cache.py`, and `decant_shader_player_cache`, which chains the tools to build that cache on a
//! player's machine.
//!
//! The DecantShader probe (`decant_shader_probe.exe`, GPL) and `spirv-val` stay external programs: the
//! tools only start them as processes and read the files they write.

pub mod batch;
pub mod cache;
pub mod json;
pub mod one;
pub mod player;
pub mod process;
pub mod summary;
