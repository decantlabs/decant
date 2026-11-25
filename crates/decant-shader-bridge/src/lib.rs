//! The DecantShader bridge between the translator and vkd3d-proton, with the same outputs as the Python tools in
//! `src/shaders/decant_shader`:
//!
//! * [`identify`]: ties each resource of a translated shader to its Direct3D register
//!   (`decant_shader_identify.py`, binary `decant_shader_identify`),
//! * [`adapt`]: rewrites the SPIR-V module for the layout vkd3d-proton chose
//!   (`decant_shader_adapt.py`, binary `decant_shader_adapt`),
//! * [`describe`]: writes a module and a JSON description of what it expects per shader
//!   (`decant_shader_describe.py`, binary `decant_shader_describe`),
//! * [`serve`]: the command the vkd3d-proton hook runs per shader (`decant_shader_serve.py`, binary
//!   `decant_shader_serve`),
//! * [`spirvopt`]: its optional spirv-opt step on the final module (`decant_shader_spirvopt.py`),
//! * [`adapt_cache`]: the offline adaptation cache the bridge reads (`decant_shader_adapt_cache.py`,
//!   binary `decant_shader_adapt_cache`).
//!
//! The translation itself runs through `decant_shader_one` and the probe, and the input and output
//! relocation through `decant_shader_locate`, each as its own process. [`shaders::Config`] says where
//! they are and how to swap them for other programs.

pub mod adapt;
pub mod adapt_cache;
pub mod container_bindings;
pub mod describe;
pub mod dxil_origin;
pub mod identify;
pub mod py;
pub mod random;
pub mod serve;
pub mod session;
pub mod shaders;
pub mod spirvopt;

#[cfg(test)]
mod adapt_tests;
