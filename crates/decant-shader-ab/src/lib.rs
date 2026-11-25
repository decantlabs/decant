//! GPU A/B comparison of shaders (issue #172).
//!
//! Shader A and shader B (two DXIL containers, or HLSL that DXC compiles) run on a real D3D12
//! device with the same generated inputs, and every writable resource is read back and compared:
//! per component, exactly and within a ULP tolerance. The result of a pair is one of identical,
//! within tolerance, different (with the first differing place and both values), A failed,
//! B failed, both failed or skipped. Separately from that, the pair reports when the two sides
//! declare different resources or signatures: that alone is a finding.
//!
//! Neither side is the truth. The old translator's output is a useful oracle where it exists, but it
//! is wrong in places itself, so the tool reports differences and does not judge them. Each side
//! gets inputs that fit its own signature, fed by semantic (or by register): a pixel shader with
//! one interpolated input and one with four both run on the same generated vertex data.
//!
//! The bindings come from the containers themselves through the D3D12 reflection of DXC. Both
//! sides run with one root signature holding the union of their bindings (or a root signature the
//! user gives), and one set of resources generated from a seed.
//!
//! Layout: [`model`] what a shader declares, [`input`] the generator, [`plan`] the pair's plan,
//! [`compare`] the comparison, [`pairs`] the batch list and results, [`dump`] the file a side's
//! subprocess writes, and `win` the DXC and D3D12 side (Windows only).

pub mod compare;
pub mod dump;
pub mod heapmap;
pub mod input;
pub mod model;
pub mod pairs;
pub mod plan;
#[cfg(windows)]
pub mod win;
