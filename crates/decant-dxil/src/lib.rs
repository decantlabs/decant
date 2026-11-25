//! `decant-dxil`: a DXIL writer without LLVM. It takes a small scalar SSA IR ([`ir::Shader`])
//! and returns the bytes of a DXBC container holding DXIL (LLVM 3.7 bitcode) for Shader Model 6.6,
//! with the container digest computed by the crate itself.
//!
//! See the README of the crate for the IR, the container layout and the test strategy.

pub mod bitstream;
pub mod cfg;
pub mod container;
pub mod dxop;
pub mod ir;
mod lower;
pub mod module;
pub mod sig;
pub mod spirv;
#[cfg(windows)]
pub mod validate;

use crate::ir::{ResourceClass, ResourceKind, Shader};

/// Whether the 14 fixed metadata kind records are written (specification 2.9).
const WRITE_METADATA_KINDS: bool = true;

/// What can go wrong, with the instruction or value named in the message.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("invalid shader: {0}")]
    Invalid(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("irreducible control flow: {0}")]
    Irreducible(String),
    #[error("use before definition: {0}")]
    UseBeforeDef(String),
}

/// A bound resource the root signature has to provide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundResource {
    pub class: ResourceClass,
    pub kind: ResourceKind,
    pub space: u32,
    pub register: u32,
    pub range_size: u32,
    /// For constant buffers, the size in bytes.
    pub size_bytes: u32,
}

/// What the layer needs to build or patch the root signature (specification 1.6).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingReport {
    pub bound: Vec<BoundResource>,
    /// `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` is needed.
    pub resource_heap: bool,
    /// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED` is needed.
    pub sampler_heap: bool,
    /// The `WaveSize` the shader declares.
    pub wave_size: Option<u32>,
}

/// The result of [`write`].
#[derive(Clone, Debug)]
pub struct Output {
    /// The signed DXBC container.
    pub container: Vec<u8>,
    pub report: BindingReport,
}

/// A phi must have exactly one input for each predecessor.
fn check_phis(shader: &Shader) -> Result<(), Error> {
    let mut preds = vec![Vec::new(); shader.blocks.len()];
    for (i, b) in shader.blocks.iter().enumerate() {
        match &b.term {
            ir::Terminator::Br(t) => preds[t.0 as usize].push(i as u32),
            ir::Terminator::CondBr(_, t, f) => {
                preds[t.0 as usize].push(i as u32);
                preds[f.0 as usize].push(i as u32);
            }
            ir::Terminator::Ret => {}
        }
    }
    for (i, b) in shader.blocks.iter().enumerate() {
        for phi in &b.phis {
            let mut got: Vec<u32> = phi.incoming.iter().map(|(p, _)| p.0).collect();
            let mut want = preds[i].clone();
            got.sort_unstable();
            want.sort_unstable();
            if got != want {
                return Err(Error::Invalid(format!("phi {} has inputs for blocks {got:?}, but its block has the predecessors {want:?}", phi.dst.0)));
            }
        }
    }
    Ok(())
}

/// Turns `shader` into a signed DXBC container: `SFI0`, `ISG1`, `OSG1`, `PSV0`, `HASH`, `DXIL`.
pub fn write(shader: &Shader) -> Result<Output, Error> {
    let mut s = shader.clone();
    cfg::normalize(&mut s)?;
    check_phis(&s)?;
    let lowered = lower::lower(&s)?;
    let bitcode = lowered.module.write_bitcode(WRITE_METADATA_KINDS)?;
    let (parts, kind) = match s.stage {
        ir::Stage::Compute { threads, wave_size } => (
            [
                (*b"SFI0", container::sfi0(lowered.features)),
                (*b"ISG1", container::empty_signature()),
                (*b"OSG1", container::empty_signature()),
                (*b"PSV0", container::psv0_compute(threads, wave_size, &lowered.psv)),
                (*b"HASH", container::hash_part()),
                (*b"DXIL", Vec::new()),
            ],
            5,
        ),
        ir::Stage::Vertex | ir::Stage::Pixel => {
            let pixel = matches!(s.stage, ir::Stage::Pixel);
            let data = lowered.sig.as_ref().expect("graphics shaders are analysed");
            let g = sig::PsvGraphics { pixel, inputs: &s.inputs, outputs: &s.outputs, data };
            (
                [
                    (*b"SFI0", container::sfi0(lowered.features)),
                    (*b"ISG1", sig::signature_part(&s.inputs, &data.inputs, false)),
                    (*b"OSG1", sig::signature_part(&s.outputs, &data.outputs, true)),
                    (*b"PSV0", sig::psv0_graphics(&g, &lowered.psv)),
                    (*b"HASH", container::hash_part()),
                    (*b"DXIL", Vec::new()),
                ],
                if pixel { 0 } else { 1 },
            )
        }
    };
    let mut parts = parts;
    parts[5].1 = container::dxil_part(&bitcode, kind, 6, 6, 1, 6);
    let report = BindingReport {
        bound: s
            .resources
            .iter()
            .filter(|r| r.binding == ir::Binding::Bound)
            .map(|r| BoundResource { class: r.class, kind: r.kind, space: r.space, register: r.lower_bound, range_size: r.range_size, size_bytes: r.size_bytes })
            .collect(),
        resource_heap: lowered.resource_heap,
        sampler_heap: lowered.sampler_heap,
        wave_size: lowered.wave_size,
    };
    Ok(Output { container: container::assemble(&parts), report })
}
