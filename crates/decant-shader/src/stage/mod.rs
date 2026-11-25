//! What depends on the stage: the registers a program finds at entry, what its inputs and
//! exports mean, and how two stages of a pipeline are linked.
//!
//! Each stage file exports `interface` (the HLSL input and output structs and how the program's
//! own numbering maps to them), `prologue` (statements that put the entry state into registers)
//! and `finish` (resolves the exports). This file holds the shared types and the dispatch.
//!
//! Linking: Direct3D links two stages by semantic name and by register, and it refuses a
//! pipeline whose pixel input signature is not what the stage before it writes. So the input
//! struct of a pixel program is generated from the output signature of the stage before it,
//! element by element in the same order, and the pixel program's own inputs are found in it by
//! semantic name and index.

pub mod compute;
pub mod geometry;
pub mod mesh;
pub mod pixel;
pub mod vertex;

use std::collections::BTreeMap;

use crate::api::{SigElement, Stage, StageProgram};
use crate::ir::{Function, Stmt};
use crate::refusal::{Refusal, RefusalKind};

/// One member of an input or output struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    /// The semantic with its index, as it is written in HLSL: `COLOR1`, `SV_Position`.
    pub semantic: String,
    /// The component type code of the signature: 1 unsigned, 2 signed, 3 float.
    pub component_type: u32,
    /// How many components the member has: from the first to the last component of the element
    /// in its register, and no more. Direct3D counts every declared component against its
    /// limits (a mesh group may write 256 bytes per vertex at 128 vertices), so a member is
    /// never wider than its element. The statements name REGISTER components
    /// (`Input::Attr.chan`, `OutputRef.chan`), the printer subtracts `first_comp`.
    pub comps: u8,
    /// The first component of the register that belongs to the element. Two elements can
    /// share one register (694 vertex, 2,770 mesh and 4,385 pixel programs of the title have
    /// such a signature): the first has components x and y, the second starts at z.
    pub first_comp: u8,
    /// The interpolation modifier printed before the type, with its trailing space:
    /// `"nointerpolation "` for an input that is constant over the primitive, empty for the
    /// default (linear with perspective).
    pub interpolation: &'static str,
}

/// The inputs and outputs of one translated stage.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interface {
    pub inputs: Vec<Field>,
    pub outputs: Vec<Field>,
    /// Logical field indices carried as pairs of binary16 components in unsigned words.
    /// These index the mesh outputs or the linked pixel inputs, respectively.
    pub packed_mesh_fields: Vec<usize>,
    /// Pixel: the input member behind hardware attribute `k`.
    pub attrs: Vec<usize>,
    /// Vertex: the input member behind signature register `r`.
    pub vertex_inputs: BTreeMap<u8, usize>,
    /// Compute and mesh: the thread group size. The entry point then takes the thread id in the
    /// group and the group id (`Input::LocalThreadId`, `Input::GroupId`).
    pub threads: Option<[u32; 3]>,
    /// Maximum dwords of workgroup memory, including translator scratch space.
    pub groupshared_limit: u32,
    /// The HLSL expression (a `uint`) for `Input::LaneIndex`. `None` for a stage without lanes
    /// to number. For a stage with `threads` the default is the group index modulo the wave
    /// size (`decant_gi`, the `SV_GroupIndex` parameter of the entry point).
    pub lane_index: Option<String>,
}

/// The struct member of a signature element.
pub fn field(e: &SigElement) -> Field {
    let semantic = if e.system_value != 0 && e.index == 0 {
        e.semantic.clone()
    } else {
        format!("{}{}", e.semantic, e.index)
    };
    Field {
        semantic,
        component_type: e.component_type,
        comps: (8 - e.mask.leading_zeros())
            .saturating_sub(e.mask.trailing_zeros().min(3))
            .max(1) as u8,
        first_comp: e.mask.trailing_zeros().min(3) as u8,
        interpolation: "",
    }
}

/// The elements of a signature in register order, which is the order of the struct members.
pub fn ordered(sig: &[SigElement]) -> Vec<&SigElement> {
    let mut out: Vec<&SigElement> = sig.iter().collect();
    out.sort_by_key(|e| (e.register, e.mask.trailing_zeros()));
    out
}

pub(crate) fn refuse(what: &'static str, detail: impl Into<String>) -> Refusal {
    Refusal::new(RefusalKind::Stage(what), detail)
}

/// The interface of `program`. `before` is the program on the other side of a link, when the
/// pipeline has one: for the pixel and the geometry stage the stage that feeds it, for a mesh
/// program the pixel program it feeds.
pub fn interface(
    program: &StageProgram,
    before: Option<&StageProgram>,
) -> Result<Interface, Refusal> {
    match program.stage {
        Stage::Pixel => pixel::interface(program, before),
        Stage::Vertex => vertex::interface(program),
        Stage::Compute => Ok(Interface {
            groupshared_limit: crate::lift::ds::LDS_DWORDS,
            threads: Some(program.threads),
            ..Interface::default()
        }),
        Stage::Mesh | Stage::Amplification => mesh::interface(program, before),
        Stage::Geometry | Stage::Hull | Stage::Domain => geometry::interface(program, before),
    }
}

/// The output elements of `before` that are members of the struct a pixel program gets from
/// it, in member order. The pixel stage builds its input struct from this list, so it is equal
/// to the output struct of the stage before. A mesh program in a pipeline only keeps what the
/// pixel program takes (`mesh::kept_outputs`), the other stages keep their whole signature.
pub fn linked_outputs<'a>(before: &'a StageProgram, pixel: &StageProgram) -> Vec<&'a SigElement> {
    match before.stage {
        Stage::Mesh => mesh::kept_outputs(before, Some(pixel)),
        _ => ordered(&before.outputs),
    }
}

/// Runs before the resource pass, on the lifted function. A stage replaces here what the
/// resource pass could only refuse (the geometry stage: one load through a system pointer).
pub fn prepare(program: &StageProgram, func: &mut Function) -> Result<(), Refusal> {
    match program.stage {
        Stage::Geometry => geometry::prepare(program, func),
        _ => Ok(()),
    }
    .map_err(|r| r.in_stage(program.stage))
}

/// The statements that run before the first instruction.
pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    // The user data registers hold what the root signature put there. They come first: the
    // stage may put its own values into registers after the user data (the group ids of a
    // compute program follow its last user data register).
    let mut out: Vec<Stmt> = program
        .stage
        .user_registers()
        .map(|k| Stmt::Set {
            dst: crate::ir::Var::S(k),
            value: crate::ir::Expr::UserData(k),
            guard: crate::ir::Guard::Always,
        })
        .collect();
    out.extend(match program.stage {
        Stage::Pixel => pixel::prologue(program),
        Stage::Vertex => vertex::prologue(program),
        Stage::Compute => compute::prologue(program),
        Stage::Mesh => mesh::prologue(program),
        Stage::Geometry => geometry::prologue(program),
        _ => Vec::new(),
    });
    out
}

/// The last word of a stage: it gets the printed text and may wrap the function into the real
/// entry point (a mesh program is a function per thread inside an entry that sets the output
/// counts). The other stages give the text back unchanged.
pub fn entry(program: &StageProgram, iface: &Interface, text: String) -> Result<String, Refusal> {
    match program.stage {
        Stage::Mesh => mesh::entry(program, iface, text),
        Stage::Geometry => geometry::entry(program, iface, text),
        _ => Ok(text),
    }
}

/// Whether the translator carries programs of this stage at all. This is the cheap answer
/// `crate::can_translate` gives before anything is decoded. The owner of a stage removes its
/// arm here in the commit that makes the stage work.
pub fn supported(stage: Stage) -> Result<(), Refusal> {
    match stage {
        Stage::Pixel | Stage::Vertex | Stage::Compute | Stage::Mesh => Ok(()),
        // The title's archive has no amplification program: nothing to measure a rule on.
        Stage::Amplification => Err(refuse("amplification-not-written", "")),
        Stage::Geometry => Ok(()),
        // The title's archive has no hull and no domain program.
        Stage::Hull | Stage::Domain => Err(refuse("tessellation-not-written", "")),
    }
    .map_err(|r| r.in_stage(stage))
}

/// Resolves the exports of `func` into output writes, and whatever else the stage owes.
pub fn finish(
    program: &StageProgram,
    iface: &Interface,
    func: &mut Function,
) -> Result<(), Refusal> {
    match program.stage {
        Stage::Pixel => pixel::finish(program, iface, func),
        Stage::Vertex => vertex::finish(program, iface, func),
        Stage::Compute => compute::finish(program, func),
        Stage::Mesh => mesh::finish(program, iface, func),
        Stage::Amplification => Err(refuse("amplification-not-written", "")),
        Stage::Geometry => geometry::finish(program, iface, func),
        Stage::Hull | Stage::Domain => Err(refuse("tessellation-not-written", "")),
    }
    .map_err(|r| r.in_stage(program.stage))
}
