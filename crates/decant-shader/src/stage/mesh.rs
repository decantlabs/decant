//! The mesh stage.
//!
//! Measured on a retail title (all mesh programs, measured with the shader batch tools): every
//! mesh program is one shape. It is a
//! real mesh shader of 128 threads, two waves of 64 lanes, around a vertex shader body:
//!
//! * the thread number is the lane number plus 64 times the wave index, which the program
//!   takes from bits 24 to 27 of `s3`,
//! * the group number is the entry value of `v5` (read through `v_readfirstlane_b32`) shifted
//!   right by 7,
//! * thread 0 sends message 9 with M0 = vertex count | primitive count << 12, and for an empty
//!   group one vertex and one null primitive,
//! * after a barrier the threads below the primitive count export one primitive word (target
//!   20: three vertex numbers at bits 0, 10 and 20, bit 31 for "no primitive"), and the threads
//!   below the vertex count export one vertex (target 12 the position, 13 the clip distances,
//!   32 and up the parameters).
//!
//! The translation keeps the model of the crate, one lane per invocation: the program becomes a
//! function for one thread, `decant_mesh_thread`, printed by the ordinary printer. It returns the
//! vertex of the thread and leaves three more results in static variables: the allocation word,
//! whether the thread exported a vertex, and its primitive word. [`entry`] wraps it into the
//! real entry point, which calls `SetMeshOutputCounts` from every thread, at the top level and
//! before any output write, as Direct3D 12 asks, and then stores vertex `t` and primitive `t`
//! for thread `t`.
//!
//! Three things make that exact:
//!
//! * **The allocation block runs in every thread.** On the console only thread 0 sends the
//!   message, behind a branch on its lane bit. Here the branch is removed ([`force`]), so every
//!   thread computes the counts. That is what the wave of thread 0 does on the machine anyway:
//!   it runs the block with one active lane, the scalar instructions for all. Everything else
//!   in the block is guarded by the lane bit. Direct3D 12 and Vulkan both take the counts from
//!   the first thread of the group, so the result is also defined when another thread has
//!   other values.
//! * **An export only counts in an active lane.** Each output component goes through a
//!   temporary that is written under the lane test, and the vertex and primitive flags are set
//!   under it.
//! * **The barriers after the allocation are dropped.** They only order the exports behind it,
//!   and one of them stands in the block for an empty group, which only one wave runs on the
//!   console. A barrier BEFORE the allocation is a real one in the five programs with local
//!   data share memory (a few threads store control points, all threads read them after the
//!   barrier): there it becomes a group sync, and the memory is the group shared array of
//!   `crate::lift::ds`. It has to stand at the top level of the function, where every thread
//!   arrives.
//!
//! Parameter slots. Slot `j` of the exports is not element `j` of the signature: registers that
//! are not the position are walked in order, a register of 32 bit elements opens a slot, and a
//! register of 16 bit elements takes half a dword per component and shares the open 16 bit slot
//! while its dwords fit. The export table of the container (`StageProgram::param_exports`) names
//! the first register of every slot and is checked against that when it is there.

use std::collections::BTreeMap;

use super::{Field, Interface, field, ordered, refuse};
use crate::api::{SigElement, StageProgram};
use crate::cfg::{self, Term};
use crate::ir::{
    Bin, Export, Expr, Function, Guard, Helper, Input, Node, OutputRef, Stmt, Ty, Var,
};
use crate::refusal::Refusal;
use crate::tables::Op;

/// The message that allocates the outputs of a group (`GS_ALLOC_REQ`).
const ALLOC: u16 = 9;

/// The name of the function for one thread.
const THREAD: &str = "decant_mesh_thread";

/// The first of the temporaries that carry an output component from a guarded export to the
/// output write: one per element and component. The lifter counts its own from zero.
const OUTPUT_TMP: u16 = 0xE000;

static STATE: Helper = Helper {
    name: "decant_mesh_state",
    result: Ty::U32,
    text: "static uint decant_mesh_alloc = 0u;\nstatic bool decant_mesh_vertex = false;\nstatic bool decant_mesh_prim = false;\nstatic uint decant_mesh_word = 0u;",
    needs: &[],
};

static BEGIN: Helper = Helper {
    name: "decant_mesh_begin",
    result: Ty::U32,
    text: "uint decant_mesh_begin() { decant_mesh_alloc = 0u; decant_mesh_vertex = false; decant_mesh_prim = false; decant_mesh_word = 0u; return 0u; }",
    needs: &[&STATE],
};

static COUNTS: Helper = Helper {
    name: "decant_mesh_counts",
    result: Ty::U32,
    text: "uint decant_mesh_counts(uint m0) { decant_mesh_alloc = m0; return 0u; }",
    needs: &[&STATE],
};

static VERTEX: Helper = Helper {
    name: "decant_mesh_has_vertex",
    result: Ty::U32,
    text: "uint decant_mesh_has_vertex() { decant_mesh_vertex = true; return 0u; }",
    needs: &[&STATE],
};

static PRIMITIVE: Helper = Helper {
    name: "decant_mesh_primitive",
    result: Ty::U32,
    text: "uint decant_mesh_primitive(uint word) { decant_mesh_prim = true; decant_mesh_word = word; return 0u; }",
    needs: &[&STATE],
};

static F16: Helper = Helper {
    name: "decant_mesh_f16",
    result: Ty::U32,
    text: "uint decant_mesh_f16(uint a) { return asuint(f16tof32(a & 0xffffu)); }",
    needs: &[],
};

/// The thread group size the translation can carry: one row of at most 128 threads, lane
/// `t & 63` of wave `t >> 6`. A program without a container has no size and gets the one
/// every mesh program of the title has.
fn threads(program: &StageProgram) -> Result<[u32; 3], Refusal> {
    match program.threads {
        [0, 0, 0] => Ok([128, 1, 1]),
        [x @ 1..=128, 1, 1] => Ok([x, 1, 1]),
        [x, y, z] => Err(refuse("mesh-threads", format!("{x} x {y} x {z} threads"))),
    }
}

/// One component of an output element and where a parameter slot holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Source {
    to: OutputRef,
    /// The dword of the slot (the source register of the export).
    dword: u8,
    /// `None` for a 32 bit component, else the half of the dword: 0 the low one.
    half: Option<u8>,
}

/// What the exports of a program mean, from its output signature.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    /// The element with the position, as an index into the ordered outputs.
    position: usize,
    /// The components the position element has.
    position_mask: u8,
    /// The clip distances and then the cull distances, in export order: target 13 carries the
    /// first four, target 14 the next four.
    distances: Vec<OutputRef>,
    /// Per parameter slot: the components it carries.
    slots: Vec<Vec<Source>>,
    /// Per parameter slot: the signature register that opened it.
    first_registers: Vec<u32>,
    /// Per parameter slot: the dwords that carry a component of the signature.
    masks: Vec<u8>,
}

impl Layout {
    fn of(program: &StageProgram) -> Result<Layout, Refusal> {
        let outputs = ordered(&program.outputs);
        let position = outputs
            .iter()
            .position(|e| e.system_value == 1)
            .ok_or_else(|| refuse("mesh-no-position", "no SV_Position in the output signature"))?;
        // System values: 1 the position, 2 a clip distance, 3 a cull distance. A layer or
        // viewport index would be a field of the second position target, which no mesh
        // program of the title writes.
        if let Some(e) = outputs.iter().find(|e| e.system_value > 3) {
            return Err(refuse(
                "mesh-system-value",
                format!("{} in the output signature", e.semantic),
            ));
        }
        let mut distance_elements: Vec<usize> = (0..outputs.len())
            .filter(|i| matches!(outputs[*i].system_value, 2 | 3))
            .collect();
        distance_elements.sort_by_key(|i| (outputs[*i].system_value, outputs[*i].index));
        let mut distances = Vec::new();
        for i in distance_elements {
            for chan in (0..4u8).filter(|c| (outputs[i].mask >> c) & 1 != 0) {
                distances.push(OutputRef {
                    element: i as u16,
                    chan,
                });
            }
        }

        // The registers that are not the position, with their elements.
        let mut registers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (i, e) in outputs.iter().enumerate() {
            if i != position {
                registers.entry(e.register).or_default().push(i);
            }
        }
        let mut slots: Vec<Vec<Source>> = Vec::new();
        let mut first_registers = Vec::new();
        // The 16 bit slot that still has room: its number and the dwords used.
        let mut open: Option<(usize, u8)> = None;
        for (register, elements) in &registers {
            let sixteen = |i: &usize| matches!(outputs[*i].component_type, 4..=6);
            let halves = elements.iter().all(sixteen);
            if !halves && elements.iter().any(sixteen) {
                return Err(refuse(
                    "mesh-export-layout",
                    format!("register {register} mixes 16 and 32 bit elements"),
                ));
            }
            // 16 bit integers would need their own conversion. The title has 16 bit floats only.
            if halves && elements.iter().any(|i| outputs[*i].component_type != 6) {
                return Err(refuse(
                    "mesh-export-layout",
                    format!("register {register} holds 16 bit integers"),
                ));
            }
            let mask = elements.iter().fold(0u8, |m, i| m | outputs[*i].mask);
            let comps = 8 - mask.leading_zeros() as u8;
            let (slot, base) = if halves {
                let need = comps.div_ceil(2);
                match open {
                    Some((slot, used)) if used + need <= 4 => {
                        open = Some((slot, used + need));
                        (slot, used)
                    }
                    _ => {
                        slots.push(Vec::new());
                        first_registers.push(*register);
                        open = Some((slots.len() - 1, need));
                        (slots.len() - 1, 0)
                    }
                }
            } else {
                slots.push(Vec::new());
                first_registers.push(*register);
                (slots.len() - 1, 0)
            };
            for chan in (0..4u8).filter(|c| (mask >> c) & 1 != 0) {
                let element = elements
                    .iter()
                    .copied()
                    .find(|i| (outputs[*i].mask >> chan) & 1 != 0)
                    .expect("the mask is the union of the element masks");
                slots[slot].push(Source {
                    to: OutputRef {
                        element: element as u16,
                        chan,
                    },
                    dword: if halves { base + chan / 2 } else { chan },
                    half: halves.then_some(chan % 2),
                });
            }
        }
        let masks = slots
            .iter()
            .map(|slot| slot.iter().fold(0, |m, s| m | 1 << s.dword))
            .collect();
        let layout = Layout {
            position,
            position_mask: outputs[position].mask,
            distances,
            slots,
            first_registers,
            masks,
        };
        layout.check_table(program)?;
        Ok(layout)
    }

    /// The layout against the members a pipeline keeps (`fields`, see [`kept_outputs`]): the
    /// element numbers become member numbers, and a component of an element that was left out
    /// has no place any more, so its export writes nothing. The slots and their dwords stay
    /// what the whole signature makes them.
    fn kept(mut self, program: &StageProgram, fields: &[Field]) -> Result<Layout, Refusal> {
        let outputs = ordered(&program.outputs);
        let member: Vec<Option<u16>> = outputs
            .iter()
            .map(|e| {
                let name = field(e).semantic;
                fields
                    .iter()
                    .position(|f| f.semantic == name)
                    .map(|i| i as u16)
            })
            .collect();
        let at = |to: OutputRef| {
            member[usize::from(to.element)].map(|element| OutputRef {
                element,
                chan: to.chan,
            })
        };
        self.position = member[self.position]
            .map(usize::from)
            .ok_or_else(|| refuse("mesh-no-position", "the position is not a member"))?;
        self.distances = self
            .distances
            .iter()
            .map(|d| {
                at(*d).ok_or_else(|| refuse("mesh-export-target", "a distance is not a member"))
            })
            .collect::<Result<_, _>>()?;
        for slot in &mut self.slots {
            *slot = slot
                .iter()
                .filter_map(|s| at(s.to).map(|to| Source { to, ..*s }))
                .collect();
        }
        Ok(self)
    }

    /// The export table of the container must name the same first register for every slot.
    fn check_table(&self, program: &StageProgram) -> Result<(), Refusal> {
        if program.param_exports.is_empty() {
            return Ok(());
        }
        let table: Vec<u32> = {
            let mut records: Vec<_> = program.param_exports.iter().collect();
            records.sort_by_key(|r| r.slot);
            records.iter().map(|r| r.register).collect()
        };
        if table != self.first_registers {
            return Err(refuse(
                "mesh-export-layout",
                format!(
                    "the export table has registers {table:?}, the signature gives {:?}",
                    self.first_registers
                ),
            ));
        }
        Ok(())
    }

    /// The dwords of a slot that carry a component.
    fn dwords(&self, slot: usize) -> u8 {
        self.masks[slot]
    }
}

/// The dwords of group shared memory a Direct3D 12 mesh shader may declare (28 KiB).
const LDS_LIMIT: u32 = 7168;

/// What the code must not have for this translation: more local data share memory than a mesh
/// shader may have, and an allocation message inside a loop.
fn check_code(program: &StageProgram) -> Result<(), Refusal> {
    if program.lds_bytes.div_ceil(4) > LDS_LIMIT {
        return Err(refuse(
            "mesh-lds-size",
            format!("{} bytes of group shared memory", program.lds_bytes),
        ));
    }
    let graph = cfg::build(&program.code)?;
    let mut sender = None;
    for (b, block) in graph.blocks.iter().enumerate() {
        for insn in &block.insns {
            if insn.op == Op::s_sendmsg {
                sender = Some((b, insn.pc));
            }
        }
    }
    let Some((sender, pc)) = sender else {
        return Ok(());
    };
    let next = |b: usize| match graph.blocks[b].term {
        Term::End => [None, None],
        Term::Jump(t) => [Some(t), None],
        Term::Branch { taken, fall, .. } => [Some(taken), Some(fall)],
    };
    let mut seen = vec![false; graph.blocks.len()];
    let mut work: Vec<usize> = next(sender).into_iter().flatten().collect();
    while let Some(b) = work.pop() {
        if b == sender {
            return Err(refuse("mesh-alloc-shape", "the allocation message is in a loop").at(pc));
        }
        if !std::mem::replace(&mut seen[b], true) {
            work.extend(next(b).into_iter().flatten());
        }
    }
    Ok(())
}

/// Direct3D 12 gives a mesh group 32,768 bytes of vertex output: the declared vertex array
/// times the bytes of one vertex, the array length rounded up to 32. DXC counts four bytes for
/// every declared component, also for 16 bit types (tried with `float16_t` and native 16 bit
/// types). Count the 32 bit words in the packed output struct when the vertex needs packing.
/// If the vertex is still too large, report the size before invoking the compiler.
fn check_size(program: &StageProgram, iface: &Interface) -> Result<(), Refusal> {
    let [vertices, ..] = threads(program)?;
    let bytes: u32 = iface
        .outputs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let comps = if iface.packed_mesh_fields.contains(&i) {
                f.comps.div_ceil(2)
            } else {
                f.comps
            };
            u32::from(comps) * 4
        })
        .sum();
    if bytes * vertices.next_multiple_of(32) > 32_768 {
        return Err(refuse(
            "mesh-output-size",
            format!("{bytes} bytes for each of {vertices} vertices"),
        ));
    }
    Ok(())
}

/// Everything about a mesh program that can be judged without lifting it: the thread group,
/// the output signature against the export table, the size of the outputs and the code shape.
/// The census of the stage rules over a corpus uses it, [`finish`] runs the same checks.
pub fn check(program: &StageProgram) -> Result<(), Refusal> {
    threads(program)?;
    Layout::of(program)?;
    check_size(program, &interface(program, None)?)?;
    check_code(program)
}

/// The output elements that are members of the vertex, in member order. Alone, a mesh program
/// keeps its whole signature. In a pipeline (`pixel` is the pixel program it feeds) it keeps
/// the position, the clip and cull distances, and the elements the pixel program takes: the
/// ones its input signature marks as read and the ones in a register its attribute table
/// names. The others would only cost output bytes, which Direct3D 12 limits (see
/// `check_size`). The pixel stage builds its input struct from the same list
/// (`stage::linked_outputs`), so both sides stay equal.
pub fn kept_outputs<'a>(
    program: &'a StageProgram,
    pixel: Option<&StageProgram>,
) -> Vec<&'a SigElement> {
    let all = ordered(&program.outputs);
    let Some(pixel) = pixel else {
        return all;
    };
    let named: Vec<u32> = pixel
        .pixel_inputs
        .iter()
        .flat_map(|a| std::iter::once(a.register).chain(a.high_register))
        .collect();
    let taken = |e: &SigElement| {
        pixel.inputs.iter().any(|i| {
            i.system_value == 0
                && i.semantic == e.semantic
                && i.index == e.index
                && (i.rw_mask != 0 || named.contains(&i.register))
        })
    };
    all.into_iter()
        .filter(|e| e.system_value != 0 || taken(e))
        .collect()
}

/// The pack plan uses the retained signature before adding rasterizer-generated pixel inputs.
pub(super) fn packed_fields(
    program: &StageProgram,
    outputs: &[&SigElement],
) -> Result<Vec<usize>, Refusal> {
    let [vertices, ..] = threads(program)?;
    let comps: u32 = outputs.iter().map(|e| u32::from(field(e).comps)).sum();
    if comps * 4 * vertices.next_multiple_of(32) <= 32_768 {
        return Ok(Vec::new());
    }
    Ok(outputs
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            (e.system_value == 0 && e.component_type == 6 && field(e).comps > 1).then_some(i)
        })
        .collect())
}

/// `pixel` is the pixel program of the pipeline, when the caller links one.
pub fn interface(
    program: &StageProgram,
    pixel: Option<&StageProgram>,
) -> Result<Interface, Refusal> {
    let outputs = kept_outputs(program, pixel);
    Ok(Interface {
        packed_mesh_fields: packed_fields(program, &outputs)?,
        outputs: outputs.into_iter().map(field).collect(),
        threads: Some(threads(program)?),
        groupshared_limit: LDS_LIMIT,
        // Two waves of 64 lanes: the programs build the thread number as lane + 64 * wave.
        lane_index: Some("(decant_tid.x & 63u)".to_owned()),
        ..Interface::default()
    })
}

/// The entry state. Measured: a mesh program reads `s3` (only bits 24 to 27, the wave index),
/// `v5` (only through `v_readfirstlane_b32`, the thread number of the dispatch) and its user
/// data. `v5` gets the value the first lane of the wave has, through a scalar register, so that
/// the read back is the read of a value that is the same in every lane. `s0` is free for that:
/// no mesh program reads it before writing it.
pub fn prologue(_program: &StageProgram) -> Vec<Stmt> {
    let set = |dst: Var, value: Expr| Stmt::Set {
        dst,
        value,
        guard: Guard::Always,
    };
    let thread = || Expr::Input(Input::LocalThreadId { axis: 0 });
    let group = Expr::Input(Input::GroupId { axis: 0 });
    vec![
        set(Var::Exec, Expr::Bool(true)),
        set(
            Var::S(3),
            Expr::bin(
                Bin::Shl,
                Expr::bin(Bin::LShr, thread(), Expr::U32(6)),
                Expr::U32(24),
            ),
        ),
        set(
            Var::S(0),
            Expr::bin(
                Bin::Or,
                Expr::bin(Bin::Shl, group, Expr::U32(7)),
                Expr::bin(Bin::And, thread(), Expr::U32(64)),
            ),
        ),
        set(Var::V(5), Expr::Var(Var::S(0))),
    ]
}

/// Whether `nodes` hold a statement `wanted` names, at any depth.
fn holds(nodes: &[Node], wanted: &dyn Fn(&Stmt) -> bool) -> bool {
    nodes.iter().any(|n| match n {
        Node::Block(stmts) => stmts.iter().any(wanted),
        Node::If { then, els, .. } => holds(then, wanted) || holds(els, wanted),
        Node::Loop(body) | Node::Once(body) => holds(body, wanted),
        Node::Break | Node::Continue | Node::Return => false,
    })
}

/// Whether a branch condition depends on the lane: it reads a lane bit or a vector register.
/// A condition on a temporary could be either, so it is refused.
fn lane_test(cond: &Expr) -> Result<bool, Refusal> {
    let (mut lane, mut unknown) = (false, false);
    crate::resource::reads(cond, &mut |v| match v {
        Var::Exec | Var::Vcc | Var::Mask(_) | Var::V(_) => lane = true,
        Var::Tmp(_) | Var::Flag(_) => unknown = true,
        _ => {}
    });
    if unknown {
        return Err(refuse(
            "mesh-alloc-shape",
            "the branch around the allocation tests a temporary",
        ));
    }
    Ok(lane)
}

/// Whether a `Break` of this block stands under a branch on the lane. The breaks of loops and
/// blocks inside are their own.
fn leaves_on_lane(nodes: &[Node], under_lane: bool) -> Result<bool, Refusal> {
    for n in nodes {
        match n {
            Node::Break if under_lane => return Ok(true),
            Node::If { cond, then, els } => {
                let lane = under_lane || lane_test(cond)?;
                if leaves_on_lane(then, lane)? || leaves_on_lane(els, lane)? {
                    return Ok(true);
                }
            }
            _ => {}
        }
    }
    Ok(false)
}

/// Makes every thread reach the statements `wanted` names (the allocation message, a barrier
/// for group shared memory): a branch on the lane that leads to one is taken by all. See the
/// module text for why that is exact. `key` names the refusal for a shape this cannot do.
fn force(
    nodes: &mut Vec<Node>,
    wanted: &dyn Fn(&Stmt) -> bool,
    key: &'static str,
) -> Result<(), Refusal> {
    let mut i = 0;
    while i < nodes.len() {
        let mut inline = None;
        match &mut nodes[i] {
            Node::If { cond, then, els } if holds(then, wanted) || holds(els, wanted) => {
                if lane_test(cond)? {
                    let (taken, other) = if holds(then, wanted) {
                        (then, els)
                    } else {
                        (els, then)
                    };
                    if !other.is_empty() {
                        return Err(refuse(key, "the branch on the lane has a second arm"));
                    }
                    force(taken, wanted, key)?;
                    inline = Some(std::mem::take(taken));
                } else {
                    // A branch every thread takes alike stays, and both arms are looked at.
                    force(then, wanted, key)?;
                    force(els, wanted, key)?;
                }
            }
            // A block that runs once is like the statements around it, unless a branch on the
            // lane leaves it early: then some threads would pass the statement by.
            Node::Once(body) if holds(body, wanted) => {
                if leaves_on_lane(body, false)? {
                    return Err(refuse(key, "inside a block that lanes leave early"));
                }
                force(body, wanted, key)?;
            }
            Node::Loop(body) if holds(body, wanted) => {
                return Err(refuse(key, "inside a loop"));
            }
            _ => {}
        }
        match inline {
            Some(body) => {
                let n = body.len();
                nodes.splice(i..=i, body);
                i += n;
            }
            None => i += 1,
        }
    }
    Ok(())
}

fn effect(helper: &'static Helper, args: Vec<Expr>, guard: Guard) -> Stmt {
    Stmt::Effect {
        dst: None,
        call: Expr::Call(helper, args),
        guard,
    }
}

/// An output component written by an export: under the lane test into its temporary, and from
/// there into the vertex. An inactive lane writes the vertex again with what it had.
fn write(to: OutputRef, value: Expr, out: &mut Vec<Stmt>) {
    let tmp = Var::Tmp(OUTPUT_TMP + to.element * 4 + u16::from(to.chan));
    out.push(Stmt::Set {
        dst: tmp,
        value,
        guard: Guard::Exec,
    });
    out.push(Stmt::Output {
        to,
        value: Expr::Var(tmp),
    });
}

fn export(layout: &Layout, e: &Export, out: &mut Vec<Stmt>) -> Result<(), Refusal> {
    let bad = |what: String| refuse("mesh-export-target", what).at(e.pc);
    if e.compressed {
        return Err(bad(format!("a compressed export to target {}", e.target)));
    }
    let enabled = |k: u8| (e.enabled >> k) & 1 != 0;
    let source = |k: u8| Expr::Var(Var::V(e.src[usize::from(k)]));
    match e.target {
        // The primitive of this thread: one word in the first source.
        20 => {
            if !enabled(0) {
                return Err(bad("a primitive export without its word".to_owned()));
            }
            out.push(effect(&PRIMITIVE, vec![source(0)], Guard::Exec));
            return Ok(());
        }
        // The position. An export with nothing enabled still makes the vertex (the one vertex
        // of an empty group).
        12 => {
            let mask = layout.position_mask;
            for chan in (0..4u8).filter(|c| enabled(*c) && (mask >> c) & 1 != 0) {
                write(
                    OutputRef {
                        element: layout.position as u16,
                        chan,
                    },
                    source(chan),
                    out,
                );
            }
        }
        // Clip and cull distances, four per target.
        13 | 14 => {
            for chan in (0..4u8).filter(|c| enabled(*c)) {
                let to = layout
                    .distances
                    .get(usize::from((e.target - 13) * 4 + chan))
                    .ok_or_else(|| {
                        bad(format!(
                            "target {} component {chan} has no clip or cull distance element",
                            e.target
                        ))
                    })?;
                write(*to, source(chan), out);
            }
        }
        32..=63 => {
            let slot = usize::from(e.target - 32);
            if slot >= layout.slots.len() || e.enabled & !layout.dwords(slot) != 0 {
                return Err(refuse(
                    "mesh-export-layout",
                    format!(
                        "parameter {slot} enables {:#x}, the signature gives {:#x}",
                        e.enabled,
                        layout.slots.get(slot).map_or(0, |_| layout.dwords(slot))
                    ),
                )
                .at(e.pc));
            }
            for s in layout.slots[slot].iter().filter(|s| enabled(s.dword)) {
                let value = match s.half {
                    None => source(s.dword),
                    Some(0) => Expr::Call(&F16, vec![source(s.dword)]),
                    Some(_) => Expr::Call(
                        &F16,
                        vec![Expr::bin(Bin::LShr, source(s.dword), Expr::U32(16))],
                    ),
                };
                write(s.to, value, out);
            }
        }
        other => return Err(bad(format!("mesh export to target {other}"))),
    }
    out.push(effect(&VERTEX, Vec::new(), Guard::Exec));
    Ok(())
}

pub fn finish(
    program: &StageProgram,
    iface: &Interface,
    func: &mut Function,
) -> Result<(), Refusal> {
    let layout = Layout::of(program)?.kept(program, &iface.outputs)?;
    check_size(program, iface)?;
    check_code(program)?;
    let mut messages = Vec::new();
    func.for_each_stmt(&mut |s| {
        if let Stmt::Message { id, pc } = s {
            messages.push((*id, *pc));
        }
    });
    if let Some((id, pc)) = messages.iter().find(|(id, _)| *id != ALLOC) {
        return Err(refuse("mesh-message", format!("message {id:#x}")).at(*pc));
    }
    if let [_, (_, pc), ..] = messages.as_slice() {
        return Err(refuse("mesh-alloc-shape", "a second allocation message").at(*pc));
    }
    force(
        &mut func.body,
        &|s| matches!(s, Stmt::Message { .. }),
        "mesh-alloc-shape",
    )?;

    // Group shared memory: the array, and the barriers that guard it. Those are the ones
    // before the allocation, and they must stand where every thread arrives once.
    let shared = crate::lift::ds::uses_lds(func);
    if shared {
        func.groupshared_dwords = match program.lds_bytes {
            0 => LDS_LIMIT,
            bytes => bytes.div_ceil(4),
        };
    }
    if shared {
        // The barriers before the allocation, in program order. Like the allocation they may
        // stand behind a branch that only skips a region no lane of the wave is active in.
        let (mut early, mut allocated) = (Vec::new(), false);
        func.for_each_stmt(&mut |s| match s {
            Stmt::Message { .. } => allocated = true,
            Stmt::Barrier { pc } if !allocated => early.push(*pc),
            _ => {}
        });
        force(
            &mut func.body,
            &|s| matches!(s, Stmt::Barrier { pc } if early.contains(pc)),
            "mesh-barrier-nested",
        )?;
    }
    let top_level: Vec<u32> = func
        .body
        .iter()
        .filter_map(|n| match n {
            Node::Block(stmts) => Some(stmts),
            _ => None,
        })
        .flatten()
        .filter_map(|s| match s {
            Stmt::Barrier { pc } => Some(*pc),
            _ => None,
        })
        .collect();

    let (mut result, mut allocated) = (Ok(()), false);
    func.for_each_block_mut(&mut |stmts| {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts.drain(..) {
            match s {
                Stmt::Export(e) => {
                    if let Err(r) = export(&layout, &e, &mut out) {
                        result = Err(r);
                    }
                }
                // Every thread reaches it after `force`, with the same counts in M0.
                Stmt::Message { .. } => {
                    allocated = true;
                    out.push(effect(&COUNTS, vec![Expr::Var(Var::M0)], Guard::Always));
                }
                Stmt::Barrier { pc } if shared && !allocated => {
                    if !top_level.contains(&pc) {
                        result = Err(refuse(
                            "mesh-barrier-nested",
                            "a barrier for group shared memory under a branch or in a loop",
                        )
                        .at(pc));
                    }
                    out.push(effect(&crate::lift::ds::BARRIER, Vec::new(), Guard::Always));
                }
                // The barriers that order the exports: nothing to wait for.
                Stmt::Barrier { .. } => {}
                other => out.push(other),
            }
        }
        *stmts = out;
    });
    result?;

    // The static results start clean, and the statement makes sure they are declared.
    let begin = effect(&BEGIN, Vec::new(), Guard::Always);
    match func.body.first_mut() {
        Some(Node::Block(stmts)) => stmts.insert(0, begin),
        _ => func.body.insert(0, Node::Block(vec![begin])),
    }
    Ok(())
}

/// Wraps the printed function for one thread into the entry point of the mesh shader.
///
/// The printer wrote the thread function as a compute style entry point: an attribute line and
/// `DecantOut main(<thread inputs>) {`. That head becomes `DecantOut decant_mesh_thread(...)`, and the
/// real entry point takes the same inputs plus the two output arrays.
pub fn entry(_program: &StageProgram, iface: &Interface, text: String) -> Result<String, Refusal> {
    let bug = |what: &str| refuse("mesh-entry", what.to_owned());
    let [threads, ..] = iface.threads.unwrap_or([128, 1, 1]);
    let head = format!(
        "[numthreads({threads}, 1, 1)]
DecantOut {}(",
        crate::hlsl::ENTRY
    );
    if text.matches(&head).count() != 1 {
        return Err(bug("the printed text has no thread function to wrap"));
    }
    let at = text.find(&head).expect("counted above") + head.len();
    let params = text[at..]
        .split_once(") {")
        .map(|(params, _)| params.to_owned())
        .ok_or_else(|| bug("the thread function has no parameter list"))?;
    // `uint3 decant_tid: SV_GroupThreadID` is passed on as `decant_tid`.
    let names: Vec<&str> = params
        .split(',')
        .filter_map(|p| p.split(':').next()?.split_whitespace().last())
        .collect();
    if !names.contains(&"decant_tid") {
        return Err(bug("the thread function does not take the thread id"));
    }
    let mut out = text.replacen(&head, &format!("DecantOut {THREAD}("), 1);
    let entry = crate::hlsl::ENTRY;
    let args = names.join(", ");
    let output_type = if iface.packed_mesh_fields.is_empty() {
        "DecantOut"
    } else {
        "DecantMeshOut"
    };
    let mut lines = vec![
        format!("[numthreads({threads}, 1, 1)]"),
        "[outputtopology(\"triangle\")]".to_owned(),
        format!(
            "void {entry}({params}, out vertices {output_type} decant_vertices[{threads}], out indices uint3 decant_triangles[{threads}]) {{"
        ),
        format!("    DecantOut decant_vertex = {THREAD}({args});"),
        "    // Every thread calls this with the same counts. Direct3D takes the first thread's."
            .to_owned(),
        format!("    uint decant_count_v = min(decant_mesh_alloc & 0x7ffu, {threads}u);"),
        format!("    uint decant_count_p = min((decant_mesh_alloc >> 12) & 0x7ffu, {threads}u);"),
        "    SetMeshOutputCounts(decant_count_v, decant_count_p);".to_owned(),
    ];
    if iface.packed_mesh_fields.is_empty() {
        lines.push("    if (decant_mesh_vertex && decant_tid.x < decant_count_v) decant_vertices[decant_tid.x] = decant_vertex;".to_owned());
    } else {
        lines.push("    if (decant_mesh_vertex && decant_tid.x < decant_count_v) {".to_owned());
        for (i, field) in iface.outputs.iter().enumerate() {
            let dest = format!("decant_vertices[decant_tid.x].o{i}");
            let source = format!("decant_vertex.o{i}");
            if !iface.packed_mesh_fields.contains(&i) {
                lines.push(format!("        {dest} = {source};"));
                continue;
            }
            for (word, pair) in b"xyzw"[..usize::from(field.comps)].chunks(2).enumerate() {
                let lo = char::from(pair[0]);
                let packed = match pair.get(1).copied().map(char::from) {
                    Some(hi) => {
                        format!("f32tof16({source}.{lo}) | (f32tof16({source}.{hi}) << 16u)")
                    }
                    None => format!("f32tof16({source}.{lo})"),
                };
                let suffix = if field.comps > 2 {
                    if word == 0 { ".x" } else { ".y" }
                } else {
                    ""
                };
                lines.push(format!("        {dest}{suffix} = {packed};"));
            }
        }
        lines.push("    }".to_owned());
    }
    lines.extend([
        "    if (decant_mesh_prim && decant_tid.x < decant_count_p) {".to_owned(),
        "        // Vertex numbers at bits 0, 10 and 20. Bit 31: no primitive, written as a triangle without area.".to_owned(),
        "        uint w = decant_mesh_word;".to_owned(),
        "        decant_triangles[decant_tid.x] = (w >> 31) != 0u ? uint3(0u, 0u, 0u) : uint3(w & 0x1ffu, (w >> 10) & 0x1ffu, (w >> 20) & 0x1ffu);".to_owned(),
        "    }".to_owned(),
        "}".to_owned(),
    ]);
    for l in lines {
        out.push_str(&l);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ParamExport, SigElement, Stage};

    fn element(
        semantic: &str,
        index: u32,
        sv: u32,
        ty: u32,
        register: u32,
        mask: u8,
    ) -> SigElement {
        SigElement {
            semantic: semantic.to_owned(),
            index,
            system_value: sv,
            component_type: ty,
            register,
            mask,
            rw_mask: 0,
        }
    }

    fn program(outputs: Vec<SigElement>) -> StageProgram {
        StageProgram {
            outputs,
            ..StageProgram::new(Stage::Mesh, Vec::new())
        }
    }

    fn dwords(layout: &Layout) -> Vec<u8> {
        (0..layout.slots.len()).map(|s| layout.dwords(s)).collect()
    }

    #[test]
    fn two_sixteen_bit_registers_share_a_slot_across_a_32_bit_one() {
        // The shape of the title's plain transform programs: COLOR0 and TEXCOORD1 are 16 bit.
        let p = program(vec![
            element("SV_Position", 0, 1, 3, 0, 15),
            element("COLOR", 0, 0, 6, 1, 15),
            element("TEXCOORD", 0, 0, 3, 2, 3),
            element("TEXCOORD", 1, 0, 6, 3, 15),
            element("TEXCOORD", 3, 0, 3, 4, 15),
        ]);
        let l = Layout::of(&p).unwrap();
        assert_eq!(l.first_registers, [1, 2, 4]);
        assert_eq!(dwords(&l), [0xf, 0x3, 0xf]);
        // TEXCOORD1.z is the low half of dword 3 of slot 0.
        let z = l.slots[0]
            .iter()
            .find(|s| s.to.element == 3 && s.to.chan == 2)
            .unwrap();
        assert_eq!((z.dword, z.half), (3, Some(0)));
    }

    #[test]
    fn a_narrow_sixteen_bit_register_is_followed_at_the_next_dword() {
        let p = program(vec![
            element("TEXCOORD", 0, 0, 6, 0, 3),
            element("TEXCOORD", 3, 0, 3, 1, 15),
            element("TEXCOORD", 1, 0, 6, 2, 15),
            element("SV_Position", 0, 1, 3, 3, 15),
        ]);
        let l = Layout::of(&p).unwrap();
        assert_eq!(l.first_registers, [0, 1]);
        assert_eq!(dwords(&l), [0x7, 0xf]);
        // Three components in two dwords, then a register that no longer fits.
        let p = program(vec![
            element("SV_Position", 0, 1, 3, 0, 15),
            element("TEXCOORD", 0, 0, 6, 1, 7),
            element("TEXCOORD", 1, 0, 6, 2, 7),
            element("TEXCOORD", 2, 0, 6, 3, 15),
        ]);
        let l = Layout::of(&p).unwrap();
        assert_eq!(l.first_registers, [1, 3]);
        assert_eq!(dwords(&l), [0xf, 0x3]);
    }

    #[test]
    fn elements_that_share_a_register_keep_their_components() {
        let p = program(vec![
            element("SV_Position", 0, 1, 3, 2, 15),
            element("TEXCOORD", 1, 0, 3, 1, 3),
            element("TEXCOORD", 5, 0, 3, 1, 12),
            element("SV_ClipDistance", 0, 2, 3, 0, 1),
        ]);
        let l = Layout::of(&p).unwrap();
        // Ordered: the clip distance, TEXCOORD1, TEXCOORD5, the position.
        assert_eq!(l.position, 3);
        assert_eq!(
            l.distances,
            [OutputRef {
                element: 0,
                chan: 0
            }]
        );
        // The clip distance register has a parameter slot as well.
        assert_eq!(l.first_registers, [0, 1]);
        let to: Vec<(u16, u8)> = l.slots[1]
            .iter()
            .map(|s| (s.to.element, s.to.chan))
            .collect();
        assert_eq!(to, [(1, 0), (1, 1), (2, 2), (2, 3)]);
    }

    #[test]
    fn the_export_table_of_the_container_is_the_authority() {
        let mut p = program(vec![
            element("SV_Position", 0, 1, 3, 0, 15),
            element("TEXCOORD", 0, 0, 3, 1, 15),
            element("TEXCOORD", 1, 0, 3, 2, 15),
        ]);
        let record = |slot: u8, register: u32| ParamExport {
            slot,
            kind: 6,
            register,
            sixteen_bit_register: 0,
            flags: 0,
        };
        p.param_exports = vec![record(0, 1), record(1, 2)];
        assert!(Layout::of(&p).is_ok());
        p.param_exports = vec![record(0, 2), record(1, 1)];
        assert_eq!(
            Layout::of(&p).unwrap_err().key(),
            "stage:mesh-export-layout"
        );
    }

    fn message() -> Node {
        Node::Block(vec![Stmt::Message { id: ALLOC, pc: 0 }])
    }

    fn branch(cond: Expr, then: Vec<Node>, els: Vec<Node>) -> Node {
        Node::If { cond, then, els }
    }

    #[test]
    fn a_structured_branch_on_the_lane_around_the_allocation_is_removed() {
        // if (exec) { if (scc) { message } }: the lane test goes, the uniform test stays.
        let inner = branch(Expr::Var(Var::Scc), vec![message()], Vec::new());
        let mut body = vec![
            Node::Block(Vec::new()),
            branch(Expr::Var(Var::Exec), vec![inner.clone()], Vec::new()),
            Node::Return,
        ];
        let wanted = |s: &Stmt| matches!(s, Stmt::Message { .. });
        force(&mut body, &wanted, "mesh-alloc-shape").unwrap();
        assert_eq!(body, [Node::Block(Vec::new()), inner, Node::Return]);
        // The message in the else arm of a test for "not active".
        let mut body = vec![branch(
            Expr::un(crate::ir::Un::LNot, Expr::Var(Var::Exec)),
            Vec::new(),
            vec![message()],
        )];
        force(&mut body, &wanted, "mesh-alloc-shape").unwrap();
        assert_eq!(body, [message()]);
        // A barrier under the same kind of branch comes to the top level the same way.
        let barrier = Node::Block(vec![Stmt::Barrier { pc: 7 }]);
        let mut body = vec![branch(
            Expr::Var(Var::Exec),
            vec![barrier.clone(), message()],
            Vec::new(),
        )];
        let wanted = |s: &Stmt| matches!(s, Stmt::Barrier { pc: 7 });
        force(&mut body, &wanted, "mesh-barrier-nested").unwrap();
        assert_eq!(body, [barrier, message()]);
    }

    #[test]
    fn allocation_shapes_that_cannot_be_forced_are_refused() {
        let wanted = |s: &Stmt| matches!(s, Stmt::Message { .. });
        let key = |mut body: Vec<Node>| {
            force(&mut body, &wanted, "mesh-alloc-shape")
                .unwrap_err()
                .key()
        };
        let exec = || Expr::Var(Var::Exec);
        // A second arm would be skipped by the threads the test let through.
        let other = Node::Block(vec![Stmt::Barrier { pc: 0 }]);
        assert_eq!(
            key(vec![branch(exec(), vec![message()], vec![other])]),
            "stage:mesh-alloc-shape"
        );
        assert_eq!(
            key(vec![Node::Loop(vec![message()])]),
            "stage:mesh-alloc-shape"
        );
        // A block that runs once is looked into, unless a branch on the lane leaves it early.
        let mut once = vec![Node::Once(vec![branch(
            exec(),
            vec![message()],
            Vec::new(),
        )])];
        force(&mut once, &wanted, "mesh-alloc-shape").unwrap();
        assert_eq!(once, [Node::Once(vec![message()])]);
        let leave = branch(exec(), vec![Node::Break], Vec::new());
        assert_eq!(
            key(vec![Node::Once(vec![leave, message()])]),
            "stage:mesh-alloc-shape"
        );
        // A temporary may hold a lane bit or a uniform condition.
        assert_eq!(
            key(vec![branch(
                Expr::Var(Var::Flag(3)),
                vec![message()],
                Vec::new()
            )]),
            "stage:mesh-alloc-shape"
        );
    }

    #[test]
    fn a_pipeline_keeps_only_the_members_its_pixel_program_takes() {
        let mesh = program(vec![
            element("SV_ClipDistance", 0, 2, 3, 0, 1),
            element("TEXCOORD", 0, 0, 3, 1, 15),
            element("TEXCOORD", 1, 0, 3, 2, 3),
            element("TEXCOORD", 5, 0, 3, 2, 12),
            element("TEXCOORD", 2, 0, 3, 3, 15),
            element("SV_Position", 0, 1, 3, 4, 15),
        ]);
        // The pixel program reads TEXCOORD5 by its read mask and TEXCOORD2 by its attribute
        // table, declares TEXCOORD0 without reading it and does not know TEXCOORD1.
        let read = |semantic: &str, index: u32, register: u32, rw_mask: u8| SigElement {
            rw_mask,
            ..element(semantic, index, 0, 3, register, 15)
        };
        let mut pixel = StageProgram {
            inputs: vec![
                element("SV_Position", 0, 1, 3, 0, 15),
                read("TEXCOORD", 0, 1, 0),
                read("TEXCOORD", 5, 2, 12),
                read("TEXCOORD", 2, 3, 0),
            ],
            ..StageProgram::new(Stage::Pixel, Vec::new())
        };
        pixel.pixel_inputs = vec![crate::api::PixelInput {
            attr: 0,
            kind: 6,
            register: 3,
            high_register: None,
            channels: 15,
            flags: 2,
        }];
        let names = |pixel: Option<&StageProgram>| -> Vec<String> {
            kept_outputs(&mesh, pixel)
                .iter()
                .map(|e| format!("{}{}", e.semantic, e.index))
                .collect()
        };
        assert_eq!(names(None).len(), 6);
        assert_eq!(
            names(Some(&pixel)),
            ["SV_ClipDistance0", "TEXCOORD5", "TEXCOORD2", "SV_Position0"]
        );
        // The layout follows the members: TEXCOORD0 and TEXCOORD1 have no place any more, the
        // slots and the dwords an export may enable are those of the whole signature.
        let fields = interface(&mesh, Some(&pixel)).unwrap().outputs;
        let l = Layout::of(&mesh).unwrap().kept(&mesh, &fields).unwrap();
        assert_eq!(l.position, 3);
        assert_eq!(
            l.distances,
            [OutputRef {
                element: 0,
                chan: 0
            }]
        );
        assert_eq!(dwords(&l), [0x1, 0xf, 0xf, 0xf]);
        assert!(l.slots[1].is_empty());
        let to: Vec<(u16, u8)> = l.slots[2]
            .iter()
            .map(|s| (s.to.element, s.to.chan))
            .collect();
        assert_eq!(to, [(1, 2), (1, 3)]);
    }

    #[test]
    fn a_vertex_larger_than_the_group_output_limit_is_refused() {
        // 128 vertices of 256 bytes are the limit: the position and 15 more vectors fit.
        let outputs = |n: u32| {
            let mut v = vec![element("SV_Position", 0, 1, 3, 0, 15)];
            v.extend((0..n).map(|k| element("TEXCOORD", k, 0, 3, 1 + k, 15)));
            v
        };
        let size = |p: &StageProgram| check_size(p, &interface(p, None).unwrap());
        assert_eq!(size(&program(outputs(15))), Ok(()));
        assert_eq!(
            size(&program(outputs(16))).unwrap_err().key(),
            "stage:mesh-output-size"
        );
        // A smaller group has room for a larger vertex.
        let mut small = program(outputs(16));
        small.threads = [64, 1, 1];
        assert_eq!(size(&small), Ok(()));
    }

    #[test]
    fn signatures_the_stage_does_not_carry_are_refused_by_name() {
        let p = program(vec![element("TEXCOORD", 0, 0, 3, 0, 15)]);
        assert_eq!(Layout::of(&p).unwrap_err().key(), "stage:mesh-no-position");
        let p = program(vec![
            element("SV_Position", 0, 1, 3, 0, 15),
            element("SV_RenderTargetArrayIndex", 0, 4, 1, 1, 1),
        ]);
        assert_eq!(Layout::of(&p).unwrap_err().key(), "stage:mesh-system-value");
        let mut p = program(vec![element("SV_Position", 0, 1, 3, 0, 15)]);
        p.threads = [8, 8, 1];
        assert_eq!(interface(&p, None).unwrap_err().key(), "stage:mesh-threads");
    }
}
