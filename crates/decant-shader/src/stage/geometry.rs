//! The geometry stage. Hull and domain programs come here too and are refused: the title's
//! archive has none.
//!
//! Measured on the title (24 Nov 2026, all 195 geometry programs, see the specification
//! `gfx10/ngg-and-geometry.md`, "Measured on the title"): one shape. The program is the geometry
//! part alone, one lane for one input primitive, and it talks to the stages around it through
//! the local data share:
//!
//! * `v0` holds the place of input vertex 0 in its low half and of vertex 1 in its high half,
//!   `v1` the place of vertex 2, counted in dwords. Component `c` of the input element in
//!   signature register `r` is dword `4r + c` of the vertex.
//! * An output vertex is a slot of `4 * output registers` dwords, component `c` of register `r`
//!   at dword `4r + c`. The slots of a lane follow each other from
//!   `lane * slots * slot size + (X << 8) + s2`, where `X` is a dword the program loads through
//!   a system pointer (`s8` and the low byte of `s9`) at byte 0xB8.
//! * Message 0x22 emits the next slot as a vertex, 0x12 ends the strip, 0x3 ends the program.
//!   There is no export.
//! * `s3` holds the number of lanes with a primitive in bits 8 to 15.
//!
//! The translation keeps that program as it is and gives it a local data share of its own: a
//! static array of the invocation, declared here under the names the data share rules use
//! (`crate::lift::ds`: `decant_lds`, `decant_lds_words`), so their loads and stores work unchanged.
//! The entry point ([`entry`]) copies the input vertices into the array at places it chose
//! through `v0` and `v1`, calls the function for one primitive, and appends the output vertices
//! it finds in the slots, in emit order:
//!
//! ```text
//! dwords 0 .. 3 * in:            the input vertices, `in` = 4 * input registers each
//! dwords 3 * in .. + n * out:    the output slots, `out` = 4 * output registers each
//! ```
//!
//! The load through the system pointer is replaced by zero before the resource pass
//! ([`prepare`]), so the slots start where `s2` says.

use std::collections::BTreeSet;

use super::{Field, Interface, field, ordered, refuse};
use crate::api::{SigElement, StageProgram};
use crate::ir::{Expr, Function, Guard, Helper, Node, Stmt, Ty, Var};
use crate::refusal::Refusal;

/// The dwords of the array that stands for the local data share.
const RING_DWORDS: u32 = 512;

/// The name of the function for one primitive.
const PRIMITIVE: &str = "decant_gs_primitive";

/// What `s8` holds at entry: nothing the program may use as a number. [`prepare`] follows it
/// to the load through the system pointer.
const SYSTEM_POINTER: u32 = 0x5347_5853;

/// The byte of the system table the measured programs read: the base of the output slots in
/// units of 256 bytes.
const SLOT_BASE_FIELD: u32 = 0xB8;

static STATE: Helper = Helper {
    name: "decant_gs_state",
    result: Ty::U32,
    text: "static uint decant_lds[512];\nstatic const uint decant_lds_words = 512u;\nstatic uint decant_gs_count = 0u;\nstatic uint decant_gs_cuts = 0u;",
    needs: &[],
};

static BEGIN: Helper = Helper {
    name: "decant_gs_begin",
    result: Ty::U32,
    text: "uint decant_gs_begin() { decant_gs_count = 0u; decant_gs_cuts = 0u; return 0u; }",
    needs: &[&STATE],
};

static EMIT: Helper = Helper {
    name: "decant_gs_emit",
    result: Ty::U32,
    text: "uint decant_gs_emit() { decant_gs_count += 1u; return 0u; }",
    needs: &[&STATE],
};

static CUT: Helper = Helper {
    name: "decant_gs_cut",
    result: Ty::U32,
    text: "uint decant_gs_cut() { if (decant_gs_count != 0u) { decant_gs_cuts |= 1u << ((decant_gs_count - 1u) & 31u); } return 0u; }",
    needs: &[&STATE],
};

/// The numbers of the ring and of the entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ring {
    /// Vertices of an input primitive.
    corners: u32,
    /// Dwords of one input vertex.
    input: u32,
    /// Dwords of one output slot.
    output: u32,
    /// Output slots.
    slots: u32,
    /// `PointStream`, `LineStream` or `TriangleStream`.
    stream: &'static str,
}

impl Ring {
    fn of(program: &StageProgram) -> Result<Ring, Refusal> {
        let registers = |sig: &[SigElement]| sig.iter().map(|e| e.register + 1).max().unwrap_or(0);
        // Without the container the title's one shape: triangles in, a strip of three out.
        let (primitive, topology, streams, slots) = match program.geometry {
            Some(g) => (
                g.input_primitive,
                g.output_topology,
                g.stream_mask,
                g.max_vertices,
            ),
            None => (3, 5, 1, 3),
        };
        // The places of the input vertices in v0 and v1 are measured for triangles only.
        if primitive != 3 {
            return Err(refuse(
                "geometry-input-primitive",
                format!("input primitive {primitive}"),
            ));
        }
        if streams != 1 {
            return Err(refuse(
                "geometry-streams",
                format!("stream mask {streams:#x}"),
            ));
        }
        let stream = match topology {
            1 => "PointStream",
            3 => "LineStream",
            5 => "TriangleStream",
            other => {
                return Err(refuse(
                    "geometry-output-topology",
                    format!("output topology {other}"),
                ));
            }
        };
        let ring = Ring {
            corners: 3,
            input: 4 * registers(&program.inputs),
            output: 4 * registers(&program.outputs),
            slots,
            stream,
        };
        // The cut flags are one word, and everything has to lie in the array.
        if slots == 0 || slots > 32 || ring.first_slot() + slots * ring.output > RING_DWORDS {
            return Err(refuse(
                "geometry-ring-size",
                format!(
                    "{slots} vertices of {} dwords after {} input dwords",
                    ring.output,
                    ring.first_slot()
                ),
            ));
        }
        Ok(ring)
    }

    /// The dword of the first output slot.
    fn first_slot(&self) -> u32 {
        self.corners * self.input
    }
}

/// What can be judged without lifting the program.
pub fn check(program: &StageProgram) -> Result<(), Refusal> {
    Ring::of(program).map(|_| ())
}

/// `before` is the vertex or domain program in front, when the pipeline was given whole: the
/// input struct is then its output signature, member by member, as Direct3D links them.
pub fn interface(
    program: &StageProgram,
    before: Option<&StageProgram>,
) -> Result<Interface, Refusal> {
    if program.stage != crate::api::Stage::Geometry {
        return Ok(Interface::default());
    }
    let source = before.map_or(&program.inputs, |b| &b.outputs);
    Ok(Interface {
        inputs: ordered(source).iter().map(|e| field(e)).collect(),
        outputs: ordered(&program.outputs).iter().map(|e| field(e)).collect(),
        // One invocation is one primitive: lane 0 of its wave.
        lane_index: Some("0u".to_owned()),
        ..Interface::default()
    })
}

/// The entry state: one lane with a primitive, the three input vertices at the start of the
/// ring, the output slots behind them.
pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    let Ok(ring) = Ring::of(program) else {
        // `finish` refuses the program with the reason.
        return Vec::new();
    };
    let set = |dst: Var, value: u32| Stmt::Set {
        dst,
        value: Expr::U32(value),
        guard: Guard::Always,
    };
    vec![
        Stmt::Set {
            dst: Var::Exec,
            value: Expr::Bool(true),
            guard: Guard::Always,
        },
        // The byte address of the first output slot (the program adds `X << 8`, see `prepare`).
        set(Var::S(2), ring.first_slot() * 4),
        // One lane has a primitive. Bits 16 to 23 go to M0, which no data share rule reads.
        set(Var::S(3), 1 << 8),
        set(Var::S(8), SYSTEM_POINTER),
        set(Var::S(9), 0),
        // Vertex 0 at dword 0, vertex 1 after it (the high half), vertex 2 after that.
        set(Var::V(0), ring.input << 16),
        set(Var::V(1), 2 * ring.input),
    ]
}

/// Replaces every scalar load through `pointer` in `e`. Returns the offset of a load that is
/// not the one field the stage knows.
fn replace_load(e: &mut Expr, pointer: &BTreeSet<u8>) -> Result<(), Refusal> {
    match e {
        Expr::Load(load) if !load.buffer && pointer.contains(&load.base) => {
            if *load.offset != Expr::U32(SLOT_BASE_FIELD) {
                return Err(refuse(
                    "geometry-system-load",
                    "a load through the system pointer at an unknown offset",
                )
                .at(load.pc));
            }
            // The slots start where `s2` says.
            *e = Expr::U32(0);
        }
        Expr::Un(_, a) => replace_load(a, pointer)?,
        Expr::Bin(_, a, b) => {
            replace_load(a, pointer)?;
            replace_load(b, pointer)?;
        }
        Expr::Select(c, a, b) => {
            replace_load(c, pointer)?;
            replace_load(a, pointer)?;
            replace_load(b, pointer)?;
        }
        Expr::Call(_, args) => {
            for a in args {
                replace_load(a, pointer)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Runs before the resource pass: the load through the system pointer, which that pass could
/// only refuse as a raw address, becomes the constant this stage chose. The pointer is
/// followed through plain moves in the first block, where the measured programs load.
pub fn prepare(program: &StageProgram, func: &mut Function) -> Result<(), Refusal> {
    if program.stage != crate::api::Stage::Geometry {
        return Ok(());
    }
    // A program the stage does not carry is refused here by its own reason, before the
    // resource pass meets a load that only this stage could have explained.
    Ring::of(program)?;
    let Some(Node::Block(stmts)) = func.body.first_mut() else {
        return Ok(());
    };
    let mut pointer: BTreeSet<u8> = BTreeSet::new();
    for s in stmts {
        let Stmt::Set { dst, value, .. } = s else {
            continue;
        };
        replace_load(value, &pointer)?;
        if let Var::S(n) = dst {
            let holds = match value {
                Expr::U32(SYSTEM_POINTER) => true,
                Expr::Var(Var::S(from)) => pointer.contains(from),
                _ => false,
            };
            if holds {
                pointer.insert(*n);
            } else {
                pointer.remove(n);
            }
        }
    }
    Ok(())
}

fn effect(helper: &'static Helper) -> Stmt {
    Stmt::Effect {
        dst: None,
        call: Expr::Call(helper, Vec::new()),
        // A lane without a primitive emits nothing.
        guard: Guard::Exec,
    }
}

/// The first emit or cut message inside a loop.
fn message_in_loop(nodes: &[Node], inside: bool) -> Option<u32> {
    nodes.iter().find_map(|n| match n {
        Node::Block(stmts) if inside => stmts.iter().find_map(|s| match s {
            Stmt::Message { pc, .. } => Some(*pc),
            _ => None,
        }),
        Node::If { then, els, .. } => {
            message_in_loop(then, inside).or_else(|| message_in_loop(els, inside))
        }
        Node::Loop(body) => message_in_loop(body, true),
        Node::Once(body) => message_in_loop(body, inside),
        _ => None,
    })
}

pub fn finish(
    program: &StageProgram,
    _iface: &Interface,
    func: &mut Function,
) -> Result<(), Refusal> {
    let ring = Ring::of(program)?;
    // The slot of an emit is its number, so the emits must be countable: none in a loop.
    if let Some(pc) = message_in_loop(&func.body, false) {
        return Err(refuse("geometry-emit-loop", "a message inside a loop").at(pc));
    }
    let (mut result, mut emits) = (Ok(()), 0u32);
    func.for_each_block_mut(&mut |stmts| {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts.drain(..) {
            match s {
                // Bits 0 to 3 name the receiver (2 the geometry stage, 3 the end of the
                // program), bit 4 ends the strip, bit 5 emits a vertex, bits 8 and 9 name the
                // stream.
                Stmt::Message { id, pc } => match id {
                    0x22 | 0x32 => {
                        emits += 1;
                        out.push(effect(&EMIT));
                        if id == 0x32 {
                            out.push(effect(&CUT));
                        }
                    }
                    0x12 => out.push(effect(&CUT)),
                    0x3 => {}
                    other => {
                        result =
                            Err(refuse("geometry-message", format!("message {other:#x}")).at(pc));
                    }
                },
                Stmt::Export(e) => {
                    result = Err(refuse(
                        "geometry-export",
                        format!("an export to target {}", e.target),
                    )
                    .at(e.pc));
                }
                // One invocation has nothing to wait for.
                Stmt::Barrier { .. } => {}
                other => out.push(other),
            }
        }
        *stmts = out;
    });
    result?;
    if emits > ring.slots {
        return Err(refuse(
            "geometry-ring-size",
            format!("{emits} emits for {} vertices", ring.slots),
        ));
    }
    // The statement declares the ring before the data share rules use it.
    let begin = Stmt::Effect {
        dst: None,
        call: Expr::Call(&BEGIN, Vec::new()),
        guard: Guard::Always,
    };
    match func.body.first_mut() {
        Some(Node::Block(stmts)) => stmts.insert(0, begin),
        _ => func.body.insert(0, Node::Block(vec![begin])),
    }
    Ok(())
}

fn component(f: &Field, c: u8) -> &'static str {
    if f.comps <= 1 {
        ""
    } else {
        [".x", ".y", ".z", ".w"][usize::from(c & 3)]
    }
}

/// Wraps the printed function for one primitive into the entry point of the geometry shader.
pub fn entry(program: &StageProgram, iface: &Interface, text: String) -> Result<String, Refusal> {
    let bug = |what: &str| refuse("geometry-entry", what.to_owned());
    let ring = Ring::of(program)?;
    let head = format!("\nDecantOut {}(", crate::hlsl::ENTRY);
    if text.matches(&head).count() != 1 {
        return Err(bug("the printed text has no primitive function to wrap"));
    }
    let at = text.find(&head).expect("counted above") + head.len();
    let params = text[at..]
        .split_once(") {")
        .map(|(params, _)| params)
        .ok_or_else(|| bug("the primitive function has no parameter list"))?;
    // The function takes one input vertex, which it never reads: its inputs are in the ring.
    let argument = match params {
        "" => "",
        "DecantIn decant_in" => "decant_in[0]",
        _ => {
            return Err(bug(
                "the primitive function takes something else than a vertex",
            ));
        }
    };
    let mut out = text.replacen(&head, &format!("\nDecantOut {PRIMITIVE}("), 1);
    let mut lines = vec![
        format!("[maxvertexcount({})]", ring.slots),
        format!(
            "void {}(triangle DecantIn decant_in[{}], inout {}<DecantOut> decant_stream) {{",
            crate::hlsl::ENTRY,
            ring.corners,
            ring.stream
        ),
    ];
    // The input vertices go to the places `prologue` named: member component `c` of the element
    // in register `r` is dword `4r + first component + c` of its vertex.
    for corner in 0..ring.corners {
        for (m, f) in iface.inputs.iter().enumerate() {
            let Some(own) = program
                .inputs
                .iter()
                .find(|e| field(e).semantic == f.semantic)
            else {
                continue;
            };
            for c in 0..f.comps {
                let dword = corner * ring.input + 4 * own.register + u32::from(f.first_comp + c);
                let value = format!("decant_in[{corner}].i{m}{}", component(f, c));
                let bits = match f.component_type {
                    1 => value,
                    2 => format!("uint({value})"),
                    _ => format!("asuint({value})"),
                };
                lines.push(format!("    decant_lds[{dword}] = {bits};"));
            }
        }
    }
    lines.push(format!("    {PRIMITIVE}({argument});"));
    lines.push(format!(
        "    for (uint decant_k = 0u; decant_k < min(decant_gs_count, {}u); decant_k++) {{",
        ring.slots
    ));
    lines.push(format!(
        "        uint decant_b = {}u + decant_k * {}u;",
        ring.first_slot(),
        ring.output
    ));
    lines.push("        DecantOut decant_v = (DecantOut)0;".to_owned());
    for (m, (f, e)) in iface
        .outputs
        .iter()
        .zip(ordered(&program.outputs))
        .enumerate()
    {
        for c in 0..f.comps {
            let word = format!(
                "decant_lds[decant_b + {}u]",
                4 * e.register + u32::from(f.first_comp + c)
            );
            let value = match f.component_type {
                1 => word,
                2 => format!("int({word})"),
                _ => format!("asfloat({word})"),
            };
            lines.push(format!("        decant_v.o{m}{} = {value};", component(f, c)));
        }
    }
    lines.push("        decant_stream.Append(decant_v);".to_owned());
    lines.push(
        "        if (((decant_gs_cuts >> decant_k) & 1u) != 0u) decant_stream.RestartStrip();".to_owned(),
    );
    lines.push("    }".to_owned());
    lines.push("}".to_owned());
    for l in lines {
        out.push_str(&l);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{GeometryInfo, Stage};
    use crate::ir::Load;

    fn element(semantic: &str, index: u32, sv: u32, register: u32, mask: u8) -> SigElement {
        SigElement {
            semantic: semantic.to_owned(),
            index,
            system_value: sv,
            component_type: 3,
            register,
            mask,
            rw_mask: 0,
        }
    }

    fn program() -> StageProgram {
        StageProgram {
            inputs: vec![
                element("SV_Position", 0, 1, 0, 15),
                element("TEXCOORD", 0, 0, 1, 3),
            ],
            outputs: (0..5)
                .map(|r| element("TEXCOORD", r, u32::from(r == 0), r, 15))
                .collect(),
            ..StageProgram::new(Stage::Geometry, Vec::new())
        }
    }

    #[test]
    fn the_ring_holds_the_input_vertices_and_then_the_output_slots() {
        let ring = Ring::of(&program()).unwrap();
        assert_eq!((ring.input, ring.output, ring.slots), (8, 20, 3));
        assert_eq!(ring.first_slot(), 24);
        assert_eq!(ring.stream, "TriangleStream");
        // The entry registers name those places.
        let p = prologue(&program());
        let value = |dst: Var| {
            p.iter().find_map(|s| match s {
                Stmt::Set {
                    dst: d,
                    value: Expr::U32(v),
                    ..
                } if *d == dst => Some(*v),
                _ => None,
            })
        };
        assert_eq!(value(Var::V(0)), Some(8 << 16));
        assert_eq!(value(Var::V(1)), Some(16));
        assert_eq!(value(Var::S(2)), Some(96));
    }

    #[test]
    fn what_the_container_says_decides_the_stream() {
        let with = |input_primitive, output_topology, stream_mask, max_vertices| StageProgram {
            geometry: Some(GeometryInfo {
                input_primitive,
                output_topology,
                stream_mask,
                max_vertices,
            }),
            ..program()
        };
        assert_eq!(Ring::of(&with(3, 3, 1, 4)).unwrap().stream, "LineStream");
        assert_eq!(Ring::of(&with(3, 5, 1, 6)).unwrap().slots, 6);
        let key = |p: &StageProgram| check(p).unwrap_err().key();
        assert_eq!(key(&with(1, 5, 1, 3)), "stage:geometry-input-primitive");
        assert_eq!(key(&with(3, 4, 1, 3)), "stage:geometry-output-topology");
        assert_eq!(key(&with(3, 5, 3, 3)), "stage:geometry-streams");
        assert_eq!(key(&with(3, 5, 1, 64)), "stage:geometry-ring-size");
    }

    #[test]
    fn the_load_through_the_system_pointer_becomes_zero() {
        let set = |dst: Var, value: Expr| Stmt::Set {
            dst,
            value,
            guard: Guard::Always,
        };
        let load = |base: u8, offset: u32| {
            Expr::Load(Load {
                buffer: false,
                base,
                offset: Box::new(Expr::U32(offset)),
                pc: 7,
            })
        };
        let body = |offset: u32| {
            let mut stmts = prologue(&program());
            stmts.push(set(Var::S(0), Expr::Var(Var::S(8))));
            stmts.push(set(Var::Tmp(1), load(0, offset)));
            // A load through user data is not this stage's.
            stmts.push(set(Var::Tmp(2), load(16, SLOT_BASE_FIELD)));
            Function {
                body: vec![Node::Block(stmts)],
                resources: Vec::new(),
                groupshared_dwords: 0,
                lanes: 64,
            }
        };
        let mut func = body(SLOT_BASE_FIELD);
        prepare(&program(), &mut func).unwrap();
        let Node::Block(stmts) = &func.body[0] else {
            panic!("one block");
        };
        assert!(stmts.contains(&set(Var::Tmp(1), Expr::U32(0))));
        assert!(stmts.contains(&set(Var::Tmp(2), load(16, SLOT_BASE_FIELD))));
        let mut func = body(0x40);
        assert_eq!(
            prepare(&program(), &mut func).unwrap_err().key(),
            "stage:geometry-system-load"
        );
    }
}
