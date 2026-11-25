//! The vertex stage.
//!
//! Inputs. A vertex program starts by calling a routine the console supplies at run time
//! (`s_swappc_b64` as its first instruction), which fetches the vertex inputs. Measured on the
//! title's programs: after the call, component `c` of the input element in signature register
//! `r` is in vector register `4 + 4r + c`, the vertex index is in `v0` and the instance index
//! in `v3`. The prologue puts those values into the registers and the call itself does nothing.
//!
//! Exports. Target 12 is the position. Target `32 + j` is parameter slot `j`. The parameter
//! export table of the container (`StageProgram::param_exports`, FSSM byte 0x1DC) names the
//! signature register that opens each slot. Further 16 bit registers follow in the open packed
//! slot by the rule both sides share (`pixel::walk`): two 16 bit registers can share a slot,
//! and two elements can share a register (a share of the vertex programs). Component `c` of
//! a register goes to the element of that register whose mask has bit `c`, and since struct
//! members are as wide as their register up to their last component, the member component is
//! `c`. A packed 16 bit register takes half `start + c` of the slot. A 16 bit register that is
//! not packed (record flags 1, the flat inputs of the pixel side) has the half float in the low
//! half of each channel. Without a container every output register opens its own slot, except
//! 16 bit registers that fit into the open packed slot.
//! The other position targets (13 to 15: point size, layer, viewport, clip distances) are not
//! written yet. A parameter export to a slot no register takes is dropped, since nothing reads
//! it.

use super::pixel::{Layout, Place, Walked, register_mask, walk};
use super::{Interface, field, ordered, refuse};
use crate::api::{SigElement, StageProgram};
use crate::ir::{Bin, Helper, Ty};

/// The system value codes of the vertex and instance index in a signature.
const SV_VERTEX_ID: u32 = 6;
const SV_INSTANCE_ID: u32 = 8;
use crate::ir::{Export, Expr, Function, Guard, Input, OutputRef, Stmt, Var};
use crate::refusal::Refusal;

pub fn interface(program: &StageProgram) -> Result<Interface, Refusal> {
    // The vertex and instance index are declared once by the printer when a program reads them,
    // so the signature's own elements for them stay out of the struct.
    let inputs: Vec<&SigElement> = ordered(&program.inputs)
        .into_iter()
        .filter(|e| !matches!(e.system_value, SV_VERTEX_ID | SV_INSTANCE_ID))
        .collect();
    let mut iface = Interface {
        inputs: inputs.iter().map(|e| field(e)).collect(),
        outputs: ordered(&program.outputs).iter().map(|e| field(e)).collect(),
        ..Interface::default()
    };
    for (i, e) in inputs.iter().enumerate() {
        if e.system_value == 0 {
            let register = u8::try_from(e.register)
                .map_err(|_| refuse("input-layout", "an input register above 255"))?;
            iface.vertex_inputs.insert(register, i);
        }
    }
    Ok(iface)
}

pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    let set = |dst: Var, input: Input| Stmt::Set {
        dst,
        value: Expr::Input(input),
        guard: Guard::Always,
    };
    let mut out = vec![
        Stmt::Set {
            dst: Var::Exec,
            value: Expr::Bool(true),
            guard: Guard::Always,
        },
        set(Var::V(0), Input::VertexId),
        set(Var::V(3), Input::InstanceId),
    ];
    for e in program
        .inputs
        .iter()
        .filter(|e| e.system_value == 0 && e.register < 63)
    {
        for chan in (0..4u8).filter(|c| (e.mask >> c) & 1 != 0) {
            out.push(set(
                Var::V(4 + 4 * e.register as u8 + chan),
                Input::VertexElement {
                    register: e.register as u8,
                    chan,
                },
            ));
        }
    }
    out
}

static F16: Helper = Helper {
    name: "decant_vertex_f16",
    result: Ty::U32,
    text: "uint decant_vertex_f16(uint a) { return asuint(f16tof32(a & 0xffffu)); }",
    needs: &[],
};

/// Where each output register lies in the parameter slots.
fn slots(program: &StageProgram) -> Vec<Place> {
    let mut registers: Vec<u32> = program
        .outputs
        .iter()
        .filter(|e| matches!(e.system_value, 0 | 2 | 3))
        .map(|e| e.register)
        .collect();
    registers.sort_unstable();
    registers.dedup();
    if !program.param_exports.is_empty() {
        return walk(registers.into_iter().map(|register| {
            let (mask, sixteen) = register_mask(&program.outputs, register);
            Walked {
                register,
                mask,
                sixteen,
                opens: program
                    .param_exports
                    .iter()
                    .find(|r| r.register == register)
                    .map(|r| (r.slot, r.flags & 3 == 3)),
            }
        }));
    }
    // Without the container's table: every register opens a slot, a 16 bit register joins the
    // open packed slot while it fits.
    let mut walked = Vec::new();
    let (mut next, mut open) = (0u8, None::<u8>);
    for register in registers {
        let (mask, sixteen) = register_mask(&program.outputs, register);
        let channels = (8 - mask.leading_zeros() as u8).div_ceil(2);
        let joins = sixteen && open.is_some_and(|used| used + channels <= 4);
        let opens = if joins {
            open = open.map(|used| used + channels);
            None
        } else {
            next += 1;
            if sixteen {
                open = Some(channels);
            }
            Some((next - 1, sixteen))
        };
        walked.push(Walked {
            register,
            mask,
            sixteen,
            opens,
        });
    }
    walk(walked)
}

/// The output element that holds component `c` of `register`, as an index of the ordered
/// outputs (the struct members).
fn output_at(outputs: &[&SigElement], register: u32, c: u8) -> Option<usize> {
    outputs.iter().position(|o| {
        o.register == register && (o.mask >> c) & 1 != 0 && matches!(o.system_value, 0 | 2 | 3)
    })
}

fn export(
    program: &StageProgram,
    slots: &[Place],
    e: &Export,
    out: &mut Vec<Stmt>,
) -> Result<(), Refusal> {
    let outputs = ordered(&program.outputs);
    if e.compressed {
        return Err(refuse("export-target", "a compressed vertex export").at(e.pc));
    }
    let word = |k: u8| Expr::Var(Var::V(e.src[usize::from(k)]));
    let enabled = |k: u8| (e.enabled >> k) & 1 != 0;
    if e.target == 12 {
        let element = outputs
            .iter()
            .position(|o| o.system_value == 1)
            .ok_or_else(|| {
                refuse("export-target", "a position export without SV_Position").at(e.pc)
            })?;
        let comps = (8 - outputs[element].mask.leading_zeros()) as u8;
        for chan in (0..comps).filter(|c| enabled(*c)) {
            out.push(Stmt::Output {
                to: OutputRef {
                    element: element as u16,
                    chan,
                },
                value: word(chan),
            });
        }
        return Ok(());
    }
    if e.target < 32 {
        return Err(refuse(
            "export-target",
            format!("vertex export to target {}", e.target),
        )
        .at(e.pc));
    }
    let slot = e.target - 32;
    for place in slots.iter().filter(|p| p.slot == slot) {
        for c in 0..4u8 {
            let Some(element) = output_at(&outputs, place.register, c) else {
                continue;
            };
            let value = match place.layout {
                Layout::Wide if enabled(c) => word(c),
                Layout::Unpacked if enabled(c) => Expr::Call(&F16, vec![word(c)]),
                Layout::Packed => {
                    let half = place.start + c;
                    if half >= 8 || !enabled(half / 2) {
                        continue;
                    }
                    let w = word(half / 2);
                    let w = if half % 2 == 1 {
                        Expr::bin(Bin::LShr, w, Expr::U32(16))
                    } else {
                        w
                    };
                    Expr::Call(&F16, vec![w])
                }
                _ => continue,
            };
            out.push(Stmt::Output {
                to: OutputRef {
                    element: element as u16,
                    chan: c,
                },
                value,
            });
        }
    }
    Ok(())
}

pub fn finish(
    program: &StageProgram,
    _iface: &Interface,
    func: &mut Function,
) -> Result<(), Refusal> {
    let slots = slots(program);
    let mut result = Ok(());
    func.for_each_block_mut(&mut |stmts| {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts.drain(..) {
            match s {
                Stmt::Export(e) => {
                    if let Err(r) = export(program, &slots, &e, &mut out) {
                        result = Err(r);
                    }
                }
                other => out.push(other),
            }
        }
        *stmts = out;
    });
    result
}
