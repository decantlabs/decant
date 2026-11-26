//! Relative VGPR moves with finite M0 analysis and a complete register fallback.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use super::{Lifter, operand};
use crate::Stage;
use crate::cfg::{Cfg, Cond, Term};
use crate::decode::{Insn, Operand, Reg};
use crate::exec::Masks;
use crate::ir::{Bin, Expr, Guard, Stmt, Ty, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

type State = BTreeMap<Var, u32>;
pub(super) type Offsets = BTreeMap<u32, Option<BTreeSet<u32>>>;

fn is_relative(insn: &Insn) -> bool {
    matches!(insn.op, Op::v_movrels_b32 | Op::v_movreld_b32)
}

fn scalar(var: Var) -> bool {
    matches!(var, Var::S(_) | Var::Ttmp(_) | Var::M0 | Var::Scc)
}

enum Step {
    Relative(u32),
    Unknown,
    Instruction(Vec<Stmt>),
}

fn evaluate(value: &Expr, state: &State) -> Option<u32> {
    fn substitute(value: &mut Expr, state: &State) {
        match value {
            Expr::Var(var) => {
                if let Some(&word) = state.get(var) {
                    *value = if var.ty() == Ty::Bool {
                        Expr::Bool(word != 0)
                    } else {
                        Expr::U32(word)
                    };
                }
            }
            Expr::Un(_, arg) => substitute(arg, state),
            Expr::Bin(_, a, b) => {
                substitute(a, state);
                substitute(b, state);
            }
            Expr::Select(cond, a, b) => {
                substitute(cond, state);
                substitute(a, state);
                substitute(b, state);
            }
            _ => {}
        }
    }

    let mut value = value.clone();
    substitute(&mut value, state);
    crate::hlsl::simplify::fold(&mut value);
    match value {
        Expr::U32(word) => Some(word),
        Expr::Bool(flag) => Some(u32::from(flag)),
        _ => None,
    }
}

/// Keep separate states at joins so a scalar counter and its exit comparison stay correlated.
/// A missing value is unknown. Bound the analysis itself for programs with unbounded counters.
pub(super) fn analyse(graph: &Cfg, stage: Stage, masks: &Masks) -> Offsets {
    const MAX_STATES_PER_BLOCK: usize = 128;
    let mut offsets: Offsets = graph
        .insns()
        .filter(|insn| is_relative(insn))
        .map(|insn| (insn.pc, Some(BTreeSet::new())))
        .collect();
    if offsets.is_empty() {
        return offsets;
    }

    let mut scratch = Lifter::new(stage, masks);
    let mut steps = Vec::with_capacity(graph.blocks.len());
    for block in &graph.blocks {
        let mut block_steps = Vec::new();
        for insn in &block.insns {
            if is_relative(insn) {
                block_steps.push(Step::Relative(insn.pc));
                continue;
            }
            scratch.next_tmp = 0;
            if super::lift(&mut scratch, insn).is_err() {
                // The real lift reports the unsupported instruction. Forget any scalar
                // facts it could have changed when counting independent lifter gaps.
                scratch.take();
                block_steps.push(Step::Unknown);
                continue;
            }
            let stmts = scratch.take();
            if stmts.iter().any(|s| {
                matches!(s,
                    Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } if scalar(*dst)
                )
            }) {
                block_steps.push(Step::Instruction(stmts));
            }
        }
        steps.push(block_steps);
    }

    let mut reached = vec![HashSet::new(); steps.len()];
    let mut pending = VecDeque::from([(0, State::new())]);
    while let Some((block, mut state)) = pending.pop_front() {
        if !reached[block].insert(state.clone()) {
            continue;
        }
        if reached[block].len() > MAX_STATES_PER_BLOCK {
            return Offsets::new();
        }
        for step in &steps[block] {
            match step {
                Step::Unknown => state.clear(),
                Step::Relative(pc) => {
                    let entry = offsets.get_mut(pc).expect("relative instruction");
                    if let (Some(values), Some(&m0)) = (entry.as_mut(), state.get(&Var::M0)) {
                        values.insert(m0);
                    } else {
                        *entry = None;
                    }
                }
                Step::Instruction(stmts) => {
                    for stmt in stmts {
                        let (dst, value) = match stmt {
                            Stmt::Set {
                                dst,
                                value,
                                guard: Guard::Always,
                            } => (*dst, evaluate(value, &state)),
                            Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } => {
                                (*dst, None)
                            }
                            _ => continue,
                        };
                        if let Some(value) = value {
                            state.insert(dst, value);
                        } else {
                            state.remove(&dst);
                        }
                    }
                    state.retain(|var, _| scalar(*var));
                }
            }
        }
        match graph.blocks[block].term {
            Term::End => {}
            Term::Jump(target) => pending.push_back((target, state)),
            Term::Branch { cond, taken, fall } => {
                let choice = match cond {
                    Cond::Scc(want) => state.get(&Var::Scc).map(|&scc| (scc != 0) == want),
                    _ => None,
                };
                match choice {
                    Some(true) => pending.push_back((taken, state)),
                    Some(false) => pending.push_back((fall, state)),
                    None => {
                        pending.push_back((taken, state.clone()));
                        pending.push_back((fall, state));
                    }
                }
            }
        }
    }
    offsets
}

fn relative_read(registers: &[(u32, Var)]) -> Expr {
    if let [(_, reg)] = registers {
        return Expr::Var(*reg);
    }
    let (lower, upper) = registers.split_at(registers.len() / 2);
    Expr::select(
        Expr::bin(Bin::ULt, Expr::Var(Var::M0), Expr::U32(upper[0].0)),
        relative_read(lower),
        relative_read(upper),
    )
}

pub(super) fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    if !is_relative(insn) {
        return Ok(false);
    }
    operand::plain(insn)?;
    operand::no_float_mods(insn)?;

    let read = insn.op == Op::v_movrels_b32;
    let base = if read { insn.src[0] } else { insn.dst };
    let Operand::Reg(Reg::V(base)) = base else {
        return Err(operand::form(insn, "relative-vgpr"));
    };
    let unknown = || operand::form(insn, "relative-m0-range");
    let dynamic;
    let offsets = match cx.relative.get(&insn.pc).and_then(Option::as_ref) {
        Some(offsets) => offsets,
        None => {
            // Enumerate every signed offset that reaches an architectural VGPR.
            dynamic = (0..=255u32)
                .map(|reg| reg.wrapping_sub(u32::from(base)))
                .collect::<BTreeSet<_>>();
            &dynamic
        }
    };
    let registers = offsets
        .iter()
        .map(|&offset| {
            i32::from(base)
                .checked_add(offset as i32)
                .and_then(|reg| u8::try_from(reg).ok())
                .map(|reg| (offset, Var::V(reg)))
                .ok_or_else(unknown)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if registers.is_empty() {
        // The scalar analysis proved this instruction unreachable.
        return Ok(true);
    }
    // A register that parks scalars holds no lane value of an array (`exec::Parking`). When the
    // scalar analysis says the move reaches one, the program is not what that rests on.
    if cx.relative.get(&insn.pc).and_then(Option::as_ref).is_some()
        && registers
            .iter()
            .any(|(_, reg)| matches!(reg, Var::V(v) if cx.masks.parks_scalars(*v)))
    {
        return Err(operand::form(insn, "relative-parking"));
    }
    if read {
        let dst = operand::var(insn, insn.dst)?;
        let valid = Expr::bin(
            Bin::ULt,
            Expr::bin(Bin::Add, Expr::U32(u32::from(base)), Expr::Var(Var::M0)),
            Expr::U32(256),
        );
        let value = relative_read(&registers);
        let value = if cx.relative.get(&insn.pc).and_then(Option::as_ref).is_some() {
            value
        } else {
            Expr::select(valid, value, Expr::Var(dst))
        };
        cx.set_lane(dst, value);
    } else {
        let source = operand::bits(insn, 0)?;
        if let [(_, reg)] = registers.as_slice() {
            cx.set_lane(*reg, source);
        } else {
            // A possible destination can alias the source. Read it before writing any VGPR.
            let saved = cx.tmp();
            cx.set_lane(saved, source);
            for (offset, reg) in registers {
                cx.set_lane(
                    reg,
                    Expr::select(
                        Expr::bin(Bin::Eq, Expr::Var(Var::M0), Expr::U32(offset)),
                        Expr::Var(saved),
                        Expr::Var(reg),
                    ),
                );
            }
        }
    }
    Ok(true)
}
