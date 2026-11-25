//! Tracks table sampler words through branches controlled by EXEC and saved lane masks.

use super::{
    Bin, Desc, Expr, Flow, Found, Source, StageProgram, State, Stmt, Sym, Tracer, Ty, Un, Var,
    forget_pair,
};
use crate::ir::{Guard, Node};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardedSampler {
    Table { sgpr: u8, offset: u32 },
    Immediate([u32; 4]),
}

pub(super) type Slots = HashMap<(u32, u8), GuardedSampler>;

const MAX_STATES: usize = 16_000;
const MAX_STATES_PER_BLOCK: usize = 128;
const MAX_EDGES: usize = 180_000;

fn truth(tracer: &Tracer, state: &State, expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Bool(b) => Some(*b),
        Expr::Var(v) => match state.get(v) {
            Some(Sym::Const(x)) => Some(*x != 0),
            _ => None,
        },
        Expr::Bin(op @ (Bin::Eq | Bin::Ne), x, y) => {
            let get = |e: &Expr| match e {
                Expr::U32(value) => Some(Sym::Const(*value)),
                Expr::Var(v) => state.get(v).copied(),
                _ => None,
            };
            let equal = if x == y {
                true
            } else {
                match (get(x)?, get(y)?) {
                    (Sym::Const(a), Sym::Const(b)) => a == b,
                    (a @ Sym::Word { desc, .. }, b)
                        if a == b
                            && matches!(tracer.sources[desc as usize], Source::Slot { .. }) =>
                    {
                        true
                    }
                    _ => return None,
                }
            };
            Some(equal == (*op == Bin::Eq))
        }
        Expr::Un(Un::LNot, x) => Some(!truth(tracer, state, x)?),
        Expr::Bin(Bin::LAnd, x, y) => match (truth(tracer, state, x), truth(tracer, state, y)) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), y) => y,
            (x, Some(true)) => x,
            _ => None,
        },
        Expr::Bin(Bin::LOr, x, y) => match (truth(tracer, state, x), truth(tracer, state, y)) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), y) => y,
            (x, Some(false)) => x,
            _ => None,
        },
        Expr::Bin(Bin::LXor, x, y) => Some(truth(tracer, state, x)? ^ truth(tracer, state, y)?),
        Expr::Select(c, x, y) => truth(tracer, state, if truth(tracer, state, c)? { x } else { y }),
        _ => None,
    }
}

fn constrain(tracer: &Tracer, state: &mut State, expr: &Expr, value: bool) -> bool {
    if let Some(known) = truth(tracer, state, expr) {
        return known == value;
    }
    match expr {
        Expr::Var(v) if v.ty() == Ty::Bool => {
            state.insert(*v, Sym::Const(u32::from(value)));
        }
        Expr::Un(Un::LNot, x) => return constrain(tracer, state, x, !value),
        _ => {}
    }
    true
}

fn live_flags(flow: &Flow<'_>) -> Vec<HashSet<Var>> {
    let dependencies: Vec<_> = flow
        .blocks
        .iter()
        .map(|block| {
            let mut reads = HashSet::new();
            let mut written = HashSet::new();
            for stmt in block.stmts.iter().flat_map(|stmts| stmts.iter()) {
                crate::resource::stmt_reads(stmt, &mut |v| {
                    if v.ty() == Ty::Bool && !written.contains(&v) {
                        reads.insert(v);
                    }
                });
                if let Stmt::Set {
                    dst,
                    guard: Guard::Always,
                    ..
                } = stmt
                    && dst.ty() == Ty::Bool
                {
                    written.insert(*dst);
                }
            }
            if let Some((cond, _)) = block.condition {
                crate::resource::reads(cond, &mut |v| {
                    reads.insert(v);
                });
            }
            (reads, written)
        })
        .collect();
    let mut live = vec![HashSet::new(); flow.blocks.len()];
    loop {
        let mut changed = false;
        for (i, block) in flow.blocks.iter().enumerate().rev() {
            let (reads, written) = &dependencies[i];
            let mut needed = reads.clone();
            for &next in &block.succs {
                needed.extend(live[next].difference(written).copied());
            }
            changed |= needed != live[i];
            live[i] = needed;
        }
        if !changed {
            return live;
        }
    }
}

fn partition(state: &State) -> Vec<(Var, Sym)> {
    let mut parts: Vec<_> = state
        .iter()
        .filter(|(v, _)| v.ty() == Ty::Bool)
        .map(|(v, value)| (*v, *value))
        .collect();
    parts.sort_unstable_by_key(|(v, _)| *v);
    parts
}

/// A sample can bind a static slot when every feasible active path supplies the same four words.
pub(super) fn sampler_slots(body: &mut [Node], program: &StageProgram, lanes: u8) -> Slots {
    let flow = Flow::build(body, true);
    let mut resources = Vec::new();
    let mut tracer = Tracer::new(program, &mut resources, lanes, false, &flow);

    let live = live_flags(&flow);
    // Keep states with different live mask values separate at a join. Within each
    // partition, keep only the register facts that agree on every incoming edge.
    let mut entries = vec![HashMap::<Vec<(Var, Sym)>, State>::new(); flow.blocks.len()];
    let mut pending = VecDeque::from([(0, State::default())]);
    let mut count = 0;
    let mut partitions = 0;
    let mut samples = HashMap::<(u32, u8), Option<GuardedSampler>>::new();
    while let Some((block, mut incoming)) = pending.pop_front() {
        count += 1;
        if count > MAX_EDGES || partitions > MAX_STATES {
            return HashMap::new();
        }
        let node = &flow.blocks[block];
        if let Some((cond, want)) = node.condition
            && !constrain(&tracer, &mut incoming, cond, want)
        {
            continue;
        }
        incoming.retain(|v, _| v.ty() != Ty::Bool || live[block].contains(v));
        let parts = partition(&incoming);
        let states = &mut entries[block];
        if let Some(old) = states.get_mut(&parts) {
            let n = old.len();
            old.retain(|v, sym| incoming.get(v) == Some(sym));
            if old.len() == n {
                continue;
            }
            incoming = old.clone();
        } else {
            if states.len() >= MAX_STATES_PER_BLOCK {
                return HashMap::new();
            }
            states.insert(parts, incoming.clone());
            partitions += 1;
        }
        let mut active = vec![incoming];
        for stmt in node.stmts.iter().flat_map(|s| s.iter()) {
            let guard = match stmt {
                Stmt::Set { guard, .. }
                | Stmt::Image(_, guard)
                | Stmt::Buffer(_, guard)
                | Stmt::Effect { guard, .. } => *guard,
                _ => Guard::Always,
            };
            let exec = Expr::Var(Var::Exec);
            if guard == Guard::Exec {
                let mut disabled = Vec::new();
                for unknown in active.iter_mut().filter(|s| !s.contains_key(&Var::Exec)) {
                    let mut other = unknown.clone();
                    other.insert(Var::Exec, Sym::Const(0));
                    unknown.insert(Var::Exec, Sym::Const(1));
                    disabled.push(other);
                }
                active.extend(disabled);
                if active.len() > MAX_STATES_PER_BLOCK {
                    return HashMap::new();
                }
            }
            for state in &mut active {
                if guard == Guard::Exec && truth(&tracer, state, &exec) == Some(false) {
                    continue;
                }
                match stmt {
                    Stmt::Set { dst, value, .. } if dst.ty() == Ty::Bool => {
                        if let Some(value) = truth(&tracer, state, value) {
                            state.insert(*dst, Sym::Const(u32::from(value)));
                            forget_pair(state, *dst, lanes);
                        } else {
                            tracer.step(state, stmt, block, &mut 0);
                        }
                    }
                    Stmt::Image(op, _) => {
                        if let Some(Desc::Regs(first)) = op.sampler {
                            let slot = match tracer.found(state, first, 4, op.pc) {
                                Ok(Found::Slot { sgpr, byte }) => {
                                    Some(GuardedSampler::Table { sgpr, offset: byte })
                                }
                                Ok(Found::Immediate(words)) => {
                                    Some(GuardedSampler::Immediate(words))
                                }
                                _ => None,
                            };
                            samples
                                .entry((op.pc, first))
                                .and_modify(|agreed| {
                                    if *agreed != slot {
                                        *agreed = None;
                                    }
                                })
                                .or_insert(slot);
                        }
                        for r in op.data_writes() {
                            state.remove(&Var::V(r));
                        }
                    }
                    Stmt::Buffer(op, _) if !op.store => {
                        for r in op.data..op.data + op.count {
                            state.remove(&Var::V(r));
                        }
                    }
                    _ => tracer.step(state, stmt, block, &mut 0),
                }
            }
        }
        for mut state in active {
            state.retain(|v, _| !matches!(v, Var::Tmp(_) | Var::Flag(_)));
            for &next in &node.succs {
                pending.push_back((next, state.clone()));
            }
        }
    }
    samples
        .into_iter()
        .filter_map(|(site, slot)| Some((site, slot?)))
        .collect()
}
