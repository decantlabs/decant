//! Tracks static descriptor origins and scalar table bases through branches controlled by
//! structured control flow flags, loop selectors, EXEC, and saved lane masks.

use super::{
    Bin, Desc, Expr, Flow, Found, Source, StageProgram, State, Stmt, Sym, Tracer, Ty, Un, Var,
    forget_pair,
};
use crate::ir::{Guard, Node};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardedDesc {
    Table { sgpr: u8, offset: u32 },
    Immediate([u32; 4]),
    RootView { sgpr: u8 },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GuardedBase {
    Table(u8),
    Root(u8),
}

#[derive(Default)]
pub(super) struct Guarded {
    pub(super) descs: HashMap<(u32, u8, u8), GuardedDesc>,
    pub(super) bases: HashMap<u32, GuardedBase>,
}

#[derive(Default)]
pub(super) struct Needed {
    pub(super) descs: HashSet<(u32, u8, u8)>,
    pub(super) bases: HashSet<u32>,
}

impl Needed {
    pub(super) fn is_empty(&self) -> bool {
        self.descs.is_empty() && self.bases.is_empty()
    }

    fn covered_by(&self, g: &Guarded) -> bool {
        self.descs.iter().all(|k| g.descs.contains_key(k))
            && self.bases.iter().all(|k| g.bases.contains_key(k))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Only track synthetic flags and selectors inserted by the structure pass.
    StructureOnly,
    /// Track all live boolean masks, EXEC guards, and structured selectors.
    Full,
}

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

fn constrain(
    tracer: &Tracer,
    state: &mut State,
    expr: &Expr,
    value: bool,
    tracked: &HashMap<Var, usize>,
) -> bool {
    if let Some(known) = truth(tracer, state, expr) {
        return known == value;
    }
    match expr {
        Expr::Var(v) if v.ty() == Ty::Bool && tracked.contains_key(v) => {
            state.insert(*v, Sym::Const(u32::from(value)));
        }
        Expr::Un(Un::LNot, x) => return constrain(tracer, state, x, !value, tracked),
        Expr::Bin(Bin::Eq, a, b) if value => match (&**a, &**b) {
            (Expr::Var(v), Expr::U32(k)) | (Expr::U32(k), Expr::Var(v))
                if tracked.contains_key(v) =>
            {
                state.insert(*v, Sym::Const(*k));
            }
            _ => {}
        },
        Expr::Bin(Bin::Ne, a, b) if !value => match (&**a, &**b) {
            (Expr::Var(v), Expr::U32(k)) | (Expr::U32(k), Expr::Var(v))
                if tracked.contains_key(v) =>
            {
                state.insert(*v, Sym::Const(*k));
            }
            _ => {}
        },
        _ => {}
    }
    true
}

/// Synthetic `structure` control variables (`Var::Flag` and `Var::Tmp`) that enclose any
/// unresolved descriptor or base site in `needed`, closed under enclosing conditions of
/// their own writes. Keeping only relevant flags prevents state explosion on large shaders
/// with hundreds of unrelated multi-level jump flags.
fn needed_synthetic(body: &[Node], needed: &Needed) -> HashSet<Var> {
    fn walk_sites(nodes: &[Node], stack: &mut Vec<Var>, needed: &Needed, out: &mut HashSet<Var>) {
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    for s in stmts {
                        let hit = match s {
                            Stmt::Image(op, _) => {
                                matches!(op.image, Desc::Regs(f) if needed.descs.contains(&(op.pc, f, 8)))
                                    || matches!(op.sampler, Some(Desc::Regs(f)) if needed.descs.contains(&(op.pc, f, 4)))
                            }
                            Stmt::Buffer(op, _) => {
                                matches!(op.buffer, Desc::Regs(f) if needed.descs.contains(&(op.pc, f, 4)))
                            }
                            Stmt::Set {
                                value: Expr::Load(load),
                                ..
                            } => {
                                if load.buffer {
                                    needed.descs.contains(&(load.pc, load.base, 4))
                                } else {
                                    needed.bases.contains(&load.pc)
                                }
                            }
                            _ => false,
                        };
                        if hit {
                            out.extend(stack.iter().copied());
                        }
                    }
                }
                Node::If { cond, then, els } => {
                    let before = stack.len();
                    crate::resource::reads(cond, &mut |v| {
                        if matches!(v, Var::Tmp(_) | Var::Flag(_)) {
                            stack.push(v);
                        }
                    });
                    walk_sites(then, stack, needed, out);
                    walk_sites(els, stack, needed, out);
                    stack.truncate(before);
                }
                Node::Loop(inner) | Node::Once(inner) => walk_sites(inner, stack, needed, out),
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
    }
    fn walk_writes(nodes: &[Node], stack: &mut Vec<Var>, out: &mut HashSet<Var>) -> bool {
        let mut changed = false;
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    for s in stmts {
                        if let Stmt::Set { dst, .. } = s
                            && out.contains(dst)
                        {
                            for &v in stack.iter() {
                                changed |= out.insert(v);
                            }
                        }
                    }
                }
                Node::If { cond, then, els } => {
                    let before = stack.len();
                    crate::resource::reads(cond, &mut |v| {
                        if matches!(v, Var::Tmp(_) | Var::Flag(_)) {
                            stack.push(v);
                        }
                    });
                    changed |= walk_writes(then, stack, out);
                    changed |= walk_writes(els, stack, out);
                    stack.truncate(before);
                }
                Node::Loop(inner) | Node::Once(inner) => {
                    changed |= walk_writes(inner, stack, out);
                }
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
        changed
    }
    let mut out = HashSet::new();
    let mut stack = Vec::new();
    walk_sites(body, &mut stack, needed, &mut out);
    while walk_writes(body, &mut stack, &mut out) {}
    out
}

fn collect_tracked(
    flow: &Flow<'_>,
    mode: Mode,
    focus: Option<&HashSet<Var>>,
) -> (HashMap<Var, usize>, HashSet<Var>) {
    let mut synth = HashSet::new();
    let mut index = HashMap::new();
    let mut add = |v: Var| {
        let next = index.len();
        index.entry(v).or_insert(next);
    };
    for block in &flow.blocks {
        if let Some((cond, _)) = block.condition {
            crate::resource::reads(cond, &mut |v| {
                if matches!(v, Var::Tmp(_) | Var::Flag(_)) {
                    if focus.is_none_or(|f| f.contains(&v)) {
                        synth.insert(v);
                        add(v);
                    }
                } else if mode == Mode::Full && v.ty() == Ty::Bool {
                    add(v);
                }
            });
        }
        if mode == Mode::Full {
            for stmt in block.stmts.iter().flat_map(|s| s.iter()) {
                match stmt {
                    Stmt::Set { dst, value, guard } => {
                        if *guard == Guard::Exec {
                            add(Var::Exec);
                        }
                        if dst.ty() == Ty::Bool {
                            add(*dst);
                            crate::resource::reads(value, &mut |v| {
                                if v.ty() == Ty::Bool {
                                    add(v);
                                }
                            });
                        }
                    }
                    Stmt::Image(_, Guard::Exec)
                    | Stmt::Buffer(_, Guard::Exec)
                    | Stmt::Effect {
                        guard: Guard::Exec, ..
                    } => add(Var::Exec),
                    _ => {}
                }
            }
        }
    }
    (index, synth)
}

/// Bitset liveness for tracked control variables at each block entry.
fn live_controls(
    flow: &Flow<'_>,
    mode: Mode,
    tracked: &HashMap<Var, usize>,
) -> (Vec<Vec<u64>>, usize) {
    let words = tracked.len().div_ceil(64);
    let n = flow.blocks.len();
    if words == 0 {
        return (vec![Vec::new(); n], 0);
    }
    let set_bit = |bits: &mut [u64], id: usize| bits[id / 64] |= 1u64 << (id % 64);
    let clear_bit = |bits: &mut [u64], id: usize| bits[id / 64] &= !(1u64 << (id % 64));
    let has_bit = |bits: &[u64], id: usize| (bits[id / 64] >> (id % 64)) & 1 != 0;

    let exec_id = tracked.get(&Var::Exec).copied();
    let mut live = vec![vec![0u64; words]; n];
    let mut cur = vec![0u64; words];
    loop {
        let mut changed = false;
        for (i, block) in flow.blocks.iter().enumerate().rev() {
            cur.fill(0);
            for &next in &block.succs {
                for (dst, &src) in cur.iter_mut().zip(&live[next]) {
                    *dst |= src;
                }
            }
            if let Some(stmts) = block.stmts.as_ref() {
                for stmt in stmts.iter().rev() {
                    if mode == Mode::Full
                        && let Some(eid) = exec_id
                        && matches!(
                            stmt,
                            Stmt::Set {
                                guard: Guard::Exec,
                                ..
                            } | Stmt::Image(_, Guard::Exec)
                                | Stmt::Buffer(_, Guard::Exec)
                                | Stmt::Effect {
                                    guard: Guard::Exec,
                                    ..
                                }
                        )
                    {
                        set_bit(&mut cur, eid);
                    }
                    if let Stmt::Set {
                        dst, value, guard, ..
                    } = stmt
                        && let Some(&did) = tracked.get(dst)
                        && has_bit(&cur, did)
                    {
                        if *guard == Guard::Always {
                            clear_bit(&mut cur, did);
                        }
                        crate::resource::reads(value, &mut |v| {
                            if let Some(&vid) = tracked.get(&v) {
                                set_bit(&mut cur, vid);
                            }
                        });
                    }
                }
            }
            if let Some((cond, _)) = block.condition {
                crate::resource::reads(cond, &mut |v| {
                    if let Some(&vid) = tracked.get(&v) {
                        set_bit(&mut cur, vid);
                    }
                });
            }
            if cur != live[i] {
                live[i].copy_from_slice(&cur);
                changed = true;
            }
        }
        if !changed {
            return (live, words);
        }
    }
}

fn partition(state: &State, tracked: &HashMap<Var, usize>) -> Vec<(Var, Sym)> {
    let mut parts: Vec<_> = state
        .iter()
        .filter(|(v, _)| tracked.contains_key(v))
        .map(|(v, value)| (*v, *value))
        .collect();
    parts.sort_unstable_by_key(|(v, _)| *v);
    parts
}

fn record_desc(
    found_map: &mut HashMap<(u32, u8, u8), Option<GuardedDesc>>,
    tracer: &Tracer,
    state: &State,
    pc: u32,
    first: u8,
    words: u8,
) {
    let slot = match tracer.found(state, first, words, pc) {
        Ok(Found::Slot { sgpr, byte }) => Some(GuardedDesc::Table { sgpr, offset: byte }),
        Ok(Found::Immediate(imm)) if words == 4 => Some(GuardedDesc::Immediate(imm)),
        Ok(Found::RootView { sgpr }) if words == 4 => Some(GuardedDesc::RootView { sgpr }),
        _ => None,
    };
    found_map
        .entry((pc, first, words))
        .and_modify(|agreed| {
            if *agreed != slot {
                *agreed = None;
            }
        })
        .or_insert(slot);
}

fn record_base(
    base_map: &mut HashMap<u32, Option<GuardedBase>>,
    state: &State,
    pc: u32,
    base: u8,
) {
    let lo = state.get(&Var::S(base)).copied();
    let hi = state.get(&Var::S(base + 1)).copied();
    let kind = match (lo, hi) {
        (Some(Sym::User(sgpr)), Some(Sym::Const(0))) => Some(GuardedBase::Table(sgpr)),
        (Some(Sym::User(sgpr)), Some(Sym::User(next))) if next == sgpr + 1 => {
            Some(GuardedBase::Root(sgpr))
        }
        _ => None,
    };
    base_map
        .entry(pc)
        .and_modify(|agreed| {
            if *agreed != kind {
                *agreed = None;
            }
        })
        .or_insert(kind);
}

fn run_mode(
    body: &mut [Node],
    program: &StageProgram,
    lanes: u8,
    mode: Mode,
    focus: Option<&HashSet<Var>>,
) -> Guarded {
    let flow = Flow::build(body, true);
    let mut resources = Vec::new();
    let mut tracer = Tracer::new(program, &mut resources, lanes, false, &flow);

    let (tracked, synth) = collect_tracked(&flow, mode, focus);
    if mode == Mode::StructureOnly && synth.is_empty() {
        return Guarded::default();
    }
    let (live, words) = live_controls(&flow, mode, &tracked);
    let is_live = |block: usize, v: Var| -> bool {
        let Some(&id) = tracked.get(&v) else {
            return true;
        };
        words > 0 && ((live[block][id / 64] >> (id % 64)) & 1 != 0)
    };
    let max_states = if mode == Mode::StructureOnly {
        MAX_STATES.max(flow.blocks.len().saturating_mul(8))
    } else {
        MAX_STATES
    };
    let max_edges = if mode == Mode::StructureOnly {
        MAX_EDGES.max(flow.blocks.len().saturating_mul(64))
    } else {
        MAX_EDGES
    };
    // Structure lowering flags are hoisted and initialised to `false` at function entry.
    let mut entry_state = State::default();
    for &v in &synth {
        if matches!(v, Var::Flag(_)) && is_live(0, v) {
            entry_state.insert(v, Sym::Const(0));
        }
    }
    // Keep states with different live control values separate at a join. Within each
    // partition, keep only the register facts that agree on every incoming edge.
    let mut entries = vec![HashMap::<Vec<(Var, Sym)>, State>::new(); flow.blocks.len()];
    let mut inbox: Vec<Vec<State>> = vec![Vec::new(); flow.blocks.len()];
    let mut queued = vec![false; flow.blocks.len()];
    let mut work = BinaryHeap::new();
    inbox[0].push(entry_state);
    queued[0] = true;
    work.push(Reverse(0usize));

    let mut count = 0;
    let mut partitions = 0;
    let mut found_map = HashMap::<(u32, u8, u8), Option<GuardedDesc>>::new();
    let mut base_map = HashMap::<u32, Option<GuardedBase>>::new();
    while let Some(Reverse(block)) = work.pop() {
        queued[block] = false;
        let batch = std::mem::take(&mut inbox[block]);
        let node = &flow.blocks[block];
        for mut incoming in batch {
            count += 1;
            if count > max_edges || partitions > max_states {
                return Guarded::default();
            }
            if let Some((cond, want)) = node.condition
                && !constrain(&tracer, &mut incoming, cond, want, &tracked)
            {
                continue;
            }
            incoming.retain(|v, _| is_live(block, *v));
            let parts = partition(&incoming, &tracked);
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
                    return Guarded::default();
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
                if mode == Mode::Full && guard == Guard::Exec {
                    let mut disabled = Vec::new();
                    for unknown in active.iter_mut().filter(|s| !s.contains_key(&Var::Exec)) {
                        let mut other = unknown.clone();
                        other.insert(Var::Exec, Sym::Const(0));
                        unknown.insert(Var::Exec, Sym::Const(1));
                        disabled.push(other);
                    }
                    active.extend(disabled);
                    if active.len() > MAX_STATES_PER_BLOCK {
                        return Guarded::default();
                    }
                }
                for state in &mut active {
                    if mode == Mode::Full
                        && guard == Guard::Exec
                        && truth(&tracer, state, &exec) == Some(false)
                    {
                        continue;
                    }
                    match stmt {
                        Stmt::Set { dst, value, .. }
                            if dst.ty() == Ty::Bool && tracked.contains_key(dst) =>
                        {
                            if let Some(value) = truth(&tracer, state, value) {
                                state.insert(*dst, Sym::Const(u32::from(value)));
                                forget_pair(state, *dst, lanes);
                            } else {
                                tracer.step(state, stmt, block, &mut 0);
                            }
                        }
                        Stmt::Set {
                            value: Expr::Load(load),
                            ..
                        } => {
                            if load.buffer {
                                record_desc(&mut found_map, &tracer, state, load.pc, load.base, 4);
                            } else {
                                record_base(&mut base_map, state, load.pc, load.base);
                            }
                            tracer.step(state, stmt, block, &mut 0);
                        }
                        Stmt::Image(op, _) => {
                            if let Desc::Regs(first) = op.image {
                                record_desc(&mut found_map, &tracer, state, op.pc, first, 8);
                            }
                            if let Some(Desc::Regs(first)) = op.sampler {
                                record_desc(&mut found_map, &tracer, state, op.pc, first, 4);
                            }
                            for r in op.data_writes() {
                                state.remove(&Var::V(r));
                            }
                        }
                        Stmt::Buffer(op, _) => {
                            if let Desc::Regs(first) = op.buffer {
                                record_desc(&mut found_map, &tracer, state, op.pc, first, 4);
                            }
                            if !op.store || (op.glc && op.op.name().contains("atomic")) {
                                for r in op.data..op.data.saturating_add(op.count) {
                                    state.remove(&Var::V(r));
                                }
                            }
                        }
                        _ => tracer.step(state, stmt, block, &mut 0),
                    }
                }
            }
            for mut state in active {
                state.retain(|v, _| !matches!(v, Var::Tmp(_) | Var::Flag(_)) || tracked.contains_key(v));
                for &next in &node.succs {
                    inbox[next].push(state.clone());
                    if !queued[next] {
                        queued[next] = true;
                        work.push(Reverse(next));
                    }
                }
            }
        }
    }
    Guarded {
        descs: found_map
            .into_iter()
            .filter_map(|(site, slot)| Some((site, slot?)))
            .collect(),
        bases: base_map
            .into_iter()
            .filter_map(|(pc, base)| Some((pc, base?)))
            .collect(),
    }
}

/// Resolves static descriptor origins and scalar table bases along feasible paths.
/// Starts with structured control flow flags and loop selectors alone (fast on large programs)
/// and falls back to full EXEC and mask tracking only when needed sites remain unresolved.
pub(super) fn solve(
    body: &mut [Node],
    program: &StageProgram,
    lanes: u8,
    needed: &Needed,
) -> Guarded {
    let focus = needed_synthetic(body, needed);
    let mut out = if focus.is_empty() {
        Guarded::default()
    } else {
        run_mode(body, program, lanes, Mode::StructureOnly, Some(&focus))
    };
    if !needed.covered_by(&out) {
        let full = run_mode(body, program, lanes, Mode::Full, None);
        for (k, v) in full.descs {
            out.descs.entry(k).or_insert(v);
        }
        for (k, v) in full.bases {
            out.bases.entry(k).or_insert(v);
        }
    }
    out
}
