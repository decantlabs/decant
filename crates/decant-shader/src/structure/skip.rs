//! Taking away a skip that stands around a derivative taking sample or a workgroup barrier.
//!
//! The machine jumps over masked code when no lane of the wave is active (`s_cbranch_execz`, or
//! `s_cbranch_scc0` right after a mask update that sets SCC from EXEC). One lane per invocation
//! makes that `if (exec) { .. }` around the code, or an early return when the skip targets
//! `s_endpgm`. For most code that is the same and saves work. Two cases break if the skip stays:
//!
//! * A sample that takes its level of detail from screen space derivatives: on the machine the
//!   inactive lanes of a quad run the sample along with the active ones and so give it their
//!   coordinates, while in HLSL a sample under a condition that differs inside a quad has no
//!   defined derivatives.
//! * A workgroup barrier (`s_barrier`): on the machine `s_cbranch_execz` only jumps when all
//!   lanes of a wave are inactive, and an early `s_endpgm` retires that wave from the barrier.
//!   In HLSL every thread of the thread group must reach `GroupMemoryBarrierWithGroupSync()`, so
//!   a per lane skip or early return around a barrier deadlocks the thread group.
//!
//! In both cases the skip is dropped: every lane runs the code, and only the active lanes write.
//! That is the same program when the skipped code does nothing observable in an inactive lane:
//!
//! * the code is entered through the skip only, and leaves to the target of the skip only (or to
//!   an empty end block when the target itself is an empty end block, for a barrier region),
//! * each of its statements is under the lane test, is a barrier or lane masked swizzle (for a
//!   barrier region), or is an assignment whose variable nothing reads once the code is left (a
//!   saved mask, a condition code, a scalar the code uses itself). Exports, messages and other
//!   effects that are not under the lane test keep the skip,
//! * the code never sets EXEC to something that can be true in a lane that came in inactive.
//!   Saving EXEC, narrowing it and restoring the saved value are all false in such a lane,
//! * for a barrier region that contains loops, every cycle reachable by an inactive lane exits
//!   only through scalar conditions that do not depend on EXEC or vector registers.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::graph::{Exit, Graph};
use crate::ir::{Bin, Expr, Guard, ImageKind, SampleLod, Stmt, Un, Var};

/// Whether `Var::Scc` holds `Var::Exec` (`Some(true)`) or `!Var::Exec` (`Some(false)`) at the
/// end of `stmts`.
fn scc_matches_exec(stmts: &[Stmt]) -> Option<bool> {
    let mut exec_alias: Option<Var> = None;
    let mut scc_eq_exec: Option<bool> = None;
    for s in stmts {
        match s {
            Stmt::Comment(_) => {}
            Stmt::Set {
                dst: Var::Exec,
                value,
                guard: Guard::Always,
            } => {
                exec_alias = match value {
                    Expr::Var(v) if *v != Var::Exec => Some(*v),
                    _ => None,
                };
                scc_eq_exec = match value {
                    Expr::Var(Var::Scc) => Some(true),
                    Expr::Un(Un::LNot, inner) if **inner == Expr::Var(Var::Scc) => Some(false),
                    _ => None,
                };
            }
            Stmt::Set {
                dst: Var::Scc,
                value,
                guard: Guard::Always,
            } => {
                scc_eq_exec = match value {
                    Expr::Var(Var::Exec) => Some(true),
                    Expr::Un(Un::LNot, inner) if **inner == Expr::Var(Var::Exec) => Some(false),
                    Expr::Var(v) if exec_alias == Some(*v) => Some(true),
                    Expr::Un(Un::LNot, inner)
                        if matches!(&**inner, Expr::Var(v) if exec_alias == Some(*v)) =>
                    {
                        Some(false)
                    }
                    _ => None,
                };
                if exec_alias == Some(Var::Scc) {
                    exec_alias = None;
                }
            }
            Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } => {
                if *dst == Var::Exec {
                    exec_alias = None;
                    scc_eq_exec = None;
                } else if *dst == Var::Scc {
                    scc_eq_exec = None;
                }
                if exec_alias == Some(*dst) {
                    exec_alias = None;
                }
            }
            _ => {}
        }
    }
    scc_eq_exec
}

/// The code a branch on this lane's EXEC bit skips, and where it skips to: `(entry, target)`
/// where an active lane enters `entry` and an inactive lane skips to `target`.
fn exec_skip(g: &Graph, branch: usize) -> Option<(usize, usize)> {
    let Exit::Branch { cond, taken, fall } = &g.exit[branch] else {
        return None;
    };
    match cond {
        Expr::Var(Var::Exec) => Some((*taken, *fall)),
        Expr::Un(Un::LNot, inner) if **inner == Expr::Var(Var::Exec) => Some((*fall, *taken)),
        Expr::Var(Var::Scc) => {
            let same = scc_matches_exec(&g.code[branch])?;
            Some(if same {
                (*taken, *fall)
            } else {
                (*fall, *taken)
            })
        }
        Expr::Un(Un::LNot, inner) if **inner == Expr::Var(Var::Scc) => {
            let same = scc_matches_exec(&g.code[branch])?;
            Some(if same {
                (*fall, *taken)
            } else {
                (*taken, *fall)
            })
        }
        _ => None,
    }
}

fn takes_derivatives(s: &Stmt) -> bool {
    match s {
        Stmt::Image(op, _) => matches!(
            op.kind,
            ImageKind::Sample {
                lod: SampleLod::Implicit | SampleLod::Bias,
                ..
            } | ImageKind::QueryLod
        ),
        _ => false,
    }
}

fn is_barrier(s: &Stmt) -> bool {
    matches!(s, Stmt::Barrier { .. })
}

/// Whether `node` ends the program without running any statement.
fn is_empty_end(g: &Graph, mut node: usize) -> bool {
    for _ in 0..8 {
        if !g.code[node].iter().all(|s| matches!(s, Stmt::Comment(_))) {
            return false;
        }
        match g.exit[node] {
            Exit::End => return true,
            Exit::Jump(next) if next != node => node = next,
            _ => return false,
        }
    }
    false
}

/// The most nodes a derivative skip region may have to be looked at.
const MAX_NODES: usize = 256;

/// The most nodes a barrier skip region may have to be looked at.
const MAX_BARRIER_NODES: usize = 4096;

/// The nodes from `entry` up to `target`, when they are a piece of code without a loop that is
/// entered at `entry` from `branch` only and left to `target` only.
fn region(
    g: &Graph,
    preds: &[Vec<usize>],
    branch: usize,
    entry: usize,
    target: usize,
) -> Option<Vec<usize>> {
    // 1 while a node is on the walk, 2 when it is done.
    let mut state: HashMap<usize, u8> = HashMap::new();
    let mut nodes = Vec::new();
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(entry, g.successors(entry).collect())];
    state.insert(entry, 1);
    if entry == target || entry == branch {
        return None;
    }
    while let Some((node, rest)) = stack.last_mut() {
        let node = *node;
        if matches!(g.exit[node], Exit::End) {
            return None;
        }
        let Some(next) = rest.pop() else {
            state.insert(node, 2);
            nodes.push(node);
            stack.pop();
            continue;
        };
        if next == target {
            continue;
        }
        match state.get(&next) {
            // Back to a node of the walk is a loop, back to the branch a loop around the skip.
            Some(1) => return None,
            Some(_) => {}
            None if next == branch || nodes.len() + stack.len() >= MAX_NODES => return None,
            None => {
                state.insert(next, 1);
                stack.push((next, g.successors(next).collect()));
            }
        }
    }
    let inside = |n: usize| state.contains_key(&n);
    let one_way_in = nodes.iter().all(|&n| {
        preds[n]
            .iter()
            .all(|&p| inside(p) || (p == branch && n == entry))
    });
    one_way_in.then_some(nodes)
}

/// The nodes from `entry` up to `target` for a barrier skip. Unlike [`region`], loops inside the
/// region are permitted (checked separately by [`loops_safe_when_inactive`]), and when `target` is
/// an empty end block other `Exit::End` blocks reached from `entry` are also treated as exits.
/// Returns the nodes in postorder and whether the region contains a cycle.
fn barrier_region(
    g: &Graph,
    preds: &[Vec<usize>],
    branch: usize,
    entry: usize,
    target: usize,
) -> Option<(Vec<usize>, bool)> {
    if entry == target || entry == branch {
        return None;
    }
    let target_is_end = is_empty_end(g, target);
    let mut state: HashMap<usize, u8> = HashMap::new();
    let mut nodes = Vec::new();
    let mut has_cycle = false;
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(entry, g.successors(entry).collect())];
    state.insert(entry, 1);
    while let Some((node, rest)) = stack.last_mut() {
        let node = *node;
        if matches!(g.exit[node], Exit::End) && !target_is_end {
            return None;
        }
        let Some(next) = rest.pop() else {
            state.insert(node, 2);
            nodes.push(node);
            stack.pop();
            continue;
        };
        if next == target {
            continue;
        }
        match state.get(&next) {
            Some(1) => has_cycle = true,
            Some(_) => {}
            None if next == branch || nodes.len() + stack.len() >= MAX_BARRIER_NODES => {
                return None;
            }
            None => {
                state.insert(next, 1);
                stack.push((next, g.successors(next).collect()));
            }
        }
    }
    let inside = |n: usize| state.contains_key(&n);
    let one_way_in = nodes.iter().all(|&n| {
        preds[n]
            .iter()
            .all(|&p| inside(p) || (p == branch && n == entry))
    });
    one_way_in.then_some((nodes, has_cycle))
}

/// Which variables a later statement can still read at the start of each node.
struct Live {
    index: HashMap<Var, usize>,
    at_start: Vec<Vec<u64>>,
}

impl Live {
    fn new(g: &Graph) -> Live {
        let mut index: HashMap<Var, usize> = HashMap::new();
        let mut number = |v: Var| {
            let next = index.len();
            *index.entry(v).or_insert(next)
        };
        // What a node reads before it writes it, and what it always writes, back to front.
        let mut summary: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(g.len());
        for n in 0..g.len() {
            let (mut reads, mut writes): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
            let read = |v: usize, reads: &mut Vec<usize>| {
                if !reads.contains(&v) {
                    reads.push(v);
                }
            };
            if let Exit::Branch { cond, .. } = &g.exit[n] {
                crate::resource::reads(cond, &mut |v| read(number(v), &mut reads));
            }
            for s in g.code[n].iter().rev() {
                // An assignment that is not under the lane test replaces the value. Any other
                // write may keep the old value, so it does not end the life of the variable.
                let written = match s {
                    Stmt::Set {
                        dst,
                        guard: Guard::Always,
                        ..
                    }
                    | Stmt::Effect {
                        dst: Some(dst),
                        guard: Guard::Always,
                        ..
                    } => Some(number(*dst)),
                    Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } => {
                        read(number(*dst), &mut reads);
                        None
                    }
                    Stmt::Image(op, _) => {
                        op.data_writes()
                            .for_each(|r| read(number(Var::V(r)), &mut reads));
                        None
                    }
                    Stmt::Buffer(op, _) if !op.store => {
                        (0..op.count).for_each(|k| read(number(Var::V(op.data + k)), &mut reads));
                        None
                    }
                    _ => None,
                };
                if let Some(w) = written {
                    reads.retain(|&r| r != w);
                    if !writes.contains(&w) {
                        writes.push(w);
                    }
                }
                crate::resource::stmt_reads(s, &mut |v| read(number(v), &mut reads));
            }
            summary.push((reads, writes));
        }
        let words = index.len().div_ceil(64).max(1);
        let mut at_start = vec![vec![0u64; words]; g.len()];
        let mut changed = true;
        while changed {
            changed = false;
            for n in (0..g.len()).rev() {
                let mut live = vec![0u64; words];
                for s in g.successors(n) {
                    for (w, from) in live.iter_mut().zip(&at_start[s]) {
                        *w |= from;
                    }
                }
                let (reads, writes) = &summary[n];
                for &w in writes {
                    live[w / 64] &= !(1 << (w % 64));
                }
                for &r in reads {
                    live[r / 64] |= 1 << (r % 64);
                }
                if live != at_start[n] {
                    at_start[n] = live;
                    changed = true;
                }
            }
        }
        Live { index, at_start }
    }

    fn at(&self, node: usize, v: Var) -> bool {
        self.index
            .get(&v)
            .is_some_and(|&i| self.at_start[node][i / 64] >> (i % 64) & 1 != 0)
    }
}

/// The variables the code of `nodes` writes outside the lane test, or `None` when it holds a
/// statement that a lane that is not active must not run.
fn always_written(g: &Graph, nodes: &[usize], allow_barrier: bool) -> Option<Vec<Var>> {
    let mut out = Vec::new();
    for &n in nodes {
        for s in &g.code[n] {
            match s {
                Stmt::Set {
                    dst,
                    guard: Guard::Always,
                    ..
                } => out.push(*dst),
                Stmt::Set { .. } | Stmt::Comment(_) => {}
                Stmt::Image(_, Guard::Exec) | Stmt::Buffer(_, Guard::Exec) => {}
                Stmt::Effect {
                    guard: Guard::Exec, ..
                } => {}
                Stmt::Effect {
                    dst,
                    call: Expr::Call(h, _),
                    guard: Guard::Always,
                } if allow_barrier
                    && (h.name == crate::lift::wave::SWIZZLE.name
                        || h.name == crate::lift::wave::SWIZZLE_WAVE.name) =>
                {
                    if let Some(dst) = dst {
                        out.push(*dst);
                    }
                }
                Stmt::Barrier { .. } if allow_barrier => {}
                Stmt::Image(..)
                | Stmt::Buffer(..)
                | Stmt::Effect { .. }
                | Stmt::Export(_)
                | Stmt::Output { .. }
                | Stmt::Discard { .. }
                | Stmt::Message { .. }
                | Stmt::Barrier { .. } => return None,
            }
        }
    }
    Some(out)
}

/// Whether a boolean expression is false (or a 32 bit expression is zero, for a parked mask)
/// when every variable of `known` is false or zero.
fn off(e: &Expr, known: &BTreeSet<Var>) -> bool {
    match e {
        Expr::Bool(b) => !b,
        Expr::U32(0) => true,
        Expr::Var(v) => known.contains(v),
        Expr::Un(Un::BoolToU32 | Un::U32ToBool, a) => off(a, known),
        Expr::Bin(Bin::LAnd, a, b) => off(a, known) || off(b, known),
        Expr::Bin(Bin::LOr | Bin::LXor, a, b) => off(a, known) && off(b, known),
        Expr::Select(_, a, b) => off(a, known) && off(b, known),
        _ => false,
    }
}

/// Whether a lane that enters `nodes` at `entry` with its EXEC bit clear keeps it clear across
/// every statement, branch and loop backedge in `nodes`, so that no statement under the lane test
/// runs for it. Returns the variables proven false at the end of each node when it holds.
fn stays_inactive(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    entry: usize,
) -> Option<HashMap<usize, BTreeSet<Var>>> {
    let mut at_end: HashMap<usize, BTreeSet<Var>> = HashMap::with_capacity(nodes.len());
    let mut changed = true;
    while changed {
        changed = false;
        for &n in nodes.iter().rev() {
            let mut known = if n == entry {
                let mut base = BTreeSet::from([Var::Exec]);
                for p in &preds[n] {
                    if let Some(s) = at_end.get(p) {
                        base = &base & s;
                    }
                }
                base
            } else {
                let mut sets = preds[n].iter().filter_map(|p| at_end.get(p));
                let first = sets.next()?;
                sets.fold(first.clone(), |all, s| &all & s)
            };
            if !known.contains(&Var::Exec) {
                return None;
            }
            for s in &g.code[n] {
                let (dst, value) = match s {
                    Stmt::Set {
                        dst,
                        value,
                        guard: Guard::Always,
                    } => (*dst, Some(value)),
                    Stmt::Effect {
                        dst: Some(dst),
                        guard: Guard::Always,
                        ..
                    } => (*dst, None),
                    _ => continue,
                };
                if value.is_some_and(|v| off(v, &known)) {
                    known.insert(dst);
                } else {
                    known.remove(&dst);
                }
                if !known.contains(&Var::Exec) {
                    return None;
                }
            }
            if at_end.get(&n) != Some(&known) {
                at_end.insert(n, known);
                changed = true;
            }
        }
    }
    Some(at_end)
}

fn is_exec_dirty(v: Var, dirty: &BTreeSet<Var>) -> bool {
    matches!(v, Var::V(_) | Var::Exec | Var::Vcc | Var::Mask(_)) || dirty.contains(&v)
}

fn expr_dirty(e: &Expr, dirty: &BTreeSet<Var>) -> bool {
    let mut bad = false;
    crate::resource::reads(e, &mut |v| {
        if is_exec_dirty(v, dirty) {
            bad = true;
        }
    });
    bad
}

/// Flow sensitive forward dataflow over `nodes`: variables at the end of each node whose value
/// may depend on `EXEC`, lane masks, vector registers or statements skipped when `EXEC` is false.
fn exec_dirty_at_end(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
) -> HashMap<usize, BTreeSet<Var>> {
    let mut at_end: HashMap<usize, BTreeSet<Var>> = HashMap::with_capacity(nodes.len());
    let mut changed = true;
    while changed {
        changed = false;
        for &n in nodes.iter().rev() {
            let mut dirty = BTreeSet::new();
            for p in &preds[n] {
                if let Some(s) = at_end.get(p) {
                    dirty.extend(s.iter().copied());
                }
            }
            for s in &g.code[n] {
                match s {
                    Stmt::Set {
                        dst,
                        value,
                        guard: Guard::Always,
                    } => {
                        if expr_dirty(value, &dirty) {
                            dirty.insert(*dst);
                        } else {
                            dirty.remove(dst);
                        }
                    }
                    Stmt::Set {
                        dst,
                        guard: Guard::Exec,
                        ..
                    }
                    | Stmt::Effect { dst: Some(dst), .. } => {
                        dirty.insert(*dst);
                    }
                    _ => {}
                }
            }
            if at_end.get(&n) != Some(&dirty) {
                at_end.insert(n, dirty);
                changed = true;
            }
        }
    }
    at_end
}

/// Strongly connected components inside `subset` that contain at least one edge, by iterative
/// Tarjan search.
fn sccs_with_cycles(adj: &HashMap<usize, Vec<usize>>, subset: &[usize]) -> Vec<Vec<usize>> {
    let in_subset: HashSet<usize> = subset.iter().copied().collect();
    let mut index: HashMap<usize, usize> = HashMap::with_capacity(subset.len());
    let mut low: HashMap<usize, usize> = HashMap::with_capacity(subset.len());
    let mut on: HashSet<usize> = HashSet::with_capacity(subset.len());
    let mut stack: Vec<usize> = Vec::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut next = 0usize;

    let ways = |v: usize| -> Vec<usize> {
        adj.get(&v)
            .into_iter()
            .flatten()
            .copied()
            .filter(|s| in_subset.contains(s))
            .collect()
    };

    for &root in subset {
        if index.contains_key(&root) {
            continue;
        }
        let mut calls: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut enter = Some(root);
        loop {
            if let Some(v) = enter.take() {
                index.insert(v, next);
                low.insert(v, next);
                next += 1;
                stack.push(v);
                on.insert(v);
                calls.push((v, ways(v)));
            }
            let Some((v, rest)) = calls.last_mut() else {
                break;
            };
            let v = *v;
            if let Some(w) = rest.pop() {
                if !index.contains_key(&w) {
                    enter = Some(w);
                } else if on.contains(&w) {
                    let lw = index[&w];
                    let lv = low.get_mut(&v).expect("visited");
                    if lw < *lv {
                        *lv = lw;
                    }
                }
                continue;
            }
            calls.pop();
            if let Some((parent, _)) = calls.last() {
                let lv = low[&v];
                let lp = low.get_mut(parent).expect("visited");
                if lv < *lp {
                    *lp = lv;
                }
            }
            if low[&v] == index[&v] {
                let mut part = Vec::new();
                loop {
                    let w = stack.pop().expect("scc root on stack");
                    on.remove(&w);
                    part.push(w);
                    if w == v {
                        break;
                    }
                }
                if part.len() > 1 || adj.get(&v).is_some_and(|s| s.contains(&v)) {
                    out.push(part);
                }
            }
        }
    }
    out
}

/// Whether every loop reachable by an inactive lane (`EXEC == false`) inside `nodes` exits only
/// through scalar conditions that do not depend on `EXEC` or vector registers.
///
/// Branches whose condition is already decided when `EXEC == false` (such as `s_cbranch_execnz`
/// or `s_cbranch_vccz` after a vector compare) are folded to their inactive lane successor first:
/// a loop exited by `s_cbranch_execnz` disappears, while a retry loop whose backedge is taken on
/// `vcc == 0` loses its exit edge and is rejected.
fn loops_safe_when_inactive(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    at_end: &HashMap<usize, BTreeSet<Var>>,
) -> bool {
    let dirty_end = exec_dirty_at_end(g, preds, nodes);
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::with_capacity(nodes.len());
    let mut clean_branch: HashSet<usize> = HashSet::new();
    for &n in nodes {
        let mut succs = Vec::new();
        match &g.exit[n] {
            Exit::End => {}
            Exit::Jump(t) => succs.push(*t),
            Exit::Branch { cond, taken, fall } => {
                let known = &at_end[&n];
                let always_fall = off(cond, known);
                let always_taken = matches!(cond, Expr::Un(Un::LNot, inner) if off(inner, known));
                if always_fall {
                    succs.push(*fall);
                } else if always_taken {
                    succs.push(*taken);
                } else {
                    succs.push(*taken);
                    succs.push(*fall);
                    if !expr_dirty(cond, &dirty_end[&n]) {
                        clean_branch.insert(n);
                    }
                }
            }
        }
        adj.insert(n, succs);
    }

    let mut work: Vec<Vec<usize>> = vec![nodes.to_vec()];
    while let Some(subset) = work.pop() {
        for part in sccs_with_cycles(&adj, &subset) {
            let in_part: HashSet<usize> = part.iter().copied().collect();
            let exits: Vec<usize> = part
                .iter()
                .copied()
                .filter(|b| {
                    adj.get(b)
                        .is_some_and(|targets| targets.iter().any(|t| !in_part.contains(t)))
                })
                .collect();
            if exits.is_empty() || !exits.iter().all(|b| clean_branch.contains(b)) {
                return false;
            }
            let exit_set: HashSet<usize> = exits.into_iter().collect();
            let rest: Vec<usize> = part
                .into_iter()
                .filter(|n| !exit_set.contains(n))
                .collect();
            if !rest.is_empty() {
                work.push(rest);
            }
        }
    }
    true
}

/// Removes nodes no longer reachable from node 0 after dropping a skip to an otherwise unused end
/// block, keeping the invariant that every node of [`Graph`] is reachable.
fn prune_unreachable(g: &mut Graph) {
    let n = g.len();
    let mut map = vec![usize::MAX; n];
    let mut order = Vec::with_capacity(n);
    map[0] = 0;
    order.push(0);
    let mut i = 0;
    while i < order.len() {
        let b = order[i];
        i += 1;
        for s in g.successors(b) {
            if map[s] == usize::MAX {
                map[s] = order.len();
                order.push(s);
            }
        }
    }
    if order.len() == n {
        return;
    }
    let mut code = Vec::with_capacity(order.len());
    let mut exit = Vec::with_capacity(order.len());
    for &old in &order {
        code.push(std::mem::take(&mut g.code[old]));
        exit.push(match &g.exit[old] {
            Exit::End => Exit::End,
            Exit::Jump(t) => Exit::Jump(map[*t]),
            Exit::Branch { cond, taken, fall } => Exit::Branch {
                cond: cond.clone(),
                taken: map[*taken],
                fall: map[*fall],
            },
        });
    }
    g.code = code;
    g.exit = exit;
}

/// Drops skips around derivative taking samples and workgroup barriers. Returns how many.
pub(super) fn drop_skips(g: &mut Graph) -> usize {
    let has_deriv = g.code.iter().flatten().any(takes_derivatives);
    let has_barrier = g.code.iter().flatten().any(is_barrier);
    if !(has_deriv || has_barrier) {
        return 0;
    }
    if !(0..g.len()).any(|b| exec_skip(g, b).is_some()) {
        return 0;
    }
    let mut preds = vec![Vec::new(); g.len()];
    for n in 0..g.len() {
        for s in g.successors(n) {
            preds[s].push(n);
        }
    }
    let mut live = None;
    let mut dropped = 0;
    loop {
        let before = dropped;
        for branch in 0..g.len() {
            let Some((entry, target)) = exec_skip(g, branch) else {
                continue;
            };
            let drop_deriv = has_deriv
                && region(g, &preds, branch, entry, target).is_some_and(|nodes| {
                    if !nodes
                        .iter()
                        .any(|&n| g.code[n].iter().any(takes_derivatives))
                    {
                        return false;
                    }
                    let Some(written) = always_written(g, &nodes, false) else {
                        return false;
                    };
                    if stays_inactive(g, &preds, &nodes, entry).is_none() {
                        return false;
                    }
                    // The liveness of the graph as it came in. A dropped skip only takes an edge
                    // away, which makes nothing live that was not.
                    let live = live.get_or_insert_with(|| Live::new(g));
                    !written.iter().any(|&v| live.at(target, v))
                });
            let drop_barrier = !drop_deriv
                && has_barrier
                && barrier_region(g, &preds, branch, entry, target).is_some_and(
                    |(nodes, has_cycle)| {
                        if !nodes.iter().any(|&n| g.code[n].iter().any(is_barrier)) {
                            return false;
                        }
                        let Some(written) = always_written(g, &nodes, true) else {
                            return false;
                        };
                        let Some(at_end) = stays_inactive(g, &preds, &nodes, entry) else {
                            return false;
                        };
                        if has_cycle && !loops_safe_when_inactive(g, &preds, &nodes, &at_end) {
                            return false;
                        }
                        let live = live.get_or_insert_with(|| Live::new(g));
                        !written.iter().any(|&v| live.at(target, v))
                    },
                );
            if drop_deriv || drop_barrier {
                g.exit[branch] = Exit::Jump(entry);
                preds[target].retain(|&p| p != branch);
                dropped += 1;
            }
        }
        if dropped == before {
            break;
        }
    }
    if dropped > 0 {
        prune_unreachable(g);
    }
    dropped
}
