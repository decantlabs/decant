//! Taking away a skip that stands around a derivative taking sample.
//!
//! The machine jumps over masked code when no lane of the wave is active (`s_cbranch_execz`).
//! One lane per invocation makes that `if (exec) { .. }` around the code. For most code that is
//! the same and saves work. Not for a sample that takes its level of detail from screen space
//! derivatives: on the machine the inactive lanes of a quad run the sample along with the
//! active ones and so give it their coordinates, while in HLSL a sample under a condition that
//! differs inside a quad has no defined derivatives. So a skip whose code holds such a sample
//! is dropped: every lane runs the code, and only the lanes that are active write.
//!
//! That is the same program when the skipped code does nothing in a lane that is not active:
//!
//! * the code is entered through the skip only, leaves to the target of the skip only, and
//!   has no loop,
//! * each of its statements is under the lane test, or is an assignment whose variable nothing
//!   reads once the code is left (a saved mask, a condition code, a scalar the code uses
//!   itself). Exports, messages, barriers and effects that are not under the lane test keep
//!   the skip,
//! * the code never sets EXEC to something that can be true in a lane that came in inactive.
//!   Saving EXEC, narrowing it and restoring the saved value are all false in such a lane.

use std::collections::{BTreeSet, HashMap};

use super::graph::{Exit, Graph};
use crate::ir::{Bin, Expr, Guard, ImageKind, SampleLod, Stmt, Un, Var};

/// The code a branch on this lane's EXEC bit skips, and where it skips to.
fn skip(exit: &Exit) -> Option<(usize, usize)> {
    let Exit::Branch { cond, taken, fall } = exit else {
        return None;
    };
    match cond {
        Expr::Var(Var::Exec) => Some((*taken, *fall)),
        Expr::Un(Un::LNot, inner) if **inner == Expr::Var(Var::Exec) => Some((*fall, *taken)),
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

/// The most nodes a skipped piece of code may have to be looked at.
const MAX_NODES: usize = 256;

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
fn always_written(g: &Graph, nodes: &[usize]) -> Option<Vec<Var>> {
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

/// Whether a boolean expression is false when every variable of `off` is.
fn off(e: &Expr, known: &BTreeSet<Var>) -> bool {
    match e {
        Expr::Bool(b) => !b,
        Expr::Var(v) => known.contains(v),
        Expr::Bin(Bin::LAnd, a, b) => off(a, known) || off(b, known),
        Expr::Bin(Bin::LOr | Bin::LXor, a, b) => off(a, known) && off(b, known),
        Expr::Select(_, a, b) => off(a, known) && off(b, known),
        _ => false,
    }
}

/// Whether a lane that enters the code of `nodes` at `entry` with its EXEC bit clear keeps it
/// clear to the end, so that no statement under the lane test runs for it. The code may save
/// and narrow EXEC and restore what it saved: all of that is false when EXEC was. `nodes` are
/// in the order a walk finished them, so the reverse has every node behind its predecessors.
fn stays_inactive(g: &Graph, preds: &[Vec<usize>], nodes: &[usize], entry: usize) -> bool {
    let mut at_end: HashMap<usize, BTreeSet<Var>> = HashMap::new();
    for &n in nodes.iter().rev() {
        let mut known = if n == entry {
            BTreeSet::from([Var::Exec])
        } else {
            let mut sets = preds[n].iter().filter_map(|p| at_end.get(p));
            let Some(first) = sets.next() else {
                return false;
            };
            sets.fold(first.clone(), |all, s| &all & s)
        };
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
                return false;
            }
        }
        at_end.insert(n, known);
    }
    true
}

/// Drops the skips around derivative taking samples. Returns how many.
pub(super) fn drop_skips(g: &mut Graph) -> usize {
    let wanted = |g: &Graph| {
        g.code.iter().flatten().any(takes_derivatives) && g.exit.iter().any(|e| skip(e).is_some())
    };
    if !wanted(g) {
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
    for branch in 0..g.len() {
        let Some((entry, target)) = skip(&g.exit[branch]) else {
            continue;
        };
        let Some(nodes) = region(g, &preds, branch, entry, target) else {
            continue;
        };
        if !nodes
            .iter()
            .any(|&n| g.code[n].iter().any(takes_derivatives))
        {
            continue;
        }
        let Some(written) = always_written(g, &nodes) else {
            continue;
        };
        if !stays_inactive(g, &preds, &nodes, entry) {
            continue;
        }
        // The liveness of the graph as it came in. A dropped skip only takes an edge away,
        // which makes nothing live that was not.
        let live = live.get_or_insert_with(|| Live::new(g));
        if written.iter().any(|&v| live.at(target, v)) {
            continue;
        }
        g.exit[branch] = Exit::Jump(entry);
        dropped += 1;
    }
    dropped
}
