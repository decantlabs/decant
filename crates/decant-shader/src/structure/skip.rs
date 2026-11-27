//! Taking away a skip that stands around a derivative taking sample or a workgroup barrier.
//!
//! The machine jumps over masked code when no lane of the wave is active (`s_cbranch_execz`,
//! `s_cbranch_scc0` after a mask update that sets SCC from EXEC, or `s_cbranch_vccz` when VCC
//! equals EXEC). One lane per invocation makes that `if (exec) { .. }` around the code, or an
//! early return when the skip targets `s_endpgm`. For most code that is the same and saves work.
//! Two cases break if the skip stays:
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
//! For an early exit (`s_endpgm`) before a barrier whose condition is not already an EXEC test,
//! the early exit is converted into `exec = false` and converges with the barrier path. That is
//! the same program when the skipped code does nothing observable in an inactive lane:
//!
//! * the code is entered through the skip only, and leaves to the target of the skip only (or to
//!   an empty end block, or an internal `Exit::End` converted to `exec = false` when the tail
//!   from `target` keeps an inactive lane inactive),
//! * each of its statements is under the lane test, is a barrier or pure/predicated wave helper
//!   (for a barrier region), or is an assignment whose variable nothing reads once the code is
//!   left (or is safely predicated under `Guard::Exec` for a barrier region). Exports, messages
//!   and other effects that are not under the lane test keep the skip,
//! * the code never sets EXEC to something that can be true in a lane that came in inactive.
//!   Saving EXEC, narrowing it and restoring the saved value are all false in such a lane,
//! * for a barrier region that contains loops, every cycle reachable by an inactive lane exits
//!   only through scalar conditions that do not depend on EXEC or vector registers.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use super::graph::{Exit, Graph};
use crate::ir::{Bin, Expr, Guard, ImageKind, SampleLod, Stmt, Un, Var};

/// Sentinel variable used locally when proving `Var::Exec == entry_exec` inside a barrier region.
/// Never clashes with lifted flags (which count up from 0) or lowered flags (which start at
/// `FIRST_FLAG = 0xFFFE` and count down).
const ENTRY_EXEC_SENTINEL: Var = Var::Flag(u16::MAX);

/// Pairwise boolean equivalence and implication relations among lane-mask variables at a program
/// point, used to recognize branches equivalent to testing `Var::Exec` (including cross-block
/// `scc`, saved-mask, spilled-mask, and `vcc` skips).
///
/// Invariant: each multi-member equivalence class uses its minimum `Var` as its canonical
/// representative `c`. Every other member `u > c` is stored in `eq` as `u -> (c, negated)` where
/// `u == c` if `!negated` or `u == !c` if `negated`. Canonical representatives are omitted from
/// `eq`. `implies_exec` holds `(c, negated)` pairs for canonical representatives proven to satisfy
/// `(if negated { !c } else { c }) => Var::Exec` (so `x && Var::Exec == x`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ExecRel {
    eq: BTreeMap<Var, (Var, bool)>,
    implies_exec: BTreeSet<(Var, bool)>,
}

impl ExecRel {
    fn find(&self, v: Var) -> (Var, bool) {
        self.eq.get(&v).copied().unwrap_or((v, false))
    }

    fn exec_class(&self) -> (Var, bool) {
        self.find(Var::Exec)
    }

    /// Whether `v` is proven equal to `Var::Exec` (`Some(true)`) or `!Var::Exec` (`Some(false)`).
    fn rel_to_exec(&self, v: Var) -> Option<bool> {
        let (vc, vn) = self.find(v);
        let (ec, en) = self.exec_class();
        (vc == ec).then_some(vn == en)
    }

    fn is_tracked(&self, v: Var) -> bool {
        v == Var::Exec
            || self.eq.contains_key(&v)
            || self.eq.values().any(|&(c, _)| c == v)
            || self.implies_exec.iter().any(|&(c, _)| c == v)
    }

    fn var_implies_exec(&self, v: Var, neg: bool) -> bool {
        let (c, n) = self.find(v);
        let r = (c, n ^ neg);
        r == self.exec_class() || self.implies_exec.contains(&r)
    }

    fn eval(&self, e: &Expr) -> Option<(Var, bool)> {
        match e {
            Expr::Var(v) => Some(self.find(*v)),
            Expr::Un(Un::LNot, inner) => self.eval(inner).map(|(c, n)| (c, !n)),
            Expr::Un(Un::BoolToU32 | Un::U32ToBool, inner) => self.eval(inner),
            Expr::Bin(Bin::Ne, a, b) => match (&**a, &**b) {
                (Expr::Var(v), Expr::U32(0)) | (Expr::U32(0), Expr::Var(v))
                    if self.is_tracked(*v) =>
                {
                    Some(self.find(*v))
                }
                _ => None,
            },
            Expr::Bin(Bin::Eq, a, b) => match (&**a, &**b) {
                (Expr::Var(v), Expr::U32(0)) | (Expr::U32(0), Expr::Var(v))
                    if self.is_tracked(*v) =>
                {
                    let (c, n) = self.find(*v);
                    Some((c, !n))
                }
                _ => None,
            },
            Expr::Select(c, a, b) => match (&**a, &**b) {
                (Expr::Bool(true), Expr::Bool(false)) => self.eval(c),
                (Expr::Bool(false), Expr::Bool(true)) => self.eval(c).map(|(k, n)| (k, !n)),
                _ => None,
            },
            Expr::Bin(Bin::LAnd, a, b) => {
                if **a == Expr::Bool(true) {
                    return self.eval(b);
                }
                if **b == Expr::Bool(true) {
                    return self.eval(a);
                }
                let ra = self.eval(a);
                let rb = self.eval(b);
                if ra.is_some() && ra == rb {
                    return ra;
                }
                let ec = self.exec_class();
                if ra == Some(ec)
                    && let Some(other) = rb
                    && self.implies_exec.contains(&other)
                {
                    return Some(other);
                }
                if rb == Some(ec)
                    && let Some(other) = ra
                    && self.implies_exec.contains(&other)
                {
                    return Some(other);
                }
                None
            }
            _ => None,
        }
    }

    fn expr_implies_exec(&self, e: &Expr) -> bool {
        if let Some(r) = self.eval(e)
            && (r == self.exec_class() || self.implies_exec.contains(&r))
        {
            return true;
        }
        match e {
            Expr::Bin(Bin::LAnd, a, b) => self.expr_implies_exec(a) || self.expr_implies_exec(b),
            Expr::Un(Un::BoolToU32 | Un::U32ToBool, inner) => self.expr_implies_exec(inner),
            _ => false,
        }
    }

    /// Picks a variable `u != avoid` currently in `(canon, neg)`'s equivalence class, returning
    /// `(u, neg_relative_to_u)` so that after `avoid` is killed, the original value can still be
    /// equated to `u`.
    fn pick_other(&self, canon: Var, neg: bool, avoid: Var) -> Option<(Var, bool)> {
        if canon != avoid {
            return Some((canon, neg));
        }
        self.eq
            .iter()
            .find(|&(&u, &(c, _))| u != avoid && c == canon)
            .map(|(&u, &(_, nu))| (u, neg ^ nu))
    }

    /// Removes `v` from its equivalence class while preserving all equalities among the remaining
    /// members of that class.
    fn kill(&mut self, v: Var) {
        if self.eq.remove(&v).is_some() {
            return;
        }
        let members: Vec<(Var, bool)> = self
            .eq
            .iter()
            .filter_map(|(&u, &(c, n))| (c == v).then_some((u, n)))
            .collect();
        if members.is_empty() {
            self.implies_exec.remove(&(v, false));
            self.implies_exec.remove(&(v, true));
            return;
        }
        // `BTreeMap` iterates in ascending `Var` order, so `members[0].0` is the new minimum rep.
        let (new_c, flip) = members[0];
        self.eq.remove(&new_c);
        for &(u, n) in &members[1..] {
            self.eq.insert(u, (new_c, n ^ flip));
        }
        let imp_f = self.implies_exec.remove(&(v, false));
        let imp_t = self.implies_exec.remove(&(v, true));
        if imp_f {
            self.implies_exec.insert((new_c, flip));
        }
        if imp_t {
            self.implies_exec.insert((new_c, !flip));
        }
    }

    /// Merges `a`'s and `b`'s equivalence classes with `a == (if neg { !b } else { b })`.
    fn equate(&mut self, a: Var, b: Var, neg: bool) {
        let (ca, na) = self.find(a);
        let (cb, nb) = self.find(b);
        if ca == cb {
            return;
        }
        let rel_neg = neg ^ na ^ nb;
        let (keep, drop) = if ca < cb { (ca, cb) } else { (cb, ca) };
        let to_move: Vec<(Var, bool)> = self
            .eq
            .iter()
            .filter_map(|(&u, &(c, n))| (c == drop).then_some((u, n)))
            .collect();
        self.eq.insert(drop, (keep, rel_neg));
        for (u, n) in to_move {
            self.eq.insert(u, (keep, n ^ rel_neg));
        }
        let imp_f = self.implies_exec.remove(&(drop, false));
        let imp_t = self.implies_exec.remove(&(drop, true));
        if imp_f {
            self.implies_exec.insert((keep, rel_neg));
        }
        if imp_t {
            self.implies_exec.insert((keep, !rel_neg));
        }
    }

    fn step(&mut self, s: &Stmt) {
        match s {
            Stmt::Comment(_) => {}
            Stmt::Set {
                dst,
                value,
                guard: Guard::Always,
            } => {
                let rhs_rel = self.eval(value);
                if *dst == Var::Exec && rhs_rel == Some(self.exec_class()) {
                    return;
                }
                let rhs_imp = self.expr_implies_exec(value);
                let target = rhs_rel.and_then(|(c, n)| self.pick_other(c, n, *dst));
                self.kill(*dst);
                if *dst == Var::Exec {
                    // Prior `x => old_exec` implications do not necessarily hold for the new EXEC.
                    self.implies_exec.clear();
                }
                if let Some((tv, tn)) = target {
                    self.equate(*dst, tv, tn);
                }
                if *dst != Var::Exec && rhs_imp {
                    let r = self.find(*dst);
                    if r != self.exec_class() {
                        self.implies_exec.insert(r);
                    }
                }
            }
            Stmt::Set {
                dst,
                guard: Guard::Exec,
                ..
            }
            | Stmt::Effect { dst: Some(dst), .. } => {
                self.kill(*dst);
                if *dst == Var::Exec {
                    self.implies_exec.clear();
                }
            }
            Stmt::Image(op, _) => {
                for r in op.data_writes() {
                    self.kill(Var::V(r));
                }
            }
            Stmt::Buffer(op, _) if !op.store => {
                for k in 0..op.count {
                    self.kill(Var::V(op.data + k));
                }
            }
            _ => {}
        }
    }

    /// Lattice meet: keeps only pairwise equalities and `implies_exec` facts that hold in every
    /// predecessor state.
    fn meet(states: &[&ExecRel]) -> ExecRel {
        if states.is_empty() {
            return ExecRel::default();
        }
        if states.len() == 1 {
            return states[0].clone();
        }
        let first = states[0];
        let mut vars = BTreeSet::new();
        vars.insert(Var::Exec);
        for (&u, &(c, _)) in &first.eq {
            vars.insert(u);
            vars.insert(c);
        }
        for &(c, _) in &first.implies_exec {
            vars.insert(c);
        }
        let vars: Vec<Var> = vars.into_iter().collect();

        let mut out = ExecRel::default();
        for (i, &u) in vars.iter().enumerate() {
            // If `u` already merged into an earlier representative, its class is already handled.
            if out.eq.contains_key(&u) {
                continue;
            }
            let (cu0, nu0) = first.find(u);
            for &v in &vars[i + 1..] {
                if out.eq.contains_key(&v) {
                    continue;
                }
                let (cv0, nv0) = first.find(v);
                if cu0 != cv0 {
                    continue;
                }
                let neg = nu0 ^ nv0;
                let all_agree = states[1..].iter().all(|st| {
                    let (cu, nu) = st.find(u);
                    let (cv, nv) = st.find(v);
                    cu == cv && (nu ^ nv) == neg
                });
                if all_agree {
                    out.equate(u, v, neg);
                }
            }
        }

        let mut reps = BTreeSet::new();
        for &v in &vars {
            reps.insert(out.find(v).0);
        }
        let ec = out.exec_class();
        for rep in reps {
            for neg in [false, true] {
                if (rep, neg) == ec {
                    continue;
                }
                if states.iter().all(|st| st.var_implies_exec(rep, neg)) {
                    out.implies_exec.insert((rep, neg));
                }
            }
        }
        out
    }
}

/// Forward dataflow over `g`: computes the `ExecRel` state at the end of every reachable node and
/// which nodes are reachable from node 0.
fn exec_rels(g: &Graph, preds: &[Vec<usize>]) -> (Vec<ExecRel>, Vec<bool>) {
    let n = g.len();
    let mut in_state: Vec<Option<ExecRel>> = vec![None; n];
    let mut out_state: Vec<Option<ExecRel>> = vec![None; n];
    let mut in_queue = vec![false; n];
    let mut work = VecDeque::new();
    in_queue[0] = true;
    work.push_back(0);

    while let Some(node) = work.pop_front() {
        in_queue[node] = false;
        let next_in = if node == 0 {
            let mut pred_states: Vec<&ExecRel> = preds[0]
                .iter()
                .filter_map(|&p| out_state[p].as_ref())
                .collect();
            let entry_default = ExecRel::default();
            pred_states.push(&entry_default);
            ExecRel::meet(&pred_states)
        } else {
            let pred_states: Vec<&ExecRel> = preds[node]
                .iter()
                .filter_map(|&p| out_state[p].as_ref())
                .collect();
            if pred_states.is_empty() {
                continue;
            }
            ExecRel::meet(&pred_states)
        };
        if in_state[node].as_ref() == Some(&next_in) {
            continue;
        }
        in_state[node] = Some(next_in.clone());
        let mut cur = next_in;
        for s in &g.code[node] {
            cur.step(s);
        }
        if out_state[node].as_ref() != Some(&cur) {
            out_state[node] = Some(cur);
            for succ in g.successors(node) {
                if !in_queue[succ] {
                    in_queue[succ] = true;
                    work.push_back(succ);
                }
            }
        }
    }

    let reachable = out_state.iter().map(Option::is_some).collect();
    let rels = out_state
        .into_iter()
        .map(Option::unwrap_or_default)
        .collect();
    (rels, reachable)
}

/// For each node in `nodes`, computes for every statement whether `Var::Exec` at that statement
/// is provably equal to `Var::Exec` upon entering `entry`. Used to verify that predicating a
/// `Guard::Always` write with `Guard::Exec` preserves behavior for outer-active lanes even when
/// `nodes` contains inner `EXEC` narrowings.
fn exec_equals_entry_per_stmt(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    entry: usize,
) -> HashMap<usize, Vec<bool>> {
    let inside: HashSet<usize> = nodes.iter().copied().collect();
    let mut entry_seed = ExecRel::default();
    entry_seed.equate(Var::Exec, ENTRY_EXEC_SENTINEL, false);

    let mut in_state: HashMap<usize, ExecRel> = HashMap::with_capacity(nodes.len());
    let mut out_state: HashMap<usize, ExecRel> = HashMap::with_capacity(nodes.len());
    let mut changed = true;
    while changed {
        changed = false;
        for &n in nodes.iter().rev() {
            let next_in = if n == entry {
                let mut states: Vec<&ExecRel> = preds[n]
                    .iter()
                    .filter(|p| inside.contains(p))
                    .filter_map(|p| out_state.get(p))
                    .collect();
                states.push(&entry_seed);
                ExecRel::meet(&states)
            } else {
                let states: Vec<&ExecRel> = preds[n]
                    .iter()
                    .filter(|p| inside.contains(p))
                    .filter_map(|p| out_state.get(p))
                    .collect();
                if states.is_empty() {
                    continue;
                }
                ExecRel::meet(&states)
            };
            if in_state.get(&n) == Some(&next_in) {
                continue;
            }
            in_state.insert(n, next_in.clone());
            let mut cur = next_in;
            for s in &g.code[n] {
                cur.step(s);
            }
            if out_state.get(&n) != Some(&cur) {
                out_state.insert(n, cur);
                changed = true;
            }
        }
    }

    let mut per_stmt = HashMap::with_capacity(nodes.len());
    for &n in nodes {
        let mut cur = in_state.get(&n).cloned().unwrap_or_default();
        let mut flags = Vec::with_capacity(g.code[n].len());
        for s in &g.code[n] {
            flags.push(cur.rel_to_exec(ENTRY_EXEC_SENTINEL) == Some(true));
            cur.step(s);
        }
        per_stmt.insert(n, flags);
    }
    per_stmt
}

/// The code a branch on this lane's EXEC bit skips, and where it skips to: `(entry, target)`
/// where an active lane enters `entry` and an inactive lane skips to `target`.
fn exec_skip(g: &Graph, rels: &[ExecRel], branch: usize) -> Option<(usize, usize)> {
    let Exit::Branch { cond, taken, fall } = &g.exit[branch] else {
        return None;
    };
    let (var, negated) = match cond {
        Expr::Var(v) => (*v, false),
        Expr::Un(Un::LNot, inner) => match &**inner {
            Expr::Var(v) => (*v, true),
            _ => return None,
        },
        _ => return None,
    };
    let same_as_exec = if var == Var::Exec {
        true
    } else {
        rels[branch].rel_to_exec(var)?
    };
    let active_is_taken = same_as_exec ^ negated;
    Some(if active_is_taken {
        (*taken, *fall)
    } else {
        (*fall, *taken)
    })
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
    let mut seen = HashSet::new();
    let limit = g.len().min(64);
    for _ in 0..limit {
        if !seen.insert(node) {
            return false;
        }
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

struct BarrierRegion {
    /// Region nodes in postorder.
    nodes: Vec<usize>,
    /// Whether the region contains a cycle.
    has_cycle: bool,
    /// Internal `Exit::End` blocks reached inside the region when `target` is not itself an empty
    /// end block; these are converted into `exec = false; jump target` when the tail is safe.
    internal_ends: Vec<usize>,
}

/// The nodes from `entry` up to `target` for a barrier skip. Unlike [`region`], loops inside the
/// region are permitted (checked separately by [`loops_safe_when_inactive`]), there is no node
/// count cap, and internal `Exit::End` blocks are collected rather than aborting immediately.
fn barrier_region(
    g: &Graph,
    preds: &[Vec<usize>],
    branch: usize,
    entry: usize,
    target: usize,
) -> Option<BarrierRegion> {
    if entry == target || entry == branch {
        return None;
    }
    let target_is_end = is_empty_end(g, target);
    let mut state: HashMap<usize, u8> = HashMap::new();
    let mut nodes = Vec::new();
    let mut internal_ends = Vec::new();
    let mut has_cycle = false;
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(entry, g.successors(entry).collect())];
    state.insert(entry, 1);
    while let Some((node, rest)) = stack.last_mut() {
        let node = *node;
        let Some(next) = rest.pop() else {
            if matches!(g.exit[node], Exit::End) && !target_is_end {
                internal_ends.push(node);
            }
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
            None if next == branch => return None,
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
    one_way_in.then_some(BarrierRegion {
        nodes,
        has_cycle,
        internal_ends,
    })
}

/// All nodes reachable from `start` in postorder, plus whether the reachable subgraph contains a
/// cycle. If `redirect` is `Some((from, old_succ, new_succ))`, edges `from -> old_succ` are
/// walked as `from -> new_succ` instead.
fn reachable_tail(
    g: &Graph,
    start: usize,
    redirect: Option<(usize, usize, usize)>,
) -> (Vec<usize>, bool) {
    let succs_of = |n: usize| -> Vec<usize> {
        let mut out = Vec::new();
        for s in g.successors(n) {
            let mapped = match redirect {
                Some((from, old, new)) if n == from && s == old => new,
                _ => s,
            };
            if !out.contains(&mapped) {
                out.push(mapped);
            }
        }
        out
    };
    let mut state: HashMap<usize, u8> = HashMap::new();
    let mut nodes = Vec::new();
    let mut has_cycle = false;
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(start, succs_of(start))];
    state.insert(start, 1);
    while let Some((node, rest)) = stack.last_mut() {
        let node = *node;
        let Some(next) = rest.pop() else {
            state.insert(node, 2);
            nodes.push(node);
            stack.pop();
            continue;
        };
        match state.get(&next) {
            Some(1) => has_cycle = true,
            Some(_) => {}
            None => {
                state.insert(next, 1);
                stack.push((next, succs_of(next)));
            }
        }
    }
    (nodes, has_cycle)
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

/// Whether a `Guard::Always` `Stmt::Effect` helper call is safe for an inactive lane to run
/// inside a barrier skip region: barriers, swizzles, and pure/predicated wave helpers (never
/// LDS or memory stores/atomics).
fn is_safe_barrier_effect(name: &str) -> bool {
    name == crate::lift::ds::BARRIER.name
        || name == crate::lift::wave::SWIZZLE.name
        || name == crate::lift::wave::SWIZZLE_WAVE.name
        || name == crate::lift::wave::READ_LANE.name
        || name == crate::lift::wave::WRITE_LANE.name
        || name == crate::lift::wave::READ_FIRST_LANE.name
        || name == crate::lift::wave::MASK_WORD.name
        || name == crate::lift::wave::DPP_ADD.name
        || name == crate::lift::wave::DPP_MOV.name
        || name == crate::lift::wave::DPP8_MOV.name
        || name == crate::lift::wave::PERMLANE.name
        || name == crate::lift::wave::BPERMUTE.name
        || name.starts_with("decant_wave_")
        || name.starts_with("decant_mbcnt_")
}

/// Whether a `Guard::Always` `Stmt::Effect` helper call may be predicated with `Guard::Exec`
/// without risking a divergent workgroup barrier inside the helper body.
///
/// Helpers that may contain `GroupMemoryBarrierWithGroupSync()` (such as `decant_barrier`,
/// `decant_swizzle`, or emulated `decant_wave_*` helpers) must never be wrapped in `if (exec)`.
fn can_predicate_barrier_effect(name: &str) -> bool {
    name == crate::lift::wave::SWIZZLE_WAVE.name
        || name == crate::lift::wave::READ_LANE.name
        || name == crate::lift::wave::WRITE_LANE.name
        || name == crate::lift::wave::READ_FIRST_LANE.name
        || name == crate::lift::wave::MASK_WORD.name
        || name == crate::lift::wave::DPP_ADD.name
        || name == crate::lift::wave::DPP_MOV.name
        || name == crate::lift::wave::DPP8_MOV.name
        || name == crate::lift::wave::PERMLANE.name
        || name == crate::lift::wave::BPERMUTE.name
        || name.starts_with("decant_mbcnt_")
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
                } if allow_barrier && is_safe_barrier_effect(h.name) => {
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
/// runs for it. Writes to variables in `predicated` are modeled as `Guard::Exec` (skipped when
/// `EXEC` is false). Returns the variables proven false at the end of each node when it holds.
fn stays_inactive(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    entry: usize,
    predicated: &BTreeSet<Var>,
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
                    } if !predicated.contains(dst) => (*dst, Some(value)),
                    Stmt::Effect {
                        dst: Some(dst),
                        guard: Guard::Always,
                        ..
                    } if !predicated.contains(dst) => (*dst, None),
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
/// may depend on `EXEC`, lane masks, vector registers or statements skipped when `EXEC` is false
/// (including `Guard::Always` writes scheduled for predication in `predicated`).
fn exec_dirty_at_end(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    predicated: &BTreeSet<Var>,
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
                    } if !predicated.contains(dst) => {
                        if expr_dirty(value, &dirty) {
                            dirty.insert(*dst);
                        } else {
                            dirty.remove(dst);
                        }
                    }
                    Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } => {
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

/// Checks whether predicating `Guard::Always` writes to `predicated` inside `nodes` could alter
/// any unfolded branch condition inside `nodes` (which could otherwise route inactive lanes away
/// from a barrier).
fn predicated_affects_branch(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    at_end: &HashMap<usize, BTreeSet<Var>>,
    predicated: &BTreeSet<Var>,
) -> bool {
    if predicated.is_empty() {
        return false;
    }
    let mut tainted_end: HashMap<usize, BTreeSet<Var>> = HashMap::with_capacity(nodes.len());
    let mut changed = true;
    while changed {
        changed = false;
        for &n in nodes.iter().rev() {
            let mut tainted = BTreeSet::new();
            for p in &preds[n] {
                if let Some(s) = tainted_end.get(p) {
                    tainted.extend(s.iter().copied());
                }
            }
            for s in &g.code[n] {
                match s {
                    Stmt::Set {
                        dst,
                        value,
                        guard: Guard::Always,
                    } => {
                        if predicated.contains(dst) {
                            tainted.insert(*dst);
                        } else {
                            let mut reads_tainted = false;
                            crate::resource::reads(value, &mut |v| {
                                if tainted.contains(&v) {
                                    reads_tainted = true;
                                }
                            });
                            if reads_tainted {
                                tainted.insert(*dst);
                            } else {
                                tainted.remove(dst);
                            }
                        }
                    }
                    Stmt::Effect {
                        dst: Some(dst),
                        call,
                        guard: Guard::Always,
                    } => {
                        let mut reads_tainted = predicated.contains(dst);
                        crate::resource::reads(call, &mut |v| {
                            if tainted.contains(&v) {
                                reads_tainted = true;
                            }
                        });
                        if reads_tainted {
                            tainted.insert(*dst);
                        } else {
                            tainted.remove(dst);
                        }
                    }
                    _ => {}
                }
            }
            if tainted_end.get(&n) != Some(&tainted) {
                tainted_end.insert(n, tainted);
                changed = true;
            }
        }
    }
    for &n in nodes {
        let Exit::Branch { cond, .. } = &g.exit[n] else {
            continue;
        };
        let known = &at_end[&n];
        let always_fall = off(cond, known);
        let always_taken = matches!(cond, Expr::Un(Un::LNot, inner) if off(inner, known));
        if always_fall || always_taken {
            continue;
        }
        let tainted = &tainted_end[&n];
        let mut hit = false;
        crate::resource::reads(cond, &mut |v| {
            if tainted.contains(&v) {
                hit = true;
            }
        });
        if hit {
            return true;
        }
    }
    false
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
    predicated: &BTreeSet<Var>,
    redirect: Option<(usize, usize, usize)>,
) -> bool {
    let dirty_end = exec_dirty_at_end(g, preds, nodes, predicated);
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::with_capacity(nodes.len());
    let mut clean_branch: HashSet<usize> = HashSet::new();
    let map_succ = |from: usize, s: usize| match redirect {
        Some((r_from, old, new)) if from == r_from && s == old => new,
        _ => s,
    };
    for &n in nodes {
        let mut succs = Vec::new();
        match &g.exit[n] {
            Exit::End => {}
            Exit::Jump(t) => succs.push(map_succ(n, *t)),
            Exit::Branch { cond, taken, fall } => {
                let taken = map_succ(n, *taken);
                let fall = map_succ(n, *fall);
                let known = &at_end[&n];
                let always_fall = off(cond, known);
                let always_taken = matches!(cond, Expr::Un(Un::LNot, inner) if off(inner, known));
                if always_fall {
                    succs.push(fall);
                } else if always_taken || taken == fall {
                    succs.push(taken);
                } else {
                    succs.push(taken);
                    succs.push(fall);
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

/// Whether the entire reachable tail from `start` can be executed safely by an inactive lane
/// (`EXEC == false`) without observable effects, EXEC resurrection, or non-terminating loops.
fn tail_safe_when_inactive(
    g: &Graph,
    preds: &[Vec<usize>],
    start: usize,
    redirect: Option<(usize, usize, usize)>,
) -> Option<(Vec<usize>, bool)> {
    let (tail_nodes, tail_has_cycle) = reachable_tail(g, start, redirect);
    let empty_pred = BTreeSet::new();
    always_written(g, &tail_nodes, true)?;
    let tail_at_end = stays_inactive(g, preds, &tail_nodes, start, &empty_pred)?;
    if tail_has_cycle
        && !loops_safe_when_inactive(
            g,
            preds,
            &tail_nodes,
            &tail_at_end,
            &empty_pred,
            redirect,
        )
    {
        return None;
    }
    Some((tail_nodes, tail_has_cycle))
}

/// Whether every `Guard::Always` write to a variable in `live_out` inside `nodes` can be safely
/// predicated with `Guard::Exec` (only `Stmt::Set` or non-barrier pure wave `Stmt::Effect`, and
/// only at statements where `Var::Exec` still equals its value on entry to `entry`).
fn can_predicate_live_out(
    g: &Graph,
    preds: &[Vec<usize>],
    nodes: &[usize],
    entry: usize,
    live_out: &BTreeSet<Var>,
) -> bool {
    if live_out.is_empty() {
        return true;
    }
    if live_out.contains(&Var::Exec) {
        return false;
    }
    let eq_entry = exec_equals_entry_per_stmt(g, preds, nodes, entry);
    for &n in nodes {
        let flags = &eq_entry[&n];
        for (idx, s) in g.code[n].iter().enumerate() {
            match s {
                Stmt::Set {
                    dst,
                    guard: Guard::Always,
                    ..
                } if live_out.contains(dst) && !flags[idx] => {
                    return false;
                }
                Stmt::Effect {
                    dst: Some(dst),
                    call: Expr::Call(h, _),
                    guard: Guard::Always,
                } if live_out.contains(dst)
                    && (!flags[idx] || !can_predicate_barrier_effect(h.name)) =>
                {
                    return false;
                }
                _ => {}
            }
        }
    }
    true
}

/// Rewrites `Guard::Always` writes to variables in `live_out` inside `nodes` to `Guard::Exec`.
fn predicate_live_out(g: &mut Graph, nodes: &[usize], live_out: &BTreeSet<Var>) {
    if live_out.is_empty() {
        return;
    }
    for &n in nodes {
        for s in &mut g.code[n] {
            match s {
                Stmt::Set { dst, guard, .. }
                    if *guard == Guard::Always && live_out.contains(dst) =>
                {
                    *guard = Guard::Exec;
                }
                Stmt::Effect {
                    dst: Some(dst),
                    guard,
                    ..
                } if *guard == Guard::Always && live_out.contains(dst) => {
                    *guard = Guard::Exec;
                }
                _ => {}
            }
        }
    }
}

/// Converts a non-`exec_skip` early exit (`Exit::Branch` where one arm terminates the shader and
/// the other reaches a workgroup barrier) into `exec = false; jump cont` so every thread of the
/// workgroup converges before the barriers.
fn try_convert_early_exit(
    g: &mut Graph,
    preds: &mut Vec<Vec<usize>>,
    live: &mut Option<Live>,
    branch: usize,
) -> bool {
    let Exit::Branch { taken, fall, .. } = g.exit[branch] else {
        return false;
    };
    if taken == fall {
        return false;
    }
    for (exit_is_taken, exit_arm, cont) in [(true, taken, fall), (false, fall, taken)] {
        if cont == branch || is_empty_end(g, cont) {
            continue;
        }
        let exit_empty = is_empty_end(g, exit_arm);
        if !exit_empty {
            // Allow a single non-empty Exit::End block when it has no barriers, no unsafe
            // unguarded effects, and no Guard::Always write live at `cont`.
            if !matches!(g.exit[exit_arm], Exit::End)
                || g.code[exit_arm].iter().any(is_barrier)
            {
                continue;
            }
            let Some(exit_written) = always_written(g, &[exit_arm], false) else {
                continue;
            };
            if exit_written.contains(&Var::Exec) {
                continue;
            }
            if !exit_written.is_empty() {
                let l = live.get_or_insert_with(|| Live::new(g));
                if exit_written.iter().any(|&v| l.at(cont, v)) {
                    continue;
                }
            }
        }
        let redirect = Some((branch, exit_arm, cont));
        let Some((tail_nodes, _)) = tail_safe_when_inactive(g, preds, cont, redirect) else {
            continue;
        };
        // A non-empty `Exit::End` block that `cont` also flows into is a shared post-barrier
        // epilogue, not a dedicated early-exit block.
        if !exit_empty && tail_nodes.contains(&exit_arm) {
            continue;
        }
        if !tail_nodes
            .iter()
            .any(|&n| g.code[n].iter().any(is_barrier))
        {
            continue;
        }

        let mut kill_code = if exit_empty {
            Vec::with_capacity(1)
        } else {
            let mut c = g.code[exit_arm].clone();
            c.reserve(1);
            c
        };
        kill_code.push(Stmt::Set {
            dst: Var::Exec,
            value: Expr::Bool(false),
            guard: Guard::Always,
        });
        let kill = g.push(kill_code, Exit::Jump(cont));
        if let Exit::Branch { taken, fall, .. } = &mut g.exit[branch] {
            if exit_is_taken {
                *taken = kill;
            } else {
                *fall = kill;
            }
        }
        preds[exit_arm].retain(|&p| p != branch);
        preds.push(vec![branch]);
        if !preds[cont].contains(&kill) {
            preds[cont].push(kill);
        }
        *live = None;
        return true;
    }
    false
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

/// Attempts to drop a barrier `exec_skip` at `branch` (`entry`, `target`). When
/// `allow_internal_ends` is `false`, regions that contain internal `Exit::End` blocks are deferred
/// so early-exit convergence can first convert any pre-barrier `Exit::End` into `exec = false`
/// before the barrier rather than jumping past it to `target`.
fn try_drop_barrier_skip(
    g: &mut Graph,
    preds: &mut [Vec<usize>],
    live: &mut Option<Live>,
    branch: usize,
    entry: usize,
    target: usize,
    allow_internal_ends: bool,
) -> bool {
    let Some(br) = barrier_region(g, preds, branch, entry, target) else {
        return false;
    };
    if !allow_internal_ends && !br.internal_ends.is_empty() {
        return false;
    }
    if !br.nodes.iter().any(|&n| g.code[n].iter().any(is_barrier)) {
        return false;
    }
    let Some(written) = always_written(g, &br.nodes, true) else {
        return false;
    };
    // Collect Guard::Always writes that are live at `target`, excluding `Var::Exec` because
    // `stays_inactive` proves an inactive lane exits with `Exec == false` (matching the skip).
    let live_out: BTreeSet<Var> = {
        let l = live.get_or_insert_with(|| Live::new(g));
        written
            .iter()
            .copied()
            .filter(|&v| v != Var::Exec && l.at(target, v))
            .collect()
    };
    // Internal Exit::End nodes cannot safely run Guard::Always writes to variables live at
    // `target` (an active lane exiting via `e` would still execute them before `exec = false`).
    let internal_ends_ok = if br.internal_ends.is_empty() {
        true
    } else {
        let ends_clobber = br.internal_ends.iter().any(|&e| {
            always_written(g, &[e], true).is_none_or(|w| {
                let l = live.get_or_insert_with(|| Live::new(g));
                w.iter().any(|&v| v != Var::Exec && l.at(target, v))
            })
        });
        !ends_clobber && tail_safe_when_inactive(g, preds, target, None).is_some()
    };
    let can_drop = internal_ends_ok
        && can_predicate_live_out(g, preds, &br.nodes, entry, &live_out)
        && stays_inactive(g, preds, &br.nodes, entry, &live_out).is_some_and(|at_end| {
            !predicated_affects_branch(g, preds, &br.nodes, &at_end, &live_out)
                && (!br.has_cycle
                    || loops_safe_when_inactive(g, preds, &br.nodes, &at_end, &live_out, None))
        });
    if !can_drop {
        return false;
    }
    predicate_live_out(g, &br.nodes, &live_out);
    for &e in &br.internal_ends {
        g.code[e].push(Stmt::Set {
            dst: Var::Exec,
            value: Expr::Bool(false),
            guard: Guard::Always,
        });
        g.exit[e] = Exit::Jump(target);
        if !preds[target].contains(&e) {
            preds[target].push(e);
        }
    }
    g.exit[branch] = Exit::Jump(entry);
    preds[target].retain(|&p| p != branch);
    if !live_out.is_empty() || !br.internal_ends.is_empty() {
        *live = None;
    }
    true
}

/// Drops skips around derivative taking samples and workgroup barriers, and converts early
/// `Exit::End` returns before barriers into `exec = false`. Returns how many skips/exits changed.
pub(super) fn drop_skips(g: &mut Graph) -> usize {
    let has_deriv = g.code.iter().flatten().any(takes_derivatives);
    let has_barrier = g.code.iter().flatten().any(is_barrier);
    if !(has_deriv || has_barrier) {
        return 0;
    }
    let mut preds = vec![Vec::new(); g.len()];
    for n in 0..g.len() {
        for s in g.successors(n) {
            if !preds[s].contains(&n) {
                preds[s].push(n);
            }
        }
    }
    let (mut rels, mut reachable) = exec_rels(g, &preds);
    if !has_barrier && !(0..g.len()).any(|b| reachable[b] && exec_skip(g, &rels, b).is_some()) {
        return 0;
    }
    let empty_pred = BTreeSet::new();
    let mut live = None;
    let mut dropped = 0;
    let mut rels_dirty = false;
    loop {
        let before = dropped;
        // Phase 1: drop derivative skips and barrier `exec_skip`s without internal `Exit::End`.
        let len = g.len();
        for branch in 0..len {
            if rels_dirty {
                (rels, reachable) = exec_rels(g, &preds);
                rels_dirty = false;
            }
            if branch >= reachable.len() || !reachable[branch] {
                continue;
            }
            let Some((entry, target)) = exec_skip(g, &rels, branch) else {
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
                    if stays_inactive(g, &preds, &nodes, entry, &empty_pred).is_none() {
                        return false;
                    }
                    // The liveness of the graph as it came in. A dropped skip only takes an
                    // edge away, which makes nothing live that was not.
                    let live = live.get_or_insert_with(|| Live::new(g));
                    !written.iter().any(|&v| live.at(target, v))
                });
            if drop_deriv {
                g.exit[branch] = Exit::Jump(entry);
                preds[target].retain(|&p| p != branch);
                dropped += 1;
                rels_dirty = true;
                continue;
            }
            if has_barrier
                && try_drop_barrier_skip(g, &mut preds, &mut live, branch, entry, target, false)
            {
                dropped += 1;
                rels_dirty = true;
            }
        }
        if dropped != before {
            continue;
        }
        // Phase 2: convert early `Exit::End` returns that skip a barrier into `exec = false`.
        if has_barrier {
            let len = g.len();
            for branch in 0..len {
                if rels_dirty {
                    (rels, reachable) = exec_rels(g, &preds);
                    rels_dirty = false;
                }
                if branch >= reachable.len() || !reachable[branch] {
                    continue;
                }
                if try_convert_early_exit(g, &mut preds, &mut live, branch) {
                    dropped += 1;
                    rels_dirty = true;
                }
            }
        }
        if dropped != before {
            continue;
        }
        // Phase 3: drop barrier `exec_skip`s whose remaining internal `Exit::End` blocks sit
        // after barriers and can safely become `exec = false; jump target`.
        if has_barrier {
            let len = g.len();
            for branch in 0..len {
                if rels_dirty {
                    (rels, reachable) = exec_rels(g, &preds);
                    rels_dirty = false;
                }
                if branch >= reachable.len() || !reachable[branch] {
                    continue;
                }
                let Some((entry, target)) = exec_skip(g, &rels, branch) else {
                    continue;
                };
                if try_drop_barrier_skip(g, &mut preds, &mut live, branch, entry, target, true) {
                    dropped += 1;
                    rels_dirty = true;
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn set(dst: Var, value: Expr, guard: Guard) -> Stmt {
        Stmt::Set { dst, value, guard }
    }

    fn barrier(pc: u32) -> Stmt {
        Stmt::Barrier { pc }
    }

    fn exec_preds(g: &Graph) -> Vec<Vec<usize>> {
        let mut preds = vec![Vec::new(); g.len()];
        for n in 0..g.len() {
            for s in g.successors(n) {
                if !preds[s].contains(&n) {
                    preds[s].push(n);
                }
            }
        }
        preds
    }

    #[test]
    fn cross_block_scc_saved_mask_and_vcc_exec_skips_are_dropped() {
        // 1. Cross-block SCC skip: block 0 narrows EXEC and sets SCC = EXEC, block 1 does scalar
        //    work without touching SCC/EXEC and branches on !SCC around a barrier.
        let mut g = Graph {
            code: vec![
                vec![
                    set(Var::Flag(0), Expr::bin(Bin::LAnd, Expr::Var(Var::Vcc), Expr::Var(Var::Exec)), Guard::Always),
                    set(Var::Mask(10), Expr::Var(Var::Exec), Guard::Always),
                    set(Var::Exec, Expr::Var(Var::Flag(0)), Guard::Always),
                    set(Var::Scc, Expr::Var(Var::Flag(0)), Guard::Always),
                ],
                vec![set(Var::S(20), Expr::U32(7), Guard::Always)],
                vec![barrier(10)],
                vec![],
            ],
            exit: vec![
                Exit::Jump(1),
                Exit::Branch {
                    cond: Expr::un(Un::LNot, Expr::Var(Var::Scc)),
                    taken: 3,
                    fall: 2,
                },
                Exit::Jump(3),
                Exit::End,
            ],
        };
        assert_eq!(drop_skips(&mut g), 1);
        assert!(matches!(g.exit[1], Exit::Jump(2)));

        // 2. Saved mask copied from EXEC, spilled across a block, and compared to 0 via SCC.
        let mut g2 = Graph {
            code: vec![
                vec![
                    set(Var::Mask(8), Expr::Var(Var::Exec), Guard::Always),
                    set(
                        Var::Spill { reg: 30, lane: 0 },
                        Expr::un(Un::BoolToU32, Expr::Var(Var::Mask(8))),
                        Guard::Always,
                    ),
                ],
                vec![
                    set(
                        Var::Mask(12),
                        Expr::un(Un::U32ToBool, Expr::Var(Var::Spill { reg: 30, lane: 0 })),
                        Guard::Always,
                    ),
                    set(Var::Scc, Expr::un(Un::LNot, Expr::Var(Var::Mask(12))), Guard::Always),
                ],
                vec![barrier(20)],
                vec![],
            ],
            exit: vec![
                Exit::Jump(1),
                Exit::Branch {
                    cond: Expr::Var(Var::Scc),
                    taken: 3,
                    fall: 2,
                },
                Exit::Jump(3),
                Exit::End,
            ],
        };
        assert_eq!(drop_skips(&mut g2), 1);
        assert!(matches!(g2.exit[1], Exit::Jump(2)));

        // 3. `v_cmp` (Vcc = Exec && cond) + `s_and_saveexec` (Exec = Vcc && Exec == Vcc) followed
        //    by `s_cbranch_vccz` (!Vcc) across a block.
        let mut g3 = Graph {
            code: vec![
                vec![
                    set(
                        Var::Vcc,
                        Expr::bin(
                            Bin::LAnd,
                            Expr::Var(Var::Exec),
                            Expr::bin(Bin::ULt, Expr::Var(Var::V(0)), Expr::Var(Var::S(4))),
                        ),
                        Guard::Always,
                    ),
                    set(Var::Flag(0), Expr::bin(Bin::LAnd, Expr::Var(Var::Vcc), Expr::Var(Var::Exec)), Guard::Always),
                    set(Var::Mask(10), Expr::Var(Var::Exec), Guard::Always),
                    set(Var::Exec, Expr::Var(Var::Flag(0)), Guard::Always),
                    set(Var::Scc, Expr::Var(Var::Flag(0)), Guard::Always),
                ],
                vec![set(Var::S(20), Expr::U32(99), Guard::Always)],
                vec![barrier(30)],
                vec![],
            ],
            exit: vec![
                Exit::Jump(1),
                Exit::Branch {
                    cond: Expr::un(Un::LNot, Expr::Var(Var::Vcc)),
                    taken: 3,
                    fall: 2,
                },
                Exit::Jump(3),
                Exit::End,
            ],
        };
        assert_eq!(drop_skips(&mut g3), 1);
        assert!(matches!(g3.exit[1], Exit::Jump(2)));
    }

    #[test]
    fn internal_early_exit_inside_barrier_region_converts_to_exec_false() {
        // Outer exec_skip jumps to non-empty `target` (block 4, Guard::Exec write then End).
        // Inside the barrier region, block 1 has an internal Exit::End (block 2).
        let mut g = Graph {
            code: vec![
                vec![],
                vec![],
                vec![], // internal early Exit::End
                vec![
                    barrier(10),
                    set(Var::V(2), Expr::U32(1), Guard::Exec),
                ],
                vec![set(Var::V(3), Expr::Var(Var::V(2)), Guard::Exec)],
            ],
            exit: vec![
                Exit::Branch {
                    cond: Expr::un(Un::LNot, Expr::Var(Var::Exec)),
                    taken: 4,
                    fall: 1,
                },
                Exit::Branch {
                    cond: Expr::Var(Var::Scc),
                    taken: 2,
                    fall: 3,
                },
                Exit::End,
                Exit::Jump(4),
                Exit::End,
            ],
        };
        let dropped = drop_skips(&mut g);
        assert!(dropped >= 1);
        // Every remaining Exit::End must come after the barrier (no early return before barrier).
        let end_count = g.exit.iter().filter(|e| matches!(e, Exit::End)).count();
        assert_eq!(end_count, 1);
        assert!(matches!(g.exit[0], Exit::Jump(1)));
    }

    #[test]
    fn barrier_region_exceeding_4096_nodes_is_not_capped() {
        // Chain of 4200 linear blocks inside an exec_skip around a barrier.
        const CHAIN: usize = 4200;
        let end = CHAIN + 1;
        let mut code = Vec::with_capacity(end + 1);
        let mut exit = Vec::with_capacity(end + 1);
        code.push(Vec::new());
        exit.push(Exit::Branch {
            cond: Expr::un(Un::LNot, Expr::Var(Var::Exec)),
            taken: end,
            fall: 1,
        });
        for i in 1..=CHAIN {
            if i == CHAIN / 2 {
                code.push(vec![barrier(i as u32)]);
            } else {
                code.push(vec![set(Var::V(1), Expr::U32(i as u32), Guard::Exec)]);
            }
            exit.push(Exit::Jump(i + 1));
        }
        code.push(Vec::new());
        exit.push(Exit::End);
        let mut g = Graph { code, exit };
        assert_eq!(drop_skips(&mut g), 1);
        assert!(matches!(g.exit[0], Exit::Jump(1)));
    }

    #[test]
    fn live_out_scalar_write_is_predicated_and_wave_effect_is_allowed() {
        // Mid-shader barrier skip writes scalar S(12) (live at join) and calls a pure wave
        // effect `READ_LANE` under Guard::Always. Both must be handled cleanly.
        let mut g = Graph {
            code: vec![
                vec![set(Var::S(12), Expr::U32(1), Guard::Always)],
                vec![
                    Stmt::Effect {
                        dst: Some(Var::Tmp(0)),
                        call: Expr::Call(
                            &crate::lift::wave::READ_LANE,
                            vec![Expr::Var(Var::V(0)), Expr::U32(0)],
                        ),
                        guard: Guard::Always,
                    },
                    set(Var::S(12), Expr::Var(Var::Tmp(0)), Guard::Always),
                    barrier(10),
                ],
                vec![set(Var::V(5), Expr::Var(Var::S(12)), Guard::Exec)],
            ],
            exit: vec![
                Exit::Branch {
                    cond: Expr::un(Un::LNot, Expr::Var(Var::Exec)),
                    taken: 2,
                    fall: 1,
                },
                Exit::Jump(2),
                Exit::End,
            ],
        };
        assert_eq!(drop_skips(&mut g), 1);
        assert!(matches!(g.exit[0], Exit::Jump(1)));
        // The live-out write to S(12) in block 1 must now be Guard::Exec.
        assert!(g.code[1].iter().any(|s| matches!(
            s,
            Stmt::Set {
                dst: Var::S(12),
                guard: Guard::Exec,
                ..
            }
        )));
    }

    #[test]
    fn early_return_that_is_sole_loop_exit_is_not_converted() {
        // A loop whose only exit is an early `Exit::End` must NOT convert that exit into
        // `exec = false; jump cont` (which would create an infinite loop).
        let mut g = Graph {
            code: vec![
                vec![set(Var::S(12), Expr::U32(0), Guard::Always)],
                vec![set(
                    Var::Scc,
                    Expr::bin(Bin::ULe, Expr::U32(10), Expr::Var(Var::S(12))),
                    Guard::Always,
                )],
                vec![
                    barrier(10),
                    set(
                        Var::S(12),
                        Expr::bin(Bin::Add, Expr::Var(Var::S(12)), Expr::U32(1)),
                        Guard::Always,
                    ),
                ],
                vec![],
            ],
            exit: vec![
                Exit::Jump(1),
                Exit::Branch {
                    cond: Expr::Var(Var::Scc),
                    taken: 3,
                    fall: 2,
                },
                Exit::Jump(1),
                Exit::End,
            ],
        };
        let preds = exec_preds(&g);
        let (rels, _) = exec_rels(&g, &preds);
        assert!(exec_skip(&g, &rels, 1).is_none());
        assert_eq!(drop_skips(&mut g), 0);
    }
}
