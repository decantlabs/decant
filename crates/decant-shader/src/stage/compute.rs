//! The compute stage: the registers a compute program finds at entry, and what it owes at the
//! end.
//!
//! Entry state, measured on the compute programs of the title:
//!
//! * `v0`, `v1`, `v2` hold the thread id in the group, x, y and z. Programs read `v0` alone,
//!   `v0` and `v1`, or all three, following the dimensions of their thread group.
//! * The user data registers start at `s4` and hold the root layout of the container
//!   (`StageProgram::user_data`). The three registers after the last one hold the group id x,
//!   y and z, always all three: programs whose group is wide in x only still read the third one
//!   when they use the group id z, and programs that use only y read the second.
//! * EXEC holds every lane of the wave. A wave at the end of a group that is no multiple of
//!   the wave size has fewer lanes, and HLSL has no invocation for the missing threads, which is
//!   the same.
//!
//! The thread group size comes from `PSV0` (`StageProgram::threads`), the size of the group
//! shared memory from the FSSM block (`StageProgram::lds_bytes`): measured on the compute
//! programs, it is not zero in exactly the programs with LDS instructions, never below the
//! highest constant offset they use, and at most 32 KiB.
//!
//! Group shared memory (`ds_*`) is `crate::lift::ds`. A compute program has no exports.

use super::refuse;
use crate::api::{StageProgram, UserDataKind};
use crate::ir::{Expr, Function, Guard, Input, Node, Stmt, Var};
use crate::refusal::Refusal;

/// The first register after the user data: the group id x. The group id y and z follow.
pub fn group_id_register(program: &StageProgram) -> u8 {
    program
        .user_data
        .iter()
        .map(|slot| {
            let width = match slot.kind {
                UserDataKind::Constants { count, .. } => count,
                UserDataKind::RootSrv { .. } => 2,
                UserDataKind::Table | UserDataKind::ResourceHeap | UserDataKind::SamplerHeap => 1,
            };
            u32::from(slot.sgpr) + width
        })
        .max()
        .map_or(program.stage.first_user_sgpr(), |end| end.min(104) as u8)
}

fn set(dst: Var, value: Expr) -> Stmt {
    Stmt::Set {
        dst,
        value,
        guard: Guard::Always,
    }
}

pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    let mut out = vec![set(Var::Exec, Expr::Bool(true))];
    for axis in 0..3u8 {
        out.push(set(
            Var::V(axis),
            Expr::Input(Input::LocalThreadId { axis }),
        ));
    }
    let first = group_id_register(program);
    for axis in 0..3u8 {
        out.push(set(
            Var::S(first + axis),
            Expr::Input(Input::GroupId { axis }),
        ));
    }
    out
}

pub fn finish(program: &StageProgram, func: &mut Function) -> Result<(), Refusal> {
    if program.threads.contains(&0) {
        return Err(refuse(
            "compute-threads-unknown",
            format!("{:?}", program.threads),
        ));
    }
    let mut export = None;
    func.for_each_stmt(&mut |s| {
        if let Stmt::Export(e) = s {
            export.get_or_insert(e.pc);
        }
    });
    if let Some(pc) = export {
        return Err(refuse("compute-export", "").at(pc));
    }
    // A barrier holds every wave of the group, whatever EXEC says: the console compiler only
    // places it where the whole group arrives.
    func.for_each_block_mut(&mut |stmts| {
        for s in stmts.iter_mut() {
            if matches!(s, Stmt::Barrier { .. }) {
                *s = Stmt::Effect {
                    dst: None,
                    call: Expr::Call(&crate::lift::ds::BARRIER, Vec::new()),
                    guard: Guard::Always,
                };
            }
        }
    });
    if crate::lift::ds::uses_lds(func) {
        // The allocation the console compiler made for the group (FSSM byte 0x10C), which no
        // access exceeds. Without it, the largest array Direct3D 12 allows.
        func.groupshared_dwords = match program.lds_bytes {
            0 => crate::lift::ds::LDS_DWORDS,
            bytes => bytes.div_ceil(4).min(crate::lift::ds::LDS_DWORDS),
        };
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct LdsFlags {
    read: bool,
    write: bool,
    atomic: bool,
}

impl LdsFlags {
    const fn union(self, other: Self) -> Self {
        Self {
            read: self.read || other.read,
            write: self.write || other.write,
            atomic: self.atomic || other.atomic,
        }
    }

    /// Whether executing `next` after unsynchronized `self` creates a cross-wave RAW, WAR, or
    /// write/atomic race when a 64-lane console wave is split across two hardware Wave32 waves.
    const fn conflicts_with(self, next: Self) -> bool {
        (self.write && (next.read || next.atomic))
            || (self.atomic && (next.read || next.write))
            || (self.read && (next.write || next.atomic))
    }
}

fn classify_lds_helper(name: &str) -> LdsFlags {
    if !name.starts_with("decant_lds_") {
        return LdsFlags::default();
    }
    if matches!(name, "decant_lds_load" | "decant_lds_load8" | "decant_lds_load16") {
        LdsFlags {
            read: true,
            ..LdsFlags::default()
        }
    } else if matches!(
        name,
        "decant_lds_store" | "decant_lds_store8" | "decant_lds_store16" | "decant_lds_store_bits"
    ) {
        LdsFlags {
            write: true,
            ..LdsFlags::default()
        }
    } else {
        LdsFlags {
            atomic: true,
            ..LdsFlags::default()
        }
    }
}

fn expr_lds_info(e: &Expr) -> (LdsFlags, bool) {
    match e {
        Expr::Call(h, args) => {
            let mut flags = classify_lds_helper(h.name);
            let mut sync = h.name == crate::lift::ds::BARRIER.name
                || crate::lift::wave::has_group_barrier(h);
            for a in args {
                let (f, s) = expr_lds_info(a);
                flags = flags.union(f);
                sync |= s;
            }
            (flags, sync)
        }
        Expr::Un(_, a) | Expr::CodeWord(a) => expr_lds_info(a),
        Expr::Bin(_, a, b) => {
            let (fa, sa) = expr_lds_info(a);
            let (fb, sb) = expr_lds_info(b);
            (fa.union(fb), sa || sb)
        }
        Expr::Select(c, a, b) => {
            let (fc, sc) = expr_lds_info(c);
            let (fa, sa) = expr_lds_info(a);
            let (fb, sb) = expr_lds_info(b);
            (fc.union(fa).union(fb), sc || sa || sb)
        }
        Expr::BufferWord { offset, .. } | Expr::VectorLoad { offset, .. } => expr_lds_info(offset),
        Expr::Load(l) => expr_lds_info(&l.offset),
        Expr::BvhWord(b, _) => {
            let mut flags = LdsFlags::default();
            let mut sync = false;
            for d in b.descriptor.iter().chain(&b.address) {
                let (f, s) = expr_lds_info(d);
                flags = flags.union(f);
                sync |= s;
            }
            (flags, sync)
        }
        _ => (LdsFlags::default(), false),
    }
}

fn stmt_lds_info(s: &Stmt) -> (LdsFlags, bool) {
    match s {
        Stmt::Set { value, .. } => expr_lds_info(value),
        Stmt::Effect { call, .. } => expr_lds_info(call),
        Stmt::Barrier { .. } => (LdsFlags::default(), true),
        Stmt::Output { value, .. } => expr_lds_info(value),
        Stmt::Discard { cond } => expr_lds_info(cond),
        _ => (LdsFlags::default(), false),
    }
}

fn barrier_stmt() -> Stmt {
    Stmt::Effect {
        dst: None,
        call: Expr::Call(&crate::lift::ds::BARRIER, Vec::new()),
        guard: Guard::Always,
    }
}

#[derive(Clone, Copy, Default)]
struct SubtreeSummary {
    /// LDS accesses that can occur on some path before any internal barrier.
    in_needs: LdsFlags,
    /// LDS accesses that may still be unsynchronized when exiting the subtree.
    out_dirty: LdsFlags,
    /// Whether every normal path through the subtree executes a workgroup barrier.
    always_syncs: bool,
    /// Whether any path inside the subtree returns early from the shader.
    has_return: bool,
    /// Whether any path inside the subtree breaks or continues an enclosing loop (not a nested one).
    has_break_or_continue: bool,
    /// Whether any top-level statement in the subtree is already an explicit workgroup barrier.
    has_top_sync: bool,
}

fn summarize_nodes(nodes: &[Node]) -> SubtreeSummary {
    let mut in_needs = LdsFlags::default();
    let mut out_dirty = LdsFlags::default();
    let mut always_syncs = false;
    let mut has_return = false;
    let mut has_break_or_continue = false;
    let mut has_top_sync = false;

    for n in nodes {
        match n {
            Node::Block(stmts) => {
                for s in stmts {
                    let (flags, sync) = stmt_lds_info(s);
                    if sync {
                        if !always_syncs {
                            in_needs = in_needs.union(flags);
                        }
                        out_dirty = LdsFlags::default();
                        always_syncs = true;
                        has_top_sync = true;
                    } else {
                        if !always_syncs {
                            in_needs = in_needs.union(flags);
                        }
                        out_dirty = out_dirty.union(flags);
                    }
                }
            }
            Node::If { cond, then, els } => {
                let (cf, cs) = expr_lds_info(cond);
                if !always_syncs {
                    in_needs = in_needs.union(cf);
                }
                if cs {
                    out_dirty = LdsFlags::default();
                    always_syncs = true;
                }
                let t = summarize_nodes(then);
                let e = summarize_nodes(els);
                has_return |= t.has_return || e.has_return;
                has_break_or_continue |= t.has_break_or_continue || e.has_break_or_continue;
                if !always_syncs {
                    in_needs = in_needs.union(t.in_needs).union(e.in_needs);
                }
                let both_sync = !els.is_empty() && t.always_syncs && e.always_syncs;
                if both_sync {
                    out_dirty = t.out_dirty.union(e.out_dirty);
                    always_syncs = true;
                } else {
                    out_dirty = out_dirty.union(t.out_dirty).union(e.out_dirty);
                }
            }
            Node::Loop(inner) => {
                let sub = summarize_nodes(inner);
                has_return |= sub.has_return;
                if !always_syncs {
                    in_needs = in_needs.union(sub.in_needs);
                }
                out_dirty = out_dirty.union(sub.out_dirty);
            }
            Node::Once(inner) => {
                let sub = summarize_nodes(inner);
                has_return |= sub.has_return;
                has_break_or_continue |= sub.has_break_or_continue;
                if !always_syncs {
                    in_needs = in_needs.union(sub.in_needs);
                }
                out_dirty = out_dirty.union(sub.out_dirty);
            }
            Node::Return => {
                has_return = true;
            }
            Node::Break | Node::Continue => {
                has_break_or_continue = true;
            }
        }
    }

    SubtreeSummary {
        in_needs,
        out_dirty,
        always_syncs,
        has_return,
        has_break_or_continue,
        has_top_sync,
    }
}

fn has_direct_break_or_continue(nodes: &[Node]) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break | Node::Continue | Node::Return => true,
        Node::If { then, els, .. } => {
            has_direct_break_or_continue(then) || has_direct_break_or_continue(els)
        }
        Node::Once(inner) => has_direct_break_or_continue(inner),
        Node::Loop(_) | Node::Block(_) => false,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// Every thread entering the slice that does not exit reaches the fallthrough converged.
    Converged,
    /// Every thread entering the slice exits unconditionally (nothing falls through).
    Exited,
    /// Threads diverged on an early return/break/continue; fallthrough is not workgroup-converged.
    Diverged,
}

fn sync_converged_slice(
    nodes: Vec<Node>,
    mut dirty: LdsFlags,
    non_uniform: &std::collections::BTreeSet<Var>,
    resources: &[crate::ir::Resource],
    in_loop: bool,
) -> (Vec<Node>, LdsFlags, Flow) {
    let mut out = Vec::with_capacity(nodes.len());
    let mut flow = Flow::Converged;

    for node in nodes {
        if flow != Flow::Converged {
            out.push(node);
            continue;
        }
        match node {
            Node::Block(stmts) => {
                let mut new_stmts = Vec::with_capacity(stmts.len());
                for s in stmts {
                    let (flags, sync) = stmt_lds_info(&s);
                    if sync {
                        new_stmts.push(s);
                        dirty = LdsFlags::default();
                    } else {
                        if dirty.conflicts_with(flags) {
                            new_stmts.push(barrier_stmt());
                            dirty = LdsFlags::default();
                        }
                        new_stmts.push(s);
                        dirty = dirty.union(flags);
                    }
                }
                out.push(Node::Block(new_stmts));
            }
            Node::If { cond, then, els } => {
                let uniform_cond =
                    crate::lift::wave::expr_is_workgroup_uniform(&cond, non_uniform, resources);
                let (cf, cs) = expr_lds_info(&cond);
                if uniform_cond && !cs && !cf.read && !cf.write && !cf.atomic {
                    let (new_then, then_dirty, then_flow) =
                        sync_converged_slice(then, dirty, non_uniform, resources, in_loop);
                    let (new_els, els_dirty, els_flow) =
                        sync_converged_slice(els, dirty, non_uniform, resources, in_loop);
                    out.push(Node::If {
                        cond,
                        then: new_then,
                        els: new_els,
                    });
                    match (then_flow, els_flow) {
                        (Flow::Converged, Flow::Converged) => {
                            dirty = then_dirty.union(els_dirty);
                        }
                        (Flow::Converged, Flow::Exited) => {
                            dirty = then_dirty;
                        }
                        (Flow::Exited, Flow::Converged) => {
                            dirty = els_dirty;
                        }
                        (Flow::Exited, Flow::Exited) => {
                            flow = Flow::Exited;
                        }
                        (Flow::Diverged, _) | (_, Flow::Diverged) => {
                            flow = Flow::Diverged;
                        }
                    }
                } else {
                    let sub = summarize_nodes(&[Node::If {
                        cond: cond.clone(),
                        then: then.clone(),
                        els: els.clone(),
                    }]);
                    if dirty.conflicts_with(sub.in_needs) {
                        out.push(Node::Block(vec![barrier_stmt()]));
                        dirty = LdsFlags::default();
                    }
                    out.push(Node::If { cond, then, els });
                    if sub.has_return || (in_loop && sub.has_break_or_continue) {
                        flow = Flow::Diverged;
                    } else if sub.always_syncs {
                        dirty = sub.out_dirty;
                    } else {
                        dirty = dirty.union(sub.out_dirty);
                    }
                }
            }
            Node::Loop(inner) => {
                let sub = summarize_nodes(&inner);
                if dirty.conflicts_with(sub.in_needs) {
                    out.push(Node::Block(vec![barrier_stmt()]));
                    dirty = LdsFlags::default();
                }
                // Only insert barriers inside a loop when every thread in the workgroup executes
                // the same iterations (proven uniform break/continue or existing top-level barrier).
                let uniform_loop =
                    crate::lift::wave::loop_breaks_uniform(&inner, non_uniform, resources);
                let loop_converged = !sub.has_return && (sub.has_top_sync || uniform_loop);
                let new_inner = if loop_converged {
                    let (mut synced, end_dirty, inner_flow) = sync_converged_slice(
                        inner,
                        LdsFlags::default(),
                        non_uniform,
                        resources,
                        true,
                    );
                    let after_sub = summarize_nodes(&synced);
                    if inner_flow == Flow::Converged
                        && end_dirty.conflicts_with(after_sub.in_needs)
                        && (!has_direct_break_or_continue(&synced)
                            || crate::lift::wave::loop_breaks_uniform(
                                &synced,
                                non_uniform,
                                resources,
                            ))
                    {
                        synced.push(Node::Block(vec![barrier_stmt()]));
                    }
                    synced
                } else {
                    inner
                };
                let final_sub = summarize_nodes(&new_inner);
                out.push(Node::Loop(new_inner));
                if final_sub.has_return {
                    flow = Flow::Diverged;
                } else {
                    dirty = dirty.union(final_sub.out_dirty);
                }
            }
            Node::Once(inner) => {
                let sub = summarize_nodes(&inner);
                if dirty.conflicts_with(sub.in_needs) {
                    out.push(Node::Block(vec![barrier_stmt()]));
                    dirty = LdsFlags::default();
                }
                out.push(Node::Once(inner));
                if sub.has_return || (in_loop && sub.has_break_or_continue) {
                    flow = Flow::Diverged;
                } else {
                    dirty = dirty.union(sub.out_dirty);
                }
            }
            Node::Return => {
                out.push(Node::Return);
                flow = Flow::Exited;
            }
            other @ (Node::Break | Node::Continue) => {
                out.push(other);
                flow = Flow::Exited;
            }
        }
    }

    (out, dirty, flow)
}

/// Inserts workgroup barriers at converged control-flow points between unsynchronized LDS writes,
/// atomics, and reads when a 64-lane console compute shader executes on narrower (Wave32) hardware.
///
/// On console Wave64, all 64 lanes execute in SIMD lockstep so intra-wave LDS init -> atomic ->
/// read sequences require no explicit `s_barrier`. On Wave32 hardware, threadgroups with more than
/// 32 threads split each console wave across multiple asynchronous hardware waves, creating RAW,
/// WAR, and WAW data races unless workgroup barriers separate conflicting LDS phases.
///
/// Barriers are ONLY inserted at workgroup-converged points (never inside divergent `if` branches,
/// non-uniform loops, or after an early `return`).
pub fn sync_lds_for_narrow_waves(program: &StageProgram, func: &mut Function) {
    let total_threads = program
        .threads
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
        .unwrap_or(0);
    if func.lanes != 64 || total_threads <= 32 || !crate::lift::ds::uses_lds(func) {
        return;
    }
    let non_uniform = crate::lift::wave::compute_non_uniform_vars(&func.body, &func.resources);
    let old_body = std::mem::take(&mut func.body);
    let (new_body, _, _) =
        sync_converged_slice(old_body, LdsFlags::default(), &non_uniform, &func.resources, false);
    func.body = new_body;
}
