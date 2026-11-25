//! Resources: from the console descriptor model to Direct3D bindings.
//!
//! A console program has no registers like `t3`. It gets user data registers from the root
//! signature (see [`root`]), loads descriptor words from the tables those point at, and hands
//! the registers holding the words to an image, buffer or sampler operand. So three steps:
//!
//! 1. [`trace`] follows the words from the load to the use and gives every use a
//!    [`crate::ir::ResId`] with an [`crate::ir::Origin`]: table and offset, root descriptor, or
//!    constants.
//! 2. [`dce`] removes the register traffic that only carried descriptor words.
//! 3. [`place`] turns each origin into a [`crate::api::Placement`] with the root signature in hand.

pub mod flow;
pub mod place;
pub mod root;
pub mod trace;

use std::collections::BTreeSet;

use crate::ir::{BufferOp, Desc, Expr, Function, ImageOp, Node, Origin, Resource, Stmt, Var};

pub use place::place;
pub use trace::trace;

/// Calls `f` on every variable an expression reads.
pub fn reads(e: &Expr, f: &mut impl FnMut(Var)) {
    match e {
        Expr::Var(v) => f(*v),
        Expr::Un(_, a) => reads(a, f),
        Expr::Bin(_, a, b) => {
            reads(a, f);
            reads(b, f);
        }
        Expr::Select(c, a, b) => {
            reads(c, f);
            reads(a, f);
            reads(b, f);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| reads(a, f)),
        Expr::Load(l) => {
            reads(&l.offset, f);
            // The base registers: an address pair or the four descriptor words.
            for k in 0..if l.buffer { 4 } else { 2 } {
                f(Var::S(l.base + k));
            }
        }
        Expr::BufferWord { offset, .. } => reads(offset, f),
        Expr::CodeWord(at) => reads(at, f),
        Expr::VectorLoad { base, offset, .. } => {
            reads(offset, f);
            f(Var::V(*base));
            f(Var::V(*base + 1));
        }
        Expr::CodeAddress(_) => {}
        Expr::U32(_)
        | Expr::U64(_)
        | Expr::Bool(_)
        | Expr::Input(_)
        | Expr::UserData(_)
        | Expr::Mirror(_) => {}
    }
}

fn desc_reads(d: Desc, words: u8, f: &mut impl FnMut(Var)) {
    if let Desc::Regs(first) = d {
        (0..words).for_each(|k| f(Var::S(first + k)));
    }
}

/// Calls `f` on every variable a statement reads.
pub fn stmt_reads(s: &Stmt, f: &mut impl FnMut(Var)) {
    match s {
        Stmt::Set { value, guard, .. } => {
            reads(value, f);
            if *guard == crate::ir::Guard::Exec {
                f(Var::Exec);
            }
        }
        Stmt::Image(op, _) => {
            let ImageOp {
                address,
                image,
                sampler,
                ..
            } = op;
            address.iter().for_each(|a| reads(a, f));
            desc_reads(*image, 8, f);
            if let Some(s) = sampler {
                desc_reads(*s, 4, f);
            }
            op.data_reads().for_each(|r| f(Var::V(r)));
            f(Var::Exec);
        }
        Stmt::Buffer(
            BufferOp {
                index,
                offset,
                buffer,
                store,
                data,
                count,
                ..
            },
            _,
        ) => {
            if let Some(i) = index {
                reads(i, f);
            }
            reads(offset, f);
            desc_reads(*buffer, 4, f);
            if *store {
                (0..*count).for_each(|k| f(Var::V(data + k)));
            }
            f(Var::Exec);
        }
        Stmt::Export(e) => {
            // A compressed export reads two registers, a plain one a register per enabled component.
            let used = if e.compressed { 2 } else { 4 };
            (0..used)
                .filter(|k| e.compressed || (e.enabled >> k) & 1 != 0)
                .for_each(|k| f(Var::V(e.src[k])));
            f(Var::Exec);
        }
        Stmt::Output { value, .. } => reads(value, f),
        Stmt::Discard { cond } => reads(cond, f),
        Stmt::Effect { call, guard, .. } => {
            reads(call, f);
            if *guard == crate::ir::Guard::Exec {
                f(Var::Exec);
            }
        }
        Stmt::Message { .. } => f(Var::M0),
        Stmt::Barrier { .. } | Stmt::Comment(_) => {}
    }
}

/// The variable that holds the heap index of a resource the program chooses at run time.
fn index_of(resources: &[Resource], res: crate::ir::ResId) -> Option<Var> {
    match resources.get(res.0 as usize)?.origin {
        Origin::HeapIndex { index, .. } => Some(index),
        _ => None,
    }
}

fn expr_indices(e: &Expr, resources: &[Resource], f: &mut impl FnMut(Var)) {
    match e {
        Expr::BufferWord { res, offset } => {
            index_of(resources, *res).into_iter().for_each(&mut *f);
            expr_indices(offset, resources, f);
        }
        Expr::Un(_, a) => expr_indices(a, resources, f),
        Expr::Bin(_, a, b) => {
            expr_indices(a, resources, f);
            expr_indices(b, resources, f);
        }
        Expr::Select(c, a, b) => {
            expr_indices(c, resources, f);
            expr_indices(a, resources, f);
            expr_indices(b, resources, f);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| expr_indices(a, resources, f)),
        _ => {}
    }
}

/// Calls `f` on the heap index variables a statement reads through its resources: a use of a
/// resource with [`Origin::HeapIndex`] reads the variable that holds the index. [`stmt_reads`]
/// cannot know them, it does not see the resource list.
pub fn stmt_index_reads(s: &Stmt, resources: &[Resource], f: &mut impl FnMut(Var)) {
    let desc = |d: &Desc, f: &mut dyn FnMut(Var)| {
        if let Desc::Res(res) = d
            && let Some(index) = index_of(resources, *res)
        {
            f(index);
        }
        if let Desc::Res(res) = d
            && let Origin::SamplerChoices { first, .. } = &resources[res.0 as usize].origin
        {
            (0..4).for_each(|k| f(Var::S(first + k)));
        }
    };
    match s {
        Stmt::Set { value, .. } | Stmt::Output { value, .. } => expr_indices(value, resources, f),
        Stmt::Discard { cond } => expr_indices(cond, resources, f),
        Stmt::Image(op, _) => {
            desc(&op.image, f);
            if let Some(sampler) = &op.sampler {
                desc(sampler, f);
            }
            op.address
                .iter()
                .for_each(|a| expr_indices(a, resources, f));
        }
        Stmt::Buffer(op, _) => {
            desc(&op.buffer, f);
            expr_indices(&op.offset, resources, f);
            if let Some(i) = &op.index {
                expr_indices(i, resources, f);
            }
        }
        Stmt::Effect { call, .. } => expr_indices(call, resources, f),
        Stmt::Export(_) | Stmt::Message { .. } | Stmt::Barrier { .. } | Stmt::Comment(_) => {}
    }
}

fn node_reads(nodes: &[Node], f: &mut impl FnMut(Var)) {
    for n in nodes {
        match n {
            Node::Block(stmts) => stmts.iter().for_each(|s| stmt_reads(s, f)),
            Node::If { cond, then, els } => {
                reads(cond, f);
                node_reads(then, f);
                node_reads(els, f);
            }
            Node::Loop(body) | Node::Once(body) => node_reads(body, f),
            Node::Break | Node::Continue | Node::Return => {}
        }
    }
}

/// Removes, inside one statement list, an assignment whose variable is assigned again before
/// anything reads it. Returns whether it removed one.
fn dead_stores(stmts: &mut Vec<Stmt>, resources: &[Resource]) -> bool {
    // Walking backwards: the variables that are overwritten later with no read in between.
    let mut overwritten: BTreeSet<Var> = BTreeSet::new();
    let mut keep = vec![true; stmts.len()];
    for (i, s) in stmts.iter().enumerate().rev() {
        if let Stmt::Set {
            dst,
            guard: crate::ir::Guard::Always,
            ..
        } = s
        {
            if overwritten.contains(dst) {
                keep[i] = false;
                continue;
            }
            overwritten.insert(*dst);
        }
        stmt_reads(s, &mut |v| {
            overwritten.remove(&v);
        });
        stmt_index_reads(s, resources, &mut |v| {
            overwritten.remove(&v);
        });
        // A write under the lane test keeps the old value in an inactive lane: it is a read too.
        if let Stmt::Set {
            dst,
            guard: crate::ir::Guard::Exec,
            ..
        } = s
        {
            overwritten.remove(dst);
        }
    }
    let mut it = keep.iter();
    stmts.retain(|_| *it.next().expect("one flag per statement"));
    keep.contains(&false)
}

/// Removes assignments to variables nothing reads and assignments that are overwritten before
/// a read in the same block, until none is left. Assignments have no side effect.
pub fn dce(func: &mut Function) {
    loop {
        let mut live = BTreeSet::new();
        node_reads(&func.body, &mut |v| {
            live.insert(v);
        });
        let resources = std::mem::take(&mut func.resources);
        func.for_each_stmt(&mut |s| {
            stmt_index_reads(s, &resources, &mut |v| {
                live.insert(v);
            })
        });
        let mut removed = false;
        func.for_each_block_mut(&mut |stmts| {
            removed |= dead_stores(stmts, &resources);
            let before = stmts.len();
            stmts.retain(|s| !matches!(s, Stmt::Set { dst, .. } if !live.contains(dst)));
            removed |= stmts.len() != before;
        });
        func.resources = resources;
        if !removed {
            return;
        }
    }
}
