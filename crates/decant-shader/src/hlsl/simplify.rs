//! Passes over the placed function that make the printed text smaller.
//!
//! * Constant folding of integer and boolean expressions (`0u >> 4`, `x + 0u`, `true && c`).
//! * A constant EXEC: when every write to EXEC in the program stores `true`, the variable and all
//!   `if (exec)` guards disappear. Most programs without divergent control flow end up here.
//!
//! The passes keep the meaning of every value, only the text changes.

use crate::ir::{Bin, Expr, Function, Guard, Node, Stmt, Un, Var};

/// Runs every pass. Returns true when EXEC was found to be constant true.
pub fn run(func: &mut Function) -> bool {
    visit_nodes(&mut func.body, &mut fold);
    let exec = exec_is_constant(func);
    if exec {
        remove_exec(func);
        visit_nodes(&mut func.body, &mut fold);
    }
    exec
}

fn fold_u32(op: Bin, a: u32, b: u32) -> Option<Expr> {
    let bool_ = |v: bool| Some(Expr::Bool(v));
    let word = |v: u32| Some(Expr::U32(v));
    match op {
        Bin::Add => word(a.wrapping_add(b)),
        Bin::Sub => word(a.wrapping_sub(b)),
        Bin::Mul => word(a.wrapping_mul(b)),
        Bin::And => word(a & b),
        Bin::Or => word(a | b),
        Bin::Xor => word(a ^ b),
        Bin::Shl => word(a << (b & 31)),
        Bin::LShr => word(a >> (b & 31)),
        Bin::AShr => word(((a as i32) >> (b & 31)) as u32),
        Bin::Eq => bool_(a == b),
        Bin::Ne => bool_(a != b),
        Bin::ULt => bool_(a < b),
        Bin::ULe => bool_(a <= b),
        Bin::SLt => bool_((a as i32) < (b as i32)),
        Bin::SLe => bool_((a as i32) <= (b as i32)),
        _ => None,
    }
}

/// Takes the expression out of `e`, leaving a constant behind.
fn take(e: &mut Expr) -> Expr {
    std::mem::replace(e, Expr::Bool(false))
}

/// Folds `e` bottom up, in place: a rebuilt tree cost an allocation per node, and a large
/// program has millions of them.
pub fn fold(e: &mut Expr) {
    match e {
        Expr::Un(op, a) => {
            fold(a);
            let folded = match (*op, &mut **a) {
                (Un::Not, Expr::U32(v)) => Expr::U32(!*v),
                (Un::Neg, Expr::U32(v)) => Expr::U32(v.wrapping_neg()),
                (Un::LNot, Expr::Bool(v)) => Expr::Bool(!*v),
                (Un::BoolToU32, Expr::Bool(v)) => Expr::U32(u32::from(*v)),
                (Un::U32ToBool, Expr::U32(v)) => Expr::Bool(*v != 0),
                (Un::U32ToU64, Expr::U32(v)) => Expr::U64(u64::from(*v)),
                (Un::U64ToU32, Expr::U64(v)) => Expr::U32(*v as u32),
                (Un::FNeg, Expr::U32(v)) => Expr::U32(*v ^ 0x8000_0000),
                (Un::FAbs, Expr::U32(v)) => Expr::U32(*v & 0x7fff_ffff),
                (Un::LNot, Expr::Un(Un::LNot, inner)) => take(inner),
                _ => return,
            };
            *e = folded;
        }
        Expr::Bin(op, a, b) => {
            fold(a);
            fold(b);
            let op = *op;
            if let (Expr::U32(x), Expr::U32(y)) = (&**a, &**b)
                && let Some(v) = fold_u32(op, *x, *y)
            {
                *e = v;
                return;
            }
            // Which operand the expression reduces to, or the constant it is.
            enum To {
                A,
                B,
                Const(Expr),
            }
            let to = match (op, &**a, &**b) {
                (Bin::Add | Bin::Or | Bin::Xor, _, Expr::U32(0))
                | (Bin::Sub | Bin::Shl | Bin::LShr, _, Expr::U32(0))
                | (Bin::Mul, _, Expr::U32(1))
                | (Bin::And, _, Expr::U32(u32::MAX)) => To::A,
                (Bin::Add | Bin::Or | Bin::Xor, Expr::U32(0), _)
                | (Bin::Mul, Expr::U32(1), _)
                | (Bin::And, Expr::U32(u32::MAX), _) => To::B,
                (Bin::Mul | Bin::And, _, Expr::U32(0)) | (Bin::Mul | Bin::And, Expr::U32(0), _) => {
                    To::Const(Expr::U32(0))
                }
                (Bin::LAnd, Expr::Bool(true), _) => To::B,
                (Bin::LAnd, _, Expr::Bool(true)) => To::A,
                (Bin::LOr, Expr::Bool(false), _) => To::B,
                (Bin::LOr, _, Expr::Bool(false)) => To::A,
                (Bin::LAnd, Expr::Bool(false), _) | (Bin::LAnd, _, Expr::Bool(false)) => {
                    To::Const(Expr::Bool(false))
                }
                (Bin::LOr, Expr::Bool(true), _) | (Bin::LOr, _, Expr::Bool(true)) => {
                    To::Const(Expr::Bool(true))
                }
                _ => return,
            };
            *e = match to {
                To::A => take(a),
                To::B => take(b),
                To::Const(c) => c,
            };
        }
        Expr::Select(c, a, b) => {
            fold(c);
            fold(a);
            fold(b);
            match **c {
                Expr::Bool(true) => *e = take(a),
                Expr::Bool(false) => *e = take(b),
                _ => {}
            }
        }
        Expr::Call(_, args) => args.iter_mut().for_each(fold),
        Expr::BufferWord { offset, .. } => fold(offset),
        // A code address is its offset: arithmetic on it folds like a constant.
        Expr::CodeAddress(k) => *e = Expr::U32(*k),
        Expr::CodeWord(at) => fold(at),
        _ => {}
    }
}

/// Replaces every read of the program's own code at a constant address by the word that is
/// there. `code` is the code of the program with its signature. A constant address outside
/// the code reads zero. Returns whether a read with a run time address is left: the printer
/// then has to keep the code words as an array.
pub fn fold_code(func: &mut Function, code: &[u32]) -> bool {
    // In place: most expressions read no code, and a rebuilt tree costs an allocation per node.
    fn go(e: &mut Expr, code: &[u32], left: &mut bool) {
        match e {
            Expr::CodeWord(at) => {
                go(at, code, left);
                if let Expr::U32(k) = **at {
                    *e = Expr::U32(code.get((k >> 2) as usize).copied().unwrap_or(0));
                } else {
                    *left = true;
                }
            }
            Expr::Un(_, a) => go(a, code, left),
            Expr::Bin(_, a, b) => {
                go(a, code, left);
                go(b, code, left);
            }
            Expr::Select(c, a, b) => {
                go(c, code, left);
                go(a, code, left);
                go(b, code, left);
            }
            Expr::Call(_, args) => args.iter_mut().for_each(|a| go(a, code, left)),
            Expr::BufferWord { offset, .. } => go(offset, code, left),
            _ => {}
        }
    }
    let mut left = false;
    visit_nodes(&mut func.body, &mut |e| go(e, code, &mut left));
    left
}

/// Calls `f` on every top level expression of the statement, to change it in place.
fn visit_stmt(s: &mut Stmt, f: &mut impl FnMut(&mut Expr)) {
    match s {
        Stmt::Set { value, .. } | Stmt::Output { value, .. } => f(value),
        Stmt::Discard { cond } => f(cond),
        Stmt::Image(op, _) => op.address.iter_mut().for_each(f),
        Stmt::Buffer(op, _) => {
            f(&mut op.offset);
            if let Some(i) = &mut op.index {
                f(i);
            }
        }
        Stmt::Effect { call, .. } => f(call),
        Stmt::Export(_) | Stmt::Message { .. } | Stmt::Barrier { .. } | Stmt::Comment(_) => {}
    }
}

/// [`visit_stmt`] for every statement and condition of the nodes.
pub(super) fn visit_nodes(ns: &mut [Node], f: &mut impl FnMut(&mut Expr)) {
    for n in ns {
        match n {
            Node::Block(stmts) => stmts.iter_mut().for_each(|s| visit_stmt(s, f)),
            Node::If { cond, then, els } => {
                f(cond);
                visit_nodes(then, f);
                visit_nodes(els, f);
            }
            Node::Loop(body) | Node::Once(body) => visit_nodes(body, f),
            Node::Break | Node::Continue | Node::Return => {}
        }
    }
}

/// True when EXEC is written at least once and only ever with `true`.
fn exec_is_constant(func: &Function) -> bool {
    let (mut writes, mut other) = (0usize, 0usize);
    func.for_each_stmt(&mut |s| {
        if let Stmt::Set {
            dst: Var::Exec,
            value,
            ..
        } = s
        {
            writes += 1;
            if *value != Expr::Bool(true) {
                other += 1;
            }
        }
    });
    writes > 0 && other == 0
}

fn replace_exec(e: &mut Expr) {
    match e {
        Expr::Var(Var::Exec) => *e = Expr::Bool(true),
        Expr::Un(_, a) | Expr::CodeWord(a) | Expr::BufferWord { offset: a, .. } => replace_exec(a),
        Expr::Bin(_, a, b) => {
            replace_exec(a);
            replace_exec(b);
        }
        Expr::Select(c, a, b) => {
            replace_exec(c);
            replace_exec(a);
            replace_exec(b);
        }
        Expr::Call(_, args) => args.iter_mut().for_each(replace_exec),
        _ => {}
    }
}

fn strip_exec(stmts: &mut Vec<Stmt>) {
    stmts.retain(|s| !matches!(s, Stmt::Set { dst: Var::Exec, .. }));
    for s in stmts {
        match s {
            Stmt::Set { guard, .. }
            | Stmt::Image(_, guard)
            | Stmt::Buffer(_, guard)
            | Stmt::Effect { guard, .. } => {
                *guard = Guard::Always;
            }
            _ => {}
        }
    }
}

fn remove_exec(func: &mut Function) {
    visit_nodes(&mut func.body, &mut replace_exec);
    func.for_each_block_mut(&mut strip_exec);
}
