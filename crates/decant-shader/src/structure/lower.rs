//! From the tree with names to the statements of the IR, which has what HLSL has: `if`, an
//! endless loop, a block that runs once (`do { } while (false)`), `break` and `continue` for
//! the innermost of them, `return`.
//!
//! A loop of the tree is a loop. A block that `simplify` left is a `Node::Once`, or plain
//! text when its jumps can all stand at its end. A jump whose target is the innermost loop or
//! block is a `break` or a `continue`. A jump further out sets a boolean of its target and
//! leaves the innermost one, and behind every loop and block it has to cross stands
//! `if (flag) break;`, until its target is the innermost one. Lanes that took different ways
//! meet again behind each of these constructs.

use std::collections::{BTreeMap, BTreeSet};

use super::tree::{Target, Tree, diverges};
use crate::ir::{Expr, Guard, Node, Stmt, Var};

/// The booleans of this pass are `Var::Flag` from this number DOWN. The lifter counts its
/// temporaries up from zero, and `Var::Tmp(0xFFFF)` is the sink of the null register.
pub const FIRST_FLAG: u16 = 0xFFFE;

/// Whether the tree can be lowered: every jump stands inside its target, the statements of
/// every node stand in it once, and no path runs out of the text or out of a loop body.
pub(super) fn sound(seq: &[Tree], nodes: usize) -> bool {
    fn walk(seq: &[Tree], names: &mut Vec<Target>, seen: &mut [u8]) -> bool {
        seq.iter().all(|t| match t {
            Tree::Code(n) => {
                seen[*n] = seen[*n].saturating_add(1);
                true
            }
            Tree::If(_, a, b) => walk(a, names, seen) && walk(b, names, seen),
            Tree::Jump(target) => names.contains(target),
            Tree::Return => true,
            Tree::Block(y, body) => {
                names.push(Target::After(*y));
                let ok = walk(body, names, seen);
                names.pop();
                ok
            }
            Tree::Loop { head, after, body } => {
                let before = names.len();
                names.push(Target::Round(*head));
                names.extend(after.map(Target::After));
                let ok = diverges(body) && walk(body, names, seen);
                names.truncate(before);
                ok
            }
            Tree::Row { first, parts, last } => {
                // A text of the row may jump to the joins behind it.
                let before = names.len();
                names.push(Target::After(*last));
                names.extend(parts.iter().rev().map(|(y, _)| Target::After(*y)));
                let mut ok = walk(first, names, seen);
                for (_, text) in parts {
                    names.pop();
                    ok = ok && walk(text, names, seen);
                }
                names.truncate(before);
                ok
            }
        })
    }
    let mut seen = vec![0u8; nodes];
    diverges(seq) && walk(seq, &mut Vec::new(), &mut seen) && seen.iter().all(|&n| n == 1)
}

/// A loop of the output and the two names a jump can mean by it.
struct Frame {
    leave: Option<Target>,
    again: Option<Target>,
    /// The joins of a row behind this text: a jump to one sets its boolean and leaves.
    joins: Vec<Target>,
    /// Targets further out that a jump inside this loop goes to.
    crossing: BTreeSet<Target>,
}

struct Lower<'a> {
    code: &'a mut [Vec<Stmt>],
    frames: Vec<Frame>,
    flags: BTreeMap<Target, Var>,
}

fn set(flag: Var, value: bool) -> Stmt {
    Stmt::Set {
        dst: flag,
        value: Expr::Bool(value),
        guard: Guard::Always,
    }
}

fn push(out: &mut Vec<Node>, mut stmts: Vec<Stmt>) {
    match out.last_mut() {
        Some(Node::Block(last)) => last.append(&mut stmts),
        _ if stmts.is_empty() => {}
        _ => out.push(Node::Block(stmts)),
    }
}

/// The first node whose statements stand in the text.
fn first_node(seq: &[Tree]) -> Option<usize> {
    seq.iter().find_map(|t| match t {
        Tree::Code(n) => Some(*n),
        Tree::If(_, a, b) => first_node(a).or_else(|| first_node(b)),
        Tree::Block(_, body) | Tree::Loop { body, .. } => first_node(body),
        Tree::Row { first, parts, .. } => {
            first_node(first).or_else(|| parts.iter().find_map(|(_, text)| first_node(text)))
        }
        Tree::Jump(_) | Tree::Return => None,
    })
}

/// Whether control never comes out at the end.
fn leaves(nodes: &[Node]) -> bool {
    match nodes.last() {
        Some(Node::Break | Node::Continue | Node::Return) => true,
        Some(Node::If { then, els, .. }) => leaves(then) && leaves(els),
        _ => false,
    }
}

/// How many times the text after an `if` may move into its `else` to free one loop that runs
/// once. Each move nests that text one level deeper.
const MOVES: u32 = 8;

/// Whether the text has a `break` of its own, not one of a loop inside it.
fn breaks(nodes: &[Node]) -> bool {
    nodes.iter().any(|n| match n {
        Node::Break => true,
        Node::If { then, els, .. } => breaks(then) || breaks(els),
        _ => false,
    })
}

/// Whether every `break` of the body of a block that runs once can be made the last thing on
/// its path, so that leaving it out goes to the same place. The same question `tree` asks for
/// a block before the jumps have become statements. `followed` says that text follows `nodes`.
fn can_unwrap(nodes: &[Node], followed: bool, moves: &mut u32) -> bool {
    for (i, n) in nodes.iter().enumerate() {
        let more = followed || i + 1 < nodes.len();
        match n {
            // What follows a `break` is never reached.
            Node::Break => return true,
            Node::If { then, els, .. } if breaks(then) || breaks(els) => {
                match (more, leaves(then), leaves(els)) {
                    (false, ..) | (true, true, true) => {
                        return can_unwrap(then, false, moves) && can_unwrap(els, false, moves);
                    }
                    (true, false, false) => return false,
                    (true, then_leaves, _) => {
                        // The text that follows moves into the side that comes out at its end.
                        let (leaving, through) = if then_leaves {
                            (then, els)
                        } else {
                            (els, then)
                        };
                        if *moves == 0 {
                            return false;
                        }
                        *moves -= 1;
                        if !can_unwrap(leaving, false, moves) || !can_unwrap(through, true, moves) {
                            return false;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    true
}

/// Does what [`can_unwrap`] found possible, and takes the `break`s out.
fn unwrap(nodes: &mut Vec<Node>) {
    let Some(i) = nodes.iter().position(|n| match n {
        Node::Break => true,
        Node::If { then, els, .. } => breaks(then) || breaks(els),
        _ => false,
    }) else {
        return;
    };
    let mut rest = nodes.split_off(i + 1);
    let gone = match &mut nodes[i] {
        Node::If { cond, then, els } => {
            match (leaves(then), leaves(els)) {
                (true, true) => {}
                (true, false) => els.append(&mut rest),
                (false, true) => then.append(&mut rest),
                (false, false) => debug_assert!(rest.is_empty(), "can_unwrap allowed this"),
            }
            unwrap(then);
            unwrap(els);
            if then.is_empty() && !els.is_empty() {
                std::mem::swap(then, els);
                *cond = super::graph::not(std::mem::replace(cond, Expr::Bool(false)));
            }
            then.is_empty()
        }
        _ => true,
    };
    if gone {
        nodes.pop();
    }
}

/// Takes away a jump that is the last thing on its path through a body and goes where the
/// end of the body goes anyway: a `continue` at the end of a loop, a `break` at the end of a
/// block that runs once.
fn drop_last(nodes: &mut Vec<Node>, jump: &Node) {
    match nodes.last_mut() {
        Some(last) if last == jump => {
            nodes.pop();
        }
        Some(Node::If { cond, then, els }) => {
            drop_last(then, jump);
            drop_last(els, jump);
            if then.is_empty() && !els.is_empty() {
                std::mem::swap(then, els);
                *cond = super::graph::not(std::mem::replace(cond, Expr::Bool(false)));
            }
            if then.is_empty() {
                nodes.pop();
            }
        }
        _ => {}
    }
}

impl Lower<'_> {
    fn seq(&mut self, items: Vec<Tree>) -> Vec<Node> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Tree::Code(n) => push(&mut out, std::mem::take(&mut self.code[n])),
                Tree::If(mut cond, mut a, mut b) => {
                    // The side that comes first in the native code comes first in the text.
                    if let (Some(x), Some(y)) = (first_node(&a), first_node(&b))
                        && y < x
                    {
                        std::mem::swap(&mut a, &mut b);
                        cond = super::graph::not(cond);
                    }
                    let then = self.seq(a);
                    let els = self.seq(b);
                    out.push(Node::If { cond, then, els });
                }
                Tree::Return => out.push(Node::Return),
                Tree::Jump(target) => self.jump(target, &mut out),
                Tree::Block(y, body) => {
                    self.looped(Some(Target::After(y)), None, Vec::new(), body, &mut out)
                }
                Tree::Loop { head, after, body } => self.looped(
                    after.map(Target::After),
                    Some(Target::Round(head)),
                    Vec::new(),
                    body,
                    &mut out,
                ),
                Tree::Row { first, parts, last } => {
                    // Each text is a block that runs once. Leaving it without a boolean set
                    // skips every join and comes out at the text of `last`.
                    let leave = Some(Target::After(last));
                    let mut joins: Vec<Target> =
                        parts.iter().map(|(y, _)| Target::After(*y)).collect();
                    self.looped(leave, None, joins.clone(), first, &mut out);
                    for (y, text) in parts {
                        joins.remove(0);
                        let flag = self.flag(Target::After(y));
                        let mut then = vec![Node::Block(vec![set(flag, false)])];
                        self.looped(leave, None, joins.clone(), text, &mut then);
                        out.push(Node::If {
                            cond: Expr::Var(flag),
                            then,
                            els: Vec::new(),
                        });
                    }
                }
            }
        }
        out
    }

    /// The boolean of a target.
    fn flag(&mut self, target: Target) -> Var {
        let next = FIRST_FLAG - self.flags.len() as u16;
        *self.flags.entry(target).or_insert(Var::Flag(next))
    }

    fn jump(&mut self, target: Target, out: &mut Vec<Node>) {
        let at = self
            .frames
            .iter()
            .rposition(|f| {
                f.leave == Some(target) || f.again == Some(target) || f.joins.contains(&target)
            })
            .expect("`sound` saw every jump inside its target");
        let join = self.frames[at].joins.contains(&target);
        if at + 1 == self.frames.len() && !join {
            out.push(if self.frames[at].leave == Some(target) {
                Node::Break
            } else {
                Node::Continue
            });
            return;
        }
        let flag = self.flag(target);
        for f in &mut self.frames[at + 1..] {
            f.crossing.insert(target);
        }
        push(out, vec![set(flag, true)]);
        out.push(Node::Break);
    }

    fn looped(
        &mut self,
        leave: Option<Target>,
        again: Option<Target>,
        joins: Vec<Target>,
        body: Vec<Tree>,
        out: &mut Vec<Node>,
    ) {
        self.frames.push(Frame {
            leave,
            again,
            joins,
            crossing: BTreeSet::new(),
        });
        let mut inner = self.seq(body);
        let frame = self.frames.pop().expect("pushed above");
        if again.is_some() {
            drop_last(&mut inner, &Node::Continue);
            out.push(Node::Loop(inner));
        } else if can_unwrap(&inner, false, &mut { MOVES }) {
            // Every `break` stands, or can be made to stand, where the text ends anyway: the
            // loop that would run once is not needed.
            unwrap(&mut inner);
            for node in inner {
                match node {
                    Node::Block(stmts) => push(out, stmts),
                    other => out.push(other),
                }
            }
        } else {
            drop_last(&mut inner, &Node::Break);
            out.push(Node::Once(inner));
        }
        for target in frame.crossing {
            let flag = self.flags[&target];
            let outer = self
                .frames
                .last()
                .expect("a jump further out has a loop further out");
            let then = if outer.leave == Some(target) {
                vec![Node::Block(vec![set(flag, false)]), Node::Break]
            } else if outer.again == Some(target) {
                vec![Node::Block(vec![set(flag, false)]), Node::Continue]
            } else {
                // Further out, or a join of the row this text is in: its test clears it.
                vec![Node::Break]
            };
            out.push(Node::If {
                cond: Expr::Var(flag),
                then,
                els: Vec::new(),
            });
        }
    }
}

/// Lowers a tree that is [`sound`]. The statements are taken out of `code`.
pub(super) fn lower(tree: Vec<Tree>, code: &mut [Vec<Stmt>]) -> Vec<Node> {
    Lower {
        code,
        frames: Vec::new(),
        flags: BTreeMap::new(),
    }
    .seq(tree)
}
