//! From a reducible graph to a tree of blocks, loops and jumps with names.
//!
//! The tree is built along the dominator tree, after N. Ramsey, "Beyond Relooper" (2022). A
//! node that one forward edge leads to is written where that edge leaves. A node that several
//! forward edges lead to is written after a block around the text of its dominator, and the
//! edges become jumps to the end of that block. A loop header is written as a loop, a back edge
//! as a jump to its next round. Two things differ from the paper, because HLSL has `break` and
//! `continue` for the innermost loop only and no blocks with a name:
//!
//! * A node that leaves a loop is written after the loop when several edges lead to it, so a
//!   jump to it is a `break` (see `Places`). The paper would write it inside.
//! * [`simplify`] then takes away every block whose jumps all stand where the text goes on to
//!   the same place anyway, which is the case for the code of a plain `if` and `else`. What it
//!   cannot take away becomes a block that runs once (see `lower`).

use super::graph::{Exit, Graph, Loops, Order, Places, not};
use crate::ir::Expr;

/// Where a jump goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Target {
    /// The next round of the loop with this header.
    Round(usize),
    /// The text of this node, which stands right after a block or a loop.
    After(usize),
}

#[derive(Clone, Debug)]
pub(super) enum Tree {
    /// The statements of a node.
    Code(usize),
    If(Expr, Vec<Tree>, Vec<Tree>),
    /// Text that is left by a jump to `After` of this node. The text of the node follows.
    Block(usize, Vec<Tree>),
    /// The body runs again and again. With `after`, the text of that node follows the loop and
    /// a jump to it leaves the loop.
    Loop {
        head: usize,
        after: Option<usize>,
        body: Vec<Tree>,
    },
    /// Several joins in a row without nesting: `first`, then the text of each join in `parts`,
    /// then the text of `last` follows. A jump to a join of the row leaves the text it stands
    /// in and skips the joins before its target. The text of a join can only jump forward.
    Row {
        first: Vec<Tree>,
        parts: Vec<(usize, Vec<Tree>)>,
        last: usize,
    },
    Jump(Target),
    Return,
}

/// The graph nests deeper than the text may.
pub(super) struct TooDeep;

/// The deepest nesting the builder follows. The text is checked again after `simplify`.
const MAX_NESTING: usize = 150;

/// From this many joins behind one node on, the builder writes a row at once. Blocks around
/// each other would nest that deep, and joins in such numbers are jumped to from all over.
const ROW_AT: usize = 6;

struct Builder<'a> {
    g: &'a Graph,
    o: &'a Order,
    loops: &'a Loops,
    places: &'a Places,
    depth: usize,
}

impl Builder<'_> {
    fn deeper(&mut self, by: usize) -> Result<(), TooDeep> {
        self.depth += by;
        if self.depth > MAX_NESTING {
            Err(TooDeep)
        } else {
            Ok(())
        }
    }

    /// What the edge is in the text: a jump, or nothing when the target is written right there.
    fn edge(&self, from: usize, to: usize) -> Option<Target> {
        if self.o.back(from, to) {
            Some(Target::Round(to))
        } else if self.places.jumped_to[to] {
            Some(Target::After(to))
        } else {
            None
        }
    }

    /// Writes `first` and what follows it on the same level, to the end of the level.
    fn run(&mut self, first: Option<usize>, out: &mut Vec<Tree>) -> Result<(), TooDeep> {
        let mut next = first;
        while let Some(x) = next {
            next = self.node(x, out)?;
        }
        Ok(())
    }

    /// Puts `text` into one block for each node of `after` and writes the nodes behind their
    /// blocks. The last of them is returned instead: the caller writes it on its own level, so
    /// a long row of joins does not nest.
    fn wrapped(
        &mut self,
        mut text: Vec<Tree>,
        after: &[usize],
        out: &mut Vec<Tree>,
    ) -> Result<Option<usize>, TooDeep> {
        let Some((&last, inner)) = after.split_last() else {
            out.append(&mut text);
            return Ok(None);
        };
        if inner.len() >= ROW_AT {
            self.deeper(2)?;
            let mut parts = Vec::with_capacity(inner.len());
            for &y in inner {
                let mut seq = Vec::new();
                self.run(Some(y), &mut seq)?;
                parts.push((y, seq));
            }
            self.depth -= 2;
            out.push(Tree::Row {
                first: text,
                parts,
                last,
            });
            return Ok(Some(last));
        }
        self.deeper(inner.len())?;
        for &y in inner {
            let mut seq = vec![Tree::Block(y, text)];
            self.run(Some(y), &mut seq)?;
            text = seq;
        }
        self.depth -= inner.len();
        out.push(Tree::Block(last, text));
        Ok(Some(last))
    }

    /// Writes node `x` with everything that is placed with it. Returns the node that goes on
    /// at the same level.
    fn node(&mut self, x: usize, out: &mut Vec<Tree>) -> Result<Option<usize>, TooDeep> {
        if !self.loops.header[x] {
            return self.within(x, out);
        }
        let mut body = Vec::new();
        self.deeper(1)?;
        let next = self.within(x, &mut body)?;
        self.run(next, &mut body)?;
        self.depth -= 1;
        let places = self.places;
        let text = vec![Tree::Loop {
            head: x,
            after: None,
            body,
        }];
        self.wrapped(text, &places.followers[x], out)
    }

    /// Node `x` and, around what it leads to, the blocks of the joins it dominates. Its own
    /// statements stand before the blocks: they jump nowhere, and outside they stay straight
    /// line code for the passes that follow.
    fn within(&mut self, x: usize, out: &mut Vec<Tree>) -> Result<Option<usize>, TooDeep> {
        let places = self.places;
        out.push(Tree::Code(x));
        if places.merges[x].is_empty() {
            return self.exit(x, out);
        }
        let mut text = Vec::new();
        self.deeper(1)?;
        let next = self.exit(x, &mut text)?;
        self.run(next, &mut text)?;
        self.depth -= 1;
        self.wrapped(text, &places.merges[x], out)
    }

    /// The exit of `x`.
    fn exit(&mut self, x: usize, out: &mut Vec<Tree>) -> Result<Option<usize>, TooDeep> {
        let g = self.g;
        Ok(match &g.exit[x] {
            Exit::End => {
                out.push(Tree::Return);
                None
            }
            Exit::Jump(t) => match self.edge(x, *t) {
                Some(jump) => {
                    out.push(Tree::Jump(jump));
                    None
                }
                None => Some(*t),
            },
            Exit::Branch { cond, taken, fall } => {
                let (taken, fall) = (*taken, *fall);
                match (self.edge(x, taken), self.edge(x, fall)) {
                    (Some(a), Some(b)) => {
                        out.push(Tree::If(
                            cond.clone(),
                            vec![Tree::Jump(a)],
                            vec![Tree::Jump(b)],
                        ));
                        None
                    }
                    (Some(a), None) => {
                        out.push(Tree::If(cond.clone(), vec![Tree::Jump(a)], Vec::new()));
                        Some(fall)
                    }
                    (None, Some(b)) => {
                        out.push(Tree::If(not(cond.clone()), vec![Tree::Jump(b)], Vec::new()));
                        Some(taken)
                    }
                    (None, None) => {
                        // Both sides are written here. Each ends in a jump or a return, so one
                        // can be the body of the `if` and the other can follow it. The smaller
                        // one goes inside: a row of early exits then stays flat.
                        let (inside, rest, cond) = if self.o.size[taken] <= self.o.size[fall] {
                            (taken, fall, cond.clone())
                        } else {
                            (fall, taken, not(cond.clone()))
                        };
                        let mut arm = Vec::new();
                        self.deeper(1)?;
                        self.run(Some(inside), &mut arm)?;
                        self.depth -= 1;
                        out.push(Tree::If(cond, arm, Vec::new()));
                        Some(rest)
                    }
                }
            }
        })
    }
}

/// The tree of a reducible graph. Every path through it ends in a jump or a return.
pub(super) fn build(
    g: &Graph,
    o: &Order,
    loops: &Loops,
    places: &Places,
) -> Result<Vec<Tree>, TooDeep> {
    let mut b = Builder {
        g,
        o,
        loops,
        places,
        depth: 0,
    };
    let mut out = Vec::new();
    b.run(Some(0), &mut out)?;
    Ok(out)
}

/// Whether the text has a jump to `After(y)`.
fn mentions(t: &Tree, y: usize) -> bool {
    match t {
        Tree::Jump(target) => *target == Target::After(y),
        Tree::If(_, a, b) => a.iter().chain(b).any(|t| mentions(t, y)),
        Tree::Block(_, body) | Tree::Loop { body, .. } => body.iter().any(|t| mentions(t, y)),
        Tree::Row { first, parts, .. } => first
            .iter()
            .chain(parts.iter().flat_map(|(_, text)| text))
            .any(|t| mentions(t, y)),
        Tree::Code(_) | Tree::Return => false,
    }
}

/// Whether control never comes out at the end of the sequence.
pub(super) fn diverges(seq: &[Tree]) -> bool {
    match seq.last() {
        Some(Tree::Jump(_) | Tree::Return | Tree::Loop { after: None, .. }) => true,
        Some(Tree::If(_, a, b)) => diverges(a) && diverges(b),
        _ => false,
    }
}

/// How many times the text after an `if` may move into its `else` to free one block. Each
/// move nests that text one level deeper.
const MOVES: u32 = 8;

/// Whether every jump to `After(y)` in `seq` can be made the last thing on its path, so that
/// leaving it out goes to the same place. `followed` says that text follows the sequence.
fn can_end(seq: &[Tree], y: usize, followed: bool, moves: &mut u32) -> bool {
    for (i, item) in seq.iter().enumerate() {
        if !mentions(item, y) {
            continue;
        }
        let more = followed || i + 1 < seq.len();
        match item {
            // What follows a jump is never reached.
            Tree::Jump(_) => return true,
            Tree::If(_, a, b) => match (more, diverges(a), diverges(b)) {
                (false, ..) | (true, true, true) => {
                    return can_end(a, y, false, moves) && can_end(b, y, false, moves);
                }
                (true, false, false) => return false,
                (true, a_leaves, _) => {
                    // The text that follows moves into the side that comes out at its end. It
                    // is judged as the walk goes on.
                    let (leaves, through) = if a_leaves { (a, b) } else { (b, a) };
                    if *moves == 0 {
                        return false;
                    }
                    *moves -= 1;
                    if !can_end(leaves, y, false, moves) || !can_end(through, y, true, moves) {
                        return false;
                    }
                }
            },
            // A loop that nothing follows yet: the node becomes what follows it.
            Tree::Loop { after: None, .. } if !more => return true,
            _ => return false,
        }
    }
    true
}

/// Does what [`can_end`] found possible, and takes the jumps out.
fn end(seq: &mut Vec<Tree>, y: usize) {
    let Some(i) = seq.iter().position(|t| mentions(t, y)) else {
        return;
    };
    let mut rest = seq.split_off(i + 1);
    let gone = match &mut seq[i] {
        Tree::Jump(_) => true,
        Tree::If(cond, a, b) => {
            match (diverges(a), diverges(b)) {
                (true, true) => {}
                (true, false) => b.append(&mut rest),
                (false, true) => a.append(&mut rest),
                (false, false) => debug_assert!(rest.is_empty(), "can_end allowed this"),
            }
            end(a, y);
            end(b, y);
            if a.is_empty() && !b.is_empty() {
                std::mem::swap(a, b);
                *cond = not(std::mem::replace(cond, Expr::Bool(false)));
            }
            a.is_empty()
        }
        Tree::Loop { after, .. } => {
            *after = Some(y);
            false
        }
        Tree::Block(..) | Tree::Row { .. } | Tree::Code(_) | Tree::Return => {
            unreachable!("can_end allowed this")
        }
    };
    if gone {
        seq.pop();
    }
}

/// Takes away the blocks that are not needed, innermost first.
pub(super) fn simplify(seq: Vec<Tree>) -> Vec<Tree> {
    let mut out = Vec::with_capacity(seq.len());
    for item in seq {
        match item {
            Tree::If(c, a, b) => out.push(Tree::If(c, simplify(a), simplify(b))),
            Tree::Loop { head, after, body } => out.push(Tree::Loop {
                head,
                after,
                body: simplify(body),
            }),
            Tree::Block(y, body) => {
                let mut body = simplify(body);
                if can_end(&body, y, false, &mut { MOVES }) {
                    end(&mut body, y);
                    out.append(&mut body);
                } else {
                    out.push(Tree::Block(y, body));
                }
            }
            Tree::Row { first, parts, last } => out.push(Tree::Row {
                first: simplify(first),
                parts: parts
                    .into_iter()
                    .map(|(y, text)| (y, simplify(text)))
                    .collect(),
                last,
            }),
            other => out.push(other),
        }
    }
    out
}

/// Turns the blocks that are left right around each other into rows. Two blocks around each
/// other need a boolean for the jump across both anyway, and a row does not nest.
pub(super) fn rows(seq: Vec<Tree>) -> Vec<Tree> {
    seq.into_iter()
        .map(|item| match item {
            Tree::If(c, a, b) => Tree::If(c, rows(a), rows(b)),
            Tree::Loop { head, after, body } => Tree::Loop {
                head,
                after,
                body: rows(body),
            },
            Tree::Row { first, parts, last } => Tree::Row {
                first: rows(first),
                parts: parts.into_iter().map(|(y, text)| (y, rows(text))).collect(),
                last,
            },
            Tree::Block(last, mut body) => {
                // From the outermost block inwards: what follows the inner block is the text
                // of the join the inner block is named after.
                let mut parts = Vec::new();
                while matches!(body.first(), Some(Tree::Block(..))) {
                    let text = body.split_off(1);
                    let Some(Tree::Block(y, inner)) = body.pop() else {
                        unreachable!("the first item is a block");
                    };
                    parts.push((y, rows(text)));
                    body = inner;
                }
                if parts.is_empty() {
                    Tree::Block(last, rows(body))
                } else {
                    parts.reverse();
                    Tree::Row {
                        first: rows(body),
                        parts,
                        last,
                    }
                }
            }
            other => other,
        })
        .collect()
}

/// The nodes that are still jumped to as the text behind a block or as a join of a row.
pub(super) fn joins(seq: &[Tree], out: &mut Vec<usize>) {
    for t in seq {
        match t {
            Tree::If(_, a, b) => {
                joins(a, out);
                joins(b, out);
            }
            Tree::Loop { body, .. } => joins(body, out),
            Tree::Block(y, body) => {
                out.push(*y);
                joins(body, out);
            }
            Tree::Row { first, parts, last } => {
                out.push(*last);
                joins(first, out);
                for (y, text) in parts {
                    out.push(*y);
                    joins(text, out);
                }
            }
            Tree::Code(_) | Tree::Jump(_) | Tree::Return => {}
        }
    }
}

/// The deepest nesting of the text.
pub(super) fn depth(seq: &[Tree]) -> usize {
    seq.iter()
        .map(|t| match t {
            Tree::If(_, a, b) => 1 + depth(a).max(depth(b)),
            Tree::Block(_, body) | Tree::Loop { body, .. } => 1 + depth(body),
            // The first text is in a block that runs once, a join in an `if` and such a block.
            Tree::Row { first, parts, .. } => parts
                .iter()
                .map(|(_, text)| 2 + depth(text))
                .max()
                .unwrap_or(0)
                .max(1 + depth(first)),
            Tree::Code(_) | Tree::Jump(_) | Tree::Return => 0,
        })
        .max()
        .unwrap_or(0)
}
