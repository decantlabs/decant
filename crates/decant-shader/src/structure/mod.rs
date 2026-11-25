//! From the native graph to structured statements: real `if`, `else` and loops with `break`
//! and `continue`, as the HLSL compiler wants them.
//!
//! Why it matters: text with real nesting compiles fast, lanes that took different ways meet
//! again behind every `if` and loop (so derivatives after a branch are defined), and the
//! resource pass sees which code runs before which.
//!
//! The pass is a pure graph pass. It knows nothing about lane masks: a branch is a condition
//! over the state of one lane (`graph::condition`), and what a mask means is decided by
//! `crate::exec` and the lifter before it.
//!
//! Steps:
//!
//! 1. `graph`: order, dominators, loops, and the place of every node in the text.
//! 2. `tree`: a tree of blocks and loops with names and jumps to them, along the dominator
//!    tree. Then the blocks that plain nesting can replace are taken away.
//! 3. `lower`: the names become `break`, `continue` and, for a jump across more than one loop,
//!    a boolean that is tested behind each loop on the way.
//!
//! No code is copied but a short end of the program that many places jump to
//! (`graph::copy_ends`), so the size of the output is linear in the size of the program. A
//! loop with several ways in is given one header first
//! (`graph::reduce`). A graph that would nest deeper than the HLSL compiler reads (256 brackets)
//! is refused with `cfg:depth`. No program of the title comes near: the deepest nests 27 levels.

pub mod check;
mod graph;
mod lower;
mod skip;
mod tree;

use crate::cfg::Cfg;
use crate::ir::{Node, Stmt};
use crate::refusal::{Refusal, RefusalKind};

pub use graph::FIRST_SELECTOR;
pub use lower::FIRST_FLAG;

/// The deepest nesting of `if` and loops the pass writes. The compiler stops at 256 open
/// brackets, and the printer needs some for the function, the lane tests and the expressions.
pub const MAX_DEPTH: usize = 150;

fn refuse(what: &'static str) -> Refusal {
    Refusal::new(RefusalKind::ControlFlow(what), "")
}

/// The tree of the graph, with the blocks that nesting replaces taken away.
fn text(g: &mut graph::Graph) -> Result<Vec<tree::Tree>, Refusal> {
    let mut order = graph::Order::new(g);
    if order.irreducible_edge(g).is_some() {
        graph::reduce(g);
        order = graph::Order::new(g);
    }
    let loops = graph::Loops::new(g, &order);
    let places = graph::Places::new(g, &order, &loops);
    let tree = tree::build(g, &order, &loops, &places).map_err(|_| refuse("depth"))?;
    Ok(tree::rows(tree::simplify(tree)))
}

/// The structured body of a program. `lifted` holds the statements of each block of `cfg`,
/// and every block of `cfg` can be reached from the first, as `cfg::build` makes it.
///
/// Refused with `cfg:depth` when the text would nest deeper than [`MAX_DEPTH`].
pub fn build(cfg: &Cfg, lifted: Vec<Vec<Stmt>>) -> Result<Vec<Node>, Refusal> {
    let mut g = graph::Graph::new(cfg, lifted);
    skip::drop_skips(&mut g);
    let mut tree = text(&mut g)?;
    // A short end of the program that is still jumped to gets a copy per jump, and the text is
    // made again: without that join it nests further.
    let mut joins = Vec::new();
    tree::joins(&tree, &mut joins);
    if graph::copy_ends(&mut g, &joins) {
        tree = text(&mut g)?;
    }
    if tree::depth(&tree) > MAX_DEPTH {
        return Err(refuse("depth"));
    }
    if !lower::sound(&tree, g.len()) {
        // Never seen: it would be a mistake of this pass, and a wrong program is worse than none.
        debug_assert!(false, "the tree of a reducible graph is sound");
        return Err(refuse("unsound"));
    }
    Ok(lower::lower(tree, &mut g.code))
}

/// How the body of a program came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shape {
    /// No branch at all.
    Straight,
    /// Nested `if` and loops.
    Structured,
}

/// The shape of a body, for the census.
pub fn shape(body: &[Node]) -> Shape {
    if body
        .iter()
        .any(|n| matches!(n, Node::If { .. } | Node::Loop(_) | Node::Once(_)))
    {
        Shape::Structured
    } else {
        Shape::Straight
    }
}
