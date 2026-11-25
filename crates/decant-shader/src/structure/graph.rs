//! The graph the structure pass works on, and what it needs to know about it: an order, the
//! dominators, the loops and where each node is placed in the structured text.
//!
//! The nodes are the native blocks with their lifted statements. The pass may add nodes of its
//! own (see `reduce`), so the graph is not the native one.

use crate::cfg::{Cfg, Cond, Term};
use crate::ir::{Expr, Stmt, Un, Var};

/// How a node ends.
#[derive(Clone, Debug)]
pub(super) enum Exit {
    End,
    Jump(usize),
    Branch {
        cond: Expr,
        taken: usize,
        fall: usize,
    },
}

/// Node 0 is the entry. Every node can be reached from it.
#[derive(Clone, Debug)]
pub(super) struct Graph {
    pub code: Vec<Vec<Stmt>>,
    pub exit: Vec<Exit>,
}

/// The condition of a native branch as an expression over the state of one lane.
///
/// A branch of the machine is one decision for the whole wave. The translation runs one lane
/// per invocation and takes the decision from this lane's bit, which is the meaning the lane
/// rules of `crate::exec` give to the registers. This function is the one place that says so.
pub(super) fn condition(cond: Cond) -> Expr {
    match cond {
        Cond::Scc(true) => Expr::Var(Var::Scc),
        Cond::Scc(false) => not(Expr::Var(Var::Scc)),
        Cond::VccZero(true) => not(Expr::Var(Var::Vcc)),
        Cond::VccZero(false) => Expr::Var(Var::Vcc),
        Cond::ExecZero(true) => not(Expr::Var(Var::Exec)),
        Cond::ExecZero(false) => Expr::Var(Var::Exec),
    }
}

/// The opposite of a condition, without stacking negations.
pub(super) fn not(e: Expr) -> Expr {
    match e {
        Expr::Un(Un::LNot, inner) => *inner,
        Expr::Bool(b) => Expr::Bool(!b),
        other => Expr::un(Un::LNot, other),
    }
}

impl Graph {
    pub fn new(cfg: &Cfg, lifted: Vec<Vec<Stmt>>) -> Graph {
        let exit = cfg
            .blocks
            .iter()
            .map(|b| match b.term {
                Term::End => Exit::End,
                Term::Jump(t) => Exit::Jump(t),
                // Both ways to one place is a jump: the condition has no effect.
                Term::Branch { taken, fall, .. } if taken == fall => Exit::Jump(taken),
                Term::Branch { cond, taken, fall } => Exit::Branch {
                    cond: condition(cond),
                    taken,
                    fall,
                },
            })
            .collect();
        Graph { code: lifted, exit }
    }

    pub fn len(&self) -> usize {
        self.exit.len()
    }

    /// The nodes `n` leads to, the taken side first.
    pub fn successors(&self, n: usize) -> impl Iterator<Item = usize> + use<> {
        let (a, b) = match self.exit[n] {
            Exit::End => (None, None),
            Exit::Jump(t) => (Some(t), None),
            Exit::Branch { taken, fall, .. } => (Some(taken), Some(fall)),
        };
        a.into_iter().chain(b)
    }

    /// Adds a node and gives its number.
    pub fn push(&mut self, code: Vec<Stmt>, exit: Exit) -> usize {
        self.code.push(code);
        self.exit.push(exit);
        self.exit.len() - 1
    }
}

const NONE: usize = usize::MAX;

/// Order and dominators of a graph.
pub(super) struct Order {
    /// The nodes in reverse postorder of a depth first walk: a node comes before the nodes it
    /// leads to, except along edges that go back.
    pub order: Vec<usize>,
    /// The position of each node in `order`.
    pub number: Vec<usize>,
    /// The immediate dominator. The entry is its own.
    pub idom: Vec<usize>,
    /// The nodes that lead to each node, each once.
    pub preds: Vec<Vec<usize>>,
    /// Entry and exit time of each node in a walk of the dominator tree.
    span: Vec<(u32, u32)>,
    /// How many nodes each node dominates, itself included.
    pub size: Vec<u32>,
}

impl Order {
    pub fn new(g: &Graph) -> Order {
        let n = g.len();
        // Postorder without recursion: programs have thousands of blocks in a row.
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let ways = |b: usize| {
            let mut v: Vec<usize> = g.successors(b).collect();
            v.reverse();
            v
        };
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, ways(0))];
        seen[0] = true;
        while let Some((node, rest)) = stack.last_mut() {
            // The taken side is walked first, so the fall through side comes right after its
            // branch in the reverse order, as in the native code.
            if let Some(next) = rest.pop() {
                if !seen[next] {
                    seen[next] = true;
                    stack.push((next, ways(next)));
                }
            } else {
                post.push(*node);
                stack.pop();
            }
        }
        let order: Vec<usize> = post.into_iter().rev().collect();
        debug_assert_eq!(order.len(), n, "every node is reachable");
        let mut number = vec![NONE; n];
        for (i, &b) in order.iter().enumerate() {
            number[b] = i;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for s in g.successors(b) {
                if !preds[s].contains(&b) {
                    preds[s].push(b);
                }
            }
        }
        // Dominators by the iteration of Cooper, Harvey and Kennedy.
        let mut idom = vec![NONE; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = NONE;
                for &p in &preds[b] {
                    if idom[p] == NONE {
                        continue;
                    }
                    new = if new == NONE {
                        p
                    } else {
                        let (mut x, mut y) = (p, new);
                        while x != y {
                            while number[x] > number[y] {
                                x = idom[x];
                            }
                            while number[y] > number[x] {
                                y = idom[y];
                            }
                        }
                        x
                    };
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        // A walk of the dominator tree gives every node a span that contains the spans of the
        // nodes it dominates. A dominator comes earlier in `order`, so one pass each way does it.
        let mut size = vec![1u32; n];
        for &b in order.iter().rev() {
            if b != 0 {
                size[idom[b]] += size[b];
            }
        }
        let mut span = vec![(0u32, 0u32); n];
        let mut next_child = vec![0u32; n];
        span[0] = (0, size[0]);
        next_child[0] = 1;
        for &b in &order[1..] {
            let d = idom[b];
            span[b] = (next_child[d], next_child[d] + size[b]);
            next_child[d] += size[b];
            next_child[b] = span[b].0 + 1;
        }
        Order {
            order,
            number,
            idom,
            preds,
            span,
            size,
        }
    }

    /// Whether every path from the entry to `b` goes through `a`. A node dominates itself.
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        self.span[a].0 <= self.span[b].0 && self.span[b].1 <= self.span[a].1
    }

    /// Whether the edge from `from` to `to` closes a loop.
    pub fn back(&self, from: usize, to: usize) -> bool {
        self.dominates(to, from)
    }

    /// An edge that goes back in the order without closing a loop of one entry, when the graph
    /// has one: the graph is then not reducible.
    pub fn irreducible_edge(&self, g: &Graph) -> Option<(usize, usize)> {
        self.order.iter().find_map(|&b| {
            g.successors(b)
                .find(|&s| self.number[s] <= self.number[b] && !self.back(b, s))
                .map(|s| (b, s))
        })
    }
}

/// The loops of a reducible graph. A loop is named by its header.
pub(super) struct Loops {
    pub header: Vec<bool>,
    /// The header of the innermost loop a node is in. A header is in its own loop.
    inner: Vec<usize>,
    /// For a header: the header of the next loop around it.
    parent: Vec<usize>,
}

impl Loops {
    pub fn new(g: &Graph, o: &Order) -> Loops {
        let n = g.len();
        let mut header = vec![false; n];
        let mut inner = vec![NONE; n];
        let mut parent = vec![NONE; n];
        // Inner loops first: their headers come later in the order.
        for &h in o.order.iter().rev() {
            let mut work: Vec<usize> = o.preds[h]
                .iter()
                .copied()
                .filter(|&p| o.back(p, h))
                .collect();
            if work.is_empty() {
                continue;
            }
            header[h] = true;
            inner[h] = h;
            while let Some(x) = work.pop() {
                if inner[x] == NONE {
                    inner[x] = h;
                    work.extend(o.preds[x].iter().copied());
                    continue;
                }
                // Already in a loop: the outermost loop known around it becomes a part.
                let mut top = inner[x];
                while parent[top] != NONE {
                    top = parent[top];
                }
                if top != h {
                    parent[top] = h;
                    work.extend(o.preds[top].iter().copied());
                }
            }
        }
        Loops {
            header,
            inner,
            parent,
        }
    }

    /// The innermost loop around a node.
    pub fn innermost(&self, node: usize) -> Option<usize> {
        Some(self.inner[node]).filter(|&h| h != NONE)
    }

    /// The loop around the loop of header `h`.
    pub fn outer(&self, h: usize) -> Option<usize> {
        Some(self.parent[h]).filter(|&p| p != NONE)
    }

    /// Whether `node` is in the loop of header `h`.
    pub fn contains(&self, h: usize, node: usize) -> bool {
        let mut l = self.innermost(node);
        while let Some(x) = l {
            if x == h {
                return true;
            }
            l = self.outer(x);
        }
        false
    }
}

/// Where the text of each node goes.
pub(super) struct Places {
    /// Nodes that several forward edges lead to, by the node whose text they follow: the node
    /// that dominates them. In order.
    pub merges: Vec<Vec<usize>>,
    /// Nodes that follow a loop, by its header. In order.
    pub followers: Vec<Vec<usize>>,
    /// Whether a node is in one of the lists above. Such a node is jumped to. Any other node
    /// has one forward edge leading to it and is written where that edge leaves.
    pub jumped_to: Vec<bool>,
}

impl Places {
    pub fn new(g: &Graph, o: &Order, loops: &Loops) -> Places {
        let n = g.len();
        let mut p = Places {
            merges: vec![Vec::new(); n],
            followers: vec![Vec::new(); n],
            jumped_to: vec![false; n],
        };
        // A node with one way in that leaves a loop, by the loop it leaves.
        let mut exits: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &y in &o.order[1..] {
            let x = o.idom[y];
            // The outermost loop that the dominator is in and the node is not: the text of the
            // node cannot be inside that loop without being run again by it.
            let mut left = None;
            let mut l = loops.innermost(x);
            while let Some(h) = l {
                if loops.contains(h, y) {
                    break;
                }
                left = Some(h);
                l = loops.outer(h);
            }
            let forward = o.preds[y].iter().filter(|&&q| !o.back(q, y)).count();
            if forward >= 2 {
                match left {
                    Some(h) => p.followers[h].push(y),
                    None => p.merges[x].push(y),
                }
                p.jumped_to[y] = true;
            } else if let Some(h) = left
                && loops.innermost(x) == Some(h)
            {
                exits[h].push(y);
            }
        }
        // A loop that nothing follows yet takes the largest of its one way exits as the text
        // after it, so a plain `while` does not hold the rest of the program in its body. The
        // other one way exits stay where they leave: they are the code of a `break`.
        for (h, exits) in exits.iter().enumerate() {
            let largest = exits
                .iter()
                .max_by_key(|&&y| (o.size[y], usize::MAX - o.number[y]));
            if p.followers[h].is_empty()
                && let Some(&y) = largest
            {
                p.followers[h].push(y);
                p.jumped_to[y] = true;
            }
        }
        p
    }
}

/// The most statements all the copies of one block that ends the program may add.
const END_COPY: usize = 64;

/// Gives each short block of `joins` that ends the program one copy for each edge that leads
/// to it. Returns whether it copied one.
///
/// Compilers share the last instructions of a program between all the places that leave it
/// early: a pixel program has one block with the export that discards, and every test for
/// "no lane left" branches there. As one node it is a join that half the program jumps to,
/// which costs a block that runs once around everything before it and a boolean for every jump
/// from inside a loop. As copies it is the body of each `if`, followed by `return`. These few
/// statements are the only code the pass ever copies, and only for a join that plain nesting
/// could not take away.
pub(super) fn copy_ends(g: &mut Graph, joins: &[usize]) -> bool {
    let n = g.len();
    let mut edges = vec![0usize; n];
    for p in 0..n {
        g.successors(p).for_each(|s| edges[s] += 1);
    }
    let mut wanted = vec![false; n];
    for &t in joins {
        let statements = g.code[t]
            .iter()
            .filter(|s| !matches!(s, Stmt::Comment(_)))
            .count();
        wanted[t] =
            matches!(g.exit[t], Exit::End) && edges[t].saturating_sub(1) * statements <= END_COPY;
    }
    if !wanted.contains(&true) {
        return false;
    }
    let mut reached = vec![false; n];
    for p in 0..n {
        let mut exit = g.exit[p].clone();
        let targets: Vec<&mut usize> = match &mut exit {
            Exit::End => Vec::new(),
            Exit::Jump(t) => vec![t],
            Exit::Branch { taken, fall, .. } => vec![taken, fall],
        };
        for t in targets {
            if !wanted[*t] {
                continue;
            }
            // The first edge keeps the block itself.
            if std::mem::replace(&mut reached[*t], true) {
                *t = g.push(g.code[*t].clone(), Exit::End);
            }
        }
        g.exit[p] = exit;
    }
    true
}

/// The selector words of [`reduce`] are `Var::Tmp` from this number DOWN, next to the booleans
/// of `lower`. `Var::Tmp(0xFFFF)` is the sink of the null register.
pub const FIRST_SELECTOR: u16 = 0xFFFE;

/// The strongly connected parts of the graph among `nodes` that hold a cycle, by Tarjan's
/// algorithm without recursion. `inside` marks `nodes`.
fn cycles(g: &Graph, nodes: &[usize], inside: &[bool]) -> Vec<Vec<usize>> {
    let n = g.len();
    let (mut index, mut low, mut on) = (vec![NONE; n], vec![0usize; n], vec![false; n]);
    let (mut stack, mut out, mut next) = (Vec::new(), Vec::new(), 0usize);
    let ways = |v: usize| {
        g.successors(v)
            .filter(|&s| inside[s])
            .collect::<Vec<usize>>()
    };
    for &root in nodes {
        if index[root] != NONE {
            continue;
        }
        let mut calls: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut enter = Some(root);
        loop {
            if let Some(v) = enter.take() {
                index[v] = next;
                low[v] = next;
                next += 1;
                stack.push(v);
                on[v] = true;
                calls.push((v, ways(v)));
            }
            let Some((v, rest)) = calls.last_mut() else {
                break;
            };
            let v = *v;
            if let Some(w) = rest.pop() {
                if index[w] == NONE {
                    enter = Some(w);
                } else if on[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            calls.pop();
            if let Some((parent, _)) = calls.last() {
                low[*parent] = low[*parent].min(low[v]);
            }
            if low[v] == index[v] {
                let mut part = Vec::new();
                loop {
                    let w = stack.pop().expect("the root of a part is on the stack");
                    on[w] = false;
                    part.push(w);
                    if w == v {
                        break;
                    }
                }
                if part.len() > 1 || g.successors(v).any(|s| s == v) {
                    out.push(part);
                }
            }
        }
    }
    out
}

/// Makes the graph reducible: every loop gets one way in.
///
/// A loop with several entries cannot be written as nested text. Such a loop gets a new header
/// that tests a selector word, and every edge to one of the entries, from outside or inside,
/// first sets the selector and goes to the header. After that the header is the one way in and
/// the edges to it from inside are plain back edges. The same is done for the loops inside,
/// with the header taken out. This is what LLVM's `FixIrreducible` pass does to a function.
/// Compilers make such loops when they move the test of a `while` to its end and jump into
/// the middle for the first round.
///
/// Returns how many headers it added.
pub(super) fn reduce(g: &mut Graph) -> u16 {
    let mut added = 0u16;
    let mut work: Vec<Vec<usize>> = vec![(0..g.len()).collect()];
    let mut inside = vec![false; g.len()];
    while let Some(nodes) = work.pop() {
        inside.resize(g.len(), false);
        nodes.iter().for_each(|&x| inside[x] = true);
        let parts = cycles(g, &nodes, &inside);
        nodes.iter().for_each(|&x| inside[x] = false);
        for mut part in parts {
            inside.resize(g.len(), false);
            part.iter().for_each(|&x| inside[x] = true);
            // The entries: nodes of the part that an edge from outside leads to.
            let mut entry = vec![false; g.len()];
            entry[0] = inside[0];
            for p in (0..g.len()).filter(|&p| !inside[p]) {
                g.successors(p)
                    .filter(|&s| inside[s])
                    .for_each(|s| entry[s] = true);
            }
            let mut entries: Vec<usize> = part.iter().copied().filter(|&x| entry[x]).collect();
            entries.sort_unstable();
            let header = if let [one] = entries[..] {
                one
            } else {
                let selector = Var::Tmp(FIRST_SELECTOR - added);
                added += 1;
                let before = g.len();
                // The header: a chain of tests, the last entry needs none.
                for (i, &e) in entries[..entries.len() - 1].iter().enumerate() {
                    let fall = if i + 2 == entries.len() {
                        entries[i + 1]
                    } else {
                        before + i + 1
                    };
                    let cond =
                        Expr::bin(crate::ir::Bin::Eq, Expr::Var(selector), Expr::U32(i as u32));
                    part.push(g.push(
                        Vec::new(),
                        Exit::Branch {
                            cond,
                            taken: e,
                            fall,
                        },
                    ));
                }
                let chain_end = g.len();
                // One stub per entry and side: it sets the selector and goes to the header.
                let mut stubs: Vec<[Option<usize>; 2]> = vec![[None; 2]; entries.len()];
                for (p, &within) in inside.iter().enumerate().take(before) {
                    let side = usize::from(within);
                    let mut exit = g.exit[p].clone();
                    let targets: Vec<&mut usize> = match &mut exit {
                        Exit::End => Vec::new(),
                        Exit::Jump(t) => vec![t],
                        Exit::Branch { taken, fall, .. } => vec![taken, fall],
                    };
                    for t in targets {
                        let Ok(i) = entries.binary_search(t) else {
                            continue;
                        };
                        *t = *stubs[i][side].get_or_insert_with(|| {
                            let set = Stmt::Set {
                                dst: selector,
                                value: Expr::U32(i as u32),
                                guard: crate::ir::Guard::Always,
                            };
                            g.push(vec![set], Exit::Jump(before))
                        });
                    }
                    g.exit[p] = exit;
                }
                part.extend(stubs.iter().filter_map(|s| s[1]));
                debug_assert!(chain_end > before, "several entries give at least one test");
                before
            };
            inside.resize(g.len(), false);
            part.iter().for_each(|&x| inside[x] = false);
            part.retain(|&x| x != header);
            work.push(part);
        }
    }
    added
}
