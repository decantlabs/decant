//! The flow of a structured function as a flat graph, for the descriptor dataflow.
//!
//! The resource pass must know what a scalar register holds at every use, whatever nesting of
//! ifs and loops the structure pass gave the program. This file lowers the [`Node`] tree to
//! blocks with successor edges. Every statement list of the tree is one block, in the order
//! [`crate::ir::Function::for_each_block_mut`] visits them, and joins and loop heads are
//! blocks without statements.
//!
//! Multi-level jumps lowered by `crate::structure` use hoisted boolean flags (`Var::Flag`)
//! that start `false`, are set `true` immediately before leaving an inner region, and are
//! tested by `if (flag)` on the way to their target where they are reset to `false`. Tracking
//! those single-active routing flags while building the graph avoids spurious merge edges
//! between unrelated jump targets without changing the block list or statement order.

use std::collections::{HashMap, HashSet};

use crate::ir::{Expr, Guard, Node, Stmt, Un, Var};

/// One block of the flat graph.
pub struct Block<'a> {
    /// The statements, when the block is a statement list of the tree.
    pub stmts: Option<&'a mut Vec<Stmt>>,
    pub succs: Vec<usize>,
    pub condition: Option<(&'a Expr, bool)>,
}

/// The flat graph. Block 0 is the entry and has no statements.
pub struct Flow<'a> {
    pub blocks: Vec<Block<'a>>,
}

/// Outgoing block indices separated by which structure routing flag, if any, is currently true.
#[derive(Default)]
struct Tails {
    /// Exits where every tracked routing flag is false.
    normal: Vec<usize>,
    /// Exits where the keyed routing flag is true and all other tracked flags are false.
    flagged: HashMap<Var, Vec<usize>>,
}

impl Tails {
    fn is_empty(&self) -> bool {
        self.normal.is_empty() && self.flagged.values().all(Vec::is_empty)
    }

    fn all_blocks(&self) -> Vec<usize> {
        let mut out = self.normal.clone();
        for blocks in self.flagged.values() {
            out.extend_from_slice(blocks);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    fn extend(&mut self, other: Tails) {
        self.normal.extend(other.normal);
        for (flag, blocks) in other.flagged {
            if !blocks.is_empty() {
                self.flagged.entry(flag).or_default().extend(blocks);
            }
        }
    }

    fn dedup(&mut self) {
        self.normal.sort_unstable();
        self.normal.dedup();
        self.flagged.retain(|_, v| {
            v.sort_unstable();
            v.dedup();
            !v.is_empty()
        });
    }
}

/// Finds `Var::Flag` variables that behave strictly as `structure` routing flags: only ever
/// assigned constant `Expr::Bool(_)` under `Guard::Always`, and only ever read as the sole
/// condition (`f` or `!f`) of `Node::If`.
fn collect_routed_flags(body: &[Node]) -> HashSet<Var> {
    fn walk_stmt(stmt: &Stmt, written: &mut HashSet<Var>, bad: &mut HashSet<Var>) {
        if let Stmt::Set { dst, value, guard } = stmt
            && matches!(dst, Var::Flag(_))
        {
            if *guard == Guard::Always && matches!(value, Expr::Bool(_)) {
                written.insert(*dst);
            } else {
                bad.insert(*dst);
            }
        }
        let mut mark_bad_reads = |e: &Expr| {
            crate::resource::reads(e, &mut |v| {
                if matches!(v, Var::Flag(_)) {
                    bad.insert(v);
                }
            });
        };
        match stmt {
            Stmt::Set { value, .. } => mark_bad_reads(value),
            Stmt::Effect { call, .. } => mark_bad_reads(call),
            Stmt::Output { value, .. } => mark_bad_reads(value),
            Stmt::Discard { cond } => mark_bad_reads(cond),
            Stmt::Image(op, _) => op.address.iter().for_each(&mut mark_bad_reads),
            Stmt::Buffer(op, _) => {
                mark_bad_reads(&op.offset);
                if let Some(i) = &op.index {
                    mark_bad_reads(i);
                }
            }
            Stmt::Export(_) | Stmt::Message { .. } | Stmt::Barrier { .. } | Stmt::Comment(_) => {}
        }
    }
    fn walk(
        nodes: &[Node],
        written: &mut HashSet<Var>,
        tested: &mut HashSet<Var>,
        bad: &mut HashSet<Var>,
    ) {
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    let mut set_true = Vec::new();
                    for stmt in stmts {
                        walk_stmt(stmt, written, bad);
                        if let Stmt::Set {
                            dst,
                            value: Expr::Bool(b),
                            guard: Guard::Always,
                        } = stmt
                            && matches!(dst, Var::Flag(_))
                        {
                            if *b {
                                set_true.push(*dst);
                            } else {
                                set_true.retain(|f| f != dst);
                            }
                        }
                    }
                    set_true.sort_unstable();
                    set_true.dedup();
                    if set_true.len() > 1 {
                        bad.extend(set_true);
                    }
                }
                Node::If { cond, then, els } => {
                    match cond {
                        Expr::Var(f) if matches!(f, Var::Flag(_)) => {
                            tested.insert(*f);
                        }
                        Expr::Un(Un::LNot, inner)
                            if matches!(&**inner, Expr::Var(f) if matches!(f, Var::Flag(_))) =>
                        {
                            if let Expr::Var(f) = &**inner {
                                tested.insert(*f);
                            }
                        }
                        _ => {
                            crate::resource::reads(cond, &mut |v| {
                                if matches!(v, Var::Flag(_)) {
                                    bad.insert(v);
                                }
                            });
                        }
                    }
                    walk(then, written, tested, bad);
                    walk(els, written, tested, bad);
                }
                Node::Loop(inner) | Node::Once(inner) => walk(inner, written, tested, bad),
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
    }
    let (mut written, mut tested, mut bad) = (HashSet::new(), HashSet::new(), HashSet::new());
    walk(body, &mut written, &mut tested, &mut bad);
    written
        .into_iter()
        .filter(|f| tested.contains(f) && !bad.contains(f))
        .collect()
}

struct Builder<'a> {
    blocks: Vec<Block<'a>>,
    /// The exits a `Break` of the innermost loop or `Once` leaves from, per open construct.
    breaks: Vec<Tails>,
    /// The head block of each open loop.
    heads: Vec<usize>,
    guarded: bool,
    routed: HashSet<Var>,
}

impl<'a> Builder<'a> {
    fn block(&mut self, stmts: Option<&'a mut Vec<Stmt>>) -> usize {
        self.blocks.push(Block {
            stmts,
            succs: Vec::new(),
            condition: None,
        });
        self.blocks.len() - 1
    }

    fn edges(&mut self, from: &[usize], to: usize) {
        for &f in from {
            if !self.blocks[f].succs.contains(&to) {
                self.blocks[f].succs.push(to);
            }
        }
    }

    /// Lowers `nodes`. `tails` are the blocks control comes from. Returns the blocks control
    /// leaves the sequence from at its end.
    fn seq(&mut self, nodes: &'a mut [Node], mut tails: Tails) -> Tails {
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    let mut cleared = Vec::new();
                    let mut set_true = None;
                    for stmt in stmts.iter() {
                        if let Stmt::Set {
                            dst,
                            value: Expr::Bool(b),
                            guard: Guard::Always,
                        } = stmt
                            && self.routed.contains(dst)
                        {
                            if *b {
                                set_true = Some(*dst);
                                cleared.retain(|f| f != dst);
                            } else {
                                if set_true == Some(*dst) {
                                    set_true = None;
                                }
                                if !cleared.contains(dst) {
                                    cleared.push(*dst);
                                }
                            }
                        }
                    }
                    let b = self.block(Some(stmts));
                    if tails.is_empty() {
                        continue;
                    }
                    let incoming = tails.all_blocks();
                    self.edges(&incoming, b);
                    let mut next = Tails::default();
                    let mut remaining_flags = Vec::new();
                    for (flag, blocks) in tails.flagged {
                        if !blocks.is_empty() && !cleared.contains(&flag) {
                            remaining_flags.push(flag);
                        }
                    }
                    if let Some(f) = set_true {
                        next.flagged.entry(f).or_default().push(b);
                    } else {
                        let cleared_any = incoming.len() > tails.normal.len()
                            && remaining_flags.is_empty();
                        if !tails.normal.is_empty() || cleared_any {
                            next.normal.push(b);
                        }
                        for flag in remaining_flags {
                            next.flagged.entry(flag).or_default().push(b);
                        }
                    }
                    tails = next;
                }
                Node::If { cond, then, els } => {
                    let flag_test = match &*cond {
                        Expr::Var(f) if self.routed.contains(f) => Some((*f, true)),
                        Expr::Un(Un::LNot, inner) => match &**inner {
                            Expr::Var(f) if self.routed.contains(f) => Some((*f, false)),
                            _ => None,
                        },
                        _ => None,
                    };
                    let (a, b) = if let Some((flag, positive)) = flag_test {
                        let taken_blocks = tails.flagged.remove(&flag).unwrap_or_default();
                        let mut true_tails = Tails::default();
                        if !taken_blocks.is_empty() {
                            if self.guarded {
                                let g = self.block(None);
                                self.blocks[g].condition = Some((cond, positive));
                                self.edges(&taken_blocks, g);
                                true_tails.flagged.insert(flag, vec![g]);
                            } else {
                                true_tails.flagged.insert(flag, taken_blocks);
                            }
                        }
                        let mut false_tails = Tails::default();
                        if self.guarded {
                            if !tails.normal.is_empty() {
                                let g = self.block(None);
                                self.blocks[g].condition = Some((cond, !positive));
                                self.edges(&tails.normal, g);
                                false_tails.normal.push(g);
                            }
                            for (other, blocks) in tails.flagged {
                                if !blocks.is_empty() {
                                    let g = self.block(None);
                                    self.blocks[g].condition = Some((cond, !positive));
                                    self.edges(&blocks, g);
                                    false_tails.flagged.insert(other, vec![g]);
                                }
                            }
                        } else {
                            false_tails = tails;
                        }
                        if positive {
                            (true_tails, false_tails)
                        } else {
                            (false_tails, true_tails)
                        }
                    } else if self.guarded {
                        let mut a = Tails::default();
                        let mut b = Tails::default();
                        if !tails.normal.is_empty() {
                            let ga = self.block(None);
                            let gb = self.block(None);
                            self.blocks[ga].condition = Some((cond, true));
                            self.blocks[gb].condition = Some((cond, false));
                            self.edges(&tails.normal, ga);
                            self.edges(&tails.normal, gb);
                            a.normal.push(ga);
                            b.normal.push(gb);
                        }
                        for (flag, blocks) in tails.flagged {
                            if !blocks.is_empty() {
                                let ga = self.block(None);
                                let gb = self.block(None);
                                self.blocks[ga].condition = Some((cond, true));
                                self.blocks[gb].condition = Some((cond, false));
                                self.edges(&blocks, ga);
                                self.edges(&blocks, gb);
                                a.flagged.insert(flag, vec![ga]);
                                b.flagged.insert(flag, vec![gb]);
                            }
                        }
                        (a, b)
                    } else {
                        let cloned = Tails {
                            normal: tails.normal.clone(),
                            flagged: tails.flagged.clone(),
                        };
                        (cloned, tails)
                    };
                    let mut out = self.seq(then, a);
                    out.extend(self.seq(els, b));
                    out.dedup();
                    tails = out;
                }
                Node::Loop(body) => {
                    let head = self.block(None);
                    let incoming = tails.all_blocks();
                    self.edges(&incoming, head);
                    self.heads.push(head);
                    self.breaks.push(Tails::default());
                    let entry = if incoming.is_empty() {
                        Tails::default()
                    } else {
                        Tails {
                            normal: vec![head],
                            flagged: HashMap::new(),
                        }
                    };
                    let end = self.seq(body, entry);
                    self.edges(&end.all_blocks(), head);
                    self.heads.pop();
                    tails = self.breaks.pop().unwrap_or_default();
                    tails.dedup();
                }
                Node::Once(body) => {
                    // No way back: control goes on from every `Break` and from the end.
                    self.breaks.push(Tails::default());
                    let mut out = self.seq(body, tails);
                    if let Some(broken) = self.breaks.pop() {
                        out.extend(broken);
                    }
                    out.dedup();
                    tails = out;
                }
                Node::Break => {
                    if let Some(open) = self.breaks.last_mut() {
                        open.extend(tails);
                    }
                    tails = Tails::default();
                }
                Node::Continue => {
                    if let Some(&head) = self.heads.last() {
                        self.edges(&tails.all_blocks(), head);
                    }
                    tails = Tails::default();
                }
                Node::Return => tails = Tails::default(),
            }
        }
        tails
    }
}

impl<'a> Flow<'a> {
    pub fn build(body: &'a mut [Node], guarded: bool) -> Flow<'a> {
        let routed = collect_routed_flags(body);
        if !routed.is_empty() && can_route(body, &routed) {
            Self::build_with(body, guarded, routed)
        } else {
            Self::build_with(body, guarded, HashSet::new())
        }
    }

    fn build_with(body: &'a mut [Node], guarded: bool, routed: HashSet<Var>) -> Flow<'a> {
        let mut b = Builder {
            blocks: Vec::new(),
            breaks: Vec::new(),
            heads: Vec::new(),
            guarded,
            routed,
        };
        let entry = b.block(None);
        b.seq(
            body,
            Tails {
                normal: vec![entry],
                flagged: HashMap::new(),
            },
        );
        Flow { blocks: b.blocks }
    }

    /// The blocks that lead to each block.
    pub fn predecessors(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.blocks.len()];
        for (i, b) in self.blocks.iter().enumerate() {
            for &s in &b.succs {
                out[s].push(i);
            }
        }
        out
    }
}

/// Verifies on `&[Node]` that single-active flag routing encounters no simultaneous active
/// flags, flagged loop entries, or flagged continues.
fn can_route(body: &[Node], routed: &HashSet<Var>) -> bool {
    #[derive(Clone, Default)]
    struct Flags {
        normal: bool,
        flagged: HashSet<Var>,
    }
    impl Flags {
        fn extend(&mut self, other: Flags) {
            self.normal |= other.normal;
            self.flagged.extend(other.flagged);
        }
    }
    fn walk(
        nodes: &[Node],
        mut cur: Flags,
        routed: &HashSet<Var>,
        breaks: &mut Vec<Flags>,
    ) -> Option<Flags> {
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    if !cur.normal && cur.flagged.is_empty() {
                        continue;
                    }
                    let mut cleared = Vec::new();
                    let mut set_true = None;
                    for stmt in stmts {
                        if let Stmt::Set {
                            dst,
                            value: Expr::Bool(b),
                            guard: Guard::Always,
                        } = stmt
                            && routed.contains(dst)
                        {
                            if *b {
                                set_true = Some(*dst);
                                cleared.retain(|f| f != dst);
                            } else {
                                if set_true == Some(*dst) {
                                    set_true = None;
                                }
                                if !cleared.contains(dst) {
                                    cleared.push(*dst);
                                }
                            }
                        }
                    }
                    let had_flags = !cur.flagged.is_empty();
                    let remaining: HashSet<Var> = cur
                        .flagged
                        .into_iter()
                        .filter(|f| !cleared.contains(f))
                        .collect();
                    let mut next = Flags::default();
                    if let Some(f) = set_true {
                        if remaining.iter().any(|&other| other != f) {
                            return None;
                        }
                        next.flagged.insert(f);
                    } else {
                        next.normal = cur.normal || (had_flags && remaining.is_empty());
                        next.flagged = remaining;
                    }
                    cur = next;
                }
                Node::If { cond, then, els } => {
                    let flag_test = match cond {
                        Expr::Var(f) if routed.contains(f) => Some((*f, true)),
                        Expr::Un(Un::LNot, inner) => match &**inner {
                            Expr::Var(f) if routed.contains(f) => Some((*f, false)),
                            _ => None,
                        },
                        _ => None,
                    };
                    let (a, b) = if let Some((flag, positive)) = flag_test {
                        let had = cur.flagged.remove(&flag);
                        let mut true_f = Flags::default();
                        if had {
                            true_f.flagged.insert(flag);
                        }
                        if positive { (true_f, cur) } else { (cur, true_f) }
                    } else {
                        (cur.clone(), cur)
                    };
                    let mut out = walk(then, a, routed, breaks)?;
                    out.extend(walk(els, b, routed, breaks)?);
                    cur = out;
                }
                Node::Loop(inner) => {
                    if !cur.flagged.is_empty() {
                        return None;
                    }
                    breaks.push(Flags::default());
                    let end = walk(
                        inner,
                        Flags {
                            normal: cur.normal,
                            flagged: HashSet::new(),
                        },
                        routed,
                        breaks,
                    )?;
                    if !end.flagged.is_empty() {
                        return None;
                    }
                    cur = breaks.pop().unwrap_or_default();
                }
                Node::Once(inner) => {
                    breaks.push(Flags::default());
                    let mut out = walk(inner, cur, routed, breaks)?;
                    if let Some(broken) = breaks.pop() {
                        out.extend(broken);
                    }
                    cur = out;
                }
                Node::Break => {
                    if let Some(open) = breaks.last_mut() {
                        open.extend(cur);
                    }
                    cur = Flags::default();
                }
                Node::Continue => {
                    if !cur.flagged.is_empty() {
                        return None;
                    }
                    cur = Flags::default();
                }
                Node::Return => cur = Flags::default(),
            }
        }
        Some(cur)
    }
    walk(
        body,
        Flags {
            normal: true,
            flagged: HashSet::new(),
        },
        routed,
        &mut Vec::new(),
    )
    .is_some()
}
