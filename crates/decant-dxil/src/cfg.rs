//! Control flow normalisation before lowering.
//!
//! What it does, in order:
//!
//! 1. drops unreachable blocks and the phi entries that came from them,
//! 2. turns a conditional branch with two equal targets into a branch,
//! 3. gives the entry block no predecessors,
//! 4. rejects irreducible graphs (a retreating edge whose target does not dominate its source),
//! 5. folds an empty latch (no phis, no instructions, only a branch back to the header, one
//!    predecessor) into its predecessor, so that the exiting block is the latch,
//! 6. gives every loop one preheader and dedicated exit blocks,
//! 7. splits the critical edges that are not back edges,
//! 8. removes dead phis (iterated: a phi only read by dead phis is dead),
//! 9. lays the blocks out in reverse postorder, so that every operand but a phi input is defined
//!    earlier in the stream.

use crate::Error;
use crate::ir::{Block, BlockId, Operand, Phi, Shader, Terminator, ValueId};

fn succs(term: &Terminator) -> Vec<usize> {
    match term {
        Terminator::Br(t) => vec![t.0 as usize],
        Terminator::CondBr(_, a, b) => vec![a.0 as usize, b.0 as usize],
        Terminator::Ret => Vec::new(),
    }
}

fn retarget(term: &mut Terminator, from: usize, to: usize) {
    let fix = |b: &mut BlockId| {
        if b.0 as usize == from {
            b.0 = to as u32;
        }
    };
    match term {
        Terminator::Br(t) => fix(t),
        Terminator::CondBr(_, a, b) => {
            fix(a);
            fix(b);
        }
        Terminator::Ret => {}
    }
}

fn preds(blocks: &[Block]) -> Vec<Vec<usize>> {
    let mut p = vec![Vec::new(); blocks.len()];
    for (i, b) in blocks.iter().enumerate() {
        for s in succs(&b.term) {
            p[s].push(i);
        }
    }
    p
}

/// Reverse postorder, immediate dominators and the retreating edges of a graph.
pub(crate) struct Analysis {
    /// Reachable blocks in reverse postorder.
    pub rpo: Vec<usize>,
    /// Position in `rpo`, `usize::MAX` for unreachable blocks.
    pub pos: Vec<usize>,
    idom: Vec<usize>,
    /// Retreating edges of the DFS (source, target).
    pub retreating: Vec<(usize, usize)>,
}

impl Analysis {
    pub fn new(blocks: &[Block]) -> Analysis {
        let n = blocks.len();
        let mut color = vec![0u8; n];
        let mut post = Vec::new();
        let mut retreating = Vec::new();
        // Iterative DFS. Successors are visited false target first, so that the true target comes
        // first in reverse postorder.
        let mut stack: Vec<(usize, Vec<usize>)> = Vec::new();
        if n > 0 {
            color[0] = 1;
            let mut s = succs(&blocks[0].term);
            s.reverse();
            stack.push((0, s));
        }
        while let Some((b, rest)) = stack.last_mut() {
            let b = *b;
            if let Some(s) = rest.pop() {
                match color[s] {
                    0 => {
                        color[s] = 1;
                        let mut ss = succs(&blocks[s].term);
                        ss.reverse();
                        stack.push((s, ss));
                    }
                    1 => retreating.push((b, s)),
                    _ => {}
                }
            } else {
                color[b] = 2;
                post.push(b);
                stack.pop();
            }
        }
        let rpo: Vec<usize> = post.into_iter().rev().collect();
        let mut pos = vec![usize::MAX; n];
        for (i, &b) in rpo.iter().enumerate() {
            pos[b] = i;
        }
        let p = preds(blocks);
        let mut idom = vec![usize::MAX; n];
        if !rpo.is_empty() {
            idom[0] = 0;
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                let mut new = usize::MAX;
                for &q in &p[b] {
                    if pos[q] == usize::MAX || idom[q] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX { q } else { intersect(&idom, &pos, q, new) };
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        Analysis { rpo, pos, idom, retreating }
    }

    /// Whether `a` dominates `b` (both reachable).
    pub fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            if b == 0 || self.idom[b] == usize::MAX {
                return false;
            }
            b = self.idom[b];
        }
    }

    pub fn is_back_edge(&self, from: usize, to: usize) -> bool {
        self.pos[from] != usize::MAX && self.dominates(to, from)
    }
}

fn intersect(idom: &[usize], pos: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while pos[a] > pos[b] {
            a = idom[a];
        }
        while pos[b] > pos[a] {
            b = idom[b];
        }
    }
    a
}

struct Cfg {
    blocks: Vec<Block>,
    next_value: u32,
}

impl Cfg {
    /// SplitBlockPredecessors: a new block that branches to `target`, takes over the edges from
    /// `from` (all predecessors of `target` in that list), and merges the phi inputs of `target`.
    fn split_edges(&mut self, target: usize, from: &[usize]) -> usize {
        let new = self.blocks.len();
        let mut new_phis = Vec::new();
        for pi in 0..self.blocks[target].phis.len() {
            let phi = &mut self.blocks[target].phis[pi];
            let moved: Vec<(BlockId, Operand)> = phi.incoming.iter().filter(|(b, _)| from.contains(&(b.0 as usize))).cloned().collect();
            phi.incoming.retain(|(b, _)| !from.contains(&(b.0 as usize)));
            let value = if moved.len() == 1 {
                moved[0].1
            } else {
                let id = ValueId(self.next_value);
                self.next_value += 1;
                new_phis.push(Phi { dst: id, ty: phi.ty, incoming: moved });
                Operand::Value(id)
            };
            phi.incoming.push((BlockId(new as u32), value));
        }
        for &p in from {
            retarget(&mut self.blocks[p].term, target, new);
        }
        self.blocks.push(Block { phis: new_phis, insts: Vec::new(), term: Terminator::Br(BlockId(target as u32)) });
        new
    }

    fn rename_pred(&mut self, block: usize, old: usize, new: usize) {
        for phi in &mut self.blocks[block].phis {
            for (b, _) in &mut phi.incoming {
                if b.0 as usize == old {
                    b.0 = new as u32;
                }
            }
        }
    }
}

/// The natural loop of `header`: the blocks that reach a latch without going through the header.
fn loop_body(blocks: &[Block], p: &[Vec<usize>], an: &Analysis, header: usize) -> Vec<bool> {
    let mut body = vec![false; blocks.len()];
    body[header] = true;
    let mut work: Vec<usize> = p[header].iter().copied().filter(|&q| an.is_back_edge(q, header)).collect();
    while let Some(b) = work.pop() {
        if body[b] || an.pos[b] == usize::MAX {
            continue;
        }
        body[b] = true;
        work.extend(p[b].iter().copied());
    }
    body
}

/// Normalises `shader` in place, see the module documentation.
pub fn normalize(shader: &mut Shader) -> Result<(), Error> {
    if shader.blocks.is_empty() {
        return Err(Error::Invalid("the shader has no blocks".into()));
    }
    let mut max = 0u32;
    for b in &shader.blocks {
        for p in &b.phis {
            max = max.max(p.dst.0 + 1);
        }
        for i in &b.insts {
            for d in &i.dsts {
                max = max.max(d.0 + 1);
            }
        }
    }
    for (i, b) in shader.blocks.iter().enumerate() {
        for s in succs(&b.term) {
            if s >= shader.blocks.len() {
                return Err(Error::Invalid(format!("block {i} branches to block {s}, which does not exist")));
            }
        }
    }
    let mut cfg = Cfg { blocks: std::mem::take(&mut shader.blocks), next_value: max };

    // 2. condbr with equal targets.
    for b in &mut cfg.blocks {
        if let Terminator::CondBr(_, t, f) = b.term
            && t == f
        {
            b.term = Terminator::Br(t);
        }
    }
    // 1. unreachable blocks: emptied here, removed by the final layout. Phi inputs from them go now.
    let an = Analysis::new(&cfg.blocks);
    for b in 0..cfg.blocks.len() {
        if an.pos[b] == usize::MAX {
            continue;
        }
        for phi in &mut cfg.blocks[b].phis {
            phi.incoming.retain(|(q, _)| an.pos[q.0 as usize] != usize::MAX);
        }
    }
    // 3. entry without predecessors.
    if preds(&cfg.blocks)[0].iter().any(|&p| an.pos[p] != usize::MAX) {
        let new = cfg.blocks.len();
        let old = std::mem::replace(&mut cfg.blocks[0], Block { phis: Vec::new(), insts: Vec::new(), term: Terminator::Br(BlockId(new as u32)) });
        cfg.blocks.push(old);
        // Every edge and phi input that named block 0 now names the moved block (block 0 itself only branches to it).
        for i in 1..cfg.blocks.len() {
            let t = &mut cfg.blocks[i].term;
            retarget(t, 0, new);
            cfg.rename_pred(i, 0, new);
        }
    }
    // 4. reducibility.
    let an = Analysis::new(&cfg.blocks);
    for &(u, v) in &an.retreating {
        if !an.dominates(v, u) {
            return Err(Error::Irreducible(format!("the edge from block {u} to block {v} is a retreating edge, but block {v} does not dominate block {u}")));
        }
    }
    // 5. latch folding.
    loop {
        let an = Analysis::new(&cfg.blocks);
        let p = preds(&cfg.blocks);
        let mut fold = None;
        for &l in &an.rpo {
            let b = &cfg.blocks[l];
            let Terminator::Br(h) = b.term else { continue };
            let h = h.0 as usize;
            if l == 0 || h == l || !b.phis.is_empty() || !b.insts.is_empty() || !an.is_back_edge(l, h) {
                continue;
            }
            let live: Vec<usize> = p[l].iter().copied().filter(|&q| an.pos[q] != usize::MAX).collect();
            if live.len() == 1 && !p[h].contains(&live[0]) {
                fold = Some((l, h, live[0]));
                break;
            }
        }
        let Some((l, h, pred)) = fold else { break };
        retarget(&mut cfg.blocks[pred].term, l, h);
        cfg.rename_pred(h, l, pred);
        // The latch is now unreachable and gets dropped by the layout.
    }
    // 6. loop simplification.
    for _ in 0..10_000 {
        let an = Analysis::new(&cfg.blocks);
        let p = preds(&cfg.blocks);
        let mut headers: Vec<usize> = Vec::new();
        for &q in &an.rpo {
            for s in succs(&cfg.blocks[q].term) {
                if an.is_back_edge(q, s) && !headers.contains(&s) {
                    headers.push(s);
                }
            }
        }
        let mut fixed = false;
        for h in headers {
            let body = loop_body(&cfg.blocks, &p, &an, h);
            let outside: Vec<usize> = p[h].iter().copied().filter(|&q| !body[q] && an.pos[q] != usize::MAX).collect();
            if outside.len() != 1 || succs(&cfg.blocks[outside[0]].term).len() != 1 {
                cfg.split_edges(h, &outside);
                fixed = true;
                break;
            }
            for b in (0..cfg.blocks.len()).filter(|&b| body[b]) {
                for s in succs(&cfg.blocks[b].term) {
                    if body[s] || p[s].iter().all(|&q| body[q]) {
                        continue;
                    }
                    let inside: Vec<usize> = p[s].iter().copied().filter(|&q| body[q]).collect();
                    cfg.split_edges(s, &inside);
                    fixed = true;
                    break;
                }
                if fixed {
                    break;
                }
            }
            if fixed {
                break;
            }
        }
        if !fixed {
            break;
        }
    }
    // 7. critical edges that are not back edges.
    let an = Analysis::new(&cfg.blocks);
    let p = preds(&cfg.blocks);
    let mut edges = Vec::new();
    for &u in &an.rpo {
        let s = succs(&cfg.blocks[u].term);
        if s.len() < 2 {
            continue;
        }
        for v in s {
            let live_preds = p[v].iter().filter(|&&q| an.pos[q] != usize::MAX).count();
            if live_preds >= 2 && !an.is_back_edge(u, v) {
                edges.push((u, v));
            }
        }
    }
    for (u, v) in edges {
        cfg.split_edges(v, &[u]);
    }
    // 8. dead phis.
    remove_dead_phis(&mut cfg.blocks);
    // 9. layout.
    let an = Analysis::new(&cfg.blocks);
    let mut slots: Vec<Option<Block>> = cfg.blocks.into_iter().map(Some).collect();
    let mut out = Vec::with_capacity(an.rpo.len());
    for &old in &an.rpo {
        let mut b = slots[old].take().expect("each block once");
        let map = |id: &mut BlockId| id.0 = an.pos[id.0 as usize] as u32;
        match &mut b.term {
            Terminator::Br(t) => map(t),
            Terminator::CondBr(_, a, c) => {
                map(a);
                map(c);
            }
            Terminator::Ret => {}
        }
        for phi in &mut b.phis {
            phi.incoming.retain(|(q, _)| an.pos[q.0 as usize] != usize::MAX);
            for (q, _) in &mut phi.incoming {
                map(q);
            }
        }
        out.push(b);
    }
    shader.blocks = out;
    Ok(())
}

fn uses(op: &Operand, mark: &mut impl FnMut(ValueId)) {
    if let Operand::Value(v) = op {
        mark(*v);
    }
}

/// Removes phis that nothing reads, iterated through chains of phis.
fn remove_dead_phis(blocks: &mut [Block]) {
    let mut live = std::collections::HashSet::new();
    for b in blocks.iter() {
        for i in &b.insts {
            crate::lower::for_each_operand(&i.op, &mut |o| uses(o, &mut |v| {
                live.insert(v);
            }));
        }
        if let Terminator::CondBr(c, _, _) = &b.term {
            uses(c, &mut |v| {
                live.insert(v);
            });
        }
    }
    loop {
        let mut grew = false;
        for b in blocks.iter() {
            for phi in &b.phis {
                if live.contains(&phi.dst) {
                    for (_, o) in &phi.incoming {
                        uses(o, &mut |v| grew |= live.insert(v));
                    }
                }
            }
        }
        if !grew {
            break;
        }
    }
    for b in blocks.iter_mut() {
        b.phis.retain(|p| live.contains(&p.dst));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    fn br(t: u32) -> Terminator {
        Terminator::Br(BlockId(t))
    }
    fn cond(t: u32, f: u32) -> Terminator {
        Terminator::CondBr(Operand::Const(Const::Bool(true)), BlockId(t), BlockId(f))
    }
    fn shader(terms: Vec<Terminator>) -> Shader {
        Shader::compute([1, 1, 1], terms.into_iter().map(|t| Block::new(Vec::new(), t)).collect())
    }

    /// Every edge goes forward in the layout unless its target dominates its source.
    pub(crate) fn assert_layout_ok(s: &Shader) {
        let an = Analysis::new(&s.blocks);
        assert_eq!(an.rpo, (0..s.blocks.len()).collect::<Vec<_>>(), "the layout is the reverse postorder");
        for (u, b) in s.blocks.iter().enumerate() {
            for v in succs(&b.term) {
                assert!(v > u || an.dominates(v, u), "edge {u} to {v} goes back without being a back edge");
            }
        }
    }

    #[test]
    fn a_diamond_is_laid_out_in_reverse_postorder() {
        let mut s = shader(vec![cond(1, 2), br(3), br(3), Terminator::Ret]);
        normalize(&mut s).unwrap();
        assert_eq!(s.blocks.len(), 4);
        assert_layout_ok(&s);
    }

    #[test]
    fn an_irreducible_graph_is_refused() {
        // 0 branches to 1 and 2, which branch to each other.
        let mut s = shader(vec![cond(1, 2), cond(2, 3), cond(1, 3), Terminator::Ret]);
        assert!(matches!(normalize(&mut s), Err(Error::Irreducible(_))));
    }

    #[test]
    fn unreachable_blocks_are_dropped() {
        let mut s = shader(vec![Terminator::Ret, br(0)]);
        normalize(&mut s).unwrap();
        assert_eq!(s.blocks.len(), 1);
    }

    #[test]
    fn a_loop_gets_a_preheader_and_dedicated_exits_and_the_latch_is_folded() {
        // 0 branches to 1 and 2, both go to the header 3. The header exits to 6 or enters the body 4,
        // which goes through the empty latch 5 back to 3.
        let mut s = shader(vec![cond(1, 2), br(3), br(3), cond(4, 6), br(5), br(3), Terminator::Ret]);
        s.blocks[4].insts.push(Inst::new(&[], Op::Barrier { flags: 1 }));
        normalize(&mut s).unwrap();
        assert_layout_ok(&s);
        // a preheader joins blocks 1 and 2, and the empty latch is folded into the body.
        assert_eq!(s.blocks.len(), 7);
        let body = s.blocks.iter().position(|b| !b.insts.is_empty()).unwrap();
        let Terminator::Br(back) = s.blocks[body].term else { panic!("the body ends in a branch") };
        assert!((back.0 as usize) < body, "the body branches straight back to the header");
    }

    #[test]
    fn critical_edges_are_split() {
        // 0 -> 1, 2; 1 -> 2, 3; 2 -> 3. Edges 0->2 and 1->3 are critical.
        let mut s = shader(vec![cond(1, 2), cond(2, 3), br(3), Terminator::Ret]);
        normalize(&mut s).unwrap();
        assert_layout_ok(&s);
        assert_eq!(s.blocks.len(), 7, "three critical edges: 0 to 2, 1 to 2 and 1 to 3");
    }

    #[test]
    fn dead_phis_are_removed_through_chains() {
        let mut s = shader(vec![br(1), cond(1, 2), Terminator::Ret]);
        let c = Operand::Const(Const::I32(0));
        s.blocks[1].phis.push(Phi { dst: ValueId(0), ty: Ty::I32, incoming: vec![(BlockId(0), c), (BlockId(1), Operand::Value(ValueId(1)))] });
        s.blocks[1].phis.push(Phi { dst: ValueId(1), ty: Ty::I32, incoming: vec![(BlockId(0), c), (BlockId(1), Operand::Value(ValueId(0)))] });
        normalize(&mut s).unwrap();
        assert!(s.blocks.iter().all(|b| b.phis.is_empty()));
    }
}
