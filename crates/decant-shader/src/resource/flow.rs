//! The flow of a structured function as a flat graph, for the descriptor dataflow.
//!
//! The resource pass must know what a scalar register holds at every use, whatever nesting of
//! ifs and loops the structure pass gave the program. This file lowers the [`Node`] tree to
//! blocks with successor edges. Every statement list of the tree is one block, in the order
//! [`crate::ir::Function::for_each_block_mut`] visits them, and joins and loop heads are
//! blocks without statements.

use crate::ir::{Expr, Node, Stmt};

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

struct Builder<'a> {
    blocks: Vec<Block<'a>>,
    /// The blocks a `Break` of the innermost loop leaves from, per open loop.
    breaks: Vec<Vec<usize>>,
    /// The head block of each open loop.
    heads: Vec<usize>,
    guarded: bool,
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
    fn seq(&mut self, nodes: &'a mut [Node], mut tails: Vec<usize>) -> Vec<usize> {
        for node in nodes {
            match node {
                Node::Block(stmts) => {
                    let b = self.block(Some(stmts));
                    self.edges(&tails, b);
                    tails = vec![b];
                }
                Node::If { cond, then, els } => {
                    let (a, b) = if self.guarded {
                        let a = self.block(None);
                        let b = self.block(None);
                        self.blocks[a].condition = Some((cond, true));
                        self.blocks[b].condition = Some((cond, false));
                        self.edges(&tails, a);
                        self.edges(&tails, b);
                        (vec![a], vec![b])
                    } else {
                        (tails.clone(), tails)
                    };
                    let mut out = self.seq(then, a);
                    out.extend(self.seq(els, b));
                    out.sort_unstable();
                    out.dedup();
                    tails = out;
                }
                Node::Loop(body) => {
                    let head = self.block(None);
                    self.edges(&tails, head);
                    self.heads.push(head);
                    self.breaks.push(Vec::new());
                    let end = self.seq(body, vec![head]);
                    self.edges(&end, head);
                    self.heads.pop();
                    tails = self.breaks.pop().unwrap_or_default();
                }
                Node::Once(body) => {
                    // No way back: control goes on from every `Break` and from the end.
                    self.breaks.push(Vec::new());
                    let mut out = self.seq(body, tails);
                    out.extend(self.breaks.pop().unwrap_or_default());
                    out.sort_unstable();
                    out.dedup();
                    tails = out;
                }
                Node::Break => {
                    if let Some(open) = self.breaks.last_mut() {
                        open.append(&mut tails);
                    }
                    tails = Vec::new();
                }
                Node::Continue => {
                    if let Some(&head) = self.heads.last() {
                        self.edges(&tails, head);
                    }
                    tails = Vec::new();
                }
                Node::Return => tails = Vec::new(),
            }
        }
        tails
    }
}

impl<'a> Flow<'a> {
    pub fn build(body: &'a mut [Node], guarded: bool) -> Flow<'a> {
        let mut b = Builder {
            blocks: Vec::new(),
            breaks: Vec::new(),
            heads: Vec::new(),
            guarded,
        };
        let entry = b.block(None);
        b.seq(body, vec![entry]);
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
