//! Randomised structured control flow (nested ifs and loops with loop carried phis): the layout
//! never needs a forward reference outside phis, and every shader validates (skipped without dxil.dll).

mod common;

use decant_dxil::cfg;
use decant_dxil::ir::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as u32
    }
}

struct Gen {
    rng: Rng,
    blocks: Vec<Block>,
    next_value: u32,
}

fn v(n: u32) -> Operand {
    Operand::value(n)
}

impl Gen {
    fn value(&mut self) -> u32 {
        self.next_value += 1;
        self.next_value - 1
    }

    fn new_block(&mut self) -> usize {
        self.blocks.push(Block::new(vec![], Terminator::Ret));
        self.blocks.len() - 1
    }

    /// A fresh `Bool` in `block`, from the constant buffer (so that nothing is known at compile time).
    fn condition(&mut self, block: usize) -> Operand {
        let (load, cmp) = (self.value(), self.value());
        let row = self.rng.next(4);
        let k = self.rng.next(5);
        self.blocks[block].insts.push(Inst::new(&[load], Op::CbufLoad { res: ResRef::bound(0), row: Operand::i32(row), comps: vec![0], ty: Ty::I32 }));
        let pred = if self.rng.next(2) == 0 { IPred::Ult } else { IPred::Ne };
        self.blocks[block].insts.push(Inst::new(&[cmp], Op::ICmp(pred, v(load), Operand::i32(k))));
        v(cmp)
    }

    fn store(&mut self, block: usize, value: Operand) {
        let at = self.rng.next(8) * 4;
        self.blocks[block].insts.push(Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: Operand::i32(at), offset: None, values: vec![value] }));
    }

    /// Appends random code to `cur` and returns the block where control continues.
    fn code(&mut self, mut cur: usize, depth: u32) -> usize {
        for _ in 0..1 + self.rng.next(3) {
            match if depth == 0 { 0 } else { self.rng.next(4) } {
                0 => self.store(cur, Operand::i32(self.rng.0 as u32)),
                1 => {
                    // if, with or without else
                    let c = self.condition(cur);
                    let (then_b, join) = (self.new_block(), self.new_block());
                    let else_b = (self.rng.next(2) == 0).then(|| self.new_block());
                    self.blocks[cur].term = Terminator::CondBr(c, BlockId(then_b as u32), BlockId(else_b.unwrap_or(join) as u32));
                    for b in [Some(then_b), else_b].into_iter().flatten() {
                        let end = self.code(b, depth - 1);
                        self.blocks[end].term = Terminator::Br(BlockId(join as u32));
                    }
                    cur = join;
                }
                _ => {
                    // a loop with a counter phi, either tested at the top or at the bottom
                    let (header, exit) = (self.new_block(), self.new_block());
                    let (i, inc) = (self.value(), self.value());
                    let limit = 1 + self.rng.next(4);
                    self.blocks[cur].term = Terminator::Br(BlockId(header as u32));
                    let top_tested = self.rng.next(2) == 0;
                    let latch;
                    if top_tested {
                        let done = self.value();
                        self.blocks[header].insts.push(Inst::new(&[done], Op::ICmp(IPred::Eq, v(i), Operand::i32(limit))));
                        let body = self.new_block();
                        self.blocks[header].term = Terminator::CondBr(v(done), BlockId(exit as u32), BlockId(body as u32));
                        latch = self.code(body, depth - 1);
                        self.blocks[latch].insts.push(Inst::new(&[inc], Op::Bin(BinOp::Add, v(i), Operand::i32(1))));
                        self.blocks[latch].term = Terminator::Br(BlockId(header as u32));
                    } else {
                        latch = self.code(header, depth - 1);
                        let done = self.value();
                        self.blocks[latch].insts.push(Inst::new(&[inc], Op::Bin(BinOp::Add, v(i), Operand::i32(1))));
                        self.blocks[latch].insts.push(Inst::new(&[done], Op::ICmp(IPred::Eq, v(inc), Operand::i32(limit))));
                        self.blocks[latch].term = Terminator::CondBr(v(done), BlockId(exit as u32), BlockId(header as u32));
                    }
                    self.store(latch, v(i));
                    self.blocks[header].phis.push(Phi { dst: ValueId(i), ty: Ty::I32, incoming: vec![(BlockId(cur as u32), Operand::i32(0)), (BlockId(latch as u32), v(inc))] });
                    cur = exit;
                }
            }
        }
        cur
    }
}

fn random_shader(seed: u64) -> Shader {
    let mut g = Gen { rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF), blocks: Vec::new(), next_value: 0 };
    let entry = g.new_block();
    let end = g.code(entry, 3);
    g.blocks[end].term = Terminator::Ret;
    let mut s = Shader::compute([1, 1, 1], g.blocks);
    s.resources.push(Resource::cbuffer(0, 0, 64));
    s.resources.push(Resource::raw(ResourceClass::Uav, 0, 0));
    s
}

/// Dominators by the plain set equations, for the check.
fn dominators(blocks: &[Block]) -> Vec<Vec<bool>> {
    let n = blocks.len();
    let succ = |b: &Block| match &b.term {
        Terminator::Br(t) => vec![t.0 as usize],
        Terminator::CondBr(_, a, c) => vec![a.0 as usize, c.0 as usize],
        Terminator::Ret => vec![],
    };
    let mut preds = vec![vec![]; n];
    for (i, b) in blocks.iter().enumerate() {
        for s in succ(b) {
            preds[s].push(i);
        }
    }
    let mut dom = vec![vec![true; n]; n];
    dom[0] = (0..n).map(|j| j == 0).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..n {
            let mut new = vec![true; n];
            for &p in &preds[b] {
                for j in 0..n {
                    new[j] &= dom[p][j];
                }
            }
            new[b] = true;
            if new != dom[b] {
                dom[b] = new;
                changed = true;
            }
        }
    }
    dom
}

#[test]
fn random_structured_graphs_lay_out_without_forward_references_and_validate() {
    let mut with_loops = 0;
    for seed in 0..300u64 {
        let shader = random_shader(seed);
        let mut normalized = shader.clone();
        cfg::normalize(&mut normalized).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let dom = dominators(&normalized.blocks);
        let mut back_edges = 0;
        for (u, b) in normalized.blocks.iter().enumerate() {
            let targets = match &b.term {
                Terminator::Br(t) => vec![t.0 as usize],
                Terminator::CondBr(_, a, c) => vec![a.0 as usize, c.0 as usize],
                Terminator::Ret => vec![],
            };
            for t in targets {
                if t <= u {
                    assert!(dom[u][t], "seed {seed}: the edge {u} to {t} goes back, but {t} does not dominate {u}");
                    back_edges += 1;
                }
            }
        }
        with_loops += (back_edges > 0) as u32;
        // Every non phi operand is defined earlier in layout order: the writer refuses forward references.
        let out = decant_dxil::write(&shader).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        if seed < 40 {
            common::check_validator(&format!("random seed {seed}"), &out.container);
        }
    }
    assert!(with_loops > 50, "the generator makes loops ({with_loops} of 300 shaders)");
}
