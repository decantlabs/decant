//! A check of the structure pass on one graph: the native graph and the structured body are
//! run side by side with the same answer to every branch, and the order in which the blocks
//! run has to be the same.
//!
//! The tests use it on graphs made at random, and a census can use it on a whole corpus,
//! because it needs the graph only and no lifter rule.

use std::collections::HashMap;

use crate::cfg::{Cfg, Cond, Term};
use crate::ir::{Bin, Expr, Guard, Node, Stmt, Un, Var};

/// The variable each block writes its number to, so the run of the body can be followed.
const MARK: Var = Var::Tmp(0);

/// The variable of the statements that stand for the instructions of a block.
const FILL: Var = Var::Tmp(1);

/// The answer to the branch of `block` when it is reached for the `visit`th time: a hash of
/// both and the seed, so both runs get the same answers without sharing state.
fn taken(seed: u64, block: usize, visit: u32) -> bool {
    let mut x = seed ^ ((block as u64) << 32 | u64::from(visit));
    x = (x ^ (x >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    x = (x ^ (x >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    (x ^ (x >> 33)) & 1 != 0
}

/// The blocks the native graph runs, at most `steps` of them, and whether the program ended.
fn native(cfg: &Cfg, seed: u64, steps: usize) -> (Vec<usize>, bool) {
    let mut visits = vec![0u32; cfg.blocks.len()];
    let mut trace = Vec::new();
    let mut at = 0;
    while trace.len() < steps {
        trace.push(at);
        visits[at] += 1;
        at = match cfg.blocks[at].term {
            Term::End => return (trace, true),
            Term::Jump(t) => t,
            Term::Branch { taken: t, fall, .. } => {
                if taken(seed, at, visits[at]) {
                    t
                } else {
                    fall
                }
            }
        };
    }
    (trace, false)
}

enum Flow {
    On,
    Break,
    Continue,
    Return,
    /// Enough blocks have run.
    Stop,
}

struct Run<'a> {
    cfg: &'a Cfg,
    seed: u64,
    steps: usize,
    visits: Vec<u32>,
    trace: Vec<usize>,
    vars: HashMap<Var, u32>,
}

impl Run<'_> {
    fn eval(&self, e: &Expr) -> Result<u32, String> {
        Ok(match e {
            Expr::Bool(b) => u32::from(*b),
            Expr::U32(v) => *v,
            Expr::Var(v @ (Var::Scc | Var::Vcc | Var::Exec)) => {
                // The register value that makes the branch of the block that ran last go the
                // way the answer says.
                let at = *self
                    .trace
                    .last()
                    .ok_or("a condition before the first block")?;
                let Term::Branch { cond, .. } = self.cfg.blocks[at].term else {
                    return Err(format!("a condition after block {at}, which has no branch"));
                };
                let go = taken(self.seed, at, self.visits[at]);
                let (var, value) = match cond {
                    Cond::Scc(set) => (Var::Scc, go == set),
                    Cond::VccZero(zero) => (Var::Vcc, go != zero),
                    Cond::ExecZero(zero) => (Var::Exec, go != zero),
                };
                if var != *v {
                    return Err(format!(
                        "the condition of block {at} reads another register"
                    ));
                }
                u32::from(value)
            }
            Expr::Var(v) => self.vars.get(v).copied().unwrap_or(0),
            Expr::Un(Un::LNot, a) => u32::from(self.eval(a)? == 0),
            Expr::Bin(Bin::Eq, a, b) => u32::from(self.eval(a)? == self.eval(b)?),
            other => return Err(format!("an expression the check does not know: {other:?}")),
        })
    }

    fn nodes(&mut self, nodes: &[Node]) -> Result<Flow, String> {
        for n in nodes {
            let flow = match n {
                Node::Block(stmts) => {
                    for s in stmts {
                        let Stmt::Set {
                            dst,
                            value,
                            guard: Guard::Always,
                        } = s
                        else {
                            return Err(format!("a statement the check does not know: {s:?}"));
                        };
                        let value = self.eval(value)?;
                        if *dst == MARK {
                            if self.trace.len() == self.steps {
                                return Ok(Flow::Stop);
                            }
                            self.trace.push(value as usize);
                            self.visits[value as usize] += 1;
                        } else {
                            self.vars.insert(*dst, value);
                        }
                    }
                    Flow::On
                }
                Node::If { cond, then, els } => {
                    if self.eval(cond)? != 0 {
                        self.nodes(then)?
                    } else {
                        self.nodes(els)?
                    }
                }
                Node::Loop(body) => loop {
                    match self.nodes(body)? {
                        Flow::On | Flow::Continue => {}
                        Flow::Break => break Flow::On,
                        other => break other,
                    }
                },
                Node::Once(body) => match self.nodes(body)? {
                    Flow::On | Flow::Break => Flow::On,
                    Flow::Continue => {
                        return Err("a continue directly in a block that runs once".to_owned());
                    }
                    other => other,
                },
                Node::Break => Flow::Break,
                Node::Continue => Flow::Continue,
                Node::Return => Flow::Return,
            };
            if !matches!(flow, Flow::On) {
                return Ok(flow);
            }
        }
        Ok(Flow::On)
    }
}

/// Structures `cfg` with a marker in every block and compares one run of at most `steps`
/// blocks, with branch answers drawn from `seed`, against the native graph. Returns the
/// structured body, or what differs.
///
/// Every block must be reachable from the first, as in a graph from `cfg::build`. Besides the
/// marker a block gets one statement per instruction, so that the pass sees blocks of the
/// size the lifter would give it (it copies a short end of the program).
pub fn same_run(cfg: &Cfg, seed: u64, steps: usize) -> Result<Vec<Node>, String> {
    let set = |dst: Var, value: u32| Stmt::Set {
        dst,
        value: Expr::U32(value),
        guard: Guard::Always,
    };
    let marks = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(i, block)| {
            let mut stmts = vec![set(MARK, i as u32)];
            stmts.extend(block.insns.iter().map(|_| set(FILL, 0)));
            stmts
        })
        .collect();
    let body = super::build(cfg, marks).map_err(|r| r.to_string())?;
    let (want, ended) = native(cfg, seed, steps);
    let mut run = Run {
        cfg,
        seed,
        steps,
        visits: vec![0; cfg.blocks.len()],
        trace: Vec::new(),
        vars: HashMap::new(),
    };
    let end = run.nodes(&body)?;
    match end {
        Flow::Return if ended => {}
        Flow::Stop if !ended => {}
        Flow::Return => return Err("the body returns where the graph goes on".to_owned()),
        Flow::Stop => return Err("the body goes on where the graph ends".to_owned()),
        _ => return Err("the body runs out of its text".to_owned()),
    }
    if run.trace != want {
        let at = run
            .trace
            .iter()
            .zip(&want)
            .take_while(|(a, b)| a == b)
            .count();
        return Err(format!(
            "after {at} blocks the graph runs block {:?} and the body block {:?}",
            want.get(at),
            run.trace.get(at)
        ));
    }
    Ok(body)
}
