//! The native control flow graph: the instructions the program can reach from its entry, cut
//! into basic blocks.
//!
//! Decoding follows the flow from the first word after the signature, so data after `s_endpgm`
//! is never decoded and code there only when a jump leads to it.
//!
//! A branch reaches 32,767 words. For a target further away compilers write a far jump:
//!
//! ```text
//! s_getpc_b64 sA              ; the byte address of the next instruction
//! s_add_u32   sA, sA, bytes   ; or s_sub_u32 for a jump back
//! s_addc_u32  sA+1, sA+1, 0   ; or s_subb_u32
//! s_setpc_b64 sA
//! ```
//!
//! usually behind a short branch with the opposite condition that jumps over it. The four
//! instructions are folded into one jump of the graph and are not part of any block: the
//! address they leave in the register pair is never read. `s_getpc_b64` alone stays an
//! instruction. Programs use it to address constant data stored behind their code.

use std::collections::{BTreeMap, BTreeSet};

use crate::container::{SIGNATURE_WORDS, has_signature};
use crate::decode::{Ext, Insn, Operand, Reg, decode_one};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::{Class, Op};

/// What a conditional branch tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// The scalar condition code is set (`true`) or clear.
    Scc(bool),
    /// VCC is zero (`true`) or not.
    VccZero(bool),
    /// EXEC is zero (`true`) or not.
    ExecZero(bool),
}

/// How a block ends. Targets are block indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    /// `s_endpgm`.
    End,
    /// Falls or jumps to one block.
    Jump(usize),
    Branch {
        cond: Cond,
        taken: usize,
        fall: usize,
    },
}

/// A basic block: a run of instructions with one entry and one exit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// Word index of the first instruction.
    pub start: u32,
    /// The instructions without the branch or `s_endpgm` that ends the block.
    pub insns: Vec<Insn>,
    pub term: Term,
}

/// The graph. Block 0 is the entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cfg {
    pub blocks: Vec<Block>,
}

impl Cfg {
    /// Every instruction, block by block.
    pub fn insns(&self) -> impl Iterator<Item = &Insn> {
        self.blocks.iter().flat_map(|b| b.insns.iter())
    }

    /// The blocks that lead to each block.
    pub fn predecessors(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.blocks.len()];
        for (i, b) in self.blocks.iter().enumerate() {
            match b.term {
                Term::End => {}
                Term::Jump(t) => out[t].push(i),
                Term::Branch { taken, fall, .. } => {
                    out[taken].push(i);
                    if fall != taken {
                        out[fall].push(i);
                    }
                }
            }
        }
        out
    }
}

fn branch_cond(op: Op) -> Option<Option<Cond>> {
    Some(match op {
        Op::s_branch => None,
        Op::s_cbranch_scc0 => Some(Cond::Scc(false)),
        Op::s_cbranch_scc1 => Some(Cond::Scc(true)),
        Op::s_cbranch_vccz => Some(Cond::VccZero(true)),
        Op::s_cbranch_vccnz => Some(Cond::VccZero(false)),
        Op::s_cbranch_execz => Some(Cond::ExecZero(true)),
        Op::s_cbranch_execnz => Some(Cond::ExecZero(false)),
        _ => return None,
    })
}

/// The target word of the far jump that starts at `pc`, when the four instructions there are one.
fn far_jump(code: &[u32], pc: u32) -> Option<u32> {
    let getpc = decode_one(code, pc).ok()?;
    let Operand::Reg(Reg::S(a)) = getpc.dst else {
        return None;
    };
    let lo = Operand::Reg(Reg::S(a));
    let hi = Operand::Reg(Reg::S(a.checked_add(1)?));
    let after = pc + u32::from(getpc.len);
    let add = decode_one(code, after).ok()?;
    let carry = decode_one(code, after + u32::from(add.len)).ok()?;
    let setpc = decode_one(code, after + u32::from(add.len) + u32::from(carry.len)).ok()?;
    let forward = match (add.op, carry.op) {
        (Op::s_add_u32, Op::s_addc_u32) => true,
        (Op::s_sub_u32, Op::s_subb_u32) => false,
        _ => return None,
    };
    let Operand::Lit(bytes) = add.src[1] else {
        return None;
    };
    let shape = getpc.op == Op::s_getpc_b64
        && (add.dst, add.src[0]) == (lo, lo)
        && (carry.dst, carry.src[0], carry.src[1]) == (hi, hi, Operand::Int(0))
        && (setpc.op, setpc.src[0]) == (Op::s_setpc_b64, lo);
    // The program counter counts bytes from wherever the code is loaded, so only the distance
    // matters. It has to be a whole number of words.
    if !shape || bytes % 4 != 0 {
        return None;
    }
    if forward {
        after.checked_add(bytes / 4)
    } else {
        after.checked_sub(bytes / 4)
    }
}

/// Builds the graph of `code` (the words of a `StageProgram`, signature included).
pub fn build(code: &[u32]) -> Result<Cfg, Refusal> {
    if !has_signature(code) {
        return Err(Refusal::new(RefusalKind::Container("no-signature"), ""));
    }
    let entry = SIGNATURE_WORDS as u32;
    // Pass 1: decode everything reachable and note where blocks start.
    let mut insns: BTreeMap<u32, Insn> = BTreeMap::new();
    let mut leaders: BTreeSet<u32> = BTreeSet::from([entry]);
    // Far jumps by the word of their first instruction.
    let mut far: BTreeMap<u32, u32> = BTreeMap::new();
    let mut work = vec![entry];
    while let Some(mut pc) = work.pop() {
        while !insns.contains_key(&pc) {
            let insn = decode_one(code, pc)
                .map_err(|e| Refusal::new(RefusalKind::Decode, e.to_string()).at(pc))?;
            insns.insert(pc, insn);
            let next = pc + u32::from(insn.len);
            if insn.op == Op::s_endpgm {
                break;
            }
            if insn.op == Op::s_getpc_b64
                && let Some(target) = far_jump(code, pc)
            {
                far.insert(pc, target);
                leaders.insert(target);
                work.push(target);
                break;
            }
            if let Some(cond) = branch_cond(insn.op) {
                let target = insn.branch_target().expect("a branch has a target");
                leaders.insert(target);
                work.push(target);
                if cond.is_none() {
                    break;
                }
                leaders.insert(next);
            } else if insn.op.info().class == Class::Branch
                || matches!(insn.op, Op::s_setpc_b64 | Op::s_call_b64 | Op::s_rfe_b64)
            {
                // `s_swappc_b64` at the entry of a vertex program calls the routine that fetches
                // the vertex inputs and comes back. The vertex stage accounts for it. Any other
                // computed jump that is not a far jump has a target only the machine knows.
                if !(insn.op == Op::s_swappc_b64 && pc == entry) {
                    return Err(Refusal::new(
                        RefusalKind::ControlFlow("indirect-jump"),
                        insn.to_string(),
                    )
                    .at(pc));
                }
            }
            pc = next;
        }
    }
    // Pass 2: cut into blocks.
    let starts: Vec<u32> = leaders
        .iter()
        .copied()
        .filter(|pc| insns.contains_key(pc))
        .collect();
    let index_of = |pc: u32| {
        starts.binary_search(&pc).map_err(|_| {
            Refusal::new(RefusalKind::ControlFlow("branch-into-instruction"), "").at(pc)
        })
    };
    let mut blocks = Vec::with_capacity(starts.len());
    for &start in &starts {
        let mut body = Vec::new();
        let mut pc = start;
        let term = loop {
            let insn = insns[&pc];
            let next = pc + u32::from(insn.len);
            if insn.op == Op::s_endpgm {
                break Term::End;
            }
            if let Some(&target) = far.get(&pc) {
                break Term::Jump(index_of(target)?);
            }
            if let Some(cond) = branch_cond(insn.op) {
                let taken = index_of(insn.branch_target().expect("a branch has a target"))?;
                break match cond {
                    None => Term::Jump(taken),
                    Some(cond) => Term::Branch {
                        cond,
                        taken,
                        fall: index_of(next)?,
                    },
                };
            }
            body.push(insn);
            if leaders.contains(&next) {
                break Term::Jump(index_of(next)?);
            }
            pc = next;
        };
        blocks.push(Block {
            start,
            insns: body,
            term,
        });
    }
    Ok(Cfg { blocks })
}

/// The 16 bit immediate of an instruction, when it has one.
pub fn imm16(insn: &Insn) -> Option<u16> {
    match insn.ext {
        Ext::Imm16(v) => Some(v),
        _ => None,
    }
}
