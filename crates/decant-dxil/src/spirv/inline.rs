//! Inlines every `OpFunctionCall` of the entry function at the SPIR-V level, before the
//! translation: DXIL entries have one function and no calls.

use std::collections::HashMap;

use rspirv::dr::{self, Block, Instruction, Operand as SOp};
use rspirv::spirv::{Op, Word};

use super::{ImportError, R, fail};

/// How many calls are expanded at most (a guard against recursion).
const MAX_EXPANSIONS: usize = 200_000;

fn remap(i: &Instruction, map: &HashMap<Word, Word>) -> Instruction {
    let mut c = i.clone();
    if let Some(r) = c.result_id {
        c.result_id = Some(*map.get(&r).unwrap_or(&r));
    }
    for o in &mut c.operands {
        match o {
            SOp::IdRef(w) | SOp::IdScope(w) | SOp::IdMemorySemantics(w) => {
                if let Some(n) = map.get(w) {
                    *w = *n;
                }
            }
            _ => {}
        }
    }
    c
}

fn label_of(b: &Block) -> Word {
    b.label.as_ref().and_then(|l| l.result_id).unwrap_or(0)
}

/// Expands the calls of the function `entry` in place.
pub(super) fn inline_calls(m: &mut dr::Module, entry: Word) -> R<()> {
    let Some(ef) = m.functions.iter().position(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(entry)) else {
        return fail("internal", "the entry function is missing");
    };
    // One forward pass: the block that was split ends in a branch, the copies of the callee and
    // the continuation follow it and are scanned next. Phis that named a split block as
    // predecessor are redirected once at the end (`moved` maps a block to the one holding its terminator now).
    let mut moved: HashMap<Word, Word> = HashMap::new();
    let mut bi = 0usize;
    let mut expansions = 0usize;
    while bi < m.functions[ef].blocks.len() {
        let Some(ii) = m.functions[ef].blocks[bi].instructions.iter().position(|i| i.class.opcode == Op::FunctionCall) else {
            bi += 1;
            continue;
        };
        expansions += 1;
        if expansions > MAX_EXPANSIONS {
            return Err(ImportError { category: "op:FunctionCall".into(), detail: "too many calls (recursion?)".into() });
        }
        let call = m.functions[ef].blocks[bi].instructions[ii].clone();
        let callee_id = match call.operands.first() {
            Some(SOp::IdRef(w)) => *w,
            _ => return fail("internal", "a call without a callee"),
        };
        if callee_id == entry {
            return fail("op:FunctionCall", "a recursive call");
        }
        let Some(callee) = m.functions.iter().find(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(callee_id)).cloned() else {
            return fail("op:FunctionCall", "a call of a function that is not defined");
        };
        let mut bound = m.header.as_ref().map(|h| h.bound).unwrap_or(1);
        let mut fresh = || {
            let w = bound;
            bound += 1;
            w
        };
        let mut map: HashMap<Word, Word> = HashMap::new();
        for (k, p) in callee.parameters.iter().enumerate() {
            let arg = match call.operands.get(1 + k) {
                Some(SOp::IdRef(w)) => *w,
                _ => return fail("internal", "a call with too few arguments"),
            };
            map.insert(p.result_id.unwrap(), arg);
        }
        for b in &callee.blocks {
            map.insert(label_of(b), fresh());
            for i in &b.instructions {
                if let Some(r) = i.result_id {
                    map.insert(r, fresh());
                }
            }
        }
        let cont = fresh();
        let old_label = label_of(&m.functions[ef].blocks[bi]);
        let mut returns: Vec<(Word, Word)> = Vec::new();
        let mut clones: Vec<Block> = Vec::new();
        for b in &callee.blocks {
            let mut nb = Block { label: b.label.as_ref().map(|l| remap(l, &map)), instructions: Vec::new() };
            let my_label = label_of(&nb);
            for i in &b.instructions {
                match i.class.opcode {
                    Op::Return => nb.instructions.push(Instruction::new(Op::Branch, None, None, vec![SOp::IdRef(cont)])),
                    Op::ReturnValue => {
                        let v = remap(i, &map);
                        let SOp::IdRef(val) = v.operands[0] else { return fail("internal", "a return value") };
                        returns.push((val, my_label));
                        nb.instructions.push(Instruction::new(Op::Branch, None, None, vec![SOp::IdRef(cont)]));
                    }
                    _ => nb.instructions.push(remap(i, &map)),
                }
            }
            clones.push(nb);
        }
        let Some(entry_clone) = clones.first().map(label_of) else { return fail("internal", "a function without blocks") };
        // The continuation: the result as a phi, then what followed the call.
        let blk = &mut m.functions[ef].blocks[bi];
        let post: Vec<Instruction> = blk.instructions.split_off(ii + 1);
        blk.instructions.pop(); // the call itself
        blk.instructions.push(Instruction::new(Op::Branch, None, None, vec![SOp::IdRef(entry_clone)]));
        let mut cont_block = Block { label: Some(Instruction::new(Op::Label, None, Some(cont), vec![])), instructions: Vec::new() };
        if let (Some(rt), Some(rid)) = (call.result_type, call.result_id) {
            let is_void = m.types_global_values.iter().any(|t| t.class.opcode == Op::TypeVoid && t.result_id == Some(rt));
            if !is_void {
                let mut ops = Vec::new();
                for (v, l) in &returns {
                    ops.push(SOp::IdRef(*v));
                    ops.push(SOp::IdRef(*l));
                }
                cont_block.instructions.push(Instruction::new(Op::Phi, Some(rt), Some(rid), ops));
            }
        }
        cont_block.instructions.extend(post);
        let f = &mut m.functions[ef];
        let mut k = bi + 1;
        for c in clones {
            f.blocks.insert(k, c);
            k += 1;
        }
        f.blocks.insert(k, cont_block);
        moved.insert(old_label, cont);
        if let Some(h) = m.header.as_mut() {
            h.bound = bound;
        }
    }
    if moved.is_empty() {
        return Ok(());
    }
    for b in &mut m.functions[ef].blocks {
        for i in &mut b.instructions {
            if i.class.opcode == Op::Phi {
                let mut n = 1;
                while n < i.operands.len() {
                    if let SOp::IdRef(w) = i.operands[n] {
                        let mut l = w;
                        while let Some(next) = moved.get(&l) {
                            l = *next;
                        }
                        i.operands[n] = SOp::IdRef(l);
                    }
                    n += 2;
                }
            }
        }
    }
    Ok(())
}
