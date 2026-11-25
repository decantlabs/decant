//! Control flow: the native graph (with far jumps folded) and the structured text made of it.
//!
//! The shape tests write a graph by hand and compare the text. The random tests run the graph
//! and the text side by side (`structure::check`) and compile whole programs with DXC. All
//! programs and graphs are made here. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::cfg::{self, Block, Cfg, Cond, Term};
use decant_shader::decode::Operand;
use decant_shader::ir::{Bin, Expr, Node, Stmt, Un, Var};
use decant_shader::structure::{self, FIRST_FLAG, FIRST_SELECTOR, Shape, check};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

const SCC: Cond = Cond::Scc(true);

fn jump(to: usize) -> Term {
    Term::Jump(to)
}

fn branch(taken: usize, fall: usize) -> Term {
    Term::Branch {
        cond: SCC,
        taken,
        fall,
    }
}

fn graph(terms: &[Term]) -> Cfg {
    Cfg {
        blocks: terms
            .iter()
            .enumerate()
            .map(|(i, &term)| Block {
                start: i as u32,
                insns: Vec::new(),
                term,
            })
            .collect(),
    }
}

fn cond(e: &Expr) -> String {
    match e {
        Expr::Var(Var::Scc) => "scc".to_owned(),
        Expr::Var(Var::Flag(n)) => format!("f{}", FIRST_FLAG - n),
        Expr::Un(Un::LNot, a) => format!("!{}", cond(a)),
        Expr::Bin(Bin::Eq, a, b) => match (&**a, &**b) {
            (Expr::Var(Var::Tmp(n)), Expr::U32(k)) => format!("t{} == {k}", FIRST_SELECTOR - n),
            _ => panic!("{e:?}"),
        },
        _ => panic!("{e:?}"),
    }
}

/// The body on one line: `B3` is native block 3, `f0` the first boolean of the pass, `t0` its
/// first selector.
fn text(nodes: &[Node]) -> String {
    let mut out = Vec::new();
    for n in nodes {
        out.push(match n {
            Node::Block(stmts) => stmts
                .iter()
                // The statements that stand for the instructions of a block are left out.
                .filter(|s| {
                    !matches!(
                        s,
                        Stmt::Set {
                            dst: Var::Tmp(1),
                            ..
                        }
                    )
                })
                .map(|s| match s {
                    Stmt::Set {
                        dst: Var::Tmp(0),
                        value: Expr::U32(b),
                        ..
                    } => format!("B{b}"),
                    Stmt::Set {
                        dst: Var::Flag(n),
                        value: Expr::Bool(v),
                        ..
                    } => format!("f{} = {v}", FIRST_FLAG - n),
                    Stmt::Set {
                        dst: Var::Tmp(n),
                        value: Expr::U32(v),
                        ..
                    } => format!("t{} = {v}", FIRST_SELECTOR - n),
                    other => panic!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" "),
            Node::If { cond: c, then, els } if els.is_empty() => {
                format!("if {} {{ {} }}", cond(c), text(then))
            }
            Node::If { cond: c, then, els } => {
                format!(
                    "if {} {{ {} }} else {{ {} }}",
                    cond(c),
                    text(then),
                    text(els)
                )
            }
            Node::Loop(body) => format!("loop {{ {} }}", text(body)),
            Node::Once(body) => format!("once {{ {} }}", text(body)),
            Node::Break => "break".to_owned(),
            Node::Continue => "continue".to_owned(),
            Node::Return => "return".to_owned(),
        });
    }
    out.join(" ")
}

/// Structures the graph, checks it against the native run for some seeds and gives the text.
fn structured(terms: &[Term]) -> String {
    let g = graph(terms);
    let mut body = Vec::new();
    for seed in 0..16 {
        body = check::same_run(&g, seed, 500).unwrap_or_else(|why| panic!("seed {seed}: {why}"));
    }
    text(&body)
}

#[test]
fn an_if_is_an_if() {
    // Without else: the branch skips block 1.
    assert_eq!(
        structured(&[branch(2, 1), jump(2), Term::End]),
        "B0 if !scc { B1 } B2 return"
    );
    // With else. The side that comes first in the native code comes first in the text.
    assert_eq!(
        structured(&[branch(2, 1), jump(3), jump(3), Term::End]),
        "B0 if !scc { B1 } else { B2 } B3 return"
    );
    // Two conditions with one else: `if (a && b) B2 else B3`, then block 4 and more. Block 3
    // is jumped to from two depths, so what comes before it is a block that runs once. No
    // boolean is needed.
    assert_eq!(
        structured(&[
            branch(3, 1),
            branch(3, 2),
            jump(4),
            jump(4),
            branch(6, 5),
            jump(6),
            Term::End
        ]),
        "B0 once { if !scc { B1 if !scc { B2 break } } B3 } B4 if !scc { B5 } B6 return"
    );
}

#[test]
fn a_short_end_that_is_jumped_to_from_all_over_is_copied() {
    // The shape of a pixel program that leaves early: blocks 0 and 2 branch to block 5, the
    // end that discards. Block 4 is the normal end.
    assert_eq!(
        structured(&[
            branch(5, 1),
            branch(4, 2),
            branch(5, 3),
            jump(4),
            Term::End,
            Term::End
        ]),
        "B0 if scc { B5 return } B1 if !scc { B2 if scc { B5 return } B3 } B4 return"
    );
    // The same end after the two conditions of the test above: it is written twice.
    assert_eq!(
        structured(&[branch(3, 1), branch(3, 2), jump(4), jump(4), Term::End]),
        "B0 if !scc { B1 if !scc { B2 B4 return } } B3 B4 return"
    );
    // A long end is not copied: it stays behind a block that runs once.
    let g = graph(&[branch(3, 1), branch(3, 2), jump(4), jump(4), Term::End]);
    let mut lifted: Vec<Vec<Stmt>> = (0..5).map(|_| Vec::new()).collect();
    lifted[4] = vec![Stmt::Comment(String::new()); 3];
    lifted[4].extend(std::iter::repeat_n(
        Stmt::Discard {
            cond: Expr::Bool(false),
        },
        70,
    ));
    let body = structure::build(&g, lifted).unwrap();
    assert!(
        matches!(body[..], [Node::Once(_), Node::Block(_), Node::Return]),
        "{body:?}"
    );
}

#[test]
fn a_loop_is_a_loop_with_break_and_continue() {
    // while (cond) body: 1 is the header, 2 the body, 3 what follows.
    assert_eq!(
        structured(&[jump(1), branch(3, 2), jump(1), Term::End]),
        "B0 loop { B1 if scc { break } B2 } B3 return"
    );
    // The test at the end: do body while (cond).
    assert_eq!(
        structured(&[jump(1), branch(1, 2), Term::End]),
        "B0 loop { B1 if !scc { break } } B2 return"
    );
    // A break from the body with code of its own (block 4), and a continue (2 to 1).
    assert_eq!(
        structured(&[
            jump(1),
            branch(5, 2),
            branch(1, 3),
            branch(4, 1),
            jump(5),
            Term::End
        ]),
        "B0 loop { B1 if scc { break } B2 if scc { continue } B3 if !scc { continue } B4 break } \
         B5 return"
    );
}

#[test]
fn a_jump_out_of_two_loops_sets_a_boolean() {
    // 1 heads the outer loop, 2 the inner. Block 3 leaves both to 5, block 4 closes both.
    assert_eq!(
        structured(&[
            jump(1),
            jump(2),
            branch(4, 3),
            branch(5, 2),
            branch(1, 5),
            Term::End
        ]),
        "B0 loop { B1 loop { B2 if scc { break } B3 if scc { f0 = true break } } \
         if f0 { f0 = false break } B4 if !scc { break } } B5 return"
    );
}

#[test]
fn a_loop_with_two_ways_in_gets_a_header_with_a_selector() {
    // Blocks 1 and 2 form a loop that block 0 enters at either. Every edge to 1 or 2 sets the
    // selector and goes to the new header, which is the test at the top of the loop.
    assert_eq!(
        structured(&[branch(2, 1), jump(2), branch(1, 3), Term::End]),
        "B0 if scc { t0 = 1 } else { t0 = 0 } loop { if t0 == 0 { B1 t0 = 1 continue } B2 \
         if !scc { break } t0 = 0 } B3 return"
    );
}

#[test]
fn joins_that_are_jumped_to_from_all_over_stand_in_a_row() {
    // 0 goes to 1 or 2, 1 to 3 or 4, 2 to 3 or 4, 3 to 4: the joins 3 and 4 behind node 0.
    let body = structured(&[
        branch(2, 1),
        branch(4, 3),
        branch(4, 3),
        jump(4),
        branch(6, 5),
        jump(6),
        Term::End,
    ]);
    assert_eq!(
        body,
        "B0 once { if !scc { B1 if scc { break } } else { B2 if scc { break } } B3 } \
         B4 if !scc { B5 } B6 return"
    );
    // Many joins behind one node: every block from 2 on is reached from block 0 and block 1.
    let mut terms = vec![branch(2, 1), branch(4, 3)];
    for i in 2..40 {
        terms.push(if i % 3 == 0 {
            branch(i + 3, i + 1)
        } else {
            branch(i + 2, i + 1)
        });
    }
    terms.extend([jump(41), jump(42), jump(43), Term::End]);
    let g = graph(&terms);
    let body = check::same_run(&g, 5, 500).unwrap();
    fn depth(nodes: &[Node]) -> usize {
        nodes
            .iter()
            .map(|n| match n {
                Node::If { then, els, .. } => 1 + depth(then).max(depth(els)),
                Node::Loop(body) | Node::Once(body) => 1 + depth(body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
    assert!(depth(&body) <= 6, "{}", text(&body));
}

/// A small generator of numbers: tests must not depend on a crate for it.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A graph with up to `max` blocks and exits chosen at random, without the blocks that the
/// entry does not reach.
fn random_graph(r: &mut Random, max: usize) -> Cfg {
    let n = 1 + r.below(max);
    let conds = [
        Cond::Scc(true),
        Cond::Scc(false),
        Cond::VccZero(true),
        Cond::VccZero(false),
        Cond::ExecZero(true),
        Cond::ExecZero(false),
    ];
    let mut terms: Vec<Term> = (0..n)
        .map(|i| match r.below(10) {
            0 => Term::End,
            1..=3 => Term::Jump(r.below(n)),
            // Mostly forward and near, as compiled code is.
            4..=6 => Term::Branch {
                cond: conds[r.below(6)],
                taken: (i + 1 + r.below(4)).min(n - 1),
                fall: (i + 1).min(n - 1),
            },
            _ => Term::Branch {
                cond: conds[r.below(6)],
                taken: r.below(n),
                fall: r.below(n),
            },
        })
        .collect();
    terms[n - 1] = Term::End;
    let mut number = vec![usize::MAX; n];
    let mut reached = vec![0usize];
    number[0] = 0;
    let mut at = 0;
    while at < reached.len() {
        let targets = match terms[reached[at]] {
            Term::End => vec![],
            Term::Jump(t) => vec![t],
            Term::Branch { taken, fall, .. } => vec![taken, fall],
        };
        for t in targets {
            if number[t] == usize::MAX {
                number[t] = reached.len();
                reached.push(t);
            }
        }
        at += 1;
    }
    let renumbered: Vec<Term> = reached
        .iter()
        .map(|&b| match terms[b] {
            Term::End => Term::End,
            Term::Jump(t) => Term::Jump(number[t]),
            Term::Branch { cond, taken, fall } => Term::Branch {
                cond,
                taken: number[taken],
                fall: number[fall],
            },
        })
        .collect();
    graph(&renumbered)
}

#[test]
fn random_graphs_run_the_same_blocks_in_the_same_order() {
    let mut r = Random(0x2545_f491_4f6c_dd1d);
    for round in 0..4000 {
        let g = random_graph(&mut r, if round % 50 == 0 { 300 } else { 24 });
        for seed in 0..4 {
            check::same_run(&g, seed, 2000)
                .unwrap_or_else(|why| panic!("round {round} seed {seed}: {why}\n{g:?}"));
        }
    }
}

/// `levels` ifs inside each other: header `i` skips to its join or goes on to header `i + 1`.
fn nested_ifs(levels: usize) -> Cfg {
    let join = |i: usize| 2 * levels - i;
    let mut terms: Vec<Term> = (0..levels).map(|i| branch(join(i), i + 1)).collect();
    terms.push(jump(join(levels - 1)));
    terms.extend((0..levels).rev().map(|i| match i {
        0 => Term::End,
        _ => jump(join(i - 1)),
    }));
    graph(&terms)
}

#[test]
fn a_graph_that_nests_deeper_than_the_compiler_reads_is_refused() {
    let empty = |g: &Cfg| vec![Vec::new(); g.blocks.len()];
    let deep = nested_ifs(structure::MAX_DEPTH + 1);
    let r = structure::build(&deep, empty(&deep)).unwrap_err();
    assert_eq!(r.key(), "cfg:depth");
    let fine = nested_ifs(structure::MAX_DEPTH);
    assert_eq!(
        structure::shape(&structure::build(&fine, empty(&fine)).unwrap()),
        Shape::Structured
    );
    check::same_run(&fine, 1, 1000).unwrap();
}

#[test]
fn a_far_jump_is_one_edge_of_the_graph() {
    // Forward: the four instructions at words 6 to 10 jump to word 13.
    let mut a = Asm::new();
    a.push(sop1(Op::s_getpc_b64, s(0), s(0)));
    a.push(sop2(Op::s_add_u32, s(0), s(0), lit(4 * (13 - 7))));
    a.push(sop2(Op::s_addc_u32, s(1), s(1), int(0)));
    a.push(sop1(Op::s_setpc_b64, s(0), s(0)));
    a.push(vop1(Op::v_mov_b32, v(1), float(1.0)));
    a.push(sopp(Op::s_endpgm, 0));
    a.push(vop1(Op::v_mov_b32, v(2), float(2.0)));
    a.push(sopp(Op::s_endpgm, 0));
    let g = cfg::build(&a.finish()).unwrap();
    assert_eq!(g.blocks.len(), 2);
    assert_eq!(
        (g.blocks[0].insns.len(), g.blocks[0].term),
        (0, Term::Jump(1))
    );
    assert_eq!((g.blocks[1].start, g.blocks[1].insns.len()), (13, 1));
    assert_eq!(g.blocks[1].term, Term::End);

    // Back, behind a branch that jumps over it: a loop whose header is far away.
    let mut a = Asm::new();
    let over = a.label();
    a.push(vop1(Op::v_mov_b32, v(1), float(1.0))); // word 6, the target
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0))); // 7
    a.branch(Op::s_cbranch_scc1, over); // 8
    a.push(sop1(Op::s_getpc_b64, s(4), s(0))); // 9, the next instruction is word 10
    a.push(sop2(Op::s_sub_u32, s(4), s(4), lit(4 * (10 - 6))));
    a.push(sop2(Op::s_subb_u32, s(5), s(5), int(0)));
    a.push(sop1(Op::s_setpc_b64, s(0), s(4)));
    a.bind(over);
    a.push(sopp(Op::s_endpgm, 0));
    let g = cfg::build(&a.finish()).unwrap();
    assert_eq!(g.blocks.len(), 3);
    assert_eq!(
        g.blocks[0].term,
        Term::Branch {
            cond: Cond::Scc(true),
            taken: 2,
            fall: 1
        }
    );
    assert_eq!(
        (g.blocks[1].insns.len(), g.blocks[1].term),
        (0, Term::Jump(0))
    );
    let body = check::same_run(&g, 3, 100).unwrap();
    assert_eq!(text(&body), "loop { B0 if scc { break } B1 } B2 return");
}

#[test]
fn a_computed_jump_that_is_no_far_jump_is_refused_and_a_lone_getpc_is_an_instruction() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_getpc_b64, s(0), s(0)));
    a.push(sop2(Op::s_add_u32, s(0), s(0), lit(64)));
    a.push(sopp(Op::s_endpgm, 0));
    let g = cfg::build(&a.finish()).unwrap();
    assert_eq!((g.blocks.len(), g.blocks[0].insns.len()), (1, 2));

    let mut a = Asm::new();
    a.push(sop1(Op::s_setpc_b64, s(0), s(4)));
    a.push(sopp(Op::s_endpgm, 0));
    assert_eq!(
        cfg::build(&a.finish()).unwrap_err().key(),
        "cfg:indirect-jump"
    );

    // The register the address is built in differs from the one that is jumped to.
    let mut a = Asm::new();
    a.push(sop1(Op::s_getpc_b64, s(0), s(0)));
    a.push(sop2(Op::s_add_u32, s(0), s(0), lit(8)));
    a.push(sop2(Op::s_addc_u32, s(1), s(1), int(0)));
    a.push(sop1(Op::s_setpc_b64, s(0), s(6)));
    a.push(sopp(Op::s_endpgm, 0));
    assert_eq!(
        cfg::build(&a.finish()).unwrap_err().key(),
        "cfg:indirect-jump"
    );
}

fn target() -> Vec<SigElement> {
    vec![SigElement {
        semantic: "SV_Target".to_owned(),
        index: 0,
        system_value: 64,
        component_type: 3,
        register: 0,
        mask: 15,
        rw_mask: 0,
    }]
}

fn pixel(code: Vec<u32>, name: &str) -> StageProgram {
    StageProgram {
        outputs: target(),
        name: name.to_owned(),
        ..StageProgram::new(Stage::Pixel, code)
    }
}

fn export_and_end(a: &mut Asm) {
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(2), v(2)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(2), v(2)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
}

#[test]
fn a_program_with_an_if_and_else_is_nested_hlsl_that_compiles() {
    // if (s8 == 0) v2 = 1.0 else v2 = 0.5, then export.
    let mut a = Asm::new();
    let (other, join) = (a.label(), a.label());
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.branch(Op::s_cbranch_scc0, other);
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    a.branch(Op::s_branch, join);
    a.bind(other);
    a.push(vop1(Op::v_mov_b32, v(2), float(0.5)));
    a.bind(join);
    export_and_end(&mut a);
    let p = pixel(a.finish(), "branchy");
    let host = CountingHost::default();
    let t = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap();
    let hlsl = &t.stages[0].hlsl;
    assert!(
        hlsl.contains("scc = (s8 == 0u);") && hlsl.contains("if (") && hlsl.contains("} else {"),
        "{hlsl}"
    );
    assert!(
        !hlsl.contains("switch") && !hlsl.contains("while"),
        "{hlsl}"
    );
    // The user data register is read as a number, so the mirror is needed.
    assert_eq!(t.stages[0].needs.mirror_dwords, [8]);
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    let options = Options::default();
    translate_to_dxil(&[p], &RootSignature::default(), &options, &host, &dxc)
        .unwrap_or_else(|r| panic!("{r}"));
}

#[test]
fn a_block_that_runs_once_is_do_while_false_in_hlsl_and_compiles() {
    // if (s8 == 0 && s9 == 0) v2 = 1.0 else v2 = 0.5, then one more if so that the join is
    // not the end of the program. The else is jumped to from two depths.
    let mut a = Asm::new();
    let (other, join, end) = (a.label(), a.label(), a.label());
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.branch(Op::s_cbranch_scc0, other);
    a.push(sopc(Op::s_cmp_eq_u32, s(9), int(0)));
    a.branch(Op::s_cbranch_scc0, other);
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    a.branch(Op::s_branch, join);
    a.bind(other);
    a.push(vop1(Op::v_mov_b32, v(2), float(0.5)));
    a.bind(join);
    a.push(sopc(Op::s_cmp_eq_u32, s(10), int(0)));
    a.branch(Op::s_cbranch_scc0, end);
    a.push(vop2(Op::v_add_f32, v(2), v(2), v(2)));
    a.bind(end);
    export_and_end(&mut a);
    let p = pixel(a.finish(), "once");
    let func = decant_shader::lift_program(&p).unwrap();
    assert!(
        func.body.iter().any(|n| matches!(n, Node::Once(_))),
        "{:?}",
        func.body
    );
    let (root, options, host) = (
        RootSignature::default(),
        Options::default(),
        CountingHost::default(),
    );
    let t = translate(std::slice::from_ref(&p), &root, &options, &host).unwrap();
    let hlsl = &t.stages[0].hlsl;
    assert!(
        hlsl.contains("do {") && hlsl.contains("} while (false);") && hlsl.contains("break;"),
        "{hlsl}"
    );
    assert!(!hlsl.contains("while (true)"), "{hlsl}");
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    translate_to_dxil(&[p], &root, &options, &host, &dxc).unwrap_or_else(|r| panic!("{r}"));
}

/// A pixel program of `blocks` blocks. Each adds to `v2`, counts in `s8`, compares the count
/// and branches somewhere at random. The last block exports.
///
/// The validator of the compiler refuses a loop that has no way out, so the programs have
/// none: only a conditional branch goes back, and its condition changes from round to round
/// (a loop on a condition that never changes is endless once it is entered).
fn random_program(r: &mut Random, blocks: usize) -> Vec<u32> {
    let mut a = Asm::new();
    let labels: Vec<Label> = (0..blocks).map(|_| a.label()).collect();
    for (i, &label) in labels.iter().enumerate() {
        a.bind(label);
        a.push(vop2(Op::v_add_f32, v(2), v(2), v(3)));
        if i + 1 == blocks {
            break;
        }
        a.push(sop2(Op::s_add_i32, s(8), s(8), int(1)));
        a.push(sopc(Op::s_cmp_eq_u32, s(8), int(r.below(8) as i8)));
        let forward = labels[(i + 1 + r.below(4)).min(blocks - 1)];
        let anywhere = if r.below(4) == 0 {
            labels[r.below(blocks)]
        } else {
            forward
        };
        match r.below(8) {
            0 => a.branch(Op::s_branch, forward),
            1..=3 => a.branch(Op::s_cbranch_scc0, anywhere),
            4..=6 => a.branch(Op::s_cbranch_scc1, anywhere),
            _ => &mut a,
        };
    }
    export_and_end(&mut a);
    a.finish()
}

#[test]
fn two_hundred_random_programs_are_structured_and_compile() {
    let dxc = common::find_dxc();
    if dxc.is_none() {
        eprintln!("skipped the DXC compiles: dxc.exe not found (set DECANT_SHADER_DXC)");
    }
    let mut r = Random(0x9e37_79b9_7f4a_7c15);
    let (root, options, host) = (
        RootSignature::default(),
        Options::default(),
        CountingHost::default(),
    );
    let mut shapes = std::collections::BTreeMap::new();
    for i in 0..200 {
        let blocks = 2 + r.below(14);
        let p = pixel(random_program(&mut r, blocks), "random");
        let func = decant_shader::lift_program(&p).unwrap_or_else(|e| panic!("{i}: {e}"));
        let shape = structure::shape(&func.body);
        *shapes.entry(shape).or_insert(0) += 1;
        let stages = std::slice::from_ref(&p);
        let t = translate(stages, &root, &options, &host).unwrap_or_else(|e| panic!("{i}: {e}"));
        if let Some(dxc) = &dxc {
            translate_to_dxil(stages, &root, &options, &host, dxc)
                .unwrap_or_else(|e| panic!("program {i}: {e}\n{}", t.stages[0].hlsl));
        }
    }
    assert!(shapes[&Shape::Structured] > 150, "{shapes:?}");
}

/// A pixel program with a masked region: `v0 < v1` narrows EXEC, the region is skipped when
/// no lane is left, and EXEC is restored behind it. `region` writes the skipped code.
fn masked_region(region: impl Fn(&mut Asm), after: impl Fn(&mut Asm)) -> StageProgram {
    let mut a = Asm::new();
    let end = a.label();
    a.push(vop3(Op::v_cmp_lt_f32, s(6), v(0), v(1), Operand::None));
    a.push(sop1(Op::s_and_saveexec_b64, s(4), s(6)));
    a.branch(Op::s_cbranch_execz, end);
    region(&mut a);
    a.bind(end);
    a.push(sop1(Op::s_mov_b64, EXEC, s(4)));
    after(&mut a);
    export_and_end(&mut a);
    pixel(a.finish(), "masked")
}

fn sample(a: &mut Asm, op: Op) {
    a.push(image(op, v(2), v(8), s(12), s(20), 1, 0xf));
}

/// How deep under `if`, loops and blocks each image statement of the body stands.
fn image_depths(nodes: &[Node], depth: usize, out: &mut Vec<usize>) {
    for n in nodes {
        match n {
            Node::Block(stmts) => out.extend(
                stmts
                    .iter()
                    .filter(|s| matches!(s, Stmt::Image(..)))
                    .map(|_| depth),
            ),
            Node::If { then, els, .. } => {
                image_depths(then, depth + 1, out);
                image_depths(els, depth + 1, out);
            }
            Node::Loop(body) | Node::Once(body) => image_depths(body, depth + 1, out),
            _ => {}
        }
    }
}

fn depths(p: &StageProgram) -> Vec<usize> {
    let func = decant_shader::lift_program(p).unwrap_or_else(|e| panic!("{e}"));
    let mut out = Vec::new();
    image_depths(&func.body, 0, &mut out);
    out
}

#[test]
fn a_skip_around_a_sample_that_takes_derivatives_is_dropped() {
    // The sample runs in every lane, so the lanes of a quad give it their coordinates.
    let p = masked_region(|a| sample(a, Op::image_sample), |_| {});
    assert_eq!(depths(&p), [0]);
    // A second masked region inside the first: both skips go, EXEC stays clear in a lane that
    // came in inactive whatever the inner region saves and restores.
    let p = masked_region(
        |a| {
            let inner = a.label();
            a.push(vop3(Op::v_cmp_lt_f32, s(26), v(2), v(3), Operand::None));
            a.push(sop1(Op::s_and_saveexec_b64, s(24), s(26)));
            a.branch(Op::s_cbranch_execz, inner);
            sample(a, Op::image_sample);
            a.bind(inner);
            a.push(sop1(Op::s_mov_b64, EXEC, s(24)));
        },
        |_| {},
    );
    assert_eq!(depths(&p), [0]);
}

#[test]
fn a_skip_stays_where_dropping_it_could_change_the_program() {
    // A sample at an explicit level needs no neighbours: the skip saves its work.
    let p = masked_region(|a| sample(a, Op::image_sample_lz), |_| {});
    assert_eq!(depths(&p), [1]);
    // The region sets a scalar register that is read behind it: a wave without an active
    // lane would have kept the old value.
    let p = masked_region(
        |a| {
            sample(a, Op::image_sample);
            a.push(sop1(Op::s_mov_b32, s(30), int(1)));
        },
        |a| {
            a.push(vop1(Op::v_mov_b32, v(3), s(30)));
        },
    );
    assert_eq!(depths(&p), [1]);
    // The same register, but nothing reads it afterwards: the skip goes.
    let p = masked_region(
        |a| {
            sample(a, Op::image_sample);
            a.push(sop1(Op::s_mov_b32, s(30), int(1)));
        },
        |_| {},
    );
    assert_eq!(depths(&p), [0]);
    // The region sets EXEC from a mask that was saved outside: a lane could become active.
    let mut a = Asm::new();
    let end = a.label();
    a.push(sop1(Op::s_mov_b64, s(34), EXEC));
    a.push(vop3(Op::v_cmp_lt_f32, s(6), v(0), v(1), Operand::None));
    a.push(sop1(Op::s_and_saveexec_b64, s(4), s(6)));
    a.branch(Op::s_cbranch_execz, end);
    sample(&mut a, Op::image_sample);
    a.push(sop1(Op::s_mov_b64, EXEC, s(34)));
    a.push(vop1(Op::v_mov_b32, v(3), float(1.0)));
    a.bind(end);
    a.push(sop1(Op::s_mov_b64, EXEC, s(4)));
    export_and_end(&mut a);
    assert_eq!(depths(&pixel(a.finish(), "masked")), [1]);
}
