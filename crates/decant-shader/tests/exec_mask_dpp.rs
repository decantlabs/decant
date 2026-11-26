//! Synthetic coverage for mask transport and scalar control flow around DPP.

mod common;

use decant_shader::api::{Stage, StageProgram};
use decant_shader::asm::*;
use decant_shader::decode::{Operand, Reg};
use decant_shader::ir::{Expr, Guard, Stmt, Var};
use decant_shader::tables::Op;
use decant_shader::{cfg, exec, lift_program};

fn finish(mut a: Asm, stage: Stage) -> StageProgram {
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram::new(stage, a.finish())
}

fn write(value: Operand, lane: Operand) -> decant_shader::decode::Insn {
    vop3(Op::v_writelane_b32_e64, v(144), value, lane, Operand::None)
}

#[test]
fn mask_words_written_to_vector_data_use_ballots() {
    for (width, high) in [(32, false), (64, false)] {
        let mut a = Asm::new();
        a.push(sop1(
            if width == 32 {
                Op::s_mov_b32
            } else {
                Op::s_mov_b64
            },
            s(0),
            EXEC,
        ));
        a.push(write(s(u8::from(high)), int(0)));
        // A vector read prevents this register from being classified as scalar storage.
        a.push(vop1(Op::v_mov_b32, v(1), v(144)));
        let p = finish(a, Stage::Compute);
        let f = lift_program(&p).unwrap();
        let mut found = false;
        f.for_each_stmt(&mut |stmt| {
            if let Stmt::Set {
                dst: Var::V(144),
                value: Expr::Call(h, args),
                guard,
            } = stmt
            {
                assert_eq!(h.name, "decant_write_lane");
                assert_eq!(*guard, Guard::Always);
                let Expr::Call(ballot, words) = &args[0] else {
                    panic!("missing ballot")
                };
                assert_eq!(ballot.name, "decant_mask_word");
                assert_eq!(
                    words,
                    &[Expr::Var(Var::Mask(0)), Expr::U32(u32::from(high))]
                );
                found = true;
            }
        });
        assert!(found);
        compile(&p);
    }
}

#[test]
fn literal_mask_constants_keep_the_same_kind_as_inline_constants() {
    for value in [0, u32::MAX] {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(0), EXEC));
        a.push(sop2(Op::s_and_b32, s(1), s(0), lit(value)));
        a.push(write(s(1), int(0)));
        a.push(write(lit(value), int(1)));
        a.push(vop3(
            Op::v_readlane_b32_e64,
            s(2),
            v(144),
            int(1),
            Operand::None,
        ));
        a.push(sop2(Op::s_or_b32, EXEC, s(1), s(2)));
        let p = finish(a, Stage::Pixel);
        let graph = cfg::build(&p.code).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
        lift_program(&p).unwrap();
    }
}

fn dpp_branch(branch: Op) -> StageProgram {
    let mut a = Asm::new();
    let done = a.label();
    a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
    a.branch(branch, done);
    a.push(dpp16(Op::v_add_u32, v(2), v(0), v(1), 0xff08_b100));
    a.bind(done);
    finish(a, Stage::Compute)
}

#[test]
fn native_dpp_accepts_scalar_branches_and_unconditional_jumps() {
    for branch in [Op::s_cbranch_scc0, Op::s_cbranch_scc1] {
        let p = dpp_branch(branch);
        lift_program(&p).unwrap();
        compile(&p);
    }
    let mut a = Asm::new();
    let next = a.label();
    a.branch(Op::s_branch, next);
    a.bind(next);
    a.push(dpp16(Op::v_add_u32, v(2), v(0), v(1), 0xff08_b100));
    let p = finish(a, Stage::Compute);
    lift_program(&p).unwrap();
    compile(&p);
}

#[test]
fn native_dpp_lowers_lane_dependent_branches() {
    for branch in [
        Op::s_cbranch_execz,
        Op::s_cbranch_execnz,
        Op::s_cbranch_vccz,
    ] {
        let p = dpp_branch(branch);
        lift_program(&p).unwrap();
        compile(&p);
    }
}

#[test]
fn whole_mask_transport_requires_converged_compute_invocations() {
    let mut a = Asm::new();
    let done = a.label();
    a.push(sop1(Op::s_mov_b32, s(0), EXEC));
    a.branch(Op::s_cbranch_execz, done);
    a.push(write(s(0), int(0)));
    a.push(vop1(Op::v_mov_b32, v(1), v(144)));
    a.bind(done);
    assert_eq!(
        lift_program(&finish(a, Stage::Compute)).unwrap_err().key(),
        "exec:mask-as-data"
    );
}

#[test]
fn direct_exec_high_word_can_be_written_as_data() {
    let mut a = Asm::new();
    a.push(write(Operand::Reg(Reg::ExecHi), s(8)));
    let p = finish(a, Stage::Compute);
    lift_program(&p).unwrap();
    compile(&p);
}

#[test]
fn scalar_high_words_without_mask_identity_remain_refused() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(0), EXEC));
    a.push(write(s(1), int(0)));
    a.push(vop1(Op::v_mov_b32, v(1), v(144)));
    assert_eq!(
        lift_program(&finish(a, Stage::Compute)).unwrap_err().key(),
        "exec:mask-as-data"
    );
}

#[test]
fn incompatible_mask_and_numeric_sources_remain_refused() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(0), EXEC));
    a.push(sop1(Op::s_mov_b32, s(4), int(7)));
    a.push(sop2(Op::s_or_b32, s(2), s(0), s(4)));
    assert_eq!(
        lift_program(&finish(a, Stage::Compute)).unwrap_err().key(),
        "exec:mixed-pair"
    );
}

fn compile(p: &StageProgram) {
    use decant_shader::api::{CountingHost, Options, RootSignature};
    let Some(dxc) = common::find_dxc() else {
        eprintln!("DXC unavailable, compile check skipped");
        return;
    };
    let mut p = p.clone();
    p.threads = [64, 1, 1];
    decant_shader::translate_to_dxil(
        &[p],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap();
}
