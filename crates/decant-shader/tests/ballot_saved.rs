//! A whole mask read as data: a saved EXEC used as a number takes the wave ballot of its
//! bits at the write (`decant_mask_word`), and a mask that is not an exact copy of EXEC stays refused.

mod common;

use decant_shader::api::{Stage, StageProgram};
use decant_shader::asm::*;
use decant_shader::decode::Operand;
use decant_shader::tables::Op;
use decant_shader::{lift_program, translate, Options, RootSignature, CountingHost};

const NONE: Operand = Operand::None;

fn finish(mut a: Asm) -> StageProgram {
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        ..StageProgram::new(Stage::Compute, a.finish())
    }
}

fn hlsl(p: &StageProgram) -> String {
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("refused: {r}"));
    t.stages[0].hlsl.clone()
}

fn refusal_key(p: &StageProgram) -> String {
    lift_program(p).unwrap_err().key().to_owned()
}

/// The saved EXEC read by a vector AND in the same block: the ballot is taken at the save.
#[test]
fn a_saved_exec_read_as_data_takes_its_ballot() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.push(vop2(Op::v_and_b32, v(2), s(4), v(2)));
    let p = finish(a);
    assert!(lift_program(&p).is_ok(), "{:?}", lift_program(&p).err());
    // The helper is defined once and called at the save.
    assert!(hlsl(&p).matches("decant_mask_word(").count() >= 2, "{}", hlsl(&p));
}

/// The saved EXEC read after a divergent branch: the read is in another block, and the value it takes is
/// still the ballot of the save (EXEC changes in between do not matter, the number is fixed at the save).
#[test]
fn a_saved_exec_read_after_a_branch_takes_its_ballot() {
    let mut a = Asm::new();
    let done = a.label();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.branch(Op::s_cbranch_execz, done);
    a.push(vop2(Op::v_and_b32, v(2), s(4), v(2)));
    a.bind(done);
    let p = finish(a);
    assert!(lift_program(&p).is_ok(), "{:?}", lift_program(&p).err());
    assert!(hlsl(&p).matches("decant_mask_word(").count() >= 2);
}

/// A mask written by a compare is not a copy of EXEC: it keeps the refusal (the invariant of
/// `what_needs_other_lanes_is_refused_by_name`).
#[test]
fn a_compare_mask_read_as_data_stays_refused() {
    let mut a = Asm::new();
    a.push(vop3(Op::v_cmp_lt_f32, s(4), v(0), v(1), NONE));
    a.push(vop2(Op::v_and_b32, v(2), s(4), v(2)));
    assert_eq!(refusal_key(&finish(a)), "exec:mask-as-data");
}

/// A compare mask copied to another register is not a copy of EXEC: the copy stays refused.
#[test]
fn a_copy_of_a_compare_mask_stays_refused() {
    let mut a = Asm::new();
    a.push(vop3(Op::v_cmp_lt_f32, s(6), v(0), v(1), NONE));
    a.push(sop1(Op::s_mov_b64, s(4), s(6)));
    a.push(vop2(Op::v_and_b32, v(2), s(4), v(2)));
    assert_eq!(refusal_key(&finish(a)), "exec:mask-as-data");
}

/// A mask built from EXEC through a select and an AND (`s2 = (scc ? exec : 0) & (s9 & exec)`) is a subset
/// of EXEC at each step: its read as data takes the ballot of the write that made it.
#[test]
fn a_mask_built_inside_exec_through_a_chain_takes_its_ballot() {
    let mut a = Asm::new();
    a.push(sopc(Op::s_cmp_eq_u32, s(5), int(0)));
    a.push(sop2(Op::s_cselect_b32, s(1), EXEC, int(0)));
    a.push(sop2(Op::s_and_b32, s(0), EXEC, EXEC));
    a.push(sop2(Op::s_and_b32, s(2), s(1), s(0)));
    a.push(vop1(Op::v_mov_b32, v(3), s(2)));
    let p = finish(a);
    assert!(lift_program(&p).is_ok(), "{:?}", lift_program(&p).err());
    // The ballot is in the lifted program: the HLSL drops the read, which exports nothing.
    let f = lift_program(&p).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(format!("{f:?}").contains("decant_mask_word"), "{f:?}");
}
