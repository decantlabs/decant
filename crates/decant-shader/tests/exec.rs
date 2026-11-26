//! Lane masks: where a scalar register holds a mask, in programs of both wave sizes, and what
//! the mask instructions become for one lane.
//!
//! The programs are written here with the assembler, in the shapes measured on the title's
//! shaders. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::Operand;
use decant_shader::ir::{Bin, Expr, Function, Guard, Input, Stmt, Un, Var};
use decant_shader::tables::Op;
use decant_shader::{cfg, exec, lift_program, translate_to_dxil};

fn pixel(code: Vec<u32>) -> StageProgram {
    let target = SigElement {
        semantic: "SV_Target".to_owned(),
        index: 0,
        system_value: 64,
        component_type: 3,
        register: 0,
        mask: 15,
        rw_mask: 0,
    };
    StageProgram {
        outputs: vec![target],
        name: "exec_test".to_owned(),
        ..StageProgram::new(Stage::Pixel, code)
    }
}

/// A compare in the two word form: the result goes to a scalar register.
fn compare_to(op: Op, dst: u8, a: Operand, b: Operand) -> decant_shader::decode::Insn {
    vop3(op, s(dst), a, b, Operand::None)
}

fn sets(func: &Function) -> Vec<(Var, Expr, Guard)> {
    let mut out = Vec::new();
    func.for_each_stmt(&mut |s| {
        if let Stmt::Set { dst, value, guard } = s {
            out.push((*dst, value.clone(), *guard));
        }
    });
    out
}

fn has_set(func: &Function, dst: Var, value: &Expr) -> bool {
    sets(func).iter().any(|(d, v, _)| *d == dst && v == value)
}

fn has_set_guarded(func: &Function, dst: Var, value: &Expr) -> bool {
    sets(func)
        .iter()
        .any(|(d, v, g)| *d == dst && v == value && *g == Guard::Exec)
}

/// The helper of this name that the function calls.
fn find_helper(func: &Function, name: &str) -> &'static decant_shader::ir::Helper {
    sets(func)
        .iter()
        .find_map(|(_, v, _)| match v {
            Expr::Call(h, _) if h.name == name => Some(*h),
            _ => None,
        })
        .expect("the helper is called")
}

fn refusal_key(code: Vec<u32>) -> String {
    lift_program(&pixel(code)).unwrap_err().key()
}

fn var(v: Var) -> Expr {
    Expr::Var(v)
}

#[test]
fn constant_exec_half_moves_preserve_every_other_lane_and_scc() {
    use decant_shader::decode::Reg;
    use decant_shader::lift::{Lifter, wave};

    fn eval(e: &Expr, lane: u32, old: bool) -> u32 {
        match e {
            Expr::U32(v) => *v,
            Expr::Bool(v) => u32::from(*v),
            Expr::Input(Input::LaneIndex) => lane,
            Expr::Var(Var::Exec) => u32::from(old),
            Expr::Un(Un::LNot, x) => u32::from(eval(x, lane, old) == 0),
            Expr::Select(c, a, b) => eval(if eval(c, lane, old) != 0 { a } else { b }, lane, old),
            Expr::Bin(op, a, b) => {
                let (a, b) = (eval(a, lane, old), eval(b, lane, old));
                match op {
                    Bin::ULe => u32::from(a <= b),
                    Bin::And => a & b,
                    Bin::LShr => a >> b,
                    Bin::Ne => u32::from(a != b),
                    _ => panic!("unexpected operation {op:?}"),
                }
            }
            _ => panic!("unexpected expression {e:?}"),
        }
    }

    for half in [Reg::ExecLo, Reg::ExecHi] {
        for source in [int(0), int(-1), int(1), lit(0xffff_0000), lit(0x8000_0001)] {
            let value = match source {
                Operand::Int(v) => i32::from(v) as u32,
                Operand::Lit(v) => v,
                _ => unreachable!(),
            };
            let insn = sop1(Op::s_mov_b32, Operand::Reg(half), source);
            let mut a = Asm::new();
            a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
            a.push(insn);
            a.push(sopp(Op::s_endpgm, 0));
            let graph = cfg::build(&a.finish()).unwrap();
            let masks = exec::analyse(&graph).unwrap();
            assert_eq!(masks.lanes(), 64);
            let mut cx = Lifter::new(Stage::Compute, &masks);
            assert!(wave::lift(&mut cx, &insn).unwrap());
            let [
                Stmt::Set {
                    dst: Var::Exec,
                    value: expr,
                    guard: Guard::Always,
                },
            ] = cx.stmts.as_slice()
            else {
                panic!("move must write only EXEC without a guard");
            };
            for lane in 0..64 {
                for old in [false, true] {
                    let in_half = (lane >= 32) == (half == Reg::ExecHi);
                    let expected = if in_half {
                        (value >> (lane % 32)) & 1
                    } else {
                        u32::from(old)
                    };
                    assert_eq!(
                        eval(expr, lane, old),
                        expected,
                        "{half:?} {value:08x} lane {lane}"
                    );
                }
            }
        }
    }
}

#[test]
fn exec_half_moves_still_refuse_unknown_bits_and_pixel_lane_patterns() {
    let code = |source| {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
        a.push(sop1(Op::s_mov_b32, EXEC, source));
        a.push(sopp(Op::s_endpgm, 0));
        a.finish()
    };
    let mut p = StageProgram::new(Stage::Compute, code(s(8)));
    p.threads = [64, 1, 1];
    assert_eq!(lift_program(&p).unwrap_err().key(), "exec:mask-as-data");
    assert_eq!(refusal_key(code(lit(0xffff_0000))), "exec:lane-pattern");
}

#[test]
fn either_exec_half_write_ends_a_full_wave_reduction() {
    use decant_shader::decode::Reg;

    for half in [Reg::ExecLo, Reg::ExecHi] {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
        a.push(vop2(Op::v_cndmask_b32, v(4), int(0), v(0)));
        a.push(sop1(Op::s_mov_b32, Operand::Reg(half), int(0)));
        a.push(dpp16(Op::v_add_u32, v(4), v(4), v(4), 0xff08_b100));
        a.push(sopp(Op::s_endpgm, 0));
        let graph = cfg::build(&a.finish()).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        let step = graph.insns().find(|i| i.op == Op::v_add_u32).unwrap();
        masks.enter(step.pc);
        assert!(masks.reduction().is_none());
    }
}

fn and(a: Expr, b: Expr) -> Expr {
    Expr::bin(Bin::LAnd, a, b)
}

fn not(a: Expr) -> Expr {
    Expr::un(Un::LNot, a)
}

fn export_and_end(a: &mut Asm) {
    a.push(export(0, 0xf, [0, 1, 2, 3], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
}

/// The kill shape of a 32 lane pixel program: the live mask and the saved EXEC are single
/// registers, the kill narrows EXEC with a compare, and a wave without a live lane leaves.
fn kill_32() -> Vec<u32> {
    let mut a = Asm::new();
    let exit = a.label();
    a.push(sop1(Op::s_mov_b32, s(4), EXEC));
    // s5 is the register after the live mask: a number at the same time.
    a.push(sop1(Op::s_mov_b32, s(5), int(7)));
    a.push(vopc(Op::v_cmp_eq_u32, int(1), v(0)));
    a.push(vop2(Op::v_cndmask_b32, v(1), v(2), v(3)));
    a.push(sop1(Op::s_mov_b32, s(0), EXEC));
    a.push(vopc(Op::v_cmpx_gt_f32, float(1.0), v(0)));
    a.push(sop2(Op::s_andn2_b32, s(4), s(4), EXEC));
    a.branch(Op::s_cbranch_scc0, exit);
    a.push(sop2(Op::s_and_b32, EXEC, s(0), s(4)));
    a.push(vop1(Op::v_mov_b32, v(2), s(5)));
    export_and_end(&mut a);
    a.bind(exit);
    a.push(sop1(Op::s_mov_b32, EXEC, s(4)));
    export_and_end(&mut a);
    a.finish()
}

#[test]
fn a_32_lane_program_keeps_a_mask_in_one_register() {
    let code = kill_32();
    let graph = cfg::build(&code).unwrap();
    let masks = exec::analyse(&graph).unwrap();
    assert_eq!(masks.lanes(), 32);
    let f = lift_program(&pixel(code)).unwrap();
    // The live mask and the saved EXEC are this lane's bit.
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
    assert!(has_set(&f, Var::Mask(0), &var(Var::Exec)));
    // The register next to the mask stays a number.
    assert!(has_set(&f, Var::S(5), &Expr::U32(7)));
    assert!(has_set(&f, Var::V(2), &var(Var::S(5))));
    // live = live and not killed, and SCC is this lane's bit of it.
    let all = sets(&f);
    let narrowed = and(var(Var::Mask(4)), not(var(Var::Exec)));
    let flag = all
        .iter()
        .find_map(|(d, v, _)| (*v == narrowed).then_some(*d))
        .expect("the live mask loses the killed lane");
    assert!(has_set(&f, Var::Mask(4), &var(flag)));
    assert!(has_set(&f, Var::Scc, &var(flag)));
    // EXEC after the kill, and EXEC of the lane that leaves.
    let after = and(var(Var::Mask(0)), var(Var::Mask(4)));
    assert!(all.iter().any(|(_, v, _)| *v == after));
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(4))));
}

/// Then and else of a 64 lane program: `s_and_saveexec`, `s_andn2 exec, saved, exec`, and the
/// restore by `s_mov`.
fn if_else_64() -> Vec<u32> {
    let mut a = Asm::new();
    let (invert, end) = (a.label(), a.label());
    a.push(compare_to(Op::v_cmp_lt_f32, 6, v(0), v(1)));
    a.push(sop1(Op::s_and_saveexec_b64, s(4), s(6)));
    a.branch(Op::s_cbranch_execz, invert);
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    a.bind(invert);
    a.push(sop2(Op::s_andn2_b64, EXEC, s(4), EXEC));
    a.branch(Op::s_cbranch_execz, end);
    a.push(vop1(Op::v_mov_b32, v(2), float(2.0)));
    a.bind(end);
    a.push(sop1(Op::s_mov_b64, EXEC, s(4)));
    export_and_end(&mut a);
    a.finish()
}

#[test]
fn the_region_shapes_of_a_64_lane_program_work_on_this_lanes_bit() {
    let code = if_else_64();
    let masks = exec::analyse(&cfg::build(&code).unwrap()).unwrap();
    assert_eq!(masks.lanes(), 64);
    let f = lift_program(&pixel(code)).unwrap();
    let all = sets(&f);
    // The compare result: zero for an inactive lane.
    assert!(matches!(
        all.iter().find(|(d, ..)| *d == Var::Mask(6)),
        Some((_, Expr::Bin(Bin::LAnd, ..), Guard::Always))
    ));
    // Save and narrow, the else lanes, the restore.
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
    assert!(
        all.iter()
            .any(|(_, v, _)| *v == and(var(Var::Mask(6)), var(Var::Exec)))
    );
    assert!(
        all.iter()
            .any(|(_, v, _)| *v == and(var(Var::Mask(4)), not(var(Var::Exec))))
    );
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(4))));
    // The vector writes of both sides are lane writes.
    assert_eq!(
        all.iter()
            .filter(|(d, _, g)| *d == Var::V(2) && *g == Guard::Exec)
            .count(),
        2
    );
}

#[test]
fn a_uniform_condition_lives_in_a_mask_register() {
    // `s_cmp`, then "all active lanes or none" in a pair, then the branch through VCC.
    let mut a = Asm::new();
    let skip = a.label();
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.push(sop2(Op::s_cselect_b64, s(6), EXEC, int(0)));
    a.push(sop2(Op::s_and_b64, VCC, EXEC, s(6)));
    a.branch(Op::s_cbranch_vccz, skip);
    a.push(vop1(Op::v_mov_b32, v(0), float(1.0)));
    a.bind(skip);
    a.push(vop3(Op::v_cndmask_b32, v(1), v(2), v(3), s(6)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    assert!(has_set(
        &f,
        Var::Mask(6),
        &Expr::select(var(Var::Scc), var(Var::Exec), Expr::Bool(false))
    ));
    let all = sets(&f);
    assert!(
        all.iter()
            .any(|(_, v, _)| *v == and(var(Var::Exec), var(Var::Mask(6))))
    );
    assert!(has_set(
        &f,
        Var::V(1),
        &Expr::select(var(Var::Mask(6)), var(Var::V(3)), var(Var::V(2)))
    ));
}

#[test]
fn a_register_is_a_mask_in_one_place_and_a_number_in_another() {
    let mut a = Asm::new();
    // A number first.
    a.push(sop1(Op::s_mov_b32, s(4), lit(0x1234)));
    a.push(sop2(Op::s_add_i32, s(4), s(4), s(4)));
    a.push(vop1(Op::v_mov_b32, v(0), s(4)));
    // Then the saved EXEC of a region.
    a.push(vopc(Op::v_cmp_lt_f32, v(0), v(1)));
    a.push(sop1(Op::s_and_saveexec_b32, s(4), VCC));
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    a.push(sop1(Op::s_mov_b32, EXEC, s(4)));
    // Then a number again, combined with the same bitwise instruction the masks use.
    a.push(sop1(Op::s_mov_b32, s(4), lit(0xff00)));
    a.push(sop2(Op::s_and_b32, s(5), s(4), s(8)));
    a.push(vop1(Op::v_mov_b32, v(3), s(5)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    assert!(has_set(&f, Var::S(4), &Expr::U32(0x1234)));
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(4))));
    assert!(has_set(&f, Var::S(4), &Expr::U32(0xff00)));
    assert!(has_set(&f, Var::V(3), &var(Var::S(5))));
    // Nothing writes a mask to the number's partner.
    assert!(!sets(&f).iter().any(|(d, ..)| *d == Var::Mask(5)));
}

#[test]
fn a_loop_leaves_lane_by_lane() {
    // `s_andn2_wrexec sD, sD`: the lanes that are done leave the mask of lanes to go, and EXEC
    // becomes what is left. The back edge is taken while EXEC is not empty.
    let mut a = Asm::new();
    let top = a.label();
    a.push(sop1(Op::s_mov_b64, s(0), EXEC));
    a.push(sop1(Op::s_mov_b64, s(36), EXEC));
    a.bind(top);
    a.push(vopc(Op::v_cmpx_eq_u32, s(24), v(5)));
    a.push(vop1(Op::v_mov_b32, v(6), float(1.0)));
    a.push(sop1(Op::s_andn2_wrexec_b64, s(36), s(36)));
    a.branch(Op::s_cbranch_execnz, top);
    a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let all = sets(&f);
    let left = and(var(Var::Mask(36)), not(var(Var::Exec)));
    let flag = all
        .iter()
        .find_map(|(d, v, _)| (*v == left).then_some(*d))
        .expect("the lanes to go lose the lanes that ran");
    assert!(has_set(&f, Var::Mask(36), &var(flag)));
    assert!(has_set(&f, Var::Exec, &var(flag)));
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(0))));
}

#[test]
fn the_first_lane_is_this_lane() {
    // The loop that serves lanes group by group: the value of the first lane, the lanes that
    // have the same value, the work, the lanes to go.
    let mut a = Asm::new();
    let top = a.label();
    a.push(sop1(Op::s_mov_b64, s(0), EXEC));
    a.push(sop1(Op::s_mov_b64, s(36), EXEC));
    a.bind(top);
    a.push(vop1(Op::v_readfirstlane_b32, s(24), v(5)));
    a.push(vopc(Op::v_cmpx_eq_u32, s(24), v(5)));
    a.push(vop1(Op::v_mov_b32, v(6), s(24)));
    a.push(sop1(Op::s_andn2_wrexec_b64, s(36), s(36)));
    a.branch(Op::s_cbranch_execnz, top);
    a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    // A scalar write of this lane's value, not guarded by EXEC.
    assert!(
        sets(&f)
            .iter()
            .any(|(d, v, g)| *d == Var::S(24) && *v == var(Var::V(5)) && *g == Guard::Always)
    );
}

#[test]
fn the_count_of_all_lanes_below_is_the_lane_number() {
    // The pair every mesh program starts with: the low and the high half of the lane number.
    let mut a = Asm::new();
    a.push(vop3(
        Op::v_mbcnt_lo_u32_b32,
        v(1),
        int(-1),
        int(0),
        Operand::None,
    ));
    a.push(vop3(
        Op::v_mbcnt_hi_u32_b32_e64,
        v(1),
        int(-1),
        v(1),
        Operand::None,
    ));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let lane = Expr::Input(Input::LaneIndex);
    let half = Expr::U32(32);
    let low = Expr::select(
        Expr::bin(Bin::ULt, lane.clone(), half.clone()),
        lane.clone(),
        half.clone(),
    );
    let high = Expr::select(
        Expr::bin(Bin::ULt, half.clone(), lane.clone()),
        Expr::bin(Bin::Sub, lane, half),
        Expr::U32(0),
    );
    let all = sets(&f);
    assert!(all.contains(&(
        Var::V(1),
        Expr::bin(Bin::Add, low, Expr::U32(0)),
        Guard::Exec
    )));
    assert!(all.contains(&(
        Var::V(1),
        Expr::bin(Bin::Add, high, var(Var::V(1))),
        Guard::Exec
    )));
}

#[test]
fn a_stale_value_on_a_path_never_taken_does_not_hide_a_mask() {
    // Two branches on the same condition. The first one saves EXEC in s4 on one side only, the
    // second one restores it on that side. Along the path the program never takes, the number
    // s4 held before reaches the restore too.
    let mut a = Asm::new();
    let (skip_save, skip_restore) = (a.label(), a.label());
    a.push(sop1(Op::s_mov_b32, s(4), lit(0x55)));
    a.push(vop1(Op::v_mov_b32, v(0), s(4)));
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.branch(Op::s_cbranch_scc1, skip_save);
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.push(vopc(Op::v_cmpx_lt_f32, v(0), v(1)));
    a.bind(skip_save);
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.branch(Op::s_cbranch_scc1, skip_restore);
    a.push(sop1(Op::s_mov_b64, EXEC, s(4)));
    a.bind(skip_restore);
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    assert!(has_set(&f, Var::S(4), &Expr::U32(0x55)));
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(4))));
}

#[test]
fn a_scalar_parked_in_a_vector_lane_keeps_its_kind() {
    // A 64 lane program parks a number in lane 0 and the two words of a saved EXEC in lanes 1
    // and 2 of v9, which nothing else uses, and takes them back into other registers.
    let park = |a: &mut Asm, reg: u8, lane: i8| {
        a.push(vop3(
            Op::v_writelane_b32_e64,
            v(9),
            s(reg),
            int(lane),
            Operand::None,
        ));
    };
    let back = |a: &mut Asm, reg: u8, lane: i8| {
        a.push(vop3(
            Op::v_readlane_b32_e64,
            s(reg),
            v(9),
            int(lane),
            Operand::None,
        ));
    };
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    park(&mut a, 8, 0);
    park(&mut a, 4, 1);
    park(&mut a, 5, 2);
    a.push(sop1(Op::s_mov_b32, s(4), lit(7)));
    a.push(sop1(Op::s_mov_b32, s(5), lit(8)));
    a.push(vopc(Op::v_cmpx_lt_f32, v(0), v(1)));
    back(&mut a, 20, 0);
    back(&mut a, 22, 1);
    back(&mut a, 23, 2);
    a.push(vop1(Op::v_mov_b32, v(2), s(20)));
    a.push(sop1(Op::s_mov_b64, EXEC, s(22)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let place = |lane| Var::Spill { reg: 9, lane };
    // The number goes through as it is.
    assert!(has_set(&f, place(0), &var(Var::S(8))));
    assert!(has_set(&f, Var::S(20), &var(place(0))));
    // The mask is parked as this lane's bit, by its low word. The high word is nothing.
    assert!(has_set(
        &f,
        place(1),
        &Expr::un(Un::BoolToU32, var(Var::Mask(4)))
    ));
    assert!(has_set(
        &f,
        Var::Mask(22),
        &Expr::un(Un::U32ToBool, var(place(1)))
    ));
    assert!(!sets(&f).iter().any(|(d, ..)| *d == place(2)));
    assert!(has_set(&f, Var::Exec, &var(Var::Mask(22))));
    // A lane of a register the program also uses as a vector is another lane's value.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.push(vop1(Op::v_mov_b32, v(9), float(1.0)));
    back(&mut a, 20, 3);
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "op:v_readlane_b32_e64");
}

#[test]
fn a_mask_taken_as_a_number_is_the_mask_of_a_wave_with_one_lane() {
    // The shape that asks whether a value is the same in every lane: the first lane of the
    // active lanes, its value, the lanes that differ, their count.
    let mut a = Asm::new();
    let slow = a.label();
    a.push(sop1(Op::s_mov_b64, s(6), EXEC));
    a.push(sop1(Op::s_ff1_i32_b64, s(8), s(6)));
    a.push(vop3(
        Op::v_readlane_b32_e64,
        s(11),
        v(18),
        s(8),
        Operand::None,
    ));
    a.push(compare_to(Op::v_cmp_eq_u32, 12, s(11), v(18)));
    a.push(sop2(Op::s_xor_b64, s(4), s(12), s(6)));
    a.push(sop1(Op::s_bcnt1_i32_b64, s(4), s(4)));
    a.branch(Op::s_cbranch_scc1, slow);
    a.push(vop1(Op::v_mov_b32, v(2), s(11)));
    a.bind(slow);
    a.push(vop3(
        Op::v_mbcnt_lo_u32_b32,
        v(3),
        s(6),
        int(0),
        Operand::None,
    ));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let all = sets(&f);
    let lane = Expr::Input(Input::LaneIndex);
    // The first lane of a mask is this lane or none.
    let flag = |mask: Var| {
        all.iter()
            .find_map(|(d, v, _)| (matches!(d, Var::Flag(_)) && *v == var(mask)).then_some(*d))
            .expect("the bit of the mask is taken once")
    };
    let first = flag(Var::Mask(6));
    assert!(has_set(
        &f,
        Var::S(8),
        &Expr::select(var(first), lane, Expr::U32(u32::MAX))
    ));
    // The lane the search found holds this lane's value.
    assert!(has_set(&f, Var::S(11), &var(Var::V(18))));
    // The count is 0 or 1, and SCC says which. The count goes to a number register that was
    // a mask register one instruction earlier.
    let differ = flag(Var::Mask(4));
    assert!(has_set(
        &f,
        Var::S(4),
        &Expr::select(var(differ), Expr::U32(1), Expr::U32(0))
    ));
    assert!(has_set(&f, Var::Scc, &var(differ)));
    // No lane of a real mask is below this one.
    assert!(all.contains(&(
        Var::V(3),
        Expr::bin(Bin::Add, Expr::U32(0), Expr::U32(0)),
        Guard::Exec
    )));
}

/// The reduction of v5 over the wave with the operation `op` in v0, as the title's compiler
/// writes it: every lane on, the identity in the lanes that do not take part, neighbours,
/// pairs, groups of four and of eight, the two rows.
fn reduction(a: &mut Asm, op: Op, mut identity: Operand, wide: bool) {
    let save = if wide {
        Op::s_or_saveexec_b64
    } else {
        Op::s_or_saveexec_b32
    };
    a.push(sop1(save, s(0), int(-1)));
    if matches!(identity, Operand::Lit(_)) {
        a.push(vop1(Op::v_mov_b32, v(0), identity));
        identity = v(0);
    }
    a.push(vop3(Op::v_cndmask_b32, v(0), identity, v(5), s(0)));
    for word in [0xde54_c100, 0x9360_1200, 0x0009_2400] {
        a.push(dpp8(op, v(0), v(0), v(0), word, false));
    }
    a.push(dpp16(op, v(0), v(0), v(0), 0xff09_4000));
    a.push(vop3(Op::v_permlanex16_b32, v(11), v(0), int(0), int(0)));
    a.push(vop2(op, v(0), v(11), v(0)));
}

#[test]
fn a_reduction_can_seed_its_identity_from_a_vector_move() {
    for (op, identity) in [
        (Op::v_min_f32, f32::INFINITY.to_bits()),
        (Op::v_max_f32, f32::NEG_INFINITY.to_bits()),
    ] {
        let mut a = Asm::new();
        reduction(&mut a, op, Operand::Lit(identity), false);
        a.push(vop3(
            Op::v_readlane_b32_e64,
            s(3),
            v(0),
            int(0),
            Operand::None,
        ));
        a.push(sop1(Op::s_mov_b32, EXEC, s(0)));
        a.push(vop1(Op::v_mov_b32, v(0), s(3)));
        export_and_end(&mut a);
        let program = pixel(a.finish());
        let lifted = lift_program(&program).unwrap();
        assert!(has_set(&lifted, Var::V(11), &Expr::U32(identity)));
        assert!(has_set(&lifted, Var::S(3), &var(Var::V(0))));
        if let Some(dxc) = common::find_dxc() {
            let compiled = translate_to_dxil(
                &[program],
                &RootSignature::default(),
                &Options::default(),
                &CountingHost::default(),
                &dxc,
            )
            .unwrap_or_else(|r| panic!("{r}"));
            assert_eq!(&compiled[0].dxil[..4], b"DXBC");
        }
    }
}

#[test]
fn a_reduction_over_the_wave_is_this_lanes_value() {
    // 32 lanes: the result is in lane 0.
    let mut a = Asm::new();
    reduction(&mut a, Op::v_min_u32, int(-1), false);
    a.push(vop3(
        Op::v_readlane_b32_e64,
        s(3),
        v(0),
        int(0),
        Operand::None,
    ));
    a.push(sop1(Op::s_mov_b32, EXEC, s(0)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    // The value of the lanes that take part, the identity for the others.
    assert!(has_set(
        &f,
        Var::V(0),
        &Expr::select(var(Var::Mask(0)), var(Var::V(5)), Expr::U32(u32::MAX))
    ));
    // The four steps change nothing, and the other row holds the identity.
    assert!(has_set(&f, Var::V(11), &Expr::U32(u32::MAX)));
    assert!(has_set(&f, Var::S(3), &var(Var::V(0))));

    // 64 lanes: the result of the upper half is combined into every lane, then lane 0 is read.
    let mut a = Asm::new();
    reduction(&mut a, Op::v_add_u32, int(0), true);
    a.push(vop3(
        Op::v_readlane_b32_e64,
        s(2),
        v(0),
        int(32),
        Operand::None,
    ));
    a.push(vop2(Op::v_add_u32, v(0), s(2), v(0)));
    a.push(vop3(
        Op::v_readlane_b32_e64,
        s(4),
        v(0),
        int(0),
        Operand::None,
    ));
    a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let upper = Expr::bin(Bin::ULe, Expr::U32(32), Expr::Input(Input::LaneIndex));
    // The upper half holds this lane's value when this lane is in it, else nothing.
    assert!(has_set(
        &f,
        Var::S(2),
        &Expr::select(upper.clone(), var(Var::V(0)), Expr::U32(0))
    ));
    // Lane 0 then holds the sum of both halves: this lane's value, wherever this lane is.
    assert!(has_set(
        &f,
        Var::S(4),
        &Expr::select(not(upper), var(Var::V(0)), var(Var::S(2)))
    ));

    // A step with the wrong identity, or out of order, is no reduction.
    let mut a = Asm::new();
    reduction(&mut a, Op::v_min_u32, int(0), false);
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "form:v_min_u32:dpp");
    let mut a = Asm::new();
    a.push(sop1(Op::s_or_saveexec_b32, s(0), int(-1)));
    a.push(vop3(Op::v_cndmask_b32, v(0), int(0), v(5), s(0)));
    a.push(dpp8(Op::v_add_u32, v(0), v(0), v(0), 0x9360_1200, false));
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "form:v_add_u32:dpp");
}

/// A pixel program that takes differences in its quad: the three swizzles of the title's
/// derivative code and a data parallel move across the diagonal.
fn quad_reads() -> Vec<u32> {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.push(sop1(Op::s_wqm_b64, EXEC, EXEC));
    // Lane 0, 1 and 2 of the quad: here, to the right, below.
    for (dst, pattern) in [(3, 0x00), (4, 0x55), (5, 0xaa)] {
        a.push(ds(
            Op::ds_swizzle_b32,
            v(dst),
            v(2),
            v(0),
            v(0),
            pattern,
            0x80,
        ));
    }
    a.push(vop2(Op::v_sub_f32, v(4), v(4), v(3)));
    a.push(vop2(Op::v_sub_f32, v(5), v(5), v(3)));
    // Quad permutation 3, 2, 1, 0 with fetch inactive and bound control, all rows and banks.
    a.push(dpp16(Op::v_mov_b32, v(6), v(0), v(0), 0xff0c_1b00));
    a.push(sop2(Op::s_and_b64, EXEC, EXEC, s(4)));
    a.push(export(0, 0xf, [3, 4, 5, 6], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    a.finish()
}

#[test]
fn a_pixel_reads_its_neighbours_in_the_quad() {
    let f = lift_program(&pixel(quad_reads())).unwrap();
    let all = sets(&f);
    let called = |dst: Var, name: &str, args: usize| {
        all.iter().any(|(d, v, g)| {
            *d == dst
                && *g == Guard::Exec
                && matches!(v, Expr::Call(h, a) if h.name == name && a.len() == args)
        })
    };
    assert!(called(Var::V(3), "decant_quad_at", 2));
    assert!(called(Var::V(4), "decant_quad_at", 2));
    assert!(called(Var::V(6), "decant_quad_diagonal", 1));
    assert!(has_set_guarded(
        &f,
        Var::V(5),
        &Expr::Call(
            find_helper(&f, "decant_quad_at"),
            vec![var(Var::V(2)), Expr::U32(2)]
        )
    ));
    // The text compiles: the quad reads exist for pixel programs.
    if let Some(dxc) = common::find_dxc() {
        let compiled = translate_to_dxil(
            &[pixel(quad_reads())],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }
    // A vertex program has no quads.
    let vertex = StageProgram {
        name: "exec_test".to_owned(),
        ..StageProgram::new(Stage::Vertex, quad_reads())
    };
    assert_eq!(lift_program(&vertex).unwrap_err().key(), "exec:cross-lane");
    // The mode that masks lane numbers within 32 lanes needs the real wave.
    let mut a = Asm::new();
    a.push(ds(Op::ds_swizzle_b32, v(3), v(2), v(0), v(0), 0x1f, 0x04));
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "exec:cross-lane");
}

#[test]
fn exec_made_from_a_lane_count_asks_for_the_lane_number() {
    // The start of a geometry program: as many lanes as there are primitives.
    let mut a = Asm::new();
    a.push(sop2(Op::s_bfm_b64, EXEC, s(8), int(0)));
    a.push(sopc(Op::s_cmp_eq_u32, s(8), lit(64)));
    a.push(sop2(Op::s_cselect_b64, EXEC, int(-1), EXEC));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    let inside = sets(&f)
        .into_iter()
        .find(|(d, v, _)| *d == Var::Exec && matches!(v, Expr::Bin(Bin::LAnd, ..)))
        .expect("EXEC is a test of the lane number");
    let Expr::Bin(_, _, below) = inside.1 else {
        unreachable!()
    };
    assert!(
        matches!(*below, Expr::Bin(Bin::ULt, ref lane, _) if **lane == Expr::Input(Input::LaneIndex))
    );
    assert!(has_set(
        &f,
        Var::Exec,
        &Expr::select(var(Var::Scc), Expr::Bool(true), var(Var::Exec))
    ));
}

#[test]
fn whole_quad_mode_and_a_mask_compared_with_zero() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(12), EXEC));
    a.push(sop1(Op::s_wqm_b32, EXEC, EXEC));
    a.push(vopc(Op::v_cmp_lt_f32, v(0), v(1)));
    a.push(sop2(Op::s_andn2_b32, s(12), s(12), VCC));
    a.push(sop1(Op::s_wqm_b32, s(13), s(12)));
    a.push(sop2(Op::s_and_b32, EXEC, EXEC, s(13)));
    a.push(sopc(Op::s_cmp_eq_u32, s(12), int(0)));
    a.push(sop2(Op::s_and_b32, EXEC, EXEC, s(12)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    assert!(has_set(&f, Var::Exec, &Expr::Bool(true)));
    assert!(has_set(&f, Var::Mask(13), &Expr::Bool(true)));
    assert!(has_set(&f, Var::Scc, &not(var(Var::Mask(12)))));
}

#[test]
fn what_needs_other_lanes_is_refused_by_name() {
    // A mask read as a number.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(4), EXEC));
    a.push(compare_to(Op::v_cmp_lt_f32, 6, v(0), v(1)));
    a.push(vop1(Op::v_mov_b32, v(2), s(6)));
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "exec:mask-as-data");
    // Lanes named by their position.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(4), EXEC));
    a.push(sop1(Op::s_mov_b32, EXEC, lit(0x0001_0001)));
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "exec:lane-pattern");
    // A number used to select lanes.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(4), EXEC));
    a.push(sop2(Op::s_add_i32, s(6), s(8), s(9)));
    a.push(vop3(Op::v_cndmask_b32, v(1), v(2), v(3), s(6)));
    export_and_end(&mut a);
    assert_eq!(refusal_key(a.finish()), "exec:data-as-mask");
}

#[test]
fn the_same_count_on_a_number_is_left_to_the_scalar_rules() {
    // `s_bcnt1_i32_b32` of a number in a 32 lane program is no lane instruction.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(4), EXEC));
    a.push(sop1(Op::s_bcnt1_i32_b32, s(2), s(8)));
    a.push(sop1(Op::s_mov_b32, EXEC, s(4)));
    export_and_end(&mut a);
    let f = lift_program(&pixel(a.finish())).unwrap();
    assert!(sets(&f).iter().any(|(d, ..)| *d == Var::S(2)));
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
}

#[test]
fn a_region_with_a_kill_compiles() {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    for code in [if_else_64(), {
        // The 32 lane kill without its early exit, which the pixel stage owns.
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(4), EXEC));
        a.push(sop1(Op::s_mov_b32, s(0), EXEC));
        a.push(vopc(Op::v_cmpx_gt_f32, float(1.0), v(0)));
        a.push(sop2(Op::s_andn2_b32, s(4), s(4), EXEC));
        a.push(sop2(Op::s_and_b32, EXEC, s(0), s(4)));
        export_and_end(&mut a);
        a.finish()
    }] {
        let compiled = translate_to_dxil(
            &[pixel(code)],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }
}

#[test]
fn constant_lane_patterns_work_in_compute_waves_and_preserve_every_lane() {
    use decant_shader::lift::{Lifter, wave};

    fn eval(e: &Expr, lane: u32, old: bool) -> u32 {
        match e {
            Expr::U32(v) => *v,
            Expr::Bool(v) => u32::from(*v),
            Expr::Input(Input::LaneIndex) => lane,
            Expr::Var(Var::Exec) => u32::from(old),
            Expr::Un(Un::LNot, x) => u32::from(eval(x, lane, old) == 0),
            Expr::Select(c, a, b) => eval(if eval(c, lane, old) != 0 { a } else { b }, lane, old),
            Expr::Bin(op, a, b) => {
                let (a, b) = (eval(a, lane, old), eval(b, lane, old));
                match op {
                    Bin::ULe => u32::from(a <= b),
                    Bin::And => a & b,
                    Bin::LShr => a >> b,
                    Bin::Ne => u32::from(a != b),
                    Bin::LAnd => u32::from(a != 0 && b != 0),
                    _ => panic!("unexpected operation {op:?}"),
                }
            }
            _ => panic!("unexpected expression {e:?}"),
        }
    }

    // Wave32 constant EXEC moves: every lane gets its exact bit without touching SCC.
    for source in [
        int(1),
        int(-2),
        lit(0xffff_0000),
        lit(0x0000_ffff),
        lit(0x5555_5555),
        lit(0x8000_0001),
    ] {
        let value = match source {
            Operand::Int(v) => i32::from(v) as u32,
            Operand::Lit(v) => v,
            _ => unreachable!(),
        };
        let insn = sop1(Op::s_mov_b32, EXEC, source);
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, EXEC, int(-1)));
        a.push(insn);
        a.push(sopp(Op::s_endpgm, 0));
        let graph = cfg::build(&a.finish()).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        assert_eq!(masks.lanes(), 32);
        let mut cx = Lifter::new(Stage::Compute, &masks);
        assert!(wave::lift(&mut cx, &insn).unwrap());
        let [
            Stmt::Set {
                dst: Var::Exec,
                value: expr,
                guard: Guard::Always,
            },
        ] = cx.stmts.as_slice()
        else {
            panic!("move must write only EXEC without a guard");
        };
        for lane in 0..32 {
            for old in [false, true] {
                let expected = (value >> lane) & 1;
                assert_eq!(eval(expr, lane, old), expected, "{value:08x} lane {lane}");
            }
        }
    }

    // Wave64 constant 64 bit EXEC moves: literals zero extend, inline integers sign extend.
    for (source, expected_mask) in [
        (lit(0xffff_0000), 0x0000_0000_ffff_0000u64),
        (int(1), 0x0000_0000_0000_0001u64),
        (int(-2), 0xffff_ffff_ffff_fffeu64),
    ] {
        let insn = sop1(Op::s_mov_b64, EXEC, source);
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
        a.push(insn);
        a.push(sopp(Op::s_endpgm, 0));
        let graph = cfg::build(&a.finish()).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        assert_eq!(masks.lanes(), 64);
        let mut cx = Lifter::new(Stage::Compute, &masks);
        assert!(wave::lift(&mut cx, &insn).unwrap());
        let [
            Stmt::Set {
                dst: Var::Exec,
                value: expr,
                guard: Guard::Always,
            },
        ] = cx.stmts.as_slice()
        else {
            panic!("move must write only EXEC without a guard");
        };
        for lane in 0..64 {
            let expected = ((expected_mask >> lane) & 1) as u32;
            assert_eq!(eval(expr, lane, false), expected, "lane {lane}");
        }
    }

    // Wave32 mask combine and saveexec with a constant lane pattern.
    let mut a = Asm::new();
    a.push(sop1(Op::s_and_saveexec_b32, s(4), lit(0xffff_0000)));
    a.push(sop2(Op::s_and_b32, EXEC, EXEC, lit(0x5555_5555)));
    a.push(sop1(Op::s_mov_b32, EXEC, s(4)));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = StageProgram::new(Stage::Compute, a.finish());
    p.threads = [32, 1, 1];
    let f = lift_program(&p).unwrap();
    assert_eq!(f.lanes, 32);
    assert!(has_set(&f, Var::Mask(4), &var(Var::Exec)));
}

#[test]
fn wave32_exclusive_scan_is_recognised_with_optional_narrow_and_waitcnt() {
    use decant_shader::decode::Ext;

    let build = |with_narrow: bool, with_wait: bool| {
        let mut a = Asm::new();
        a.push(sop1(Op::s_or_saveexec_b32, s(1), int(-1)));
        if with_wait {
            a.push(sopp(Op::s_waitcnt, 0xc07f));
        }
        let mask = if with_narrow {
            a.push(sop2(Op::s_and_b32, s(2), s(8), s(1)));
            s(2)
        } else {
            s(1)
        };
        a.push(vop3(Op::v_cndmask_b32, v(1), int(0), v(0), mask));
        a.push(vop1(Op::v_mov_b32, v(2), v(1)));
        for control in [0xff09_1100, 0xff01_1200, 0xff01_1400, 0xff01_1800] {
            a.push(dpp16(Op::v_add_u32, v(2), v(2), v(2), control));
        }
        a.push(sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)));
        let mut p = vop3(Op::v_permlanex16_b32, v(3), v(2), int(-1), int(-1));
        p.ext = Ext::Valu {
            abs: 0,
            neg: 0,
            opsel: 9,
            omod: 0,
            clamp: false,
        };
        a.push(p);
        a.push(vop2(Op::v_add_u32, v(2), v(3), v(2)));
        a.push(sop1(Op::s_mov_b32, EXEC, int(-1)));
        a.push(vop2(Op::v_sub_u32, v(2), v(2), v(1)));
        a.push(sop1(Op::s_mov_b32, EXEC, s(1)));
        a.push(sopp(Op::s_endpgm, 0));
        a.finish()
    };

    for (narrow, wait) in [(false, false), (true, false), (false, true), (true, true)] {
        let code = build(narrow, wait);
        let graph = cfg::build(&code).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        assert_eq!(masks.lanes(), 32);
        let steps: Vec<_> = graph
            .insns()
            .filter(|i| {
                masks.enter(i.pc);
                masks.reduction() == Some(exec::Reduction::Step)
            })
            .collect();
        assert_eq!(steps.len(), 4);
        let perm = graph
            .insns()
            .find(|i| i.op == Op::v_permlanex16_b32)
            .unwrap();
        masks.enter(perm.pc);
        assert_eq!(
            masks.reduction(),
            Some(exec::Reduction::OtherRow { identity: 0 })
        );
    }
}

#[test]
fn bvh_and_r128_mimg_read_four_descriptor_sgprs() {
    use decant_shader::decode::{Ext, Insn};

    let with_r128 = |mut insn: Insn, r128: bool| {
        if let Ext::Mimg(m) = &mut insn.ext {
            m.r128 = r128;
            m.unrm = true;
        }
        insn
    };

    // Wave32 compute shape of `cs 027b5c340a1d62d5`: `s0..s3` hold the 128-bit BVH descriptor
    // while `s5`, `s6`, `s7` hold live saved EXEC masks.
    let build = |insn: Insn, mask_in_low_half: bool| {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(5), EXEC));
        a.push(sop1(Op::s_mov_b32, s(6), EXEC));
        a.push(sop1(Op::s_mov_b32, s(7), EXEC));
        a.push(sop1(Op::s_mov_b32, s(0), int(0)));
        a.push(sop1(Op::s_mov_b32, s(1), lit(0x8000_0000)));
        a.push(sop1(Op::s_mov_b32, s(2), int(-1)));
        if mask_in_low_half {
            a.push(sop1(Op::s_mov_b32, s(3), EXEC));
        } else {
            a.push(sop1(Op::s_mov_b32, s(3), lit(0x8900_03ff)));
        }
        a.push(insn);
        if mask_in_low_half {
            a.push(sop1(Op::s_mov_b32, EXEC, s(3)));
        }
        a.push(sop2(Op::s_and_b32, EXEC, s(5), s(6)));
        a.push(sop1(Op::s_mov_b32, EXEC, s(7)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut p = StageProgram::new(Stage::Compute, a.finish());
        p.threads = [32, 1, 1];
        p
    };

    let bvh64_nsa = with_r128(
        image_nsa(
            Op::image_bvh64_intersect_ray,
            v(32),
            v(32),
            &[33, 12, 16, 41, 46, 24, 40, 23, 34, 35, 36],
            s(0),
            s(0),
            0,
            0xf,
        ),
        true,
    );
    let bvh32_nsa = with_r128(
        image_nsa(
            Op::image_bvh_intersect_ray,
            v(17),
            v(17),
            &[4, 1, 2, 3, 6, 9, 11, 8, 10, 12],
            s(0),
            s(0),
            0,
            0xf,
        ),
        true,
    );
    let bvh32_no_r128 = with_r128(
        image(Op::image_bvh_intersect_ray, v(17), v(0), s(0), s(0), 0, 0xf),
        false,
    );
    let load_r128 = with_r128(image(Op::image_load, v(4), v(0), s(0), s(0), 1, 0xf), true);
    let load_256 = with_r128(image(Op::image_load, v(4), v(0), s(0), s(0), 1, 0xf), false);

    for (insn, expected_key) in [
        (bvh64_nsa, None),
        (bvh32_nsa, None),
        (bvh32_no_r128, None),
        (load_r128, Some("form:image_load:image-r128")),
    ] {
        let p = build(insn, false);
        let graph = cfg::build(&p.code).unwrap();
        let masks = exec::analyse(&graph).unwrap();
        let target = graph.insns().find(|i| i.op == insn.op).unwrap();
        masks.enter(target.pc);
        assert_eq!(masks.refusal(), None, "{}", insn.op.name());
        match expected_key {
            Some(key) => assert_eq!(lift_program(&p).unwrap_err().key(), key),
            None => {
                lift_program(&p).unwrap();
            }
        }
    }

    // A live lane mask inside the 4-dword descriptor (`s3`) is still refused as `mask-as-data`.
    let p_low = build(bvh64_nsa, true);
    let graph_low = cfg::build(&p_low.code).unwrap();
    let masks_low = exec::analyse(&graph_low).unwrap();
    let target_low = graph_low
        .insns()
        .find(|i| i.op == Op::image_bvh64_intersect_ray)
        .unwrap();
    masks_low.enter(target_low.pc);
    assert_eq!(masks_low.refusal(), Some("mask-as-data"));
    assert_eq!(lift_program(&p_low).unwrap_err().key(), "exec:mask-as-data");

    // An ordinary 256-bit MIMG (`r128 == false`) still reads `s0..s7` and refuses masks in `s5..s7`.
    let p_wide = build(load_256, false);
    let graph_wide = cfg::build(&p_wide.code).unwrap();
    let masks_wide = exec::analyse(&graph_wide).unwrap();
    let target_wide = graph_wide.insns().find(|i| i.op == Op::image_load).unwrap();
    masks_wide.enter(target_wide.pc);
    assert_eq!(masks_wide.refusal(), Some("mask-as-data"));
    assert_eq!(
        lift_program(&p_wide).unwrap_err().key(),
        "exec:mask-as-data"
    );
}

#[test]
fn non_nsa_bvh_marks_its_full_vector_address_span_against_parking() {
    use decant_shader::decode::Ext;

    let mut bvh = image(
        Op::image_bvh64_intersect_ray,
        v(32),
        v(10),
        s(0),
        s(0),
        0,
        0xf,
    );
    if let Ext::Mimg(m) = &mut bvh.ext {
        m.r128 = true;
        m.unrm = true;
    }

    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(5), EXEC));
    // `v15` lies inside `v10..v21` (12 sequential address VGPRs of non-NSA `image_bvh64`), so it
    // must not be treated as a scalar parking register; `v22` lies just past the span and can park.
    a.push(vop3(
        Op::v_writelane_b32_e64,
        v(15),
        s(5),
        int(0),
        Operand::None,
    ));
    a.push(vop3(
        Op::v_writelane_b32_e64,
        v(22),
        s(5),
        int(0),
        Operand::None,
    ));
    a.push(bvh);
    a.push(vop3(
        Op::v_readlane_b32_e64,
        s(6),
        v(22),
        int(0),
        Operand::None,
    ));
    a.push(sop1(Op::s_mov_b32, EXEC, s(6)));
    a.push(sopp(Op::s_endpgm, 0));

    let graph = cfg::build(&a.finish()).unwrap();
    let masks = exec::analyse(&graph).unwrap();
    let writes: Vec<_> = graph
        .insns()
        .filter(|i| i.op == Op::v_writelane_b32_e64)
        .collect();
    assert_eq!(writes.len(), 2);
    masks.enter(writes[0].pc);
    assert_eq!(masks.parked(), None);
    masks.enter(writes[1].pc);
    assert_eq!(masks.parked(), Some(exec::Parked::Mask));
}

#[test]
fn dpp_staged_64bit_carry_chain_wave32_reduction() {
    use decant_shader::decode::Ext;

    let build = |valid_carry: bool| {
        let mut a = Asm::new();
        a.push(sop1(Op::s_or_saveexec_b32, s(0), int(-1)));
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(vop1(Op::v_mov_b32, v(2), int(0)));
        a.push(vop3(Op::v_cndmask_b32, v(1), v(1), v(3), s(0)));
        a.push(vop3(Op::v_cndmask_b32, v(2), v(2), v(4), s(0)));
        let steps: [(bool, u32); 4] = [
            (true, 0xde54_c100),
            (true, 0x9360_1200),
            (true, 0x0009_2400),
            (false, 0xff09_4000),
        ];
        for (is_dpp8, word) in steps {
            if is_dpp8 {
                a.push(dpp8(Op::v_mov_b32, v(5), v(1), Operand::None, word, false));
                a.push(dpp8(Op::v_mov_b32, v(6), v(2), Operand::None, word, false));
            } else {
                a.push(dpp16(Op::v_mov_b32, v(5), v(1), Operand::None, word));
                a.push(dpp16(Op::v_mov_b32, v(6), v(2), Operand::None, word));
            }
            let mut add_lo = vop3(Op::v_add_co_u32_e64, v(1), v(5), v(1), Operand::None);
            add_lo.dst2 = VCC;
            a.push(add_lo);
            let mut add_hi = vop3(
                Op::v_addc_co_u32,
                v(2),
                v(6),
                v(2),
                if valid_carry { VCC } else { s(8) },
            );
            add_hi.dst2 = VCC;
            a.push(add_hi);
        }
        a.push(vop3(Op::v_permlanex16_b32, v(5), v(1), int(0), int(0)));
        a.push(vop3(Op::v_permlanex16_b32, v(6), v(2), int(0), int(0)));
        let mut add_lo = vop3(Op::v_add_co_u32_e64, v(1), v(5), v(1), Operand::None);
        add_lo.dst2 = VCC;
        a.push(add_lo);
        let mut add_hi = vop3(
            Op::v_addc_co_u32,
            v(2),
            v(6),
            v(2),
            if valid_carry { VCC } else { s(8) },
        );
        add_hi.dst2 = VCC;
        a.push(add_hi);
        a.push(vop3(Op::v_readlane_b32_e64, s(2), v(1), int(0), Operand::None));
        a.push(vop3(Op::v_readlane_b32_e64, s(3), v(2), int(0), Operand::None));
        a.push(sop1(Op::s_mov_b32, EXEC, s(0)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut p = StageProgram::new(Stage::Compute, a.finish());
        p.threads = [32, 1, 1];
        let _ = Ext::None;
        p
    };

    let good = build(true);
    let graph = cfg::build(&good.code).unwrap();
    let masks = exec::analyse(&graph).unwrap();
    let other_rows = graph
        .insns()
        .filter(|i| {
            masks.enter(i.pc);
            masks.reduction() == Some(exec::Reduction::OtherRow { identity: 0 })
        })
        .count();
    let results = graph
        .insns()
        .filter(|i| {
            masks.enter(i.pc);
            matches!(masks.reduction(), Some(exec::Reduction::Result { .. }))
        })
        .count();
    assert_eq!(other_rows, 10);
    assert_eq!(results, 2);
    lift_program(&good).unwrap();
    if let Some(dxc) = common::find_dxc() {
        let compiled = translate_to_dxil(
            &[good],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }

    // Without a proven zero carry from the low word combine, the high word DPP stages stay unmatched.
    let bad = build(false);
    let bad_graph = cfg::build(&bad.code).unwrap();
    let bad_masks = exec::analyse(&bad_graph).unwrap();
    let bad_other_rows = bad_graph
        .insns()
        .filter(|i| {
            bad_masks.enter(i.pc);
            bad_masks.reduction() == Some(exec::Reduction::OtherRow { identity: 0 })
        })
        .count();
    assert_eq!(bad_other_rows, 5);
}

#[test]
fn multi_block_64_lane_vcmp_ballot_materializes_for_mov_and_cselect() {
    let build = |from_vcmp: bool| {
        let mut a = Asm::new();
        let skip = a.label();
        let join = a.label();
        a.push(sop1(Op::s_mov_b64, s(2), EXEC));
        if from_vcmp {
            a.push(compare_to(Op::v_cmp_lt_u32, 4, v(0), int(8)));
        } else {
            a.push(sop1(Op::s_mov_b64, s(4), EXEC));
        }
        a.push(sopc(Op::s_cmp_eq_u32, s(6), int(0)));
        a.branch(Op::s_cbranch_scc1, skip);
        a.push(vop1(Op::v_mov_b32, v(1), int(1)));
        a.branch(Op::s_branch, join);
        a.bind(skip);
        a.push(vop1(Op::v_mov_b32, v(1), int(2)));
        a.bind(join);
        a.push(sop1(Op::s_mov_b32, s(8), s(4)));
        a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
        a.push(sop2(Op::s_cselect_b32, s(9), s(5), s(8)));
        a.push(vop1(Op::v_mov_b32, v(2), s(9)));
        a.push(sop1(Op::s_mov_b64, EXEC, s(2)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut p = StageProgram::new(Stage::Compute, a.finish());
        p.threads = [64, 1, 1];
        p
    };

    let good = build(true);
    let graph = cfg::build(&good.code).unwrap();
    let masks = exec::analyse(&graph).unwrap();
    let cmp = graph
        .insns()
        .find(|i| i.op == Op::v_cmp_lt_u32)
        .unwrap();
    masks.enter(cmp.pc);
    assert_eq!(masks.materialize_ballot(), 3);
    let mov = graph
        .insns()
        .find(|i| i.op == Op::s_mov_b32 && i.dst == s(8))
        .unwrap();
    masks.enter(mov.pc);
    assert!(masks.ballot_ready());
    let csel = graph
        .insns()
        .find(|i| i.op == Op::s_cselect_b32)
        .unwrap();
    masks.enter(csel.pc);
    assert!(masks.ballot_ready());
    let translated = decant_shader::translate(
        std::slice::from_ref(&good),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap();
    let hlsl = &translated.stages[0].hlsl;
    assert!(hlsl.contains("[WaveSize(64)]"), "{hlsl}");
    assert!(hlsl.contains("decant_mask_word("), "{hlsl}");
    if let Some(dxc) = common::find_dxc() {
        let compiled = translate_to_dxil(
            &[good],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }

    // A non-v_cmp mask (saved EXEC) across multiple blocks must stay refused.
    let bad = build(false);
    assert_eq!(lift_program(&bad).unwrap_err().key(), "exec:mask-as-data");
}

#[test]
fn zero_test_resolves_to_number_when_mask_only_reaches_under_empty_exec() {
    let build = |mask_path_v47: i8, observe_scc: bool| {
        let mut a = Asm::new();
        let num_path = a.label();
        let join = a.label();
        let exit = a.label();
        a.push(sop1(Op::s_mov_b32, s(10), EXEC));
        a.push(sopc(Op::s_cmp_eq_u32, s(6), int(0)));
        a.branch(Op::s_cbranch_scc1, num_path);
        a.push(sop1(Op::s_mov_b32, s(1), EXEC));
        a.push(vop1(Op::v_mov_b32, v(47), int(mask_path_v47)));
        a.branch(Op::s_branch, join);
        a.bind(num_path);
        a.push(sop1(Op::s_mov_b32, s(1), int(7)));
        a.push(vop1(Op::v_mov_b32, v(47), s(7)));
        a.bind(join);
        a.push(vopc(Op::v_cmp_eq_u32, int(-1), v(47)));
        a.push(sop2(Op::s_andn2_b32, VCC, EXEC, VCC));
        a.branch(Op::s_cbranch_vccnz, exit);
        a.push(sopc(Op::s_cmp_eq_u32, s(1), int(0)));
        if observe_scc {
            a.push(sop2(Op::s_cselect_b32, s(0), int(1), int(0)));
        } else {
            a.push(sop2(Op::s_cselect_b32, s(0), EXEC, int(0)));
        }
        a.push(sop2(Op::s_and_b32, s(1), s(8), int(4)));
        a.push(vop1(Op::v_mov_b32, v(2), s(1)));
        a.bind(exit);
        a.push(sop1(Op::s_mov_b32, EXEC, s(10)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut p = StageProgram::new(Stage::Compute, a.finish());
        p.threads = [32, 1, 1];
        p
    };

    let good = build(0, false);
    lift_program(&good).unwrap();
    if let Some(dxc) = common::find_dxc() {
        let compiled = translate_to_dxil(
            &[good],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }

    // Observing SCC with a non-EXEC select keeps `exec:mixed-pair`.
    assert_eq!(
        lift_program(&build(0, true)).unwrap_err().key(),
        "exec:mixed-pair"
    );
    // When the mask path matches the compare constant (-1), active lanes can fall through.
    assert_eq!(
        lift_program(&build(-1, false)).unwrap_err().key(),
        "exec:mixed-pair"
    );
}

#[test]
fn wave64_scan_with_waitcnt_and_cross_lane_read_lowers_with_real_wave_intrinsics() {
    use decant_shader::decode::Ext;

    let build = |with_cross_lane_read: bool| {
        let mut a = Asm::new();
        let skip = a.label();
        // Wave64 exclusive prefix sum with intervening `s_waitcnt`.
        a.push(sop1(Op::s_or_saveexec_b64, s(2), int(-1)));
        a.push(sopp(Op::s_waitcnt, 0x3f70));
        a.push(vop3(Op::v_cndmask_b32, v(1), int(0), v(5), s(2)));
        a.push(vop1(Op::v_mov_b32, v(2), v(1)));
        for control in [0xff09_1100, 0xff01_1200, 0xff01_1400, 0xff01_1800] {
            a.push(dpp16(Op::v_add_u32, v(2), v(2), v(2), control));
        }
        a.push(sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)));
        a.push(sop1(Op::s_mov_b32, Operand::Reg(decant_shader::decode::Reg::ExecHi), lit(0xffff_0000)));
        let mut p = vop3(Op::v_permlanex16_b32, v(3), v(2), int(-1), int(-1));
        p.ext = Ext::Valu {
            abs: 0,
            neg: 0,
            opsel: 9,
            omod: 0,
            clamp: false,
        };
        a.push(p);
        a.push(vop2(Op::v_add_u32, v(2), v(3), v(2)));
        a.push(vop3(Op::v_readlane_b32_e64, s(10), v(2), int(31), Operand::None));
        a.push(sop1(Op::s_mov_b32, EXEC, int(0)));
        a.push(sop1(Op::s_mov_b32, Operand::Reg(decant_shader::decode::Reg::ExecHi), int(-1)));
        a.push(vop2(Op::v_add_u32, v(2), s(10), v(2)));
        a.push(sop1(Op::s_mov_b32, EXEC, int(-1)));
        a.push(sop1(Op::s_mov_b32, Operand::Reg(decant_shader::decode::Reg::ExecHi), int(-1)));
        a.push(vop2(Op::v_sub_u32, v(2), v(2), v(1)));
        a.push(sop1(Op::s_mov_b64, EXEC, s(2)));
        a.push(vop1(Op::v_mov_b32, v(6), v(2)));
        // Wave64 sum reduction, `s_ff1_i32_b64`, and `v_readfirstlane_b32`.
        a.push(vop3(Op::v_mbcnt_lo_u32_b32, v(8), int(-1), int(0), Operand::None));
        a.push(vop3(Op::v_mbcnt_hi_u32_b32_e64, v(10), int(-1), v(8), Operand::None));
        a.push(sop1(Op::s_mov_b64, s(2), EXEC));
        a.push(vopc(Op::v_cmpx_eq_u32, int(63), v(10)));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop2(Op::v_add_u32, v(8), v(6), v(5)));
        a.push(sop1(Op::s_or_saveexec_b64, s(10), int(-1)));
        a.push(vop3(Op::v_cndmask_b32, v(1), int(0), v(8), s(10)));
        a.push(dpp8(Op::v_add_u32, v(1), v(1), v(1), 0xde54_c100, false));
        a.push(dpp8(Op::v_add_u32, v(1), v(1), v(1), 0x9360_1200, false));
        a.push(dpp8(Op::v_add_u32, v(1), v(1), v(1), 0x0009_2400, false));
        a.push(dpp16(Op::v_add_u32, v(1), v(1), v(1), 0xff09_4001));
        a.push(vop3(Op::v_permlanex16_b32, v(2), v(1), int(0), int(0)));
        a.push(vop2(Op::v_add_u32, v(1), v(2), v(1)));
        a.push(vop3(Op::v_readlane_b32_e64, s(18), v(1), int(32), Operand::None));
        a.push(vop2(Op::v_add_u32, v(1), s(18), v(1)));
        a.push(vop3(Op::v_readlane_b32_e64, s(18), v(1), int(0), Operand::None));
        a.push(sop1(Op::s_mov_b64, EXEC, s(10)));
        a.push(sop1(Op::s_ff1_i32_b64, s(19), EXEC));
        a.push(vop2(Op::v_add_u32, v(8), s(18), v(8)));
        a.push(vop2(Op::v_add_u32, v(8), s(19), v(8)));
        a.push(vop1(Op::v_readfirstlane_b32, s(18), v(8)));
        a.push(vop2(Op::v_add_u32, v(8), s(18), v(8)));
        a.bind(skip);
        a.push(sop1(Op::s_mov_b64, EXEC, s(2)));
        if with_cross_lane_read {
            a.push(vop3(Op::v_readlane_b32_e64, s(2), v(8), int(63), Operand::None));
            a.push(vop2(Op::v_add_u32, v(10), s(2), v(6)));
        }
        a.push(sopp(Op::s_endpgm, 0));
        let mut prog = StageProgram::new(Stage::Compute, a.finish());
        prog.threads = [64, 1, 1];
        prog
    };

    let real = build(true);
    let graph = cfg::build(&real.code).unwrap();
    let masks = exec::analyse(&graph).unwrap();
    assert_eq!(masks.lanes(), 64);
    assert!(masks.real_wave());
    let translated = decant_shader::translate(
        std::slice::from_ref(&real),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap();
    let hlsl = &translated.stages[0].hlsl;
    assert!(hlsl.contains("[WaveSize(64)]"), "{hlsl}");
    assert!(hlsl.contains("WavePrefixSum("), "{hlsl}");
    assert!(hlsl.contains("WaveActiveSum("), "{hlsl}");
    assert!(hlsl.contains("decant_wave_ff1_b64("), "{hlsl}");
    assert!(hlsl.contains("decant_read_first_lane("), "{hlsl}");
    assert!(hlsl.contains("decant_read_lane("), "{hlsl}");
    if let Some(dxc) = common::find_dxc() {
        let compiled = translate_to_dxil(
            &[real],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}"));
        assert_eq!(&compiled[0].dxil[..4], b"DXBC");
    }

    // Without an extra cross lane `v_readlane`, single lane projection stays active.
    let single = build(false);
    let s_graph = cfg::build(&single.code).unwrap();
    let s_masks = exec::analyse(&s_graph).unwrap();
    assert!(!s_masks.real_wave());
    lift_program(&single).unwrap();
}
