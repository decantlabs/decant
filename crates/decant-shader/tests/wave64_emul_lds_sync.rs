//! Integration tests for Wave64 emulation on narrow (Wave32) hardware, Wave64-on-Wave32 LDS
//! auto-synchronization, Wave32 intrinsic fixes, and bounded/OOB-safe CAS atomics.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn, Operand, Reg};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

const NONE: Operand = Operand::None;

fn ds(op: Op, dst: Operand, src: [Operand; 3], offset0: u8, offset1: u8) -> Insn {
    Insn {
        pc: 0,
        len: 0,
        op,
        form: Form::Plain,
        dst,
        dst2: Operand::None,
        src: [src[0], src[1], src[2], Operand::None],
        ext: Ext::Ds {
            offset0,
            offset1,
            gds: false,
        },
    }
}

fn compute(threads: [u32; 3], build: impl FnOnce(&mut Asm)) -> StageProgram {
    let mut a = Asm::new();
    build(&mut a);
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads,
        name: "wave64 emul test".to_owned(),
        ..StageProgram::new(Stage::Compute, a.finish())
    }
}

fn translate_with_mode(p: &StageProgram, wave: WaveMode) -> String {
    let opts = Options {
        wave,
        ..Options::default()
    };
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("refused: {r}"));
    t.stages[0].hlsl.clone()
}

fn compile_with_mode(p: &StageProgram, wave: WaveMode) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("dxc.exe not found, compile check skipped");
        return;
    };
    let opts = Options {
        wave,
        ..Options::default()
    };
    if let Err(r) = translate_to_dxil(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
        &dxc,
    ) {
        panic!("{r}\n{}", translate_with_mode(p, wave));
    }
}

#[test]
fn emulated_wave64_readlane_and_writelane_omit_wavesize64_and_compile() {
    let mut p = compute([8, 8, 1], |a| {
        a.push(sop1(Op::s_and_saveexec_b64, s(10), int(-1)));
        a.push(sop1(Op::s_mov_b32, s(2), int(35)));
        a.push(vop1(Op::v_mov_b32, v(4), v(0)));
        a.push(vop3(Op::v_writelane_b32_e64, v(4), s(2), int(33), NONE));
        a.push(vop3(Op::v_readlane_b32_e64, s(3), v(4), int(33), NONE));
        a.push(vop3(Op::v_readlane_b32_e64, s(4), v(4), int(63), NONE));
        a.push(vop3(Op::v_readlane_b32_e64, s(5), v(4), s(2), NONE));
        a.push(vop2(Op::v_add_u32, v(5), s(3), s(4)));
        a.push(vop2(Op::v_add_u32, v(5), v(5), s(5)));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(5), NONE], 0, 0));
    });
    p.lds_bytes = 256;

    let native = translate_with_mode(&p, WaveMode::Native { lanes: 64 });
    assert!(native.contains("[WaveSize(64)]"), "{native}");

    let emul = translate_with_mode(&p, WaveMode::Emulated);
    assert!(!emul.contains("[WaveSize(64)]"), "{emul}");
    assert!(!emul.contains("WaveReadLaneAt("), "{emul}");
    assert!(emul.contains("groupshared uint decant_wave_lds[64];"), "{emul}");
    assert!(emul.contains("decant_emul_read_lane(decant_gi,"), "{emul}");
    assert!(emul.contains("decant_emul_write_lane(decant_gi,"), "{emul}");
    compile_with_mode(&p, WaveMode::Emulated);
}

fn exclusive_scan() -> Vec<Insn> {
    let hi = Operand::Reg(Reg::ExecHi);
    let mut pl = vop3(Op::v_permlanex16_b32, v(8), v(7), int(-1), int(-1));
    pl.ext = Ext::Valu {
        abs: 0,
        neg: 0,
        opsel: 9,
        omod: 0,
        clamp: false,
    };
    let mut code = vec![
        sop1(Op::s_or_saveexec_b64, s(20), int(-1)),
        vop3(Op::v_cndmask_b32, v(6), int(0), v(0), s(20)),
        vop1(Op::v_mov_b32, v(7), v(6)),
    ];
    for control in [0xff09_1100, 0xff01_1200, 0xff01_1400, 0xff01_1800] {
        code.push(dpp16(Op::v_add_u32, v(7), v(7), v(7), control));
    }
    code.extend([
        sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)),
        sop1(Op::s_mov_b32, hi, lit(0xffff_0000)),
        pl,
        vop2(Op::v_add_u32, v(7), v(8), v(7)),
        vop3(Op::v_readlane_b32_e64, s(22), v(7), int(31), NONE),
        sop1(Op::s_mov_b32, EXEC, int(0)),
        sop1(Op::s_mov_b32, hi, int(-1)),
        vop2(Op::v_add_u32, v(7), s(22), v(7)),
        sop1(Op::s_mov_b32, EXEC, int(-1)),
        sop1(Op::s_mov_b32, hi, int(-1)),
        vop2(Op::v_sub_u32, v(7), v(7), v(6)),
        sop1(Op::s_mov_b64, EXEC, s(20)),
    ]);
    code
}

#[test]
fn emulated_wave64_ballots_prefix_sum_and_readfirstlane_use_groupshared_and_compile() {
    let mut p = compute([64, 1, 1], |a| {
        // Matched Wave64 exclusive scan + fixed lane 63 read activates real-wave mode.
        for i in exclusive_scan() {
            a.push(i);
        }
        a.push(vop3(Op::v_readlane_b32_e64, s(30), v(7), int(63), NONE));
        a.push(vopc(Op::v_cmp_lt_u32, v(0), int(40)));
        a.push(sop1(Op::s_and_saveexec_b64, s(12), VCC));
        a.push(sop1(Op::s_bcnt1_i32_b64, s(2), EXEC));
        a.push(sop1(Op::s_bcnt0_i32_b64, s(3), EXEC));
        a.push(sop1(Op::s_ff1_i32_b64, s(4), EXEC));
        a.push(sop1(Op::s_ff0_i32_b64, s(5), EXEC));
        a.push(sop1(Op::s_flbit_i32_b64, s(6), EXEC));
        a.push(sop2(Op::s_bitcmp1_b64, NONE, EXEC, int(37)));
        a.push(sop2(Op::s_cselect_b32, s(7), int(1), int(0)));
        a.push(vop1(Op::v_readfirstlane_b32, s(8), v(0)));
        a.push(sop1(Op::s_mov_b64, EXEC, s(12)));
        a.push(sop1(Op::s_mov_b32, s(9), Operand::Reg(Reg::ExecHi)));
        a.push(vop1(Op::v_mov_b32, v(1), s(30)));
        a.push(vop2(Op::v_add_u32, v(1), s(2), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(3), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(4), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(5), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(6), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(7), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(8), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), s(9), v(1)));
        a.push(vop2(Op::v_add_u32, v(1), v(7), v(1)));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(1), NONE], 0, 0));
    });
    p.lds_bytes = 256;

    // Also verify native Wave32 fixes in the native helpers.
    let native = translate_with_mode(&p, WaveMode::Scalar);
    assert!(native.contains("WaveGetLaneCount() > 32u ? ~b.y : 0u"), "{native}");
    assert!(native.contains("WaveGetLaneCount() - (countbits(b.x) + countbits(b.y))"), "{native}");
    assert!(native.contains("(WaveGetLaneCount() - 1u) - firstbithigh(b.x)"), "{native}");

    let emul = translate_with_mode(&p, WaveMode::Emulated);
    assert!(!emul.contains("[WaveSize(64)]"), "{emul}");
    assert!(emul.contains("groupshared uint decant_wave_mask[2];"), "{emul}");
    assert!(emul.contains("groupshared uint decant_wave_lds[64];"), "{emul}");
    assert!(emul.contains("decant_emul_ballot64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_prefix_sum_u32("), "{emul}");
    assert!(emul.contains("decant_emul_wave_bcnt1_b64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_bcnt0_b64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_ff1_b64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_ff0_b64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_flbit_b64("), "{emul}");
    assert!(emul.contains("decant_emul_wave_bitcmp_b64("), "{emul}");
    assert!(emul.contains("decant_emul_read_first_lane("), "{emul}");
    assert!(emul.contains("decant_emul_mask_word("), "{emul}");
    compile_with_mode(&p, WaveMode::Emulated);
}

#[test]
fn emulated_wave64_dpp_permlane_and_bpermute_never_emit_wavesize64() {
    let mut p = compute([64, 1, 1], |a| {
        // Asymmetric row_mask 0xc (rows 2 and 3): previously forced [WaveSize(64)].
        a.push(dpp16(Op::v_mov_b32, v(4), v(0), NONE, 0xcf09_1100));
        a.push(dpp16(Op::v_add_u32, v(5), v(4), v(0), 0xcf01_1100));
        let mut pl = vop3(Op::v_permlanex16_b32, v(6), v(5), s(4), s(5));
        pl.ext = Ext::Valu {
            abs: 0,
            neg: 0,
            opsel: 3,
            omod: 0,
            clamp: false,
        };
        a.push(pl);
        a.push(ds(Op::ds_bpermute_b32, v(7), [v(0), v(6), NONE], 0, 0));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(7), NONE], 0, 0));
    });
    p.lds_bytes = 256;

    // Native DPP_ADD also has the `source < WaveGetLaneCount()` guard.
    let native = translate_with_mode(&p, WaveMode::Scalar);
    assert!(native.contains("[WaveSize(64)]"), "{native}");
    assert!(native.contains("bool inWave = inRange && source < WaveGetLaneCount();"), "{native}");

    let emul = translate_with_mode(&p, WaveMode::Emulated);
    assert!(!emul.contains("[WaveSize(64)]"), "{emul}");
    assert!(emul.contains("[WaveSize(32)]"), "{emul}");
    assert!(emul.contains("decant_emul_dpp_mov(decant_gi,"), "{emul}");
    assert!(emul.contains("decant_emul_dpp_add(decant_gi,"), "{emul}");
    assert!(emul.contains("decant_emul_permlane("), "{emul}");
    assert!(emul.contains("decant_emul_bpermute(decant_gi,"), "{emul}");
    compile_with_mode(&p, WaveMode::Emulated);
}

#[test]
fn emulated_wave64_flattens_divergent_exec_skip_around_readlane() {
    let mut p = compute([64, 1, 1], |a| {
        let skip = a.label();
        a.push(vopc(Op::v_cmp_lt_u32, v(0), int(16)));
        a.push(sop1(Op::s_and_saveexec_b64, s(12), VCC));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop3(Op::v_readlane_b32_e64, s(3), v(0), int(40), NONE));
        a.push(vop2(Op::v_add_u32, v(4), s(3), v(0)));
        a.bind(skip);
        a.push(sop1(Op::s_mov_b64, EXEC, s(12)));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
    });
    p.lds_bytes = 256;

    let emul = translate_with_mode(&p, WaveMode::Emulated);
    assert!(!emul.contains("[WaveSize(64)]"), "{emul}");
    // The emulated read_lane call must be hoisted to top level (4-space indent, never nested under if).
    let rl_line = emul
        .lines()
        .find(|l| l.contains("= decant_emul_read_lane("))
        .unwrap_or_else(|| panic!("missing decant_emul_read_lane call:\n{emul}"));
    assert!(
        rl_line.starts_with("    ") && !rl_line.starts_with("        "),
        "expected top-level converged emulated read_lane, got: {rl_line:?}\n{emul}"
    );
    compile_with_mode(&p, WaveMode::Emulated);
}

#[test]
fn lds_auto_sync_inserts_converged_barriers_between_write_atomic_and_read_phases() {
    // Models the Wave64 lockstep pattern (lane 0 init -> all-lane LDS atomics -> all-lane LDS read)
    // with zero explicit `s_barrier` instructions.
    let mut p = compute([64, 1, 1], |a| {
        let after_init = a.label();
        a.push(vopc(Op::v_cmp_eq_u32, v(0), int(0)));
        a.push(sop1(Op::s_and_saveexec_b64, s(10), VCC));
        a.branch(Op::s_cbranch_execz, after_init);
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(1), NONE], 0, 0));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(1), NONE], 4, 0));
        a.bind(after_init);
        a.push(sop1(Op::s_mov_b64, EXEC, s(10)));
        a.push(vop1(Op::v_mov_b32, v(2), int(0)));
        a.push(ds(Op::ds_or_b32, NONE, [v(2), v(0), NONE], 0, 0));
        a.push(ds(Op::ds_min_u32, NONE, [v(2), v(0), NONE], 4, 0));
        a.push(ds(Op::ds_read_b32, v(3), [v(2), NONE, NONE], 0, 0));
        a.push(ds(Op::ds_read_b32, v(4), [v(2), NONE, NONE], 4, 0));
        a.push(vop2(Op::v_add_u32, v(5), v(3), v(4)));
        a.push(ds(Op::ds_write_b32, NONE, [v(2), v(5), NONE], 8, 0));
    });
    p.lds_bytes = 16;

    // Native Wave64 does not need extra workgroup barriers.
    let native64 = translate_with_mode(&p, WaveMode::Native { lanes: 64 });
    assert_eq!(native64.matches("decant_barrier();").count(), 0, "{native64}");

    // Emulated (Wave32 hardware) inserts top-level converged barriers between W->A, A->R, R->W.
    let emul = translate_with_mode(&p, WaveMode::Emulated);
    assert!(emul.matches("decant_barrier();").count() >= 3, "{emul}");
    for line in emul.lines().filter(|l| l.contains("decant_barrier();")) {
        assert!(
            line.starts_with("    decant_barrier();"),
            "auto-inserted barrier must be top-level converged, got {line:?}\n{emul}"
        );
    }
    compile_with_mode(&p, WaveMode::Emulated);

    // Groups with <= 32 threads fit in a single Wave32 and need no auto-inserted barriers.
    let mut small = p.clone();
    small.threads = [32, 1, 1];
    let small_emul = translate_with_mode(&small, WaveMode::Emulated);
    assert_eq!(small_emul.matches("decant_barrier();").count(), 0, "{small_emul}");
}

#[test]
fn lds_auto_sync_never_inserts_barriers_after_early_return_or_inside_divergent_branch() {
    let mut p = compute([64, 1, 1], |a| {
        let early = a.label();
        a.push(vopc(Op::v_cmp_eq_u32, v(0), int(0)));
        a.push(sop1(Op::s_and_saveexec_b64, s(10), VCC));
        a.branch(Op::s_cbranch_execz, early);
        a.push(sopp(Op::s_endpgm, 0));
        a.bind(early);
        a.push(sop1(Op::s_mov_b64, EXEC, s(10)));
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(0), NONE], 0, 0));
        a.push(ds(Op::ds_read_b32, v(2), [v(1), NONE, NONE], 0, 0));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(2), NONE], 4, 0));
    });
    p.lds_bytes = 16;

    let emul = translate_with_mode(&p, WaveMode::Emulated);
    // An early return precedes all LDS ops, so control flow is non-converged: 0 barriers inserted.
    assert_eq!(emul.matches("decant_barrier();").count(), 0, "{emul}");
    compile_with_mode(&p, WaveMode::Emulated);
}

#[test]
fn lds_and_buffer_cas_helpers_are_bounded_and_oob_cmpst_returns_compare() {
    // LDS CAS (`ds_max_i32`, `ds_inc_u32`, `ds_cmpst_rtn_b32`).
    let mut p = compute([64, 1, 1], |a| {
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(vop1(Op::v_mov_b32, v(2), int(7)));
        a.push(vop1(Op::v_mov_b32, v(3), int(9)));
        a.push(ds(Op::ds_max_i32, NONE, [v(1), v(2), NONE], 0, 0));
        a.push(ds(Op::ds_inc_u32, NONE, [v(1), v(2), NONE], 4, 0));
        a.push(ds(Op::ds_cmpst_rtn_b32, v(4), [v(1), v(2), v(3)], 8, 0));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(4), NONE], 12, 0));
    });
    p.lds_bytes = 32;
    let text = translate_with_mode(&p, WaveMode::Emulated);
    assert!(!text.contains("for (;;)"), "{text}");
    assert!(text.contains("decant_cas < 1024u"), "{text}");
    assert!(
        text.contains("uint decant_lds_cmpst(uint a, uint v, uint w) { uint i = a >> 2; uint o = v;"),
        "{text}"
    );
    compile_with_mode(&p, WaveMode::Emulated);
}

#[test]
fn lds_auto_sync_handles_uniform_loop_war_and_buffer_cas_is_bounded() {
    // 1. Uniform loop with LDS read followed by LDS write (WAR hazard across iterations).
    let mut loop_prog = compute([64, 1, 1], |a| {
        let header = a.label();
        let exit = a.label();
        a.push(sop1(Op::s_mov_b32, s(6), int(0)));
        a.bind(header);
        a.push(sopc(Op::s_cmp_ge_u32, s(6), int(4)));
        a.branch(Op::s_cbranch_scc1, exit);
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(ds(Op::ds_read_b32, v(2), [v(1), NONE, NONE], 0, 0));
        a.push(vop2(Op::v_add_u32, v(2), v(2), v(0)));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(2), NONE], 0, 0));
        a.push(sop2(Op::s_add_u32, s(6), s(6), int(1)));
        a.branch(Op::s_branch, header);
        a.bind(exit);
    });
    loop_prog.lds_bytes = 16;
    let emul_loop = translate_with_mode(&loop_prog, WaveMode::Emulated);
    // Needs both R->W barrier inside the loop and W->R back-edge barrier before next iteration.
    assert!(emul_loop.matches("decant_barrier();").count() >= 2, "{emul_loop}");
    compile_with_mode(&loop_prog, WaveMode::Emulated);

    // 2. Buffer atomic Inc (bounded CAS) + CmpSwap (GetDimensions guard + OOB compare return).
    let buf_atomic = |op: Op, vdata: u8| {
        let mut i = buffer_load(op, v(vdata), v(0), s(8), 0, false);
        if let Ext::Buf { offen, glc, .. } = &mut i.ext {
            *offen = true;
            *glc = true;
        }
        i
    };
    let mut buf_prog = compute([64, 1, 1], |a| {
        a.push(sop1(Op::s_mov_b32, s(5), int(0)));
        a.push(smem(Op::s_load_dwordx4, s(8), s(4), 0));
        a.push(vop1(Op::v_mov_b32, v(4), int(10)));
        a.push(vop1(Op::v_mov_b32, v(6), int(2)));
        a.push(vop1(Op::v_mov_b32, v(7), int(5)));
        a.push(buf_atomic(Op::buffer_atomic_inc, 4));
        a.push(buf_atomic(Op::buffer_atomic_cmpswap, 6));
    });
    buf_prog.user_data = vec![UserDataSlot {
        sgpr: 4,
        kind: UserDataKind::Table,
    }];
    buf_prog.resources = vec![DeclaredResource {
        class: RegisterClass::Uav,
        lower: 0,
        upper: 0,
        space: 0,
        psv_type: 7,
        psv_kind: 11,
    }];
    buf_prog.binding_table = vec![TableBinding {
        class: RegisterClass::Uav,
        register: 0,
        sgpr: 4,
        word_offset: 0,
    }];
    let root = RootSignature {
        params: vec![RootParam {
            kind: RootParamKind::Table(vec![RootRange {
                class: RegisterClass::Uav,
                count: 1,
                base_register: 0,
                space: 0,
                offset: 0,
            }]),
            visibility: 0,
        }],
        static_samplers: Vec::new(),
        flags: 0xC00,
    };
    let opts = Options {
        wave: WaveMode::Emulated,
        ..Options::default()
    };
    let tr = translate(
        std::slice::from_ref(&buf_prog),
        &root,
        &opts,
        &CountingHost::default(),
    )
    .expect("translate buffer CAS");
    let hlsl = &tr.stages[0].hlsl;
    assert!(!hlsl.contains("while (true)"), "{hlsl}");
    assert!(hlsl.contains("< 1024u; ++decant_cas"), "{hlsl}");
    assert!(hlsl.contains(".GetDimensions(decant_dim"), "{hlsl}");
    if let Some(dxc) = common::find_dxc() {
        translate_to_dxil(
            std::slice::from_ref(&buf_prog),
            &root,
            &opts,
            &CountingHost::default(),
            &dxc,
        )
        .expect("compile buffer CAS");
    }
}

#[test]
fn lds_auto_sync_handles_read_to_atomic_war_and_uniform_if_branch() {
    // A workgroup-uniform `if` (branching on group id `s4`) that does LDS load -> LDS atomic add
    // receives a converged barrier between the read and the modifying atomic (WAR hazard).
    let mut prog = compute([64, 1, 1], |a| {
        let skip = a.label();
        a.push(sop1(Op::s_mov_b64, s(10), EXEC));
        a.push(sopc(Op::s_cmp_eq_u32, s(4), int(0)));
        a.branch(Op::s_cbranch_scc1, skip);
        a.push(vop1(Op::v_mov_b32, v(4), int(0)));
        a.push(ds(Op::ds_read_b32, v(6), [v(4), NONE, NONE], 0, 0));
        a.push(ds(Op::ds_add_u32, NONE, [v(4), v(6), NONE], 0, 0));
        a.bind(skip);
    });
    prog.lds_bytes = 64;

    let hlsl = translate_with_mode(&prog, WaveMode::Emulated);
    assert!(!hlsl.contains("[WaveSize(64)]"), "{hlsl}");
    assert!(
        hlsl.contains("decant_barrier();"),
        "expected read->atomic WAR barrier in uniform if:\n{hlsl}"
    );
    compile_with_mode(&prog, WaveMode::Emulated);
}

#[test]
#[ignore = "requires DECANT_TEST_WAVE64_BIN pointing to a Wave64 compute container and DXC"]
fn optional_external_wave64_container_emulates_without_wavesize64() {
    let path = std::env::var_os("DECANT_TEST_WAVE64_BIN")
        .expect("set DECANT_TEST_WAVE64_BIN to a compute container path");
    let bytes = std::fs::read(&path).expect("read container");
    let program = decant_shader::container::program(&bytes, "external").expect("parse container");
    let dxc = common::find_dxc().expect("load DXC");
    let opts = Options {
        wave: WaveMode::Emulated,
        ..Options::default()
    };
    let translated = translate(
        std::slice::from_ref(&program),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
    )
    .expect("translate under WaveMode::Emulated");
    let hlsl = &translated.stages[0].hlsl;
    assert!(!hlsl.contains("[WaveSize(64)]"), "{hlsl}");
    eprintln!(
        "emulated external container: threads={:?} wave32={} barriers={} emul_helpers={}",
        program.threads,
        hlsl.contains("[WaveSize(32)]"),
        hlsl.matches("decant_barrier()").count(),
        hlsl.matches("decant_emul_").count(),
    );
    let compiled = translate_to_dxil(
        std::slice::from_ref(&program),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
        &dxc,
    )
    .expect("compile under WaveMode::Emulated");
    assert!(!compiled[0].dxil.is_empty());
}
