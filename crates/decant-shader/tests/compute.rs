//! The compute stage: entry registers (thread id, group id after the user data), group shared
//! memory and barriers. Programs are built with the assembler, no game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn, Operand, Reg};
use decant_shader::ir::{Expr, Guard, Stmt};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

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

const NONE: Operand = Operand::None;

fn compute(
    threads: [u32; 3],
    user_data: Vec<UserDataSlot>,
    build: impl FnOnce(&mut Asm),
) -> StageProgram {
    let mut a = Asm::new();
    build(&mut a);
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads,
        user_data,
        name: "compute test".to_owned(),
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

fn compiles(p: &StageProgram) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("dxc.exe not found, compile check skipped");
        return;
    };
    if let Err(r) = translate_to_dxil(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    ) {
        panic!("{r}\n{}", hlsl(p));
    }
}

fn table(sgpr: u8) -> UserDataSlot {
    UserDataSlot {
        sgpr,
        kind: UserDataKind::Table,
    }
}

#[test]
fn an_empty_program_translates_with_its_group_size() {
    let p = compute([64, 1, 1], Vec::new(), |_| {});
    let text = hlsl(&p);
    assert!(text.contains("numthreads(64, 1, 1)"), "{text}");
    compiles(&p);
}

#[test]
fn lane_accesses_compile_with_the_console_wave_size() {
    for (lanes, save_exec) in [(32, Op::s_and_saveexec_b32), (64, Op::s_and_saveexec_b64)] {
        let mut p = compute([lanes, 1, 1], Vec::new(), |a| {
            a.push(sop1(save_exec, s(10), int(-1)));
            a.push(sop1(Op::s_mov_b32, s(2), int(7)));
            a.push(vop1(Op::v_mov_b32, v(4), v(0)));
            a.push(vop3(Op::v_writelane_b32_e64, v(4), s(2), int(3), NONE));
            a.push(vop3(Op::v_readlane_b32_e64, s(3), v(4), int(3), NONE));
            a.push(vop3(Op::v_readlane_b32_e64, s(4), v(4), s(2), NONE));
            a.push(vop1(Op::v_mov_b32, v(5), s(3)));
            a.push(vop1(Op::v_mov_b32, v(6), s(4)));
            a.push(ds(Op::ds_write_b32, NONE, [v(0), v(5), NONE], 0, 0));
            a.push(ds(Op::ds_write_b32, NONE, [v(0), v(6), NONE], 4, 0));
        });
        p.lds_bytes = 128;
        let text = hlsl(&p);
        assert!(text.contains(&format!("[WaveSize({lanes})]")), "{text}");
        assert!(text.contains("WaveReadLaneAt(value, lane)"), "{text}");
        assert!(
            text.contains("WaveGetLaneIndex() == lane ? value : old"),
            "{text}"
        );
        assert_eq!(text.matches("decant_read_lane(").count(), 3, "{text}");
        compiles(&p);
    }
}

#[test]
#[ignore = "requires DECANT_SHADER_CORPUS and DXC"]
fn corpus_lane_reads_compile() {
    let path = std::env::var_os("DECANT_SHADER_CORPUS").expect("set DECANT_SHADER_CORPUS to a shader archive");
    let selection = decant_shader::corpus::Selection {
        hashes: Some([0xdf23_1d75_ee93_0b98].into()),
        ..Default::default()
    };
    let (_, parts) = decant_shader::corpus::par_corpus(
        std::path::Path::new(&path),
        &selection,
        1,
        Vec::new,
        |programs, entry| programs.push(entry.program.expect("decode the compute shader")),
    )
    .expect("open the shader archive");
    let programs: Vec<_> = parts.into_iter().flatten().collect();
    assert_eq!(programs.len(), 1, "find the shader by its container hash");
    let dxc = common::find_dxc().expect("load DXC");
    let compiled = translate_to_dxil(
        &programs,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .expect("translate and compile df231d75ee930b98");
    assert!(!compiled[0].dxil.is_empty());
}

#[test]
fn a_bitmask_swizzle_exchanges_group_values_outside_program_lds() {
    let mut p = compute([8, 8, 1], Vec::new(), |a| {
        a.push(ds(Op::ds_swizzle_b32, v(4), [v(0), NONE, NONE], 31, 4));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
    });
    p.lds_bytes = 128;
    let text = hlsl(&p);
    assert!(text.contains("groupshared uint decant_lds[96];"), "{text}");
    assert!(text.contains("decant_lds_words = 32;"), "{text}");
    assert!(text.contains("decant_swizzle(decant_gi, 1055u,"), "{text}");
    assert_eq!(text.matches("GroupMemoryBarrierWithGroupSync()").count(), 2);
    compiles(&p);
}

#[test]
fn a_swizzle_inside_a_branch_uses_wave_intrinsics_without_group_barriers() {
    let p = compute([64, 1, 1], Vec::new(), |a| {
        let done = a.label();
        a.push(vopc(Op::v_cmpx_eq_u32, int(0), v(0)));
        a.branch(Op::s_cbranch_execz, done);
        a.push(ds(Op::ds_swizzle_b32, v(4), [v(0), NONE, NONE], 31, 4));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
        a.bind(done);
    });
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_swizzle_wave(1055u,"), "{text}");
    assert!(text.contains("WaveActiveBallot(exec)"), "{text}");
    assert!(
        text.contains("WaveReadLaneAt(value, valid ? source : lane)"),
        "{text}"
    );
    assert!(
        !text.contains("GroupMemoryBarrierWithGroupSync()"),
        "{text}"
    );
    assert!(!text.contains("decant_swizzle_threads"), "{text}");
    compiles(&p);
}

#[test]
fn divergent_quad_and_bitmask_swizzles_across_shapes_compile_with_dxc() {
    // 1. Wave32 compute with early exit + quad derivative swizzles (0x8000, 0x8055, 0x80aa)
    //    and in-place aliasing (`v4 -> v4`), matching raytracing resolve shaders.
    let p = compute([32, 1, 1], Vec::new(), |a| {
        let exit = a.label();
        a.push(sop1(Op::s_mov_b32, s(20), EXEC));
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(24)));
        a.branch(Op::s_cbranch_execz, exit);
        a.push(ds(Op::ds_swizzle_b32, v(5), [v(4), NONE, NONE], 0, 128));
        a.push(ds(Op::ds_swizzle_b32, v(6), [v(4), NONE, NONE], 85, 128));
        a.push(ds(Op::ds_swizzle_b32, v(4), [v(4), NONE, NONE], 170, 128));
        a.push(vop2(Op::v_add_u32, v(4), v(5), v(4)));
        a.push(vop2(Op::v_add_u32, v(4), v(6), v(4)));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
        a.bind(exit);
    });
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_swizzle_wave(32768u,"), "{text}");
    assert!(text.contains("decant_swizzle_wave(32853u,"), "{text}");
    assert!(text.contains("decant_swizzle_wave(32938u,"), "{text}");
    assert!(
        !text.contains("GroupMemoryBarrierWithGroupSync()"),
        "{text}"
    );
    compiles(&p);

    // 2. Wave64 2D threadgroup with a swizzle after a loop (the loop marks control flow
    //    potentially non-converged).
    let p = compute([8, 8, 1], Vec::new(), |a| {
        let header = a.label();
        a.push(sop1(Op::s_mov_b64, s(20), EXEC));
        a.push(vop1(Op::v_mov_b32, v(4), v(0)));
        a.bind(header);
        a.push(vop2(Op::v_add_u32, v(4), int(1), v(4)));
        a.push(vopc(Op::v_cmp_lt_u32, v(4), int(4)));
        a.branch(Op::s_cbranch_vccnz, header);
        a.push(ds(Op::ds_swizzle_b32, v(5), [v(4), NONE, NONE], 31, 32));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(5), NONE], 0, 0));
    });
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_swizzle_wave(8223u,"), "{text}");
    assert!(
        !text.contains("GroupMemoryBarrierWithGroupSync()"),
        "{text}"
    );
    compiles(&p);

    // 3. Wave64 with both `v_readlane_b32` and a divergent swizzle keeps `[WaveSize(64)]`.
    let p = compute([64, 1, 1], Vec::new(), |a| {
        let done = a.label();
        a.push(sop1(Op::s_and_saveexec_b64, s(20), int(-1)));
        a.push(vop3(Op::v_readlane_b32_e64, s(2), v(0), int(33), NONE));
        a.push(vopc(Op::v_cmpx_gt_u32, s(2), v(0)));
        a.branch(Op::s_cbranch_execz, done);
        a.push(ds(Op::ds_swizzle_b32, v(4), [v(0), NONE, NONE], 31, 4));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
        a.bind(done);
    });
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(64)]"), "{text}");
    assert!(text.contains("decant_swizzle_wave(1055u,"), "{text}");
    compiles(&p);

    // 4. Partial threadgroup `[3, 2, 1]` under a branch compiles and does not touch LDS scratch.
    let p = compute([3, 2, 1], Vec::new(), |a| {
        let done = a.label();
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(2)));
        a.branch(Op::s_cbranch_execz, done);
        a.push(ds(Op::ds_swizzle_b32, v(4), [v(0), NONE, NONE], 0x1b, 0x80));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
        a.bind(done);
    });
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_swizzle_wave(32795u,"), "{text}");
    compiles(&p);

    // 5. Shader Model < 6 refuses `[WaveSize(32)]` cleanly.
    let opts = Options {
        shader_model_minor: 5,
        ..Options::default()
    };
    let err = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(err.key(), "stage:dpp-shader-model");
}

#[test]
fn a_program_without_a_group_size_is_refused() {
    let p = compute([0, 0, 0], Vec::new(), |_| {});
    let r = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(r.key(), "stage:compute-threads-unknown");
}

#[test]
fn the_group_id_follows_the_user_data() {
    // Two tables in s4 and s5, four root constants in s6 to s9: the group id x is s10 and y s11.
    let user = vec![
        table(4),
        table(5),
        UserDataSlot {
            sgpr: 6,
            kind: UserDataKind::Constants {
                register: 0,
                count: 4,
            },
        },
    ];
    let p = compute([8, 8, 1], user, |a| {
        // lds[(gid.x + gid.y + tid.x + tid.y) * 4] = 1
        a.push(vop2(Op::v_add_u32, v(3), s(10), v(0)));
        a.push(vop2(Op::v_add_u32, v(3), s(11), v(3)));
        a.push(vop2(Op::v_add_u32, v(3), v(1), v(3)));
        a.push(vop2(Op::v_lshlrev_b32, v(3), int(2), v(3)));
        a.push(vop1(Op::v_mov_b32, v(4), int(1)));
        a.push(ds(Op::ds_write_b32, NONE, [v(3), v(4), NONE], 0, 0));
    });
    let text = hlsl(&p);
    for want in [
        "SV_GroupThreadID",
        "SV_GroupID",
        "groupshared uint decant_lds[",
        "decant_lds_store(",
    ] {
        assert!(text.contains(want), "{want} missing:\n{text}");
    }
    compiles(&p);
}

#[test]
fn shared_memory_round_trips_through_a_barrier() {
    let p = compute([64, 1, 1], Vec::new(), |a| {
        a.push(vop2(Op::v_lshlrev_b32, v(1), int(2), v(0)));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(0), NONE], 0, 0));
        // Two neighbours at once: dwords 1 and 2 after this lane's own.
        a.push(ds(Op::ds_read2_b32, v(2), [v(1), NONE, NONE], 1, 2));
        a.push(vop2(Op::v_add_u32, v(2), v(2), v(3)));
        a.push(ds(Op::ds_write2st64_b32, NONE, [v(1), v(2), v(3)], 1, 2));
        a.push(ds(Op::ds_write_b16, NONE, [v(1), v(2), NONE], 0x40, 0));
        a.push(ds(
            Op::ds_read_u16_d16_hi,
            v(5),
            [v(1), NONE, NONE],
            0x42,
            0,
        ));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(5), NONE], 0, 1));
    });
    let text = hlsl(&p);
    // The two halves of the second read: offset1 * 4 bytes.
    assert!(text.contains("8u"), "{text}");
    compiles(&p);
}

#[test]
fn atomics_are_kept_even_when_nothing_reads_their_result() {
    let p = compute([32, 1, 1], Vec::new(), |a| {
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
        a.push(vop1(Op::v_mov_b32, v(2), int(1)));
        a.push(ds(Op::ds_add_u32, NONE, [v(1), v(2), NONE], 0, 0));
        a.push(ds(Op::ds_max_i32, NONE, [v(1), v(0), NONE], 4, 0));
        a.push(ds(Op::ds_add_rtn_u32, v(3), [v(1), v(2), NONE], 8, 0));
        a.push(ds(Op::ds_cmpst_rtn_b32, v(4), [v(1), v(3), v(2)], 12, 0));
        a.push(ds(Op::ds_or_b32, NONE, [v(1), v(4), NONE], 16, 0));
    });
    let text = hlsl(&p);
    for want in [
        "decant_lds_add(",
        "decant_lds_smax(",
        "decant_lds_cmpst(",
        "decant_lds_or(",
    ] {
        assert!(text.contains(want), "{want} missing:\n{text}");
    }
    compiles(&p);
}

#[test]
fn global_data_share_is_refused_by_form() {
    let p = compute([32, 1, 1], Vec::new(), |a| {
        let mut i = ds(Op::ds_add_u32, NONE, [v(0), v(0), NONE], 0, 0);
        i.ext = Ext::Ds {
            offset0: 0,
            offset1: 0,
            gds: true,
        };
        a.push(i);
    });
    let r = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(r.key(), "form:ds_add_u32:gds");
}

#[test]
fn a_barrier_becomes_a_group_sync_for_every_lane() {
    let p = compute([64, 1, 1], Vec::new(), |a| {
        a.push(vop1(Op::v_mov_b32, v(1), int(0)));
    });
    let mut func = decant_shader::lift_program(&p).unwrap();
    // The lane rule file of the mesh package lifts `s_barrier` into this raw statement.
    func.for_each_block_mut(&mut |stmts| stmts.push(Stmt::Barrier { pc: 6 }));
    decant_shader::stage::compute::finish(&p, &mut func).unwrap();
    let mut syncs = 0;
    func.for_each_stmt(&mut |s| match s {
        Stmt::Barrier { .. } => panic!("barrier left"),
        Stmt::Effect {
            dst: None,
            call: Expr::Call(h, _),
            guard: Guard::Always,
        } if h.name == "decant_barrier" => syncs += 1,
        _ => {}
    });
    assert!(syncs >= 1);
    assert!(
        decant_shader::lift::ds::BARRIER
            .text
            .contains("GroupMemoryBarrierWithGroupSync")
    );
}

#[test]
fn the_array_has_the_size_the_container_gives() {
    let mut p = compute([64, 1, 1], Vec::new(), |a| {
        a.push(vop2(Op::v_lshlrev_b32, v(1), int(2), v(0)));
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(0), NONE], 0, 0));
    });
    p.lds_bytes = 258;
    let text = hlsl(&p);
    assert!(text.contains("groupshared uint decant_lds[65];"), "{text}");
    p.lds_bytes = 0;
    let text = hlsl(&p);
    assert!(text.contains("groupshared uint decant_lds[8192];"), "{text}");
}

#[test]
fn dpp_add_emits_native_row_controls_and_masks() {
    let controls = [
        (0x1b, "(ctrl >> (2u * (lane & 3u))) & 3u"),
        (0x103, "source = lane + delta;"),
        (0x113, "source = lane - delta;"),
        (0x123, "source = row | ((col - delta) & 15u);"),
        (0x140, "source = lane ^ 15u;"),
        (0x141, "source = lane ^ 7u;"),
        (0x154, "source = row | delta;"),
        (0x164, "source = lane ^ delta;"),
    ];
    let mut p = compute([64, 1, 1], Vec::new(), |a| {
        for (i, (ctrl, _)) in controls.iter().enumerate() {
            let dst = v(4 + i as u8);
            let bound = (i as u32 & 1) << 19;
            a.push(dpp16(
                Op::v_add_u32,
                dst,
                v(0),
                v(0),
                0xa500_0000 | (ctrl << 8) | bound,
            ));
            a.push(ds(
                Op::ds_write_b32,
                NONE,
                [v(0), dst, NONE],
                (4 * i) as u8,
                0,
            ));
        }
    });
    p.lds_bytes = 512;
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(64)]"), "{text}");
    assert!(text.contains("WaveActiveBallot(exec)"), "{text}");
    assert!(
        text.contains("WaveReadLaneAt(a, valid ? safeSource : lane)"),
        "{text}"
    );
    assert!(
        text.contains("return write && (valid || bound) ? sum : old;"),
        "{text}"
    );
    for (i, (ctrl, source)) in controls.into_iter().enumerate() {
        assert!(text.contains(source), "{text}");
        let args = format!(", {ctrl}u, {}, 10u, 5u, false)", i % 2 != 0);
        assert!(text.contains(&args), "{text}");
    }
    compiles(&p);
}

#[test]
fn constant_exec_halves_compile_with_row_reduction_patterns() {
    use decant_shader::decode::Reg;

    for half in [Reg::ExecLo, Reg::ExecHi] {
        for value in [0, u32::MAX, 0xffff_0000, 0x8000_0001] {
            let mut p = compute([64, 1, 1], Vec::new(), |a| {
                a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
                a.push(vop1(Op::v_mov_b32, v(4), v(0)));
                a.push(sop1(Op::s_mov_b32, Operand::Reg(half), lit(value)));
                a.push(vop2(Op::v_add_u32, v(4), int(1), v(4)));
                a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
                a.push(vop2(Op::v_cndmask_b32, v(4), int(0), v(4)));
                // A reduction across each row, followed by selecting a half.
                for control in [0xb1, 0x4e, 0x141, 0x140] {
                    a.push(dpp16(
                        Op::v_add_u32,
                        v(4),
                        v(4),
                        v(4),
                        0xff08_0000 | (control << 8),
                    ));
                }
                a.push(sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)));
                a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
            });
            p.lds_bytes = 512;
            compiles(&p);
        }
    }
}

#[test]
fn dpp_add_refuses_unsupported_controls_precisely() {
    for (word, reason) in [
        (0x13000, "dpp-wave-shl"),
        (0x13400, "dpp-wave-rol"),
        (0x13800, "dpp-wave-shr"),
        (0x13c00, "dpp-wave-ror"),
        (0x14200, "dpp-row-bcast15"),
        (0x14300, "dpp-row-bcast31"),
        (0x17000, "dpp-reserved-control"),
        (0x40000, "dpp-fetch-inactive"),
        (0x20000, "dpp-reserved-bit"),
        (0x100000, "dpp-absneg"),
    ] {
        let p = compute([64, 1, 1], Vec::new(), |a| {
            a.push(dpp16(Op::v_add_u32, v(4), v(0), v(0), 0xff00_0000 | word));
        });
        assert_eq!(
            decant_shader::lift_program(&p).unwrap_err().key(),
            format!("form:v_add_u32:{reason}")
        );
    }
}

#[test]
fn native_dpp_add_allows_readfirstlane() {
    let p = compute([64, 1, 1], Vec::new(), |a| {
        a.push(dpp16(Op::v_add_u32, v(4), v(0), v(0), 0xff01_1100));
        a.push(vop1(Op::v_readfirstlane_b32, s(8), v(4)));
    });
    decant_shader::lift_program(&p).unwrap();
}

fn perm(op: Op, dst: Operand, opsel: u8) -> Insn {
    let mut i = vop3(op, dst, v(0), s(4), s(5));
    i.ext = Ext::Valu {
        abs: 0,
        neg: 0,
        opsel,
        omod: 0,
        clamp: false,
    };
    i
}

#[test]
fn permlane_selectors_flags_and_aliases_compile() {
    for op in [Op::v_permlane16_b32, Op::v_permlanex16_b32] {
        for lanes in [32, 64] {
            for flags in [0, 1, 2, 3, 8, 9, 10, 11] {
                let p = compute([64, 1, 1], Vec::new(), |a| {
                    if lanes == 32 {
                        // A numeric VCC scratch register supplies wave32 width evidence.
                        a.push(sop1(Op::s_mov_b32, VCC, int(0)));
                        a.push(vop2(Op::v_add_u32, v(0), VCC, v(0)));
                    }
                    // Distinct nibbles exercise both selector words and their high bits.
                    a.push(sop1(Op::s_mov_b32, s(4), lit(0x7654_3210)));
                    a.push(sop1(Op::s_mov_b32, s(5), lit(0xfedc_ba98)));
                    // Source aliases destination. The gather must precede its guarded write.
                    a.push(perm(op, v(0), flags));
                    a.push(ds(Op::ds_write_b32, NONE, [v(1), v(0), NONE], 0, 0));
                });
                let text = hlsl(&p);
                assert!(text.contains(&format!("[WaveSize({lanes})]")), "{text}");
                assert!(text.contains("WaveActiveBallot(exec || fi)"), "{text}");
                assert!(text.contains("(col & 7u) * 4u"), "{text}");
                let args = format!(
                    ", {}, {}, {})",
                    op == Op::v_permlanex16_b32,
                    flags & 1 != 0,
                    flags & 2 != 0
                );
                assert!(text.contains(&args), "{text}");
                compiles(&p);
            }
        }
    }
}

#[test]
fn permlane_refuses_unsupported_forms_and_scalar_wave_contexts() {
    for op in [Op::v_permlane16_b32, Op::v_permlanex16_b32] {
        for (change, reason) in [
            (0, "permlane-opsel"),
            (1, "permlane-modifiers"),
            (2, "permlane-modifiers"),
            (3, "permlane-modifiers"),
            (4, "permlane-modifiers"),
            (5, "permlane-operands"),
            (6, "permlane-operands"),
        ] {
            let p = compute([64, 1, 1], Vec::new(), |a| {
                let mut i = perm(op, v(2), 0);
                match change {
                    0 => {
                        i.ext = Ext::Valu {
                            abs: 0,
                            neg: 0,
                            opsel: 4,
                            omod: 0,
                            clamp: false,
                        }
                    }
                    1 => {
                        i.ext = Ext::Valu {
                            abs: 1,
                            neg: 0,
                            opsel: 0,
                            omod: 0,
                            clamp: false,
                        }
                    }
                    2 => {
                        i.ext = Ext::Valu {
                            abs: 0,
                            neg: 1,
                            opsel: 0,
                            omod: 0,
                            clamp: false,
                        }
                    }
                    3 => {
                        i.ext = Ext::Valu {
                            abs: 0,
                            neg: 0,
                            opsel: 0,
                            omod: 1,
                            clamp: false,
                        }
                    }
                    4 => {
                        i.ext = Ext::Valu {
                            abs: 0,
                            neg: 0,
                            opsel: 0,
                            omod: 0,
                            clamp: true,
                        }
                    }
                    5 => i.src[0] = s(0),
                    _ => i.src[1] = v(1),
                }
                a.push(i);
            });
            assert_eq!(
                decant_shader::lift_program(&p).unwrap_err().key(),
                format!("form:{}:{reason}", op.info().name)
            );
        }
        let p = compute([64, 1, 1], Vec::new(), |a| {
            a.push(perm(op, v(2), 0));
            a.push(vop1(Op::v_readfirstlane_b32, s(8), v(2)));
        });
        decant_shader::lift_program(&p).unwrap();
    }
}

fn exclusive_scan() -> Vec<Insn> {
    use decant_shader::decode::Reg;
    let hi = Operand::Reg(Reg::ExecHi);
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
        {
            let mut i = perm(Op::v_permlanex16_b32, v(8), 9);
            i.src[..3].copy_from_slice(&[v(7), int(-1), int(-1)]);
            i
        },
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
fn a_complete_exclusive_scan_uses_the_scalar_wave_projection() {
    let p = compute([64, 1, 1], Vec::new(), |a| {
        for i in exclusive_scan() {
            a.push(i);
        }
        a.push(ds(Op::ds_write_b32, NONE, [v(1), v(7), NONE], 0, 0));
    });
    let text = hlsl(&p);
    assert!(!text.contains("WaveReadLaneAt"), "{text}");
    assert!(!text.contains("decant_dpp_add"), "{text}");
    compiles(&p);
    // A different shift, lane selector, half mask or source is no longer
    // the proved scan: it must either be refused or fall back to native
    // wave instructions instead of using the scalar wave projection.
    for changed in [3, 7, 9, 11, 14, 17, 18, 19] {
        let p = compute([64, 1, 1], Vec::new(), |a| {
            let mut code = exclusive_scan();
            match changed {
                3 => code[3] = dpp16(Op::v_add_u32, v(7), v(7), v(7), 0xff09_1200),
                7 => code[7].src[0] = lit(0xff00_ff00),
                9 => code[9].src[1] = int(0),
                11 => code[11].src[1] = int(30),
                14 => code[14].src[0] = s(23),
                17 => code[17].src[1] = v(5),
                18 => code[18].src[0] = s(24),
                _ => {
                    code[11].dst = s(21);
                    code[14].src[0] = s(21);
                }
            }
            for i in code {
                a.push(i);
            }
        });
        if decant_shader::lift_program(&p).is_ok() {
            let text = hlsl(&p);
            assert!(
                text.contains("WaveReadLaneAt") || text.contains("decant_dpp_add"),
                "mutation {changed} unexpectedly used scalar wave projection:\n{text}"
            );
        }
    }
}

#[test]
#[ignore = "requires a local shader archive and DXC"]
fn instance_culling_shaders_compile_with_dxc() {
    use decant_shader::corpus::{Selection, par_corpus};

    let hashes = [
        0x2e0055c25f49848a,
        0x6e148dbd563de205,
        0xfaac0f5ebe046c11,
        0x62b584e061c05fc5,
    ];
    let archive = std::env::var_os("DECANT_SHADER_CORPUS").expect("set DECANT_SHADER_CORPUS");
    let dxc = common::find_dxc().expect("DXC is required for this test");
    let selection = Selection {
        hashes: Some(hashes.into_iter().collect()),
        ..Selection::default()
    };
    let (_, entries) = par_corpus(
        std::path::Path::new(&archive),
        &selection,
        1,
        Vec::new,
        |entries, entry| entries.push(entry),
    )
    .expect("read shader archive");
    let mut entries: std::collections::BTreeMap<_, _> = entries
        .into_iter()
        .flatten()
        .map(|entry| (entry.hash, entry.program))
        .collect();
    let mut failures = Vec::new();
    for hash in hashes {
        let program = entries
            .remove(&hash)
            .unwrap_or_else(|| panic!("missing {hash:016x}"));
        let compiled = program.and_then(|program| {
            translate_to_dxil(
                &[program],
                &RootSignature::default(),
                &Options::default(),
                &CountingHost::default(),
                &dxc,
            )
        });
        match compiled {
            Ok(stages) if stages.len() == 1 && !stages[0].dxil.is_empty() => {
                eprintln!("{hash:016x} compiled with DXC");
            }
            Ok(_) => failures.push(format!("{hash:016x} produced no DXIL")),
            Err(r) => failures.push(format!("{hash:016x} {r}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn dpp_mov_emits_controls_masks_fi_and_compiles() {
    // All GFX10 DPP16 control modes with fi, bound_ctrl, row_mask, bank_mask, aliasing and branches.
    let controls = [
        (
            0xb1,
            "source = (lane & ~3u) | ((ctrl >> (2u * (lane & 3u))) & 3u);",
        ),
        (
            0x4e,
            "source = (lane & ~3u) | ((ctrl >> (2u * (lane & 3u))) & 3u);",
        ),
        (
            0x1b,
            "source = (lane & ~3u) | ((ctrl >> (2u * (lane & 3u))) & 3u);",
        ),
        (0x103, "source = lane + delta;"),
        (0x113, "source = lane - delta;"),
        (0x123, "source = row | ((col - delta) & 15u);"),
        (0x140, "source = lane ^ 15u;"),
        (0x141, "source = lane ^ 7u;"),
        (0x154, "source = row | delta;"),
        (0x164, "source = lane ^ delta;"),
    ];
    let mut p = compute([8, 8, 1], Vec::new(), |a| {
        let done = a.label();
        a.push(sop1(Op::s_mov_b64, s(20), EXEC));
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(7)));
        a.branch(Op::s_cbranch_execz, done);
        for (i, (ctrl, _)) in controls.iter().enumerate() {
            let dst = v(4 + i as u8);
            let fi = ((i as u32) & 1) << 18;
            let bound = (((i as u32) >> 1) & 1) << 19;
            // Symmetric row_mask 0xa (rows 1 and 3) matches WaveSize 32 and WaveSize 64.
            a.push(dpp16(
                Op::v_mov_b32,
                dst,
                v(0),
                NONE,
                0xa500_0000 | (ctrl << 8) | fi | bound,
            ));
            a.push(ds(
                Op::ds_write_b32,
                NONE,
                [v(0), dst, NONE],
                (4 * i) as u8,
                0,
            ));
        }
        // In place aliasing (dst == src0) with fi=1 and bound_ctrl=1 (dpp=0xff0cb104).
        a.push(dpp16(Op::v_mov_b32, v(4), v(4), NONE, 0xff0c_b100));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 64, 0));
        a.bind(done);
        a.push(sop1(Op::s_mov_b64, EXEC, s(20)));
    });
    p.lds_bytes = 512;
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_dpp_mov("), "{text}");
    assert!(text.contains("WaveActiveBallot(exec || fi)"), "{text}");
    assert!(
        text.contains("WaveReadLaneAt(a, valid ? safeSource : lane)"),
        "{text}"
    );
    assert!(
        text.contains("return write && (valid || bound) ? (valid ? selected : 0u) : old;"),
        "{text}"
    );
    for (i, (ctrl, source)) in controls.into_iter().enumerate() {
        assert!(text.contains(source), "{text}");
        let args = format!(", {ctrl}u, {}, {}, 10u, 5u)", i % 2 != 0, (i / 2) % 2 != 0);
        assert!(text.contains(&args), "{text}");
    }
    compiles(&p);

    // An asymmetric row_mask (0xc selects rows 2 and 3) in a 64 lane wave keeps WaveSize 64.
    let mut wide = compute([64, 1, 1], Vec::new(), |a| {
        a.push(sop1(Op::s_mov_b64, s(20), EXEC));
        a.push(dpp16(Op::v_mov_b32, v(4), v(0), NONE, 0xcf0c_b100));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
    });
    wide.lds_bytes = 256;
    let text = hlsl(&wide);
    assert!(text.contains("[WaveSize(64)]"), "{text}");
    assert!(text.contains(", 177u, true, true, 12u, 15u)"), "{text}");
    compiles(&wide);

    // Shader Model 6.5 refuses WaveSize cleanly.
    let opts = Options {
        shader_model_minor: 5,
        ..Options::default()
    };
    let err = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &opts,
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(err.key(), "stage:dpp-shader-model");

    // Unsupported DPP16 controls and bits refuse precisely.
    for (word, reason) in [
        (0x13000, "dpp-wave-shl"),
        (0x13400, "dpp-wave-rol"),
        (0x13800, "dpp-wave-shr"),
        (0x13c00, "dpp-wave-ror"),
        (0x14200, "dpp-row-bcast15"),
        (0x14300, "dpp-row-bcast31"),
        (0x17000, "dpp-reserved-control"),
        (0x20000, "dpp-reserved-bit"),
        (0x100000, "dpp-absneg"),
    ] {
        let bad = compute([64, 1, 1], Vec::new(), |a| {
            a.push(dpp16(Op::v_mov_b32, v(4), v(0), NONE, 0xff00_0000 | word));
        });
        assert_eq!(
            decant_shader::lift_program(&bad).unwrap_err().key(),
            format!("form:v_mov_b32:{reason}")
        );
    }
}

#[test]
fn compute_quad_dpp_alu_lowers_via_dpp_mov_and_compiles() {
    let mut p = compute([8, 8, 1], Vec::new(), |a| {
        a.push(dpp16(Op::v_add_f32, v(4), v(0), v(1), 0xff0c_b100));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(4), NONE], 0, 0));
    });
    p.lds_bytes = 256;
    let text = hlsl(&p);
    assert!(text.contains("[WaveSize(32)]"), "{text}");
    assert!(text.contains("decant_dpp_mov("), "{text}");
    compiles(&p);
}

#[test]
fn wave32_constant_lane_patterns_and_exclusive_scan_compile() {
    let mut p = compute([32, 1, 1], Vec::new(), |a| {
        let done = a.label();
        a.push(sop1(Op::s_mov_b32, s(10), EXEC));
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(24)));
        a.branch(Op::s_cbranch_execz, done);
        // Standalone constant lane pattern EXEC update.
        a.push(sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)));
        a.push(vop2(Op::v_add_u32, v(5), int(1), v(0)));
        a.push(sop2(Op::s_and_b32, EXEC, EXEC, lit(0x5555_5555)));
        a.push(vop2(Op::v_add_u32, v(5), int(2), v(5)));
        a.push(sop1(Op::s_mov_b32, EXEC, s(10)));
        // Complete Wave32 exclusive scan with waitcnt and mask narrowing.
        a.push(sop1(Op::s_or_saveexec_b32, s(1), int(-1)));
        a.push(sopp(Op::s_waitcnt, 0xc07f));
        a.push(sop2(Op::s_and_b32, s(2), s(10), s(1)));
        a.push(vop3(Op::v_cndmask_b32, v(1), int(0), v(5), s(2)));
        a.push(vop1(Op::v_mov_b32, v(2), v(1)));
        for control in [0xff09_1100, 0xff01_1200, 0xff01_1400, 0xff01_1800] {
            a.push(dpp16(Op::v_add_u32, v(2), v(2), v(2), control));
        }
        a.push(sop1(Op::s_mov_b32, EXEC, lit(0xffff_0000)));
        let mut perm_insn = perm(Op::v_permlanex16_b32, v(3), 9);
        perm_insn.src[..3].copy_from_slice(&[v(2), int(-1), int(-1)]);
        a.push(perm_insn);
        a.push(vop2(Op::v_add_u32, v(2), v(3), v(2)));
        a.push(sop1(Op::s_mov_b32, EXEC, int(-1)));
        a.push(vop2(Op::v_sub_u32, v(2), v(2), v(1)));
        a.push(sop1(Op::s_mov_b32, EXEC, s(1)));
        a.push(ds(Op::ds_write_b32, NONE, [v(0), v(2), NONE], 0, 0));
        a.bind(done);
    });
    p.lds_bytes = 256;
    let text = hlsl(&p);
    assert!(!text.contains("decant_dpp_add"), "{text}");
    assert!(!text.contains("decant_permlane"), "{text}");
    assert!(text.contains("4294901760u"), "{text}");
    compiles(&p);
}

#[test]
#[ignore = "requires a local shader archive and DXC"]
fn dpp_and_lane_pattern_shaders_compile_with_dxc() {
    use decant_shader::corpus::{Selection, par_corpus};

    let hashes = [
        0x8e2c_56a1_aed7_3165,
        0xf487_8819_ab94_8e0a,
        0x89e1_d116_1389_7fb1,
        0x5f07_b049_4873_8f66,
    ];
    let archive = std::env::var_os("DECANT_SHADER_CORPUS").expect("set DECANT_SHADER_CORPUS");
    let dxc = common::find_dxc().expect("DXC is required for this test");
    let selection = Selection {
        hashes: Some(hashes.into_iter().collect()),
        ..Selection::default()
    };
    let (_, entries) = par_corpus(
        std::path::Path::new(&archive),
        &selection,
        1,
        Vec::new,
        |entries, entry| entries.push(entry),
    )
    .expect("read shader archive");
    let mut entries: std::collections::BTreeMap<_, _> = entries
        .into_iter()
        .flatten()
        .map(|entry| (entry.hash, entry.program))
        .collect();
    let mut failures = Vec::new();
    for hash in hashes {
        let program = entries
            .remove(&hash)
            .unwrap_or_else(|| panic!("missing {hash:016x}"));
        let compiled = program.and_then(|program| {
            translate_to_dxil(
                &[program],
                &RootSignature::default(),
                &Options::default(),
                &CountingHost::default(),
                &dxc,
            )
        });
        match compiled {
            Ok(stages) if stages.len() == 1 && !stages[0].dxil.is_empty() => {
                eprintln!("{hash:016x} compiled with DXC");
            }
            Ok(_) => failures.push(format!("{hash:016x} produced no DXIL")),
            Err(r) => failures.push(format!("{hash:016x} {r}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn merged_image_descriptor_after_many_diamonds_compiles_cs_6_6() {
    let p = compute([8, 4, 1], vec![table(4)], |a| {
        let branch_b = a.label();
        let join = a.label();
        a.push(sop1(Op::s_mov_b32, s(24), s(4)));
        a.push(sop1(Op::s_mov_b32, s(25), int(0)));
        a.push(sopc(Op::s_cmp_eq_u32, s(6), int(0)));
        a.branch(Op::s_cbranch_scc1, branch_b);

        a.push(smem(Op::s_load_dwordx8, s(4), s(24), 0x120));
        for i in 0..6 {
            let skip = a.label();
            a.push(vopc(Op::v_cmp_eq_u32, v(0), int(i)));
            a.push(sop1(
                Op::s_and_saveexec_b32,
                s(14),
                Operand::Reg(Reg::VccLo),
            ));
            a.branch(Op::s_cbranch_execz, skip);
            a.push(vop1(Op::v_mov_b32, v(2), int(i)));
            a.bind(skip);
            a.push(sop1(Op::s_mov_b32, Operand::Reg(Reg::ExecLo), s(14)));
        }
        a.branch(Op::s_branch, join);

        a.bind(branch_b);
        a.push(smem(Op::s_load_dwordx8, s(4), s(24), 0x140));
        for i in 0..6 {
            let skip = a.label();
            a.push(vopc(Op::v_cmp_eq_u32, v(1), int(i)));
            a.push(sop1(
                Op::s_and_saveexec_b32,
                s(14),
                Operand::Reg(Reg::VccLo),
            ));
            a.branch(Op::s_cbranch_execz, skip);
            a.push(vop1(Op::v_mov_b32, v(2), int(i)));
            a.bind(skip);
            a.push(sop1(Op::s_mov_b32, Operand::Reg(Reg::ExecLo), s(14)));
        }

        a.bind(join);
        a.push(image(Op::image_store, v(2), v(0), s(4), s(0), 1, 0xf));
    });
    let text = hlsl(&p);
    assert!(text.contains("ResourceDescriptorHeap"), "{text}");
    compiles(&p);
}
