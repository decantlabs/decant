//! Relative VGPR moves in synthetic programs and the title's compression shaders.

mod common;

use std::path::Path;

use decant_shader::api::{CountingHost, Options, RootSignature, Stage, StageProgram};
use decant_shader::asm::*;
use decant_shader::corpus::{self, Entry, Selection};
use decant_shader::decode::Operand;
use decant_shader::tables::Op;
use decant_shader::{lift_program, translate_to_dxil};

fn compute(mut asm: Asm) -> StageProgram {
    asm.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        ..StageProgram::new(Stage::Compute, asm.finish())
    }
}

#[test]
fn signed_offset_selects_a_register_below_the_base() {
    let mut asm = Asm::new();
    asm.push(sop1(Op::s_mov_b32, M0, int(-1)))
        .push(vop1(Op::v_movrels_b32, v(2), v(5)));
    let text = lift_program(&compute(asm)).unwrap().to_string();
    assert!(text.contains("v2 = v4"), "{text}");
}

#[test]
fn loop_bounds_both_relative_moves_and_preserves_an_aliased_source() {
    let mut asm = Asm::new();
    let head = asm.label();
    asm.push(sop1(Op::s_mov_b32, s(0), int(0)))
        .bind(head)
        .push(sop1(Op::s_mov_b32, M0, s(0)))
        .push(vop1(Op::v_movreld_b32, v(5), v(5)))
        .push(vop1(Op::v_movrels_b32, v(2), v(5)))
        .push(sop2(Op::s_add_i32, s(0), s(0), int(1)))
        .push(sopc(Op::s_cmp_eq_u32, s(0), int(2)))
        .branch(Op::s_cbranch_scc0, head);
    let text = lift_program(&compute(asm)).unwrap().to_string();
    assert!(text.contains("@exec t0 = v5"), "{text}");
    assert!(text.contains("@exec v5 = ((m0 == 0) ? t0 : v5)"), "{text}");
    assert!(text.contains("@exec v6 = ((m0 == 1) ? t0 : v6)"), "{text}");
    assert!(text.contains("@exec v2 = ((m0 <u 1) ? v5 : v6)"), "{text}");
    assert!(!text.contains("v7"), "{text}");
}

#[test]
fn out_of_range_m0_is_refused() {
    for offset in [int(-2), lit(256)] {
        let mut asm = Asm::new();
        asm.push(sop1(Op::s_mov_b32, M0, offset))
            .push(vop1(Op::v_movrels_b32, v(2), v(1)));
        let refusal = lift_program(&compute(asm)).unwrap_err();
        assert_eq!(refusal.key(), "form:v_movrels_b32:relative-m0-range");
    }
}

#[test]
#[ignore = "needs DECANT_SHADER_CORPUS and DXC"]
fn bc_compression_corpus_compiles() {
    let path =
        std::env::var_os("DECANT_SHADER_CORPUS").expect("set DECANT_SHADER_CORPUS to the archive");
    let dxc = common::find_dxc().expect("DXC");
    let hashes = [0x9cde95cb518ed7c3, 0xacaa614050409d4f, 0x37e70e7affa8e849];
    let selection = Selection {
        hashes: Some(hashes.into_iter().collect()),
        ..Selection::default()
    };
    let (_, parts) = corpus::par_corpus(
        Path::new(&path),
        &selection,
        1,
        Vec::new,
        |entries: &mut Vec<Entry>, entry| entries.push(entry),
    )
    .expect("read archive");
    let entries: Vec<_> = parts.into_iter().flatten().collect();
    assert_eq!(entries.len(), hashes.len());
    for entry in entries {
        let program = entry
            .program
            .unwrap_or_else(|err| panic!("{}: {err}", entry.name));
        translate_to_dxil(
            &[program],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|err| panic!("{}: {err}", entry.name));
        println!("compiled {} ({:016x})", entry.name, entry.hash);
    }
}

#[test]
fn dynamic_m0_covers_signed_offsets_and_saves_an_aliased_source() {
    let mut asm = Asm::new();
    asm.push(sop1(Op::s_mov_b32, M0, s(0)))
        .push(vop1(Op::v_movreld_b32, v(5), v(5)))
        .push(vop1(Op::v_movrels_b32, v(2), v(5)));
    let text = lift_program(&compute(asm)).unwrap().to_string();
    assert!(text.contains("@exec t0 = v5"), "{text}");
    assert!(
        text.contains("@exec v0 = ((m0 == 0xfffffffb) ? t0 : v0)"),
        "{text}"
    );
    assert!(
        text.contains("@exec v255 = ((m0 == 0xfa) ? t0 : v255)"),
        "{text}"
    );
    assert!(text.contains("((5 + m0) <u 0x100)"), "{text}");
}

/// A compute program that saves EXEC into two lanes of `v128` around a branch and indexes a
/// private array with a relative move inside the branch. `m0` is the relative index.
fn parked_exec_around_relative_move(m0: Operand) -> Asm {
    let mut asm = Asm::new();
    let skip = asm.label();
    asm.push(vopc(Op::v_cmp_gt_u32, v(0), int(3)))
        .push(sop1(Op::s_and_saveexec_b64, s(6), VCC))
        .push(vop3(Op::v_writelane_b32_e64, v(128), s(6), int(4), Operand::None))
        .push(vop3(Op::v_writelane_b32_e64, v(128), s(7), int(5), Operand::None))
        .branch(Op::s_cbranch_execz, skip)
        .push(sop1(Op::s_mov_b32, M0, m0))
        .push(vop1(Op::v_movrels_b32, v(2), v(5)))
        .bind(skip)
        .push(vop3(Op::v_readlane_b32_e64, s(6), v(128), int(4), Operand::None))
        .push(vop3(Op::v_readlane_b32_e64, s(7), v(128), int(5), Operand::None))
        .push(sop2(Op::s_or_b64, EXEC, EXEC, s(6)));
    asm
}

#[test]
fn parks_beside_relative_moves() {
    for m0 in [s(0), int(1)] {
        let program = compute(parked_exec_around_relative_move(m0));
        let text = lift_program(&program)
            .unwrap_or_else(|r| panic!("{m0:?}: {r}"))
            .to_string();
        assert!(text.contains("spill128_4"), "{text}");
        assert!(!text.contains("decant_write_lane"), "{text}");
        let Some(dxc) = common::find_dxc() else {
            eprintln!("no DXC, compile check skipped");
            continue;
        };
        translate_to_dxil(
            &[program],
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{m0:?}: {r}"));
    }
}

#[test]
fn a_lane_read_without_a_lane_write_is_no_parking_place_beside_a_relative_move() {
    let mut asm = Asm::new();
    asm.push(sop1(Op::s_mov_b32, M0, s(0)))
        .push(vop1(Op::v_movreld_b32, v(5), v(1)))
        .push(vop3(Op::v_readlane_b32_e64, s(6), v(9), int(3), Operand::None))
        .push(vop1(Op::v_mov_b32, v(2), s(6)));
    let text = lift_program(&compute(asm)).unwrap().to_string();
    assert!(text.contains("decant_read_lane(v9, "), "{text}");
    assert!(!text.contains("spill9_3"), "{text}");
}

#[test]
fn a_relative_move_proven_to_reach_a_parking_register_is_refused() {
    let mut asm = Asm::new();
    asm.push(sop1(Op::s_mov_b32, M0, int(3)))
        .push(vop3(Op::v_writelane_b32_e64, v(8), s(6), int(0), Operand::None))
        .push(vop1(Op::v_movrels_b32, v(2), v(5)))
        .push(vop3(Op::v_readlane_b32_e64, s(7), v(8), int(0), Operand::None));
    let refusal = lift_program(&compute(asm)).unwrap_err();
    assert_eq!(refusal.key(), "form:v_movrels_b32:relative-parking");
}
