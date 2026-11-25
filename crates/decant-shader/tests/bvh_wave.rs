//! Synthetic node intersection and lane operation coverage.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Insn};
use decant_shader::ir::{Expr, Guard, Stmt};
use decant_shader::tables::Op;
use decant_shader::{lift_program, translate};

fn program(insn: Insn) -> StageProgram {
    let mut asm = Asm::new();
    asm.push(insn).push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        ..StageProgram::new(Stage::Compute, asm.finish())
    }
}

/// `insn` after a load of its descriptor into s8 to s11 from slot 1 of the table in s4, with
/// the first result word written to group shared memory so the intersection stays live.
fn bvh_program(insn: Insn) -> StageProgram {
    let none = decant_shader::decode::Operand::None;
    let mut asm = Asm::new();
    asm.push(sop1(Op::s_mov_b32, s(5), int(0)))
        .push(smem(Op::s_load_dwordx4, s(8), s(4), 0x20))
        .push(insn)
        .push(ds(Op::ds_write_b32, none, v(1), v(20), none, 0, 0))
        .push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        lds_bytes: 256,
        user_data: vec![UserDataSlot {
            sgpr: 4,
            kind: UserDataKind::Table,
        }],
        ..StageProgram::new(Stage::Compute, asm.finish())
    }
}

#[test]
fn bvh_forms_preserve_descriptor_address_width_and_all_result_words() {
    for op in [Op::image_bvh_intersect_ray, Op::image_bvh64_intersect_ray] {
        for a16 in [false, true] {
            for nsa in [false, true] {
                let wide = op == Op::image_bvh64_intersect_ray;
                let count = usize::from(wide) + 5 + if a16 { 3 } else { 6 };
                let more = (0..count - 1).map(|i| 90 + i as u8 * 2).collect::<Vec<_>>();
                let mut insn = if nsa {
                    image_nsa(op, v(20), v(20), &more, s(8), s(0), 0, 15)
                } else {
                    image(op, v(20), v(20), s(8), s(0), 0, 15)
                };
                if let Ext::Mimg(m) = &mut insn.ext {
                    m.a16 = a16;
                }
                let p = bvh_program(insn);
                let func = lift_program(&p).unwrap();
                let mut words = Vec::new();
                func.for_each_stmt(&mut |stmt| {
                    if let Stmt::Set {
                        value: Expr::BvhWord(node, word),
                        guard,
                        ..
                    } = stmt
                    {
                        assert_eq!(*guard, Guard::Exec);
                        assert_eq!(node.wide, wide);
                        assert_eq!(node.encoding.a16, a16);
                        assert_eq!(node.address.len(), count);
                        assert_eq!(node.descriptor.len(), 4);
                        assert_eq!(
                            node.address[1],
                            Expr::Var(decant_shader::ir::Var::V(if nsa { 90 } else { 21 }))
                        );
                        words.push(*word);
                    }
                });
                assert_eq!(words, [0, 1, 2, 3]);
                // The descriptor becomes the node memory buffer and the intersection lowers to
                // one call of the HLSL helper. `tests/bvh_hlsl.rs` covers the text and DXC.
                let translated = translate(
                    &[p],
                    &RootSignature::default(),
                    &Options::default(),
                    &CountingHost::default(),
                )
                .unwrap_or_else(|r| panic!("refused: {r}"));
                let text = &translated.stages[0].hlsl;
                assert!(
                    text.contains("= decant_bvh_intersect(heap_t_s4_1, "),
                    "{text}"
                );
            }
        }
    }
}

#[test]
fn bvh_rejects_address_and_result_register_overflow() {
    for (dst, addr) in [(253, 0), (0, 250)] {
        let p = program(image(
            Op::image_bvh64_intersect_ray,
            v(dst),
            v(addr),
            s(8),
            s(0),
            0,
            15,
        ));
        assert!(lift_program(&p).is_err());
    }
}

#[test]
fn half_dpp_add_preserves_masks_and_the_old_high_half() {
    let p = program(dpp16(
        Op::v_add_f16,
        v(2),
        v(3),
        v(4),
        (0x111 << 8) | (1 << 19) | (5 << 24) | (10 << 28),
    ));
    let text = lift_program(&p).unwrap().to_string();
    assert!(
        text.contains("decant_dpp_add(v3, v4, v2, exec, 0x111, true, 0xa, 5, true)"),
        "{text}"
    );
}

#[test]
fn dpp8_move_preserves_lane_selectors_and_fetch_inactive() {
    for fi in [false, true] {
        let p = program(dpp8(
            Op::v_mov_b32,
            v(2),
            v(3),
            decant_shader::decode::Operand::None,
            0x12345600,
            fi,
        ));
        let text = lift_program(&p).unwrap().to_string();
        assert!(
            text.contains(&format!("decant_dpp8_mov(v3, 0x123456, exec, {fi})")),
            "{text}"
        );
    }
}

#[test]
fn bpermute_keeps_byte_offset_and_reads_before_writing() {
    let p = program(ds(
        Op::ds_bpermute_b32,
        v(2),
        v(1),
        v(2),
        decant_shader::decode::Operand::None,
        20,
        1,
    ));
    let text = lift_program(&p).unwrap().to_string();
    assert!(
        text.contains("t0 = decant_bpermute(v2, v1, 0x114, exec)"),
        "{text}"
    );
    assert!(text.contains("@exec v2 = t0"), "{text}");
}

#[test]
fn native_lane_forms_translate_and_compile_for_both_wave_sizes() {
    let none = decant_shader::decode::Operand::None;
    for lanes in [32, 64] {
        for insn in [
            dpp16(
                Op::v_add_f16,
                v(2),
                v(0),
                v(1),
                (0x111 << 8) | (1 << 19) | 0xff000000,
            ),
            dpp8(Op::v_mov_b32, v(2), v(0), none, 0x12345600, false),
            ds(Op::ds_bpermute_b32, v(2), v(0), v(1), none, 20, 0),
        ] {
            let mut asm = Asm::new();
            if lanes == 32 {
                asm.push(sop1(Op::s_mov_b32, VCC, int(0))).push(vop2(
                    Op::v_add_u32,
                    v(0),
                    VCC,
                    v(0),
                ));
            }
            asm.push(insn)
                .push(ds(Op::ds_write_b32, none, v(0), v(2), none, 0, 0))
                .push(sopp(Op::s_endpgm, 0));
            let p = StageProgram {
                threads: [lanes, 1, 1],
                lds_bytes: 256,
                ..StageProgram::new(Stage::Compute, asm.finish())
            };
            let translated = translate(
                &[p],
                &RootSignature::default(),
                &Options::default(),
                &CountingHost::default(),
            )
            .unwrap();
            let stage = &translated.stages[0];
            assert!(
                stage.hlsl.contains(&format!("[WaveSize({lanes})]")),
                "{}",
                stage.hlsl
            );
            if let Some(dxc) = common::find_dxc() {
                let bytes = dxc
                    .compile(&stage.hlsl, &stage.entry, &stage.profile, &stage.dxc_args)
                    .unwrap();
                assert!(!bytes.is_empty());
            } else {
                eprintln!("DXC unavailable, native lane compilation skipped");
            }
        }
    }
}
