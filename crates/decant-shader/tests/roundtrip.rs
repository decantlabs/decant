//! The decoder and the encoder are inverses: every instruction the assembler can build decodes
//! back to the same record, format by format.

use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn, Operand, decode_all, decode_one, encode, listing};
use decant_shader::tables::{Format, OPS, Op};

fn round(insn: Insn) -> Insn {
    let words = encode(&insn);
    let back = decode_one(&words, 0).unwrap_or_else(|e| panic!("{}: {e}", insn.op.name()));
    assert_eq!(usize::from(back.len), words.len(), "{}", insn.op.name());
    assert_eq!(Insn { len: 0, ..back }, insn, "{}", insn.op.name());
    back
}

#[test]
fn one_instruction_of_every_format_round_trips() {
    round(sop1(Op::s_mov_b32, s(4), lit(0x447f_028f)));
    round(sop1(Op::s_mov_b64, s(12), EXEC));
    round(sop2(Op::s_add_u32, s(1), s(2), int(-3)));
    round(sop2(Op::s_and_b64, EXEC, EXEC, s(12)));
    round(sopc(Op::s_cmp_lt_u32, s(3), int(64)));
    round(sopk(Op::s_movk_i32, s(8), 0x92));
    round(sopp(Op::s_waitcnt, 0x3f70));
    round(smem(Op::s_load_dwordx8, s(4), s(8), 0x20));
    round(smem(Op::s_buffer_load_dwordx4, s(0), s(4), 0x20));
    round(vop1(Op::v_mov_b32, v(7), float(1.0)));
    round(vop1(Op::v_cvt_i32_f32, v(0), v(0)));
    round(vop2(Op::v_mul_legacy_f32, v(2), s(0), v(2)));
    round(vop2(Op::v_add_f32, v(3), lit(0x4006_6666), v(9)));
    // The constant of the `ak` form is always the literal word after the instruction.
    let fmaak = round(Insn {
        src: [s(56), v(47), lit(0x3f00_0000), Operand::None],
        ..vop2(Op::v_fmaak_f32, v(35), s(56), v(47))
    });
    assert_eq!(fmaak.len, 2);
    round(vopc(Op::v_cmp_lt_f32, v(1), v(2)));
    round(vopc(Op::v_cmpx_gt_u32, int(5), v(2)));
    round(vop3(Op::v_fma_f32, v(0), v(10), s(4), lit(0x4006_6666)));
    round(vop3(Op::v_med3_f32, v(1), v(16), int(0), float(1.0)));
    round(vop3(Op::v_add_f32, v(1), v(2), v(3), Operand::None));
    round(interp(Op::v_interp_p1_f32, v(2), v(0), 3, 1));
    round(interp(Op::v_interp_p2_f32, v(2), v(1), 3, 1));
    round(buffer_load(
        Op::buffer_load_dwordx4,
        v(0),
        v(0),
        s(0),
        0,
        true,
    ));
    round(image(Op::image_sample, v(11), v(2), s(4), s(0), 1, 0xf));
    round(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    round(export(33, 0xf, [12, 13, 2, 7], false, false, false));
}

#[test]
fn the_two_word_form_of_a_compare_writes_a_scalar_pair() {
    let insn = Insn {
        dst: s(10),
        ..vop3(Op::v_cmp_ge_f32, s(10), v(1), v(2), Operand::None)
    };
    let back = round(insn);
    assert_eq!(back.form, Form::AsVop3);
    assert_eq!(back.dst, s(10));
}

#[test]
fn a_program_lists_and_a_branch_finds_its_target() {
    let mut a = Asm::new();
    let end = a.label();
    a.push(sopc(Op::s_cmp_eq_u32, s(8), int(0)));
    a.branch(Op::s_cbranch_scc1, end);
    a.push(vop1(Op::v_mov_b32, v(0), int(1)));
    a.bind(end);
    a.push(sopp(Op::s_endpgm, 0));
    let code = a.finish();
    let insns = decode_all(&code[6..]).unwrap();
    assert_eq!(insns.len(), 4);
    assert_eq!(insns[1].branch_target(), Some(3));
    assert!(matches!(insns[1].ext, Ext::Imm16(1)));
    let text = listing(&insns);
    assert!(text.contains("s_cbranch_scc1 -> 3"), "{text}");
    assert!(text.contains("v_mov_b32 v0, 1"), "{text}");
}

/// Every scalar instruction with plain operands survives the round trip. This walks the whole
/// table, so a wrong field position for an opcode number shows up here.
#[test]
fn every_scalar_alu_instruction_round_trips() {
    let mut count = 0;
    for info in OPS.iter() {
        let insn = match info.format {
            Format::Sop1 => sop1(info.op, s(3), s(5)),
            Format::Sop2 => sop2(info.op, s(3), s(5), s(7)),
            Format::Sopc => sopc(info.op, s(5), s(7)),
            Format::Sopp => sopp(info.op, 2),
            Format::Sopk if info.op != Op::s_setreg_imm32_b32 => sopk(info.op, s(3), 0x1234),
            _ => continue,
        };
        round(insn);
        count += 1;
    }
    assert!(count > 150, "{count}");
}

/// The carry forms (`Vop3B`) keep their second result in bits 8 to 14 of the first word. Those
/// bits are not `abs` and `opsel`, whatever register the second result is.
#[test]
fn every_two_result_vop3_instruction_round_trips_with_its_second_result() {
    let mut count = 0;
    for info in OPS.iter() {
        if info.defs.len() != 2 || !matches!(info.format, Format::Vop2 | Format::Vop3) {
            continue;
        }
        for sdst in [s(0), s(100), s(105), VCC] {
            let mut insn = vop3(info.op, v(1), v(2), v(3), v(4));
            insn.dst2 = sdst;
            // Unused source slots come back empty: start from the decoded record.
            let canonical = decode_one(&encode(&insn), 0).unwrap();
            let back = round(Insn {
                len: 0,
                ..canonical
            });
            assert_eq!(back.dst2, sdst, "{}", info.name);
            let Ext::Valu { abs, opsel, .. } = back.ext else {
                panic!("{}", info.name)
            };
            assert_eq!((abs, opsel), (0, 0), "{}", info.name);
        }
        count += 1;
    }
    assert!(count >= 10, "{count}");
}

/// The record the decoder gives for the words of `insn` (unused fields empty), and a check that
/// it survives another round.
fn canonical(insn: Insn) -> Insn {
    let first = decode_one(&encode(&insn), 0).unwrap_or_else(|e| panic!("{}: {e}", insn.op.name()));
    round(Insn { len: 0, ..first })
}

#[test]
fn every_vector_one_word_instruction_round_trips_in_every_form() {
    use decant_shader::decode::{Dpp8, Dpp16, DppCtrl, SdwaSel, SdwaUnused};
    let mut count = 0;
    for info in OPS.iter() {
        if !matches!(info.format, Format::Vop1 | Format::Vop2 | Format::Vopc) {
            continue;
        }
        if info.op.name().ends_with("_e64") {
            continue;
        }
        let dst = if info.format == Format::Vopc {
            if info.op.name().starts_with("v_cmpx") {
                EXEC
            } else {
                VCC
            }
        } else {
            v(5)
        };
        let src1 = if info.format == Format::Vop1 {
            Operand::None
        } else {
            v(9)
        };
        let plain = Insn {
            src: [v(3), src1, Operand::None, Operand::None],
            ..vop2(info.op, dst, v(3), src1)
        };
        if matches!(
            info.op,
            Op::v_madmk_f32
                | Op::v_madak_f32
                | Op::v_fmamk_f32
                | Op::v_fmaak_f32
                | Op::v_fmamk_f16
                | Op::v_fmaak_f16
        ) {
            // The constant is the literal word, which no extra word form combines with.
            canonical(Insn {
                src: [v(3), v(9), lit(0x3f80_0000), Operand::None],
                ..plain
            });
            continue;
        }
        canonical(plain);
        // Sub dword: byte 2 of the destination, word 1 of source 0, byte 3 of source 1, with
        // negate on source 0 and sign extend on source 1.
        let word = (2 << 8) | (1 << 11) | (5 << 16) | (1 << 20) | (3 << 24) | (1 << 27);
        let sd = canonical(sdwa(info.op, dst, v(3), src1, word));
        let x = sd.sdwa().unwrap();
        assert!(
            x.src0_neg() && x.src1_sext() && !x.src0_is_scalar(),
            "{}",
            info.name
        );
        assert_eq!(x.src0_sel(), Some(SdwaSel::Word1));
        assert_eq!(x.src1_sel(), Some(SdwaSel::Byte3));
        if info.format != Format::Vopc {
            assert_eq!(x.dst_sel(), Some(SdwaSel::Byte2));
            assert_eq!(x.dst_unused(), Some(SdwaUnused::Sext));
        }
        let dp = canonical(dpp16(
            info.op,
            dst,
            v(3),
            src1,
            (0xB1 << 8) | (1 << 19) | (0xA << 24) | (0xF << 28),
        ));
        let y: Dpp16 = dp.dpp16().unwrap();
        assert_eq!(y.ctrl(), DppCtrl::QuadPerm([1, 0, 3, 2]));
        assert!(y.bound_ctrl() && y.bank_mask() == 0xA && y.row_mask() == 0xF);
        for fi in [false, true] {
            let d8 = canonical(dpp8(info.op, dst, v(3), src1, 0xFAC6_8800, fi));
            let (z, got): (Dpp8, bool) = d8.dpp8().unwrap();
            assert_eq!(got, fi);
            assert_eq!(
                (0..8).map(|l| z.lane_sel(l)).collect::<Vec<_>>(),
                (0..8).collect::<Vec<_>>()
            );
        }
        count += 1;
    }
    assert!(count > 250, "{count}");
}

/// Every remaining table row, one record each (two for the image instructions: with and without
/// the separate address registers). Together with the tests above the whole table is covered.
#[test]
fn every_other_instruction_of_the_table_round_trips() {
    let mut seen = std::collections::BTreeMap::<Format, u32>::new();
    for info in OPS.iter() {
        let o = info.op;
        let insn = match info.format {
            Format::Vop3 | Format::Vop3p => {
                let scalar_dst = info
                    .defs
                    .first()
                    .is_some_and(|d| d.base == decant_shader::tables::Base::LaneMask)
                    || matches!(o, Op::v_readlane_b32_e64);
                let mut i = vop3(o, if scalar_dst { s(6) } else { v(1) }, v(2), v(3), v(4));
                i.form = Form::Plain;
                if info.format == Format::Vop3p {
                    i.ext = Ext::Valu {
                        abs: 0,
                        neg: 5,
                        opsel: 3,
                        omod: 1,
                        clamp: true,
                    };
                }
                i
            }
            Format::Smem => smem(o, s(4), s(8), 0x1234),
            Format::Ds => ds(o, v(1), v(2), v(3), v(4), 8, 9),
            Format::Mubuf => buffer_store(o, v(1), v(2), s(8), 0x40, true),
            Format::Mtbuf => tbuffer(o, v(1), v(2), s(8), s(3), 0x20, 22, true, false),
            Format::Mimg => {
                canonical(image(o, v(1), v(2), s(8), s(12), 1, 0xf));
                image_nsa(o, v(1), v(2), &[7, 9, 11, 13, 15], s(8), s(12), 5, 0x3)
            }
            Format::Vintrp => interp(o, v(1), v(2), 3, 1),
            Format::Exp => export(0, 0xf, [0, 1, 2, 3], false, true, true),
            _ => continue,
        };
        canonical(insn);
        *seen.entry(info.format).or_default() += 1;
    }
    assert!(seen.len() >= 9, "{seen:?}");
}

#[test]
fn image_address_registers_come_from_the_separate_words_when_there_are_some() {
    let plain = image(Op::image_sample, v(1), v(4), s(8), s(12), 1, 0xf);
    let Ext::Mimg(m) = plain.ext else {
        unreachable!()
    };
    assert_eq!(
        (0..3).map(|i| m.address(4, i)).collect::<Vec<_>>(),
        [Some(4), Some(5), Some(6)]
    );
    let split = canonical(image_nsa(
        Op::image_sample,
        v(1),
        v(4),
        &[9, 7, 20],
        s(8),
        s(12),
        1,
        0xf,
    ));
    let Ext::Mimg(m) = split.ext else {
        unreachable!()
    };
    assert_eq!(split.len, 3);
    assert_eq!(
        (0..5).map(|i| m.address(4, i)).collect::<Vec<_>>(),
        [Some(4), Some(9), Some(7), Some(20), Some(0)]
    );
}
