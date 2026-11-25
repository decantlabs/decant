//! Node intersections (`image_bvh_intersect_ray`, `image_bvh64_intersect_ray`) translated to
//! HLSL and compiled with DXC.
//!
//! Every program here is a compute program of 64 lanes. It takes the descriptor of the node
//! memory from a table or builds it around a root view, runs one node intersection into v20 to
//! v23 and writes the four result words to group shared memory so they stay live. The programs
//! are made with the assembler for the test. No console data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Insn, Operand};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

const NONE: Operand = Operand::None;

/// The node intersection `op` into v20 to v23 with the descriptor in s8 to s11 and the address
/// operands from v0, or from v0 and then v90, v92 and on for the NSA form.
fn intersection(op: Op, a16: bool, nsa: bool) -> Insn {
    let wide = op == Op::image_bvh64_intersect_ray;
    let count = usize::from(wide) + 5 + if a16 { 3 } else { 6 };
    let more = (0..count - 1).map(|i| 90 + i as u8 * 2).collect::<Vec<_>>();
    let mut insn = if nsa {
        image_nsa(op, v(20), v(0), &more, s(8), s(0), 0, 15)
    } else {
        image(op, v(20), v(0), s(8), s(0), 0, 15)
    };
    if let Ext::Mimg(m) = &mut insn.ext {
        m.a16 = a16;
    }
    insn
}

/// A compute program: `setup` puts the descriptor in s8 to s11, then the intersection runs and
/// its four result words go to group shared memory at the lane's address in v1.
fn program(setup: &[Insn], insn: Insn, user_data: Vec<UserDataSlot>) -> StageProgram {
    let mut a = Asm::new();
    for i in setup {
        a.push(*i);
    }
    a.push(insn);
    for word in 0..4u8 {
        a.push(ds(
            Op::ds_write_b32,
            NONE,
            v(1),
            v(20 + word),
            NONE,
            word * 4,
            0,
        ));
    }
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        lds_bytes: 1024,
        user_data,
        name: "bvh hlsl".to_owned(),
        ..StageProgram::new(Stage::Compute, a.finish())
    }
}

/// The descriptor at slot 1 of the table whose address is in s4, the first user data register
/// of a compute program.
fn from_table(insn: Insn) -> StageProgram {
    program(
        &[
            sop1(Op::s_mov_b32, s(5), int(0)),
            smem(Op::s_load_dwordx4, s(8), s(4), 0x20),
        ],
        insn,
        vec![UserDataSlot {
            sgpr: 4,
            kind: UserDataKind::Table,
        }],
    )
}

fn translated(p: &StageProgram) -> TranslatedStage {
    let mut t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("refused: {r}"));
    t.stages.remove(0)
}

fn compiles(p: &StageProgram) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("DXC unavailable, the node intersection compile is skipped");
        return;
    };
    let compiled = translate_to_dxil(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}\n{}", translated(p).hlsl));
    assert_eq!(&compiled[0].dxil[..4], b"DXBC");
}

#[test]
fn a_table_descriptor_becomes_a_byte_address_buffer_for_every_form() {
    for op in [Op::image_bvh_intersect_ray, Op::image_bvh64_intersect_ray] {
        for a16 in [false, true] {
            for nsa in [false, true] {
                let p = from_table(intersection(op, a16, nsa));
                let stage = translated(&p);
                let text = &stage.hlsl;
                assert!(
                    text.contains(
                        "ByteAddressBuffer heap_t_s4_1 = ResourceDescriptorHeap[decant_mirror[1].x + 1u];"
                    ),
                    "{text}"
                );
                assert_eq!(
                    text.matches("= decant_bvh_intersect(heap_t_s4_1, v0, ")
                        .count(),
                    1,
                    "one call for the four words\n{text}"
                );
                for word in ["x", "y", "z", "w"] {
                    assert!(text.contains(&format!("decant_bvh1.{word};")), "{text}");
                }
                let second = if nsa { "v90" } else { "v1" };
                let hi = if op == Op::image_bvh64_intersect_ray {
                    second
                } else {
                    "0u"
                };
                assert!(
                    text.contains(&format!("decant_bvh_intersect(heap_t_s4_1, v0, {hi}, ")),
                    "{text}"
                );
                let call = text
                    .lines()
                    .find(|l| l.contains("= decant_bvh_intersect("))
                    .expect("the call");
                assert_eq!(
                    call.contains("f16tof32(uint3("),
                    a16,
                    "the packed direction is unpacked only with A16\n{text}"
                );
                assert_eq!(
                    stage.resources.len(),
                    1,
                    "the descriptor is one resource\n{text}"
                );
                assert_eq!(
                    stage.resources[0].placement,
                    Placement::HeapAtMirror {
                        class: RegisterClass::Srv,
                        dword: 4,
                        slot: 1
                    }
                );
                compiles(&p);
            }
        }
    }
}

#[test]
fn the_helper_reads_the_public_node_layout() {
    let p = from_table(intersection(Op::image_bvh_intersect_ray, false, false));
    let text = translated(&p).hlsl;
    for fact in [
        // The node pointer counts 8 byte units above its three type bits.
        "uint kind = node_lo & 7u;",
        "uint at = (node_lo & 0xfffffff8u) << 3u;",
        // Box nodes: four child pointers, then four boxes of six floats or six half floats.
        "uint4 children = nodes.Load4(at);",
        "lo = asfloat(nodes.Load3(at + 16u + 24u * i));",
        "hi = asfloat(nodes.Load3(at + 28u + 24u * i));",
        "uint3 w = nodes.Load3(at + 16u + 12u * i);",
        "decant_bvh_box(nodes, at, kind == 4u, asfloat(extent), origin, inv_dir)",
        // Triangle nodes: three vertices of three floats.
        "float3 b = asfloat(nodes.Load3(at + 12u)) - origin;",
        "float3 c = asfloat(nodes.Load3(at + 24u)) - origin;",
        // A triangle miss and a box miss.
        "uint4(0x7f800000u, 0x3f800000u, 0u, 0u)",
        "uint4 hit = (uint4)0xffffffffu;",
    ] {
        assert!(text.contains(fact), "{fact}\n{text}");
    }
}

#[test]
fn a_descriptor_around_a_root_view_binds_its_register() {
    let mut p = program(
        &[
            sop1(Op::s_mov_b32, s(10), int(-1)),
            sop1(Op::s_mov_b32, s(11), lit(0x8000_0000)),
        ],
        intersection(Op::image_bvh64_intersect_ray, false, true),
        vec![UserDataSlot {
            sgpr: 8,
            kind: UserDataKind::RootSrv { register: 4 },
        }],
    );
    p.resources.push(DeclaredResource {
        class: RegisterClass::Srv,
        lower: 4,
        upper: 4,
        space: 1,
        psv_type: 4,
        psv_kind: 0,
    });
    let stage = translated(&p);
    let text = &stage.hlsl;
    assert!(
        text.contains("ByteAddressBuffer t4_sp1 : register(t4, space1);"),
        "{text}"
    );
    assert!(
        text.contains("= decant_bvh_intersect(t4_sp1, v0, v90, "),
        "{text}"
    );
    compiles(&p);
}

#[test]
fn a_descriptor_that_is_not_traced_is_refused_by_name() {
    // s8 to s11 hold user data at entry: no table, no root view, no constants.
    let p = program(
        &[],
        intersection(Op::image_bvh_intersect_ray, false, false),
        Vec::new(),
    );
    let refusal = translate(
        &[p],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(refusal.key(), "resource:partial-descriptor");
}

#[test]
fn unused_result_words_keep_the_call_for_the_words_that_are_read() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(5), int(0)))
        .push(smem(Op::s_load_dwordx4, s(8), s(4), 0x20))
        .push(intersection(Op::image_bvh_intersect_ray, false, false))
        .push(ds(Op::ds_write_b32, NONE, v(1), v(22), NONE, 0, 0))
        .push(sopp(Op::s_endpgm, 0));
    let p = StageProgram {
        threads: [64, 1, 1],
        lds_bytes: 256,
        user_data: vec![UserDataSlot {
            sgpr: 4,
            kind: UserDataKind::Table,
        }],
        ..StageProgram::new(Stage::Compute, a.finish())
    };
    let text = translated(&p).hlsl;
    assert_eq!(text.matches("= decant_bvh_intersect(").count(), 1, "{text}");
    assert!(text.contains("decant_bvh1.z;"), "{text}");
    assert!(!text.contains("decant_bvh1.x;"), "{text}");
    compiles(&p);
}
