//! The vertical slice: the shapes of the title's first shaders, built with the assembler,
//! translated to HLSL and compiled with DXC.
//!
//! * a blit pixel program: a constant buffer in a descriptor table times the vertex colour,
//!   exported as packed 16 bit floats,
//! * a textured pixel program: a texture in a table slot the root signature does not name, a
//!   sampler built from constants, whole quad mode around the interpolation,
//! * a user interface pair: a vertex program that reads rows of a constant buffer by an index
//!   from a vertex colour and writes four parameters, and a pixel program that reads all four.
//!
//! The programs are written here by hand. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::tables::Op;
use decant_shader::{RefusalKind, translate, translate_to_dxil};

fn element(semantic: &str, index: u32, register: u32, mask: u8, read: u8) -> SigElement {
    let system_value = match semantic {
        "SV_Position" => 1,
        "SV_Target" => 64,
        _ => 0,
    };
    SigElement {
        semantic: semantic.to_owned(),
        index,
        system_value,
        component_type: 3,
        register,
        mask,
        rw_mask: read,
    }
}

fn program(
    stage: Stage,
    code: Vec<u32>,
    inputs: Vec<SigElement>,
    outputs: Vec<SigElement>,
    name: &str,
) -> StageProgram {
    StageProgram {
        inputs,
        outputs,
        name: name.to_owned(),
        ..StageProgram::new(stage, code)
    }
}

fn target() -> Vec<SigElement> {
    vec![element("SV_Target", 0, 0, 15, 0)]
}

/// Interpolates the four components of attribute `attr` into `regs`, with the weights in v0, v1.
fn interpolate(a: &mut Asm, attr: u8, regs: [u8; 4]) {
    for (chan, r) in regs.into_iter().enumerate() {
        a.push(interp(Op::v_interp_p1_f32, v(r), v(0), attr, chan as u8));
    }
    for (chan, r) in regs.into_iter().enumerate() {
        a.push(interp(Op::v_interp_p2_f32, v(r), v(1), attr, chan as u8));
    }
}

fn pack_and_export(a: &mut Asm, rgba: [u8; 4]) {
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(rgba[0]), v(rgba[1])));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(rgba[2]), v(rgba[3])));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
}

/// Colour times dwords 8 to 11 of the constant buffer at slot 0 of the table in s8.
fn blit_pixel() -> StageProgram {
    let mut a = Asm::new();
    a.push(sopp(Op::s_inst_prefetch, 3));
    a.push(sop1(Op::s_mov_b32, s(2), s(9)));
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(sop1(Op::s_mov_b32, M0, s(2)));
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0));
    interpolate(&mut a, 0, [2, 3, 4, 5]);
    a.push(smem(Op::s_buffer_load_dwordx4, s(0), s(4), 0x20));
    for k in 0..4 {
        a.push(vop2(Op::v_mul_legacy_f32, v(2 + k), s(k), v(2 + k)));
    }
    pack_and_export(&mut a, [2, 3, 4, 5]);
    let inputs = vec![
        element("SV_Position", 0, 0, 15, 0),
        element("COLOR", 0, 1, 15, 15),
        element("TEXCOORD", 0, 2, 3, 0),
    ];
    let mut p = program(Stage::Pixel, a.finish(), inputs, target(), "blit_pixel");
    p.resources.push(DeclaredResource {
        class: RegisterClass::Cbv,
        lower: 27,
        upper: 27,
        space: 1,
        psv_type: 2,
        psv_kind: 13,
    });
    p.binding_table.push(TableBinding {
        class: RegisterClass::Cbv,
        register: 27,
        sgpr: 8,
        word_offset: 0,
    });
    p
}

/// A texture at slot 1 of the table in s8, a sampler from constants, times the colour.
fn textured_pixel() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b64, s(12), EXEC));
    a.push(sop1(Op::s_wqm_b64, EXEC, EXEC));
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x20));
    a.push(interp(Op::v_interp_p1_f32, v(2), v(0), 1, 0));
    a.push(interp(Op::v_interp_p1_f32, v(3), v(0), 1, 1));
    a.push(interp(Op::v_interp_p2_f32, v(2), v(1), 1, 0));
    a.push(interp(Op::v_interp_p2_f32, v(3), v(1), 1, 1));
    a.push(sop2(Op::s_and_b64, EXEC, EXEC, s(12)));
    a.push(sopk(Op::s_movk_i32, s(0), 0x400));
    a.push(sop1(Op::s_mov_b32, s(1), lit(0x00c0_0000)));
    a.push(sop1(Op::s_mov_b32, s(2), lit(0x18f0_0000)));
    a.push(sop1(Op::s_mov_b32, s(3), int(0)));
    a.push(image(Op::image_sample, v(11), v(2), s(4), s(0), 1, 0xf));
    interpolate(&mut a, 0, [6, 7, 8, 9]);
    for k in 0..4 {
        a.push(vop2(Op::v_mul_legacy_f32, v(2 + k), v(11 + k), v(6 + k)));
    }
    pack_and_export(&mut a, [2, 3, 4, 5]);
    let inputs = vec![
        element("SV_Position", 0, 0, 15, 0),
        element("COLOR", 0, 1, 15, 15),
        element("TEXCOORD", 0, 2, 3, 3),
    ];
    let mut p = program(Stage::Pixel, a.finish(), inputs, target(), "textured_pixel");
    p.binding_table.push(TableBinding {
        class: RegisterClass::Srv,
        register: 277,
        sgpr: 8,
        word_offset: 8,
    });
    p
}

fn ui_outputs() -> Vec<SigElement> {
    vec![
        element("COLOR", 0, 0, 15, 0),
        element("COLOR", 1, 1, 15, 0),
        element("TEXCOORD", 0, 2, 15, 0),
        element("TEXCOORD", 1, 3, 15, 0),
        element("SV_Position", 0, 4, 15, 0),
    ]
}

/// Position from two rows of a constant buffer chosen by the third component of COLOR1, colours
/// passed through, two more rows as texture coordinates.
fn ui_vertex() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_swappc_b64, s(4), s(4)));
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(sop1(Op::s_mov_b32, s(4), lit(255.0f32.to_bits())));
    a.push(smem(Op::s_load_dwordx4, s(0), s(8), 0));
    // Rows k + 2 and k + 3 for the position, rows k and k + 1 for the pixel program.
    for (row, first) in [(2.5f32, 0u8), (3.5, 14), (0.5, 18), (1.5, 22)] {
        a.push(vop3(Op::v_fma_f32, v(26), v(10), s(4), lit(row.to_bits())));
        a.push(vop1(Op::v_cvt_i32_f32, v(26), v(26)));
        a.push(buffer_load(
            Op::buffer_load_dwordx4,
            v(first),
            v(26),
            s(0),
            0,
            true,
        ));
    }
    a.push(vop3(Op::v_fma_f32, v(1), v(13), v(1), v(3)));
    a.push(vop3(Op::v_fma_f32, v(2), v(13), v(15), v(17)));
    a.push(vop2(Op::v_fmac_f32, v(1), v(12), v(0)));
    a.push(vop2(Op::v_fmac_f32, v(2), v(12), v(14)));
    a.push(vop1(Op::v_mov_b32, v(3), int(0)));
    a.push(vop1(Op::v_mov_b32, v(16), float(1.0)));
    a.push(export(12, 0xf, [1, 2, 3, 16], false, true, false));
    a.push(export(32, 0xf, [4, 5, 6, 7], false, false, false));
    a.push(export(33, 0xf, [8, 9, 10, 11], false, false, false));
    a.push(export(34, 0xf, [18, 19, 20, 21], false, false, false));
    a.push(export(35, 0xf, [22, 23, 24, 25], false, false, false));
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("COLOR", 0, 0, 15, 15),
        element("COLOR", 1, 1, 15, 15),
        element("POSITION", 0, 2, 3, 3),
    ];
    let mut p = program(Stage::Vertex, a.finish(), inputs, ui_outputs(), "ui_vertex");
    p.resources.push(DeclaredResource {
        class: RegisterClass::Cbv,
        lower: 0,
        upper: 0,
        space: 1,
        psv_type: 2,
        psv_kind: 13,
    });
    p
}

/// rgb = saturate(tc0 + colour0 * tc1), alpha times the alpha of colour 1. Four inputs, read
/// through four different attribute numbers.
fn ui_pixel() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, M0, s(12)));
    interpolate(&mut a, 0, [10, 11, 12, 13]);
    interpolate(&mut a, 2, [14, 15, 16, 17]);
    interpolate(&mut a, 3, [18, 19, 20, 21]);
    a.push(interp(Op::v_interp_p1_f32, v(22), v(0), 1, 3));
    a.push(interp(Op::v_interp_p2_f32, v(22), v(1), 1, 3));
    for k in 0..4 {
        a.push(vop2(Op::v_fmac_f32, v(14 + k), v(10 + k), v(18 + k)));
        a.push(vop3(
            Op::v_med3_f32,
            v(2 + k),
            v(14 + k),
            int(0),
            float(1.0),
        ));
    }
    a.push(vop2(Op::v_mul_legacy_f32, v(5), v(5), v(22)));
    pack_and_export(&mut a, [2, 3, 4, 5]);
    // The inputs are the vertex outputs without the position. COLOR1 is read in alpha only.
    let inputs = vec![
        element("COLOR", 0, 0, 15, 15),
        element("COLOR", 1, 1, 15, 8),
        element("TEXCOORD", 0, 2, 15, 15),
        element("TEXCOORD", 1, 3, 15, 15),
    ];
    program(Stage::Pixel, a.finish(), inputs, target(), "ui_pixel")
}

/// The root signature of the blit pipelines: one table with one constant buffer, `b27` space 1.
fn one_table_root() -> RootSignature {
    let range = RootRange {
        class: RegisterClass::Cbv,
        count: 1,
        base_register: 27,
        space: 1,
        offset: 0,
    };
    RootSignature {
        params: vec![RootParam {
            kind: RootParamKind::Table(vec![range]),
            visibility: 0,
        }],
        ..RootSignature::default()
    }
}

/// Compiles every stage when `dxc.exe` is there, and says so when it is not.
fn compile(stages: &[StageProgram], root: &RootSignature, host: &CountingHost) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    let compiled = translate_to_dxil(stages, root, &Options::default(), host, &dxc)
        .unwrap_or_else(|r| panic!("{r}"));
    for c in compiled {
        assert_eq!(&c.dxil[..4], b"DXBC", "{:?}", c.stage);
    }
}

#[test]
fn a_table_buffer_is_read_from_the_heap_even_when_a_range_names_it() {
    let (p, root, host) = (blit_pixel(), one_table_root(), CountingHost::default());
    let t = translate(std::slice::from_ref(&p), &root, &Options::default(), &host).unwrap();
    let s = &t.stages[0];
    let placement = Placement::HeapAtMirror {
        class: RegisterClass::Cbv,
        dword: 8,
        slot: 0,
    };
    assert_eq!(
        s.resources,
        [BoundResource {
            name: "heap_b_s8_0".to_owned(),
            placement
        }]
    );
    assert!(
        s.hlsl.contains(
            "ConstantBuffer<decant_words> heap_b_s8_0 = ResourceDescriptorHeap[decant_mirror[2].x + 0u];"
        ),
        "{}",
        s.hlsl
    );
    assert!(!s.hlsl.contains("register(b27"), "{}", s.hlsl);
    // Dwords 8 to 11: byte 32 and up.
    assert!(s.hlsl.contains("heap_b_s8_0.w[2][0]"), "{}", s.hlsl);
    // Attribute 0 is the colour: the first input that is no system value and is read.
    assert!(
        s.hlsl.contains("float4 i1 : COLOR0;") && s.hlsl.contains("asuint(decant_in.i1.w)"),
        "{}",
        s.hlsl
    );
    // Only the table base is read from the mirror: the register the program copies to M0 is dead.
    assert_eq!(
        s.needs,
        Needs {
            mirror_dwords: vec![8],
            resource_heap: true,
            sampler_heap: false,
            ..Needs::default()
        }
    );
    assert!(!s.hlsl.contains("decant_mirror[2].y"), "{}", s.hlsl);
    assert_eq!((s.profile.as_str(), s.entry.as_str()), ("ps_6_6", "main"));
    assert_eq!(s.dxc_args, ["-Gis", "-denorm", "ftz"]);
    compile(&[p], &root, &host);
}

/// The table of the user interface pipeline: `t2` then `b0`, so the PC layout has `b0` at slot 1
/// while the console program reads its constant buffer at slot 0.
#[test]
fn the_slot_comes_from_the_machine_code_not_from_the_range_order() {
    let srv = RootRange {
        class: RegisterClass::Srv,
        count: 1,
        base_register: 2,
        space: 1,
        offset: 0,
    };
    let cbv = RootRange {
        class: RegisterClass::Cbv,
        count: 1,
        base_register: 0,
        space: 1,
        offset: 1,
    };
    let root = RootSignature {
        params: vec![RootParam {
            kind: RootParamKind::Table(vec![srv, cbv]),
            visibility: 0,
        }],
        ..RootSignature::default()
    };
    let mut vs = ui_vertex();
    vs.binding_table.push(TableBinding {
        class: RegisterClass::Cbv,
        register: 0,
        sgpr: 8,
        word_offset: 0,
    });
    let host = CountingHost::default();
    let t = translate(std::slice::from_ref(&vs), &root, &Options::default(), &host).unwrap();
    // Not `t2` (the range at slot 0) and not `b0` (which the PC would read from slot 1): the heap
    // at the slot the machine code names. Without the binding table record the result is the same,
    // the class then comes from the declared resources.
    assert_eq!(
        t.stages[0].resources[0].placement,
        Placement::HeapAtMirror {
            class: RegisterClass::Cbv,
            dword: 8,
            slot: 0
        }
    );
    compile(std::slice::from_ref(&vs), &root, &host);
    vs.binding_table.clear();
    let t = translate(&[vs], &root, &Options::default(), &host).unwrap();
    assert_eq!(
        t.stages[0].resources[0].placement,
        Placement::HeapAtMirror {
            class: RegisterClass::Cbv,
            dword: 8,
            slot: 0
        }
    );
}

#[test]
fn a_write_to_the_null_register_is_dropped() {
    let mut a = Asm::new();
    a.push(sop1(
        Op::s_mov_b32,
        decant_shader::decode::Operand::Reg(decant_shader::decode::Reg::Null),
        int(0),
    ));
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    pack_and_export(&mut a, [2, 2, 2, 2]);
    let t = translate(
        &[program(
            Stage::Pixel,
            a.finish(),
            Vec::new(),
            target(),
            "null",
        )],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap();
    assert!(!t.stages[0].hlsl.contains("t65535"), "{}", t.stages[0].hlsl);
}

#[test]
fn a_table_slot_the_root_signature_lacks_is_read_from_the_heap_through_the_mirror() {
    let (p, root, host) = (textured_pixel(), one_table_root(), CountingHost::default());
    let t = translate(std::slice::from_ref(&p), &root, &Options::default(), &host).unwrap();
    let s = &t.stages[0];
    let words = [0x400, 0x00c0_0000, 0x18f0_0000, 0];
    assert_eq!(
        s.resources.iter().map(|r| r.placement).collect::<Vec<_>>(),
        [
            Placement::HeapAtMirror {
                class: RegisterClass::Srv,
                dword: 8,
                slot: 1
            },
            Placement::HeapSampler { words, index: 0 }
        ]
    );
    assert_eq!(
        s.needs,
        Needs {
            mirror_dwords: vec![8],
            resource_heap: true,
            sampler_heap: true,
            ..Needs::default()
        }
    );
    assert_eq!(host.samplers(), [words]);
    assert!(
        s.hlsl.contains(
            "Texture2D<float4> heap_t_s8_1 = ResourceDescriptorHeap[decant_mirror[2].x + 1u];"
        ),
        "{}",
        s.hlsl
    );
    assert!(
        s.hlsl
            .contains("SamplerState smp_0 = SamplerDescriptorHeap[0u];"),
        "{}",
        s.hlsl
    );
    assert!(
        s.hlsl
            .contains("heap_t_s8_1.Sample(smp_0, float2(asfloat(v2), asfloat(v3)))"),
        "{}",
        s.hlsl
    );
    // The texture coordinate is attribute 1, the third struct member.
    assert!(s.hlsl.contains("asuint(decant_in.i2.y)"), "{}", s.hlsl);
    compile(&[p], &root, &host);
    // Without the mirror the same program is refused by name.
    let refusal = translate(
        &[textured_pixel()],
        &root,
        &Options {
            mirror: None,
            ..Options::default()
        },
        &host,
    )
    .unwrap_err();
    assert_eq!(refusal.key(), "resource:no-mirror");
    assert_eq!(refusal.stage, Some(Stage::Pixel));
}

#[test]
fn vertex_parameters_and_pixel_attributes_link_element_by_element() {
    let (stages, host) = ([ui_vertex(), ui_pixel()], CountingHost::default());
    let t = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap();
    let (vs, ps) = (&t.stages[0].hlsl, &t.stages[1].hlsl);
    // The pixel input struct is the vertex output struct, member by member.
    let members = |text: &str, name: &str| {
        let body = text
            .split(&format!("struct {name} {{"))
            .nth(1)
            .unwrap()
            .split("};")
            .next()
            .unwrap()
            .to_owned();
        // Type and semantic of each member, without the member name.
        body.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split_whitespace()
                    .enumerate()
                    .filter(|(i, _)| *i != 1)
                    .map(|(_, w)| w)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
    };
    let out = members(vs, "DecantOut");
    assert_eq!(
        out,
        [
            "float4 : COLOR0;",
            "float4 : COLOR1;",
            "float4 : TEXCOORD0;",
            "float4 : TEXCOORD1;",
            "float4 : SV_Position;"
        ]
    );
    assert_eq!(members(ps, "DecantIn"), out);
    // Parameter exports 32 to 35 fill members 0 to 3, the position export fills member 4.
    for (member, reg) in [(0, 4), (1, 8), (2, 18), (3, 22), (4, 1)] {
        assert!(
            vs.contains(&format!("decant_out.o{member}.x = asfloat(v{reg});")),
            "member {member}\n{vs}"
        );
    }
    // The pixel program reads four different members: attribute 1 is COLOR1, read in alpha.
    for (member, comp, reg) in [(0, "x", 10), (1, "w", 22), (2, "y", 15), (3, "z", 20)] {
        assert!(
            ps.contains(&format!("v{reg} = asuint(decant_in.i{member}.{comp});")),
            "member {member}\n{ps}"
        );
    }
    // The constant buffer rows are indexed: 16 bytes per row.
    assert!(vs.contains("(v26 * 16u + 0u)"), "{vs}");
    assert_eq!(
        t.stages[0].needs,
        Needs {
            mirror_dwords: vec![8],
            resource_heap: true,
            sampler_heap: false,
            ..Needs::default()
        }
    );
    assert_eq!(t.stages[1].needs, Needs::default());
    compile(&stages, &RootSignature::default(), &host);
}

#[test]
fn a_pixel_input_the_stage_before_does_not_write_is_refused() {
    let mut ps = ui_pixel();
    ps.inputs[3].semantic = "NORMAL".to_owned();
    let r = translate(
        &[ui_vertex(), ps],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(r.key(), "stage:link-missing");
}

#[test]
fn manifest_census_uses_the_root_and_links_the_stages() {
    use decant_shader::census::pipeline::{PipelineLists, run};
    use decant_shader::container::fnv1a64;
    use std::collections::HashMap;
    use std::io::Write;

    let mirror = "2 0xc01 table:0:t.2.1.1.0.8+b.0.1.1.1.0 cbv:0:1.15";
    let no_mirror = "2 0x1 table:0:t.2.1.1.0.8+b.0.1.1.1.0";
    let mirror_id = fnv1a64(mirror.as_bytes());
    let no_mirror_id = fnv1a64(no_mirror.as_bytes());
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "# decant pipeline manifest 1").unwrap();
    for (root, msaa, pixel) in [
        (mirror_id, 1, 2),
        (mirror_id, 4, 2),
        (mirror_id, 1, 3),
        (no_mirror_id, 1, 2),
    ] {
        writeln!(file, "P {root:016x} topo=3 msaa={msaa} ds=0 rt=28 so=0 in= VS 0000000000000001 PS {pixel:016x}").unwrap();
    }
    writeln!(file, "R {mirror_id:016x} {mirror}").unwrap();
    writeln!(file, "R {no_mirror_id:016x} {no_mirror}").unwrap();

    let mut lists = PipelineLists::default();
    lists.read(file.path()).unwrap();
    let mut bad_pixel = ui_pixel();
    bad_pixel.inputs[3].semantic = "NORMAL".to_owned();
    let programs = HashMap::from([(1, ui_vertex()), (2, ui_pixel()), (3, bad_pixel)]);
    let result = run(&programs, &lists, 2);
    assert_eq!(result.pipelines, 4);
    assert_eq!(result.translated, 2, "{:?}", result.outcomes);
    assert_eq!(result.outcomes.len(), 4);
    assert_eq!(result.blockers["stage:link-missing"].count, 1);
    assert_eq!(result.blockers["resource:no-mirror"].count, 1);
}

/// A flat address load: the title's programs never use that format and the lifter has no rule
/// file for it, so this test does not depend on which instructions the rule files know.
#[test]
fn what_the_lifter_does_not_know_is_refused_by_instruction_name() {
    use decant_shader::decode::{Ext, Form, Insn, Operand};
    let op = Op::flat_load_dword;
    let word = (0b110111 << 26) | (u32::from(op.info().opcode) << 18);
    let flat = Insn {
        pc: 0,
        len: 0,
        op,
        form: Form::Plain,
        dst: Operand::None,
        dst2: Operand::None,
        src: [Operand::None; 4],
        ext: Ext::Flat(word, 0),
    };
    let mut a = Asm::new();
    a.push(flat);
    a.push(sopp(Op::s_endpgm, 0));
    let stages = [program(
        Stage::Pixel,
        a.finish(),
        Vec::new(),
        target(),
        "gap",
    )];
    let r = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(r.kind, RefusalKind::Op(op));
    assert_eq!((r.key().as_str(), r.pc), ("op:flat_load_dword", Some(6)));
}

/// A stage that is not written yet is refused before anything is decoded: the code here is
/// the signature followed by words that are no instructions.
#[test]
fn a_stage_that_is_not_written_is_refused_without_decoding() {
    let mut code = SIGNATURE.to_vec();
    code.extend([0xFFFF_FFFF; 4]);
    let hull = StageProgram::new(Stage::Hull, code.clone());
    let stages = [ui_vertex(), hull];
    let r = decant_shader::can_translate(&stages).unwrap_err();
    assert_eq!(
        (r.key().as_str(), r.stage),
        ("stage:tessellation-not-written", Some(Stage::Hull))
    );
    let host = CountingHost::default();
    let r = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap_err();
    assert_eq!(r.key(), "stage:tessellation-not-written");
    // The probe goes on and finds the words that do not decode.
    let hull = StageProgram::new(Stage::Hull, code);
    let r = decant_shader::translate_probe(
        &[hull],
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap_err();
    assert_eq!(r.key(), "decode");
    assert!(decant_shader::can_translate(&[ui_vertex(), ui_pixel()]).is_ok());
    // Code without the signature is refused as cheaply.
    let r =
        decant_shader::can_translate(&[StageProgram::new(Stage::Pixel, vec![0; 3])]).unwrap_err();
    assert_eq!(r.key(), "container:no-signature");
}

#[test]
fn the_cache_key_follows_code_root_signature_and_options_but_not_names() {
    use decant_shader::cache::key;
    let (p, root, opts) = (blit_pixel(), one_table_root(), Options::default());
    let base = key(std::slice::from_ref(&p), &root, &opts, 0);
    let mut renamed = p.clone();
    renamed.name = "other".to_owned();
    assert_eq!(key(&[renamed], &root, &opts, 0), base);
    let mut changed = p.clone();
    *changed.code.last_mut().unwrap() ^= 1;
    assert_ne!(key(&[changed], &root, &opts, 0), base);
    assert_ne!(
        key(
            std::slice::from_ref(&p),
            &RootSignature::default(),
            &opts,
            0
        ),
        base
    );
    assert_ne!(
        key(
            std::slice::from_ref(&p),
            &root,
            &Options {
                mirror: None,
                ..opts.clone()
            },
            0
        ),
        base
    );
    assert_ne!(key(std::slice::from_ref(&p), &root, &opts, 1), base);
}

/// A table the compiler placed behind the code: `s_getpc_b64` gives the address of the next
/// instruction as an offset into the program's own words, and a read of the code at a constant
/// address is the word that is there.
#[test]
fn a_read_of_the_programs_own_code_at_a_constant_address_is_its_word() {
    use decant_shader::hlsl::{self, Context};
    use decant_shader::ir::{Expr, Function, Guard, Node, OutputRef, Stmt, Var};
    use decant_shader::stage::{Field, Interface};
    // The lifter: the pair gets the offset of the word after the instruction.
    let mut a = Asm::new();
    a.push(sop1(
        Op::s_getpc_b64,
        s(2),
        decant_shader::decode::Operand::None,
    ));
    a.push(sopp(Op::s_endpgm, 0));
    let p = program(Stage::Pixel, a.finish(), Vec::new(), target(), "getpc");
    let func = decant_shader::lift_program(&p).unwrap();
    let mut found = false;
    func.for_each_stmt(&mut |st| {
        found |= matches!(
            st,
            Stmt::Set {
                value: Expr::CodeAddress(28),
                ..
            }
        )
    });
    assert!(found, "word 6 is the instruction, word 7 the next: byte 28");
    // The printer: a constant address folds to the word, a computed one keeps the array.
    let code = [10u32, 11, 12, 0x3f80_0000];
    let field = Field {
        semantic: "SV_Target".to_owned(),
        component_type: 3,
        comps: 1,
        first_comp: 0,
        interpolation: "",
    };
    let iface = Interface {
        outputs: vec![field],
        ..Interface::default()
    };
    let print = |at: Expr| {
        let read = Expr::CodeWord(Box::new(Expr::bin(
            decant_shader::ir::Bin::Add,
            Expr::CodeAddress(8),
            at,
        )));
        let set = Stmt::Set {
            dst: Var::S(0),
            value: read,
            guard: Guard::Always,
        };
        let out = Stmt::Output {
            to: OutputRef {
                element: 0,
                chan: 0,
            },
            value: Expr::Var(Var::S(0)),
        };
        let f = Function {
            body: vec![Node::Block(vec![set, out]), Node::Return],
            resources: Vec::new(),
            groupshared_dwords: 0,
            lanes: 64,
        };
        hlsl::print(
            &f,
            &Context {
                iface: &iface,
                mirror: None,
                header: &[],
                code: &code,
            },
        )
        .unwrap()
    };
    let constant = print(Expr::U32(4));
    assert!(
        constant.contains("1065353216u") && !constant.contains("decant_code"),
        "{constant}"
    );
    let computed = print(Expr::Var(Var::S(5)));
    assert!(
        computed.contains("static const uint decant_code[4] = { 0xa, 0xb, 0xc, 0x3f800000 };"),
        "{computed}"
    );
    // A computed address goes through the bounded read: a lane that is not active computes it
    // too, from whatever its registers hold.
    assert!(
        computed.contains(
            "uint decant_code_at(uint a) { uint i = a >> 2; return i < 4u ? decant_code[i] : 0u; }"
        ),
        "{computed}"
    );
    assert!(computed.contains("decant_code_at((8u + s5))"), "{computed}");
}

/// A global load decodes with its registers and round trips through the assembler.
#[test]
fn a_global_load_carries_its_registers() {
    use decant_shader::decode::{Operand, Reg, decode_one, encode};
    let i = global_load(Op::global_load_dword, 56, 42, 8);
    let words = encode(&i);
    let back = decode_one(&words, 0).unwrap();
    assert_eq!(back.op, Op::global_load_dword);
    assert_eq!(
        (back.dst, back.src[0], back.src[1]),
        (v(56), v(42), Operand::Reg(Reg::Null))
    );
    assert_eq!(words.len(), 2);
}

/// The 64 bit vector shifts work on a register pair: both halves of the result are computed
/// from the old pair before either is written.
#[test]
fn a_64_bit_vector_shift_moves_both_halves_of_its_pair() {
    let mut a = Asm::new();
    a.push(vop3(
        Op::v_lshlrev_b64_e64,
        v(2),
        v(4),
        v(2),
        decant_shader::decode::Operand::None,
    ));
    pack_and_export(&mut a, [2, 3, 2, 3]);
    let p = program(Stage::Pixel, a.finish(), Vec::new(), target(), "shift64");
    let host = CountingHost::default();
    let t = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap();
    let text = &t.stages[0].hlsl;
    assert!(text.contains("decant_shl64(decant_pack64(v2, v3), v4)"), "{text}");
    assert!(text.contains("decant_hi64("), "{text}");
    compile(&[p], &RootSignature::default(), &host);
}
