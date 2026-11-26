//! The vertex stage, with programs built by the assembler. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

fn element(semantic: &str, system_value: u32, mask: u8) -> SigElement {
    SigElement {
        semantic: semantic.to_owned(),
        index: 0,
        system_value,
        component_type: 3,
        register: 0,
        mask,
        rw_mask: 0,
    }
}

fn vertex(code: Vec<u32>, inputs: Vec<SigElement>, outputs: Vec<SigElement>) -> StageProgram {
    StageProgram {
        inputs,
        outputs,
        name: "vertex_test".to_owned(),
        ..StageProgram::new(Stage::Vertex, code)
    }
}

fn hlsl_of(p: &StageProgram) -> Result<String, decant_shader::Refusal> {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .map(|t| t.stages[0].hlsl.clone())
}

/// A position export and a parameter export that the output signature does not list: the
/// compiler left the parameter in because the pixel stage never reads it.
#[test]
fn a_parameter_nothing_reads_is_dropped() {
    let mut a = Asm::new();
    a.push(export(12, 0xf, [4, 5, 6, 7], false, true, true));
    a.push(export(32, 0xf, [6, 6, 6, 7], false, false, false));
    a.push(sopp(Op::s_endpgm, 0));
    let p = vertex(
        a.finish(),
        vec![element("POSITION", 0, 7)],
        vec![element("SV_Position", 1, 15)],
    );
    let hlsl = hlsl_of(&p).expect("translates");
    assert!(hlsl.contains("SV_Position"));
}

/// The signature lists the vertex and instance index and the program reads both: the struct
/// declares each once, and DXC accepts it.
#[test]
fn vertex_and_instance_index_compile() {
    let mut a = Asm::new();
    a.push(vop1(Op::v_mov_b32, v(4), v(0)));
    a.push(vop1(Op::v_mov_b32, v(5), v(3)));
    a.push(export(12, 0xf, [4, 5, 4, 5], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut instance = element("SV_InstanceID", 8, 1);
    instance.register = 2;
    instance.rw_mask = 1;
    let mut index = element("SV_VertexID", 6, 1);
    index.rw_mask = 1;
    let p = vertex(
        a.finish(),
        vec![index, instance],
        vec![element("SV_Position", 1, 15)],
    );
    let hlsl = hlsl_of(&p).expect("translates");
    assert_eq!(hlsl.matches("SV_VertexID").count(), 1, "{hlsl}");
    assert_eq!(hlsl.matches("SV_InstanceID").count(), 1, "{hlsl}");
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped: no dxc");
        return;
    };
    translate_to_dxil(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .expect("DXC compiles it");
}

/// An element of an interface, for the linking tests.
fn el(semantic: &str, index: u32, sv: u32, register: u32, mask: u8, ty: u32) -> SigElement {
    SigElement {
        semantic: semantic.to_owned(),
        index,
        system_value: sv,
        component_type: ty,
        register,
        mask,
        rw_mask: 0,
    }
}

fn param(slot: u8, kind: u8, register: u32, flags: u8) -> ParamExport {
    ParamExport {
        slot,
        kind,
        register,
        sixteen_bit_register: if flags == 3 { register } else { 0 },
        flags,
    }
}

fn input(attr: u8, kind: u8, register: u32, high: Option<u32>, flags: u8) -> PixelInput {
    PixelInput {
        attr,
        kind,
        register,
        high_register: high,
        channels: 0xf,
        flags,
    }
}

/// Translates a vertex and pixel pair as one pipeline and compiles both stages.
fn link(vs: &StageProgram, ps: &StageProgram) -> (String, String) {
    let stages = [vs.clone(), ps.clone()];
    let t = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .expect("the pipeline translates");
    if let Some(dxc) = common::find_dxc() {
        translate_to_dxil(
            &stages,
            &RootSignature::default(),
            &Options::default(),
            &CountingHost::default(),
            &dxc,
        )
        .expect("DXC compiles both stages");
    } else {
        eprintln!("skipped the DXC compile: no dxc");
    }
    (t.stages[0].hlsl.clone(), t.stages[1].hlsl.clone())
}

/// Builds a pixel program that interpolates `reads` (attribute, channel) into v2 upward and
/// exports the first four as one colour.
fn pixel_reading(
    reads: &[(u8, u8)],
    inputs: Vec<SigElement>,
    table: Vec<PixelInput>,
) -> StageProgram {
    let mut a = Asm::new();
    for (k, (attr, chan)) in reads.iter().enumerate() {
        a.push(interp(
            Op::v_interp_p1_f32,
            v(2 + k as u8),
            v(0),
            *attr,
            *chan,
        ));
        a.push(interp(
            Op::v_interp_p2_f32,
            v(2 + k as u8),
            v(1),
            *attr,
            *chan,
        ));
    }
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(10), v(2), v(3)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(11), v(2), v(3)));
    a.push(export(0, 0xf, [10, 11, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs,
        outputs: vec![el("SV_Target", 0, 64, 0, 15, 3)],
        pixel_inputs: table,
        name: "pixel_linked".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

/// Exports `slots` (target, enable mask, four registers) after the position.
fn vertex_exporting(
    slots: &[(u8, u8, [u8; 4])],
    outputs: Vec<SigElement>,
    table: Vec<ParamExport>,
) -> StageProgram {
    let mut a = Asm::new();
    a.push(export(12, 0xf, [4, 5, 6, 7], false, true, false));
    for (target, en, regs) in slots {
        a.push(export(*target, *en, *regs, false, false, false));
    }
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs: vec![el("POSITION", 0, 0, 0, 15, 3)],
        outputs,
        param_exports: table,
        name: "vertex_linked".to_owned(),
        ..StageProgram::new(Stage::Vertex, a.finish())
    }
}

#[test]
fn two_elements_in_one_register_link_by_name_at_their_own_components() {
    // The vertex side puts TEXCOORD0 at r0.xy and TEXCOORD1 at r0.zw. The pixel side declares
    // them the other way round: TEXCOORD1 at r0.xy, TEXCOORD0 at r0.zw.
    let vs = vertex_exporting(
        &[(32, 0xf, [8, 9, 10, 11])],
        vec![
            el("TEXCOORD", 0, 0, 0, 0b0011, 3),
            el("TEXCOORD", 1, 0, 0, 0b1100, 3),
            el("SV_Position", 0, 1, 1, 15, 3),
        ],
        vec![param(0, 6, 0, 0)],
    );
    let ps = pixel_reading(
        &[(0, 0), (0, 2)],
        vec![
            el("TEXCOORD", 1, 0, 0, 0b0011, 3),
            el("TEXCOORD", 0, 0, 0, 0b1100, 3),
        ],
        vec![input(0, 6, 0, None, 0x02)],
    );
    let (vhlsl, phlsl) = link(&vs, &ps);
    // The vertex side writes each register component to its own element, and a member is as
    // wide as its element: TEXCOORD1 is a float2 that starts at .z of the register.
    assert!(vhlsl.contains("float2 o1 : TEXCOORD1;"), "{vhlsl}");
    assert!(vhlsl.contains("decant_out.o0.x = asfloat(v8)"), "{vhlsl}");
    assert!(vhlsl.contains("decant_out.o1.x = asfloat(v10)"), "{vhlsl}");
    // Pixel attribute 0 channel 0 is its TEXCOORD1.x. The vertex side has that element at .z of
    // the register, which is the first component of its member.
    assert!(phlsl.contains("v2 = asuint(decant_in.i1.x)"), "{phlsl}");
    // Channel 2 is the pixel side's TEXCOORD0.x, at .x on the vertex side.
    assert!(phlsl.contains("v3 = asuint(decant_in.i0.x)"), "{phlsl}");
}

#[test]
fn a_16_bit_pair_shares_one_slot() {
    // TEXCOORD0.xy and TEXCOORD1.xyzw, both 16 bit: slot 0 channel 0 holds TEXCOORD0, channels
    // 1 and 2 hold TEXCOORD1 (half 5 is TEXCOORD1.w).
    let outputs = vec![
        el("TEXCOORD", 0, 0, 0, 0b0011, 6),
        el("TEXCOORD", 1, 0, 1, 0b1111, 6),
        el("SV_Position", 0, 1, 2, 15, 3),
    ];
    let vs = vertex_exporting(
        &[(32, 0x7, [8, 9, 10, 0])],
        outputs.clone(),
        vec![param(0, 6, 0, 3)],
    );
    let mut a = Asm::new();
    let f16 = |op: Op, dst: u8, chan: u8, high: bool| decant_shader::decode::Insn {
        pc: 0,
        len: 0,
        op,
        form: decant_shader::decode::Form::Plain,
        dst: v(dst),
        dst2: decant_shader::decode::Operand::None,
        src: [
            v(1),
            if op == Op::v_interp_p2_f16 {
                v(dst)
            } else {
                decant_shader::decode::Operand::None
            },
            decant_shader::decode::Operand::None,
            decant_shader::decode::Operand::None,
        ],
        ext: decant_shader::decode::Ext::VintrpE64 {
            attr: 0,
            chan,
            high,
            abs: 0,
            neg: 0,
            opsel: 0,
            omod: 0,
            clamp: false,
        },
    };
    a.push(f16(Op::v_interp_p1ll_f16, 6, 2, true));
    a.push(f16(Op::v_interp_p2_f16, 2, 2, true));
    a.push(export(0, 0xf, [2, 2, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut ins = outputs.clone();
    ins.truncate(2);
    let ps = StageProgram {
        inputs: ins,
        outputs: vec![el("SV_Target", 0, 64, 0, 15, 3)],
        pixel_inputs: vec![input(0, 6, 0, Some(1), 0x1a)],
        name: "pixel_16".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    let (vhlsl, phlsl) = link(&vs, &ps);
    // TEXCOORD1.w is the high half of channel 2 (v10) on the vertex side.
    assert!(
        vhlsl.contains("decant_out.o1.w = asfloat(decant_vertex_f16((v10 >> (16u & 31u))))"),
        "{vhlsl}"
    );
    assert!(
        vhlsl.contains("decant_out.o0.y = asfloat(decant_vertex_f16((v8 >> (16u & 31u))))"),
        "{vhlsl}"
    );
    assert!(phlsl.contains("decant_in.i1.w"), "{phlsl}");
}

#[test]
fn a_clip_distance_slot_links() {
    let vs = vertex_exporting(
        &[(32, 0xf, [8, 9, 10, 11]), (33, 0x1, [12, 0, 0, 0])],
        vec![
            el("TEXCOORD", 0, 0, 0, 15, 3),
            el("SV_ClipDistance", 0, 2, 1, 1, 3),
            el("SV_Position", 0, 1, 2, 15, 3),
        ],
        vec![param(0, 6, 0, 0), param(1, 7, 1, 0)],
    );
    let ps = pixel_reading(
        &[(0, 0), (1, 0)],
        vec![
            el("TEXCOORD", 0, 0, 0, 15, 3),
            el("SV_ClipDistance", 0, 2, 1, 1, 3),
        ],
        vec![input(0, 6, 0, None, 0x02), input(1, 7, 1, None, 0x02)],
    );
    let (vhlsl, phlsl) = link(&vs, &ps);
    assert!(vhlsl.contains("decant_out.o1 = asfloat(v12)"), "{vhlsl}");
    assert!(phlsl.contains("v3 = asuint(decant_in.i1)"), "{phlsl}");
}

#[test]
fn pos1_exports_render_target_and_viewport_array_indices() {
    let mut a = Asm::new();
    a.push(sopp(Op::s_swappc_b64, 0));
    a.push(export(32, 0xf, [8, 9, 10, 11], false, false, false));
    a.push(export(12, 0xf, [4, 5, 6, 7], false, false, false));
    a.push(export(13, 0b1101, [12, 0, 13, 14], false, true, false));
    a.push(sopp(Op::s_endpgm, 0));
    let vs = StageProgram {
        inputs: vec![el("POSITION", 0, 0, 0, 15, 3)],
        outputs: vec![
            el("TEXCOORD", 0, 0, 0, 15, 3),
            el("SV_Position", 0, 1, 1, 15, 3),
            el("SV_RenderTargetArrayIndex", 0, 4, 2, 1, 1),
            el("SV_ViewportArrayIndex", 0, 5, 3, 1, 1),
        ],
        param_exports: vec![param(0, 6, 0, 0)],
        name: "vertex_pos1".to_owned(),
        ..StageProgram::new(Stage::Vertex, a.finish())
    };
    let ps = pixel_reading(
        &[(0, 0)],
        vec![el("TEXCOORD", 0, 0, 0, 15, 3)],
        vec![input(0, 6, 0, None, 0x02)],
    );
    let (vhlsl, _phlsl) = link(&vs, &ps);
    assert!(vhlsl.contains("uint o2 : SV_RenderTargetArrayIndex;"), "{vhlsl}");
    assert!(vhlsl.contains("uint o3 : SV_ViewportArrayIndex;"), "{vhlsl}");
    assert!(vhlsl.contains("decant_out.o2 = v13;"), "{vhlsl}");
    assert!(vhlsl.contains("decant_out.o3 = v14;"), "{vhlsl}");
}
