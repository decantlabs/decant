//! The pixel stage: attributes from the container's input table, 16 bit and flat
//! interpolation, the entry registers (position, front face), the export targets (compacted
//! colour targets, integer targets, depth, the null target of the kill exit).
//!
//! The programs are written here by hand. No game data is involved. Each translation is also
//! compiled with DXC when `dxc.exe` is found.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn, Operand};
use decant_shader::tables::Op;
use decant_shader::{Refusal, RefusalKind, translate, translate_to_dxil};

fn element(semantic: &str, index: u32, register: u32, mask: u8, read: u8, ty: u32) -> SigElement {
    let system_value = match semantic {
        "SV_Position" => 1,
        "SV_IsFrontFace" => 9,
        "SV_Target" => 64,
        "SV_Depth" => 65,
        "SV_DepthLessEqual" => 68,
        _ => 0,
    };
    SigElement {
        semantic: semantic.to_owned(),
        index,
        system_value,
        component_type: ty,
        register,
        mask,
        rw_mask: read,
    }
}

fn record(attr: u8, register: u32, high: Option<u32>, flags: u8) -> PixelInput {
    PixelInput {
        attr,
        kind: 6,
        register,
        high_register: high,
        channels: 0xf,
        flags,
    }
}

fn pixel(code: Vec<u32>, inputs: Vec<SigElement>, outputs: Vec<SigElement>) -> StageProgram {
    StageProgram {
        inputs,
        outputs,
        name: "pixel test".to_owned(),
        ..StageProgram::new(Stage::Pixel, code)
    }
}

fn target(index: u32, ty: u32) -> SigElement {
    element("SV_Target", index, index, 15, 0, ty)
}

/// A 16 bit interpolation step in the two word encoding.
fn interp16(op: Op, dst: u8, weight: u8, attr: u8, chan: u8, high: bool, opsel: u8) -> Insn {
    let src1 = if op == Op::v_interp_p2_f16 {
        v(dst)
    } else {
        Operand::None
    };
    Insn {
        pc: 0,
        len: 0,
        op,
        form: Form::Plain,
        dst: v(dst),
        dst2: Operand::None,
        src: [v(weight), src1, Operand::None, Operand::None],
        ext: Ext::VintrpE64 {
            attr,
            chan,
            high,
            abs: 0,
            neg: 0,
            opsel,
            omod: 0,
            clamp: false,
        },
    }
}

fn translate_one(p: &StageProgram) -> Result<String, Refusal> {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .map(|t| t.stages[0].hlsl.clone())
}

fn compile(stages: &[StageProgram]) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    if let Err(r) = translate_to_dxil(
        stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    ) {
        panic!("DXC refused the translation: {r}");
    }
}

/// Packs registers `rgba` as half floats into v0, v1 and exports them to colour target `t`.
fn pack_export(a: &mut Asm, rgba: [u8; 4], t: u8, last: bool) {
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(rgba[0]), v(rgba[1])));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(rgba[2]), v(rgba[3])));
    a.push(export(t, 0xf, [0, 1, 0, 0], true, last, last));
}

#[test]
fn two_16_bit_registers_share_one_attribute() {
    // TEXCOORD1.xyz and TEXCOORD2.xyzw, both 16 bit, are attribute 1: halves 0 to 2 the first,
    // halves 4 to 7 the second. Attribute 0 is TEXCOORD0.xy at 32 bits.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, M0, s(2)));
    a.push(interp(Op::v_interp_p1_f32, v(2), v(0), 0, 1));
    a.push(interp(Op::v_interp_p2_f32, v(2), v(1), 0, 1));
    // Half 1 of attribute 1 (TEXCOORD1.y) into the high half of v3, half 6 (TEXCOORD2.z) into
    // the low half of v3.
    a.push(interp16(Op::v_interp_p1ll_f16, 6, 0, 1, 0, true, 0));
    a.push(interp16(Op::v_interp_p2_f16, 3, 1, 1, 0, true, 8));
    a.push(interp16(Op::v_interp_p1ll_f16, 6, 0, 1, 3, false, 0));
    a.push(interp16(Op::v_interp_p2_f16, 3, 1, 1, 3, false, 0));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(2), v(2)));
    a.push(export(0, 0xf, [0, 3, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("SV_Position", 0, 0, 15, 0, 3),
        element("TEXCOORD", 0, 1, 3, 2, 3),
        element("TEXCOORD", 1, 2, 7, 2, 6),
        element("TEXCOORD", 2, 3, 15, 4, 6),
    ];
    let mut p = pixel(a.finish(), inputs, vec![target(0, 3)]);
    p.pixel_inputs = vec![record(0, 1, None, 0x02), record(1, 2, Some(3), 0x1a)];
    let hlsl = translate_one(&p).unwrap();
    // Members: i0 SV_Position, i1 TEXCOORD0, i2 TEXCOORD1, i3 TEXCOORD2.
    assert!(hlsl.contains("decant_in.i1.y"), "{hlsl}");
    assert!(hlsl.contains("decant_in.i2.y"), "{hlsl}");
    assert!(hlsl.contains("decant_in.i3.z"), "{hlsl}");
    assert!(!hlsl.contains("decant_in.i2.z"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn a_flat_input_is_not_interpolated() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, M0, s(2)));
    let mut mov = interp(Op::v_interp_mov_f32, v(2), int(2), 1, 0);
    mov.src[0] = Operand::Int(2);
    a.push(mov);
    a.push(interp(Op::v_interp_p1_f32, v(3), v(0), 0, 0));
    a.push(interp(Op::v_interp_p2_f32, v(3), v(1), 0, 0));
    a.push(vop1(Op::v_cvt_f32_u32, v(2), v(2)));
    pack_export(&mut a, [2, 3, 2, 3], 0, true);
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("TEXCOORD", 0, 0, 1, 1, 3),
        element("COLOR", 0, 1, 1, 1, 1),
    ];
    let mut p = pixel(a.finish(), inputs, vec![target(0, 3)]);
    p.pixel_inputs = vec![record(0, 0, None, 0x02), record(1, 1, None, 0x01)];
    let hlsl = translate_one(&p).unwrap();
    assert!(hlsl.contains("nointerpolation uint i1 : COLOR0"), "{hlsl}");
    assert!(!hlsl.contains("nointerpolation float i0"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn a_flat_16_bit_attribute_has_one_component_per_channel() {
    // Record flags 0x09: 16 bit and flat, not interpolated with the 16 bit instructions. The
    // console reads channel 1 with a 32 bit move and finds component y as a half float in the
    // low half.
    let mut a = Asm::new();
    let mut mov = interp(Op::v_interp_mov_f32, v(2), int(2), 0, 1);
    mov.src[0] = Operand::Int(2);
    a.push(mov);
    a.push(export(0, 0x3, [2, 2, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![element("TEXCOORD", 3, 0, 15, 2, 6)];
    let mut p = pixel(
        a.finish(),
        inputs,
        vec![element("SV_Target", 0, 0, 3, 0, 3)],
    );
    p.pixel_inputs = vec![record(0, 0, None, 0x09)];
    let hlsl = translate_one(&p).unwrap();
    assert!(
        hlsl.contains("nointerpolation float4 i0 : TEXCOORD3"),
        "{hlsl}"
    );
    assert!(hlsl.contains("decant_in.i0.y"), "{hlsl}");
    assert!(!hlsl.contains("decant_in.i0.x"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn a_16_bit_register_follows_a_short_one_at_the_next_channel() {
    // TEXCOORD1.xy (16 bit, one channel) opens attribute 0, TEXCOORD2.xyzw (16 bit) follows at
    // channel 1: half 2 is TEXCOORD2.x, half 5 is TEXCOORD2.w. Read from a stage before that
    // places the same elements in the same registers.
    let mut a = Asm::new();
    a.push(interp16(Op::v_interp_p1ll_f16, 6, 0, 0, 1, false, 0));
    a.push(interp16(Op::v_interp_p2_f16, 3, 1, 0, 1, false, 0));
    a.push(interp16(Op::v_interp_p1ll_f16, 6, 0, 0, 2, true, 0));
    a.push(interp16(Op::v_interp_p2_f16, 3, 1, 0, 2, true, 8));
    a.push(export(0, 0xf, [3, 3, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("TEXCOORD", 1, 0, 3, 3, 6),
        element("TEXCOORD", 2, 1, 15, 15, 6),
    ];
    let mut p = pixel(a.finish(), inputs, vec![target(0, 3)]);
    p.pixel_inputs = vec![record(0, 0, Some(1), 0x1a)];
    let hlsl = translate_one(&p).unwrap();
    assert!(hlsl.contains("decant_in.i1.x"), "{hlsl}");
    assert!(hlsl.contains("decant_in.i1.w"), "{hlsl}");
    assert!(!hlsl.contains("decant_in.i1.z"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn the_rank_rule_is_the_fallback_without_a_table() {
    // No table: attribute k is the read register of rank k. TEXCOORD1 is not read.
    let mut a = Asm::new();
    a.push(interp(Op::v_interp_p1_f32, v(2), v(0), 1, 0));
    a.push(interp(Op::v_interp_p2_f32, v(2), v(1), 1, 0));
    pack_export(&mut a, [2, 2, 2, 2], 0, true);
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("TEXCOORD", 0, 0, 15, 1, 3),
        element("TEXCOORD", 1, 1, 15, 0, 3),
        element("TEXCOORD", 2, 2, 15, 1, 3),
    ];
    let p = pixel(a.finish(), inputs, vec![target(0, 3)]);
    let hlsl = translate_one(&p).unwrap();
    assert!(hlsl.contains("decant_in.i2.x"), "{hlsl}");
}

#[test]
fn position_and_front_face_come_from_the_entry_registers() {
    // Position x, y and w are read: v2, v3, v4. The front face follows in v5.
    let mut a = Asm::new();
    a.push(vopc(Op::v_cmp_lg_u32, int(0), v(5)));
    a.push(vop3(Op::v_cndmask_b32, v(6), int(0), float(1.0), VCC));
    a.push(vop2(Op::v_mul_f32, v(4), v(4), v(4)));
    pack_export(&mut a, [2, 3, 4, 6], 0, true);
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![
        element("SV_Position", 0, 0, 15, 0b1011, 3),
        element("TEXCOORD", 5, 1, 1, 0, 1),
        element("SV_IsFrontFace", 0, 1, 2, 2, 1),
    ];
    let p = pixel(a.finish(), inputs, vec![target(0, 3)]);
    let hlsl = translate_one(&p).unwrap();
    // A scalar unsigned front face, which DXC accepts.
    assert!(hlsl.contains("uint i2 : SV_IsFrontFace"), "{hlsl}");
    assert!(
        hlsl.contains("decant_in.i0.x") && hlsl.contains("decant_in.i0.w"),
        "{hlsl}"
    );
    assert!(hlsl.contains("decant_pixel_rcp"), "{hlsl}");
    // EXEC starts without the helper pixels.
    assert!(!hlsl.contains("exec = true;"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn a_linked_pixel_program_adds_the_front_face_to_the_vertex_outputs() {
    let mut a = Asm::new();
    a.push(export(12, 0xf, [0, 1, 2, 3], false, true, false));
    a.push(export(32, 0xf, [0, 1, 2, 3], false, false, false));
    a.push(sopp(Op::s_endpgm, 0));
    let vs = StageProgram {
        outputs: vec![
            element("TEXCOORD", 0, 0, 15, 0, 3),
            element("SV_Position", 0, 1, 15, 0, 3),
        ],
        name: "vertex test".to_owned(),
        ..StageProgram::new(Stage::Vertex, a.finish())
    };
    let mut a = Asm::new();
    a.push(interp(Op::v_interp_p1_f32, v(3), v(0), 0, 0));
    a.push(interp(Op::v_interp_p2_f32, v(3), v(1), 0, 0));
    a.push(vopc(Op::v_cmp_lg_u32, int(0), v(2)));
    a.push(vop3(Op::v_cndmask_b32, v(3), int(0), v(3), VCC));
    pack_export(&mut a, [3, 3, 3, 3], 0, true);
    a.push(sopp(Op::s_endpgm, 0));
    let mut ps = pixel(
        a.finish(),
        vec![
            element("TEXCOORD", 0, 0, 15, 1, 3),
            element("SV_IsFrontFace", 0, 1, 1, 1, 1),
        ],
        vec![target(0, 3)],
    );
    ps.pixel_inputs = vec![record(0, 0, None, 0x02)];
    let t = translate(
        &[vs.clone(), ps.clone()],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap();
    let hlsl = &t.stages[1].hlsl;
    assert!(hlsl.contains("uint i2 : SV_IsFrontFace"), "{hlsl}");
    compile(&[vs, ps]);
}

#[test]
fn colour_targets_are_compacted() {
    // The program writes SV_Target0 and SV_Target2: export targets 0 and 1.
    let mut a = Asm::new();
    a.push(vop1(Op::v_mov_b32, v(2), float(1.0)));
    pack_export(&mut a, [2, 2, 2, 2], 0, false);
    pack_export(&mut a, [2, 2, 2, 2], 1, true);
    a.push(sopp(Op::s_endpgm, 0));
    let p = pixel(a.finish(), Vec::new(), vec![target(0, 3), target(2, 3)]);
    let hlsl = translate_one(&p).unwrap();
    assert!(hlsl.contains("decant_out.o1.w"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn an_integer_target_takes_16_bit_integers() {
    let mut a = Asm::new();
    a.push(vop1(Op::v_mov_b32, v(2), lit(0x0002_0001)));
    a.push(export(0, 0x3, [2, 2, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let p = pixel(
        a.finish(),
        Vec::new(),
        vec![element("SV_Target", 0, 0, 3, 0, 1)],
    );
    let hlsl = translate_one(&p).unwrap();
    assert!(!hlsl.contains("decant_f16_to_f32"), "{hlsl}");
    assert!(hlsl.contains("65535u"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn depth_and_the_kill_exit_are_exported() {
    // The kill exit: the surviving lanes in EXEC, then the null target with the valid mask.
    let mut a = Asm::new();
    a.push(vopc(Op::v_cmpx_lt_f32, int(0), v(2)));
    a.push(vop1(Op::v_mov_b32, v(3), float(0.5)));
    a.push(export(8, 0x1, [3, 0, 0, 0], false, false, false));
    a.push(export(9, 0x0, [0, 0, 0, 0], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let inputs = vec![element("SV_Position", 0, 0, 15, 1, 3)];
    let p = pixel(
        a.finish(),
        inputs,
        vec![element("SV_DepthLessEqual", 0, u32::MAX, 1, 0xe, 3)],
    );
    let hlsl = translate_one(&p).unwrap();
    assert!(hlsl.contains("SV_DepthLessEqual"), "{hlsl}");
    // DXC wants a centroid position with a conservative depth output.
    assert!(
        hlsl.contains("noperspective centroid float4 i0 : SV_Position"),
        "{hlsl}"
    );
    assert!(hlsl.contains("discard"), "{hlsl}");
    compile(&[p]);
}

#[test]
fn a_read_of_a_vertex_difference_is_refused() {
    let mut a = Asm::new();
    let mut mov = interp(Op::v_interp_mov_f32, v(2), int(0), 0, 0);
    mov.src[0] = Operand::Int(0);
    a.push(mov);
    a.push(sopp(Op::s_endpgm, 0));
    let p = pixel(
        a.finish(),
        vec![element("TEXCOORD", 0, 0, 1, 1, 3)],
        Vec::new(),
    );
    let r = translate_one(&p).unwrap_err();
    assert_eq!(r.kind, RefusalKind::Form(Op::v_interp_mov_f32, "vertex"));
}
