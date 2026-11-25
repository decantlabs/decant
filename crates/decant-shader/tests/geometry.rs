//! The geometry stage: the one shape the geometry programs of the title have, built with the
//! assembler, translated to an HLSL geometry shader and compiled with DXC.
//!
//! The shape (see `stage/geometry.rs`): the program for one primitive reads its three input
//! vertices from the local data share at the places `v0` and `v1` name, writes each output
//! vertex into a slot behind them and sends a message for each emit. The programs are written
//! here by hand. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::Operand;
use decant_shader::stage::geometry;
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

const NONE: Operand = Operand::None;

fn element(semantic: &str, index: u32, ty: u32, register: u32, mask: u8) -> SigElement {
    SigElement {
        semantic: semantic.to_owned(),
        index,
        system_value: u32::from(semantic == "SV_Position"),
        component_type: ty,
        register,
        mask,
        rw_mask: mask,
    }
}

/// Position and one texture coordinate in, position, an integer and a vector out: 2 input
/// registers, 3 output registers.
fn signature() -> (Vec<SigElement>, Vec<SigElement>) {
    (
        vec![
            element("SV_Position", 0, 3, 0, 15),
            element("TEXCOORD", 0, 3, 1, 3),
        ],
        vec![
            element("SV_Position", 0, 3, 0, 15),
            element("TEXCOORD", 0, 1, 1, 1),
            element("TEXCOORD", 1, 3, 2, 3),
        ],
    )
}

/// The skeleton of a geometry program of the title. `messages` are sent after the three
/// vertices are written, with the first three between the vertices as the title does.
fn skeleton(messages: &[u16], system_offset: u32) -> Vec<u32> {
    let mut a = Asm::new();
    a.push(sopp(Op::s_inst_prefetch, 3));
    // Every lane has a primitive. A real program makes this mask from the lane count in s3.
    a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
    // The base of the output slots: a dword of the system table times 256, plus s2.
    a.push(sop2(Op::s_and_b32, s(1), s(9), lit(0xff)));
    a.push(sop1(Op::s_mov_b32, s(0), s(8)));
    a.push(smem(Op::s_load_dword, s(0), s(0), system_offset));
    a.push(sop2(Op::s_lshl_b32, s(0), s(0), int(8)));
    a.push(sop2(Op::s_add_i32, s(0), s(0), s(2)));
    a.push(vop1(Op::v_mov_b32, v(12), s(0)));
    // The byte addresses of the three input vertices: the two halves of v0 and v1, times 4.
    a.push(vop2(Op::v_lshlrev_b32, v(3), int(2), v(0)));
    a.push(vop2(Op::v_and_b32, v(10), lit(0x3fffc), v(3)));
    a.push(vop2(Op::v_lshrrev_b32, v(3), int(14), v(0)));
    a.push(vop2(Op::v_and_b32, v(8), lit(0x3fffc), v(3)));
    a.push(vop2(Op::v_lshlrev_b32, v(3), int(2), v(1)));
    a.push(vop2(Op::v_and_b32, v(6), lit(0x3fffc), v(3)));
    // Output vertex k is input vertex k: the position (dwords 0 to 3), a number in register 1
    // and the texture coordinate (dwords 4 and 5 of the input) in register 2. A slot has 12
    // dwords.
    for (k, addr) in [10u8, 8, 6].into_iter().enumerate() {
        let slot = 12 * k as u8;
        a.push(ds(Op::ds_read2_b32, v(20), v(addr), NONE, NONE, 0, 1));
        a.push(ds(Op::ds_read2_b32, v(22), v(addr), NONE, NONE, 2, 3));
        a.push(ds(Op::ds_read2_b32, v(24), v(addr), NONE, NONE, 4, 5));
        a.push(vop1(Op::v_mov_b32, v(26), int(k as i8 + 1)));
        a.push(ds(
            Op::ds_write2_b32,
            NONE,
            v(12),
            v(20),
            v(21),
            slot,
            slot + 1,
        ));
        a.push(ds(
            Op::ds_write2_b32,
            NONE,
            v(12),
            v(22),
            v(23),
            slot + 2,
            slot + 3,
        ));
        a.push(ds(
            Op::ds_write2_b32,
            NONE,
            v(12),
            v(26),
            v(26),
            slot + 4,
            slot + 4,
        ));
        a.push(ds(
            Op::ds_write2_b32,
            NONE,
            v(12),
            v(24),
            v(25),
            slot + 8,
            slot + 9,
        ));
        if let Some(m) = messages.get(k) {
            a.push(sopp(Op::s_sendmsg, *m));
        }
    }
    for m in messages.iter().skip(3) {
        a.push(sopp(Op::s_sendmsg, *m));
    }
    a.push(sopp(Op::s_endpgm, 0));
    a.finish()
}

/// Three emits, the end of the strip, the end of the program.
const TITLE: [u16; 5] = [0x22, 0x22, 0x22, 0x12, 0x3];

fn program(code: Vec<u32>, name: &str) -> StageProgram {
    let (inputs, outputs) = signature();
    StageProgram {
        inputs,
        outputs,
        geometry: Some(GeometryInfo {
            input_primitive: 3,
            output_topology: 5,
            stream_mask: 1,
            max_vertices: 3,
        }),
        name: name.to_owned(),
        ..StageProgram::new(Stage::Geometry, code)
    }
}

fn hlsl(p: &StageProgram) -> String {
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{}: {r}", p.name));
    assert_eq!(t.stages[0].profile, "gs_6_6");
    assert_eq!(t.stages[0].entry, "main");
    t.stages[0].hlsl.clone()
}

fn refusal(p: &StageProgram) -> String {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .expect_err("the program is refused")
    .key()
}

fn compile(stages: &[StageProgram]) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("dxc.exe not found: the compile step is skipped");
        return;
    };
    let compiled = translate_to_dxil(
        stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(compiled.len(), stages.len());
    assert!(compiled.iter().all(|c| c.dxil.starts_with(b"DXBC")));
}

#[test]
fn the_skeleton_becomes_a_geometry_shader() {
    let p = program(skeleton(&TITLE, 0xb8), "geometry_plain");
    assert_eq!(geometry::check(&p), Ok(()));
    let text = hlsl(&p);
    // One function for one primitive, and the entry point around it.
    assert!(
        text.contains("\nDecantOut decant_gs_primitive(DecantIn decant_in) {"),
        "{text}"
    );
    assert!(
        text.contains(
            "\n[maxvertexcount(3)]\nvoid main(triangle DecantIn decant_in[3], inout TriangleStream<DecantOut> decant_stream) {"
        ),
        "{text}"
    );
    // The local data share is an array of the invocation, not group shared memory.
    assert!(text.contains("static uint decant_lds[512];"), "{text}");
    assert!(!text.contains("groupshared"), "{text}");
    // The input vertices lie 8 dwords apart: the second texture coordinate component of
    // vertex 2 is dword 2 * 8 + 4 + 1.
    assert!(
        text.contains("decant_lds[21] = asuint(decant_in[2].i1.y);"),
        "{text}"
    );
    // The output slots follow at dword 24, 12 dwords each. Register 1 is an integer.
    assert!(text.contains("uint decant_b = 24u + decant_k * 12u;"), "{text}");
    assert!(text.contains("decant_v.o1 = decant_lds[decant_b + 4u];"), "{text}");
    assert!(
        text.contains("decant_v.o2.y = asfloat(decant_lds[decant_b + 9u]);"),
        "{text}"
    );
    // Three emits and the end of the strip.
    assert_eq!(text.matches("decant_gs_emit();").count(), 3, "{text}");
    assert_eq!(text.matches("decant_gs_cut();").count(), 1, "{text}");
    assert!(text.contains("decant_stream.Append(decant_v);"), "{text}");
    assert!(text.contains("decant_stream.RestartStrip();"), "{text}");
    // The load through the system pointer is gone: no raw address, no leftover of the pointer.
    assert!(!text.contains("1397184595"), "{text}");
}

#[test]
fn the_geometry_shader_compiles() {
    compile(&[program(skeleton(&TITLE, 0xb8), "geometry_plain")]);
}

#[test]
fn a_pixel_program_links_to_the_geometry_program() {
    // The pixel program reads TEXCOORD1, which the geometry program writes in register 2.
    let mut a = Asm::new();
    for chan in 0..2u8 {
        a.push(interp(Op::v_interp_p1_f32, v(2 + chan), v(0), 0, chan));
    }
    for chan in 0..2u8 {
        a.push(interp(Op::v_interp_p2_f32, v(2 + chan), v(1), 0, chan));
    }
    a.push(export(0, 0xf, [2, 3, 2, 3], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let pixel = StageProgram {
        inputs: vec![
            SigElement {
                rw_mask: 0,
                ..element("SV_Position", 0, 3, 0, 15)
            },
            element("TEXCOORD", 1, 3, 2, 3),
        ],
        outputs: vec![SigElement {
            system_value: 64,
            ..element("SV_Target", 0, 3, 0, 15)
        }],
        name: "geometry_pixel".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    compile(&[program(skeleton(&TITLE, 0xb8), "geometry_plain"), pixel]);
}

#[test]
fn what_the_stage_does_not_carry_is_refused_by_name() {
    // A mesh message in a geometry program.
    let p = program(skeleton(&[0x22, 0x22, 0x22, 0x9], 0xb8), "geometry_message");
    assert_eq!(refusal(&p), "stage:geometry-message");
    // More emits than the container allows vertices.
    let p = program(
        skeleton(&[0x22, 0x22, 0x22, 0x22, 0x3], 0xb8),
        "geometry_emits",
    );
    assert_eq!(refusal(&p), "stage:geometry-ring-size");
    // Another field of the system table.
    let p = program(skeleton(&TITLE, 0x40), "geometry_system");
    assert_eq!(refusal(&p), "stage:geometry-system-load");
    // Lines in.
    let mut p = program(skeleton(&TITLE, 0xb8), "geometry_lines");
    p.geometry = Some(GeometryInfo {
        input_primitive: 2,
        output_topology: 5,
        stream_mask: 1,
        max_vertices: 3,
    });
    assert_eq!(refusal(&p), "stage:geometry-input-primitive");
    // Hull and domain programs are not written: the title has none.
    let hull = StageProgram::new(Stage::Hull, skeleton(&TITLE, 0xb8));
    assert_eq!(refusal(&hull), "stage:tessellation-not-written");
}
