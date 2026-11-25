//! Synthetic console shaders for the tests of the direct shader path (`direct_source` and the
//! device case of `d3d12xs_test`): programs written with the assembler of `decant-shader`, and
//! the console container around a program, so a test hands the layer the same kind of bytes the
//! title does. No game data is involved.
//!
//! The container is what `decant_shader::container::program` reads back: a DXBC container with
//! the `ISG1`, `OSG1` and `PSV0` parts and a `DXIL` part whose `XDAT` table holds one `XBCH`
//! part, a zlib stream (one stored block) of the `FSSM` block with the code and its three tables:
//! the root layout, the binding table and the pixel inputs.

use decant_shader::api::{DeclaredResource, PixelInput, RegisterClass, SigElement, Stage, StageProgram, TableBinding, UserDataKind, UserDataSlot};
use decant_shader::asm::*;
use decant_shader::tables::Op;

fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// An `ISG1` or `OSG1` part: count, offset of the first element, 32 byte elements, then the names.
fn signature_part(elements: &[SigElement]) -> Vec<u8> {
    let names_at = 8 + 32 * elements.len();
    let mut names: Vec<u8> = Vec::new();
    let mut out = Vec::new();
    put32(&mut out, elements.len() as u32);
    put32(&mut out, 8);
    for e in elements {
        put32(&mut out, 0);
        put32(&mut out, (names_at + names.len()) as u32);
        for v in [e.index, e.system_value, e.component_type, e.register] {
            put32(&mut out, v);
        }
        out.extend_from_slice(&[e.mask, e.rw_mask, 0, 0, 0, 0, 0, 0]);
        names.extend_from_slice(e.semantic.as_bytes());
        names.push(0);
    }
    out.extend_from_slice(&names);
    out
}

/// A `PSV0` part: eight bytes of runtime information, then the resource records in the long form.
fn psv0_part(resources: &[DeclaredResource]) -> Vec<u8> {
    let mut out = Vec::new();
    put32(&mut out, 8);
    out.extend_from_slice(&[0; 8]);
    put32(&mut out, resources.len() as u32);
    if !resources.is_empty() {
        put32(&mut out, 24);
    }
    for r in resources {
        for v in [r.psv_type, r.space, r.lower, r.upper, r.psv_kind, 0] {
            put32(&mut out, v);
        }
    }
    out
}

/// The `FSSM` block: a 0x240 byte header with the place of the code (0xC0, 0xC8), of the root
/// layout (0xE8, 0xF8) and of the binding table (0xF0, 0xFC) and with the 32 pixel input records
/// (0x138), then a 16 byte sub header, the code with one padding word, and the two tables.
fn fssm_block(program: &StageProgram) -> Vec<u8> {
    const HEADER: usize = 0x240;
    const CODE_END: u32 = 0xBF9F_0000;
    let mut out = b"FSSM".to_vec();
    out.resize(HEADER, 0);
    let code_bytes = (program.code.len() + 1) * 4;
    out[0xC0..0xC8].copy_from_slice(&(HEADER as u64).to_le_bytes());
    out[0xC8..0xD0].copy_from_slice(&(code_bytes as u64).to_le_bytes());
    for p in &program.pixel_inputs {
        // Kind, signature register, channels, attribute number, 0, second register, 0, flags.
        let at = 0x138 + 8 * usize::from(p.attr);
        let second = p.high_register.map_or(0, |r| r as u8);
        out[at..at + 8].copy_from_slice(&[p.kind, p.register as u8, p.channels, p.attr, 0, second, 0, p.flags]);
    }
    out.extend_from_slice(&[0; 16]);
    for w in program.code.iter().chain(&[CODE_END]) {
        put32(&mut out, *w);
    }
    // The first record of a table is 16 bytes after the position the header names.
    // The root layout: kind, 0, count, register, the user data register in the low byte.
    let layout_at = out.len() as u64;
    out[0xE8..0xF0].copy_from_slice(&layout_at.to_le_bytes());
    out[0xF8..0xFC].copy_from_slice(&(program.user_data.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for u in &program.user_data {
        let (kind, count, register) = match u.kind {
            UserDataKind::Table => (3, 0, 0),
            UserDataKind::Constants { register, count } => (0, count, register),
            UserDataKind::RootSrv { register } => (1, 0, register),
            UserDataKind::ResourceHeap => (5, 0, 0),
            UserDataKind::SamplerHeap => (6, 0, 0),
        };
        for v in [kind, 0, count, register, u32::from(u.sgpr)] {
            put32(&mut out, v);
        }
    }
    // The binding table: kind, offset in dwords, register, 0, the table register in bits 8 to 15.
    let table_at = out.len() as u64;
    out[0xF0..0xF8].copy_from_slice(&table_at.to_le_bytes());
    out[0xFC..0x100].copy_from_slice(&(program.binding_table.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0; 16]);
    for b in &program.binding_table {
        let kind = match b.class {
            RegisterClass::Srv => 0x300,
            RegisterClass::Cbv => 0x302,
            RegisterClass::Uav => 0x303,
            RegisterClass::Sampler => 0x401,
        };
        for v in [kind, b.word_offset, b.register, 0, u32::from(b.sgpr) << 8] {
            put32(&mut out, v);
        }
    }
    out
}

/// A zlib stream of one stored block.
fn stored_zlib(data: &[u8]) -> Vec<u8> {
    let len = u16::try_from(data.len()).expect("a synthetic program fits one stored block");
    let mut out = vec![0x78, 0x01, 0x01];
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&(!len).to_le_bytes());
    out.extend_from_slice(data);
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

/// The `DXIL` part: the program version word (stage and shader model 6.6), then the `XDAT` table
/// with one `XBCH` part.
fn dxil_part(stage: Stage, fssm: &[u8]) -> Vec<u8> {
    let stage_code: u32 = match stage {
        Stage::Pixel => 0,
        Stage::Vertex => 1,
        Stage::Geometry => 2,
        Stage::Hull => 3,
        Stage::Domain => 4,
        Stage::Compute => 5,
        Stage::Mesh => 13,
        Stage::Amplification => 14,
    };
    let stream = stored_zlib(fssm);
    let mut out = Vec::new();
    put32(&mut out, stage_code << 16 | 6 << 4 | 6);
    put32(&mut out, 0);
    out.extend_from_slice(b"XDAT");
    put32(&mut out, 1);
    put32(&mut out, 0);
    // One entry: kind 0x0D, the place of the XBCH part in this DXIL part, its size.
    let xbch_at = out.len() + 12;
    for v in [0x0D, xbch_at as u32, (24 + stream.len()) as u32] {
        put32(&mut out, v);
    }
    out.extend_from_slice(b"XBCH\0\0\0\0");
    out.extend_from_slice(&(stream.len() as u64).to_le_bytes());
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(&stream);
    out
}

/// The console container of `program`.
pub fn container(program: &StageProgram) -> Vec<u8> {
    let parts: [(&[u8; 4], Vec<u8>); 4] = [
        (b"ISG1", signature_part(&program.inputs)),
        (b"OSG1", signature_part(&program.outputs)),
        (b"PSV0", psv0_part(&program.resources)),
        (b"DXIL", dxil_part(program.stage, &fssm_block(program))),
    ];
    let mut body: Vec<u8> = Vec::new();
    let mut offsets = Vec::new();
    let table_end = 32 + 4 * parts.len();
    for (tag, data) in &parts {
        offsets.push((table_end + body.len()) as u32);
        body.extend_from_slice(*tag);
        put32(&mut body, data.len() as u32);
        body.extend_from_slice(data);
        body.resize(body.len().next_multiple_of(4), 0);
    }
    let mut out = b"DXBC".to_vec();
    // The checksum field: not the DXBC checksum (nothing checks it on a console container), but
    // different for different programs, as the layer tells shader containers apart by it.
    let mut id = 0xcbf2_9ce4_8422_2325u64;
    for b in &body {
        id = (id ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    out.extend_from_slice(&id.to_le_bytes());
    out.extend_from_slice(&(!id).to_le_bytes());
    put32(&mut out, 1);
    put32(&mut out, (table_end + body.len()) as u32);
    put32(&mut out, parts.len() as u32);
    for o in offsets {
        put32(&mut out, o);
    }
    out.extend_from_slice(&body);
    out
}

fn element(semantic: &str, index: u32, register: u32, mask: u8, rw_mask: u8) -> SigElement {
    let system_value = match semantic {
        "SV_Position" => 1,
        "SV_Target" => 64,
        _ => 0,
    };
    SigElement { semantic: semantic.to_owned(), index, system_value, component_type: 3, register, mask, rw_mask }
}

/// The user interface vertex program of the translator's slice test (`decant-shader`,
/// `tests/slice.rs`): the position from two rows of a constant buffer chosen by the third component
/// of `COLOR1`, the colours passed through, two more rows as texture coordinates. The buffer is
/// descriptor 0 of the table in `s8`, declared as `b0` of space 1.
///
/// With `k = COLOR1.z * 255` and rows of four floats: `x = dot(POSITION.xy, row[k + 2].xy) +
/// row[k + 2].w`, `y = dot(POSITION.xy, row[k + 3].xy) + row[k + 3].w`, `z = 0`, `w = 1`,
/// `TEXCOORD0 = row[k]`, `TEXCOORD1 = row[k + 1]`.
pub fn ui_vertex() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_swappc_b64, s(4), s(4)));
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(sop1(Op::s_mov_b32, s(4), lit(255.0f32.to_bits())));
    a.push(smem(Op::s_load_dwordx4, s(0), s(8), 0));
    for (row, first) in [(2.5f32, 0u8), (3.5, 14), (0.5, 18), (1.5, 22)] {
        a.push(vop3(Op::v_fma_f32, v(26), v(10), s(4), lit(row.to_bits())));
        a.push(vop1(Op::v_cvt_i32_f32, v(26), v(26)));
        a.push(buffer_load(Op::buffer_load_dwordx4, v(first), v(26), s(0), 0, true));
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
    StageProgram {
        inputs: vec![element("COLOR", 0, 0, 15, 15), element("COLOR", 1, 1, 15, 15), element("POSITION", 0, 2, 3, 3)],
        outputs: vec![
            element("COLOR", 0, 0, 15, 0),
            element("COLOR", 1, 1, 15, 0),
            element("TEXCOORD", 0, 2, 15, 0),
            element("TEXCOORD", 1, 3, 15, 0),
            element("SV_Position", 0, 4, 15, 0),
        ],
        resources: vec![DeclaredResource { class: RegisterClass::Cbv, lower: 0, upper: 0, space: 1, psv_type: 2, psv_kind: 13 }],
        binding_table: vec![TableBinding { class: RegisterClass::Cbv, register: 0, sgpr: 8, word_offset: 0 }],
        user_data: vec![UserDataSlot { sgpr: 8, kind: UserDataKind::Table }],
        name: "ui_vertex".to_owned(),
        ..StageProgram::new(Stage::Vertex, a.finish())
    }
}

/// The user interface pixel program of the same test: `rgb = saturate(TEXCOORD0 + COLOR0 *
/// TEXCOORD1)`, and the alpha of that times the alpha of `COLOR1`. Four inputs, read through four
/// different attribute numbers.
pub fn ui_pixel() -> StageProgram {
    let mut a = Asm::new();
    let interpolate = |a: &mut Asm, attr: u8, regs: [u8; 4]| {
        for (chan, r) in regs.into_iter().enumerate() {
            a.push(interp(Op::v_interp_p1_f32, v(r), v(0), attr, chan as u8));
        }
        for (chan, r) in regs.into_iter().enumerate() {
            a.push(interp(Op::v_interp_p2_f32, v(r), v(1), attr, chan as u8));
        }
    };
    a.push(sop1(Op::s_mov_b32, M0, s(12)));
    interpolate(&mut a, 0, [10, 11, 12, 13]);
    interpolate(&mut a, 2, [14, 15, 16, 17]);
    interpolate(&mut a, 3, [18, 19, 20, 21]);
    a.push(interp(Op::v_interp_p1_f32, v(22), v(0), 1, 3));
    a.push(interp(Op::v_interp_p2_f32, v(22), v(1), 1, 3));
    for k in 0..4 {
        a.push(vop2(Op::v_fmac_f32, v(14 + k), v(10 + k), v(18 + k)));
        a.push(vop3(Op::v_med3_f32, v(2 + k), v(14 + k), int(0), float(1.0)));
    }
    a.push(vop2(Op::v_mul_legacy_f32, v(5), v(5), v(22)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(2), v(3)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(4), v(5)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    // Attribute `n` carries the input in signature register `n`, interpolated with perspective.
    let pixel_inputs = (0..4).map(|n| PixelInput { attr: n, kind: 6, register: u32::from(n), high_register: None, channels: 0xF, flags: 0x02 }).collect();
    StageProgram {
        inputs: vec![element("COLOR", 0, 0, 15, 15), element("COLOR", 1, 1, 15, 8), element("TEXCOORD", 0, 2, 15, 15), element("TEXCOORD", 1, 3, 15, 15)],
        outputs: vec![element("SV_Target", 0, 0, 15, 0)],
        pixel_inputs,
        name: "ui_pixel".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

/// A mesh triangle with the same link signature as the UI vertex shader and one unused output.
pub fn mesh_triangle() -> StageProgram {
    let mut code = Asm::new();
    code.push(sopp(Op::s_inst_prefetch, 3));
    code.push(sop1(Op::s_mov_b32, M0, lit(1 << 12 | 3)));
    code.push(sopp(Op::s_sendmsg, 9));
    code.push(vop3(Op::v_mbcnt_lo_u32_b32, v(0), int(-1), int(0), decant_shader::decode::Operand::None));
    // The first three lane positions are (-1, -1), (-1, 3) and (3, -1).
    code.push(vopc(Op::v_cmp_eq_u32, int(2), v(0)));
    code.push(vop3(Op::v_cndmask_b32, v(1), float(-1.0), lit(3.0f32.to_bits()), VCC));
    code.push(vopc(Op::v_cmp_eq_u32, int(1), v(0)));
    code.push(vop3(Op::v_cndmask_b32, v(2), float(-1.0), lit(3.0f32.to_bits()), VCC));
    code.push(vop1(Op::v_mov_b32, v(3), int(0)));
    code.push(vop1(Op::v_mov_b32, v(4), float(1.0)));
    code.push(export(12, 15, [1, 2, 3, 4], false, true, false));
    for slot in 0..5 {
        code.push(export(32 + slot, 15, [4; 4], false, false, false));
    }
    code.push(vop1(Op::v_mov_b32, v(5), lit(1 << 10 | 2 << 20)));
    code.push(export(20, 1, [5; 4], false, true, false));
    code.push(sopp(Op::s_endpgm, 0));

    let mut mesh = StageProgram::new(Stage::Mesh, code.finish());
    mesh.outputs = ui_vertex().outputs;
    mesh.outputs.push(SigElement { semantic: "UNUSED".into(), register: 5, system_value: 0, ..mesh.outputs[0].clone() });
    mesh
}
