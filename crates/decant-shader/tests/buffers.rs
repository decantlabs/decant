//! Vector buffer accesses: how each kind of view is addressed and written, translated to HLSL
//! and compiled with DXC.
//!
//! Every program here is a pixel program that loads the descriptor at slot 1 of the table in s8
//! (`s_load_dwordx4 s[0:3]`), does its buffer work and exports v8 to v11. The container declares
//! the descriptor as register 3 with the `PSV0` type of the test. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Insn};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

#[derive(Clone, Copy)]
enum Layer {
    WithShadow,
    NoShadow,
}

use Layer::{NoShadow, WithShadow};

fn options(layer: Layer) -> Options {
    let mut o = Options::default();
    if let Some(m) = &mut o.mirror {
        m.view_shadow = matches!(layer, WithShadow).then_some(1);
    }
    o
}

/// A buffer instruction on the descriptor in s[0:3], with the address registers from v4.
fn buf(op: Op, data: u8, offset: u16, offen: bool, idxen: bool) -> Insn {
    let mut i = buffer_load(op, v(data), v(4), s(0), offset, idxen);
    if let Ext::Buf { offen: o, .. } = &mut i.ext {
        *o = offen;
    }
    i
}

fn with_format(mut i: Insn, format: u8) -> Insn {
    if let Ext::Buf { format: f, .. } = &mut i.ext {
        *f = format;
    }
    i
}

fn with_glc(mut i: Insn) -> Insn {
    if let Ext::Buf { glc, .. } = &mut i.ext {
        *glc = true;
    }
    i
}

/// The program around `body`. `psv` is the declared `PSV0` type of the descriptor, `None` for a
/// descriptor the container does not declare.
fn pixel(body: &[Insn], class: RegisterClass, psv: Option<u32>) -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(0), s(8), 0x20));
    for i in body {
        a.push(*i);
    }
    a.push(export(0, 0xf, [8, 9, 10, 11], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let sig = |semantic: &str, sv| SigElement {
        semantic: semantic.to_owned(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register: 0,
        mask: 15,
        rw_mask: 0,
    };
    let mut p = StageProgram {
        inputs: vec![sig("SV_Position", 1)],
        outputs: vec![sig("SV_Target", 64)],
        name: "buffers".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    if let Some(psv_type) = psv {
        p.resources.push(DeclaredResource {
            class,
            lower: 3,
            upper: 3,
            space: 0,
            psv_type,
            psv_kind: 0,
        });
        p.binding_table.push(TableBinding {
            class,
            register: 3,
            sgpr: 8,
            word_offset: 8,
        });
    }
    p
}

fn hlsl(p: &StageProgram, layer: Layer) -> String {
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &options(layer),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{r}"));
    t.stages[0].hlsl.clone()
}

fn refusal(p: &StageProgram, layer: Layer) -> String {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &options(layer),
        &CountingHost::default(),
    )
    .map(|t| t.stages[0].hlsl.clone())
    .expect_err("refused")
    .key()
}

fn compile(p: &StageProgram, layer: Layer) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    let compiled = translate_to_dxil(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &options(layer),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(&compiled[0].dxil[..4], b"DXBC");
}

/// The slot of the descriptor: the heap index the resource local and the shadow row use.
const SLOT: &str = "(decant_mirror[2].x + 1u)";

#[test]
fn an_indexed_structured_read_takes_stride_and_count_from_the_shadow_row() {
    let p = pixel(
        &[buf(Op::buffer_load_dwordx4, 8, 0x20, false, true)],
        RegisterClass::Srv,
        Some(5),
    );
    let text = hlsl(&p, WithShadow);
    assert!(
        text.contains("ByteAddressBuffer heap_t_s8_1 = ResourceDescriptorHeap["),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "uint2 decant_row1 = decant_view_shadow_load2({SLOT} * 16u);"
        )),
        "{text}"
    );
    // The index register is v4, the record stride is the row's first dword.
    assert!(text.contains("uint decant_i1 = v4;"), "{text}");
    // The view of the slot starts on 16 bytes at or below the title's first byte: the offset
    // of the row (its fourth dword) is added to every byte address.
    assert!(
        text.contains(
            "uint decant_view_offset(uint slot) { return decant_view_shadow_load(slot * 16u + 12u) & 15u; }"
        ),
        "{text}"
    );
    assert!(
        text.contains(&format!(
            "uint decant_a1 = decant_i1 * decant_row1.x + (32u + decant_view_offset({SLOT}));"
        )),
        "{text}"
    );
    // Every dword is checked against the record count.
    for k in 0..4 {
        assert!(
            text.contains(&format!(
                "((decant_row1.x != 0u && decant_i1 < decant_row1.y) ? heap_t_s8_1.Load((decant_a1 + {}u)) : 0u)",
                4 * k
            )),
            "{text}"
        );
        assert!(
            text.contains(&format!("v{} = decant_t1_{k};", 8 + k)),
            "{text}"
        );
    }
    compile(&p, WithShadow);
}

#[test]
fn an_offset_only_structured_read_is_a_raw_read_without_a_row() {
    let p = pixel(
        &[buf(Op::buffer_load_dwordx2, 8, 0x10, true, false)],
        RegisterClass::Srv,
        Some(5),
    );
    let text = hlsl(&p, WithShadow);
    assert!(!text.contains("decant_row"), "{text}");
    assert!(
        text.contains(&format!(
            "uint decant_a1 = ((16u + v4) + decant_view_offset({SLOT}));"
        )),
        "{text}"
    );
    assert!(text.contains("heap_t_s8_1.Load((decant_a1 + 4u))"), "{text}");
    compile(&p, WithShadow);
}

#[test]
fn with_both_address_registers_the_index_comes_first_and_the_offset_second() {
    let p = pixel(
        &[buf(Op::buffer_load_dword, 8, 4, true, true)],
        RegisterClass::Srv,
        Some(5),
    );
    let text = hlsl(&p, WithShadow);
    assert!(text.contains("uint decant_i1 = v4;"), "{text}");
    assert!(
        text.contains(&format!(
            "uint decant_a1 = decant_i1 * decant_row1.x + ((4u + v5) + decant_view_offset({SLOT}));"
        )),
        "{text}"
    );
    compile(&p, WithShadow);
}

#[test]
fn a_constant_buffer_index_counts_16_byte_rows_with_and_without_an_offset_register() {
    let idx = pixel(
        &[buf(Op::buffer_load_dwordx4, 8, 0, false, true)],
        RegisterClass::Cbv,
        Some(2),
    );
    let text = hlsl(&idx, WithShadow);
    assert!(text.contains("uint decant_a1 = (v4 * 16u + 0u);"), "{text}");
    assert!(!text.contains("decant_row"), "{text}");
    compile(&idx, WithShadow);
    let both = pixel(
        &[buf(Op::buffer_load_dwordx4, 8, 0, true, true)],
        RegisterClass::Cbv,
        Some(2),
    );
    let text = hlsl(&both, WithShadow);
    assert!(text.contains("uint decant_a1 = (v4 * 16u + v5);"), "{text}");
    compile(&both, WithShadow);
}

#[test]
fn without_a_shadow_structured_reads_are_refused_and_counted_apart() {
    let indexed = pixel(
        &[buf(Op::buffer_load_dword, 8, 0, false, true)],
        RegisterClass::Srv,
        Some(5),
    );
    assert_eq!(refusal(&indexed, NoShadow), "resource:buffer-structured");
    let offset = pixel(
        &[buf(Op::buffer_load_dword, 8, 0, true, false)],
        RegisterClass::Srv,
        Some(5),
    );
    assert_eq!(refusal(&offset, NoShadow), "resource:buffer-structured-raw");
    // A descriptor the binding table does not name, read with an index, is taken for a
    // structured view.
    let mut guessed = pixel(
        &[buf(Op::buffer_load_dword, 8, 0, false, true)],
        RegisterClass::Srv,
        Some(5),
    );
    guessed.binding_table.clear();
    assert_eq!(refusal(&guessed, NoShadow), "resource:buffer-structured");
}

#[test]
fn a_four_component_store_encodes_all_four_registers() {
    let p = pixel(
        &[buf(Op::buffer_store_format_xyzw, 8, 0, false, true)],
        RegisterClass::Uav,
        Some(6),
    );
    let text = hlsl(&p, WithShadow);
    assert!(text.contains("RWByteAddressBuffer heap_u_s8_1"), "{text}");
    assert!(
        text.contains("decant_typed_encode(uint4(v8, v9, v10, v11), decant_row1.z, decant_m1)"),
        "{text}"
    );
    compile(&p, WithShadow);
}

#[test]
fn a_format_load_reads_the_dwords_of_the_element_and_decodes_by_the_format_of_the_row() {
    let p = pixel(
        &[buf(Op::buffer_load_format_xy, 8, 0, false, true)],
        RegisterClass::Srv,
        Some(3),
    );
    let text = hlsl(&p, WithShadow);
    // The layer made the typed view raw: no text declares a typed buffer.
    assert!(
        text.contains("ByteAddressBuffer heap_t_s8_1 = ResourceDescriptorHeap["),
        "{text}"
    );
    assert!(!text.contains("Buffer<"), "{text}");
    // Element size, element count and DXGI format of the host slot.
    assert!(
        text.contains(&format!(
            "uint4 decant_row1 = decant_view_shadow_load4({SLOT} * 16u);"
        )),
        "{text}"
    );
    assert!(text.contains("uint decant_i1 = v4;"), "{text}");
    // The view starts on 16 bytes, the title's first byte is the offset of the row behind it.
    assert!(
        text.contains("uint decant_a1 = decant_i1 * decant_row1.x + (decant_row1.w & 15u);"),
        "{text}"
    );
    // An element past the count, or a slot without elements, reads as zero dwords.
    assert!(
        text.contains("if (decant_row1.x != 0u && decant_i1 < decant_row1.y) {"),
        "{text}"
    );
    assert!(
        text.contains(
            "decant_m1 = uint4(heap_t_s8_1.Load((decant_a1 & ~3u)) >> ((decant_a1 & 3u) * 8u), decant_row1.x > 4u ? heap_t_s8_1.Load(decant_a1 + 4u) : 0u, decant_row1.x > 8u ? heap_t_s8_1.Load(decant_a1 + 8u) : 0u, decant_row1.x > 12u ? heap_t_s8_1.Load(decant_a1 + 12u) : 0u);"
        ),
        "{text}"
    );
    assert!(
        text.contains("uint4 decant_t1 = decant_typed_decode(decant_m1, decant_row1.z);"),
        "{text}"
    );
    assert!(text.contains("v8 = decant_t1.x;"), "{text}");
    assert!(text.contains("v9 = decant_t1.y;"), "{text}");
    assert!(!text.contains("decant_typed_encode"), "{text}");
    compile(&p, WithShadow);
    // Without an index the element is 0, an offset inside an element is not an element.
    let first = pixel(
        &[buf(Op::buffer_load_format_x, 8, 0, false, false)],
        RegisterClass::Srv,
        Some(3),
    );
    assert!(hlsl(&first, WithShadow).contains("uint decant_i1 = 0u;"));
    let inside = pixel(
        &[buf(Op::buffer_load_format_x, 8, 4, false, true)],
        RegisterClass::Srv,
        Some(3),
    );
    assert_eq!(
        refusal(&inside, WithShadow),
        "form:buffer_load_format_x:typed-offset"
    );
}

/// The case of one format in the decoder or encoder of `text`.
fn case_of<'a>(text: &'a str, function: &str, format: u32) -> &'a str {
    let body = &text[text.find(function).expect("the function")..];
    let label = format!("case {format}u:");
    body.lines()
        .find(|l| l.contains(&label))
        .unwrap_or_else(|| panic!("no {label} in\n{text}"))
}

#[test]
fn the_decoder_gives_the_registers_the_hardware_gives_for_each_format() {
    let p = pixel(
        &[buf(Op::buffer_load_format_xyzw, 8, 0, false, true)],
        RegisterClass::Srv,
        Some(3),
    );
    let text = hlsl(&p, WithShadow);
    let decode = "uint4 decant_typed_decode(uint4 m, uint format) {";
    // Floats of 32 bits are their bits, with 0, 0 and 1.0 for the channels a format lacks.
    assert!(
        case_of(&text, decode, 2).ends_with("return uint4(m.x, m.y, m.z, m.w);"),
        "{text}"
    );
    assert!(
        case_of(&text, decode, 41).ends_with("case 41u: return uint4(m.x, 0u, 0u, 0x3f800000u);"),
        "{text}"
    );
    // Integers: the missing fourth channel is the integer 1.
    assert!(
        case_of(&text, decode, 42).ends_with("return uint4(m.x, 0u, 0u, 1u);"),
        "{text}"
    );
    assert!(case_of(&text, decode, 43).contains("case 42u:"), "{text}");
    assert!(
        case_of(&text, decode, 32).ends_with(
            "return uint4(uint((int(((m.x >> 0u) & 255u) << 24u) >> 24u)), uint((int(((m.x >> 8u) & 255u) << 24u) >> 24u)), uint((int(((m.x >> 16u) & 255u) << 24u) >> 24u)), uint((int(((m.x >> 24u) & 255u) << 24u) >> 24u)));"
        ),
        "{text}"
    );
    // 8 bit normalized, in the R8G8B8A8 and in the B8G8R8A8 order.
    assert!(
        case_of(&text, decode, 28).ends_with(
            "return uint4(asuint(float(((m.x >> 0u) & 255u)) / 255.0), asuint(float(((m.x >> 8u) & 255u)) / 255.0), asuint(float(((m.x >> 16u) & 255u)) / 255.0), asuint(float(((m.x >> 24u) & 255u)) / 255.0));"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, decode, 87).ends_with(
            "return uint4(asuint(float(((m.x >> 16u) & 255u)) / 255.0), asuint(float(((m.x >> 8u) & 255u)) / 255.0), asuint(float(((m.x >> 0u) & 255u)) / 255.0), asuint(float(((m.x >> 24u) & 255u)) / 255.0));"
        ),
        "{text}"
    );
    // 16 bit: floats, normalized with and without a sign, over two dwords.
    assert!(
        case_of(&text, decode, 10).ends_with(
            "return uint4(asuint(f16tof32(((m.x >> 0u) & 65535u))), asuint(f16tof32(((m.x >> 16u) & 65535u))), asuint(f16tof32(((m.y >> 0u) & 65535u))), asuint(f16tof32(((m.y >> 16u) & 65535u))));"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, decode, 35).ends_with(
            "return uint4(asuint(float(((m.x >> 0u) & 65535u)) / 65535.0), asuint(float(((m.x >> 16u) & 65535u)) / 65535.0), 0u, 0x3f800000u);"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, decode, 37).contains(
            "asuint(max(float((int(((m.x >> 16u) & 65535u) << 16u) >> 16u)) / 32767.0, -1.0))"
        ),
        "{text}"
    );
    // The packed formats: 10, 10, 10 and 2 bits, and the floats of 11, 11 and 10 bits.
    assert!(
        case_of(&text, decode, 24).ends_with(
            "asuint(float(((m.x >> 20u) & 1023u)) / 1023.0), asuint(float(((m.x >> 30u) & 3u)) / 3.0));"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, decode, 26).ends_with(
            "return uint4(asuint(f16tof32(((m.x >> 0u) & 2047u) << 4u)), asuint(f16tof32(((m.x >> 11u) & 2047u) << 4u)), asuint(f16tof32(((m.x >> 22u) & 1023u) << 5u)), 0x3f800000u);"
        ),
        "{text}"
    );
    // A format the switch does not know reads zeros.
    assert!(
        text.contains("    default: return uint4(0u, 0u, 0u, 0u);"),
        "{text}"
    );
    assert!(!text.contains("case 0u:"), "{text}");
}

#[test]
fn a_typed_view_is_also_read_raw_by_the_same_program() {
    // The buffer of one heap slot read three ways: a format load by element, dwords at an
    // index and an offset, and a scalar read.
    let p = pixel(
        &[
            buf(Op::buffer_load_format_x, 8, 0, false, true),
            buf(Op::buffer_load_dword, 9, 8, true, true),
            buf(Op::buffer_load_dword, 11, 0x10, true, false),
            smem(Op::s_buffer_load_dword, s(6), s(0), 4),
            vop1(Op::v_mov_b32, v(10), s(6)),
        ],
        RegisterClass::Srv,
        Some(3),
    );
    let text = hlsl(&p, WithShadow);
    assert_eq!(text.matches("ByteAddressBuffer heap_t_s8_1 = ").count(), 1);
    assert!(
        text.contains("uint4 decant_t1 = decant_typed_decode(decant_m1, decant_row1.z);"),
        "{text}"
    );
    // A raw access with an index: the element size of the row is the stride.
    assert!(
        text.contains(&format!(
            "uint decant_a2 = decant_i2 * decant_row2.x + ((8u + v5) + decant_view_offset({SLOT}));"
        )),
        "{text}"
    );
    assert!(
        text.contains(
            "((decant_row2.x != 0u && decant_i2 < decant_row2.y) ? heap_t_s8_1.Load((decant_a2 + 0u)) : 0u)"
        ),
        "{text}"
    );
    // Without an index: the byte address alone.
    assert!(
        text.contains(&format!(
            "uint decant_a3 = ((16u + v4) + decant_view_offset({SLOT}));"
        )),
        "{text}"
    );
    assert!(text.contains("heap_t_s8_1.Load((decant_a3 + 0u))"), "{text}");
    assert!(
        text.contains(&format!("heap_t_s8_1.Load((4u + decant_view_offset({SLOT})))")),
        "{text}"
    );
    compile(&p, WithShadow);
}

#[test]
fn a_raw_typed_view_needs_the_shadow_and_a_view_the_title_made_raw_has_no_format() {
    let typed = pixel(
        &[buf(Op::buffer_load_format_x, 8, 0, false, true)],
        RegisterClass::Srv,
        Some(3),
    );
    assert_eq!(refusal(&typed, NoShadow), "resource:buffer-structured");
    let raw = pixel(
        &[buf(Op::buffer_load_dword, 8, 0, true, false)],
        RegisterClass::Srv,
        Some(3),
    );
    assert_eq!(refusal(&raw, NoShadow), "resource:buffer-structured-raw");
    for psv in [4, 5] {
        let p = pixel(
            &[buf(Op::buffer_load_format_x, 8, 0, false, true)],
            RegisterClass::Srv,
            Some(psv),
        );
        assert_eq!(refusal(&p, WithShadow), "resource:buffer-view-mismatch");
    }
}

#[test]
fn a_format_store_encodes_by_the_format_of_the_row_and_writes_the_dwords_of_the_element() {
    let p = pixel(
        &[buf(Op::buffer_store_format_xy, 8, 0, false, true)],
        RegisterClass::Uav,
        Some(6),
    );
    let text = hlsl(&p, WithShadow);
    assert!(
        text.contains("RWByteAddressBuffer heap_u_s8_1 = ResourceDescriptorHeap["),
        "{text}"
    );
    assert!(!text.contains("Buffer<"), "{text}");
    // Only in an active lane, only inside the view and only for a format the encoder knows.
    for line in [
        "    if (exec) {",
        "        if (decant_row1.x != 0u && decant_i1 < decant_row1.y) {",
        "            if (decant_typed_encode(uint4(v8, v9, 0u, 0u), decant_row1.z, decant_m1)) {",
        // An element of 1 or 2 bytes changes only its bytes of the dword.
        "                if (decant_row1.x < 4u) {",
        "                    uint decant_k1 = decant_row1.x == 1u ? 0xffu : 0xffffu;",
        "                    heap_u_s8_1.InterlockedAnd((decant_a1 & ~3u), ~(decant_k1 << ((decant_a1 & 3u) * 8u)));",
        "                    heap_u_s8_1.InterlockedOr((decant_a1 & ~3u), (decant_m1.x & decant_k1) << ((decant_a1 & 3u) * 8u));",
        "                } else {",
        "                    heap_u_s8_1.Store(decant_a1, decant_m1.x);",
        "                    if (decant_row1.x > 4u) heap_u_s8_1.Store(decant_a1 + 4u, decant_m1.y);",
        "                    if (decant_row1.x > 12u) heap_u_s8_1.Store(decant_a1 + 12u, decant_m1.w);",
    ] {
        assert!(text.contains(line), "missing {line} in\n{text}");
    }
    let encode = "bool decant_typed_encode(uint4 d, uint format, out uint4 m) {";
    assert!(
        case_of(&text, encode, 16).ends_with("m = uint4(d.x, d.y, 0u, 0u); return true;"),
        "{text}"
    );
    assert!(
        case_of(&text, encode, 28).ends_with(
            "m = uint4(uint(saturate(asfloat(d.x)) * 255.0 + 0.5) | (uint(saturate(asfloat(d.y)) * 255.0 + 0.5) << 8u) | (uint(saturate(asfloat(d.z)) * 255.0 + 0.5) << 16u) | (uint(saturate(asfloat(d.w)) * 255.0 + 0.5) << 24u), 0u, 0u, 0u); return true;"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, encode, 87).contains("uint(saturate(asfloat(d.z)) * 255.0 + 0.5) | "),
        "{text}"
    );
    assert!(
        case_of(&text, encode, 34).ends_with(
            "m = uint4((decant_pack_small_float(d.x, 10u) | ((d.x >> 16u) & 32768u)) | ((decant_pack_small_float(d.y, 10u) | ((d.y >> 16u) & 32768u)) << 16u), 0u, 0u, 0u); return true;"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, encode, 13).contains(
            "(uint(decant_to_snorm(asfloat(d.z), 32767.0)) & 65535u) | ((uint(decant_to_snorm(asfloat(d.w), 32767.0)) & 65535u) << 16u)"
        ),
        "{text}"
    );
    assert!(
        case_of(&text, encode, 30).contains("(d.x & 255u) | ((d.y & 255u) << 8u)"),
        "{text}"
    );
    assert!(case_of(&text, encode, 26).contains(
        "decant_pack_unsigned_float(d.x, 6u) | (decant_pack_unsigned_float(d.y, 6u) << 11u) | (decant_pack_unsigned_float(d.z, 5u) << 22u)"
    ), "{text}");
    assert!(text.contains("if (exponent == 255u) return infinity | uint(mantissa != 0u);"));
    assert!(text.contains("if (exponent > 142u) return infinity - 1u;"));
    assert!(text.contains("(mantissa >> shift)"));
    assert!(text.contains("    default: return false;"), "{text}");
    assert!(!text.contains("decant_typed_decode"), "{text}");
    compile(&p, WithShadow);
}

#[test]
fn d16_format_loads_pack_pairs_and_preserve_the_unused_half() {
    for (op, components) in [
        (Op::buffer_load_format_d16_x, 1u8),
        (Op::buffer_load_format_d16_xy, 2),
        (Op::buffer_load_format_d16_xyz, 3),
        (Op::buffer_load_format_d16_xyzw, 4),
    ] {
        let p = pixel(&[buf(op, 8, 0, false, true)], RegisterClass::Srv, Some(3));
        let text = hlsl(&p, WithShadow);
        assert!(text.contains("f32tof16(asfloat(decant_t1.x))"), "{text}");
        assert!(text.contains("decant_t1.x & 65535u"), "{text}");
        let last = 8 + components.div_ceil(2) - 1;
        assert!(text.contains(&format!("v{last} = (decant_kind1")), "{text}");
        assert!(
            !text.contains(&format!("v{} = (decant_kind1", last + 1)),
            "{text}"
        );
        assert_eq!(
            text.contains(&format!("v{last} & 0xffff0000u")),
            components % 2 != 0,
            "{text}"
        );
        compile(&p, WithShadow);
    }
}

#[test]
fn d16_format_stores_unpack_halves_using_the_view_type() {
    for (op, components) in [
        (Op::buffer_store_format_d16_x, 1u8),
        (Op::buffer_store_format_d16_xy, 2),
        (Op::buffer_store_format_d16_xyz, 3),
        (Op::buffer_store_format_d16_xyzw, 4),
    ] {
        let p = pixel(&[buf(op, 8, 0, false, true)], RegisterClass::Uav, Some(6));
        let text = hlsl(&p, WithShadow);
        assert!(text.contains("decant_typed_d16_kind(decant_row1.z)"), "{text}");
        for (i, field) in [
            (0, "v8 >> 0u"),
            (1, "v8 >> 16u"),
            (2, "v9 >> 0u"),
            (3, "v9 >> 16u"),
        ] {
            assert_eq!(
                text.contains(&format!("f16tof32((({field}) & 65535u))")),
                i < components,
                "{text}"
            );
        }
        assert!(text.contains("decant_kind1 == 2u ? uint(int("), "{text}");
        assert!(text.contains("decant_typed_encode(uint4("), "{text}");
        compile(&p, WithShadow);
    }
}

#[test]
fn a_tbuffer_load_converts_with_the_format_of_the_instruction() {
    // 8_8_8_8_UNORM, then 16_16_SNORM.
    let p = pixel(
        &[
            with_format(buf(Op::tbuffer_load_format_xyzw, 8, 0, true, false), 56),
            with_format(buf(Op::tbuffer_load_format_xy, 12, 4, true, false), 24),
        ],
        RegisterClass::Srv,
        Some(4),
    );
    let text = hlsl(&p, NoShadow);
    assert!(
        text.contains(
            "uint4 decant_tfmt56(uint4 m) { return uint4(asuint(float(((m.x >> 0u) & 255u)) / 255.0)"
        ),
        "{text}"
    );
    assert!(
        text.contains("asuint(max(float((int(((m.x >> 16u) & 65535u) << 16u) >> 16u)) / 32767.0, -1.0)), 0u, 0x3f800000u)"),
        "{text}"
    );
    assert!(
        text.contains("decant_tfmt56(uint4(heap_t_s8_1.Load((decant_a1 + 0u)), 0u, 0u, 0u))"),
        "{text}"
    );
    compile(&p, NoShadow);
}

#[test]
fn byte_and_short_loads_extract_and_extend() {
    let p = pixel(
        &[
            buf(Op::buffer_load_sbyte, 8, 1, true, false),
            buf(Op::buffer_load_ushort, 9, 2, true, false),
        ],
        RegisterClass::Srv,
        Some(4),
    );
    let text = hlsl(&p, NoShadow);
    assert!(
        text.contains("uint decant_t1_0 = uint(int(((heap_t_s8_1.Load((decant_a1 & ~3u)) >> ((decant_a1 & 3u) * 8u)) & 255u) << 24u) >> 24u);"),
        "{text}"
    );
    assert!(
        text.contains(
            "uint decant_t2_0 = ((heap_t_s8_1.Load((decant_a2 & ~3u)) >> ((decant_a2 & 3u) * 8u)) & 65535u);"
        ),
        "{text}"
    );
    compile(&p, NoShadow);
}

#[test]
fn raw_stores_and_byte_stores_happen_only_in_active_lanes() {
    let p = pixel(
        &[
            buf(Op::buffer_store_dwordx2, 8, 8, true, false),
            buf(Op::buffer_store_byte, 10, 3, true, false),
        ],
        RegisterClass::Uav,
        Some(7),
    );
    let text = hlsl(&p, NoShadow);
    // Without a shadow the views are where the title put them.
    assert!(!text.contains("decant_view_offset"), "{text}");
    assert!(text.contains("uint decant_a1 = (8u + v4);"), "{text}");
    assert!(text.contains("RWByteAddressBuffer heap_u_s8_1"), "{text}");
    assert!(
        text.contains("    heap_u_s8_1.Store(decant_a1 + 4u, v9);"),
        "{text}"
    );
    assert!(
        text.contains("    heap_u_s8_1.InterlockedOr(decant_d2, (v10 & 0xffu) << decant_s2);"),
        "{text}"
    );
    compile(&p, NoShadow);
}

#[test]
fn a_raw_view_in_a_heap_slot_is_read_and_written_behind_the_view_offset_of_its_row() {
    let p = pixel(
        &[
            buf(Op::buffer_load_dword, 8, 4, true, false),
            buf(Op::buffer_store_dword, 8, 8, true, false),
            with_glc(buf(Op::buffer_atomic_add, 9, 0, false, false)),
            smem(Op::s_buffer_load_dword, s(6), s(0), 0),
            vop1(Op::v_mov_b32, v(10), s(6)),
        ],
        RegisterClass::Uav,
        Some(7),
    );
    let text = hlsl(&p, WithShadow);
    for needle in [
        format!("uint decant_a1 = ((4u + v4) + decant_view_offset({SLOT}));"),
        format!("uint decant_a2 = ((8u + v4) + decant_view_offset({SLOT}));"),
        "heap_u_s8_1.Store(decant_a2 + 0u, v8);".to_owned(),
        format!("uint decant_a3 = decant_view_offset({SLOT});"),
        "heap_u_s8_1.InterlockedAdd(decant_a3, v9, decant_o3);".to_owned(),
        format!("heap_u_s8_1.Load(decant_view_offset({SLOT}))"),
    ] {
        assert!(
            text.contains(&needle),
            "missing {needle} in
{text}"
        );
    }
    compile(&p, WithShadow);
}

#[test]
fn atomics_return_the_old_value_only_with_glc_and_loop_where_hlsl_has_none() {
    let p = pixel(
        &[
            with_glc(buf(Op::buffer_atomic_add, 8, 0, true, false)),
            buf(Op::buffer_atomic_smax, 9, 4, true, false),
            with_glc(buf(Op::buffer_atomic_inc, 10, 8, true, false)),
            with_glc(buf(Op::buffer_atomic_cmpswap, 12, 12, true, false)),
        ],
        RegisterClass::Uav,
        Some(7),
    );
    let text = hlsl(&p, NoShadow);
    assert!(
        text.contains("heap_u_s8_1.InterlockedAdd(decant_a1, v8, decant_o1);"),
        "{text}"
    );
    assert!(text.contains("    v8 = decant_o1;"), "{text}");
    assert!(
        text.contains("heap_u_s8_1.InterlockedMax(decant_a2, int(v9), decant_so2);"),
        "{text}"
    );
    assert!(!text.contains("v9 = decant_o2;"), "{text}");
    assert!(
        text.contains("heap_u_s8_1.InterlockedCompareExchange(decant_a3, decant_o3, (decant_o3 >= v10 ? 0u : decant_o3 + 1u), decant_seen3);"),
        "{text}"
    );
    assert!(
        text.contains("heap_u_s8_1.InterlockedCompareExchange(decant_a4, v13, v12, decant_o4);"),
        "{text}"
    );
    compile(&p, NoShadow);
}

#[test]
fn a_uav_with_a_counter_is_refused_by_name() {
    let p = pixel(
        &[buf(Op::buffer_store_dword, 8, 4, false, true)],
        RegisterClass::Uav,
        Some(9),
    );
    assert_eq!(refusal(&p, WithShadow), "resource:buffer-counter");
}

#[test]
fn an_indexed_structured_store_is_dropped_past_the_record_count() {
    let p = pixel(
        &[buf(Op::buffer_store_dword, 8, 4, false, true)],
        RegisterClass::Uav,
        Some(8),
    );
    let text = hlsl(&p, WithShadow);
    assert!(
        text.contains("    if (decant_row1.x != 0u && decant_i1 < decant_row1.y) {"),
        "{text}"
    );
    assert!(
        text.contains("        heap_u_s8_1.Store(decant_a1 + 0u, v8);"),
        "{text}"
    );
    compile(&p, WithShadow);
}

fn with_soffset(mut i: Insn, reg: u8) -> Insn {
    if let Ext::Smem { soffset, .. } = &mut i.ext {
        *soffset = s(reg);
    }
    i
}

#[test]
fn a_scalar_constant_read_drops_the_low_bits_of_each_offset_on_its_own() {
    // s[4:7] = dwords 0x13 + (s20 & ~3) .. of the constant buffer, moved to v8 to v9.
    let p = pixel(
        &[
            with_soffset(smem(Op::s_buffer_load_dwordx2, s(4), s(0), 0x13), 20),
            vop1(Op::v_mov_b32, v(8), s(4)),
            vop1(Op::v_mov_b32, v(9), s(5)),
        ],
        RegisterClass::Cbv,
        Some(2),
    );
    let text = hlsl(&p, NoShadow);
    assert!(
        text.contains("heap_b_s8_1.w[((s20 & 4294967292u) + 16u) >> 4]"),
        "{text}"
    );
    assert!(
        text.contains("heap_b_s8_1.w[((s20 & 4294967292u) + 20u) >> 4]"),
        "{text}"
    );
    compile(&p, NoShadow);
}

#[test]
fn a_scalar_read_into_vcc_is_a_mask_from_memory() {
    let p = pixel(
        &[smem(Op::s_buffer_load_dword, VCC, s(0), 0)],
        RegisterClass::Cbv,
        Some(2),
    );
    assert_eq!(refusal(&p, NoShadow), "exec:mask-as-data");
}

/// Descriptors a program takes by heap index at run time (`Origin::HeapIndex`): the printer
/// gets them from the resource pass, so these tests build the function directly. The index
/// register s4 holds a HOST heap index at each use.
mod by_heap_index {
    use super::common;
    use decant_shader::api::*;
    use decant_shader::hlsl::{self, Context};
    use decant_shader::ir::*;
    use decant_shader::stage::Interface;
    use decant_shader::tables::Op;

    fn resource(written: bool, psv: Option<u32>) -> Resource {
        let class = if written {
            RegisterClass::Uav
        } else {
            RegisterClass::Srv
        };
        Resource {
            origin: Origin::HeapIndex {
                index: Var::S(4),
                sampler: false,
            },
            usage: ResourceUse::Words { written },
            declared: psv.map(|psv_type| DeclaredResource {
                class,
                lower: 0,
                upper: 0,
                space: 0,
                psv_type,
                psv_kind: 0,
            }),
            placement: Some(Placement::HeapAtConsoleIndex { class }),
            name: format!("con_{}", class.letter()),
        }
    }

    fn op(op: Op, store: bool, count: u8, index: Option<Expr>, glc: bool) -> Stmt {
        // A typed view is addressed by element, the others at byte 16.
        let offset = if op.name().contains("format") { 0 } else { 16 };
        Stmt::Buffer(
            BufferOp {
                op,
                glc,
                store,
                bytes: 4,
                count,
                data: 8,
                index,
                offset: Expr::U32(offset),
                format: 0,
                buffer: Desc::Res(ResId(0)),
                pc: 0,
            },
            Guard::Exec,
        )
    }

    /// Prints the statements, the index changing between them, and compiles the text.
    fn print(stmts: Vec<Stmt>, res: Resource) -> String {
        let mut body = vec![Stmt::Set {
            dst: Var::S(4),
            value: Expr::U32(7),
            guard: Guard::Always,
        }];
        for s in stmts {
            body.push(s);
            body.push(Stmt::Set {
                dst: Var::S(4),
                value: Expr::bin(Bin::Add, Expr::Var(Var::S(4)), Expr::U32(1)),
                guard: Guard::Always,
            });
        }
        let func = Function {
            body: vec![Node::Block(body), Node::Return],
            resources: vec![res],
            groupshared_dwords: 0,
            lanes: 0,
        };
        let iface = Interface::default();
        let cx = Context {
            iface: &iface,
            mirror: Some(MirrorAbi::default()),
            header: &[],
            code: &[],
        };
        let text = hlsl::print(&func, &cx).unwrap_or_else(|r| panic!("{r}"));
        if let Some(dxc) = common::find_dxc() {
            dxc.compile(&text, "main", "ps_6_6", &[])
                .unwrap_or_else(|e| panic!("{e}\n{text}"));
        }
        text
    }

    #[test]
    fn a_raw_buffer_is_made_at_each_use_from_the_index_it_has_then() {
        let text = print(
            vec![
                op(Op::buffer_load_dword, false, 1, None, false),
                op(Op::buffer_load_dwordx2, false, 2, None, false),
            ],
            resource(false, Some(4)),
        );
        assert!(
            text.contains("ByteAddressBuffer con_t_2 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(
            text.contains("ByteAddressBuffer con_t_4 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(text.contains("con_t_4.Load((decant_a4 + 4u))"), "{text}");
        // Nothing is declared once at the top.
        assert!(!text.contains("con_t ="), "{text}");
    }

    #[test]
    fn a_structured_buffer_by_index_reads_the_shadow_row_of_that_host_slot() {
        let text = print(
            vec![op(
                Op::buffer_load_dword,
                false,
                1,
                Some(Expr::Var(Var::V(4))),
                false,
            )],
            resource(false, Some(5)),
        );
        assert!(
            text.contains("uint2 decant_row2 = decant_view_shadow_load2(s4 * 16u);"),
            "{text}"
        );
        assert!(
            text.contains("uint decant_a2 = decant_i2 * decant_row2.x + (16u + decant_view_offset(s4));"),
            "{text}"
        );
    }

    #[test]
    fn a_typed_buffer_by_index_is_a_raw_buffer_decoded_by_the_row_of_that_host_slot() {
        let text = print(
            vec![
                op(
                    Op::buffer_load_format_xyzw,
                    false,
                    4,
                    Some(Expr::Var(Var::V(4))),
                    false,
                ),
                // The same descriptor read raw, which one typed view could not serve.
                op(Op::buffer_load_dword, false, 1, None, false),
                Stmt::Set {
                    dst: Var::S(9),
                    value: Expr::BufferWord {
                        res: ResId(0),
                        offset: Box::new(Expr::U32(8)),
                    },
                    guard: Guard::Always,
                },
            ],
            resource(false, Some(3)),
        );
        assert!(
            text.contains("ByteAddressBuffer con_t_2 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(
            text.contains("uint4 decant_row2 = decant_view_shadow_load4(s4 * 16u);"),
            "{text}"
        );
        assert!(
            text.contains("uint4 decant_t2 = decant_typed_decode(decant_m2, decant_row2.z);"),
            "{text}"
        );
        assert!(text.contains("con_t_4.Load((decant_a4 + 0u))"), "{text}");
        assert!(
            text.contains("s9 = con_t_5.Load((8u + decant_view_offset(s4)));"),
            "{text}"
        );
        assert!(!text.contains("Buffer<"), "{text}");
    }

    #[test]
    fn a_store_and_an_atomic_by_index_write_through_the_descriptor_of_their_moment() {
        let text = print(
            vec![
                op(Op::buffer_store_dwordx2, true, 2, None, false),
                op(Op::buffer_atomic_add, true, 1, None, true),
            ],
            resource(true, None),
        );
        assert!(
            text.contains("RWByteAddressBuffer con_u_2 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(text.contains("con_u_2.Store(decant_a2 + 4u, v9);"), "{text}");
        assert!(
            text.contains("RWByteAddressBuffer con_u_4 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(
            text.contains("con_u_4.InterlockedAdd(decant_a4, v8, decant_o4);"),
            "{text}"
        );
    }

    #[test]
    fn a_scalar_read_by_index_makes_the_buffer_before_its_statement() {
        let read = Stmt::Set {
            dst: Var::S(9),
            value: Expr::BufferWord {
                res: ResId(0),
                offset: Box::new(Expr::U32(8)),
            },
            guard: Guard::Always,
        };
        let text = print(vec![read], resource(false, Some(4)));
        assert!(
            text.contains("ByteAddressBuffer con_t_1 = ResourceDescriptorHeap[NonUniformResourceIndex(s4)];"),
            "{text}"
        );
        assert!(
            text.contains("s9 = con_t_1.Load((8u + decant_view_offset(s4)));"),
            "{text}"
        );
    }
}

#[test]
fn atomics_on_a_typed_view_work_on_the_dword_of_the_element() {
    let p = pixel(
        &[
            buf(Op::buffer_atomic_umin, 8, 0, false, true),
            with_glc(buf(Op::buffer_atomic_add, 9, 0, false, true)),
            with_glc(buf(Op::buffer_atomic_smax, 11, 0, false, true)),
            with_glc(buf(Op::buffer_atomic_inc, 12, 0, false, true)),
            with_glc(buf(Op::buffer_atomic_cmpswap, 13, 0, false, true)),
            buf(Op::buffer_atomic_or, 10, 0, false, true),
        ],
        RegisterClass::Uav,
        Some(6),
    );
    let text = hlsl(&p, WithShadow);
    for needle in [
        "RWByteAddressBuffer heap_u_s8_1 = ResourceDescriptorHeap[NonUniformResourceIndex(decant_mirror[2].x + 1u)];",
        "uint decant_a1 = decant_i1 * decant_row1.x + decant_view_offset((decant_mirror[2].x + 1u));",
        "    if (decant_row1.x != 0u && decant_i1 < decant_row1.y) {",
        "heap_u_s8_1.InterlockedOr(decant_a6, v10, decant_o6);",
        "heap_u_s8_1.InterlockedMin(decant_a1, v8, decant_o1);",
        "heap_u_s8_1.InterlockedAdd(decant_a2, v9, decant_o2);",
        "heap_u_s8_1.InterlockedMax(decant_a3, int(v11), decant_so3);",
        "heap_u_s8_1.InterlockedCompareExchange(decant_a4, decant_o4, (decant_o4 >= v12 ? 0u : decant_o4 + 1u), decant_seen4);",
        "heap_u_s8_1.InterlockedCompareExchange(decant_a5, v14, v13, decant_o5);",
    ] {
        assert!(text.contains(needle), "missing {needle} in\n{text}");
    }
    assert!(!text.contains("Buffer<"), "{text}");
    compile(&p, WithShadow);
}
