//! Image instructions: each form built with the assembler, translated, and compiled with DXC.
//!
//! The programs are pixel programs of one shape: the image descriptor comes from slot 1 of the
//! table in s8, the sampler is built from constants, the address registers start at v2, the
//! result goes to v10 and up and is exported as the colour. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Insn, Operand};
use decant_shader::tables::Op;
use decant_shader::translate;

const SAMPLER_WORDS: [u32; 4] = [0x400, 0x00c0_0000, 0x18f0_0000, 0];

fn element(semantic: &str, register: u32, mask: u8, read: u8, sv: u32) -> SigElement {
    SigElement {
        semantic: semantic.to_owned(),
        index: 0,
        system_value: sv,
        component_type: 3,
        register,
        mask,
        rw_mask: read,
    }
}

/// Sets the image fields of an instruction made by `asm::image`.
fn with(mut insn: Insn, f: impl FnOnce(&mut decant_shader::decode::Mimg)) -> Insn {
    if let Ext::Mimg(m) = &mut insn.ext {
        f(m);
    }
    insn
}

/// A pixel program: table slot 1 of s8 into s[4:11], the sampler words into s[0:3], the
/// `setup` instructions (which fill v2 and up), `op`, then v10 to v13 exported. `sampler_words`
/// changes the constant sampler.
fn pixel_with(setup: &[Insn], op: Insn, sampler_words: [u32; 4]) -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x20));
    for (k, w) in sampler_words.iter().enumerate() {
        a.push(sop1(Op::s_mov_b32, s(k as u8), lit(*w)));
    }
    for i in setup {
        a.push(*i);
    }
    a.push(op);
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(10), v(11)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(12), v(13)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs: vec![element("SV_Position", 0, 15, 0, 1)],
        outputs: vec![element("SV_Target", 0, 15, 0, 64)],
        name: "image_test".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

fn pixel(setup: &[Insn], op: Insn) -> StageProgram {
    pixel_with(setup, op, SAMPLER_WORDS)
}

/// Fills v2 to v2 + n - 1 with distinct values: floats 0.25 apart, or small integers.
fn regs(n: u8, float_values: bool) -> Vec<Insn> {
    (0..n)
        .map(|k| {
            let value = if float_values {
                lit((0.25 + 0.25 * f32::from(k)).to_bits())
            } else {
                int(k as i8 + 1)
            };
            vop1(Op::v_mov_b32, v(2 + k), value)
        })
        .collect()
}

/// Translates and compiles. Returns the HLSL.
fn hlsl(p: &StageProgram) -> String {
    let host = CountingHost::default();
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    let text = t.stages[0].hlsl.clone();
    if let Some(dxc) = common::find_dxc() {
        let s = &t.stages[0];
        if let Err(e) = decant_shader::api::Compiler::compile(
            &dxc,
            &s.hlsl,
            &s.entry,
            &s.profile,
            &s.dxc_args,
        ) {
            panic!("{e}\n{text}");
        }
    } else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
    }
    text
}

fn refusal(p: &StageProgram) -> String {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err()
    .key()
}

fn img(op: Op, dim: u8, dmask: u8) -> Insn {
    image(op, v(10), v(2), s(4), s(0), dim, dmask)
}

fn has(text: &str, needle: &str) {
    assert!(text.contains(needle), "missing `{needle}` in\n{text}");
}

#[test]
fn a_load_reads_the_texel_at_integer_coordinates_and_mip_zero() {
    let t = hlsl(&pixel(&regs(2, false), img(Op::image_load, 1, 0xf)));
    has(&t, "Texture2D<float4> heap_t_s8_1");
    has(&t, ".Load(int3(int(v2), int(v3), 0))");
    has(&t, "v13 = asuint(decant_t1.w);");
}

#[test]
fn a_load_with_a_mip_takes_the_operand_after_the_coordinates() {
    let t = hlsl(&pixel(&regs(4, false), img(Op::image_load_mip, 5, 0x3)));
    has(&t, "Texture2DArray<float4>");
    has(&t, ".Load(int4(int(v2), int(v3), int(v4), int(v5)))");
    has(&t, "v11 = asuint(decant_t1.y);");
}

#[test]
fn a_load_of_one_and_three_dimensional_images() {
    let t = hlsl(&pixel(&regs(2, false), img(Op::image_load_mip, 0, 0x1)));
    has(&t, "Texture1D<float4>");
    has(&t, ".Load(int2(int(v2), int(v3)))");
    let t = hlsl(&pixel(&regs(3, false), img(Op::image_load, 2, 0x1)));
    has(&t, "Texture3D<float4>");
    has(&t, ".Load(int4(int(v2), int(v3), int(v4), 0))");
}

#[test]
fn a_multisample_load_takes_the_fragment_index_as_the_sample() {
    let t = hlsl(&pixel(&regs(3, false), img(Op::image_load, 6, 0x1)));
    has(&t, "Texture2DMS<float4>");
    has(&t, ".Load(int2(int(v2), int(v3)), int(v4))");
}

#[test]
fn sixteen_bit_coordinates_are_unpacked_two_per_register() {
    let op = with(img(Op::image_load_mip, 1, 0x1), |m| m.a16 = true);
    let t = hlsl(&pixel(&regs(2, false), op));
    has(&t, "int((v2 & 65535u))");
    has(&t, "int((v2 >> (16u & 31u)))");
    has(&t, "int((v3 & 65535u))");
    let op = with(img(Op::image_sample, 1, 0x1), |m| m.a16 = true);
    let t = hlsl(&pixel(&regs(1, false), op));
    has(&t, "decant_img_f16lo(v2)");
    has(&t, "decant_img_f16hi(v2)");
}

#[test]
fn sixteen_bit_results_are_packed_two_per_register() {
    let op = with(img(Op::image_sample_lz, 1, 0x7), |m| m.d16 = true);
    let t = hlsl(&pixel(&regs(2, true), op));
    has(
        &t,
        "v10 = (f32tof16(decant_t1.x) | (f32tof16(decant_t1.y) << 16));",
    );
    has(&t, "v11 = f32tof16(decant_t1.z);");
}

#[test]
fn the_status_register_follows_the_data() {
    let op = with(img(Op::image_load, 1, 0x3), |m| m.tfe = true);
    let t = hlsl(&pixel(&regs(2, false), op));
    has(&t, "v12 = 0u;");
}

#[test]
fn non_sequential_addresses_come_from_the_extra_words() {
    let op = with(img(Op::image_sample_l, 1, 0xf), |m| {
        m.nsa_words = 1;
        m.nsa[..2].copy_from_slice(&[7, 5]);
    });
    let mut setup = regs(1, true);
    setup.push(vop1(Op::v_mov_b32, v(7), float(0.5)));
    setup.push(vop1(Op::v_mov_b32, v(5), float(2.0)));
    let t = hlsl(&pixel(&setup, op));
    has(
        &t,
        ".SampleLevel(smp_0, float2(asfloat(v2), asfloat(v7)), asfloat(v5))",
    );
}

#[test]
fn compare_samples_use_a_comparison_sampler() {
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_sample_c_lz, 1, 0x1)));
    has(&t, "SamplerComparisonState smp_0");
    has(
        &t,
        ".SampleCmpLevelZero(smp_0, float2(asfloat(v3), asfloat(v4)), asfloat(v2)).xxxx",
    );
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_sample_c, 1, 0x1)));
    has(
        &t,
        ".SampleCmp(smp_0, float2(asfloat(v3), asfloat(v4)), asfloat(v2))",
    );
}

#[test]
fn an_offset_is_unpacked_from_its_six_bit_fields() {
    // x = 1, y = -2 at bits 5:0 and 13:8.
    let mut setup = vec![vop1(Op::v_mov_b32, v(2), lit(0x3e01))];
    setup.extend(regs(3, true).into_iter().skip(1));
    let t = hlsl(&pixel(&setup, img(Op::image_sample_lz_o, 1, 0xf)));
    has(&t, "(int(v2 << 26u) >> 26), (int(v2 << 18u) >> 26)");
}

#[test]
fn gathers_pick_the_channel_from_the_mask() {
    let t = hlsl(&pixel(&regs(2, true), img(Op::image_gather4_lz, 1, 0x2)));
    has(&t, ".GatherGreen(smp_0, float2(asfloat(v2), asfloat(v3)))");
    has(&t, "v13 = asuint(decant_t1.w);");
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_gather4_c_lz, 1, 0x1)));
    has(
        &t,
        ".GatherCmpRed(smp_0, float2(asfloat(v3), asfloat(v4)), asfloat(v2))",
    );
}

#[test]
fn explicit_derivatives_come_before_the_coordinates() {
    let t = hlsl(&pixel(&regs(6, true), img(Op::image_sample_d, 1, 0xf)));
    has(
        &t,
        ".SampleGrad(smp_0, float2(asfloat(v6), asfloat(v7)), float2(asfloat(v2), asfloat(v3)), float2(asfloat(v4), asfloat(v5)))",
    );
    let t = hlsl(&pixel(&regs(7, true), img(Op::image_sample_d_cl, 1, 0xf)));
    has(&t, ", int2(0, 0), asfloat(v8))");
}

#[test]
fn a_bias_comes_first() {
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_sample_b, 1, 0xf)));
    has(
        &t,
        ".SampleBias(smp_0, float2(asfloat(v3), asfloat(v4)), asfloat(v2))",
    );
}

#[test]
fn a_cube_sample_rebuilds_the_direction_from_the_face() {
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_sample_lz, 3, 0xf)));
    has(&t, "TextureCube<float4>");
    has(&t, "float decant_t1_a = 2.0 * (asfloat(v2) - 1.5);");
    has(&t, ".SampleLevel(smp_0, decant_t1_d, 0.0)");
}

#[test]
fn a_sampler_with_texel_coordinates_divides_by_the_size() {
    let mut words = SAMPLER_WORDS;
    words[0] |= 0x8000;
    let t = hlsl(&pixel_with(
        &regs(2, true),
        img(Op::image_sample_lz, 1, 0xf),
        words,
    ));
    has(&t, "GetDimensions(decant_t1_w, decant_t1_h);");
    has(&t, "(asfloat(v2) / decant_t1_w)");
}

#[test]
fn the_level_of_detail_query_gives_clamped_and_raw_levels() {
    let t = hlsl(&pixel(&regs(2, true), img(Op::image_get_lod, 1, 0x3)));
    has(
        &t,
        "CalculateLevelOfDetail(smp_0, float2(asfloat(v2), asfloat(v3)))",
    );
    has(&t, "CalculateLevelOfDetailUnclamped(");
}

#[test]
fn resinfo_gives_size_and_mip_count_as_integers() {
    let t = hlsl(&pixel(&regs(1, false), img(Op::image_get_resinfo, 1, 0xf)));
    has(&t, ".GetDimensions(v2, decant_t1.x, decant_t1.y, decant_t1.w);");
    has(&t, "v12 = decant_t1.z;");
}

#[test]
fn a_store_writes_the_masked_components_under_the_lane_test() {
    let p = pixel(
        &regs(2, false),
        image(Op::image_store, v(10), v(2), s(4), s(0), 1, 0x3),
    );
    let t = hlsl(&p);
    has(&t, "RWTexture2D<float4> heap_u_s8_1");
    has(
        &t,
        "heap_u_s8_1[uint2(uint(int(v2)), uint(int(v3)))] = float4(asfloat(v10), asfloat(v11), 0.0, 0.0);",
    );
}

#[test]
fn an_atomic_reads_an_integer_view_of_the_same_slot() {
    let mut op = image(Op::image_atomic_umax, v(10), v(2), s(4), s(0), 1, 0x1);
    op = with(op, |m| m.glc = true);
    let t = hlsl(&pixel(&regs(2, false), op));
    has(
        &t,
        "RWTexture2D<uint> decant_t1_u = ResourceDescriptorHeap[decant_mirror[2].x + 1u];",
    );
    has(
        &t,
        "InterlockedMax(decant_t1_u[uint2(uint(int(v2)), uint(int(v3)))], uint(v10), decant_t1_o);",
    );
    has(&t, "v10 = uint(decant_t1_o);");
}

#[test]
fn forms_without_an_hlsl_equivalent_are_refused_by_name() {
    let p = pixel(&regs(4, true), img(Op::image_gather4_c_l, 1, 0x1));
    assert_eq!(refusal(&p), "form:image_sample:image-gather-lod");
    let p = pixel(&regs(4, true), img(Op::image_sample_c_l, 1, 0x1));
    assert_eq!(refusal(&p), "form:image_sample:image-compare-lod");
    let p = pixel(&regs(2, false), img(Op::image_atomic_fmax, 1, 0x1));
    assert_eq!(refusal(&p), "form:image_atomic_fmax:image-float-atomic");
}

#[test]
fn a_sixty_four_bit_atomic_joins_two_registers() {
    let op = image(Op::image_atomic_umax, v(10), v(2), s(4), s(0), 1, 0x3);
    let mut setup = regs(2, false);
    setup.push(vop1(Op::v_mov_b32, v(10), int(7)));
    setup.push(vop1(Op::v_mov_b32, v(11), int(9)));
    let t = hlsl(&pixel(&setup, op));
    has(&t, "RWTexture2D<uint64_t> decant_t1_u");
    has(
        &t,
        "InterlockedMax(decant_t1_u[uint2(uint(int(v2)), uint(int(v3)))], uint64_t((uint64_t(v11) << 32) | uint64_t(v10)), decant_t1_o);",
    );
}

#[test]
fn a_cube_with_derivatives_turns_face_derivatives_into_direction_derivatives() {
    let t = hlsl(&pixel(&regs(7, true), img(Op::image_sample_d, 3, 0x7)));
    has(
        &t,
        "SampleGrad(smp_0, decant_t1_d, ((decant_t1_f & 7u) == 0u ? float3(0.0, -(2.0 * asfloat(v3)), -(2.0 * asfloat(v2)))",
    );
    let t = hlsl(&pixel(&regs(8, true), img(Op::image_sample_d_cl, 3, 0x7)));
    has(&t, ", asfloat(v9))");
}

#[test]
fn a_gather_at_a_level_reads_the_four_texels_of_that_mip() {
    let t = hlsl(&pixel(&regs(3, true), img(Op::image_gather4_l, 1, 0x1)));
    has(
        &t,
        "uint decant_t1_m = min(uint(max(round(asfloat(v4)), 0.0)), decant_t1_n - 1u);",
    );
    has(
        &t,
        ".Load(int3(clamp(decant_t1_i + int2(0, 1), int2(0, 0), decant_t1_x), int(decant_t1_m))).x",
    );
}

/// A pixel program whose image and sampler are taken by run time heap indices: the image from
/// the resource heap by a console index out of a root constant, the sampler from the sampler
/// heap. The root layout of the container says which registers are the heaps.
fn bindless(setup: &[Insn], op: Insn) -> StageProgram {
    let at = |op: Op, dst: u8, base: u8, soffset: u8| {
        let mut insn = smem(op, s(dst), s(base), 0);
        insn.ext = decant_shader::decode::Ext::Smem {
            offset: 0,
            soffset: s(soffset),
            glc: false,
            dlc: false,
        };
        insn
    };
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(20), s(15)));
    a.push(sop1(Op::s_mov_b32, s(21), int(0)));
    a.push(sop1(Op::s_mov_b32, s(22), s(16)));
    a.push(sop1(Op::s_mov_b32, s(23), int(0)));
    a.push(sop2(Op::s_lshl_b32, s(24), s(13), int(5)));
    a.push(sop2(Op::s_lshl_b32, s(25), s(14), int(4)));
    a.push(at(Op::s_load_dwordx8, 4, 20, 24));
    a.push(at(Op::s_load_dwordx4, 0, 22, 25));
    for i in setup {
        a.push(*i);
    }
    a.push(op);
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(10), v(11)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(12), v(13)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let slot = |sgpr, kind| UserDataSlot { sgpr, kind };
    StageProgram {
        inputs: vec![element("SV_Position", 0, 15, 0, 1)],
        outputs: vec![element("SV_Target", 0, 15, 0, 64)],
        user_data: vec![
            slot(
                13,
                UserDataKind::Constants {
                    register: 500,
                    count: 2,
                },
            ),
            slot(15, UserDataKind::ResourceHeap),
            slot(16, UserDataKind::SamplerHeap),
        ],
        name: "bindless".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

#[test]
fn an_image_and_a_sampler_taken_by_heap_indices_are_made_at_the_use() {
    let t = hlsl(&bindless(&regs(2, true), img(Op::image_sample, 1, 0xf)));
    // The loads leave the host indices in the first descriptor registers: the console index
    // through the heap map, the sampler index as it is.
    has(
        &t,
        "ByteAddressBuffer decant_heap_map : register(t0, space15);",
    );
    has(&t, "decant_heap_host((s24 >> (5u & 31u)))");
    has(&t, "(decant_mirror[4].x + (s25 >> (4u & 31u)))");
    // The resources are made right before the sample, from those registers.
    has(
        &t,
        "Texture2D<float4> con_t_1 = ResourceDescriptorHeap[s4];",
    );
    has(&t, "SamplerState con_s_3 = SamplerDescriptorHeap[s0];");
    has(
        &t,
        "con_t_1.Sample(con_s_3, float2(asfloat(v2), asfloat(v3)))",
    );
}

#[test]
fn every_image_dimension_is_made_from_a_heap_index() {
    for (dim, coords, ty) in [
        (0u8, 1u8, "Texture1D<float4>"),
        (1, 2, "Texture2D<float4>"),
        (2, 3, "Texture3D<float4>"),
        (3, 3, "TextureCube<float4>"),
        (4, 2, "Texture1DArray<float4>"),
        (5, 3, "Texture2DArray<float4>"),
    ] {
        let t = hlsl(&bindless(
            &regs(coords, true),
            img(Op::image_sample_lz, dim, 0xf),
        ));
        has(&t, &format!("{ty} con_t_1 = ResourceDescriptorHeap[s4];"));
    }
    // A multisample image is loaded, not sampled.
    let t = hlsl(&bindless(&regs(3, false), img(Op::image_load, 6, 0xf)));
    has(
        &t,
        "Texture2DMS<float4> con_t_1 = ResourceDescriptorHeap[s4];",
    );
    // A store types the image as writable.
    let t = hlsl(&bindless(&regs(2, false), img(Op::image_store, 1, 0xf)));
    has(
        &t,
        "RWTexture2D<float4> con_u_1 = ResourceDescriptorHeap[s4];",
    );
}

#[test]
fn an_atomic_on_an_image_taken_by_a_heap_index_views_the_same_slot() {
    let mut op = image(Op::image_atomic_umax, v(10), v(2), s(4), s(0), 1, 0x1);
    op = with(op, |m| m.glc = true);
    let t = hlsl(&bindless(&regs(2, false), op));
    has(
        &t,
        "RWTexture2D<uint> decant_t1_u = ResourceDescriptorHeap[s4];",
    );
    has(
        &t,
        "InterlockedMax(decant_t1_u[uint2(uint(int(v2)), uint(int(v3)))], uint(v10), decant_t1_o);",
    );
}

#[test]
fn sampler_from_lane_indexed_code_table() {
    let marker = 0xc0de_1234;
    let mut setup = regs(2, true);
    setup.extend([
        sop1(Op::s_getpc_b64, s(16), int(0)),
        sop2(Op::s_add_u32, s(16), s(16), lit(marker)),
        vop1(Op::v_mov_b32, v(16), int(16)),
        vop2(Op::v_cndmask_b32, v(16), int(0), v(16)),
        vop2(Op::v_add_u32, v(16), s(16), v(16)),
        vop1(Op::v_mov_b32, v(17), int(0)),
        global_load(Op::global_load_dwordx4, 20, 16, 0),
    ]);
    setup.extend((0..4).map(|k| vop1(Op::v_readfirstlane_b32, s(k), v(20 + k))));
    let mut p = pixel(&setup, img(Op::image_sample_lz, 1, 0xf));
    let literal = p.code.iter().position(|word| *word == marker).unwrap();
    p.code[literal] = ((p.code.len() - (literal - 1)) * 4) as u32;
    p.code.extend(SAMPLER_WORDS);
    let mut other = SAMPLER_WORDS;
    other[0] = 0x92;
    p.code.extend(other);
    let text = hlsl(&p);
    assert!(
        text.contains("SamplerDescriptorHeap[NonUniformResourceIndex("),
        "{text}"
    );
    assert!(text.contains("s0 == 1024u"), "{text}");
}

#[test]
fn sampler_with_first_word_chosen_by_branch() {
    let mut setup = regs(2, true);
    setup.extend([
        sopc(Op::s_cmp_eq_u32, s(18), int(0)),
        sopp(Op::s_cbranch_scc1, 2),
        sop1(Op::s_mov_b32, s(0), lit(0x92)),
    ]);
    let p = pixel(&setup, img(Op::image_sample_lz, 1, 0xf));
    let text = hlsl(&p);
    assert!(
        text.contains("SamplerDescriptorHeap[NonUniformResourceIndex("),
        "{text}"
    );
    assert!(text.contains("s0 == 1024u"), "{text}");
}

fn masked_sampler(other_offset: u32) -> StageProgram {
    let mut a = Asm::new();
    let other = a.label();
    let join = a.label();
    a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(sop1(Op::s_mov_b32, s(17), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(0), s(8), 0x20));
    a.push(vop3(Op::v_cmp_lt_f32, s(14), v(0), v(1), Operand::None));
    a.push(sop1(Op::s_and_saveexec_b64, s(12), s(14)));
    a.branch(Op::s_cbranch_execz, other);
    a.push(smem(Op::s_load_dwordx4, s(20), s(16), 0));
    for word in 0..4 {
        a.push(vop1(Op::v_mov_b32, v(20 + word), s(20 + word)));
    }
    a.bind(other);
    a.push(sop2(Op::s_andn2_b64, EXEC, s(12), EXEC));
    a.branch(Op::s_cbranch_execz, join);
    a.push(smem(Op::s_load_dwordx4, s(24), s(16), other_offset));
    for word in 0..4 {
        a.push(vop1(Op::v_mov_b32, v(20 + word), s(24 + word)));
    }
    a.bind(join);
    a.push(sop1(Op::s_mov_b64, EXEC, s(12)));
    for word in 0..4 {
        a.push(vop1(Op::v_readfirstlane_b32, s(20 + word), v(20 + word)));
    }
    a.push(image(Op::image_sample_lz, v(10), v(0), s(0), s(20), 1, 0xf));
    a.push(export(0, 0xf, [10, 11, 12, 13], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        inputs: vec![element("SV_Position", 0, 15, 0, 1)],
        outputs: vec![element("SV_Target", 0, 15, 0, 64)],
        user_data: vec![
            UserDataSlot {
                sgpr: 8,
                kind: UserDataKind::Table,
            },
            UserDataSlot {
                sgpr: 16,
                kind: UserDataKind::Table,
            },
        ],
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

#[test]
fn a_sampler_survives_complementary_exec_branches() {
    let text = hlsl(&masked_sampler(0));
    has(&text, "heap_s_s16_0");
}

#[test]
fn a_sampler_with_different_slots_on_active_paths_is_not_fixed_to_one() {
    assert_eq!(refusal(&masked_sampler(16)), "resource:untraced-descriptor");
}
