//! The vector float instructions: each one lifted in a small pixel program, translated and
//! compiled with DXC, and the bit level helpers checked on the CPU against references that do
//! not share their algorithm.
//!
//! The CPU checks run Rust copies of the integer HLSL helpers (`decant_vf_h2f`, `decant_vf_f2h`,
//! `decant_vf_f2h_rtz`, the minimum family). The device comparison of the HLSL text itself follows
//! once the GPU comparer is on main.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn};
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

fn pixel(code: Vec<u32>) -> StageProgram {
    StageProgram {
        stage: Stage::Pixel,
        code,
        inputs: Vec::new(),
        outputs: vec![SigElement {
            semantic: "SV_Target".to_owned(),
            index: 0,
            system_value: 64,
            component_type: 3,
            register: 0,
            mask: 15,
            rw_mask: 0,
        }],
        resources: Vec::new(),
        binding_table: Vec::new(),
        name: "valu_float".to_owned(),
        ..StageProgram::new(Stage::Pixel, Vec::new())
    }
}

/// Translates `body` followed by an export of v2 to v5, compiles it when DXC is there, and
/// returns the HLSL.
fn run(body: &[Insn]) -> String {
    let mut a = Asm::new();
    for i in body {
        a.push(*i);
    }
    a.push(export(0, 0xf, [2, 3, 4, 5], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let p = pixel(a.finish());
    let host = CountingHost::default();
    let t = translate(
        std::slice::from_ref(&p),
        &RootSignature::default(),
        &Options::default(),
        &host,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    if let Some(dxc) = common::find_dxc() {
        translate_to_dxil(
            &[p],
            &RootSignature::default(),
            &Options::default(),
            &host,
            &dxc,
        )
        .unwrap_or_else(|r| panic!("{r}\n{}", t.stages[0].hlsl));
    } else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
    }
    t.stages[0].hlsl.clone()
}

fn refused(body: &[Insn]) -> String {
    let mut a = Asm::new();
    for i in body {
        a.push(*i);
    }
    a.push(sopp(Op::s_endpgm, 0));
    translate(
        &[pixel(a.finish())],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err()
    .key()
}

/// A two word instruction with modifiers.
fn with(mut i: Insn, abs: u8, neg: u8, opsel: u8, omod: u8, clamp: bool) -> Insn {
    i.ext = Ext::Valu {
        abs,
        neg,
        opsel,
        omod,
        clamp,
    };
    i
}

#[test]
fn every_one_operand_float_instruction_lifts_and_compiles() {
    let ops = [
        (Op::v_rcp_f32, "decant_vf_rcp"),
        (Op::v_rcp_iflag_f32, "decant_vf_rcp"),
        (Op::v_rsq_f32, "decant_vf_rsq"),
        (Op::v_sqrt_f32, "decant_vf_sqrt"),
        (Op::v_exp_f32, "decant_vf_exp"),
        (Op::v_log_f32, "decant_vf_log"),
        (Op::v_sin_f32, "decant_vf_sin"),
        (Op::v_cos_f32, "decant_vf_cos"),
        (Op::v_fract_f32, "decant_vf_fract"),
        (Op::v_trunc_f32, "decant_vf_trunc"),
        (Op::v_floor_f32, "decant_vf_floor"),
        (Op::v_ceil_f32, "decant_vf_ceil"),
        (Op::v_rndne_f32, "decant_vf_rndne"),
        (Op::v_cvt_f32_f16, "decant_vf_h2f"),
        (Op::v_cvt_f16_f32, "decant_vf_f2h"),
        (Op::v_cvt_i32_f32, "decant_vf_cvt_i32_f32"),
        (Op::v_cvt_u32_f32, "decant_vf_cvt_u32_f32"),
        (Op::v_cvt_f32_i32, "decant_vf_cvt_f32_i32"),
        (Op::v_cvt_f32_u32, "decant_vf_cvt_f32_u32"),
        (Op::v_rcp_f16, "decant_vf_rcp"),
        (Op::v_rsq_f16, "decant_vf_rsq"),
        (Op::v_sqrt_f16, "decant_vf_sqrt"),
        (Op::v_exp_f16, "decant_vf_exp"),
        (Op::v_sin_f16, "decant_vf_sin"),
        (Op::v_cos_f16, "decant_vf_cos"),
        (Op::v_trunc_f16, "decant_vf_trunc"),
    ];
    for (op, helper) in ops {
        let text = run(&[vop1(op, v(2), v(1))]);
        assert!(
            text.contains(&format!("{helper}(")),
            "{}\n{text}",
            op.name()
        );
    }
}

#[test]
fn every_two_and_three_operand_float_instruction_lifts_and_compiles() {
    let two = [
        Op::v_add_f32,
        Op::v_sub_f32,
        Op::v_subrev_f32,
        Op::v_mul_f32,
        Op::v_mul_legacy_f32,
        Op::v_min_f32,
        Op::v_max_f32,
        Op::v_fmac_f32,
        Op::v_mac_f32,
        Op::v_cvt_pkrtz_f16_f32,
        Op::v_add_f16,
        Op::v_sub_f16,
        Op::v_mul_f16,
        Op::v_fmac_f16,
        Op::v_min_f16,
        Op::v_max_f16,
    ];
    for op in two {
        run(&[vop2(op, v(2), v(0), v(1))]);
    }
    let three = [
        Op::v_fma_f32,
        Op::v_fma_legacy_f32,
        Op::v_mad_f32,
        Op::v_min3_f32,
        Op::v_max3_f32,
        Op::v_med3_f32,
        Op::v_cubeid_f32,
        Op::v_cubesc_f32,
        Op::v_cubetc_f32,
        Op::v_cubema_f32,
        Op::v_fma_f16,
        Op::v_med3_f16,
        Op::v_pack_b32_f16,
        Op::v_cvt_pk_u8_f32,
    ];
    for op in three {
        run(&[vop3(op, v(2), v(0), v(1), v(3))]);
    }
}

#[test]
fn the_constant_multiply_add_forms_take_the_literal_as_their_third_operand() {
    // madak: v0 * v1 + K, madmk: v0 * K + v1.
    let k = 3.0f32.to_bits();
    let ak = run(&[vop2(Op::v_madak_f32, v(2), v(0), v(1)).tap(k)]);
    assert!(ak.contains(&format!("decant_vf_mad(v0, v1, {k}u)")), "{ak}");
    let mk = run(&[vop2(Op::v_madmk_f32, v(2), v(0), v(1)).tap(k)]);
    assert!(mk.contains(&format!("decant_vf_mad(v0, {k}u, v1)")), "{mk}");
    let fk = run(&[vop2(Op::v_fmaak_f32, v(2), v(0), v(1)).tap(k)]);
    assert!(fk.contains(&format!("decant_vf_fma(v0, v1, {k}u)")), "{fk}");
}

trait Tap {
    fn tap(self, literal: u32) -> Insn;
}

impl Tap for Insn {
    /// Sets the literal of a `mk` or `ak` form.
    fn tap(mut self, literal: u32) -> Insn {
        self.src[2] = lit(literal);
        self
    }
}

#[test]
fn the_accumulating_forms_read_the_destination() {
    let text = run(&[vop2(Op::v_fmac_f32, v(2), v(0), v(1))]);
    assert!(text.contains("decant_vf_fma(v0, v1, v2)"), "{text}");
}

#[test]
fn abs_and_neg_act_on_the_sign_bit_then_omod_then_clamp() {
    let i = with(
        vop3(Op::v_add_f32, v(2), v(0), v(1), Operand::None),
        0b01,
        0b10,
        0,
        1,
        true,
    );
    let text = run(&[i]);
    assert!(
        text.contains(
            "decant_vf_sat(asuint(asfloat(asuint(asfloat((v0 & 2147483647u)) + asfloat((v1 ^ 2147483648u)))) * asfloat(1073741824u)))"
        ),
        "{text}"
    );
}

#[test]
fn a_16_bit_result_keeps_the_other_half_and_opsel_picks_halves() {
    // Plain form: low half written, high half kept.
    let text = run(&[vop2(Op::v_add_f16, v(2), v(0), v(1))]);
    assert!(text.contains("((v2 & 4294901760u) | decant_vf_f2h("), "{text}");
    // v_fma_f16 with op_sel: source 0 high, destination high.
    let i = with(
        vop3(Op::v_fma_f16, v(2), v(0), v(1), v(3)),
        0,
        0,
        0b1001,
        0,
        false,
    );
    let text = run(&[i]);
    assert!(
        text.contains("((v2 & 65535u) | (decant_vf_f2h(decant_vf_fma(decant_vf_h2f((v0 >> (16u & 31u)))"),
        "{text}"
    );
    assert!(text.contains("decant_vf_h2f((v0 >> (16u & 31u)))"), "{text}");
    // A 16 bit instruction that has no op_sel on GFX10 refuses it.
    let i = with(
        vop3(Op::v_add_f16, v(2), v(0), v(1), Operand::None),
        0,
        0,
        0b0001,
        0,
        false,
    );
    assert_eq!(refused(&[i]), "form:v_add_f16:opsel");
}

#[test]
fn the_mixed_multiply_add_reads_16_bit_sources_where_op_sel_hi_says_so() {
    // Source 1 is the high half of v1 as a 16 bit float, sources 0 and 2 are 32 bit.
    let mut i = vop3(Op::v_fma_mix_f32, v(2), v(0), v(1), v(3));
    i.form = Form::Plain;
    i.ext = Ext::Valu {
        abs: 0b001,
        neg: 0,
        opsel: 0b0010,
        omod: 0b10,
        clamp: false,
    };
    let text = run(&[i]);
    assert!(
        text.contains("decant_vf_fma((v0 & 2147483647u), decant_vf_h2f((v1 >> (16u & 31u))), v3)"),
        "{text}"
    );
    let mut lo = i;
    lo.op = Op::v_fma_mixlo_f16;
    let text = run(&[lo]);
    assert!(
        text.contains("((v2 & 4294901760u) | decant_vf_f2h(decant_vf_fma("),
        "{text}"
    );
}

use decant_shader::decode::Operand;

// CPU copies of the integer HLSL helpers, checked against independent references.

fn h2f(h: u32) -> u32 {
    let s = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    if e == 31 {
        return s | 0x7f80_0000 | (m << 13);
    }
    if e != 0 {
        return s | ((e + 112) << 23) | (m << 13);
    }
    if m == 0 {
        return s;
    }
    let sh = 10 - (31 - m.leading_zeros());
    s | ((113 - sh) << 23) | (((m << sh) & 0x3ff) << 13)
}

fn f2h(f: u32) -> u32 {
    let s = (f >> 16) & 0x8000;
    let a = f & 0x7fff_ffff;
    if a > 0x7f80_0000 {
        return s | 0x7e00 | ((a >> 13) & 0x3ff);
    }
    if a >= 0x477f_f000 {
        return s | 0x7c00;
    }
    if a >= 0x3880_0000 {
        let r = a - 0x3800_0000;
        return s | ((r + 0xfff + ((r >> 13) & 1)) >> 13);
    }
    if a <= 0x3300_0000 {
        return s;
    }
    let m = (a & 0x7f_ffff) | 0x80_0000;
    let sh = 126 - (a >> 23);
    let q = m >> sh;
    let rem = m & ((1 << sh) - 1);
    let half = 1 << (sh - 1);
    s | (q + u32::from(rem > half || (rem == half && q & 1 != 0)))
}

fn f2h_rtz(f: u32) -> u32 {
    let s = (f >> 16) & 0x8000;
    let m = f & 0x7f_ffff;
    let mut e = ((f >> 23) & 0xff) as i32;
    if e == 255 {
        return s | if m != 0 { 0x7e00 | (m >> 13) } else { 0x7c00 };
    }
    e -= 112;
    if e >= 31 {
        return s | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return s;
        }
        return s | ((m | 0x80_0000) >> (14 - e) as u32);
    }
    s | ((e as u32) << 10) | (m >> 13)
}

/// The value of a 16 bit pattern, by the definition of the format.
fn half_value(h: u32) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = f64::from(h & 0x3ff);
    sign * if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

#[test]
fn h2f_is_exact_for_every_16_bit_pattern() {
    for h in 0..=0xffffu32 {
        let f = h2f(h);
        if (h >> 10) & 0x1f == 31 {
            assert_eq!(f & 0x7f80_0000, 0x7f80_0000, "{h:#x}");
            assert_eq!(f & 0x7f_ffff, (h & 0x3ff) << 13, "{h:#x}");
            continue;
        }
        assert_eq!(f64::from(f32::from_bits(f)), half_value(h), "{h:#x}");
        assert_eq!(f >> 31, h >> 15, "{h:#x}");
    }
}

/// Nearest even rounding to 16 bit by search over the finite patterns.
fn f2h_reference(f: f32) -> u32 {
    let s = if f.is_sign_negative() { 0x8000 } else { 0 };
    let x = f64::from(f.abs());
    // 65520 and above round to infinity.
    if x >= 65520.0 {
        return s | 0x7c00;
    }
    let mut lo = 0u32;
    let mut hi = 0x7bffu32;
    // Largest pattern not above x.
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if half_value(mid) <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = lo;
    if half_value(below) == x || below == 0x7bff {
        return s | below;
    }
    let above = below + 1;
    let (db, da) = (x - half_value(below), half_value(above) - x);
    s | if db < da || (db == da && below & 1 == 0) {
        below
    } else {
        above
    }
}

fn sample_floats() -> impl Iterator<Item = u32> {
    // Every exponent with mantissas around the rounding points, plus a stride over everything.
    let exps = (0..=255u32).flat_map(|e| {
        [
            0u32, 1, 0xfff, 0x1000, 0x1001, 0x1fff, 0x2000, 0x2001, 0x3fff, 0x7f_e000, 0x7f_f000,
            0x7f_f001, 0x7f_ffff, 0x40_0000, 0x40_1000,
        ]
        .into_iter()
        .map(move |m| (e << 23) | m)
    });
    exps.chain((0..0x8000_0000u32).step_by(9973))
        .flat_map(|u| [u, u | 0x8000_0000])
}

#[test]
fn f2h_rounds_to_nearest_even_like_the_reference() {
    for u in sample_floats() {
        let f = f32::from_bits(u);
        if f.is_nan() {
            let h = f2h(u);
            assert_eq!(h & 0x7e00, 0x7e00, "{u:#x}");
            continue;
        }
        assert_eq!(f2h(u), f2h_reference(f), "{u:#x} {f:e}");
    }
}

/// Toward zero: the largest pattern whose magnitude is not above the input.
fn f2h_rtz_reference(f: f32) -> u32 {
    let s = if f.is_sign_negative() { 0x8000 } else { 0 };
    if f.is_infinite() {
        return s | 0x7c00;
    }
    let x = f64::from(f.abs());
    let (mut lo, mut hi) = (0u32, 0x7bffu32);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if half_value(mid) <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    s | lo
}

#[test]
fn f2h_rtz_truncates_like_the_reference() {
    for u in sample_floats() {
        let f = f32::from_bits(u);
        if f.is_nan() {
            assert_eq!(f2h_rtz(u) & 0x7e00, 0x7e00, "{u:#x}");
            continue;
        }
        assert_eq!(f2h_rtz(u), f2h_rtz_reference(f), "{u:#x} {f:e}");
    }
}

/// A denormal pattern as the zero of its sign.
fn ftz(a: u32) -> u32 {
    if a & 0x7f80_0000 == 0 {
        a & 0x8000_0000
    } else {
        a
    }
}

fn nan(a: u32) -> bool {
    a & 0x7fff_ffff > 0x7f80_0000
}

fn fmin(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    if nan(a) {
        return if nan(b) { a } else { b };
    }
    if nan(b) {
        return a;
    }
    if (a | b) & 0x7fff_ffff == 0 {
        return a | b;
    }
    if f32::from_bits(a) < f32::from_bits(b) {
        a
    } else {
        b
    }
}

fn fmax(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    if nan(a) {
        return if nan(b) { a } else { b };
    }
    if nan(b) {
        return a;
    }
    if (a | b) & 0x7fff_ffff == 0 {
        return a & b;
    }
    if f32::from_bits(a) > f32::from_bits(b) {
        a
    } else {
        b
    }
}

fn med3(a: u32, b: u32, c: u32) -> u32 {
    if nan(a) {
        return fmin(b, c);
    }
    if nan(b) {
        return fmin(a, c);
    }
    if nan(c) {
        return fmin(a, b);
    }
    fmax(fmin(a, b), fmin(fmax(a, b), c))
}

#[test]
fn min_max_and_median_follow_the_gfx10_rules() {
    let (pz, nz, one, two, qnan) = (
        0u32,
        0x8000_0000,
        1f32.to_bits(),
        2f32.to_bits(),
        0x7fc0_0000,
    );
    // Not a number loses against a number, either side.
    assert_eq!(fmin(qnan, one), one);
    assert_eq!(fmax(one, qnan), one);
    // Negative zero is below positive zero, either order.
    assert_eq!(fmin(pz, nz), nz);
    assert_eq!(fmin(nz, pz), nz);
    assert_eq!(fmax(pz, nz), pz);
    assert_eq!(fmax(nz, pz), pz);
    // The median is the middle value of every order.
    let vals = [one, two, nz, (-3f32).to_bits()];
    for &a in &vals {
        for &b in &vals {
            for &c in &vals {
                let mut v = [a, b, c].map(f32::from_bits);
                v.sort_by(|x, y| x.total_cmp(y));
                assert_eq!(med3(a, b, c), v[1].to_bits(), "{a:#x} {b:#x} {c:#x}");
            }
        }
    }
    // A denormal counts as the zero of its sign, also in the result.
    assert_eq!(fmin(pz, 0x8000_0001), nz);
    assert_eq!(fmax(0x0000_0001, nz), pz);
    // With a not a number the median is the minimum of the other two (LLVM, ieee=0).
    assert_eq!(med3(qnan, two, one), one);
    assert_eq!(med3(two, qnan, one), one);
    assert_eq!(med3(two, one, qnan), one);
}

#[test]
fn every_compare_type_lifts_and_compiles() {
    let ops = [
        Op::v_cmp_lt_f16,
        Op::v_cmp_neq_f16,
        Op::v_cmp_nlt_f16,
        Op::v_cmp_u_f32,
        Op::v_cmp_nlg_f32,
        Op::v_cmp_class_f32,
        Op::v_cmp_class_f16,
        Op::v_cmp_eq_u64,
        Op::v_cmp_lg_u64,
        Op::v_cmp_lt_u64,
        Op::v_cmp_gt_i64,
        Op::v_cmp_eq_u16,
        Op::v_cmp_lt_i16,
    ];
    for op in ops {
        // The mask selects between two registers, so the compare is used.
        let text = run(&[
            vopc(op, v(0), v(2)),
            vop2(Op::v_cndmask_b32, v(2), v(0), v(1)),
        ]);
        assert!(text.contains("vcc = "), "{}\n{text}", op.name());
    }
    for op in [Op::v_cmpx_lt_f16, Op::v_cmpx_lg_u64, Op::v_cmpx_class_f32] {
        let text = run(&[vopc(op, v(0), v(2))]);
        assert!(text.contains("exec = (exec && "), "{}\n{text}", op.name());
    }
}

#[test]
fn a_64_bit_compare_orders_by_the_high_half_first() {
    let text = run(&[
        vopc(Op::v_cmp_lt_u64, v(0), v(2)),
        vop2(Op::v_cndmask_b32, v(2), v(0), v(1)),
    ]);
    assert!(
        text.contains("((v1 < v3) || ((v1 == v3) && (v0 < v2)))"),
        "{text}"
    );
}

#[test]
fn a_two_word_cmpx_writes_exec_whatever_its_destination_field_holds() {
    let i = vop3(Op::v_cmpx_gt_f32, s(20), v(0), v(1), Operand::None);
    let text = run(&[i]);
    assert!(
        text.contains("exec = (exec && (asfloat(v1) < asfloat(v0)))"),
        "{text}"
    );
}

#[test]
fn the_class_compare_reads_the_bit_pattern() {
    let i = with(
        vop3(Op::v_cmp_class_f32, VCC, v(0), lit(0x3fc), Operand::None),
        1,
        0,
        0,
        0,
        false,
    );
    let text = run(&[i, vop2(Op::v_cndmask_b32, v(2), v(0), v(1))]);
    assert!(
        text.contains("decant_vc_class32((v0 & 2147483647u), 1020u)"),
        "{text}"
    );
}

/// The classes of the 32 bit patterns, as the helper computes them.
fn class32(a: u32) -> u32 {
    let (e, f, s) = ((a >> 23) & 0xff, a & 0x7f_ffff, a >> 31 != 0);
    if e == 0xff {
        if f == 0 {
            if s { 4 } else { 512 }
        } else if f & 0x40_0000 != 0 {
            2
        } else {
            1
        }
    } else if e == 0 {
        if f == 0 {
            if s { 32 } else { 64 }
        } else if s {
            16
        } else {
            128
        }
    } else if s {
        8
    } else {
        256
    }
}

#[test]
fn the_class_bits_match_the_float_categories() {
    use std::num::FpCategory::*;
    for u in sample_floats() {
        let f = f32::from_bits(u);
        let neg = f.is_sign_negative();
        let want = match (f.classify(), neg) {
            (Nan, _) => {
                if u & 0x40_0000 != 0 {
                    2
                } else {
                    1
                }
            }
            (Infinite, true) => 4,
            (Normal, true) => 8,
            (Subnormal, true) => 16,
            (Zero, true) => 32,
            (Zero, false) => 64,
            (Subnormal, false) => 128,
            (Normal, false) => 256,
            (Infinite, false) => 512,
        };
        assert_eq!(class32(u), want, "{u:#x}");
    }
}

/// The instruction in the sub dword encoding with this word.
fn sdwa(mut i: Insn, word: u32) -> Insn {
    // The first source register sits in the low byte of the word.
    let src0 = match i.src[0] {
        Operand::Reg(decant_shader::decode::Reg::V(n)) => u32::from(n),
        _ => panic!("the sub dword tests use a vector register first"),
    };
    i.form = Form::Sdwa;
    i.ext = Ext::Sdwa(word | src0);
    i
}

/// Sub dword word fields: destination select and unused rule, source selects.
fn sdwa_word(dst_sel: u32, unused: u32, src0_sel: u32, src1_sel: u32) -> u32 {
    (dst_sel << 8) | (unused << 11) | (src0_sel << 16) | (src1_sel << 24)
}

#[test]
fn the_sub_dword_form_picks_the_half_a_16_bit_source_reads() {
    let i = sdwa(vop1(Op::v_cvt_f32_f16, v(2), v(1)), sdwa_word(6, 0, 5, 6));
    let text = run(&[i]);
    assert!(
        text.contains("decant_vf_h2f(((v1 >> (16u & 31u)) & 65535u))"),
        "{text}"
    );
}

#[test]
fn the_sub_dword_form_writes_a_16_bit_result_where_the_destination_select_says() {
    // High word written, the low word preserved.
    let i = sdwa(vop2(Op::v_add_f16, v(2), v(0), v(1)), sdwa_word(5, 2, 4, 5));
    let text = run(&[i]);
    assert!(text.contains("((v2 & 65535u) | ("), "{text}");
    // Clamp from the sub dword word on a 32 bit result.
    let i = sdwa(
        vop2(Op::v_add_f32, v(2), v(0), v(1)),
        sdwa_word(6, 0, 6, 6) | 1 << 13,
    );
    assert!(run(&[i]).contains("decant_vf_sat("));
}

#[test]
fn a_sub_dword_compare_reads_the_selected_byte() {
    let i = sdwa(vopc(Op::v_cmp_eq_u32, v(0), v(1)), (1 << 16) | (6 << 24));
    let text = run(&[i, vop2(Op::v_cndmask_b32, v(2), v(0), v(1))]);
    assert!(
        text.contains("(((v0 >> (8u & 31u)) & 255u) == v1)"),
        "{text}"
    );
}

#[test]
fn the_packed_16_bit_instructions_compute_each_half_from_its_own_selects() {
    // op_sel 0 (low halves for the low result), op_sel_hi all set (high halves for the high
    // result), NEG of source 1 on the low result only.
    let mut i = vop3(Op::v_pk_add_f16, v(2), v(0), v(1), Operand::None);
    i.form = Form::Plain;
    i.ext = Ext::Valu {
        abs: 0,
        neg: 0b10,
        opsel: 0b1000,
        omod: 0b11,
        clamp: false,
    };
    let text = run(&[i]);
    assert!(
        text.contains("decant_vf_h2f(((v1 & 65535u) ^ 32768u))")
            && text.contains("decant_vf_h2f((v1 >> (16u & 31u)))"),
        "{text}"
    );
    for op in [
        Op::v_pk_mul_f16,
        Op::v_pk_fma_f16,
        Op::v_pk_min_f16,
        Op::v_pk_max_f16,
    ] {
        let mut i = vop3(op, v(2), v(0), v(1), v(3));
        i.form = Form::Plain;
        run(&[i]);
    }
}

#[test]
fn the_64_bit_float_instructions_work_on_register_pairs() {
    let text = run(&[vop3(Op::v_add_f64_e64, v(2), v(0), v(2), Operand::None)]);
    // The result goes through temporaries: v2 is a source and the low destination.
    assert!(text.contains("decant_vf_add_f64(v0, v1, v2, v3)"), "{text}");
    run(&[vop1(Op::v_cvt_f32_f64, v(2), v(0))]);
    run(&[vop1(Op::v_cvt_f64_f32, v(2), v(0))]);
    run(&[vop3(Op::v_fma_f64, v(2), v(0), float(2.0), v(4))]);
}

#[test]
fn a_float_data_parallel_form_takes_the_given_first_source_and_its_modifiers() {
    use decant_shader::exec::Masks;
    use decant_shader::ir::{Expr, Stmt};
    use decant_shader::lift::Lifter;
    let mut i = vop2(Op::v_min_f32, v(0), v(1), v(2));
    i.form = Form::Dpp16;
    // NEG of source 0.
    i.ext = Ext::Dpp(0x140 | 1 << 20);
    let masks = Masks::default();
    let mut cx = Lifter::new(Stage::Pixel, &masks);
    assert!(decant_shader::lift::valu_float::lift_with_src0(&mut cx, &i, Expr::U32(5)).unwrap());
    let stmts = cx.take();
    let Stmt::Set { value, .. } = &stmts[0] else {
        panic!("{stmts:?}")
    };
    assert_eq!(
        format!("{value:?}").matches("2147483648").count(),
        1,
        "{value:?}"
    );
}
