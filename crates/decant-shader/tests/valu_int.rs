//! Vector integer instructions: the lifted statements are evaluated against a Rust reference,
//! and a program that uses all of them is compiled with DXC when it is there.
//!
//! No game data is involved.

mod common;

use std::collections::HashMap;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Form, Insn, Operand};
use decant_shader::exec::Masks;
use decant_shader::ir::{Bin, Expr, Stmt, Un, Var};
use decant_shader::lift::{Lifter, lift};
use decant_shader::tables::Op;
use decant_shader::translate_to_dxil;

type State = HashMap<Var, u32>;

fn eval(e: &Expr, st: &State) -> u32 {
    let b = |x: bool| u32::from(x);
    match e {
        Expr::U32(v) => *v,
        Expr::Bool(v) => b(*v),
        Expr::Var(v) => *st.get(v).unwrap_or(&0),
        Expr::Un(op, a) => {
            let a = eval(a, st);
            match op {
                Un::Not => !a,
                Un::LNot => b(a == 0),
                Un::Neg => a.wrapping_neg(),
                Un::FNeg => a ^ 0x8000_0000,
                Un::FAbs => a & 0x7fff_ffff,
                Un::BoolToU32 => a,
                Un::U32ToBool => b(a != 0),
                _ => panic!("{op:?}"),
            }
        }
        Expr::Bin(op, a, c) => {
            let (a, c) = (eval(a, st), eval(c, st));
            match op {
                Bin::Add => a.wrapping_add(c),
                Bin::Sub => a.wrapping_sub(c),
                Bin::Mul => a.wrapping_mul(c),
                Bin::And => a & c,
                Bin::Or => a | c,
                Bin::Xor => a ^ c,
                Bin::Shl => a << (c & 31),
                Bin::LShr => a >> (c & 31),
                Bin::AShr => ((a as i32) >> (c & 31)) as u32,
                Bin::Eq => b(a == c),
                Bin::Ne => b(a != c),
                Bin::ULt => b(a < c),
                Bin::ULe => b(a <= c),
                Bin::LAnd => b(a != 0 && c != 0),
                Bin::LOr => b(a != 0 || c != 0),
                Bin::FMul => (f32::from_bits(a) * f32::from_bits(c)).to_bits(),
                _ => panic!("{op:?}"),
            }
        }
        Expr::Select(c, a, d) => {
            if eval(c, st) != 0 {
                eval(a, st)
            } else {
                eval(d, st)
            }
        }
        Expr::Call(h, args) => {
            let a: Vec<u32> = args.iter().map(|x| eval(x, st)).collect();
            reference(h.name, &a)
        }
        _ => panic!("{e:?}"),
    }
}

fn perm_byte(a: u32, b: u32, s: u32) -> u32 {
    let wide = (u64::from(a) << 32) | u64::from(b);
    let sign = |v: u32, bit: u32| if v & (1 << bit) != 0 { 0xff } else { 0 };
    match s {
        0..=7 => ((wide >> (8 * s)) & 0xff) as u32,
        8 => sign(b, 15),
        9 => sign(b, 31),
        10 => sign(a, 15),
        11 => sign(a, 31),
        12 => 0,
        _ => 0xff,
    }
}

/// The reference of every helper of the file, written with 64 bit arithmetic.
fn reference(name: &str, a: &[u32]) -> u32 {
    let (x, y, z) = (a[0], *a.get(1).unwrap_or(&0), *a.get(2).unwrap_or(&0));
    let (sx, sy, sz) = (x as i32, y as i32, z as i32);
    match name {
        "decant_vi_mul_hi_u32" => ((u64::from(x) * u64::from(y)) >> 32) as u32,
        "decant_vi_mul_hi_i32" => ((i64::from(sx) * i64::from(sy)) >> 32) as u32,
        "decant_vi_min_i32" => sx.min(sy) as u32,
        "decant_vi_max_i32" => sx.max(sy) as u32,
        "decant_vi_min_u32" => x.min(y),
        "decant_vi_max_u32" => x.max(y),
        "decant_vi_min3_i32" => sx.min(sy).min(sz) as u32,
        "decant_vi_max3_i32" => sx.max(sy).max(sz) as u32,
        "decant_vi_min3_u32" => x.min(y).min(z),
        "decant_vi_max3_u32" => x.max(y).max(z),
        "decant_vi_med3_i32" => {
            let mut v = [sx, sy, sz];
            v.sort();
            v[1] as u32
        }
        "decant_vi_med3_u32" => {
            let mut v = [x, y, z];
            v.sort();
            v[1]
        }
        "decant_vi_bfe_u32" => {
            let w = z & 31;
            if w == 0 {
                0
            } else {
                (x >> (y & 31)) & ((1u32 << w) - 1)
            }
        }
        "decant_vi_bfe_i32" => {
            let w = z & 31;
            if w == 0 {
                0
            } else {
                let v = sx >> (y & 31);
                ((v << (32 - w)) >> (32 - w)) as u32
            }
        }
        "decant_vi_bfi" => (x & y) | (!x & z),
        "decant_vi_bfm" => ((1u32 << (x & 31)) - 1) << (y & 31),
        "decant_vi_bcnt" => x.count_ones().wrapping_add(y),
        "decant_vi_bfrev" => x.reverse_bits(),
        "decant_vi_ffbh_u32" => {
            if x == 0 {
                !0
            } else {
                x.leading_zeros()
            }
        }
        "decant_vi_ffbh_i32" => {
            if x == 0 || x == !0 {
                !0
            } else if sx < 0 {
                (!x).leading_zeros()
            } else {
                x.leading_zeros()
            }
        }
        "decant_vi_ffbl" => {
            if x == 0 {
                !0
            } else {
                x.trailing_zeros()
            }
        }
        "decant_vi_alignbit" => {
            ((((u64::from(x) << 32) | u64::from(y)) >> (z & 31)) & 0xffff_ffff) as u32
        }
        "decant_vi_alignbyte" => {
            ((((u64::from(x) << 32) | u64::from(y)) >> (8 * (z & 3))) & 0xffff_ffff) as u32
        }
        "decant_vi_perm" => (0..4).fold(0, |r, i| {
            r | (perm_byte(x, y, (z >> (8 * i)) & 0xff) << (8 * i))
        }),
        "decant_vi_sad_u32" => x.abs_diff(y).wrapping_add(z),
        "decant_vi_sad_u8" => (0..4).fold(z, |r, i| {
            r.wrapping_add(((x >> (8 * i)) & 0xff).abs_diff((y >> (8 * i)) & 0xff))
        }),
        "decant_vi_lerp_u8" => (0..4).fold(0, |r, i| {
            let s = ((x >> (8 * i)) & 0xff) + ((y >> (8 * i)) & 0xff) + ((z >> (8 * i)) & 1);
            r | (((s >> 1) & 0xff) << (8 * i))
        }),
        // The integer to float conversions are float rules, they meet the integer sub dword
        // selects here.
        "decant_vf_cvt_f32_u32" => (x as f32).to_bits(),
        "decant_vf_cvt_f32_i32" => (sx as f32).to_bits(),
        _ => panic!("no reference for {name}"),
    }
}

/// Lifts `insn` and runs the statements on `st`.
fn exec(insn: &Insn, mut st: State) -> State {
    let masks = Masks::default();
    let mut cx = Lifter::new(Stage::Pixel, &masks);
    lift(&mut cx, insn).unwrap_or_else(|r| panic!("{insn}: {r}"));
    st.insert(Var::Exec, 1);
    for s in cx.take() {
        let Stmt::Set { dst, value, .. } = s else {
            panic!("unexpected {s:?}")
        };
        let v = eval(&value, &st);
        st.insert(dst, v);
    }
    st
}

/// Runs one instruction on a state with `v1 v2 v3` set.
fn run(insn: &Insn, vals: [u32; 3]) -> State {
    let st = vals
        .into_iter()
        .enumerate()
        .map(|(i, v)| (Var::V(i as u8 + 1), v))
        .collect();
    exec(insn, st)
}

const SAMPLES: [u32; 9] = [
    0,
    1,
    7,
    0x80,
    0xffff,
    0x1234_5678,
    0x8000_0000,
    0xdead_beef,
    0xffff_ffff,
];
const THIRDS: [u32; 7] = [0, 3, 0x0504_0302, 0x0c0b_0a09, 31, 33, 0xffff_ffff];

fn each3(mut f: impl FnMut([u32; 3])) {
    for &a in &SAMPLES {
        for &b in &SAMPLES {
            for &c in &THIRDS {
                f([a, b, c]);
            }
        }
    }
}

#[test]
fn three_operand_forms_match_the_reference() {
    type F = fn(u32, u32, u32) -> u32;
    let table: &[(Op, F)] = &[
        (Op::v_bfe_u32, |a, b, c| {
            reference("decant_vi_bfe_u32", &[a, b, c])
        }),
        (Op::v_bfe_i32, |a, b, c| {
            reference("decant_vi_bfe_i32", &[a, b, c])
        }),
        (Op::v_bfi_b32, |a, b, c| (a & b) | (!a & c)),
        (Op::v_alignbit_b32, |a, b, c| {
            reference("decant_vi_alignbit", &[a, b, c])
        }),
        (Op::v_alignbyte_b32, |a, b, c| {
            reference("decant_vi_alignbyte", &[a, b, c])
        }),
        (Op::v_perm_b32, |a, b, c| {
            reference("decant_vi_perm", &[a, b, c])
        }),
        (Op::v_min3_i32, |a, b, c| {
            reference("decant_vi_min3_i32", &[a, b, c])
        }),
        (Op::v_max3_u32, |a, b, c| {
            reference("decant_vi_max3_u32", &[a, b, c])
        }),
        (Op::v_med3_i32, |a, b, c| {
            reference("decant_vi_med3_i32", &[a, b, c])
        }),
        (Op::v_med3_u32, |a, b, c| {
            reference("decant_vi_med3_u32", &[a, b, c])
        }),
        (Op::v_sad_u8, |a, b, c| {
            reference("decant_vi_sad_u8", &[a, b, c])
        }),
        (Op::v_sad_u32, |a, b, c| {
            reference("decant_vi_sad_u32", &[a, b, c])
        }),
        (Op::v_lerp_u8, |a, b, c| {
            reference("decant_vi_lerp_u8", &[a, b, c])
        }),
        (Op::v_add3_u32, |a, b, c| a.wrapping_add(b).wrapping_add(c)),
        (Op::v_or3_b32, |a, b, c| a | b | c),
        (Op::v_xor3_b32, |a, b, c| a ^ b ^ c),
        (Op::v_and_or_b32, |a, b, c| (a & b) | c),
        (Op::v_lshl_or_b32, |a, b, c| (a << (b & 31)) | c),
        (Op::v_lshl_add_u32, |a, b, c| {
            (a << (b & 31)).wrapping_add(c)
        }),
        (Op::v_add_lshl_u32, |a, b, c| a.wrapping_add(b) << (c & 31)),
        (Op::v_xad_u32, |a, b, c| (a ^ b).wrapping_add(c)),
        (Op::v_mad_u32_u24, |a, b, c| {
            (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff).wrapping_add(c)
        }),
        (Op::v_mad_i32_i24, |a, b, c| {
            let s = |x: u32| ((x << 8) as i32) >> 8;
            s(a).wrapping_mul(s(b)).wrapping_add(c as i32) as u32
        }),
    ];
    for (op, f) in table {
        let insn = vop3(*op, v(0), v(1), v(2), v(3));
        each3(|[a, b, c]| {
            let st = run(&insn, [a, b, c]);
            assert_eq!(
                st[&Var::V(0)],
                f(a, b, c),
                "{insn} with {a:#x} {b:#x} {c:#x}"
            );
        });
    }
}

#[test]
fn two_operand_forms_match_the_reference() {
    type F = fn(u32, u32) -> u32;
    let table: &[(Op, F)] = &[
        (Op::v_add_u32, |a, b| a.wrapping_add(b)),
        (Op::v_sub_u32, |a, b| a.wrapping_sub(b)),
        (Op::v_subrev_u32, |a, b| b.wrapping_sub(a)),
        (Op::v_mul_lo_u32, |a, b| a.wrapping_mul(b)),
        (Op::v_mul_hi_u32, |a, b| {
            reference("decant_vi_mul_hi_u32", &[a, b])
        }),
        (Op::v_mul_hi_i32, |a, b| {
            reference("decant_vi_mul_hi_i32", &[a, b])
        }),
        (Op::v_mul_u32_u24, |a, b| {
            (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff)
        }),
        (Op::v_mul_hi_u32_u24, |a, b| {
            ((u64::from(a & 0xff_ffff) * u64::from(b & 0xff_ffff)) >> 32) as u32
        }),
        (Op::v_mul_i32_i24, |a, b| {
            let s = |x: u32| ((x << 8) as i32) >> 8;
            s(a).wrapping_mul(s(b)) as u32
        }),
        (Op::v_mul_hi_i32_i24, |a, b| {
            let s = |x: u32| i64::from(((x << 8) as i32) >> 8);
            ((s(a) * s(b)) >> 32) as u32
        }),
        (Op::v_and_b32, |a, b| a & b),
        (Op::v_or_b32, |a, b| a | b),
        (Op::v_xor_b32, |a, b| a ^ b),
        (Op::v_xnor_b32, |a, b| !(a ^ b)),
        (Op::v_lshlrev_b32, |a, b| b << (a & 31)),
        (Op::v_lshrrev_b32, |a, b| b >> (a & 31)),
        (Op::v_ashrrev_i32, |a, b| ((b as i32) >> (a & 31)) as u32),
        (Op::v_min_i32, |a, b| reference("decant_vi_min_i32", &[a, b])),
        (Op::v_max_i32, |a, b| reference("decant_vi_max_i32", &[a, b])),
        (Op::v_min_u32, |a, b| a.min(b)),
        (Op::v_max_u32, |a, b| a.max(b)),
        (Op::v_bfm_b32, |a, b| reference("decant_vi_bfm", &[a, b])),
        (Op::v_bcnt_u32_b32, |a, b| reference("decant_vi_bcnt", &[a, b])),
    ];
    for (op, f) in table {
        let insn = if op.info().format == decant_shader::tables::Format::Vop3 {
            vop3(*op, v(0), v(1), v(2), Operand::None)
        } else {
            vop2(*op, v(0), v(1), v(2))
        };
        each3(|[a, b, _]| {
            let st = run(&insn, [a, b, 0]);
            assert_eq!(st[&Var::V(0)], f(a, b), "{insn} with {a:#x} {b:#x}");
        });
    }
}

#[test]
fn one_operand_forms_match_the_reference() {
    for (op, name) in [
        (Op::v_bfrev_b32, "decant_vi_bfrev"),
        (Op::v_ffbh_u32, "decant_vi_ffbh_u32"),
        (Op::v_ffbh_i32, "decant_vi_ffbh_i32"),
        (Op::v_ffbl_b32, "decant_vi_ffbl"),
    ] {
        let insn = vop1(op, v(0), v(1));
        for &a in &SAMPLES {
            let st = run(&insn, [a, 0, 0]);
            assert_eq!(st[&Var::V(0)], reference(name, &[a]), "{insn} with {a:#x}");
        }
    }
    for (op, n) in [
        (Op::v_cvt_f32_ubyte0, 0),
        (Op::v_cvt_f32_ubyte1, 1),
        (Op::v_cvt_f32_ubyte2, 2),
        (Op::v_cvt_f32_ubyte3, 3),
    ] {
        let st = run(&vop1(op, v(0), v(1)), [0x1234_80ff, 0, 0]);
        assert_eq!(
            st[&Var::V(0)],
            (((0x1234_80ffu32 >> (8 * n)) & 0xff) as f32).to_bits()
        );
    }
    let st = run(&vop1(Op::v_not_b32, v(0), v(1)), [5, 0, 0]);
    assert_eq!(st[&Var::V(0)], !5);
}

#[test]
fn a_destination_that_is_a_source_reads_the_old_value() {
    let st = run(&vop3(Op::v_add3_u32, v(1), v(1), v(2), v(1)), [10, 20, 0]);
    assert_eq!(st[&Var::V(1)], 40);
    let st = run(&vop1(Op::v_swap_b32, v(1), v(2)), [10, 20, 0]);
    assert_eq!((st[&Var::V(1)], st[&Var::V(2)]), (20, 10));
}

#[test]
fn the_carry_forms_produce_and_consume_the_carry() {
    let mut add = vop3(Op::v_add_co_u32_e64, v(0), v(1), v(2), Operand::None);
    add.dst2 = VCC;
    let mut sub = add;
    sub.op = Op::v_sub_co_u32_e64;
    let mut rev = add;
    rev.op = Op::v_subrev_co_u32_e64;
    let addc = vop2(Op::v_addc_co_u32, v(0), v(1), v(2));
    let subb = vop2(Op::v_subb_co_u32, v(0), v(1), v(2));
    let subbrev = vop2(Op::v_subbrev_co_u32, v(0), v(1), v(2));
    for &a in &SAMPLES {
        for &b in &SAMPLES {
            let pair = |st: &State| (st[&Var::V(0)], st[&Var::Vcc]);
            let (r, c) = a.overflowing_add(b);
            assert_eq!(pair(&run(&add, [a, b, 0])), (r, u32::from(c)));
            let (r, c) = a.overflowing_sub(b);
            assert_eq!(pair(&run(&sub, [a, b, 0])), (r, u32::from(c)));
            let (r, c) = b.overflowing_sub(a);
            assert_eq!(pair(&run(&rev, [a, b, 0])), (r, u32::from(c)));
            for cin in [0u32, 1] {
                let st = |i: &Insn| {
                    pair(&exec(
                        i,
                        [(Var::V(1), a), (Var::V(2), b), (Var::Vcc, cin)].into(),
                    ))
                };
                let full = u64::from(a) + u64::from(b) + u64::from(cin);
                assert_eq!(st(&addc), (full as u32, (full >> 32) as u32), "addc");
                let full = i64::from(a) - i64::from(b) - i64::from(cin);
                assert_eq!(st(&subb), (full as u32, u32::from(full < 0)), "subb");
                let full = i64::from(b) - i64::from(a) - i64::from(cin);
                assert_eq!(st(&subbrev), (full as u32, u32::from(full < 0)), "subbrev");
            }
        }
    }
}

#[test]
fn cndmask_takes_the_mask_from_vcc() {
    let insn = vop2(Op::v_cndmask_b32, v(0), v(1), v(2));
    for (vcc, want) in [(0, 5), (1, 9)] {
        let st = exec(
            &insn,
            [(Var::V(1), 5), (Var::V(2), 9), (Var::Vcc, vcc)].into(),
        );
        assert_eq!(st[&Var::V(0)], want);
    }
    // The negate modifier of the two word form flips the sign bit of the chosen source.
    let mut neg = vop3(Op::v_cndmask_b32, v(0), v(1), v(2), VCC);
    neg.ext = Ext::Valu {
        abs: 0,
        neg: 2,
        opsel: 0,
        omod: 0,
        clamp: false,
    };
    let st = exec(&neg, [(Var::V(1), 5), (Var::V(2), 9), (Var::Vcc, 1)].into());
    assert_eq!(st[&Var::V(0)], 9 ^ 0x8000_0000);
}

fn sdwa(op: Op, word: u32) -> Insn {
    let mut i = vop2(op, v(0), v(1), v(2));
    i.form = Form::Sdwa;
    i.ext = Ext::Sdwa(word);
    i
}

#[test]
fn sdwa_selects_bytes_and_words_and_places_the_result() {
    // src0 BYTE1 zero extended, src1 WORD1 sign extended, dword result.
    let word = (1 << 16) | (5 << 24) | (1 << 27) | (6 << 8);
    let st = run(&sdwa(Op::v_add_u32, word), [0x0000_ab00, 0x8001_0000, 0]);
    assert_eq!(st[&Var::V(0)], 0xabu32.wrapping_add(0xffff_8001));
    // The result goes to BYTE2 and the other bytes are kept.
    let word = (6 << 16) | (6 << 24) | (2 << 8) | (2 << 11);
    let st = exec(
        &sdwa(Op::v_add_u32, word),
        [(Var::V(0), 0x1122_3344), (Var::V(1), 1), (Var::V(2), 2)].into(),
    );
    assert_eq!(st[&Var::V(0)], 0x1103_3344);
    // Padding with zeros, result in WORD1.
    let word = (6 << 16) | (6 << 24) | (5 << 8);
    let st = run(&sdwa(Op::v_add_u32, word), [1, 2, 0]);
    assert_eq!(st[&Var::V(0)], 0x0003_0000);
}

#[test]
fn the_sub_dword_forms_of_the_int_to_float_conversions_pick_the_field() {
    let mut i = vop1(Op::v_cvt_f32_u32, v(0), v(1));
    i.form = Form::Sdwa;
    // src0 BYTE2.
    i.ext = Ext::Sdwa((2 << 16) | (6 << 8));
    let st = run(&i, [0x00ab_0000, 0, 0]);
    assert_eq!(st[&Var::V(0)], 171f32.to_bits());
    // The form the title uses: src0 WORD0, dword result, OMOD 1 (times two). The output
    // modifier acts on the float result.
    i.ext = Ext::Sdwa(0x0004_4600);
    let st = run(&i, [0xdead_0007, 0, 0]);
    assert_eq!(st[&Var::V(0)], (2.0f32 * 7.0).to_bits());
    // The signed conversion with a sign extended word.
    let mut s = i;
    s.op = Op::v_cvt_f32_i32;
    s.ext = Ext::Sdwa((4 << 16) | (1 << 19) | (6 << 8));
    let st = run(&s, [0x0000_fffe, 0, 0]);
    assert_eq!(st[&Var::V(0)], (-2.0f32).to_bits());
}

#[test]
fn sixteen_bit_forms_use_the_halves_the_opsel_names() {
    let mk = |op: Op, opsel: u8| {
        let mut i = vop3(op, v(0), v(1), v(2), Operand::None);
        i.ext = Ext::Valu {
            abs: 0,
            neg: 0,
            opsel,
            omod: 0,
            clamp: false,
        };
        i
    };
    let st = run(&mk(Op::v_add_u16_e64, 0), [0x0001_ffff, 2, 0]);
    assert_eq!(st[&Var::V(0)], 1);
    let st = run(&mk(Op::v_lshrrev_b16_e64, 2), [3, 0x8000_0000, 0]);
    assert_eq!(st[&Var::V(0)], 0x8000 >> 3);
    let st = run(&mk(Op::v_ashrrev_i16_e64, 0), [4, 0x8000, 0]);
    assert_eq!(st[&Var::V(0)], 0xf800);
    let st = run(&mk(Op::v_max_i16_e64, 0), [0xffff, 1, 0]);
    assert_eq!(st[&Var::V(0)], 1);
    // The high destination half keeps the low half.
    let st = exec(
        &mk(Op::v_add_u16_e64, 8),
        [(Var::V(0), 0xaaaa_bbbb), (Var::V(1), 1), (Var::V(2), 2)].into(),
    );
    assert_eq!(st[&Var::V(0)], 0x0003_bbbb);
}

#[test]
fn a_program_with_every_integer_instruction_compiles() {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    let mut a = Asm::new();
    for (chan, r) in [2u8, 3, 4, 5].into_iter().enumerate() {
        a.push(interp(Op::v_interp_p1_f32, v(r), v(0), 0, chan as u8));
    }
    for (chan, r) in [2u8, 3, 4, 5].into_iter().enumerate() {
        a.push(interp(Op::v_interp_p2_f32, v(r), v(1), 0, chan as u8));
    }
    let threes = [
        Op::v_bfe_u32,
        Op::v_bfe_i32,
        Op::v_bfi_b32,
        Op::v_alignbit_b32,
        Op::v_alignbyte_b32,
        Op::v_perm_b32,
        Op::v_min3_i32,
        Op::v_max3_u32,
        Op::v_med3_i32,
        Op::v_med3_u32,
        Op::v_sad_u8,
        Op::v_sad_u32,
        Op::v_lerp_u8,
        Op::v_add3_u32,
        Op::v_or3_b32,
        Op::v_xor3_b32,
        Op::v_and_or_b32,
        Op::v_lshl_or_b32,
        Op::v_lshl_add_u32,
        Op::v_add_lshl_u32,
        Op::v_xad_u32,
        Op::v_mad_u32_u24,
        Op::v_mad_i32_i24,
        Op::v_mul_hi_u32,
        Op::v_mul_hi_i32,
        Op::v_bfm_b32,
        Op::v_bcnt_u32_b32,
        Op::v_mul_hi_u32_u24,
        Op::v_mul_hi_i32_i24,
    ];
    for op in threes {
        a.push(vop3(op, v(6), v(2), v(3), v(4)));
        a.push(vop2(Op::v_xor_b32, v(5), v(5), v(6)));
    }
    for op in [
        Op::v_bfrev_b32,
        Op::v_ffbh_u32,
        Op::v_ffbh_i32,
        Op::v_ffbl_b32,
        Op::v_not_b32,
        Op::v_cvt_f32_ubyte1,
    ] {
        a.push(vop1(op, v(6), v(2)));
        a.push(vop2(Op::v_xor_b32, v(5), v(5), v(6)));
    }
    let mut add = vop3(Op::v_add_co_u32_e64, v(6), v(2), v(3), Operand::None);
    add.dst2 = VCC;
    a.push(add);
    a.push(vop2(Op::v_addc_co_u32, v(7), v(2), v(3)));
    a.push(vop2(Op::v_cndmask_b32, v(8), v(7), v(6)));
    a.push(vop2(Op::v_xor_b32, v(5), v(5), v(8)));
    let mut sh = vop3(Op::v_lshrrev_b16_e64, v(6), v(2), v(3), Operand::None);
    sh.ext = Ext::Valu {
        abs: 0,
        neg: 0,
        opsel: 0,
        omod: 0,
        clamp: false,
    };
    a.push(sh);
    a.push(sdwa(Op::v_add_u32, (1 << 16) | (5 << 24) | (6 << 8)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(5), v(2)));
    a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(3), v(4)));
    a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let element = |name: &str, system_value: u32, mask: u8, read: u8| SigElement {
        semantic: name.to_owned(),
        index: 0,
        system_value,
        component_type: 3,
        register: 0,
        mask,
        rw_mask: read,
    };
    let p = StageProgram {
        stage: Stage::Pixel,
        code: a.finish(),
        inputs: vec![element("COLOR", 0, 15, 15)],
        outputs: vec![element("SV_Target", 64, 15, 0)],
        resources: Vec::new(),
        binding_table: Vec::new(),
        name: "valu_int".to_owned(),
        ..StageProgram::new(Stage::Pixel, Vec::new())
    };
    let compiled = translate_to_dxil(
        &[p],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(&compiled[0].dxil[..4], b"DXBC");
}

/// Lifts `insn` with its first source given, as the lane rules do for the data parallel forms.
fn exec_given(insn: &Insn, src0: u32, st: &mut State) {
    let masks = Masks::default();
    let mut cx = Lifter::new(Stage::Pixel, &masks);
    assert!(
        decant_shader::lift::valu_int::lift_with_src0(&mut cx, insn, Expr::U32(src0))
            .unwrap_or_else(|r| panic!("{insn}: {r}"))
    );
    st.insert(Var::Exec, 1);
    for s in cx.take() {
        let Stmt::Set { dst, value, .. } = s else {
            panic!("unexpected {s:?}")
        };
        let v = eval(&value, st);
        st.insert(dst, v);
    }
}

#[test]
fn a_data_parallel_form_computes_with_the_first_source_the_lane_rules_give() {
    let dpp = |op: Op, word: u32| {
        let mut i = vop2(op, v(0), v(1), v(2));
        i.form = Form::Dpp16;
        i.ext = Ext::Dpp(word);
        i
    };
    let mut st: State = [(Var::V(1), 1), (Var::V(2), 0xf0)].into();
    exec_given(&dpp(Op::v_mov_b32, 0xb1), 7, &mut st);
    assert_eq!(st[&Var::V(0)], 7);
    exec_given(&dpp(Op::v_or_b32, 0x140), 0x0f, &mut st);
    assert_eq!(st[&Var::V(0)], 0xff);
    exec_given(&dpp(Op::v_max_u32, 0x140), 0x100, &mut st);
    assert_eq!(st[&Var::V(0)], 0x100);
    // The reverse forms are refused.
    let masks = Masks::default();
    let mut cx = Lifter::new(Stage::Pixel, &masks);
    let r = decant_shader::lift::valu_int::lift_with_src0(
        &mut cx,
        &dpp(Op::v_lshlrev_b32, 0x140),
        Expr::U32(1),
    )
    .unwrap_err();
    assert_eq!(r.key(), "form:v_lshlrev_b32:dpp");
}

#[test]
fn the_packed_shifts_shift_each_half() {
    let mut i = vop3(Op::v_pk_lshlrev_b16, v(0), v(1), v(2), Operand::None);
    i.form = Form::Plain;
    // op_sel 0, op_sel_hi 0b11: the high result reads the high halves.
    i.ext = Ext::Valu {
        abs: 0,
        neg: 0,
        opsel: 0,
        omod: 0b11,
        clamp: false,
    };
    let st = run(&i, [0x0004_0011, 0x8001_0003, 0]);
    assert_eq!(st[&Var::V(0)], (0x0010 << 16) | 0x0006);
    i.op = Op::v_pk_lshrrev_b16;
    let st = run(&i, [0x0004_0001, 0x8000_0006, 0]);
    assert_eq!(st[&Var::V(0)], (0x0800 << 16) | 0x0003);
}
