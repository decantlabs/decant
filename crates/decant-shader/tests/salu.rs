//! Scalar instructions: the lifted statements are run by a small evaluator and compared with a
//! model of each instruction written from the semantics table (bit loops and wide integers
//! instead of the shift and mask formulas the lifter uses). The helpers the lifter calls are
//! mirrored by name in `helper`, a port of their HLSL text. A last test compiles the HLSL of a
//! program that uses all of them with DXC.

mod common;

use std::collections::HashMap;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::Insn;
use decant_shader::exec::Masks;
use decant_shader::ir::{Bin, Expr, Stmt, Un, Var};
use decant_shader::lift::{Lifter, lift};
use decant_shader::tables::Op;
use decant_shader::translate_to_dxil;

type State = HashMap<Var, u64>;

/// The helpers by name: the same arithmetic as their HLSL text.
fn helper(name: &str, a: &[u64]) -> u64 {
    let (x, y) = (
        a.first().copied().unwrap_or(0),
        a.get(1).copied().unwrap_or(0),
    );
    let (lo, hi) = (x as u32, (x >> 32) as u32);
    let first_low = |v: u32| if v == 0 { u32::MAX } else { v.trailing_zeros() };
    let first_high = |v: u32| 31 - v.leading_zeros();
    match name {
        "decant_pack64" => (y << 32) | (x & 0xffff_ffff),
        "decant_hi64" => x >> 32,
        "decant_shl64" => x << (y & 63),
        "decant_lshr64" => x >> (y & 63),
        "decant_ashr64" => ((x as i64) >> (y & 63)) as u64,
        "decant_brev64" => x.reverse_bits(),
        "decant_brev" => u64::from((x as u32).reverse_bits()),
        "decant_bfe_u32" => {
            let w = (y >> 16) & 127;
            let v = (x as u32) >> (y & 31);
            u64::from(if w == 0 {
                0
            } else if w >= 32 {
                v
            } else {
                v & ((1u32 << w) - 1)
            })
        }
        "decant_bfe_i32" => {
            let w = ((y >> 16) & 127).min(32) as u32;
            let v = (x as u32 as i32) >> (y & 31);
            u64::from(if w == 0 {
                0
            } else {
                let up = 32 - w;
                ((v << up) >> up) as u32
            })
        }
        "decant_bfe_u64" => {
            let w = (y >> 16) & 127;
            let v = x >> (y & 63);
            if w == 0 {
                0
            } else if w >= 64 {
                v
            } else {
                v & ((1u64 << w) - 1)
            }
        }
        "decant_bfe_i64" => {
            let w = ((y >> 16) & 127).min(64) as u32;
            let v = (x as i64) >> (y & 63);
            if w == 0 {
                0
            } else {
                let up = 64 - w;
                ((v << up) >> up) as u64
            }
        }
        "decant_bfm_b32" => u64::from(((1u32 << (x & 31)) - 1) << (y & 31)),
        "decant_bfm_b64" => ((1u64 << (x & 63)).wrapping_sub(1)) << (y & 63),
        "decant_mul_hi_u32" => (x * y) >> 32,
        "decant_mul_hi_i32" => {
            u64::from(((i64::from(x as u32 as i32) * i64::from(y as u32 as i32)) >> 32) as u32)
        }
        "decant_shl_add_carry" => u64::from((((x << a[2]) + y) >> 32) != 0),
        "decant_ff1_b32" => u64::from(first_low(x as u32)),
        "decant_ff1_b64" => u64::from(if lo != 0 {
            first_low(lo)
        } else if hi != 0 {
            32 + first_low(hi)
        } else {
            u32::MAX
        }),
        "decant_flbit_b32" => u64::from(if x as u32 == 0 {
            u32::MAX
        } else {
            first_high_inv(x as u32)
        }),
        "decant_flbit_b64" => u64::from(if hi != 0 {
            31 - first_high(hi)
        } else if lo != 0 {
            63 - first_high(lo)
        } else {
            u32::MAX
        }),
        "decant_flbit_i32" => {
            let v = x as u32;
            let y = if (v as i32) < 0 { !v } else { v };
            u64::from(if y == 0 { u32::MAX } else { first_high_inv(y) })
        }
        "decant_flbit_i64" => {
            let y = if (x as i64) < 0 { !x } else { x };
            helper("decant_flbit_b64", &[y])
        }
        "decant_bcnt_b32" => u64::from((x as u32).count_ones()),
        "decant_bcnt_b64" => u64::from(x.count_ones()),
        "decant_bit_b64" => (x >> (y & 63)) & 1,
        "decant_bitset1_b64" => x | (1u64 << (y & 63)),
        "decant_bitset0_b64" => x & !(1u64 << (y & 63)),
        "decant_sext_i8" => u64::from(i32::from(x as u8 as i8) as u32),
        "decant_sext_i16" => u64::from(i32::from(x as u16 as i16) as u32),
        other => panic!("no model for the helper {other}"),
    }
}

/// `31 - firstbithigh`, the leading zeros of a nonzero value.
fn first_high_inv(v: u32) -> u32 {
    v.leading_zeros()
}

fn eval(st: &State, e: &Expr) -> u64 {
    let get = |v: &Var| st.get(v).copied().unwrap_or(0);
    match e {
        Expr::U32(v) => u64::from(*v),
        Expr::U64(v) => *v,
        Expr::Bool(b) => u64::from(*b),
        Expr::Var(v) => get(v),
        Expr::Un(op, a) => {
            let x = eval(st, a);
            match op {
                Un::Not if a.ty() == decant_shader::ir::Ty::U64 => !x,
                Un::Not => u64::from(!(x as u32)),
                Un::LNot => u64::from(x == 0),
                Un::Neg => u64::from(0u32.wrapping_sub(x as u32)),
                Un::BoolToU32 => x,
                Un::U32ToBool => u64::from(x as u32 != 0),
                Un::U32ToU64 => x,
                Un::U64ToU32 => x & 0xffff_ffff,
                Un::FNeg | Un::FAbs => panic!("float"),
            }
        }
        Expr::Bin(op, a, b) => {
            let (x, y) = (eval(st, a) as u32, eval(st, b) as u32);
            let w = |v: u32| u64::from(v);
            let t = |c: bool| u64::from(c);
            match op {
                Bin::Add => w(x.wrapping_add(y)),
                Bin::Sub => w(x.wrapping_sub(y)),
                Bin::Mul => w(x.wrapping_mul(y)),
                Bin::And => w(x & y),
                Bin::Or => w(x | y),
                Bin::Xor => w(x ^ y),
                Bin::Shl => w(x << (y & 31)),
                Bin::LShr => w(x >> (y & 31)),
                Bin::AShr => w(((x as i32) >> (y & 31)) as u32),
                Bin::Eq => t(x == y),
                Bin::Ne => t(x != y),
                Bin::ULt => t(x < y),
                Bin::ULe => t(x <= y),
                Bin::SLt => t((x as i32) < (y as i32)),
                Bin::SLe => t((x as i32) <= (y as i32)),
                Bin::LAnd => t(x != 0 && y != 0),
                Bin::LOr => t(x != 0 || y != 0),
                Bin::LXor => t((x != 0) != (y != 0)),
                _ => panic!("float"),
            }
        }
        Expr::Select(c, a, b) => {
            if eval(st, c) != 0 {
                eval(st, a)
            } else {
                eval(st, b)
            }
        }
        Expr::Call(h, args) => {
            let a: Vec<u64> = args.iter().map(|a| eval(st, a)).collect();
            helper(h.name, &a)
        }
        other => panic!("not evaluated: {other:?}"),
    }
}

/// Lifts and runs the instructions on the given registers.
fn run(insns: &[Insn], init: &[(Var, u64)]) -> State {
    let masks = Masks::default();
    let mut cx = Lifter::new(Stage::Pixel, &masks);
    for i in insns {
        lift(&mut cx, i).unwrap_or_else(|r| panic!("{i}: {r}"));
    }
    let mut st: State = init.iter().copied().collect();
    for stmt in cx.take() {
        match stmt {
            Stmt::Set { dst, value, .. } => {
                let v = eval(&st, &value);
                st.insert(dst, v);
            }
            other => panic!("unexpected statement {other:?}"),
        }
    }
    st
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // A mix of small values, edges and random words so that carries and widths are hit.
        let r = (self.0 >> 20) as u32;
        match r % 7 {
            0 => r % 40,
            1 => u32::MAX - (r % 5),
            2 => 0x8000_0000u32.wrapping_add(r % 5).wrapping_sub(2),
            _ => (self.0 >> 8) as u32,
        }
    }
}

/// The result of a two operand instruction: the destination (when it writes one) and SCC.
struct Model {
    d: Option<u32>,
    scc: Option<bool>,
}

fn bit_field(a: u32, off: u32, w: u32, signed: bool) -> u32 {
    let w = w.min(32);
    let source = |i: u32| {
        if off + i < 32 {
            (a >> (off + i)) & 1
        } else {
            // The bits above the register are copies of the top bit for the signed form.
            if signed { a >> 31 } else { 0 }
        }
    };
    let mut d = 0u32;
    for i in 0..w {
        d |= source(i) << i;
    }
    if signed && w > 0 && w < 32 && (d >> (w - 1)) & 1 != 0 {
        d |= u32::MAX << w;
    }
    d
}

fn bit_field64(a: u64, off: u32, w: u32, signed: bool) -> u64 {
    let w = w.min(64);
    let source = |i: u32| {
        if off + i < 64 {
            (a >> (off + i)) & 1
        } else if signed {
            a >> 63
        } else {
            0
        }
    };
    let mut d = 0u64;
    for i in 0..w {
        d |= source(i) << i;
    }
    if signed && w > 0 && w < 64 && (d >> (w - 1)) & 1 != 0 {
        d |= u64::MAX << w;
    }
    d
}

fn model32(op: Op, a: u32, b: u32, scc: bool) -> Model {
    let nz = |d: u32| Model {
        d: Some(d),
        scc: Some(d != 0),
    };
    let plain = |d: u32| Model {
        d: Some(d),
        scc: None,
    };
    let (ia, ib) = (i64::from(a as i32), i64::from(b as i32));
    let wide = |d: i64| Model {
        d: Some(d as u32),
        scc: Some(d != i64::from(d as i32)),
    };
    match op {
        Op::s_add_u32 => {
            let t = u64::from(a) + u64::from(b);
            Model {
                d: Some(t as u32),
                scc: Some(t >> 32 != 0),
            }
        }
        Op::s_sub_u32 => Model {
            d: Some(a.wrapping_sub(b)),
            scc: Some(b > a),
        },
        Op::s_add_i32 => wide(ia + ib),
        Op::s_sub_i32 => wide(ia - ib),
        Op::s_addc_u32 => {
            let t = u64::from(a) + u64::from(b) + u64::from(scc);
            Model {
                d: Some(t as u32),
                scc: Some(t >> 32 != 0),
            }
        }
        Op::s_subb_u32 => Model {
            d: Some(a.wrapping_sub(b).wrapping_sub(u32::from(scc))),
            scc: Some(u64::from(b) + u64::from(scc) > u64::from(a)),
        },
        Op::s_min_u32 => Model {
            d: Some(a.min(b)),
            scc: Some(a < b),
        },
        Op::s_max_u32 => Model {
            d: Some(a.max(b)),
            scc: Some(a > b),
        },
        Op::s_min_i32 => Model {
            d: Some((a as i32).min(b as i32) as u32),
            scc: Some(ia < ib),
        },
        Op::s_max_i32 => Model {
            d: Some((a as i32).max(b as i32) as u32),
            scc: Some(ia > ib),
        },
        Op::s_and_b32 => nz(a & b),
        Op::s_or_b32 => nz(a | b),
        Op::s_xor_b32 => nz(a ^ b),
        Op::s_andn2_b32 => nz(a & !b),
        Op::s_orn2_b32 => nz(a | !b),
        Op::s_nand_b32 => nz(!(a & b)),
        Op::s_nor_b32 => nz(!(a | b)),
        Op::s_xnor_b32 => nz(!(a ^ b)),
        Op::s_lshl_b32 => nz(a << (b & 31)),
        Op::s_lshr_b32 => nz(a >> (b & 31)),
        Op::s_ashr_i32 => nz(((a as i32) >> (b & 31)) as u32),
        Op::s_mul_i32 => plain(a.wrapping_mul(b)),
        Op::s_mul_hi_u32 => plain(((u64::from(a) * u64::from(b)) >> 32) as u32),
        Op::s_mul_hi_i32 => plain(((ia * ib) >> 32) as u32),
        Op::s_absdiff_i32 => nz((a.wrapping_sub(b) as i32).wrapping_abs() as u32),
        Op::s_bfe_u32 => nz(bit_field(a, b & 31, (b >> 16) & 127, false)),
        Op::s_bfe_i32 => nz(bit_field(a, b & 31, (b >> 16) & 127, true)),
        Op::s_bfm_b32 => {
            let (w, o) = (a & 31, b & 31);
            let mut d = 0u32;
            for i in 0..w {
                if i + o < 32 {
                    d |= 1 << (i + o);
                }
            }
            plain(d)
        }
        Op::s_cselect_b32 => plain(if scc { a } else { b }),
        Op::s_lshl1_add_u32 | Op::s_lshl2_add_u32 | Op::s_lshl3_add_u32 | Op::s_lshl4_add_u32 => {
            let n = match op {
                Op::s_lshl1_add_u32 => 1,
                Op::s_lshl2_add_u32 => 2,
                Op::s_lshl3_add_u32 => 3,
                _ => 4,
            };
            let t = (u64::from(a) << n) + u64::from(b);
            Model {
                d: Some(t as u32),
                scc: Some(t >> 32 != 0),
            }
        }
        Op::s_pack_ll_b32_b16 => plain((a & 0xffff) | (b << 16)),
        Op::s_pack_lh_b32_b16 => plain((a & 0xffff) | (b & 0xffff_0000)),
        Op::s_pack_hh_b32_b16 => plain((a >> 16) | (b & 0xffff_0000)),
        other => panic!("no model for {other:?}"),
    }
}

const SOP2_32: &[Op] = &[
    Op::s_add_u32,
    Op::s_sub_u32,
    Op::s_add_i32,
    Op::s_sub_i32,
    Op::s_addc_u32,
    Op::s_subb_u32,
    Op::s_min_u32,
    Op::s_max_u32,
    Op::s_min_i32,
    Op::s_max_i32,
    Op::s_and_b32,
    Op::s_or_b32,
    Op::s_xor_b32,
    Op::s_andn2_b32,
    Op::s_orn2_b32,
    Op::s_nand_b32,
    Op::s_nor_b32,
    Op::s_xnor_b32,
    Op::s_lshl_b32,
    Op::s_lshr_b32,
    Op::s_ashr_i32,
    Op::s_mul_i32,
    Op::s_mul_hi_u32,
    Op::s_mul_hi_i32,
    Op::s_absdiff_i32,
    Op::s_bfe_u32,
    Op::s_bfe_i32,
    Op::s_bfm_b32,
    Op::s_cselect_b32,
    Op::s_lshl1_add_u32,
    Op::s_lshl2_add_u32,
    Op::s_lshl3_add_u32,
    Op::s_lshl4_add_u32,
    Op::s_pack_ll_b32_b16,
    Op::s_pack_lh_b32_b16,
    Op::s_pack_hh_b32_b16,
];

#[test]
fn every_two_operand_32_bit_instruction_matches_its_model_with_scc() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for &op in SOP2_32 {
        for round in 0..400 {
            let (a, mut b) = (rng.next(), rng.next());
            if matches!(op, Op::s_bfe_u32 | Op::s_bfe_i32) && round % 2 == 0 {
                // Offsets and widths in range more often than a random word gives them.
                b = (b & 31) | ((rng.next() % 40) << 16);
            }
            let scc = rng.next() & 1 != 0;
            let st = run(
                &[sop2(op, s(3), s(1), s(2))],
                &[
                    (Var::S(1), u64::from(a)),
                    (Var::S(2), u64::from(b)),
                    (Var::Scc, u64::from(scc)),
                ],
            );
            let m = model32(op, a, b, scc);
            assert_eq!(
                st.get(&Var::S(3)).copied(),
                m.d.map(u64::from),
                "{op:?} a={a:#x} b={b:#x} scc={scc}"
            );
            let want_scc = m.scc.unwrap_or(scc);
            assert_eq!(
                st.get(&Var::Scc).copied().unwrap_or(0) != 0,
                want_scc,
                "SCC of {op:?} a={a:#x} b={b:#x} scc={scc}"
            );
        }
    }
}

#[test]
fn a_destination_that_is_also_a_source_reads_the_old_value() {
    // s_addc_u32 s1, s1, s2 with the carry in: the carry out must not see the new s1.
    let st = run(
        &[sop2(Op::s_addc_u32, s(1), s(1), s(2))],
        &[(Var::S(1), 0xffff_ffff), (Var::S(2), 0), (Var::Scc, 1)],
    );
    assert_eq!((st[&Var::S(1)], st[&Var::Scc]), (0, 1));
    let st = run(
        &[sop2(Op::s_subb_u32, s(1), s(1), s(2))],
        &[(Var::S(1), 5), (Var::S(2), 5), (Var::Scc, 1)],
    );
    assert_eq!((st[&Var::S(1)], st[&Var::Scc]), (u64::from(u32::MAX), 1));
}

#[test]
fn a_write_to_the_null_register_only_sets_scc() {
    let st = run(
        &[
            sop1(Op::s_mov_b32, Operand::Reg(Reg::Null), int(5)),
            sop2(Op::s_and_b32, Operand::Reg(Reg::Null), s(1), s(2)),
        ],
        &[(Var::S(1), 6), (Var::S(2), 1)],
    );
    assert_eq!(st[&Var::Scc], 0);
    assert!(!st.contains_key(&Var::S(0)));
}

use decant_shader::decode::{Operand, Reg};

#[test]
fn sopk_forms_extend_their_immediate_as_the_table_says() {
    let st = run(&[sopk(Op::s_addk_i32, s(1), 0xfffe)], &[(Var::S(1), 5)]);
    assert_eq!((st[&Var::S(1)], st[&Var::Scc]), (3, 0));
    let st = run(
        &[sopk(Op::s_addk_i32, s(1), 1)],
        &[(Var::S(1), 0x7fff_ffff)],
    );
    assert_eq!((st[&Var::S(1)], st[&Var::Scc]), (0x8000_0000, 1));
    let st = run(&[sopk(Op::s_mulk_i32, s(1), 0xfffd)], &[(Var::S(1), 7)]);
    assert_eq!(st[&Var::S(1)], u64::from((-21i32) as u32));
    let st = run(
        &[sopk(Op::s_cmovk_i32, s(1), 0x8000)],
        &[(Var::S(1), 9), (Var::Scc, 0)],
    );
    assert_eq!(st[&Var::S(1)], 9);
    let st = run(
        &[sopk(Op::s_cmovk_i32, s(1), 0x8000)],
        &[(Var::S(1), 9), (Var::Scc, 1)],
    );
    assert_eq!(st[&Var::S(1)], 0xffff_8000);
}

#[test]
fn one_operand_instructions_follow_the_table() {
    let mut rng = Rng(12345);
    for _ in 0..300 {
        let a = rng.next();
        let one = |op: Op| run(&[sop1(op, s(3), s(1))], &[(Var::S(1), u64::from(a))]);
        let st = one(Op::s_brev_b32);
        assert_eq!(st[&Var::S(3)], u64::from(a.reverse_bits()));
        let st = one(Op::s_not_b32);
        assert_eq!(
            (st[&Var::S(3)], st[&Var::Scc] != 0),
            (u64::from(!a), !a != 0)
        );
        let st = one(Op::s_abs_i32);
        let abs = (a as i32).wrapping_abs() as u32;
        assert_eq!(
            (st[&Var::S(3)], st[&Var::Scc] != 0),
            (u64::from(abs), abs != 0)
        );
        let st = one(Op::s_bcnt1_i32_b32);
        assert_eq!(
            (st[&Var::S(3)], st[&Var::Scc] != 0),
            (u64::from(a.count_ones()), a != 0)
        );
        let st = one(Op::s_bcnt0_i32_b32);
        assert_eq!(
            (st[&Var::S(3)], st[&Var::Scc] != 0),
            (u64::from(a.count_zeros()), a != u32::MAX)
        );
        let st = one(Op::s_ff1_i32_b32);
        let ff1 = (0..32).find(|i| (a >> i) & 1 == 1).unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(3)], u64::from(ff1));
        let st = one(Op::s_ff0_i32_b32);
        let ff0 = (0..32).find(|i| (a >> i) & 1 == 0).unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(3)], u64::from(ff0));
        let st = one(Op::s_flbit_i32_b32);
        let clz = (0..32)
            .find(|i| (a >> (31 - i)) & 1 == 1)
            .unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(3)], u64::from(clz));
        let st = one(Op::s_flbit_i32);
        let sign = a >> 31;
        let lead = (0..32)
            .find(|i| (a >> (31 - i)) & 1 != sign)
            .unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(3)], u64::from(lead), "s_flbit_i32 {a:#x}");
        let st = one(Op::s_sext_i32_i8);
        assert_eq!(st[&Var::S(3)], u64::from(i32::from(a as u8 as i8) as u32));
        let st = one(Op::s_sext_i32_i16);
        assert_eq!(st[&Var::S(3)], u64::from(i32::from(a as u16 as i16) as u32));
    }
}

#[test]
fn bit_set_and_bit_compare_use_five_bit_indices() {
    let mut rng = Rng(777);
    for _ in 0..200 {
        let (d, i) = (rng.next(), rng.next());
        let st = run(
            &[sop1(Op::s_bitset1_b32, s(1), s(2))],
            &[(Var::S(1), u64::from(d)), (Var::S(2), u64::from(i))],
        );
        assert_eq!(st[&Var::S(1)], u64::from(d | (1 << (i & 31))));
        let st = run(
            &[sop1(Op::s_bitset0_b32, s(1), s(2))],
            &[(Var::S(1), u64::from(d)), (Var::S(2), u64::from(i))],
        );
        assert_eq!(st[&Var::S(1)], u64::from(d & !(1 << (i & 31))));
        let bit = (d >> (i & 31)) & 1;
        for (op, want) in [(Op::s_bitcmp1_b32, bit == 1), (Op::s_bitcmp0_b32, bit == 0)] {
            let st = run(
                &[sopc(op, s(1), s(2))],
                &[(Var::S(1), u64::from(d)), (Var::S(2), u64::from(i))],
            );
            assert_eq!(st[&Var::Scc] != 0, want, "{op:?}");
        }
    }
}

fn pair_state(reg: u8, v: u64) -> [(Var, u64); 2] {
    [(Var::S(reg), v & 0xffff_ffff), (Var::S(reg + 1), v >> 32)]
}

fn read_pair(st: &State, reg: u8) -> u64 {
    st[&Var::S(reg)] | (st[&Var::S(reg + 1)] << 32)
}

#[test]
fn sixty_four_bit_data_forms_match_wide_integer_models() {
    let mut rng = Rng(4242);
    for round in 0..300 {
        let a = (u64::from(rng.next()) << 32) | u64::from(rng.next());
        let b = (u64::from(rng.next()) << 32) | u64::from(rng.next());
        let n = rng.next();
        let run2 = |op: Op, count: bool| {
            let mut init = Vec::new();
            init.extend(pair_state(2, a));
            if count {
                init.push((Var::S(4), u64::from(n)));
            } else {
                init.extend(pair_state(4, b));
            }
            run(&[sop2(op, s(6), s(2), s(4))], &init)
        };
        let check = |op: Op, count: bool, want: u64| {
            let st = run2(op, count);
            assert_eq!(read_pair(&st, 6), want, "{op:?} a={a:#x} b={b:#x} n={n:#x}");
            assert_eq!(st[&Var::Scc] != 0, want != 0, "SCC of {op:?}");
        };
        check(Op::s_and_b64, false, a & b);
        check(Op::s_or_b64, false, a | b);
        check(Op::s_xor_b64, false, a ^ b);
        check(Op::s_andn2_b64, false, a & !b);
        check(Op::s_orn2_b64, false, a | !b);
        check(Op::s_nand_b64, false, !(a & b));
        check(Op::s_nor_b64, false, !(a | b));
        check(Op::s_xnor_b64, false, !(a ^ b));
        check(Op::s_lshl_b64, true, a << (n & 63));
        check(Op::s_lshr_b64, true, a >> (n & 63));
        check(Op::s_ashr_i64, true, ((a as i64) >> (n & 63)) as u64);
        let (off, w) = (
            n & 63,
            if round % 2 == 0 {
                (n >> 8) % 70
            } else {
                n >> 16 & 127
            },
        );
        let sel = u64::from(off | (w << 16));
        let run_bfe = |op: Op| {
            let mut init: Vec<(Var, u64)> = pair_state(2, a).into();
            init.push((Var::S(4), sel));
            run(&[sop2(op, s(6), s(2), s(4))], &init)
        };
        let st = run_bfe(Op::s_bfe_u64);
        assert_eq!(read_pair(&st, 6), bit_field64(a, off, w, false));
        let st = run_bfe(Op::s_bfe_i64);
        assert_eq!(read_pair(&st, 6), bit_field64(a, off, w, true));
        let st = run(
            &[sop2(Op::s_bfm_b64, s(6), s(2), s(4))],
            &[
                (Var::S(2), u64::from(n)),
                (Var::S(4), u64::from(rng.next())),
            ],
        );
        let (width, o) = (u64::from(n) & 63, st[&Var::S(4)] & 63);
        let mut want = 0u64;
        for i in 0..width {
            if i + o < 64 {
                want |= 1 << (i + o);
            }
        }
        assert_eq!(read_pair(&st, 6), want, "bfm_b64");
        // Moves, selects and compares.
        for scc in [false, true] {
            let mut init: Vec<(Var, u64)> = pair_state(2, a).into();
            init.extend(pair_state(4, b));
            init.extend(pair_state(6, 0x1111_2222_3333_4444));
            init.push((Var::Scc, u64::from(scc)));
            let st = run(&[sop2(Op::s_cselect_b64, s(8), s(2), s(4))], &init);
            assert_eq!(read_pair(&st, 8), if scc { a } else { b });
            assert_eq!(st[&Var::Scc], u64::from(scc));
            let st = run(&[sop1(Op::s_cmov_b64, s(6), s(2))], &init);
            assert_eq!(
                read_pair(&st, 6),
                if scc { a } else { 0x1111_2222_3333_4444 }
            );
        }
        let mut init: Vec<(Var, u64)> = pair_state(2, a).into();
        init.extend(pair_state(4, if round % 3 == 0 { a } else { b }));
        let same = if round % 3 == 0 { true } else { a == b };
        let st = run(&[sopc(Op::s_cmp_eq_u64, s(2), s(4))], &init);
        assert_eq!(st[&Var::Scc] != 0, same);
        let st = run(&[sopc(Op::s_cmp_lg_u64, s(2), s(4))], &init);
        assert_eq!(st[&Var::Scc] != 0, !same);
        // One operand forms.
        let one = |op: Op| {
            let init: Vec<(Var, u64)> = pair_state(2, a).into();
            run(&[sop1(op, s(6), s(2))], &init)
        };
        assert_eq!(read_pair(&one(Op::s_brev_b64), 6), a.reverse_bits());
        let st = one(Op::s_not_b64);
        assert_eq!((read_pair(&st, 6), st[&Var::Scc] != 0), (!a, !a != 0));
        let st = one(Op::s_mov_b64);
        assert_eq!(read_pair(&st, 6), a);
        let st = one(Op::s_bcnt1_i32_b64);
        assert_eq!(st[&Var::S(6)], u64::from(a.count_ones()));
        let st = one(Op::s_bcnt0_i32_b64);
        assert_eq!(st[&Var::S(6)], u64::from(a.count_zeros()));
        let st = one(Op::s_ff1_i32_b64);
        let ff1 = (0..64).find(|i| (a >> i) & 1 == 1).unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(6)], u64::from(ff1));
        let st = one(Op::s_ff0_i32_b64);
        let ff0 = (0..64).find(|i| (a >> i) & 1 == 0).unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(6)], u64::from(ff0));
        let st = one(Op::s_flbit_i32_b64);
        let clz = (0..64)
            .find(|i| (a >> (63 - i)) & 1 == 1)
            .unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(6)], u64::from(clz));
        let st = one(Op::s_flbit_i32_i64);
        let sign = a >> 63;
        let lead = (0..64)
            .find(|i| (a >> (63 - i)) & 1 != sign)
            .unwrap_or(u32::MAX);
        assert_eq!(st[&Var::S(6)], u64::from(lead), "s_flbit_i32_i64 {a:#x}");
        // Bit set and compare on a pair.
        let mut init: Vec<(Var, u64)> = pair_state(6, a).into();
        init.push((Var::S(4), u64::from(n)));
        let st = run(&[sop1(Op::s_bitset1_b64, s(6), s(4))], &init);
        assert_eq!(read_pair(&st, 6), a | (1 << (n & 63)));
        let st = run(&[sop1(Op::s_bitset0_b64, s(6), s(4))], &init);
        assert_eq!(read_pair(&st, 6), a & !(1 << (n & 63)));
        let mut init: Vec<(Var, u64)> = pair_state(2, a).into();
        init.push((Var::S(4), u64::from(n)));
        let bit = (a >> (n & 63)) & 1;
        let st = run(&[sopc(Op::s_bitcmp1_b64, s(2), s(4))], &init);
        assert_eq!(st[&Var::Scc], bit);
        let st = run(&[sopc(Op::s_bitcmp0_b64, s(2), s(4))], &init);
        assert_eq!(st[&Var::Scc], bit ^ 1);
    }
}

#[test]
fn sixty_four_bit_constants_are_sign_extended_when_inline() {
    let st = run(&[sop1(Op::s_mov_b64, s(2), int(-1))], &[]);
    assert_eq!(read_pair(&st, 2), u64::MAX);
    let st = run(&[sop1(Op::s_mov_b64, s(2), int(5))], &[]);
    assert_eq!(read_pair(&st, 2), 5);
    let st = run(&[sop2(Op::s_lshl_b64, s(2), int(-1), int(60))], &[]);
    assert_eq!(read_pair(&st, 2), 0xf000_0000_0000_0000);
}

#[test]
fn instructions_that_only_order_or_configure_do_nothing() {
    let st = run(
        &[
            sopk(Op::s_waitcnt_vscnt, Operand::Reg(Reg::Null), 0),
            sopp(Op::s_sleep, 1),
            sopk(Op::s_setreg_b32, s(1), 0x881),
        ],
        &[],
    );
    assert!(st.is_empty());
}

fn element(semantic: &str, register: u32) -> SigElement {
    SigElement {
        semantic: semantic.to_owned(),
        index: 0,
        system_value: if semantic == "SV_Target" { 64 } else { 0 },
        component_type: 3,
        register,
        mask: 15,
        rw_mask: 0,
    }
}

#[test]
fn the_helpers_compile_with_dxc() {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("skipped the DXC compile: dxc.exe not found (set DECANT_SHADER_DXC)");
        return;
    };
    // Every two operand instruction on literals, the results folded into four vector registers
    // with s_mov and v_mov so that nothing is dropped as dead before the printer.
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(1), lit(0x1234_5678)));
    a.push(sop1(Op::s_mov_b32, s(2), lit(0x0010_0007)));
    for (next, &op) in (8u8..).zip(SOP2_32.iter()) {
        a.push(sop2(op, s(next % 40 + 8), s(1), s(2)));
    }
    for op in [
        Op::s_lshl_b64,
        Op::s_lshr_b64,
        Op::s_ashr_i64,
        Op::s_bfe_u64,
        Op::s_bfe_i64,
        Op::s_bfm_b64,
    ] {
        a.push(sop2(op, s(60), s(4), s(2)));
    }
    for op in [
        Op::s_brev_b64,
        Op::s_bcnt1_i32_b64,
        Op::s_bcnt0_i32_b64,
        Op::s_ff1_i32_b64,
        Op::s_ff0_i32_b64,
        Op::s_flbit_i32_b64,
        Op::s_flbit_i32_i64,
        Op::s_not_b64,
    ] {
        a.push(sop1(op, s(62), s(4)));
    }
    for op in [Op::s_bitset1_b64, Op::s_bitset0_b64] {
        a.push(sop1(op, s(62), s(2)));
    }
    for op in [
        Op::s_brev_b32,
        Op::s_abs_i32,
        Op::s_bcnt1_i32_b32,
        Op::s_bcnt0_i32_b32,
        Op::s_ff1_i32_b32,
        Op::s_ff0_i32_b32,
        Op::s_flbit_i32_b32,
        Op::s_flbit_i32,
        Op::s_sext_i32_i8,
        Op::s_sext_i32_i16,
    ] {
        a.push(sop1(op, s(64), s(1)));
    }
    a.push(sopc(Op::s_bitcmp1_b64, s(4), s(2)));
    a.push(sopc(Op::s_cmp_eq_u64, s(4), s(6)));
    a.push(sop2(Op::s_cselect_b64, s(66), s(4), s(6)));
    a.push(sopk(Op::s_addk_i32, s(1), 12));
    a.push(sopk(Op::s_mulk_i32, s(1), 3));
    for k in 0..4u8 {
        a.push(vop1(Op::v_mov_b32, v(k), s(8 + k * 9)));
    }
    a.push(export(0, 0xf, [0, 1, 2, 3], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let program = StageProgram {
        outputs: vec![element("SV_Target", 0)],
        name: "salu".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    let compiled = translate_to_dxil(
        &[program],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(&compiled[0].dxil[..4], b"DXBC");
}
