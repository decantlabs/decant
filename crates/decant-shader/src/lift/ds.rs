//! Local data share instructions (`ds_*`): group shared memory.
//!
//! The LDS of a group becomes one HLSL array, `groupshared uint decant_lds[N]`, that the printer
//! declares when [`crate::ir::Function::groupshared_dwords`] is not zero (the stage sets it from
//! the allocation the container gives when [`uses_lds`]), with the constant `decant_lds_words`
//! equal to `N`. Every access goes through a helper of this file by byte address: the machine
//! address is the vector register plus the 16 bit offset of the instruction (or the two 8 bit
//! offsets of the `2` forms, scaled by the element size and by 64 for the `st64` forms).
//!
//! Sources: the operand layout and the offset rules are Mesa's ACO (`aco_assembler.cpp`, the DS
//! encoder, and `aco_select_nir_intrinsics.cpp`, `visit_shared_atomic`: on GFX10 the compare
//! value of `ds_cmpst_*` is the first data operand and the new value the second). On GFX9 and
//! later only the `addtid` forms use M0 (Mesa `README-ISA.md`).
//!
//! Inference, not checked on the machine: an access at an address at or past the group's
//! allocation reads 0 and writes nothing. The helpers do the same against `decant_lds_words`, so an
//! address the program computes wrongly never touches memory outside the array, which HLSL
//! leaves undefined. Addresses of dword accesses are taken as aligned (the low two bits are
//! dropped).
//!
//! Byte and short stores share a dword with their neighbours. Two lanes may write the two halves
//! of one dword at the same time, so such a store is an atomic clear of its bits and an atomic
//! set, never a read, modify and write.
//!
//! What stays out: the cross lane instructions `ds_swizzle_b32`, `ds_permute_b32` and
//! `ds_bpermute_b32` (they move data between lanes and belong to `wave.rs`), global data share
//! (`gds`), the ordered count and append instructions, the `src2` and 64 bit atomics.

use super::Lifter;
use super::operand::form;
use crate::decode::{Ext, Insn, Operand, Reg};
use crate::ir::{Bin, Expr, Function, Guard, Helper, Stmt, Ty, Var};
use crate::refusal::Refusal;
use crate::tables::Op;

/// The dwords of group shared memory a Direct3D 12 compute shader may declare (32 KiB): the
/// array when the container does not give the allocation. A console group may allocate up to
/// 64 KiB, but no program of the title does.
pub const LDS_DWORDS: u32 = 8192;

pub static LOAD: Helper = Helper {
    name: "decant_lds_load",
    result: Ty::U32,
    text: "uint decant_lds_load(uint a) { uint i = a >> 2; uint w = decant_lds[min(i, decant_lds_words - 1u)]; return i < decant_lds_words ? w : 0u; }",
    needs: &[],
};

pub static LOAD8: Helper = Helper {
    name: "decant_lds_load8",
    result: Ty::U32,
    text: "uint decant_lds_load8(uint a) { return (decant_lds_load(a) >> ((a & 3u) * 8u)) & 0xFFu; }",
    needs: &[&LOAD],
};

pub static LOAD16: Helper = Helper {
    name: "decant_lds_load16",
    result: Ty::U32,
    text: "uint decant_lds_load16(uint a) { return (decant_lds_load(a) >> ((a & 2u) * 8u)) & 0xFFFFu; }",
    needs: &[&LOAD],
};

pub static STORE: Helper = Helper {
    name: "decant_lds_store",
    result: Ty::U32,
    text: "uint decant_lds_store(uint a, uint v) { uint i = a >> 2; if (i < decant_lds_words) { decant_lds[i] = v; } return 0u; }",
    needs: &[],
};

static STORE_BITS: Helper = Helper {
    name: "decant_lds_store_bits",
    result: Ty::U32,
    text: "uint decant_lds_store_bits(uint a, uint v, uint m) { uint i = a >> 2; if (i < decant_lds_words) { InterlockedAnd(decant_lds[i], ~m); InterlockedOr(decant_lds[i], v & m); } return 0u; }",
    needs: &[],
};

pub static STORE8: Helper = Helper {
    name: "decant_lds_store8",
    result: Ty::U32,
    text: "uint decant_lds_store8(uint a, uint v) { uint s = (a & 3u) * 8u; return decant_lds_store_bits(a, (v & 0xFFu) << s, 0xFFu << s); }",
    needs: &[&STORE_BITS],
};

pub static STORE16: Helper = Helper {
    name: "decant_lds_store16",
    result: Ty::U32,
    text: "uint decant_lds_store16(uint a, uint v) { uint s = (a & 2u) * 8u; return decant_lds_store_bits(a, (v & 0xFFFFu) << s, 0xFFFFu << s); }",
    needs: &[&STORE_BITS],
};

/// An atomic with an HLSL intrinsic: `uint name(uint a, uint v, uint w)` gives the old value.
macro_rules! intrinsic {
    ($id:ident, $name:literal, $call:literal) => {
        pub static $id: Helper = Helper {
            name: $name,
            result: Ty::U32,
            text: concat!(
                "uint ",
                $name,
                "(uint a, uint v, uint w) { uint i = a >> 2; uint o = 0u; if (i < decant_lds_words) { ",
                $call,
                "; } return o; }"
            ),
            needs: &[],
        };
    };
}

intrinsic!(ADD, "decant_lds_add", "InterlockedAdd(decant_lds[i], v, o)");
intrinsic!(SUB, "decant_lds_sub", "InterlockedAdd(decant_lds[i], 0u - v, o)");
intrinsic!(UMIN, "decant_lds_umin", "InterlockedMin(decant_lds[i], v, o)");
intrinsic!(UMAX, "decant_lds_umax", "InterlockedMax(decant_lds[i], v, o)");
intrinsic!(AND, "decant_lds_and", "InterlockedAnd(decant_lds[i], v, o)");
intrinsic!(OR, "decant_lds_or", "InterlockedOr(decant_lds[i], v, o)");
intrinsic!(XOR, "decant_lds_xor", "InterlockedXor(decant_lds[i], v, o)");
intrinsic!(
    XCHG,
    "decant_lds_xchg",
    "InterlockedExchange(decant_lds[i], v, o)"
);
pub static CMPST: Helper = Helper {
    name: "decant_lds_cmpst",
    result: Ty::U32,
    text: "uint decant_lds_cmpst(uint a, uint v, uint w) { uint i = a >> 2; uint o = v; if (i < decant_lds_words) { InterlockedCompareExchange(decant_lds[i], v, w, o); } return o; }",
    needs: &[],
};

/// Maximum compare-and-exchange retry iterations for emulated LDS atomics.
pub const LDS_CAS_MAX_ITERS: u32 = 1024;

/// An atomic without an HLSL intrinsic: a bounded compare and exchange loop. `$new` is the new
/// value from the current one `c` and the operands `v` and `w`.
macro_rules! cas {
    ($id:ident, $name:literal, $new:literal) => {
        pub static $id: Helper = Helper {
            name: $name,
            result: Ty::U32,
            text: concat!(
                "uint ",
                $name,
                "(uint a, uint v, uint w) { uint i = a >> 2; uint o = 0u; if (i < decant_lds_words) { uint c = decant_lds[i]; o = c; [loop] for (uint decant_cas = 0u; decant_cas < 1024u; ++decant_cas) { uint n = ",
                $new,
                "; uint p; InterlockedCompareExchange(decant_lds[i], c, n, p); if (p == c) { o = c; break; } c = p; o = c; } } return o; }"
            ),
            needs: &[],
        };
    };
}

cas!(RSUB, "decant_lds_rsub", "v - c");
// Wrapping increment and decrement (Mesa maps the wrapping NIR atomics to them).
cas!(INC, "decant_lds_inc", "c >= v ? 0u : c + 1u");
cas!(DEC, "decant_lds_dec", "(c == 0u || c > v) ? v : c - 1u");
cas!(SMIN, "decant_lds_smin", "(int)v < (int)c ? v : c");
cas!(SMAX, "decant_lds_smax", "(int)v > (int)c ? v : c");
cas!(MSKOR, "decant_lds_mskor", "(c & ~v) | w");
cas!(FADD, "decant_lds_fadd", "asuint(asfloat(c) + asfloat(v))");
cas!(FMIN, "decant_lds_fmin", "asuint(min(asfloat(c), asfloat(v)))");
cas!(FMAX, "decant_lds_fmax", "asuint(max(asfloat(c), asfloat(v)))");

/// `s_barrier` (the raw `Stmt::Barrier`, resolved by the stage): every wave of the group waits
/// here, and the group shared memory written before it is visible after it.
pub static BARRIER: Helper = Helper {
    name: "decant_barrier",
    result: Ty::U32,
    text: "uint decant_barrier() { GroupMemoryBarrierWithGroupSync(); return 0u; }",
    needs: &[],
};

/// Whether `func` touches group shared memory, so that the stage declares it.
pub fn uses_lds(func: &Function) -> bool {
    fn walk(e: &Expr) -> bool {
        match e {
            Expr::Call(h, args) => h.name.starts_with("decant_lds_") || args.iter().any(walk),
            Expr::Un(_, a) => walk(a),
            Expr::Bin(_, a, b) => walk(a) || walk(b),
            Expr::Select(c, a, b) => walk(c) || walk(a) || walk(b),
            _ => false,
        }
    }
    let mut found = false;
    func.for_each_stmt(&mut |s| {
        found |= match s {
            Stmt::Set { value, .. } => walk(value),
            Stmt::Effect { call, .. } => walk(call),
            _ => false,
        };
    });
    found
}

fn vreg(insn: &Insn, op: Operand) -> Result<u8, Refusal> {
    match op {
        Operand::Reg(Reg::V(n)) => Ok(n),
        _ => Err(form(insn, "operand")),
    }
}

fn data(insn: &Insn, i: usize, k: u8) -> Result<Expr, Refusal> {
    Ok(Expr::Var(Var::V(vreg(insn, insn.src[i])? + k)))
}

fn call(h: &'static Helper, args: Vec<Expr>) -> Expr {
    Expr::Call(h, args)
}

/// What one instruction does, by its name.
enum Kind {
    /// `count` dwords from one address.
    Read {
        count: u8,
    },
    /// Two groups of `dwords` from two addresses (`offset0` and `offset1` times `scale`).
    Read2 {
        dwords: u8,
        scale: u32,
    },
    /// A sub dword read: bytes, sign extension, and for the `d16` forms which half it replaces.
    ReadSmall {
        bytes: u8,
        signed: bool,
        half: Option<bool>,
    },
    Write {
        count: u8,
    },
    Write2 {
        dwords: u8,
        scale: u32,
    },
    /// A sub dword write of the low (or, with `hi`, the high) half of the data register.
    WriteSmall {
        bytes: u8,
        hi: bool,
    },
    Atomic {
        helper: &'static Helper,
        returns: bool,
        two: bool,
    },
}

fn kind(op: Op) -> Option<Kind> {
    use Kind::*;
    let atomic = |helper, returns, two| {
        Some(Atomic {
            helper,
            returns,
            two,
        })
    };
    match op {
        Op::ds_read_b32 => Some(Read { count: 1 }),
        Op::ds_read_b64 => Some(Read { count: 2 }),
        Op::ds_read_b96 => Some(Read { count: 3 }),
        Op::ds_read_b128 => Some(Read { count: 4 }),
        Op::ds_read2_b32 => Some(Read2 {
            dwords: 1,
            scale: 4,
        }),
        Op::ds_read2st64_b32 => Some(Read2 {
            dwords: 1,
            scale: 256,
        }),
        Op::ds_read2_b64 => Some(Read2 {
            dwords: 2,
            scale: 8,
        }),
        Op::ds_read2st64_b64 => Some(Read2 {
            dwords: 2,
            scale: 512,
        }),
        Op::ds_read_u8 => Some(ReadSmall {
            bytes: 1,
            signed: false,
            half: None,
        }),
        Op::ds_read_i8 => Some(ReadSmall {
            bytes: 1,
            signed: true,
            half: None,
        }),
        Op::ds_read_u16 => Some(ReadSmall {
            bytes: 2,
            signed: false,
            half: None,
        }),
        Op::ds_read_i16 => Some(ReadSmall {
            bytes: 2,
            signed: true,
            half: None,
        }),
        Op::ds_read_u8_d16 => Some(ReadSmall {
            bytes: 1,
            signed: false,
            half: Some(false),
        }),
        Op::ds_read_u8_d16_hi => Some(ReadSmall {
            bytes: 1,
            signed: false,
            half: Some(true),
        }),
        Op::ds_read_i8_d16 => Some(ReadSmall {
            bytes: 1,
            signed: true,
            half: Some(false),
        }),
        Op::ds_read_i8_d16_hi => Some(ReadSmall {
            bytes: 1,
            signed: true,
            half: Some(true),
        }),
        Op::ds_read_u16_d16 => Some(ReadSmall {
            bytes: 2,
            signed: false,
            half: Some(false),
        }),
        Op::ds_read_u16_d16_hi => Some(ReadSmall {
            bytes: 2,
            signed: false,
            half: Some(true),
        }),
        Op::ds_write_b32 => Some(Write { count: 1 }),
        Op::ds_write_b64 => Some(Write { count: 2 }),
        Op::ds_write_b96 => Some(Write { count: 3 }),
        Op::ds_write_b128 => Some(Write { count: 4 }),
        Op::ds_write2_b32 => Some(Write2 {
            dwords: 1,
            scale: 4,
        }),
        Op::ds_write2st64_b32 => Some(Write2 {
            dwords: 1,
            scale: 256,
        }),
        Op::ds_write2_b64 => Some(Write2 {
            dwords: 2,
            scale: 8,
        }),
        Op::ds_write2st64_b64 => Some(Write2 {
            dwords: 2,
            scale: 512,
        }),
        Op::ds_write_b8 => Some(WriteSmall {
            bytes: 1,
            hi: false,
        }),
        Op::ds_write_b8_d16_hi => Some(WriteSmall { bytes: 1, hi: true }),
        Op::ds_write_b16 => Some(WriteSmall {
            bytes: 2,
            hi: false,
        }),
        Op::ds_write_b16_d16_hi => Some(WriteSmall { bytes: 2, hi: true }),
        Op::ds_add_u32 => atomic(&ADD, false, false),
        Op::ds_add_rtn_u32 => atomic(&ADD, true, false),
        Op::ds_sub_u32 => atomic(&SUB, false, false),
        Op::ds_sub_rtn_u32 => atomic(&SUB, true, false),
        Op::ds_rsub_u32 => atomic(&RSUB, false, false),
        Op::ds_rsub_rtn_u32 => atomic(&RSUB, true, false),
        Op::ds_inc_u32 => atomic(&INC, false, false),
        Op::ds_inc_rtn_u32 => atomic(&INC, true, false),
        Op::ds_dec_u32 => atomic(&DEC, false, false),
        Op::ds_dec_rtn_u32 => atomic(&DEC, true, false),
        Op::ds_min_i32 => atomic(&SMIN, false, false),
        Op::ds_min_rtn_i32 => atomic(&SMIN, true, false),
        Op::ds_max_i32 => atomic(&SMAX, false, false),
        Op::ds_max_rtn_i32 => atomic(&SMAX, true, false),
        Op::ds_min_u32 => atomic(&UMIN, false, false),
        Op::ds_min_rtn_u32 => atomic(&UMIN, true, false),
        Op::ds_max_u32 => atomic(&UMAX, false, false),
        Op::ds_max_rtn_u32 => atomic(&UMAX, true, false),
        Op::ds_and_b32 => atomic(&AND, false, false),
        Op::ds_and_rtn_b32 => atomic(&AND, true, false),
        Op::ds_or_b32 => atomic(&OR, false, false),
        Op::ds_or_rtn_b32 => atomic(&OR, true, false),
        Op::ds_xor_b32 => atomic(&XOR, false, false),
        Op::ds_xor_rtn_b32 => atomic(&XOR, true, false),
        Op::ds_mskor_b32 => atomic(&MSKOR, false, true),
        Op::ds_mskor_rtn_b32 => atomic(&MSKOR, true, true),
        Op::ds_wrxchg_rtn_b32 => atomic(&XCHG, true, false),
        Op::ds_cmpst_b32 => atomic(&CMPST, false, true),
        Op::ds_cmpst_rtn_b32 => atomic(&CMPST, true, true),
        Op::ds_add_f32 => atomic(&FADD, false, false),
        Op::ds_add_rtn_f32 => atomic(&FADD, true, false),
        Op::ds_min_f32 => atomic(&FMIN, false, false),
        Op::ds_min_rtn_f32 => atomic(&FMIN, true, false),
        Op::ds_max_f32 => atomic(&FMAX, false, false),
        Op::ds_max_rtn_f32 => atomic(&FMAX, true, false),
        _ => None,
    }
}

/// A statement with a side effect: it runs in an active lane and is never removed.
fn effect(cx: &mut Lifter, dst: Option<Var>, call: Expr, guard: Guard) {
    cx.stmts.push(Stmt::Effect { dst, call, guard });
}

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    let Some(kind) = kind(insn.op) else {
        return Ok(false);
    };
    let Ext::Ds {
        offset0,
        offset1,
        gds,
    } = insn.ext
    else {
        return Err(form(insn, "ext"));
    };
    if gds {
        return Err(form(insn, "gds"));
    }
    let base = Expr::Var(Var::V(vreg(insn, insn.src[0])?));
    let at = |offset: u32| {
        if offset == 0 {
            base.clone()
        } else {
            Expr::bin(Bin::Add, base.clone(), Expr::U32(offset))
        }
    };
    let offset = u32::from(offset0) | (u32::from(offset1) << 8);
    match kind {
        Kind::Read { count } => {
            let dst = vreg(insn, insn.dst)?;
            // All reads first: the destination may overlap the address register.
            let values: Vec<Expr> = (0..count)
                .map(|k| call(&LOAD, vec![at(offset + 4 * u32::from(k))]))
                .collect();
            read_into(cx, dst, values);
        }
        Kind::Read2 { dwords, scale } => {
            let dst = vreg(insn, insn.dst)?;
            let mut values = Vec::new();
            for o in [offset0, offset1] {
                for k in 0..dwords {
                    values.push(call(
                        &LOAD,
                        vec![at(u32::from(o) * scale + 4 * u32::from(k))],
                    ));
                }
            }
            read_into(cx, dst, values);
        }
        Kind::ReadSmall {
            bytes,
            signed,
            half,
        } => {
            let dst = vreg(insn, insn.dst)?;
            let helper = if bytes == 1 { &LOAD8 } else { &LOAD16 };
            let mut value = call(helper, vec![at(offset)]);
            if signed {
                // Sign extension from the top bit of the read value.
                let shift = Expr::U32(32 - 8 * u32::from(bytes));
                value = Expr::bin(Bin::AShr, Expr::bin(Bin::Shl, value, shift.clone()), shift);
            }
            let old = Expr::Var(Var::V(dst));
            let value = match half {
                None => value,
                // The low half is written, the high half of the register stays.
                Some(false) => Expr::bin(
                    Bin::Or,
                    Expr::bin(Bin::And, old, Expr::U32(0xFFFF_0000)),
                    Expr::bin(Bin::And, value, Expr::U32(0xFFFF)),
                ),
                Some(true) => Expr::bin(
                    Bin::Or,
                    Expr::bin(Bin::And, old, Expr::U32(0xFFFF)),
                    Expr::bin(Bin::Shl, value, Expr::U32(16)),
                ),
            };
            cx.set_lane(Var::V(dst), value);
        }
        Kind::Write { count } => {
            for k in 0..count {
                let value = data(insn, 1, k)?;
                effect(
                    cx,
                    None,
                    call(&STORE, vec![at(offset + 4 * u32::from(k)), value]),
                    Guard::Exec,
                );
            }
        }
        Kind::Write2 { dwords, scale } => {
            for (i, o) in [(1, offset0), (2, offset1)] {
                for k in 0..dwords {
                    let value = data(insn, i, k)?;
                    effect(
                        cx,
                        None,
                        call(
                            &STORE,
                            vec![at(u32::from(o) * scale + 4 * u32::from(k)), value],
                        ),
                        Guard::Exec,
                    );
                }
            }
        }
        Kind::WriteSmall { bytes, hi } => {
            let mut value = data(insn, 1, 0)?;
            if hi {
                value = Expr::bin(Bin::LShr, value, Expr::U32(16));
            }
            let helper = if bytes == 1 { &STORE8 } else { &STORE16 };
            effect(cx, None, call(helper, vec![at(offset), value]), Guard::Exec);
        }
        Kind::Atomic {
            helper,
            returns,
            two,
        } => {
            let v = data(insn, 1, 0)?;
            let w = if two { data(insn, 2, 0)? } else { Expr::U32(0) };
            let dst = if returns {
                Some(Var::V(vreg(insn, insn.dst)?))
            } else {
                None
            };
            effect(cx, dst, call(helper, vec![at(offset), v, w]), Guard::Exec);
        }
    }
    Ok(true)
}

/// Writes `values` to the registers from `dst` on. The values are read into temporaries first
/// when a destination register is also an operand of a later value.
fn read_into(cx: &mut Lifter, dst: u8, values: Vec<Expr>) {
    if values.len() == 1 {
        let v = values.into_iter().next().expect("one value");
        cx.set_lane(Var::V(dst), v);
        return;
    }
    let tmps: Vec<Var> = values
        .into_iter()
        .map(|v| {
            let t = cx.tmp();
            cx.set(t, v);
            t
        })
        .collect();
    for (k, t) in tmps.into_iter().enumerate() {
        cx.set_lane(Var::V(dst + k as u8), Expr::Var(t));
    }
}
