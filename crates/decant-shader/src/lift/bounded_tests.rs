//! Synthetic lane exchanges and byte addressing. No archive data is used here.

use std::collections::HashMap;

use crate::api::*;
use crate::asm::*;
use crate::decode::{Ext, Form, Insn, Operand};
use crate::ir::{Bin, Expr, Guard, Stmt, Var};
use crate::tables::Op;

const NONE: Operand = Operand::None;

fn ds(op: Op, dst: Operand, value: Operand, pattern: u16) -> Insn {
    Insn {
        pc: 0,
        len: 0,
        op,
        form: Form::Plain,
        dst,
        dst2: NONE,
        src: [value, v(4), NONE, NONE],
        ext: Ext::Ds {
            offset0: pattern as u8,
            offset1: (pattern >> 8) as u8,
            gds: false,
        },
    }
}

fn program(body: impl FnOnce(&mut Asm)) -> StageProgram {
    let mut a = Asm::new();
    body(&mut a);
    a.push(sopp(Op::s_endpgm, 0));
    StageProgram {
        threads: [64, 1, 1],
        lds_bytes: 256,
        ..StageProgram::new(Stage::Compute, a.finish())
    }
}

fn text(p: &StageProgram) -> String {
    crate::translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{r}"))
    .stages[0]
        .hlsl
        .clone()
}

fn compile(text: &str) {
    let dxc = match decant_dxc::Dxc::load() {
        Ok(dxc) => dxc,
        Err(why) => {
            eprintln!("DXC test skipped: {why}");
            return;
        }
    };
    let result = dxc
        .compile(text, "main", "cs_6_6", &[])
        .unwrap_or_else(|why| panic!("{why}\n{text}"));
    assert_eq!(&result[..4], b"DXBC");
}

#[test]
fn quad_swizzles_use_bounded_group_scratch_and_compile() {
    for pattern in [0x8000, 0x8055, 0x80aa, 0x801b, 0x80e4] {
        let p = program(|a| {
            a.push(ds(Op::ds_swizzle_b32, v(4), v(0), pattern));
            a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
        });
        let out = text(&p);
        assert!(out.contains("source < decant_swizzle_threads"), "{out}");
        assert!(out.contains("thread & ~3u"), "{out}");
        assert_eq!(out.matches("GroupMemoryBarrierWithGroupSync();").count(), 2);
        compile(&out);
    }
}

#[test]
fn quad_swizzles_mask_sources_and_destinations_in_partial_groups() {
    let mut p = program(|a| {
        a.push(ds(Op::ds_swizzle_b32, v(4), v(0), 0x801b));
        a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
    });
    p.threads = [3, 2, 1];
    let f = crate::lift_program(&p).unwrap();
    let mut exchange = false;
    let mut result = false;
    f.for_each_stmt(&mut |stmt| match stmt {
        Stmt::Effect {
            call: Expr::Call(h, args),
            guard,
            ..
        } if h.name == "decant_swizzle" => {
            assert_eq!(*guard, Guard::Always);
            assert_eq!(
                args[2],
                Expr::select(Expr::Var(Var::Exec), Expr::Var(Var::V(0)), Expr::U32(0))
            );
            exchange = true;
        }
        Stmt::Set {
            dst: Var::V(4),
            guard,
            ..
        } => {
            assert_eq!(*guard, Guard::Exec);
            result = true;
        }
        _ => {}
    });
    assert!(exchange && result);
    let out = text(&p);
    assert!(out.contains("decant_swizzle_threads = 6u"), "{out}");
    compile(&out);
}

#[test]
fn quad_swizzles_lower_bypasses_to_wave_and_reject_gds() {
    let p = program(|a| {
        let end = a.label();
        a.branch(Op::s_cbranch_scc0, end);
        a.push(ds(Op::ds_swizzle_b32, v(4), v(0), 0x801b));
        a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
        a.bind(end);
    });
    let out = text(&p);
    assert!(out.contains("[WaveSize(32)]"), "{out}");
    assert!(out.contains("decant_swizzle_wave("), "{out}");
    assert!(out.contains("WaveActiveBallot(exec)"), "{out}");
    assert!(
        out.contains("WaveReadLaneAt(value, valid ? source : lane)"),
        "{out}"
    );
    assert!(!out.contains("GroupMemoryBarrierWithGroupSync()"), "{out}");
    compile(&out);
    let p = program(|a| {
        let mut i = ds(Op::ds_swizzle_b32, v(4), v(0), 0x801b);
        if let Ext::Ds { gds, .. } = &mut i.ext {
            *gds = true;
        }
        a.push(i);
    });
    assert_eq!(
        crate::lift_program(&p).unwrap_err().key(),
        "exec:cross-lane"
    );
}

#[test]
fn writelane_ignores_exec_and_preserves_other_lanes() {
    let p = program(|a| {
        a.push(vop1(Op::v_mov_b32, v(4), v(0)));
        a.push(sop1(Op::s_mov_b64, EXEC, int(0)));
        a.push(vop3(Op::v_writelane_b32_e64, v(4), s(2), int(63), NONE));
        a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
        a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
    });
    let f = crate::lift_program(&p).unwrap();
    let mut found = false;
    f.for_each_stmt(&mut |stmt| {
        if let Stmt::Set {
            dst: Var::V(4),
            value: Expr::Call(h, args),
            guard,
        } = stmt
        {
            assert_eq!(h.name, "decant_write_lane");
            assert_eq!(*guard, Guard::Always);
            assert_eq!(args[2], Expr::Var(Var::V(4)));
            found = true;
        }
    });
    assert!(found);
    let out = text(&p);
    assert!(out.contains("[WaveSize(64)]"));
    assert!(out.contains("WaveGetLaneIndex() == lane ? value : old"));
    compile(&out);
}

fn eval(e: &Expr, vars: &HashMap<Var, u32>, bytes: &[u8]) -> u32 {
    match e {
        Expr::U32(n) => *n,
        Expr::Var(v) => vars[v],
        Expr::Bin(op, a, b) => {
            let (a, b) = (eval(a, vars, bytes), eval(b, vars, bytes));
            match op {
                Bin::Add => a.wrapping_add(b),
                Bin::Mul => a.wrapping_mul(b),
                Bin::And => a & b,
                Bin::Or => a | b,
                Bin::Shl => a << (b & 31),
                Bin::LShr => a >> (b & 31),
                _ => panic!("unexpected operation {op:?}"),
            }
        }
        Expr::VectorLoad { base, offset, .. } => {
            let at = (vars[&Var::V(*base)].wrapping_add(eval(offset, vars, bytes)) & !3) as usize;
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        }
        _ => panic!("unexpected expression {e:?}"),
    }
}

#[test]
fn ushort_zero_extends_at_every_byte_alignment_with_address_aliases() {
    let p = program(|_| {});
    let graph = crate::cfg::build(&p.code).unwrap();
    let masks = crate::exec::analyse(&graph).unwrap();
    let bytes = [0x19, 0x80, 0xff, 0xab, 0xcd, 0xef, 0x87, 0x65, 0, 0, 0, 0];
    for dst in [2, 3, 4] {
        for address in 0..4 {
            for offset in 0..4 {
                let mut cx = super::Lifter::new(Stage::Compute, &masks);
                assert!(
                    super::smem::lift_global(
                        &mut cx,
                        &global_load(Op::global_load_ushort, dst, 2, offset)
                    )
                    .unwrap()
                );
                for active in [false, true] {
                    let mut vars =
                        HashMap::from([(Var::V(2), address), (Var::V(3), 0), (Var::V(4), 99)]);
                    let old = vars[&Var::V(dst)];
                    for stmt in &cx.stmts {
                        if let Stmt::Set { dst, value, guard } = stmt
                            && (*guard == Guard::Always || active)
                        {
                            let value = eval(value, &vars, &bytes);
                            vars.insert(*dst, value);
                        }
                    }
                    let at = (address + u32::from(offset)) as usize;
                    let expected = u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
                    assert_eq!(vars[&Var::V(dst)], if active { expected } else { old });
                }
            }
        }
    }
}

#[test]
fn ushort_from_traced_code_compiles_and_unknown_addresses_stay_refused() {
    let p = program(|a| {
        a.push(sop1(Op::s_getpc_b64, s(22), int(0)));
        a.push(vop2(Op::v_add_u32, v(2), s(22), v(0)));
        a.push(vop1(Op::v_mov_b32, v(3), int(0)));
        a.push(global_load(Op::global_load_ushort, 4, 2, 3));
        a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
    });
    let out = text(&p);
    assert!(out.contains("255u"), "{out}");
    compile(&out);
    let p = program(|a| {
        a.push(global_load(Op::global_load_ushort, 4, 2, 0));
        a.push(ds(Op::ds_write_b32, NONE, v(0), 0));
    });
    let r = crate::translate(
        &[p],
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_err();
    assert_eq!(r.key(), "resource:raw-address");
}
