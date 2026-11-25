//! The compute stage: the registers a compute program finds at entry, and what it owes at the
//! end.
//!
//! Entry state, measured on the compute programs of the title:
//!
//! * `v0`, `v1`, `v2` hold the thread id in the group, x, y and z. Programs read `v0` alone,
//!   `v0` and `v1`, or all three, following the dimensions of their thread group.
//! * The user data registers start at `s4` and hold the root layout of the container
//!   (`StageProgram::user_data`). The three registers after the last one hold the group id x,
//!   y and z, always all three: programs whose group is wide in x only still read the third one
//!   when they use the group id z, and programs that use only y read the second.
//! * EXEC holds every lane of the wave. A wave at the end of a group that is no multiple of
//!   the wave size has fewer lanes, and HLSL has no invocation for the missing threads, which is
//!   the same.
//!
//! The thread group size comes from `PSV0` (`StageProgram::threads`), the size of the group
//! shared memory from the FSSM block (`StageProgram::lds_bytes`): measured on the compute
//! programs, it is not zero in exactly the programs with LDS instructions, never below the
//! highest constant offset they use, and at most 32 KiB.
//!
//! Group shared memory (`ds_*`) is `crate::lift::ds`. A compute program has no exports.

use super::refuse;
use crate::api::{StageProgram, UserDataKind};
use crate::ir::{Expr, Function, Guard, Input, Stmt, Var};
use crate::refusal::Refusal;

/// The first register after the user data: the group id x. The group id y and z follow.
pub fn group_id_register(program: &StageProgram) -> u8 {
    program
        .user_data
        .iter()
        .map(|slot| {
            let width = match slot.kind {
                UserDataKind::Constants { count, .. } => count,
                UserDataKind::RootSrv { .. } => 2,
                UserDataKind::Table | UserDataKind::ResourceHeap | UserDataKind::SamplerHeap => 1,
            };
            u32::from(slot.sgpr) + width
        })
        .max()
        .map_or(program.stage.first_user_sgpr(), |end| end.min(104) as u8)
}

fn set(dst: Var, value: Expr) -> Stmt {
    Stmt::Set {
        dst,
        value,
        guard: Guard::Always,
    }
}

pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    let mut out = vec![set(Var::Exec, Expr::Bool(true))];
    for axis in 0..3u8 {
        out.push(set(
            Var::V(axis),
            Expr::Input(Input::LocalThreadId { axis }),
        ));
    }
    let first = group_id_register(program);
    for axis in 0..3u8 {
        out.push(set(
            Var::S(first + axis),
            Expr::Input(Input::GroupId { axis }),
        ));
    }
    out
}

pub fn finish(program: &StageProgram, func: &mut Function) -> Result<(), Refusal> {
    if program.threads.contains(&0) {
        return Err(refuse(
            "compute-threads-unknown",
            format!("{:?}", program.threads),
        ));
    }
    let mut export = None;
    func.for_each_stmt(&mut |s| {
        if let Stmt::Export(e) = s {
            export.get_or_insert(e.pc);
        }
    });
    if let Some(pc) = export {
        return Err(refuse("compute-export", "").at(pc));
    }
    // A barrier holds every wave of the group, whatever EXEC says: the console compiler only
    // places it where the whole group arrives.
    func.for_each_block_mut(&mut |stmts| {
        for s in stmts.iter_mut() {
            if matches!(s, Stmt::Barrier { .. }) {
                *s = Stmt::Effect {
                    dst: None,
                    call: Expr::Call(&crate::lift::ds::BARRIER, Vec::new()),
                    guard: Guard::Always,
                };
            }
        }
    });
    if crate::lift::ds::uses_lds(func) {
        // The allocation the console compiler made for the group (FSSM byte 0x10C), which no
        // access exceeds. Without it, the largest array Direct3D 12 allows.
        func.groupshared_dwords = match program.lds_bytes {
            0 => crate::lift::ds::LDS_DWORDS,
            bytes => bytes.div_ceil(4).min(crate::lift::ds::LDS_DWORDS),
        };
    }
    Ok(())
}
