//! Messages and barriers (`s_sendmsg`, `s_barrier`). The lifter only records them as raw
//! statements (`Stmt::Message`, `Stmt::Barrier`): what a message means depends on the stage
//! (message 9 allocates the outputs of a mesh group, 0x22 emits a geometry vertex), and a
//! barrier is only needed where group shared memory is. So `crate::stage` resolves or drops
//! them, and the printer refuses one that is left.

use super::Lifter;
use crate::decode::{Ext, Insn};
use crate::ir::Stmt;
use crate::refusal::Refusal;
use crate::tables::Op;

pub fn lift(cx: &mut Lifter, insn: &Insn) -> Result<bool, Refusal> {
    match (insn.op, insn.ext) {
        // The message number is the immediate. The payload is whatever M0 holds.
        (Op::s_sendmsg, Ext::Imm16(id)) => cx.stmts.push(Stmt::Message { id, pc: insn.pc }),
        (Op::s_barrier, _) => cx.stmts.push(Stmt::Barrier { pc: insn.pc }),
        // `s_sendmsghalt` stops the wave after the message. No program of the title has it.
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Stage;
    use crate::asm::sopp;
    use crate::exec::Masks;

    #[test]
    fn a_message_and_a_barrier_become_raw_statements() {
        let masks = Masks::default();
        let mut cx = Lifter::new(Stage::Mesh, &masks);
        assert!(lift(&mut cx, &sopp(Op::s_sendmsg, 0x22)).unwrap());
        assert!(lift(&mut cx, &sopp(Op::s_barrier, 0)).unwrap());
        assert!(!lift(&mut cx, &sopp(Op::s_endpgm, 0)).unwrap());
        assert!(matches!(
            cx.stmts.as_slice(),
            [Stmt::Message { id: 0x22, .. }, Stmt::Barrier { .. }]
        ));
    }
}
