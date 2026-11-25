//! A listing of decoded instructions, for the census, for refusal messages and for tests. The
//! text is for people. Nothing parses it.

use std::fmt;

use super::{Ext, Form, Insn, Operand, Reg};
use crate::tables::Format;

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::S(n) => write!(f, "s{n}"),
            Reg::Ttmp(n) => write!(f, "ttmp{n}"),
            Reg::V(n) => write!(f, "v{n}"),
            Reg::VccLo => f.write_str("vcc_lo"),
            Reg::VccHi => f.write_str("vcc_hi"),
            Reg::M0 => f.write_str("m0"),
            Reg::Null => f.write_str("null"),
            Reg::ExecLo => f.write_str("exec_lo"),
            Reg::ExecHi => f.write_str("exec_hi"),
            Reg::Scc => f.write_str("scc"),
            Reg::Vccz => f.write_str("vccz"),
            Reg::Execz => f.write_str("execz"),
            Reg::SharedBase => f.write_str("src_shared_base"),
            Reg::SharedLimit => f.write_str("src_shared_limit"),
            Reg::PrivateBase => f.write_str("src_private_base"),
            Reg::PrivateLimit => f.write_str("src_private_limit"),
            Reg::PopsExitingWaveId => f.write_str("src_pops_exiting_wave_id"),
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::None => f.write_str("_"),
            Operand::Reg(r) => r.fmt(f),
            Operand::Int(v) => write!(f, "{v}"),
            Operand::Float(c) => write!(f, "{:?}", c.f32()),
            Operand::Lit(v) => write!(f, "{v:#x}"),
            Operand::Unknown(c) => write!(f, "?{c}"),
        }
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op.name())?;
        match self.form {
            Form::Plain => {}
            Form::AsVop3 => f.write_str("_e64")?,
            Form::Sdwa => f.write_str("_sdwa")?,
            Form::Dpp16 | Form::Dpp8 => f.write_str("_dpp")?,
        }
        let mut sep = " ";
        for op in [self.dst, self.dst2]
            .into_iter()
            .chain(self.src)
            .filter(|o| *o != Operand::None)
        {
            write!(f, "{sep}{op}")?;
            sep = ", ";
        }
        match self.ext {
            Ext::None => Ok(()),
            Ext::Imm16(imm) => match self.branch_target() {
                Some(target) => write!(f, "{sep}-> {target}"),
                None => write!(f, "{sep}{imm:#x}"),
            },
            Ext::Valu {
                abs,
                neg,
                opsel,
                omod,
                clamp,
            } => {
                if abs != 0 || neg != 0 || opsel != 0 || omod != 0 || clamp {
                    write!(
                        f,
                        " ; abs={abs} neg={neg} opsel={opsel} omod={omod} clamp={}",
                        u8::from(clamp)
                    )?;
                }
                Ok(())
            }
            Ext::VintrpE64 {
                attr,
                chan,
                high,
                abs,
                neg,
                opsel,
                omod,
                clamp,
            } => write!(
                f,
                " ; attr={attr}.{chan} high={} abs={abs} neg={neg} opsel={opsel} omod={omod} clamp={}",
                u8::from(high),
                u8::from(clamp)
            ),
            Ext::Sdwa(x) => write!(f, " ; sdwa={x:#010x}"),
            Ext::Dpp8 {
                word,
                fetch_inactive,
            } => {
                write!(f, " ; dpp8={word:#010x} fi={}", u8::from(fetch_inactive))
            }
            Ext::Dpp(x) => write!(f, " ; dpp={x:#010x}"),
            Ext::Smem {
                offset,
                soffset,
                glc,
                dlc,
            } => write!(
                f,
                " ; offset={offset:#x} soffset={soffset} glc={} dlc={}",
                u8::from(glc),
                u8::from(dlc)
            ),
            Ext::Buf {
                offset,
                offen,
                idxen,
                glc,
                slc,
                dlc,
                lds,
                tfe,
                format,
            } => {
                write!(
                    f,
                    " ; offset={offset:#x} offen={} idxen={}",
                    u8::from(offen),
                    u8::from(idxen)
                )?;
                if self.format() == Format::Mtbuf {
                    write!(f, " format={format}")?;
                }
                for (name, on) in [
                    ("glc", glc),
                    ("slc", slc),
                    ("dlc", dlc),
                    ("lds", lds),
                    ("tfe", tfe),
                ] {
                    if on {
                        write!(f, " {name}")?;
                    }
                }
                Ok(())
            }
            Ext::Mimg(m) => {
                write!(f, " ; dmask={:#x} dim={}", m.dmask, m.dim)?;
                if m.nsa_words != 0 {
                    write!(f, " nsa={:?}", &m.nsa[..usize::from(m.nsa_words) * 4])?;
                }
                for (name, on) in [
                    ("unrm", m.unrm),
                    ("glc", m.glc),
                    ("slc", m.slc),
                    ("dlc", m.dlc),
                    ("r128", m.r128),
                    ("a16", m.a16),
                    ("d16", m.d16),
                    ("tfe", m.tfe),
                    ("lwe", m.lwe),
                ] {
                    if on {
                        write!(f, " {name}")?;
                    }
                }
                Ok(())
            }
            Ext::Ds {
                offset0,
                offset1,
                gds,
            } => write!(
                f,
                " ; offset0={offset0} offset1={offset1} gds={}",
                u8::from(gds)
            ),
            Ext::Exp {
                enabled,
                target,
                compressed,
                done,
                valid_mask,
            } => {
                write!(f, " ; target={target} en={enabled:#x}")?;
                for (name, on) in [("compr", compressed), ("done", done), ("vm", valid_mask)] {
                    if on {
                        write!(f, " {name}")?;
                    }
                }
                Ok(())
            }
            Ext::Vintrp { attr, chan } => write!(
                f,
                " ; attr{attr}.{}",
                ["x", "y", "z", "w"][usize::from(chan & 3)]
            ),
            Ext::Flat(w0, w1) => write!(f, " ; {w0:#010x} {w1:#010x}"),
        }
    }
}

/// A listing: one line per instruction, with its word index.
pub fn listing(insns: &[Insn]) -> String {
    let mut out = String::new();
    for insn in insns {
        out.push_str(&format!("{:5}: {insn}\n", insn.pc));
    }
    out
}
