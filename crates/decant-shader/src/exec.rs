//! Lane masks.
//!
//! The machine runs up to 64 lanes at once and keeps one bit per lane in EXEC, in VCC and in
//! scalar registers that save or combine masks. The translation runs one lane per shader
//! invocation, so a mask becomes one boolean: this lane's bit. That is exact as long as the
//! program only moves and combines masks bit by bit, which is all that compiled HLSL does
//! outside the wave intrinsics.
//!
//! What this file decides is where a scalar register holds a mask. Measured on the title's
//! programs: the wave size differs per program. A 64 lane program keeps a mask in
//! a register pair and combines masks with the `_b64` instructions. A 32 lane program keeps it
//! in ONE register and uses the same `_b32` instructions for masks and for numbers. The compiler
//! also reuses a register for a mask in one place and for a number in another, and uses VCC as a
//! plain scratch register. So the answer is per instruction, not per program.
//!
//! How it is found:
//!
//! 1. The wave size, from the width of the instructions that name EXEC or VCC.
//! 2. For every instruction, which scalar registers it reads and writes and in which role:
//!    surely a mask (a compare result, a saved EXEC, a select operand), surely a number
//!    (arithmetic, an address, a descriptor), or whatever its partners are (a bitwise instruction
//!    of the wave's width: its operands and its result have one common kind).
//! 3. Kinds flow forward from the writes: at every point, the kinds each register may hold. A
//!    bitwise instruction takes the kind its sources have in common, and gives it to its
//!    result. Kinds also flow backward from the reads: a result that only masks are read from
//!    is a mask. This repeats until nothing changes.
//! 4. A read that is sure of its kind is right when ONE value of that kind reaches it. The
//!    compiler reuses registers, so a stale value of the other kind often reaches the same read
//!    along a path the program never takes (two branches on the same condition). All values
//!    that really arrive are one value of the source program and have one kind. Only a read
//!    that no value of its kind reaches is refused: `exec:mask-as-data` for a mask read as a
//!    number, `exec:data-as-mask` for a number used as a mask. A bitwise instruction whose
//!    sources and readers leave both kinds open is refused as `exec:mixed-pair`.
//!
//! Scalars parked in vector lanes: when a program runs out of scalar registers it puts one into
//! a lane of a vector register with `v_writelane_b32` and takes it back with `v_readlane_b32`.
//! Such a place is followed like a register, so a parked mask comes back as a mask.
//! A lane write into vector data can also read a whole mask word. The compute lifter collects
//! its bits with a ballot when the graph has one block. Literal zero has the same mask role
//! as inline zero. Literal all ones has that role for a 32 bit mask word.
//!
//! The lifter asks through [`Masks::is_mask`] for the instruction it is at. `lift::wave` moves
//! that position with [`Masks::enter`] before any rule runs.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::cfg::{Cfg, Cond, Term};
use crate::decode::{DppCtrl, Ext, Form, Insn, Operand, Reg};
use crate::refusal::{Refusal, RefusalKind};
use crate::tables::{Base, Fixed, Format, Op, Slot};

/// The scalar registers the typing follows: `s0` to `s105`, then the two halves of VCC. This is
/// the number of the low half of VCC, for [`Masks::reads_mask`] and [`Masks::writes_mask`].
pub const VCC: u8 = 106;
const REGS: usize = 108;

/// A set of followed registers.
type Set = u128;

fn bit(reg: u8) -> Set {
    1 << reg
}

/// A bitwise operation on masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Logic {
    Mov,
    Not,
    And,
    Or,
    Xor,
    /// Not the first operand, and the second.
    Andn1,
    /// The first operand, and not the second.
    Andn2,
    Orn1,
    Orn2,
    Nand,
    Nor,
    Xnor,
}

/// What a scalar instruction does when its operands are masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaskOp {
    /// The result is the operation on the sources.
    Logic(Logic),
    /// The first source when SCC is set, else the second.
    Select,
    /// The result is the old EXEC, and EXEC becomes the operation on the source and EXEC.
    SaveExec(Logic),
    /// The result and EXEC both become the operation on the source and EXEC.
    WrExec(Logic),
    /// Whole quad mode: every lane of a quad that has a lane in the source.
    Wqm,
}

/// The mask meaning of a scalar instruction and the wave size it is for (32 or 64 lanes).
pub fn mask_op(op: Op) -> Option<(MaskOp, u8)> {
    use Logic::*;
    use MaskOp::{Logic as L, SaveExec as S, Select, Wqm, WrExec as W};
    Some(match op {
        Op::s_mov_b32 => (L(Mov), 32),
        Op::s_not_b32 => (L(Not), 32),
        Op::s_and_b32 => (L(And), 32),
        Op::s_or_b32 => (L(Or), 32),
        Op::s_xor_b32 => (L(Xor), 32),
        Op::s_andn2_b32 => (L(Andn2), 32),
        Op::s_orn2_b32 => (L(Orn2), 32),
        Op::s_nand_b32 => (L(Nand), 32),
        Op::s_nor_b32 => (L(Nor), 32),
        Op::s_xnor_b32 => (L(Xnor), 32),
        Op::s_cselect_b32 => (Select, 32),
        Op::s_and_saveexec_b32 => (S(And), 32),
        Op::s_or_saveexec_b32 => (S(Or), 32),
        Op::s_xor_saveexec_b32 => (S(Xor), 32),
        Op::s_andn1_saveexec_b32 => (S(Andn1), 32),
        Op::s_andn2_saveexec_b32 => (S(Andn2), 32),
        Op::s_orn1_saveexec_b32 => (S(Orn1), 32),
        Op::s_orn2_saveexec_b32 => (S(Orn2), 32),
        Op::s_nand_saveexec_b32 => (S(Nand), 32),
        Op::s_nor_saveexec_b32 => (S(Nor), 32),
        Op::s_xnor_saveexec_b32 => (S(Xnor), 32),
        Op::s_andn1_wrexec_b32 => (W(Andn1), 32),
        Op::s_andn2_wrexec_b32 => (W(Andn2), 32),
        Op::s_wqm_b32 => (Wqm, 32),
        Op::s_mov_b64 => (L(Mov), 64),
        Op::s_not_b64 => (L(Not), 64),
        Op::s_and_b64 => (L(And), 64),
        Op::s_or_b64 => (L(Or), 64),
        Op::s_xor_b64 => (L(Xor), 64),
        Op::s_andn2_b64 => (L(Andn2), 64),
        Op::s_orn2_b64 => (L(Orn2), 64),
        Op::s_nand_b64 => (L(Nand), 64),
        Op::s_nor_b64 => (L(Nor), 64),
        Op::s_xnor_b64 => (L(Xnor), 64),
        Op::s_cselect_b64 => (Select, 64),
        Op::s_and_saveexec_b64 => (S(And), 64),
        Op::s_or_saveexec_b64 => (S(Or), 64),
        Op::s_xor_saveexec_b64 => (S(Xor), 64),
        Op::s_andn1_saveexec_b64 => (S(Andn1), 64),
        Op::s_andn2_saveexec_b64 => (S(Andn2), 64),
        Op::s_orn1_saveexec_b64 => (S(Orn1), 64),
        Op::s_orn2_saveexec_b64 => (S(Orn2), 64),
        Op::s_nand_saveexec_b64 => (S(Nand), 64),
        Op::s_nor_saveexec_b64 => (S(Nor), 64),
        Op::s_xnor_saveexec_b64 => (S(Xnor), 64),
        Op::s_andn1_wrexec_b64 => (W(Andn1), 64),
        Op::s_andn2_wrexec_b64 => (W(Andn2), 64),
        Op::s_wqm_b64 => (Wqm, 64),
        _ => return None,
    })
}

/// A scalar instruction that reads its first source as one number of as many bits as the wave
/// has lanes: a count, a search, a bit test. On a mask it looks at other lanes. The result is
/// the wave size the instruction is for.
pub fn mask_number(op: Op) -> Option<u8> {
    Some(match op {
        Op::s_bcnt0_i32_b32
        | Op::s_bcnt1_i32_b32
        | Op::s_ff0_i32_b32
        | Op::s_ff1_i32_b32
        | Op::s_flbit_i32_b32
        | Op::s_bitcmp0_b32
        | Op::s_bitcmp1_b32 => 32,
        Op::s_bcnt0_i32_b64
        | Op::s_bcnt1_i32_b64
        | Op::s_ff0_i32_b64
        | Op::s_ff1_i32_b64
        | Op::s_flbit_i32_b64
        | Op::s_bitcmp0_b64
        | Op::s_bitcmp1_b64 => 64,
        _ => return None,
    })
}

/// A compare of a register with zero: on a mask it asks whether any lane is set. The result is
/// the source index of the register, the wave size the compare is for, and whether the
/// compare is "equal" (no lane) or "not equal" (some lane).
pub fn zero_test(insn: &Insn) -> Option<(usize, u8, bool)> {
    let (lanes, equal) = match insn.op {
        Op::s_cmp_eq_u32 | Op::s_cmp_eq_i32 => (32, true),
        Op::s_cmp_lg_u32 | Op::s_cmp_lg_i32 => (32, false),
        Op::s_cmp_eq_u64 => (64, true),
        Op::s_cmp_lg_u64 => (64, false),
        _ => return None,
    };
    match (insn.src[0], insn.src[1]) {
        (Operand::Reg(_), Operand::Int(0)) => Some((0, lanes, equal)),
        (Operand::Int(0), Operand::Reg(_)) => Some((1, lanes, equal)),
        _ => None,
    }
}

/// The instructions that save EXEC to their destination and then combine EXEC with their source.
pub fn saveexec(op: Op) -> bool {
    matches!(mask_op(op), Some((MaskOp::SaveExec(_), _)))
}

/// The followed register an operand starts at.
fn followed(op: Operand) -> Option<u8> {
    match op {
        Operand::Reg(Reg::S(n)) if n < VCC => Some(n),
        Operand::Reg(Reg::VccLo) => Some(VCC),
        Operand::Reg(Reg::VccHi) => Some(VCC + 1),
        _ => None,
    }
}

fn names(insn: &Insn, reg: Reg) -> bool {
    std::iter::once(insn.dst)
        .chain(insn.src)
        .any(|o| o == Operand::Reg(reg))
}

/// The role a location has in one instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Mask,
    Data,
    /// Whatever kind the other `Same` operands of the instruction with the same `half` have.
    Same,
    /// Read as a number when it is a number, as a whole mask when it is a mask. No evidence.
    Number,
    /// Read as a 32 bit word of a 64 lane mask or as a numeric scalar, writing a numeric scalar.
    MaskWord,
}

/// One location an instruction reads or writes: a followed register or a parked scalar.
#[derive(Clone, Copy, Debug)]
struct Touch {
    loc: u16,
    /// Which word of its operand the location is: 0 the low half of a mask, 1 the high half.
    half: u8,
    role: Role,
    def: bool,
    /// For `Role::Same`: the group of touches that share one kind.
    group: u32,
}

/// Vector registers a program only uses to park scalars: `v_writelane_b32` puts a scalar
/// register into one lane, `v_readlane_b32` takes it back, both with a constant lane, and no
/// other instruction names the register. Each (register, lane) is a location of its own, after
/// the followed registers.
#[derive(Default)]
struct Parking {
    slots: BTreeMap<(u8, u8), u16>,
}

impl Parking {
    /// The register and lane of a parking instruction, when its lane is a constant.
    fn place(insn: &Insn) -> Option<(u8, u8)> {
        let (reg, lane) = match insn.op {
            Op::v_writelane_b32_e64 => (insn.dst, insn.src[1]),
            Op::v_readlane_b32_e64 => (insn.src[0], insn.src[1]),
            _ => return None,
        };
        match (reg, lane) {
            (Operand::Reg(Reg::V(v)), Operand::Int(l)) if (0..64).contains(&l) => {
                Some((v, l as u8))
            }
            _ => None,
        }
    }

    fn find(cfg: &Cfg) -> Parking {
        let mut places = BTreeSet::new();
        let mut written = BTreeSet::new();
        // Vector registers any other instruction names, with the registers after them that the
        // instruction also touches (results and addresses of memory instructions, 64 bit pairs).
        // The parking register is often the very next one after the last data register.
        let mut other = [false; 256];
        let mut mark = |op: Operand, count: usize| {
            if let Operand::Reg(Reg::V(first)) = op {
                for r in other.iter_mut().skip(usize::from(first)).take(count) {
                    *r = true;
                }
            }
        };
        let mut relative = false;
        for insn in cfg.insns() {
            if let Some(place) = Parking::place(insn) {
                places.insert(place);
                if insn.op == Op::v_writelane_b32_e64 {
                    written.insert(place);
                }
                continue;
            }
            // A relative move reaches a register chosen at run time, inside the per lane array
            // it indexes. See `parks_beside_relative_moves` for why that is never a parking place.
            relative |= insn.op.name().starts_with("v_movrel");
            let (dst, dst2, src) = vector_spans(insn);
            mark(insn.dst, dst);
            mark(insn.dst2, dst2);
            for (s, count) in insn.src.into_iter().zip(src) {
                mark(s, count);
            }
            if let Ext::Mimg(m) = insn.ext {
                for r in m.nsa.iter().take(usize::from(m.nsa_words) * 4) {
                    mark(Operand::Reg(Reg::V(*r)), 1);
                }
            }
        }
        if relative {
            Parking::drop_unwritten(&places, &written, &mut other);
        }
        let mut out = Parking::default();
        for place in places {
            let loc = REGS + out.slots.len();
            if !other[usize::from(place.0)] && loc < LOCATIONS {
                out.slots.insert(place, loc as u16);
            }
        }
        out
    }

    /// In a program with relative moves, keeps only the registers that look like a spill of
    /// scalars: every lane that is read back was written with `v_writelane_b32` first.
    ///
    /// Why a relative move never reaches such a register (`parks_beside_relative_moves`): it
    /// indexes an array that holds one value per lane, the private array of the source
    /// program. HLSL has no instruction that writes one chosen lane of a vector register, so
    /// a constant lane write only comes from the compiler parking a scalar. A register whose
    /// lanes are each a different parked scalar holds no per lane value, so no array of the
    /// source program can be in it. A register that is only read with constant lanes and never
    /// written that way could be an array element read across lanes, so it stays out.
    fn drop_unwritten(
        places: &BTreeSet<(u8, u8)>,
        written: &BTreeSet<(u8, u8)>,
        other: &mut [bool; 256],
    ) {
        for place in places.difference(written) {
            other[usize::from(place.0)] = true;
        }
    }

    fn slot(&self, insn: &Insn) -> Option<u16> {
        Parking::place(insn).and_then(|p| self.slots.get(&p).copied())
    }
}

/// How many vector registers an instruction touches from each register it names: the result,
/// the second result and the four sources. A count may be too large, never too small.
fn vector_spans(insn: &Insn) -> (usize, usize, [usize; 4]) {
    let name = insn.op.name();
    let info = insn.op.info();
    let words = |s: &Slot| usize::from((s.bits / 32).max(1));
    match (insn.format(), insn.ext) {
        (Format::Mimg, Ext::Mimg(m)) => {
            if matches!(
                insn.op,
                Op::image_bvh_intersect_ray | Op::image_bvh64_intersect_ray
            ) {
                let address = if m.nsa_words > 0 {
                    1
                } else {
                    let node = if insn.op == Op::image_bvh64_intersect_ray {
                        2
                    } else {
                        1
                    };
                    node + 4 + if m.a16 { 3 } else { 6 }
                };
                return (4, 1, [address, 1, 1, 1]);
            }
            let data = if name.contains("gather4") {
                4
            } else {
                m.dmask.count_ones() as usize
            } + usize::from(m.tfe);
            // Without extra address words the address registers follow the first one: the
            // coordinates of the dimension, and what the name adds (bias, compare value,
            // offset, level, clamp, derivatives, mip level).
            let dim = usize::from(m.dim & 7);
            let coordinates = [1, 2, 3, 3, 2, 3, 3, 4][dim];
            let has = |part: &str| name.split('_').any(|p| p == part);
            let extras = usize::from(has("b"))
                + usize::from(has("c"))
                + usize::from(has("o"))
                + usize::from(has("cl"))
                + usize::from(has("l"))
                + usize::from(has("mip"))
                + if has("d") {
                    2 * [1, 2, 3, 3, 1, 2, 2, 2][dim]
                } else {
                    0
                };
            // With extra address words every address register is named: the first one here,
            // the others in the words.
            let address = if m.nsa_words > 0 {
                1
            } else {
                coordinates + extras
            };
            (data.max(1), 1, [address, 1, 1, 1])
        }
        (
            Format::Mubuf | Format::Mtbuf,
            Ext::Buf {
                offen, idxen, tfe, ..
            },
        ) => {
            let data = name
                .rsplit_once("dwordx")
                .and_then(|(_, n)| n.parse().ok())
                .or_else(|| name.rsplit_once("format_").map(|(_, c)| c.len()))
                .unwrap_or(if name.contains("cmpswap") { 2 } else { 1 });
            (
                data + usize::from(tfe),
                1,
                [1 + usize::from(offen && idxen), 1, 1, 1],
            )
        }
        (Format::Ds, _) => {
            let wide = if name.ends_with("64") { 2 } else { 1 };
            let pair = if name.contains("read2") { 2 } else { 1 };
            (wide * pair, 1, [1, wide, wide, 1])
        }
        (Format::Flat | Format::Global | Format::Scratch, _) => (256, 256, [256; 4]),
        _ => {
            let mut defs = info.defs.iter().filter(|s| s.fixed == Fixed::No).map(words);
            let mut src = [1; 4];
            for (count, s) in src
                .iter_mut()
                .zip(info.srcs.iter().filter(|s| s.fixed == Fixed::No))
            {
                *count = words(s);
            }
            (defs.next().unwrap_or(1), defs.next().unwrap_or(1), src)
        }
    }
}

const NO_GROUP: u32 = u32::MAX;

/// Collects the touches of one instruction.
struct Collector<'a> {
    lanes: u8,
    parking: &'a Parking,
    out: &'a mut Vec<Touch>,
}

impl Collector<'_> {
    fn mask_words(&self) -> u8 {
        self.lanes / 32
    }

    fn at(&mut self, loc: u16, half: u8, role: Role, def: bool) {
        self.out.push(Touch {
            loc,
            half,
            role,
            def,
            group: NO_GROUP,
        });
    }

    fn push(&mut self, op: Operand, words: u8, role: Role, def: bool) {
        let Some(first) = followed(op) else { return };
        for half in 0..words {
            let reg = usize::from(first) + usize::from(half);
            if reg < REGS {
                self.at(reg as u16, half, role, def);
            }
        }
    }

    fn uses(&mut self, op: Operand, words: u8, role: Role) {
        self.push(op, words, role, false);
    }

    fn defs(&mut self, op: Operand, words: u8, role: Role) {
        self.push(op, words, role, true);
    }

    /// Scalar instructions with register operands.
    fn scalar(&mut self, insn: &Insn) {
        let info = insn.op.info();
        let words = |s: &Slot| (s.bits / 32).max(1);
        let ours = |w: u8| w == self.lanes;
        if let Some((kind, w)) = mask_op(insn.op) {
            let role = match kind {
                MaskOp::SaveExec(_) | MaskOp::WrExec(_) | MaskOp::Wqm => Role::Mask,
                MaskOp::Logic(Logic::Mov) | MaskOp::Select
                    if !ours(w)
                        && matches!(insn.op, Op::s_mov_b32 | Op::s_cselect_b32)
                        && !names(insn, Reg::ExecLo) =>
                {
                    for s in insn.src {
                        self.uses(s, 1, Role::MaskWord);
                    }
                    self.defs(insn.dst, 1, Role::Data);
                    return;
                }
                MaskOp::Logic(_) | MaskOp::Select if !ours(w) => Role::Data,
                MaskOp::Logic(_) | MaskOp::Select => {
                    // A constant other than "no lane" and "all lanes" is a number.
                    let constant = insn.src.iter().any(|s| {
                        matches!(s, Operand::Lit(v) if *v != 0 && !(w == 32 && *v == u32::MAX))
                            || matches!(s, Operand::Float(_))
                            || matches!(s, Operand::Int(v) if *v != 0 && *v != -1)
                    });
                    if names(insn, Reg::ExecLo) {
                        Role::Mask
                    } else if constant {
                        Role::Data
                    } else {
                        Role::Same
                    }
                }
            };
            let width = if role == Role::Data {
                w / 32
            } else {
                self.mask_words()
            };
            for s in insn.src {
                self.uses(s, width, role);
            }
            self.defs(insn.dst, width, role);
            return;
        }
        let number = mask_number(insn.op).filter(|w| ours(*w));
        let zero = zero_test(insn).filter(|(_, w, _)| ours(*w));
        let name = insn.op.name();
        if insn.format() == Format::Sopk {
            // The register field of the other immediate instructions is not a register.
            let (reads, writes) = match insn.op {
                Op::s_movk_i32 | Op::s_getreg_b32 => (false, true),
                Op::s_addk_i32 | Op::s_mulk_i32 | Op::s_cmovk_i32 => (true, true),
                Op::s_setreg_b32 => (true, false),
                _ => (name.starts_with("s_cmpk"), false),
            };
            if reads {
                self.uses(insn.dst, 1, Role::Data);
            }
            if writes {
                self.defs(insn.dst, 1, Role::Data);
            }
            return;
        }
        // The instructions that change their destination in place: the table lists the
        // destination as their first source, the encoding has only the other one.
        let in_place = matches!(
            insn.op,
            Op::s_cmov_b32
                | Op::s_cmov_b64
                | Op::s_bitset0_b32
                | Op::s_bitset1_b32
                | Op::s_bitset0_b64
                | Op::s_bitset1_b64
        );
        let mut slots = info
            .srcs
            .iter()
            .filter(|s| s.fixed == Fixed::No)
            .skip(usize::from(in_place && !name.starts_with("s_cmov")));
        // `s_getpc_b64` has no source: its source field is not a register.
        if insn.op != Op::s_getpc_b64 {
            for (i, s) in insn.src.into_iter().enumerate() {
                let role = match (number, zero) {
                    (Some(_), _) if i == 0 => Role::Number,
                    (_, Some((at, ..))) if i == at => Role::Number,
                    _ => Role::Data,
                };
                self.uses(s, slots.next().map_or(1, words), role);
            }
        }
        if insn.format() == Format::Sopc {
            return;
        }
        let dst_words = info
            .defs
            .iter()
            .find(|s| s.fixed == Fixed::No)
            .map_or(1, words);
        if in_place {
            self.uses(insn.dst, dst_words, Role::Data);
        }
        self.defs(insn.dst, dst_words, Role::Data);
    }

    /// A scalar parked in a lane of a vector register, or taken back: a move between a
    /// register and the parking place, of whatever kind the scalar is. Whether the instruction
    /// was one.
    fn parked(&mut self, insn: &Insn) -> bool {
        let Some(slot) = self.parking.slot(insn) else {
            return false;
        };
        // EXEC itself parked or taken back: surely a mask, one half per instruction.
        let exec = |op: Operand| match op {
            Operand::Reg(Reg::ExecLo) => Some(0),
            Operand::Reg(Reg::ExecHi) => Some(1),
            _ => None,
        };
        if insn.op == Op::v_writelane_b32_e64 {
            match insn.src[0] {
                op if exec(op).is_some() => self.at(slot, exec(op).unwrap_or(0), Role::Mask, true),
                // "No lane" and "all lanes" are constants of both kinds.
                Operand::Int(0 | -1) | Operand::Lit(0 | u32::MAX) => {
                    self.at(slot, 0, Role::Same, true)
                }
                Operand::Int(_) | Operand::Lit(_) | Operand::Float(_) => {
                    self.at(slot, 0, Role::Data, true)
                }
                op => {
                    self.uses(op, 1, Role::Same);
                    self.at(slot, 0, Role::Same, true);
                }
            }
        } else if let Some(half) = exec(insn.dst) {
            self.at(slot, half, Role::Mask, false);
        } else {
            self.at(slot, 0, Role::Same, false);
            self.defs(insn.dst, 1, Role::Same);
        }
        true
    }

    /// Vector instructions: the table says which slots are lane masks.
    fn vector(&mut self, insn: &Insn) {
        if insn.op == Op::v_writelane_b32_e64 {
            self.uses(insn.src[0], 1, Role::MaskWord);
            self.uses(insn.src[1], 1, Role::Data);
            return;
        }
        if matches!(insn.op, Op::v_mbcnt_lo_u32_b32 | Op::v_mbcnt_hi_u32_b32_e64) {
            // The count of the bits below the lane takes one word of a mask, or a number.
            self.uses(insn.src[0], 1, Role::Number);
            self.uses(insn.src[1], 1, Role::Data);
            return;
        }
        let info = insn.op.info();
        let mask_words = self.mask_words();
        let slot = |s: &Slot, op: Operand| {
            if s.base == Base::LaneMask {
                // A mask slot the encoding does not name is VCC.
                let op = if op == Operand::None {
                    Operand::Reg(Reg::VccLo)
                } else {
                    op
                };
                (op, mask_words, Role::Mask)
            } else {
                (op, (s.bits / 32).max(1), Role::Data)
            }
        };
        // How many sources the encoding really has. The two word form of a one or two source
        // instruction has unused source fields that decode as `s0`, and the table lists the
        // register an instruction accumulates into as one more source.
        let encoded = match (insn.format(), insn.op) {
            (Format::Vop1, _) => 1,
            (Format::Vop2 | Format::Vopc, _) | (_, Op::v_writelane_b32_e64) => 2,
            _ => 4,
        };
        for (i, (s, op)) in info
            .srcs
            .iter()
            .filter(|s| s.fixed == Fixed::No)
            .zip(insn.src)
            .enumerate()
        {
            if i < encoded || s.base == Base::LaneMask {
                let (op, words, role) = slot(s, op);
                self.uses(op, words, role);
            }
        }
        for (s, op) in info
            .defs
            .iter()
            .filter(|s| s.fixed == Fixed::No)
            .zip([insn.dst, insn.dst2])
        {
            let (op, words, role) = slot(s, op);
            self.defs(op, words, role);
        }
    }

    /// Memory instructions: the table lists no operands for them.
    fn memory(&mut self, insn: &Insn) {
        let name = insn.op.name();
        match insn.format() {
            Format::Smem => {
                let count = name
                    .rsplit_once("dwordx")
                    .and_then(|(_, n)| n.parse().ok())
                    .unwrap_or(1u8);
                let descriptor = name.starts_with("s_buffer") || name.starts_with("s_scratch");
                self.uses(insn.src[0], if descriptor { 4 } else { 2 }, Role::Data);
                if let Ext::Smem { soffset, .. } = insn.ext {
                    self.uses(soffset, 1, Role::Data);
                }
                if name.contains("load") {
                    self.defs(insn.dst, count, Role::Data);
                } else {
                    self.uses(insn.dst, count, Role::Data);
                }
            }
            Format::Mubuf | Format::Mtbuf => {
                self.uses(insn.src[1], 4, Role::Data);
                self.uses(insn.src[2], 1, Role::Data);
            }
            Format::Mimg => {
                // `r128` and the BVH ray-intersection ops read a 4-dword (128-bit) descriptor,
                // not an 8-dword T#: marking the upper four SGPRs as data reads would falsely
                // collide with live lane masks kept in those registers.
                let rsrc_words = match insn.ext {
                    Ext::Mimg(m)
                        if m.r128
                            || matches!(
                                insn.op,
                                Op::image_bvh_intersect_ray | Op::image_bvh64_intersect_ray
                            ) =>
                    {
                        4
                    }
                    _ => 8,
                };
                self.uses(insn.src[1], rsrc_words, Role::Data);
                // Only the filtered forms have a sampler. The field decodes as `s0` in the others.
                if ["image_sample", "image_gather4", "image_get_lod"]
                    .iter()
                    .any(|p| name.starts_with(p))
                {
                    self.uses(insn.src[2], 4, Role::Data);
                }
            }
            _ => {}
        }
    }

    fn insn(&mut self, insn: &Insn) {
        match insn.format() {
            Format::Sop1 | Format::Sop2 | Format::Sopk | Format::Sopc => self.scalar(insn),
            // The interpolation forms of the vector formats have no scalar operand.
            Format::Vop1 | Format::Vop2 | Format::Vop3 | Format::Vop3p | Format::Vopc
                if !insn.op.name().starts_with("v_interp") =>
            {
                if !self.parked(insn) {
                    self.vector(insn)
                }
            }
            Format::Smem | Format::Mubuf | Format::Mtbuf | Format::Mimg => self.memory(insn),
            Format::Vop1
            | Format::Vop2
            | Format::Vop3
            | Format::Vop3p
            | Format::Vopc
            | Format::Sopp
            | Format::Ds
            | Format::Exp
            | Format::Vintrp
            | Format::Flat
            | Format::Global
            | Format::Scratch => {}
        }
    }
}

/// The kinds of value a location can hold: a number, a mask (the low word of one in a 64 lane
/// program), the high word of a 64 lane mask. `SURE` marks a set that rests on an instruction
/// that is sure of its kind, and not only on bitwise instructions that could be either.
const NUMBER: u8 = 1;
const LOW: u8 = 2;
const HIGH: u8 = 4;
const ANY: u8 = NUMBER | LOW | HIGH;
const SURE: u8 = 8;

/// The word of a mask a touch is.
fn word_kind(half: u8) -> u8 {
    if half == 0 { LOW } else { HIGH }
}

/// How many locations the typing can follow: the registers and the parking places.
const LOCATIONS: usize = 512;

/// A set of locations.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Bits([u64; LOCATIONS / 64]);

impl Bits {
    const NONE: Bits = Bits([0; LOCATIONS / 64]);

    fn has(&self, loc: u16) -> bool {
        (self.0[usize::from(loc) / 64] >> (loc % 64)) & 1 != 0
    }

    fn put(&mut self, loc: u16, on: bool) {
        let word = &mut self.0[usize::from(loc) / 64];
        if on {
            *word |= 1 << (loc % 64);
        } else {
            *word &= !(1 << (loc % 64));
        }
    }
}

/// A set of kinds for every location: one plane per kind bit.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Kinds([Bits; 4]);

impl Kinds {
    const NONE: Kinds = Kinds([Bits::NONE; 4]);

    fn get(&self, loc: u16) -> u8 {
        (0..4).fold(0, |k, p| k | (u8::from(self.0[p].has(loc)) << p))
    }

    fn set(&mut self, loc: u16, kinds: u8) {
        for (p, plane) in self.0.iter_mut().enumerate() {
            plane.put(loc, (kinds >> p) & 1 != 0);
        }
    }

    fn add(&mut self, loc: u16, kinds: u8) {
        for (p, plane) in self.0.iter_mut().enumerate() {
            if (kinds >> p) & 1 != 0 {
                plane.put(loc, true);
            }
        }
    }

    fn merge(&mut self, other: &Kinds) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            for (x, y) in a.0.iter_mut().zip(&b.0) {
                *x |= y;
            }
        }
    }

    /// What is left of `self` after the locations of `kill` are rewritten with `made`.
    fn through(&self, kill: &Bits, made: &Kinds) -> Kinds {
        let mut out = *self;
        for (plane, new) in out.0.iter_mut().zip(&made.0) {
            for ((x, k), m) in plane.0.iter_mut().zip(&kill.0).zip(&new.0) {
                *x = (*x & !k) | m;
            }
        }
        out
    }
}

/// The touches of one instruction that share one kind.
#[derive(Clone, Copy)]
struct Group {
    /// The kinds still possible.
    kinds: u8,
    /// Whether a sure instruction reaches the group, through what it reads or what reads it.
    sure: bool,
    /// What it reads and what reads it have no kind in common.
    clash: bool,
    /// Whether any definition in this group is read by a later instruction.
    def_live: bool,
}

impl Group {
    /// Takes the kinds a neighbour allows. Whether anything changed.
    fn narrow(&mut self, set: u8) -> bool {
        if set & ANY == 0 {
            return false;
        }
        let mut changed = false;
        if set & SURE != 0 && !self.sure {
            self.sure = true;
            changed = true;
        }
        let kinds = self.kinds & set;
        if kinds == 0 {
            self.clash = true;
        } else if kinds != self.kinds {
            self.kinds = kinds;
            changed = true;
        }
        changed
    }

    /// The kind of the group, or why it has none. A group nothing sure reaches is a number:
    /// the bitwise instructions do the same to both.
    fn verdict(&self) -> Result<u8, &'static str> {
        if self.clash {
            Err("mixed-pair")
        } else if self.kinds.count_ones() == 1 {
            Ok(self.kinds)
        } else if self.sure {
            Err("mixed-pair")
        } else {
            Ok(NUMBER)
        }
    }
}

/// The kinds a write leaves in its location.
fn written(t: &Touch, groups: &[Group]) -> u8 {
    match t.role {
        Role::Mask => word_kind(t.half) | SURE,
        Role::Data | Role::Number | Role::MaskWord => NUMBER | SURE,
        Role::Same => {
            let g = groups[t.group as usize];
            g.kinds | if g.sure { SURE } else { 0 }
        }
    }
}

/// The kinds a read accepts in its location.
fn wanted(t: &Touch, groups: &[Group]) -> u8 {
    match t.role {
        Role::Number | Role::MaskWord => 0,
        _ => written(t, groups),
    }
}

/// What a parking instruction moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parked {
    Number,
    /// A mask, or the low word of one in a 64 lane program.
    Mask,
    /// The high word of a 64 lane mask. The low word carries this lane's bit.
    MaskHigh,
}

/// What the typing found for one instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Site {
    /// The registers that hold a mask where the instruction reads them.
    reads: Set,
    /// Reads of the high word of a mask.
    high_reads: Set,
    /// The registers the instruction writes a mask to.
    writes: Set,
    /// Why the instruction cannot be translated one lane at a time.
    refusal: Option<&'static str>,
    /// For a parking instruction: the kind it moves.
    parked: u8,
    /// The part the instruction has in a reduction over the wave.
    reduction: Option<Reduction>,
    /// The part the instruction has in a wave exclusive prefix scan when lifted with real wave
    /// intrinsics.
    scan: Option<WaveScan>,
    /// The part the instruction has in a complete wave reduction when lifted with real wave
    /// intrinsics.
    wave_reduce: Option<WaveReducePart>,
    /// Bitmask of 32 bit ballot words (bit 0 low, bit 1 high) this `v_cmp` must
    /// materialise into its scalar destination pair for later numeric word reads.
    materialize_ballot: u8,
    /// Whether this `s_mov_b32` or `s_cselect_b32` reads mask words already materialised
    /// into scalar registers at their defining `v_cmp`.
    ballot_ready: bool,
}

/// Where the scalar registers of a program hold lane masks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Masks {
    lanes: u8,
    /// Whole mask words can only be collected where all invocations stay together.
    converged: bool,
    /// Whether matched scans and reductions must lower with real HLSL wave intrinsics because
    /// the program also reads values across lanes with constant `v_readlane_b32`.
    real_wave: bool,
    sites: BTreeMap<u32, Site>,
    /// The vector registers that only park scalars, one bit per register.
    parking: [u64; 4],
    /// The instruction the lifter is at.
    at: Cell<Site>,
}

impl Masks {
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Whether compute wave operations in this program must use real wave intrinsics rather than
    /// a single lane projection.
    pub fn real_wave(&self) -> bool {
        self.real_wave
    }

    /// Whether vector register `v` only parks scalars (`v_writelane_b32` and `v_readlane_b32`
    /// with constant lanes). Its lanes are scalars of their own, not the register's value.
    pub fn parks_scalars(&self, v: u8) -> bool {
        (self.parking[usize::from(v / 64)] >> (v % 64)) & 1 != 0
    }

    pub fn reads_high_mask(&self, first: u8) -> bool {
        usize::from(first) < REGS && self.at.get().high_reads & bit(first) != 0
    }

    /// The lanes of a wave of this program: 32 or 64.
    pub fn lanes(&self) -> u8 {
        if self.lanes == 32 { 32 } else { 64 }
    }

    /// Moves to the instruction at word `pc`. The questions below are about that instruction.
    pub fn enter(&self, pc: u32) {
        self.at
            .set(self.sites.get(&pc).copied().unwrap_or_default());
    }

    /// Whether the register (the first of a pair in a 64 lane program) is a mask operand of the
    /// instruction: a mask where it is read, or the mask the instruction writes.
    pub fn is_mask(&self, first: u8) -> bool {
        self.reads_mask(first) || self.writes_mask(first)
    }

    pub fn reads_mask(&self, first: u8) -> bool {
        usize::from(first) < REGS && self.at.get().reads & bit(first) != 0
    }

    pub fn writes_mask(&self, first: u8) -> bool {
        usize::from(first) < REGS && self.at.get().writes & bit(first) != 0
    }

    /// Whether any scalar register operand of the instruction is a mask.
    pub fn touches_mask(&self) -> bool {
        let s = self.at.get();
        s.reads | s.writes != 0
    }

    /// The refusal key of the instruction, when the typing found a contradiction at it.
    pub fn refusal(&self) -> Option<&'static str> {
        self.at.get().refusal
    }

    /// When the instruction parks a scalar in a lane of a vector register that the program
    /// only uses for that, or takes it back: what it moves. `None` for a `v_writelane_b32` or
    /// `v_readlane_b32` that really reaches into another lane.
    pub fn parked(&self) -> Option<Parked> {
        match self.at.get().parked {
            NUMBER => Some(Parked::Number),
            LOW => Some(Parked::Mask),
            HIGH => Some(Parked::MaskHigh),
            _ => None,
        }
    }

    /// The part the instruction has in a reduction over the wave, when it has one.
    pub fn reduction(&self) -> Option<Reduction> {
        self.at.get().reduction
    }

    /// The part the instruction has in a matched wave exclusive prefix scan.
    pub fn scan(&self) -> Option<WaveScan> {
        self.at.get().scan
    }

    /// The part the instruction has in a complete wave reduction for real wave lowering.
    pub fn wave_reduce(&self) -> Option<WaveReducePart> {
        self.at.get().wave_reduce
    }

    /// Which 32 bit ballot words (bit 0 low, bit 1 high) this `v_cmp` must materialise into
    /// its scalar registers.
    pub fn materialize_ballot(&self) -> u8 {
        self.at.get().materialize_ballot
    }

    /// Whether this `s_mov_b32` or `s_cselect_b32` reads mask words already materialised
    /// into scalar registers at their defining `v_cmp`.
    pub fn ballot_ready(&self) -> bool {
        self.at.get().ballot_ready
    }
}

/// How an instruction inside a matched wave exclusive prefix scan lowers in real wave mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveScan {
    /// Intermediate lane shuffle or temporary half wave mask step, skipped in real wave mode.
    Skip,
    /// Final `v_sub_u32 sum, sum, input`, lowered as `WavePrefixSum(input)`.
    PrefixSum,
}

/// Operation of a complete single word wave reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveReduce {
    AddU32,
    AddF32,
    MulU32,
    MulF32,
    MinU32,
    MaxU32,
    MinI32,
    MaxI32,
    MinF32,
    MaxF32,
    Or,
    And,
    Xor,
}

/// How a `v_readlane_b32` result instruction of a matched wave reduction lowers in real wave mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveReducePart {
    /// First half wave read in a 64 lane reduction, replaced by the operation identity so the
    /// following combine leaves each lane's masked input intact for the full wave intrinsic.
    Identity(u32),
    /// Final read of the reduced wave value, lowered with the matching `WaveActive*` intrinsic.
    Active(WaveReduce),
}

fn wave_reduce_op(op: Op) -> Option<WaveReduce> {
    Some(match op {
        Op::v_add_u32 => WaveReduce::AddU32,
        Op::v_add_f32 => WaveReduce::AddF32,
        Op::v_mul_lo_u32 => WaveReduce::MulU32,
        Op::v_mul_f32 => WaveReduce::MulF32,
        Op::v_min_u32 => WaveReduce::MinU32,
        Op::v_max_u32 => WaveReduce::MaxU32,
        Op::v_min_i32 => WaveReduce::MinI32,
        Op::v_max_i32 => WaveReduce::MaxI32,
        Op::v_min_f32 => WaveReduce::MinF32,
        Op::v_max_f32 => WaveReduce::MaxF32,
        Op::v_or_b32 => WaveReduce::Or,
        Op::v_and_b32 => WaveReduce::And,
        Op::v_xor_b32 => WaveReduce::Xor,
        _ => return None,
    })
}

/// The part of one instruction in a reduction over the wave (a sum, a minimum, a bitwise or of
/// one value per lane).
///
/// The compiler builds a reduction from lane permutations, measured on the title's programs in
/// one shape: all lanes on (`s_or_saveexec sT, -1`), `v_cndmask_b32 vX, identity, value, mask`
/// (the lanes that do not take part get the identity of the operation), three or four steps
/// `v_<op> vX, vX, vX` in the data parallel forms that combine neighbours, pairs, groups of four
/// and of eight, `v_permlanex16_b32 vY, vX, 0, 0` and `v_<op> vX, vY, vX` for the two rows of
/// 16, and `v_readlane_b32 sR, vX, 0`. A 64 lane program then has the result of each half of
/// the wave: it reads lane 32, combines the scalar into every lane and reads lane 0.
///
/// After each step every lane holds the reduction of its group. In a wave with one active lane
/// (see `lift::wave`) every other lane holds the identity, so the group of this lane holds this
/// lane's value and every other group the identity: the steps change nothing in this lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// A step that combines this lane's group with another: nothing changes in this lane.
    Step,
    /// `v_permlanex16_b32`: the other row of 16 holds the identity.
    OtherRow { identity: u32 },
    /// `v_readlane_b32` of the reduced register: the reduction of the half of the wave the
    /// lane read is in. It is this lane's value when this lane is in that half (`upper` says
    /// which). Else it is the identity, or, after the result of the other half was combined
    /// into every lane, the scalar register `combined` that holds it.
    Result {
        identity: u32,
        upper: bool,
        combined: Option<u8>,
    },
}

/// The value an operation leaves unchanged, for the operations compilers reduce with.
fn identities(op: Op) -> &'static [u32] {
    match op {
        Op::v_or_b32 | Op::v_xor_b32 | Op::v_add_u32 | Op::v_max_u32 => &[0],
        Op::v_and_b32 | Op::v_min_u32 => &[u32::MAX],
        Op::v_min_i32 => &[0x7fff_ffff],
        Op::v_max_i32 => &[0x8000_0000],
        Op::v_mul_lo_u32 => &[1],
        // Plus zero is what compilers use. It turns a sum of minus zero into plus zero.
        Op::v_add_f32 => &[0, 0x8000_0000],
        Op::v_mul_f32 => &[0x3f80_0000],
        Op::v_min_f32 => &[0x7f80_0000],
        Op::v_max_f32 => &[0xff80_0000],
        _ => &[],
    }
}

/// A constant operand as its 32 bit pattern.
fn constant(op: Operand) -> Option<u32> {
    match op {
        Operand::Int(v) => Some(i32::from(v) as u32),
        Operand::Float(c) => Some(c.f32().to_bits()),
        Operand::Lit(v) => Some(v),
        _ => None,
    }
}

/// When the data parallel word of a step is a permutation that combines each group of `2^k`
/// lanes with its neighbour group: that `k` (0 to 3). No lane reads itself, every lane reads a
/// lane of the other group of its pair of groups. Such a step doubles the group every lane
/// holds the reduction of.
fn combines_groups(insn: &Insn) -> Option<u8> {
    // The lane each of the 16 lanes of a row reads.
    let mut from = [0u8; 16];
    if let Some((word, _)) = insn.dpp8() {
        for lane in 0..16u8 {
            from[usize::from(lane)] = (lane & 8) | word.lane_sel(lane & 7);
        }
    } else {
        let word = insn.dpp16()?;
        // Every row and bank, and no float modifier on the lane read.
        if word.row_mask() != 0xf
            || word.bank_mask() != 0xf
            || word.src0_neg()
            || word.src0_abs()
            || word.src1_neg()
            || word.src1_abs()
        {
            return None;
        }
        for lane in 0..16u8 {
            from[usize::from(lane)] = match word.ctrl() {
                DppCtrl::QuadPerm(p) => (lane & 12) | (p[usize::from(lane & 3)] & 3),
                DppCtrl::RowMirror => 15 - lane,
                DppCtrl::RowHalfMirror => (lane & 8) | (7 - (lane & 7)),
                _ => return None,
            };
        }
    }
    (0..4u8).find(|k| {
        let group = 1u8 << k;
        (0..16u8).all(|lane| {
            let other = from[usize::from(lane)];
            // The same pair of groups, the other group of the pair.
            other / (2 * group) == lane / (2 * group) && other / group != lane / group
        })
    })
}

/// Finds the reductions of a program (see [`Reduction`]): for each instruction that is part of
/// one, its part.
/// A complete unsigned exclusive scan has the same single active lane
/// projection as a sum reduction. Each row shift adds only zero in that
/// projection. The final subtraction removes this lane's inclusive value.
/// Match the complete sequence so an arbitrary lane permutation is never
/// mistaken for a scan step.
fn scans(cfg: &Cfg, lanes: u8) -> (BTreeMap<u32, Reduction>, BTreeMap<u32, WaveScan>) {
    let mut out = BTreeMap::new();
    let mut wave_scans = BTreeMap::new();
    let exec = Operand::Reg(Reg::ExecLo);
    let high = Operand::Reg(Reg::ExecHi);
    let scalar = |o| matches!(o, Operand::Reg(Reg::S(_)));
    let vector = |o| matches!(o, Operand::Reg(Reg::V(_)));
    let plain = |i: &Insn| {
        matches!(i.form, Form::Plain | Form::AsVop3)
            && matches!(
                i.ext,
                Ext::None
                    | Ext::Valu {
                        abs: 0,
                        neg: 0,
                        opsel: 0,
                        omod: 0,
                        clamp: false
                    }
            )
    };
    let row_steps = |steps: &[&Insn], sum: Operand| {
        steps
            .iter()
            .zip([0xff09_1100, 0xff01_1200, 0xff01_1400, 0xff01_1800])
            .all(|(i, control)| {
                i.op == Op::v_add_u32
                    && i.dst == sum
                    && i.src[..2] == [sum, sum]
                    && i.dpp16().is_some_and(|word| word.0 & !0xff == control)
            })
    };
    let move_exec = |insn: &Insn, dst, value| {
        insn.op == Op::s_mov_b32 && insn.dst == dst && constant(insn.src[0]) == Some(value)
    };
    let is_nop = |i: &Insn| {
        matches!(
            i.op,
            Op::s_nop | Op::v_nop | Op::s_waitcnt | Op::s_waitcnt_depctr
        )
    };
    if lanes == 32 {
        for block in &cfg.blocks {
            let active: Vec<&Insn> = block.insns.iter().filter(|i| !is_nop(i)).collect();
            for start in 0..active.len() {
                let first = active[start];
                let saved = first.dst;
                if first.op != Op::s_or_saveexec_b32
                    || !plain(first)
                    || !scalar(saved)
                    || constant(first.src[0]) != Some(u32::MAX)
                {
                    continue;
                }
                // Optional `s_and_b32 mask, other, saved` narrows which active lanes contribute
                // to the exclusive prefix sum while keeping `saved` intact for the restore.
                let (mask, rest) = if let Some(&narrow) = active.get(start + 1)
                    && narrow.op == Op::s_and_b32
                    && plain(narrow)
                    && scalar(narrow.dst)
                    && narrow.dst != saved
                    && ((narrow.src[0] == saved && scalar(narrow.src[1]))
                        || (narrow.src[1] == saved && scalar(narrow.src[0])))
                {
                    (narrow.dst, start + 2)
                } else {
                    (saved, start + 1)
                };
                let Some(w) = active.get(rest..rest + 12) else {
                    continue;
                };
                let input = w[0].dst;
                let sum = w[1].dst;
                let other = w[7].dst;
                if w[0].op != Op::v_cndmask_b32
                    || !vector(input)
                    || constant(w[0].src[0]) != Some(0)
                    || w[0].src[2] != mask
                    || w[1].op != Op::v_mov_b32
                    || !vector(sum)
                    || w[1].src[0] != input
                    || sum == input
                    || ![0, 1, 8, 10, 11].into_iter().all(|i| plain(w[i]))
                    || !move_exec(w[6], exec, 0xffff_0000)
                    || w[7].op != Op::v_permlanex16_b32
                    || !vector(other)
                    || other == input
                    || other == sum
                    || w[7].src[..3] != [sum, Operand::Int(-1), Operand::Int(-1)]
                    || !matches!(
                        w[7].ext,
                        Ext::Valu {
                            abs: 0,
                            neg: 0,
                            opsel: 1 | 9,
                            omod: 0,
                            clamp: false
                        }
                    )
                    || w[8].op != Op::v_add_u32
                    || w[8].dst != sum
                    || w[8].src[..2] != [other, sum]
                    || !move_exec(w[9], exec, u32::MAX)
                    || w[10].op != Op::v_sub_u32
                    || w[10].dst != sum
                    || w[10].src[..2] != [sum, input]
                    || w[11].op != Op::s_mov_b32
                    || w[11].dst != exec
                    || w[11].src[0] != saved
                {
                    continue;
                }
                let steps = &w[2..6];
                if !row_steps(steps, sum) {
                    continue;
                }
                for i in steps {
                    out.insert(i.pc, Reduction::Step);
                }
                out.insert(w[7].pc, Reduction::OtherRow { identity: 0 });
                for i in &w[2..10] {
                    wave_scans.insert(i.pc, WaveScan::Skip);
                }
                wave_scans.insert(w[10].pc, WaveScan::PrefixSum);
            }
        }
        return (out, wave_scans);
    }
    if lanes != 64 {
        return (out, wave_scans);
    }
    for block in &cfg.blocks {
        let active: Vec<&Insn> = block.insns.iter().filter(|i| !is_nop(i)).collect();
        for w in active.windows(19) {
            let saved = w[0].dst;
            let input = w[1].dst;
            let sum = w[2].dst;
            let other = w[9].dst;
            let total = w[11].dst;
            if w[0].op != Op::s_or_saveexec_b64
                || !scalar(saved)
                || constant(w[0].src[0]) != Some(u32::MAX)
                || w[1].op != Op::v_cndmask_b32
                || !vector(input)
                || constant(w[1].src[0]) != Some(0)
                || w[1].src[2] != saved
                || w[2].op != Op::v_mov_b32
                || !vector(sum)
                || w[2].src[0] != input
                || sum == input
                || ![0, 1, 2, 10, 11, 14, 17, 18]
                    .into_iter()
                    .all(|i| plain(w[i]))
                || !move_exec(w[7], exec, 0xffff_0000)
                || !move_exec(w[8], high, 0xffff_0000)
                || w[9].op != Op::v_permlanex16_b32
                || !vector(other)
                || other == input
                || other == sum
                || w[9].src[..3] != [sum, Operand::Int(-1), Operand::Int(-1)]
                || !matches!(
                    w[9].ext,
                    Ext::Valu {
                        abs: 0,
                        neg: 0,
                        opsel: 1 | 9,
                        omod: 0,
                        clamp: false
                    }
                )
                || w[10].op != Op::v_add_u32
                || w[10].dst != sum
                || w[10].src[..2] != [other, sum]
                || w[11].op != Op::v_readlane_b32_e64
                || !scalar(total)
                || matches!((saved, total), (Operand::Reg(Reg::S(a)), Operand::Reg(Reg::S(b)))
                    if b == a || b == a.saturating_add(1))
                || w[11].src[..2] != [sum, Operand::Int(31)]
                || !move_exec(w[12], exec, 0)
                || !move_exec(w[13], high, u32::MAX)
                || w[14].op != Op::v_add_u32
                || w[14].dst != sum
                || w[14].src[..2] != [total, sum]
                || !move_exec(w[15], exec, u32::MAX)
                || !move_exec(w[16], high, u32::MAX)
                || w[17].op != Op::v_sub_u32
                || w[17].dst != sum
                || w[17].src[..2] != [sum, input]
                || w[18].op != Op::s_mov_b64
                || w[18].dst != exec
                || w[18].src[0] != saved
            {
                continue;
            }
            let steps = &w[3..7];
            if !row_steps(steps, sum) {
                continue;
            }
            for i in steps {
                out.insert(i.pc, Reduction::Step);
            }
            out.insert(w[9].pc, Reduction::OtherRow { identity: 0 });
            out.insert(
                w[11].pc,
                Reduction::Result {
                    identity: 0,
                    upper: false,
                    combined: None,
                },
            );
            for i in &w[3..17] {
                wave_scans.insert(i.pc, WaveScan::Skip);
            }
            wave_scans.insert(w[17].pc, WaveScan::PrefixSum);
        }
    }
    (out, wave_scans)
}

fn reductions(cfg: &Cfg, lanes: u8) -> (BTreeMap<u32, Reduction>, BTreeMap<u32, WaveReducePart>) {
    /// A register that is being reduced.
    #[derive(Clone, Copy)]
    struct Chain {
        identity: u32,
        /// The scalar register that holds the result of one half of the wave, read from the
        /// lane in the half `upper`, and whether it was combined into every lane yet.
        half: Option<(u8, bool, bool)>,
        /// Every lane holds the reduction of its group of `2^level` lanes: 4 is a row, 5 a
        /// half of 32 lanes. The steps have to come in this order, or a lane reads a lane
        /// that already holds its value.
        level: u8,
        /// Consistent single word reduction op across all steps of the chain.
        op: Option<WaveReduce>,
        /// Set when steps mix operations or use multi word carry chains.
        mixed_op: bool,
        /// PC of the first half wave `v_readlane_b32` in a 64 lane reduction.
        first_result: Option<u32>,
    }
    impl Chain {
        fn record_op(&mut self, op: Op) {
            match (self.op, wave_reduce_op(op)) {
                (None, Some(r)) => self.op = Some(r),
                (Some(cur), Some(r)) if cur == r => {}
                _ => self.mixed_op = true,
            }
        }
    }
    /// A sibling group value staged in a temporary VGPR by `v_mov_b32_dpp`.
    #[derive(Clone, Copy)]
    struct Staged {
        chain: u8,
        level: u8,
        identity: u32,
        pc: u32,
    }
    let carry_reg = |op: Operand| match op {
        Operand::None | Operand::Reg(Reg::VccLo) => Some(VCC),
        Operand::Reg(Reg::S(r)) if r < VCC => Some(r),
        _ => None,
    };
    let mut out = BTreeMap::new();
    let mut wave_reduces = BTreeMap::new();
    for block in &cfg.blocks {
        let mut all_on = false;
        let mut chains: BTreeMap<u8, Chain> = BTreeMap::new();
        let mut literal_moves: BTreeMap<u8, u32> = BTreeMap::new();
        // Registers that hold the identity in this lane: staged DPP moves or the other row
        // of a reduced register.
        let mut staged: BTreeMap<u8, Staged> = BTreeMap::new();
        let mut rows: BTreeMap<u8, (u8, u32)> = BTreeMap::new();
        // A carry register guaranteed to be zero from a zero-identity `v_add_co_u32` or
        // `v_addc_co_u32` reduction combine at the recorded level.
        let mut carry_zero: Option<(u8, u8)> = None;
        for insn in &block.insns {
            let v = |op: Operand| match op {
                Operand::Reg(Reg::V(n)) => Some(n),
                _ => None,
            };
            let ids = identities(insn.op);
            let dst = v(insn.dst);
            // Instructions without an effect on registers.
            if matches!(
                insn.op,
                Op::s_nop | Op::v_nop | Op::s_waitcnt | Op::s_waitcnt_depctr
            ) {
                continue;
            }
            // EXEC: only "every lane" starts reductions, any other write ends them.
            if matches!(insn.format(), Format::Sop1 | Format::Sop2)
                && let Some((kind, w)) = mask_op(insn.op)
                && (insn.dst == Operand::Reg(Reg::ExecLo)
                    || matches!(kind, MaskOp::SaveExec(_) | MaskOp::WrExec(_)))
            {
                all_on = w == lanes
                    && insn.src[0] == Operand::Int(-1)
                    && matches!(
                        kind,
                        MaskOp::SaveExec(Logic::Or) | MaskOp::Logic(Logic::Mov)
                    );
                chains.clear();
                rows.clear();
                staged.clear();
                literal_moves.clear();
                carry_zero = None;
                continue;
            }
            if matches!(insn.dst, Operand::Reg(Reg::ExecLo | Reg::ExecHi)) {
                all_on = false;
                chains.clear();
                rows.clear();
                staged.clear();
                literal_moves.clear();
                carry_zero = None;
                continue;
            }
            let plain = matches!(insn.form, Form::Plain | Form::AsVop3);
            let no_mods = matches!(
                insn.ext,
                Ext::None
                    | Ext::Valu {
                        abs: 0,
                        neg: 0,
                        opsel: 0,
                        omod: 0,
                        clamp: false
                    }
            );
            // A literal move under full EXEC can supply the identity to a later cndmask
            // in this block. Only the ordinary VOP1 move writes a new known constant.
            let identity = dst
                .filter(|&x| insn.src[0] == Operand::Reg(Reg::V(x)))
                .and_then(|x| literal_moves.get(&x).copied());
            // Other instructions may have multi-register destinations. Drop the facts for
            // any VGPR write, even for instructions the reduction recognises below.
            if matches!(
                insn.format(),
                Format::Mimg | Format::Mubuf | Format::Mtbuf | Format::Ds
            ) {
                literal_moves.clear();
            } else {
                let (d0, d1, _) = vector_spans(insn);
                for (op, count) in [(insn.dst, d0), (insn.dst2, d1)] {
                    if let Some(x) = v(op) {
                        for k in 0..count {
                            literal_moves.remove(&x.wrapping_add(k as u8));
                        }
                    }
                }
            }
            if all_on
                && insn.op == Op::v_mov_b32
                && insn.form == Form::Plain
                && let (Some(x), Some(value)) = (dst, constant(insn.src[0]))
            {
                literal_moves.insert(x, value);
            }
            match (insn.op, insn.form) {
                // The start: the lanes that do not take part get the identity.
                (Op::v_cndmask_b32, Form::Plain | Form::AsVop3) if all_on => {
                    if let (Some(x), Some(identity)) = (
                        dst,
                        constant(insn.src[0]).or_else(|| identity.filter(|_| no_mods)),
                    ) {
                        rows.remove(&x);
                        rows.retain(|_, (ch, _)| *ch != x);
                        staged.remove(&x);
                        staged.retain(|_, s| s.chain != x);
                        chains.insert(
                            x,
                            Chain {
                                identity,
                                half: None,
                                level: 0,
                                op: None,
                                mixed_op: false,
                                first_result: None,
                            },
                        );
                        continue;
                    }
                }
                // A DPP move staging the sibling group's value into a temporary register for
                // an immediately following combine (for example a 64 bit carry-chain sum).
                (Op::v_mov_b32, Form::Dpp8 | Form::Dpp16) => {
                    if let (Some(y), Some(x)) = (dst, v(insn.src[0]))
                        && y != x
                        && let Some(c) = chains.get(&x).copied()
                        && combines_groups(insn) == Some(c.level)
                    {
                        chains.remove(&y);
                        rows.remove(&y);
                        rows.retain(|_, (ch, _)| *ch != y);
                        staged.retain(|_, s| s.chain != y);
                        staged.insert(
                            y,
                            Staged {
                                chain: x,
                                level: c.level,
                                identity: c.identity,
                                pc: insn.pc,
                            },
                        );
                        continue;
                    }
                }
                // A step in a data parallel form.
                (_, Form::Dpp8 | Form::Dpp16) => {
                    if let Some(x) = dst
                        && let Some(c) = chains.get_mut(&x)
                        && v(insn.src[0]) == Some(x)
                        && v(insn.src[1]) == Some(x)
                        && ids.contains(&c.identity)
                        && combines_groups(insn) == Some(c.level)
                    {
                        c.level += 1;
                        c.record_op(insn.op);
                        rows.retain(|_, (ch, _)| *ch != x);
                        staged.retain(|_, s| s.chain != x);
                        out.insert(insn.pc, Reduction::Step);
                        continue;
                    }
                }
                // The other row of 16 lanes.
                (Op::v_permlanex16_b32, _) => {
                    if let (Some(y), Some(x)) = (dst, v(insn.src[0]))
                        && let Some(c) = chains.get(&x).copied()
                        && c.level == 4
                        && insn.src[1] == Operand::Int(0)
                        && insn.src[2] == Operand::Int(0)
                        && y != x
                    {
                        out.insert(
                            insn.pc,
                            Reduction::OtherRow {
                                identity: c.identity,
                            },
                        );
                        chains.remove(&y);
                        staged.remove(&y);
                        staged.retain(|_, s| s.chain != y);
                        rows.retain(|_, (ch, _)| *ch != y);
                        rows.insert(y, (x, c.identity));
                        continue;
                    }
                }
                // The result of one half of the wave, or of the whole wave.
                (Op::v_readlane_b32_e64, _) => {
                    if let (Operand::Reg(Reg::S(r)), Some(x), Operand::Int(lane)) =
                        (insn.dst, v(insn.src[0]), insn.src[1])
                        && let Some(c) = chains.get_mut(&x)
                        && c.level == 5
                        && (0..i8::try_from(lanes).unwrap_or(0)).contains(&lane)
                    {
                        let upper = lane >= 32;
                        match c.half {
                            // Every lane holds the reduction of its half.
                            None => {
                                out.insert(
                                    insn.pc,
                                    Reduction::Result {
                                        identity: c.identity,
                                        upper,
                                        combined: None,
                                    },
                                );
                                if lanes == 32 {
                                    if !c.mixed_op
                                        && let Some(rop) = c.op
                                    {
                                        wave_reduces.insert(insn.pc, WaveReducePart::Active(rop));
                                    }
                                } else {
                                    c.first_result = Some(insn.pc);
                                }
                                c.half = Some((r, upper, false));
                                if carry_zero.is_some_and(|(cr, _)| cr == r) {
                                    carry_zero = None;
                                }
                                continue;
                            }
                            // The other half was combined in: a lane of this half holds the
                            // reduction of the whole wave.
                            Some((scalar, first, true)) if first != upper => {
                                out.insert(
                                    insn.pc,
                                    Reduction::Result {
                                        identity: c.identity,
                                        upper,
                                        combined: Some(scalar),
                                    },
                                );
                                if !c.mixed_op
                                    && let (Some(rop), Some(pc1)) = (c.op, c.first_result)
                                {
                                    wave_reduces.insert(pc1, WaveReducePart::Identity(c.identity));
                                    wave_reduces.insert(insn.pc, WaveReducePart::Active(rop));
                                }
                                chains.remove(&x);
                                if carry_zero.is_some_and(|(cr, _)| cr == r) {
                                    carry_zero = None;
                                }
                                continue;
                            }
                            _ => {}
                        }
                    }
                }
                // The plain operation of the reduction: with a staged DPP move, with the
                // other row, or with the result of the other half. The ordinary rules lift it.
                _ if plain && !ids.is_empty() => {
                    if let Some(x) = dst
                        && let Some(c) = chains.get_mut(&x)
                        && ids.contains(&c.identity)
                        && (v(insn.src[0]) == Some(x) || v(insn.src[1]) == Some(x))
                    {
                        let other = if v(insn.src[1]) == Some(x) {
                            insn.src[0]
                        } else {
                            insn.src[1]
                        };
                        if c.level < 4
                            && let Some(y) = v(other)
                            && let Some(&st) = staged.get(&y)
                            && st.chain == x
                            && st.level == c.level
                            && st.identity == c.identity
                        {
                            out.insert(
                                st.pc,
                                Reduction::OtherRow {
                                    identity: st.identity,
                                },
                            );
                            staged.remove(&y);
                            c.level += 1;
                            c.record_op(insn.op);
                            continue;
                        }
                        if c.level == 4
                            && v(other).is_some_and(|y| rows.get(&y) == Some(&(x, c.identity)))
                        {
                            c.level = 5;
                            c.record_op(insn.op);
                            continue;
                        }
                        if let Some((scalar, upper, false)) = c.half
                            && other == Operand::Reg(Reg::S(scalar))
                        {
                            c.half = Some((scalar, upper, true));
                            c.record_op(insn.op);
                            continue;
                        }
                    }
                }
                // Multi word unsigned sum reduction: `v_add_co_u32_e64` on the low word
                // produces a zero carry in a single lane projection, enabling `v_addc_co_u32`
                // on the high word at the same reduction level.
                (Op::v_add_co_u32_e64 | Op::v_addc_co_u32, _) if plain && no_mods => {
                    if let Some(x) = dst
                        && let Some(c) = chains.get_mut(&x)
                        && c.identity == 0
                        && (v(insn.src[0]) == Some(x) || v(insn.src[1]) == Some(x))
                    {
                        let other = if v(insn.src[1]) == Some(x) {
                            insn.src[0]
                        } else {
                            insn.src[1]
                        };
                        let carry_ok = if insn.op == Op::v_addc_co_u32 {
                            let cin = if insn.form == Form::AsVop3
                                || (insn.form == Form::Plain && insn.src[2] != Operand::None)
                            {
                                carry_reg(insn.src[2])
                            } else {
                                Some(VCC)
                            };
                            cin.is_some_and(|r| carry_zero == Some((r, c.level)))
                        } else {
                            true
                        };
                        if carry_ok
                            && c.level < 4
                            && let Some(y) = v(other)
                            && let Some(&st) = staged.get(&y)
                            && st.chain == x
                            && st.level == c.level
                            && st.identity == 0
                        {
                            let old_level = c.level;
                            out.insert(st.pc, Reduction::OtherRow { identity: 0 });
                            staged.remove(&y);
                            c.level += 1;
                            c.mixed_op = true;
                            carry_zero = carry_reg(insn.dst2).map(|r| (r, old_level));
                            if let Operand::Reg(Reg::S(r)) = insn.dst2 {
                                chains.retain(|_, ch| {
                                    ch.half.is_none_or(|(scalar, ..)| {
                                        !(r..=r.saturating_add(1)).contains(&scalar)
                                    })
                                });
                            }
                            continue;
                        }
                        if carry_ok
                            && c.level == 4
                            && v(other).is_some_and(|y| rows.get(&y) == Some(&(x, 0)))
                        {
                            let old_level = c.level;
                            c.level = 5;
                            c.mixed_op = true;
                            carry_zero = carry_reg(insn.dst2).map(|r| (r, old_level));
                            if let Operand::Reg(Reg::S(r)) = insn.dst2 {
                                chains.retain(|_, ch| {
                                    ch.half.is_none_or(|(scalar, ..)| {
                                        !(r..=r.saturating_add(1)).contains(&scalar)
                                    })
                                });
                            }
                            continue;
                        }
                    }
                }
                _ => {}
            }
            // Anything else: a write to a register ends what was known about it. Memory
            // instructions write several registers from the one they name.
            if matches!(
                insn.format(),
                Format::Mimg | Format::Mubuf | Format::Mtbuf | Format::Ds
            ) {
                chains.clear();
                rows.clear();
                staged.clear();
            }
            if insn.format() == Format::Vopc {
                carry_zero = None;
            }
            for op in [insn.dst, insn.dst2] {
                if let Some(x) = v(op) {
                    let next = x.wrapping_add(1);
                    chains.remove(&x);
                    rows.remove(&x);
                    staged.remove(&x);
                    // A 64 bit result takes the next register too.
                    chains.remove(&next);
                    rows.remove(&next);
                    staged.remove(&next);
                    rows.retain(|_, (ch, _)| *ch != x && *ch != next);
                    staged.retain(|_, s| s.chain != x && s.chain != next);
                }
                // The scalar that holds the result of a half must still hold it when it is
                // combined and read.
                if let Operand::Reg(Reg::S(r)) = op {
                    if carry_zero
                        .is_some_and(|(cr, _)| (r..=r.saturating_add(15)).contains(&cr))
                    {
                        carry_zero = None;
                    }
                    // A scalar load writes up to 16 registers from the one it names.
                    chains.retain(|_, c| {
                        c.half.is_none_or(|(scalar, ..)| {
                            !(r..=r.saturating_add(15)).contains(&scalar)
                        })
                    });
                }
                if matches!(op, Operand::Reg(Reg::VccLo | Reg::VccHi))
                    && carry_zero.is_some_and(|(cr, _)| cr == VCC)
                {
                    carry_zero = None;
                }
            }
        }
    }
    (out, wave_reduces)
}

/// The wave size the instructions that name EXEC or VCC are written for. `None` when the
/// program has no such instruction.
fn wave_size(cfg: &Cfg) -> Option<u8> {
    let mut found = None;
    for insn in cfg.insns() {
        if !matches!(insn.format(), Format::Sop1 | Format::Sop2) {
            continue;
        }
        let Some((kind, lanes)) = mask_op(insn.op) else {
            continue;
        };
        let evidence = match kind {
            MaskOp::SaveExec(_) | MaskOp::WrExec(_) | MaskOp::Wqm => true,
            MaskOp::Logic(_) | MaskOp::Select => {
                names(insn, Reg::ExecLo) || names(insn, Reg::VccLo)
            }
        };
        if evidence {
            // A 64 lane program also has 32 bit instructions that use VCC as a scratch register.
            if lanes == 64 {
                return Some(64);
            }
            found = Some(32);
        }
    }
    found
}

/// The touches of one instruction as a range of the flat list. The branch on VCC at the end of
/// a block has no instruction: its `pc` is `NO_PC`.
struct Span {
    pc: u32,
    start: usize,
    end: usize,
    /// The first of the two groups of the instruction (one per mask word), when it has
    /// touches that share a kind.
    groups: Option<u32>,
    /// Whether the second group has touches: a bitwise instruction of a 64 lane program.
    wide: bool,
    /// A parking instruction.
    parked: bool,
}

const NO_PC: u32 = u32::MAX;

/// The program as the typing sees it.
struct Typing<'a> {
    cfg: &'a Cfg,
    lanes: u8,
    touches: Vec<Touch>,
    spans: Vec<Vec<Span>>,
    groups: Vec<Group>,
    preds: Vec<Vec<usize>>,
}

impl Typing<'_> {
    fn own(&self, span: &Span) -> &[Touch] {
        &self.touches[span.start..span.end]
    }

    /// The kinds every location may hold at the start of each block, with the kinds the
    /// groups have now.
    fn forward(&self) -> Vec<Kinds> {
        let n = self.spans.len();
        let mut kill = vec![Bits::NONE; n];
        let mut made = vec![Kinds::NONE; n];
        for (b, list) in self.spans.iter().enumerate() {
            for t in list.iter().flat_map(|s| self.own(s)).filter(|t| t.def) {
                kill[b].put(t.loc, true);
                made[b].set(t.loc, written(t, &self.groups));
            }
        }
        // The scalar registers hold numbers at entry (what the root signature put there, or
        // nothing the program may rely on). VCC and the parking places hold nothing.
        let mut entry = Kinds::NONE;
        for r in 0..u16::from(VCC) {
            entry.set(r, NUMBER | SURE);
        }
        let mut ins = vec![Kinds::NONE; n];
        ins[0] = entry;
        let mut outs: Vec<Kinds> = (0..n).map(|b| ins[b].through(&kill[b], &made[b])).collect();
        loop {
            let mut changed = false;
            for b in 0..n {
                let mut new = if b == 0 { entry } else { Kinds::NONE };
                for &p in &self.preds[b] {
                    new.merge(&outs[p]);
                }
                if new != ins[b] {
                    ins[b] = new;
                    outs[b] = new.through(&kill[b], &made[b]);
                    changed = true;
                }
            }
            if !changed {
                return ins;
            }
        }
    }

    /// The kinds the readers of every location accept at the end of each block.
    fn backward(&self) -> Vec<Kinds> {
        let n = self.spans.len();
        let mut kill = vec![Bits::NONE; n];
        let mut made = vec![Kinds::NONE; n];
        for (b, list) in self.spans.iter().enumerate() {
            for span in list.iter().rev() {
                // A write ends the demand of the reads after it. The reads of the same
                // instruction come before its writes.
                for t in self.own(span).iter().filter(|t| t.def) {
                    kill[b].put(t.loc, true);
                    made[b].set(t.loc, 0);
                }
                for t in self.own(span).iter().filter(|t| !t.def) {
                    made[b].add(t.loc, wanted(t, &self.groups));
                }
            }
        }
        let successors = |b: usize| -> [Option<usize>; 2] {
            match self.cfg.blocks[b].term {
                Term::End => [None, None],
                Term::Jump(t) => [Some(t), None],
                Term::Branch { taken, fall, .. } => [Some(taken), Some(fall)],
            }
        };
        let mut outs = vec![Kinds::NONE; n];
        let mut ins: Vec<Kinds> = (0..n)
            .map(|b| outs[b].through(&kill[b], &made[b]))
            .collect();
        loop {
            let mut changed = false;
            for b in (0..n).rev() {
                let mut new = Kinds::NONE;
                for s in successors(b).into_iter().flatten() {
                    new.merge(&ins[s]);
                }
                if new != outs[b] {
                    outs[b] = new;
                    ins[b] = new.through(&kill[b], &made[b]);
                    changed = true;
                }
            }
            if !changed {
                return outs;
            }
        }
    }

    /// Narrows the groups by what their instructions read. Whether a group changed.
    fn narrow_by_sources(&mut self, ins: &[Kinds]) -> bool {
        let mut changed = false;
        for (b, start) in ins.iter().enumerate() {
            let mut cur = *start;
            for s in 0..self.spans[b].len() {
                let (start, end) = (self.spans[b][s].start, self.spans[b][s].end);
                for i in start..end {
                    let t = self.touches[i];
                    if !t.def && t.role == Role::Same {
                        changed |= self.groups[t.group as usize].narrow(cur.get(t.loc));
                    }
                }
                for i in start..end {
                    let t = self.touches[i];
                    if t.def {
                        cur.set(t.loc, written(&t, &self.groups));
                    }
                }
            }
        }
        changed
    }

    /// Which locations are read by any later instruction before being redefined.
    fn live_outs(&self) -> Vec<Bits> {
        let n = self.spans.len();
        let mut kill = vec![Bits::NONE; n];
        let mut used = vec![Bits::NONE; n];
        for (b, list) in self.spans.iter().enumerate() {
            for span in list.iter().rev() {
                for t in self.own(span).iter().filter(|t| t.def) {
                    kill[b].put(t.loc, true);
                    used[b].put(t.loc, false);
                }
                for t in self.own(span).iter().filter(|t| !t.def) {
                    used[b].put(t.loc, true);
                }
            }
        }
        let successors = |b: usize| -> [Option<usize>; 2] {
            match self.cfg.blocks[b].term {
                Term::End => [None, None],
                Term::Jump(t) => [Some(t), None],
                Term::Branch { taken, fall, .. } => [Some(taken), Some(fall)],
            }
        };
        let mut outs = vec![Bits::NONE; n];
        let mut ins = vec![Bits::NONE; n];
        for b in 0..n {
            for w in 0..ins[b].0.len() {
                ins[b].0[w] = (outs[b].0[w] & !kill[b].0[w]) | used[b].0[w];
            }
        }
        loop {
            let mut changed = false;
            for b in (0..n).rev() {
                let mut new = Bits::NONE;
                for s in successors(b).into_iter().flatten() {
                    for w in 0..new.0.len() {
                        new.0[w] |= ins[s].0[w];
                    }
                }
                if new != outs[b] {
                    outs[b] = new;
                    for w in 0..ins[b].0.len() {
                        ins[b].0[w] = (new.0[w] & !kill[b].0[w]) | used[b].0[w];
                    }
                    changed = true;
                }
            }
            if !changed {
                return outs;
            }
        }
    }

    /// Narrows the groups by what reads their results. Whether a group changed.
    fn narrow_by_readers(&mut self, outs: &[Kinds]) -> bool {
        let mut changed = false;
        for (b, end) in outs.iter().enumerate() {
            let mut cur = *end;
            for s in (0..self.spans[b].len()).rev() {
                let (start, end) = (self.spans[b][s].start, self.spans[b][s].end);
                for i in start..end {
                    let t = self.touches[i];
                    if t.def && t.role == Role::Same {
                        changed |= self.groups[t.group as usize].narrow(cur.get(t.loc));
                    }
                }
                for i in start..end {
                    let t = self.touches[i];
                    if t.def {
                        cur.set(t.loc, 0);
                    }
                }
                for i in start..end {
                    let t = self.touches[i];
                    if !t.def {
                        cur.add(t.loc, wanted(&t, &self.groups));
                    }
                }
            }
        }
        changed
    }

    /// For every read that takes either kind (`Role::Number`): the kinds the later readers
    /// of the same value accept, 0 when the instruction writes the location itself. Indexed
    /// like the touches.
    fn later_readers(&self, outs: &[Kinds]) -> Vec<u8> {
        let mut out = vec![0; self.touches.len()];
        for (b, end) in outs.iter().enumerate() {
            let mut cur = *end;
            for span in self.spans[b].iter().rev() {
                let own = self.own(span);
                for (i, t) in own.iter().enumerate() {
                    let rewritten = own.iter().any(|d| d.def && d.loc == t.loc);
                    if !t.def && t.role == Role::Number && !rewritten {
                        out[span.start + i] = cur.get(t.loc);
                    }
                }
                for t in own.iter().filter(|t| t.def) {
                    cur.set(t.loc, 0);
                }
                for t in own.iter().filter(|t| !t.def) {
                    cur.add(t.loc, wanted(t, &self.groups));
                }
            }
        }
        out
    }

    /// Checks whether a `Role::Number` zero test at `(target_block, target_span_idx)` can only
    /// receive a lane mask in `loc` along paths where `EXEC` is guaranteed empty and the `SCC`
    /// written by the compare is unobservable when `EXEC` is empty.
    fn inactive_mask_only(&self, target_block: usize, target_span_idx: usize, loc: u16) -> bool {
        let block = &self.cfg.blocks[target_block];
        let Some(target) = block.insns.get(target_span_idx) else {
            return false;
        };
        if !zero_test(target).is_some_and(|(_, w, _)| w == self.lanes) {
            return false;
        }
        // Step 1: confirm `SCC` written by `target` is only read by `s_cselect _, exec, 0` (or
        // `0, exec`) before being unconditionally overwritten in the same block with `EXEC`
        // untouched. Under empty `EXEC`, `select(scc, false, false)` is always false.
        let mut scc_killed = false;
        for insn in &block.insns[target_span_idx + 1..] {
            let info = insn.op.info();
            let writes_exec = matches!(insn.dst, Operand::Reg(Reg::ExecLo | Reg::ExecHi))
                || info.defs.iter().any(|s| s.fixed == Fixed::Exec);
            if writes_exec {
                return false;
            }
            let reads_scc = info.srcs.iter().any(|s| s.fixed == Fixed::Scc)
                || insn.src.iter().any(|o| matches!(o, Operand::Reg(Reg::Scc)));
            let writes_scc = info.defs.iter().any(|s| s.fixed == Fixed::Scc);
            if reads_scc {
                let exec_or_zero = matches!(mask_op(insn.op), Some((MaskOp::Select, w)) if w == self.lanes)
                    && ((insn.src[0] == Operand::Reg(Reg::ExecLo) && constant(insn.src[1]) == Some(0))
                        || (constant(insn.src[0]) == Some(0) && insn.src[1] == Operand::Reg(Reg::ExecLo)));
                if !exec_or_zero {
                    return false;
                }
            } else if writes_scc {
                scc_killed = true;
                break;
            }
        }
        if !scc_killed {
            return false;
        }
        // Step 2: find blocks guarded by `v_cmp_eq_u32 vcc, C, vK` followed by
        // `s_andn2 vcc, exec, vcc` and a `Cond::VccZero` branch. When `vK != C` in every active
        // lane, `vcc` equals `exec`, so the zero VCC edge guarantees empty `EXEC`.
        let plain_valu = |i: &Insn| {
            matches!(i.form, Form::Plain | Form::AsVop3)
                && matches!(
                    i.ext,
                    Ext::None
                        | Ext::Valu {
                            abs: 0,
                            neg: 0,
                            opsel: 0,
                            omod: 0,
                            clamp: false
                        }
                )
        };
        let mut guards: Vec<(u8, u32)> = Vec::new();
        let mut block_guard: Vec<Option<(usize, usize)>> = vec![None; self.cfg.blocks.len()];
        for (b, blk) in self.cfg.blocks.iter().enumerate() {
            if !matches!(
                blk.term,
                Term::Branch {
                    cond: Cond::VccZero(_),
                    ..
                }
            ) {
                continue;
            }
            let mut andn2_idx = None;
            for (i, insn) in blk.insns.iter().enumerate().rev() {
                if matches!(mask_op(insn.op), Some((MaskOp::Logic(Logic::Andn2), w)) if w == self.lanes)
                    && insn.dst == Operand::Reg(Reg::VccLo)
                    && insn.src[0] == Operand::Reg(Reg::ExecLo)
                    && insn.src[1] == Operand::Reg(Reg::VccLo)
                {
                    andn2_idx = Some(i);
                    break;
                }
                let info = insn.op.info();
                if matches!(
                    insn.dst,
                    Operand::Reg(Reg::VccLo | Reg::VccHi | Reg::ExecLo | Reg::ExecHi)
                ) || matches!(
                    insn.dst2,
                    Operand::Reg(Reg::VccLo | Reg::VccHi | Reg::ExecLo | Reg::ExecHi)
                ) || info.defs.iter().any(|s| s.fixed == Fixed::Exec)
                    || insn.format() == Format::Vopc
                {
                    break;
                }
            }
            let Some(andn2_at) = andn2_idx else {
                continue;
            };
            for (i, insn) in blk.insns[..andn2_at].iter().enumerate().rev() {
                if insn.op == Op::v_cmp_eq_u32
                    && plain_valu(insn)
                    && matches!(insn.dst, Operand::None | Operand::Reg(Reg::VccLo))
                {
                    let pair = match (insn.src[0], insn.src[1]) {
                        (Operand::Reg(Reg::V(v)), other) => constant(other).map(|c| (v, c)),
                        (other, Operand::Reg(Reg::V(v))) => constant(other).map(|c| (v, c)),
                        _ => None,
                    };
                    if let Some(pair) = pair {
                        let g_bit = if let Some(pos) = guards.iter().position(|p| *p == pair) {
                            Some(pos)
                        } else if guards.len() < 3 {
                            guards.push(pair);
                            Some(guards.len() - 1)
                        } else {
                            None
                        };
                        if let Some(g_bit) = g_bit {
                            block_guard[b] = Some((i, g_bit));
                        }
                    }
                    break;
                }
                let info = insn.op.info();
                if matches!(
                    insn.dst,
                    Operand::Reg(Reg::VccLo | Reg::VccHi | Reg::ExecLo | Reg::ExecHi)
                ) || matches!(
                    insn.dst2,
                    Operand::Reg(Reg::VccLo | Reg::VccHi | Reg::ExecLo | Reg::ExecHi)
                ) || info.defs.iter().any(|s| s.fixed == Fixed::Exec)
                    || insn.format() == Format::Vopc
                {
                    break;
                }
            }
        }
        // Step 3: bounded path sensitive reachability over `(has_mask, exec_zero, v_ne_mask)`
        // with intersected sets of scalar registers proven zero.
        let n_states = 4usize << guards.len();
        let state_index = |has_mask: bool, exec_zero: bool, v_ne: u8| -> usize {
            (usize::from(has_mask) << (guards.len() + 1))
                | (usize::from(exec_zero) << guards.len())
                | usize::from(v_ne)
        };
        let mut seen: Vec<Vec<Option<u128>>> = vec![vec![None; n_states]; self.cfg.blocks.len()];
        let mut queue: VecDeque<(usize, bool, bool, u8)> = VecDeque::new();
        let push = |seen: &mut [Vec<Option<u128>>],
                    queue: &mut VecDeque<(usize, bool, bool, u8)>,
                    b: usize,
                    has_mask: bool,
                    exec_zero: bool,
                    v_ne: u8,
                    zero_regs: u128| {
            let idx = state_index(has_mask, exec_zero, v_ne);
            match seen[b][idx] {
                None => {
                    seen[b][idx] = Some(zero_regs);
                    queue.push_back((b, has_mask, exec_zero, v_ne));
                }
                Some(old) => {
                    let merged = old & zero_regs;
                    if merged != old {
                        seen[b][idx] = Some(merged);
                        queue.push_back((b, has_mask, exec_zero, v_ne));
                    }
                }
            }
        };
        let wide_mask = self.lanes == 64;
        let is_zero_mask = |op: Operand, zero_regs: u128| -> bool {
            if constant(op) == Some(0) {
                return true;
            }
            let Some(r) = followed(op) else {
                return false;
            };
            (zero_regs & bit(r)) != 0
                && (!wide_mask || (usize::from(r) + 1 < REGS && (zero_regs & bit(r + 1)) != 0))
        };
        let v_written = |insn: &Insn, v_reg: u8| -> bool {
            if insn.op.name().starts_with("v_movrel")
                || matches!(
                    insn.format(),
                    Format::Flat | Format::Global | Format::Scratch
                )
            {
                return true;
            }
            let (d0, d1, _) = vector_spans(insn);
            for (op, count) in [(insn.dst, d0), (insn.dst2, d1)] {
                if let Operand::Reg(Reg::V(first)) = op
                    && v_reg >= first
                    && usize::from(v_reg - first) < count
                {
                    return true;
                }
            }
            false
        };
        push(&mut seen, &mut queue, 0, false, false, 0, 0);
        while let Some((b, mut has_mask, mut exec_zero, mut v_ne)) = queue.pop_front() {
            let idx = state_index(has_mask, exec_zero, v_ne);
            let mut zero_regs = seen[b][idx].unwrap_or(0);
            let mut vcc_eq_exec = false;
            let blk = &self.cfg.blocks[b];
            for (i, insn) in blk.insns.iter().enumerate() {
                if b == target_block && i == target_span_idx && has_mask && !exec_zero {
                    return false;
                }
                if let Some((cmp_i, g_bit)) = block_guard[b]
                    && cmp_i == i
                    && (exec_zero || (v_ne & (1 << g_bit)) != 0)
                {
                    vcc_eq_exec = true;
                }
                for (g_bit, &(v_k, c_test)) in guards.iter().enumerate() {
                    if insn.op == Op::v_mov_b32
                        && insn.form == Form::Plain
                        && insn.dst == Operand::Reg(Reg::V(v_k))
                        && constant(insn.src[0]).is_some_and(|c| c != c_test)
                    {
                        v_ne |= 1 << g_bit;
                    } else if v_written(insn, v_k) {
                        v_ne &= !(1 << g_bit);
                    }
                }
                let span = &self.spans[b][i];
                let own = self.own(span);
                let mut zero_defs = 0u128;
                let mut all_defs = 0u128;
                for t in own.iter().filter(|t| t.def) {
                    if t.loc == loc {
                        has_mask = written(t, &self.groups) & (LOW | HIGH) != 0;
                    }
                    if usize::from(t.loc) < REGS {
                        all_defs |= bit(t.loc as u8);
                    }
                }
                if insn.format() == Format::Vopc && !insn.op.name().starts_with("v_cmpx_") {
                    let dst_r = if insn.dst == Operand::None {
                        Some(VCC)
                    } else {
                        followed(insn.dst)
                    };
                    if let Some(r) = dst_r {
                        all_defs |= bit(r);
                        if wide_mask && usize::from(r) + 1 < REGS {
                            all_defs |= bit(r + 1);
                        }
                        if exec_zero {
                            zero_defs |= bit(r);
                            if wide_mask && usize::from(r) + 1 < REGS {
                                zero_defs |= bit(r + 1);
                            }
                        }
                    }
                } else if let Some((kind, w)) = mask_op(insn.op)
                    && w == self.lanes
                    && let Some(r) = followed(insn.dst)
                {
                    let writes_zero = match kind {
                        MaskOp::Logic(Logic::Mov) => {
                            (exec_zero && insn.src[0] == Operand::Reg(Reg::ExecLo))
                                || is_zero_mask(insn.src[0], zero_regs)
                        }
                        MaskOp::SaveExec(_) => exec_zero,
                        MaskOp::Logic(Logic::And) => {
                            (exec_zero
                                && (insn.src[0] == Operand::Reg(Reg::ExecLo)
                                    || insn.src[1] == Operand::Reg(Reg::ExecLo)))
                                || is_zero_mask(insn.src[0], zero_regs)
                                || is_zero_mask(insn.src[1], zero_regs)
                        }
                        MaskOp::Logic(Logic::Andn2) => {
                            (exec_zero && insn.src[0] == Operand::Reg(Reg::ExecLo))
                                || is_zero_mask(insn.src[0], zero_regs)
                        }
                        MaskOp::Select => {
                            let z0 = (exec_zero && insn.src[0] == Operand::Reg(Reg::ExecLo))
                                || is_zero_mask(insn.src[0], zero_regs);
                            let z1 = (exec_zero && insn.src[1] == Operand::Reg(Reg::ExecLo))
                                || is_zero_mask(insn.src[1], zero_regs);
                            z0 && z1
                        }
                        _ => false,
                    };
                    if writes_zero {
                        zero_defs |= bit(r);
                        if wide_mask && usize::from(r) + 1 < REGS {
                            zero_defs |= bit(r + 1);
                        }
                    }
                }
                zero_regs = (zero_regs & !all_defs) | zero_defs;
                let info = insn.op.info();
                let writes_exec = matches!(insn.dst, Operand::Reg(Reg::ExecLo | Reg::ExecHi))
                    || info.defs.iter().any(|s| s.fixed == Fixed::Exec);
                if writes_exec {
                    if insn.format() == Format::Vopc && insn.op.name().starts_with("v_cmpx_") {
                        // Narrows EXEC only.
                    } else if let Some((kind, w)) = mask_op(insn.op)
                        && w == self.lanes
                    {
                        match kind {
                            MaskOp::SaveExec(Logic::And) => {
                                if is_zero_mask(insn.src[0], zero_regs) {
                                    exec_zero = true;
                                }
                            }
                            MaskOp::Logic(Logic::And) if insn.dst == Operand::Reg(Reg::ExecLo) => {
                                if insn.src[0] == Operand::Reg(Reg::ExecLo) {
                                    if is_zero_mask(insn.src[1], zero_regs) {
                                        exec_zero = true;
                                    }
                                } else if insn.src[1] == Operand::Reg(Reg::ExecLo) {
                                    if is_zero_mask(insn.src[0], zero_regs) {
                                        exec_zero = true;
                                    }
                                } else if is_zero_mask(insn.src[0], zero_regs)
                                    || is_zero_mask(insn.src[1], zero_regs)
                                {
                                    exec_zero = true;
                                } else {
                                    exec_zero = false;
                                    v_ne = 0;
                                }
                            }
                            MaskOp::Logic(Logic::Andn2)
                                if insn.dst == Operand::Reg(Reg::ExecLo) =>
                            {
                                if insn.src[0] == Operand::Reg(Reg::ExecLo) {
                                    // Narrows EXEC only.
                                } else if is_zero_mask(insn.src[0], zero_regs) {
                                    exec_zero = true;
                                } else {
                                    exec_zero = false;
                                    v_ne = 0;
                                }
                            }
                            MaskOp::Logic(Logic::Mov)
                                if insn.dst == Operand::Reg(Reg::ExecLo)
                                    && is_zero_mask(insn.src[0], zero_regs) =>
                            {
                                exec_zero = true;
                            }
                            _ => {
                                exec_zero = false;
                                v_ne = 0;
                            }
                        }
                    } else {
                        exec_zero = false;
                        v_ne = 0;
                    }
                }
            }
            let vcc_zero = (zero_regs & bit(VCC)) != 0
                && (!wide_mask || (zero_regs & bit(VCC + 1)) != 0);
            match blk.term {
                Term::End => {}
                Term::Jump(t) => push(&mut seen, &mut queue, t, has_mask, exec_zero, v_ne, zero_regs),
                Term::Branch { cond, taken, fall } => match cond {
                    Cond::ExecZero(true) => {
                        push(&mut seen, &mut queue, taken, has_mask, true, v_ne, zero_regs);
                        if !exec_zero {
                            push(&mut seen, &mut queue, fall, has_mask, false, v_ne, zero_regs);
                        }
                    }
                    Cond::ExecZero(false) => {
                        if !exec_zero {
                            push(&mut seen, &mut queue, taken, has_mask, false, v_ne, zero_regs);
                        }
                        push(&mut seen, &mut queue, fall, has_mask, true, v_ne, zero_regs);
                    }
                    Cond::VccZero(false) if vcc_eq_exec => {
                        if !exec_zero {
                            push(&mut seen, &mut queue, taken, has_mask, false, v_ne, zero_regs);
                        }
                        push(&mut seen, &mut queue, fall, has_mask, true, v_ne, zero_regs);
                    }
                    Cond::VccZero(true) if vcc_eq_exec => {
                        push(&mut seen, &mut queue, taken, has_mask, true, v_ne, zero_regs);
                        if !exec_zero {
                            push(&mut seen, &mut queue, fall, has_mask, false, v_ne, zero_regs);
                        }
                    }
                    Cond::VccZero(false) if vcc_zero => {
                        push(&mut seen, &mut queue, fall, has_mask, exec_zero, v_ne, zero_regs);
                    }
                    Cond::VccZero(true) if vcc_zero => {
                        push(&mut seen, &mut queue, taken, has_mask, exec_zero, v_ne, zero_regs);
                    }
                    _ => {
                        push(&mut seen, &mut queue, taken, has_mask, exec_zero, v_ne, zero_regs);
                        push(&mut seen, &mut queue, fall, has_mask, exec_zero, v_ne, zero_regs);
                    }
                },
            }
        }
        true
    }

    /// What the instruction of `span` reads and writes as masks, with `cur` the kinds before it
    /// and `later` the answer of [`Typing::later_readers`].
    fn site(&self, block_idx: usize, span_idx: usize, span: &Span, cur: &mut Kinds, later: &[u8]) -> Site {
        let mut site = Site::default();
        // The first reason found stays: it is the one closest to the cause.
        fn refuse(site: &mut Site, what: &'static str) {
            site.refusal.get_or_insert(what);
        }
        let group_kind = |site: &mut Site, t: &Touch| match self.groups[t.group as usize].verdict()
        {
            Ok(kind) => kind,
            Err(what) => {
                refuse(site, what);
                NUMBER
            }
        };
        for (i, t) in self.own(span).iter().enumerate().filter(|(_, t)| !t.def) {
            let set = cur.get(t.loc) & ANY;
            let kind = match t.role {
                Role::Mask => {
                    // Every path that can really be taken brings the same kind, so one mask
                    // among the values that reach the read is enough. None at all is a number
                    // used as a mask.
                    if set != 0 && set & word_kind(t.half) == 0 {
                        refuse(&mut site, "data-as-mask");
                    }
                    word_kind(t.half)
                }
                Role::Data => {
                    if set != 0 && set & NUMBER == 0 {
                        refuse(&mut site, "mask-as-data");
                    }
                    NUMBER
                }
                Role::MaskWord => {
                    if set != 0 && set & NUMBER == 0 {
                        set
                    } else {
                        NUMBER
                    }
                }
                Role::Number => {
                    if set & NUMBER == 0 && set != 0 {
                        set
                    } else if set & (LOW | HIGH) == 0 {
                        NUMBER
                    } else {
                        // Values of both kinds reach the read, one of them along a path the
                        // program never takes. The instructions that read the same value
                        // later say which kind really arrives.
                        let readers = later[span.start + i] & ANY;
                        if readers == NUMBER {
                            NUMBER
                        } else if readers != 0 && readers & NUMBER == 0 {
                            set & readers
                        } else if self.inactive_mask_only(block_idx, span_idx, t.loc) {
                            NUMBER
                        } else {
                            refuse(&mut site, "mixed-pair");
                            NUMBER
                        }
                    }
                }
                Role::Same => group_kind(&mut site, t),
            };
            if usize::from(t.loc) >= REGS {
                site.parked = kind;
            } else if kind & (LOW | HIGH) != 0 {
                site.reads |= bit(t.loc as u8);
                if kind == HIGH {
                    site.high_reads |= bit(t.loc as u8);
                }
            }
        }
        for t in self.own(span).iter().filter(|t| t.def) {
            let kind = match t.role {
                Role::Mask => word_kind(t.half),
                Role::Data | Role::Number | Role::MaskWord => NUMBER,
                Role::Same => group_kind(&mut site, t),
            };
            if usize::from(t.loc) >= REGS {
                site.parked = kind;
            } else if kind & (LOW | HIGH) != 0 {
                site.writes |= bit(t.loc as u8);
            }
            cur.set(t.loc, written(t, &self.groups));
        }
        if let Some(base) = span.groups {
            let low = self.groups[base as usize].verdict().unwrap_or(NUMBER);
            if !span.parked {
                // A bitwise instruction works on whole masks: the low word in the low half,
                // and in a 64 lane program the high word in the high half.
                let high_group = &self.groups[base as usize + 1];
                let high = high_group.verdict().unwrap_or(NUMBER);
                let fits = match (low, span.wide) {
                    (NUMBER, false) | (LOW, false) => true,
                    (NUMBER, true) => high == NUMBER,
                    (LOW, true) => high == HIGH || !high_group.def_live,
                    _ => false,
                };
                if !fits {
                    refuse(&mut site, "mixed-pair");
                }
            }
        }
        site
    }
}

/// The typing for one assumed wave size. The second result counts the refused instructions.
fn analyse_for(cfg: &Cfg, lanes: u8, parking: &Parking) -> Result<(Masks, usize), Refusal> {
    let mut touches = Vec::new();
    let mut groups = Vec::new();
    let mut spans: Vec<Vec<Span>> = Vec::with_capacity(cfg.blocks.len());
    let mut close = |touches: &mut Vec<Touch>, pc: u32, start: usize, parked: bool| {
        let own = &mut touches[start..];
        let shared = own.iter().any(|t| t.role == Role::Same);
        let base = groups.len() as u32;
        if shared {
            groups.extend(
                [Group {
                    kinds: ANY,
                    sure: false,
                    clash: false,
                    def_live: false,
                }; 2],
            );
            for t in own.iter_mut().filter(|t| t.role == Role::Same) {
                t.group = base + u32::from(t.half.min(1));
            }
        }
        Span {
            pc,
            start,
            end: touches.len(),
            groups: shared.then_some(base),
            wide: touches[start..]
                .iter()
                .any(|t| t.role == Role::Same && t.half != 0),
            parked,
        }
    };
    for block in &cfg.blocks {
        let mut list = Vec::with_capacity(block.insns.len() + 1);
        for insn in &block.insns {
            let start = touches.len();
            Collector {
                lanes,
                parking,
                out: &mut touches,
            }
            .insn(insn);
            let parked = parking.slot(insn).is_some();
            list.push(close(&mut touches, insn.pc, start, parked));
        }
        // A branch on VCC reads it as a mask.
        if let Term::Branch {
            cond: Cond::VccZero(_),
            ..
        } = block.term
        {
            let start = touches.len();
            Collector {
                lanes,
                parking,
                out: &mut touches,
            }
            .uses(Operand::Reg(Reg::VccLo), lanes / 32, Role::Mask);
            list.push(close(&mut touches, NO_PC, start, false));
        }
        spans.push(list);
    }
    let mut typing = Typing {
        cfg,
        lanes,
        touches,
        spans,
        groups,
        preds: cfg.predecessors(),
    };
    // Each round carries what is known one step further: along a block at once, across blocks
    // one block per round. The groups only lose kinds, so this ends.
    for _ in 0..ROUNDS {
        let ins = typing.forward();
        let mut changed = typing.narrow_by_sources(&ins);
        let outs = typing.backward();
        changed |= typing.narrow_by_readers(&outs);
        if !changed {
            break;
        }
    }
    let live_outs = typing.live_outs();
    for (b, end) in live_outs.iter().enumerate() {
        let mut live = *end;
        for span in typing.spans[b].iter().rev() {
            let own = &typing.touches[span.start..span.end];
            for t in own.iter().filter(|t| t.def && t.role == Role::Same) {
                if live.has(t.loc) {
                    typing.groups[t.group as usize].def_live = true;
                }
            }
            for t in own.iter().filter(|t| t.def) {
                live.put(t.loc, false);
            }
            for t in own.iter().filter(|t| !t.def) {
                live.put(t.loc, true);
            }
        }
    }
    // An ambiguous group nothing sure reaches, whose result is never read, or where mixed mask and
    // numeric provenance reaches with no reader requiring a mask is a number from here on. Groups
    // already narrowed to a single kind stay as they are: turning an unambiguous dead mask copy or
    // restore into a number would contradict its live mask producer without helping any reader.
    for g in &mut typing.groups {
        if !g.clash
            && g.kinds.count_ones() != 1
            && (!g.def_live || !g.sure || (g.kinds & NUMBER) != 0)
        {
            g.kinds = NUMBER;
            g.sure = false;
        }
    }
    let ins = typing.forward();
    let later = typing.later_readers(&typing.backward());
    let mut masks = Masks {
        lanes,
        converged: cfg.blocks.len() == 1,
        ..Masks::default()
    };
    for &(reg, _) in parking.slots.keys() {
        masks.parking[usize::from(reg / 64)] |= 1 << (reg % 64);
    }
    let mut refused = 0;
    for (b, list) in typing.spans.iter().enumerate() {
        let mut cur = ins[b];
        for (span_idx, span) in list.iter().enumerate() {
            let site = typing.site(b, span_idx, span, &mut cur, &later);
            if site.refusal.is_some() {
                refused += 1;
                if span.pc == NO_PC {
                    return Err(Refusal::new(
                        RefusalKind::Exec("data-as-mask"),
                        "a branch on VCC that holds a number",
                    )
                    .at(cfg.blocks[b].start));
                }
            }
            if span.pc != NO_PC && site != Site::default() {
                masks.sites.insert(span.pc, site);
            }
        }
    }
    let (red_parts, wave_reduces) = reductions(cfg, lanes);
    let (scan_parts, wave_scans) = scans(cfg, lanes);
    for (pc, part) in red_parts {
        masks.sites.entry(pc).or_default().reduction = Some(part);
    }
    for (pc, part) in wave_reduces {
        masks.sites.entry(pc).or_default().wave_reduce = Some(part);
    }
    for (pc, part) in scan_parts {
        masks.sites.entry(pc).or_default().reduction = Some(part);
    }
    for (pc, part) in wave_scans {
        masks.sites.entry(pc).or_default().scan = Some(part);
    }
    // Enable real HLSL wave lowering for matched scans and reductions when the program also
    // performs an explicit constant cross lane `v_readlane_b32` outside those patterns and every
    // matched reduction result has a real wave lowering.
    let has_wave_pattern = masks
        .sites
        .values()
        .any(|s| s.scan.is_some() || s.wave_reduce.is_some());
    if has_wave_pattern {
        let has_cross_lane_read = cfg.insns().any(|insn| {
            insn.op == Op::v_readlane_b32_e64
                && matches!(insn.src[1], Operand::Int(l) if (0..i8::try_from(lanes).unwrap_or(0)).contains(&l))
                && parking.slot(insn).is_none()
                && masks
                    .sites
                    .get(&insn.pc)
                    .is_none_or(|s| s.reduction.is_none() && s.scan.is_none())
        });
        let all_results_supported = masks.sites.values().all(|s| {
            !matches!(s.reduction, Some(Reduction::Result { .. }))
                || s.scan.is_some()
                || s.wave_reduce.is_some()
        });
        masks.real_wave = has_cross_lane_read && all_results_supported;
    }
    if lanes == 64 && !masks.converged {
        mark_vcmp_ballots(cfg, &typing, &mut masks);
    }
    Ok((masks, refused))
}

/// Marks 64 lane `v_cmp_*` definitions whose scalar destination words are read directly as
/// numeric ballot words by `s_mov_b32` or `s_cselect_b32` without any intervening overwrite.
/// Because `v_cmp_*` masks its output by EXEC, inactive HLSL invocations already have zero
/// bits, so materialising `WaveActiveBallot` at the `v_cmp_*` definition preserves the exact
/// 32 bit words even when later branches or loop iterations read one half at a time.
fn mark_vcmp_ballots(cfg: &Cfg, typing: &Typing<'_>, masks: &mut Masks) {
    let is_vcmp_pair = |insn: &Insn| -> Option<u8> {
        if insn.format() == Format::Vopc
            && insn.op.name().starts_with("v_cmp_")
            && let Operand::Reg(Reg::S(n)) = insn.dst
            && n + 1 < VCC
        {
            return Some(n);
        }
        None
    };
    let reads_word_mask = |insn: &Insn, masks: &Masks| -> bool {
        matches!(insn.op, Op::s_mov_b32 | Op::s_cselect_b32)
            && !names(insn, Reg::ExecLo)
            && masks.sites.get(&insn.pc).is_some_and(|s| s.reads != 0)
    };
    if !cfg.insns().any(|insn| reads_word_mask(insn, masks)) {
        return;
    }
    let n = cfg.blocks.len();
    let mut ins = vec![0u128; n];
    let mut outs = vec![0u128; n];
    let mut seen = vec![false; n];
    let transfer = |b: usize, mut cur: u128| -> u128 {
        for (idx, insn) in cfg.blocks[b].insns.iter().enumerate() {
            let span = &typing.spans[b][idx];
            if let Some(first) = is_vcmp_pair(insn) {
                cur |= bit(first) | bit(first + 1);
            } else {
                for t in typing.own(span).iter().filter(|t| t.def) {
                    if usize::from(t.loc) < usize::from(VCC) {
                        cur &= !bit(t.loc as u8);
                    }
                }
            }
        }
        cur
    };
    seen[0] = true;
    outs[0] = transfer(0, 0);
    loop {
        let mut changed = false;
        for b in 0..n {
            let mut incoming = if b == 0 {
                0u128
            } else {
                let mut acc = None;
                for &p in &typing.preds[b] {
                    if seen[p] {
                        acc = Some(acc.map_or(outs[p], |old: u128| old & outs[p]));
                    }
                }
                let Some(v) = acc else { continue };
                v
            };
            if !seen[b] || incoming != ins[b] {
                seen[b] = true;
                ins[b] = incoming;
                incoming = transfer(b, incoming);
                if incoming != outs[b] {
                    outs[b] = incoming;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut needed_regs = 0u128;
    for b in 0..n {
        if !seen[b] {
            continue;
        }
        let mut cur = ins[b];
        for (idx, insn) in cfg.blocks[b].insns.iter().enumerate() {
            if reads_word_mask(insn, masks) {
                let reads = masks.sites.get(&insn.pc).map_or(0, |s| s.reads);
                if reads != 0 && (reads & !cur) == 0 {
                    needed_regs |= reads;
                    masks.sites.entry(insn.pc).or_default().ballot_ready = true;
                }
            }
            let span = &typing.spans[b][idx];
            if let Some(first) = is_vcmp_pair(insn) {
                cur |= bit(first) | bit(first + 1);
            } else {
                for t in typing.own(span).iter().filter(|t| t.def) {
                    if usize::from(t.loc) < usize::from(VCC) {
                        cur &= !bit(t.loc as u8);
                    }
                }
            }
        }
    }
    if needed_regs == 0 {
        return;
    }
    for insn in cfg.insns() {
        if let Some(first) = is_vcmp_pair(insn) {
            let lo = u8::from(needed_regs & bit(first) != 0);
            let hi = u8::from(needed_regs & bit(first + 1) != 0);
            let words = lo | (hi << 1);
            if words != 0 {
                masks.sites.entry(insn.pc).or_default().materialize_ballot = words;
            }
        }
    }
}

/// How many rounds the typing runs at most. A program that needs more keeps what it has: the
/// groups left open are refused or numbers, never guessed.
const ROUNDS: usize = 48;

/// Finds where the scalar registers of a program hold masks.
pub fn analyse(cfg: &Cfg) -> Result<Masks, Refusal> {
    let parking = Parking::find(cfg);
    if let Some(lanes) = wave_size(cfg) {
        return analyse_for(cfg, lanes, &parking).map(|(m, _)| m);
    }
    // No instruction names EXEC or VCC with a width. The size that explains the program with
    // fewer contradictions is taken, 64 lanes when both do.
    match (
        analyse_for(cfg, 64, &parking),
        analyse_for(cfg, 32, &parking),
    ) {
        (Ok((wide, w)), Ok((_, n))) if w <= n => Ok(wide),
        (_, Ok((narrow, _))) => Ok(narrow),
        (wide, Err(_)) => wide.map(|(m, _)| m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::*;

    fn analyse_asm(mut a: Asm) -> (crate::cfg::Cfg, Masks) {
        a.push(sopp(Op::s_endpgm, 0));
        let graph = crate::cfg::build(&a.finish()).unwrap();
        let masks = analyse(&graph).unwrap();
        (graph, masks)
    }

    /// In a 64 lane program, `s_mov_b32` on a live 64 lane mask word reads the mask word and
    /// writes a numeric scalar (`NUMBER`) so `lift::wave` can extract it via `WaveActiveBallot`.
    #[test]
    fn move_b32_of_live_64_lane_mask_word_reads_mask_and_writes_number() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, s(26), EXEC));
        a.push(sop1(Op::s_mov_b32, s(40), s(26)));
        let (graph, masks) = analyse_asm(a);
        assert_eq!(masks.lanes(), 64);
        let mov = graph.insns().find(|i| i.op == Op::s_mov_b32).unwrap();
        masks.enter(mov.pc);
        assert_eq!(masks.refusal(), None);
        assert!(masks.reads_mask(26));
        assert!(!masks.writes_mask(40));
    }

    #[test]
    fn move_b32_allows_register_when_numeric_path_also_reaches_source() {
        let mut a = Asm::new();
        let numeric = a.label();
        let join = a.label();
        a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
        a.branch(Op::s_cbranch_scc1, numeric);
        a.push(sop1(Op::s_mov_b64, s(26), EXEC));
        a.branch(Op::s_branch, join);
        a.bind(numeric);
        a.push(sop1(Op::s_mov_b32, s(26), int(7)));
        a.bind(join);
        a.push(sop1(Op::s_mov_b32, s(40), s(26)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }

    #[test]
    fn move_b32_propagates_32_lane_mask_cleanly() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(26), EXEC));
        a.push(sop1(Op::s_mov_b32, s(40), s(26)));
        a.push(sop1(Op::s_mov_b32, EXEC, s(40)));
        let (graph, masks) = analyse_asm(a);
        assert_eq!(masks.lanes(), 32);
        let mov = graph.insns().nth(1).unwrap();
        masks.enter(mov.pc);
        assert!(masks.reads_mask(26));
        assert!(masks.writes_mask(40));
        assert_eq!(masks.refusal(), None);
    }

    #[test]
    fn move_b64_with_mixed_provenance_and_dead_result_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        let numeric = a.label();
        let join = a.label();
        a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
        a.branch(Op::s_cbranch_scc1, numeric);
        a.push(sop1(Op::s_mov_b64, s(32), EXEC));
        a.branch(Op::s_branch, join);
        a.bind(numeric);
        a.push(sop1(Op::s_mov_b64, s(32), int(0)));
        a.bind(join);
        a.push(sop1(Op::s_mov_b64, s(62), s(32)));
        let (graph, masks) = analyse_asm(a);
        assert_eq!(masks.lanes(), 64);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }

    #[test]
    fn writelane_with_mixed_provenance_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        let numeric = a.label();
        let join = a.label();
        a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
        a.branch(Op::s_cbranch_scc1, numeric);
        a.push(sop1(Op::s_mov_b64, s(2), EXEC));
        a.branch(Op::s_branch, join);
        a.bind(numeric);
        a.push(sop1(Op::s_mov_b32, s(2), int(7)));
        a.bind(join);
        a.push(vop3(Op::v_writelane_b32_e64, v(128), s(2), int(0), int(0)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }

    #[test]
    fn move_b64_with_live_low_mask_and_dead_high_word_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, s(32), EXEC));
        a.push(sop1(Op::s_mov_b64, s(62), s(32)));
        a.push(sop1(Op::s_mov_b32, s(40), s(62)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }

    #[test]
    fn writelane_readlane_chain_with_mixed_provenance_and_dead_result_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        let numeric = a.label();
        let join = a.label();
        a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
        a.branch(Op::s_cbranch_scc1, numeric);
        a.push(sop1(Op::s_mov_b64, s(2), EXEC));
        a.branch(Op::s_branch, join);
        a.bind(numeric);
        a.push(sop1(Op::s_mov_b32, s(2), int(7)));
        a.bind(join);
        a.push(vop3(Op::v_writelane_b32_e64, v(128), s(2), int(0), int(0)));
        a.push(vop3(Op::v_readlane_b32_e64, s(3), v(128), int(0), int(0)));
        a.push(sop1(Op::s_mov_b32, s(4), s(3)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }

    #[test]
    fn a_dead_restore_of_a_parked_mask_does_not_refuse_mixed_pair() {
        // 32 lanes: EXEC is saved and parked around a branch, taken back after it and then
        // overwritten with a number before anything reads it.
        let mut a = Asm::new();
        let skip = a.label();
        a.push(vopc(Op::v_cmp_lg_u32, int(0), v(1)));
        a.push(sop1(Op::s_and_saveexec_b32, s(2), crate::asm::VCC));
        a.push(vop3(Op::v_writelane_b32_e64, v(128), s(2), int(0), Operand::None));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop1(Op::v_mov_b32, v(3), int(1)));
        a.bind(skip);
        a.push(vop3(Op::v_readlane_b32_e64, s(2), v(128), int(0), Operand::None));
        a.push(sop1(Op::s_mov_b32, s(2), int(7)));
        a.push(vop1(Op::v_mov_b32, v(4), s(2)));
        let (graph, masks) = analyse_asm(a);
        assert_eq!(masks.lanes(), 32);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
            if insn.op == Op::v_writelane_b32_e64 {
                assert_eq!(masks.parked(), Some(Parked::Mask), "{insn}");
            }
            if insn.op == Op::v_readlane_b32_e64 {
                assert_eq!(masks.parked(), Some(Parked::Mask), "{insn}");
            }
        }
    }

    #[test]
    fn a_dead_scalar_mask_restore_chain_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(71), EXEC));
        a.push(sop1(Op::s_mov_b32, s(24), s(71)));
        a.push(sop1(Op::s_mov_b32, s(71), s(24)));
        let (graph, masks) = analyse_asm(a);
        assert_eq!(masks.lanes(), 32);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
        let save = graph.insns().nth(1).unwrap();
        masks.enter(save.pc);
        assert!(masks.reads_mask(71));
        assert!(masks.writes_mask(24));
        let restore = graph.insns().nth(2).unwrap();
        masks.enter(restore.pc);
        assert!(masks.reads_mask(24));
        assert!(masks.writes_mask(71));
    }

    #[test]
    fn a_dead_destination_mask_and_with_live_scc_stays_a_mask() {
        let mut a = Asm::new();
        let done = a.label();
        a.push(sop1(Op::s_mov_b32, s(71), EXEC));
        a.push(sop2(Op::s_and_b32, s(24), s(71), EXEC));
        a.branch(Op::s_cbranch_scc0, done);
        a.push(vop1(Op::v_mov_b32, v(1), int(1)));
        a.bind(done);
        let (graph, masks) = analyse_asm(a);
        let and = graph.insns().find(|i| i.op == Op::s_and_b32).unwrap();
        masks.enter(and.pc);
        assert_eq!(masks.refusal(), None);
        assert!(masks.reads_mask(71));
        assert!(masks.writes_mask(24));
    }

    #[test]
    fn scratch_vcc_64_bit_move_lifts_and_preserves_mask_rejection() {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b64, s(0), int(7)));
        a.push(sop1(Op::s_mov_b64, crate::asm::VCC, s(0)));
        a.push(sop1(Op::s_mov_b64, s(0), EXEC));
        a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
        a.push(sop1(Op::s_mov_b64, s(0), crate::asm::VCC));
        a.push(vop1(Op::v_mov_b32, v(2), s(0)));
        a.push(vop1(Op::v_mov_b32, v(3), s(1)));
        a.push(sopp(Op::s_endpgm, 0));
        let prog = crate::api::StageProgram {
            threads: [64, 1, 1],
            ..crate::api::StageProgram::new(crate::api::Stage::Compute, a.finish())
        };
        let text = crate::lift_program(&prog).unwrap().to_string();
        assert!(text.contains("s106"), "{text}");
        assert!(text.contains("s107"), "{text}");

        // VCC holding a real mask in a numeric 64 bit shift is still refused.
        let mut b = Asm::new();
        b.push(vopc(Op::v_cmp_lt_u32, v(0), int(4)));
        b.push(sop2(Op::s_lshl_b64, s(4), crate::asm::VCC, int(1)));
        b.push(sopp(Op::s_endpgm, 0));
        let bad = crate::api::StageProgram {
            threads: [64, 1, 1],
            ..crate::api::StageProgram::new(crate::api::Stage::Compute, b.finish())
        };
        let refusal = crate::lift_program(&bad).unwrap_err();
        assert_eq!(refusal.key(), "exec:mask-as-data");
    }

    #[test]
    fn a_used_restore_of_a_parked_mask_stays_a_mask() {
        let mut a = Asm::new();
        let skip = a.label();
        a.push(vopc(Op::v_cmp_lg_u32, int(0), v(1)));
        a.push(sop1(Op::s_and_saveexec_b32, s(2), crate::asm::VCC));
        a.push(vop3(Op::v_writelane_b32_e64, v(128), s(2), int(0), Operand::None));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop1(Op::v_mov_b32, v(3), int(1)));
        a.bind(skip);
        a.push(vop3(Op::v_readlane_b32_e64, s(2), v(128), int(0), Operand::None));
        a.push(sop1(Op::s_mov_b32, EXEC, s(2)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
            if matches!(insn.op, Op::v_writelane_b32_e64 | Op::v_readlane_b32_e64) {
                assert_eq!(masks.parked(), Some(Parked::Mask), "{insn}");
            }
        }
    }

    #[test]
    fn move_b64_chain_with_mixed_provenance_and_dead_result_does_not_refuse_mixed_pair() {
        let mut a = Asm::new();
        let numeric = a.label();
        let join = a.label();
        a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
        a.branch(Op::s_cbranch_scc1, numeric);
        a.push(sop1(Op::s_mov_b64, s(32), EXEC));
        a.branch(Op::s_branch, join);
        a.bind(numeric);
        a.push(sop1(Op::s_mov_b64, s(32), int(7)));
        a.bind(join);
        a.push(sop1(Op::s_mov_b64, s(62), s(32)));
        a.push(sop1(Op::s_mov_b64, s(64), s(62)));
        let (graph, masks) = analyse_asm(a);
        for insn in graph.insns() {
            masks.enter(insn.pc);
            assert_eq!(masks.refusal(), None, "{insn}");
        }
    }
}
