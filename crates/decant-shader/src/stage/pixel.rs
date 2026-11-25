//! The pixel stage.
//!
//! Facts measured on the title's 12,101 pixel programs (24 Nov 2026), the basis of this file:
//!
//! * **Attributes.** The FSSM block of every pixel container holds 32 records of 8 bytes at byte
//!   0x138 (`api::PixelInput`): attribute `k` of the interpolation instructions starts with
//!   signature register `register`. More 16 bit registers can follow in the same attribute
//!   ([`walk`]). With that layout every interpolation read of 12,093 programs lies inside the
//!   declared mask of its register. Six programs read a padding half past a 16 bit element,
//!   which reads zero here, and two read a register the console split into a flat and an
//!   interpolated attribute (refused as `stage:attr-unknown`). The older rule "attribute `k` is the register of rank `k` among the registers with a
//!   non zero read mask" holds for 7,722 programs only: the console compiler gives attributes
//!   to some registers whose read mask is zero, and packs two 16 bit registers into one
//!   attribute. That rule is kept as the fallback for programs without a container (packs and
//!   the tests), with adjacent 16 bit registers paired.
//! * **16 bit attributes.** Half `h` of a 16 bit attribute lies in channel `h / 2` (low half for
//!   even `h`). An interpolated 16 bit register (record flag 0x10) packs its components into
//!   halves, and the next 16 bit register follows at the next whole channel: after a register
//!   with two components at half 2, after one with three or four at half 4. A flat 16 bit
//!   register without flag 0x10 has one component per channel, the half float in the low half
//!   (programs convert it with `v_cvt_f32_f16`). A 32 bit read of a packed channel gets both
//!   halves.
//! * **Linking.** Attribute `k` is a register, and the member is the element of that register
//!   whose mask has the component, found by name in the struct made from the stage before.
//!   When the two stages place the element at different components of their registers, the
//!   member component is the component minus the element's first component here plus the
//!   member's first component there.
//! * **Interpolation modes.** `PSV0` lists only two modes for ordinary inputs: constant (6,333
//!   elements) and perspective at the pixel centre (79,577). An attribute whose record is flat
//!   (flag 0x01) becomes a `nointerpolation` member.
//! * **Entry registers.** The barycentric pair is always perspective centre in `v0`, `v1`, and
//!   it is allocated even when the program interpolates nothing (1,841 programs). Then follow
//!   the components of `SV_Position` the program reads (by the read mask, x to w), then the
//!   front face. The layout was checked against the registers read before written in 10,157
//!   programs, the rest are artefacts of that linear scan. The front face register is non zero
//!   for a front face (programs test it with `v_cmp_lg_u32 0`), here 0 or all bits. The
//!   position w register holds 1/w, while Direct3D's `SV_Position.w` is w.
//! * **Exports.** Every colour export of the title is compressed (16 bit formats): two halves
//!   per register, red and green in the first, blue and alpha in the second. A float target
//!   gets half floats (FP16), an integer target 16 bit integers. Export target `n` is the `n`-th
//!   colour target the program writes, in ascending index (the hardware compacts the targets:
//!   221 programs write for example SV_Target0 and SV_Target2 with targets 0 and 1). Target 8
//!   carries the depth in channel 0 (1,171 programs, `SV_Depth` or a conservative form), the
//!   stencil reference in 1 and the coverage in 2. Target 9 is the null target the kill exit
//!   uses. With the valid mask bit an export also carries EXEC as the kill mask: a lane that is
//!   not active there is a discarded pixel.
//!
//! The lifter names attributes by their hardware number ([`crate::lift::interp`]) and the entry
//! registers by [`Input::FragCoord`] and [`Input::FrontFace`]. [`finish`] turns all of them into
//! reads of input struct members: [`Interface::attrs`] is the identity here, so an
//! [`Input::Attr`] after `finish` names a member and its component.

use super::{Field, Interface, field, ordered, refuse};
use crate::api::{PixelInput, SigElement, StageProgram};
use crate::decode::{self, Ext};
use crate::ir::{Bin, Export, Expr, Function, Guard, Helper, Input, OutputRef, Stmt, Ty, Un, Var};
use crate::lift::interp::HALF;
use crate::refusal::Refusal;
use crate::tables::Op;

static F16: Helper = Helper {
    name: "decant_f16_to_f32",
    result: Ty::U32,
    text: "uint decant_f16_to_f32(uint a) { return asuint(f16tof32(a & 0xffffu)); }",
    needs: &[],
};

static TO_F16: Helper = Helper {
    name: "decant_pixel_to_f16",
    result: Ty::U32,
    text: "uint decant_pixel_to_f16(uint a) { return f32tof16(asfloat(a)); }",
    needs: &[],
};

static RCP: Helper = Helper {
    name: "decant_pixel_rcp",
    result: Ty::U32,
    text: "uint decant_pixel_rcp(uint a) { return asuint(1.0f / asfloat(a)); }",
    needs: &[],
};

/// System value codes of the signatures.
const SV_POSITION: u32 = 1;
const SV_PRIMITIVE_ID: u32 = 7;
const SV_IS_FRONT_FACE: u32 = 9;
const SV_SAMPLE_INDEX: u32 = 10;
const SV_TARGET: u32 = 64;
const SV_DEPTH: u32 = 65;
const SV_COVERAGE: u32 = 66;
const SV_DEPTH_GREATER_EQUAL: u32 = 67;
const SV_DEPTH_LESS_EQUAL: u32 = 68;
const SV_STENCIL_REF: u32 = 69;

/// Flag bits of a [`PixelInput`] record.
const FLAT: u8 = 0x01;
const SIXTEEN: u8 = 0x08;
const PACKED: u8 = 0x10;

/// The attribute table: the container's, or the fallback rule when the program has none.
fn attributes(program: &StageProgram) -> Vec<PixelInput> {
    if !program.pixel_inputs.is_empty() {
        return program.pixel_inputs.clone();
    }
    // Registers with ordinary elements that are read, with their merged read mask and whether
    // they hold 16 bit elements.
    let mut regs: Vec<(u32, u8, bool)> = Vec::new();
    for e in ordered(&program.inputs) {
        if e.system_value != 0 || e.rw_mask == 0 {
            continue;
        }
        let sixteen = matches!(e.component_type, 4..=6);
        match regs.last_mut() {
            Some(r) if r.0 == e.register => {
                r.1 |= e.rw_mask;
                r.2 |= sixteen;
            }
            _ => regs.push((e.register, e.rw_mask, sixteen)),
        }
    }
    let mut out: Vec<PixelInput> = Vec::new();
    let mut open = false;
    for (register, _, sixteen) in regs {
        if sixteen && open {
            let last = out.last_mut().expect("an open pair has its first register");
            last.high_register = Some(register);
            open = false;
            continue;
        }
        out.push(PixelInput {
            attr: out.len() as u8,
            kind: 6,
            register,
            high_register: None,
            channels: 0xf,
            flags: if sixteen { SIXTEEN | 0x12 } else { 0x02 },
        });
        open = sixteen;
    }
    out
}

/// How a signature register lies in its attribute (pixel) or parameter slot (vertex).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layout {
    /// One 32 bit component per channel.
    Wide,
    /// A 16 bit register that is not interpolated with the 16 bit instructions (the flat ones):
    /// one component per channel, the half float in the low half.
    Unpacked,
    /// Packed halves: component `c` is half `start + c` of the attribute.
    Packed,
}

/// One signature register in an attribute or parameter slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Place {
    pub slot: u8,
    pub register: u32,
    /// The first half of the slot the register takes (packed layout only).
    pub start: u8,
    pub layout: Layout,
}

/// One register for [`walk`]: its merged declared mask, whether it is 16 bit, and the slot its
/// record opens with whether that slot is packed, when the container has a record for it.
pub(super) struct Walked {
    pub register: u32,
    pub mask: u8,
    pub sixteen: bool,
    pub opens: Option<(u8, bool)>,
}

/// The packing both sides of the interface use (measured on the pixel input table at FSSM
/// 0x138 and, by the mesh package, on the parameter export table at 0x1DC). In register
/// order: a register with a record takes its own slot, at the start. A packed 16 bit register
/// with a record opens a packed slot. A 16 bit register without a record goes into the open
/// packed slot at the next whole channel, its components rounded up to whole channels, while
/// it fits in four channels. The open packed slot stays open across 32 bit registers. With this
/// layout every interpolation read of 12,093 of the 12,101 pixel programs lies inside the
/// declared mask of its register.
pub(super) fn walk(registers: impl IntoIterator<Item = Walked>) -> Vec<Place> {
    let mut out = Vec::new();
    // The open packed slot and the channels it has used.
    let mut open: Option<(u8, u8)> = None;
    for r in registers {
        let channels = (8 - r.mask.leading_zeros() as u8).div_ceil(2);
        match r.opens {
            Some((slot, _)) if !r.sixteen => out.push(Place {
                slot,
                register: r.register,
                start: 0,
                layout: Layout::Wide,
            }),
            Some((slot, false)) => out.push(Place {
                slot,
                register: r.register,
                start: 0,
                layout: Layout::Unpacked,
            }),
            Some((slot, true)) => {
                out.push(Place {
                    slot,
                    register: r.register,
                    start: 0,
                    layout: Layout::Packed,
                });
                open = Some((slot, channels));
            }
            None if r.sixteen => {
                if let Some((slot, used)) = open
                    && used + channels <= 4
                {
                    out.push(Place {
                        slot,
                        register: r.register,
                        start: used * 2,
                        layout: Layout::Packed,
                    });
                    open = Some((slot, used + channels));
                }
            }
            None => {}
        }
    }
    out
}

/// The merged declared mask of the elements of `register` in `sig` that take part in the
/// interface (ordinary elements and clip or cull distances), and whether one is 16 bit.
pub(super) fn register_mask(sig: &[SigElement], register: u32) -> (u8, bool) {
    sig.iter()
        .filter(|e| e.register == register && matches!(e.system_value, 0 | 2 | 3))
        .fold((0, false), |(m, s), e| {
            (m | e.mask, s || matches!(e.component_type, 4..=6))
        })
}

/// Where each input register of the program lies in the attributes.
fn placements(program: &StageProgram, table: &[PixelInput]) -> Vec<Place> {
    let mut registers: std::collections::BTreeSet<u32> = table
        .iter()
        .flat_map(|a| std::iter::once(a.register).chain(a.high_register))
        .collect();
    for e in &program.inputs {
        if e.system_value == 0 && e.rw_mask != 0 && matches!(e.component_type, 4..=6) {
            registers.insert(e.register);
        }
    }
    walk(registers.into_iter().map(|register| {
        let (mask, sixteen) = register_mask(&program.inputs, register);
        Walked {
            register,
            mask,
            sixteen,
            opens: table
                .iter()
                .find(|a| a.register == register)
                .map(|a| (a.attr, a.flags & PACKED != 0)),
        }
    }))
}

/// The attributes `v_interp_mov_f32` reads, for programs without the container's table.
fn moved_attributes(program: &StageProgram) -> Vec<u8> {
    let start = crate::container::SIGNATURE_WORDS.min(program.code.len());
    decode::decode_lossy(&program.code[start..])
        .into_iter()
        .filter_map(Result::ok)
        .filter(|i| i.op == Op::v_interp_mov_f32)
        .filter_map(|i| match i.ext {
            Ext::Vintrp { attr, .. } | Ext::VintrpE64 { attr, .. } => Some(attr),
            _ => None,
        })
        .collect()
}

/// The element of the program's own signature that holds component `comp` of `register`.
fn element_at(program: &StageProgram, register: u32, comp: u8) -> Option<&SigElement> {
    let at = |e: &&SigElement| e.register == register && (e.mask >> comp) & 1 != 0;
    let ordinary = program
        .inputs
        .iter()
        .filter(at)
        .find(|e| e.system_value == 0);
    ordinary.or_else(|| {
        program.inputs.iter().filter(at).find(|e| {
            !matches!(
                e.system_value,
                SV_POSITION | SV_IS_FRONT_FACE | SV_SAMPLE_INDEX | SV_PRIMITIVE_ID
            )
        })
    })
}

/// The input struct members: the stage before's outputs (or the own inputs), then the system
/// values only the rasteriser makes.
fn members(program: &StageProgram, before: Option<&StageProgram>) -> Vec<SigElement> {
    let mut source: Vec<SigElement> = match before {
        Some(b) => super::linked_outputs(b, program)
            .into_iter()
            .cloned()
            .collect(),
        None => ordered(&program.inputs).into_iter().cloned().collect(),
    };
    if before.is_some() {
        for e in ordered(&program.inputs) {
            let generated = matches!(
                e.system_value,
                SV_IS_FRONT_FACE | SV_SAMPLE_INDEX | SV_PRIMITIVE_ID
            );
            if generated
                && e.rw_mask != 0
                && !source
                    .iter()
                    .any(|s| s.semantic == e.semantic && s.index == e.index)
            {
                source.push(e.clone());
            }
        }
    }
    source
}

fn member_field(e: &SigElement) -> Field {
    let mut f = field(e);
    // The scalar system values: the console signature may place them in a later component,
    // Direct3D wants them scalar.
    match e.system_value {
        SV_IS_FRONT_FACE | SV_SAMPLE_INDEX | SV_PRIMITIVE_ID => {
            f.comps = 1;
            f.component_type = 1;
            f.interpolation = "nointerpolation ";
        }
        _ => {}
    }
    if matches!(e.component_type, 1 | 2) && e.system_value == 0 {
        f.interpolation = "nointerpolation ";
    }
    f
}

/// The input struct member of the own element `e`. Its components are numbered as the
/// register components of the element in the stage before, which Direct3D requires to equal
/// those of the pixel side.
fn member_of(inputs: &[Field], e: &SigElement) -> Option<usize> {
    let name = field(e).semantic;
    inputs.iter().position(|f| f.semantic == name)
}

pub fn interface(
    program: &StageProgram,
    before: Option<&StageProgram>,
) -> Result<Interface, Refusal> {
    let packed_mesh_fields = match before {
        Some(mesh) if mesh.stage == crate::api::Stage::Mesh => {
            super::mesh::packed_fields(mesh, &super::linked_outputs(mesh, program))?
        }
        _ => Vec::new(),
    };
    let members = members(program, before);
    let mut inputs: Vec<Field> = members.iter().map(member_field).collect();
    for e in &program.inputs {
        if e.rw_mask != 0 && !matches!(e.system_value, 0 | SV_POSITION | SV_IS_FRONT_FACE | 2 | 3) {
            return Err(refuse(
                "pixel-input-system-value",
                format!("{}{} (code {})", e.semantic, e.index, e.system_value),
            ));
        }
    }
    // A conservative depth output needs the position at the centroid (a DXC validation rule).
    // The title's 15 programs that write one declare that mode in PSV0.
    let conservative = program
        .outputs
        .iter()
        .any(|o| matches!(o.system_value, SV_DEPTH_GREATER_EQUAL | SV_DEPTH_LESS_EQUAL));
    if conservative {
        for (f, m) in inputs.iter_mut().zip(&members) {
            if m.system_value == SV_POSITION {
                f.interpolation = "noperspective centroid ";
            }
        }
    }
    // Flat attributes give their members the modifier.
    let moved = moved_attributes(program);
    let table = attributes(program);
    for place in placements(program, &table) {
        let flat = table
            .iter()
            .any(|a| a.attr == place.slot && a.flags & FLAT != 0);
        if !flat && !moved.contains(&place.slot) {
            continue;
        }
        for e in program
            .inputs
            .iter()
            .filter(|e| e.register == place.register && e.system_value == 0)
        {
            if let Some(m) = member_of(&inputs, e) {
                inputs[m].interpolation = "nointerpolation ";
            }
        }
    }
    Ok(Interface {
        packed_mesh_fields,
        attrs: (0..inputs.len()).collect(),
        inputs,
        outputs: ordered(&program.outputs).iter().map(|e| field(e)).collect(),
        ..Interface::default()
    })
}

/// What `finish` resolves inputs with.
struct Resolver<'a> {
    program: &'a StageProgram,
    inputs: &'a [Field],
    places: Vec<Place>,
}

impl Resolver<'_> {
    /// Component `comp` of signature register `register` as a 32 bit pattern of a member.
    fn component(&self, register: u32, comp: u8) -> Result<Expr, Refusal> {
        let Some(e) = element_at(self.program, register, comp) else {
            // A padding component past the element: the console reads whatever the
            // parameter memory holds there, the translation reads zero.
            return Ok(Expr::U32(0));
        };
        let m = member_of(self.inputs, e).ok_or_else(|| {
            refuse(
                "link-missing",
                format!(
                    "{}{} is not written by the stage before",
                    e.semantic, e.index
                ),
            )
        })?;
        // The member's components are those of the register in the stage before, which may
        // place the element at another component than this program does.
        let chan =
            i16::from(comp) - i16::from(field(e).first_comp) + i16::from(self.inputs[m].first_comp);
        let first = i16::from(self.inputs[m].first_comp);
        if !(first..first + i16::from(self.inputs[m].comps)).contains(&chan) {
            return Ok(Expr::U32(0));
        }
        Ok(Expr::Input(Input::Attr {
            attr: m as u8,
            chan: chan as u8,
        }))
    }

    /// The place of attribute `attr` that holds half `half` (or channel `half / 2`).
    fn place(&self, attr: u8, half: u8) -> Result<Place, Refusal> {
        self.places
            .iter()
            .filter(|p| p.slot == attr && p.start <= half)
            .max_by_key(|p| p.start)
            .copied()
            .ok_or_else(|| {
                refuse(
                    "attr-unknown",
                    format!("attribute {attr} has no input record"),
                )
            })
    }

    /// Half `h` of a packed 16 bit attribute as a 32 bit float pattern.
    fn half(&self, attr: u8, h: u8) -> Result<Expr, Refusal> {
        let p = self.place(attr, h)?;
        if p.layout != Layout::Packed {
            return Err(refuse(
                "half-of-unpacked-attribute",
                format!("attribute {attr} half {h}"),
            ));
        }
        let comp = h - p.start;
        if comp >= 4 {
            // A padding half past the register (six programs): zero.
            return Ok(Expr::U32(0));
        }
        self.component(p.register, comp)
    }

    /// An attribute read of the lifter, see [`crate::lift::interp`].
    fn attr(&self, attr: u8, chan: u8) -> Result<Expr, Refusal> {
        if chan >= HALF {
            return self.half(attr, chan - HALF);
        }
        let p = self.place(attr, chan * 2)?;
        match p.layout {
            Layout::Wide => self.component(p.register, chan),
            Layout::Unpacked => Ok(Expr::Call(&TO_F16, vec![self.component(p.register, chan)?])),
            // A 32 bit read of a packed channel: both halves.
            Layout::Packed => {
                let lo = Expr::Call(&TO_F16, vec![self.half(attr, chan * 2)?]);
                let hi = Expr::Call(&TO_F16, vec![self.half(attr, chan * 2 + 1)?]);
                Ok(Expr::bin(
                    Bin::Or,
                    lo,
                    Expr::bin(Bin::Shl, hi, Expr::U32(16)),
                ))
            }
        }
    }

    fn system(&self, sv: u32, comp: u8) -> Result<Expr, Refusal> {
        let e = self
            .program
            .inputs
            .iter()
            .find(|e| e.system_value == sv)
            .ok_or_else(|| refuse("entry-register", format!("system value {sv} not declared")))?;
        let m = member_of(self.inputs, e).ok_or_else(|| {
            refuse(
                "link-missing",
                format!(
                    "{}{} is not written by the stage before",
                    e.semantic, e.index
                ),
            )
        })?;
        Ok(Expr::Input(Input::Attr {
            attr: m as u8,
            chan: comp,
        }))
    }

    fn rewrite(&self, e: &mut Expr) -> Result<(), Refusal> {
        match e {
            Expr::Input(Input::Attr { attr, chan }) => *e = self.attr(*attr, *chan)?,
            Expr::Input(Input::FragCoord { chan }) => {
                let c = self.system(SV_POSITION, *chan)?;
                *e = if *chan == 3 {
                    Expr::Call(&RCP, vec![c])
                } else {
                    c
                };
            }
            Expr::Input(Input::FrontFace) => {
                *e = Expr::un(Un::U32ToBool, self.system(SV_IS_FRONT_FACE, 0)?);
            }
            Expr::Un(_, a) => self.rewrite(a)?,
            Expr::Bin(_, a, b) => {
                self.rewrite(a)?;
                self.rewrite(b)?;
            }
            Expr::Select(c, a, b) => {
                self.rewrite(c)?;
                self.rewrite(a)?;
                self.rewrite(b)?;
            }
            Expr::Call(_, args) => {
                for a in args {
                    self.rewrite(a)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// The entry registers: the barycentric pair, the position components the program reads, the
/// front face. EXEC starts with the covered pixels only: the hardware starts a pixel wave
/// without the helper pixels, and `s_wqm` turns them on (the lane rules of `lift::wave`).
pub fn prologue(program: &StageProgram) -> Vec<Stmt> {
    let mut out = vec![Stmt::Set {
        dst: Var::Exec,
        value: Expr::un(Un::LNot, Expr::Input(Input::HelperLane)),
        guard: Guard::Always,
    }];
    let set = |out: &mut Vec<Stmt>, n: u8, value: Expr| {
        out.push(Stmt::Set {
            dst: Var::V(n),
            value,
            guard: Guard::Always,
        })
    };
    // The barycentric weights: the interpolation rules do not read them.
    set(&mut out, 0, Expr::U32(0));
    set(&mut out, 1, Expr::U32(0));
    let mut next = 2u8;
    let read = |sv: u32| {
        program
            .inputs
            .iter()
            .filter(|e| e.system_value == sv)
            .fold(0u8, |m, e| m | e.rw_mask)
    };
    let position = read(SV_POSITION);
    for chan in (0..4).filter(|c| (position >> c) & 1 != 0) {
        set(&mut out, next, Expr::Input(Input::FragCoord { chan }));
        next += 1;
    }
    if read(SV_IS_FRONT_FACE) != 0 {
        set(
            &mut out,
            next,
            Expr::select(
                Expr::Input(Input::FrontFace),
                Expr::U32(0xffff_ffff),
                Expr::U32(0),
            ),
        );
    }
    out
}

/// The output element (index into the ordered outputs) with system value `sv`, among `svs`.
fn output_with(program: &StageProgram, svs: &[u32]) -> Option<usize> {
    ordered(&program.outputs)
        .iter()
        .position(|o| svs.contains(&o.system_value))
}

fn export(program: &StageProgram, e: &Export, out: &mut Vec<Stmt>) -> Result<(), Refusal> {
    let outputs = ordered(&program.outputs);
    if e.valid_mask {
        out.push(Stmt::Discard {
            cond: Expr::un(Un::LNot, Expr::Var(Var::Exec)),
        });
    }
    let missing = |what: String| refuse("export-target", what).at(e.pc);
    let word = |k: usize| Expr::Var(Var::V(e.src[k]));
    match e.target {
        0..=7 => {
            // The n-th colour target the program writes: the hardware compacts the targets.
            let mut targets: Vec<(usize, &SigElement)> = outputs
                .iter()
                .enumerate()
                .filter(|(_, o)| o.system_value == SV_TARGET)
                .map(|(i, o)| (i, *o))
                .collect();
            targets.sort_by_key(|(_, o)| o.index);
            let (element, o) = *targets
                .get(usize::from(e.target))
                .ok_or_else(|| missing(format!("no colour target {} in the outputs", e.target)))?;
            let comps = (8 - o.mask.leading_zeros()) as u8;
            for chan in (0..comps).filter(|c| (e.enabled >> c) & 1 != 0) {
                let value = if e.compressed {
                    let w = word(usize::from(chan / 2));
                    let half = if chan % 2 == 1 {
                        Expr::bin(Bin::LShr, w, Expr::U32(16))
                    } else {
                        w
                    };
                    match o.component_type {
                        // 16 bit unsigned and signed integers.
                        1 => Expr::bin(Bin::And, half, Expr::U32(0xffff)),
                        2 => Expr::bin(
                            Bin::AShr,
                            Expr::bin(Bin::Shl, half, Expr::U32(16)),
                            Expr::U32(16),
                        ),
                        _ => Expr::Call(&F16, vec![half]),
                    }
                } else {
                    word(usize::from(chan))
                };
                out.push(Stmt::Output {
                    to: OutputRef {
                        element: element as u16,
                        chan,
                    },
                    value,
                });
            }
        }
        8 => {
            if e.compressed {
                return Err(refuse("export-depth-compressed", "").at(e.pc));
            }
            let parts: [(&[u32], usize); 3] = [
                (&[SV_DEPTH, SV_DEPTH_GREATER_EQUAL, SV_DEPTH_LESS_EQUAL], 0),
                (&[SV_STENCIL_REF], 1),
                (&[SV_COVERAGE], 2),
            ];
            for (svs, chan) in parts {
                if (e.enabled >> chan) & 1 == 0 {
                    continue;
                }
                let element = output_with(program, svs)
                    .ok_or_else(|| missing(format!("depth export channel {chan}")))?;
                out.push(Stmt::Output {
                    to: OutputRef {
                        element: element as u16,
                        chan: 0,
                    },
                    value: word(chan),
                });
            }
        }
        // The null target: only the valid mask matters.
        9 => {}
        t => return Err(missing(format!("pixel export to target {t}"))),
    }
    Ok(())
}

pub fn finish(
    program: &StageProgram,
    iface: &Interface,
    func: &mut Function,
) -> Result<(), Refusal> {
    let resolver = Resolver {
        program,
        inputs: &iface.inputs,
        places: placements(program, &attributes(program)),
    };
    let mut result = Ok(());
    func.for_each_block_mut(&mut |stmts| {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts.drain(..) {
            let step = match s {
                Stmt::Export(e) => export(program, &e, &mut out),
                Stmt::Set {
                    dst,
                    mut value,
                    guard,
                } => resolver.rewrite(&mut value).map(|()| {
                    out.push(Stmt::Set { dst, value, guard });
                }),
                other => {
                    out.push(other);
                    Ok(())
                }
            };
            if let Err(r) = step
                && result.is_ok()
            {
                result = Err(r);
            }
        }
        *stmts = out;
    });
    result
}
