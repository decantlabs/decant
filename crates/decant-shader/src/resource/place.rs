//! Placement: every traced resource gets a Direct3D binding, with the real root signature of
//! the pipeline in hand.
//!
//! * **A descriptor in a table is always read from the descriptor heap** at `mirror[K] + slot`,
//!   with the slot from the byte offset in the machine code. It is never declared at the register
//!   of a table range. A console table is indexed by the offset the program uses. A PC table is
//!   laid out by its ranges, in their order when the offsets are appended, and the two disagree.
//!   Measured in a retail title: the table of the user interface pipeline is `t2` then
//!   `b0`, the title writes the constant buffer of `b0` at slot 0, and a shader that declares
//!   `b0` reads slot 1 and draws nothing, while the heap read at slot 0 shows the picture. The
//!   title also pairs programs with root signatures that lack their registers altogether.
//! * A buffer behind a root descriptor is declared at the register of that parameter.
//! * A sampler built from constants is read from the sampler heap at the index the host gives.
//!
//! A table slot is 32 bytes in a resource table and 16 bytes in a sampler table.

use super::root;
use crate::api::{
    Host, Needs, Options, Placement, RegisterClass, RootParamKind, RootSignature, Stage,
    StageProgram, UserDataKind,
};
use crate::ir::{Expr, Function, Node, Origin, ResourceUse, Stmt};
use crate::refusal::{Refusal, RefusalKind};

fn refuse(what: &'static str, detail: String) -> Refusal {
    Refusal::new(RefusalKind::Resource(what), detail)
}

/// The class of a table descriptor: from the binding table of the container, else from how
/// the program uses it, else from the declared buffers when they leave no choice. As a last
/// hint a table with ONE range is asked: with a single range the PC layout and the console
/// layout cannot disagree about what kind of descriptor the table holds.
fn class_of(
    program: &StageProgram,
    root_sig: &RootSignature,
    sgpr: u8,
    dword: u8,
    slot: u32,
    offset: u32,
    usage: ResourceUse,
) -> Result<RegisterClass, Refusal> {
    if let Some(record) = program
        .binding_table
        .iter()
        .find(|r| r.sgpr == dword && r.word_offset * 4 == offset)
    {
        return Ok(record.class);
    }
    match usage {
        ResourceUse::Sampler { .. } => Ok(RegisterClass::Sampler),
        ResourceUse::Image { written: true, .. } | ResourceUse::Words { written: true } => {
            Ok(RegisterClass::Uav)
        }
        ResourceUse::Image { .. } => Ok(RegisterClass::Srv),
        ResourceUse::Words { .. } => {
            // A read only buffer is a constant buffer or a buffer view. Without a record the
            // declared resources decide, when they leave no choice.
            let buffers: Vec<RegisterClass> = program
                .resources
                .iter()
                .filter(|r| matches!(r.psv_type, 2 | 4 | 5))
                .map(|r| r.class)
                .collect();
            match buffers.first() {
                Some(first) if buffers.iter().all(|c| c == first) => Ok(*first),
                _ => match root::slot(root_sig, program.stage, sgpr).map(|s| &s.param.kind) {
                    Some(RootParamKind::Table(ranges))
                        if ranges.len() == 1
                            && slot < ranges[0].count
                            && usage_fits(ranges[0].class, usage) =>
                    {
                        Ok(ranges[0].class)
                    }
                    _ => Err(refuse(
                        "class-unknown",
                        format!("table s{sgpr} offset {offset:#x}"),
                    )),
                },
            }
        }
    }
}

/// The resource the container declares for the descriptor the binding table puts at this
/// offset of the table in mirror dword `dword`.
fn declared(
    program: &StageProgram,
    dword: u8,
    offset: u32,
) -> Option<crate::api::DeclaredResource> {
    let record = program
        .binding_table
        .iter()
        .find(|b| b.sgpr == dword && b.word_offset * 4 == offset)?;
    program
        .resources
        .iter()
        .find(|r| r.class == record.class && (r.lower..=r.upper).contains(&record.register))
        .copied()
}

fn usage_fits(class: RegisterClass, usage: ResourceUse) -> bool {
    match usage {
        ResourceUse::Sampler { .. } => class == RegisterClass::Sampler,
        ResourceUse::Image { written, .. } => {
            class
                == if written {
                    RegisterClass::Uav
                } else {
                    RegisterClass::Srv
                }
                || class == RegisterClass::Uav
        }
        ResourceUse::Words { written } => {
            class != RegisterClass::Sampler && (!written || class == RegisterClass::Uav)
        }
    }
}

fn name_of(p: &Placement) -> String {
    match p {
        Placement::Register {
            class,
            register,
            space,
        } => format!("{}{register}_sp{space}", class.letter()),
        Placement::HeapAtMirror { class, dword, slot } => {
            format!("heap_{}_s{dword}_{slot}", class.letter())
        }
        Placement::HeapSampler { index, .. } => format!("smp_{index}"),
        Placement::HeapAtConsoleIndex { class } => format!("con_{}", class.letter()),
    }
}

/// The root shader resource view the container's root layout puts at mirror dword `dword`,
/// as a register. For callers without a root signature (the census): the layout names the `t`
/// register, the declared resources its space. `None` when two declared buffers could be meant.
fn layout_root_view(program: &StageProgram, dword: u8) -> Option<Placement> {
    let register = program.user_data.iter().find_map(|u| match u.kind {
        UserDataKind::RootSrv { register } if u.sgpr == dword => Some(register),
        _ => None,
    })?;
    let in_table = |r: &crate::api::DeclaredResource| {
        program
            .binding_table
            .iter()
            .any(|b| b.class == RegisterClass::Srv && (r.lower..=r.upper).contains(&b.register))
    };
    let mut found: Vec<_> = program
        .resources
        .iter()
        .filter(|r| {
            r.class == RegisterClass::Srv
                && matches!(r.psv_type, 4 | 5)
                && (r.lower..=r.upper).contains(&register)
        })
        .collect();
    if found.len() > 1 {
        found.retain(|r| !in_table(r));
    }
    match found.as_slice() {
        [one] => Some(Placement::Register {
            class: RegisterClass::Srv,
            register,
            space: one.space,
        }),
        _ => None,
    }
}

/// The entry of the list of used mirror dwords that says the heap map is read.
const HEAP_MAP_USED: u8 = u8::MAX;
/// The same for a size query of an image by its heap slot.
const IMAGE_SIZE_USED: u8 = u8::MAX - 2;
/// The same for the view shadow.
const VIEW_SHADOW_USED: u8 = u8::MAX - 1;

/// Turns every user data read into a read of the mirror dword of that machine register.
fn rewrite_user_data(e: &mut Expr, stage: Stage, used: &mut Vec<u8>) {
    match e {
        Expr::BvhWord(op, _) => {
            for value in op.descriptor.iter_mut().chain(&mut op.address) {
                rewrite_user_data(value, stage, used);
            }
        }
        Expr::UserData(k) => {
            let dword = stage.mirror_dword(*k);
            used.push(dword);
            *e = Expr::Mirror(dword);
        }
        Expr::Un(_, a) => rewrite_user_data(a, stage, used),
        Expr::Bin(_, a, b) => {
            rewrite_user_data(a, stage, used);
            rewrite_user_data(b, stage, used);
        }
        Expr::Select(c, a, b) => {
            rewrite_user_data(c, stage, used);
            rewrite_user_data(a, stage, used);
            rewrite_user_data(b, stage, used);
        }
        Expr::Call(h, args) => {
            if std::ptr::eq(*h, &super::trace::HEAP_HOST) {
                // Not a register: the note that the program reads the heap map.
                used.push(HEAP_MAP_USED);
            }
            if h.name == "decant_img_row" {
                used.push(VIEW_SHADOW_USED);
            } else if h.name.starts_with("decant_img_") {
                used.push(IMAGE_SIZE_USED);
            }
            if std::ptr::eq(*h, &super::trace::SHADOW_STRIDE)
                || std::ptr::eq(*h, &super::trace::SHADOW_COUNT)
            {
                used.push(VIEW_SHADOW_USED);
            }
            args.iter_mut()
                .for_each(|a| rewrite_user_data(a, stage, used))
        }
        Expr::BufferWord { offset, .. } => rewrite_user_data(offset, stage, used),
        Expr::Load(l) => rewrite_user_data(&mut l.offset, stage, used),
        Expr::CodeWord(at) => rewrite_user_data(at, stage, used),
        Expr::VectorLoad { offset, .. } => rewrite_user_data(offset, stage, used),
        Expr::CodeAddress(_) => {}
        Expr::U32(_)
        | Expr::U64(_)
        | Expr::Bool(_)
        | Expr::Var(_)
        | Expr::Input(_)
        | Expr::Mirror(_) => {}
    }
}

fn rewrite_nodes(nodes: &mut [Node], stage: Stage, used: &mut Vec<u8>) {
    for n in nodes {
        match n {
            Node::Block(stmts) => {
                for s in stmts {
                    match s {
                        Stmt::Set { value, .. } | Stmt::Output { value, .. } => {
                            rewrite_user_data(value, stage, used)
                        }
                        Stmt::Discard { cond } => rewrite_user_data(cond, stage, used),
                        Stmt::Image(op, _) => op
                            .address
                            .iter_mut()
                            .for_each(|a| rewrite_user_data(a, stage, used)),
                        Stmt::Buffer(op, _) => {
                            rewrite_user_data(&mut op.offset, stage, used);
                            if let Some(i) = &mut op.index {
                                rewrite_user_data(i, stage, used);
                            }
                        }
                        Stmt::Effect { call, .. } => rewrite_user_data(call, stage, used),
                        Stmt::Export(_)
                        | Stmt::Message { .. }
                        | Stmt::Barrier { .. }
                        | Stmt::Comment(_) => {}
                    }
                }
            }
            Node::If { cond, then, els } => {
                rewrite_user_data(cond, stage, used);
                rewrite_nodes(then, stage, used);
                rewrite_nodes(els, stage, used);
            }
            Node::Loop(body) | Node::Once(body) => rewrite_nodes(body, stage, used),
            Node::Break | Node::Continue | Node::Return => {}
        }
    }
}

/// Places every resource of `func` and turns the user data registers the program still reads as
/// numbers (root constants) into mirror reads. Returns what the layer has to provide.
pub fn place(
    func: &mut Function,
    program: &StageProgram,
    root_sig: &RootSignature,
    options: &Options,
    host: &dyn Host,
) -> Result<Needs, Refusal> {
    let mut needs = Needs::default();
    for res in &mut func.resources {
        let placement = match res.origin {
            Origin::Table { sgpr, offset } => {
                let sampler = matches!(res.usage, ResourceUse::Sampler { .. });
                let stride = if sampler { 16 } else { 32 };
                if offset % stride != 0 {
                    return Err(refuse(
                        "table-offset",
                        format!("table s{sgpr} offset {offset:#x}"),
                    ));
                }
                let slot = offset / stride;
                if options.mirror.is_none() {
                    return Err(refuse(
                        "no-mirror",
                        format!("table s{sgpr} offset {offset:#x} needs the mirror"),
                    ));
                }
                // The container's tables and the mirror count the registers of every graphics
                // stage from 8, whatever machine register the stage starts at.
                let dword = program.stage.mirror_dword(sgpr);
                let class = class_of(program, root_sig, sgpr, dword, slot, offset, res.usage)?;
                res.declared = declared(program, dword, offset);
                needs.mirror_dwords.push(dword);
                if sampler {
                    needs.sampler_heap = true;
                } else {
                    needs.resource_heap = true;
                }
                Placement::HeapAtMirror { class, dword, slot }
            }
            Origin::RootDescriptor { sgpr } => {
                match root::slot(root_sig, program.stage, sgpr).map(|s| (&s.param.kind, s.dword)) {
                    Some((
                        RootParamKind::Descriptor {
                            class,
                            register,
                            space,
                        },
                        0,
                    )) => Placement::Register {
                        class: *class,
                        register: *register,
                        space: *space,
                    },
                    _ => layout_root_view(program, program.stage.mirror_dword(sgpr)).ok_or_else(
                        || {
                            refuse(
                                "root-descriptor",
                                format!("s{sgpr} is no root descriptor of the root signature"),
                            )
                        },
                    )?,
                }
            }
            Origin::HeapIndex { sampler, .. } => {
                let class = match res.usage {
                    ResourceUse::Sampler { .. } => RegisterClass::Sampler,
                    ResourceUse::Image { written: true, .. }
                    | ResourceUse::Words { written: true } => RegisterClass::Uav,
                    _ => RegisterClass::Srv,
                };
                if sampler {
                    needs.sampler_heap = true;
                } else {
                    needs.resource_heap = true;
                }
                Placement::HeapAtConsoleIndex { class }
            }
            Origin::SamplerChoices {
                ref mut choices, ..
            } => {
                for (words, index) in choices {
                    *index = host
                        .sampler_index(*words)
                        .ok_or_else(|| refuse("no-sampler-registry", String::new()))?;
                }
                needs.sampler_heap = true;
                Placement::HeapAtConsoleIndex {
                    class: RegisterClass::Sampler,
                }
            }
            Origin::Immediate { words } => {
                let index = host
                    .sampler_index(words)
                    .ok_or_else(|| refuse("no-sampler-registry", String::new()))?;
                needs.sampler_heap = true;
                Placement::HeapSampler { words, index }
            }
        };
        res.name = name_of(&placement);
        res.placement = Some(placement);
    }
    let mut used = Vec::new();
    rewrite_nodes(&mut func.body, program.stage, &mut used);
    if used.contains(&HEAP_MAP_USED) {
        used.retain(|k| *k != HEAP_MAP_USED);
        if options.mirror.and_then(|m| m.heap_map).is_none() {
            return Err(refuse(
                "no-heap-map",
                "a descriptor is taken by its console heap index".to_owned(),
            ));
        }
    }
    if used.contains(&IMAGE_SIZE_USED) {
        used.retain(|k| *k != IMAGE_SIZE_USED);
        needs.resource_heap = true;
    }
    if used.contains(&VIEW_SHADOW_USED) {
        used.retain(|k| *k != VIEW_SHADOW_USED);
        if options.mirror.and_then(|m| m.view_shadow).is_none() {
            return Err(refuse(
                "no-view-shadow",
                "the stride or record count of a buffer is read".to_owned(),
            ));
        }
    }
    if !used.is_empty() && options.mirror.is_none() {
        return Err(refuse(
            "no-mirror",
            format!("user data s{} is read as a number", used[0]),
        ));
    }
    needs.mirror_dwords.extend(used);
    needs.mirror_dwords.sort_unstable();
    needs.mirror_dwords.dedup();
    Ok(needs)
}
