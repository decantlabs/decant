//! The HLSL printer: a placed [`Function`] and its [`Interface`] to the text DXC compiles.
//!
//! Shape of the text: the helpers the program calls, the resources at their registers, the
//! mirror constant buffer when it is read, the input and output structs, and one function
//! `main`. Inside it every register the program touches is a local variable (`uint s4`,
//! `uint v2`, `bool exec`), heap resources are locals initialised from the descriptor heaps, and
//! the body follows the statements one to one. DXC turns the locals into SSA values.
//!
//! Every 32 bit value is a `uint`. A float operation reinterprets its operands with `asfloat`
//! and its result with `asuint`.

mod buffer;
mod bvh;
pub mod expr;
mod image;
pub(crate) mod simplify;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::api::{MirrorAbi, Placement, RegisterClass};
use crate::ir::{
    Desc, Expr, Function, Guard, Helper, Node, Origin, ResId, Resource, ResourceUse, Stmt, Ty, Var,
};
use crate::refusal::{Refusal, RefusalKind};
use crate::stage::{Field, Interface};

pub use expr::var_name;

/// The name of the heap map buffer in the text, see `MirrorAbi`. A rule that needs the host
/// index of a console heap index reads it through [`HEAP_MAP_LOAD`].
pub const HEAP_MAP: &str = "decant_heap_map";
/// The name of the view shadow buffer in the text, see `MirrorAbi`: the row of host heap slot
/// `n` starts at byte `n * 16`.
pub const VIEW_SHADOW: &str = "decant_view_shadow";

/// Bounds-checked 32-bit read from [`HEAP_MAP`]. Queries buffer size via `GetDimensions` and
/// returns `0u` when `[off, off + 4)` is out of bounds.
pub static HEAP_MAP_LOAD: Helper = Helper {
    name: "decant_heap_map_load",
    result: Ty::U32,
    text: "uint decant_heap_map_load(uint off) { uint dim; decant_heap_map.GetDimensions(dim); return off <= dim && dim - off >= 4u ? decant_heap_map.Load(off) : 0u; }",
    needs: &[],
};

/// Bounds-checked 32-bit read from [`VIEW_SHADOW`]. Queries buffer size via `GetDimensions` and
/// returns `0u` when `[off, off + 4)` is out of bounds.
pub static VIEW_SHADOW_LOAD: Helper = Helper {
    name: "decant_view_shadow_load",
    result: Ty::U32,
    text: "uint decant_view_shadow_load(uint off) { uint dim; decant_view_shadow.GetDimensions(dim); return off <= dim && dim - off >= 4u ? decant_view_shadow.Load(off) : 0u; }",
    needs: &[],
};

/// Bounds-checked 64-bit read from [`VIEW_SHADOW`]. Queries buffer size via `GetDimensions` and
/// returns `uint2(0u, 0u)` when `[off, off + 8)` is out of bounds.
pub static VIEW_SHADOW_LOAD2: Helper = Helper {
    name: "decant_view_shadow_load2",
    result: Ty::U32,
    text: "uint2 decant_view_shadow_load2(uint off) { uint dim; decant_view_shadow.GetDimensions(dim); return off <= dim && dim - off >= 8u ? decant_view_shadow.Load2(off) : uint2(0u, 0u); }",
    needs: &[],
};

/// Bounds-checked 128-bit read from [`VIEW_SHADOW`]. Queries buffer size via `GetDimensions` and
/// returns `uint4(0u, 0u, 0u, 0u)` when `[off, off + 16)` is out of bounds.
pub static VIEW_SHADOW_LOAD4: Helper = Helper {
    name: "decant_view_shadow_load4",
    result: Ty::U32,
    text: "uint4 decant_view_shadow_load4(uint off) { uint dim; decant_view_shadow.GetDimensions(dim); return off <= dim && dim - off >= 16u ? decant_view_shadow.Load4(off) : uint4(0u, 0u, 0u, 0u); }",
    needs: &[],
};

/// The entry point name.
pub const ENTRY: &str = "main";

/// What the printer needs besides the function.
pub struct Context<'a> {
    pub iface: &'a Interface,
    pub mirror: Option<MirrorAbi>,
    /// Lines put at the top as comments.
    pub header: &'a [String],
    /// The code words of the program with its signature, for reads of constant tables the
    /// compiler placed behind the code (`Expr::CodeWord`).
    pub code: &'a [u32],
}

fn refuse(what: &'static str, pc: u32) -> Refusal {
    Refusal::new(RefusalKind::Form(crate::tables::Op::image_sample, what), "").at(pc)
}

fn scalar_type(component_type: u32) -> &'static str {
    match component_type {
        1 => "uint",
        2 => "int",
        _ => "float",
    }
}

fn field_type(f: &Field) -> String {
    let base = scalar_type(f.component_type);
    if f.comps <= 1 {
        base.to_owned()
    } else {
        format!("{base}{}", f.comps)
    }
}

/// The HLSL type of a resource.
fn resource_type(func: &Function, r: &Resource, class: RegisterClass) -> Option<String> {
    Some(match r.usage {
        ResourceUse::Words { .. } => buffer::resource_type(func, r, class)?,
        ResourceUse::Image { .. } => image::resource_type(r, class)?,
        ResourceUse::Sampler { compare } => if compare {
            "SamplerComparisonState"
        } else {
            "SamplerState"
        }
        .to_owned(),
    })
}

fn class_of(p: &Placement) -> RegisterClass {
    match p {
        Placement::Register { class, .. }
        | Placement::HeapAtMirror { class, .. }
        | Placement::HeapAtConsoleIndex { class } => *class,
        Placement::HeapSampler { .. } => RegisterClass::Sampler,
    }
}

fn mirror_word(k: u8) -> String {
    format!(
        "decant_mirror[{}].{}",
        k >> 2,
        ["x", "y", "z", "w"][usize::from(k & 3)]
    )
}

/// Wraps a heap index that is read at run time (from a register or the mirror), so the shader
/// compiler does not assume that every lane of a wave picks the same descriptor. A compile-time
/// constant index (a literal slot number) needs no wrap.
fn nonuniform_index(index: &str) -> String {
    format!("NonUniformResourceIndex({index})")
}

struct Printer<'a> {
    cx: &'a Context<'a>,
    func: &'a Function,
    out: String,
    depth: usize,
    unique: u32,
    /// Registers declared where the program first sets them, either at the top level of the
    /// function or as single-assignment temporaries whose uses stay inside one block.
    inline: BTreeSet<Var>,
    /// Hoisted `decant_view_shadow_load4` row locals for typed buffer resources with repeated uses.
    typed_rows: BTreeMap<ResId, String>,
    /// Whether single-component non-d16 typed loads use the scalar `decant_typed_decode_x` helper.
    typed_decode_x: bool,
    /// The node intersection of the statement before and the local that holds its result
    /// words, see [`bvh::assign`].
    bvh: Option<(crate::ir::BvhIntersect, String)>,
}

impl Printer<'_> {
    fn line(&mut self, text: &str) {
        self.indent();
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// The start of a line at the current depth.
    fn indent(&mut self) {
        for _ in 0..self.depth {
            self.out.push_str("    ");
        }
    }

    /// Appends the text of `e` to the output.
    fn push_e(&mut self, e: &Expr) {
        expr::write(e, self.cx.iface, &self.func.resources, &mut self.out);
    }

    /// The line `dst = value;`, with the type when it declares `dst`, or `value;` without a
    /// destination, written straight into the output. Most lines of a program are this, and a
    /// string per line and per operand made the printer a large allocator of the translation.
    fn assign(&mut self, guard: Guard, ty: &str, dst: Option<Var>, value: &Expr) {
        self.indent();
        if matches!(guard, Guard::Exec) {
            self.out.push_str("if (exec) ");
        }
        self.out.push_str(ty);
        if let Some(dst) = dst {
            expr::push_var(dst, &mut self.out);
            self.out.push_str(" = ");
        }
        self.push_e(value);
        self.out.push_str(";\n");
    }

    fn e(&self, e: &Expr) -> String {
        expr::print(e, self.cx.iface, &self.func.resources)
    }

    fn res(&self, d: Desc) -> &Resource {
        match d {
            Desc::Res(ResId(i)) => &self.func.resources[i as usize],
            Desc::Regs(_) => unreachable!("the resource pass replaced every register descriptor"),
        }
    }

    /// The name to use resource `id` by at this statement. A descriptor the program chose by
    /// its heap index at run time is made here, from the index it holds now: the same resource
    /// can be a different descriptor at each statement.
    fn use_resource(&mut self, id: ResId, ty: &str) -> String {
        let r = &self.func.resources[id.0 as usize];
        match (r.placement, &r.origin) {
            (Some(Placement::HeapAtConsoleIndex { .. }), Origin::HeapIndex { index, sampler }) => {
                self.unique += 1;
                let local = format!("{}_{}", r.name, self.unique);
                let heap = if *sampler {
                    "SamplerDescriptorHeap"
                } else {
                    "ResourceDescriptorHeap"
                };
                let idx = nonuniform_index(&var_name(*index));
                self.line(&format!("{ty} {local} = {heap}[{idx}];"));
                local
            }
            _ => r.name.clone(),
        }
    }

    /// Whether resource `id` is taken by a heap index in a register and so made at each use.
    fn by_index(&self, id: ResId) -> bool {
        matches!(
            self.func.resources[id.0 as usize].placement,
            Some(Placement::HeapAtConsoleIndex { .. })
        )
    }

    /// The host heap index of resource `id` as text: the mirror dword of its table plus its
    /// slot, or the register the program chose it by. `None` for a view without a heap slot.
    fn host_slot(&self, id: ResId) -> Option<String> {
        let r = &self.func.resources[id.0 as usize];
        match (r.placement, &r.origin) {
            (Some(Placement::HeapAtMirror { dword, slot, .. }), _) => {
                Some(format!("({} + {slot}u)", mirror_word(dword)))
            }
            (Some(Placement::HeapAtConsoleIndex { .. }), Origin::HeapIndex { index, .. }) => {
                Some(var_name(*index))
            }
            _ => None,
        }
    }

    fn guarded(&mut self, guard: Guard, lines: Vec<String>) {
        match (guard, lines.len()) {
            (Guard::Always, _) => lines.iter().for_each(|l| self.line(l)),
            (Guard::Exec, 1) => self.line(&format!("if (exec) {}", lines[0])),
            (Guard::Exec, _) => {
                self.line("if (exec) {");
                self.depth += 1;
                lines.iter().for_each(|l| self.line(l));
                self.depth -= 1;
                self.line("}");
            }
        }
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), Refusal> {
        if !matches!(
            s,
            Stmt::Set {
                value: Expr::BvhWord(..),
                ..
            }
        ) {
            self.bvh = None;
        }
        match s {
            Stmt::Set { dst, value, guard } => {
                let ty = if self.inline.remove(dst) {
                    if dst.ty() == Ty::Bool {
                        "bool "
                    } else {
                        "uint "
                    }
                } else {
                    ""
                };
                if let Expr::BvhWord(op, word) = value {
                    return bvh::assign(self, *guard, ty, *dst, op, *word);
                }
                if !matches!(value, Expr::BufferWord { res, .. } if self.by_index(*res)) {
                    self.assign(*guard, ty, Some(*dst), value);
                    return Ok(());
                }
                let text = match value {
                    // A scalar read through a descriptor the program chose by its heap index:
                    // the buffer is made in a line before the assignment.
                    Expr::BufferWord { res, offset } if self.by_index(*res) => {
                        let func = self.func;
                        let r = &func.resources[res.0 as usize];
                        let class = r.placement.as_ref().map(class_of);
                        let rty =
                            class
                                .and_then(|c| resource_type(func, r, c))
                                .ok_or_else(|| {
                                    Refusal::new(
                                        RefusalKind::Resource("class-mismatch"),
                                        r.name.clone(),
                                    )
                                })?;
                        let local = self.use_resource(*res, &rty);
                        let at = self.e(offset);
                        format!(
                            "{ty}{} = {};",
                            var_name(*dst),
                            buffer::word(&local, class, &at)
                        )
                    }
                    _ => unreachable!("assigned above"),
                };
                self.guarded(*guard, vec![text]);
            }
            Stmt::Image(op, guard) => self.image(op, *guard)?,
            Stmt::Buffer(op, guard) => self.buffer(op, *guard)?,
            Stmt::Export(e) => {
                return Err(Refusal::new(
                    RefusalKind::Stage("export-target"),
                    "an export the stage did not resolve",
                )
                .at(e.pc));
            }
            Stmt::Output { to, value } => {
                let f = &self.cx.iface.outputs[usize::from(to.element)];
                // `chan` is a component of the register, the member starts at the first
                // component of its element. A write outside the element has no member.
                if let Some(rel) = to.chan.checked_sub(f.first_comp).filter(|r| *r < f.comps) {
                    let member = if f.comps <= 1 {
                        String::new()
                    } else {
                        format!(".{}", ["x", "y", "z", "w"][usize::from(rel & 3)])
                    };
                    let value = match f.component_type {
                        1 => self.e(value),
                        2 => format!("int({})", self.e(value)),
                        _ => format!("asfloat({})", self.e(value)),
                    };
                    self.line(&format!("decant_out.o{}{member} = {value};", to.element));
                }
            }
            Stmt::Discard { cond } => self.line(&format!("if ({}) discard;", self.e(cond))),
            Stmt::Effect { dst, call, guard } => self.assign(*guard, "", *dst, call),
            Stmt::Message { pc, .. } => {
                return Err(Refusal::new(RefusalKind::Stage("message-unresolved"), "").at(*pc));
            }
            Stmt::Barrier { pc } => {
                return Err(Refusal::new(RefusalKind::Stage("barrier-unresolved"), "").at(*pc));
            }
            Stmt::Comment(text) => self.line(&format!("// {text}")),
        }
        Ok(())
    }

    fn nodes(&mut self, nodes: &[Node]) -> Result<(), Refusal> {
        for n in nodes {
            self.bvh = None;
            match n {
                Node::Block(stmts) => {
                    for s in stmts {
                        self.stmt(s)?;
                    }
                }
                Node::If { cond, then, els } => {
                    self.indent();
                    self.out.push_str("if (");
                    self.push_e(cond);
                    self.out.push_str(") {\n");
                    self.depth += 1;
                    self.nodes(then)?;
                    self.depth -= 1;
                    if !els.is_empty() {
                        self.line("} else {");
                        self.depth += 1;
                        self.nodes(els)?;
                        self.depth -= 1;
                    }
                    self.line("}");
                }
                Node::Loop(body) => {
                    self.line("while (true) {");
                    self.depth += 1;
                    self.nodes(body)?;
                    self.depth -= 1;
                    self.line("}");
                }
                Node::Once(body) => {
                    self.line("do {");
                    self.depth += 1;
                    self.nodes(body)?;
                    self.depth -= 1;
                    self.line("} while (false);");
                }
                Node::Break => self.line("break;"),
                Node::Continue => self.line("continue;"),
                Node::Return => self.line(if self.cx.iface.outputs.is_empty() {
                    "return;"
                } else {
                    "return decant_out;"
                }),
            }
        }
        Ok(())
    }
}

fn collect_helpers(func: &Function) -> Vec<&'static Helper> {
    fn add(h: &'static Helper, out: &mut Vec<&'static Helper>) {
        if out.iter().any(|o| std::ptr::eq(*o, h)) {
            return;
        }
        h.needs.iter().for_each(|n| add(n, out));
        out.push(h);
    }
    fn walk(e: &Expr, out: &mut Vec<&'static Helper>) {
        match e {
            Expr::Call(h, args) => {
                add(h, out);
                args.iter().for_each(|a| walk(a, out));
            }
            Expr::Un(_, a) => walk(a, out),
            Expr::Bin(_, a, b) => {
                walk(a, out);
                walk(b, out);
            }
            Expr::Select(c, a, b) => {
                walk(c, out);
                walk(a, out);
                walk(b, out);
            }
            Expr::BufferWord { offset, .. } => walk(offset, out),
            Expr::CodeWord(at) => walk(at, out),
            Expr::BvhWord(op, _) => op.address.iter().for_each(|a| walk(a, out)),
            _ => {}
        }
    }
    fn nodes(ns: &[Node], out: &mut Vec<&'static Helper>) {
        for n in ns {
            match n {
                Node::Block(stmts) => {
                    for s in stmts {
                        match s {
                            Stmt::Set { value, .. } | Stmt::Output { value, .. } => {
                                walk(value, out)
                            }
                            Stmt::Discard { cond } => walk(cond, out),
                            Stmt::Effect { call, .. } => walk(call, out),
                            Stmt::Image(op, _) => op.address.iter().for_each(|a| walk(a, out)),
                            Stmt::Buffer(op, _) => {
                                walk(&op.offset, out);
                                if let Some(i) = &op.index {
                                    walk(i, out);
                                }
                            }
                            Stmt::Export(_)
                            | Stmt::Message { .. }
                            | Stmt::Barrier { .. }
                            | Stmt::Comment(_) => {}
                        }
                    }
                }
                Node::If { cond, then, els } => {
                    walk(cond, out);
                    nodes(then, out);
                    nodes(els, out);
                }
                Node::Loop(body) | Node::Once(body) => nodes(body, out),
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
    }
    let mut out = Vec::new();
    nodes(&func.body, &mut out);
    if bvh::used(&func.body) {
        add(&bvh::INTERSECT, &mut out);
    }
    for h in buffer::shadow_helpers(func) {
        add(h, &mut out);
    }
    out
}

fn is_temp_var(v: Var) -> bool {
    matches!(v, Var::Tmp(_) | Var::Flag(_))
}

/// The first word index in `code` after all reachable instructions when the program has a
/// signature, ends with `s_endpgm`, and has trailing data words behind it. Returns `0` when the
/// whole `code` slice must be kept (for example, synthetic code without a signature, no trailing
/// words, or a read that addresses words before the end of code).
fn code_table_start(code: &[u32], func: &Function) -> usize {
    if !crate::container::has_signature(code) {
        return 0;
    }
    let Ok(graph) = crate::cfg::build(code) else {
        return 0;
    };
    if !graph.blocks.iter().any(|b| b.term == crate::cfg::Term::End) {
        return 0;
    }
    let starts: BTreeSet<u32> = graph.blocks.iter().map(|b| b.start).collect();
    let mut end_word = 0u32;
    for block in &graph.blocks {
        let after_insns = block
            .insns
            .last()
            .map_or(block.start, |i| i.pc + u32::from(i.len));
        let block_end = match block.term {
            // `s_endpgm` is a 1-word SOPP instruction immediately after `block.insns`.
            crate::cfg::Term::End => after_insns + 1,
            crate::cfg::Term::Jump(_) | crate::cfg::Term::Branch { .. } => {
                if starts.contains(&after_insns) {
                    after_insns
                } else if let Ok(insn) = crate::decode::decode_one(code, after_insns) {
                    if insn.op == crate::tables::Op::s_getpc_b64 {
                        // A far jump folds four instructions out of `block.insns`.
                        let mut pc = after_insns + u32::from(insn.len);
                        for _ in 0..3 {
                            if let Ok(next) = crate::decode::decode_one(code, pc) {
                                pc += u32::from(next.len);
                            }
                        }
                        pc
                    } else {
                        after_insns + u32::from(insn.len)
                    }
                } else {
                    after_insns
                }
            }
        };
        end_word = end_word.max(block_end);
    }
    let mut base = end_word as usize;
    if base == 0 || base >= code.len() {
        return 0;
    }
    // If any runtime `CodeWord` explicitly adds a constant byte address below `base * 4`, lower
    // `base` so the slice still covers it.
    expr::for_each_expr(func, &mut |e| {
        if let Expr::CodeWord(at) = e {
            let const_byte = match &**at {
                Expr::Bin(crate::ir::Bin::Add, a, b) => match (&**a, &**b) {
                    (Expr::U32(c), _) | (_, Expr::U32(c)) => Some(*c),
                    _ => None,
                },
                Expr::U32(c) => Some(*c),
                _ => None,
            };
            if let Some(byte) = const_byte {
                let word = (byte >> 2) as usize;
                if word < base {
                    base = word;
                }
            }
        }
    });
    base
}

/// Which registers to declare. `hoisted` are declared with a zero at the top of the function.
/// `inline` are declared by their first statement: either an unconditional assignment at the top
/// level of the function that does not read the register itself, or a single-assignment `Var::Tmp`
/// / `Var::Flag` whose definition and every read stay within the same `Node::Block`. EXEC is left
/// out when it is a constant.
fn declarations(func: &Function, exec_const: bool) -> (BTreeSet<Var>, BTreeSet<Var>) {
    let mut seen = BTreeSet::new();
    let mut hoisted = BTreeSet::new();
    let mut inline = BTreeSet::new();
    // For each `Var::Tmp` / `Var::Flag`: `Some(block_id)` means it has one unconditional `Set`
    // in `block_id` and every read so far happened later in the same block; `None` means it must
    // stay hoisted (read before definition, cross-block use, guarded/effect write, or multiple
    // writes).
    let mut block_local: BTreeMap<Var, Option<usize>> = BTreeMap::new();
    let mut next_block = 0usize;
    let mut touch = |v: Var, seen: &mut BTreeSet<Var>, hoisted: &mut BTreeSet<Var>| {
        if !(exec_const && v == Var::Exec) && seen.insert(v) {
            hoisted.insert(v);
        }
    };
    fn note_read(v: Var, block: Option<usize>, block_local: &mut BTreeMap<Var, Option<usize>>) {
        if !is_temp_var(v) {
            return;
        }
        match (block, block_local.get(&v)) {
            (Some(bid), Some(Some(def_bid))) if *def_bid == bid => {}
            _ => {
                block_local.insert(v, None);
            }
        }
    }
    fn note_non_inline_write(v: Var, block_local: &mut BTreeMap<Var, Option<usize>>) {
        if is_temp_var(v) {
            block_local.insert(v, None);
        }
    }
    fn touch_stmt(
        s: &Stmt,
        resources: &[Resource],
        bid: usize,
        touch: &mut impl FnMut(Var, &mut BTreeSet<Var>, &mut BTreeSet<Var>),
        seen: &mut BTreeSet<Var>,
        hoisted: &mut BTreeSet<Var>,
        block_local: &mut BTreeMap<Var, Option<usize>>,
    ) {
        // Straight to `touch`, without a list per statement: the sets do not depend on the
        // order within one statement.
        crate::resource::stmt_reads(s, &mut |v| {
            touch(v, seen, hoisted);
            note_read(v, Some(bid), block_local);
        });
        crate::resource::stmt_index_reads(s, resources, &mut |v| {
            note_read(v, Some(bid), block_local);
        });
        match s {
            Stmt::Set { dst, .. } | Stmt::Effect { dst: Some(dst), .. } => {
                touch(*dst, seen, hoisted);
                note_non_inline_write(*dst, block_local);
            }
            Stmt::Image(op, _) => image::written(op).for_each(|v| touch(v, seen, hoisted)),
            Stmt::Buffer(op, _) => {
                (0..op.count).for_each(|k| touch(Var::V(op.data + k), seen, hoisted))
            }
            _ => {}
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn walk(
        ns: &[Node],
        top: bool,
        resources: &[Resource],
        next_block: &mut usize,
        touch: &mut impl FnMut(Var, &mut BTreeSet<Var>, &mut BTreeSet<Var>),
        seen: &mut BTreeSet<Var>,
        hoisted: &mut BTreeSet<Var>,
        inline: &mut BTreeSet<Var>,
        block_local: &mut BTreeMap<Var, Option<usize>>,
    ) {
        for n in ns {
            match n {
                Node::Block(stmts) => {
                    let bid = *next_block;
                    *next_block += 1;
                    for s in stmts {
                        if let Stmt::Set {
                            dst,
                            value,
                            guard: Guard::Always,
                        } = s
                        {
                            crate::resource::reads(value, &mut |v| {
                                touch(v, seen, hoisted);
                                note_read(v, Some(bid), block_local);
                            });
                            crate::resource::stmt_index_reads(s, resources, &mut |v| {
                                note_read(v, Some(bid), block_local);
                            });
                            if top {
                                if seen.insert(*dst) {
                                    inline.insert(*dst);
                                }
                            } else {
                                touch(*dst, seen, hoisted);
                            }
                            if is_temp_var(*dst) {
                                match block_local.get(dst) {
                                    None => {
                                        block_local.insert(*dst, Some(bid));
                                    }
                                    Some(_) => {
                                        block_local.insert(*dst, None);
                                    }
                                }
                            }
                        } else {
                            touch_stmt(s, resources, bid, touch, seen, hoisted, block_local);
                        }
                    }
                }
                Node::If { cond, then, els } => {
                    crate::resource::reads(cond, &mut |v| {
                        touch(v, seen, hoisted);
                        note_read(v, None, block_local);
                    });
                    walk(
                        then,
                        false,
                        resources,
                        next_block,
                        touch,
                        seen,
                        hoisted,
                        inline,
                        block_local,
                    );
                    walk(
                        els,
                        false,
                        resources,
                        next_block,
                        touch,
                        seen,
                        hoisted,
                        inline,
                        block_local,
                    );
                }
                Node::Loop(body) | Node::Once(body) => walk(
                    body,
                    false,
                    resources,
                    next_block,
                    touch,
                    seen,
                    hoisted,
                    inline,
                    block_local,
                ),
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
    }
    walk(
        &func.body,
        true,
        &func.resources,
        &mut next_block,
        &mut touch,
        &mut seen,
        &mut hoisted,
        &mut inline,
        &mut block_local,
    );
    for (v, state) in block_local {
        if state.is_some() {
            hoisted.remove(&v);
            inline.insert(v);
        }
    }
    (hoisted, inline)
}

/// Prints `func`. Every resource must be placed.
pub fn print(func: &Function, cx: &Context) -> Result<String, Refusal> {
    let mut simplified = func.clone();
    let exec_const = simplify::run(&mut simplified);
    // After the constants are folded: a code read at a constant address is its word.
    let code_array = simplify::fold_code(&mut simplified, cx.code);
    // The raw view of a heap slot may start before the title's first byte.
    buffer::add_view_offsets(&mut simplified, cx);
    let func = &simplified;
    bvh::check(func)?;
    let mut out = String::new();
    for l in cx.header {
        writeln!(out, "// {l}").expect("writing to a string");
    }
    // The two buffers of the mirror ABI come first, so that helpers can read them. DXC drops
    // the one a program does not use.
    if let Some(m) = cx.mirror {
        for (name, register) in [(HEAP_MAP, m.heap_map), (VIEW_SHADOW, m.view_shadow)] {
            if let Some(r) = register {
                writeln!(
                    out,
                    "ByteAddressBuffer {name} : register(t{r}, space{});",
                    m.space
                )
                .expect("writing to a string");
            }
        }
    }
    if code_array {
        // Slice off the executable instructions up to `s_endpgm` when the trailing constant table
        // sits after the code, so DXC does not lower the whole instruction stream to a stack array.
        let base = code_table_start(cx.code, func);
        let slice = &cx.code[base..];
        let words: Vec<String> = slice.iter().map(|w| format!("{w:#x}")).collect();
        let count = slice.len().max(1);
        let end = cx.code.len().max(1);
        writeln!(
            out,
            "static const uint {0}[{count}] = {{ {1} }};",
            expr::CODE,
            if words.is_empty() {
                "0".to_owned()
            } else {
                words.join(", ")
            }
        )
        .expect("writing to a string");
        // A lane that is not active computes the address too, from whatever its registers hold:
        // an address outside the table reads zero and never indexes past the array.
        if base == 0 {
            writeln!(
                out,
                "uint {0}_at(uint a) {{ uint i = a >> 2; return i < {count}u ? {0}[i] : 0u; }}",
                expr::CODE
            )
            .expect("writing to a string");
        } else {
            writeln!(
                out,
                "uint {0}_at(uint a) {{ uint i = a >> 2; return i >= {base}u && i < {end}u ? {0}[i - {base}u] : 0u; }}",
                expr::CODE
            )
            .expect("writing to a string");
        }
    }
    let helpers = collect_helpers(func);
    let uses_wave_lanes = helpers.iter().any(|h| {
        h.name == crate::lift::wave::READ_LANE.name || h.name == crate::lift::wave::WRITE_LANE.name
    });
    let uses_wave_swizzle = helpers
        .iter()
        .any(|h| h.name == crate::lift::wave::SWIZZLE_WAVE.name);
    let swizzle_threads = if helpers
        .iter()
        .any(|h| h.name == crate::lift::wave::SWIZZLE.name)
    {
        let threads = cx
            .iface
            .threads
            .and_then(|dims| dims.into_iter().try_fold(1u32, u32::checked_mul));
        let Some(threads @ 1..=1024) = threads else {
            return Err(Refusal::new(
                RefusalKind::Exec("cross-lane"),
                "invalid swizzle group size",
            ));
        };
        writeln!(out, "static const uint decant_swizzle_threads = {threads}u;")
            .expect("writing to a string");
        threads
    } else {
        0
    };
    let shared_words = func.groupshared_dwords + swizzle_threads;
    if swizzle_threads != 0 && shared_words > cx.iface.groupshared_limit {
        return Err(Refusal::new(
            RefusalKind::Exec("cross-lane"),
            "swizzle scratch exceeds group shared memory limit",
        ));
    }
    if shared_words != 0 {
        writeln!(
            out,
            "groupshared uint decant_lds[{shared_words}];\nstatic const uint decant_lds_words = {};",
            func.groupshared_dwords
        )
        .expect("writing to a string");
    }
    let dpp = helpers.iter().any(|h| {
        h.name == crate::lift::wave::DPP_ADD.name || h.name == crate::lift::wave::PERMLANE.name
            || h.name == crate::lift::wave::BPERMUTE.name
            || h.name == crate::lift::wave::DPP8_MOV.name
    });
    let mut uses_dpp_mov = false;
    let mut dpp_mov_needs_full_wave = false;
    expr::for_each_expr(func, &mut |e| {
        if let Expr::Call(h, args) = e
            && h.name == crate::lift::wave::DPP_MOV.name
        {
            uses_dpp_mov = true;
            if let Some(Expr::U32(rows)) = args.get(6) {
                if (rows & 3) != ((rows >> 2) & 3) {
                    dpp_mov_needs_full_wave = true;
                }
            } else {
                dpp_mov_needs_full_wave = true;
            }
        }
    });
    for h in helpers {
        out.push_str(h.text);
        out.push('\n');
    }
    buffer::preamble(func, cx, &mut out)?;
    // A buffer word read through a heap index in a register is printed where it is the whole
    // value of an assignment (what a scalar load lifts to). Anywhere else nothing declares the
    // buffer before the expression.
    let by_index = |e: &Expr| {
        matches!(e, Expr::BufferWord { res, .. } if matches!(
            func.resources[res.0 as usize].placement,
            Some(Placement::HeapAtConsoleIndex { .. })
        ))
    };
    let (mut all, mut whole) = (0u32, 0u32);
    expr::for_each_expr(func, &mut |e| all += u32::from(by_index(e)));
    func.for_each_stmt(&mut |s| {
        if let Stmt::Set { value, .. } = s {
            whole += u32::from(by_index(value) || bvh::by_index(func, value));
        }
    });
    if all != whole {
        return Err(Refusal::new(
            RefusalKind::Resource("dynamic-descriptor-print"),
            "a buffer word inside an expression",
        ));
    }
    let mut heap_locals = Vec::new();
    let mut uses_mirror = false;
    expr::any_mirror(func, &mut uses_mirror);
    for r in &func.resources {
        let placement = r
            .placement
            .as_ref()
            .ok_or_else(|| Refusal::new(RefusalKind::Resource("unplaced"), ""))?;
        let ty = resource_type(func, r, class_of(placement))
            .ok_or_else(|| Refusal::new(RefusalKind::Resource("class-mismatch"), r.name.clone()))?;
        match placement {
            Placement::Register {
                class,
                register,
                space,
            } => writeln!(
                out,
                "{ty} {} : register({}{register}, space{space});",
                r.name,
                class.letter()
            )
            .expect("writing to a string"),
            Placement::HeapAtMirror { class, dword, slot } => {
                uses_mirror = true;
                let heap = if *class == RegisterClass::Sampler {
                    "SamplerDescriptorHeap"
                } else {
                    "ResourceDescriptorHeap"
                };
                let idx = nonuniform_index(&format!("{} + {slot}u", mirror_word(*dword)));
                heap_locals.push(format!("{ty} {} = {heap}[{idx}];", r.name));
            }
            // A resource whose heap index is in a register is made at each use
            // (`Printer::use_resource`): by the image and buffer statements, and by an
            // assignment that reads a buffer word.
            Placement::HeapAtConsoleIndex { .. } => {}
            Placement::HeapSampler { index, .. } => heap_locals.push(format!(
                "{ty} {} = SamplerDescriptorHeap[{index}u];",
                r.name
            )),
        }
    }
    if uses_mirror {
        let m = cx
            .mirror
            .ok_or_else(|| Refusal::new(RefusalKind::Resource("no-mirror"), ""))?;
        writeln!(
            out,
            "cbuffer decant_mirror_cb : register(b{}, space{}) {{ uint4 decant_mirror[8]; }};",
            m.register, m.space
        )
        .expect("writing to a string");
    }
    let uses = expr::input_uses(func);
    let structs = |name: &str, prefix: &str, fields: &[Field], extra: &[&str], out: &mut String| {
        writeln!(out, "struct {name} {{").expect("writing to a string");
        for (i, f) in fields.iter().enumerate() {
            writeln!(
                out,
                "    {}{} {prefix}{i} : {};",
                f.interpolation,
                field_type(f),
                f.semantic
            )
            .expect("writing to a string");
        }
        for e in extra {
            writeln!(out, "    {e};").expect("writing to a string");
        }
        out.push_str("};\n");
    };
    let packed = &cx.iface.packed_mesh_fields;
    let transport = |fields: &[Field]| -> Vec<Field> {
        let mut fields = fields.to_vec();
        for &i in packed {
            fields[i].component_type = 1;
            fields[i].comps = fields[i].comps.div_ceil(2);
            fields[i].interpolation = "nointerpolation ";
        }
        fields
    };
    let packed_input = !packed.is_empty() && !cx.iface.inputs.is_empty();
    let needs_barycentrics = packed_input
        && packed
            .iter()
            .any(|&i| cx.iface.inputs[i].interpolation != "nointerpolation ");
    let mut extra = Vec::new();
    if uses.vertex_id {
        extra.push("uint decant_vertex_id : SV_VertexID");
    }
    if uses.instance_id {
        extra.push("uint decant_instance_id : SV_InstanceID");
    }
    let has_in = !cx.iface.inputs.is_empty() || !extra.is_empty();
    if has_in {
        structs("DecantIn", "i", &cx.iface.inputs, &extra, &mut out);
        if packed_input {
            if needs_barycentrics {
                extra.push("float3 decant_bary : SV_Barycentrics");
            }
            structs(
                "DecantMeshIn",
                "i",
                &transport(&cx.iface.inputs),
                &extra,
                &mut out,
            );
        }
    }
    if !cx.iface.outputs.is_empty() {
        structs("DecantOut", "o", &cx.iface.outputs, &[], &mut out);
        if !packed.is_empty() && !packed_input {
            structs(
                "DecantMeshOut",
                "o",
                &transport(&cx.iface.outputs),
                &[],
                &mut out,
            );
        }
    }
    let ret = if cx.iface.outputs.is_empty() {
        "void"
    } else {
        "DecantOut"
    };
    // A compute entry takes the thread id in the group and the group id as parameters.
    let params = match cx.iface.threads {
        Some([x, y, z]) => {
            if uses_wave_lanes || dpp || dpp_mov_needs_full_wave {
                writeln!(out, "[WaveSize({})]", func.lanes).expect("writing to a string");
            } else if (uses_wave_swizzle || uses_dpp_mov) && cx.iface.outputs.is_empty() {
                writeln!(out, "[WaveSize(32)]").expect("writing to a string");
            }
            writeln!(
                out,
                "[numthreads({}, {}, {})]",
                x.max(1),
                y.max(1),
                z.max(1)
            )
            .expect("writing to a string");
            "uint3 decant_tid : SV_GroupThreadID, uint3 decant_gid : SV_GroupID, uint decant_gi : SV_GroupIndex"
        }
        None if packed_input => "DecantMeshIn decant_mesh_in",
        None if has_in => "DecantIn decant_in",
        None => "",
    };
    writeln!(out, "{ret} {ENTRY}({params}) {{").expect("writing to a string");
    let (hoisted, inline) = declarations(func, exec_const);
    let (typed_rows, typed_row_lines) = buffer::hoisted_typed_rows(func, cx);
    let typed_decode_x = buffer::uses_typed_decode_x(func);
    let mut p = Printer {
        cx,
        func,
        out,
        depth: 1,
        unique: 0,
        inline,
        typed_rows,
        typed_decode_x,
        bvh: None,
    };
    if !cx.iface.outputs.is_empty() {
        p.line("DecantOut decant_out = (DecantOut)0;");
    }
    if packed_input {
        // Keep the logical input used by the lifted program. For smooth fields, fetch each
        // vertex's original bits before interpolation because each word contains two values.
        p.line("DecantIn decant_in;");
        for (i, field) in cx.iface.inputs.iter().enumerate() {
            if !packed.contains(&i) {
                p.line(&format!("decant_in.i{i} = decant_mesh_in.i{i};"));
                continue;
            }
            let flat = field.interpolation == "nointerpolation ";
            for (c, chan) in "xyzw".chars().take(usize::from(field.comps)).enumerate() {
                let word = if field.comps > 2 {
                    if c < 2 { ".x" } else { ".y" }
                } else {
                    ""
                };
                let source = format!("decant_mesh_in.i{i}{word}");
                let shift = (c % 2) * 16;
                let unpack = |src: String| format!("f16tof32(({src} >> {shift}u) & 0xffffu)");
                let value = if flat {
                    unpack(source)
                } else {
                    let values = (0..3)
                        .map(|v| unpack(format!("GetAttributeAtVertex({source}, {v})")))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("dot(float3({values}), decant_mesh_in.decant_bary)")
                };
                p.line(&format!("decant_in.i{i}.{chan} = {value};"));
            }
        }
    }
    for l in &heap_locals {
        p.line(l);
    }
    for l in &typed_row_lines {
        p.line(l);
    }
    for ty in [Ty::U32, Ty::Bool] {
        let (kind, init) = if ty == Ty::Bool {
            ("bool", "false")
        } else {
            ("uint", "0u")
        };
        let names: Vec<String> = hoisted
            .iter()
            .filter(|v| v.ty() == ty)
            .map(|v| format!("{} = {init}", var_name(*v)))
            .collect();
        for chunk in names.chunks(8) {
            p.line(&format!("{kind} {};", chunk.join(", ")));
        }
    }
    p.nodes(&func.body)?;
    // A body that ends without a return statement still has to give its outputs back.
    if !matches!(func.body.last(), Some(Node::Return)) && !cx.iface.outputs.is_empty() {
        p.line("return decant_out;");
    }
    p.out.push_str("}\n");
    Ok(p.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asm::{Asm, lit, s, sop1, sopp};
    use crate::ir::{Bin, OutputRef};
    use crate::tables::Op;

    fn compile_hlsl(text: &str, profile: &str) {
        if let Ok(dxc) = decant_dxc::Dxc::load() {
            let args = vec!["-Gis".to_owned(), "-denorm".to_owned(), "ftz".to_owned()];
            dxc.compile(text, ENTRY, profile, &args)
                .unwrap_or_else(|e| panic!("DXC compile failed:\n{e}\n{text}"));
        }
    }

    fn single_output_iface() -> Interface {
        Interface {
            outputs: vec![Field {
                semantic: "SV_Target".to_owned(),
                component_type: 1,
                comps: 1,
                first_comp: 0,
                interpolation: "",
            }],
            ..Interface::default()
        }
    }

    #[test]
    fn block_local_single_assignment_temps_are_declared_inline() {
        let iface = single_output_iface();
        let func = Function {
            body: vec![
                Node::Block(vec![
                    Stmt::Set {
                        dst: Var::Exec,
                        value: Expr::bin(Bin::Ne, Expr::Var(Var::V(0)), Expr::U32(0)),
                        guard: Guard::Always,
                    },
                    Stmt::Set {
                        dst: Var::S(0),
                        value: Expr::U32(0),
                        guard: Guard::Always,
                    },
                ]),
                Node::If {
                    cond: Expr::bin(Bin::Ne, Expr::Var(Var::V(1)), Expr::U32(0)),
                    then: vec![
                        Node::Block(vec![
                            // Single-assignment block-local Tmp and Flag: declared inline.
                            Stmt::Set {
                                dst: Var::Tmp(0),
                                value: Expr::bin(Bin::Add, Expr::Var(Var::V(0)), Expr::U32(7)),
                                guard: Guard::Always,
                            },
                            Stmt::Set {
                                dst: Var::Flag(1),
                                value: Expr::bin(Bin::ULt, Expr::Var(Var::Tmp(0)), Expr::U32(100)),
                                guard: Guard::Always,
                            },
                            Stmt::Set {
                                dst: Var::S(0),
                                value: Expr::select(
                                    Expr::Var(Var::Flag(1)),
                                    Expr::Var(Var::Tmp(0)),
                                    Expr::U32(1),
                                ),
                                guard: Guard::Always,
                            },
                            // Assigned twice in the same block: stays hoisted.
                            Stmt::Set {
                                dst: Var::Tmp(2),
                                value: Expr::U32(3),
                                guard: Guard::Always,
                            },
                            Stmt::Set {
                                dst: Var::Tmp(2),
                                value: Expr::bin(Bin::Add, Expr::Var(Var::Tmp(2)), Expr::U32(4)),
                                guard: Guard::Always,
                            },
                            // Read in a later block: stays hoisted.
                            Stmt::Set {
                                dst: Var::Tmp(3),
                                value: Expr::bin(Bin::Add, Expr::Var(Var::Tmp(2)), Expr::U32(5)),
                                guard: Guard::Always,
                            },
                            // Read in a following `if` condition: stays hoisted.
                            Stmt::Set {
                                dst: Var::Flag(4),
                                value: Expr::bin(Bin::Eq, Expr::Var(Var::Tmp(3)), Expr::U32(12)),
                                guard: Guard::Always,
                            },
                            // Guarded write under `if (exec)`: stays hoisted.
                            Stmt::Set {
                                dst: Var::Tmp(5),
                                value: Expr::U32(9),
                                guard: Guard::Exec,
                            },
                            // Reads itself on RHS: stays hoisted.
                            Stmt::Set {
                                dst: Var::Tmp(6),
                                value: Expr::bin(Bin::Add, Expr::Var(Var::Tmp(6)), Expr::U32(1)),
                                guard: Guard::Always,
                            },
                        ]),
                        Node::If {
                            cond: Expr::Var(Var::Flag(4)),
                            then: vec![Node::Block(vec![Stmt::Set {
                                dst: Var::S(0),
                                value: Expr::bin(
                                    Bin::Add,
                                    Expr::Var(Var::S(0)),
                                    Expr::bin(
                                        Bin::Add,
                                        Expr::Var(Var::Tmp(3)),
                                        Expr::bin(
                                            Bin::Add,
                                            Expr::Var(Var::Tmp(5)),
                                            Expr::Var(Var::Tmp(6)),
                                        ),
                                    ),
                                ),
                                guard: Guard::Always,
                            }])],
                            els: Vec::new(),
                        },
                    ],
                    els: Vec::new(),
                },
                Node::Loop(vec![
                    Node::Block(vec![
                        // Single-assignment loop-block-local Tmp: declared inline.
                        Stmt::Set {
                            dst: Var::Tmp(7),
                            value: Expr::bin(Bin::Add, Expr::Var(Var::S(0)), Expr::U32(2)),
                            guard: Guard::Always,
                        },
                        Stmt::Set {
                            dst: Var::S(0),
                            value: Expr::Var(Var::Tmp(7)),
                            guard: Guard::Always,
                        },
                    ]),
                    Node::Break,
                ]),
                Node::Block(vec![Stmt::Output {
                    to: OutputRef {
                        element: 0,
                        chan: 0,
                    },
                    value: Expr::Var(Var::S(0)),
                }]),
                Node::Return,
            ],
            resources: Vec::new(),
            groupshared_dwords: 0,
            lanes: 32,
        };
        let text = print(
            &func,
            &Context {
                iface: &iface,
                mirror: None,
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        assert!(text.contains("uint t0 = (v0 + 7u);"), "{text}");
        assert!(text.contains("bool f1 = (t0 < 100u);"), "{text}");
        assert!(text.contains("uint t7 = (s0 + 2u);"), "{text}");
        assert!(!text.contains("t0 = 0u"), "{text}");
        assert!(!text.contains("f1 = false"), "{text}");
        assert!(!text.contains("t7 = 0u"), "{text}");
        for hoisted in ["t2 = 0u", "t3 = 0u", "f4 = false", "t5 = 0u", "t6 = 0u"] {
            assert!(text.contains(hoisted), "missing {hoisted} in\n{text}");
        }
        compile_hlsl(&text, "ps_6_6");
    }

    #[test]
    fn code_table_slices_past_s_endpgm_and_preserves_indexing() {
        let iface = single_output_iface();
        let mut a = Asm::new();
        // Include a literal equal to the encoding of `s_endpgm` (0xbf810000) and an early
        // conditional `s_endpgm` followed by more reachable code before the final `s_endpgm`.
        let skip = a.label();
        a.push(sop1(Op::s_mov_b32, s(1), lit(0xbf81_0000)));
        a.branch(Op::s_cbranch_scc0, skip);
        a.push(sopp(Op::s_endpgm, 0));
        a.bind(skip);
        a.push(sop1(Op::s_mov_b32, s(2), lit(0x1234_5678)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut code = a.finish();
        let table_word = code.len();
        code.extend_from_slice(&[0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444]);
        let end_word = code.len();
        let func = Function {
            body: vec![
                Node::Block(vec![
                    Stmt::Set {
                        dst: Var::S(0),
                        value: Expr::CodeWord(Box::new(Expr::bin(
                            Bin::Add,
                            Expr::CodeAddress((table_word * 4) as u32),
                            Expr::bin(Bin::Shl, Expr::Var(Var::V(0)), Expr::U32(2)),
                        ))),
                        guard: Guard::Always,
                    },
                    Stmt::Output {
                        to: OutputRef {
                            element: 0,
                            chan: 0,
                        },
                        value: Expr::Var(Var::S(0)),
                    },
                ]),
                Node::Return,
            ],
            resources: Vec::new(),
            groupshared_dwords: 0,
            lanes: 32,
        };
        let text = print(
            &func,
            &Context {
                iface: &iface,
                mirror: None,
                header: &[],
                code: &code,
            },
        )
        .unwrap();
        assert!(
            text.contains(
                "static const uint decant_code[4] = { 0x11111111, 0x22222222, 0x33333333, 0x44444444 };"
            ),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "uint decant_code_at(uint a) {{ uint i = a >> 2; return i >= {table_word}u && i < {end_word}u ? decant_code[i - {table_word}u] : 0u; }}"
            )),
            "{text}"
        );
        compile_hlsl(&text, "ps_6_6");

        // Without trailing words after `s_endpgm`, keep the full code slice instead of an empty one.
        let no_tail = &code[..table_word];
        let text_no_tail = print(
            &func,
            &Context {
                iface: &iface,
                mirror: None,
                header: &[],
                code: no_tail,
            },
        )
        .unwrap();
        assert!(
            text_no_tail.contains(&format!("static const uint decant_code[{table_word}] = {{")),
            "{text_no_tail}"
        );
        compile_hlsl(&text_no_tail, "ps_6_6");
    }

    #[test]
    fn metadata_root_reads_route_through_getdimensions_helpers_for_cs_and_ps() {
        use crate::api::{DeclaredResource, MirrorAbi};
        use crate::ir::BufferOp;
        use crate::resource::trace::{HEAP_HOST, SHADOW_COUNT, SHADOW_STRIDE};

        let make_func = |with_output: bool| -> Function {
            let mut stmts = vec![
                Stmt::Set {
                    dst: Var::S(4),
                    value: Expr::Call(&HEAP_HOST, vec![Expr::Var(Var::V(0))]),
                    guard: Guard::Always,
                },
                Stmt::Set {
                    dst: Var::S(5),
                    value: Expr::bin(
                        Bin::Add,
                        Expr::Call(&SHADOW_STRIDE, vec![Expr::Var(Var::S(4))]),
                        Expr::Call(&SHADOW_COUNT, vec![Expr::Var(Var::S(4))]),
                    ),
                    guard: Guard::Always,
                },
                Stmt::Buffer(
                    BufferOp {
                        op: Op::buffer_load_dword,
                        glc: false,
                        store: false,
                        bytes: 4,
                        count: 1,
                        data: 8,
                        index: Some(Expr::Var(Var::V(0))),
                        offset: Expr::U32(0),
                        format: 0,
                        buffer: Desc::Res(ResId(0)),
                        pc: 0,
                    },
                    Guard::Always,
                ),
                Stmt::Buffer(
                    BufferOp {
                        op: Op::buffer_load_format_x,
                        glc: false,
                        store: false,
                        bytes: 4,
                        count: 1,
                        data: 9,
                        index: Some(Expr::Var(Var::V(1))),
                        offset: Expr::U32(0),
                        format: 0,
                        buffer: Desc::Res(ResId(1)),
                        pc: 0,
                    },
                    Guard::Always,
                ),
                Stmt::Buffer(
                    BufferOp {
                        op: Op::buffer_load_format_x,
                        glc: false,
                        store: false,
                        bytes: 4,
                        count: 1,
                        data: 10,
                        index: Some(Expr::Var(Var::V(2))),
                        offset: Expr::U32(0),
                        format: 0,
                        buffer: Desc::Res(ResId(1)),
                        pc: 0,
                    },
                    Guard::Always,
                ),
            ];
            let mut resources = vec![
                Resource {
                    origin: Origin::Table { sgpr: 4, offset: 0 },
                    usage: ResourceUse::Words { written: false },
                    declared: Some(DeclaredResource {
                        class: RegisterClass::Srv,
                        lower: 0,
                        upper: 0,
                        space: 0,
                        psv_type: 5,
                        psv_kind: 0,
                    }),
                    placement: Some(Placement::HeapAtMirror {
                        class: RegisterClass::Srv,
                        dword: 4,
                        slot: 0,
                    }),
                    name: "heap_t_s4_0".to_owned(),
                },
                Resource {
                    origin: Origin::Table {
                        sgpr: 4,
                        offset: 16,
                    },
                    usage: ResourceUse::Words { written: false },
                    declared: Some(DeclaredResource {
                        class: RegisterClass::Srv,
                        lower: 1,
                        upper: 1,
                        space: 0,
                        psv_type: 3,
                        psv_kind: 0,
                    }),
                    placement: Some(Placement::HeapAtMirror {
                        class: RegisterClass::Srv,
                        dword: 4,
                        slot: 1,
                    }),
                    name: "heap_t_s4_1".to_owned(),
                },
            ];
            if with_output {
                stmts.push(Stmt::Output {
                    to: OutputRef {
                        element: 0,
                        chan: 0,
                    },
                    value: Expr::bin(
                        Bin::Add,
                        Expr::Var(Var::S(5)),
                        Expr::bin(Bin::Add, Expr::Var(Var::V(8)), Expr::Var(Var::V(10))),
                    ),
                });
            } else {
                resources.push(Resource {
                    origin: Origin::Table { sgpr: 5, offset: 0 },
                    usage: ResourceUse::Words { written: true },
                    declared: Some(DeclaredResource {
                        class: RegisterClass::Uav,
                        lower: 0,
                        upper: 0,
                        space: 0,
                        psv_type: 7,
                        psv_kind: 0,
                    }),
                    placement: Some(Placement::HeapAtMirror {
                        class: RegisterClass::Uav,
                        dword: 5,
                        slot: 0,
                    }),
                    name: "heap_u_s5_0".to_owned(),
                });
                stmts.push(Stmt::Set {
                    dst: Var::V(11),
                    value: Expr::bin(
                        Bin::Add,
                        Expr::Var(Var::S(5)),
                        Expr::bin(Bin::Add, Expr::Var(Var::V(8)), Expr::Var(Var::V(10))),
                    ),
                    guard: Guard::Always,
                });
                stmts.push(Stmt::Buffer(
                    BufferOp {
                        op: Op::buffer_store_dword,
                        glc: false,
                        store: true,
                        bytes: 4,
                        count: 1,
                        data: 11,
                        index: None,
                        offset: Expr::U32(0),
                        format: 0,
                        buffer: Desc::Res(ResId(2)),
                        pc: 0,
                    },
                    Guard::Always,
                ));
            }
            Function {
                body: vec![Node::Block(stmts), Node::Return],
                resources,
                groupshared_dwords: 0,
                lanes: 32,
            }
        };

        let check = |text: &str| {
            assert!(text.contains(HEAP_MAP_LOAD.text), "{text}");
            assert!(text.contains(VIEW_SHADOW_LOAD.text), "{text}");
            assert!(text.contains(VIEW_SHADOW_LOAD2.text), "{text}");
            assert!(text.contains(VIEW_SHADOW_LOAD4.text), "{text}");
            // Each raw ByteAddressBuffer load occurs exactly once, inside its bounds helper.
            assert_eq!(text.matches("decant_heap_map.Load(").count(), 1, "{text}");
            assert_eq!(text.matches("decant_view_shadow.Load(").count(), 1, "{text}");
            assert_eq!(text.matches("decant_view_shadow.Load2(").count(), 1, "{text}");
            assert_eq!(text.matches("decant_view_shadow.Load4(").count(), 1, "{text}");
            // GetDimensions appears once in each of the 4 helpers, never inline in main.
            assert_eq!(text.matches(".GetDimensions(dim)").count(), 4, "{text}");
            let main_body = text.split(" main(").nth(1).expect("main body");
            assert!(!main_body.contains("GetDimensions"), "{main_body}");
            assert!(!main_body.contains("decant_heap_map.Load"), "{main_body}");
            assert!(!main_body.contains("decant_view_shadow.Load"), "{main_body}");
        };

        let cs_iface = Interface {
            threads: Some([32, 1, 1]),
            ..Interface::default()
        };
        let cs_text = print(
            &make_func(false),
            &Context {
                iface: &cs_iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        check(&cs_text);
        compile_hlsl(&cs_text, "cs_6_6");

        let ps_iface = single_output_iface();
        let ps_text = print(
            &make_func(true),
            &Context {
                iface: &ps_iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        check(&ps_text);
        compile_hlsl(&ps_text, "ps_6_6");

        // Shaders that do not read metadata roots must not emit unused helper definitions.
        let empty = Function {
            body: vec![
                Node::Block(vec![Stmt::Output {
                    to: OutputRef {
                        element: 0,
                        chan: 0,
                    },
                    value: Expr::U32(1),
                }]),
                Node::Return,
            ],
            resources: Vec::new(),
            groupshared_dwords: 0,
            lanes: 32,
        };
        let empty_text = print(
            &empty,
            &Context {
                iface: &ps_iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        assert!(!empty_text.contains("decant_heap_map_load"), "{empty_text}");
        assert!(!empty_text.contains("decant_view_shadow_load"), "{empty_text}");
        assert_eq!(empty_text.matches(HEAP_MAP).count(), 1, "{empty_text}");
        assert_eq!(empty_text.matches(VIEW_SHADOW).count(), 1, "{empty_text}");
    }
}
