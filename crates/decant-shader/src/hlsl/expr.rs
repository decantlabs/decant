//! Expressions as HLSL text.

use std::fmt::Write as _;

use crate::api::{Placement, RegisterClass};
use crate::ir::{Bin, Expr, Function, Input, Node, Resource, Stmt, Ty, Un, Var};
use crate::stage::Interface;

/// The name of the array that holds the program's own code words, for `Expr::CodeWord`.
pub const CODE: &str = "decant_code";

/// The local variable of a register.
pub fn var_name(v: Var) -> String {
    let mut out = String::new();
    push_var(v, &mut out);
    out
}

/// Appends the local variable of a register to `o`.
pub fn push_var(v: Var, o: &mut String) {
    let _ = match v {
        Var::S(n) => write!(o, "s{n}"),
        Var::V(n) => write!(o, "v{n}"),
        Var::Ttmp(n) => write!(o, "ttmp{n}"),
        Var::M0 => write!(o, "m0"),
        Var::Scc => write!(o, "scc"),
        Var::Vcc => write!(o, "vcc"),
        Var::Exec => write!(o, "exec"),
        Var::Mask(n) => write!(o, "mask{n}"),
        Var::Tmp(n) => write!(o, "t{n}"),
        Var::Flag(n) => write!(o, "f{n}"),
        Var::Spill { reg, lane } => write!(o, "spill{reg}_{lane}"),
    };
}

fn component(chan: u8) -> &'static str {
    ["x", "y", "z", "w"][usize::from(chan & 3)]
}

/// One component of an input struct member as a 32 bit pattern. A component the member does not
/// have reads as zero.
fn input_member(iface: &Interface, member: Option<usize>, chan: u8) -> String {
    let Some((i, f)) = member.and_then(|i| iface.inputs.get(i).map(|f| (i, f))) else {
        return "0u".to_owned();
    };
    // `chan` is a component of the register, the member starts at the first component of its
    // element.
    let Some(rel) = chan.checked_sub(f.first_comp).filter(|r| *r < f.comps) else {
        return "0u".to_owned();
    };
    let value = if f.comps <= 1 {
        format!("decant_in.i{i}")
    } else {
        format!("decant_in.i{i}.{}", component(rel))
    };
    match f.component_type {
        1 => value,
        2 => format!("uint({value})"),
        _ => format!("asuint({value})"),
    }
}

/// The HLSL text of `e`.
pub fn print(e: &Expr, iface: &Interface, resources: &[Resource]) -> String {
    let mut out = String::new();
    write(e, iface, resources, &mut out);
    out
}

/// Appends the HLSL text of `e` to `o`. One buffer for the whole expression: a large program
/// has millions of expression nodes, and a string per node made the printer the second
/// largest allocator of the translation.
pub fn write(e: &Expr, iface: &Interface, resources: &[Resource], o: &mut String) {
    let w = |e: &Expr, o: &mut String| write(e, iface, resources, o);
    // Text, the first operand, text, the second operand, text.
    let around = |o: &mut String, pre: &str, a: &Expr, mid: &str, b: &Expr, post: &str| {
        o.push_str(pre);
        w(a, o);
        o.push_str(mid);
        w(b, o);
        o.push_str(post);
    };
    match e {
        Expr::U32(v) => {
            let _ = write!(o, "{v}u");
        }
        Expr::U64(v) => {
            let _ = write!(o, "{v}ull");
        }
        Expr::Bool(v) => o.push_str(if *v { "true" } else { "false" }),
        Expr::Var(v) => push_var(*v, o),
        Expr::Un(op, a) => {
            let wide = a.ty() == Ty::U64;
            let (pre, post) = match op {
                Un::Neg if wide => ("(0ull - ", ")"),
                Un::Not => ("(~", ")"),
                Un::LNot => ("(!", ")"),
                Un::Neg => ("(0u - ", ")"),
                Un::FNeg => ("(", " ^ 0x80000000u)"),
                Un::FAbs => ("(", " & 0x7fffffffu)"),
                Un::BoolToU32 => ("(", " ? 1u : 0u)"),
                Un::U32ToBool => ("(", " != 0u)"),
                Un::U32ToU64 => ("uint64_t(", ")"),
                Un::U64ToU32 => ("uint(", ")"),
            };
            o.push_str(pre);
            w(a, o);
            o.push_str(post);
        }
        Expr::Bin(op, a, b) => {
            let wide = a.ty() == Ty::U64;
            if wide {
                let shift = match op {
                    Bin::Shl => Some(("(", " << ")),
                    Bin::LShr => Some(("(", " >> ")),
                    Bin::AShr => Some(("uint64_t(int64_t(", ") >> ")),
                    _ => None,
                };
                if let Some((pre, mid)) = shift {
                    o.push_str(pre);
                    w(a, o);
                    o.push_str(mid);
                    // The shift count of a 64 bit shift.
                    if b.ty() == Ty::U64 {
                        o.push_str("(uint(");
                        w(b, o);
                        o.push_str(") & 63u))");
                    } else {
                        o.push('(');
                        w(b, o);
                        o.push_str(" & 63u))");
                    }
                    return;
                }
                match op {
                    Bin::SLt => return around(o, "(int64_t(", a, ") < int64_t(", b, "))"),
                    Bin::SLe => return around(o, "(int64_t(", a, ") <= int64_t(", b, "))"),
                    _ => {}
                }
            }
            let (pre, mid, post) = match op {
                Bin::Add => ("(", " + ", ")"),
                Bin::Sub => ("(", " - ", ")"),
                Bin::Mul => ("(", " * ", ")"),
                Bin::And => ("(", " & ", ")"),
                Bin::Or => ("(", " | ", ")"),
                Bin::Xor => ("(", " ^ ", ")"),
                Bin::Shl => ("(", " << (", " & 31u))"),
                Bin::LShr => ("(", " >> (", " & 31u))"),
                Bin::AShr => ("uint(int(", ") >> (", " & 31u))"),
                Bin::FAdd => ("asuint(asfloat(", ") + asfloat(", "))"),
                Bin::FSub => ("asuint(asfloat(", ") - asfloat(", "))"),
                Bin::FMul => ("asuint(asfloat(", ") * asfloat(", "))"),
                Bin::Eq => ("(", " == ", ")"),
                Bin::Ne => ("(", " != ", ")"),
                Bin::ULt => ("(", " < ", ")"),
                Bin::ULe => ("(", " <= ", ")"),
                Bin::SLt => ("(int(", ") < int(", "))"),
                Bin::SLe => ("(int(", ") <= int(", "))"),
                Bin::FLt => ("(asfloat(", ") < asfloat(", "))"),
                Bin::FLe => ("(asfloat(", ") <= asfloat(", "))"),
                Bin::FEq => ("(asfloat(", ") == asfloat(", "))"),
                Bin::FNe => ("(asfloat(", ") != asfloat(", "))"),
                Bin::LAnd => ("(", " && ", ")"),
                Bin::LOr => ("(", " || ", ")"),
                Bin::LXor => ("(", " != ", ")"),
            };
            around(o, pre, a, mid, b, post);
        }
        Expr::Select(c, a, b) => {
            o.push('(');
            w(c, o);
            around(o, " ? ", a, " : ", b, ")");
        }
        Expr::Call(h, args) => {
            o.push_str(h.name);
            o.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    o.push_str(", ");
                }
                w(a, o);
            }
            o.push(')');
        }
        Expr::Input(i) => match i {
            Input::Attr { attr, chan } => o.push_str(&input_member(
                iface,
                iface.attrs.get(usize::from(*attr)).copied(),
                *chan,
            )),
            Input::VertexElement { register, chan } => o.push_str(&input_member(
                iface,
                iface.vertex_inputs.get(register).copied(),
                *chan,
            )),
            Input::VertexId => o.push_str("decant_in.decant_vertex_id"),
            Input::InstanceId => o.push_str("decant_in.decant_instance_id"),
            Input::LocalThreadId { axis } => {
                o.push_str("decant_tid.");
                o.push_str(component(*axis));
            }
            Input::GroupIndex => o.push_str("decant_gi"),
            Input::GroupId { axis } => {
                o.push_str("decant_gid.");
                o.push_str(component(*axis));
            }
            Input::LaneIndex => o.push_str(iface.lane_index.as_deref().unwrap_or("0u")),
            // The stage files do not produce the other inputs yet.
            Input::FragCoord { .. } => o.push_str("0u"),
            Input::FrontFace | Input::HelperLane => o.push_str("false"),
        },
        Expr::Mirror(k) => {
            let _ = write!(o, "decant_mirror[{}].{}", k >> 2, component(k & 3));
        }
        Expr::CodeAddress(k) => {
            let _ = write!(o, "{k}u");
        }
        // Gone before printing: `translate` refuses a program that still has one.
        Expr::VectorLoad { .. } => o.push_str("0u"),
        // A constant address was folded to its word before printing (`simplify::fold_code`).
        Expr::CodeWord(at) => {
            o.push_str(CODE);
            o.push_str("_at(");
            w(at, o);
            o.push(')');
        }
        Expr::BufferWord { res, offset } => {
            let r = &resources[res.0 as usize];
            let class = r.placement.as_ref().map(|p| match p {
                Placement::Register { class, .. }
                | Placement::HeapAtMirror { class, .. }
                | Placement::HeapAtConsoleIndex { class } => *class,
                Placement::HeapSampler { .. } => RegisterClass::Sampler,
            });
            match (&**offset, class) {
                (Expr::U32(k), Some(RegisterClass::Cbv)) => {
                    let _ = write!(o, "{}.w[{}][{}]", r.name, k >> 4, (k >> 2) & 3);
                }
                _ => o.push_str(&super::buffer::word(
                    &r.name,
                    class,
                    &print(offset, iface, resources),
                )),
            }
        }
        // Both are gone before printing: the resource pass replaces loads, placement the user data.
        Expr::UserData(_) | Expr::Load(_) => {
            o.push_str(if e.ty() == Ty::Bool { "false" } else { "0u" });
        }
    }
}

fn walk(e: &Expr, f: &mut impl FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Un(_, a) => walk(a, f),
        Expr::Bin(_, a, b) => {
            walk(a, f);
            walk(b, f);
        }
        Expr::Select(c, a, b) => {
            walk(c, f);
            walk(a, f);
            walk(b, f);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| walk(a, f)),
        Expr::BufferWord { offset, .. } => walk(offset, f),
        Expr::CodeWord(at) => walk(at, f),
        _ => {}
    }
}

/// Calls `f` on every expression of the function, sub expressions included.
pub fn for_each_expr(func: &Function, f: &mut impl FnMut(&Expr)) {
    fn nodes(ns: &[Node], f: &mut impl FnMut(&Expr)) {
        for n in ns {
            match n {
                Node::Block(stmts) => {
                    for s in stmts {
                        match s {
                            Stmt::Set { value, .. } | Stmt::Output { value, .. } => walk(value, f),
                            Stmt::Discard { cond } => walk(cond, f),
                            Stmt::Effect { call, .. } => walk(call, f),
                            Stmt::Image(op, _) => op.address.iter().for_each(|a| walk(a, f)),
                            Stmt::Buffer(op, _) => {
                                walk(&op.offset, f);
                                if let Some(i) = &op.index {
                                    walk(i, f);
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
                    walk(cond, f);
                    nodes(then, f);
                    nodes(els, f);
                }
                Node::Loop(body) | Node::Once(body) => nodes(body, f),
                Node::Break | Node::Continue | Node::Return => {}
            }
        }
    }
    nodes(&func.body, f);
}

/// Sets `found` when the function reads the mirror as a number.
pub fn any_mirror(func: &Function, found: &mut bool) {
    for_each_expr(func, &mut |e| *found |= matches!(e, Expr::Mirror(_)));
}

/// Which system inputs the function reads.
#[derive(Default)]
pub struct InputUses {
    pub vertex_id: bool,
    pub instance_id: bool,
}

pub fn input_uses(func: &Function) -> InputUses {
    let mut uses = InputUses::default();
    for_each_expr(func, &mut |e| match e {
        Expr::Input(Input::VertexId) => uses.vertex_id = true,
        Expr::Input(Input::InstanceId) => uses.instance_id = true,
        _ => {}
    });
    uses
}
