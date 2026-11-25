//! A text form of the IR for debug listings and golden tests: `Function` implements `Display`.
//!
//! One statement per line, two spaces of indent per level, every binary operator in
//! parentheses so that the text has one reading. Vector statements carry the prefix `@exec`.

use std::fmt::{self, Display, Formatter, Write};

use super::{Bin, BufferOp, Desc, Expr, Function, Guard, ImageOp, Input, Node, Stmt, Un, Var};

impl Display for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Var::S(n) => write!(f, "s{n}"),
            Var::V(n) => write!(f, "v{n}"),
            Var::Ttmp(n) => write!(f, "ttmp{n}"),
            Var::M0 => f.write_str("m0"),
            Var::Scc => f.write_str("scc"),
            Var::Vcc => f.write_str("vcc"),
            Var::Exec => f.write_str("exec"),
            Var::Mask(n) => write!(f, "mask{n}"),
            Var::Tmp(n) => write!(f, "t{n}"),
            Var::Flag(n) => write!(f, "f{n}"),
            Var::Spill { reg, lane } => write!(f, "spill{reg}_{lane}"),
        }
    }
}

fn un_name(op: Un) -> &'static str {
    match op {
        Un::Not => "not",
        Un::LNot => "lnot",
        Un::Neg => "neg",
        Un::FNeg => "fneg",
        Un::FAbs => "fabs",
        Un::BoolToU32 => "b2u",
        Un::U32ToBool => "u2b",
        Un::U32ToU64 => "zext64",
        Un::U64ToU32 => "trunc32",
    }
}

fn bin_name(op: Bin) -> &'static str {
    match op {
        Bin::Add => "+",
        Bin::Sub => "-",
        Bin::Mul => "*",
        Bin::And => "&",
        Bin::Or => "|",
        Bin::Xor => "^",
        Bin::Shl => "<<",
        Bin::LShr => ">>",
        Bin::AShr => ">>s",
        Bin::FAdd => "+f",
        Bin::FSub => "-f",
        Bin::FMul => "*f",
        Bin::Eq => "==",
        Bin::Ne => "!=",
        Bin::ULt => "<u",
        Bin::ULe => "<=u",
        Bin::SLt => "<s",
        Bin::SLe => "<=s",
        Bin::FLt => "<f",
        Bin::FLe => "<=f",
        Bin::FEq => "==f",
        Bin::FNe => "!=f",
        Bin::LAnd => "&&",
        Bin::LOr => "||",
        Bin::LXor => "^^",
    }
}

impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Input::Attr { attr, chan } => write!(f, "attr{attr}.{chan}"),
            Input::VertexElement { register, chan } => write!(f, "vin{register}.{chan}"),
            Input::VertexId => f.write_str("vertex_id"),
            Input::InstanceId => f.write_str("instance_id"),
            Input::FragCoord { chan } => write!(f, "frag_coord.{chan}"),
            Input::FrontFace => f.write_str("front_face"),
            Input::HelperLane => f.write_str("helper_lane"),
            Input::LocalThreadId { axis } => write!(f, "local_id.{axis}"),
            Input::GroupId { axis } => write!(f, "group_id.{axis}"),
            Input::LaneIndex => f.write_str("lane_index"),
            Input::GroupIndex => f.write_str("group_index"),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BvhWord(op, word) => write!(
                f,
                "bvh{}[{}] descriptor[{}] address[{}] a16={} pc={}",
                if op.wide { "64" } else { "" },
                word,
                joined(&op.descriptor),
                joined(&op.address),
                op.encoding.a16,
                op.pc
            ),
            Expr::U32(v) if *v < 10 => write!(f, "{v}"),
            Expr::U32(v) => write!(f, "{v:#x}"),
            Expr::U64(v) => write!(f, "{v:#x}L"),
            Expr::Bool(v) => write!(f, "{v}"),
            Expr::Var(v) => write!(f, "{v}"),
            Expr::Un(op, a) => write!(f, "{}({a})", un_name(*op)),
            Expr::Bin(op, a, b) => write!(f, "({a} {} {b})", bin_name(*op)),
            Expr::Select(c, a, b) => write!(f, "({c} ? {a} : {b})"),
            Expr::Call(h, args) => write!(f, "{}({})", h.name, joined(args)),
            Expr::Input(i) => write!(f, "in.{i}"),
            Expr::UserData(k) => write!(f, "user{k}"),
            Expr::Load(l) => write!(
                f,
                "{}[s{}, {}]",
                if l.buffer { "buffer_load" } else { "load" },
                l.base,
                l.offset
            ),
            Expr::BufferWord { res, offset } => write!(f, "res{}[{offset}]", res.0),
            Expr::Mirror(k) => write!(f, "mirror{k}"),
            Expr::CodeAddress(k) => write!(f, "code+{k:#x}"),
            Expr::VectorLoad { base, offset, .. } => write!(f, "load[v{base}, {offset}]"),
            Expr::CodeWord(at) => write!(f, "code[{at}]"),
        }
    }
}

fn joined(items: &[Expr]) -> String {
    items
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn desc(d: &Desc) -> String {
    match d {
        Desc::Regs(n) => format!("s[{n}..]"),
        Desc::Res(r) => format!("res{}", r.0),
    }
}

fn image(op: &ImageOp) -> String {
    let sampler = op
        .sampler
        .as_ref()
        .map(|s| format!(", {}", desc(s)))
        .unwrap_or_default();
    format!(
        "image {:?} dim{} dmask{:#x} -> v{} ({}{sampler}) at [{}]",
        op.kind,
        op.dim,
        op.dmask,
        op.data,
        desc(&op.image),
        joined(&op.address)
    )
}

fn buffer(op: &BufferOp) -> String {
    let format = if op.format != 0 {
        format!(" fmt{}", op.format)
    } else {
        String::new()
    };
    let index = op
        .index
        .as_ref()
        .map(|i| format!(" [idx {i}]"))
        .unwrap_or_default();
    format!(
        "buffer {} {}x{}B v{} {}{format}{index} +{}",
        if op.store { "store" } else { "load" },
        op.count,
        op.bytes,
        op.data,
        desc(&op.buffer),
        op.offset
    )
}

fn guard(g: Guard) -> &'static str {
    match g {
        Guard::Always => "",
        Guard::Exec => "@exec ",
    }
}

fn stmt(s: &Stmt) -> String {
    match s {
        Stmt::Set {
            dst,
            value,
            guard: g,
        } => format!("{}{dst} = {value}", guard(*g)),
        Stmt::Image(op, g) => format!("{}{}", guard(*g), image(op)),
        Stmt::Buffer(op, g) => format!("{}{}", guard(*g), buffer(op)),
        Stmt::Export(e) => format!(
            "export target{} en{:#x} src{:?}{}{}{}",
            e.target,
            e.enabled,
            e.src,
            if e.compressed { " compr" } else { "" },
            if e.done { " done" } else { "" },
            if e.valid_mask { " vm" } else { "" }
        ),
        Stmt::Output { to, value } => format!("out{}.{} = {value}", to.element, to.chan),
        Stmt::Discard { cond } => format!("discard if {cond}"),
        Stmt::Effect {
            dst: Some(dst),
            call,
            guard: g,
        } => format!("{}{dst} = effect {call}", guard(*g)),
        Stmt::Effect {
            dst: None,
            call,
            guard: g,
        } => format!("{}effect {call}", guard(*g)),
        Stmt::Message { id, .. } => format!("message {id}"),
        Stmt::Barrier { .. } => "barrier".to_owned(),
        Stmt::Comment(t) => format!("// {t}"),
    }
}

fn line(out: &mut String, depth: usize, text: &str) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push_str(text);
    out.push('\n');
}

fn nodes(out: &mut String, depth: usize, ns: &[Node]) {
    for n in ns {
        match n {
            Node::Block(stmts) => stmts.iter().for_each(|s| line(out, depth, &stmt(s))),
            Node::If { cond, then, els } => {
                line(out, depth, &format!("if {cond} {{"));
                nodes(out, depth + 1, then);
                if !els.is_empty() {
                    line(out, depth, "} else {");
                    nodes(out, depth + 1, els);
                }
                line(out, depth, "}");
            }
            Node::Loop(body) => {
                line(out, depth, "loop {");
                nodes(out, depth + 1, body);
                line(out, depth, "}");
            }
            Node::Once(body) => {
                line(out, depth, "once {");
                nodes(out, depth + 1, body);
                line(out, depth, "}");
            }
            Node::Break => line(out, depth, "break"),
            Node::Continue => line(out, depth, "continue"),
            Node::Return => line(out, depth, "return"),
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        for (i, r) in self.resources.iter().enumerate() {
            writeln!(out, "res{i}: {:?} {:?} {}", r.origin, r.usage, r.name)?;
        }
        nodes(&mut out, 0, &self.body);
        f.write_str(&out)
    }
}
