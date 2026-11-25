//! Turns the origin of a console descriptor (the expression the `.src` file records) into a heap
//! index the shader can compute with the user data mirror (see the README of `decant-dxil`).
//!
//! The console reads a descriptor at a byte offset of the table that user register `sK` holds. With
//! the mirror, dword `K` is the heap index of the table's first descriptor, so a descriptor 32 bytes
//! (16 for a sampler) wide at byte `O` is heap element `mirror[K] + O / 32`, and a read at an offset
//! the shader computes (`ShiftLeftLogical32(index, #5)`) is `mirror[K] + index`.
//!
//! The expression language is the probe's print of the console program: `Name(arg,arg,)` with hex
//! literals `#1f`, `=Name(..)` for the value a scalar read stands for, and subtrees printed as a bare name
//! when they repeat an earlier one. Offsets of `ReadConstBuffer` are taken in bytes, as the console
//! instruction takes them.

use decant_dxil::spirv::IndexExpr;

use crate::identify::Resource;

/// One node of an origin expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub name: String,
    pub args: Vec<Node>,
}

/// Parses an origin expression. `None` when the text is not well formed.
pub fn parse(text: &str) -> Option<Node> {
    let bytes = text.as_bytes();
    let mut at = 0;
    let node = parse_node(bytes, &mut at)?;
    (at == bytes.len()).then_some(node)
}

fn parse_node(b: &[u8], at: &mut usize) -> Option<Node> {
    let start = *at;
    while *at < b.len() && !matches!(b[*at], b'(' | b')' | b',') {
        *at += 1;
    }
    let name = std::str::from_utf8(&b[start..*at]).ok()?.to_owned();
    let mut args = Vec::new();
    if *at < b.len() && b[*at] == b'(' {
        *at += 1;
        loop {
            match b.get(*at)? {
                b')' => {
                    *at += 1;
                    break;
                }
                b',' => *at += 1,
                _ => args.push(parse_node(b, at)?),
            }
        }
    }
    Some(Node { name, args })
}

fn literal(n: &Node) -> Option<u32> {
    u32::from_str_radix(n.name.strip_prefix('#')?, 16).ok()
}

fn user_register(n: &Node) -> Option<u32> {
    if n.name != "GetUserData" {
        return None;
    }
    n.args.first()?.name.strip_prefix('s')?.parse().ok()
}

fn at_offset(n: &Node, base: &str) -> Option<u32> {
    n.name.strip_prefix(base)?.strip_prefix('@')?.parse().ok()
}

/// The descriptor of a table read: where it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableRead {
    /// The user register that holds the table.
    pub register: u32,
    /// The byte offset that is a constant.
    pub offset: u32,
    /// The descriptor number that is computed (`None` for a constant offset only), without the shift.
    pub index: Option<IndexExpr>,
    /// The shift that turned the index into bytes.
    pub shift: u32,
}

/// Reads `LoadAddressU32@A(GetAddressResource(GetUserData(sK),..),DYN,#X,imm,)`. `buffer_of` finds the
/// shader buffer whose descriptor is stored at a table offset: `(table register if printed, offset)`.
pub fn table_read(origin: &str, buffer_of: &dyn Fn(Option<u32>, u32) -> Option<(u32, u32, u32)>, current: &mut Option<(u32, u32, u32)>) -> Result<TableRead, String> {
    let mut node = parse(origin).ok_or("expression does not parse")?;
    // A table read the console stored in its scalar registers: `ReadConst(GetSrtResource,#i,=LoadAddressU32@O(..))`.
    if node.name == "ReadConst"
        && let Some(inner) = node.args.iter().find(|a| a.name.starts_with('=')).cloned()
    {
        node = Node { name: inner.name.trim_start_matches('=').to_owned(), args: inner.args };
    }
    let base = at_offset(&node, "LoadAddressU32").ok_or_else(|| {
        if user_register(&node).is_some() { "root descriptor".to_owned() } else { format!("origin {}", node.name) }
    })?;
    let table = node.args.first().filter(|n| n.name == "GetAddressResource").and_then(|n| n.args.first()).and_then(user_register).ok_or("table is not a user register")?;
    let dynamic = node.args.get(1).ok_or("no index argument")?;
    let extra = node.args.get(2).and_then(literal).ok_or("no constant offset")?;
    let offset = base.checked_add(extra).ok_or("offset overflow")?;
    if literal(dynamic).is_some() {
        return Ok(TableRead { register: table, offset: offset + literal(dynamic).unwrap_or(0), index: None, shift: 0 });
    }
    if dynamic.name != "ShiftLeftLogical32" {
        return Err(format!("index is {}", dynamic.name));
    }
    let shift = dynamic.args.get(1).and_then(literal).ok_or("no shift")?;
    let index = index_expr(dynamic.args.first().ok_or("no index")?, buffer_of, current)?;
    Ok(TableRead { register: table, offset, index: Some(index), shift })
}

/// An index computation: user registers, constants, adds, shifts and words of the shader's buffers.
fn index_expr(n: &Node, buffer_of: &dyn Fn(Option<u32>, u32) -> Option<(u32, u32, u32)>, current: &mut Option<(u32, u32, u32)>) -> Result<IndexExpr, String> {
    if let Some(v) = literal(n) {
        return Ok(IndexExpr::Const(v));
    }
    if let Some(k) = user_register(n) {
        return Ok(IndexExpr::User(k));
    }
    match n.name.as_str() {
        "IAdd32" => {
            let a = index_expr(n.args.first().ok_or("IAdd32 needs two arguments")?, buffer_of, current)?;
            let b = index_expr(n.args.get(1).ok_or("IAdd32 needs two arguments")?, buffer_of, current)?;
            return Ok(a.plus(b));
        }
        "IMul32" | "BitwiseAnd32" => {
            let a = Box::new(index_expr(n.args.first().ok_or("an operation needs two arguments")?, buffer_of, current)?);
            let b = Box::new(index_expr(n.args.get(1).ok_or("an operation needs two arguments")?, buffer_of, current)?);
            return Ok(if n.name == "IMul32" { IndexExpr::Mul(a, b) } else { IndexExpr::And(a, b) });
        }
        "ShiftLeftLogical32" | "ShiftRightLogical32" => {
            let a = index_expr(n.args.first().ok_or("a shift needs an operand")?, buffer_of, current)?;
            let by = n.args.get(1).and_then(literal).ok_or("a shift by a value that is not constant")?;
            return Ok(if n.name == "ShiftLeftLogical32" { IndexExpr::Shl(Box::new(a), by) } else { IndexExpr::Shr(Box::new(a), by) });
        }
        // A scalar read of the resource table: its value is what the `=` says.
        "ReadConst" => {
            let inner = n.args.iter().find(|a| a.name.starts_with('=')).ok_or("a scalar read without a value")?;
            let inner = Node { name: inner.name.trim_start_matches('=').to_owned(), args: inner.args.clone() };
            return index_expr(&inner, buffer_of, current);
        }
        _ => {}
    }
    if let Some(constant) = at_offset(n, "ReadConstBuffer") {
        let descriptor = n.args.first().filter(|a| a.name == "GetBufferResource").ok_or("a buffer read without a buffer")?;
        // The four dwords of the buffer descriptor: the first says where it is stored. A bare
        // `GetBufferResource` repeats the buffer printed before it in the same expression.
        let key = match descriptor.args.first() {
            Some(first) => {
                let stored = first.args.iter().find(|a| a.name.starts_with("=LoadAddressU32")).ok_or("a buffer read of an elided buffer")?;
                let offset = at_offset(&Node { name: stored.name.trim_start_matches('=').to_owned(), args: Vec::new() }, "LoadAddressU32").ok_or("a buffer descriptor without an offset")?;
                let table = stored.args.first().filter(|a| a.name == "GetAddressResource").and_then(|a| a.args.first()).and_then(user_register);
                let key = buffer_of(table, offset).ok_or_else(|| format!("the buffer stored at table offset {offset} is not a buffer of the shader"))?;
                *current = Some(key);
                key
            }
            None => current.ok_or("a buffer read of an elided buffer")?,
        };
        let at = match n.args.get(1) {
            Some(o) => index_expr(o, buffer_of, current)?,
            None => IndexExpr::Const(0),
        };
        return Ok(IndexExpr::BufferWord { buffer: key, offset: Box::new(IndexExpr::Const(constant).plus(at)) });
    }
    Err(format!("expression {}", n.name))
}

/// The user register of a root descriptor: an origin that is `GetUserData(sK)` itself.
pub fn root_descriptor(origin: &str) -> Option<u32> {
    user_register(&parse(origin)?)
}

/// The heap index of a table read: `mirror[K] + offset / size + index`. `size` is 32 for resources, 16 for samplers.
pub fn heap_index(read: &TableRead, size: u32) -> Result<IndexExpr, String> {
    let per = size.trailing_zeros();
    if read.index.is_some() && read.shift != per {
        return Err(format!("descriptors are {size} bytes, the index is shifted by {}", read.shift));
    }
    if !read.offset.is_multiple_of(size) {
        return Err(format!("offset {} is not a multiple of {size}", read.offset));
    }
    let mut e = IndexExpr::User(read.register).plus(IndexExpr::Const(read.offset / size));
    if let Some(i) = &read.index {
        e = e.plus(i.clone());
    }
    Ok(e)
}

/// The heap placement of every resource that has a table origin or a computed one, or why not.
///
/// `current` is the buffer the probe printed last: a bare `GetBufferResource` repeats it, also from the
/// expression of the next resource, so pass the same value for the resources of one module in order.
pub fn placement(origin_text: &str, kind: &str, resources: &[Resource], base: i64, current: &mut Option<(u32, u32, u32)>) -> Result<IndexExpr, String> {
    let buffer_of = |table: Option<u32>, offset: u32| -> Option<(u32, u32, u32)> {
        let r = resources.iter().find(|r| r.kind == "buffer" && r.offset == Some(u64::from(offset)) && table.is_none_or(|t| r.table == Some(u64::from(t))))?;
        Some((0, (base + r.binding?) as u32, r.rank.filter(|&k| k < 0xFFFF_FFFF).unwrap_or(0) as u32))
    };
    let read = table_read(origin_text, &buffer_of, current)?;
    heap_index(&read, if kind == "sampler" { 16 } else { 32 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_buffers(_: Option<u32>, _: u32) -> Option<(u32, u32, u32)> {
        None
    }

    #[test]
    fn the_parser_reads_nested_calls_literals_and_equals() {
        let n = parse("ReadConst(GetSrtResource,#34,=LoadAddressU32@64(GetAddressResource(GetUserData(s4),#0,),#0,#0,imm,))").unwrap();
        assert_eq!(n.name, "ReadConst");
        assert_eq!(n.args.len(), 3);
        assert_eq!(n.args[2].name, "=LoadAddressU32@64");
        assert!(parse("Foo(").is_none());
    }

    #[test]
    fn a_static_read_is_the_table_plus_the_descriptor_number() {
        let origin = "LoadAddressU32@352(GetAddressResource(GetUserData(s9),#0,),#0,#0,imm,)";
        let read = table_read(origin, &no_buffers, &mut None).unwrap();
        assert_eq!((read.register, read.offset, read.index.clone()), (9, 352, None));
        assert_eq!(heap_index(&read, 32).unwrap(), IndexExpr::User(9).plus(IndexExpr::Const(11)));
    }

    #[test]
    fn a_sampler_table_has_sixteen_byte_descriptors() {
        let origin = "LoadAddressU32@0(GetAddressResource(GetUserData(s8),#0,),ShiftLeftLogical32(GetUserData(s9),#4,),#0,imm,)";
        let read = table_read(origin, &no_buffers, &mut None).unwrap();
        assert_eq!(heap_index(&read, 16).unwrap(), IndexExpr::User(8).plus(IndexExpr::Const(0)).plus(IndexExpr::User(9)));
        assert!(heap_index(&read, 32).is_err());
    }

    #[test]
    fn a_bindless_index_comes_from_a_buffer_of_the_shader() {
        let origin = "LoadAddressU32@0(GetAddressResource(GetUserData(s15),#0,),ShiftLeftLogical32(ReadConstBuffer@0(GetBufferResource(ReadConst(GetSrtResource,#0,=LoadAddressU32@1856(GetAddressResource,#0,#0,imm,)),ReadConst(GetSrtResource,#1,=LoadAddressU32@1860(GetAddressResource,#0,#0,imm,)),),IAdd32(GetUserData(s14),#4,),),#5,),#0,imm,)";
        let buffer_of = |_: Option<u32>, offset: u32| (offset == 1856).then_some((0, 0, 0));
        let read = table_read(origin, &buffer_of, &mut None).unwrap();
        let want = IndexExpr::BufferWord { buffer: (0, 0, 0), offset: Box::new(IndexExpr::Const(0).plus(IndexExpr::User(14).plus(IndexExpr::Const(4)))) };
        assert_eq!(heap_index(&read, 32).unwrap(), IndexExpr::User(15).plus(IndexExpr::Const(0)).plus(want));
    }

    #[test]
    fn an_elided_buffer_is_the_one_printed_before_it() {
        let origin = "LoadAddressU32@0(GetAddressResource(GetUserData(s15),#0,),ShiftLeftLogical32(ReadConstBuffer@4(GetBufferResource(ReadConst(GetSrtResource,#37,=LoadAddressU32@160(GetAddressResource,#0,#0,imm,)),),ShiftRightLogical32(ReadConst(GetSrtResource,#57,=ReadConstBuffer@32(GetBufferResource,#0,)),#10,),),#5,),#0,imm,)";
        let buffer_of = |_: Option<u32>, offset: u32| (offset == 160).then_some((0, 60, 3));
        let read = table_read(origin, &buffer_of, &mut None).unwrap();
        let inner = IndexExpr::BufferWord { buffer: (0, 60, 3), offset: Box::new(IndexExpr::Const(32).plus(IndexExpr::Const(0))) };
        let outer = IndexExpr::BufferWord { buffer: (0, 60, 3), offset: Box::new(IndexExpr::Const(4).plus(IndexExpr::Shr(Box::new(inner), 16))) };
        assert_eq!(read.index, Some(outer));
    }

    #[test]
    fn a_root_descriptor_and_an_elided_buffer_are_refused_with_a_reason() {
        assert_eq!(table_read("GetUserData(s10)", &no_buffers, &mut None).unwrap_err(), "root descriptor");
        let elided = "LoadAddressU32@0(GetAddressResource(GetUserData(s9),#0,),ShiftLeftLogical32(ReadConstBuffer@4(GetBufferResource,#0,),#5,),#0,imm,)";
        assert!(table_read(elided, &no_buffers, &mut None).unwrap_err().contains("elided"));
    }
}
