//! Which descriptor heap slots a shader reads through `ResourceDescriptorHeap` and
//! `SamplerDescriptorHeap`, and as what, from its DXIL disassembly.
//!
//! A heap slot has to hold a descriptor of the type the shader uses there: WARP removes the device
//! when a constant buffer is read through a view descriptor or the other way round. The index of a
//! slot is usually a value of the user data mirror (`b1` of space 15) plus a constant. The runner
//! fills the mirror of such shaders with [`mirror_bytes`]: dword `k` holds `k * MIRROR_STRIDE`, so
//! slots reached from different dwords stay apart, and every index that is a mirror dword at a
//! constant row plus or times a constant becomes a known number here. The slot gets a resource of
//! the kind the following `annotateHandle` names.
//!
//! The facts used are the public DXIL operations `createHandleFromBinding`, `createHandleFromHeap`,
//! `annotateHandle` and `cbufferLoadLegacy` and the layout of `ResourceProperties` (word 0: the
//! resource kind in bits 0 to 7, the UAV bit 12, word 1: the component type and count of typed
//! resources, the stride of structured buffers, the size of constant buffers).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::{Class, Ret, Shape};

/// The distance between the heap indices two neighbouring mirror dwords give.
pub const MIRROR_STRIDE: u32 = 64;

/// The mirror contents of a shader that indexes the descriptor heap: dword `k` is `k * MIRROR_STRIDE`.
pub fn mirror_bytes(len: usize) -> Vec<u8> {
    (0..len.div_ceil(4) as u32).flat_map(|k| (k * MIRROR_STRIDE).to_le_bytes()).take(len).collect()
}

/// One heap slot a shader uses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeapSlot {
    /// Its index in the heap, with the mirror filled by [`mirror_bytes`].
    pub index: u32,
    /// `Sampler` for the sampler heap, else the class of the view.
    pub class: Class,
    pub shape: Shape,
}

/// A value the evaluator knows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Val {
    Int(i64),
    /// A handle of the mirror (`b1` of space 15).
    MirrorHandle,
    /// The result of a constant buffer load from the mirror at a known row.
    MirrorRow(i64),
    /// A heap handle with its index, `None` when the index is not known.
    Heap { index: Option<i64>, sampler: bool },
}

/// The class and shape of `ResourceProperties { p0, p1 }`.
pub fn properties(p0: u32, p1: u32) -> (Class, Shape) {
    let kind = p0 & 0xFF;
    let uav = p0 & (1 << 12) != 0;
    let class = if kind == 13 {
        Class::Cbv
    } else if kind == 14 {
        Class::Sampler
    } else if uav {
        Class::Uav
    } else {
        Class::Srv
    };
    // Component types of DXIL: 4 and 5 are 32 bit signed and unsigned integers, 2 and 3 their
    // 16 bit forms, 6 and 7 the 64 bit ones. Everything else is read as float.
    let ret = match p1 & 0xFF {
        2 | 4 | 6 => Ret::Sint,
        3 | 5 | 7 => Ret::Uint,
        _ => Ret::Float,
    };
    let shape = match kind {
        1 => Shape::Texture1D(ret),
        2 => Shape::Texture2D(ret),
        3 | 8 => Shape::Texture2DMS(ret),
        4 => Shape::Texture3D(ret),
        5 => Shape::TextureCube(ret),
        6 => Shape::Texture1DArray(ret),
        7 => Shape::Texture2DArray(ret),
        9 => Shape::TextureCubeArray(ret),
        10 => Shape::TypedBuffer(ret),
        11 => Shape::RawBuffer,
        12 => Shape::StructuredBuffer { stride: p1 },
        13 => Shape::ConstantBuffer,
        14 => Shape::Sampler { comparison: p0 & (1 << 15) != 0 },
        other => Shape::Other(other),
    };
    (class, shape)
}

fn operand(text: &str, vals: &HashMap<String, Val>) -> Option<Val> {
    let t = text.trim();
    let t = t.rsplit(' ').next().unwrap_or(t);
    if let Some(name) = t.strip_prefix('%') {
        return vals.get(name).copied();
    }
    t.parse::<i64>().ok().map(Val::Int)
}

fn ints_after<'a>(text: &'a str, key: &str) -> Vec<&'a str> {
    let Some(at) = text.find(key) else { return Vec::new() };
    let rest = &text[at + key.len()..];
    let end = rest.find(')').unwrap_or(rest.len());
    rest[..end].split(',').map(str::trim).collect()
}

fn braced(text: &str, key: &str) -> Vec<u32> {
    let Some(at) = text.find(key) else { return Vec::new() };
    let rest = &text[at + key.len()..];
    let Some(open) = rest.find('{') else { return Vec::new() };
    let close = rest.find('}').unwrap_or(rest.len());
    rest[open + 1..close].split(',').filter_map(|w| w.split_whitespace().last()?.parse().ok()).collect()
}

/// The heap slots of a disassembled DXIL module. Slots whose index is not known are left out.
pub fn slots(disassembly: &str) -> Vec<HeapSlot> {
    let mut vals: HashMap<String, Val> = HashMap::new();
    let mut out: Vec<HeapSlot> = Vec::new();
    for raw in disassembly.lines() {
        let line = raw.split(';').next().unwrap_or("").trim();
        let Some((lhs, rhs)) = line.split_once(" = ") else { continue };
        let Some(name) = lhs.trim().strip_prefix('%') else { continue };
        let rhs = rhs.trim();
        let val = if rhs.contains("@dx.op.createHandleFromBinding") {
            // ResBind { lower, upper, space, class }: class 2 is a CBV.
            let b = braced(rhs, "%dx.types.ResBind");
            (b.len() == 4 && b[0] == 1 && b[2] == crate::plan::MIRROR_SPACE && b[3] == 2).then_some(Val::MirrorHandle)
        } else if rhs.contains("@dx.op.createHandleFromHeap") {
            let args = ints_after(rhs, "@dx.op.createHandleFromHeap(");
            let index = args.get(1).and_then(|a| match operand(a, &vals) {
                Some(Val::Int(i)) => Some(i),
                _ => None,
            });
            let sampler = args.get(2).is_some_and(|a| a.ends_with("true"));
            Some(Val::Heap { index, sampler })
        } else if rhs.contains("@dx.op.annotateHandle") {
            let args = ints_after(rhs, "@dx.op.annotateHandle(");
            let handle = args.get(1).and_then(|a| operand(a, &vals));
            let p = braced(rhs, "%dx.types.ResourceProperties");
            match handle {
                Some(Val::Heap { index: Some(i), sampler }) if p.len() == 2 && (0..=u32::MAX as i64).contains(&i) => {
                    let (class, shape) = properties(p[0], p[1]);
                    let class = if sampler { Class::Sampler } else { class };
                    let slot = HeapSlot { index: i as u32, class, shape };
                    if !out.contains(&slot) {
                        out.push(slot);
                    }
                    None
                }
                Some(Val::MirrorHandle) => Some(Val::MirrorHandle),
                _ => None,
            }
        } else if rhs.contains("@dx.op.cbufferLoadLegacy") {
            let args = ints_after(rhs, "@dx.op.cbufferLoadLegacy");
            let handle = args.iter().find(|a| a.contains("%dx.types.Handle")).and_then(|a| operand(a, &vals));
            let row = args.last().and_then(|a| match operand(a, &vals) {
                Some(Val::Int(r)) => Some(r),
                _ => None,
            });
            match (handle, row) {
                (Some(Val::MirrorHandle), Some(r)) => Some(Val::MirrorRow(r)),
                _ => None,
            }
        } else if let Some(rest) = rhs.strip_prefix("extractvalue ") {
            let mut parts = rest.split(',');
            let src = parts.next().and_then(|a| operand(a, &vals));
            let component = parts.next().and_then(|c| c.trim().parse::<i64>().ok());
            match (src, component) {
                (Some(Val::MirrorRow(r)), Some(c)) => Some(Val::Int((4 * r + c) * i64::from(MIRROR_STRIDE))),
                _ => None,
            }
        } else if let Some(rest) = rhs.strip_prefix("bitcast ") {
            // A float bitcast keeps the bits: only an integer known as such passes.
            rest.split(" to ").next().and_then(|a| operand(a, &vals)).filter(|v| matches!(v, Val::Int(_)))
        } else {
            let mut words = rhs.split_whitespace();
            let op = words.next().unwrap_or("");
            let args: Vec<&str> = rhs.split_once(" i32 ").map(|(_, a)| a.split(',').collect()).unwrap_or_default();
            if args.len() == 2 {
                let (a, b) = (operand(args[0], &vals), operand(args[1], &vals));
                match (a, b) {
                    (Some(Val::Int(x)), Some(Val::Int(y))) => match op {
                        "add" => Some(Val::Int(x.wrapping_add(y))),
                        "sub" => Some(Val::Int(x.wrapping_sub(y))),
                        "mul" => Some(Val::Int(x.wrapping_mul(y))),
                        "shl" => Some(Val::Int(x.wrapping_shl(y as u32))),
                        "lshr" | "ashr" => Some(Val::Int(x.wrapping_shr(y as u32))),
                        "and" => Some(Val::Int(x & y)),
                        "or" => Some(Val::Int(x | y)),
                        "xor" => Some(Val::Int(x ^ y)),
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            }
        };
        if let Some(v) = val {
            vals.insert(name.to_string(), v);
        }
    }
    out.sort_by_key(|s| (s.class == Class::Sampler, s.index));
    out
}

/// Rewrites every `createHandleFromBinding` whose constant index lies outside its own range
/// `[lower, upper]` to index `lower + index`, and returns the text and how many it changed.
///
/// The index of `createHandleFromBinding` is the absolute register. The DXIL writer of the old
/// path passes the index inside the range (0) instead: hardware drivers tolerate it, WARP takes
/// the register below the range and removes the device. Rewritten, such a container runs on WARP.
pub fn fix_binding_indices(disassembly: &str) -> (String, usize) {
    let mut fixed = 0;
    let mut out = String::with_capacity(disassembly.len());
    for line in disassembly.split_inclusive('\n') {
        let key = "@dx.op.createHandleFromBinding(";
        let (Some(at), Some(close)) = (line.find(key), line.find("}, i32 ")) else {
            out.push_str(line);
            continue;
        };
        let b = braced(&line[at..], "%dx.types.ResBind");
        let rest = &line[close + 7..];
        let end = rest.find(',').unwrap_or(rest.len());
        match (b.as_slice(), rest[..end].trim().parse::<u32>()) {
            ([lower, upper, _, _], Ok(index)) if index < *lower || index > *upper => {
                out.push_str(&line[..close + 7]);
                out.push_str(&(lower + index).to_string());
                out.push_str(&rest[end..]);
                fixed += 1;
            }
            _ => out.push_str(line),
        }
    }
    (out, fixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODULE: &str = r#"
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 1, i32 1, i32 15, i8 2 }, i32 1, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 13, i32 128 })  ; AnnotateHandle(res,props)  resource: CBuffer
  %8 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %2, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %9 = extractvalue %dx.types.CBufRet.i32 %8, 1
  %10 = add i32 %9, 3
  %11 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %10, i1 false, i1 false)  ; CreateHandleFromHeap(index,samplerHeap,nonUniformIndex)
  %12 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %11, %dx.types.ResourceProperties { i32 2, i32 1033 })  ; AnnotateHandle(res,props)  resource: Texture2D<4xF32>
  %13 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 0, i1 true, i1 false)  ; CreateHandleFromHeap(index,samplerHeap,nonUniformIndex)
  %14 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %13, %dx.types.ResourceProperties { i32 14, i32 0 })  ; AnnotateHandle(res,props)  resource: SamplerState
  %15 = add i32 %9, 5
  %16 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %15, i1 false, i1 false)  ; CreateHandleFromHeap(index,samplerHeap,nonUniformIndex)
  %17 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %16, %dx.types.ResourceProperties { i32 13, i32 65536 })  ; AnnotateHandle(res,props)  resource: CBuffer
  %18 = extractvalue %dx.types.CBufRet.i32 %8, 0
  %19 = shl i32 %18, 2
  %20 = add nuw i32 %19, 7
  %21 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %20, i1 false, i1 false)  ; CreateHandleFromHeap(index,samplerHeap,nonUniformIndex)
  %22 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %21, %dx.types.ResourceProperties { i32 4107, i32 0 })  ; AnnotateHandle(res,props)  resource: RWByteAddressBuffer
  %23 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %17, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %24 = extractvalue %dx.types.CBufRet.i32 %23, 0
  %25 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 %24, i1 false, i1 false)  ; CreateHandleFromHeap(index,samplerHeap,nonUniformIndex)
  %26 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %25, %dx.types.ResourceProperties { i32 11, i32 0 })  ; AnnotateHandle(res,props)  resource: ByteAddressBuffer
"#;

    #[test]
    fn slots_indexed_by_the_mirror_plus_a_constant_are_found_with_their_kind() {
        let s = slots(MODULE);
        assert_eq!(
            s,
            vec![
                // Row 2, component 1 is mirror dword 9: 9 * 64 + 3 and + 5.
                HeapSlot { index: 579, class: Class::Srv, shape: Shape::Texture2D(Ret::Float) },
                HeapSlot { index: 581, class: Class::Cbv, shape: Shape::ConstantBuffer },
                // Row 2, component 0 is dword 8: (8 * 64) << 2 + 7.
                HeapSlot { index: 2055, class: Class::Uav, shape: Shape::RawBuffer },
                HeapSlot { index: 0, class: Class::Sampler, shape: Shape::Sampler { comparison: false } },
            ]
        );
    }

    #[test]
    fn binding_indices_below_their_range_are_rewritten_and_others_kept() {
        let text = "  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 27, i32 27, i32 1, i8 2 }, i32 0, i1 false)  ; x\n  %2 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 1, i32 1, i32 15, i8 2 }, i32 1, i1 false)\n";
        let (fixed, n) = fix_binding_indices(text);
        assert_eq!(n, 1);
        assert!(fixed.contains("i8 2 }, i32 27, i1 false)  ; x\n"), "{fixed}");
        assert!(fixed.contains("i32 15, i8 2 }, i32 1, i1 false)\n"));
    }

    #[test]
    fn resource_properties_decode_kind_uav_bit_and_types() {
        assert_eq!(properties(2, 1033), (Class::Srv, Shape::Texture2D(Ret::Float)));
        assert_eq!(properties(4098, 0x405), (Class::Uav, Shape::Texture2D(Ret::Uint)));
        assert_eq!(properties(4108, 16), (Class::Uav, Shape::StructuredBuffer { stride: 16 }));
        assert_eq!(properties(13, 128), (Class::Cbv, Shape::ConstantBuffer));
        assert_eq!(properties(11, 0), (Class::Srv, Shape::RawBuffer));
    }
}
