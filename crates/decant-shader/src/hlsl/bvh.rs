//! Node intersections (`image_bvh_intersect_ray`, `image_bvh64_intersect_ray`) as HLSL.
//!
//! The instruction tests a ray against one node of a bounding volume hierarchy and writes four
//! result words. The node memory is the buffer the resource pass made of the descriptor
//! ([`crate::resource::node_memory`]). It is read as a `ByteAddressBuffer`.
//!
//! The node layout, the node pointer and the result words follow the public RDNA2 layout in Mesa
//! `src/amd/vulkan/bvh/bvh_defines.h` and the software intersection in
//! `src/amd/vulkan/nir/radv_nir_rt_common.c` (MIT), both at revision `f8281cfa33e2`. The helper
//! text is written here from those facts.
//!
//! * The node pointer counts 8 byte units. Its low three bits are the node type, so the byte
//!   offset of the node is `(pointer & ~7) << 3`. Types 0 to 3 are triangles, 4 a box of half
//!   floats, 5 a box of floats.
//! * A box node is four child pointers and then four boxes (six floats each, or six half floats
//!   each). The result is the child pointers of the boxes the ray enters before its extent,
//!   nearest first, then `0xffffffff` for every box it misses. A box whose minimum x is not a
//!   number is inactive.
//! * A triangle node starts with its three vertices (nine floats). The result is the distance
//!   numerator, the denominator (its sign is the facing), and the numerators of the second and
//!   third barycentric coordinates. A miss is (infinity, 1, 0, 0). This is the result mode Mesa
//!   selects in its descriptor ("return IJ").
//! * The address operands are the node pointer (two words for the 64 bit form), the extent, the
//!   origin, the direction and the inverse direction. With A16 the direction and the inverse
//!   direction are six half floats in three words, from the operand packing of the LLVM AMDGPU
//!   backend (`SIISelLowering.cpp`).
//!
//! Where the measurement stops: the helper follows the software path of Mesa, not a measurement
//! of the console hardware. Float rounding, the edge rules of the triangle test and the order of
//! two boxes at the same distance were not compared with hardware results. The descriptor flags
//! (box sorting, the triangle result mode) are not read, the helper always sorts and always
//! returns barycentric numerators. Triangle types 1 to 3 read the same single triangle as type 0.
//! The public sources define no hardware result for instance (6) and procedural (7) nodes, and
//! the helper answers them with four `0xffffffff` words. A node more than 4 GiB past the start of
//! the node memory reads as a miss of its type.

use std::fmt::Write as _;

use super::Printer;
use crate::ir::{BvhIntersect, Expr, Function, Guard, Helper, Node, Stmt, Ty, Var};
use crate::refusal::{Refusal, RefusalKind};

/// The box test: four children, sorted by the distance at which the ray enters their box.
static BOX: Helper = Helper {
    name: "decant_bvh_box",
    result: Ty::U32,
    text: r#"void decant_bvh_order(inout float4 d, inout uint4 c, uint a, uint b) {
    if (d[b] < d[a]) {
        float t = d[a]; d[a] = d[b]; d[b] = t;
        uint n = c[a]; c[a] = c[b]; c[b] = n;
    }
}
uint4 decant_bvh_box(ByteAddressBuffer nodes, uint at, bool half_box, float extent, float3 origin, float3 inv_dir) {
    uint4 children = nodes.Load4(at);
    float3 inv = clamp(inv_dir, -3.402823466e+38f, 3.402823466e+38f);
    float4 dist = (float4)asfloat(0x7f800000u);
    uint4 hit = (uint4)0xffffffffu;
    [unroll] for (uint i = 0u; i < 4u; i++) {
        float3 lo;
        float3 hi;
        if (half_box) {
            uint3 w = nodes.Load3(at + 16u + 12u * i);
            lo = f16tof32(uint3(w.x, w.x >> 16u, w.y));
            hi = f16tof32(uint3(w.y >> 16u, w.z, w.z >> 16u));
        } else {
            lo = asfloat(nodes.Load3(at + 16u + 24u * i));
            hi = asfloat(nodes.Load3(at + 28u + 24u * i));
        }
        float3 b0 = (lo - origin) * inv;
        float3 b1 = (hi - origin) * inv;
        float3 enter = min(b0, b1);
        float3 leave = max(b0, b1);
        float tmin = max(max(enter.x, enter.y), enter.z);
        float tmax = min(min(leave.x, leave.y), leave.z);
        if (!isnan(lo.x) && tmax >= max(0.0f, tmin) && tmin < extent) {
            hit[i] = children[i];
            dist[i] = tmin;
        }
    }
    decant_bvh_order(dist, hit, 0u, 1u);
    decant_bvh_order(dist, hit, 2u, 3u);
    decant_bvh_order(dist, hit, 0u, 2u);
    decant_bvh_order(dist, hit, 1u, 3u);
    decant_bvh_order(dist, hit, 1u, 2u);
    return hit;
}"#,
    needs: &[],
};

/// The watertight triangle test, with the edge and vertex rules of the software path.
static TRIANGLE: Helper = Helper {
    name: "decant_bvh_triangle",
    result: Ty::U32,
    text: r#"bool decant_bvh_edge(float v0x, float v0y, float v1x, float v1y) {
    float tx = v1x - v0x;
    return tx == 0.0f ? (v1y - v0y) < 0.0f : tx < 0.0f;
}
bool decant_bvh_corner(float v0x, float v1x, float v2x) {
    return (v1x - v0x) <= 0.0f && (v2x - v0x) > 0.0f;
}
uint4 decant_bvh_triangle(ByteAddressBuffer nodes, uint at, float3 origin, float3 dir, float3 inv_dir) {
    float3 a = asfloat(nodes.Load3(at)) - origin;
    float3 b = asfloat(nodes.Load3(at + 12u)) - origin;
    float3 c = asfloat(nodes.Load3(at + 24u)) - origin;
    float3 m = abs(dir);
    uint3 k = uint3(0u, 1u, 2u);
    if (m.x >= m.y) {
        if (m.x >= m.z) {
            k = uint3(1u, 2u, 0u);
        }
    } else if (m.y >= m.z) {
        k = uint3(2u, 0u, 1u);
    }
    if (dir[k.z] < 0.0f) {
        k = k.yxz;
    }
    float sz = inv_dir[k.z];
    precise float sx = dir[k.x] * sz;
    precise float sy = dir[k.y] * sz;
    precise float ax = a[k.x] - sx * a[k.z];
    precise float ay = a[k.y] - sy * a[k.z];
    precise float bx = b[k.x] - sx * b[k.z];
    precise float by = b[k.y] - sy * b[k.z];
    precise float cx = c[k.x] - sx * c[k.z];
    precise float cy = c[k.y] - sy * c[k.z];
    precise float u = cx * by - cy * bx;
    precise float v = ax * cy - ay * cx;
    precise float w = bx * ay - by * ax;
    bool back = u < 0.0f || v < 0.0f || w < 0.0f;
    bool front = u > 0.0f || v > 0.0f || w > 0.0f;
    bool hit = !(back && front);
    bool ea = u == 0.0f;
    bool eb = v == 0.0f;
    bool ec = w == 0.0f;
    if (ea || eb || ec) {
        bool edge = (ea && decant_bvh_edge(bx, by, cx, cy)) || (eb && decant_bvh_edge(cx, cy, ax, ay))
            || (ec && decant_bvh_edge(ax, ay, bx, by));
        bool va = eb && ec;
        bool vb = ea && ec;
        bool vc = ea && eb;
        bool corner = (va && decant_bvh_corner(ax, bx, cx)) || (vb && decant_bvh_corner(bx, cx, ax))
            || (vc && decant_bvh_corner(cx, ax, bx));
        hit = hit && edge && (!(va || vb || vc) || corner);
    }
    uint4 result = uint4(0x7f800000u, 0x3f800000u, 0u, 0u);
    if (hit) {
        precise float det = u + (v + w);
        precise float t = (u * (sz * a[k.z]) + v * (sz * b[k.z])) + w * (sz * c[k.z]);
        float s = float(sign(det));
        if (!(s * t < 0.0f)) {
            float d = abs(det);
            result = uint4(asuint(t / d), asuint(s), asuint(v / d), asuint(w / d));
        }
    }
    return result;
}"#,
    needs: &[],
};

/// One node intersection: the node type picks the test. It returns the four result words as a
/// `uint4`. The printer calls it in a statement of its own, never through `Expr::Call`.
pub(super) static INTERSECT: Helper = Helper {
    name: "decant_bvh_intersect",
    result: Ty::U32,
    text: r#"uint4 decant_bvh_intersect(ByteAddressBuffer nodes, uint node_lo, uint node_hi, uint extent, float3 origin, float3 dir, float3 inv_dir) {
    uint kind = node_lo & 7u;
    bool reachable = ((node_hi << 3u) | (node_lo >> 29u)) == 0u;
    uint at = (node_lo & 0xfffffff8u) << 3u;
    if (kind < 4u) {
        return reachable ? decant_bvh_triangle(nodes, at, origin, dir, inv_dir) : uint4(0x7f800000u, 0x3f800000u, 0u, 0u);
    }
    if (reachable && kind <= 5u) {
        return decant_bvh_box(nodes, at, kind == 4u, asfloat(extent), origin, inv_dir);
    }
    return (uint4)0xffffffffu;
}"#,
    needs: &[&BOX, &TRIANGLE],
};

fn refuse(what: &'static str, detail: &str, pc: u32) -> Refusal {
    Refusal::new(RefusalKind::Resource(what), detail).at(pc)
}

/// Checks that every node intersection is the whole value of an assignment and has its node
/// memory bound. The printer writes one call per intersection in a line of its own.
pub(super) fn check(func: &Function) -> Result<(), Refusal> {
    let mut all = Vec::new();
    super::expr::for_each_expr(func, &mut |e| {
        if let Expr::BvhWord(op, _) = e {
            all.push(op.pc);
        }
    });
    let mut whole = 0usize;
    func.for_each_stmt(&mut |s| {
        if let Stmt::Set {
            value: Expr::BvhWord(..),
            ..
        } = s
        {
            whole += 1;
        }
    });
    if let Some(pc) = all.first().copied()
        && all.len() != whole
    {
        return Err(refuse(
            "bvh-in-expression",
            "a node intersection inside an expression",
            pc,
        ));
    }
    let mut unbound = None;
    super::expr::for_each_expr(func, &mut |e| {
        if let Expr::BvhWord(op, _) = e
            && crate::resource::node_memory(op).is_none()
        {
            unbound.get_or_insert(op.pc);
        }
    });
    match unbound {
        Some(pc) => Err(refuse("bvh-descriptor", "the node memory is not bound", pc)),
        None => Ok(()),
    }
}

/// Whether a node intersection takes its node memory by a heap index in a register: the
/// assignment then declares the buffer in a line of its own, like a buffer word read.
pub(super) fn by_index(func: &Function, e: &Expr) -> bool {
    matches!(e, Expr::BvhWord(op, _) if crate::resource::node_memory(op).is_some_and(|res| matches!(
        func.resources[res.0 as usize].placement,
        Some(crate::api::Placement::HeapAtConsoleIndex { .. })
    )))
}

/// Whether the function has a node intersection, so the printer needs [`INTERSECT`].
pub(super) fn used(nodes: &[Node]) -> bool {
    nodes.iter().any(|n| match n {
        Node::Block(stmts) => stmts.iter().any(|s| {
            matches!(
                s,
                Stmt::Set {
                    value: Expr::BvhWord(..),
                    ..
                }
            )
        }),
        Node::If { then, els, .. } => used(then) || used(els),
        Node::Loop(body) | Node::Once(body) => used(body),
        Node::Break | Node::Continue | Node::Return => false,
    })
}

/// The ray of an intersection as the three `float3` arguments of [`INTERSECT`]: origin,
/// direction and inverse direction. `ray` is the address operands after the node pointer and
/// the extent.
fn ray(p: &Printer, ray: &[Expr], a16: bool) -> [String; 3] {
    let e = |i: usize| p.e(&ray[i]);
    let origin = format!("asfloat(uint3({}, {}, {}))", e(0), e(1), e(2));
    if a16 {
        let (xy, zx, yz) = (e(3), e(4), e(5));
        [
            origin,
            format!("f16tof32(uint3({xy}, {xy} >> 16u, {zx}))"),
            format!("f16tof32(uint3({zx} >> 16u, {yz}, {yz} >> 16u))"),
        ]
    } else {
        [
            origin,
            format!("asfloat(uint3({}, {}, {}))", e(3), e(4), e(5)),
            format!("asfloat(uint3({}, {}, {}))", e(6), e(7), e(8)),
        ]
    }
}

/// Writes the call of [`INTERSECT`] for `op` into a new local and returns its name.
fn intersect(p: &mut Printer, op: &BvhIntersect) -> Result<String, Refusal> {
    let res = crate::resource::node_memory(op)
        .ok_or_else(|| refuse("bvh-descriptor", "the node memory is not bound", op.pc))?;
    let func = p.func;
    let r = &func.resources[res.0 as usize];
    let ty = r
        .placement
        .as_ref()
        .map(super::class_of)
        .and_then(|class| super::resource_type(func, r, class));
    if ty.as_deref() != Some("ByteAddressBuffer") {
        return Err(refuse(
            "bvh-node-view",
            "the node memory is no read only raw buffer view",
            op.pc,
        ));
    }
    let pointer = usize::from(op.wide) + 1;
    let words = pointer + 4 + if op.encoding.a16 { 3 } else { 6 };
    if op.address.len() != words {
        return Err(refuse(
            "bvh-address",
            "the address operands do not match the encoding",
            op.pc,
        ));
    }
    let nodes = p.use_resource(res, "ByteAddressBuffer");
    let lo = p.e(&op.address[0]);
    let hi = if op.wide {
        p.e(&op.address[1])
    } else {
        "0u".to_owned()
    };
    let extent = p.e(&op.address[pointer]);
    let [origin, dir, inv_dir] = ray(p, &op.address[pointer + 1..], op.encoding.a16);
    p.unique += 1;
    let local = format!("decant_bvh{}", p.unique);
    p.line(&format!(
        "uint4 {local} = decant_bvh_intersect({nodes}, {lo}, {hi}, {extent}, {origin}, {dir}, {inv_dir});"
    ));
    Ok(local)
}

/// Prints `dst = word` of a node intersection. Consecutive assignments of the words of one
/// intersection share one call.
pub(super) fn assign(
    p: &mut Printer,
    guard: Guard,
    ty: &str,
    dst: Var,
    op: &BvhIntersect,
    word: u8,
) -> Result<(), Refusal> {
    let local = match &p.bvh {
        Some((last, local)) if last == op => local.clone(),
        _ => {
            let local = intersect(p, op)?;
            p.bvh = Some((op.clone(), local.clone()));
            local
        }
    };
    p.indent();
    if guard == Guard::Exec {
        p.out.push_str("if (exec) ");
    }
    let _ = writeln!(
        p.out,
        "{ty}{} = {local}.{};",
        super::expr::var_name(dst),
        ["x", "y", "z", "w"][usize::from(word & 3)]
    );
    Ok(())
}
