//! The test shaders of the specification, section 5.2, as IR.

use decant_dxil::ir::*;

use super::br;

fn v(n: u32) -> Operand {
    Operand::value(n)
}
fn c(n: u32) -> Operand {
    Operand::i32(n)
}
fn uav_raw(register: u32) -> Resource {
    Resource::raw(ResourceClass::Uav, 0, register)
}

/// Test 1: an empty compute shader.
pub fn empty() -> Shader {
    Shader::compute([1, 1, 1], vec![Block::new(vec![], Terminator::Ret)])
}

/// Test 2: `u0[tid] = tid` (a raw UAV, 64 threads).
pub fn thread_id_store() -> Shader {
    thread_id_store_at(0)
}

/// Test 2 with the raw UAV at register `register`.
pub fn thread_id_store_at(register: u32) -> Shader {
    let insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[1], Op::Bin(BinOp::Mul, v(0), c(4))),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(0), index: v(1), offset: None, values: vec![v(0)] }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(uav_raw(register));
    s
}

/// Test 3: copies four dwords per thread from the raw SRV t0 to the raw UAV u0.
pub fn buffer_copy() -> Shader {
    let insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[1], Op::Bin(BinOp::Shl, v(0), c(4))),
        Inst::new(&[2, 3, 4, 5], Op::BufLoad { res: ResRef::bound(0), index: v(1), offset: None, count: 4, ty: Ty::I32 }),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: v(1), offset: None, values: vec![v(2), v(3), v(4), v(5)] }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(Resource::raw(ResourceClass::Srv, 0, 0));
    s.resources.push(uav_raw(0));
    s
}

/// The 32 byte constant buffer of the arithmetic test: row 0 is `a`, `b` (u32), row 1 is `x`, `y`, `z` (f32).
pub struct ArithInput {
    pub a: u32,
    pub b: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl ArithInput {
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = Vec::new();
        for w in [self.a, self.b, 0, 0, self.x.to_bits(), self.y.to_bits(), self.z.to_bits(), 0] {
            v.extend_from_slice(&w.to_le_bytes());
        }
        v
    }
}

/// How many dwords the arithmetic shader writes.
pub const ARITH_OUTPUTS: usize = 17;

/// The expected output of the arithmetic shader, written from the definitions of the operations.
pub fn arith_reference(i: &ArithInput) -> Vec<u32> {
    let (a, b) = (i.a, i.b);
    vec![
        a.wrapping_add(b),
        a.wrapping_mul(b),
        a / b,
        (i.x * i.y + i.z).to_bits(),
        i.x.min(i.y).to_bits(),
        a.max(b),
        a.count_ones(),
        if a == 0 { u32::MAX } else { a.leading_zeros() },
        (a >> 3) & 0x1F,
        if a < b { a } else { b },
        i.x as i32 as u32,
        a.wrapping_sub(b),
        a << (b & 31),
        a ^ b,
        ((a as i32).wrapping_div(b as i32)) as u32,
        a.overflowing_add(b).1 as u32,
        ((a as i32) >> (b & 31)) as u32,
    ]
}

/// Test 4: arithmetic on `a`, `b`, `x`, `y`, `z` from a constant buffer, results to a raw UAV.
pub fn arithmetic() -> Shader {
    let mut insts = vec![
        Inst::new(&[0, 1], Op::CbufLoad { res: ResRef::bound(0), row: c(0), comps: vec![0, 1], ty: Ty::I32 }),
        Inst::new(&[2, 3, 4], Op::CbufLoad { res: ResRef::bound(0), row: c(1), comps: vec![0, 1, 2], ty: Ty::F32 }),
    ];
    let (a, b, x, y, z) = (v(0), v(1), v(2), v(3), v(4));
    let mut next = 5;
    let mut out: Vec<(Op, Ty)> = vec![
        (Op::Bin(BinOp::Add, a, b), Ty::I32),
        (Op::Bin(BinOp::Mul, a, b), Ty::I32),
        (Op::Bin(BinOp::UDiv, a, b), Ty::I32),
        (Op::Mad(MadOp::FMad, x, y, z), Ty::F32),
        (Op::MinMax(MinMaxOp::FMin, x, y), Ty::F32),
        (Op::MinMax(MinMaxOp::UMax, a, b), Ty::I32),
        (Op::Un(UnOp::CountBits, a), Ty::I32),
        (Op::Un(UnOp::FirstBitHigh, a), Ty::I32),
        (Op::BitfieldExtract { signed: false, width: c(5), offset: c(3), value: a }, Ty::I32),
    ];
    // select(a < b, a, b) needs the compare first.
    let cmp = next;
    next += 1;
    insts.push(Inst::new(&[cmp], Op::ICmp(IPred::Ult, a, b)));
    out.push((Op::Select(v(cmp), a, b), Ty::I32));
    out.push((Op::Cast(CastOp::FpToSi, x, Ty::I32), Ty::I32));
    out.push((Op::Bin(BinOp::Sub, a, b), Ty::I32));
    out.push((Op::Bin(BinOp::Shl, a, b), Ty::I32));
    out.push((Op::Bin(BinOp::Xor, a, b), Ty::I32));
    out.push((Op::Bin(BinOp::SDiv, a, b), Ty::I32));
    for (i, (op, _)) in out.iter().enumerate() {
        let id = next;
        next += 1;
        insts.push(Inst::new(&[id], op.clone()));
        insts.push(Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: c(4 * i as u32), offset: None, values: vec![v(id)] }));
    }
    // The carry of a + b, widened to 0 or 1, and an arithmetic shift.
    let (sum, carry, wide) = (next, next + 1, next + 2);
    insts.push(Inst::new(&[sum, carry], Op::AddCarry(a, b)));
    insts.push(Inst::new(&[wide], Op::Cast(CastOp::ZExt, v(carry), Ty::I32)));
    insts.push(Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: c(4 * 15), offset: None, values: vec![v(wide)] }));
    let sh = next + 3;
    insts.push(Inst::new(&[sh], Op::Bin(BinOp::AShr, a, b)));
    insts.push(Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: c(4 * 16), offset: None, values: vec![v(sh)] }));
    let mut s = Shader::compute([1, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(Resource::cbuffer(0, 0, 32));
    s.resources.push(uav_raw(0));
    s
}

/// Test 5: the loop with a branch. `n` comes from a constant buffer, the sum goes to `u0[0]`.
///
/// ```text
/// sum = 0; i = 0;
/// if n != 0 { do { if i & 1 { sum += i } else { sum -= 1 }; i += 1 } while i != n }
/// u0[0] = sum
/// ```
pub fn loop_with_branch() -> Shader {
    let phi = |dst, incoming| Phi { dst: ValueId(dst), ty: Ty::I32, incoming };
    let b = BlockId;
    let zero = c(0);
    let entry = Block::new(
        vec![
            Inst::new(&[0], Op::CbufLoad { res: ResRef::bound(0), row: c(0), comps: vec![0], ty: Ty::I32 }),
            Inst::new(&[1], Op::ICmp(IPred::Eq, v(0), zero)),
        ],
        Terminator::CondBr(v(1), b(4), b(1)),
    );
    let header = Block {
        phis: vec![phi(2, vec![(b(0), zero), (b(3), v(9))]), phi(3, vec![(b(0), zero), (b(3), v(8))])],
        insts: vec![Inst::new(&[4], Op::Bin(BinOp::And, v(2), c(1))), Inst::new(&[5], Op::ICmp(IPred::Ne, v(4), zero))],
        term: Terminator::CondBr(v(5), b(2), b(5)),
    };
    let then = Block::new(vec![Inst::new(&[6], Op::Bin(BinOp::Add, v(3), v(2)))], br(3));
    let latch = Block {
        phis: vec![phi(8, vec![(b(2), v(6)), (b(5), v(7))])],
        insts: vec![Inst::new(&[9], Op::Bin(BinOp::Add, v(2), c(1))), Inst::new(&[10], Op::ICmp(IPred::Eq, v(9), v(0)))],
        term: Terminator::CondBr(v(10), b(4), b(1)),
    };
    let exit = Block {
        phis: vec![phi(11, vec![(b(0), zero), (b(3), v(8))])],
        insts: vec![Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: zero, offset: None, values: vec![v(11)] })],
        term: Terminator::Ret,
    };
    let els = Block::new(vec![Inst::new(&[7], Op::Bin(BinOp::Sub, v(3), c(1)))], br(3));
    let mut s = Shader::compute([1, 1, 1], vec![entry, header, then, latch, exit, els]);
    s.resources.push(Resource::cbuffer(0, 0, 16));
    s.resources.push(uav_raw(0));
    s
}

/// The reference result of [`loop_with_branch`].
pub fn loop_reference(n: u32) -> u32 {
    let mut sum = 0u32;
    if n != 0 {
        let mut i = 0u32;
        loop {
            if i & 1 != 0 {
                sum = sum.wrapping_add(i);
            } else {
                sum = sum.wrapping_sub(1);
            }
            i += 1;
            if i == n {
                break;
            }
        }
    }
    sum
}

/// Test 6: groupshared and a barrier. `u0[tid] = 63 - tid % 64` through a `[64 x i32]` array.
pub fn groupshared_barrier() -> Shader {
    let insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::InGroup, 0)),
        Inst::new(&[1], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[], Op::GsStore { var: 0, index: v(0), value: v(0) }),
        Inst::new(&[], Op::Barrier { flags: barrier::SYNC_GROUP | barrier::GROUPSHARED_FENCE }),
        Inst::new(&[2], Op::Bin(BinOp::Sub, c(63), v(0))),
        Inst::new(&[3], Op::GsLoad { var: 0, index: v(2) }),
        Inst::new(&[4], Op::Bin(BinOp::Shl, v(1), c(2))),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(0), index: v(4), offset: None, values: vec![v(3)] }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.groupshared.push(Groupshared { len: 64, ty: Ty::I32 });
    s.resources.push(uav_raw(0));
    s
}

/// Test 7: atomics on a UAV and on groupshared memory, 64 threads per group.
///
/// With two groups the expected `u0` is `[128, 64, 2, 5]`: all threads, the groupshared count, one compare exchange winner per group, one buffer compare exchange winner.
pub fn atomics() -> Shader {
    let insts = vec![
        Inst::new(&[], Op::GsStore { var: 0, index: c(0), value: c(0) }),
        Inst::new(&[], Op::GsStore { var: 0, index: c(1), value: c(0) }),
        Inst::new(&[], Op::Barrier { flags: barrier::SYNC_GROUP | barrier::GROUPSHARED_FENCE }),
        Inst::new(&[0], Op::BufAtomic { op: AtomicOp::Add, res: ResRef::bound(0), index: c(0), offset: None, value: c(1) }),
        Inst::new(&[1], Op::GsAtomic { op: GsAtomicOp::Add, var: 0, index: c(0), value: c(1) }),
        Inst::new(&[2], Op::GsCmpXchg { var: 0, index: c(1), compare: c(0), new: c(1) }),
        Inst::new(&[3], Op::ICmp(IPred::Eq, v(2), c(0))),
        Inst::new(&[4], Op::Select(v(3), c(1), c(0))),
        Inst::new(&[5], Op::BufAtomic { op: AtomicOp::Add, res: ResRef::bound(0), index: c(8), offset: None, value: v(4) }),
        Inst::new(&[6], Op::BufCmpXchg { res: ResRef::bound(0), index: c(12), offset: None, compare: c(0), new: c(5) }),
        Inst::new(&[], Op::Barrier { flags: barrier::SYNC_GROUP | barrier::GROUPSHARED_FENCE }),
        Inst::new(&[7], Op::GsLoad { var: 0, index: c(0) }),
        Inst::new(&[8], Op::BufAtomic { op: AtomicOp::Xchg, res: ResRef::bound(0), index: c(4), offset: None, value: v(7) }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.groupshared.push(Groupshared { len: 2, ty: Ty::I32 });
    s.resources.push(uav_raw(0));
    s
}

/// Words written per thread by [`wave_ops`].
pub const WAVE_WORDS: usize = 8;

/// Test 8: wave operations in a group of 64 threads with `WaveSize` `size`. Per thread `t`, 8 words:
/// the wave sum of `t`, the ballot mask word 0 of `t` being even, the first lane's `t`, the lane index,
/// the lane count, whether this is the first lane, the exclusive prefix sum of `t`, the count of even `t`.
pub fn wave_ops(size: u32) -> Shader {
    let (t, base) = (v(0), v(1));
    let mut insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[1], Op::Bin(BinOp::Mul, t, c(4 * WAVE_WORDS as u32))),
        Inst::new(&[2], Op::WaveActive { op: WaveOp::Sum, signed: false, value: t }),
        Inst::new(&[3], Op::Bin(BinOp::And, t, c(1))),
        Inst::new(&[4], Op::ICmp(IPred::Eq, v(3), c(0))),
        Inst::new(&[5, 6, 7, 8], Op::WaveBallot(v(4))),
        Inst::new(&[9], Op::WaveReadLaneFirst(t)),
        Inst::new(&[10], Op::WaveLaneIndex),
        Inst::new(&[11], Op::WaveLaneCount),
        Inst::new(&[12], Op::WaveIsFirstLane),
        Inst::new(&[13], Op::Select(v(12), c(1), c(0))),
        Inst::new(&[14], Op::WavePrefix { op: WaveOp::Sum, signed: false, value: t }),
        Inst::new(&[15], Op::WaveActiveCountBits(v(4))),
    ];
    for (k, value) in [2, 5, 9, 10, 11, 13, 14, 15].into_iter().enumerate() {
        let at = 16 + k as u32;
        insts.push(Inst::new(&[at], Op::Bin(BinOp::Add, base, c(4 * k as u32))));
        insts.push(Inst::new(&[], Op::BufStore { res: ResRef::bound(0), index: v(at), offset: None, values: vec![v(value)] }));
    }
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.stage = Stage::Compute { threads: [64, 1, 1], wave_size: Some(size) };
    s.resources.push(uav_raw(0));
    s
}

/// The expected words of [`wave_ops`] for thread `t` with waves of `n` lanes (4 to 32, a power of two).
pub fn wave_reference(t: u32, n: u32) -> [u32; WAVE_WORDS] {
    let (w, l) = (t / n, t % n);
    let mask = if n == 32 { 0x5555_5555 } else { 0x5555_5555 & ((1u32 << n) - 1) };
    [n * n * w + n * (n - 1) / 2, mask, n * w, l, n, (l == 0) as u32, n * w * l + l * l.saturating_sub(1) / 2, n / 2]
}

/// Test 9: `ResourceDescriptorHeap[b0.x]` is a raw UAV, `u[tid] = tid + 100`, 64 threads.
pub fn heap_uav() -> Shader {
    let insts = vec![
        Inst::new(&[0], Op::CbufLoad { res: ResRef::bound(0), row: c(0), comps: vec![0], ty: Ty::I32 }),
        Inst::new(&[1], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[2], Op::Bin(BinOp::Mul, v(1), c(4))),
        Inst::new(&[3], Op::Bin(BinOp::Add, v(1), c(100))),
        Inst::new(&[], Op::BufStore { res: ResRef::heap(1, v(0), false), index: v(2), offset: None, values: vec![v(3)] }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(Resource::cbuffer(0, 0, 16));
    s.resources.push(uav_raw(0).in_heap());
    s
}

fn load(elem: u32, col: u8, dst: u32) -> Inst {
    Inst::new(&[dst], Op::LoadInput { elem, row: c(0), col })
}

fn store(elem: u32, col: u8, value: Operand) -> Inst {
    Inst::new(&[], Op::StoreOutput { elem, row: c(0), col, value })
}

fn position_in() -> SigElement {
    SigElement::new("POSITION", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Undefined, 0, 4)
}

fn position_out() -> SigElement {
    SigElement::new("SV_Position", 0, Semantic::Position, ComponentType::F32, Interp::LinearNoperspective, 0, 4)
}

fn target_out() -> SigElement {
    SigElement::new("SV_Target", 0, Semantic::Target, ComponentType::F32, Interp::Undefined, 0, 4)
}

/// Test 10: a vertex shader that passes `POSITION` through to `SV_Position`.
pub fn vertex_passthrough() -> Shader {
    let mut insts: Vec<Inst> = (0..4).map(|k| load(0, k, k as u32)).collect();
    insts.extend((0..4).map(|k| store(0, k, v(k as u32))));
    let mut s = Shader::vertex(vec![Block::new(insts, Terminator::Ret)]);
    s.inputs.push(position_in());
    s.outputs.push(position_out());
    s
}

/// [`vertex_passthrough`] that also stores `POSITION.x` to a raw UAV of the heap. A UAV of the heap is
/// not a bound UAV, so the "UAVs at every stage" flag stays clear (the validator checks it).
pub fn vertex_heap_uav() -> Shader {
    let mut s = vertex_passthrough();
    let cast = Inst::new(&[4], Op::Cast(CastOp::Bitcast, v(0), Ty::I32));
    let store = Inst::new(&[], Op::BufStore { res: ResRef::heap(0, c(7), false), index: c(0), offset: None, values: vec![v(4)] });
    s.blocks[0].insts.extend([cast, store]);
    s.resources.push(uav_raw(0).in_heap());
    s
}

/// A vertex shader with a second input and output, `TEXCOORD0` (float2), passed through.
pub fn vertex_with_uv() -> Shader {
    let mut insts: Vec<Inst> = (0..4).map(|k| load(0, k, k as u32)).collect();
    insts.extend((0..2).map(|k| load(1, k, 4 + k as u32)));
    insts.extend((0..4).map(|k| store(0, k, v(k as u32))));
    insts.extend((0..2).map(|k| store(1, k, v(4 + k as u32))));
    let mut s = Shader::vertex(vec![Block::new(insts, Terminator::Ret)]);
    s.inputs.push(position_in());
    s.inputs.push(SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Undefined, 1, 2));
    s.outputs.push(position_out());
    s.outputs.push(SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Linear, 1, 2));
    s
}

/// The colour of [`pixel_constant`].
pub const PIXEL_COLOR: [f32; 4] = [1.0, 0.5, 0.25, 1.0];

/// Test 11: a pixel shader that writes a constant colour to `SV_Target`.
pub fn pixel_constant() -> Shader {
    let insts: Vec<Inst> = (0..4).map(|k| store(0, k, Operand::f32(PIXEL_COLOR[k as usize]))).collect();
    let mut s = Shader::pixel(vec![Block::new(insts, Terminator::Ret)]);
    s.outputs.push(target_out());
    s
}

/// Test 12: a pixel shader that samples `Texture2D t0` with `SamplerState s0` at `TEXCOORD0`.
pub fn pixel_texture() -> Shader {
    let sample = Inst::new(
        &[2, 3, 4, 5],
        Op::Sample { tex: ResRef::bound(0), sampler: ResRef::bound(1), coords: vec![v(0), v(1)], offsets: vec![], mode: SampleMode::Implicit, clamp: None, ty: Ty::F32 },
    );
    let mut insts = vec![load(1, 0, 0), load(1, 1, 1), sample];
    insts.extend((0..4).map(|k| store(0, k, v(2 + k as u32))));
    let mut s = Shader::pixel(vec![Block::new(insts, Terminator::Ret)]);
    s.inputs.push(position_out());
    s.inputs.push(SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Linear, 1, 2));
    s.outputs.push(target_out());
    s.resources.push(Resource::texture(ResourceClass::Srv, ResourceKind::Texture2D, 0, 0, Element { component: ComponentType::F32, count: 4 }));
    s.resources.push(Resource::sampler(0, 0, false));
    s
}

/// A pixel shader with many kinds of input and output, written to produce the same signatures as this HLSL
/// compiled by DXC (`dxc -T ps_6_6 -validator-version 1.6`):
///
/// ```text
/// struct I { float4 pos : SV_Position; float2 uv : TEXCOORD0; nointerpolation uint id : ID; bool ff : SV_IsFrontFace;
///            uint prim : SV_PrimitiveID; uint si : SV_SampleIndex; centroid float4 col : COLOR0; };
/// struct O { float4 c0 : SV_Target0; float4 c1 : SV_Target1; float d : SV_Depth; uint cov : SV_Coverage; };
/// O main(I i) { O o; o.c0 = i.col + i.uv.x; o.c1 = float4(i.uv, i.id, i.prim + i.si);
///               if (i.ff) o.c1.x = 1; o.d = i.pos.z; o.cov = 3; return o; }
/// ```
pub fn pixel_zoo() -> Shader {
    let elem = |name: &str, semantic, comp, interp, row, col, cols| SigElement { start_col: col, ..SigElement::new(name, 0, semantic, comp, interp, row, cols) };
    let (f, u) = (ComponentType::F32, ComponentType::U32);
    let mut s = Shader::pixel(vec![]);
    s.inputs = vec![
        elem("SV_Position", Semantic::Position, f, Interp::LinearNoperspective, 0, 0, 4),
        elem("TEXCOORD", Semantic::Arbitrary, f, Interp::Linear, 1, 0, 2),
        elem("ID", Semantic::Arbitrary, u, Interp::Constant, 2, 0, 1),
        elem("SV_IsFrontFace", Semantic::IsFrontFace, u, Interp::Constant, 2, 1, 1),
        elem("SV_PrimitiveID", Semantic::PrimitiveId, u, Interp::Constant, 2, 2, 1),
        elem("SV_SampleIndex", Semantic::SampleIndex, u, Interp::Constant, SigElement::UNALLOCATED, 0, 1),
        elem("COLOR", Semantic::Arbitrary, f, Interp::LinearCentroid, 3, 0, 4),
    ];
    let target1 = SigElement { index: 1, ..elem("SV_Target", Semantic::Target, f, Interp::Undefined, 1, 0, 4) };
    s.outputs = vec![
        elem("SV_Target", Semantic::Target, f, Interp::Undefined, 0, 0, 4),
        target1,
        elem("SV_Depth", Semantic::Depth, f, Interp::Undefined, SigElement::UNALLOCATED, 0, 1),
        elem("SV_Coverage", Semantic::Coverage, u, Interp::Undefined, SigElement::UNALLOCATED, 0, 1),
    ];
    let mut b0 = vec![load(1, 0, 0), load(1, 1, 1)];
    b0.extend((0..4).map(|k| load(6, k, 2 + k as u32)));
    b0.extend([load(2, 0, 6), load(3, 0, 7), load(4, 0, 8), load(5, 0, 9), load(0, 2, 10)]);
    for k in 0..4u32 {
        b0.push(Inst::new(&[11 + k], Op::Bin(BinOp::FAdd, v(2 + k), v(0))));
        b0.push(store(0, k as u8, v(11 + k)));
    }
    b0.push(Inst::new(&[15], Op::Cast(CastOp::UiToFp, v(6), Ty::F32)));
    b0.push(Inst::new(&[16], Op::Bin(BinOp::Add, v(8), v(9))));
    b0.push(Inst::new(&[17], Op::Cast(CastOp::UiToFp, v(16), Ty::F32)));
    b0.push(Inst::new(&[18], Op::ICmp(IPred::Ne, v(7), c(0))));
    let phi = Phi { dst: ValueId(19), ty: Ty::F32, incoming: vec![(BlockId(0), v(0)), (BlockId(1), Operand::f32(1.0))] };
    let b2 = Block {
        phis: vec![phi],
        insts: vec![store(1, 0, v(19)), store(1, 1, v(1)), store(1, 2, v(15)), store(1, 3, v(17)), store(2, 0, v(10)), store(3, 0, c(3))],
        term: Terminator::Ret,
    };
    s.blocks = vec![Block::new(b0, Terminator::CondBr(v(18), BlockId(1), BlockId(2))), Block::new(vec![], br(2)), b2];
    s
}

/// [`pixel_texture`] with the texture taken from the resource heap and the sampler from the sampler heap.
pub fn pixel_heap_texture() -> Shader {
    let mut s = pixel_texture();
    let Op::Sample { tex, sampler, .. } = &mut s.blocks[0].insts[2].op else { unreachable!() };
    *tex = ResRef::heap(0, c(3), false);
    *sampler = ResRef::heap(1, c(2), true);
    s.resources[0] = s.resources[0].clone().in_heap();
    s.resources[1] = s.resources[1].clone().in_heap();
    s
}

/// A vertex shader with system value inputs, an array output written in a loop with a dynamic row,
/// an output written under a condition, a clip distance and an integer output.
pub fn vertex_zoo() -> Shader {
    let (f, u) = (ComponentType::F32, ComponentType::U32);
    let elem = |name: &str, semantic, comp, interp, row, col, cols| SigElement { start_col: col, ..SigElement::new(name, 0, semantic, comp, interp, row, cols) };
    let mut s = Shader::vertex(vec![]);
    s.inputs = vec![
        elem("POSITION", Semantic::Arbitrary, f, Interp::Undefined, 0, 0, 4),
        elem("SV_VertexID", Semantic::VertexId, u, Interp::Undefined, 1, 0, 1),
        elem("SV_InstanceID", Semantic::InstanceId, u, Interp::Undefined, 1, 1, 1),
    ];
    let array = SigElement { rows: 3, ..elem("TEXCOORD", Semantic::Arbitrary, f, Interp::Linear, 1, 0, 2) };
    s.outputs = vec![
        elem("SV_Position", Semantic::Position, f, Interp::LinearNoperspective, 0, 0, 4),
        array,
        elem("COLOR", Semantic::Arbitrary, f, Interp::Linear, 4, 0, 4),
        elem("SV_ClipDistance", Semantic::ClipDistance, f, Interp::Linear, 5, 0, 2),
        elem("ID", Semantic::Arbitrary, u, Interp::Constant, 6, 0, 1),
    ];
    let mut b0: Vec<Inst> = (0..4).map(|k| load(0, k, k as u32)).collect();
    b0.extend([load(1, 0, 4), load(2, 0, 5)]);
    b0.extend((0..4).map(|k| store(0, k, v(k as u32))));
    // loop: for i in 0..3 { TEXCOORD[i] = (pos.x * i, pos.y + instance) }
    let phi_i = Phi { dst: ValueId(6), ty: Ty::I32, incoming: vec![(BlockId(0), c(0)), (BlockId(1), v(11))] };
    let header = Block {
        phis: vec![phi_i],
        insts: vec![
            Inst::new(&[7], Op::Cast(CastOp::UiToFp, v(6), Ty::F32)),
            Inst::new(&[8], Op::Bin(BinOp::FMul, v(0), v(7))),
            Inst::new(&[9], Op::Cast(CastOp::UiToFp, v(5), Ty::F32)),
            Inst::new(&[10], Op::Bin(BinOp::FAdd, v(1), v(9))),
            Inst::new(&[], Op::StoreOutput { elem: 1, row: v(6), col: 0, value: v(8) }),
            Inst::new(&[], Op::StoreOutput { elem: 1, row: v(6), col: 1, value: v(10) }),
            Inst::new(&[11], Op::Bin(BinOp::Add, v(6), c(1))),
            Inst::new(&[12], Op::ICmp(IPred::Eq, v(11), c(3))),
        ],
        term: Terminator::CondBr(v(12), BlockId(2), BlockId(1)),
    };
    let after = Block::new(
        vec![
            Inst::new(&[13], Op::Bin(BinOp::And, v(4), c(1))),
            Inst::new(&[14], Op::ICmp(IPred::Ne, v(13), c(0))),
            store(3, 0, v(2)),
            store(3, 1, v(3)),
            store(4, 0, v(4)),
        ],
        Terminator::CondBr(v(14), BlockId(3), BlockId(4)),
    );
    let then = Block::new(vec![store(2, 0, v(0))], br(4));
    let join = Block::new(vec![], Terminator::Ret);
    s.blocks = vec![Block::new(b0, br(1)), header, after, then, join];
    s
}

/// A pixel shader that discards the right half of the target (`TEXCOORD0.x > 0.5`) and writes green elsewhere.
pub fn pixel_discard() -> Shader {
    let insts = vec![
        load(1, 0, 0),
        Inst::new(&[1], Op::FCmp(FPred::Ogt, v(0), Operand::f32(0.5))),
        Inst::new(&[], Op::Discard(v(1))),
        store(0, 0, Operand::f32(0.0)),
        store(0, 1, Operand::f32(1.0)),
        store(0, 2, Operand::f32(0.0)),
        store(0, 3, Operand::f32(1.0)),
    ];
    let mut s = Shader::pixel(vec![Block::new(insts, Terminator::Ret)]);
    s.inputs.push(position_out());
    s.inputs.push(SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Linear, 1, 2));
    s.outputs.push(target_out());
    s
}

/// A pixel shader that uses every form of the texture operations, for the validator only.
pub fn texture_zoo() -> Shader {
    let f4 = Element { component: ComponentType::F32, count: 4 };
    let mut s = Shader::pixel(vec![]);
    s.inputs.push(position_out());
    s.inputs.push(SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Linear, 1, 2));
    s.outputs.push(target_out());
    let tex = |kind, reg| Resource::texture(ResourceClass::Srv, kind, 0, reg, f4);
    s.resources.push(tex(ResourceKind::Texture2D, 0)); // 0
    s.resources.push(tex(ResourceKind::Texture2DArray, 1)); // 1
    s.resources.push(tex(ResourceKind::Texture3D, 2)); // 2
    s.resources.push(tex(ResourceKind::TextureCube, 3)); // 3
    s.resources.push(tex(ResourceKind::Texture1D, 4)); // 4
    s.resources.push(Resource::texture(ResourceClass::Srv, ResourceKind::Texture2D, 0, 5, Element { component: ComponentType::F32, count: 1 })); // 5 depth
    s.resources.push(Resource::texture(ResourceClass::Uav, ResourceKind::Texture2D, 0, 0, f4)); // 6
    s.resources.push(Resource::texture(ResourceClass::Srv, ResourceKind::Texture2D, 1, 0, Element { component: ComponentType::U32, count: 4 })); // 7 uint texture
    s.resources.push(Resource::sampler(0, 0, false)); // 8
    s.resources.push(Resource::sampler(0, 1, true)); // 9
    let r = ResRef::bound;
    let mut n = 0u32;
    let mut next = |count: u32| -> Vec<u32> {
        let ids: Vec<u32> = (n..n + count).collect();
        n += count;
        ids
    };
    let f = Operand::f32;
    let mut insts = vec![load(1, 0, next(1)[0]), load(1, 1, next(1)[0])];
    let (u, w) = (v(0), v(1));
    let sample = |tex, mode: SampleMode, coords: Vec<Operand>, offsets: Vec<i32>, dst: Vec<u32>| {
        Inst { dsts: dst.into_iter().map(ValueId).collect(), op: Op::Sample { tex: r(tex), sampler: r(8), coords, offsets, mode, clamp: None, ty: Ty::F32 } }
    };
    let mut results: Vec<u32> = Vec::new();
    let mut push = |insts: &mut Vec<Inst>, i: Inst| {
        results.push(i.dsts[0].0);
        insts.push(i);
    };
    push(&mut insts, sample(0, SampleMode::Implicit, vec![u, w], vec![1, -1], next(4)));
    push(&mut insts, sample(0, SampleMode::Bias(f(0.5)), vec![u, w], vec![], next(4)));
    push(&mut insts, sample(0, SampleMode::Level(f(2.0)), vec![u, w], vec![], next(4)));
    push(&mut insts, sample(0, SampleMode::Grad { ddx: vec![f(0.01), f(0.0)], ddy: vec![f(0.0), f(0.01)] }, vec![u, w], vec![], next(4)));
    push(&mut insts, sample(1, SampleMode::Implicit, vec![u, w, f(1.0)], vec![], next(4)));
    push(&mut insts, sample(2, SampleMode::Level(f(0.0)), vec![u, w, f(0.5)], vec![], next(4)));
    push(&mut insts, sample(3, SampleMode::Implicit, vec![u, w, f(1.0)], vec![], next(4)));
    push(&mut insts, sample(4, SampleMode::Implicit, vec![u], vec![], next(4)));
    let cmp_sample = |mode, dst: Vec<u32>| {
        Inst { dsts: dst.into_iter().map(ValueId).collect(), op: Op::Sample { tex: r(5), sampler: r(9), coords: vec![u, w], offsets: vec![], mode, clamp: None, ty: Ty::F32 } }
    };
    push(&mut insts, cmp_sample(SampleMode::Compare(f(0.5)), next(4)));
    push(&mut insts, cmp_sample(SampleMode::CompareLevelZero(f(0.5)), next(4)));
    let gather = Inst { dsts: next(4).into_iter().map(ValueId).collect(), op: Op::Gather { tex: r(0), sampler: r(8), coords: vec![u, w], offsets: vec![], channel: 1, compare: None, ty: Ty::F32 } };
    push(&mut insts, gather);
    let gather_cmp = Inst { dsts: next(4).into_iter().map(ValueId).collect(), op: Op::Gather { tex: r(5), sampler: r(9), coords: vec![u, w], offsets: vec![], channel: 0, compare: Some(f(0.5)), ty: Ty::F32 } };
    push(&mut insts, gather_cmp);
    let dims = next(4);
    insts.push(Inst { dsts: dims.iter().copied().map(ValueId).collect(), op: Op::GetDimensions { tex: r(0), mip: Some(c(0)) } });
    let uav_dims = next(4);
    insts.push(Inst { dsts: uav_dims.iter().copied().map(ValueId).collect(), op: Op::GetDimensions { tex: r(6), mip: None } });
    let lod = next(1);
    insts.push(Inst { dsts: vec![ValueId(lod[0])], op: Op::CalculateLod { tex: r(0), sampler: r(8), coords: vec![u, w], clamped: true } });
    let lod2 = next(1);
    insts.push(Inst { dsts: vec![ValueId(lod2[0])], op: Op::CalculateLod { tex: r(1), sampler: r(8), coords: vec![u, w], clamped: false } });
    push(&mut insts, Inst { dsts: next(4).into_iter().map(ValueId).collect(), op: Op::TexLoad { tex: r(0), mip: Some(c(1)), coords: vec![c(3), c(4)], offsets: vec![0, 1], ty: Ty::F32 } });
    push(&mut insts, Inst { dsts: next(4).into_iter().map(ValueId).collect(), op: Op::TexLoad { tex: r(7), mip: Some(c(0)), coords: vec![c(1), c(2)], offsets: vec![], ty: Ty::I32 } });
    push(&mut insts, Inst { dsts: next(4).into_iter().map(ValueId).collect(), op: Op::TexLoad { tex: r(6), mip: None, coords: vec![c(1), c(2)], offsets: vec![], ty: Ty::F32 } });
    insts.push(Inst { dsts: vec![], op: Op::TexStore { tex: r(6), coords: vec![c(5), c(6)], values: vec![f(1.0), f(2.0), f(3.0), f(4.0)] } });
    // Derivatives of the coordinate and their sum with the first sample.
    let dd = next(4);
    for (d, op) in dd.iter().zip([UnOp::DdxCoarse, UnOp::DdyCoarse, UnOp::DdxFine, UnOp::DdyFine]) {
        insts.push(Inst::new(&[*d], Op::Un(op, u)));
    }
    // The colour: the first sample plus the derivatives.
    let acc0 = next(1)[0];
    insts.push(Inst::new(&[acc0], Op::Bin(BinOp::FAdd, v(2), v(dd[0]))));
    let acc1 = next(1)[0];
    insts.push(Inst::new(&[acc1], Op::Bin(BinOp::FAdd, v(acc0), v(dd[1]))));
    let acc2 = next(1)[0];
    insts.push(Inst::new(&[acc2], Op::Bin(BinOp::FAdd, v(acc1), v(dd[2]))));
    let acc3 = next(1)[0];
    insts.push(Inst::new(&[acc3], Op::Bin(BinOp::FAdd, v(acc2), v(dd[3]))));
    for k in 0..4 {
        insts.push(store(0, k, if k == 0 { v(acc3) } else { v(2 + k as u32) }));
    }
    let _ = results;
    s.blocks.push(Block::new(insts, Terminator::Ret));
    s
}

/// A compute shader on a `RWTexture2D<float4>`: stores a value and loads the neighbour back.
pub fn texture_compute() -> Shader {
    let f4 = Element { component: ComponentType::F32, count: 4 };
    let insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[1], Op::ThreadId(ThreadComponentKind::Dispatch, 1)),
        Inst::new(&[2], Op::Cast(CastOp::UiToFp, v(0), Ty::F32)),
        Inst { dsts: vec![], op: Op::TexStore { tex: ResRef::bound(0), coords: vec![v(0), v(1)], values: vec![v(2), Operand::f32(0.0), Operand::f32(1.0), Operand::f32(1.0)] } },
        Inst::new(&[3], Op::Bin(BinOp::Add, v(0), c(1))),
        Inst { dsts: vec![ValueId(4), ValueId(5), ValueId(6), ValueId(7)], op: Op::TexLoad { tex: ResRef::bound(0), mip: None, coords: vec![v(3), v(1)], offsets: vec![], ty: Ty::F32 } },
        Inst::new(&[8], Op::Bin(BinOp::FMul, v(4), v(5))),
        Inst { dsts: vec![], op: Op::BufStore { res: ResRef::bound(1), index: c(0), offset: None, values: vec![v(8)] } },
    ];
    let mut s = Shader::compute([8, 8, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(Resource::texture(ResourceClass::Uav, ResourceKind::Texture2D, 0, 0, f4));
    s.resources.push(uav_raw(1));
    s
}

/// A structured copy: components 1 to 3 of element `tid` of the SRV t0 (stride 16) to components
/// 0 to 2 of element `tid` of the UAV u0 (stride 16).
pub fn structured_copy() -> Shader {
    let insts = vec![
        Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
        Inst::new(&[1, 2, 3], Op::BufLoad { res: ResRef::bound(0), index: v(0), offset: Some(c(4)), count: 3, ty: Ty::I32 }),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(1), index: v(0), offset: Some(c(0)), values: vec![v(1), v(2), v(3)] }),
    ];
    let mut s = Shader::compute([64, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    s.resources.push(Resource::structured(ResourceClass::Srv, 0, 0, 16));
    s.resources.push(Resource::structured(ResourceClass::Uav, 0, 0, 16));
    s
}

/// Every resource class and kind at once, for the validator only (typed buffers need descriptors):
/// a constant buffer, raw, structured and typed SRVs, raw, structured and typed UAVs, in two spaces.
pub fn resource_zoo() -> Shader {
    let typed = Element { component: ComponentType::F32, count: 4 };
    let mut s = Shader::compute([8, 8, 1], vec![]);
    s.resources.push(Resource::cbuffer(0, 0, 16)); // 0
    s.resources.push(Resource::raw(ResourceClass::Srv, 0, 0)); // 1
    s.resources.push(Resource::structured(ResourceClass::Srv, 1, 3, 12)); // 2
    s.resources.push(Resource::typed(ResourceClass::Srv, 0, 2, typed)); // 3
    s.resources.push(uav_raw(0)); // 4
    s.resources.push(Resource::structured(ResourceClass::Uav, 0, 1, 8)); // 5
    s.resources.push(Resource::typed(ResourceClass::Uav, 0, 2, typed)); // 6
    let insts = vec![
        Inst::new(&[0], Op::CbufLoad { res: ResRef::bound(0), row: c(0), comps: vec![0], ty: Ty::I32 }),
        Inst::new(&[1], Op::BufLoad { res: ResRef::bound(1), index: v(0), offset: None, count: 1, ty: Ty::I32 }),
        Inst::new(&[2], Op::BufLoad { res: ResRef::bound(2), index: v(0), offset: Some(c(4)), count: 1, ty: Ty::I32 }),
        Inst::new(&[3, 4, 5, 6], Op::BufLoad { res: ResRef::bound(3), index: v(0), offset: None, count: 4, ty: Ty::F32 }),
        Inst::new(&[7, 8, 9, 10], Op::BufLoad { res: ResRef::bound(6), index: v(0), offset: None, count: 4, ty: Ty::F32 }),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(6), index: v(0), offset: None, values: vec![v(7), v(8), v(9), v(10)] }),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(4), index: v(0), offset: None, values: vec![v(1), v(2)] }),
        Inst::new(&[], Op::BufStore { res: ResRef::bound(5), index: v(0), offset: Some(c(4)), values: vec![v(2)] }),
        Inst::new(&[11], Op::BufAtomic { op: AtomicOp::UMax, res: ResRef::bound(5), index: v(0), offset: Some(c(0)), value: v(1) }),
    ];
    s.blocks.push(Block::new(insts, Terminator::Ret));
    s
}
