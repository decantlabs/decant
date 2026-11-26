//! Lowering of the IR to the module of `module.rs`: instruction selection, `dx.op` declarations,
//! resource handles, DXIL metadata, shader flags.

use std::collections::HashMap;

use crate::Error;
use crate::dxop::{self, DxOp};
use crate::ir::*;
use crate::module::*;

/// Calls `f` for every operand of `op`.
pub(crate) fn for_each_operand(op: &Op, f: &mut dyn FnMut(&Operand)) {
    for r in res_refs(op) {
        if let Some(i) = &r.index {
            f(i);
        }
    }
    match op {
        Op::Bin(_, a, b) | Op::MinMax(_, a, b) | Op::AddCarry(a, b) | Op::SubBorrow(a, b) | Op::ICmp(_, a, b) | Op::FCmp(_, a, b) => {
            f(a);
            f(b);
        }
        Op::LoadInput { row, .. } => f(row),
        Op::StoreOutput { row, value, .. } => {
            f(row);
            f(value);
        }
        Op::Discard(a) => f(a),
        Op::TexLoad { mip, coords, .. } => {
            mip.iter().for_each(&mut *f);
            coords.iter().for_each(&mut *f);
        }
        Op::TexAtomic { coords, value, .. } => {
            coords.iter().for_each(&mut *f);
            f(value);
        }
        Op::TexCmpXchg { coords, compare, new, .. } => {
            coords.iter().for_each(&mut *f);
            f(compare);
            f(new);
        }
        Op::TexStore { coords, values, .. } => {
            coords.iter().for_each(&mut *f);
            values.iter().for_each(&mut *f);
        }
        Op::Sample { coords, mode, clamp, .. } => {
            coords.iter().for_each(&mut *f);
            clamp.iter().for_each(&mut *f);
            match mode {
                SampleMode::Implicit => {}
                SampleMode::Bias(a) | SampleMode::Level(a) | SampleMode::Compare(a) | SampleMode::CompareLevelZero(a) => f(a),
                SampleMode::Grad { ddx, ddy } => {
                    ddx.iter().for_each(&mut *f);
                    ddy.iter().for_each(&mut *f);
                }
            }
        }
        Op::Gather { coords, compare, .. } => {
            coords.iter().for_each(&mut *f);
            compare.iter().for_each(&mut *f);
        }
        Op::GetDimensions { mip, .. } => mip.iter().for_each(&mut *f),
        Op::CalculateLod { coords, .. } => coords.iter().for_each(&mut *f),
        Op::Un(_, a)
        | Op::Cast(_, a, _)
        | Op::WaveAnyTrue(a)
        | Op::WaveAllTrue(a)
        | Op::WaveAllEqual(a)
        | Op::WaveBallot(a)
        | Op::WaveReadLaneFirst(a)
        | Op::WaveActiveCountBits(a)
        | Op::WavePrefixCountBits(a)
        | Op::WaveActive { value: a, .. }
        | Op::WaveActiveBit { value: a, .. }
        | Op::WavePrefix { value: a, .. }
        | Op::QuadOp { value: a, .. } => f(a),
        Op::WaveReadLaneAt { value, lane } | Op::QuadReadLaneAt { value, lane } => {
            f(value);
            f(lane);
        }
        Op::Mad(_, a, b, c) | Op::Select(a, b, c) => {
            f(a);
            f(b);
            f(c);
        }
        Op::BitfieldExtract { width, offset, value, .. } => {
            f(width);
            f(offset);
            f(value);
        }
        Op::BitfieldInsert { width, offset, insert, base } => {
            f(width);
            f(offset);
            f(insert);
            f(base);
        }
        Op::ThreadId(..) | Op::FlattenedThreadIdInGroup | Op::Barrier { .. } | Op::IsHelperLane | Op::WaveLaneIndex | Op::WaveLaneCount | Op::WaveIsFirstLane => {}
        Op::CbufLoad { row, .. } => f(row),
        Op::BufLoad { index, offset, .. } => {
            f(index);
            offset.iter().for_each(&mut *f);
        }
        Op::BufStore { index, offset, values, .. } => {
            f(index);
            offset.iter().for_each(&mut *f);
            values.iter().for_each(&mut *f);
        }
        Op::BufAtomic { index, offset, value, .. } => {
            f(index);
            offset.iter().for_each(&mut *f);
            f(value);
        }
        Op::BufCmpXchg { index, offset, compare, new, .. } => {
            f(index);
            offset.iter().for_each(&mut *f);
            f(compare);
            f(new);
        }
        Op::GsLoad { index, .. } => f(index),
        Op::GsStore { index, value, .. } | Op::GsAtomic { index, value, .. } => {
            f(index);
            f(value);
        }
        Op::GsCmpXchg { index, compare, new, .. } => {
            f(index);
            f(compare);
            f(new);
        }
    }
}

struct ResRefs<'a> {
    items: [Option<&'a ResRef>; 2],
    idx: usize,
}

impl<'a> Iterator for ResRefs<'a> {
    type Item = &'a ResRef;
    #[inline]
    fn next(&mut self) -> Option<&'a ResRef> {
        while self.idx < 2 {
            let item = self.items[self.idx];
            self.idx += 1;
            if item.is_some() {
                return item;
            }
        }
        None
    }
}

/// The resources an instruction names.
#[inline]
fn res_refs(op: &Op) -> ResRefs<'_> {
    let items = match op {
        Op::CbufLoad { res, .. } | Op::BufLoad { res, .. } | Op::BufStore { res, .. } | Op::BufAtomic { res, .. } | Op::BufCmpXchg { res, .. } => [Some(res), None],
        Op::TexLoad { tex, .. } | Op::TexStore { tex, .. } | Op::TexAtomic { tex, .. } | Op::TexCmpXchg { tex, .. } | Op::GetDimensions { tex, .. } => [Some(tex), None],
        Op::Sample { tex, sampler, .. } | Op::Gather { tex, sampler, .. } | Op::CalculateLod { tex, sampler, .. } => [Some(tex), Some(sampler)],
        _ => [None, None],
    };
    ResRefs { items, idx: 0 }
}

/// The last register of a range (the range can be unbounded: `u32::MAX` registers).
fn upper_bound(r: &Resource) -> u32 {
    if r.range_size == u32::MAX { u32::MAX } else { r.lower_bound + r.range_size - 1 }
}

/// What the lowering found out besides the module.
pub struct Lowered {
    pub module: Module,
    /// Feature info bits (`SFI0`).
    pub features: u64,
    /// `createHandleFromHeap` is used for resources.
    pub resource_heap: bool,
    /// `createHandleFromHeap` is used for samplers.
    pub sampler_heap: bool,
    pub wave_size: Option<u32>,
    /// The signature analysis of a vertex or pixel shader.
    pub sig: Option<crate::sig::SigData>,
    /// PSV0 resource table, in PSV order.
    pub psv: Vec<crate::container::PsvResource>,
}

/// Instruction record codes (LLVMBitCodes 3.7 `FunctionCodes`).
mod fc {
    pub const BINOP: u32 = 2;
    pub const CAST: u32 = 3;
    pub const RET: u32 = 10;
    pub const BR: u32 = 11;
    pub const PHI: u32 = 16;
    pub const LOAD: u32 = 20;
    pub const EXTRACTVAL: u32 = 26;
    pub const CMP2: u32 = 28;
    pub const VSELECT: u32 = 29;
    pub const CALL: u32 = 34;
    pub const ATOMICRMW: u32 = 38;
    pub const GEP: u32 = 43;
    pub const STORE: u32 = 44;
    pub const CMPXCHG: u32 = 46;
}

struct ValMap {
    dense: Vec<Option<(Ref, Ty)>>,
    sparse: HashMap<u32, (Ref, Ty)>,
}

impl ValMap {
    fn new(max_dense: usize) -> Self {
        Self { dense: vec![None; max_dense], sparse: HashMap::new() }
    }
    #[inline]
    fn get(&self, id: u32) -> Option<(Ref, Ty)> {
        let idx = id as usize;
        if idx < self.dense.len() { self.dense[idx] } else { self.sparse.get(&id).copied() }
    }
    #[inline]
    fn insert(&mut self, id: u32, val: (Ref, Ty)) -> Option<(Ref, Ty)> {
        let idx = id as usize;
        if idx < self.dense.len() {
            self.dense[idx].replace(val)
        } else {
            self.sparse.insert(id, val)
        }
    }
}

struct Lower<'a> {
    sh: &'a Shader,
    m: Module,
    vals: ValMap,
    next_inst: u32,
    handles: Vec<Option<Ref>>,
    dx_funcs: HashMap<(&'static str, Option<&'static str>), u32>,
    gs: Vec<u32>,
    flags: u64,
    features: u64,
    resource_heap: bool,
    sampler_heap: bool,
    sig: Option<crate::sig::SigData>,
    handle_ty_id: Option<TypeId>,
    resbind_ty_id: Option<TypeId>,
    props_ty_id: Option<TypeId>,
    cbufret_f32_id: Option<TypeId>,
    cbufret_i32_id: Option<TypeId>,
    resret_f32_id: Option<TypeId>,
    resret_i32_id: Option<TypeId>,
}

fn suffix(t: Ty) -> &'static str {
    match t {
        Ty::Bool => "i1",
        Ty::I32 => "i32",
        Ty::I64 => "i64",
        Ty::F16 => "f16",
        Ty::F32 => "f32",
    }
}

fn unsupported<T>(what: &str) -> Result<T, Error> {
    Err(Error::Unsupported(what.to_owned()))
}

/// DXIL `ComponentType` numbers.
fn component_code(c: ComponentType) -> u32 {
    match c {
        ComponentType::I32 => 4,
        ComponentType::U32 => 5,
        ComponentType::F32 => 9,
    }
}

fn kind_code(k: ResourceKind) -> u32 {
    match k {
        ResourceKind::TypedBuffer => 10,
        ResourceKind::RawBuffer => 11,
        ResourceKind::StructuredBuffer => 12,
        ResourceKind::CBuffer => 13,
        ResourceKind::Sampler => 14,
        ResourceKind::Texture1D => 1,
        ResourceKind::Texture2D => 2,
        ResourceKind::Texture3D => 4,
        ResourceKind::TextureCube => 5,
        ResourceKind::Texture1DArray => 6,
        ResourceKind::Texture2DArray => 7,
        ResourceKind::TextureCubeArray => 9,
    }
}

impl<'a> Lower<'a> {
    fn ty(&mut self, t: Ty) -> Result<TypeId, Error> {
        Ok(match t {
            Ty::Bool => self.m.types.int(1),
            Ty::I32 => self.m.types.int(32),
            Ty::I64 => {
                self.flags |= 1 << 20;
                self.features |= 0x8000;
                self.m.types.int(64)
            }
            Ty::F16 => return unsupported("f16 values (native 16 bit types are out of scope)"),
            Ty::F32 => self.m.types.float(),
        })
    }

    fn int_const(&mut self, ty: TypeId, v: i64) -> Ref {
        Ref::Const(self.m.consts.intern(ty, ConstKind::Int(v)))
    }
    fn i32c(&mut self, v: u32) -> Ref {
        let t = self.m.types.int(32);
        self.int_const(t, v as i32 as i64)
    }
    fn i8c(&mut self, v: u32) -> Ref {
        let t = self.m.types.int(8);
        self.int_const(t, v as u8 as i8 as i64)
    }
    fn undef(&mut self, t: TypeId) -> Ref {
        Ref::Const(self.m.consts.intern(t, ConstKind::Undef))
    }

    fn constant(&mut self, c: Const) -> Result<(Ref, Ty), Error> {
        let t = c.ty();
        let id = self.ty(t)?;
        let r = match c {
            Const::Bool(b) => self.int_const(id, if b { -1 } else { 0 }),
            Const::I32(v) => self.int_const(id, v as i32 as i64),
            Const::I64(v) => self.int_const(id, v as i64),
            Const::F32(bits) => Ref::Const(self.m.consts.intern(id, ConstKind::Float(bits as u64))),
            Const::F16(_) => return unsupported("f16 constants"),
        };
        Ok((r, t))
    }

    #[inline]
    fn opnd(&mut self, o: &Operand) -> Result<(Ref, Ty), Error> {
        match o {
            Operand::Const(c) => self.constant(*c),
            Operand::Value(v) => self.vals.get(v.0).ok_or_else(|| Error::UseBeforeDef(format!("value {} is used before its definition", v.0))),
        }
    }

    fn expect(&mut self, o: &Operand, want: Ty, what: &str) -> Result<Ref, Error> {
        let (r, t) = self.opnd(o)?;
        if t != want {
            return Err(Error::Invalid(format!("{what}: expected {want:?}, got {t:?}")));
        }
        Ok(r)
    }

    fn emit(&mut self, code: u32, ops: Vec<LOp>, has_value: bool) -> Option<Ref> {
        self.m.body.push(LInst { code, ops, has_value });
        has_value.then(|| {
            let r = Ref::Inst(self.next_inst);
            self.next_inst += 1;
            r
        })
    }

    fn def(&mut self, inst: &Inst, i: usize, r: Ref, t: Ty) -> Result<(), Error> {
        let d = inst.dsts.get(i).ok_or_else(|| Error::Invalid(format!("instruction {:?} needs result id number {i}", inst.op)))?;
        if self.vals.insert(d.0, (r, t)).is_some() {
            return Err(Error::Invalid(format!("value {} is defined twice", d.0)));
        }
        Ok(())
    }

    fn check_dsts(inst: &Inst, n: usize) -> Result<(), Error> {
        if inst.dsts.len() != n {
            return Err(Error::Invalid(format!("{:?} has {} result ids, expected {n}", inst.op, inst.dsts.len())));
        }
        Ok(())
    }

    /// A `dx.op` call: the opcode is the first argument.
    fn dx(&mut self, op: DxOp, overload: Option<&'static str>, ret: TypeId, args: &[(TypeId, Ref)]) -> Option<Ref> {
        let i32t = self.m.types.int(32);
        let code = self.i32c(op.opcode);
        let idx = match self.dx_funcs.get(&(op.class, overload)) {
            Some(&i) => i,
            None => {
                let name = match overload {
                    Some(o) => format!("dx.op.{}.{o}", op.class),
                    None => format!("dx.op.{}", op.class),
                };
                let mut params = Vec::with_capacity(1 + args.len());
                params.push(i32t);
                params.extend(args.iter().map(|a| a.0));
                let fty = self.m.types.func(ret, params);
                self.m.types.ptr(fty, 0);
                let i = self.m.funcs.len() as u32;
                self.m.funcs.push(Func { name, ty: fty, attr: Some(op.attr), proto: true });
                self.dx_funcs.insert((op.class, overload), i);
                i
            }
        };
        let fty = self.m.funcs[idx as usize].ty;
        let void = self.m.types.void();
        let mut ops = Vec::with_capacity(5 + args.len());
        ops.push(LOp::Imm(0));
        ops.push(LOp::Imm(1 << 15));
        ops.push(LOp::Imm(fty as u64));
        ops.push(LOp::Rel(Ref::Func(idx)));
        ops.push(LOp::Rel(code));
        ops.extend(args.iter().map(|a| LOp::Rel(a.1)));
        self.emit(fc::CALL, ops, ret != void)
    }

    #[inline]
    fn extract(&mut self, agg: Ref, idx: u64) -> Ref {
        self.emit(fc::EXTRACTVAL, vec![LOp::Rel(agg), LOp::Imm(idx)], true).expect("has a value")
    }

    // DXIL types (cached on `Lower`).
    fn handle_ty(&mut self) -> TypeId {
        if let Some(id) = self.handle_ty_id {
            return id;
        }
        let i8t = self.m.types.int(8);
        let p = self.m.types.ptr(i8t, 0);
        let id = self.m.types.named("dx.types.Handle", vec![p]);
        self.handle_ty_id = Some(id);
        id
    }
    fn resbind_ty(&mut self) -> TypeId {
        if let Some(id) = self.resbind_ty_id {
            return id;
        }
        let (i32t, i8t) = (self.m.types.int(32), self.m.types.int(8));
        let id = self.m.types.named("dx.types.ResBind", vec![i32t, i32t, i32t, i8t]);
        self.resbind_ty_id = Some(id);
        id
    }
    fn props_ty(&mut self) -> TypeId {
        if let Some(id) = self.props_ty_id {
            return id;
        }
        let i32t = self.m.types.int(32);
        let id = self.m.types.named("dx.types.ResourceProperties", vec![i32t, i32t]);
        self.props_ty_id = Some(id);
        id
    }
    fn cbufret_ty(&mut self, t: Ty) -> Result<TypeId, Error> {
        match t {
            Ty::F32 => {
                if let Some(id) = self.cbufret_f32_id {
                    return Ok(id);
                }
                let e = self.ty(t)?;
                let id = self.m.types.named("dx.types.CBufRet.f32", vec![e; 4]);
                self.cbufret_f32_id = Some(id);
                Ok(id)
            }
            Ty::I32 => {
                if let Some(id) = self.cbufret_i32_id {
                    return Ok(id);
                }
                let e = self.ty(t)?;
                let id = self.m.types.named("dx.types.CBufRet.i32", vec![e; 4]);
                self.cbufret_i32_id = Some(id);
                Ok(id)
            }
            _ => {
                let e = self.ty(t)?;
                Ok(self.m.types.named(&format!("dx.types.CBufRet.{}", suffix(t)), vec![e; 4]))
            }
        }
    }
    fn resret_ty(&mut self, t: Ty) -> Result<TypeId, Error> {
        match t {
            Ty::F32 => {
                if let Some(id) = self.resret_f32_id {
                    return Ok(id);
                }
                let e = self.ty(t)?;
                let i32t = self.m.types.int(32);
                let id = self.m.types.named("dx.types.ResRet.f32", vec![e, e, e, e, i32t]);
                self.resret_f32_id = Some(id);
                Ok(id)
            }
            Ty::I32 => {
                if let Some(id) = self.resret_i32_id {
                    return Ok(id);
                }
                let e = self.ty(t)?;
                let i32t = self.m.types.int(32);
                let id = self.m.types.named("dx.types.ResRet.i32", vec![e, e, e, e, i32t]);
                self.resret_i32_id = Some(id);
                Ok(id)
            }
            _ => {
                let e = self.ty(t)?;
                let i32t = self.m.types.int(32);
                Ok(self.m.types.named(&format!("dx.types.ResRet.{}", suffix(t)), vec![e, e, e, e, i32t]))
            }
        }
    }

    /// The handles of the bound resources: `createHandleFromBinding` then `annotateHandle`, once per
    /// used resource, in the entry block.
    fn make_handles(&mut self) -> Result<(), Error> {
        let sh = self.sh;
        let mut used = vec![false; sh.resources.len()];
        for b in &sh.blocks {
            for i in &b.insts {
                for r in res_refs(&i.op) {
                    let decl = sh.resources.get(r.res as usize).ok_or_else(|| Error::Invalid(format!("resource {} does not exist", r.res)))?;
                    if decl.binding == Binding::Bound && r.index.is_none() {
                        used[r.res as usize] = true;
                    }
                }
            }
        }
        let (handle, resbind, i32t, i1t) = {
            let h = self.handle_ty();
            let rb = self.resbind_ty();
            (h, rb, self.m.types.int(32), self.m.types.int(1))
        };
        for (n, r) in sh.resources.iter().enumerate() {
            if !used[n] {
                continue;
            }
            let class = match r.class {
                ResourceClass::Srv => 0,
                ResourceClass::Uav => 1,
                ResourceClass::CBuffer => 2,
                ResourceClass::Sampler => 3,
            };
            let bind = self.bind_const(r, class, resbind);
            // The index operand is the absolute register, inside [lower bound, upper bound] of the range.
            let register = self.i32c(r.lower_bound);
            let nonuniform = self.int_const(i1t, 0);
            let h = self.dx(dxop::CREATE_HANDLE_FROM_BINDING, None, handle, &[(resbind, bind), (i32t, register), (i1t, nonuniform)]).expect("handle");
            self.handles[n] = Some(self.annotate(n, h)?);
        }
        Ok(())
    }

    /// The `ResBind` constant of a bound resource.
    fn bind_const(&mut self, r: &Resource, class: u32, resbind: TypeId) -> Ref {
        let (lb, ub) = (self.i32c(r.lower_bound), self.i32c(upper_bound(r)));
        let (sp, cl) = (self.i32c(r.space), self.i8c(class));
        let (Ref::Const(a), Ref::Const(b), Ref::Const(c), Ref::Const(d)) = (lb, ub, sp, cl) else { unreachable!() };
        Ref::Const(self.m.consts.intern(resbind, ConstKind::Aggregate(vec![a, b, c, d])))
    }

    /// `annotateHandle` of the handle `h` of resource `n`.
    fn annotate(&mut self, n: usize, h: Ref) -> Result<Ref, Error> {
        let r = &self.sh.resources[n];
        let (handle, props_t) = (self.handle_ty(), self.props_ty());
        let mut d0 = kind_code(r.kind);
        if r.class == ResourceClass::Uav {
            d0 |= 0x1000;
            if r.rasterizer_ordered {
                d0 |= 0x2000;
            }
            if r.globally_coherent {
                d0 |= 0x4000;
            }
            if r.has_counter {
                d0 |= 0x8000;
            }
        }
        if r.class == ResourceClass::Sampler && r.comparison {
            d0 |= 0x8000;
        }
        let d1 = match r.kind {
            ResourceKind::TypedBuffer => {
                let e = r.element.ok_or_else(|| Error::Invalid(format!("typed buffer {n} has no element type")))?;
                component_code(e.component) | (e.count as u32) << 8
            }
            k if k.is_texture() => {
                let e = r.element.ok_or_else(|| Error::Invalid(format!("texture {n} has no element type")))?;
                component_code(e.component) | (e.count as u32) << 8
            }
            ResourceKind::StructuredBuffer => r.stride,
            ResourceKind::CBuffer => r.size_bytes,
            ResourceKind::RawBuffer | ResourceKind::Sampler => 0,
            _ => unreachable!(),
        };
        let (p0, p1) = (self.i32c(d0), self.i32c(d1));
        let (Ref::Const(p0), Ref::Const(p1)) = (p0, p1) else { unreachable!() };
        let props = Ref::Const(self.m.consts.intern(props_t, ConstKind::Aggregate(vec![p0, p1])));
        Ok(self.dx(dxop::ANNOTATE_HANDLE, None, handle, &[(handle, h), (props_t, props)]).expect("handle"))
    }

    fn resource(&self, res: &ResRef) -> Result<&'a Resource, Error> {
        let sh: &'a Shader = self.sh;
        let r = sh.resources.get(res.res as usize).ok_or_else(|| Error::Invalid(format!("resource {} does not exist", res.res)))?;
        match (r.binding, &res.index) {
            (Binding::Bound, Some(_)) if r.range_size == 1 => Err(Error::Invalid(format!("resource {} is bound with a range of one, it takes no index", res.res))),
            (Binding::Heap, None) => Err(Error::Invalid(format!("resource {} is a heap resource, it needs a heap index", res.res))),
            _ => Ok(r),
        }
    }

    /// The annotated handle for a use: the one made in the entry block for a bound resource,
    /// `createHandleFromHeap` then `annotateHandle` at the use for a heap resource.
    fn handle(&mut self, res: &ResRef) -> Result<Ref, Error> {
        let decl = self.resource(res)?;
        let Some(index) = &res.index else {
            return Ok(self.handles[res.res as usize].expect("handles are made for every used bound resource"));
        };
        let idx = self.expect(index, Ty::I32, "heap index")?;
        let (handle, i32t, i1t) = (self.handle_ty(), self.m.types.int(32), self.m.types.int(1));
        if decl.binding == Binding::Bound {
            // An element of a bound range, selected at the use.
            let class = match decl.class {
                ResourceClass::Srv => 0,
                ResourceClass::Uav => 1,
                ResourceClass::CBuffer => 2,
                ResourceClass::Sampler => 3,
            };
            let resbind = self.resbind_ty();
            let bind = self.bind_const(decl, class, resbind);
            let nonuniform = self.int_const(i1t, if res.non_uniform { -1 } else { 0 });
            // The use names the element of the range, DXIL wants the absolute register.
            let register = if decl.lower_bound == 0 {
                idx
            } else {
                let base = self.i32c(decl.lower_bound);
                self.emit(fc::BINOP, vec![LOp::Rel(idx), LOp::Rel(base), LOp::Imm(0)], true).expect("value")
            };
            let h = self.dx(dxop::CREATE_HANDLE_FROM_BINDING, None, handle, &[(resbind, bind), (i32t, register), (i1t, nonuniform)]).expect("handle");
            return self.annotate(res.res as usize, h);
        }
        let is_sampler = decl.class == ResourceClass::Sampler;
        let sampler = self.int_const(i1t, if is_sampler { -1 } else { 0 });
        let nonuniform = self.int_const(i1t, if res.non_uniform { -1 } else { 0 });
        if is_sampler {
            self.flags |= 1 << 31;
            self.features |= 0x400_0000;
            self.sampler_heap = true;
        } else {
            self.flags |= 1 << 30;
            self.features |= 0x200_0000;
            self.resource_heap = true;
        }
        let h = self.dx(dxop::CREATE_HANDLE_FROM_HEAP, None, handle, &[(i32t, idx), (i1t, sampler), (i1t, nonuniform)]).expect("handle");
        self.annotate(res.res as usize, h)
    }

    /// The `(index, offset)` addressing arguments of raw and structured buffers.
    fn buffer_address(&mut self, r: &Resource, index: &Operand, offset: &Option<Operand>) -> Result<(Ref, Ref), Error> {
        let idx = self.expect(index, Ty::I32, "buffer index")?;
        let i32t = self.m.types.int(32);
        let off = match (r.kind, offset) {
            (ResourceKind::RawBuffer, None) => self.undef(i32t),
            (ResourceKind::RawBuffer, Some(_)) => return Err(Error::Invalid("a raw buffer takes no element offset".into())),
            (ResourceKind::StructuredBuffer, None) => self.i32c(0),
            (ResourceKind::StructuredBuffer, Some(o)) => self.expect(o, Ty::I32, "buffer offset")?,
            (ResourceKind::TypedBuffer, None) => self.undef(i32t),
            (ResourceKind::TypedBuffer, Some(_)) => return Err(Error::Invalid("a typed buffer takes no element offset".into())),
            _ => return Err(Error::Invalid("this resource is not a buffer, it cannot be addressed like one".into())),
        };
        Ok((idx, off))
    }

    fn scalar32(t: Ty, what: &str) -> Result<(), Error> {
        if matches!(t, Ty::I32 | Ty::F32) { Ok(()) } else { Err(Error::Invalid(format!("{what}: only I32 and F32 are supported, got {t:?}"))) }
    }

    fn lower_inst(&mut self, inst: &Inst) -> Result<(), Error> {
        let compute = matches!(self.sh.stage, Stage::Compute { .. });
        let compute_only = matches!(
            inst.op,
            Op::ThreadId(..) | Op::FlattenedThreadIdInGroup | Op::GsLoad { .. } | Op::GsStore { .. } | Op::GsAtomic { .. } | Op::GsCmpXchg { .. }
        );
        let compute_only = compute_only || matches!(&inst.op, Op::Barrier { flags } if flags & barrier::SYNC_GROUP != 0);
        let graphics_only = matches!(inst.op, Op::LoadInput { .. } | Op::StoreOutput { .. } | Op::Discard(_));
        if (compute_only && !compute) || (graphics_only && compute) {
            return Err(Error::Invalid(format!("{:?} is not available in this stage", inst.op)));
        }
        let i32t = self.m.types.int(32);
        let i1t = self.m.types.int(1);
        match &inst.op {
            Op::Bin(b, x, y) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let (mut ry, ty_) = self.opnd(y)?;
                if tx != ty_ {
                    return Err(Error::Invalid(format!("{b:?}: operand types differ ({tx:?}, {ty_:?})")));
                }
                use BinOp::*;
                let float = matches!(b, FAdd | FSub | FMul | FDiv | FRem);
                if float && tx != Ty::F32 {
                    return Err(Error::Invalid(format!("{b:?} needs F32 operands, got {tx:?}")));
                }
                if !float && !(matches!(tx, Ty::I32 | Ty::I64) || (tx == Ty::Bool && matches!(b, And | Or | Xor))) {
                    return Err(Error::Invalid(format!("{b:?} needs integer operands, got {tx:?}")));
                }
                if matches!(b, Shl | LShr | AShr) {
                    let bits = if tx == Ty::I64 { 64 } else { 32 };
                    let small = matches!(y, Operand::Const(Const::I32(c)) if *c < bits) || matches!(y, Operand::Const(Const::I64(c)) if *c < bits as u64);
                    if !small {
                        let t = self.ty(tx)?;
                        let mask = self.int_const(t, bits as i64 - 1);
                        ry = self.emit(fc::BINOP, vec![LOp::Rel(ry), LOp::Rel(mask), LOp::Imm(10)], true).expect("value");
                    }
                }
                let opc = match b {
                    Add | FAdd => 0,
                    Sub | FSub => 1,
                    Mul | FMul => 2,
                    UDiv => 3,
                    SDiv | FDiv => 4,
                    URem => 5,
                    SRem | FRem => 6,
                    Shl => 7,
                    LShr => 8,
                    AShr => 9,
                    And => 10,
                    Or => 11,
                    Xor => 12,
                };
                let r = self.emit(fc::BINOP, vec![LOp::Rel(rx), LOp::Rel(ry), LOp::Imm(opc)], true).expect("value");
                self.def(inst, 0, r, tx)
            }
            Op::Un(u, x) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let f32t = self.m.types.float();
                use UnOp::*;
                let (r, t) = match u {
                    FAbs | Saturate | Floor | Ceil | Trunc | RoundNe | Fract | Sqrt | Rsqrt | Exp2 | Log2 | Sin | Cos => {
                        if tx != Ty::F32 {
                            return Err(Error::Invalid(format!("{u:?} needs an F32 operand, got {tx:?}")));
                        }
                        let op = match u {
                            FAbs => dxop::FABS,
                            Saturate => dxop::SATURATE,
                            Floor => dxop::FLOOR,
                            Ceil => dxop::CEIL,
                            Trunc => dxop::TRUNC,
                            RoundNe => dxop::ROUND_NE,
                            Fract => dxop::FRACT,
                            Sqrt => dxop::SQRT,
                            Rsqrt => dxop::RSQRT,
                            Exp2 => dxop::EXP2,
                            Log2 => dxop::LOG2,
                            Sin => dxop::SIN,
                            _ => dxop::COS,
                        };
                        (self.dx(op, Some("f32"), f32t, &[(f32t, rx)]).expect("value"), Ty::F32)
                    }
                    IsNan | IsInf | IsFinite => {
                        if tx != Ty::F32 {
                            return Err(Error::Invalid(format!("{u:?} needs an F32 operand, got {tx:?}")));
                        }
                        let op = match u {
                            IsNan => dxop::IS_NAN,
                            IsInf => dxop::IS_INF,
                            _ => dxop::IS_FINITE,
                        };
                        (self.dx(op, Some("f32"), i1t, &[(f32t, rx)]).expect("value"), Ty::Bool)
                    }
                    BitReverse | CountBits | FirstBitLow | FirstBitHigh | FirstBitHighSigned => {
                        if !matches!(tx, Ty::I32 | Ty::I64) {
                            return Err(Error::Invalid(format!("{u:?} needs an integer operand, got {tx:?}")));
                        }
                        let t = self.ty(tx)?;
                        match u {
                            BitReverse => (self.dx(dxop::BIT_REVERSE, Some(suffix(tx)), t, &[(t, rx)]).expect("value"), tx),
                            _ => {
                                let op = match u {
                                    CountBits => dxop::COUNT_BITS,
                                    FirstBitLow => dxop::FIRST_BIT_LOW,
                                    FirstBitHigh => dxop::FIRST_BIT_HIGH,
                                    _ => dxop::FIRST_BIT_HIGH_SIGNED,
                                };
                                (self.dx(op, Some(suffix(tx)), i32t, &[(t, rx)]).expect("value"), Ty::I32)
                            }
                        }
                    }
                    F32ToF16Bits => {
                        if tx != Ty::F32 {
                            return Err(Error::Invalid(format!("{u:?} needs an F32 operand, got {tx:?}")));
                        }
                        (self.dx(dxop::LEGACY_F32_TO_F16, None, i32t, &[(f32t, rx)]).expect("value"), Ty::I32)
                    }
                    DdxCoarse | DdyCoarse | DdxFine | DdyFine => {
                        if tx != Ty::F32 {
                            return Err(Error::Invalid(format!("{u:?} needs an F32 operand, got {tx:?}")));
                        }
                        let op = match u {
                            DdxCoarse => dxop::DDX_COARSE,
                            DdyCoarse => dxop::DDY_COARSE,
                            DdxFine => dxop::DDX_FINE,
                            _ => dxop::DDY_FINE,
                        };
                        (self.dx(op, Some("f32"), f32t, &[(f32t, rx)]).expect("value"), Ty::F32)
                    }
                    F16BitsToF32 => {
                        if tx != Ty::I32 {
                            return Err(Error::Invalid(format!("{u:?} needs an I32 operand, got {tx:?}")));
                        }
                        (self.dx(dxop::LEGACY_F16_TO_F32, None, f32t, &[(i32t, rx)]).expect("value"), Ty::F32)
                    }
                };
                self.def(inst, 0, r, t)
            }
            Op::MinMax(m, x, y) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let ry = self.expect(y, tx, "min/max operands")?;
                let float = matches!(m, MinMaxOp::FMin | MinMaxOp::FMax);
                if (float && tx != Ty::F32) || (!float && !matches!(tx, Ty::I32 | Ty::I64)) {
                    return Err(Error::Invalid(format!("{m:?} does not take {tx:?}")));
                }
                let op = match m {
                    MinMaxOp::FMin => dxop::FMIN,
                    MinMaxOp::FMax => dxop::FMAX,
                    MinMaxOp::UMin => dxop::UMIN,
                    MinMaxOp::UMax => dxop::UMAX,
                    MinMaxOp::SMin => dxop::SMIN,
                    MinMaxOp::SMax => dxop::SMAX,
                };
                let t = self.ty(tx)?;
                let r = self.dx(op, Some(suffix(tx)), t, &[(t, rx), (t, ry)]).expect("value");
                self.def(inst, 0, r, tx)
            }
            Op::Mad(m, x, y, z) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let ry = self.expect(y, tx, "mad operands")?;
                let rz = self.expect(z, tx, "mad operands")?;
                let (op, want) = match m {
                    MadOp::FMad => (dxop::FMAD, &[Ty::F32][..]),
                    MadOp::IMad => (dxop::IMAD, &[Ty::I32, Ty::I64][..]),
                    MadOp::UMad => (dxop::UMAD, &[Ty::I32, Ty::I64][..]),
                };
                if !want.contains(&tx) {
                    return Err(Error::Invalid(format!("{m:?} does not take {tx:?}")));
                }
                let t = self.ty(tx)?;
                let r = self.dx(op, Some(suffix(tx)), t, &[(t, rx), (t, ry), (t, rz)]).expect("value");
                self.def(inst, 0, r, tx)
            }
            Op::BitfieldExtract { signed, width, offset, value } => {
                Self::check_dsts(inst, 1)?;
                let (w, o, v) = (self.expect(width, Ty::I32, "bfe width")?, self.expect(offset, Ty::I32, "bfe offset")?, self.expect(value, Ty::I32, "bfe value")?);
                let op = if *signed { dxop::BFE_SIGNED } else { dxop::BFE_UNSIGNED };
                let r = self.dx(op, Some("i32"), i32t, &[(i32t, w), (i32t, o), (i32t, v)]).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::BitfieldInsert { width, offset, insert, base } => {
                Self::check_dsts(inst, 1)?;
                let w = self.expect(width, Ty::I32, "bfi width")?;
                let o = self.expect(offset, Ty::I32, "bfi offset")?;
                let i = self.expect(insert, Ty::I32, "bfi insert")?;
                let b = self.expect(base, Ty::I32, "bfi base")?;
                let r = self.dx(dxop::BFI, Some("i32"), i32t, &[(i32t, w), (i32t, o), (i32t, i), (i32t, b)]).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::AddCarry(x, y) | Op::SubBorrow(x, y) => {
                Self::check_dsts(inst, 2)?;
                let rx = self.expect(x, Ty::I32, "carry operands")?;
                let ry = self.expect(y, Ty::I32, "carry operands")?;
                let st = self.m.types.named("dx.types.i32c", vec![i32t, i1t]);
                let op = if matches!(inst.op, Op::AddCarry(..)) { dxop::ADD_CARRY } else { dxop::SUB_BORROW };
                let agg = self.dx(op, Some("i32"), st, &[(i32t, rx), (i32t, ry)]).expect("value");
                let (v, c) = (self.extract(agg, 0), self.extract(agg, 1));
                self.def(inst, 0, v, Ty::I32)?;
                self.def(inst, 1, c, Ty::Bool)
            }
            Op::Cast(c, x, to) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let dst = self.ty(*to)?;
                let int = |t: Ty| matches!(t, Ty::I32 | Ty::I64);
                let (opc, ok) = match c {
                    CastOp::Trunc => (0, (int(tx) || tx == Ty::Bool) && (int(*to) || *to == Ty::Bool)),
                    CastOp::ZExt => (1, (int(tx) || tx == Ty::Bool) && int(*to)),
                    CastOp::SExt => (2, (int(tx) || tx == Ty::Bool) && int(*to)),
                    CastOp::FpToUi => (3, tx == Ty::F32 && int(*to)),
                    CastOp::FpToSi => (4, tx == Ty::F32 && int(*to)),
                    CastOp::UiToFp => (5, int(tx) && *to == Ty::F32),
                    CastOp::SiToFp => (6, int(tx) && *to == Ty::F32),
                    CastOp::FpTrunc | CastOp::FpExt => return unsupported("fptrunc and fpext (f16 values)"),
                    CastOp::Bitcast => (11, matches!((tx, *to), (Ty::I32, Ty::F32) | (Ty::F32, Ty::I32))),
                };
                if !ok {
                    return Err(Error::Invalid(format!("{c:?} from {tx:?} to {to:?} is not supported")));
                }
                let r = self.emit(fc::CAST, vec![LOp::Rel(rx), LOp::Imm(dst as u64), LOp::Imm(opc)], true).expect("value");
                self.def(inst, 0, r, *to)
            }
            Op::ICmp(p, x, y) => {
                Self::check_dsts(inst, 1)?;
                let (rx, tx) = self.opnd(x)?;
                let ry = self.expect(y, tx, "icmp operands")?;
                if !matches!(tx, Ty::I32 | Ty::I64 | Ty::Bool) {
                    return Err(Error::Invalid(format!("icmp does not take {tx:?}")));
                }
                use IPred::*;
                let code = match p {
                    Eq => 32,
                    Ne => 33,
                    Ugt => 34,
                    Uge => 35,
                    Ult => 36,
                    Ule => 37,
                    Sgt => 38,
                    Sge => 39,
                    Slt => 40,
                    Sle => 41,
                };
                let r = self.emit(fc::CMP2, vec![LOp::Rel(rx), LOp::Rel(ry), LOp::Imm(code)], true).expect("value");
                self.def(inst, 0, r, Ty::Bool)
            }
            Op::FCmp(p, x, y) => {
                Self::check_dsts(inst, 1)?;
                let rx = self.expect(x, Ty::F32, "fcmp operands")?;
                let ry = self.expect(y, Ty::F32, "fcmp operands")?;
                use FPred::*;
                let code = match p {
                    Oeq => 1,
                    Ogt => 2,
                    Oge => 3,
                    Olt => 4,
                    Ole => 5,
                    One => 6,
                    Ord => 7,
                    Uno => 8,
                    Ueq => 9,
                    Ugt => 10,
                    Uge => 11,
                    Ult => 12,
                    Ule => 13,
                    Une => 14,
                };
                let r = self.emit(fc::CMP2, vec![LOp::Rel(rx), LOp::Rel(ry), LOp::Imm(code)], true).expect("value");
                self.def(inst, 0, r, Ty::Bool)
            }
            Op::Select(c, x, y) => {
                Self::check_dsts(inst, 1)?;
                let rc = self.expect(c, Ty::Bool, "select condition")?;
                let (rx, tx) = self.opnd(x)?;
                let ry = self.expect(y, tx, "select values")?;
                let r = self.emit(fc::VSELECT, vec![LOp::Rel(rx), LOp::Rel(ry), LOp::Rel(rc)], true).expect("value");
                self.def(inst, 0, r, tx)
            }
            Op::ThreadId(kind, comp) => {
                Self::check_dsts(inst, 1)?;
                if *comp > 2 {
                    return Err(Error::Invalid(format!("thread id component {comp}")));
                }
                let op = match kind {
                    ThreadComponentKind::Dispatch => dxop::THREAD_ID,
                    ThreadComponentKind::Group => dxop::GROUP_ID,
                    ThreadComponentKind::InGroup => dxop::THREAD_ID_IN_GROUP,
                };
                let c = self.i32c(*comp as u32);
                let r = self.dx(op, Some("i32"), i32t, &[(i32t, c)]).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::FlattenedThreadIdInGroup => {
                Self::check_dsts(inst, 1)?;
                let r = self.dx(dxop::FLATTENED_THREAD_ID_IN_GROUP, Some("i32"), i32t, &[]).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::CbufLoad { res, row, comps, ty } => {
                Self::check_dsts(inst, comps.len())?;
                let r = self.resource(res)?;
                if r.kind != ResourceKind::CBuffer {
                    return Err(Error::Invalid(format!("resource {} is not a constant buffer", res.res)));
                }
                Self::scalar32(*ty, "CbufLoad")?;
                if comps.iter().any(|&c| c > 3) {
                    return Err(Error::Invalid("a constant buffer row has 4 components".into()));
                }
                let row = self.expect(row, Ty::I32, "cbuffer row")?;
                let ret = self.cbufret_ty(*ty)?;
                let handle = self.handle_ty();
                let h = self.handle(res)?;
                let agg = self.dx(dxop::CBUFFER_LOAD, Some(suffix(*ty)), ret, &[(handle, h), (i32t, row)]).expect("value");
                for (i, &c) in comps.iter().enumerate() {
                    let v = self.extract(agg, c as u64);
                    self.def(inst, i, v, *ty)?;
                }
                Ok(())
            }
            Op::BufLoad { res, index, offset, count, ty } => {
                Self::check_dsts(inst, *count as usize)?;
                if !(1..=4).contains(count) {
                    return Err(Error::Invalid("a buffer load reads 1 to 4 components".into()));
                }
                Self::scalar32(*ty, "BufLoad")?;
                let r = self.resource(res)?;
                let (idx, off) = self.buffer_address(r, index, offset)?;
                let handle = self.handle_ty();
                let ret = self.resret_ty(*ty)?;
                let h = self.handle(res)?;
                let agg = match r.kind {
                    ResourceKind::TypedBuffer => {
                        self.check_typed(r, *ty)?;
                        if r.class == ResourceClass::Uav {
                            self.flags |= 1 << 13;
                            self.features |= 0x800;
                        }
                        self.dx(dxop::BUFFER_LOAD, Some(suffix(*ty)), ret, &[(handle, h), (i32t, idx), (i32t, off)])
                    }
                    _ => {
                        let i8t = self.m.types.int(8);
                        let mask = self.i8c((1u32 << count) - 1);
                        let align = self.i32c(4);
                        self.dx(dxop::RAW_BUFFER_LOAD, Some(suffix(*ty)), ret, &[(handle, h), (i32t, idx), (i32t, off), (i8t, mask), (i32t, align)])
                    }
                }
                .expect("value");
                for i in 0..*count as usize {
                    let v = self.extract(agg, i as u64);
                    self.def(inst, i, v, *ty)?;
                }
                Ok(())
            }
            Op::BufStore { res, index, offset, values } => {
                Self::check_dsts(inst, 0)?;
                let r = self.resource(res)?;
                if r.class != ResourceClass::Uav {
                    return Err(Error::Invalid(format!("resource {} is not a UAV", res.res)));
                }
                if !(1..=4).contains(&values.len()) {
                    return Err(Error::Invalid("a buffer store writes 1 to 4 components".into()));
                }
                let (idx, off) = self.buffer_address(r, index, offset)?;
                let (first, vt) = self.opnd(&values[0])?;
                Self::scalar32(vt, "BufStore")?;
                let mut vs = vec![first];
                for v in &values[1..] {
                    vs.push(self.expect(v, vt, "stored values")?);
                }
                let vty = self.ty(vt)?;
                let u = self.undef(vty);
                vs.resize(4, u);
                let void = self.m.types.void();
                let (handle, i8t) = (self.handle_ty(), self.m.types.int(8));
                let h = self.handle(res)?;
                if r.kind == ResourceKind::TypedBuffer {
                    if values.len() != 4 {
                        return Err(Error::Invalid("a typed buffer store writes all 4 components".into()));
                    }
                    self.check_typed(r, vt)?;
                    let mask = self.i8c(15);
                    let args: Vec<(TypeId, Ref)> = [(handle, h), (i32t, idx), (i32t, off)].into_iter().chain(vs.iter().map(|&v| (vty, v))).chain([(i8t, mask)]).collect();
                    self.dx(dxop::BUFFER_STORE, Some(suffix(vt)), void, &args);
                } else {
                    let mask = self.i8c((1u32 << values.len()) - 1);
                    let align = self.i32c(4);
                    let args: Vec<(TypeId, Ref)> =
                        [(handle, h), (i32t, idx), (i32t, off)].into_iter().chain(vs.iter().map(|&v| (vty, v))).chain([(i8t, mask), (i32t, align)]).collect();
                    self.dx(dxop::RAW_BUFFER_STORE, Some(suffix(vt)), void, &args);
                }
                Ok(())
            }
            Op::BufAtomic { op, res, index, offset, value } => {
                Self::check_dsts(inst, 1)?;
                let r = self.resource(res)?;
                let (c0, c1, v) = self.atomic_args(r, index, offset, &[value])?;
                let handle = self.handle_ty();
                let code = self.i32c(dxop::atomic_code(*op));
                let u = self.undef(i32t);
                let h = self.handle(res)?;
                let old = self.dx(dxop::ATOMIC_BIN_OP, Some("i32"), i32t, &[(handle, h), (i32t, code), (i32t, c0), (i32t, c1), (i32t, u), (i32t, v[0])]).expect("value");
                self.def(inst, 0, old, Ty::I32)
            }
            Op::BufCmpXchg { res, index, offset, compare, new } => {
                Self::check_dsts(inst, 1)?;
                let r = self.resource(res)?;
                let (c0, c1, v) = self.atomic_args(r, index, offset, &[compare, new])?;
                let handle = self.handle_ty();
                let u = self.undef(i32t);
                let h = self.handle(res)?;
                let old = self.dx(dxop::ATOMIC_COMPARE_EXCHANGE, Some("i32"), i32t, &[(handle, h), (i32t, c0), (i32t, c1), (i32t, u), (i32t, v[0]), (i32t, v[1])]).expect("value");
                self.def(inst, 0, old, Ty::I32)
            }
            Op::GsLoad { var, index } => {
                Self::check_dsts(inst, 1)?;
                let (ptr, et, ty) = self.gs_element(*var, index)?;
                let r = self.emit(fc::LOAD, vec![LOp::Rel(ptr), LOp::Imm(et as u64), LOp::Imm(3), LOp::Imm(0)], true).expect("value");
                self.def(inst, 0, r, ty)
            }
            Op::GsStore { var, index, value } => {
                Self::check_dsts(inst, 0)?;
                let (ptr, _, ty) = self.gs_element(*var, index)?;
                let v = self.expect(value, ty, "groupshared store value")?;
                self.emit(fc::STORE, vec![LOp::Rel(ptr), LOp::Rel(v), LOp::Imm(3), LOp::Imm(0)], false);
                Ok(())
            }
            Op::GsAtomic { op, var, index, value } => {
                Self::check_dsts(inst, 1)?;
                let (ptr, _, ty) = self.gs_element(*var, index)?;
                if ty != Ty::I32 {
                    return Err(Error::Invalid("groupshared atomics need an I32 variable".into()));
                }
                let v = self.expect(value, Ty::I32, "groupshared atomic value")?;
                use GsAtomicOp::*;
                let opc = match op {
                    Xchg => 0,
                    Add => 1,
                    Sub => 2,
                    And => 3,
                    Or => 5,
                    Xor => 6,
                    SMax => 7,
                    SMin => 8,
                    UMax => 9,
                    UMin => 10,
                };
                let r = self.emit(fc::ATOMICRMW, vec![LOp::Rel(ptr), LOp::Rel(v), LOp::Imm(opc), LOp::Imm(0), LOp::Imm(6), LOp::Imm(1)], true).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::GsCmpXchg { var, index, compare, new } => {
                Self::check_dsts(inst, 1)?;
                let (ptr, _, ty) = self.gs_element(*var, index)?;
                if ty != Ty::I32 {
                    return Err(Error::Invalid("groupshared atomics need an I32 variable".into()));
                }
                let c = self.expect(compare, Ty::I32, "compare value")?;
                let n = self.expect(new, Ty::I32, "new value")?;
                let agg = self.emit(fc::CMPXCHG, vec![LOp::Rel(ptr), LOp::Rel(c), LOp::Rel(n), LOp::Imm(0), LOp::Imm(6), LOp::Imm(1), LOp::Imm(6), LOp::Imm(0)], true).expect("value");
                self.m.types.anon(vec![i32t, i1t]);
                let old = self.extract(agg, 0);
                self.def(inst, 0, old, Ty::I32)
            }
            Op::WaveLaneIndex | Op::WaveLaneCount | Op::WaveIsFirstLane => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (op, ret, t) = match inst.op {
                    Op::WaveLaneIndex => (dxop::WAVE_LANE_INDEX, i32t, Ty::I32),
                    Op::WaveLaneCount => (dxop::WAVE_LANE_COUNT, i32t, Ty::I32),
                    _ => (dxop::WAVE_IS_FIRST_LANE, i1t, Ty::Bool),
                };
                let r = self.dx(op, None, ret, &[]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::WaveAnyTrue(c) | Op::WaveAllTrue(c) => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let c = self.expect(c, Ty::Bool, "wave condition")?;
                let op = if matches!(inst.op, Op::WaveAnyTrue(_)) { dxop::WAVE_ANY_TRUE } else { dxop::WAVE_ALL_TRUE };
                let r = self.dx(op, None, i1t, &[(i1t, c)]).expect("value");
                self.def(inst, 0, r, Ty::Bool)
            }
            Op::WaveAllEqual(v) => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(v)?;
                Self::wave_value(t, false)?;
                let ti = self.ty(t)?;
                let r = self.dx(dxop::WAVE_ALL_EQUAL, Some(suffix(t)), i1t, &[(ti, rv)]).expect("value");
                self.def(inst, 0, r, Ty::Bool)
            }
            Op::WaveBallot(c) => {
                Self::check_dsts(inst, 4)?;
                self.wave_used();
                let c = self.expect(c, Ty::Bool, "wave condition")?;
                let st = self.m.types.named("dx.types.fouri32", vec![i32t; 4]);
                let agg = self.dx(dxop::WAVE_BALLOT, None, st, &[(i1t, c)]).expect("value");
                for i in 0..4 {
                    let v = self.extract(agg, i as u64);
                    self.def(inst, i, v, Ty::I32)?;
                }
                Ok(())
            }
            Op::WaveReadLaneAt { value, lane } | Op::QuadReadLaneAt { value, lane } => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(value)?;
                Self::wave_value(t, false)?;
                let l = self.expect(lane, Ty::I32, "lane index")?;
                let ti = self.ty(t)?;
                let op = if matches!(inst.op, Op::WaveReadLaneAt { .. }) { dxop::WAVE_READ_LANE_AT } else { dxop::QUAD_READ_LANE_AT };
                let r = self.dx(op, Some(suffix(t)), ti, &[(ti, rv), (i32t, l)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::WaveReadLaneFirst(value) => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(value)?;
                Self::wave_value(t, false)?;
                let ti = self.ty(t)?;
                let r = self.dx(dxop::WAVE_READ_LANE_FIRST, Some(suffix(t)), ti, &[(ti, rv)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::WaveActive { op, signed, value } | Op::WavePrefix { op, signed, value } => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(value)?;
                Self::wave_value(t, false)?;
                let ti = self.ty(t)?;
                let kind = self.i8c(match op {
                    WaveOp::Sum => 0,
                    WaveOp::Product => 1,
                    WaveOp::Min => 2,
                    WaveOp::Max => 3,
                });
                let sign = self.i8c(if *signed { 0 } else { 1 });
                let i8t = self.m.types.int(8);
                let dxo = if matches!(inst.op, Op::WaveActive { .. }) { dxop::WAVE_ACTIVE_OP } else { dxop::WAVE_PREFIX_OP };
                let r = self.dx(dxo, Some(suffix(t)), ti, &[(ti, rv), (i8t, kind), (i8t, sign)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::WaveActiveBit { op, value } => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(value)?;
                Self::wave_value(t, true)?;
                let ti = self.ty(t)?;
                let kind = self.i8c(match op {
                    WaveBitOp::And => 0,
                    WaveBitOp::Or => 1,
                    WaveBitOp::Xor => 2,
                });
                let i8t = self.m.types.int(8);
                let r = self.dx(dxop::WAVE_ACTIVE_BIT, Some(suffix(t)), ti, &[(ti, rv), (i8t, kind)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::WaveActiveCountBits(c) | Op::WavePrefixCountBits(c) => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let c = self.expect(c, Ty::Bool, "wave condition")?;
                let op = if matches!(inst.op, Op::WaveActiveCountBits(_)) { dxop::WAVE_ALL_BIT_COUNT } else { dxop::WAVE_PREFIX_BIT_COUNT };
                let r = self.dx(op, None, i32t, &[(i1t, c)]).expect("value");
                self.def(inst, 0, r, Ty::I32)
            }
            Op::QuadOp { kind, value } => {
                Self::check_dsts(inst, 1)?;
                self.wave_used();
                let (rv, t) = self.opnd(value)?;
                Self::wave_value(t, false)?;
                let ti = self.ty(t)?;
                let k = self.i8c(match kind {
                    QuadKind::AcrossX => 0,
                    QuadKind::AcrossY => 1,
                    QuadKind::Diagonal => 2,
                });
                let i8t = self.m.types.int(8);
                let r = self.dx(dxop::QUAD_OP, Some(suffix(t)), ti, &[(ti, rv), (i8t, k)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::LoadInput { elem, row, col } => {
                Self::check_dsts(inst, 1)?;
                let e = self.sh.inputs.get(*elem as usize).ok_or_else(|| Error::Invalid(format!("input element {elem} does not exist")))?;
                if *col >= e.cols {
                    return Err(Error::Invalid(format!("column {col} of input element {elem} ({}), which has {} columns", e.name, e.cols)));
                }
                let t = if e.component == ComponentType::F32 { Ty::F32 } else { Ty::I32 };
                let ti = self.ty(t)?;
                let (el, row, c, u) = (self.i32c(*elem), self.expect(row, Ty::I32, "input row")?, self.i8c(*col as u32), self.undef(i32t));
                let i8t = self.m.types.int(8);
                let r = self.dx(dxop::LOAD_INPUT, Some(suffix(t)), ti, &[(i32t, el), (i32t, row), (i8t, c), (i32t, u)]).expect("value");
                self.def(inst, 0, r, t)
            }
            Op::StoreOutput { elem, row, col, value } => {
                Self::check_dsts(inst, 0)?;
                let e = self.sh.outputs.get(*elem as usize).ok_or_else(|| Error::Invalid(format!("output element {elem} does not exist")))?;
                if *col >= e.cols {
                    return Err(Error::Invalid(format!("column {col} of output element {elem} ({}), which has {} columns", e.name, e.cols)));
                }
                let t = if e.component == ComponentType::F32 { Ty::F32 } else { Ty::I32 };
                let ti = self.ty(t)?;
                let v = self.expect(value, t, "stored output value")?;
                let (el, row, c) = (self.i32c(*elem), self.expect(row, Ty::I32, "output row")?, self.i8c(*col as u32));
                let (i8t, void) = (self.m.types.int(8), self.m.types.void());
                self.dx(dxop::STORE_OUTPUT, Some(suffix(t)), void, &[(i32t, el), (i32t, row), (i8t, c), (ti, v)]);
                Ok(())
            }
            Op::Discard(c) => {
                Self::check_dsts(inst, 0)?;
                if !matches!(self.sh.stage, Stage::Pixel) {
                    return Err(Error::Invalid("discard is only available in pixel shaders".into()));
                }
                let c = self.expect(c, Ty::Bool, "discard condition")?;
                let void = self.m.types.void();
                self.dx(dxop::DISCARD, None, void, &[(i1t, c)]);
                Ok(())
            }
            Op::TexLoad { .. } | Op::TexStore { .. } | Op::TexAtomic { .. } | Op::TexCmpXchg { .. } | Op::Sample { .. } | Op::Gather { .. } | Op::GetDimensions { .. } | Op::CalculateLod { .. } => self.lower_tex(inst),
            Op::IsHelperLane => {
                Self::check_dsts(inst, 1)?;
                if !matches!(self.sh.stage, Stage::Pixel) {
                    return Err(Error::Invalid("IsHelperLane is only available in pixel shaders".into()));
                }
                let r = self.dx(dxop::IS_HELPER_LANE, Some("i1"), i1t, &[]).expect("value");
                self.def(inst, 0, r, Ty::Bool)
            }
            Op::Barrier { flags } => {
                Self::check_dsts(inst, 0)?;
                if *flags == 0 || flags & !0xF != 0 {
                    return Err(Error::Invalid(format!("barrier flags {flags:#x}")));
                }
                let void = self.m.types.void();
                let f = self.i32c(*flags);
                self.dx(dxop::BARRIER, None, void, &[(i32t, f)]);
                Ok(())
            }
        }
    }

    /// Operands padded with `undef` of type `t` up to `n`.
    fn padded(&mut self, ops: &[Operand], want: Ty, n: usize) -> Result<Vec<Ref>, Error> {
        let mut out = Vec::new();
        for o in ops {
            out.push(self.expect(o, want, "texture coordinate")?);
        }
        let ti = self.ty(want)?;
        let u = self.undef(ti);
        out.resize(n, u);
        Ok(out)
    }

    /// Constant offsets as `i32` arguments, `n` of them: the given ones, zeros for the dimensions of the
    /// texture when none are given, `undef` after.
    fn offsets(&mut self, kind: ResourceKind, offsets: &[i32], n: usize) -> Result<Vec<Ref>, Error> {
        let dims = kind.offset_count();
        if !offsets.is_empty() && offsets.len() != dims {
            return Err(Error::Invalid(format!("{kind:?} takes {dims} offsets, got {}", offsets.len())));
        }
        let i32t = self.m.types.int(32);
        let mut out = Vec::new();
        for d in 0..dims.min(n) {
            out.push(self.i32c(offsets.get(d).copied().unwrap_or(0) as u32));
        }
        let u = self.undef(i32t);
        out.resize(n, u);
        Ok(out)
    }

    fn texel_ty(t: Ty) -> Result<(), Error> {
        if matches!(t, Ty::F32 | Ty::I32) { Ok(()) } else { Err(Error::Invalid(format!("texel type {t:?} is not supported, use F32 or I32"))) }
    }

    fn lower_tex(&mut self, inst: &Inst) -> Result<(), Error> {
        let (i32t, i1t, f32t, i8t) = (self.m.types.int(32), self.m.types.int(1), self.m.types.float(), self.m.types.int(8));
        let handle = self.handle_ty();
        let need_texture = |r: &Resource, uav_ok: bool, what: &str| -> Result<(), Error> {
            if !r.kind.is_texture() || !(r.class == ResourceClass::Srv || (uav_ok && r.class == ResourceClass::Uav)) {
                return Err(Error::Invalid(format!("{what} needs a texture {}", if uav_ok { "SRV or UAV" } else { "SRV" })));
            }
            Ok(())
        };
        let need_sampler = |r: &Resource| -> Result<(), Error> {
            if r.class == ResourceClass::Sampler { Ok(()) } else { Err(Error::Invalid("this operation needs a sampler".into())) }
        };
        match &inst.op {
            Op::TexLoad { tex, mip, coords, offsets, ty } => {
                Self::check_dsts(inst, 4)?;
                Self::texel_ty(*ty)?;
                let r = self.resource(tex)?;
                need_texture(r, true, "TexLoad")?;
                if coords.len() != r.kind.coord_count() {
                    return Err(Error::Invalid(format!("{:?} takes {} coordinates, got {}", r.kind, r.kind.coord_count(), coords.len())));
                }
                self.check_texel(r, *ty)?;
                if r.class == ResourceClass::Uav {
                    self.flags |= 1 << 13;
                    self.features |= 0x800;
                }
                let h = self.handle(tex)?;
                let m = match mip {
                    // A UAV has one level: its load takes no mip level, level 0 means none.
                    Some(Operand::Const(Const::I32(0))) if r.class == ResourceClass::Uav => self.undef(i32t),
                    Some(_) if r.class == ResourceClass::Uav => return Err(Error::Invalid("a UAV load takes no mip level".into())),
                    Some(m) => self.expect(m, Ty::I32, "mip level")?,
                    None => self.undef(i32t),
                };
                let c = self.padded(coords, Ty::I32, 3)?;
                let o = if r.class == ResourceClass::Uav {
                    if !offsets.is_empty() {
                        return Err(Error::Invalid("a UAV load takes no offsets".into()));
                    }
                    let u = self.undef(i32t);
                    vec![u; 3]
                } else {
                    self.offsets(r.kind, offsets, 3)?
                };
                let ret = self.resret_ty(*ty)?;
                let agg = self.dx(dxop::TEXTURE_LOAD, Some(suffix(*ty)), ret, &[(handle, h), (i32t, m), (i32t, c[0]), (i32t, c[1]), (i32t, c[2]), (i32t, o[0]), (i32t, o[1]), (i32t, o[2])]).expect("value");
                for i in 0..4 {
                    let v = self.extract(agg, i as u64);
                    self.def(inst, i, v, *ty)?;
                }
                Ok(())
            }
            Op::TexAtomic { op, tex, coords, value } => {
                Self::check_dsts(inst, 1)?;
                let r = self.resource(tex)?;
                if !r.kind.is_texture() || r.class != ResourceClass::Uav || coords.len() != r.kind.coord_count() {
                    return Err(Error::Invalid("TexAtomic needs a texture UAV and its coordinates".into()));
                }
                let h = self.handle(tex)?;
                let c = self.padded(coords, Ty::I32, 3)?;
                let v = self.expect(value, Ty::I32, "atomic operand")?;
                let code = self.i32c(dxop::atomic_code(*op));
                let old = self.dx(dxop::ATOMIC_BIN_OP, Some("i32"), i32t, &[(handle, h), (i32t, code), (i32t, c[0]), (i32t, c[1]), (i32t, c[2]), (i32t, v)]).expect("value");
                self.def(inst, 0, old, Ty::I32)
            }
            Op::TexCmpXchg { tex, coords, compare, new } => {
                Self::check_dsts(inst, 1)?;
                let r = self.resource(tex)?;
                if !r.kind.is_texture() || r.class != ResourceClass::Uav || coords.len() != r.kind.coord_count() {
                    return Err(Error::Invalid("TexCmpXchg needs a texture UAV and its coordinates".into()));
                }
                let h = self.handle(tex)?;
                let c = self.padded(coords, Ty::I32, 3)?;
                let (cmp, new) = (self.expect(compare, Ty::I32, "compare value")?, self.expect(new, Ty::I32, "new value")?);
                let old = self.dx(dxop::ATOMIC_COMPARE_EXCHANGE, Some("i32"), i32t, &[(handle, h), (i32t, c[0]), (i32t, c[1]), (i32t, c[2]), (i32t, cmp), (i32t, new)]).expect("value");
                self.def(inst, 0, old, Ty::I32)
            }
            Op::TexStore { tex, coords, values } => {
                Self::check_dsts(inst, 0)?;
                let r = self.resource(tex)?;
                if !r.kind.is_texture() || r.class != ResourceClass::Uav {
                    return Err(Error::Invalid("TexStore needs a texture UAV".into()));
                }
                if coords.len() != r.kind.coord_count() || values.len() != 4 {
                    return Err(Error::Invalid(format!("TexStore of {:?} takes {} coordinates and 4 values", r.kind, r.kind.coord_count())));
                }
                let (first, vt) = self.opnd(&values[0])?;
                Self::texel_ty(vt)?;
                self.check_texel(r, vt)?;
                let mut vs = vec![first];
                for v in &values[1..] {
                    vs.push(self.expect(v, vt, "stored texel")?);
                }
                let vty = self.ty(vt)?;
                let h = self.handle(tex)?;
                let c = self.padded(coords, Ty::I32, 3)?;
                let mask = self.i8c(15);
                let void = self.m.types.void();
                let mut args = vec![(handle, h), (i32t, c[0]), (i32t, c[1]), (i32t, c[2])];
                args.extend(vs.iter().map(|&v| (vty, v)));
                args.push((i8t, mask));
                self.dx(dxop::TEXTURE_STORE, Some(suffix(vt)), void, &args);
                Ok(())
            }
            Op::Sample { tex, sampler, coords, offsets, mode, clamp, ty } => {
                Self::check_dsts(inst, 4)?;
                Self::texel_ty(*ty)?;
                let r = self.resource(tex)?;
                need_texture(r, false, "Sample")?;
                need_sampler(self.resource(sampler)?)?;
                if coords.len() != r.kind.coord_count() {
                    return Err(Error::Invalid(format!("{:?} takes {} coordinates, got {}", r.kind, r.kind.coord_count(), coords.len())));
                }
                self.check_texel(r, *ty)?;
                let (th, sh) = (self.handle(tex)?, self.handle(sampler)?);
                let c = self.padded(coords, Ty::F32, 4)?;
                let o = self.offsets(r.kind, offsets, 3)?;
                let u = self.undef(f32t);
                let clamp = match clamp {
                    Some(c) => self.expect(c, Ty::F32, "clamp")?,
                    None => u,
                };
                let ret = self.resret_ty(*ty)?;
                let mut args = vec![(handle, th), (handle, sh)];
                args.extend(c.iter().map(|&x| (f32t, x)));
                args.extend(o.iter().map(|&x| (i32t, x)));
                let dxo = match mode {
                    SampleMode::Implicit => {
                        args.push((f32t, clamp));
                        dxop::SAMPLE
                    }
                    SampleMode::Bias(b) => {
                        let b = self.expect(b, Ty::F32, "bias")?;
                        args.push((f32t, b));
                        args.push((f32t, clamp));
                        dxop::SAMPLE_BIAS
                    }
                    SampleMode::Level(l) => {
                        let l = self.expect(l, Ty::F32, "level")?;
                        args.push((f32t, l));
                        dxop::SAMPLE_LEVEL
                    }
                    SampleMode::Grad { ddx, ddy } => {
                        let n = r.kind.coord_count() - usize::from(matches!(r.kind, ResourceKind::Texture1DArray | ResourceKind::Texture2DArray | ResourceKind::TextureCubeArray));
                        if ddx.len() != n || ddy.len() != n {
                            return Err(Error::Invalid(format!("SampleGrad of {:?} takes {n} derivatives of each kind", r.kind)));
                        }
                        let dx = self.padded(ddx, Ty::F32, 3)?;
                        let dy = self.padded(ddy, Ty::F32, 3)?;
                        args.extend(dx.iter().map(|&x| (f32t, x)));
                        args.extend(dy.iter().map(|&x| (f32t, x)));
                        args.push((f32t, clamp));
                        dxop::SAMPLE_GRAD
                    }
                    SampleMode::Compare(cmp) => {
                        let cmp = self.expect(cmp, Ty::F32, "compare value")?;
                        args.push((f32t, cmp));
                        args.push((f32t, clamp));
                        dxop::SAMPLE_CMP
                    }
                    SampleMode::CompareLevelZero(cmp) => {
                        let cmp = self.expect(cmp, Ty::F32, "compare value")?;
                        args.push((f32t, cmp));
                        dxop::SAMPLE_CMP_LEVEL_ZERO
                    }
                };
                let agg = self.dx(dxo, Some(suffix(*ty)), ret, &args).expect("value");
                for i in 0..4 {
                    let v = self.extract(agg, i as u64);
                    self.def(inst, i, v, *ty)?;
                }
                Ok(())
            }
            Op::Gather { tex, sampler, coords, offsets, channel, compare, ty } => {
                Self::check_dsts(inst, 4)?;
                Self::texel_ty(*ty)?;
                let r = self.resource(tex)?;
                need_texture(r, false, "Gather")?;
                need_sampler(self.resource(sampler)?)?;
                if !matches!(r.kind, ResourceKind::Texture2D | ResourceKind::Texture2DArray | ResourceKind::TextureCube | ResourceKind::TextureCubeArray) || coords.len() != r.kind.coord_count() || *channel > 3 {
                    return Err(Error::Invalid(format!("Gather of {:?} with {} coordinates and channel {channel}", r.kind, coords.len())));
                }
                self.check_texel(r, *ty)?;
                let (th, sh) = (self.handle(tex)?, self.handle(sampler)?);
                let c = self.padded(coords, Ty::F32, 4)?;
                let o = self.offsets(r.kind, offsets, 2)?;
                let ch = self.i32c(*channel as u32);
                let ret = self.resret_ty(*ty)?;
                let mut args = vec![(handle, th), (handle, sh)];
                args.extend(c.iter().map(|&x| (f32t, x)));
                args.extend(o.iter().map(|&x| (i32t, x)));
                args.push((i32t, ch));
                let dxo = match compare {
                    Some(cmp) => {
                        let cmp = self.expect(cmp, Ty::F32, "compare value")?;
                        args.push((f32t, cmp));
                        dxop::TEXTURE_GATHER_CMP
                    }
                    None => dxop::TEXTURE_GATHER,
                };
                let agg = self.dx(dxo, Some(suffix(*ty)), ret, &args).expect("value");
                for i in 0..4 {
                    let v = self.extract(agg, i as u64);
                    self.def(inst, i, v, *ty)?;
                }
                Ok(())
            }
            Op::GetDimensions { tex, mip } => {
                Self::check_dsts(inst, 4)?;
                let r = self.resource(tex)?;
                // A raw buffer answers with its size in bytes.
                let raw = r.kind == ResourceKind::RawBuffer && matches!(r.class, ResourceClass::Srv | ResourceClass::Uav);
                if !raw {
                    need_texture(r, true, "GetDimensions")?;
                }
                let h = self.handle(tex)?;
                let m = match mip {
                    _ if raw => self.undef(i32t),
                    Some(m) => self.expect(m, Ty::I32, "mip level")?,
                    None if r.class == ResourceClass::Srv => self.i32c(0),
                    None => self.undef(i32t),
                };
                let st = self.m.types.named("dx.types.Dimensions", vec![i32t; 4]);
                let agg = self.dx(dxop::GET_DIMENSIONS, None, st, &[(handle, h), (i32t, m)]).expect("value");
                // Only the components the kind defines can be read, the others are 0: width, height,
                // depth or array size, mip count (SRVs only).
                use ResourceKind::*;
                let mut valid = match r.kind {
                    Texture1D => [true, false, false, true],
                    Texture1DArray | Texture2D | TextureCube => [true, true, false, true],
                    Texture2DArray | Texture3D | TextureCubeArray => [true; 4],
                    RawBuffer => [true, false, false, false],
                    _ => [false; 4],
                };
                if r.class == ResourceClass::Uav {
                    valid[3] = false;
                }
                for (i, ok) in valid.into_iter().enumerate() {
                    let v = if ok { self.extract(agg, i as u64) } else { self.i32c(0) };
                    self.def(inst, i, v, Ty::I32)?;
                }
                Ok(())
            }
            Op::CalculateLod { tex, sampler, coords, clamped } => {
                Self::check_dsts(inst, 1)?;
                let r = self.resource(tex)?;
                need_texture(r, false, "CalculateLod")?;
                need_sampler(self.resource(sampler)?)?;
                let arrays = usize::from(matches!(r.kind, ResourceKind::Texture1DArray | ResourceKind::Texture2DArray | ResourceKind::TextureCubeArray));
                if coords.len() != r.kind.coord_count() - arrays {
                    return Err(Error::Invalid(format!("CalculateLod of {:?} takes {} coordinates", r.kind, r.kind.coord_count() - arrays)));
                }
                let (th, sh) = (self.handle(tex)?, self.handle(sampler)?);
                let c = self.padded(coords, Ty::F32, 3)?;
                let cl = self.int_const(i1t, if *clamped { -1 } else { 0 });
                let v = self.dx(dxop::CALCULATE_LOD, Some("f32"), f32t, &[(handle, th), (handle, sh), (f32t, c[0]), (f32t, c[1]), (f32t, c[2]), (i1t, cl)]).expect("value");
                self.def(inst, 0, v, Ty::F32)
            }
            _ => unreachable!("lower_tex is called for texture operations only"),
        }
    }

    /// A texel access must match the component type of the resource.
    fn check_texel(&self, r: &Resource, ty: Ty) -> Result<(), Error> {
        let e = r.element.ok_or_else(|| Error::Invalid("texture without element type".into()))?;
        let ok = match e.component {
            ComponentType::F32 => ty == Ty::F32,
            ComponentType::I32 | ComponentType::U32 => ty == Ty::I32,
        };
        if ok { Ok(()) } else { Err(Error::Invalid(format!("texture element {:?} does not match {ty:?} access", e.component))) }
    }

    /// Wave operations set the wave feature bits.
    fn wave_used(&mut self) {
        self.flags |= 1 << 19;
        self.features |= 0x4000;
    }

    fn wave_value(t: Ty, bits_only: bool) -> Result<(), Error> {
        let ok = if bits_only { matches!(t, Ty::I32 | Ty::I64) } else { matches!(t, Ty::I32 | Ty::I64 | Ty::F32) };
        if ok { Ok(()) } else { Err(Error::Invalid(format!("a wave operation does not take {t:?}"))) }
    }

    fn check_typed(&self, r: &Resource, ty: Ty) -> Result<(), Error> {
        let e = r.element.ok_or_else(|| Error::Invalid("typed buffer without element type".into()))?;
        let ok = match e.component {
            ComponentType::F32 => ty == Ty::F32,
            ComponentType::I32 | ComponentType::U32 => ty == Ty::I32,
        };
        if ok { Ok(()) } else { Err(Error::Invalid(format!("typed buffer element {:?} does not match {ty:?} access", e.component))) }
    }

    /// Address and value arguments of the buffer atomics: `(coord0, coord1, values)`.
    fn atomic_args(&mut self, r: &Resource, index: &Operand, offset: &Option<Operand>, values: &[&Operand]) -> Result<(Ref, Ref, Vec<Ref>), Error> {
        if r.class != ResourceClass::Uav || r.kind == ResourceKind::CBuffer {
            return Err(Error::Invalid("atomics need a UAV".into()));
        }
        let (idx, off) = self.buffer_address(r, index, offset)?;
        let i32t = self.m.types.int(32);
        let c1 = if r.kind == ResourceKind::StructuredBuffer { off } else { self.undef(i32t) };
        let mut vs = Vec::new();
        for v in values {
            vs.push(self.expect(v, Ty::I32, "atomic operand")?);
        }
        Ok((idx, c1, vs))
    }

    /// The address of element `index` of groupshared variable `var`: pointer value, element type, IR type.
    fn gs_element(&mut self, var: u32, index: &Operand) -> Result<(Ref, TypeId, Ty), Error> {
        let sh: &'a Shader = self.sh;
        let decl = sh.groupshared.get(var as usize).ok_or_else(|| Error::Invalid(format!("groupshared variable {var} does not exist")))?;
        let ty = decl.ty;
        let et = self.ty(ty)?;
        let arr = self.m.types.intern(Type::Array(decl.len as u64, et));
        self.m.types.ptr(et, 3);
        let idx = self.expect(index, Ty::I32, "groupshared index")?;
        let zero = self.i32c(0);
        let ptr = self.emit(fc::GEP, vec![LOp::Imm(1), LOp::Imm(arr as u64), LOp::Rel(Ref::Global(self.gs[var as usize])), LOp::Rel(zero), LOp::Rel(idx)], true).expect("value");
        Ok((ptr, et, ty))
    }

    fn lower_blocks(&mut self) -> Result<(), Error> {
        let sh = self.sh;
        self.m.block_count = sh.blocks.len() as u32;
        for (bi, b) in sh.blocks.iter().enumerate() {
            if bi == 0 {
                if !b.phis.is_empty() {
                    return Err(Error::Invalid("the entry block has phis".into()));
                }
                self.make_handles()?;
            }
            for phi in &b.phis {
                let t = self.ty(phi.ty)?;
                let mut ops = vec![LOp::Imm(t as u64)];
                for (blk, o) in &phi.incoming {
                    let r = match o {
                        Operand::Const(c) => {
                            let (r, ct) = self.constant(*c)?;
                            if ct != phi.ty {
                                return Err(Error::Invalid(format!("phi {}: incoming constant of type {ct:?}, expected {:?}", phi.dst.0, phi.ty)));
                            }
                            r
                        }
                        Operand::Value(v) => self.vals.get(v.0).map_or(Ref::Late(v.0), |x| x.0),
                    };
                    ops.push(LOp::SRel(r));
                    ops.push(LOp::Imm(blk.0 as u64));
                }
                let r = self.emit(fc::PHI, ops, true).expect("value");
                if self.vals.insert(phi.dst.0, (r, phi.ty)).is_some() {
                    return Err(Error::Invalid(format!("value {} is defined twice", phi.dst.0)));
                }
            }
            for inst in &b.insts {
                self.lower_inst(inst)?;
            }
            match &b.term {
                Terminator::Br(t) => {
                    self.emit(fc::BR, vec![LOp::Imm(t.0 as u64)], false);
                }
                Terminator::CondBr(c, t, f) => {
                    let rc = self.expect(c, Ty::Bool, "branch condition")?;
                    self.emit(fc::BR, vec![LOp::Imm(t.0 as u64), LOp::Imm(f.0 as u64), LOp::Rel(rc)], false);
                }
                Terminator::Ret => {
                    self.emit(fc::RET, vec![], false);
                }
            }
        }
        // Resolve the phi inputs that were defined later.
        let vals = &self.vals;
        for inst in &mut self.m.body {
            for op in &mut inst.ops {
                if let LOp::SRel(Ref::Late(v)) = op {
                    let (r, _) = vals.get(*v).ok_or_else(|| Error::UseBeforeDef(format!("phi input value {v} is never defined")))?;
                    *op = LOp::SRel(r);
                }
            }
        }
        Ok(())
    }

    fn md_int(&mut self, bits: u32, v: i64) -> MdId {
        let t = self.m.types.int(bits);
        let c = self.int_const(t, v);
        self.m.md.intern(Md::Value(t, c))
    }
    fn md_i32(&mut self, v: u32) -> MdId {
        self.md_int(32, v as i32 as i64)
    }
    fn md_i1(&mut self, v: bool) -> MdId {
        self.md_int(1, if v { -1 } else { 0 })
    }

    fn pair_node(&mut self, tag: u32, value: u32) -> MdId {
        let (t, v) = (self.md_i32(tag), self.md_i32(value));
        self.m.md.node(vec![Some(t), Some(v)])
    }

    fn resource_metadata(&mut self) -> Result<Option<MdId>, Error> {
        let sh = self.sh;
        let mut lists: [Vec<MdId>; 4] = Default::default();
        let mut counts = [0u32; 4];
        for r in sh.resources.iter().filter(|r| r.binding == Binding::Bound) {
            let ci = match r.class {
                ResourceClass::Srv => 0,
                ResourceClass::Uav => 1,
                ResourceClass::CBuffer => 2,
                ResourceClass::Sampler => 3,
            };
            let id = counts[ci];
            counts[ci] += 1;
            let sym_name = match (r.class, r.kind) {
                (ResourceClass::Srv, ResourceKind::RawBuffer) => "struct.ByteAddressBuffer",
                (ResourceClass::Uav, ResourceKind::RawBuffer) => "struct.RWByteAddressBuffer",
                (ResourceClass::Srv, ResourceKind::StructuredBuffer) => "struct.StructuredBuffer",
                (ResourceClass::Uav, ResourceKind::StructuredBuffer) => "struct.RWStructuredBuffer",
                (ResourceClass::Srv, ResourceKind::TypedBuffer) => "class.Buffer",
                (ResourceClass::Uav, ResourceKind::TypedBuffer) => "class.RWBuffer",
                (ResourceClass::Srv, k) if k.is_texture() => "class.Texture",
                (ResourceClass::Uav, k) if k.is_texture() => "class.RWTexture",
                (ResourceClass::Sampler, _) if r.comparison => "struct.SamplerComparisonState",
                (ResourceClass::Sampler, _) => "struct.SamplerState",
                _ => "struct.cbuffer",
            };
            let i32t = self.m.types.int(32);
            let st = self.m.types.named(sym_name, vec![i32t]);
            let pt = self.m.types.ptr(st, 0);
            let u = self.undef(pt);
            let sym = self.m.md.intern(Md::Value(pt, u));
            let name = self.m.md.string(&r.name);
            let mut ops: Vec<Option<MdId>> = vec![Some(self.md_i32(id)), Some(sym), Some(name), Some(self.md_i32(r.space)), Some(self.md_i32(r.lower_bound)), Some(self.md_i32(r.range_size))];
            let extra = match r.kind {
                k if k == ResourceKind::TypedBuffer || k.is_texture() => {
                    let e = r.element.ok_or_else(|| Error::Invalid("typed resource without element type".into()))?;
                    Some(self.pair_node(0, component_code(e.component)))
                }
                ResourceKind::StructuredBuffer => Some(self.pair_node(1, r.stride)),
                _ => None,
            };
            match r.class {
                ResourceClass::Srv => {
                    ops.push(Some(self.md_i32(kind_code(r.kind))));
                    ops.push(Some(self.md_i32(0)));
                    ops.push(extra);
                }
                ResourceClass::Uav => {
                    ops.push(Some(self.md_i32(kind_code(r.kind))));
                    ops.push(Some(self.md_i1(r.globally_coherent)));
                    ops.push(Some(self.md_i1(r.has_counter)));
                    ops.push(Some(self.md_i1(r.rasterizer_ordered)));
                    ops.push(extra);
                }
                ResourceClass::CBuffer => {
                    ops.push(Some(self.md_i32(r.size_bytes)));
                    ops.push(None);
                }
                ResourceClass::Sampler => {
                    ops.push(Some(self.md_i32(r.comparison as u32)));
                    ops.push(None);
                }
            }
            let node = self.m.md.node(ops);
            lists[ci].push(node);
        }
        if sh.resources.iter().all(|r| r.binding == Binding::Heap) {
            return Ok(None);
        }
        let mut top = Vec::new();
        for l in lists {
            top.push(if l.is_empty() { None } else { Some(self.m.md.node(l.into_iter().map(Some).collect())) });
        }
        // The order of the lists is SRVs, UAVs, CBVs, samplers.
        Ok(Some(self.m.md.node(top)))
    }

    fn md_i8(&mut self, v: u32) -> MdId {
        self.md_int(8, v as u8 as i8 as i64)
    }

    /// One signature list: `null` when empty, else a node of element records.
    fn sig_list(&mut self, elems: &'a [SigElement], uses: &[crate::sig::ElemUse]) -> Option<MdId> {
        use crate::sig::{declared_mask, interp_code, semantic_code};
        if elems.is_empty() {
            return None;
        }
        let mut nodes = Vec::new();
        for (i, e) in elems.iter().enumerate() {
            let id = self.md_i32(i as u32);
            let name = self.m.md.string(&e.name);
            let comp = self.md_i8(component_code(e.component));
            let sem = self.md_i8(semantic_code(e.semantic));
            let idx: Vec<Option<MdId>> = (0..e.rows).map(|r| Some(self.md_i32(e.index + r))).collect();
            let idx = self.m.md.node(idx);
            let interp = self.md_i8(interp_code(e.interp));
            let rows = self.md_i32(e.rows);
            let cols = self.md_i8(e.cols as u32);
            let (sr, sc) = if e.allocated() { (self.md_i32(e.start_row), self.md_i8(e.start_col as u32)) } else { (self.md_i32(u32::MAX), self.md_i8(u32::MAX)) };
            let extra = if uses[i].used != 0 { Some(self.pair_node(3, uses[i].used as u32)) } else { None };
            let _ = declared_mask(e);
            nodes.push(Some(self.m.md.node(vec![Some(id), Some(name), Some(comp), Some(sem), Some(idx), Some(interp), Some(rows), Some(cols), Some(sr), Some(sc), extra])));
        }
        Some(self.m.md.node(nodes))
    }

    fn metadata(&mut self) -> Result<(), Error> {
        let sh: &'a Shader = self.sh;
        let (model_name, program) = match sh.stage {
            Stage::Compute { .. } => ("cs", 5),
            Stage::Vertex => ("vs", 1),
            Stage::Pixel => ("ps", 0),
        };
        let _ = program;
        let version = {
            let (a, b) = (self.md_i32(1), self.md_i32(6));
            self.m.md.node(vec![Some(a), Some(b)])
        };
        let model = {
            let cs = self.m.md.string(model_name);
            let (a, b) = (self.md_i32(6), self.md_i32(6));
            self.m.md.node(vec![Some(cs), Some(a), Some(b)])
        };
        let resources = self.resource_metadata()?;
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(Some(self.md_i32(0)));
            props.push(Some(self.md_int(64, self.flags as i64)));
        }
        if let Stage::Compute { threads, wave_size } = sh.stage {
            props.push(Some(self.md_i32(4)));
            let th: Vec<Option<MdId>> = threads.iter().map(|&t| Some(self.md_i32(t))).collect();
            props.push(Some(self.m.md.node(th)));
            if let Some(n) = wave_size {
                props.push(Some(self.md_i32(11)));
                let n = self.md_i32(n);
                props.push(Some(self.m.md.node(vec![Some(n)])));
            }
        }
        let props = if props.is_empty() { None } else { Some(self.m.md.node(props)) };
        let sigs = match self.sig.clone() {
            Some(data) => {
                let (i, o) = (self.sig_list(&sh.inputs, &data.inputs), self.sig_list(&sh.outputs, &data.outputs));
                Some(self.m.md.node(vec![i, o, None]))
            }
            None => None,
        };
        let view = match self.sig.clone() {
            Some(data) => {
                let i32t = self.m.types.int(32);
                let mut words = vec![data.in_scalars, data.out_scalars];
                words.extend_from_slice(&data.table);
                let arr = self.m.types.intern(Type::Array(words.len() as u64, i32t));
                let elems: Vec<ConstId> = words.iter().map(|&w| match self.i32c(w) {
                    Ref::Const(c) => c,
                    _ => unreachable!(),
                }).collect();
                let c = self.m.consts.intern(arr, ConstKind::Aggregate(elems));
                let v = self.m.md.intern(Md::Value(arr, Ref::Const(c)));
                Some(self.m.md.node(vec![Some(v)]))
            }
            None => None,
        };
        let fty = self.m.funcs[0].ty;
        let fptr = self.m.types.ptr(fty, 0);
        let fval = self.m.md.intern(Md::Value(fptr, Ref::Func(0)));
        let main = self.m.md.string("main");
        let entry = self.m.md.node(vec![Some(fval), Some(main), sigs, resources, props]);
        self.m.md.named.push(("dx.version".into(), vec![version]));
        self.m.md.named.push(("dx.valver".into(), vec![version]));
        self.m.md.named.push(("dx.shaderModel".into(), vec![model]));
        if let Some(r) = resources {
            self.m.md.named.push(("dx.resources".into(), vec![r]));
        }
        if let Some(v) = view {
            self.m.md.named.push(("dx.viewIdState".into(), vec![v]));
        }
        self.m.md.named.push(("dx.entryPoints".into(), vec![entry]));
        Ok(())
    }
}

/// Checks the signature declarations of a graphics shader.
fn check_signature(shader: &Shader) -> Result<(), Error> {
    for (what, elems) in [("input", &shader.inputs), ("output", &shader.outputs)] {
        for (i, e) in elems.iter().enumerate() {
            let register_ok = !e.allocated() || (e.start_col <= 3 && u32::from(e.start_col) + u32::from(e.cols) <= 4 && e.start_row <= 31);
            if e.cols == 0 || e.cols > 4 || e.rows == 0 || !register_ok || e.name.is_empty() {
                return Err(Error::Invalid(format!("{what} element {i} ({}): rows {}, columns {} from {}", e.name, e.rows, e.cols, e.start_col)));
            }
        }
    }
    Ok(())
}

/// Lowers a normalised shader.
pub fn lower(shader: &Shader) -> Result<Lowered, Error> {
    let (wave_size, graphics) = match shader.stage {
        Stage::Compute { threads, wave_size } => {
            let [x, y, z] = threads;
            if x == 0 || y == 0 || z == 0 || x > 1024 || y > 1024 || z > 64 || x * y * z > 1024 {
                return Err(Error::Invalid(format!("thread group size {x}x{y}x{z} is outside the limits (1024 threads, 1024 x 1024 x 64)")));
            }
            (wave_size, false)
        }
        Stage::Vertex | Stage::Pixel => {
            if !shader.groupshared.is_empty() {
                return Err(Error::Invalid("groupshared memory is only available in compute shaders".into()));
            }
            (None, true)
        }
    };
    if !graphics && (!shader.inputs.is_empty() || !shader.outputs.is_empty()) {
        return Err(Error::Invalid("a compute shader has no signature".into()));
    }
    let mut max_val = 0u32;
    for b in &shader.blocks {
        for p in &b.phis {
            max_val = max_val.max(p.dst.0 + 1);
        }
        for i in &b.insts {
            for d in &i.dsts {
                max_val = max_val.max(d.0 + 1);
            }
        }
    }
    let mut l = Lower {
        sh: shader,
        m: Module::default(),
        vals: ValMap::new((max_val as usize).min(262_144)),
        next_inst: 0,
        handles: vec![None; shader.resources.len()],
        dx_funcs: HashMap::new(),
        gs: Vec::new(),
        flags: 0,
        features: 0,
        resource_heap: false,
        sampler_heap: false,
        sig: None,
        handle_ty_id: None,
        resbind_ty_id: None,
        props_ty_id: None,
        cbufret_f32_id: None,
        cbufret_i32_id: None,
        resret_f32_id: None,
        resret_i32_id: None,
    };
    // The entry function is function 0.
    let void = l.m.types.void();
    let fty = l.m.types.func(void, vec![]);
    l.m.types.ptr(fty, 0);
    l.m.funcs.push(Func { name: "main".into(), ty: fty, attr: None, proto: false });

    if let Some(n) = wave_size
        && !(n.is_power_of_two() && (4..=128).contains(&n))
    {
        return Err(Error::Invalid(format!("WaveSize {n} is not a power of two from 4 to 128")));
    }
    let mut total_gs = 0u64;
    for (i, g) in shader.groupshared.iter().enumerate() {
        if !matches!(g.ty, Ty::I32 | Ty::F32) {
            return Err(Error::Invalid(format!("groupshared variable {i} must be I32 or F32")));
        }
        total_gs += 4 * g.len as u64;
        let et = l.ty(g.ty)?;
        let arr = l.m.types.intern(Type::Array(g.len as u64, et));
        l.m.types.ptr(arr, 3);
        let init = l.undef(arr);
        let Ref::Const(init) = init else { unreachable!() };
        l.gs.push(l.m.globals.len() as u32);
        l.m.globals.push(Global { name: format!("gs{i}"), ty: arr, addrspace: 3, init: Some(init) });
    }
    if total_gs > 32 * 1024 {
        return Err(Error::Invalid(format!("groupshared memory of {total_gs} bytes exceeds 32 KiB")));
    }
    for (i, r) in shader.resources.iter().enumerate() {
        if r.range_size == 0 {
            return Err(Error::Invalid(format!("resource {i}: an empty range")));
        }
        if r.kind == ResourceKind::CBuffer && (r.class != ResourceClass::CBuffer || r.size_bytes == 0 || r.size_bytes % 16 != 0 || r.size_bytes > 65536) {
            return Err(Error::Invalid(format!("constant buffer {i}: size must be a multiple of 16 up to 65536")));
        }
        if (r.class == ResourceClass::CBuffer) != (r.kind == ResourceKind::CBuffer) || (r.class == ResourceClass::Sampler) != (r.kind == ResourceKind::Sampler) {
            return Err(Error::Invalid(format!("resource {i}: class and kind do not agree")));
        }
        if r.kind.is_texture() && r.class == ResourceClass::Uav && !matches!(r.kind, ResourceKind::Texture1D | ResourceKind::Texture2D | ResourceKind::Texture3D | ResourceKind::Texture1DArray | ResourceKind::Texture2DArray) {
            return Err(Error::Invalid(format!("resource {i}: a UAV cannot be a cube texture")));
        }
        if r.binding == Binding::Bound && matches!(r.kind, ResourceKind::RawBuffer | ResourceKind::StructuredBuffer) {
            l.flags |= 1 << 4;
        }
        if r.kind == ResourceKind::StructuredBuffer && (r.stride == 0 || r.stride % 4 != 0) {
            return Err(Error::Invalid(format!("structured buffer {i}: stride must be a nonzero multiple of 4")));
        }
    }
    // Bound UAVs in a stage other than pixel and compute set the "UAVs at every stage" flag. A UAV of
    // the heap does not, the validator refuses the flag for it ("Flags must match usage").
    if matches!(shader.stage, Stage::Vertex) && shader.resources.iter().any(|r| r.class == ResourceClass::Uav && r.binding == Binding::Bound) {
        l.flags |= 1 << 16;
        l.features |= 0x4;
    }
    // More than eight UAVs set the "64 UAVs" flag.
    if shader.resources.iter().filter(|r| r.class == ResourceClass::Uav && r.binding == Binding::Bound).fold(0u64, |n, r| n + r.range_size as u64) > 8 {
        l.flags |= 1 << 15;
        l.features |= 0x8;
    }
    if graphics {
        check_signature(shader)?;
        l.sig = Some(crate::sig::analyze(shader)?);
        for e in shader.inputs.iter().chain(&shader.outputs) {
            match e.semantic {
                Semantic::StencilRef => {
                    l.flags |= 1 << 11;
                    l.features |= 0x200;
                }
                Semantic::RenderTargetArrayIndex | Semantic::ViewportArrayIndex => {
                    l.flags |= 1 << 9;
                    l.features |= 0x2000;
                }
                Semantic::InnerCoverage => {
                    l.flags |= 1 << 10;
                    l.features |= 0x400;
                }
                Semantic::Barycentrics => {
                    l.flags |= 1 << 22;
                    l.features |= 0x2_0000;
                }
                _ => {}
            }
        }
    }
    l.lower_blocks()?;
    l.metadata()?;

    // PSV0 resources: CBVs, samplers, SRVs, UAVs.
    let mut psv = Vec::new();
    for class in [ResourceClass::CBuffer, ResourceClass::Sampler, ResourceClass::Srv, ResourceClass::Uav] {
        for r in shader.resources.iter().filter(|r| r.class == class && r.binding == Binding::Bound) {
            let ty = match (class, r.kind) {
                (ResourceClass::CBuffer, _) => 2,
                (ResourceClass::Sampler, _) => 1,
                (ResourceClass::Srv, ResourceKind::RawBuffer) => 4,
                (ResourceClass::Srv, ResourceKind::StructuredBuffer) => 5,
                (ResourceClass::Srv, _) => 3,
                (ResourceClass::Uav, ResourceKind::RawBuffer) => 7,
                (ResourceClass::Uav, ResourceKind::TypedBuffer) => 6,
                (ResourceClass::Uav, k) if k.is_texture() => 6,
                (ResourceClass::Uav, _) if r.has_counter => 9,
                (ResourceClass::Uav, _) => 8,
            };
            psv.push(crate::container::PsvResource { ty, space: r.space, lower: r.lower_bound, upper: upper_bound(r), kind: kind_code(r.kind), flags: 0 });
        }
    }
    Ok(Lowered { module: l.m, features: l.features, psv, resource_heap: l.resource_heap, sampler_heap: l.sampler_heap, wave_size, sig: l.sig })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constant index operands of every `createHandleFromBinding` call of the lowered module.
    fn binding_indices(shader: &Shader) -> Vec<u32> {
        let l = lower(shader).expect("lowers");
        let m = &l.module;
        let callee = m.funcs.iter().position(|f| f.name == "dx.op.createHandleFromBinding").expect("the op is declared") as u32;
        let konst = |r: &LOp| match r {
            LOp::Rel(Ref::Const(c)) => match &m.consts.list[*c as usize].1 {
                ConstKind::Int(v) => Some(*v as u32),
                _ => None,
            },
            _ => None,
        };
        m.body.iter().filter(|i| is_binding_call(i, callee)).map(|i| konst(&i.ops[6]).expect("a constant index")).collect()
    }

    /// Call operands: attributes, flags, function type, callee, then opcode, ResBind, index, non uniform.
    fn is_binding_call(i: &LInst, callee: u32) -> bool {
        i.code == fc::CALL && matches!(i.ops.get(3), Some(LOp::Rel(Ref::Func(f))) if *f == callee)
    }

    fn store_of(res: Resource) -> Shader {
        let insts = vec![Inst::new(&[], Op::BufStore { res: ResRef::bound(0), index: Operand::Const(Const::I32(0)), offset: None, values: vec![Operand::Const(Const::I32(7))] })];
        let mut s = Shader::compute([1, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
        s.resources.push(res);
        s
    }

    #[test]
    fn the_handle_index_is_the_register_of_the_binding() {
        for register in [0, 1, 27, 4096] {
            assert_eq!(binding_indices(&store_of(Resource::raw(ResourceClass::Uav, 0, register))), [register], "u{register}");
        }
        assert_eq!(binding_indices(&store_of(Resource::raw(ResourceClass::Uav, 3, 9))), [9], "u9 of space 3");
    }

    #[test]
    fn an_array_element_index_is_offset_by_the_register() {
        let mut array = Resource::raw(ResourceClass::Uav, 0, 8);
        array.range_size = 4;
        let res = ResRef { res: 0, index: Some(Operand::Value(ValueId(0))), non_uniform: false };
        let insts = vec![
            Inst::new(&[0], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
            Inst::new(&[], Op::BufStore { res, index: Operand::Const(Const::I32(0)), offset: None, values: vec![Operand::Const(Const::I32(7))] }),
        ];
        let mut s = Shader::compute([4, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
        s.resources.push(array);
        let l = lower(&s).expect("lowers");
        let m = &l.module;
        let callee = m.funcs.iter().position(|f| f.name == "dx.op.createHandleFromBinding").expect("declared") as u32;
        let at = m.body.iter().position(|i| is_binding_call(i, callee)).expect("a call");
        let LOp::Rel(Ref::Inst(_)) = m.body[at].ops[6] else { panic!("the index is computed") };
        // The instruction before the call adds the register 8 to the element.
        let add = &m.body[at - 1];
        assert_eq!(add.code, fc::BINOP);
        assert!(matches!(add.ops[2], LOp::Imm(0)));
        assert!(matches!(add.ops[1], LOp::Rel(Ref::Const(c)) if matches!(m.consts.list[c as usize].1, ConstKind::Int(8))));
    }

    #[test]
    fn cutscene_pixel_shader_with_mrt_packed_half_derivatives_loop_and_heap_lowers_and_validates() {
        let pos_in = SigElement::new("SV_Position", 0, Semantic::Position, ComponentType::F32, Interp::LinearNoperspective, 0, 4);
        let uv_in = SigElement::new("TEXCOORD", 0, Semantic::Arbitrary, ComponentType::F32, Interp::Linear, 1, 2);
        let t0 = SigElement::new("SV_Target", 0, Semantic::Target, ComponentType::F32, Interp::Undefined, 0, 4);
        let t1 = SigElement::new("SV_Target", 1, Semantic::Target, ComponentType::F32, Interp::Undefined, 1, 4);
        let depth_out = SigElement::new("SV_Depth", 0, Semantic::Depth, ComponentType::F32, Interp::Undefined, SigElement::UNALLOCATED, 1);

        // Resources:
        // 0: bound cbuffer b0 (space 0)
        // 1: bound Texture2D array t4..t7 (range_size 4, space 1)
        // 2: bound Sampler s0 (space 0)
        // 3: heap Texture2D
        // 4: heap Sampler
        let tex_elem = Element { component: ComponentType::F32, count: 4 };
        let mut tex_arr = Resource::texture(ResourceClass::Srv, ResourceKind::Texture2D, 1, 4, tex_elem);
        tex_arr.range_size = 4;
        let mut heap_tex = Resource::texture(ResourceClass::Srv, ResourceKind::Texture2D, 0, 0, tex_elem);
        heap_tex.binding = Binding::Heap;
        let mut heap_smp = Resource::sampler(0, 0, false);
        heap_smp.binding = Binding::Heap;

        let arr_ref = ResRef { res: 1, index: Some(Operand::i32(2)), non_uniform: true };
        let heap_tex_ref = ResRef { res: 3, index: Some(Operand::i32(5)), non_uniform: false };
        let heap_smp_ref = ResRef { res: 4, index: Some(Operand::i32(1)), non_uniform: false };

        let b0 = Block::new(
            vec![
                Inst::new(&[0], Op::LoadInput { elem: 1, row: Operand::i32(0), col: 0 }),
                Inst::new(&[1], Op::LoadInput { elem: 1, row: Operand::i32(0), col: 1 }),
                Inst::new(&[2], Op::CbufLoad { res: ResRef::bound(0), row: Operand::i32(0), comps: vec![0], ty: Ty::F32 }),
                Inst::new(&[3], Op::Un(UnOp::DdxFine, Operand::value(0))),
                Inst::new(&[4], Op::Un(UnOp::DdyFine, Operand::value(1))),
                Inst::new(&[5], Op::IsHelperLane),
                Inst::new(&[6], Op::Select(Operand::value(5), Operand::f32(0.0), Operand::value(3))),
                Inst::new(&[7, 8, 9, 10], Op::Sample {
                    tex: arr_ref,
                    sampler: ResRef::bound(2),
                    coords: vec![Operand::value(0), Operand::value(1)],
                    offsets: vec![],
                    mode: SampleMode::Grad { ddx: vec![Operand::value(6), Operand::f32(0.0)], ddy: vec![Operand::f32(0.0), Operand::value(4)] },
                    clamp: None,
                    ty: Ty::F32,
                }),
                // Packed half-float encode/decode + bitfield ops.
                Inst::new(&[11], Op::Un(UnOp::F32ToF16Bits, Operand::value(7))),
                Inst::new(&[12], Op::BitfieldInsert { width: Operand::i32(16), offset: Operand::i32(16), insert: Operand::value(11), base: Operand::value(11) }),
                Inst::new(&[13], Op::BitfieldExtract { signed: false, width: Operand::i32(16), offset: Operand::i32(16), value: Operand::value(12) }),
                Inst::new(&[14], Op::Un(UnOp::F16BitsToF32, Operand::value(13))),
                Inst::new(&[15, 16, 17, 18], Op::Sample {
                    tex: heap_tex_ref,
                    sampler: heap_smp_ref,
                    coords: vec![Operand::value(0), Operand::value(1)],
                    offsets: vec![],
                    mode: SampleMode::Level(Operand::f32(0.0)),
                    clamp: None,
                    ty: Ty::F32,
                }),
                Inst::new(&[19], Op::FCmp(FPred::Olt, Operand::value(15), Operand::f32(0.01))),
                Inst::new(&[], Op::Discard(Operand::value(19))),
            ],
            Terminator::Br(BlockId(1)),
        );

        // Loop with forward phi (value 21 defined after phi 20 in block 1).
        let mut b1 = Block::new(
            vec![
                Inst::new(&[21], Op::Bin(BinOp::Add, Operand::value(20), Operand::i32(1))),
                Inst::new(&[22], Op::ICmp(IPred::Ult, Operand::value(21), Operand::i32(4))),
            ],
            Terminator::CondBr(Operand::value(22), BlockId(1), BlockId(2)),
        );
        b1.phis.push(Phi {
            dst: ValueId(20),
            ty: Ty::I32,
            incoming: vec![(BlockId(0), Operand::i32(0)), (BlockId(1), Operand::value(21))],
        });

        let b2 = Block::new(
            vec![
                Inst::new(&[23], Op::Cast(CastOp::UiToFp, Operand::value(20), Ty::F32)),
                Inst::new(&[24], Op::Bin(BinOp::FAdd, Operand::value(14), Operand::value(2))),
                Inst::new(&[], Op::StoreOutput { elem: 0, row: Operand::i32(0), col: 0, value: Operand::value(24) }),
                Inst::new(&[], Op::StoreOutput { elem: 1, row: Operand::i32(0), col: 0, value: Operand::value(23) }),
                Inst::new(&[], Op::StoreOutput { elem: 2, row: Operand::i32(0), col: 0, value: Operand::f32(0.5) }),
            ],
            Terminator::Ret,
        );

        let mut sh = Shader::pixel(vec![b0, b1, b2]);
        sh.inputs = vec![pos_in, uv_in];
        sh.outputs = vec![t0, t1, depth_out];
        sh.resources.push(Resource::cbuffer(0, 0, 64));
        sh.resources.push(tex_arr);
        sh.resources.push(Resource::sampler(0, 0, false));
        sh.resources.push(heap_tex);
        sh.resources.push(heap_smp);

        let mut out = crate::write(&sh).expect("writes cutscene pixel shader");
        assert!(out.report.resource_heap);
        assert!(out.report.sampler_heap);
        assert_eq!(out.report.bound.len(), 3);
        let sum = crate::container::digest(&out.container);
        assert!((0..4).all(|i| crate::container::rd32(&out.container, 4 + 4 * i) == sum[i]));

        #[cfg(windows)]
        if let Err(e) = crate::validate::validate_in_place(&mut out.container)
            && !matches!(e, crate::validate::ValidateError::Missing(_))
        {
            panic!("dxil validator rejected cutscene pixel shader: {e}");
        }
    }

    #[test]
    fn sparse_high_value_ids_lower_correctly() {
        let insts = vec![
            Inst::new(&[300_000], Op::ThreadId(ThreadComponentKind::Dispatch, 0)),
            Inst::new(&[], Op::BufStore { res: ResRef::bound(0), index: Operand::value(300_000), offset: None, values: vec![Operand::value(300_000)] }),
        ];
        let mut s = Shader::compute([1, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
        s.resources.push(Resource::raw(ResourceClass::Uav, 0, 0));
        let out = crate::write(&s).expect("lowers sparse high value ids");
        assert!(!out.container.is_empty());
    }
}
