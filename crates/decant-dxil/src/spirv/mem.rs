//! Pointers, descriptors and memory: access chains, loads, stores and atomics for each storage
//! class, and the creation of the IR resources behind descriptor variables.

use rspirv::spirv::{BuiltIn, Decoration, Dim, StorageClass, Word};

use super::*;

fn byte_layout(sc: StorageClass) -> bool {
    matches!(sc, StorageClass::StorageBuffer | StorageClass::Uniform | StorageClass::PushConstant | StorageClass::PhysicalStorageBuffer)
}

fn scalar_bytes(t: Ty) -> u32 {
    if t == Ty::I64 { 8 } else { 4 }
}

/// One scalar of a buffer layout.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leaf {
    pub ty: Ty,
    pub off: u32,
    /// 8 and 16 bit integers and half floats are parts of a dword, as 32 bit values here.
    pub bits: u32,
    pub signed: bool,
}

impl Leaf {
    fn sub_word(&self) -> bool {
        self.bits < 32
    }
    /// A half precision float, kept as an `F32`.
    fn half(&self) -> bool {
        self.bits == 16 && self.ty == Ty::F32
    }
}

/// A placement in a heap: the index expression and whether it is the sampler heap.
fn heap_expr(t: &Target) -> Option<(IndexExpr, bool)> {
    match t {
        Target::Heap(i) => Some((IndexExpr::Const(*i), false)),
        Target::SamplerHeap(i) => Some((IndexExpr::Const(*i), true)),
        Target::HeapAt(e) | Target::ConstantHeapAt(e) => Some((e.clone(), false)),
        Target::SamplerHeapAt(e) => Some((e.clone(), true)),
        Target::Register { .. } | Target::Address(_) => None,
    }
}

/// The element of a descriptor array that stands for "the one the run time index picks".
const DYNAMIC: u32 = 0x4000_0000;

/// The pseudo variable of the device memory window.
const WINDOW_VAR: Word = Word::MAX;

impl Imp<'_> {
    /// The raw UAV that is the window of device memory.
    fn window_resource(&mut self) -> u32 {
        if let Some(w) = self.window {
            return w;
        }
        let mut r = Resource::raw(ResourceClass::Uav, DEVICE_MEMORY_SPACE, 0);
        r.name = "device_memory".into();
        let index = self.resources.len() as u32;
        self.resources.push(r);
        self.imported.push(ImportedResource { set: DEVICE_MEMORY_SPACE, binding: 0, space: DEVICE_MEMORY_SPACE, array_index: 0, index, push_constant: false });
        self.window = Some(index);
        index
    }

    /// The 64 bit base address of the window, from its constant buffer.
    fn window_base_address(&mut self) -> Operand {
        let b = match self.window_base {
            Some(b) => b,
            None => {
                let mut r = Resource::cbuffer(DEVICE_MEMORY_SPACE, 0, 16);
                r.name = "device_memory_base".into();
                let index = self.resources.len() as u32;
                self.resources.push(r);
                self.imported.push(ImportedResource { set: DEVICE_MEMORY_SPACE, binding: 0, space: DEVICE_MEMORY_SPACE, array_index: 1, index, push_constant: false });
                self.window_base = Some(index);
                index
            }
        };
        let lo = self.emit1(Op_::CbufLoad { res: ResRef::bound(b), row: Operand::i32(0), comps: vec![0], ty: Ty::I32 });
        let hi = self.emit1(Op_::CbufLoad { res: ResRef::bound(b), row: Operand::i32(0), comps: vec![1], ty: Ty::I32 });
        self.join64(lo, hi)
    }

    /// Turns a device address pointer into a pointer into the window: a byte offset (`I32`) from its base.
    fn window_pointer(&mut self, p: &Ptr) -> R<Ptr> {
        let Some(addr) = p.addr else { return fail("internal", "a device pointer without an address") };
        let off = self.offset_operand(p.dynoff, p.off);
        let off64 = self.emit1(Op_::Cast(CastOp::ZExt, off, Ty::I64));
        let a = self.emit1(Op_::Bin(BinOp::Add, addr, off64));
        let base = self.window_base_address();
        let rel = self.emit1(Op_::Bin(BinOp::Sub, a, base));
        let byte = self.emit1(Op_::Cast(CastOp::Trunc, rel, Ty::I32));
        self.window_resource();
        Ok(Ptr { storage: StorageClass::StorageBuffer, var: WINDOW_VAR, ty: p.ty, arr: 0, arr_idx: None, off: 0, dynoff: Some(byte), root: false, addr: None })
    }

    /// A pointer to the start of a variable.
    pub fn var_ptr(&self, var: Word) -> R<Ptr> {
        let v = self.vars.get(&var).ok_or_else(|| ImportError { category: "internal".into(), detail: format!("variable {var}") })?;
        Ok(Ptr { storage: v.storage, var, ty: v.ty, arr: 0, arr_idx: None, off: 0, dynoff: None, root: true, addr: None })
    }

    /// The scalars of a type with their byte offsets, buffer layout (`Offset`, `ArrayStride`,
    /// `MatrixStride` decorations).
    pub fn leaves(&self, t: Word, base: u32, mstride: u32, out: &mut Vec<(Ty, u32)>) -> R<()> {
        let mut info = Vec::new();
        self.leaf_info(t, base, mstride, &mut info)?;
        out.extend(info.iter().map(|l| (l.ty, l.off)));
        Ok(())
    }

    /// The size in bytes of a scalar element of a buffer.
    fn elem_bytes(&self, t: Word) -> R<u32> {
        let (s, w) = self.scalar_of(t)?;
        Ok(if s == Ty::I64 { 8 } else { w.max(8) / 8 })
    }

    /// The scalars of a type with their byte offsets, widths and kinds.
    pub fn leaf_info(&self, t: Word, base: u32, mstride: u32, out: &mut Vec<Leaf>) -> R<()> {
        match self.types.get(&t).cloned() {
            Some(T::Bool | T::Int(..) | T::Float(_)) => {
                let (s, w) = self.scalar_of(t)?;
                if s == Ty::Bool {
                    return fail("memory:bool", "a boolean in a buffer");
                }
                let signed = matches!(self.types.get(&t), Some(T::Int(_, true)));
                out.push(Leaf { ty: s, off: base, bits: w, signed });
            }
            Some(T::Vector(e, n)) => {
                let size = self.elem_bytes(e)?;
                for k in 0..n {
                    self.leaf_info(e, base + k * size, mstride, out)?;
                }
            }
            Some(T::Matrix(c, n)) => {
                if self.has_deco(t, Decoration::RowMajor) {
                    return fail("memory:row-major", "row major matrices");
                }
                for k in 0..n {
                    self.leaf_info(c, base + k * mstride, mstride, out)?;
                }
            }
            Some(T::Array(e, Some(n))) => {
                let stride = self.deco_lit(t, Decoration::ArrayStride).ok_or_else(|| ImportError { category: "memory:layout".into(), detail: "array without a stride".into() })?;
                for k in 0..n {
                    self.leaf_info(e, base + k as u32 * stride, mstride, out)?;
                }
            }
            Some(T::Struct(members)) => {
                for (k, m) in members.iter().enumerate() {
                    let off = self.member_deco_lit(t, k as u32, Decoration::Offset).ok_or_else(|| ImportError { category: "memory:layout".into(), detail: "member without an offset".into() })?;
                    let ms = self.member_deco_lit(t, k as u32, Decoration::MatrixStride).unwrap_or(16);
                    self.leaf_info(*m, base + off, ms, out)?;
                }
            }
            Some(T::Array(_, None)) => return fail("memory:runtime-array", "a runtime array is loaded as a whole"),
            other => return fail("internal", format!("leaves of {other:?}")),
        }
        Ok(())
    }

    fn const_index(&self, id: Word) -> Option<u32> {
        match self.vals.get(&id) {
            Some(Val::S(v)) if v.len() == 1 => match v[0] {
                Operand::Const(Const::I32(x)) => Some(x),
                Operand::Const(Const::I64(x)) => Some(x as u32),
                _ => None,
            },
            _ => None,
        }
    }

    /// Steps into a pointer: `OpAccessChain`.
    pub fn access_chain(&mut self, base: &Ptr, indices: &[Word]) -> R<Ptr> {
        let mut p = base.clone();
        let bytes = byte_layout(p.storage);
        for &ix in indices {
            let t = self.types.get(&p.ty).cloned();
            match t {
                Some(T::Array(elem, _)) if p.root && matches!(p.storage, StorageClass::StorageBuffer | StorageClass::Uniform | StorageClass::UniformConstant) => {
                    let single = matches!(t, Some(T::Array(_, Some(1))));
                    if single {
                        p.arr = self.const_index(ix).unwrap_or(0);
                    } else if let (Some(_), Some(k)) = (&self.bindings, self.const_index(ix)) {
                        // A placed module has one resource per element, so a constant index picks it now.
                        p.arr = k;
                    } else {
                        p.arr_idx = Some(match self.vals.get(&ix) {
                            Some(Val::S(v)) if v.len() == 1 => v[0],
                            _ => return fail("internal", "a descriptor index that is not a scalar"),
                        });
                    }
                    p.ty = elem;
                }
                Some(T::Struct(members)) => {
                    let m = self.const_index(ix).ok_or_else(|| ImportError { category: "internal".into(), detail: "a struct is indexed dynamically".into() })? as usize;
                    if bytes {
                        p.off += self.member_deco_lit(p.ty, m as u32, Decoration::Offset).ok_or_else(|| ImportError { category: "memory:layout".into(), detail: "member without an offset".into() })?;
                    } else {
                        let mut skip = 0;
                        for member in &members[..m] {
                            skip += self.flat_count(*member)?;
                        }
                        p.off += skip;
                    }
                    p.ty = members[m];
                }
                Some(T::Array(elem, _)) | Some(T::Vector(elem, _)) | Some(T::Matrix(elem, _)) => {
                    let is_vec = matches!(t, Some(T::Vector(..)));
                    let is_mat = matches!(t, Some(T::Matrix(..)));
                    let stride = if bytes {
                        if is_vec {
                            self.elem_bytes(elem)?
                        } else if is_mat {
                            self.deco_matrix_stride(p.ty)
                        } else {
                            self.deco_lit(p.ty, Decoration::ArrayStride).ok_or_else(|| ImportError { category: "memory:layout".into(), detail: "array without a stride".into() })?
                        }
                    } else {
                        self.flat_count(elem)?
                    };
                    match self.const_index(ix) {
                        Some(c) => p.off += c * stride,
                        None => {
                            let idx = match self.vals.get(&ix) {
                                Some(Val::S(v)) if v.len() == 1 => v[0],
                                _ => return fail("internal", "a non scalar index"),
                            };
                            let scaled = if stride == 1 { idx } else { self.emit1(Op_::Bin(BinOp::Mul, idx, Operand::i32(stride))) };
                            p.dynoff = Some(match p.dynoff {
                                Some(d) => self.emit1(Op_::Bin(BinOp::Add, d, scaled)),
                                None => scaled,
                            });
                        }
                    }
                    p.ty = elem;
                }
                other => return fail("internal", format!("access chain through {other:?}")),
            }
            p.root = false;
        }
        Ok(p)
    }

    fn deco_matrix_stride(&self, _ty: Word) -> u32 {
        16
    }

    /// An `I32` operand for `dynoff + constant`.
    pub fn offset_operand(&mut self, dynoff: Option<Operand>, constant: u32) -> Operand {
        match dynoff {
            None => Operand::i32(constant),
            Some(d) if constant == 0 => d,
            Some(d) => self.emit1(Op_::Bin(BinOp::Add, d, Operand::i32(constant))),
        }
    }

    // ---- resources

    fn next_register(&mut self, class: u8, set: u32) -> u32 {
        let e = self.next_reg.entry((class, set)).or_insert(0);
        let r = *e;
        *e += 1;
        r
    }

    /// The range of a variable that is an array of more than one descriptor (`u32::MAX`: unbounded).
    fn descriptor_range(&self, var: Word) -> Option<u32> {
        // A placed module puts every element of an array at a register of its own.
        if self.bindings.is_some() {
            return None;
        }
        match self.types.get(&self.vars.get(&var)?.ty) {
            Some(T::Array(_, Some(1))) => None,
            Some(T::Array(_, Some(n))) => Some(*n as u32),
            Some(T::Array(_, None)) => Some(u32::MAX),
            _ => None,
        }
    }

    /// Gives a resource that is an array of `n` descriptors its range: the registers after it are
    /// skipped, an unbounded array gets a register space of its own.
    fn give_range(&mut self, res: &mut Resource, n: u32) {
        let code = match res.class {
            ResourceClass::Srv => 0,
            ResourceClass::Uav => 1,
            ResourceClass::CBuffer => 2,
            ResourceClass::Sampler => 3,
        };
        res.range_size = n;
        if n == u32::MAX {
            res.space = self.next_space;
            self.next_space += 1;
            res.lower_bound = 0;
        } else {
            *self.next_reg.entry((code, res.space)).or_insert(0) += n - 1;
        }
    }

    /// Adds a resource to the shader, at the place the host chose for the descriptor `(var, element)`
    /// when the module is placed. Another variable at the same register gives back the resource
    /// that is already there.
    fn register_resource(&mut self, mut res: Resource, range: Option<u32>, var: Word, element: u32, mappable: bool) -> R<u32> {
        if let Some(n) = range {
            self.give_range(&mut res, n);
        }
        let mut target = None;
        let mut heap_ref: Option<(HeapRef, bool)> = None;
        let mut constant = false;
        if let (Some(b), true) = (&self.bindings, mappable) {
            let v = &self.vars[&var];
            let (set, binding) = (v.set, v.binding);
            if element == DYNAMIC {
                // An element picked at run time: only heap elements can be chosen by an index.
                let list = b.elements(set, binding);
                let mut picks = Vec::new();
                let mut sampler = None;
                // Whether every element is a constant buffer: which one the index reaches is not known.
                constant = !list.is_empty() && list.iter().all(|(_, t)| matches!(t, Target::ConstantHeapAt(_)));
                for (k, t) in &list {
                    match heap_expr(t) {
                        Some((e, smp)) if sampler.is_none_or(|s| s == smp) => {
                            sampler = Some(smp);
                            picks.push((*k, e));
                        }
                        _ => return fail("binding:dynamic-index", format!("set {set} binding {binding} is indexed at run time and element {k} is not in a heap")),
                    }
                }
                if picks.is_empty() {
                    return fail("binding:unmapped", format!("set {set} binding {binding} is indexed at run time and no element is placed"));
                }
                heap_ref = Some((HeapRef::Select(picks), sampler == Some(true)));
            } else {
                match b.target(set, binding, element) {
                    Some(Target::Address(_)) => return fail("binding:root-descriptor", format!("set {set} binding {binding} element {element} is a root descriptor, it has no resource of its own")),
                    Some(t) => match heap_expr(&t) {
                        Some((e, smp)) => {
                            constant = matches!(t, Target::ConstantHeapAt(_));
                            heap_ref = Some((HeapRef::One(e), smp));
                        }
                        None => target = Some(t),
                    },
                    None => return fail("binding:unmapped", format!("set {set} binding {binding} element {element} is not placed")),
                }
            }
        }
        let mut heap = None;
        if let Some((h, smp)) = heap_ref {
            if (res.class == ResourceClass::Sampler) != smp {
                return fail("binding:class", "a sampler placed in the resource heap or a resource in the sampler heap");
            }
            match res.kind {
                ResourceKind::RawBuffer if constant => res = Resource::cbuffer(0, 0, 65536),
                ResourceKind::CBuffer => {}
                _ if constant => {
                    return fail("binding:class", format!("a {:?} {:?} placed at a constant buffer of the heap", res.class, res.kind));
                }
                _ => {}
            }
            res = res.in_heap();
            heap = Some(h);
        }
        match target {
            None => {}
            Some(Target::Register { class, space, register }) => {
                let compatible = match (res.kind, class) {
                    (ResourceKind::RawBuffer, RegisterClass::Srv | RegisterClass::Uav | RegisterClass::Cbv) => true,
                    (ResourceKind::CBuffer, RegisterClass::Cbv) => true,
                    (ResourceKind::CBuffer | ResourceKind::Sampler, _) => res.kind == ResourceKind::Sampler && class == RegisterClass::Sampler,
                    (_, RegisterClass::Srv) => res.class == ResourceClass::Srv,
                    // A texture read with fetches only may sit in a UAV register: the console shader read a
                    // read write texture. It becomes a UAV (typed loads), samples then fail in the lowering.
                    (_, RegisterClass::Uav) => res.class == ResourceClass::Uav || (res.class == ResourceClass::Srv && res.kind.is_texture()),
                    _ => false,
                };
                if !compatible {
                    return fail("binding:class", format!("a {:?} {:?} placed at a {class:?} register", res.class, res.kind));
                }
                match (res.kind, class) {
                    (ResourceKind::RawBuffer, RegisterClass::Srv) => res.class = ResourceClass::Srv,
                    (ResourceKind::RawBuffer, RegisterClass::Uav) => res.class = ResourceClass::Uav,
                    (ResourceKind::RawBuffer, RegisterClass::Cbv) => res = Resource::cbuffer(space, register, 65536),
                    (_, RegisterClass::Uav) => res.class = ResourceClass::Uav,
                    _ => {}
                }
                res.space = space;
                res.lower_bound = register;
                res.range_size = 1;
                let key = (class.code(), space, register);
                if let Some(&existing) = self.placed.get(&key) {
                    let e = &self.resources[existing as usize];
                    if e.kind != res.kind || e.comparison != res.comparison || e.element != res.element {
                        return fail("binding:conflict", format!("two different resources at {class:?} register {register} of space {space}"));
                    }
                    return Ok(existing);
                }
                self.placed.insert(key, self.resources.len() as u32);
            }
            Some(_) => {}
        }
        let index = self.resources.len() as u32;
        self.resources.push(res);
        if let Some(h) = heap {
            self.heap_of.insert(index, h);
        }
        Ok(index)
    }

    /// A reference to resource `r`, with the index of the descriptor when it is an array.
    pub fn attach(&mut self, r: u32, idx: Option<Operand>) -> R<ResRef> {
        if let Some(heap) = self.heap_of.get(&r).cloned() {
            return Ok(match (heap, idx) {
                (HeapRef::One(e), _) => {
                    let index = self.eval_index(&e)?;
                    ResRef::heap(r, index, false)
                }
                (HeapRef::Select(list), Some(i)) => {
                    // The element the index picks: a chain of selects over the heap indices.
                    let mut it = list.iter();
                    let (_, first) = it.next().ok_or_else(|| ImportError { category: "binding:dynamic-index".into(), detail: "an array without a placed element".into() })?;
                    let mut v = self.eval_index(first)?;
                    for (k, e) in it {
                        let x = self.eval_index(e)?;
                        let is = self.emit1(Op_::ICmp(IPred::Eq, i, Operand::i32(*k)));
                        v = self.emit1(Op_::Select(is, x, v));
                    }
                    ResRef::heap(r, v, true)
                }
                (HeapRef::Select(_), None) => return fail("internal", "a run time array without an index"),
            });
        }
        Ok(match idx {
            Some(i) if self.resources[r as usize].range_size != 1 => ResRef { res: r, index: Some(i), non_uniform: true },
            _ => ResRef::bound(r),
        })
    }

    /// A pointer into the device memory window when the buffer `p` points into is a root descriptor.
    fn root_window(&mut self, p: &Ptr) -> R<Option<Ptr>> {
        if self.bindings.is_none() || p.var == WINDOW_VAR || p.arr_idx.is_some() || !matches!(p.storage, StorageClass::StorageBuffer | StorageClass::Uniform) {
            return Ok(None);
        }
        let Some(v) = self.vars.get(&p.var) else { return Ok(None) };
        let Some(Target::Address(k)) = self.bindings.as_ref().and_then(|b| b.target(v.set, v.binding, p.arr)) else { return Ok(None) };
        let (lo, hi) = (self.user_word(k)?, self.user_word(k + 1)?);
        let addr = self.join64(lo, hi);
        let device = Ptr { storage: StorageClass::PhysicalStorageBuffer, var: 0, ty: p.ty, arr: 0, arr_idx: None, off: p.off, dynoff: p.dynoff, root: false, addr: Some(addr) };
        self.window_ok = true;
        self.window_pointer(&device).map(Some)
    }

    /// The user data mirror: dword `k` of the root constant buffer the host fills.
    fn user_word(&mut self, k: u32) -> R<Operand> {
        if k >= USER_DATA_DWORDS {
            return fail("binding:user-data", format!("user register s{k} is outside the mirror"));
        }
        let m = match self.mirror {
            Some(m) => m,
            None => {
                let mut r = Resource::cbuffer(USER_DATA_SPACE, USER_DATA_REGISTER, USER_DATA_DWORDS * 4);
                r.name = "user_data".into();
                let index = self.resources.len() as u32;
                self.resources.push(r);
                self.imported.push(ImportedResource { set: USER_DATA_SPACE, binding: 1, space: USER_DATA_SPACE, array_index: 0, index, push_constant: false });
                self.mirror = Some(index);
                index
            }
        };
        self.mirror_words.insert(k);
        Ok(self.emit1(Op_::CbufLoad { res: ResRef::bound(m), row: Operand::i32(k / 4), comps: vec![(k % 4) as u8], ty: Ty::I32 }))
    }

    /// The buffer a descriptor key names, as a resource reference (raw buffers and constant buffers).
    fn buffer_by_key(&mut self, set: u32, binding: u32, element: u32) -> R<ResRef> {
        let var = self.vars.iter().filter(|(_, v)| v.set == set && v.binding == binding && matches!(v.storage, StorageClass::StorageBuffer | StorageClass::Uniform)).map(|(id, _)| *id).min();
        let Some(var) = var else {
            return fail("binding:expr-buffer", format!("no buffer variable at set {set} binding {binding}"));
        };
        let v = &self.vars[&var];
        let ptr = Ptr { storage: v.storage, var, ty: v.ty, arr: element, arr_idx: None, off: 0, dynoff: None, root: false, addr: None };
        self.buffer_resource(&ptr)
    }

    /// Emits the computation of a heap index.
    pub fn eval_index(&mut self, e: &IndexExpr) -> R<Operand> {
        Ok(match e {
            IndexExpr::Const(c) => Operand::i32(*c),
            IndexExpr::User(k) => self.user_word(*k)?,
            IndexExpr::Add(a, b) => {
                let (a, b) = (self.eval_index(a)?, self.eval_index(b)?);
                match (a, b) {
                    (Operand::Const(Const::I32(x)), Operand::Const(Const::I32(y))) => Operand::i32(x.wrapping_add(y)),
                    (x, Operand::Const(Const::I32(0))) | (Operand::Const(Const::I32(0)), x) => x,
                    _ => self.emit1(Op_::Bin(BinOp::Add, a, b)),
                }
            }
            IndexExpr::Mul(a, b) | IndexExpr::And(a, b) => {
                let (x, y) = (self.eval_index(a)?, self.eval_index(b)?);
                self.emit1(Op_::Bin(if matches!(e, IndexExpr::Mul(..)) { BinOp::Mul } else { BinOp::And }, x, y))
            }
            IndexExpr::Shl(a, n) => {
                let a = self.eval_index(a)?;
                self.emit1(Op_::Bin(BinOp::Shl, a, Operand::i32(*n)))
            }
            IndexExpr::Shr(a, n) => {
                let a = self.eval_index(a)?;
                self.emit1(Op_::Bin(BinOp::LShr, a, Operand::i32(*n)))
            }
            IndexExpr::BufferWord { buffer: (set, binding, element), offset } => {
                let byte = self.eval_index(offset)?;
                let r = self.buffer_by_key(*set, *binding, *element)?;
                if self.resources[r.res as usize].kind == ResourceKind::CBuffer {
                    let p = Ptr { storage: StorageClass::Uniform, var: 0, ty: 0, arr: 0, arr_idx: None, off: 0, dynoff: Some(byte), root: false, addr: None };
                    self.cbuffer_scalar(r, &p, 0, Ty::I32)?
                } else {
                    self.emit1(Op_::BufLoad { res: r, index: byte, offset: None, count: 1, ty: Ty::I32 })
                }
            }
        })
    }

    fn note(&mut self, var: Word, arr: u32, index: u32, push: bool) {
        let (set, binding) = self.vars.get(&var).map(|v| (v.set, v.binding)).unwrap_or((0, 0));
        let space = self.resources[index as usize].space;
        self.imported.push(ImportedResource { set, binding, space, array_index: arr, index, push_constant: push });
    }

    fn writable(&self, var: Word) -> bool {
        if self.has_deco(var, Decoration::NonWritable) {
            return false;
        }
        let v = &self.vars[&var];
        let block = match self.types.get(&v.ty) {
            Some(T::Array(e, _)) => *e,
            _ => v.ty,
        };
        if let Some(T::Struct(members)) = self.types.get(&block) {
            return !(0..members.len()).all(|k| self.member_has_deco(block, k as u32, Decoration::NonWritable));
        }
        true
    }

    /// The size in bytes of a block, rounded up to 16.
    fn block_size(&self, t: Word) -> R<u32> {
        let mut leaves = Vec::new();
        self.leaves(t, 0, 16, &mut leaves)?;
        let end = leaves.iter().map(|(s, o)| o + scalar_bytes(*s)).max().unwrap_or(16);
        Ok(end.div_ceil(16).max(1) * 16)
    }

    /// The 64 bit device address a pointer holds (`OpConvertPtrToU`).
    pub fn pointer_address(&mut self, p: &Ptr) -> R<Operand> {
        let Some(addr) = p.addr else { return fail("memory:pointer-to-integer", "a pointer that is not a device address") };
        let off = self.offset_operand(p.dynoff, p.off);
        let off64 = self.emit1(Op_::Cast(CastOp::ZExt, off, Ty::I64));
        Ok(self.emit1(Op_::Bin(BinOp::Add, addr, off64)))
    }

    /// The IR resource of the buffer a pointer points into: a raw buffer for storage buffers and
    /// a constant buffer for uniform blocks and push constants.
    pub fn buffer_resource(&mut self, p: &Ptr) -> R<ResRef> {
        let r = self.buffer_resource_index(p)?;
        self.attach(r, p.arr_idx)
    }

    fn buffer_resource_index(&mut self, p: &Ptr) -> R<u32> {
        if p.var == WINDOW_VAR {
            return Ok(self.window_resource());
        }
        let range = self.descriptor_range(p.var);
        let dynamic = self.bindings.is_some() && p.arr_idx.is_some();
        let key = (p.var, if dynamic { DYNAMIC } else if p.storage == StorageClass::PushConstant || range.is_some() { 0 } else { p.arr });
        if let Some(r) = self.res_map.get(&key) {
            return Ok(*r);
        }
        let v_set = self.vars[&p.var].set;
        let block = match self.types.get(&self.vars[&p.var].ty) {
            Some(T::Array(e, _)) => *e,
            _ => self.vars[&p.var].ty,
        };
        let uniform_block = p.storage == StorageClass::Uniform && !self.has_deco(block, Decoration::BufferBlock);
        let res = if p.storage == StorageClass::PushConstant {
            let size = self.block_size(block)?;
            Resource::cbuffer(15, 0, size.min(65536))
        } else if uniform_block {
            let size = self.block_size(block)?;
            let reg = self.next_register(2, v_set);
            Resource::cbuffer(v_set, reg, size.min(65536))
        } else if matches!(p.storage, StorageClass::StorageBuffer | StorageClass::Uniform) {
            let (class, code) = if self.writable(p.var) { (ResourceClass::Uav, 1) } else { (ResourceClass::Srv, 0) };
            let reg = self.next_register(code, v_set);
            Resource::raw(class, v_set, reg)
        } else {
            return fail(format!("storage:{:?}", p.storage), "a buffer in this storage class");
        };
        let index = self.register_resource(res, range, p.var, key.1, p.storage != StorageClass::PushConstant)?;
        self.res_map.insert(key, index);
        self.note(p.var, if dynamic { 0 } else { key.1 }, index, p.storage == StorageClass::PushConstant);
        Ok(index)
    }

    /// The IR resource of an image or sampler descriptor.
    pub fn descriptor_ref(&mut self, d: &DescRef) -> R<ResRef> {
        self.descriptor_ref_as(d, false)
    }

    /// The resource of a descriptor; a sampler used for depth compare operations is a separate comparison sampler.
    pub fn descriptor_ref_as(&mut self, d: &DescRef, comparison: bool) -> R<ResRef> {
        let r = self.descriptor_resource(d, comparison)?;
        self.attach(r, d.idx)
    }

    fn descriptor_resource(&mut self, d: &DescRef, comparison: bool) -> R<u32> {
        let range = self.descriptor_range(d.var);
        let dynamic = self.bindings.is_some() && d.idx.is_some();
        let d = &DescRef { arr: if dynamic { DYNAMIC } else if range.is_some() { 0 } else { d.arr }, ..d.clone() };
        let variant = if comparison { 0x8000_0000 } else { 0 };
        if let Some(r) = self.res_map.get(&(d.var, d.arr | variant)) {
            return Ok(*r);
        }
        let set = self.vars[&d.var].set;
        let res = match self.types.get(&d.ty).cloned() {
            Some(T::Sampler) => {
                let reg = self.next_register(3, set);
                Resource::sampler(set, reg, comparison)
            }
            Some(T::Image { sampled, dim, arrayed, ms, kind, .. }) => {
                if ms {
                    return fail("image:multisampled", "multisampled images");
                }
                let comp = match self.types.get(&sampled) {
                    Some(T::Float(32)) => ComponentType::F32,
                    Some(T::Int(32, false)) => ComponentType::U32,
                    Some(T::Int(32, true)) => ComponentType::I32,
                    _ => return fail("image:texel-type", "a texel type other than 32 bit float or integer"),
                };
                let (class, code) = match kind {
                    1 => (ResourceClass::Srv, 0),
                    2 => (ResourceClass::Uav, 1),
                    _ => return fail("image:usage", "an image whose usage is not known"),
                };
                let element = Element { component: comp, count: 4 };
                let rk = match (dim, arrayed) {
                    (Dim::Dim1D, false) => ResourceKind::Texture1D,
                    (Dim::Dim1D, true) => ResourceKind::Texture1DArray,
                    (Dim::Dim2D, false) => ResourceKind::Texture2D,
                    (Dim::Dim2D, true) => ResourceKind::Texture2DArray,
                    (Dim::Dim3D, false) => ResourceKind::Texture3D,
                    (Dim::DimCube, false) => ResourceKind::TextureCube,
                    (Dim::DimCube, true) => ResourceKind::TextureCubeArray,
                    (Dim::DimBuffer, _) => ResourceKind::TypedBuffer,
                    _ => return fail("image:dim", format!("{dim:?} arrayed {arrayed}")),
                };
                let reg = self.next_register(code, set);
                if rk == ResourceKind::TypedBuffer { Resource::typed(class, set, reg, element) } else { Resource::texture(class, rk, set, reg, element) }
            }
            other => return fail("internal", format!("descriptor of type {other:?}")),
        };
        let index = self.register_resource(res, range, d.var, d.arr, true)?;
        self.res_map.insert((d.var, d.arr | variant), index);
        self.note(d.var, if dynamic { 0 } else { d.arr }, index, false);
        Ok(index)
    }

    // ---- loads and stores

    /// `OpLoad`.
    pub fn load(&mut self, p: &Ptr) -> R<Val> {
        let window;
        let p = match self.root_window(p)? {
            Some(w) => {
                window = w;
                &window
            }
            None => p,
        };
        match p.storage {
            StorageClass::UniformConstant => {
                let d = DescRef { var: p.var, arr: p.arr, idx: p.arr_idx, ty: p.ty };
                match self.types.get(&p.ty) {
                    Some(T::Image { .. }) => Ok(Val::Image(d)),
                    Some(T::Sampler) => Ok(Val::Sampler(d)),
                    other => fail("internal", format!("a load of {other:?} from UniformConstant")),
                }
            }
            StorageClass::Function | StorageClass::Private => {
                let l = self.local(p)?;
                let n = self.flat_count(p.ty)? as usize;
                let first = l + p.off as usize;
                Ok(Val::S((0..n).map(|k| self.read_slot(first + k, self.cur)).collect()))
            }
            StorageClass::Workgroup => {
                let tys = self.flat(p.ty)?;
                let g = *self.gs_map.get(&p.var).ok_or_else(|| ImportError { category: "internal".into(), detail: "workgroup variable".into() })? as u32;
                let mut out = Vec::new();
                for (k, t) in tys.iter().enumerate() {
                    let idx = self.offset_operand(p.dynoff, p.off + k as u32);
                    let raw = self.emit1(Op_::GsLoad { var: g, index: idx });
                    out.push(self.word_to_value(raw, *t)?);
                }
                Ok(Val::S(out))
            }
            StorageClass::Input if self.io.contains_key(&p.var) => self.load_io(p),
            StorageClass::Input => self.load_builtin(p),
            StorageClass::StorageBuffer | StorageClass::Uniform | StorageClass::PushConstant => self.load_buffer(p),
            StorageClass::PhysicalStorageBuffer => {
                let w = self.window_pointer(p)?;
                self.load_buffer(&w)
            }
            other => fail(format!("storage:{other:?}"), "a load from this storage class"),
        }
    }

    fn local(&self, p: &Ptr) -> R<usize> {
        if p.dynoff.is_some() {
            return fail("memory:dynamic-local-index", "a local array indexed dynamically");
        }
        self.locals.get(&p.var).map(|l| l.base).ok_or_else(|| ImportError { category: "internal".into(), detail: "local variable".into() })
    }

    /// Converts the `I32` of a workgroup slot to the type of the value.
    fn word_to_value(&mut self, raw: Operand, t: Ty) -> R<Operand> {
        Ok(match t {
            Ty::I32 => raw,
            Ty::F32 => self.emit1(Op_::Cast(CastOp::Bitcast, raw, Ty::F32)),
            Ty::Bool => self.emit1(Op_::ICmp(IPred::Ne, raw, Operand::i32(0))),
            _ => return fail("memory:workgroup-type", format!("{t:?} in workgroup memory")),
        })
    }

    fn value_to_word(&mut self, v: Operand, t: Ty) -> R<Operand> {
        Ok(match t {
            Ty::I32 => v,
            Ty::F32 => self.emit1(Op_::Cast(CastOp::Bitcast, v, Ty::I32)),
            Ty::Bool => self.emit1(Op_::Select(v, Operand::i32(1), Operand::i32(0))),
            _ => return fail("memory:workgroup-type", format!("{t:?} in workgroup memory")),
        })
    }

    fn load_builtin(&mut self, p: &Ptr) -> R<Val> {
        let Some(SOp::BuiltIn(b)) = self.decos.get(&p.var).and_then(|d| d.iter().find(|(x, _)| *x == Decoration::BuiltIn)).and_then(|(_, ops)| ops.first().cloned()) else {
            return fail("input:not-builtin", "a stage input that is not a built in");
        };
        let n = self.flat_count(p.ty)? as usize;
        let mut out = Vec::new();
        for k in 0..n {
            let comp = (p.off as usize + k) as u8;
            out.push(match b {
                BuiltIn::GlobalInvocationId => self.emit1(Op_::ThreadId(ThreadComponentKind::Dispatch, comp)),
                BuiltIn::LocalInvocationId => self.emit1(Op_::ThreadId(ThreadComponentKind::InGroup, comp)),
                BuiltIn::WorkgroupId => self.emit1(Op_::ThreadId(ThreadComponentKind::Group, comp)),
                BuiltIn::LocalInvocationIndex => self.emit1(Op_::FlattenedThreadIdInGroup),
                BuiltIn::SubgroupLocalInvocationId => self.emit1(Op_::WaveLaneIndex),
                BuiltIn::SubgroupSize => self.emit1(Op_::WaveLaneCount),
                BuiltIn::HelperInvocation => self.emit1(Op_::IsHelperLane),
                other => return fail(format!("builtin:{other:?}"), "this built in has no DXIL operation yet"),
            });
        }
        Ok(Val::S(out))
    }

    /// The push constant block of a placed module: no constant buffer, each word is what the host's
    /// layout says it stands for (see [`PushLayout`]).
    fn load_push(&mut self, p: &Ptr, layout: &PushLayout) -> R<Val> {
        if p.dynoff.is_some() {
            return fail("push:dynamic-index", "the push constants are indexed at run time");
        }
        let mut leaves = Vec::new();
        self.leaf_info(p.ty, 0, 16, &mut leaves)?;
        let mut out = Vec::new();
        for l in leaves {
            if l.ty == Ty::I64 || l.sub_word() {
                return fail("push:width", "a push constant that is not a 32 bit word");
            }
            let raw = self.push_word(layout, (p.off + l.off) / 4)?;
            out.push(self.word_to_value(raw, l.ty)?);
        }
        Ok(Val::S(out))
    }

    fn push_word(&mut self, layout: &PushLayout, word: u32) -> R<Operand> {
        if word < layout.data_end {
            if let Some((_, sgpr)) = layout.registers.iter().find(|(w, _)| *w == word) {
                return self.user_word(*sgpr);
            }
            return match layout.group_id.iter().find(|(w, _)| *w == word) {
                Some((_, axis)) => Ok(self.emit1(Op_::ThreadId(ThreadComponentKind::Group, *axis))),
                None => fail("push:prefix", format!("push word {word} is the value of a console register")),
            };
        }
        if word < layout.offsets_end {
            // A Direct3D view starts at its first byte.
            return Ok(Operand::i32(0));
        }
        if word >= layout.sizes_end {
            return Ok(Operand::i32(0));
        }
        let k = (word - layout.offsets_end) as usize;
        let Some(Some((set, binding, element))) = layout.buffers.get(k).copied() else {
            return fail("push:size-unmapped", format!("push word {word} is the size of buffer {k}, which is not known"));
        };
        let var = self.vars.iter().filter(|(_, v)| v.set == set && v.binding == binding && matches!(v.storage, StorageClass::StorageBuffer | StorageClass::Uniform)).map(|(id, _)| *id).min();
        let Some(var) = var else {
            return fail("push:size-unmapped", format!("no buffer variable at set {set} binding {binding}"));
        };
        // A root descriptor has no size: a bounds check on it passes.
        if let Some(Target::Address(_)) = self.bindings.as_ref().and_then(|b| b.target(set, binding, element)) {
            return Ok(Operand::i32(u32::MAX));
        }
        // A constant buffer cannot be asked its size: it is 64 KiB at most, so a bounds check on it passes.
        if let Some(Target::Register { class: RegisterClass::Cbv, .. }) = self.bindings.as_ref().and_then(|b| b.target(set, binding, element)) {
            return Ok(Operand::i32(65536));
        }
        let v = &self.vars[&var];
        let ptr = Ptr { storage: v.storage, var, ty: v.ty, arr: element, arr_idx: None, off: 0, dynoff: None, root: false, addr: None };
        let r = self.buffer_resource(&ptr)?;
        if self.resources[r.res as usize].kind != ResourceKind::RawBuffer {
            return fail("push:size-of-buffer", format!("buffer {k} is neither a raw buffer nor a constant buffer"));
        }
        let dims = self.emit(4, Op_::GetDimensions { tex: r, mip: None });
        Ok(dims[0])
    }

    fn load_buffer(&mut self, p: &Ptr) -> R<Val> {
        if p.storage == StorageClass::PushConstant
            && let Some(layout) = self.bindings.as_ref().and_then(|b| b.push.clone())
        {
            return self.load_push(p, &layout);
        }
        let r = self.buffer_resource(p)?;
        let mut leaves = Vec::new();
        self.leaf_info(p.ty, 0, 16, &mut leaves)?;
        let constant_buffer = matches!(self.resources[r.res as usize].kind, ResourceKind::CBuffer);
        let mut out: Vec<Operand> = Vec::new();
        if constant_buffer {
            for l in leaves {
                if l.ty == Ty::I64 {
                    let lo = self.cbuffer_scalar(r, p, l.off, Ty::I32)?;
                    let hi = self.cbuffer_scalar(r, p, l.off + 4, Ty::I32)?;
                    out.push(self.join64(lo, hi));
                } else if l.sub_word() {
                    return fail("memory:sub-word-constant-buffer", "an 8 or 16 bit value in a constant buffer");
                } else {
                    out.push(self.cbuffer_scalar(r, p, l.off, l.ty)?);
                }
            }
        } else {
            let mut k = 0;
            while k < leaves.len() {
                let l = leaves[k];
                if l.ty == Ty::I64 {
                    let index = self.offset_operand(p.dynoff, p.off + l.off);
                    let w = self.emit(2, Op_::BufLoad { res: r, index, offset: None, count: 2, ty: Ty::I32 });
                    out.push(self.join64(w[0], w[1]));
                    k += 1;
                    continue;
                }
                if l.sub_word() {
                    let byte = self.offset_operand(p.dynoff, p.off + l.off);
                    out.push(self.sub_word_load(r, byte, l));
                    k += 1;
                    continue;
                }
                let mut count = 1;
                while count < 4 && k + count < leaves.len() && leaves[k + count].ty == l.ty && !leaves[k + count].sub_word() && leaves[k + count].off == l.off + 4 * count as u32 {
                    count += 1;
                }
                let index = self.offset_operand(p.dynoff, p.off + l.off);
                let vals = self.emit(count, Op_::BufLoad { res: r, index, offset: None, count: count as u8, ty: l.ty });
                out.extend(vals);
                k += count;
            }
        }
        Ok(Val::S(out))
    }

    /// An 8 or 16 bit value: the dword that holds it, shifted and masked (and extended).
    fn sub_word_load(&mut self, r: ResRef, byte: Operand, l: Leaf) -> Operand {
        let (aligned, shift) = self.sub_word_address(byte);
        let w = self.emit1(Op_::BufLoad { res: r, index: aligned, offset: None, count: 1, ty: Ty::I32 });
        let v = self.emit1(Op_::Bin(BinOp::LShr, w, shift));
        let mask = (1u64 << l.bits) as u32 - 1;
        let m = self.emit1(Op_::Bin(BinOp::And, v, Operand::i32(mask)));
        if l.half() {
            self.emit1(Op_::Un(UnOp::F16BitsToF32, m))
        } else if l.signed {
            let up = self.emit1(Op_::Bin(BinOp::Shl, m, Operand::i32(32 - l.bits)));
            self.emit1(Op_::Bin(BinOp::AShr, up, Operand::i32(32 - l.bits)))
        } else {
            m
        }
    }

    /// The aligned dword offset and the bit shift of a byte offset.
    fn sub_word_address(&mut self, byte: Operand) -> (Operand, Operand) {
        match byte {
            Operand::Const(Const::I32(c)) => (Operand::i32(c & !3), Operand::i32((c & 3) * 8)),
            _ => {
                let aligned = self.emit1(Op_::Bin(BinOp::And, byte, Operand::i32(!3)));
                let low = self.emit1(Op_::Bin(BinOp::And, byte, Operand::i32(3)));
                (aligned, self.emit1(Op_::Bin(BinOp::Shl, low, Operand::i32(3))))
            }
        }
    }

    /// Stores an 8 or 16 bit value into the dword that holds it with two atomics (clear, then set),
    /// so that the neighbours written by other invocations survive.
    fn sub_word_store(&mut self, r: ResRef, byte: Operand, l: Leaf, value: Operand) {
        let (aligned, shift) = self.sub_word_address(byte);
        let bits = if l.half() { self.emit1(Op_::Un(UnOp::F32ToF16Bits, value)) } else { value };
        let mask = (1u64 << l.bits) as u32 - 1;
        let m = self.emit1(Op_::Bin(BinOp::And, bits, Operand::i32(mask)));
        let placed = self.emit1(Op_::Bin(BinOp::Shl, m, shift));
        let hole = self.emit1(Op_::Bin(BinOp::Shl, Operand::i32(mask), shift));
        let keep = self.emit1(Op_::Bin(BinOp::Xor, hole, Operand::i32(u32::MAX)));
        self.emit1(Op_::BufAtomic { op: AtomicOp::And, res: r, index: aligned, offset: None, value: keep });
        self.emit1(Op_::BufAtomic { op: AtomicOp::Or, res: r, index: aligned, offset: None, value: placed });
    }

    /// Two 32 bit halves as a 64 bit integer.
    pub fn join64(&mut self, lo: Operand, hi: Operand) -> Operand {
        let (l, h) = (self.emit1(Op_::Cast(CastOp::ZExt, lo, Ty::I64)), self.emit1(Op_::Cast(CastOp::ZExt, hi, Ty::I64)));
        let sh = self.emit1(Op_::Bin(BinOp::Shl, h, Operand::Const(Const::I64(32))));
        self.emit1(Op_::Bin(BinOp::Or, l, sh))
    }

    /// One scalar of a constant buffer.
    fn cbuffer_scalar(&mut self, r: ResRef, p: &Ptr, off: u32, t: Ty) -> R<Operand> {
        let total = p.off + off;
        match p.dynoff {
            None => Ok(self.emit1(Op_::CbufLoad { res: r, row: Operand::i32(total / 16), comps: vec![((total % 16) / 4) as u8], ty: t })),
            Some(d) => {
                let byte = self.emit1(Op_::Bin(BinOp::Add, d, Operand::i32(total)));
                let row = self.emit1(Op_::Bin(BinOp::LShr, byte, Operand::i32(4)));
                let word = self.emit1(Op_::Bin(BinOp::LShr, byte, Operand::i32(2)));
                let comp = self.emit1(Op_::Bin(BinOp::And, word, Operand::i32(3)));
                let four = self.emit(4, Op_::CbufLoad { res: r, row, comps: vec![0, 1, 2, 3], ty: t });
                let mut v = four[0];
                for (k, x) in four.iter().enumerate().skip(1) {
                    let is = self.emit1(Op_::ICmp(IPred::Eq, comp, Operand::i32(k as u32)));
                    v = self.emit1(Op_::Select(is, *x, v));
                }
                Ok(v)
            }
        }
    }

    /// `OpStore`.
    pub fn store(&mut self, p: &Ptr, value: &[Operand]) -> R<()> {
        let window;
        let p = match self.root_window(p)? {
            Some(w) => {
                window = w;
                &window
            }
            None => p,
        };
        match p.storage {
            StorageClass::Function | StorageClass::Private => {
                let first = self.local(p)? + p.off as usize;
                for (k, v) in value.iter().enumerate() {
                    self.defs.insert((first + k, self.cur), *v);
                }
                Ok(())
            }
            StorageClass::Output => self.store_io(p, value),
            StorageClass::Workgroup => {
                let tys = self.flat(p.ty)?;
                let g = *self.gs_map.get(&p.var).ok_or_else(|| ImportError { category: "internal".into(), detail: "workgroup variable".into() })? as u32;
                for (k, t) in tys.iter().enumerate() {
                    let idx = self.offset_operand(p.dynoff, p.off + k as u32);
                    let raw = self.value_to_word(value[k], *t)?;
                    self.emit0(Op_::GsStore { var: g, index: idx, value: raw });
                }
                Ok(())
            }
            StorageClass::StorageBuffer | StorageClass::Uniform => {
                let r = self.buffer_resource(p)?;
                if self.resources[r.res as usize].class != ResourceClass::Uav {
                    return fail("memory:store-to-readonly", "a store to a buffer that is not writable");
                }
                let mut leaves = Vec::new();
                self.leaf_info(p.ty, 0, 16, &mut leaves)?;
                let mut k = 0;
                while k < leaves.len() {
                    let l = leaves[k];
                    if l.ty == Ty::I64 {
                        let index = self.offset_operand(p.dynoff, p.off + l.off);
                        let lo = self.emit1(Op_::Cast(CastOp::Trunc, value[k], Ty::I32));
                        let sh = self.emit1(Op_::Bin(BinOp::LShr, value[k], Operand::Const(Const::I64(32))));
                        let hi = self.emit1(Op_::Cast(CastOp::Trunc, sh, Ty::I32));
                        self.emit0(Op_::BufStore { res: r, index, offset: None, values: vec![lo, hi] });
                        k += 1;
                        continue;
                    }
                    if l.sub_word() {
                        let byte = self.offset_operand(p.dynoff, p.off + l.off);
                        self.sub_word_store(r, byte, l, value[k]);
                        k += 1;
                        continue;
                    }
                    let mut count = 1;
                    while count < 4 && k + count < leaves.len() && leaves[k + count].ty == l.ty && !leaves[k + count].sub_word() && leaves[k + count].off == l.off + 4 * count as u32 {
                        count += 1;
                    }
                    let index = self.offset_operand(p.dynoff, p.off + l.off);
                    self.emit0(Op_::BufStore { res: r, index, offset: None, values: value[k..k + count].to_vec() });
                    k += count;
                }
                Ok(())
            }
            StorageClass::PhysicalStorageBuffer => {
                let w = self.window_pointer(p)?;
                self.store(&w, value)
            }
            other => fail(format!("storage:{other:?}"), "a store to this storage class"),
        }
    }

    /// The pointee scalar location of an atomic: a buffer offset operand with its resource, or a workgroup slot.
    pub fn atomic_target(&mut self, p: &Ptr) -> R<AtomicTarget> {
        let window;
        let p = match self.root_window(p)? {
            Some(w) => {
                window = w;
                &window
            }
            None => p,
        };
        match p.storage {
            StorageClass::StorageBuffer | StorageClass::Uniform => {
                let r = self.buffer_resource(p)?;
                if self.resources[r.res as usize].class != ResourceClass::Uav {
                    return fail("memory:atomic-on-readonly", "an atomic on a buffer that is not writable");
                }
                let index = self.offset_operand(p.dynoff, p.off);
                Ok(AtomicTarget::Buffer(r, index))
            }
            StorageClass::Workgroup => {
                let g = *self.gs_map.get(&p.var).ok_or_else(|| ImportError { category: "internal".into(), detail: "workgroup variable".into() })? as u32;
                let index = self.offset_operand(p.dynoff, p.off);
                Ok(AtomicTarget::Group(g, index))
            }
            StorageClass::PhysicalStorageBuffer => {
                let w = self.window_pointer(p)?;
                self.atomic_target(&w)
            }
            other => fail(format!("storage:{other:?}"), "an atomic on this storage class"),
        }
    }
}

/// Where an atomic operation acts.
pub(crate) enum AtomicTarget {
    Buffer(ResRef, Operand),
    Group(u32, Operand),
}
