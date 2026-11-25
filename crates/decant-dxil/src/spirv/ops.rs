//! The instruction translation: arithmetic, composites, conversions, GLSL.std.450, atomics,
//! barriers, subgroup operations and image operations.

use rspirv::dr::Instruction;
use rspirv::spirv::{GroupOperation, ImageOperands, Op, Word};

use super::mem::AtomicTarget;
use super::*;

impl Imp<'_> {
    /// Defines the result of `i` as a value of its result type.
    fn define(&mut self, i: &Instruction, v: Vec<Operand>) {
        let id = i.result_id.expect("a result id");
        self.vals.insert(id, Val::S(v));
        if let Some(t) = i.result_type {
            self.val_types.insert(id, t);
        }
        self.note_types(id);
    }

    /// Remembers the IR types of the value operands of `id`.
    pub fn note_types(&mut self, id: Word) {
        let (Some(Val::S(v)), Some(t)) = (self.vals.get(&id), self.val_types.get(&id)) else { return };
        let Ok(tys) = self.flat(*t) else { return };
        for (o, ty) in v.clone().into_iter().zip(tys) {
            if let Operand::Value(x) = o {
                self.op_ty.insert(x.0, ty);
            }
        }
    }

    fn arg(&self, i: &Instruction, n: usize) -> R<Vec<Operand>> {
        self.scalars(operand_id(i, n)?)
    }

    fn map1(&mut self, a: Vec<Operand>, f: impl Fn(&mut Self, Operand) -> Operand) -> Vec<Operand> {
        a.into_iter().map(|x| f(self, x)).collect()
    }

    fn map2(&mut self, a: Vec<Operand>, b: Vec<Operand>, f: impl Fn(&mut Self, Operand, Operand) -> Operand) -> R<Vec<Operand>> {
        if a.len() != b.len() {
            return fail("internal", format!("operand widths differ ({} and {})", a.len(), b.len()));
        }
        Ok(a.into_iter().zip(b).map(|(x, y)| f(self, x, y)).collect())
    }

    fn binop(&mut self, i: &Instruction, op: BinOp) -> R<()> {
        let (a, mut b) = (self.arg(i, 0)?, self.arg(i, 1)?);
        if matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr) {
            // The shift amount may be narrower or wider than the value.
            let want = self.result_scalar_ty(i)?;
            for x in b.iter_mut() {
                let have = self.val_ty_operand(x);
                if have != want {
                    *x = match (*x, want) {
                        (Operand::Const(Const::I32(c)), Ty::I64) => Operand::Const(Const::I64(c as u64)),
                        (Operand::Const(Const::I64(c)), Ty::I32) => Operand::Const(Const::I32(c as u32)),
                        (v, Ty::I64) => self.emit1(Op_::Cast(CastOp::ZExt, v, Ty::I64)),
                        (v, _) => self.emit1(Op_::Cast(CastOp::Trunc, v, Ty::I32)),
                    };
                }
            }
        }
        let r = self.map2(a, b, |s, x, y| s.emit1(Op_::Bin(op, x, y)))?;
        self.define(i, r);
        Ok(())
    }

    fn cmp_i(&mut self, i: &Instruction, p: IPred) -> R<()> {
        let (a, b) = (self.arg(i, 0)?, self.arg(i, 1)?);
        let r = self.map2(a, b, |s, x, y| s.emit1(Op_::ICmp(p, x, y)))?;
        self.define(i, r);
        Ok(())
    }

    fn cmp_f(&mut self, i: &Instruction, p: FPred) -> R<()> {
        let (a, b) = (self.arg(i, 0)?, self.arg(i, 1)?);
        let r = self.map2(a, b, |s, x, y| s.emit1(Op_::FCmp(p, x, y)))?;
        self.define(i, r);
        Ok(())
    }

    fn result_scalar_ty(&self, i: &Instruction) -> R<Ty> {
        let t = i.result_type.ok_or_else(|| ImportError { category: "internal".into(), detail: "no result type".into() })?;
        Ok(self.flat(t)?[0])
    }

    fn fconst(v: f32) -> Operand {
        Operand::f32(v)
    }

    pub fn register_vars(&mut self) {
        let ids: Vec<Word> = self.vars.keys().copied().collect();
        for id in ids {
            if let Ok(p) = self.var_ptr(id) {
                self.vals.insert(id, Val::Ptr(p));
            }
        }
    }

    fn ptr_of(&self, id: Word) -> R<Ptr> {
        match self.vals.get(&id) {
            Some(Val::Ptr(p)) => Ok(p.clone()),
            other => fail("internal", format!("id {id} is {other:?}, expected a pointer")),
        }
    }

    /// Reads the value of an operand id as an image descriptor with its sampler, if any.
    fn image_of(&self, id: Word) -> R<(DescRef, Option<DescRef>)> {
        match self.vals.get(&id) {
            Some(Val::Image(d)) => Ok((d.clone(), None)),
            Some(Val::Sampled(d, s)) => Ok((d.clone(), Some(s.clone()))),
            other => fail("internal", format!("id {id} is {other:?}, expected an image")),
        }
    }

    /// Translates one instruction of the block `b`.
    pub fn instruction(&mut self, i: &Instruction, b: usize) -> R<()> {
        let op = i.class.opcode;
        match op {
            Op::Label | Op::Nop | Op::Line | Op::NoLine | Op::SelectionMerge | Op::LoopMerge | Op::Name | Op::LifetimeStart | Op::LifetimeStop => Ok(()),
            Op::Variable => Ok(()),
            Op::Load => {
                let p = self.ptr_of(operand_id(i, 0)?)?;
                let v = self.load(&p)?;
                let id = i.result_id.unwrap();
                self.vals.insert(id, v);
                self.val_types.insert(id, i.result_type.unwrap());
                self.note_types(id);
                Ok(())
            }
            Op::Store => {
                let p = self.ptr_of(operand_id(i, 0)?)?;
                let v = self.arg(i, 1)?;
                self.store(&p, &v)
            }
            Op::CopyObject => {
                let id = operand_id(i, 0)?;
                let v = self.vals.get(&id).cloned().ok_or_else(|| ImportError { category: "internal".into(), detail: "CopyObject of an unknown id".into() })?;
                self.vals.insert(i.result_id.unwrap(), v);
                if let Some(t) = i.result_type {
                    self.val_types.insert(i.result_id.unwrap(), t);
                }
                Ok(())
            }
            Op::AccessChain | Op::InBoundsAccessChain => {
                let base = self.ptr_of(operand_id(i, 0)?)?;
                let idx = (1..i.operands.len()).map(|n| operand_id(i, n)).collect::<R<Vec<_>>>()?;
                let p = self.access_chain(&base, &idx)?;
                self.vals.insert(i.result_id.unwrap(), Val::Ptr(p));
                Ok(())
            }
            Op::ConvertUToPtr => {
                let addr = self.scalar(operand_id(i, 0)?)?;
                let Some(T::Pointer(sc, pointee)) = self.types.get(&i.result_type.unwrap()).cloned() else { return fail("internal", "ConvertUToPtr to a non pointer") };
                if sc != spirv::StorageClass::PhysicalStorageBuffer {
                    return fail("memory:convert-to-pointer", format!("an integer converted to a {sc:?} pointer"));
                }
                let p = Ptr { storage: sc, var: 0, ty: pointee, arr: 0, arr_idx: None, off: 0, dynoff: None, root: false, addr: Some(addr) };
                self.vals.insert(i.result_id.unwrap(), Val::Ptr(p));
                Ok(())
            }
            Op::ConvertPtrToU => {
                let p = self.ptr_of(operand_id(i, 0)?)?;
                let a = self.pointer_address(&p)?;
                self.define(i, vec![a]);
                Ok(())
            }
            Op::Phi => {
                let t = i.result_type.unwrap();
                let tys = self.flat(t)?;
                let mut dsts = Vec::new();
                let mut ops = Vec::new();
                for ty in &tys {
                    let dst = self.value();
                    self.blocks[b].phis.push(Phi { dst, ty: *ty, incoming: Vec::new() });
                    dsts.push(dst.0 as usize);
                    ops.push(Operand::Value(dst));
                }
                let mut incoming = Vec::new();
                let mut k = 0;
                while k + 1 < i.operands.len() {
                    incoming.push((operand_id(i, k)?, self.label(operand_id(i, k + 1)?)?));
                    k += 2;
                }
                self.phi_fix.push((b, dsts, incoming));
                self.define(i, ops);
                Ok(())
            }
            Op::Undef => {
                let tys = self.flat(i.result_type.unwrap())?;
                self.define(i, tys.iter().map(|t| zero(*t)).collect());
                Ok(())
            }
            // composites
            Op::CompositeConstruct => {
                let mut v = Vec::new();
                for n in 0..i.operands.len() {
                    v.extend(self.arg(i, n)?);
                }
                self.define(i, v);
                Ok(())
            }
            Op::CompositeExtract => {
                let comp = self.arg(i, 0)?;
                let ty = *self.val_types.get(&operand_id(i, 0)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "composite type".into() })?;
                let lits = (1..i.operands.len()).map(|n| operand_lit(i, n)).collect::<R<Vec<_>>>()?;
                let (off, len) = self.composite_range(ty, &lits)?;
                self.define(i, comp[off..off + len].to_vec());
                Ok(())
            }
            Op::CompositeInsert => {
                let obj = self.arg(i, 0)?;
                let mut comp = self.arg(i, 1)?;
                let ty = *self.val_types.get(&operand_id(i, 1)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "composite type".into() })?;
                let lits = (2..i.operands.len()).map(|n| operand_lit(i, n)).collect::<R<Vec<_>>>()?;
                let (off, len) = self.composite_range(ty, &lits)?;
                comp.splice(off..off + len, obj);
                self.define(i, comp);
                Ok(())
            }
            Op::VectorShuffle => {
                let (a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?);
                let mut out = Vec::new();
                for n in 2..i.operands.len() {
                    let c = operand_lit(i, n)? as usize;
                    out.push(if c == u32::MAX as usize {
                        zero(self.result_scalar_ty(i)?)
                    } else if c < a.len() {
                        a[c]
                    } else {
                        bv[c - a.len()]
                    });
                }
                self.define(i, out);
                Ok(())
            }
            Op::VectorExtractDynamic => {
                let v = self.arg(i, 0)?;
                let idx = self.scalar(operand_id(i, 1)?)?;
                let mut r = v[0];
                for (k, x) in v.iter().enumerate().skip(1) {
                    let is = self.emit1(Op_::ICmp(IPred::Eq, idx, Operand::i32(k as u32)));
                    r = self.emit1(Op_::Select(is, *x, r));
                }
                self.define(i, vec![r]);
                Ok(())
            }
            Op::VectorInsertDynamic => {
                let v = self.arg(i, 0)?;
                let c = self.scalar(operand_id(i, 1)?)?;
                let idx = self.scalar(operand_id(i, 2)?)?;
                let mut out = Vec::new();
                for (k, x) in v.iter().enumerate() {
                    let is = self.emit1(Op_::ICmp(IPred::Eq, idx, Operand::i32(k as u32)));
                    out.push(self.emit1(Op_::Select(is, c, *x)));
                }
                self.define(i, out);
                Ok(())
            }
            // integer arithmetic
            Op::IAdd => self.binop(i, BinOp::Add),
            Op::ISub => self.binop(i, BinOp::Sub),
            Op::IMul => self.binop(i, BinOp::Mul),
            Op::UDiv => self.binop(i, BinOp::UDiv),
            Op::SDiv => self.binop(i, BinOp::SDiv),
            Op::UMod => self.binop(i, BinOp::URem),
            Op::SRem => self.binop(i, BinOp::SRem),
            Op::SMod => {
                // The result takes the sign of the divisor: ((a srem b) + b) srem b.
                let (a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?);
                let r = self.map2(a, bv, |s, x, y| {
                    let r = s.emit1(Op_::Bin(BinOp::SRem, x, y));
                    let t = s.emit1(Op_::Bin(BinOp::Add, r, y));
                    s.emit1(Op_::Bin(BinOp::SRem, t, y))
                })?;
                self.define(i, r);
                Ok(())
            }
            Op::ShiftLeftLogical => self.binop(i, BinOp::Shl),
            Op::ShiftRightLogical => self.binop(i, BinOp::LShr),
            Op::ShiftRightArithmetic => self.binop(i, BinOp::AShr),
            Op::BitwiseAnd => self.binop(i, BinOp::And),
            Op::BitwiseOr => self.binop(i, BinOp::Or),
            Op::BitwiseXor => self.binop(i, BinOp::Xor),
            Op::Not => {
                let a = self.arg(i, 0)?;
                let r = self.map1(a, |s, x| {
                    let ty = if s.val_ty_operand(&x) == Ty::I64 { Const::I64(u64::MAX) } else { Const::I32(u32::MAX) };
                    s.emit1(Op_::Bin(BinOp::Xor, x, Operand::Const(ty)))
                });
                self.define(i, r);
                Ok(())
            }
            Op::SNegate => {
                let t = self.result_scalar_ty(i)?;
                let a = self.arg(i, 0)?;
                let z = zero(t);
                let r = self.map1(a, |s, x| s.emit1(Op_::Bin(BinOp::Sub, z, x)));
                self.define(i, r);
                Ok(())
            }
            Op::FNegate => {
                let a = self.arg(i, 0)?;
                let r = self.map1(a, |s, x| s.emit1(Op_::Bin(BinOp::FMul, x, Self::fconst(-1.0))));
                self.define(i, r);
                Ok(())
            }
            Op::FAdd => self.binop(i, BinOp::FAdd),
            Op::FSub => self.binop(i, BinOp::FSub),
            Op::FMul => self.binop(i, BinOp::FMul),
            Op::FDiv => self.binop(i, BinOp::FDiv),
            Op::FRem => self.binop(i, BinOp::FRem),
            Op::FMod => {
                // x - y * floor(x / y)
                let (a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?);
                let r = self.map2(a, bv, |s, x, y| {
                    let q = s.emit1(Op_::Bin(BinOp::FDiv, x, y));
                    let f = s.emit1(Op_::Un(UnOp::Floor, q));
                    let m = s.emit1(Op_::Bin(BinOp::FMul, y, f));
                    s.emit1(Op_::Bin(BinOp::FSub, x, m))
                })?;
                self.define(i, r);
                Ok(())
            }
            Op::VectorTimesScalar => {
                let (v, s) = (self.arg(i, 0)?, self.scalar(operand_id(i, 1)?)?);
                let r = self.map1(v, |m, x| m.emit1(Op_::Bin(BinOp::FMul, x, s)));
                self.define(i, r);
                Ok(())
            }
            Op::Dot => {
                let (a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?);
                let mut acc: Option<Operand> = None;
                for (x, y) in a.into_iter().zip(bv) {
                    let m = self.emit1(Op_::Bin(BinOp::FMul, x, y));
                    acc = Some(match acc {
                        Some(s) => self.emit1(Op_::Bin(BinOp::FAdd, s, m)),
                        None => m,
                    });
                }
                self.define(i, vec![acc.unwrap()]);
                Ok(())
            }
            Op::IAddCarry | Op::ISubBorrow => {
                let (a, bv) = (self.scalar(operand_id(i, 0)?)?, self.scalar(operand_id(i, 1)?)?);
                let r = self.emit(2, if op == Op::IAddCarry { Op_::AddCarry(a, bv) } else { Op_::SubBorrow(a, bv) });
                let carry = self.emit1(Op_::Select(r[1], Operand::i32(1), Operand::i32(0)));
                self.define(i, vec![r[0], carry]);
                Ok(())
            }
            Op::DPdx | Op::DPdxFine => self.unop(i, UnOp::DdxFine),
            Op::DPdy | Op::DPdyFine => self.unop(i, UnOp::DdyFine),
            Op::DPdxCoarse => self.unop(i, UnOp::DdxCoarse),
            Op::DPdyCoarse => self.unop(i, UnOp::DdyCoarse),
            Op::Fwidth | Op::FwidthFine | Op::FwidthCoarse => {
                let (dx, dy) = if i.class.opcode == Op::FwidthCoarse { (UnOp::DdxCoarse, UnOp::DdyCoarse) } else { (UnOp::DdxFine, UnOp::DdyFine) };
                let a = self.arg(i, 0)?;
                let r = self.map1(a, |s, x| {
                    let gx = s.emit1(Op_::Un(dx, x));
                    let gy = s.emit1(Op_::Un(dy, x));
                    let ax = s.emit1(Op_::Un(UnOp::FAbs, gx));
                    let ay = s.emit1(Op_::Un(UnOp::FAbs, gy));
                    s.emit1(Op_::Bin(BinOp::FAdd, ax, ay))
                });
                self.define(i, r);
                Ok(())
            }
            Op::BitCount => self.unop(i, UnOp::CountBits),
            Op::BitReverse => self.unop(i, UnOp::BitReverse),
            Op::BitFieldUExtract | Op::BitFieldSExtract => {
                let (base, off, cnt) = (self.arg(i, 0)?, self.scalar(operand_id(i, 1)?)?, self.scalar(operand_id(i, 2)?)?);
                let signed = op == Op::BitFieldSExtract;
                let r = self.map1(base, |s, x| s.emit1(Op_::BitfieldExtract { signed, width: cnt, offset: off, value: x }));
                self.define(i, r);
                Ok(())
            }
            Op::BitFieldInsert => {
                let (base, ins, off, cnt) = (self.arg(i, 0)?, self.arg(i, 1)?, self.scalar(operand_id(i, 2)?)?, self.scalar(operand_id(i, 3)?)?);
                let r = self.map2(base, ins, |s, x, y| s.emit1(Op_::BitfieldInsert { width: cnt, offset: off, insert: y, base: x }))?;
                self.define(i, r);
                Ok(())
            }
            // comparisons
            Op::IEqual => self.cmp_i(i, IPred::Eq),
            Op::INotEqual => self.cmp_i(i, IPred::Ne),
            Op::UGreaterThan => self.cmp_i(i, IPred::Ugt),
            Op::SGreaterThan => self.cmp_i(i, IPred::Sgt),
            Op::UGreaterThanEqual => self.cmp_i(i, IPred::Uge),
            Op::SGreaterThanEqual => self.cmp_i(i, IPred::Sge),
            Op::ULessThan => self.cmp_i(i, IPred::Ult),
            Op::SLessThan => self.cmp_i(i, IPred::Slt),
            Op::ULessThanEqual => self.cmp_i(i, IPred::Ule),
            Op::SLessThanEqual => self.cmp_i(i, IPred::Sle),
            Op::FOrdEqual => self.cmp_f(i, FPred::Oeq),
            Op::FUnordEqual => self.cmp_f(i, FPred::Ueq),
            Op::FOrdNotEqual => self.cmp_f(i, FPred::One),
            Op::FUnordNotEqual => self.cmp_f(i, FPred::Une),
            Op::FOrdLessThan => self.cmp_f(i, FPred::Olt),
            Op::FUnordLessThan => self.cmp_f(i, FPred::Ult),
            Op::FOrdGreaterThan => self.cmp_f(i, FPred::Ogt),
            Op::FUnordGreaterThan => self.cmp_f(i, FPred::Ugt),
            Op::FOrdLessThanEqual => self.cmp_f(i, FPred::Ole),
            Op::FUnordLessThanEqual => self.cmp_f(i, FPred::Ule),
            Op::FOrdGreaterThanEqual => self.cmp_f(i, FPred::Oge),
            Op::FUnordGreaterThanEqual => self.cmp_f(i, FPred::Uge),
            Op::LogicalAnd => self.binop(i, BinOp::And),
            Op::LogicalOr => self.binop(i, BinOp::Or),
            Op::LogicalNotEqual => self.binop(i, BinOp::Xor),
            Op::LogicalEqual => {
                let (a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?);
                let r = self.map2(a, bv, |s, x, y| s.emit1(Op_::ICmp(IPred::Eq, x, y)))?;
                self.define(i, r);
                Ok(())
            }
            Op::LogicalNot => {
                let a = self.arg(i, 0)?;
                let r = self.map1(a, |s, x| s.emit1(Op_::ICmp(IPred::Eq, x, Operand::Const(Const::Bool(false)))));
                self.define(i, r);
                Ok(())
            }
            Op::Select => {
                let (c, a, bv) = (self.arg(i, 0)?, self.arg(i, 1)?, self.arg(i, 2)?);
                let mut out = Vec::new();
                for (k, (x, y)) in a.into_iter().zip(bv).enumerate() {
                    let cond = if c.len() == 1 { c[0] } else { c[k] };
                    out.push(self.emit1(Op_::Select(cond, x, y)));
                }
                self.define(i, out);
                Ok(())
            }
            Op::IsNan => self.unop(i, UnOp::IsNan),
            Op::IsInf => self.unop(i, UnOp::IsInf),
            // conversions
            Op::All | Op::Any => {
                let v = self.arg(i, 0)?;
                let mut acc = v[0];
                for x in &v[1..] {
                    acc = self.emit1(Op_::Bin(if i.class.opcode == Op::All { BinOp::And } else { BinOp::Or }, acc, *x));
                }
                self.define(i, vec![acc]);
                Ok(())
            }
            Op::UConvert | Op::SConvert if self.narrow_conversion(i)? => Ok(()),
            Op::ConvertFToU | Op::ConvertFToS | Op::ConvertSToF | Op::ConvertUToF | Op::UConvert | Op::SConvert => {
                let to = self.result_scalar_ty(i)?;
                let src = *self.val_types.get(&operand_id(i, 0)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "conversion source type".into() })?;
                let from = self.flat(src)?[0];
                let cast = match op {
                    Op::ConvertFToU => CastOp::FpToUi,
                    Op::ConvertFToS => CastOp::FpToSi,
                    Op::ConvertSToF => CastOp::SiToFp,
                    Op::ConvertUToF => CastOp::UiToFp,
                    Op::UConvert if from == Ty::I32 && to == Ty::I64 => CastOp::ZExt,
                    Op::SConvert if from == Ty::I32 && to == Ty::I64 => CastOp::SExt,
                    _ if from == Ty::I64 && to == Ty::I32 => CastOp::Trunc,
                    _ if from == to => {
                        let a = self.arg(i, 0)?;
                        self.define(i, a);
                        return Ok(());
                    }
                    _ => return fail("op:conversion", format!("{op:?} from {from:?} to {to:?}")),
                };
                let a = self.arg(i, 0)?;
                let r = self.map1(a, |s, x| s.emit1(Op_::Cast(cast, x, to)));
                self.define(i, r);
                Ok(())
            }
            Op::Bitcast => self.bitcast(i),
            Op::FConvert => {
                let src = *self.val_types.get(&operand_id(i, 0)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "conversion source type".into() })?;
                let from = self.float_width(src);
                let to = self.float_width(i.result_type.unwrap());
                let a = self.arg(i, 0)?;
                let r = if to == 16 && from != 16 {
                    // Round to half precision through the bits.
                    self.map1(a, |s, x| {
                        let h = s.emit1(Op_::Un(UnOp::F32ToF16Bits, x));
                        s.emit1(Op_::Un(UnOp::F16BitsToF32, h))
                    })
                } else if from == 32 || from == 16 {
                    a
                } else {
                    return fail("type:float", format!("a conversion from {from} bit floats"));
                };
                self.define(i, r);
                Ok(())
            }
            Op::UMulExtended | Op::SMulExtended => {
                let signed = op == Op::SMulExtended;
                let (a, bv) = (self.scalar(operand_id(i, 0)?)?, self.scalar(operand_id(i, 1)?)?);
                let ext = if signed { CastOp::SExt } else { CastOp::ZExt };
                let (wa, wb) = (self.emit1(Op_::Cast(ext, a, Ty::I64)), self.emit1(Op_::Cast(ext, bv, Ty::I64)));
                let m = self.emit1(Op_::Bin(BinOp::Mul, wa, wb));
                let lo = self.emit1(Op_::Cast(CastOp::Trunc, m, Ty::I32));
                let sh = self.emit1(Op_::Bin(if signed { BinOp::AShr } else { BinOp::LShr }, m, Operand::Const(Const::I64(32))));
                let hi = self.emit1(Op_::Cast(CastOp::Trunc, sh, Ty::I32));
                self.define(i, vec![lo, hi]);
                Ok(())
            }
            Op::ExtInst => self.ext_inst(i),
            // synchronisation
            Op::ControlBarrier | Op::MemoryBarrier => self.barrier(i),
            // atomics
            Op::AtomicLoad | Op::AtomicStore | Op::AtomicExchange | Op::AtomicCompareExchange | Op::AtomicIIncrement | Op::AtomicIDecrement | Op::AtomicIAdd | Op::AtomicISub | Op::AtomicSMin | Op::AtomicUMin | Op::AtomicSMax | Op::AtomicUMax | Op::AtomicAnd | Op::AtomicOr | Op::AtomicXor => self.atomic(i),
            // subgroup
            Op::GroupNonUniformElect
            | Op::GroupNonUniformAll
            | Op::GroupNonUniformAny
            | Op::GroupNonUniformAllEqual
            | Op::GroupNonUniformBroadcast
            | Op::GroupNonUniformBroadcastFirst
            | Op::GroupNonUniformBallot
            | Op::GroupNonUniformBallotBitCount
            | Op::GroupNonUniformBallotBitExtract
            | Op::GroupNonUniformShuffle
            | Op::GroupNonUniformIAdd
            | Op::GroupNonUniformFAdd
            | Op::GroupNonUniformIMul
            | Op::GroupNonUniformFMul
            | Op::GroupNonUniformSMin
            | Op::GroupNonUniformUMin
            | Op::GroupNonUniformFMin
            | Op::GroupNonUniformSMax
            | Op::GroupNonUniformUMax
            | Op::GroupNonUniformFMax
            | Op::GroupNonUniformBitwiseAnd
            | Op::GroupNonUniformBitwiseOr
            | Op::GroupNonUniformBitwiseXor => self.subgroup(i),
            // images
            Op::SampledImage => {
                let (img, _) = self.image_of(operand_id(i, 0)?)?;
                let Some(Val::Sampler(s)) = self.vals.get(&operand_id(i, 1)?).cloned() else { return fail("internal", "a sampled image without a sampler") };
                self.vals.insert(i.result_id.unwrap(), Val::Sampled(img, s));
                Ok(())
            }
            Op::ImageTexelPointer => {
                let p = self.ptr_of(operand_id(i, 0)?)?;
                if p.storage != spirv::StorageClass::UniformConstant {
                    return fail("internal", "a texel pointer into something that is not an image");
                }
                let d = DescRef { var: p.var, arr: p.arr, idx: p.arr_idx, ty: p.ty };
                let coords = self.arg(i, 1)?;
                self.vals.insert(i.result_id.unwrap(), Val::Texel(d, coords));
                Ok(())
            }
            Op::Image => {
                let (img, _) = self.image_of(operand_id(i, 0)?)?;
                self.vals.insert(i.result_id.unwrap(), Val::Image(img));
                Ok(())
            }
            Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod | Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod | Op::ImageFetch | Op::ImageRead | Op::ImageWrite | Op::ImageGather | Op::ImageDrefGather | Op::ImageQuerySizeLod | Op::ImageQuerySize | Op::ImageQueryLevels | Op::ImageQueryLod => self.image_op(i),
            other => fail(format!("op:{other:?}"), "this instruction is not translated yet"),
        }
    }

    /// `OpUConvert` and `OpSConvert` that involve an 8 or 16 bit integer: the value is masked or sign extended
    /// to the width and signedness of the result. Returns whether it handled the instruction.
    fn narrow_conversion(&mut self, i: &Instruction) -> R<bool> {
        let src = *self.val_types.get(&operand_id(i, 0)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "conversion source type".into() })?;
        let from = self.flat_widths(src)?[0];
        let to_t = i.result_type.unwrap();
        let to = self.flat_widths(to_t)?[0];
        if from.1 >= 32 && to.1 >= 32 {
            return Ok(false);
        }
        let signed_src = |s: &Self, t: Word| -> bool {
            match s.types.get(&t) {
                Some(T::Vector(e, _)) => matches!(s.types.get(e), Some(T::Int(_, true))),
                other => matches!(other, Some(T::Int(_, true))),
            }
        };
        let (src_signed, dst_signed) = (signed_src(self, src), signed_src(self, to_t));
        let op = i.class.opcode;
        let a = self.arg(i, 0)?;
        let mut out = Vec::new();
        for x in a {
            // The bits of the source, as an unsigned or signed value of its width.
            let mut v = x;
            if from.1 < 32 {
                let mask = (1u32 << from.1) - 1;
                v = self.emit1(Op_::Bin(BinOp::And, v, Operand::i32(mask)));
                if op == Op::SConvert {
                    let up = self.emit1(Op_::Bin(BinOp::Shl, v, Operand::i32(32 - from.1)));
                    v = self.emit1(Op_::Bin(BinOp::AShr, up, Operand::i32(32 - from.1)));
                }
            }
            let _ = src_signed;
            // To the width of the result.
            if to.1 < 32 {
                let mask = (1u32 << to.1) - 1;
                v = self.emit1(Op_::Bin(BinOp::And, v, Operand::i32(mask)));
                if dst_signed {
                    let up = self.emit1(Op_::Bin(BinOp::Shl, v, Operand::i32(32 - to.1)));
                    v = self.emit1(Op_::Bin(BinOp::AShr, up, Operand::i32(32 - to.1)));
                }
            } else if to.0 == Ty::I64 {
                v = self.emit1(Op_::Cast(if op == Op::SConvert { CastOp::SExt } else { CastOp::ZExt }, v, Ty::I64));
            }
            out.push(v);
        }
        self.define(i, out);
        Ok(true)
    }

    fn float_width(&self, t: Word) -> u32 {
        match self.types.get(&t) {
            Some(T::Float(w)) => *w,
            Some(T::Vector(e, _)) => self.float_width(*e),
            _ => 0,
        }
    }

    /// The scalar types with their widths, in layout order (a half precision float is 16).
    fn flat_widths(&self, t: Word) -> R<Vec<(Ty, u32)>> {
        Ok(match self.types.get(&t) {
            Some(T::Vector(e, n)) => {
                let one = self.flat_widths(*e)?;
                (0..*n).flat_map(|_| one.clone()).collect()
            }
            Some(T::Array(e, Some(n))) => {
                let one = self.flat_widths(*e)?;
                (0..*n).flat_map(|_| one.clone()).collect()
            }
            _ => vec![self.scalar_of(t)?],
        })
    }

    fn bitcast(&mut self, i: &Instruction) -> R<()> {
        if let Some(Val::Ptr(p)) = self.vals.get(&operand_id(i, 0)?).cloned() {
            let Some(T::Pointer(_, pointee)) = self.types.get(&i.result_type.unwrap()).cloned() else { return fail("op:bitcast", "a pointer cast to a non pointer") };
            let mut q = p;
            q.ty = pointee;
            self.vals.insert(i.result_id.unwrap(), Val::Ptr(q));
            return Ok(());
        }
        let src = *self.val_types.get(&operand_id(i, 0)?).ok_or_else(|| ImportError { category: "internal".into(), detail: "bitcast source type".into() })?;
        let from = self.flat_widths(src)?;
        let to = self.flat_widths(i.result_type.unwrap())?;
        let a = self.arg(i, 0)?;
        let mut out = Vec::new();
        if from.len() == to.len() && from.iter().zip(&to).all(|(f, t)| f.1 == t.1 && f.1 != 16) {
            for ((x, f), t) in a.into_iter().zip(&from).zip(&to) {
                out.push(if f.0 == t.0 { x } else if matches!((f.0, t.0), (Ty::I32, Ty::F32) | (Ty::F32, Ty::I32)) { self.emit1(Op_::Cast(CastOp::Bitcast, x, t.0)) } else { return fail("op:bitcast", format!("{:?} to {:?}", f.0, t.0)) });
            }
        } else if from.iter().all(|f| f.1 == 32) && to.len() * 2 == from.len() && to.iter().all(|t| t.1 == 64) {
            // pairs of 32 bit words to 64 bit integers
            for pair in a.chunks(2) {
                let lo = self.as_u32(pair[0], from[0].0);
                let hi = self.as_u32(pair[1], from[0].0);
                let (lo, hi) = (self.emit1(Op_::Cast(CastOp::ZExt, lo, Ty::I64)), self.emit1(Op_::Cast(CastOp::ZExt, hi, Ty::I64)));
                let sh = self.emit1(Op_::Bin(BinOp::Shl, hi, Operand::Const(Const::I64(32))));
                out.push(self.emit1(Op_::Bin(BinOp::Or, lo, sh)));
            }
        } else if from.iter().all(|f| f.1 == 64) && to.len() == from.len() * 2 && to.iter().all(|t| t.1 == 32) {
            for x in a {
                let lo = self.emit1(Op_::Cast(CastOp::Trunc, x, Ty::I32));
                let sh = self.emit1(Op_::Bin(BinOp::LShr, x, Operand::Const(Const::I64(32))));
                let hi = self.emit1(Op_::Cast(CastOp::Trunc, sh, Ty::I32));
                for w in [lo, hi] {
                    out.push(if to[0].0 == Ty::F32 { self.emit1(Op_::Cast(CastOp::Bitcast, w, Ty::F32)) } else { w });
                }
            }
        } else if from.len() * 2 == to.len() && from.iter().all(|f| f.1 == 32) && to.iter().all(|t| t.1 == 16) {
            // a 32 bit word to two half precision floats
            for x in a {
                let w = self.as_u32(x, from[0].0);
                let lo = self.emit1(Op_::Bin(BinOp::And, w, Operand::i32(0xFFFF)));
                let hi = self.emit1(Op_::Bin(BinOp::LShr, w, Operand::i32(16)));
                out.push(self.emit1(Op_::Un(UnOp::F16BitsToF32, lo)));
                out.push(self.emit1(Op_::Un(UnOp::F16BitsToF32, hi)));
            }
        } else if from.len() == to.len() * 2 && from.iter().all(|f| f.1 == 16) && to.iter().all(|t| t.1 == 32) {
            for pair in a.chunks(2) {
                let lo = self.emit1(Op_::Un(UnOp::F32ToF16Bits, pair[0]));
                let hi = self.emit1(Op_::Un(UnOp::F32ToF16Bits, pair[1]));
                let sh = self.emit1(Op_::Bin(BinOp::Shl, hi, Operand::i32(16)));
                let w = self.emit1(Op_::Bin(BinOp::Or, lo, sh));
                out.push(if to[0].0 == Ty::F32 { self.emit1(Op_::Cast(CastOp::Bitcast, w, Ty::F32)) } else { w });
            }
        } else {
            return fail("op:bitcast", format!("{from:?} to {to:?}"));
        }
        self.define(i, out);
        Ok(())
    }

    /// A 32 bit word as an `I32` operand.
    fn as_u32(&mut self, x: Operand, t: Ty) -> Operand {
        if t == Ty::F32 { self.emit1(Op_::Cast(CastOp::Bitcast, x, Ty::I32)) } else { x }
    }

    fn unop(&mut self, i: &Instruction, u: UnOp) -> R<()> {
        let a = self.arg(i, 0)?;
        let r = self.map1(a, |s, x| s.emit1(Op_::Un(u, x)));
        self.define(i, r);
        Ok(())
    }

    /// The offset and length in scalars of a member selected by literal indices.
    fn composite_range(&self, mut t: Word, lits: &[u32]) -> R<(usize, usize)> {
        let mut off = 0usize;
        for &l in lits {
            match self.types.get(&t).cloned() {
                Some(T::Struct(members)) => {
                    for m in &members[..l as usize] {
                        off += self.flat_count(*m)? as usize;
                    }
                    t = members[l as usize];
                }
                Some(T::Vector(e, _)) | Some(T::Matrix(e, _)) | Some(T::Array(e, _)) => {
                    off += l as usize * self.flat_count(e)? as usize;
                    t = e;
                }
                other => return fail("internal", format!("composite index into {other:?}")),
            }
        }
        Ok((off, self.flat_count(t)? as usize))
    }

    // ---- GLSL.std.450

    fn ext_inst(&mut self, i: &Instruction) -> R<()> {
        let set = operand_id(i, 0)?;
        if Some(set) != self.ext_glsl {
            return fail("ext:unknown-set", "an extended instruction set other than GLSL.std.450");
        }
        let n = operand_lit(i, 1)?;
        let args: Vec<Vec<Operand>> = (2..i.operands.len()).map(|k| self.arg(i, k)).collect::<R<_>>()?;
        let f = Self::fconst;
        let un = |s: &mut Self, u: UnOp, a: &[Operand]| -> Vec<Operand> { a.iter().map(|&x| s.emit1(Op_::Un(u, x))).collect() };
        let result: Vec<Operand> = match n {
            1 => {
                // Round: half away from zero, sign(x) * floor(|x| + 0.5)
                let a = args[0].clone();
                self.map1(a, |s, x| {
                    let ab = s.emit1(Op_::Un(UnOp::FAbs, x));
                    let h = s.emit1(Op_::Bin(BinOp::FAdd, ab, f(0.5)));
                    let fl = s.emit1(Op_::Un(UnOp::Floor, h));
                    let neg = s.emit1(Op_::Bin(BinOp::FMul, fl, f(-1.0)));
                    let is_neg = s.emit1(Op_::FCmp(FPred::Olt, x, f(0.0)));
                    s.emit1(Op_::Select(is_neg, neg, fl))
                })
            }
            2 => un(self, UnOp::RoundNe, &args[0]),
            3 => un(self, UnOp::Trunc, &args[0]),
            4 => un(self, UnOp::FAbs, &args[0]),
            5 => self.map1(args[0].clone(), |s, x| {
                let neg = s.emit1(Op_::Bin(BinOp::Sub, Operand::i32(0), x));
                let is_neg = s.emit1(Op_::ICmp(IPred::Slt, x, Operand::i32(0)));
                s.emit1(Op_::Select(is_neg, neg, x))
            }),
            6 => self.map1(args[0].clone(), |s, x| {
                let pos = s.emit1(Op_::FCmp(FPred::Ogt, x, f(0.0)));
                let neg = s.emit1(Op_::FCmp(FPred::Olt, x, f(0.0)));
                let r = s.emit1(Op_::Select(neg, f(-1.0), f(0.0)));
                s.emit1(Op_::Select(pos, f(1.0), r))
            }),
            7 => self.map1(args[0].clone(), |s, x| {
                let pos = s.emit1(Op_::ICmp(IPred::Sgt, x, Operand::i32(0)));
                let neg = s.emit1(Op_::ICmp(IPred::Slt, x, Operand::i32(0)));
                let r = s.emit1(Op_::Select(neg, Operand::i32(u32::MAX), Operand::i32(0)));
                s.emit1(Op_::Select(pos, Operand::i32(1), r))
            }),
            8 => un(self, UnOp::Floor, &args[0]),
            9 => un(self, UnOp::Ceil, &args[0]),
            10 => un(self, UnOp::Fract, &args[0]),
            11 => self.map1(args[0].clone(), |s, x| s.emit1(Op_::Bin(BinOp::FMul, x, f(std::f32::consts::PI / 180.0)))),
            12 => self.map1(args[0].clone(), |s, x| s.emit1(Op_::Bin(BinOp::FMul, x, f(180.0 / std::f32::consts::PI)))),
            13 => un(self, UnOp::Sin, &args[0]),
            14 => un(self, UnOp::Cos, &args[0]),
            15 => self.map1(args[0].clone(), |s, x| {
                let sn = s.emit1(Op_::Un(UnOp::Sin, x));
                let cs = s.emit1(Op_::Un(UnOp::Cos, x));
                s.emit1(Op_::Bin(BinOp::FDiv, sn, cs))
            }),
            26 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| {
                let l = s.emit1(Op_::Un(UnOp::Log2, x));
                let m = s.emit1(Op_::Bin(BinOp::FMul, y, l));
                s.emit1(Op_::Un(UnOp::Exp2, m))
            })?,
            27 => self.map1(args[0].clone(), |s, x| {
                let m = s.emit1(Op_::Bin(BinOp::FMul, x, f(std::f32::consts::LOG2_E)));
                s.emit1(Op_::Un(UnOp::Exp2, m))
            }),
            28 => self.map1(args[0].clone(), |s, x| {
                let l = s.emit1(Op_::Un(UnOp::Log2, x));
                s.emit1(Op_::Bin(BinOp::FMul, l, f(std::f32::consts::LN_2)))
            }),
            29 => un(self, UnOp::Exp2, &args[0]),
            30 => un(self, UnOp::Log2, &args[0]),
            31 => un(self, UnOp::Sqrt, &args[0]),
            32 => un(self, UnOp::Rsqrt, &args[0]),
            37 | 79 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::FMin, x, y)))?,
            38 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::UMin, x, y)))?,
            39 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::SMin, x, y)))?,
            40 | 80 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::FMax, x, y)))?,
            41 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::UMax, x, y)))?,
            42 => self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::MinMax(MinMaxOp::SMax, x, y)))?,
            43 | 44 | 45 | 81 => {
                let (lo, hi) = match n {
                    43 | 81 => (MinMaxOp::FMax, MinMaxOp::FMin),
                    44 => (MinMaxOp::UMax, MinMaxOp::UMin),
                    _ => (MinMaxOp::SMax, MinMaxOp::SMin),
                };
                let mut out = Vec::new();
                for k in 0..args[0].len() {
                    let (l, h) = (args[1][if args[1].len() == 1 { 0 } else { k }], args[2][if args[2].len() == 1 { 0 } else { k }]);
                    let m = self.emit1(Op_::MinMax(lo, args[0][k], l));
                    out.push(self.emit1(Op_::MinMax(hi, m, h)));
                }
                out
            }
            46 => {
                let mut out = Vec::new();
                for (k, (&x0, &y0)) in args[0].iter().zip(&args[1]).enumerate() {
                    let a = args[2][if args[2].len() == 1 { 0 } else { k }];
                    let om = self.emit1(Op_::Bin(BinOp::FSub, f(1.0), a));
                    let x = self.emit1(Op_::Bin(BinOp::FMul, x0, om));
                    let y = self.emit1(Op_::Bin(BinOp::FMul, y0, a));
                    out.push(self.emit1(Op_::Bin(BinOp::FAdd, x, y)));
                }
                out
            }
            48 => {
                let mut out = Vec::new();
                for k in 0..args[1].len() {
                    let e = args[0][if args[0].len() == 1 { 0 } else { k }];
                    let lt = self.emit1(Op_::FCmp(FPred::Olt, args[1][k], e));
                    out.push(self.emit1(Op_::Select(lt, f(0.0), f(1.0))));
                }
                out
            }
            49 => {
                let mut out = Vec::new();
                for k in 0..args[2].len() {
                    let e0 = args[0][if args[0].len() == 1 { 0 } else { k }];
                    let e1 = args[1][if args[1].len() == 1 { 0 } else { k }];
                    let num = self.emit1(Op_::Bin(BinOp::FSub, args[2][k], e0));
                    let den = self.emit1(Op_::Bin(BinOp::FSub, e1, e0));
                    let q = self.emit1(Op_::Bin(BinOp::FDiv, num, den));
                    let t = self.emit1(Op_::Un(UnOp::Saturate, q));
                    let t2 = self.emit1(Op_::Bin(BinOp::FMul, t, f(2.0)));
                    let c = self.emit1(Op_::Bin(BinOp::FSub, f(3.0), t2));
                    let tt = self.emit1(Op_::Bin(BinOp::FMul, t, t));
                    out.push(self.emit1(Op_::Bin(BinOp::FMul, tt, c)));
                }
                out
            }
            50 => {
                let mut out = Vec::new();
                for ((&x, &y), &z) in args[0].iter().zip(&args[1]).zip(&args[2]) {
                    out.push(self.emit1(Op_::Mad(MadOp::FMad, x, y, z)));
                }
                out
            }
            66 | 67 | 69 => {
                // Length, Distance and Normalize through the sum of squares.
                let v: Vec<Operand> = if n == 67 { self.map2(args[0].clone(), args[1].clone(), |s, x, y| s.emit1(Op_::Bin(BinOp::FSub, x, y)))? } else { args[0].clone() };
                let mut sum: Option<Operand> = None;
                for x in &v {
                    let m = self.emit1(Op_::Bin(BinOp::FMul, *x, *x));
                    sum = Some(match sum {
                        Some(s) => self.emit1(Op_::Bin(BinOp::FAdd, s, m)),
                        None => m,
                    });
                }
                let sum = sum.unwrap();
                if n == 69 {
                    let r = self.emit1(Op_::Un(UnOp::Rsqrt, sum));
                    self.map1(v, |s, x| s.emit1(Op_::Bin(BinOp::FMul, x, r)))
                } else {
                    vec![self.emit1(Op_::Un(UnOp::Sqrt, sum))]
                }
            }
            73 => un(self, UnOp::FirstBitLow, &args[0]),
            74 | 75 => {
                let u = if n == 74 { UnOp::FirstBitHighSigned } else { UnOp::FirstBitHigh };
                self.map1(args[0].clone(), |s, x| {
                    let h = s.emit1(Op_::Un(u, x));
                    let none = s.emit1(Op_::ICmp(IPred::Eq, h, Operand::i32(u32::MAX)));
                    let idx = s.emit1(Op_::Bin(BinOp::Sub, Operand::i32(31), h));
                    s.emit1(Op_::Select(none, Operand::i32(u32::MAX), idx))
                })
            }
            54..=57 => {
                // Pack Snorm4x8, Unorm4x8, Snorm2x16 and Unorm2x16.
                let (bits, signed, scale) = match n {
                    54 => (8u32, true, 127.0f32),
                    55 => (8, false, 255.0),
                    56 => (16, true, 32767.0),
                    _ => (16, false, 65535.0),
                };
                let mut acc: Option<Operand> = None;
                for (k, x) in args[0].clone().into_iter().enumerate() {
                    let lo = if signed { -1.0 } else { 0.0 };
                    let c = self.emit1(Op_::MinMax(MinMaxOp::FMax, x, f(lo)));
                    let c = self.emit1(Op_::MinMax(MinMaxOp::FMin, c, f(1.0)));
                    let m = self.emit1(Op_::Bin(BinOp::FMul, c, f(scale)));
                    let r = self.emit1(Op_::Un(UnOp::RoundNe, m));
                    let q = self.emit1(Op_::Cast(CastOp::FpToSi, r, Ty::I32));
                    let masked = self.emit1(Op_::Bin(BinOp::And, q, Operand::i32((1u32 << bits) - 1)));
                    let placed = if k == 0 { masked } else { self.emit1(Op_::Bin(BinOp::Shl, masked, Operand::i32(k as u32 * bits))) };
                    acc = Some(match acc {
                        Some(a) => self.emit1(Op_::Bin(BinOp::Or, a, placed)),
                        None => placed,
                    });
                }
                vec![acc.unwrap()]
            }
            60 | 61 | 63 | 64 => {
                // Unpack Snorm2x16, Unorm2x16, Snorm4x8 and Unorm4x8 (59 is PackDouble2x32, 62
                // UnpackHalf2x16).
                let (bits, count, signed, scale) = match n {
                    60 => (16u32, 2u32, true, 32767.0f32),
                    61 => (16, 2, false, 65535.0),
                    63 => (8, 4, true, 127.0),
                    _ => (8, 4, false, 255.0),
                };
                let x = args[0][0];
                let mut out = Vec::new();
                for k in 0..count {
                    let v = if signed {
                        let up = self.emit1(Op_::Bin(BinOp::Shl, x, Operand::i32(32 - bits * (k + 1))));
                        self.emit1(Op_::Bin(BinOp::AShr, up, Operand::i32(32 - bits)))
                    } else {
                        let sh = if k == 0 { x } else { self.emit1(Op_::Bin(BinOp::LShr, x, Operand::i32(k * bits))) };
                        self.emit1(Op_::Bin(BinOp::And, sh, Operand::i32((1u32 << bits) - 1)))
                    };
                    let fl = self.emit1(Op_::Cast(if signed { CastOp::SiToFp } else { CastOp::UiToFp }, v, Ty::F32));
                    let d = self.emit1(Op_::Bin(BinOp::FDiv, fl, f(scale)));
                    out.push(if signed { self.emit1(Op_::MinMax(MinMaxOp::FMax, d, f(-1.0))) } else { d });
                }
                out
            }
            58 => {
                // PackHalf2x16
                let lo = self.emit1(Op_::Un(UnOp::F32ToF16Bits, args[0][0]));
                let hi = self.emit1(Op_::Un(UnOp::F32ToF16Bits, args[0][1]));
                let sh = self.emit1(Op_::Bin(BinOp::Shl, hi, Operand::i32(16)));
                vec![self.emit1(Op_::Bin(BinOp::Or, lo, sh))]
            }
            62 => {
                // UnpackHalf2x16
                let x = args[0][0];
                let lo = self.emit1(Op_::Bin(BinOp::And, x, Operand::i32(0xFFFF)));
                let hi = self.emit1(Op_::Bin(BinOp::LShr, x, Operand::i32(16)));
                vec![self.emit1(Op_::Un(UnOp::F16BitsToF32, lo)), self.emit1(Op_::Un(UnOp::F16BitsToF32, hi))]
            }
            other => return fail(format!("ext:glsl:{other}"), "this GLSL.std.450 instruction is not translated yet"),
        };
        self.define(i, result);
        Ok(())
    }

    // ---- barriers and atomics

    fn const_word(&self, i: &Instruction, n: usize) -> R<u32> {
        self.const_u32(operand_id(i, n)?)
    }

    fn barrier(&mut self, i: &Instruction) -> R<()> {
        let (exec, mem, sem) = if i.class.opcode == Op::ControlBarrier { (Some(self.const_word(i, 0)?), self.const_word(i, 1)?, self.const_word(i, 2)?) } else { (None, self.const_word(i, 0)?, self.const_word(i, 1)?) };
        let mut flags = 0;
        if exec == Some(2) {
            flags |= barrier::SYNC_GROUP;
        }
        if sem & 0x100 != 0 {
            flags |= barrier::GROUPSHARED_FENCE;
        }
        if sem & (0x40 | 0x800 | 0x80) != 0 && mem != 4 {
            flags |= if mem <= 1 { barrier::UAV_FENCE_GLOBAL } else { barrier::UAV_FENCE_GROUP };
        }
        if flags != 0 {
            self.emit0(Op_::Barrier { flags });
        }
        Ok(())
    }

    fn texel_atomic(&mut self, i: &Instruction, d: DescRef, coords: Vec<Operand>) -> R<()> {
        let op = i.class.opcode;
        let tex_ref = self.descriptor_ref(&d)?;
        let kind = self.resources[tex_ref.res as usize].kind;
        if self.resources[tex_ref.res as usize].class != ResourceClass::Uav {
            return fail("image:atomic-on-readonly", "an atomic on a texture that is not a storage image");
        }
        let n = if kind == ResourceKind::TypedBuffer { 1 } else { kind.coord_count() };
        if coords.len() < n {
            return fail("image:coordinates", "too few coordinates for an image atomic");
        }
        let coords = coords[..n].to_vec();
        let value = |s: &Self, k: usize| -> R<Operand> { s.scalar(operand_id(i, k)?) };
        let old = match op {
            Op::AtomicCompareExchange => {
                let (new, cmp) = (value(self, 4)?, value(self, 5)?);
                if kind == ResourceKind::TypedBuffer {
                    self.emit1(Op_::BufCmpXchg { res: tex_ref, index: coords[0], offset: None, compare: cmp, new })
                } else {
                    self.emit1(Op_::TexCmpXchg { tex: tex_ref, coords, compare: cmp, new })
                }
            }
            Op::AtomicLoad | Op::AtomicStore => return fail("image:atomic-load-store", "an atomic load or store of a texel"),
            _ => {
                let v = match op {
                    Op::AtomicIIncrement | Op::AtomicIDecrement => Operand::i32(1),
                    _ => value(self, 3)?,
                };
                let (aop, v) = match op {
                    Op::AtomicIAdd | Op::AtomicIIncrement => (AtomicOp::Add, v),
                    Op::AtomicISub | Op::AtomicIDecrement => (AtomicOp::Add, self.emit1(Op_::Bin(BinOp::Sub, Operand::i32(0), v))),
                    Op::AtomicSMin => (AtomicOp::SMin, v),
                    Op::AtomicUMin => (AtomicOp::UMin, v),
                    Op::AtomicSMax => (AtomicOp::SMax, v),
                    Op::AtomicUMax => (AtomicOp::UMax, v),
                    Op::AtomicAnd => (AtomicOp::And, v),
                    Op::AtomicOr => (AtomicOp::Or, v),
                    Op::AtomicXor => (AtomicOp::Xor, v),
                    _ => (AtomicOp::Xchg, v),
                };
                if kind == ResourceKind::TypedBuffer {
                    self.emit1(Op_::BufAtomic { op: aop, res: tex_ref, index: coords[0], offset: None, value: v })
                } else {
                    self.emit1(Op_::TexAtomic { op: aop, tex: tex_ref, coords, value: v })
                }
            }
        };
        self.define(i, vec![old]);
        Ok(())
    }

    fn atomic(&mut self, i: &Instruction) -> R<()> {
        let op = i.class.opcode;
        if let Some(Val::Texel(d, c)) = self.vals.get(&operand_id(i, 0)?).cloned() {
            return self.texel_atomic(i, d, c);
        }
        let p = self.ptr_of(operand_id(i, 0)?)?;
        let target = self.atomic_target(&p)?;
        let ty = self.flat(self.deref_ty(&p)?)?;
        if ty.len() != 1 || ty[0] != Ty::I32 {
            return fail("atomic:type", "atomics on values other than 32 bit integers");
        }
        // The value operand sits after the scope and the semantics (two for the exchange forms).
        let value = |s: &Self, k: usize| -> R<Operand> { s.scalar(operand_id(i, k)?) };
        let old: Operand = match op {
            Op::AtomicLoad => match target {
                AtomicTarget::Buffer(r, idx) => self.emit1(Op_::BufLoad { res: r, index: idx, offset: None, count: 1, ty: Ty::I32 }),
                AtomicTarget::Group(g, idx) => self.emit1(Op_::GsLoad { var: g, index: idx }),
            },
            Op::AtomicStore => {
                let v = value(self, 3)?;
                match target {
                    AtomicTarget::Buffer(r, idx) => self.emit0(Op_::BufStore { res: r, index: idx, offset: None, values: vec![v] }),
                    AtomicTarget::Group(g, idx) => self.emit0(Op_::GsStore { var: g, index: idx, value: v }),
                }
                return Ok(());
            }
            Op::AtomicCompareExchange => {
                let (new, cmp) = (value(self, 4)?, value(self, 5)?);
                match target {
                    AtomicTarget::Buffer(r, idx) => self.emit1(Op_::BufCmpXchg { res: r, index: idx, offset: None, compare: cmp, new }),
                    AtomicTarget::Group(g, idx) => self.emit1(Op_::GsCmpXchg { var: g, index: idx, compare: cmp, new }),
                }
            }
            _ => {
                let v = match op {
                    Op::AtomicIIncrement | Op::AtomicIDecrement => Operand::i32(1),
                    _ => value(self, 3)?,
                };
                match target {
                    AtomicTarget::Buffer(r, idx) => {
                        let (aop, v) = match op {
                            Op::AtomicIAdd | Op::AtomicIIncrement => (AtomicOp::Add, v),
                            Op::AtomicISub | Op::AtomicIDecrement => (AtomicOp::Add, self.emit1(Op_::Bin(BinOp::Sub, Operand::i32(0), v))),
                            Op::AtomicSMin => (AtomicOp::SMin, v),
                            Op::AtomicUMin => (AtomicOp::UMin, v),
                            Op::AtomicSMax => (AtomicOp::SMax, v),
                            Op::AtomicUMax => (AtomicOp::UMax, v),
                            Op::AtomicAnd => (AtomicOp::And, v),
                            Op::AtomicOr => (AtomicOp::Or, v),
                            Op::AtomicXor => (AtomicOp::Xor, v),
                            _ => (AtomicOp::Xchg, v),
                        };
                        self.emit1(Op_::BufAtomic { op: aop, res: r, index: idx, offset: None, value: v })
                    }
                    AtomicTarget::Group(g, idx) => {
                        let aop = match op {
                            Op::AtomicIAdd | Op::AtomicIIncrement => GsAtomicOp::Add,
                            Op::AtomicISub | Op::AtomicIDecrement => GsAtomicOp::Sub,
                            Op::AtomicSMin => GsAtomicOp::SMin,
                            Op::AtomicUMin => GsAtomicOp::UMin,
                            Op::AtomicSMax => GsAtomicOp::SMax,
                            Op::AtomicUMax => GsAtomicOp::UMax,
                            Op::AtomicAnd => GsAtomicOp::And,
                            Op::AtomicOr => GsAtomicOp::Or,
                            Op::AtomicXor => GsAtomicOp::Xor,
                            _ => GsAtomicOp::Xchg,
                        };
                        self.emit1(Op_::GsAtomic { op: aop, var: g, index: idx, value: v })
                    }
                }
            }
        };
        self.define(i, vec![old]);
        Ok(())
    }

    fn deref_ty(&self, p: &Ptr) -> R<Word> {
        Ok(p.ty)
    }

    // ---- subgroup operations

    fn subgroup(&mut self, i: &Instruction) -> R<()> {
        let op = i.class.opcode;
        if self.const_word(i, 0)? != 3 {
            return fail("subgroup:scope", "a group operation at a scope other than Subgroup");
        }
        let reduce_kind = |op: Op| -> Option<(WaveOp, bool, bool)> {
            // (operation, signed, float)
            Some(match op {
                Op::GroupNonUniformIAdd => (WaveOp::Sum, false, false),
                Op::GroupNonUniformFAdd => (WaveOp::Sum, false, true),
                Op::GroupNonUniformIMul => (WaveOp::Product, false, false),
                Op::GroupNonUniformFMul => (WaveOp::Product, false, true),
                Op::GroupNonUniformSMin => (WaveOp::Min, true, false),
                Op::GroupNonUniformUMin => (WaveOp::Min, false, false),
                Op::GroupNonUniformFMin => (WaveOp::Min, false, true),
                Op::GroupNonUniformSMax => (WaveOp::Max, true, false),
                Op::GroupNonUniformUMax => (WaveOp::Max, false, false),
                Op::GroupNonUniformFMax => (WaveOp::Max, false, true),
                _ => return None,
            })
        };
        let r: Vec<Operand> = match op {
            Op::GroupNonUniformElect => vec![self.emit1(Op_::WaveIsFirstLane)],
            Op::GroupNonUniformAll => vec![self.emit1(Op_::WaveAllTrue(self.scalar(operand_id(i, 1)?)?))],
            Op::GroupNonUniformAny => vec![self.emit1(Op_::WaveAnyTrue(self.scalar(operand_id(i, 1)?)?))],
            Op::GroupNonUniformAllEqual => {
                let x = self.scalar(operand_id(i, 1)?)?;
                let (x, _) = self.wave_in(x);
                vec![self.emit1(Op_::WaveAllEqual(x))]
            }
            Op::GroupNonUniformBroadcast | Op::GroupNonUniformShuffle => {
                let (v, lane) = (self.arg(i, 1)?, self.scalar(operand_id(i, 2)?)?);
                let mut out = Vec::new();
                for x in v {
                    let (xi, was_bool) = self.wave_in(x);
                    let r = self.emit1(Op_::WaveReadLaneAt { value: xi, lane });
                    out.push(self.wave_out(r, was_bool));
                }
                out
            }
            Op::GroupNonUniformBroadcastFirst => {
                let v = self.arg(i, 1)?;
                let mut out = Vec::new();
                for x in v {
                    let (xi, was_bool) = self.wave_in(x);
                    let r = self.emit1(Op_::WaveReadLaneFirst(xi));
                    out.push(self.wave_out(r, was_bool));
                }
                out
            }
            Op::GroupNonUniformBallot => self.emit(4, Op_::WaveBallot(self.scalar(operand_id(i, 1)?)?)),
            Op::GroupNonUniformBallotBitCount => {
                let Some(SOp::GroupOperation(g)) = i.operands.get(1) else { return fail("internal", "group operation") };
                if *g != GroupOperation::Reduce {
                    return fail("subgroup:ballot-scan", "a ballot bit count that is not a reduction");
                }
                let words = self.arg(i, 2)?;
                let mut sum: Option<Operand> = None;
                for w in words {
                    let c = self.emit1(Op_::Un(UnOp::CountBits, w));
                    sum = Some(match sum {
                        Some(s) => self.emit1(Op_::Bin(BinOp::Add, s, c)),
                        None => c,
                    });
                }
                vec![sum.unwrap()]
            }
            Op::GroupNonUniformBallotBitExtract => {
                let (words, idx) = (self.arg(i, 1)?, self.scalar(operand_id(i, 2)?)?);
                let wi = self.emit1(Op_::Bin(BinOp::LShr, idx, Operand::i32(5)));
                let mut w = words[0];
                for (k, x) in words.iter().enumerate().skip(1) {
                    let is = self.emit1(Op_::ICmp(IPred::Eq, wi, Operand::i32(k as u32)));
                    w = self.emit1(Op_::Select(is, *x, w));
                }
                let bit = self.emit1(Op_::Bin(BinOp::And, idx, Operand::i32(31)));
                let sh = self.emit1(Op_::Bin(BinOp::LShr, w, bit));
                let one = self.emit1(Op_::Bin(BinOp::And, sh, Operand::i32(1)));
                vec![self.emit1(Op_::ICmp(IPred::Ne, one, Operand::i32(0)))]
            }
            other => {
                let Some((wop, signed, _float)) = reduce_kind(other) else {
                    let bit = match other {
                        Op::GroupNonUniformBitwiseAnd => WaveBitOp::And,
                        Op::GroupNonUniformBitwiseOr => WaveBitOp::Or,
                        _ => WaveBitOp::Xor,
                    };
                    let Some(SOp::GroupOperation(g)) = i.operands.get(1) else { return fail("internal", "group operation") };
                    if *g != GroupOperation::Reduce {
                        return fail("subgroup:bit-scan", "a bitwise scan");
                    }
                    let v = self.arg(i, 2)?;
                    let out = v.into_iter().map(|x| self.emit1(Op_::WaveActiveBit { op: bit, value: x })).collect();
                    self.define(i, out);
                    return Ok(());
                };
                let Some(SOp::GroupOperation(g)) = i.operands.get(1) else { return fail("internal", "group operation") };
                let v = self.arg(i, 2)?;
                let mut out = Vec::new();
                for x in v {
                    out.push(match g {
                        GroupOperation::Reduce => self.emit1(Op_::WaveActive { op: wop, signed, value: x }),
                        GroupOperation::ExclusiveScan if matches!(wop, WaveOp::Sum | WaveOp::Product) => self.emit1(Op_::WavePrefix { op: wop, signed, value: x }),
                        GroupOperation::InclusiveScan if wop == WaveOp::Sum => {
                            let p = self.emit1(Op_::WavePrefix { op: wop, signed, value: x });
                            let is_float = self.val_ty_operand(&x) == Ty::F32;
                            self.emit1(Op_::Bin(if is_float { BinOp::FAdd } else { BinOp::Add }, p, x))
                        }
                        _ => return fail("subgroup:scan", format!("{g:?} of {wop:?}")),
                    });
                }
                out
            }
        };
        self.define(i, r);
        Ok(())
    }

    /// A wave operand: booleans travel as 0 or 1.
    fn wave_in(&mut self, x: Operand) -> (Operand, bool) {
        if self.val_ty_operand(&x) == Ty::Bool { (self.emit1(Op_::Select(x, Operand::i32(1), Operand::i32(0))), true) } else { (x, false) }
    }

    fn wave_out(&mut self, r: Operand, was_bool: bool) -> Operand {
        if was_bool { self.emit1(Op_::ICmp(IPred::Ne, r, Operand::i32(0))) } else { r }
    }

    /// The type of an operand (constants know theirs, values are looked up in the IR).
    fn val_ty_operand(&self, o: &Operand) -> Ty {
        match o {
            Operand::Const(c) => c.ty(),
            Operand::Value(v) => self.op_ty.get(&v.0).copied().unwrap_or(Ty::I32),
        }
    }

    // ---- images

    fn image_operands(&self, i: &Instruction, first: usize) -> R<ImgOps> {
        let mut o = ImgOps::default();
        let Some(SOp::ImageOperands(mask)) = i.operands.get(first) else { return Ok(o) };
        let mut k = first + 1;
        let take = |k: &mut usize| -> R<Word> {
            let w = operand_id(i, *k)?;
            *k += 1;
            Ok(w)
        };
        if mask.contains(ImageOperands::BIAS) {
            o.bias = Some(take(&mut k)?);
        }
        if mask.contains(ImageOperands::LOD) {
            o.lod = Some(take(&mut k)?);
        }
        if mask.contains(ImageOperands::GRAD) {
            o.grad = Some((take(&mut k)?, take(&mut k)?));
        }
        if mask.contains(ImageOperands::CONST_OFFSET) {
            o.offset = Some(take(&mut k)?);
        }
        if mask.contains(ImageOperands::OFFSET) {
            o.offset = Some(take(&mut k)?);
        }
        if mask.contains(ImageOperands::CONST_OFFSETS) {
            return fail("image:const-offsets", "gather with four offsets");
        }
        if mask.contains(ImageOperands::SAMPLE) {
            return fail("image:multisampled", "a sample index");
        }
        if mask.contains(ImageOperands::MIN_LOD) {
            o.min_lod = Some(take(&mut k)?);
        }
        Ok(o)
    }

    fn offsets_of(&self, o: &ImgOps) -> R<Vec<i32>> {
        let Some(id) = o.offset else { return Ok(Vec::new()) };
        let v = self.scalars(id)?;
        v.iter()
            .map(|x| match x {
                Operand::Const(Const::I32(c)) => Ok(*c as i32),
                _ => fail("image:dynamic-offset", "an image offset that is not a constant"),
            })
            .collect()
    }

    fn image_op(&mut self, i: &Instruction) -> R<()> {
        let op = i.class.opcode;
        let (img, smp) = self.image_of(operand_id(i, 0)?)?;
        let tex_ref = self.descriptor_ref(&img)?;
        let tex = tex_ref.res;
        let kind = self.resources[tex as usize].kind;
        let class = self.resources[tex as usize].class;
        let texel_ty = match self.resources[tex as usize].element.map(|e| e.component) {
            Some(ComponentType::F32) => Ty::F32,
            _ => Ty::I32,
        };
        let ncoords = if kind == ResourceKind::TypedBuffer { 1 } else { kind.coord_count() };
        let is_dref = matches!(op, Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod | Op::ImageDrefGather);
        let sampler_ref = match smp {
            Some(s) => Some(self.descriptor_ref_as(&s, is_dref)?),
            None => None,
        };
        let sampler = sampler_ref.as_ref().map(|r| r.res);
        let coords = |s: &Self, n: usize| -> R<Vec<Operand>> {
            let c = s.arg(i, 1)?;
            if c.len() < n {
                return fail("image:coordinates", format!("{} coordinates for a {kind:?}", c.len()));
            }
            Ok(c[..n].to_vec())
        };
        match op {
            Op::ImageFetch | Op::ImageRead => {
                let c = coords(self, ncoords)?;
                let o = self.image_operands(i, 2)?;
                if kind == ResourceKind::TypedBuffer {
                    let r = self.emit(4, Op_::BufLoad { res: tex_ref, index: c[0], offset: None, count: 4, ty: texel_ty });
                    self.define(i, r);
                    return Ok(());
                }
                let mip = match o.lod {
                    Some(l) => Some(self.scalar(l)?),
                    None if class == ResourceClass::Srv => Some(Operand::i32(0)),
                    None => None,
                };
                let offsets = self.offsets_of(&o)?;
                let r = self.emit(4, Op_::TexLoad { tex: tex_ref, mip, coords: c, offsets, ty: texel_ty });
                self.define(i, r);
                Ok(())
            }
            Op::ImageWrite => {
                let c = coords(self, ncoords)?;
                let texel = self.arg(i, 2)?;
                if kind == ResourceKind::TypedBuffer {
                    self.emit0(Op_::BufStore { res: tex_ref, index: c[0], offset: None, values: texel });
                } else {
                    self.emit0(Op_::TexStore { tex: tex_ref, coords: c, values: texel });
                }
                Ok(())
            }
            Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod | Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod => {
                let Some(_) = sampler else { return fail("internal", "a sample without a sampler") };
                let dref = matches!(op, Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod);
                let c = coords(self, ncoords)?;
                let first = if dref { 3 } else { 2 };
                let o = self.image_operands(i, first)?;
                let offsets = self.offsets_of(&o)?;
                let clamp = match o.min_lod {
                    Some(m) => Some(self.scalar(m)?),
                    None => None,
                };
                let mode = if dref {
                    let cmp = self.scalar(operand_id(i, 2)?)?;
                    if matches!(op, Op::ImageSampleDrefExplicitLod) {
                        match o.lod {
                            Some(l) if matches!(self.scalar(l)?, Operand::Const(Const::F32(0))) => SampleMode::CompareLevelZero(cmp),
                            _ => return fail("image:dref-lod", "a depth compare sample at a level other than zero"),
                        }
                    } else {
                        SampleMode::Compare(cmp)
                    }
                } else if let Some(l) = o.lod {
                    SampleMode::Level(self.scalar(l)?)
                } else if let Some((dx, dy)) = o.grad {
                    SampleMode::Grad { ddx: self.scalars(dx)?, ddy: self.scalars(dy)? }
                } else if let Some(bias) = o.bias {
                    SampleMode::Bias(self.scalar(bias)?)
                } else if op == Op::ImageSampleExplicitLod {
                    return fail("image:explicit-without-lod", "an explicit sample without a level");
                } else {
                    SampleMode::Implicit
                };
                let r = self.emit(4, Op_::Sample { tex: tex_ref, sampler: sampler_ref.unwrap(), coords: c, offsets, mode, clamp, ty: texel_ty });
                if dref { self.define(i, vec![r[0]]) } else { self.define(i, r) }
                Ok(())
            }
            Op::ImageGather | Op::ImageDrefGather => {
                let Some(_) = sampler else { return fail("internal", "a gather without a sampler") };
                let c = coords(self, ncoords)?;
                let dref = op == Op::ImageDrefGather;
                let third = operand_id(i, 2)?;
                let o = self.image_operands(i, 3)?;
                let offsets = self.offsets_of(&o)?;
                let (channel, compare) = if dref {
                    (0, Some(self.scalar(third)?))
                } else {
                    (self.const_u32(third)? as u8, None)
                };
                let r = self.emit(4, Op_::Gather { tex: tex_ref, sampler: sampler_ref.unwrap(), coords: c, offsets, channel, compare, ty: texel_ty });
                self.define(i, r);
                Ok(())
            }
            Op::ImageQuerySizeLod | Op::ImageQuerySize | Op::ImageQueryLevels => {
                let mip = if op == Op::ImageQuerySizeLod { Some(self.scalar(operand_id(i, 1)?)?) } else { None };
                let d = self.emit(4, Op_::GetDimensions { tex: tex_ref, mip });
                let out = match op {
                    Op::ImageQueryLevels => vec![d[3]],
                    _ => match kind {
                        ResourceKind::Texture1D | ResourceKind::TypedBuffer => vec![d[0]],
                        ResourceKind::Texture1DArray | ResourceKind::Texture2D | ResourceKind::TextureCube => vec![d[0], d[1]],
                        _ => vec![d[0], d[1], d[2]],
                    },
                };
                self.define(i, out);
                Ok(())
            }
            Op::ImageQueryLod => {
                let Some(_s) = sampler else { return fail("internal", "a LOD query without a sampler") };
                let c = coords(self, kind.coord_count() - usize::from(matches!(kind, ResourceKind::Texture1DArray | ResourceKind::Texture2DArray | ResourceKind::TextureCubeArray)))?;
                let a = self.emit1(Op_::CalculateLod { tex: tex_ref, sampler: sampler_ref.unwrap(), coords: c.clone(), clamped: true });
                let b = self.emit1(Op_::CalculateLod { tex: tex_ref, sampler: sampler_ref.unwrap(), coords: c, clamped: false });
                self.define(i, vec![a, b]);
                Ok(())
            }
            _ => unreachable!(),
        }
    }
}

/// The parsed image operands.
#[derive(Default)]
struct ImgOps {
    bias: Option<Word>,
    lod: Option<Word>,
    grad: Option<(Word, Word)>,
    offset: Option<Word>,
    min_lod: Option<Word>,
}
