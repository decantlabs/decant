//! Signatures: the use of input and output elements, the data flow from inputs to outputs for
//! `dx.viewIdState` and `PSV0`, and the `ISG1`, `OSG1` and graphics `PSV0` parts.

use crate::Error;
use crate::container::{PsvResource, wr32};
use crate::ir::*;

/// A growable set of small integers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bits(Vec<u64>);

impl Bits {
    pub fn set(&mut self, i: usize) {
        if self.0.len() <= i / 64 {
            self.0.resize(i / 64 + 1, 0);
        }
        self.0[i / 64] |= 1 << (i % 64);
    }
    pub fn get(&self, i: usize) -> bool {
        self.0.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 != 0)
    }
    /// Adds every member of `other`, returns whether anything changed.
    pub fn union(&mut self, other: &Bits) -> bool {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        let mut changed = false;
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            let n = *a | *b;
            changed |= n != *a;
            *a = n;
        }
        changed
    }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.0.len() * 64).filter(|&i| self.get(i))
    }
}

/// How the shader uses one signature element.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ElemUse {
    /// The same per row of the element (a dynamic row index counts for all rows).
    pub row_used: Vec<u8>,
    pub row_always: Vec<u8>,
    /// Components accessed with a row index that is not a constant, same numbering.
    pub dynamic: u8,
    /// Mask of the components of the element read or written anywhere (bit 0 is the first column).
    pub used: u8,
    /// Components read on every path (inputs), same numbering.
    pub always: u8,
}

/// The result of the analysis of a graphics shader.
#[derive(Clone, Debug, Default)]
pub struct SigData {
    pub inputs: Vec<ElemUse>,
    pub outputs: Vec<ElemUse>,
    /// Scalars of the input and output signature (the dimensions of the view id table).
    pub in_scalars: u32,
    pub out_scalars: u32,
    /// `PSV0` and `dx.viewIdState` table: for each input scalar, the mask words of the outputs it feeds.
    pub table: Vec<u32>,
    /// The same for `PSV0`, sized by whole vectors.
    pub psv_table: Vec<u32>,
    /// Highest register plus one, of the inputs and outputs.
    pub in_vectors: u32,
    pub out_vectors: u32,
}

/// The scalar index of a component: register times four plus the component. `None` for an
/// element without a register (see [`SigElement::UNALLOCATED`]).
fn scalar(e: &SigElement, row: u32, col: u32) -> Option<u32> {
    e.allocated().then(|| (e.start_row + row) * 4 + e.start_col as u32 + col)
}

/// The number of scalars of a signature: the highest declared scalar plus one.
fn scalar_count(elems: &[SigElement]) -> u32 {
    elems.iter().filter_map(|e| scalar(e, e.rows - 1, e.cols as u32 - 1)).map(|s| s + 1).max().unwrap_or(0)
}

/// Postdominator sets: `pdom[b]` holds the blocks that every path from `b` to a return passes.
fn postdominators(shader: &Shader) -> Vec<Bits> {
    let n = shader.blocks.len();
    let succ = |b: &Block| match &b.term {
        Terminator::Br(t) => vec![t.0 as usize],
        Terminator::CondBr(_, a, c) => vec![a.0 as usize, c.0 as usize],
        Terminator::Ret => vec![],
    };
    let mut all = Bits::default();
    for i in 0..n {
        all.set(i);
    }
    let mut pdom: Vec<Bits> = (0..n)
        .map(|i| {
            if matches!(shader.blocks[i].term, Terminator::Ret) {
                let mut b = Bits::default();
                b.set(i);
                b
            } else {
                all.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            let s = succ(&shader.blocks[i]);
            if s.is_empty() {
                continue;
            }
            let mut inter = pdom[s[0]].clone();
            for &t in &s[1..] {
                for w in 0..inter.0.len() {
                    inter.0[w] &= pdom[t].0.get(w).copied().unwrap_or(0);
                }
            }
            inter.set(i);
            if inter != pdom[i] {
                pdom[i] = inter;
                changed = true;
            }
        }
    }
    pdom
}

/// The rows of `e` an access with this row operand touches.
fn hit_rows(e: &SigElement, row: &Operand) -> Vec<usize> {
    match row {
        Operand::Const(Const::I32(r)) if *r < e.rows => vec![*r as usize],
        _ => (0..e.rows as usize).collect(),
    }
}

/// Analyses the signature use of a normalised graphics shader.
pub fn analyze(shader: &Shader) -> Result<SigData, Error> {
    let (in_scalars, out_scalars) = (scalar_count(&shader.inputs), scalar_count(&shader.outputs));
    let n = shader.blocks.len();
    let pdom = postdominators(shader);
    // Blocks control dependent on a branch block: `cd[b]` lists the branches `b` depends on.
    let succ = |b: &Block| match &b.term {
        Terminator::Br(t) => vec![t.0 as usize],
        Terminator::CondBr(_, a, c) => vec![a.0 as usize, c.0 as usize],
        Terminator::Ret => vec![],
    };
    let mut cd: Vec<Vec<usize>> = vec![Vec::new(); n];
    for a in 0..n {
        let s = succ(&shader.blocks[a]);
        if s.len() < 2 {
            continue;
        }
        for (b, deps_of_b) in cd.iter_mut().enumerate() {
            let postdominates_a_strictly = b != a && pdom[a].get(b);
            if !postdominates_a_strictly && s.iter().any(|&t| pdom[t].get(b)) {
                deps_of_b.push(a);
            }
        }
    }
    // Close the dependence: a branch is itself controlled by other branches.
    let direct = cd.clone();
    for b in 0..n {
        let mut seen: Vec<usize> = direct[b].clone();
        let mut work = seen.clone();
        while let Some(a) = work.pop() {
            for &x in &direct[a] {
                if !seen.contains(&x) {
                    seen.push(x);
                    work.push(x);
                }
            }
        }
        cd[b] = seen;
    }

    let mut data = SigData { in_scalars, out_scalars, inputs: shader.inputs.iter().map(|e| ElemUse { row_used: vec![0; e.rows as usize], row_always: vec![0; e.rows as usize], ..Default::default() }).collect(), outputs: shader.outputs.iter().map(|e| ElemUse { row_used: vec![0; e.rows as usize], row_always: vec![0; e.rows as usize], ..Default::default() }).collect(), ..Default::default() };
    data.in_vectors = shader.inputs.iter().filter(|e| e.allocated()).map(|e| e.start_row + e.rows).max().unwrap_or(0);
    data.out_vectors = shader.outputs.iter().filter(|e| e.allocated()).map(|e| e.start_row + e.rows).max().unwrap_or(0);

    let mut deps: std::collections::HashMap<u32, Bits> = std::collections::HashMap::new();
    let mut out_deps: Vec<Bits> = vec![Bits::default(); out_scalars as usize];
    let value_deps = |deps: &std::collections::HashMap<u32, Bits>, o: &Operand| -> Bits {
        match o {
            Operand::Value(v) => deps.get(&v.0).cloned().unwrap_or_default(),
            Operand::Const(_) => Bits::default(),
        }
    };
    let elem_range = |elems: &[SigElement], elem: u32, row: &Operand, col: u32| -> Result<Vec<u32>, Error> {
        let e = elems.get(elem as usize).ok_or_else(|| Error::Invalid(format!("signature element {elem} does not exist")))?;
        let rows: Vec<u32> = match row {
            Operand::Const(Const::I32(r)) => vec![*r],
            _ => (0..e.rows).collect(),
        };
        for &r in &rows {
            if r >= e.rows {
                return Err(Error::Invalid(format!("row {r} of element {} ({}) is out of range", elem, e.name)));
            }
        }
        Ok(rows.into_iter().filter_map(|r| scalar(e, r, col)).collect())
    };
    let mut changed = true;
    let mut first = true;
    while changed {
        changed = false;
        // Dependence of the branch conditions, as known now.
        let cond_deps: Vec<Bits> = shader
            .blocks
            .iter()
            .map(|b| if let Terminator::CondBr(c, _, _) = &b.term { value_deps(&deps, c) } else { Bits::default() })
            .collect();
        for (bi, b) in shader.blocks.iter().enumerate() {
            let mut ctrl = Bits::default();
            for &a in &cd[bi] {
                ctrl.union(&cond_deps[a]);
            }
            for phi in &b.phis {
                let mut d = ctrl.clone();
                for (q, o) in &phi.incoming {
                    d.union(&value_deps(&deps, o));
                    for &a in &cd[q.0 as usize] {
                        d.union(&cond_deps[a]);
                    }
                }
                changed |= deps.entry(phi.dst.0).or_default().union(&d);
            }
            let always_runs = pdom[0].get(bi);
            for inst in &b.insts {
                let mut d = Bits::default();
                crate::lower::for_each_operand(&inst.op, &mut |o| {
                    d.union(&value_deps(&deps, o));
                });
                match &inst.op {
                    Op::LoadInput { elem, row, col } => {
                        shader.inputs.get(*elem as usize).ok_or_else(|| Error::Invalid(format!("input element {elem} does not exist")))?;
                        for sc in elem_range(&shader.inputs, *elem, row, *col as u32)? {
                            d.set(sc as usize);
                        }
                        if first && (shader.inputs[*elem as usize].allocated() || shader.inputs[*elem as usize].semantic != Semantic::SampleIndex) {
                            let bit = 1u8 << col;
                            let e = &shader.inputs[*elem as usize];
                            let u = &mut data.inputs[*elem as usize];
                            u.used |= bit;
                            let dynamic = !matches!(row, Operand::Const(_));
                            if dynamic {
                                u.dynamic |= bit;
                            }
                            for r in hit_rows(e, row) {
                                u.row_used[r] |= bit;
                                if always_runs {
                                    u.row_always[r] |= bit;
                                }
                            }
                            if always_runs {
                                u.always |= bit;
                            }
                        }
                    }
                    Op::StoreOutput { elem, row, col, value } => {
                        shader.outputs.get(*elem as usize).ok_or_else(|| Error::Invalid(format!("output element {elem} does not exist")))?;
                        let mut v = value_deps(&deps, value);
                        v.union(&ctrl);
                        for sc in elem_range(&shader.outputs, *elem, row, *col as u32)? {
                            changed |= out_deps[sc as usize].union(&v);
                        }
                        if first {
                            let e = &shader.outputs[*elem as usize];
                            let u = &mut data.outputs[*elem as usize];
                            u.used |= 1u8 << col;
                            if !matches!(row, Operand::Const(_)) {
                                u.dynamic |= 1u8 << col;
                            }
                            for r in hit_rows(e, row) {
                                u.row_used[r] |= 1u8 << col;
                            }
                        }
                    }
                    _ => {}
                }
                for dst in &inst.dsts {
                    changed |= deps.entry(dst.0).or_default().union(&d);
                }
            }
        }
        first = false;
    }
    // The tables, input major: for each input scalar, the outputs it feeds. The metadata table
    // covers the used scalars, the `PSV0` table whole vectors.
    let words = out_scalars.div_ceil(32) as usize;
    data.table = vec![0; in_scalars as usize * words];
    let psv_words = (data.out_vectors * 4).div_ceil(32) as usize;
    data.psv_table = vec![0; data.in_vectors as usize * 4 * psv_words];
    for (o, d) in out_deps.iter().enumerate() {
        for i in d.iter() {
            if i < in_scalars as usize {
                data.table[i * words + o / 32] |= 1 << (o % 32);
                data.psv_table[i * psv_words + o / 32] |= 1 << (o % 32);
            }
        }
    }
    Ok(data)
}

/// D3D_NAME of a semantic, for the signature parts.
pub fn system_value(s: Semantic) -> u32 {
    match s {
        Semantic::Arbitrary => 0,
        Semantic::Position => 1,
        Semantic::ClipDistance => 2,
        Semantic::CullDistance => 3,
        Semantic::RenderTargetArrayIndex => 4,
        Semantic::ViewportArrayIndex => 5,
        Semantic::VertexId => 6,
        Semantic::PrimitiveId => 7,
        Semantic::InstanceId => 8,
        Semantic::IsFrontFace => 9,
        Semantic::SampleIndex => 10,
        Semantic::Target => 64,
        Semantic::Depth => 65,
        Semantic::Coverage => 66,
        Semantic::DepthGreaterEqual => 67,
        Semantic::DepthLessEqual => 68,
        Semantic::StencilRef => 69,
        Semantic::InnerCoverage => 70,
        Semantic::Barycentrics => 23,
    }
}

/// DXIL `SemanticKind`.
pub fn semantic_code(s: Semantic) -> u32 {
    match s {
        Semantic::Arbitrary => 0,
        Semantic::VertexId => 1,
        Semantic::InstanceId => 2,
        Semantic::Position => 3,
        Semantic::RenderTargetArrayIndex => 4,
        Semantic::ViewportArrayIndex => 5,
        Semantic::ClipDistance => 6,
        Semantic::CullDistance => 7,
        Semantic::PrimitiveId => 10,
        Semantic::SampleIndex => 12,
        Semantic::IsFrontFace => 13,
        Semantic::Coverage => 14,
        Semantic::InnerCoverage => 15,
        Semantic::Barycentrics => 28,
        Semantic::Target => 16,
        Semantic::Depth => 17,
        Semantic::DepthLessEqual => 18,
        Semantic::DepthGreaterEqual => 19,
        Semantic::StencilRef => 20,
    }
}

pub fn interp_code(i: Interp) -> u32 {
    match i {
        Interp::Undefined => 0,
        Interp::Constant => 1,
        Interp::Linear => 2,
        Interp::LinearCentroid => 3,
        Interp::LinearNoperspective => 4,
        Interp::LinearNoperspectiveCentroid => 5,
        Interp::LinearSample => 6,
        Interp::LinearNoperspectiveSample => 7,
    }
}

/// The register mask a declaration covers (components 0 to `cols` for an element without register).
pub fn declared_mask(e: &SigElement) -> u8 {
    let start = if e.allocated() { e.start_col } else { 0 };
    (((1u32 << e.cols) - 1) << start) as u8
}

/// `D3D_REGISTER_COMPONENT_TYPE`.
fn register_component(c: ComponentType) -> u32 {
    match c {
        ComponentType::U32 => 1,
        ComponentType::I32 => 2,
        ComponentType::F32 => 3,
    }
}

/// An `ISG1` or `OSG1` part.
pub fn signature_part(elems: &[SigElement], uses: &[ElemUse], output: bool) -> Vec<u8> {
    // One row of the part for each row of an element: the same name string, the semantic index
    // and the register counting up. The rows are sorted by register, rows without a register last
    // (the metadata keeps the declaration order).
    struct Row<'a> {
        e: &'a SigElement,
        elem: usize,
        row: u32,
    }
    let mut rows: Vec<Row> = elems.iter().enumerate().flat_map(|(elem, e)| (0..e.rows).map(move |row| Row { e, elem, row })).collect();
    rows.sort_by_key(|r| (r.e.start_row.saturating_add(r.row), if r.e.allocated() { r.e.start_col } else { 0 }));
    let mut v = vec![0u8; 8 + 32 * rows.len()];
    wr32(&mut v, 0, rows.len() as u32);
    wr32(&mut v, 4, 8);
    // The names are laid out in declaration order, every arbitrary semantic with its own string and
    // system values sharing theirs.
    let mut names: Vec<(String, u32)> = Vec::new();
    let mut tail: Vec<u8> = Vec::new();
    let base = v.len() as u32;
    let mut offsets = vec![0u32; elems.len()];
    for (i, e) in elems.iter().enumerate() {
        let shared = if e.semantic == Semantic::Arbitrary { None } else { names.iter().find(|(n, _)| *n == e.name).map(|(_, o)| *o) };
        offsets[i] = match shared {
            Some(o) => o,
            None => {
                let o = base + tail.len() as u32;
                tail.extend_from_slice(e.name.as_bytes());
                tail.push(0);
                names.push((e.name.clone(), o));
                o
            }
        };
    }
    for (i, r) in rows.iter().enumerate() {
        let e = r.e;
        let at = 8 + 32 * i;
        wr32(&mut v, at + 4, offsets[r.elem]);
        wr32(&mut v, at + 8, e.index + r.row);
        wr32(&mut v, at + 12, system_value(e.semantic));
        wr32(&mut v, at + 16, register_component(e.component));
        wr32(&mut v, at + 20, if e.allocated() { e.start_row + r.row } else { 0xFFFF_FFFF });
        let mask = declared_mask(e);
        v[at + 24] = mask;
        let shift = if e.allocated() { e.start_col } else { 0 };
        let u = &uses[r.elem];
        // The use is per element: every row of an array carries the components used in any row.
        let (used, always) = (u.used, u.always);
        let _ = always;
        v[at + 25] = if output { 0xF & !(used << shift) } else { used << shift };
    }
    v.extend_from_slice(&tail);
    v
}

/// Everything a graphics `PSV0` needs besides the resources.
pub struct PsvGraphics<'a> {
    pub pixel: bool,
    pub inputs: &'a [SigElement],
    pub outputs: &'a [SigElement],
    pub data: &'a SigData,
}

/// `PSV0` of a vertex or pixel shader (validator 1.6, runtime info version 2).
pub fn psv0_graphics(g: &PsvGraphics, resources: &[PsvResource]) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    let put = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
    put(&mut v, 48);
    let mut union = [0u8; 16];
    if g.pixel {
        union[0] = g.outputs.iter().any(|e| matches!(e.semantic, Semantic::Depth | Semantic::DepthLessEqual | Semantic::DepthGreaterEqual)) as u8;
        union[1] = g.inputs.iter().any(|e| e.semantic == Semantic::SampleIndex || matches!(e.interp, Interp::LinearSample | Interp::LinearNoperspectiveSample)) as u8;
    } else {
        union[0] = g.outputs.iter().any(|e| e.semantic == Semantic::Position) as u8;
    }
    v.extend_from_slice(&union);
    put(&mut v, 0);
    put(&mut v, 0xFFFF_FFFF);
    v.extend_from_slice(&[if g.pixel { 0 } else { 1 }, 0, 0, 0, g.inputs.len() as u8, g.outputs.len() as u8, 0, g.data.in_vectors as u8, g.data.out_vectors as u8, 0, 0, 0]);
    for _ in 0..3 {
        put(&mut v, 0);
    }
    put(&mut v, resources.len() as u32);
    if !resources.is_empty() {
        put(&mut v, 24);
        for r in resources {
            for x in [r.ty, r.space, r.lower, r.upper, r.kind, r.flags] {
                put(&mut v, x);
            }
        }
    }
    // String table: the empty string, then the names of the arbitrary semantics.
    let mut strings: Vec<u8> = vec![0];
    let mut indices: Vec<u32> = Vec::new();
    let mut elems: Vec<[u8; 16]> = Vec::new();
    let uses = g.data.inputs.iter().chain(&g.data.outputs);
    for (n, (e, u)) in g.inputs.iter().chain(g.outputs).zip(uses).enumerate() {
        // Arbitrary semantics get their own string, system values none.
        let name_off = if e.semantic == Semantic::Arbitrary {
            let o = strings.len() as u32;
            strings.extend_from_slice(e.name.as_bytes());
            strings.push(0);
            o
        } else {
            0
        };
        let wanted: Vec<u32> = (0..e.rows).map(|r| e.index + r).collect();
        let at = indices.windows(wanted.len()).position(|w| w == wanted.as_slice()).unwrap_or_else(|| {
            let at = indices.len();
            indices.extend_from_slice(&wanted);
            at
        });
        let mut el = [0u8; 16];
        el[0..4].copy_from_slice(&name_off.to_le_bytes());
        el[4..8].copy_from_slice(&(at as u32).to_le_bytes());
        el[8] = e.rows as u8;
        el[9] = if e.allocated() { e.start_row as u8 } else { 0 };
        el[10] = if e.allocated() { e.cols | e.start_col << 4 | 0x40 } else { e.cols };
        el[11] = semantic_code(e.semantic) as u8;
        el[12] = register_component(e.component) as u8;
        el[13] = interp_code(e.interp) as u8;
        // Only the inputs carry the mask of the components indexed dynamically.
        el[14] = if n < g.inputs.len() { u.dynamic << if e.allocated() { e.start_col } else { 0 } } else { 0 };
        elems.push(el);
    }
    while !strings.len().is_multiple_of(4) {
        strings.push(0);
    }
    put(&mut v, strings.len() as u32);
    v.extend_from_slice(&strings);
    put(&mut v, indices.len() as u32);
    for i in &indices {
        put(&mut v, *i);
    }
    if !elems.is_empty() {
        put(&mut v, 16);
        // Inputs, then outputs, no patch constant elements.
        for el in &elems {
            v.extend_from_slice(el);
        }
    }
    for w in &g.data.psv_table {
        put(&mut v, *w);
    }
    v
}
