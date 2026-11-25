//! Stage inputs and outputs of vertex and pixel shaders: the signature elements behind the
//! `Input` and `Output` variables (and the members of interface blocks such as `gl_PerVertex`), and
//! the loads and stores through them.
//!
//! The mapping of Vulkan interface variables to the signature is the one a host has to know:
//!
//! * a vertex input with `Location` L is the element `ATTRIB` with semantic index L in register L,
//! * a vertex output and a pixel input with `Location` L is the element `TEXCOORD` with semantic
//!   index L in register L + 1 (register 0 is `SV_Position`),
//! * a pixel output with `Location` L is `SV_Target` L in register L,
//! * built ins become the system values `SV_Position`, `SV_ClipDistance`, `SV_CullDistance`,
//!   `SV_VertexID`, `SV_InstanceID`, `SV_IsFrontFace`, `SV_PrimitiveID`, `SV_SampleIndex`,
//!   `SV_Depth`, `SV_Coverage` and `SV_StencilRef`. Built ins without a signature element
//!   (`SubgroupLocalInvocationId` and `SubgroupSize`) are wave operations.

use rspirv::spirv::{BuiltIn, Decoration, StorageClass, Word};

use super::*;

/// One signature element and the part of an interface variable it holds.
#[derive(Clone, Debug)]
pub(crate) struct IoSlot {
    /// First scalar of the variable (in layout order) that belongs to the element.
    pub start: u32,
    pub len: u32,
    pub elem: u32,
    pub cols: u32,
    pub output: bool,
    /// The variable is a boolean the signature holds as an integer.
    pub bool_value: bool,
    /// A built in with no signature element (point size): writes are dropped.
    pub ignored: bool,
}

/// Where the decorations of an interface variable or of one member of an interface block are.
#[derive(Clone, Copy)]
struct Site {
    id: Word,
    member: Option<u32>,
}

impl Imp<'_> {
    fn site_lit(&self, s: Site, d: Decoration) -> Option<u32> {
        match s.member {
            Some(m) => self.member_deco_lit(s.id, m, d),
            None => self.deco_lit(s.id, d),
        }
    }

    fn site_has(&self, s: Site, d: Decoration) -> bool {
        match s.member {
            Some(m) => self.member_has_deco(s.id, m, d),
            None => self.has_deco(s.id, d),
        }
    }

    fn site_builtin(&self, s: Site) -> Option<BuiltIn> {
        let list = match s.member {
            Some(m) => self.member_decos.get(&(s.id, m)),
            None => self.decos.get(&s.id),
        }?;
        match list.iter().find(|(d, _)| *d == Decoration::BuiltIn)?.1.first() {
            Some(SOp::BuiltIn(b)) => Some(*b),
            _ => None,
        }
    }

    /// The rows and columns of an interface variable type.
    fn io_dims(&self, t: Word) -> R<(u32, u32)> {
        Ok(match self.types.get(&t) {
            Some(T::Bool | T::Int(..) | T::Float(_)) => (1, 1),
            Some(T::Vector(_, n)) => (1, *n),
            Some(T::Matrix(c, n)) => {
                let (_, rows_cols) = self.io_dims(*c)?;
                (*n, rows_cols)
            }
            Some(T::Array(e, Some(n))) => {
                let (r, c) = self.io_dims(*e)?;
                if r != 1 {
                    return fail("io:array-of-matrices", "an array of matrices on the interface");
                }
                (*n as u32, c)
            }
            other => return fail("io:type", format!("an interface variable of type {other:?}")),
        })
    }

    fn io_component(&self, t: Word) -> R<ComponentType> {
        match self.types.get(&t) {
            Some(T::Vector(e, _)) | Some(T::Matrix(e, _)) | Some(T::Array(e, _)) => self.io_component(*e),
            Some(T::Float(32)) => Ok(ComponentType::F32),
            Some(T::Int(32, true)) => Ok(ComponentType::I32),
            Some(T::Int(32, false)) => Ok(ComponentType::U32),
            Some(T::Bool) => Ok(ComponentType::U32),
            other => fail("io:component", format!("interface components of type {other:?}")),
        }
    }

    fn interpolation(&self, s: Site, component: ComponentType) -> Interp {
        if component != ComponentType::F32 || self.site_has(s, Decoration::Flat) {
            return Interp::Constant;
        }
        let (centroid, sample, noperspective) = (self.site_has(s, Decoration::Centroid), self.site_has(s, Decoration::Sample), self.site_has(s, Decoration::NoPerspective));
        match (noperspective, centroid, sample) {
            (true, _, true) => Interp::LinearNoperspectiveSample,
            (true, true, _) => Interp::LinearNoperspectiveCentroid,
            (true, ..) => Interp::LinearNoperspective,
            (_, _, true) => Interp::LinearSample,
            (_, true, _) => Interp::LinearCentroid,
            _ => Interp::Linear,
        }
    }

    /// Builds the signature from the interface variables of the entry point.
    pub fn setup_io(&mut self) -> R<()> {
        let stage = self.stage.clone();
        if matches!(stage, Some(Stage::Compute { .. }) | None) {
            return Ok(());
        }
        let pixel = matches!(stage, Some(Stage::Pixel));
        let mut ids: Vec<Word> = self.vars.iter().filter(|(_, v)| matches!(v.storage, StorageClass::Input | StorageClass::Output)).map(|(id, _)| *id).collect();
        ids.sort_unstable();
        // The sites: a variable, or the members of an interface block.
        let mut sites: Vec<(Word, Site, Word, bool)> = Vec::new();
        for id in &ids {
            let v = &self.vars[id];
            let output = v.storage == StorageClass::Output;
            match self.types.get(&v.ty).cloned() {
                Some(T::Struct(members)) => {
                    for (m, mt) in members.iter().enumerate() {
                        sites.push((*id, Site { id: v.ty, member: Some(m as u32) }, *mt, output));
                    }
                }
                _ => sites.push((*id, Site { id: *id, member: None }, v.ty, output)),
            }
        }
        // The registers after the last location go to the system values that need one.
        let mut next_in = 0u32;
        for (_, site, ty, output) in &sites {
            if let Some(loc) = self.site_lit(*site, Decoration::Location) {
                let (rows, _) = self.io_dims(*ty)?;
                let base = if *output == pixel { 0 } else { 1 };
                if !*output {
                    next_in = next_in.max(loc + base + rows);
                }
            }
        }
        next_in = next_in.max(1);
        let mut start_of: HashMap<Word, u32> = HashMap::new();
        for (var, site, ty, output) in sites {
            let builtin = self.site_builtin(site);
            let flat_len = self.flat_count(ty)?;
            let start = *start_of.entry(var).or_insert(0);
            *start_of.get_mut(&var).unwrap() += flat_len;
            // Built ins that are wave operations in the shader, not signature elements.
            if matches!(builtin, Some(BuiltIn::SubgroupLocalInvocationId | BuiltIn::SubgroupSize | BuiltIn::HelperInvocation)) {
                continue;
            }
            let (mut rows, mut cols) = self.io_dims(ty)?;
            let component = self.io_component(ty)?;
            let mut el = SigElement { name: String::new(), index: 0, semantic: Semantic::Arbitrary, component, interp: Interp::Undefined, start_row: 0, start_col: 0, rows, cols: cols as u8 };
            let mut bool_value = false;
            let mut ignored = false;
            let mut location = None;
            match builtin {
                None => {
                    let loc = self.site_lit(site, Decoration::Location).ok_or_else(|| ImportError { category: "io:no-location".into(), detail: "an interface variable without a location".into() })?;
                    location = Some(loc);
                    el.start_col = self.site_lit(site, Decoration::Component).unwrap_or(0) as u8;
                    el.index = loc;
                    match (pixel, output) {
                        (false, false) => {
                            el.name = "ATTRIB".into();
                            el.start_row = loc;
                        }
                        (true, true) => {
                            el.name = "SV_Target".into();
                            el.semantic = Semantic::Target;
                            el.start_row = loc;
                        }
                        _ => {
                            el.name = "TEXCOORD".into();
                            el.start_row = loc + 1;
                            el.interp = self.interpolation(site, component);
                        }
                    }
                    if el.semantic == Semantic::Arbitrary
                        && let Some((name, index)) = self.bindings.as_ref().and_then(|b| b.name_of(output, loc, u32::from(el.start_col)))
                    {
                        el.name = name.clone();
                        el.index = *index;
                        // The location is the console register, so an interpolant sits there too.
                        el.start_row = loc;
                    }
                }
                Some(b) => {
                    let mut alloc = || {
                        let r = next_in;
                        next_in += 1;
                        r
                    };
                    match (pixel, output, b) {
                        (false, true, BuiltIn::Position) | (true, false, BuiltIn::FragCoord) => {
                            el.name = "SV_Position".into();
                            el.semantic = Semantic::Position;
                            el.interp = Interp::LinearNoperspective;
                            el.start_row = self.bindings.as_ref().and_then(|b| b.system_of(output, "SV_Position")).unwrap_or(0);
                        }
                        (false, true, BuiltIn::PointSize) => ignored = true,
                        (false, true, BuiltIn::ClipDistance | BuiltIn::CullDistance) | (true, false, BuiltIn::ClipDistance | BuiltIn::CullDistance) => {
                            // An array of floats packed four to a register.
                            let n = flat_len;
                            cols = n.min(4);
                            rows = n.div_ceil(4);
                            el.name = if b == BuiltIn::ClipDistance { "SV_ClipDistance".into() } else { "SV_CullDistance".into() };
                            el.semantic = if b == BuiltIn::ClipDistance { Semantic::ClipDistance } else { Semantic::CullDistance };
                            el.interp = Interp::Linear;
                            el.rows = rows;
                            el.cols = cols as u8;
                            let console = self.bindings.as_ref().and_then(|b| b.system_of(output, &el.name));
                            el.start_row = match console {
                                Some(r) => r,
                                None if output => 1 + self.deco_next_location(pixel),
                                None => alloc(),
                            };
                        }
                        (false, false, BuiltIn::VertexIndex) => {
                            el.name = "SV_VertexID".into();
                            el.semantic = Semantic::VertexId;
                            el.component = ComponentType::U32;
                            el.start_row = alloc();
                        }
                        (false, false, BuiltIn::InstanceIndex) => {
                            el.name = "SV_InstanceID".into();
                            el.semantic = Semantic::InstanceId;
                            el.component = ComponentType::U32;
                            el.start_row = alloc();
                        }
                        (true, false, BuiltIn::FrontFacing) => {
                            el.name = "SV_IsFrontFace".into();
                            el.semantic = Semantic::IsFrontFace;
                            el.interp = Interp::Constant;
                            el.start_row = alloc();
                            bool_value = true;
                        }
                        (true, false, BuiltIn::PrimitiveId) => {
                            el.name = "SV_PrimitiveID".into();
                            el.semantic = Semantic::PrimitiveId;
                            el.interp = Interp::Constant;
                            el.start_row = alloc();
                        }
                        (true, false, BuiltIn::BaryCoordKHR | BuiltIn::BaryCoordNoPerspKHR) => {
                            let perspective = b == BuiltIn::BaryCoordKHR;
                            el.name = "SV_Barycentrics".into();
                            el.semantic = Semantic::Barycentrics;
                            el.index = if perspective { 0 } else { 1 };
                            el.interp = if perspective { Interp::Linear } else { Interp::LinearNoperspective };
                            el.start_row = SigElement::UNALLOCATED;
                        }
                        (true, false, BuiltIn::SampleId) => {
                            el.name = "SV_SampleIndex".into();
                            el.semantic = Semantic::SampleIndex;
                            el.interp = Interp::Constant;
                            el.start_row = SigElement::UNALLOCATED;
                        }
                        (true, true, BuiltIn::FragDepth) => {
                            el.name = "SV_Depth".into();
                            el.semantic = Semantic::Depth;
                            el.start_row = SigElement::UNALLOCATED;
                        }
                        (true, true, BuiltIn::SampleMask) => {
                            el.name = "SV_Coverage".into();
                            el.semantic = Semantic::Coverage;
                            el.start_row = SigElement::UNALLOCATED;
                            el.component = ComponentType::U32;
                        }
                        (true, true, BuiltIn::FragStencilRefEXT) => {
                            el.name = "SV_StencilRef".into();
                            el.semantic = Semantic::StencilRef;
                            el.start_row = SigElement::UNALLOCATED;
                        }
                        (_, _, other) => return fail(format!("builtin:{other:?}"), "this built in has no signature element yet"),
                    }
                }
            }
            if ignored {
                self.io.entry(var).or_default().push(IoSlot { start, len: flat_len, elem: 0, cols, output, bool_value, ignored: true });
                continue;
            }
            if el.allocated() && el.start_row + el.rows > 32 {
                return fail("io:too-many-registers", "an interface that needs more than 32 registers");
            }
            if matches!(el.semantic, Semantic::Coverage | Semantic::Depth | Semantic::StencilRef | Semantic::SampleIndex) {
                el.rows = 1;
            }
            let list = if output { &mut self.outputs } else { &mut self.inputs };
            let elem = list.len() as u32;
            list.push(el);
            self.io_notes.push(ImportedIo { output, element: elem, location, builtin: builtin.map(|b| format!("{b:?}")) });
            self.io.entry(var).or_default().push(IoSlot { start, len: flat_len, elem, cols, output, bool_value, ignored: false });
        }
        Ok(())
    }

    /// The first register after the interpolated outputs (for clip and cull distances).
    fn deco_next_location(&self, pixel: bool) -> u32 {
        let mut top = 0u32;
        for (id, v) in &self.vars {
            if v.storage != StorageClass::Output || pixel {
                continue;
            }
            if let Some(T::Struct(ms)) = self.types.get(&v.ty) {
                for (m, mt) in ms.iter().enumerate() {
                    if let Some(l) = self.member_deco_lit(v.ty, m as u32, Decoration::Location) {
                        top = top.max(l + self.io_dims(*mt).map(|d| d.0).unwrap_or(1));
                    }
                }
            } else if let Some(l) = self.deco_lit(*id, Decoration::Location) {
                top = top.max(l + self.io_dims(v.ty).map(|d| d.0).unwrap_or(1));
            }
        }
        top
    }

    /// The row operand and column of the scalar `local` of a slot, for a pointer into the variable.
    fn io_slot_pos(&mut self, slot: &IoSlot, p: &Ptr, local: u32) -> (Operand, u8) {
        let (row, col) = (local / slot.cols, local % slot.cols);
        let row = match p.dynoff {
            None => Operand::i32(row),
            Some(d) => {
                // The dynamic part counts scalars: whole elements of an array of rows.
                let r = if slot.cols == 1 { d } else { self.emit1(Op_::Bin(BinOp::UDiv, d, Operand::i32(slot.cols))) };
                if row == 0 { r } else { self.emit1(Op_::Bin(BinOp::Add, r, Operand::i32(row))) }
            }
        };
        (row, col as u8)
    }

    fn slot_of(&self, p: &Ptr, flat: u32) -> R<IoSlot> {
        let slots = self.io.get(&p.var).ok_or_else(|| ImportError { category: "io:unknown".into(), detail: "an interface variable the signature does not know".into() })?;
        slots.iter().find(|s| flat >= s.start && flat < s.start + s.len).cloned().ok_or_else(|| ImportError { category: "io:unknown".into(), detail: "a scalar of an interface block that has no element".into() })
    }

    /// A load from an input variable of a vertex or pixel shader.
    pub fn load_io(&mut self, p: &Ptr) -> R<Val> {
        let n = self.flat_count(p.ty)?;
        let mut out = Vec::new();
        for k in 0..n {
            let slot = self.slot_of(p, p.off + k)?;
            if slot.output {
                return fail("io:output-read", "a read of an output variable");
            }
            let (row, col) = self.io_slot_pos(&slot, p, p.off + k - slot.start);
            let v = self.emit1(Op_::LoadInput { elem: slot.elem, row, col });
            out.push(if slot.bool_value { self.emit1(Op_::ICmp(IPred::Ne, v, Operand::i32(0))) } else { v });
        }
        Ok(Val::S(out))
    }

    /// A store to an output variable.
    pub fn store_io(&mut self, p: &Ptr, value: &[Operand]) -> R<()> {
        for (k, v) in value.iter().enumerate() {
            let slot = self.slot_of(p, p.off + k as u32)?;
            if slot.ignored {
                continue;
            }
            let (row, col) = self.io_slot_pos(&slot, p, p.off + k as u32 - slot.start);
            self.emit0(Op_::StoreOutput { elem: slot.elem, row, col, value: *v });
        }
        Ok(())
    }
}
