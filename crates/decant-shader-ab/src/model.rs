//! What a shader declares, independent of where the facts came from: its stage, its resource
//! bindings and its input and output signatures. The Windows side fills it from the D3D12
//! reflection of DXC; everything here works without a GPU.

use serde::{Deserialize, Serialize};

/// The pipeline stage of a shader. Only these three are run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    Vertex,
    Pixel,
    Compute,
}

impl Stage {
    /// `vs`, `ps` or `cs`, as the TSV and the command line write it.
    pub fn short(self) -> &'static str {
        match self {
            Stage::Vertex => "vs",
            Stage::Pixel => "ps",
            Stage::Compute => "cs",
        }
    }

    /// Reads `vs`, `ps`, `cs` (or `vertex`, `pixel`, `compute`). `auto` and `-` give `None`.
    pub fn parse(text: &str) -> Result<Option<Stage>, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "vs" | "vertex" => Ok(Some(Stage::Vertex)),
            "ps" | "pixel" => Ok(Some(Stage::Pixel)),
            "cs" | "compute" => Ok(Some(Stage::Compute)),
            "auto" | "-" | "" => Ok(None),
            other => Err(format!("unknown stage {other:?} (vs, ps, cs or auto)")),
        }
    }
}

/// The register class of a binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Class {
    Cbv,
    Srv,
    Uav,
    Sampler,
}

impl Class {
    /// The register letter: `b`, `t`, `u` or `s`.
    pub fn letter(self) -> char {
        match self {
            Class::Cbv => 'b',
            Class::Srv => 't',
            Class::Uav => 'u',
            Class::Sampler => 's',
        }
    }
}

/// The element type a texture or typed buffer returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ret {
    Float,
    Uint,
    Sint,
}

/// What a binding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    ConstantBuffer,
    Sampler { comparison: bool },
    RawBuffer,
    StructuredBuffer { stride: u32 },
    TypedBuffer(Ret),
    Texture1D(Ret),
    Texture1DArray(Ret),
    Texture2D(Ret),
    Texture2DArray(Ret),
    Texture2DMS(Ret),
    Texture3D(Ret),
    TextureCube(Ret),
    TextureCubeArray(Ret),
    /// Anything the runner does not create (acceleration structures, feedback maps). Bound as a
    /// null descriptor.
    Other(u32),
}

/// One resource binding: `count` registers from `lower` in `space` (`u32::MAX` when unbounded).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub class: Class,
    pub space: u32,
    pub lower: u32,
    pub count: u32,
    pub shape: Shape,
    /// The name the shader gave it, for messages only. Never compared.
    pub name: String,
}

impl Binding {
    /// Whether register `register` of `space` and `class` falls in this binding.
    pub fn covers(&self, class: Class, space: u32, register: u32) -> bool {
        self.class == class && self.space == space && register >= self.lower && (self.count == u32::MAX || register - self.lower < self.count)
    }

    /// `t3 space1 Texture2D(Float)`, without the name.
    pub fn describe(&self) -> String {
        let range = match self.count {
            1 => format!("{}{}", self.class.letter(), self.lower),
            u32::MAX => format!("{}{}..", self.class.letter(), self.lower),
            n => format!("{}{}..{}", self.class.letter(), self.lower, self.lower + n - 1),
        };
        format!("{range} space{} {:?}", self.space, self.shape)
    }
}

/// The scalar type of a signature element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Component {
    Float,
    Uint,
    Sint,
}

/// One input or output signature element.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Element {
    pub semantic: String,
    pub index: u32,
    pub register: u32,
    /// Component mask, bit 0 is `x`.
    pub mask: u8,
    /// `D3D_NAME`: 0 for an arbitrary semantic, 1 `SV_Position`, 64 to 71 the targets...
    pub system: u32,
    pub component: Component,
}

impl Element {
    /// The first component the mask names and how many follow it.
    pub fn span(&self) -> (u8, u8) {
        let mask = self.mask & 0xF;
        if mask == 0 {
            return (0, 0);
        }
        let first = mask.trailing_zeros() as u8;
        let last = 7 - mask.leading_zeros() as u8;
        (first, last - first + 1)
    }
}

/// `D3D_NAME_POSITION`.
pub const SV_POSITION: u32 = 1;
/// `D3D_NAME_TARGET`.
pub const SV_TARGET: u32 = 64;

/// What one shader declares.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reflection {
    pub stage: Stage,
    /// The shader model, for example `(6, 6)`.
    pub model: (u32, u32),
    /// Threads per group of a compute shader, `[0; 3]` otherwise.
    pub threads: [u32; 3],
    pub bindings: Vec<Binding>,
    pub inputs: Vec<Element>,
    pub outputs: Vec<Element>,
    /// It indexes `ResourceDescriptorHeap` directly.
    pub resource_heap: bool,
    /// It indexes `SamplerDescriptorHeap` directly.
    pub sampler_heap: bool,
    /// The heap slots it reads through those heaps whose index is known (see [`crate::heapmap`]).
    #[serde(default)]
    pub heap: Vec<crate::heapmap::HeapSlot>,
}

/// The differences between what two shaders declare, one line each, empty when they declare the
/// same. Names are ignored, everything else counts: a different resource set alone is a finding.
pub fn declaration_differences(a: &Reflection, b: &Reflection) -> Vec<String> {
    let mut out = Vec::new();
    if a.stage != b.stage {
        out.push(format!("stage {:?} against {:?}", a.stage, b.stage));
    }
    if a.stage == Stage::Compute && b.stage == Stage::Compute && a.threads != b.threads {
        out.push(format!("threads {:?} against {:?}", a.threads, b.threads));
    }
    let key = |x: &Binding| (x.class, x.space, x.lower);
    let mut left: Vec<&Binding> = a.bindings.iter().collect();
    let mut right: Vec<&Binding> = b.bindings.iter().collect();
    left.sort_by_key(|x| key(x));
    right.sort_by_key(|x| key(x));
    for x in &left {
        match right.iter().find(|y| key(y) == key(x)) {
            None => out.push(format!("only A: {}", x.describe())),
            Some(y) if y.count != x.count || y.shape != x.shape => out.push(format!("A {} against B {}", x.describe(), y.describe())),
            Some(_) => {}
        }
    }
    for y in &right {
        if !left.iter().any(|x| key(x) == key(y)) {
            out.push(format!("only B: {}", y.describe()));
        }
    }
    for (what, xs, ys) in [("input", &a.inputs, &b.inputs), ("output", &a.outputs, &b.outputs)] {
        let sig = |e: &Element| format!("{}{} r{} mask{:x} {:?}", e.semantic.to_ascii_uppercase(), e.index, e.register, e.mask, e.component);
        let xs: Vec<String> = xs.iter().map(sig).collect();
        let ys: Vec<String> = ys.iter().map(sig).collect();
        if xs != ys {
            out.push(format!("{what} signature [{}] against [{}]", xs.join(", "), ys.join(", ")));
        }
    }
    if (a.resource_heap, a.sampler_heap) != (b.resource_heap, b.sampler_heap) {
        out.push(format!("heap indexing resource {} sampler {} against resource {} sampler {}", a.resource_heap, a.sampler_heap, b.resource_heap, b.sampler_heap));
    }
    // Heap slots compare as sets: one index can be read as several kinds.
    let slot = |s: &crate::heapmap::HeapSlot| format!("{}{} {:?}", if s.class == Class::Sampler { "sampler heap " } else { "heap " }, s.index, s.shape);
    for s in a.heap.iter().filter(|s| !b.heap.contains(s)) {
        out.push(format!("only A: {}", slot(s)));
    }
    for s in b.heap.iter().filter(|s| !a.heap.contains(s)) {
        out.push(format!("only B: {}", slot(s)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refl(bindings: Vec<Binding>) -> Reflection {
        Reflection { stage: Stage::Compute, model: (6, 0), threads: [8, 8, 1], bindings, inputs: vec![], outputs: vec![], resource_heap: false, sampler_heap: false, heap: vec![] }
    }

    fn bind(class: Class, lower: u32, shape: Shape, name: &str) -> Binding {
        Binding { class, space: 0, lower, count: 1, shape, name: name.into() }
    }

    #[test]
    fn same_declarations_with_other_names_do_not_differ() {
        let a = refl(vec![bind(Class::Uav, 0, Shape::RawBuffer, "out"), bind(Class::Cbv, 0, Shape::ConstantBuffer, "cb")]);
        let b = refl(vec![bind(Class::Cbv, 0, Shape::ConstantBuffer, "k"), bind(Class::Uav, 0, Shape::RawBuffer, "o")]);
        assert!(declaration_differences(&a, &b).is_empty());
    }

    #[test]
    fn a_missing_and_a_changed_binding_are_found() {
        let a = refl(vec![bind(Class::Srv, 3, Shape::Texture2D(Ret::Float), "t"), bind(Class::Uav, 0, Shape::RawBuffer, "o")]);
        let b = refl(vec![bind(Class::Srv, 3, Shape::TypedBuffer(Ret::Float), "t"), bind(Class::Cbv, 2, Shape::ConstantBuffer, "c")]);
        let d = declaration_differences(&a, &b);
        assert_eq!(d.len(), 3, "{d:?}");
        assert!(d[0].starts_with("A t3 space0 Texture2D"), "{d:?}");
        assert!(d.iter().any(|l| l == "only A: u0 space0 RawBuffer"));
        assert!(d.iter().any(|l| l == "only B: b2 space0 ConstantBuffer"));
    }

    #[test]
    fn one_heap_index_read_as_two_kinds_does_not_differ_from_itself() {
        use crate::heapmap::HeapSlot;
        let mut a = refl(vec![]);
        a.heap = vec![HeapSlot { index: 36, class: Class::Srv, shape: Shape::Texture2D(Ret::Float) }, HeapSlot { index: 36, class: Class::Srv, shape: Shape::Texture3D(Ret::Float) }];
        assert!(declaration_differences(&a, &a.clone()).is_empty());
        let mut b = a.clone();
        b.heap.pop();
        assert_eq!(declaration_differences(&a, &b), vec!["only A: heap 36 Texture3D(Float)".to_string()]);
    }

    #[test]
    fn element_span_follows_the_mask() {
        let e = |mask| Element { semantic: "T".into(), index: 0, register: 0, mask, system: 0, component: Component::Float };
        assert_eq!(e(0b0011).span(), (0, 2));
        assert_eq!(e(0b1100).span(), (2, 2));
        assert_eq!(e(0b1111).span(), (0, 4));
        assert_eq!(e(0b0100).span(), (2, 1));
    }

    #[test]
    fn unbounded_bindings_cover_every_register_above() {
        let b = Binding { class: Class::Srv, space: 1, lower: 4, count: u32::MAX, shape: Shape::RawBuffer, name: String::new() };
        assert!(b.covers(Class::Srv, 1, 4) && b.covers(Class::Srv, 1, 1000));
        assert!(!b.covers(Class::Srv, 1, 3) && !b.covers(Class::Srv, 0, 4) && !b.covers(Class::Uav, 1, 4));
    }
}
