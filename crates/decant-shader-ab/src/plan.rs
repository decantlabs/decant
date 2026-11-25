//! The plan of a pair: one root layout and one set of generated resources for both sides, and per
//! side what its own signatures need (the vertex shader that feeds a pixel shader, the input layout
//! and stream output of a vertex shader, the render targets). Built from the two reflections
//! without a GPU, so it can be tested and written to a file for the subprocess of each side.
//!
//! Neither side is taken as the truth: each side gets inputs that fit its own signature, fed by
//! semantic (or by register), so a side that declares one interpolated input and a side that
//! declares four still run on the same generated values.

use serde::{Deserialize, Serialize};

use crate::model::{Binding, Class, Component, Element, Reflection, Ret, SV_POSITION, SV_TARGET, Shape, Stage};

/// A texture or typed buffer format the runner creates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    Rgba32F,
    Rgba32U,
    Rgba32I,
    Rgba16F,
    Rgba8,
}

impl Format {
    pub fn parse(text: &str) -> Result<Format, String> {
        match text.to_ascii_lowercase().as_str() {
            "rgba32f" => Ok(Format::Rgba32F),
            "rgba32u" => Ok(Format::Rgba32U),
            "rgba32i" => Ok(Format::Rgba32I),
            "rgba16f" => Ok(Format::Rgba16F),
            "rgba8" => Ok(Format::Rgba8),
            other => Err(format!("unknown format {other:?} (rgba32f, rgba32u, rgba32i, rgba16f, rgba8)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Rgba32F => "rgba32f",
            Format::Rgba32U => "rgba32u",
            Format::Rgba32I => "rgba32i",
            Format::Rgba16F => "rgba16f",
            Format::Rgba8 => "rgba8",
        }
    }

    /// Bytes of one texel.
    pub fn texel_size(self) -> u32 {
        match self {
            Format::Rgba32F | Format::Rgba32U | Format::Rgba32I => 16,
            Format::Rgba16F => 8,
            Format::Rgba8 => 4,
        }
    }

    pub fn elem(self) -> crate::compare::Elem {
        use crate::compare::Elem;
        match self {
            Format::Rgba32F => Elem::F32,
            Format::Rgba32U => Elem::U32,
            Format::Rgba32I => Elem::I32,
            Format::Rgba16F => Elem::F16,
            Format::Rgba8 => Elem::Unorm8,
        }
    }

    fn for_ret(ret: Ret, sampled: bool) -> Format {
        match ret {
            Ret::Float if sampled => Format::Rgba16F,
            Ret::Float => Format::Rgba32F,
            Ret::Uint => Format::Rgba32U,
            Ret::Sint => Format::Rgba32I,
        }
    }
}

/// The dimension of a generated texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    D1,
    D1Array,
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// How a buffer is viewed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufView {
    Raw,
    Structured(u32),
    Typed(Format),
}

/// What fills one descriptor or root parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Res {
    /// A null descriptor of the class: nothing the shaders declare lives here.
    Null,
    Cbv { bytes: u32 },
    Buffer { bytes: u32, view: BufView, uav: bool },
    /// `size` is width, height and depth (3D) or layers (arrays, six per cube).
    Texture { dim: Dim, size: [u32; 3], mips: u32, format: Format, uav: bool },
}

/// One descriptor range of a table, with the resource of every descriptor in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    pub class: Class,
    pub base: u32,
    pub space: u32,
    /// The range as the root signature declares it (`u32::MAX` unbounded).
    pub declared: u32,
    /// Its offset in the table, `None` to follow the range before it.
    pub offset: Option<u32>,
    /// One entry per descriptor the runner fills.
    pub slots: Vec<Res>,
}

/// One root parameter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Param {
    Table(Vec<Range>),
    Constants { register: u32, space: u32, count: u32 },
    /// A root CBV, SRV or UAV.
    Root { class: Class, register: u32, space: u32, res: Res },
}

/// A static sampler of a synthesized root signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sampler {
    pub register: u32,
    pub space: u32,
    pub comparison: bool,
}

/// The root signature both sides run with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Root {
    /// `D3D12_ROOT_SIGNATURE_FLAGS`.
    pub flags: u32,
    pub params: Vec<Param>,
    /// Static samplers to create when the layout is synthesized. A given blob carries its own.
    pub samplers: Vec<Sampler>,
    /// The serialized root signature given by the user, used as it is.
    pub blob: Option<Vec<u8>>,
}

pub const FLAG_INPUT_LAYOUT: u32 = 0x1;
pub const FLAG_STREAM_OUTPUT: u32 = 0x40;
pub const FLAG_RESOURCE_HEAP: u32 = 0x400;
pub const FLAG_SAMPLER_HEAP: u32 = 0x800;

/// Register space and registers of the device memory window of the SPIR-V importer: the root UAV
/// `u0` and the CBV `b0` whose first eight bytes hold the address of that UAV.
pub const WINDOW_SPACE: u32 = 14;

/// Register space and register of the user data mirror, the constant buffer through which translated
/// shaders get their user data (descriptor heap indices among it).
pub const MIRROR_SPACE: u32 = 15;
pub const MIRROR_REGISTER: u32 = 1;

/// What one side runs besides its own container.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Side {
    pub reflection: Reflection,
    /// Pixel pairs: the HLSL of the vertex shader that feeds this side's inputs.
    pub feeder: Option<String>,
    /// Vertex pairs: this side's vertex input elements, each a four component 32 bit attribute.
    pub vertex_inputs: Vec<Element>,
}

/// How inputs are matched between the sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feed {
    /// By semantic name and index: `TEXCOORD3` gets the same values on both sides.
    Semantic,
    /// By signature register: register 3 gets the same values on both sides.
    Location,
}

/// The user's options for a plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub seed: u64,
    /// Render target size, pixels per side.
    pub size: u32,
    /// Compute dispatch.
    pub groups: [u32; 3],
    /// Vertices of a vertex pair, drawn as a point list.
    pub points: u32,
    /// Grid cells per side of the pixel pair's mesh.
    pub grid: u32,
    /// Render target format instead of the one the output type implies.
    pub target_format: Option<Format>,
    pub feed: Feed,
    /// Explicit constant buffer contents: `(space, register, bytes)`.
    pub constants: Vec<(u32, u32, Vec<u8>)>,
}

impl Default for Options {
    fn default() -> Self {
        Options { seed: 1, size: 64, groups: [2, 2, 1], points: 256, grid: 4, target_format: None, feed: Feed::Semantic, constants: Vec::new() }
    }
}

/// One render target of a pixel pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    pub index: u32,
    pub format: Format,
}

/// Everything both sides need.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub stage: Stage,
    pub options: Options,
    pub root: Root,
    pub targets: Vec<Target>,
    pub sides: [Side; 2],
    /// What the two sides declare differently: a finding of its own.
    pub declarations: Vec<String>,
    /// Approximations the runner made.
    pub notes: Vec<String>,
    /// The heap slots of both sides whose index is known, A's kind first where they disagree.
    pub heap: Vec<crate::heapmap::HeapSlot>,
}

/// Descriptors filled for an unbounded range.
pub const UNBOUNDED_FILL: u32 = 16;
/// Bytes of generated buffers and constant buffers.
pub const BUFFER_BYTES: u32 = 65536;

/// The resource for register `register` of `space` and `class`, from the first binding that covers
/// it (A's before B's), or a null descriptor.
pub fn resource_for(bindings: &[&Binding], class: Class, space: u32, register: u32) -> Res {
    let Some(b) = bindings.iter().find(|b| b.covers(class, space, register)) else {
        return if class == Class::Cbv { Res::Cbv { bytes: BUFFER_BYTES } } else { Res::Null };
    };
    res_of(class, b.shape)
}

/// The generated resource for a view of `class` and `shape`.
pub fn res_of(class: Class, shape: Shape) -> Res {
    let uav = class == Class::Uav;
    let tex = |dim, size, ret| Res::Texture { dim, size, mips: if uav { 1 } else { mip_count(size, dim) }, format: Format::for_ret(ret, !uav), uav };
    match shape {
        Shape::ConstantBuffer => Res::Cbv { bytes: BUFFER_BYTES },
        Shape::RawBuffer => Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Raw, uav },
        Shape::StructuredBuffer { stride } => Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Structured(stride.max(4)), uav },
        Shape::TypedBuffer(ret) => Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Typed(Format::for_ret(ret, false)), uav },
        Shape::Texture1D(r) => tex(Dim::D1, [256, 1, 1], r),
        Shape::Texture1DArray(r) => tex(Dim::D1Array, [256, 1, 4], r),
        Shape::Texture2D(r) => tex(Dim::D2, [64, 64, 1], r),
        Shape::Texture2DArray(r) => tex(Dim::D2Array, [64, 64, 4], r),
        Shape::Texture3D(r) => tex(Dim::D3, [16, 16, 16], r),
        Shape::TextureCube(r) => tex(Dim::Cube, [32, 32, 6], r),
        Shape::TextureCubeArray(r) => tex(Dim::CubeArray, [32, 32, 12], r),
        Shape::Texture2DMS(_) | Shape::Other(_) | Shape::Sampler { .. } => Res::Null,
    }
}

fn mip_count(size: [u32; 3], dim: Dim) -> u32 {
    let largest = match dim {
        Dim::D3 => size[0].max(size[1]).max(size[2]),
        _ => size[0].max(size[1]),
    };
    32 - largest.max(1).leading_zeros()
}

/// Every binding of both sides, A first.
fn union<'a>(a: &'a Reflection, b: &'a Reflection) -> Vec<&'a Binding> {
    a.bindings.iter().chain(&b.bindings).collect()
}

/// A root layout with every binding of both sides: one descriptor table for constant buffers and
/// views, static samplers for samplers. Constant buffers stay in the table rather than in root
/// descriptors, because views are bounds checked and root descriptors are not: a shader that
/// indexes a constant buffer with generated data must not read past it.
pub fn synthesize_root(a: &Reflection, b: &Reflection, stage: Stage, notes: &mut Vec<String>) -> Root {
    let all = union(a, b);
    // One entry per (class, space, register range), A's shape wins.
    let mut keys: Vec<(Class, u32, u32, u32)> = Vec::new();
    for x in &all {
        let count = if x.count == u32::MAX { UNBOUNDED_FILL } else { x.count };
        if !keys.iter().any(|k| (k.0, k.1) == (x.class, x.space) && ranges_overlap(k.2, k.3, x.lower, count)) {
            keys.push((x.class, x.space, x.lower, count));
        }
        if x.count == u32::MAX {
            notes.push(format!("unbounded {} filled with {UNBOUNDED_FILL} descriptors", x.describe()));
        }
    }
    keys.sort();
    let mut params = Vec::new();
    let mut ranges = Vec::new();
    let mut samplers = Vec::new();
    for &(class, space, lower, count) in &keys {
        match class {
            Class::Sampler => {
                for r in lower..lower + count {
                    let comparison = all.iter().find(|x| x.covers(class, space, r)).is_some_and(|x| matches!(x.shape, Shape::Sampler { comparison: true }));
                    samplers.push(Sampler { register: r, space, comparison });
                }
            }
            _ => ranges.push(Range { class, base: lower, space, declared: count, offset: None, slots: (lower..lower + count).map(|r| resource_for(&all, class, space, r)).collect() }),
        }
    }
    if !ranges.is_empty() {
        params.push(Param::Table(ranges));
    }
    let mut flags = match stage {
        Stage::Vertex => FLAG_INPUT_LAYOUT | FLAG_STREAM_OUTPUT,
        Stage::Pixel => FLAG_INPUT_LAYOUT,
        Stage::Compute => 0,
    };
    if a.resource_heap || b.resource_heap {
        flags |= FLAG_RESOURCE_HEAP;
    }
    if a.sampler_heap || b.sampler_heap {
        flags |= FLAG_SAMPLER_HEAP;
    }
    Root { flags, params, samplers, blob: None }
}

fn ranges_overlap(a: u32, an: u32, b: u32, bn: u32) -> bool {
    a < b.saturating_add(bn) && b < a.saturating_add(an)
}

/// Fills the resources of a root layout read from a given root signature.
pub fn fill_given_root(root: &mut Root, a: &Reflection, b: &Reflection, notes: &mut Vec<String>) {
    let all = union(a, b);
    for p in &mut root.params {
        match p {
            Param::Table(ranges) => {
                for r in ranges {
                    let n = if r.declared == u32::MAX {
                        // Fill as far as a binding of either side reaches, at least the default.
                        let reach = all.iter().filter(|x| x.class == r.class && x.space == r.space && x.lower >= r.base).map(|x| x.lower - r.base + if x.count == u32::MAX { UNBOUNDED_FILL } else { x.count }).max().unwrap_or(0);
                        notes.push(format!("unbounded range {}{} space{} filled with {} descriptors", r.class.letter(), r.base, r.space, reach.max(UNBOUNDED_FILL)));
                        reach.max(UNBOUNDED_FILL)
                    } else {
                        r.declared
                    };
                    if r.class != Class::Sampler {
                        r.slots = (r.base..r.base.saturating_add(n)).map(|reg| resource_for(&all, r.class, r.space, reg)).collect();
                    } else {
                        r.slots = vec![Res::Null; n as usize];
                    }
                }
            }
            Param::Root { class, register, space, res } => {
                *res = match resource_for(&all, *class, *space, *register) {
                    // A root SRV or UAV is a buffer address: textures cannot go there.
                    Res::Null | Res::Texture { .. } if *class != Class::Cbv => Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Raw, uav: *class == Class::Uav },
                    other => other,
                };
            }
            Param::Constants { .. } => {}
        }
    }
}

/// The 32 bit hash the feeder shaders use (a public domain integer hash by Chris Wellons).
pub fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The key that selects the values of an input element: its semantic and index, or its register.
pub fn feed_key(e: &Element, feed: Feed, seed: u64) -> u32 {
    let text = match feed {
        Feed::Semantic => format!("{}{}", e.semantic.to_ascii_uppercase(), e.index),
        Feed::Location => format!("r{}", e.register),
    };
    let mut rng = crate::input::Rng::new(seed, &text);
    rng.next_u32()
}

fn hlsl_type(c: Component, n: u8) -> String {
    let base = match c {
        Component::Float => "float",
        Component::Uint => "uint",
        Component::Sint => "int",
    };
    if n == 1 { base.to_string() } else { format!("{base}{n}") }
}

/// `D3D_NAME_CLIP_DISTANCE` and `D3D_NAME_CULL_DISTANCE`: system values a vertex shader writes.
const SV_CLIP_DISTANCE: u32 = 2;
const SV_CULL_DISTANCE: u32 = 3;

/// The inputs of a pixel shader the feeder writes: arbitrary semantics, and the clip and cull
/// distances, which get values in `[0, 1)` so that nothing is clipped or culled.
pub fn fed_inputs(inputs: &[Element]) -> Vec<&Element> {
    let mut v: Vec<&Element> = inputs.iter().filter(|e| matches!(e.system, 0 | SV_CLIP_DISTANCE | SV_CULL_DISTANCE)).collect();
    v.sort_by_key(|e| (e.register, e.span().0));
    v
}

/// The HLSL of a vertex shader that draws a `grid` by `grid` mesh over the whole target and writes
/// every non system input of the pixel shader. Each grid point gets values from the hash of its
/// index and the element's key, floats in `[0, 1)` and integers in `[0, 64)`, so neighbouring
/// triangles agree and the pixel shader sees smooth gradients.
pub fn feeder_hlsl(inputs: &[Element], grid: u32, feed: Feed, seed: u64) -> String {
    let fed = fed_inputs(inputs);
    let mut s = String::from("// Generated by decant-shader-ab: feeds the pixel shader's inputs.\n");
    s.push_str("uint h(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }\n");
    s.push_str("float f(uint x) { return (h(x) >> 8) * (1.0 / 16777216.0); }\n");
    // The outputs follow the registers of the pixel shader's inputs, so that DXC packs them into
    // the same registers: unused rows and leading components get padding outputs, the position
    // goes where the pixel shader has it, else after everything else.
    let position = inputs.iter().find(|e| e.system == SV_POSITION).map_or(u32::MAX, |e| e.register);
    s.push_str("struct VOut {\n");
    let mut pads = 0u32;
    let mut pad = |s: &mut String, n: u8| {
        let t = if n == 1 { "float".to_string() } else { format!("float{n}") };
        s.push_str(&format!("    {t} pad{pads} : DECANT_AB_PAD{pads};\n"));
        pads += 1;
    };
    let mut row = 0u32;
    let mut i = 0usize;
    let mut placed = false;
    loop {
        let next = fed.get(i).map_or(u32::MAX, |e| e.register);
        if !placed && (position == row || (position == u32::MAX && next == u32::MAX)) {
            s.push_str("    float4 pos : SV_Position;\n");
            placed = true;
            row += 1;
            continue;
        }
        if next == u32::MAX {
            if placed {
                break;
            }
            // The position sits in a row past every fed input.
            pad(&mut s, 4);
            row += 1;
            continue;
        }
        if row < next {
            pad(&mut s, 4);
            row += 1;
            continue;
        }
        // Every element of this row, in component order, with padding before a gap.
        let mut col = 0u8;
        while let Some(e) = fed.get(i).filter(|e| e.register == next) {
            let (start, n) = e.span();
            if start > col {
                pad(&mut s, start - col);
            }
            let interp = if e.component == Component::Float { "" } else { "nointerpolation " };
            s.push_str(&format!("    {interp}{} v{i} : {}{};\n", hlsl_type(e.component, n.max(1)), e.semantic, e.index));
            col = start + n.max(1);
            i += 1;
        }
        row = next + 1;
    }
    s.push_str("};\n");
    s.push_str(&format!("static const uint N = {grid};\n"));
    s.push_str("static const uint2 corner[6] = { uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(0, 1), uint2(1, 0), uint2(1, 1) };\n");
    s.push_str("VOut main(uint vid : SV_VertexID) {\n");
    s.push_str("    uint q = vid / 6;\n    uint2 p = uint2(q % N, q / N) + corner[vid % 6];\n    uint g = p.y * (N + 1) + p.x;\n");
    s.push_str("    VOut o = (VOut)0;\n    o.pos = float4(p.x * 2.0 / N - 1.0, 1.0 - p.y * 2.0 / N, 0.5, 1.0);\n");
    for (i, e) in fed.iter().enumerate() {
        let n = e.span().1.max(1);
        let key = feed_key(e, feed, seed);
        let comps: Vec<String> = (0..n)
            .map(|c| {
                let arg = format!("g * 4u + {}u + 0x{key:08x}u", c + e.span().0);
                match e.component {
                    Component::Float => format!("f({arg})"),
                    Component::Uint => format!("h({arg}) % 64u"),
                    Component::Sint => format!("int(h({arg}) % 64u)"),
                }
            })
            .collect();
        s.push_str(&format!("    o.v{i} = {}({});\n", hlsl_type(e.component, n), comps.join(", ")));
    }
    s.push_str("    return o;\n}\n");
    s
}

/// The value of component `c` of vertex input element `e` at vertex `v`, as the vertex buffer of a
/// vertex pair holds it: floats in `[-1, 1)`, integers in `[0, 64)`.
pub fn vertex_value(e: &Element, feed: Feed, seed: u64, v: u32, c: u32) -> u32 {
    let x = hash32(v.wrapping_mul(4).wrapping_add(c).wrapping_add(feed_key(e, feed, seed)));
    match e.component {
        Component::Float => ((x >> 8) as f32 / 8_388_608.0 - 1.0).to_bits(),
        Component::Uint | Component::Sint => x % 64,
    }
}

/// The render targets of a pixel pair: every `SV_Target` either side writes.
fn targets(a: &Reflection, b: &Reflection, forced: Option<Format>) -> Vec<Target> {
    let mut out: Vec<Target> = Vec::new();
    for e in a.outputs.iter().chain(&b.outputs) {
        if e.system != SV_TARGET && !e.semantic.eq_ignore_ascii_case("SV_Target") {
            continue;
        }
        if out.iter().any(|t| t.index == e.index) {
            continue;
        }
        let format = forced.unwrap_or(match e.component {
            Component::Float => Format::Rgba32F,
            Component::Uint => Format::Rgba32U,
            Component::Sint => Format::Rgba32I,
        });
        out.push(Target { index: e.index, format });
    }
    out.sort_by_key(|t| t.index);
    out
}

/// Builds the plan of a pair. `stage` overrides the reflected stage. `root` is a layout read from a
/// root signature the user gave, or `None` to synthesize one.
pub fn build(a: Reflection, b: Reflection, stage: Option<Stage>, root: Option<Root>, options: Options) -> Result<Plan, String> {
    let stage = stage.unwrap_or(a.stage);
    let declarations = crate::model::declaration_differences(&a, &b);
    if a.stage != stage && b.stage != stage {
        return Err(format!("neither side is a {} shader", stage.short()));
    }
    let mut notes = Vec::new();
    let root = match root {
        Some(mut r) => {
            fill_given_root(&mut r, &a, &b, &mut notes);
            r
        }
        None => synthesize_root(&a, &b, stage, &mut notes),
    };
    let targets = if stage == Stage::Pixel { targets(&a, &b, options.target_format) } else { Vec::new() };
    if stage == Stage::Pixel {
        for e in a.outputs.iter().chain(&b.outputs) {
            if e.system != SV_TARGET && !e.semantic.to_ascii_uppercase().starts_with("SV_TARGET") {
                notes.push(format!("output {}{} is not compared", e.semantic, e.index));
            }
        }
        for e in a.inputs.iter().chain(&b.inputs) {
            if !matches!(e.system, 0 | SV_POSITION | SV_CLIP_DISTANCE | SV_CULL_DISTANCE) {
                notes.push(format!("system input {}{} takes the rasterizer's value", e.semantic, e.index));
            }
        }
    }
    let mut heap: Vec<crate::heapmap::HeapSlot> = Vec::new();
    for s in a.heap.iter().chain(&b.heap) {
        if !heap.iter().any(|h| (h.index, h.class == Class::Sampler) == (s.index, s.class == Class::Sampler)) {
            heap.push(s.clone());
        }
    }
    if a.resource_heap || b.resource_heap || a.sampler_heap || b.sampler_heap {
        notes.push(format!("descriptor heap indexed: the user data mirror holds 64 times its dword number, {} slots of known kind, the others null", heap.len()));
    }
    let side = |r: Reflection| {
        let feeder = (stage == Stage::Pixel).then(|| feeder_hlsl(&r.inputs, options.grid, options.feed, options.seed));
        let vertex_inputs = if stage == Stage::Vertex { r.inputs.iter().filter(|e| e.system == 0).cloned().collect() } else { Vec::new() };
        Side { reflection: r, feeder, vertex_inputs }
    };
    notes.sort();
    notes.dedup();
    Ok(Plan { stage, root, targets, sides: [side(a), side(b)], declarations, notes, options, heap })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(semantic: &str, index: u32, register: u32, mask: u8, component: Component) -> Element {
        Element { semantic: semantic.into(), index, register, mask, system: 0, component }
    }

    fn refl(stage: Stage, bindings: Vec<Binding>, inputs: Vec<Element>, outputs: Vec<Element>) -> Reflection {
        Reflection { stage, model: (6, 0), threads: [1, 1, 1], bindings, inputs, outputs, resource_heap: false, sampler_heap: false, heap: vec![] }
    }

    fn bind(class: Class, space: u32, lower: u32, count: u32, shape: Shape) -> Binding {
        Binding { class, space, lower, count, shape, name: String::new() }
    }

    #[test]
    fn a_synthesized_root_holds_the_union_in_one_table_and_static_samplers() {
        let a = refl(Stage::Compute, vec![bind(Class::Cbv, 0, 0, 1, Shape::ConstantBuffer), bind(Class::Uav, 0, 0, 1, Shape::RawBuffer), bind(Class::Sampler, 0, 1, 1, Shape::Sampler { comparison: true })], vec![], vec![]);
        let b = refl(Stage::Compute, vec![bind(Class::Srv, 1, 2, 1, Shape::Texture2D(Ret::Float)), bind(Class::Uav, 0, 0, 1, Shape::RawBuffer)], vec![], vec![]);
        let plan = build(a, b, None, None, Options::default()).unwrap();
        assert_eq!(plan.declarations.len(), 3, "{:?}", plan.declarations);
        let root = &plan.root;
        assert_eq!(root.samplers, vec![Sampler { register: 1, space: 0, comparison: true }]);
        assert_eq!(root.params.len(), 1);
        let Param::Table(ranges) = &root.params[0] else { panic!("{:?}", root.params) };
        assert_eq!(ranges.len(), 3);
        assert_eq!((ranges[0].class, ranges[0].slots.clone()), (Class::Cbv, vec![Res::Cbv { bytes: BUFFER_BYTES }]));
        assert_eq!(ranges[1].class, Class::Srv);
        assert!(matches!(ranges[1].slots[0], Res::Texture { dim: Dim::D2, mips: 7, format: Format::Rgba16F, uav: false, .. }));
        assert_eq!(ranges[2].slots[0], Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Raw, uav: true });
    }

    #[test]
    fn a_given_root_gets_resources_and_null_descriptors() {
        let a = refl(Stage::Compute, vec![bind(Class::Srv, 0, 5, 1, Shape::StructuredBuffer { stride: 12 })], vec![], vec![]);
        let root = Root {
            flags: 0,
            params: vec![
                Param::Table(vec![Range { class: Class::Srv, base: 4, space: 0, declared: 3, offset: None, slots: vec![] }]),
                Param::Root { class: Class::Uav, register: 0, space: 0, res: Res::Null },
                Param::Constants { register: 0, space: 0, count: 4 },
            ],
            samplers: vec![],
            blob: Some(vec![1, 2, 3]),
        };
        let plan = build(a.clone(), a, None, Some(root), Options::default()).unwrap();
        let Param::Table(r) = &plan.root.params[0] else { panic!() };
        assert_eq!(r[0].slots, vec![Res::Null, Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Structured(12), uav: false }, Res::Null]);
        assert_eq!(plan.root.params[1], Param::Root { class: Class::Uav, register: 0, space: 0, res: Res::Buffer { bytes: BUFFER_BYTES, view: BufView::Raw, uav: true } });
    }

    #[test]
    fn each_side_of_a_pixel_pair_gets_a_feeder_for_its_own_inputs() {
        let one = refl(Stage::Pixel, vec![], vec![el("TEXCOORD", 0, 1, 0xF, Component::Float)], vec![Element { semantic: "SV_Target".into(), index: 0, register: 0, mask: 0xF, system: SV_TARGET, component: Component::Float }]);
        let four = refl(
            Stage::Pixel,
            vec![],
            vec![el("TEXCOORD", 0, 1, 0xF, Component::Float), el("TEXCOORD", 1, 2, 0x3, Component::Float), el("TEXCOORD", 2, 2, 0xC, Component::Float), el("BLENDINDICES", 0, 3, 0x1, Component::Uint)],
            one.outputs.clone(),
        );
        let plan = build(one, four, None, None, Options::default()).unwrap();
        assert!(plan.declarations.iter().any(|d| d.starts_with("input signature")), "{:?}", plan.declarations);
        let fa = plan.sides[0].feeder.as_ref().unwrap();
        let fb = plan.sides[1].feeder.as_ref().unwrap();
        assert!(fa.contains("float4 v0 : TEXCOORD0;") && !fa.contains("TEXCOORD1"));
        assert!(fb.contains("float2 v1 : TEXCOORD1;") && fb.contains("float2 v2 : TEXCOORD2;") && fb.contains("nointerpolation uint v3 : BLENDINDICES0;"));
        // TEXCOORD0 gets the same expression on both sides.
        let line = |s: &str| s.lines().find(|l| l.contains("o.v0 =")).unwrap().to_string();
        assert_eq!(line(fa), line(fb));
        // TEXCOORD2 starts at z: its values come from components 2 and 3.
        assert!(fb.lines().find(|l| l.contains("o.v2 =")).unwrap().contains("g * 4u + 2u"));
        assert_eq!(plan.targets, vec![Target { index: 0, format: Format::Rgba32F }]);
        assert_eq!(plan.root.flags, FLAG_INPUT_LAYOUT);
    }

    #[test]
    fn feeding_by_location_ignores_the_semantic() {
        let a = el("TEXCOORD", 0, 3, 0xF, Component::Float);
        let b = el("COLOR", 1, 3, 0xF, Component::Float);
        assert_eq!(feed_key(&a, Feed::Location, 9), feed_key(&b, Feed::Location, 9));
        assert_ne!(feed_key(&a, Feed::Semantic, 9), feed_key(&b, Feed::Semantic, 9));
        assert_eq!(vertex_value(&a, Feed::Location, 9, 5, 1), vertex_value(&b, Feed::Location, 9, 5, 1));
    }

    #[test]
    fn the_feeder_pads_unused_rows_and_places_the_position_like_the_pixel_shader() {
        // Inputs at registers 4 and 5, no SV_Position: rows 0 to 3 are padding, the position comes last.
        let inputs = vec![el("POSITION", 1, 4, 0xF, Component::Float), el("POSITION", 2, 5, 0xC, Component::Float)];
        let src = feeder_hlsl(&inputs, 4, Feed::Semantic, 1);
        let fields: Vec<&str> = src.lines().skip_while(|l| !l.starts_with("struct VOut")).skip(1).take_while(|l| *l != "};").map(str::trim).collect();
        assert_eq!(
            fields,
            [
                "float4 pad0 : DECANT_AB_PAD0;",
                "float4 pad1 : DECANT_AB_PAD1;",
                "float4 pad2 : DECANT_AB_PAD2;",
                "float4 pad3 : DECANT_AB_PAD3;",
                "float4 v0 : POSITION1;",
                "float2 pad4 : DECANT_AB_PAD4;",
                "float2 v1 : POSITION2;",
                "float4 pos : SV_Position;",
            ]
        );
        // With SV_Position at register 0 the position comes first and nothing is padded.
        let mut with_pos = vec![Element { semantic: "SV_Position".into(), index: 0, register: 0, mask: 0xF, system: SV_POSITION, component: Component::Float }];
        with_pos.push(el("TEXCOORD", 0, 1, 0x3, Component::Float));
        let src = feeder_hlsl(&with_pos, 4, Feed::Semantic, 1);
        assert!(src.contains("struct VOut {
    float4 pos : SV_Position;
    float2 v0 : TEXCOORD0;
};"), "{src}");
    }
}
