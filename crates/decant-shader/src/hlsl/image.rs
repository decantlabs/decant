//! Image printing: the HLSL texture types and the statements of image instructions.
//!
//! The address operands arrive in machine order, one 32 bit pattern each (see `lift::mimg`).
//! [`Layout`] finds each group in them. Cube images arrive as face coordinates: the direction
//! `TextureCube` wants is rebuilt from the face and the two coordinates on it
//! (GFX10 image instruction semantics).

use super::{Printer, class_of, mirror_word, nonuniform_index, refuse};
use crate::api::{Placement, RegisterClass};
use crate::ir::{
    Desc, Expr, Guard, ImageAtomic, ImageKind, ImageOp, Origin, Resource, ResourceUse, SampleLod,
    Var,
};
use crate::refusal::Refusal;

/// The `PSV0` kind of a cube array view.
const KIND_CUBE_ARRAY: u32 = 9;

/// The HLSL type of dimension code `dim`, a cube array when the declared view is one.
fn image_type(dim: u8, declared_kind: u32) -> Option<&'static str> {
    Some(match dim {
        0 => "Texture1D",
        1 => "Texture2D",
        2 => "Texture3D",
        3 if declared_kind == KIND_CUBE_ARRAY => "TextureCubeArray",
        3 => "TextureCube",
        4 => "Texture1DArray",
        5 => "Texture2DArray",
        6 => "Texture2DMS",
        7 => "Texture2DMSArray",
        _ => return None,
    })
}

fn declared_kind(r: &Resource) -> u32 {
    r.declared.map_or(0, |d| d.psv_kind)
}

/// The HLSL type of an image resource.
pub(super) fn resource_type(r: &Resource, class: RegisterClass) -> Option<String> {
    let ResourceUse::Image { dim, written, .. } = r.usage else {
        return None;
    };
    let rw = written || class == RegisterClass::Uav;
    // Writable views have no cube or multisample form.
    if rw && matches!(dim, 3 | 6 | 7) {
        return None;
    }
    Some(format!(
        "{}{}<float4>",
        if rw { "RW" } else { "" },
        image_type(dim, declared_kind(r))?
    ))
}

/// The registers an image instruction writes.
pub(super) fn written(op: &ImageOp) -> impl Iterator<Item = Var> + '_ {
    op.data_writes().map(Var::V)
}

/// Components of the texture coordinates without the slice: 1D 1, 2D and cube 2, 3D 3.
fn spatial(dim: u8) -> usize {
    match dim {
        0 | 4 => 1,
        2 => 3,
        _ => 2,
    }
}

/// Components of the coordinate operands, slice, face and fragment index included.
fn coordinates(dim: u8) -> usize {
    match dim {
        0 => 1,
        1 | 4 => 2,
        7 => 4,
        _ => 3,
    }
}

/// Where each group of operands sits in [`ImageOp::address`].
struct Layout {
    offset: Option<usize>,
    bias: Option<usize>,
    compare: Option<usize>,
    /// First gradient and the components per direction.
    grads: Option<(usize, usize)>,
    coords: usize,
    /// The level, clamp or mip operand after the coordinates.
    last: Option<usize>,
    len: usize,
}

impl Layout {
    fn of(op: &ImageOp) -> Layout {
        let mut at = 0;
        let mut take = |n: usize| {
            let here = at;
            at += n;
            here
        };
        let (offset, bias, compare, grad, last) = match op.kind {
            ImageKind::Sample {
                lod,
                compare,
                offset,
                clamp,
                ..
            } => (
                offset,
                lod == SampleLod::Bias,
                compare,
                lod == SampleLod::Grad,
                lod == SampleLod::Level || clamp,
            ),
            ImageKind::Fetch { mip } | ImageKind::Store { mip } => {
                (false, false, false, false, mip)
            }
            ImageKind::Info => {
                return Layout {
                    offset: None,
                    bias: None,
                    compare: None,
                    grads: None,
                    coords: 0,
                    last: Some(0),
                    len: 1,
                };
            }
            ImageKind::QueryLod | ImageKind::Atomic { .. } => (false, false, false, false, false),
        };
        let offset = offset.then(|| take(1));
        let bias = bias.then(|| take(1));
        let compare = compare.then(|| take(1));
        let per = spatial(op.dim);
        let grads = grad.then(|| (take(2 * per), per));
        let coords = take(coordinates(op.dim));
        let last = last.then(|| take(1));
        Layout {
            offset,
            bias,
            compare,
            grads,
            coords,
            last,
            len: at,
        }
    }
}

/// Is the sampler built from constants with `FORCE_UNNORMALIZED` (word 0 bit 15) set?
fn unnormalized_sampler(r: &Resource) -> bool {
    match &r.origin {
        Origin::Immediate { words } => words[0] & 0x8000 != 0,
        Origin::SamplerChoices { choices, .. } => choices[0].0[0] & 0x8000 != 0,
        _ => false,
    }
}

/// The direction on face `{t}_f & 7` whose face coordinates (the cube S and T of
/// `v_cubesc_f32` and `v_cubetc_f32`, divided by the major axis) are `a` and `b` and whose
/// major axis component is `m`. Linear in all three, so it also turns face derivatives (with
/// `m` 0) into direction derivatives.
fn cube_direction(t: &str, a: &str, b: &str, m: &str) -> String {
    let f = format!("({t}_f & 7u)");
    format!(
        "({f} == 0u ? float3({m}, -({b}), -({a})) \
         : {f} == 1u ? float3(-({m}), -({b}), {a}) \
         : {f} == 2u ? float3({a}, {m}, {b}) \
         : {f} == 3u ? float3({a}, -({m}), -({b})) \
         : {f} == 4u ? float3({a}, -({b}), {m}) \
         : float3(-({a}), -({b}), -({m})))"
    )
}

fn vector(kind: &str, parts: &[String]) -> String {
    if parts.len() == 1 {
        parts[0].clone()
    } else {
        format!("{kind}{}({})", parts.len(), parts.join(", "))
    }
}

/// The value read as a result component: a float or an integer of the sample variable.
struct Read {
    var: String,
    integer: bool,
}

impl Printer<'_> {
    /// The name of the image or sampler of an operand for this statement. A descriptor the
    /// program chose by its heap index is made here, in a line before the statement, with the
    /// type its use needs.
    fn use_desc(&mut self, d: Desc, pc: u32) -> Result<String, Refusal> {
        let Desc::Res(id) = d else {
            unreachable!("the resource pass replaced every register descriptor")
        };
        let func = self.func;
        let r = &func.resources[id.0 as usize];
        let ty = r
            .placement
            .as_ref()
            .and_then(|p| super::resource_type(func, r, class_of(p)))
            .ok_or_else(|| refuse("image-kind", pc))?;
        if let Origin::SamplerChoices { first, choices } = &r.origin {
            self.unique += 1;
            let local = format!("decant_sampler_{}", self.unique);
            let index = format!("{local}_index");
            self.line(&format!("uint {index} = {}u;", choices[0].1));
            for (words, slot) in &choices[1..] {
                let matches = words
                    .iter()
                    .enumerate()
                    .map(|(i, word)| format!("s{} == {word}u", first + i as u8))
                    .collect::<Vec<_>>()
                    .join(" && ");
                self.line(&format!("if ({matches}) {index} = {slot}u;"));
            }
            self.line(&format!(
                "{ty} {local} = SamplerDescriptorHeap[NonUniformResourceIndex({index})];"
            ));
            return Ok(local);
        }
        Ok(self.use_resource(id, &ty))
    }

    pub(super) fn image(&mut self, op: &ImageOp, guard: Guard) -> Result<(), Refusal> {
        let res = self.res(op.image);
        let kind = declared_kind(res);
        image_type(op.dim, kind).ok_or_else(|| refuse("image-dim", op.pc))?;
        let layout = Layout::of(op);
        if op.address.len() < layout.len {
            return Err(refuse("image-address", op.pc));
        }
        let class = res.placement.as_ref().map(class_of);
        let rw = class == Some(RegisterClass::Uav);
        // An atomic makes its own integer view of the slot.
        let image = if matches!(op.kind, ImageKind::Atomic { .. }) {
            String::new()
        } else {
            self.use_desc(op.image, op.pc)?
        };
        self.unique += 1;
        let t = format!("decant_t{}", self.unique);
        let read = match op.kind {
            ImageKind::Sample { .. } | ImageKind::QueryLod => {
                if rw {
                    return Err(refuse("image-sample-uav", op.pc));
                }
                let desc = op.sampler.ok_or_else(|| refuse("image-kind", op.pc))?;
                let unnormalized = op.unnormalized || unnormalized_sampler(self.res(desc));
                let sampler = self.use_desc(desc, op.pc)?;
                let cube_array = kind == KIND_CUBE_ARRAY;
                let value =
                    self.sample(op, &layout, &image, &sampler, unnormalized, cube_array, &t)?;
                // Outside the lane test: an implicit level of detail needs every lane of the quad.
                self.line(&format!("float4 {t} = {value};"));
                Read {
                    var: t,
                    integer: false,
                }
            }
            ImageKind::Fetch { .. } => {
                let value = self.fetch(op, &layout, &image, rw)?;
                self.line(&format!("float4 {t} = {value};"));
                Read {
                    var: t,
                    integer: false,
                }
            }
            ImageKind::Info => {
                self.info(op, &image, rw, kind == KIND_CUBE_ARRAY, &t)?;
                Read {
                    var: t,
                    integer: true,
                }
            }
            ImageKind::Store { .. } => return self.store(op, &layout, &image, rw, guard),
            ImageKind::Atomic { op: what, returns } => {
                return self.atomic(op, &layout, what, returns, guard, &t);
            }
        };
        self.results(op, &read, guard);
        Ok(())
    }

    /// Writes the result registers: the components `dmask` selects (all four of a gather),
    /// two to a register with D16, then the residency status with TFE.
    fn results(&mut self, op: &ImageOp, read: &Read, guard: Guard) {
        let gather = matches!(op.kind, ImageKind::Sample { gather: true, .. });
        let comps: Vec<&str> = ["x", "y", "z", "w"]
            .into_iter()
            .enumerate()
            .filter(|(c, _)| gather || (op.dmask >> c) & 1 != 0)
            .map(|(_, n)| n)
            .collect();
        let t = &read.var;
        let word = |c: &str| {
            if read.integer {
                format!("{t}.{c}")
            } else {
                format!("asuint({t}.{c})")
            }
        };
        let half = |c: &str| {
            if read.integer {
                format!("({t}.{c} & 0xFFFFu)")
            } else {
                format!("f32tof16({t}.{c})")
            }
        };
        let mut values: Vec<String> = if op.d16 {
            comps
                .chunks(2)
                .map(|pair| match pair {
                    [lo, hi] => format!("({} | ({} << 16))", half(lo), half(hi)),
                    [lo] => half(lo),
                    _ => unreachable!("chunks of two"),
                })
                .collect()
        } else {
            comps.iter().map(|c| word(c)).collect()
        };
        if op.tfe && !values.is_empty() {
            // Every texel is resident: the status is zero.
            values.push("0u".to_owned());
        }
        let lines = values
            .into_iter()
            .enumerate()
            .map(|(k, v)| format!("v{} = {v};", op.data as usize + k))
            .collect();
        self.guarded(guard, lines);
    }

    fn float(&self, e: &Expr) -> String {
        format!("asfloat({})", self.e(e))
    }

    fn int(&self, e: &Expr) -> String {
        format!("int({})", self.e(e))
    }

    /// The sample location: the coordinates as floats, a direction for a cube, with texel
    /// coordinates turned into normalised ones when `unnormalized`. Second: the location
    /// without the array slice, which the level of detail queries take.
    #[allow(clippy::too_many_arguments)]
    fn location(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        image: &str,
        unnormalized: bool,
        cube_array: bool,
        t: &str,
    ) -> Result<(String, String), Refusal> {
        let at = layout.coords;
        let mut parts: Vec<String> = (0..coordinates(op.dim))
            .map(|k| self.float(&op.address[at + k]))
            .collect();
        if unnormalized {
            // Only 1D and 2D images can be read with texel coordinates.
            let dims: &[&str] = match op.dim {
                0 => &["w"],
                1 => &["w", "h"],
                4 => &["w", "e"],
                5 => &["w", "h", "e"],
                _ => return Err(refuse("image-unnormalized-dim", op.pc)),
            };
            let names: Vec<String> = dims.iter().map(|d| format!("{t}_{d}")).collect();
            self.line(&format!("float {};", names.join(", ")));
            self.line(&format!("{image}.GetDimensions({});", names.join(", ")));
            for (k, part) in parts.iter_mut().enumerate().take(spatial(op.dim)) {
                *part = format!("({part} / {})", names[k]);
            }
        }
        if op.dim == 3 {
            // s and t are the face coordinates plus 1.5, the third operand is 8 * layer + face.
            let (s, tc, face) = (&parts[0], &parts[1], &parts[2]);
            self.line(&format!("float {t}_a = 2.0 * ({s} - 1.5);"));
            self.line(&format!("float {t}_b = 2.0 * ({tc} - 1.5);"));
            self.line(&format!("uint {t}_f = uint(max({face}, 0.0));"));
            let dir = cube_direction(t, &format!("{t}_a"), &format!("{t}_b"), "1.0");
            self.line(&format!("float3 {t}_d = {dir};"));
            let dir = format!("{t}_d");
            if cube_array {
                return Ok((format!("float4({dir}, float({t}_f >> 3))"), dir));
            }
            return Ok((dir.clone(), dir));
        }
        Ok((
            vector("float", &parts),
            vector("float", &parts[..spatial(op.dim)]),
        ))
    }

    /// Components of a texel offset: 1D 1, 3D 3, else 2.
    fn offset_comps(dim: u8) -> usize {
        match dim {
            0 | 4 => 1,
            2 => 3,
            _ => 2,
        }
    }

    /// The texel offset of a sample: the 6 bit fields of the offset operand.
    fn offset(&self, op: &ImageOp, layout: &Layout) -> Option<String> {
        let at = layout.offset?;
        let o = self.e(&op.address[at]);
        let parts: Vec<String> = (0..Self::offset_comps(op.dim))
            .map(|k| format!("(int({o} << {}u) >> 26)", 26 - 8 * k))
            .collect();
        Some(vector("int", &parts))
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        image: &str,
        sampler: &str,
        unnormalized: bool,
        cube_array: bool,
        t: &str,
    ) -> Result<String, Refusal> {
        let (loc, flat) = self.location(op, layout, image, unnormalized, cube_array, t)?;
        let (lod, gather, clamp) = match op.kind {
            ImageKind::Sample {
                lod, gather, clamp, ..
            } => (lod, gather, clamp),
            ImageKind::QueryLod => {
                return Ok(format!(
                    "float4({image}.CalculateLevelOfDetail({sampler}, {flat}), \
                     {image}.CalculateLevelOfDetailUnclamped({sampler}, {flat}), 0.0, 0.0)"
                ));
            }
            _ => unreachable!("only samples come here"),
        };
        let offset = self.offset(op, layout);
        if offset.is_some() && op.dim == 3 {
            return Err(refuse("image-cube-offset", op.pc));
        }
        // A clamp operand goes after an offset, which is then zero when the program has none.
        let mut tail = String::new();
        if clamp && op.dim == 3 {
            // A cube takes no offset: the clamp follows the location directly.
            let c = self.float(&op.address[layout.last.expect("a clamp operand")]);
            tail = format!(", {c}");
        } else if clamp {
            let zero = vector("int", &vec!["0".to_owned(); Self::offset_comps(op.dim)]);
            let c = self.float(&op.address[layout.last.expect("a clamp operand")]);
            tail = format!(", {}, {c}", offset.clone().unwrap_or(zero));
        } else if let Some(o) = &offset {
            tail = format!(", {o}");
        }
        let cmp = layout.compare.map(|k| self.float(&op.address[k]));
        if gather {
            if op.dmask.count_ones() != 1 {
                return Err(refuse("image-gather-dmask", op.pc));
            }
            let channel = ["Red", "Green", "Blue", "Alpha"][op.dmask.trailing_zeros() as usize];
            if lod == SampleLod::Level && cmp.is_none() && !clamp {
                return self.gather_level(op, layout, image, &loc, t);
            }
            if lod != SampleLod::Zero || clamp {
                return Err(refuse("image-gather-lod", op.pc));
            }
            return Ok(match cmp {
                Some(c) => format!("{image}.GatherCmp{channel}({sampler}, {loc}, {c}{tail})"),
                None => format!("{image}.Gather{channel}({sampler}, {loc}{tail})"),
            });
        }
        let compared = cmp.is_some();
        let call = match (lod, cmp) {
            (SampleLod::Implicit, None) => format!("{image}.Sample({sampler}, {loc}{tail})"),
            (SampleLod::Bias, None) => {
                let b = self.float(&op.address[layout.bias.expect("a bias operand")]);
                format!("{image}.SampleBias({sampler}, {loc}, {b}{tail})")
            }
            (SampleLod::Level, None) => {
                let l = self.float(&op.address[layout.last.expect("a level operand")]);
                format!("{image}.SampleLevel({sampler}, {loc}, {l}{tail})")
            }
            (SampleLod::Zero, None) => format!("{image}.SampleLevel({sampler}, {loc}, 0.0{tail})"),
            (SampleLod::Grad, None) => {
                let (first, per) = layout.grads.expect("gradient operands");
                let dx: Vec<String> = (0..per)
                    .map(|k| self.float(&op.address[first + k]))
                    .collect();
                let dy: Vec<String> = (0..per)
                    .map(|k| self.float(&op.address[first + per + k]))
                    .collect();
                let (dx, dy) = if op.dim == 3 {
                    // Face derivatives are in units of the face coordinate plus 1.5, half of the
                    // coordinate divided by the major axis. The major axis does not move.
                    let d = |g: &[String]| {
                        cube_direction(
                            t,
                            &format!("2.0 * {}", g[0]),
                            &format!("2.0 * {}", g[1]),
                            "0.0",
                        )
                    };
                    (d(&dx), d(&dy))
                } else {
                    (vector("float", &dx), vector("float", &dy))
                };
                format!("{image}.SampleGrad({sampler}, {loc}, {dx}, {dy}{tail})")
            }
            (SampleLod::Implicit, Some(c)) => {
                format!("{image}.SampleCmp({sampler}, {loc}, {c}{tail})")
            }
            (SampleLod::Zero, Some(c)) if !clamp => {
                format!("{image}.SampleCmpLevelZero({sampler}, {loc}, {c}{tail})")
            }
            (_, Some(_)) => return Err(refuse("image-compare-lod", op.pc)),
        };
        // A depth compare gives one value.
        if compared {
            return Ok(format!("{call}.xxxx"));
        }
        Ok(call)
    }

    /// A gather at an explicit level, which HLSL lacks: the four texels of the footprint read
    /// one by one at the nearest mip, in the gather order (x at the lower left, then
    /// counterclockwise, w at the upper left). Texel coordinates are clamped to the edge
    /// (Inference: the sampler's address mode is not known here).
    fn gather_level(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        image: &str,
        loc: &str,
        t: &str,
    ) -> Result<String, Refusal> {
        let array = match op.dim {
            1 => false,
            5 => true,
            _ => return Err(refuse("image-gather-lod", op.pc)),
        };
        let channel = ["x", "y", "z", "w"][op.dmask.trailing_zeros() as usize];
        let level = self.float(&op.address[layout.last.expect("a level operand")]);
        self.line(&format!(
            "float4 {t}_l = float4({loc}{});",
            if array { ", 0.0" } else { ", 0.0, 0.0" }
        ));
        self.line(&format!("uint {t}_w, {t}_h, {t}_e, {t}_n;"));
        let size = if array {
            format!("{t}_w, {t}_h, {t}_e, {t}_n")
        } else {
            format!("{t}_w, {t}_h, {t}_n")
        };
        self.line(&format!("{image}.GetDimensions(0u, {size});"));
        self.line(&format!(
            "uint {t}_m = min(uint(max(round({level}), 0.0)), {t}_n - 1u);"
        ));
        self.line(&format!("{image}.GetDimensions({t}_m, {size});"));
        let offset = self
            .offset(op, layout)
            .unwrap_or_else(|| "int2(0, 0)".to_owned());
        self.line(&format!(
            "int2 {t}_i = int2(floor({t}_l.xy * float2({t}_w, {t}_h) - 0.5)) + {offset};"
        ));
        self.line(&format!("int2 {t}_x = int2({t}_w, {t}_h) - 1;"));
        let texel = |dx: i32, dy: i32| {
            let at = format!("clamp({t}_i + int2({dx}, {dy}), int2(0, 0), {t}_x)");
            if array {
                format!("{image}.Load(int4({at}, int(round({t}_l.z)), int({t}_m))).{channel}")
            } else {
                format!("{image}.Load(int3({at}, int({t}_m))).{channel}")
            }
        };
        Ok(format!(
            "float4({}, {}, {}, {})",
            texel(0, 1),
            texel(1, 1),
            texel(1, 0),
            texel(0, 0)
        ))
    }

    /// The integer coordinates of a load, store or atomic, slice and fragment index included.
    fn texel(&self, op: &ImageOp, layout: &Layout) -> Vec<String> {
        (0..coordinates(op.dim))
            .map(|k| self.int(&op.address[layout.coords + k]))
            .collect()
    }

    /// A writable view has one mip. A mip operand other than 0 is refused.
    fn single_mip(&self, op: &ImageOp, layout: &Layout) -> Result<(), Refusal> {
        match layout.last.map(|k| &op.address[k]) {
            None | Some(Expr::U32(0)) => Ok(()),
            Some(_) => Err(refuse("image-uav-mip", op.pc)),
        }
    }

    fn rw_index(&self, op: &ImageOp, layout: &Layout) -> Result<String, Refusal> {
        if matches!(op.dim, 3 | 6 | 7) {
            return Err(refuse("image-uav-dim", op.pc));
        }
        self.single_mip(op, layout)?;
        let at: Vec<String> = self
            .texel(op, layout)
            .iter()
            .map(|c| format!("uint({c})"))
            .collect();
        Ok(vector("uint", &at))
    }

    fn fetch(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        image: &str,
        rw: bool,
    ) -> Result<String, Refusal> {
        if op.dim == 3 {
            return Err(refuse("image-cube-load", op.pc));
        }
        if rw {
            return Ok(format!("{image}[{}]", self.rw_index(op, layout)?));
        }
        let mut coords = self.texel(op, layout);
        if matches!(op.dim, 6 | 7) {
            let sample = coords.pop().expect("a fragment index");
            return Ok(format!(
                "{image}.Load({}, {sample})",
                vector("int", &coords)
            ));
        }
        coords.push(match layout.last {
            Some(k) => self.int(&op.address[k]),
            None => "0".to_owned(),
        });
        Ok(format!("{image}.Load({})", vector("int", &coords)))
    }

    fn store(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        image: &str,
        rw: bool,
        guard: Guard,
    ) -> Result<(), Refusal> {
        if !rw {
            return Err(refuse("image-store-srv", op.pc));
        }
        let at = self.rw_index(op, layout)?;
        // The stored components in x, y, z, w order, the others zero.
        let mut regs = op.data_reads();
        let mut halves = Vec::new();
        let mut comps = Vec::new();
        for c in 0..4 {
            if (op.dmask >> c) & 1 == 0 {
                comps.push("0.0".to_owned());
                continue;
            }
            if op.d16 {
                if halves.is_empty() {
                    let r = regs.next().ok_or_else(|| refuse("image-data", op.pc))?;
                    halves = vec![format!("f16tof32(v{r} >> 16)"), format!("f16tof32(v{r})")];
                }
                comps.push(halves.pop().expect("two halves per register"));
            } else {
                let r = regs.next().ok_or_else(|| refuse("image-data", op.pc))?;
                comps.push(format!("asfloat(v{r})"));
            }
        }
        let line = format!("{image}[{at}] = float4({});", comps.join(", "));
        self.guarded(guard, vec![line]);
        Ok(())
    }

    /// An atomic needs an integer view: a second variable on the same heap slot, typed `uint`
    /// (`int` for the signed minimum and maximum).
    fn atomic(
        &mut self,
        op: &ImageOp,
        layout: &Layout,
        what: ImageAtomic,
        returns: bool,
        guard: Guard,
        t: &str,
    ) -> Result<(), Refusal> {
        // The slot by its host heap index: a table slot through the mirror, or the register
        // that holds the index of a descriptor the program chose.
        let heap = match (op.image, self.res(op.image).placement) {
            (_, Some(Placement::HeapAtMirror { dword, slot, .. })) => {
                let idx = nonuniform_index(&format!("{} + {slot}u", mirror_word(dword)));
                format!("ResourceDescriptorHeap[{idx}]")
            }
            (Desc::Res(id), Some(Placement::HeapAtConsoleIndex { .. })) => {
                let index = self
                    .host_slot(id)
                    .ok_or_else(|| refuse("image-atomic-placement", op.pc))?;
                let idx = nonuniform_index(&index);
                format!("ResourceDescriptorHeap[{idx}]")
            }
            _ => return Err(refuse("image-atomic-placement", op.pc)),
        };
        let ty = match op.dim {
            0 => "RWTexture1D",
            1 => "RWTexture2D",
            2 => "RWTexture3D",
            4 => "RWTexture1DArray",
            5 => "RWTexture2DArray",
            _ => return Err(refuse("image-uav-dim", op.pc)),
        };
        let signed = matches!(what, ImageAtomic::SMin | ImageAtomic::SMax);
        // The mask names the operand dwords: two per operand for the 64 bit forms.
        let operands = if what == ImageAtomic::CmpSwap { 2 } else { 1 };
        let wide = op.dmask.count_ones() == 2 * operands;
        let elem = match (wide, signed) {
            (false, false) => "uint",
            (false, true) => "int",
            (true, false) => "uint64_t",
            (true, true) => "int64_t",
        };
        let at = self.rw_index(op, layout)?;
        let d = op.data;
        // Operand `k` of the data registers, in the element type.
        let operand = |k: u8| {
            if wide {
                let r = d + 2 * k;
                format!("{elem}((uint64_t(v{}) << 32) | uint64_t(v{r}))", r + 1)
            } else {
                format!("{elem}(v{})", d + k)
            }
        };
        let value = operand(0);
        let dest = format!("{t}_u[{at}]");
        let call = match what {
            ImageAtomic::Swap => format!("InterlockedExchange({dest}, {value}, {t}_o)"),
            ImageAtomic::CmpSwap => format!(
                "InterlockedCompareExchange({dest}, {}, {value}, {t}_o)",
                operand(1)
            ),
            ImageAtomic::Add => format!("InterlockedAdd({dest}, {value}, {t}_o)"),
            ImageAtomic::Sub => format!("InterlockedAdd({dest}, 0 - {value}, {t}_o)"),
            ImageAtomic::SMin | ImageAtomic::UMin => {
                format!("InterlockedMin({dest}, {value}, {t}_o)")
            }
            ImageAtomic::SMax | ImageAtomic::UMax => {
                format!("InterlockedMax({dest}, {value}, {t}_o)")
            }
            ImageAtomic::And => format!("InterlockedAnd({dest}, {value}, {t}_o)"),
            ImageAtomic::Or => format!("InterlockedOr({dest}, {value}, {t}_o)"),
            ImageAtomic::Xor => format!("InterlockedXor({dest}, {value}, {t}_o)"),
            ImageAtomic::Inc | ImageAtomic::Dec => {
                return Err(refuse("image-atomic-wrap", op.pc));
            }
        };
        self.line(&format!("{ty}<{elem}> {t}_u = {heap};"));
        self.line(&format!("{elem} {t}_o = 0;"));
        let mut lines = vec![format!("{call};")];
        if returns {
            lines.push(format!("v{d} = uint({t}_o);"));
            if wide {
                lines.push(format!("v{} = uint(uint64_t({t}_o) >> 32);", d + 1));
            }
        }
        self.guarded(guard, lines);
        Ok(())
    }

    /// `image_get_resinfo`: width, height, depth or layers, mip count, all integers.
    fn info(
        &mut self,
        op: &ImageOp,
        image: &str,
        rw: bool,
        cube_array: bool,
        t: &str,
    ) -> Result<(), Refusal> {
        let mip = self.e(&op.address[0]);
        self.line(&format!("uint4 {t} = uint4(0u, 0u, 0u, 1u);"));
        let fields: &[&str] = match (op.dim, rw) {
            (0, false) => &["x", "w"],
            (1 | 3, false) if !cube_array => &["x", "y", "w"],
            (2 | 3 | 5, false) => &["x", "y", "z", "w"],
            (4, false) => &["x", "z", "w"],
            (0, true) => &["x"],
            (1, true) => &["x", "y"],
            (2 | 5, true) => &["x", "y", "z"],
            (4, true) => &["x", "z"],
            (6, false) => &["x", "y", "w"],
            (7, false) => &["x", "y", "z", "w"],
            _ => return Err(refuse("image-info-dim", op.pc)),
        };
        let outs: Vec<String> = fields.iter().map(|f| format!("{t}.{f}")).collect();
        let call = if rw || matches!(op.dim, 6 | 7) {
            format!("{image}.GetDimensions({});", outs.join(", "))
        } else {
            format!("{image}.GetDimensions({mip}, {});", outs.join(", "))
        };
        self.line(&call);
        if op.dim == 3 {
            // A cube counts its faces as layers.
            if cube_array {
                self.line(&format!("{t}.z *= 6u;"));
            } else {
                self.line(&format!("{t}.z = 6u;"));
            }
        }
        if matches!(op.dim, 6 | 7) {
            // The last field of a multisample image is the sample count: the machine gives its log2.
            self.line(&format!("{t}.w = firstbithigh({t}.w);"));
        }
        Ok(())
    }
}
