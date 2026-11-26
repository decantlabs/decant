//! Buffers as HLSL text: the types of buffer resources, the dwords of a constant buffer, and the
//! vector buffer statements (`buffer_load_*`, `buffer_store_*`, the typed forms, atomics).
//!
//! The machine computes the address of an access itself, `index * stride + offset`, with the
//! record stride of the descriptor. The HLSL form depends on the view the title made, which the
//! container declares (`Resource::declared`, its `PSV0` type):
//!
//! * a constant buffer is an array of 16 byte rows, so an index counts rows,
//! * a raw view is a `ByteAddressBuffer` read at the byte offset,
//! * a structured view is turned into a raw view by the layer (the view shadow of
//!   `MirrorAbi`): it is read as a `ByteAddressBuffer` too, an index is multiplied by the
//!   record stride of the shadow row of its host heap slot and checked against the record
//!   count of that row,
//! * a typed view uses the same raw addressing. Format loads and stores convert the
//!   dwords using the DXGI format in the shadow row.
//!
//! A raw view on PC starts on a 16 byte boundary, a console view on 4. So the layer starts the
//! view of a heap slot on the 16 bytes at or below the title's first byte, and dword 3 of the
//! shadow row is the byte offset of the title's first byte inside the view (0, 4, 8 or 12).
//! Every byte address of a buffer in a heap slot gets that offset added when the layer keeps a
//! shadow ([`add_view_offsets`]).
//!
//! When the container does not declare the descriptor, the instructions decide: a format
//! access means a typed view, an index a structured view, else a raw view.
//!
//! Out of bounds reads give 0 and out of bounds writes are dropped, on the console and in
//! Direct3D alike, so only the record count of a structured or typed view needs a test of its
//! own.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::{Context, Printer};
use crate::api::{Placement, RegisterClass};
use crate::ir::{
    Bin, BufferOp, Desc, Expr, Function, Guard, Helper, Origin, ResId, Resource, ResourceUse, Stmt,
    Ty,
};
use crate::lift::buf::{Atomic, Conversion, Shape, shape};
use crate::refusal::{Refusal, RefusalKind};

/// The kind of view a buffer descriptor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum View {
    Constant,
    Raw,
    Structured,
    Typed,
}

fn res_index(func: &Function, r: &Resource) -> usize {
    func.resources
        .iter()
        .position(|x| std::ptr::eq(x, r))
        .expect("a resource of this function")
}

fn class(r: &Resource) -> Option<RegisterClass> {
    r.placement.as_ref().map(super::class_of)
}

/// Calls `f` on every buffer operation on resource `id`.
fn ops_on(func: &Function, id: usize, f: &mut impl FnMut(&BufferOp)) {
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && op.buffer == Desc::Res(ResId(id as u32))
        {
            f(op);
        }
    });
}

fn view(func: &Function, id: usize) -> View {
    let r = &func.resources[id];
    if class(r) == Some(RegisterClass::Cbv) {
        return View::Constant;
    }
    if let Some(d) = r.declared {
        match d.psv_type {
            2 => return View::Constant,
            3 | 6 => return View::Typed,
            4 | 7 => return View::Raw,
            5 | 8 | 9 => return View::Structured,
            _ => {}
        }
    }
    let (mut typed, mut indexed) = (false, false);
    ops_on(func, id, &mut |op| {
        typed |= shape(op.op).is_some_and(|s| s.conversion == Conversion::View);
        indexed |= op.index.is_some();
    });
    if typed {
        View::Typed
    } else if indexed {
        View::Structured
    } else {
        View::Raw
    }
}

fn writable(r: &Resource) -> bool {
    matches!(r.usage, ResourceUse::Words { written: true }) || class(r) == Some(RegisterClass::Uav)
}

/// The byte offset of the title's first byte inside the raw view of a host heap slot: dword 3
/// of the shadow row. Its upper bits are zero for a buffer.
static VIEW_OFFSET: Helper = Helper {
    name: "decant_view_offset",
    result: Ty::U32,
    text: "uint decant_view_offset(uint slot) { return decant_view_shadow_load(slot * 16u + 12u) & 15u; }",
    needs: &[&super::VIEW_SHADOW_LOAD],
};

/// Returns the view shadow read helpers (`decant_view_shadow_load2` / `decant_view_shadow_load4`)
/// emitted directly by buffer statements in `func`.
pub(super) fn shadow_helpers(func: &Function) -> Vec<&'static Helper> {
    let (mut load2, mut load4) = (false, false);
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && let Desc::Res(ResId(id)) = op.buffer
            && let Some(sh) = shape(op.op)
        {
            let v = view(func, id as usize);
            if sh.conversion == Conversion::View {
                if v == View::Typed {
                    load4 = true;
                }
            } else if op.index.is_some() && matches!(v, View::Structured | View::Typed) {
                load2 = true;
            }
        }
    });
    let mut out = Vec::new();
    if load2 {
        out.push(&super::VIEW_SHADOW_LOAD2);
    }
    if load4 {
        out.push(&super::VIEW_SHADOW_LOAD4);
    }
    out
}

/// The host heap slot of buffer resource `id` as an expression, when the resource is a raw
/// view in a heap slot: a raw or structured view of the title, or a typed one the layer made
/// raw.
fn raw_slot(func: &Function, id: usize) -> Option<Expr> {
    let r = &func.resources[id];
    if !matches!(r.usage, ResourceUse::Words { .. }) {
        return None;
    }
    if view(func, id) == View::Constant {
        return None;
    }
    match (r.placement, &r.origin) {
        (Some(Placement::HeapAtMirror { dword, slot, .. }), _) => {
            Some(Expr::bin(Bin::Add, Expr::Mirror(dword), Expr::U32(slot)))
        }
        (
            Some(Placement::HeapAtConsoleIndex { .. }),
            Origin::HeapIndex {
                index,
                sampler: false,
            },
        ) => Some(Expr::Var(*index)),
        _ => None,
    }
}

/// Adds the view offset of the shadow row to every byte address of a buffer in a heap slot:
/// the scalar reads and the raw accesses of the buffer statements. A format access of a typed
/// view has no byte offset, its statement adds the view offset itself. Without a shadow the
/// views are where the title put them.
pub(super) fn add_view_offsets(func: &mut Function, cx: &Context) {
    if cx.mirror.and_then(|m| m.view_shadow).is_none() {
        return;
    }
    let slots: Vec<Option<Expr>> = (0..func.resources.len())
        .map(|id| raw_slot(func, id))
        .collect();
    if slots.iter().all(Option::is_none) {
        return;
    }
    let moved = |offset: Expr, slot: &Expr| {
        let view = Expr::Call(&VIEW_OFFSET, vec![slot.clone()]);
        if offset == Expr::U32(0) {
            view
        } else {
            Expr::bin(Bin::Add, offset, view)
        }
    };
    // In place: a rebuilt tree cost an allocation per node.
    fn words(e: &mut Expr, slots: &[Option<Expr>], moved: &impl Fn(Expr, &Expr) -> Expr) {
        match e {
            Expr::BufferWord { res, offset } => {
                words(offset, slots, moved);
                if let Some(slot) = &slots[res.0 as usize] {
                    let inner = std::mem::replace(&mut **offset, Expr::U32(0));
                    **offset = moved(inner, slot);
                }
            }
            Expr::Un(_, a) | Expr::CodeWord(a) => words(a, slots, moved),
            Expr::Bin(_, a, b) => {
                words(a, slots, moved);
                words(b, slots, moved);
            }
            Expr::Select(c, a, b) => {
                words(c, slots, moved);
                words(a, slots, moved);
                words(b, slots, moved);
            }
            Expr::Call(_, args) => args.iter_mut().for_each(|a| words(a, slots, moved)),
            _ => {}
        }
    }
    super::simplify::visit_nodes(&mut func.body, &mut |e| words(e, &slots, &moved));
    func.for_each_block_mut(&mut |stmts| {
        for s in stmts {
            if let Stmt::Buffer(op, _) = s
                && let Desc::Res(ResId(id)) = op.buffer
                && let Some(slot) = &slots[id as usize]
                && shape(op.op).is_none_or(|s| s.conversion != Conversion::View)
            {
                op.offset = moved(op.offset.clone(), slot);
            }
        }
    });
}

/// The HLSL type of a resource read or written as dwords.
pub(super) fn resource_type(func: &Function, r: &Resource, class: RegisterClass) -> Option<String> {
    if class == RegisterClass::Sampler {
        return None;
    }
    let rw = if writable(r) { "RW" } else { "" };
    Some(match view(func, res_index(func, r)) {
        View::Constant => "ConstantBuffer<decant_words>".to_owned(),
        // The layer makes these views raw.
        View::Raw | View::Structured | View::Typed => format!("{rw}ByteAddressBuffer"),
    })
}

fn res_refusal(what: &'static str, detail: String) -> Refusal {
    Refusal::new(RefusalKind::Resource(what), detail)
}

/// Whether `op` is a single-component non-d16 typed view load (`buffer_load_format_x`).
fn is_single_typed_load(op: &BufferOp) -> bool {
    !op.store
        && op.count == 1
        && shape(op.op).is_some_and(|sh| sh.conversion == Conversion::View && !sh.d16)
}

/// Whether the function benefits from the scalar `decant_typed_decode_x` helper: used when a shader
/// has repeated `buffer_load_format_x` loads, avoiding a second helper for single-load shaders.
pub(super) fn uses_typed_decode_x(func: &Function) -> bool {
    let mut count = 0usize;
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && is_single_typed_load(op)
        {
            count += 1;
        }
    });
    count >= 2
}

/// Shadow rows hoisted to the start of `main()` for fixed-slot typed buffer resources used by
/// more than one format operation. Returns the map of `ResId` to local variable name and the
/// HLSL declaration lines to emit at entry.
pub(super) fn hoisted_typed_rows(
    func: &Function,
    cx: &Context,
) -> (BTreeMap<ResId, String>, Vec<String>) {
    let mut rows = BTreeMap::new();
    let mut lines = Vec::new();
    if cx.mirror.and_then(|m| m.view_shadow).is_none() {
        return (rows, lines);
    }
    let mut counts = vec![0usize; func.resources.len()];
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && let Desc::Res(ResId(id)) = op.buffer
            && shape(op.op).is_some_and(|sh| sh.conversion == Conversion::View)
            && view(func, id as usize) == View::Typed
        {
            counts[id as usize] += 1;
        }
    });
    for (id, &count) in counts.iter().enumerate() {
        if count < 2 {
            continue;
        }
        let r = &func.resources[id];
        if let Some(Placement::HeapAtMirror { dword, slot, .. }) = r.placement {
            let var = format!("decant_row_{}", r.name);
            lines.push(format!(
                "uint4 {var} = decant_view_shadow_load4(({} + {slot}u) * 16u);",
                super::mirror_word(dword),
            ));
            rows.insert(ResId(id as u32), var);
        }
    }
    (rows, lines)
}

/// Checks that every buffer resource can be written with the views and the host at hand, and
/// writes the declarations the buffer statements need: the constant buffer row type, the
/// format conversions of the typed `tbuffer` loads and the conversions of a typed view.
pub(super) fn preamble(func: &Function, cx: &Context, out: &mut String) -> Result<(), Refusal> {
    let mut constant = false;
    // Node memory is read by the node intersection helper, which checks its view itself.
    let nodes = crate::resource::node_memories(func);
    for (id, r) in func.resources.iter().enumerate() {
        if !matches!(r.usage, ResourceUse::Words { .. }) || nodes.contains(&ResId(id as u32)) {
            continue;
        }
        if r.declared.is_some_and(|d| d.psv_type == 9) {
            // The layer makes structured views raw, and a raw view carries no counter.
            return Err(res_refusal("buffer-counter", r.name.clone()));
        }
        let v = view(func, id);
        constant |= v == View::Constant;
        if v == View::Constant && class(r) != Some(RegisterClass::Cbv) {
            return Err(res_refusal("buffer-view-mismatch", r.name.clone()));
        }
        let raw = matches!(v, View::Structured | View::Typed);
        if raw && cx.mirror.and_then(|m| m.view_shadow).is_none() {
            let mut indexed = false;
            ops_on(func, id, &mut |op| indexed |= op.index.is_some());
            return Err(res_refusal(
                if indexed {
                    "buffer-structured"
                } else {
                    "buffer-structured-raw"
                },
                r.name.clone(),
            ));
        }
    }
    if constant {
        // A constant buffer as 16 byte registers of four dwords, the largest size a view can have.
        writeln!(out, "struct decant_words {{ uint4 w[4096]; }};").expect("writing to a string");
    }
    let mut formats = BTreeSet::new();
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && shape(op.op).is_some_and(|s| s.conversion == Conversion::Instruction)
        {
            formats.insert((op.format, op.op, op.pc));
        }
    });
    let mut written = BTreeSet::new();
    for (format, op, pc) in formats {
        let f = Format::of(format)
            .ok_or_else(|| Refusal::new(RefusalKind::Form(op, "format"), "").at(pc))?;
        if written.insert(format) {
            out.push_str(&f.decoder(format));
        }
    }
    // The conversions of a typed view, when a format access uses them.
    let use_decode_x = uses_typed_decode_x(func);
    let (mut decode, mut encode, mut d16) = (false, false, false);
    func.for_each_stmt(&mut |s| {
        if let Stmt::Buffer(op, _) = s
            && let Some(sh) = shape(op.op)
            && sh.conversion == Conversion::View
        {
            d16 |= sh.d16;
            if !op.store {
                decode |= !(use_decode_x && is_single_typed_load(op));
            }
            encode |= op.store;
        }
    });
    if d16 {
        out.push_str(&typed_d16_kind());
    }
    if use_decode_x {
        out.push_str(&typed_decoder_x());
    }
    if decode {
        out.push_str(&typed_decoder());
    }
    if encode {
        out.push_str(&typed_encoder());
    }
    Ok(())
}

/// The dword at byte address `at` of a buffer resource.
pub(super) fn word(name: &str, class: Option<RegisterClass>, at: &str) -> String {
    match class {
        Some(RegisterClass::Cbv) => format!("{name}.w[{at} >> 4][({at} >> 2) & 3u]"),
        _ => format!("{name}.Load({at})"),
    }
}

/// How the bits of one channel of a buffer format become a 32 bit register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// The format of a typed buffer access: channel widths from the low bits up.
struct Format {
    widths: &'static [u32],
    num: Num,
    /// The first and the third channel in memory are the third and the first register (the
    /// `B8G8R8A8` order).
    swapped: bool,
}

/// The DXGI formats of typed buffer views the conversions know, by their number.
const DXGI_FORMATS: [u32; 43] = [
    2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 24, 25, 26, 28, 30, 31, 32, 34, 35, 36, 37,
    38, 41, 42, 43, 49, 50, 51, 52, 54, 56, 57, 58, 59, 61, 62, 63, 64, 87,
];

impl Format {
    /// The formats a buffer access can name. The name of a format lists the channels from the
    /// high bits down (`2_10_10_10` has x in the low ten bits).
    fn of(format: u8) -> Option<Format> {
        use Num::*;
        const SIX: [Num; 6] = [Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
        const SEVEN: [Num; 7] = [Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
        const WIDE: [Num; 3] = [Uint, Sint, Float];
        let groups: [(u8, &'static [u32], &[Num]); 12] = [
            (1, &[8], &SIX),
            (7, &[16], &SEVEN),
            (14, &[8, 8], &SIX),
            (20, &[32], &WIDE),
            (23, &[16, 16], &SEVEN),
            (44, &[2, 10, 10, 10], &SIX),
            (50, &[10, 10, 10, 2], &SIX),
            (56, &[8, 8, 8, 8], &SIX),
            (62, &[32, 32], &WIDE),
            (65, &[16, 16, 16, 16], &SEVEN),
            (72, &[32, 32, 32], &WIDE),
            (75, &[32, 32, 32, 32], &WIDE),
        ];
        groups.iter().find_map(|(first, widths, nums)| {
            let k = usize::from(format.checked_sub(*first)?);
            nums.get(k).map(|&num| Format {
                widths,
                num,
                swapped: false,
            })
        })
    }

    /// The layout of DXGI format `format` (the number of `DXGI_FORMAT`), for the formats a
    /// typed buffer view of the title can have. A DXGI name lists the channels from the low
    /// bits up.
    fn dxgi(format: u32) -> Option<Format> {
        use Num::*;
        let (widths, num): (&'static [u32], Num) = match format {
            2 => (&[32, 32, 32, 32], Float),
            3 => (&[32, 32, 32, 32], Uint),
            4 => (&[32, 32, 32, 32], Sint),
            6 => (&[32, 32, 32], Float),
            7 => (&[32, 32, 32], Uint),
            8 => (&[32, 32, 32], Sint),
            10 => (&[16, 16, 16, 16], Float),
            11 => (&[16, 16, 16, 16], Unorm),
            12 => (&[16, 16, 16, 16], Uint),
            13 => (&[16, 16, 16, 16], Snorm),
            14 => (&[16, 16, 16, 16], Sint),
            16 => (&[32, 32], Float),
            17 => (&[32, 32], Uint),
            18 => (&[32, 32], Sint),
            24 => (&[10, 10, 10, 2], Unorm),
            25 => (&[10, 10, 10, 2], Uint),
            // Floats without a sign: 5 bits of exponent, 6 and 5 bits of mantissa.
            26 => (&[11, 11, 10], Float),
            28 | 87 => (&[8, 8, 8, 8], Unorm),
            30 => (&[8, 8, 8, 8], Uint),
            31 => (&[8, 8, 8, 8], Snorm),
            32 => (&[8, 8, 8, 8], Sint),
            34 => (&[16, 16], Float),
            35 => (&[16, 16], Unorm),
            36 => (&[16, 16], Uint),
            37 => (&[16, 16], Snorm),
            38 => (&[16, 16], Sint),
            41 => (&[32], Float),
            42 => (&[32], Uint),
            43 => (&[32], Sint),
            49 => (&[8, 8], Unorm),
            50 => (&[8, 8], Uint),
            51 => (&[8, 8], Snorm),
            52 => (&[8, 8], Sint),
            54 => (&[16], Float),
            56 => (&[16], Unorm),
            57 => (&[16], Uint),
            58 => (&[16], Snorm),
            59 => (&[16], Sint),
            61 => (&[8], Unorm),
            62 => (&[8], Uint),
            63 => (&[8], Snorm),
            64 => (&[8], Sint),
            _ => return None,
        };
        Some(Format {
            widths,
            num,
            swapped: format == 87,
        })
    }

    fn bytes(&self) -> u32 {
        self.widths.iter().sum::<u32>() / 8
    }

    /// The four registers a format load gives for the element whose bits start at bit 0 of the
    /// `uint4 m`, as expressions. A channel the format does not have reads 0, the fourth reads 1.
    fn registers(&self) -> Vec<String> {
        let mut chans = Vec::new();
        let mut bit = 0u32;
        for &w in self.widths {
            let (d, s) = (["m.x", "m.y", "m.z", "m.w"][(bit / 32) as usize], bit % 32);
            let raw = if w == 32 {
                d.to_owned()
            } else {
                format!("(({d} >> {s}u) & {}u)", (1u64 << w) - 1)
            };
            let sext = format!("(int({raw} << {}u) >> {}u)", 32 - w, 32 - w);
            let max = ((1u64 << w) - 1) as f64;
            let smax = ((1u64 << (w - 1)) - 1) as f64;
            chans.push(match self.num {
                Num::Unorm => format!("asuint(float({raw}) / {max:.1})"),
                Num::Snorm => format!("asuint(max(float({sext}) / {smax:.1}, -1.0))"),
                Num::Uscaled => format!("asuint(float({raw}))"),
                Num::Sscaled => format!("asuint(float({sext}))"),
                Num::Uint => raw,
                Num::Sint if w == 32 => raw,
                Num::Sint => format!("uint({sext})"),
                Num::Float if w == 16 => format!("asuint(f16tof32({raw}))"),
                Num::Float if w == 32 => raw,
                // A float of 11 or 10 bits has no sign: its bits are those of a 16 bit float
                // from the exponent down.
                Num::Float => format!("asuint(f16tof32({raw} << {}u))", 15 - w),
            });
            bit += w;
        }
        if self.swapped {
            chans.swap(0, 2);
        }
        let one = if matches!(self.num, Num::Uint | Num::Sint) {
            "1u"
        } else {
            "0x3f800000u"
        };
        while chans.len() < 3 {
            chans.push("0u".to_owned());
        }
        if chans.len() < 4 {
            chans.push(one.to_owned());
        }
        chans
    }

    /// `uint4 decant_tfmt<format>(uint4 m)`: [`Format::registers`] as a function of its own.
    fn decoder(&self, format: u8) -> String {
        format!(
            "uint4 decant_tfmt{format}(uint4 m) {{ return uint4({}); }}\n",
            self.registers().join(", ")
        )
    }

    /// The four dwords of the element a format store writes for the registers `uint4 d`, as
    /// expressions, for a DXGI format. The conversions are those of Direct3D for a write to a
    /// view of the format: a normalized value is clamped and rounded to the nearest step, an
    /// integer keeps its low bits.
    fn element(&self) -> [String; 4] {
        let mut words: [Vec<String>; 4] = Default::default();
        let mut bit = 0u32;
        for (k, &w) in self.widths.iter().enumerate() {
            let from = if self.swapped && k != 3 { 2 - k } else { k };
            let d = ["d.x", "d.y", "d.z", "d.w"][from];
            let mask = (1u64 << w) - 1;
            let max = mask as f64;
            let smax = ((1u64 << (w - 1)) - 1) as f64;
            let field = match self.num {
                _ if w == 32 => d.to_owned(),
                Num::Unorm => format!("uint(saturate(asfloat({d})) * {max:.1} + 0.5)"),
                Num::Snorm => format!("(uint(decant_to_snorm(asfloat({d}), {smax:.1})) & {mask}u)"),
                Num::Uint | Num::Sint => format!("({d} & {mask}u)"),
                Num::Float if w == 16 => {
                    format!("(decant_pack_small_float({d}, 10u) | (({d} >> 16u) & 32768u))")
                }
                // Without a sign: a negative value is stored as 0.
                Num::Float => format!("decant_pack_unsigned_float({d}, {}u)", w - 5),
                Num::Uscaled | Num::Sscaled => unreachable!("no DXGI format is scaled"),
            };
            let (word, shift) = ((bit / 32) as usize, bit % 32);
            words[word].push(if shift == 0 {
                field
            } else {
                format!("({field} << {shift}u)")
            });
            bit += w;
        }
        words.map(|parts| {
            if parts.is_empty() {
                "0u".to_owned()
            } else {
                parts.join(" | ")
            }
        })
    }
}

/// The cases of a switch over the DXGI formats, formats with the same text under one body.
fn dxgi_cases(body: impl Fn(&Format) -> String) -> String {
    let mut bodies: Vec<(String, Vec<u32>)> = Vec::new();
    for n in DXGI_FORMATS {
        let Some(f) = Format::dxgi(n) else { continue };
        let text = body(&f);
        match bodies.iter_mut().find(|(t, _)| *t == text) {
            Some((_, numbers)) => numbers.push(n),
            None => bodies.push((text, vec![n])),
        }
    }
    let mut out = String::new();
    for (text, numbers) in bodies {
        let labels: Vec<String> = numbers.iter().map(|n| format!("case {n}u:")).collect();
        writeln!(out, "    {} {text}", labels.join(" ")).expect("writing to a string");
    }
    out
}

/// The VGPR half is a float, unsigned integer, or signed integer according to the view.
fn typed_d16_kind() -> String {
    format!(
        "uint decant_typed_d16_kind(uint format) {{\n    switch (format) {{\n{}    default: return 0u;\n    }}\n}}\n",
        dxgi_cases(|f| match f.num {
            Num::Uint => "return 1u;".to_owned(),
            Num::Sint => "return 2u;".to_owned(),
            _ => "return 0u;".to_owned(),
        })
    )
}

/// Builds a branchless HLSL predicate testing whether `format` belongs to the subset of
/// [`DXGI_FORMATS`] selected by `pred`, using `b0` (`format < 32`) and `b1` (`32..64`).
fn dxgi_format_test(pred: impl Fn(u32, &Format) -> bool) -> String {
    let (mut lo, mut hi) = (0u32, 0u32);
    let mut extra = Vec::new();
    for n in DXGI_FORMATS {
        let Some(f) = Format::dxgi(n) else { continue };
        if pred(n, &f) {
            if n < 32 {
                lo |= 1u32 << n;
            } else if n < 64 {
                hi |= 1u32 << (n - 32);
            } else {
                extra.push(n);
            }
        }
    }
    let mut parts = Vec::new();
    match (lo != 0, hi != 0) {
        (true, true) => parts.push(format!("(((b0 & {lo:#x}u) | (b1 & {hi:#x}u)) != 0u)")),
        (true, false) => parts.push(format!("((b0 & {lo:#x}u) != 0u)")),
        (false, true) => parts.push(format!("((b1 & {hi:#x}u) != 0u)")),
        (false, false) => {}
    }
    for n in extra {
        parts.push(format!("(format == {n}u)"));
    }
    if parts.is_empty() {
        "false".to_owned()
    } else {
        parts.join(" || ")
    }
}

/// `uint decant_typed_decode_x(uint m, uint format)`: channel `x` of [`typed_decoder`] for the first
/// dword `m` of an element, emitted branchlessly so DXC's unconditional inlining does not add
/// switch blocks per `buffer_load_format_x` call.
fn typed_decoder_x() -> String {
    let ch0_width = |f: &Format| {
        if f.swapped { f.widths[2] } else { f.widths[0] }
    };
    let w32 = dxgi_format_test(|_, f| ch0_width(f) == 32);
    let w16 = dxgi_format_test(|_, f| ch0_width(f) == 16);
    let w11 = dxgi_format_test(|_, f| ch0_width(f) == 11);
    let w10 = dxgi_format_test(|_, f| ch0_width(f) == 10);
    let w8 = dxgi_format_test(|_, f| ch0_width(f) == 8);
    let swapped = dxgi_format_test(|_, f| f.swapped);
    let is_raw = dxgi_format_test(|_, f| ch0_width(f) == 32 || f.num == Num::Uint);
    let is_sint = dxgi_format_test(|_, f| ch0_width(f) < 32 && f.num == Num::Sint);
    let is_unorm = dxgi_format_test(|_, f| f.num == Num::Unorm);
    let is_snorm = dxgi_format_test(|_, f| f.num == Num::Snorm);
    let is_sflt = dxgi_format_test(|_, f| ch0_width(f) < 32 && f.num == Num::Float);
    format!(
        "uint decant_typed_decode_x(uint m, uint format) {{\n\
        \x20   uint b0 = format < 32u ? (1u << (format & 31u)) : 0u;\n\
        \x20   uint b1 = (format >= 32u && format < 64u) ? (1u << ((format - 32u) & 31u)) : 0u;\n\
        \x20   bool w32 = {w32};\n\
        \x20   bool w16 = {w16};\n\
        \x20   bool w11 = {w11};\n\
        \x20   bool w10 = {w10};\n\
        \x20   bool w8 = {w8};\n\
        \x20   uint mask = w16 ? 65535u : (w8 ? 255u : (w10 ? 1023u : (w11 ? 2047u : 0u)));\n\
        \x20   uint raw = w32 ? m : ((m >> (({swapped}) ? 16u : 0u)) & mask);\n\
        \x20   uint sh = w16 ? 16u : 24u;\n\
        \x20   int sraw = int(raw << sh) >> sh;\n\
        \x20   float udiv = w16 ? 65535.0 : (w10 ? 1023.0 : 255.0);\n\
        \x20   float sdiv = w16 ? 32767.0 : 127.0;\n\
        \x20   return ({is_raw}) ? raw\n\
        \x20       : ({is_sint}) ? uint(sraw)\n\
        \x20       : ({is_unorm}) ? asuint(float(raw) / udiv)\n\
        \x20       : ({is_snorm}) ? asuint(max(float(sraw) / sdiv, -1.0))\n\
        \x20       : ({is_sflt}) ? asuint(f16tof32(raw << (w11 ? 4u : 0u)))\n\
        \x20       : 0u;\n\
        }}\n"
    )
}

/// `uint4 decant_typed_decode(uint4 m, uint format)`: the four registers a format load gives for
/// the element whose dwords are `m`, in a view of DXGI format `format`. A format the switch
/// does not know reads zeros.
fn typed_decoder() -> String {
    format!(
        "uint4 decant_typed_decode(uint4 m, uint format) {{\n    switch (format) {{\n{}    default: return uint4(0u, 0u, 0u, 0u);\n    }}\n}}\n",
        dxgi_cases(|f| format!("return uint4({});", f.registers().join(", ")))
    )
}

/// `bool decant_typed_encode(uint4 d, uint format, out uint4 m)`: the dwords `m` of the element a
/// format store writes for the registers `d`, in a view of DXGI format `format`. False for a
/// format the switch does not know: nothing is written then.
fn typed_encoder() -> String {
    // Direct3D float narrowing rounds toward zero and saturates finite overflow.
    let float_pack = r#"uint decant_pack_small_float(uint bits, uint fraction) {
    uint exponent = (bits >> 23u) & 255u;
    uint mantissa = bits & 0x7fffffu;
    uint infinity = 31u << fraction;
    if (exponent == 255u) return infinity | uint(mantissa != 0u);
    if (exponent > 142u) return infinity - 1u;
    uint shift = 23u - fraction;
    if (exponent >= 113u) return ((exponent - 112u) << fraction) | (mantissa >> shift);
    if (exponent < 113u - fraction) return 0u;
    return (mantissa | 0x800000u) >> (shift + 113u - exponent);
}
uint decant_pack_unsigned_float(uint bits, uint fraction) {
    bool nan = (bits & 0x7fffffffu) > 0x7f800000u;
    return bits < 0x80000000u || nan ? decant_pack_small_float(bits, fraction) : 0u;
}
"#;
    format!(
        "{float_pack}int decant_to_snorm(float f, float scale) {{ f = (f != f) ? 0.0 : clamp(f, -1.0, 1.0) * scale; return int(f >= 0.0 ? f + 0.5 : f - 0.5); }}\nbool decant_typed_encode(uint4 d, uint format, out uint4 m) {{\n    m = uint4(0u, 0u, 0u, 0u);\n    switch (format) {{\n{}    default: return false;\n    }}\n}}\n",
        dxgi_cases(|f| format!("m = uint4({}); return true;", f.element().join(", ")))
    )
}

/// The byte address of an access on a byte addressed view, and the condition under which it is
/// inside the view (`None`: always, the view bounds it).
struct Address {
    at: String,
    inside: Option<String>,
}

impl Printer<'_> {
    fn buf_refuse_form(op: &BufferOp, what: &'static str) -> Refusal {
        Refusal::new(RefusalKind::Form(op.op, what), "").at(op.pc)
    }

    fn buf_refuse_res(op: &BufferOp, what: &'static str) -> Refusal {
        Refusal::new(RefusalKind::Resource(what), op.op.name()).at(op.pc)
    }

    /// The address of a raw access `op`, with the lines that compute it printed. A typed view
    /// is addressed like a structured one: its row has the element size as the stride.
    fn buf_address(&mut self, op: &BufferOp, view: View, id: ResId) -> Result<Address, Refusal> {
        let u = self.unique;
        let off = self.e(&op.offset);
        let Some(index) = &op.index else {
            self.line(&format!("uint decant_a{u} = {off};"));
            return Ok(Address {
                at: format!("decant_a{u}"),
                inside: None,
            });
        };
        let index = self.e(index);
        match view {
            View::Constant => {
                self.line(&format!("uint decant_a{u} = ({index} * 16u + {off});"));
                Ok(Address {
                    at: format!("decant_a{u}"),
                    inside: None,
                })
            }
            View::Structured | View::Typed => {
                let Some(slot) = self.host_slot(id) else {
                    return Err(Self::buf_refuse_res(op, "buffer-structured-root"));
                };
                // The row of the host heap slot: record stride and record count. The preamble
                // refused the program when the layer keeps no shadow.
                self.line(&format!(
                    "uint2 decant_row{u} = decant_view_shadow_load2({slot} * 16u);"
                ));
                self.line(&format!("uint decant_i{u} = {index};"));
                self.line(&format!("uint decant_a{u} = decant_i{u} * decant_row{u}.x + {off};"));
                Ok(Address {
                    at: format!("decant_a{u}"),
                    // A stride of 0 marks a slot without a raw view of records: nothing is read
                    // or written there.
                    inside: Some(format!("(decant_row{u}.x != 0u && decant_i{u} < decant_row{u}.y)")),
                })
            }
            // A raw descriptor has stride 0: the index adds nothing. Not seen yet, so refused.
            View::Raw => Err(Self::buf_refuse_res(op, "buffer-raw-index")),
        }
    }

    pub(super) fn buffer(&mut self, op: &BufferOp, guard: Guard) -> Result<(), Refusal> {
        let Desc::Res(ResId(id)) = op.buffer else {
            unreachable!("the resource pass replaced every register descriptor")
        };
        let r = &self.func.resources[id as usize];
        let view = view(self.func, id as usize);
        let sh = shape(op.op).ok_or_else(|| Self::buf_refuse_form(op, "operand"))?;
        self.unique += 1;
        let ty = resource_type(
            self.func,
            r,
            super::class_of(r.placement.as_ref().expect("placed")),
        )
        .ok_or_else(|| Self::buf_refuse_res(op, "class-mismatch"))?;
        let name = self.use_resource(ResId(id), &ty);
        if sh.conversion == Conversion::View {
            if view != View::Typed {
                // The title made the view raw or structured: its row names no format.
                return Err(Self::buf_refuse_res(op, "buffer-view-mismatch"));
            }
            return self.buf_typed(op, sh, &name, ResId(id), guard);
        }
        if view == View::Constant && sh.store {
            return Err(Self::buf_refuse_res(op, "buffer-view-mismatch"));
        }
        let cls = if view == View::Constant {
            Some(RegisterClass::Cbv)
        } else {
            None
        };
        let addr = self.buf_address(op, view, ResId(id))?;
        let u = self.unique;
        let read = |at: &str| {
            let w = word(&name, cls, at);
            match &addr.inside {
                Some(c) => format!("({c} ? {w} : 0u)"),
                None => w,
            }
        };
        let a = addr.at.clone();
        if let Some(kind) = sh.atomic {
            let lines = self.buf_atomic(op, kind, &name, &a);
            let oob = op.glc.then(|| format!("v{} = 0u;", op.data));
            return self.buf_finish_write(guard, &addr, lines, oob.as_deref());
        }
        if sh.store {
            if sh.conversion == Conversion::Instruction {
                return Err(Self::buf_refuse_form(op, "tbuffer-store"));
            }
            let lines = if sh.bytes == 4 {
                (0..op.count)
                    .map(|k| format!("{name}.Store({a} + {}u, v{});", 4 * k, op.data + k))
                    .collect()
            } else {
                // Clear the bytes, then set them: two atomics that leave the other bytes of the
                // dword to other lanes.
                let mask = if sh.bytes == 1 { "0xffu" } else { "0xffffu" };
                vec![
                    format!("uint decant_d{u} = {a} & ~3u;"),
                    format!("uint decant_s{u} = ({a} & 3u) * 8u;"),
                    format!("{name}.InterlockedAnd(decant_d{u}, ~({mask} << decant_s{u}));"),
                    format!(
                        "{name}.InterlockedOr(decant_d{u}, (v{} & {mask}) << decant_s{u});",
                        op.data
                    ),
                ]
            };
            return self.buf_finish_write(guard, &addr, lines, None);
        }
        // Loads: all dwords are read before the first register is written, the address may be
        // in one of them.
        let mut values = Vec::new();
        match sh.conversion {
            Conversion::Instruction => {
                let f = Format::of(op.format).ok_or_else(|| Self::buf_refuse_form(op, "format"))?;
                let bytes = f.bytes();
                let m = if bytes < 4 {
                    format!(
                        "uint4({} >> (({a} & 3u) * 8u), 0u, 0u, 0u)",
                        read(&format!("({a} & ~3u)"))
                    )
                } else {
                    let words: Vec<String> = (0..4)
                        .map(|k| {
                            if k < bytes / 4 {
                                read(&format!("({a} + {}u)", 4 * k))
                            } else {
                                "0u".to_owned()
                            }
                        })
                        .collect();
                    format!("uint4({})", words.join(", "))
                };
                self.line(&format!("uint4 decant_t{u} = decant_tfmt{}({m});", op.format));
                for k in 0..op.count {
                    values.push(format!("decant_t{u}.{}", ["x", "y", "z", "w"][usize::from(k)]));
                }
            }
            Conversion::None if sh.bytes == 4 => {
                for k in 0..op.count {
                    self.line(&format!(
                        "uint decant_t{u}_{k} = {};",
                        read(&format!("({a} + {}u)", 4 * u32::from(k)))
                    ));
                    values.push(format!("decant_t{u}_{k}"));
                }
            }
            Conversion::None => {
                let bits = u32::from(sh.bytes) * 8;
                let field = format!(
                    "(({} >> (({a} & 3u) * 8u)) & {}u)",
                    read(&format!("({a} & ~3u)")),
                    (1u64 << bits) - 1
                );
                let value = if sh.signed {
                    format!("uint(int({field} << {}u) >> {}u)", 32 - bits, 32 - bits)
                } else {
                    field
                };
                self.line(&format!("uint decant_t{u}_0 = {value};"));
                values.push(format!("decant_t{u}_0"));
            }
            Conversion::View => unreachable!("refused above"),
        }
        let lines = values
            .iter()
            .enumerate()
            .map(|(k, v)| format!("v{} = {v};", op.data + k as u8))
            .collect();
        self.guarded(guard, lines);
        Ok(())
    }

    /// A store or atomic: only in an active lane, and only inside a structured view.
    /// A returning atomic (`glc`) outside the view writes zero to its data register,
    /// matching hardware and out of bounds loads.
    fn buf_finish_write(
        &mut self,
        guard: Guard,
        addr: &Address,
        lines: Vec<String>,
        oob: Option<&str>,
    ) -> Result<(), Refusal> {
        let lines = match &addr.inside {
            Some(c) => {
                let mut l = vec![format!("if {c} {{")];
                l.extend(lines.into_iter().map(|x| format!("    {x}")));
                if let Some(fallback) = oob {
                    l.push("} else {".to_owned());
                    l.push(format!("    {fallback}"));
                }
                l.push("}".to_owned());
                l
            }
            None => lines,
        };
        self.guarded(guard, lines);
        Ok(())
    }

    /// Emits an atomic on the dword at byte address `a`.
    fn buf_atomic(&mut self, op: &BufferOp, kind: Atomic, name: &str, a: &str) -> Vec<String> {
        let u = self.unique;
        let (v, w) = (format!("v{}", op.data), format!("v{}", op.data + 1));
        let old = format!("decant_o{u}");
        let mut lines = vec![format!("uint {old} = 0u;")];
        let call = |f: &str, args: &str| format!("{name}.{f}({a}, {args});");
        let load = format!("{name}.Load({a})");
        // The operations HLSL has: one call.
        let direct = match kind {
            Atomic::Swap => Some(call("InterlockedExchange", &format!("{v}, {old}"))),
            Atomic::Add => Some(call("InterlockedAdd", &format!("{v}, {old}"))),
            Atomic::Sub => Some(call("InterlockedAdd", &format!("0u - {v}, {old}"))),
            Atomic::UMin => Some(call("InterlockedMin", &format!("{v}, {old}"))),
            Atomic::UMax => Some(call("InterlockedMax", &format!("{v}, {old}"))),
            Atomic::And => Some(call("InterlockedAnd", &format!("{v}, {old}"))),
            Atomic::Or => Some(call("InterlockedOr", &format!("{v}, {old}"))),
            Atomic::Xor => Some(call("InterlockedXor", &format!("{v}, {old}"))),
            Atomic::CmpSwap => Some(call(
                "InterlockedCompareExchange",
                &format!("{w}, {v}, {old}"),
            )),
            Atomic::SMin | Atomic::SMax => {
                let f = if kind == Atomic::SMin {
                    "InterlockedMin"
                } else {
                    "InterlockedMax"
                };
                lines.push(format!("int decant_so{u} = 0;"));
                lines.push(call(f, &format!("int({v}), decant_so{u}")));
                lines.push(format!("{old} = uint(decant_so{u});"));
                Some(String::new())
            }
            _ => None,
        };
        match direct {
            Some(l) if !l.is_empty() => lines.push(l),
            Some(_) => {}
            None => {
                // The rest is a compare exchange loop over the new value of each kind.
                let new = match kind {
                    Atomic::CSub => format!("({old} >= {v} ? {old} - {v} : 0u)"),
                    Atomic::Inc => format!("({old} >= {v} ? 0u : {old} + 1u)"),
                    Atomic::Dec => format!("(({old} == 0u || {old} > {v}) ? {v} : {old} - 1u)"),
                    Atomic::FMin => format!("asuint(min(asfloat({old}), asfloat({v})))"),
                    Atomic::FMax => format!("asuint(max(asfloat({old}), asfloat({v})))"),
                    Atomic::FCmpSwap => {
                        format!("(asfloat({old}) == asfloat({w}) ? {v} : {old})")
                    }
                    _ => unreachable!("handled by one call"),
                };
                lines.push(format!("{old} = {load};"));
                lines.push("[loop] while (true) {".to_owned());
                lines.push(format!("    uint decant_seen{u} = 0u;"));
                lines.push(format!(
                    "    {}",
                    call(
                        "InterlockedCompareExchange",
                        &format!("{old}, {new}, decant_seen{u}")
                    )
                ));
                lines.push(format!("    if (decant_seen{u} == {old}) break;"));
                lines.push(format!("    {old} = decant_seen{u};"));
                lines.push("}".to_owned());
            }
        }
        if op.glc {
            lines.push(format!("{v} = {old};"));
        }
        lines
    }

    /// A format access of a typed view the layer made raw: the dwords of the element,
    /// converted by the format of the shadow row of the host heap slot. An element past the
    /// count of the row reads as zero dwords and is not written. An element of 1 or 2 bytes is
    /// shifted out of the dword that holds it, and a store changes only its bytes.
    fn buf_typed(
        &mut self,
        op: &BufferOp,
        sh: Shape,
        name: &str,
        id: ResId,
        guard: Guard,
    ) -> Result<(), Refusal> {
        // The stride of a typed view is its element: an offset inside one is not an element.
        if op.offset != Expr::U32(0) {
            return Err(Self::buf_refuse_form(op, "typed-offset"));
        }
        // The preamble refused the program when the layer keeps no shadow.
        let Some(slot) = self.host_slot(id) else {
            return Err(Self::buf_refuse_res(op, "buffer-structured-root"));
        };
        let u = self.unique;
        let e = match &op.index {
            Some(i) => self.e(i),
            None => "0u".to_owned(),
        };
        // The row: element size, element count, DXGI format, and the byte offset of the
        // title's first byte inside the view. Hoisted once at the start of `main` for fixed-slot
        // resources with repeated typed accesses.
        let row = if let Some(hoisted) = self.typed_rows.get(&id) {
            hoisted.clone()
        } else {
            self.line(&format!(
                "uint4 decant_row{u} = decant_view_shadow_load4({slot} * 16u);"
            ));
            format!("decant_row{u}")
        };
        self.line(&format!("uint decant_i{u} = {e};"));
        self.line(&format!(
            "uint decant_a{u} = decant_i{u} * {row}.x + ({row}.w & 15u);"
        ));
        let addr = Address {
            at: format!("decant_a{u}"),
            inside: Some(format!("({row}.x != 0u && decant_i{u} < {row}.y)")),
        };
        // The dwords of an element after the first exist when the element is that large.
        let rest = |k: u32| format!("{row}.x > {}u", 4 * k);
        // An element of 1 or 2 bytes (R8, R16, R8G8 ...) lies inside one dword, at the shift of
        // its first byte; a larger one starts on a dword, its shift is 0.
        let dword = format!("(decant_a{u} & ~3u)");
        let shift = format!("((decant_a{u} & 3u) * 8u)");
        if sh.d16 {
            self.line(&format!("uint decant_kind{u} = decant_typed_d16_kind({row}.z);"));
        }
        if sh.store {
            let data: Vec<String> = (0..4u8)
                .map(|k| {
                    if k >= sh.count {
                        return "0u".to_owned();
                    }
                    if !sh.d16 {
                        return format!("v{}", op.data + k);
                    }
                    let v = op.data + k / 2;
                    let shift = (k % 2) * 16;
                    let half = format!("((v{v} >> {shift}u) & 65535u)");
                    format!(
                        "(decant_kind{u} == 0u ? asuint(f16tof32({half})) : \
                         (decant_kind{u} == 2u ? uint(int({half} << 16u) >> 16u) : {half}))"
                    )
                })
                .collect();
            let mut lines = vec![
                format!("uint4 decant_m{u};"),
                format!(
                    "if (decant_typed_encode(uint4({}), {row}.z, decant_m{u})) {{",
                    data.join(", ")
                ),
                // A small element changes only its bytes of the dword, by two atomics that leave
                // the other bytes to the lanes that write them.
                format!("    if ({row}.x < 4u) {{"),
                format!("        uint decant_k{u} = {row}.x == 1u ? 0xffu : 0xffffu;"),
                format!("        {name}.InterlockedAnd({dword}, ~(decant_k{u} << {shift}));"),
                format!(
                    "        {name}.InterlockedOr({dword}, (decant_m{u}.x & decant_k{u}) << {shift});"
                ),
                "    } else {".to_owned(),
                format!("        {name}.Store(decant_a{u}, decant_m{u}.x);"),
            ];
            for (k, c) in [(1, "y"), (2, "z"), (3, "w")] {
                lines.push(format!(
                    "        if ({}) {name}.Store(decant_a{u} + {}u, decant_m{u}.{c});",
                    rest(k),
                    4 * k
                ));
            }
            lines.push("    }".to_owned());
            lines.push("}".to_owned());
            return self.buf_finish_write(guard, &addr, lines, None);
        }
        let inside = addr.inside.as_deref().expect("set above");
        if self.typed_decode_x && is_single_typed_load(op) {
            self.line(&format!(
                "uint decant_m{u} = {inside} ? ({name}.Load({dword}) >> {shift}) : 0u;"
            ));
            self.line(&format!(
                "uint decant_t{u} = decant_typed_decode_x(decant_m{u}, {row}.z);"
            ));
            self.guarded(guard, vec![format!("v{} = decant_t{u};", op.data)]);
            return Ok(());
        }
        self.line(&format!("uint4 decant_m{u} = uint4(0u, 0u, 0u, 0u);"));
        self.line(&format!("if {inside} {{"));
        self.line(&format!(
            "    decant_m{u} = uint4({name}.Load({dword}) >> {shift}, {});",
            (1..4)
                .map(|k| format!("{} ? {name}.Load(decant_a{u} + {}u) : 0u", rest(k), 4 * k))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        self.line("}");
        self.line(&format!(
            "uint4 decant_t{u} = decant_typed_decode(decant_m{u}, {row}.z);"
        ));
        let lines = (0..op.count)
            .map(|k| {
                let register = op.data + k;
                let component = |n| ["x", "y", "z", "w"][usize::from(n)];
                if !sh.d16 {
                    return format!("v{register} = decant_t{u}.{};", component(k));
                }
                let half = |n| {
                    let value = format!("decant_t{u}.{}", component(n));
                    format!("(decant_kind{u} == 0u ? f32tof16(asfloat({value})) : ({value} & 65535u))")
                };
                let low = half(2 * k);
                let high = if 2 * k + 1 < sh.count {
                    format!("({} << 16u)", half(2 * k + 1))
                } else {
                    format!("(v{register} & 0xffff0000u)")
                };
                format!("v{register} = {low} | {high};")
            })
            .collect();
        self.guarded(guard, lines);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DeclaredResource, MirrorAbi};
    use crate::hlsl::{self, Context, ENTRY};
    use crate::ir::{Node, Var};
    use crate::stage::Interface;
    use crate::tables::Op;

    fn compile_cs(text: &str) {
        if let Ok(dxc) = decant_dxc::Dxc::load() {
            let args = vec!["-Gis".to_owned(), "-denorm".to_owned(), "ftz".to_owned()];
            dxc.compile(text, ENTRY, "cs_6_6", &args)
                .unwrap_or_else(|e| panic!("DXC compile failed:\n{e}\n{text}"));
        }
    }

    /// Reference IEEE 754 binary16 to binary32 conversion matching HLSL `f16tof32`.
    fn f16_to_f32_bits(h: u32) -> u32 {
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        if exp == 0 {
            if mant == 0 {
                return sign;
            }
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            return sign | ((e as u32) << 23) | ((m & 0x3ff) << 13);
        }
        if exp == 31 {
            return sign | 0x7f80_0000 | (mant << 13);
        }
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    }

    /// Evaluates channel 0 of `Format::dxgi(format)` directly from its `Format` description.
    fn ref_decode_x(m: u32, format: u32) -> u32 {
        let Some(f) = Format::dxgi(format) else {
            return 0;
        };
        let (bit, w) = if f.swapped {
            (16u32, f.widths[2])
        } else {
            (0u32, f.widths[0])
        };
        let raw = if w == 32 {
            m
        } else {
            (m >> bit) & (((1u64 << w) - 1) as u32)
        };
        let sext = ((raw << (32 - w)) as i32) >> (32 - w);
        let max = ((1u64 << w) - 1) as f32;
        let smax = ((1u64 << (w - 1)) - 1) as f32;
        match f.num {
            Num::Unorm => ((raw as f32) / max).to_bits(),
            Num::Snorm => ((sext as f32) / smax).max(-1.0f32).to_bits(),
            Num::Uint => raw,
            Num::Sint if w == 32 => raw,
            Num::Sint => sext as u32,
            Num::Float if w == 32 => raw,
            Num::Float if w == 16 => f16_to_f32_bits(raw),
            Num::Float => f16_to_f32_bits(raw << (15 - w)),
            Num::Uscaled | Num::Sscaled => unreachable!("no DXGI format is scaled"),
        }
    }

    /// Evaluates the exact branchless formula emitted by [`typed_decoder_x`].
    fn fast_decode_x(m: u32, format: u32) -> u32 {
        let ch0_width = |f: &Format| {
            if f.swapped { f.widths[2] } else { f.widths[0] }
        };
        let in_set = |pred: &dyn Fn(&Format) -> bool| -> bool {
            DXGI_FORMATS
                .iter()
                .any(|&n| n == format && Format::dxgi(n).is_some_and(|f| pred(&f)))
        };
        let w32 = in_set(&|f| ch0_width(f) == 32);
        let w16 = in_set(&|f| ch0_width(f) == 16);
        let w11 = in_set(&|f| ch0_width(f) == 11);
        let w10 = in_set(&|f| ch0_width(f) == 10);
        let w8 = in_set(&|f| ch0_width(f) == 8);
        let swapped = in_set(&|f| f.swapped);
        let is_raw = in_set(&|f| ch0_width(f) == 32 || f.num == Num::Uint);
        let is_sint = in_set(&|f| ch0_width(f) < 32 && f.num == Num::Sint);
        let is_unorm = in_set(&|f| f.num == Num::Unorm);
        let is_snorm = in_set(&|f| f.num == Num::Snorm);
        let is_sflt = in_set(&|f| ch0_width(f) < 32 && f.num == Num::Float);
        let mask = if w16 {
            65535u32
        } else if w8 {
            255u32
        } else if w10 {
            1023u32
        } else if w11 {
            2047u32
        } else {
            0u32
        };
        let raw = if w32 {
            m
        } else {
            (m >> if swapped { 16u32 } else { 0u32 }) & mask
        };
        let sh = if w16 { 16u32 } else { 24u32 };
        let sraw = ((raw << sh) as i32) >> sh;
        let udiv = if w16 {
            65535.0f32
        } else if w10 {
            1023.0f32
        } else {
            255.0f32
        };
        let sdiv = if w16 { 32767.0f32 } else { 127.0f32 };
        if is_raw {
            raw
        } else if is_sint {
            sraw as u32
        } else if is_unorm {
            ((raw as f32) / udiv).to_bits()
        } else if is_snorm {
            ((sraw as f32) / sdiv).max(-1.0f32).to_bits()
        } else if is_sflt {
            f16_to_f32_bits(raw << if w11 { 4u32 } else { 0u32 })
        } else {
            0u32
        }
    }

    #[test]
    fn typed_decoder_x_matches_four_channel_decoder_for_all_formats() {
        let samples = [
            0u32,
            1,
            0x0000_007f,
            0x0000_0080,
            0x0000_00ff,
            0x0000_03ff,
            0x0000_07ff,
            0x0000_7fff,
            0x0000_8000,
            0x0000_ffff,
            0x0080_0000,
            0x00ff_0080,
            0x3c00_3c00,
            0x7bff_03ff,
            0x1234_5678,
            0x8000_8080,
            0xffff_ffff,
        ];
        for fmt in 0..=100u32 {
            for &word in &samples {
                assert_eq!(
                    fast_decode_x(word, fmt),
                    ref_decode_x(word, fmt),
                    "mismatch for DXGI format {fmt} word {word:#x}"
                );
            }
        }
    }

    fn typed_mirror_res(name: &str, dword: u8, slot: u32) -> Resource {
        Resource {
            origin: Origin::Table {
                sgpr: 4,
                offset: slot * 16,
            },
            usage: ResourceUse::Words { written: false },
            declared: Some(DeclaredResource {
                class: RegisterClass::Srv,
                lower: slot,
                upper: slot,
                space: 0,
                psv_type: 3,
                psv_kind: 0,
            }),
            placement: Some(Placement::HeapAtMirror {
                class: RegisterClass::Srv,
                dword,
                slot,
            }),
            name: name.to_owned(),
        }
    }

    fn typed_index_res(name: &str, index: Var) -> Resource {
        Resource {
            origin: Origin::HeapIndex {
                index,
                sampler: false,
            },
            usage: ResourceUse::Words { written: false },
            declared: Some(DeclaredResource {
                class: RegisterClass::Srv,
                lower: 0,
                upper: 0,
                space: 0,
                psv_type: 3,
                psv_kind: 0,
            }),
            placement: Some(Placement::HeapAtConsoleIndex {
                class: RegisterClass::Srv,
            }),
            name: name.to_owned(),
        }
    }

    fn raw_uav_res(name: &str, dword: u8, slot: u32) -> Resource {
        Resource {
            origin: Origin::Table {
                sgpr: 5,
                offset: slot * 16,
            },
            usage: ResourceUse::Words { written: true },
            declared: Some(DeclaredResource {
                class: RegisterClass::Uav,
                lower: slot,
                upper: slot,
                space: 0,
                psv_type: 7,
                psv_kind: 0,
            }),
            placement: Some(Placement::HeapAtMirror {
                class: RegisterClass::Uav,
                dword,
                slot,
            }),
            name: name.to_owned(),
        }
    }

    fn fmt_load(op: Op, res: u32, data: u8, idx: Expr) -> Stmt {
        let sh = shape(op).unwrap();
        Stmt::Buffer(
            BufferOp {
                op,
                glc: false,
                store: false,
                bytes: 4,
                count: if sh.d16 {
                    sh.count.div_ceil(2)
                } else {
                    sh.count
                },
                data,
                index: Some(idx),
                offset: Expr::U32(0),
                format: 0,
                buffer: Desc::Res(ResId(res)),
                pc: 0,
            },
            Guard::Always,
        )
    }

    #[test]
    fn repeated_typed_loads_hoist_shadow_row_and_use_scalar_decode_x() {
        let iface = Interface {
            threads: Some([32, 1, 1]),
            ..Interface::default()
        };
        let func = Function {
            body: vec![
                Node::Block(vec![
                    Stmt::Set {
                        dst: Var::S(4),
                        value: Expr::U32(3),
                        guard: Guard::Always,
                    },
                    fmt_load(Op::buffer_load_format_x, 0, 8, Expr::Var(Var::V(0))),
                    fmt_load(Op::buffer_load_format_x, 0, 9, Expr::Var(Var::V(1))),
                    fmt_load(Op::buffer_load_format_x, 1, 10, Expr::Var(Var::V(0))),
                    fmt_load(Op::buffer_load_format_x, 2, 11, Expr::Var(Var::V(0))),
                    Stmt::Set {
                        dst: Var::S(4),
                        value: Expr::U32(4),
                        guard: Guard::Always,
                    },
                    fmt_load(Op::buffer_load_format_x, 2, 12, Expr::Var(Var::V(1))),
                ]),
                Node::If {
                    cond: Expr::bin(Bin::Ne, Expr::Var(Var::V(0)), Expr::U32(0)),
                    then: vec![Node::Block(vec![
                        fmt_load(Op::buffer_load_format_x, 0, 13, Expr::Var(Var::V(2))),
                        Stmt::Buffer(
                            BufferOp {
                                op: Op::buffer_store_dword,
                                glc: false,
                                store: true,
                                bytes: 4,
                                count: 1,
                                data: 13,
                                index: None,
                                offset: Expr::U32(0),
                                format: 0,
                                buffer: Desc::Res(ResId(3)),
                                pc: 0,
                            },
                            Guard::Always,
                        ),
                    ])],
                    els: Vec::new(),
                },
                Node::Return,
            ],
            resources: vec![
                typed_mirror_res("heap_t_s4_50", 4, 50),
                typed_mirror_res("heap_t_s4_12", 4, 12),
                typed_index_res("con_t", Var::S(4)),
                raw_uav_res("heap_u_s5_0", 5, 0),
            ],
            groupshared_dwords: 0,
            lanes: 32,
        };
        let text = hlsl::print(
            &func,
            &Context {
                iface: &iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        // Multi-use fixed-slot resource hoists one `decant_view_shadow_load4` at the start of `main`.
        assert!(
            text.contains(
                "uint4 decant_row_heap_t_s4_50 = decant_view_shadow_load4((decant_mirror[1].x + 50u) * 16u);"
            ),
            "{text}"
        );
        assert_eq!(
            text.matches("decant_view_shadow_load4((decant_mirror[1].x + 50u) * 16u)")
                .count(),
            1,
            "{text}"
        );
        // Single-use fixed-slot resource keeps its single inline load; dynamic heap index is never hoisted.
        assert!(!text.contains("decant_row_heap_t_s4_12"), "{text}");
        assert!(!text.contains("decant_row_con_t"), "{text}");
        assert_eq!(
            text.matches("decant_view_shadow_load4(s4 * 16u)").count(),
            2,
            "{text}"
        );
        assert_eq!(text.matches("decant_view_shadow.Load4(").count(), 1, "{text}");
        // Scalar `decant_typed_decode_x` is emitted instead of the 4-channel `decant_typed_decode` switch.
        assert!(
            text.contains("uint decant_typed_decode_x(uint m, uint format)"),
            "{text}"
        );
        assert!(!text.contains("decant_typed_decode("), "{text}");
        assert!(!text.contains("uint4 decant_t"), "{text}");
        assert!(!text.contains(".x > 4u"), "{text}");
        assert_eq!(text.matches("decant_typed_decode_x(").count(), 1 + 6, "{text}");
        compile_cs(&text);
    }

    #[test]
    fn d16_and_wide_typed_loads_keep_four_channel_decoder_while_hoisting_row() {
        let iface = Interface {
            threads: Some([32, 1, 1]),
            ..Interface::default()
        };
        let func = Function {
            body: vec![
                Node::Block(vec![
                    fmt_load(Op::buffer_load_format_x, 0, 8, Expr::Var(Var::V(0))),
                    fmt_load(Op::buffer_load_format_x, 0, 9, Expr::Var(Var::V(1))),
                    fmt_load(Op::buffer_load_format_xyzw, 0, 10, Expr::Var(Var::V(0))),
                    fmt_load(Op::buffer_load_format_d16_xy, 0, 14, Expr::Var(Var::V(1))),
                    Stmt::Buffer(
                        BufferOp {
                            op: Op::buffer_store_dword,
                            glc: false,
                            store: true,
                            bytes: 4,
                            count: 1,
                            data: 14,
                            index: None,
                            offset: Expr::U32(0),
                            format: 0,
                            buffer: Desc::Res(ResId(1)),
                            pc: 0,
                        },
                        Guard::Always,
                    ),
                ]),
                Node::Return,
            ],
            resources: vec![
                typed_mirror_res("heap_t_s4_50", 4, 50),
                raw_uav_res("heap_u_s5_0", 5, 0),
            ],
            groupshared_dwords: 0,
            lanes: 32,
        };
        let text = hlsl::print(
            &func,
            &Context {
                iface: &iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        assert_eq!(
            text.matches("decant_view_shadow_load4((decant_mirror[1].x + 50u) * 16u)")
                .count(),
            1,
            "{text}"
        );
        assert_eq!(text.matches("decant_view_shadow.Load4(").count(), 1, "{text}");
        assert!(text.contains("decant_typed_decode_x("), "{text}");
        assert!(text.contains("decant_typed_decode("), "{text}");
        assert!(
            text.contains("decant_typed_d16_kind(decant_row_heap_t_s4_50.z)"),
            "{text}"
        );
        compile_cs(&text);
    }

    #[test]
    fn structured_returning_atomic_zeros_destination_when_out_of_bounds() {
        let iface = Interface {
            threads: Some([32, 1, 1]),
            ..Interface::default()
        };
        let structured_uav = Resource {
            origin: Origin::Table { sgpr: 5, offset: 0 },
            usage: ResourceUse::Words { written: true },
            declared: Some(DeclaredResource {
                class: RegisterClass::Uav,
                lower: 0,
                upper: 0,
                space: 0,
                psv_type: 8,
                psv_kind: 0,
            }),
            placement: Some(Placement::HeapAtMirror {
                class: RegisterClass::Uav,
                dword: 5,
                slot: 0,
            }),
            name: "heap_u_s5_0".to_owned(),
        };
        let func = Function {
            body: vec![
                Node::Block(vec![
                    Stmt::Buffer(
                        BufferOp {
                            op: Op::buffer_atomic_cmpswap,
                            glc: true,
                            store: true,
                            bytes: 4,
                            count: 2,
                            data: 5,
                            index: Some(Expr::Var(Var::V(0))),
                            offset: Expr::U32(0),
                            format: 0,
                            buffer: Desc::Res(ResId(0)),
                            pc: 0,
                        },
                        Guard::Exec,
                    ),
                    Stmt::Buffer(
                        BufferOp {
                            op: Op::buffer_atomic_add,
                            glc: false,
                            store: true,
                            bytes: 4,
                            count: 1,
                            data: 8,
                            index: Some(Expr::Var(Var::V(0))),
                            offset: Expr::U32(0),
                            format: 0,
                            buffer: Desc::Res(ResId(0)),
                            pc: 4,
                        },
                        Guard::Exec,
                    ),
                ]),
                Node::Return,
            ],
            resources: vec![structured_uav],
            groupshared_dwords: 0,
            lanes: 32,
        };
        let text = hlsl::print(
            &func,
            &Context {
                iface: &iface,
                mirror: Some(MirrorAbi::default()),
                header: &[],
                code: &[],
            },
        )
        .unwrap();
        // Returning atomic writes zero when out of bounds, non-returning atomic has no else branch.
        assert!(
            text.contains("} else {\n            v5 = 0u;\n        }"),
            "{text}"
        );
        assert_eq!(text.matches("} else {").count(), 1, "{text}");
        compile_cs(&text);
    }
}
