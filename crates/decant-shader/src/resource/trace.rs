//! Descriptor tracing: which descriptor does each image, buffer and sampler operand hold?
//!
//! A forward dataflow over the flat graph of the function ([`super::flow`]) keeps a symbolic
//! value for every scalar register and temporary: a constant, the entry value of a user data
//! register, or a word of a descriptor. Where paths meet, equal values stay, and the words of
//! two different descriptors become the words of one "chosen by the flow" descriptor, so a
//! descriptor a branch picks is still known as a whole. The states are computed to a fixed point
//! first (loops), then one pass rewrites the statements with them.
//!
//! Where a descriptor comes from (measured on the programs of the title, see the design
//! document, "Resources"):
//!
//! * A scalar load whose base is (user data `K`, constant 0) reads the table `K` points at. A
//!   resource table has 32 byte slots (a buffer descriptor is the first four dwords of its
//!   slot), a sampler table 16 byte slots. Table parameters are only read at constant offsets.
//!   A load at an offset computed at run time always has a heap register as its base: the whole
//!   view heap or sampler heap, indexed by a number the title supplies.
//! * A buffer descriptor built in registers around a root descriptor: word 0 is user data `K`
//!   (the low half of the address), word 1 is user data `K + 1` with constant bits or'ed in
//!   (the stride field), words 2 and 3 are constants. It is the buffer behind that root
//!   descriptor.
//! * Four constants are a sampler the program builds itself.
//! * A scalar load whose base is (user data `K`, user data `K + 1`) reads the buffer behind a
//!   root descriptor directly.
//! * The four descriptor registers of a node intersection (`image_bvh_intersect_ray`,
//!   `image_bvh64_intersect_ray`) come from any of the places above. They become one buffer
//!   resource read as words, the node memory, see [`node_memory`].
//!
//! Anything else is refused by name: a load at an address the program computed
//! (`raw-address`), descriptor words used as numbers (`descriptor-as-data`), a descriptor that
//! is only partly known (`partial-descriptor`, `untraced-descriptor`), one chosen per lane
//! (`per-lane-descriptor`).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use super::flow::Flow;
use crate::api::{HEAP_PAGE_SHIFT, StageProgram, UserDataKind};
use crate::hlsl::{HEAP_MAP_LOAD, VIEW_SHADOW_LOAD, VIEW_SHADOW_LOAD2};
use crate::ir::{
    Bin, BvhIntersect, Desc, Expr, Function, Helper, ImageKind, Load, Origin, ResId, Resource,
    ResourceUse, Stmt, Ty, Un, Var,
};
use crate::refusal::{Refusal, RefusalKind};

mod guarded;

/// The host heap index of a console heap index, through the heap map of the mirror ABI
/// (`crate::api::MirrorAbi`): the layer places console heap pages where the PC heap has room.
pub static HEAP_HOST: Helper = Helper {
    name: "decant_heap_host",
    result: Ty::U32,
    text: "uint decant_heap_host(uint i) { return decant_heap_map_load((i >> 12) << 2) + (i & 4095u); }",
    needs: &[&HEAP_MAP_LOAD],
};
/// The record stride in bytes of the buffer view in a host heap slot, from the view shadow of
/// the mirror ABI (four dwords per host slot: stride, record count, two zeros).
pub static SHADOW_STRIDE: Helper = Helper {
    name: "decant_shadow_stride",
    result: Ty::U32,
    text: "uint decant_shadow_stride(uint slot) { return decant_view_shadow_load(slot * 16u); }",
    needs: &[&VIEW_SHADOW_LOAD],
};

/// The record count of the buffer view in a host heap slot, from the view shadow.
pub static SHADOW_COUNT: Helper = Helper {
    name: "decant_shadow_count",
    result: Ty::U32,
    text: "uint decant_shadow_count(uint slot) { return decant_view_shadow_load(slot * 16u + 4u); }",
    needs: &[&VIEW_SHADOW_LOAD],
};

/// One dword of a console image descriptor, made from the size of the PC texture: the fields a
/// program reads where the source asked for the size of a texture. GFX10 layout (Mesa,
/// `registers/gfx10-rsrc.json` and `ac_descriptors.c`): dword 1 bits 30 and 31 the low bits of
/// width minus one, dword 2 bits 0 to 11 the rest of it and bits 14 to 27 height minus one,
/// dword 3 bits 16 to 19 the last mip level (the base level, bits 12 to 15, is 0) and bits 28
/// to 31 the type, dword 4 bits 0 to 12 the last slice (depth minus one). Addresses, the
/// format, the swizzle and the tiling fields read as zero.
static IMG_WORD: Helper = Helper {
    name: "decant_img_word",
    result: Ty::U32,
    text: "uint decant_img_word(uint k, uint w, uint h, uint d, uint last, uint type) { if (k == 1u) return ((w - 1u) & 3u) << 30; if (k == 2u) return ((w - 1u) >> 2) | ((h - 1u) << 14) | 0x80000000u; if (k == 3u) return (last << 16) | (type << 28); if (k == 4u) return (d - 1u) & 0x1fffu; return 0u; }",
    needs: &[],
};

/// The same words from the view shadow row of the host slot, for an image whose kind nothing
/// says. Dwords 2 and 3 of the row of an image view: width minus one in bits 0 to 15 and
/// height minus one in bits 16 to 31, then the last slice in bits 0 to 12, the last mip
/// level in bits 16 to 19 and the type in bits 28 to 31 (the positions of descriptor dwords 4
/// and 3).
static IMG_ROW: Helper = Helper {
    name: "decant_img_row",
    result: Ty::U32,
    text: "uint decant_img_row(uint slot, uint k) { uint2 r = decant_view_shadow_load2(slot * 16u + 8u); uint w = r.x & 0xffffu; uint h = r.x >> 16; if (k == 1u) return (w & 3u) << 30; if (k == 2u) return (w >> 2) | (h << 14) | 0x80000000u; if (k == 3u) return r.y & 0xf00f0000u; if (k == 4u) return r.y & 0x1fffu; return 0u; }",
    needs: &[&VIEW_SHADOW_LOAD2],
};

macro_rules! image_words {
    ($($id:ident, $name:literal, $text:literal;)*) => {
        $(static $id: Helper = Helper { name: $name, result: Ty::U32, text: $text, needs: &[&IMG_WORD] };)*
    };
}

image_words! {
    IMG_1D, "decant_img_1d", "uint decant_img_1d(uint slot, uint k) { Texture1D<float4> t = ResourceDescriptorHeap[slot]; uint w, l; t.GetDimensions(0u, w, l); return decant_img_word(k, w, 1u, 1u, l - 1u, 8u); }";
    IMG_2D, "decant_img_2d", "uint decant_img_2d(uint slot, uint k) { Texture2D<float4> t = ResourceDescriptorHeap[slot]; uint w, h, l; t.GetDimensions(0u, w, h, l); return decant_img_word(k, w, h, 1u, l - 1u, 9u); }";
    IMG_3D, "decant_img_3d", "uint decant_img_3d(uint slot, uint k) { Texture3D<float4> t = ResourceDescriptorHeap[slot]; uint w, h, d, l; t.GetDimensions(0u, w, h, d, l); return decant_img_word(k, w, h, d, l - 1u, 10u); }";
    IMG_CUBE, "decant_img_cube", "uint decant_img_cube(uint slot, uint k) { TextureCube<float4> t = ResourceDescriptorHeap[slot]; uint w, h, l; t.GetDimensions(0u, w, h, l); return decant_img_word(k, w, h, 6u, l - 1u, 11u); }";
    IMG_CUBE_ARRAY, "decant_img_cube_array", "uint decant_img_cube_array(uint slot, uint k) { TextureCubeArray<float4> t = ResourceDescriptorHeap[slot]; uint w, h, e, l; t.GetDimensions(0u, w, h, e, l); return decant_img_word(k, w, h, 6u * e, l - 1u, 11u); }";
    IMG_1D_ARRAY, "decant_img_1d_array", "uint decant_img_1d_array(uint slot, uint k) { Texture1DArray<float4> t = ResourceDescriptorHeap[slot]; uint w, e, l; t.GetDimensions(0u, w, e, l); return decant_img_word(k, w, 1u, e, l - 1u, 12u); }";
    IMG_2D_ARRAY, "decant_img_2d_array", "uint decant_img_2d_array(uint slot, uint k) { Texture2DArray<float4> t = ResourceDescriptorHeap[slot]; uint w, h, e, l; t.GetDimensions(0u, w, h, e, l); return decant_img_word(k, w, h, e, l - 1u, 13u); }";
    IMG_2D_MS, "decant_img_2d_ms", "uint decant_img_2d_ms(uint slot, uint k) { Texture2DMS<float4> t = ResourceDescriptorHeap[slot]; uint w, h, n; t.GetDimensions(w, h, n); return decant_img_word(k, w, h, 1u, firstbithigh(n), 14u); }";
    IMG_2D_MS_ARRAY, "decant_img_2d_ms_array", "uint decant_img_2d_ms_array(uint slot, uint k) { Texture2DMSArray<float4> t = ResourceDescriptorHeap[slot]; uint w, h, e, n; t.GetDimensions(w, h, e, n); return decant_img_word(k, w, h, e, firstbithigh(n), 15u); }";
    IMG_RW_1D, "decant_img_rw_1d", "uint decant_img_rw_1d(uint slot, uint k) { RWTexture1D<float4> t = ResourceDescriptorHeap[slot]; uint w; t.GetDimensions(w); return decant_img_word(k, w, 1u, 1u, 0u, 8u); }";
    IMG_RW_2D, "decant_img_rw_2d", "uint decant_img_rw_2d(uint slot, uint k) { RWTexture2D<float4> t = ResourceDescriptorHeap[slot]; uint w, h; t.GetDimensions(w, h); return decant_img_word(k, w, h, 1u, 0u, 9u); }";
    IMG_RW_3D, "decant_img_rw_3d", "uint decant_img_rw_3d(uint slot, uint k) { RWTexture3D<float4> t = ResourceDescriptorHeap[slot]; uint w, h, d; t.GetDimensions(w, h, d); return decant_img_word(k, w, h, d, 0u, 10u); }";
    IMG_RW_1D_ARRAY, "decant_img_rw_1d_array", "uint decant_img_rw_1d_array(uint slot, uint k) { RWTexture1DArray<float4> t = ResourceDescriptorHeap[slot]; uint w, e; t.GetDimensions(w, e); return decant_img_word(k, w, 1u, e, 0u, 12u); }";
    IMG_RW_2D_ARRAY, "decant_img_rw_2d_array", "uint decant_img_rw_2d_array(uint slot, uint k) { RWTexture2DArray<float4> t = ResourceDescriptorHeap[slot]; uint w, h, e; t.GetDimensions(w, h, e); return decant_img_word(k, w, h, e, 0u, 13u); }";
}

/// What kind of image a descriptor is, as far as its size query goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageShape {
    /// The dimension code of the image instructions (0 1D, 1 2D, 2 3D, 3 cube, 4 1D array,
    /// 5 2D array, 6 2D multisample, 7 2D multisample array).
    dim: u8,
    cube_array: bool,
    writable: bool,
}

impl ImageShape {
    /// The helper that makes the descriptor words of such an image.
    fn helper(self) -> Option<&'static Helper> {
        Some(match (self.writable, self.dim, self.cube_array) {
            (false, 3, true) => &IMG_CUBE_ARRAY,
            (false, 0, _) => &IMG_1D,
            (false, 1, _) => &IMG_2D,
            (false, 2, _) => &IMG_3D,
            (false, 3, _) => &IMG_CUBE,
            (false, 4, _) => &IMG_1D_ARRAY,
            (false, 5, _) => &IMG_2D_ARRAY,
            (false, 6, _) => &IMG_2D_MS,
            (false, 7, _) => &IMG_2D_MS_ARRAY,
            (true, 0, _) => &IMG_RW_1D,
            (true, 1, _) => &IMG_RW_2D,
            (true, 2, _) => &IMG_RW_3D,
            (true, 4, _) => &IMG_RW_1D_ARRAY,
            (true, 5, _) => &IMG_RW_2D_ARRAY,
            _ => return None,
        })
    }

    /// From the resource kind of the container (`PSV0`, the DXIL resource kinds).
    fn of_kind(kind: u32, writable: bool) -> Option<ImageShape> {
        let (dim, cube_array) = match kind {
            1 => (0, false),
            2 => (1, false),
            3 => (6, false),
            4 => (2, false),
            5 => (3, false),
            6 => (4, false),
            7 => (5, false),
            8 => (7, false),
            9 => (3, true),
            _ => return None,
        };
        Some(ImageShape {
            dim,
            cube_array,
            writable,
        })
    }
}

// The helper text spells the page size out.
const _: () = assert!(HEAP_PAGE_SHIFT == 12);

/// What a scalar register holds, as far as the pass knows. An unknown value has no entry in the
/// state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Sym {
    Const(u32),
    /// The entry value of a user data register.
    User(u8),
    /// The entry value of a user data register with constant bits or'ed in: the high address
    /// word of a buffer descriptor the program builds around a root descriptor.
    UserOr {
        sgpr: u8,
        bits: u32,
    },
    /// Dword `dword` of the descriptor `desc` (an index into [`Tracer::sources`]).
    Word {
        desc: u32,
        dword: u8,
    },
    /// A value read from one lane of a vector register.
    Lane,
    /// A byte address in the program's own code (what `s_getpc_b64` gives) plus numbers: a
    /// place in the constant tables the compiler puts behind the code.
    Code(Option<u32>),
    /// A finite set of numbers, or byte addresses into the code.
    Choices(u32),
    CodeChoices(u32),
    /// One word of a vector load from the same finite set of code addresses.
    CodePart {
        pc: u32,
        addresses: u32,
        offset: u32,
    },
}

/// Where a descriptor that was loaded from memory comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Source {
    /// The slot whose first byte is `byte` of the table user data register `sgpr` points at.
    Slot { sgpr: u8, byte: u32 },
    /// The same with an offset computed at run time: the load at word `pc` read it at its
    /// register offset plus `byte`.
    Dynamic { sgpr: u8, pc: u32, byte: u32 },
    /// One of several descriptors, chosen by the path taken to the block `at` or by a select.
    Merge { at: u64, of: Vec<u32> },
}

/// The hash of a register name for [`State`]. The dataflow looks registers up millions of
/// times in a large program, and the default hash, which resists crafted keys, is several times
/// slower on keys this small. The keys come from the program, not from an attacker of the
/// translator. A multiply and rotate per field, as in the hash of the Firefox engine.
#[derive(Default, Clone, Copy)]
struct VarHasher(u64);

impl VarHasher {
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for VarHasher {
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.add(u64::from(*b));
        }
    }
    fn write_u8(&mut self, x: u8) {
        self.add(u64::from(x));
    }
    fn write_u16(&mut self, x: u16) {
        self.add(u64::from(x));
    }
    fn write_u32(&mut self, x: u32) {
        self.add(u64::from(x));
    }
    fn write_u64(&mut self, x: u64) {
        self.add(x);
    }
    fn write_usize(&mut self, x: usize) {
        self.add(x as u64);
    }
    fn write_isize(&mut self, x: isize) {
        self.add(x as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type State = HashMap<Var, Sym, BuildHasherDefault<VarHasher>>;

/// Maximum number of values kept for a word during the sampler retry.
const MAX_CHOICES: usize = 8;
/// How often a block is evaluated before values that still change there are given up.
const WIDEN_AFTER: u32 = 12;

fn refuse(what: &'static str, pc: u32, detail: impl Into<String>) -> Refusal {
    Refusal::new(RefusalKind::Resource(what), detail).at(pc)
}

fn fold(op: Bin, a: u32, b: u32) -> Option<u32> {
    Some(match op {
        Bin::Add => a.wrapping_add(b),
        Bin::Sub => a.wrapping_sub(b),
        Bin::Mul => a.wrapping_mul(b),
        Bin::And => a & b,
        Bin::Or => a | b,
        Bin::Xor => a ^ b,
        Bin::Shl => a << (b & 31),
        Bin::LShr => a >> (b & 31),
        Bin::AShr => ((a as i32) >> (b & 31)) as u32,
        _ => return None,
    })
}

/// What the descriptor registers of a use hold.
enum Found {
    Slot {
        sgpr: u8,
        byte: u32,
    },
    /// A descriptor only known at run time: a dynamic slot or a choice.
    Chosen(u32),
    Immediate([u32; 4]),
    /// The buffer behind the root descriptor in user data `sgpr` and `sgpr + 1`.
    RootView {
        sgpr: u8,
    },
}

/// What a scalar load reads.
enum Loaded {
    /// A descriptor word of a table.
    Table(Sym),
    /// Data of the buffer whose descriptor is in the base registers.
    Buffer,
    /// Data of the buffer behind the root descriptor at this user data register.
    Root(u8),
    /// A dword of the program's own code: the low half of the address is a code address.
    Code,
    /// Memory at an address the pass cannot name.
    Raw,
}

struct Tracer<'a> {
    program: &'a StageProgram,
    resources: &'a mut Vec<Resource>,
    sources: Vec<Source>,
    choices: Vec<Vec<u32>>,
    finite_samplers: bool,
    guarded_slots: guarded::Slots,
    ids: HashMap<Source, u32>,
    /// Dwords the scalar load at each word index reads.
    widths: HashMap<u32, u8>,
    /// The loaded descriptors a use takes by a heap index computed at run time, and whether
    /// that index is one of the sampler heap. The load of such a descriptor leaves the index
    /// in its first register instead of nothing.
    tokens: HashMap<u32, bool>,
    /// Descriptors that one use takes as a sampler and another as a resource.
    two_ways: Vec<u32>,
    /// The lanes of a wave of the program: 32 or 64.
    lanes: u8,
    /// Loaded descriptors of eight dwords (images). The others are buffers or samplers.
    wide: HashSet<u32>,
    /// The words of loaded buffer descriptors the program reads as numbers and the pass can
    /// supply: (descriptor, dword). The load of such a dword leaves a made up word behind with
    /// the field the program reads in its place.
    synth: HashSet<(u32, u8)>,
    /// What the image instructions of the program say a loaded image descriptor is.
    shapes: HashMap<u32, ImageShape>,
    /// The loaded descriptor behind each resource that is taken by a heap index, by its place
    /// in the resource list, and the descriptor of the use being interned.
    chosen: HashMap<usize, u32>,
    using: Option<u32>,
    /// The values of one register at the end of each predecessor, kept between calls of
    /// [`Tracer::join`] so a join allocates nothing per register.
    scratch: Vec<Sym>,
}

fn show(sym: Option<Sym>) -> String {
    match sym {
        None => "unknown".to_owned(),
        Some(Sym::Const(_)) => "constant".to_owned(),
        Some(Sym::User(_)) => "user data".to_owned(),
        Some(Sym::UserOr { .. }) => "user data with bits".to_owned(),
        Some(Sym::Word { dword, .. }) => format!("word {dword}"),
        Some(Sym::Lane) => "lane value".to_owned(),
        Some(Sym::Code(_) | Sym::CodeChoices(_)) => "code address".to_owned(),
        Some(Sym::CodePart { .. }) => "code word".to_owned(),
        Some(Sym::Choices(_)) => "finite choice".to_owned(),
    }
}

impl Tracer<'_> {
    /// What the root layout of the container says machine register `sgpr` holds.
    fn layout(&self, sgpr: u8) -> Option<UserDataKind> {
        let dword = self.program.stage.mirror_dword(sgpr);
        self.program
            .user_data
            .iter()
            .find(|u| u.sgpr == dword)
            .map(|u| u.kind)
    }

    fn source(&mut self, s: Source) -> u32 {
        if let Some(id) = self.ids.get(&s) {
            return *id;
        }
        self.sources.push(s.clone());
        let id = self.sources.len() as u32 - 1;
        self.ids.insert(s, id);
        id
    }

    fn choices(&mut self, mut values: Vec<u32>) -> Option<Sym> {
        if !self.finite_samplers {
            return None;
        }
        values.sort_unstable();
        values.dedup();
        match values.as_slice() {
            [] => None,
            [one] => Some(Sym::Const(*one)),
            many if many.len() <= MAX_CHOICES => {
                let id = self
                    .choices
                    .iter()
                    .position(|set| set == many)
                    .unwrap_or_else(|| {
                        self.choices.push(values);
                        self.choices.len() - 1
                    });
                Some(Sym::Choices(id as u32))
            }
            _ => None,
        }
    }

    fn finite_values(&self, sym: Sym) -> Option<Vec<u32>> {
        match sym {
            Sym::Const(v) => Some(vec![v]),
            Sym::Choices(id) => Some(self.choices[id as usize].clone()),
            _ => None,
        }
    }

    fn add_to_code(&mut self, op: Bin, address: Sym, other: Option<Sym>) -> Option<Sym> {
        let result = match (address, other) {
            (
                _,
                Some(
                    Sym::Word { .. }
                    | Sym::Lane
                    | Sym::Code(_)
                    | Sym::CodeChoices(_)
                    | Sym::CodePart { .. },
                ),
            ) => return None,
            (Sym::Code(Some(base)), Some(Sym::Const(v))) => fold(op, base, v).map(Sym::Const),
            (Sym::Code(Some(base)), Some(Sym::Choices(id))) => {
                let values = self.choices[id as usize]
                    .iter()
                    .filter_map(|v| fold(op, base, *v))
                    .collect();
                self.choices(values)
            }
            (Sym::CodeChoices(id), Some(Sym::Const(v))) => {
                let values = self.choices[id as usize]
                    .iter()
                    .filter_map(|base| fold(op, *base, v))
                    .collect();
                self.choices(values)
            }
            _ => None,
        };
        Some(match result {
            Some(Sym::Choices(id)) => Sym::CodeChoices(id),
            Some(Sym::Const(value)) => Sym::Code(Some(value)),
            _ => Sym::Code(None),
        })
    }

    /// The symbolic value of an expression that is no load. `at` names the place, for the
    /// identity of a descriptor a select chooses.
    fn eval(&mut self, state: &State, e: &Expr, at: u64) -> Option<Sym> {
        match e {
            Expr::U32(v) => Some(Sym::Const(*v)),
            Expr::Var(v) => state.get(v).copied(),
            Expr::UserData(k) => Some(Sym::User(*k)),
            Expr::CodeAddress(address) => Some(Sym::Code(self.finite_samplers.then_some(*address))),
            Expr::Un(Un::Not, a) => match self.eval(state, a, at)? {
                Sym::Const(v) => Some(Sym::Const(!v)),
                _ => None,
            },
            Expr::Bin(op, a, b) => {
                let (a, b) = (self.eval(state, a, at), self.eval(state, b, at));
                if matches!(op, Bin::Add | Bin::Sub)
                    && let Some(address @ (Sym::Code(_) | Sym::CodeChoices(_))) = a
                {
                    return self.add_to_code(*op, address, b);
                }
                if *op == Bin::Add
                    && let Some(address @ (Sym::Code(_) | Sym::CodeChoices(_))) = b
                {
                    return self.add_to_code(*op, address, a);
                }
                match (op, a?, b?) {
                    (_, Sym::Const(x), Sym::Const(y)) => fold(*op, x, y).map(Sym::Const),
                    (_, x @ Sym::Choices(_), y @ Sym::Const(_))
                    | (_, x @ Sym::Const(_), y @ Sym::Choices(_)) => {
                        let right = self.finite_values(y)?;
                        let values = self
                            .finite_values(x)?
                            .into_iter()
                            .flat_map(|a| right.iter().filter_map(move |b| fold(*op, a, *b)))
                            .collect();
                        self.choices(values)
                    }
                    (Bin::Or, Sym::User(sgpr), Sym::Const(bits))
                    | (Bin::Or, Sym::Const(bits), Sym::User(sgpr)) => {
                        Some(Sym::UserOr { sgpr, bits })
                    }
                    (Bin::Or, Sym::UserOr { sgpr, bits }, Sym::Const(more))
                    | (Bin::Or, Sym::Const(more), Sym::UserOr { sgpr, bits }) => {
                        Some(Sym::UserOr {
                            sgpr,
                            bits: bits | more,
                        })
                    }
                    // The address half masked to its 16 bits before the stride goes in.
                    (Bin::And, Sym::User(sgpr), Sym::Const(0xffff))
                    | (Bin::And, Sym::Const(0xffff), Sym::User(sgpr)) => {
                        Some(Sym::UserOr { sgpr, bits: 0 })
                    }
                    _ => None,
                }
            }
            Expr::Select(_, a, b) => {
                let (a, b) = (self.eval(state, a, at)?, self.eval(state, b, at)?);
                match (a, b) {
                    _ if a == b => Some(a),
                    (Sym::Code(_) | Sym::CodeChoices(_), Sym::Code(_) | Sym::CodeChoices(_)) => {
                        Some(Sym::Code(None))
                    }
                    (
                        Sym::Word { desc: x, dword },
                        Sym::Word {
                            desc: y,
                            dword: other,
                        },
                    ) if dword == other => Some(Sym::Word {
                        desc: self.source(Source::Merge { at, of: vec![x, y] }),
                        dword,
                    }),
                    _ if self.finite_samplers => {
                        let mut values = self.finite_values(a)?;
                        values.extend(self.finite_values(b)?);
                        self.choices(values)
                    }
                    _ => None,
                }
            }
            Expr::Call(h, _) if h.name.contains("readfirstlane") || h.name.contains("readlane") => {
                Some(Sym::Lane)
            }
            // The compiler builds constants whose set bits are at the top by reversing a small
            // number (the instruction s_brev_b32): some programs make a sampler word that way.
            Expr::Call(h, args) if h.name == "decant_brev" && args.len() == 1 => {
                match self.eval(state, &args[0], at)? {
                    Sym::Const(v) => Some(Sym::Const(v.reverse_bits())),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// An offset expression as a constant part and whether a run time part is added to it.
    fn split(&mut self, state: &State, e: &Expr) -> (bool, u32) {
        if let Some(Sym::Const(v)) = self.eval(state, e, 0) {
            return (false, v);
        }
        match e {
            Expr::Bin(Bin::Add, a, b) => {
                let (da, ca) = self.split(state, a);
                let (db, cb) = self.split(state, b);
                (da || db, ca.wrapping_add(cb))
            }
            _ => (true, 0),
        }
    }

    fn loaded(&mut self, state: &State, load: &Load) -> Loaded {
        if load.buffer {
            return Loaded::Buffer;
        }
        let (lo, hi) = (
            state.get(&Var::S(load.base)).copied(),
            state.get(&Var::S(load.base + 1)).copied(),
        );
        match (lo, hi) {
            (Some(Sym::User(sgpr)), Some(Sym::Const(0))) => {
                let (dynamic, at) = self.split(state, &load.offset);
                // The slot the dword is in: a load of eight dwords or more reads image
                // descriptors, a smaller one a buffer or sampler descriptor.
                let slot = if self.widths.get(&load.pc).copied().unwrap_or(4) >= 8 {
                    32
                } else {
                    16
                };
                let dword = (at % slot) / 4;
                let byte = at - 4 * dword;
                let heap = self.layout(sgpr) == Some(UserDataKind::ResourceHeap);
                let source = if dynamic || heap {
                    Source::Dynamic {
                        sgpr,
                        pc: load.pc,
                        byte,
                    }
                } else {
                    Source::Slot { sgpr, byte }
                };
                let desc = self.source(source);
                if slot == 32 {
                    self.wide.insert(desc);
                }
                Loaded::Table(Sym::Word {
                    desc,
                    dword: dword as u8,
                })
            }
            (Some(Sym::User(sgpr)), Some(Sym::User(next))) if next == sgpr + 1 => {
                Loaded::Root(sgpr)
            }
            // The low half is a code address plus an index. The high half is then the carry
            // and the sign of the same 64 bit sum, which a program that reads its own tables
            // keeps at zero: it is accepted unless it is known to be something else.
            (Some(Sym::Code(_) | Sym::CodeChoices(_)), None | Some(Sym::Const(0))) => Loaded::Code,
            _ => Loaded::Raw,
        }
    }

    /// The value a `Set` gives its destination.
    fn value(&mut self, state: &State, value: &Expr, at: u64) -> Option<Sym> {
        match value {
            Expr::Load(load) => match self.loaded(state, load) {
                Loaded::Table(sym) => Some(sym),
                _ => None,
            },
            Expr::VectorLoad { base, offset, pc } if self.finite_samplers => {
                if !matches!(state.get(&Var::V(*base + 1)), None | Some(Sym::Const(0))) {
                    return None;
                }
                let Sym::Const(offset) = self.eval(state, offset, at)? else {
                    return None;
                };
                match state.get(&Var::V(*base)).copied()? {
                    Sym::Code(Some(address)) => self.code_word(address, offset).map(Sym::Const),
                    Sym::CodeChoices(addresses) => Some(Sym::CodePart {
                        pc: *pc,
                        addresses,
                        offset,
                    }),
                    _ => None,
                }
            }
            other => self.eval(state, other, at),
        }
    }

    fn code_word(&self, base: u32, offset: u32) -> Option<u32> {
        let address = base.checked_add(offset)?;
        if address % 4 != 0 {
            return None;
        }
        self.program.code.get((address / 4) as usize).copied()
    }

    /// Whether `desc` is a buffer descriptor on every path: loaded as four dwords.
    fn narrow(&self, desc: u32) -> bool {
        match &self.sources[desc as usize] {
            Source::Merge { of, .. } => of.iter().all(|d| self.narrow(*d)),
            _ => !self.wide.contains(&desc),
        }
    }

    /// The lowest bit of dword `dword` of `desc` that holds a field the pass can supply, with
    /// every bit above it either such a field or zero in the made up word. `None` when the
    /// dword has no such field.
    ///
    /// A console buffer descriptor holds the record stride in bits 16 to 29 of dword 1 and the
    /// record count in dword 2. An image descriptor holds its size in dwords 1 to 4, see
    /// [`IMG_WORD`]. A program reads them where the source asked for the size of a buffer or a
    /// texture. The address in dword 0 and the low part of dword 1, the format and the swizzle
    /// have no value the pass could put there.
    fn lowest_field(&self, desc: u32, dword: u8) -> Option<u32> {
        match (self.narrow(desc), dword) {
            (true, 1) => Some(16),
            (true, 2) => Some(0),
            (false, 1) if self.all_wide(desc) => Some(30),
            (false, 2 | 4) if self.all_wide(desc) => Some(0),
            (false, 3) if self.all_wide(desc) => Some(12),
            _ => None,
        }
    }

    /// Whether `desc` is an image descriptor on every path.
    fn all_wide(&self, desc: u32) -> bool {
        match &self.sources[desc as usize] {
            Source::Merge { of, .. } => of.iter().all(|d| self.all_wide(*d)),
            _ => self.wide.contains(&desc),
        }
    }

    /// Whether every descriptor word `e` reads as a number is a field the pass can supply.
    /// The fields are pushed to `out`.
    fn data_reads(&self, state: &State, e: &Expr, out: &mut Vec<(u32, u8)>) -> bool {
        let word = |e: &Expr| match e {
            Expr::Var(v) => match state.get(v) {
                Some(Sym::Word { desc, dword }) => Some((*desc, *dword)),
                _ => None,
            },
            _ => None,
        };
        // A read of the bits from `from` up of a word.
        let upper = |e: &Expr, from: u32, out: &mut Vec<(u32, u8)>| match word(e) {
            Some((desc, dword)) => match self.lowest_field(desc, dword) {
                Some(lowest) if from >= lowest => {
                    out.push((desc, dword));
                    Some(true)
                }
                _ => Some(false),
            },
            None => None,
        };
        match e {
            Expr::Bin(Bin::LShr, a, b) => {
                let shift = match &**b {
                    Expr::U32(by) if *by < 32 => Some(*by),
                    _ => None,
                };
                match shift.and_then(|by| upper(a, by, out)) {
                    Some(ok) => ok,
                    None => self.data_reads(state, a, out) & self.data_reads(state, b, out),
                }
            }
            // A bit field: its first bit is the low five bits of the second operand.
            Expr::Call(h, args) if h.name == "decant_bfe_u32" && args.len() == 2 => {
                let first = match &args[1] {
                    Expr::U32(field) => Some(field & 31),
                    _ => None,
                };
                match first.and_then(|from| upper(&args[0], from, out)) {
                    Some(ok) => ok,
                    None => args
                        .iter()
                        .fold(true, |ok, a| ok & self.data_reads(state, a, out)),
                }
            }
            // Two words shifted right as one: the second operand is the low word.
            Expr::Call(h, args) if h.name == "decant_vi_alignbit" && args.len() == 3 => {
                let shift = match &args[2] {
                    Expr::U32(by) => Some(by & 31),
                    _ => None,
                };
                let low = match shift.and_then(|by| upper(&args[1], by, out)) {
                    Some(ok) => ok,
                    None => self.data_reads(state, &args[1], out),
                };
                low & self.data_reads(state, &args[0], out) & self.data_reads(state, &args[2], out)
            }
            Expr::Var(_) => upper(e, 0, out).unwrap_or(true),
            Expr::Un(_, a) => self.data_reads(state, a, out),
            Expr::Bin(Bin::Eq | Bin::Ne, a, b) => {
                if let (Some(wa), Some(wb)) = (word(a), word(b))
                    && wa == wb
                {
                    true
                } else {
                    self.data_reads(state, a, out) & self.data_reads(state, b, out)
                }
            }
            Expr::Bin(Bin::And, a, b) => {
                let const_val = |e: &Expr| match e {
                    Expr::U32(v) => Some(*v),
                    Expr::Var(v) => match state.get(v) {
                        Some(Sym::Const(v)) => Some(*v),
                        _ => None,
                    },
                    _ => None,
                };
                let ca = const_val(a);
                let cb = const_val(b);
                if matches!(ca, Some(0)) || matches!(cb, Some(0)) {
                    true
                } else if let Some(mask) = cb
                    && let Some(ok) = upper(a, mask.trailing_zeros(), out)
                {
                    ok
                } else if let Some(mask) = ca
                    && let Some(ok) = upper(b, mask.trailing_zeros(), out)
                {
                    ok
                } else {
                    self.data_reads(state, a, out) & self.data_reads(state, b, out)
                }
            }
            Expr::Bin(_, a, b) => self.data_reads(state, a, out) & self.data_reads(state, b, out),
            Expr::Select(c, a, b) => {
                self.data_reads(state, c, out)
                    & self.data_reads(state, a, out)
                    & self.data_reads(state, b, out)
            }
            Expr::Call(_, args) => args
                .iter()
                .fold(true, |ok, a| ok & self.data_reads(state, a, out)),
            Expr::BufferWord { offset, .. } => self.data_reads(state, offset, out),
            _ => true,
        }
    }

    /// What kind of image the loaded descriptor `desc` is: by the image instructions that use
    /// it, else by the resource the container declares for its table slot.
    fn shape_of_image(&self, desc: u32) -> Option<ImageShape> {
        if let Some(shape) = self.shapes.get(&desc) {
            return Some(*shape);
        }
        let Source::Slot { sgpr, byte } = &self.sources[desc as usize] else {
            return self.heap_image_shape();
        };
        let dword = self.program.stage.mirror_dword(*sgpr);
        let record = self
            .program
            .binding_table
            .iter()
            .find(|b| b.sgpr == dword && b.word_offset * 4 == *byte)?;
        let declared =
            self.program.resources.iter().find(|r| {
                r.class == record.class && (r.lower..=r.upper).contains(&record.register)
            })?;
        ImageShape::of_kind(
            declared.psv_kind,
            record.class == crate::api::RegisterClass::Uav,
        )
    }

    /// What kind of image a descriptor taken from the whole heap is when no image instruction
    /// uses it (a program that only asks for its size): the kind of the textures the container
    /// declares outside its tables, when they are all of one kind. A table slot has a binding
    /// record, so a declared texture without one is reached through the heap.
    fn heap_image_shape(&self) -> Option<ImageShape> {
        let p = self.program;
        let mut kinds =
            p.resources
                .iter()
                .filter(|r| {
                    r.class == crate::api::RegisterClass::Srv
                        && r.psv_type == 3
                        && (1..=9).contains(&r.psv_kind)
                        && !p.binding_table.iter().any(|b| {
                            b.class == r.class && (r.lower..=r.upper).contains(&b.register)
                        })
                })
                .map(|r| r.psv_kind);
        let first = kinds.next()?;
        kinds
            .all(|k| k == first)
            .then(|| ImageShape::of_kind(first, false))?
    }

    /// Notes what an image instruction says about the descriptor in the registers from `first`.
    fn note_image(&mut self, state: &State, first: u8, dim: u8, writable: bool) {
        fn leaves(t: &Tracer, desc: u32, out: &mut Vec<u32>) {
            match &t.sources[desc as usize] {
                Source::Merge { of, .. } => of.iter().for_each(|d| leaves(t, *d, out)),
                _ => out.push(desc),
            }
        }
        if let Some(Sym::Word { desc, dword: 0 }) = state.get(&Var::S(first)).copied() {
            let mut all = Vec::new();
            leaves(self, desc, &mut all);
            for leaf in all {
                let shape = self.shapes.entry(leaf).or_insert(ImageShape {
                    dim,
                    cube_array: false,
                    writable,
                });
                shape.writable |= writable;
            }
        }
    }

    /// Notes that a statement reads dword `dword` of the loaded descriptors behind `desc`.
    fn mark_synth(&mut self, desc: u32, dword: u8) {
        match self.sources[desc as usize].clone() {
            Source::Merge { of, .. } => of.into_iter().for_each(|d| self.mark_synth(d, dword)),
            _ => {
                self.synth.insert((desc, dword));
            }
        }
    }

    /// The host heap index of a loaded descriptor at its load, whose byte offset expression is
    /// `offset` (the offset of any of its dwords: a slot is 32 bytes).
    fn host_of(&self, desc: u32, offset: &Expr, pc: u32) -> Result<Expr, Refusal> {
        match &self.sources[desc as usize] {
            Source::Slot { sgpr, byte } => Ok(Expr::bin(
                Bin::Add,
                Expr::UserData(*sgpr),
                Expr::U32(byte / 32),
            )),
            Source::Dynamic { sgpr, .. }
                if self.layout(*sgpr) == Some(UserDataKind::ResourceHeap) =>
            {
                let index = Expr::bin(Bin::LShr, offset.clone(), Expr::U32(5));
                Ok(Expr::Call(&HEAP_HOST, vec![index]))
            }
            _ => Err(refuse(
                "descriptor-as-data",
                pc,
                "a field of a descriptor without a host slot",
            )),
        }
    }

    /// The expressions of a statement that compute with numbers (not the descriptor operands,
    /// and not a load, which has its own rules).
    fn numbers(s: &Stmt) -> Vec<&Expr> {
        match s {
            Stmt::Set { value, .. } if !matches!(value, Expr::Load(_)) => vec![value],
            Stmt::Effect { call, .. } => vec![call],
            Stmt::Output { value, .. } => vec![value],
            Stmt::Discard { cond } => vec![cond],
            Stmt::Image(op, _) => op.address.iter().collect(),
            Stmt::Buffer(op, _) => std::iter::once(&op.offset).chain(&op.index).collect(),
            _ => Vec::new(),
        }
    }

    /// The descriptor fields a statement reads as numbers, see [`Tracer::synth`]. `Err` when it
    /// reads a word the pass cannot supply.
    fn fields(&mut self, state: &State, s: &Stmt) -> Result<Vec<(u32, u8)>, Refusal> {
        let mut out = Vec::new();
        for e in Self::numbers(s) {
            // A move or a select of whole words is no read of their value.
            if matches!(s, Stmt::Set { .. })
                && matches!(self.eval(state, e, 0), Some(Sym::Word { .. }))
            {
                continue;
            }
            if !self.data_reads(state, e, &mut out) {
                let mut word = None;
                super::reads(e, &mut |v| {
                    if let Some(Sym::Word { dword, desc }) = state.get(&v) {
                        word.get_or_insert((*dword, self.narrow(*desc)));
                    }
                });
                let (dword, narrow) = word.unwrap_or((0, true));
                return Err(refuse(
                    "descriptor-as-data",
                    0,
                    format!(
                        "word {dword} of {} in {}",
                        if narrow { "a buffer" } else { "an image" },
                        shape_of(e)
                    ),
                ));
            }
        }
        Ok(out)
    }

    /// The uses of one statement that take a descriptor by a heap index, see [`Tracer::tokens`].
    fn scan(&mut self, state: &State, s: &Stmt) {
        if let Ok(fields) = self.fields(state, s) {
            fields
                .into_iter()
                .for_each(|(desc, dword)| self.mark_synth(desc, dword));
        }
        let note = |t: &mut Self, d: &Desc, words: u8, sampler: bool| {
            if let Desc::Regs(first) = *d
                && let Ok(Found::Chosen(desc)) = t.found(state, first, words, 0)
            {
                t.mark(desc, sampler);
            }
        };
        match s {
            Stmt::Set {
                value: Expr::Load(load),
                ..
            } if load.buffer => note(self, &Desc::Regs(load.base), 4, false),
            Stmt::Set {
                value: Expr::BvhWord(op, _),
                ..
            } => {
                if let Some(first) = descriptor_registers(op) {
                    note(self, &Desc::Regs(first), 4, false);
                }
            }
            Stmt::Image(op, _) => {
                if let Desc::Regs(first) = op.image {
                    let writable =
                        matches!(op.kind, ImageKind::Store { .. } | ImageKind::Atomic { .. });
                    self.note_image(state, first, op.dim, writable);
                }
                note(self, &op.image, 8, false);
                if let Some(sampler) = &op.sampler {
                    note(self, sampler, 4, true);
                }
            }
            Stmt::Buffer(op, _) => note(self, &op.buffer, 4, false),
            _ => {}
        }
    }

    /// One statement in the analysis: only assignments change the state.
    fn step(&mut self, state: &mut State, s: &Stmt, block: usize, epoch: &mut u64) {
        match s {
            Stmt::Set { dst, value, .. } => {
                if *dst == Var::Scc {
                    *epoch += 1;
                }
                match self.value(state, value, select_key(block, *epoch)) {
                    Some(sym) => state.insert(*dst, sym),
                    None => state.remove(dst),
                };
                forget_pair(state, *dst, self.lanes);
            }
            Stmt::Effect { dst: Some(dst), .. } => {
                state.remove(dst);
            }
            Stmt::Image(op, _) => {
                for r in op.data_writes() {
                    state.remove(&Var::V(r));
                }
            }
            Stmt::Buffer(op, _) if !op.store || (op.glc && op.op.name().contains("atomic")) => {
                for r in op.data..op.data.saturating_add(op.count) {
                    state.remove(&Var::V(r));
                }
            }
            _ => {}
        }
    }

    /// The state at the entry of `block` from the states its predecessors leave.
    fn join(
        &mut self,
        block: usize,
        preds: &[usize],
        exit: &[Option<State>],
        widen: bool,
    ) -> State {
        let mut known = preds.iter().filter_map(|p| exit[*p].as_ref());
        let Some(first) = known.next() else {
            return State::default();
        };
        let rest = known;
        let mut out = State::with_capacity_and_hasher(first.len(), Default::default());
        let mut all = std::mem::take(&mut self.scratch);
        for (var, sym) in first {
            all.clear();
            all.push(*sym);
            for other in rest.clone() {
                match other.get(var) {
                    Some(s) => all.push(*s),
                    None => {
                        all.clear();
                        break;
                    }
                }
            }
            let Some(head) = all.first().copied() else {
                continue;
            };
            if all.iter().all(|s| *s == head) {
                out.insert(*var, head);
                continue;
            }
            if all
                .iter()
                .all(|s| matches!(s, Sym::Code(_) | Sym::CodeChoices(_)))
            {
                out.insert(*var, Sym::Code(None));
                continue;
            }
            if widen {
                continue;
            }
            if self.finite_samplers
                && let Some(values) = all
                    .iter()
                    .map(|s| self.finite_values(*s))
                    .collect::<Option<Vec<_>>>()
                && let Some(choice) = self.choices(values.into_iter().flatten().collect())
            {
                out.insert(*var, choice);
                continue;
            }
            // Words of different descriptors with the same position: the descriptor the path chose.
            let words: Option<Vec<u32>> = all
                .iter()
                .map(|s| match (s, head) {
                    (Sym::Word { desc, dword }, Sym::Word { dword: want, .. })
                        if *dword == want =>
                    {
                        Some(*desc)
                    }
                    _ => None,
                })
                .collect();
            if let (Some(of), Sym::Word { dword, .. }) = (words, head) {
                let desc = self.source(Source::Merge {
                    at: block as u64,
                    of,
                });
                out.insert(*var, Sym::Word { desc, dword });
            }
        }
        self.scratch = all;
        out
    }

    fn intern(&mut self, origin: Origin, usage: ResourceUse, pc: u32) -> Result<ResId, Refusal> {
        let by_index = matches!(origin, Origin::HeapIndex { .. });
        // What the use takes by a heap index: one resource per loaded descriptor, because the
        // same register holds a typed buffer at one place and a raw one at another, and the
        // printer decides the view of a buffer per resource.
        let chosen = self.using.take();
        for (i, r) in self.resources.iter_mut().enumerate() {
            if r.origin != origin {
                continue;
            }
            if by_index && self.chosen.get(&i) != chosen.as_ref() {
                continue;
            }
            // A register can hold the index of a buffer at one use and of an image at another:
            // those are two resources.
            if by_index && std::mem::discriminant(&r.usage) != std::mem::discriminant(&usage) {
                continue;
            }
            if let (true, ResourceUse::Image { dim: a, .. }, ResourceUse::Image { dim: b, .. }) =
                (by_index, r.usage, usage)
                && a != b
            {
                continue;
            }
            // One descriptor used twice: the uses must agree on what it is.
            r.usage = match (r.usage, usage) {
                (ResourceUse::Words { written: a }, ResourceUse::Words { written: b }) => {
                    ResourceUse::Words { written: a || b }
                }
                (
                    ResourceUse::Image {
                        dim,
                        written: a,
                        compare: c,
                    },
                    ResourceUse::Image {
                        dim: d2,
                        written: b,
                        compare: c2,
                    },
                ) if dim == d2 => ResourceUse::Image {
                    dim,
                    written: a || b,
                    compare: c || c2,
                },
                (ResourceUse::Sampler { compare: a }, ResourceUse::Sampler { compare: b })
                    if a == b =>
                {
                    ResourceUse::Sampler { compare: a }
                }
                _ => return Err(refuse("descriptor-used-two-ways", pc, "")),
            };
            return Ok(ResId(i as u32));
        }
        self.resources.push(Resource {
            origin,
            usage,
            declared: None,
            placement: None,
            name: String::new(),
        });
        if let (true, Some(desc)) = (by_index, chosen) {
            self.chosen.insert(self.resources.len() - 1, desc);
        }
        Ok(ResId(self.resources.len() as u32 - 1))
    }

    /// What the `words` registers from `first` hold together.
    fn found(&self, state: &State, first: u8, words: u8, pc: u32) -> Result<Found, Refusal> {
        let syms: Vec<Option<Sym>> = (0..words)
            .map(|k| state.get(&Var::S(first + k)).copied())
            .collect();
        let shape = || syms.iter().map(|s| show(*s)).collect::<Vec<_>>().join(", ");
        match syms[0] {
            Some(Sym::Word { desc, dword: 0 }) => {
                let whole = syms.iter().enumerate().all(|(k, s)| {
                    *s == Some(Sym::Word {
                        desc,
                        dword: k as u8,
                    })
                });
                if !whole {
                    return Err(refuse("partial-descriptor", pc, shape()));
                }
                Ok(match &self.sources[desc as usize] {
                    Source::Slot { sgpr, byte } => Found::Slot {
                        sgpr: *sgpr,
                        byte: *byte,
                    },
                    _ => Found::Chosen(desc),
                })
            }
            Some(Sym::Const(_)) if words == 4 => {
                let mut out = [0u32; 4];
                for (slot, s) in out.iter_mut().zip(&syms) {
                    let Some(Sym::Const(v)) = s else {
                        return Err(refuse("partial-descriptor", pc, shape()));
                    };
                    *slot = *v;
                }
                Ok(Found::Immediate(out))
            }
            Some(Sym::User(sgpr)) if words == 4 => match (syms[1], syms[2], syms[3]) {
                (
                    Some(Sym::UserOr { sgpr: hi, .. } | Sym::User(hi)),
                    Some(Sym::Const(_)),
                    Some(Sym::Const(_)),
                ) if hi == sgpr + 1 => Ok(Found::RootView { sgpr }),
                _ => Err(refuse("partial-descriptor", pc, shape())),
            },
            Some(Sym::Word { .. }) => Err(refuse("partial-descriptor", pc, shape())),
            _ if syms.contains(&Some(Sym::Lane)) => Err(refuse("per-lane-descriptor", pc, shape())),
            _ => Err(refuse("untraced-descriptor", pc, shape())),
        }
    }

    /// Notes that a use takes the loaded descriptors behind `desc` by a heap index.
    fn mark(&mut self, desc: u32, sampler: bool) {
        match self.sources[desc as usize].clone() {
            Source::Merge { of, .. } => of.into_iter().for_each(|d| self.mark(d, sampler)),
            _ => {
                if *self.tokens.entry(desc).or_insert(sampler) != sampler {
                    self.two_ways.push(desc);
                }
            }
        }
    }

    /// Whether `desc` or a descriptor it is chosen from is used both as a sampler and not.
    fn used_two_ways(&self, desc: u32) -> bool {
        match &self.sources[desc as usize] {
            Source::Merge { of, .. } => of.iter().any(|d| self.used_two_ways(*d)),
            _ => self.two_ways.contains(&desc),
        }
    }

    /// The heap index the load of dword 0 of `desc` leaves in its register. `offset` is the
    /// byte offset expression of that load.
    fn token(&self, desc: u32, sampler: bool, offset: &Expr, pc: u32) -> Result<Expr, Refusal> {
        let (stride, shift) = if sampler { (16, 4) } else { (32, 5) };
        match &self.sources[desc as usize] {
            Source::Slot { sgpr, byte } => {
                if byte % stride != 0 {
                    return Err(refuse(
                        "table-offset",
                        pc,
                        format!("table s{sgpr} offset {byte:#x}"),
                    ));
                }
                // The mirror holds the host index of the table's first slot.
                Ok(Expr::bin(
                    Bin::Add,
                    Expr::UserData(*sgpr),
                    Expr::U32(byte / stride),
                ))
            }
            Source::Dynamic { sgpr, .. } => {
                let index = Expr::bin(Bin::LShr, offset.clone(), Expr::U32(shift));
                match (self.layout(*sgpr), sampler) {
                    // A console index into the whole view heap: the heap map gives the host index.
                    (Some(UserDataKind::ResourceHeap), false) => {
                        Ok(Expr::Call(&HEAP_HOST, vec![index]))
                    }
                    // Sampler heaps are small and keep their indices.
                    (Some(UserDataKind::SamplerHeap), true) => {
                        Ok(Expr::bin(Bin::Add, Expr::UserData(*sgpr), index))
                    }
                    (kind, _) => Err(refuse(
                        "dynamic-table-offset",
                        pc,
                        match kind {
                            Some(UserDataKind::ResourceHeap) => "a sampler in the resource heap",
                            Some(UserDataKind::SamplerHeap) => "a resource in the sampler heap",
                            Some(_) => "the base is no heap register",
                            None => "the container has no root layout for the base",
                        },
                    )),
                }
            }
            Source::Merge { .. } => Err(refuse("descriptor-by-branch", pc, "a load of a choice")),
        }
    }

    /// Complete descriptor rows. Keep the four words of each table entry together.
    fn sampler_choices(&self, state: &State, first: u8) -> Option<Vec<([u32; 4], u32)>> {
        let mut syms = [Sym::Const(0); 4];
        for (i, sym) in syms.iter_mut().enumerate() {
            *sym = *state.get(&Var::S(first + i as u8))?;
        }
        let mut rows = Vec::new();
        if let Sym::CodePart {
            pc,
            addresses,
            offset,
        } = syms[0]
        {
            for (i, sym) in syms.into_iter().enumerate() {
                if sym
                    != (Sym::CodePart {
                        pc,
                        addresses,
                        offset: offset + 4 * i as u32,
                    })
                {
                    return None;
                }
            }
            for &address in &self.choices[addresses as usize] {
                let mut words = [0; 4];
                for (i, word) in words.iter_mut().enumerate() {
                    *word = self.code_word(address, offset + 4 * i as u32)?;
                }
                rows.push((words, 0));
            }
        } else {
            let mut words = [0; 4];
            let mut varying = None;
            for (i, sym) in syms.into_iter().enumerate() {
                match sym {
                    Sym::Const(value) => words[i] = value,
                    Sym::Choices(id) if varying.is_none() => varying = Some((i, id)),
                    _ => return None,
                }
            }
            let (i, id) = varying?;
            for &value in &self.choices[id as usize] {
                words[i] = value;
                rows.push((words, 0));
            }
        }
        rows.sort_unstable();
        rows.dedup();
        // The image printer uses one coordinate convention for the entire operation.
        let unnormalized = rows.first()?.0[0] & 0x8000;
        rows.iter()
            .all(|(words, _)| words[0] & 0x8000 == unnormalized)
            .then_some(rows)
    }

    /// The resource of a descriptor operand.
    fn resource(
        &mut self,
        state: &State,
        first: u8,
        words: u8,
        usage: ResourceUse,
        pc: u32,
    ) -> Result<ResId, Refusal> {
        let sampler = matches!(usage, ResourceUse::Sampler { .. });
        let buffer = matches!(usage, ResourceUse::Words { .. });
        let found = self.found(state, first, words, pc);
        if sampler
            && found.is_err()
            && let Some(choices) = self.sampler_choices(state, first)
        {
            return self.intern(Origin::SamplerChoices { first, choices }, usage, pc);
        }
        if sampler
            && found.is_err()
            && let Some(&guarded) = self.guarded_slots.get(&(pc, first))
        {
            let origin = match guarded {
                guarded::GuardedSampler::Table { sgpr, offset } => Origin::Table { sgpr, offset },
                guarded::GuardedSampler::Immediate(words) => Origin::Immediate { words },
            };
            return self.intern(origin, usage, pc);
        }
        let origin = match found? {
            Found::Slot { sgpr, byte } => Origin::Table { sgpr, offset: byte },
            Found::Chosen(desc) => {
                if self.used_two_ways(desc) {
                    return Err(refuse("descriptor-used-two-ways", pc, ""));
                }
                // The first register of the descriptor holds the heap index at this point.
                self.using = Some(desc);
                Origin::HeapIndex {
                    index: Var::S(first),
                    sampler,
                }
            }
            Found::Immediate(words) if sampler => Origin::Immediate { words },
            Found::Immediate(_) => {
                return Err(refuse(
                    "immediate-descriptor",
                    pc,
                    "a buffer or image descriptor made of constants",
                ));
            }
            Found::RootView { sgpr } if buffer => Origin::RootDescriptor { sgpr },
            Found::RootView { .. } => {
                return Err(refuse(
                    "untraced-descriptor",
                    pc,
                    "a root descriptor address used as an image or sampler",
                ));
            }
        };
        self.intern(origin, usage, pc)
    }

    fn desc(
        &mut self,
        state: &State,
        d: &mut Desc,
        words: u8,
        usage: ResourceUse,
        pc: u32,
    ) -> Result<(), Refusal> {
        if let Desc::Regs(first) = *d {
            *d = Desc::Res(self.resource(state, first, words, usage, pc)?);
        }
        Ok(())
    }

    /// Replaces the four descriptor registers of a node intersection by the buffer resource
    /// they describe, see [`node_memory`].
    fn bind_node_memory(&mut self, state: &State, op: &mut BvhIntersect) -> Result<(), Refusal> {
        if node_memory(op).is_some() {
            return Ok(());
        }
        let first = descriptor_registers(op).ok_or_else(|| {
            refuse(
                "bvh-descriptor",
                op.pc,
                "descriptor words outside four consecutive scalar registers",
            )
        })?;
        let res = self.resource(
            state,
            first,
            4,
            ResourceUse::Words { written: false },
            op.pc,
        )?;
        op.descriptor = vec![Expr::BufferWord {
            res,
            offset: Box::new(Expr::U32(0)),
        }];
        Ok(())
    }

    /// Whether an expression reads a register that holds a descriptor word.
    fn reads_word(state: &State, e: &Expr) -> bool {
        let mut found = false;
        super::reads(e, &mut |v| {
            found |= matches!(state.get(&v), Some(Sym::Word { .. }))
        });
        found
    }

    /// One statement in the rewrite: descriptor operands become resources, loads become
    /// buffer reads or vanish, and what cannot be carried is refused.
    fn rewrite(
        &mut self,
        state: &mut State,
        s: &mut Stmt,
        block: usize,
        epoch: &mut u64,
    ) -> Result<(), Refusal> {
        // Descriptor words read as numbers: only the fields the loads supply.
        self.fields(state, s)?;
        match s {
            Stmt::Set { dst, value, .. } => {
                if *dst == Var::Scc {
                    *epoch += 1;
                }
                let at = select_key(block, *epoch);
                // A global load: the address is in a vector register pair of the lane. So far
                // the title only reads the tables behind its own code that way, with the lane's
                // index added to the code address. The high half is the carry of that sum.
                let mut code_part = None;
                if let Expr::VectorLoad { .. } = value {
                    code_part = self.value(state, value, at);
                }
                if let Expr::VectorLoad { base, offset, pc } = value {
                    let lo = state.get(&Var::V(*base)).copied();
                    let hi = state.get(&Var::V(*base + 1)).copied();
                    if !matches!(lo, Some(Sym::Code(_) | Sym::CodeChoices(_)))
                        || !matches!(hi, None | Some(Sym::Const(0)))
                    {
                        return Err(refuse(
                            "raw-address",
                            *pc,
                            format!("a global load at ({}, {})", show(lo), show(hi)),
                        ));
                    }
                    *value = Expr::CodeWord(Box::new(Expr::bin(
                        Bin::Add,
                        Expr::Var(Var::V(*base)),
                        (**offset).clone(),
                    )));
                }
                if let Expr::BvhWord(op, _) = value {
                    self.bind_node_memory(state, op)?;
                }
                let sym = if let Expr::Load(load) = value {
                    let (load, pc) = (load.clone(), load.pc);
                    if Self::reads_word(state, &load.offset) {
                        return Err(refuse("descriptor-as-data", pc, "in a load offset"));
                    }
                    match self.loaded(state, &load) {
                        Loaded::Table(sym) => {
                            // The words are never used as numbers. A use takes the descriptor
                            // by its origin, or by the heap index that the load of its first
                            // dword leaves in the register.
                            *value = match sym {
                                Sym::Word { desc, dword: 0 } => match self.tokens.get(&desc) {
                                    Some(sampler) => {
                                        self.token(desc, *sampler, &load.offset, pc)?
                                    }
                                    None => Expr::U32(0),
                                },
                                // A field the program reads as a number: the stride in its
                                // place in dword 1, the record count as dword 2.
                                Sym::Word { desc, dword }
                                    if self.synth.contains(&(desc, dword)) =>
                                {
                                    let host = self.host_of(desc, &load.offset, pc)?;
                                    if self.wide.contains(&desc) {
                                        // The kind of the image decides the query. When
                                        // nothing in the program or the container says it
                                        // (a texture taken from the whole heap that is only
                                        // asked for its size), the view shadow row has the size.
                                        let helper = self
                                            .shape_of_image(desc)
                                            .and_then(ImageShape::helper)
                                            .unwrap_or(&IMG_ROW);
                                        Expr::Call(helper, vec![host, Expr::U32(u32::from(dword))])
                                    } else if dword == 1 {
                                        Expr::bin(
                                            Bin::Shl,
                                            Expr::Call(&SHADOW_STRIDE, vec![host]),
                                            Expr::U32(16),
                                        )
                                    } else {
                                        Expr::Call(&SHADOW_COUNT, vec![host])
                                    }
                                }
                                _ => Expr::U32(0),
                            };
                            Some(sym)
                        }
                        Loaded::Buffer => {
                            let res = self.resource(
                                state,
                                load.base,
                                4,
                                ResourceUse::Words { written: false },
                                pc,
                            )?;
                            *value = Expr::BufferWord {
                                res,
                                offset: load.offset,
                            };
                            None
                        }
                        Loaded::Root(sgpr) => {
                            let res = self.intern(
                                Origin::RootDescriptor { sgpr },
                                ResourceUse::Words { written: false },
                                pc,
                            )?;
                            *value = Expr::BufferWord {
                                res,
                                offset: load.offset,
                            };
                            None
                        }
                        Loaded::Code => {
                            // The register holds the byte address in the code at this point.
                            *value = Expr::CodeWord(Box::new(Expr::bin(
                                Bin::Add,
                                Expr::Var(Var::S(load.base)),
                                (*load.offset).clone(),
                            )));
                            None
                        }
                        Loaded::Raw => {
                            let lo = state.get(&Var::S(load.base)).copied();
                            let hi = state.get(&Var::S(load.base + 1)).copied();
                            return Err(refuse(
                                "raw-address",
                                pc,
                                format!("base ({}, {})", show(lo), show(hi)),
                            ));
                        }
                    }
                } else {
                    code_part.or_else(|| self.eval(state, value, at))
                };
                match sym {
                    Some(sym) => state.insert(*dst, sym),
                    None => state.remove(dst),
                };
                forget_pair(state, *dst, self.lanes);
            }
            Stmt::Image(op, _) => {
                let (written, compare) = match op.kind {
                    ImageKind::Store { .. } | ImageKind::Atomic { .. } => (true, false),
                    ImageKind::Sample { compare, .. } => (false, compare),
                    _ => (false, false),
                };
                let pc = op.pc;
                let usage = ResourceUse::Image {
                    dim: op.dim,
                    written,
                    compare,
                };
                self.desc(state, &mut op.image, 8, usage, pc)?;
                if let Some(sampler) = &mut op.sampler {
                    self.desc(state, sampler, 4, ResourceUse::Sampler { compare }, pc)?;
                }
                for r in op.data_writes() {
                    state.remove(&Var::V(r));
                }
            }
            Stmt::Buffer(op, _) => {
                let pc = op.pc;
                let usage = ResourceUse::Words { written: op.store };
                self.desc(state, &mut op.buffer, 4, usage, pc)?;
                if !op.store || (op.glc && op.op.name().contains("atomic")) {
                    for r in op.data..op.data.saturating_add(op.count) {
                        state.remove(&Var::V(r));
                    }
                }
            }
            Stmt::Effect { dst, .. } => {
                // What the call gives back is a number the pass knows nothing about.
                if let Some(dst) = dst {
                    state.remove(dst);
                }
            }
            Stmt::Export(_)
            | Stmt::Output { .. }
            | Stmt::Discard { .. }
            | Stmt::Message { .. }
            | Stmt::Barrier { .. }
            | Stmt::Comment(_) => {}
        }
        Ok(())
    }
}

/// The first of the four consecutive scalar registers a node intersection takes its
/// descriptor from, before the pass binds it.
fn descriptor_registers(op: &BvhIntersect) -> Option<u8> {
    let Some(Expr::Var(Var::S(first))) = op.descriptor.first() else {
        return None;
    };
    let consecutive = op.descriptor.len() == 4
        && op.descriptor.iter().enumerate().all(|(k, word)| {
            u8::try_from(k)
                .ok()
                .and_then(|k| first.checked_add(k))
                .is_some_and(|reg| *word == Expr::Var(Var::S(reg)))
        });
    consecutive.then_some(*first)
}

/// The node memory of a node intersection after this pass: the buffer resource its
/// descriptor describes. The pass replaces the four descriptor words by one
/// [`Expr::BufferWord`] of that resource at offset 0. It marks the resource and is never
/// printed as a read. The node pointer in the address operands counts from the start of that
/// buffer.
pub fn node_memory(op: &BvhIntersect) -> Option<ResId> {
    match op.descriptor.as_slice() {
        [Expr::BufferWord { res, .. }] => Some(*res),
        _ => None,
    }
}

/// A lane mask written to a scalar register pair replaces what the two registers held.
/// In a program of 32 lanes a mask is one register.
fn forget_pair(state: &mut State, dst: Var, lanes: u8) {
    if let Var::Mask(first) = dst {
        state.remove(&Var::S(first));
        if lanes != 32 {
            state.remove(&Var::S(first + 1));
        }
    }
}

/// What kind of expression a value is, for the detail of a refusal.
fn shape_of(e: &Expr) -> String {
    match e {
        Expr::Bin(op, ..) => format!("{op:?}"),
        Expr::Un(op, _) => format!("{op:?}"),
        Expr::Call(h, _) => h.name.to_owned(),
        Expr::Select(..) => "select".to_owned(),
        _ => "expression".to_owned(),
    }
}

/// The identity of the descriptor a select chooses: all selects between two tests of the same
/// block choose together.
fn select_key(block: usize, epoch: u64) -> u64 {
    (1 << 63) | ((block as u64) << 24) | epoch
}

/// Replaces every raw descriptor use in `func` by a resource and fills `func.resources`. The
/// program gives the stage, the tables of the container and the code words.
pub fn trace(func: &mut Function, program: &StageProgram) -> Result<(), Refusal> {
    let mut body = std::mem::take(&mut func.body);
    let result = (|| {
        if run(
            &mut body,
            &mut func.resources,
            program,
            func.lanes,
            false,
            None,
        )? == Retry::Finite
            && run(
                &mut body,
                &mut func.resources,
                program,
                func.lanes,
                true,
                None,
            )? == Retry::Guarded
        {
            let slots = guarded::sampler_slots(&mut body, program, func.lanes);
            run(
                &mut body,
                &mut func.resources,
                program,
                func.lanes,
                true,
                Some(slots),
            )?;
        }
        Ok(())
    })();
    func.body = body;
    result
}

#[derive(PartialEq, Eq)]
enum Retry {
    Done,
    Finite,
    Guarded,
}

impl<'a> Tracer<'a> {
    fn new(
        program: &'a StageProgram,
        resources: &'a mut Vec<Resource>,
        lanes: u8,
        finite_samplers: bool,
        flow: &Flow<'_>,
    ) -> Self {
        let mut tracer = Self {
            program,
            resources,
            sources: Vec::new(),
            choices: Vec::new(),
            finite_samplers,
            guarded_slots: HashMap::new(),
            ids: HashMap::new(),
            widths: HashMap::new(),
            tokens: HashMap::new(),
            two_ways: Vec::new(),
            lanes,
            wide: HashSet::new(),
            synth: HashSet::new(),
            shapes: HashMap::new(),
            chosen: HashMap::new(),
            using: None,
            scratch: Vec::new(),
        };
        for block in &flow.blocks {
            for s in block.stmts.iter().flat_map(|s| s.iter()) {
                if let Stmt::Set {
                    value: Expr::Load(load),
                    ..
                } = s
                {
                    *tracer.widths.entry(load.pc).or_default() += 1;
                }
            }
        }
        tracer
    }
}

fn run(
    body: &mut [crate::ir::Node],
    resources: &mut Vec<Resource>,
    program: &StageProgram,
    lanes: u8,
    finite_samplers: bool,
    guarded_slots: Option<guarded::Slots>,
) -> Result<Retry, Refusal> {
    let mut flow = Flow::build(body, false);
    let mut tracer = Tracer::new(program, resources, lanes, finite_samplers, &flow);
    let check_guards = guarded_slots.is_none();
    if let Some(slots) = guarded_slots {
        tracer.guarded_slots = slots;
    }
    // The states to a fixed point.
    let preds = flow.predecessors();
    let count = flow.blocks.len();
    let mut entry: Vec<Option<State>> = vec![None; count];
    let mut exit: Vec<Option<State>> = vec![None; count];
    let mut visits = vec![0u32; count];
    let mut queued = vec![false; count];
    // Evaluate the lowest block first: Flow numbers blocks in program order, so forward
    // predecessors precede join blocks and each join runs once instead of once per path.
    let mut work = BinaryHeap::from([Reverse(0usize)]);
    while let Some(Reverse(b)) = work.pop() {
        queued[b] = false;
        visits[b] += 1;
        if visits[b] > 4 * WIDEN_AFTER {
            return Err(refuse("flow-budget", 0, ""));
        }
        let mut state = tracer.join(b, &preds[b], &exit, visits[b] > WIDEN_AFTER);
        entry[b] = Some(state.clone());
        let mut epoch = 0;
        for s in flow.blocks[b].stmts.iter().flat_map(|s| s.iter()) {
            tracer.step(&mut state, s, b, &mut epoch);
        }
        // A temporary lives inside the instruction that made it, so it is dropped where the
        // block ends. That keeps the states the size of the register file.
        state.retain(|v, _| !matches!(v, Var::Tmp(_) | Var::Flag(_)));
        if exit[b].as_ref() != Some(&state) {
            exit[b] = Some(state);
            for &s in &flow.blocks[b].succs {
                if !queued[s] {
                    queued[s] = true;
                    work.push(Reverse(s));
                }
            }
        }
    }
    // Which loads have to leave a heap index behind: the uses say.
    for (b, block) in flow.blocks.iter().enumerate() {
        let (Some(stmts), Some(mut state)) = (block.stmts.as_ref(), entry[b].clone()) else {
            continue;
        };
        let mut epoch = 0;
        for s in stmts.iter() {
            // Numeric alternatives can change how offsets are classified elsewhere in the
            // program. Only spend that precision on stages with an unresolved sampler.
            if check_guards
                && let Stmt::Image(op, _) = s
                && let Some(Desc::Regs(first)) = op.sampler
                && tracer.found(&state, first, 4, op.pc).is_err()
            {
                if !finite_samplers {
                    return Ok(Retry::Finite);
                }
                if tracer.sampler_choices(&state, first).is_none() {
                    return Ok(Retry::Guarded);
                }
            }
            tracer.scan(&state, s);
            tracer.step(&mut state, s, b, &mut epoch);
        }
    }
    // The rewrite, block by block with the state each one starts in.
    for (b, block) in flow.blocks.iter_mut().enumerate() {
        let Some(stmts) = block.stmts.as_mut() else {
            continue;
        };
        let Some(mut state) = entry[b].take() else {
            // Nothing leads here.
            stmts.clear();
            continue;
        };
        let mut epoch = 0;
        for s in stmts.iter_mut() {
            tracer.rewrite(&mut state, s, b, &mut epoch)?;
        }
    }
    Ok(Retry::Done)
}
