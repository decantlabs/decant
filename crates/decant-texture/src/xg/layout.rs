//! Texture layouts of the console library (`docs/CONSOLE_LIBS.md` 1.3 and 1.6): linear and tiled, 1D, 2D, 3D, MSAA, the
//! CMASK plane, the byte offset of an element and the `XG_RESOURCE_LAYOUT` record.

use super::XgError;
use super::equations::{Equation, equation_2d, equation_3d};
use super::formats::{Format, classify};
use super::modes::{self, Volume};

/// The texture kind of a description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    D1,
    D2,
    D3,
}

/// One description, whichever of `XG_TEXTURE1D_DESC`, `XG_TEXTURE2D_DESC` and `XG_TEXTURE3D_DESC` it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDesc {
    pub dim: Dim,
    pub width: u32,
    pub height: u32,
    /// Array size (1D, 2D) or depth (3D).
    pub depth_or_array: u32,
    /// 0 means the full chain.
    pub mip_levels: u32,
    pub format: u32,
    pub sample_count: u32,
    pub usage: u32,
    pub bind_flags: u32,
    pub misc_flags: u32,
    pub swizzle_mode: u32,
    /// 1D and 3D only: elements, 0 means minimal.
    pub pitch: u32,
}

/// One mip level of a plane, field for field the 0x70 byte record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Level {
    pub size_bytes: u64,
    pub offset_bytes: u64,
    pub slice2d_size_bytes: u64,
    pub pitch_pixels: u32,
    pub pitch_bytes: u32,
    pub alignment_bytes: u32,
    pub padded_width: u32,
    pub padded_height: u32,
    pub padded_depth_or_array: u32,
    pub width_elements: u32,
    pub height_elements: u32,
    pub width_texels: u32,
    pub height_texels: u32,
    pub depth_or_array: u32,
    pub sample_count: u32,
    pub swizzle_mode: u32,
    pub constant_50: u32,
    pub constant_58: u32,
    pub block_depth: u32,
    pub in_tail: bool,
    /// Position of the level inside the tail block, in elements (0 outside the tail).
    pub tail_x: u32,
    pub tail_y: u32,
    /// 3D only: the position reported at offset 0x68.
    pub tail_position: u32,
}

/// One plane: the data, or the CMASK.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plane {
    /// 1 main data, 2 CMASK.
    pub usage: u32,
    pub size_bytes: u64,
    pub slice_size_bytes: u64,
    pub base_offset_bytes: u64,
    pub base_alignment_bytes: u64,
    pub bytes_per_element: u32,
    pub first_mip_in_tail: u32,
    pub mip_levels_in_tail: u32,
    pub levels: Vec<Level>,
}

/// The layout of a resource, the content of `XG_RESOURCE_LAYOUT`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceLayout {
    pub size_bytes: u64,
    pub base_alignment_bytes: u64,
    pub mip_levels: u32,
    pub planes: Vec<Plane>,
    /// 1 buffer, 2 1D, 3 2D, 4 3D (written as 1 to 4 in the record: the record uses 2 for 1D, 3 for 2D, 4 for 3D).
    pub dimension: u32,
}

/// The size of the `XG_RESOURCE_LAYOUT` record.
pub const RECORD_BYTES: usize = 0x1B40;
const PLANE_BYTES: usize = 0x6C8;
const LEVEL_BYTES: usize = 0x70;
const MAX_LEVELS: usize = 15;

/// "256 byte units modulo 2^24" of a size field.
fn reduce_size(v: u64) -> u64 {
    if v < 256 { v } else { ((v / 256 - 1) % (1 << 24) + 1) * 256 }
}

/// "Modulo 16384" of a padded width or height.
fn reduce_dim(v: u64) -> u64 {
    if v == 0 { 0 } else { (v - 1) % 16384 + 1 }
}

impl ResourceLayout {
    /// The record as the console library writes it: the whole structure zeroed, then filled.
    pub fn to_record(&self) -> Vec<u8> {
        let mut r = vec![0u8; RECORD_BYTES];
        let put32 = |r: &mut Vec<u8>, at: usize, v: u32| r[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put64 = |r: &mut Vec<u8>, at: usize, v: u64| r[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put64(&mut r, 0, self.size_bytes);
        put64(&mut r, 8, self.base_alignment_bytes);
        put32(&mut r, 0x10, self.mip_levels);
        put32(&mut r, 0x14, self.planes.len() as u32);
        for (p, plane) in self.planes.iter().take(4).enumerate() {
            let base = 0x18 + p * PLANE_BYTES;
            put32(&mut r, base, plane.usage);
            put64(&mut r, base + 0x08, plane.size_bytes);
            put64(&mut r, base + 0x10, plane.slice_size_bytes);
            put64(&mut r, base + 0x18, plane.base_offset_bytes);
            put64(&mut r, base + 0x20, plane.base_alignment_bytes);
            put32(&mut r, base + 0x28, plane.bytes_per_element);
            put32(&mut r, base + 0x2C, plane.first_mip_in_tail);
            put32(&mut r, base + 0x30, plane.mip_levels_in_tail);
            for (i, l) in plane.levels.iter().take(MAX_LEVELS).enumerate() {
                let at = base + 0x38 + i * LEVEL_BYTES;
                let (padded_w, padded_h) = (reduce_dim(u64::from(l.padded_width)), reduce_dim(u64::from(l.padded_height)));
                let per_element = |pitch: u32| u64::from(pitch / l.padded_width.max(1));
                let (blk, bpe) = (per_element(l.pitch_pixels), per_element(l.pitch_bytes));
                put64(&mut r, at, reduce_size(l.size_bytes));
                put64(&mut r, at + 0x08, l.offset_bytes);
                put64(&mut r, at + 0x10, reduce_size(l.slice2d_size_bytes));
                put32(&mut r, at + 0x18, (padded_w * blk) as u32);
                put32(&mut r, at + 0x1C, (padded_w * bpe) as u32);
                put32(&mut r, at + 0x20, l.alignment_bytes);
                put32(&mut r, at + 0x24, padded_w as u32);
                put32(&mut r, at + 0x28, padded_h as u32);
                put32(&mut r, at + 0x2C, l.padded_depth_or_array);
                put32(&mut r, at + 0x30, l.width_elements);
                put32(&mut r, at + 0x34, l.height_elements);
                put32(&mut r, at + 0x38, l.width_texels);
                put32(&mut r, at + 0x3C, l.height_texels);
                put32(&mut r, at + 0x40, l.depth_or_array);
                put32(&mut r, at + 0x44, l.sample_count);
                put32(&mut r, at + 0x48, l.swizzle_mode);
                put32(&mut r, at + 0x50, l.constant_50);
                put32(&mut r, at + 0x58, l.constant_58);
                put32(&mut r, at + 0x60, l.block_depth);
                put32(&mut r, at + 0x64, u32::from(l.in_tail));
                put32(&mut r, at + 0x68, l.tail_position);
            }
        }
        put32(&mut r, 0x1B38, self.dimension);
        r
    }
}

/// The layout of a buffer (`XGComputeBufferLayout`): no planes, no levels.
pub fn buffer_layout(byte_width: u32, bind_flags: u32, misc_flags: u32) -> Result<ResourceLayout, XgError> {
    if byte_width == 0 || bind_flags & BIND_MASK != 0 || misc_flags & MISC_MASK != 0 || (misc_flags & MISC_64K_MULTIPLE != 0 && !byte_width.is_multiple_of(65536)) {
        return Err(XgError::InvalidArg);
    }
    let big = misc_flags & (MISC_64K_MULTIPLE | MISC_64K_D) != 0;
    Ok(ResourceLayout { size_bytes: u64::from(byte_width), base_alignment_bytes: if big { 65536 } else { 4 }, mip_levels: 1, planes: Vec::new(), dimension: 1 })
}

/// Bind flag bits that fail for buffers, 1D and 3D textures.
const BIND_MASK: u32 = 0xFFFF_F900;
/// Misc flag bits that fail for buffers, 1D and 3D textures.
const MISC_MASK: u32 = 0xF385_6D0A;
const MISC_64K_MULTIPLE: u32 = 0x400_0000;
const MISC_64K_D: u32 = 0x800_0000;
const MISC_400000: u32 = 0x40_0000;
const BIND_RENDER_TARGET: u32 = 0x20;

/// A computed layout with what the element addresses need.
#[derive(Clone, Debug)]
pub struct TextureLayout {
    pub layout: ResourceLayout,
    desc_dim: Dim,
    mode: u32,
    bpe: u32,
    v4: bool,
    /// The equation, `None` for linear.
    equation: Option<Equation>,
    block_bits: u32,
    block_w: u32,
    block_h: u32,
    block_d: u32,
    thick: bool,
    /// Distance between array slices (2D, 1D) or depth slices or slabs (3D).
    slice_size: u64,
    array_or_depth: u32,
    /// Texels along one side of an element: 4 for block compressed formats, else 1.
    texels_per_element: u32,
}

fn align_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

/// Elements along one side of level `i`: ceil(ceil(size / block) / 2^i), at least 1.
fn level_elements(texels: u32, block: u32, i: u32) -> u32 {
    let base = u64::from(texels).div_ceil(u64::from(block));
    (base.div_ceil(1u64 << i)).max(1) as u32
}

fn level_texels(texels: u32, i: u32) -> u32 {
    (texels >> i).max(1)
}

fn full_chain(size: u32) -> u32 {
    size.max(1).ilog2() + 1
}

const SLOT_SEQUENCE: [u64; 16] = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 6, 5, 4, 3, 2, 1, 0];

/// Position inside a tail block of the 256 byte block index `t`, in 256 byte blocks along x and y.
fn tail_block_position(t: u64, kx: u32, ky: u32) -> (u32, u32) {
    let (mut rx, mut ry, mut x_turn) = (kx, ky, true);
    let (mut x, mut y) = (0u32, 0u32);
    for k in (0..kx + ky).rev() {
        let bit = ((t >> k) & 1) as u32;
        if (x_turn && rx > 0) || ry == 0 {
            x = (x << 1) | bit;
            rx -= 1;
        } else {
            y = (y << 1) | bit;
            ry -= 1;
        }
        x_turn = !x_turn;
    }
    (x, y)
}

/// Block shape of `2^bits` bytes for `bpe` byte elements and `ls` log2 samples.
fn block_shape(bits: u32, lb: u32, ls: u32) -> (u32, u32) {
    if bits.is_multiple_of(2) {
        let e = bits - lb;
        (1 << (e.div_ceil(2) - ls.div_ceil(2)), 1 << (e / 2 - ls / 2))
    } else {
        let e = bits - lb - ls;
        (1 << e.div_ceil(2), 1 << (e / 2))
    }
}

/// The 1 KiB thick block shapes for 1, 2, 4, 8 and 16 byte elements and the 256 byte micro blocks.
const THICK_1K: [(u32, u32, u32); 5] = [(16, 8, 8), (8, 8, 8), (8, 8, 4), (8, 4, 4), (4, 4, 4)];
const THICK_MICRO: [(u32, u32, u32); 5] = [(8, 4, 8), (4, 4, 8), (4, 4, 4), (4, 2, 4), (2, 2, 4)];
/// Positions of the tail levels of a thick block, in micro blocks.
const THICK_TAIL: [(u32, u32); 10] = [(4, 0), (0, 4), (2, 0), (1, 2), (0, 3), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)];

fn thick_shape(bits: u32, lb: u32) -> (u32, u32, u32) {
    let (w, h, d) = THICK_1K[lb as usize];
    let (a, r) = ((bits - 10) / 3, (bits - 10) % 3);
    (w << a, h << (a + r / 2), d << (a + u32::from(r > 0)))
}

/// What the validated description gives to the chain builders.
struct Common {
    mode: u32,
    bpe: u32,
    blk: u32,
    w: u32,
    h: u32,
    /// Array size (1D, 2D) or depth (3D).
    a: u32,
    n: u32,
    s: u32,
    v4: bool,
    pitch: u32,
    three_d: bool,
}

struct Chain {
    levels: Vec<Level>,
    slice_size: u64,
    first_tail: u32,
    alignment: u64,
    block: (u32, u32, u32),
    bits: u32,
}

fn base_level(c: &Common, i: u32) -> Level {
    let d_i = if c.three_d { (u64::from(c.a).div_ceil(1u64 << i)).max(1) as u32 } else { c.a };
    Level {
        width_elements: level_elements(c.w, c.blk, i),
        height_elements: level_elements(c.h, c.blk, i),
        width_texels: level_texels(c.w, i),
        height_texels: level_texels(c.h, i),
        depth_or_array: d_i,
        sample_count: c.s,
        block_depth: 1,
        ..Level::default()
    }
}

fn linear_chain(c: &Common) -> Chain {
    let mut levels: Vec<Level> = (0..c.n).map(|i| base_level(c, i)).collect();
    let mut run = 0u64;
    for i in (0..c.n as usize).rev() {
        let l = &mut levels[i];
        let wanted = if i == 0 { l.width_elements.max(c.pitch) } else { l.width_elements };
        let pitch_bytes = align_up(u64::from(wanted) * u64::from(c.bpe), 256);
        let slice2d = pitch_bytes * u64::from(l.height_elements);
        l.pitch_bytes = pitch_bytes as u32;
        l.padded_width = (pitch_bytes / u64::from(c.bpe)) as u32;
        l.padded_height = l.height_elements;
        l.pitch_pixels = l.padded_width * c.blk;
        l.alignment_bytes = 256;
        l.offset_bytes = run;
        l.slice2d_size_bytes = slice2d;
        l.size_bytes = if c.three_d { slice2d * u64::from(c.a) } else { slice2d };
        l.padded_depth_or_array = l.depth_or_array;
        l.swizzle_mode = 0;
        run += slice2d;
    }
    Chain { levels, slice_size: run, first_tail: c.n, alignment: 256, block: (1, 1, 1), bits: 0 }
}

/// 2D chain of a tiled mode (also 1D, and the depth slices of a thin volume).
fn tiled_chain(c: &Common) -> Chain {
    let lb = c.bpe.ilog2();
    let bits = modes::block_bits(c.mode, c.v4);
    let block_size = 1u64 << bits;
    let (bw, bh) = block_shape(bits, lb, c.s.ilog2());
    let (mw, mh) = block_shape(8, lb, 0);
    let mut levels: Vec<Level> = (0..c.n).map(|i| base_level(c, i)).collect();
    let (tw, th) = (bw / 2, bh);
    let (fw, fh) = match (c.mode & 0x1F, c.bpe) {
        (24 | 28, 1) => (tw / 2, th / 2),
        (24 | 28, 2) => (tw / 2, th),
        _ => (tw, th),
    };
    let mut first = c.n;
    if bits > 8 && c.n > 1 && let Some(i0) = levels.iter().position(|l| l.width_elements <= fw && l.height_elements <= fh) {
        first = (i0 as u32).max(c.n.saturating_sub(bits - 4));
    }
    let mut run = if first < c.n { block_size } else { 0 };
    for i in (0..first as usize).rev() {
        let l = &mut levels[i];
        let pad_w = align_up(u64::from(l.width_elements), u64::from(bw));
        let pad_h = align_up(u64::from(l.height_elements), u64::from(bh));
        let size = pad_w * pad_h * u64::from(c.bpe) * u64::from(c.s);
        l.padded_width = pad_w as u32;
        l.padded_height = pad_h as u32;
        l.pitch_bytes = (pad_w * u64::from(c.bpe)) as u32;
        l.pitch_pixels = pad_w as u32 * c.blk;
        l.alignment_bytes = block_size as u32;
        l.offset_bytes = run;
        l.size_bytes = size;
        l.slice2d_size_bytes = size;
        l.swizzle_mode = c.mode & 0x1F;
        if (24..=27).contains(&(c.mode & 0x1F)) {
            (l.constant_50, l.constant_58) = if c.v4 { (0xE000, 0x2000) } else { (0x8000, 0x8000) };
        }
        run += size;
    }
    let slice_size = align_up(run, block_size);
    if first < c.n {
        let tail_mode = modes::tail_mode(c.mode).unwrap_or(c.mode & 0x1F);
        let (kx, ky) = ((bw / mw).ilog2(), (bh / mh).ilog2());
        for j in 0..(c.n - first) {
            let l = &mut levels[(first + j) as usize];
            let t = SLOT_SEQUENCE[((20 - bits) + j).min(15) as usize];
            let (px, py) = tail_block_position(t, kx, ky);
            let pad_w = (tw >> j.min(31)).max(mw);
            let pad_h = (th >> j.min(31)).max(mh);
            let size = u64::from(pad_w) * u64::from(pad_h) * u64::from(c.bpe);
            l.padded_width = pad_w;
            l.padded_height = pad_h;
            l.pitch_bytes = pad_w * c.bpe;
            l.pitch_pixels = pad_w * c.blk;
            l.alignment_bytes = 256;
            l.offset_bytes = if c.three_d { 0 } else { t * 256 };
            l.size_bytes = size;
            l.slice2d_size_bytes = size;
            l.swizzle_mode = tail_mode;
            l.in_tail = true;
            l.tail_x = px * mw;
            l.tail_y = py * mh;
        }
    }
    Chain { levels, slice_size, first_tail: first, alignment: block_size, block: (bw, bh, 1), bits }
}

/// Thick 3D chain: one slab holds a block of depth.
fn thick_chain(c: &Common) -> Chain {
    let lb = c.bpe.ilog2();
    let bits = modes::block_bits(c.mode, c.v4);
    let block_size = 1u64 << bits;
    let (bw, bh, bd) = thick_shape(bits, lb);
    let (mw, mh, md) = THICK_MICRO[lb as usize];
    let mut levels: Vec<Level> = (0..c.n).map(|i| base_level(c, i)).collect();
    let (fw, fh) = if bits.is_multiple_of(3) { (bw, bh / 2) } else { (bw / 2, bh) };
    let max_tail = if bits == 12 { 5 } else { 10 };
    let mut first = c.n;
    if c.n > 1 && let Some(i0) = levels.iter().position(|l| l.width_elements <= fw && l.height_elements <= fh) {
        first = (i0 as u32).max(c.n.saturating_sub(max_tail));
    }
    let mut run = if first < c.n { block_size } else { 0 };
    for i in (0..first as usize).rev() {
        let l = &mut levels[i];
        let pad_w = align_up(u64::from(l.width_elements), u64::from(bw));
        let pad_h = align_up(u64::from(l.height_elements), u64::from(bh));
        let pad_d = align_up(u64::from(l.depth_or_array), u64::from(bd));
        let slice2d = pad_w * pad_h * u64::from(c.bpe);
        l.padded_width = pad_w as u32;
        l.padded_height = pad_h as u32;
        l.padded_depth_or_array = pad_d as u32;
        l.pitch_bytes = (pad_w * u64::from(c.bpe)) as u32;
        l.pitch_pixels = pad_w as u32 * c.blk;
        l.alignment_bytes = block_size as u32;
        l.offset_bytes = run;
        l.slice2d_size_bytes = slice2d;
        l.size_bytes = slice2d * pad_d;
        l.swizzle_mode = c.mode & 0x1F;
        l.block_depth = bd;
        if (24..=27).contains(&(c.mode & 0x1F)) {
            (l.constant_50, l.constant_58) = if c.v4 { (0xE000, 0x2000) } else { (0x8000, 0x8000) };
        }
        run += slice2d * u64::from(bd);
    }
    let slab = align_up(run, block_size);
    if first < c.n {
        let tail_mode = modes::tail_mode(c.mode).unwrap_or(c.mode & 0x1F);
        let slot0 = if bits == 12 { 5 } else { 0 };
        for j in 0..(c.n - first) {
            let l = &mut levels[(first + j) as usize];
            let pad_w = (fw >> j.min(31)).max(mw);
            let pad_h = (fh >> j.min(31)).max(mh);
            let pad_d = align_up(u64::from(l.depth_or_array), u64::from(md));
            let (px, py) = THICK_TAIL[(slot0 + j as usize).min(THICK_TAIL.len() - 1)];
            let slice2d = u64::from(pad_w) * u64::from(pad_h) * u64::from(c.bpe);
            l.padded_width = pad_w;
            l.padded_height = pad_h;
            l.padded_depth_or_array = pad_d as u32;
            l.pitch_bytes = pad_w * c.bpe;
            l.pitch_pixels = pad_w * c.blk;
            l.alignment_bytes = 256;
            l.offset_bytes = 0;
            l.slice2d_size_bytes = slice2d;
            l.size_bytes = slice2d * pad_d;
            l.swizzle_mode = tail_mode;
            l.block_depth = bd;
            l.in_tail = true;
            l.tail_x = px * mw;
            l.tail_y = py * mh;
            l.tail_position = (l.tail_x & 0x7FF) | (l.tail_y << 11);
        }
    }
    Chain { levels, slice_size: slab, first_tail: first, alignment: block_size, block: (bw, bh, bd), bits }
}

/// Validates a description and computes its layout. `v4` is the hardware version flag of `XGSetHardwareVersion`.
pub fn compute_layout(desc: &TextureDesc, v4: bool) -> Result<TextureLayout, XgError> {
    match desc.dim {
        Dim::D2 => compute_2d(desc, v4),
        Dim::D1 | Dim::D3 => compute_1d_3d(desc, v4),
    }
}

fn wrap_dimension(v: u32) -> u32 {
    ((v - 1) % 65536) + 1
}

fn compute_2d(d: &TextureDesc, v4: bool) -> Result<TextureLayout, XgError> {
    if d.misc_flags & MISC_400000 != 0 && d.misc_flags & MISC_64K_D != 0 {
        return Err(XgError::InvalidArg);
    }
    if d.width == 0 || d.height == 0 || d.depth_or_array == 0 {
        return Err(XgError::InvalidArg);
    }
    let (w, h) = (wrap_dimension(d.width), wrap_dimension(d.height));
    if d.sample_count > 15 {
        return Err(XgError::InvalidArg);
    }
    let s = if d.sample_count <= 1 { 1 } else { 1 << d.sample_count.ilog2() };
    let mut n = if d.mip_levels == 0 { full_chain(w.max(h)) } else { d.mip_levels };
    if n > 15 {
        return Err(XgError::InvalidArg);
    }
    if s > 1 {
        n = 1;
    }
    let (bpe, blk) = match classify(d.format) {
        Format::Element { bpe, block } => (bpe, block),
        Format::NotModelled => return Err(XgError::NotImpl),
        Format::Unknown => return Err(XgError::InvalidArg),
    };
    let mode = d.swizzle_mode & 0x1F;
    if d.usage == 3 && mode != 0 || mode == 0 && blk == 4 && d.usage != 3 {
        return Err(XgError::InvalidArg);
    }
    if s > 1 {
        let allowed = if bpe <= 4 { matches!(mode, 24 | 27 | 28 | 31) } else { matches!(mode, 27 | 31) };
        if !allowed || blk == 4 {
            return Err(XgError::InvalidArg);
        }
    }
    if mode != 0 && !modes::valid_element(mode, bpe) {
        return Err(XgError::InvalidArg);
    }
    if matches!(mode, 24 | 28) && blk == 4 && !(d.bind_flags & BIND_RENDER_TARGET != 0 && s == 1 && n == 1) {
        return Err(XgError::InvalidArg);
    }
    let c = Common { mode, bpe, blk, w, h, a: d.depth_or_array, n, s, v4, pitch: 0, three_d: false };
    build(&c, Dim::D2, d, Volume::No)
}

fn compute_1d_3d(d: &TextureDesc, v4: bool) -> Result<TextureLayout, XgError> {
    let is_1d = d.dim == Dim::D1;
    let height = if is_1d { 1 } else { d.height };
    if d.width == 0 || height == 0 || d.depth_or_array == 0 {
        return Err(XgError::InvalidArg);
    }
    let mode = d.swizzle_mode & 0x1F;
    if d.usage == 3 && mode != 0 {
        return Err(XgError::InvalidArg);
    }
    if d.pitch != 0 && d.pitch < d.width {
        return Err(XgError::InvalidArg);
    }
    if d.bind_flags & BIND_MASK != 0 || d.misc_flags & MISC_MASK != 0 {
        return Err(XgError::InvalidArg);
    }
    let format = classify(d.format);
    let (bpe, blk) = match format {
        Format::Element { bpe, block } => (bpe, block),
        _ => (0, 0),
    };
    if blk == 4 && mode == 0 && d.usage != 3 {
        return Err(XgError::InvalidArg);
    }
    if (5..=8).contains(&d.format) || d.misc_flags & (MISC_64K_D | MISC_64K_MULTIPLE | MISC_400000) != 0 {
        return Err(XgError::InvalidArg);
    }
    let n = if d.mip_levels == 0 {
        if is_1d { full_chain(d.width) } else { full_chain(d.width.max(height).max(d.depth_or_array)) }
    } else {
        d.mip_levels
    };
    if n > 15 {
        return Err(XgError::InvalidArg);
    }
    match format {
        Format::Unknown => return Err(XgError::InvalidArg),
        Format::NotModelled => return Err(XgError::NotImpl),
        Format::Element { .. } => {}
    }
    let depth_limit = if is_1d { 0 } else { d.depth_or_array };
    if d.width > 16384 || height > 16384 || depth_limit > 16384 {
        return Err(XgError::NotImpl);
    }
    let volume = modes::volume(mode);
    let mode_ok = if is_1d { modes::accepts_1d(mode) } else { !matches!(volume, Volume::No) };
    if !mode_ok || is_1d && blk == 4 {
        return Err(XgError::InvalidArg);
    }
    if mode != 0 && !modes::valid_element(mode, bpe) {
        return Err(XgError::InvalidArg);
    }
    let c = Common { mode, bpe, blk, w: d.width, h: height, a: d.depth_or_array, n, s: 1, v4, pitch: d.pitch, three_d: !is_1d };
    let layout = build(&c, d.dim, d, if is_1d { Volume::No } else { volume })?;
    if layout.layout.size_bytes >= 1 << 40 {
        return Err(XgError::NotImpl);
    }
    Ok(layout)
}

/// Runs the chain builder for the validated parameters and assembles the planes.
fn build(c: &Common, dim: Dim, d: &TextureDesc, volume: Volume) -> Result<TextureLayout, XgError> {
    let thick = c.mode != 0 && c.three_d && volume == Volume::Thick;
    let equation = if c.mode == 0 {
        None
    } else if thick {
        match equation_3d(c.mode, c.bpe, c.v4) {
            Some(e) => Some(e),
            None => return Err(XgError::NotImpl),
        }
    } else {
        Some(equation_2d(c.mode, c.bpe, c.s, c.v4).ok_or(XgError::InvalidArg)?)
    };
    let mut chain = if c.mode == 0 {
        linear_chain(c)
    } else if thick {
        thick_chain(c)
    } else {
        tiled_chain(c)
    };
    // Reported depth fields of 3D levels.
    if c.three_d && !thick {
        for l in chain.levels.iter_mut() {
            l.padded_depth_or_array = if c.mode == 0 { l.depth_or_array } else if c.n == 1 { c.a } else { 1 };
        }
    } else if !c.three_d {
        for l in chain.levels.iter_mut() {
            l.padded_depth_or_array = c.a;
        }
    }
    let slices: u64 = if thick { u64::from(align_up(u64::from(c.a), u64::from(chain.block.2)) as u32) / u64::from(chain.block.2) } else { u64::from(c.a) };
    let main_size = u128::from(chain.slice_size) * u128::from(slices);
    let main_size = u64::try_from(main_size).map_err(|_| XgError::InvalidArg)?;
    let first_tail = chain.first_tail;
    let main = Plane {
        usage: 1,
        size_bytes: main_size,
        slice_size_bytes: chain.slice_size,
        base_offset_bytes: 0,
        base_alignment_bytes: chain.alignment,
        bytes_per_element: c.bpe,
        first_mip_in_tail: first_tail,
        mip_levels_in_tail: c.n - first_tail,
        levels: chain.levels,
    };
    let mut layout = ResourceLayout { size_bytes: main_size, base_alignment_bytes: chain.alignment, mip_levels: c.n, planes: vec![main], dimension: match dim { Dim::D1 => 2, Dim::D2 => 3, Dim::D3 => 4 } };
    let compressed_z = matches!(c.mode, 24 | 28) && c.blk == 4;
    if dim == Dim::D2 && d.bind_flags & BIND_RENDER_TARGET != 0 && c.s == 1 && c.n == 1 && d.misc_flags & MISC_64K_D == 0 && !compressed_z {
        let size = align_up(u64::from(c.w), 1024) * align_up(u64::from(c.h), 1024) / 128 * u64::from(c.a);
        let base = align_up(main_size, 0x2000);
        let level = Level { size_bytes: size, alignment_bytes: 0x2000, ..Level::default() };
        layout.planes.push(Plane { usage: 2, size_bytes: size, slice_size_bytes: 0, base_offset_bytes: base, base_alignment_bytes: 0x2000, bytes_per_element: 0, first_mip_in_tail: 0, mip_levels_in_tail: 0, levels: vec![level] });
        layout.size_bytes = base + size;
        layout.base_alignment_bytes = chain.alignment.max(0x2000);
    }
    Ok(TextureLayout {
        layout,
        desc_dim: dim,
        mode: c.mode,
        bpe: c.bpe,
        v4: c.v4,
        equation,
        block_bits: chain.bits,
        block_w: chain.block.0,
        block_h: chain.block.1,
        block_d: chain.block.2,
        thick,
        slice_size: chain.slice_size,
        array_or_depth: c.a,
        texels_per_element: c.blk,
    })
}

impl TextureLayout {
    /// 1D, 2D or 3D.
    pub fn dim(&self) -> Dim {
        self.desc_dim
    }

    /// The main plane.
    pub fn plane0(&self) -> &Plane {
        &self.layout.planes[0]
    }

    /// Array size (1D, 2D) or depth (3D).
    pub fn array_or_depth(&self) -> u32 {
        self.array_or_depth
    }

    /// Bytes per element.
    pub fn bytes_per_element(&self) -> u32 {
        self.bpe
    }

    /// Texels along one side of an element: 4 for block compressed formats, else 1.
    pub fn texels_per_element(&self) -> u32 {
        self.texels_per_element
    }

    /// The block shape in elements along depth, 1 for thin modes.
    pub fn block_depth(&self) -> u32 {
        self.block_d
    }

    /// The address equation of a tiled mode, `None` for linear.
    pub fn equation(&self) -> Option<&Equation> {
        self.equation.as_ref()
    }

    /// log2 of the block size in bytes (0 for linear).
    pub fn block_bits(&self) -> u32 {
        self.block_bits
    }

    /// The block shape in elements (width, height), (1, 1) for linear.
    pub fn block_shape(&self) -> (u32, u32) {
        (self.block_w, self.block_h)
    }

    /// Whether the mode lays 3D textures out in thick blocks (the slice index then selects a slab, not an array slice).
    pub fn is_thick(&self) -> bool {
        self.thick
    }

    /// Distance in bytes between array slices (2D, 1D) or depth slices or slabs (3D).
    pub fn slice_size(&self) -> u64 {
        self.slice_size
    }

    /// Whether version 4 equations were used.
    pub fn hardware_v4(&self) -> bool {
        self.v4
    }

    /// The swizzle mode of the resource.
    pub fn swizzle_mode(&self) -> u32 {
        self.mode
    }

    /// The byte offset of element (x, y) of mip `mip`, array or depth slice `z`, sample `s` in plane 0. `None` when the
    /// mip, slice or sample is out of range or x or y is beyond the padded level.
    pub fn element_offset(&self, mip: u32, x: u64, y: u32, z: u32, s: u32) -> Option<u64> {
        let l = self.layout.planes[0].levels.get(mip as usize)?;
        if x >= u64::from(l.padded_width) || y >= l.padded_height || z >= l.depth_or_array || s >= l.sample_count {
            return None;
        }
        let Some(eq) = &self.equation else {
            return Some(u64::from(z) * self.slice_size + l.offset_bytes + u64::from(y) * u64::from(l.pitch_bytes) + x * u64::from(self.bpe));
        };
        let block = 1u64 << self.block_bits;
        let reduce = |e: u32| u64::from(e) & (block - 1);
        let z_base = if self.thick { u64::from(z / self.block_d) * self.slice_size } else { u64::from(z) * self.slice_size };
        if l.in_tail {
            let e = eq.eval((x as u32).wrapping_add(l.tail_x), y.wrapping_add(l.tail_y), z, s);
            return Some(z_base + reduce(e));
        }
        let blocks_x = l.padded_width / self.block_w;
        let block_index = u64::from((y / self.block_h).wrapping_mul(blocks_x).wrapping_add(x as u32 / self.block_w));
        Some(z_base + l.offset_bytes + block_index * block + reduce(eq.eval(x as u32, y, z, s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(dim: Dim, w: u32, h: u32, a: u32, n: u32, format: u32, mode: u32) -> TextureDesc {
        TextureDesc { dim, width: w, height: h, depth_or_array: a, mip_levels: n, format, sample_count: 1, usage: u32::from(mode == 0) * 3, bind_flags: 8, misc_flags: 0, swizzle_mode: mode, pitch: 0 }
    }

    #[test]
    fn tail_positions_decode_the_block_index() {
        // 4 KiB blocks of 4 byte elements: 32 x 32 elements in 8 x 8 micro blocks, 2 bits each way.
        assert_eq!(tail_block_position(8, 2, 2), (2, 0));
        // 64 KiB blocks of 4 byte elements: 128 x 128 in 8 x 8, 4 bits each way.
        assert_eq!(tail_block_position(128, 4, 4), (8, 0));
        assert_eq!(tail_block_position(0, 4, 4), (0, 0));
    }

    #[test]
    fn linear_level_sizes_and_the_record_reductions() {
        assert_eq!(reduce_size(100), 100);
        assert_eq!(reduce_size(0x4000_0000), 0x4000_0000);
        assert_eq!(reduce_size(256 * (1 << 24) + 256), 256);
        assert_eq!(reduce_dim(16384), 16384);
        assert_eq!(reduce_dim(16385), 1);
        assert_eq!(reduce_dim(0), 0);
        let t = compute_layout(&desc(Dim::D2, 100, 20, 1, 1, 28, 0), false).unwrap();
        assert_eq!(t.layout.size_bytes, 512 * 20);
        assert_eq!(t.plane0().levels[0].padded_width, 128);
    }

    #[test]
    fn thick_blocks_have_the_listed_shapes() {
        assert_eq!(thick_shape(12, 2), (8, 16, 8));
        assert_eq!(thick_shape(16, 2), (32, 32, 16));
        assert_eq!(thick_shape(12, 0), (16, 16, 16));
    }

    #[test]
    fn the_2d_equations_cover_every_block_of_a_thick_mode_once() {
        // The thick 3D equations are one to one over a block of the thick shape.
        for mode in [5u32, 9, 17, 21, 25, 26] {
            for bpe in [1u32, 2, 4, 8, 16] {
                for v4 in [false, true] {
                    let Some(e) = equation_3d(mode, bpe, v4) else { continue };
                    let bits = modes::block_bits(mode, v4);
                    let (bw, bh, bd) = thick_shape(bits, bpe.ilog2());
                    for (ox, oy, oz) in [(0u32, 0u32, 0u32), (3 * bw, 2 * bh, 5 * bd)] {
                        let mut seen = std::collections::HashSet::new();
                        for z in 0..bd {
                            for y in 0..bh {
                                for x in 0..bw {
                                    let o = e.eval(ox + x, oy + y, oz + z, 0) & ((1 << bits) - 1);
                                    assert_eq!(o % bpe, 0);
                                    assert!(seen.insert(o), "mode {mode} bpe {bpe} v4 {v4}: {o:#x} twice");
                                }
                            }
                        }
                        assert_eq!(seen.len() as u32, (1 << bits) / bpe, "mode {mode} bpe {bpe} v4 {v4}");
                    }
                }
            }
        }
    }

    #[test]
    fn errors_of_the_validation() {
        let ok = desc(Dim::D2, 64, 64, 1, 1, 28, 5);
        assert!(compute_layout(&ok, false).is_ok());
        let bad = |d: TextureDesc| compute_layout(&d, false).unwrap_err();
        assert_eq!(bad(TextureDesc { width: 0, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { height: 0, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { depth_or_array: 0, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { sample_count: 16, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { mip_levels: 16, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { format: 5, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { format: 0, ..ok }), XgError::NotImpl);
        assert_eq!(bad(TextureDesc { format: 45, ..ok }), XgError::NotImpl);
        assert_eq!(bad(TextureDesc { usage: 3, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { swizzle_mode: 3, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { misc_flags: 0x40_0000 | 0x800_0000, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { format: 71, swizzle_mode: 0, usage: 0, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { sample_count: 2, ..ok }), XgError::InvalidArg, "MSAA needs 24, 27, 28 or 31");
        assert_eq!(bad(TextureDesc { format: 71, swizzle_mode: 24, ..ok }), XgError::InvalidArg, "block compressed mode 24");
        assert!(compute_layout(&TextureDesc { format: 71, swizzle_mode: 24, bind_flags: 0x20, ..ok }, false).is_ok());
        assert_eq!(bad(TextureDesc { dim: Dim::D3, swizzle_mode: 6, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { dim: Dim::D3, width: 16385, ..ok }), XgError::NotImpl);
        assert_eq!(bad(TextureDesc { dim: Dim::D3, pitch: 10, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { dim: Dim::D3, swizzle_mode: 26, format: 10, ..ok }), XgError::NotImpl, "no thick equation for mode 26 with 8 byte elements");
        assert_eq!(bad(TextureDesc { dim: Dim::D1, swizzle_mode: 5, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { dim: Dim::D1, format: 71, swizzle_mode: 0, usage: 3, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { dim: Dim::D3, bind_flags: 0x100, ..ok }), XgError::InvalidArg);
        assert_eq!(bad(TextureDesc { dim: Dim::D3, misc_flags: 0x2, ..ok }), XgError::InvalidArg);
    }

    #[test]
    fn element_offsets_reject_out_of_range_arguments() {
        let t = compute_layout(&desc(Dim::D2, 64, 64, 2, 1, 28, 5), false).unwrap();
        assert_eq!(t.element_offset(0, 0, 0, 0, 0), Some(0));
        assert_eq!(t.element_offset(1, 0, 0, 0, 0), None);
        assert_eq!(t.element_offset(0, 64, 0, 0, 0), None);
        assert_eq!(t.element_offset(0, 0, 0, 2, 0), None);
        assert_eq!(t.element_offset(0, 0, 0, 0, 1), None);
    }

    #[test]
    fn two_d_layouts_agree_with_the_surface_calculation() {
        // Where the specification says so: the modes `surface::compute` knows, one sample, no Z fit (modes 24 with 1 and 2 byte
        // elements fit differently, see 1.6).
        use crate::surface::{Resource, Surface, compute};
        let mut compared = 0;
        for mode in [1u32, 2, 5, 6, 9, 10, 21, 22, 24, 25, 26, 27] {
            let core_mode = modes::core_mode(mode).unwrap();
            for (bpe, format) in [(1u32, 61u32), (2, 56), (4, 28), (8, 10), (16, 2)] {
                if mode == 24 && bpe != 4 && bpe != 8 {
                    continue;
                }
                // Powers of two: `surface::compute` halves a level size rounding down, the console library rounds up (vector AH).
                for (w, h) in [(1u32, 1u32), (16, 16), (16, 64), (64, 64), (256, 8), (256, 256), (1024, 512)] {
                    for (levels, slices) in [(1u32, 1u32), (3, 1), (0, 1), (6, 3), (0, 4)] {
                        let d = TextureDesc { bind_flags: 8, ..desc(Dim::D2, w, h, slices, levels, format, mode) };
                        let t = compute_layout(&d, false).unwrap();
                        let n = t.layout.mip_levels;
                        let info = compute(&Surface { mode: core_mode, resource: Resource::Tex2d, bpe: bpe as usize, width: w as usize, height: h as usize, slices: slices as usize, levels: n as usize }).unwrap();
                        let p = t.plane0();
                        let what = format!("mode {mode} bpe {bpe} {w}x{h} levels {n} slices {slices}");
                        assert_eq!(p.slice_size_bytes, info.slice_size, "{what}");
                        assert_eq!(t.layout.size_bytes, info.surf_size, "{what}");
                        assert_eq!(p.first_mip_in_tail as usize, info.first_mip_in_tail, "{what}");
                        for (i, (l, m)) in p.levels.iter().zip(&info.mips).enumerate() {
                            assert_eq!(l.in_tail, m.in_tail, "{what} level {i}");
                            if m.in_tail {
                                assert_eq!((l.offset_bytes, l.tail_x as usize, l.tail_y as usize), (u64::from(m.tail_offset), m.tail_x, m.tail_y), "{what} tail level {i}");
                            } else {
                                assert_eq!(l.offset_bytes, m.macro_block_offset, "{what} level {i}");
                            }
                            assert_eq!((l.padded_width as usize, l.padded_height as usize), (m.pitch, m.height), "{what} level {i} padded size");
                        }
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000);
    }

    #[test]
    fn buffers() {
        let b = buffer_layout(0x64, 0, 0).unwrap();
        assert_eq!((b.size_bytes, b.base_alignment_bytes, b.dimension), (0x64, 4, 1));
        let r = b.to_record();
        assert_eq!(u64::from_le_bytes(r[0x10..0x18].try_into().unwrap()), 1);
        assert_eq!(buffer_layout(0, 0, 0), Err(XgError::InvalidArg));
        assert_eq!(buffer_layout(1, 0x100, 0), Err(XgError::InvalidArg));
        assert_eq!(buffer_layout(0x64, 0, 0x400_0000), Err(XgError::InvalidArg));
        assert_eq!(buffer_layout(0x64, 0, 0x8000_0000), Err(XgError::InvalidArg));
        assert_eq!(buffer_layout(0x20000, 0, 0x400_0000).unwrap().base_alignment_bytes, 0x10000);
        assert_eq!(buffer_layout(0x64, 0, 0x800_0000).unwrap().base_alignment_bytes, 0x10000);
    }
}
