//! Structured and typed buffer views of the title become raw views over the same bytes.
//!
//! A console buffer descriptor carries its record stride, and the machine computes
//! `index * stride + offset` itself. On PC a structured buffer read uses the stride the shader
//! declares on some drivers (WARP, vkd3d-proton) and the stride of the view on others (measured on
//! an RTX 3090). And a console program may read one buffer both with format loads and
//! with dword loads, which no single typed PC view serves. The translated shaders (and the old
//! override shaders, which declare every buffer a `ByteAddressBuffer`) therefore read buffers raw and
//! do the arithmetic and the format conversion themselves, with the element size, count and format
//! from the view shadow (`view_shadow`).
//!
//! The view keeps its bytes: format `R32_TYPELESS`, flag `RAW`, the default component mapping, and
//! the dwords that hold the bytes from `FirstElement * size` for `NumElements * size`, where `size`
//! is the structure stride or the element size of the format. A typed view of 1 or 2 byte
//! elements (`R8_UNORM`, `R16_UINT`, `R8G8_SNORM` ...) is made raw the same way: the program reads
//! the dword that holds an element and shifts it out, and writes only its bytes. Any other size
//! that is not a multiple of 4 has no raw form, so such a view stays as it is.
//!
//! A raw view on PC must start on 16 bytes (`D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT`), the console's on
//! 4. A view whose first byte is not on 16 bytes starts at the 16 bytes below and is longer by the
//! difference, and the row says by how much (`Records::offset`, 0 to 15, a multiple of 4 unless
//! the elements are smaller): the program adds it to every address. The same goes for a raw view
//! of the title's own.
//!
//! A raw view clears differently from a typed one, so [`clear_words`] gives the words of a clear
//! of a view made raw (`list_clear_uav_uint`, `list_clear_uav_float` of the proxy).
//!
//! The two description layouts (40 bytes each):
//! - `D3D12_SHADER_RESOURCE_VIEW_DESC`: format 0, dimension 4, component mapping 8,
//!   `FirstElement` 0x10 (u64), `NumElements` 0x18, `StructureByteStride` 0x1C, `Flags` 0x20.
//! - `D3D12_UNORDERED_ACCESS_VIEW_DESC`: format 0, dimension 4, `FirstElement` 0x08 (u64),
//!   `NumElements` 0x10, `StructureByteStride` 0x14, `CounterOffsetInBytes` 0x18 (u64), `Flags` 0x20.

/// `DXGI_FORMAT_R32_TYPELESS`.
const R32_TYPELESS: u32 = 39;
/// `DXGI_FORMAT_R32_UINT`.
const R32_UINT: u32 = 42;
/// `D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING`.
const DEFAULT_MAPPING: u32 = 0x1688;
/// `D3D12_SRV_DIMENSION_BUFFER` and `D3D12_UAV_DIMENSION_BUFFER`.
const DIMENSION_BUFFER: u32 = 1;
/// `D3D12_BUFFER_SRV_FLAG_RAW` and `D3D12_BUFFER_UAV_FLAG_RAW`.
const FLAG_RAW: u32 = 1;
/// Where the flags of a buffer view lie, in both layouts.
const FLAGS_AT: usize = 0x20;

/// The bytes of a view description.
pub const DESC_BYTES: usize = 40;

/// Which kind of view a description is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKind {
    ShaderResource,
    UnorderedAccess,
}

impl ViewKind {
    /// Offsets of `FirstElement`, `NumElements` and `StructureByteStride`.
    fn fields(self) -> (usize, usize, usize) {
        match self {
            ViewKind::ShaderResource => (0x10, 0x18, 0x1C),
            ViewKind::UnorderedAccess => (0x08, 0x10, 0x14),
        }
    }
}

/// The element layout of a buffer view, as the console descriptor had it: the row of the view
/// shadow for the slot that holds the view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Records {
    /// The record stride, or the element size of the format, in bytes.
    pub stride: u32,
    /// The number of records or elements.
    pub count: u32,
    /// The DXGI format of a typed view, 0 for a structured one.
    pub format: u32,
    /// Where the title's first byte is in the PC view: 0 to 15.
    pub offset: u32,
}

/// What [`make_raw`] did to a description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No buffer view of records, elements or dwords: the description is unchanged.
    NotStructured,
    /// The description is now the raw view of these records (a raw view of the title's: stride and
    /// format 0, its dwords and the offset, the description unchanged when the offset is 0).
    Raw(Records),
    /// A view that has no raw form (a size that is not a multiple of 4, an unknown format, or more
    /// than 2^32 dwords): the description is unchanged.
    Kept(Records),
    /// A buffer view of no elements, which the console takes and the PC runtime calls invalid and
    /// removes the device for (a world load: a raw UAV with
    /// `NumElements` 0). The description is now that of a null view: the caller passes no resource,
    /// so the slot reads zeros and drops writes, as an empty view does.
    Empty,
}

impl Outcome {
    /// The records of a view made raw.
    pub fn raw(self) -> Option<Records> {
        match self {
            Outcome::Raw(r) => Some(r),
            _ => None,
        }
    }
}

fn get32(d: &[u8; DESC_BYTES], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

fn put32(d: &mut [u8; DESC_BYTES], at: usize, value: u32) {
    d[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// Turns the structured or typed buffer view `desc` into the raw view of the same bytes, and moves
/// a raw view that does not start on 16 bytes to the 16 bytes below.
pub fn make_raw(desc: &mut [u8; DESC_BYTES], kind: ViewKind) -> Outcome {
    let (first_at, count_at, stride_at) = kind.fields();
    let (format, stride) = (get32(desc, 0), get32(desc, stride_at));
    let title_raw = get32(desc, FLAGS_AT) & FLAG_RAW != 0;
    if get32(desc, 4) != DIMENSION_BUFFER {
        return Outcome::NotStructured;
    }
    if get32(desc, count_at) == 0 {
        // a null buffer view: a typed format, one element (the runtime looks at no range without a resource)
        put32(desc, 0, R32_UINT);
        desc[first_at..first_at + 8].fill(0);
        put32(desc, count_at, 1);
        put32(desc, stride_at, 0);
        put32(desc, FLAGS_AT, 0);
        match kind {
            ViewKind::ShaderResource => put32(desc, 8, DEFAULT_MAPPING),
            ViewKind::UnorderedAccess => desc[0x18..0x20].fill(0),
        }
        return Outcome::Empty;
    }
    if !title_raw && stride == 0 && format == 0 {
        return Outcome::NotStructured;
    }
    // a raw view counts dwords, a stride makes the view structured whatever the format says
    let (size, mut records) = if title_raw {
        (4, Records { count: get32(desc, count_at), ..Records::default() })
    } else if stride != 0 {
        (stride, Records { stride, count: get32(desc, count_at), format: 0, offset: 0 })
    } else {
        let size = crate::views_at::format_bytes(format);
        (size, Records { stride: size, count: get32(desc, count_at), format, offset: 0 })
    };
    // elements of 1 or 2 bytes are read from their dword, when the program knows their format
    let small = size < 4 && format != 0 && channels(format).is_some();
    if size == 0 || !(size.is_multiple_of(4) || small) {
        return Outcome::Kept(records);
    }
    let first = u64::from_le_bytes(desc[first_at..first_at + 8].try_into().unwrap_or_default());
    let size = u64::from(size);
    let (Some(start), Some(bytes)) = (first.checked_mul(size), u64::from(get32(desc, count_at)).checked_mul(size)) else {
        return Outcome::Kept(records);
    };
    // the PC view starts on 16 bytes and ends on the dword that holds the last byte
    let lead = start % 16;
    let Ok(count) = u32::try_from((bytes + lead).div_ceil(4)) else { return Outcome::Kept(records) };
    records.offset = lead as u32;
    // a raw view of the title's on 16 bytes stays as it is, its row names its dwords
    if title_raw && lead == 0 {
        return Outcome::Raw(records);
    }
    put32(desc, 0, R32_TYPELESS);
    desc[first_at..first_at + 8].copy_from_slice(&((start - lead) / 4).to_le_bytes());
    put32(desc, count_at, count);
    put32(desc, stride_at, 0);
    match kind {
        ViewKind::ShaderResource => put32(desc, 8, DEFAULT_MAPPING),
        // a raw view has no counter
        ViewKind::UnorderedAccess => desc[0x18..0x20].fill(0),
    }
    put32(desc, FLAGS_AT, get32(desc, FLAGS_AT) | FLAG_RAW);
    Outcome::Raw(records)
}

/// Cuts the buffer view `desc` (as it goes to the PC) to a buffer of `width` bytes: elements past
/// the end are left out (the console reads them as zeros, the PC runtime removes the device on a
/// view that does not fit: a raw SRV one dword longer than its
/// buffer). Whether the view still has elements; with none it must become a null view ([`make_raw`]
/// makes the description of one from a view of no elements). A view that is no buffer view, or
/// whose element size is unknown, is left as it is.
pub fn fit(desc: &mut [u8; DESC_BYTES], kind: ViewKind, width: u64) -> bool {
    let (first_at, count_at, stride_at) = kind.fields();
    if get32(desc, 4) != DIMENSION_BUFFER {
        return true;
    }
    let size = if get32(desc, FLAGS_AT) & FLAG_RAW != 0 {
        4
    } else if get32(desc, stride_at) != 0 {
        get32(desc, stride_at)
    } else {
        crate::views_at::format_bytes(get32(desc, 0))
    };
    if size == 0 {
        return true;
    }
    let first = u64::from_le_bytes(desc[first_at..first_at + 8].try_into().unwrap_or_default());
    let fitting = (width / u64::from(size)).saturating_sub(first);
    let count = u64::from(get32(desc, count_at));
    if count > fitting {
        put32(desc, count_at, fitting as u32);
        static CUT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
        if crate::proxy::log_due(&CUT) {
            crate::log::xs_log!("buffer view of {count} elements of {size} bytes from element {first} is cut to the {fitting} that fit its buffer of {width} bytes");
        }
    }
    fitting > 0
}

/// [`make_raw`] on the description of a view the title makes (`None`: no description, the
/// resource's own view), with a log line for a view that stays as it is and for a counter that
/// the raw view drops. `what` names the call for the log.
pub fn convert(desc: Option<&mut [u8; DESC_BYTES]>, kind: ViewKind, has_counter: bool, what: &str) -> Outcome {
    static KEPT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    static COUNTER: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    let Some(desc) = desc else { return Outcome::NotStructured };
    let outcome = make_raw(desc, kind);
    match outcome {
        Outcome::Kept(r) if r.format != 0 && crate::proxy::log_due(&KEPT) => {
            crate::log::xs_log!("{what}: typed buffer view of format {} ({} bytes) and {} elements has no raw form, it stays typed", r.format, r.stride, r.count);
        }
        Outcome::Kept(r) if crate::proxy::log_due(&KEPT) => {
            crate::log::xs_log!("{what}: structured buffer view of stride {} and {} records has no raw form, it stays structured", r.stride, r.count);
        }
        Outcome::Empty => {
            static EMPTY: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
            if crate::proxy::log_due(&EMPTY) {
                crate::log::xs_log!("{what}: buffer view of no elements becomes a null view (the PC runtime removes the device on it)");
            }
        }
        Outcome::Raw(r) if has_counter && r.stride != 0 && crate::proxy::log_due(&COUNTER) => {
            crate::log::xs_log!("{what}: structured buffer view of stride {} with a counter resource made raw, the counter is left out", r.stride);
        }
        _ => {}
    }
    outcome
}

/// How the channels of a format are stored: their bits, from the lowest, and their kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    Uint,
    Sint,
    Unorm,
    Snorm,
    Float,
}

fn channels(format: u32) -> Option<(&'static [u32], Channel)> {
    use Channel::{Float, Sint, Snorm, Uint, Unorm};
    Some(match format {
        1 | 3 => (&[32; 4], Uint),
        2 => (&[32; 4], Float),
        4 => (&[32; 4], Sint),
        5 | 7 => (&[32; 3], Uint),
        6 => (&[32; 3], Float),
        8 => (&[32; 3], Sint),
        9 | 12 => (&[16; 4], Uint),
        10 => (&[16; 4], Float),
        11 => (&[16; 4], Unorm),
        13 => (&[16; 4], Snorm),
        14 => (&[16; 4], Sint),
        15 | 17 => (&[32; 2], Uint),
        16 => (&[32; 2], Float),
        18 => (&[32; 2], Sint),
        23 | 25 => (&[10, 10, 10, 2], Uint),
        24 => (&[10, 10, 10, 2], Unorm),
        27 | 30 => (&[8; 4], Uint),
        28 | 29 => (&[8; 4], Unorm),
        31 => (&[8; 4], Snorm),
        32 => (&[8; 4], Sint),
        33 | 36 => (&[16; 2], Uint),
        34 => (&[16; 2], Float),
        35 => (&[16; 2], Unorm),
        37 => (&[16; 2], Snorm),
        38 => (&[16; 2], Sint),
        39 | 42 => (&[32], Uint),
        41 => (&[32], Float),
        43 => (&[32], Sint),
        // elements of 1 or 2 bytes, the formats the translated programs decode
        49 => (&[8; 2], Unorm),
        50 => (&[8; 2], Uint),
        51 => (&[8; 2], Snorm),
        52 => (&[8; 2], Sint),
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
    })
}

/// `f` as a half float, rounded half away from zero.
fn half_bits(f: f32) -> u32 {
    let x = f.to_bits();
    let sign = (x >> 16) & 0x8000;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e) as u32;
        return sign | ((m + 0x1000) >> 13);
    }
    sign | (((e as u32) << 10) + ((man + 0x1000) >> 13)).min(0x7C00)
}

/// One channel of `bits` bits of a clear: from the title's uint value (its low bits, as a typed
/// clear stores them) or float value (converted as the format stores it).
fn channel_bits(bits: u32, kind: Channel, value: ClearValue) -> u32 {
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let v = match value {
        ClearValue::Uint(u) => u,
        ClearValue::Float(f) => match kind {
            Channel::Float if bits == 32 => f.to_bits(),
            Channel::Float => half_bits(f),
            Channel::Unorm => (f.clamp(0.0, 1.0) * mask as f32).round() as u32,
            Channel::Snorm => {
                let max = (mask >> 1) as f32;
                ((f.clamp(-1.0, 1.0) * max).round() as i32) as u32
            }
            Channel::Uint => f as u32,
            Channel::Sint => (f as i32) as u32,
        },
    };
    v & mask
}

/// One value of a clear.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    Uint(u32),
    Float(f32),
}

/// What a clear of a view in a slot with `row` (its view shadow row) is on PC.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clear {
    /// The view is the title's: the clear as it is.
    AsIs,
    /// A view made raw: a uint clear of this word (a raw view clears with the first value).
    Words(u32),
    /// A view made raw whose clear no single word gives (elements of different dwords, a float
    /// clear of a structured view, a format without a layout here).
    Unsupported,
}

/// The clear of a view whose view shadow row is `row`, with the four values of the title's clear.
pub fn clear_words(row: [u32; 4], values: [ClearValue; 4]) -> Clear {
    let [stride, _, format, shape] = row;
    // views the layer left alone, raw views of the title's, images
    if stride == 0 || shape >> 28 != 0 {
        return Clear::AsIs;
    }
    if format == 0 {
        // a raw view from a structured one: the first value, as on a structured view
        return match values[0] {
            ClearValue::Uint(u) => Clear::Words(u),
            ClearValue::Float(_) => Clear::Unsupported,
        };
    }
    let Some((bits, kind)) = channels(format) else { return Clear::Unsupported };
    let mut element = 0u128;
    let mut at = 0;
    for (i, &b) in bits.iter().enumerate() {
        element |= u128::from(channel_bits(b, kind, values[i])) << at;
        at += b;
    }
    // an element of 1 or 2 bytes fills the word as often as it fits
    if at < 32 {
        let e = element as u32;
        return Clear::Words(if at == 8 { e * 0x0101_0101 } else { e | e << 16 });
    }
    let first = element as u32;
    for i in 1..at / 32 {
        if (element >> (32 * i)) as u32 != first {
            return Clear::Unsupported;
        }
    }
    Clear::Words(first)
}

/// The rectangles (left, top, right, bottom) of a clear of a view whose view shadow row is `row`,
/// when they differ from the title's `rects`: a view made raw counts dwords from the start of the
/// PC view, the title's rectangles count its elements from its first one. With no rectangles the
/// clear covers the title's elements only, not the dwords the PC view has before them.
pub fn clear_rects(row: [u32; 4], rects: &[[i32; 4]]) -> Rects {
    let [stride, count, _, shape] = row;
    if shape >> 28 != 0 || (stride == 0 && shape == 0) {
        return Rects::AsIs;
    }
    // a raw view of the title's counts dwords
    let size = i64::from(if stride == 0 { 4 } else { stride });
    let lead = i64::from(shape & 0xF);
    if size == 4 && lead == 0 {
        return Rects::AsIs;
    }
    // the dwords of the title's elements `l` to `r`; elements of 1 or 2 bytes that do not fill
    // whole dwords share them with bytes the clear must not change
    let dwords = |l: i32, r: i32| -> Option<(i32, i32)> {
        let (from, to) = (lead + i64::from(l) * size, lead + i64::from(r) * size);
        if from % 4 != 0 || to % 4 != 0 {
            return None;
        }
        Some((i32::try_from(from / 4).ok()?, i32::try_from(to / 4).ok()?))
    };
    let whole = [[0, 0, i32::try_from(count).unwrap_or(i32::MAX), 1]];
    let title = if rects.is_empty() { &whole[..] } else { rects };
    title
        .iter()
        .map(|&[l, t, r, b]| dwords(l, r).map(|(l, r)| [l, t, r, b]))
        .collect::<Option<Vec<_>>>()
        .map_or(Rects::Unaligned, Rects::Moved)
}

/// The rectangles of a clear on PC ([`clear_rects`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rects {
    /// The title's rectangles as they are.
    AsIs,
    /// These rectangles, in dwords of the PC view.
    Moved(Vec<[i32; 4]>),
    /// A rectangle of elements of 1 or 2 bytes that starts or ends inside a dword: no clear of the
    /// raw view gives it.
    Unaligned,
}

#[cfg(test)]
mod tests {
    use super::{Clear, ClearValue, DESC_BYTES, Outcome, Records, Rects, ViewKind, clear_rects, clear_words, half_bits, make_raw};

    #[test]
    fn the_rectangles_of_a_clear_count_dwords_of_the_pc_view() {
        assert_eq!(clear_rects([0; 4], &[]), Rects::AsIs, "a view the layer left alone");
        assert_eq!(clear_rects([4, 8, 42, 0], &[]), Rects::AsIs, "elements of a dword and no offset");
        assert_eq!(clear_rects([4, 8, 42, 4], &[]), Rects::Moved(vec![[1, 0, 9, 1]]));
        assert_eq!(clear_rects([16, 2, 2, 0], &[[1, 0, 2, 1]]), Rects::Moved(vec![[4, 0, 8, 1]]));
        assert_eq!(clear_rects([0, 10, 0, 12], &[]), Rects::Moved(vec![[3, 0, 13, 1]]), "a raw view of the title's that starts lower");
        assert_eq!(clear_rects([0, 0, 0x000F_000F, 9 << 28], &[]), Rects::AsIs, "an image");
    }

    #[test]
    fn a_clear_of_small_elements_covers_whole_dwords_or_is_refused() {
        // R16_UINT, 8 elements from byte 4 of the PC view: bytes 4 to 20, dwords 1 to 5
        assert_eq!(clear_rects([2, 8, 57, 4], &[]), Rects::Moved(vec![[1, 0, 5, 1]]));
        assert_eq!(clear_rects([2, 8, 57, 4], &[[2, 0, 6, 1]]), Rects::Moved(vec![[2, 0, 4, 1]]));
        assert_eq!(clear_rects([2, 8, 57, 4], &[[1, 0, 6, 1]]), Rects::Unaligned, "element 1 shares its dword with element 0");
        // R8_UNORM, 6 elements from byte 2: the first and the last dword are shared
        assert_eq!(clear_rects([1, 6, 61, 2], &[]), Rects::Unaligned);
        // the word repeats the element
        let u = |v: [u32; 4]| v.map(ClearValue::Uint);
        assert_eq!(clear_words([2, 8, 57, 4], u([0x1_2345, 0, 0, 0])), Clear::Words(0x2345_2345));
        assert_eq!(clear_words([1, 6, 62, 2], u([0xAB, 0, 0, 0])), Clear::Words(0xABAB_ABAB));
        assert_eq!(clear_words([2, 8, 49, 0], [ClearValue::Float(1.0), ClearValue::Float(0.0), ClearValue::Float(0.0), ClearValue::Float(0.0)]), Clear::Words(0x00FF_00FF), "R8G8_UNORM");
        assert_eq!(clear_words([2, 8, 54, 0], [ClearValue::Float(1.0); 4]), Clear::Words(0x3C00_3C00), "R16_FLOAT");
    }

    #[test]
    fn a_typed_view_of_one_or_two_byte_elements_becomes_raw_with_the_byte_offset() {
        // R16_UINT, elements 1 to 8: bytes 2 to 18, the PC view dwords 0 to 5 and offset 2
        let mut d = srv(57, 1, 8, 0, 0);
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { stride: 2, count: 8, format: 57, offset: 2 }));
        assert_eq!(d, srv(39, 0, 5, 0, 1));
        // R8_UNORM, elements 21 to 23: bytes 21 to 24, view from byte 16, offset 5, 2 dwords
        let mut d = uav(61, 21, 3, 0, 0);
        assert_eq!(make_raw(&mut d, ViewKind::UnorderedAccess), Outcome::Raw(Records { stride: 1, count: 3, format: 61, offset: 5 }));
        let mut want = uav(39, 4, 2, 0, 0);
        want[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(d, want);
        // R8G8_SNORM from byte 32: on 16 bytes, offset 0
        let mut d = srv(51, 16, 4, 0, 0);
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { stride: 2, count: 4, format: 51, offset: 0 }));
        assert_eq!(d, srv(39, 8, 2, 0, 1));
        // a format the programs do not decode (B5G6R5) keeps its typed view
        let mut d = srv(85, 0, 8, 0, 0);
        let before = d;
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Kept(Records { stride: 2, count: 8, format: 85, offset: 0 }));
        assert_eq!(d, before);
    }

    fn srv(format: u32, first: u64, count: u32, stride: u32, flags: u32) -> [u8; DESC_BYTES] {
        let mut d = [0u8; DESC_BYTES];
        d[0..4].copy_from_slice(&format.to_le_bytes());
        d[4..8].copy_from_slice(&1u32.to_le_bytes());
        d[8..12].copy_from_slice(&0x1688u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&first.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&count.to_le_bytes());
        d[0x1C..0x20].copy_from_slice(&stride.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&flags.to_le_bytes());
        d
    }

    fn uav(format: u32, first: u64, count: u32, stride: u32, counter_offset: u64) -> [u8; DESC_BYTES] {
        let mut d = [0u8; DESC_BYTES];
        d[0..4].copy_from_slice(&format.to_le_bytes());
        d[4..8].copy_from_slice(&1u32.to_le_bytes());
        d[0x08..0x10].copy_from_slice(&first.to_le_bytes());
        d[0x10..0x14].copy_from_slice(&count.to_le_bytes());
        d[0x14..0x18].copy_from_slice(&stride.to_le_bytes());
        d[0x18..0x20].copy_from_slice(&counter_offset.to_le_bytes());
        d
    }

    #[test]
    fn a_structured_shader_resource_view_becomes_the_raw_view_of_the_same_bytes() {
        let mut d = srv(0, 3, 10, 16, 0);
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { stride: 16, count: 10, format: 0, offset: 0 }));
        assert_eq!(d, srv(39, 12, 40, 0, 1), "first byte 48 and 160 bytes, as before");
    }

    #[test]
    fn a_structured_unordered_access_view_becomes_raw_and_loses_its_counter() {
        // first byte 24: the PC view starts at 16, 8 bytes earlier and 2 dwords longer
        let mut d = uav(0, 2, 5, 12, 4096);
        assert_eq!(make_raw(&mut d, ViewKind::UnorderedAccess), Outcome::Raw(Records { stride: 12, count: 5, format: 0, offset: 8 }));
        let mut want = uav(39, 4, 17, 0, 0);
        want[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(d, want);
    }

    #[test]
    fn a_typed_view_becomes_the_raw_view_of_its_elements_with_the_default_mapping() {
        // R32G32B32A32_FLOAT: elements 2 and 3 are bytes 32 to 64
        let mut d = srv(2, 2, 2, 0, 0);
        d[8..12].copy_from_slice(&0x1234u32.to_le_bytes());
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { stride: 16, count: 2, format: 2, offset: 0 }));
        assert_eq!(d, srv(39, 8, 8, 0, 1));
        let mut d = uav(42, 5, 7, 0, 0);
        assert_eq!(make_raw(&mut d, ViewKind::UnorderedAccess), Outcome::Raw(Records { stride: 4, count: 7, format: 42, offset: 4 }));
        let mut want = uav(39, 4, 8, 0, 0);
        want[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(d, want);
    }

    #[test]
    fn raw_and_texture_views_are_left_alone() {
        for mut d in [srv(39, 0, 64, 0, 1), srv(39, 8, 64, 0, 1)] {
            let before = d;
            assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { count: 64, ..Records::default() }), "a raw view of the title's on 16 bytes: its dwords");
            assert_eq!(d, before);
        }
        let mut d = srv(0, 0, 64, 0, 0);
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::NotStructured);
        let mut texture = srv(28, 0, 1, 16, 0);
        texture[4..8].copy_from_slice(&4u32.to_le_bytes());
        assert_eq!(make_raw(&mut texture, ViewKind::ShaderResource), Outcome::NotStructured);
    }

    #[test]
    fn a_buffer_view_longer_than_its_buffer_is_cut_and_one_past_its_end_has_no_elements() {
        use super::fit;
        // the raw SRV of run early-debug: 2957493 dwords on a buffer of 2957492
        let mut d = srv(39, 0, 2_957_493, 0, 1);
        assert!(fit(&mut d, ViewKind::ShaderResource, 2_957_492 * 4));
        assert_eq!(d, srv(39, 0, 2_957_492, 0, 1));
        // a structured view keeps the records that fit whole
        let mut s = uav(0, 2, 10, 12, 0);
        assert!(fit(&mut s, ViewKind::UnorderedAccess, 100));
        assert_eq!(s, uav(0, 2, 6, 12, 0), "8 records of 12 bytes fit, 6 of them from record 2");
        // a view that fits stays, one that starts past the end has no element left
        let mut ok = srv(39, 4, 4, 0, 1);
        assert!(fit(&mut ok, ViewKind::ShaderResource, 32));
        assert_eq!(ok, srv(39, 4, 4, 0, 1));
        let mut past = srv(39, 9, 4, 0, 1);
        assert!(!fit(&mut past, ViewKind::ShaderResource, 32));
        assert_eq!(make_raw(&mut past, ViewKind::ShaderResource), Outcome::Empty);
    }

    #[test]
    fn a_buffer_view_of_no_elements_becomes_a_null_view() {
        let mut d = uav(39, 10_394_464, 0, 0, 0);
        d[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(make_raw(&mut d, ViewKind::UnorderedAccess), Outcome::Empty);
        assert_eq!(d, uav(42, 0, 1, 0, 0), "typed, one element, no raw flag");
        let mut s = srv(0, 5, 0, 16, 0);
        assert_eq!(make_raw(&mut s, ViewKind::ShaderResource), Outcome::Empty);
        assert_eq!(s, srv(42, 0, 1, 0, 0));
    }

    #[test]
    fn a_raw_view_of_the_title_that_does_not_start_on_16_bytes_starts_lower() {
        let mut d = srv(39, 7, 10, 0, 1);
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Raw(Records { count: 10, offset: 12, ..Records::default() }));
        assert_eq!(d, srv(39, 4, 13, 0, 1));
    }

    #[test]
    fn a_stride_that_is_not_a_multiple_of_four_or_too_many_dwords_keeps_the_structured_view() {
        let mut d = srv(0, 1, 7, 6, 0);
        let before = d;
        assert_eq!(make_raw(&mut d, ViewKind::ShaderResource), Outcome::Kept(Records { stride: 6, count: 7, format: 0, offset: 0 }));
        assert_eq!(d, before);
        let mut big = srv(0, 0, u32::MAX, 2048, 0);
        assert_eq!(make_raw(&mut big, ViewKind::ShaderResource), Outcome::Kept(Records { stride: 2048, count: u32::MAX, format: 0, offset: 0 }));
    }

    #[test]
    fn clears_of_views_made_raw_become_one_word_or_are_refused() {
        let u = |v: [u32; 4]| v.map(ClearValue::Uint);
        let f = |v: [f32; 4]| v.map(ClearValue::Float);
        // views the layer left alone and images clear as they are
        assert_eq!(clear_words([0; 4], u([1, 2, 3, 4])), Clear::AsIs);
        assert_eq!(clear_words([0, 0, 0x000F_000F, 9 << 28], f([1.0; 4])), Clear::AsIs);
        // a structured view: the first value, no float clear
        assert_eq!(clear_words([16, 4, 0, 0], u([7, 8, 9, 10])), Clear::Words(7));
        assert_eq!(clear_words([16, 4, 0, 0], f([1.0; 4])), Clear::Unsupported);
        // R8G8B8A8_UNORM: channels packed from the lowest byte
        assert_eq!(clear_words([4, 4, 28, 0], f([1.0, 0.5, 0.0, 1.0])), Clear::Words(0xFF00_80FF));
        assert_eq!(clear_words([4, 4, 30, 0], u([0x101, 2, 3, 4])), Clear::Words(0x0403_0201), "a uint clear keeps the low bits");
        // R16G16_FLOAT and R16G16_UNORM
        assert_eq!(clear_words([4, 4, 34, 0], f([1.0, -2.0, 0.0, 0.0])), Clear::Words(0xC000_3C00));
        assert_eq!(clear_words([4, 4, 35, 0], f([1.0, 0.0, 0.0, 0.0])), Clear::Words(0x0000_FFFF));
        // R32G32B32A32: one word only when the four are equal
        assert_eq!(clear_words([16, 4, 2, 0], f([2.0; 4])), Clear::Words(0x4000_0000));
        assert_eq!(clear_words([16, 4, 2, 0], f([1.0, 2.0, 3.0, 4.0])), Clear::Unsupported);
        assert_eq!(clear_words([4, 4, 42, 4], u([5, 0, 0, 0])), Clear::Words(5), "an offset does not change the word");
        assert_eq!((half_bits(65504.0), half_bits(1e6), half_bits(f32::NAN) & 0x7C00), (0x7BFF, 0x7C00, 0x7C00));
    }
}
