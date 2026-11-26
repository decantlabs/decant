//! Reads and patches the root signature (RTS0 part) of a DXBC container. The Rust form of the
//! former `src/gdk/dxbc_rts0.h`.
//!
//! Why: the console root signatures carry flag bits the PC runtime does not define
//! (`XboxBvhBuilder.dll` ships 52 DXIL containers whose RTS0 v2 flags are `0x03000000`).
//! [`strip_unknown_flags`] clears those bits in a copy and rewrites the container checksum, so a PC
//! D3D12 runtime (Windows or vkd3d-proton) can parse it.
//!
//! Layouts used here are the public ones:
//!
//! * DXBC container: "DXBC", 16 byte checksum, u32 1, u32 total size, u32 part count, u32 part
//!   offsets. Each part is a FourCC, a u32 size, then the data.
//! * RTS0 part: u32 version, u32 parameter count, u32 parameter offset, u32 static sampler count,
//!   u32 static sampler offset, u32 flags (`D3D12_ROOT_SIGNATURE_FLAGS`).
//! * The checksum is MD5 with DXBC's own padding, over the container from byte 20 to the end.
//!
//! Written from those descriptions, with no code taken from another project.

/// Every `D3D12_ROOT_SIGNATURE_FLAGS` value the public d3d12.h defines (up to
/// `D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED` = 0x800). Anything above is console
/// only and unknown to a PC runtime.
pub const PUBLIC_FLAGS: u32 = 0x0000_0FFF;

/// A little endian u32 at `at`.
pub fn rd32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Writes a little endian u32 at `at`.
pub fn wr32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// Offset of the RTS0 flags word inside a DXBC container, or `None` when the blob is not a
/// well formed container holding an RTS0 part. Every offset is bounds checked.
pub fn flags_offset(blob: &[u8]) -> Option<usize> {
    rts0_part(blob).map(|(start, _)| start + 20)
}

/// The class of a register a root signature or a shader names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegisterClass {
    /// `t` registers.
    Srv,
    /// `u` registers.
    Uav,
    /// `b` registers.
    Cbv,
    /// `s` registers.
    Sampler,
}

/// One descriptor range of a root descriptor table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootRange {
    /// What the range holds.
    pub class: RegisterClass,
    /// Number of descriptors, `u32::MAX` for an unbounded range.
    pub count: u32,
    /// First register of the range.
    pub base: u32,
    /// Register space.
    pub space: u32,
    /// `D3D12_DESCRIPTOR_RANGE_FLAGS` (0 in a 1.0 blob).
    pub flags: u32,
    /// Offset in descriptors from the start of the table, `OFFSET_APPEND` resolved from the range
    /// before (`u32::MAX` when that one is unbounded).
    pub offset: u32,
}

/// What one root parameter is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootParamKind {
    /// A descriptor table.
    Table(Vec<RootRange>),
    /// Root constants: `count` 32 bit values at constant buffer `register`.
    Constants { register: u32, space: u32, count: u32 },
    /// A root descriptor of the given class (`Cbv`, `Srv` or `Uav`).
    Descriptor { class: RegisterClass, register: u32, space: u32 },
}

/// One root parameter with its shader visibility (`D3D12_SHADER_VISIBILITY`, 0 is all stages).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootParam {
    /// The parameter itself.
    pub kind: RootParamKind,
    /// Shader visibility.
    pub visibility: u32,
}

/// A static sampler reduced to what coverage needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticSampler {
    /// Shader register (`s`).
    pub register: u32,
    /// Register space.
    pub space: u32,
    /// Shader visibility.
    pub visibility: u32,
}

/// The parsed layout of a root signature: what the root signature has in each register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootLayout {
    /// RTS0 version word: 1 for root signature 1.0, 2 for 1.1, 3 for 1.2.
    pub version: u32,
    /// `D3D12_ROOT_SIGNATURE_FLAGS`.
    pub flags: u32,
    /// Parameters in root index order.
    pub params: Vec<RootParam>,
    /// Static samplers.
    pub static_samplers: Vec<StaticSampler>,
}

/// A bounds checked little endian u32 of `data` at `at`.
fn get32(data: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    data.get(at..end).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parses the root signature of a DXBC container (RTS0 versions 1, 2 and 3, that is root signature
/// 1.0, 1.1 and 1.2) into a [`RootLayout`]. `None` for anything malformed or unknown: a bad offset,
/// a count beyond the part, an unknown parameter or range type. Never reads outside `blob`.
///
/// Layout (offsets inside the RTS0 data): header of six u32 (version, parameter count, parameter
/// offset, static sampler count, static sampler offset, flags). A parameter is three u32 (type,
/// visibility, offset of its data). Type 0 data is a table (range count, range offset), type 1
/// constants (register, space, value count), types 2 to 4 a root descriptor (register, space and,
/// from 1.1 on, flags). A range is five u32 in 1.0 (type, count, base register, space, table
/// offset) and six from 1.1 on (flags after the space). A static sampler is 13 u32 in 1.0 and 1.1
/// and 14 in 1.2, its register, space and visibility at words 10, 11 and 12.
pub fn parse_root_layout(blob: &[u8]) -> Option<RootLayout> {
    let (start, len) = rts0_part(blob)?;
    let data = &blob[start..start + len];
    let version = get32(data, 0)?;
    if !(1..=3).contains(&version) {
        return None;
    }
    let param_count = get32(data, 4)? as usize;
    let param_off = get32(data, 8)? as usize;
    let sampler_count = get32(data, 12)? as usize;
    let sampler_off = get32(data, 16)? as usize;
    let flags = get32(data, 20)?;
    if param_count > len / 12 || sampler_count > len / 52 {
        return None;
    }
    let mut params = Vec::with_capacity(param_count);
    for i in 0..param_count {
        let at = param_off.checked_add(i * 12)?;
        let (kind, visibility, body) = (get32(data, at)?, get32(data, at + 4)?, get32(data, at + 8)? as usize);
        let kind = match kind {
            0 => {
                let count = get32(data, body)? as usize;
                let off = get32(data, body.checked_add(4)?)? as usize;
                let stride = if version == 1 { 20 } else { 24 };
                if count > len / stride {
                    return None;
                }
                let mut ranges: Vec<RootRange> = Vec::with_capacity(count);
                for r in 0..count {
                    let at = off.checked_add(r * stride)?;
                    let class = match get32(data, at)? {
                        0 => RegisterClass::Srv,
                        1 => RegisterClass::Uav,
                        2 => RegisterClass::Cbv,
                        3 => RegisterClass::Sampler,
                        _ => return None,
                    };
                    let flags = if version == 1 { 0 } else { get32(data, at + 16)? };
                    let range_count = get32(data, at + 4)?;
                    // D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND continues after the range before it.
                    let offset = match get32(data, at + if version == 1 { 16 } else { 20 })? {
                        u32::MAX => ranges.last().map_or(0, |p| if p.count == u32::MAX || p.offset == u32::MAX { u32::MAX } else { p.offset.saturating_add(p.count) }),
                        o => o,
                    };
                    ranges.push(RootRange { class, count: range_count, base: get32(data, at + 8)?, space: get32(data, at + 12)?, flags, offset });
                }
                RootParamKind::Table(ranges)
            }
            1 => RootParamKind::Constants { register: get32(data, body)?, space: get32(data, body + 4)?, count: get32(data, body + 8)? },
            2..=4 => {
                let class = match kind {
                    2 => RegisterClass::Cbv,
                    3 => RegisterClass::Srv,
                    _ => RegisterClass::Uav,
                };
                RootParamKind::Descriptor { class, register: get32(data, body)?, space: get32(data, body + 4)? }
            }
            _ => return None,
        };
        params.push(RootParam { kind, visibility });
    }
    let stride = if version == 3 { 56 } else { 52 };
    let mut static_samplers = Vec::with_capacity(sampler_count);
    for i in 0..sampler_count {
        let at = sampler_off.checked_add(i * stride)?;
        static_samplers.push(StaticSampler { register: get32(data, at + 40)?, space: get32(data, at + 44)?, visibility: get32(data, at + 48)? });
    }
    Some(RootLayout { version, flags, params, static_samplers })
}

/// Start and length of the data of the RTS0 part inside a DXBC container, or `None` when the blob is
/// not a well formed container holding an RTS0 part of at least the 24 byte header. Every offset is
/// bounds checked.
fn rts0_part(blob: &[u8]) -> Option<(usize, usize)> {
    if blob.len() < 32 || &blob[..4] != b"DXBC" {
        return None;
    }
    let (total, parts) = (rd32(blob, 24) as usize, rd32(blob, 28) as usize);
    if total > blob.len() || total < 32 || parts > (total - 32) / 4 {
        return None;
    }
    for i in 0..parts {
        let off = rd32(blob, 32 + 4 * i) as usize;
        if off > total || total - off < 8 {
            return None;
        }
        let len = rd32(blob, off + 4) as usize;
        if len > total - off - 8 {
            return None;
        }
        if &blob[off..off + 4] == b"RTS0" {
            return (len >= 24).then_some((off + 8, len));
        }
    }
    None
}

/// MD5 round constants (RFC 1321).
const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1,
    0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453,
    0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942,
    0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d,
    0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// MD5 rotation amounts.
const R: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4,
    11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// One MD5 block.
fn md5_block(state: &mut [u32; 4], block: &[u8]) {
    let mut m = [0u32; 16];
    for (i, word) in m.iter_mut().enumerate() {
        *word = rd32(block, 4 * i);
    }
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64 {
        let (f, g) = match i {
            0..=15 => ((b & c) | (!b & d), i),
            16..=31 => ((d & b) | (!d & c), (5 * i + 1) & 15),
            32..=47 => (b ^ c ^ d, (3 * i + 5) & 15),
            _ => (c ^ (b | !d), (7 * i) & 15),
        };
        let t = d;
        d = c;
        c = b;
        b = b.wrapping_add(a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]).rotate_left(R[i]));
        a = t;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// DXBC checksum of a container: MD5 state over bytes `[20, total)`, with DXBC's final block layout
/// instead of MD5's. The bit count goes in the first word of the last block and `(bits >> 2) | 1`
/// in its last word.
pub fn dxbc_checksum(blob: &[u8], total: usize) -> [u32; 4] {
    let data = &blob[20..total];
    let n = data.len();
    let full = n & !63;
    let rest = n - full;
    let mut state = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let bits = (n as u32).wrapping_mul(8);
    let mut block = [0u8; 64];
    for chunk in data[..full].as_chunks::<64>().0 {
        md5_block(&mut state, chunk);
    }
    if rest >= 56 {
        block[..rest].copy_from_slice(&data[full..]);
        block[rest] = 0x80;
        md5_block(&mut state, &block);
        block = [0u8; 64];
    } else {
        block[4..4 + rest].copy_from_slice(&data[full..]);
        block[4 + rest] = 0x80;
    }
    wr32(&mut block, 0, bits);
    wr32(&mut block, 60, (bits >> 2) | 1);
    md5_block(&mut state, &block);
    state
}

/// Clears the console only RTS0 flag bits of a container in place and rewrites its checksum.
/// Returns the bits that were cleared (0 when there was nothing to do or no RTS0 part).
pub fn strip_unknown_flags(blob: &mut [u8]) -> u32 {
    let Some(at) = flags_offset(blob) else {
        return 0;
    };
    let flags = rd32(blob, at);
    let extra = flags & !PUBLIC_FLAGS;
    if extra == 0 {
        return 0;
    }
    wr32(blob, at, flags & PUBLIC_FLAGS);
    let sum = dxbc_checksum(blob, rd32(blob, 24) as usize);
    for (i, word) in sum.iter().enumerate() {
        wr32(blob, 4 + 4 * i, *word);
    }
    extra
}

/// Sets `flag` (a PC defined bit, `PUBLIC_FLAGS`) in the RTS0 flags word of a container in place
/// and rewrites its checksum. Returns whether the bit was not already set (and so the container
/// changed). No RTS0 part, or the bit already set, leaves the container untouched and returns
/// `false`.
///
/// `D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` (0x400) has to reach the PC
/// runtime for a shader that indexes the descriptor heap directly (SM 6.6 `createHandleFromHeap`)
/// to get the binding it needs; the console compiler does not always set it.
pub fn add_public_flag(blob: &mut [u8], flag: u32) -> bool {
    let Some(at) = flags_offset(blob) else {
        return false;
    };
    let flags = rd32(blob, at);
    if flags & flag != 0 {
        return false;
    }
    wr32(blob, at, flags | flag);
    let sum = dxbc_checksum(blob, rd32(blob, 24) as usize);
    for (i, word) in sum.iter().enumerate() {
        wr32(blob, 4 + 4 * i, *word);
    }
    true
}

/// Register of the user data mirror, a root constant buffer view `b1` of [`MIRROR_SPACE`] (the
/// contract of `decant-dxil`, `USER_DATA_REGISTER`).
pub const MIRROR_REGISTER: u32 = 1;
/// Register space of the user data mirror (`USER_DATA_SPACE`).
pub const MIRROR_SPACE: u32 = 15;
/// Register of the heap map in the mirror's space, a root shader resource view `t0`: one dword per
/// page of the bound view heap (`title_heap`, "page map"), through which a directly translated
/// shader turns a console heap index into a host one (`decant-shader`, `MirrorAbi`).
pub const HEAP_MAP_REGISTER: u32 = 0;
/// Register of the view shadow in the mirror's space, a root shader resource view `t1`: record stride
/// and count per host slot of the bound view heap (`view_shadow`), for structured buffers read
/// through raw views.
pub const VIEW_SHADOW_REGISTER: u32 = 1;
/// The most dwords a root signature may use (`D3D12_MAX_ROOT_COST`).
pub const MAX_ROOT_DWORDS: u32 = 64;

/// Register space of the device memory window (`u0` the window UAV, `b0` the base address).
pub const WINDOW_SPACE: u32 = 14;
/// Register space of the translator's own buffers (`t0` the page table, `u0` to `u2` the others).
pub const SERVICE_SPACE: u32 = 13;

impl RootLayout {
    /// The index of a root descriptor `class` `register` of `space` visible to all stages.
    pub fn descriptor_index(&self, class: RegisterClass, register: u32, space: u32) -> Option<usize> {
        self.params.iter().position(|p| p.visibility == 0 && p.kind == RootParamKind::Descriptor { class, register, space })
    }

    /// The index of the user data mirror parameter: a root CBV `b1` of space 15 visible to all stages.
    pub fn mirror_index(&self) -> Option<usize> {
        self.descriptor_index(RegisterClass::Cbv, MIRROR_REGISTER, MIRROR_SPACE)
    }

    /// The index of the heap map parameter: a root SRV `t0` of space 15 visible to all stages.
    pub fn heap_map_index(&self) -> Option<usize> {
        self.descriptor_index(RegisterClass::Srv, HEAP_MAP_REGISTER, MIRROR_SPACE)
    }

    /// The index of the view shadow parameter: a root SRV `t1` of space 15 visible to all stages.
    pub fn view_shadow_index(&self) -> Option<usize> {
        self.descriptor_index(RegisterClass::Srv, VIEW_SHADOW_REGISTER, MIRROR_SPACE)
    }

    /// The index of the window parameter: a root UAV `u0` of space 14 visible to all stages.
    pub fn window_uav_index(&self) -> Option<usize> {
        self.descriptor_index(RegisterClass::Uav, 0, WINDOW_SPACE)
    }

    /// The index of the base address parameter of the window: a root CBV `b0` of space 14 visible to
    /// all stages (the layer makes a CBV, root constants at that register are not looked for).
    pub fn window_base_index(&self) -> Option<usize> {
        self.descriptor_index(RegisterClass::Cbv, 0, WINDOW_SPACE)
    }

    /// The root signature as the title made it: without the root descriptors the layer appends
    /// (user data mirror, device memory window, service buffers in spaces 13 to 15). They are
    /// appended after the title's parameters, so the indices of the title's parameters stay.
    pub fn without_layer_params(&self) -> RootLayout {
        let mut out = self.clone();
        while let Some(last) = out.params.last() {
            match last.kind {
                RootParamKind::Descriptor { space, .. } if last.visibility == 0 && (SERVICE_SPACE..=MIRROR_SPACE).contains(&space) => {
                    out.params.pop();
                }
                _ => break,
            }
        }
        out
    }

    /// The indices of the service parameters: page table `t0`, then `u0`, `u1` and `u2` of space 13.
    pub fn service_indices(&self) -> [Option<usize>; 4] {
        [
            self.descriptor_index(RegisterClass::Srv, 0, SERVICE_SPACE),
            self.descriptor_index(RegisterClass::Uav, 0, SERVICE_SPACE),
            self.descriptor_index(RegisterClass::Uav, 1, SERVICE_SPACE),
            self.descriptor_index(RegisterClass::Uav, 2, SERVICE_SPACE),
        ]
    }

    /// The dwords the parameters take in the root signature: a table one, root constants their count,
    /// a root descriptor two.
    pub fn dwords(&self) -> u32 {
        self.params
            .iter()
            .map(|p| match &p.kind {
                RootParamKind::Table(_) => 1,
                RootParamKind::Constants { count, .. } => *count,
                RootParamKind::Descriptor { .. } => 2,
            })
            .fold(0u32, u32::saturating_add)
    }
}

/// Why [`append_abi_params`] left the root signature as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendRefusal {
    /// The blob is not a container with a root signature this code parses.
    NotParsed,
    /// The root signature has every wanted group already.
    Present,
    /// Not even the first wanted group fits: the title's parameters and it are more than 64 dwords.
    TooLarge { dwords: u32 },
}

/// Which parameter groups [`append_abi_params`] should add.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbiWant {
    /// The user data mirror, a root CBV (two dwords).
    pub mirror: bool,
    /// The device memory window and the translator's service buffers (4 and 8 dwords).
    pub window: bool,
    /// The heap map and the view shadow, a root SRV each (two dwords each). Only added to a root
    /// signature with the mirror: a shader without the mirror reads neither.
    pub heap_map: bool,
    pub view_shadow: bool,
}

/// Which groups [`append_abi_params`] added. A group that does not fit is left out, and the service
/// group is only added together with the window (a shader that reads the page table forms addresses in it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AbiAdded {
    pub mirror: bool,
    pub window: bool,
    pub service: bool,
    pub heap_map: bool,
    pub view_shadow: bool,
}

/// One root descriptor parameter to append: `kind` is the `D3D12_ROOT_PARAMETER_TYPE` (2 CBV, 3 SRV, 4 UAV).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NewParam {
    kind: u32,
    register: u32,
    space: u32,
}

const CBV: u32 = 2;
const SRV: u32 = 3;
const UAV: u32 = 4;

const MIRROR_PARAMS: [NewParam; 1] = [NewParam { kind: CBV, register: MIRROR_REGISTER, space: MIRROR_SPACE }];
const HEAP_MAP_PARAMS: [NewParam; 1] = [NewParam { kind: SRV, register: HEAP_MAP_REGISTER, space: MIRROR_SPACE }];
const VIEW_SHADOW_PARAMS: [NewParam; 1] = [NewParam { kind: SRV, register: VIEW_SHADOW_REGISTER, space: MIRROR_SPACE }];
const WINDOW_PARAMS: [NewParam; 2] = [NewParam { kind: UAV, register: 0, space: WINDOW_SPACE }, NewParam { kind: CBV, register: 0, space: WINDOW_SPACE }];
const SERVICE_PARAMS: [NewParam; 4] = [
    NewParam { kind: SRV, register: 0, space: SERVICE_SPACE },
    NewParam { kind: UAV, register: 0, space: SERVICE_SPACE },
    NewParam { kind: UAV, register: 1, space: SERVICE_SPACE },
    NewParam { kind: UAV, register: 2, space: SERVICE_SPACE },
];

const MAX_STACK_PARTS: usize = 16;

#[allow(clippy::large_enum_variant)]
enum ContainerParts {
    Stack([(usize, usize); MAX_STACK_PARTS], usize),
    Heap(Vec<(usize, usize)>),
}

impl std::ops::Deref for ContainerParts {
    type Target = [(usize, usize)];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Stack(items, len) => &items[..*len],
            Self::Heap(vec) => vec,
        }
    }
}

/// Start and length of every part of a well formed container: `(offset of the part header, data length)`.
fn container_parts(blob: &[u8]) -> Option<ContainerParts> {
    rts0_part(blob)?;
    let (total, parts) = (rd32(blob, 24) as usize, rd32(blob, 28) as usize);
    if parts <= MAX_STACK_PARTS {
        let mut items = [(0, 0); MAX_STACK_PARTS];
        for (i, slot) in items[..parts].iter_mut().enumerate() {
            let off = rd32(blob, 32 + 4 * i) as usize;
            let len = rd32(blob, off + 4) as usize;
            if len > total - off - 8 {
                return None;
            }
            *slot = (off, len);
        }
        return Some(ContainerParts::Stack(items, parts));
    }
    let mut out = Vec::with_capacity(parts);
    for i in 0..parts {
        let off = rd32(blob, 32 + 4 * i) as usize;
        let len = rd32(blob, off + 4) as usize;
        if len > total - off - 8 {
            return None;
        }
        out.push((off, len));
    }
    Some(ContainerParts::Heap(out))
}

/// The RTS0 data with the root descriptor parameters `add` (visibility all) appended after the existing
/// parameters. The parameter headers go behind the last header, so every offset that points at or past
/// that place moves by 12 bytes per new parameter, and the parameters' own data goes to the end of the
/// part. Version 1.0 root descriptors have two words, 1.1 and 1.2 three (with flags).
fn rts0_with_params(data: &[u8], add: &[NewParam]) -> Option<Vec<u8>> {
    const HEADER: usize = 12;
    let version = get32(data, 0)?;
    let count = get32(data, 4)? as usize;
    let param_off = get32(data, 8)? as usize;
    let sampler_count = get32(data, 12)? as usize;
    let sampler_off = get32(data, 16)?;
    let insert = param_off.checked_add(count.checked_mul(HEADER)?)?;
    if insert > data.len() {
        return None;
    }
    let grow = HEADER * add.len();
    let mut out = Vec::with_capacity(data.len() + grow + 12 * add.len());
    out.extend_from_slice(&data[..insert]);
    out.resize(out.len() + grow, 0);
    out.extend_from_slice(&data[insert..]);
    let shift = |at: u32| if at as usize >= insert { at + grow as u32 } else { at };
    wr32(&mut out, 4, (count + add.len()) as u32);
    if sampler_count > 0 {
        wr32(&mut out, 16, shift(sampler_off));
    }
    for i in 0..count {
        let header = param_off + i * HEADER;
        let body = shift(get32(&out, header + 8)?) as usize;
        wr32(&mut out, header + 8, body as u32);
        if get32(&out, header)? == 0 {
            let ranges = get32(&out, body.checked_add(4)?)?;
            wr32(&mut out, body + 4, shift(ranges));
        }
    }
    for (i, param) in add.iter().enumerate() {
        let body = out.len() as u32;
        let words: &[u32] = if version == 1 { &[param.register, param.space] } else { &[param.register, param.space, 0] };
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        let header = insert + i * HEADER;
        wr32(&mut out, header, param.kind);
        wr32(&mut out, header + 4, 0); // D3D12_SHADER_VISIBILITY_ALL
        wr32(&mut out, header + 8, body);
    }
    Some(out)
}

/// A copy of the container `blob` whose root signature has the groups of `want` appended after the
/// title's parameters (in the order mirror, heap map, view shadow, window, service), so that every title parameter keeps its
/// index. The container is rebuilt with the new part sizes and offsets and a new checksum, and the
/// result parses ([`parse_root_layout`]). Root signature 1.0, 1.1 and 1.2 are handled. A group that is
/// present already is skipped and one that would pass 64 dwords is left out (the result says which
/// were added), so the result always stays within the limit.
pub fn append_abi_params(blob: &[u8], want: AbiWant) -> Result<(Vec<u8>, AbiAdded), AppendRefusal> {
    let layout = parse_root_layout(blob).ok_or(AppendRefusal::NotParsed)?;
    let mut dwords = layout.dwords();
    let mut added = AbiAdded::default();
    let mut add = [NewParam { kind: 0, register: 0, space: 0 }; 9];
    let mut add_len = 0usize;
    let mut first_need = None;
    let mut try_group = |wanted: bool, has: bool, params: &[NewParam], flag: &mut bool| {
        if !wanted || has {
            return;
        }
        let cost = 2 * params.len() as u32;
        first_need.get_or_insert(cost);
        if dwords + cost <= MAX_ROOT_DWORDS {
            dwords += cost;
            *flag = true;
            add[add_len..add_len + params.len()].copy_from_slice(params);
            add_len += params.len();
        }
    };
    try_group(want.mirror, layout.mirror_index().is_some(), &MIRROR_PARAMS, &mut added.mirror);
    let with_mirror = added.mirror || layout.mirror_index().is_some();
    try_group(want.heap_map && with_mirror, layout.heap_map_index().is_some(), &HEAP_MAP_PARAMS, &mut added.heap_map);
    try_group(want.view_shadow && with_mirror, layout.view_shadow_index().is_some(), &VIEW_SHADOW_PARAMS, &mut added.view_shadow);
    let has_window = layout.window_uav_index().is_some() && layout.window_base_index().is_some();
    try_group(want.window, has_window, &WINDOW_PARAMS, &mut added.window);
    if want.window && (added.window || has_window) {
        let has_service = layout.service_indices().iter().all(Option::is_some);
        try_group(true, has_service, &SERVICE_PARAMS, &mut added.service);
    }
    if add_len == 0 {
        return Err(match first_need {
            Some(cost) => AppendRefusal::TooLarge { dwords: layout.dwords().saturating_add(cost) },
            None => AppendRefusal::Present,
        });
    }
    let out = with_new_rts0(blob, |data| rts0_with_params(data, &add[..add_len])).ok_or(AppendRefusal::NotParsed)?;
    Ok((out, added))
}

/// A copy of the container `blob` with its RTS0 data replaced by what `change` makes of it, the
/// other parts kept, and new part offsets, size and checksum. `None` when the container does not
/// parse or `change` gives `None`.
fn with_new_rts0(blob: &[u8], change: impl FnOnce(&[u8]) -> Option<Vec<u8>>) -> Option<Vec<u8>> {
    let parts = container_parts(blob)?;
    let (rts0_at, _) = rts0_part(blob)?;
    let mut change = Some(change);
    let mut out = blob[..32].to_vec();
    let mut offsets = Vec::with_capacity(parts.len());
    let mut body: Vec<u8> = Vec::new();
    let table_end = 32 + 4 * parts.len();
    for &(off, len) in &*parts {
        offsets.push((table_end + body.len()) as u32);
        if off + 8 == rts0_at {
            let new = change.take()?(&blob[off + 8..off + 8 + len])?;
            body.extend_from_slice(b"RTS0");
            body.extend_from_slice(&(new.len() as u32).to_le_bytes());
            body.extend_from_slice(&new);
        } else {
            body.extend_from_slice(&blob[off..off + 8 + len]);
        }
        body.resize(body.len().next_multiple_of(4), 0);
    }
    for off in offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    out.extend_from_slice(&body);
    let total = out.len();
    wr32(&mut out, 24, total as u32);
    let sum = dxbc_checksum(&out, total);
    for (i, word) in sum.iter().enumerate() {
        wr32(&mut out, 4 + 4 * i, *word);
    }
    Some(out)
}

/// One static sampler as its words: the 13 of root signature 1.0 and 1.1 (register, space and
/// visibility at 10, 11 and 12) and the flags of 1.2 last, 0 when the root signature had none.
pub type SamplerWords = [u32; 14];

/// The static samplers of the root signature in the container `blob`. Empty when it does not parse.
pub fn static_sampler_words(blob: &[u8]) -> Vec<SamplerWords> {
    let Some((start, len)) = rts0_part(blob) else { return Vec::new() };
    let data = &blob[start..start + len];
    let (Some(version), Some(count), Some(off)) = (get32(data, 0), get32(data, 12), get32(data, 16)) else { return Vec::new() };
    let stride = if version == 3 { 14 } else { 13 };
    let mut out = Vec::new();
    for i in 0..count as usize {
        let mut words = [0u32; 14];
        for (k, w) in words.iter_mut().enumerate().take(stride) {
            let Some(v) = (off as usize).checked_add(4 * (i * stride + k)).and_then(|at| get32(data, at)) else { return out };
            *w = v;
        }
        out.push(words);
    }
    out
}

/// Register space of the samplers a console shader holds as immediates: `decant_shader_dxil_cache` places
/// such a sampler at register `s<first descriptor word>` of this space.
pub const IMMEDIATE_SAMPLER_SPACE: u32 = 12;

/// A static sampler at `register` of [`IMMEDIATE_SAMPLER_SPACE`] rebuilt from the first word of a
/// console sampler descriptor (the public AMD sampler layout): the address modes of x, y and z in bits
/// 0 to 8, the anisotropy ratio in bits 9 to 11 and the comparison in bits 12 to 14. The filter is not
/// in that word: linear, anisotropic when the ratio is not 0, comparing when the comparison is not
/// never. A comparing sampler borders opaque white, the others transparent black.
pub fn immediate_sampler(word: u32) -> SamplerWords {
    // console clamp modes 0 to 7 to D3D12_TEXTURE_ADDRESS_MODE: wrap, mirror, clamp, mirror once,
    // half border, mirror once half border, border, mirror once border
    const ADDRESS: [u32; 8] = [1, 2, 3, 5, 4, 5, 4, 5];
    let address = |shift: u32| ADDRESS[((word >> shift) & 7) as usize];
    let ratio = (word >> 9) & 7;
    let compare = (word >> 12) & 7;
    let anisotropic = ratio != 0;
    let filter = match (compare != 0, anisotropic) {
        (false, false) => 0x15, // MIN_MAG_MIP_LINEAR
        (false, true) => 0x55,  // ANISOTROPIC
        (true, false) => 0x95,  // COMPARISON_MIN_MAG_MIP_LINEAR
        (true, true) => 0xD5,   // COMPARISON_ANISOTROPIC
    };
    let max_anisotropy = if anisotropic { 1u32 << ratio.min(4) } else { 1 };
    // D3D12_COMPARISON_FUNC is the console function plus one (never 1 to always 8).
    let border = if compare != 0 { 2 } else { 0 }; // OPAQUE_WHITE, TRANSPARENT_BLACK
    [filter, address(0), address(3), address(6), 0, max_anisotropy, compare + 1, border, 0, f32::MAX.to_bits(), word, IMMEDIATE_SAMPLER_SPACE, 0, 0]
}

/// A static sampler at `register` of [`IMMEDIATE_SAMPLER_SPACE`] rebuilt from the four words of a
/// console sampler descriptor (the public AMD RDNA2 sampler layout). Word 0 as in [`immediate_sampler`].
/// Word 1: minimum LOD in bits 0 to 11 and maximum LOD in bits 12 to 23 (unsigned 4.8). Word 2: LOD
/// bias in bits 0 to 13 (signed 5.8), then the filters, 0 point, 1 linear, 2 and 3 anisotropic
/// point and linear: xy magnification in bits 20 and 21, xy minification in 22 and 23, mip in 26 and
/// 27 (0 none, 1 point, 2 linear). Word 3: border color type in bits 30 and 31 (transparent black,
/// opaque black, opaque white, a table entry, the last taken as transparent black).
pub fn immediate_sampler_words(register: u32, words: [u32; 4]) -> SamplerWords {
    let mut s = immediate_sampler(words[0]);
    let (mag, min, mip) = ((words[2] >> 20) & 3, (words[2] >> 22) & 3, (words[2] >> 26) & 3);
    let compare = (words[0] >> 12) & 7 != 0;
    // D3D12_ENCODE_BASIC_FILTER(min, mag, mip, reduction): 0 point, 1 linear; the anisotropic bit 0x40.
    let linear = |f: u32| u32::from(f & 1 == 1);
    let anisotropic = mag >= 2 || min >= 2;
    let mut filter = (linear(min) << 4) | (linear(mag) << 2) | u32::from(mip == 2) | (u32::from(compare) << 7);
    if anisotropic {
        filter = 0x55 | (u32::from(compare) << 7);
    }
    let lod = |bits: u32| (bits & 0xFFF) as f32 / 256.0;
    let min_lod = lod(words[1]);
    // Without a mip filter only the first level is read.
    let max_lod = if mip == 0 { min_lod } else { lod(words[1] >> 12) };
    let bias = (((words[2] & 0x3FFF) << 18) as i32 >> 18) as f32 / 256.0;
    let ratio = (words[0] >> 9) & 7;
    s[0] = filter;
    s[4] = bias.to_bits();
    s[5] = if anisotropic { (1u32 << ratio.min(4)).max(2) } else { 1 };
    s[7] = match words[3] >> 30 {
        1 => 1,
        2 => 2,
        _ => 0,
    };
    s[8] = min_lod.to_bits();
    s[9] = max_lod.to_bits();
    s[10] = register;
    s
}

/// The static samplers of an `immediate_samplers.txt` written by `decant_shader_dxil_cache`: per line the
/// register alone (a first word, [`immediate_sampler`]) or the register and the four words
/// ([`immediate_sampler_words`]), in hex. Lines that do not parse are skipped.
pub fn immediate_samplers(text: &str) -> Vec<SamplerWords> {
    text.lines()
        .filter_map(|line| {
            let hex: Vec<u32> = line.split_whitespace().map(|w| u32::from_str_radix(w, 16).ok()).collect::<Option<_>>()?;
            match hex[..] {
                [word] => Some(immediate_sampler(word)),
                [register, a, b, c, d] => Some(immediate_sampler_words(register, [a, b, c, d])),
                _ => None,
            }
        })
        .collect()
}

/// Whether `layout` has sampler `register` of `space` already, as a static sampler or in a table.
fn has_sampler(layout: &RootLayout, register: u32, space: u32) -> bool {
    layout.static_samplers.iter().any(|s| s.register == register && s.space == space)
        || layout.params.iter().any(|p| match &p.kind {
            RootParamKind::Table(ranges) => ranges.iter().any(|r| {
                r.class == RegisterClass::Sampler && r.space == space && register >= r.base && (r.count == u32::MAX || u64::from(register) < u64::from(r.base) + u64::from(r.count))
            }),
            _ => false,
        })
}

/// A copy of the container `blob` whose root signature also has every sampler of `samplers` whose
/// register it does not have yet, with the number added. The static sampler array moves to the end
/// of the RTS0 data with the new samplers behind the title's. A sampler with 1.2 flags is not added
/// to an older root signature, which cannot carry them. `None` when nothing is added or the blob does
/// not parse.
pub fn append_static_samplers(blob: &[u8], samplers: &[SamplerWords]) -> Option<(Vec<u8>, usize)> {
    let layout = parse_root_layout(blob)?;
    let mut add: Vec<SamplerWords> = Vec::new();
    for s in samplers {
        let (register, space) = (s[10], s[11]);
        if has_sampler(&layout, register, space) || add.iter().any(|a| (a[10], a[11]) == (register, space)) || (layout.version != 3 && s[13] != 0) {
            continue;
        }
        add.push(*s);
    }
    if add.is_empty() {
        return None;
    }
    let old = static_sampler_words(blob);
    let stride = if layout.version == 3 { 14 } else { 13 };
    let out = with_new_rts0(blob, |data| {
        let mut new = data.to_vec();
        let at = new.len() as u32;
        for s in old.iter().chain(&add) {
            for w in &s[..stride] {
                new.extend_from_slice(&w.to_le_bytes());
            }
        }
        wr32(&mut new, 12, (old.len() + add.len()) as u32);
        wr32(&mut new, 16, at);
        Some(new)
    })?;
    Some((out, add.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A container with one RTS0 part (version 2, no parameters) carrying `flags`.
    fn container(flags: u32) -> Vec<u8> {
        let mut blob = vec![0u8; 32 + 4 + 8 + 24];
        blob[..4].copy_from_slice(b"DXBC");
        wr32(&mut blob, 20, 1);
        let total = blob.len() as u32;
        wr32(&mut blob, 24, total);
        wr32(&mut blob, 28, 1);
        wr32(&mut blob, 32, 36);
        blob[36..40].copy_from_slice(b"RTS0");
        wr32(&mut blob, 40, 24);
        wr32(&mut blob, 44, 2);
        wr32(&mut blob, 36 + 8 + 20, flags);
        let sum = dxbc_checksum(&blob, total as usize);
        for (i, word) in sum.iter().enumerate() {
            wr32(&mut blob, 4 + 4 * i, *word);
        }
        blob
    }

    #[test]
    fn the_flags_word_is_found_in_a_well_formed_container() {
        let blob = container(0x0300_0001);
        assert_eq!(flags_offset(&blob), Some(64));
        assert_eq!(rd32(&blob, 64), 0x0300_0001);
    }

    #[test]
    fn malformed_containers_are_refused_without_reading_outside() {
        assert_eq!(flags_offset(&[]), None);
        assert_eq!(flags_offset(&[0u8; 40]), None);
        let mut blob = container(1);
        blob.truncate(50);
        assert_eq!(flags_offset(&blob), None, "total size beyond the buffer");
        let mut blob = container(1);
        wr32(&mut blob, 28, 0x4000_0000);
        assert_eq!(flags_offset(&blob), None, "part count beyond the container");
        let mut blob = container(1);
        wr32(&mut blob, 32, 0xFFFF_FFF0);
        assert_eq!(flags_offset(&blob), None, "part offset beyond the container");
        let mut blob = container(1);
        wr32(&mut blob, 40, 0xFFFF_FFF0);
        assert_eq!(flags_offset(&blob), None, "part size beyond the container");
        let mut blob = container(1);
        wr32(&mut blob, 40, 20);
        assert_eq!(flags_offset(&blob), None, "RTS0 part too short for the flags word");
    }

    #[test]
    fn stripping_clears_the_console_bits_and_fixes_the_checksum() {
        let mut blob = container(0x0300_0001);
        assert_eq!(strip_unknown_flags(&mut blob), 0x0300_0000);
        assert_eq!(rd32(&blob, 64), 1);
        let sum = dxbc_checksum(&blob, blob.len());
        assert!((0..4).all(|i| rd32(&blob, 4 + 4 * i) == sum[i]));
        assert_eq!(blob, container(1), "same bytes as a container built with the public flags");
        assert_eq!(strip_unknown_flags(&mut blob), 0, "nothing left to clear");
    }

    #[test]
    fn adding_a_public_flag_sets_the_bit_and_fixes_the_checksum() {
        let mut blob = container(1);
        assert!(add_public_flag(&mut blob, 0x400), "the bit was not set yet");
        assert_eq!(rd32(&blob, 64), 0x401);
        let sum = dxbc_checksum(&blob, blob.len());
        assert!((0..4).all(|i| rd32(&blob, 4 + 4 * i) == sum[i]));
        assert_eq!(blob, container(0x401), "same bytes as a container built with the bit already set");
    }

    #[test]
    fn adding_an_already_set_public_flag_changes_nothing() {
        let mut blob = container(0x401);
        let before = blob.clone();
        assert!(!add_public_flag(&mut blob, 0x400), "the bit was already set");
        assert_eq!(blob, before);
    }

    #[test]
    fn adding_a_flag_to_a_malformed_blob_does_nothing() {
        let mut blob = vec![0u8; 10];
        assert!(!add_public_flag(&mut blob, 0x400));
    }

    /// A DXBC container holding an RTS0 part built the way the D3D12 serializer lays it out:
    /// header, parameter headers, then each parameter's data, then ranges, then static samplers.
    /// `params` is (type, visibility, data words) where a table's data words are its ranges, each
    /// a Vec of words, and `samplers` the (register, space, visibility) of each static sampler.
    fn root_blob(version: u32, flags: u32, params: &[(u32, u32, Vec<Vec<u32>>)], samplers: &[(u32, u32, u32)]) -> Vec<u8> {
        let sampler_stride = if version == 3 { 14 } else { 13 };
        let mut words: Vec<u32> = vec![version, params.len() as u32, 24, samplers.len() as u32, 0, flags];
        let mut body_at = 24 + 12 * params.len();
        let mut bodies: Vec<u32> = Vec::new();
        let mut headers: Vec<u32> = Vec::new();
        for (kind, vis, data) in params {
            headers.extend([*kind, *vis, (body_at + 4 * bodies.len()) as u32]);
            if *kind == 0 {
                // table: range count, range offset, then the ranges right after this body
                let here = body_at + 4 * bodies.len();
                bodies.extend([data.len() as u32, (here + 8) as u32]);
                for range in data {
                    bodies.extend(range);
                }
            } else {
                bodies.extend(&data[0]);
            }
        }
        body_at += 4 * bodies.len();
        words[4] = body_at as u32;
        words.extend(headers);
        words.extend(bodies);
        for (register, space, vis) in samplers {
            let mut s = vec![0u32; sampler_stride];
            (s[10], s[11], s[12]) = (*register, *space, *vis);
            words.extend(s);
        }
        let part: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut blob = vec![0u8; 36];
        blob[..4].copy_from_slice(b"DXBC");
        wr32(&mut blob, 20, 1);
        wr32(&mut blob, 28, 1);
        wr32(&mut blob, 32, 36);
        blob.extend(b"RTS0");
        blob.extend((part.len() as u32).to_le_bytes());
        blob.extend(part);
        let total = blob.len() as u32;
        wr32(&mut blob, 24, total);
        blob
    }

    /// The title's translated programs read root SRVs `t0` and `t4` of space 3 as raw buffers: every
    /// rewrite the layer makes of a root signature (console flags cleared, heap flags added, static
    /// samplers and the layer's parameters appended) keeps them as root SRVs at the same index,
    /// register, space and visibility.
    #[test]
    fn the_titles_root_srvs_in_space_3_survive_every_rewrite() {
        for version in [1u32, 2, 3] {
            let srv = |register: u32| if version == 1 { vec![vec![register, 3]] } else { vec![vec![register, 3, 0]] };
            let range = if version == 1 { vec![0, 8, 0, 0, 0] } else { vec![0, 8, 0, 0, 0, 0] };
            let blob = root_blob(version, 0x0300_0001, &[(0, 0, vec![range]), (3, 0, srv(0)), (1, 5, vec![vec![0, 0, 4]]), (3, 2, srv(4))], &[(0, 0, 0)]);
            let title = parse_root_layout(&blob).expect("the title's root signature parses").params;
            assert_eq!(title[1].kind, RootParamKind::Descriptor { class: RegisterClass::Srv, register: 0, space: 3 });
            assert_eq!(title[3].kind, RootParamKind::Descriptor { class: RegisterClass::Srv, register: 4, space: 3 });
            let mut c = blob.clone();
            strip_unknown_flags(&mut c);
            add_public_flag(&mut c, 0x400);
            add_public_flag(&mut c, 0x800);
            let mut sampler = [0u32; 14];
            (sampler[10], sampler[11]) = (7, 0);
            let (c, added) = append_static_samplers(&c, &[sampler]).expect("a sampler is added");
            assert_eq!(added, 1);
            let (c, _) = append_abi_params(&c, AbiWant { mirror: true, window: true, ..AbiWant::default() }).expect("the layer's parameters fit");
            let layout = parse_root_layout(&c).expect("the rewritten root signature parses");
            assert_eq!(layout.params[..4], title[..], "version {version}: the title's parameters keep index, kind, register, space and visibility");
            assert_eq!(layout.flags & 0x0300_0000, 0);
            assert_eq!(layout.without_layer_params().params, title, "version {version}");
        }
    }

    #[test]
    fn a_version_1_1_root_signature_is_parsed() {
        // table (SRV t0 x4 space 0, UAV u4 x unbounded space 1), root CBV b2, constants b5 x4 space 3
        let blob = root_blob(
            2,
            0x401,
            &[
                (0, 0, vec![vec![0, 4, 0, 0, 0, 0], vec![1, u32::MAX, 4, 1, 0x8, 4]]),
                (2, 1, vec![vec![2, 0, 0]]),
                (1, 5, vec![vec![5, 3, 4]]),
                (3, 0, vec![vec![9, 0, 0]]),
            ],
            &[(1, 0, 5)],
        );
        let layout = parse_root_layout(&blob).expect("a well formed blob");
        assert_eq!((layout.version, layout.flags), (2, 0x401));
        assert_eq!(layout.params.len(), 4);
        let RootParamKind::Table(ranges) = &layout.params[0].kind else { panic!("a table") };
        assert_eq!(ranges[0], RootRange { class: RegisterClass::Srv, count: 4, base: 0, space: 0, flags: 0, offset: 0 });
        assert_eq!(ranges[1], RootRange { class: RegisterClass::Uav, count: u32::MAX, base: 4, space: 1, flags: 8, offset: 4 });
        assert_eq!(layout.params[1].kind, RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 2, space: 0 });
        assert_eq!(layout.params[1].visibility, 1);
        assert_eq!(layout.params[2].kind, RootParamKind::Constants { register: 5, space: 3, count: 4 });
        assert_eq!(layout.params[3].kind, RootParamKind::Descriptor { class: RegisterClass::Srv, register: 9, space: 0 });
        assert_eq!(layout.static_samplers, vec![StaticSampler { register: 1, space: 0, visibility: 5 }]);
    }

    #[test]
    fn the_title_view_drops_only_the_trailing_layer_parameters() {
        let d = |class, register, space| RootParam { kind: RootParamKind::Descriptor { class, register, space }, visibility: 0 };
        let title = vec![d(RegisterClass::Cbv, 0, 0), d(RegisterClass::Srv, 0, MIRROR_SPACE)];
        let mut layout = RootLayout { version: 2, flags: 0, params: title.clone(), static_samplers: Vec::new() };
        layout.params.extend([d(RegisterClass::Cbv, MIRROR_REGISTER, MIRROR_SPACE), d(RegisterClass::Uav, 0, WINDOW_SPACE), d(RegisterClass::Uav, 2, SERVICE_SPACE)]);
        let view = layout.without_layer_params();
        // The title's own parameter in space 15 stays: it is not after the layer's.
        assert_eq!(view.params, title[..1].to_vec(), "trailing parameters of the layer spaces go");
        assert_eq!(RootLayout { params: title.clone(), ..layout.clone() }.without_layer_params().params, title[..1]);
    }

    #[test]
    fn a_version_1_0_root_signature_has_five_word_ranges() {
        let blob = root_blob(1, 0, &[(0, 0, vec![vec![3, 2, 7, 1, 0], vec![2, 1, 0, 0, u32::MAX]])], &[(3, 2, 0)]);
        let layout = parse_root_layout(&blob).expect("a well formed blob");
        let RootParamKind::Table(ranges) = &layout.params[0].kind else { panic!("a table") };
        assert_eq!(ranges[0], RootRange { class: RegisterClass::Sampler, count: 2, base: 7, space: 1, flags: 0, offset: 0 });
        assert_eq!((ranges[1].class, ranges[1].offset), (RegisterClass::Cbv, 2), "an appended range follows the one before");
        assert_eq!(layout.static_samplers, vec![StaticSampler { register: 3, space: 2, visibility: 0 }]);
    }

    #[test]
    fn a_version_1_2_root_signature_has_longer_static_samplers() {
        let blob = root_blob(3, 0, &[], &[(1, 0, 0), (2, 4, 1)]);
        let layout = parse_root_layout(&blob).expect("a well formed blob");
        assert_eq!(layout.static_samplers.iter().map(|s| (s.register, s.space, s.visibility)).collect::<Vec<_>>(), [(1, 0, 0), (2, 4, 1)]);
    }

    #[test]
    fn a_malformed_root_signature_is_refused_without_reading_outside() {
        assert_eq!(parse_root_layout(&[]), None);
        assert_eq!(parse_root_layout(&container(0)).map(|l| (l.params.len(), l.static_samplers.len())), Some((0, 0)), "an empty root signature");
        let good = root_blob(2, 0, &[(0, 0, vec![vec![0, 1, 0, 0, 0, 0]])], &[(0, 0, 0)]);
        assert!(parse_root_layout(&good).is_some());
        let part = 44;
        for (at, value, what) in [
            (part, 9, "unknown version"),
            (part + 4, 0x4000_0000, "parameter count beyond the part"),
            (part + 8, 0xFFFF_FFF0, "parameter offset beyond the part"),
            (part + 12, 0x4000_0000, "sampler count beyond the part"),
            (part + 16, 0xFFFF_FFF0, "sampler offset beyond the part"),
            (part + 24, 9, "unknown parameter type"),
            (part + 32, 0xFFFF_FFF0, "parameter body beyond the part"),
        ] {
            let mut blob = good.clone();
            wr32(&mut blob, at, value);
            assert_eq!(parse_root_layout(&blob), None, "{what}");
        }
        let mut blob = good.clone();
        let table = rd32(&blob, part + 32) as usize + part;
        wr32(&mut blob, table, 0x4000_0000);
        assert_eq!(parse_root_layout(&blob), None, "range count beyond the part");
        let mut blob = good;
        let table = rd32(&blob, part + 32) as usize + part;
        let ranges = rd32(&blob, table + 4) as usize + part;
        wr32(&mut blob, ranges, 7);
        assert_eq!(parse_root_layout(&blob), None, "unknown range type");
    }

    #[test]
    fn every_truncation_of_a_root_signature_is_handled() {
        let good = root_blob(2, 0, &[(0, 0, vec![vec![0, 1, 0, 0, 0, 0]]), (2, 0, vec![vec![0, 0, 0]])], &[(0, 0, 0)]);
        for len in 0..good.len() {
            let _ = parse_root_layout(&good[..len]);
        }
    }

    /// The layout the real serializer writes, parsed back (Windows only: it loads d3d12.dll).
    #[cfg(windows)]
    #[test]
    #[allow(unsafe_code)]
    fn the_d3d12_serializer_output_is_parsed() {
        use windows::Win32::Graphics::Direct3D12::*;
        let ranges = [
            D3D12_DESCRIPTOR_RANGE1 { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 4, BaseShaderRegister: 188, RegisterSpace: 0, Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE, OffsetInDescriptorsFromTableStart: 0 },
            D3D12_DESCRIPTOR_RANGE1 { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV, NumDescriptors: u32::MAX, BaseShaderRegister: 9, RegisterSpace: 2, Flags: D3D12_DESCRIPTOR_RANGE_FLAG_NONE, OffsetInDescriptorsFromTableStart: 4 },
        ];
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER1_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 2, pDescriptorRanges: ranges.as_ptr() } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 { Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: 27, RegisterSpace: 1, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 5, RegisterSpace: 0, Num32BitValues: 3 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let samplers = [D3D12_STATIC_SAMPLER_DESC { ShaderRegister: 6, RegisterSpace: 0, ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            MaxLOD: 1000.0,
            ..Default::default()
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 3,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 1,
                    pStaticSamplers: samplers.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // The crate exports a function of the same name, so the system one is looked up by name.
        type Serialize = unsafe extern "system" fn(*const D3D12_VERSIONED_ROOT_SIGNATURE_DESC, *mut *mut core::ffi::c_void, *mut *mut core::ffi::c_void) -> i32;
        let dll: Vec<u16> = "d3d12.dll\0".encode_utf16().collect();
        // SAFETY: a NUL terminated name, and the exported function has the signature above.
        let serialize: Serialize = unsafe {
            let module = windows_sys::Win32::System::LibraryLoader::LoadLibraryW(dll.as_ptr());
            assert!(!module.is_null(), "d3d12.dll");
            std::mem::transmute(windows_sys::Win32::System::LibraryLoader::GetProcAddress(module, c"D3D12SerializeVersionedRootSignature".as_ptr().cast()).expect("export"))
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the description and everything it points to outlive the call.
        let hr = unsafe { serialize(&desc, &mut raw, std::ptr::null_mut()) };
        assert!(hr >= 0 && !raw.is_null(), "serialization failed, hr={hr:#x}");
        // SAFETY: `raw` is the new ID3DBlob the call returned, owned by us now.
        let blob = unsafe { <windows::Win32::Graphics::Direct3D::ID3DBlob as windows::core::Interface>::from_raw(raw) };
        // SAFETY: the blob owns that buffer for as long as `blob` lives.
        let bytes = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
        let layout = parse_root_layout(bytes).expect("the serializer's own output parses");
        assert_eq!(layout.flags, 0x400);
        assert_eq!(layout.params.len(), 3);
        let RootParamKind::Table(r) = &layout.params[0].kind else { panic!("a table") };
        assert_eq!((r[0].class, r[0].base, r[0].count, r[0].space), (RegisterClass::Srv, 188, 4, 0));
        assert_eq!((r[1].class, r[1].base, r[1].count, r[1].space), (RegisterClass::Uav, 9, u32::MAX, 2));
        assert_eq!(layout.params[1].kind, RootParamKind::Descriptor { class: RegisterClass::Cbv, register: 27, space: 1 });
        assert_eq!(layout.params[1].visibility, 5);
        assert_eq!(layout.params[2].kind, RootParamKind::Constants { register: 5, space: 0, count: 3 });
        assert_eq!(layout.static_samplers, vec![StaticSampler { register: 6, space: 0, visibility: 5 }]);
        // the serializer's own layout takes the mirror too
        let patched = parse_root_layout(&mirror_only(bytes).expect("patched")).expect("parses");
        assert_eq!(patched.params[..3], layout.params[..]);
        assert_eq!(patched.mirror_index(), Some(3));
        assert_eq!(patched.static_samplers, layout.static_samplers);
    }

    fn mirror_only(blob: &[u8]) -> Result<Vec<u8>, AppendRefusal> {
        append_abi_params(blob, AbiWant { mirror: true, ..AbiWant::default() }).map(|(b, _)| b)
    }

    fn all_groups() -> AbiWant {
        AbiWant { mirror: true, window: true, ..AbiWant::default() }
    }

    fn desc(class: RegisterClass, register: u32, space: u32) -> RootParam {
        RootParam { kind: RootParamKind::Descriptor { class, register, space }, visibility: 0 }
    }

    #[test]
    fn window_and_service_parameters_are_appended_in_order_to_every_version() {
        for version in [1u32, 2, 3] {
            let range = |class: u32, count: u32, base: u32| if version == 1 { vec![class, count, base, 0, 0] } else { vec![class, count, base, 0, 0, 0] };
            let d = if version == 1 { vec![vec![2, 0]] } else { vec![vec![2, 0, 0]] };
            let blob = root_blob(version, 0xC01, &[(0, 0, vec![range(0, 4, 0), range(1, 8, 2)]), (2, 1, d), (1, 5, vec![vec![5, 3, 4]])], &[(1, 0, 5)]);
            let before = parse_root_layout(&blob).unwrap();
            let (patched, added) = append_abi_params(&blob, all_groups()).unwrap_or_else(|e| panic!("version {version}: {e:?}"));
            assert_eq!(added, AbiAdded { mirror: true, window: true, service: true, heap_map: false, view_shadow: false });
            let after = parse_root_layout(&patched).unwrap_or_else(|| panic!("version {version}: must parse"));
            let n = before.params.len();
            assert_eq!(&after.params[..n], before.params.as_slice());
            assert_eq!(after.params[n..].to_vec(), vec![
                mirror_param(),
                desc(RegisterClass::Uav, 0, WINDOW_SPACE),
                desc(RegisterClass::Cbv, 0, WINDOW_SPACE),
                desc(RegisterClass::Srv, 0, SERVICE_SPACE),
                desc(RegisterClass::Uav, 0, SERVICE_SPACE),
                desc(RegisterClass::Uav, 1, SERVICE_SPACE),
                desc(RegisterClass::Uav, 2, SERVICE_SPACE),
            ]);
            assert_eq!((after.mirror_index(), after.window_uav_index(), after.window_base_index()), (Some(n), Some(n + 1), Some(n + 2)));
            assert_eq!(after.service_indices(), [Some(n + 3), Some(n + 4), Some(n + 5), Some(n + 6)]);
            assert_eq!(after.dwords(), before.dwords() + 14);
            assert_eq!((after.version, after.flags, &after.static_samplers), (before.version, before.flags, &before.static_samplers));
            assert_eq!(rd32(&patched, 24) as usize, patched.len());
            let sum = dxbc_checksum(&patched, patched.len());
            assert!((0..4).all(|i| rd32(&patched, 4 + 4 * i) == sum[i]));
            // a second pass finds everything present
            assert_eq!(append_abi_params(&patched, all_groups()), Err(AppendRefusal::Present));
        }
    }

    #[test]
    fn groups_that_do_not_fit_the_64_dwords_are_left_out() {
        // 50 dwords: all three groups (2 + 4 + 8) fit exactly
        let fifty = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 50]])], &[]);
        let (_, added) = append_abi_params(&fifty, all_groups()).unwrap();
        assert_eq!(added, AbiAdded { mirror: true, window: true, service: true, heap_map: false, view_shadow: false });
        // 51 dwords: the service group does not fit, the others do
        let more = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 51]])], &[]);
        let (blob, added) = append_abi_params(&more, all_groups()).unwrap();
        assert_eq!(added, AbiAdded { mirror: true, window: true, service: false, heap_map: false, view_shadow: false });
        assert!(parse_root_layout(&blob).unwrap().dwords() <= MAX_ROOT_DWORDS);
        // 60 dwords: only the mirror and not the window (4 more would be 66)
        let near = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 60]])], &[]);
        let (_, added) = append_abi_params(&near, all_groups()).unwrap();
        assert_eq!(added, AbiAdded { mirror: true, window: false, service: false, heap_map: false, view_shadow: false });
        // the window alone is wanted and does not fit
        let full = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 62]])], &[]);
        assert_eq!(append_abi_params(&full, AbiWant { window: true, ..AbiWant::default() }), Err(AppendRefusal::TooLarge { dwords: 66 }));
        // a title root signature with the mirror only gets the others
        let with_mirror = mirror_only(&root_blob(2, 0, &[], &[])).unwrap();
        let (_, added) = append_abi_params(&with_mirror, all_groups()).unwrap();
        assert_eq!(added, AbiAdded { mirror: false, window: true, service: true, heap_map: false, view_shadow: false });
    }

    #[test]
    fn the_heap_map_and_the_view_shadow_follow_the_mirror_and_only_come_with_it() {
        let blob = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 4]])], &[]);
        let want = AbiWant { mirror: true, window: true, heap_map: true, view_shadow: true };
        let (patched, added) = append_abi_params(&blob, want).unwrap();
        assert_eq!(added, AbiAdded { mirror: true, window: true, service: true, heap_map: true, view_shadow: true });
        let after = parse_root_layout(&patched).unwrap();
        let indices = (after.mirror_index(), after.heap_map_index(), after.view_shadow_index(), after.window_uav_index());
        assert_eq!(indices, (Some(1), Some(2), Some(3), Some(4)));
        assert_eq!(after.params[2], desc(RegisterClass::Srv, HEAP_MAP_REGISTER, MIRROR_SPACE));
        assert_eq!(after.params[3], desc(RegisterClass::Srv, VIEW_SHADOW_REGISTER, MIRROR_SPACE));
        assert_eq!(after.without_layer_params().params.len(), 1, "both are parameters of the layer");
        assert_eq!(append_abi_params(&patched, want), Err(AppendRefusal::Present));
        // without the mirror neither comes, with a mirror already there they come alone
        let alone = AbiWant { heap_map: true, view_shadow: true, ..AbiWant::default() };
        assert_eq!(append_abi_params(&blob, alone), Err(AppendRefusal::Present));
        let mirrored = mirror_only(&blob).unwrap();
        let (out, added) = append_abi_params(&mirrored, alone).unwrap();
        assert_eq!(added, AbiAdded { heap_map: true, view_shadow: true, ..AbiAdded::default() });
        assert_eq!(parse_root_layout(&out).unwrap().view_shadow_index(), Some(3));
        // 60 dwords: the mirror and the heap map fit, the view shadow does not
        let near = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 60]])], &[]);
        let (_, added) = append_abi_params(&near, want).unwrap();
        assert_eq!(added, AbiAdded { mirror: true, heap_map: true, ..AbiAdded::default() });
    }

    fn mirror_param() -> RootParam {
        RootParam { kind: RootParamKind::Descriptor { class: RegisterClass::Cbv, register: MIRROR_REGISTER, space: MIRROR_SPACE }, visibility: 0 }
    }

    #[test]
    fn the_mirror_parameter_is_appended_to_every_root_signature_version() {
        for version in [1u32, 2, 3] {
            let range = |class: u32, count: u32, base: u32| if version == 1 { vec![class, count, base, 0, 0] } else { vec![class, count, base, 0, 0, 0] };
            let desc = if version == 1 { vec![vec![2, 0]] } else { vec![vec![2, 0, 0]] };
            let blob = root_blob(
                version,
                0xC01,
                &[(0, 0, vec![range(0, 4, 0), range(1, 8, 2)]), (2, 1, desc.clone()), (1, 5, vec![vec![5, 3, 4]]), (0, 0, vec![range(3, 2, 0)])],
                &[(1, 0, 5), (2, 0, 0)],
            );
            let before = parse_root_layout(&blob).unwrap();
            let patched = mirror_only(&blob).unwrap_or_else(|e| panic!("version {version}: {e:?}"));
            let after = parse_root_layout(&patched).unwrap_or_else(|| panic!("version {version}: the patched blob must parse"));
            assert_eq!(after.params.len(), before.params.len() + 1);
            assert_eq!(&after.params[..before.params.len()], before.params.as_slice(), "title parameters keep their index and content");
            assert_eq!(after.params.last(), Some(&mirror_param()));
            assert_eq!(after.mirror_index(), Some(before.params.len()));
            assert_eq!((after.version, after.flags, &after.static_samplers), (before.version, before.flags, &before.static_samplers));
            assert_eq!(rd32(&patched, 24) as usize, patched.len());
            let sum = dxbc_checksum(&patched, patched.len());
            assert!((0..4).all(|i| rd32(&patched, 4 + 4 * i) == sum[i]), "checksum of version {version}");
            assert_eq!(before.mirror_index(), None);
        }
    }

    #[test]
    fn a_root_signature_with_no_parameters_and_other_container_parts_is_patched() {
        // an RTS0 part followed by another part: the second part moves and keeps its bytes
        let single = root_blob(2, 0, &[], &[]);
        let rts0 = &single[36..];
        let mut table = single[..32].to_vec();
        table.extend(40u32.to_le_bytes());
        table.extend((40 + rts0.len() as u32).to_le_bytes());
        table.extend_from_slice(rts0);
        table.extend(b"PSV0");
        table.extend(8u32.to_le_bytes());
        table.extend([1, 2, 3, 4, 5, 6, 7, 8]);
        wr32(&mut table, 28, 2);
        let total = table.len() as u32;
        wr32(&mut table, 24, total);
        let patched = mirror_only(&table).expect("patched");
        assert_eq!(parse_root_layout(&patched).unwrap().params, vec![mirror_param()]);
        assert_eq!(rd32(&patched, 28), 2);
        let second = rd32(&patched, 36) as usize;
        assert_eq!(&patched[second..second + 4], b"PSV0");
        assert_eq!(&patched[second + 8..second + 16], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn the_patch_is_refused_when_it_cannot_be_made() {
        assert_eq!(mirror_only(&[1, 2, 3]), Err(AppendRefusal::NotParsed));
        let once = mirror_only(&root_blob(2, 0, &[(2, 0, vec![vec![0, 0, 0]])], &[])).unwrap();
        assert_eq!(mirror_only(&once), Err(AppendRefusal::Present));
        // 63 dwords of constants and a root descriptor: 65 with the mirror
        let big = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 62]]), (2, 0, vec![vec![0, 0, 0]])], &[]);
        assert_eq!(mirror_only(&big), Err(AppendRefusal::TooLarge { dwords: 66 }));
        let fits = root_blob(2, 0, &[(1, 0, vec![vec![0, 0, 62]])], &[]);
        assert!(mirror_only(&fits).is_ok(), "64 dwords exactly");
        for len in 0..fits.len() {
            let _ = mirror_only(&fits[..len]);
        }
    }

    #[test]
    fn missing_static_samplers_are_appended_behind_the_titles() {
        let full = root_blob(2, 0, &[], &[(0, 1, 0), (3, 1, 0), (5, 1, 0)]);
        let learned = static_sampler_words(&full);
        assert_eq!(learned.len(), 3);
        // has s3 space 1 already, and a sampler table over s5 space 1
        let partial = root_blob(2, 0, &[(0, 0, vec![vec![3, 2, 4, 1, 0, 0]])], &[(3, 1, 5)]);
        let (out, added) = append_static_samplers(&partial, &learned).unwrap();
        assert_eq!(added, 1, "only s0 is missing");
        let layout = parse_root_layout(&out).unwrap();
        let regs: Vec<(u32, u32, u32)> = layout.static_samplers.iter().map(|s| (s.register, s.space, s.visibility)).collect();
        assert_eq!(regs, vec![(3, 1, 5), (0, 1, 0)], "the title's sampler first, its visibility kept");
        assert_eq!(layout.params, parse_root_layout(&partial).unwrap().params);
        let sum = dxbc_checksum(&out, out.len());
        assert!((0..4).all(|i| rd32(&out, 4 + 4 * i) == sum[i]));
        assert_eq!(append_static_samplers(&out, &learned), None, "nothing left to add");
        assert_eq!(append_static_samplers(&full, &learned), None);
    }

    #[test]
    fn an_immediate_sampler_word_gives_address_modes_anisotropy_and_comparison() {
        let clamp = immediate_sampler(0x92);
        assert_eq!(&clamp[..4], &[0x15, 3, 3, 3], "linear, clamp in x, y and z");
        assert_eq!((clamp[5], clamp[7], clamp[10], clamp[11]), (1, 0, 0x92, 12));
        let shadow = immediate_sampler(0x30b6);
        assert_eq!(&shadow[..4], &[0x95, 4, 4, 3], "comparing, border, border, clamp");
        assert_eq!((shadow[6], shadow[7]), (4, 2), "less or equal, opaque white");
        let aniso = immediate_sampler(0x400);
        assert_eq!((aniso[0], aniso[1], aniso[5]), (0x55, 1, 4), "wrap, four times anisotropic");
        assert_eq!(f32::from_bits(aniso[9]), f32::MAX);
    }

    #[test]
    fn four_sampler_words_give_the_filter_lod_range_bias_and_border() {
        // Words seen in the shader pack of the title.
        let linear = immediate_sampler_words(0x0100_0001, [0x92, 0xc0_0000, 0x1850_0000, 0]);
        assert_eq!(&linear[..4], &[0x15, 3, 3, 3], "linear with linear mips, clamp");
        assert_eq!((f32::from_bits(linear[8]), f32::from_bits(linear[9]), f32::from_bits(linear[4])), (0.0, 12.0, 0.0));
        assert_eq!((linear[10], linear[11]), (0x0100_0001, 12));
        let point = immediate_sampler_words(2, [0x92, 0xc0_0000, 0x1400_0000, 0]);
        assert_eq!(point[0], 0, "MIN_MAG_MIP_POINT");
        let bilinear = immediate_sampler_words(3, [0x0, 0xc0_0000, 0x1450_0000, 0]);
        assert_eq!((bilinear[0], bilinear[1]), (0x14, 1), "MIN_MAG_LINEAR_MIP_POINT, wrap");
        let shadow = immediate_sampler_words(4, [0x30b6, 0xc0_0000, 0x1850_0000, 0x8000_0000]);
        assert_eq!((shadow[0], shadow[6], shadow[7]), (0x95, 4, 2), "comparing linear, less or equal, opaque white");
        let aniso = immediate_sampler_words(5, [0x400, 0xc0_0000, 0x18f0_0000, 0]);
        assert_eq!((aniso[0], aniso[5]), (0x55, 4));
        let biased = immediate_sampler_words(6, [0x200, 0xc0_0000, 0x18f0_3e80, 0]);
        assert_eq!(f32::from_bits(biased[4]), -1.5);
        let first_level = immediate_sampler_words(7, [0, 0, 0, 0]);
        assert_eq!((first_level[0], f32::from_bits(first_level[9])), (0, 0.0), "point, no mips: LOD 0 only");
    }

    #[test]
    fn the_immediate_samplers_file_has_both_line_forms() {
        let s = immediate_samplers("00000092\n0100abcd 00000092 00c00000 14000000 00000000\nzz\n1 2\n");
        assert_eq!(s.len(), 2);
        assert_eq!((s[0][0], s[0][10]), (0x15, 0x92), "a first word alone stays linear");
        assert_eq!((s[1][0], s[1][10], s[1][11]), (0, 0x0100_abcd, 12));
    }

    #[test]
    fn a_sampler_with_flags_is_not_added_to_an_older_root_signature() {
        let mut s = [0u32; 14];
        (s[10], s[11], s[13]) = (2, 1, 1);
        assert_eq!(append_static_samplers(&root_blob(2, 0, &[], &[]), &[s]), None);
        let (out, added) = append_static_samplers(&root_blob(3, 0, &[], &[]), &[s]).unwrap();
        assert_eq!(added, 1);
        assert_eq!(static_sampler_words(&out), vec![s]);
    }

    #[test]
    fn the_layout_counts_root_dwords() {
        let blob = root_blob(2, 0, &[(0, 0, vec![vec![0, 1, 0, 0, 0, 0]]), (1, 0, vec![vec![0, 0, 5]]), (3, 0, vec![vec![0, 0, 0]])], &[]);
        assert_eq!(parse_root_layout(&blob).unwrap().dwords(), 1 + 5 + 2);
    }

    #[test]
    fn checksum_covers_every_padding_branch() {
        // Different lengths take the two final block layouts (fewer than 56 bytes left, or more).
        let mut seen = std::collections::HashSet::new();
        for len in [20usize, 21, 75, 76, 77, 83, 84, 100, 147, 148, 149, 200] {
            let mut blob = vec![0x5Au8; len];
            blob[..4].copy_from_slice(b"DXBC");
            assert!(seen.insert(dxbc_checksum(&blob, len)), "length {len} repeats another checksum");
        }
    }

    #[test]
    fn container_parts_handles_stack_and_heap_part_counts() {
        let base = container(0);
        assert_eq!(container_parts(&base).unwrap().len(), 1);
    }
}
