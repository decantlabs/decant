//! Turns the pipeline state stream of console device slot 86 (`CreatePipelineState`) into the plain
//! PC descriptions: a `D3D12_COMPUTE_PIPELINE_STATE_DESC`, a `D3D12_GRAPHICS_PIPELINE_STATE_DESC`
//! or our own description of a mesh pipeline. On the way root signatures can be unwrapped, compute
//! shaders go through `dxil_override`, console DXGI formats (`117..=120`) become their PC
//! equivalents (`44..=47`) and enum values only the console knows are replaced.
//!
//! A stream is a run of subobjects. Each starts with a `u32` type, then the payload at the
//! alignment its type needs, and the next subobject starts at the next multiple of 8.

use crate::abi::pc_dxgi_format;
use crate::mesh_pso::MESH_DESC_SIZE;

/// Payload alignment and size of every subobject type of a console stream, on x86_64 Windows.
/// Types up to 21 are the PC ones. The console numbers its amplification and mesh shaders 23 and
/// 24, one below the 24 and 25 of the PC. A stream builder of a retail title was measured to give
/// a mesh program type 24 where a vertex shader goes and type 23 for an amplification program.
/// Read with the PC numbers, such streams look like streams without any shader.
const SUBOBJECTS: [(u32, usize, usize); 25] = [
    (0, 8, 8),    // root signature: a pointer
    (1, 8, 16),   // VS, a D3D12_SHADER_BYTECODE { pointer, length }
    (2, 8, 16),   // PS
    (3, 8, 16),   // DS
    (4, 8, 16),   // HS
    (5, 8, 16),   // GS
    (6, 8, 16),   // CS
    (7, 8, 32),   // stream output: two pointers and three u32 with padding
    (8, 4, 328),  // blend: two u32 and eight render target blocks of 40 bytes
    (9, 4, 4),    // sample mask
    (10, 4, 44),  // rasterizer: eleven u32
    (11, 4, 52),  // depth stencil
    (12, 8, 16),  // input layout: { pointer, count }
    (13, 4, 4),   // index buffer strip cut value
    (14, 4, 4),   // primitive topology type
    (15, 4, 36),  // render target formats: eight formats, then the count
    (16, 4, 4),   // depth stencil format
    (17, 4, 8),   // sample desc: { count, quality }
    (18, 4, 4),   // node mask
    (19, 8, 16),  // cached pipeline: { pointer, length }
    (20, 4, 4),   // flags
    (21, 4, 56),  // depth stencil with the depth bounds test flag appended
    (22, 8, 24),  // view instancing
    (23, 8, 16),  // amplification shader
    (24, 8, 16),  // mesh shader
];

/// Alignment and payload size of the subobject type `ty`, `None` when the type is not known.
pub fn subobject_layout(ty: u32) -> Option<(usize, usize)> {
    SUBOBJECTS.iter().find(|s| s.0 == ty).map(|&(_, align, size)| (align, size))
}

/// The bytes of a x86_64 `D3D12_COMPUTE_PIPELINE_STATE_DESC` (56 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlainComputeDesc {
    pub root_sig: usize,
    pub cs_ptr: usize,
    pub cs_len: usize,
    pub node_mask: u32,
    pub _pad0: u32,
    pub cached_ptr: usize,
    pub cached_len: usize,
    pub flags: u32,
    pub _pad1: u32,
}

/// Size of a x86_64 `D3D12_GRAPHICS_PIPELINE_STATE_DESC` in the Windows SDK layout.
pub const GRAPHICS_DESC_SIZE: usize = 656;

/// What a stream unpacks to.
#[derive(Clone, Debug)]
pub enum UnpackedStream {
    Compute(PlainComputeDesc),
    Graphics(Box<[u8; GRAPHICS_DESC_SIZE]>),
    /// A mesh pipeline in our description of one (`mesh_pso`).
    Mesh(Box<[u8; MESH_DESC_SIZE]>),
}

/// Byte offsets of the fields of the plain graphics description.
mod at {
    pub const ROOT_SIGNATURE: usize = 0x000;
    pub const VS: usize = 0x008;
    pub const PS: usize = 0x018;
    pub const DS: usize = 0x028;
    pub const HS: usize = 0x038;
    pub const GS: usize = 0x048;
    pub const STREAM_OUTPUT: usize = 0x058;
    pub const BLEND: usize = 0x078;
    pub const INDEPENDENT_BLEND: usize = BLEND + 4;
    pub const RENDER_TARGET0: usize = BLEND + 8;
    pub const RENDER_TARGET_STRIDE: usize = 40;
    pub const SAMPLE_MASK: usize = 0x1C0;
    pub const RASTERIZER: usize = 0x1C4;
    pub const DEPTH_STENCIL: usize = 0x1F0;
    pub const INPUT_LAYOUT: usize = 0x228;
    pub const STRIP_CUT: usize = 0x238;
    pub const TOPOLOGY: usize = 0x23C;
    pub const RT_COUNT: usize = 0x240;
    pub const RT_FORMATS: usize = 0x244;
    pub const DEPTH_FORMAT: usize = 0x264;
    pub const SAMPLE_DESC: usize = 0x268;
    pub const NODE_MASK: usize = 0x270;
    pub const CACHED: usize = 0x278;
    pub const FLAGS: usize = 0x288;

    /// Offset of the block of render target `i` inside the blend state.
    pub const fn render_target(i: usize) -> usize {
        RENDER_TARGET0 + i * RENDER_TARGET_STRIDE
    }
}

/// Where a subobject payload is copied to in the graphics description: type, offset, byte count.
/// The payload of a subobject can be longer than the field (depth stencil with the bounds flag),
/// then the first bytes are taken. Types 15, 16, 23 and 24 are not plain copies and are handled
/// in `unpack_graphics`.
const COPIES: [(u32, usize, usize); 19] = [
    (0, at::ROOT_SIGNATURE, 8),
    (1, at::VS, 16),
    (2, at::PS, 16),
    (3, at::DS, 16),
    (4, at::HS, 16),
    (5, at::GS, 16),
    (7, at::STREAM_OUTPUT, 32),
    (8, at::BLEND, 328),
    (9, at::SAMPLE_MASK, 4),
    (10, at::RASTERIZER, 44),
    (11, at::DEPTH_STENCIL, 52),
    (21, at::DEPTH_STENCIL, 52),
    (12, at::INPUT_LAYOUT, 16),
    (13, at::STRIP_CUT, 4),
    (14, at::TOPOLOGY, 4),
    (17, at::SAMPLE_DESC, 8),
    (18, at::NODE_MASK, 4),
    (19, at::CACHED, 16),
    (20, at::FLAGS, 4),
];

fn rd_u32(b: &[u8], off: usize) -> u32 {
    b.get(off..off + 4).and_then(|s| s.try_into().ok()).map_or(0, u32::from_le_bytes)
}

fn rd_usize(b: &[u8], off: usize) -> usize {
    b.get(off..off + 8).and_then(|s| s.try_into().ok()).map_or(0, |a| u64::from_le_bytes(a) as usize)
}

fn wr_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// The description a graphics stream starts from: the D3D12 defaults of every state a stream may
/// leave out (solid fill, back face culling, depth test less, one sample, all colour channels).
fn default_graphics_desc() -> [u8; GRAPHICS_DESC_SIZE] {
    // Enum values used below (D3D12): BLEND_ZERO 1, BLEND_ONE 2, BLEND_OP_ADD 1, LOGIC_OP_NOOP 4,
    // FILL_MODE_SOLID 3, CULL_MODE_BACK 3, COMPARISON_FUNC_LESS 2 and _ALWAYS 8, STENCIL_OP_KEEP 1.
    let mut g = [0u8; GRAPHICS_DESC_SIZE];
    for i in 0..8 {
        let rt = at::render_target(i);
        for (field, v) in [(8, 2), (12, 1), (16, 1), (20, 2), (24, 1), (28, 1), (32, 4)] {
            wr_u32(&mut g, rt + field, v);
        }
        g[rt + 36] = 0x0F;
    }
    wr_u32(&mut g, at::SAMPLE_MASK, u32::MAX);
    wr_u32(&mut g, at::RASTERIZER, 3);
    wr_u32(&mut g, at::RASTERIZER + 4, 3);
    wr_u32(&mut g, at::RASTERIZER + 24, 1);
    let ds = at::DEPTH_STENCIL;
    wr_u32(&mut g, ds, 1);
    wr_u32(&mut g, ds + 4, 1);
    wr_u32(&mut g, ds + 8, 2);
    g[ds + 16] = 0xFF;
    g[ds + 17] = 0xFF;
    for face in [ds + 20, ds + 36] {
        for (field, v) in [(0, 1), (4, 1), (8, 1), (12, 8)] {
            wr_u32(&mut g, face + field, v);
        }
    }
    wr_u32(&mut g, at::SAMPLE_DESC, 1);
    g
}

/// Cuts a stream into `(type, payload offset, payload size)`. `Err(type)` for a type that is not
/// known or a payload that runs past the end of the stream.
fn split_stream(stream: &[u8]) -> Result<Vec<(u32, usize, usize)>, u32> {
    let mut found = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= stream.len() {
        let ty = rd_u32(stream, pos);
        let (align, size) = subobject_layout(ty).ok_or(ty)?;
        let start = (pos + 4).next_multiple_of(align);
        if start + size > stream.len() {
            return Err(ty);
        }
        found.push((ty, start, size));
        pos = (start + size).next_multiple_of(8);
    }
    Ok(found)
}

fn unpack_compute(stream: &[u8], subs: &[(u32, usize, usize)]) -> PlainComputeDesc {
    let mut c = PlainComputeDesc { root_sig: 0, cs_ptr: 0, cs_len: 0, node_mask: 0, _pad0: 0, cached_ptr: 0, cached_len: 0, flags: 0, _pad1: 0 };
    for &(ty, at, _) in subs {
        match ty {
            0 => c.root_sig = rd_usize(stream, at),
            6 => (c.cs_ptr, c.cs_len) = (rd_usize(stream, at), rd_usize(stream, at + 8)),
            18 => c.node_mask = rd_u32(stream, at),
            19 => (c.cached_ptr, c.cached_len) = (rd_usize(stream, at), rd_usize(stream, at + 8)),
            20 => c.flags = rd_u32(stream, at),
            _ => {}
        }
    }
    c
}

/// Reads a stream into a graphics description and the amplification and mesh shader pair
/// (`[AS, MS]`, 32 bytes) that the plain description has no room for.
fn unpack_graphics(stream: &[u8], subs: &[(u32, usize, usize)]) -> ([u8; GRAPHICS_DESC_SIZE], [u8; 32]) {
    let mut g = default_graphics_desc();
    let mut mesh_shaders = [0u8; 32];
    for &(ty, start, size) in subs {
        let payload = &stream[start..start + size];
        if let Some(&(_, dst, len)) = COPIES.iter().find(|c| c.0 == ty) {
            g[dst..dst + len].copy_from_slice(&payload[..len]);
            continue;
        }
        match ty {
            // Eight formats come first and the count after them. Read the other way round, a
            // pipeline with one target had UNKNOWN in slot 0 and wrote nothing, so every frame
            // was black.
            15 => {
                wr_u32(&mut g, at::RT_COUNT, rd_u32(payload, 32).min(8));
                for i in 0..8 {
                    wr_u32(&mut g, at::RT_FORMATS + i * 4, pc_dxgi_format(rd_u32(payload, i * 4)));
                }
            }
            16 => wr_u32(&mut g, at::DEPTH_FORMAT, pc_dxgi_format(rd_u32(payload, 0))),
            23 => mesh_shaders[..16].copy_from_slice(payload),
            24 => mesh_shaders[16..].copy_from_slice(payload),
            _ => {}
        }
    }
    (g, mesh_shaders)
}

/// Unpacks a `D3D12_PIPELINE_STATE_STREAM_DESC` into a plain compute or graphics description, or
/// into our mesh description when the stream has a mesh shader. `Err(type)` for a subobject type
/// that is not known or cut off, `Ok(None)` when the stream has no CS, VS or MS.
pub fn unpack_stream(stream: &[u8]) -> Result<Option<UnpackedStream>, u32> {
    let subs = split_stream(stream)?;
    let has = |ty: u32| subs.iter().any(|s| s.0 == ty);
    let has_mesh = has(24);
    if has(6) && !has_mesh {
        return Ok(Some(UnpackedStream::Compute(unpack_compute(stream, &subs))));
    }
    if !has(1) && !has_mesh {
        return Ok(None);
    }
    let (mut g, mesh_shaders) = unpack_graphics(stream, &subs);
    // The console has topology types the PC does not (13 is a rectangle list). Triangles stand in
    // for them, or patches when a hull shader is set.
    if !(1..=4).contains(&rd_u32(&g, at::TOPOLOGY)) {
        let patches = rd_usize(&g, at::HS) != 0;
        wr_u32(&mut g, at::TOPOLOGY, if patches { 4 } else { 3 });
    }
    valid_states(&mut g);
    if has_mesh {
        // Our mesh description is the graphics one followed by the amplification and mesh shader.
        let mut m = Box::new([0u8; MESH_DESC_SIZE]);
        m[..GRAPHICS_DESC_SIZE].copy_from_slice(&g);
        m[GRAPHICS_DESC_SIZE..].copy_from_slice(&mesh_shaders);
        return Ok(Some(UnpackedStream::Mesh(m)));
    }
    Ok(Some(UnpackedStream::Graphics(Box::new(g))))
}

/// Replaces enum values the PC runtime does not know in the blend, rasterizer and depth stencil
/// parts of a plain graphics description. A console subobject carries zeros and console only
/// values where the PC enums start at 1. Native D3D12 refuses them, and the NVIDIA driver faulted
/// on "unrecognized" fill mode, cull mode and stencil fail operation.
pub fn valid_states(g: &mut [u8; GRAPHICS_DESC_SIZE]) {
    const BLEND_FACTORS: std::ops::RangeInclusive<u32> = 1..=19;
    const BLEND_OPS: std::ops::RangeInclusive<u32> = 1..=5;
    const STENCIL_FIELDS: [(usize, u32); 4] = [(0, 1), (4, 1), (8, 1), (12, 8)];
    let mut fix = |off: usize, valid: std::ops::RangeInclusive<u32>, default: u32| {
        if !valid.contains(&rd_u32(g, off)) {
            wr_u32(g, off, default);
        }
    };
    for i in 0..8 {
        let rt = at::render_target(i);
        fix(rt + 8, BLEND_FACTORS, 2);
        fix(rt + 12, BLEND_FACTORS, 1);
        fix(rt + 16, BLEND_OPS, 1);
        fix(rt + 20, BLEND_FACTORS, 2);
        fix(rt + 24, BLEND_FACTORS, 1);
        fix(rt + 28, BLEND_OPS, 1);
        fix(rt + 32, 0..=15, 4);
    }
    fix(at::RASTERIZER, 2..=3, 3);
    fix(at::RASTERIZER + 4, 1..=3, 1);
    fix(at::DEPTH_STENCIL + 4, 0..=1, 1);
    fix(at::DEPTH_STENCIL + 8, 1..=8, 8);
    for face in [at::DEPTH_STENCIL + 20, at::DEPTH_STENCIL + 36] {
        for (field, default) in STENCIL_FIELDS {
            fix(face + field, 1..=8, default);
        }
    }
    sanitize_integer_rt_blend(g);
}

/// DXGI formats of the integer kinds (`*_UINT` and `*_SINT`). PC D3D12 forbids blending on a
/// render target of these, while console hardware ignores it silently.
const INTEGER_FORMATS: [u32; 21] = [3, 4, 7, 8, 12, 14, 17, 18, 25, 30, 32, 36, 38, 42, 43, 50, 52, 57, 59, 62, 64];

/// Whether `fmt` is an integer (`*_UINT` or `*_SINT`) DXGI render target format.
pub fn is_integer_dxgi_format(fmt: u32) -> bool {
    INTEGER_FORMATS.contains(&fmt)
}

/// Switches blending and logic operations off on every active integer render target so that PC
/// D3D12 accepts a console pipeline that left them on.
///
/// Without independent blend only slot 0 holds the state. When it blends and integer and other
/// targets are mixed, the state is promoted to independent blend: slot 0 is copied to all slots,
/// logic operations go off everywhere (D3D12 wants that with independent blend) and the integer
/// slots stop blending, so the other targets keep it.
fn sanitize_integer_rt_blend(g: &mut [u8; GRAPHICS_DESC_SIZE]) {
    let active = (rd_u32(g, at::RT_COUNT) as usize).min(8);
    let integer: Vec<bool> = (0..active).map(|i| is_integer_dxgi_format(rd_u32(g, at::RT_FORMATS + i * 4))).collect();
    if !integer.contains(&true) {
        return;
    }
    // Blend and logic operation enable flags are the first two words of a target block.
    let switch_off = |g: &mut [u8; GRAPHICS_DESC_SIZE], slot: usize| {
        wr_u32(g, at::render_target(slot), 0);
        wr_u32(g, at::render_target(slot) + 4, 0);
    };
    if rd_u32(g, at::INDEPENDENT_BLEND) != 0 {
        for slot in (0..active).filter(|&i| integer[i]) {
            switch_off(g, slot);
        }
        return;
    }
    let blends = rd_u32(g, at::render_target(0)) != 0;
    let logic = rd_u32(g, at::render_target(0) + 4) != 0;
    if !blends && !logic {
        return;
    }
    if blends && integer.contains(&false) {
        wr_u32(g, at::INDEPENDENT_BLEND, 1);
        wr_u32(g, at::render_target(0) + 4, 0);
        let first = at::render_target(0);
        for slot in 1..8 {
            g.copy_within(first..first + at::RENDER_TARGET_STRIDE, at::render_target(slot));
        }
        for slot in (0..active).filter(|&i| integer[i]) {
            switch_off(g, slot);
        }
    } else {
        switch_off(g, 0);
    }
}

/// Reads `pRootSignature` of a plain graphics description.
pub fn graphics_root_sig(g: &[u8; GRAPHICS_DESC_SIZE]) -> usize {
    rd_usize(g, at::ROOT_SIGNATURE)
}

/// Replaces `pRootSignature` of a plain graphics description.
pub fn set_graphics_root_sig(g: &mut [u8; GRAPHICS_DESC_SIZE], rs: usize) {
    g[at::ROOT_SIGNATURE..at::ROOT_SIGNATURE + 8].copy_from_slice(&(rs as u64).to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Appends one subobject to a stream the way a builder lays it out.
    fn push_sub(stream: &mut Vec<u8>, ty: u32, payload: &[u8]) {
        let (align, size) = subobject_layout(ty).expect("a known type");
        assert_eq!(payload.len(), size, "payload of type {ty}");
        stream.resize(stream.len().next_multiple_of(8), 0);
        stream.extend_from_slice(&ty.to_le_bytes());
        stream.resize(stream.len().next_multiple_of(align), 0);
        stream.extend_from_slice(payload);
    }

    fn bytecode(ptr: u64, len: u64) -> Vec<u8> {
        [ptr.to_le_bytes(), len.to_le_bytes()].concat()
    }

    fn graphics(stream: &[u8]) -> Box<[u8; GRAPHICS_DESC_SIZE]> {
        match unpack_stream(stream) {
            Ok(Some(UnpackedStream::Graphics(g))) => g,
            other => panic!("expected a graphics pipeline, got {other:?}"),
        }
    }

    fn vs_only() -> Vec<u8> {
        let mut s = Vec::new();
        push_sub(&mut s, 1, &bytecode(0x9000, 64));
        s
    }

    #[cfg(windows)]
    #[test]
    fn field_offsets_agree_with_the_windows_sdk_structure() {
        use std::mem::{offset_of, size_of};
        use windows::Win32::Graphics::Direct3D12::*;
        type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        assert_eq!(size_of::<G>(), GRAPHICS_DESC_SIZE);
        let expected = [
            (offset_of!(G, pRootSignature), at::ROOT_SIGNATURE),
            (offset_of!(G, VS), at::VS),
            (offset_of!(G, PS), at::PS),
            (offset_of!(G, DS), at::DS),
            (offset_of!(G, HS), at::HS),
            (offset_of!(G, GS), at::GS),
            (offset_of!(G, StreamOutput), at::STREAM_OUTPUT),
            (offset_of!(G, BlendState), at::BLEND),
            (offset_of!(G, SampleMask), at::SAMPLE_MASK),
            (offset_of!(G, RasterizerState), at::RASTERIZER),
            (offset_of!(G, DepthStencilState), at::DEPTH_STENCIL),
            (offset_of!(G, InputLayout), at::INPUT_LAYOUT),
            (offset_of!(G, IBStripCutValue), at::STRIP_CUT),
            (offset_of!(G, PrimitiveTopologyType), at::TOPOLOGY),
            (offset_of!(G, NumRenderTargets), at::RT_COUNT),
            (offset_of!(G, RTVFormats), at::RT_FORMATS),
            (offset_of!(G, DSVFormat), at::DEPTH_FORMAT),
            (offset_of!(G, SampleDesc), at::SAMPLE_DESC),
            (offset_of!(G, NodeMask), at::NODE_MASK),
            (offset_of!(G, CachedPSO), at::CACHED),
            (offset_of!(G, Flags), at::FLAGS),
        ];
        for (sdk, ours) in expected {
            assert_eq!(sdk, ours);
        }
        assert_eq!(size_of::<D3D12_STREAM_OUTPUT_DESC>(), 32);
        assert_eq!(size_of::<D3D12_RASTERIZER_DESC>(), 44);
        assert_eq!(size_of::<D3D12_BLEND_DESC>(), 328);
    }

    #[cfg(windows)]
    #[test]
    fn console_numbers_match_the_pc_ones_except_for_the_mesh_shaders() {
        use windows::Win32::Graphics::Direct3D12::*;
        for (ours, pc) in [
            (1, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS),
            (6, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS),
            (11, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL),
            (15, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS),
            (21, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1),
        ] {
            assert_eq!(ours, pc.0);
        }
        assert_eq!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS.0, 24);
        assert_eq!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS.0, 25);
        assert!(subobject_layout(25).is_none());
    }

    #[test]
    fn the_layout_table_has_one_entry_per_type_in_order() {
        for (i, &(ty, align, size)) in SUBOBJECTS.iter().enumerate() {
            assert_eq!(ty as usize, i);
            assert!(align == 4 || align == 8);
            assert!(size >= 4);
        }
        assert_eq!(subobject_layout(7), Some((8, 32)));
        assert_eq!(subobject_layout(25), None);
    }

    #[test]
    fn a_compute_stream_keeps_its_pointers_flags_and_cache() {
        let mut s = Vec::new();
        push_sub(&mut s, 0, &0xAB00u64.to_le_bytes());
        push_sub(&mut s, 6, &bytecode(0xC000, 123));
        push_sub(&mut s, 18, &5u32.to_le_bytes());
        push_sub(&mut s, 19, &bytecode(0xD000, 77));
        push_sub(&mut s, 20, &2u32.to_le_bytes());
        let Ok(Some(UnpackedStream::Compute(c))) = unpack_stream(&s) else { panic!("expected compute") };
        assert_eq!((c.root_sig, c.cs_ptr, c.cs_len), (0xAB00, 0xC000, 123));
        assert_eq!((c.node_mask, c.cached_ptr, c.cached_len, c.flags), (5, 0xD000, 77, 2));
        assert_eq!(std::mem::size_of::<PlainComputeDesc>(), 56);
    }

    #[test]
    fn streams_without_a_shader_or_with_a_bad_subobject_are_told_apart() {
        let mut s = Vec::new();
        push_sub(&mut s, 9, &1u32.to_le_bytes());
        assert!(matches!(unpack_stream(&s), Ok(None)));
        assert!(matches!(unpack_stream(&[]), Ok(None)));
        let mut unknown = vs_only();
        unknown.resize(unknown.len().next_multiple_of(8), 0);
        unknown.extend_from_slice(&77u32.to_le_bytes());
        assert!(matches!(unpack_stream(&unknown), Err(77)));
        let mut cut = vs_only();
        cut.truncate(cut.len() - 4);
        assert!(matches!(unpack_stream(&cut), Err(1)));
    }

    #[test]
    fn a_graphics_stream_fills_fields_and_defaults_the_rest() {
        let mut s = Vec::new();
        push_sub(&mut s, 0, &0x1234_5678u64.to_le_bytes());
        push_sub(&mut s, 1, &bytecode(0xAA00, 992));
        push_sub(&mut s, 2, &bytecode(0xBB00, 100));
        push_sub(&mut s, 16, &118u32.to_le_bytes());
        let g = graphics(&s);
        assert_eq!(graphics_root_sig(&g), 0x1234_5678);
        assert_eq!(rd_usize(&g[..], at::VS), 0xAA00);
        assert_eq!(rd_usize(&g[..], at::VS + 8), 992);
        assert_eq!(rd_usize(&g[..], at::PS), 0xBB00);
        assert_eq!(rd_u32(&g[..], at::DEPTH_FORMAT), 45, "console D16_UNORM_S8_UINT becomes D24_UNORM_S8_UINT");
        assert_eq!(rd_u32(&g[..], at::SAMPLE_DESC), 1);
        assert_eq!(rd_u32(&g[..], at::SAMPLE_MASK), u32::MAX);
        assert_eq!(rd_u32(&g[..], at::DEPTH_STENCIL + 8), 2, "depth test less");
    }

    #[test]
    fn every_graphics_subobject_type_is_accepted_even_when_zeroed() {
        let mut s = Vec::new();
        for ty in (0..=21).filter(|&t| t != 6) {
            let (_, size) = subobject_layout(ty).unwrap();
            push_sub(&mut s, ty, &vec![0u8; size]);
        }
        let g = graphics(&s);
        assert_eq!(rd_u32(&g[..], at::RASTERIZER), 3, "a zero fill mode is made valid");
    }

    #[test]
    fn the_depth_stencil_with_bounds_flag_copies_only_the_common_part() {
        let mut payload = vec![0u8; 56];
        payload[4] = 1;
        payload[52] = 0xEE;
        let mut s = vs_only();
        push_sub(&mut s, 21, &payload);
        let g = graphics(&s);
        assert_eq!(g[at::DEPTH_STENCIL + 4], 1);
        assert_eq!(rd_u32(&g[..], at::DEPTH_STENCIL + 52), 0, "the bounds flag is not in the plain description");
    }

    #[test]
    fn render_target_formats_come_before_their_count() {
        let mut s = vs_only();
        let mut payload = Vec::new();
        for v in [28u32, 10, 0, 0, 0, 0, 0, 0, 2] {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        push_sub(&mut s, 15, &payload);
        let g = graphics(&s);
        assert_eq!(rd_u32(&g[..], at::RT_COUNT), 2);
        assert_eq!(rd_u32(&g[..], at::RT_FORMATS), 28);
        assert_eq!(rd_u32(&g[..], at::RT_FORMATS + 4), 10);
        assert_eq!(rd_u32(&g[..], at::RT_FORMATS + 8), 0);
    }

    #[test]
    fn a_target_count_above_eight_is_clamped() {
        let mut s = vs_only();
        let mut payload = vec![0u8; 36];
        payload[32] = 200;
        push_sub(&mut s, 15, &payload);
        assert_eq!(rd_u32(&graphics(&s)[..], at::RT_COUNT), 8);
    }

    #[test]
    fn console_only_topology_types_fall_back_to_triangles_or_patches() {
        let mut s = vs_only();
        push_sub(&mut s, 14, &13u32.to_le_bytes());
        assert_eq!(rd_u32(&graphics(&s)[..], at::TOPOLOGY), 3);
        let mut with_hull = vs_only();
        push_sub(&mut with_hull, 4, &bytecode(0x7000, 8));
        push_sub(&mut with_hull, 14, &0u32.to_le_bytes());
        assert_eq!(rd_u32(&graphics(&with_hull)[..], at::TOPOLOGY), 4);
        let mut line = vs_only();
        push_sub(&mut line, 14, &2u32.to_le_bytes());
        assert_eq!(rd_u32(&graphics(&line)[..], at::TOPOLOGY), 2, "a PC value stays");
    }

    #[test]
    fn a_mesh_shader_makes_a_mesh_description_with_both_shaders() {
        let mut s = Vec::new();
        push_sub(&mut s, 23, &bytecode(0xA000, 10));
        push_sub(&mut s, 24, &bytecode(0xB000, 20));
        let Ok(Some(UnpackedStream::Mesh(m))) = unpack_stream(&s) else { panic!("expected mesh") };
        assert_eq!(rd_usize(&m[..], GRAPHICS_DESC_SIZE), 0xA000);
        assert_eq!(rd_usize(&m[..], GRAPHICS_DESC_SIZE + 8), 10);
        assert_eq!(rd_usize(&m[..], GRAPHICS_DESC_SIZE + 16), 0xB000);
        assert_eq!(rd_usize(&m[..], GRAPHICS_DESC_SIZE + 24), 20);
    }

    #[test]
    fn zeroed_enums_become_valid_and_valid_ones_stay() {
        let mut g = [0u8; GRAPHICS_DESC_SIZE];
        wr_u32(&mut g, at::RASTERIZER + 4, 2);
        valid_states(&mut g);
        assert_eq!(rd_u32(&g, at::RASTERIZER), 3);
        assert_eq!(rd_u32(&g, at::RASTERIZER + 4), 2);
        assert_eq!(rd_u32(&g, at::DEPTH_STENCIL + 20), 1);
        assert_eq!(rd_u32(&g, at::DEPTH_STENCIL + 32), 8);
        assert_eq!(rd_u32(&g, at::render_target(0) + 8), 2);
        assert_eq!(rd_u32(&g, at::render_target(7) + 12), 1);
        assert_eq!(rd_u32(&g, at::render_target(0) + 32), 0, "logic op CLEAR is valid");
    }

    const UNORM: u32 = 28;
    const R32_UINT: u32 = 42;
    const R16G16_SINT: u32 = 38;

    fn targets(formats: &[u32], independent: bool, blend: &[usize]) -> [u8; GRAPHICS_DESC_SIZE] {
        let mut g = default_graphics_desc();
        wr_u32(&mut g, at::RT_COUNT, formats.len() as u32);
        for (i, &f) in formats.iter().enumerate() {
            wr_u32(&mut g, at::RT_FORMATS + i * 4, f);
        }
        wr_u32(&mut g, at::INDEPENDENT_BLEND, independent as u32);
        for &slot in blend {
            wr_u32(&mut g, at::render_target(slot), 1);
            wr_u32(&mut g, at::render_target(slot) + 4, 1);
        }
        valid_states(&mut g);
        g
    }

    #[test]
    fn integer_targets_stop_blending_under_independent_blend() {
        let g = targets(&[UNORM, R32_UINT, R16G16_SINT], true, &[0, 1, 2]);
        let flags = |slot| (rd_u32(&g, at::render_target(slot)), rd_u32(&g, at::render_target(slot) + 4));
        assert_eq!(flags(0), (1, 1), "a normalized target keeps both");
        assert_eq!(flags(1), (0, 0));
        assert_eq!(flags(2), (0, 0));
        assert_eq!(g[at::render_target(1) + 36], 0x0F, "the write mask stays");
    }

    #[test]
    fn mixed_targets_without_independent_blend_are_promoted() {
        let mut g = default_graphics_desc();
        wr_u32(&mut g, at::render_target(0) + 8, 5);
        wr_u32(&mut g, at::RT_COUNT, 2);
        wr_u32(&mut g, at::RT_FORMATS, UNORM);
        wr_u32(&mut g, at::RT_FORMATS + 4, R32_UINT);
        wr_u32(&mut g, at::render_target(0), 1);
        valid_states(&mut g);
        assert_eq!(rd_u32(&g, at::INDEPENDENT_BLEND), 1);
        assert_eq!(rd_u32(&g, at::render_target(0)), 1);
        assert_eq!(rd_u32(&g, at::render_target(1)), 0);
        assert_eq!(rd_u32(&g, at::render_target(1) + 8), 5, "the other fields follow slot 0");
        assert_eq!(rd_u32(&g, at::render_target(5) + 8), 5);
    }

    #[test]
    fn only_integer_targets_without_independent_blend_switch_slot_zero_off() {
        let g = targets(&[R32_UINT], false, &[0]);
        assert_eq!(rd_u32(&g, at::INDEPENDENT_BLEND), 0);
        assert_eq!(rd_u32(&g, at::render_target(0)), 0);
        assert_eq!(rd_u32(&g, at::render_target(0) + 4), 0);
    }

    #[test]
    fn pipelines_without_integer_targets_or_targets_at_all_are_left_alone() {
        let g = targets(&[UNORM], false, &[0]);
        assert_eq!((rd_u32(&g, at::render_target(0)), rd_u32(&g, at::INDEPENDENT_BLEND)), (1, 0));
        let none = targets(&[], false, &[0]);
        assert_eq!(rd_u32(&none, at::render_target(0)), 1);
    }

    #[test]
    fn root_signature_accessors_round_trip() {
        let mut g = [0u8; GRAPHICS_DESC_SIZE];
        set_graphics_root_sig(&mut g, 0xDEAD_BEEF_0000);
        assert_eq!(graphics_root_sig(&g), 0xDEAD_BEEF_0000);
    }

    #[test]
    fn integer_formats_are_the_uint_and_sint_ones() {
        for f in INTEGER_FORMATS {
            assert!(is_integer_dxgi_format(f));
        }
        for f in [0, 2, 10, 28, 41, 45, 87] {
            assert!(!is_integer_dxgi_format(f), "{f}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn integer_format_numbers_are_those_of_the_windows_sdk() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let sdk = [
            DXGI_FORMAT_R32G32B32A32_UINT,
            DXGI_FORMAT_R32G32B32A32_SINT,
            DXGI_FORMAT_R32G32B32_UINT,
            DXGI_FORMAT_R32G32B32_SINT,
            DXGI_FORMAT_R16G16B16A16_UINT,
            DXGI_FORMAT_R16G16B16A16_SINT,
            DXGI_FORMAT_R32G32_UINT,
            DXGI_FORMAT_R32G32_SINT,
            DXGI_FORMAT_R10G10B10A2_UINT,
            DXGI_FORMAT_R8G8B8A8_UINT,
            DXGI_FORMAT_R8G8B8A8_SINT,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_R16G16_SINT,
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R32_SINT,
            DXGI_FORMAT_R8G8_UINT,
            DXGI_FORMAT_R8G8_SINT,
            DXGI_FORMAT_R16_UINT,
            DXGI_FORMAT_R16_SINT,
            DXGI_FORMAT_R8_UINT,
            DXGI_FORMAT_R8_SINT,
        ];
        let mut ids: Vec<u32> = sdk.iter().map(|f| f.0 as u32).collect();
        ids.sort_unstable();
        assert_eq!(ids, INTEGER_FORMATS);
    }
}
