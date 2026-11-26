//! Mesh pipelines (console device slot 86 with a mesh shader in its stream). The title creates about
//! of them while it boots.
//!
//! Our description of a mesh pipeline is a plain `D3D12_GRAPHICS_PIPELINE_STATE_DESC` (no vertex,
//! hull, domain or geometry shader, the pixel shader at its usual place, the fixed function state as
//! for any graphics pipeline, `stream_pso`) followed by the amplification and the mesh shader
//! (`D3D12_SHADER_BYTECODE` each). So the state code of the graphics pipelines serves it unchanged,
//! and the length tells it apart from the plain descriptions (`lazy_pso`, `direct_pso`).
//!
//! The PC runtime takes a mesh pipeline only as a stream (`ID3D12Device2::CreatePipelineState`):
//! [`pc_stream`] writes one from the description.

use crate::stream_pso::GRAPHICS_DESC_SIZE;

/// Offset of the amplification shader's `D3D12_SHADER_BYTECODE`.
pub const AS_AT: usize = GRAPHICS_DESC_SIZE;
/// Offset of the mesh shader's `D3D12_SHADER_BYTECODE`.
pub const MS_AT: usize = GRAPHICS_DESC_SIZE + 16;
/// The size of our mesh description.
pub const MESH_DESC_SIZE: usize = GRAPHICS_DESC_SIZE + 32;
/// Offset of the pixel shader in the graphics part.
pub const PS_AT: usize = 0x18;

/// `D3D12_PIPELINE_STATE_SUBOBJECT_TYPE` values of the PC stream.
const ROOT_SIGNATURE: u32 = 0;
const PS: u32 = 2;
const BLEND: u32 = 8;
const SAMPLE_MASK: u32 = 9;
const RASTERIZER: u32 = 10;
const DEPTH_STENCIL: u32 = 11;
const PRIMITIVE_TOPOLOGY: u32 = 14;
const RENDER_TARGET_FORMATS: u32 = 15;
const DEPTH_STENCIL_FORMAT: u32 = 16;
const SAMPLE_DESC: u32 = 17;
const NODE_MASK: u32 = 18;
const FLAGS: u32 = 20;
/// The PC numbers of the amplification and the mesh shader (the console's are one below).
const AS: u32 = 24;
const MS: u32 = 25;

fn read_u64(d: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(d[at..at + 8].try_into().unwrap_or_default())
}

/// Whether the `D3D12_SHADER_BYTECODE` at `at` names a shader.
pub fn has_shader(desc: &[u8], at: usize) -> bool {
    read_u64(desc, at) != 0 && read_u64(desc, at + 8) != 0
}

const MAX_PC_STREAM_BYTES: usize = 624;

/// Appends one subobject: its type, padding up to the payload's alignment, the payload, padding up
/// to the next pointer boundary (each subobject of a stream is pointer aligned).
#[inline]
fn push(stream: &mut Vec<u8>, ty: u32, align: usize, payload: &[u8]) {
    stream.extend_from_slice(&ty.to_le_bytes());
    let pre_pad = (align - (stream.len() & (align - 1))) & (align - 1);
    if pre_pad != 0 {
        stream.resize(stream.len() + pre_pad, 0);
    }
    stream.extend_from_slice(payload);
    let post_pad = (8 - (stream.len() & 7)) & 7;
    if post_pad != 0 {
        stream.resize(stream.len() + post_pad, 0);
    }
}

/// The PC pipeline stream of our mesh description `desc`: the root signature, the shaders it names,
/// the fixed function state, the targets, the sample description, the node mask and the flags. The
/// title's cached blob is the console driver's and is left out.
pub fn pc_stream(desc: &[u8; MESH_DESC_SIZE]) -> Vec<u8> {
    let mut s = Vec::with_capacity(MAX_PC_STREAM_BYTES);
    push(&mut s, ROOT_SIGNATURE, 8, &desc[0..8]);
    for (ty, at) in [(AS, AS_AT), (MS, MS_AT), (PS, PS_AT)] {
        if has_shader(desc, at) {
            push(&mut s, ty, 8, &desc[at..at + 16]);
        }
    }
    push(&mut s, BLEND, 4, &desc[0x078..0x1C0]);
    push(&mut s, SAMPLE_MASK, 4, &desc[0x1C0..0x1C4]);
    push(&mut s, RASTERIZER, 4, &desc[0x1C4..0x1F0]);
    push(&mut s, DEPTH_STENCIL, 4, &desc[0x1F0..0x224]);
    push(&mut s, PRIMITIVE_TOPOLOGY, 4, &desc[0x23C..0x240]);
    // D3D12_RT_FORMAT_ARRAY: the eight formats, then the count.
    let mut targets = [0u8; 36];
    targets[..32].copy_from_slice(&desc[0x244..0x264]);
    targets[32..].copy_from_slice(&desc[0x240..0x244]);
    push(&mut s, RENDER_TARGET_FORMATS, 4, &targets);
    push(&mut s, DEPTH_STENCIL_FORMAT, 4, &desc[0x264..0x268]);
    push(&mut s, SAMPLE_DESC, 4, &desc[0x268..0x270]);
    push(&mut s, NODE_MASK, 4, &desc[0x270..0x274]);
    push(&mut s, FLAGS, 4, &desc[0x288..0x28C]);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_pso::{UnpackedStream, subobject_layout, unpack_stream};

    /// The console's numbers of the amplification and the mesh shader (`stream_pso`).
    const CONSOLE_AS: u32 = 23;
    const CONSOLE_MS: u32 = 24;

    /// A console stream as the title passes it: root signature, mesh and pixel shader, one target.
    fn mesh_stream(with_as: bool) -> Vec<u8> {
        let mut s = Vec::new();
        push(&mut s, ROOT_SIGNATURE, 8, &0x1234_5678u64.to_le_bytes());
        if with_as {
            push(&mut s, CONSOLE_AS, 8, &[0x10u64.to_le_bytes(), 11u64.to_le_bytes()].concat());
        }
        push(&mut s, CONSOLE_MS, 8, &[0xA000u64.to_le_bytes(), 22u64.to_le_bytes()].concat());
        push(&mut s, PS, 8, &[0xB000u64.to_le_bytes(), 33u64.to_le_bytes()].concat());
        let mut targets = [0u8; 36];
        targets[0..4].copy_from_slice(&10u32.to_le_bytes());
        targets[32..36].copy_from_slice(&1u32.to_le_bytes());
        push(&mut s, RENDER_TARGET_FORMATS, 4, &targets);
        push(&mut s, DEPTH_STENCIL_FORMAT, 4, &118u32.to_le_bytes());
        s
    }

    #[test]
    fn a_mesh_stream_unpacks_to_a_mesh_description_with_its_stage_kinds() {
        let Ok(Some(UnpackedStream::Mesh(m))) = unpack_stream(&mesh_stream(true)) else {
            panic!("expected a mesh pipeline");
        };
        assert_eq!(read_u64(&m[..], 0), 0x1234_5678, "root signature");
        assert_eq!((read_u64(&m[..], AS_AT), read_u64(&m[..], AS_AT + 8)), (0x10, 11), "amplification shader");
        assert_eq!((read_u64(&m[..], MS_AT), read_u64(&m[..], MS_AT + 8)), (0xA000, 22), "mesh shader");
        assert_eq!((read_u64(&m[..], PS_AT), read_u64(&m[..], PS_AT + 8)), (0xB000, 33), "pixel shader");
        assert!(!has_shader(&m[..], 0x08), "no vertex shader");
        assert_eq!(u32::from_le_bytes(m[0x264..0x268].try_into().unwrap()), 45, "console depth format 118 mapped");
        assert_eq!(u32::from_le_bytes(m[0x1C4..0x1C8].try_into().unwrap()), 3, "fill mode made valid");
        let Ok(Some(UnpackedStream::Mesh(m))) = unpack_stream(&mesh_stream(false)) else {
            panic!("expected a mesh pipeline");
        };
        assert!(!has_shader(&m[..], AS_AT), "no amplification shader");
    }

    #[test]
    fn the_pc_stream_parses_back_with_the_same_shaders_and_targets() {
        let Ok(Some(UnpackedStream::Mesh(m))) = unpack_stream(&mesh_stream(true)) else {
            panic!("expected a mesh pipeline");
        };
        let pc = pc_stream(&m);
        assert!(pc.len().is_multiple_of(8));
        // Walk it as the PC runtime does.
        let mut types = Vec::new();
        let mut pos = 0;
        while pos < pc.len() {
            let ty = u32::from_le_bytes(pc[pos..pos + 4].try_into().unwrap());
            // The PC mesh numbers are not the console ones `subobject_layout` reads.
            let (align, size) = if ty == AS || ty == MS { (8, 16) } else { subobject_layout(ty).expect("a PC subobject") };
            let at = pos + align.max(4);
            if ty == MS {
                assert_eq!(read_u64(&pc, at), 0xA000);
            }
            if ty == RENDER_TARGET_FORMATS {
                assert_eq!(u32::from_le_bytes(pc[at..at + 4].try_into().unwrap()), 10, "target 0 format");
                assert_eq!(u32::from_le_bytes(pc[at + 32..at + 36].try_into().unwrap()), 1, "one target");
            }
            types.push(ty);
            pos = (at + size).next_multiple_of(8);
        }
        assert_eq!(types, [ROOT_SIGNATURE, AS, MS, PS, BLEND, SAMPLE_MASK, RASTERIZER, DEPTH_STENCIL, PRIMITIVE_TOPOLOGY, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT, SAMPLE_DESC, NODE_MASK, FLAGS]);
    }

    #[test]
    fn pc_stream_stays_within_max_capacity_and_eight_byte_alignment() {
        let Ok(Some(UnpackedStream::Mesh(with_as))) = unpack_stream(&mesh_stream(true)) else {
            panic!("expected a mesh pipeline");
        };
        let full = pc_stream(&with_as);
        assert!(full.len() <= MAX_PC_STREAM_BYTES);
        assert!(full.len().is_multiple_of(8));

        let Ok(Some(UnpackedStream::Mesh(without_as))) = unpack_stream(&mesh_stream(false)) else {
            panic!("expected a mesh pipeline");
        };
        let compact = pc_stream(&without_as);
        assert!(compact.len() < full.len());
        assert!(compact.len().is_multiple_of(8));
    }
}
