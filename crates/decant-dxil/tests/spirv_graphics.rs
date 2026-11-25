//! The SPIR-V importer on hand written vertex and fragment modules, drawn on a D3D12 device.
//! Skipped with a message when `spirv-as` or the device is missing.
#![cfg(windows)]

mod common;

use std::path::PathBuf;

use common::gfx::{Attribute, TARGET_SIZE, Texture, pixel};
use common::gpu::open_device;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};

fn spirv_as() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SPIRV_AS") {
        return Some(PathBuf::from(p));
    }
    let rd = std::fs::read_dir("C:\\VulkanSDK").ok()?;
    let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path().join("Bin").join("spirv-as.exe")).filter(|p| p.is_file()).collect();
    dirs.sort();
    dirs.pop()
}

fn assemble(name: &str, text: &str) -> Option<Vec<u32>> {
    let tool = spirv_as()?;
    let dir = std::env::temp_dir().join(format!("decant-dxil-spirv-gfx-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join(format!("{name}.spvasm")), dir.join(format!("{name}.spv")));
    std::fs::write(&src, text).ok()?;
    let status = std::process::Command::new(tool).arg("--target-env").arg("vulkan1.1").arg(&src).arg("-o").arg(&dst).status().ok()?;
    assert!(status.success(), "spirv-as rejected {name}");
    let bytes = std::fs::read(&dst).ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Passes the position and a texture coordinate through.
const VERTEX: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %inpos %inuv %outuv %pos
OpDecorate %inpos Location 0
OpDecorate %inuv Location 1
OpDecorate %outuv Location 0
OpDecorate %pos BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%pi4 = OpTypePointer Input %v4
%pi2 = OpTypePointer Input %v2
%po4 = OpTypePointer Output %v4
%po2 = OpTypePointer Output %v2
%inpos = OpVariable %pi4 Input
%inuv = OpVariable %pi2 Input
%outuv = OpVariable %po2 Output
%pos = OpVariable %po4 Output
%main = OpFunction %void None %fn
%entry = OpLabel
%p = OpLoad %v4 %inpos
%u = OpLoad %v2 %inuv
OpStore %pos %p
OpStore %outuv %u
OpReturn
OpFunctionEnd
"#;

/// Samples a texture at the interpolated coordinate and discards the right edge (x above 0.9).
const FRAGMENT: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %uv Location 0
OpDecorate %color Location 0
OpDecorate %tex DescriptorSet 0
OpDecorate %tex Binding 0
OpDecorate %smp DescriptorSet 0
OpDecorate %smp Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%pi2 = OpTypePointer Input %v2
%po4 = OpTypePointer Output %v4
%pif = OpTypePointer Input %float
%uv = OpVariable %pi2 Input
%color = OpVariable %po4 Output
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%pimg = OpTypePointer UniformConstant %img
%tex = OpVariable %pimg UniformConstant
%sam = OpTypeSampler
%psam = OpTypePointer UniformConstant %sam
%smp = OpVariable %psam UniformConstant
%simg = OpTypeSampledImage %img
%uint = OpTypeInt 32 0
%c0 = OpConstant %uint 0
%f09 = OpConstant %float 0.9
%main = OpFunction %void None %fn
%entry = OpLabel
%t = OpLoad %img %tex
%s = OpLoad %sam %smp
%si = OpSampledImage %simg %t %s
%c = OpLoad %v2 %uv
%x_p = OpAccessChain %pif %uv %c0
%x = OpLoad %float %x_p
%edge = OpFOrdGreaterThan %bool %x %f09
OpSelectionMerge %go None
OpBranchConditional %edge %kill %go
%kill = OpLabel
OpKill
%go = OpLabel
%r = OpImageSampleImplicitLod %v4 %si %c
OpStore %color %r
OpReturn
OpFunctionEnd
"#;

/// The colour export of a translated console pixel shader: two halves packed with PackHalf2x16
/// and read back with UnpackHalf2x16 (opcode 62), the last two channels from UnpackUnorm4x8
/// (opcode 64) of 0xFF400000. The colour is (1, 0.5, 0.25, 1).
const UNPACK: &str = r#"
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %uv %color
OpExecutionMode %main OriginUpperLeft
OpDecorate %uv Location 0
OpDecorate %color Location 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%pi2 = OpTypePointer Input %v2
%po4 = OpTypePointer Output %v4
%uv = OpVariable %pi2 Input
%color = OpVariable %po4 Output
%f1 = OpConstant %float 1
%fh = OpConstant %float 0.5
%rg = OpConstantComposite %v2 %f1 %fh
%word = OpConstant %uint 0xFF400000
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpLoad %v2 %uv
%packed = OpExtInst %uint %glsl PackHalf2x16 %rg
%halves = OpExtInst %v2 %glsl UnpackHalf2x16 %packed
%bytes = OpExtInst %v4 %glsl UnpackUnorm4x8 %word
%out = OpVectorShuffle %v4 %halves %bytes 0 1 4 5
OpStore %color %out
OpReturn
OpFunctionEnd
"#;

#[test]
fn unpack_half_and_unorm_read_their_own_opcodes() {
    let (Some(vs_words), Some(ps_words)) = (assemble("unpack_vs", VERTEX), assemble("unpack_ps", UNPACK)) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let vs = decant_dxil::spirv::import(&vs_words).expect("vertex import");
    let ps = decant_dxil::spirv::import(&ps_words).expect("fragment import");
    let vsc = common::write_and_validate("unpack_vs", &vs.shader);
    let psc = common::write_and_validate("unpack_ps", &ps.shader);
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    let attributes = [
        Attribute { name: b"ATTRIB\0", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        Attribute { name: b"ATTRIB\0", index: 1, format: DXGI_FORMAT_R32G32_FLOAT, offset: 16 },
    ];
    let vertices = floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, 2.0, 1.0, -1.0, 3.0, 0.0, 1.0, 0.0, -1.0]);
    let out = gpu.render(&vsc, &psc, &attributes, 24, &vertices, 3, None).expect("render");
    let centre = pixel(&out, TARGET_SIZE / 2, TARGET_SIZE / 2);
    // 0.5 is 127.5 in eight bits, which the driver may round either way.
    assert!(matches!(centre, [255, 127 | 128, 64, 255]), "centre pixel {centre:?}");
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
fn a_vertex_and_a_fragment_module_draw_a_textured_triangle() {
    let (Some(vs_words), Some(ps_words)) = (assemble("gfx_vs", VERTEX), assemble("gfx_ps", FRAGMENT)) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let vs = decant_dxil::spirv::import(&vs_words).expect("vertex import");
    let ps = decant_dxil::spirv::import(&ps_words).expect("fragment import");
    assert_eq!(vs.shader.stage, decant_dxil::ir::Stage::Vertex);
    assert_eq!(ps.shader.stage, decant_dxil::ir::Stage::Pixel);
    // The interface the host wires: vertex inputs ATTRIB0 and ATTRIB1, the pixel input TEXCOORD0.
    assert_eq!(vs.shader.inputs.iter().map(|e| (e.name.as_str(), e.index)).collect::<Vec<_>>(), [("ATTRIB", 0), ("ATTRIB", 1)]);
    assert_eq!(ps.shader.inputs[0].name, "TEXCOORD");
    let vsc = common::write_and_validate("spirv_vs", &vs.shader);
    let psc = common::write_and_validate("spirv_ps", &ps.shader);
    let Some(gpu) = open_device() else {
        println!("SKIPPED: no D3D12 device with Shader Model 6.6");
        return;
    };
    println!("device: {}", gpu.name);
    let attributes = [
        Attribute { name: b"ATTRIB\0", index: 0, format: DXGI_FORMAT_R32G32B32A32_FLOAT, offset: 0 },
        Attribute { name: b"ATTRIB\0", index: 1, format: DXGI_FORMAT_R32G32_FLOAT, offset: 16 },
    ];
    let vertices = floats(&[-1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, 2.0, 1.0, -1.0, 3.0, 0.0, 1.0, 0.0, -1.0]);
    let texture = Texture { width: 2, height: 2, rgba: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255] };
    let out = gpu.render(&vsc, &psc, &attributes, 24, &vertices, 3, Some(&texture)).expect("render");
    let q = TARGET_SIZE / 4;
    assert_eq!(pixel(&out, q, q), [255, 0, 0, 255], "top left");
    assert_eq!(pixel(&out, 3 * q - 4, q), [0, 255, 0, 255], "top right");
    assert_eq!(pixel(&out, q, 3 * q), [0, 0, 255, 255], "bottom left");
    assert_eq!(pixel(&out, 3 * q - 4, 3 * q), [255, 255, 255, 255], "bottom right");
    assert_eq!(pixel(&out, TARGET_SIZE - 2, q), [0, 0, 0, 0], "the right edge is discarded");
}
