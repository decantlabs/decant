//! Console semantic names on the interface of an imported vertex shader: the title's input layout
//! names `POSITION`, `NORMAL` and `TEXCOORD`, so the signature parts must be those DXC writes for the
//! same shader in HLSL. Compared with `dxc` (Windows SDK or Vulkan SDK) when it is installed.

mod common;

use std::path::PathBuf;

use decant_dxil::spirv::{Bindings, import_with};

fn find(tool: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(tool.to_uppercase().replace('-', "_")) {
        return Some(PathBuf::from(p));
    }
    let exe = format!("{tool}.exe");
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir("C:\\VulkanSDK") {
        roots.extend(rd.filter_map(|e| e.ok()).map(|e| e.path().join("Bin")));
    }
    if let Ok(rd) = std::fs::read_dir("C:\\Program Files (x86)\\Windows Kits\\10\\bin") {
        roots.extend(rd.filter_map(|e| e.ok()).map(|e| e.path().join("x64")));
    }
    roots.sort();
    roots.into_iter().rev().map(|r| r.join(&exe)).find(|p| p.is_file())
}

fn assemble(name: &str, text: &str) -> Option<Vec<u32>> {
    let tool = find("spirv-as")?;
    let dir = std::env::temp_dir().join(format!("decant-dxil-names-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join(format!("{name}.spvasm")), dir.join(format!("{name}.spv")));
    std::fs::write(&src, text).ok()?;
    let status = std::process::Command::new(tool).arg("--target-env").arg("vulkan1.1").arg(&src).arg("-o").arg(&dst).status().ok()?;
    assert!(status.success(), "spirv-as rejected {name}");
    let bytes = std::fs::read(&dst).ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// The data of the part `fourcc` of a DXBC container.
fn part<'a>(container: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    let count = u32::from_le_bytes(container[28..32].try_into().ok()?) as usize;
    for i in 0..count {
        let off = u32::from_le_bytes(container[32 + 4 * i..36 + 4 * i].try_into().ok()?) as usize;
        let size = u32::from_le_bytes(container[off + 4..off + 8].try_into().ok()?) as usize;
        if &container[off..off + 4] == fourcc {
            return Some(&container[off + 8..off + 8 + size]);
        }
    }
    None
}

const HLSL: &str = "struct I { float3 p : POSITION; float3 n : NORMAL; float2 uv : TEXCOORD0; };
struct O { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
O main(I i) { O o; o.pos = float4(i.p + i.n, 1); o.uv = i.uv; return o; }
";

const SPIRV: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %p %n %uv %pos %ouv
OpDecorate %p Location 0
OpDecorate %n Location 1
OpDecorate %uv Location 2
OpDecorate %pos BuiltIn Position
OpDecorate %ouv Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v3 = OpTypeVector %float 3
%v4 = OpTypeVector %float 4
%pi3 = OpTypePointer Input %v3
%pi2 = OpTypePointer Input %v2
%po4 = OpTypePointer Output %v4
%po2 = OpTypePointer Output %v2
%p = OpVariable %pi3 Input
%n = OpVariable %pi3 Input
%uv = OpVariable %pi2 Input
%pos = OpVariable %po4 Output
%ouv = OpVariable %po2 Output
%one = OpConstant %float 1
%main = OpFunction %void None %fn
%entry = OpLabel
%lp = OpLoad %v3 %p
%ln = OpLoad %v3 %n
%luv = OpLoad %v2 %uv
%s = OpFAdd %v3 %lp %ln
%x = OpCompositeExtract %float %s 0
%y = OpCompositeExtract %float %s 1
%z = OpCompositeExtract %float %s 2
%q = OpCompositeConstruct %v4 %x %y %z %one
OpStore %pos %q
OpStore %ouv %luv
OpReturn
OpFunctionEnd
"#;

#[test]
fn console_names_give_the_signature_dxc_writes() {
    let (Some(words), Some(dxc)) = (assemble("names", SPIRV), find("dxc")) else {
        println!("SKIPPED: spirv-as or dxc not found (set SPIRV_AS, DXC)");
        return;
    };
    let dir = std::env::temp_dir().join(format!("decant-dxil-names-hlsl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, dst) = (dir.join("v.hlsl"), dir.join("v.dxbc"));
    std::fs::write(&src, HLSL).unwrap();
    let status = std::process::Command::new(&dxc).args(["-T", "vs_6_6", "-E", "main", "-validator-version", "1.6", "-Fo"]).arg(&dst).arg(&src).status().expect("dxc");
    assert!(status.success());
    let reference = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    let mut b = Bindings::new();
    b.name(false, 0, 0, "POSITION", 0).name(false, 1, 0, "NORMAL", 0).name(false, 2, 0, "TEXCOORD", 0).name(true, 1, 0, "TEXCOORD", 0);
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    let container = common::write_and_validate("spirv_names", &imported.shader);
    for fourcc in [b"ISG1", b"OSG1"] {
        assert_eq!(part(&container, fourcc), part(&reference, fourcc), "{}", String::from_utf8_lossy(fourcc));
    }
}
