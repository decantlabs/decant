//! The SPIR-V importer with a binding map: descriptors land at the registers (or heap indices) the
//! host chose, as for the console shaders whose root signature is the title's. Hand written
//! modules, assembled with `spirv-as` when it is installed, validated with dxil.dll.

mod common;

use std::path::PathBuf;

use decant_dxil::ir::{Binding, ResourceClass};
use decant_dxil::spirv::{Bindings, PushLayout, RegisterClass, Target, import_with};

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
    let dir = std::env::temp_dir().join(format!("decant-dxil-spirv-bind-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join(format!("{name}.spvasm")), dir.join(format!("{name}.spv")));
    std::fs::write(&src, text).ok()?;
    let status = std::process::Command::new(tool).arg("--target-env").arg("vulkan1.1").arg(&src).arg("-o").arg(&dst).status().ok()?;
    assert!(status.success(), "spirv-as rejected {name}");
    let bytes = std::fs::read(&dst).ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Binding 0 is an array of four storage buffers, as in the cache. Each invocation reads word
/// `gid.x` of element 2 and of element 1, and writes their sum to element 3.
const BUFFERS: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %bufs DescriptorSet 0
OpDecorate %bufs Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%c4 = OpConstant %uint 4
%bufs_t = OpTypeArray %SSBO %c4
%pbufs = OpTypePointer StorageBuffer %bufs_t
%bufs = OpVariable %pbufs StorageBuffer
%pu = OpTypePointer StorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%main = OpFunction %void None %fn
%entry = OpLabel
%gp = OpAccessChain %pu_in %gid %c0
%x = OpLoad %uint %gp
%a_p = OpAccessChain %pu %bufs %c2 %c0 %x
%b_p = OpAccessChain %pu %bufs %c1 %c0 %x
%o_p = OpAccessChain %pu %bufs %c3 %c0 %x
%a = OpLoad %uint %a_p
%b = OpLoad %uint %b_p
%s = OpIAdd %uint %a %b
OpStore %o_p %s
OpReturn
OpFunctionEnd
"#;

fn resources_of(name: &str, bindings: &Bindings) -> Vec<(ResourceClass, u32, u32, Binding)> {
    let words = assemble(name, BUFFERS).expect("spirv-as");
    let imported = import_with(&words, bindings).unwrap_or_else(|e| panic!("import failed: {e}"));
    common::write_and_validate(&format!("spirv_bindings_{name}"), &imported.shader);
    imported.shader.resources.iter().map(|r| (r.class, r.space, r.lower_bound, r.binding)).collect()
}

#[test]
fn buffers_land_at_the_registers_the_host_chose() {
    if spirv_as().is_none() {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    }
    let mut b = Bindings::new();
    b.bind(0, 0, 2, Target::Register { class: RegisterClass::Srv, space: 0, register: 188 });
    b.bind(0, 0, 1, Target::Register { class: RegisterClass::Cbv, space: 0, register: 27 });
    b.bind(0, 0, 3, Target::Register { class: RegisterClass::Uav, space: 0, register: 9 });
    let got = resources_of("registers", &b);
    assert!(got.contains(&(ResourceClass::Srv, 0, 188, Binding::Bound)), "{got:?}");
    assert!(got.contains(&(ResourceClass::CBuffer, 0, 27, Binding::Bound)), "{got:?}");
    assert!(got.contains(&(ResourceClass::Uav, 0, 9, Binding::Bound)), "{got:?}");
    assert_eq!(got.len(), 3);
}

#[test]
fn an_unplaced_descriptor_and_a_runtime_index_are_refused() {
    let Some(words) = assemble("buffers_refused", BUFFERS) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    b.bind(0, 0, 2, Target::Register { class: RegisterClass::Srv, space: 0, register: 188 });
    let e = import_with(&words, &b).unwrap_err();
    assert_eq!(e.category, "binding:unmapped", "{e}");
    // Without a map the old numbering still works.
    assert!(decant_dxil::spirv::import(&words).is_ok());
}

#[test]
fn buffers_in_the_heap_are_indexed_directly() {
    if spirv_as().is_none() {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    }
    let mut b = Bindings::new();
    b.bind(0, 0, 2, Target::Heap(17));
    b.bind(0, 0, 1, Target::Register { class: RegisterClass::Srv, space: 0, register: 3 });
    b.bind(0, 0, 3, Target::Heap(40));
    let got = resources_of("heap", &b);
    assert_eq!(got.iter().filter(|r| r.3 == Binding::Heap).count(), 2, "{got:?}");
    let words = assemble("buffers_heap", BUFFERS).expect("spirv-as");
    let out = decant_dxil::write(&import_with(&words, &b).unwrap().shader).unwrap();
    assert!(out.report.resource_heap);
    assert!(!out.report.sampler_heap);
}

/// A texture and a sampler (bindings 1 and 2), sampled at the invocation id and stored to buffer 0.
const SAMPLED: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 8 8 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %tex DescriptorSet 0
OpDecorate %tex Binding 1
OpDecorate %smp DescriptorSet 0
OpDecorate %smp Binding 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v3uint = OpTypeVector %uint 3
%v2float = OpTypeVector %float 2
%v4float = OpTypeVector %float 4
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%pssbo = OpTypePointer StorageBuffer %SSBO
%out = OpVariable %pssbo StorageBuffer
%pu = OpTypePointer StorageBuffer %uint
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%pimg = OpTypePointer UniformConstant %img
%tex = OpVariable %pimg UniformConstant
%samp = OpTypeSampler
%psamp = OpTypePointer UniformConstant %samp
%smp = OpVariable %psamp UniformConstant
%simg = OpTypeSampledImage %img
%c0 = OpConstant %uint 0
%f0 = OpConstant %float 0
%f64 = OpConstant %float 0.015625
%main = OpFunction %void None %fn
%entry = OpLabel
%g = OpLoad %v3uint %gid
%x = OpCompositeExtract %uint %g 0
%y = OpCompositeExtract %uint %g 1
%fx = OpConvertUToF %float %x
%fy = OpConvertUToF %float %y
%u = OpFMul %float %fx %f64
%v = OpFMul %float %fy %f64
%uv = OpCompositeConstruct %v2float %u %v
%t = OpLoad %img %tex
%s = OpLoad %samp %smp
%si = OpSampledImage %simg %t %s
%c = OpImageSampleExplicitLod %v4float %si %uv Lod %f0
%r = OpCompositeExtract %float %c 0
%ru = OpBitcast %uint %r
%o = OpAccessChain %pu %out %c0 %x
OpStore %o %ru
OpReturn
OpFunctionEnd
"#;

#[test]
fn textures_and_samplers_come_from_the_heaps() {
    let Some(words) = assemble("sampled", SAMPLED) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::Register { class: RegisterClass::Uav, space: 0, register: 0 });
    b.bind(0, 1, 0, Target::Heap(5));
    b.bind(0, 2, 0, Target::SamplerHeap(2));
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    let out = decant_dxil::write(&imported.shader).unwrap();
    common::check_validator("spirv_bindings_sampled_heap", &out.container);
    assert!(out.report.resource_heap && out.report.sampler_heap, "{:?}", out.report);
    // The same module with registers: no heap.
    let mut r = Bindings::new();
    r.bind(0, 0, 0, Target::Register { class: RegisterClass::Uav, space: 0, register: 0 });
    r.bind(0, 1, 0, Target::Register { class: RegisterClass::Srv, space: 0, register: 12 });
    r.bind(0, 2, 0, Target::Register { class: RegisterClass::Sampler, space: 0, register: 3 });
    let imported = import_with(&words, &r).unwrap();
    let out = decant_dxil::write(&imported.shader).unwrap();
    common::check_validator("spirv_bindings_sampled_regs", &out.container);
    assert!(!out.report.resource_heap && !out.report.sampler_heap);
    assert_eq!(out.report.bound.iter().filter(|b| b.class == ResourceClass::Sampler && b.register == 3).count(), 1, "{:?}", out.report);
    // A sampler cannot be placed in the resource heap.
    let mut bad = Bindings::new();
    bad.bind(0, 0, 0, Target::Register { class: RegisterClass::Uav, space: 0, register: 0 });
    bad.bind(0, 1, 0, Target::Heap(5));
    bad.bind(0, 2, 0, Target::Heap(2));
    assert_eq!(import_with(&words, &bad).unwrap_err().category, "binding:class");
}

/// Buffers 0 (input) and 1 (output) are elements of binding 0. The push words are: 0 the group id
/// (data), 1 an offset, 2 and 3 the sizes of buffers 0 and 1. Invocation x of groups of 4 writes
/// `in[x] + pc[0] * 1000 + pc[1] + pc[2] * 7 + pc[3] * 11`.
const PUSH: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 4 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %bufs DescriptorSet 0
OpDecorate %bufs Binding 0
OpDecorate %pcarr ArrayStride 4
OpMemberDecorate %PC 0 Offset 0
OpDecorate %PC Block
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%c2 = OpConstant %uint 2
%bufs_t = OpTypeArray %SSBO %c2
%pbufs = OpTypePointer StorageBuffer %bufs_t
%bufs = OpVariable %pbufs StorageBuffer
%pu = OpTypePointer StorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c3 = OpConstant %uint 3
%c4 = OpConstant %uint 4
%c7 = OpConstant %uint 7
%c11 = OpConstant %uint 11
%c1000 = OpConstant %uint 1000
%pcarr = OpTypeArray %uint %c4
%PC = OpTypeStruct %pcarr
%ppc = OpTypePointer PushConstant %PC
%pc = OpVariable %ppc PushConstant
%ppcu = OpTypePointer PushConstant %uint
%main = OpFunction %void None %fn
%entry = OpLabel
%gp = OpAccessChain %pu_in %gid %c0
%x = OpLoad %uint %gp
%p0 = OpAccessChain %ppcu %pc %c0 %c0
%p1 = OpAccessChain %ppcu %pc %c0 %c1
%p2 = OpAccessChain %ppcu %pc %c0 %c2
%p3 = OpAccessChain %ppcu %pc %c0 %c3
%w0 = OpLoad %uint %p0
%w1 = OpLoad %uint %p1
%w2 = OpLoad %uint %p2
%w3 = OpLoad %uint %p3
%ip = OpAccessChain %pu %bufs %c0 %c0 %x
%op = OpAccessChain %pu %bufs %c1 %c0 %x
%in = OpLoad %uint %ip
%t0 = OpIMul %uint %w0 %c1000
%t2 = OpIMul %uint %w2 %c7
%t3 = OpIMul %uint %w3 %c11
%a = OpIAdd %uint %in %t0
%b = OpIAdd %uint %a %w1
%c = OpIAdd %uint %b %t2
%d = OpIAdd %uint %c %t3
OpStore %op %d
OpReturn
OpFunctionEnd
"#;

fn push_bindings() -> Bindings {
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::Register { class: RegisterClass::Srv, space: 0, register: 0 });
    b.bind(0, 0, 1, Target::Register { class: RegisterClass::Uav, space: 0, register: 0 });
    b.push = Some(PushLayout { data_end: 1, offsets_end: 2, sizes_end: 4, buffers: vec![Some((0, 0, 0)), Some((0, 0, 1))], group_id: vec![(0, 0)], registers: Vec::new() });
    b
}

#[test]
fn push_words_are_group_id_zero_offsets_and_buffer_sizes() {
    let Some(words) = assemble("push", PUSH) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let imported = import_with(&words, &push_bindings()).unwrap_or_else(|e| panic!("import failed: {e}"));
    // The block is gone: only the two buffers remain.
    assert_eq!(imported.shader.resources.len(), 2, "{:?}", imported.shader.resources);
    common::write_and_validate("spirv_bindings_push", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words as to_words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        // A root descriptor has no size, so the buffers sit in a heap as views of 64 bytes.
        let mut heap = Bindings::new();
        heap.bind(0, 0, 0, Target::Heap(0));
        heap.bind(0, 0, 1, Target::Heap(1));
        heap.push = push_bindings().push;
        let imported = import_with(&words, &heap).unwrap();
        let c = common::write_and_validate("spirv_bindings_push_heap", &imported.shader);
        let out = gpu.run(&c, &[Bind::HeapUav { slot: 0, size: 64 }, Bind::HeapUav { slot: 1, size: 64 }], [4, 1, 1]).expect("run");
        let got = to_words(&out[1]);
        for (x, g) in got.iter().enumerate() {
            let want = (x as u32 / 4) * 1000 + 64 * 7 + 64 * 11;
            assert_eq!(*g, want, "invocation {x}");
        }
        println!("device: {}", gpu.name);
    }
}

#[test]
fn the_size_of_a_constant_buffer_is_its_maximum() {
    let Some(words) = assemble("push_cbv", PUSH) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = push_bindings();
    b.bind(0, 0, 0, Target::Register { class: RegisterClass::Cbv, space: 0, register: 27 });
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    common::write_and_validate("spirv_bindings_push_cbv", &imported.shader);
}

#[test]
fn a_push_data_word_that_names_a_console_register_reads_the_mirror() {
    let Some(words) = assemble("push_mirror", PUSH) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::Heap(0));
    b.bind(0, 0, 1, Target::Heap(1));
    b.push = Some(PushLayout { data_end: 1, offsets_end: 2, sizes_end: 4, buffers: vec![Some((0, 0, 0)), Some((0, 0, 1))], group_id: Vec::new(), registers: vec![(0, 5)] });
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    assert_eq!(imported.user_data.as_ref().map(|u| u.dwords.clone()), Some(vec![5]));
    let c = common::write_and_validate("spirv_bindings_push_mirror", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words as to_words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let mut m = [0u32; 32];
        m[5] = 3;
        let mirror: Vec<u8> = m.iter().flat_map(|w| w.to_le_bytes()).collect();
        let out = gpu.run(&c, &[Bind::HeapUav { slot: 0, size: 64 }, Bind::HeapUav { slot: 1, size: 64 }, Bind::Mirror(mirror)], [4, 1, 1]).expect("run");
        for (x, g) in to_words(&out[1]).iter().enumerate() {
            assert_eq!(*g, 3 * 1000 + 64 * 7 + 64 * 11, "invocation {x}");
        }
    }
    #[cfg(not(windows))]
    let _ = c;
}

#[test]
fn a_push_data_word_that_is_not_the_group_id_is_refused() {
    let Some(words) = assemble("push_prefix", PUSH) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = push_bindings();
    b.push.as_mut().unwrap().group_id.clear();
    assert_eq!(import_with(&words, &b).unwrap_err().category, "push:prefix");
    let mut b = push_bindings();
    b.push.as_mut().unwrap().buffers[1] = None;
    assert_eq!(import_with(&words, &b).unwrap_err().category, "push:size-unmapped");
}
