//! The SPIR-V importer on hand written compute modules: assembled with `spirv-as` when it is
//! installed (the Vulkan SDK), imported, written, validated and run on a D3D12 device. Skipped
//! with a message when `spirv-as` or the device is missing.

mod common;

use std::path::PathBuf;

/// Finds `spirv-as`: `SPIRV_AS`, then the Vulkan SDK folders.
fn spirv_as() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SPIRV_AS") {
        return Some(PathBuf::from(p));
    }
    for root in ["C:\\VulkanSDK"] {
        let Ok(rd) = std::fs::read_dir(root) else { continue };
        let mut dirs: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path().join("Bin").join("spirv-as.exe")).filter(|p| p.is_file()).collect();
        dirs.sort();
        if let Some(p) = dirs.pop() {
            return Some(p);
        }
    }
    None
}

/// Assembles `text` to SPIR-V words, or `None` when there is no assembler.
fn assemble(name: &str, text: &str) -> Option<Vec<u32>> {
    let tool = spirv_as()?;
    let dir = std::env::temp_dir().join(format!("decant-dxil-spirv-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join(format!("{name}.spvasm")), dir.join(format!("{name}.spv")));
    std::fs::write(&src, text).ok()?;
    let status = std::process::Command::new(tool).arg("--target-env").arg("vulkan1.1").arg(&src).arg("-o").arg(&dst).status().ok()?;
    assert!(status.success(), "spirv-as rejected {name}");
    let bytes = std::fs::read(&dst).ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Each invocation sums or decrements over four words of the input depending on the parity of the
/// index, through function variables (SSA construction with phis), a loop and a branch.
const LOOP: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %in DescriptorSet 0
OpDecorate %in Binding 0
OpDecorate %in NonWritable
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%pssbo = OpTypePointer StorageBuffer %SSBO
%in = OpVariable %pssbo StorageBuffer
%out = OpVariable %pssbo StorageBuffer
%pu_sb = OpTypePointer StorageBuffer %uint
%pu_fn = OpTypePointer Function %uint
%pu_in = OpTypePointer Input %uint
%bool = OpTypeBool
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c4 = OpConstant %uint 4
%main = OpFunction %void None %fn
%entry = OpLabel
%acc = OpVariable %pu_fn Function
%i = OpVariable %pu_fn Function
%gx_p = OpAccessChain %pu_in %gid %c0
%gx = OpLoad %uint %gx_p
OpStore %acc %c0
OpStore %i %c0
OpBranch %head
%head = OpLabel
OpLoopMerge %merge %cont None
OpBranch %cond
%cond = OpLabel
%iv = OpLoad %uint %i
%lt = OpULessThan %bool %iv %c4
OpBranchConditional %lt %body %merge
%body = OpLabel
%base = OpIMul %uint %gx %c4
%idx = OpIAdd %uint %base %iv
%p = OpAccessChain %pu_sb %in %c0 %idx
%v = OpLoad %uint %p
%a = OpLoad %uint %acc
%odd = OpBitwiseAnd %uint %iv %c1
%isodd = OpINotEqual %bool %odd %c0
OpSelectionMerge %join None
OpBranchConditional %isodd %then %else
%then = OpLabel
%a1 = OpIAdd %uint %a %v
OpStore %acc %a1
OpBranch %join
%else = OpLabel
%a2 = OpISub %uint %a %c1
OpStore %acc %a2
OpBranch %join
%join = OpLabel
OpBranch %cont
%cont = OpLabel
%iv2 = OpLoad %uint %i
%in2 = OpIAdd %uint %iv2 %c1
OpStore %i %in2
OpBranch %head
%merge = OpLabel
%res = OpLoad %uint %acc
%op = OpAccessChain %pu_sb %out %c0 %gx
OpStore %op %res
OpReturn
OpFunctionEnd
"#;

fn loop_reference(input: &[u32], gx: usize) -> u32 {
    let mut acc = 0u32;
    for iv in 0..4usize {
        if iv & 1 != 0 {
            acc = acc.wrapping_add(input[gx * 4 + iv]);
        } else {
            acc = acc.wrapping_sub(1);
        }
    }
    acc
}

/// A function called with a pointer and a value (inlined), a workgroup array, a barrier, an
/// atomic add, a float conversion with GLSL.std.450 `Floor` and an `OpSwitch`.
const MIXED: &str = r#"
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %lid %gid
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %lid BuiltIn LocalInvocationId
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%bool = OpTypeBool
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%lid = OpVariable %pv3 Input
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%pssbo = OpTypePointer StorageBuffer %SSBO
%out = OpVariable %pssbo StorageBuffer
%pu_sb = OpTypePointer StorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%c63 = OpConstant %uint 63
%c64 = OpConstant %uint 64
%c264 = OpConstant %uint 264
%c2_5 = OpConstant %float 2.5
%arr64 = OpTypeArray %uint %c64
%pws = OpTypePointer Workgroup %arr64
%ws = OpVariable %pws Workgroup
%pu_ws = OpTypePointer Workgroup %uint
%fnadd = OpTypeFunction %uint %uint %uint
%add3 = OpFunction %uint None %fnadd
%pa = OpFunctionParameter %uint
%pb = OpFunctionParameter %uint
%lbl = OpLabel
%s = OpIAdd %uint %pa %pb
%s3 = OpIAdd %uint %s %c3
OpReturnValue %s3
OpFunctionEnd
%main = OpFunction %void None %fn
%entry = OpLabel
%lx_p = OpAccessChain %pu_in %lid %c0
%lx = OpLoad %uint %lx_p
%gx_p = OpAccessChain %pu_in %gid %c0
%gx = OpLoad %uint %gx_p
%slot = OpAccessChain %pu_ws %ws %lx
OpStore %slot %lx
OpControlBarrier %c2 %c2 %c264
%rev = OpISub %uint %c63 %lx
%rslot = OpAccessChain %pu_ws %ws %rev
%rv = OpLoad %uint %rslot
%call = OpFunctionCall %uint %add3 %rv %gx
%f = OpConvertUToF %float %call
%fm = OpFMul %float %f %c2_5
%fl = OpExtInst %float %glsl Floor %fm
%back = OpConvertFToU %uint %fl
%sel = OpUMod %uint %lx %c3
OpSelectionMerge %done None
OpSwitch %sel %other 0 %zero 1 %one
%zero = OpLabel
%r0 = OpIAdd %uint %back %c1
OpBranch %done
%one = OpLabel
%r1 = OpIAdd %uint %back %c2
OpBranch %done
%other = OpLabel
OpBranch %done
%done = OpLabel
%r = OpPhi %uint %r0 %zero %r1 %one %back %other
%op = OpAccessChain %pu_sb %out %c0 %gx
OpStore %op %r
OpReturn
OpFunctionEnd
"#;

fn mixed_reference(gx: u32) -> u32 {
    let lx = gx % 64;
    let rv = 63 - lx;
    let call = rv + gx + 3;
    let back = ((call as f32) * 2.5).floor() as u32;
    match lx % 3 {
        0 => back + 1,
        1 => back + 2,
        _ => back,
    }
}

/// Memory read and written through device addresses: the address of the window comes from a
/// storage buffer, the shader loads `window[4 + gx]` and stores `2 * it + 1` to `window[68 + gx]`.
const DEVICE_ADDRESS: &str = r#"
OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses
OpExtension "SPV_KHR_physical_storage_buffer"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %in DescriptorSet 0
OpDecorate %in Binding 0
OpDecorate %in NonWritable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%ulong = OpTypeInt 64 0
%v2uint = OpTypeVector %uint 2
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%pssbo = OpTypePointer StorageBuffer %SSBO
%in = OpVariable %pssbo StorageBuffer
%pu_sb = OpTypePointer StorageBuffer %uint
%psb_uint = OpTypePointer PhysicalStorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%l16 = OpConstant %ulong 16
%l256 = OpConstant %ulong 256
%main = OpFunction %void None %fn
%entry = OpLabel
%gx_p = OpAccessChain %pu_in %gid %c0
%gx = OpLoad %uint %gx_p
%p0 = OpAccessChain %pu_sb %in %c0 %c0
%lo = OpLoad %uint %p0
%p1 = OpAccessChain %pu_sb %in %c0 %c1
%hi = OpLoad %uint %p1
%pair = OpCompositeConstruct %v2uint %lo %hi
%base = OpBitcast %ulong %pair
%off = OpUConvert %ulong %gx
%off4 = OpShiftLeftLogical %ulong %off %c2
%a0 = OpIAdd %ulong %base %off4
%src = OpIAdd %ulong %a0 %l16
%dst = OpIAdd %ulong %src %l256
%ps = OpConvertUToPtr %psb_uint %src
%v = OpLoad %uint %ps Aligned 4
%d = OpIAdd %uint %v %v
%r = OpIAdd %uint %d %c1
%pd = OpConvertUToPtr %psb_uint %dst
OpStore %pd %r Aligned 4
OpReturn
OpFunctionEnd
"#;

#[test]
fn device_addresses_become_raw_buffer_accesses_in_a_window() {
    let Some(words) = assemble("device_address", DEVICE_ADDRESS) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let imported = decant_dxil::spirv::import(&words).expect("import");
    let dm = imported.device_memory.expect("the module uses device memory");
    assert_eq!(imported.shader.resources[dm.window as usize].space, decant_dxil::spirv::DEVICE_MEMORY_SPACE);
    assert_eq!(imported.shader.resources[dm.base as usize].size_bytes, 16);
    let c = common::write_and_validate("spirv_device_address", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, bytes_of, open_device, words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let init: Vec<u32> = (0..256u32).map(|k| k * 3 + 7).collect();
        let out = gpu.run(&c, &[Bind::SrvWithWindowAddress(vec![0u8; 16]), Bind::Window(bytes_of(&init))], [1, 1, 1]).expect("run");
        let got = words(&out[0]);
        for gx in 0..64usize {
            assert_eq!(got[68 + gx], init[4 + gx] * 2 + 1, "invocation {gx}");
        }
        assert_eq!(&got[..68], &init[..68], "the rest of the window is untouched");
        println!("device: {}", gpu.name);
    }
    #[cfg(not(windows))]
    let _ = c;
}

/// Atomics on a texel of a storage image and a descriptor array indexed by the invocation.
const IMAGE_ATOMICS: &str = r#"
OpCapability Shader
OpCapability RuntimeDescriptorArray
OpExtension "SPV_EXT_descriptor_indexing"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 8 8 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %img DescriptorSet 0
OpDecorate %img Binding 0
OpDecorate %imgs DescriptorSet 0
OpDecorate %imgs Binding 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%v2uint = OpTypeVector %uint 2
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%tex = OpTypeImage %uint 2D 0 0 0 2 R32ui
%ptex = OpTypePointer UniformConstant %tex
%img = OpVariable %ptex UniformConstant
%texarr = OpTypeRuntimeArray %tex
%ptexarr = OpTypePointer UniformConstant %texarr
%imgs = OpVariable %ptexarr UniformConstant
%pimg = OpTypePointer Image %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%scope = OpConstant %uint 1
%sem = OpConstant %uint 0
%main = OpFunction %void None %fn
%entry = OpLabel
%g = OpLoad %v3uint %gid
%xy = OpVectorShuffle %v2uint %g %g 0 1
%gx = OpCompositeExtract %uint %g 0
%ptr = OpImageTexelPointer %pimg %img %xy %c0
%old = OpAtomicIAdd %uint %ptr %scope %sem %c1
%slot = OpUMod %uint %gx %c1
%arrimg = OpAccessChain %ptex %imgs %slot
%ptr2 = OpImageTexelPointer %pimg %arrimg %xy %c0
%old2 = OpAtomicUMax %uint %ptr2 %scope %sem %old
OpReturn
OpFunctionEnd
"#;

#[test]
fn image_atomics_and_descriptor_arrays_validate() {
    let Some(words) = assemble("image_atomics", IMAGE_ATOMICS) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let imported = decant_dxil::spirv::import(&words).expect("import");
    // The unbounded array is one resource with a range that has no end, in a space of its own.
    let array = imported.shader.resources.iter().find(|r| r.range_size == u32::MAX).expect("the descriptor array");
    assert!(array.space >= 1000);
    common::write_and_validate("spirv_image_atomics", &imported.shader);
}

fn import_and_validate(name: &str, words: &[u32]) -> Vec<u8> {
    let imported = decant_dxil::spirv::import(words).unwrap_or_else(|e| panic!("{name}: import failed: {e}"));
    // The IR survives its text form.
    let text = imported.shader.to_sxir();
    assert_eq!(decant_dxil::ir::Shader::from_sxir(&text).unwrap(), imported.shader);
    common::write_and_validate(name, &imported.shader)
}

#[test]
fn a_loop_through_function_variables_matches_the_reference() {
    let Some(words) = assemble("loop", LOOP) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let c = import_and_validate("spirv_loop", &words);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, bytes_of, open_device, words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let input: Vec<u32> = (0..128 * 4).map(|i: u32| i.wrapping_mul(2_654_435_761) ^ 0x1234).collect();
        let out = gpu.run(&c, &[Bind::Srv(bytes_of(&input)), Bind::Uav(128 * 4)], [2, 1, 1]).expect("run");
        let got = words(&out[0]);
        for (gx, g) in got.iter().enumerate() {
            assert_eq!(*g, loop_reference(&input, gx), "invocation {gx}");
        }
        println!("device: {}", gpu.name);
    }
    #[cfg(not(windows))]
    let _ = c;
}

#[test]
fn a_function_call_a_workgroup_array_a_barrier_and_a_switch_match_the_reference() {
    let Some(words) = assemble("mixed", MIXED) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let c = import_and_validate("spirv_mixed", &words);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let out = gpu.run(&c, &[Bind::Uav(128 * 4)], [2, 1, 1]).expect("run");
        let got = words(&out[0]);
        for gx in 0..128u32 {
            assert_eq!(got[gx as usize], mixed_reference(gx), "invocation {gx}");
        }
        println!("device: {}", gpu.name);
    }
    #[cfg(not(windows))]
    let _ = c;
}
