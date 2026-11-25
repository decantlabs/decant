//! Heap indices chosen at run time: read from the user data mirror (a root constant buffer, `b1` of
//! space 15, the console user SGPRs), or picked by a run time index among the elements of an array.
//! Hand written modules, assembled with `spirv-as`, validated with dxil.dll and run on a device.

mod common;

use std::path::PathBuf;

use decant_dxil::spirv::{Bindings, IndexExpr, RegisterClass, Target, USER_DATA_REGISTER, USER_DATA_SPACE, import_with};

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
    let dir = std::env::temp_dir().join(format!("decant-dxil-spirv-mirror-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join(format!("{name}.spvasm")), dir.join(format!("{name}.spv")));
    std::fs::write(&src, text).ok()?;
    let status = std::process::Command::new(tool).arg("--target-env").arg("vulkan1.1").arg(&src).arg("-o").arg(&dst).status().ok()?;
    assert!(status.success(), "spirv-as rejected {name}");
    let bytes = std::fs::read(&dst).ok()?;
    let _ = std::fs::remove_dir_all(&dir);
    Some(bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Invocation x writes `x * 3 + 1` to word x of one storage buffer (set 0 binding 0).
const OUT: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid
OpExecutionMode %main LocalSize 4 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %arr ArrayStride 4
OpMemberDecorate %SSBO 0 Offset 0
OpDecorate %SSBO Block
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%pssbo = OpTypePointer StorageBuffer %SSBO
%out = OpVariable %pssbo StorageBuffer
%pu = OpTypePointer StorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c3 = OpConstant %uint 3
%main = OpFunction %void None %fn
%entry = OpLabel
%gp = OpAccessChain %pu_in %gid %c0
%x = OpLoad %uint %gp
%o = OpAccessChain %pu %out %c0 %x
%t = OpIMul %uint %x %c3
%v = OpIAdd %uint %t %c1
OpStore %o %v
OpReturn
OpFunctionEnd
"#;

/// Invocation x writes `x + 1` to word x of buffer `x % 3` of an array of three (set 0 binding 0).
const DYNAMIC: &str = r#"
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
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%pv3 = OpTypePointer Input %v3uint
%gid = OpVariable %pv3 Input
%pu_in = OpTypePointer Input %uint
%arr = OpTypeRuntimeArray %uint
%SSBO = OpTypeStruct %arr
%c3 = OpConstant %uint 3
%bufs_t = OpTypeArray %SSBO %c3
%pbufs = OpTypePointer StorageBuffer %bufs_t
%bufs = OpVariable %pbufs StorageBuffer
%pu = OpTypePointer StorageBuffer %uint
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%gp = OpAccessChain %pu_in %gid %c0
%x = OpLoad %uint %gp
%e = OpUMod %uint %x %c3
%o = OpAccessChain %pu %bufs %e %c0 %x
%v = OpIAdd %uint %x %c1
OpStore %o %v
OpReturn
OpFunctionEnd
"#;

fn mirror_bytes(words: &[(usize, u32)]) -> Vec<u8> {
    let mut m = [0u32; 32];
    for (k, v) in words {
        m[*k] = *v;
    }
    m.iter().flat_map(|w| w.to_le_bytes()).collect()
}

#[test]
fn a_heap_index_is_a_table_from_the_mirror_plus_an_offset() {
    let Some(words) = assemble("out", OUT) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    // The console shader reads descriptor 2 of the table that user register s5 holds.
    b.bind(0, 0, 0, Target::HeapAt(IndexExpr::User(5).plus(IndexExpr::Const(2))));
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    let ud = imported.user_data.as_ref().expect("the shader reads the mirror");
    assert_eq!(ud.dwords, vec![5]);
    // The mirror is the constant buffer b1 of space 15 and the buffer itself is in the heap.
    let mirror = imported.shader.resources.iter().find(|r| r.space == USER_DATA_SPACE && r.lower_bound == USER_DATA_REGISTER).expect("mirror");
    assert_eq!(mirror.size_bytes, 128);
    let c = common::write_and_validate("spirv_mirror_out", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words as to_words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let binds: Vec<Bind> = (0..4).map(|slot| Bind::HeapUav { slot, size: 16 }).chain([Bind::Mirror(mirror_bytes(&[(5, 1)]))]).collect();
        let out = gpu.run(&c, &binds, [1, 1, 1]).expect("run");
        for (slot, bytes) in out.iter().enumerate() {
            let got = to_words(bytes);
            // Mirror word 5 is 1, so the table starts at slot 1 and descriptor 2 is slot 3.
            let want: Vec<u32> = if slot == 3 { vec![1, 4, 7, 10] } else { vec![0; 4] };
            assert_eq!(got, want, "slot {slot}");
        }
        println!("device: {}", gpu.name);
    }
    #[cfg(not(windows))]
    let _ = c;
}

#[test]
fn an_array_indexed_at_run_time_picks_among_heap_indices() {
    let Some(words) = assemble("dynamic", DYNAMIC) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::Heap(0));
    b.bind(0, 0, 1, Target::HeapAt(IndexExpr::User(2)));
    b.bind(0, 0, 2, Target::Heap(2));
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    let c = common::write_and_validate("spirv_mirror_dynamic", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words as to_words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        let binds: Vec<Bind> = (0..3).map(|slot| Bind::HeapUav { slot, size: 16 }).chain([Bind::Mirror(mirror_bytes(&[(2, 1)]))]).collect();
        let out = gpu.run(&c, &binds, [1, 1, 1]).expect("run");
        let want = [vec![1, 0, 0, 4], vec![0, 2, 0, 0], vec![0, 0, 3, 0]];
        for (slot, bytes) in out.iter().enumerate() {
            assert_eq!(to_words(bytes), want[slot], "slot {slot}");
        }
    }
    #[cfg(not(windows))]
    let _ = c;
}

#[test]
fn registers_cannot_be_picked_by_an_index_unless_a_heap_placement_is_given() {
    let Some(words) = assemble("dynamic_regs", DYNAMIC) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    for k in 0..3 {
        b.bind(0, 0, k, Target::Register { class: RegisterClass::Uav, space: 0, register: 10 + k });
    }
    assert_eq!(import_with(&words, &b).unwrap_err().category, "binding:dynamic-index");
    // With a heap placement for the run time case the same module imports.
    for k in 0..3 {
        b.bind_dynamic(0, 0, k, Target::HeapAt(IndexExpr::User(4).plus(IndexExpr::Const(k))));
    }
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    assert_eq!(imported.user_data.as_ref().map(|u| u.dwords.clone()), Some(vec![4]));
    common::write_and_validate("spirv_mirror_dynamic_regs", &imported.shader);
    // Constant indices still use the register.
    assert_eq!(b.target(0, 0, 1), Some(Target::Register { class: RegisterClass::Uav, space: 0, register: 11 }));
}

#[test]
fn a_root_descriptor_is_reached_through_the_window_at_the_address_in_the_mirror() {
    let Some(words) = assemble("root", OUT) else {
        println!("SKIPPED: spirv-as not found (set SPIRV_AS)");
        return;
    };
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::Address(2));
    let imported = import_with(&words, &b).unwrap_or_else(|e| panic!("import failed: {e}"));
    assert_eq!(imported.user_data.as_ref().map(|u| u.dwords.clone()), Some(vec![2, 3]));
    assert!(imported.device_memory.is_some());
    let c = common::write_and_validate("spirv_mirror_root", &imported.shader);
    #[cfg(windows)]
    {
        use common::gpu::{Bind, open_device, words as to_words};
        let Some(gpu) = open_device() else {
            println!("SKIPPED: no D3D12 device with Shader Model 6.6");
            return;
        };
        // The buffer is 32 bytes into the window, the mirror holds its address in dwords 2 and 3.
        let binds = [Bind::MirrorWithWindowAddress { bytes: mirror_bytes(&[]), at: 2, offset: 32 }, Bind::Window(vec![0xAA; 64])];
        let out = gpu.run(&c, &binds, [1, 1, 1]).expect("run");
        let got = to_words(&out[0]);
        assert_eq!(&got[8..12], &[1, 4, 7, 10]);
        assert!(got[..8].iter().all(|w| *w == 0xAAAA_AAAA) && got[12..].iter().all(|w| *w == 0xAAAA_AAAA), "{got:08x?}");
    }
    #[cfg(not(windows))]
    let _ = c;
}
