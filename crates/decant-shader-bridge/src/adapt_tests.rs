//! `adapt` on small modules assembled here with `rspirv`, shaped like the translator's output:
//! buffers, images and samplers in arrays at the base bindings, and the translator's push constants.

use rspirv::binary::{Assemble, Disassemble};
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{AddressingModel, Capability, Decoration, Dim, ExecutionMode, ExecutionModel, FunctionControl, ImageFormat, MemoryModel, Op, StorageClass};

use crate::adapt::{Layout, Target, adapt, id_operands};
use crate::identify::Resource;
use crate::py::Failure;

/// What a test changes in the module.
#[derive(Clone, Copy, Default)]
struct Shape {
    /// SPIR-V 1.5 instead of 1.0, so new variables join the entry point interface.
    v15: bool,
    /// Read the push block at a run time index instead of a constant.
    push_runtime_index: bool,
    /// Pass a push chain to OpCopyObject instead of OpLoad.
    push_copied: bool,
    /// Ask OpArrayLength of the moved buffer.
    array_length: bool,
}

/// A pixel shader: storage buffers `struct { uint[] }[4]` at binding 60, images `image[2]` at
/// binding 61, samplers `sampler[1]` at binding 62, a push block of four uints.
fn module(shape: Shape) -> Vec<u8> {
    let mut b = Builder::new();
    b.set_version(1, if shape.v15 { 5 } else { 0 });
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let float = b.type_float(32, None);
    let k: Vec<u32> = (0..5).map(|v| b.constant_bit32(uint, v)).collect();
    let runtime = b.type_runtime_array(uint);
    let element = b.type_struct([runtime]);
    let buffers_type = b.type_array(element, k[4]);
    let buffers_ptr = b.type_pointer(None, StorageClass::StorageBuffer, buffers_type);
    let buffers = b.variable(buffers_ptr, None, StorageClass::StorageBuffer, None);
    let image = b.type_image(float, Dim::Dim2D, 0, 0, 0, 1, ImageFormat::Unknown, None);
    let images_type = b.type_array(image, k[2]);
    let images_ptr = b.type_pointer(None, StorageClass::UniformConstant, images_type);
    let images = b.variable(images_ptr, None, StorageClass::UniformConstant, None);
    let sampler = b.type_sampler();
    let samplers_type = b.type_array(sampler, k[1]);
    let samplers_ptr = b.type_pointer(None, StorageClass::UniformConstant, samplers_type);
    let samplers = b.variable(samplers_ptr, None, StorageClass::UniformConstant, None);
    let push_type = b.type_struct([uint, uint, uint, uint]);
    let push_ptr = b.type_pointer(None, StorageClass::PushConstant, push_type);
    let push = b.variable(push_ptr, None, StorageClass::PushConstant, None);
    let uint_sb = b.type_pointer(None, StorageClass::StorageBuffer, uint);
    let element_sb = b.type_pointer(None, StorageClass::StorageBuffer, element);
    let uint_pc = b.type_pointer(None, StorageClass::PushConstant, uint);
    let image_uc = b.type_pointer(None, StorageClass::UniformConstant, image);
    let uint_fn = b.type_pointer(None, StorageClass::Function, uint);
    for (variable, binding) in [(buffers, 60), (images, 61), (samplers, 62)] {
        b.decorate(variable, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(variable, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }
    b.decorate(runtime, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    b.decorate(element, Decoration::Block, []);
    b.member_decorate(element, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    b.decorate(push_type, Decoration::Block, []);
    for member in 0..4 {
        b.member_decorate(push_type, member, Decoration::Offset, [Operand::LiteralBit32(4 * member)]);
    }
    let fn_type = b.type_function(void, []);
    let main = b.begin_function(void, None, FunctionControl::NONE, fn_type).unwrap();
    b.begin_block(None).unwrap();
    let local = b.variable(uint_fn, None, StorageClass::Function, None);
    // Buffer at rank 1, member 0, word 2, then the same buffer as a whole.
    let word = b.access_chain(uint_sb, None, buffers, [k[1], k[0], k[2]]).unwrap();
    let value = b.load(uint, None, word, None, []).unwrap();
    let whole = b.access_chain(element_sb, None, buffers, [k[1]]).unwrap();
    let inner = b.access_chain(uint_sb, None, whole, [k[0], k[3]]).unwrap();
    b.store(inner, value, None, []).unwrap();
    if shape.array_length {
        b.array_length(uint, None, whole, 0).unwrap();
    }
    // Buffer at rank 2 stays where it is.
    let other = b.access_chain(uint_sb, None, buffers, [k[2], k[0], k[0]]).unwrap();
    b.load(uint, None, other, None, []).unwrap();
    // Image at rank 0 as a whole, and sampler 0.
    let texture = b.access_chain(image_uc, None, images, [k[0]]).unwrap();
    b.load(image, None, texture, None, []).unwrap();
    // Push words 1 (an offset) and 2 (a size).
    let offset = if shape.push_runtime_index { b.load(uint, None, local, None, []).unwrap() } else { k[1] };
    let first = b.access_chain(uint_pc, None, push, [offset]).unwrap();
    if shape.push_copied {
        b.copy_object(uint_pc, None, first).unwrap();
    } else {
        let read = b.load(uint, None, first, None, []).unwrap();
        b.store(local, read, None, []).unwrap();
    }
    let second = b.access_chain(uint_pc, None, push, [k[2]]).unwrap();
    let read = b.load(uint, None, second, None, []).unwrap();
    b.store(local, read, None, []).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    let interface: Vec<u32> = if shape.v15 { vec![buffers, images, samplers, push] } else { Vec::new() };
    b.entry_point(ExecutionModel::Fragment, main, "main", interface);
    b.execution_mode(main, ExecutionMode::OriginUpperLeft, []);
    b.module().assemble().iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn resource(kind: &str, index: i64, register: Option<&str>, binding: i64, rank: i64) -> Resource {
    Resource {
        kind: kind.into(),
        index,
        register: register.map(str::to_owned),
        origin: if register.is_some() { "table" } else { "autre" },
        heap: None,
        binding: Some(binding),
        rank: Some(rank),
        table: None,
        offset: None,
    }
}

fn resources() -> Vec<Resource> {
    vec![
        resource("buffer", 0, Some("b3"), 0, 1),
        resource("buffer", 1, None, 0, 2),
        resource("image", 0, Some("t5"), 1, 0),
        resource("sampler", 0, Some("s0"), 2, 0),
    ]
}

fn layout(uniform: bool) -> Layout {
    vec![
        ("b3".into(), Target { set: 103, binding: 0, uniform }),
        ("t5".into(), Target { set: 0, binding: 4, uniform: false }),
    ]
}

const PUSH: &[i64] = &[0, 2, 4];

fn disassemble(bytes: &[u8]) -> String {
    rspirv::dr::load_bytes(bytes).unwrap().disassemble()
}

fn refusal(result: Result<(Vec<u8>, Vec<String>), Failure>) -> String {
    match result {
        Err(Failure::Exit(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn report_lists_every_resource() {
    let (_, report) = adapt(&module(Shape::default()), &resources(), Some(PUSH), &layout(true), true).unwrap();
    assert_eq!(
        report,
        [
            "buffer 0 : b3 -> jeu 103, binding 0, tampon uniforme",
            "buffer 1 : laisse en place (autre)",
            "image 0 : t5 -> jeu 0, binding 4",
            "sampler 0 : laisse en place (s0)",
        ]
    );
}

/// The whole rewritten module, checked instruction by instruction against the Python rules:
/// moved buffer and image get their own variable, the converted buffer becomes a uniform block of
/// 16384 words, push chains become OpNop words and their loads constants.
#[test]
fn moved_resources_get_variables_and_push_reads_become_constants() {
    let input = module(Shape::default());
    let (output, _) = adapt(&input, &resources(), Some(PUSH), &layout(true), true).unwrap();
    let text = disassemble(&output);
    assert_eq!(text, EXPECTED, "{text}");
    // Every rewritten instruction keeps its size (OpNop words fill the gaps), so the module grows
    // only by the new globals (35 words) and decorations (28 words).
    assert_eq!(output.len() / 4, input.len() / 4 + 35 + 28);
}

#[test]
fn storage_buffers_stay_storage_when_not_uniform() {
    let (output, _) = adapt(&module(Shape::default()), &resources(), Some(PUSH), &layout(false), true).unwrap();
    let text = disassemble(&output);
    assert!(!text.contains("OpTypeStruct %uint_16384") && !text.contains("16384\n"), "{text}");
    assert!(text.contains("OpDecorate %60 DescriptorSet 103") || text.contains("DescriptorSet 103"), "{text}");
}

#[test]
fn new_variables_join_the_interface_from_spirv_1_4() {
    let shape = Shape { v15: true, ..Shape::default() };
    let (output, _) = adapt(&module(shape), &resources(), Some(PUSH), &layout(true), true).unwrap();
    let m = rspirv::dr::load_bytes(&output).unwrap();
    let interface: Vec<&Operand> = m.entry_points[0].operands.iter().skip(3).collect();
    assert_eq!(interface.len(), 4 + 2);
    let (output, _) = adapt(&module(Shape::default()), &resources(), Some(PUSH), &layout(true), true).unwrap();
    assert_eq!(rspirv::dr::load_bytes(&output).unwrap().entry_points[0].operands.len(), 3);
}

#[test]
fn a_vertex_shader_looks_at_bindings_from_zero() {
    let message = refusal(adapt(&module(Shape::default()), &resources(), Some(PUSH), &layout(true), false));
    assert_eq!(message, "buffer 0 (b3) : pas de place dans le module");
}

#[test]
fn unsupported_forms_are_refused_with_the_python_messages() {
    let all = resources();
    let base = module(Shape::default());
    let runtime = refusal(adapt(&module(Shape { push_runtime_index: true, ..Shape::default() }), &all, Some(PUSH), &layout(true), true));
    assert!(runtime.starts_with("constantes poussees lues a un indice non constant (resultat %") && runtime.ends_with(") : non gere"), "{runtime}");
    let copied = refusal(adapt(&module(Shape { push_copied: true, ..Shape::default() }), &all, Some(PUSH), &layout(true), true));
    assert!(copied.contains(&format!("utilisee par l'operation {}, pas seulement lue par OpLoad : non gere", Op::CopyObject as u32)), "{copied}");
    let length = refusal(adapt(&module(Shape { array_length: true, ..Shape::default() }), &all, Some(PUSH), &layout(true), true));
    assert_eq!(length, "OpArrayLength sur un tampon converti : non gere");
    assert_eq!(refusal(adapt(&base, &all, None, &layout(true), true)), "constante poussee 1 : sens inconnu, non gere");
    assert_eq!(refusal(adapt(&base, &all, Some(&[0, 1, 2]), &layout(true), true)), "constante poussee 2 : sens inconnu, non gere");
    let mut unplaced = all.clone();
    unplaced[0].rank = Some(0xFFFF_FFFF);
    assert_eq!(refusal(adapt(&base, &unplaced, Some(PUSH), &layout(true), true)), "buffer 0 (b3) : pas de place dans le module");
    let mut unbound = all;
    unbound[0].binding = None;
    assert!(matches!(adapt(&base, &unbound, Some(PUSH), &layout(true), true), Err(Failure::Error { .. })));
}

#[test]
fn nothing_to_move_keeps_the_module_apart_from_push_reads() {
    let input = module(Shape::default());
    let (output, report) = adapt(&input, &resources(), Some(PUSH), &Layout::new(), true).unwrap();
    assert_eq!(report.len(), 4);
    let text = disassemble(&output);
    assert_eq!(text.matches("OpNop").count(), 10, "two push chains of five words");
    assert_eq!(output.len(), input.len() + 2 * 16, "two constants added, loads and copies have the same size");
}

/// rspirv's loader writes its own generator number: the translator's must survive the rewrite.
#[test]
fn the_generator_word_is_kept() {
    let mut input = module(Shape::default());
    input[8..12].copy_from_slice(&0x0013_0002u32.to_le_bytes());
    let (output, _) = adapt(&input, &resources(), Some(PUSH), &Layout::new(), true).unwrap();
    assert_eq!(output[8..12], input[8..12]);
}

#[test]
fn modules_rspirv_cannot_read_are_refused() {
    let mut input = module(Shape::default());
    input[0] ^= 0xFF;
    assert!(refusal(adapt(&input, &resources(), Some(PUSH), &layout(true), true)).starts_with("module SPIR-V illisible"));
    assert!(matches!(adapt(&input[..7], &[], None, &Layout::new(), true), Err(Failure::Error { .. })));
}

#[test]
fn only_real_ids_count_as_users() {
    // OpExtInst %type %result %set <instruction> operands: the instruction number is a literal.
    assert_eq!(id_operands(&[7 << 16 | 12, 1, 2, 3, 7, 5, 6]), [1, 2, 3, 5, 6]);
}

/// The disassembly of the rewritten module, checked by hand and against the Python output.
const EXPECTED: &str = r#"; SPIR-V
; Version: 1.0
; Generator: rspirv
; Bound: 55
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %31 "main"
OpExecutionMode %31 OriginUpperLeft
OpDecorate %13 DescriptorSet 0
OpDecorate %13 Binding 60
OpDecorate %17 DescriptorSet 0
OpDecorate %17 Binding 61
OpDecorate %21 DescriptorSet 0
OpDecorate %21 Binding 62
OpDecorate %9 ArrayStride 4
OpDecorate %10 Block
OpMemberDecorate %10 0 Offset 0
OpDecorate %22 Block
OpMemberDecorate %22 0 Offset 0
OpMemberDecorate %22 1 Offset 4
OpMemberDecorate %22 2 Offset 8
OpMemberDecorate %22 3 Offset 12
OpDecorate %46 ArrayStride 4
OpDecorate %47 Block
OpMemberDecorate %47 0 Offset 0
OpDecorate %51 DescriptorSet 103
OpDecorate %51 Binding 0
OpDecorate %52 DescriptorSet 0
OpDecorate %52 Binding 4
%1 = OpTypeVoid
%2 = OpTypeInt 32 0
%3 = OpTypeFloat 32
%4 = OpConstant  %2  0
%5 = OpConstant  %2  1
%6 = OpConstant  %2  2
%7 = OpConstant  %2  3
%8 = OpConstant  %2  4
%9 = OpTypeRuntimeArray %2
%10 = OpTypeStruct %9
%11 = OpTypeArray %10 %8
%12 = OpTypePointer StorageBuffer %11
%13 = OpVariable  %12  StorageBuffer
%14 = OpTypeImage %3 2D 0 0 0 1 Unknown
%15 = OpTypeArray %14 %6
%16 = OpTypePointer UniformConstant %15
%17 = OpVariable  %16  UniformConstant
%18 = OpTypeSampler
%19 = OpTypeArray %18 %5
%20 = OpTypePointer UniformConstant %19
%21 = OpVariable  %20  UniformConstant
%22 = OpTypeStruct %2 %2 %2 %2
%23 = OpTypePointer PushConstant %22
%24 = OpVariable  %23  PushConstant
%25 = OpTypePointer StorageBuffer %2
%26 = OpTypePointer StorageBuffer %10
%27 = OpTypePointer PushConstant %2
%28 = OpTypePointer UniformConstant %14
%29 = OpTypePointer Function %2
%30 = OpTypeFunction %1
%48 = OpConstant  %2  16384
%46 = OpTypeArray %2 %48
%47 = OpTypeStruct %46
%49 = OpTypePointer Uniform %47
%50 = OpTypePointer Uniform %2
%51 = OpVariable  %49  Uniform
%52 = OpVariable  %28  UniformConstant
%53 = OpConstant  %2  0
%54 = OpConstant  %2  65536
%31 = OpFunction  %1  None %30
%32 = OpLabel
%33 = OpVariable  %29  Function
%34 = OpAccessChain  %50  %51 %4 %6
OpNop
%35 = OpLoad  %2  %34
%36 = OpCopyObject  %49  %51
OpNop
%37 = OpAccessChain  %50  %36 %4 %7
OpStore %37 %35
%38 = OpAccessChain  %25  %13 %6 %4 %4
%39 = OpLoad  %2  %38
%40 = OpCopyObject  %28  %52
OpNop
%41 = OpLoad  %14  %40
OpNop
OpNop
OpNop
OpNop
OpNop
%43 = OpCopyObject  %2  %53
OpStore %33 %43
OpNop
OpNop
OpNop
OpNop
OpNop
%45 = OpCopyObject  %2  %54
OpStore %33 %45
OpReturn
OpFunctionEnd"#;
