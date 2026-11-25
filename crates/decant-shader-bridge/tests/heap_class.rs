//! The class of a buffer the DXIL importer places in the descriptor heap. The modules are built with
//! `rspirv`, so these tests need no tool, no DLL and no device (the tests of `decant-dxil` itself only
//! build on Windows).

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel, StorageClass};
use decant_dxil::ir::{Binding, ResourceClass, ResourceKind};
use decant_dxil::spirv::{Bindings, ImportError, Imported, IndexExpr, RegisterClass, Target, import_with};

/// Binding 0 is an array of four storage buffers, as in the cache. Each invocation reads word `gid.x`
/// of element 3 and stores it in element `to`.
fn module(to: u32) -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = b.type_void();
    let uint = b.type_int(32, 0);
    let v3 = b.type_vector(uint, 3);
    let fn_t = b.type_function(void, vec![]);
    let words = b.type_runtime_array(uint);
    b.decorate(words, Decoration::ArrayStride, vec![Operand::LiteralBit32(4)]);
    let ssbo = b.type_struct(vec![words]);
    b.decorate(ssbo, Decoration::Block, vec![]);
    b.member_decorate(ssbo, 0, Decoration::Offset, vec![Operand::LiteralBit32(0)]);
    let four = b.constant_bit32(uint, 4);
    let zero = b.constant_bit32(uint, 0);
    let from = b.constant_bit32(uint, 3);
    let to = b.constant_bit32(uint, to);
    let array = b.type_array(ssbo, four);
    let p_array = b.type_pointer(None, StorageClass::StorageBuffer, array);
    let bufs = b.variable(p_array, None, StorageClass::StorageBuffer, None);
    b.decorate(bufs, Decoration::DescriptorSet, vec![Operand::LiteralBit32(0)]);
    b.decorate(bufs, Decoration::Binding, vec![Operand::LiteralBit32(0)]);
    let p_v3 = b.type_pointer(None, StorageClass::Input, v3);
    let gid = b.variable(p_v3, None, StorageClass::Input, None);
    b.decorate(gid, Decoration::BuiltIn, vec![Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let p_in = b.type_pointer(None, StorageClass::Input, uint);
    let p_word = b.type_pointer(None, StorageClass::StorageBuffer, uint);
    let main = b.begin_function(void, None, FunctionControl::NONE, fn_t).unwrap();
    b.begin_block(None).unwrap();
    let gx = b.access_chain(p_in, None, gid, vec![zero]).unwrap();
    let x = b.load(uint, None, gx, None, vec![]).unwrap();
    let src = b.access_chain(p_word, None, bufs, vec![from, zero, x]).unwrap();
    let v = b.load(uint, None, src, None, vec![]).unwrap();
    let dst = b.access_chain(p_word, None, bufs, vec![to, zero, x]).unwrap();
    b.store(dst, v, None, vec![]).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, main, "main", vec![gid]);
    b.execution_mode(main, ExecutionMode::LocalSize, vec![64, 1, 1]);
    b.module().assemble()
}

fn at(n: u32) -> IndexExpr {
    IndexExpr::User(5).plus(IndexExpr::Const(n))
}

fn import(to: u32, third: Target) -> Result<Imported, ImportError> {
    let mut b = Bindings::new();
    b.bind(0, 0, 0, Target::HeapAt(at(0)));
    b.bind(0, 0, 3, third);
    import_with(&module(to), &b)
}

/// The class and kind of the resources the shader takes from the heap.
fn in_heap(i: &Imported) -> Vec<(ResourceClass, ResourceKind)> {
    i.shader.resources.iter().filter(|r| r.binding == Binding::Heap).map(|r| (r.class, r.kind)).collect()
}

#[test]
fn a_constant_buffer_of_the_heap_is_read_as_one() {
    let imported = import(0, Target::ConstantHeapAt(at(3))).unwrap_or_else(|e| panic!("{e}"));
    let got = in_heap(&imported);
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got.contains(&(ResourceClass::CBuffer, ResourceKind::CBuffer)), "{got:?}");
    assert!(got.contains(&(ResourceClass::Uav, ResourceKind::RawBuffer)), "{got:?}");
    // The module can be written, its handle is a constant buffer of the heap.
    assert!(decant_dxil::write(&imported.shader).unwrap().report.resource_heap);
}

#[test]
fn a_buffer_of_the_heap_without_a_class_stays_raw() {
    let imported = import(0, Target::HeapAt(at(3))).unwrap_or_else(|e| panic!("{e}"));
    let got = in_heap(&imported);
    assert!(got.iter().all(|r| r.1 == ResourceKind::RawBuffer), "{got:?}");
}

#[test]
fn a_heap_constant_buffer_gives_the_code_a_register_gives() {
    let heap = import(0, Target::ConstantHeapAt(at(3))).unwrap();
    let register = import(0, Target::Register { class: RegisterClass::Cbv, space: 1, register: 23 }).unwrap();
    let size = |i: &Imported| i.shader.resources.iter().filter(|r| r.class == ResourceClass::CBuffer).map(|r| r.size_bytes).max();
    assert_eq!(size(&heap), Some(65536));
    assert_eq!(size(&heap), size(&register));
}

#[test]
fn a_write_to_a_heap_constant_buffer_is_refused() {
    let outcome = import(3, Target::ConstantHeapAt(at(3))).map(|i| decant_dxil::write(&i.shader).map(|_| ()));
    assert!(!matches!(outcome, Ok(Ok(()))), "a store needs a writable view");
}
