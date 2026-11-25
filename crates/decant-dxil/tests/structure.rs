//! Tests that need no DLL and no device: the structure of the output, the IR text form, and the errors.

mod common;

use common::shaders;
use decant_dxil::bitstream::reader::{Item, read};
use decant_dxil::container::{self, rd32};
use decant_dxil::ir::*;
use decant_dxil::{Error, write};

fn bitcode_of(container: &[u8]) -> Vec<u8> {
    let parts = container::parts(container).expect("a well formed container");
    let (_, dxil) = parts.iter().find(|(cc, _)| cc == b"DXIL").expect("a DXIL part");
    assert_eq!(&dxil[8..12], b"DXIL");
    let (off, size) = (rd32(dxil, 16) as usize, rd32(dxil, 20) as usize);
    assert_eq!(off, 16);
    assert_eq!(rd32(dxil, 4) as usize * 4, dxil.len(), "the part size in words");
    dxil[8 + off..8 + off + size].to_vec()
}

fn block_ids(items: &[Item]) -> Vec<u64> {
    items.iter().filter_map(|i| if let Item::Block { id, .. } = i { Some(*id) } else { None }).collect()
}

#[test]
fn the_empty_shader_has_the_parts_of_the_specification() {
    let c = write(&shaders::empty()).unwrap().container;
    let parts = container::parts(&c).unwrap();
    let names: Vec<_> = parts.iter().map(|(cc, _)| String::from_utf8_lossy(cc).into_owned()).collect();
    assert_eq!(names, ["SFI0", "ISG1", "OSG1", "PSV0", "HASH", "DXIL"]);
    let size = |n: &str| parts.iter().find(|(cc, _)| cc == n.as_bytes()).unwrap().1.len();
    assert_eq!((size("SFI0"), size("ISG1"), size("OSG1"), size("PSV0"), size("HASH")), (8, 8, 8, 68, 20));
    assert_eq!(rd32(&c, 24) as usize, c.len());
    let sum = container::digest(&c);
    assert!((0..4).all(|i| rd32(&c, 4 + 4 * i) == sum[i]));
    let dxil = parts[5].1;
    assert_eq!(rd32(dxil, 0), 0x0005_0066, "compute shader, model 6.6");
    assert_eq!(rd32(dxil, 12), 0x106, "DXIL 1.6");
}

#[test]
fn the_bitcode_decodes_into_the_blocks_of_a_module() {
    let c = write(&shaders::buffer_copy()).unwrap().container;
    let items = read(&bitcode_of(&c)).expect("our reader accepts what the writer wrote");
    assert_eq!(items.len(), 1);
    let Item::Block { id: 8, items: module, .. } = &items[0] else { panic!("a module block is expected") };
    // attribute groups, attributes, types, constants, metadata, kinds, symbol table, the function.
    assert_eq!(block_ids(module), [10, 9, 17, 11, 15, 15, 14, 12]);
    assert!(matches!(&module[0], Item::Record { code: 1, ops } if ops == &[1]), "version 1 comes first");
    let Some(Item::Block { items: body, .. }) = module.iter().find(|i| matches!(i, Item::Block { id: 12, .. })) else { panic!() };
    assert!(matches!(&body[0], Item::Record { code: 1, ops } if ops == &[1]), "DECLAREBLOCKS 1");
    assert!(matches!(body.last(), Some(Item::Record { code: 10, .. })), "the function ends with ret");
}

#[test]
fn the_loop_shader_lays_blocks_out_so_that_only_phis_refer_forward() {
    let c = write(&shaders::loop_with_branch()).unwrap().container;
    let items = read(&bitcode_of(&c)).unwrap();
    let Item::Block { items: module, .. } = &items[0] else { panic!() };
    let Some(Item::Block { items: body, .. }) = module.iter().find(|i| matches!(i, Item::Block { id: 12, .. })) else { panic!() };
    let Item::Record { ops, .. } = &body[0] else { panic!() };
    // A preheader, a dedicated loop exit and one split critical edge were added to the six blocks of the source.
    assert_eq!(ops[0], 9);
    let phis = body.iter().filter(|i| matches!(i, Item::Record { code: 16, .. })).count();
    assert_eq!(phis, 4, "i, sum, the merged sum and the exit sum");
}

#[test]
fn the_ir_survives_its_text_form() {
    for s in [shaders::empty(), shaders::arithmetic(), shaders::loop_with_branch(), shaders::atomics(), shaders::resource_zoo()] {
        let text = s.to_sxir();
        let back = Shader::from_sxir(&text).expect("parses");
        assert_eq!(back, s);
        assert_eq!(write(&back).unwrap().container, write(&s).unwrap().container);
    }
}

#[test]
fn output_is_deterministic() {
    assert_eq!(write(&shaders::arithmetic()).unwrap().container, write(&shaders::arithmetic()).unwrap().container);
}

#[test]
fn the_binding_report_lists_the_bound_resources() {
    let r = write(&shaders::buffer_copy()).unwrap().report;
    assert_eq!(r.bound.len(), 2);
    assert_eq!((r.bound[0].class, r.bound[0].register), (ResourceClass::Srv, 0));
    assert_eq!((r.bound[1].class, r.bound[1].kind), (ResourceClass::Uav, ResourceKind::RawBuffer));
    assert!(!r.resource_heap && !r.sampler_heap && r.wave_size.is_none());
}

#[test]
fn a_use_before_definition_is_named() {
    let insts = vec![Inst::new(&[1], Op::Bin(BinOp::Add, Operand::value(0), Operand::i32(1)))];
    let s = Shader::compute([1, 1, 1], vec![Block::new(insts, Terminator::Ret)]);
    assert!(matches!(write(&s), Err(Error::UseBeforeDef(m)) if m.contains("value 0")));
}

#[test]
fn irreducible_flow_is_refused() {
    let c = Operand::Const(Const::Bool(true));
    let t = |t: Terminator| Block::new(vec![], t);
    let s = Shader::compute(
        [1, 1, 1],
        vec![
            t(Terminator::CondBr(c, BlockId(1), BlockId(2))),
            t(Terminator::CondBr(c, BlockId(2), BlockId(3))),
            t(Terminator::CondBr(c, BlockId(1), BlockId(3))),
            t(Terminator::Ret),
        ],
    );
    assert!(matches!(write(&s), Err(Error::Irreducible(_))));
}

#[test]
fn type_mismatches_and_bad_shapes_are_errors() {
    let one = |op: Op, dsts: &[u32]| Shader::compute([1, 1, 1], vec![Block::new(vec![Inst::new(dsts, op)], Terminator::Ret)]);
    let bad = [
        one(Op::Bin(BinOp::Add, Operand::i32(1), Operand::f32(1.0)), &[0]),
        one(Op::Bin(BinOp::FAdd, Operand::i32(1), Operand::i32(1)), &[0]),
        one(Op::Select(Operand::i32(1), Operand::i32(1), Operand::i32(2)), &[0]),
        one(Op::Bin(BinOp::Add, Operand::i32(1), Operand::i32(1)), &[]),
        one(Op::ThreadId(ThreadComponentKind::Dispatch, 3), &[0]),
        one(Op::Barrier { flags: 0 }, &[]),
        one(Op::BufLoad { res: ResRef::bound(0), index: Operand::i32(0), offset: None, count: 1, ty: Ty::I32 }, &[0]),
    ];
    for (i, s) in bad.iter().enumerate() {
        assert!(matches!(write(s), Err(Error::Invalid(_))), "case {i}: {:?}", write(s).err());
    }
    let mut big = shaders::empty();
    big.stage = Stage::Compute { threads: [1024, 2, 1], wave_size: None };
    assert!(matches!(write(&big), Err(Error::Invalid(_))));
    let mut f16 = shaders::empty();
    f16.blocks[0].insts.push(Inst::new(&[0], Op::Cast(CastOp::FpTrunc, Operand::f32(1.0), Ty::F16)));
    assert!(write(&f16).is_err());
}

#[test]
fn a_phi_needs_one_input_per_predecessor() {
    let mut s = shaders::loop_with_branch();
    s.blocks[3].phis[0].incoming.pop();
    assert!(matches!(write(&s), Err(Error::Invalid(m)) if m.contains("phi 8")));
}

#[test]
fn a_loop_whose_header_is_the_entry_gets_a_new_entry_block() {
    // 0 loops on itself: the entry has a predecessor, which DXIL does not allow.
    let c = Operand::Const(Const::Bool(true));
    let s = Shader::compute([1, 1, 1], vec![Block::new(vec![], Terminator::CondBr(c, BlockId(0), BlockId(1))), Block::new(vec![], Terminator::Ret)]);
    let c = write(&s).unwrap().container;
    let items = read(&bitcode_of(&c)).unwrap();
    let Item::Block { items: module, .. } = &items[0] else { panic!() };
    let Some(Item::Block { items: body, .. }) = module.iter().find(|i| matches!(i, Item::Block { id: 12, .. })) else { panic!() };
    let Item::Record { ops, .. } = &body[0] else { panic!() };
    assert_eq!(ops[0], 3, "a new entry, the old entry and the exit");
}

/// The golden container of the empty shader: any change of the output shows here. Regenerate with
/// `DECANT_SHADER_PRINT_GOLDEN=1 cargo test -p decant-dxil --test structure golden -- --nocapture`
/// after the new output has been through the validator tests.


#[test]
fn golden_bytes_of_the_empty_shader() {
    let c = write(&shaders::empty()).unwrap().container;
    let hex: String = c.iter().map(|b| format!("{b:02x}")).collect();
    if std::env::var_os("DECANT_SHADER_PRINT_GOLDEN").is_some() {
        println!("{hex}");
    }
    assert_eq!(hex, common::golden::empty(), "the output of the empty shader changed");
}
