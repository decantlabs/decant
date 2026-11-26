//! Resource tracing and placement: where the descriptors of synthetic programs come from and
//! where they are bound. The programs are written here with the assembler, shaped like the
//! cases measured on the title's shaders. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::{Ext, Insn, Operand, Reg};
use decant_shader::ir::{Bin, Expr, Function, Origin, Stmt, Var};
use decant_shader::tables::Op;
use decant_shader::{Refusal, lift_program, resource};

/// The resource pass alone: lift, trace, remove dead register traffic, place.
fn placed_with(
    p: &StageProgram,
    root: &RootSignature,
    options: &Options,
) -> Result<(Function, Needs), Refusal> {
    let mut func = lift_program(p)?;
    resource::trace(&mut func, p)?;
    resource::dce(&mut func);
    let needs = resource::place(&mut func, p, root, options, &CountingHost::default())?;
    Ok((func, needs))
}

fn placed(p: &StageProgram) -> Result<(Function, Needs), Refusal> {
    placed_with(p, &RootSignature::default(), &Options::default())
}

fn placements(func: &Function) -> Vec<Placement> {
    func.resources.iter().filter_map(|r| r.placement).collect()
}

fn program(stage: Stage, a: Asm) -> StageProgram {
    StageProgram {
        name: "test".to_owned(),
        ..StageProgram::new(stage, a.finish())
    }
}

fn record(class: RegisterClass, register: u32, sgpr: u8, word_offset: u32) -> TableBinding {
    TableBinding {
        class,
        register,
        sgpr,
        word_offset,
    }
}

/// A sampler from four constants in `s36` to `s39`.
fn constant_sampler(a: &mut Asm) -> [u32; 4] {
    a.push(sopk(Op::s_movk_i32, s(36), 0x92));
    a.push(sop1(Op::s_mov_b32, s(37), lit(0x00c0_0000)));
    a.push(sop1(Op::s_mov_b32, s(38), lit(0x1450_0000)));
    a.push(sop1(Op::s_mov_b32, s(39), int(0)));
    [0x92, 0x00c0_0000, 0x1450_0000, 0]
}

/// The shape of the video pixel program of the title: the table pointer is set up in the first
/// block, and one texture is only loaded and sampled behind a branch.
#[test]
fn a_table_read_behind_a_branch_is_still_a_table_read() {
    let mut a = Asm::new();
    let skip = a.label();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0xc0));
    let words = constant_sampler(&mut a);
    a.push(smem(Op::s_buffer_load_dword, s(0), s(4), 0x70));
    a.push(sopc(Op::s_cmp_eq_u32, s(0), int(0)));
    a.branch(Op::s_cbranch_scc1, skip);
    a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x80));
    a.push(image(Op::image_sample_lz, v(2), v(2), s(4), s(36), 2, 0x7));
    a.bind(skip);
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = program(Stage::Pixel, a);
    p.binding_table = vec![
        record(RegisterClass::Srv, 10, 8, 32),
        record(RegisterClass::Cbv, 3, 8, 48),
    ];
    let (func, needs) = placed(&p).unwrap();
    assert_eq!(
        placements(&func),
        [
            Placement::HeapAtMirror {
                class: RegisterClass::Cbv,
                dword: 8,
                slot: 6
            },
            Placement::HeapAtMirror {
                class: RegisterClass::Srv,
                dword: 8,
                slot: 4
            },
            Placement::HeapSampler { words, index: 0 },
        ]
    );
    assert_eq!(needs.mirror_dwords, [8]);
    assert!(needs.resource_heap && needs.sampler_heap);
}

/// A sampler built before a loop and used inside it: the constants hold around the back edge.
#[test]
fn constants_set_before_a_loop_are_known_inside_it() {
    let mut a = Asm::new();
    let top = a.label();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(12), s(8), 0x20));
    let words = constant_sampler(&mut a);
    a.push(sop1(Op::s_mov_b32, s(24), int(0)));
    a.bind(top);
    a.push(image(Op::image_sample_lz, v(2), v(2), s(12), s(36), 1, 0x7));
    a.push(sop2(Op::s_add_u32, s(24), s(24), int(1)));
    a.push(sopc(Op::s_cmp_eq_u32, s(24), int(13)));
    a.branch(Op::s_cbranch_scc0, top);
    a.push(sopp(Op::s_endpgm, 0));
    let (func, _) = placed(&program(Stage::Pixel, a)).unwrap();
    assert_eq!(
        placements(&func),
        [
            Placement::HeapAtMirror {
                class: RegisterClass::Srv,
                dword: 8,
                slot: 1
            },
            Placement::HeapSampler { words, index: 0 },
        ]
    );
}

/// A register that holds a descriptor on one path only is not a descriptor after the join.
#[test]
fn a_descriptor_loaded_on_one_path_only_is_refused_at_its_use() {
    let mut a = Asm::new();
    let join = a.label();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(sopc(Op::s_cmp_eq_u32, s(10), int(0)));
    a.branch(Op::s_cbranch_scc1, join);
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0x20));
    a.bind(join);
    a.push(smem(Op::s_buffer_load_dword, s(0), s(4), 0));
    a.push(sopp(Op::s_endpgm, 0));
    let refusal = placed(&program(Stage::Pixel, a)).unwrap_err();
    assert_eq!(refusal.key(), "resource:untraced-descriptor");
}

/// The buffer descriptor the title's vertex programs build around a root descriptor: the
/// address halves from two user data registers, the stride or'ed into the high half, then two
/// constants.
fn root_view_vertex() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(3), lit(0x1101_4fac)));
    a.push(sop1(Op::s_mov_b32, s(2), int(-1)));
    a.push(sop1(Op::s_mov_b32, s(0), s(8)));
    a.push(sop2(Op::s_or_b32, s(1), s(9), lit(0x10000)));
    a.push(smem(Op::s_buffer_load_dword, s(4), s(0), 0));
    a.push(vop1(Op::v_mov_b32, v(1), s(4)));
    a.push(buffer_load(
        Op::buffer_load_dwordx3,
        v(1),
        v(1),
        s(0),
        0,
        false,
    ));
    a.push(sopp(Op::s_endpgm, 0));
    program(Stage::Vertex, a)
}

#[test]
fn a_buffer_descriptor_built_around_a_root_descriptor_is_that_root_view() {
    let p = root_view_vertex();
    let root = RootSignature {
        params: vec![RootParam {
            kind: RootParamKind::Descriptor {
                class: RegisterClass::Srv,
                register: 0,
                space: 3,
            },
            visibility: 0,
        }],
        ..RootSignature::default()
    };
    let view = Placement::Register {
        class: RegisterClass::Srv,
        register: 0,
        space: 3,
    };
    let (func, needs) = placed_with(&p, &root, &Options::default()).unwrap();
    assert_eq!(func.resources.len(), 1);
    assert_eq!(func.resources[0].origin, Origin::RootDescriptor { sgpr: 8 });
    assert_eq!(placements(&func), [view]);
    // The address is not read as a number: the root view is bound by the layer.
    assert!(needs.mirror_dwords.is_empty());

    // Without a root signature the root layout of the container names the view.
    assert_eq!(
        placed(&p).unwrap_err().key(),
        "resource:root-descriptor",
        "no root signature and no layout"
    );
    let mut with_layout = p.clone();
    with_layout.user_data = vec![UserDataSlot {
        sgpr: 8,
        kind: UserDataKind::RootSrv { register: 0 },
    }];
    with_layout.resources = vec![DeclaredResource {
        class: RegisterClass::Srv,
        lower: 0,
        upper: 0,
        space: 3,
        psv_type: 4,
        psv_kind: 11,
    }];
    assert_eq!(placements(&placed(&with_layout).unwrap().0), [view]);
}

/// Mesh programs find their user data eight registers later, while the container's tables and
/// the mirror count from 8 as for every graphics stage.
#[test]
fn mesh_registers_map_to_mirror_dwords() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(10), s(17)));
    a.push(sop1(Op::s_mov_b32, s(11), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(24), s(10), 0x40));
    a.push(smem(Op::s_buffer_load_dword, s(4), s(24), 0));
    // A root constant read as a number.
    a.push(sop2(Op::s_and_b32, s(6), s(20), lit(0x3ff)));
    a.push(vop2(Op::v_add_u32, v(1), s(4), v(1)));
    a.push(vop2(Op::v_add_u32, v(1), s(6), v(1)));
    a.push(export(12, 0xf, [1, 1, 1, 1], false, true, false));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = program(Stage::Mesh, a);
    p.binding_table = vec![record(RegisterClass::Cbv, 0, 9, 16)];
    let (func, needs) = placed(&p).unwrap();
    assert_eq!(
        func.resources[0].origin,
        Origin::Table {
            sgpr: 17,
            offset: 0x40
        }
    );
    assert_eq!(
        placements(&func),
        [Placement::HeapAtMirror {
            class: RegisterClass::Cbv,
            dword: 9,
            slot: 2
        }]
    );
    assert_eq!(needs.mirror_dwords, [9, 12]);
}

#[test]
fn descriptor_words_masked_safely_in_and_are_accepted() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0));
    a.push(sop2(Op::s_and_b32, s(0), s(4), int(0)));
    a.push(sop2(Op::s_and_b32, s(1), s(5), lit(0x3fff_0000)));
    a.push(sopp(Op::s_endpgm, 0));
    placed(&program(Stage::Compute, a)).unwrap();

    let mut b = Asm::new();
    b.push(sop1(Op::s_mov_b32, s(9), int(0)));
    b.push(smem(Op::s_load_dwordx4, s(4), s(8), 0));
    b.push(sop2(Op::s_and_b32, s(0), s(4), lit(0xff)));
    b.push(sopp(Op::s_endpgm, 0));
    let refusal = placed(&program(Stage::Compute, b)).unwrap_err();
    assert_eq!(refusal.key(), "resource:descriptor-as-data");
}

#[test]
fn descriptor_words_used_as_numbers_are_refused() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0));
    a.push(sop2(Op::s_lshr_b32, s(0), s(7), int(4)));
    a.push(sopp(Op::s_endpgm, 0));
    let refusal = placed(&program(Stage::Pixel, a)).unwrap_err();
    assert_eq!(refusal.key(), "resource:descriptor-as-data");
}

#[test]
fn a_load_at_an_address_the_program_computed_is_refused() {
    let mut a = Asm::new();
    a.push(sop2(Op::s_add_u32, s(2), s(8), lit(0x100)));
    a.push(sop1(Op::s_mov_b32, s(3), int(0)));
    a.push(smem(Op::s_load_dword, s(4), s(2), 0));
    a.push(sopp(Op::s_endpgm, 0));
    let refusal = placed(&program(Stage::Pixel, a)).unwrap_err();
    assert_eq!(refusal.key(), "resource:raw-address");
    assert_eq!(refusal.pc, Some(9));
}

/// A scalar load with a register offset.
fn smem_at(op: Op, dst: Operand, base: Operand, offset: u32, soffset: Operand) -> Insn {
    let mut insn = smem(op, dst, base, offset);
    insn.ext = Ext::Smem {
        offset,
        soffset,
        glc: false,
        dlc: false,
    };
    insn
}

/// The root layout of a typical pixel program of the title: two tables, two root constants,
/// then the bases of the two heaps.
fn pixel_layout() -> Vec<UserDataSlot> {
    let slot = |sgpr, kind| UserDataSlot { sgpr, kind };
    vec![
        slot(8, UserDataKind::Table),
        slot(9, UserDataKind::Table),
        slot(
            13,
            UserDataKind::Constants {
                register: 500,
                count: 2,
            },
        ),
        slot(15, UserDataKind::ResourceHeap),
        slot(16, UserDataKind::SamplerHeap),
    ]
}

/// Every `Set` value of the function that satisfies `want`.
fn values(func: &Function, want: impl Fn(&Expr) -> bool) -> Vec<Expr> {
    let mut out = Vec::new();
    func.for_each_stmt(&mut |s| {
        if let Stmt::Set { value, .. } = s
            && want(value)
        {
            out.push(value.clone());
        }
    });
    out
}

/// The bindless shape: a texture taken from the whole view heap by an index out of a root
/// constant, and a sampler from the sampler heap at a constant slot.
fn bindless_pixel() -> StageProgram {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(4), s(15)));
    a.push(sop1(Op::s_mov_b32, s(5), int(0)));
    a.push(sop1(Op::s_mov_b32, s(2), s(16)));
    a.push(sop1(Op::s_mov_b32, s(3), int(0)));
    a.push(sop2(Op::s_and_b32, s(6), s(13), lit(0x3ff)));
    a.push(sop2(Op::s_lshl_b32, s(6), s(6), int(5)));
    a.push(smem_at(Op::s_load_dwordx8, s(20), s(4), 0, s(6)));
    a.push(smem(Op::s_load_dwordx4, s(36), s(2), 0x30));
    a.push(image(Op::image_sample_lz, v(2), v(2), s(20), s(36), 1, 0x7));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = program(Stage::Pixel, a);
    p.user_data = pixel_layout();
    p
}

#[test]
fn a_descriptor_taken_by_a_console_heap_index_goes_through_the_heap_map() {
    let p = bindless_pixel();
    let (func, needs) = placed(&p).unwrap();
    assert_eq!(
        func.resources[0].origin,
        Origin::HeapIndex {
            index: Var::S(20),
            sampler: false
        }
    );
    assert_eq!(
        placements(&func),
        [
            Placement::HeapAtConsoleIndex {
                class: RegisterClass::Srv
            },
            // The sampler heap keeps its indices: the slot is the byte offset over 16.
            Placement::HeapAtMirror {
                class: RegisterClass::Sampler,
                dword: 16,
                slot: 3
            },
        ]
    );
    // The load of the first descriptor word leaves the host index behind: the console index
    // (the byte offset over 32) through the heap map.
    let tokens = values(
        &func,
        |e| matches!(e, Expr::Call(h, _) if h.name == "decant_heap_host"),
    );
    assert_eq!(tokens.len(), 1, "{tokens:?}");
    let Expr::Call(_, args) = &tokens[0] else {
        unreachable!()
    };
    assert!(
        matches!(&args[0], Expr::Bin(Bin::LShr, _, shift) if **shift == Expr::U32(5)),
        "{args:?}"
    );
    // The root constant and the sampler heap base are read from the mirror, the heap map is
    // found through the helper by `lib.rs`.
    assert_eq!(needs.mirror_dwords, [13, 16]);
    assert!(needs.resource_heap && needs.sampler_heap);

    // Without a heap map in the mirror ABI the program is refused by name.
    let no_map = Options {
        mirror: Some(MirrorAbi {
            heap_map: None,
            ..MirrorAbi::default()
        }),
        ..Options::default()
    };
    let refusal = placed_with(&p, &RootSignature::default(), &no_map).unwrap_err();
    assert_eq!(refusal.key(), "resource:no-heap-map");

    // Without the root layout nothing says that s15 is the heap.
    let mut bare = p.clone();
    bare.user_data.clear();
    assert_eq!(
        placed(&bare).unwrap_err().key(),
        "resource:dynamic-table-offset"
    );
}

/// Two table slots, one per path, in the same registers: the use takes whichever was loaded.
#[test]
fn a_descriptor_chosen_by_a_branch_is_taken_by_the_index_its_load_left() {
    let mut a = Asm::new();
    let (other, join) = (a.label(), a.label());
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    let words = constant_sampler(&mut a);
    a.push(sopc(Op::s_cmp_eq_u32, s(10), int(0)));
    a.branch(Op::s_cbranch_scc1, other);
    a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x20));
    a.branch(Op::s_branch, join);
    a.bind(other);
    a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x40));
    a.bind(join);
    a.push(image(Op::image_sample_lz, v(2), v(2), s(4), s(36), 1, 0x7));
    a.push(sopp(Op::s_endpgm, 0));
    let (func, needs) = placed(&program(Stage::Pixel, a)).unwrap();
    assert_eq!(
        placements(&func),
        [
            Placement::HeapAtConsoleIndex {
                class: RegisterClass::Srv
            },
            Placement::HeapSampler { words, index: 0 },
        ]
    );
    assert_eq!(
        func.resources[0].origin,
        Origin::HeapIndex {
            index: Var::S(4),
            sampler: false
        }
    );
    // Each load leaves the host index of its slot: the table base from the mirror plus the slot.
    let slots: Vec<Expr> = values(
        &func,
        |e| matches!(e, Expr::Bin(Bin::Add, base, _) if **base == Expr::Mirror(8)),
    );
    assert_eq!(
        slots,
        [
            Expr::bin(Bin::Add, Expr::Mirror(8), Expr::U32(1)),
            Expr::bin(Bin::Add, Expr::Mirror(8), Expr::U32(2)),
        ]
    );
    // No heap map: both descriptors are in a table.
    assert_eq!(needs.mirror_dwords, [8, 10]);
}

/// The compiler makes a constant with its bits at the top by reversing a small number. A
/// sampler word built that way is still a constant.
#[test]
fn a_sampler_word_made_by_reversing_bits_is_a_constant() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(12), s(8), 0x20));
    a.push(sopk(Op::s_movk_i32, s(24), 0x92));
    a.push(sop1(Op::s_mov_b32, s(25), lit(0x00c0_0000)));
    a.push(sop1(Op::s_brev_b32, s(26), int(40)));
    a.push(sop1(Op::s_mov_b32, s(27), int(0)));
    a.push(image(Op::image_sample_lz, v(4), v(2), s(12), s(24), 1, 0xf));
    a.push(sopp(Op::s_endpgm, 0));
    let (func, _) = placed(&program(Stage::Pixel, a)).unwrap();
    assert_eq!(
        placements(&func)[1],
        Placement::HeapSampler {
            words: [0x92, 0x00c0_0000, 0x1400_0000, 0],
            index: 0
        }
    );
}

/// A lane mask saved into a register pair overwrites a descriptor word that was there.
#[test]
fn a_mask_written_over_a_descriptor_word_ends_the_descriptor() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(4), s(8), 0x20));
    a.push(sop1(Op::s_mov_b64, s(6), EXEC));
    a.push(smem(Op::s_buffer_load_dword, s(0), s(4), 0));
    a.push(sopp(Op::s_endpgm, 0));
    let refusal = placed(&program(Stage::Pixel, a)).unwrap_err();
    // Whichever pass sees it first refuses it: the lane rules know that the buffer read takes a
    // mask register as a descriptor word, the tracer knows that the descriptor is cut.
    let key = refusal.key();
    assert!(
        key == "resource:partial-descriptor" || key == "exec:mask-as-data",
        "{key}"
    );
}

/// The source asked for the size of a buffer: the code reads the stride out of dword 1 and the
/// record count out of dword 2 of the console descriptor. The loads leave those fields behind,
/// taken from the view shadow row of the host slot.
#[test]
fn the_stride_and_record_count_of_a_buffer_come_from_the_view_shadow() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(9), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(48), s(8), 0x60));
    a.push(sop2(Op::s_lshr_b32, s(0), s(49), int(16)));
    a.push(sop2(Op::s_add_i32, s(1), s(50), int(-1)));
    a.push(sop2(Op::s_mul_i32, s(2), s(1), s(0)));
    a.push(smem(Op::s_buffer_load_dword, s(4), s(48), 0));
    a.push(vop2(Op::v_add_u32, v(1), s(2), v(1)));
    a.push(vop2(Op::v_add_u32, v(1), s(4), v(1)));
    a.push(export(0, 0xf, [1, 1, 1, 1], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = program(Stage::Pixel, a);
    p.binding_table = vec![record(RegisterClass::Srv, 20, 8, 24)];
    let (func, needs) = placed(&p).unwrap();
    let slot = Expr::bin(Bin::Add, Expr::Mirror(8), Expr::U32(3));
    let fields = values(&func, |e| {
        matches!(e, Expr::Call(h, _) if h.name == "decant_shadow_count")
            || matches!(e, Expr::Bin(Bin::Shl, f, _) if matches!(&**f, Expr::Call(h, _) if h.name == "decant_shadow_stride"))
    });
    assert_eq!(fields.len(), 2, "{fields:?}");
    for f in &fields {
        let call = match f {
            Expr::Bin(_, call, _) => &**call,
            call => call,
        };
        let Expr::Call(_, args) = call else {
            unreachable!()
        };
        assert_eq!(args[0], slot);
    }
    assert_eq!(needs.mirror_dwords, [8]);

    // Without the view shadow the program is refused by name.
    let bare = Options {
        mirror: Some(MirrorAbi {
            view_shadow: None,
            ..MirrorAbi::default()
        }),
        ..Options::default()
    };
    let refusal = placed_with(&p, &RootSignature::default(), &bare).unwrap_err();
    assert_eq!(refusal.key(), "resource:no-view-shadow");
}

/// The address in dword 0 of a buffer and the format bits in dword 1 of an image have no value to give.
#[test]
fn the_address_and_image_words_stay_refused_as_numbers() {
    for (load, reg) in [(Op::s_load_dwordx4, 4u8), (Op::s_load_dwordx8, 5)] {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(9), int(0)));
        a.push(smem(load, s(4), s(8), 0x20));
        a.push(sop2(Op::s_lshr_b32, s(0), s(reg), int(8)));
        a.push(sopp(Op::s_endpgm, 0));
        let refusal = placed(&program(Stage::Pixel, a)).unwrap_err();
        assert_eq!(refusal.key(), "resource:descriptor-as-data");
    }
}

/// The source asked for the size of a texture: the code reads width, height, the mip count and
/// the slice count out of the console image descriptor. The loads leave made up words with
/// those fields, from the size of the PC texture in the same heap slot. One program per kind
/// of image the container can declare, each compiled.
#[test]
fn the_size_fields_of_an_image_descriptor_come_from_the_texture() {
    let sig = |semantic: &str, system_value: u32| SigElement {
        semantic: semantic.to_owned(),
        index: 0,
        system_value,
        component_type: 3,
        register: 0,
        mask: 15,
        rw_mask: 0,
    };
    for (kind, class, helper) in [
        (1u32, RegisterClass::Srv, "decant_img_1d("),
        (2, RegisterClass::Srv, "decant_img_2d("),
        (3, RegisterClass::Srv, "decant_img_2d_ms("),
        (4, RegisterClass::Srv, "decant_img_3d("),
        (5, RegisterClass::Srv, "decant_img_cube("),
        (6, RegisterClass::Srv, "decant_img_1d_array("),
        (7, RegisterClass::Srv, "decant_img_2d_array("),
        (8, RegisterClass::Srv, "decant_img_2d_ms_array("),
        (9, RegisterClass::Srv, "decant_img_cube_array("),
        (1, RegisterClass::Uav, "decant_img_rw_1d("),
        (2, RegisterClass::Uav, "decant_img_rw_2d("),
        (4, RegisterClass::Uav, "decant_img_rw_3d("),
        (6, RegisterClass::Uav, "decant_img_rw_1d_array("),
        (7, RegisterClass::Uav, "decant_img_rw_2d_array("),
    ] {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(9), int(0)));
        a.push(smem(Op::s_load_dwordx8, s(4), s(8), 0x20));
        // Width minus one: two bits of dword 1 and twelve of dword 2.
        a.push(sop2(Op::s_lshr_b32, s(0), s(5), int(30)));
        a.push(sop2(Op::s_lshl_b32, s(1), s(6), int(2)));
        a.push(sop2(Op::s_or_b32, s(0), s(0), s(1)));
        // Height minus one, the last level and the last slice.
        a.push(sop2(Op::s_bfe_u32, s(1), s(6), lit(0x000e_000e)));
        a.push(sop2(Op::s_bfe_u32, s(2), s(7), lit(0x0004_0010)));
        a.push(sop2(Op::s_and_b32, s(3), s(8), lit(0x1fff)));
        for k in 0..4 {
            a.push(vop1(Op::v_mov_b32, v(10 + k), s(k)));
        }
        a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(0), v(10), v(11)));
        a.push(vop2(Op::v_cvt_pkrtz_f16_f32, v(1), v(12), v(13)));
        a.push(export(0, 0xf, [0, 1, 0, 0], true, true, true));
        a.push(sopp(Op::s_endpgm, 0));
        let p = StageProgram {
            inputs: vec![sig("SV_Position", 1)],
            outputs: vec![sig("SV_Target", 64)],
            binding_table: vec![record(class, 7, 8, 8)],
            resources: vec![DeclaredResource {
                class,
                lower: 7,
                upper: 7,
                space: 1,
                psv_type: 3,
                psv_kind: kind,
            }],
            name: "image_size".to_owned(),
            ..StageProgram::new(Stage::Pixel, a.finish())
        };
        let host = CountingHost::default();
        let t = decant_shader::translate(
            std::slice::from_ref(&p),
            &RootSignature::default(),
            &Options::default(),
            &host,
        )
        .unwrap_or_else(|r| panic!("kind {kind}: {r}"));
        let stage = &t.stages[0];
        // Dwords 1 to 4 of the slot at mirror dword 8 plus 1.
        for k in 1..=4 {
            let call = format!("{helper}(decant_mirror[2].x + 1u), {k}u)");
            assert!(
                stage.hlsl.contains(&call),
                "missing {call} in\n{}",
                stage.hlsl
            );
        }
        assert!(stage.needs.resource_heap, "a size query indexes the heap");
        if let Some(dxc) = common::find_dxc()
            && let Err(e) = Compiler::compile(
                &dxc,
                &stage.hlsl,
                &stage.entry,
                &stage.profile,
                &stage.dxc_args,
            )
        {
            panic!(
                "kind {kind}: {e}
{}",
                stage.hlsl
            );
        }
    }
}

/// A texture taken from the whole heap that the program only asks for its size: nothing says
/// what kind of image it is, so the size comes from the view shadow row of its host slot.
#[test]
fn the_size_of_an_image_of_unknown_kind_comes_from_the_view_shadow_row() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_mov_b32, s(20), s(15)));
    a.push(sop1(Op::s_mov_b32, s(21), int(0)));
    a.push(sop2(Op::s_lshl_b32, s(24), s(13), int(5)));
    a.push(smem_at(Op::s_load_dwordx8, s(4), s(20), 0, s(24)));
    a.push(vop3(Op::v_alignbit_b32, v(16), s(6), s(5), int(30)));
    a.push(export(0, 0xf, [16, 16, 16, 16], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut p = program(Stage::Pixel, a);
    p.user_data = pixel_layout();
    let (func, needs) = placed(&p).unwrap();
    let rows = values(
        &func,
        |e| matches!(e, Expr::Call(h, _) if h.name == "decant_img_row"),
    );
    assert_eq!(rows.len(), 2, "dwords 1 and 2: {rows:?}");
    assert_eq!(needs.mirror_dwords, [13]);
    let bare = Options {
        mirror: Some(MirrorAbi {
            view_shadow: None,
            ..MirrorAbi::default()
        }),
        ..Options::default()
    };
    let refusal = placed_with(&p, &RootSignature::default(), &bare).unwrap_err();
    assert_eq!(refusal.key(), "resource:no-view-shadow");
}

/// A table of constants the compiler put behind the code, read with an index: the address is
/// `s_getpc_b64` plus a constant plus the index, and the load becomes a read of the program's
/// own words.
#[test]
fn a_load_from_a_table_behind_the_code_reads_the_programs_own_words() {
    let mut a = Asm::new();
    a.push(sop1(Op::s_getpc_b64, s(22), int(0)));
    a.push(sop2(Op::s_add_u32, s(22), s(22), lit(0x40)));
    // The index: a root constant times four.
    a.push(sop2(Op::s_lshl_b32, s(4), s(13), int(2)));
    a.push(sop1(Op::s_mov_b32, s(5), int(0)));
    a.push(sop2(Op::s_add_u32, s(8), s(4), s(22)));
    a.push(sop2(Op::s_add_u32, s(9), s(5), s(23)));
    a.push(smem(Op::s_load_dword, s(8), s(8), 0));
    a.push(vop1(Op::v_mov_b32, v(1), s(8)));
    a.push(export(0, 0xf, [1, 1, 1, 1], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let (func, needs) = placed(&program(Stage::Pixel, a)).unwrap();
    assert!(func.resources.is_empty());
    let words = values(&func, |e| matches!(e, Expr::CodeWord(_)));
    assert_eq!(words.len(), 1, "{words:?}");
    // Only the index comes from the mirror.
    assert_eq!(needs.mirror_dwords, [13]);
}

/// Two branches each load a different image descriptor and pass through a chain of diamond
/// `if`s before joining at an `image_store`. Evaluating the dataflow lowest-block-first keeps the
/// join from hitting the widening threshold and dropping the descriptor.
#[test]
fn merged_image_descriptor_after_many_diamonds_is_traced() {
    let mut a = Asm::new();
    let branch_b = a.label();
    let join = a.label();
    a.push(sop1(Op::s_mov_b32, s(24), s(4)));
    a.push(sop1(Op::s_mov_b32, s(25), int(0)));
    a.push(sopc(Op::s_cmp_eq_u32, s(6), int(0)));
    a.branch(Op::s_cbranch_scc1, branch_b);

    a.push(smem(Op::s_load_dwordx8, s(4), s(24), 0x120));
    for i in 0..6 {
        let skip = a.label();
        a.push(vopc(Op::v_cmp_eq_u32, v(0), int(i)));
        a.push(sop1(
            Op::s_and_saveexec_b32,
            s(14),
            Operand::Reg(Reg::VccLo),
        ));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop1(Op::v_mov_b32, v(2), int(i)));
        a.bind(skip);
        a.push(sop1(Op::s_mov_b32, Operand::Reg(Reg::ExecLo), s(14)));
    }
    a.branch(Op::s_branch, join);

    a.bind(branch_b);
    a.push(smem(Op::s_load_dwordx8, s(4), s(24), 0x140));
    for i in 0..6 {
        let skip = a.label();
        a.push(vopc(Op::v_cmp_eq_u32, v(1), int(i)));
        a.push(sop1(
            Op::s_and_saveexec_b32,
            s(14),
            Operand::Reg(Reg::VccLo),
        ));
        a.branch(Op::s_cbranch_execz, skip);
        a.push(vop1(Op::v_mov_b32, v(2), int(i)));
        a.bind(skip);
        a.push(sop1(Op::s_mov_b32, Operand::Reg(Reg::ExecLo), s(14)));
    }

    a.bind(join);
    a.push(image(Op::image_store, v(2), v(0), s(4), s(0), 1, 0xf));
    a.push(sopp(Op::s_endpgm, 0));

    let mut p = program(Stage::Compute, a);
    p.threads = [8, 4, 1];
    p.user_data = vec![UserDataSlot {
        sgpr: 4,
        kind: UserDataKind::Table,
    }];
    let (func, needs) = placed(&p).unwrap();
    assert_eq!(func.resources.len(), 1);
    assert_eq!(
        func.resources[0].origin,
        Origin::HeapIndex {
            index: Var::S(4),
            sampler: false,
        }
    );
    assert_eq!(
        placements(&func),
        [Placement::HeapAtConsoleIndex {
            class: RegisterClass::Uav,
        }]
    );
    assert_eq!(needs.mirror_dwords, [4]);
    assert!(needs.resource_heap);
}

#[test]
fn waterfall_comparison_of_identical_descriptor_words_is_accepted() {
    let mut a = Asm::new();
    let again = a.label();
    a.push(sop1(Op::s_mov_b32, s(5), int(0)));
    a.push(smem(Op::s_load_dwordx8, s(8), s(4), 0x40));
    for i in 0..8u8 {
        a.push(vop1(Op::v_mov_b32, v(57 + i), s(8 + i)));
    }
    a.push(sop1(Op::s_mov_b64, s(2), EXEC));
    a.bind(again);
    for i in 0..8u8 {
        a.push(vop1(Op::v_readfirstlane_b32, s(12 + i), v(57 + i)));
    }
    a.push(vopc(Op::v_cmp_eq_u64, s(12), v(57)));
    a.push(vop3(Op::v_cmp_eq_u64, s(0), s(14), v(59), Operand::None));
    a.push(sop2(Op::s_and_b64, s(0), VCC, s(0)));
    a.push(sop1(Op::s_and_saveexec_b64, s(0), s(0)));
    a.push(image(Op::image_load, v(2), v(0), s(12), s(0), 1, 0xf));
    a.push(sop2(Op::s_xor_b64, EXEC, EXEC, s(0)));
    a.branch(Op::s_cbranch_execnz, again);
    a.push(sop1(Op::s_mov_b64, EXEC, s(2)));
    a.push(sopp(Op::s_endpgm, 0));

    let mut p = program(Stage::Compute, a);
    p.threads = [64, 1, 1];
    p.user_data = vec![UserDataSlot {
        sgpr: 4,
        kind: UserDataKind::Table,
    }];
    let (func, _) = placed(&p).unwrap();
    assert_eq!(func.resources.len(), 1);
}

#[test]
fn image_and_buffer_loads_clear_stale_descriptor_words_in_vgprs() {
    let build = |overwrite_with: Option<Insn>| {
        let mut a = Asm::new();
        a.push(sop1(Op::s_mov_b32, s(5), int(0)));
        a.push(smem(Op::s_load_dwordx8, s(8), s(4), 0x40));
        a.push(smem(Op::s_load_dwordx4, s(16), s(4), 0x80));
        a.push(vop1(Op::v_mov_b32, v(57), s(8)));
        if let Some(insn) = overwrite_with {
            a.push(insn);
        }
        a.push(vop2(Op::v_add_f32, v(57), v(57), float(1.0)));
        a.push(sopp(Op::s_endpgm, 0));
        let mut p = program(Stage::Compute, a);
        p.threads = [32, 1, 1];
        p.user_data = vec![UserDataSlot {
            sgpr: 4,
            kind: UserDataKind::Table,
        }];
        p.binding_table = vec![record(RegisterClass::Srv, 0, 4, 32)];
        p
    };

    // Without an overwrite, using the moved descriptor word in arithmetic is still refused.
    let refusal = placed(&build(None)).unwrap_err();
    assert_eq!(refusal.key(), "resource:descriptor-as-data");

    // An image load writing v56..v57 (dmask=0x3) clears v57's stale descriptor provenance.
    let img = image(Op::image_load, v(56), v(0), s(8), s(0), 1, 0x3);
    placed(&build(Some(img))).unwrap();

    // A buffer load writing v57 clears v57's stale descriptor provenance.
    let buf = buffer_load(Op::buffer_load_dword, v(57), v(0), s(16), 0, true);
    placed(&build(Some(buf))).unwrap();
}
