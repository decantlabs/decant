//! The mesh stage: the one shape every mesh program of the title has, built with the assembler,
//! translated to an HLSL mesh shader and compiled with DXC.
//!
//! The shape (see `stage/mesh.rs`): counts from a buffer, a vertex body for the threads below
//! the vertex count, a primitive word for the threads below the primitive count, the
//! allocation message from thread 0 with the fallback for an empty group, two barriers, then
//! the exports. The programs are written here by hand. No game data is involved.

mod common;

use decant_shader::api::*;
use decant_shader::asm::*;
use decant_shader::decode::Operand;
use decant_shader::stage::mesh;
use decant_shader::tables::Op;
use decant_shader::{translate, translate_to_dxil};

fn element(semantic: &str, index: u32, ty: u32, register: u32, mask: u8) -> SigElement {
    let system_value = match semantic {
        "SV_Position" => 1,
        "SV_ClipDistance" => 2,
        "SV_Target" => 64,
        _ => 0,
    };
    SigElement {
        semantic: semantic.to_owned(),
        index,
        system_value,
        component_type: ty,
        register,
        mask,
        rw_mask: 0,
    }
}

/// How the skeleton below varies.
struct Shape {
    /// The message number thread 0 sends.
    message: u16,
    /// Jump back to the allocation block after the exports (never in a real program).
    looped: bool,
    /// Every thread stores a value in the local data share and reads its neighbour's after a
    /// barrier, before the vertex body (the programs with control points do that).
    shared: bool,
    swizzle: bool,
}

impl Default for Shape {
    fn default() -> Shape {
        Shape {
            message: 9,
            looped: false,
            shared: false,
            swizzle: false,
        }
    }
}

/// The skeleton of a mesh program of the title. `vertex_exports` writes the exports of a vertex
/// thread besides the position: registers v8 to v15 hold values for them.
fn skeleton(shape: &Shape, vertex_exports: &dyn Fn(&mut Asm)) -> Vec<u32> {
    let mut a = Asm::new();
    let (after_vertices, after_indices) = (a.label(), a.label());
    let (alloc, skip, no_primitive, end) = (a.label(), a.label(), a.label(), a.label());
    a.push(sopp(Op::s_inst_prefetch, 3));
    a.push(sop1(Op::s_mov_b64, EXEC, int(-1)));
    // The meshlet header: the vertex count in s22 and the primitive count in s28, read from
    // the buffer in slot 0 of the table in s17.
    a.push(sop1(Op::s_mov_b32, s(10), s(17)));
    a.push(sop1(Op::s_mov_b32, s(11), int(0)));
    a.push(smem(Op::s_load_dwordx4, s(24), s(10), 0));
    a.push(smem(Op::s_buffer_load_dword, s(22), s(24), 0));
    a.push(smem(Op::s_buffer_load_dword, s(28), s(24), 4));
    // The thread number: the lane number plus 64 times the wave index from s3. And the group
    // number: the entry value of v5, read back through the first lane, shifted right by 7.
    a.push(vop3(
        Op::v_mbcnt_lo_u32_b32,
        v(0),
        int(-1),
        int(0),
        Operand::None,
    ));
    a.push(vop1(Op::v_readfirstlane_b32, s(0), v(5)));
    a.push(sop2(Op::s_bfe_u32, s(1), s(3), lit(0x0004_0018)));
    a.push(vop3(
        Op::v_mbcnt_hi_u32_b32_e64,
        v(0),
        int(-1),
        v(0),
        Operand::None,
    ));
    a.push(sop2(Op::s_lshr_b32, s(12), s(0), int(7)));
    a.push(vop3(Op::v_lshl_add_u32, v(0), s(1), int(6), v(0)));
    if shape.swizzle {
        a.push(ds(Op::ds_swizzle_b32, v(31), v(0), v(0), v(0), 92, 0));
    }
    if shape.shared {
        // Dword `thread` of the memory gets the thread number, then every thread reads the
        // dword of the thread after it.
        a.push(vop2(Op::v_lshlrev_b32, v(30), int(2), v(0)));
        a.push(ds(
            Op::ds_write_b32,
            Operand::None,
            v(30),
            v(0),
            Operand::None,
            0,
            0,
        ));
        a.push(sopp(Op::s_barrier, 0));
        a.push(ds(
            Op::ds_read_b32,
            v(31),
            v(30),
            Operand::None,
            Operand::None,
            4,
            0,
        ));
    }
    // The vertex body, for the threads below the vertex count.
    a.push(sop1(Op::s_mov_b64, s(20), EXEC));
    a.push(vopc(Op::v_cmpx_gt_u32, s(22), v(0)));
    a.branch(Op::s_cbranch_execz, after_vertices);
    for k in 0..4 {
        a.push(vop1(Op::v_mov_b32, v(1 + k), float(1.0)));
    }
    // Parameters: the group number, so that it is not dead code.
    for k in 0..8 {
        a.push(vop1(Op::v_mov_b32, v(8 + k), s(12)));
    }
    a.bind(after_vertices);
    a.push(sop1(Op::s_mov_b64, EXEC, s(20)));
    // The primitive word, for the threads below the primitive count.
    a.push(sop1(Op::s_mov_b64, s(0), EXEC));
    a.push(vopc(Op::v_cmpx_gt_u32, s(28), v(0)));
    a.branch(Op::s_cbranch_execz, after_indices);
    a.push(vop1(Op::v_mov_b32, v(5), lit(1 << 10 | 2 << 20)));
    a.bind(after_indices);
    a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
    // A group without vertices is a bad group: the mask in s2 says so, s4 is its opposite.
    a.push(sopc(Op::s_cmp_eq_u32, s(22), int(0)));
    a.push(sop2(Op::s_cselect_b64, s(2), EXEC, int(0)));
    a.push(sop2(Op::s_xor_b64, s(4), s(2), int(-1)));
    // Thread 0 allocates.
    a.push(sop1(Op::s_mov_b64, s(0), EXEC));
    a.bind(alloc);
    a.push(vopc(Op::v_cmpx_eq_u32, int(0), v(0)));
    a.branch(Op::s_cbranch_execz, skip);
    a.push(sop2(Op::s_lshl_b32, s(7), s(28), int(12)));
    a.push(sop2(Op::s_andn2_b64, VCC, EXEC, s(2)));
    a.push(sop2(Op::s_or_b32, s(6), s(7), s(22)));
    a.push(sop1(Op::s_mov_b32, M0, s(6)));
    a.push(sopp(Op::s_sendmsg, shape.message));
    a.branch(Op::s_cbranch_vccnz, skip);
    // The bad group: one vertex without a position and one null primitive.
    a.push(sopp(Op::s_barrier, 0));
    a.push(vop1(Op::v_bfrev_b32, v(6), int(1)));
    a.push(export(12, 0, [0, 0, 0, 0], false, true, false));
    a.push(export(20, 1, [6, 0, 0, 0], false, true, false));
    a.bind(skip);
    a.push(sop1(Op::s_mov_b64, EXEC, s(0)));
    a.push(sopp(Op::s_barrier, 0));
    // The primitive of this thread.
    a.push(vopc(Op::v_cmp_gt_u32, s(28), v(0)));
    a.push(sop2(Op::s_and_b64, s(0), VCC, s(4)));
    a.push(sop1(Op::s_and_saveexec_b64, s(6), s(0)));
    a.branch(Op::s_cbranch_execz, no_primitive);
    a.push(export(20, 1, [5, 0, 0, 0], false, true, false));
    a.bind(no_primitive);
    a.push(sop1(Op::s_mov_b64, EXEC, s(6)));
    // The vertex of this thread.
    a.push(vopc(Op::v_cmp_gt_u32, s(22), v(0)));
    a.push(sop2(Op::s_and_b64, s(0), VCC, s(4)));
    a.push(sop1(Op::s_and_saveexec_b64, s(0), s(0)));
    a.branch(Op::s_cbranch_execz, end);
    vertex_exports(&mut a);
    a.push(export(12, 0xf, [1, 2, 3, 4], false, true, false));
    a.bind(end);
    if shape.looped {
        a.push(sopc(Op::s_cmp_eq_u32, s(22), int(0)));
        a.branch(Op::s_cbranch_scc1, alloc);
    }
    a.push(sopp(Op::s_endpgm, 0));
    a.finish()
}

fn program(code: Vec<u32>, outputs: Vec<SigElement>, name: &str) -> StageProgram {
    StageProgram {
        outputs,
        threads: [128, 1, 1],
        // The meshlet buffer: `t66` in space 1, the first descriptor of the table in s17.
        resources: vec![DeclaredResource {
            class: RegisterClass::Srv,
            lower: 66,
            upper: 66,
            space: 1,
            psv_type: 4,
            psv_kind: 11,
        }],
        binding_table: vec![TableBinding {
            class: RegisterClass::Srv,
            register: 66,
            sgpr: Stage::Mesh.mirror_dword(17),
            word_offset: 0,
        }],
        name: name.to_owned(),
        ..StageProgram::new(Stage::Mesh, code)
    }
}

/// A position, one 32 bit parameter and a clip distance that is also a parameter.
fn plain() -> StageProgram {
    let code = skeleton(&Shape::default(), &|a| {
        a.push(export(13, 0x1, [12, 0, 0, 0], false, false, false));
        a.push(export(32, 0x1, [12, 0, 0, 0], false, false, false));
        a.push(export(33, 0xf, [8, 9, 10, 11], false, false, false));
    });
    let outputs = vec![
        element("SV_ClipDistance", 0, 3, 0, 1),
        element("TEXCOORD", 0, 3, 1, 15),
        element("SV_Position", 0, 3, 2, 15),
    ];
    program(code, outputs, "mesh_plain")
}

fn hlsl(p: &StageProgram) -> String {
    let t = translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{}: {r}", p.name));
    assert_eq!(t.stages[0].profile, "ms_6_6");
    assert_eq!(t.stages[0].entry, "main");
    t.stages[0].hlsl.clone()
}

fn refusal(p: &StageProgram) -> String {
    translate(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .expect_err("the program is refused")
    .key()
}

fn compile(stages: &[StageProgram]) {
    let Some(dxc) = common::find_dxc() else {
        eprintln!("dxc.exe not found: the compile step is skipped");
        return;
    };
    let compiled = translate_to_dxil(
        stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
        &dxc,
    )
    .unwrap_or_else(|r| panic!("{r}"));
    assert_eq!(compiled.len(), stages.len());
    assert!(compiled.iter().all(|c| c.dxil.starts_with(b"DXBC")));
}

#[test]
fn the_skeleton_becomes_a_mesh_shader() {
    let p = plain();
    assert_eq!(mesh::check(&p), Ok(()));
    let text = hlsl(&p);
    // One function per thread, and the entry point around it.
    assert!(
        text.contains("DecantOut decant_mesh_thread(uint3 decant_tid"),
        "{text}"
    );
    assert!(
        text.contains(
            "
[outputtopology(\"triangle\")]
void main(uint3 decant_tid"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "out vertices DecantOut decant_vertices[128], out indices uint3 decant_triangles[128]"
        ),
        "{text}"
    );
    assert_eq!(text.matches("[numthreads(128, 1, 1)]").count(), 1, "{text}");
    // The counts are set once, in the entry point, after the thread function ran.
    assert_eq!(text.matches("SetMeshOutputCounts(").count(), 1, "{text}");
    let call = text.find("= decant_mesh_thread(decant_tid, ").expect("the call");
    assert!(call < text.find("SetMeshOutputCounts(").unwrap());
    assert!(
        text.find("SetMeshOutputCounts(").unwrap()
            < text.find("decant_vertices[decant_tid.x] =").unwrap()
    );
    // The barriers are gone, the message became the count word of every thread.
    assert!(!text.contains("Barrier"), "{text}");
    let counts = text
        .lines()
        .find(|l| l.contains("decant_mesh_counts(m0)"))
        .expect("the allocation");
    assert!(
        counts.trim_start().starts_with("decant_mesh_counts"),
        "{counts}"
    );
    // The lane number is the thread number in its wave, the group number comes from the group id.
    assert!(text.contains("(decant_tid.x & 63u)"), "{text}");
    assert!(text.contains("decant_gid.x"), "{text}");
    // Exports only count in an active lane.
    assert!(text.contains("if (exec) decant_mesh_primitive(v5);"), "{text}");
    assert!(text.contains("if (exec) decant_mesh_has_vertex();"), "{text}");
}

#[test]
fn every_thread_runs_the_allocation_block() {
    // On the console only thread 0 sends the message, behind a branch on its lane bit. After the
    // stage forced that branch, the count word is written at the top level of the thread
    // function: no `if` and no loop is open around it.
    let text = hlsl(&plain());
    let line = text
        .lines()
        .find(|l| l.contains("decant_mesh_counts(m0)"))
        .unwrap_or_else(|| {
            panic!(
                "the allocation
{text}"
            )
        });
    let indent = line.len() - line.trim_start().len();
    assert_eq!(
        indent, 4,
        "the allocation is nested:
{text}"
    );
    assert!(line.trim_start().starts_with("decant_mesh_counts"), "{line}");
}

#[test]
fn the_mesh_shader_compiles() {
    compile(&[plain()]);
}

#[test]
fn sixteen_bit_registers_are_unpacked_from_their_shared_slot() {
    // COLOR0 and TEXCOORD1 are 16 bit and share parameter 0, TEXCOORD0 is parameter 1.
    let code = skeleton(&Shape::default(), &|a| {
        a.push(export(32, 0xf, [8, 9, 10, 11], false, false, false));
        a.push(export(33, 0x3, [12, 13, 0, 0], false, false, false));
    });
    let outputs = vec![
        element("SV_Position", 0, 3, 0, 15),
        element("COLOR", 0, 6, 1, 15),
        element("TEXCOORD", 0, 3, 2, 3),
        element("TEXCOORD", 1, 6, 3, 15),
    ];
    let mut p = program(code, outputs, "mesh_sixteen");
    p.param_exports = vec![
        ParamExport {
            slot: 0,
            kind: 6,
            register: 1,
            sixteen_bit_register: 1,
            flags: 3,
        },
        ParamExport {
            slot: 1,
            kind: 6,
            register: 2,
            sixteen_bit_register: 0,
            flags: 0,
        },
    ];
    let text = hlsl(&p);
    // COLOR0.y is the high half of dword 0 (v8), TEXCOORD1.z the low half of dword 3 (v11).
    assert!(
        text.contains("decant_mesh_f16((v8 >> (16u & 31u)))")
            || text.contains("decant_mesh_f16((v8 >> 16u))"),
        "{text}"
    );
    assert!(text.contains("decant_mesh_f16(v11)"), "{text}");
    // TEXCOORD0 is 32 bit: its components are the registers.
    assert!(!text.contains("decant_mesh_f16(v12)"), "{text}");
    compile(&[p]);
}

#[test]
fn a_pixel_program_links_to_the_mesh_program() {
    // The pixel program reads TEXCOORD0, which the mesh program writes as parameter 1.
    let mut a = Asm::new();
    for chan in 0..4u8 {
        a.push(interp(Op::v_interp_p1_f32, v(2 + chan), v(0), 0, chan));
    }
    for chan in 0..4u8 {
        a.push(interp(Op::v_interp_p2_f32, v(2 + chan), v(1), 0, chan));
    }
    a.push(export(0, 0xf, [2, 3, 4, 5], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut read = element("TEXCOORD", 0, 3, 1, 15);
    read.rw_mask = 15;
    let pixel = StageProgram {
        inputs: vec![element("SV_Position", 0, 3, 0, 15), read],
        outputs: vec![element("SV_Target", 0, 3, 0, 15)],
        name: "mesh_pixel".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    let stages = [plain(), pixel];
    let t = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{r}"));
    // The pixel input struct has the members of the mesh output struct, in the same order.
    let members = |text: &str, name: &str| -> Vec<String> {
        let start = text.find(&format!("struct {name} {{")).expect("the struct");
        text[start..]
            .lines()
            .skip(1)
            .take_while(|l| !l.starts_with("};"))
            .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_owned())
            .collect()
    };
    assert_eq!(
        members(&t.stages[0].hlsl, "DecantOut"),
        members(&t.stages[1].hlsl, "DecantIn")
    );
    compile(&stages);
}

/// A pixel program that reads the four components of `TEXCOORD<index>`, declared in input
/// register `register`.
fn pixel_reading(index: u32, register: u32) -> StageProgram {
    let mut a = Asm::new();
    for chan in 0..4u8 {
        a.push(interp(Op::v_interp_p1_f32, v(2 + chan), v(0), 0, chan));
    }
    for chan in 0..4u8 {
        a.push(interp(Op::v_interp_p2_f32, v(2 + chan), v(1), 0, chan));
    }
    a.push(export(0, 0xf, [2, 3, 4, 5], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut read = element("TEXCOORD", index, 3, register, 15);
    read.rw_mask = 15;
    StageProgram {
        inputs: vec![element("SV_Position", 0, 3, 0, 15), read],
        outputs: vec![element("SV_Target", 0, 3, 0, 15)],
        name: "mesh_pixel".to_owned(),
        ..StageProgram::new(Stage::Pixel, a.finish())
    }
}

#[test]
fn a_pipeline_leaves_out_what_its_pixel_program_does_not_read() {
    // The mesh program writes 17 vectors besides the position: 288 bytes, over the 256 that
    // Direct3D 12 allows with 128 vertices. Its pixel program reads TEXCOORD3 alone.
    let code = skeleton(&Shape::default(), &|a| {
        for slot in 0..17u8 {
            a.push(export(32 + slot, 0xf, [8, 9, 10, 11], false, false, false));
        }
    });
    let mut outputs = vec![element("SV_Position", 0, 3, 0, 15)];
    outputs.extend((0..17).map(|k| element("TEXCOORD", k, 3, 1 + k, 15)));
    let mesh_program = program(code, outputs, "mesh_wide_vertex");
    // Alone it is refused with its size.
    let alone = translate(
        std::slice::from_ref(&mesh_program),
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .expect_err("too large");
    assert_eq!(alone.key(), "stage:mesh-output-size");
    assert!(alone.detail.contains("288 bytes"), "{}", alone.detail);
    // In the pipeline both structs have the position and TEXCOORD3, and nothing else.
    let stages = [mesh_program, pixel_reading(3, 1)];
    let t = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{r}"));
    let members = |text: &str, name: &str| -> Vec<String> {
        let start = text.find(&format!("struct {name} {{")).expect("the struct");
        text[start..]
            .lines()
            .skip(1)
            .take_while(|l| !l.starts_with("};"))
            .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_owned())
            .collect()
    };
    assert_eq!(
        members(&t.stages[0].hlsl, "DecantOut"),
        ["SV_Position;", "TEXCOORD3;"]
    );
    assert_eq!(
        members(&t.stages[0].hlsl, "DecantOut"),
        members(&t.stages[1].hlsl, "DecantIn")
    );
    // The export of parameter 3 is the one that still writes a member.
    assert_eq!(t.stages[0].hlsl.matches("decant_out.o1.x = ").count(), 1);
    compile(&stages);
}

#[test]
fn a_mesh_bitmask_swizzle_can_read_a_patch_corner() {
    let shape = Shape {
        swizzle: true,
        ..Shape::default()
    };
    let code = skeleton(&shape, &|a| {
        a.push(export(32, 0x1, [31, 0, 0, 0], false, false, false));
    });
    let p = program(
        code,
        vec![
            element("SV_Position", 0, 3, 0, 15),
            element("TEXCOORD", 0, 1, 1, 1),
        ],
        "mesh_swizzle",
    );
    let text = hlsl(&p);
    assert!(text.contains("groupshared uint decant_lds[128];"), "{text}");
    assert!(text.contains("decant_swizzle(decant_gi, 92u,"), "{text}");
    compile(&[p]);
}

#[test]
fn a_mesh_divergent_quad_swizzle_uses_wave_intrinsics_and_compiles() {
    let shape = Shape::default();
    let code = skeleton(&shape, &|a| {
        let done = a.label();
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(64)));
        a.branch(Op::s_cbranch_execz, done);
        a.push(ds(Op::ds_swizzle_b32, v(31), v(0), v(0), v(0), 92, 0));
        a.bind(done);
        a.push(export(32, 0x1, [31, 0, 0, 0], false, false, false));
    });
    let p = program(
        code,
        vec![
            element("SV_Position", 0, 3, 0, 15),
            element("TEXCOORD", 0, 1, 1, 1),
        ],
        "mesh_divergent_swizzle",
    );
    let text = hlsl(&p);
    assert!(text.contains("decant_swizzle_wave(92u,"), "{text}");
    assert!(!text.contains("[WaveSize("), "{text}");
    assert!(
        !text.contains("GroupMemoryBarrierWithGroupSync()"),
        "{text}"
    );
    compile(&[p]);

    // A non-quad divergent mesh swizzle crosses 4-lane quads and cannot declare WaveSize in SM 6.6.
    let wide = skeleton(&shape, &|a| {
        let done = a.label();
        a.push(vopc(Op::v_cmpx_lt_u32, v(0), int(64)));
        a.branch(Op::s_cbranch_execz, done);
        a.push(ds(Op::ds_swizzle_b32, v(31), v(0), v(0), v(0), 31, 32));
        a.bind(done);
        a.push(export(32, 0x1, [31, 0, 0, 0], false, false, false));
    });
    let wide_p = program(
        wide,
        vec![
            element("SV_Position", 0, 3, 0, 15),
            element("TEXCOORD", 0, 1, 1, 1),
        ],
        "mesh_divergent_wide_swizzle",
    );
    assert_eq!(
        decant_shader::lift_program(&wide_p).unwrap_err().key(),
        "exec:cross-lane"
    );
}

#[test]
fn local_data_share_memory_is_group_shared_with_its_barrier() {
    let shape = Shape {
        shared: true,
        ..Shape::default()
    };
    let code = skeleton(&shape, &|a| {
        // What the thread read from its neighbour is a parameter.
        a.push(export(32, 0x1, [31, 0, 0, 0], false, false, false));
    });
    let outputs = vec![
        element("SV_Position", 0, 3, 0, 15),
        element("TEXCOORD", 0, 1, 1, 1),
    ];
    let mut p = program(code, outputs, "mesh_shared");
    p.lds_bytes = 512;
    let text = hlsl(&p);
    // The array has the size the container gives, and only the barrier before the allocation
    // is a group sync: the two behind it order exports.
    assert!(text.contains("groupshared uint decant_lds[128];"), "{text}");
    assert_eq!(text.matches("decant_barrier();").count(), 1, "{text}");
    let sync = text.find("decant_barrier();").unwrap();
    assert!(sync < text.find("decant_mesh_counts(m0)").unwrap(), "{text}");
    compile(&[p]);
    // More memory than a mesh shader may declare.
    let mut p = program(
        skeleton(&shape, &|_| {}),
        vec![element("SV_Position", 0, 3, 0, 15)],
        "mesh_shared_large",
    );
    p.lds_bytes = 32 * 1024;
    assert_eq!(refusal(&p), "stage:mesh-lds-size");
}

#[test]
fn what_the_stage_does_not_carry_is_refused_by_name() {
    let outputs = || vec![element("SV_Position", 0, 3, 0, 15)];
    // A geometry message in a mesh program.
    let shape = Shape {
        message: 0x22,
        ..Shape::default()
    };
    let p = program(skeleton(&shape, &|_| {}), outputs(), "mesh_message");
    assert_eq!(refusal(&p), "stage:mesh-message");
    // The allocation in a loop.
    let shape = Shape {
        looped: true,
        ..Shape::default()
    };
    let p = program(skeleton(&shape, &|_| {}), outputs(), "mesh_loop");
    assert_eq!(mesh::check(&p).unwrap_err().key(), "stage:mesh-alloc-shape");
    assert_eq!(refusal(&p), "stage:mesh-alloc-shape");
    // A clip distance export without a clip distance element.
    let code = skeleton(&Shape::default(), &|a| {
        a.push(export(13, 0x1, [12, 0, 0, 0], false, false, false));
    });
    assert_eq!(
        refusal(&program(code, outputs(), "mesh_clip")),
        "stage:mesh-export-target"
    );
    // A parameter export wider than the signature register.
    let code = skeleton(&Shape::default(), &|a| {
        a.push(export(32, 0xf, [8, 9, 10, 11], false, false, false));
    });
    let mut wide = outputs();
    wide.push(element("TEXCOORD", 0, 3, 1, 3));
    assert_eq!(
        refusal(&program(code, wide, "mesh_wide")),
        "stage:mesh-export-layout"
    );
    // A second position target that is no clip distance (point size, layer, viewport).
    let code = skeleton(&Shape::default(), &|a| {
        a.push(export(15, 0x1, [12, 0, 0, 0], false, false, false));
    });
    assert_eq!(
        refusal(&program(code, outputs(), "mesh_misc")),
        "stage:mesh-export-target"
    );
}

#[test]
fn oversized_mesh_outputs_link_as_packed_half_pairs() {
    let code = skeleton(&Shape::default(), &|a| {
        // Parameter 0 carries two half4 registers. Parameters 1 through 14 are wide.
        for slot in 0..15u8 {
            a.push(export(32 + slot, 0xf, [8, 9, 10, 11], false, false, false));
        }
    });
    let mut outputs = vec![element("SV_Position", 0, 3, 0, 15)];
    outputs.extend((0..2).map(|k| element("TEXCOORD", k, 6, 1 + k, 15)));
    outputs.extend((2..16).map(|k| element("TEXCOORD", k, 3, 1 + k, 15)));
    let mut a = Asm::new();
    a.push(interp(Op::v_interp_p1_f32, v(2), v(0), 0, 0));
    a.push(interp(Op::v_interp_p2_f32, v(2), v(1), 0, 0));
    let mut flat = interp(Op::v_interp_mov_f32, v(3), int(2), 1, 3);
    flat.src[0] = Operand::Int(2);
    a.push(flat);
    a.push(vop1(Op::v_cvt_f32_f16, v(3), v(3)));
    a.push(export(0, 0xf, [2, 3, 2, 3], false, true, true));
    a.push(sopp(Op::s_endpgm, 0));
    let mut pixel = StageProgram {
        inputs: outputs.clone(),
        outputs: vec![element("SV_Target", 0, 3, 0, 15)],
        ..StageProgram::new(Stage::Pixel, a.finish())
    };
    for input in &mut pixel.inputs[1..] {
        input.rw_mask = input.mask;
    }
    pixel.pixel_inputs = vec![
        PixelInput {
            attr: 0,
            kind: 6,
            register: 1,
            high_register: None,
            channels: 15,
            flags: 0x1a,
        },
        PixelInput {
            attr: 1,
            kind: 6,
            register: 2,
            high_register: None,
            channels: 15,
            flags: 0x09,
        },
    ];
    let stages = [program(code, outputs, "mesh_half_pairs"), pixel];
    let t = translate(
        &stages,
        &RootSignature::default(),
        &Options::default(),
        &CountingHost::default(),
    )
    .unwrap_or_else(|r| panic!("{r}"));
    let ms = &t.stages[0].hlsl;
    let ps = &t.stages[1].hlsl;
    assert!(ms.contains("out vertices DecantMeshOut decant_vertices[128]"));
    assert!(ms.contains("nointerpolation uint2 o1 : TEXCOORD0;"));
    assert!(ms.contains("f32tof16(decant_vertex.o1.x) | (f32tof16(decant_vertex.o1.y) << 16u)"));
    assert!(ps.contains("nointerpolation uint2 i1 : TEXCOORD0;"));
    assert!(ps.contains("GetAttributeAtVertex(decant_mesh_in.i1.x, 2)"));
    assert!(ps.contains("decant_in.i2.w = f16tof32((decant_mesh_in.i2.y >> 16u) & 0xffffu);"));
    assert!(ps.contains("float3 decant_bary : SV_Barycentrics;"));
    compile(&stages);
}
