//! The probe translates a T# chosen by a branch merge (finite descriptor choice, or whole-T# loads
//! from a descriptor table) into a valid compute module that writes an image. Skipped when the probe (`DECANT_SHADER_PROBE`, otherwise
//! `src/shaders/decant_shader/decant_shader_probe.exe`) or spirv-val (`DECANT_SHADER_TEST_SPIRV_BIN`, otherwise
//! the `PATH`) is missing. Every shader here is assembled by hand from the RDNA2 instruction
//! formats (no game shader).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// One element of a hand-assembled program: words, a label, or a branch to a label.
enum Item {
    Words(Vec<u32>),
    Label(&'static str),
    /// SOPP branch (s_branch 2, s_cbranch_scc1 5, s_cbranch_execz 8): offset in words after it.
    Branch(u32, &'static str),
}

fn assemble(items: &[Item]) -> Vec<u32> {
    let (mut labels, mut pc) = (HashMap::new(), 0usize);
    for item in items {
        match item {
            Item::Words(words) => pc += words.len(),
            Item::Label(name) => {
                labels.insert(*name, pc);
            }
            Item::Branch(..) => pc += 1,
        }
    }
    let mut words = Vec::new();
    for item in items {
        match item {
            Item::Words(w) => words.extend(w),
            Item::Label(_) => {}
            Item::Branch(opcode, target) => {
                let offset = (labels[target] as i64 - words.len() as i64 - 1) as u32 & 0xFFFF;
                words.push(0xBF80_0000 | opcode << 16 | offset);
            }
        }
    }
    words
}

/// s_mov_b32 s[first + i], literal[i] for the eight words of a T#.
fn s_mov_descriptor(first: u32, literal: &[u32; 8]) -> Item {
    Item::Words(literal.iter().enumerate().flat_map(|(i, w)| [0xBE80_03FF | (first + i as u32) << 16, *w]).collect())
}

const T_A: [u32; 8] = [0x1111_0000, 0x1111_0001, 0x1111_0002, 0x1111_0003, 0x1111_0004, 0x1111_0005, 0x1111_0006, 0x1111_0007];
const T_B: [u32; 8] = [0x2222_0000, 0x2222_0001, 0x2222_0002, 0x2222_0003, 0x2222_0004, 0x2222_0005, 0x2222_0006, 0x2222_0007];
const PROLOGUE: [u32; 10] = [
    0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000,
    0x7E08_0280, 0x7E0A_0280, 0x7E0C_0280, 0x7E0E_0280, // v_mov_b32 v4..v7, 0
];
/// image_store v[4:7], v0, s[20:27] dmask:0xf dim:1D, then s_endpgm.
const STORE: [u32; 3] = [0xF020_0F00, 0x0005_0400, 0xBF81_0000];

fn spirv_val() -> Option<PathBuf> {
    let file = format!("spirv-val{}", std::env::consts::EXE_SUFFIX);
    let mut dirs: Vec<PathBuf> = std::env::var_os("DECANT_SHADER_TEST_SPIRV_BIN").map(Into::into).into_iter().collect();
    dirs.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
    dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// Opcodes of the module's instructions after the header.
fn opcodes(module: &[u8]) -> Vec<u32> {
    let words: Vec<u32> = module.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    let (mut out, mut at) = (Vec::new(), 5);
    while at < words.len() {
        out.push(words[at] & 0xFFFF);
        at += ((words[at] >> 16) as usize).max(1);
    }
    out
}

/// Translates a compute shader with the probe; the opcodes of its valid module, or None when skipped.
fn translate(code: &[u32]) -> Option<Vec<u32>> {
    let probe = decant_shader_core::paths::probe();
    let (true, Some(validator)) = (probe.is_file(), spirv_val()) else {
        eprintln!("skipped: probe (DECANT_SHADER_PROBE) or spirv-val missing");
        return None;
    };
    let dir = tempfile::tempdir().unwrap();
    let (pack, log) = (dir.path().join("cs.pack"), dir.path().join("cs.tsv"));
    let name = b"CS_Synthetic";
    let mut record = (name.len() as u16).to_le_bytes().to_vec();
    record.extend(name);
    record.push(5);
    record.extend([0u8; 16]);
    record.extend((code.len() as u32).to_le_bytes());
    record.extend(code.iter().flat_map(|w| w.to_le_bytes()));
    std::fs::write(&pack, decant_extract_shaders::pack::write([record.as_slice()].into_iter())).unwrap();
    let status = Command::new(&probe).arg(&pack).arg("--stage").arg(&log).args(["0", "5"]).arg(dir.path()).arg("1").output().unwrap().status;
    assert!(status.success());
    assert!(String::from_utf8_lossy(&std::fs::read(&log).unwrap()).contains("C\t0\tOK"));
    let spv = dir.path().join("0.spv");
    assert!(Command::new(&validator).args(["--target-env", "vulkan1.2"]).arg(&spv).status().unwrap().success());
    Some(opcodes(&std::fs::read(&spv).unwrap()))
}

const OP_IMAGE_WRITE: u32 = 99;

/// A condition read from memory (s_load_dword) chooses between two literal T#s; image_store writes
/// with the merged T#. A Phi of two fixed T#s used by a write: the finite choice admits it.
#[test]
fn fixed_descriptors_chosen_by_a_merge_are_written_by_image_store() {
    let code = assemble(&[
        Item::Words(PROLOGUE.to_vec()),
        Item::Words(vec![0xF400_0400, 0xFA00_0000, 0xBF8C_C07F]), // s_load_dword s16, s[0:1], 0 ; s_waitcnt
        Item::Words(vec![0xBF06_8010]),                           // s_cmp_eq_u32 s16, 0
        Item::Branch(5, "other"),                                 // s_cbranch_scc1
        s_mov_descriptor(20, &T_A),
        Item::Branch(2, "store"),                                 // s_branch
        Item::Label("other"),
        s_mov_descriptor(20, &T_B),
        Item::Label("store"),
        Item::Words(STORE.to_vec()),
    ]);
    if let Some(list) = translate(&code) {
        assert!(list.contains(&OP_IMAGE_WRITE));
    }
}

/// A rotated loop: s[20:27] first holds lane data (v_readfirstlane of v0), and the entry jumps over the
/// image to the loop body, which loads the whole T# from the descriptor table at s[0:1] (s_load_dwordx8
/// s[20:27] at s2 << 5 or at s3 << 5, chosen by a value read from memory) before branching back to the
/// image. In the IR the image's words are Phis that also carry the lane data, but on the native CFG only
/// the two whole-T# loads reach the image: each gives its own key to the heap table.
#[test]
fn two_whole_descriptor_loads_reaching_an_image_each_give_their_table_key() {
    let code = assemble(&[
        Item::Words(PROLOGUE.to_vec()),
        Item::Words(vec![0xF400_0400, 0xFA00_0000, 0xBF8C_C07F]), // s_load_dword s16, s[0:1], 0 ; s_waitcnt
        Item::Words(vec![0x8F1E_8502, 0x8F1F_8503]),              // s_lshl_b32 s30, s2, 5 ; s_lshl_b32 s31, s3, 5
        Item::Words((0..8).map(|w| 0x7E00_0500 | (20 + w) << 17).collect()), // v_readfirstlane_b32 s20..s27, v0
        Item::Words(vec![0xBE91_0380]),                           // s_mov_b32 s17, 0
        Item::Branch(2, "load"),                                  // s_branch
        Item::Label("read"),
        Item::Words(vec![0xBF8C_0000, 0xF000_0F00, 0x0005_0400, 0xBF8C_0000]), // s_waitcnt ; image_load v[4:7], v0, s[20:27]
        s_mov_descriptor(40, &T_A),
        Item::Words(vec![0xF020_0F00, 0x000A_0400]),              // image_store v[4:7], v0, s[40:47]
        Item::Label("load"),
        Item::Words(vec![0xBF06_8010]),                           // s_cmp_eq_u32 s16, 0
        Item::Branch(5, "other"),                                 // s_cbranch_scc1
        Item::Words(vec![0xF40C_0500, 0x3C00_0000]),              // s_load_dwordx8 s[20:27], s[0:1], s30
        Item::Branch(2, "next"),                                  // s_branch
        Item::Label("other"),
        Item::Words(vec![0xF40C_0500, 0x3E00_0000]),              // s_load_dwordx8 s[20:27], s[0:1], s31
        Item::Label("next"),
        Item::Words(vec![0xBF8C_0000, 0x8011_8111, 0xBF0A_8411]), // s_waitcnt ; s_add_u32 s17, s17, 1 ; s_cmp_lt_u32 s17, 4
        Item::Branch(5, "read"),                                  // s_cbranch_scc1
        Item::Words(vec![0xBF81_0000]),                           // s_endpgm
    ]);
    if let Some(list) = translate(&code) {
        assert!(list.contains(&OP_IMAGE_WRITE));
    }
}

/// v_cmpx_ne_u32 keeps the lanes where v0 != 0; s_cbranch_execz (no lane active) jumps straight to
/// the image_store while s[20:27] still holds lane data (v_readfirstlane of v0), not a T#. Otherwise
/// s[20:27] receives a literal T#. The EXECZ arm writes nothing: it reads a null T# instead of
/// failing the translation.
#[test]
fn an_execz_arm_without_active_lane_reads_a_null_descriptor() {
    let code = assemble(&[
        Item::Words(PROLOGUE.to_vec()),
        Item::Words((0..8).map(|w| 0x7E00_0500 | (20 + w) << 17).collect()), // v_readfirstlane_b32 s20..s27, v0
        Item::Words(vec![0x7DAA_0080]),                                      // v_cmpx_ne_u32 exec, 0, v0
        Item::Branch(8, "store"),                                            // s_cbranch_execz
        s_mov_descriptor(20, &T_A),
        Item::Label("store"),
        Item::Words(STORE.to_vec()),
    ]);
    if let Some(list) = translate(&code) {
        assert!(list.contains(&OP_IMAGE_WRITE));
    }
}
