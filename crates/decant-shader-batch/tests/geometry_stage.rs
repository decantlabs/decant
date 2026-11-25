//! The probe turns a synthetic geometry shader into a valid Vulkan geometry module. Skipped when
//! the probe (`DECANT_SHADER_PROBE`, otherwise `src/shaders/decant_shader/decant_shader_probe.exe`) or spirv-val
//! (`DECANT_SHADER_TEST_SPIRV_BIN`, otherwise the `PATH`) is missing.

use std::path::PathBuf;
use std::process::Command;

/// A geometry shader assembled by hand from the RDNA2 instruction formats (no game shader): reads
/// one word of vertex 0 (v0 & 0xffff) and one of vertex 2 (v1 & 0xffff) from LDS, writes three
/// vertices of two words to the ring (address v2, 16 byte step) with a MSG_GS emit after each,
/// then cut and MSG_GS_DONE.
const SYNTHETIC_GS: &[u32] = &[
    0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0xBF80_0000,
    0x360E_00FF, 0x0000_FFFF, // v_and_b32 v7, 0xffff, v0
    0x340E_0E82, // v_lshlrev_b32 v7, 2, v7
    0x3610_02FF, 0x0000_FFFF, // v_and_b32 v8, 0xffff, v1
    0x3410_1082, // v_lshlrev_b32 v8, 2, v8
    0xD8D8_0000, 0x0400_0007, // ds_read_b32 v4, v7
    0xD8D8_0004, 0x0500_0008, // ds_read_b32 v5, v8 offset:4
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xD834_0000, 0x0000_0402, 0xD834_0004, 0x0000_0502, 0xBF90_0022, // ds_write_b32 v2, v4 and v5, s_sendmsg emit
    0xD834_0010, 0x0000_0402, 0xD834_0014, 0x0000_0502, 0xBF90_0022,
    0xD834_0020, 0x0000_0402, 0xD834_0024, 0x0000_0502, 0xBF90_0022,
    0xBF90_0012, 0xBF90_0003, 0xBF81_0000, // s_sendmsg cut, s_sendmsg gs_done, s_endpgm
];

fn spirv_val() -> Option<PathBuf> {
    let file = format!("spirv-val{}", std::env::consts::EXE_SUFFIX);
    let mut dirs: Vec<PathBuf> = std::env::var_os("DECANT_SHADER_TEST_SPIRV_BIN").map(Into::into).into_iter().collect();
    dirs.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
    dirs.into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
}

/// (opcode, operands) of every instruction after the header.
fn instructions(module: &[u8]) -> Vec<(u32, Vec<u32>)> {
    let words: Vec<u32> = module.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
    let (mut out, mut at) = (Vec::new(), 5);
    while at < words.len() {
        let (count, opcode) = ((words[at] >> 16) as usize, words[at] & 0xFFFF);
        out.push((opcode, words[at + 1..at + count.max(1)].to_vec()));
        at += count.max(1);
    }
    out
}

#[test]
fn a_synthetic_gs_becomes_a_valid_geometry_module() {
    let probe = decant_shader_core::paths::probe();
    let (true, Some(validator)) = (probe.is_file(), spirv_val()) else {
        eprintln!("skipped: probe (DECANT_SHADER_PROBE) or spirv-val missing");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (pack, log) = (dir.path().join("gs.pack"), dir.path().join("gs.tsv"));
    let name = b"GS_Synthetic";
    let mut record = (name.len() as u16).to_le_bytes().to_vec();
    record.extend(name);
    record.push(2);
    record.extend([0u8; 16]);
    record.extend((SYNTHETIC_GS.len() as u32).to_le_bytes());
    record.extend(SYNTHETIC_GS.iter().flat_map(|w| w.to_le_bytes()));
    std::fs::write(&pack, decant_extract_shaders::pack::write([record.as_slice()].into_iter())).unwrap();
    let status = Command::new(&probe).arg(&pack).arg("--stage").arg(&log).args(["0", "2"]).arg(dir.path()).arg("1").output().unwrap().status;
    assert!(status.success());
    assert!(String::from_utf8_lossy(&std::fs::read(&log).unwrap()).contains("C\t0\tOK"));
    let spv = dir.path().join("0.spv");
    assert!(Command::new(&validator).args(["--target-env", "vulkan1.2"]).arg(&spv).status().unwrap().success());
    let list = instructions(&std::fs::read(&spv).unwrap());
    // OpEntryPoint Geometry.
    assert!(list.iter().any(|(op, o)| *op == 15 && o.first() == Some(&3)));
    // OpExecutionMode: Triangles, OutputTriangleStrip, OutputVertices 3, Invocations 1.
    let modes: Vec<&[u32]> = list.iter().filter(|(op, _)| *op == 16).map(|(_, o)| &o[1..]).collect();
    for wanted in [&[22][..], &[29], &[26, 3], &[0, 1]] {
        assert!(modes.contains(&wanted), "execution mode {wanted:?}");
    }
    // OpEmitVertex three times, OpEndPrimitive once.
    let count = |code| list.iter().filter(|(op, _)| *op == code).count();
    assert_eq!((count(218), count(219)), (3, 1));
}
