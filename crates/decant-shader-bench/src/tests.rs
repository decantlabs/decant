use std::ffi::{OsStr, OsString};

use crate::cfmt::{Fields, atoi, fixed4, strtoul0};
use crate::layout::{self, LayoutEntry};
use crate::run_args::{self, RunArgs};
use crate::{Exit, Vertex, Vkd3dOptions, d3d12core_beside, pad256, pattern};

/// Output of `printf("%.4f", f)` with the Universal CRT (MSVC 2022), for these float bit patterns.
const MSVC: &[(u32, &str)] = &[
    (0x7FC0_0000, "nan"),
    (0xFFC0_0000, "-nan(ind)"),
    (0x7F80_0001, "nan"),
    (0xFF80_0001, "-nan"),
    (0x7FC0_0001, "nan"),
    (0xFFC0_0001, "-nan"),
    (0x7FFF_FFFF, "nan"),
    (0xFFFF_FFFF, "-nan"),
    (0x7F80_0000, "inf"),
    (0xFF80_0000, "-inf"),
    (0x8000_0000, "-0.0000"),
    (0x0000_0000, "0.0000"),
    (0x3D00_0000, "0.0312"),
    (0x3D80_0000, "0.0625"),
    (0x3F00_0000, "0.5000"),
    (0x0000_0001, "0.0000"),
    (0x7F7F_FFFF, "340282346638528859811704183484516925440.0000"),
    (0xBF80_0000, "-1.0000"),
    (0x3851_B717, "0.0000"),
    (0x3F7F_FCB9, "0.9999"),
    (0x3DFC_D35B, "0.1235"),
    (0x4020_0000, "2.5000"),
    (0xB827_C5AC, "-0.0000"),
    (0x3F80_01A3, "1.0000"),
    (0x3F1E_7D56, "0.6191"),
    (0x5015_02F9, "10000000000.0000"),
    (0x391D_4952, "0.0002"),
    (0x3983_126F, "0.0003"),
];

#[test]
fn fixed4_matches_the_universal_crt() {
    for &(bits, expected) in MSVC {
        assert_eq!(fixed4(f32::from_bits(bits)), expected, "{bits:08x}");
    }
}

#[test]
fn atoi_as_the_c_runtime() {
    assert_eq!(atoi("63"), 63);
    assert_eq!(atoi("  -7x"), -7);
    assert_eq!(atoi("+12"), 12);
    assert_eq!(atoi("abc"), 0);
    assert_eq!(atoi(""), 0);
    assert_eq!(atoi("99999999999"), i32::MAX);
    assert_eq!(atoi("-99999999999"), i32::MIN);
}

#[test]
fn strtoul_base_zero() {
    assert_eq!(strtoul0("0x8000"), 0x8000);
    assert_eq!(strtoul0("0X1f"), 31);
    assert_eq!(strtoul0("010"), 8);
    assert_eq!(strtoul0("65536"), 65536);
    assert_eq!(strtoul0("-1"), u32::MAX);
    assert_eq!(strtoul0("0x"), 0);
    assert_eq!(strtoul0("0xg"), 0);
    assert_eq!(strtoul0("99999999999"), u32::MAX);
    assert_eq!(strtoul0(" 12z"), 12);
    assert_eq!(strtoul0("z"), 0);
    assert_eq!(strtoul0("09"), 0);
}

#[test]
fn fields_read_as_an_istream() {
    let mut f = Fields::new(b" 12abc -3 x\r\n 4\x1a 5");
    assert_eq!(f.unsigned(), Some(12));
    assert_eq!(f.word().as_deref(), Some("abc"));
    assert_eq!(f.signed(), Some(-3));
    assert_eq!(f.signed(), None);
    let mut f = Fields::new(b"4\x1a 5");
    assert_eq!(f.unsigned(), Some(4));
    assert_eq!(f.unsigned(), None);
    assert_eq!(Fields::new(b"-1").unsigned(), Some(u32::MAX));
    assert_eq!(Fields::new(b"4294967296").unsigned(), None);
}

fn entry(line: &str) -> LayoutEntry {
    let mut entries = layout::parse(line.as_bytes());
    assert_eq!(entries.len(), 1, "{line}");
    entries.remove(0)
}

#[test]
fn layout_lines_parse_and_unknown_counts_become_64() {
    let entries = layout::parse(b"0 3 storage_buffer 1 - 0 - 0 0 0\n1 0 sampled_image 0 2d 1 float 0 0 0\n2 1 broken\n3 3 sampler 1 - 0 - 0 0 0\n");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].kind, "storage_buffer");
    assert_eq!(entries[1].count, 64);
    assert_eq!(entries[1].view_key(), "2d[]");
    assert_eq!(
        entry("0 1 storage_image 1 3d 0 float 2 0 0").storage_key(),
        "storage 3d2"
    );
}

fn refused(line: &str, robust: bool) -> Option<Exit> {
    layout::check(&[entry(line)], robust).err()
}

#[test]
fn layout_checks_in_the_cpp_order() {
    let reason = |line: &str| refused(line, true).map(|e| (e.code, e.message));
    assert_eq!(
        reason("0 1 acceleration 1 - 0 - 0 0 0"),
        Some((
            5,
            "non gere : acceleration au jeu 0 binding 1 (genre)".into()
        ))
    );
    assert_eq!(
        reason("0 1 sampled_image 1 1d 0 float 0 0 0"),
        Some((5, "non gere : sampled_image au jeu 0 binding 1 (1d)".into()))
    );
    assert_eq!(
        reason("2 4 combined_image 1 2d 0 sint 0 0 0").unwrap().1,
        "non gere : combined_image au jeu 2 binding 4 (texels entiers)"
    );
    assert_eq!(
        reason("0 1 storage_image 1 2d 0 sint 1 0 0").unwrap().1,
        "non gere : storage_image au jeu 0 binding 1 (type des texels et format)"
    );
    assert_eq!(
        reason("0 1 storage_image 1 2d 0 float 33 0 0").unwrap().1,
        "non gere : storage_image au jeu 0 binding 1 (type des texels et format)"
    );
    assert_eq!(
        reason("0 1 sampled_image 1 2d 0 float 0 1 0").unwrap().1,
        "non gere : sampled_image au jeu 0 binding 1 (multi-echantillons)"
    );
    assert_eq!(
        reason("0 1 sampled_image 1 2d 0 float 0 0 1").unwrap().1,
        "non gere : sampled_image au jeu 0 binding 1 (comparaison de profondeur)"
    );
    assert_eq!(
        reason("0 1 storage_image 1 cube 0 float 1 0 0").unwrap().1,
        "non gere : storage_image au jeu 0 binding 1 (format d'image)"
    );
    assert_eq!(
        reason("0 1 storage_image 1 2d 0 float 40 0 0").unwrap().1,
        "non gere : storage_image au jeu 0 binding 1 (format d'image)"
    );
    assert_eq!(reason("0 1 storage_image 1 2d 0 uint 33 0 0"), None);
    assert_eq!(reason("0 1 bda_pagetable 1 - 0 - 0 0 0"), None);
    assert_eq!(
        refused("0 1 bda_pagetable 1 - 0 - 0 0 0", false)
            .unwrap()
            .message,
        "non gere : bda_pagetable au jeu 0 binding 1 (table de pages sans robustesse)"
    );
}

fn args(list: &[&str]) -> Vec<OsString> {
    list.iter().map(OsString::from).collect()
}

fn fake_read(path: &OsStr) -> Result<Vec<u8>, Exit> {
    match path.to_str() {
        Some("missing") => Err(Exit::new(2, "lecture impossible : missing")),
        Some(name) => Ok(name.as_bytes().to_vec()),
        None => unreachable!(),
    }
}

fn parse(list: &[&str]) -> Result<RunArgs, Exit> {
    run_args::parse(&args(list), fake_read)
}

#[test]
fn run_arguments_as_run_tests_passes_them() {
    let run = parse(&[
        "b",
        "vs",
        "ps",
        "o.f32",
        "--texture",
        "63",
        "111",
        "--buffer",
        "119",
        "zero.bin",
        "--push",
        "0",
        "0x8000",
    ])
    .unwrap();
    assert_eq!((run.texture_binding, run.sampler_binding), (63, 111));
    assert_eq!(run.bound[0].binding, 119);
    assert_eq!(run.bound[0].data, b"zero.bin");
    assert_eq!(run.user_data[0], 0x8000);
    assert!(run.robust);
    let run = parse(&[
        "b",
        "vs",
        "ps",
        "o",
        "--layout",
        "l.txt",
        "--no-robust",
        "--texture-file",
        "t.rgba",
        "8",
        "4",
        "--push",
        "31",
        "65536",
    ])
    .unwrap();
    assert_eq!(run.layout_path.as_deref(), Some(OsStr::new("l.txt")));
    assert!(!run.robust);
    assert_eq!(
        (run.real_texture.as_slice(), run.real_w, run.real_h),
        (b"t.rgba".as_slice(), 8, 4)
    );
    assert_eq!(run.user_data[31], 65536);
    assert_eq!(run.pipeline_cache_path, None);
}

#[test]
fn run_arguments_read_the_pipeline_cache_path() {
    let run = parse(&["b", "vs", "ps", "o", "--pipeline-cache", "cache.bin", "--no-robust"]).unwrap();
    assert_eq!(run.pipeline_cache_path.as_deref(), Some(OsStr::new("cache.bin")));
    assert!(!run.robust);
    // The path is kept as given: the file is read only when the pipeline is created.
    let missing = parse(&["b", "vs", "ps", "o", "--pipeline-cache"]).unwrap_err();
    assert_eq!(
        (missing.code, missing.message.as_str()),
        (1, "option inconnue : --pipeline-cache")
    );
}

#[test]
fn run_argument_errors_keep_their_order_and_codes() {
    assert_eq!(parse(&["b", "vs", "ps"]).unwrap_err().code, 1);
    let unknown = parse(&["b", "vs", "ps", "o", "--push", "32", "1"]).unwrap_err();
    assert_eq!(
        (unknown.code, unknown.message.as_str()),
        (1, "option inconnue : --push")
    );
    assert_eq!(
        parse(&["b", "vs", "ps", "o", "--push", "-1", "1"])
            .unwrap_err()
            .code,
        1
    );
    assert_eq!(
        parse(&["b", "vs", "ps", "o", "--buffer", "1"])
            .unwrap_err()
            .message,
        "option inconnue : --buffer"
    );
    // The file is read when its option is read, before the unknown option after it.
    assert_eq!(
        parse(&["b", "vs", "ps", "o", "--buffer", "1", "missing", "--bogus"])
            .unwrap_err()
            .code,
        2
    );
    // Too few values for --texture-file: it falls through to the unknown option.
    assert_eq!(
        parse(&["b", "vs", "ps", "o", "--texture-file", "t", "8"])
            .unwrap_err()
            .message,
        "option inconnue : --texture-file"
    );
}

#[test]
fn layout_bindings_taken_twice_are_refused() {
    let run = parse(&["b", "vs", "ps", "o", "--buffer", "3", "x", "--layout", "l"]).unwrap();
    let text = |body: &'static str| move |_: &OsStr| Some(body.as_bytes().to_vec());
    let err = run_args::load_layout(&run, text("0 3 storage_buffer 1 - 0 - 0 0 0\n")).unwrap_err();
    assert_eq!(
        (err.code, err.message.as_str()),
        (
            2,
            "binding 3 du jeu 0 donne deux fois (--layout et --buffer/--texture)"
        )
    );
    assert_eq!(
        run_args::load_layout(&run, text("1 3 storage_buffer 1 - 0 - 0 0 0\n"))
            .unwrap()
            .len(),
        1
    );
    // The checks come before the duplicate test.
    assert_eq!(
        run_args::load_layout(&run, text("0 3 what 1 - 0 - 0 0 0\n"))
            .unwrap_err()
            .code,
        5
    );
    let missing = run_args::load_layout(&run, |_| None).unwrap_err();
    assert_eq!(
        (missing.code, missing.message.as_str()),
        (2, "lecture impossible : l")
    );
}

#[test]
fn bench_inputs() {
    let texels = pattern(16 * 6);
    assert_eq!(&texels[..4], &[10, 20, 0, 255]);
    assert_eq!(&texels[15 * 4..16 * 4], &[130, 200, 225, 255]);
    assert_eq!(&texels[16 * 4..17 * 4], &[10, 20, 7, 255]);
    assert_eq!(texels[95 * 4 + 2], ((15 * 15 + 7 * 5) % 256) as u8);
    let bytes = Vertex::bytes(&[Vertex {
        position: [1.0, 2.0, 3.0],
        color: [4.0, 5.0, 6.0, 7.0],
        uv: [8.0, 9.0],
    }]);
    assert_eq!(bytes.len(), Vertex::SIZE as usize);
    assert_eq!(
        &bytes[Vertex::UV as usize..Vertex::UV as usize + 4],
        &8.0f32.to_ne_bytes()
    );
    assert_eq!(std::mem::size_of::<Vertex>(), Vertex::SIZE as usize);
}

#[test]
fn vkd3d_arguments() {
    assert_eq!(d3d12core_beside(r"D:\v\d3d12.dll"), r"D:\v\d3d12core.dll");
    assert_eq!(d3d12core_beside("x/y\\d3d12.dll"), "x/y\\d3d12core.dll");
    assert_eq!(d3d12core_beside("d3d12.dll"), "d3d12core.dll");
    let base = ["t", "d3d12.dll", "vs", "ps", "o"];
    let with = |extra: &[&str]| Vkd3dOptions::parse(&args(&[&base[..], extra].concat()));
    assert_eq!(
        with(&["--cbv", "27", "c.bin"]),
        Vkd3dOptions {
            cbv_register: 27,
            cbv_file: Some("c.bin".into()),
            srv_register: -1
        }
    );
    assert_eq!(
        with(&["--srv", "188"]),
        Vkd3dOptions {
            cbv_register: -1,
            cbv_file: None,
            srv_register: 188
        }
    );
    assert_eq!(
        with(&["--cbv", "27"]),
        Vkd3dOptions {
            cbv_register: -1,
            cbv_file: None,
            srv_register: -1
        }
    );
    assert_eq!(
        with(&[]),
        Vkd3dOptions {
            cbv_register: -1,
            cbv_file: None,
            srv_register: -1
        }
    );
    assert_eq!(pad256(vec![1; 64]).len(), 256);
    assert_eq!(pad256(vec![1; 256]).len(), 256);
    assert_eq!(pad256(Vec::new()).len(), 0);
}
