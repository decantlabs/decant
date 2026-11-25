//! The steps against a recording runner: the commands each step runs (program, arguments,
//! redirections, exported variables), the files it writes and every line it prints, compared
//! with what `launch_prep.sh`, `run_airgapped.sh` and `test_isolation.sh` run and print.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::rc::Rc;

use crate::ctx::{Cmd, Ctx, Exit, Outcome, Runner, Sink};
use crate::prep::{self, CONTAINER_SMOKE, DLL_OVERRIDES, Prep};
use crate::sandbox;

#[derive(Clone, Default)]
struct Buf(Rc<RefCell<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buf {
    fn text(&self) -> String {
        String::from_utf8(self.0.borrow().clone()).unwrap()
    }
}

type Respond = Box<dyn FnMut(&Cmd) -> Outcome>;

/// Records each command, answers with `respond`, and writes the answer's stdout where the
/// command's stdout goes.
struct Recorder {
    log: Rc<RefCell<Vec<Cmd>>>,
    respond: Respond,
    found: Vec<&'static str>,
}

impl Runner for Recorder {
    fn run(&mut self, cmd: &Cmd, echo: &mut dyn Write) -> Outcome {
        self.log.borrow_mut().push(cmd.clone());
        let out = (self.respond)(cmd);
        match &cmd.stdout {
            Sink::File(path, append) => {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(*append)
                    .write(true)
                    .truncate(!append)
                    .open(path)
                    .unwrap();
                file.write_all(out.stdout.as_bytes()).unwrap();
            }
            Sink::Tee(path) => {
                std::fs::write(path, &out.stdout).unwrap();
                echo.write_all(out.stdout.as_bytes()).unwrap();
            }
            Sink::Capture => return out,
            _ => echo.write_all(out.stdout.as_bytes()).unwrap(),
        }
        Outcome { code: out.code, stdout: String::new() }
    }

    fn find(&self, program: &str) -> bool {
        self.found.contains(&program)
    }
}

struct Run {
    ctx: Ctx,
    log: Rc<RefCell<Vec<Cmd>>>,
    out: Buf,
    err: Buf,
}

const PROG: &str = "src/gdk/launch_prep.sh";

fn run_with(env: &[(&str, &str)], found: Vec<&'static str>, respond: Respond) -> Run {
    let base: BTreeMap<String, String> = env.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut ctx = Ctx::new(PROG.to_owned(), base, Box::new(Recorder { log: log.clone(), respond, found }));
    let (out, err) = (Buf::default(), Buf::default());
    ctx.out = Box::new(out.clone());
    ctx.err = Box::new(err.clone());
    ctx.stamp = Some("20261121-120000".to_owned());
    Run { ctx, log, out, err }
}

fn ok() -> Respond {
    Box::new(|_| Outcome::default())
}

/// A scratch checkout: `root` with forward slashes, as the script's paths are.
fn root() -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().replace('\\', "/");
    (dir, path)
}

fn write(path: &str, data: &[u8]) {
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
    std::fs::write(path, data).unwrap();
}

fn say(text: &str) -> String {
    format!("\x1b[1m:: {text}\x1b[0m\n")
}
fn warn(text: &str) -> String {
    format!("\x1b[33m** {text}\x1b[0m\n")
}
fn die(text: &str) -> String {
    format!("\x1b[31m!! {text}\x1b[0m\n")
}

/// The argument vectors run, program first.
fn argvs(log: &Rc<RefCell<Vec<Cmd>>>) -> Vec<Vec<String>> {
    log.borrow().iter().map(|c| std::iter::once(c.program.clone()).chain(c.args.iter().cloned()).collect()).collect()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// A synthetic config: a dev only tool listed first, then the title (invented names).
const CONFIG: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Contoso.SyntheticTitle" Publisher="CN=Contoso" Version="1.0.0.0"/>
  <ExecutableList>
    <Executable Name="DevTool.exe" TargetDeviceFamily="Scarlett" IsDevOnly="true"/>
    <Executable Name="Title.exe" TargetDeviceFamily="Scarlett"/>
  </ExecutableList>
</Game>
"#;

/// The smallest header that passes the PE pre-flight: x86-64, PE32+, GUI subsystem.
fn plain_pe() -> Vec<u8> {
    let mut image = vec![0u8; 0x200];
    image[..2].copy_from_slice(b"MZ");
    image[0x3c] = 0x40;
    image[0x40..0x44].copy_from_slice(b"PE\0\0");
    image[0x44..0x46].copy_from_slice(&0x8664u16.to_le_bytes());
    image[0x54..0x56].copy_from_slice(&240u16.to_le_bytes());
    image[0x58..0x5a].copy_from_slice(&0x20bu16.to_le_bytes());
    image[0x58 + 68] = 2;
    image
}

fn code(result: Result<(), Exit>) -> i32 {
    result.err().map_or(0, |Exit(c)| c)
}

#[test]
fn unknown_command_prints_the_usage_and_exits_2() {
    let mut r = run_with(&[], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["bogus"])), 2);
    assert_eq!(r.out.text(), prep::USAGE);
    assert_eq!(r.out.text().lines().count(), 18, "lines 2 to 19 of the script's header");
}

#[test]
fn status_reports_everything_then_fails() {
    let (_keep, root) = root();
    let mut r = run_with(&[("DECANT_ROOT", &root)], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &[]), 1, "status is the default command");
    let base = format!("{root}/runtime");
    let expected = [
        "config reader          built in (gdk-config)".to_owned(),
        format!("runtime archive        MISSING (run: {PROG} runtime)"),
        format!("runtime                MISSING (run: {PROG} runtime)"),
        "shims                  xmem.dll(MISSING) d3d12_xs.dll(MISSING)".to_owned(),
        format!("prefix                 MISSING (run: {PROG} prefix)"),
        format!("game dir               MISSING (run: {PROG} stage)"),
        format!("staged config          MISSING (run: {PROG} stage)"),
        "executable             MISSING: no primary executable (config unreadable; set DECANT_EXE_NAME)".to_owned(),
        "launch mode            air-gapped container (default; host opt-out: DECANT_HOST_LAUNCH=1)".to_owned(),
    ];
    assert_eq!(r.out.text(), expected.map(|l| l + "\n").concat());
    assert_eq!(r.err.text(), die("status: missing components above"), "the reader's own error stays hidden");
    assert!(!std::path::Path::new(&base).exists(), "status creates nothing");
}

#[test]
fn status_all_present_with_host_launch() {
    let (_keep, root) = root();
    let archive = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1.tar.xz");
    write(&archive, b"abc");
    let rt = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1");
    write(&format!("{rt}/proton"), b"#!/bin/sh\n");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(format!("{rt}/proton"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    write(&format!("{rt}/version"), b"11.0-test\n");
    for f in ["xmem.dll", "d3d12_xs.dll"] {
        write(&format!("{root}/build/gdk/{f}"), b"x");
    }
    write(&format!("{root}/runtime/prefix/pfx/system.reg"), b"x");
    let game = format!("{root}/install/title");
    write(&format!("{game}/MicrosoftGame.Config"), CONFIG.as_bytes());
    write(&format!("{game}/Title.exe"), &plain_pe());
    write(&format!("{root}/extracted/package/MicrosoftGame.Config"), CONFIG.as_bytes());
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let env = [("DECANT_ROOT", root.as_str()), ("RUNTIME_SHA256", abc), ("DECANT_HOST_LAUNCH", "1")];
    let mut r = run_with(&env, vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["status"])), 0);
    let out = r.out.text();
    assert!(out.contains(&format!("runtime archive        {archive} (present, hash OK)\n")), "{out}");
    assert!(out.contains(&format!("runtime                {rt} (11.0-test)\n")));
    assert!(out.contains("shims                  xmem.dll d3d12_xs.dll\n"));
    assert!(out.contains(&format!("game dir               {game} (2 entries)\n")));
    assert!(out.contains(&format!("executable             {game}/Title.exe\n")), "the dev only tool is skipped");
    assert!(
        out.contains("launch mode            HOST (DECANT_HOST_LAUNCH=1: package code on this host, with network)\n")
    );
    assert!(out.ends_with(&say("status: all components present")));
    assert_eq!(r.err.text(), warn("host launch enabled; the air-gapped container is the project default"));
}

#[test]
fn env_prints_the_launch_environment_sorted() {
    let env = [
        ("DECANT_ROOT", "/repo"),
        ("WINEDEBUG", "-all"),
        ("WINEPREFIX", "/home/u/.wine"),
        ("DECANT_PACKAGE_DIR", "/pkg"),
        ("PATH", "/usr/bin"),
    ];
    let mut r = run_with(&env, vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["env"])), 0);
    let expected = [
        "DECANT_INSTALL_DIR=/repo/install/title",
        "DECANT_LOG=1",
        "DECANT_PACKAGE_DIR=/pkg",
        "DECANT_ROOT=/repo",
        "PROTON_LOG=1",
        "PROTON_LOG_DIR=/repo/logs",
        "STEAM_COMPAT_CLIENT_INSTALL_PATH=/repo/runtime/fake-steam",
        "STEAM_COMPAT_DATA_PATH=/repo/runtime/prefix",
        "VKD3D_CONFIG=dxr",
        "VKD3D_FEATURE_LEVEL=12_2",
        "WINEDEBUG=-all",
        &format!("WINEDLLOVERRIDES={DLL_OVERRIDES}"),
        "WINE_GAMEINPUT=1",
    ];
    assert_eq!(r.out.text(), expected.map(|l| format!("{l}\n")).concat());
}

#[test]
fn sha256_matches_sha256sum() {
    let (_keep, root) = root();
    write(&format!("{root}/f"), b"abc");
    assert_eq!(
        prep::sha256_file(&format!("{root}/f")).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(prep::RUNTIME_SHA256_PIN, "b5af9be3a87e025eba5c941e11dac1e51c42923270b123eb06e7c9ff1016f83d");
}

#[test]
fn runtime_downloads_verifies_and_extracts() {
    let (_keep, root) = root();
    let archive = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1.tar.xz");
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let fetched = archive.clone();
    let respond: Respond = Box::new(move |cmd| {
        if cmd.program == "curl" {
            write(&fetched, b"abc");
        }
        Outcome::default()
    });
    let mut r = run_with(&[("DECANT_ROOT", &root), ("RUNTIME_SHA256", abc)], vec![], respond);
    assert_eq!(prep::main(&mut r.ctx, &strings(&["runtime"])), 1, "the fake tar unpacks nothing");
    let url = "https://github.com/icex/ferestre/releases/download/runtime-11.0.20261107.1";
    assert_eq!(
        argvs(&r.log),
        [
            strings(&["curl", "-fSL", "-o", &archive, &format!("{url}/ferestre-runtime-11.0.20261107.1.tar.xz")]),
            strings(&["tar", "-C", &format!("{root}/runtime"), "-xJf", &archive]),
        ]
    );
    assert_eq!(
        r.out.text(),
        say("downloading ferestre-runtime-11.0.20261107.1 (~300 MB)")
            + &say(&format!("runtime archive hash OK ({abc})"))
    );
    assert_eq!(r.err.text(), die("runtime archive did not contain ferestre-runtime-11.0.20261107.1/proton"));
}

#[test]
fn runtime_refuses_a_wrong_archive_with_the_pinned_digest() {
    let (_keep, root) = root();
    let archive = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1.tar.xz");
    write(&archive, b"abc");
    let mut r = run_with(&[("DECANT_ROOT", &root), ("RUNTIME_SHA256", "")], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["runtime"])), 1);
    assert!(r.log.borrow().is_empty(), "nothing downloaded or extracted");
    let got = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        r.err.text(),
        die(&format!("runtime archive hash mismatch: got {got}, pinned {}", prep::RUNTIME_SHA256_PIN))
    );
}

#[test]
fn runtime_minisign_and_no_anchor() {
    let (_keep, root) = root();
    let archive = format!("{root}/a.tar.xz");
    write(&archive, b"abc");
    let mut r = run_with(&[("RUNTIME_MINISIGN_PUB", "/keys/ferestre.pub")], vec!["minisign"], ok());
    let mut p = Prep::new(&mut r.ctx);
    p.cfg.runtime_sha256.clear();
    assert_eq!(code(p.verify_runtime_archive(&archive)), 0);
    let url = "https://github.com/icex/ferestre/releases/download/runtime-11.0.20261107.1";
    assert_eq!(
        argvs(&r.log),
        [
            strings(&["curl", "-fSL", "-o", &format!("{archive}.minisig"), &format!("{url}/a.tar.xz.minisig")]),
            strings(&["minisign", "-Vm", &archive, "-p", "/keys/ferestre.pub"]),
        ]
    );
    assert_eq!(r.out.text(), say("runtime minisig OK (key: /keys/ferestre.pub)"));

    let mut r = run_with(&[("RUNTIME_MINISIGN_PUB", "/k")], vec![], ok());
    let mut p = Prep::new(&mut r.ctx);
    p.cfg.runtime_sha256.clear();
    assert_eq!(code(p.verify_runtime_archive(&archive)), 1);
    assert_eq!(r.err.text(), die("minisign is required to verify the runtime against /k"));

    let mut r = run_with(&[], vec![], ok());
    let mut p = Prep::new(&mut r.ctx);
    p.cfg.runtime_sha256.clear();
    assert_eq!(code(p.verify_runtime_archive(&archive)), 1);
    assert_eq!(
        r.err.text(),
        die("no pinned trust anchor for ferestre-runtime-11.0.20261107.1; refusing to extract. Verify the release digest out-of-band, then either:
    - set RUNTIME_SHA256=<digest> (and commit it as RUNTIME_SHA256_PIN in this script), or
    - point RUNTIME_MINISIGN_PUB at the release's minisign public key (minisign required).")
    );
}

#[test]
fn stubs_runs_make_and_lists_the_shims() {
    let (_keep, root) = root();
    for f in ["xmem.dll", "d3d12_xs.dll", "loadtest.exe"] {
        write(&format!("{root}/build/gdk/{f}"), b"x");
    }
    let mut r = run_with(&[("DECANT_ROOT", &root)], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stubs"])), 1);
    assert_eq!(r.err.text(), die("need mingw-w64 (x86_64-w64-mingw32-gcc)"));

    let mut r = run_with(&[("DECANT_ROOT", &root)], vec!["x86_64-w64-mingw32-gcc"], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stubs"])), 0);
    let make = &r.log.borrow()[0];
    let out = format!("OUT={root}/build/gdk");
    assert_eq!(make.args, strings(&["-C", &format!("{root}/src/gdk"), &out, "all", "tests"]));
    assert_eq!((&make.stdout, &make.stderr), (&Sink::Null, &Sink::Inherit));
    assert_eq!(r.out.text(), say("shims built: d3d12_xs.dll xmem.dll") + &say("config reader: built in (gdk-config)"));

    let mut r = run_with(
        &[("DECANT_ROOT", &root)],
        vec!["x86_64-w64-mingw32-gcc"],
        Box::new(|_| Outcome { code: 2, stdout: String::new() }),
    );
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stubs"])), 2, "make's own exit code, as set -e");
}

#[test]
fn prefix_runs_wineboot_and_regsvr32_in_the_launch_environment() {
    let (_keep, root) = root();
    let rt = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1");
    write(&format!("{rt}/proton"), b"x");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(format!("{rt}/proton"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let respond: Respond = Box::new(|cmd| Outcome {
        code: if cmd.args.contains(&"wineboot".to_owned()) { 5 } else { 0 },
        stdout: String::new(),
    });
    let mut r = run_with(&[("DECANT_ROOT", &root), ("WINEPREFIX", "/x")], vec![], respond);
    assert_eq!(prep::main(&mut r.ctx, &strings(&["prefix"])), 0);
    let log = r.log.borrow();
    let logfile = format!("{root}/logs/prefix-init.log");
    assert_eq!(log[0].program, format!("{rt}/proton"));
    assert_eq!(log[0].args, strings(&["run", "wineboot", "-u"]));
    assert_eq!((&log[0].stdout, &log[0].stderr), (&Sink::File(logfile.clone(), false), &Sink::Stdout));
    assert!(log[0].env.contains(&("DISPLAY".to_owned(), Some(String::new()))), "DISPLAY=''");
    assert!(log[0].env.contains(&("WINEPREFIX".to_owned(), None)), "launch_env unsets WINEPREFIX");
    assert!(log[0].env.contains(&("STEAM_COMPAT_DATA_PATH".to_owned(), Some(format!("{root}/runtime/prefix")))));
    assert_eq!(log[1].args, strings(&["run", "regsvr32", "/s", "xgameruntime.dll"]));
    assert_eq!(log[1].stdout, Sink::File(logfile.clone(), true));
    assert!(!log[1].env.iter().any(|(k, _)| k == "DISPLAY"));
    assert_eq!(
        r.err.text(),
        warn(&format!("wineboot returned 5 (see {logfile})")) + &warn("vkd3d-proton d3d12.dll not in prefix yet")
    );
    assert_eq!(
        r.out.text(),
        say(&format!("creating Proton prefix at {root}/runtime/prefix"))
            + &say(&format!("prefix ready: {root}/runtime/prefix/pfx"))
    );
}

/// A payload with the config, a manifest, two companion DLLs in both cases, a data folder and a
/// hidden file, and the two shims.
fn payload(root: &str) -> String {
    let src = format!("{root}/pkg");
    write(&format!("{src}/MicrosoftGame.Config"), CONFIG.as_bytes());
    write(&format!("{src}/appxmanifest.xml"), b"<Package/>");
    write(&format!("{src}/Companion.DLL"), b"c");
    write(&format!("{src}/other.dll"), b"o");
    write(&format!("{src}/.hidden"), b"h");
    for f in ["xmem.dll", "d3d12_xs.dll"] {
        write(&format!("{root}/build/gdk/{f}"), f.as_bytes());
    }
    src
}

#[test]
fn stage_copies_metadata_dlls_and_shims_then_links_the_rest() {
    let (_keep, root) = root();
    let src = payload(&root);
    write(&format!("{src}/data/archive.bin"), b"d");
    let game = format!("{root}/install/title");
    let mut r = run_with(&[("DECANT_ROOT", &root), ("DECANT_PACKAGE_DIR", &src)], vec![], ok());
    r.ctx.dry_run = true;
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stage"])), 0);
    // Paths go through the same quoting as the program: a temp dir such as `RUNNER~1` (the CI
    // runner's 8.3 short name) contains `~`, which gets quoted, while a local temp dir may not.
    let q = |p: String| crate::ctx::quote(&p);
    let expected = [
        format!("mkdir -p {}", q(game.clone())),
        format!("cp -f {} {}", q(format!("{src}/MicrosoftGame.Config")), q(format!("{game}/MicrosoftGame.Config"))),
        format!("cp -f {} {}", q(format!("{src}/appxmanifest.xml")), q(format!("{game}/appxmanifest.xml"))),
        format!("cp -f {} {}", q(format!("{src}/Companion.DLL")), q(format!("{game}/Companion.DLL"))),
        format!("cp -f {} {}", q(format!("{src}/other.dll")), q(format!("{game}/other.dll"))),
        format!("cp -f {} {}", q(format!("{root}/build/gdk/xmem.dll")), q(format!("{game}/xmem.dll"))),
        format!("cp -f {} {}", q(format!("{root}/build/gdk/d3d12_xs.dll")), q(format!("{game}/d3d12_xs.dll"))),
        format!("ln -sfn {} {}", q(format!("{src}/data")), q(format!("{game}/data"))),
        format!(
            "printf %s 'Place the title executable here as Title.exe, then run: {PROG} launch\n' >{}",
            q(format!("{game}/Title.exe.PLACEHOLDER.txt"))
        ),
    ];
    assert_eq!(r.err.text(), expected.map(|l| format!("[dry-run] {l}\n")).concat());
    assert_eq!(
        r.out.text(),
        say("copied 2 companion DLL(s) app-local") + &say(&format!("staged {game} (exe expected: Title.exe)"))
    );
    assert!(!std::path::Path::new(&game).exists(), "a dry run changes nothing");
}

#[test]
fn stage_on_disk() {
    let (_keep, root) = root();
    let src = payload(&root);
    std::fs::remove_file(format!("{src}/appxmanifest.xml")).unwrap();
    let game = format!("{root}/install/title");
    let mut r = run_with(&[("DECANT_ROOT", &root), ("DECANT_PACKAGE_DIR", &src)], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stage"])), 0);
    assert_eq!(r.err.text(), warn("missing payload file: appxmanifest.xml"));
    assert_eq!(std::fs::read(format!("{game}/xmem.dll")).unwrap(), b"xmem.dll");
    assert_eq!(std::fs::read(format!("{game}/Companion.DLL")).unwrap(), b"c");
    let placeholder = std::fs::read_to_string(format!("{game}/Title.exe.PLACEHOLDER.txt")).unwrap();
    assert_eq!(placeholder, format!("Place the title executable here as Title.exe, then run: {PROG} launch\n"));
    assert!(!std::path::Path::new(&format!("{game}/.hidden")).exists());

    // Again with the title in place: no placeholder, files replaced.
    write(&format!("{game}/Title.exe"), &plain_pe());
    std::fs::remove_file(format!("{game}/Title.exe.PLACEHOLDER.txt")).unwrap();
    let mut r = run_with(&[("DECANT_ROOT", &root), ("DECANT_PACKAGE_DIR", &src)], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stage"])), 0);
    assert!(!std::path::Path::new(&format!("{game}/Title.exe.PLACEHOLDER.txt")).exists());

    // Without shims, or without a payload.
    let mut r = run_with(&[("DECANT_ROOT", &root), ("DECANT_PACKAGE_DIR", &src), ("STUBS_DIR", "/none")], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stage"])), 1);
    assert_eq!(r.err.text(), die(&format!("shims not built; run: {PROG} stubs")));
    let mut r = run_with(&[("DECANT_ROOT", &root)], vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["stage"])), 1);
    assert_eq!(r.err.text(), die(&format!("no package payload at {root}/extracted/package (set DECANT_PACKAGE_DIR)")));
}

/// A checkout ready for `test` and `launch`: prefix, smoke test, compose file, staged title.
fn ready(root: &str) -> String {
    let src = payload(root);
    write(&format!("{root}/runtime/prefix/pfx/system.reg"), b"x");
    write(&format!("{root}/build/gdk/gdk_smoketest.exe"), b"x");
    write(&format!("{root}/sandbox/docker-compose.yml"), b"services: {}\n");
    write(&format!("{root}/install/title/Title.exe"), &plain_pe());
    src
}

/// The docker side of an isolated sandbox: only `lo`, every probe blocked, and the smoke test
/// passing.
fn isolated_docker(smoke: &'static str) -> Respond {
    Box::new(move |cmd| {
        let line = cmd.args.join(" ");
        if line.ends_with("ls /sys/class/net") {
            return Outcome { code: 0, stdout: "lo\n".to_owned() };
        }
        if line.contains("/dev/tcp/") || line.contains("google.com") {
            return Outcome { code: 1, stdout: String::new() };
        }
        if line.contains("gdk_smoketest.exe") && line.starts_with("exec") {
            return Outcome { code: 0, stdout: smoke.to_owned() };
        }
        if cmd.program == "id" {
            return Outcome { code: 0, stdout: "1000\n".to_owned() };
        }
        Outcome::default()
    })
}

/// The commands of `run_airgapped.sh` for `dir`, as argument vectors.
fn airgapped_commands(dir: &str) -> Vec<Vec<String>> {
    let _ = dir;
    vec![
        strings(&["id", "-u"]),
        strings(&["id", "-g"]),
        strings(&["docker", "compose", "build", "--quiet"]),
        strings(&["docker", "compose", "up", "-d"]),
        strings(&["docker", "exec", "decant-sandbox", "ls", "/sys/class/net"]),
        strings(&["docker", "exec", "decant-sandbox", "timeout", "3", "bash", "-c", "echo > /dev/tcp/1.1.1.1/53"]),
        strings(&["docker", "exec", "decant-sandbox", "curl", "-s", "--max-time", "2", "https://google.com"]),
    ]
}

#[test]
fn test_runs_the_smoke_test_in_the_container() {
    let (_keep, root) = root();
    let src = ready(&root);
    let smoke = "RESULT OK   load xmem.dll                                    err=0\r\nnoise\r\nSUMMARY failures=0\r\n";
    let env = [("DECANT_ROOT", root.as_str()), ("DECANT_PACKAGE_DIR", src.as_str())];
    let mut r = run_with(&env, vec!["docker"], isolated_docker(smoke));
    assert_eq!(prep::main(&mut r.ctx, &strings(&["test"])), 0);
    let mut expected = airgapped_commands(&format!("{root}/sandbox"));
    expected.push(strings(&[
        "docker",
        "cp",
        &format!("{root}/build/gdk/gdk_smoketest.exe"),
        "decant-sandbox:/tmp/gdk_smoketest.exe",
    ]));
    expected.push(strings(&[
        "docker",
        "exec",
        "-e",
        &format!("WINEDLLOVERRIDES={DLL_OVERRIDES}"),
        "-e",
        "WINEDEBUG=err+module",
        "decant-sandbox",
        "bash",
        "-c",
        CONTAINER_SMOKE,
    ]));
    assert_eq!(argvs(&r.log), expected);
    let log = r.log.borrow();
    let sandbox = format!("{root}/sandbox");
    assert!(
        log[2..7].iter().all(|c| c.cwd.as_deref() == Some(sandbox.as_str())),
        "compose runs from the sandbox folder"
    );
    assert!(log[2].env.contains(&("DECANT_PACKAGE_DIR".to_owned(), Some(src.clone()))));
    assert!(log[2].env.contains(&("DECANT_UID".to_owned(), Some("1000".to_owned()))));
    assert!(!log[8].env.iter().any(|(k, _)| k == "DECANT_UID"), "the sandbox script's exports end with it");
    let smoke_log = format!("{root}/logs/smoketest.log");
    assert_eq!(log[8].stdout, Sink::File(smoke_log.clone(), false));
    let out = r.out.text();
    assert!(out.contains("=== DECANT AIR-GAPPED SANDBOX ===\n"));
    assert!(out.contains(&format!("Stop: docker compose -f {sandbox}/docker-compose.yml down\n")));
    assert!(out.ends_with(
        &(say(&format!("smoke test inside the air-gapped container (log: {smoke_log})"))
            + "RESULT OK   load xmem.dll                                    err=0\nSUMMARY failures=0\n")
    ));
}

#[test]
fn test_fails_on_smoke_failures_and_without_docker() {
    let (_keep, root) = root();
    let src = ready(&root);
    let env = [("DECANT_ROOT", root.as_str()), ("DECANT_PACKAGE_DIR", src.as_str())];
    let mut r = run_with(&env, vec!["docker"], isolated_docker("SUMMARY failures=2\r\n"));
    assert_eq!(prep::main(&mut r.ctx, &strings(&["test"])), 1);
    assert_eq!(r.err.text(), die(&format!("smoke test reported failures (see {root}/logs/smoketest.log)")));

    let mut r = run_with(&env, vec![], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["test"])), 1);
    assert_eq!(
        r.err.text(),
        die(
            "the smoke test loads package DLLs; default is the air-gapped container (docker), or DECANT_HOST_LAUNCH=1 to run it on this host — against the project rules"
        )
    );
}

#[test]
fn host_test_runs_wine_in_a_scratch_folder() {
    let (_keep, root) = root();
    let src = ready(&root);
    write(&format!("{root}/install/title/xmem.dll"), b"x");
    let tmp = format!("{root}/tmp");
    std::fs::create_dir_all(&tmp).unwrap();
    let env = [
        ("DECANT_ROOT", root.as_str()),
        ("DECANT_PACKAGE_DIR", src.as_str()),
        ("DECANT_HOST_LAUNCH", "1"),
        ("TMPDIR", tmp.as_str()),
    ];
    let mut r = run_with(&env, vec![], Box::new(|_| Outcome { code: 3, stdout: "SUMMARY failures=0\r\n".to_owned() }));
    assert_eq!(prep::main(&mut r.ctx, &strings(&["test"])), 0, "wine's own exit code is ignored (|| true)");
    let log = r.log.borrow();
    assert_eq!(log.len(), 1);
    let wine = &log[0];
    assert_eq!(wine.program, format!("{root}/runtime/ferestre-runtime-11.0.20261107.1/files/bin/wine"));
    assert_eq!(wine.args, strings(&["gdk_smoketest.exe"]));
    let work = wine.cwd.clone().unwrap();
    assert!(work.starts_with(&format!("{tmp}/decant-prep.")) && work.ends_with("/smoke"), "{work}");
    assert!(!std::path::Path::new(&work).exists(), "the scratch folder is gone after the run");
    for (k, v) in [("DISPLAY", ""), ("WINEPREFIX", &format!("{root}/runtime/prefix/pfx")), ("WINEDEBUG", "err+module")]
    {
        assert_eq!(wine.env.iter().rev().find(|(n, _)| n == k).unwrap().1.as_deref(), Some(v));
    }
    assert_eq!(
        r.err.text(),
        warn("DECANT_HOST_LAUNCH=1: running package DLLs on this host, with network access")
            + &warn(
                "this contradicts the project rules (see src/gdk/README.md); the air-gapped container is the default"
            )
    );
}

#[test]
fn launch_in_the_container() {
    let (_keep, root) = root();
    let src = ready(&root);
    let env = [("DECANT_ROOT", root.as_str()), ("DECANT_PACKAGE_DIR", src.as_str()), ("XGR_XUID", "123")];
    let respond = isolated_docker("");
    let mut r = run_with(&env, vec!["docker"], respond);
    assert_eq!(prep::main(&mut r.ctx, &strings(&["launch"])), 0);
    let exe = format!("{root}/install/title/Title.exe");
    let mut expected = airgapped_commands(&format!("{root}/sandbox"));
    expected.push(strings(&["docker", "cp", &exe, "decant-sandbox:/tmp/Title.exe"]));
    expected.push(strings(&[
        "docker",
        "exec",
        "-e",
        &format!("WINEDLLOVERRIDES={DLL_OVERRIDES}"),
        "-e",
        "WINE_GAMEINPUT=1",
        "-e",
        "WINEDEBUG=+debugstr,err+module,+loaddll,+xgameruntime",
        "-e",
        "DECANT_LOG=1",
        "-e",
        "XGR_XUID=123",
        "decant-sandbox",
        "bash",
        "-c",
        "cd /data/game && exec wine64 /tmp/Title.exe",
    ]));
    assert_eq!(argvs(&r.log), expected);
    let log = format!("{root}/logs/launch-20261121-120000.log");
    assert_eq!(r.log.borrow()[8].stdout, Sink::Tee(log.clone()));
    let out = r.out.text();
    assert!(out.starts_with(
        &(say(&format!("PE pre-flight: {exe}")) + &format!("ok: {exe} is a plain x86-64 PE (GUI/CUI subsystem)\n"))
    ));
    assert!(out.ends_with(
        &(say(&format!("launching Title.exe inside the air-gapped container (log: {log})"))
            + &say(&format!("container launch finished (log: {log})"))
            + &say(&format!("stop the sandbox: docker compose -f {root}/sandbox/docker-compose.yml down")))
    ));
    assert_eq!(
        r.err.text(),
        warn(
            "container note: distribution Wine with its own prefix (sandbox/prefix); Ferestre's GDK runtime is wired for the host path only so far"
        )
    );
}

#[test]
fn launch_exit_codes() {
    let (_keep, root) = root();
    let src = ready(&root);
    let env = [("DECANT_ROOT", root.as_str()), ("DECANT_PACKAGE_DIR", src.as_str())];
    // The title's exit code is the launch's.
    let mut docker = isolated_docker("");
    let respond: Respond = Box::new(move |cmd| {
        if cmd.args.last().is_some_and(|a| a.contains("exec wine64 /tmp/Title.exe")) {
            return Outcome { code: 42, stdout: String::new() };
        }
        docker(cmd)
    });
    let mut r = run_with(&env, vec!["docker"], respond);
    assert_eq!(prep::main(&mut r.ctx, &strings(&["launch"])), 42);
    let log = format!("{root}/logs/launch-20261121-120000.log");
    assert!(r.err.text().ends_with(&warn(&format!("container launch exited 42 (log: {log})"))));

    // A file that is not a plain PE never runs.
    write(&format!("{root}/install/title/Title.exe"), b"MZ not really");
    let mut r = run_with(&env, vec!["docker"], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["launch"])), 1);
    assert!(r.log.borrow().is_empty());
    assert_eq!(
        r.err.text(),
        format!("{root}/install/title/Title.exe: truncated: 13 bytes is too small for a DOS header\n")
            + &die("executable failed the PE pre-flight check")
    );

    // No title staged.
    std::fs::remove_file(format!("{root}/install/title/Title.exe")).unwrap();
    let mut r = run_with(&env, vec!["docker"], ok());
    assert_eq!(prep::main(&mut r.ctx, &strings(&["launch"])), 1);
    assert_eq!(
        r.err.text(),
        die(&format!("title executable not present: {root}/install/title/Title.exe (placeholder only)"))
    );
}

#[test]
fn host_launch_tees_proton() {
    let (_keep, root) = root();
    let src = ready(&root);
    let rt = format!("{root}/runtime/ferestre-runtime-11.0.20261107.1");
    write(&format!("{rt}/proton"), b"x");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(format!("{rt}/proton"), std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let env = [("DECANT_ROOT", root.as_str()), ("DECANT_PACKAGE_DIR", src.as_str()), ("DECANT_HOST_LAUNCH", "1")];
    let mut r = run_with(&env, vec![], Box::new(|_| Outcome { code: 7, stdout: "title output\n".to_owned() }));
    assert_eq!(prep::main(&mut r.ctx, &strings(&["launch"])), 7, "proton's code through the pipe (pipefail)");
    let exe = format!("{root}/install/title/Title.exe");
    let log = format!("{root}/logs/launch-20261121-120000.log");
    let cmd = &r.log.borrow()[0];
    assert_eq!(
        (cmd.program.as_str(), cmd.args.clone()),
        (format!("{rt}/proton").as_str(), strings(&["waitforexitandrun", &exe]))
    );
    assert_eq!((&cmd.stdout, &cmd.stderr), (&Sink::Tee(log.clone()), &Sink::Stdout));
    assert_eq!(std::fs::read_to_string(&log).unwrap(), "title output\n");
    assert!(r.out.text().ends_with(&(say(&format!("launching {exe} on the host (log: {log})")) + "title output\n")));
}

#[test]
fn sandbox_aborts_when_not_isolated() {
    let (_keep, root) = root();
    let dir = format!("{root}/sandbox");
    let respond: Respond = Box::new(|cmd| {
        if cmd.args.join(" ").ends_with("ls /sys/class/net") {
            return Outcome { code: 0, stdout: "eth0\nlo\n".to_owned() };
        }
        Outcome::default()
    });
    let mut r = run_with(&[("DECANT_PACKAGE_DIR", "/pkg"), ("DECANT_UID", "1"), ("DECANT_GID", "2")], vec![], respond);
    r.ctx.prog = "sandbox/run_airgapped.sh".to_owned();
    assert_eq!(code(sandbox::run_airgapped(&mut r.ctx, &dir)), 1);
    assert_eq!(argvs(&r.log).last().unwrap(), &strings(&["docker", "compose", "down"]));
    assert!(r.out.text().ends_with(
        "[3/4] Verifying network isolation...\nFATAL: container has a network interface besides lo! Aborting.\n"
    ));
    for sub in ["prefix", "logs", "../build/gdk"] {
        assert!(std::path::Path::new(&format!("{dir}/{sub}")).is_dir());
    }

    let mut r = run_with(&[], vec![], ok());
    r.ctx.prog = "sandbox/run_airgapped.sh".to_owned();
    assert_eq!(code(sandbox::run_airgapped(&mut r.ctx, &dir)), 1);
    assert!(r.log.borrow().is_empty());
    assert_eq!(
        r.err.text(),
        "sandbox/run_airgapped.sh: line 15: DECANT_PACKAGE_DIR: Set DECANT_PACKAGE_DIR to the (read-only) game package directory\n"
    );
}

#[test]
fn isolation_suite() {
    let (_keep, root) = root();
    let dir = format!("{root}/sandbox");
    let env = [("DECANT_PACKAGE_DIR", "/pkg")];
    let respond: Respond = Box::new(|cmd| {
        let line = cmd.args.join(" ");
        let blocked =
            ["/dev/tcp/", "getent", "google.com", "/data/game/test", "/data/stubs/test", "test $(id -u) -eq 0"];
        let code = if blocked.iter().any(|b| line.contains(b)) { 1 } else { 0 };
        let stdout = if line.ends_with("ls /sys/class/net") {
            "lo\n"
        } else if cmd.program == "id" {
            "1000\n"
        } else {
            ""
        };
        Outcome { code, stdout: stdout.to_owned() }
    });
    let mut r = run_with(&env, vec![], respond);
    assert_eq!(code(sandbox::test_isolation(&mut r.ctx, &dir)), 0);
    let expected = "=== ISOLATION TEST SUITE ===

--- Network isolation ---
  ✓ Only the loopback interface exists
  ✓ No TCP to 1.1.1.1:53 (correctly blocked)
  ✓ No TCP to docker0 gateway 172.17.0.1:22 (correctly blocked)
  ✓ No DNS resolution (correctly blocked)
  ✓ No HTTP outbound (correctly blocked)
  ✓ No HTTPS outbound (correctly blocked)

--- Filesystem isolation ---
  ✓ Cannot write to game data (correctly blocked)
  ✓ Cannot write to stubs (correctly blocked)
  ✓ Can write to logs
  ✓ Can write to Wine prefix

--- Privilege isolation ---
  ✓ Not running as root (correctly blocked)
  ✓ no-new-privileges is set

--- Results ---
Passed: 12 / 12
Failed: 0 / 12
ALL TESTS PASSED - sandbox is properly isolated
";
    assert_eq!(r.out.text(), expected);
    let x = |args: &[&str]| -> Vec<String> {
        ["docker", "exec", "decant-sandbox"].iter().chain(args).map(|s| (*s).to_owned()).collect()
    };
    let expected_commands = vec![
        strings(&["docker", "compose", "down"]),
        strings(&["id", "-u"]),
        strings(&["id", "-g"]),
        strings(&["docker", "compose", "up", "-d"]),
        x(&["ls", "/sys/class/net"]),
        x(&["timeout", "3", "bash", "-c", "echo > /dev/tcp/1.1.1.1/53"]),
        x(&["timeout", "3", "bash", "-c", "echo > /dev/tcp/172.17.0.1/22"]),
        x(&["getent", "hosts", "google.com"]),
        x(&["curl", "-s", "--max-time", "2", "http://google.com"]),
        x(&["curl", "-s", "--max-time", "2", "https://google.com"]),
        x(&["touch", "/data/game/test"]),
        x(&["touch", "/data/stubs/test"]),
        x(&["touch", "/data/logs/test"]),
        x(&["rm", "/data/logs/test"]),
        x(&["touch", "/home/runner/.wine/test"]),
        x(&["rm", "/home/runner/.wine/test"]),
        x(&["sh", "-c", "test $(id -u) -eq 0"]),
        x(&["grep", "-q", "NoNewPrivs:[[:space:]]*1", "/proc/1/status"]),
    ];
    assert_eq!(argvs(&r.log), expected_commands);

    // One probe getting through fails the suite.
    let respond: Respond = Box::new(|cmd| Outcome {
        code: 0,
        stdout: if cmd.args.join(" ").ends_with("ls /sys/class/net") { "lo\n".to_owned() } else { String::new() },
    });
    let mut r = run_with(&env, vec![], respond);
    assert_eq!(code(sandbox::test_isolation(&mut r.ctx, &dir)), 1);
    let out = r.out.text();
    assert!(out.contains("  ✗ SECURITY FAILURE: No DNS resolution\n"));
    assert!(out.ends_with("Passed: 4 / 12\nFailed: 8 / 12\nSECURITY ISSUES DETECTED\n"), "{out}");
}

#[test]
fn dry_run_lines_quote_like_a_shell() {
    let r = run_with(&[], vec![], ok());
    let cmd = r
        .ctx
        .command("docker")
        .env("DISPLAY", "")
        .args(["exec", "a b", "it's"])
        .cwd("/w")
        .stdout(Sink::Tee("/l".to_owned()))
        .stderr(Sink::Stdout);
    assert_eq!(cmd.shell_line(), "(cd /w && DISPLAY='' docker exec 'a b' 'it'\\''s' 2>&1 | tee /l)");
}
