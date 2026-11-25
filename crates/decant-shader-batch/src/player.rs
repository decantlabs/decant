//! `decant_shader_player_cache`: builds the ahead-of-time translation cache (`DECANT_SHADER_SPV_CACHE`) on a
//! player's machine, from the player's own game package, by chaining the existing tools as
//! separate processes. It adds no translation logic of its own:
//!
//! 1. `extract_shaders <shaders> --out <work>\out` writes `shaders_gpu.pack` and `shaders.csv`.
//! 2. `decant_shader_metadata --out <work>\out` (shaders from `DECANT_SHADERS_RPF`) writes
//!    `decant_shader_metadata.jsonl`.
//! 3. `decant_shader_outputs --out <work>\out` (same shaders) writes `decant_shader_outputs.jsonl`, which
//!    `decant_shader_locate` reads when `decant_shader_serve` places a module.
//! 4. `decant_shader_batch <work>\measure 0 1 2 5 13 --metadata <jsonl> --placeholders`, with
//!    `DECANT_SHADER_SOURCES=1` and `DECANT_SHADER_NO_MIP_STATS=1`, the pack from `DECANT_SHADERS_DIR`
//!    (`<work>`, whose `out` folder steps 1 to 3 fill) and the probe from `DECANT_SHADER_PROBE`.
//! 5. `decant_shader_cache <work>\measure <root>\spv` writes the modules and `cache.json`.
//!
//! Steps 4 and 5 are skipped when `<root>\spv\cache.json` records the SHA-256 of the current probe
//! and pack. The probe is GPL code and is only ever started as a separate process.
//!
//! Everything this module decides (paths, commands, environment, `run_env.txt`, the incremental
//! decision) is a pure function of the options, so it is tested without the probe or game files.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

/// The stages the cache covers, as `decant_shader_cache` expects the measure.
pub const STAGES: [&str; 5] = ["0", "1", "2", "5", "13"];

/// The command line, before defaults are applied.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub package: PathBuf,
    pub cache: PathBuf,
    pub probe: Option<PathBuf>,
    pub shaders: Option<PathBuf>,
    pub work: Option<PathBuf>,
    pub tools: Option<PathBuf>,
    pub plan: bool,
}

pub const USAGE: &str = "usage: decant_shader_player_cache --package <package dir> --cache <cache root> [--probe <decant_shader_probe.exe>]
                             [--shaders <file or folder>] [--work <dir>] [--tools <dir>] [--plan]

Builds the ahead-of-time shader translation cache from your own game package:
  extract_shaders -> decant_shader_metadata -> decant_shader_batch -> decant_shader_cache
  --package : the game package folder, scanned for console shader containers
  --cache   : output root: spv\\ (the cache), adapt\\, override\\, scratch\\ and run_env.txt
  --probe   : the DecantShader probe (default: decant_shader_probe.exe in --tools)
  --shaders : one file or a folder of files with the shader containers, instead of the package
  --work    : scratch folder for the pack, metadata and measure (default: <cache>\\work)
  --tools   : folder of the sibling programs (default: the folder of this program)
  --plan    : print the steps (commands and environment) without running anything
Everything written comes from title files: keep it locally, never in a repository.";

/// Parses the arguments (without the program name).
pub fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let (mut package, mut cache) = (None, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || it.next().map(PathBuf::from).ok_or_else(|| format!("{arg} needs a value\n{USAGE}"));
        match arg.as_str() {
            "--package" => package = Some(value()?),
            "--cache" => cache = Some(value()?),
            "--probe" => o.probe = Some(value()?),
            "--shaders" => o.shaders = Some(value()?),
            "--work" => o.work = Some(value()?),
            "--tools" => o.tools = Some(value()?),
            "--plan" => o.plan = true,
            other => return Err(format!("unexpected argument {other}\n{USAGE}")),
        }
    }
    o.package = package.ok_or_else(|| format!("--package is required\n{USAGE}"))?;
    o.cache = cache.ok_or_else(|| format!("--cache is required\n{USAGE}"))?;
    Ok(o)
}

/// Every path a run uses, with the defaults applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub root: PathBuf,
    pub spv: PathBuf,
    pub adapt: PathBuf,
    pub override_dir: PathBuf,
    pub scratch: PathBuf,
    /// `DECANT_SHADERS_DIR` of every tool: `out\` holds the pack, the table and the metadata,
    /// `decant_shader\` is the working folder `decant_shader_serve` starts its helpers in.
    pub work: PathBuf,
    pub out: PathBuf,
    pub measure: PathBuf,
    pub shaders: PathBuf,
    pub probe: PathBuf,
    pub tools: PathBuf,
}

impl Layout {
    /// `shaders` is the package folder (or `--shaders`), `default_tools` the folder of the
    /// running program.
    pub fn new(o: &Options, shaders: PathBuf, default_tools: &Path) -> Layout {
        let root = o.cache.clone();
        let tools = o.tools.clone().unwrap_or_else(|| default_tools.to_path_buf());
        let work = o.work.clone().unwrap_or_else(|| root.join("work"));
        Layout {
            spv: root.join("spv"),
            adapt: root.join("adapt"),
            override_dir: root.join("override"),
            scratch: root.join("scratch"),
            out: work.join("out"),
            measure: work.join("measure"),
            probe: o.probe.clone().unwrap_or_else(|| program(&tools, "decant_shader_probe")),
            shaders,
            work,
            tools,
            root,
        }
    }

    pub fn pack(&self) -> PathBuf {
        self.out.join("shaders_gpu.pack")
    }

    pub fn metadata(&self) -> PathBuf {
        self.out.join("decant_shader_metadata.jsonl")
    }

    /// The container index `decant_shader_serve` keeps (pack SHA-1 to pack number); stale when the pack
    /// changes.
    pub fn container_index(&self) -> PathBuf {
        self.work.join("decant_shader").join("harness").join("out").join("index_conteneurs_pack.json")
    }

    pub fn cache_json(&self) -> PathBuf {
        self.spv.join("cache.json")
    }

    pub fn run_env_file(&self) -> PathBuf {
        self.root.join("run_env.txt")
    }
}

/// The program `name` in the tools folder.
pub fn program(tools: &Path, name: &str) -> PathBuf {
    tools.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// One step: a program run as a separate process with extra environment variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub name: &'static str,
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(&'static str, OsString)>,
}

impl Step {
    /// The command as one line, for the log and `--plan`.
    pub fn line(&self) -> String {
        let mut parts = vec![quote(&self.program.to_string_lossy())];
        parts.extend(self.args.iter().map(|a| quote(&a.to_string_lossy())));
        parts.join(" ")
    }
}

/// A word quoted for display (and for `cmd`) when it holds whitespace.
pub fn quote(word: &str) -> String {
    if word.chars().any(char::is_whitespace) { format!("\"{word}\"") } else { word.to_owned() }
}

/// The five steps, in order: extract, metadata, outputs, measure, cache. The output signatures
/// (`decant_shader_outputs.jsonl`) are read by `decant_shader_locate` when `decant_shader_serve` places a module.
pub fn plan(l: &Layout) -> [Step; 5] {
    let shaders_dir = || ("DECANT_SHADERS_DIR", l.work.clone().into_os_string());
    let probe = || ("DECANT_SHADER_PROBE", l.probe.clone().into_os_string());
    let mut measure: Vec<OsString> = vec![l.measure.clone().into()];
    measure.extend(STAGES.iter().map(OsString::from));
    measure.extend(["--metadata".into(), l.metadata().into(), "--placeholders".into()]);
    [
        Step {
            name: "extract_shaders",
            program: program(&l.tools, "extract_shaders"),
            args: vec![l.shaders.clone().into(), "--out".into(), l.out.clone().into()],
            env: vec![],
        },
        Step {
            name: "decant_shader_metadata",
            program: program(&l.tools, "decant_shader_metadata"),
            args: vec!["--out".into(), l.out.clone().into()],
            env: vec![("DECANT_SHADERS_RPF", l.shaders.clone().into())],
        },
        Step {
            name: "decant_shader_outputs",
            program: program(&l.tools, "decant_shader_outputs"),
            args: vec!["--out".into(), l.out.clone().into()],
            env: vec![("DECANT_SHADERS_RPF", l.shaders.clone().into())],
        },
        Step {
            name: "decant_shader_batch",
            program: program(&l.tools, "decant_shader_batch"),
            args: measure,
            env: vec![shaders_dir(), probe(), ("DECANT_SHADER_SOURCES", "1".into()), ("DECANT_SHADER_NO_MIP_STATS", "1".into())],
        },
        Step {
            name: "decant_shader_cache",
            program: program(&l.tools, "decant_shader_cache"),
            args: vec![l.measure.clone().into(), l.spv.clone().into()],
            env: vec![shaders_dir(), probe()],
        },
    ]
}

/// The environment a game run needs to use the cache through the vkd3d-proton hook. The hook runs
/// `VKD3D_DECANT_SHADER_COMMAND <16 hex digit hash>` through `system()`, so the program is quoted when
/// its path holds a space. `decant_shader_serve` reads the pack table and metadata from
/// `DECANT_SHADERS_DIR`, the containers from `DECANT_SHADERS_RPF`, and starts `decant_shader_one`
/// (with `DECANT_SHADER_PROBE`) for a shader the cache lacks.
pub fn run_env(l: &Layout) -> Vec<(&'static str, String)> {
    let show = |p: &Path| p.display().to_string();
    vec![
        ("DECANT_SHADER_SPV_CACHE", show(&l.spv)),
        ("DECANT_SHADER_ADAPT_CACHE", show(&l.adapt)),
        ("VKD3D_SHADER_OVERRIDE", show(&l.override_dir)),
        ("VKD3D_DECANT_SHADER_DIR", show(&l.scratch)),
        ("VKD3D_DECANT_SHADER_COMMAND", quote(&show(&program(&l.tools, "decant_shader_serve")))),
        ("DECANT_SHADERS_DIR", show(&l.work)),
        ("DECANT_SHADERS_RPF", show(&l.shaders)),
        ("DECANT_SHADER_PROBE", show(&l.probe)),
    ]
}

/// `run_env.txt`: one `NAME=value` per line.
pub fn run_env_text(l: &Layout) -> String {
    run_env(l).iter().map(|(k, v)| format!("{k}={v}\n")).collect()
}

/// Whether the cache in `spv\` matches the current probe and pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// Both SHA-256 match: steps 4 and 5 can be skipped.
    Current,
    /// No `cache.json` yet.
    Missing,
    /// A `cache.json` for another probe or pack (or unreadable), with the reason.
    Stale(String),
}

/// Compares `cache.json` (as `decant_shader_cache` writes it: `pack_sha256`, `probe_sha256`) with the
/// current hashes.
pub fn freshness(cache_json: &Path, pack_sha256: &str, probe_sha256: &str) -> Freshness {
    let Ok(text) = fs::read_to_string(cache_json) else {
        return Freshness::Missing;
    };
    let value: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => return Freshness::Stale(format!("{} is not valid JSON: {e}", cache_json.display())),
    };
    let field = |name: &str| value.get(name).and_then(|v| v.as_str()).map(str::to_owned);
    match (field("pack_sha256"), field("probe_sha256")) {
        (Some(pack), _) if !pack.eq_ignore_ascii_case(pack_sha256) => Freshness::Stale("the pack changed".into()),
        (None, _) => Freshness::Stale("cache.json records no pack_sha256".into()),
        (_, None) => Freshness::Stale("cache.json records no probe_sha256".into()),
        (_, Some(probe)) if !probe.eq_ignore_ascii_case(probe_sha256) => Freshness::Stale("the probe changed".into()),
        _ => Freshness::Current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(extra: &[&str]) -> Options {
        let mut args = vec!["--package", "P", "--cache", "C"];
        args.extend_from_slice(extra);
        parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    fn os(items: &[&Path]) -> Vec<OsString> {
        items.iter().map(|p| p.as_os_str().to_owned()).collect()
    }

    #[test]
    fn arguments() {
        let o = opts(&["--plan", "--probe", "probe.exe"]);
        assert_eq!((o.package, o.cache, o.plan, o.probe), ("P".into(), "C".into(), true, Some("probe.exe".into())));
        let err = |a: &[&str]| parse(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_err();
        assert!(err(&["--cache", "C"]).starts_with("--package is required"));
        assert!(err(&["--package", "P"]).starts_with("--cache is required"));
        assert!(err(&["--package", "P", "--cache"]).starts_with("--cache needs a value"));
        assert!(err(&["--package", "P", "--cache", "C", "--bogus"]).starts_with("unexpected argument --bogus"));
    }

    #[test]
    fn the_plan_with_defaults() {
        let root = Path::new("root");
        let exe = Path::new("bin");
        let l = Layout::new(&opts(&[]), Path::new("P").join("shaders"), exe);
        let work = root.join("work");
        let out = work.join("out");
        let measure = work.join("measure");
        assert_eq!((l.root.as_path(), l.work.as_path(), l.tools.as_path()), (Path::new("C"), Path::new("C").join("work").as_path(), exe));
        let l = Layout::new(&Options { cache: root.into(), ..opts(&[]) }, Path::new("P").join("shaders"), exe);
        let probe = program(exe, "decant_shader_probe");
        assert_eq!(l.probe, probe);
        let [extract, meta, outputs, batch, cache] = plan(&l);

        assert_eq!(extract.program, program(exe, "extract_shaders"));
        assert_eq!(extract.args, os(&[&Path::new("P").join("shaders"), Path::new("--out"), &out]));
        assert!(extract.env.is_empty());

        assert_eq!(meta.program, program(exe, "decant_shader_metadata"));
        assert_eq!(meta.args, os(&[Path::new("--out"), &out]));
        assert_eq!(meta.env, vec![("DECANT_SHADERS_RPF", Path::new("P").join("shaders").into_os_string())]);

        assert_eq!(outputs.program, program(exe, "decant_shader_outputs"));
        assert_eq!(outputs.args, meta.args);
        assert_eq!(outputs.env, meta.env);

        assert_eq!(batch.program, program(exe, "decant_shader_batch"));
        let jsonl = out.join("decant_shader_metadata.jsonl");
        let expected: Vec<&Path> =
            vec![&measure, "0".as_ref(), "1".as_ref(), "2".as_ref(), "5".as_ref(), "13".as_ref(), "--metadata".as_ref(), &jsonl, "--placeholders".as_ref()];
        assert_eq!(batch.args, os(&expected));
        let names: Vec<&str> = batch.env.iter().map(|(k, _)| *k).collect();
        assert_eq!(names, ["DECANT_SHADERS_DIR", "DECANT_SHADER_PROBE", "DECANT_SHADER_SOURCES", "DECANT_SHADER_NO_MIP_STATS"]);
        assert_eq!(batch.env[0].1, work.as_os_str());
        assert_eq!(batch.env[1].1, probe.as_os_str());
        assert_eq!((batch.env[2].1.as_os_str(), batch.env[3].1.as_os_str()), ("1".as_ref(), "1".as_ref()));

        assert_eq!(cache.program, program(exe, "decant_shader_cache"));
        assert_eq!(cache.args, os(&[&measure, &root.join("spv")]));
        assert_eq!(cache.env, batch.env[..2]);
    }

    #[test]
    fn the_plan_with_overrides() {
        let tools = Path::new("my tools");
        let o = opts(&["--shaders", "elsewhere", "--tools", "my tools", "--work", "W", "--probe", "p.exe"]);
        assert_eq!(o.shaders.as_deref(), Some(Path::new("elsewhere")));
        let l = Layout::new(&o, o.shaders.clone().unwrap(), Path::new("ignored"));
        let [extract, meta, outputs, batch, cache] = plan(&l);
        for (step, name) in [(&extract, "extract_shaders"), (&meta, "decant_shader_metadata"), (&outputs, "decant_shader_outputs"), (&batch, "decant_shader_batch"), (&cache, "decant_shader_cache")] {
            assert_eq!(step.program, program(tools, name));
            assert!(step.line().starts_with('"'), "a program path with a space is quoted: {}", step.line());
        }
        assert_eq!(extract.args[0], "elsewhere");
        assert_eq!(meta.env[0].1, "elsewhere");
        assert_eq!(batch.args[0], Path::new("W").join("measure").as_os_str());
        assert_eq!(batch.env[0].1, "W");
        assert_eq!(batch.env[1].1, "p.exe");
        assert_eq!(cache.args[1], Path::new("C").join("spv").as_os_str());
    }

    #[test]
    fn run_env_lists_the_keys_and_paths() {
        let o = opts(&["--tools", "T dir", "--probe", "probe.exe"]);
        let l = Layout::new(&o, "a".into(), Path::new("bin"));
        let text = run_env_text(&l);
        let lines: Vec<(&str, &str)> = text.lines().map(|line| line.split_once('=').unwrap()).collect();
        let c = Path::new("C");
        let show = |p: PathBuf| p.display().to_string();
        assert_eq!(
            lines,
            [
                ("DECANT_SHADER_SPV_CACHE", show(c.join("spv")).as_str()),
                ("DECANT_SHADER_ADAPT_CACHE", show(c.join("adapt")).as_str()),
                ("VKD3D_SHADER_OVERRIDE", show(c.join("override")).as_str()),
                ("VKD3D_DECANT_SHADER_DIR", show(c.join("scratch")).as_str()),
                ("VKD3D_DECANT_SHADER_COMMAND", format!("\"{}\"", show(program(Path::new("T dir"), "decant_shader_serve"))).as_str()),
                ("DECANT_SHADERS_DIR", show(c.join("work")).as_str()),
                ("DECANT_SHADERS_RPF", "a"),
                ("DECANT_SHADER_PROBE", "probe.exe"),
            ]
        );
        assert!(text.ends_with('\n'));
        let plain = Layout::new(&opts(&[]), "a".into(), Path::new("bin"));
        assert_eq!(run_env(&plain)[4].1, show(program(Path::new("bin"), "decant_shader_serve")));
    }

    #[test]
    fn the_incremental_decision() {
        let dir = tempfile::tempdir().unwrap();
        let json = dir.path().join("cache.json");
        let (pack, probe) = ("a".repeat(64), "b".repeat(64));
        assert_eq!(freshness(&json, &pack, &probe), Freshness::Missing);
        let write = |pack: &str, probe: &str| {
            let info = crate::cache::Info {
                modules: 3,
                copied: crate::cache::Copied { copied: 3, missing_src: 0 },
                pack_sha256: pack.into(),
                probe_sha256: Some(probe.into()),
            };
            fs::write(&json, info.file()).unwrap();
        };
        write(&pack, &probe);
        assert_eq!(freshness(&json, &pack, &probe), Freshness::Current);
        assert_eq!(freshness(&json, &pack.to_uppercase(), &probe), Freshness::Current);
        assert_eq!(freshness(&json, &"c".repeat(64), &probe), Freshness::Stale("the pack changed".into()));
        assert_eq!(freshness(&json, &pack, &"c".repeat(64)), Freshness::Stale("the probe changed".into()));
        fs::write(&json, format!("{{\"pack_sha256\": \"{pack}\", \"probe_sha256\": null}}")).unwrap();
        assert_eq!(freshness(&json, &pack, &probe), Freshness::Stale("cache.json records no probe_sha256".into()));
        fs::write(&json, "{").unwrap();
        assert!(matches!(freshness(&json, &pack, &probe), Freshness::Stale(_)));
    }
}
