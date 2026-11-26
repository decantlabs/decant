//! `decant_shader_direct_census <shader archive or shaders_gpu.pack> [options]`
//!
//! Without options: the census of the whole corpus (see `decant_shader::census`).
//!
//! | Option | Effect |
//! |---|---|
//! | `--top N` | rows per list of the report (default 40) |
//! | `--stage S` | only programs of stage `vertex`, `pixel`, `compute`, `mesh`, `geometry` or `amplification` |
//! | `--show ID...` | instead of the census, print the listing and the HLSL (or the refusal) of the programs with these positions or container hashes (16 hex digits) |
//! | `--name PATTERN` | only programs whose name matches (case insensitive, `*` and `?`, else a substring) |
//! | `--hash H` | only the program with this container hash (16 hex digits), repeatable |
//! | `--hashes FILE` | only programs whose container hash is listed in FILE (one hex hash per line, `#` starts a comment) |
//! | `--sample N` `--seed S` | a random sample of N of the programs the filters keep, the same seed gives the same sample |
//! | `--threads N` | worker threads (default: every core) |
//! | `--quiet` | print only the counts, the first blockers and the lifter gaps |
//! | `--wave32` | `--show` translates as for a host whose waves have 32 lanes: a 64 lane wave is emulated through group shared memory or refused (`hlsl::wave_emu`) |
//! | `--ab DIR` | write the translated HLSL of the selected programs and the batch list `DIR/pairs.tsv` for `decant-shader-ab` (A = the old override `<hash>.dxil` from the `--override-dir` folders, B = the new HLSL) |
//! | `--override-dir DIR` | folder of old overrides, repeatable, the first one that has the hash wins |
//! | `--ab-run EXE` | then run `EXE batch` on that list on WARP only (`--ab-jobs N`, `--ab-timeout S`) and print its table |
//! | `--bin FILE` | also count the raw console container in FILE (a layer dump), repeatable, whatever the filters say. Also adds its program to the pipeline census |
//! | `--line LABEL` | print only the progress line of this run (translated per stage against the baseline, top blockers with their change) |
//! | `--pipelines FILE` | also count pipelines in FILE (a version 1 manifest or a layer log, repeatable). The pipeline census uses all referenced stages and the manifest root signature |
//! | `--json FILE` | also write the report as JSON (stable keys, per program outcomes) |
//! | `--baseline FILE` | after the report, print the delta to the stored report FILE (also set by the environment variable `DECANT_SHADER_CENSUS_BASELINE`) |
//! | `--save` | then store this run as the baseline |
//! | `--diff OLD NEW` (first argument) | no census: print the delta between two stored reports (use `--top`) |
//! | `--blocked FILE` | also write one line per program that is not translated: position, name, refusal key |
//!
//! An archive gives whole containers (signatures, resource lists, hashes). A pack only has the
//! code, which is enough for the instruction counts but not for the stage interfaces.
//! Everything this tool prints comes from title files: keep it local.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use decant_shader::api::{CountingHost, Options, RootSignature, Stage, StageProgram, WaveMode};

/// `--wave32`: `--show` translates as for a host whose waves have 32 lanes (`WaveMode::Emulated`).
static WAVE32: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
use decant_shader::census::ab::{self, Emitted};
use decant_shader::census::pipeline::{self, PipelineCensus, PipelineLists};
use decant_shader::census::{Census, Report};
use decant_shader::corpus::{self, Entry, Selection};
use decant_shader::{container, decode};

fn stage_of(name: &str) -> Option<Stage> {
    Some(match name {
        "vertex" => Stage::Vertex,
        "pixel" => Stage::Pixel,
        "compute" => Stage::Compute,
        "mesh" => Stage::Mesh,
        "geometry" => Stage::Geometry,
        "amplification" => Stage::Amplification,
        _ => return None,
    })
}

fn show(e: &Entry) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "== {} {} {:016x}", e.index, e.name, e.hash);
    let p = match &e.program {
        Ok(p) => p,
        Err(r) => {
            let _ = writeln!(o, "   {r}");
            return o;
        }
    };
    let _ = writeln!(
        o,
        "   all lifter gaps: {}",
        decant_shader::census::lifter_gaps(p).join(" ")
    );
    let _ = writeln!(
        o,
        "   stage {:?}
   inputs {:?}
   outputs {:?}
   resources {:?}
   table {:?}
   threads {:?}
   user data {:?}
   param exports {:?}
   pixel inputs {:?}",
        p.stage,
        p.inputs,
        p.outputs,
        p.resources,
        p.binding_table,
        p.threads,
        p.user_data,
        p.param_exports,
        p.pixel_inputs
    );
    for item in decode::decode_lossy(&p.code[container::SIGNATURE_WORDS.min(p.code.len())..]) {
        match item {
            Ok(i) => {
                let _ = writeln!(o, "{:5}: {i}", i.pc as usize + container::SIGNATURE_WORDS);
            }
            Err(e) => {
                let _ = writeln!(o, "   !! {e}");
            }
        }
    }
    match decant_shader::translate_probe(
        std::slice::from_ref(p),
        &RootSignature::default(),
        &Options { wave: if WAVE32.load(std::sync::atomic::Ordering::Relaxed) { WaveMode::Emulated } else { WaveMode::Scalar }, ..Options::default() },
        &CountingHost::default(),
    ) {
        Ok(t) => {
            let _ = writeln!(o, "{}", t.stages[0].hlsl);
        }
        Err(r) => {
            let _ = writeln!(o, "   refused: {r}");
        }
    }
    o
}

/// What one worker thread collects.
#[derive(Default)]
struct Acc {
    census: Census,
    /// `(position, line)` of each program that is not translated.
    lines: Vec<(usize, String)>,
    shown: Vec<(usize, String)>,
    /// Position and line of the comparer batch list.
    pairs: Vec<(usize, String)>,
    /// Programs by what became of them for the comparer.
    emitted: [u64; 4],
}

/// Runs the comparer over the batch list on WARP (never the real GPU: the old overrides with
/// generated inputs can hang it, and the game runs on this machine) and prints its table.
fn run_comparer(exe: &str, dir: &Path, jobs: usize, timeout: u64) -> Result<(), String> {
    let out = std::process::Command::new(exe)
        .arg("batch")
        .arg(dir.join("pairs.tsv"))
        .args(["-j", &jobs.to_string(), "--timeout", &timeout.to_string()])
        .arg("--out")
        .arg(dir.join("results.tsv"))
        .arg("--json-out")
        .arg(dir.join("results.jsonl"))
        .env("DECANT_SHADER_DEVICE", "warp")
        .output()
        .map_err(|e| format!("{exe}: {e}"))?;
    print!("{}", String::from_utf8_lossy(&out.stdout));
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    println!("results: {}", dir.join("results.tsv").display());
    Ok(())
}

/// Where a report goes after it was printed.
struct Output {
    /// Print the progress line of this list instead of the delta.
    line: Option<String>,
    json: Option<String>,
    baseline: Option<String>,
    save: bool,
    top: usize,
}

/// Writes the JSON report, prints the delta to the baseline and stores a new baseline.
fn finish(report: &Report, out: &Output) -> Result<(), String> {
    if let Some(file) = &out.json {
        std::fs::write(file, report.to_json()).map_err(|e| format!("{file}: {e}"))?;
    }
    if let Some(label) = &out.line {
        let old = out
            .baseline
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .map(|t| Report::from_json(&t))
            .transpose()?;
        print!("{}", report.progress_line(old.as_ref(), label, out.top));
    }
    if let Some(file) = &out.baseline {
        match std::fs::read_to_string(file) {
            Ok(_) if out.line.is_some() => {}
            Err(_) if out.line.is_some() => {}
            Ok(text) => println!(
                "
{}",
                report.diff(&Report::from_json(&text)?, out.top)
            ),
            Err(_) => println!(
                "
no stored report at {file}"
            ),
        }
        if out.save {
            std::fs::write(file, report.to_json()).map_err(|e| format!("{file}: {e}"))?;
            println!("stored as the new baseline: {file}");
        }
    } else if out.save {
        return Err("--save needs --baseline FILE".to_owned());
    }
    Ok(())
}

/// The pipeline census: every pipeline of the lists, its programs translated together.
fn pipelines(
    path: &Path,
    lists: &[String],
    bins: &[PathBuf],
    threads: usize,
) -> Result<PipelineCensus, String> {
    let mut pipelines = PipelineLists::default();
    for f in lists {
        pipelines.read(Path::new(f))?;
    }
    let hashes: HashSet<u64> = pipelines
        .seen
        .keys()
        .flat_map(|p| p.stages.iter().map(|(_, h)| *h))
        .collect();
    let sel = Selection {
        hashes: Some(hashes),
        ..Selection::default()
    };
    let (_, parts) = corpus::par_corpus(
        path,
        &sel,
        threads,
        HashMap::new,
        |m: &mut HashMap<u64, StageProgram>, e: Entry| {
            if let Ok(p) = e.program {
                m.insert(e.hash, p);
            }
        },
    )?;
    let mut programs: HashMap<u64, StageProgram> = parts.into_iter().flatten().collect();
    for e in corpus::bin_entries(bins, usize::MAX / 2)? {
        if let Ok(p) = e.program {
            programs.insert(e.hash, p);
        }
    }
    let started = std::time::Instant::now();
    let census = pipeline::run(&programs, &pipelines, threads);
    eprintln!("pipeline census: {:.1} s", started.elapsed().as_secs_f64());
    Ok(census)
}

fn number<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    what: &str,
) -> Result<T, String> {
    args.next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{what} needs a number"))
}

fn run() -> Result<(), String> {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    // `--diff OLD NEW` stands alone: the archive is not needed.
    if argv.first().is_some_and(|a| a == "--diff") {
        argv.insert(0, String::new());
    }
    let mut args = argv.into_iter();
    let path = args.next().ok_or("usage: decant_shader_direct_census <archive or pack> [--json FILE] [--baseline FILE] [--save] [--diff OLD NEW] [--top N] [--stage S] [--name PATTERN] [--hash H] [--hashes FILE] [--sample N] [--seed S] [--threads N] [--quiet] [--show ID...] [--blocked FILE]")?;
    let (mut top, mut blocked, mut threads, mut quiet) = (40usize, None, 0usize, false);
    let mut sel = Selection::default();
    let (mut sample, mut seed) = (None, 0u64);
    let (mut json, mut baseline, mut save) =
        (None, std::env::var("DECANT_SHADER_CENSUS_BASELINE").ok(), false);
    let mut pipeline_lists: Vec<String> = Vec::new();
    let mut bins: Vec<PathBuf> = Vec::new();
    let mut line: Option<String> = None;
    let (mut ab_dir, mut ab_exe, mut overrides) = (None, None, Vec::new());
    let (mut ab_jobs, mut ab_timeout) = (2usize, 60u64);
    let mut wanted = Vec::new();
    let mut showing = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--top" => top = number(&mut args, "--top")?,
            "--threads" => threads = number(&mut args, "--threads")?,
            "--sample" => sample = Some(number::<usize>(&mut args, "--sample")?),
            "--seed" => seed = number(&mut args, "--seed")?,
            "--quiet" => quiet = true,
            "--wave32" => WAVE32.store(true, std::sync::atomic::Ordering::Relaxed),
            "--line" => line = Some(args.next().ok_or("--line needs a label")?),
            "--bin" => bins.push(PathBuf::from(args.next().ok_or("--bin needs a file")?)),
            "--ab" => ab_dir = Some(PathBuf::from(args.next().ok_or("--ab needs a folder")?)),
            "--override-dir" => overrides.push(PathBuf::from(
                args.next().ok_or("--override-dir needs a folder")?,
            )),
            "--ab-run" => ab_exe = Some(args.next().ok_or("--ab-run needs decant-shader-ab.exe")?),
            "--ab-jobs" => ab_jobs = number(&mut args, "--ab-jobs")?,
            "--ab-timeout" => ab_timeout = number(&mut args, "--ab-timeout")?,
            "--pipelines" => pipeline_lists.push(args.next().ok_or("--pipelines needs a file")?),
            "--save" => save = true,
            "--json" => json = Some(args.next().ok_or("--json needs a file")?),
            "--baseline" => baseline = Some(args.next().ok_or("--baseline needs a file")?),
            "--diff" => {
                let (a, b) = (
                    args.next().ok_or("--diff needs two reports")?,
                    args.next().ok_or("--diff needs two reports")?,
                );
                let read = |f: &str| {
                    std::fs::read_to_string(f)
                        .map_err(|e| format!("{f}: {e}"))
                        .and_then(|t| Report::from_json(&t))
                };
                print!("{}", read(&b)?.diff(&read(&a)?, top));
                return Ok(());
            }
            "--name" => sel.name = Some(args.next().ok_or("--name needs a pattern")?),
            "--hash" => {
                let h = args.next().ok_or("--hash needs a hash")?;
                let h = u64::from_str_radix(h.trim_start_matches("0x"), 16)
                    .map_err(|_| format!("--hash: {h} is not hex"))?;
                sel.hashes.get_or_insert_with(HashSet::new).insert(h);
            }
            "--hashes" => {
                let f = args.next().ok_or("--hashes needs a file")?;
                let text = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
                sel.hashes
                    .get_or_insert_with(HashSet::new)
                    .extend(corpus::parse_hashes(&text));
            }
            "--stage" => {
                sel.stage = Some(args.next().as_deref().and_then(stage_of).ok_or(
                    "--stage needs vertex, pixel, compute, mesh, geometry or amplification",
                )?)
            }
            "--blocked" => blocked = Some(args.next().ok_or("--blocked needs a file")?),
            "--show" => showing = true,
            id if showing => wanted.push(id.to_owned()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    sel.sample = sample.map(|n| (n, seed));
    let path = Path::new(&path);
    let pack = path.extension().is_some_and(|e| e == "pack");
    let started = std::time::Instant::now();
    let visit = |acc: &mut Acc, e: Entry| {
        if showing {
            let hash = format!("{:016x}", e.hash);
            if wanted
                .iter()
                .any(|w| *w == hash || w.parse() == Ok(e.index))
            {
                acc.shown.push((e.index, show(&e)));
            }
            return;
        }
        let Ok(p) = &e.program else {
            acc.census.add_undecodable(&e.id());
            return;
        };
        if let Some(dir) = &ab_dir {
            match ab::emit(dir, &overrides, e.hash, p) {
                Ok(Emitted::Pair(line)) => {
                    acc.emitted[0] += 1;
                    acc.pairs.push((e.index, line));
                }
                Ok(Emitted::NoOverride) => acc.emitted[1] += 1,
                Ok(Emitted::NotTranslated) => acc.emitted[2] += 1,
                Ok(Emitted::Unsupported) => acc.emitted[3] += 1,
                Err(err) => eprintln!("{err}"),
            }
        }
        if let Some(key) = acc.census.add(e.index, &e.id(), p) {
            acc.lines
                .push((e.index, format!("{}	{}	{key}", e.index, e.name)));
        }
    };
    let (total, mut accs) = if pack {
        corpus::par_pack(path, &sel, threads, Acc::default, visit)?
    } else {
        corpus::par_corpus(path, &sel, threads, Acc::default, visit)?
    };
    // Raw containers named with `--bin` are always counted, whatever the filters say.
    let mut extra = Acc::default();
    for e in corpus::bin_entries(&bins, total)? {
        visit(&mut extra, e);
    }
    accs.push(extra);
    let mut all = Acc::default();
    for a in accs {
        all.census.merge(a.census);
        all.lines.extend(a.lines);
        all.shown.extend(a.shown);
        all.pairs.extend(a.pairs);
        for (n, m) in all.emitted.iter_mut().zip(a.emitted) {
            *n += m;
        }
    }
    if showing {
        all.shown.sort();
        for (_, text) in &all.shown {
            print!("{text}");
        }
        return Ok(());
    }
    eprintln!("census: {:.1} s", started.elapsed().as_secs_f64());
    if line.is_none() {
        println!(
            "corpus: {} shaders in {}",
            total,
            if pack { "a pack" } else { "an archive" }
        );
    }
    if line.is_none() {
        if quiet {
            print!("{}", all.census.summary(top));
        } else {
            print!("{}", all.census.report(top));
        }
    }
    let selection = format!(
        "{} name={:?} stage={:?} hashes={} sample={:?}",
        path.file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        sel.name,
        sel.stage,
        sel.hashes.as_ref().map_or(0, |h| h.len()),
        sel.sample
    );
    if let Some(dir) = &ab_dir {
        all.pairs.sort();
        let list = dir.join("pairs.tsv");
        let text: String = all.pairs.iter().map(|(_, l)| format!("{l}\n")).collect();
        std::fs::write(&list, text).map_err(|e| format!("{}: {e}", list.display()))?;
        let [pairs, none, refused, other] = all.emitted;
        println!(
            "\ncomparer list {}: {pairs} pairs, {none} translated without an old override, {refused} not translated, {other} of a stage the comparer does not run",
            list.display()
        );
        if let Some(exe) = &ab_exe {
            run_comparer(exe, dir, ab_jobs, ab_timeout)?;
        }
    }
    let mut report = all.census.to_report(&selection);
    if !pipeline_lists.is_empty() {
        let census = pipelines(path, &pipeline_lists, &bins, threads)?;
        if line.is_none() {
            print!("{}", census.report(top));
        }
        report = report.with_pipelines(&census);
    }
    finish(
        &report,
        &Output {
            line,
            json,
            baseline,
            save,
            top,
        },
    )?;
    if let Some(file) = blocked {
        all.lines.sort();
        let mut f = std::fs::File::create(&file).map_err(|e| format!("{file}: {e}"))?;
        for (_, l) in all.lines {
            writeln!(f, "{l}").map_err(|e| format!("{file}: {e}"))?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
