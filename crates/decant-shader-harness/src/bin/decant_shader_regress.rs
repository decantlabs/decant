//! The regression guard of the translator. See `decant_shader_harness::regress`.
//!
//! Verdicts per shader:
//! * `identique`: both modules are the same
//! * `modifie`: the module changes and still validates
//! * `ne_valide_plus`: valid before, refused after (REGRESSION)
//! * `plus_traduit`: translated before, no longer after (REGRESSION)
//! * `valide_maintenant`: refused before, valid after
//! * `toujours_invalide`: translated on both sides, refused on both sides
//! * `nouvellement_traduit`: not translated before, translated after
//! * `toujours_non_traduit`: translated on neither side
//!
//! The exit code is 1 when there is at least one regression, so the tool can guard a merge of the
//! translator.
//!
//! Usage: `decant_shader_regress <output> --before <probe> --after <probe> (--ids 1,2,3 | --sample N
//! [--seed S]) [--stages 0 1 2 5 13] [--skip 24238,24240] [--pack ...] [--metadata ...]
//! [--mode 0|1] [--before-env KEY=VAL] [--after-env KEY=VAL] [--timeout S] [--max-memory-mb N]
//! [--diff-timeout S] [--diff-memory-mb N] [--spirv-val ...] [--spirv-diff ...]`
//!
//! The pack and metadata are those of the measurement run (`src/shaders/out`). Everything written
//! to `<output>` (modules, differences, report) comes from title files: keep it locally.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use serde_json::{Value, json};
use decant_shader_batch::batch::write_inputs;
use decant_shader_bridge::describe::{compact, pretty};
use decant_shader_core::py::native_newlines;
use decant_shader_harness::regress::{self, ALL_STAGES, Limits, STAGES};

const USAGE: &str = "usage : decant_shader_regress <sortie> --before <sonde> --after <sonde> (--ids 1,2,3 | --sample N [--seed S])\n        [--stages 0 1 2 5 13] [--skip 24238,24240] [--pack ...] [--metadata ...] [--mode 0|1]\n        [--before-env CLE=VAL] [--after-env CLE=VAL] [--timeout S] [--max-memory-mb N]\n        [--diff-timeout S] [--diff-memory-mb N] [--spirv-val ...] [--spirv-diff ...]";

#[derive(Debug)]
struct Args {
    output: PathBuf,
    before: PathBuf,
    after: PathBuf,
    ids: String,
    sample: usize,
    seed: i64,
    stages: Vec<u8>,
    skip: String,
    pack: PathBuf,
    metadata: PathBuf,
    mode: u8,
    before_env: Vec<(String, String)>,
    after_env: Vec<(String, String)>,
    limits: Limits,
    spirv_val: Option<PathBuf>,
    spirv_diff: Option<PathBuf>,
}

fn error(message: &str) -> String {
    format!("{USAGE}\ndecant_shader_regress: error: {message}")
}

fn number<T: std::str::FromStr>(option: &str, value: &str) -> Result<T, String> {
    value.trim().parse().map_err(|_| error(&format!("argument {option}: invalid value: '{value}'")))
}

fn pair(value: &str) -> (String, String) {
    let (k, v) = value.split_once('=').unwrap_or((value, ""));
    (k.to_owned(), v.to_owned())
}

fn parse(args: &[String]) -> Result<Args, String> {
    let out = decant_shader_core::paths::out();
    let mut a = Args {
        output: PathBuf::new(),
        before: PathBuf::new(),
        after: PathBuf::new(),
        ids: String::new(),
        sample: 0,
        seed: 7,
        stages: STAGES.to_vec(),
        skip: String::new(),
        pack: out.join("shaders_gpu.pack"),
        metadata: out.join("decant_shader_metadata.jsonl"),
        mode: 1,
        before_env: Vec::new(),
        after_env: Vec::new(),
        limits: Limits { timeout: 45.0, max_memory_mb: 4096, diff_timeout: 30.0, diff_memory_mb: 1024 },
        spirv_val: None,
        spirv_diff: None,
    };
    let (mut positional, mut before, mut after) = (Vec::new(), false, false);
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if !arg.starts_with("--") {
            positional.push(arg.clone());
            continue;
        }
        let (option, inline) = match arg.split_once('=') {
            Some((o, v)) => (o, Some(v.to_owned())),
            None => (arg.as_str(), None),
        };
        if option == "--help" {
            return Err(USAGE.into());
        }
        if option == "--stages" {
            a.stages.clear();
            while let Some(v) = args.get(i).filter(|v| !v.starts_with("--")) {
                let stage: u8 = number(option, v)?;
                if !ALL_STAGES.contains(&stage) {
                    return Err(error(&format!("argument --stages: invalid choice: {stage} (choose from 0, 1, 2, 5, 13)")));
                }
                a.stages.push(stage);
                i += 1;
            }
            if a.stages.is_empty() {
                return Err(error("argument --stages: expected at least one argument"));
            }
            continue;
        }
        let value = match inline {
            Some(v) => v,
            None => {
                let v = args.get(i).ok_or_else(|| error(&format!("argument {option}: expected one argument")))?.clone();
                i += 1;
                v
            }
        };
        match option {
            "--before" => (a.before, before) = (PathBuf::from(value), true),
            "--after" => (a.after, after) = (PathBuf::from(value), true),
            "--ids" => a.ids = value,
            "--sample" => a.sample = number(option, &value)?,
            "--seed" => a.seed = number(option, &value)?,
            "--skip" => a.skip = value,
            "--pack" => a.pack = PathBuf::from(value),
            "--metadata" => a.metadata = PathBuf::from(value),
            "--mode" => {
                a.mode = number(option, &value)?;
                if a.mode > 1 {
                    return Err(error(&format!("argument --mode: invalid choice: {} (choose from 0, 1)", a.mode)));
                }
            }
            "--before-env" => a.before_env.push(pair(&value)),
            "--after-env" => a.after_env.push(pair(&value)),
            "--timeout" => a.limits.timeout = number(option, &value)?,
            "--max-memory-mb" => a.limits.max_memory_mb = number(option, &value)?,
            "--diff-timeout" => a.limits.diff_timeout = number(option, &value)?,
            "--diff-memory-mb" => a.limits.diff_memory_mb = number(option, &value)?,
            "--spirv-val" => a.spirv_val = Some(PathBuf::from(value)),
            "--spirv-diff" => a.spirv_diff = Some(PathBuf::from(value)),
            _ => return Err(error(&format!("unrecognized arguments: {arg}"))),
        }
    }
    match positional.as_slice() {
        [output] => a.output = PathBuf::from(output),
        [] => return Err(error("the following arguments are required: output")),
        [_, rest @ ..] => return Err(error(&format!("unrecognized arguments: {}", rest.join(" ")))),
    }
    if !before || !after {
        return Err(error("the following arguments are required: --before, --after"));
    }
    Ok(a)
}

fn numbers(text: &str) -> Result<Vec<i64>, String> {
    text.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().map_err(|_| format!("ValueError: invalid literal for int() with base 10: '{x}'"))).collect()
}

fn tool(given: &Option<PathBuf>, variable: &str, name: &str) -> Option<PathBuf> {
    given.clone().or_else(|| std::env::var_os(variable).filter(|v| !v.is_empty()).map(PathBuf::from)).or_else(|| regress::which(name))
}

fn io(path: &std::path::Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

fn run(args: &Args) -> Result<bool, String> {
    let validator = tool(&args.spirv_val, "SPIRV_VAL", "spirv-val").ok_or("spirv-val introuvable (--spirv-val, SPIRV_VAL ou PATH)")?;
    let differ = tool(&args.spirv_diff, "SPIRV_DIFF", "spirv-diff");
    let skip: HashSet<i64> = numbers(&args.skip)?.into_iter().collect();
    let ids = numbers(&args.ids)?;
    if ids.is_empty() && args.sample == 0 {
        return Err("donner --ids ou --sample".into());
    }
    let stages: HashSet<u8> = args.stages.iter().copied().collect();
    let chosen = regress::choose(&args.pack, &ids, args.sample, args.seed, &stages, &skip)?;
    let records = regress::records(&args.pack, &chosen.iter().copied().collect())?;
    let meta = if args.metadata.exists() { Some(decant_shader_core::meta::rows(&args.metadata)?) } else { None };

    let out = std::path::absolute(&args.output).map_err(io(&args.output))?;
    std::fs::create_dir_all(&out).map_err(io(&out))?;
    let absolute = |p: &PathBuf| std::path::absolute(p).unwrap_or_else(|_| p.clone());
    let sides = [("avant", absolute(&args.before), &args.before_env), ("apres", absolute(&args.after), &args.after_env)];
    let limits = args.limits;
    let mut rows: Vec<Value> = Vec::new();
    for number in &chosen {
        let record = records.get(number).ok_or_else(|| format!("KeyError: {number}"))?;
        let folder = out.join(number.to_string());
        let mut results = Vec::new();
        for (side, probe, env) in &sides {
            let work = folder.join(side);
            std::fs::create_dir_all(&work).map_err(io(&work))?;
            let pack = work.join("one.pack");
            write_inputs(&pack, std::slice::from_ref(record), meta.as_deref())?;
            results.push(regress::translate(probe, env, &pack, record.stage, &work, args.mode, &limits)?);
            let _ = std::fs::remove_file(&pack);
        }
        let verdict = regress::classify(
            &results[0],
            &results[1],
            &|m| regress::validate(&validator, m, limits.max_memory_mb),
            &|a, b| regress::diff(differ.as_deref(), a, b, &limits),
        );
        if !verdict.difference.is_empty() {
            let path = folder.join("difference.txt");
            std::fs::write(&path, native_newlines(&verdict.difference)).map_err(io(&path))?;
        }
        rows.push(json!({
            "valide_avant": verdict.valid_before,
            "valide_apres": verdict.valid_after,
            "verdict": verdict.verdict,
            "pack_id": number,
            "name": record.name,
            "stage": record.stage,
            "avant": results[0].state,
            "apres": results[1].state,
            "detail_avant": results[0].detail,
            "detail_apres": results[1].detail,
            "lignes_differentes": decant_shader_core::py::splitlines(&verdict.difference).len(),
        }));
        println!("{number}\t{}", verdict.verdict);
        let _ = std::io::stdout().flush();
    }
    let path = out.join("regress.jsonl");
    let text: String = rows.iter().map(|r| compact(r) + "\n").collect();
    std::fs::write(&path, native_newlines(&text)).map_err(io(&path))?;
    let summary = regress::summary(&rows);
    let mut result = summary.as_object().cloned().unwrap_or_default();
    let mut skipped: Vec<i64> = skip.into_iter().collect();
    skipped.sort_unstable();
    result.insert("before".into(), json!(sides[0].1.display().to_string()));
    result.insert("after".into(), json!(sides[1].1.display().to_string()));
    result.insert("mode".into(), json!(args.mode));
    result.insert("skip".into(), json!(skipped));
    let path = out.join("summary.json");
    std::fs::write(&path, native_newlines(&pretty(&Value::Object(result)))).map_err(io(&path))?;
    println!("{}", compact(&summary));
    Ok(summary["regressions"].as_array().is_some_and(|r| r.is_empty()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        parse(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn arguments() {
        let a = args(&["out", "--before", "a.exe", "--after=b.exe", "--sample", "40", "--stages", "2", "5", "--before-env", "K=V=W", "--mode", "0"]).unwrap();
        assert_eq!((a.sample, a.stages.as_slice(), a.mode, a.seed), (40, [2u8, 5].as_slice(), 0, 7));
        assert_eq!(a.before_env, [("K".to_owned(), "V=W".to_owned())]);
        assert_eq!((a.limits.timeout, a.limits.max_memory_mb, a.limits.diff_memory_mb), (45.0, 4096, 1024));
        assert!(args(&["out", "--before", "a"]).unwrap_err().ends_with("--before, --after"));
        assert!(args(&["out", "--before", "a", "--after", "b", "--stages", "3"]).unwrap_err().contains("invalid choice: 3"));
        assert!(args(&["out", "--before", "a", "--after", "b", "--mode", "2"]).is_err());
        assert_eq!(numbers(" 1, 2,,3 ").unwrap(), [1, 2, 3]);
    }
}
