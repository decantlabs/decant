//! The regression guard of the translator (binary `decant_shader_regress`): two probes, before and after
//! a change, translate the same fixed sample of shaders, and each pair of modules is compared.
//!
//! Each shader is translated by both probes, one process at a time, with a deadline and a memory
//! ceiling per module (some compute modules make a compiler ask for tens of gigabytes). The two
//! SPIR-V modules are compared byte for byte, then with spirv-diff, and each goes through
//! spirv-val (Vulkan 1.2, with `--allow-offset-texture-operand`: the probe assumes
//! VK_KHR_maintenance8).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use decant_extract_shaders::pack::{self, Record};
use serde_json::{Value, json};
use decant_shader_batch::process::{Ended, run_limited};
use decant_shader_bridge::random::Random;

/// The default sample, unchanged so the same seed keeps drawing the same shaders.
pub const STAGES: [u8; 4] = [0, 1, 5, 13];
/// Every stage `--stages` accepts (2 is geometry).
pub const ALL_STAGES: [u8; 5] = [0, 1, 2, 5, 13];
pub const REGRESSIONS: [&str; 2] = ["ne_valide_plus", "plus_traduit"];

/// The pack numbers to measure: the given list, or a sample the seed makes reproducible.
pub fn choose(pack_path: &Path, ids: &[i64], sample: usize, seed: i64, stages: &HashSet<u8>, skip: &HashSet<i64>) -> Result<Vec<i64>, String> {
    if !ids.is_empty() {
        let mut chosen: Vec<i64> = ids.iter().copied().filter(|i| !skip.contains(i)).collect::<HashSet<_>>().into_iter().collect();
        chosen.sort_unstable();
        return Ok(chosen);
    }
    let mut pool = Vec::new();
    pack::read(pack_path, |index, _, stage| {
        if stages.contains(&stage) && !skip.contains(&(index as i64)) {
            pool.push(index as i64);
        }
        false
    })
    .map_err(|e| format!("{}: {e}", pack_path.display()))?;
    let mut chosen = Random::seed(seed.unsigned_abs()).sample(&pool, sample.min(pool.len()));
    chosen.sort_unstable();
    Ok(chosen)
}

/// The records of `wanted`, by pack number. The code of the others is not read.
pub fn records(pack_path: &Path, wanted: &HashSet<i64>) -> Result<HashMap<i64, Record>, String> {
    let (header, records) = pack::read(pack_path, |index, _, _| wanted.contains(&(index as i64))).map_err(|e| format!("{}: {e}", pack_path.display()))?;
    if !header.is_current() {
        return Err(format!("{} : pas un pack XSGC version 1", pack_path.display()));
    }
    Ok(records.into_iter().map(|r| (r.index as i64, r)).collect())
}

/// What one probe made of one shader.
#[derive(Clone, Debug, PartialEq)]
pub struct Translation {
    /// `traduit`, `echec`, `delai` or `plantage`.
    pub state: &'static str,
    pub detail: String,
    pub module: Option<PathBuf>,
}

/// The limits of every external program.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub timeout: f64,
    pub max_memory_mb: u64,
    pub diff_timeout: f64,
    pub diff_memory_mb: u64,
}

/// `{x:g}` for the timeouts printed in the journal.
pub fn seconds(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e16 { format!("{}", x as i64) } else { format!("{x}") }
}

/// Reads the probe's tab separated journal: tag, then values after the second field.
fn journal(text: &str) -> HashMap<String, Vec<String>> {
    let mut rows = HashMap::new();
    for line in decant_shader_core::py::splitlines(text) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() >= 2 {
            rows.insert(fields[0].to_owned(), fields[2..].iter().map(|s| (*s).to_owned()).collect());
        }
    }
    rows
}

/// Translates the shader of the one shader pack `pack` with `probe`, in `folder`.
pub fn translate(probe: &Path, env: &[(String, String)], pack: &Path, stage: u8, folder: &Path, mode: u8, limits: &Limits) -> Result<Translation, String> {
    let log = folder.join("one.tsv");
    let module = folder.join("0.spv");
    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&module);
    let mut command = Command::new(probe);
    command.arg(pack).arg("--stage").arg(&log).arg("0").arg(stage.to_string()).arg(folder).arg(mode.to_string());
    command.envs(env.iter().map(|(k, v)| (k, v)));
    let ended = run_limited(&mut command, Duration::from_secs_f64(limits.timeout), limits.max_memory_mb, None).map_err(|e| format!("{}: {e}", probe.display()))?;
    let rows = std::fs::read(&log).map(|b| journal(&String::from_utf8_lossy(&b))).unwrap_or_default();
    let code = match ended {
        Ended::Timeout => return Ok(Translation { state: "delai", detail: format!("plus de {} s", seconds(limits.timeout)), module: None }),
        Ended::Exit(code) => code,
    };
    let first = |tag: &str| rows.get(tag).and_then(|v| v.first()).map_or("", String::as_str);
    if first("C") == "OK" && module.exists() {
        return Ok(Translation { state: "traduit", detail: String::new(), module: Some(module) });
    }
    for tag in ["C", "T"] {
        if first(tag) == "FAIL" {
            let reason: String = rows[tag].get(1).map(|r| r.chars().take(200).collect()).unwrap_or_default();
            return Ok(Translation { state: "echec", detail: format!("{tag} : {reason}"), module: None });
        }
    }
    Ok(Translation { state: "plantage", detail: format!("code {code}"), module: None })
}

/// The verdict of one shader and what it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub verdict: &'static str,
    pub valid_before: Option<bool>,
    pub valid_after: Option<bool>,
    pub difference: String,
}

/// Compares the two translations of a shader. `validate` runs spirv-val, `diff` spirv-diff.
pub fn classify(before: &Translation, after: &Translation, validate: &dyn Fn(&Path) -> bool, diff: &dyn Fn(&Path, &Path) -> String) -> Verdict {
    let (a, b) = (before.module.as_deref().filter(|_| before.state == "traduit"), after.module.as_deref().filter(|_| after.state == "traduit"));
    let mut row = Verdict { verdict: "", valid_before: a.map(validate), valid_after: b.map(validate), difference: String::new() };
    row.verdict = match (a, b) {
        (Some(_), None) => "plus_traduit",
        (None, Some(_)) => "nouvellement_traduit",
        (None, None) => "toujours_non_traduit",
        (Some(a), Some(b)) => {
            if std::fs::read(a).ok() == std::fs::read(b).ok() {
                "identique"
            } else {
                row.difference = diff(a, b);
                match (row.valid_before == Some(true), row.valid_after == Some(true)) {
                    (true, false) => "ne_valide_plus",
                    (false, true) => "valide_maintenant",
                    (false, false) => "toujours_invalide",
                    (true, true) => "modifie",
                }
            }
        }
    };
    row
}

/// The summary of the rows: count per verdict (sorted by name) and the regressed pack numbers.
pub fn summary(rows: &[Value]) -> Value {
    let mut counts: BTreeMap<String, i64> = BTreeMap::new();
    let mut regressions = Vec::new();
    for r in rows {
        let verdict = r["verdict"].as_str().unwrap_or_default();
        *counts.entry(verdict.to_owned()).or_default() += 1;
        if REGRESSIONS.contains(&verdict) {
            regressions.push(r["pack_id"].clone());
        }
    }
    json!({"shaders": rows.len(), "verdicts": counts, "regressions": regressions})
}

/// spirv-val on one module, with the probe's memory ceiling.
pub fn validate(validator: &Path, module: &Path, max_memory_mb: u64) -> bool {
    let mut command = Command::new(validator);
    command.args(["--target-env", "vulkan1.2", "--allow-offset-texture-operand"]).arg(module);
    matches!(run_limited(&mut command, Duration::from_secs(30), max_memory_mb, None), Ok(Ended::Exit(0)))
}

/// spirv-diff of two modules under its own ceiling: on a large module it easily asks for several
/// gigabytes. When it fails, the byte comparison and spirv-val are enough for the verdict, only the
/// detail is missing.
pub fn diff(differ: Option<&Path>, a: &Path, b: &Path, limits: &Limits) -> String {
    let Some(differ) = differ else { return "(spirv-diff absent)".into() };
    let report = a.parent().and_then(Path::parent).unwrap_or(Path::new(".")).join("difference.tmp");
    let ended = std::fs::File::create(&report).and_then(|sink| {
        let mut command = Command::new(differ);
        command.arg(a).arg(b);
        run_limited(&mut command, Duration::from_secs_f64(limits.diff_timeout), limits.diff_memory_mb, Some(sink))
    });
    let text = std::fs::read(&report).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let _ = std::fs::remove_file(&report);
    match ended {
        Ok(Ended::Timeout) => format!("(spirv-diff : plus de {} s, detail non calcule)\n", seconds(limits.diff_timeout)),
        Ok(Ended::Exit(code)) if code != 0 && text.is_empty() => {
            format!("(spirv-diff : arret code {code}, probablement le plafond de {} Mo, detail non calcule)\n", limits.diff_memory_mb)
        }
        Err(e) => format!("(spirv-diff : {e}, detail non calcule)\n"),
        _ if text.is_empty() => "(spirv-diff : aucune difference d'instruction, seulement l'encodage)\n".into(),
        _ => text,
    }
}

/// The program `name` on the `PATH`, as `shutil.which` finds it.
pub fn which(name: &str) -> Option<PathBuf> {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::var_os("PATH").iter().flat_map(std::env::split_paths).map(|d| d.join(&file)).find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(state: &'static str, module: Option<&Path>) -> Translation {
        Translation { state, detail: String::new(), module: module.map(Path::to_path_buf) }
    }

    #[test]
    fn verdicts_follow_translation_bytes_and_validity() {
        let dir = tempfile::tempdir().unwrap();
        let (one, same, other, bad) = (dir.path().join("1"), dir.path().join("2"), dir.path().join("3"), dir.path().join("4"));
        std::fs::write(&one, b"module").unwrap();
        std::fs::write(&same, b"module").unwrap();
        std::fs::write(&other, b"changed").unwrap();
        std::fs::write(&bad, b"broken").unwrap();
        let valid = |p: &Path| !p.ends_with("4");
        let diff = |_: &Path, _: &Path| "- a\n+ b\n".to_owned();
        let v = |a: Translation, b: Translation| classify(&a, &b, &valid, &diff);
        let none = translation("echec", None);
        assert_eq!(v(translation("traduit", Some(&one)), translation("traduit", Some(&same))).verdict, "identique");
        assert_eq!(v(translation("traduit", Some(&one)), translation("traduit", Some(&other))).verdict, "modifie");
        assert_eq!(v(translation("traduit", Some(&one)), translation("traduit", Some(&bad))).verdict, "ne_valide_plus");
        assert_eq!(v(translation("traduit", Some(&bad)), translation("traduit", Some(&one))).verdict, "valide_maintenant");
        assert_eq!(v(translation("traduit", Some(&bad)), translation("traduit", Some(&bad))).verdict, "identique");
        assert_eq!(v(translation("traduit", Some(&one)), none.clone()).verdict, "plus_traduit");
        assert_eq!(v(none.clone(), translation("traduit", Some(&bad))).verdict, "nouvellement_traduit");
        assert_eq!(v(none.clone(), none).verdict, "toujours_non_traduit");
        let changed = v(translation("traduit", Some(&one)), translation("traduit", Some(&other)));
        assert_eq!((changed.valid_before, changed.valid_after, changed.difference.as_str()), (Some(true), Some(true), "- a\n+ b\n"));
    }

    #[test]
    fn the_summary_sorts_verdicts_and_lists_regressions() {
        let rows = [
            json!({"verdict": "modifie", "pack_id": 3}),
            json!({"verdict": "ne_valide_plus", "pack_id": 9}),
            json!({"verdict": "identique", "pack_id": 1}),
            json!({"verdict": "plus_traduit", "pack_id": 4}),
        ];
        let s = summary(&rows);
        assert_eq!(decant_shader_bridge::describe::compact(&s), r#"{"shaders": 4, "verdicts": {"identique": 1, "modifie": 1, "ne_valide_plus": 1, "plus_traduit": 1}, "regressions": [9, 4]}"#);
    }

    #[test]
    fn the_sample_is_reproducible_and_skips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.pack");
        let mut records = Vec::new();
        for (i, stage) in [0u8, 1, 2, 5, 13, 0, 1, 2, 5, 13].iter().enumerate() {
            let name = format!("S{i}");
            let mut r = (name.len() as u16).to_le_bytes().to_vec();
            r.extend(name.as_bytes());
            let mut tail = [0u8; 21];
            tail[0] = *stage;
            tail[17] = 1;
            r.extend(tail);
            r.extend([0u8; 4]);
            records.push(r);
        }
        std::fs::write(&path, pack::write(records.iter().map(Vec::as_slice))).unwrap();
        let stages: HashSet<u8> = STAGES.into();
        let skip = HashSet::from([3]);
        let a = choose(&path, &[], 4, 7, &stages, &skip).unwrap();
        assert_eq!(a, choose(&path, &[], 4, 7, &stages, &skip).unwrap());
        assert_eq!(a.len(), 4);
        assert!(a.iter().all(|i| ![2, 7, 3].contains(i)));
        assert_eq!(choose(&path, &[5, 3, 5, 1], 0, 7, &stages, &skip).unwrap(), [1, 5]);
        let wanted = HashSet::from([1, 8]);
        let read = records_of(&path, &wanted);
        assert_eq!(read, [(1, "S1".to_owned(), 1), (8, "S8".to_owned(), 5)]);
    }

    fn records_of(path: &Path, wanted: &HashSet<i64>) -> Vec<(i64, String, u8)> {
        let mut v: Vec<_> = records(path, wanted).unwrap().into_iter().map(|(i, r)| (i, r.name, r.stage)).collect();
        v.sort();
        v
    }

    #[test]
    fn the_probe_journal_gives_the_state() {
        let rows = journal("C\t0\tFAIL\tbad opcode\nT\t0\tOK\nshort\n");
        assert_eq!(rows["C"], ["FAIL", "bad opcode"]);
        assert!(!rows.contains_key("short"));
        assert_eq!((seconds(45.0), seconds(2.5)), ("45".to_owned(), "2.5".to_owned()));
    }
}
