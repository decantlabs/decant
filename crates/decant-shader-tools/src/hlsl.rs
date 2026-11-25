//! `decant_shader_hlsl_probe.py`: converts a sample of the SPIR-V modules of a measurement to HLSL
//! (`spirv-cross`) and then to DXIL (`dxc`), and sorts the refusals. It weighs the Direct3D 12
//! route. Both programs run as separate processes, as in the script.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use regex::Regex;
use decant_shader_analysis::NL;
use decant_shader_analysis::py::Counter;
use decant_shader_analysis::random::Random;
use decant_shader_core::py::{read_text, splitlines};

use crate::files::{batches, modules, py_str};

/// Where one module stopped: `ok`, `spirv-cross` or `dxc`, with the reason.
pub type Outcome = (String, String);

/// `str.strip()`, whose whitespace also holds U+001C to U+001F.
fn strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn first_chars(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// The last line `spirv-cross` wrote to its error stream, numbers and names replaced by `#`.
pub fn cross_reason(stderr: &str) -> String {
    static NUMBERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"ID \d+|index \d+|\(name: [^)]*\)|%\d+|\d+").unwrap());
    let line = splitlines(strip(stderr)).last().copied().unwrap_or("?");
    first_chars(&NUMBERS.replace_all(line, "#"), 110)
}

/// The first line of `dxc` holding `error`, without what precedes the last `error: `.
pub fn dxc_reason(stderr: &str, stdout: &str) -> String {
    static PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^.*error: ").unwrap());
    let both = format!("{stderr}{stdout}");
    let line = splitlines(&both).into_iter().find(|l| l.contains("error")).unwrap_or("?");
    first_chars(&PREFIX.replace(line, ""), 110)
}

/// The two converters.
pub struct Tools {
    pub cross: PathBuf,
    pub dxc: PathBuf,
}

impl Tools {
    fn run(&self, exe: &Path, args: &[String]) -> Result<(bool, String, String), String> {
        let out = Command::new(exe).args(args).stdin(Stdio::null()).output().map_err(|e| format!("{}: {e}", exe.display()))?;
        Ok((out.status.success(), read_text(&out.stdout), read_text(&out.stderr)))
    }

    /// One module through both converters, the HLSL and DXIL files named after `profile` and
    /// `index` in `tmp`.
    pub fn convert(&self, module: &Path, profile: &str, index: usize, tmp: &Path) -> Result<Outcome, String> {
        let hlsl = py_str(&tmp.join(format!("{profile}_{index}.hlsl")));
        let (ok, _, err) = self.run(&self.cross, &[py_str(module), "--hlsl".into(), "--shader-model".into(), "66".into(), "--output".into(), hlsl.clone()])?;
        if !ok {
            return Ok(("spirv-cross".into(), cross_reason(&err)));
        }
        let dxil = py_str(&Path::new(&hlsl).with_extension("dxil"));
        let (ok, out, err) = self.run(&self.dxc, &["-T".into(), profile.into(), "-E".into(), "main".into(), hlsl, "-Fo".into(), dxil])?;
        if !ok {
            return Ok(("dxc".into(), dxc_reason(&err, &out)));
        }
        Ok(("ok".into(), String::new()))
    }
}

/// `convert` on every job, on eight workers, with the results in job order.
pub fn convert_all(jobs: &[(PathBuf, String, usize)], convert: &(dyn Fn(&Path, &str, usize) -> Result<Outcome, String> + Sync)) -> Result<Vec<Outcome>, String> {
    let next = AtomicUsize::new(0);
    let mut results: Vec<Option<Result<Outcome, String>>> = (0..jobs.len()).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<&mut Option<Result<Outcome, String>>>> = results.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let (module, profile, index) = &jobs[i];
                    **slots[i].lock().unwrap() = Some(convert(module, profile, *index));
                }
            });
        }
    });
    drop(slots);
    results.into_iter().map(|r| r.expect("every job ran")).collect()
}

/// The report of one `<folder>:<profile>` specification: `sample` drawn from `random`.
pub fn report(folder: &str, profile: &str, sample: &[PathBuf], results: &[Outcome]) -> String {
    let mut tally: Counter<Outcome> = Counter::default();
    for r in results {
        tally.add(r.clone(), 1);
    }
    let ok = tally.get(&("ok".to_owned(), String::new()));
    let mut out = format!("== {folder} ({profile}) : {ok} / {} convertis jusqu'au DXIL{NL}", sample.len());
    for ((step, why), n) in tally.most_common(Some(8)) {
        if step != "ok" {
            write!(out, "   {n:4}  {step}: {why}{NL}").unwrap();
        }
    }
    out
}

/// The modules of a measurement folder, `min(count, len)` of them drawn from `random`.
pub fn draw(random: &mut Random, folder: &Path, count: i64) -> Result<Vec<PathBuf>, String> {
    let mut list = Vec::new();
    for batch in batches(folder)? {
        list.extend(modules(&batch)?);
    }
    let k = usize::try_from(count.min(list.len() as i64)).map_err(|_| "ValueError: Sample larger than population or is negative")?;
    random.sample(&list, k)
}

/// The whole script for its arguments after the temporary folder and the count.
pub fn probe(
    count: i64,
    specs: &[String],
    convert: &(dyn Fn(&Path, &str, usize) -> Result<Outcome, String> + Sync),
    print: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let mut random = Random::seed(21);
    for spec in specs {
        let parts: Vec<&str> = spec.split(':').collect();
        let [folder, profile] = parts[..] else {
            return Err(format!("ValueError: {spec} is not <folder>:<profile>"));
        };
        let sample = draw(&mut random, Path::new(folder), count)?;
        let jobs: Vec<(PathBuf, String, usize)> = sample.iter().enumerate().map(|(i, m)| (m.clone(), profile.to_owned(), i)).collect();
        let results = convert_all(&jobs, convert)?;
        print(&report(folder, profile, &sample, &results));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_are_normalised_like_the_script() {
        let err = "warning: something\nSPIRV-Cross threw an exception: ID 42 (name: foo) at index 7, %12 has 3 uses\n\n";
        assert_eq!(cross_reason(err), "SPIRV-Cross threw an exception: # # at #, # has # uses");
        assert_eq!(cross_reason("  \n "), "?");
        assert_eq!(cross_reason(&"x".repeat(200)).len(), 110);
        let dxc = "C:\\t\\ps_6_6_0.hlsl:3:5: error: something: error: deep\nnote: x\n";
        assert_eq!(dxc_reason(dxc, ""), "deep");
        assert_eq!(dxc_reason("all good", "fatal error\n"), "all goodfatal error", "both streams are joined as they are");
        assert_eq!(dxc_reason("", ""), "?");
    }

    #[test]
    fn report_lists_the_refusals_most_common_first() {
        let o = |s: &str, w: &str| (s.to_owned(), w.to_owned());
        let results = [o("ok", ""), o("dxc", "b"), o("spirv-cross", "a"), o("dxc", "b"), o("ok", "")];
        let sample: Vec<PathBuf> = (0..5).map(|i| PathBuf::from(i.to_string())).collect();
        assert_eq!(
            report("m", "ps_6_6", &sample, &results),
            [
                "== m (ps_6_6) : 2 / 5 convertis jusqu'au DXIL".to_owned(),
                "      2  dxc: b".to_owned(),
                "      1  spirv-cross: a".to_owned()
            ]
            .map(|l| format!("{l}{NL}"))
            .concat()
        );
    }
}
