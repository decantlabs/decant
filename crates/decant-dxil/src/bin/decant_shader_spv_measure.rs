//! Measures the SPIR-V importer on a folder of modules: how many import, how many become a
//! container, how many the validator of dxil.dll accepts, and the most frequent reasons for failure.
//!
//! Usage: `decant_shader_spv_measure <folder> [--limit N] [--examples K] [--jobs J] [--filter <text in the path>] [--every N] [--dump <folder>]`
//!
//! The folder is searched recursively for `*.spv`. The modules are data: nothing is written
//! next to them. `--dump` keeps the containers that validate, and for the others the container and
//! the error text. The modules are processed by `J` threads (default 8). With `DECANT_SHADER_PROGRESS` set
//! the name of each module is printed when it starts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|e| e == "spv") {
            out.push(p);
        }
    }
}

/// Makes a message comparable: digits and ids replaced.
fn normalise(msg: &str) -> String {
    let first = msg.lines().find(|l| l.contains("error") || !l.trim().is_empty()).unwrap_or("").trim();
    let mut out = String::new();
    let mut digits = false;
    for c in first.chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('N');
            }
            digits = true;
        } else {
            digits = false;
            out.push(c);
        }
    }
    out.chars().take(160).collect()
}

#[derive(Default)]
struct Totals {
    imported: usize,
    written: usize,
    validated: usize,
    skipped_validation: usize,
    failures: BTreeMap<String, (usize, Vec<String>)>,
}

enum Outcome {
    Passed { validated: bool },
    /// `reached`: 0 nothing, 1 imported, 2 written.
    Failed { reached: u8, key: String },
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(folder) = args.next() else {
        eprintln!("usage: decant_shader_spv_measure <folder> [--limit N] [--examples K] [--jobs J] [--filter TEXT] [--every N] [--dump <folder>]");
        return ExitCode::from(2);
    };
    let (mut limit, mut examples, mut jobs, mut dump) = (usize::MAX, 2usize, 8usize, None::<PathBuf>);
    let (mut filter, mut every) = (None::<String>, 1usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
            "--examples" => examples = args.next().and_then(|v| v.parse().ok()).unwrap_or(2),
            "--jobs" => jobs = args.next().and_then(|v| v.parse().ok()).unwrap_or(8).max(1),
            "--dump" => dump = args.next().map(PathBuf::from),
            "--filter" => filter = args.next(),
            "--every" => every = args.next().and_then(|v| v.parse().ok()).unwrap_or(1).max(1),
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    let mut files = Vec::new();
    collect(Path::new(&folder), &mut files);
    if let Some(f) = &filter {
        files.retain(|p| p.to_string_lossy().contains(f.as_str()));
    }
    if every > 1 {
        files = files.into_iter().enumerate().filter(|(i, _)| i % every == 0).map(|(_, p)| p).collect();
    }
    files.truncate(limit);
    let total = files.len();
    let progress = std::env::var_os("DECANT_SHADER_PROGRESS").is_some();
    let totals = Mutex::new(Totals::default());
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let n = next.fetch_add(1, Ordering::SeqCst);
            let Some(file) = files.get(n) else { break };
            if progress {
                eprintln!("{n} {}", file.display());
            }
            let outcome = process(file, n, dump.as_deref());
            let mut t = totals.lock().unwrap();
            match outcome {
                Outcome::Passed { validated } => {
                    t.imported += 1;
                    t.written += 1;
                    if validated {
                        t.validated += 1;
                    } else {
                        t.skipped_validation += 1;
                    }
                }
                Outcome::Failed { reached, key } => {
                    if reached >= 1 {
                        t.imported += 1;
                    }
                    if reached >= 2 {
                        t.written += 1;
                    }
                    let e = t.failures.entry(key).or_default();
                    e.0 += 1;
                    if e.1.len() < examples {
                        e.1.push(file.display().to_string());
                    }
                }
            }
        }
    };
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            // A big stack: the importer recurses through long chains of blocks.
            std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(scope, work).expect("a worker thread");
        }
    });
    let t = totals.into_inner().unwrap();
    println!("modules {total}");
    println!("imported {}", t.imported);
    println!("written {}", t.written);
    println!("validated {} (validation skipped for {})", t.validated, t.skipped_validation);
    let mut rows: Vec<(&String, &(usize, Vec<String>))> = t.failures.iter().collect();
    rows.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(b.0)));
    println!("failures by cause:");
    for (key, (count, ex)) in rows {
        println!("{count:6}  {key}");
        for e in ex {
            println!("          e.g. {e}");
        }
    }
    ExitCode::SUCCESS
}

fn process(file: &Path, n: usize, dump: Option<&Path>) -> Outcome {
    let Ok(bytes) = std::fs::read(file) else {
        return Outcome::Failed { reached: 0, key: "read: unreadable".into() };
    };
    let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let imp = match std::panic::catch_unwind(|| decant_dxil::spirv::import(&words)) {
        Err(_) => return Outcome::Failed { reached: 0, key: "import: panic".into() },
        Ok(Err(e)) => {
            let key = if e.category == "internal" || e.category.starts_with("type") { format!("{} {}", e.category, normalise(&e.detail)) } else { e.category.clone() };
            return Outcome::Failed { reached: 0, key: format!("import: {key}") };
        }
        Ok(Ok(i)) => i,
    };
    let out = match std::panic::catch_unwind(|| decant_dxil::write(&imp.shader)) {
        Err(_) => return Outcome::Failed { reached: 1, key: "write: panic".into() },
        Ok(Err(e)) => {
            let key = match &e {
                decant_dxil::Error::Unsupported(m) => format!("unsupported {}", normalise(m)),
                other => normalise(&other.to_string()),
            };
            return Outcome::Failed { reached: 1, key: format!("write: {key}") };
        }
        Ok(Ok(o)) => o,
    };
    #[cfg(windows)]
    {
        let mut blob = out.container.clone();
        // `<folder of the module>_<module>`, for example `1-00048_7`: the measure folder keeps
        // `<stage>-<offset>/<local>.spv`, so the dump names one module and its manifest line.
        let stem = file.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let parent = file.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let name = format!("{parent}_{stem}");
        match decant_dxil::validate::validate_in_place(&mut blob) {
            Ok(()) => {
                if let Some(d) = dump {
                    let _ = std::fs::create_dir_all(d);
                    let _ = std::fs::write(d.join(format!("{n}_{name}.dxbc")), &out.container);
                }
                Outcome::Passed { validated: true }
            }
            Err(decant_dxil::validate::ValidateError::Missing(_)) => Outcome::Passed { validated: false },
            Err(decant_dxil::validate::ValidateError::Rejected(m)) => {
                if let Some(d) = dump {
                    let _ = std::fs::create_dir_all(d);
                    let _ = std::fs::write(d.join(format!("fail_{n}_{name}.dxbc")), &out.container);
                    let _ = std::fs::write(d.join(format!("fail_{n}_{name}.err")), &m);
                }
                Outcome::Failed { reached: 2, key: format!("validate: {}", normalise(&m)) }
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = (&out, dump, n);
        Outcome::Passed { validated: false }
    }
}
