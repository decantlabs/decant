//! `decant_shader_batch.py`: measures the pack in batches of 48 shaders. A probe run that crashes or
//! hangs is stopped and the batch goes on after the shader the probe log names.
//!
//! Each batch gets a folder `<stage>-<offset>` with its own `batch.pack`, the `.inputs` and
//! `.vinputs` files the probe reads, the probe log `batch.tsv` and the modules. The rows of all
//! batches go to `results.jsonl`, in batch order.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use decant_extract_shaders::pack::{self, Record};
use serde_json::Value;
use decant_shader_core::{meta, py};

use crate::json::{Field, Row};
use crate::process::{self, Ended};

/// Shaders per batch.
pub const BATCH: usize = 48;
/// Batches measured at the same time.
pub const WORKERS: usize = 5;
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(45);
pub const VALIDATE_TIMEOUT: Duration = Duration::from_secs(10);

/// What `spirv-val` said about one module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Validation {
    Ok,
    /// Its error output.
    Fail(String),
    Timeout,
}

/// The external programs a batch runs. [`External`] runs the real ones, tests replace them.
pub trait Tools: Sync {
    /// `decant_shader_probe.exe <pack> --stage <log> <first> <stage> <folder> <mode>`.
    fn probe(&self, pack: &Path, log: &Path, first: usize, stage: u8, folder: &Path, mode: u8) -> Result<Ended, String>;
    /// Whether a validator was given.
    fn validates(&self) -> bool;
    /// `spirv-val --target-env vulkan1.2 --allow-offset-texture-operand <module>`: the probe assumes a device with
    /// `VK_KHR_maintenance8` (dynamic texel offsets on image samples), so the validator does too.
    fn validate(&self, module: &Path) -> Result<Validation, String>;
}

/// The DecantShader probe and an optional `spirv-val`, run as separate processes.
pub struct External {
    pub probe: PathBuf,
    pub validator: Option<PathBuf>,
}

impl Tools for External {
    fn probe(&self, pack: &Path, log: &Path, first: usize, stage: u8, folder: &Path, mode: u8) -> Result<Ended, String> {
        let mut command = Command::new(&self.probe);
        command.arg(pack).arg("--stage").arg(log).arg(first.to_string()).arg(stage.to_string()).arg(folder).arg(mode.to_string());
        Ok(process::run(&mut command, PROBE_TIMEOUT, false).map_err(|e| format!("{}: {e}", self.probe.display()))?.ended)
    }

    fn validates(&self) -> bool {
        self.validator.is_some()
    }

    fn validate(&self, module: &Path) -> Result<Validation, String> {
        let validator = self.validator.as_ref().ok_or("no validator")?;
        let mut command = Command::new(validator);
        command.arg("--target-env").arg("vulkan1.2").arg("--allow-offset-texture-operand").arg(module);
        let out = process::run(&mut command, VALIDATE_TIMEOUT, true).map_err(|e| format!("{}: {e}", validator.display()))?;
        Ok(match out.ended {
            Ended::Exit(0) => Validation::Ok,
            Ended::Exit(_) => Validation::Fail(String::from_utf8_lossy(&out.stderr).into_owned()),
            Ended::Timeout => Validation::Timeout,
        })
    }
}

/// One batch: up to [`BATCH`] records of one stage, from `offset` in that stage's list.
pub struct Job<'a> {
    pub stage: u8,
    pub offset: usize,
    pub records: &'a [Record],
}

/// Records grouped by stage in the order the stages were given (repeats count once), then cut
/// into batches.
pub fn group(stages: &[u8], records: Vec<Record>) -> Vec<(u8, Vec<Record>)> {
    let mut buckets: Vec<(u8, Vec<Record>)> = Vec::new();
    for &s in stages {
        if !buckets.iter().any(|(b, _)| *b == s) {
            buckets.push((s, Vec::new()));
        }
    }
    for record in records {
        if let Some((_, list)) = buckets.iter_mut().find(|(s, _)| *s == record.stage) {
            list.push(record);
        }
    }
    buckets
}

pub fn jobs(buckets: &[(u8, Vec<Record>)]) -> Vec<Job<'_>> {
    buckets
        .iter()
        .flat_map(|(stage, records)| {
            records.chunks(BATCH).enumerate().map(|(n, chunk)| Job { stage: *stage, offset: n * BATCH, records: chunk })
        })
        .collect()
}

/// The probe mode argument: 2 for `--ir-only`, otherwise 1 with `--placeholders` and 0 without.
pub fn mode(ir_only: bool, placeholders: bool) -> u8 {
    if ir_only { 2 } else { u8::from(placeholders) }
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> String + '_ {
    move |e| format!("{}: {e}", path.display())
}

/// Writes the files the probe reads for a set of records: the pack, and with metadata the
/// `.inputs` and `.vinputs` files next to it.
pub fn write_inputs(pack_path: &Path, records: &[Record], meta: Option<&[Value]>) -> Result<(), String> {
    fs::write(pack_path, pack::write(records.iter().map(|r| r.bytes.as_slice()))).map_err(io(pack_path))?;
    let Some(meta) = meta else { return Ok(()) };
    let row = |r: &Record| -> Result<&Value, String> {
        let m = meta.get(r.index).ok_or("IndexError: list index out of range")?;
        meta::check_name(m, &r.name)?;
        Ok(m)
    };
    let mut inputs = String::new();
    for r in records {
        inputs += &(meta::inputs_line(row(r)?)? + "\n");
    }
    let path = PathBuf::from(format!("{}.inputs", pack_path.display()));
    fs::write(&path, py::native_newlines(&inputs)).map_err(io(&path))?;
    let mut vinputs = String::new();
    for r in records {
        vinputs += &(meta::vinputs_line(row(r)?)? + "\n");
    }
    let path = PathBuf::from(format!("{}.vinputs", pack_path.display()));
    fs::write(&path, py::native_newlines(&vinputs)).map_err(io(&path))
}

fn read_log(log: &Path) -> Result<String, String> {
    Ok(py::read_text(&fs::read(log).map_err(io(log))?))
}

/// Measures one batch and returns its rows.
pub fn run(out: &Path, job: &Job, meta: Option<&[Value]>, mode: u8, tools: &dyn Tools) -> Result<Vec<Row>, String> {
    let folder = out.join(format!("{}-{:05}", job.stage, job.offset));
    fs::create_dir(&folder).map_err(io(&folder))?;
    let pack_path = folder.join("batch.pack");
    let log = folder.join("batch.tsv");
    write_inputs(&pack_path, job.records, meta)?;

    let mut first = 0;
    while first < job.records.len() {
        let error = match tools.probe(&pack_path, &log, first, job.stage, &folder, mode)? {
            Ended::Exit(0) => break,
            Ended::Exit(code) => code.to_string(),
            Ended::Timeout => format!("timeout {}s", PROBE_TIMEOUT.as_secs()),
        };
        let text = read_log(&log)?;
        let mut last = None;
        for line in py::splitlines(&text) {
            if line.starts_with("B\t") {
                last = Some(py::int(line.split('\t').nth(1).unwrap_or_default())?);
            }
        }
        let last = match last {
            Some(l) if l >= first as i64 => l,
            _ => return Err(format!("RuntimeError: (({}, {}), {first}, {})", job.stage, job.offset, py::repr_str(&error))),
        };
        let mut file = OpenOptions::new().append(true).create(true).open(&log).map_err(io(&log))?;
        file.write_all(py::native_newlines(&format!("X\t{last}\t{error}\n")).as_bytes()).map_err(io(&log))?;
        first = last as usize + 1;
    }
    fs::remove_file(&pack_path).map_err(io(&pack_path))?;

    let mut rows: Vec<Row> = job
        .records
        .iter()
        .map(|r| Row(vec![("pack_id".into(), Field::Int(r.index as i64)), ("name".into(), Field::Str(r.name.clone())), ("stage".into(), Field::Int(i64::from(job.stage)))]))
        .collect();
    let text = read_log(&log)?;
    for line in py::splitlines(&text) {
        let mut parts = line.split('\t');
        let (Some(tag), Some(i)) = (parts.next(), parts.next()) else {
            return Err("ValueError: not enough values to unpack (expected at least 2, got 1)".into());
        };
        let i = py::int(i)?;
        let len = rows.len() as i64;
        let slot = if i < 0 { i + len } else { i };
        let row = usize::try_from(slot).ok().and_then(|s| rows.get_mut(s)).ok_or("IndexError: list index out of range")?;
        row.set(tag, Field::List(parts.map(str::to_owned).collect()));
    }
    if tools.validates() {
        for (i, row) in rows.iter_mut().enumerate() {
            // Only log lines set "C", so it is always a list.
            let compiled = match row.get("C") {
                Some(Field::List(values)) => values.first().ok_or("IndexError: list index out of range")? == "OK",
                _ => false,
            };
            if compiled {
                let v = match tools.validate(&folder.join(format!("{i}.spv")))? {
                    Validation::Ok => vec!["OK".to_owned()],
                    Validation::Fail(message) => vec!["FAIL".to_owned(), message],
                    Validation::Timeout => vec!["TIMEOUT".to_owned()],
                };
                row.set("V", Field::List(v));
            }
        }
    }
    Ok(rows)
}

/// Runs every job on [`WORKERS`] threads and hands the rows to `sink` in job order. After a
/// failed job no new job starts and the rows of later jobs are not handed over, as with
/// `ThreadPoolExecutor.map`. The jobs already running finish first.
pub fn run_all(
    out: &Path,
    jobs: &[Job],
    meta: Option<&[Value]>,
    mode: u8,
    tools: &dyn Tools,
    sink: &mut dyn FnMut(&[Row]) -> Result<(), String>,
) -> Result<(), String> {
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let (send, receive) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..WORKERS.min(jobs.len()) {
            let send = send.clone();
            let (next, stop) = (&next, &stop);
            scope.spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    let Some(job) = jobs.get(n) else { break };
                    if send.send((n, run(out, job, meta, mode, tools))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(send);
        let mut waiting = BTreeMap::new();
        let mut written = 0;
        let mut failure = None;
        for (n, result) in receive {
            waiting.insert(n, result);
            while failure.is_none()
                && let Some(result) = waiting.remove(&written)
            {
                match result.and_then(|rows| sink(&rows)) {
                    Ok(()) => written += 1,
                    Err(e) => {
                        stop.store(true, Ordering::SeqCst);
                        failure = Some(e);
                    }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use decant_extract_shaders::fssm::Code;
    use decant_extract_shaders::output::pack_record;
    use serde_json::json;

    fn record(index: usize, name: &str, stage: &str) -> Record {
        let mut bytes = Vec::new();
        let code = Code { sub_header: [index as u8; 16], words: vec![index as u32; 3], padding: 0 };
        pack_record(&mut bytes, name, stage, &code).unwrap();
        Record { index, name: name.into(), stage: decant_extract_shaders::output::stage_code(stage), bytes }
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("decant-shader-batch-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// An invented probe. It writes a `B` line per shader, then `T` and `C` lines, crashes on
    /// shader names starting with `crash`, hangs on names starting with `hang`, and writes a module
    /// file for each compiled shader.
    struct Fake {
        calls: Mutex<Vec<usize>>,
        validator: bool,
    }

    fn fake(validator: bool) -> Fake {
        Fake { calls: Mutex::new(Vec::new()), validator }
    }

    impl Tools for Fake {
        fn probe(&self, pack: &Path, log: &Path, first: usize, stage: u8, folder: &Path, mode: u8) -> Result<Ended, String> {
            assert_eq!((stage, mode), (0, 1));
            self.calls.lock().unwrap().push(first);
            let (header, records) = pack::read(pack, |_, _, _| true).unwrap();
            assert!(header.is_current());
            let mut file = OpenOptions::new().append(true).create(true).open(log).unwrap();
            for (i, name) in records.iter().map(|r| &r.name).enumerate().skip(first) {
                writeln!(file, "B\t{i}\t{name}\t64\t3").unwrap();
                if name.starts_with("crash") {
                    return Ok(Ended::Exit(3_221_225_477));
                }
                if name.starts_with("hang") {
                    return Ok(Ended::Timeout);
                }
                writeln!(file, "T\t{i}\tOK").unwrap();
                let compiled = if name.starts_with("bad") { "FAIL\tpc=0x1f0 unknown op 12" } else { "OK" };
                writeln!(file, "C\t{i}\t{compiled}").unwrap();
                fs::write(folder.join(format!("{i}.spv")), name).unwrap();
            }
            Ok(Ended::Exit(0))
        }

        fn validates(&self) -> bool {
            self.validator
        }

        fn validate(&self, module: &Path) -> Result<Validation, String> {
            let text = fs::read_to_string(module).unwrap();
            Ok(if text.starts_with("invalid") { Validation::Fail("error: line 1: ID 5 \u{e9}\n".into()) } else { Validation::Ok })
        }
    }

    #[test]
    fn crashes_and_hangs_restart_after_the_named_shader() {
        let out = temp("restart");
        let names = ["ok0", "crash1", "ok2", "hang3", "invalid4", "bad5"];
        let records: Vec<Record> = names.iter().enumerate().map(|(i, n)| record(10 + i, n, "pixel")).collect();
        let fake = fake(true);
        let meta: Vec<Value> = (0..16)
            .map(|i| {
                let name = if i >= 10 { names[i - 10] } else { "other" };
                json!({"name": name, "threads": null, "wave": 64, "mesh_limits": [0, 0], "inputs": [["TEXCOORD", 0, 1, 3, 3, 0, 3, 0, 0]]})
            })
            .collect();
        let job = Job { stage: 0, offset: 0, records: &records };
        let rows = run(&out, &job, Some(&meta), 1, &fake).unwrap();
        assert_eq!(*fake.calls.lock().unwrap(), [0, 2, 4]);

        let folder = out.join("0-00000");
        assert!(!folder.join("batch.pack").exists());
        let nl = if cfg!(windows) { "\r\n" } else { "\n" };
        assert_eq!(fs::read_to_string(folder.join("batch.pack.inputs")).unwrap(), format!("0 0 0 64 0 0{nl}").repeat(6));
        let log = fs::read_to_string(folder.join("batch.tsv")).unwrap();
        assert!(log.contains(&format!("X\t1\t3221225477{nl}")) && log.contains(&format!("X\t3\ttimeout 45s{nl}")));

        let dumped: Vec<String> = rows.iter().map(Row::dumps).collect();
        assert_eq!(dumped[0], r#"{"pack_id": 10, "name": "ok0", "stage": 0, "B": ["ok0", "64", "3"], "T": ["OK"], "C": ["OK"], "V": ["OK"]}"#);
        assert_eq!(dumped[1], r#"{"pack_id": 11, "name": "crash1", "stage": 0, "B": ["crash1", "64", "3"], "X": ["3221225477"]}"#);
        assert_eq!(dumped[3], r#"{"pack_id": 13, "name": "hang3", "stage": 0, "B": ["hang3", "64", "3"], "X": ["timeout 45s"]}"#);
        assert_eq!(
            dumped[4],
            r#"{"pack_id": 14, "name": "invalid4", "stage": 0, "B": ["invalid4", "64", "3"], "T": ["OK"], "C": ["OK"], "V": ["FAIL", "error: line 1: ID 5 \u00e9\n"]}"#
        );
        assert_eq!(dumped[5], r#"{"pack_id": 15, "name": "bad5", "stage": 0, "B": ["bad5", "64", "3"], "T": ["OK"], "C": ["FAIL", "pc=0x1f0 unknown op 12"]}"#);
        fs::remove_dir_all(&out).unwrap();
    }

    /// A probe that ends with `code` and writes an empty log or none.
    struct Silent {
        code: i64,
        log: bool,
    }

    impl Tools for Silent {
        fn probe(&self, _: &Path, log: &Path, _: usize, _: u8, _: &Path, _: u8) -> Result<Ended, String> {
            if self.log {
                fs::write(log, "").unwrap();
            }
            Ok(Ended::Exit(self.code))
        }
        fn validates(&self) -> bool {
            false
        }
        fn validate(&self, _: &Path) -> Result<Validation, String> {
            unreachable!()
        }
    }

    #[test]
    fn a_crash_before_any_log_line_stops_the_batch() {
        let out = temp("stuck");
        let records = vec![record(0, "crash0", "pixel")];
        let job = Job { stage: 0, offset: 48, records: &records };
        assert_eq!(run(&out, &job, None, 1, &Silent { code: 7, log: true }).unwrap_err(), "RuntimeError: ((0, 48), 0, '7')");
        let job = Job { stage: 0, offset: 96, records: &records };
        assert!(run(&out, &job, None, 1, &Silent { code: 0, log: false }).unwrap_err().contains("batch.tsv"));
        let job = Job { stage: 0, offset: 144, records: &records };
        let rows = run(&out, &job, None, 1, &Silent { code: 0, log: true }).unwrap();
        assert_eq!(rows[0].dumps(), r#"{"pack_id": 0, "name": "crash0", "stage": 0}"#);
        fs::remove_dir_all(&out).unwrap();
    }

    #[test]
    fn jobs_follow_the_stage_order_in_batches_of_48() {
        let records: Vec<Record> = (0..120).map(|i| record(i, "s", if i % 3 == 0 { "vertex" } else { "pixel" })).collect();
        let buckets = group(&[1, 0, 1, 5], records);
        let list: Vec<(u8, usize, usize)> = jobs(&buckets).iter().map(|j| (j.stage, j.offset, j.records.len())).collect();
        assert_eq!(list, [(1, 0, 40), (0, 0, 48), (0, 48, 32)]);
        assert_eq!((mode(true, true), mode(false, true), mode(false, false)), (2, 1, 0));
    }

    #[test]
    fn rows_reach_the_sink_in_job_order_until_a_failure() {
        let out = temp("order");
        let records: Vec<Record> = (0..300).map(|i| record(i, &format!("ok{i}"), "pixel")).collect();
        let fake = fake(false);
        let buckets = group(&[0], records);
        let list = jobs(&buckets);
        let mut seen = Vec::new();
        run_all(&out, &list, None, 1, &fake, &mut |rows| {
            seen.push(rows.len());
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, [48, 48, 48, 48, 48, 48, 12]);

        let again = temp("order2");
        fs::create_dir(again.join("0-00096")).unwrap();
        let mut seen = Vec::new();
        let error = run_all(&again, &list, None, 1, &fake, &mut |rows| {
            seen.push(rows.len());
            Ok(())
        })
        .unwrap_err();
        assert!(error.contains("0-00096"));
        assert_eq!(seen, [48, 48]);
        fs::remove_dir_all(&out).unwrap();
        fs::remove_dir_all(&again).unwrap();
    }
}
