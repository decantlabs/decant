//! `DECANT_GENTLE=1` against the stand-in title `tools_probe`: its CPU and GPU scheduling priority
//! classes are read back from outside while it runs.

#![cfg(all(windows, target_arch = "x86_64"))]
#![allow(unsafe_code)]

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use decant_launch::run::{Sink, run_cancellable};
use windows_sys::Win32::Foundation::CloseHandle;
use windows_sys::Win32::System::Threading::{BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, GetPriorityClass, OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_SET_INFORMATION};

struct Lines(Arc<Mutex<Vec<String>>>);

impl Sink for Lines {
    fn line(&mut self, text: &str) {
        self.0.lock().unwrap().push(text.to_owned());
    }
}

/// Runs the probe in `gpu` mode (it opens a DXGI adapter a second after its start, from when the
/// kernel keeps GPU state for it) with `DECANT_GENTLE=<switch>`, reads its classes while it waits and
/// ends it. The GPU class is read once it is readable (the probe has used the GPU) and, with the
/// switch on, once the launcher has lowered it. Returns (CPU class, GPU class, log).
fn classes(switch: &str) -> (u32, Result<i32, String>, Vec<String>) {
    let dir = tempfile::tempdir().unwrap();
    let folder = std::fs::canonicalize(dir.path()).unwrap();
    let folder = folder.to_str().and_then(|s| s.strip_prefix(r"\\?\")).map_or(folder.clone(), std::path::PathBuf::from);
    let exe = folder.join("tools_probe.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_tools_probe"), &exe).unwrap();
    let done = folder.join("done.flag");
    let args = vec!["gpu".to_owned(), done.display().to_string()];
    let env = vec![("DECANT_GENTLE".to_owned(), switch.to_owned())];
    let lines = Arc::new(Mutex::new(Vec::new()));
    let mut sink = Lines(lines.clone());
    let seen = std::thread::scope(|s| {
        let runner = s.spawn(|| run_cancellable(&exe, &args, &folder, &env, Some(Duration::from_secs(150)), &AtomicBool::new(false), &mut sink).unwrap());
        let end = Instant::now() + Duration::from_secs(100);
        let mut pid = None;
        while pid.is_none() && Instant::now() < end && !runner.is_finished() {
            pid = lines.lock().unwrap().iter().find_map(|l| l.split("(process ").nth(1).and_then(|r| r.trim_end_matches(')').parse::<u32>().ok()));
            std::thread::sleep(Duration::from_millis(20));
        }
        let pid = pid.expect("the start line");
        // SAFETY: a query handle of a process of this test, closed below.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_SET_INFORMATION, 0, pid) };
        assert!(!handle.is_null(), "OpenProcess {pid}");
        let wanted = if switch == "1" { Some(decant_launch::gentle::GPU_BELOW_NORMAL) } else { None };
        let mut gpu = decant_launch::gentle::gpu_class(handle);
        let end = Instant::now() + Duration::from_secs(60);
        while Instant::now() < end && !runner.is_finished() && !gpu.as_ref().is_ok_and(|c| wanted.is_none_or(|w| *c == w)) {
            std::thread::sleep(Duration::from_millis(50));
            gpu = decant_launch::gentle::gpu_class(handle);
        }
        // SAFETY: the handle is valid.
        let cpu = unsafe { GetPriorityClass(handle) };
        // SAFETY: closing our handle.
        unsafe { CloseHandle(handle) };
        std::fs::write(&done, "").unwrap();
        runner.join().unwrap();
        (cpu, gpu)
    });
    let log = lines.lock().unwrap().clone();
    (seen.0, seen.1, log)
}

#[test]
fn the_title_runs_below_normal_with_the_switch() {
    let (cpu, gpu, log) = classes("1");
    assert_eq!(cpu, BELOW_NORMAL_PRIORITY_CLASS, "{log:#?}");
    assert_eq!(gpu, Ok(decant_launch::gentle::GPU_BELOW_NORMAL), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("DECANT_GENTLE: CPU priority below normal, GPU scheduling priority") && l.contains("memory priority left alone")), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("DECANT_GENTLE: GPU scheduling priority below normal, ")), "{log:#?}");
}

#[test]
fn the_title_keeps_its_priority_without_the_switch() {
    let (cpu, gpu, log) = classes("0");
    // The title inherits the priority class of this test process, whatever it is.
    // SAFETY: the pseudo handle of this process needs no close.
    let inherited = unsafe { GetPriorityClass(GetCurrentProcess()) };
    assert_ne!(inherited, 0, "GetPriorityClass of the test process");
    assert_eq!(cpu, inherited, "{log:#?}");
    assert_eq!(gpu, Ok(2), "{log:#?}");
    assert!(!log.iter().any(|l| l.contains("DECANT_GENTLE:")), "{log:#?}");
}
