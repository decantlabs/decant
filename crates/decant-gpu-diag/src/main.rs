//! Anonymous diagnostic runner for the native hardware tests, as docs/GPU_TESTING.md specifies.
//!
//! It lists the DXGI adapters with their D3D12 ray tracing tier, runs `loadtest.exe`,
//! `decant_bvh_test.exe` and `ffxtest.exe` from the build folder one after another, each with a 45 s
//! deadline on its process tree, and prints an allowlisted record in the order of the GPU form.
//! Nothing is written to disk or sent anywhere: the record is shown for review, and sharing it is
//! up to the person who ran it.
//!
//! Usage: `gpu_diag [--build <folder>] [--backend <FidelityFX backend DLL>] [--adapter <n>]
//! [--commit <text>] [--json]`

mod runner;

use decant_gpu_diag::{probe, record, sanitize};

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use record::{Adapter, Category, Component, Record, TestResult};
use runner::{DEADLINE, Run};
use sanitize::Scrubber;

struct Args {
    build: PathBuf,
    backend: Option<PathBuf>,
    adapter: usize,
    commit: Option<String>,
    json: bool,
}

fn parse() -> Option<Args> {
    let mut args = Args { build: PathBuf::from("build"), backend: None, adapter: 0, commit: None, json: false };
    let mut it = std::env::args_os().skip(1);
    while let Some(arg) = it.next() {
        match arg.to_str()? {
            "--build" => args.build = it.next()?.into(),
            "--backend" => args.backend = Some(it.next()?.into()),
            "--adapter" => args.adapter = it.next()?.to_str()?.parse().ok()?,
            "--commit" => args.commit = Some(it.next()?.into_string().ok()?),
            "--json" => args.json = true,
            _ => return None,
        }
    }
    Some(args)
}

fn main() -> ExitCode {
    let Some(args) = parse() else {
        eprintln!("usage: gpu_diag [--build <folder>] [--backend <FidelityFX backend DLL>] [--adapter <n>] [--commit <text>] [--json]");
        return ExitCode::from(2);
    };
    let build = std::path::absolute(&args.build).unwrap_or(args.build.clone());
    let adapters = probe::adapters(args.adapter);
    let tests = Component::ALL.iter().map(|&c| test(c, &build, args.backend.as_deref(), &adapters)).collect();
    let commit = args.commit.clone().or_else(|| git_commit(&build)).unwrap_or_else(|| "unknown".to_owned());
    let record = Record { tool: format!("gpu_diag {}, commit {commit}", env!("CARGO_PKG_VERSION")), system: probe::system(), adapters, tests };

    let mut local = vec![build.display().to_string(), args.build.display().to_string()];
    local.extend(args.backend.iter().map(|b| b.display().to_string()));
    local.extend(std::env::current_dir().ok().map(|d| d.display().to_string()));
    let record = record.scrubbed(&Scrubber::for_this_run(&local));

    println!("{}", if args.json { record.json() } else { record.markdown() });
    eprintln!("\nNothing was saved or sent. Read the record above before sharing it.");
    ExitCode::SUCCESS
}

/// Runs one component's test, or says why it cannot run here.
fn test(component: Component, build: &Path, backend: Option<&Path>, adapters: &[Adapter]) -> TestResult {
    let program = build.join(component.executable());
    let run = match component {
        Component::Xmem => {
            let dll = build.join("xmem.dll");
            if dll.is_file() { runner::run(&program, &[dll.into_os_string()], DEADLINE) } else { Run::unavailable(Category::MissingInput) }
        }
        Component::Dxr => match adapters.iter().find(|a| a.default) {
            Some(a) if a.raytracing_tier.unwrap_or(0) == 0 => Run::unavailable(Category::NoRaytracing),
            _ => runner::run(&program, &[], DEADLINE),
        },
        Component::FidelityFx => match backend.filter(|b| b.is_file()) {
            Some(backend) => runner::run(&program, &[OsString::from(build), OsString::from(backend)], DEADLINE),
            None => Run::unavailable(Category::MissingBackend),
        },
    };
    TestResult { component, outcome: run.outcome, elapsed_ms: run.elapsed_ms, category: run.category }
}

/// The short commit of the checkout the tested build lives in, when git is there to say.
fn git_commit(build: &Path) -> Option<String> {
    let out = Command::new("git").args(["rev-parse", "--short", "HEAD"]).current_dir(build).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (out.status.success() && !text.is_empty() && text.chars().all(|c| c.is_ascii_hexdigit())).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Outcome;

    fn adapter(tier: Option<u32>) -> Adapter {
        Adapter { name: "Invented GPU".into(), vendor_id: 1, device_id: 2, driver: None, raytracing_tier: tier, software: false, default: true, selected: true }
    }

    #[test]
    fn missing_inputs_make_each_component_unavailable_with_its_own_reason() {
        let empty = std::env::temp_dir().join(format!("gpu-diag-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let result = |c, adapters: &[Adapter]| {
            let r = test(c, &empty, Some(Path::new(r"Z:\no\backend.dll")), adapters);
            (r.outcome, r.category)
        };
        assert_eq!(result(Component::Xmem, &[adapter(Some(12))]), (Outcome::Unavailable, Some(Category::MissingInput)));
        assert_eq!(result(Component::Dxr, &[adapter(Some(0))]), (Outcome::Unavailable, Some(Category::NoRaytracing)));
        assert_eq!(result(Component::Dxr, &[adapter(Some(12))]), (Outcome::Unavailable, Some(Category::MissingExecutable)));
        assert_eq!(result(Component::FidelityFx, &[adapter(Some(12))]), (Outcome::Unavailable, Some(Category::MissingBackend)));
        std::fs::remove_dir_all(empty).unwrap();
    }
}
