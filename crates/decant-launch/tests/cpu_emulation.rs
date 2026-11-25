//! The debugger emulates CLZERO for a thread that raises an illegal instruction.

#![cfg(all(windows, target_arch = "x86_64"))]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use decant_launch::run::run_cancellable;

fn run_probe(mode: &str, env: &[(String, String)]) -> (Option<u32>, Vec<String>) {
    let exe = Path::new(env!("CARGO_BIN_EXE_clzero_probe"));
    let mut log: Vec<String> = Vec::new();
    let outcome = run_cancellable(exe, &[mode.to_owned()], exe.parent().unwrap(), env, Some(Duration::from_secs(60)), &AtomicBool::new(false), &mut log).unwrap();
    (outcome.exit_code, log)
}

fn skip() -> bool {
    if decant_launch::cpu_emu::cpu_has_clzero() {
        eprintln!("skipped: this CPU runs CLZERO itself (CPUID 0x80000008 EBX bit 0), nothing to emulate");
        return true;
    }
    false
}

#[test]
fn a_clzero_loop_is_zeroed_in_one_step() {
    if skip() {
        return;
    }
    let (code, log) = run_probe("fast", &[]);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("emulated clzero 16,")), "{log:#?}");
    assert_eq!(log.iter().filter(|l| l.contains("emulate clzero at")).count(), 1, "{log:#?}");
}

#[test]
fn an_unknown_loop_is_emulated_per_instruction() {
    if skip() {
        return;
    }
    let (code, log) = run_probe("slow", &[]);
    assert_eq!(code, Some(0), "{log:#?}");
    assert!(log.iter().any(|l| l.contains("emulated clzero 16,")), "{log:#?}");
}

#[test]
fn switched_off_the_exception_stays_unhandled() {
    if skip() {
        return;
    }
    let (code, log) = run_probe("fast", &[("DECANT_CPU_EMULATION".into(), "0".into())]);
    assert_eq!(code, Some(0xC000_001D), "{log:#?}");
    assert!(!log.iter().any(|l| l.contains("emulated")), "{log:#?}");
}
