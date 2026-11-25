#![allow(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn temp_log_file(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("decant_bench_{name}_{}.log", std::process::id()));
    dir
}

#[test]
fn session_log_repeated_lines_are_counted_not_written() {
    let _guard = TEST_LOCK.lock().unwrap();
    decant_log::reset();

    let log_path = temp_log_file("repeat");
    let _ = fs::remove_file(&log_path);
    unsafe {
        std::env::set_var("DECANT_LOG_PATH", &log_path);
        std::env::set_var("DECANT_LOG_RATE_LIMIT", "1");
    }

    let line = "[decant_xgr] QueryApiImpl(api XNetworking) -> 0x0";
    for _ in 0..10 {
        decant_log::session_log(line);
    }
    decant_log::flush();

    let content = fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0], line);
    assert_eq!(lines[1], "repeated 9 times");

    let _ = fs::remove_file(log_path);
}

#[test]
fn session_log_error_line_is_always_written() {
    let _guard = TEST_LOCK.lock().unwrap();
    decant_log::reset();

    let log_path = temp_log_file("error");
    let _ = fs::remove_file(&log_path);
    unsafe {
        std::env::set_var("DECANT_LOG_PATH", &log_path);
        std::env::set_var("DECANT_LOG_RATE_LIMIT", "1");
    }

    let err = "error: failed to bind socket";
    for _ in 0..5 {
        decant_log::session_log(err);
    }
    decant_log::flush();

    let content = fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 5);
    for l in lines {
        assert_eq!(l, err);
    }

    let _ = fs::remove_file(log_path);
}

#[test]
fn session_log_different_line_ends_run_with_summary() {
    let _guard = TEST_LOCK.lock().unwrap();
    decant_log::reset();

    let log_path = temp_log_file("different");
    let _ = fs::remove_file(&log_path);
    unsafe {
        std::env::set_var("DECANT_LOG_PATH", &log_path);
        std::env::set_var("DECANT_LOG_RATE_LIMIT", "1");
    }

    let line_a = "[decant_xgr] QueryApiImpl(api XNetworking) -> 0x0";
    let line_b = "[decant_xgr] QueryApiImpl(api XUser) -> 0x0";

    for _ in 0..4 {
        decant_log::session_log(line_a);
    }
    decant_log::session_log(line_b);
    decant_log::flush();

    let content = fs::read_to_string(&log_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], line_a);
    assert_eq!(lines[1], "repeated 3 times");
    assert_eq!(lines[2], line_b);

    let _ = fs::remove_file(log_path);
}

#[test]
#[ignore]
fn bench_80_lines_per_second_for_10_seconds() {
    let _guard = TEST_LOCK.lock().unwrap();
    decant_log::reset();

    let sample = "[decant_xgr] QueryApiImpl(api {37E56907-2F10-41E8-B72F-36EDB185331A} = XNetworking [hypothesis], riid {BF2346B2-39AF-4658-B5EA-44713C7E83B3} = riid {bf2346b2-...}) -> 0x00000000, table 00007FFF9F43E3E8, local XNetworking table (no PC runtime)";
    let total_calls = 800usize;
    let call_interval = Duration::from_micros(12500);

    // Phase 1: Before rate limiting
    let before_path = temp_log_file("bench_before");
    let _ = fs::remove_file(&before_path);
    unsafe {
        std::env::set_var("DECANT_LOG_PATH", &before_path);
        std::env::set_var("DECANT_LOG_RATE_LIMIT", "0");
    }

    let bench_start = Instant::now();
    let mut total_time_before = Duration::ZERO;
    for i in 0..total_calls {
        let call_start = Instant::now();
        decant_log::session_log(sample);
        total_time_before += call_start.elapsed();

        let expected_elapsed = call_interval * (i as u32 + 1);
        let actual_elapsed = bench_start.elapsed();
        if actual_elapsed < expected_elapsed {
            std::thread::sleep(expected_elapsed - actual_elapsed);
        }
    }

    let before_content = fs::read_to_string(&before_path).unwrap();
    let lines_written_before = before_content.lines().count();
    let bytes_before = before_content.len();
    let time_per_call_before_ns = total_time_before.as_nanos() as f64 / total_calls as f64;
    let time_per_call_before_us = time_per_call_before_ns / 1000.0;

    // Phase 2: After rate limiting
    decant_log::reset();
    let after_path = temp_log_file("bench_after");
    let _ = fs::remove_file(&after_path);
    unsafe {
        std::env::set_var("DECANT_LOG_PATH", &after_path);
        std::env::set_var("DECANT_LOG_RATE_LIMIT", "1");
    }

    let bench_start = Instant::now();
    let mut total_time_after = Duration::ZERO;
    for i in 0..total_calls {
        let call_start = Instant::now();
        decant_log::session_log(sample);
        total_time_after += call_start.elapsed();

        let expected_elapsed = call_interval * (i as u32 + 1);
        let actual_elapsed = bench_start.elapsed();
        if actual_elapsed < expected_elapsed {
            std::thread::sleep(expected_elapsed - actual_elapsed);
        }
    }
    decant_log::flush();

    let after_content = fs::read_to_string(&after_path).unwrap();
    let lines_written_after = after_content.lines().count();
    let bytes_after = after_content.len();
    let time_per_call_after_ns = total_time_after.as_nanos() as f64 / total_calls as f64;
    let time_per_call_after_us = time_per_call_after_ns / 1000.0;

    println!("\n================== LOG COST BENCHMARK ==================");
    println!("Benchmark parameters: 80 calls/s over 10 s (800 calls total)");
    println!("Before (rate limit off):");
    println!("  Lines written:       {lines_written_before}");
    println!("  Total bytes written: {bytes_before} bytes");
    println!("  Total log call time: {total_time_before:?}");
    println!("  Time spent per call: {time_per_call_before_us:.3} us ({time_per_call_before_ns:.1} ns)");
    println!("After (rate limit on):");
    println!("  Lines written:       {lines_written_after}");
    println!("  Total bytes written: {bytes_after} bytes");
    println!("  Total log call time: {total_time_after:?}");
    println!("  Time spent per call: {time_per_call_after_us:.3} us ({time_per_call_after_ns:.1} ns)");
    let line_reduction = (1.0 - (lines_written_after as f64 / lines_written_before as f64)) * 100.0;
    let time_reduction = (1.0 - (time_per_call_after_ns / time_per_call_before_ns)) * 100.0;
    println!("Reduction:");
    println!("  Lines written reduction: {line_reduction:.1}%");
    println!("  Time per call reduction: {time_reduction:.1}%");
    println!("========================================================\n");

    let _ = fs::remove_file(before_path);
    let _ = fs::remove_file(after_path);
}
