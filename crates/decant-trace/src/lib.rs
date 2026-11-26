//! Low overhead tracing for the Decant layers.
//!
//! Tracing is off unless the environment variable `DECANT_TRACE` is set. `1` writes `decant_trace.json`
//! in the working directory and any other value is the path of the file. While tracing is
//! off, every call here costs one relaxed load of an `AtomicBool` and allocates nothing.
//!
//! When tracing is on, each thread records into its own buffer. The buffers are written as one Chrome and
//! Perfetto JSON document by `flush` and when the process exits. The `tracy` feature also forwards spans,
//! plots and frame marks to the Tracy profiler. See the README of the crate.

use std::cell::OnceCell;
use std::io;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock, PoisonError};
use std::time::Instant;

mod perfetto;
mod tracy_bridge;

use perfetto::{Event, Kind, ThreadView};
pub use perfetto::{FRAME_CAT, LOCK_WAIT_CAT, LockStat, SpanStat, StutterFrame, Summary, ThreadStat, WindowStat, summarize, summarize_range};

/// The environment variable that turns tracing on.
pub const ENV_VAR: &str = "DECANT_TRACE";
/// The file written when `DECANT_TRACE` is `1`.
pub const DEFAULT_FILE: &str = "decant_trace.json";
/// Events one thread keeps. Later events are counted as dropped and reported in the trace.
pub const MAX_EVENTS_PER_THREAD: usize = if cfg!(test) { 1000 } else { 1 << 21 };
/// True when the crate was built with the `tracy` feature.
pub const TRACY_BUILT: bool = tracy_bridge::ENABLED;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ENV_READ: Once = Once::new();
static ATEXIT: Once = Once::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();
static NEXT_TID: AtomicU32 = AtomicU32::new(1);
static STATE: Mutex<State> = Mutex::new(State { path: None, threads: Vec::new() });

struct State {
    path: Option<PathBuf>,
    threads: Vec<Arc<Mutex<ThreadBuf>>>,
}

struct ThreadBuf {
    tid: u32,
    name: String,
    events: Vec<Event>,
    dropped: u64,
}

thread_local! {
    static LOCAL: OnceCell<Arc<Mutex<ThreadBuf>>> = const { OnceCell::new() };
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads the value of `DECANT_TRACE`. `None` means tracing stays off.
fn path_from_env(value: Option<&str>) -> Option<PathBuf> {
    match value.map(str::trim) {
        None | Some("" | "0") => None,
        Some("1") => Some(PathBuf::from(DEFAULT_FILE)),
        Some(path) => Some(PathBuf::from(path)),
    }
}

fn read_env() {
    let value = std::env::var(ENV_VAR).ok();
    if let Some(path) = path_from_env(value.as_deref()) {
        activate(path);
    }
}

/// True when events are being recorded. The first call reads `DECANT_TRACE`.
#[inline]
pub fn enabled() -> bool {
    ENV_READ.call_once(read_env);
    ENABLED.load(Ordering::Relaxed)
}

/// Starts recording into `path` whatever `DECANT_TRACE` says, and drops anything recorded before.
/// The file is written by `flush` and at process exit.
pub fn start(path: impl Into<PathBuf>) {
    ENV_READ.call_once(|| {});
    activate(path.into());
}

/// Stops recording. What was recorded stays available to `flush` and `render_json`.
pub fn stop() {
    ENV_READ.call_once(|| {});
    ENABLED.store(false, Ordering::Relaxed);
}

fn activate(path: PathBuf) {
    EPOCH.get_or_init(Instant::now);
    {
        let mut state = lock(&STATE);
        state.path = Some(path);
        // Buffers of threads that ended are dropped, the others are emptied.
        state.threads.retain(|t| Arc::strong_count(t) > 1);
        for t in &state.threads {
            let mut t = lock(t);
            t.events.clear();
            t.dropped = 0;
        }
    }
    register_atexit();
    tracy_bridge::start();
    ENABLED.store(true, Ordering::Relaxed);
}

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> i32;
}

extern "C" fn write_at_exit() {
    let _ = flush();
}

fn register_atexit() {
    ATEXIT.call_once(|| {
        // SAFETY: `write_at_exit` is a plain function that stays valid for the life of the process and
        // does not unwind, as `flush` reports errors by value.
        let _ = unsafe { atexit(write_at_exit) };
    });
}

fn now_ns() -> u64 {
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn register_thread() -> Arc<Mutex<ThreadBuf>> {
    let tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
    let current = std::thread::current();
    let name = current.name().map_or_else(|| format!("thread {tid}"), str::to_owned);
    let buf = Arc::new(Mutex::new(ThreadBuf { tid, name, events: Vec::new(), dropped: 0 }));
    lock(&STATE).threads.push(Arc::clone(&buf));
    buf
}

fn record(e: Event) {
    let _ = LOCAL.try_with(|cell| {
        let buf = cell.get_or_init(register_thread);
        let mut buf = lock(buf);
        if buf.events.len() < MAX_EVENTS_PER_THREAD {
            buf.events.push(e);
        } else {
            buf.dropped += 1;
        }
    });
}

struct Live {
    cat: &'static str,
    name: &'static str,
    start_ns: u64,
    _zone: tracy_bridge::Zone,
}

/// A span that ends when the guard is dropped. Made by `span`.
#[must_use = "the span ends when the guard is dropped"]
pub struct SpanGuard {
    live: Option<Live>,
    // Spans belong to the thread that began them, which Tracy requires.
    _not_send: PhantomData<*const ()>,
}

impl SpanGuard {
    /// True when this guard records a span, false when tracing was off at `span`.
    pub fn is_recording(&self) -> bool {
        self.live.is_some()
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if let Some(live) = self.live.take() {
            let end = now_ns();
            record(Event { kind: Kind::Span, cat: live.cat, name: live.name, ts_ns: live.start_ns, dur_ns: end.saturating_sub(live.start_ns), value: 0.0 });
        }
    }
}

/// Begins a span of category `cat` and name `name` on the current thread.
#[inline]
pub fn span(cat: &'static str, name: &'static str) -> SpanGuard {
    let live = enabled().then(|| Live { cat, name, start_ns: now_ns(), _zone: tracy_bridge::Zone::enter(cat, name) });
    SpanGuard { live, _not_send: PhantomData }
}

/// Records that the current thread waited `micros` microseconds for the lock `name`, ending now.
#[inline]
pub fn record_lock_wait(name: &'static str, micros: u64) {
    if !enabled() {
        return;
    }
    let dur_ns = micros.saturating_mul(1000);
    let ts_ns = now_ns().saturating_sub(dur_ns);
    record(Event { kind: Kind::LockWait, cat: LOCK_WAIT_CAT, name, ts_ns, dur_ns, value: 0.0 });
    tracy_bridge::plot("lock wait us: ", name, micros as f64);
}

/// Runs `acquire`, usually a lock call, and records how long it took as a wait for the lock `name`.
#[inline]
pub fn timed_lock<R>(name: &'static str, acquire: impl FnOnce() -> R) -> R {
    if !enabled() {
        return acquire();
    }
    let began = Instant::now();
    let result = acquire();
    record_lock_wait(name, u64::try_from(began.elapsed().as_micros()).unwrap_or(u64::MAX));
    result
}

/// Records the value of the counter `name` now.
#[inline]
pub fn record_counter(name: &'static str, value: f64) {
    if !enabled() {
        return;
    }
    record(Event { kind: Kind::Counter, cat: "counter", name, ts_ns: now_ns(), dur_ns: 0, value });
    tracy_bridge::plot("", name, value);
}

/// Marks the end of a frame.
#[inline]
pub fn frame_mark() {
    if !enabled() {
        return;
    }
    record(Event { kind: Kind::Frame, cat: FRAME_CAT, name: FRAME_CAT, ts_ns: now_ns(), dur_ns: 0, value: 0.0 });
    tracy_bridge::frame();
}

/// Renders everything recorded so far as a Chrome and Perfetto JSON document.
pub fn render_json() -> String {
    let threads: Vec<Arc<Mutex<ThreadBuf>>> = lock(&STATE).threads.clone();
    let guards: Vec<MutexGuard<'_, ThreadBuf>> = threads.iter().map(|t| lock(t)).collect();
    let views: Vec<ThreadView<'_>> = guards.iter().map(|g| ThreadView { tid: g.tid, name: &g.name, events: &g.events, dropped: g.dropped }).collect();
    perfetto::render(std::process::id(), &views)
}

/// The file `flush` writes, when tracing was turned on.
pub fn output_path() -> Option<PathBuf> {
    lock(&STATE).path.clone()
}

/// Writes everything recorded so far to the trace file. Does nothing when tracing was never turned on.
/// Calling it again writes a longer trace over the same file.
pub fn flush() -> io::Result<()> {
    match output_path() {
        Some(path) => write_to(&path),
        None => Ok(()),
    }
}

fn write_to(path: &Path) -> io::Result<()> {
    let data = render_json();
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, &data).is_ok() && std::fs::rename(&tmp, path).is_ok() {
        return Ok(());
    }
    let _ = std::fs::remove_file(&tmp);
    std::fs::write(path, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The recorder is process wide, so the tests that turn it on run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        lock(&SERIAL)
    }

    fn events(json: &str) -> Vec<Value> {
        let v: Value = serde_json::from_str(json).unwrap();
        v["traceEvents"].as_array().unwrap().clone()
    }

    fn named<'a>(list: &'a [Value], ph: &str, name: &str) -> Vec<&'a Value> {
        list.iter().filter(|e| e["ph"] == ph && e["name"] == name).collect()
    }

    #[test]
    fn env_values_choose_the_file() {
        assert_eq!(path_from_env(None), None);
        assert_eq!(path_from_env(Some("")), None);
        assert_eq!(path_from_env(Some("0")), None);
        assert_eq!(path_from_env(Some(" 0 ")), None);
        assert_eq!(path_from_env(Some("1")), Some(PathBuf::from("decant_trace.json")));
        assert_eq!(path_from_env(Some("out/run.json")), Some(PathBuf::from("out/run.json")));
    }

    #[test]
    fn disabled_mode_records_and_registers_nothing() {
        let _g = serial();
        stop();
        let before = lock(&STATE).threads.len();
        std::thread::spawn(|| {
            let guard = span("cat", "off");
            assert!(!guard.is_recording());
            record_counter("off", 1.0);
            record_lock_wait("off", 5);
            frame_mark();
            assert_eq!(timed_lock("off", || 42), 42);
        })
        .join()
        .unwrap();
        assert_eq!(lock(&STATE).threads.len(), before);
        assert!(named(&events(&render_json()), "X", "off").is_empty());
        assert!(!enabled());
    }

    #[test]
    fn records_spans_counters_lock_waits_and_frames_from_two_threads() {
        let _g = serial();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.json");
        start(&path);
        assert!(enabled());
        {
            let outer = span("test", "outer");
            assert!(outer.is_recording());
            let _inner = span("test", "inner");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        record_counter("depth", 3.0);
        record_lock_wait("queue", 1500);
        frame_mark();
        frame_mark();
        std::thread::Builder::new()
            .name("worker".to_owned())
            .spawn(|| {
                let _s = span("test", "worker_job");
                record_counter("depth", 4.0);
            })
            .unwrap()
            .join()
            .unwrap();
        stop();
        flush().unwrap();

        let list = events(&std::fs::read_to_string(&path).unwrap());
        let outer = named(&list, "X", "outer");
        let inner = named(&list, "X", "inner");
        assert_eq!((outer.len(), inner.len()), (1, 1));
        let (o, i) = (outer[0], inner[0]);
        assert_eq!(o["cat"], "test");
        assert!(o["dur"].as_f64().unwrap() >= 2000.0);
        assert!(o["ts"].as_f64().unwrap() <= i["ts"].as_f64().unwrap());
        assert_eq!(o["tid"], i["tid"]);
        let wait = named(&list, "X", "queue");
        assert_eq!(wait[0]["cat"], LOCK_WAIT_CAT);
        assert_eq!(wait[0]["dur"].as_f64(), Some(1500.0));
        let depth = named(&list, "C", "depth");
        let mut values: Vec<f64> = depth.iter().map(|e| e["args"]["value"].as_f64().unwrap()).collect();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, [3.0, 4.0]);
        assert_eq!(named(&list, "i", "frame").len(), 2);
        let job = named(&list, "X", "worker_job")[0];
        assert_ne!(job["tid"], o["tid"]);
        let worker_name = list.iter().find(|e| e["ph"] == "M" && e["name"] == "thread_name" && e["tid"] == job["tid"]).unwrap();
        assert_eq!(worker_name["args"]["name"], "worker");

        let summary = summarize(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(summary.frames, 2);
        assert_eq!(summary.counter_samples, 2);
        assert_eq!(summary.locks[0].name, "queue");
        assert!(summary.spans.iter().any(|s| s.name == "worker_job"));
    }

    #[test]
    fn start_drops_earlier_events_and_stop_keeps_them_for_flush() {
        let _g = serial();
        let dir = tempfile::tempdir().unwrap();
        start(dir.path().join("a.json"));
        drop(span("test", "first_session"));
        start(dir.path().join("b.json"));
        drop(span("test", "second_session"));
        stop();
        drop(span("test", "after_stop"));
        let list = events(&render_json());
        assert!(named(&list, "X", "first_session").is_empty());
        assert_eq!(named(&list, "X", "second_session").len(), 1);
        assert!(named(&list, "X", "after_stop").is_empty());
        assert_eq!(output_path(), Some(dir.path().join("b.json")));
        flush().unwrap();
        assert!(dir.path().join("b.json").exists());
        assert!(!dir.path().join("a.json").exists());
    }

    #[test]
    fn a_span_that_started_before_stop_still_ends() {
        let _g = serial();
        start(std::env::temp_dir().join("decant_trace_unused.json"));
        let guard = span("test", "straddles_stop");
        stop();
        drop(guard);
        assert_eq!(named(&events(&render_json()), "X", "straddles_stop").len(), 1);
    }

    #[test]
    fn timed_lock_records_the_wait() {
        let _g = serial();
        start(std::env::temp_dir().join("decant_trace_unused.json"));
        let m = Mutex::new(5);
        let value = timed_lock("test_mutex", || *lock(&m));
        stop();
        assert_eq!(value, 5);
        let list = events(&render_json());
        assert_eq!(named(&list, "X", "test_mutex").len(), 1);
    }

    #[test]
    fn full_buffers_count_dropped_events() {
        let _g = serial();
        start(std::env::temp_dir().join("decant_trace_unused.json"));
        std::thread::spawn(|| {
            for _ in 0..MAX_EVENTS_PER_THREAD + 3 {
                record_counter("flood", 1.0);
            }
        })
        .join()
        .unwrap();
        stop();
        let summary = summarize(&render_json()).unwrap();
        assert_eq!(summary.dropped, 3);
        assert_eq!(summary.counter_samples, MAX_EVENTS_PER_THREAD as u64);
    }

    #[test]
    fn flush_without_a_session_path_is_a_no_op() {
        let _g = serial();
        let saved = lock(&STATE).path.take();
        assert!(flush().is_ok());
        lock(&STATE).path = saved;
    }

    #[cfg(feature = "tracy")]
    #[test]
    fn tracy_forwarding_does_not_disturb_recording() {
        let _g = serial();
        const { assert!(TRACY_BUILT) };
        start(std::env::temp_dir().join("decant_trace_unused.json"));
        drop(span("test", "tracy_span"));
        record_counter("tracy_plot", 1.0);
        record_lock_wait("tracy_lock", 3);
        frame_mark();
        stop();
        assert_eq!(named(&events(&render_json()), "X", "tracy_span").len(), 1);
    }

    #[cfg(not(feature = "tracy"))]
    #[test]
    fn tracy_is_not_built_by_default() {
        const { assert!(!TRACY_BUILT) };
    }
}
