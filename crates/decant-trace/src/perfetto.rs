//! The Chrome and Perfetto JSON side of the crate: the writer used by `flush` and the reader and
//! aggregation used by `decant_trace_summary`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::Value;

/// Category of the events written by `record_lock_wait`. The reader treats them as blocked time.
pub const LOCK_WAIT_CAT: &str = "lock_wait";
/// Category and name of the events written by `frame_mark`.
pub const FRAME_CAT: &str = "frame";

/// What an `Event` records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Span,
    LockWait,
    Counter,
    Frame,
}

/// One recorded event. Times are nanoseconds since the start of the session.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    pub kind: Kind,
    pub cat: &'static str,
    pub name: &'static str,
    pub ts_ns: u64,
    pub dur_ns: u64,
    pub value: f64,
}

/// A read only view of one thread buffer, as the writer needs it.
pub(crate) struct ThreadView<'a> {
    pub tid: u32,
    pub name: &'a str,
    pub events: &'a [Event],
    pub dropped: u64,
}

/// Appends `s` as a JSON string literal.
fn push_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Appends nanoseconds as microseconds with three decimals, which is how the trace format counts time.
fn push_micros(out: &mut String, ns: u64) {
    let _ = write!(out, "{}.{:03}", ns / 1000, ns % 1000);
}

/// Appends a number, writing 0 for values JSON cannot hold.
fn push_f64(out: &mut String, v: f64) {
    if v.is_finite() {
        let _ = write!(out, "{v}");
    } else {
        out.push('0');
    }
}

fn push_metadata(out: &mut String, first: &mut bool, name: &str, pid: u32, tid: u32, arg: &str, value: &str) {
    if !std::mem::take(first) {
        out.push(',');
    }
    out.push_str("\n{\"name\":");
    push_str(out, name);
    let _ = write!(out, ",\"ph\":\"M\",\"pid\":{pid},\"tid\":{tid},\"args\":{{\"{arg}\":");
    push_str(out, value);
    out.push_str("}}");
}

fn push_event(out: &mut String, pid: u32, tid: u32, e: &Event) {
    out.push_str(",\n{\"name\":");
    push_str(out, e.name);
    match e.kind {
        Kind::Span | Kind::LockWait => {
            out.push_str(",\"cat\":");
            push_str(out, if e.kind == Kind::LockWait { LOCK_WAIT_CAT } else { e.cat });
            out.push_str(",\"ph\":\"X\",\"ts\":");
            push_micros(out, e.ts_ns);
            out.push_str(",\"dur\":");
            push_micros(out, e.dur_ns);
        }
        Kind::Counter => {
            out.push_str(",\"cat\":\"counter\",\"ph\":\"C\",\"ts\":");
            push_micros(out, e.ts_ns);
        }
        Kind::Frame => {
            out.push_str(",\"cat\":\"frame\",\"ph\":\"i\",\"s\":\"g\",\"ts\":");
            push_micros(out, e.ts_ns);
        }
    }
    let _ = write!(out, ",\"pid\":{pid},\"tid\":{tid}");
    if e.kind == Kind::Counter {
        out.push_str(",\"args\":{\"value\":");
        push_f64(out, e.value);
        out.push('}');
    }
    out.push('}');
}

/// Renders the threads as one Chrome trace document. Events are ordered by time.
pub(crate) fn render(pid: u32, threads: &[ThreadView<'_>]) -> String {
    let total: usize = threads.iter().map(|t| t.events.len()).sum();
    let mut out = String::with_capacity(128 + total * 120);
    out.push_str("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[");
    let mut first = true;
    push_metadata(&mut out, &mut first, "process_name", pid, 0, "name", "decant");
    for t in threads {
        push_metadata(&mut out, &mut first, "thread_name", pid, t.tid, "name", t.name);
    }
    let dropped: u64 = threads.iter().map(|t| t.dropped).sum();
    if dropped > 0 {
        push_metadata(&mut out, &mut first, DROPPED_NAME, pid, 0, "events", &dropped.to_string());
    }
    let mut all: Vec<(u32, &Event)> = threads.iter().flat_map(|t| t.events.iter().map(move |e| (t.tid, e))).collect();
    all.sort_by_key(|(_, e)| e.ts_ns);
    for (tid, e) in all {
        push_event(&mut out, pid, tid, e);
    }
    out.push_str("\n]}\n");
    out
}

/// Name of the metadata event that carries the number of events lost to the buffer limit.
const DROPPED_NAME: &str = "decant_dropped_events";

/// Aggregate of the spans that share a category and a name.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanStat {
    /// Category of the span.
    pub cat: String,
    /// Name of the span.
    pub name: String,
    /// Number of spans seen.
    pub count: u64,
    /// Sum of their durations in microseconds. Nested spans are counted in the parent too.
    pub total_us: f64,
    /// Longest single duration in microseconds.
    pub max_us: f64,
}

/// Aggregate of the lock waits that share a name.
#[derive(Clone, Debug, PartialEq)]
pub struct LockStat {
    /// Name of the lock.
    pub name: String,
    /// Number of waits seen.
    pub count: u64,
    /// Sum of the waits in microseconds.
    pub total_us: f64,
    /// Longest single wait in microseconds.
    pub max_us: f64,
}

/// What one thread did.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreadStat {
    /// Process id of the thread.
    pub pid: u64,
    /// Thread id in the trace.
    pub tid: u64,
    /// Name from the `thread_name` metadata, or a generated one.
    pub name: String,
    /// Number of spans on the thread.
    pub spans: u64,
    /// Time covered by at least one span, in microseconds. Nested and overlapping spans count once.
    pub active_us: f64,
    /// Sum of the lock waits of the thread in microseconds.
    pub lock_wait_us: f64,
}

impl ThreadStat {
    /// Share of `wall_us` the thread was inside a span, from 0 to 1.
    pub fn utilization(&self, wall_us: f64) -> f64 {
        if wall_us > 0.0 { (self.active_us / wall_us).min(1.0) } else { 0.0 }
    }
}

/// Frame statistics for one 10-second window.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowStat {
    /// Window start in seconds relative to trace start.
    pub start_s: u64,
    /// Window end in seconds relative to trace start.
    pub end_s: u64,
    /// Number of frame marks in this window.
    pub frames: u64,
    /// Frames per second in this window.
    pub fps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// One worst stutter frame (`> 50 ms`) and its overlapping spans/locks.
#[derive(Clone, Debug, PartialEq)]
pub struct StutterFrame {
    /// Timestamp in seconds relative to trace start.
    pub at_s: f64,
    /// Frame interval duration in milliseconds.
    pub dur_ms: f64,
    /// Top overlapping non-lock spans `(label, overlap_ms)`.
    pub spans: Vec<(String, f64)>,
    /// Top overlapping lock waits `(lock_name, overlap_ms)`.
    pub locks: Vec<(String, f64)>,
}

/// Aggregate of the counter samples that share a name.
#[derive(Clone, Debug, PartialEq)]
pub struct CounterStat {
    pub name: String,
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub last: f64,
}

/// The result of reading a trace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    /// Number of events that carry a time (spans, counters and frame marks).
    pub events: u64,
    /// Time from the first event to the last end, in microseconds.
    pub wall_us: f64,
    /// Span aggregates, ordered by category and name.
    pub spans: Vec<SpanStat>,
    /// Lock wait aggregates, ordered by name.
    pub locks: Vec<LockStat>,
    /// Thread aggregates, busiest first.
    pub threads: Vec<ThreadStat>,
    /// 10-second windowed frame statistics.
    pub windows: Vec<WindowStat>,
    /// Top 15 worst stutter frames (`> 50 ms`).
    pub stutters: Vec<StutterFrame>,
    /// Counter aggregates, ordered by name.
    pub counters: Vec<CounterStat>,
    /// Number of frame marks.
    pub frames: u64,
    /// Mean time between frame marks in microseconds, 0 for fewer than two frames.
    pub frame_mean_us: f64,
    /// Overall frames per second across the frame span.
    pub fps: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub hitches_50ms: u64,
    pub hitches_100ms: u64,
    pub hitches_500ms: u64,
    /// Number of counter samples.
    pub counter_samples: u64,
    /// Events the recorder dropped because a thread buffer was full.
    pub dropped: u64,
}

#[derive(Default)]
struct Agg {
    count: u64,
    total: f64,
    max: f64,
}

impl Agg {
    fn add(&mut self, dur: f64) {
        self.count += 1;
        self.total += dur;
        self.max = self.max.max(dur);
    }
}

#[derive(Default)]
struct ThreadAcc {
    intervals: Vec<(f64, f64)>,
    lock_wait: f64,
}

/// Length of the union of the intervals.
fn union_length(intervals: &mut [(f64, f64)]) -> f64 {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let mut current: Option<(f64, f64)> = None;
    for &(start, end) in intervals.iter() {
        match current {
            Some((s, e)) if start <= e => current = Some((s, e.max(end))),
            Some((s, e)) => {
                total += e - s;
                current = Some((start, end));
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((s, e)) = current {
        total += e - s;
    }
    total
}

fn num(e: &Value, key: &str) -> f64 {
    e.get(key).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0)
}

fn id(e: &Value, key: &str) -> u64 {
    e.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn text<'a>(e: &'a Value, key: &str) -> &'a str {
    e.get(key).and_then(Value::as_str).unwrap_or("")
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn compute_windows(frame_ts: &[f64], t0: f64) -> Vec<WindowStat> {
    if frame_ts.is_empty() {
        return Vec::new();
    }
    let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
    let mut intervals: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for (i, &ts) in frame_ts.iter().enumerate() {
        let rel_s = ((ts - t0).max(0.0)) / 1_000_000.0;
        let bucket = (rel_s / 10.0).floor() as u64;
        *counts.entry(bucket).or_insert(0) += 1;
        if i > 0 {
            let dt_ms = (ts - frame_ts[i - 1]).max(0.0) / 1_000.0;
            intervals.entry(bucket).or_default().push(dt_ms);
        }
    }
    counts
        .into_iter()
        .map(|(bucket, frames)| {
            let mut dts = intervals.remove(&bucket).unwrap_or_default();
            dts.sort_by(f64::total_cmp);
            WindowStat {
                start_s: bucket * 10,
                end_s: (bucket + 1) * 10,
                frames,
                fps: frames as f64 / 10.0,
                p50_ms: percentile(&dts, 0.50),
                p95_ms: percentile(&dts, 0.95),
                p99_ms: percentile(&dts, 0.99),
                max_ms: dts.last().copied().unwrap_or(0.0),
            }
        })
        .collect()
}

fn compute_stutters(frame_ts: &[f64], t0: f64, span_intervals: &[(f64, f64, bool, String)]) -> Vec<StutterFrame> {
    const STUTTER_MIN_MS: f64 = 50.0;
    const TOP_STUTTERS: usize = 15;
    let mut worst: Vec<(f64, f64, f64)> = Vec::new();
    for i in 1..frame_ts.len() {
        let (prev, cur) = (frame_ts[i - 1], frame_ts[i]);
        let dt_ms = (cur - prev) / 1_000.0;
        if dt_ms > STUTTER_MIN_MS {
            worst.push((dt_ms, prev, cur));
        }
    }
    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    worst.truncate(TOP_STUTTERS);
    worst
        .into_iter()
        .map(|(dur_ms, prev, cur)| {
            let mut span_agg: BTreeMap<&str, f64> = BTreeMap::new();
            let mut lock_agg: BTreeMap<&str, f64> = BTreeMap::new();
            for (s, e, is_lock, label) in span_intervals {
                let ov = (e.min(cur) - s.max(prev)) / 1_000.0;
                if ov > 0.0 {
                    let target = if *is_lock { &mut lock_agg } else { &mut span_agg };
                    *target.entry(label.as_str()).or_insert(0.0) += ov;
                }
            }
            let mut spans: Vec<(String, f64)> = span_agg.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
            spans.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            spans.truncate(3);
            let mut locks: Vec<(String, f64)> = lock_agg.into_iter().map(|(k, v)| (k.to_owned(), v)).collect();
            locks.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            locks.truncate(3);
            StutterFrame { at_s: (cur - t0).max(0.0) / 1_000_000.0, dur_ms, spans, locks }
        })
        .collect()
}

/// Reads a Chrome or Perfetto JSON trace, either an object with `traceEvents` or a bare array.
pub fn summarize(json: &str) -> Result<Summary, String> {
    summarize_range(json, None, None)
}

/// Reads a Chrome or Perfetto JSON trace, optionally filtering events to `[from_s, to_s]` seconds
/// relative to the first event timestamp.
pub fn summarize_range(json: &str, from_s: Option<f64>, to_s: Option<f64>) -> Result<Summary, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| format!("not valid JSON: {e}"))?;
    let events: &Vec<Value> = match &root {
        Value::Array(a) => a,
        Value::Object(o) => o.get("traceEvents").and_then(Value::as_array).ok_or("no traceEvents array")?,
        _ => return Err("expected an object or an array".to_owned()),
    };
    let mut t0 = f64::INFINITY;
    for e in events {
        if text(e, "ph") != "M" {
            t0 = t0.min(num(e, "ts"));
        }
    }
    let from_us = from_s.map(|s| t0 + s * 1_000_000.0);
    let to_us = to_s.map(|s| t0 + s * 1_000_000.0);
    let in_range = |ts: f64, dur: f64| -> bool {
        let lo_ok = from_us.is_none_or(|lo| ts + dur >= lo);
        let hi_ok = to_us.is_none_or(|hi| ts <= hi);
        lo_ok && hi_ok
    };

    let mut spans: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut locks: BTreeMap<String, Agg> = BTreeMap::new();
    let mut counters: BTreeMap<String, (u64, f64, f64, f64, f64)> = BTreeMap::new();
    let mut threads: BTreeMap<(u64, u64), ThreadAcc> = BTreeMap::new();
    let mut names: BTreeMap<(u64, u64), String> = BTreeMap::new();
    let mut frame_ts: Vec<f64> = Vec::new();
    let mut span_intervals: Vec<(f64, f64, bool, String)> = Vec::new();
    let mut s = Summary::default();
    let (mut first, mut last) = (f64::INFINITY, 0.0_f64);
    for e in events {
        let key = (id(e, "pid"), id(e, "tid"));
        let ph = text(e, "ph");
        if ph == "M" {
            let arg = |k: &str| e.get("args").and_then(|a| a.get(k));
            match text(e, "name") {
                "thread_name" => {
                    if let Some(n) = arg("name").and_then(Value::as_str) {
                        names.insert(key, n.to_owned());
                    }
                }
                DROPPED_NAME => {
                    s.dropped += arg("events").and_then(Value::as_str).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
                }
                _ => {}
            }
            continue;
        }
        let (ts, dur) = (num(e, "ts"), num(e, "dur").max(0.0));
        if !in_range(ts, dur) {
            continue;
        }
        match ph {
            "X" => {
                let (cat, name) = (text(e, "cat"), text(e, "name"));
                let acc = threads.entry(key).or_default();
                if cat == LOCK_WAIT_CAT {
                    locks.entry(name.to_owned()).or_default().add(dur);
                    acc.lock_wait += dur;
                    span_intervals.push((ts, ts + dur, true, name.to_owned()));
                } else {
                    spans.entry((cat.to_owned(), name.to_owned())).or_default().add(dur);
                    acc.intervals.push((ts, ts + dur));
                    span_intervals.push((ts, ts + dur, false, format!("{cat}:{name}")));
                }
            }
            "C" => {
                s.counter_samples += 1;
                let cname = text(e, "name");
                if !cname.is_empty() {
                    let val = e.get("args").and_then(|a| a.get("value")).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0);
                    counters
                        .entry(cname.to_owned())
                        .and_modify(|(cnt, min, max, sum, last_v)| {
                            *cnt += 1;
                            *min = min.min(val);
                            *max = max.max(val);
                            *sum += val;
                            *last_v = val;
                        })
                        .or_insert((1, val, val, val, val));
                }
            }
            "i" | "I" if text(e, "cat") == FRAME_CAT => frame_ts.push(ts),
            "i" | "I" => {}
            _ => continue,
        }
        s.events += 1;
        first = first.min(ts);
        last = last.max(ts + dur);
    }
    s.wall_us = if first.is_finite() { (last - first).max(0.0) } else { 0.0 };
    s.spans = spans.into_iter().map(|((cat, name), a)| SpanStat { cat, name, count: a.count, total_us: a.total, max_us: a.max }).collect();
    s.locks = locks.into_iter().map(|(name, a)| LockStat { name, count: a.count, total_us: a.total, max_us: a.max }).collect();
    s.counters = counters
        .into_iter()
        .map(|(name, (count, min, max, sum, last))| CounterStat {
            name,
            count,
            min,
            max,
            mean: sum / count.max(1) as f64,
            last,
        })
        .collect();
    s.threads = threads
        .into_iter()
        .map(|(key, mut acc)| ThreadStat {
            pid: key.0,
            tid: key.1,
            name: names.get(&key).cloned().unwrap_or_else(|| format!("thread {}", key.1)),
            spans: u64::try_from(acc.intervals.len()).unwrap_or(u64::MAX),
            active_us: union_length(&mut acc.intervals),
            lock_wait_us: acc.lock_wait,
        })
        .collect();
    s.threads.sort_by(|a, b| b.active_us.total_cmp(&a.active_us).then(a.tid.cmp(&b.tid)));
    frame_ts.sort_by(f64::total_cmp);
    s.frames = u64::try_from(frame_ts.len()).unwrap_or(u64::MAX);
    if let (Some(&lo), Some(&hi)) = (frame_ts.first(), frame_ts.last())
        && frame_ts.len() > 1
    {
        s.frame_mean_us = (hi - lo) / (frame_ts.len() - 1) as f64;
        let span_s = (hi - lo) / 1_000_000.0;
        if span_s > 0.0 {
            s.fps = (frame_ts.len() - 1) as f64 / span_s;
        }
        let mut all_dts: Vec<f64> = frame_ts.windows(2).map(|w| (w[1] - w[0]).max(0.0) / 1_000.0).collect();
        for &dt in &all_dts {
            if dt > 50.0 {
                s.hitches_50ms += 1;
            }
            if dt > 100.0 {
                s.hitches_100ms += 1;
            }
            if dt > 500.0 {
                s.hitches_500ms += 1;
            }
        }
        all_dts.sort_by(f64::total_cmp);
        s.p50_ms = percentile(&all_dts, 0.50);
        s.p95_ms = percentile(&all_dts, 0.95);
        s.p99_ms = percentile(&all_dts, 0.99);
        s.max_ms = all_dts.last().copied().unwrap_or(0.0);
    }
    let origin = if t0.is_finite() { t0 } else { 0.0 };
    s.windows = compute_windows(&frame_ts, origin);
    s.stutters = compute_stutters(&frame_ts, origin, &span_intervals);
    Ok(s)
}

impl Summary {
    /// The `n` spans with the largest total time.
    pub fn top_by_total(&self, n: usize) -> Vec<&SpanStat> {
        let mut v: Vec<&SpanStat> = self.spans.iter().collect();
        v.sort_by(|a, b| b.total_us.total_cmp(&a.total_us).then_with(|| a.name.cmp(&b.name)));
        v.truncate(n);
        v
    }

    /// The `n` spans with the longest single duration.
    pub fn top_by_max(&self, n: usize) -> Vec<&SpanStat> {
        let mut v: Vec<&SpanStat> = self.spans.iter().collect();
        v.sort_by(|a, b| b.max_us.total_cmp(&a.max_us).then_with(|| a.name.cmp(&b.name)));
        v.truncate(n);
        v
    }

    /// The `n` locks with the largest total wait.
    pub fn top_locks(&self, n: usize) -> Vec<&LockStat> {
        let mut v: Vec<&LockStat> = self.locks.iter().collect();
        v.sort_by(|a, b| b.total_us.total_cmp(&a.total_us).then_with(|| a.name.cmp(&b.name)));
        v.truncate(n);
        v
    }

    /// The text report of `decant_trace_summary`, with `top` rows in each list.
    pub fn report(&self, top: usize) -> String {
        let ms = |us: f64| us / 1000.0;
        let mut o = String::new();
        let _ = writeln!(o, "trace: {} events, {} threads, {:.3} ms wall", self.events, self.threads.len(), ms(self.wall_us));
        if self.frames > 0 {
            let _ = writeln!(o, "frames: {}, mean interval {:.3} ms", self.frames, ms(self.frame_mean_us));
        }
        if self.counter_samples > 0 {
            let _ = writeln!(o, "counter samples: {}", self.counter_samples);
        }
        if self.dropped > 0 {
            let _ = writeln!(o, "dropped events: {} (a thread buffer was full)", self.dropped);
        }
        let span_rows = |o: &mut String, rows: Vec<&SpanStat>| {
            let _ = writeln!(o, "{:>12} {:>8} {:>12} {:>12}  span", "total ms", "count", "mean ms", "max ms");
            for r in rows {
                let _ = writeln!(o, "{:>12.3} {:>8} {:>12.3} {:>12.3}  {}:{}", ms(r.total_us), r.count, ms(r.total_us / r.count.max(1) as f64), ms(r.max_us), r.cat, r.name);
            }
        };
        let _ = writeln!(o, "\ntop spans by total time (a parent includes its children)");
        span_rows(&mut o, self.top_by_total(top));
        let _ = writeln!(o, "\ntop spans by max time");
        span_rows(&mut o, self.top_by_max(top));
        let _ = writeln!(o, "\nthread active time");
        let _ = writeln!(o, "{:>12} {:>7} {:>12} {:>8}  thread", "active ms", "util %", "lock ms", "spans");
        for t in self.threads.iter().take(top) {
            let _ = writeln!(o, "{:>12.3} {:>7.1} {:>12.3} {:>8}  {} (tid {})", ms(t.active_us), t.utilization(self.wall_us) * 100.0, ms(t.lock_wait_us), t.spans, t.name, t.tid);
        }
        let _ = writeln!(o, "\ntop lock wait hotspots");
        if self.locks.is_empty() {
            let _ = writeln!(o, "none recorded");
        } else {
            let _ = writeln!(o, "{:>12} {:>8} {:>12}  lock", "total ms", "count", "max ms");
            for l in self.top_locks(top) {
                let _ = writeln!(o, "{:>12.3} {:>8} {:>12.3}  {}", ms(l.total_us), l.count, ms(l.max_us), l.name);
            }
        }
        if !self.windows.is_empty() {
            let _ = writeln!(o, "\nframe windows (10s)");
            let _ = writeln!(o, "{:>11} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8}", "window", "frames", "FPS", "p50 ms", "p95 ms", "p99 ms", "max ms");
            for w in &self.windows {
                let label = format!("{}..{}s", w.start_s, w.end_s);
                let _ = writeln!(
                    o,
                    "{:>11} {:>7} {:>7.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                    label, w.frames, w.fps, w.p50_ms, w.p95_ms, w.p99_ms, w.max_ms
                );
            }
        }
        if !self.stutters.is_empty() {
            let _ = writeln!(o, "\ntop {} stutter frames (> 50 ms)", self.stutters.len());
            let _ = writeln!(o, "{:>9} {:>9}  overlapping spans & locks", "at_s", "dur ms");
            for st in &self.stutters {
                let mut parts: Vec<String> = st.spans.iter().map(|(n, d)| format!("{n}={d:.1}ms")).collect();
                parts.extend(st.locks.iter().map(|(n, d)| format!("lock:{n}={d:.1}ms")));
                let detail = if parts.is_empty() { "untraced".to_owned() } else { parts.join(", ") };
                let _ = writeln!(o, "{:>9.3} {:>9.2}  {}", st.at_s, st.dur_ms, detail);
            }
        }
        if !self.counters.is_empty() {
            let _ = writeln!(o, "\ncounter summaries");
            let _ = writeln!(o, "{:>8} {:>12} {:>12} {:>12} {:>12}  counter", "count", "min", "mean", "max", "last");
            for c in &self.counters {
                let _ = writeln!(o, "{:>8} {:>12.2} {:>12.2} {:>12.2} {:>12.2}  {}", c.count, c.min, c.mean, c.max, c.last, c.name);
            }
        }
        o
    }

    /// Serializes this summary to a structured `metrics.json` string.
    pub fn to_metrics_json(&self, top: usize) -> String {
        let top_spans: Vec<Value> = self
            .top_by_total(top)
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "cat": r.cat,
                    "name": r.name,
                    "count": r.count,
                    "total_ms": r.total_us / 1000.0,
                    "mean_ms": r.total_us / r.count.max(1) as f64 / 1000.0,
                    "max_ms": r.max_us / 1000.0,
                })
            })
            .collect();
        let top_locks: Vec<Value> = self
            .top_locks(top)
            .into_iter()
            .map(|l| {
                serde_json::json!({
                    "name": l.name,
                    "count": l.count,
                    "total_ms": l.total_us / 1000.0,
                    "max_ms": l.max_us / 1000.0,
                })
            })
            .collect();
        let windows: Vec<Value> = self
            .windows
            .iter()
            .map(|w| {
                serde_json::json!({
                    "start_s": w.start_s,
                    "end_s": w.end_s,
                    "frames": w.frames,
                    "fps": w.fps,
                    "p50_ms": w.p50_ms,
                    "p95_ms": w.p95_ms,
                    "p99_ms": w.p99_ms,
                    "max_ms": w.max_ms,
                })
            })
            .collect();
        let top_stutters: Vec<Value> = self
            .stutters
            .iter()
            .map(|st| {
                let spans: Vec<Value> = st.spans.iter().map(|(label, ms)| serde_json::json!({"label": label, "overlap_ms": ms})).collect();
                let locks: Vec<Value> = st.locks.iter().map(|(name, ms)| serde_json::json!({"name": name, "overlap_ms": ms})).collect();
                serde_json::json!({
                    "at_s": st.at_s,
                    "dur_ms": st.dur_ms,
                    "spans": spans,
                    "locks": locks,
                })
            })
            .collect();
        let counters: Vec<Value> = self
            .counters
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "count": c.count,
                    "min": c.min,
                    "mean": c.mean,
                    "max": c.max,
                    "last": c.last,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "overall": {
                "frames": self.frames,
                "fps": self.fps,
                "mean_ms": self.frame_mean_us / 1000.0,
                "p50_ms": self.p50_ms,
                "p95_ms": self.p95_ms,
                "p99_ms": self.p99_ms,
                "max_ms": self.max_ms,
                "hitches_50ms": self.hitches_50ms,
                "hitches_100ms": self.hitches_100ms,
                "hitches_500ms": self.hitches_500ms,
                "wall_ms": self.wall_us / 1000.0,
                "events": self.events,
                "dropped": self.dropped,
            },
            "windows": windows,
            "top_spans": top_spans,
            "top_locks": top_locks,
            "top_stutters": top_stutters,
            "counters": counters,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_owned())
    }

    /// Compares two `metrics.json` documents produced by [`Summary::to_metrics_json`] and formats a diff table.
    pub fn compare_metrics_json(base_json: &str, head_json: &str) -> Result<String, String> {
        compare_metrics_json_impl(base_json, head_json)
    }
}

fn pct_delta(base: f64, head: f64) -> String {
    if base.abs() < 1e-9 {
        if (head - base).abs() < 1e-9 { "0.0%".to_owned() } else { "n/a".to_owned() }
    } else {
        format!("{:+.1}%", (head - base) / base * 100.0)
    }
}

fn compare_metrics_json_impl(base_json: &str, head_json: &str) -> Result<String, String> {
    let base: Value = serde_json::from_str(base_json).map_err(|e| format!("base metrics.json invalid: {e}"))?;
    let head: Value = serde_json::from_str(head_json).map_err(|e| format!("head metrics.json invalid: {e}"))?;
    let bo = base.get("overall").ok_or("base missing overall")?;
    let ho = head.get("overall").ok_or("head missing overall")?;
    let mut o = String::new();
    let _ = writeln!(o, "overall comparison (base vs head)");
    let _ = writeln!(o, "{:<16} {:>12} {:>12} {:>12} {:>9}", "metric", "base", "head", "delta", "change");
    for (key, label, prec) in [
        ("frames", "frames", 0usize),
        ("fps", "FPS", 2),
        ("mean_ms", "mean_ms", 3),
        ("p50_ms", "p50_ms", 2),
        ("p95_ms", "p95_ms", 2),
        ("p99_ms", "p99_ms", 2),
        ("max_ms", "max_ms", 2),
        ("hitches_50ms", "hitches>50ms", 0),
        ("hitches_100ms", "hitches>100ms", 0),
        ("hitches_500ms", "hitches>500ms", 0),
    ] {
        let bv = num(bo, key);
        let hv = num(ho, key);
        let dv = hv - bv;
        let _ = writeln!(
            o,
            "{:<16} {:>12.*} {:>12.*} {:>+12.*} {:>9}",
            label,
            prec,
            bv,
            prec,
            hv,
            prec,
            dv,
            pct_delta(bv, hv)
        );
    }

    let mut bw: BTreeMap<(u64, u64), &Value> = BTreeMap::new();
    if let Some(arr) = base.get("windows").and_then(Value::as_array) {
        for w in arr {
            bw.insert((id(w, "start_s"), id(w, "end_s")), w);
        }
    }
    let mut hw: BTreeMap<(u64, u64), &Value> = BTreeMap::new();
    if let Some(arr) = head.get("windows").and_then(Value::as_array) {
        for w in arr {
            hw.insert((id(w, "start_s"), id(w, "end_s")), w);
        }
    }
    if !bw.is_empty() || !hw.is_empty() {
        let mut keys: Vec<(u64, u64)> = bw.keys().chain(hw.keys()).copied().collect();
        keys.sort();
        keys.dedup();
        let _ = writeln!(o, "\nwindow comparison (10s)");
        let _ = writeln!(
            o,
            "{:>11} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "window", "base fps", "head fps", "fps %", "base p95", "head p95", "p95 %", "base max", "head max"
        );
        for k in keys {
            let b_fps = bw.get(&k).map_or(0.0, |w| num(w, "fps"));
            let h_fps = hw.get(&k).map_or(0.0, |w| num(w, "fps"));
            let b_p95 = bw.get(&k).map_or(0.0, |w| num(w, "p95_ms"));
            let h_p95 = hw.get(&k).map_or(0.0, |w| num(w, "p95_ms"));
            let b_max = bw.get(&k).map_or(0.0, |w| num(w, "max_ms"));
            let h_max = hw.get(&k).map_or(0.0, |w| num(w, "max_ms"));
            let label = format!("{}..{}s", k.0, k.1);
            let _ = writeln!(
                o,
                "{:>11} {:>8.1} {:>8.1} {:>9} {:>9.2} {:>9.2} {:>9} {:>9.2} {:>9.2}",
                label,
                b_fps,
                h_fps,
                pct_delta(b_fps, h_fps),
                b_p95,
                h_p95,
                pct_delta(b_p95, h_p95),
                b_max,
                h_max
            );
        }
    }

    let mut bc: BTreeMap<&str, &Value> = BTreeMap::new();
    if let Some(arr) = base.get("counters").and_then(Value::as_array) {
        for c in arr {
            let n = text(c, "name");
            if !n.is_empty() {
                bc.insert(n, c);
            }
        }
    }
    let mut hc: BTreeMap<&str, &Value> = BTreeMap::new();
    if let Some(arr) = head.get("counters").and_then(Value::as_array) {
        for c in arr {
            let n = text(c, "name");
            if !n.is_empty() {
                hc.insert(n, c);
            }
        }
    }
    if !bc.is_empty() || !hc.is_empty() {
        let mut names: Vec<&str> = bc.keys().chain(hc.keys()).copied().collect();
        names.sort();
        names.dedup();
        let _ = writeln!(o, "\ncounter comparison");
        let _ = writeln!(o, "{:<28} {:>10} {:>10} {:>9} {:>10} {:>10}", "counter", "base mean", "head mean", "mean %", "base max", "head max");
        for n in names {
            let bm = bc.get(n).map_or(0.0, |c| num(c, "mean"));
            let hm = hc.get(n).map_or(0.0, |c| num(c, "mean"));
            let bx = bc.get(n).map_or(0.0, |c| num(c, "max"));
            let hx = hc.get(n).map_or(0.0, |c| num(c, "max"));
            let _ = writeln!(o, "{:<28} {:>10.2} {:>10.2} {:>9} {:>10.2} {:>10.2}", n, bm, hm, pct_delta(bm, hm), bx, hx);
        }
    }
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(kind: Kind, name: &'static str, ts_ns: u64, dur_ns: u64) -> Event {
        Event { kind, cat: "test", name, ts_ns, dur_ns, value: 0.0 }
    }

    #[test]
    fn writer_output_is_valid_json_with_the_three_phases() {
        let events = [
            ev(Kind::Span, "work", 2_000, 1_500),
            Event { value: 4.5, ..ev(Kind::Counter, "queue", 1_000, 0) },
            ev(Kind::Frame, "frame", 3_000, 0),
        ];
        let json = render(7, &[ThreadView { tid: 1, name: "main", events: &events, dropped: 0 }]);
        let v: Value = serde_json::from_str(&json).unwrap();
        let list = v["traceEvents"].as_array().unwrap();
        let phases: Vec<&str> = list.iter().map(|e| e["ph"].as_str().unwrap()).collect();
        assert_eq!(phases, ["M", "M", "C", "X", "i"]);
        assert_eq!(list[1]["name"], "thread_name");
        assert_eq!(list[1]["args"]["name"], "main");
        assert_eq!(list[2]["args"]["value"], 4.5);
        assert_eq!(list[3]["ts"].as_f64(), Some(2.0));
        assert_eq!(list[3]["dur"].as_f64(), Some(1.5));
        assert_eq!(list[3]["pid"], 7);
    }

    #[test]
    fn strings_are_escaped_and_round_trip() {
        let name = "we\"ird\\name\n\t\u{1}é";
        let json = render(1, &[ThreadView { tid: 2, name, events: &[], dropped: 0 }]);
        let v: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["traceEvents"][1]["args"]["name"], name);
        let s = summarize(&json).unwrap();
        assert_eq!(s.events, 0);
    }

    #[test]
    fn non_finite_counters_stay_valid_json() {
        let events = [Event { value: f64::NAN, ..ev(Kind::Counter, "bad", 0, 0) }, Event { value: f64::INFINITY, ..ev(Kind::Counter, "inf", 1, 0) }];
        let json = render(1, &[ThreadView { tid: 1, name: "t", events: &events, dropped: 0 }]);
        let v: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["traceEvents"][2]["args"]["value"], 0.0);
        assert_eq!(summarize(&json).unwrap().counter_samples, 2);
    }

    #[test]
    fn round_trip_through_the_reader() {
        let a = [ev(Kind::Span, "frame_work", 0, 10_000_000), ev(Kind::LockWait, "queue", 2_000_000, 3_000_000), ev(Kind::Frame, "frame", 0, 0), ev(Kind::Frame, "frame", 10_000_000, 0)];
        let b = [ev(Kind::Span, "upload", 1_000_000, 4_000_000), ev(Kind::Span, "upload", 6_000_000, 2_000_000)];
        let json = render(1, &[ThreadView { tid: 1, name: "main", events: &a, dropped: 3 }, ThreadView { tid: 2, name: "worker", events: &b, dropped: 2 }]);
        let s = summarize(&json).unwrap();
        assert_eq!(s.events, 6);
        assert_eq!(s.dropped, 5);
        assert_eq!(s.frames, 2);
        assert_eq!(s.frame_mean_us, 10_000.0);
        assert_eq!(s.wall_us, 10_000.0);
        let upload = s.spans.iter().find(|x| x.name == "upload").unwrap();
        assert_eq!((upload.count, upload.total_us, upload.max_us), (2, 6_000.0, 4_000.0));
        assert_eq!(s.locks, [LockStat { name: "queue".into(), count: 1, total_us: 3_000.0, max_us: 3_000.0 }]);
        let main = s.threads.iter().find(|t| t.name == "main").unwrap();
        assert_eq!((main.active_us, main.lock_wait_us, main.spans), (10_000.0, 3_000.0, 1));
        assert_eq!(s.threads[0].name, "main");
    }

    #[test]
    fn aggregation_of_handwritten_trace() {
        let json = r#"[
            {"name":"thread_name","ph":"M","pid":1,"tid":5,"args":{"name":"render"}},
            {"name":"a","cat":"gpu","ph":"X","ts":0,"dur":100,"pid":1,"tid":5},
            {"name":"b","cat":"gpu","ph":"X","ts":10,"dur":20,"pid":1,"tid":5},
            {"name":"a","cat":"gpu","ph":"X","ts":200,"dur":50,"pid":1,"tid":5},
            {"name":"big","cat":"io","ph":"X","ts":0,"dur":400,"pid":1,"tid":6},
            {"name":"m","cat":"lock_wait","ph":"X","ts":5,"dur":7,"pid":1,"tid":6},
            {"name":"m","cat":"lock_wait","ph":"X","ts":50,"dur":9,"pid":1,"tid":5},
            {"name":"k","cat":"lock_wait","ph":"X","ts":50,"dur":1,"pid":1,"tid":5},
            {"name":"x","ph":"B","ts":1,"pid":1,"tid":5}
        ]"#;
        let s = summarize(json).unwrap();
        assert_eq!(s.events, 7);
        assert_eq!(s.wall_us, 400.0);
        let by_total: Vec<&str> = s.top_by_total(10).iter().map(|x| x.name.as_str()).collect();
        assert_eq!(by_total, ["big", "a", "b"]);
        let by_max: Vec<&str> = s.top_by_max(2).iter().map(|x| x.name.as_str()).collect();
        assert_eq!(by_max, ["big", "a"]);
        let render_thread = s.threads.iter().find(|t| t.tid == 5).unwrap();
        assert_eq!(render_thread.name, "render");
        assert_eq!(render_thread.active_us, 150.0);
        assert_eq!(render_thread.lock_wait_us, 10.0);
        assert_eq!(s.threads.iter().find(|t| t.tid == 6).unwrap().name, "thread 6");
        assert!((render_thread.utilization(s.wall_us) - 0.375).abs() < 1e-9);
        let hot: Vec<(&str, f64)> = s.top_locks(5).iter().map(|l| (l.name.as_str(), l.total_us)).collect();
        assert_eq!(hot, [("m", 16.0), ("k", 1.0)]);
    }

    #[test]
    fn overlapping_spans_count_once_in_active_time() {
        assert_eq!(union_length(&mut [(0.0, 10.0), (2.0, 5.0), (8.0, 12.0), (20.0, 25.0)]), 17.0);
        assert_eq!(union_length(&mut []), 0.0);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(summarize("not json").unwrap_err().contains("not valid JSON"));
        assert!(summarize("{\"other\":[]}").unwrap_err().contains("traceEvents"));
        assert!(summarize("3").is_err());
        let empty = summarize("{\"traceEvents\":[]}").unwrap();
        assert_eq!((empty.events, empty.wall_us), (0, 0.0));
    }

    #[test]
    fn report_lists_every_section() {
        let json = r#"{"traceEvents":[
            {"name":"thread_name","ph":"M","pid":1,"tid":1,"args":{"name":"main"}},
            {"name":"present","cat":"d3d12","ph":"X","ts":0,"dur":2500,"pid":1,"tid":1},
            {"name":"pso","cat":"lock_wait","ph":"X","ts":0,"dur":500,"pid":1,"tid":1}
        ]}"#;
        let text = summarize(json).unwrap().report(5);
        for needle in ["top spans by total time", "top spans by max time", "thread active time", "top lock wait hotspots", "d3d12:present", "main (tid 1)", "pso"] {
            assert!(text.contains(needle), "missing {needle} in\n{text}");
        }
        let none = summarize("[]").unwrap().report(3);
        assert!(none.contains("none recorded"));
    }

    #[test]
    fn windows_stutters_and_range_filter_attribute_spans_and_locks() {
        let json = r#"{"traceEvents":[
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":0,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":20000,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":120000,"pid":1,"tid":1},
            {"name":"sync","cat":"title_mem","ph":"X","ts":30000,"dur":60000,"pid":1,"tid":1},
            {"name":"STATE","cat":"lock_wait","ph":"X","ts":40000,"dur":30000,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":12000000,"pid":1,"tid":1},
            {"name":"late","cat":"detile","ph":"X","ts":12000000,"dur":5000,"pid":1,"tid":1}
        ]}"#;
        let full = summarize(json).unwrap();
        assert_eq!(full.windows.len(), 2);
        assert_eq!((full.windows[0].start_s, full.windows[0].end_s, full.windows[0].frames), (0, 10, 3));
        assert_eq!((full.windows[1].start_s, full.windows[1].end_s, full.windows[1].frames), (10, 20, 1));
        assert_eq!(full.stutters.len(), 2);
        // Worst stutter is 12.0s - 0.12s = 11880 ms, second is 100 ms (with title_mem:sync=60ms and lock STATE=30ms).
        let s100 = full.stutters.iter().find(|s| (s.dur_ms - 100.0).abs() < 1e-6).unwrap();
        assert_eq!(s100.spans, vec![("title_mem:sync".to_owned(), 60.0)]);
        assert_eq!(s100.locks, vec![("STATE".to_owned(), 30.0)]);

        let ranged = summarize_range(json, Some(10.0), Some(15.0)).unwrap();
        assert_eq!(ranged.frames, 1);
        assert_eq!(ranged.spans.len(), 1);
        assert_eq!(ranged.spans[0].name, "late");
    }

    #[test]
    fn metrics_json_round_trip_and_compare_diff() {
        let base_trace = r#"{"traceEvents":[
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":0,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":40000,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":120000,"pid":1,"tid":1},
            {"name":"WorkingSetMiB","cat":"counter","ph":"C","ts":10000,"pid":1,"tid":1,"args":{"value":1000.0}}
        ]}"#;
        let head_trace = r#"{"traceEvents":[
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":0,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":20000,"pid":1,"tid":1},
            {"name":"frame","cat":"frame","ph":"i","s":"g","ts":40000,"pid":1,"tid":1},
            {"name":"WorkingSetMiB","cat":"counter","ph":"C","ts":10000,"pid":1,"tid":1,"args":{"value":800.0}}
        ]}"#;
        let base_s = summarize(base_trace).unwrap();
        assert_eq!(base_s.hitches_50ms, 1);
        assert_eq!(base_s.counters.len(), 1);
        let base_json = base_s.to_metrics_json(5);
        let head_json = summarize(head_trace).unwrap().to_metrics_json(5);
        let parsed: Value = serde_json::from_str(&base_json).unwrap();
        assert_eq!(parsed["overall"]["frames"], 3);
        assert_eq!(parsed["overall"]["hitches_50ms"], 1);
        assert_eq!(parsed["counters"][0]["name"], "WorkingSetMiB");

        let diff = Summary::compare_metrics_json(&base_json, &head_json).unwrap();
        assert!(diff.contains("overall comparison"), "{diff}");
        assert!(diff.contains("hitches>50ms"), "{diff}");
        assert!(diff.contains("WorkingSetMiB"), "{diff}");
        assert!(diff.contains("-20.0%"), "{diff}");
    }
}
