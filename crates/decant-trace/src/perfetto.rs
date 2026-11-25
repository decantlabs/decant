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
    /// Number of frame marks.
    pub frames: u64,
    /// Mean time between frame marks in microseconds, 0 for fewer than two frames.
    pub frame_mean_us: f64,
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

/// Reads a Chrome or Perfetto JSON trace, either an object with `traceEvents` or a bare array.
pub fn summarize(json: &str) -> Result<Summary, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| format!("not valid JSON: {e}"))?;
    let events: &Vec<Value> = match &root {
        Value::Array(a) => a,
        Value::Object(o) => o.get("traceEvents").and_then(Value::as_array).ok_or("no traceEvents array")?,
        _ => return Err("expected an object or an array".to_owned()),
    };
    let mut spans: BTreeMap<(String, String), Agg> = BTreeMap::new();
    let mut locks: BTreeMap<String, Agg> = BTreeMap::new();
    let mut threads: BTreeMap<(u64, u64), ThreadAcc> = BTreeMap::new();
    let mut names: BTreeMap<(u64, u64), String> = BTreeMap::new();
    let mut frame_ts: Vec<f64> = Vec::new();
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
        match ph {
            "X" => {
                let (cat, name) = (text(e, "cat"), text(e, "name"));
                let acc = threads.entry(key).or_default();
                if cat == LOCK_WAIT_CAT {
                    locks.entry(name.to_owned()).or_default().add(dur);
                    acc.lock_wait += dur;
                } else {
                    spans.entry((cat.to_owned(), name.to_owned())).or_default().add(dur);
                    acc.intervals.push((ts, ts + dur));
                }
            }
            "C" => s.counter_samples += 1,
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
    s.frames = u64::try_from(frame_ts.len()).unwrap_or(u64::MAX);
    if let (Some(lo), Some(hi)) = (frame_ts.iter().copied().reduce(f64::min), frame_ts.iter().copied().reduce(f64::max))
        && frame_ts.len() > 1
    {
        s.frame_mean_us = (hi - lo) / (frame_ts.len() - 1) as f64;
    }
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
        o
    }
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
}
