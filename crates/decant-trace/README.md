# decant-trace

Low overhead tracing for the Decant layers. It writes a Chrome and Perfetto JSON trace, can forward the same events to the Tracy profiler, and comes with `decant_trace_summary`, a command line analyzer for the traces it writes.

## Turning it on

Tracing is controlled by the environment variable `DECANT_TRACE`.

* Unset, empty or `0`: off. Every call costs one relaxed load of an `AtomicBool` and allocates nothing. A span guard holds no data.
* `1`: on, writes `decant_trace.json` in the working directory.
* Any other value: on, and the value is the path of the file.

The variable is read on the first call into the crate. `start(path)` and `stop()` do the same from code, for tools and tests.

## Recording

```rust
let _frame = decant_trace::span("d3d12", "present");   // ends when the guard drops
decant_trace::record_counter("pending_uploads", 12.0);
decant_trace::record_lock_wait("pipeline_cache", 340);  // microseconds, ending now
let guard = decant_trace::timed_lock("pipeline_cache", || cache.lock());
decant_trace::frame_mark();
```

Names are `&'static str`, so recording never formats or allocates a string. Each thread appends to its own buffer. The buffers are registered once per thread, and the thread takes only its own lock, which no other thread holds while it records. A thread keeps at most `MAX_EVENTS_PER_THREAD` events (2 097 152). Later events are counted and the count is written to the trace.

`flush()` writes everything recorded so far to the trace file. It also runs when the process exits through `atexit`. Inside a DLL the C runtime runs that handler when the DLL unloads, so call `flush()` yourself at a point you control when the exit order matters.

## Trace format

The file is one JSON object with a `traceEvents` array. Times are microseconds with three decimals since the first recorded event.

| Event | Written as |
|---|---|
| Span | `ph` `X` with `cat`, `ts` and `dur` |
| Lock wait | `ph` `X` with `cat` `lock_wait`, ending at the time of the call |
| Counter | `ph` `C` with `args.value` |
| Frame mark | `ph` `i`, `cat` `frame`, global scope |
| Thread name | `ph` `M`, `thread_name`, from the name given to the thread, otherwise `thread N` |

Open the file in `ui.perfetto.dev` or `chrome://tracing`.

## Tracy

The `tracy` feature depends on `tracy-client` (MIT or Apache-2.0) with only its `enable` feature. It needs a C++ compiler to build. When the feature is on and tracing is on, spans become Tracy zones, counters become plots, lock waits become plots named `lock wait us: <name>` and frame marks become Tracy frame marks. The JSON trace is written as well. Without the feature the hook has no code.

```sh
cargo build -p decant-trace --features tracy
```

## Reading a trace

```sh
cargo run --locked -p decant-trace --bin decant_trace_summary -- decant_trace.json --top 15
```

The report has four parts.

* The top spans by total time and by longest single span. A parent span includes the time of its children.
* Each thread with its active time, its share of the wall time, its lock wait time and its span count. Nested and overlapping spans on one thread count once.
* The lock wait hotspots, by total wait.
* The frame count and the mean frame interval when frame marks exist, and the number of dropped events when a buffer filled up.

`decant_trace::summarize` gives the same numbers as a `Summary` value for other tools.

## What was tested

The unit tests use synthetic events and need no GPU and no console files. They cover recording from two threads, the JSON the writer produces, the round trip through the reader, the aggregation of a handwritten trace, the disabled path and the buffer limit. The `atexit` write was checked once by hand with a small program that never called `flush()`. The Tracy path was built and tested with the `tracy` feature, with no profiler connected, so what a profiler shows was not checked.
