//! Content keyed scheduling for deferred shader pipelines. Workers publish DXIL cache keys,
//! while the title still creates the actual pipeline at bind time.
//!
//! * A pipeline whose disk record and stage files are already there is ready at registration
//!   and again checked at bind, so a warm cache hit never waits behind a running compile.
//! * Binds promote pending jobs to the urgent queue in the order they came (first in, first out).
//! * With two or more workers, worker 0 is a fast lane: it takes urgent jobs whose stages are all
//!   below [`O0_HLSL_BYTES`] of HLSL. A larger job it translated goes back to the queue with its
//!   HLSL, for the other workers.
//! * Urgent jobs never wait for the busy gate. Background jobs start while the title does not
//!   bind. While it binds, up to [`Limits::busy_background`] of them run, one on a small machine
//!   and more on a machine with many cores. Running urgent jobs hold background jobs back only
//!   from [`Limits::urgent_yield`] of them on. Both limits follow the core count and are chosen,
//!   not measured.
//! * A progress line (`direct precompile: warmup ...`) with the queued, active, ready and failed
//!   pipelines is logged every [`PROGRESS_PERIOD`] while those numbers change, and at exit.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

#[cfg(windows)]
use crate::compile_service::O0_HLSL_BYTES;
use crate::direct_cache::Fnv128;

const MAX_ENTRIES: usize = 131_072;
const MAGIC: &[u8] = b"SXBO\x01\0\0\0";
const NAME: &str = "bind-order.bin";
/// The bind history is written every this many progress periods.
#[cfg_attr(not(windows), allow(dead_code))]
const SAVE_EVERY_TICKS: u32 = 6;

fn read_history(dir: &Path) -> Option<Vec<u128>> {
    const LIMIT: usize = 12 + MAX_ENTRIES * 16 + 16;
    let mut bytes = Vec::new();
    File::open(dir.join(NAME)).ok()?.take((LIMIT + 1) as u64).read_to_end(&mut bytes).ok()?;
    let end = bytes.len().checked_sub(16)?;
    let body = &bytes[..end];
    let count = u32::from_le_bytes(body.get(8..12)?.try_into().ok()?) as usize;
    if !body.starts_with(MAGIC)
        || count > MAX_ENTRIES
        || body.len() != 12 + count * 16
        || Fnv128::default().bytes(body).finish().to_le_bytes() != bytes[end..]
    {
        return None;
    }
    let keys: Vec<_> = body[12..].as_chunks::<16>().0.iter().copied().map(u128::from_le_bytes).collect();
    (keys.iter().collect::<HashSet<_>>().len() == keys.len()).then_some(keys)
}

fn write_history(dir: &Path, keys: &[u128]) -> io::Result<()> {
    let keys = &keys[..keys.len().min(MAX_ENTRIES)];
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for key in keys {
        bytes.extend_from_slice(&key.to_le_bytes());
    }
    let check = Fnv128::default().bytes(&bytes).finish();
    bytes.extend_from_slice(&check.to_le_bytes());
    fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".bind-order.{}.tmp", std::process::id()));
    let result = fs::write(&temp, bytes).and_then(|()| fs::rename(&temp, dir.join(NAME)));
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Priority {
    History(usize),
    Compute(usize),
    Simple(usize),
    Bulk(usize),
}

/// What a precompile worker takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    /// The only worker: every job.
    All,
    /// The reserved fast lane: urgent jobs whose stages are all small.
    Fast,
    /// Urgent and background jobs of any size.
    Bulk,
}

impl Role {
    /// The role of worker `index` of `count`: worker 0 is the fast lane when there are two or more.
    fn of(index: usize, count: usize) -> Role {
        match (index, count) {
            (_, 0 | 1) => Role::All,
            (0, _) => Role::Fast,
            _ => Role::Bulk,
        }
    }

    /// Whether a worker of this role may start a background job now. Urgent jobs never ask: a
    /// busy title or a running background compile does not hold them back.
    fn background(self, busy: bool, background_running: usize, urgent_running: usize, limits: Limits) -> bool {
        self != Role::Fast && urgent_running < limits.urgent_yield && (!busy || background_running < limits.busy_background)
    }
}

/// How many jobs may run together before the gate of [`Role::background`] closes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Limits {
    /// Background jobs that may run while the title binds pipelines.
    busy_background: usize,
    /// Running urgent jobs from which no new background job starts.
    urgent_yield: usize,
}

impl Limits {
    /// The strictest limits: one background job while the title binds, none beside an urgent job.
    /// This is what a machine of up to 7 cores gets.
    #[cfg_attr(not(test), allow(dead_code))]
    const SERIAL: Limits = Limits { busy_background: 1, urgent_yield: 1 };

    /// The limits for `workers` precompile workers on a machine with `cores` logical cores. Every
    /// 4 cores let one more background job run while the title binds, never more than there are
    /// workers. Every 8 cores let one more urgent job run before background jobs yield.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn for_machine(workers: usize, cores: usize) -> Limits {
        Limits { busy_background: (cores / 4).clamp(1, workers.max(1)), urgent_yield: (cores / 8).max(1) }
    }
}

/// How often the progress line is considered.
#[cfg_attr(not(windows), allow(dead_code))]
const PROGRESS_PERIOD: std::time::Duration = std::time::Duration::from_secs(5);

/// The pipelines of the queue by state, at one moment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Progress {
    /// Waiting for a worker.
    queued: u64,
    /// Being translated or compiled.
    active: u64,
    /// With DXIL, whether compiled or read from the disk record.
    ready: u64,
    /// Of those, ready from the disk record before any compile.
    cached: u64,
    /// Refused or failed to compile.
    failed: u64,
}

impl Progress {
    fn total(&self) -> u64 {
        self.queued + self.active + self.ready + self.failed
    }

    /// Whether no pipeline waits or runs.
    fn drained(&self) -> bool {
        self.queued == 0 && self.active == 0
    }

    /// The share of pipelines that are done, ready or failed, in percent.
    fn percent(&self) -> u64 {
        match self.total() {
            0 => 100,
            total => (self.ready + self.failed) * 100 / total,
        }
    }

    fn line(&self) -> String {
        format!(
            "direct precompile: warmup {} of {} pipelines done ({}%), {} queued, {} active, {} ready ({} from cache), {} failed{}",
            self.ready + self.failed,
            self.total(),
            self.percent(),
            self.queued,
            self.active,
            self.ready,
            self.cached,
            self.failed,
            if self.drained() { ", queue is idle" } else { "" }
        )
    }
}

/// Decides which progress lines are logged: one per change, none while nothing moves.
#[derive(Default)]
struct ProgressLog {
    last: Option<Progress>,
}

impl ProgressLog {
    /// The line to log for `now`, `None` when it is the same as the last logged one or nothing was
    /// ever queued.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn next(&mut self, now: Progress) -> Option<String> {
        if now.total() == 0 || self.last == Some(now) {
            return None;
        }
        self.last = Some(now);
        Some(now.line())
    }
}

type Heap = BinaryHeap<Reverse<(Priority, u32, u128)>>;

struct Entry<T> {
    job: T,
    priority: Priority,
    sequence: u32,
    compute: bool,
    urgent: bool,
    /// The fast lane found a stage of at least `O0_HLSL_BYTES`: only the other workers take it.
    large: bool,
}

/// A job a worker took, with what it needs to go back in the queue.
#[derive(Debug, PartialEq)]
struct Popped<T> {
    key: u128,
    job: T,
    priority: Priority,
    sequence: u32,
    compute: bool,
    urgent: bool,
}

struct Schedule<T> {
    /// Urgent compute binds are served ahead of urgent graphics/mesh binds so waited compute
    /// dispatches are never buried behind hundreds of unwaited cold world graphics binds.
    urgent_compute: VecDeque<u128>,
    urgent_other: VecDeque<u128>,
    small: Heap,
    large: Heap,
    jobs: HashMap<u128, Entry<T>>,
}

impl<T> Default for Schedule<T> {
    fn default() -> Self {
        Self {
            urgent_compute: VecDeque::new(),
            urgent_other: VecDeque::new(),
            small: BinaryHeap::new(),
            large: BinaryHeap::new(),
            jobs: HashMap::new(),
        }
    }
}

/// The first live entry of `heap` for jobs of the size class `large`, stale entries dropped.
fn top<T>(heap: &mut Heap, jobs: &HashMap<u128, Entry<T>>, large: bool) -> Option<(Priority, u32, u128)> {
    while let Some(&Reverse(entry)) = heap.peek() {
        if jobs.get(&entry.2).is_some_and(|job| !job.urgent && job.large == large) {
            return Some(entry);
        }
        heap.pop();
    }
    None
}

impl<T> Schedule<T> {
    fn insert(&mut self, key: u128, job: T, priority: Priority, sequence: u32, compute: bool) {
        self.jobs.insert(key, Entry { job, priority, sequence, compute, urgent: false, large: false });
        self.small.push(Reverse((priority, sequence, key)));
    }

    /// Moves a pending job to the back of its urgent queue (compute ahead of graphics/mesh).
    fn promote(&mut self, key: u128) {
        if let Some(entry) = self.jobs.get_mut(&key)
            && !entry.urgent
        {
            entry.urgent = true;
            if entry.compute {
                self.urgent_compute.push_back(key);
            } else {
                self.urgent_other.push_back(key);
            }
        }
    }

    /// Puts back a job the fast lane found too large, for the other workers. An urgent job keeps
    /// its place at the head of its urgent queue, as it was the oldest there.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn requeue_large(&mut self, popped: Popped<T>) {
        let Popped { key, job, priority, sequence, compute, urgent } = popped;
        self.jobs.insert(key, Entry { job, priority, sequence, compute, urgent, large: true });
        if urgent {
            if compute {
                self.urgent_compute.push_front(key);
            } else {
                self.urgent_other.push_front(key);
            }
        } else {
            self.large.push(Reverse((priority, sequence, key)));
        }
    }

    #[cfg_attr(not(windows), allow(dead_code))]
    fn get(&self, key: u128) -> Option<&T> {
        self.jobs.get(&key).map(|entry| &entry.job)
    }

    /// Drops a pending job that no longer needs a worker.
    #[cfg_attr(not(windows), allow(dead_code))]
    fn remove(&mut self, key: u128) -> Option<T> {
        let entry = self.jobs.remove(&key)?;
        if entry.urgent {
            if entry.compute {
                self.urgent_compute.retain(|queued| *queued != key);
            } else {
                self.urgent_other.retain(|queued| *queued != key);
            }
        }
        Some(entry.job)
    }

    fn take(&mut self, key: u128) -> Option<Popped<T>> {
        let Entry { job, priority, sequence, compute, urgent, .. } = self.jobs.remove(&key)?;
        Some(Popped { key, job, priority, sequence, compute, urgent })
    }

    fn pop_urgent(queue: &mut VecDeque<u128>, jobs: &HashMap<u128, Entry<T>>, small_only: bool) -> Option<u128> {
        let mut at = 0;
        while let Some(&key) = queue.get(at) {
            match jobs.get(&key) {
                Some(entry) if entry.urgent && small_only && entry.large => at += 1,
                Some(entry) if entry.urgent => {
                    queue.remove(at);
                    return Some(key);
                }
                _ => {
                    queue.remove(at);
                }
            }
        }
        None
    }

    /// The next job: the oldest urgent compute one, then the oldest urgent graphics/mesh one,
    /// else with `background` the background job of the best priority (only bind history while
    /// `low_memory`). `small_only` skips large jobs.
    fn pop(&mut self, background: bool, low_memory: bool, small_only: bool) -> Option<Popped<T>> {
        if let Some(key) = Self::pop_urgent(&mut self.urgent_compute, &self.jobs, small_only) {
            return self.take(key);
        }
        if let Some(key) = Self::pop_urgent(&mut self.urgent_other, &self.jobs, small_only) {
            return self.take(key);
        }
        if !background {
            return None;
        }
        let small = top(&mut self.small, &self.jobs, false);
        let large = if small_only { None } else { top(&mut self.large, &self.jobs, true) };
        let (entry, from_large) = match (small, large) {
            (Some(small), Some(large)) if large < small => (large, true),
            (Some(small), _) => (small, false),
            (None, Some(large)) => (large, true),
            (None, None) => return None,
        };
        if low_memory && !matches!(entry.0, Priority::History(_)) {
            return None;
        }
        if from_large {
            self.large.pop();
        } else {
            self.small.pop();
        }
        self.take(entry.2)
    }
}

#[cfg(windows)]
mod runtime {
    #![allow(unsafe_code)]

    use super::*;
    use crate::compile_service::{Poll, Service, Translated};
    use crate::direct_cache::DiskCache;
    use crate::direct_source::DirectSource;
    use crate::log::xs_log;
    use crate::shader_source::{Pipeline, StageKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetCurrentThreadId, OpenThread, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL, THREAD_SET_INFORMATION,
    };

    /// One queued pipeline, with its translation once a fast lane worker made it.
    pub(crate) struct Job {
        pub(crate) pipeline: Pipeline,
        pub(crate) translated: Option<Translated>,
    }

    /// The final answer of a worker, or `None` when the fast lane leaves the job to the others.
    pub(crate) type Outcome = Option<(Vec<u128>, Poll)>;

    /// The work of one queue job on the layer's service: translates when needed, leaves a stage of
    /// at least `O0_HLSL_BYTES` without DXIL to the other workers when `fast`, else compiles.
    pub(crate) fn run_job(service: &Service, source: &DirectSource, job: &mut Job, fast: bool) -> Outcome {
        let translated = match job.translated.take() {
            Some(translated) => translated,
            None => match service.translate_deferred(source, &job.pipeline) {
                Ok(translated) => translated,
                Err(refused) => return Some((Vec::new(), refused)),
            },
        };
        if fast && !service.small_or_cached(&translated) {
            job.translated = Some(translated);
            return None;
        }
        Some(service.compile_translated(&job.pipeline, translated))
    }

    pub(super) enum Status {
        Pending,
        Running { thread: u32, boosted: bool },
        /// Stage keys of the DXIL, `cached` when the disk record was there before any compile.
        Ready { keys: Arc<[u128]>, cached: bool },
        Refused,
    }

    #[derive(Default)]
    pub(super) struct State {
        pub(super) schedule: Schedule<Job>,
        pub(super) slots: HashMap<u128, Status>,
        history: Vec<u128>,
        bound: HashSet<u128>,
        saved: usize,
        pub(super) background_running: usize,
        pub(super) urgent_running: usize,
        left_large: u64,
        memory_sample: Option<(Instant, bool)>,
    }

    impl State {
        /// The pipelines of the queue by state. Counted over the slots, so it cannot drift from them.
        pub(super) fn progress(&self) -> Progress {
            let mut progress = Progress::default();
            for status in self.slots.values() {
                match status {
                    Status::Pending => progress.queued += 1,
                    Status::Running { .. } => progress.active += 1,
                    Status::Ready { cached, .. } => {
                        progress.ready += 1;
                        progress.cached += u64::from(*cached);
                    }
                    Status::Refused => progress.failed += 1,
                }
            }
            progress
        }
    }

    type Compile = dyn Fn(&mut Job, bool) -> Outcome + Send + Sync;

    pub(crate) struct Queue {
        cache: DiskCache,
        ranks: HashMap<u128, usize>,
        epoch: Instant,
        last_bind_ms: AtomicU64,
        min_free: u64,
        limits: Limits,
        progress_log: Mutex<ProgressLog>,
        pub(super) state: Mutex<State>,
        changed: Condvar,
        save_lock: Mutex<()>,
    }

    impl Queue {
        /// Starts `count` workers. `compile` gets each job and whether the worker is the fast lane.
        pub(crate) fn start(
            cache: DiskCache,
            count: usize,
            compile: impl Fn(&mut Job, bool) -> Outcome + Send + Sync + 'static,
        ) -> Option<Arc<Self>> {
            let cores = std::thread::available_parallelism().map_or(4, |cores| cores.get());
            Self::start_with(cache, count, Limits::for_machine(count, cores), compile, |i, role, f| {
                let _ = role;
                std::thread::Builder::new()
                    .name(format!("decant-precompile-{i}"))
                    .stack_size(32 << 20)
                    .spawn(f)
                    .map(|_| ())
            })
        }

        pub(super) fn start_with(
            cache: DiskCache,
            count: usize,
            limits: Limits,
            compile: impl Fn(&mut Job, bool) -> Outcome + Send + Sync + 'static,
            mut spawn: impl FnMut(usize, Role, Box<dyn FnOnce() + Send>) -> io::Result<()>,
        ) -> Option<Arc<Self>> {
            let previous = read_history(cache.dir()).unwrap_or_default();
            let queue = Arc::new(Self {
                cache,
                ranks: previous.into_iter().enumerate().map(|(rank, key)| (key, rank)).collect(),
                epoch: Instant::now(),
                last_bind_ms: AtomicU64::new(0),
                min_free: std::env::var("DECANT_D3D12X_PRECOMPILE_MIN_FREE_MIB")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(2048)
                    .saturating_mul(1 << 20),
                limits,
                progress_log: Mutex::new(ProgressLog::default()),
                state: Mutex::new(State::default()),
                changed: Condvar::new(),
                save_lock: Mutex::new(()),
            });
            let compile: Arc<Compile> = Arc::new(compile);
            let mut bulk_workers = 0usize;
            for i in (1..count).rev() {
                let (worker, compile) = (queue.clone(), compile.clone());
                match spawn(i, Role::Bulk, Box::new(move || worker.run(Role::Bulk, &*compile))) {
                    Ok(()) => bulk_workers += 1,
                    Err(error) => xs_log!("direct precompile: could not start worker {i} ({error})"),
                }
            }
            let first_role = Role::of(0, bulk_workers + 1);
            let (worker, compile) = (queue.clone(), compile);
            if let Err(error) = spawn(0, first_role, Box::new(move || worker.run(first_role, &*compile))) {
                xs_log!("direct precompile: could not start worker 0 ({error})");
                if bulk_workers == 0 {
                    return None;
                }
            }
            let saver = queue.clone();
            if let Err(error) = std::thread::Builder::new().name("decant-bind-order".into()).spawn(move || {
                let mut tick = 0u32;
                loop {
                    std::thread::sleep(PROGRESS_PERIOD);
                    saver.log_progress();
                    tick += 1;
                    if tick.is_multiple_of(SAVE_EVERY_TICKS) {
                        saver.save(false);
                    }
                }
            }) {
                xs_log!("direct precompile: could not start bind history writer ({error})");
            }
            xs_log!(
                "direct precompile: {count} configured workers, fast lane {} below {} HLSL bytes, {} recorded pipelines, {} background jobs while the title binds, background yields from {} urgent jobs",
                if first_role == Role::Fast { "on" } else { "off" },
                O0_HLSL_BYTES,
                queue.ranks.len(),
                queue.limits.busy_background,
                queue.limits.urgent_yield
            );
            Some(queue)
        }

        pub(crate) fn register(&self, service: &Service, pipeline: Pipeline, sequence: u32) -> Option<u128> {
            let key = service.input_key(&pipeline);
            {
                let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.slots.contains_key(&key) {
                    return Some(key);
                }
                if state.slots.len() >= MAX_ENTRIES {
                    return None;
                }
            }
            // A warm disk record is ready at once, so its bind never waits behind a running compile.
            if let Some(keys) = self.cache.read_pipeline(key).filter(|keys| keys.len() == pipeline.stages.len()) {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                state.slots.entry(key).or_insert(Status::Ready { keys: keys.into(), cached: true });
                return Some(key);
            }
            let size = pipeline.stages.iter().map(|stage| stage.bytes.len()).sum();
            let compute = pipeline.linkage.stages == [StageKind::Compute];
            let priority = if let Some(&rank) = self.ranks.get(&key) {
                Priority::History(rank)
            } else if compute {
                Priority::Compute(size)
            } else if pipeline.linkage.targets.len() <= 1
                && !pipeline.linkage.stream_output
                && size <= 16 * 1024
                && pipeline.linkage.stages.contains(&StageKind::Vertex)
            {
                Priority::Simple(size)
            } else {
                Priority::Bulk(size)
            };
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.slots.contains_key(&key) {
                if state.slots.len() >= MAX_ENTRIES {
                    return None;
                }
                state.slots.insert(key, Status::Pending);
                state.schedule.insert(key, Job { pipeline, translated: None }, priority, sequence, compute);
                self.changed.notify_all();
            }
            Some(key)
        }

        pub(crate) fn note_bind(&self) {
            self.last_bind_ms
                .fetch_max(self.epoch.elapsed().as_millis() as u64 + 1, Ordering::Relaxed);
        }

        pub(crate) fn bind(&self, service: &Service, key: u128, wait: Duration) -> Option<Poll> {
            let started = Instant::now();
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.history.len() < MAX_ENTRIES && state.bound.insert(key) {
                state.history.push(key);
            }
            if matches!(state.slots.get(&key), Some(Status::Pending)) {
                // The record may have appeared since registration (another path compiled the same
                // content): read it without the lock, then take it or promote the job.
                drop(state);
                let found = self.cache.read_pipeline(key);
                state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if matches!(state.slots.get(&key), Some(Status::Pending)) {
                    let fits = |keys: &Vec<u128>| state.schedule.get(key).is_some_and(|job| job.pipeline.stages.len() == keys.len());
                    if let Some(keys) = found.filter(fits) {
                        state.schedule.remove(key);
                        state.slots.insert(key, Status::Ready { keys: keys.into(), cached: true });
                    } else {
                        state.schedule.promote(key);
                    }
                    self.changed.notify_all();
                }
            }
            if let Some(Status::Running { thread, boosted }) = state.slots.get_mut(&key)
                && !*boosted
            {
                *boosted = boost(*thread);
            }
            let remaining = wait.saturating_sub(started.elapsed());
            let state = if !remaining.is_zero()
                && matches!(state.slots.get(&key), Some(Status::Pending | Status::Running { .. }))
            {
                let wait_started = Instant::now();
                let (state, _) = self
                    .changed
                    .wait_timeout_while(state, remaining, |state| {
                        matches!(state.slots.get(&key), Some(Status::Pending | Status::Running { .. }))
                    })
                    .unwrap_or_else(|e| e.into_inner());
                service.record_wait(wait_started.elapsed());
                state
            } else {
                state
            };
            match state.slots.get(&key)? {
                Status::Pending | Status::Running { .. } => Some(Poll::Pending),
                Status::Refused => Some(Poll::Refused(String::new())),
                Status::Ready { keys, cached } => {
                    let (keys, cached) = (keys.clone(), *cached);
                    drop(state);
                    let dxil = keys.iter()
                        .map(|key| self.cache.read(*key).map(Arc::from))
                        .collect::<Option<Vec<_>>>()?;
                    Some(if cached { service.pipe_hit(key, dxil) } else { Poll::Ready(dxil) })
                }
            }
        }

        /// Logs the progress line when the numbers changed since the last one.
        pub(super) fn log_progress(&self) {
            let progress = self.state.lock().unwrap_or_else(|e| e.into_inner()).progress();
            let line = self.progress_log.lock().unwrap_or_else(|e| e.into_inner()).next(progress);
            if let Some(line) = line {
                xs_log!("{line}");
            }
        }

        pub(crate) fn save(&self, exiting: bool) {
            if exiting {
                // The last numbers of the run, unless a worker holds the lock at teardown.
                if let Ok(state) = self.state.try_lock() {
                    let progress = state.progress();
                    drop(state);
                    if let Ok(mut log) = self.progress_log.try_lock()
                        && let Some(line) = log.next(progress)
                    {
                        xs_log!("{line}");
                    }
                }
            }
            // DLL teardown can run with other threads already stopped. Leave the last completed
            // snapshot in place if a worker or the periodic writer held a lock.
            let _writer = if exiting {
                let Ok(guard) = self.save_lock.try_lock() else { return };
                guard
            } else {
                self.save_lock.lock().unwrap_or_else(|e| e.into_inner())
            };
            let history = {
                let state = if exiting {
                    let Ok(guard) = self.state.try_lock() else { return };
                    guard
                } else {
                    self.state.lock().unwrap_or_else(|e| e.into_inner())
                };
                if state.saved == state.history.len() {
                    return;
                }
                state.history.clone()
            };
            if let Err(error) = write_history(self.cache.dir(), &history) {
                xs_log!("direct precompile: could not save bind history ({error})");
            } else if !exiting {
                self.state.lock().unwrap_or_else(|e| e.into_inner()).saved = history.len();
            }
        }

        fn run(&self, role: Role, compile: &Compile) {
            let fast = role == Role::Fast;
            if !fast {
                set_priority(THREAD_PRIORITY_LOWEST);
            }
            // SAFETY: this retrieves an identifier for the current worker.
            let thread = unsafe { GetCurrentThreadId() };
            loop {
                let mut popped = {
                    let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        let last_bind = self.last_bind_ms.load(Ordering::Relaxed);
                        let busy = last_bind != 0 && (self.epoch.elapsed().as_millis() as u64 + 1).saturating_sub(last_bind) < 1000;
                        let background = role.background(busy, state.background_running, state.urgent_running, self.limits);
                        let low_memory = background && self.low_memory(&mut state);
                        if let Some(popped) = state.schedule.pop(background, low_memory, fast) {
                            if popped.urgent {
                                state.urgent_running += 1;
                            } else {
                                state.background_running += 1;
                            }
                            state.slots.insert(popped.key, Status::Running { thread, boosted: popped.urgent });
                            break popped;
                        }
                        let (next, _) = self
                            .changed
                            .wait_timeout(state, Duration::from_secs(1))
                            .unwrap_or_else(|e| e.into_inner());
                        state = next;
                    }
                };
                if popped.urgent && !fast {
                    set_priority(THREAD_PRIORITY_NORMAL);
                }
                let outcome = compile(&mut popped.job, fast);
                self.finish_popped(popped, outcome, fast);
            }
        }

        pub(super) fn finish_popped(&self, mut popped: Popped<Job>, outcome: Outcome, fast: bool) {
            let key = popped.key;
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let bound_while_running = matches!(state.slots.get(&key), Some(Status::Running { boosted: true, .. }))
                || state.bound.contains(&key);
            if !fast && bound_while_running {
                set_priority(THREAD_PRIORITY_LOWEST);
            }
            if popped.urgent {
                state.urgent_running -= 1;
            } else {
                state.background_running -= 1;
            }
            let status = match outcome {
                Some((keys, Poll::Ready(_))) => Status::Ready { keys: keys.into(), cached: false },
                Some((_, Poll::Refused(_) | Poll::Pending)) => Status::Refused,
                None => {
                    popped.urgent |= bound_while_running;
                    let bytes = popped.job.translated.as_ref().map_or(0, Translated::largest);
                    state.left_large += 1;
                    if state.left_large <= 16 || state.left_large.is_power_of_two() {
                        xs_log!(
                            "direct precompile: pipeline {key:032x} has a stage of {bytes} HLSL bytes, the fast lane leaves it to the other workers ({} so far)",
                            state.left_large
                        );
                    }
                    state.schedule.requeue_large(popped);
                    Status::Pending
                }
            };
            state.slots.insert(key, status);
            self.changed.notify_all();
        }

        fn low_memory(&self, state: &mut State) -> bool {
            if let Some((sampled, low)) = state.memory_sample
                && sampled.elapsed() < Duration::from_secs(1)
            {
                return low;
            }
            let mut memory = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..Default::default()
            };
            // SAFETY: the writable buffer carries the size required by Windows.
            let low = unsafe { GlobalMemoryStatusEx(&mut memory) } != 0 && memory.ullAvailPhys.min(memory.ullAvailPageFile) < self.min_free;
            state.memory_sample = Some((Instant::now(), low));
            low
        }
    }

    fn boost(thread: u32) -> bool {
        // SAFETY: the queue lock holds the worker at this job until its state can be published.
        unsafe {
            let handle = OpenThread(THREAD_SET_INFORMATION, 0, thread);
            if handle.is_null() {
                return false;
            }
            let changed = SetThreadPriority(handle, THREAD_PRIORITY_NORMAL) != 0;
            CloseHandle(handle);
            changed
        }
    }

    fn set_priority(priority: i32) {
        // SAFETY: this pseudo handle names the calling thread.
        unsafe { SetThreadPriority(GetCurrentThread(), priority) };
    }
}

#[cfg(windows)]
pub(crate) use runtime::{run_job, Queue};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_cache::tests::TempDir;

    fn keys<T>(queue: &mut Schedule<T>, background: bool, low_memory: bool, small_only: bool) -> Vec<(u128, bool)> {
        std::iter::from_fn(|| queue.pop(background, low_memory, small_only)).map(|p| (p.key, p.urgent)).collect()
    }

    #[test]
    fn four_tiers_use_bind_rank_then_kind_size_and_creation_order() {
        let mut queue = Schedule::default();
        for (key, priority, seq) in [
            (10, Priority::Bulk(32), 1),
            (11, Priority::Bulk(8), 4),
            (12, Priority::Bulk(8), 3),
            (20, Priority::Simple(8), 5),
            (21, Priority::Compute(128), 6),
            (30, Priority::History(1), 7),
            (31, Priority::History(0), 8),
            (40, Priority::Bulk(64), 9),
        ] {
            queue.insert(key, key, priority, seq, matches!(priority, Priority::Compute(_)));
        }
        queue.promote(40);
        let order: Vec<_> = keys(&mut queue, true, false, false).into_iter().map(|(key, _)| key).collect();
        assert_eq!(order, [40, 31, 30, 21, 20, 12, 11, 10]);
    }

    #[test]
    fn urgent_promotion_is_first_in_first_out_and_the_stale_entry_is_skipped() {
        let mut queue = Schedule::default();
        queue.insert(1, (), Priority::Bulk(1), 0, false);
        queue.insert(2, (), Priority::Bulk(2), 1, false);
        queue.insert(3, (), Priority::Bulk(3), 2, false);
        queue.insert(4, (), Priority::Bulk(4), 3, false);
        queue.promote(2);
        queue.promote(3);
        queue.promote(3);
        queue.promote(4);
        // The bind that timed out first is served first, a later bind does not bury it.
        assert_eq!(keys(&mut queue, false, true, false), [(2, true), (3, true), (4, true)]);
        assert_eq!(keys(&mut queue, true, false, false), [(1, false)]);
    }

    #[test]
    fn urgent_compute_jumps_ahead_of_urgent_graphics_and_keeps_fifo_within_each() {
        let mut queue = Schedule::default();
        queue.insert(1, (), Priority::Bulk(1), 0, false);
        queue.insert(2, (), Priority::Bulk(2), 1, false);
        queue.insert(3, (), Priority::History(0), 2, true);
        queue.insert(4, (), Priority::Compute(8), 3, true);
        queue.promote(1);
        queue.promote(2);
        queue.promote(3);
        queue.promote(4);
        assert_eq!(keys(&mut queue, false, false, false), [(3, true), (4, true), (1, true), (2, true)]);
    }

    #[test]
    fn urgent_pops_never_wait_for_a_busy_title_or_a_background_compile() {
        for role in [Role::All, Role::Fast, Role::Bulk] {
            // The title binds and a background compile runs: no new background job may start.
            let background = role.background(true, 1, 0, Limits::SERIAL);
            assert!(!background, "{role:?}");
            let mut queue = Schedule::default();
            queue.insert(1, (), Priority::History(0), 0, false);
            queue.insert(2, (), Priority::Bulk(9), 1, false);
            queue.promote(2);
            assert_eq!(keys(&mut queue, background, false, role == Role::Fast), [(2, true)], "{role:?}");
        }
        // An idle title lets the bulk workers start background work, never the fast lane.
        assert!(Role::Bulk.background(false, 3, 0, Limits::SERIAL));
        assert!(Role::All.background(true, 0, 0, Limits::SERIAL));
        assert!(!Role::Fast.background(false, 0, 0, Limits::SERIAL));
        assert!(!Role::Bulk.background(false, 0, 1, Limits::SERIAL));
        assert_eq!([Role::of(0, 1), Role::of(0, 4), Role::of(3, 4)], [Role::All, Role::Fast, Role::Bulk]);
    }

    #[test]
    fn the_fast_lane_skips_large_jobs_and_they_keep_their_place_for_the_others() {
        let mut queue = Schedule::default();
        for key in 1..=4 {
            queue.insert(key, key, Priority::Bulk(key as usize), key as u32, false);
        }
        queue.promote(1);
        let first = queue.pop(false, false, true).unwrap();
        assert_eq!((first.key, first.urgent), (1, true));
        queue.requeue_large(first);
        // The large urgent job stays queued, the fast lane goes on with background work.
        let background = queue.pop(true, false, true).unwrap();
        assert_eq!((background.key, background.urgent), (2, false));
        queue.requeue_large(background);
        queue.promote(3);
        // The fast lane takes the small urgent job behind the large one, then the small background one.
        assert_eq!(keys(&mut queue, true, false, true), [(3, true), (4, false)]);
        // The others take the large urgent job first, then the large background one.
        assert_eq!(keys(&mut queue, true, false, false), [(1, true), (2, false)]);
    }

    #[test]
    fn a_removed_job_leaves_no_stale_urgent_entry() {
        let mut queue = Schedule::default();
        queue.insert(1, 'a', Priority::Bulk(1), 0, false);
        queue.insert(2, 'b', Priority::Bulk(1), 1, false);
        queue.promote(1);
        assert_eq!(queue.get(1), Some(&'a'));
        assert_eq!(queue.remove(1), Some('a'));
        assert!(queue.urgent_other.is_empty() && queue.urgent_compute.is_empty());
        assert_eq!(keys(&mut queue, true, false, false), [(2, false)]);
    }

    #[cfg(windows)]
    fn source() -> std::sync::Arc<crate::direct_source::DirectSource> {
        use std::sync::Arc;
        Arc::new(crate::direct_source::DirectSource::new(Arc::new(decant_shader::api::CountingHost::default()), |_| {}, None))
    }

    #[cfg(windows)]
    #[test]
    fn promoted_pipeline_starts_while_background_compile_runs() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-urgent");
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let blocked = release.clone();
        let queue = Queue::start(DiskCache::new(&dir.0), 2, move |job, fast| {
            let stage = job.pipeline.stages[0].bytes[0];
            let _ = started.send((stage, fast));
            if stage == 1 {
                let (lock, changed) = &*blocked;
                let _guard = changed.wait_timeout_while(
                    lock.lock().unwrap(), Duration::from_secs(5), |released| !*released,
                ).unwrap();
            }
            Some((Vec::new(), Poll::Refused("test".into())))
        }).unwrap();

        queue.register(&service, pipeline(&[(StageKind::Compute, &[1])]), 0).unwrap();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (1, false));
        queue.note_bind();
        let urgent = queue.register(&service, pipeline(&[(StageKind::Compute, &[2])]), 1).unwrap();
        // Refresh the busy period while waiting so the result cannot come from an idle pop.
        let mut started_urgent = None;
        for _ in 0..20 {
            queue.note_bind();
            queue.bind(&service, urgent, Duration::ZERO);
            if let Ok(stage) = jobs.recv_timeout(Duration::from_millis(50)) {
                started_urgent = Some(stage);
                break;
            }
        }
        let (lock, changed) = &*release;
        *lock.lock().unwrap() = true;
        changed.notify_one();
        assert_eq!(started_urgent, Some((2, true)), "the fast lane takes the urgent job");
    }

    #[cfg(windows)]
    #[test]
    fn the_fast_lane_hands_a_large_job_to_a_bulk_worker_and_serves_the_next_bind() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-fast-lane");
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let blocked = release.clone();
        let queue = Queue::start(DiskCache::new(&dir.0), 2, move |job, fast| {
            let stage = job.pipeline.stages[0].bytes[0];
            let _ = started.send((stage, fast));
            match stage {
                1 => {
                    let (lock, changed) = &*blocked;
                    let _guard = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |released| !*released).unwrap();
                }
                // Stage 9 stands for a stage of at least 512 KiB of HLSL.
                9 if fast => return None,
                _ => {}
            }
            Some((Vec::new(), Poll::Refused("test".into())))
        }).unwrap();

        queue.register(&service, pipeline(&[(StageKind::Compute, &[1])]), 0).unwrap();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (1, false));
        queue.note_bind();
        let large = queue.register(&service, pipeline(&[(StageKind::Compute, &[9])]), 1).unwrap();
        let small = queue.register(&service, pipeline(&[(StageKind::Compute, &[2])]), 2).unwrap();
        queue.bind(&service, large, Duration::ZERO);
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (9, true), "the fast lane looks at the large job first");
        queue.bind(&service, small, Duration::ZERO);
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (2, true), "then serves the small bind behind it");
        assert_eq!(queue.bind(&service, small, Duration::from_secs(2)), Some(Poll::Refused(String::new())));
        assert_eq!(queue.bind(&service, large, Duration::ZERO), Some(Poll::Pending), "the large job waits for a bulk worker");
        let (lock, changed) = &*release;
        *lock.lock().unwrap() = true;
        changed.notify_one();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(5)).unwrap(), (9, false), "a bulk worker takes the large job");
        assert_eq!(queue.bind(&service, large, Duration::from_secs(2)), Some(Poll::Refused(String::new())));
    }

    #[cfg(windows)]
    #[test]
    fn bind_during_fast_lane_translation_keeps_urgency_across_requeue_large() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-requeue-urgent");
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let release_bg = Arc::new((Mutex::new(false), Condvar::new()));
        let bg_wait = release_bg.clone();
        let queue = Queue::start(DiskCache::new(&dir.0), 3, move |job, fast| {
            let stage = job.pipeline.stages[0].bytes[0];
            let _ = started.send((stage, fast));
            if stage == 1 && !fast {
                let (lock, changed) = &*bg_wait;
                let _guard = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |released| !*released).unwrap();
            }
            Some((Vec::new(), Poll::Refused("test".into())))
        }).unwrap();

        let key1 = queue.register(&service, pipeline(&[(StageKind::Compute, &[1])]), 0).unwrap();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (1, false));
        queue.note_bind();
        let large = queue.register(&service, pipeline(&[(StageKind::Compute, &[9])]), 1).unwrap();
        // Simulate the fast lane taking `large` before its first bind (`popped.urgent == false`),
        // then the title binding `large` while it is `Running`, and the fast lane handing it off.
        let popped = {
            let mut state = queue.state.lock().unwrap();
            let popped = state.schedule.pop(true, false, true).unwrap();
            assert_eq!((popped.key, popped.urgent), (large, false));
            state.background_running += 1;
            state.slots.insert(large, runtime::Status::Running { thread: 0, boosted: false });
            popped
        };
        queue.note_bind();
        assert_eq!(queue.bind(&service, large, Duration::ZERO), Some(Poll::Pending));
        queue.finish_popped(popped, None, true);
        // Keep the title busy while worker 1 still holds its background compile (`background_running == 1`):
        // worker 2 can only pop `large` if `requeue_large` preserved first-bind urgency.
        let mut taken_by_bulk = None;
        for _ in 0..20 {
            queue.note_bind();
            if let Ok(item) = jobs.recv_timeout(Duration::from_millis(50)) {
                taken_by_bulk = Some(item);
                break;
            }
        }
        let (lock, changed) = &*release_bg;
        *lock.lock().unwrap() = true;
        changed.notify_one();
        assert_eq!(taken_by_bulk, Some((9, false)), "bulk worker takes the bound large job despite busy gate");
        assert_eq!(queue.bind(&service, large, Duration::from_secs(2)), Some(Poll::Refused(String::new())));
        assert_eq!(queue.bind(&service, key1, Duration::from_secs(2)), Some(Poll::Refused(String::new())));
    }

    #[cfg(windows)]
    #[test]
    fn worker_zero_falls_back_to_all_jobs_when_bulk_worker_spawn_fails() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-spawn-fallback");
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let queue = Queue::start_with(
            DiskCache::new(&dir.0),
            2,
            Limits::SERIAL,
            move |job, fast| {
                let stage = job.pipeline.stages[0].bytes[0];
                let _ = started.send((stage, fast));
                if fast {
                    return None;
                }
                Some((Vec::new(), Poll::Refused("test".into())))
            },
            |i, role, f| {
                if i > 0 {
                    return Err(io::Error::other("simulated bulk worker spawn failure"));
                }
                assert_eq!(role, Role::All);
                std::thread::Builder::new().name(format!("decant-precompile-test-{i}")).spawn(f).map(|_| ())
            },
        )
        .unwrap();

        let large = queue.register(&service, pipeline(&[(StageKind::Compute, &[9])]), 0).unwrap();
        queue.bind(&service, large, Duration::ZERO);
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), (9, false));
        assert_eq!(queue.bind(&service, large, Duration::from_secs(2)), Some(Poll::Refused(String::new())));
    }

    #[cfg(windows)]
    #[test]
    fn a_warm_disk_record_is_ready_while_every_worker_is_busy() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::atomic::Ordering;
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-warm");
        let cache = DiskCache::new(&dir.0);
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let blocked = release.clone();
        let queue = Queue::start(cache.clone(), 1, move |job, _| {
            let _ = started.send(job.pipeline.stages[0].bytes[0]);
            let (lock, changed) = &*blocked;
            let _guard = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |released| !*released).unwrap();
            Some((Vec::new(), Poll::Refused("test".into())))
        }).unwrap();
        queue.register(&service, pipeline(&[(StageKind::Compute, &[1])]), 0).unwrap();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), 1);

        // Warm at registration.
        let warm = pipeline(&[(StageKind::Compute, &[3])]);
        cache.write(0x33, b"DXBCwarm").unwrap();
        cache.write_pipeline(service.input_key(&warm), &[0x33]).unwrap();
        let key = queue.register(&service, warm, 1).unwrap();
        assert_eq!(queue.bind(&service, key, Duration::ZERO), Some(Poll::Ready(vec![Arc::from(&b"DXBCwarm"[..])])));

        // Warm only by the time of the bind.
        let late = pipeline(&[(StageKind::Compute, &[4])]);
        let late_key = queue.register(&service, late.clone(), 2).unwrap();
        cache.write(0x44, b"DXBClate").unwrap();
        cache.write_pipeline(service.input_key(&late), &[0x44]).unwrap();
        assert_eq!(queue.bind(&service, late_key, Duration::ZERO), Some(Poll::Ready(vec![Arc::from(&b"DXBClate"[..])])));
        assert_eq!(service.counters().pipe_hits.load(Ordering::Relaxed), 2);

        let (lock, changed) = &*release;
        *lock.lock().unwrap() = true;
        changed.notify_one();
        assert!(jobs.recv_timeout(Duration::from_millis(300)).is_err(), "no worker ever compiles the warm pipelines");
    }

    #[cfg(windows)]
    #[test]
    fn a_queued_pipeline_uses_the_compile_service_and_disk_cache() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::direct_fixture::{container, mesh_triangle};
        use crate::rts0::{RegisterClass, RootParam, RootParamKind};
        use crate::shader_source::StageKind;
        use std::sync::atomic::Ordering;
        use std::sync::{Arc, mpsc};
        use std::time::Duration;

        let dir = TempDir::new("precompile-dxil");
        let cache = DiskCache::new(&dir.0);
        let mut settings = Settings::quiet(1);
        settings.cache = Some(cache.clone());
        let compiler = Arc::new(FakeCompiler::default());
        let source = source();
        let service = Arc::new(Service::new(source.clone(), compiler.clone(), settings));
        let worker = service.clone();
        let (finished, done) = mpsc::channel();
        let queue = Queue::start(cache.clone(), 1, move |job, fast| {
            let answer = run_job(&worker, &source, job, fast);
            if let Some((keys, _)) = &answer {
                finished.send(keys.clone()).unwrap();
            }
            answer
        })
        .unwrap();
        let mut mesh = pipeline(&[(StageKind::Mesh, &container(&mesh_triangle()))]);
        mesh.root.flags = 0xC00;
        mesh.root.params.push(RootParam {
            kind: RootParamKind::Descriptor {
                class: RegisterClass::Cbv,
                register: 1,
                space: 15,
            },
            visibility: 0,
        });
        let key = queue.register(&service, mesh.clone(), 0).unwrap();
        assert_eq!(queue.register(&service, mesh, 1), Some(key));
        let stage_keys = done.recv_timeout(Duration::from_secs(2)).unwrap();
        let Some(Poll::Ready(dxil)) = queue.bind(&service, key, Duration::from_secs(1)) else {
            panic!("compiled pipeline is unavailable");
        };
        assert_eq!(&*dxil[0], b"DXBCms_6_6");
        assert_eq!(cache.read(stage_keys[0]).as_deref(), Some(dxil[0].as_ref()));
        assert_eq!(compiler.calls.load(Ordering::Relaxed), 1);
        assert_eq!(service.counters().pipe_hits.load(Ordering::Relaxed), 0, "a compiled pipeline is no pipeline hit");
    }

    #[test]
    fn limits_follow_the_core_count_and_never_exceed_the_workers() {
        // Up to 7 cores keep the serial gate: one background job while the title binds.
        for cores in [1, 2, 4, 7] {
            assert_eq!(Limits::for_machine(4, cores), Limits::SERIAL, "{cores} cores");
        }
        assert_eq!(Limits::for_machine(4, 8), Limits { busy_background: 2, urgent_yield: 1 });
        assert_eq!(Limits::for_machine(4, 16), Limits { busy_background: 4, urgent_yield: 2 });
        assert_eq!(Limits::for_machine(4, 64), Limits { busy_background: 4, urgent_yield: 8 }, "capped at the workers");
        assert_eq!(Limits::for_machine(0, 64).busy_background, 1, "at least one");
    }

    #[test]
    fn a_wide_machine_runs_more_background_jobs_while_the_title_binds() {
        let wide = Limits::for_machine(4, 16);
        // A busy title with two background jobs running: serial machines stop, wide ones go on.
        assert!(!Role::Bulk.background(true, 2, 0, Limits::SERIAL));
        assert!(Role::Bulk.background(true, 2, 0, wide));
        assert!(Role::All.background(true, 3, 0, wide));
        assert!(!Role::Bulk.background(true, 4, 0, wide), "the limit still holds");
        // One urgent job holds background work back on a serial machine, not on a wide one.
        assert!(!Role::Bulk.background(false, 0, 1, Limits::SERIAL));
        assert!(Role::Bulk.background(false, 0, 1, wide));
        assert!(!Role::Bulk.background(false, 0, 2, wide));
        // The fast lane never takes background work.
        assert!(!Role::Fast.background(false, 0, 0, wide));
    }

    #[test]
    fn progress_counts_the_shares_and_prints_one_line() {
        let progress = Progress { queued: 6, active: 2, ready: 10, cached: 4, failed: 2 };
        assert_eq!(progress.total(), 20);
        assert_eq!(progress.percent(), 60);
        assert!(!progress.drained());
        assert_eq!(
            progress.line(),
            "direct precompile: warmup 12 of 20 pipelines done (60%), 6 queued, 2 active, 10 ready (4 from cache), 2 failed"
        );
        let idle = Progress { ready: 3, failed: 1, ..Progress::default() };
        assert!(idle.drained());
        assert!(idle.line().ends_with("3 ready (0 from cache), 1 failed, queue is idle"), "{}", idle.line());
        assert_eq!(Progress::default().percent(), 100);
    }

    #[test]
    fn the_progress_log_speaks_once_per_change_and_never_for_an_empty_queue() {
        let mut log = ProgressLog::default();
        assert_eq!(log.next(Progress::default()), None, "nothing was ever queued");
        let first = Progress { queued: 5, ..Progress::default() };
        assert_eq!(log.next(first), Some(first.line()));
        assert_eq!(log.next(first), None, "no change, no line");
        let second = Progress { queued: 3, active: 2, ..Progress::default() };
        assert_eq!(log.next(second), Some(second.line()));
        let done = Progress { ready: 4, failed: 1, ..Progress::default() };
        assert_eq!(log.next(done), Some(done.line()));
        assert_eq!(log.next(done), None, "an idle queue stays quiet");
    }

    #[cfg(windows)]
    #[test]
    fn the_queue_counts_queued_active_ready_and_failed_pipelines() {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::time::{Duration, Instant};

        let dir = TempDir::new("precompile-progress");
        let cache = DiskCache::new(&dir.0);
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (started, jobs) = mpsc::channel();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let blocked = release.clone();
        let queue = Queue::start(cache.clone(), 1, move |job, _| {
            let stage = job.pipeline.stages[0].bytes[0];
            let _ = started.send(stage);
            if stage == 1 {
                let (lock, changed) = &*blocked;
                let _guard = changed.wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |released| !*released).unwrap();
            }
            match stage {
                3 => Some((Vec::new(), Poll::Refused("test".into()))),
                _ => Some((vec![u128::from(stage)], Poll::Ready(Vec::new()))),
            }
        }).unwrap();
        let progress = || queue.state.lock().unwrap().progress();
        assert_eq!(progress(), Progress::default());

        queue.register(&service, pipeline(&[(StageKind::Compute, &[1])]), 0).unwrap();
        assert_eq!(jobs.recv_timeout(Duration::from_secs(2)).unwrap(), 1);
        queue.register(&service, pipeline(&[(StageKind::Compute, &[2])]), 1).unwrap();
        queue.register(&service, pipeline(&[(StageKind::Compute, &[3])]), 2).unwrap();
        // One pipeline with a disk record is ready at registration and counts as cached.
        let warm = pipeline(&[(StageKind::Compute, &[4])]);
        cache.write(0x44, b"DXBCwarm").unwrap();
        cache.write_pipeline(service.input_key(&warm), &[0x44]).unwrap();
        queue.register(&service, warm, 3).unwrap();
        assert_eq!(progress(), Progress { queued: 2, active: 1, ready: 1, cached: 1, failed: 0 });

        let (lock, changed) = &*release;
        *lock.lock().unwrap() = true;
        changed.notify_all();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !progress().drained() {
            assert!(Instant::now() < deadline, "the queue never drained: {:?}", progress());
            std::thread::sleep(Duration::from_millis(5));
        }
        let done = progress();
        assert_eq!(done, Progress { queued: 0, active: 0, ready: 3, cached: 1, failed: 1 });
        assert_eq!(done.percent(), 100);
        queue.log_progress();
    }

    /// Runs `count` background compute jobs on a queue of 4 workers while the title keeps binding,
    /// and returns the most jobs that ran at the same time. With `meet` the jobs wait for each
    /// other for up to 2 seconds, else each one takes 100 ms.
    #[cfg(windows)]
    fn busiest_background(limits: Limits, count: u8, meet: bool) -> usize {
        use crate::compile_service::tests::{FakeCompiler, pipeline};
        use crate::compile_service::{Poll, Service, Settings};
        use crate::direct_cache::DiskCache;
        use crate::shader_source::StageKind;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, mpsc};
        use std::time::{Duration, Instant};

        let dir = TempDir::new("precompile-parallel");
        let service = Service::new(source(), Arc::new(FakeCompiler::default()), Settings::quiet(1));
        let (running, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (done, finished) = mpsc::channel();
        let (now, high) = (running.clone(), peak.clone());
        let queue = Queue::start_with(
            DiskCache::new(&dir.0),
            4,
            limits,
            move |_, _| {
                high.fetch_max(now.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                if meet {
                    let deadline = Instant::now() + Duration::from_secs(2);
                    while now.load(Ordering::SeqCst) < usize::from(count) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(2));
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(100));
                }
                now.fetch_sub(1, Ordering::SeqCst);
                let _ = done.send(());
                Some((Vec::new(), Poll::Refused("test".into())))
            },
            |i, _, f| std::thread::Builder::new().name(format!("decant-precompile-test-{i}")).spawn(f).map(|_| ()),
        )
        .unwrap();
        // The title binds, so the busy gate is closed to all but `limits.busy_background` jobs.
        queue.note_bind();
        for stage in 1..=count {
            queue.register(&service, pipeline(&[(StageKind::Compute, &[stage])]), u32::from(stage)).unwrap();
        }
        for _ in 0..count {
            queue.note_bind();
            finished.recv_timeout(Duration::from_secs(10)).expect("a job never finished");
        }
        peak.load(Ordering::SeqCst)
    }

    #[cfg(windows)]
    #[test]
    fn a_wide_machine_compiles_background_jobs_in_parallel_while_the_title_binds() {
        // Three bulk workers and room for three background jobs: all three run together.
        assert_eq!(busiest_background(Limits { busy_background: 3, urgent_yield: 1 }, 3, true), 3);
    }

    #[cfg(windows)]
    #[test]
    fn a_serial_gate_keeps_one_background_job_while_the_title_binds() {
        assert_eq!(busiest_background(Limits::SERIAL, 3, false), 1);
    }

    #[test]
    fn bind_history_round_trips_and_ignores_corrupt_or_empty_files() {
        let dir = TempDir::new("bind-order");
        write_history(&dir.0, &[71, 42, u128::MAX]).unwrap();
        assert_eq!(read_history(&dir.0), Some(vec![71, 42, u128::MAX]));
        write_history(&dir.0, &[42]).unwrap();
        assert_eq!(read_history(&dir.0), Some(vec![42]));
        let path = dir.0.join(NAME);
        let mut corrupt = fs::read(&path).unwrap();
        corrupt[12] ^= 1;
        fs::write(&path, corrupt).unwrap();
        assert_eq!(read_history(&dir.0), None);
        fs::write(&path, []).unwrap();
        assert_eq!(read_history(&dir.0), None);
        write_history(&dir.0, &[]).unwrap();
        assert_eq!(read_history(&dir.0), Some(Vec::new()));
    }
}
