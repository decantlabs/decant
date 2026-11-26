//! The decode pool: a few threads shared by every queue, which decode what the queues' workers
//! have read (zlib, GDeflate, BCPack) and write it to its destination. A BCPack stream decodes on
//! one thread, so the title's many requests are what runs in parallel.
//!
//! The threads start on first use and live as long as the process, scaling with logical
//! processors (clamped to `4..=24`) so decodes saturate available cores.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

/// One decode.
pub type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    idle_workers: AtomicUsize,
    ready: Condvar,
}

/// The pool, or `None` when not one of its threads could start.
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool { jobs: Mutex::new(VecDeque::new()), idle_workers: AtomicUsize::new(0), ready: Condvar::new() }));
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(4, 24);
        let started = (0..threads).filter(|k| std::thread::Builder::new().name(format!("dstorage decode {k}")).spawn(move || work(pool)).is_ok()).count();
        (started > 0).then_some(pool)
    })
}

fn work(pool: &Pool) {
    loop {
        let job = {
            let mut jobs = pool.jobs.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                if let Some(job) = jobs.pop_front() {
                    break job;
                }
                pool.idle_workers.fetch_add(1, Ordering::AcqRel);
                jobs = pool.ready.wait(jobs).unwrap_or_else(PoisonError::into_inner);
                pool.idle_workers.fetch_sub(1, Ordering::AcqRel);
            }
        };
        job();
    }
}

/// Runs `job` on a decode thread, or here when the pool has no thread.
pub fn run(job: Job) {
    let Some(pool) = pool() else { return job() };
    pool.jobs.lock().unwrap_or_else(PoisonError::into_inner).push_back(job);
    if pool.idle_workers.load(Ordering::Acquire) > 0 {
        pool.ready.notify_one();
    }
}

/// Enqueues a batch of decode jobs under a single lock acquisition.
pub fn run_batch<I: IntoIterator<Item = Job>>(jobs_iter: I) {
    let Some(pool) = pool() else {
        for job in jobs_iter {
            job();
        }
        return;
    };
    let count = {
        let mut jobs = pool.jobs.lock().unwrap_or_else(PoisonError::into_inner);
        let before = jobs.len();
        jobs.extend(jobs_iter);
        jobs.len() - before
    };
    if count > 0 && pool.idle_workers.load(Ordering::Acquire) > 0 {
        if count == 1 {
            pool.ready.notify_one();
        } else {
            pool.ready.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn run_batch_executes_all_jobs() {
        let done = Arc::new(AtomicUsize::new(0));
        let jobs = (0..16).map(|_| {
            let done = done.clone();
            Box::new(move || {
                done.fetch_add(1, Ordering::Relaxed);
            }) as super::Job
        });
        super::run_batch(jobs);
        let deadline = Instant::now() + Duration::from_secs(5);
        while done.load(Ordering::Relaxed) < 16 {
            assert!(Instant::now() < deadline, "batch jobs timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(done.load(Ordering::Relaxed), 16);
    }
}
