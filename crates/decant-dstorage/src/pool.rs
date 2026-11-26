//! The decode pool: a few threads shared by every queue, which decode what the queues' workers
//! have read (zlib, GDeflate, BCPack) and write it to its destination. A BCPack stream decodes on
//! one thread, so the title's many requests are what runs in parallel.
//!
//! The threads start on first use and live as long as the process. Their number is three
//! quarters of the logical processors, at least 4 and at most 16, so decodes scale across cores.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

/// One decode.
pub type Job = Box<dyn FnOnce() + Send>;

struct Pool {
    jobs: Mutex<VecDeque<Job>>,
    ready: Condvar,
}

/// The pool, or `None` when not one of its threads could start.
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool { jobs: Mutex::new(VecDeque::new()), ready: Condvar::new() }));
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get() * 3 / 4).clamp(4, 16);
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
                jobs = pool.ready.wait(jobs).unwrap_or_else(PoisonError::into_inner);
            }
        };
        job();
    }
}

/// Runs `job` on a decode thread, or here when the pool has no thread.
pub fn run(job: Job) {
    let Some(pool) = pool() else { return job() };
    pool.jobs.lock().unwrap_or_else(PoisonError::into_inner).push_back(job);
    pool.ready.notify_one();
}
