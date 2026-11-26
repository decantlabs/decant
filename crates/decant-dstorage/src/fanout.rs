//! A persistent pool of helper threads for splitting one decode into independent parts (the tiles
//! of a GDeflate stream) without starting an OS thread per decode.
//!
//! [`parallel_for`] runs `f(0..tasks)`. The caller always takes part: it claims tasks itself until
//! none are left, so a call finishes even when every helper is busy, and a call made from inside a
//! task (or from a thread of the decode pool) cannot deadlock waiting for a helper. The helpers are
//! separate from the decode pool of `pool` for the same reason: a decode job that fans out never
//! queues behind jobs that are themselves waiting for it.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

/// Most helper threads, in addition to the callers.
const MAX_HELPERS: usize = 15;

/// A task function with its lifetime erased. It is only called for a task claimed before the batch
/// was exhausted, and `parallel_for` does not return before every claimed task finished.
#[derive(Clone, Copy)]
struct TaskFn(*const (dyn Fn(usize) + Sync));

// SAFETY: the function is `Sync`, and `parallel_for` keeps it alive while any task can still run.
unsafe impl Send for TaskFn {}
// SAFETY: as above.
unsafe impl Sync for TaskFn {}

struct Batch {
    task: TaskFn,
    total: usize,
    next: AtomicUsize,
    finished: Mutex<usize>,
    all_done: Condvar,
    panicked: AtomicBool,
}

impl Batch {
    /// Claims and runs tasks until none are left.
    fn drive(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::AcqRel);
            if index >= self.total {
                return;
            }
            // SAFETY: `index < total` was claimed, so `parallel_for` is still waiting for this task and `f` is alive.
            let task = unsafe { &*self.task.0 };
            if catch_unwind(AssertUnwindSafe(|| task(index))).is_err() {
                self.panicked.store(true, Ordering::Release);
            }
            let mut finished = self.finished.lock().unwrap_or_else(PoisonError::into_inner);
            *finished += 1;
            if *finished == self.total {
                self.all_done.notify_all();
            }
        }
    }

    fn exhausted(&self) -> bool {
        self.next.load(Ordering::Acquire) >= self.total
    }
}

struct Pool {
    batches: Mutex<VecDeque<Arc<Batch>>>,
    ready: Condvar,
    helpers: usize,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let wanted = std::thread::available_parallelism().map_or(2, |n| n.get()).saturating_sub(1).clamp(1, MAX_HELPERS);
        let pool: &'static Pool = Box::leak(Box::new(Pool { batches: Mutex::new(VecDeque::new()), ready: Condvar::new(), helpers: wanted }));
        for k in 0..wanted {
            // A helper that cannot start only means fewer helpers: callers do the work themselves.
            let _ = std::thread::Builder::new().name(format!("dstorage fanout {k}")).spawn(move || helper(pool));
        }
        pool
    })
}

fn helper(pool: &Pool) {
    loop {
        let batch = {
            let mut batches = pool.batches.lock().unwrap_or_else(PoisonError::into_inner);
            loop {
                while batches.front().is_some_and(|b| b.exhausted()) {
                    batches.pop_front();
                }
                if let Some(batch) = batches.front() {
                    break Arc::clone(batch);
                }
                batches = pool.ready.wait(batches).unwrap_or_else(PoisonError::into_inner);
            }
        };
        batch.drive();
    }
}

/// Runs `f(0)`, `f(1)`, ..., `f(tasks - 1)`, on the persistent helpers and the calling thread, and
/// returns when every one has finished. Panics when a task panicked.
pub fn parallel_for<F: Fn(usize) + Sync>(tasks: usize, f: F) {
    if tasks <= 1 {
        if tasks == 1 {
            f(0);
        }
        return;
    }
    let pool = pool();
    let erased: &(dyn Fn(usize) + Sync) = &f;
    // SAFETY: only the lifetime is erased; this function waits for every claimed task before it returns, and a task is never claimed after that.
    let task = TaskFn(unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), *const (dyn Fn(usize) + Sync)>(erased) });
    let batch = Arc::new(Batch { task, total: tasks, next: AtomicUsize::new(0), finished: Mutex::new(0), all_done: Condvar::new(), panicked: AtomicBool::new(false) });
    {
        let mut batches = pool.batches.lock().unwrap_or_else(PoisonError::into_inner);
        batches.push_back(Arc::clone(&batch));
    }
    let wake = (tasks - 1).min(pool.helpers);
    if wake >= pool.helpers {
        pool.ready.notify_all();
    } else {
        for _ in 0..wake {
            pool.ready.notify_one();
        }
    }
    batch.drive();
    let mut finished = batch.finished.lock().unwrap_or_else(PoisonError::into_inner);
    while *finished < tasks {
        finished = batch.all_done.wait(finished).unwrap_or_else(PoisonError::into_inner);
    }
    drop(finished);
    assert!(!batch.panicked.load(Ordering::Acquire), "a fan-out task panicked");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::parallel_for;

    #[test]
    fn every_task_runs_exactly_once() {
        for tasks in [0usize, 1, 2, 7, 64, 500] {
            let hits: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(tasks, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "tasks={tasks}");
        }
    }

    #[test]
    fn tasks_can_write_disjoint_parts_of_a_buffer() {
        let mut data = vec![0u8; 8 * 1024];
        let base = data.as_mut_ptr() as usize;
        parallel_for(8, |i| {
            // SAFETY: each task owns its own 1 KiB part of `data`, which outlives the call.
            let part = unsafe { std::slice::from_raw_parts_mut((base as *mut u8).add(i * 1024), 1024) };
            part.fill(i as u8 + 1);
        });
        for (i, chunk) in data.chunks(1024).enumerate() {
            assert!(chunk.iter().all(|&b| b == i as u8 + 1));
        }
    }

    #[test]
    fn a_task_can_fan_out_again_without_deadlock() {
        let total = AtomicUsize::new(0);
        // More outer tasks than there can be helpers, each fanning out again from a helper thread.
        parallel_for(64, |_| {
            parallel_for(8, |_| {
                parallel_for(3, |_| {
                    total.fetch_add(1, Ordering::Relaxed);
                });
            });
        });
        assert_eq!(total.load(Ordering::Relaxed), 64 * 8 * 3);
    }

    #[test]
    fn concurrent_callers_all_finish() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        parallel_for(5, |_| {
                            total.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 16 * 50 * 5);
    }

    #[test]
    fn a_panicking_task_is_reported_and_the_pool_survives() {
        let result = std::panic::catch_unwind(|| parallel_for(4, |i| assert!(i != 2, "boom")));
        assert!(result.is_err());
        let done = AtomicUsize::new(0);
        parallel_for(4, |_| {
            done.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(done.load(Ordering::Relaxed), 4);
    }
}
