//! Running independent pieces of a scan on every core the host makes available.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

/// Worker threads for a scan: what the operating system reports as available to this process
/// (`std::thread::available_parallelism` follows CPU affinity and container quotas), or
/// `DECANT_THREADS` when set to a positive number.
pub fn threads() -> usize {
    let from_env = std::env::var("DECANT_THREADS").ok().and_then(|v| v.trim().parse::<NonZeroUsize>().ok());
    from_env.or_else(|| std::thread::available_parallelism().ok()).map_or(1, NonZeroUsize::get)
}

/// Runs `work(i)` for every `i` in `0..count` on up to `threads()` workers and returns the results
/// in index order. `weight(i)` is the share of the total each piece stands for, reported through
/// `progress` as pieces finish. `None` when `cancel` was set or a piece returned `None`.
pub fn map_pieces<T: Send>(
    count: usize,
    weight: impl Fn(usize) -> f64,
    work: impl Fn(usize) -> Option<T> + Sync,
    progress: &mut dyn FnMut(f64),
    cancel: &AtomicBool,
) -> Option<Vec<T>> {
    let workers = threads().min(count).max(1);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let mut results: Vec<Option<T>> = std::iter::repeat_with(|| None).take(count).collect();
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel::<(usize, Option<T>)>();
        for _ in 0..workers {
            let tx = tx.clone();
            let (next, work, failed) = (&next, &work, &failed);
            scope.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= count || cancel.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let result = work(i);
                    if result.is_none() {
                        failed.store(true, Ordering::Relaxed);
                    }
                    if tx.send((i, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0.0;
        for (i, result) in rx {
            done += weight(i);
            results[i] = result;
            progress(done.min(1.0));
        }
    });
    if cancel.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed) {
        return None;
    }
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_order_and_progress_reaches_one() {
        let mut last = 0.0;
        let out = map_pieces(100, |_| 0.01, |i| Some(i * 2), &mut |p| last = p, &AtomicBool::new(false)).unwrap();
        assert_eq!(out, (0..100).map(|i| i * 2).collect::<Vec<_>>());
        assert!((last - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_failed_piece_or_a_cancel_gives_none() {
        assert!(map_pieces(10, |_| 0.1, |i| (i != 3).then_some(i), &mut |_| {}, &AtomicBool::new(false)).is_none());
        assert!(map_pieces(10, |_| 0.1, Some, &mut |_| {}, &AtomicBool::new(true)).is_none());
        assert!(threads() >= 1);
    }
}
