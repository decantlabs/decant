//! Per-list global UAV barrier deduplication.
//!
//! Ported from vitutu1/victory-perf (GPL-3.0-or-later) and adapted to this codebase.
//!
//! On the console titles and middleware emit global UAV barriers conservatively between
//! passes. Consecutive global UAV barriers with no shader draw, compute/mesh dispatch, indirect
//! execution, UAV clear, or memory-copy shader in between flush the GPU pipeline redundantly.
//!
//! Each command list starts conservatively in the `pending` state on creation and after every
//! `Reset`, so the first global UAV barrier on a list never drops a cross-list UAV hazard. Only
//! a list where a global UAV barrier has been emitted with no intervening work is tracked in
//! `CLEAN`; any draw, dispatch, indirect execution, UAV clear, or memory-fill/copy pass clears
//! its clean bit (re-marking UAV writes pending).
//!
//! Switch: `DECANT_D3D12X_BARRIER_MERGE=0` disables deduplication (default on; `0` = forward every
//! barrier unchanged).

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
use crate::log::xs_log;
#[cfg(not(windows))]
macro_rules! xs_log {
    ($($t:tt)*) => {};
}

/// Size in bytes of one `D3D12_RESOURCE_BARRIER` on x86-64 (Type 4, Flags 4, union 24).
pub const BARRIER_BYTES: usize = 32;
/// `D3D12_RESOURCE_BARRIER_TYPE_UAV`.
const BARRIER_TYPE_UAV: u32 = 2;
/// Log summary cadence in total barrier decisions.
const LOG_EVERY: u64 = 4096;

/// Lists that have emitted a global UAV barrier with no subsequent UAV-writing command.
/// Absence means `uav_pending == true` (conservative default for new/reset lists).
static CLEAN: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// Fast-path count of entries in `CLEAN` so `mark_pending` costs one relaxed load when empty.
static CLEAN_LEN: AtomicUsize = AtomicUsize::new(0);
static FIRST_SKIP_LOGGED: AtomicBool = AtomicBool::new(false);

/// `[title_recorded, title_skipped, clamp_recorded, clamp_skipped]`.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether global UAV barrier deduplication is enabled (`DECANT_D3D12X_BARRIER_MERGE`, default on;
/// `0` = off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("DECANT_D3D12X_BARRIER_MERGE").is_none_or(|v| v != "0");
        if !on {
            xs_log!("uav barrier merge: off (DECANT_D3D12X_BARRIER_MERGE=0)");
        }
        on
    })
}

fn with_clean<R>(f: impl FnOnce(&mut HashSet<usize>) -> R) -> R {
    let mut guard = CLEAN.lock().unwrap_or_else(|e| e.into_inner());
    let set = guard.get_or_insert_with(HashSet::new);
    let out = f(set);
    CLEAN_LEN.store(set.len(), Ordering::Relaxed);
    out
}

/// Returns whether `list` may have pending UAV writes (`true` when unknown, new, or reset).
pub fn is_pending(list: usize) -> bool {
    if !enabled() || CLEAN_LEN.load(Ordering::Relaxed) == 0 {
        return true;
    }
    with_clean(|s| !s.contains(&list))
}

/// Marks `list` as having pending UAV writes (called on every draw, dispatch, indirect call, UAV
/// clear, or memory copy/fill shader recorded on `list`).
#[inline]
pub fn mark_pending(list: usize) {
    if CLEAN_LEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    with_clean(|s| {
        s.remove(&list);
    });
}

/// Marks `list` clean after a global UAV barrier has been emitted on it (used by indirect clamp
/// when its pre-wait was a global UAV barrier).
pub fn clear_pending(list: usize) {
    if !enabled() {
        return;
    }
    with_clean(|s| {
        s.insert(list);
    });
}

/// Resets `list` back to the conservative `pending` state (called on `Reset` and `Release`).
#[inline]
pub fn reset(list: usize) {
    mark_pending(list);
}

/// Decides whether a title global UAV barrier on `list` must be recorded (`true`) or can be
/// skipped (`false`, because no UAV-writing work ran since the last global UAV barrier on `list`).
pub fn take_global(list: usize) -> bool {
    if !enabled() {
        return true;
    }
    let emit = with_clean(|s| s.insert(list));
    let total = if emit {
        STATS[0].fetch_add(1, Ordering::Relaxed) + 1 + STATS[1].load(Ordering::Relaxed)
    } else {
        let skipped = STATS[1].fetch_add(1, Ordering::Relaxed) + 1;
        if !FIRST_SKIP_LOGGED.swap(true, Ordering::Relaxed) {
            xs_log!("uav barrier merge: active (DECANT_D3D12X_BARRIER_MERGE=1), first redundant global UAV barrier skipped");
        }
        STATS[0].load(Ordering::Relaxed) + skipped
    };
    if total.is_multiple_of(LOG_EVERY) {
        log_summary();
    }
    emit
}

/// Records indirect-clamp barrier accounting (`recorded` global/scoped pre-waits, `skipped`
/// omitted pre/post global UAV barriers).
pub fn note_clamp(recorded: u64, skipped: u64) {
    if !enabled() {
        return;
    }
    if recorded > 0 {
        STATS[2].fetch_add(recorded, Ordering::Relaxed);
    }
    if skipped > 0 {
        STATS[3].fetch_add(skipped, Ordering::Relaxed);
    }
}

/// Returns whether raw 72-byte `D3D12_RESOURCE_BARRIER` `b` is a global UAV barrier
/// (`Type == D3D12_RESOURCE_BARRIER_TYPE_UAV` and `pResource == NULL`).
#[inline]
pub fn is_global_uav_barrier(b: &[u8; BARRIER_BYTES]) -> bool {
    let ty = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    if ty != BARRIER_TYPE_UAV {
        return false;
    }
    let res = u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    res == 0
}

/// Filters redundant global UAV barriers out of `barriers` in place while preserving all
/// transition, aliasing, and resource-specific UAV barriers. Returns the number of dropped
/// barriers.
pub fn filter_barriers(list: usize, barriers: &mut Vec<[u8; BARRIER_BYTES]>) -> usize {
    if !enabled() || barriers.is_empty() {
        return 0;
    }
    let before = barriers.len();
    barriers.retain(|b| !is_global_uav_barrier(b) || take_global(list));
    before - barriers.len()
}

/// Emits a cumulative summary line of global UAV barriers recorded vs skipped.
pub fn log_summary() {
    #[cfg(windows)]
    {
        let rec = STATS[0].load(Ordering::Relaxed);
        let skip = STATS[1].load(Ordering::Relaxed);
        let crec = STATS[2].load(Ordering::Relaxed);
        let cskip = STATS[3].load(Ordering::Relaxed);
        xs_log!("uav barrier merge: {rec} recorded / {skip} skipped title global UAV barriers; indirect clamps {crec} recorded / {cskip} skipped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_barrier(ty: u32, res: u64) -> [u8; BARRIER_BYTES] {
        let mut b = [0u8; BARRIER_BYTES];
        b[0..4].copy_from_slice(&ty.to_le_bytes());
        b[8..16].copy_from_slice(&res.to_le_bytes());
        b
    }

    #[test]
    fn consecutive_global_uav_barriers_are_deduplicated_until_work_or_reset() {
        let list = 0xDEAD_1001;
        reset(list);
        assert!(is_pending(list), "new/reset list starts conservatively pending");
        assert!(take_global(list), "first global UAV barrier emits");
        assert!(!is_pending(list));
        assert!(!take_global(list), "back-to-back global UAV barrier is skipped");
        mark_pending(list);
        assert!(is_pending(list));
        assert!(take_global(list), "global UAV barrier emits again after work");
        assert!(!take_global(list), "subsequent global UAV barrier skipped again");
        reset(list);
        assert!(take_global(list), "reset restores conservative pending state");
        reset(list);
    }

    #[test]
    fn filter_barriers_keeps_transitions_and_specific_uav_and_drops_redundant_globals() {
        let list = 0xDEAD_1002;
        reset(list);
        let transition = make_barrier(0, 0x1000);
        let specific_uav = make_barrier(2, 0x2000);
        let global_uav = make_barrier(2, 0);
        let aliasing = make_barrier(1, 0);

        let mut batch = vec![transition, global_uav, specific_uav, global_uav, aliasing];
        let dropped = filter_barriers(list, &mut batch);
        assert_eq!(dropped, 1, "second global UAV barrier in same batch dropped");
        assert_eq!(batch, vec![transition, global_uav, specific_uav, aliasing]);

        // Next batch with no intervening work drops its global UAV barrier too.
        let mut batch2 = vec![global_uav];
        assert_eq!(filter_barriers(list, &mut batch2), 1);
        assert!(batch2.is_empty());

        // After work, global UAV barrier passes again.
        mark_pending(list);
        let mut batch3 = vec![global_uav];
        assert_eq!(filter_barriers(list, &mut batch3), 0);
        assert_eq!(batch3.len(), 1);
        reset(list);
    }
}
