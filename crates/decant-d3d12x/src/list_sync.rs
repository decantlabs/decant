//! What a title records on a command list that the PC GPU cannot carry out itself, kept per list.
//!
//! A console title marks GPU progress by writing values into its own memory from the bottom of the
//! pipe (`Write32/64BitValueBottomOfPipeX`, list slots 71 and 72 of the newer layout) and waits for
//! such values (`Wait32/64BitValueX`, slots 73 and 74). The PC GPU has no access to that memory, so
//! the layer keeps the writes and the waits next to the list that recorded them. The CPU makes the
//! writes once the submission has finished (`cpu_writes`) and checks the waits before it submits.
//! A submission whose waits are not met yet is queued on its queue (`queue_fifo`) and its worker
//! waits for them, so the title's thread never waits on a memory wait.
//!
//! The state is a table keyed by the address of the list proxy. A reset of the list drops all of it.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// How long a memory wait may hold a list back when nothing that could still write the value is
/// on its way (no write of `cpu_writes` pending, no work queued on another queue).
pub const WAIT_IDLE: Duration = Duration::from_millis(200);
/// How long a memory wait may hold a list back at most before the list runs anyway.
pub const WAIT_HARD: Duration = Duration::from_secs(2);

/// The limits of one memory wait (see `cpu_writes::wait_until`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitLimits {
    pub idle: Duration,
    pub hard: Duration,
}

/// The limits from the settings `DECANT_D3D12X_MEMORY_WAIT_MS` (hard) and
/// `DECANT_D3D12X_MEMORY_WAIT_IDLE_MS` (idle), given as milliseconds. A missing or unparsable
/// setting keeps its default. The idle limit is never above the hard one.
pub fn limits_from(hard: Option<&str>, idle: Option<&str>) -> WaitLimits {
    let ms = |text: Option<&str>| text.and_then(|t| t.trim().parse::<u64>().ok()).map(Duration::from_millis);
    let hard = ms(hard).unwrap_or(WAIT_HARD);
    let idle = ms(idle).unwrap_or(WAIT_IDLE).min(hard);
    WaitLimits { idle, hard }
}

/// The limits of the memory waits of this process, read from the settings once.
pub fn wait_limits() -> WaitLimits {
    static LIMITS: OnceLock<WaitLimits> = OnceLock::new();
    *LIMITS.get_or_init(|| {
        let hard = std::env::var("DECANT_D3D12X_MEMORY_WAIT_MS").ok();
        let idle = std::env::var("DECANT_D3D12X_MEMORY_WAIT_IDLE_MS").ok();
        limits_from(hard.as_deref(), idle.as_deref())
    })
}

/// A write the CPU makes after the GPU: the title address and the dwords stored there in order.
/// An empty dword list is a GPU timestamp: eight bytes that the CPU fills from the GPU clock at the
/// moment it makes the write (`cpu_writes`).
pub type CpuWrite = (usize, Vec<u32>);

/// How many bytes a CPU write of `words` covers. A timestamp covers eight.
pub fn write_bytes(words: &[u32]) -> usize {
    match words.len() {
        0 => 8,
        n => n * 4,
    }
}

/// A memory wait of the title (`Wait32BitValueX` or `Wait64BitValueX`): the `width` bytes at `va`,
/// ANDed with `mask`, must pass `comparison` against `value` before the list may run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingWait {
    pub va: u64,
    pub width: u8,
    pub comparison: u32,
    pub value: u64,
    pub mask: u64,
}

/// `D3D12_COMPARISON_FUNC` numbers of a comparison that never and that always holds.
pub const COMPARISON_NEVER: u32 = 1;
pub const COMPARISON_ALWAYS: u32 = 8;

/// Applies the `D3D12_COMPARISON_FUNC` numbered `op` (1 to 8) to `lhs` and `rhs`. Any other number
/// gives `None`.
pub fn compare(op: u32, lhs: u64, rhs: u64) -> Option<bool> {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let order = lhs.cmp(&rhs);
    Some(match op {
        COMPARISON_NEVER => false,
        2 => order == Less,
        3 => order == Equal,
        4 => order != Greater,
        5 => order == Greater,
        6 => order != Equal,
        7 => order != Less,
        COMPARISON_ALWAYS => true,
        _ => return None,
    })
}

impl PendingWait {
    /// Whether memory that currently reads `current` meets the wait. Every wait is compared as
    /// `current & mask` (the fourth argument of `Wait64BitValueX` is no mask, see `proxy::list_wait64`,
    /// so the caller passes all ones for it).
    pub fn satisfied_by(&self, current: u64) -> bool {
        compare(self.comparison, current & self.mask, self.value).unwrap_or(true)
    }
}

/// Everything recorded on one list since its last reset.
#[derive(Default)]
struct Recording {
    writes: Vec<CpuWrite>,
    waits: Vec<PendingWait>,
    index_view: Option<[u8; 16]>,
}

#[derive(Default)]
struct FastUsizeHasher(u64);

impl Hasher for FastUsizeHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
        }
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<V> = HashMap<usize, V, BuildHasherDefault<FastUsizeHasher>>;

static TABLE: Mutex<Option<FastMap<Recording>>> = Mutex::new(None);

/// Runs `f` on the recording of `list`, made if there is none yet.
fn with_recording<R>(list: usize, f: impl FnOnce(&mut Recording) -> R) -> R {
    let mut guard = lock();
    f(guard.get_or_insert_with(FastMap::default).entry(list).or_default())
}

/// Runs `f` on the recording of `list` if it has one.
fn peek<R>(list: usize, f: impl FnOnce(&Recording) -> R) -> Option<R> {
    lock().as_ref().and_then(|t| t.get(&list)).map(f)
}

fn lock() -> MutexGuard<'static, Option<FastMap<Recording>>> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Records on `list` a write of `words` at `va`, memory the GPU cannot reach.
pub fn record_cpu_write(list: usize, va: usize, words: Vec<u32>) {
    with_recording(list, |r| r.writes.push((va, words)));
}

/// The writes of the given lists, list by list in the order given. A list that is executed twice
/// has its writes made twice, so they stay until the list is reset.
pub fn cpu_writes_of(list_keys: &[usize]) -> Vec<CpuWrite> {
    let guard = lock();
    let Some(table) = guard.as_ref() else { return Vec::new() };
    list_keys.iter().filter_map(|k| table.get(k)).flat_map(|r| r.writes.iter().cloned()).collect()
}

/// Notes the index buffer view (a 16 byte `D3D12_INDEX_BUFFER_VIEW`) bound on `list`, or that none is.
pub fn set_index_view(list: usize, view: Option<[u8; 16]>) {
    let mut guard = lock();
    match view {
        Some(_) => guard.get_or_insert_with(FastMap::default).entry(list).or_default().index_view = view,
        None => {
            if let Some(r) = guard.as_mut().and_then(|t| t.get_mut(&list)) {
                r.index_view = None;
            }
        }
    }
}

/// The index buffer view bound on `list` since its last reset.
pub fn index_view(list: usize) -> Option<[u8; 16]> {
    peek(list, |r| r.index_view).flatten()
}

/// `list` was reset: all that was recorded on it belonged to the old recording, keeping vector
/// capacities for reuse on the next recording.
pub fn forget(list: usize) {
    if let Some(r) = lock().as_mut().and_then(|t| t.get_mut(&list)) {
        r.writes.clear();
        r.waits.clear();
        r.index_view = None;
    }
}

/// Records a memory wait on `list`.
pub fn record_wait(list: usize, wait: PendingWait) {
    with_recording(list, |r| r.waits.push(wait));
}

/// The waits recorded on `list` since its last reset, in order.
pub fn waits_of(list: usize) -> Vec<PendingWait> {
    peek(list, |r| r.waits.clone()).unwrap_or_default()
}

/// Whether every wait recorded on `list` passes `met`. A list without waits passes.
pub fn all_waits(list: usize, met: impl Fn(&PendingWait) -> bool) -> bool {
    peek(list, |r| r.waits.iter().all(&met)).unwrap_or(true)
}

/// Whether `list` holds a CPU write that overlaps `[va, va + bytes)`. Such a write only happens
/// after the list ran, so a wait on that memory in the same list can never be met beforehand and the
/// caller must not record it.
pub fn has_cpu_write_over(list: usize, va: u64, bytes: u64) -> bool {
    let end = va.saturating_add(bytes);
    peek(list, |r| {
        r.writes.iter().any(|(at, words)| {
            let start = *at as u64;
            va < start + write_bytes(words) as u64 && start < end
        })
    })
    .unwrap_or(false)
}

/// A `D3D12_RESOURCE_BARRIER` (32 bytes) for one global UAV barrier. In the public layout `Type`
/// (UAV is 2) is at offset 0, `Flags` (none) at 4 and the `pResource` of the UAV member (null) at 8.
pub fn global_uav_barrier() -> [u8; 32] {
    let mut barrier = [0u8; 32];
    barrier[0] = 2;
    barrier
}

#[cfg(test)]
mod tests {
    use super::*;

    // The table is shared by the whole process and tests run in parallel, so each test has list
    // keys of its own.

    #[test]
    fn barrier_bytes_follow_the_public_layout() {
        let b = global_uav_barrier();
        let (ty, flags, resource) = (&b[..4], &b[4..8], &b[8..16]);
        assert_eq!(ty, 2u32.to_le_bytes());
        assert_eq!(flags, [0; 4]);
        assert_eq!(resource, [0; 8]);
        assert!(b[16..].iter().all(|&x| x == 0));
    }

    #[test]
    fn writes_keep_their_recording_order() {
        record_cpu_write(0x1_0000, 0xA0, vec![1]);
        record_cpu_write(0x1_0000, 0x90, vec![2, 3]);
        assert_eq!(cpu_writes_of(&[0x1_0000]), [(0xA0, vec![1]), (0x90, vec![2, 3])]);
        forget(0x1_0000);
    }

    #[test]
    fn writes_survive_reads_and_end_with_a_reset() {
        record_cpu_write(0x2_0000, 0xB0, vec![5]);
        for _ in 0..3 {
            assert_eq!(cpu_writes_of(&[0x2_0000]).len(), 1);
        }
        forget(0x2_0000);
        assert!(cpu_writes_of(&[0x2_0000]).is_empty());
    }

    #[test]
    fn writes_of_several_lists_follow_the_key_order_asked_for() {
        record_cpu_write(0x3_0000, 1, vec![10]);
        record_cpu_write(0x3_0001, 2, vec![20]);
        assert_eq!(cpu_writes_of(&[0x3_0001, 0x3_0000]), [(2, vec![20]), (1, vec![10])]);
        assert!(cpu_writes_of(&[0x3_0002]).is_empty());
        assert!(cpu_writes_of(&[]).is_empty());
        forget(0x3_0000);
        forget(0x3_0001);
    }

    #[test]
    fn compare_covers_the_eight_functions_and_rejects_the_rest() {
        let table = [(1, [false, false, false]), (2, [true, false, false]), (3, [false, true, false]), (4, [true, true, false]), (5, [false, false, true]), (6, [true, false, true]), (7, [false, true, true]), (8, [true, true, true])];
        for (op, [lt, eq, gt]) in table {
            assert_eq!(compare(op, 1, 2), Some(lt), "op {op} on 1, 2");
            assert_eq!(compare(op, 2, 2), Some(eq), "op {op} on 2, 2");
            assert_eq!(compare(op, 3, 2), Some(gt), "op {op} on 3, 2");
        }
        assert_eq!(compare(0, 1, 1), None);
        assert_eq!(compare(9, 1, 1), None);
    }

    #[test]
    fn a_wait_applies_its_mask_before_comparing() {
        let masked = PendingWait { va: 0, width: 4, comparison: 3, value: 2, mask: 0xF };
        assert!(masked.satisfied_by(0x72), "0x72 & 0xF is 2");
        assert!(!masked.satisfied_by(0x73));
        let whole = PendingWait { mask: u64::MAX, comparison: 7, value: 1, width: 8, va: 0 };
        assert!(!whole.satisfied_by(0));
        assert!(whole.satisfied_by(0x1_0000_0000));
    }

    #[test]
    fn a_wait_with_an_unknown_comparison_counts_as_met() {
        let w = PendingWait { va: 0, width: 8, comparison: 99, value: 5, mask: u64::MAX };
        assert!(w.satisfied_by(0));
    }

    #[test]
    fn waits_are_per_list_and_end_with_a_reset() {
        let wait = |va| PendingWait { va, width: 4, comparison: 5, value: 0, mask: u64::MAX };
        record_wait(0x4_0000, wait(1));
        record_wait(0x4_0000, wait(2));
        assert_eq!(waits_of(0x4_0000), [wait(1), wait(2)]);
        assert!(all_waits(0x4_0000, |w| w.va > 0));
        assert!(!all_waits(0x4_0000, |w| w.va == 1));
        assert!(waits_of(0x4_0001).is_empty());
        assert!(all_waits(0x4_0001, |_| false), "a list without waits passes");
        forget(0x4_0000);
        assert!(waits_of(0x4_0000).is_empty());
    }

    #[test]
    fn wait_limits_take_the_settings_or_the_defaults() {
        assert_eq!(limits_from(None, None), WaitLimits { idle: WAIT_IDLE, hard: WAIT_HARD });
        assert_eq!(limits_from(Some(" 500 "), Some("50")), WaitLimits { idle: Duration::from_millis(50), hard: Duration::from_millis(500) });
        assert_eq!(limits_from(Some("100"), None), WaitLimits { idle: Duration::from_millis(100), hard: Duration::from_millis(100) }, "idle never above hard");
        assert_eq!(limits_from(Some("soon"), Some("-1")), WaitLimits { idle: WAIT_IDLE, hard: WAIT_HARD });
        assert_eq!(limits_from(Some("0"), None).hard, Duration::ZERO, "0 runs every list at once");
    }

    #[test]
    fn a_reset_also_drops_the_index_view() {
        set_index_view(0x5_0000, Some([7; 16]));
        record_wait(0x5_0000, PendingWait { va: 0, width: 4, comparison: 8, value: 0, mask: 0 });
        assert_eq!(index_view(0x5_0000), Some([7; 16]));
        set_index_view(0x5_0000, None);
        assert_eq!(index_view(0x5_0000), None);
        assert_eq!(waits_of(0x5_0000).len(), 1, "unbinding the view leaves the waits alone");
        set_index_view(0x5_0000, Some([1; 16]));
        forget(0x5_0000);
        assert_eq!(index_view(0x5_0000), None);
        assert!(waits_of(0x5_0000).is_empty());
    }

    #[test]
    fn overlap_uses_eight_bytes_for_a_timestamp_and_four_per_dword() {
        assert_eq!(write_bytes(&[]), 8);
        assert_eq!(write_bytes(&[1, 2, 3]), 12);
        record_cpu_write(0x6_0000, 0x1000, Vec::new());
        record_cpu_write(0x6_0000, 0x2000, vec![1, 2]);
        assert!(has_cpu_write_over(0x6_0000, 0x1004, 4));
        assert!(!has_cpu_write_over(0x6_0000, 0x1008, 8));
        assert!(has_cpu_write_over(0x6_0000, 0x1FFC, 8));
        assert!(!has_cpu_write_over(0x6_0000, 0x2008, 4));
        assert!(!has_cpu_write_over(0x6_0001, 0x1000, 8), "another list");
        forget(0x6_0000);
    }

    #[test]
    fn forget_reuses_recording_allocations() {
        let list = 0x7_0000usize;
        record_cpu_write(list, 0x1000, vec![1, 2]);
        record_wait(list, PendingWait { va: 0, width: 4, comparison: 5, value: 0, mask: u64::MAX });
        let caps_before = peek(list, |r| (r.writes.capacity(), r.waits.capacity())).unwrap();
        assert!(caps_before.0 >= 1 && caps_before.1 >= 1);
        forget(list);
        let state_after = peek(list, |r| (r.writes.len(), r.writes.capacity(), r.waits.len(), r.waits.capacity(), r.index_view));
        assert_eq!(state_after, Some((0, caps_before.0, 0, caps_before.1, None)));
        if let Some(table) = lock().as_mut() {
            table.remove(&list);
        }
    }
}
