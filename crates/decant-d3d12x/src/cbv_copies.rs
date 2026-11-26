//! The copies of constant buffers that do not start on 256 bytes, kept per list until it is
//! submitted.
//!
//! A PC constant buffer view must start on 256 bytes, a console one on 16. So when a draw reads a
//! table slot that holds such a view, the layer copies the title's bytes into upload memory of the
//! list and points the host slot at the copy (`list_mirror::copied_cbv`). On the console the GPU
//! reads the constant buffer when the draw runs, and every other buffer of the title reaches the GPU
//! at submission (`title_mem::sync`). A title that fills its constants after recording the draw (a
//! text batch writes its glyph transforms once the whole string is laid out) would read the bytes of
//! the recording time from the copy. So the copies of a list are taken again from the title's memory
//! when the list is first submitted after its reset, where they differ.
//!
//! A list submitted a second time without a reset keeps its copies: the GPU may still read them.
//!
//! The same host slot written twice in one list (two draws whose tables share the slot with
//! different constant buffers) is counted: the GPU reads only the last view written there.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const SHARDS: usize = 64;
const SHARD_MASK: usize = SHARDS - 1;

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
            self.0 = self.0.wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[inline]
fn shard_of(list: usize) -> usize {
    ((list.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u64) >> (64 - 6)) as usize & SHARD_MASK
}

/// One copy: the host slot that points at it, the title's range, the copy's CPU address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CbvCopy {
    pub dst: usize,
    pub va: u64,
    pub size: u32,
    pub cpu: usize,
}

/// The copies of one list since its reset.
#[derive(Default)]
struct ListCopies {
    copies: Vec<(CbvCopy, u64)>,
    destinations: FastMap<usize, Destination>,
    /// The number of this recording of the list (`Ledger::recording`), 0 when nobody asked.
    recording: u64,
    submitted: bool,
    /// Tables moved for the last draw, per bind (0 graphics, 1 compute): (the title's host handle,
    /// the moved one), to point back before the next draw.
    moved: [Vec<(u64, u64)>; 2],
}

/// If two different ranges have used a slot, every future write conflicts with at least one.
struct Destination {
    first: (u64, u32),
    mixed: bool,
}

impl Destination {
    fn conflicts(&self, range: (u64, u32)) -> bool {
        self.mixed || self.first != range
    }
}

/// What a refresh at submission found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Refresh {
    /// Copies of lists submitted for the first time since their reset.
    pub copies: usize,
    /// Of them, those whose title bytes changed since the recording and were taken again.
    pub stale: usize,
    /// Copies of lists submitted again without a reset, left as they are.
    pub kept: usize,
}

/// The copies of every list, apart from any memory access (tested on synthetic data).
#[cfg(test)]
#[derive(Default)]
pub(crate) struct Ledger {
    lists: FastMap<usize, ListCopies>,
    recordings: u64,
}

#[cfg(test)]
impl Ledger {
    /// Notes a copy recorded in `list`; true when its host slot already pointed at another copy of
    /// a different title range in this list (the GPU reads only the last).
    pub fn recorded(&mut self, list: usize, copy: CbvCopy, hash: u64) -> bool {
        let l = self.lists.entry(list).or_default();
        let range = (copy.va, copy.size);
        let dst = l.destinations.entry(copy.dst).or_insert(Destination { first: range, mixed: false });
        dst.mixed = dst.conflicts(range);
        l.copies.push((copy, hash));
        dst.mixed
    }

    /// Whether host slot `dst` already points at a copy of another title range in `list`: a copy
    /// of `va` written there would change what an earlier draw of the list reads.
    pub fn conflicts(&self, list: usize, dst: usize, va: u64, size: u32) -> bool {
        self.lists.get(&list).and_then(|l| l.destinations.get(&dst)).is_some_and(|d| d.conflicts((va, size)))
    }

    /// The number of the recording `list` is in: another one after each reset, never 0.
    pub fn recording(&mut self, list: usize) -> u64 {
        let l = self.lists.entry(list).or_default();
        if l.recording == 0 {
            self.recordings += 1;
            l.recording = self.recordings;
        }
        l.recording
    }

    /// Whether `list` was not reset since [`recording`](Self::recording) gave `recording`, so its copies are there.
    pub fn is_live(&self, list: usize, recording: u64) -> bool {
        recording != 0 && self.lists.get(&list).is_some_and(|l| l.recording == recording)
    }

    /// The table at host handle `from` of bind `bind` (0 graphics, 1 compute) was moved to `to` for
    /// the next draw of `list`.
    pub fn moved(&mut self, list: usize, bind: usize, from: u64, to: u64) {
        self.lists.entry(list).or_default().moved[bind].push((from, to));
    }

    /// The tables of `bind` moved for the last draw of `list`, taken out.
    pub fn take_moved(&mut self, list: usize, bind: usize) -> Vec<(u64, u64)> {
        self.lists.get_mut(&list).map(|l| std::mem::take(&mut l.moved[bind])).unwrap_or_default()
    }

    /// `list` was reset: its copies are no longer read.
    pub fn forget(&mut self, list: usize) {
        self.lists.remove(&list);
    }

    /// `lists` are submitted: the copies of a list submitted for the first time since its reset are
    /// refreshed by `refresh(copy, hash)`, which returns true when the copy was taken again.
    pub fn submitted(&mut self, lists: &[usize], mut refresh: impl FnMut(&CbvCopy, u64) -> bool) -> Refresh {
        let mut r = Refresh::default();
        for list in lists {
            let Some(l) = self.lists.get_mut(list) else { continue };
            if l.submitted {
                r.kept += l.copies.len();
                continue;
            }
            l.submitted = true;
            for (c, hash) in &l.copies {
                r.copies += 1;
                if refresh(c, *hash) {
                    r.stale += 1;
                }
            }
        }
        r
    }
}

static LEDGER_SHARDS: [Mutex<Option<FastMap<usize, ListCopies>>>; SHARDS] = [const { Mutex::new(None) }; SHARDS];
static RECORDINGS: AtomicU64 = AtomicU64::new(1);

#[inline]
fn with_shard<R>(list: usize, f: impl FnOnce(&mut FastMap<usize, ListCopies>) -> R) -> R {
    let mut guard = LEDGER_SHARDS[shard_of(list)].lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Default::default))
}

/// Copy `copy` was recorded in list `list`; true when its host slot was already written by another
/// copy of this list.
pub(crate) fn recorded(list: usize, copy: CbvCopy) -> bool {
    // SAFETY: the caller just copied this readable title range while recording the draw.
    let title = unsafe { std::slice::from_raw_parts(copy.va as *const u8, copy.size as usize) };
    let hash = fingerprint(title);
    with_shard(list, |lists| {
        let l = lists.entry(list).or_default();
        let range = (copy.va, copy.size);
        let dst = l.destinations.entry(copy.dst).or_insert(Destination { first: range, mixed: false });
        dst.mixed = dst.conflicts(range);
        l.copies.push((copy, hash));
        dst.mixed
    })
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = (bytes.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x517C_C1B7_2722_0A95;
    let (chunks, rem) = bytes.as_chunks::<8>();
    for chunk in chunks {
        let v = u64::from_le_bytes(*chunk);
        h = (h ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27);
    }
    if !rem.is_empty() {
        let mut tail = 0u64;
        for (i, &b) in rem.iter().enumerate() {
            tail |= u64::from(b) << (i * 8);
        }
        h = (h ^ tail).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(27);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h
}

/// Whether a copy of `va` into host slot `dst` would change what an earlier draw of `list` reads.
pub(crate) fn conflicts(list: usize, dst: usize, va: u64, size: u32) -> bool {
    with_shard(list, |lists| lists.get(&list).and_then(|l| l.destinations.get(&dst)).is_some_and(|d| d.conflicts((va, size))))
}

/// The number of the recording `list` is in.
pub(crate) fn recording(list: usize) -> u64 {
    with_shard(list, |lists| {
        let l = lists.entry(list).or_default();
        if l.recording == 0 {
            l.recording = RECORDINGS.fetch_add(1, Ordering::Relaxed);
        }
        l.recording
    })
}

/// Whether `list` is still in recording `recording` or that one was not reset yet.
pub(crate) fn is_live(list: usize, recording: u64) -> bool {
    if recording == 0 {
        return false;
    }
    with_shard(list, |lists| lists.get(&list).is_some_and(|l| l.recording == recording))
}

/// The table at `from` of `bind` was moved to `to` for the next draw of `list`.
pub(crate) fn moved(list: usize, bind: usize, from: u64, to: u64) {
    with_shard(list, |lists| lists.entry(list).or_default().moved[bind].push((from, to)));
}

/// The tables of `bind` moved for the last draw of `list`.
pub(crate) fn take_moved(list: usize, bind: usize) -> Vec<(u64, u64)> {
    with_shard(list, |lists| lists.get_mut(&list).map(|l| std::mem::take(&mut l.moved[bind])).unwrap_or_default())
}

/// List `list` was reset.
pub(crate) fn forget(list: usize) {
    with_shard(list, |lists| {
        lists.remove(&list);
    });
}

/// The lists `lists` (console list proxies) are about to be submitted: their copies are brought up
/// to date with the title's bytes.
pub(crate) fn submitted(lists: &[usize]) -> Refresh {
    let mut r = Refresh::default();
    for &list in lists {
        with_shard(list, |shard| {
            let Some(l) = shard.get_mut(&list) else { return };
            if l.submitted {
                r.kept += l.copies.len();
                return;
            }
            l.submitted = true;
            for (c, hash) in &l.copies {
                r.copies += 1;
                if refresh_copy(c, *hash) {
                    r.stale += 1;
                }
            }
        });
    }
    r
}

fn refresh_copy(copy: &CbvCopy, recorded_hash: u64) -> bool {
    // SAFETY: the title keeps this range readable while its lists use it.
    let title = unsafe { std::slice::from_raw_parts(copy.va as *const u8, copy.size as usize) };
    if fingerprint(title) == recorded_hash {
        return false;
    }
    // SAFETY: the list owns a separate mapped upload block of at least `size` bytes until reset.
    unsafe { std::ptr::copy_nonoverlapping(title.as_ptr(), copy.cpu as *mut u8, title.len()) };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(dst: usize, va: u64) -> CbvCopy {
        CbvCopy { dst, va, size: 64, cpu: 0x9000 + dst }
    }

    #[test]
    fn a_slot_written_twice_in_one_list_with_another_range_is_counted() {
        let mut l = Ledger::default();
        assert!(!l.recorded(1, copy(10, 0x100), 0));
        assert!(!l.recorded(1, copy(10, 0x100), 0), "the same range again reads the same bytes");
        assert!(!l.recorded(1, copy(11, 0x200), 0));
        assert!(l.recorded(1, copy(10, 0x300), 0), "slot 10 now points at another range");
        assert!(!l.recorded(2, copy(10, 0x400), 0), "another list");
        l.forget(1);
        assert!(!l.recorded(1, copy(10, 0x500), 0), "a reset list starts over");
    }

    #[test]
    fn a_conflicting_copy_is_found_before_it_is_written_and_moves_are_taken_once() {
        let mut l = Ledger::default();
        l.recorded(1, copy(10, 0x100), 0);
        assert!(!l.conflicts(1, 10, 0x100, 64));
        assert!(l.conflicts(1, 10, 0x300, 64));
        assert!(!l.conflicts(2, 10, 0x300, 64));
        l.moved(1, 0, 0x40, 0x900);
        assert_eq!(l.take_moved(1, 1), []);
        assert_eq!(l.take_moved(1, 0), [(0x40, 0x900)]);
        assert_eq!(l.take_moved(1, 0), []);
    }

    #[test]
    fn a_recording_is_live_until_its_list_is_reset() {
        let mut l = Ledger::default();
        let (a, b) = (l.recording(1), l.recording(2));
        assert!(a != 0 && b != 0 && a != b);
        assert_eq!(l.recording(1), a);
        l.recorded(1, copy(10, 0x100), 0);
        assert!(l.is_live(1, a) && !l.is_live(2, a) && !l.is_live(1, 0));
        l.forget(1);
        assert!(!l.is_live(1, a));
        assert_ne!(l.recording(1), a);
        assert!(!l.is_live(1, a) && l.is_live(2, b));
    }

    #[test]
    fn destination_index_preserves_conflicts_and_recording_order() {
        let mut ledger = Ledger::default();
        let mut history: Vec<CbvCopy> = Vec::new();
        let short = CbvCopy { size: 16, ..copy(10, 0x100) };
        for next in [copy(10, 0x100), copy(11, 0x200), copy(10, 0x100), short, copy(10, 0x100), short] {
            let expected = history.iter().any(|c| c.dst == next.dst && (c.va, c.size) != (next.va, next.size));
            assert_eq!(ledger.conflicts(1, next.dst, next.va, next.size), expected);
            assert_eq!(ledger.recorded(1, next, 0), expected);
            history.push(next);
        }
        let mut submitted = Vec::new();
        ledger.submitted(&[1], |c, _| {
            submitted.push(*c);
            false
        });
        assert_eq!(submitted, history);
    }

    #[test]
    fn submission_matches_byte_comparison_with_the_recorded_title_data() {
        let original = [7u8; 64];
        let mut title = original;
        let mut upload = original;
        let hash = fingerprint(&original);
        for index in 0..=title.len() {
            title = original;
            upload.copy_from_slice(&original);
            if let Some(byte) = title.get_mut(index) {
                *byte = 8;
            }
            let stale = title != upload;
            let c = CbvCopy { dst: 10, va: title.as_ptr() as u64, size: 64, cpu: upload.as_mut_ptr() as usize };
            assert_eq!(refresh_copy(&c, hash), stale);
            assert_eq!(upload, title);
        }
    }

    #[test]
    fn the_first_submission_takes_changed_copies_again_and_a_second_one_keeps_them() {
        let mut l = Ledger::default();
        l.recorded(1, copy(10, 0x100), 0);
        l.recorded(1, copy(11, 0x200), 0);
        l.recorded(2, copy(12, 0x300), 0);
        let mut retaken = Vec::new();
        let r = l.submitted(&[1, 3], |c, _| {
            if c.va == 0x200 {
                retaken.push(c.va);
                true
            } else {
                false
            }
        });
        assert_eq!(r, Refresh { copies: 2, stale: 1, kept: 0 });
        assert_eq!(retaken, [0x200]);
        let r = l.submitted(&[1, 2], |c, _| {
            assert_eq!(c.va, 0x300, "list 1 may still run");
            true
        });
        assert_eq!(r, Refresh { copies: 1, stale: 1, kept: 2 });
        l.forget(1);
        l.recorded(1, copy(10, 0x100), 0);
        assert_eq!(l.submitted(&[1], |_, _| false), Refresh { copies: 1, stale: 0, kept: 0 });
    }

    #[test]
    fn fingerprint_detects_single_byte_changes_across_unaligned_lengths() {
        for len in [1usize, 7, 8, 9, 15, 16, 31, 64, 127] {
            let base = vec![0xA5u8; len];
            let base_hash = fingerprint(&base);
            for pos in 0..len {
                let mut mutated = base.clone();
                mutated[pos] ^= 0x01;
                assert_ne!(fingerprint(&mutated), base_hash, "len {len} pos {pos}");
            }
        }
    }

    #[test]
    fn global_sharded_ledger_isolates_concurrent_lists() {
        std::thread::scope(|scope| {
            for t in 0..4usize {
                scope.spawn(move || {
                    let list = 0x5000_0000 + t * 32;
                    let rec = recording(list);
                    assert!(is_live(list, rec));
                    let mut title = [1u8; 32];
                    let mut upload = [1u8; 32];
                    let c = CbvCopy { dst: 0x100 + t, va: title.as_ptr() as u64, size: 32, cpu: upload.as_mut_ptr() as usize };
                    assert!(!recorded(list, c));
                    assert!(!conflicts(list, 0x100 + t, title.as_ptr() as u64, 32));
                    assert!(conflicts(list, 0x100 + t, 0xDEAD, 32));
                    moved(list, 0, 0x10, 0x20);
                    assert_eq!(take_moved(list, 0), [(0x10, 0x20)]);
                    title[17] = 9;
                    let r = submitted(&[list]);
                    assert_eq!(r, Refresh { copies: 1, stale: 1, kept: 0 });
                    assert_eq!(upload, title);
                    forget(list);
                    assert!(!is_live(list, rec));
                });
            }
        });
    }
}
