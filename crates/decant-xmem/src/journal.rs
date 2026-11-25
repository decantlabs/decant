//! A ring of the changes the engine made to its address space, for a reader that mirrors it (the
//! GPU side of `d3d12_xs.dll`). Every change has a sequence number. A reader that asks for the
//! changes after a number it saw gets them in order, or learns that the ring has overrun and
//! starts again from a snapshot. Plain safe code.

/// What a change did.
pub const RESERVE: u32 = 1;
pub const RELEASE: u32 = 2;
pub const MAP: u32 = 3;
pub const UNMAP: u32 = 4;

/// `flags` of a `RESERVE`: the reservation is a commit of the title (`XMemVirtualAlloc` with a console
/// page bit) whose pages reach the mirror only once asked for (`DecantXmemGpuUse`). Its `MAP` records come
/// for those pages only, as they are asked for.
pub const ON_DEMAND: u32 = 1;

/// The records the ring holds.
pub const RING: usize = 65536;

/// One change. `va` is the title address (a reservation base for `RESERVE` and `RELEASE`, the first
/// page for `MAP` and `UNMAP`), `pages` the 64 KiB pages it covers and `first_page` the first pool
/// page index of a `MAP` (the pages are consecutive). `flags` holds `ON_DEMAND` for a `RESERVE`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuChange {
    pub seq: u64,
    pub kind: u32,
    pub pages: u32,
    pub va: u64,
    pub first_page: u32,
    pub flags: u32,
}

#[derive(Debug, Default)]
pub struct Journal {
    ring: Vec<GpuChange>,
    /// The sequence number of the last change, 0 before the first.
    seq: u64,
}

impl Journal {
    pub fn new() -> Journal {
        Journal::default()
    }

    /// The sequence number of the last change.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Records a change and returns its sequence number.
    pub fn push(&mut self, kind: u32, va: usize, pages: usize, first_page: u32, flags: u32) -> u64 {
        self.seq += 1;
        let record = GpuChange { seq: self.seq, kind, pages: pages as u32, va: va as u64, first_page, flags };
        if self.ring.len() < RING {
            self.ring.push(record);
        } else {
            let slot = (self.seq as usize - 1) % RING;
            self.ring[slot] = record;
        }
        self.seq
    }

    /// The changes after `after`, in order, at most `cap` of them. `None` when the ring no longer
    /// holds the change that follows `after` (the reader must start from a snapshot).
    pub fn since(&self, after: u64, cap: usize) -> Option<Vec<GpuChange>> {
        if after > self.seq || self.seq - after > RING as u64 {
            return None;
        }
        let last = (after + cap as u64).min(self.seq);
        Some((after + 1..=last).map(|n| self.ring[(n as usize - 1) % RING]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_come_back_in_order_after_a_sequence_number() {
        let mut j = Journal::new();
        assert_eq!(j.since(0, 10), Some(vec![]));
        j.push(RESERVE, 0x1_0000, 4, 0, 0);
        j.push(MAP, 0x1_0000, 2, 7, 0);
        j.push(UNMAP, 0x2_0000, 1, 0, 0);
        let all = j.since(0, 10).unwrap();
        assert_eq!(all.iter().map(|c| (c.seq, c.kind)).collect::<Vec<_>>(), [(1, RESERVE), (2, MAP), (3, UNMAP)]);
        assert_eq!(all[1], GpuChange { seq: 2, kind: MAP, pages: 2, va: 0x1_0000, first_page: 7, flags: 0 });
        assert_eq!(j.since(2, 10).unwrap().len(), 1);
        assert_eq!(j.since(1, 1).unwrap()[0].seq, 2, "the cap cuts the answer");
        assert_eq!(j.since(3, 10), Some(vec![]));
        assert_eq!(j.since(4, 10), None, "a sequence number from the future is not ours");
    }

    #[test]
    fn an_overrun_is_detected() {
        let mut j = Journal::new();
        for i in 0..RING as u64 + 5 {
            j.push(MAP, 0x1_0000, 1, i as u32, 0);
        }
        assert_eq!(j.since(0, 10), None, "the first records are gone");
        assert_eq!(j.since(4, 10), None);
        let from = j.since(5, 3).unwrap();
        assert_eq!(from.iter().map(|c| c.seq).collect::<Vec<_>>(), [6, 7, 8]);
        assert_eq!(j.since(j.seq() - 1, 10).unwrap()[0].seq, j.seq());
    }

    #[test]
    fn a_snapshot_then_the_changes_after_it_equal_the_history() {
        // The state is a map of va to first page, rebuilt by replaying records.
        fn replay(state: &mut std::collections::BTreeMap<u64, u32>, c: &GpuChange) {
            match c.kind {
                MAP => {
                    state.insert(c.va, c.first_page);
                }
                UNMAP => {
                    state.remove(&c.va);
                }
                _ => {}
            }
        }
        let mut j = Journal::new();
        let mut full = std::collections::BTreeMap::new();
        for i in 0..100u32 {
            j.push(if i % 3 == 0 { UNMAP } else { MAP }, 0x1_0000 * (i as usize % 7 + 1), 1, i, 0);
        }
        let mut at_snapshot = std::collections::BTreeMap::new();
        for c in j.since(0, 1000).unwrap() {
            replay(&mut at_snapshot, &c);
        }
        let seq = j.seq();
        for i in 100..140u32 {
            j.push(MAP, 0x1_0000 * (i as usize % 5 + 1), 1, i, 0);
        }
        for c in j.since(0, 1000).unwrap() {
            replay(&mut full, &c);
        }
        for c in j.since(seq, 1000).unwrap() {
            replay(&mut at_snapshot, &c);
        }
        assert_eq!(at_snapshot, full);
    }
}
