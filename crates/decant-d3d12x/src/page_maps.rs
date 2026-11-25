//! The page mapping copies of a console command queue. A console title registers a pool of 64 KiB
//! pages with the device and then tells its queue which pool pages its GPU shall see at which
//! addresses, in batches: each batch names a destination address and a list of ranges that follow
//! each other from there. A range is one of four kinds: a run of consecutive pool pages, one pool
//! page shown over the whole range, nothing (the pages are unmapped), or "leave as it is".
//!
//! The title gets every streamed texture to its GPU this way, and the planes of its videos: the decoder
//! writes pool pages at the pool's address, the textures are placed at addresses in a reservation
//! where the title itself maps nothing, and this call makes the two meet.
//!
//! On the PC the title's memory and the GPU's view of it are one thing (`title_mem`), so the layer
//! asks `xmem.dll` to show the pool pages at the destination too (`DecantXmemShowPages`). That is a
//! second view of the same pages for the CPU and, through the journal of `xmem.dll`, for the GPU
//! window. This module turns the ranges into those requests. It has no D3D12 in it.

/// The page of a pool and of a mapping: 64 KiB.
pub const PAGE: u64 = 0x1_0000;

/// A run of consecutive pool pages from `first`.
pub const KIND_RUN: u32 = 0;
/// Pool page `first` at every page of the range.
pub const KIND_REPEAT: u32 = 1;
/// The pages of the range show nothing.
pub const KIND_UNMAP: u32 = 2;
/// The pages of the range stay as they are.
pub const KIND_KEEP: u32 = 3;

/// One range of a batch as the title passes it: three 32 bit words.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub kind: u32,
    /// The first pool page (a page index of the pool).
    pub first: u32,
    /// How many pages the range covers at the destination.
    pub pages: u32,
}

/// One batch as the title passes it.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Batch {
    pub destination: u64,
    pub count: u32,
    pub ranges: *const Range,
}

/// One request to `xmem.dll`: `pages` pages from `to` show what the pages from `from` show, or
/// nothing when `from` is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Show {
    pub to: u64,
    pub from: u64,
    pub pages: u32,
}

/// The requests for the `ranges` of one batch whose first page goes to `destination`, for a pool of
/// `pool_pages` pages at `pool`, and how many ranges were refused (a kind that is not known, or
/// pool pages past the end of the pool). A refused range still takes its room at the destination.
pub fn requests(destination: u64, ranges: &[Range], pool: u64, pool_pages: u32) -> (Vec<Show>, u32) {
    let mut out = Vec::with_capacity(ranges.len());
    let mut refused = 0;
    let mut to = destination;
    for r in ranges {
        let inside = |pages: u32| r.first.checked_add(pages).is_some_and(|end| end <= pool_pages);
        let from = pool + u64::from(r.first) * PAGE;
        match r.kind {
            KIND_RUN if inside(r.pages) => {
                if r.pages > 0 {
                    out.push(Show { to, from, pages: r.pages });
                }
            }
            KIND_REPEAT if inside(1) => out.extend((0..u64::from(r.pages)).map(|k| Show { to: to + k * PAGE, from, pages: 1 })),
            KIND_UNMAP => {
                if r.pages > 0 {
                    out.push(Show { to, from: 0, pages: r.pages });
                }
            }
            KIND_KEEP => {}
            _ => refused += 1,
        }
        to += u64::from(r.pages) * PAGE;
    }
    (out, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    const POOL: u64 = 0x17E_3E8A_0000;

    #[test]
    fn the_ranges_of_a_batch_follow_each_other_at_the_destination() {
        let ranges = [
            Range { kind: KIND_RUN, first: 10, pages: 3 },
            Range { kind: KIND_KEEP, first: 0, pages: 2 },
            Range { kind: KIND_UNMAP, first: 0, pages: 4 },
            Range { kind: KIND_REPEAT, first: 7, pages: 2 },
        ];
        let (shows, refused) = requests(0x1000_0000, &ranges, POOL, 100);
        assert_eq!(refused, 0);
        assert_eq!(
            shows,
            vec![
                Show { to: 0x1000_0000, from: POOL + 10 * PAGE, pages: 3 },
                Show { to: 0x1000_0000 + 5 * PAGE, from: 0, pages: 4 },
                Show { to: 0x1000_0000 + 9 * PAGE, from: POOL + 7 * PAGE, pages: 1 },
                Show { to: 0x1000_0000 + 10 * PAGE, from: POOL + 7 * PAGE, pages: 1 },
            ]
        );
    }

    #[test]
    fn pool_pages_past_the_pool_and_unknown_kinds_are_refused_and_keep_their_room() {
        let ranges = [
            Range { kind: KIND_RUN, first: 98, pages: 3 },
            Range { kind: 9, first: 0, pages: 1 },
            Range { kind: KIND_REPEAT, first: 100, pages: 2 },
            Range { kind: KIND_RUN, first: 99, pages: 1 },
        ];
        let (shows, refused) = requests(0, &ranges, POOL, 100);
        assert_eq!(refused, 3);
        assert_eq!(shows, vec![Show { to: 6 * PAGE, from: POOL + 99 * PAGE, pages: 1 }]);
    }

    #[test]
    fn empty_ranges_ask_for_nothing() {
        let ranges = [Range { kind: KIND_RUN, first: 1, pages: 0 }, Range { kind: KIND_UNMAP, first: 0, pages: 0 }, Range { kind: KIND_REPEAT, first: 1, pages: 0 }];
        assert_eq!(requests(0x2_0000, &ranges, POOL, 4), (Vec::new(), 0));
    }

    #[test]
    fn a_batch_has_the_layout_the_title_passes() {
        assert_eq!(std::mem::size_of::<Range>(), 12);
        assert_eq!(std::mem::size_of::<Batch>(), 24);
        assert_eq!(std::mem::offset_of!(Batch, count), 8);
        assert_eq!(std::mem::offset_of!(Batch, ranges), 16);
    }
}
