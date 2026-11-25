//! The physical page model of the console, as plain safe code. A physical page is 64 KiB. Its
//! frame number counts in 4 KiB units: page `i` of the backing section is frame `(i + 1) * 16`,
//! and the frame 0 means "no page". A frame is always a non zero multiple of 16.

use std::collections::BTreeSet;

/// Bytes of one physical page.
pub const PAGE_BYTES: usize = 0x1_0000;

/// 4 KiB units in one physical page: the step between two consecutive frame numbers.
pub const FRAMES_PER_PAGE: u32 = 16;

/// The frame number that means "no page".
pub const NO_FRAME: u32 = 0;

/// The frame number of page `index` of the backing section, `None` when it does not fit 32 bits.
pub fn frame_of(index: usize) -> Option<u32> {
    let frame = (u64::try_from(index).ok()?.checked_add(1)?).checked_mul(u64::from(FRAMES_PER_PAGE))?;
    u32::try_from(frame).ok()
}

/// The page index a frame number names, `None` for 0 and for a number that is not a multiple of 16.
pub fn index_of(frame: u32) -> Option<usize> {
    (frame != NO_FRAME && frame.is_multiple_of(FRAMES_PER_PAGE)).then(|| (frame / FRAMES_PER_PAGE - 1) as usize)
}

/// Splits a list of frames into runs of consecutive pages: `(first frame, page count)`.
pub fn runs(frames: &[u32]) -> Vec<(u32, usize)> {
    let mut out: Vec<(u32, usize)> = Vec::new();
    for &frame in frames {
        match out.last_mut() {
            Some((first, count)) if first.checked_add(FRAMES_PER_PAGE * *count as u32) == Some(frame) => *count += 1,
            _ => out.push((frame, 1)),
        }
    }
    out
}

/// The pool grows in steps of this many pages (256 MiB), so that many small requests do not each
/// create a section.
pub const GROW_STEP_PAGES: usize = 4096;

/// Commit charge that growth leaves free for the rest of the system: 1 GiB.
pub const HEADROOM_BYTES: u64 = 1 << 30;

/// The most pages a frame number can name: frame `(i + 1) * 16` has to fit 32 bits.
pub const MAX_POOL_PAGES: usize = 0x0FFF_FFFE;

/// What the pool does for a request of `count` pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    /// Enough pages are free already.
    None,
    /// Add this many pages (a section of that size).
    Add(usize),
    /// The request cannot be met: the pool may grow by at most `room` pages more, but `missing`
    /// are needed.
    Bound { room: usize, missing: usize },
}

/// How the pool of `pool` pages, of which `free` are free, grows for a request of `count` pages,
/// when the system has `available_commit` bytes of commit charge left. The new pages are charged
/// against the page file, so the bound is that charge minus `HEADROOM_BYTES`, and the number of
/// frames that fit 32 bits. A request that does not fit is refused whole, with no growth. A growth
/// that fits is rounded up to `GROW_STEP_PAGES` as far as the room allows.
pub fn growth(pool: usize, free: usize, count: usize, available_commit: u64) -> Growth {
    let missing = count.saturating_sub(free);
    if missing == 0 {
        return Growth::None;
    }
    let by_commit = usize::try_from(available_commit.saturating_sub(HEADROOM_BYTES) / PAGE_BYTES as u64).unwrap_or(usize::MAX);
    let room = by_commit.min(MAX_POOL_PAGES.saturating_sub(pool));
    if missing > room {
        return Growth::Bound { room, missing };
    }
    Growth::Add(missing.div_ceil(GROW_STEP_PAGES).saturating_mul(GROW_STEP_PAGES).min(room))
}

/// Why a page operation was refused. Nothing changes when one is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageError {
    /// The frame is 0, not a multiple of 16, or past the end of the backing section.
    BadFrame(u32),
    /// The frame is not allocated.
    NotAllocated(u32),
    /// The same frame appears twice in a list to free.
    Duplicate(u32),
}

/// Which pages of the backing sections are in use, and how many views map each of them.
#[derive(Debug)]
pub struct PageAllocator {
    /// The page is not free: the title holds it, or a view still maps it after the title freed it.
    used: Vec<bool>,
    /// The title freed the page while views still mapped it. It goes back to the pool when the
    /// last view ends.
    freed: Vec<bool>,
    views: Vec<u32>,
    /// The page went back to the pool after use, so it may hold old bytes.
    dirty: Vec<bool>,
}

impl PageAllocator {
    /// An allocator over `capacity` pages, all free.
    pub fn new(capacity: usize) -> PageAllocator {
        PageAllocator { used: vec![false; capacity], freed: vec![false; capacity], views: vec![0; capacity], dirty: vec![false; capacity] }
    }

    pub fn capacity(&self) -> usize {
        self.used.len()
    }

    /// Adds `extra` free pages at the end (one more section of the engine). A run may cross from one
    /// section into the next: the engine maps it as one view per section, and the GPU mirror sees
    /// pool pages, not sections.
    pub fn grow(&mut self, extra: usize) {
        if extra == 0 {
            return;
        }
        let total = self.used.len() + extra;
        self.used.resize(total, false);
        self.freed.resize(total, false);
        self.views.resize(total, 0);
        self.dirty.resize(total, false);
    }

    /// Whether `count` consecutive pages are free.
    pub fn has_run(&self, count: usize) -> bool {
        count != 0 && self.find_run(count).is_some()
    }

    /// Whether the page went back to the pool after use (it may hold old bytes). False for a frame that is not valid.
    pub fn is_dirty(&self, frame: u32) -> bool {
        self.slot(frame).is_ok_and(|i| self.dirty[i])
    }

    /// Notes that the page holds nothing old any more, because the caller cleared it.
    pub fn mark_clean(&mut self, frame: u32) {
        if let Ok(i) = self.slot(frame) {
            self.dirty[i] = false;
        }
    }

    fn release_page(&mut self, i: usize) {
        self.used[i] = false;
        self.dirty[i] = true;
    }

    pub fn used_count(&self) -> usize {
        self.used.iter().filter(|&&u| u).count()
    }

    /// Pages that at least one view maps.
    pub fn mapped_count(&self) -> usize {
        self.views.iter().filter(|&&v| v > 0).count()
    }

    fn slot(&self, frame: u32) -> Result<usize, PageError> {
        index_of(frame).filter(|&i| i < self.used.len()).ok_or(PageError::BadFrame(frame))
    }

    /// Whether the title holds the frame: allocated and not freed since.
    pub fn is_allocated(&self, frame: u32) -> bool {
        self.slot(frame).is_ok_and(|i| self.used[i] && !self.freed[i])
    }

    /// The first free run of `count` consecutive pages, as a start index.
    fn find_run(&self, count: usize) -> Option<usize> {
        let mut start = 0;
        let mut length = 0;
        for (i, &used) in self.used.iter().enumerate() {
            if used {
                length = 0;
                start = i + 1;
            } else {
                length += 1;
                if length == count {
                    return Some(start);
                }
            }
        }
        None
    }

    /// `count` consecutive pages, as the frame of the first one. `None` when `count` is 0 or no such
    /// run is free.
    pub fn allocate_run(&mut self, count: usize) -> Option<u32> {
        if count == 0 {
            return None;
        }
        let start = self.find_run(count)?;
        self.used[start..start + count].fill(true);
        frame_of(start)
    }

    /// `count` pages, consecutive when such a run is free and scattered otherwise. All or nothing.
    pub fn allocate(&mut self, count: usize) -> Option<Vec<u32>> {
        if count == 0 {
            return None;
        }
        if let Some(first) = self.allocate_run(count) {
            return Some((0..count as u32).map(|k| first + k * FRAMES_PER_PAGE).collect());
        }
        if self.used.len() - self.used_count() < count {
            return None;
        }
        let picked: Vec<usize> = (0..self.used.len()).filter(|&i| !self.used[i]).take(count).collect();
        let mut frames = Vec::with_capacity(count);
        for &i in &picked {
            self.used[i] = true;
            frames.push(frame_of(i)?);
        }
        Some(frames)
    }

    /// Frees pages. All or nothing: one frame that is bad, not allocated or listed twice makes the
    /// call fail and changes nothing. A page that a view still maps cannot be handed out again
    /// and stays readable through the view until the last view ends.
    pub fn free(&mut self, frames: &[u32]) -> Result<(), PageError> {
        let mut seen = BTreeSet::new();
        for &frame in frames {
            let i = self.slot(frame)?;
            if !self.used[i] || self.freed[i] {
                return Err(PageError::NotAllocated(frame));
            }
            if !seen.insert(i) {
                return Err(PageError::Duplicate(frame));
            }
        }
        for i in seen {
            if self.views[i] == 0 {
                self.release_page(i);
            } else {
                self.freed[i] = true;
            }
        }
        Ok(())
    }

    /// Counts one more view on each page of the run. The pages must be allocated.
    pub fn map(&mut self, first: u32, count: usize) -> Result<(), PageError> {
        let slots = self.run_slots(first, count)?;
        for i in slots {
            self.views[i] += 1;
        }
        Ok(())
    }

    /// Counts one view less on each page of the run.
    pub fn unmap(&mut self, first: u32, count: usize) -> Result<(), PageError> {
        let slots = self.run_slots(first, count)?;
        for i in slots {
            self.views[i] = self.views[i].saturating_sub(1);
            if self.views[i] == 0 && self.freed[i] {
                self.freed[i] = false;
                self.release_page(i);
            }
        }
        Ok(())
    }

    /// Whether the page is out of the pool, held by the title or by views of a freed page.
    pub fn is_in_use(&self, frame: u32) -> bool {
        self.slot(frame).is_ok_and(|i| self.used[i])
    }

    /// Views that map the page, 0 for a frame that is not valid.
    pub fn view_count(&self, frame: u32) -> u32 {
        self.slot(frame).map_or(0, |i| self.views[i])
    }

    fn run_slots(&self, first: u32, count: usize) -> Result<std::ops::Range<usize>, PageError> {
        let start = self.slot(first)?;
        let end = start.checked_add(count).filter(|&e| e <= self.used.len()).ok_or(PageError::BadFrame(first))?;
        if let Some(free) = (start..end).find(|&i| !self.used[i]) {
            return Err(PageError::NotAllocated(frame_of(free).unwrap_or(NO_FRAME)));
        }
        Ok(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_i_is_frame_i_plus_one_times_sixteen() {
        assert_eq!(frame_of(0), Some(16));
        assert_eq!(frame_of(1), Some(32));
        assert_eq!(frame_of(9), Some(160));
        assert_eq!(frame_of(usize::MAX), None);
        assert_eq!(frame_of(0x0FFF_FFFE), Some(0xFFFF_FFF0));
        assert_eq!(frame_of(0x0FFF_FFFF), None);
    }

    #[test]
    fn zero_means_no_page_and_the_inverse_checks_the_multiple() {
        assert_eq!(index_of(NO_FRAME), None);
        assert_eq!(index_of(16), Some(0));
        assert_eq!(index_of(160), Some(9));
        for odd in [1, 8, 15, 17, 31] {
            assert_eq!(index_of(odd), None, "{odd}");
        }
        for i in [0usize, 1, 7, 4095] {
            assert_eq!(frame_of(i).and_then(index_of), Some(i));
        }
    }

    #[test]
    fn runs_join_consecutive_pages() {
        assert_eq!(runs(&[16, 32, 48, 96, 112, 16]), [(16, 3), (96, 2), (16, 1)]);
        assert_eq!(runs(&[]), []);
        assert_eq!(runs(&[48, 32]), [(48, 1), (32, 1)]);
    }

    #[test]
    fn allocation_gives_frames_from_the_first_page() {
        let mut pages = PageAllocator::new(8);
        assert_eq!(pages.allocate(3), Some(vec![16, 32, 48]));
        assert_eq!(pages.allocate_run(1), Some(64));
        assert!(pages.is_allocated(16) && pages.is_allocated(64) && !pages.is_allocated(80));
        assert!(!pages.is_allocated(NO_FRAME) && !pages.is_allocated(17) && !pages.is_allocated(16 * 9));
        assert_eq!(pages.used_count(), 4);
    }

    #[test]
    fn nothing_for_zero_pages_or_a_full_section() {
        let mut pages = PageAllocator::new(4);
        assert_eq!(pages.allocate(0), None);
        assert_eq!(pages.allocate_run(0), None);
        assert_eq!(pages.allocate(5), None);
        assert!(pages.allocate(4).is_some());
        assert_eq!(pages.allocate(1), None);
        assert_eq!(pages.allocate_run(1), None);
    }

    #[test]
    fn a_fragmented_section_still_serves_scattered_pages() {
        let mut pages = PageAllocator::new(6);
        let all = pages.allocate(6).unwrap();
        pages.free(&[all[1], all[3], all[5]]).unwrap();
        assert_eq!(pages.allocate_run(2), None, "no two free pages are next to each other");
        assert_eq!(pages.allocate(3), Some(vec![all[1], all[3], all[5]]));
        assert_eq!(pages.used_count(), 6);
    }

    #[test]
    fn freed_pages_come_back_and_a_bad_free_changes_nothing() {
        let mut pages = PageAllocator::new(4);
        let frames = pages.allocate(2).unwrap();
        assert_eq!(pages.free(&[frames[0], 0]), Err(PageError::BadFrame(0)));
        assert_eq!(pages.free(&[frames[0], 17]), Err(PageError::BadFrame(17)));
        assert_eq!(pages.free(&[frames[0], 16 * 5]), Err(PageError::BadFrame(16 * 5)));
        assert_eq!(pages.free(&[frames[0], 48]), Err(PageError::NotAllocated(48)));
        assert_eq!(pages.free(&[frames[0], frames[0]]), Err(PageError::Duplicate(frames[0])));
        assert_eq!(pages.used_count(), 2);
        assert_eq!(pages.free(&frames), Ok(()));
        assert_eq!(pages.used_count(), 0);
        assert_eq!(pages.free(&frames), Err(PageError::NotAllocated(frames[0])), "a second free is refused");
    }

    #[test]
    fn a_mapped_page_that_is_freed_stays_in_use_until_unmapped_everywhere() {
        let mut pages = PageAllocator::new(4);
        let first = pages.allocate_run(2).unwrap();
        assert_eq!(pages.map(first, 2), Ok(()));
        assert_eq!(pages.map(first, 1), Ok(()), "the same page at a second address");
        assert_eq!((pages.view_count(first), pages.view_count(first + 16)), (2, 1));
        assert_eq!(pages.free(&[first]), Ok(()), "the title may free a page that is mapped");
        assert!(!pages.is_allocated(first) && pages.is_in_use(first), "no longer the title's, not yet back in the pool");
        assert_eq!(pages.free(&[first]), Err(PageError::NotAllocated(first)), "a second free is refused");
        assert_eq!(pages.allocate(4), None, "the page is not handed out again");
        assert_eq!(pages.unmap(first, 2), Ok(()));
        assert!(pages.is_in_use(first), "one view is left");
        assert_eq!(pages.unmap(first, 1), Ok(()));
        assert!(!pages.is_in_use(first) && pages.is_in_use(first + 16));
        assert_eq!(pages.free(&[first + 16]), Ok(()));
        assert_eq!(pages.used_count(), 0);
        assert_eq!(pages.map(first, 1), Err(PageError::NotAllocated(first)));
        assert_eq!(pages.map(first, 99), Err(PageError::BadFrame(first)));
    }

    #[test]
    fn runs_are_first_fit() {
        let mut pages = PageAllocator::new(4);
        pages.grow(4);
        assert_eq!(pages.allocate_run(2), Some(16), "pages 0 and 1");
        assert_eq!(pages.allocate_run(3), Some(48), "pages 2 to 4");
        assert_eq!(pages.allocate_run(2), Some(16 * 6), "pages 5 and 6");
        assert!(!pages.has_run(2), "page 7 is the last free page");
        assert_eq!(pages.allocate_run(2), None);
        assert_eq!(pages.allocate(2), None, "only one page is free");
        pages.free(&[16 * 4]).unwrap();
        assert!(!pages.has_run(2), "pages 3 and 7 are free, not next to each other");
        pages.free(&[16 * 5]).unwrap();
        assert!(pages.has_run(2), "pages 3 and 4 are free");
        assert_eq!(pages.allocate_run(2), Some(16 * 4));
    }

    #[test]
    fn a_run_may_cross_a_section_boundary() {
        // A commit larger than a section (a title asks for 0x89700000 bytes, a section holds 2 GiB) is one run.
        let mut pages = PageAllocator::new(3);
        pages.grow(3);
        let all = pages.allocate(6).unwrap();
        // Page 2 (last of the first section) and page 3 (first of the second) are free and adjacent.
        pages.free(&[all[2], all[3]]).unwrap();
        assert!(pages.has_run(2), "the two free pages are next to each other");
        assert_eq!(pages.allocate_run(2), Some(all[2]));
    }

    #[test]
    fn a_page_that_went_back_to_the_pool_is_dirty_until_cleared() {
        let mut pages = PageAllocator::new(3);
        let frames = pages.allocate(2).unwrap();
        assert!(!pages.is_dirty(frames[0]), "a fresh page holds zeros");
        pages.free(&frames).unwrap();
        assert!(pages.is_dirty(frames[0]) && pages.is_dirty(frames[1]));
        pages.mark_clean(frames[0]);
        assert!(!pages.is_dirty(frames[0]) && pages.is_dirty(frames[1]));
        // A freed page that a view keeps becomes dirty when the last view ends.
        let mapped = pages.allocate_run(1).unwrap();
        pages.map(mapped, 1).unwrap();
        pages.free(&[mapped]).unwrap();
        assert!(!pages.is_dirty(mapped));
        pages.unmap(mapped, 1).unwrap();
        assert!(pages.is_dirty(mapped));
        assert!(!pages.is_dirty(0) && !pages.is_dirty(16 * 9));
    }

    #[test]
    fn the_pool_does_not_grow_when_enough_pages_are_free() {
        assert_eq!(growth(100, 100, 100, 0), Growth::None);
        assert_eq!(growth(100, 5, 0, 0), Growth::None);
    }

    #[test]
    fn growth_is_rounded_up_to_a_step_within_the_room() {
        let gib = 1u64 << 30;
        // 99776 wanted, 1920 free: 97856 missing, 24 steps of 4096 are 98304.
        assert_eq!(growth(16384, 1920, 99776, 40 * gib), Growth::Add(98304));
        assert_eq!(growth(16384, 0, 1, 40 * gib), Growth::Add(GROW_STEP_PAGES));
        // Room for 5000 pages (320 MB above the headroom): the step is cut to the room.
        let commit = HEADROOM_BYTES + 5000 * PAGE_BYTES as u64;
        assert_eq!(growth(0, 0, 4100, commit), Growth::Add(5000));
        assert_eq!(growth(0, 0, 5000, commit), Growth::Add(5000));
    }

    #[test]
    fn a_request_beyond_the_commit_charge_or_the_frame_range_is_a_bound() {
        let gib = 1u64 << 30;
        assert_eq!(growth(0, 0, 10, 0), Growth::Bound { room: 0, missing: 10 });
        assert_eq!(growth(0, 0, 10, HEADROOM_BYTES), Growth::Bound { room: 0, missing: 10 });
        let commit = HEADROOM_BYTES + 5000 * PAGE_BYTES as u64;
        assert_eq!(growth(0, 0, 5001, commit), Growth::Bound { room: 5000, missing: 5001 });
        assert_eq!(growth(MAX_POOL_PAGES - 3, 0, 4, 100 * gib), Growth::Bound { room: 3, missing: 4 });
        assert_eq!(growth(MAX_POOL_PAGES - 3, 0, 3, 100 * gib), Growth::Add(3));
    }

    #[test]
    fn a_grown_allocator_serves_the_new_pages_and_keeps_the_old_state() {
        let mut pages = PageAllocator::new(2);
        let first = pages.allocate(2).unwrap();
        assert_eq!(pages.allocate(1), None);
        pages.grow(3);
        assert_eq!((pages.capacity(), pages.used_count()), (5, 2));
        assert_eq!(pages.allocate(3), Some(vec![48, 64, 80]));
        assert!(pages.is_allocated(first[0]));
    }
}
