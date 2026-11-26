//! The address space the engine manages, as plain safe code: placeholders and the views mapped
//! over them. The registry mirrors what Windows holds, so that the unsafe code only has to make
//! the calls. Every change here follows a Windows call that succeeded.

use std::collections::BTreeMap;

use crate::pages::{PAGE_BYTES, index_of};

/// What occupies a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Placeholder,
    /// A view of `size / PAGE_BYTES` consecutive pages of the backing section, from `first_frame`.
    View { first_frame: u32 },
    /// Private memory that `XMemVirtualAlloc` placed inside a reservation (the placeholder was replaced
    /// by a `VirtualAlloc2` allocation). `XMemFree` of its base turns it back into a placeholder.
    Allocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub size: usize,
    /// The reservation this range came from. Only placeholders of one origin can be coalesced.
    pub origin: u32,
    /// The protection the title reserved the range with, which every view in it takes.
    pub protect: u32,
    pub kind: Kind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionError {
    /// The range is empty, wraps around, or is not aligned on 64 KiB.
    BadRange,
    /// The range overlaps something already registered.
    Overlap,
    /// The range is not inside one placeholder.
    NotInPlaceholder,
    /// No region starts at this address.
    Unknown,
    /// The range cuts a view in two.
    Partial,
}

/// The cuts that make `[base, base + size)` a placeholder of its own, when it lies inside the
/// placeholder `[start, start + whole)`. Each step is `(address, first)`: the placeholder at
/// `address` is cut into `first` bytes and the rest. A front cut comes first, then the tail cut.
/// Empty when the range is the whole placeholder, or does not lie inside it.
pub fn split_steps(start: usize, whole: usize, base: usize, size: usize) -> Vec<(usize, usize)> {
    let Some(end) = base.checked_add(size).filter(|_| size != 0) else {
        return Vec::new();
    };
    let Some(whole_end) = start.checked_add(whole) else {
        return Vec::new();
    };
    if base < start || end > whole_end {
        return Vec::new();
    }
    let mut steps = Vec::new();
    if base > start {
        steps.push((start, base - start));
    }
    if end < whole_end {
        steps.push((base, size));
    }
    steps
}

#[derive(Debug, Default)]
pub struct Regions {
    map: BTreeMap<usize, Region>,
    next_origin: u32,
    /// Which views show each pool page: `(pool page index, view base)` to the pool page index of the first page
    /// of that view. Kept by `set_view`, `clear_view` and `remove`, so that a page finds its views without a scan
    /// of all of them (`views_of_page`).
    by_page: BTreeMap<(usize, usize), usize>,
}

fn end_of(base: usize, size: usize) -> Option<usize> {
    base.checked_add(size)
}

impl Regions {
    pub fn new() -> Regions {
        Regions::default()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The region that holds `address`, with its base.
    pub fn containing(&self, address: usize) -> Option<(usize, Region)> {
        let (&base, &region) = self.map.range(..=address).next_back()?;
        (address - base < region.size).then_some((base, region))
    }

    pub fn at(&self, base: usize) -> Option<Region> {
        self.map.get(&base).copied()
    }

    /// Registers a new reservation as one placeholder.
    pub fn insert_placeholder(&mut self, base: usize, size: usize, protect: u32) -> Result<(), RegionError> {
        let end = end_of(base, size).filter(|_| size != 0).ok_or(RegionError::BadRange)?;
        if self.map.range(..end).next_back().is_some_and(|(&b, r)| b + r.size > base) {
            return Err(RegionError::Overlap);
        }
        self.next_origin += 1;
        self.map.insert(base, Region { size, origin: self.next_origin, protect, kind: Kind::Placeholder });
        Ok(())
    }

    /// The placeholder that holds `[base, base + size)`, with its base and size.
    pub fn placeholder_holding(&self, base: usize, size: usize) -> Result<(usize, usize), RegionError> {
        let end = end_of(base, size).filter(|_| size != 0).ok_or(RegionError::BadRange)?;
        match self.containing(base) {
            Some((b, r)) if r.kind == Kind::Placeholder && end <= b + r.size => Ok((b, r.size)),
            _ => Err(RegionError::NotInPlaceholder),
        }
    }

    /// Cuts the placeholder at `base` into `[base, base + first)` and the rest.
    pub fn split(&mut self, base: usize, first: usize) -> Result<(), RegionError> {
        let region = self.at(base).filter(|r| r.kind == Kind::Placeholder).ok_or(RegionError::Unknown)?;
        if first == 0 || first >= region.size {
            return Err(RegionError::BadRange);
        }
        self.map.insert(base, Region { size: first, ..region });
        self.map.insert(base + first, Region { size: region.size - first, ..region });
        Ok(())
    }

    /// The regions that lie in `[base, base + size)` or touch it, in order, when together they
    /// cover it without a gap and belong to one reservation. The first one starts at or before
    /// `base`.
    pub fn covering(&self, base: usize, size: usize) -> Result<Vec<(usize, Region)>, RegionError> {
        let end = end_of(base, size).filter(|_| size != 0).ok_or(RegionError::BadRange)?;
        let (first, _) = self.containing(base).ok_or(RegionError::NotInPlaceholder)?;
        let found: Vec<(usize, Region)> = self.map.range(first..end).map(|(&b, &r)| (b, r)).collect();
        let mut next = first;
        for &(b, r) in &found {
            if b != next || r.origin != found[0].1.origin {
                return Err(RegionError::NotInPlaceholder);
            }
            next = b + r.size;
        }
        if next < end { Err(RegionError::NotInPlaceholder) } else { Ok(found) }
    }

    /// The base of every region of one reservation, in order, when `base` is the first of them.
    pub fn reservation_at(&self, base: usize) -> Result<Vec<usize>, RegionError> {
        let origin = self.at(base).ok_or(RegionError::Unknown)?.origin;
        let all: Vec<usize> = self.map.iter().filter(|(_, r)| r.origin == origin).map(|(&b, _)| b).collect();
        if all.first() == Some(&base) { Ok(all) } else { Err(RegionError::Unknown) }
    }

    /// Turns the whole placeholder at `base` into a view.
    pub fn set_view(&mut self, base: usize, first_frame: u32) -> Result<(), RegionError> {
        let region = self.map.get_mut(&base).filter(|r| r.kind == Kind::Placeholder).ok_or(RegionError::Unknown)?;
        region.kind = Kind::View { first_frame };
        let pages = region.size / PAGE_BYTES;
        if let Some(first) = index_of(first_frame) {
            for page in first..first + pages {
                self.by_page.insert((page, base), first);
            }
        }
        Ok(())
    }

    /// Turns the view at `base` back into a placeholder and returns what it was.
    pub fn clear_view(&mut self, base: usize) -> Result<Region, RegionError> {
        let region = self.map.get_mut(&base).ok_or(RegionError::Unknown)?;
        let before = *region;
        if !matches!(before.kind, Kind::View { .. }) {
            return Err(RegionError::Unknown);
        }
        region.kind = Kind::Placeholder;
        self.unindex(base, before);
        Ok(before)
    }

    /// Turns the whole placeholder at `base` into an allocation placed by `XMemVirtualAlloc`.
    pub fn set_allocation(&mut self, base: usize) -> Result<(), RegionError> {
        let region = self.map.get_mut(&base).filter(|r| r.kind == Kind::Placeholder).ok_or(RegionError::Unknown)?;
        region.kind = Kind::Allocation;
        Ok(())
    }

    /// Turns the allocation at `base` back into a placeholder and returns its size.
    pub fn clear_allocation(&mut self, base: usize) -> Result<usize, RegionError> {
        let region = self.map.get_mut(&base).filter(|r| r.kind == Kind::Allocation).ok_or(RegionError::Unknown)?;
        region.kind = Kind::Placeholder;
        Ok(region.size)
    }

    /// Forgets the region at `base`.
    pub fn remove(&mut self, base: usize) -> Result<Region, RegionError> {
        let region = self.map.remove(&base).ok_or(RegionError::Unknown)?;
        self.unindex(base, region);
        Ok(region)
    }

    /// Drops the page index of the view `region` at `base` (nothing for another kind).
    fn unindex(&mut self, base: usize, region: Region) {
        if let Kind::View { first_frame } = region.kind
            && let Some(first) = index_of(first_frame)
        {
            for page in first..first + region.size / PAGE_BYTES {
                self.by_page.remove(&(page, base));
            }
        }
    }

    /// The views that show pool page `index`, as `(base, pool page index of the view's first page)`, in address
    /// order. The page is at `base + (index - first) * PAGE_BYTES`. A lookup, not a scan of every view.
    pub fn views_of_page(&self, index: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.by_page.range((index, 0)..=(index, usize::MAX)).map(|(&(_, base), &first)| (base, first))
    }

    /// The bases of the views inside `[base, base + size)`, in order. The range may hold
    /// placeholders too. It must not cut a region in two, and must hold at least one view.
    pub fn views_in(&self, base: usize, size: usize) -> Result<Vec<usize>, RegionError> {
        let end = end_of(base, size).filter(|_| size != 0).ok_or(RegionError::BadRange)?;
        let mut views = Vec::new();
        for (&b, r) in self.map.range(..end) {
            if b + r.size <= base {
                continue;
            }
            if b < base || b + r.size > end {
                return Err(RegionError::Partial);
            }
            if matches!(r.kind, Kind::View { .. }) {
                views.push(b);
            }
        }
        if views.is_empty() { Err(RegionError::Unknown) } else { Ok(views) }
    }

    /// The widest run of placeholders of one origin that touch each other and hold
    /// `[base, base + size)`, as `(start, length)`. `None` when it is a single placeholder
    /// already, so there is nothing to coalesce.
    pub fn coalescible(&self, base: usize, size: usize) -> Option<(usize, usize)> {
        let (first, region) = self.containing(base)?;
        if region.kind != Kind::Placeholder {
            return None;
        }
        let origin = region.origin;
        let mut start = first;
        while let Some((&b, r)) = self.map.range(..start).next_back() {
            if r.kind == Kind::Placeholder && r.origin == origin && b + r.size == start {
                start = b;
            } else {
                break;
            }
        }
        let mut end = first + region.size;
        let wanted = base.checked_add(size)?;
        while let Some(r) = self.map.get(&end).filter(|r| r.kind == Kind::Placeholder && r.origin == origin) {
            end += r.size;
        }
        (end >= wanted && end - start > region.size).then_some((start, end - start))
    }

    /// Joins the placeholders of `[start, start + length)` into one, after `coalescible`.
    pub fn merge(&mut self, start: usize, length: usize) {
        let end = start + length;
        let inside: Vec<usize> = self.map.range(start..end).map(|(&b, _)| b).collect();
        if let Some(&head) = inside.first()
            && let Some(mut region) = self.map.get(&head).copied()
        {
            for b in &inside {
                self.map.remove(b);
            }
            region.size = length;
            self.map.insert(head, region);
        }
    }

    /// Every view, in address order, with its base.
    pub fn views(&self) -> Vec<(usize, Region)> {
        self.map.iter().filter(|(_, r)| matches!(r.kind, Kind::View { .. })).map(|(&b, &r)| (b, r)).collect()
    }

    /// The bytes of private memory placed inside reservations (`Kind::Allocation`).
    pub fn allocation_bytes(&self) -> usize {
        self.map.values().filter(|r| r.kind == Kind::Allocation).map(|r| r.size).sum()
    }

    /// Every reservation as its base and its size, in address order. The regions that came from one
    /// `insert_placeholder` are one reservation, however they were cut.
    pub fn reservations(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        let mut index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (&base, r) in &self.map {
            match index.get(&r.origin) {
                Some(&i) => out[i].1 += r.size,
                None => {
                    index.insert(r.origin, out.len());
                    out.push((base, r.size));
                }
            }
        }
        out
    }

    /// A view needs a whole number of pages, starting on a page boundary.
    pub fn page_aligned(base: usize, size: usize) -> bool {
        base.is_multiple_of(PAGE_BYTES) && size != 0 && size.is_multiple_of(PAGE_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = PAGE_BYTES;

    #[test]
    fn a_reservation_is_one_placeholder_and_overlaps_are_refused() {
        let mut r = Regions::new();
        assert_eq!(r.insert_placeholder(0x10_0000, 4 * P, 4), Ok(()));
        assert_eq!(r.insert_placeholder(0x10_0000 + 3 * P, P, 4), Err(RegionError::Overlap));
        assert_eq!(r.insert_placeholder(0x0F_0000, 2 * P, 4), Err(RegionError::Overlap));
        assert_eq!(r.insert_placeholder(0x10_0000 + 4 * P, P, 4), Ok(()));
        assert_eq!(r.insert_placeholder(0, 0, 4), Err(RegionError::BadRange));
        assert_eq!(r.insert_placeholder(usize::MAX - 5, 100, 4), Err(RegionError::BadRange));
        assert_eq!(r.containing(0x10_0000 + 4 * P - 1).map(|(b, _)| b), Some(0x10_0000));
        assert_eq!(r.containing(0x10_0000 + 5 * P), None);
    }

    #[test]
    fn views_and_reservations_are_listed_in_order() {
        let mut r = Regions::new();
        let base = 0x100_0000;
        r.insert_placeholder(base, 8 * P, 4).unwrap();
        r.insert_placeholder(0x200_0000, 2 * P, 4).unwrap();
        // Views: [P, 3P) frames 32 and 48, [4P, 5P) frame 160. The rest stays placeholders.
        for (at, size) in [(0, P), (P, 2 * P), (3 * P, P), (4 * P, P), (5 * P, P)] {
            r.split(base + at, size).unwrap();
        }
        r.set_view(base + P, 32).unwrap();
        r.set_view(base + 4 * P, 160).unwrap();
        let views: Vec<(usize, usize, Kind)> = r.views().iter().map(|(b, v)| (*b, v.size, v.kind)).collect();
        assert_eq!(views, [(base + P, 2 * P, Kind::View { first_frame: 32 }), (base + 4 * P, P, Kind::View { first_frame: 160 })]);
        assert_eq!(r.reservations(), [(base, 8 * P), (0x200_0000, 2 * P)], "a cut reservation is still one");
    }

    #[test]
    fn views_of_a_page_are_found_without_a_scan_and_follow_clear_and_remove() {
        let mut r = Regions::new();
        let base = 0x100_0000;
        r.insert_placeholder(base, 8 * P, 4).unwrap();
        for (at, size) in [(0, 2 * P), (2 * P, 2 * P), (4 * P, 4 * P)] {
            r.split(base + at, size).ok();
        }
        // Two views of pool pages 1 and 2 (frames 32 and 48), one more of page 2 and 3 (frames 48 and 64).
        r.set_view(base, 32).unwrap();
        r.set_view(base + 2 * P, 48).unwrap();
        let of = |r: &Regions, i: usize| r.views_of_page(i).collect::<Vec<_>>();
        assert_eq!(of(&r, 0), []);
        assert_eq!(of(&r, 1), [(base, 1)]);
        assert_eq!(of(&r, 2), [(base, 1), (base + 2 * P, 2)], "two views show page 2, in address order");
        assert_eq!(of(&r, 3), [(base + 2 * P, 2)]);
        r.clear_view(base).unwrap();
        assert_eq!(of(&r, 1), []);
        assert_eq!(of(&r, 2), [(base + 2 * P, 2)]);
        r.remove(base + 2 * P).unwrap();
        assert_eq!(of(&r, 2), []);
        assert_eq!(of(&r, 3), []);
    }

    #[test]
    fn splitting_keeps_the_origin() {
        let mut r = Regions::new();
        r.insert_placeholder(0x10_0000, 4 * P, 4).unwrap();
        assert_eq!(r.split(0x10_0000, P), Ok(()));
        assert_eq!(r.split(0x10_0000, P), Err(RegionError::BadRange));
        assert_eq!(r.split(0x10_0000 + P, 3 * P), Err(RegionError::BadRange));
        assert_eq!(r.split(0x10_0000 + 2 * P, P), Err(RegionError::Unknown));
        let (a, b) = (r.at(0x10_0000).unwrap(), r.at(0x10_0000 + P).unwrap());
        assert_eq!((a.size, b.size, a.origin == b.origin), (P, 3 * P, true));
        assert_eq!(r.placeholder_holding(0x10_0000 + P, 2 * P), Ok((0x10_0000 + P, 3 * P)));
        assert_eq!(r.placeholder_holding(0x10_0000, 2 * P), Err(RegionError::NotInPlaceholder));
    }

    #[test]
    fn views_replace_placeholders_and_go_back() {
        let mut r = Regions::new();
        r.insert_placeholder(0x10_0000, 4 * P, 4).unwrap();
        r.split(0x10_0000, 2 * P).unwrap();
        r.set_view(0x10_0000, 48).unwrap();
        assert_eq!(r.placeholder_holding(0x10_0000, P), Err(RegionError::NotInPlaceholder));
        assert_eq!(r.views_in(0x10_0000, 4 * P), Ok(vec![0x10_0000]));
        assert_eq!(r.views_in(0x10_0000, P), Err(RegionError::Partial), "half of a view");
        assert_eq!(r.views_in(0x10_0000 + 2 * P, 2 * P), Err(RegionError::Unknown), "only placeholders there");
        let was = r.clear_view(0x10_0000).unwrap();
        assert_eq!(was.kind, Kind::View { first_frame: 48 });
        assert_eq!(r.clear_view(0x10_0000), Err(RegionError::Unknown), "already a placeholder");
        assert_eq!(r.set_view(0x10_0000 + P, 16), Err(RegionError::Unknown), "inside a placeholder, not at its start");
    }

    #[test]
    fn only_neighbours_of_one_reservation_coalesce() {
        let mut r = Regions::new();
        r.insert_placeholder(0x10_0000, 3 * P, 4).unwrap();
        r.insert_placeholder(0x10_0000 + 3 * P, P, 4).unwrap();
        r.split(0x10_0000, P).unwrap();
        r.split(0x10_0000 + P, P).unwrap();
        assert_eq!(r.coalescible(0x10_0000 + P, P), Some((0x10_0000, 3 * P)), "the other reservation stays apart");
        r.set_view(0x10_0000 + P, 16).unwrap();
        assert_eq!(r.coalescible(0x10_0000, P), None, "a view in the way");
        r.clear_view(0x10_0000 + P).unwrap();
        r.merge(0x10_0000, 3 * P);
        assert_eq!(r.at(0x10_0000).map(|x| x.size), Some(3 * P));
        assert_eq!(r.coalescible(0x10_0000, 3 * P), None);
        assert!(r.at(0x10_0000 + P).is_none());
    }

    #[test]
    fn the_cuts_for_a_range_inside_a_placeholder() {
        let (s, w) = (0x10_0000, 8 * P);
        assert!(split_steps(s, w, s, w).is_empty(), "the whole placeholder");
        assert_eq!(split_steps(s, w, s, 2 * P), [(s, 2 * P)], "front: one cut");
        assert_eq!(split_steps(s, w, s + 6 * P, 2 * P), [(s, 6 * P)], "tail: one cut");
        assert_eq!(split_steps(s, w, s + 3 * P, 2 * P), [(s, 3 * P), (s + 3 * P, 2 * P)], "middle: two cuts");
        assert!(split_steps(s, w, s - P, 2 * P).is_empty(), "starts before");
        assert!(split_steps(s, w, s + 7 * P, 2 * P).is_empty(), "ends after");
        assert!(split_steps(s, w, s, 0).is_empty(), "empty");
        assert!(split_steps(s, w, usize::MAX - P, 2 * P).is_empty(), "wraps");
    }

    #[test]
    fn an_allocation_goes_back_to_a_placeholder() {
        let mut r = Regions::new();
        r.insert_placeholder(0x10_0000, 4 * P, 4).unwrap();
        r.split(0x10_0000, P).unwrap();
        assert_eq!(r.set_allocation(0x10_0000), Ok(()));
        assert_eq!(r.set_allocation(0x10_0000), Err(RegionError::Unknown), "no longer a placeholder");
        assert_eq!(r.placeholder_holding(0x10_0000, P), Err(RegionError::NotInPlaceholder));
        assert_eq!(r.clear_view(0x10_0000), Err(RegionError::Unknown), "not a view");
        assert_eq!(r.clear_allocation(0x10_0000 + P), Err(RegionError::Unknown));
        assert_eq!(r.clear_allocation(0x10_0000), Ok(P));
        assert_eq!(r.coalescible(0x10_0000, P), Some((0x10_0000, 4 * P)));
    }

    #[test]
    fn removing_forgets_a_region() {
        let mut r = Regions::new();
        r.insert_placeholder(0x20_0000, P, 4).unwrap();
        assert!(!r.is_empty());
        assert!(r.remove(0x20_0000).is_ok());
        assert_eq!(r.remove(0x20_0000), Err(RegionError::Unknown));
        assert!(r.is_empty());
    }

    #[test]
    fn a_range_is_covered_by_one_reservation_without_gaps() {
        let mut r = Regions::new();
        r.insert_placeholder(0x10_0000, 4 * P, 4).unwrap();
        r.insert_placeholder(0x10_0000 + 4 * P, 2 * P, 4).unwrap();
        r.split(0x10_0000, P).unwrap();
        r.set_view(0x10_0000, 16).unwrap();
        let found = r.covering(0x10_0000 + P / 2, P).unwrap();
        assert_eq!(found.iter().map(|&(b, _)| b).collect::<Vec<_>>(), [0x10_0000, 0x10_0000 + P]);
        assert_eq!(r.covering(0x10_0000 + 3 * P, 2 * P), Err(RegionError::NotInPlaceholder), "two reservations");
        assert_eq!(r.covering(0x10_0000 + 5 * P, 2 * P), Err(RegionError::NotInPlaceholder), "past the end");
        assert_eq!(r.covering(0x0F_0000, P), Err(RegionError::NotInPlaceholder), "nothing there");
        assert_eq!(r.reservation_at(0x10_0000).unwrap(), [0x10_0000, 0x10_0000 + P]);
        assert_eq!(r.reservation_at(0x10_0000 + P), Err(RegionError::Unknown), "not the first of its reservation");
        assert_eq!(r.reservation_at(0x10_0000 + 4 * P).unwrap(), [0x10_0000 + 4 * P]);
    }

    #[test]
    fn views_need_whole_pages() {
        assert!(Regions::page_aligned(0x10_0000, P));
        assert!(!Regions::page_aligned(0x10_1000, P));
        assert!(!Regions::page_aligned(0x10_0000, 0x1000));
        assert!(!Regions::page_aligned(0x10_0000, 0));
    }
}
