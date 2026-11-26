//! Which pool pages the GPU mirror of `d3d12_xs.dll` has to copy again. Plain safe code: the
//! transitions answer which title view pages must change protection, the engine makes the calls.
//!
//! A page is `Dirty` (the mirror may be stale, every read write view is read write) or `Clean` (the
//! mirror equals the page, every view of it whose title protection is read write is `PAGE_READONLY`
//! so that the next write faults and is seen). Views the title made read only are never touched.
//!
//! A page written again and again soon after each copy (a video decoder writes every plane page once a
//! frame) would fault once per frame. After `HOT_FAULTS` such faults in a row it is held like a page a
//! kernel write may reach: unprotected and copied at every sync for `HOT_HOLD`, then protected again.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::pages::PAGE_BYTES;

/// How long a page that a kernel write may still reach (`prepare`) stays unprotected by default.
pub const IO_HOLD: Duration = Duration::from_secs(2);
/// How long a page stays open for a write that ends when a call returns (`Engine::begin_write`) at most,
/// should the call never come back.
pub const CALL_HOLD: Duration = Duration::from_secs(600);

/// A write fault this soon after the page was made clean counts as a quick one.
pub const HOT_WINDOW: Duration = Duration::from_millis(500);
/// Quick faults in a row that make a page hot.
pub const HOT_FAULTS: u8 = 3;
/// How long a hot page stays unprotected and is copied at every sync by default.
pub const HOT_HOLD: Duration = Duration::from_millis(1000);
/// Minimum writable rest after a CPU write fault before `take` re-protects the page by default.
pub const GUARD_REST: Duration = Duration::from_millis(32);

/// Parses a hold duration in milliseconds from an optional string, falling back to `default`.
fn parse_hold_ms(raw: Option<&str>, default: Duration) -> Duration {
    match raw.and_then(|s| s.trim().parse::<u64>().ok()) {
        Some(ms) => Duration::from_millis(ms),
        None => default,
    }
}

/// Parses `DECANT_XMEM_HOT_HOLD_MS` from an optional string, falling back to [`HOT_HOLD`].
pub fn parse_hot_hold_ms(raw: Option<&str>) -> Duration {
    parse_hold_ms(raw, HOT_HOLD)
}

/// Parses `DECANT_XMEM_IO_HOLD_MS` from an optional string, falling back to [`IO_HOLD`].
pub fn parse_io_hold_ms(raw: Option<&str>) -> Duration {
    parse_hold_ms(raw, IO_HOLD)
}

/// Parses `DECANT_XMEM_GUARD_REST_MS` from an optional string, falling back to [`GUARD_REST`].
pub fn parse_guard_rest_ms(raw: Option<&str>) -> Duration {
    parse_hold_ms(raw, GUARD_REST)
}

/// How long a hot page stays unprotected and is copied at every sync.
pub fn hot_hold() -> Duration {
    static HOLD: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *HOLD.get_or_init(|| parse_hot_hold_ms(std::env::var("DECANT_XMEM_HOT_HOLD_MS").ok().as_deref()))
}

/// How long a faulted page stays writable before `take` re-protects it.
pub fn guard_rest() -> Duration {
    static REST: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *REST.get_or_init(|| parse_guard_rest_ms(std::env::var("DECANT_XMEM_GUARD_REST_MS").ok().as_deref()))
}

/// How long a page that a kernel write may still reach stays unprotected.
pub fn io_hold() -> Duration {
    static HOLD: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *HOLD.get_or_init(|| parse_io_hold_ms(std::env::var("DECANT_XMEM_IO_HOLD_MS").ok().as_deref()))
}

/// Scales the hold for large overlapped reads so multi-hundred-MB buffers do not expire mid-read.
pub fn io_hold_for(bytes: usize) -> Duration {
    io_hold().max(Duration::from_millis(((bytes as u64) / (32 * 1024)).min(30_000)))
}

/// One title view page of a pool page: its address, whether the title protection is read write,
/// and whether that alias is currently open (`PAGE_READWRITE`) rather than guarded (`PAGE_READONLY`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Alias {
    addr: usize,
    rw: bool,
    open: bool,
}

#[derive(Debug, Default)]
pub struct Track {
    /// Per pool page: the mirror equals the page and the read write views are protected.
    clean: Vec<bool>,
    /// Per pool page: it is in `pending`.
    queued: Vec<bool>,
    aliases: HashMap<u32, Vec<Alias>>,
    addresses: HashMap<usize, (u32, bool)>,
    held: HashMap<u32, Instant>,
    io_held: HashSet<u32>,
    active_writes: HashMap<u32, u32>,
    /// When `take` last made the page clean, and the quick faults in a row since (`HOT_WINDOW`).
    cleaned: HashMap<u32, (Instant, u8)>,
    /// The dirty pages that have a view, waiting for `take`.
    pending: Vec<u32>,
    /// Per pool page: `take` saw it committed. A pool page is never decommitted, so the question is
    /// asked once per page (a `VirtualQuery` each time took 5 to 7 s of every 200 syncs while a retail title
    /// loads its world, most of it for held pages taken at every sync).
    committed: Vec<bool>,
}

/// The pages of `addrs` (title view page addresses) as runs `(start, bytes)`, adjacent pages merged.
pub fn runs(mut addrs: Vec<usize>) -> Vec<(usize, usize)> {
    addrs.sort_unstable();
    addrs.dedup();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for a in addrs {
        match out.last_mut() {
            Some((start, bytes)) if *start + *bytes == a => *bytes += PAGE_BYTES,
            _ => out.push((a, PAGE_BYTES)),
        }
    }
    out
}

impl Track {
    /// A tracker over `pages` pool pages, none of them mapped.
    pub fn new(pages: usize) -> Track {
        Track { clean: vec![false; pages], queued: vec![false; pages], committed: vec![false; pages], ..Track::default() }
    }

    /// The pool grew to `pages` pages.
    pub fn grow(&mut self, pages: usize) {
        self.clean.resize(pages, false);
        self.queued.resize(pages, false);
        self.committed.resize(pages, false);
    }

    /// The pool page at a writable view address, without consulting the address space registry.
    pub fn writable_page(&self, addr: usize) -> Option<u32> {
        self.addresses.get(&addr).and_then(|&(page, rw)| rw.then_some(page))
    }

    /// Whether `page` is known to be committed in its backing section.
    pub fn is_committed(&self, page: u32) -> bool {
        self.committed.get(page as usize).copied().unwrap_or(false)
    }

    /// Records that `page` is committed in its backing section.
    pub fn mark_committed(&mut self, page: u32) {
        if let Some(slot) = self.committed.get_mut(page as usize) {
            *slot = true;
        }
    }

    /// Appends the addresses of the open read write views of `page` to `out` and marks them closed.
    fn take_open_rw(&mut self, page: u32, out: &mut Vec<usize>) {
        if let Some(v) = self.aliases.get_mut(&page) {
            for a in v.iter_mut() {
                if a.rw && a.open {
                    a.open = false;
                    out.push(a.addr);
                }
            }
        }
    }

    /// Makes the page dirty and queues it when it has a view, appending any closed read write views
    /// to `out` and marking them open.
    fn dirty_into(&mut self, page: u32, out: &mut Vec<usize>) {
        let i = page as usize;
        self.clean[i] = false;
        if !self.queued[i] && self.aliases.contains_key(&page) {
            self.queued[i] = true;
            self.pending.push(page);
        }
        if let Some(v) = self.aliases.get_mut(&page) {
            for a in v.iter_mut() {
                if a.rw && !a.open {
                    a.open = true;
                    out.push(a.addr);
                }
            }
        }
    }

    /// Makes the page dirty and queues it when it has a view. Returns the read write views to make
    /// read write again when it was clean.
    fn dirty(&mut self, page: u32) -> Vec<usize> {
        let mut out = Vec::new();
        self.dirty_into(page, &mut out);
        out
    }

    /// The bytes of the page changed outside its views: the allocator handed it out, or the engine
    /// wrote it through the pool.
    pub fn on_changed(&mut self, page: u32) -> Vec<usize> {
        self.dirty(page)
    }

    /// A view of the page appeared at `addr`, with the title protection `rw` (a new map counts as a write).
    pub fn on_map(&mut self, page: u32, addr: usize, rw: bool) -> Vec<usize> {
        self.addresses.insert(addr, (page, rw));
        self.aliases.entry(page).or_default().push(Alias { addr, rw, open: false });
        self.dirty(page)
    }

    /// The view of the page at `addr` is gone.
    pub fn on_unmap(&mut self, page: u32, addr: usize) {
        self.addresses.remove(&addr);
        if let Some(v) = self.aliases.get_mut(&page) {
            v.retain(|a| a.addr != addr);
            if v.is_empty() {
                self.aliases.remove(&page);
                self.held.remove(&page);
                self.io_held.remove(&page);
                self.active_writes.remove(&page);
                self.cleaned.remove(&page);
            }
        }
    }

    /// A write faulted on a read write view `fault_addr` of `page` at `now`. `Some(views)` when
    /// `fault_addr` needs opening (`PAGE_READWRITE`): only `fault_addr` is opened and returned, while
    /// sibling aliases stay guarded until they fault or are opened for I/O. `None` when `fault_addr`
    /// was already open (another thread won the race, the faulting write may simply run again).
    /// The `HOT_FAULTS`th quick fault in a row makes the page hot: held for `hot_hold()`.
    pub fn on_fault(&mut self, page: u32, fault_addr: usize, now: Instant) -> Option<Vec<usize>> {
        let i = page as usize;
        let was_clean = self.clean[i];
        let needs_open = self
            .aliases
            .get(&page)
            .and_then(|v| v.iter().find(|a| a.addr == fault_addr && a.rw))
            .is_some_and(|a| !a.open);
        if !was_clean && !needs_open {
            return None;
        }
        if was_clean && let Some((at, quick)) = self.cleaned.get_mut(&page) {
            *quick = if now.saturating_duration_since(*at) < HOT_WINDOW { *quick + 1 } else { 0 };
            if *quick >= HOT_FAULTS {
                *quick = 0;
                self.held.insert(page, now + hot_hold());
                self.io_held.remove(&page);
            }
        }
        let rest = guard_rest();
        if !rest.is_zero() && self.aliases.contains_key(&page) {
            let until = now + rest;
            let held = self.held.entry(page).or_insert(until);
            *held = (*held).max(until);
        }
        self.clean[i] = false;
        if !self.queued[i] && self.aliases.contains_key(&page) {
            self.queued[i] = true;
            self.pending.push(page);
        }
        if let Some(v) = self.aliases.get_mut(&page)
            && let Some(alias) = v.iter_mut().find(|a| a.addr == fault_addr && a.rw)
            && (was_clean || !alias.open)
        {
            alias.open = true;
            return Some(vec![fault_addr]);
        }
        None
    }

    /// Whether the page is held now (a kernel write may reach it, or it is hot).
    pub fn is_held(&self, page: u32, now: Instant) -> bool {
        self.held.get(&page).is_some_and(|&until| now < until)
    }

    /// Pins `page` open for an active asynchronous write, appending read write views to `out`.
    pub fn begin_write_pin_into(&mut self, page: u32, out: &mut Vec<usize>) {
        if self.aliases.contains_key(&page) {
            *self.active_writes.entry(page).or_insert(0) += 1;
        }
        self.dirty_into(page, out);
    }

    /// Pins `page` open for an active asynchronous write. Marks the page dirty, increments the
    /// active write pin count, and returns the read write views to make writable.
    pub fn begin_write_pin(&mut self, page: u32) -> Vec<usize> {
        let mut out = Vec::new();
        self.begin_write_pin_into(page, &mut out);
        out
    }

    /// Unpins `page` after an active asynchronous write finishes. Decrements the pin count and
    /// marks the page dirty once with no hold so the next `take` copies it and protects it again.
    pub fn finish_write_pin(&mut self, page: u32) {
        if let Some(count) = self.active_writes.get_mut(&page) {
            if *count <= 1 {
                self.active_writes.remove(&page);
            } else {
                *count -= 1;
            }
        }
        let _ = self.dirty(page);
    }

    /// Transitions an active write pin on `page` into a timed fallback hold until `until` when
    /// an overlapped read returns `ERROR_IO_PENDING`.
    pub fn convert_pin_to_hold(&mut self, page: u32, until: Instant) {
        if let Some(count) = self.active_writes.get_mut(&page) {
            if *count <= 1 {
                self.active_writes.remove(&page);
            } else {
                *count -= 1;
            }
        }
        if self.aliases.contains_key(&page) {
            let held = self.held.entry(page).or_insert(until);
            *held = (*held).max(until);
            self.io_held.insert(page);
        }
        let _ = self.dirty(page);
    }

    /// Clears an I/O fallback hold on `page` when the overlapped read completes and marks the
    /// page dirty once so the next `take` copies it and protects it immediately.
    pub fn clear_io_hold(&mut self, page: u32) {
        if self.io_held.remove(&page) {
            self.held.remove(&page);
        }
        let _ = self.dirty(page);
    }

    /// A kernel write may land in the page: it is dirty, unprotected and held for `io_hold()`. Returns the
    /// views to make read write again.
    pub fn prepare(&mut self, page: u32, now: Instant) -> Vec<usize> {
        self.prepare_until(page, now + io_hold())
    }

    /// As `prepare`, held until `until`, appending read write views to `out`.
    pub fn prepare_until_into(&mut self, page: u32, until: Instant, out: &mut Vec<usize>) {
        if self.aliases.contains_key(&page) {
            let held = self.held.entry(page).or_insert(until);
            *held = (*held).max(until);
        }
        self.dirty_into(page, out);
    }

    /// As `prepare`, held until `until` (a longer hold already there stays).
    pub fn prepare_until(&mut self, page: u32, until: Instant) -> Vec<usize> {
        let mut out = Vec::new();
        self.prepare_until_into(page, until, &mut out);
        out
    }

    /// The write a long hold was for ended at `now`: the page is held for `io_hold()` from now. That
    /// shortens a `CALL_HOLD` (any hold that ends later than the longest other one, `hot_hold()`) and
    /// keeps a hot or kernel write hold that ends later.
    pub fn hold_from(&mut self, page: u32, now: Instant) {
        if let Some(until) = self.held.get_mut(&page) {
            let hold = io_hold();
            let end = now + hold;
            if *until > now + hot_hold().max(hold) || *until < end {
                *until = end;
            }
        }
    }

    /// The dirty pages waiting for `take`.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Pending pages with at least one view that are not yet marked committed, scanning up to `cap`
    /// already committed pages.
    pub fn uncommitted_pending(&self, cap: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut count = 0usize;
        for &page in &self.pending {
            if count >= cap {
                break;
            }
            if !self.aliases.contains_key(&page) {
                continue;
            }
            if !self.committed[page as usize] {
                out.push(page);
            } else {
                count += 1;
            }
        }
        out
    }

    /// Whether `address` currently maps a writable view that is clean or guarded (`!open`).
    pub fn is_clean_addr(&self, address: usize) -> bool {
        let Some(&(page, rw)) = self.addresses.get(&address) else { return false };
        if !rw {
            return false;
        }
        self.clean[page as usize]
            || self
                .aliases
                .get(&page)
                .and_then(|v| v.iter().find(|a| a.addr == address))
                .is_some_and(|a| !a.open)
    }

    /// Whether the page has a view with a read write title protection.
    pub fn writable(&self, page: u32) -> bool {
        self.aliases.get(&page).is_some_and(|v| v.iter().any(|a| a.rw))
    }

    /// The dirty pages with a view, at most `cap`, for the caller to copy. Each page that is not held is
    /// marked clean and its read write views are returned to be protected, before the caller copies. A
    /// page without a view stays dirty and unprotected, and is dropped from the queue until a view appears.
    /// A page for which `committed` is false was never touched: it holds zeros as its mirror does, so it is
    /// clean without a copy and without protection (it cannot be protected), and its first access, which
    /// commits it, makes it dirty (`on_fault`).
    pub fn take(&mut self, now: Instant, cap: usize, committed: impl Fn(u32) -> bool) -> (Vec<u32>, Vec<usize>) {
        let mut taken = Vec::new();
        let mut protect = Vec::new();
        let pending = std::mem::take(&mut self.pending);
        let mut keep = Vec::new();
        for (idx, &page) in pending.iter().enumerate() {
            let i = page as usize;
            if taken.len() >= cap {
                keep.extend_from_slice(&pending[idx..]);
                break;
            } else if !self.aliases.contains_key(&page) {
                self.queued[i] = false;
            } else if !self.committed[i] && !committed(page) {
                self.clean[i] = true;
                self.queued[i] = false;
            } else if self.active_writes.get(&page).is_some_and(|&n| n > 0)
                || self.held.get(&page).is_some_and(|&until| now < until)
            {
                self.committed[i] = true;
                taken.push(page);
                keep.push(page);
            } else {
                self.committed[i] = true;
                self.held.remove(&page);
                self.io_held.remove(&page);
                self.take_open_rw(page, &mut protect);
                let quick = self.cleaned.get(&page).map_or(0, |&(_, q)| q);
                self.cleaned.insert(page, (now, quick));
                self.clean[i] = true;
                self.queued[i] = false;
                taken.push(page);
            }
        }
        self.pending = keep;
        (taken, protect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = PAGE_BYTES;

    fn track() -> Track {
        Track::new(16)
    }

    fn all(_: u32) -> bool {
        true
    }

    #[test]
    fn writable_addresses_follow_alias_removal_and_reuse() {
        let mut t = track();
        t.on_map(2, 0x10_0000, true);
        t.on_map(2, 0x20_0000, false);
        assert_eq!(t.writable_page(0x10_0000), Some(2));
        assert_eq!(t.writable_page(0x20_0000), None);
        t.on_unmap(2, 0x10_0000);
        assert_eq!(t.writable_page(0x10_0000), None);
        t.on_map(3, 0x10_0000, true);
        assert_eq!(t.writable_page(0x10_0000), Some(3));
        t.on_unmap(2, 0x20_0000);
        assert_eq!(t.writable_page(0x10_0000), Some(3));
    }

    #[test]
    fn a_page_seen_committed_is_not_asked_about_again() {
        let mut t = track();
        t.on_map(3, 0x10_0000, true);
        let asked = std::cell::Cell::new(0);
        let committed = |_: u32| {
            asked.set(asked.get() + 1);
            true
        };
        assert_eq!(t.take(Instant::now(), 16, committed).0, [3]);
        t.on_changed(3);
        assert_eq!(t.take(Instant::now(), 16, committed).0, [3]);
        assert_eq!(asked.get(), 1, "the second take knows the page is committed");
    }

    #[test]
    fn a_new_map_is_dirty_and_taken_once() {
        let mut t = track();
        t.on_changed(3);
        t.on_map(3, 0x10_0000, true);
        let now = Instant::now();
        let (pages, protect) = t.take(now, 100, all);
        assert_eq!(pages, [3]);
        assert_eq!(protect, [0x10_0000], "the read write view is protected");
        assert_eq!(t.take(now, 100, all), (vec![], vec![]), "clean pages are not returned again");
    }

    #[test]
    fn one_view_per_fault_opens_only_faulting_alias_and_protects_only_opened_aliases() {
        let mut t = track();
        t.on_map(2, 0x10_0000, true);
        t.on_map(2, 0x20_0000, true);
        t.on_map(2, 0x30_0000, false);
        let (_, mut protect) = t.take(Instant::now(), 10, all);
        protect.sort_unstable();
        assert_eq!(protect, [0x10_0000, 0x20_0000], "the read only view is never touched");

        let now = Instant::now();
        assert_eq!(t.on_fault(2, 0x10_0000, now), Some(vec![0x10_0000]), "only faulting alias opens");
        assert_eq!(t.on_fault(2, 0x10_0000, now), None, "a second fault on an already open alias has nothing to do");
        assert_eq!(t.on_fault(2, 0x20_0000, now), Some(vec![0x20_0000]), "closed sibling alias opens when it faults while dirty");
        assert_eq!(t.on_fault(2, 0x20_0000, now), None);

        let (pages, mut protect2) = t.take(now + GUARD_REST, 10, all);
        protect2.sort_unstable();
        assert_eq!(pages, [2], "the page is back in the dirty list");
        assert_eq!(protect2, [0x10_0000, 0x20_0000]);

        // Faulting on only one alias protects only that opened alias on the next take.
        assert_eq!(t.on_fault(2, 0x10_0000, now + GUARD_REST), Some(vec![0x10_0000]));
        let (_, protect3) = t.take(now + GUARD_REST * 2, 10, all);
        assert_eq!(protect3, [0x10_0000]);
    }

    #[test]
    fn begin_write_pin_and_prepare_open_closed_sibling_aliases() {
        let mut t = track();
        t.on_map(3, 0x10_0000, true);
        t.on_map(3, 0x20_0000, true);
        let now = Instant::now();
        let _ = t.take(now, 10, all);
        // Only alias 0x10_0000 faults, leaving 0x20_0000 closed.
        assert_eq!(t.on_fault(3, 0x10_0000, now), Some(vec![0x10_0000]));
        // Kernel I/O pin must open the closed sibling 0x20_0000 as well.
        assert_eq!(t.begin_write_pin(3), vec![0x20_0000]);
        t.finish_write_pin(3);
        let (_, mut protect) = t.take(now + GUARD_REST, 10, all);
        protect.sort_unstable();
        assert_eq!(protect, [0x10_0000, 0x20_0000]);
    }

    #[test]
    fn a_new_alias_of_a_clean_page_opens_the_others() {
        let mut t = track();
        t.on_map(1, 0x10_0000, true);
        t.take(Instant::now(), 10, all);
        let mut open = t.on_map(1, 0x20_0000, true);
        open.sort_unstable();
        assert_eq!(open, [0x10_0000, 0x20_0000]);
        assert_eq!(t.take(Instant::now(), 10, all).0, [1]);
    }

    #[test]
    fn a_page_without_a_view_stays_dirty_and_is_taken_after_its_map() {
        let mut t = track();
        assert!(t.on_changed(4).is_empty());
        assert_eq!(t.take(Instant::now(), 10, all), (vec![], vec![]));
        t.on_map(4, 0x10_0000, true);
        assert_eq!(t.take(Instant::now(), 10, all).0, [4]);
        t.on_unmap(4, 0x10_0000);
        t.on_changed(4);
        assert_eq!(t.take(Instant::now(), 10, all).0, Vec::<u32>::new(), "no view left");
        assert!(!t.writable(4));
    }

    #[test]
    fn a_held_page_is_returned_each_time_and_protected_only_after_the_hold() {
        let mut t = track();
        t.on_map(5, 0x10_0000, true);
        let now = Instant::now();
        t.take(now, 10, all);
        assert_eq!(t.prepare(5, now), [0x10_0000], "the clean page is opened");
        let (pages, protect) = t.take(now, 10, all);
        assert_eq!((pages, protect), (vec![5], vec![]));
        assert_eq!(t.take(now + IO_HOLD / 2, 10, all), (vec![5], vec![]), "still held");
        let (pages, protect) = t.take(now + IO_HOLD, 10, all);
        assert_eq!((pages, protect), (vec![5], vec![0x10_0000]), "the hold ended, the page is protected again");
        assert_eq!(t.take(now + IO_HOLD, 10, all).0, Vec::<u32>::new());
    }

    #[test]
    fn the_cap_leaves_the_rest_for_the_next_call_and_growth_keeps_states() {
        let mut t = track();
        for p in 0..4u32 {
            t.on_map(p, 0x10_0000 + p as usize * P, true);
        }
        let now = Instant::now();
        assert_eq!(t.take(now, 3, all).0, [0, 1, 2]);
        t.grow(64);
        t.on_map(40, 0x90_0000, true);
        assert_eq!(t.take(now, 3, all).0, [3, 40]);
        assert_eq!(t.on_fault(0, 0x10_0000, Instant::now()).map(|v| v.len()), Some(1), "page 0 kept its clean state over the growth");
    }

    #[test]
    fn a_page_never_committed_is_clean_without_a_copy_until_its_first_access() {
        let mut t = track();
        t.on_map(6, 0x10_0000, true);
        t.on_map(7, 0x20_0000, true);
        let now = Instant::now();
        let (pages, protect) = t.take(now, 10, |page| page == 7);
        assert_eq!((pages, protect), (vec![7], vec![0x20_0000]), "the untouched page is neither copied nor protected");
        assert_eq!(t.take(now, 10, all).0, Vec::<u32>::new(), "and not queued again");
        assert_eq!(t.on_fault(6, 0x10_0000, Instant::now()), Some(vec![0x10_0000]), "its first access makes it dirty");
        assert_eq!(t.take(now, 10, all).0, [6]);
    }

    #[test]
    fn a_page_written_soon_after_each_copy_becomes_hot_and_stays_open_for_the_hold() {
        let mut t = track();
        t.on_map(8, 0x10_0000, true);
        t.on_map(8, 0x20_0000, true);
        let mut now = Instant::now();
        t.take(now, 10, all);
        for round in 1..HOT_FAULTS {
            now += Duration::from_millis(30);
            assert!(t.on_fault(8, 0x10_0000, now).is_some());
            now += GUARD_REST;
            let (pages, protect) = t.take(now, 10, all);
            assert_eq!((pages, protect.len()), (vec![8], 1), "quick fault {round}: copied and protected again");
        }
        now += Duration::from_millis(30);
        assert!(t.on_fault(8, 0x10_0000, now).is_some());
        assert!(t.is_held(8, now + GUARD_REST), "the third quick fault in a row makes it hot");
        for _ in 0..3 {
            now += Duration::from_millis(30);
            assert_eq!(t.take(now, 10, all), (vec![8], vec![]), "a hot page is copied at every sync and stays writable");
        }
        let (pages, protect) = t.take(now + HOT_HOLD, 10, all);
        assert_eq!((pages, protect.len()), (vec![8], 1), "after the hold it is protected again");
    }

    #[test]
    fn a_slow_fault_does_not_count_towards_hot() {
        let mut t = track();
        t.on_map(9, 0x10_0000, true);
        let mut now = Instant::now();
        t.take(now, 10, all);
        for _ in 0..10 {
            now += HOT_WINDOW + Duration::from_millis(1);
            assert!(t.on_fault(9, 0x10_0000, now).is_some());
            now += GUARD_REST;
            assert!(!t.is_held(9, now), "a page written now and then stays tracked by faults");
            t.take(now, 10, all);
        }
    }

    #[test]
    fn a_page_of_a_blocking_read_stays_open_until_the_read_returns() {
        let mut t = track();
        let now = Instant::now();
        t.on_map(5, 0x10_0000, true);
        t.take(now, 10, all);
        assert_eq!(t.prepare_until(5, now + CALL_HOLD), [0x10_0000], "the clean page is opened");
        let late = now + IO_HOLD * 5;
        assert_eq!(t.take(late, 10, all), (vec![5], vec![]), "a read longer than IO_HOLD keeps it open and copied");
        t.hold_from(5, late);
        assert_eq!(t.take(late + IO_HOLD / 2, 10, all), (vec![5], vec![]), "held for IO_HOLD after the read");
        assert_eq!(t.take(late + IO_HOLD, 10, all), (vec![5], vec![0x10_0000]), "then copied once more and protected");
        t.prepare(5, late);
        t.hold_from(5, late);
        assert!(t.is_held(5, late + IO_HOLD / 2), "a short hold is not cut by the end of another read");
    }

    #[test]
    fn runs_merge_adjacent_pages() {
        assert_eq!(runs(vec![0x30_0000, 0x10_0000, 0x10_0000 + P, 0x10_0000]), [(0x10_0000, 2 * P), (0x30_0000, P)]);
        assert!(runs(vec![]).is_empty());
    }

    #[test]
    fn write_pins_keep_page_open_until_last_writer_finishes_then_protect_immediately() {
        let mut t = track();
        let now = Instant::now();
        t.on_map(5, 0x10_0000, true);
        assert_eq!(t.take(now, 10, all), (vec![5], vec![0x10_0000]));

        assert_eq!(t.begin_write_pin(5), [0x10_0000]);
        assert!(t.begin_write_pin(5).is_empty());
        assert_eq!(t.take(now, 10, all), (vec![5], vec![]));

        t.finish_write_pin(5);
        assert_eq!(t.take(now, 10, all), (vec![5], vec![]));

        t.finish_write_pin(5);
        assert_eq!(t.take(now, 10, all), (vec![5], vec![0x10_0000]));
        assert_eq!(t.take(now, 10, all), (vec![], vec![]));
    }

    #[test]
    fn parse_hot_hold_ms_parses_override_and_defaults_to_1000_ms() {
        assert_eq!(HOT_WINDOW, Duration::from_millis(500));
        assert_eq!(HOT_HOLD, Duration::from_millis(1000));
        assert_eq!(parse_hot_hold_ms(None), Duration::from_millis(1000));
        assert_eq!(parse_hot_hold_ms(Some("")), Duration::from_millis(1000));
        assert_eq!(parse_hot_hold_ms(Some("invalid")), Duration::from_millis(1000));
        assert_eq!(parse_hot_hold_ms(Some(" 750 ")), Duration::from_millis(750));
    }

    #[test]
    fn parse_io_hold_ms_parses_override_and_defaults_to_2000_ms() {
        assert_eq!(IO_HOLD, Duration::from_secs(2));
        assert_eq!(parse_io_hold_ms(None), Duration::from_millis(2000));
        assert_eq!(parse_io_hold_ms(Some("")), Duration::from_millis(2000));
        assert_eq!(parse_io_hold_ms(Some("invalid")), Duration::from_millis(2000));
        assert_eq!(parse_io_hold_ms(Some(" 1200 ")), Duration::from_millis(1200));
        assert_eq!(io_hold(), Duration::from_millis(2000));
        assert!(io_hold_for(271 * 1024 * 1024) >= Duration::from_secs(8));
        assert_eq!(io_hold_for(4096), Duration::from_millis(2000));
    }

    #[test]
    fn convert_pin_to_hold_and_clear_io_hold_release_page_immediately() {
        let mut t = track();
        let now = Instant::now();
        t.on_map(5, 0x10_0000, true);
        assert_eq!(t.take(now, 10, all), (vec![5], vec![0x10_0000]));

        assert_eq!(t.begin_write_pin(5), [0x10_0000]);
        t.convert_pin_to_hold(5, now + Duration::from_secs(10));
        assert_eq!(t.take(now + Duration::from_secs(1), 10, all), (vec![5], vec![]));

        t.clear_io_hold(5);
        assert_eq!(t.take(now + Duration::from_secs(1), 10, all), (vec![5], vec![0x10_0000]));
        assert_eq!(t.take(now + Duration::from_secs(1), 10, all), (vec![], vec![]));
    }

    #[test]
    fn take_with_cap_preserves_remaining_pending_order() {
        let mut t = track();
        let now = Instant::now();
        for p in 0..5u32 {
            t.on_map(p, 0x10_0000 + p as usize * P, true);
        }
        assert_eq!(t.take(now, 2, all).0, vec![0, 1]);
        assert_eq!(t.pending_len(), 3);
        assert_eq!(t.take(now, 2, all).0, vec![2, 3]);
        assert_eq!(t.take(now, 2, all).0, vec![4]);
        assert_eq!(t.pending_len(), 0);
    }

    #[test]
    fn uncommitted_pending_and_is_clean_addr_reflect_state() {
        let mut t = track();
        let now = Instant::now();
        t.on_map(0, 0x10_0000, true);
        t.on_map(1, 0x10_1000, true);
        t.mark_committed(0);
        assert_eq!(t.uncommitted_pending(10), vec![1]);
        assert!(!t.is_clean_addr(0x10_0000));
        let (taken, protect) = t.take(now, 10, |p| p == 0);
        assert_eq!(taken, vec![0]);
        assert_eq!(protect, vec![0x10_0000]);
        assert!(t.is_clean_addr(0x10_0000));
        assert!(t.is_clean_addr(0x10_1000));
        let _ = t.on_fault(0, 0x10_0000, now);
        assert!(!t.is_clean_addr(0x10_0000));
    }

    #[test]
    fn parse_guard_rest_ms_parses_override_and_defaults_to_32_ms() {
        assert_eq!(GUARD_REST, Duration::from_millis(32));
        assert_eq!(parse_guard_rest_ms(None), GUARD_REST);
        assert_eq!(parse_guard_rest_ms(Some("0")), Duration::ZERO);
        assert_eq!(parse_guard_rest_ms(Some(" 16 ")), Duration::from_millis(16));
        assert_eq!(parse_guard_rest_ms(Some("bad")), GUARD_REST);
    }

    #[test]
    fn guard_rest_keeps_faulted_page_writable_until_rest_elapses() {
        let mut t = track();
        let t0 = Instant::now();
        t.on_map(1, 0x10_0000, true);
        assert_eq!(t.take(t0, 10, all), (vec![1], vec![0x10_0000]));

        assert_eq!(t.on_fault(1, 0x10_0000, t0), Some(vec![0x10_0000]));
        // Immediately after fault, take copies the page but keeps it writable during guard rest.
        assert_eq!(t.take(t0 + Duration::from_millis(1), 10, all), (vec![1], vec![]));
        // Once guard rest elapses, take copies and protects it again.
        assert_eq!(t.take(t0 + GUARD_REST, 10, all), (vec![1], vec![0x10_0000]));
        assert_eq!(t.take(t0 + GUARD_REST, 10, all), (vec![], vec![]));
    }
}
