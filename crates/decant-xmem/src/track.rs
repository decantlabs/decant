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

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::pages::PAGE_BYTES;

/// How long a page that a kernel write may still reach (`prepare`) stays unprotected by default.
pub const IO_HOLD: Duration = Duration::from_secs(2);
/// How long a page stays open for a write that ends when a call returns (`Engine::begin_write`) at most,
/// should the call never come back.
pub const CALL_HOLD: Duration = Duration::from_secs(600);

/// A write fault this soon after the page was made clean counts as a quick one.
pub const HOT_WINDOW: Duration = Duration::from_millis(250);
/// Quick faults in a row that make a page hot.
pub const HOT_FAULTS: u8 = 3;
/// How long a hot page stays unprotected and is copied at every sync by default.
pub const HOT_HOLD: Duration = Duration::from_millis(250);
/// Minimum writable rest after a CPU write fault before `take` re-protects the page by default.
pub const GUARD_REST: Duration = Duration::from_millis(8);
/// Minimum interval between `VirtualProtect(PAGE_READONLY)` waves in `Engine::take_dirty` by default.
pub const REPROTECT_INTERVAL: Duration = Duration::from_millis(12);
/// Pending page count that forces an immediate re-protect wave regardless of `REPROTECT_INTERVAL`.
pub const REPROTECT_WATERMARK: usize = 1024;

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

/// Parses `DECANT_XMEM_REPROTECT_MS` from an optional string, falling back to [`REPROTECT_INTERVAL`].
pub fn parse_reprotect_ms(raw: Option<&str>) -> Duration {
    parse_hold_ms(raw, REPROTECT_INTERVAL)
}

/// Minimum interval between batched re-protection waves.
pub fn reprotect_interval() -> Duration {
    static INTERVAL: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *INTERVAL.get_or_init(|| parse_reprotect_ms(std::env::var("DECANT_XMEM_REPROTECT_MS").ok().as_deref()))
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

#[derive(Clone, Debug, Default)]
enum AliasSlot {
    #[default]
    Empty,
    One(Alias),
    #[allow(clippy::box_collection)]
    Many(Box<Vec<Alias>>),
}

impl AliasSlot {
    fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    fn as_slice(&self) -> &[Alias] {
        match self {
            Self::Empty => &[],
            Self::One(a) => std::slice::from_ref(a),
            Self::Many(v) => v.as_slice(),
        }
    }

    fn push(&mut self, alias: Alias) {
        match std::mem::take(self) {
            Self::Empty => *self = Self::One(alias),
            Self::One(first) => *self = Self::Many(Box::new(vec![first, alias])),
            Self::Many(mut v) => {
                v.push(alias);
                *self = Self::Many(v);
            }
        }
    }

    fn retain_addr(&mut self, addr: usize) {
        match std::mem::take(self) {
            Self::Empty => {}
            Self::One(a) => {
                if a.addr != addr {
                    *self = Self::One(a);
                }
            }
            Self::Many(mut v) => {
                v.retain(|a| a.addr != addr);
                match v.len() {
                    0 => {}
                    1 => *self = Self::One(v[0]),
                    _ => *self = Self::Many(v),
                }
            }
        }
    }

    fn for_each_mut(&mut self, mut f: impl FnMut(&mut Alias)) {
        match self {
            Self::Empty => {}
            Self::One(a) => f(a),
            Self::Many(v) => v.iter_mut().for_each(f),
        }
    }

    fn find_mut(&mut self, mut pred: impl FnMut(&Alias) -> bool) -> Option<&mut Alias> {
        match self {
            Self::Empty => None,
            Self::One(a) => pred(a).then_some(a),
            Self::Many(v) => v.iter_mut().find(|a| pred(a)),
        }
    }
}

const FLAG_CLEAN: u8 = 1 << 0;
const FLAG_QUEUED: u8 = 1 << 1;
const FLAG_COMMITTED: u8 = 1 << 2;
const FLAG_IO_HELD: u8 = 1 << 3;

#[derive(Debug, Default)]
pub struct Track {
    /// Per pool page packed flags: clean, queued, committed, and io_held.
    flags: Vec<u8>,
    /// Per pool page view aliases.
    aliases: Vec<AliasSlot>,
    addresses: HashMap<usize, (u32, bool)>,
    /// Epoch for compact microsecond timestamps in `held_until_us` and `cleaned_us`.
    epoch: Option<Instant>,
    /// Per pool page hold deadline in microseconds from `epoch` (`0` means not held).
    held_until_us: Vec<u64>,
    /// Per pool page active asynchronous write pin count.
    active_writes: Vec<u32>,
    /// When `take` last made the page clean in microseconds from `epoch` (`0` means never).
    cleaned_us: Vec<u64>,
    /// Quick faults in a row since `cleaned_us` (`HOT_WINDOW`).
    quick_faults: Vec<u8>,
    /// The dirty pages that have a view, waiting for `take`.
    pending: Vec<u32>,
    /// Reusable scratch buffers for `take`.
    keep_scratch: Vec<u32>,
    protect_scratch: Vec<usize>,
    /// Whether `pending` may contain uncommitted pages.
    has_uncommitted_pending: bool,
    /// Last timestamp when `take_batched` emitted a protection wave.
    last_reprotect: Option<Instant>,
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
        Track {
            flags: vec![0; pages],
            aliases: vec![AliasSlot::Empty; pages],
            held_until_us: vec![0; pages],
            active_writes: vec![0; pages],
            cleaned_us: vec![0; pages],
            quick_faults: vec![0; pages],
            ..Track::default()
        }
    }

    /// The pool grew to `pages` pages.
    pub fn grow(&mut self, pages: usize) {
        self.flags.resize(pages, 0);
        self.aliases.resize(pages, AliasSlot::Empty);
        self.held_until_us.resize(pages, 0);
        self.active_writes.resize(pages, 0);
        self.cleaned_us.resize(pages, 0);
        self.quick_faults.resize(pages, 0);
    }

    #[inline]
    fn stamp_us(&mut self, t: Instant) -> u64 {
        let epoch = *self.epoch.get_or_insert(t);
        (t.saturating_duration_since(epoch).as_micros() as u64).saturating_add(1)
    }

    #[inline]
    fn read_us(&self, t: Instant) -> u64 {
        let Some(epoch) = self.epoch else { return 1 };
        (t.saturating_duration_since(epoch).as_micros() as u64).saturating_add(1)
    }

    /// The pool page at a writable view address, without consulting the address space registry.
    pub fn writable_page(&self, addr: usize) -> Option<u32> {
        self.addresses.get(&addr).and_then(|&(page, rw)| rw.then_some(page))
    }

    /// Whether `page` is known to be committed in its backing section.
    pub fn is_committed(&self, page: u32) -> bool {
        self.flags.get(page as usize).is_some_and(|&f| (f & FLAG_COMMITTED) != 0)
    }

    /// Records that `page` is committed in its backing section.
    pub fn mark_committed(&mut self, page: u32) {
        if let Some(slot) = self.flags.get_mut(page as usize) {
            *slot |= FLAG_COMMITTED;
        }
    }

    /// Appends the addresses of the open read write views of `page` to `out` and marks them closed.
    fn take_open_rw(&mut self, page: u32, out: &mut Vec<usize>) {
        self.aliases[page as usize].for_each_mut(|a| {
            if a.rw && a.open {
                a.open = false;
                out.push(a.addr);
            }
        });
    }

    /// Makes the page dirty and queues it when it has a view, appending any closed read write views
    /// to `out` and marking them open.
    fn dirty_into(&mut self, page: u32, out: &mut Vec<usize>) {
        let i = page as usize;
        self.flags[i] &= !FLAG_CLEAN;
        if (self.flags[i] & FLAG_QUEUED) == 0 && !self.aliases[i].is_empty() {
            self.flags[i] |= FLAG_QUEUED;
            self.pending.push(page);
            if (self.flags[i] & FLAG_COMMITTED) == 0 {
                self.has_uncommitted_pending = true;
            }
        }
        self.aliases[i].for_each_mut(|a| {
            if a.rw && !a.open {
                a.open = true;
                out.push(a.addr);
            }
        });
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
        self.aliases[page as usize].push(Alias { addr, rw, open: false });
        self.dirty(page)
    }

    /// The view of the page at `addr` is gone.
    pub fn on_unmap(&mut self, page: u32, addr: usize) {
        self.addresses.remove(&addr);
        let i = page as usize;
        self.aliases[i].retain_addr(addr);
        if self.aliases[i].is_empty() {
            self.held_until_us[i] = 0;
            self.flags[i] &= !FLAG_IO_HELD;
            self.active_writes[i] = 0;
            self.cleaned_us[i] = 0;
            self.quick_faults[i] = 0;
        }
    }

    /// A write faulted on a read write view `fault_addr` of `page` at `now`. `Some(views)` when
    /// `fault_addr` needs opening (`PAGE_READWRITE`): only `fault_addr` is opened and returned, while
    /// sibling aliases stay guarded until they fault or are opened for I/O. `None` when `fault_addr`
    /// was already open (another thread won the race, the faulting write may simply run again).
    /// The `HOT_FAULTS`th quick fault in a row makes the page hot: held for `hot_hold()`.
    pub fn on_fault(&mut self, page: u32, fault_addr: usize, now: Instant) -> Option<Vec<usize>> {
        let i = page as usize;
        let was_clean = (self.flags[i] & FLAG_CLEAN) != 0;
        let needs_open = self.aliases[i]
            .as_slice()
            .iter()
            .find(|a| a.addr == fault_addr && a.rw)
            .is_some_and(|a| !a.open);
        if !was_clean && !needs_open {
            return None;
        }
        if was_clean && self.cleaned_us[i] != 0 {
            let at_us = self.cleaned_us[i];
            let now_us = self.read_us(now);
            let elapsed = Duration::from_micros(now_us.saturating_sub(at_us));
            self.quick_faults[i] = if elapsed < HOT_WINDOW { self.quick_faults[i] + 1 } else { 0 };
            if self.quick_faults[i] >= HOT_FAULTS {
                self.quick_faults[i] = 0;
                self.held_until_us[i] = self.stamp_us(now + hot_hold());
                self.flags[i] &= !FLAG_IO_HELD;
            }
        }
        let rest = guard_rest();
        if !rest.is_zero() && !self.aliases[i].is_empty() {
            let until = self.stamp_us(now + rest);
            self.held_until_us[i] = self.held_until_us[i].max(until);
        }
        self.flags[i] &= !FLAG_CLEAN;
        if (self.flags[i] & FLAG_QUEUED) == 0 && !self.aliases[i].is_empty() {
            self.flags[i] |= FLAG_QUEUED;
            self.pending.push(page);
            if (self.flags[i] & FLAG_COMMITTED) == 0 {
                self.has_uncommitted_pending = true;
            }
        }
        if let Some(alias) = self.aliases[i].find_mut(|a| a.addr == fault_addr && a.rw)
            && (was_clean || !alias.open)
        {
            alias.open = true;
            return Some(vec![fault_addr]);
        }
        None
    }

    /// Whether the page is held now (a kernel write may reach it, or it is hot).
    pub fn is_held(&self, page: u32, now: Instant) -> bool {
        let until = self.held_until_us[page as usize];
        until != 0 && self.read_us(now) < until
    }

    /// Pins `page` open for an active asynchronous write, appending read write views to `out`.
    pub fn begin_write_pin_into(&mut self, page: u32, out: &mut Vec<usize>) {
        let i = page as usize;
        if !self.aliases[i].is_empty() {
            self.active_writes[i] += 1;
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
        let i = page as usize;
        self.active_writes[i] = self.active_writes[i].saturating_sub(1);
        let _ = self.dirty(page);
    }

    /// Transitions an active write pin on `page` into a timed fallback hold until `until` when
    /// an overlapped read returns `ERROR_IO_PENDING`.
    pub fn convert_pin_to_hold(&mut self, page: u32, until: Instant) {
        let i = page as usize;
        self.active_writes[i] = self.active_writes[i].saturating_sub(1);
        if !self.aliases[i].is_empty() {
            let until_us = self.stamp_us(until);
            self.held_until_us[i] = self.held_until_us[i].max(until_us);
            self.flags[i] |= FLAG_IO_HELD;
        }
        let _ = self.dirty(page);
    }

    /// Clears an I/O fallback hold on `page` when the overlapped read completes and marks the
    /// page dirty once so the next `take` copies it and protects it immediately.
    pub fn clear_io_hold(&mut self, page: u32) {
        let i = page as usize;
        if (self.flags[i] & FLAG_IO_HELD) != 0 {
            self.flags[i] &= !FLAG_IO_HELD;
            self.held_until_us[i] = 0;
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
        let i = page as usize;
        if !self.aliases[i].is_empty() {
            let until_us = self.stamp_us(until);
            self.held_until_us[i] = self.held_until_us[i].max(until_us);
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
        let i = page as usize;
        if self.held_until_us[i] != 0 {
            let hold = io_hold();
            let end = self.stamp_us(now + hold);
            let cap_end = self.stamp_us(now + hot_hold().max(hold));
            let cur = &mut self.held_until_us[i];
            if *cur > cap_end || *cur < end {
                *cur = end;
            }
        }
    }

    /// The dirty pages waiting for `take`.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Pending pages with at least one view that are not yet marked committed, scanning up to `cap`
    /// already committed pages.
    pub fn uncommitted_pending(&mut self, cap: usize) -> Vec<u32> {
        if !self.has_uncommitted_pending {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut count = 0usize;
        let mut completed_scan = true;
        for &page in &self.pending {
            if count >= cap {
                completed_scan = false;
                break;
            }
            let i = page as usize;
            if self.aliases[i].is_empty() {
                continue;
            }
            if (self.flags[i] & FLAG_COMMITTED) == 0 {
                out.push(page);
            } else {
                count += 1;
            }
        }
        if completed_scan && out.is_empty() {
            self.has_uncommitted_pending = false;
        }
        out
    }

    /// Whether `address` currently maps a writable view that is clean or guarded (`!open`).
    pub fn is_clean_addr(&self, address: usize) -> bool {
        let Some(&(page, rw)) = self.addresses.get(&address) else { return false };
        if !rw {
            return false;
        }
        let i = page as usize;
        (self.flags[i] & FLAG_CLEAN) != 0
            || self.aliases[i]
                .as_slice()
                .iter()
                .find(|a| a.addr == address)
                .is_some_and(|a| !a.open)
    }

    /// Whether the page has a view with a read write title protection.
    pub fn writable(&self, page: u32) -> bool {
        self.aliases[page as usize].as_slice().iter().any(|a| a.rw)
    }

    /// The dirty pages with a view, at most `cap`, for the caller to copy. Each page that is not held is
    /// marked clean and its read write views are returned to be protected, before the caller copies. A
    /// page without a view stays dirty and unprotected, and is dropped from the queue until a view appears.
    /// A page for which `committed` is false was never touched: it holds zeros as its mirror does, so it is
    /// clean without a copy and without protection (it cannot be protected), and its first access, which
    /// commits it, makes it dirty (`on_fault`).
    pub fn take(&mut self, now: Instant, cap: usize, committed: impl Fn(u32) -> bool) -> (Vec<u32>, Vec<usize>) {
        self.take_batched(now, cap, Duration::ZERO, committed)
    }

    /// Like `take`, but defers returning `protect` addresses until `interval` has elapsed since
    /// the last protection wave or `pending` reaches `REPROTECT_WATERMARK`. While deferred, dirty
    /// pages stay queued in `pending` and are still returned in `taken` for GPU coherency.
    pub fn take_batched(
        &mut self,
        now: Instant,
        cap: usize,
        interval: Duration,
        committed: impl Fn(u32) -> bool,
    ) -> (Vec<u32>, Vec<usize>) {
        let protect_now = interval.is_zero()
            || self.pending.len() >= REPROTECT_WATERMARK
            || self.last_reprotect.is_none_or(|last| now >= last + interval);
        let mut taken = Vec::new();
        let mut protect = std::mem::take(&mut self.protect_scratch);
        protect.clear();
        let now_us = self.stamp_us(now);
        let mut pending = std::mem::take(&mut self.pending);
        let mut keep = std::mem::take(&mut self.keep_scratch);
        keep.clear();
        let mut hit_cap = false;
        for (idx, &page) in pending.iter().enumerate() {
            let i = page as usize;
            if taken.len() >= cap {
                keep.extend_from_slice(&pending[idx..]);
                hit_cap = true;
                break;
            } else if self.aliases[i].is_empty() {
                self.flags[i] &= !FLAG_QUEUED;
            } else if (self.flags[i] & FLAG_COMMITTED) == 0 && !committed(page) {
                self.flags[i] |= FLAG_CLEAN;
                self.flags[i] &= !FLAG_QUEUED;
            } else if !protect_now
                || self.active_writes[i] > 0
                || (self.held_until_us[i] != 0 && now_us < self.held_until_us[i])
            {
                self.flags[i] |= FLAG_COMMITTED;
                taken.push(page);
                keep.push(page);
            } else {
                self.flags[i] |= FLAG_COMMITTED;
                self.held_until_us[i] = 0;
                self.flags[i] &= !FLAG_IO_HELD;
                self.take_open_rw(page, &mut protect);
                self.cleaned_us[i] = now_us;
                self.flags[i] |= FLAG_CLEAN;
                self.flags[i] &= !FLAG_QUEUED;
                taken.push(page);
            }
        }
        if protect_now {
            if !hit_cap {
                self.has_uncommitted_pending = false;
            }
            if !protect.is_empty() || self.last_reprotect.is_none() {
                self.last_reprotect = Some(now);
            }
        }
        pending.clear();
        self.keep_scratch = pending;
        self.pending = keep;
        let out_protect = protect.clone();
        protect.clear();
        self.protect_scratch = protect;
        (taken, out_protect)
    }

    fn is_bridgeable_clean_page(&self, addr: usize, expected_page: u32) -> bool {
        let Some(&(page, rw)) = self.addresses.get(&addr) else {
            return false;
        };
        if !rw || page != expected_page {
            return false;
        }
        let i = page as usize;
        let mask = FLAG_CLEAN | FLAG_COMMITTED | FLAG_QUEUED | FLAG_IO_HELD;
        if (self.flags[i] & mask) != (FLAG_CLEAN | FLAG_COMMITTED) {
            return false;
        }
        if self.active_writes[i] != 0 {
            return false;
        }
        self.aliases[i]
            .as_slice()
            .iter()
            .any(|a| a.addr == addr && a.rw && !a.open)
    }

    /// Bridges `1..=max_gap` clean, committed, guarded, non-queued, non-held pages between two
    /// adjacent protect addresses from the same contiguous pool-page view so `VirtualProtect` runs
    /// coalesce without changing protection semantics.
    pub fn bridge_clean_gaps(&self, mut addrs: Vec<usize>, max_gap: usize) -> Vec<usize> {
        if addrs.len() < 2 || max_gap == 0 {
            return addrs;
        }
        addrs.sort_unstable();
        addrs.dedup();
        let mut out = Vec::with_capacity(addrs.len() + addrs.len() / 2);
        for i in 0..addrs.len() {
            let cur = addrs[i];
            if i > 0 {
                let prev = addrs[i - 1];
                if cur > prev {
                    let diff = cur - prev;
                    if diff.is_multiple_of(PAGE_BYTES) {
                        let gap = diff / PAGE_BYTES - 1;
                        if (1..=max_gap).contains(&gap)
                            && let Some(&(prev_page, true)) = self.addresses.get(&prev)
                            && let Some(&(cur_page, true)) = self.addresses.get(&cur)
                            && cur_page == prev_page + (gap as u32 + 1)
                            && (1..=gap).all(|k| self.is_bridgeable_clean_page(prev + k * PAGE_BYTES, prev_page + k as u32))
                        {
                            for k in 1..=gap {
                                out.push(prev + k * PAGE_BYTES);
                            }
                        }
                    }
                }
            }
            out.push(cur);
        }
        out
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
    fn parse_hot_hold_ms_parses_override_and_defaults_to_250_ms() {
        assert_eq!(HOT_WINDOW, Duration::from_millis(250));
        assert_eq!(HOT_HOLD, Duration::from_millis(250));
        assert_eq!(parse_hot_hold_ms(None), Duration::from_millis(250));
        assert_eq!(parse_hot_hold_ms(Some("")), Duration::from_millis(250));
        assert_eq!(parse_hot_hold_ms(Some("invalid")), Duration::from_millis(250));
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
    fn parse_guard_rest_ms_parses_override_and_defaults_to_8_ms() {
        assert_eq!(GUARD_REST, Duration::from_millis(8));
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

    #[test]
    fn batched_reprotect_defers_protection_until_interval_or_watermark() {
        assert_eq!(REPROTECT_INTERVAL, Duration::from_millis(12));
        assert_eq!(parse_reprotect_ms(None), REPROTECT_INTERVAL);
        assert_eq!(parse_reprotect_ms(Some("0")), Duration::ZERO);
        assert_eq!(parse_reprotect_ms(Some(" 20 ")), Duration::from_millis(20));

        let mut t = track();
        let t0 = Instant::now();
        t.on_map(1, 0x10_0000, true);
        // First wave protects immediately.
        assert_eq!(t.take_batched(t0, 10, REPROTECT_INTERVAL, all), (vec![1], vec![0x10_0000]));

        // Map page 2 within the interval: taken is returned every call, protect is deferred.
        t.on_map(2, 0x11_0000, true);
        assert_eq!(t.take_batched(t0 + Duration::from_millis(2), 10, REPROTECT_INTERVAL, all), (vec![2], vec![]));
        assert!(!t.is_clean_addr(0x11_0000));
        assert_eq!(t.take_batched(t0 + Duration::from_millis(6), 10, REPROTECT_INTERVAL, all), (vec![2], vec![]));

        // Once the 12 ms interval elapses, protection runs and marks page 2 clean.
        assert_eq!(t.take_batched(t0 + REPROTECT_INTERVAL, 10, REPROTECT_INTERVAL, all), (vec![2], vec![0x11_0000]));
        assert!(t.is_clean_addr(0x11_0000));
        assert_eq!(t.take_batched(t0 + REPROTECT_INTERVAL + Duration::from_millis(1), 10, REPROTECT_INTERVAL, all), (vec![], vec![]));

        // Watermark forces immediate protection even before the interval elapses.
        let mut big = Track::new(REPROTECT_WATERMARK + 4);
        big.on_map(0, 0x20_0000, true);
        assert_eq!(big.take_batched(t0, 10, REPROTECT_INTERVAL, all), (vec![0], vec![0x20_0000]));
        for p in 1..=REPROTECT_WATERMARK as u32 {
            big.on_map(p, 0x20_0000 + p as usize * P, true);
        }
        let (taken, protect) = big.take_batched(t0 + Duration::from_millis(1), REPROTECT_WATERMARK + 4, REPROTECT_INTERVAL, all);
        assert_eq!(taken.len(), REPROTECT_WATERMARK);
        assert_eq!(protect.len(), REPROTECT_WATERMARK);
    }

    #[test]
    fn bridge_clean_gaps_bridges_only_safe_clean_committed_guarded_pages() {
        let mut t = Track::new(16);
        let t0 = Instant::now();
        for p in 0..7u32 {
            t.on_map(p, 0x10_0000 + p as usize * P, true);
        }
        let _ = t.take(t0, 16, all);

        // Bridge a 3 page clean committed guarded gap between pages 0 and 4.
        assert_eq!(
            t.bridge_clean_gaps(vec![0x10_0000, 0x14_0000], 4),
            vec![0x10_0000, 0x11_0000, 0x12_0000, 0x13_0000, 0x14_0000]
        );

        // Do not bridge when gap exceeds max_gap.
        assert_eq!(t.bridge_clean_gaps(vec![0x10_0000, 0x16_0000], 4), vec![0x10_0000, 0x16_0000]);

        // Do not bridge across active_writes > 0.
        t.begin_write_pin(2);
        assert_eq!(t.bridge_clean_gaps(vec![0x10_0000, 0x14_0000], 4), vec![0x10_0000, 0x14_0000]);
        t.finish_write_pin(2);
        let _ = t.take(t0 + Duration::from_millis(1), 16, all);

        // Do not bridge across dirty or queued pages.
        let _ = t.on_changed(2);
        assert_eq!(t.bridge_clean_gaps(vec![0x10_0000, 0x14_0000], 4), vec![0x10_0000, 0x14_0000]);
        let _ = t.take(t0 + Duration::from_millis(2), 16, all);

        // Do not bridge across unmapped pages.
        t.on_unmap(2, 0x12_0000);
        assert_eq!(t.bridge_clean_gaps(vec![0x10_0000, 0x14_0000], 4), vec![0x10_0000, 0x14_0000]);

        // Do not bridge across uncommitted pages.
        t.on_map(8, 0x30_0000, true);
        t.on_map(9, 0x31_0000, true);
        t.on_map(10, 0x32_0000, true);
        let _ = t.take(t0 + Duration::from_millis(3), 16, |p| p != 9);
        assert_eq!(t.bridge_clean_gaps(vec![0x30_0000, 0x32_0000], 4), vec![0x30_0000, 0x32_0000]);

        // Do not bridge across non contiguous pool page mappings.
        t.on_map(12, 0x40_0000, true);
        t.on_map(14, 0x41_0000, true);
        t.on_map(13, 0x42_0000, true);
        let _ = t.take(t0 + Duration::from_millis(4), 16, all);
        assert_eq!(t.bridge_clean_gaps(vec![0x40_0000, 0x42_0000], 4), vec![0x40_0000, 0x42_0000]);
    }
}
